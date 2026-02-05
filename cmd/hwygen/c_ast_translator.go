package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

// CASTTranslator walks a ParsedFunc's Go AST and emits GOAT-compatible C code
// with target-specific SIMD intrinsics. Unlike the template-based CEmitter
// (which pattern-matches math function names), this translator handles
// arbitrary control flow: matmul, transpose, dot product, etc.
type CASTTranslator struct {
	profile  *CIntrinsicProfile
	tier     string // primary SIMD tier: "q" for NEON, "ymm" for AVX2, "zmm" for AVX-512
	lanes    int    // number of elements per vector at primary tier
	elemType string // "float32", "float64"

	// Variable tracking
	vars   map[string]cVarInfo   // declared local variables
	params map[string]cParamInfo // function parameters

	buf    *bytes.Buffer
	indent int
}

// cVarInfo tracks the C type of a local variable.
type cVarInfo struct {
	cType    string // "float32x4_t", "float", "long", "float *"
	isVector bool
	isPtr    bool
}

// cParamInfo tracks function parameter translation details.
type cParamInfo struct {
	goName  string // "a", "b", "c", "m", "n", "k"
	goType  string // "[]T", "int"
	cName   string // "a", "b", "c", "pm", "pn", "pk"
	cType   string // "float *", "long *"
	isSlice bool
	isInt   bool
}

// NewCASTTranslator creates a translator for the given profile and element type.
func NewCASTTranslator(profile *CIntrinsicProfile, elemType string) *CASTTranslator {
	tier, lanes := primaryTier(profile)
	return &CASTTranslator{
		profile:  profile,
		tier:     tier,
		lanes:    lanes,
		elemType: elemType,
		vars:     make(map[string]cVarInfo),
		params:   make(map[string]cParamInfo),
		buf:      &bytes.Buffer{},
	}
}

// primaryTier returns the first non-scalar tier name and its lane count.
func primaryTier(p *CIntrinsicProfile) (string, int) {
	for _, t := range p.Tiers {
		if !t.IsScalar {
			return t.Name, t.Lanes
		}
	}
	// Fallback
	return "q", 4
}

// TranslateToC translates a ParsedFunc to GOAT-compatible C source code.
func (t *CASTTranslator) TranslateToC(pf *ParsedFunc) (string, error) {
	t.buf.Reset()
	t.vars = make(map[string]cVarInfo)
	t.params = make(map[string]cParamInfo)
	t.indent = 0

	// Build parameter map
	t.buildParamMap(pf)

	// Emit function signature
	t.emitFuncSignature(pf)

	// Emit int parameter dereferences
	t.indent = 1
	t.emitParamDerefs()

	// Translate function body
	if pf.Body != nil {
		t.translateBlockStmtContents(pf.Body)
	}

	// Close function
	t.indent = 0
	t.writef("}\n")

	return t.buf.String(), nil
}

// buildParamMap creates cParamInfo entries for each Go parameter.
func (t *CASTTranslator) buildParamMap(pf *ParsedFunc) {
	for _, p := range pf.Params {
		info := cParamInfo{
			goName: p.Name,
			goType: p.Type,
		}
		if strings.HasPrefix(p.Type, "[]") {
			// Slice param → pointer
			info.isSlice = true
			info.cName = p.Name
			info.cType = t.profile.CType + " *"
		} else if p.Type == "int" || p.Type == "int64" {
			// Int param → long pointer (GOAT convention)
			info.isInt = true
			info.cName = "p" + p.Name
			info.cType = "long *"
		} else {
			info.cName = p.Name
			info.cType = t.profile.CType
		}
		t.params[p.Name] = info
	}
}

// emitFuncSignature emits: void funcname(float *a, float *b, ..., long *pm, long *pn, ...)
func (t *CASTTranslator) emitFuncSignature(pf *ParsedFunc) {
	funcName := t.cFuncName(pf.Name)
	var params []string
	for _, p := range pf.Params {
		info := t.params[p.Name]
		// Pointer types already end with "* " so just concatenate; non-pointer need a space
		if strings.HasSuffix(info.cType, "*") {
			params = append(params, info.cType+info.cName)
		} else {
			params = append(params, info.cType+" "+info.cName)
		}
	}
	t.writef("void %s(%s) {\n", funcName, strings.Join(params, ", "))
}

// cFuncName builds the C function name from the Go function name.
// BaseMatMul → basematmul_c_f32_neon
func (t *CASTTranslator) cFuncName(baseName string) string {
	name := strings.ToLower(baseName)
	name = strings.TrimPrefix(name, "base")
	targetSuffix := strings.ToLower(t.profile.TargetName)
	return name + "_c_" + cTypeSuffix(t.elemType) + "_" + targetSuffix
}

// emitParamDerefs emits: long m = *pm; for each int parameter.
func (t *CASTTranslator) emitParamDerefs() {
	for _, info := range sortedParams(t.params) {
		if info.isInt {
			t.writef("long %s = *%s;\n", info.goName, info.cName)
			// Register the dereferenced variable as a long
			t.vars[info.goName] = cVarInfo{cType: "long"}
		}
	}
}

// sortedParams returns params in stable order (by original param order in the function).
// We iterate over t.params, but need deterministic order. We reconstruct from params.
func sortedParams(params map[string]cParamInfo) []cParamInfo {
	// Since we need order, collect values sorted by goName which gives deterministic output.
	// In practice the caller should use the original pf.Params order, but since we
	// only have the map here, we need a stable approach.
	var result []cParamInfo
	for _, v := range params {
		result = append(result, v)
	}
	// Sort by cName for stable output
	for i := range result {
		for j := i + 1; j < len(result); j++ {
			if result[i].goName > result[j].goName {
				result[i], result[j] = result[j], result[i]
			}
		}
	}
	return result
}

// cDeclVar formats a C variable declaration, e.g. "float *x" or "long y".
// For pointer types ending in "*", the name is placed directly after the star.
func cDeclVar(cType, name string) string {
	if strings.HasSuffix(cType, "*") {
		return cType + name
	}
	return cType + " " + name
}

// ---------------------------------------------------------------------------
// Statement translators
// ---------------------------------------------------------------------------

// translateBlockStmtContents translates each statement in a block.
func (t *CASTTranslator) translateBlockStmtContents(block *ast.BlockStmt) {
	if block == nil {
		return
	}
	for _, stmt := range block.List {
		t.translateStmt(stmt)
	}
}

// translateStmt dispatches to the appropriate statement handler.
func (t *CASTTranslator) translateStmt(stmt ast.Stmt) {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		t.translateAssignStmt(s)
	case *ast.ForStmt:
		t.translateForStmt(s)
	case *ast.RangeStmt:
		t.translateRangeStmt(s)
	case *ast.ExprStmt:
		t.translateExprStmt(s)
	case *ast.DeclStmt:
		t.translateDeclStmt(s)
	case *ast.IfStmt:
		t.translateIfStmt(s)
	case *ast.BlockStmt:
		t.writef("{\n")
		t.indent++
		t.translateBlockStmtContents(s)
		t.indent--
		t.writef("}\n")
	case *ast.IncDecStmt:
		t.translateIncDecStmt(s)
	default:
		// Skip unsupported statements
	}
}

// translateAssignStmt handles := and = assignments.
func (t *CASTTranslator) translateAssignStmt(s *ast.AssignStmt) {
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		// Multi-assign not supported
		return
	}

	lhs := s.Lhs[0]
	rhs := s.Rhs[0]

	lhsName := t.translateExpr(lhs)

	switch s.Tok {
	case token.DEFINE: // :=
		varInfo := t.inferType(rhs)
		t.vars[lhsName] = varInfo
		rhsStr := t.translateExpr(rhs)
		t.writef("%s = %s;\n", cDeclVar(varInfo.cType, lhsName), rhsStr)

	case token.ASSIGN: // =
		rhsStr := t.translateExpr(rhs)
		t.writef("%s = %s;\n", lhsName, rhsStr)

	case token.ADD_ASSIGN: // +=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s += %s;\n", lhsName, rhsStr)

	case token.SUB_ASSIGN: // -=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s -= %s;\n", lhsName, rhsStr)

	case token.MUL_ASSIGN: // *=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s *= %s;\n", lhsName, rhsStr)
	}
}

// translateForStmt handles C-style for loops.
func (t *CASTTranslator) translateForStmt(s *ast.ForStmt) {
	initStr := ""
	condStr := ""
	postStr := ""

	// Init
	if s.Init != nil {
		initStr = t.translateForInit(s.Init)
	}

	// Condition
	if s.Cond != nil {
		condStr = t.translateExpr(s.Cond)
	}

	// Post
	if s.Post != nil {
		postStr = t.translateForPost(s.Post)
	}

	t.writef("for (%s; %s; %s) {\n", initStr, condStr, postStr)
	t.indent++
	t.translateBlockStmtContents(s.Body)
	t.indent--
	t.writef("}\n")
}

// translateForInit translates a for-loop init statement.
func (t *CASTTranslator) translateForInit(stmt ast.Stmt) string {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
			lhs := t.translateExpr(s.Lhs[0])
			rhs := t.translateExpr(s.Rhs[0])
			if s.Tok == token.DEFINE {
				// Declare variable with type
				varInfo := t.inferType(s.Rhs[0])
				t.vars[lhs] = varInfo
				return fmt.Sprintf("%s = %s", cDeclVar(varInfo.cType, lhs), rhs)
			}
			return fmt.Sprintf("%s = %s", lhs, rhs)
		}
	}
	return ""
}

// translateForPost translates a for-loop post statement.
func (t *CASTTranslator) translateForPost(stmt ast.Stmt) string {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
			lhs := t.translateExpr(s.Lhs[0])
			rhs := t.translateExpr(s.Rhs[0])
			switch s.Tok {
			case token.ADD_ASSIGN:
				return fmt.Sprintf("%s += %s", lhs, rhs)
			case token.ASSIGN:
				return fmt.Sprintf("%s = %s", lhs, rhs)
			}
		}
	case *ast.IncDecStmt:
		name := t.translateExpr(s.X)
		if s.Tok == token.INC {
			return name + "++"
		}
		return name + "--"
	}
	return ""
}

// translateRangeStmt translates `for i := range m` to `for (long i = 0; i < m; i++)`.
func (t *CASTTranslator) translateRangeStmt(s *ast.RangeStmt) {
	// `for i := range m` → key is i, X is m
	if s.Key == nil {
		return
	}

	iter := t.translateExpr(s.Key)
	rangeOver := t.translateExpr(s.X)

	// Register the iterator variable
	t.vars[iter] = cVarInfo{cType: "long"}

	t.writef("for (long %s = 0; %s < %s; %s++) {\n", iter, iter, rangeOver, iter)
	t.indent++
	t.translateBlockStmtContents(s.Body)
	t.indent--
	t.writef("}\n")
}

// translateExprStmt handles standalone expression statements (function calls).
func (t *CASTTranslator) translateExprStmt(s *ast.ExprStmt) {
	call, ok := s.X.(*ast.CallExpr)
	if !ok {
		return
	}

	// Check for panic() calls - skip them
	if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "panic" {
		return
	}

	// Check for hwy.Store calls
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			if sel.Sel.Name == "Store" {
				t.emitHwyStore(call.Args)
				return
			}
		}
	}

	// Generic function call
	expr := t.translateExpr(s.X)
	t.writef("%s;\n", expr)
}

// translateDeclStmt handles `var j int`.
func (t *CASTTranslator) translateDeclStmt(s *ast.DeclStmt) {
	genDecl, ok := s.Decl.(*ast.GenDecl)
	if !ok || genDecl.Tok != token.VAR {
		return
	}

	for _, spec := range genDecl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}

		cType := t.goTypeToCType(exprToString(valueSpec.Type))
		for _, name := range valueSpec.Names {
			t.vars[name.Name] = cVarInfo{cType: cType}
			t.writef("%s;\n", cDeclVar(cType, name.Name))
		}
	}
}

// translateIfStmt handles if statements. Skips panic guards (bounds checks).
func (t *CASTTranslator) translateIfStmt(s *ast.IfStmt) {
	// Skip if the body contains only a panic call (bounds check)
	if isPanicGuard(s) {
		return
	}

	condStr := t.translateExpr(s.Cond)
	t.writef("if (%s) {\n", condStr)
	t.indent++
	t.translateBlockStmtContents(s.Body)
	t.indent--
	if s.Else != nil {
		t.writef("} else ")
		// Don't add newline - let the else clause handle it
		switch e := s.Else.(type) {
		case *ast.BlockStmt:
			t.writefRaw("{\n")
			t.indent++
			t.translateBlockStmtContents(e)
			t.indent--
			t.writef("}\n")
		case *ast.IfStmt:
			// else if
			t.translateIfStmt(e)
		}
	} else {
		t.writef("}\n")
	}
}

// translateIncDecStmt handles i++ and i--.
func (t *CASTTranslator) translateIncDecStmt(s *ast.IncDecStmt) {
	name := t.translateExpr(s.X)
	if s.Tok == token.INC {
		t.writef("%s++;\n", name)
	} else {
		t.writef("%s--;\n", name)
	}
}

// isPanicGuard returns true if the if statement's body only contains panic().
func isPanicGuard(s *ast.IfStmt) bool {
	if s.Body == nil || len(s.Body.List) == 0 {
		return false
	}
	for _, stmt := range s.Body.List {
		exprStmt, ok := stmt.(*ast.ExprStmt)
		if !ok {
			return false
		}
		call, ok := exprStmt.X.(*ast.CallExpr)
		if !ok {
			return false
		}
		ident, ok := call.Fun.(*ast.Ident)
		if !ok || ident.Name != "panic" {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Expression translators
// ---------------------------------------------------------------------------

// translateExpr converts a Go AST expression to a C expression string.
func (t *CASTTranslator) translateExpr(expr ast.Expr) string {
	if expr == nil {
		return ""
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.BasicLit:
		return t.translateBasicLit(e)
	case *ast.BinaryExpr:
		return t.translateBinaryExpr(e)
	case *ast.CallExpr:
		return t.translateCallExpr(e)
	case *ast.IndexExpr:
		return t.translateIndexExpr(e)
	case *ast.SliceExpr:
		return t.translateSliceExpr(e)
	case *ast.SelectorExpr:
		return t.translateSelectorExpr(e)
	case *ast.ParenExpr:
		return "(" + t.translateExpr(e.X) + ")"
	case *ast.UnaryExpr:
		return t.translateUnaryExpr(e)
	case *ast.StarExpr:
		return "*" + t.translateExpr(e.X)
	default:
		return exprToString(expr)
	}
}

// translateBasicLit translates a literal (int, float, string).
func (t *CASTTranslator) translateBasicLit(lit *ast.BasicLit) string {
	switch lit.Kind {
	case token.INT:
		return lit.Value
	case token.FLOAT:
		if t.elemType == "float32" {
			return lit.Value + "f"
		}
		return lit.Value
	default:
		return lit.Value
	}
}

// translateBinaryExpr translates binary operations.
func (t *CASTTranslator) translateBinaryExpr(e *ast.BinaryExpr) string {
	left := t.translateExpr(e.X)
	right := t.translateExpr(e.Y)
	return left + " " + e.Op.String() + " " + right
}

// translateCallExpr translates function calls, dispatching hwy.* calls to intrinsics.
func (t *CASTTranslator) translateCallExpr(e *ast.CallExpr) string {
	// Check for hwy.Func(...) or hwy.Func[T](...)
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			return t.translateHwyCall(sel.Sel.Name, e.Args)
		}
	}

	// Check for v.NumLanes() method calls
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements" {
			return fmt.Sprintf("%d", t.lanes)
		}
	}

	// Generic call - just translate arguments
	fun := t.translateExpr(e.Fun)
	var args []string
	for _, arg := range e.Args {
		args = append(args, t.translateExpr(arg))
	}
	return fmt.Sprintf("%s(%s)", fun, strings.Join(args, ", "))
}

// extractSelectorExpr extracts the SelectorExpr from a call expression's Fun,
// handling both direct calls (hwy.Load) and generic calls (hwy.Zero[T]).
func extractSelectorExpr(fun ast.Expr) *ast.SelectorExpr {
	switch f := fun.(type) {
	case *ast.SelectorExpr:
		return f
	case *ast.IndexExpr:
		// hwy.Zero[T]() → IndexExpr{X: SelectorExpr{hwy, Zero}, Index: T}
		if sel, ok := f.X.(*ast.SelectorExpr); ok {
			return sel
		}
	case *ast.IndexListExpr:
		if sel, ok := f.X.(*ast.SelectorExpr); ok {
			return sel
		}
	}
	return nil
}

// translateIndexExpr translates array indexing: a[i*k+p].
func (t *CASTTranslator) translateIndexExpr(e *ast.IndexExpr) string {
	x := t.translateExpr(e.X)
	idx := t.translateExpr(e.Index)
	return fmt.Sprintf("%s[%s]", x, idx)
}

// translateSliceExpr translates slice expressions to C pointer arithmetic.
// c[i*n : (i+1)*n] → c + i*n   (as a pointer alias)
// bRow[j:]          → bRow + j  (as a pointer argument)
func (t *CASTTranslator) translateSliceExpr(e *ast.SliceExpr) string {
	x := t.translateExpr(e.X)

	if e.Low != nil {
		low := t.translateExpr(e.Low)
		return x + " + " + low
	}
	return x
}

// translateSelectorExpr translates field access / method references.
func (t *CASTTranslator) translateSelectorExpr(e *ast.SelectorExpr) string {
	x := t.translateExpr(e.X)
	return x + "." + e.Sel.Name
}

// translateUnaryExpr translates unary expressions.
func (t *CASTTranslator) translateUnaryExpr(e *ast.UnaryExpr) string {
	operand := t.translateExpr(e.X)
	return e.Op.String() + operand
}

// ---------------------------------------------------------------------------
// hwy.* call mapping
// ---------------------------------------------------------------------------

// translateHwyCall maps hwy.FuncName to the appropriate C intrinsic.
func (t *CASTTranslator) translateHwyCall(funcName string, args []ast.Expr) string {
	switch funcName {
	case "Load":
		return t.emitHwyLoad(args)
	case "Store":
		// Store is typically a statement, not expression — but handle both
		return t.emitHwyStoreExpr(args)
	case "Set":
		return t.emitHwySet(args)
	case "Zero":
		return t.emitHwyZero()
	case "MulAdd":
		return t.emitHwyMulAdd(args)
	case "Add":
		return t.emitHwyBinaryOp(t.profile.AddFn, args)
	case "Sub":
		return t.emitHwyBinaryOp(t.profile.SubFn, args)
	case "Mul":
		return t.emitHwyBinaryOp(t.profile.MulFn, args)
	case "Div":
		return t.emitHwyBinaryOp(t.profile.DivFn, args)
	case "Min":
		return t.emitHwyBinaryOp(t.profile.MinFn, args)
	case "Max":
		return t.emitHwyBinaryOp(t.profile.MaxFn, args)
	case "Neg":
		return t.emitHwyUnaryOp(t.profile.NegFn, args)
	case "Abs":
		return t.emitHwyUnaryOp(t.profile.AbsFn, args)
	case "Sqrt":
		return t.emitHwyUnaryOp(t.profile.SqrtFn, args)
	default:
		// Unknown hwy call — emit as-is
		var argStrs []string
		for _, a := range args {
			argStrs = append(argStrs, t.translateExpr(a))
		}
		return fmt.Sprintf("hwy_%s(%s)", strings.ToLower(funcName), strings.Join(argStrs, ", "))
	}
}

// emitHwyLoad: hwy.Load(slice[off:]) → vld1q_f32(ptr + off)
func (t *CASTTranslator) emitHwyLoad(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* Load: missing args */"
	}
	loadFn := t.profile.LoadFn[t.tier]
	ptr := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", loadFn, ptr)
}

// emitHwyStore: hwy.Store(vec, slice[off:]) → vst1q_f32(ptr + off, vec)
func (t *CASTTranslator) emitHwyStore(args []ast.Expr) {
	if len(args) < 2 {
		t.writef("/* Store: missing args */\n")
		return
	}
	storeFn := t.profile.StoreFn[t.tier]
	vec := t.translateExpr(args[0])
	ptr := t.translateExpr(args[1])
	// NEON Store: vst1q_f32(ptr, vec) — pointer first, vector second
	// AVX Store: _mm256_storeu_ps(ptr, vec) — same convention
	t.writef("%s(%s, %s);\n", storeFn, ptr, vec)
}

// emitHwyStoreExpr returns Store as a string expression (for edge cases).
func (t *CASTTranslator) emitHwyStoreExpr(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* Store: missing args */"
	}
	storeFn := t.profile.StoreFn[t.tier]
	vec := t.translateExpr(args[0])
	ptr := t.translateExpr(args[1])
	return fmt.Sprintf("%s(%s, %s)", storeFn, ptr, vec)
}

// emitHwySet: hwy.Set(val) → vdupq_n_f32(val)
func (t *CASTTranslator) emitHwySet(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* Set: missing args */"
	}
	dupFn := t.profile.DupFn[t.tier]
	val := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", dupFn, val)
}

// emitHwyZero: hwy.Zero[T]() → vdupq_n_f32(0.0f)
func (t *CASTTranslator) emitHwyZero() string {
	dupFn := t.profile.DupFn[t.tier]
	zero := "0.0f"
	if t.elemType == "float64" {
		zero = "0.0"
	}
	return fmt.Sprintf("%s(%s)", dupFn, zero)
}

// emitHwyMulAdd: hwy.MulAdd(a, b, acc) → target-specific FMA.
// Go convention (like AVX): MulAdd(a, b, acc) = a*b + acc
// NEON: vfmaq_f32(acc, a, b) — accumulator first
// AVX: _mm256_fmadd_ps(a, b, acc) — accumulator last
func (t *CASTTranslator) emitHwyMulAdd(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* MulAdd: missing args */"
	}
	fmaFn := t.profile.FmaFn[t.tier]
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	acc := t.translateExpr(args[2])

	if t.profile.FmaArgOrder == "acc_first" {
		// NEON: FMA(acc, a, b)
		return fmt.Sprintf("%s(%s, %s, %s)", fmaFn, acc, a, b)
	}
	// AVX: FMA(a, b, acc)
	return fmt.Sprintf("%s(%s, %s, %s)", fmaFn, a, b, acc)
}

// emitHwyBinaryOp: hwy.Add(a, b) → vaddq_f32(a, b)
func (t *CASTTranslator) emitHwyBinaryOp(fnMap map[string]string, args []ast.Expr) string {
	if len(args) < 2 {
		return "/* binary op: missing args */"
	}
	fn := fnMap[t.tier]
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	return fmt.Sprintf("%s(%s, %s)", fn, a, b)
}

// emitHwyUnaryOp: hwy.Neg(x) → vnegq_f32(x)
func (t *CASTTranslator) emitHwyUnaryOp(fnMap map[string]string, args []ast.Expr) string {
	if len(args) < 1 {
		return "/* unary op: missing args */"
	}
	fn := fnMap[t.tier]
	x := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", fn, x)
}

// ---------------------------------------------------------------------------
// Type inference
// ---------------------------------------------------------------------------

// inferType infers the C type from the RHS of an assignment.
func (t *CASTTranslator) inferType(expr ast.Expr) cVarInfo {
	switch e := expr.(type) {
	case *ast.CallExpr:
		return t.inferCallType(e)
	case *ast.SliceExpr:
		return cVarInfo{cType: t.profile.CType + " *", isPtr: true}
	case *ast.IndexExpr:
		// Array element access → scalar type
		return cVarInfo{cType: t.profile.CType}
	case *ast.BinaryExpr:
		// Arithmetic on scalars → scalar type
		return cVarInfo{cType: t.profile.CType}
	case *ast.BasicLit:
		if e.Kind == token.INT {
			return cVarInfo{cType: "long"}
		}
		return cVarInfo{cType: t.profile.CType}
	case *ast.Ident:
		// Variable reference — look up its type
		if info, ok := t.vars[e.Name]; ok {
			return info
		}
		if info, ok := t.params[e.Name]; ok {
			if info.isSlice {
				return cVarInfo{cType: t.profile.CType + " *", isPtr: true}
			}
			return cVarInfo{cType: t.profile.CType}
		}
		return cVarInfo{cType: t.profile.CType}
	default:
		return cVarInfo{cType: t.profile.CType}
	}
}

// inferCallType infers the return type of a function call.
func (t *CASTTranslator) inferCallType(e *ast.CallExpr) cVarInfo {
	vecType := t.profile.VecTypes[t.tier]

	// Check for hwy.Func calls
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			switch sel.Sel.Name {
			case "Load", "Zero", "Set", "MulAdd", "Add", "Sub", "Mul", "Div",
				"Min", "Max", "Neg", "Abs", "Sqrt":
				return cVarInfo{cType: vecType, isVector: true}
			}
		}
	}

	// Check for v.NumLanes() method calls
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements" {
			return cVarInfo{cType: "long"}
		}
	}

	return cVarInfo{cType: t.profile.CType}
}

// goTypeToCType converts Go type names to C type names.
func (t *CASTTranslator) goTypeToCType(goType string) string {
	switch goType {
	case "int", "int64":
		return "long"
	case "int32":
		return "int"
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "T":
		return t.profile.CType
	default:
		if strings.HasPrefix(goType, "[]") {
			return t.profile.CType + " *"
		}
		return "long" // default for unknown types
	}
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

// writef writes indented formatted output.
func (t *CASTTranslator) writef(format string, args ...any) {
	for range t.indent {
		t.buf.WriteString("    ")
	}
	fmt.Fprintf(t.buf, format, args...)
}

// writefRaw writes formatted output without indentation (for continuing same line).
func (t *CASTTranslator) writefRaw(format string, args ...any) {
	fmt.Fprintf(t.buf, format, args...)
}

// ---------------------------------------------------------------------------
// Eligibility check
// ---------------------------------------------------------------------------

// IsASTCEligible returns true if a function should be translated using the
// AST-walking translator rather than the template-based CEmitter.
// Eligible functions have slice parameters, use hwy.* operations, and are
// NOT composite math functions handled by the template path.
func IsASTCEligible(pf *ParsedFunc) bool {
	// Must have slice params (not Vec→Vec)
	hasSlice := false
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			hasSlice = true
			break
		}
	}
	if !hasSlice {
		return false
	}

	// Must NOT have Vec in signature (those go through IsCEligible)
	if hasVecInSignature(*pf) {
		return false
	}

	// Must use hwy.* operations
	hasHwyOps := false
	for _, call := range pf.HwyCalls {
		if call.Package == "hwy" {
			hasHwyOps = true
			break
		}
	}
	if !hasHwyOps {
		return false
	}

	// Must NOT be a composite math function (those use the template path)
	if mathOpFromFuncName(pf.Name) != "" {
		return false
	}

	// Must have int params (indicates a multi-dimensional function, not a simple
	// array transform). This distinguishes matmul/transpose from GELU/softmax.
	hasIntParam := false
	for _, p := range pf.Params {
		if p.Type == "int" || p.Type == "int64" {
			hasIntParam = true
			break
		}
	}

	return hasIntParam
}
