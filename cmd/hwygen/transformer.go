package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// Transform transforms a parsed function for a specific target and element type.
// It clones the AST, specializes generics, and transforms hwy operations.
func Transform(pf *ParsedFunc, target Target, elemType string) *ast.FuncDecl {
	// Create new function declaration (don't copy Doc - emitter handles comments)
	funcDecl := &ast.FuncDecl{
		Name: ast.NewIdent(pf.Name + target.Suffix()),
		Type: &ast.FuncType{
			Params:  &ast.FieldList{},
			Results: pf.buildResults(elemType),
		},
		Body: cloneBlockStmt(pf.Body),
	}

	// Build parameter list with specialized types
	for _, param := range pf.Params {
		paramType := specializeType(param.Type, pf.TypeParams, elemType)
		field := &ast.Field{
			Names: []*ast.Ident{ast.NewIdent(param.Name)},
			Type:  parseTypeExpr(paramType),
		}
		funcDecl.Type.Params.List = append(funcDecl.Type.Params.List, field)
	}

	// Transform the function body
	ctx := &transformContext{
		target:     target,
		elemType:   elemType,
		typeParams: pf.TypeParams,
		loopInfo:   pf.LoopInfo,
	}
	transformNode(funcDecl.Body, ctx)

	// Post-process to replace NumLanes() calls and ReduceSum() calls
	if target.Name != "Fallback" {
		postProcessSIMD(funcDecl.Body, ctx)
	}

	// Insert tail handling if there's a loop
	if pf.LoopInfo != nil {
		insertTailHandling(funcDecl.Body, pf.LoopInfo, elemType, target, pf.Name, pf.Params)
	}

	return funcDecl
}

type transformContext struct {
	target     Target
	elemType   string
	typeParams []TypeParam
	loopInfo   *LoopInfo
}

// transformNode recursively transforms AST nodes.
func transformNode(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CallExpr:
			transformCallExpr(node, ctx)
			// Also check for type conversions like T(1)
			transformTypeConversion(node, ctx)
		case *ast.DeclStmt:
			// Transform variable declarations
			if genDecl, ok := node.Decl.(*ast.GenDecl); ok {
				transformGenDecl(genDecl, ctx)
			}
		case *ast.AssignStmt:
			// Transform assignments
			transformAssignStmt(node, ctx)
		case *ast.ForStmt:
			// Transform for loop for SIMD (stride, condition)
			transformForStmt(node, ctx)
		}
		return true
	})
}

// transformForStmt transforms for loops for SIMD targets.
// Changes: for ii := 0; ii < size; ii += v.NumLanes()
// To:      for ii := 0; ii+8 <= size; ii += 8
func transformForStmt(stmt *ast.ForStmt, ctx *transformContext) {
	if ctx.target.Name == "Fallback" || ctx.loopInfo == nil {
		return
	}

	lanes := ctx.target.LanesFor(ctx.elemType)

	// Transform post: ii += v.NumLanes() -> ii += lanes
	if assignStmt, ok := stmt.Post.(*ast.AssignStmt); ok {
		if len(assignStmt.Rhs) == 1 {
			if call, ok := assignStmt.Rhs[0].(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
					if sel.Sel.Name == "NumElements" || sel.Sel.Name == "NumLanes" {
						assignStmt.Rhs[0] = &ast.BasicLit{
							Kind:  token.INT,
							Value: strconv.Itoa(lanes),
						}
					}
				}
			}
		}
	}

	// Transform condition: ii < size -> ii+lanes <= size
	if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
		if binExpr.Op == token.LSS {
			// Change ii < size to ii+lanes <= size
			binExpr.Op = token.LEQ
			binExpr.X = &ast.BinaryExpr{
				X:  binExpr.X,
				Op: token.ADD,
				Y: &ast.BasicLit{
					Kind:  token.INT,
					Value: strconv.Itoa(lanes),
				},
			}
		}
	}
}

// transformTypeConversion converts T(1) to float32(1) for generic type parameters.
func transformTypeConversion(call *ast.CallExpr, ctx *transformContext) {
	// Check if this is a type conversion T(value) where T is a type parameter
	ident, ok := call.Fun.(*ast.Ident)
	if !ok {
		return
	}

	// Check if the identifier is a type parameter
	for _, tp := range ctx.typeParams {
		if ident.Name == tp.Name {
			// Replace T with the concrete element type
			ident.Name = ctx.elemType
			return
		}
	}
}

// transformCallExpr transforms hwy.* and contrib.* function calls.
func transformCallExpr(call *ast.CallExpr, ctx *transformContext) {
	selExpr, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}

	ident, ok := selExpr.X.(*ast.Ident)
	if !ok {
		return
	}

	// Handle hwy.* and contrib subpackage calls (math.*, dot.*, matvec.*, algo.*)
	switch ident.Name {
	case "hwy", "contrib", "math", "dot", "matvec", "algo":
		// Continue processing
	default:
		return
	}

	funcName := selExpr.Sel.Name
	opInfo, ok := ctx.target.OpMap[funcName]
	if !ok {
		// Unknown operation, leave as-is
		return
	}

	// Transform based on operation type
	if opInfo.IsMethod {
		transformToMethod(call, funcName, opInfo, ctx)
	} else {
		transformToFunction(call, funcName, opInfo, ctx)
	}
}

// transformToMethod converts hwy.Add(a, b) to a.Add(b) for SIMD targets.
// For Fallback, keeps hwy.Add(a, b) as-is.
func transformToMethod(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	if len(call.Args) < 1 {
		return
	}

	// For fallback, keep hwy calls as-is (don't convert to method calls)
	if ctx.target.Name == "Fallback" {
		// Just update the package name if needed
		if selExpr, ok := call.Fun.(*ast.SelectorExpr); ok {
			selExpr.X = ast.NewIdent("hwy")
		}
		return
	}

	// For SIMD targets, convert to method calls on archsimd types
	switch funcName {
	case "Store":
		// hwy.Store(v, dst) -> v.StoreSlice(dst)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("StoreSlice"),
			}
			call.Args = call.Args[1:]
		}

	case "MaskStore":
		// hwy.MaskStore(mask, v, dst) -> v.MaskStoreSlice(mask, dst)
		if len(call.Args) >= 3 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[1],
				Sel: ast.NewIdent("MaskStoreSlice"),
			}
			call.Args = []ast.Expr{call.Args[0], call.Args[2]}
		}

	case "Neg":
		// hwy.Neg(x) -> archsimd.BroadcastFloat32x8(0).Sub(x) for SIMD
		// (archsimd types don't have a Neg method, so we use 0 - x)
		if len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			// Create archsimd.BroadcastFloat32x8(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("archsimd"),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			call.Fun = &ast.SelectorExpr{
				X:   zeroCall,
				Sel: ast.NewIdent("Sub"),
			}
			// Args stays as [x]
		}

	default:
		// Binary operations: hwy.Add(a, b) -> a.Add(b)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			call.Args = call.Args[1:]
		} else if len(call.Args) == 1 {
			// Other unary operations
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			call.Args = nil
		}
	}
}

// transformToFunction converts hwy.Load(src) to archsimd.LoadFloat32x8Slice(src).
func transformToFunction(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	selExpr := call.Fun.(*ast.SelectorExpr)

	if ctx.target.Name == "Fallback" {
		// For fallback, use the appropriate package
		if opInfo.SubPackage != "" {
			// Contrib functions stay in their subpackage (e.g., math.Exp stays math.Exp)
			selExpr.X = ast.NewIdent(opInfo.SubPackage)
		} else {
			// Core ops use hwy package
			selExpr.X = ast.NewIdent("hwy")
		}
		selExpr.Sel.Name = funcName
		return
	}

	// For SIMD targets, transform to archsimd calls
	var fullName string
	vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)

	switch funcName {
	case "Load":
		fullName = fmt.Sprintf("Load%sSlice", vecTypeName)
		selExpr.X = ast.NewIdent("archsimd")
	case "Set":
		fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
		selExpr.X = ast.NewIdent("archsimd")
	case "Zero":
		fullName = fmt.Sprintf("Zero%s", vecTypeName)
		selExpr.X = ast.NewIdent("archsimd")
	case "MaskLoad":
		fullName = fmt.Sprintf("MaskLoad%sSlice", vecTypeName)
		selExpr.X = ast.NewIdent("archsimd")
	default:
		// For contrib functions, add target and type suffix (e.g., math.Exp_AVX2_F32x8)
		if opInfo.SubPackage != "" {
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent(opInfo.SubPackage) // math, dot, matvec, algo
		} else if opInfo.Package == "hwy" {
			// Core ops from hwy package (e.g., hwy.Sqrt_AVX2_F32x8)
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = opInfo.Name
			selExpr.X = ast.NewIdent("archsimd")
		}
	}

	selExpr.Sel.Name = fullName
}

// getShortTypeName returns the short type name like F32x8 for contrib functions.
func getShortTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	switch elemType {
	case "float32":
		return fmt.Sprintf("F32x%d", lanes)
	case "float64":
		return fmt.Sprintf("F64x%d", lanes)
	case "int32":
		return fmt.Sprintf("I32x%d", lanes)
	case "int64":
		return fmt.Sprintf("I64x%d", lanes)
	default:
		return "Vec"
	}
}

// transformGenDecl transforms variable declarations with generic types.
func transformGenDecl(decl *ast.GenDecl, ctx *transformContext) {
	if decl.Tok != token.VAR && decl.Tok != token.CONST {
		return
	}

	for _, spec := range decl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}

		// Transform type if present
		if valueSpec.Type != nil {
			typeStr := exprToString(valueSpec.Type)
			specialized := specializeType(typeStr, ctx.typeParams, ctx.elemType)
			if specialized != typeStr {
				valueSpec.Type = parseTypeExpr(specialized)
			}
		}
	}
}

// transformAssignStmt transforms assignments, particularly for loop stride calculations.
func transformAssignStmt(stmt *ast.AssignStmt, ctx *transformContext) {
	if ctx.loopInfo == nil {
		return
	}

	// For fallback, don't replace NumLanes with a constant - keep it dynamic
	if ctx.target.Name == "Fallback" {
		return
	}

	// Look for v.NumElements() or similar and replace with constant
	for i, rhs := range stmt.Rhs {
		if call, ok := rhs.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if sel.Sel.Name == "NumElements" || sel.Sel.Name == "NumLanes" {
					// Replace with constant lane count
					lanes := ctx.target.LanesFor(ctx.elemType)
					stmt.Rhs[i] = &ast.BasicLit{
						Kind:  token.INT,
						Value: strconv.Itoa(lanes),
					}
				}
			}
		}
	}
}

// insertTailHandling adds scalar tail handling after the vectorized loop.
func insertTailHandling(body *ast.BlockStmt, loopInfo *LoopInfo, elemType string, target Target, funcName string, params []Param) {
	if body == nil || loopInfo == nil {
		return
	}

	// For fallback, don't add tail handling - the hwy API handles it
	if target.Name == "Fallback" {
		return
	}

	// Find the main loop
	var loopIdx int
	var mainLoop *ast.ForStmt
	for i, stmt := range body.List {
		if forStmt, ok := stmt.(*ast.ForStmt); ok {
			loopIdx = i
			mainLoop = forStmt
			break
		}
	}

	if mainLoop == nil {
		return
	}

	// Declare the iterator before the loop so it's in scope for the tail
	// Change: for ii := 0; ... to: ii := 0; for ; ...
	var initStmt ast.Stmt
	if mainLoop.Init != nil {
		initStmt = mainLoop.Init
		mainLoop.Init = nil
	}

	// Build tail handling that calls the fallback function for remaining elements
	// if ii < size {
	//     BaseSigmoid_fallback(in[ii:size], out[ii:size])
	// }
	fallbackFuncName := funcName + "_fallback"
	// Add type suffix for non-float32 types (matches how generator.go names functions)
	if elemType != "float32" {
		fallbackFuncName = fallbackFuncName + "_" + strings.Title(elemType)
	}

	// Build slice expressions for each parameter: param[ii:size]
	var callArgs []ast.Expr
	for _, param := range params {
		// Create param[ii:size] for slice parameters
		if strings.HasPrefix(param.Type, "[]") {
			sliceExpr := &ast.SliceExpr{
				X:    ast.NewIdent(param.Name),
				Low:  ast.NewIdent(loopInfo.Iterator),
				High: ast.NewIdent(loopInfo.End),
			}
			callArgs = append(callArgs, sliceExpr)
		}
	}

	// Create the fallback call: BaseSigmoid_fallback(in[ii:size], out[ii:size])
	fallbackCall := &ast.CallExpr{
		Fun:  ast.NewIdent(fallbackFuncName),
		Args: callArgs,
	}

	// Wrap in if statement: if ii < size { ... }
	tailIf := &ast.IfStmt{
		Cond: &ast.BinaryExpr{
			X:  ast.NewIdent(loopInfo.Iterator),
			Op: token.LSS,
			Y:  ast.NewIdent(loopInfo.End),
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				&ast.ExprStmt{X: fallbackCall},
			},
		},
	}

	// Insert init statement, main loop, and tail handling
	newStmts := make([]ast.Stmt, 0, len(body.List)+2)
	newStmts = append(newStmts, body.List[:loopIdx]...)
	if initStmt != nil {
		newStmts = append(newStmts, initStmt)
	}
	newStmts = append(newStmts, mainLoop)
	newStmts = append(newStmts, tailIf)
	newStmts = append(newStmts, body.List[loopIdx+1:]...)
	body.List = newStmts
}

// specializeType replaces generic type parameters with concrete types.
func specializeType(typeStr string, typeParams []TypeParam, elemType string) string {
	for _, tp := range typeParams {
		// Replace []T with []float32, etc.
		typeStr = strings.ReplaceAll(typeStr, "[]"+tp.Name, "[]"+elemType)
		typeStr = strings.ReplaceAll(typeStr, tp.Name, elemType)
	}
	return typeStr
}

// getVectorTypeName returns the vector type name for archsimd functions.
func getVectorTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	switch elemType {
	case "float32":
		return fmt.Sprintf("Float32x%d", lanes)
	case "float64":
		return fmt.Sprintf("Float64x%d", lanes)
	case "int32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "int64":
		return fmt.Sprintf("Int64x%d", lanes)
	default:
		return "Vec"
	}
}

// parseTypeExpr converts a type string back to an AST expression.
func parseTypeExpr(typeStr string) ast.Expr {
	// Handle slice types
	if strings.HasPrefix(typeStr, "[]") {
		return &ast.ArrayType{
			Elt: parseTypeExpr(typeStr[2:]),
		}
	}

	// Handle pointer types
	if strings.HasPrefix(typeStr, "*") {
		return &ast.StarExpr{
			X: parseTypeExpr(typeStr[1:]),
		}
	}

	// Handle qualified names (pkg.Type)
	if idx := strings.Index(typeStr, "."); idx >= 0 {
		return &ast.SelectorExpr{
			X:   ast.NewIdent(typeStr[:idx]),
			Sel: ast.NewIdent(typeStr[idx+1:]),
		}
	}

	// Simple identifier
	return ast.NewIdent(typeStr)
}

// cloneBlockStmt creates a deep copy of a block statement.
func cloneBlockStmt(block *ast.BlockStmt) *ast.BlockStmt {
	if block == nil {
		return nil
	}

	newBlock := &ast.BlockStmt{
		List: make([]ast.Stmt, len(block.List)),
	}

	for i, stmt := range block.List {
		newBlock.List[i] = cloneStmt(stmt)
	}

	return newBlock
}

// cloneStmt creates a deep copy of a statement.
func cloneStmt(stmt ast.Stmt) ast.Stmt {
	if stmt == nil {
		return nil
	}

	switch s := stmt.(type) {
	case *ast.ExprStmt:
		return &ast.ExprStmt{X: cloneExpr(s.X)}
	case *ast.AssignStmt:
		return cloneAssignStmt(s)
	case *ast.DeclStmt:
		return &ast.DeclStmt{Decl: cloneDecl(s.Decl)}
	case *ast.ReturnStmt:
		return cloneReturnStmt(s)
	case *ast.ForStmt:
		return cloneForStmt(s)
	case *ast.IfStmt:
		return cloneIfStmt(s)
	case *ast.IncDecStmt:
		return &ast.IncDecStmt{X: cloneExpr(s.X), Tok: s.Tok}
	case *ast.BranchStmt:
		return &ast.BranchStmt{Tok: s.Tok, Label: s.Label}
	case *ast.BlockStmt:
		return cloneBlockStmt(s)
	case *ast.RangeStmt:
		return &ast.RangeStmt{
			Key:   cloneExpr(s.Key),
			Value: cloneExpr(s.Value),
			Tok:   s.Tok,
			X:     cloneExpr(s.X),
			Body:  cloneBlockStmt(s.Body),
		}
	default:
		// For other statement types, return as-is
		return stmt
	}
}

// cloneExpr creates a deep copy of an expression.
func cloneExpr(expr ast.Expr) ast.Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return &ast.Ident{Name: e.Name}
	case *ast.BasicLit:
		return &ast.BasicLit{Kind: e.Kind, Value: e.Value}
	case *ast.SelectorExpr:
		return &ast.SelectorExpr{
			X:   cloneExpr(e.X),
			Sel: ast.NewIdent(e.Sel.Name),
		}
	case *ast.CallExpr:
		args := make([]ast.Expr, len(e.Args))
		for i, arg := range e.Args {
			args[i] = cloneExpr(arg)
		}
		return &ast.CallExpr{
			Fun:  cloneExpr(e.Fun),
			Args: args,
		}
	case *ast.BinaryExpr:
		return &ast.BinaryExpr{
			X:  cloneExpr(e.X),
			Op: e.Op,
			Y:  cloneExpr(e.Y),
		}
	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			Op: e.Op,
			X:  cloneExpr(e.X),
		}
	case *ast.ParenExpr:
		return &ast.ParenExpr{X: cloneExpr(e.X)}
	case *ast.IndexExpr:
		return &ast.IndexExpr{
			X:     cloneExpr(e.X),
			Index: cloneExpr(e.Index),
		}
	case *ast.SliceExpr:
		return &ast.SliceExpr{
			X:      cloneExpr(e.X),
			Low:    cloneExpr(e.Low),
			High:   cloneExpr(e.High),
			Max:    cloneExpr(e.Max),
			Slice3: e.Slice3,
		}
	case *ast.StarExpr:
		return &ast.StarExpr{X: cloneExpr(e.X)}
	case *ast.ArrayType:
		return &ast.ArrayType{
			Len: cloneExpr(e.Len),
			Elt: cloneExpr(e.Elt),
		}
	case *ast.CompositeLit:
		elts := make([]ast.Expr, len(e.Elts))
		for i, elt := range e.Elts {
			elts[i] = cloneExpr(elt)
		}
		return &ast.CompositeLit{
			Type: cloneExpr(e.Type),
			Elts: elts,
		}
	default:
		// For unsupported types, return as-is (may cause issues for complex expressions)
		return expr
	}
}

// cloneAssignStmt clones an assignment statement.
func cloneAssignStmt(stmt *ast.AssignStmt) *ast.AssignStmt {
	newStmt := &ast.AssignStmt{
		Lhs: make([]ast.Expr, len(stmt.Lhs)),
		Rhs: make([]ast.Expr, len(stmt.Rhs)),
		Tok: stmt.Tok,
	}
	for i, lhs := range stmt.Lhs {
		newStmt.Lhs[i] = cloneExpr(lhs)
	}
	for i, rhs := range stmt.Rhs {
		newStmt.Rhs[i] = cloneExpr(rhs)
	}
	return newStmt
}

// cloneDecl clones a declaration.
func cloneDecl(decl ast.Decl) ast.Decl {
	if decl == nil {
		return nil
	}

	switch d := decl.(type) {
	case *ast.GenDecl:
		newSpecs := make([]ast.Spec, len(d.Specs))
		for i, spec := range d.Specs {
			newSpecs[i] = cloneSpec(spec)
		}
		return &ast.GenDecl{
			Tok:   d.Tok,
			Specs: newSpecs,
		}
	default:
		return decl
	}
}

// cloneSpec clones a declaration spec (e.g., variable declaration).
func cloneSpec(spec ast.Spec) ast.Spec {
	if spec == nil {
		return nil
	}

	switch s := spec.(type) {
	case *ast.ValueSpec:
		var newValues []ast.Expr
		if len(s.Values) > 0 {
			newValues = make([]ast.Expr, len(s.Values))
			for i, v := range s.Values {
				newValues[i] = cloneExpr(v)
			}
		}
		newNames := make([]*ast.Ident, len(s.Names))
		for i, n := range s.Names {
			newNames[i] = &ast.Ident{Name: n.Name}
		}
		return &ast.ValueSpec{
			Names:  newNames,
			Type:   cloneExpr(s.Type),
			Values: newValues,
		}
	default:
		return spec
	}
}

// cloneReturnStmt clones a return statement.
func cloneReturnStmt(stmt *ast.ReturnStmt) *ast.ReturnStmt {
	newStmt := &ast.ReturnStmt{
		Results: make([]ast.Expr, len(stmt.Results)),
	}
	for i, result := range stmt.Results {
		newStmt.Results[i] = cloneExpr(result)
	}
	return newStmt
}

// cloneForStmt clones a for loop.
func cloneForStmt(stmt *ast.ForStmt) *ast.ForStmt {
	return &ast.ForStmt{
		Init: cloneStmt(stmt.Init),
		Cond: cloneExpr(stmt.Cond),
		Post: cloneStmt(stmt.Post),
		Body: cloneBlockStmt(stmt.Body),
	}
}

// cloneIfStmt clones an if statement.
func cloneIfStmt(stmt *ast.IfStmt) *ast.IfStmt {
	return &ast.IfStmt{
		Init: cloneStmt(stmt.Init),
		Cond: cloneExpr(stmt.Cond),
		Body: cloneBlockStmt(stmt.Body),
		Else: cloneStmt(stmt.Else),
	}
}

// buildResults builds the return type list for a function.
func (pf *ParsedFunc) buildResults(elemType string) *ast.FieldList {
	if len(pf.Returns) == 0 {
		return nil
	}

	fieldList := &ast.FieldList{
		List: make([]*ast.Field, 0, len(pf.Returns)),
	}

	for _, ret := range pf.Returns {
		retType := specializeType(ret.Type, pf.TypeParams, elemType)
		field := &ast.Field{
			Type: parseTypeExpr(retType),
		}
		if ret.Name != "" {
			field.Names = []*ast.Ident{ast.NewIdent(ret.Name)}
		}
		fieldList.List = append(fieldList.List, field)
	}

	return fieldList
}

// postProcessSIMD walks the AST and replaces NumLanes() calls with constants
// and transforms ReduceSum() calls to store+sum patterns.
func postProcessSIMD(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	lanes := ctx.target.LanesFor(ctx.elemType)
	vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)

	// Walk all statements and expressions, replacing as needed
	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.IfStmt:
			// Replace comparisons like: remaining >= v.NumLanes()
			if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
				replaceNumLanesInExpr(binExpr, lanes)
			}
		case *ast.AssignStmt:
			// Replace: sum += v.ReduceSum() or sum += hwy.ReduceSum(v)
			for i, rhs := range stmt.Rhs {
				if call, ok := rhs.(*ast.CallExpr); ok {
					if isReduceSumCall(call) {
						// Transform to store + sum pattern
						stmt.Rhs[i] = createReduceSumExpr(call, lanes, vecTypeName, ctx.elemType)
					}
				}
			}
		case *ast.ExprStmt:
			// Handle standalone expressions if needed
		}
		return true
	})
}

// replaceNumLanesInExpr replaces v.NumLanes() with a constant in a binary expression.
func replaceNumLanesInExpr(binExpr *ast.BinaryExpr, lanes int) {
	// Check RHS
	if call, ok := binExpr.Y.(*ast.CallExpr); ok {
		if isNumLanesCall(call) {
			binExpr.Y = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
	// Check LHS (less common but possible)
	if call, ok := binExpr.X.(*ast.CallExpr); ok {
		if isNumLanesCall(call) {
			binExpr.X = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
}

// isNumLanesCall checks if a call expression is v.NumLanes() or v.NumElements().
func isNumLanesCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements"
}

// isReduceSumCall checks if a call expression is v.ReduceSum() or hwy.ReduceSum(v).
func isReduceSumCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	if sel.Sel.Name == "ReduceSum" {
		return true
	}
	return false
}

// createReduceSumExpr creates an expression that stores the vector and sums elements.
// For now, we generate a function call that we'll define in a helper.
// Actually, archsimd vectors don't have a built-in ReduceSum, so we need to
// generate inline code that stores to temp and sums.
// Since we can't inject statements here, we'll generate a compound expression.
func createReduceSumExpr(call *ast.CallExpr, lanes int, vecTypeName, elemType string) ast.Expr {
	// Get the vector argument
	var vecExpr ast.Expr
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if _, ok := sel.X.(*ast.Ident); ok {
			// It's v.ReduceSum() - the receiver is the vector
			vecExpr = sel.X
		}
	}
	if vecExpr == nil && len(call.Args) > 0 {
		// It's hwy.ReduceSum(v) - first arg is the vector
		vecExpr = call.Args[0]
	}
	if vecExpr == nil {
		return call // Can't transform, leave as-is
	}

	// Generate a function call to a helper we'll need to add
	// For now, generate inline reduction: func() T { var t [N]T; v.StoreSlice(t[:]); return t[0]+t[1]+... }()
	// This is verbose but works without injecting statements

	// Build: t[0] + t[1] + ... + t[lanes-1]
	var sumExpr ast.Expr
	for i := 0; i < lanes; i++ {
		indexExpr := &ast.IndexExpr{
			X: ast.NewIdent("_simd_temp"),
			Index: &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(i),
			},
		}
		if sumExpr == nil {
			sumExpr = indexExpr
		} else {
			sumExpr = &ast.BinaryExpr{
				X:  sumExpr,
				Op: token.ADD,
				Y:  indexExpr,
			}
		}
	}

	// Build the function literal:
	// func() elemType {
	//     var _simd_temp [lanes]elemType
	//     vec.StoreSlice(_simd_temp[:])
	//     return t[0] + t[1] + ...
	// }()
	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: ast.NewIdent(elemType)},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				// var _simd_temp [lanes]elemType
				&ast.DeclStmt{
					Decl: &ast.GenDecl{
						Tok: token.VAR,
						Specs: []ast.Spec{
							&ast.ValueSpec{
								Names: []*ast.Ident{ast.NewIdent("_simd_temp")},
								Type: &ast.ArrayType{
									Len: &ast.BasicLit{
										Kind:  token.INT,
										Value: strconv.Itoa(lanes),
									},
									Elt: ast.NewIdent(elemType),
								},
							},
						},
					},
				},
				// vec.StoreSlice(_simd_temp[:])
				&ast.ExprStmt{
					X: &ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X:   vecExpr,
							Sel: ast.NewIdent("StoreSlice"),
						},
						Args: []ast.Expr{
							&ast.SliceExpr{
								X: ast.NewIdent("_simd_temp"),
							},
						},
					},
				},
				// return sum expression
				&ast.ReturnStmt{
					Results: []ast.Expr{sumExpr},
				},
			},
		},
	}

	// Call the function literal immediately
	return &ast.CallExpr{
		Fun: funcLit,
	}
}
