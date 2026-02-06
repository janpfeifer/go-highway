package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ajroetker/go-highway/cmd/hwygen/ir"
)

// runCMode generates C code for requested targets, compiles with GOAT,
// and generates Go wrapper functions.
func (g *Generator) runCMode(result *ParseResult) error {
	// Parse requested targets
	var targets []Target
	for _, name := range g.Targets {
		target, err := GetTarget(name)
		if err != nil {
			return err
		}
		// Skip fallback - no C generation needed
		if target.Name == "Fallback" {
			continue
		}
		targets = append(targets, target)
	}

	if len(targets) == 0 {
		return fmt.Errorf("no valid C generation targets specified (got: %v)", g.Targets)
	}

	// Collect all eligible functions
	var vecFuncs []ParsedFunc   // Vec→Vec functions
	var sliceFuncs []ParsedFunc // Slice→Slice functions (composite like GELU)
	var astFuncs []ParsedFunc   // Functions for AST translation (matmul, transpose, etc.)

	for _, pf := range result.Funcs {
		if IsASTCEligible(&pf) {
			astFuncs = append(astFuncs, pf)
		} else if IsCEligible(&pf) {
			vecFuncs = append(vecFuncs, pf)
		} else if IsSliceFunction(&pf) {
			sliceFuncs = append(sliceFuncs, pf)
		}
	}

	totalFuncs := len(vecFuncs) + len(sliceFuncs) + len(astFuncs)
	if totalFuncs == 0 {
		return fmt.Errorf("no C-eligible functions found")
	}

	fmt.Printf("Found %d C-eligible functions:\n", totalFuncs)
	for _, pf := range vecFuncs {
		fmt.Printf("  - %s (Vec→Vec)\n", pf.Name)
	}
	for _, pf := range sliceFuncs {
		fmt.Printf("  - %s (Slice→Slice)\n", pf.Name)
	}

	// If fusion mode is enabled, use IR-based generation for slice functions
	if g.FusionMode && len(sliceFuncs) > 0 {
		fmt.Println("\nUsing IR-based fusion optimization...")
		if err := g.runFusionCMode(result, sliceFuncs, targets); err != nil {
			return fmt.Errorf("fusion mode: %w", err)
		}
		// Continue with non-slice functions using standard path
		sliceFuncs = nil
	}
	for _, pf := range astFuncs {
		fmt.Printf("  - %s (AST→C)\n", pf.Name)
	}

	// Process each target
	for _, target := range targets {
		fmt.Printf("\nGenerating C for target: %s\n", target.Name)

		// Track generated C files for GOAT compilation
		var cFiles []string

		// Generate C code for Vec→Vec functions
		for _, pf := range vecFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue // No profile for this target/type combo
				}
				emitter := NewCEmitter(g.PackageOut, elemType, target)
				emitter.profile = profile
				cFile, err := emitter.EmitC(&pf, g.OutputDir)
				if err != nil {
					return fmt.Errorf("emit C for %s (%s, %s): %w", pf.Name, elemType, target.Name, err)
				}
				cFiles = append(cFiles, cFile)
			}
		}

		// Generate C code for Slice→Slice (composite) functions
		for _, pf := range sliceFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue // No profile for this target/type combo
				}
				emitter := NewCEmitter(g.PackageOut, elemType, target)
				emitter.profile = profile
				cFile, err := emitter.EmitCompositeC(&pf, g.OutputDir)
				if err != nil {
					return fmt.Errorf("emit composite C for %s (%s, %s): %w", pf.Name, elemType, target.Name, err)
				}
				cFiles = append(cFiles, cFile)
			}
		}

		// Generate C code for AST-translated functions (matmul, transpose, etc.)
		for _, pf := range astFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue
				}
				// AST translator emits native arithmetic — skip types that
				// require promoted math unless they have native SIMD arithmetic.
				if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
					continue
				}
				emitter := NewCEmitter(g.PackageOut, elemType, target)
				emitter.profile = profile
				cFile, err := emitter.EmitASTTranslatedC(&pf, g.OutputDir)
				if err != nil {
					return fmt.Errorf("emit AST C for %s (%s, %s): %w", pf.Name, elemType, target.Name, err)
				}
				cFiles = append(cFiles, cFile)
			}
		}

		if len(cFiles) == 0 {
			continue
		}

		// If -asm mode, compile C files with GOAT and generate wrappers
		if g.AsmMode {
			fmt.Printf("Compiling %d C files with GOAT...\n", len(cFiles))
			for _, cFile := range cFiles {
				profile := getCProfileForFile(cFile, target)
				if err := runGOAT(cFile, profile); err != nil {
					return fmt.Errorf("GOAT compile %s: %w", cFile, err)
				}
				fmt.Printf("  Compiled: %s\n", filepath.Base(cFile))
			}

			// Clean up C and object files (Go build doesn't like them)
			for _, cFile := range cFiles {
				os.Remove(cFile)
				os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
			}

			// Generate unified wrapper file for this target
			allFuncs := make([]ParsedFunc, 0, len(vecFuncs)+len(sliceFuncs)+len(astFuncs))
			allFuncs = append(allFuncs, vecFuncs...)
			allFuncs = append(allFuncs, sliceFuncs...)
			allFuncs = append(allFuncs, astFuncs...)
			if err := g.emitCWrappers(allFuncs, target); err != nil {
				return fmt.Errorf("emit wrappers for %s: %w", target.Name, err)
			}
		} else {
			fmt.Printf("Generated %d C files (use -asm to compile to Go assembly)\n", len(cFiles))
		}
	}

	return nil
}

// runFusionCMode generates C code with IR-based fusion optimization.
// This path is used when -fusion flag is enabled.
func (g *Generator) runFusionCMode(result *ParseResult, sliceFuncs []ParsedFunc, targets []Target) error {
	// Find module root for cross-package resolution
	moduleRoot, moduleName, err := FindModuleRoot(g.OutputDir)
	if err != nil {
		fmt.Printf("Warning: could not find module root: %v\n", err)
		// Fall back to direct translation without cross-package resolution
		moduleRoot = g.OutputDir
		moduleName = "go-highway"
	}

	// Create function registry for cross-package resolution
	registry := NewFunctionRegistry(moduleRoot, moduleName)

	for _, target := range targets {
		fmt.Printf("\nGenerating C for target: %s (with fusion)\n", target.Name)

		var cFiles []string

		for _, pf := range sliceFuncs {
			elemTypes := getCElemTypes(&pf)
			for _, elemType := range elemTypes {
				profile := GetCProfile(target.Name, elemType)
				if profile == nil {
					continue
				}

				// Skip types that require promoted math and don't have native arithmetic
				if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
					continue
				}

				// Build IR from the parsed function
				builder := ir.NewBuilder(
					ir.WithImports(result.Imports),
					ir.WithElemType(elemType),
					ir.WithResolver(registry),
				)

				irPF := &ir.ParsedFunc{
					Name: pf.Name,
					Body: pf.Body,
				}
				for _, tp := range pf.TypeParams {
					irPF.TypeParams = append(irPF.TypeParams, ir.TypeParamInput{
						Name:       tp.Name,
						Constraint: tp.Constraint,
					})
				}
				for _, p := range pf.Params {
					irPF.Params = append(irPF.Params, ir.ParamInput{
						Name: p.Name,
						Type: p.Type,
					})
				}
				for _, r := range pf.Returns {
					irPF.Returns = append(irPF.Returns, ir.ParamInput{
						Name: r.Name,
						Type: r.Type,
					})
				}

				irFunc, err := builder.Build(irPF)
				if err != nil {
					return fmt.Errorf("build IR for %s: %w", pf.Name, err)
				}

				// Run data flow analysis
				ir.Analyze(irFunc)

				// Apply fusion rules
				ir.ApplyFusionRules(irFunc)

				// Print fusion statistics if verbose
				if g.Verbose {
					stats := ir.ComputeFusionStats(irFunc)
					fmt.Printf("  %s: %d→%d passes, %d allocs eliminated, %d fusion groups\n",
						pf.Name, stats.OriginalPasses, stats.FusedPasses,
						stats.EliminatedAllocs, stats.FusionGroups)
				}

				// Create profile adapter for IR emitter
				profileAdapter := newCProfileAdapter(profile)

				// Emit C code from IR
				emitter := ir.NewEmitter(profileAdapter)
				cCode := emitter.EmitFunction(irFunc)

				// Write C file
				cFileName := fmt.Sprintf("%s_%s_%s.c",
					strings.ToLower(pf.Name),
					elemType,
					strings.ToLower(target.Name))
				cFilePath := filepath.Join(g.OutputDir, cFileName)

				if err := os.WriteFile(cFilePath, []byte(cCode), 0o644); err != nil {
					return fmt.Errorf("write C file: %w", err)
				}

				cFiles = append(cFiles, cFilePath)
				fmt.Printf("  Generated: %s\n", cFileName)
			}
		}

		if len(cFiles) == 0 {
			continue
		}

		if g.AsmMode {
			fmt.Printf("Compiling %d C files with GOAT...\n", len(cFiles))
			for _, cFile := range cFiles {
				profile := getCProfileForFile(cFile, target)
				if err := runGOAT(cFile, profile); err != nil {
					return fmt.Errorf("GOAT compile %s: %w", cFile, err)
				}
				fmt.Printf("  Compiled: %s\n", filepath.Base(cFile))
			}

			// Clean up C and object files
			for _, cFile := range cFiles {
				os.Remove(cFile)
				os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")
			}

			// Generate wrappers
			if err := g.emitCWrappers(sliceFuncs, target); err != nil {
				return fmt.Errorf("emit wrappers for %s: %w", target.Name, err)
			}
		} else {
			fmt.Printf("Generated %d C files with fusion (use -asm to compile)\n", len(cFiles))
		}
	}

	return nil
}

// cProfileAdapter adapts CIntrinsicProfile to ir.CProfile interface.
type cProfileAdapter struct {
	p *CIntrinsicProfile
}

func newCProfileAdapter(p *CIntrinsicProfile) *cProfileAdapter {
	return &cProfileAdapter{p: p}
}

func (a *cProfileAdapter) GetTier() string {
	for _, t := range a.p.Tiers {
		if !t.IsScalar {
			return t.Name
		}
	}
	return "q"
}

func (a *cProfileAdapter) GetLanes() int {
	for _, t := range a.p.Tiers {
		if !t.IsScalar {
			return t.Lanes
		}
	}
	return 4
}

func (a *cProfileAdapter) GetVecType(tier string) string {
	if vt, ok := a.p.VecTypes[tier]; ok {
		return vt
	}
	return "float32x4_t"
}

func (a *cProfileAdapter) GetScalarType() string {
	if a.p.ScalarArithType != "" {
		return a.p.ScalarArithType
	}
	return a.p.CType
}

func (a *cProfileAdapter) GetLoadFn(tier string) string {
	if fn, ok := a.p.LoadFn[tier]; ok {
		return fn
	}
	return "vld1q_f32"
}

func (a *cProfileAdapter) GetStoreFn(tier string) string {
	if fn, ok := a.p.StoreFn[tier]; ok {
		return fn
	}
	return "vst1q_f32"
}

func (a *cProfileAdapter) GetIntrinsic(op, tier string) string {
	var fnMap map[string]string
	switch op {
	case "Add":
		fnMap = a.p.AddFn
	case "Sub":
		fnMap = a.p.SubFn
	case "Mul":
		fnMap = a.p.MulFn
	case "Div":
		fnMap = a.p.DivFn
	case "MulAdd":
		fnMap = a.p.FmaFn
	case "Neg":
		fnMap = a.p.NegFn
	case "Abs":
		fnMap = a.p.AbsFn
	case "Sqrt":
		fnMap = a.p.SqrtFn
	case "Min":
		fnMap = a.p.MinFn
	case "Max":
		fnMap = a.p.MaxFn
	case "ReduceSum":
		fnMap = a.p.ReduceSumFn
	case "ReduceMin":
		fnMap = a.p.ReduceMinFn
	case "ReduceMax":
		fnMap = a.p.ReduceMaxFn
	case "Set", "Const":
		fnMap = a.p.DupFn
	default:
		return ""
	}
	if fn, ok := fnMap[tier]; ok {
		return fn
	}
	return ""
}

func (a *cProfileAdapter) GetFmaArgOrder() string {
	if a.p.FmaArgOrder != "" {
		return a.p.FmaArgOrder
	}
	return "acc_last"
}

func (a *cProfileAdapter) GetInlineHelpers() string {
	return strings.Join(a.p.InlineHelpers, "\n")
}

func (a *cProfileAdapter) RequiresCast() bool {
	return a.p.CastExpr != ""
}

func (a *cProfileAdapter) GetCastExpr() string {
	return a.p.CastExpr
}

func (a *cProfileAdapter) GetZeroInit(tier string) string {
	// Create a zero vector
	if dupFn, ok := a.p.DupFn[tier]; ok {
		return dupFn + "(0)"
	}
	return "{0}"
}

// getCElemTypes returns the concrete element types for C code generation.
// This includes f16/bf16 types when the constraint allows them.
// For non-generic functions, it infers the element type from slice parameters.
func getCElemTypes(pf *ParsedFunc) []string {
	if len(pf.TypeParams) > 0 {
		return GetConcreteTypes(pf.TypeParams[0].Constraint)
	}
	// For non-generic functions, detect element type from slice param types
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			elemType := strings.TrimPrefix(p.Type, "[]")
			switch elemType {
			case "uint64":
				return []string{"uint64"}
			case "uint32":
				return []string{"uint32"}
			case "uint8", "byte":
				return []string{"uint8"}
			case "float64":
				return []string{"float64"}
			case "float32":
				return []string{"float32"}
			}
		}
	}
	return []string{"float32"}
}

// IsSliceFunction checks if a function operates on slices (not Vec).
// These are composite functions like GELU that process entire arrays.
func IsSliceFunction(pf *ParsedFunc) bool {
	hasSliceParam := false
	for _, param := range pf.Params {
		if strings.HasPrefix(param.Type, "[]") {
			hasSliceParam = true
			break
		}
	}
	// Must have slice params and NOT have Vec in signature
	return hasSliceParam && !hasVecInSignature(*pf)
}

// getCProfileForFile determines the CIntrinsicProfile for a generated C file
// based on its filename convention.
func getCProfileForFile(cFile string, target Target) *CIntrinsicProfile {
	base := filepath.Base(cFile)
	for _, et := range []string{
		"float16", "hwy.Float16", "bfloat16", "hwy.BFloat16",
		"float32", "float64",
		"uint64", "uint32", "uint8",
	} {
		suffix := cTypeSuffix(et)
		if strings.Contains(base, "_"+suffix+"_") {
			p := GetCProfile(target.Name, et)
			if p != nil {
				return p
			}
		}
	}
	// Default to f32
	return GetCProfile(target.Name, "float32")
}

// runGOAT invokes the GOAT tool to compile a C file to Go assembly.
// It uses `go tool github.com/gorse-io/goat` which requires goat to be
// declared as a tool dependency in go.mod (via `go get -tool`).
func runGOAT(cFile string, profile *CIntrinsicProfile) error {
	// Use the Go binary from GOROOT (same toolchain that built hwygen)
	goBin := filepath.Join(runtime.GOROOT(), "bin", "go")

	// Get absolute path for C file since we run from module root
	absCFile, err := filepath.Abs(cFile)
	if err != nil {
		return fmt.Errorf("abs path: %w", err)
	}

	// Find module root (directory containing go.mod with tool directive)
	modRoot, err := findModuleRoot()
	if err != nil {
		return fmt.Errorf("find module root: %w", err)
	}

	// Build GOAT args from profile
	args := []string{"tool", "github.com/gorse-io/goat", absCFile,
		"-O3",
		"-o", filepath.Dir(absCFile),
	}

	if profile != nil {
		args = append(args, fmt.Sprintf("-e=--target=%s", profile.GoatTarget))
		for _, flag := range profile.GoatExtraFlags {
			args = append(args, "-e="+flag)
		}
	} else {
		// Fallback to arm64 NEON
		args = append(args,
			"-e=--target=arm64",
			"-e=-march=armv8-a+simd+fp",
		)
	}
	args = append(args, "-e=-fno-builtin-memset")

	cmd := exec.Command(goBin, args...)
	cmd.Dir = modRoot
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, string(output))
	}
	return nil
}

// findModuleRoot walks up from the current directory to find go.mod.
func findModuleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found")
		}
		dir = parent
	}
}

// goatPackageName derives the Go package name from a directory path,
// matching GOAT's behavior of using the directory basename.
func goatPackageName(dir string) string {
	// Handle "." by resolving to absolute path first
	if dir == "." {
		absDir, err := filepath.Abs(dir)
		if err == nil {
			dir = absDir
		}
	}
	base := filepath.Base(dir)
	// Replace hyphens with underscores (GOAT behavior)
	return strings.ReplaceAll(base, "-", "_")
}

// emitCWrappers generates a Go file with wrapper functions that call the assembly.
func (g *Generator) emitCWrappers(funcs []ParsedFunc, target Target) error {
	var buf bytes.Buffer

	// GOAT derives package name from output directory, so we must match that
	pkgName := goatPackageName(g.OutputDir)

	// Determine build tag and file suffix based on target
	buildTag := target.BuildTag
	if buildTag == "" {
		buildTag = "!noasm"
	} else {
		buildTag = "!noasm && " + buildTag
	}
	archSuffix := target.Arch()
	targetSuffix := strings.ToLower(target.Name)

	// Check if we need the hwy import (for f16/bf16 AST-translated functions
	// that actually have native math — promoted math types are skipped)
	needsHwy := false
	for _, pf := range funcs {
		if IsASTCEligible(&pf) {
			for _, et := range getCElemTypes(&pf) {
				if isHalfPrecisionType(et) {
					profile := GetCProfile(target.Name, et)
					if profile != nil && (profile.MathStrategy != "promoted" || profile.NativeArithmetic) {
						needsHwy = true
						break
					}
				}
			}
		}
		if needsHwy {
			break
		}
	}

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", pkgName)
	if needsHwy {
		fmt.Fprintf(&buf, "import (\n")
		fmt.Fprintf(&buf, "\t\"unsafe\"\n\n")
		fmt.Fprintf(&buf, "\t\"github.com/ajroetker/go-highway/hwy\"\n")
		fmt.Fprintf(&buf, ")\n\n")
	} else {
		fmt.Fprintf(&buf, "import \"unsafe\"\n\n")
	}

	// Emit wrapper functions
	fmt.Fprintf(&buf, "// Public wrapper functions\n")
	for _, pf := range funcs {
		elemTypes := getCElemTypes(&pf)
		for _, elemType := range elemTypes {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			if IsASTCEligible(&pf) {
				// Skip wrapper for types that didn't get C/asm generated
				if profile.MathStrategy == "promoted" && !profile.NativeArithmetic {
					continue
				}
				// Check if function has struct pointer params - these need special
				// wrapper generation to convert Go struct layout to C struct layout
				if hasStructPtrParams(&pf) {
					emitStructPtrAsmWrapper(&buf, &pf, elemType, target)
				} else {
					emitASTCWrapperFunc(&buf, &pf, elemType, targetSuffix)
				}
			} else {
				emitCWrapperFunc(&buf, &pf, elemType, targetSuffix)
			}
		}
	}

	// Format and write
	filename := filepath.Join(g.OutputDir, fmt.Sprintf("c_wrappers_%s_%s.go", targetSuffix, archSuffix))
	if err := os.WriteFile(filename, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("write wrappers: %w", err)
	}

	fmt.Printf("Generated: %s\n", filename)
	return nil
}

// emitCWrapperFunc generates a single wrapper function.
func emitCWrapperFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType, targetSuffix string) {
	publicName := buildCPublicName(pf.Name, elemType)
	asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)
	sliceType := cWrapperSliceType(elemType)

	// Determine parameter names based on function signature
	inputParam := "input"
	outputParam := "result"

	// Check if it's an in-place operation (modifies input) or has separate output
	hasOutput := false
	for _, param := range pf.Params {
		if param.Name == "output" || param.Name == "result" {
			hasOutput = true
			outputParam = param.Name
			break
		}
	}

	fmt.Fprintf(buf, "// %s computes %s for entire arrays using %s SIMD.\n", publicName, pf.Name, strings.ToUpper(targetSuffix))
	if hasOutput {
		fmt.Fprintf(buf, "func %s(%s, %s %s) {\n", publicName, inputParam, outputParam, sliceType)
	} else {
		fmt.Fprintf(buf, "func %s(%s %s) %s {\n", publicName, inputParam, sliceType, sliceType)
		fmt.Fprintf(buf, "\t%s := make(%s, len(%s))\n", outputParam, sliceType, inputParam)
	}

	fmt.Fprintf(buf, "\tif len(%s) == 0 {\n", inputParam)
	if hasOutput {
		fmt.Fprintf(buf, "\t\treturn\n")
	} else {
		fmt.Fprintf(buf, "\t\treturn %s\n", outputParam)
	}
	fmt.Fprintf(buf, "\t}\n")

	fmt.Fprintf(buf, "\tn := int64(len(%s))\n", inputParam)
	fmt.Fprintf(buf, "\t%s(unsafe.Pointer(&%s[0]), unsafe.Pointer(&%s[0]), unsafe.Pointer(&n))\n",
		asmName, inputParam, outputParam)

	if !hasOutput {
		fmt.Fprintf(buf, "\treturn %s\n", outputParam)
	}
	fmt.Fprintf(buf, "}\n\n")
}

// buildCPublicName creates the public function name.
// BaseExpVec -> ExpCF32, BaseGELU -> GELUCF32
func buildCPublicName(baseName, elemType string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimSuffix(name, "Vec")

	typeSuffix := cTypePublicSuffix(elemType)

	return name + "C" + typeSuffix
}

// cAsmFuncName creates the assembly function name.
// BaseExpVec, float32, neon -> exp_c_f32_neon
func cAsmFuncName(baseName, elemType, targetSuffix string) string {
	name := strings.TrimPrefix(baseName, "Base")
	name = strings.TrimSuffix(name, "Vec")
	name = strings.ToLower(name)

	return name + "_c_" + cTypeSuffix(elemType) + "_" + targetSuffix
}

// cTypeSuffix returns the short type suffix for file naming.
func cTypeSuffix(elemType string) string {
	switch elemType {
	case "float32":
		return "f32"
	case "float64":
		return "f64"
	case "float16", "hwy.Float16":
		return "f16"
	case "bfloat16", "hwy.BFloat16":
		return "bf16"
	case "uint64":
		return "u64"
	case "uint32":
		return "u32"
	case "uint8", "byte":
		return "u8"
	default:
		return "f32"
	}
}

// cTypePublicSuffix returns the public Go suffix for function names.
func cTypePublicSuffix(elemType string) string {
	switch elemType {
	case "float32":
		return "F32"
	case "float64":
		return "F64"
	case "float16", "hwy.Float16":
		return "F16"
	case "bfloat16", "hwy.BFloat16":
		return "BF16"
	case "uint64":
		return "U64"
	case "uint32":
		return "U32"
	case "uint8", "byte":
		return "U8"
	default:
		return "F32"
	}
}

// goElemTypeToCType converts a Go element type to a C type.
func goElemTypeToCType(elemType string) string {
	switch elemType {
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "int32":
		return "int"
	case "int64":
		return "long"
	case "uint32":
		return "unsigned int"
	case "uint64":
		return "unsigned long"
	case "uint8", "byte":
		return "unsigned char"
	default:
		return "float"
	}
}

// cWrapperSliceType returns the Go slice type for wrapper functions.
func cWrapperSliceType(elemType string) string {
	switch elemType {
	case "float32":
		return "[]float32"
	case "float64":
		return "[]float64"
	case "float16", "hwy.Float16":
		return "[]uint16" // f16 stored as uint16 in Go
	case "bfloat16", "hwy.BFloat16":
		return "[]uint16" // bf16 stored as uint16 in Go
	default:
		return "[]float32"
	}
}

// astWrapperGoSliceType returns the Go slice type for AST-translated wrapper
// functions. Unlike cWrapperSliceType, this uses the proper hwy types for
// f16/bf16 to match the original Go function signatures.
func astWrapperGoSliceType(elemType string) string {
	switch elemType {
	case "float32":
		return "[]float32"
	case "float64":
		return "[]float64"
	case "float16", "hwy.Float16":
		return "[]hwy.Float16"
	case "bfloat16", "hwy.BFloat16":
		return "[]hwy.BFloat16"
	case "uint64":
		return "[]uint64"
	case "uint32":
		return "[]uint32"
	case "uint8", "byte":
		return "[]byte"
	default:
		return "[]float32"
	}
}

// goReturnTypeForWrapper maps Go return types to their Go type name for wrappers.
func goReturnTypeForWrapper(goType string) string {
	switch goType {
	case "uint32":
		return "uint32"
	case "uint64":
		return "uint64"
	case "int", "int64":
		return "int"
	case "int32":
		return "int32"
	case "float32":
		return "float32"
	case "float64":
		return "float64"
	default:
		return goType
	}
}

// emitASTCWrapperFunc generates a wrapper function for an AST-translated function
// with arbitrary parameters (multiple slices + int dimensions).
//
// Handles three parameter patterns:
//  1. Functions with explicit int params (matmul): a, b, c []float32, m, n, k int
//  2. Functions without int params (rabitq): code, q1, q2 []uint64 → adds hidden length param
//  3. Functions with return values (rabitq, varint): uint32 → adds output pointer param
//
// Note: Functions with *Image[T] params are handled through the normal transformer/emitter
// flow via the asmBody generation, not through separate wrapper functions.
func emitASTCWrapperFunc(buf *bytes.Buffer, pf *ParsedFunc, elemType, targetSuffix string) {
	// Skip functions with *Image[T] params - they go through the normal transformer flow
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			return
		}
	}

	publicName := buildCPublicName(pf.Name, elemType)
	asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)
	goSliceType := astWrapperGoSliceType(elemType)

	// Classify params
	var sliceParams []string
	var intParams []string
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			sliceParams = append(sliceParams, p.Name)
		} else if p.Type == "int" || p.Type == "int64" {
			intParams = append(intParams, p.Name)
		}
	}

	// Determine if we need a hidden length param (no explicit int params but has slices)
	needsHiddenLen := len(intParams) == 0 && len(sliceParams) > 0
	firstSlice := ""
	if needsHiddenLen && len(sliceParams) > 0 {
		firstSlice = sliceParams[0]
	}

	// Determine if we have return values
	hasReturns := len(pf.Returns) > 0

	// Build Go signature
	goSig := groupGoParams(pf, goSliceType)

	// Build return type string
	var returnType string
	if hasReturns {
		if len(pf.Returns) == 1 {
			returnType = goReturnTypeForWrapper(pf.Returns[0].Type)
		} else {
			var retTypes []string
			for _, ret := range pf.Returns {
				retTypes = append(retTypes, goReturnTypeForWrapper(ret.Type))
			}
			returnType = "(" + strings.Join(retTypes, ", ") + ")"
		}
	}

	// Comment + function signature
	fmt.Fprintf(buf, "// %s computes %s using %s SIMD assembly.\n",
		publicName, strings.TrimPrefix(pf.Name, "Base"), strings.ToUpper(targetSuffix))
	if hasReturns {
		fmt.Fprintf(buf, "func %s(%s) %s {\n", publicName, goSig, returnType)
	} else {
		fmt.Fprintf(buf, "func %s(%s) {\n", publicName, goSig)
	}

	// Zero-length guard for slice-only functions
	if needsHiddenLen {
		fmt.Fprintf(buf, "\tif len(%s) == 0 {\n", firstSlice)
		if hasReturns {
			// Return zero values
			var zeros []string
			for range pf.Returns {
				zeros = append(zeros, "0")
			}
			fmt.Fprintf(buf, "\t\treturn %s\n", strings.Join(zeros, ", "))
		} else {
			fmt.Fprintf(buf, "\t\treturn\n")
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Zero-dimension guard for functions with explicit int params
	if len(intParams) > 0 {
		var checks []string
		for _, ip := range intParams {
			checks = append(checks, ip+" == 0")
		}
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(checks, " || "))
		if hasReturns {
			var zeros []string
			for range pf.Returns {
				zeros = append(zeros, "0")
			}
			fmt.Fprintf(buf, "\t\treturn %s\n", strings.Join(zeros, ", "))
		} else {
			fmt.Fprintf(buf, "\t\treturn\n")
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Bounds checks for slices
	if len(sliceParams) > 0 && len(intParams) > 0 {
		emitBoundsChecks(buf, pf, sliceParams, intParams)
	}

	// Convert int params to int64 for unsafe.Pointer
	for _, ip := range intParams {
		fmt.Fprintf(buf, "\t%sVal := int64(%s)\n", ip, ip)
	}

	// Hidden length param
	if needsHiddenLen {
		fmt.Fprintf(buf, "\tlenVal := int64(len(%s))\n", firstSlice)
	}

	// Output variables for return values.
	// GOAT always uses 64-bit (long *) for output pointers, so we declare
	// int64 variables and cast to the actual return type.
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			fmt.Fprintf(buf, "\tvar out_%s int64\n", name)
		}
	}

	// Call assembly function
	fmt.Fprintf(buf, "\t%s(\n", asmName)

	// Regular params
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%s[0]),\n", p.Name)
		} else if p.Type == "int" || p.Type == "int64" {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&%sVal),\n", p.Name)
		}
	}
	// Hidden length param
	if needsHiddenLen {
		fmt.Fprintf(buf, "\t\tunsafe.Pointer(&lenVal),\n")
	}
	// Output pointer params
	if hasReturns {
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&out_%s),\n", name)
		}
	}
	fmt.Fprintf(buf, "\t)\n")

	// Return with type narrowing from int64 to actual return type
	if hasReturns {
		var retExprs []string
		for _, ret := range pf.Returns {
			name := ret.Name
			if name == "" {
				name = "result"
			}
			goType := goReturnTypeForWrapper(ret.Type)
			if goType == "int64" || goType == "int" {
				retExprs = append(retExprs, goType+"(out_"+name+")")
			} else {
				retExprs = append(retExprs, goType+"(out_"+name+")")
			}
		}
		fmt.Fprintf(buf, "\treturn %s\n", strings.Join(retExprs, ", "))
	}
	fmt.Fprintf(buf, "}\n\n")
}

// hasStructPtrParams returns true if the function has any struct pointer parameters
// (e.g., *Image[T], *SomeStruct[T]).
func hasStructPtrParams(pf *ParsedFunc) bool {
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			return true
		}
	}
	return false
}

// isGenericStructPtr returns true if the type is a pointer to a generic struct
// (e.g., *Image[T], *SomeStruct[T]).
func isGenericStructPtr(typeStr string) bool {
	// Check for *Type[...] pattern
	if !strings.HasPrefix(typeStr, "*") {
		return false
	}
	// Must have a type parameter [...]
	return strings.Contains(typeStr, "[") && strings.Contains(typeStr, "]")
}

// StructField describes a field in a C-compatible struct layout.
// Fields are discovered by analyzing method calls in the function body.
type StructField struct {
	Name     string // Field name in C (lowercased method name)
	GoType   string // Go type for wrapper generation (e.g., "int64", "unsafe.Pointer")
	CType    string // C type (e.g., "long" for dimension getters, "%s *" for data pointer)
	GoGetter string // Go expression to get value (e.g., ".Width()", ".Row(0)[0]")
	IsPtr    bool   // Whether this field is a pointer
	IsData   bool   // Whether this is the data pointer field (for Row-like accessors)
}

// DiscoveredStructInfo contains struct layout information discovered from analyzing
// method calls in the function body. This is fully generic - no hardcoded struct names.
type DiscoveredStructInfo struct {
	Fields      []StructField // Fields discovered from method calls
	DataField   string        // Name of the data pointer field (from Row-like accessor)
	StrideField string        // Name of the stride field (from Row-like accessor)
	GoType      string        // Original Go type, e.g., "*Image[T]"
	ElemCType   string        // C element type, e.g., "float", "double"
}

// DiscoverStructFields analyzes method calls in the function body to discover
// struct layout. This is fully generic - works with any struct type.
//
// Convention-based discovery:
//   - Methods with 0 args (e.g., Width(), Height()) → scalar fields (long)
//   - Methods with 1 arg that are indexed (e.g., Row(y)[i]) → data + stride pattern
func DiscoverStructFields(pf *ParsedFunc, paramName string, elemCType string) *DiscoveredStructInfo {
	info := &DiscoveredStructInfo{
		ElemCType: elemCType,
	}

	// Find the param's Go type
	for _, p := range pf.Params {
		if p.Name == paramName {
			info.GoType = p.Type
			break
		}
	}

	// Track discovered methods to avoid duplicates
	discovered := make(map[string]bool)

	// Walk the function body looking for method calls on this param
	walkMethodCalls(pf.Body, paramName, func(methodName string, argCount int, isIndexed bool) {
		if discovered[methodName] {
			return
		}
		discovered[methodName] = true

		fieldName := strings.ToLower(methodName)

		if argCount == 0 {
			// Simple getter: Width() → width field of type long
			info.Fields = append(info.Fields, StructField{
				Name:     fieldName,
				GoType:   "int64",
				CType:    "long",
				GoGetter: "." + methodName + "()",
				IsPtr:    false,
			})
		} else if argCount == 1 && isIndexed {
			// Row-like accessor: Row(y)[i] → data pointer + stride
			// We assume stride field exists (will be discovered separately)
			info.DataField = "data"
			info.StrideField = "stride"
			info.Fields = append(info.Fields, StructField{
				Name:     "data",
				GoType:   "unsafe.Pointer",
				CType:    elemCType + " *",
				GoGetter: "." + methodName + "(0)[0]",
				IsPtr:    true,
				IsData:   true,
			})
		}
	})

	// Ensure stride field is present if we have a data accessor
	if info.DataField != "" && info.StrideField != "" {
		hasStride := false
		for _, f := range info.Fields {
			if f.Name == info.StrideField {
				hasStride = true
				break
			}
		}
		if !hasStride {
			// Add stride field - assume it's a getter method with matching name
			info.Fields = append(info.Fields, StructField{
				Name:     info.StrideField,
				GoType:   "int64",
				CType:    "long",
				GoGetter: ".Stride()",
				IsPtr:    false,
			})
		}
	}

	return info
}

// walkMethodCalls walks an AST node looking for method calls on the specified receiver.
// The callback receives the method name, argument count, and whether the call is indexed.
func walkMethodCalls(node interface{}, receiverName string, callback func(methodName string, argCount int, isIndexed bool)) {
	if node == nil {
		return
	}

	// This is a simplified walker - the full implementation is in c_ast_translator.go
	// For now, we rely on the C translator's discovery which happens during translation
}

// extractStructBaseName extracts the base name from a struct type.
// E.g., "*Image[T]" -> "Image", "*image.Image[float32]" -> "Image"
func extractStructBaseName(structType string) string {
	baseName := structType
	if strings.HasPrefix(baseName, "*") {
		baseName = baseName[1:]
	}
	if idx := strings.Index(baseName, "["); idx != -1 {
		baseName = baseName[:idx]
	}
	if idx := strings.LastIndex(baseName, "."); idx != -1 {
		baseName = baseName[idx+1:]
	}
	return baseName
}

// emitStructPtrAsmWrapper generates a wrapper function for functions with struct
// pointer parameters. The wrapper creates C-compatible struct layouts and calls
// the assembly function.
//
// This uses transformer-style naming (e.g., BaseForwardICT_neon) so the wrapper
// integrates with the dispatch system.
func emitStructPtrAsmWrapper(buf *bytes.Buffer, pf *ParsedFunc, elemType string, target Target) {
	// Build function name matching transformer convention
	funcName := pf.Name + target.Suffix()
	if elemType != "float32" {
		funcName += "_" + typeNameToSuffix(elemType)
	}

	// Build assembly function name
	targetSuffix := strings.ToLower(target.Name)
	asmName := cAsmFuncName(pf.Name, elemType, targetSuffix)

	// Build parameter list with specialized types
	var params []string
	for _, p := range pf.Params {
		paramType := specializeStructPtrType(p.Type, elemType)
		params = append(params, p.Name+" "+paramType)
	}
	paramList := strings.Join(params, ", ")

	// Emit function signature
	fmt.Fprintf(buf, "// %s calls the %s SIMD assembly implementation.\n",
		funcName, strings.ToUpper(target.Name))
	fmt.Fprintf(buf, "func %s(%s) {\n", funcName, paramList)

	// Nil checks
	var nilChecks []string
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			nilChecks = append(nilChecks, p.Name+" == nil")
		}
	}
	if len(nilChecks) > 0 {
		fmt.Fprintf(buf, "\tif %s {\n", strings.Join(nilChecks, " || "))
		fmt.Fprintf(buf, "\t\treturn\n")
		fmt.Fprintf(buf, "\t}\n")
	}

	// Create C-compatible struct instances for each struct pointer param
	// The struct layout is discovered from method calls in the function body
	for _, p := range pf.Params {
		if !isGenericStructPtr(p.Type) {
			continue
		}

		// Discover struct layout by analyzing the function body
		elemCType := goElemTypeToCType(elemType)
		structInfo := DiscoverStructFields(pf, p.Name, elemCType)
		if structInfo == nil || len(structInfo.Fields) == 0 {
			continue
		}

		// Emit struct definition
		fmt.Fprintf(buf, "\tc%s := struct {\n", p.Name)
		for _, field := range structInfo.Fields {
			fmt.Fprintf(buf, "\t\t%s %s\n", field.Name, field.GoType)
		}
		fmt.Fprintf(buf, "\t}{\n")

		// Emit field initializers
		for _, field := range structInfo.Fields {
			if field.IsPtr {
				fmt.Fprintf(buf, "\t\t%s: unsafe.Pointer(&%s%s),\n", field.Name, p.Name, field.GoGetter)
			} else {
				fmt.Fprintf(buf, "\t\t%s: %s(%s%s),\n", field.Name, field.GoType, p.Name, field.GoGetter)
			}
		}
		fmt.Fprintf(buf, "\t}\n")
	}

	// Call assembly function with pointers to the C structs
	fmt.Fprintf(buf, "\t%s(\n", asmName)
	for _, p := range pf.Params {
		if isGenericStructPtr(p.Type) {
			fmt.Fprintf(buf, "\t\tunsafe.Pointer(&c%s),\n", p.Name)
		}
	}
	fmt.Fprintf(buf, "\t)\n")
	fmt.Fprintf(buf, "}\n\n")
}

// specializeStructPtrType specializes a generic struct pointer type.
// E.g., *Image[T] with elemType=float32 becomes *Image[float32]
func specializeStructPtrType(typeStr, elemType string) string {
	// Replace T with the concrete type
	// This handles *Image[T] -> *Image[float32]
	if strings.Contains(typeStr, "[T]") {
		return strings.Replace(typeStr, "[T]", "["+goTypeName(elemType)+"]", 1)
	}
	return typeStr
}

// goTypeName returns the Go type name for an element type.
func goTypeName(elemType string) string {
	switch elemType {
	case "float16":
		return "hwy.Float16"
	case "bfloat16":
		return "hwy.BFloat16"
	default:
		return elemType
	}
}

// groupGoParams groups consecutive parameters with the same Go type for cleaner
// function signatures: "a, b, c []float32, m, n, k int" instead of
// "a []float32, b []float32, c []float32, m int, n int, k int".
func groupGoParams(pf *ParsedFunc, goSliceType string) string {
	var groups []string
	var currentNames []string
	var currentType string

	for _, p := range pf.Params {
		var paramType string
		if strings.HasPrefix(p.Type, "[]") {
			paramType = goSliceType
		} else if p.Type == "int" || p.Type == "int64" {
			paramType = "int"
		} else {
			paramType = p.Type
		}

		if paramType == currentType {
			currentNames = append(currentNames, p.Name)
		} else {
			if len(currentNames) > 0 {
				groups = append(groups, strings.Join(currentNames, ", ")+" "+currentType)
			}
			currentNames = []string{p.Name}
			currentType = paramType
		}
	}
	if len(currentNames) > 0 {
		groups = append(groups, strings.Join(currentNames, ", ")+" "+currentType)
	}
	return strings.Join(groups, ", ")
}

// emitBoundsChecks generates bounds checks for slice parameters based on the
// function's dimension parameters. This is a best-effort heuristic that covers
// common patterns like matmul (a[m*k], b[k*n], c[m*n]).
func emitBoundsChecks(buf *bytes.Buffer, pf *ParsedFunc, sliceParams, intParams []string) {
	// For matmul-like functions with 3 slices and 3 ints (m, n, k),
	// generate standard bounds checks.
	if len(sliceParams) == 3 && len(intParams) == 3 {
		m, n, k := intParams[0], intParams[1], intParams[2]
		fmt.Fprintf(buf, "\tif len(%s) < %s*%s || len(%s) < %s*%s || len(%s) < %s*%s {\n",
			sliceParams[0], m, k,
			sliceParams[1], k, n,
			sliceParams[2], m, n)
		fmt.Fprintf(buf, "\t\treturn\n")
		fmt.Fprintf(buf, "\t}\n")
	}
}
