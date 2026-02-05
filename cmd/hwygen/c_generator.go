package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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

	for _, pf := range result.Funcs {
		if IsCEligible(&pf) {
			vecFuncs = append(vecFuncs, pf)
		} else if IsSliceFunction(&pf) {
			sliceFuncs = append(sliceFuncs, pf)
		}
	}

	totalFuncs := len(vecFuncs) + len(sliceFuncs)
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

		if len(cFiles) == 0 {
			continue
		}

		// Compile all C files with GOAT
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
		allFuncs := append(vecFuncs, sliceFuncs...)
		if err := g.emitCWrappers(allFuncs, target); err != nil {
			return fmt.Errorf("emit wrappers for %s: %w", target.Name, err)
		}
	}

	return nil
}

// getCElemTypes returns the concrete element types for C code generation.
// This includes f16/bf16 types when the constraint allows them.
func getCElemTypes(pf *ParsedFunc) []string {
	if len(pf.TypeParams) > 0 {
		return GetConcreteTypes(pf.TypeParams[0].Constraint)
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
	for _, et := range []string{"float16", "hwy.Float16", "bfloat16", "hwy.BFloat16", "float32", "float64"} {
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

	// File header
	fmt.Fprintf(&buf, "//go:build %s\n", buildTag)
	fmt.Fprintf(&buf, "// Code generated by hwygen -c. DO NOT EDIT.\n\n")
	fmt.Fprintf(&buf, "package %s\n\n", pkgName)
	fmt.Fprintf(&buf, "import \"unsafe\"\n\n")

	// Emit wrapper functions
	fmt.Fprintf(&buf, "// Public wrapper functions\n")
	for _, pf := range funcs {
		elemTypes := getCElemTypes(&pf)
		for _, elemType := range elemTypes {
			profile := GetCProfile(target.Name, elemType)
			if profile == nil {
				continue
			}
			emitCWrapperFunc(&buf, &pf, elemType, targetSuffix)
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
	default:
		return "F32"
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
