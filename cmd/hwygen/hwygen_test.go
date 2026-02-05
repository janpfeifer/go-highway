package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGetTarget(t *testing.T) {
	tests := []struct {
		name    string
		target  string
		wantErr bool
	}{
		{"AVX2", "avx2", false},
		{"AVX512", "avx512", false},
		{"Fallback", "fallback", false},
		{"Unknown", "unknown", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := GetTarget(tt.target)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetTarget(%q) error = %v, wantErr %v", tt.target, err, tt.wantErr)
			}
		})
	}
}

func TestTargetLanesFor(t *testing.T) {
	avx2 := AVX2Target()

	tests := []struct {
		elemType string
		want     int
	}{
		{"float32", 8}, // 32 bytes / 4 = 8
		{"float64", 4}, // 32 bytes / 8 = 4
		{"int32", 8},   // 32 bytes / 4 = 8
		{"int64", 4},   // 32 bytes / 8 = 4
	}

	for _, tt := range tests {
		t.Run(tt.elemType, func(t *testing.T) {
			got := avx2.LanesFor(tt.elemType)
			if got != tt.want {
				t.Errorf("AVX2Target.LanesFor(%q) = %d, want %d", tt.elemType, got, tt.want)
			}
		})
	}
}

func TestGetConcreteTypes(t *testing.T) {
	tests := []struct {
		constraint string
		wantLen    int
		wantFirst  string
	}{
		{"hwy.Floats", 4, "hwy.Float16"},   // Float16, BFloat16, float32, float64
		{"hwy.FloatsNative", 2, "float32"}, // float32, float64 only (no half-precision)
		{"hwy.SignedInts", 2, "int32"},
		{"hwy.Integers", 4, "int32"},
		{"unknown", 2, "float32"}, // Default
	}

	for _, tt := range tests {
		t.Run(tt.constraint, func(t *testing.T) {
			got := GetConcreteTypes(tt.constraint)
			if len(got) != tt.wantLen {
				t.Errorf("GetConcreteTypes(%q) returned %d types, want %d", tt.constraint, len(got), tt.wantLen)
			}
			if len(got) > 0 && got[0] != tt.wantFirst {
				t.Errorf("GetConcreteTypes(%q) first type = %q, want %q", tt.constraint, got[0], tt.wantFirst)
			}
		})
	}
}

func TestParseSimpleFunction(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.go")

	content := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](a, b, result []T) {
	size := min(len(a), len(b), len(result))
	for i := 0; i < size; i += 8 {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		vr := hwy.Add(va, vb)
		hwy.Store(vr, result[i:])
	}
}
`

	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Parse the file
	result, err := Parse(testFile)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Verify results
	if result.PackageName != "test" {
		t.Errorf("Package name = %q, want %q", result.PackageName, "test")
	}

	if len(result.Funcs) != 1 {
		t.Fatalf("Got %d functions, want 1", len(result.Funcs))
	}

	f := result.Funcs[0]
	if f.Name != "BaseAdd" {
		t.Errorf("Function name = %q, want %q", f.Name, "BaseAdd")
	}

	if len(f.TypeParams) != 1 {
		t.Errorf("Got %d type params, want 1", len(f.TypeParams))
	}

	if len(f.HwyCalls) < 3 {
		t.Errorf("Got %d hwy calls, want at least 3 (Load, Load, Add, Store)", len(f.HwyCalls))
	}

	// Check for specific operations
	var hasLoad, hasAdd, hasStore bool
	for _, call := range f.HwyCalls {
		switch call.FuncName {
		case "Load":
			hasLoad = true
		case "Add":
			hasAdd = true
		case "Store":
			hasStore = true
		}
	}

	if !hasLoad {
		t.Error("Expected to find Load operation")
	}
	if !hasAdd {
		t.Error("Expected to find Add operation")
	}
	if !hasStore {
		t.Error("Expected to find Store operation")
	}
}

func TestGeneratorEndToEnd(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "add.go")
	content := `package testadd

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](a, b, result []T) {
	size := min(len(a), len(b), len(result))
	vOne := hwy.Set(T(1))
	for i := 0; i < size; i += vOne.NumElements() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		vr := hwy.Add(va, vb)
		hwy.Store(vr, result[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator
	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"avx2", "fallback"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check that expected files were created
	// Note: dispatch files are now architecture-specific and prefixed with function name
	expectedFiles := []string{
		"dispatch_add_amd64.gen.go",
		"dispatch_add_other.gen.go",
		"add_avx2.gen.go",
		"add_fallback.gen.go",
	}

	for _, filename := range expectedFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %q was not created", filename)
		} else {
			// Read and check basic content
			content, err := os.ReadFile(path)
			if err != nil {
				t.Errorf("Failed to read %q: %v", filename, err)
				continue
			}

			contentStr := string(content)

			// Check for package declaration
			if !strings.Contains(contentStr, "package testadd") {
				t.Errorf("File %q missing package declaration", filename)
			}

			// Check for "Code generated" comment
			if !strings.Contains(contentStr, HeaderNote) {
				t.Errorf("File %q missing generation comment", filename)
			}
		}
	}

	// Check dispatcher file specifically
	dispatchPath := filepath.Join(tmpDir, "dispatch_add_amd64.gen.go")
	dispatchContent, err := os.ReadFile(dispatchPath)
	if err != nil {
		t.Fatalf("Failed to read dispatcher: %v", err)
	}

	dispatchStr := string(dispatchContent)

	// Should have function variables (now with type suffix)
	if !strings.Contains(dispatchStr, "var AddFloat32 func") {
		t.Error("Dispatcher missing AddFloat32 function variable")
	}

	// Should have init function
	if !strings.Contains(dispatchStr, "func init()") {
		t.Error("Dispatcher missing init function")
	}

	// Should have target-specific init functions
	if !strings.Contains(dispatchStr, "func initAddAVX2()") {
		t.Error("Dispatcher missing initAddAVX2 function")
	}
	if !strings.Contains(dispatchStr, "func initAddFallback()") {
		t.Error("Dispatcher missing initAddFallback function")
	}
}

func TestSpecializeType(t *testing.T) {
	typeParams := []TypeParam{
		{Name: "T", Constraint: "hwy.Floats"},
	}

	tests := []struct {
		input    string
		elemType string
		want     string
	}{
		{"[]T", "float32", "[]float32"},
		{"T", "float64", "float64"},
		{"[][]T", "int32", "[][]int32"},
		{"map[string]T", "float32", "map[string]float32"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := specializeType(tt.input, typeParams, tt.elemType)
			if got != tt.want {
				t.Errorf("specializeType(%q, _, %q) = %q, want %q", tt.input, tt.elemType, got, tt.want)
			}
		})
	}
}

func TestBuildDispatchFuncName(t *testing.T) {
	tests := []struct {
		baseName  string
		elemType  string
		isGeneric bool
		private   bool
		want      string
	}{
		// Generic functions get type suffixes
		{"BaseSigmoid", "float32", true, false, "SigmoidFloat32"},
		{"BaseSigmoid", "float64", true, false, "SigmoidFloat64"},
		{"BaseAdd", "float32", true, false, "AddFloat32"},
		{"BaseAdd", "int32", true, false, "AddInt32"},
		// Non-generic functions don't get type suffixes
		{"BaseDecodeStreamVByte32Into", "uint8", false, false, "DecodeStreamVByte32Into"},
		{"BasePack32", "uint32", false, false, "Pack32"},
		// Private (lowercase "base" prefix) functions produce unexported names
		{"baseSigmoid", "float32", true, true, "sigmoidFloat32"},
		{"baseSigmoid", "float64", true, true, "sigmoidFloat64"},
		{"baseAdd", "float32", true, true, "addFloat32"},
		{"basePack32", "uint32", false, true, "pack32"},
	}

	for _, tt := range tests {
		t.Run(tt.baseName+"_"+tt.elemType, func(t *testing.T) {
			got := buildDispatchFuncName(tt.baseName, tt.elemType, tt.isGeneric, tt.private)
			if got != tt.want {
				t.Errorf("buildDispatchFuncName(%q, %q, %v, %v) = %q, want %q", tt.baseName, tt.elemType, tt.isGeneric, tt.private, got, tt.want)
			}
		})
	}
}

func TestContribSubpackageImports(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a test input file that uses contrib/math functions
	inputFile := filepath.Join(tmpDir, "sigmoid.go")
	content := `package testsigmoid

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib"
)

func BaseSigmoid[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	vOne := hwy.Set(T(1))
	for i := 0; i < size; i += vOne.NumElements() {
		x := hwy.Load(input[i:])
		y := contrib.Sigmoid(x)
		hwy.Store(y, output[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator
	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"avx2", "fallback"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check AVX2 output file has contrib/math import
	avx2Path := filepath.Join(tmpDir, "sigmoid_avx2.gen.go")
	avx2Content, err := os.ReadFile(avx2Path)
	if err != nil {
		t.Fatalf("Failed to read AVX2 output: %v", err)
	}

	avx2Str := string(avx2Content)

	// Should have contrib/math import for SIMD target
	if !strings.Contains(avx2Str, `"github.com/ajroetker/go-highway/hwy/contrib/math"`) {
		t.Error("AVX2 output missing contrib/math import")
	}

	// Should use math.BaseSigmoidVec_avx2 style function calls
	if !strings.Contains(avx2Str, "math.BaseSigmoidVec_avx2") {
		t.Error("AVX2 output missing math.BaseSigmoidVec_avx2 function call")
	}

	// Check fallback output uses hwy package
	fallbackPath := filepath.Join(tmpDir, "sigmoid_fallback.gen.go")
	fallbackContent, err := os.ReadFile(fallbackPath)
	if err != nil {
		t.Fatalf("Failed to read fallback output: %v", err)
	}

	fallbackStr := string(fallbackContent)

	// Fallback should use hwy package for core ops
	if !strings.Contains(fallbackStr, `"github.com/ajroetker/go-highway/hwy"`) {
		t.Error("Fallback output missing hwy import")
	}

	// Fallback should use math.BaseSigmoidVec for contrib functions (not hwy.Sigmoid)
	if !strings.Contains(fallbackStr, "math.BaseSigmoidVec_fallback") {
		t.Error("Fallback output missing math.BaseSigmoidVec_fallback function call")
	}
}

func TestDetectContribPackages(t *testing.T) {
	targets := []Target{AVX2Target(), FallbackTarget()}

	tests := []struct {
		name     string
		calls    []HwyCall
		wantMath bool
		wantVec  bool
		wantAlgo bool
	}{
		{
			name:     "No contrib",
			calls:    []HwyCall{{Package: "hwy", FuncName: "Add"}},
			wantMath: false, wantVec: false, wantAlgo: false,
		},
		{
			name:     "Math function",
			calls:    []HwyCall{{Package: "contrib", FuncName: "Exp"}},
			wantMath: true, wantVec: false, wantAlgo: false,
		},
		{
			name:     "Dot function",
			calls:    []HwyCall{{Package: "contrib", FuncName: "Dot"}},
			wantMath: false, wantVec: true, wantAlgo: false,
		},
		{
			name:     "Multiple functions",
			calls:    []HwyCall{{Package: "contrib", FuncName: "Sigmoid"}, {Package: "contrib", FuncName: "Dot"}},
			wantMath: true,
			wantVec:  true,
			wantAlgo: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			funcs := []ParsedFunc{{HwyCalls: tt.calls}}
			pkgs := detectContribPackages(funcs, targets)

			if pkgs.Math != tt.wantMath {
				t.Errorf("Math = %v, want %v", pkgs.Math, tt.wantMath)
			}
			if pkgs.Vec != tt.wantVec {
				t.Errorf("Dot = %v, want %v", pkgs.Vec, tt.wantVec)
			}
			if pkgs.Algo != tt.wantAlgo {
				t.Errorf("Algo = %v, want %v", pkgs.Algo, tt.wantAlgo)
			}
		})
	}
}

func TestParseCondition(t *testing.T) {
	tests := []struct {
		name      string
		condition string
		target    string
		elemType  string
		want      bool
	}{
		// Simple type conditions
		{"f32 matches float32", "f32", "AVX2", "float32", true},
		{"f32 doesn't match float64", "f32", "AVX2", "float64", false},
		{"f64 matches float64", "f64", "AVX2", "float64", true},
		{"f64 doesn't match float32", "f64", "AVX2", "float32", false},

		// Simple target conditions
		{"avx2 matches AVX2", "avx2", "AVX2", "float32", true},
		{"avx2 doesn't match AVX512", "avx2", "AVX512", "float32", false},
		{"avx512 matches AVX512", "avx512", "AVX512", "float32", true},
		{"neon matches NEON", "neon", "NEON", "float32", true},
		{"fallback matches Fallback", "fallback", "Fallback", "float32", true},

		// Compound conditions with AND
		{"f64 && avx2 matches", "f64 && avx2", "AVX2", "float64", true},
		{"f64 && avx2 doesn't match f32", "f64 && avx2", "AVX2", "float32", false},
		{"f64 && avx2 doesn't match avx512", "f64 && avx2", "AVX512", "float64", false},
		{"f32 && neon matches", "f32 && neon", "NEON", "float32", true},

		// Compound conditions with OR
		{"f32 || f64 matches f32", "f32 || f64", "AVX2", "float32", true},
		{"f32 || f64 matches f64", "f32 || f64", "AVX2", "float64", true},
		{"avx2 || avx512 matches avx2", "avx2 || avx512", "AVX2", "float32", true},
		{"avx2 || avx512 matches avx512", "avx2 || avx512", "AVX512", "float32", true},
		{"avx2 || avx512 doesn't match neon", "avx2 || avx512", "NEON", "float32", false},

		// Negation
		{"!avx2 matches avx512", "!avx2", "AVX512", "float32", true},
		{"!avx2 doesn't match avx2", "!avx2", "AVX2", "float32", false},
		{"!f64 matches f32", "!f64", "AVX2", "float32", true},
		{"!f64 doesn't match f64", "!f64", "AVX2", "float64", false},

		// Complex compound with mixed AND/OR
		{"(f64 && avx2) matches", "f64 && avx2", "AVX2", "float64", true},
		{"f64 && !avx512 matches avx2", "f64 && !avx512", "AVX2", "float64", true},
		{"f64 && !avx512 doesn't match avx512", "f64 && !avx512", "AVX512", "float64", false},

		// Category conditions
		{"float matches float32", "float", "AVX2", "float32", true},
		{"float matches float64", "float", "AVX2", "float64", true},
		{"float matches Float16", "float", "NEON", "hwy.Float16", true},
		{"float matches BFloat16", "float", "NEON", "hwy.BFloat16", true},
		{"float doesn't match int32", "float", "AVX2", "int32", false},
		{"float doesn't match uint64", "float", "AVX2", "uint64", false},
		{"int matches int32", "int", "AVX2", "int32", true},
		{"int matches int64", "int", "AVX2", "int64", true},
		{"int doesn't match uint32", "int", "AVX2", "uint32", false},
		{"int doesn't match float32", "int", "AVX2", "float32", false},
		{"uint matches uint32", "uint", "AVX2", "uint32", true},
		{"uint matches uint64", "uint", "AVX2", "uint64", true},
		{"uint doesn't match int32", "uint", "AVX2", "int32", false},
		{"uint doesn't match float64", "uint", "AVX2", "float64", false},

		// Category with negation
		{"!float matches int32", "!float", "AVX2", "int32", true},
		{"!float doesn't match float32", "!float", "AVX2", "float32", false},

		// Category compound conditions
		{"float && avx2 matches", "float && avx2", "AVX2", "float32", true},
		{"float && avx2 doesn't match int32", "float && avx2", "AVX2", "int32", false},
		{"int || uint matches int32", "int || uint", "AVX2", "int32", true},
		{"int || uint matches uint64", "int || uint", "AVX2", "uint64", true},
		{"int || uint doesn't match float32", "int || uint", "AVX2", "float32", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pc := parseCondition(tt.condition)
			got := pc.Evaluate(tt.target, tt.elemType)
			if got != tt.want {
				t.Errorf("parseCondition(%q).Evaluate(%q, %q) = %v, want %v",
					tt.condition, tt.target, tt.elemType, got, tt.want)
			}
		})
	}
}

func TestIsScalarTailLoop(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		iterator string
		end      string
		want     bool
	}{
		{
			name:     "valid scalar tail loop",
			code:     "for ; i < n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     true,
		},
		{
			name:     "scalar tail loop with different iterator",
			code:     "for ; j < n; j++ { dst[j] += s[j] }",
			iterator: "j",
			end:      "n",
			want:     true,
		},
		{
			name:     "scalar tail loop with different end var",
			code:     "for ; i < size; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "size",
			want:     true,
		},
		{
			name:     "not a scalar tail - has init",
			code:     "for i := 0; i < n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - wrong iterator in condition",
			code:     "for ; j < n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - wrong end variable",
			code:     "for ; i < m; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - decrement instead of increment",
			code:     "for ; i < n; i-- { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - >= condition",
			code:     "for ; i >= n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "scalar tail loop with len(dst) end - simple copy",
			code:     "for ; i < len(dst); i++ { dst[i] = src[i] }",
			iterator: "i",
			end:      "len(dst)",
			want:     true,
		},
		{
			name:     "scalar tail loop with len(src) end - simple copy",
			code:     "for ; i < len(src); i++ { dst[i] = src[i] }",
			iterator: "i",
			end:      "len(src)",
			want:     true,
		},
		{
			name:     "not a scalar tail - uses external variable scale (multiply)",
			code:     "for ; i < len(dst); i++ { dst[i] *= scale }",
			iterator: "i",
			end:      "len(dst)",
			want:     false,
		},
		{
			name:     "not a scalar tail - uses external variable scale (expression)",
			code:     "for ; i < len(src); i++ { dst[i] = src[i] * scale }",
			iterator: "i",
			end:      "len(src)",
			want:     false,
		},
		{
			name:     "not a scalar tail - assigns to local variable",
			code:     "for ; i < len(src); i++ { dst[i] = src[i] - prev; prev = src[i] }",
			iterator: "i",
			end:      "len(src)",
			want:     false,
		},
		{
			name:     "not a scalar tail - assigns to local variable only",
			code:     "for ; i < n; i++ { sum += src[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "scalar tail with type conversion via selector",
			code:     "for ; i < len(dst); i++ { dst[i] = hwy.Float32ToFloat16(src[i].Float32()) }",
			iterator: "i",
			end:      "len(dst)",
			want:     true,
		},
		{
			name:     "scalar tail with method call on indexed element",
			code:     "for ; i < len(dst); i++ { dst[i] = pkg.Convert(dst[i].Value() + s[i].Value()) }",
			iterator: "i",
			end:      "len(dst)",
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Parse the for loop
			src := "package test\nfunc f() { " + tt.code + " }"
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "", src, 0)
			if err != nil {
				t.Fatalf("Failed to parse test code: %v", err)
			}

			// Find the for statement
			var forStmt *ast.ForStmt
			ast.Inspect(file, func(n ast.Node) bool {
				if f, ok := n.(*ast.ForStmt); ok {
					forStmt = f
					return false
				}
				return true
			})

			if forStmt == nil {
				t.Fatal("No for statement found in parsed code")
			}

			got := isScalarTailLoop(forStmt, tt.iterator, tt.end)
			if got != tt.want {
				t.Errorf("isScalarTailLoop() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConditionalBlockFiltering(t *testing.T) {
	// Create a temporary test file with hwy:if directives
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.go")

	content := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseTest[T hwy.Floats](a, result []T) {
	size := len(a)
	for i := 0; i < size; i++ {
		x := a[i]
		//hwy:if f64 && avx2
		result[i] = x * 2 // scalar fallback for f64 on avx2
		//hwy:else
		result[i] = x * 3 // normal simd path
		//hwy:endif
	}
}
`

	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Parse the file
	result, err := Parse(testFile)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Verify conditional blocks were parsed
	if len(result.ConditionalBlocks) != 1 {
		t.Fatalf("Got %d conditional blocks, want 1", len(result.ConditionalBlocks))
	}

	block := result.ConditionalBlocks[0]
	if block.Condition != "f64 && avx2" {
		t.Errorf("Block condition = %q, want %q", block.Condition, "f64 && avx2")
	}

	// Verify the parsed condition structure
	if block.ParsedCondition == nil {
		t.Fatal("ParsedCondition is nil")
	}
	if block.ParsedCondition.Op != "&&" {
		t.Errorf("ParsedCondition.Op = %q, want %q", block.ParsedCondition.Op, "&&")
	}

	// Test evaluation
	if !block.ParsedCondition.Evaluate("AVX2", "float64") {
		t.Error("Condition should match f64 && avx2")
	}
	if block.ParsedCondition.Evaluate("AVX2", "float32") {
		t.Error("Condition should not match f32 on avx2")
	}
	if block.ParsedCondition.Evaluate("AVX512", "float64") {
		t.Error("Condition should not match f64 on avx512")
	}
}

func TestOutputPrefix(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "prefix_test.go")
	content := `package testprefix

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](a, b, result []T) {
	size := min(len(a), len(b), len(result))
	vOne := hwy.Set(T(1))
	for i := 0; i < size; i += vOne.NumElements() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		vr := hwy.Add(va, vb)
		hwy.Store(vr, result[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator with OutputPrefix
	gen := &Generator{
		InputFile:    inputFile,
		OutputDir:    tmpDir,
		OutputPrefix: "custom_prefix",
		Targets:      []string{"avx2", "fallback"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check that files with custom prefix were created
	expectedFiles := []string{
		"custom_prefix_avx2.gen.go",
		"custom_prefix_fallback.gen.go",
	}

	for _, filename := range expectedFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %q was not created", filename)
		} else {
			content, err := os.ReadFile(path)
			if err != nil {
				t.Errorf("Failed to read %q: %v", filename, err)
				continue
			}
			if !strings.Contains(string(content), HeaderNote) {
				t.Errorf("File %q missing generation comment", filename)
			}
		}
	}

	// Ensure default files are NOT created
	defaultFiles := []string{
		"prefix_test_avx2.gen.go",
		"prefix_test_fallback.gen.go",
	}
	for _, filename := range defaultFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); err == nil {
			t.Errorf("Unexpected default file %q was created", filename)
		}
	}
}

func TestDispatchPrefix(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "dispatch_prefix.go")
	content := `package testdispatch

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](a, b, result []T) {
	size := min(len(a), len(b), len(result))
	vOne := hwy.Set(T(1))
	for i := 0; i < size; i += vOne.NumElements() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		vr := hwy.Add(va, vb)
		hwy.Store(vr, result[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator with DispatchPrefix
	gen := &Generator{
		InputFile:      inputFile,
		OutputDir:      tmpDir,
		DispatchPrefix: "custom_dispatch",
		Targets:        []string{"avx2", "fallback"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check that dispatch files with custom prefix were created WITHOUT "dispatch_" prepended
	expectedFiles := []string{
		"custom_dispatch_amd64.gen.go",
		"custom_dispatch_other.gen.go",
	}

	for _, filename := range expectedFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %q was not created", filename)
		}
	}

	// Check that default dispatch file was NOT created
	defaultDispatch := "dispatch_custom_dispatch_amd64.gen.go"
	if _, err := os.Stat(filepath.Join(tmpDir, defaultDispatch)); err == nil {
		t.Errorf("Unexpected default dispatch file %q was created", defaultDispatch)
	}
}

func TestLoadStoreTransformation(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a test input file using Load and Store
	// Note: Function must start with "Base" to be processed by hwygen
	inputFile := filepath.Join(tmpDir, "load.go")
	content := `package testload

import "github.com/ajroetker/go-highway/hwy"

func BaseCopy[T hwy.FloatsNative](src, dst []T) {
	n := hwy.NumLanes[T]()
	for i := 0; i <= len(src)-n; i += n {
		v := hwy.Load(src[i:])
		hwy.Store(v, dst[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator for AVX2
	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"avx2"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Read the generated AVX2 file
	avx2Path := filepath.Join(tmpDir, "load_avx2.gen.go")
	avx2Content, err := os.ReadFile(avx2Path)
	if err != nil {
		t.Fatalf("Failed to read AVX2 output: %v", err)
	}

	contentStr := string(avx2Content)

	// Verify Load was transformed to use unsafe.Pointer
	if !strings.Contains(contentStr, "unsafe.Pointer") {
		t.Error("Generated code missing unsafe.Pointer for Load/Store")
	}

	// Verify the pointer-to-array cast pattern exists
	if !strings.Contains(contentStr, "(*[8]float32)") && !strings.Contains(contentStr, "(*[4]float64)") {
		t.Error("Generated code missing pointer-to-array cast pattern")
	}

	// Verify unsafe import was added
	if !strings.Contains(contentStr, `"unsafe"`) {
		t.Error("Generated code missing unsafe import")
	}

	// Verify archsimd.LoadFloat32x8 is called (not LoadFloat32x8Slice)
	if !strings.Contains(contentStr, "archsimd.LoadFloat32x8(") {
		t.Error("Generated code should use archsimd.LoadFloat32x8 (pointer-based), not LoadFloat32x8Slice")
	}

	// Verify v.Store is called with the cast (not StoreSlice)
	if strings.Contains(contentStr, "StoreSlice") {
		t.Error("Generated code should use v.Store (pointer-based), not StoreSlice")
	}

	// Verify the slice-to-pointer optimization is applied:
	// Should generate &src[i] instead of &src[i:][0]
	// The pattern [i:][0] should NOT appear in the output
	if strings.Contains(contentStr, ":][0]") {
		t.Error("Generated code has unoptimized slice access &slice[i:][0], should be &slice[i]")
	}

	// Verify the optimized pattern &src[i]) or &dst[i]) appears
	if !strings.Contains(contentStr, "&src[i])") && !strings.Contains(contentStr, "&dst[i])") {
		t.Error("Generated code missing optimized pointer pattern &slice[i]")
	}
}

// TestNumLanesTypeParameter verifies that hwy.NumLanes[T]() uses the explicit type parameter T
// for lane count calculation, not the function's first slice parameter type.
// This is a regression test for a bug where functions like:
//
//	func Decode(dst []float32, src []byte) { lanes := hwy.NumLanes[uint8]() ... }
//
// would incorrectly get lanes=8 (float32 AVX2 lanes) instead of lanes=32 (uint8 AVX2 lanes).
func TestNumLanesTypeParameter(t *testing.T) {
	tmpDir := t.TempDir()

	// Create test input that has a first parameter type different from the Load type
	// This triggers the bug: function signature has float32, but operations are on uint8
	inputFile := filepath.Join(tmpDir, "numlanes_type.go")
	content := `package testnumlanes

import (
	"unsafe"
	"github.com/ajroetker/go-highway/hwy"
)

// BaseDecodeFloat32s has float32 as first slice parameter, but uses uint8 operations.
// The lanes variable should be computed from uint8 (32 lanes for AVX2), not float32 (8 lanes).
func BaseDecodeFloat32s(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}
	totalBytes := len(dst) * 4
	if len(src) < totalBytes {
		return
	}
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), totalBytes)

	// This is the key: hwy.NumLanes[uint8]() should give 32 for AVX2, not 8
	lanes := hwy.NumLanes[uint8]()
	i := 0
	for ; i+lanes <= totalBytes; i += lanes {
		v := hwy.Load[uint8](src[i:])
		hwy.Store(v, dstBytes[i:])
	}
	for ; i < totalBytes; i++ {
		dstBytes[i] = src[i]
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Test AVX2 target (should get lanes=32 for uint8, not lanes=8 for float32)
	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"avx2"},
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	avx2Path := filepath.Join(tmpDir, "numlanes_type_avx2.gen.go")
	avx2Content, err := os.ReadFile(avx2Path)
	if err != nil {
		t.Fatalf("Failed to read AVX2 output: %v", err)
	}

	contentStr := string(avx2Content)

	// The critical check: lanes should be 32 (uint8 AVX2 = 32 bytes = 256 bits)
	// NOT 8 (float32 AVX2 = 8 floats = 256 bits)
	if strings.Contains(contentStr, "lanes := 8") {
		t.Error("Bug: hwy.NumLanes[uint8]() incorrectly generated lanes=8 (float32 lanes) instead of lanes=32 (uint8 lanes)")
	}
	if !strings.Contains(contentStr, "lanes := 32") {
		t.Error("Expected lanes := 32 for uint8 AVX2, but not found in generated code")
	}

	// Also verify the SIMD operations use uint8 vectors
	// hwy.Load generates pointer-based LoadUint8x32 (not LoadUint8x32Slice)
	if !strings.Contains(contentStr, "LoadUint8x32") {
		t.Error("Expected LoadUint8x32 for uint8 AVX2 operations")
	}

	// Test NEON target (should get lanes=16 for uint8, not lanes=4 for float32)
	gen2 := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"neon"},
	}

	if err := gen2.Run(); err != nil {
		t.Fatalf("Generator.Run() for NEON failed: %v", err)
	}

	neonPath := filepath.Join(tmpDir, "numlanes_type_neon.gen.go")
	neonContent, err := os.ReadFile(neonPath)
	if err != nil {
		t.Fatalf("Failed to read NEON output: %v", err)
	}

	neonStr := string(neonContent)

	// For NEON: lanes should be 16 (uint8 NEON = 16 bytes = 128 bits)
	// NOT 4 (float32 NEON = 4 floats = 128 bits)
	if strings.Contains(neonStr, "lanes := 4") {
		t.Error("Bug: hwy.NumLanes[uint8]() incorrectly generated lanes=4 (float32 lanes) instead of lanes=16 (uint8 lanes)")
	}
	if !strings.Contains(neonStr, "lanes := 16") {
		t.Error("Expected lanes := 16 for uint8 NEON, but not found in generated code")
	}
}

func TestNEONHalfPrecisionStoreSlice(t *testing.T) {
	// Test that half-precision StoreSlice on NEON properly casts to []uint16
	tmpDir := t.TempDir()

	inputFile := filepath.Join(tmpDir, "halfprec_store.go")
	content := `package testhalf

import "github.com/ajroetker/go-highway/hwy"

func BaseScale[T hwy.Floats](data []T, scale T) {
	lanes := hwy.MaxLanes[T]()
	scaleVec := hwy.Set(scale)
	for i := 0; i < len(data); i += lanes {
		v := hwy.LoadSlice(data[i:])
		result := hwy.Mul(v, scaleVec)
		hwy.StoreSlice(result, data[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"neon"},
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	neonPath := filepath.Join(tmpDir, "halfprec_store_neon.gen.go")
	neonContent, err := os.ReadFile(neonPath)
	if err != nil {
		t.Fatalf("Failed to read NEON output: %v", err)
	}

	neonStr := string(neonContent)

	// For Float16 function, StoreSlice should cast slice to []uint16
	// Look for the unsafe.Slice conversion pattern
	if !strings.Contains(neonStr, "unsafe.Slice((*uint16)") {
		t.Error("NEON Float16 StoreSlice should cast slice to []uint16 using unsafe.Slice")
	}

	// Extract just the Float16 function to check for proper conversion
	float16Start := strings.Index(neonStr, "func BaseScale_neon_Float16")
	float16End := strings.Index(neonStr, "func BaseScale_neon_BFloat16")
	if float16Start == -1 || float16End == -1 {
		t.Fatal("Could not find Float16 function in generated code")
	}
	float16Func := neonStr[float16Start:float16End]

	// In the Float16 function, all StoreSlice calls should use unsafe.Slice
	// Count StoreSlice calls and ensure they all have unsafe.Slice
	storeSliceCalls := strings.Count(float16Func, ".StoreSlice(")
	unsafeSliceCalls := strings.Count(float16Func, ".StoreSlice(unsafe.Slice(")

	if storeSliceCalls != unsafeSliceCalls {
		t.Errorf("Float16 function has %d StoreSlice calls but only %d use unsafe.Slice conversion",
			storeSliceCalls, unsafeSliceCalls)
	}

	if storeSliceCalls == 0 {
		t.Error("Float16 function should have StoreSlice calls")
	}
}

// TestCModeExpGeneration verifies that -c mode generates correct C code for the
// exp function across element types. This validates:
// - f32: native NEON path with correct intrinsics and hex constants
// - f16: promoted math path (load f16, promote to f32, compute, demote back)
// - bf16: promoted math path with bfloat16 load/store intrinsics
//
// These were previously hand-written in hwy/c/ files; hwygen now generates them.
func TestCModeExpGeneration(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a minimal Vec→Vec function named BaseExpVec.
	// The C emitter dispatches on function name, not body content.
	inputFile := filepath.Join(tmpDir, "exp_base.go")
	content := `package testexp

import "github.com/ajroetker/go-highway/hwy"

func BaseExpVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	return hwy.Mul(x, x)
}
`
	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"neon"},
		CMode:     true,
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() in CMode failed: %v", err)
	}

	// ---- f32: native NEON path ----
	f32Path := filepath.Join(tmpDir, "baseexpvec_c_f32_neon_arm64.c")
	f32Content, err := os.ReadFile(f32Path)
	if err != nil {
		t.Fatalf("Failed to read f32 C file: %v", err)
	}
	f32 := string(f32Content)

	// Verify function signature uses float pointers
	if !strings.Contains(f32, "void exp_c_f32_neon(float *input, float *result, long *len)") {
		t.Error("f32: expected float pointer signature")
	}

	// Verify NEON f32 intrinsics
	for _, intrinsic := range []string{
		"vld1q_f32",  // vector load
		"vst1q_f32",  // vector store
		"vfmaq_f32",  // fused multiply-add (Horner polynomial)
		"vrndnq_f32", // round to nearest (range reduction)
		"vfmsq_f32",  // fused multiply-subtract
	} {
		if !strings.Contains(f32, intrinsic) {
			t.Errorf("f32: missing NEON intrinsic %q", intrinsic)
		}
	}

	// Verify hex constant for invLn2 (1/ln(2) = 1.4426950408889634)
	if !strings.Contains(f32, "0x3FB8AA3B") {
		t.Error("f32: missing hex constant 0x3FB8AA3B (invLn2)")
	}

	// Verify 3-tier loop structure: main (4 vectors), single, scalar
	if !strings.Contains(f32, "i + 15 < n; i += 16") {
		t.Error("f32: missing main loop (16 elements = 4 vectors of 4)")
	}
	if !strings.Contains(f32, "i + 3 < n; i += 4") {
		t.Error("f32: missing single vector loop (4 elements)")
	}
	if !strings.Contains(f32, "for (; i < n; i++)") {
		t.Error("f32: missing scalar tail loop")
	}

	// ---- f16: promoted math path (load f16, compute in f32, store f16) ----
	f16Path := filepath.Join(tmpDir, "baseexpvec_c_f16_neon_arm64.c")
	f16Content, err := os.ReadFile(f16Path)
	if err != nil {
		t.Fatalf("Failed to read f16 C file: %v", err)
	}
	f16 := string(f16Content)

	// Verify function signature uses unsigned short (f16 storage type)
	if !strings.Contains(f16, "void exp_c_f16_neon(unsigned short *input, unsigned short *result, long *len)") {
		t.Error("f16: expected unsigned short pointer signature")
	}

	// Verify promoted math comment
	if !strings.Contains(f16, "promoted Exp via f32 polynomial") {
		t.Error("f16: missing promoted math comment")
	}

	// Verify f16 load intrinsic
	if !strings.Contains(f16, "vld1q_f16") {
		t.Error("f16: missing vld1q_f16 load intrinsic")
	}

	// Verify f16→f32 promotion intrinsics
	if !strings.Contains(f16, "vcvt_f32_f16") {
		t.Error("f16: missing vcvt_f32_f16 promote intrinsic")
	}

	// Verify computation happens in f32 (not f16)
	if !strings.Contains(f16, "float32x4_t") {
		t.Error("f16: computation should use float32x4_t vectors")
	}
	if !strings.Contains(f16, "vfmaq_f32") {
		t.Error("f16: Horner polynomial should use f32 FMA")
	}

	// Verify f32→f16 demotion and store
	if !strings.Contains(f16, "vcvt_f16_f32") {
		t.Error("f16: missing vcvt_f16_f32 demote intrinsic")
	}
	if !strings.Contains(f16, "vst1q_f16") {
		t.Error("f16: missing vst1q_f16 store intrinsic")
	}

	// Verify constants are f32 (not f64)
	if !strings.Contains(f16, "float32x4_t invLn2") {
		t.Error("f16: promoted constants should be float32x4_t")
	}
	if strings.Contains(f16, "float64x2_t") {
		t.Error("f16: should NOT contain float64x2_t (was using wrong legacy path)")
	}

	// ---- bf16: promoted math with bfloat16 intrinsics ----
	bf16Path := filepath.Join(tmpDir, "baseexpvec_c_bf16_neon_arm64.c")
	bf16Content, err := os.ReadFile(bf16Path)
	if err != nil {
		t.Fatalf("Failed to read bf16 C file: %v", err)
	}
	bf16 := string(bf16Content)

	// Verify bf16 function signature
	if !strings.Contains(bf16, "void exp_c_bf16_neon(unsigned short *input, unsigned short *result, long *len)") {
		t.Error("bf16: expected unsigned short pointer signature")
	}

	// Verify bf16 load intrinsic
	if !strings.Contains(bf16, "vld1q_bf16") {
		t.Error("bf16: missing vld1q_bf16 load intrinsic")
	}

	// Verify bf16→f32 promotion (bit-shift pattern for bfloat16)
	if !strings.Contains(bf16, "vshll_n_u16") {
		t.Error("bf16: missing vshll_n_u16 for bf16→f32 promotion")
	}

	// Verify computation in f32
	if !strings.Contains(bf16, "float32x4_t") {
		t.Error("bf16: computation should use float32x4_t vectors")
	}

	// Verify f32→bf16 demotion
	if !strings.Contains(bf16, "vst1q_bf16") {
		t.Error("bf16: missing vst1q_bf16 store intrinsic")
	}
}
