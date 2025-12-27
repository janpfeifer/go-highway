//go:build amd64 && goexperiment.simd

package algo

import (
	"math"
	"simd/archsimd"
	"testing"

	hwymath "github.com/ajroetker/go-highway/hwy/contrib/math"
)

const benchSize = 1024

// Test correctness using input ranges that match the existing accuracy tests.
// The SIMD implementations have been validated for these ranges.

func TestExpTransform(t *testing.T) {
	// Use values from TestExp32_Accuracy that are known to work
	input := []float32{-10, -1, 0, 0.5, 1, 2, 5, 10, -5, -2, 0.1, 0.9, 3, 4, 6, 7}
	output := make([]float32, len(input))

	ExpTransform(input, output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ExpTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestLogTransform(t *testing.T) {
	// Use positive values from reasonable range
	input := []float32{0.1, 0.5, 1.0, 2.0, 2.718, 5.0, 10.0, 100.0, 0.01, 0.25, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0}
	output := make([]float32, len(input))

	LogTransform(input, output)

	for i := range input {
		expected := float32(math.Log(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("LogTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestSinTransform(t *testing.T) {
	// Use values in [-2π, 2π] range
	input := []float32{0, 0.5, 1.0, 1.57, 2.0, 3.14, 4.0, 5.0, -0.5, -1.0, -1.57, -2.0, -3.14, -4.0, 6.0, 6.28}
	output := make([]float32, len(input))

	SinTransform(input, output)

	for i := range input {
		expected := float32(math.Sin(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("SinTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestCosTransform(t *testing.T) {
	// Use values in [-2π, 2π] range
	input := []float32{0, 0.5, 1.0, 1.57, 2.0, 3.14, 4.0, 5.0, -0.5, -1.0, -1.57, -2.0, -3.14, -4.0, 6.0, 6.28}
	output := make([]float32, len(input))

	CosTransform(input, output)

	for i := range input {
		expected := float32(math.Cos(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("CosTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestTanhTransform(t *testing.T) {
	// Use values in [-5, 5] range (tanh saturates beyond this)
	input := []float32{-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, -3, -0.1, 0.1, 3, -4, 4, 1.5}
	output := make([]float32, len(input))

	TanhTransform(input, output)

	for i := range input {
		expected := float32(math.Tanh(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("TanhTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestSigmoidTransform(t *testing.T) {
	// Use values in [-10, 10] range
	input := []float32{-10, -5, -2, -1, 0, 1, 2, 5, 10, -3, -0.5, 0.5, 3, -4, 4, 6}
	output := make([]float32, len(input))

	SigmoidTransform(input, output)

	for i := range input {
		expected := float32(1.0 / (1.0 + math.Exp(-float64(input[i]))))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("SigmoidTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

func TestErfTransform(t *testing.T) {
	// Use values in [-3, 3] range (erf saturates beyond this)
	input := []float32{-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, -1.5, -0.25, 0.25, 1.5, -2.5, 2.5, 0.75}
	output := make([]float32, len(input))

	ErfTransform(input, output)

	for i := range input {
		expected := float32(math.Erf(float64(input[i])))
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ErfTransform[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

// Debug test to check if broadcast constants work
func TestBroadcastConstants(t *testing.T) {
	// Create a broadcast constant and verify all lanes have the same value
	bcast := archsimd.BroadcastFloat32x8(3.14159)
	output := make([]float32, 8)
	bcast.StoreSlice(output)

	t.Logf("Broadcast 3.14159: %v", output)

	for i, v := range output {
		if v != 3.14159 {
			t.Errorf("BroadcastFloat32x8[%d]: got %v, want 3.14159", i, v)
		}
	}
}

// Test basic SIMD arithmetic
func TestSimdArithmetic(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	x := archsimd.LoadFloat32x8Slice(input)

	// Test multiplication by broadcast constant
	two := archsimd.BroadcastFloat32x8(2.0)
	result := x.Mul(two)

	output := make([]float32, 8)
	result.StoreSlice(output)

	t.Logf("Input * 2: %v", output)

	for i, v := range output {
		expected := input[i] * 2
		if v != expected {
			t.Errorf("Mul[%d]: got %v, want %v", i, v, expected)
		}
	}
}

// Test MulAdd operation which is heavily used in polynomial evaluation
func TestMulAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2}
	c := []float32{10, 10, 10, 10, 10, 10, 10, 10}

	aVec := archsimd.LoadFloat32x8Slice(a)
	bVec := archsimd.LoadFloat32x8Slice(b)
	cVec := archsimd.LoadFloat32x8Slice(c)

	// MulAdd(a, b, c) = a*b + c
	result := aVec.MulAdd(bVec, cVec)
	output := make([]float32, 8)
	result.StoreSlice(output)

	t.Logf("a*b + c: %v", output)

	for i := range a {
		expected := a[i]*b[i] + c[i]
		if output[i] != expected {
			t.Errorf("MulAdd[%d]: got %v, want %v", i, output[i], expected)
		}
	}
}

// Test RoundToEven which is used in range reduction
func TestRoundToEven(t *testing.T) {
	input := []float32{0.4, 0.5, 0.6, 1.4, 1.5, 2.5, -0.5, -1.5}
	x := archsimd.LoadFloat32x8Slice(input)

	result := x.RoundToEven()
	output := make([]float32, 8)
	result.StoreSlice(output)

	t.Logf("RoundToEven: input=%v output=%v", input, output)

	// RoundToEven rounds to nearest even: 0.5->0, 1.5->2, 2.5->2
	expected := []float32{0, 0, 1, 1, 2, 2, 0, -2}
	for i := range expected {
		if output[i] != expected[i] {
			t.Errorf("RoundToEven[%d]: got %v, want %v", i, output[i], expected[i])
		}
	}
}

// Test ConvertToInt32 and back
func TestConvertToInt32(t *testing.T) {
	input := []float32{0, 1, 2, 3, -1, -2, 100, -100}
	x := archsimd.LoadFloat32x8Slice(input)

	intVec := x.ConvertToInt32()

	intOutput := make([]int32, 8)
	intVec.StoreSlice(intOutput)
	t.Logf("ConvertToInt32: input=%v output=%v", input, intOutput)

	for i := range input {
		expected := int32(input[i])
		if intOutput[i] != expected {
			t.Errorf("ConvertToInt32[%d]: got %v, want %v", i, intOutput[i], expected)
		}
	}
}


// Test the 2^k scaling computation which is the heart of exp
func TestTwoToTheK(t *testing.T) {
	// For exp(x), we compute 2^k where k = round(x / ln(2))
	// For x=0: k=0, 2^0=1
	// For x=1: k=round(1/0.693)=round(1.44)=1, 2^1=2
	kValues := []int32{0, 1, 2, -1, -2, 3, 4, 5}

	kVec := archsimd.LoadInt32x8Slice(kValues)
	bias := archsimd.BroadcastInt32x8(127)

	// 2^k = ((k + 127) << 23) reinterpreted as float
	expBits := kVec.Add(bias).ShiftAllLeft(23)
	scale := expBits.AsFloat32x8()

	output := make([]float32, 8)
	scale.StoreSlice(output)

	t.Logf("2^k: k=%v scale=%v", kValues, output)

	// Expected: 2^0=1, 2^1=2, 2^2=4, 2^-1=0.5, 2^-2=0.25, etc
	for i := range kValues {
		expected := float32(math.Pow(2, float64(kValues[i])))
		if !closeEnough32(output[i], expected, 1e-6) {
			t.Errorf("TwoToTheK[%d]: k=%d got %v, want %v", i, kValues[i], output[i], expected)
		}
	}
}

// Debug test to isolate where the SIMD issue is
func TestLoadStoreRoundtrip(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	output := make([]float32, 8)

	// Test that LoadFloat32x8Slice and StoreSlice work correctly
	x := archsimd.LoadFloat32x8Slice(input)
	x.StoreSlice(output)

	for i := range input {
		if output[i] != input[i] {
			t.Errorf("LoadStoreRoundtrip[%d]: got %v, want %v", i, output[i], input[i])
		}
	}
}

func TestExpAVX2Direct(t *testing.T) {
	// Test calling Exp_AVX2_F32x8 directly with known values
	input := []float32{0, 1, 2, -1, 0.5, -0.5, 0.1, -0.1}
	output := make([]float32, 8)

	x := archsimd.LoadFloat32x8Slice(input)
	result := hwymath.Exp_AVX2_F32x8(x)
	result.StoreSlice(output)

	t.Logf("Input:  %v", input)
	t.Logf("Output: %v", output)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		t.Logf("[%d] input=%v output=%v expected=%v", i, input[i], output[i], expected)
		if !closeEnough32(output[i], expected, 1e-4) {
			t.Errorf("ExpAVX2Direct[%d] input=%v: got %v, want %v", i, input[i], output[i], expected)
		}
	}
}

// Test that mimics exactly what the SIMD exp loop does
func TestExpMimicLoop(t *testing.T) {
	input := []float32{0, 1, 2, -1, 0.5, -0.5, 0.1, -0.1}

	data := input
	n := len(input)
	result := make([]float32, n)

	t.Logf("n=%d", n)

	for i := 0; i+8 <= n; i += 8 {
		t.Logf("Loop iteration i=%d", i)
		x := archsimd.LoadFloat32x8Slice(data[i:])

		// Log the loaded values
		loaded := make([]float32, 8)
		x.StoreSlice(loaded)
		t.Logf("Loaded: %v", loaded)

		out := hwymath.Exp_AVX2_F32x8(x)

		// Log the output values
		outVals := make([]float32, 8)
		out.StoreSlice(outVals)
		t.Logf("Exp output: %v", outVals)

		out.StoreSlice(result[i:])
	}

	t.Logf("Result: %v", result)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		if !closeEnough32(result[i], expected, 1e-4) {
			t.Errorf("ExpMimicLoop[%d] input=%v: got %v, want %v", i, input[i], result[i], expected)
		}
	}
}



func closeEnough32(a, b, tol float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	if math.IsInf(float64(a), 0) && math.IsInf(float64(b), 0) {
		return (a > 0) == (b > 0)
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tol
}

// Benchmarks - Transform API (zero allocation)

func BenchmarkExpTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ExpTransform(input, output)
	}
}

func BenchmarkLogTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		LogTransform(input, output)
	}
}

func BenchmarkSinTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SinTransform(input, output)
	}
}

func BenchmarkCosTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		CosTransform(input, output)
	}
}

func BenchmarkTanhTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		TanhTransform(input, output)
	}
}

func BenchmarkSigmoidTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SigmoidTransform(input, output)
	}
}

func BenchmarkErfTransform(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ErfTransform(input, output)
	}
}

// Benchmarks - Stdlib comparison

func BenchmarkExpTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Exp(float64(input[j])))
		}
	}
}

func BenchmarkLogTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Log(float64(input[j])))
		}
	}
}

func BenchmarkSinTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Sin(float64(input[j])))
		}
	}
}

func BenchmarkCosTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Cos(float64(input[j])))
		}
	}
}

func BenchmarkTanhTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Tanh(float64(input[j])))
		}
	}
}

func BenchmarkSigmoidTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(1.0 / (1.0 + math.Exp(-float64(input[j]))))
		}
	}
}

func BenchmarkErfTransform_Stdlib(b *testing.B) {
	input := make([]float32, benchSize)
	output := make([]float32, benchSize)
	for i := range input {
		input[i] = float32(i-benchSize/2) * 0.01
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range input {
			output[j] = float32(math.Erf(float64(input[j])))
		}
	}
}
