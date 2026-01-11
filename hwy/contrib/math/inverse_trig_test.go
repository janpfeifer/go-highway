//go:build (amd64 && goexperiment.simd) || arm64

package math

import (
	stdmath "math"
	"testing"
)

// Tests for: Tan, Atan, Atan2, Asin, Acos, Expm1, Log1p

// Minimum slice sizes for SIMD operations
const (
	minF32 = 16
	minF64 = 8
)

// Helper to fill slice with value
func fillF32(n int, v float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = v
	}
	return s
}

func fillF64(n int, v float64) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = v
	}
	return s
}

// ============================================================================
// Tan Tests
// ============================================================================

func TestTanPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"tan(0)", 0.0},
		{"tan(pi/4)", float32(stdmath.Pi / 4)},
		{"tan(-pi/4)", float32(-stdmath.Pi / 4)},
		{"tan(pi/6)", float32(stdmath.Pi / 6)},
		{"tan(0.5)", 0.5},
		{"tan(1.0)", 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			TanPoly(input, output)
			got := output[0]
			want := float32(stdmath.Tan(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Tan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// ============================================================================
// Atan Tests
// ============================================================================

func TestAtanPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"atan(0)", 0.0},
		{"atan(1)", 1.0},
		{"atan(-1)", -1.0},
		{"atan(0.5)", 0.5},
		{"atan(2)", 2.0},
		{"atan(-2)", -2.0},
		{"atan(10)", 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			AtanPoly(input, output)
			got := output[0]
			want := float32(stdmath.Atan(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Atan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestAtan2Poly(t *testing.T) {
	tests := []struct {
		name string
		y, x float32
	}{
		{"atan2(1, 1)", 1.0, 1.0},
		{"atan2(1, -1)", 1.0, -1.0},
		{"atan2(-1, 1)", -1.0, 1.0},
		{"atan2(-1, -1)", -1.0, -1.0},
		{"atan2(0, 1)", 0.0, 1.0},
		{"atan2(1, 0)", 1.0, 0.0},
		{"atan2(3, 4)", 3.0, 4.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputY := fillF32(minF32, tt.y)
			inputX := fillF32(minF32, tt.x)
			output := make([]float32, minF32)
			Atan2Poly(inputY, inputX, output)
			got := output[0]
			want := float32(stdmath.Atan2(float64(tt.y), float64(tt.x)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Atan2(%v, %v) = %v, want %v", tt.y, tt.x, got, want)
			}
		})
	}
}

// ============================================================================
// Asin Tests
// ============================================================================

func TestAsinPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"asin(0)", 0.0},
		{"asin(0.5)", 0.5},
		{"asin(-0.5)", -0.5},
		{"asin(0.3)", 0.3},
		{"asin(0.7)", 0.7},
		{"asin(0.9)", 0.9},
		{"asin(1)", 1.0},
		{"asin(-1)", -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			AsinPoly(input, output)
			got := output[0]
			want := float32(stdmath.Asin(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Asin(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestAsin_OutOfRange(t *testing.T) {
	// |x| > 1 should return NaN
	input := fillF32(minF32, 1.5)
	output := make([]float32, minF32)
	AsinPoly(input, output)
	got := output[0]

	if !stdmath.IsNaN(float64(got)) {
		t.Errorf("Asin(1.5) = %v, want NaN", got)
	}
}

// ============================================================================
// Acos Tests
// ============================================================================

func TestAcosPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"acos(0)", 0.0},
		{"acos(0.5)", 0.5},
		{"acos(-0.5)", -0.5},
		{"acos(1)", 1.0},
		{"acos(-1)", -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			AcosPoly(input, output)
			got := output[0]
			want := float32(stdmath.Acos(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Acos(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// ============================================================================
// Expm1 Tests
// ============================================================================

func TestExpm1Poly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"expm1(0)", 0.0},
		{"expm1(0.001)", 0.001},   // Small value - uses Taylor series
		{"expm1(-0.001)", -0.001}, // Small negative
		{"expm1(0.1)", 0.1},
		{"expm1(1)", 1.0},
		{"expm1(-1)", -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			Expm1Poly(input, output)
			got := output[0]
			want := float32(stdmath.Expm1(float64(tt.input)))

			relErr := stdmath.Abs(float64(got-want)) / (stdmath.Abs(float64(want)) + 1e-10)
			if relErr > 1e-5 {
				t.Errorf("Expm1(%v) = %v, want %v (relErr=%v)", tt.input, got, want, relErr)
			}
		})
	}
}

// ============================================================================
// Log1p Tests
// ============================================================================

func TestLog1pPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"log1p(0)", 0.0},
		{"log1p(0.001)", 0.001},   // Small value - uses Taylor series
		{"log1p(-0.001)", -0.001}, // Small negative
		{"log1p(0.1)", 0.1},
		{"log1p(1)", 1.0},
		{"log1p(10)", 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			Log1pPoly(input, output)
			got := output[0]
			want := float32(stdmath.Log1p(float64(tt.input)))

			relErr := stdmath.Abs(float64(got-want)) / (stdmath.Abs(float64(want)) + 1e-10)
			if relErr > 1e-5 {
				t.Errorf("Log1p(%v) = %v, want %v (relErr=%v)", tt.input, got, want, relErr)
			}
		})
	}
}

func TestLog1p_SpecialCases(t *testing.T) {
	// log1p(-1) = -Inf
	input := fillF32(minF32, -1.0)
	output := make([]float32, minF32)
	Log1pPoly(input, output)
	got := output[0]

	if !stdmath.IsInf(float64(got), -1) {
		t.Errorf("Log1p(-1) = %v, want -Inf", got)
	}

	// log1p(x) = NaN for x < -1
	input = fillF32(minF32, -2.0)
	Log1pPoly(input, output)
	got = output[0]

	if !stdmath.IsNaN(float64(got)) {
		t.Errorf("Log1p(-2) = %v, want NaN", got)
	}
}

// ============================================================================
// Float64 Tests
// ============================================================================

func TestAtanPoly_Float64(t *testing.T) {
	tests := []struct {
		name  string
		input float64
	}{
		{"atan(0)", 0.0},
		{"atan(1)", 1.0},
		{"atan(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF64(minF64, tt.input)
			output := make([]float64, minF64)
			AtanPoly(input, output)
			got := output[0]
			want := stdmath.Atan(tt.input)

			if stdmath.Abs(got-want) > 1e-6 {
				t.Errorf("Atan(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestAsinPoly_Float64(t *testing.T) {
	tests := []struct {
		name  string
		input float64
	}{
		{"asin(0)", 0.0},
		{"asin(0.5)", 0.5},
		{"asin(0.9)", 0.9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF64(minF64, tt.input)
			output := make([]float64, minF64)
			AsinPoly(input, output)
			got := output[0]
			want := stdmath.Asin(tt.input)

			if stdmath.Abs(got-want) > 1e-6 {
				t.Errorf("Asin(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestExpm1Poly_Float64(t *testing.T) {
	input := fillF64(minF64, 0.001)
	output := make([]float64, minF64)
	Expm1Poly(input, output)
	got := output[0]
	want := stdmath.Expm1(0.001)

	if stdmath.Abs(got-want) > 1e-6 {
		t.Errorf("Expm1(0.001) = %v, want %v", got, want)
	}
}

func TestLog1pPoly_Float64(t *testing.T) {
	input := fillF64(minF64, 0.001)
	output := make([]float64, minF64)
	Log1pPoly(input, output)
	got := output[0]
	want := stdmath.Log1p(0.001)

	if stdmath.Abs(got-want) > 1e-6 {
		t.Errorf("Log1p(0.001) = %v, want %v", got, want)
	}
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkTanPoly(b *testing.B) {
	input := fillF32(minF32, 0.5)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TanPoly(input, output)
	}
}

func BenchmarkAtanPoly(b *testing.B) {
	input := fillF32(minF32, 0.5)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AtanPoly(input, output)
	}
}

func BenchmarkAsinPoly(b *testing.B) {
	input := fillF32(minF32, 0.5)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AsinPoly(input, output)
	}
}

func BenchmarkExpm1Poly(b *testing.B) {
	input := fillF32(minF32, 0.001)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Expm1Poly(input, output)
	}
}

func BenchmarkLog1pPoly(b *testing.B) {
	input := fillF32(minF32, 0.001)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Log1pPoly(input, output)
	}
}
