//go:build (amd64 && goexperiment.simd) || arm64

package math

import (
	stdmath "math"
	"testing"
)

// Minimum slice sizes for SIMD operations (defined in inverse_trig_test.go)
// const minF32 = 16
// const minF64 = 8

func TestLog2Poly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"log2(1) = 0", 1.0, 0.0},
		{"log2(2) = 1", 2.0, 1.0},
		{"log2(4) = 2", 4.0, 2.0},
		{"log2(8) = 3", 8.0, 3.0},
		{"log2(0.5) = -1", 0.5, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			Log2Poly(input, output)
			got := output[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Log2(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestLog10Poly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"log10(1) = 0", 1.0, 0.0},
		{"log10(10) = 1", 10.0, 1.0},
		{"log10(100) = 2", 100.0, 2.0},
		{"log10(0.1) = -1", 0.1, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			Log10Poly(input, output)
			got := output[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Log10(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestExp2Poly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"2^0 = 1", 0.0, 1.0},
		{"2^1 = 2", 1.0, 2.0},
		{"2^2 = 4", 2.0, 4.0},
		{"2^3 = 8", 3.0, 8.0},
		{"2^-1 = 0.5", -1.0, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			Exp2Poly(input, output)
			got := output[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("Exp2(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestSinhPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"sinh(0)", 0.0},
		{"sinh(1)", 1.0},
		{"sinh(-1)", -1.0},
		{"sinh(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			SinhPoly(input, output)
			got := output[0]
			want := float32(stdmath.Sinh(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Sinh(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

func TestCoshPoly(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"cosh(0) = 1", 0.0},
		{"cosh(1)", 1.0},
		{"cosh(-1)", -1.0},
		{"cosh(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.input)
			output := make([]float32, minF32)
			CoshPoly(input, output)
			got := output[0]
			want := float32(stdmath.Cosh(float64(tt.input)))

			if stdmath.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("Cosh(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// Float64 tests

func TestLog2Poly_Float64(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  float64
	}{
		{"log2(1) = 0", 1.0, 0.0},
		{"log2(2) = 1", 2.0, 1.0},
		{"log2(4) = 2", 4.0, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF64(minF64, tt.input)
			output := make([]float64, minF64)
			Log2Poly(input, output)
			got := output[0]

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Log2(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestSinhPoly_Float64(t *testing.T) {
	tests := []struct {
		name  string
		input float64
	}{
		{"sinh(0)", 0.0},
		{"sinh(1)", 1.0},
		{"sinh(2)", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF64(minF64, tt.input)
			output := make([]float64, minF64)
			SinhPoly(input, output)
			got := output[0]
			want := stdmath.Sinh(tt.input)

			if stdmath.Abs(got-want) > 1e-6 {
				t.Errorf("Sinh(%v) = %v, want %v", tt.input, got, want)
			}
		})
	}
}

// Benchmarks

func BenchmarkLog2Poly(b *testing.B) {
	input := fillF32(minF32, 2.5)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Log2Poly(input, output)
	}
}

func BenchmarkSinhPoly(b *testing.B) {
	input := fillF32(minF32, 1.5)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SinhPoly(input, output)
	}
}
