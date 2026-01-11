//go:build (amd64 && goexperiment.simd) || arm64

package math

import (
	stdmath "math"
	"testing"
)

// Minimum slice sizes for SIMD operations
const (
	minFloat32Lanes = 16
	minFloat64Lanes = 8
)

func TestPowPoly(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
		{"e^1 = e", float32(stdmath.E), 1.0, float32(stdmath.E)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputX := make([]float32, minFloat32Lanes)
			inputY := make([]float32, minFloat32Lanes)
			output := make([]float32, minFloat32Lanes)
			for i := range inputX {
				inputX[i] = tt.x
				inputY[i] = tt.y
			}
			PowPoly(inputX, inputY, output)
			got := output[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPowPoly_SpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float32
		y       float32
		wantInf bool
		wantNaN bool
		want    float32
	}{
		{"0^1 = 0", 0.0, 1.0, false, false, 0.0},
		{"0^0 = 1", 0.0, 0.0, false, false, 1.0},
		{"0^-1 = +Inf", 0.0, -1.0, true, false, 0.0},
		{"1^0 = 1", 1.0, 0.0, false, false, 1.0},
		{"1^NaN = 1", 1.0, float32(stdmath.NaN()), false, false, 1.0},
		{"NaN^0 = 1", float32(stdmath.NaN()), 0.0, false, false, 1.0},
		{"+Inf^1 = +Inf", float32(stdmath.Inf(1)), 1.0, true, false, 0.0},
		{"+Inf^-1 = 0", float32(stdmath.Inf(1)), -1.0, false, false, 0.0},
		{"+Inf^0 = 1", float32(stdmath.Inf(1)), 0.0, false, false, 1.0},
		{"-1^2 = NaN", -1.0, 2.0, false, true, 0.0}, // negative base -> NaN in our impl
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputX := make([]float32, minFloat32Lanes)
			inputY := make([]float32, minFloat32Lanes)
			output := make([]float32, minFloat32Lanes)
			for i := range inputX {
				inputX[i] = tt.x
				inputY[i] = tt.y
			}
			PowPoly(inputX, inputY, output)
			got := output[0]

			if tt.wantInf {
				if !stdmath.IsInf(float64(got), 1) {
					t.Errorf("Pow(%v, %v) = %v, want +Inf", tt.x, tt.y, got)
				}
			} else if tt.wantNaN {
				if !stdmath.IsNaN(float64(got)) {
					t.Errorf("Pow(%v, %v) = %v, want NaN", tt.x, tt.y, got)
				}
			} else {
				if stdmath.Abs(float64(got-tt.want)) > 1e-5 {
					t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
				}
			}
		})
	}
}

func TestPowPoly_Float64(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		y    float64
		want float64
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputX := make([]float64, minFloat64Lanes)
			inputY := make([]float64, minFloat64Lanes)
			output := make([]float64, minFloat64Lanes)
			for i := range inputX {
				inputX[i] = tt.x
				inputY[i] = tt.y
			}
			PowPoly(inputX, inputY, output)
			got := output[0]

			relErr := stdmath.Abs(got-tt.want) / (stdmath.Abs(tt.want) + 1e-10)
			if relErr > 1e-6 {
				t.Errorf("Pow(%v, %v) = %v, want %v (relErr=%v)", tt.x, tt.y, got, tt.want, relErr)
			}
		})
	}
}

// Benchmarks

func BenchmarkPowPoly(b *testing.B) {
	inputX := make([]float32, minFloat32Lanes)
	inputY := make([]float32, minFloat32Lanes)
	output := make([]float32, minFloat32Lanes)
	for i := range inputX {
		inputX[i] = 2.5
		inputY[i] = 3.7
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PowPoly(inputX, inputY, output)
	}
}

func BenchmarkPowPoly_Float64(b *testing.B) {
	inputX := make([]float64, minFloat64Lanes)
	inputY := make([]float64, minFloat64Lanes)
	output := make([]float64, minFloat64Lanes)
	for i := range inputX {
		inputX[i] = 2.5
		inputY[i] = 3.7
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PowPoly(inputX, inputY, output)
	}
}

func BenchmarkPow_Scalar(b *testing.B) {
	x := 2.5
	y := 3.7
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = stdmath.Pow(x, y)
	}
}
