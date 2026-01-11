//go:build (amd64 && goexperiment.simd) || arm64

package math

import (
	stdmath "math"
	"testing"
)

// Minimum slice sizes for SIMD operations (defined in inverse_trig_test.go)
// const minF32 = 16
// const minF64 = 8

func TestCbrtPoly(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		want float32
	}{
		{"cbrt(8) = 2", 8.0, 2.0},
		{"cbrt(27) = 3", 27.0, 3.0},
		{"cbrt(1) = 1", 1.0, 1.0},
		{"cbrt(0) = 0", 0.0, 0.0},
		{"cbrt(-8) = -2", -8.0, -2.0},
		{"cbrt(-27) = -3", -27.0, -3.0},
		{"cbrt(64) = 4", 64.0, 4.0},
		{"cbrt(125) = 5", 125.0, 5.0},
		{"cbrt(0.001) = 0.1", 0.001, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF32(minF32, tt.x)
			output := make([]float32, minF32)
			CbrtPoly(input, output)
			got := output[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Cbrt(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestCbrtPoly_Float64(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{"cbrt(8) = 2", 8.0, 2.0},
		{"cbrt(27) = 3", 27.0, 3.0},
		{"cbrt(1) = 1", 1.0, 1.0},
		{"cbrt(0) = 0", 0.0, 0.0},
		{"cbrt(-8) = -2", -8.0, -2.0},
		{"cbrt(-27) = -3", -27.0, -3.0},
		{"cbrt(64) = 4", 64.0, 4.0},
		{"cbrt(125) = 5", 125.0, 5.0},
		{"cbrt(0.001) = 0.1", 0.001, 0.1},
		{"cbrt(1000) = 10", 1000.0, 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF64(minF64, tt.x)
			output := make([]float64, minF64)
			CbrtPoly(input, output)
			got := output[0]

			if stdmath.Abs(got-tt.want) > 1e-6 {
				t.Errorf("Cbrt(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestCbrtPoly_SpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float64
		wantInf int
		wantNaN bool
		want    float64
	}{
		{"+Inf", stdmath.Inf(1), 1, false, 0.0},
		{"-Inf", stdmath.Inf(-1), -1, false, 0.0},
		{"NaN", stdmath.NaN(), 0, true, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := fillF64(minF64, tt.x)
			output := make([]float64, minF64)
			CbrtPoly(input, output)
			got := output[0]

			if tt.wantInf != 0 {
				if !stdmath.IsInf(got, tt.wantInf) {
					t.Errorf("Cbrt(%v) = %v, want Inf(%d)", tt.x, got, tt.wantInf)
				}
			} else if tt.wantNaN {
				if !stdmath.IsNaN(got) {
					t.Errorf("Cbrt(%v) = %v, want NaN", tt.x, got)
				}
			}
		})
	}
}

func TestHypotPoly(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"hypot(3, 4) = 5", 3.0, 4.0, 5.0},
		{"hypot(5, 12) = 13", 5.0, 12.0, 13.0},
		{"hypot(8, 15) = 17", 8.0, 15.0, 17.0},
		{"hypot(0, 0) = 0", 0.0, 0.0, 0.0},
		{"hypot(1, 0) = 1", 1.0, 0.0, 1.0},
		{"hypot(0, 1) = 1", 0.0, 1.0, 1.0},
		{"hypot(-3, 4) = 5", -3.0, 4.0, 5.0},
		{"hypot(3, -4) = 5", 3.0, -4.0, 5.0},
		{"hypot(-3, -4) = 5", -3.0, -4.0, 5.0},
		{"hypot(1, 1) = sqrt(2)", 1.0, 1.0, float32(stdmath.Sqrt(2))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputX := fillF32(minF32, tt.x)
			inputY := fillF32(minF32, tt.y)
			output := make([]float32, minF32)
			HypotPoly(inputX, inputY, output)
			got := output[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Hypot(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestHypotPoly_Float64(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		y    float64
		want float64
	}{
		{"hypot(3, 4) = 5", 3.0, 4.0, 5.0},
		{"hypot(5, 12) = 13", 5.0, 12.0, 13.0},
		{"hypot(8, 15) = 17", 8.0, 15.0, 17.0},
		{"hypot(0, 0) = 0", 0.0, 0.0, 0.0},
		{"hypot(1, 0) = 1", 1.0, 0.0, 1.0},
		{"hypot(0, 1) = 1", 0.0, 1.0, 1.0},
		{"hypot(-3, 4) = 5", -3.0, 4.0, 5.0},
		{"hypot(3, -4) = 5", 3.0, -4.0, 5.0},
		{"hypot(-3, -4) = 5", -3.0, -4.0, 5.0},
		{"hypot(1, 1) = sqrt(2)", 1.0, 1.0, stdmath.Sqrt(2)},
		{"hypot(7, 24) = 25", 7.0, 24.0, 25.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputX := fillF64(minF64, tt.x)
			inputY := fillF64(minF64, tt.y)
			output := make([]float64, minF64)
			HypotPoly(inputX, inputY, output)
			got := output[0]

			if stdmath.Abs(got-tt.want) > 1e-6 {
				t.Errorf("Hypot(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestHypotPoly_SpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float64
		y       float64
		wantInf bool
	}{
		{"+Inf, any", stdmath.Inf(1), 5.0, true},
		{"any, +Inf", 5.0, stdmath.Inf(1), true},
		{"-Inf, any", stdmath.Inf(-1), 5.0, true},
		{"any, -Inf", 5.0, stdmath.Inf(-1), true},
		{"+Inf, -Inf", stdmath.Inf(1), stdmath.Inf(-1), true},
		{"+Inf, NaN", stdmath.Inf(1), stdmath.NaN(), true}, // Inf takes precedence
		{"NaN, +Inf", stdmath.NaN(), stdmath.Inf(1), true}, // Inf takes precedence
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputX := fillF64(minF64, tt.x)
			inputY := fillF64(minF64, tt.y)
			output := make([]float64, minF64)
			HypotPoly(inputX, inputY, output)
			got := output[0]

			if tt.wantInf && !stdmath.IsInf(got, 1) {
				t.Errorf("Hypot(%v, %v) = %v, want +Inf", tt.x, tt.y, got)
			}
		})
	}
}

func TestHypotPoly_NaN(t *testing.T) {
	// NaN with finite number should produce NaN
	inputX := fillF64(minF64, stdmath.NaN())
	inputY := fillF64(minF64, 5.0)
	output := make([]float64, minF64)
	HypotPoly(inputX, inputY, output)
	got := output[0]

	if !stdmath.IsNaN(got) {
		t.Errorf("Hypot(NaN, 5) = %v, want NaN", got)
	}
}

func BenchmarkCbrtPoly(b *testing.B) {
	input := fillF32(minF32, 27.0)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CbrtPoly(input, output)
	}
}

func BenchmarkCbrtPoly_Float64(b *testing.B) {
	input := fillF64(minF64, 27.0)
	output := make([]float64, minF64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CbrtPoly(input, output)
	}
}

func BenchmarkHypotPoly(b *testing.B) {
	inputX := fillF32(minF32, 3.0)
	inputY := fillF32(minF32, 4.0)
	output := make([]float32, minF32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HypotPoly(inputX, inputY, output)
	}
}

func BenchmarkHypotPoly_Float64(b *testing.B) {
	inputX := fillF64(minF64, 3.0)
	inputY := fillF64(minF64, 4.0)
	output := make([]float64, minF64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HypotPoly(inputX, inputY, output)
	}
}
