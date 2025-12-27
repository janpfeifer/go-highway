package gelu

import (
	"fmt"
	stdmath "math"
	"testing"
)

// referenceGELU computes GELU using standard library for comparison.
func referenceGELU(x float64) float64 {
	return x * 0.5 * (1.0 + stdmath.Erf(x/stdmath.Sqrt2))
}

func TestBaseGELU(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "zeros",
			input: []float32{0, 0, 0, 0},
		},
		{
			name:  "positive",
			input: []float32{0.5, 1.0, 1.5, 2.0},
		},
		{
			name:  "negative",
			input: []float32{-0.5, -1.0, -1.5, -2.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "simd width",
			input: []float32{-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5},
		},
		{
			name:  "larger than simd",
			input: []float32{-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			BaseGELU(tt.input, output)

			for i, x := range tt.input {
				expected := float32(referenceGELU(float64(x)))
				got := output[i]

				// Allow small tolerance for floating point differences
				diff := stdmath.Abs(float64(got - expected))
				if diff > 1e-5 {
					t.Errorf("GELU(%v) = %v, want %v (diff: %v)", x, got, expected, diff)
				}
			}
		})
	}
}

func TestBaseGELU64(t *testing.T) {
	input := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	output := make([]float64, len(input))

	BaseGELU(input, output)

	for i, x := range input {
		expected := referenceGELU(x)
		got := output[i]

		diff := stdmath.Abs(got - expected)
		if diff > 1e-10 {
			t.Errorf("GELU(%v) = %v, want %v (diff: %v)", x, got, expected, diff)
		}
	}
}

func TestBaseGELUApprox(t *testing.T) {
	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	output := make([]float32, len(input))

	BaseGELUApprox(input, output)

	// The approximation should be close but not exact
	for i, x := range input {
		expected := float32(referenceGELU(float64(x)))
		got := output[i]

		// Approximation allows larger tolerance
		diff := stdmath.Abs(float64(got - expected))
		if diff > 0.05 { // 5% tolerance for approximation
			t.Errorf("GELUApprox(%v) = %v, want ~%v (diff: %v)", x, got, expected, diff)
		}
	}
}

func TestGELUProperties(t *testing.T) {
	// GELU(0) should be 0
	input := []float32{0}
	output := make([]float32, 1)
	BaseGELU(input, output)
	if stdmath.Abs(float64(output[0])) > 1e-6 {
		t.Errorf("GELU(0) = %v, want 0", output[0])
	}

	// GELU is approximately linear for large positive x
	input = []float32{10}
	BaseGELU(input, output)
	if stdmath.Abs(float64(output[0]-10)) > 1e-3 {
		t.Errorf("GELU(10) = %v, want ~10", output[0])
	}

	// GELU approaches 0 for large negative x
	input = []float32{-10}
	BaseGELU(input, output)
	if stdmath.Abs(float64(output[0])) > 1e-3 {
		t.Errorf("GELU(-10) = %v, want ~0", output[0])
	}
}

func BenchmarkBaseGELU(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i-size/2) * 0.1
		}

		b.Run(fmt.Sprintf("exact_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				BaseGELU(input, output)
			}
		})

		b.Run(fmt.Sprintf("approx_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				BaseGELUApprox(input, output)
			}
		})
	}
}
