package math

import (
	"math"
	"testing"
)

// TestBasePoly2_F32 tests the Poly2 function for float32.
func TestBasePoly2_F32(t *testing.T) {
	// Test polynomial: 1 + 2x + 3x^2
	c0, c1, c2 := float32(1), float32(2), float32(3)

	tests := []struct {
		x    float32
		want float32
	}{
		{0, 1},  // 1 + 0 + 0 = 1
		{1, 6},  // 1 + 2 + 3 = 6
		{2, 17}, // 1 + 4 + 12 = 17
		{-1, 2}, // 1 - 2 + 3 = 2
	}

	for _, tt := range tests {
		x := []float32{tt.x}
		result := make([]float32, 1)
		BasePoly2_fallback(x, c0, c1, c2, result)

		if math.Abs(float64(result[0]-tt.want)) > 1e-5 {
			t.Errorf("BasePoly2(%v) = %v, want %v", tt.x, result[0], tt.want)
		}
	}
}

// TestBasePoly2_Vector tests the Poly2 function with multiple values.
func TestBasePoly2_Vector(t *testing.T) {
	// Test polynomial: 1 + 2x + 3x^2
	c0, c1, c2 := float32(1), float32(2), float32(3)

	x := make([]float32, 16)
	result := make([]float32, 16)
	expected := make([]float32, 16)

	for i := range x {
		xi := float32(i)
		x[i] = xi
		// 1 + 2*xi + 3*xi^2
		expected[i] = c0 + c1*xi + c2*xi*xi
	}

	BasePoly2_fallback(x, c0, c1, c2, result)

	for i := range result {
		if math.Abs(float64(result[i]-expected[i])) > 1e-4 {
			t.Errorf("BasePoly2 at index %d: got %v, want %v", i, result[i], expected[i])
		}
	}
}

// TestBaseClamp_F32 tests the Clamp function for float32.
func TestBaseClamp_F32(t *testing.T) {
	tests := []struct {
		input  float32
		minVal float32
		maxVal float32
		want   float32
	}{
		{5, 0, 10, 5},   // in range
		{-5, 0, 10, 0},  // below min
		{15, 0, 10, 10}, // above max
		{0, 0, 10, 0},   // at min
		{10, 0, 10, 10}, // at max
	}

	for _, tt := range tests {
		input := []float32{tt.input}
		output := make([]float32, 1)
		BaseClamp_fallback(input, tt.minVal, tt.maxVal, output)

		if output[0] != tt.want {
			t.Errorf("BaseClamp(%v, %v, %v) = %v, want %v", tt.input, tt.minVal, tt.maxVal, output[0], tt.want)
		}
	}
}

// TestBaseClamp_Vector tests the Clamp function with multiple values.
func TestBaseClamp_Vector(t *testing.T) {
	input := []float32{-5, -2, 0, 2, 5, 8, 10, 12, 15}
	expected := []float32{0, 0, 0, 2, 5, 8, 10, 10, 10}
	output := make([]float32, len(input))

	BaseClamp_fallback(input, 0, 10, output)

	for i := range output {
		if output[i] != expected[i] {
			t.Errorf("BaseClamp at index %d: got %v, want %v", i, output[i], expected[i])
		}
	}
}
