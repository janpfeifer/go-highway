package softmax

import (
	"fmt"
	stdmath "math"
	"testing"
)

func TestBaseSoftmax(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "simple",
			input: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:  "negative",
			input: []float32{-1.0, -2.0, -3.0, -4.0},
		},
		{
			name:  "mixed",
			input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			name:  "large values",
			input: []float32{100.0, 101.0, 102.0, 103.0},
		},
		{
			name:  "simd width",
			input: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:  "larger than simd",
			input: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := make([]float32, len(tt.input))
			BaseSoftmax(tt.input, output)

			// Verify properties of softmax:
			// 1. All values between 0 and 1
			// 2. Sum equals 1
			var sum float32
			for i, v := range output {
				if v < 0 || v > 1 {
					t.Errorf("output[%d] = %v, want value in [0, 1]", i, v)
				}
				sum += v
			}

			if stdmath.Abs(float64(sum-1.0)) > 1e-5 {
				t.Errorf("sum of softmax = %v, want 1.0", sum)
			}

			// Verify relative ordering is preserved (larger input -> larger output)
			for i := 0; i < len(tt.input)-1; i++ {
				for j := i + 1; j < len(tt.input); j++ {
					if tt.input[i] > tt.input[j] && output[i] <= output[j] {
						t.Errorf("ordering not preserved: input[%d]=%v > input[%d]=%v but output[%d]=%v <= output[%d]=%v",
							i, tt.input[i], j, tt.input[j], i, output[i], j, output[j])
					}
				}
			}
		})
	}
}

func TestBaseSoftmax64(t *testing.T) {
	input := []float64{1.0, 2.0, 3.0, 4.0}
	output := make([]float64, len(input))

	BaseSoftmax(input, output)

	var sum float64
	for _, v := range output {
		sum += v
	}

	if stdmath.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum of softmax = %v, want 1.0", sum)
	}
}

func BenchmarkBaseSoftmax(b *testing.B) {
	sizes := []int{8, 64, 256, 1024}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i) * 0.1
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Softmax(input, output)
			}
		})
	}
}
