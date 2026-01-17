//go:build !noasm && darwin && arm64

package asm

import (
	"math"
	"testing"
)

// transposeMatrix transposes M×K matrix A into K×M matrix AT
func transposeMatrix(a []float32, m, k int, at []float32) {
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			at[j*m+i] = a[i*k+j]
		}
	}
}

// transposeMatrix64 transposes M×K matrix A into K×M matrix AT for float64
func transposeMatrix64(a []float64, m, k int, at []float64) {
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			at[j*m+i] = a[i*k+j]
		}
	}
}

// matmulReference computes C = A * B using naive triple loop
func matmulReference(a, b, c []float32, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// matmulReference64 computes C = A * B for float64
func matmulReference64(a, b, c []float64, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// TestGoatGeneratedF32 tests correctness of the goat-generated SME f32 implementation
func TestGoatGeneratedF32(t *testing.T) {
	sizes := []int{16, 32, 64, 128}

	for _, size := range sizes {
		m, n, k := size, size, size
		t.Run(sizeStr(size), func(t *testing.T) {
			a := make([]float32, m*k)
			b := make([]float32, k*n)
			at := make([]float32, k*m)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			// Fill with test values
			for i := range a {
				a[i] = float32(i%7) + 0.5
			}
			for i := range b {
				b[i] = float32(i%11) + 0.25
			}

			// Transpose A for the AT-based function
			transposeMatrix(a, m, k, at)

			// Reference implementation
			matmulReference(a, b, expected, m, n, k)

			// Goat-generated SME implementation
			MatMulFMOPAF32(at, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}
			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestGoatGeneratedF64 tests correctness of the goat-generated SME f64 implementation
func TestGoatGeneratedF64(t *testing.T) {
	sizes := []int{8, 16, 32, 64}

	for _, size := range sizes {
		m, n, k := size, size, size
		t.Run(sizeStr(size), func(t *testing.T) {
			a := make([]float64, m*k)
			b := make([]float64, k*n)
			at := make([]float64, k*m)
			c := make([]float64, m*n)
			expected := make([]float64, m*n)

			// Fill with test values
			for i := range a {
				a[i] = float64(i%7) + 0.5
			}
			for i := range b {
				b[i] = float64(i%11) + 0.25
			}

			// Transpose A for the AT-based function
			transposeMatrix64(a, m, k, at)

			// Reference implementation
			matmulReference64(a, b, expected, m, n, k)

			// Goat-generated SME implementation
			MatMulFMOPAF64(at, b, c, m, n, k)

			var maxErr float64
			for i := range c {
				err := math.Abs(c[i] - expected[i])
				if err > maxErr {
					maxErr = err
				}
			}
			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			if maxErr > 1e-9 {
				t.Errorf("max error %e exceeds threshold 1e-9", maxErr)
			}
		})
	}
}

func sizeStr(n int) string {
	s := ""
	if n >= 100 {
		s += string(rune('0' + n/100))
	}
	if n >= 10 {
		s += string(rune('0' + (n/10)%10))
	}
	s += string(rune('0' + n%10))
	return s
}
