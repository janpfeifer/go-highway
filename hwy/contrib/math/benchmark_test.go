//go:build (amd64 && goexperiment.simd) || arm64

package math_test

import (
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	hwymath "github.com/ajroetker/go-highway/hwy/contrib/math"
)

// ============================================================================
// Dispatch-based Benchmarks (auto-selects best SIMD implementation)
// ============================================================================

func BenchmarkExpPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.ExpPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.ExpPoly(input64, output64)
		}
	})
}

func BenchmarkLogPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100+1) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.LogPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100+1) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.LogPoly(input64, output64)
		}
	})
}

func BenchmarkSinPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.SinPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.SinPoly(input64, output64)
		}
	})
}

func BenchmarkCosPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.CosPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.CosPoly(input64, output64)
		}
	})
}

func BenchmarkTanhPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.TanhPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100-50) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.TanhPoly(input64, output64)
		}
	})
}

func BenchmarkSigmoidPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.SigmoidPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100-50) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.SigmoidPoly(input64, output64)
		}
	})
}

func BenchmarkErfPoly(b *testing.B) {
	size := 4096
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("Float32", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.ErfPoly(input, output)
		}
	})

	input64 := make([]float64, size)
	output64 := make([]float64, size)
	for i := range input64 {
		input64[i] = float64(i%100-50) * 0.1
	}

	b.Run("Float64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			hwymath.ErfPoly(input64, output64)
		}
	})
}

// ============================================================================
// Scaling Benchmarks (different sizes)
// ============================================================================

func BenchmarkExpPoly_Scaling(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i%100) * 0.1
		}

		name := sizeLabel(size)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				hwymath.ExpPoly(input, output)
			}
		})
	}
}

func BenchmarkSigmoidPoly_Scaling(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, size := range sizes {
		input := make([]float32, size)
		output := make([]float32, size)
		for i := range input {
			input[i] = float32(i%100-50) * 0.1
		}

		name := sizeLabel(size)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				hwymath.SigmoidPoly(input, output)
			}
		})
	}
}

// ============================================================================
// Throughput Benchmarks (bytes/sec)
// ============================================================================

func BenchmarkThroughput_Exp(b *testing.B) {
	size := 16384
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 4)) // 4 bytes per float32
		for i := 0; i < b.N; i++ {
			hwymath.ExpPoly(input, output)
		}
	})

	b.Run("Stdlib", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			for j := range input {
				output[j] = float32(stdmath.Exp(float64(input[j])))
			}
		}
	})
}

func BenchmarkThroughput_Sigmoid(b *testing.B) {
	size := 16384
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100-50) * 0.1
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			hwymath.SigmoidPoly(input, output)
		}
	})

	b.Run("Stdlib", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			for j := range input {
				output[j] = float32(1.0 / (1.0 + stdmath.Exp(-float64(input[j]))))
			}
		}
	})
}

// ============================================================================
// Helper Functions
// ============================================================================

func sizeLabel(size int) string {
	switch {
	case size >= 16384:
		return "16K"
	case size >= 4096:
		return "4K"
	case size >= 1024:
		return "1K"
	case size >= 256:
		return "256"
	default:
		return "64"
	}
}

// ============================================================================
// SIMD Level Info
// ============================================================================

func TestPrintSIMDLevel(t *testing.T) {
	t.Logf("SIMD Level: %s", hwy.CurrentLevel())
	t.Logf("SIMD Width: %d bytes", hwy.CurrentWidth())
	t.Logf("AVX2 Available: %v", hwy.CurrentLevel() >= hwy.DispatchAVX2)
	t.Logf("AVX512 Available: %v", hwy.CurrentLevel() >= hwy.DispatchAVX512)
}
