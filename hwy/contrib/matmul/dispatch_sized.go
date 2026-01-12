package matmul

import "github.com/ajroetker/go-highway/hwy"

// Size-based dispatch thresholds.
// Tuned empirically - adjust based on benchmarks on target hardware.
const (
	// Below this total ops count, streaming is faster (less overhead)
	SmallMatrixThreshold = 64 * 64 * 64 // 262144 ops

	// When K/N ratio exceeds this, blocking helps reduce C traffic
	DeepKRatio = 4
)

// MatMulAuto automatically selects the best algorithm based on matrix dimensions.
// For small matrices, uses streaming (lower overhead).
// For large matrices, uses cache-tiled blocking (better cache efficiency).
func MatMulAuto[T hwy.Floats](a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	if totalOps < SmallMatrixThreshold {
		// Small matrices: streaming is faster (no blocking overhead)
		MatMul(a, b, c, m, n, k)
	} else {
		// Large matrices: blocked is faster (cache efficiency)
		BlockedMatMul(a, b, c, m, n, k)
	}
}

// MatMulAutoFloat32 is the non-generic version for float32.
func MatMulAutoFloat32(a, b, c []float32, m, n, k int) {
	totalOps := m * n * k

	if totalOps < SmallMatrixThreshold {
		MatMulFloat32(a, b, c, m, n, k)
	} else {
		BlockedMatMulFloat32(a, b, c, m, n, k)
	}
}

// MatMulAutoFloat64 is the non-generic version for float64.
func MatMulAutoFloat64(a, b, c []float64, m, n, k int) {
	totalOps := m * n * k

	if totalOps < SmallMatrixThreshold {
		MatMulFloat64(a, b, c, m, n, k)
	} else {
		BlockedMatMulFloat64(a, b, c, m, n, k)
	}
}
