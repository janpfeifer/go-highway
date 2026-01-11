//go:build !noasm && arm64

package matmul

import (
	"unsafe"
)

// Minimum dimensions to use NEON vectorization
const minDimForNEON = 16

//go:noescape
func matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)

// matmulNEON uses ARM NEON FMLA instructions for matrix multiplication.
// Falls back to scalar for small matrices.
func matmulNEON(a, b, c []float32, m, n, k int) {
	// For small matrices, use scalar
	if m < minDimForNEON || n < minDimForNEON || k < minDimForNEON {
		matmulScalar(a, b, c, m, n, k)
		return
	}

	matmul_neon_f32(
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)
}

func init() {
	// Use hand-written NEON implementation on arm64
	// This overrides the generated dispatch for better performance
	// Will be overridden by SME if available
	MatMulFloat32 = matmulNEON
}
