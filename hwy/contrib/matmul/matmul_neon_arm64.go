//go:build !noasm && arm64

package matmul

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Minimum dimensions to use NEON vectorization
const minDimForNEON = 16

//go:noescape
func matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_neon_f16(a, b, c, pm, pn, pk unsafe.Pointer)

//go:noescape
func matmul_neon_bf16(a, b, c, pm, pn, pk unsafe.Pointer)

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

// matmulNEONF16 uses ARM NEON instructions for float16 matrix multiplication.
// Falls back to generated implementation for small matrices.
func matmulNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	// For small matrices, use generated fallback
	if m < minDimForNEON || n < minDimForNEON || k < minDimForNEON {
		BaseMatMul_neon_Float16(a, b, c, m, n, k)
		return
	}

	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	matmul_neon_f16(
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// matmulNEONBF16 uses ARM NEON instructions for bfloat16 matrix multiplication.
// Falls back to generated implementation for small matrices.
func matmulNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For small matrices, use generated fallback
	if m < minDimForNEON || n < minDimForNEON || k < minDimForNEON {
		BaseMatMul_neon_BFloat16(a, b, c, m, n, k)
		return
	}

	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	matmul_neon_bf16(
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

func init() {
	// Use hand-written NEON implementation on arm64
	// This overrides the generated dispatch for better performance
	// Will be overridden by SME if available
	MatMulFloat32 = matmulNEON
}
