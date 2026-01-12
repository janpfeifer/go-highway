//go:build !noasm && darwin && arm64

package matmul

import (
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// NOTE: SME implementation confirmed working on Apple M4!
//
// Initial testing showed zeros from ZA due to incorrect MOVA encodings.
// After fixing instruction encodings from LLVM reference:
//
// What WORKS on Apple M4:
//   - SMSTART/SMSTOP (streaming mode entry/exit)
//   - PTRUE, ZERO {ZA}, DUP
//   - ST1W from Z registers (SVE stores in streaming mode)
//   - FMOPA (outer product accumulate) - 512 FP32 ops per instruction!
//   - MOVA Z→ZA (write to ZA): encoding 0xc080XXXX
//   - MOVA ZA→Z (read from ZA): encoding 0xc082XXXX (bit 17 set!)
//
// Key encoding corrections:
//   - MOVA tile-to-vector requires bit 17 set (0xc082 prefix, not 0xc080)
//   - FMOPA encoding: 0x8081XXXX for za.s with z registers
//   - PTRUE p0.s for FP32 operations, not p0.b
//
// Apple M4 SVL = 512 bits, meaning:
//   - Z registers hold 16 × float32
//   - ZA tiles are 16×16 = 256 float32 elements
//   - Single FMOPA does 16×16×2 = 512 FP32 ops
//
// Two implementations available:
//   1. matmul_fmopa_f32: original with scalar A column loading (slower)
//   2. matmul_fmopa_at_f32: with pre-transposed A for contiguous loads (faster!)

// Minimum dimensions to use SME FMOPA MatMul
// With pre-transposed A, FMOPA uses contiguous loads and is competitive with NEON.
// Below this threshold, NEON is used (streaming mode has fixed overhead).
const minDimForSME = 64

//go:noescape
func matmul_sme_f32(a, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_fmopa_f32(a, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)

// Transpose buffer pools to avoid allocations
var transposePool = sync.Pool{
	New: func() interface{} {
		return make([]float32, 0, 256*256)
	},
}

var transposePool64 = sync.Pool{
	New: func() interface{} {
		return make([]float64, 0, 256*256)
	},
}

// transposeMatrix transposes M×K matrix A into K×M matrix AT (row-major to column-major)
// AT[k,i] = A[i,k]
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

// matmulStreamingSVE uses ARM streaming SVE mode with SVE instructions.
// NOTE: This does NOT work on Apple M4 - ld1w doesn't function in streaming mode.
// Kept for reference, but use matmulFMOPA instead.
func matmulStreamingSVE(a, b, c []float32, m, n, k int) {
	// For small matrices, fall back to NEON (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	matmul_sme_f32(
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)
}

// matmulFMOPA uses ARM SME FMOPA instruction for matrix multiplication.
// Uses outer product accumulate with ZA tiles - confirmed working on Apple M4!
// Processes matrices in 16x16 tiles using the ZA accumulator.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA(a, b, c []float32, m, n, k int) {
	// For non-aligned sizes, fall back to generated NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	matmul_fmopa_at_f32(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)

	// Return buffer to pool
	transposePool.Put(atBuf)
}

// matmulFMOPA64 uses ARM SME FMOPA instruction for float64 matrix multiplication.
// Uses outer product accumulate with ZA tiles - 8×8 tiles for float64.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for float64), fall back to scalar
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// For small matrices, scalar is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix64(a, m, k, atBuf)

	// Call FMOPA with transposed A
	matmul_fmopa_at_f64(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)

	// Return buffer to pool
	transposePool64.Put(atBuf)
}

func init() {
	if hwy.HasSME() {
		// Use FMOPA implementation which works on Apple M4
		// This overrides the generated dispatch for large aligned matrices
		MatMulFloat32 = matmulFMOPA
		MatMulFloat64 = matmulFMOPA64
	}
}
