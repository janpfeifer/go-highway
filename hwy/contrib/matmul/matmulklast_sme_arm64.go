// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && darwin && arm64

package matmul

import (
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum dimensions to use SME FMOPA for MatMulKLast
// SME with transpose is faster than NEON dot-product even at small sizes
// (2.2x faster at 64x64, 3x+ faster at larger sizes).
// Only use NEON for very small matrices where transpose overhead dominates.
const minDimForSMEKLast = 32

// Buffer pools for MatMulKLast transpose operations
// These are separate from the regular matmul pools since MatMulKLast
// transposes both A and B, potentially needing different sizes.
var klastTransposePoolA32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var klastTransposePoolB32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var klastTransposePoolA64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var klastTransposePoolB64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var klastTransposePoolAF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var klastTransposePoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var klastTransposePoolABF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

var klastTransposePoolBBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// matmulKLastFMOPA uses ARM SME FMOPA for MatMulKLast.
//
// MatMulKLast computes C = A @ B^T where:
//   - A is M x K (row-major)
//   - B is N x K (row-major)
//   - C is M x N (row-major)
//
// The existing FMOPA kernel expects:
//   - AT: K x M (transposed A)
//   - B:  K x N (normal B)
//   - Computes: (AT)^T @ B = A @ B
//
// For MatMulKLast (C = A @ B^T):
//   - Transpose A to get AT [K, M]
//   - Transpose B to get BT [K, N]
//   - Call FMOPA(AT, BT) which computes (AT)^T @ BT = A @ B^T ✓
func matmulKLastFMOPA(a, b, c []float32, m, n, k int) {
	// For non-aligned sizes, fall back to NEON dot-product assembly
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulKLastNEONF32(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (transpose + streaming mode overhead)
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONF32(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolA32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolB32.Get().([]float32)
	if cap(btBuf) < btSize {
		btBuf = make([]float32, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Call FMOPA: (AT)^T @ BT = A @ B^T
	asm.MatMulFMOPAF32(atBuf, btBuf, c, m, n, k)

	// Return buffers to pool
	klastTransposePoolA32.Put(atBuf)
	klastTransposePoolB32.Put(btBuf)
}

// matmulKLastFMOPA64 uses ARM SME FMOPA for float64 MatMulKLast.
func matmulKLastFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for float64), fall back to NEON assembly
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		asm.MatMulKLastNEONF64(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONF64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolA64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolB64.Get().([]float64)
	if cap(btBuf) < btSize {
		btBuf = make([]float64, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Call FMOPA: (AT)^T @ BT = A @ B^T
	asm.MatMulFMOPAF64(atBuf, btBuf, c, m, n, k)

	// Return buffers to pool
	klastTransposePoolA64.Put(atBuf)
	klastTransposePoolB64.Put(btBuf)
}

// matmulKLastFMOPAF16 uses ARM SME FMOPA for float16 MatMulKLast.
func matmulKLastFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// For non-aligned sizes, fall back to NEON assembly
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulKLastNEONF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolAF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolBF16.Get().([]hwy.Float16)
	if cap(btBuf) < btSize {
		btBuf = make([]hwy.Float16, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Scratch buffer for f32->f16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call FMOPA: (AT)^T @ BT = A @ B^T
	asm.MatmulFmopaAtF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(btBuf)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffers to pool
	klastTransposePoolAF16.Put(atBuf)
	klastTransposePoolBF16.Put(btBuf)
}

// matmulKLastFMOPABF16 uses ARM SME BFMOPA for bfloat16 MatMulKLast.
func matmulKLastFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For non-aligned sizes, fall back to NEON assembly
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster
	if m < minDimForSMEKLast || n < minDimForSMEKLast || k < minDimForSMEKLast {
		asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffers from pool
	atSize := k * m
	btSize := k * n

	atBuf := klastTransposePoolABF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	btBuf := klastTransposePoolBBF16.Get().([]hwy.BFloat16)
	if cap(btBuf) < btSize {
		btBuf = make([]hwy.BFloat16, btSize)
	} else {
		btBuf = btBuf[:btSize]
	}

	// Transpose A (M×K) to AT (K×M)
	transposeMatrix(a, m, k, atBuf)

	// Transpose B (N×K) to BT (K×N)
	transposeMatrix(b, n, k, btBuf)

	// Scratch buffer for f32->bf16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call BFMOPA: (AT)^T @ BT = A @ B^T
	asm.MatmulBfmopaAtBF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(btBuf)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffers to pool
	klastTransposePoolABF16.Put(atBuf)
	klastTransposePoolBBF16.Put(btBuf)
}

func init() {
	if hwy.HasSME() {
		// Use FMOPA implementation for large aligned matrices
		MatMulKLastFloat32 = matmulKLastFMOPA
		MatMulKLastFloat64 = matmulKLastFMOPA64
		MatMulKLastFloat16 = matmulKLastFMOPAF16
		MatMulKLastBFloat16 = matmulKLastFMOPABF16

		// Blocked versions use the same approach
		MatMulKLastBlockedFloat32 = matmulKLastFMOPA
		MatMulKLastBlockedFloat64 = matmulKLastFMOPA64
		MatMulKLastBlockedFloat16 = matmulKLastFMOPAF16
		MatMulKLastBlockedBFloat16 = matmulKLastFMOPABF16
	}
}
