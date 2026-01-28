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

//go:build !noasm && arm64

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum dimensions to use NEON KLast vectorization
const minDimForNEONKLast = 16

// matmulKLastNEON uses ARM NEON for KLast matrix multiplication.
// Uses optimized tiled dot-product algorithm via GOAT-generated assembly.
// C = A * B^T where A is [M,K] and B is [N,K] (K-last layout).
func matmulKLastNEON(a, b, c []float32, m, n, k int) {
	// Fall back to scalar for small matrices
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONF32(a, b, c, m, n, k)
}

// matmulKLastNEONF64 uses ARM NEON for float64 KLast matrix multiplication.
func matmulKLastNEONF64(a, b, c []float64, m, n, k int) {
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONF64(a, b, c, m, n, k)
}

// matmulKLastNEONF16 uses ARM NEON for float16 KLast matrix multiplication.
// Uses f32 accumulation for precision.
func matmulKLastNEONF16(a, b, c []hwy.Float16, m, n, k int) {
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONF16(a, b, c, m, n, k)
}

// matmulKLastNEONBF16 uses ARM NEON for bfloat16 KLast matrix multiplication.
// Uses BFDOT for computation with f32 accumulation.
func matmulKLastNEONBF16(a, b, c []hwy.BFloat16, m, n, k int) {
	if m < minDimForNEONKLast || n < minDimForNEONKLast || k < minDimForNEONKLast {
		BaseMatMulKLast(a, b, c, m, n, k)
		return
	}
	asm.MatMulKLastNEONBF16(a, b, c, m, n, k)
}

func init() {
	// Use GOAT-generated NEON assembly for arm64
	// This overrides the hwygen-generated dispatch for better performance
	// Will be overridden by SME if available (for large matrices)
	MatMulKLastFloat32 = matmulKLastNEON
	MatMulKLastFloat64 = matmulKLastNEONF64

	// Blocked versions use the same NEON implementations
	MatMulKLastBlockedFloat32 = matmulKLastNEON
	MatMulKLastBlockedFloat64 = matmulKLastNEONF64

	// FP16/BF16 require ARMv8.2+ extensions
	if hwy.HasARMFP16() {
		MatMulKLastFloat16 = matmulKLastNEONF16
		MatMulKLastBlockedFloat16 = matmulKLastNEONF16
	}
	if hwy.HasARMBF16() {
		MatMulKLastBFloat16 = matmulKLastNEONBF16
		MatMulKLastBlockedBFloat16 = matmulKLastNEONBF16
	}
}
