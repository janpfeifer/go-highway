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

package matvec

import (
	"unsafe"
)

// Minimum dimensions to use hand-written NEON
const minDimForNEON = 16

//go:noescape
func matvec_neon_f32(m, v, result unsafe.Pointer, rows, cols int64)

//go:noescape
func matvec_neon_f64(m, v, result unsafe.Pointer, rows, cols int64)

// matvecNEON uses ARM NEON FMLA instructions for matrix-vector multiplication.
// Falls back to generated code for small matrices.
func matvecNEON(m []float32, rows, cols int, v, result []float32) {
	// For small matrices, use generated code
	if rows < minDimForNEON || cols < minDimForNEON {
		BaseMatVec_neon(m, rows, cols, v, result)
		return
	}

	matvec_neon_f32(
		unsafe.Pointer(unsafe.SliceData(m)),
		unsafe.Pointer(unsafe.SliceData(v)),
		unsafe.Pointer(unsafe.SliceData(result)),
		int64(rows),
		int64(cols),
	)
}

// matvecNEON64 uses ARM NEON for float64 matrix-vector multiplication.
func matvecNEON64(m []float64, rows, cols int, v, result []float64) {
	// For small matrices, use generated code
	if rows < minDimForNEON || cols < minDimForNEON {
		BaseMatVec_neon_Float64(m, rows, cols, v, result)
		return
	}

	matvec_neon_f64(
		unsafe.Pointer(unsafe.SliceData(m)),
		unsafe.Pointer(unsafe.SliceData(v)),
		unsafe.Pointer(unsafe.SliceData(result)),
		int64(rows),
		int64(cols),
	)
}

func init() {
	// Use hand-written NEON implementation on arm64
	// This overrides the generated dispatch for better performance
	// Will be overridden by SME if available (for appropriate sizes)
	MatVecFloat32 = matvecNEON
	MatVecFloat64 = matvecNEON64
}
