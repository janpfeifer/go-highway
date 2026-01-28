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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Minimum size for SME (streaming mode has fixed overhead)
// Benchmarks on M4 show:
// - Float32/Float64: SME wins at 256x256 and above
// - Float16/BFloat16: SME wins at 512x512 and above (higher overhead)
const (
	minSizeForSMETransposeF32 = 256
	minSizeForSMETransposeF64 = 256
	minSizeForSMETransposeF16 = 512
)

func init() {
	if !hwy.HasSME() {
		return
	}

	// Override with SME for large matrices
	neonF32 := Transpose2DFloat32
	Transpose2DFloat32 = func(src []float32, m, k int, dst []float32) {
		if m >= minSizeForSMETransposeF32 && k >= minSizeForSMETransposeF32 {
			asm.TransposeSMEF32(src, m, k, dst)
		} else {
			neonF32(src, m, k, dst)
		}
	}

	neonF64 := Transpose2DFloat64
	Transpose2DFloat64 = func(src []float64, m, k int, dst []float64) {
		if m >= minSizeForSMETransposeF64 && k >= minSizeForSMETransposeF64 {
			asm.TransposeSMEF64(src, m, k, dst)
		} else {
			neonF64(src, m, k, dst)
		}
	}

	neonF16 := Transpose2DFloat16
	Transpose2DFloat16 = func(src []hwy.Float16, m, k int, dst []hwy.Float16) {
		if m >= minSizeForSMETransposeF16 && k >= minSizeForSMETransposeF16 {
			asm.TransposeSMEF16(src, m, k, dst)
		} else {
			neonF16(src, m, k, dst)
		}
	}

	neonBF16 := Transpose2DBFloat16
	Transpose2DBFloat16 = func(src []hwy.BFloat16, m, k int, dst []hwy.BFloat16) {
		if m >= minSizeForSMETransposeF16 && k >= minSizeForSMETransposeF16 {
			asm.TransposeSMEBF16(src, m, k, dst)
		} else {
			neonBF16(src, m, k, dst)
		}
	}
}
