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

package matmul

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input transpose_base.go -output . -targets avx2,avx512,neon,fallback -dispatch transpose

// BaseTranspose2D transposes an M×K row-major matrix to K×M.
// Uses block-based approach: load lanes×lanes block, transpose in-register, store.
func BaseTranspose2D[T hwy.Floats](src []T, m, k int, dst []T) {
	if len(src) < m*k || len(dst) < k*m {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// Process lanes×lanes blocks with SIMD
	for i := 0; i <= m-lanes; i += lanes {
		for j := 0; j <= k-lanes; j += lanes {
			transposeBlockSIMD(src, dst, i, j, m, k, lanes)
		}
	}

	// Handle edges with scalar
	transposeEdgesScalar(src, m, k, dst, lanes)
}

// transposeBlockSIMD transposes a lanes×lanes block using SIMD interleave ops.
func transposeBlockSIMD[T hwy.Floats](src, dst []T, startI, startJ, m, k, lanes int) {
	// Load `lanes` rows
	rows := make([]hwy.Vec[T], lanes)
	for r := 0; r < lanes; r++ {
		rows[r] = hwy.LoadFull(src[(startI+r)*k+startJ:])
	}

	// In-register transpose using butterfly pattern with InterleaveLower/Upper
	// For 4 lanes: 2 levels of interleave
	// For 8 lanes: 3 levels of interleave
	for level := 0; (1 << level) < lanes; level++ {
		stride := 1 << level
		newRows := make([]hwy.Vec[T], lanes)
		for i := 0; i < lanes; i += 2 * stride {
			for j := 0; j < stride; j++ {
				newRows[i+j] = hwy.InterleaveLower(rows[i+j], rows[i+j+stride])
				newRows[i+j+stride] = hwy.InterleaveUpper(rows[i+j], rows[i+j+stride])
			}
		}
		rows = newRows
	}

	// Store transposed: column c of input -> row c of output
	for c := 0; c < lanes; c++ {
		hwy.StoreFull(rows[c], dst[(startJ+c)*m+startI:])
	}
}

// transposeEdgesScalar handles non-block-aligned edges.
func transposeEdgesScalar[T hwy.Floats](src []T, m, k int, dst []T, lanes int) {
	blockM := (m / lanes) * lanes
	blockK := (k / lanes) * lanes

	// Right edge: columns [blockK, k)
	for i := 0; i < m; i++ {
		for j := blockK; j < k; j++ {
			dst[j*m+i] = src[i*k+j]
		}
	}

	// Bottom edge: rows [blockM, m), columns [0, blockK)
	for i := blockM; i < m; i++ {
		for j := 0; j < blockK; j++ {
			dst[j*m+i] = src[i*k+j]
		}
	}
}
