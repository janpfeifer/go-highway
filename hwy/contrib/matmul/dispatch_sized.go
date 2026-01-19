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
// For large matrices, uses parallel blocked (cache efficiency + multicore).
func MatMulAuto[T hwy.Floats](a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	if totalOps < SmallMatrixThreshold {
		// Small matrices: streaming is faster (no blocking overhead)
		MatMul(a, b, c, m, n, k)
	} else {
		// Large matrices: parallel blocked (handles single-core fallback internally)
		ParallelMatMul(a, b, c, m, n, k)
	}
}

// MatMulAutoFloat32 is the non-generic version for float32.
func MatMulAutoFloat32(a, b, c []float32, m, n, k int) {
	MatMulAuto(a, b, c, m, n, k)
}

// MatMulAutoFloat64 is the non-generic version for float64.
func MatMulAutoFloat64(a, b, c []float64, m, n, k int) {
	MatMulAuto(a, b, c, m, n, k)
}
