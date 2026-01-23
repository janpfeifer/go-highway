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

import (
	"runtime"

	"github.com/ajroetker/go-highway/hwy"
)

// Size-based dispatch thresholds.
// Tuned empirically - adjust based on benchmarks on target hardware.
const (
	// Below this total ops count, streaming is faster (less overhead)
	SmallMatrixThreshold = 64 * 64 * 64 // 262144 ops

	// Above this total ops count, use packed matmul for best cache efficiency
	// 256^3 = 16M ops, where K-blocking benefit outweighs packing overhead
	LargeMatrixThreshold = 256 * 256 * 256 // 16777216 ops

	// When K/N ratio exceeds this, blocking helps reduce C traffic
	DeepKRatio = 4
)

// MatMulAuto automatically selects the best algorithm based on matrix dimensions.
//
// Algorithm selection based on matrix size (total ops = M * N * K):
//   - Small (<64^3): Streaming MatMul - lowest overhead, fits in cache
//   - Medium/Large: Parallel BlockedMatMul - cache tiling + parallelism
//
// On AMD64, large matrices (>=256^3) may use PackedMatMul with K-blocking.
// On ARM64, the SME/NEON blocked implementation is faster due to FMOPA
// outer product instructions that prefer full-K accumulation over K-blocking.
func MatMulAuto[T hwy.Floats](a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	if totalOps < SmallMatrixThreshold {
		// Small matrices: streaming is faster (no blocking overhead)
		MatMul(a, b, c, m, n, k)
	} else if totalOps < LargeMatrixThreshold {
		// Medium matrices: parallel blocked
		ParallelMatMul(a, b, c, m, n, k)
	} else {
		// Large matrices: architecture-dependent
		// On ARM64, SME FMOPA with full-K accumulation is faster than K-blocked packed
		// On AMD64, packed GEBP with K-blocking can help L1 cache utilization
		if runtime.GOARCH == "arm64" {
			// Use blocked (SME FMOPA or NEON) - better for outer product hardware
			ParallelMatMul(a, b, c, m, n, k)
		} else {
			// Use packed GEBP with K-blocking on AMD64
			ParallelPackedMatMul(a, b, c, m, n, k)
		}
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
