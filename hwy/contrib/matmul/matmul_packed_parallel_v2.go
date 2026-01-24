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

// getCacheParamsV2 returns V2-optimized cache parameters for the current architecture.
// V2 uses smaller Mc/Nc for the packed output buffer pattern.
func getCacheParamsV2[T hwy.Floats]() CacheParams {
	lanes := hwy.Zero[T]().NumLanes()

	// Detect architecture from vector width
	switch lanes {
	case 16: // AVX-512 float32
		var zero T
		if isFloat64(zero) {
			// This would be unusual (AVX2 doesn't have 16-lane float64)
			return CacheParamsV2AVX2()
		}
		return CacheParamsV2AVX512()
	case 8: // AVX2 float32, AVX-512 float64, or NEON float64
		var zero T
		if isFloat64(zero) {
			// AVX-512 float64 or NEON float64
			if runtime.GOARCH == "arm64" {
				return CacheParamsV2NEON()
			}
			return CacheParamsV2AVX512()
		}
		// AVX2 float32
		return CacheParamsV2AVX2()
	case 4: // NEON float32 or fallback float64
		if runtime.GOARCH == "arm64" {
			return CacheParamsV2NEON()
		}
		return CacheParamsV2Fallback()
	case 2: // NEON float64
		return CacheParamsV2NEON()
	default:
		return CacheParamsV2Fallback()
	}
}

// ParallelPackedMatMulV2 computes C = A * B using the optimized parallel
// algorithm inspired by gomlx's packgemm-simd-large-opt.
//
// Key optimizations over V1:
//   - WorkersPool with Saturate for efficient worker management
//   - feedWorkItems for intelligent work distribution
//   - Packed output buffer for faster micro-kernel writes
//   - SIMD-optimized output application
//
// For small matrices, falls back to single-threaded PackedMatMul.
//
// Parameters:
//   - a: Input matrix A in row-major order (M × K)
//   - b: Input matrix B in row-major order (K × N)
//   - c: Output matrix C in row-major order (M × N), will be zeroed
//   - m, n, k: Matrix dimensions
func ParallelPackedMatMulV2[T hwy.Floats](a, b, c []T, m, n, k int) {
	totalOps := m * n * k

	// For small matrices, use single-threaded version
	if totalOps < MinPackedParallelOps {
		PackedMatMul(a, b, c, m, n, k)
		return
	}

	pool := NewWorkersPool()
	params := getCacheParamsV2[T]()
	maxWorkers := pool.AdjustedMaxParallelism()

	// Zero the output matrix once (shared across all workers)
	zeroMatrix(c, m*n)

	// For single-worker case, run without parallelization overhead
	if maxWorkers <= 1 {
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())
		packedOut := make([]T, params.PackedOutputSize())
		processGEMMSliceV2(a, b, c, m, n, k, 0, m, 0, n, packedA, packedB, packedOut, params)
		return
	}

	// Create work channel and feed work items
	// Use batchSize=1 since we're treating the whole matrix as one batch
	workChan := make(chan workItem, max(2*maxWorkers, 100))
	go feedWorkItems(1, m, n, params, maxWorkers, workChan)

	// Saturate workers to consume work items
	pool.Saturate(func() {
		// Each worker has its own buffers
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())
		packedOut := make([]T, params.PackedOutputSize())

		for item := range workChan {
			// Process this work item's slice of the output
			processGEMMSliceV2(
				a, b, c, m, n, k,
				item.lhsRowStart, item.lhsRowEnd,
				item.rhsColStart, item.rhsColEnd,
				packedA, packedB, packedOut, params,
			)
		}
	})
}

// processGEMMSliceV2 computes a slice of C using the packed output buffer pattern.
// C[lhsRowStart:lhsRowEnd, rhsColStart:rhsColEnd] += A[lhsRowStart:lhsRowEnd, :] * B[:, rhsColStart:rhsColEnd]
//
// This function uses an intermediate packed output buffer for faster writes,
// then applies the result to the final output with SIMD.
func processGEMMSliceV2[T hwy.Floats](
	a, b, c []T, m, n, k int,
	lhsRowStart, lhsRowEnd int,
	rhsColStart, rhsColEnd int,
	packedA, packedB, packedOut []T,
	params CacheParams,
) {
	mr, nr := params.Mr, params.Nr
	kc, mc, nc := params.Kc, params.Mc, params.Nc

	sliceM := lhsRowEnd - lhsRowStart
	sliceN := rhsColEnd - rhsColStart

	// Loop 5: B panels (L3 blocking) - within slice N range
	for jc := 0; jc < sliceN; jc += nc {
		jcEnd := min(jc+nc, sliceN)
		panelCols := jcEnd - jc
		globalJC := rhsColStart + jc

		// Loop 4: K blocking (L1)
		for pc := 0; pc < k; pc += kc {
			pcEnd := min(pc+kc, k)
			panelK := pcEnd - pc

			// Pack B panel: B[pc:pcEnd, globalJC:globalJC+panelCols]
			PackRHSFast(b, packedB, n, pc, globalJC, panelK, panelCols, nr)

			// Loop 3: A panels (L2 blocking) - within slice M range
			for ic := 0; ic < sliceM; ic += mc {
				icEnd := min(ic+mc, sliceM)
				panelRows := icEnd - ic
				globalIC := lhsRowStart + ic

				// Pack A panel: A[globalIC:globalIC+panelRows, pc:pcEnd]
				PackLHS(a, packedA, m, k, globalIC, pc, panelRows, panelK, mr)

				// GEBP with packed output buffer
				gebpWithPackedOutput(
					packedA, packedB, packedOut, c,
					n, globalIC, globalJC,
					panelRows, panelCols, panelK,
					mr, nr, nc,
					pc > 0, // accumulate if not first K panel
				)
			}
		}
	}
}

// gebpWithPackedOutput performs GEBP using an intermediate packed output buffer.
// The micro-kernel writes to packedOut, then we apply to c with alpha/beta.
//
// This allows the micro-kernel to write full tiles without bounds checking,
// improving performance.
func gebpWithPackedOutput[T hwy.Floats](
	packedA, packedB, packedOut, c []T,
	cStride int,
	outputRowStart, outputColStart int,
	panelRows, panelCols, panelK int,
	mr, nr, ncStride int,
	accumulate bool,
) {
	numMicroPanelsA := (panelRows + mr - 1) / mr
	numMicroPanelsB := (panelCols + nr - 1) / nr

	// Compute active rows/cols in last micro-panels
	activeRowsLast := panelRows - (numMicroPanelsA-1)*mr
	if activeRowsLast <= 0 {
		activeRowsLast = mr
	}
	activeColsLast := panelCols - (numMicroPanelsB-1)*nr
	if activeColsLast <= 0 {
		activeColsLast = nr
	}

	// Zero the packed output buffer for this panel
	zeroSlice(packedOut, panelRows*ncStride)

	// Loop 2: micro-tile columns (jr)
	for jPanel := 0; jPanel < numMicroPanelsB; jPanel++ {
		bPanelOffset := jPanel * panelK * nr

		// Loop 1: micro-tile rows (ir)
		for iPanel := 0; iPanel < numMicroPanelsA; iPanel++ {
			aPanelOffset := iPanel * panelK * mr

			// Compute output position in packed buffer
			outRowStart := iPanel * mr
			outColStart := jPanel * nr

			// Call micro-kernel to write to packed output
			packedMicroKernelToBuffer(
				packedA[aPanelOffset:],
				packedB[bPanelOffset:],
				packedOut,
				ncStride, // stride in packed output
				outRowStart, outColStart,
				panelK, mr, nr,
			)
		}
	}

	// Apply packed output to final output matrix with proper active region handling
	if accumulate {
		// Accumulate: c += packedOut (for K panels after the first)
		ApplyPackedOutputAccum(
			packedOut, c,
			ncStride,
			outputRowStart, outputColStart,
			cStride,
			panelRows, panelCols,
		)
	} else {
		// First K panel: just copy (alpha=1, beta=0)
		ApplyPackedOutputSimple(
			packedOut, c,
			ncStride,
			outputRowStart, outputColStart,
			cStride,
			panelRows, panelCols,
		)
	}
}

// packedMicroKernelToBuffer computes a micro-tile and writes to a buffer.
// This writes a full Mr x Nr tile without bounds checking.
//
// Optimized for mr=4 and nr=2*lanes (8 accumulators).
func packedMicroKernelToBuffer[T hwy.Floats](
	packedA, packedB []T,
	output []T,
	outputStride int,
	outRowStart, outColStart int,
	panelK, mr, nr int,
) {
	lanes := hwy.Zero[T]().NumLanes()
	numBVecs := nr / lanes

	// For typical 4x(2*lanes) config, we have 8 accumulators.
	// Unroll for common case.
	if mr == 4 && numBVecs == 2 {
		packedMicroKernel4x2(packedA, packedB, output, outputStride, outRowStart, outColStart, panelK, lanes)
		return
	}

	// Generic fallback (allocates, but rare)
	packedMicroKernelGeneric(packedA, packedB, output, outputStride, outRowStart, outColStart, panelK, mr, nr, lanes)
}

// packedMicroKernel4x2 is optimized for mr=4, numBVecs=2 (common case).
// Includes 4x K-loop unrolling for better instruction-level parallelism.
func packedMicroKernel4x2[T hwy.Floats](
	packedA, packedB []T,
	output []T,
	outputStride int,
	outRowStart, outColStart int,
	panelK, lanes int,
) {
	// 4 rows × 2 B vectors = 8 accumulator vectors
	acc00 := hwy.Zero[T]()
	acc01 := hwy.Zero[T]()
	acc10 := hwy.Zero[T]()
	acc11 := hwy.Zero[T]()
	acc20 := hwy.Zero[T]()
	acc21 := hwy.Zero[T]()
	acc30 := hwy.Zero[T]()
	acc31 := hwy.Zero[T]()

	nr := 2 * lanes
	mr := 4
	aIdx := 0
	bIdx := 0

	// BCE hints for bounds check elimination
	_ = packedA[panelK*mr-1]
	_ = packedB[panelK*nr-1]

	// Main loop: 4x unrolled for better ILP
	p := 0
	for ; p+3 < panelK; p += 4 {
		// --- Step 0 ---
		bVec0_0 := hwy.Load(packedB[bIdx:])
		bVec1_0 := hwy.Load(packedB[bIdx+lanes:])
		a0_0 := hwy.Set(packedA[aIdx])
		a1_0 := hwy.Set(packedA[aIdx+1])
		a2_0 := hwy.Set(packedA[aIdx+2])
		a3_0 := hwy.Set(packedA[aIdx+3])

		acc00 = hwy.MulAdd(a0_0, bVec0_0, acc00)
		acc01 = hwy.MulAdd(a0_0, bVec1_0, acc01)
		acc10 = hwy.MulAdd(a1_0, bVec0_0, acc10)
		acc11 = hwy.MulAdd(a1_0, bVec1_0, acc11)
		acc20 = hwy.MulAdd(a2_0, bVec0_0, acc20)
		acc21 = hwy.MulAdd(a2_0, bVec1_0, acc21)
		acc30 = hwy.MulAdd(a3_0, bVec0_0, acc30)
		acc31 = hwy.MulAdd(a3_0, bVec1_0, acc31)

		// --- Step 1 ---
		bVec0_1 := hwy.Load(packedB[bIdx+nr:])
		bVec1_1 := hwy.Load(packedB[bIdx+nr+lanes:])
		a0_1 := hwy.Set(packedA[aIdx+mr])
		a1_1 := hwy.Set(packedA[aIdx+mr+1])
		a2_1 := hwy.Set(packedA[aIdx+mr+2])
		a3_1 := hwy.Set(packedA[aIdx+mr+3])

		acc00 = hwy.MulAdd(a0_1, bVec0_1, acc00)
		acc01 = hwy.MulAdd(a0_1, bVec1_1, acc01)
		acc10 = hwy.MulAdd(a1_1, bVec0_1, acc10)
		acc11 = hwy.MulAdd(a1_1, bVec1_1, acc11)
		acc20 = hwy.MulAdd(a2_1, bVec0_1, acc20)
		acc21 = hwy.MulAdd(a2_1, bVec1_1, acc21)
		acc30 = hwy.MulAdd(a3_1, bVec0_1, acc30)
		acc31 = hwy.MulAdd(a3_1, bVec1_1, acc31)

		// --- Step 2 ---
		bVec0_2 := hwy.Load(packedB[bIdx+2*nr:])
		bVec1_2 := hwy.Load(packedB[bIdx+2*nr+lanes:])
		a0_2 := hwy.Set(packedA[aIdx+2*mr])
		a1_2 := hwy.Set(packedA[aIdx+2*mr+1])
		a2_2 := hwy.Set(packedA[aIdx+2*mr+2])
		a3_2 := hwy.Set(packedA[aIdx+2*mr+3])

		acc00 = hwy.MulAdd(a0_2, bVec0_2, acc00)
		acc01 = hwy.MulAdd(a0_2, bVec1_2, acc01)
		acc10 = hwy.MulAdd(a1_2, bVec0_2, acc10)
		acc11 = hwy.MulAdd(a1_2, bVec1_2, acc11)
		acc20 = hwy.MulAdd(a2_2, bVec0_2, acc20)
		acc21 = hwy.MulAdd(a2_2, bVec1_2, acc21)
		acc30 = hwy.MulAdd(a3_2, bVec0_2, acc30)
		acc31 = hwy.MulAdd(a3_2, bVec1_2, acc31)

		// --- Step 3 ---
		bVec0_3 := hwy.Load(packedB[bIdx+3*nr:])
		bVec1_3 := hwy.Load(packedB[bIdx+3*nr+lanes:])
		a0_3 := hwy.Set(packedA[aIdx+3*mr])
		a1_3 := hwy.Set(packedA[aIdx+3*mr+1])
		a2_3 := hwy.Set(packedA[aIdx+3*mr+2])
		a3_3 := hwy.Set(packedA[aIdx+3*mr+3])

		acc00 = hwy.MulAdd(a0_3, bVec0_3, acc00)
		acc01 = hwy.MulAdd(a0_3, bVec1_3, acc01)
		acc10 = hwy.MulAdd(a1_3, bVec0_3, acc10)
		acc11 = hwy.MulAdd(a1_3, bVec1_3, acc11)
		acc20 = hwy.MulAdd(a2_3, bVec0_3, acc20)
		acc21 = hwy.MulAdd(a2_3, bVec1_3, acc21)
		acc30 = hwy.MulAdd(a3_3, bVec0_3, acc30)
		acc31 = hwy.MulAdd(a3_3, bVec1_3, acc31)

		aIdx += 4 * mr
		bIdx += 4 * nr
	}

	// Handle remaining iterations (0-3)
	for ; p < panelK; p++ {
		bVec0 := hwy.Load(packedB[bIdx:])
		bVec1 := hwy.Load(packedB[bIdx+lanes:])
		bIdx += nr

		a0 := hwy.Set(packedA[aIdx])
		a1 := hwy.Set(packedA[aIdx+1])
		a2 := hwy.Set(packedA[aIdx+2])
		a3 := hwy.Set(packedA[aIdx+3])
		aIdx += mr

		acc00 = hwy.MulAdd(a0, bVec0, acc00)
		acc01 = hwy.MulAdd(a0, bVec1, acc01)
		acc10 = hwy.MulAdd(a1, bVec0, acc10)
		acc11 = hwy.MulAdd(a1, bVec1, acc11)
		acc20 = hwy.MulAdd(a2, bVec0, acc20)
		acc21 = hwy.MulAdd(a2, bVec1, acc21)
		acc30 = hwy.MulAdd(a3, bVec0, acc30)
		acc31 = hwy.MulAdd(a3, bVec1, acc31)
	}

	// Write accumulators to output
	outIdx0 := outRowStart*outputStride + outColStart
	outIdx1 := outIdx0 + outputStride
	outIdx2 := outIdx1 + outputStride
	outIdx3 := outIdx2 + outputStride

	hwy.Store(acc00, output[outIdx0:])
	hwy.Store(acc01, output[outIdx0+lanes:])
	hwy.Store(acc10, output[outIdx1:])
	hwy.Store(acc11, output[outIdx1+lanes:])
	hwy.Store(acc20, output[outIdx2:])
	hwy.Store(acc21, output[outIdx2+lanes:])
	hwy.Store(acc30, output[outIdx3:])
	hwy.Store(acc31, output[outIdx3+lanes:])
}

// packedMicroKernelGeneric is the generic fallback for non-4x2 configs.
func packedMicroKernelGeneric[T hwy.Floats](
	packedA, packedB []T,
	output []T,
	outputStride int,
	outRowStart, outColStart int,
	panelK, mr, nr, lanes int,
) {
	numBVecs := nr / lanes

	// Allocate accumulators (this path is rarely taken)
	acc := make([]hwy.Vec[T], mr*numBVecs)
	for i := range acc {
		acc[i] = hwy.Zero[T]()
	}

	aIdx := 0
	bIdx := 0

	for p := 0; p < panelK; p++ {
		// Load B vectors
		for v := 0; v < numBVecs; v++ {
			bVec := hwy.Load(packedB[bIdx+v*lanes:])

			// FMA with each A row
			for row := 0; row < mr; row++ {
				aVec := hwy.Set(packedA[aIdx+row])
				accIdx := row*numBVecs + v
				acc[accIdx] = hwy.MulAdd(aVec, bVec, acc[accIdx])
			}
		}
		aIdx += mr
		bIdx += nr
	}

	// Write accumulators
	for row := 0; row < mr; row++ {
		outIdx := (outRowStart+row)*outputStride + outColStart
		for v := 0; v < numBVecs; v++ {
			hwy.Store(acc[row*numBVecs+v], output[outIdx+v*lanes:])
		}
	}
}

// zeroSlice zeros a slice using SIMD.
func zeroSlice[T hwy.Floats](s []T, n int) {
	vZero := hwy.Zero[T]()
	lanes := vZero.NumLanes()

	var idx int
	for idx = 0; idx+lanes <= n; idx += lanes {
		hwy.Store(vZero, s[idx:])
	}
	for ; idx < n; idx++ {
		s[idx] = 0
	}
}

// ParallelPackedMatMulV2Float32 is the non-generic version for float32.
func ParallelPackedMatMulV2Float32(a, b, c []float32, m, n, k int) {
	ParallelPackedMatMulV2(a, b, c, m, n, k)
}

// ParallelPackedMatMulV2Float64 is the non-generic version for float64.
func ParallelPackedMatMulV2Float64(a, b, c []float64, m, n, k int) {
	ParallelPackedMatMulV2(a, b, c, m, n, k)
}

// BatchParallelPackedMatMulV2 computes batched C = A * B using the optimized
// parallel algorithm.
//
// Parameters:
//   - a: Batched input matrix A [batchSize, M, K] in row-major order
//   - b: Batched input matrix B [batchSize, K, N] in row-major order
//   - c: Batched output matrix C [batchSize, M, N] in row-major order
//   - batchSize, m, n, k: Dimensions
func BatchParallelPackedMatMulV2[T hwy.Floats](a, b, c []T, batchSize, m, n, k int) {
	totalOps := batchSize * m * n * k

	// For small total work, use single-threaded version
	if totalOps < MinPackedParallelOps {
		lhsStride := m * k
		rhsStride := k * n
		outStride := m * n
		for batch := 0; batch < batchSize; batch++ {
			PackedMatMul(
				a[batch*lhsStride:(batch+1)*lhsStride],
				b[batch*rhsStride:(batch+1)*rhsStride],
				c[batch*outStride:(batch+1)*outStride],
				m, n, k,
			)
		}
		return
	}

	pool := NewWorkersPool()
	params := getCacheParamsV2[T]()
	maxWorkers := pool.AdjustedMaxParallelism()

	lhsStride := m * k
	rhsStride := k * n
	outStride := m * n

	// Zero all output matrices
	zeroMatrix(c, batchSize*m*n)

	// Create work channel with intelligent work distribution
	workChan := make(chan workItem, max(2*maxWorkers, 100))
	go feedWorkItems(batchSize, m, n, params, maxWorkers, workChan)

	// Saturate workers
	pool.Saturate(func() {
		// Each worker has its own buffers
		packedA := make([]T, params.PackedASize())
		packedB := make([]T, params.PackedBSize())
		packedOut := make([]T, params.PackedOutputSize())

		for item := range workChan {
			// Process each batch in this work item
			for batch := item.batchStart; batch < item.batchEnd; batch++ {
				batchA := a[batch*lhsStride : (batch+1)*lhsStride]
				batchB := b[batch*rhsStride : (batch+1)*rhsStride]
				batchC := c[batch*outStride : (batch+1)*outStride]

				processGEMMSliceV2(
					batchA, batchB, batchC, m, n, k,
					item.lhsRowStart, item.lhsRowEnd,
					item.rhsColStart, item.rhsColEnd,
					packedA, packedB, packedOut, params,
				)
			}
		}
	})
}
