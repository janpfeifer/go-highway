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
	"math"
	"math/rand"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// TestPackLHS verifies that LHS packing produces the expected layout.
func TestPackLHS(t *testing.T) {
	// 6x4 matrix A, Mr=2
	// A = [[ 1  2  3  4]
	//      [ 5  6  7  8]
	//      [ 9 10 11 12]
	//      [13 14 15 16]
	//      [17 18 19 20]
	//      [21 22 23 24]]
	m, k := 6, 4
	mr := 2
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(i + 1)
	}

	// Pack entire A (rowStart=0, colStart=0, panelRows=m, panelK=k)
	numPanels := (m + mr - 1) / mr
	packed := make([]float32, numPanels*k*mr)

	activeRows := BasePackLHS(a, packed, m, k, 0, 0, m, k, mr)

	// Expected packed layout: [panel, k, mr]
	// Panel 0: rows 0-1 -> for k=0: [1,5], k=1: [2,6], k=2: [3,7], k=3: [4,8]
	// Panel 1: rows 2-3 -> for k=0: [9,13], k=1: [10,14], k=2: [11,15], k=3: [12,16]
	// Panel 2: rows 4-5 -> for k=0: [17,21], k=1: [18,22], k=2: [19,23], k=3: [20,24]
	expected := []float32{
		1, 5, 2, 6, 3, 7, 4, 8, // Panel 0
		9, 13, 10, 14, 11, 15, 12, 16, // Panel 1
		17, 21, 18, 22, 19, 23, 20, 24, // Panel 2
	}

	if activeRows != mr {
		t.Errorf("activeRows = %d, want %d", activeRows, mr)
	}

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %f, want %f", i, packed[i], expected[i])
		}
	}
}

// TestPackRHS verifies that RHS packing produces the expected layout.
func TestPackRHS(t *testing.T) {
	// 4x6 matrix B, Nr=2
	// B = [[ 1  2  3  4  5  6]
	//      [ 7  8  9 10 11 12]
	//      [13 14 15 16 17 18]
	//      [19 20 21 22 23 24]]
	k, n := 4, 6
	nr := 2
	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(i + 1)
	}

	// Pack entire B (rowStart=0, colStart=0, panelK=k, panelCols=n)
	numPanels := (n + nr - 1) / nr
	packed := make([]float32, numPanels*k*nr)

	activeCols := BasePackRHS(b, packed, k, n, 0, 0, k, n, nr)

	// Expected packed layout: [panel, k, nr]
	// Panel 0: cols 0-1 -> for k=0: [1,2], k=1: [7,8], k=2: [13,14], k=3: [19,20]
	// Panel 1: cols 2-3 -> for k=0: [3,4], k=1: [9,10], k=2: [15,16], k=3: [21,22]
	// Panel 2: cols 4-5 -> for k=0: [5,6], k=1: [11,12], k=2: [17,18], k=3: [23,24]
	expected := []float32{
		1, 2, 7, 8, 13, 14, 19, 20, // Panel 0
		3, 4, 9, 10, 15, 16, 21, 22, // Panel 1
		5, 6, 11, 12, 17, 18, 23, 24, // Panel 2
	}

	if activeCols != nr {
		t.Errorf("activeCols = %d, want %d", activeCols, nr)
	}

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %f, want %f", i, packed[i], expected[i])
		}
	}
}

// TestPackedMatMul verifies packed matmul produces correct results.
func TestPackedMatMul(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{16, 32, 48, 64, 96, 128, 256}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			PackedMatMul(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			// Allow accumulated floating point error proportional to K
			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestPackedMatMulNonSquare verifies packed matmul with non-square matrices.
func TestPackedMatMulNonSquare(t *testing.T) {
	testCases := []struct {
		m, n, k int
	}{
		{64, 128, 32},
		{128, 64, 96},
		{100, 200, 150},
		{37, 53, 41}, // Odd sizes to test edge handling
		{256, 512, 128},
	}

	for _, tc := range testCases {
		name := sizeStr(tc.m) + "x" + sizeStr(tc.n) + "x" + sizeStr(tc.k)
		t.Run(name, func(t *testing.T) {
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.k*tc.n)
			c := make([]float32, tc.m*tc.n)
			expected := make([]float32, tc.m*tc.n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, tc.m, tc.n, tc.k)
			PackedMatMul(a, b, c, tc.m, tc.n, tc.k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%dx%d: max error %e", tc.m, tc.n, tc.k, maxErr)

			tolerance := float32(1e-4) * float32(tc.k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestParallelPackedMatMul verifies parallel packed matmul produces correct results.
func TestParallelPackedMatMul(t *testing.T) {
	sizes := []int{256, 512}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			ParallelPackedMatMul(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// BenchmarkPackedMatMul benchmarks the packed matmul.
func BenchmarkPackedMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkParallelPackedMatMul benchmarks the parallel packed matmul.
func BenchmarkParallelPackedMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelPackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkPackedVsBlocked compares packed and blocked matmul side-by-side.
func BenchmarkPackedVsBlocked(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Packed", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/ParallelBlocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/ParallelPacked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelPackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkAllAlgorithms compares all matmul algorithms.
func BenchmarkAllAlgorithms(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{64, 128, 256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size)+"/Streaming", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Packed", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Auto", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulAuto(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkPacking benchmarks the packing operations themselves.
func BenchmarkPacking(b *testing.B) {
	size := 512
	m, n, k := size, size, size

	a := make([]float32, m*k)
	bMat := make([]float32, k*n)

	for i := range a {
		a[i] = rand.Float32()
	}
	for i := range bMat {
		bMat[i] = rand.Float32()
	}

	params := getCacheParams[float32]()
	packedA := make([]float32, params.PackedASize())
	packedB := make([]float32, params.PackedBSize())

	b.Run("PackLHS", func(b *testing.B) {
		b.SetBytes(int64(params.Mc * params.Kc * 4))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			PackLHS(a, packedA, m, k, 0, 0, params.Mc, params.Kc, params.Mr)
		}
	})

	b.Run("PackRHS", func(b *testing.B) {
		b.SetBytes(int64(params.Nc * params.Kc * 4))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			PackRHS(bMat, packedB, k, n, 0, 0, params.Kc, params.Nc, params.Nr)
		}
	})
}
