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

package wavelet

import (
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// Test sizes covering various boundary conditions
var testSizes = []int{2, 3, 4, 7, 8, 15, 16, 17, 32, 63, 64, 100}

func TestSynthesize53_RoundTrip(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				// Create original data
				original := make([]int32, size)
				for i := range original {
					original[i] = int32(i*7 - size/2)
				}

				// Copy for transform
				data := make([]int32, size)
				copy(data, original)

				// Forward then inverse should recover original
				Analyze53(data, phase)
				Synthesize53(data, phase)

				// Verify exact match (5/3 is lossless)
				for i := range original {
					if data[i] != original[i] {
						t.Errorf("at %d: got %d, want %d", i, data[i], original[i])
					}
				}
			})
		}
	}
}

func TestAnalyze53_RoundTrip(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				// Create original data in wavelet domain (low|high format)
				original := make([]int32, size)
				for i := range original {
					original[i] = int32(i*5 - size/3)
				}

				// Copy for transform
				data := make([]int32, size)
				copy(data, original)

				// Inverse then forward should recover original
				Synthesize53(data, phase)
				Analyze53(data, phase)

				// Verify exact match
				for i := range original {
					if data[i] != original[i] {
						t.Errorf("at %d: got %d, want %d", i, data[i], original[i])
					}
				}
			})
		}
	}
}

func TestSynthesize53Bufs_MatchesNonBufs(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				// Create identical data for both variants
				dataNonBufs := make([]int32, size)
				dataBufs := make([]int32, size)
				for i := range dataNonBufs {
					dataNonBufs[i] = int32(i*7 - size/2)
					dataBufs[i] = dataNonBufs[i]
				}

				// Run non-Bufs
				Synthesize53(dataNonBufs, phase)

				// Run Bufs with pre-allocated buffers
				maxHalf := (size + 1) / 2
				low := make([]int32, maxHalf)
				high := make([]int32, maxHalf)
				Synthesize53Bufs(dataBufs, phase, low, high)

				// Verify identical results
				for i := range dataNonBufs {
					if dataBufs[i] != dataNonBufs[i] {
						t.Errorf("at %d: Bufs got %d, non-Bufs got %d", i, dataBufs[i], dataNonBufs[i])
					}
				}
			})
		}
	}
}

func TestAnalyze53Bufs_MatchesNonBufs(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				dataNonBufs := make([]int32, size)
				dataBufs := make([]int32, size)
				for i := range dataNonBufs {
					dataNonBufs[i] = int32(i*5 - size/3)
					dataBufs[i] = dataNonBufs[i]
				}

				Analyze53(dataNonBufs, phase)

				maxHalf := (size + 1) / 2
				low := make([]int32, maxHalf)
				high := make([]int32, maxHalf)
				Analyze53Bufs(dataBufs, phase, low, high)

				for i := range dataNonBufs {
					if dataBufs[i] != dataNonBufs[i] {
						t.Errorf("at %d: Bufs got %d, non-Bufs got %d", i, dataBufs[i], dataNonBufs[i])
					}
				}
			})
		}
	}
}

func TestSynthesize53Bufs_RoundTrip(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				original := make([]int32, size)
				data := make([]int32, size)
				for i := range original {
					original[i] = int32(i*7 - size/2)
					data[i] = original[i]
				}

				maxHalf := (size + 1) / 2
				low := make([]int32, maxHalf)
				high := make([]int32, maxHalf)

				Analyze53Bufs(data, phase, low, high)
				Synthesize53Bufs(data, phase, low, high)

				for i := range original {
					if data[i] != original[i] {
						t.Errorf("at %d: got %d, want %d", i, data[i], original[i])
					}
				}
			})
		}
	}
}

func TestSynthesize53Core_MatchesSynthesize53(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				// Create identical wavelet-domain data for both paths
				dataRef := make([]int32, size)
				dataFused := make([]int32, size)
				for i := range dataRef {
					dataRef[i] = int32(i*11 - size/3)
					dataFused[i] = dataRef[i]
				}

				// Reference: scalar Synthesize53
				Synthesize53(dataRef, phase)

				// Fused: Synthesize53Bufs (now calls Synthesize53Core)
				maxHalf := (size + 1) / 2
				low := make([]int32, maxHalf)
				high := make([]int32, maxHalf)
				Synthesize53Bufs(dataFused, phase, low, high)

				for i := range dataRef {
					if dataFused[i] != dataRef[i] {
						t.Errorf("at %d: fused got %d, ref got %d", i, dataFused[i], dataRef[i])
					}
				}
			})
		}
	}
}

func TestSynthesize53CoreCols_MatchesSynthesize53(t *testing.T) {
	for _, height := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(height, phase), func(t *testing.T) {
				// Use MaxLanes columns to match the dispatched kernel width
				lanes := hwy.MaxLanes[int32]()

				// Generate lanes independent column signals in wavelet domain
				cols := make([][]int32, lanes)
				for c := range lanes {
					cols[c] = make([]int32, height)
					for y := range height {
						cols[c][y] = int32(y*7 + c*13 - height/2)
					}
				}

				// Reference: run scalar Synthesize53 on each column independently
				refCols := make([][]int32, lanes)
				for c := range lanes {
					refCols[c] = make([]int32, height)
					copy(refCols[c], cols[c])
					Synthesize53(refCols[c], phase)
				}

				// Column-interleaved buffer: colBuf[y*lanes + c]
				colBuf := make([]int32, height*lanes)
				for y := range height {
					for c := range lanes {
						colBuf[y*lanes+c] = cols[c][y]
					}
				}

				maxHalf := (height + 1) / 2
				lowBuf := make([]int32, maxHalf*lanes)
				highBuf := make([]int32, maxHalf*lanes)

				Synthesize53BufsCols(colBuf, height, phase, lowBuf, highBuf)

				// Verify each column matches reference
				for y := range height {
					for c := range lanes {
						got := colBuf[y*lanes+c]
						want := refCols[c][y]
						if got != want {
							t.Errorf("col %d row %d: got %d, want %d", c, y, got, want)
						}
					}
				}
			})
		}
	}
}

func TestSynthesize97_RoundTrip_Float32(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				// Create original data
				original := make([]float32, size)
				for i := range original {
					original[i] = float32(i)*0.1 - float32(size)/20.0
				}

				// Copy for transform
				data := make([]float32, size)
				copy(data, original)

				// Forward then inverse
				Analyze97(data, phase)
				Synthesize97(data, phase)

				// Verify approximate match (float32 precision)
				for i := range original {
					if !almostEqualF32(data[i], original[i], 1e-4) {
						t.Errorf("at %d: got %v, want %v (diff %v)", i, data[i], original[i], data[i]-original[i])
					}
				}
			})
		}
	}
}

func TestSynthesize97_RoundTrip_Float64(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				// Create original data
				original := make([]float64, size)
				for i := range original {
					original[i] = float64(i)*0.1 - float64(size)/20.0
				}

				// Copy for transform
				data := make([]float64, size)
				copy(data, original)

				// Forward then inverse
				Analyze97(data, phase)
				Synthesize97(data, phase)

				// Verify very close match (float64 precision)
				for i := range original {
					if !almostEqualF64(data[i], original[i], 1e-10) {
						t.Errorf("at %d: got %v, want %v (diff %v)", i, data[i], original[i], data[i]-original[i])
					}
				}
			})
		}
	}
}

func TestInterleaveDeinterleave(t *testing.T) {
	for _, size := range testSizes {
		for phase := 0; phase <= 1; phase++ {
			t.Run(sizePhaseString(size, phase), func(t *testing.T) {
				sn := (size + 1 - phase) / 2
				dn := size - sn

				// Create low and high
				low := make([]int32, sn)
				high := make([]int32, dn)
				for i := range low {
					low[i] = int32(i * 2)
				}
				for i := range high {
					high[i] = int32(i*2 + 1)
				}

				// Interleave
				dst := make([]int32, size)
				Interleave(dst, low, sn, high, dn, phase)

				// Deinterleave
				lowOut := make([]int32, sn)
				highOut := make([]int32, dn)
				Deinterleave(dst, lowOut, sn, highOut, dn, phase)

				// Verify match
				for i := range low {
					if lowOut[i] != low[i] {
						t.Errorf("low at %d: got %d, want %d", i, lowOut[i], low[i])
					}
				}
				for i := range high {
					if highOut[i] != high[i] {
						t.Errorf("high at %d: got %d, want %d", i, highOut[i], high[i])
					}
				}
			})
		}
	}
}

func TestScaleSlice(t *testing.T) {
	sizes := []int{1, 7, 8, 16, 17, 100}
	for _, size := range sizes {
		t.Run(sizeString(size), func(t *testing.T) {
			data := make([]float32, size)
			for i := range data {
				data[i] = float32(i) + 1.0
			}

			scale := float32(2.5)
			ScaleSlice(data, size, scale)

			for i := range data {
				expected := (float32(i) + 1.0) * scale
				if !almostEqualF32(data[i], expected, 1e-6) {
					t.Errorf("at %d: got %v, want %v", i, data[i], expected)
				}
			}
		})
	}
}

func TestSmallSizes(t *testing.T) {
	// Test edge cases with very small arrays
	t.Run("size1", func(t *testing.T) {
		data := []int32{42}
		Analyze53(data, 0)
		Synthesize53(data, 0)
		if data[0] != 42 {
			t.Errorf("size 1: got %d, want 42", data[0])
		}
	})

	t.Run("size2", func(t *testing.T) {
		original := []int32{10, 20}
		data := make([]int32, 2)
		copy(data, original)

		Analyze53(data, 0)
		Synthesize53(data, 0)

		for i := range original {
			if data[i] != original[i] {
				t.Errorf("size 2 at %d: got %d, want %d", i, data[i], original[i])
			}
		}
	})
}

func TestConstants(t *testing.T) {
	// Verify constants are correct
	if math.Abs(float64(K97*InvK97)-1.0) > 1e-10 {
		t.Errorf("K97 * InvK97 should equal 1, got %v", K97*InvK97)
	}

	// Verify coefficient signs for lifting steps
	if Alpha97 >= 0 {
		t.Errorf("Alpha97 should be negative, got %v", Alpha97)
	}
	if Beta97 >= 0 {
		t.Errorf("Beta97 should be negative, got %v", Beta97)
	}
	if Gamma97 <= 0 {
		t.Errorf("Gamma97 should be positive, got %v", Gamma97)
	}
	if Delta97 <= 0 {
		t.Errorf("Delta97 should be positive, got %v", Delta97)
	}
}

// Helper functions

func sizePhaseString(size, phase int) string {
	return sizeString(size) + "_phase" + string(rune('0'+phase))
}

func sizeString(size int) string {
	switch {
	case size < 4:
		return "tiny"
	case size < 8:
		return "small"
	case size == 8 || size == 16 || size == 32 || size == 64:
		return "aligned"
	case size < 100:
		return "medium"
	default:
		return "large"
	}
}

func almostEqualF32(a, b, tol float32) bool {
	return math.Abs(float64(a-b)) < float64(tol)
}

func almostEqualF64(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}
