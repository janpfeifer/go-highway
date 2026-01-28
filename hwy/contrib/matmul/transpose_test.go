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
	"fmt"
	"slices"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestTranspose2D(t *testing.T) {
	sizes := []struct{ m, k int }{
		{4, 4}, {8, 8}, {16, 16}, {32, 32},
		{64, 64}, {256, 256},
		{5, 7}, {17, 23}, {100, 200}, // Non-aligned
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size.m, size.k), func(t *testing.T) {
			src := make([]float32, size.m*size.k)
			for i := range src {
				src[i] = float32(i)
			}

			got := make([]float32, size.k*size.m)
			want := make([]float32, size.k*size.m)

			// Reference scalar transpose
			for i := 0; i < size.m; i++ {
				for j := 0; j < size.k; j++ {
					want[j*size.m+i] = src[i*size.k+j]
				}
			}

			Transpose2DFloat32(src, size.m, size.k, got)

			if !slices.Equal(got, want) {
				t.Errorf("mismatch at size %dx%d", size.m, size.k)
				// Print first difference
				for i := range got {
					if got[i] != want[i] {
						t.Errorf("first difference at index %d: got %v, want %v", i, got[i], want[i])
						break
					}
				}
			}
		})
	}
}

func TestTranspose2DFloat64(t *testing.T) {
	sizes := []struct{ m, k int }{
		{2, 2}, {4, 4}, {8, 8}, {16, 16},
		{64, 64}, {128, 128},
		{3, 5}, {11, 17}, // Non-aligned
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size.m, size.k), func(t *testing.T) {
			src := make([]float64, size.m*size.k)
			for i := range src {
				src[i] = float64(i)
			}

			got := make([]float64, size.k*size.m)
			want := make([]float64, size.k*size.m)

			// Reference scalar transpose
			for i := 0; i < size.m; i++ {
				for j := 0; j < size.k; j++ {
					want[j*size.m+i] = src[i*size.k+j]
				}
			}

			Transpose2DFloat64(src, size.m, size.k, got)

			if !slices.Equal(got, want) {
				t.Errorf("mismatch at size %dx%d", size.m, size.k)
			}
		})
	}
}

func TestTranspose2DFloat16(t *testing.T) {
	sizes := []struct{ m, k int }{
		{8, 8}, {16, 16}, {32, 32},
		{64, 64}, {128, 128},
		{5, 11}, {13, 19}, // Non-aligned
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size.m, size.k), func(t *testing.T) {
			src := make([]hwy.Float16, size.m*size.k)
			for i := range src {
				src[i] = hwy.Float32ToFloat16(float32(i))
			}

			got := make([]hwy.Float16, size.k*size.m)
			want := make([]hwy.Float16, size.k*size.m)

			// Reference scalar transpose
			for i := 0; i < size.m; i++ {
				for j := 0; j < size.k; j++ {
					want[j*size.m+i] = src[i*size.k+j]
				}
			}

			Transpose2DFloat16(src, size.m, size.k, got)

			if !slices.Equal(got, want) {
				t.Errorf("mismatch at size %dx%d", size.m, size.k)
			}
		})
	}
}

func TestTranspose2DBFloat16(t *testing.T) {
	sizes := []struct{ m, k int }{
		{8, 8}, {16, 16}, {32, 32},
		{64, 64}, {128, 128},
		{5, 11}, {13, 19}, // Non-aligned
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size.m, size.k), func(t *testing.T) {
			src := make([]hwy.BFloat16, size.m*size.k)
			for i := range src {
				src[i] = hwy.Float32ToBFloat16(float32(i))
			}

			got := make([]hwy.BFloat16, size.k*size.m)
			want := make([]hwy.BFloat16, size.k*size.m)

			// Reference scalar transpose
			for i := 0; i < size.m; i++ {
				for j := 0; j < size.k; j++ {
					want[j*size.m+i] = src[i*size.k+j]
				}
			}

			Transpose2DBFloat16(src, size.m, size.k, got)

			if !slices.Equal(got, want) {
				t.Errorf("mismatch at size %dx%d", size.m, size.k)
			}
		})
	}
}

func BenchmarkTranspose(b *testing.B) {
	for _, size := range []int{16, 32, 64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			src := make([]float32, size*size)
			dst := make([]float32, size*size)
			for i := range src {
				src[i] = float32(i)
			}
			b.SetBytes(int64(size * size * 4 * 2)) // read + write
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Transpose2DFloat32(src, size, size, dst)
			}
		})
	}
}

func BenchmarkTransposeFloat64(b *testing.B) {
	for _, size := range []int{64, 256, 512} {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			src := make([]float64, size*size)
			dst := make([]float64, size*size)
			for i := range src {
				src[i] = float64(i)
			}
			b.SetBytes(int64(size * size * 8 * 2)) // read + write
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Transpose2DFloat64(src, size, size, dst)
			}
		})
	}
}

func BenchmarkTransposeFloat16(b *testing.B) {
	for _, size := range []int{64, 256, 1024} {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			src := make([]hwy.Float16, size*size)
			dst := make([]hwy.Float16, size*size)
			for i := range src {
				src[i] = hwy.Float32ToFloat16(float32(i))
			}
			b.SetBytes(int64(size * size * 2 * 2)) // read + write
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Transpose2DFloat16(src, size, size, dst)
			}
		})
	}
}
