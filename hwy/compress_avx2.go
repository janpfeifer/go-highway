//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of compress and expand operations.
// These work directly with archsimd vector types.
// Since archsimd doesn't have vcompressps, we use store/scalar/load pattern.
//
// Note: archsimd doesn't have explicit mask types. Masks are represented as
// integer vectors (Int32x8, Int64x4) where each lane is all-1s (true) or
// all-0s (false). Comparison operations like Equal() return these masks.

// Compress_AVX2_F32x8 compresses elements where mask lane is non-zero to the front.
// The mask should be the result of a comparison operation (e.g., Equal, Less).
// Returns compressed vector and count of valid elements.
func Compress_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.Int32x8) (archsimd.Float32x8, int) {
	var data [8]float32
	v.StoreSlice(data[:])

	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	var result [8]float32
	count := 0

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat32x8Slice(result[:]), count
}

// Compress_AVX2_F64x4 compresses elements where mask lane is non-zero to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.Int64x4) (archsimd.Float64x4, int) {
	var data [4]float64
	v.StoreSlice(data[:])

	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	var result [4]float64
	count := 0

	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat64x4Slice(result[:]), count
}

// Expand_AVX2_F32x8 expands elements into positions where mask lane is non-zero.
func Expand_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.Int32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])

	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	var result [8]float32
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
		// else: leave as zero
	}

	return archsimd.LoadFloat32x8Slice(result[:])
}

// Expand_AVX2_F64x4 expands elements into positions where mask lane is non-zero.
func Expand_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.Int64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])

	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	var result [4]float64
	srcIdx := 0

	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat64x4Slice(result[:])
}

// CompressStore_AVX2_F32x8 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.Int32x8, dst []float32) int {
	var data [8]float32
	v.StoreSlice(data[:])

	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX2_F64x4 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.Int64x4, dst []float64) int {
	var data [4]float64
	v.StoreSlice(data[:])

	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue_AVX2_F32x8 counts true (non-zero) lanes in mask.
func CountTrue_AVX2_F32x8(mask archsimd.Int32x8) int {
	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			count++
		}
	}
	return count
}

// CountTrue_AVX2_F64x4 counts true (non-zero) lanes in mask.
func CountTrue_AVX2_F64x4(mask archsimd.Int64x4) int {
	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			count++
		}
	}
	return count
}

// AllTrue_AVX2_F32x8 returns true if all lanes are true (non-zero).
func AllTrue_AVX2_F32x8(mask archsimd.Int32x8) bool {
	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] == 0 {
			return false
		}
	}
	return true
}

// AllTrue_AVX2_F64x4 returns true if all lanes are true (non-zero).
func AllTrue_AVX2_F64x4(mask archsimd.Int64x4) bool {
	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 4; i++ {
		if maskData[i] == 0 {
			return false
		}
	}
	return true
}

// AllFalse_AVX2_F32x8 returns true if all lanes are false (zero).
func AllFalse_AVX2_F32x8(mask archsimd.Int32x8) bool {
	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			return false
		}
	}
	return true
}

// AllFalse_AVX2_F64x4 returns true if all lanes are false (zero).
func AllFalse_AVX2_F64x4(mask archsimd.Int64x4) bool {
	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			return false
		}
	}
	return true
}

// FindFirstTrue_AVX2_F32x8 returns index of first true (non-zero) lane, or -1 if none.
func FindFirstTrue_AVX2_F32x8(mask archsimd.Int32x8) int {
	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FindFirstTrue_AVX2_F64x4 returns index of first true (non-zero) lane, or -1 if none.
func FindFirstTrue_AVX2_F64x4(mask archsimd.Int64x4) int {
	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX2_F32x8 returns index of last true (non-zero) lane, or -1 if none.
func FindLastTrue_AVX2_F32x8(mask archsimd.Int32x8) int {
	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	for i := 7; i >= 0; i-- {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX2_F64x4 returns index of last true (non-zero) lane, or -1 if none.
func FindLastTrue_AVX2_F64x4(mask archsimd.Int64x4) int {
	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	for i := 3; i >= 0; i-- {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FirstN_AVX2_F32x8 creates a mask with the first n lanes set to true.
func FirstN_AVX2_F32x8(n int) archsimd.Int32x8 {
	var vals [8]int32
	for i := 0; i < 8 && i < n; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt32x8Slice(vals[:])
}

// FirstN_AVX2_F64x4 creates a mask with the first n lanes set to true.
func FirstN_AVX2_F64x4(n int) archsimd.Int64x4 {
	var vals [4]int64
	for i := 0; i < 4 && i < n; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt64x4Slice(vals[:])
}

// LastN_AVX2_F32x8 creates a mask with the last n lanes set to true.
func LastN_AVX2_F32x8(n int) archsimd.Int32x8 {
	var vals [8]int32
	start := 8 - n
	if start < 0 {
		start = 0
	}
	for i := start; i < 8; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt32x8Slice(vals[:])
}

// LastN_AVX2_F64x4 creates a mask with the last n lanes set to true.
func LastN_AVX2_F64x4(n int) archsimd.Int64x4 {
	var vals [4]int64
	start := 4 - n
	if start < 0 {
		start = 0
	}
	for i := start; i < 4; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt64x4Slice(vals[:])
}

// MaskFromBits_AVX2_F32x8 creates a mask from a bitmask integer.
func MaskFromBits_AVX2_F32x8(bits uint64) archsimd.Int32x8 {
	var vals [8]int32
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = -1 // All 1s for true
		}
	}
	return archsimd.LoadInt32x8Slice(vals[:])
}

// MaskFromBits_AVX2_F64x4 creates a mask from a bitmask integer.
func MaskFromBits_AVX2_F64x4(bits uint64) archsimd.Int64x4 {
	var vals [4]int64
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = -1 // All 1s for true
		}
	}
	return archsimd.LoadInt64x4Slice(vals[:])
}

// BitsFromMask_AVX2_F32x8 converts mask to bitmask integer.
func BitsFromMask_AVX2_F32x8(mask archsimd.Int32x8) uint64 {
	var maskData [8]int32
	mask.StoreSlice(maskData[:])

	var result uint64
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_AVX2_F64x4 converts mask to bitmask integer.
func BitsFromMask_AVX2_F64x4(mask archsimd.Int64x4) uint64 {
	var maskData [4]int64
	mask.StoreSlice(maskData[:])

	var result uint64
	for i := 0; i < 4; i++ {
		if maskData[i] != 0 {
			result |= 1 << i
		}
	}
	return result
}
