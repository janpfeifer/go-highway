//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of compress and expand operations.
// These work directly with archsimd vector types for Float32x16 and Float64x8.
//
// Note: archsimd doesn't have explicit mask types. Masks are represented as
// integer vectors (Int32x16, Int64x8) where each lane is all-1s (true) or
// all-0s (false). Comparison operations like Equal() return these masks.

// Compress_AVX512_F32x16 compresses elements where mask lane is non-zero to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.Int32x16) (archsimd.Float32x16, int) {
	var data [16]float32
	v.StoreSlice(data[:])

	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	var result [16]float32
	count := 0

	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat32x16Slice(result[:]), count
}

// Compress_AVX512_F64x8 compresses elements where mask lane is non-zero to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.Int64x8) (archsimd.Float64x8, int) {
	var data [8]float64
	v.StoreSlice(data[:])

	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	var result [8]float64
	count := 0

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat64x8Slice(result[:]), count
}

// Expand_AVX512_F32x16 expands elements into positions where mask lane is non-zero.
func Expand_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.Int32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])

	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	var result [16]float32
	srcIdx := 0

	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat32x16Slice(result[:])
}

// Expand_AVX512_F64x8 expands elements into positions where mask lane is non-zero.
func Expand_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.Int64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])

	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	var result [8]float64
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat64x8Slice(result[:])
}

// CompressStore_AVX512_F32x16 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.Int32x16, dst []float32) int {
	var data [16]float32
	v.StoreSlice(data[:])

	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CompressStore_AVX512_F64x8 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.Int64x8, dst []float64) int {
	var data [8]float64
	v.StoreSlice(data[:])

	var maskData [8]int64
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

// CountTrue_AVX512_F32x16 counts true (non-zero) lanes in mask.
func CountTrue_AVX512_F32x16(mask archsimd.Int32x16) int {
	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			count++
		}
	}
	return count
}

// CountTrue_AVX512_F64x8 counts true (non-zero) lanes in mask.
func CountTrue_AVX512_F64x8(mask archsimd.Int64x8) int {
	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	count := 0
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			count++
		}
	}
	return count
}

// AllTrue_AVX512_F32x16 returns true if all lanes are true (non-zero).
func AllTrue_AVX512_F32x16(mask archsimd.Int32x16) bool {
	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 16; i++ {
		if maskData[i] == 0 {
			return false
		}
	}
	return true
}

// AllTrue_AVX512_F64x8 returns true if all lanes are true (non-zero).
func AllTrue_AVX512_F64x8(mask archsimd.Int64x8) bool {
	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] == 0 {
			return false
		}
	}
	return true
}

// AllFalse_AVX512_F32x16 returns true if all lanes are false (zero).
func AllFalse_AVX512_F32x16(mask archsimd.Int32x16) bool {
	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			return false
		}
	}
	return true
}

// AllFalse_AVX512_F64x8 returns true if all lanes are false (zero).
func AllFalse_AVX512_F64x8(mask archsimd.Int64x8) bool {
	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			return false
		}
	}
	return true
}

// FindFirstTrue_AVX512_F32x16 returns index of first true (non-zero) lane, or -1 if none.
func FindFirstTrue_AVX512_F32x16(mask archsimd.Int32x16) int {
	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FindFirstTrue_AVX512_F64x8 returns index of first true (non-zero) lane, or -1 if none.
func FindFirstTrue_AVX512_F64x8(mask archsimd.Int64x8) int {
	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX512_F32x16 returns index of last true (non-zero) lane, or -1 if none.
func FindLastTrue_AVX512_F32x16(mask archsimd.Int32x16) int {
	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	for i := 15; i >= 0; i-- {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX512_F64x8 returns index of last true (non-zero) lane, or -1 if none.
func FindLastTrue_AVX512_F64x8(mask archsimd.Int64x8) int {
	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	for i := 7; i >= 0; i-- {
		if maskData[i] != 0 {
			return i
		}
	}
	return -1
}

// FirstN_AVX512_F32x16 creates a mask with the first n lanes set to true.
func FirstN_AVX512_F32x16(n int) archsimd.Int32x16 {
	var vals [16]int32
	for i := 0; i < 16 && i < n; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt32x16Slice(vals[:])
}

// FirstN_AVX512_F64x8 creates a mask with the first n lanes set to true.
func FirstN_AVX512_F64x8(n int) archsimd.Int64x8 {
	var vals [8]int64
	for i := 0; i < 8 && i < n; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt64x8Slice(vals[:])
}

// LastN_AVX512_F32x16 creates a mask with the last n lanes set to true.
func LastN_AVX512_F32x16(n int) archsimd.Int32x16 {
	var vals [16]int32
	start := 16 - n
	if start < 0 {
		start = 0
	}
	for i := start; i < 16; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt32x16Slice(vals[:])
}

// LastN_AVX512_F64x8 creates a mask with the last n lanes set to true.
func LastN_AVX512_F64x8(n int) archsimd.Int64x8 {
	var vals [8]int64
	start := 8 - n
	if start < 0 {
		start = 0
	}
	for i := start; i < 8; i++ {
		vals[i] = -1 // All 1s for true
	}
	return archsimd.LoadInt64x8Slice(vals[:])
}

// MaskFromBits_AVX512_F32x16 creates a mask from a bitmask integer.
func MaskFromBits_AVX512_F32x16(bits uint64) archsimd.Int32x16 {
	var vals [16]int32
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = -1 // All 1s for true
		}
	}
	return archsimd.LoadInt32x16Slice(vals[:])
}

// MaskFromBits_AVX512_F64x8 creates a mask from a bitmask integer.
func MaskFromBits_AVX512_F64x8(bits uint64) archsimd.Int64x8 {
	var vals [8]int64
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = -1 // All 1s for true
		}
	}
	return archsimd.LoadInt64x8Slice(vals[:])
}

// BitsFromMask_AVX512_F32x16 converts mask to bitmask integer.
func BitsFromMask_AVX512_F32x16(mask archsimd.Int32x16) uint64 {
	var maskData [16]int32
	mask.StoreSlice(maskData[:])

	var result uint64
	for i := 0; i < 16; i++ {
		if maskData[i] != 0 {
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_AVX512_F64x8 converts mask to bitmask integer.
func BitsFromMask_AVX512_F64x8(mask archsimd.Int64x8) uint64 {
	var maskData [8]int64
	mask.StoreSlice(maskData[:])

	var result uint64
	for i := 0; i < 8; i++ {
		if maskData[i] != 0 {
			result |= 1 << i
		}
	}
	return result
}
