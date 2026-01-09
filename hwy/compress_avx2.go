//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of compress and expand operations.
// These work directly with archsimd vector types.
// Since archsimd doesn't have vcompressps, we use store/scalar/load pattern.

// Compress_AVX2_F32x8 compresses elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.MaskFloat32x8) (archsimd.Float32x8, int) {
	var data [8]float32
	v.StoreSlice(data[:])

	var result [8]float32
	count := 0

	// Extract mask bits - check each lane individually
	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat32x8Slice(result[:]), count
}

// Compress_AVX2_F64x4 compresses elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.MaskFloat64x4) (archsimd.Float64x4, int) {
	var data [4]float64
	v.StoreSlice(data[:])

	var result [4]float64
	count := 0

	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat64x4Slice(result[:]), count
}

// Expand_AVX2_F32x8 expands elements into positions where mask is true.
func Expand_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.MaskFloat32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])

	var result [8]float32
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
			result[i] = data[srcIdx]
			srcIdx++
		}
		// else: leave as zero
	}

	return archsimd.LoadFloat32x8Slice(result[:])
}

// Expand_AVX2_F64x4 expands elements into positions where mask is true.
func Expand_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.MaskFloat64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])

	var result [4]float64
	srcIdx := 0

	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat64x4Slice(result[:])
}

// CompressStore_AVX2_F32x8 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX2_F32x8(v archsimd.Float32x8, mask archsimd.MaskFloat32x8, dst []float32) int {
	var data [8]float32
	v.StoreSlice(data[:])

	count := 0
	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
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
func CompressStore_AVX2_F64x4(v archsimd.Float64x4, mask archsimd.MaskFloat64x4, dst []float64) int {
	var data [4]float64
	v.StoreSlice(data[:])

	count := 0
	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue_AVX2_F32x8 counts true lanes in mask.
func CountTrue_AVX2_F32x8(mask archsimd.MaskFloat32x8) int {
	count := 0
	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
			count++
		}
	}
	return count
}

// CountTrue_AVX2_F64x4 counts true lanes in mask.
func CountTrue_AVX2_F64x4(mask archsimd.MaskFloat64x4) int {
	count := 0
	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			count++
		}
	}
	return count
}

// AllTrue_AVX2_F32x8 returns true if all lanes are true.
func AllTrue_AVX2_F32x8(mask archsimd.MaskFloat32x8) bool {
	for i := 0; i < 8; i++ {
		if !getMaskBitF32x8(mask, i) {
			return false
		}
	}
	return true
}

// AllTrue_AVX2_F64x4 returns true if all lanes are true.
func AllTrue_AVX2_F64x4(mask archsimd.MaskFloat64x4) bool {
	for i := 0; i < 4; i++ {
		if !getMaskBitF64x4(mask, i) {
			return false
		}
	}
	return true
}

// AllFalse_AVX2_F32x8 returns true if all lanes are false.
func AllFalse_AVX2_F32x8(mask archsimd.MaskFloat32x8) bool {
	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
			return false
		}
	}
	return true
}

// AllFalse_AVX2_F64x4 returns true if all lanes are false.
func AllFalse_AVX2_F64x4(mask archsimd.MaskFloat64x4) bool {
	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			return false
		}
	}
	return true
}

// FindFirstTrue_AVX2_F32x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_F32x8(mask archsimd.MaskFloat32x8) int {
	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
			return i
		}
	}
	return -1
}

// FindFirstTrue_AVX2_F64x4 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX2_F64x4(mask archsimd.MaskFloat64x4) int {
	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX2_F32x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_F32x8(mask archsimd.MaskFloat32x8) int {
	for i := 7; i >= 0; i-- {
		if getMaskBitF32x8(mask, i) {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX2_F64x4 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX2_F64x4(mask archsimd.MaskFloat64x4) int {
	for i := 3; i >= 0; i-- {
		if getMaskBitF64x4(mask, i) {
			return i
		}
	}
	return -1
}

// FirstN_AVX2_F32x8 creates a mask with the first n lanes set to true.
func FirstN_AVX2_F32x8(n int) archsimd.MaskFloat32x8 {
	if n <= 0 {
		return archsimd.MaskFloat32x8{}
	}
	if n >= 8 {
		// All true mask
		allTrue := archsimd.BroadcastFloat32x8(1.0)
		return allTrue.Equal(allTrue)
	}
	// Create a mask by comparing with appropriate values
	// Build mask manually using comparison
	var vals [8]float32
	for i := 0; i < n; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat32x8Slice(vals[:])
	ones := archsimd.BroadcastFloat32x8(1.0)
	return vec.Equal(ones)
}

// FirstN_AVX2_F64x4 creates a mask with the first n lanes set to true.
func FirstN_AVX2_F64x4(n int) archsimd.MaskFloat64x4 {
	if n <= 0 {
		return archsimd.MaskFloat64x4{}
	}
	if n >= 4 {
		allTrue := archsimd.BroadcastFloat64x4(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [4]float64
	for i := 0; i < n; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat64x4Slice(vals[:])
	ones := archsimd.BroadcastFloat64x4(1.0)
	return vec.Equal(ones)
}

// LastN_AVX2_F32x8 creates a mask with the last n lanes set to true.
func LastN_AVX2_F32x8(n int) archsimd.MaskFloat32x8 {
	if n <= 0 {
		return archsimd.MaskFloat32x8{}
	}
	if n >= 8 {
		allTrue := archsimd.BroadcastFloat32x8(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [8]float32
	start := 8 - n
	for i := start; i < 8; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat32x8Slice(vals[:])
	ones := archsimd.BroadcastFloat32x8(1.0)
	return vec.Equal(ones)
}

// LastN_AVX2_F64x4 creates a mask with the last n lanes set to true.
func LastN_AVX2_F64x4(n int) archsimd.MaskFloat64x4 {
	if n <= 0 {
		return archsimd.MaskFloat64x4{}
	}
	if n >= 4 {
		allTrue := archsimd.BroadcastFloat64x4(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [4]float64
	start := 4 - n
	for i := start; i < 4; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat64x4Slice(vals[:])
	ones := archsimd.BroadcastFloat64x4(1.0)
	return vec.Equal(ones)
}

// Helper functions to extract mask bits by using blend operations

// getMaskBitF32x8 extracts whether bit i is set in the mask.
func getMaskBitF32x8(mask archsimd.MaskFloat32x8, i int) bool {
	if i < 0 || i >= 8 {
		return false
	}
	// Use Blend to test the mask: if bit i is set, result[i] will be 1.0
	zeros := archsimd.Float32x8{}
	ones := archsimd.BroadcastFloat32x8(1.0)
	blended := mask.Blend(ones, zeros)
	var data [8]float32
	blended.StoreSlice(data[:])
	return data[i] == 1.0
}

// getMaskBitF64x4 extracts whether bit i is set in the mask.
func getMaskBitF64x4(mask archsimd.MaskFloat64x4, i int) bool {
	if i < 0 || i >= 4 {
		return false
	}
	zeros := archsimd.Float64x4{}
	ones := archsimd.BroadcastFloat64x4(1.0)
	blended := mask.Blend(ones, zeros)
	var data [4]float64
	blended.StoreSlice(data[:])
	return data[i] == 1.0
}

// MaskFromBits_AVX2_F32x8 creates a mask from a bitmask integer.
func MaskFromBits_AVX2_F32x8(bits uint64) archsimd.MaskFloat32x8 {
	var vals [8]float32
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1.0
		}
	}
	vec := archsimd.LoadFloat32x8Slice(vals[:])
	ones := archsimd.BroadcastFloat32x8(1.0)
	return vec.Equal(ones)
}

// MaskFromBits_AVX2_F64x4 creates a mask from a bitmask integer.
func MaskFromBits_AVX2_F64x4(bits uint64) archsimd.MaskFloat64x4 {
	var vals [4]float64
	for i := 0; i < 4; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1.0
		}
	}
	vec := archsimd.LoadFloat64x4Slice(vals[:])
	ones := archsimd.BroadcastFloat64x4(1.0)
	return vec.Equal(ones)
}

// BitsFromMask_AVX2_F32x8 converts mask to bitmask integer.
func BitsFromMask_AVX2_F32x8(mask archsimd.MaskFloat32x8) uint64 {
	var result uint64
	for i := 0; i < 8; i++ {
		if getMaskBitF32x8(mask, i) {
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_AVX2_F64x4 converts mask to bitmask integer.
func BitsFromMask_AVX2_F64x4(mask archsimd.MaskFloat64x4) uint64 {
	var result uint64
	for i := 0; i < 4; i++ {
		if getMaskBitF64x4(mask, i) {
			result |= 1 << i
		}
	}
	return result
}
