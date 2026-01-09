//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of compress and expand operations.
// These work directly with archsimd vector types for Float32x16 and Float64x8.

// Compress_AVX512_F32x16 compresses elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.MaskFloat32x16) (archsimd.Float32x16, int) {
	var data [16]float32
	v.StoreSlice(data[:])

	var result [16]float32
	count := 0

	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat32x16Slice(result[:]), count
}

// Compress_AVX512_F64x8 compresses elements where mask is true to the front.
// Returns compressed vector and count of valid elements.
func Compress_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.MaskFloat64x8) (archsimd.Float64x8, int) {
	var data [8]float64
	v.StoreSlice(data[:])

	var result [8]float64
	count := 0

	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			result[count] = data[i]
			count++
		}
	}

	return archsimd.LoadFloat64x8Slice(result[:]), count
}

// Expand_AVX512_F32x16 expands elements into positions where mask is true.
func Expand_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.MaskFloat32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])

	var result [16]float32
	srcIdx := 0

	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat32x16Slice(result[:])
}

// Expand_AVX512_F64x8 expands elements into positions where mask is true.
func Expand_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.MaskFloat64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])

	var result [8]float64
	srcIdx := 0

	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			result[i] = data[srcIdx]
			srcIdx++
		}
	}

	return archsimd.LoadFloat64x8Slice(result[:])
}

// CompressStore_AVX512_F32x16 compresses and stores directly to slice.
// Returns number of elements stored.
func CompressStore_AVX512_F32x16(v archsimd.Float32x16, mask archsimd.MaskFloat32x16, dst []float32) int {
	var data [16]float32
	v.StoreSlice(data[:])

	count := 0
	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
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
func CompressStore_AVX512_F64x8(v archsimd.Float64x8, mask archsimd.MaskFloat64x8, dst []float64) int {
	var data [8]float64
	v.StoreSlice(data[:])

	count := 0
	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			if count < len(dst) {
				dst[count] = data[i]
			}
			count++
		}
	}
	return count
}

// CountTrue_AVX512_F32x16 counts true lanes in mask.
func CountTrue_AVX512_F32x16(mask archsimd.MaskFloat32x16) int {
	count := 0
	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
			count++
		}
	}
	return count
}

// CountTrue_AVX512_F64x8 counts true lanes in mask.
func CountTrue_AVX512_F64x8(mask archsimd.MaskFloat64x8) int {
	count := 0
	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			count++
		}
	}
	return count
}

// AllTrue_AVX512_F32x16 returns true if all lanes are true.
func AllTrue_AVX512_F32x16(mask archsimd.MaskFloat32x16) bool {
	for i := 0; i < 16; i++ {
		if !getMaskBitF32x16(mask, i) {
			return false
		}
	}
	return true
}

// AllTrue_AVX512_F64x8 returns true if all lanes are true.
func AllTrue_AVX512_F64x8(mask archsimd.MaskFloat64x8) bool {
	for i := 0; i < 8; i++ {
		if !getMaskBitF64x8(mask, i) {
			return false
		}
	}
	return true
}

// AllFalse_AVX512_F32x16 returns true if all lanes are false.
func AllFalse_AVX512_F32x16(mask archsimd.MaskFloat32x16) bool {
	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
			return false
		}
	}
	return true
}

// AllFalse_AVX512_F64x8 returns true if all lanes are false.
func AllFalse_AVX512_F64x8(mask archsimd.MaskFloat64x8) bool {
	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			return false
		}
	}
	return true
}

// FindFirstTrue_AVX512_F32x16 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX512_F32x16(mask archsimd.MaskFloat32x16) int {
	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
			return i
		}
	}
	return -1
}

// FindFirstTrue_AVX512_F64x8 returns index of first true lane, or -1 if none.
func FindFirstTrue_AVX512_F64x8(mask archsimd.MaskFloat64x8) int {
	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX512_F32x16 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX512_F32x16(mask archsimd.MaskFloat32x16) int {
	for i := 15; i >= 0; i-- {
		if getMaskBitF32x16(mask, i) {
			return i
		}
	}
	return -1
}

// FindLastTrue_AVX512_F64x8 returns index of last true lane, or -1 if none.
func FindLastTrue_AVX512_F64x8(mask archsimd.MaskFloat64x8) int {
	for i := 7; i >= 0; i-- {
		if getMaskBitF64x8(mask, i) {
			return i
		}
	}
	return -1
}

// FirstN_AVX512_F32x16 creates a mask with the first n lanes set to true.
func FirstN_AVX512_F32x16(n int) archsimd.MaskFloat32x16 {
	if n <= 0 {
		return archsimd.MaskFloat32x16{}
	}
	if n >= 16 {
		allTrue := archsimd.BroadcastFloat32x16(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [16]float32
	for i := 0; i < n; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat32x16Slice(vals[:])
	ones := archsimd.BroadcastFloat32x16(1.0)
	return vec.Equal(ones)
}

// FirstN_AVX512_F64x8 creates a mask with the first n lanes set to true.
func FirstN_AVX512_F64x8(n int) archsimd.MaskFloat64x8 {
	if n <= 0 {
		return archsimd.MaskFloat64x8{}
	}
	if n >= 8 {
		allTrue := archsimd.BroadcastFloat64x8(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [8]float64
	for i := 0; i < n; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat64x8Slice(vals[:])
	ones := archsimd.BroadcastFloat64x8(1.0)
	return vec.Equal(ones)
}

// LastN_AVX512_F32x16 creates a mask with the last n lanes set to true.
func LastN_AVX512_F32x16(n int) archsimd.MaskFloat32x16 {
	if n <= 0 {
		return archsimd.MaskFloat32x16{}
	}
	if n >= 16 {
		allTrue := archsimd.BroadcastFloat32x16(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [16]float32
	start := 16 - n
	for i := start; i < 16; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat32x16Slice(vals[:])
	ones := archsimd.BroadcastFloat32x16(1.0)
	return vec.Equal(ones)
}

// LastN_AVX512_F64x8 creates a mask with the last n lanes set to true.
func LastN_AVX512_F64x8(n int) archsimd.MaskFloat64x8 {
	if n <= 0 {
		return archsimd.MaskFloat64x8{}
	}
	if n >= 8 {
		allTrue := archsimd.BroadcastFloat64x8(1.0)
		return allTrue.Equal(allTrue)
	}
	var vals [8]float64
	start := 8 - n
	for i := start; i < 8; i++ {
		vals[i] = 1.0
	}
	vec := archsimd.LoadFloat64x8Slice(vals[:])
	ones := archsimd.BroadcastFloat64x8(1.0)
	return vec.Equal(ones)
}

// Helper functions to extract mask bits

// getMaskBitF32x16 extracts whether bit i is set in the mask.
func getMaskBitF32x16(mask archsimd.MaskFloat32x16, i int) bool {
	if i < 0 || i >= 16 {
		return false
	}
	zeros := archsimd.Float32x16{}
	ones := archsimd.BroadcastFloat32x16(1.0)
	blended := mask.Blend(ones, zeros)
	var data [16]float32
	blended.StoreSlice(data[:])
	return data[i] == 1.0
}

// getMaskBitF64x8 extracts whether bit i is set in the mask.
func getMaskBitF64x8(mask archsimd.MaskFloat64x8, i int) bool {
	if i < 0 || i >= 8 {
		return false
	}
	zeros := archsimd.Float64x8{}
	ones := archsimd.BroadcastFloat64x8(1.0)
	blended := mask.Blend(ones, zeros)
	var data [8]float64
	blended.StoreSlice(data[:])
	return data[i] == 1.0
}

// MaskFromBits_AVX512_F32x16 creates a mask from a bitmask integer.
func MaskFromBits_AVX512_F32x16(bits uint64) archsimd.MaskFloat32x16 {
	var vals [16]float32
	for i := 0; i < 16; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1.0
		}
	}
	vec := archsimd.LoadFloat32x16Slice(vals[:])
	ones := archsimd.BroadcastFloat32x16(1.0)
	return vec.Equal(ones)
}

// MaskFromBits_AVX512_F64x8 creates a mask from a bitmask integer.
func MaskFromBits_AVX512_F64x8(bits uint64) archsimd.MaskFloat64x8 {
	var vals [8]float64
	for i := 0; i < 8; i++ {
		if (bits & (1 << i)) != 0 {
			vals[i] = 1.0
		}
	}
	vec := archsimd.LoadFloat64x8Slice(vals[:])
	ones := archsimd.BroadcastFloat64x8(1.0)
	return vec.Equal(ones)
}

// BitsFromMask_AVX512_F32x16 converts mask to bitmask integer.
func BitsFromMask_AVX512_F32x16(mask archsimd.MaskFloat32x16) uint64 {
	var result uint64
	for i := 0; i < 16; i++ {
		if getMaskBitF32x16(mask, i) {
			result |= 1 << i
		}
	}
	return result
}

// BitsFromMask_AVX512_F64x8 converts mask to bitmask integer.
func BitsFromMask_AVX512_F64x8(mask archsimd.MaskFloat64x8) uint64 {
	var result uint64
	for i := 0; i < 8; i++ {
		if getMaskBitF64x8(mask, i) {
			result |= 1 << i
		}
	}
	return result
}
