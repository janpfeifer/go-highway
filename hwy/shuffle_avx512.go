//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of shuffle operations.
// These work directly with archsimd vector types.

// Reverse_AVX512_F32x16 reverses all lanes using archsimd Reverse.
func Reverse_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	return v.Reverse()
}

// Reverse_AVX512_F64x8 reverses all lanes using archsimd Reverse.
func Reverse_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	return v.Reverse()
}

// Reverse2_AVX512_F32x16 reverses pairs of lanes.
func Reverse2_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i += 2 {
		data[i], data[i+1] = data[i+1], data[i]
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Reverse2_AVX512_F64x8 reverses pairs of lanes.
func Reverse2_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i += 2 {
		data[i], data[i+1] = data[i+1], data[i]
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}

// Reverse4_AVX512_F32x16 reverses groups of 4 lanes.
func Reverse4_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i += 4 {
		data[i], data[i+1], data[i+2], data[i+3] = data[i+3], data[i+2], data[i+1], data[i]
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Reverse4_AVX512_F64x8 reverses groups of 4 lanes.
func Reverse4_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i += 4 {
		data[i], data[i+1], data[i+2], data[i+3] = data[i+3], data[i+2], data[i+1], data[i]
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}

// Reverse8_AVX512_F32x16 reverses groups of 8 lanes.
func Reverse8_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i += 8 {
		for j := 0; j < 4; j++ {
			data[i+j], data[i+7-j] = data[i+7-j], data[i+j]
		}
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Reverse8_AVX512_F64x8 reverses all 8 lanes (same as Reverse for F64x8).
func Reverse8_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	return v.Reverse()
}

// GetLane_AVX512_F32x16 extracts a single lane value.
func GetLane_AVX512_F32x16(v archsimd.Float32x16, idx int) float32 {
	if idx < 0 || idx >= 16 {
		return 0
	}
	var data [16]float32
	v.StoreSlice(data[:])
	return data[idx]
}

// GetLane_AVX512_F64x8 extracts a single lane value.
func GetLane_AVX512_F64x8(v archsimd.Float64x8, idx int) float64 {
	if idx < 0 || idx >= 8 {
		return 0
	}
	var data [8]float64
	v.StoreSlice(data[:])
	return data[idx]
}

// InsertLane_AVX512_F32x16 inserts a value at the given lane.
func InsertLane_AVX512_F32x16(v archsimd.Float32x16, idx int, val float32) archsimd.Float32x16 {
	if idx < 0 || idx >= 16 {
		return v
	}
	var data [16]float32
	v.StoreSlice(data[:])
	data[idx] = val
	return archsimd.LoadFloat32x16Slice(data[:])
}

// InsertLane_AVX512_F64x8 inserts a value at the given lane.
func InsertLane_AVX512_F64x8(v archsimd.Float64x8, idx int, val float64) archsimd.Float64x8 {
	if idx < 0 || idx >= 8 {
		return v
	}
	var data [8]float64
	v.StoreSlice(data[:])
	data[idx] = val
	return archsimd.LoadFloat64x8Slice(data[:])
}

// InterleaveLower_AVX512_F32x16 interleaves lower halves.
func InterleaveLower_AVX512_F32x16(a, b archsimd.Float32x16) archsimd.Float32x16 {
	var dataA, dataB [16]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [16]float32
	for i := 0; i < 8; i++ {
		result[2*i] = dataA[i]
		result[2*i+1] = dataB[i]
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// InterleaveLower_AVX512_F64x8 interleaves lower halves.
func InterleaveLower_AVX512_F64x8(a, b archsimd.Float64x8) archsimd.Float64x8 {
	var dataA, dataB [8]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [8]float64
	for i := 0; i < 4; i++ {
		result[2*i] = dataA[i]
		result[2*i+1] = dataB[i]
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// InterleaveUpper_AVX512_F32x16 interleaves upper halves.
func InterleaveUpper_AVX512_F32x16(a, b archsimd.Float32x16) archsimd.Float32x16 {
	var dataA, dataB [16]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [16]float32
	for i := 0; i < 8; i++ {
		result[2*i] = dataA[8+i]
		result[2*i+1] = dataB[8+i]
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// InterleaveUpper_AVX512_F64x8 interleaves upper halves.
func InterleaveUpper_AVX512_F64x8(a, b archsimd.Float64x8) archsimd.Float64x8 {
	var dataA, dataB [8]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [8]float64
	for i := 0; i < 4; i++ {
		result[2*i] = dataA[4+i]
		result[2*i+1] = dataB[4+i]
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// ConcatLowerLower_AVX512_F32x16 concatenates lower halves.
func ConcatLowerLower_AVX512_F32x16(a, b archsimd.Float32x16) archsimd.Float32x16 {
	var dataA, dataB [16]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [16]float32
	copy(result[:8], dataA[:8])
	copy(result[8:], dataB[:8])
	return archsimd.LoadFloat32x16Slice(result[:])
}

// ConcatLowerLower_AVX512_F64x8 concatenates lower halves.
func ConcatLowerLower_AVX512_F64x8(a, b archsimd.Float64x8) archsimd.Float64x8 {
	var dataA, dataB [8]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [8]float64
	copy(result[:4], dataA[:4])
	copy(result[4:], dataB[:4])
	return archsimd.LoadFloat64x8Slice(result[:])
}

// OddEven_AVX512_F32x16 combines odd lanes from a with even lanes from b.
func OddEven_AVX512_F32x16(a, b archsimd.Float32x16) archsimd.Float32x16 {
	var dataA, dataB [16]float32
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [16]float32
	for i := 0; i < 16; i++ {
		if i%2 == 0 {
			result[i] = dataB[i]
		} else {
			result[i] = dataA[i]
		}
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// OddEven_AVX512_F64x8 combines odd lanes from a with even lanes from b.
func OddEven_AVX512_F64x8(a, b archsimd.Float64x8) archsimd.Float64x8 {
	var dataA, dataB [8]float64
	a.StoreSlice(dataA[:])
	b.StoreSlice(dataB[:])
	var result [8]float64
	for i := 0; i < 8; i++ {
		if i%2 == 0 {
			result[i] = dataB[i]
		} else {
			result[i] = dataA[i]
		}
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// DupEven_AVX512_F32x16 duplicates even lanes.
func DupEven_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	var result [16]float32
	for i := 0; i < 16; i += 2 {
		result[i] = data[i]
		result[i+1] = data[i]
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// DupEven_AVX512_F64x8 duplicates even lanes.
func DupEven_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	var result [8]float64
	for i := 0; i < 8; i += 2 {
		result[i] = data[i]
		result[i+1] = data[i]
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// DupOdd_AVX512_F32x16 duplicates odd lanes.
func DupOdd_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	var result [16]float32
	for i := 0; i < 16; i += 2 {
		result[i] = data[i+1]
		result[i+1] = data[i+1]
	}
	return archsimd.LoadFloat32x16Slice(result[:])
}

// DupOdd_AVX512_F64x8 duplicates odd lanes.
func DupOdd_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	var result [8]float64
	for i := 0; i < 8; i += 2 {
		result[i] = data[i+1]
		result[i+1] = data[i+1]
	}
	return archsimd.LoadFloat64x8Slice(result[:])
}

// SwapAdjacentBlocks_AVX512_F32x16 swaps adjacent 128-bit blocks.
// Each 128-bit block = 4 float32 lanes
func SwapAdjacentBlocks_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	// Swap blocks: [0-3] <-> [4-7], [8-11] <-> [12-15]
	var result [16]float32
	copy(result[0:4], data[4:8])
	copy(result[4:8], data[0:4])
	copy(result[8:12], data[12:16])
	copy(result[12:16], data[8:12])
	return archsimd.LoadFloat32x16Slice(result[:])
}

// SwapAdjacentBlocks_AVX512_F64x8 swaps adjacent 128-bit blocks.
// Each 128-bit block = 2 float64 lanes
func SwapAdjacentBlocks_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	// Swap blocks: [0-1] <-> [2-3], [4-5] <-> [6-7]
	var result [8]float64
	copy(result[0:2], data[2:4])
	copy(result[2:4], data[0:2])
	copy(result[4:6], data[6:8])
	copy(result[6:8], data[4:6])
	return archsimd.LoadFloat64x8Slice(result[:])
}
