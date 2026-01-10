//go:build amd64 && goexperiment.simd

package hwy

import (
	"math"
	"simd/archsimd"
)

// This file provides AVX-512 SIMD implementations of type conversion operations.
// These work directly with archsimd vector types for maximum performance.

// ConvertToInt32_AVX512_F32x16 converts float32 to int32 using AVX-512 VCVTTPS2DQ.
// Uses truncation toward zero (same as C-style float to int cast).
func ConvertToInt32_AVX512_F32x16(v archsimd.Float32x16) archsimd.Int32x16 {
	return v.ConvertToInt32()
}

// ConvertToFloat32_AVX512_I32x16 converts int32 to float32 using AVX-512 VCVTDQ2PS.
func ConvertToFloat32_AVX512_I32x16(v archsimd.Int32x16) archsimd.Float32x16 {
	return v.ConvertToFloat32()
}

// ConvertToInt64_AVX512_F64x8 converts float64 to int64 using AVX-512.
func ConvertToInt64_AVX512_F64x8(v archsimd.Float64x8) archsimd.Int64x8 {
	return v.ConvertToInt64()
}

// ConvertToFloat64_AVX512_I64x8 converts int64 to float64 using AVX-512.
func ConvertToFloat64_AVX512_I64x8(v archsimd.Int64x8) archsimd.Float64x8 {
	return v.ConvertToFloat64()
}

// BitCastF32ToI32_AVX512 reinterprets Float32x16 bits as Int32x16.
func BitCastF32ToI32_AVX512(v archsimd.Float32x16) archsimd.Int32x16 {
	return v.AsInt32x16()
}

// BitCastI32ToF32_AVX512 reinterprets Int32x16 bits as Float32x16.
func BitCastI32ToF32_AVX512(v archsimd.Int32x16) archsimd.Float32x16 {
	return v.AsFloat32x16()
}

// BitCastF64ToI64_AVX512 reinterprets Float64x8 bits as Int64x8.
func BitCastF64ToI64_AVX512(v archsimd.Float64x8) archsimd.Int64x8 {
	return v.AsInt64x8()
}

// BitCastI64ToF64_AVX512 reinterprets Int64x8 bits as Float64x8.
func BitCastI64ToF64_AVX512(v archsimd.Int64x8) archsimd.Float64x8 {
	return v.AsFloat64x8()
}

// Round_AVX512_F32x16 rounds to nearest integer.
func Round_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Round(float64(data[i])))
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Round_AVX512_F64x8 rounds to nearest integer.
func Round_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = math.Round(data[i])
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}

// Trunc_AVX512_F32x16 truncates toward zero.
func Trunc_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Trunc(float64(data[i])))
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Trunc_AVX512_F64x8 truncates toward zero.
func Trunc_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = math.Trunc(data[i])
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}

// Ceil_AVX512_F32x16 rounds up toward positive infinity.
func Ceil_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Ceil(float64(data[i])))
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Ceil_AVX512_F64x8 rounds up toward positive infinity.
func Ceil_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = math.Ceil(data[i])
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}

// Floor_AVX512_F32x16 rounds down toward negative infinity.
func Floor_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = float32(math.Floor(float64(data[i])))
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// Floor_AVX512_F64x8 rounds down toward negative infinity.
func Floor_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = math.Floor(data[i])
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}

// NearestInt_AVX512_F32x16 rounds to nearest even integer.
func NearestInt_AVX512_F32x16(v archsimd.Float32x16) archsimd.Float32x16 {
	var data [16]float32
	v.StoreSlice(data[:])
	for i := 0; i < 16; i++ {
		data[i] = float32(math.RoundToEven(float64(data[i])))
	}
	return archsimd.LoadFloat32x16Slice(data[:])
}

// NearestInt_AVX512_F64x8 rounds to nearest even integer.
func NearestInt_AVX512_F64x8(v archsimd.Float64x8) archsimd.Float64x8 {
	var data [8]float64
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = math.RoundToEven(data[i])
	}
	return archsimd.LoadFloat64x8Slice(data[:])
}
