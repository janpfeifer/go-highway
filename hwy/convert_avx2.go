//go:build amd64 && goexperiment.simd

package hwy

import (
	"math"
	"simd/archsimd"
)

// This file provides AVX2 SIMD implementations of type conversion operations.
// These work directly with archsimd vector types for maximum performance.

// ConvertToInt32_AVX2_F32x8 converts float32 to int32 using AVX2 VCVTTPS2DQ.
// Uses truncation toward zero (same as C-style float to int cast).
func ConvertToInt32_AVX2_F32x8(v archsimd.Float32x8) archsimd.Int32x8 {
	return v.ConvertToInt32()
}

// ConvertToFloat32_AVX2_I32x8 converts int32 to float32 using AVX2 VCVTDQ2PS.
func ConvertToFloat32_AVX2_I32x8(v archsimd.Int32x8) archsimd.Float32x8 {
	return v.ConvertToFloat32()
}

// BitCastF32ToI32_AVX2 reinterprets Float32x8 bits as Int32x8.
func BitCastF32ToI32_AVX2(v archsimd.Float32x8) archsimd.Int32x8 {
	return v.AsInt32x8()
}

// BitCastI32ToF32_AVX2 reinterprets Int32x8 bits as Float32x8.
func BitCastI32ToF32_AVX2(v archsimd.Int32x8) archsimd.Float32x8 {
	return v.AsFloat32x8()
}

// BitCastF64ToI64_AVX2 reinterprets Float64x4 bits as Int64x4.
func BitCastF64ToI64_AVX2(v archsimd.Float64x4) archsimd.Int64x4 {
	return v.AsInt64x4()
}

// BitCastI64ToF64_AVX2 reinterprets Int64x4 bits as Float64x4.
func BitCastI64ToF64_AVX2(v archsimd.Int64x4) archsimd.Float64x4 {
	return v.AsFloat64x4()
}

// Round_AVX2_F32x8 rounds to nearest integer using store/scalar/load pattern.
// AVX2 has VROUNDPS but archsimd may not expose it directly.
func Round_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Round(float64(data[i])))
	}
	return archsimd.LoadFloat32x8Slice(data[:])
}

// Round_AVX2_F64x4 rounds to nearest integer using store/scalar/load pattern.
func Round_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = math.Round(data[i])
	}
	return archsimd.LoadFloat64x4Slice(data[:])
}

// Trunc_AVX2_F32x8 truncates toward zero using store/scalar/load pattern.
func Trunc_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Trunc(float64(data[i])))
	}
	return archsimd.LoadFloat32x8Slice(data[:])
}

// Trunc_AVX2_F64x4 truncates toward zero using store/scalar/load pattern.
func Trunc_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = math.Trunc(data[i])
	}
	return archsimd.LoadFloat64x4Slice(data[:])
}

// Ceil_AVX2_F32x8 rounds up toward positive infinity.
func Ceil_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Ceil(float64(data[i])))
	}
	return archsimd.LoadFloat32x8Slice(data[:])
}

// Ceil_AVX2_F64x4 rounds up toward positive infinity.
func Ceil_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = math.Ceil(data[i])
	}
	return archsimd.LoadFloat64x4Slice(data[:])
}

// Floor_AVX2_F32x8 rounds down toward negative infinity.
func Floor_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = float32(math.Floor(float64(data[i])))
	}
	return archsimd.LoadFloat32x8Slice(data[:])
}

// Floor_AVX2_F64x4 rounds down toward negative infinity.
func Floor_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = math.Floor(data[i])
	}
	return archsimd.LoadFloat64x4Slice(data[:])
}

// NearestInt_AVX2_F32x8 rounds to nearest even integer.
func NearestInt_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8 {
	var data [8]float32
	v.StoreSlice(data[:])
	for i := 0; i < 8; i++ {
		data[i] = float32(math.RoundToEven(float64(data[i])))
	}
	return archsimd.LoadFloat32x8Slice(data[:])
}

// NearestInt_AVX2_F64x4 rounds to nearest even integer.
func NearestInt_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4 {
	var data [4]float64
	v.StoreSlice(data[:])
	for i := 0; i < 4; i++ {
		data[i] = math.RoundToEven(data[i])
	}
	return archsimd.LoadFloat64x4Slice(data[:])
}
