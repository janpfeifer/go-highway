package hwy

import "math"

// This file provides pure Go (scalar) implementations of type conversion operations.
// When SIMD implementations are available (convert_avx2.go, convert_avx512.go),
// they can be used for higher performance on supported hardware.

// ConvertToInt32 converts float32 or float64 to int32 (truncate toward zero).
// For values outside the int32 range, the result is undefined.
func ConvertToInt32[T ~float32 | ~float64](v Vec[T]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int32(v.data[i])
	}
	return Vec[int32]{data: result}
}

// ConvertToFloat32 converts int32 or int64 to float32.
// Large int64 values may lose precision.
func ConvertToFloat32[T ~int32 | ~int64](v Vec[T]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float32(v.data[i])
	}
	return Vec[float32]{data: result}
}

// ConvertToInt64 converts float64 to int64 (truncate toward zero).
// For values outside the int64 range, the result is undefined.
func ConvertToInt64[T ~float64](v Vec[T]) Vec[int64] {
	result := make([]int64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int64(v.data[i])
	}
	return Vec[int64]{data: result}
}

// ConvertToFloat64 converts int32 or int64 to float64.
// Large int64 values may lose precision.
func ConvertToFloat64[T ~int32 | ~int64](v Vec[T]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = float64(v.data[i])
	}
	return Vec[float64]{data: result}
}

// Round rounds each lane to the nearest integer (using banker's rounding).
func Round[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = T(math.Round(float64(v.data[i])))
	}
	return Vec[T]{data: result}
}

// Trunc truncates each lane toward zero.
func Trunc[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = T(math.Trunc(float64(v.data[i])))
	}
	return Vec[T]{data: result}
}

// Ceil rounds each lane up (toward positive infinity).
func Ceil[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = T(math.Ceil(float64(v.data[i])))
	}
	return Vec[T]{data: result}
}

// Floor rounds each lane down (toward negative infinity).
func Floor[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = T(math.Floor(float64(v.data[i])))
	}
	return Vec[T]{data: result}
}

// NearestInt rounds each lane to the nearest integer and returns the same type.
// This is similar to Round but ensures the result is an integral value
// stored in the floating-point type.
func NearestInt[T Floats](v Vec[T]) Vec[T] {
	result := make([]T, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = T(math.RoundToEven(float64(v.data[i])))
	}
	return Vec[T]{data: result}
}

// BitCastF32ToI32 reinterprets float32 bits as int32 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastF32ToI32(v Vec[float32]) Vec[int32] {
	result := make([]int32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int32(math.Float32bits(v.data[i]))
	}
	return Vec[int32]{data: result}
}

// BitCastI32ToF32 reinterprets int32 bits as float32 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastI32ToF32(v Vec[int32]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float32frombits(uint32(v.data[i]))
	}
	return Vec[float32]{data: result}
}

// BitCastF64ToI64 reinterprets float64 bits as int64 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastF64ToI64(v Vec[float64]) Vec[int64] {
	result := make([]int64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = int64(math.Float64bits(v.data[i]))
	}
	return Vec[int64]{data: result}
}

// BitCastI64ToF64 reinterprets int64 bits as float64 without conversion.
// This is a bitwise reinterpretation, not a numeric conversion.
func BitCastI64ToF64(v Vec[int64]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float64frombits(uint64(v.data[i]))
	}
	return Vec[float64]{data: result}
}

// BitCastU32ToF32 reinterprets uint32 bits as float32 without conversion.
func BitCastU32ToF32(v Vec[uint32]) Vec[float32] {
	result := make([]float32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float32frombits(v.data[i])
	}
	return Vec[float32]{data: result}
}

// BitCastF32ToU32 reinterprets float32 bits as uint32 without conversion.
func BitCastF32ToU32(v Vec[float32]) Vec[uint32] {
	result := make([]uint32, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float32bits(v.data[i])
	}
	return Vec[uint32]{data: result}
}

// BitCastU64ToF64 reinterprets uint64 bits as float64 without conversion.
func BitCastU64ToF64(v Vec[uint64]) Vec[float64] {
	result := make([]float64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float64frombits(v.data[i])
	}
	return Vec[float64]{data: result}
}

// BitCastF64ToU64 reinterprets float64 bits as uint64 without conversion.
func BitCastF64ToU64(v Vec[float64]) Vec[uint64] {
	result := make([]uint64, len(v.data))
	for i := 0; i < len(v.data); i++ {
		result[i] = math.Float64bits(v.data[i])
	}
	return Vec[uint64]{data: result}
}
