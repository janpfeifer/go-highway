//go:build !noasm && arm64

package asm

import (
	"math"
)

// Float32x4 represents a 128-bit NEON vector of 4 float32 values.
// This provides an API similar to archsimd.Float32x4 for NEON.
type Float32x4 [4]float32

// Float64x2 represents a 128-bit NEON vector of 2 float64 values.
// This provides an API similar to archsimd.Float64x2 for NEON.
type Float64x2 [2]float64

// Int32x4 represents a 128-bit NEON vector of 4 int32 values.
type Int32x4 [4]int32

// Int64x2 represents a 128-bit NEON vector of 2 int64 values.
type Int64x2 [2]int64

// ===== Float32x4 constructors =====

// BroadcastFloat32x4 creates a vector with all lanes set to the given value.
func BroadcastFloat32x4(v float32) Float32x4 {
	return Float32x4{v, v, v, v}
}

// LoadFloat32x4 loads 4 float32 values from a slice.
func LoadFloat32x4(s []float32) Float32x4 {
	var v Float32x4
	copy(v[:], s[:4])
	return v
}

// LoadFloat32x4Slice is an alias for LoadFloat32x4 (matches archsimd naming).
func LoadFloat32x4Slice(s []float32) Float32x4 {
	return LoadFloat32x4(s)
}

// ZeroFloat32x4 returns a zero vector.
func ZeroFloat32x4() Float32x4 {
	return Float32x4{}
}

// ===== Float32x4 methods =====

// StoreSlice stores the vector to a slice.
func (v Float32x4) StoreSlice(s []float32) {
	copy(s[:4], v[:])
}

// Add performs element-wise addition.
func (v Float32x4) Add(other Float32x4) Float32x4 {
	var result Float32x4
	AddF32(v[:], other[:], result[:])
	return result
}

// Sub performs element-wise subtraction.
func (v Float32x4) Sub(other Float32x4) Float32x4 {
	var result Float32x4
	SubF32(v[:], other[:], result[:])
	return result
}

// Mul performs element-wise multiplication.
func (v Float32x4) Mul(other Float32x4) Float32x4 {
	var result Float32x4
	MulF32(v[:], other[:], result[:])
	return result
}

// Div performs element-wise division.
func (v Float32x4) Div(other Float32x4) Float32x4 {
	var result Float32x4
	DivF32(v[:], other[:], result[:])
	return result
}

// Min performs element-wise minimum.
func (v Float32x4) Min(other Float32x4) Float32x4 {
	var result Float32x4
	MinF32(v[:], other[:], result[:])
	return result
}

// Max performs element-wise maximum.
func (v Float32x4) Max(other Float32x4) Float32x4 {
	var result Float32x4
	MaxF32(v[:], other[:], result[:])
	return result
}

// Sqrt performs element-wise square root.
func (v Float32x4) Sqrt() Float32x4 {
	var result Float32x4
	SqrtF32(v[:], result[:])
	return result
}

// Abs performs element-wise absolute value.
func (v Float32x4) Abs() Float32x4 {
	var result Float32x4
	AbsF32(v[:], result[:])
	return result
}

// Neg performs element-wise negation.
func (v Float32x4) Neg() Float32x4 {
	var result Float32x4
	NegF32(v[:], result[:])
	return result
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float32x4) MulAdd(a, b Float32x4) Float32x4 {
	var result Float32x4
	FmaF32(v[:], a[:], b[:], result[:])
	return result
}

// ReduceSum returns the sum of all elements.
func (v Float32x4) ReduceSum() float32 {
	return ReduceSumF32(v[:])
}

// ReduceMin returns the minimum element.
func (v Float32x4) ReduceMin() float32 {
	return ReduceMinF32(v[:])
}

// ReduceMax returns the maximum element.
func (v Float32x4) ReduceMax() float32 {
	return ReduceMaxF32(v[:])
}

// Greater returns a mask where v > other.
func (v Float32x4) Greater(other Float32x4) Int32x4 {
	var result Int32x4
	GtF32(v[:], other[:], result[:])
	return result
}

// Less returns a mask where v < other.
func (v Float32x4) Less(other Float32x4) Int32x4 {
	var result Int32x4
	LtF32(v[:], other[:], result[:])
	return result
}

// GreaterEqual returns a mask where v >= other.
func (v Float32x4) GreaterEqual(other Float32x4) Int32x4 {
	var result Int32x4
	GeF32(v[:], other[:], result[:])
	return result
}

// LessEqual returns a mask where v <= other.
func (v Float32x4) LessEqual(other Float32x4) Int32x4 {
	var result Int32x4
	LeF32(v[:], other[:], result[:])
	return result
}

// Equal returns a mask where v == other.
func (v Float32x4) Equal(other Float32x4) Int32x4 {
	var result Int32x4
	EqF32(v[:], other[:], result[:])
	return result
}

// Merge selects elements: mask ? v : other
// Note: mask values should be all-ones (-1) for true, all-zeros (0) for false.
func (v Float32x4) Merge(other Float32x4, mask Int32x4) Float32x4 {
	var result Float32x4
	IfThenElseF32(mask[:], v[:], other[:], result[:])
	return result
}

// RoundToEven rounds to nearest even.
func (v Float32x4) RoundToEven() Float32x4 {
	var result Float32x4
	RoundF32(v[:], result[:])
	return result
}

// ConvertToInt32 converts to int32.
func (v Float32x4) ConvertToInt32() Int32x4 {
	var result Int32x4
	ConvertF32ToI32(v[:], result[:])
	return result
}

// ===== Float64x2 constructors =====

// BroadcastFloat64x2 creates a vector with all lanes set to the given value.
func BroadcastFloat64x2(v float64) Float64x2 {
	return Float64x2{v, v}
}

// LoadFloat64x2 loads 2 float64 values from a slice.
func LoadFloat64x2(s []float64) Float64x2 {
	var v Float64x2
	copy(v[:], s[:2])
	return v
}

// LoadFloat64x2Slice is an alias for LoadFloat64x2 (matches archsimd naming).
func LoadFloat64x2Slice(s []float64) Float64x2 {
	return LoadFloat64x2(s)
}

// ZeroFloat64x2 returns a zero vector.
func ZeroFloat64x2() Float64x2 {
	return Float64x2{}
}

// ===== Float64x2 methods =====

// StoreSlice stores the vector to a slice.
func (v Float64x2) StoreSlice(s []float64) {
	copy(s[:2], v[:])
}

// Add performs element-wise addition.
func (v Float64x2) Add(other Float64x2) Float64x2 {
	var result Float64x2
	AddF64(v[:], other[:], result[:])
	return result
}

// Sub performs element-wise subtraction.
func (v Float64x2) Sub(other Float64x2) Float64x2 {
	var result Float64x2
	SubF64(v[:], other[:], result[:])
	return result
}

// Mul performs element-wise multiplication.
func (v Float64x2) Mul(other Float64x2) Float64x2 {
	var result Float64x2
	MulF64(v[:], other[:], result[:])
	return result
}

// Div performs element-wise division.
func (v Float64x2) Div(other Float64x2) Float64x2 {
	var result Float64x2
	DivF64(v[:], other[:], result[:])
	return result
}

// Min performs element-wise minimum.
func (v Float64x2) Min(other Float64x2) Float64x2 {
	var result Float64x2
	MinF64(v[:], other[:], result[:])
	return result
}

// Max performs element-wise maximum.
func (v Float64x2) Max(other Float64x2) Float64x2 {
	var result Float64x2
	MaxF64(v[:], other[:], result[:])
	return result
}

// Sqrt performs element-wise square root.
func (v Float64x2) Sqrt() Float64x2 {
	var result Float64x2
	SqrtF64(v[:], result[:])
	return result
}

// Abs performs element-wise absolute value.
func (v Float64x2) Abs() Float64x2 {
	var result Float64x2
	AbsF64(v[:], result[:])
	return result
}

// Neg performs element-wise negation.
func (v Float64x2) Neg() Float64x2 {
	var result Float64x2
	NegF64(v[:], result[:])
	return result
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float64x2) MulAdd(a, b Float64x2) Float64x2 {
	var result Float64x2
	FmaF64(v[:], a[:], b[:], result[:])
	return result
}

// ReduceSum returns the sum of all elements.
func (v Float64x2) ReduceSum() float64 {
	return ReduceSumF64(v[:])
}

// ReduceMin returns the minimum element.
func (v Float64x2) ReduceMin() float64 {
	return ReduceMinF64(v[:])
}

// ReduceMax returns the maximum element.
func (v Float64x2) ReduceMax() float64 {
	return ReduceMaxF64(v[:])
}

// Greater returns a mask where v > other.
// Note: Returns Int64x2 mask for Float64x2 comparisons.
func (v Float64x2) Greater(other Float64x2) Int64x2 {
	var result Int64x2
	// Use scalar comparison since we don't have F64 comparison in asm yet
	for i := 0; i < 2; i++ {
		if v[i] > other[i] {
			result[i] = -1
		}
	}
	return result
}

// Less returns a mask where v < other.
func (v Float64x2) Less(other Float64x2) Int64x2 {
	var result Int64x2
	for i := 0; i < 2; i++ {
		if v[i] < other[i] {
			result[i] = -1
		}
	}
	return result
}

// Merge selects elements: mask ? v : other
func (v Float64x2) Merge(other Float64x2, mask Int64x2) Float64x2 {
	var result Float64x2
	for i := 0; i < 2; i++ {
		if mask[i] != 0 {
			result[i] = v[i]
		} else {
			result[i] = other[i]
		}
	}
	return result
}

// RoundToEven rounds to nearest even.
func (v Float64x2) RoundToEven() Float64x2 {
	// Use scalar rounding for now
	var result Float64x2
	for i := 0; i < 2; i++ {
		result[i] = float64(int64(v[i] + 0.5))
	}
	return result
}

// ===== Int32x4 constructors and methods =====

// BroadcastInt32x4 creates a vector with all lanes set to the given value.
func BroadcastInt32x4(v int32) Int32x4 {
	return Int32x4{v, v, v, v}
}

// Add performs element-wise addition.
func (v Int32x4) Add(other Int32x4) Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		result[i] = v[i] + other[i]
	}
	return result
}

// ShiftAllLeft shifts all elements left by the given count.
func (v Int32x4) ShiftAllLeft(count int) Int32x4 {
	var result Int32x4
	for i := 0; i < 4; i++ {
		result[i] = v[i] << count
	}
	return result
}

// AsFloat32x4 reinterprets bits as float32.
func (v Int32x4) AsFloat32x4() Float32x4 {
	var result Float32x4
	for i := 0; i < 4; i++ {
		result[i] = float32FromBits(uint32(v[i]))
	}
	return result
}

// ===== Int64x2 constructors and methods =====

// BroadcastInt64x2 creates a vector with all lanes set to the given value.
func BroadcastInt64x2(v int64) Int64x2 {
	return Int64x2{v, v}
}

// ===== Helper functions =====

func float32FromBits(b uint32) float32 {
	return math.Float32frombits(b)
}
