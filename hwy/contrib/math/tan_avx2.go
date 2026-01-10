//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
)

// AVX2 vectorized constants for tan32
var (
	tan32_zero   = archsimd.BroadcastFloat32x8(0.0)
	tan32_nan    = archsimd.BroadcastFloat32x8(float32(stdmath.NaN()))
	tan32_inf    = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(1)))
	tan32_negInf = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(-1)))

	// Small threshold for division safety
	tan32_tiny = archsimd.BroadcastFloat32x8(1e-30)
)

// AVX2 vectorized constants for tan64
var (
	tan64_zero   = archsimd.BroadcastFloat64x4(0.0)
	tan64_nan    = archsimd.BroadcastFloat64x4(stdmath.NaN())
	tan64_inf    = archsimd.BroadcastFloat64x4(stdmath.Inf(1))
	tan64_negInf = archsimd.BroadcastFloat64x4(stdmath.Inf(-1))
	tan64_tiny   = archsimd.BroadcastFloat64x4(1e-100)
)

// Tan_AVX2_F32x8 computes tan(x) for a single Float32x8 vector.
//
// Algorithm: tan(x) = sin(x) / cos(x)
// Uses the existing SinCos implementation for efficiency.
//
// Special cases:
//   - tan(±0) = ±0
//   - tan(±Inf) = NaN
//   - tan(NaN) = NaN
//   - tan(x) = ±Inf where cos(x) = 0
func Tan_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Compute sin and cos together (shares range reduction)
	sin, cos := sinCos32Core(x)

	// tan = sin / cos
	// Handle division by zero: where |cos| is tiny, result should be ±Inf
	result := sin.Div(cos)

	// Handle special cases: ±Inf -> NaN
	infMask := x.Equal(tan32_inf).Or(x.Equal(tan32_negInf))
	result = tan32_nan.Merge(result, infMask)

	return result
}

// Tan_AVX2_F64x4 computes tan(x) for a single Float64x4 vector.
//
// Note: Uses scalar fallback for float64 to match Sin/Cos behavior.
func Tan_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var in, out [4]float64
	x.StoreSlice(in[:])
	for i := range in {
		out[i] = stdmath.Tan(in[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}
