//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"sync"
)

// Lazy initialization for AVX-512 tan constants to avoid executing AVX-512
// instructions at package load time on machines without AVX-512 support.

var tan512Init sync.Once

// AVX-512 vectorized constants for tan32
var (
	tan512_32_zero   archsimd.Float32x16
	tan512_32_nan    archsimd.Float32x16
	tan512_32_inf    archsimd.Float32x16
	tan512_32_negInf archsimd.Float32x16
)

// AVX-512 vectorized constants for tan64
var (
	tan512_64_zero   archsimd.Float64x8
	tan512_64_nan    archsimd.Float64x8
	tan512_64_inf    archsimd.Float64x8
	tan512_64_negInf archsimd.Float64x8
)

func initTan512Constants() {
	// Float32 constants
	tan512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	tan512_32_nan = archsimd.BroadcastFloat32x16(float32(stdmath.NaN()))
	tan512_32_inf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(1)))
	tan512_32_negInf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(-1)))

	// Float64 constants
	tan512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	tan512_64_nan = archsimd.BroadcastFloat64x8(stdmath.NaN())
	tan512_64_inf = archsimd.BroadcastFloat64x8(stdmath.Inf(1))
	tan512_64_negInf = archsimd.BroadcastFloat64x8(stdmath.Inf(-1))
}

// Tan_AVX512_F32x16 computes tan(x) for a single Float32x16 vector.
//
// Algorithm: tan(x) = sin(x) / cos(x)
// Uses the existing SinCos implementation for efficiency.
//
// Special cases:
//   - tan(±0) = ±0
//   - tan(±Inf) = NaN
//   - tan(NaN) = NaN
//   - tan(x) = ±Inf where cos(x) = 0
func Tan_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	tan512Init.Do(initTan512Constants)
	trig512Init.Do(initTrig512Constants)

	// Compute sin and cos together (shares range reduction)
	sin, cos := sinCos512_32Core(x)

	// tan = sin / cos
	// Handle division by zero: where |cos| is tiny, result should be ±Inf
	result := sin.Div(cos)

	// Handle special cases: ±Inf -> NaN
	infMask := x.Equal(tan512_32_inf).Or(x.Equal(tan512_32_negInf))
	result = tan512_32_nan.Merge(result, infMask)

	return result
}

// Tan_AVX512_F64x8 computes tan(x) for a single Float64x8 vector.
//
// Algorithm: tan(x) = sin(x) / cos(x)
// Uses the existing SinCos implementation for efficiency.
//
// Special cases:
//   - tan(±0) = ±0
//   - tan(±Inf) = NaN
//   - tan(NaN) = NaN
//   - tan(x) = ±Inf where cos(x) = 0
func Tan_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	tan512Init.Do(initTan512Constants)
	trig512Init.Do(initTrig512Constants)

	// Compute sin and cos together (shares range reduction)
	sin, cos := sinCos512_64Core(x)

	// tan = sin / cos
	// Handle division by zero: where |cos| is tiny, result should be ±Inf
	result := sin.Div(cos)

	// Handle special cases: ±Inf -> NaN
	infMask := x.Equal(tan512_64_inf).Or(x.Equal(tan512_64_negInf))
	result = tan512_64_nan.Merge(result, infMask)

	return result
}
