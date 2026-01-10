//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"sync"
)

// Lazy initialization for AVX-512 constants to avoid executing AVX-512
// instructions at package load time on machines without AVX-512 support.

var expm1log1p512Init sync.Once

// AVX-512 vectorized constants for expm1 (Float32x16)
var (
	expm1_512_32_threshold archsimd.Float32x16
	expm1_512_32_c1        archsimd.Float32x16
	expm1_512_32_c2        archsimd.Float32x16
	expm1_512_32_c3        archsimd.Float32x16
	expm1_512_32_c4        archsimd.Float32x16
	expm1_512_32_c5        archsimd.Float32x16
	expm1_512_32_c6        archsimd.Float32x16
	expm1_512_32_c7        archsimd.Float32x16
	expm1_512_32_c8        archsimd.Float32x16
	expm1_512_32_one       archsimd.Float32x16
	expm1_512_32_negOne    archsimd.Float32x16
	expm1_512_32_zero      archsimd.Float32x16
	expm1_512_32_posInf    archsimd.Float32x16
	expm1_512_32_negInf    archsimd.Float32x16
)

// AVX-512 vectorized constants for expm1 (Float64x8)
var (
	expm1_512_64_threshold archsimd.Float64x8
	expm1_512_64_c1        archsimd.Float64x8
	expm1_512_64_c2        archsimd.Float64x8
	expm1_512_64_c3        archsimd.Float64x8
	expm1_512_64_c4        archsimd.Float64x8
	expm1_512_64_c5        archsimd.Float64x8
	expm1_512_64_c6        archsimd.Float64x8
	expm1_512_64_c7        archsimd.Float64x8
	expm1_512_64_c8        archsimd.Float64x8
	expm1_512_64_c9        archsimd.Float64x8
	expm1_512_64_c10       archsimd.Float64x8
	expm1_512_64_one       archsimd.Float64x8
	expm1_512_64_negOne    archsimd.Float64x8
	expm1_512_64_zero      archsimd.Float64x8
	expm1_512_64_posInf    archsimd.Float64x8
	expm1_512_64_negInf    archsimd.Float64x8
)

// AVX-512 vectorized constants for log1p (Float32x16)
var (
	log1p_512_32_threshold archsimd.Float32x16
	log1p_512_32_c1        archsimd.Float32x16
	log1p_512_32_c2        archsimd.Float32x16
	log1p_512_32_c3        archsimd.Float32x16
	log1p_512_32_c4        archsimd.Float32x16
	log1p_512_32_c5        archsimd.Float32x16
	log1p_512_32_c6        archsimd.Float32x16
	log1p_512_32_c7        archsimd.Float32x16
	log1p_512_32_c8        archsimd.Float32x16
	log1p_512_32_one       archsimd.Float32x16
	log1p_512_32_negOne    archsimd.Float32x16
	log1p_512_32_zero      archsimd.Float32x16
	log1p_512_32_posInf    archsimd.Float32x16
	log1p_512_32_negInf    archsimd.Float32x16
	log1p_512_32_nan       archsimd.Float32x16
)

// AVX-512 vectorized constants for log1p (Float64x8)
var (
	log1p_512_64_threshold archsimd.Float64x8
	log1p_512_64_c1        archsimd.Float64x8
	log1p_512_64_c2        archsimd.Float64x8
	log1p_512_64_c3        archsimd.Float64x8
	log1p_512_64_c4        archsimd.Float64x8
	log1p_512_64_c5        archsimd.Float64x8
	log1p_512_64_c6        archsimd.Float64x8
	log1p_512_64_c7        archsimd.Float64x8
	log1p_512_64_c8        archsimd.Float64x8
	log1p_512_64_c9        archsimd.Float64x8
	log1p_512_64_c10       archsimd.Float64x8
	log1p_512_64_one       archsimd.Float64x8
	log1p_512_64_negOne    archsimd.Float64x8
	log1p_512_64_zero      archsimd.Float64x8
	log1p_512_64_posInf    archsimd.Float64x8
	log1p_512_64_negInf    archsimd.Float64x8
	log1p_512_64_nan       archsimd.Float64x8
)

func initExpm1Log1p512Constants() {
	// Float32 expm1 constants
	expm1_512_32_threshold = archsimd.BroadcastFloat32x16(0.5)
	expm1_512_32_c1 = archsimd.BroadcastFloat32x16(1.0)
	expm1_512_32_c2 = archsimd.BroadcastFloat32x16(0.5)
	expm1_512_32_c3 = archsimd.BroadcastFloat32x16(0.16666666666666666)
	expm1_512_32_c4 = archsimd.BroadcastFloat32x16(0.041666666666666664)
	expm1_512_32_c5 = archsimd.BroadcastFloat32x16(0.008333333333333333)
	expm1_512_32_c6 = archsimd.BroadcastFloat32x16(0.001388888888888889)
	expm1_512_32_c7 = archsimd.BroadcastFloat32x16(0.0001984126984126984)
	expm1_512_32_c8 = archsimd.BroadcastFloat32x16(0.0000248015873015873)
	expm1_512_32_one = archsimd.BroadcastFloat32x16(1.0)
	expm1_512_32_negOne = archsimd.BroadcastFloat32x16(-1.0)
	expm1_512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	expm1_512_32_posInf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(1)))
	expm1_512_32_negInf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(-1)))

	// Float64 expm1 constants (higher-degree polynomial for precision)
	expm1_512_64_threshold = archsimd.BroadcastFloat64x8(0.5)
	expm1_512_64_c1 = archsimd.BroadcastFloat64x8(1.0)
	expm1_512_64_c2 = archsimd.BroadcastFloat64x8(0.5)
	expm1_512_64_c3 = archsimd.BroadcastFloat64x8(0.16666666666666666)
	expm1_512_64_c4 = archsimd.BroadcastFloat64x8(0.041666666666666664)
	expm1_512_64_c5 = archsimd.BroadcastFloat64x8(0.008333333333333333)
	expm1_512_64_c6 = archsimd.BroadcastFloat64x8(0.001388888888888889)
	expm1_512_64_c7 = archsimd.BroadcastFloat64x8(0.0001984126984126984)
	expm1_512_64_c8 = archsimd.BroadcastFloat64x8(2.48015873015873e-05)
	expm1_512_64_c9 = archsimd.BroadcastFloat64x8(2.7557319223985893e-06)
	expm1_512_64_c10 = archsimd.BroadcastFloat64x8(2.755731922398589e-07)
	expm1_512_64_one = archsimd.BroadcastFloat64x8(1.0)
	expm1_512_64_negOne = archsimd.BroadcastFloat64x8(-1.0)
	expm1_512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	expm1_512_64_posInf = archsimd.BroadcastFloat64x8(stdmath.Inf(1))
	expm1_512_64_negInf = archsimd.BroadcastFloat64x8(stdmath.Inf(-1))

	// Float32 log1p constants
	log1p_512_32_threshold = archsimd.BroadcastFloat32x16(0.5)
	log1p_512_32_c1 = archsimd.BroadcastFloat32x16(1.0)
	log1p_512_32_c2 = archsimd.BroadcastFloat32x16(-0.5)
	log1p_512_32_c3 = archsimd.BroadcastFloat32x16(0.3333333333333333)
	log1p_512_32_c4 = archsimd.BroadcastFloat32x16(-0.25)
	log1p_512_32_c5 = archsimd.BroadcastFloat32x16(0.2)
	log1p_512_32_c6 = archsimd.BroadcastFloat32x16(-0.16666666666666666)
	log1p_512_32_c7 = archsimd.BroadcastFloat32x16(0.14285714285714285)
	log1p_512_32_c8 = archsimd.BroadcastFloat32x16(-0.125)
	log1p_512_32_one = archsimd.BroadcastFloat32x16(1.0)
	log1p_512_32_negOne = archsimd.BroadcastFloat32x16(-1.0)
	log1p_512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	log1p_512_32_posInf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(1)))
	log1p_512_32_negInf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(-1)))
	log1p_512_32_nan = archsimd.BroadcastFloat32x16(float32(stdmath.NaN()))

	// Float64 log1p constants (higher-degree polynomial for precision)
	log1p_512_64_threshold = archsimd.BroadcastFloat64x8(0.5)
	log1p_512_64_c1 = archsimd.BroadcastFloat64x8(1.0)
	log1p_512_64_c2 = archsimd.BroadcastFloat64x8(-0.5)
	log1p_512_64_c3 = archsimd.BroadcastFloat64x8(0.3333333333333333)
	log1p_512_64_c4 = archsimd.BroadcastFloat64x8(-0.25)
	log1p_512_64_c5 = archsimd.BroadcastFloat64x8(0.2)
	log1p_512_64_c6 = archsimd.BroadcastFloat64x8(-0.16666666666666666)
	log1p_512_64_c7 = archsimd.BroadcastFloat64x8(0.14285714285714285)
	log1p_512_64_c8 = archsimd.BroadcastFloat64x8(-0.125)
	log1p_512_64_c9 = archsimd.BroadcastFloat64x8(0.1111111111111111)
	log1p_512_64_c10 = archsimd.BroadcastFloat64x8(-0.1)
	log1p_512_64_one = archsimd.BroadcastFloat64x8(1.0)
	log1p_512_64_negOne = archsimd.BroadcastFloat64x8(-1.0)
	log1p_512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	log1p_512_64_posInf = archsimd.BroadcastFloat64x8(stdmath.Inf(1))
	log1p_512_64_negInf = archsimd.BroadcastFloat64x8(stdmath.Inf(-1))
	log1p_512_64_nan = archsimd.BroadcastFloat64x8(stdmath.NaN())
}

// Expm1_AVX512_F32x16 computes exp(x) - 1 for a single Float32x16 vector.
//
// This function is more accurate than computing exp(x) - 1 directly when x
// is close to zero, avoiding catastrophic cancellation.
//
// Algorithm:
// - For |x| > threshold: compute exp(x) - 1 directly using Exp_AVX512_F32x16
// - For |x| <= threshold: use Taylor series x + x^2/2! + x^3/3! + ...
func Expm1_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	expm1log1p512Init.Do(initExpm1Log1p512Constants)

	// Compute absolute value for threshold comparison: max(x, -x)
	absX := x.Max(expm1_512_32_zero.Sub(x))

	// Mask for small values where Taylor series should be used
	// LessOrEqual = Less OR Equal
	smallMask := absX.Less(expm1_512_32_threshold).Or(absX.Equal(expm1_512_32_threshold))

	// Compute Taylor series for small values: x + x^2/2! + x^3/3! + ...
	// Using Horner's method: x * (1 + x * (1/2! + x * (1/3! + x * (1/4! + ...))))
	// Horner's method from inside out for the series after the first term
	p := expm1_512_32_c8.MulAdd(x, expm1_512_32_c7)
	p = p.MulAdd(x, expm1_512_32_c6)
	p = p.MulAdd(x, expm1_512_32_c5)
	p = p.MulAdd(x, expm1_512_32_c4)
	p = p.MulAdd(x, expm1_512_32_c3)
	p = p.MulAdd(x, expm1_512_32_c2)
	p = p.MulAdd(x, expm1_512_32_c1)
	// Now p = 1 + x/2! + x^2/3! + x^3/4! + ...
	// expm1(x) = x * p
	taylorResult := x.Mul(p)

	// Compute exp(x) - 1 for large values
	expResult := Exp_AVX512_F32x16(x)
	largeResult := expResult.Sub(expm1_512_32_one)

	// Merge results: use Taylor series for small values, exp(x)-1 for large
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	result := taylorResult.Merge(largeResult, smallMask)

	// Handle special cases
	// +Inf -> +Inf
	posInfMask := x.Equal(expm1_512_32_posInf)
	result = expm1_512_32_posInf.Merge(result, posInfMask)

	// -Inf -> -1
	negInfMask := x.Equal(expm1_512_32_negInf)
	result = expm1_512_32_negOne.Merge(result, negInfMask)

	return result
}

// Expm1_AVX512_F64x8 computes exp(x) - 1 for a single Float64x8 vector.
//
// This function is more accurate than computing exp(x) - 1 directly when x
// is close to zero, avoiding catastrophic cancellation.
//
// Algorithm:
// - For |x| > threshold: compute exp(x) - 1 directly using Exp_AVX512_F64x8
// - For |x| <= threshold: use Taylor series x + x^2/2! + x^3/3! + ...
func Expm1_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	expm1log1p512Init.Do(initExpm1Log1p512Constants)

	// Compute absolute value for threshold comparison: max(x, -x)
	absX := x.Max(expm1_512_64_zero.Sub(x))

	// Mask for small values where Taylor series should be used
	// LessOrEqual = Less OR Equal
	smallMask := absX.Less(expm1_512_64_threshold).Or(absX.Equal(expm1_512_64_threshold))

	// Compute Taylor series for small values (higher degree for float64)
	// Using Horner's method
	p := expm1_512_64_c10.MulAdd(x, expm1_512_64_c9)
	p = p.MulAdd(x, expm1_512_64_c8)
	p = p.MulAdd(x, expm1_512_64_c7)
	p = p.MulAdd(x, expm1_512_64_c6)
	p = p.MulAdd(x, expm1_512_64_c5)
	p = p.MulAdd(x, expm1_512_64_c4)
	p = p.MulAdd(x, expm1_512_64_c3)
	p = p.MulAdd(x, expm1_512_64_c2)
	p = p.MulAdd(x, expm1_512_64_c1)
	// expm1(x) = x * p
	taylorResult := x.Mul(p)

	// Compute exp(x) - 1 for large values
	expResult := Exp_AVX512_F64x8(x)
	largeResult := expResult.Sub(expm1_512_64_one)

	// Merge results
	result := taylorResult.Merge(largeResult, smallMask)

	// Handle special cases
	posInfMask := x.Equal(expm1_512_64_posInf)
	result = expm1_512_64_posInf.Merge(result, posInfMask)

	negInfMask := x.Equal(expm1_512_64_negInf)
	result = expm1_512_64_negOne.Merge(result, negInfMask)

	return result
}

// Log1p_AVX512_F32x16 computes log(1 + x) for a single Float32x16 vector.
//
// This function is more accurate than computing log(1 + x) directly when x
// is close to zero, avoiding catastrophic cancellation.
//
// Algorithm:
// - For |x| > threshold: compute log(1 + x) directly using Log_AVX512_F32x16
// - For |x| <= threshold: use Taylor series x - x^2/2 + x^3/3 - x^4/4 + ...
func Log1p_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	expm1log1p512Init.Do(initExpm1Log1p512Constants)

	// Save original for special case detection
	origX := x

	// Compute absolute value for threshold comparison: max(x, -x)
	absX := x.Max(log1p_512_32_zero.Sub(x))

	// Mask for small values where Taylor series should be used
	// LessOrEqual = Less OR Equal
	smallMask := absX.Less(log1p_512_32_threshold).Or(absX.Equal(log1p_512_32_threshold))

	// Compute Taylor series for small values: x - x^2/2 + x^3/3 - x^4/4 + ...
	// Using Horner's method
	p := log1p_512_32_c8.MulAdd(x, log1p_512_32_c7)
	p = p.MulAdd(x, log1p_512_32_c6)
	p = p.MulAdd(x, log1p_512_32_c5)
	p = p.MulAdd(x, log1p_512_32_c4)
	p = p.MulAdd(x, log1p_512_32_c3)
	p = p.MulAdd(x, log1p_512_32_c2)
	p = p.MulAdd(x, log1p_512_32_c1)
	// log1p(x) = x * p
	taylorResult := x.Mul(p)

	// Compute log(1 + x) for large values
	onePlusX := log1p_512_32_one.Add(x)
	largeResult := Log_AVX512_F32x16(onePlusX)

	// Merge results
	result := taylorResult.Merge(largeResult, smallMask)

	// Handle special cases
	// x == -1: return -Inf
	negOneMask := origX.Equal(log1p_512_32_negOne)
	result = log1p_512_32_negInf.Merge(result, negOneMask)

	// x < -1: return NaN
	lessThanNegOneMask := origX.Less(log1p_512_32_negOne)
	result = log1p_512_32_nan.Merge(result, lessThanNegOneMask)

	// x == +Inf: return +Inf
	posInfMask := origX.Equal(log1p_512_32_posInf)
	result = log1p_512_32_posInf.Merge(result, posInfMask)

	return result
}

// Log1p_AVX512_F64x8 computes log(1 + x) for a single Float64x8 vector.
//
// This function is more accurate than computing log(1 + x) directly when x
// is close to zero, avoiding catastrophic cancellation.
//
// Algorithm:
// - For |x| > threshold: compute log(1 + x) directly using Log_AVX512_F64x8
// - For |x| <= threshold: use Taylor series x - x^2/2 + x^3/3 - x^4/4 + ...
func Log1p_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	expm1log1p512Init.Do(initExpm1Log1p512Constants)

	// Save original for special case detection
	origX := x

	// Compute absolute value for threshold comparison: max(x, -x)
	absX := x.Max(log1p_512_64_zero.Sub(x))

	// Mask for small values where Taylor series should be used
	// LessOrEqual = Less OR Equal
	smallMask := absX.Less(log1p_512_64_threshold).Or(absX.Equal(log1p_512_64_threshold))

	// Compute Taylor series for small values (higher degree for float64)
	// Using Horner's method
	p := log1p_512_64_c10.MulAdd(x, log1p_512_64_c9)
	p = p.MulAdd(x, log1p_512_64_c8)
	p = p.MulAdd(x, log1p_512_64_c7)
	p = p.MulAdd(x, log1p_512_64_c6)
	p = p.MulAdd(x, log1p_512_64_c5)
	p = p.MulAdd(x, log1p_512_64_c4)
	p = p.MulAdd(x, log1p_512_64_c3)
	p = p.MulAdd(x, log1p_512_64_c2)
	p = p.MulAdd(x, log1p_512_64_c1)
	// log1p(x) = x * p
	taylorResult := x.Mul(p)

	// Compute log(1 + x) for large values
	onePlusX := log1p_512_64_one.Add(x)
	largeResult := Log_AVX512_F64x8(onePlusX)

	// Merge results
	result := taylorResult.Merge(largeResult, smallMask)

	// Handle special cases
	// x == -1: return -Inf
	negOneMask := origX.Equal(log1p_512_64_negOne)
	result = log1p_512_64_negInf.Merge(result, negOneMask)

	// x < -1: return NaN
	lessThanNegOneMask := origX.Less(log1p_512_64_negOne)
	result = log1p_512_64_nan.Merge(result, lessThanNegOneMask)

	// x == +Inf: return +Inf
	posInfMask := origX.Equal(log1p_512_64_posInf)
	result = log1p_512_64_posInf.Merge(result, posInfMask)

	return result
}
