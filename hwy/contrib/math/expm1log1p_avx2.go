//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
)

// AVX2 vectorized constants for expm1/log1p
var (
	// Threshold for switching between Taylor series and direct computation
	// For |x| < threshold, use Taylor series for better accuracy
	expm1_32_threshold = archsimd.BroadcastFloat32x8(0.5)
	log1p_32_threshold = archsimd.BroadcastFloat32x8(0.5)

	// Taylor series coefficients for expm1: x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6!
	expm1_32_c1 = archsimd.BroadcastFloat32x8(1.0)                    // x
	expm1_32_c2 = archsimd.BroadcastFloat32x8(0.5)                    // 1/2!
	expm1_32_c3 = archsimd.BroadcastFloat32x8(0.16666666666666666)    // 1/3!
	expm1_32_c4 = archsimd.BroadcastFloat32x8(0.041666666666666664)   // 1/4!
	expm1_32_c5 = archsimd.BroadcastFloat32x8(0.008333333333333333)   // 1/5!
	expm1_32_c6 = archsimd.BroadcastFloat32x8(0.001388888888888889)   // 1/6!
	expm1_32_c7 = archsimd.BroadcastFloat32x8(0.0001984126984126984)  // 1/7!
	expm1_32_c8 = archsimd.BroadcastFloat32x8(0.0000248015873015873)  // 1/8!

	// Taylor series coefficients for log1p: x - x^2/2 + x^3/3 - x^4/4 + x^5/5 - ...
	log1p_32_c1 = archsimd.BroadcastFloat32x8(1.0)                    // x
	log1p_32_c2 = archsimd.BroadcastFloat32x8(-0.5)                   // -1/2
	log1p_32_c3 = archsimd.BroadcastFloat32x8(0.3333333333333333)     // 1/3
	log1p_32_c4 = archsimd.BroadcastFloat32x8(-0.25)                  // -1/4
	log1p_32_c5 = archsimd.BroadcastFloat32x8(0.2)                    // 1/5
	log1p_32_c6 = archsimd.BroadcastFloat32x8(-0.16666666666666666)   // -1/6
	log1p_32_c7 = archsimd.BroadcastFloat32x8(0.14285714285714285)    // 1/7
	log1p_32_c8 = archsimd.BroadcastFloat32x8(-0.125)                 // -1/8

	// Constants
	expm1_32_one     = archsimd.BroadcastFloat32x8(1.0)
	expm1_32_negOne  = archsimd.BroadcastFloat32x8(-1.0)
	expm1_32_zero    = archsimd.BroadcastFloat32x8(0.0)
	expm1_32_negInf  = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(-1)))
	expm1_32_posInf  = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(1)))
	expm1_32_nan     = archsimd.BroadcastFloat32x8(float32(stdmath.NaN()))

	log1p_32_one     = archsimd.BroadcastFloat32x8(1.0)
	log1p_32_negOne  = archsimd.BroadcastFloat32x8(-1.0)
	log1p_32_zero    = archsimd.BroadcastFloat32x8(0.0)
	log1p_32_negInf  = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(-1)))
	log1p_32_posInf  = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(1)))
	log1p_32_nan     = archsimd.BroadcastFloat32x8(float32(stdmath.NaN()))
)

// Expm1_AVX2_F32x8 computes exp(x) - 1 for a single Float32x8 vector.
//
// This function is more accurate than computing exp(x) - 1 directly when x
// is close to zero, avoiding catastrophic cancellation.
//
// Algorithm:
// - For |x| > threshold: compute exp(x) - 1 directly using Exp_AVX2_F32x8
// - For |x| <= threshold: use Taylor series x + x^2/2! + x^3/3! + ...
func Expm1_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Compute absolute value for threshold comparison: max(x, -x)
	absX := x.Max(expm1_32_zero.Sub(x))

	// Mask for small values where Taylor series should be used
	// LessOrEqual = Less OR Equal
	smallMask := absX.Less(expm1_32_threshold).Or(absX.Equal(expm1_32_threshold))

	// Compute Taylor series for small values: x + x^2/2! + x^3/3! + ...
	// Using Horner's method: x * (1 + x * (1/2! + x * (1/3! + x * (1/4! + ...))))
	// p = c8 + c7*x + c6*x^2 + c5*x^3 + ...
	// Horner's method from inside out for the series after the first term
	p := expm1_32_c8.MulAdd(x, expm1_32_c7)
	p = p.MulAdd(x, expm1_32_c6)
	p = p.MulAdd(x, expm1_32_c5)
	p = p.MulAdd(x, expm1_32_c4)
	p = p.MulAdd(x, expm1_32_c3)
	p = p.MulAdd(x, expm1_32_c2)
	p = p.MulAdd(x, expm1_32_c1)
	// Now p = 1 + x/2! + x^2/3! + x^3/4! + ...
	// expm1(x) = x * p
	taylorResult := x.Mul(p)

	// Compute exp(x) - 1 for large values
	expResult := Exp_AVX2_F32x8(x)
	largeResult := expResult.Sub(expm1_32_one)

	// Merge results: use Taylor series for small values, exp(x)-1 for large
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	result := taylorResult.Merge(largeResult, smallMask)

	// Handle special cases
	// +Inf -> +Inf
	posInfMask := x.Equal(expm1_32_posInf)
	result = expm1_32_posInf.Merge(result, posInfMask)

	// -Inf -> -1
	negInfMask := x.Equal(expm1_32_negInf)
	result = expm1_32_negOne.Merge(result, negInfMask)

	return result
}

// Log1p_AVX2_F32x8 computes log(1 + x) for a single Float32x8 vector.
//
// This function is more accurate than computing log(1 + x) directly when x
// is close to zero, avoiding catastrophic cancellation.
//
// Algorithm:
// - For |x| > threshold: compute log(1 + x) directly using Log_AVX2_F32x8
// - For |x| <= threshold: use Taylor series x - x^2/2 + x^3/3 - x^4/4 + ...
func Log1p_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Save original for special case detection
	origX := x

	// Compute absolute value for threshold comparison: max(x, -x)
	absX := x.Max(log1p_32_zero.Sub(x))

	// Mask for small values where Taylor series should be used
	// LessOrEqual = Less OR Equal
	smallMask := absX.Less(log1p_32_threshold).Or(absX.Equal(log1p_32_threshold))

	// Compute Taylor series for small values: x - x^2/2 + x^3/3 - x^4/4 + ...
	// Using Horner's method: x * (1 + x * (-1/2 + x * (1/3 + x * (-1/4 + ...))))
	// p = c8 + c7*x + c6*x^2 + ...
	p := log1p_32_c8.MulAdd(x, log1p_32_c7)
	p = p.MulAdd(x, log1p_32_c6)
	p = p.MulAdd(x, log1p_32_c5)
	p = p.MulAdd(x, log1p_32_c4)
	p = p.MulAdd(x, log1p_32_c3)
	p = p.MulAdd(x, log1p_32_c2)
	p = p.MulAdd(x, log1p_32_c1)
	// Now p = 1 - x/2 + x^2/3 - x^3/4 + ...
	// log1p(x) = x * p
	taylorResult := x.Mul(p)

	// Compute log(1 + x) for large values
	onePlusX := log1p_32_one.Add(x)
	largeResult := Log_AVX2_F32x8(onePlusX)

	// Merge results: use Taylor series for small values, log(1+x) for large
	// Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE
	result := taylorResult.Merge(largeResult, smallMask)

	// Handle special cases
	// x == -1: return -Inf
	negOneMask := origX.Equal(log1p_32_negOne)
	result = log1p_32_negInf.Merge(result, negOneMask)

	// x < -1: return NaN
	lessThanNegOneMask := origX.Less(log1p_32_negOne)
	result = log1p_32_nan.Merge(result, lessThanNegOneMask)

	// x == +Inf: return +Inf
	posInfMask := origX.Equal(log1p_32_posInf)
	result = log1p_32_posInf.Merge(result, posInfMask)

	return result
}

// Expm1_AVX2_F64x4 computes exp(x) - 1 for a single Float64x4 vector.
//
// Note: Uses scalar fallback because AVX2 lacks proper 64-bit integer shift
// support. The Go compiler generates AVX-512 EVEX instructions for Int64x4
// shifts, which fail on AVX2-only hardware.
func Expm1_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var in, out [4]float64
	x.StoreSlice(in[:])
	for i := range in {
		out[i] = stdmath.Expm1(in[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}

// Log1p_AVX2_F64x4 computes log(1 + x) for a single Float64x4 vector.
//
// Note: Uses scalar fallback because AVX2 lacks proper 64-bit integer shift
// support. The Go compiler generates AVX-512 EVEX instructions for Int64x4
// shifts, which fail on AVX2-only hardware.
func Log1p_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var in, out [4]float64
	x.StoreSlice(in[:])
	for i := range in {
		out[i] = stdmath.Log1p(in[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}
