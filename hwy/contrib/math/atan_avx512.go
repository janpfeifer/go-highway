//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"sync"
)

// Lazy initialization for AVX-512 atan constants to avoid executing AVX-512
// instructions at package load time on machines without AVX-512 support.

var atan512Init sync.Once

// AVX-512 vectorized constants for atan32
var (
	// Constants
	atan512_32_zero       archsimd.Float32x16
	atan512_32_one        archsimd.Float32x16
	atan512_32_piOver2    archsimd.Float32x16
	atan512_32_piOver4    archsimd.Float32x16
	atan512_32_pi         archsimd.Float32x16
	atan512_32_negPi      archsimd.Float32x16
	atan512_32_tanPiOver8 archsimd.Float32x16

	// Polynomial coefficients for atan(x) on [0, tan(π/8)]
	atan512_32_c1 archsimd.Float32x16
	atan512_32_c2 archsimd.Float32x16
	atan512_32_c3 archsimd.Float32x16
	atan512_32_c4 archsimd.Float32x16
	atan512_32_c5 archsimd.Float32x16

	// Sign mask for float32
	atan512_32_signMask archsimd.Int32x16
	atan512_32_absMask  archsimd.Int32x16
)

// AVX-512 vectorized constants for atan64
var (
	// Constants
	atan512_64_zero       archsimd.Float64x8
	atan512_64_one        archsimd.Float64x8
	atan512_64_piOver2    archsimd.Float64x8
	atan512_64_piOver4    archsimd.Float64x8
	atan512_64_pi         archsimd.Float64x8
	atan512_64_negPi      archsimd.Float64x8
	atan512_64_tanPiOver8 archsimd.Float64x8

	// Polynomial coefficients for atan(x) on [0, tan(π/8)]
	atan512_64_c1 archsimd.Float64x8
	atan512_64_c2 archsimd.Float64x8
	atan512_64_c3 archsimd.Float64x8
	atan512_64_c4 archsimd.Float64x8
	atan512_64_c5 archsimd.Float64x8
	atan512_64_c6 archsimd.Float64x8
	atan512_64_c7 archsimd.Float64x8

	// Sign mask for float64
	atan512_64_signMask archsimd.Int64x8
	atan512_64_absMask  archsimd.Int64x8
)

func initAtan512Constants() {
	// Float32 constants
	atan512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	atan512_32_one = archsimd.BroadcastFloat32x16(1.0)
	atan512_32_piOver2 = archsimd.BroadcastFloat32x16(1.5707963267948966)
	atan512_32_piOver4 = archsimd.BroadcastFloat32x16(0.7853981633974483)
	atan512_32_pi = archsimd.BroadcastFloat32x16(3.141592653589793)
	atan512_32_negPi = archsimd.BroadcastFloat32x16(-3.141592653589793)
	atan512_32_tanPiOver8 = archsimd.BroadcastFloat32x16(0.4142135623730950488) // tan(π/8) = sqrt(2) - 1
	atan512_32_signMask = archsimd.BroadcastInt32x16(-2147483648)
	atan512_32_absMask = archsimd.BroadcastInt32x16(0x7FFFFFFF)

	// Float32 polynomial coefficients
	atan512_32_c1 = archsimd.BroadcastFloat32x16(-0.3333333333)
	atan512_32_c2 = archsimd.BroadcastFloat32x16(0.2)
	atan512_32_c3 = archsimd.BroadcastFloat32x16(-0.1428571429)
	atan512_32_c4 = archsimd.BroadcastFloat32x16(0.1111111111)
	atan512_32_c5 = archsimd.BroadcastFloat32x16(-0.0909090909)

	// Float64 constants
	atan512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	atan512_64_one = archsimd.BroadcastFloat64x8(1.0)
	atan512_64_piOver2 = archsimd.BroadcastFloat64x8(1.5707963267948966)
	atan512_64_piOver4 = archsimd.BroadcastFloat64x8(0.7853981633974483)
	atan512_64_pi = archsimd.BroadcastFloat64x8(3.141592653589793)
	atan512_64_negPi = archsimd.BroadcastFloat64x8(-3.141592653589793)
	atan512_64_tanPiOver8 = archsimd.BroadcastFloat64x8(0.4142135623730950488)
	atan512_64_signMask = archsimd.BroadcastInt64x8(-9223372036854775808)
	atan512_64_absMask = archsimd.BroadcastInt64x8(0x7FFFFFFFFFFFFFFF)

	// Float64 polynomial coefficients (more terms for higher precision)
	atan512_64_c1 = archsimd.BroadcastFloat64x8(-0.3333333333333333)
	atan512_64_c2 = archsimd.BroadcastFloat64x8(0.2)
	atan512_64_c3 = archsimd.BroadcastFloat64x8(-0.14285714285714285)
	atan512_64_c4 = archsimd.BroadcastFloat64x8(0.1111111111111111)
	atan512_64_c5 = archsimd.BroadcastFloat64x8(-0.09090909090909091)
	atan512_64_c6 = archsimd.BroadcastFloat64x8(0.07692307692307693)
	atan512_64_c7 = archsimd.BroadcastFloat64x8(-0.06666666666666667)
}

// Atan_AVX512_F32x16 computes atan(x) for a single Float32x16 vector.
//
// Algorithm uses two-level range reduction for better accuracy:
// 1. If |x| > 1: use atan(x) = π/2 - atan(1/x)
// 2. If |x| > tan(π/8) ≈ 0.414: use atan(x) = π/4 + atan((x-1)/(x+1))
//
// This reduces the argument to [0, tan(π/8)] where the polynomial is accurate.
func Atan_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	atan512Init.Do(initAtan512Constants)

	// Get absolute value and sign
	xBits := x.AsInt32x16()
	signBits := xBits.And(atan512_32_signMask)
	absXBits := xBits.And(atan512_32_absMask)
	absX := absXBits.AsFloat32x16()

	// Range reduction level 1: if |x| > 1, use atan(x) = π/2 - atan(1/x)
	useReciprocalMask := absX.Greater(atan512_32_one)
	recipAbsX := atan512_32_one.Div(absX)
	// Merge semantics: a.Merge(b, mask) = a where mask TRUE, b where mask FALSE
	reduced := recipAbsX.AsInt32x16().Merge(absX.AsInt32x16(), useReciprocalMask).AsFloat32x16()

	// Range reduction level 2: if reduced > tan(π/8), use atan(x) = π/4 + atan((x-1)/(x+1))
	useIdentityMask := reduced.Greater(atan512_32_tanPiOver8)
	xMinus1 := reduced.Sub(atan512_32_one)
	xPlus1 := reduced.Add(atan512_32_one)
	transformed := xMinus1.Div(xPlus1)
	reduced = transformed.AsInt32x16().Merge(reduced.AsInt32x16(), useIdentityMask).AsFloat32x16()

	// Compute polynomial
	z2 := reduced.Mul(reduced)
	poly := atan512_32_c5.MulAdd(z2, atan512_32_c4)
	poly = poly.MulAdd(z2, atan512_32_c3)
	poly = poly.MulAdd(z2, atan512_32_c2)
	poly = poly.MulAdd(z2, atan512_32_c1)
	poly = poly.MulAdd(z2, atan512_32_one)
	atanCore := reduced.Mul(poly)

	// Adjust for identity transform
	atanWithIdentity := atan512_32_piOver4.Add(atanCore)
	atanReduced := atanWithIdentity.AsInt32x16().Merge(atanCore.AsInt32x16(), useIdentityMask).AsFloat32x16()

	// Adjust for reciprocal
	atanWithReciprocal := atan512_32_piOver2.Sub(atanReduced)
	resultAbs := atanWithReciprocal.AsInt32x16().Merge(atanReduced.AsInt32x16(), useReciprocalMask).AsFloat32x16()

	// Restore sign
	resultBits := resultAbs.AsInt32x16().Or(signBits)
	return resultBits.AsFloat32x16()
}

// Atan_AVX512_F64x8 computes atan(x) for a single Float64x8 vector.
//
// Algorithm uses two-level range reduction for better accuracy:
// 1. If |x| > 1: use atan(x) = π/2 - atan(1/x)
// 2. If |x| > tan(π/8) ≈ 0.414: use atan(x) = π/4 + atan((x-1)/(x+1))
func Atan_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	atan512Init.Do(initAtan512Constants)

	// Get absolute value and sign
	xBits := x.AsInt64x8()
	signBits := xBits.And(atan512_64_signMask)
	absXBits := xBits.And(atan512_64_absMask)
	absX := absXBits.AsFloat64x8()

	// Range reduction level 1
	useReciprocalMask := absX.Greater(atan512_64_one)
	recipAbsX := atan512_64_one.Div(absX)
	reduced := recipAbsX.AsInt64x8().Merge(absX.AsInt64x8(), useReciprocalMask).AsFloat64x8()

	// Range reduction level 2
	useIdentityMask := reduced.Greater(atan512_64_tanPiOver8)
	xMinus1 := reduced.Sub(atan512_64_one)
	xPlus1 := reduced.Add(atan512_64_one)
	transformed := xMinus1.Div(xPlus1)
	reduced = transformed.AsInt64x8().Merge(reduced.AsInt64x8(), useIdentityMask).AsFloat64x8()

	// Compute polynomial (more terms for float64 precision)
	z2 := reduced.Mul(reduced)
	poly := atan512_64_c7.MulAdd(z2, atan512_64_c6)
	poly = poly.MulAdd(z2, atan512_64_c5)
	poly = poly.MulAdd(z2, atan512_64_c4)
	poly = poly.MulAdd(z2, atan512_64_c3)
	poly = poly.MulAdd(z2, atan512_64_c2)
	poly = poly.MulAdd(z2, atan512_64_c1)
	poly = poly.MulAdd(z2, atan512_64_one)
	atanCore := reduced.Mul(poly)

	// Adjust for identity transform
	atanWithIdentity := atan512_64_piOver4.Add(atanCore)
	atanReduced := atanWithIdentity.AsInt64x8().Merge(atanCore.AsInt64x8(), useIdentityMask).AsFloat64x8()

	// Adjust for reciprocal
	atanWithReciprocal := atan512_64_piOver2.Sub(atanReduced)
	resultAbs := atanWithReciprocal.AsInt64x8().Merge(atanReduced.AsInt64x8(), useReciprocalMask).AsFloat64x8()

	// Restore sign
	resultBits := resultAbs.AsInt64x8().Or(signBits)
	return resultBits.AsFloat64x8()
}

// Atan2_AVX512_F32x16 computes atan2(y, x) for Float32x16 vectors.
//
// Uses two-level range reduction for accuracy.
func Atan2_AVX512_F32x16(y, x archsimd.Float32x16) archsimd.Float32x16 {
	atan512Init.Do(initAtan512Constants)

	// Get masks for signs and zeros
	xZeroMask := x.Equal(atan512_32_zero)
	yZeroMask := y.Equal(atan512_32_zero)
	xNegMask := x.Less(atan512_32_zero)
	yNegMask := y.Less(atan512_32_zero)
	yPosMask := y.Greater(atan512_32_zero)

	// Safe division (replace x=0 with 1 temporarily)
	safeX := atan512_32_one.AsInt32x16().Merge(x.AsInt32x16(), xZeroMask).AsFloat32x16()
	ratio := y.Div(safeX)

	// Get sign and absolute value of ratio
	ratioBits := ratio.AsInt32x16()
	ratioSign := ratioBits.And(atan512_32_signMask)
	absRatio := ratioBits.And(atan512_32_absMask).AsFloat32x16()

	// Range reduction level 1
	useReciprocalMask := absRatio.Greater(atan512_32_one)
	recipRatio := atan512_32_one.Div(absRatio)
	reduced := recipRatio.AsInt32x16().Merge(absRatio.AsInt32x16(), useReciprocalMask).AsFloat32x16()

	// Range reduction level 2
	useIdentityMask := reduced.Greater(atan512_32_tanPiOver8)
	rMinus1 := reduced.Sub(atan512_32_one)
	rPlus1 := reduced.Add(atan512_32_one)
	transformed := rMinus1.Div(rPlus1)
	reduced = transformed.AsInt32x16().Merge(reduced.AsInt32x16(), useIdentityMask).AsFloat32x16()

	// Compute polynomial
	r2 := reduced.Mul(reduced)
	poly := atan512_32_c5.MulAdd(r2, atan512_32_c4)
	poly = poly.MulAdd(r2, atan512_32_c3)
	poly = poly.MulAdd(r2, atan512_32_c2)
	poly = poly.MulAdd(r2, atan512_32_c1)
	poly = poly.MulAdd(r2, atan512_32_one)
	atanCore := reduced.Mul(poly)

	// Adjust for identity transform
	atanWithIdentity := atan512_32_piOver4.Add(atanCore)
	atanReduced := atanWithIdentity.AsInt32x16().Merge(atanCore.AsInt32x16(), useIdentityMask).AsFloat32x16()

	// Adjust for reciprocal
	atanWithReciprocal := atan512_32_piOver2.Sub(atanReduced)
	atanAbs := atanWithReciprocal.AsInt32x16().Merge(atanReduced.AsInt32x16(), useReciprocalMask).AsFloat32x16()

	// Apply ratio sign
	atanVal := atanAbs.AsInt32x16().Or(ratioSign).AsFloat32x16()

	// Quadrant adjustment
	yNonNegMask := y.GreaterEqual(atan512_32_zero)
	needAddPiMask := xNegMask.And(yNonNegMask)
	needSubPiMask := xNegMask.And(yNegMask)

	atanPlusPi := atanVal.Add(atan512_32_pi)
	atanMinusPi := atanVal.Add(atan512_32_negPi)

	atanVal = atanPlusPi.AsInt32x16().Merge(atanVal.AsInt32x16(), needAddPiMask).AsFloat32x16()
	atanVal = atanMinusPi.AsInt32x16().Merge(atanVal.AsInt32x16(), needSubPiMask).AsFloat32x16()

	// Handle x = 0 cases
	negPiOver2 := atan512_32_zero.Sub(atan512_32_piOver2)

	xZeroYPosMask := xZeroMask.And(yPosMask)
	atanVal = atan512_32_piOver2.AsInt32x16().Merge(atanVal.AsInt32x16(), xZeroYPosMask).AsFloat32x16()

	xZeroYNegMask := xZeroMask.And(yNegMask)
	atanVal = negPiOver2.AsInt32x16().Merge(atanVal.AsInt32x16(), xZeroYNegMask).AsFloat32x16()

	xZeroYZeroMask := xZeroMask.And(yZeroMask)
	atanVal = atan512_32_zero.AsInt32x16().Merge(atanVal.AsInt32x16(), xZeroYZeroMask).AsFloat32x16()

	return atanVal
}

// Atan2_AVX512_F64x8 computes atan2(y, x) for Float64x8 vectors.
//
// Note: Uses scalar fallback for simplicity and correctness.
func Atan2_AVX512_F64x8(y, x archsimd.Float64x8) archsimd.Float64x8 {
	var yIn, xIn, out [8]float64
	y.StoreSlice(yIn[:])
	x.StoreSlice(xIn[:])
	for i := range yIn {
		out[i] = stdmath.Atan2(yIn[i], xIn[i])
	}
	return archsimd.LoadFloat64x8Slice(out[:])
}
