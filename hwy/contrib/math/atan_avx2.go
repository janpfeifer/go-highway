//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
)

// AVX2 vectorized constants for atan32
var (
	// Constants
	atan32_zero      = archsimd.BroadcastFloat32x8(0.0)
	atan32_one       = archsimd.BroadcastFloat32x8(1.0)
	atan32_piOver2   = archsimd.BroadcastFloat32x8(1.5707963267948966)
	atan32_piOver4   = archsimd.BroadcastFloat32x8(0.7853981633974483)
	atan32_pi        = archsimd.BroadcastFloat32x8(3.141592653589793)
	atan32_negPi     = archsimd.BroadcastFloat32x8(-3.141592653589793)
	atan32_tanPiOver8 = archsimd.BroadcastFloat32x8(0.4142135623730950488) // tan(π/8) = sqrt(2) - 1

	// Polynomial coefficients for atan(x) on [0, tan(π/8)]
	// atan(x) ≈ x * (1 + x² * (c1 + x² * (c2 + x² * (c3 + x² * (c4 + x² * c5)))))
	atan32_c1 = archsimd.BroadcastFloat32x8(-0.3333333333)
	atan32_c2 = archsimd.BroadcastFloat32x8(0.2)
	atan32_c3 = archsimd.BroadcastFloat32x8(-0.1428571429)
	atan32_c4 = archsimd.BroadcastFloat32x8(0.1111111111)
	atan32_c5 = archsimd.BroadcastFloat32x8(-0.0909090909)

	// Sign mask for float32
	atan32_signMask = archsimd.BroadcastInt32x8(-2147483648) // 0x80000000
	atan32_absMask  = archsimd.BroadcastInt32x8(0x7FFFFFFF)
)

// AVX2 vectorized constants for atan64
var (
	atan64_zero      = archsimd.BroadcastFloat64x4(0.0)
	atan64_one       = archsimd.BroadcastFloat64x4(1.0)
	atan64_piOver2   = archsimd.BroadcastFloat64x4(1.5707963267948966)
	atan64_piOver4   = archsimd.BroadcastFloat64x4(0.7853981633974483)
	atan64_pi        = archsimd.BroadcastFloat64x4(3.141592653589793)
	atan64_negPi     = archsimd.BroadcastFloat64x4(-3.141592653589793)
	atan64_tanPiOver8 = archsimd.BroadcastFloat64x4(0.4142135623730950488)
)

// Atan_AVX2_F32x8 computes atan(x) for a single Float32x8 vector.
//
// Algorithm uses two-level range reduction for better accuracy:
// 1. If |x| > 1: use atan(x) = π/2 - atan(1/x)
// 2. If |x| > tan(π/8) ≈ 0.414: use atan(x) = π/4 + atan((x-1)/(x+1))
//
// This reduces the argument to [0, tan(π/8)] where the polynomial is accurate.
//
// Special cases:
//   - atan(±0) = ±0
//   - atan(±Inf) = ±π/2
//   - atan(NaN) = NaN
func Atan_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	// Get absolute value and sign
	xBits := x.AsInt32x8()
	signBits := xBits.And(atan32_signMask)
	absXBits := xBits.And(atan32_absMask)
	absX := absXBits.AsFloat32x8()

	// Range reduction level 1: if |x| > 1, use atan(x) = π/2 - atan(1/x)
	useReciprocalMask := absX.Greater(atan32_one)
	recipAbsX := atan32_one.Div(absX)
	// Select: if useReciprocal then recipAbsX else absX
	reduced := absX.AsInt32x8().Merge(recipAbsX.AsInt32x8(), useReciprocalMask).AsFloat32x8()

	// Range reduction level 2: if reduced > tan(π/8), use atan(x) = π/4 + atan((x-1)/(x+1))
	useIdentityMask := reduced.Greater(atan32_tanPiOver8)
	xMinus1 := reduced.Sub(atan32_one)
	xPlus1 := reduced.Add(atan32_one)
	transformed := xMinus1.Div(xPlus1)
	// Select: if useIdentity then transformed else reduced
	reduced = reduced.AsInt32x8().Merge(transformed.AsInt32x8(), useIdentityMask).AsFloat32x8()

	// Compute polynomial: atan(z) ≈ z * (1 + z² * (c1 + z² * (c2 + ...)))
	// Now z is in [0, tan(π/8)] where the polynomial converges quickly
	z2 := reduced.Mul(reduced)

	// Horner's method from highest degree
	poly := atan32_c5.MulAdd(z2, atan32_c4)
	poly = poly.MulAdd(z2, atan32_c3)
	poly = poly.MulAdd(z2, atan32_c2)
	poly = poly.MulAdd(z2, atan32_c1)
	poly = poly.MulAdd(z2, atan32_one)
	atanCore := reduced.Mul(poly)

	// Adjust for identity transform: add π/4 if we used (x-1)/(x+1)
	atanWithIdentity := atan32_piOver4.Add(atanCore)
	atanReduced := atanCore.AsInt32x8().Merge(atanWithIdentity.AsInt32x8(), useIdentityMask).AsFloat32x8()

	// Adjust for reciprocal: if |x| > 1, result = π/2 - atan(1/|x|)
	atanWithReciprocal := atan32_piOver2.Sub(atanReduced)
	resultAbs := atanReduced.AsInt32x8().Merge(atanWithReciprocal.AsInt32x8(), useReciprocalMask).AsFloat32x8()

	// Restore sign
	resultBits := resultAbs.AsInt32x8().Or(signBits)
	return resultBits.AsFloat32x8()
}

// Atan_AVX2_F64x4 computes atan(x) for a single Float64x4 vector.
//
// Note: Uses scalar fallback for float64 to avoid AVX-512 dependency.
func Atan_AVX2_F64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	var in, out [4]float64
	x.StoreSlice(in[:])
	for i := range in {
		out[i] = stdmath.Atan(in[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}

// Atan2_AVX2_F32x8 computes atan2(y, x) for Float32x8 vectors.
//
// Uses two-level range reduction for accuracy:
// 1. Compute base = atan(|y/x|) using range-reduced polynomial
// 2. Apply quadrant correction based on signs of x and y
func Atan2_AVX2_F32x8(y, x archsimd.Float32x8) archsimd.Float32x8 {
	// Get masks for signs and zeros
	xZeroMask := x.Equal(atan32_zero)
	yZeroMask := y.Equal(atan32_zero)
	xNegMask := x.Less(atan32_zero)
	yNegMask := y.Less(atan32_zero)
	yPosMask := y.Greater(atan32_zero)

	// Safe division (replace x=0 with 1 temporarily)
	safeX := atan32_one.AsInt32x8().Merge(x.AsInt32x8(), xZeroMask).AsFloat32x8()
	ratio := y.Div(safeX)

	// Get sign and absolute value of ratio
	ratioBits := ratio.AsInt32x8()
	ratioSign := ratioBits.And(atan32_signMask)
	absRatio := ratioBits.And(atan32_absMask).AsFloat32x8()

	// Range reduction level 1: if |ratio| > 1, use atan(r) = π/2 - atan(1/r)
	useReciprocalMask := absRatio.Greater(atan32_one)
	recipRatio := atan32_one.Div(absRatio)
	reduced := absRatio.AsInt32x8().Merge(recipRatio.AsInt32x8(), useReciprocalMask).AsFloat32x8()

	// Range reduction level 2: if reduced > tan(π/8), use identity
	useIdentityMask := reduced.Greater(atan32_tanPiOver8)
	rMinus1 := reduced.Sub(atan32_one)
	rPlus1 := reduced.Add(atan32_one)
	transformed := rMinus1.Div(rPlus1)
	reduced = reduced.AsInt32x8().Merge(transformed.AsInt32x8(), useIdentityMask).AsFloat32x8()

	// Compute polynomial
	r2 := reduced.Mul(reduced)
	poly := atan32_c5.MulAdd(r2, atan32_c4)
	poly = poly.MulAdd(r2, atan32_c3)
	poly = poly.MulAdd(r2, atan32_c2)
	poly = poly.MulAdd(r2, atan32_c1)
	poly = poly.MulAdd(r2, atan32_one)
	atanCore := reduced.Mul(poly)

	// Adjust for identity transform
	atanWithIdentity := atan32_piOver4.Add(atanCore)
	atanReduced := atanCore.AsInt32x8().Merge(atanWithIdentity.AsInt32x8(), useIdentityMask).AsFloat32x8()

	// Adjust for reciprocal
	atanWithReciprocal := atan32_piOver2.Sub(atanReduced)
	atanAbs := atanReduced.AsInt32x8().Merge(atanWithReciprocal.AsInt32x8(), useReciprocalMask).AsFloat32x8()

	// Apply ratio sign
	atanVal := atanAbs.AsInt32x8().Or(ratioSign).AsFloat32x8()

	// Quadrant adjustment based on sign of x
	// x < 0, y >= 0: add π
	// x < 0, y < 0: subtract π
	yNonNegMask := y.GreaterEqual(atan32_zero)
	needAddPiMask := xNegMask.And(yNonNegMask)
	needSubPiMask := xNegMask.And(yNegMask)

	atanPlusPi := atanVal.Add(atan32_pi)
	atanMinusPi := atanVal.Add(atan32_negPi)

	atanVal = atanVal.AsInt32x8().Merge(atanPlusPi.AsInt32x8(), needAddPiMask).AsFloat32x8()
	atanVal = atanVal.AsInt32x8().Merge(atanMinusPi.AsInt32x8(), needSubPiMask).AsFloat32x8()

	// Handle x = 0 cases
	negPiOver2 := atan32_zero.Sub(atan32_piOver2)

	// x=0, y>0 -> π/2
	xZeroYPosMask := xZeroMask.And(yPosMask)
	atanVal = atan32_piOver2.AsInt32x8().Merge(atanVal.AsInt32x8(), xZeroYPosMask).AsFloat32x8()

	// x=0, y<0 -> -π/2
	xZeroYNegMask := xZeroMask.And(yNegMask)
	atanVal = negPiOver2.AsInt32x8().Merge(atanVal.AsInt32x8(), xZeroYNegMask).AsFloat32x8()

	// x=0, y=0 -> 0
	xZeroYZeroMask := xZeroMask.And(yZeroMask)
	atanVal = atan32_zero.AsInt32x8().Merge(atanVal.AsInt32x8(), xZeroYZeroMask).AsFloat32x8()

	return atanVal
}

// Atan2_AVX2_F64x4 computes atan2(y, x) for Float64x4 vectors.
//
// Note: Uses scalar fallback for float64 to avoid AVX-512 dependency.
func Atan2_AVX2_F64x4(y, x archsimd.Float64x4) archsimd.Float64x4 {
	var yIn, xIn, out [4]float64
	y.StoreSlice(yIn[:])
	x.StoreSlice(xIn[:])
	for i := range yIn {
		out[i] = stdmath.Atan2(yIn[i], xIn[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}
