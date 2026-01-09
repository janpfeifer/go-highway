//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
)

// AVX2 vectorized constants for atan32
var (
	// atan(x) polynomial coefficients for |x| <= 1
	// Using minimax polynomial approximation for atan(x)/x
	// atan(x) = x * (1 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸ + c5*x¹⁰ + c6*x¹² + c7*x¹⁴)
	atan32_c1 = archsimd.BroadcastFloat32x8(-0.3333333333)  // -1/3
	atan32_c2 = archsimd.BroadcastFloat32x8(0.2)            // 1/5
	atan32_c3 = archsimd.BroadcastFloat32x8(-0.1428571429)  // -1/7
	atan32_c4 = archsimd.BroadcastFloat32x8(0.1111111111)   // 1/9
	atan32_c5 = archsimd.BroadcastFloat32x8(-0.0909090909)  // -1/11
	atan32_c6 = archsimd.BroadcastFloat32x8(0.0769230769)   // 1/13
	atan32_c7 = archsimd.BroadcastFloat32x8(-0.0666666667)  // -1/15

	// Constants
	atan32_zero     = archsimd.BroadcastFloat32x8(0.0)
	atan32_one      = archsimd.BroadcastFloat32x8(1.0)
	atan32_negOne   = archsimd.BroadcastFloat32x8(-1.0)
	atan32_piOver2  = archsimd.BroadcastFloat32x8(1.5707963267948966)
	atan32_pi       = archsimd.BroadcastFloat32x8(3.141592653589793)
	atan32_negPi    = archsimd.BroadcastFloat32x8(-3.141592653589793)
	atan32_inf      = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(1)))
	atan32_negInf   = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(-1)))

	// Sign mask for float32 (0x80000000 as negative value to avoid overflow)
	atan32_signMask = archsimd.BroadcastInt32x8(-2147483648)
	atan32_absMask  = archsimd.BroadcastInt32x8(0x7FFFFFFF)
)

// AVX2 vectorized constants for atan64
var (
	atan64_zero    = archsimd.BroadcastFloat64x4(0.0)
	atan64_one     = archsimd.BroadcastFloat64x4(1.0)
	atan64_piOver2 = archsimd.BroadcastFloat64x4(1.5707963267948966)
	atan64_pi      = archsimd.BroadcastFloat64x4(3.141592653589793)
	atan64_negPi   = archsimd.BroadcastFloat64x4(-3.141592653589793)
	atan64_inf     = archsimd.BroadcastFloat64x4(stdmath.Inf(1))
	atan64_negInf  = archsimd.BroadcastFloat64x4(stdmath.Inf(-1))
)

// Atan_AVX2_F32x8 computes atan(x) for a single Float32x8 vector.
//
// Algorithm:
// For |x| <= 1: use polynomial approximation directly
// For |x| > 1: use identity atan(x) = π/2 - atan(1/x)
//
// The polynomial is a minimax approximation for atan(x) on [-1, 1]:
// atan(x) ≈ x * (1 - x²/3 + x⁴/5 - x⁶/7 + ...)
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

	// Check if |x| > 1
	largeXMask := absX.Greater(atan32_one)

	// For |x| > 1, compute 1/|x|; otherwise use |x|
	invAbsX := atan32_one.Div(absX)
	// Use merge: largeMask true -> invAbsX, otherwise absX
	z := invAbsX.AsInt32x8().Merge(absX.AsInt32x8(), largeXMask).AsFloat32x8()

	// Compute polynomial: atan(z) ≈ z * (1 + c1*z² + c2*z⁴ + ...)
	z2 := z.Mul(z)

	// Horner's method from the highest degree
	poly := atan32_c7.MulAdd(z2, atan32_c6)
	poly = poly.MulAdd(z2, atan32_c5)
	poly = poly.MulAdd(z2, atan32_c4)
	poly = poly.MulAdd(z2, atan32_c3)
	poly = poly.MulAdd(z2, atan32_c2)
	poly = poly.MulAdd(z2, atan32_c1)
	poly = poly.MulAdd(z2, atan32_one)
	atanZ := z.Mul(poly)

	// For |x| > 1: result = π/2 - atan(1/|x|)
	atanLarge := atan32_piOver2.Sub(atanZ)
	resultAbs := atanLarge.AsInt32x8().Merge(atanZ.AsInt32x8(), largeXMask).AsFloat32x8()

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
// Algorithm:
// 1. Handle special cases (x=0, infinities)
// 2. Compute base = atan(|y/x|)
// 3. Apply quadrant correction based on signs of x and y:
//   - x > 0: result = base * sign(y)
//   - x < 0: result = (π - base) * sign(y)
//   - x = 0, y > 0: result = π/2
//   - x = 0, y < 0: result = -π/2
//   - x = 0, y = 0: result = 0 (or ±π for -0)
func Atan2_AVX2_F32x8(y, x archsimd.Float32x8) archsimd.Float32x8 {
	// Get signs
	yBits := y.AsInt32x8()
	xBits := x.AsInt32x8()
	ySign := yBits.And(atan32_signMask)
	xSign := xBits.And(atan32_signMask)

	// Get absolute values
	absY := yBits.And(atan32_absMask).AsFloat32x8()
	absX := xBits.And(atan32_absMask).AsFloat32x8()

	// Compute |y/x| safely (handle x=0 later)
	ratio := absY.Div(absX)

	// Compute atan(|y/x|)
	base := Atan_AVX2_F32x8(ratio)

	// Determine quadrant adjustments
	xNegMask := x.Less(atan32_zero)
	xZeroMask := x.Equal(atan32_zero)
	yNegMask := y.Less(atan32_zero)
	yZeroMask := y.Equal(atan32_zero)
	yPosMask := y.Greater(atan32_zero)

	// For x < 0: result = π - base
	// For x > 0: result = base
	piMinusBase := atan32_pi.Sub(base)
	result := piMinusBase.AsInt32x8().Merge(base.AsInt32x8(), xNegMask).AsFloat32x8()

	// Apply y sign
	resultBits := result.AsInt32x8().Or(ySign)
	result = resultBits.AsFloat32x8()

	// Handle x = 0 cases:
	// x = 0, y > 0: π/2
	// x = 0, y < 0: -π/2
	// x = 0, y = 0: 0 (simplified; actual behavior depends on sign of zeros)
	negPiOver2 := atan32_zero.Sub(atan32_piOver2)

	// x=0, y>0 -> π/2
	xZeroYPosMask := xZeroMask.And(yPosMask)
	result = atan32_piOver2.Merge(result, xZeroYPosMask)

	// x=0, y<0 -> -π/2
	xZeroYNegMask := xZeroMask.And(yNegMask)
	result = negPiOver2.Merge(result, xZeroYNegMask)

	// x=0, y=0: handle based on sign of x
	// x=-0, y=±0 -> ±π
	// x=+0, y=±0 -> ±0
	xZeroYZeroMask := xZeroMask.And(yZeroMask)
	xNegZero := xSign.Equal(atan32_signMask)
	xZeroYZeroXNegMask := xZeroYZeroMask.And(xNegZero)
	piWithYSign := atan32_pi.AsInt32x8().Or(ySign).AsFloat32x8()
	zeroWithYSign := atan32_zero.AsInt32x8().Or(ySign).AsFloat32x8()
	result = piWithYSign.Merge(result, xZeroYZeroXNegMask)
	// For positive zero, xSign is 0, so compare against all-zeros
	zeroInt := archsimd.BroadcastInt32x8(0)
	xPosZero := xSign.Equal(zeroInt)
	xZeroYZeroXPosMask := xZeroYZeroMask.And(xPosZero)
	result = zeroWithYSign.Merge(result, xZeroYZeroXPosMask)

	return result
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
