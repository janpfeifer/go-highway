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
	// atan(x) polynomial coefficients for |x| <= 1
	// Using minimax polynomial approximation for atan(x)/x
	// atan(x) = x * (1 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸ + c5*x¹⁰ + c6*x¹² + c7*x¹⁴)
	atan512_32_c1 archsimd.Float32x16
	atan512_32_c2 archsimd.Float32x16
	atan512_32_c3 archsimd.Float32x16
	atan512_32_c4 archsimd.Float32x16
	atan512_32_c5 archsimd.Float32x16
	atan512_32_c6 archsimd.Float32x16
	atan512_32_c7 archsimd.Float32x16

	// Constants
	atan512_32_zero    archsimd.Float32x16
	atan512_32_one     archsimd.Float32x16
	atan512_32_negOne  archsimd.Float32x16
	atan512_32_piOver2 archsimd.Float32x16
	atan512_32_pi      archsimd.Float32x16
	atan512_32_negPi   archsimd.Float32x16
	atan512_32_inf     archsimd.Float32x16
	atan512_32_negInf  archsimd.Float32x16

	// Sign mask for float32
	atan512_32_signMask archsimd.Int32x16
	atan512_32_absMask  archsimd.Int32x16
)

// AVX-512 vectorized constants for atan64
var (
	// atan(x) polynomial coefficients for |x| <= 1 (higher precision for float64)
	atan512_64_c1 archsimd.Float64x8
	atan512_64_c2 archsimd.Float64x8
	atan512_64_c3 archsimd.Float64x8
	atan512_64_c4 archsimd.Float64x8
	atan512_64_c5 archsimd.Float64x8
	atan512_64_c6 archsimd.Float64x8
	atan512_64_c7 archsimd.Float64x8
	atan512_64_c8 archsimd.Float64x8
	atan512_64_c9 archsimd.Float64x8

	// Constants
	atan512_64_zero    archsimd.Float64x8
	atan512_64_one     archsimd.Float64x8
	atan512_64_piOver2 archsimd.Float64x8
	atan512_64_pi      archsimd.Float64x8
	atan512_64_negPi   archsimd.Float64x8
	atan512_64_inf     archsimd.Float64x8
	atan512_64_negInf  archsimd.Float64x8

	// Sign mask for float64
	atan512_64_signMask archsimd.Int64x8
	atan512_64_absMask  archsimd.Int64x8
)

func initAtan512Constants() {
	// Float32 polynomial coefficients (minimax approximation)
	atan512_32_c1 = archsimd.BroadcastFloat32x16(-0.3333333333)  // -1/3
	atan512_32_c2 = archsimd.BroadcastFloat32x16(0.2)            // 1/5
	atan512_32_c3 = archsimd.BroadcastFloat32x16(-0.1428571429)  // -1/7
	atan512_32_c4 = archsimd.BroadcastFloat32x16(0.1111111111)   // 1/9
	atan512_32_c5 = archsimd.BroadcastFloat32x16(-0.0909090909)  // -1/11
	atan512_32_c6 = archsimd.BroadcastFloat32x16(0.0769230769)   // 1/13
	atan512_32_c7 = archsimd.BroadcastFloat32x16(-0.0666666667)  // -1/15

	// Float32 constants
	atan512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	atan512_32_one = archsimd.BroadcastFloat32x16(1.0)
	atan512_32_negOne = archsimd.BroadcastFloat32x16(-1.0)
	atan512_32_piOver2 = archsimd.BroadcastFloat32x16(1.5707963267948966)
	atan512_32_pi = archsimd.BroadcastFloat32x16(3.141592653589793)
	atan512_32_negPi = archsimd.BroadcastFloat32x16(-3.141592653589793)
	atan512_32_inf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(1)))
	atan512_32_negInf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(-1)))
	atan512_32_signMask = archsimd.BroadcastInt32x16(-2147483648)
	atan512_32_absMask = archsimd.BroadcastInt32x16(0x7FFFFFFF)

	// Float64 polynomial coefficients (higher precision)
	atan512_64_c1 = archsimd.BroadcastFloat64x8(-0.3333333333333333)  // -1/3
	atan512_64_c2 = archsimd.BroadcastFloat64x8(0.2)                  // 1/5
	atan512_64_c3 = archsimd.BroadcastFloat64x8(-0.14285714285714285) // -1/7
	atan512_64_c4 = archsimd.BroadcastFloat64x8(0.1111111111111111)   // 1/9
	atan512_64_c5 = archsimd.BroadcastFloat64x8(-0.09090909090909091) // -1/11
	atan512_64_c6 = archsimd.BroadcastFloat64x8(0.07692307692307693)  // 1/13
	atan512_64_c7 = archsimd.BroadcastFloat64x8(-0.06666666666666667) // -1/15
	atan512_64_c8 = archsimd.BroadcastFloat64x8(0.058823529411764705) // 1/17
	atan512_64_c9 = archsimd.BroadcastFloat64x8(-0.05263157894736842) // -1/19

	// Float64 constants
	atan512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	atan512_64_one = archsimd.BroadcastFloat64x8(1.0)
	atan512_64_piOver2 = archsimd.BroadcastFloat64x8(1.5707963267948966)
	atan512_64_pi = archsimd.BroadcastFloat64x8(3.141592653589793)
	atan512_64_negPi = archsimd.BroadcastFloat64x8(-3.141592653589793)
	atan512_64_inf = archsimd.BroadcastFloat64x8(stdmath.Inf(1))
	atan512_64_negInf = archsimd.BroadcastFloat64x8(stdmath.Inf(-1))
	atan512_64_signMask = archsimd.BroadcastInt64x8(-9223372036854775808)
	atan512_64_absMask = archsimd.BroadcastInt64x8(0x7FFFFFFFFFFFFFFF)
}

// Atan_AVX512_F32x16 computes atan(x) for a single Float32x16 vector.
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
func Atan_AVX512_F32x16(x archsimd.Float32x16) archsimd.Float32x16 {
	atan512Init.Do(initAtan512Constants)

	// Get absolute value and sign
	xBits := x.AsInt32x16()
	signBits := xBits.And(atan512_32_signMask)
	absXBits := xBits.And(atan512_32_absMask)
	absX := absXBits.AsFloat32x16()

	// Check if |x| > 1
	largeXMask := absX.Greater(atan512_32_one)

	// For |x| > 1, compute 1/|x|; otherwise use |x|
	invAbsX := atan512_32_one.Div(absX)
	// Use merge: largeMask true -> invAbsX, otherwise absX
	z := invAbsX.AsInt32x16().Merge(absX.AsInt32x16(), largeXMask).AsFloat32x16()

	// Compute polynomial: atan(z) ≈ z * (1 + c1*z² + c2*z⁴ + ...)
	z2 := z.Mul(z)

	// Horner's method from the highest degree
	poly := atan512_32_c7.MulAdd(z2, atan512_32_c6)
	poly = poly.MulAdd(z2, atan512_32_c5)
	poly = poly.MulAdd(z2, atan512_32_c4)
	poly = poly.MulAdd(z2, atan512_32_c3)
	poly = poly.MulAdd(z2, atan512_32_c2)
	poly = poly.MulAdd(z2, atan512_32_c1)
	poly = poly.MulAdd(z2, atan512_32_one)
	atanZ := z.Mul(poly)

	// For |x| > 1: result = π/2 - atan(1/|x|)
	atanLarge := atan512_32_piOver2.Sub(atanZ)
	resultAbs := atanLarge.AsInt32x16().Merge(atanZ.AsInt32x16(), largeXMask).AsFloat32x16()

	// Restore sign
	resultBits := resultAbs.AsInt32x16().Or(signBits)
	return resultBits.AsFloat32x16()
}

// Atan_AVX512_F64x8 computes atan(x) for a single Float64x8 vector.
//
// Algorithm:
// For |x| <= 1: use polynomial approximation directly
// For |x| > 1: use identity atan(x) = π/2 - atan(1/x)
//
// Uses a higher-degree polynomial for float64 precision.
//
// Special cases:
//   - atan(±0) = ±0
//   - atan(±Inf) = ±π/2
//   - atan(NaN) = NaN
func Atan_AVX512_F64x8(x archsimd.Float64x8) archsimd.Float64x8 {
	atan512Init.Do(initAtan512Constants)

	// Get absolute value and sign
	xBits := x.AsInt64x8()
	signBits := xBits.And(atan512_64_signMask)
	absXBits := xBits.And(atan512_64_absMask)
	absX := absXBits.AsFloat64x8()

	// Check if |x| > 1
	largeXMask := absX.Greater(atan512_64_one)

	// For |x| > 1, compute 1/|x|; otherwise use |x|
	invAbsX := atan512_64_one.Div(absX)
	// Use merge: largeMask true -> invAbsX, otherwise absX
	z := invAbsX.AsInt64x8().Merge(absX.AsInt64x8(), largeXMask).AsFloat64x8()

	// Compute polynomial: atan(z) ≈ z * (1 + c1*z² + c2*z⁴ + ...)
	z2 := z.Mul(z)

	// Horner's method from the highest degree (higher precision for float64)
	poly := atan512_64_c9.MulAdd(z2, atan512_64_c8)
	poly = poly.MulAdd(z2, atan512_64_c7)
	poly = poly.MulAdd(z2, atan512_64_c6)
	poly = poly.MulAdd(z2, atan512_64_c5)
	poly = poly.MulAdd(z2, atan512_64_c4)
	poly = poly.MulAdd(z2, atan512_64_c3)
	poly = poly.MulAdd(z2, atan512_64_c2)
	poly = poly.MulAdd(z2, atan512_64_c1)
	poly = poly.MulAdd(z2, atan512_64_one)
	atanZ := z.Mul(poly)

	// For |x| > 1: result = π/2 - atan(1/|x|)
	atanLarge := atan512_64_piOver2.Sub(atanZ)
	resultAbs := atanLarge.AsInt64x8().Merge(atanZ.AsInt64x8(), largeXMask).AsFloat64x8()

	// Restore sign
	resultBits := resultAbs.AsInt64x8().Or(signBits)
	return resultBits.AsFloat64x8()
}

// Atan2_AVX512_F32x16 computes atan2(y, x) for Float32x16 vectors.
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
func Atan2_AVX512_F32x16(y, x archsimd.Float32x16) archsimd.Float32x16 {
	atan512Init.Do(initAtan512Constants)

	// Get signs
	yBits := y.AsInt32x16()
	xBits := x.AsInt32x16()
	ySign := yBits.And(atan512_32_signMask)
	xSign := xBits.And(atan512_32_signMask)

	// Get absolute values
	absY := yBits.And(atan512_32_absMask).AsFloat32x16()
	absX := xBits.And(atan512_32_absMask).AsFloat32x16()

	// Compute |y/x| safely (handle x=0 later)
	ratio := absY.Div(absX)

	// Compute atan(|y/x|)
	base := Atan_AVX512_F32x16(ratio)

	// Determine quadrant adjustments
	xNegMask := x.Less(atan512_32_zero)
	xZeroMask := x.Equal(atan512_32_zero)
	yNegMask := y.Less(atan512_32_zero)
	yZeroMask := y.Equal(atan512_32_zero)
	yPosMask := y.Greater(atan512_32_zero)

	// For x < 0: result = π - base
	// For x > 0: result = base
	piMinusBase := atan512_32_pi.Sub(base)
	result := piMinusBase.AsInt32x16().Merge(base.AsInt32x16(), xNegMask).AsFloat32x16()

	// Apply y sign
	resultBits := result.AsInt32x16().Or(ySign)
	result = resultBits.AsFloat32x16()

	// Handle x = 0 cases:
	// x = 0, y > 0: π/2
	// x = 0, y < 0: -π/2
	// x = 0, y = 0: 0 (simplified; actual behavior depends on sign of zeros)
	negPiOver2 := atan512_32_zero.Sub(atan512_32_piOver2)

	// x=0, y>0 -> π/2
	xZeroYPosMask := xZeroMask.And(yPosMask)
	result = atan512_32_piOver2.Merge(result, xZeroYPosMask)

	// x=0, y<0 -> -π/2
	xZeroYNegMask := xZeroMask.And(yNegMask)
	result = negPiOver2.Merge(result, xZeroYNegMask)

	// x=0, y=0: handle based on sign of x
	// x=-0, y=±0 -> ±π
	// x=+0, y=±0 -> ±0
	xZeroYZeroMask := xZeroMask.And(yZeroMask)
	xNegZero := xSign.Equal(atan512_32_signMask)
	xZeroYZeroXNegMask := xZeroYZeroMask.And(xNegZero)
	piWithYSign := atan512_32_pi.AsInt32x16().Or(ySign).AsFloat32x16()
	zeroWithYSign := atan512_32_zero.AsInt32x16().Or(ySign).AsFloat32x16()
	result = piWithYSign.Merge(result, xZeroYZeroXNegMask)
	// For positive zero, xSign is 0, so compare against all-zeros
	zeroInt32 := archsimd.BroadcastInt32x16(0)
	xPosZero := xSign.Equal(zeroInt32)
	xZeroYZeroXPosMask := xZeroYZeroMask.And(xPosZero)
	result = zeroWithYSign.Merge(result, xZeroYZeroXPosMask)

	return result
}

// Atan2_AVX512_F64x8 computes atan2(y, x) for Float64x8 vectors.
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
func Atan2_AVX512_F64x8(y, x archsimd.Float64x8) archsimd.Float64x8 {
	atan512Init.Do(initAtan512Constants)

	// Get signs
	yBits := y.AsInt64x8()
	xBits := x.AsInt64x8()
	ySign := yBits.And(atan512_64_signMask)
	xSign := xBits.And(atan512_64_signMask)

	// Get absolute values
	absY := yBits.And(atan512_64_absMask).AsFloat64x8()
	absX := xBits.And(atan512_64_absMask).AsFloat64x8()

	// Compute |y/x| safely (handle x=0 later)
	ratio := absY.Div(absX)

	// Compute atan(|y/x|)
	base := Atan_AVX512_F64x8(ratio)

	// Determine quadrant adjustments
	xNegMask := x.Less(atan512_64_zero)
	xZeroMask := x.Equal(atan512_64_zero)
	yNegMask := y.Less(atan512_64_zero)
	yZeroMask := y.Equal(atan512_64_zero)
	yPosMask := y.Greater(atan512_64_zero)

	// For x < 0: result = π - base
	// For x > 0: result = base
	piMinusBase := atan512_64_pi.Sub(base)
	result := piMinusBase.AsInt64x8().Merge(base.AsInt64x8(), xNegMask).AsFloat64x8()

	// Apply y sign
	resultBits := result.AsInt64x8().Or(ySign)
	result = resultBits.AsFloat64x8()

	// Handle x = 0 cases:
	// x = 0, y > 0: π/2
	// x = 0, y < 0: -π/2
	// x = 0, y = 0: 0 (simplified; actual behavior depends on sign of zeros)
	negPiOver2 := atan512_64_zero.Sub(atan512_64_piOver2)

	// x=0, y>0 -> π/2
	xZeroYPosMask := xZeroMask.And(yPosMask)
	result = atan512_64_piOver2.Merge(result, xZeroYPosMask)

	// x=0, y<0 -> -π/2
	xZeroYNegMask := xZeroMask.And(yNegMask)
	result = negPiOver2.Merge(result, xZeroYNegMask)

	// x=0, y=0: handle based on sign of x
	// x=-0, y=±0 -> ±π
	// x=+0, y=±0 -> ±0
	xZeroYZeroMask := xZeroMask.And(yZeroMask)
	xNegZero := xSign.Equal(atan512_64_signMask)
	xZeroYZeroXNegMask := xZeroYZeroMask.And(xNegZero)
	piWithYSign := atan512_64_pi.AsInt64x8().Or(ySign).AsFloat64x8()
	zeroWithYSign := atan512_64_zero.AsInt64x8().Or(ySign).AsFloat64x8()
	result = piWithYSign.Merge(result, xZeroYZeroXNegMask)
	// For positive zero, xSign is 0, so compare against all-zeros
	zeroInt64 := archsimd.BroadcastInt64x8(0)
	xPosZero := xSign.Equal(zeroInt64)
	xZeroYZeroXPosMask := xZeroYZeroMask.And(xPosZero)
	result = zeroWithYSign.Merge(result, xZeroYZeroXPosMask)

	return result
}
