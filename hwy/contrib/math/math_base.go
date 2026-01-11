//go:generate hwygen -input math_base.go -output . -targets avx2,avx512,neon,fallback

package math

import (
	"math"

	"github.com/ajroetker/go-highway/hwy"
)

// =============================================================================
// Constants for mathematical functions
// =============================================================================

// Float32 constants for Exp
var (
	expLn2Hi_f32  float32 = 0.693359375
	expLn2Lo_f32  float32 = -2.12194440e-4
	expInvLn2_f32 float32 = 1.44269504088896341

	// Overflow/underflow thresholds
	expOverflow_f32  float32 = 88.72283905206835
	expUnderflow_f32 float32 = -87.33654475055310

	// Polynomial coefficients for exp(r) on [-ln(2)/2, ln(2)/2]
	// Taylor series: 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6!
	expC1_f32 float32 = 1.0
	expC2_f32 float32 = 0.5
	expC3_f32 float32 = 0.16666666666666666
	expC4_f32 float32 = 0.041666666666666664
	expC5_f32 float32 = 0.008333333333333333
	expC6_f32 float32 = 0.001388888888888889

	expOne_f32  float32 = 1.0
	expZero_f32 float32 = 0.0
)

// Float64 constants for Exp
var (
	expLn2Hi_f64  float64 = 0.6931471803691238
	expLn2Lo_f64  float64 = 1.9082149292705877e-10
	expInvLn2_f64 float64 = 1.4426950408889634

	expOverflow_f64  float64 = 709.782712893384
	expUnderflow_f64 float64 = -708.3964185322641

	// Higher-degree polynomial for float64
	expC1_f64  float64 = 1.0
	expC2_f64  float64 = 0.5
	expC3_f64  float64 = 0.16666666666666666
	expC4_f64  float64 = 0.041666666666666664
	expC5_f64  float64 = 0.008333333333333333
	expC6_f64  float64 = 0.001388888888888889
	expC7_f64  float64 = 0.0001984126984126984
	expC8_f64  float64 = 2.48015873015873e-05
	expC9_f64  float64 = 2.7557319223985893e-06
	expC10_f64 float64 = 2.755731922398589e-07

	expOne_f64  float64 = 1.0
	expZero_f64 float64 = 0.0
)

// Float32 constants for Log
var (
	// Polynomial coefficients for atanh-based log approximation
	// log(m) = 2*atanh((m-1)/(m+1)) = 2*y*(1 + y²/3 + y⁴/5 + y⁶/7 + ...)
	// poly = c1 + c2*y² + c3*y⁴ + c4*y⁶ + c5*y⁸
	logC1_f32 float32 = 1.0                   // constant term
	logC2_f32 float32 = 0.3333333333333367565 // 1/3
	logC3_f32 float32 = 0.1999999999970470954 // 1/5
	logC4_f32 float32 = 0.1428571437183119574 // 1/7
	logC5_f32 float32 = 0.1111109921607489198 // 1/9

	logLn2Hi_f32    float32 = 0.693359375
	logLn2Lo_f32    float32 = -2.12194440e-4
	logOne_f32      float32 = 1.0
	logTwo_f32      float32 = 2.0
	logSqrtHalf_f32 float32 = 0.7071067811865476
)

// Float64 constants for Log
var (
	// Polynomial coefficients for atanh-based log approximation
	// log(m) = 2*atanh((m-1)/(m+1)) = 2*y*(1 + y²/3 + y⁴/5 + y⁶/7 + ...)
	// poly = c1 + c2*y² + c3*y⁴ + c4*y⁶ + c5*y⁸ + c6*y¹⁰ + c7*y¹²
	logC1_f64 float64 = 1.0                   // constant term
	logC2_f64 float64 = 0.3333333333333367565 // 1/3
	logC3_f64 float64 = 0.1999999999970470954 // 1/5
	logC4_f64 float64 = 0.1428571437183119574 // 1/7
	logC5_f64 float64 = 0.1111109921607489198 // 1/9
	logC6_f64 float64 = 0.0909178608080902506 // 1/11
	logC7_f64 float64 = 0.0765691884960468666 // 1/13

	logLn2Hi_f64    float64 = 0.6931471803691238
	logLn2Lo_f64    float64 = 1.9082149292705877e-10
	logOne_f64      float64 = 1.0
	logTwo_f64      float64 = 2.0
	logSqrtHalf_f64 float64 = 0.7071067811865476
)

// Float32 constants for Trig (Sin, Cos)
var (
	trig2OverPi_f32   float32 = 0.6366197723675814     // 2/π
	trigPiOver2Hi_f32 float32 = 1.5707963267948966     // π/2 high
	trigPiOver2Lo_f32 float32 = 6.123233995736766e-17  // π/2 low

	// sin(x) polynomial coefficients for |x| <= π/4
	trigS1_f32 float32 = -0.16666666641626524     // -1/3!
	trigS2_f32 float32 = 0.008333329385889463     // 1/5!
	trigS3_f32 float32 = -0.00019839334836096632  // -1/7!
	trigS4_f32 float32 = 2.718311493989822e-6     // 1/9!

	// cos(x) polynomial coefficients for |x| <= π/4
	trigC1_f32 float32 = -0.4999999963229337      // -1/2!
	trigC2_f32 float32 = 0.04166662453689337      // 1/4!
	trigC3_f32 float32 = -0.001388731625493765    // -1/6!
	trigC4_f32 float32 = 2.443315711809948e-5     // 1/8!

	trigOne_f32    float32 = 1.0
	trigNegOne_f32 float32 = -1.0
	trigZero_f32   float32 = 0.0
)

// Float64 constants for Trig
var (
	trig2OverPi_f64   float64 = 0.6366197723675814
	trigPiOver2Hi_f64 float64 = 1.5707963267948966192313216916398
	trigPiOver2Lo_f64 float64 = 6.123233995736766035868820147292e-17

	trigS1_f64 float64 = -0.16666666666666632
	trigS2_f64 float64 = 0.008333333333332249
	trigS3_f64 float64 = -0.00019841269840885721
	trigS4_f64 float64 = 2.7557316103728803e-6
	trigS5_f64 float64 = -2.5051132068021698e-8
	trigS6_f64 float64 = 1.5896230157221844e-10

	trigC1_f64 float64 = -0.5
	trigC2_f64 float64 = 0.04166666666666621
	trigC3_f64 float64 = -0.001388888888887411
	trigC4_f64 float64 = 2.4801587288851704e-5
	trigC5_f64 float64 = -2.7557314351390663e-7
	trigC6_f64 float64 = 2.0875723212981748e-9

	trigOne_f64    float64 = 1.0
	trigNegOne_f64 float64 = -1.0
	trigZero_f64   float64 = 0.0
)

// Float32 constants for Tanh
var (
	tanhClamp_f32  float32 = 9.0
	tanhOne_f32    float32 = 1.0
	tanhNegOne_f32 float32 = -1.0

	tanhC1_f32 float32 = 1.0
	tanhC3_f32 float32 = -0.3333333333333333
	tanhC5_f32 float32 = 0.13333333333333333
)

// Float64 constants for Tanh
var (
	tanhClamp_f64  float64 = 19.0
	tanhOne_f64    float64 = 1.0
	tanhNegOne_f64 float64 = -1.0

	tanhC1_f64 float64 = 1.0
	tanhC3_f64 float64 = -0.3333333333333333
	tanhC5_f64 float64 = 0.13333333333333333
	tanhC7_f64 float64 = -0.05396825396825397
	tanhC9_f64 float64 = 0.021869488536155203
)

// Float32 constants for Sigmoid
var (
	sigmoidOne_f32     float32 = 1.0
	sigmoidHalf_f32    float32 = 0.5
	sigmoidQuarter_f32 float32 = 0.25
	sigmoidNegOne_f32  float32 = -1.0
)

// Float64 constants for Sigmoid
var (
	sigmoidOne_f64     float64 = 1.0
	sigmoidHalf_f64    float64 = 0.5
	sigmoidQuarter_f64 float64 = 0.25
	sigmoidNegOne_f64  float64 = -1.0
)

// Float32 constants for Atan
var (
	atanPiOver2_f32    float32 = 1.5707963267948966
	atanPiOver4_f32    float32 = 0.7853981633974483
	atanPi_f32         float32 = 3.141592653589793
	atanTanPiOver8_f32 float32 = 0.4142135623730950488 // tan(π/8) = sqrt(2) - 1

	atanC1_f32 float32 = -0.3333333333
	atanC2_f32 float32 = 0.2
	atanC3_f32 float32 = -0.1428571429
	atanC4_f32 float32 = 0.1111111111
	atanC5_f32 float32 = -0.0909090909

	atanOne_f32  float32 = 1.0
	atanZero_f32 float32 = 0.0
)

// Float64 constants for Atan
var (
	atanPiOver2_f64    float64 = 1.5707963267948966
	atanPiOver4_f64    float64 = 0.7853981633974483
	atanPi_f64         float64 = 3.141592653589793
	atanTanPiOver8_f64 float64 = 0.4142135623730950488

	atanC1_f64 float64 = -0.3333333333
	atanC2_f64 float64 = 0.2
	atanC3_f64 float64 = -0.1428571429
	atanC4_f64 float64 = 0.1111111111
	atanC5_f64 float64 = -0.0909090909

	atanOne_f64  float64 = 1.0
	atanZero_f64 float64 = 0.0
)

// Float32 constants for Asin/Acos
var (
	asinPiOver2_f32 float32 = 1.5707963267948966
	asinHalf_f32    float32 = 0.5
	asinTwo_f32     float32 = 2.0

	asinP1_f32 float32 = 0.16666666666666666
	asinP2_f32 float32 = 0.075
	asinP3_f32 float32 = 0.04464285714285714
	asinP4_f32 float32 = 0.030381944444444444
	asinP5_f32 float32 = 0.022372159090909092
	asinP6_f32 float32 = 0.017352764423076923

	asinOne_f32     float32 = 1.0
	asinNegOne_f32  float32 = -1.0
	asinZero_f32    float32 = 0.0
)

// Float64 constants for Asin/Acos
var (
	asinPiOver2_f64 float64 = 1.5707963267948966192313216916398
	asinHalf_f64    float64 = 0.5
	asinTwo_f64     float64 = 2.0

	asinP1_f64 float64 = 0.16666666666666666
	asinP2_f64 float64 = 0.075
	asinP3_f64 float64 = 0.04464285714285714
	asinP4_f64 float64 = 0.030381944444444444
	asinP5_f64 float64 = 0.022372159090909092
	asinP6_f64 float64 = 0.017352764423076923

	asinOne_f64     float64 = 1.0
	asinNegOne_f64  float64 = -1.0
	asinZero_f64    float64 = 0.0
)

// =============================================================================
// Mathematical Functions
// =============================================================================

// BaseExpPoly computes e^x using full-range polynomial approximation with
// proper range reduction and IEEE 754 reconstruction.
//
// Algorithm:
// 1. Range reduction: x = k*ln(2) + r, where |r| <= ln(2)/2
// 2. Polynomial approximation: e^r ≈ 1 + r + r²/2! + r³/3! + ...
// 3. Reconstruction: e^x = 2^k * e^r using IEEE 754 bit manipulation
func BaseExpPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	overflow := hwy.Set[T](T(expOverflow_f32))
	underflow := hwy.Set[T](T(expUnderflow_f32))
	one := hwy.Set[T](T(expOne_f32))
	zero := hwy.Set[T](T(expZero_f32))
	inf := hwy.Set[T](T(expOverflow_f32 * 2)) // Will become Inf
	invLn2 := hwy.Set[T](T(expInvLn2_f32))
	ln2Hi := hwy.Set[T](T(expLn2Hi_f32))
	ln2Lo := hwy.Set[T](T(expLn2Lo_f32))

	c1 := hwy.Set[T](T(expC1_f32))
	c2 := hwy.Set[T](T(expC2_f32))
	c3 := hwy.Set[T](T(expC3_f32))
	c4 := hwy.Set[T](T(expC4_f32))
	c5 := hwy.Set[T](T(expC5_f32))
	c6 := hwy.Set[T](T(expC6_f32))

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Check overflow/underflow
		overflowMask := hwy.Greater(x, overflow)
		underflowMask := hwy.Less(x, underflow)

		// Range reduction: k = round(x / ln(2)), r = x - k * ln(2)
		kFloat := hwy.RoundToEven(hwy.Mul(x, invLn2))

		// r = x - k*ln(2) using high/low split for precision
		r := hwy.Sub(x, hwy.Mul(kFloat, ln2Hi))
		r = hwy.Sub(r, hwy.Mul(kFloat, ln2Lo))

		// Polynomial approximation using Horner's method
		// p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
		p := hwy.MulAdd(c6, r, c5)
		p = hwy.MulAdd(p, r, c4)
		p = hwy.MulAdd(p, r, c3)
		p = hwy.MulAdd(p, r, c2)
		p = hwy.MulAdd(p, r, c1)
		p = hwy.MulAdd(p, r, one)

		// Scale by 2^k using IEEE 754 bit manipulation
		// Convert k to integer for Pow2
		kInt := hwy.ConvertToInt32(kFloat)
		scale := hwy.Pow2[T](kInt)
		result := hwy.Mul(p, scale)

		// Handle special cases
		result = hwy.Merge(inf, result, overflowMask)
		result = hwy.Merge(zero, result, underflowMask)

		hwy.Store(result, output[ii:])
	}
}

// BaseTanhPoly computes tanh(x) using the exp-based formula.
// Algorithm: tanh(x) = 2*sigmoid(2x) - 1
// For large |x|, tanh saturates to ±1.
func BaseTanhPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	two := hwy.Set[T](T(2.0))
	one := hwy.Set[T](T(tanhOne_f32))
	negOne := hwy.Set[T](T(tanhNegOne_f32))
	threshold := hwy.Set[T](T(tanhClamp_f32))
	negThreshold := hwy.Neg(threshold)

	lanes := one.NumLanes()

	// Allocate temporary buffers for sigmoid computation
	tempIn := make([]T, lanes)
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// tanh(x) = 2*sigmoid(2x) - 1
		twoX := hwy.Mul(two, x)
		hwy.Store(twoX, tempIn)
		BaseSigmoidPoly(tempIn, tempOut)
		sigTwoX := hwy.Load(tempOut)
		result := hwy.Sub(hwy.Mul(two, sigTwoX), one)

		// Handle saturation: for x > threshold, tanh ≈ 1; for x < -threshold, tanh ≈ -1
		result = hwy.Merge(one, result, hwy.Greater(x, threshold))
		result = hwy.Merge(negOne, result, hwy.Less(x, negThreshold))

		hwy.Store(result, output[ii:])
	}
}

// BaseSigmoidPoly computes sigmoid(x) = 1 / (1 + e^(-x)) using the exp function.
// Algorithm: sigmoid(x) = 1 / (1 + exp(-x))
// For numerical stability, we clamp x to avoid exp overflow.
func BaseSigmoidPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(sigmoidOne_f32))
	zero := hwy.Set[T](T(0.0))
	satHi := hwy.Set[T](T(20.0)) // sigmoid saturates to 1 for x > 20
	satLo := hwy.Set[T](T(-20.0)) // sigmoid saturates to 0 for x < -20

	lanes := one.NumLanes()

	// Allocate temporary buffers for exp computation
	tempIn := make([]T, lanes)
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Clamp to avoid exp overflow
		clampedX := hwy.Max(hwy.Min(x, satHi), satLo)

		// Compute exp(-x)
		negX := hwy.Neg(clampedX)
		hwy.Store(negX, tempIn)
		BaseExpPoly(tempIn, tempOut)
		expNegX := hwy.Load(tempOut)

		// sigmoid(x) = 1 / (1 + exp(-x))
		result := hwy.Div(one, hwy.Add(one, expNegX))

		// Handle saturation
		result = hwy.Merge(one, result, hwy.Greater(x, satHi))
		result = hwy.Merge(zero, result, hwy.Less(x, satLo))

		hwy.Store(result, output[ii:])
	}
}

// BaseSinPoly computes sin(x) using polynomial approximation.
// Uses Cody-Waite range reduction with proper integer octant selection.
//
// Algorithm:
// 1. Range reduction: k = round(x * 2/π), r = x - k*(π/2)
// 2. Compute sin(r) and cos(r) polynomials
// 3. Select based on quadrant: k mod 4
//   - 0: sin(r)
//   - 1: cos(r)
//   - 2: -sin(r)
//   - 3: -cos(r)
func BaseSinPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	twoOverPi := hwy.Set[T](T(trig2OverPi_f32))
	piOver2Hi := hwy.Set[T](T(trigPiOver2Hi_f32))
	piOver2Lo := hwy.Set[T](T(trigPiOver2Lo_f32))
	one := hwy.Set[T](T(trigOne_f32))
	s1 := hwy.Set[T](T(trigS1_f32))
	s2 := hwy.Set[T](T(trigS2_f32))
	s3 := hwy.Set[T](T(trigS3_f32))
	s4 := hwy.Set[T](T(trigS4_f32))
	c1 := hwy.Set[T](T(trigC1_f32))
	c2 := hwy.Set[T](T(trigC2_f32))
	c3 := hwy.Set[T](T(trigC3_f32))
	c4 := hwy.Set[T](T(trigC4_f32))

	// Integer constants for octant selection
	intOne := hwy.Set[int32](1)
	intTwo := hwy.Set[int32](2)
	intThree := hwy.Set[int32](3)

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Range reduction: k = round(x * 2/π)
		kFloat := hwy.RoundToEven(hwy.Mul(x, twoOverPi))
		kInt := hwy.ConvertToInt32(kFloat)

		// r = x - k * (π/2) using Cody-Waite high/low for precision
		r := hwy.Sub(x, hwy.Mul(kFloat, piOver2Hi))
		r = hwy.Sub(r, hwy.Mul(kFloat, piOver2Lo))
		r2 := hwy.Mul(r, r)

		// Compute sin(r) polynomial: sin(r) ≈ r * (1 + s1*r² + s2*r⁴ + s3*r⁶ + s4*r⁸)
		sinPoly := hwy.MulAdd(s4, r2, s3)
		sinPoly = hwy.MulAdd(sinPoly, r2, s2)
		sinPoly = hwy.MulAdd(sinPoly, r2, s1)
		sinPoly = hwy.MulAdd(sinPoly, r2, one)
		sinR := hwy.Mul(r, sinPoly)

		// Compute cos(r) polynomial: cos(r) ≈ 1 + c1*r² + c2*r⁴ + c3*r⁶ + c4*r⁸
		cosPoly := hwy.MulAdd(c4, r2, c3)
		cosPoly = hwy.MulAdd(cosPoly, r2, c2)
		cosPoly = hwy.MulAdd(cosPoly, r2, c1)
		cosR := hwy.MulAdd(cosPoly, r2, one)

		// Octant selection based on k mod 4 (using integer operations)
		octant := hwy.And(kInt, intThree)

		// Determine if we should use cos instead of sin (octant & 1 == 1)
		useCosMask := hwy.Equal(hwy.And(octant, intOne), intOne)
		// Determine if we should negate (octant & 2 == 2)
		negateMask := hwy.Equal(hwy.And(octant, intTwo), intTwo)

		// Select between sin(r) and cos(r) based on octant
		// We need to convert int32 masks to float masks by expanding element-wise
		sinRData := sinR.Data()
		cosRData := cosR.Data()
		resultData := make([]T, len(sinRData))
		for i := range sinRData {
			if useCosMask.GetBit(i) {
				resultData[i] = cosRData[i]
			} else {
				resultData[i] = sinRData[i]
			}
		}
		result := hwy.Load(resultData)

		// Negate if needed (octant 2 or 3)
		negResult := hwy.Neg(result)
		negResultData := negResult.Data()
		for i := range resultData {
			if negateMask.GetBit(i) {
				resultData[i] = negResultData[i]
			}
		}
		result = hwy.Load(resultData)

		hwy.Store(result, output[ii:])
	}
}

// BaseCosPoly computes cos(x) using polynomial approximation.
// Uses Cody-Waite range reduction with proper integer octant selection.
//
// Algorithm: cos(x) = sin(x + π/2)
// So we use k+1 for octant selection instead of k.
// For cosine, the quadrant mapping becomes:
//   - 0: cos(r)
//   - 1: -sin(r)
//   - 2: -cos(r)
//   - 3: sin(r)
func BaseCosPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	twoOverPi := hwy.Set[T](T(trig2OverPi_f32))
	piOver2Hi := hwy.Set[T](T(trigPiOver2Hi_f32))
	piOver2Lo := hwy.Set[T](T(trigPiOver2Lo_f32))
	one := hwy.Set[T](T(trigOne_f32))
	s1 := hwy.Set[T](T(trigS1_f32))
	s2 := hwy.Set[T](T(trigS2_f32))
	s3 := hwy.Set[T](T(trigS3_f32))
	s4 := hwy.Set[T](T(trigS4_f32))
	c1 := hwy.Set[T](T(trigC1_f32))
	c2 := hwy.Set[T](T(trigC2_f32))
	c3 := hwy.Set[T](T(trigC3_f32))
	c4 := hwy.Set[T](T(trigC4_f32))

	// Integer constants for octant selection
	intOne := hwy.Set[int32](1)
	intTwo := hwy.Set[int32](2)
	intThree := hwy.Set[int32](3)

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Range reduction: k = round(x * 2/π)
		kFloat := hwy.RoundToEven(hwy.Mul(x, twoOverPi))
		kInt := hwy.ConvertToInt32(kFloat)

		// r = x - k * (π/2) using Cody-Waite high/low for precision
		r := hwy.Sub(x, hwy.Mul(kFloat, piOver2Hi))
		r = hwy.Sub(r, hwy.Mul(kFloat, piOver2Lo))
		r2 := hwy.Mul(r, r)

		// Compute sin(r) polynomial: sin(r) ≈ r * (1 + s1*r² + s2*r⁴ + s3*r⁶ + s4*r⁸)
		sinPoly := hwy.MulAdd(s4, r2, s3)
		sinPoly = hwy.MulAdd(sinPoly, r2, s2)
		sinPoly = hwy.MulAdd(sinPoly, r2, s1)
		sinPoly = hwy.MulAdd(sinPoly, r2, one)
		sinR := hwy.Mul(r, sinPoly)

		// Compute cos(r) polynomial: cos(r) ≈ 1 + c1*r² + c2*r⁴ + c3*r⁶ + c4*r⁸
		cosPoly := hwy.MulAdd(c4, r2, c3)
		cosPoly = hwy.MulAdd(cosPoly, r2, c2)
		cosPoly = hwy.MulAdd(cosPoly, r2, c1)
		cosR := hwy.MulAdd(cosPoly, r2, one)

		// For cos(x), we shift by 1 octant: cos(x) = sin(x + π/2)
		cosOctant := hwy.And(hwy.Add(kInt, intOne), intThree)

		// Determine if we should use cos instead of sin (octant & 1 == 1)
		useCosMask := hwy.Equal(hwy.And(cosOctant, intOne), intOne)
		// Determine if we should negate (octant & 2 == 2)
		negateMask := hwy.Equal(hwy.And(cosOctant, intTwo), intTwo)

		// Select between sin(r) and cos(r) based on octant
		sinRData := sinR.Data()
		cosRData := cosR.Data()
		resultData := make([]T, len(sinRData))
		for i := range sinRData {
			if useCosMask.GetBit(i) {
				resultData[i] = cosRData[i]
			} else {
				resultData[i] = sinRData[i]
			}
		}
		result := hwy.Load(resultData)

		// Negate if needed (octant 2 or 3)
		negResult := hwy.Neg(result)
		negResultData := negResult.Data()
		for i := range resultData {
			if negateMask.GetBit(i) {
				resultData[i] = negResultData[i]
			}
		}
		result = hwy.Load(resultData)

		hwy.Store(result, output[ii:])
	}
}

// BaseAtanPoly computes atan(x) using polynomial approximation.
// Uses range reduction: if |x| > 1, use atan(x) = π/2 - atan(1/x)
func BaseAtanPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	piOver2 := hwy.Set[T](T(atanPiOver2_f32))
	piOver4 := hwy.Set[T](T(atanPiOver4_f32))
	tanPiOver8 := hwy.Set[T](T(atanTanPiOver8_f32))
	one := hwy.Set[T](T(atanOne_f32))
	zero := hwy.Set[T](T(atanZero_f32))
	c1 := hwy.Set[T](T(atanC1_f32))
	c2 := hwy.Set[T](T(atanC2_f32))
	c3 := hwy.Set[T](T(atanC3_f32))
	c4 := hwy.Set[T](T(atanC4_f32))
	c5 := hwy.Set[T](T(atanC5_f32))

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Get absolute value and track sign
		absX := hwy.Abs(x)
		signMask := hwy.Less(x, zero)

		// Range reduction level 1: if |x| > 1, use atan(x) = π/2 - atan(1/x)
		useReciprocalMask := hwy.Greater(absX, one)
		recipAbsX := hwy.Div(one, absX)
		reduced := hwy.Merge(recipAbsX, absX, useReciprocalMask)

		// Range reduction level 2: if reduced > tan(π/8), use identity
		useIdentityMask := hwy.Greater(reduced, tanPiOver8)
		xMinus1 := hwy.Sub(reduced, one)
		xPlus1 := hwy.Add(reduced, one)
		transformed := hwy.Div(xMinus1, xPlus1)
		reduced = hwy.Merge(transformed, reduced, useIdentityMask)

		// Compute polynomial
		z2 := hwy.Mul(reduced, reduced)
		poly := hwy.MulAdd(c5, z2, c4)
		poly = hwy.MulAdd(poly, z2, c3)
		poly = hwy.MulAdd(poly, z2, c2)
		poly = hwy.MulAdd(poly, z2, c1)
		poly = hwy.MulAdd(poly, z2, one)
		atanCore := hwy.Mul(reduced, poly)

		// Adjust for identity transform
		atanWithIdentity := hwy.Add(piOver4, atanCore)
		atanReduced := hwy.Merge(atanWithIdentity, atanCore, useIdentityMask)

		// Adjust for reciprocal
		atanWithReciprocal := hwy.Sub(piOver2, atanReduced)
		resultAbs := hwy.Merge(atanWithReciprocal, atanReduced, useReciprocalMask)

		// Restore sign
		negResult := hwy.Neg(resultAbs)
		result := hwy.Merge(negResult, resultAbs, signMask)

		hwy.Store(result, output[ii:])
	}
}

// BaseAsinPoly computes asin(x) using polynomial approximation.
// For |x| < 0.5: uses Taylor series
// For |x| >= 0.5: uses asin(x) = pi/2 - 2*asin(sqrt((1-x)/2))
func BaseAsinPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	piOver2 := hwy.Set[T](T(asinPiOver2_f32))
	half := hwy.Set[T](T(asinHalf_f32))
	two := hwy.Set[T](T(asinTwo_f32))
	one := hwy.Set[T](T(asinOne_f32))
	zero := hwy.Set[T](T(asinZero_f32))
	p1 := hwy.Set[T](T(asinP1_f32))
	p2 := hwy.Set[T](T(asinP2_f32))
	p3 := hwy.Set[T](T(asinP3_f32))
	p4 := hwy.Set[T](T(asinP4_f32))
	p5 := hwy.Set[T](T(asinP5_f32))
	p6 := hwy.Set[T](T(asinP6_f32))

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Get absolute value
		absX := hwy.Abs(x)
		signMask := hwy.Less(x, zero)

		// Check for large argument
		largeMask := hwy.Greater(absX, half)

		// Small argument path: direct polynomial
		x2Small := hwy.Mul(x, x)
		poly := hwy.MulAdd(p6, x2Small, p5)
		poly = hwy.MulAdd(poly, x2Small, p4)
		poly = hwy.MulAdd(poly, x2Small, p3)
		poly = hwy.MulAdd(poly, x2Small, p2)
		poly = hwy.MulAdd(poly, x2Small, p1)
		smallResult := hwy.Add(x, hwy.Mul(hwy.Mul(x, x2Small), poly))

		// Large argument path: asin(x) = pi/2 - 2*asin(sqrt((1-|x|)/2))
		oneMinusAbsX := hwy.Sub(one, absX)
		halfOneMinusAbsX := hwy.Mul(oneMinusAbsX, half)
		sqrtArg := hwy.Sqrt(halfOneMinusAbsX)

		sqrtArg2 := hwy.Mul(sqrtArg, sqrtArg)
		polyLarge := hwy.MulAdd(p6, sqrtArg2, p5)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p4)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p3)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p2)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p1)
		asinSqrtArg := hwy.Add(sqrtArg, hwy.Mul(hwy.Mul(sqrtArg, sqrtArg2), polyLarge))

		largeResultPos := hwy.Sub(piOver2, hwy.Mul(two, asinSqrtArg))

		// Restore sign for large result
		largeResultNeg := hwy.Neg(largeResultPos)
		largeResult := hwy.Merge(largeResultNeg, largeResultPos, signMask)

		// Select between small and large paths
		result := hwy.Merge(largeResult, smallResult, largeMask)

		hwy.Store(result, output[ii:])
	}
}

// BaseAcosPoly computes acos(x) = pi/2 - asin(x)
func BaseAcosPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	piOver2 := hwy.Set[T](T(asinPiOver2_f32))
	half := hwy.Set[T](T(asinHalf_f32))
	two := hwy.Set[T](T(asinTwo_f32))
	one := hwy.Set[T](T(asinOne_f32))
	zero := hwy.Set[T](T(asinZero_f32))
	p1 := hwy.Set[T](T(asinP1_f32))
	p2 := hwy.Set[T](T(asinP2_f32))
	p3 := hwy.Set[T](T(asinP3_f32))
	p4 := hwy.Set[T](T(asinP4_f32))
	p5 := hwy.Set[T](T(asinP5_f32))
	p6 := hwy.Set[T](T(asinP6_f32))

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Compute asin first
		absX := hwy.Abs(x)
		signMask := hwy.Less(x, zero)
		largeMask := hwy.Greater(absX, half)

		// Small path
		x2Small := hwy.Mul(x, x)
		poly := hwy.MulAdd(p6, x2Small, p5)
		poly = hwy.MulAdd(poly, x2Small, p4)
		poly = hwy.MulAdd(poly, x2Small, p3)
		poly = hwy.MulAdd(poly, x2Small, p2)
		poly = hwy.MulAdd(poly, x2Small, p1)
		asinSmall := hwy.Add(x, hwy.Mul(hwy.Mul(x, x2Small), poly))

		// Large path
		oneMinusAbsX := hwy.Sub(one, absX)
		halfOneMinusAbsX := hwy.Mul(oneMinusAbsX, half)
		sqrtArg := hwy.Sqrt(halfOneMinusAbsX)
		sqrtArg2 := hwy.Mul(sqrtArg, sqrtArg)
		polyLarge := hwy.MulAdd(p6, sqrtArg2, p5)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p4)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p3)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p2)
		polyLarge = hwy.MulAdd(polyLarge, sqrtArg2, p1)
		asinSqrtArg := hwy.Add(sqrtArg, hwy.Mul(hwy.Mul(sqrtArg, sqrtArg2), polyLarge))
		largeResultPos := hwy.Sub(piOver2, hwy.Mul(two, asinSqrtArg))
		largeResultNeg := hwy.Neg(largeResultPos)
		asinLarge := hwy.Merge(largeResultNeg, largeResultPos, signMask)

		asinX := hwy.Merge(asinLarge, asinSmall, largeMask)

		// acos(x) = pi/2 - asin(x)
		result := hwy.Sub(piOver2, asinX)

		hwy.Store(result, output[ii:])
	}
}

// =============================================================================
// Additional Hyperbolic Functions
// =============================================================================

// Float32 constants for Sinh/Cosh
var (
	sinhOverflow_f32 float32 = 89.0 // sinh saturates
	sinhOne_f32      float32 = 1.0
	sinhHalf_f32     float32 = 0.5
	sinhZero_f32     float32 = 0.0

	// Polynomial coefficients for sinh(x) for small |x|
	sinhC3_f32 float32 = 0.16666666666666666 // 1/6
	sinhC5_f32 float32 = 0.008333333333333333 // 1/120
	sinhC7_f32 float32 = 0.0001984126984126984 // 1/5040
)

// Float64 constants for Sinh/Cosh
var (
	sinhOverflow_f64 float64 = 710.0
	sinhOne_f64      float64 = 1.0
	sinhHalf_f64     float64 = 0.5
	sinhZero_f64     float64 = 0.0

	sinhC3_f64 float64 = 0.16666666666666666
	sinhC5_f64 float64 = 0.008333333333333333
	sinhC7_f64 float64 = 0.0001984126984126984
)

// BaseSinhPoly computes sinh(x) using the exponential formula.
// sinh(x) = (exp(x) - exp(-x)) / 2
func BaseSinhPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	half := hwy.Set[T](T(0.5))
	lanes := half.NumLanes()

	// Temporary storage for exp computations
	expPos := make([]T, lanes)
	expNeg := make([]T, lanes)
	negInput := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// sinh(x) = (exp(x) - exp(-x)) / 2
		hwy.Store(x, expPos)
		BaseExpPoly[T](expPos, expPos)
		expX := hwy.Load(expPos)

		negX := hwy.Neg(x)
		hwy.Store(negX, negInput)
		BaseExpPoly[T](negInput, expNeg)
		expNegX := hwy.Load(expNeg)

		diff := hwy.Sub(expX, expNegX)
		result := hwy.Mul(diff, half)

		hwy.Store(result, output[ii:])
	}
}

// BaseCoshPoly computes cosh(x) using the exponential formula.
// cosh(x) = (exp(x) + exp(-x)) / 2
func BaseCoshPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	half := hwy.Set[T](T(0.5))
	lanes := half.NumLanes()

	// Temporary storage for exp computations
	expPos := make([]T, lanes)
	expNeg := make([]T, lanes)
	negInput := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// cosh(x) = (exp(x) + exp(-x)) / 2
		hwy.Store(x, expPos)
		BaseExpPoly[T](expPos, expPos)
		expX := hwy.Load(expPos)

		negX := hwy.Neg(x)
		hwy.Store(negX, negInput)
		BaseExpPoly[T](negInput, expNeg)
		expNegX := hwy.Load(expNeg)

		sum := hwy.Add(expX, expNegX)
		result := hwy.Mul(sum, half)

		hwy.Store(result, output[ii:])
	}
}

// =============================================================================
// Error Function (Erf)
// =============================================================================

// Float32 constants for Erf
var (
	erfA1_f32 float32 = 0.254829592
	erfA2_f32 float32 = -0.284496736
	erfA3_f32 float32 = 1.421413741
	erfA4_f32 float32 = -1.453152027
	erfA5_f32 float32 = 1.061405429
	erfP_f32  float32 = 0.3275911

	erfOne_f32  float32 = 1.0
	erfZero_f32 float32 = 0.0
)

// Float64 constants for Erf
var (
	erfA1_f64 float64 = 0.254829592
	erfA2_f64 float64 = -0.284496736
	erfA3_f64 float64 = 1.421413741
	erfA4_f64 float64 = -1.453152027
	erfA5_f64 float64 = 1.061405429
	erfP_f64  float64 = 0.3275911

	erfOne_f64  float64 = 1.0
	erfZero_f64 float64 = 0.0
)

// BaseErfPoly computes erf(x) using Horner form of the Abramowitz and Stegun approximation.
// Algorithm: erf(x) ≈ 1 - (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵) * exp(-x²)
// where t = 1 / (1 + 0.3275911*|x|)
// Accuracy: |error| < 1.5e-7
func BaseErfPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	a1 := hwy.Set[T](T(erfA1_f32))
	a2 := hwy.Set[T](T(erfA2_f32))
	a3 := hwy.Set[T](T(erfA3_f32))
	a4 := hwy.Set[T](T(erfA4_f32))
	a5 := hwy.Set[T](T(erfA5_f32))
	p := hwy.Set[T](T(erfP_f32))
	one := hwy.Set[T](T(erfOne_f32))
	zero := hwy.Set[T](T(erfZero_f32))

	lanes := one.NumLanes()

	// Allocate temporary buffer for exp(-x²) computation
	tempIn := make([]T, lanes)
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// erf(-x) = -erf(x), so work with |x|
		absX := hwy.Abs(x)
		signMask := hwy.Less(x, zero)

		// t = 1 / (1 + p * |x|)
		t := hwy.Div(one, hwy.Add(one, hwy.Mul(p, absX)))

		// Polynomial approximation using Horner's method
		// poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
		poly := hwy.MulAdd(a5, t, a4)
		poly = hwy.MulAdd(poly, t, a3)
		poly = hwy.MulAdd(poly, t, a2)
		poly = hwy.MulAdd(poly, t, a1)
		poly = hwy.Mul(poly, t)

		// Compute exp(-x²) using the actual exp function
		x2 := hwy.Mul(absX, absX)
		negX2 := hwy.Neg(x2)
		hwy.Store(negX2, tempIn)
		BaseExpPoly(tempIn, tempOut)
		expNegX2 := hwy.Load(tempOut)

		// erf(|x|) = 1 - poly * exp(-x²)
		erfAbs := hwy.Sub(one, hwy.Mul(poly, expNegX2))

		// Clamp to [0, 1]
		erfAbs = hwy.Max(hwy.Min(erfAbs, one), zero)

		// Apply sign
		negErfAbs := hwy.Neg(erfAbs)
		result := hwy.Merge(negErfAbs, erfAbs, signMask)

		hwy.Store(result, output[ii:])
	}
}

// =============================================================================
// Expm1 and Log1p (for better accuracy near zero)
// =============================================================================

// Float32 constants for Expm1
var (
	expm1Threshold_f32 float32 = 0.5 // Use polynomial for |x| < threshold
	expm1One_f32       float32 = 1.0
	expm1Half_f32      float32 = 0.5

	// Taylor series coefficients: x + x²/2 + x³/6 + x⁴/24 + ...
	expm1C2_f32 float32 = 0.5
	expm1C3_f32 float32 = 0.16666666666666666
	expm1C4_f32 float32 = 0.041666666666666664
	expm1C5_f32 float32 = 0.008333333333333333
)

// Float64 constants for Expm1
var (
	expm1Threshold_f64 float64 = 0.5
	expm1One_f64       float64 = 1.0
	expm1Half_f64      float64 = 0.5

	expm1C2_f64 float64 = 0.5
	expm1C3_f64 float64 = 0.16666666666666666
	expm1C4_f64 float64 = 0.041666666666666664
	expm1C5_f64 float64 = 0.008333333333333333
	expm1C6_f64 float64 = 0.001388888888888889
)

// BaseExpm1Poly computes exp(x) - 1 with proper range switching.
// For small |x| < 0.5: uses Taylor series for better accuracy
// For larger |x|: uses exp(x) - 1 directly
// Handles special cases: expm1(0)=0, expm1(-Inf)=-1, expm1(+Inf)=+Inf, expm1(NaN)=NaN
func BaseExpm1Poly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(expm1One_f32))
	negOne := hwy.Set[T](T(-1.0))
	zero := hwy.Set[T](T(0.0))
	threshold := hwy.Set[T](T(expm1Threshold_f32))
	c2 := hwy.Set[T](T(expm1C2_f32))
	c3 := hwy.Set[T](T(expm1C3_f32))
	c4 := hwy.Set[T](T(expm1C4_f32))
	c5 := hwy.Set[T](T(expm1C5_f32))

	lanes := one.NumLanes()

	// Temp buffers for exp computation
	tempIn := make([]T, lanes)
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Check for small |x| (use Taylor series for better accuracy)
		absX := hwy.Abs(x)
		smallMask := hwy.Less(absX, threshold)

		// Taylor series path: x + x²/2 + x³/6 + x⁴/24 + x⁵/120
		// = x * (1 + x/2 + x²/6 + x³/24 + x⁴/120)
		poly := hwy.MulAdd(c5, x, c4)
		poly = hwy.MulAdd(poly, x, c3)
		poly = hwy.MulAdd(poly, x, c2)
		poly = hwy.MulAdd(poly, x, one)
		taylorResult := hwy.Mul(x, poly)

		// Large |x| path: exp(x) - 1
		hwy.Store(x, tempIn)
		BaseExpPoly(tempIn, tempOut)
		expX := hwy.Load(tempOut)
		expResult := hwy.Sub(expX, one)

		// Select based on range
		result := hwy.Merge(taylorResult, expResult, smallMask)

		// Handle special cases
		// For very large positive x, exp(x)-1 ≈ exp(x) → +Inf (handled by exp)
		// For very negative x, exp(x)-1 → -1
		veryNegMask := hwy.Less(x, hwy.Set[T](T(-20.0)))
		result = hwy.Merge(negOne, result, veryNegMask)

		// expm1(0) = 0 (exact)
		zeroMask := hwy.Equal(x, zero)
		result = hwy.Merge(zero, result, zeroMask)

		hwy.Store(result, output[ii:])
	}
}

// Float32 constants for Log1p
var (
	log1pOne_f32  float32 = 1.0
	log1pHalf_f32 float32 = 0.5

	// Coefficients for log(1+x) ≈ x - x²/2 + x³/3 - x⁴/4 + ...
	log1pC2_f32 float32 = -0.5
	log1pC3_f32 float32 = 0.3333333333333333
	log1pC4_f32 float32 = -0.25
	log1pC5_f32 float32 = 0.2
)

// Float64 constants for Log1p
var (
	log1pOne_f64  float64 = 1.0
	log1pHalf_f64 float64 = 0.5

	log1pC2_f64 float64 = -0.5
	log1pC3_f64 float64 = 0.3333333333333333
	log1pC4_f64 float64 = -0.25
	log1pC5_f64 float64 = 0.2
	log1pC6_f64 float64 = -0.16666666666666666
)

// BaseLog1pPoly computes log(1+x) with proper range switching.
// For small |x| < 0.5: uses Taylor series for better accuracy
// For larger |x|: computes log(1+x) using BaseLogPoly
// Handles special cases: log1p(0)=0, log1p(-1)=-Inf, log1p(x<-1)=NaN, log1p(+Inf)=+Inf
func BaseLog1pPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(log1pOne_f32))
	zero := hwy.Set[T](T(0.0))
	negOne := hwy.Set[T](T(-1.0))
	threshold := hwy.Set[T](T(log1pHalf_f32))
	c2 := hwy.Set[T](T(log1pC2_f32))
	c3 := hwy.Set[T](T(log1pC3_f32))
	c4 := hwy.Set[T](T(log1pC4_f32))
	c5 := hwy.Set[T](T(log1pC5_f32))
	negInf := hwy.Set[T](T(math.Inf(-1)))
	nan := hwy.Set[T](T(math.NaN()))

	lanes := one.NumLanes()

	// Temp buffers for log computation
	tempIn := make([]T, lanes)
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Check for small |x| (use Taylor series for better accuracy)
		absX := hwy.Abs(x)
		smallMask := hwy.Less(absX, threshold)

		// Taylor series path: x - x²/2 + x³/3 - x⁴/4 + x⁵/5
		// = x * (1 - x/2 + x²/3 - x³/4 + x⁴/5)
		poly := hwy.MulAdd(c5, x, c4)
		poly = hwy.MulAdd(poly, x, c3)
		poly = hwy.MulAdd(poly, x, c2)
		poly = hwy.MulAdd(poly, x, one)
		taylorResult := hwy.Mul(x, poly)

		// Large |x| path: log(1+x) using BaseLogPoly
		onePlusX := hwy.Add(one, x)
		hwy.Store(onePlusX, tempIn)
		BaseLogPoly(tempIn, tempOut)
		logResult := hwy.Load(tempOut)

		// Select based on range
		result := hwy.Merge(taylorResult, logResult, smallMask)

		// Handle special cases
		// log1p(0) = 0 (exact)
		zeroMask := hwy.Equal(x, zero)
		result = hwy.Merge(zero, result, zeroMask)

		// log1p(-1) = -Inf
		negOneMask := hwy.Equal(x, negOne)
		result = hwy.Merge(negInf, result, negOneMask)

		// log1p(x < -1) = NaN
		invalidMask := hwy.Less(x, negOne)
		result = hwy.Merge(nan, result, invalidMask)

		hwy.Store(result, output[ii:])
	}
}

// =============================================================================
// Logarithm Functions
// =============================================================================

// BaseLogPoly computes natural logarithm ln(x) using IEEE 754 exponent extraction
// and polynomial approximation.
//
// Algorithm: log(x) = log(2^e * m) = e*ln(2) + log(m), where m ∈ [1, 2)
// For log(m), we use: let y = (m-1)/(m+1), then log(m) = 2*(y + y³/3 + y⁵/5 + ...)
//
// Special cases: log(0)=-Inf, log(neg)=NaN, log(+Inf)=+Inf, log(1)=0
func BaseLogPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(logOne_f32))
	two := hwy.Set[T](T(logTwo_f32))
	zero := hwy.Set[T](T(0.0))
	ln2Hi := hwy.Set[T](T(logLn2Hi_f32))
	ln2Lo := hwy.Set[T](T(logLn2Lo_f32))
	negInf := hwy.Set[T](T(math.Inf(-1)))
	nan := hwy.Set[T](T(math.NaN()))

	// Polynomial coefficients for atanh-based log
	c1 := hwy.Set[T](T(logC1_f32))
	c2 := hwy.Set[T](T(logC2_f32))
	c3 := hwy.Set[T](T(logC3_f32))
	c4 := hwy.Set[T](T(logC4_f32))
	c5 := hwy.Set[T](T(logC5_f32))

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Handle special cases first
		zeroMask := hwy.Equal(x, zero)
		negMask := hwy.Less(x, zero)
		oneMask := hwy.Equal(x, one)

		// Extract exponent and mantissa using IEEE 754 bit manipulation
		// log(x) = log(2^e * m) = e*ln(2) + log(m)
		e := hwy.GetExponent(x)
		m := hwy.GetMantissa(x) // Returns mantissa in [1, 2)

		// For better accuracy near 1, adjust if m > sqrt(2)
		mLarge := hwy.Greater(m, hwy.Set[T](T(1.414)))
		mAdjusted := hwy.Merge(hwy.Mul(m, hwy.Set[T](T(0.5))), m, mLarge)

		// Convert int32 exponent to float T (scalar conversion)
		eData := e.Data()
		eFloatData := make([]T, len(eData))
		for i, v := range eData {
			eFloatData[i] = T(v)
		}
		eFloat := hwy.Load(eFloatData)
		eAdjusted := hwy.Merge(hwy.Add(eFloat, one), eFloat, mLarge)

		// Compute log(m) using y = (m-1)/(m+1), log(m) = 2*(y + y³/3 + y⁵/5 + ...)
		mMinus1 := hwy.Sub(mAdjusted, one)
		mPlus1 := hwy.Add(mAdjusted, one)
		y := hwy.Div(mMinus1, mPlus1)
		y2 := hwy.Mul(y, y)

		// Polynomial: 1 + y²/3 + y⁴/5 + y⁶/7 + ...
		poly := hwy.MulAdd(c5, y2, c4)
		poly = hwy.MulAdd(poly, y2, c3)
		poly = hwy.MulAdd(poly, y2, c2)
		poly = hwy.MulAdd(poly, y2, c1)
		logM := hwy.Mul(hwy.Mul(two, y), poly)

		// log(x) = e*ln(2) + log(m)
		result := hwy.Add(hwy.MulAdd(eAdjusted, ln2Hi, logM), hwy.Mul(eAdjusted, ln2Lo))

		// Handle special cases
		result = hwy.Merge(negInf, result, zeroMask)
		result = hwy.Merge(nan, result, negMask)
		result = hwy.Merge(zero, result, oneMask)

		hwy.Store(result, output[ii:])
	}
}

// Log conversion constants
var (
	log2E_f32  float32 = 1.4426950408889634 // 1/ln(2) = log2(e)
	log10E_f32 float32 = 0.4342944819032518 // 1/ln(10) = log10(e)

	log2E_f64  float64 = 1.4426950408889634
	log10E_f64 float64 = 0.4342944819032518
)

// BaseLog2Poly computes base-2 logarithm log2(x).
// log2(x) = log(x) / ln(2) = log(x) * log2(e)
func BaseLog2Poly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	log2E := hwy.Set[T](T(log2E_f32))
	lanes := log2E.NumLanes()

	// Temp buffers for log computation
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Compute ln(x) first
		hwy.Store(x, output[ii:]) // Use output as temp
		BaseLogPoly(output[ii:ii+lanes], tempOut)
		lnX := hwy.Load(tempOut)

		// log2(x) = ln(x) * log2(e)
		result := hwy.Mul(lnX, log2E)

		hwy.Store(result, output[ii:])
	}
}

// BaseLog10Poly computes base-10 logarithm log10(x).
// log10(x) = log(x) / ln(10) = log(x) * log10(e)
func BaseLog10Poly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	log10E := hwy.Set[T](T(log10E_f32))
	lanes := log10E.NumLanes()

	// Temp buffers for log computation
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Compute ln(x) first
		hwy.Store(x, output[ii:]) // Use output as temp
		BaseLogPoly(output[ii:ii+lanes], tempOut)
		lnX := hwy.Load(tempOut)

		// log10(x) = ln(x) * log10(e)
		result := hwy.Mul(lnX, log10E)

		hwy.Store(result, output[ii:])
	}
}

// =============================================================================
// Additional Trigonometric Functions
// =============================================================================

// BaseTanPoly computes tan(x) = sin(x) / cos(x)
func BaseTanPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	lanes := hwy.Set[T](T(1.0)).NumLanes()

	// Temp buffers for sin/cos computation
	sinOut := make([]T, lanes)
	cosOut := make([]T, lanes)
	tempIn := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Compute sin(x) and cos(x)
		hwy.Store(x, tempIn)
		BaseSinPoly(tempIn, sinOut)
		hwy.Store(x, tempIn) // restore for cos
		BaseCosPoly(tempIn, cosOut)

		sinX := hwy.Load(sinOut)
		cosX := hwy.Load(cosOut)

		// tan(x) = sin(x) / cos(x)
		result := hwy.Div(sinX, cosX)

		hwy.Store(result, output[ii:])
	}
}

// BaseAtan2Poly computes atan2(y, x) - the angle in radians between the positive x-axis
// and the point (x, y), in the range [-π, π].
func BaseAtan2Poly[T hwy.Floats](inputY, inputX, output []T) {
	size := len(inputY)
	if len(inputX) < size {
		size = len(inputX)
	}
	if len(output) < size {
		size = len(output)
	}

	pi := hwy.Set[T](T(atanPi_f32))
	piOver2 := hwy.Set[T](T(atanPiOver2_f32))
	zero := hwy.Set[T](T(0.0))

	lanes := pi.NumLanes()

	// Temp buffers for atan computation
	tempIn := make([]T, lanes)
	tempOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		y := hwy.Load(inputY[ii:])
		x := hwy.Load(inputX[ii:])

		// Compute atan(y/x)
		ratio := hwy.Div(y, x)
		hwy.Store(ratio, tempIn)
		BaseAtanPoly(tempIn, tempOut)
		atanRatio := hwy.Load(tempOut)

		// Adjust based on quadrant
		xNegMask := hwy.Less(x, zero)
		yNonNegMask := hwy.GreaterEqual(y, zero)
		xZeroMask := hwy.Equal(x, zero)
		yPosZeroMask := hwy.Greater(y, zero)
		yNegMask := hwy.Less(y, zero)

		// Start with atan(y/x)
		result := atanRatio

		// If x < 0 and y >= 0: add π
		addPiMask := hwy.MaskAnd(xNegMask, yNonNegMask)
		result = hwy.Merge(hwy.Add(result, pi), result, addPiMask)

		// If x < 0 and y < 0: subtract π
		subPiMask := hwy.MaskAnd(xNegMask, yNegMask)
		result = hwy.Merge(hwy.Sub(result, pi), result, subPiMask)

		// If x = 0 and y > 0: π/2
		result = hwy.Merge(piOver2, result, hwy.MaskAnd(xZeroMask, yPosZeroMask))

		// If x = 0 and y < 0: -π/2
		result = hwy.Merge(hwy.Neg(piOver2), result, hwy.MaskAnd(xZeroMask, yNegMask))

		hwy.Store(result, output[ii:])
	}
}

// BaseSinCosPoly computes both sin(x) and cos(x) simultaneously.
// This is more efficient when both values are needed.
func BaseSinCosPoly[T hwy.Floats](input, sinOutput, cosOutput []T) {
	size := len(input)
	if len(sinOutput) < size {
		size = len(sinOutput)
	}
	if len(cosOutput) < size {
		size = len(cosOutput)
	}

	twoOverPi := hwy.Set[T](T(trig2OverPi_f32))
	piOver2Hi := hwy.Set[T](T(trigPiOver2Hi_f32))
	piOver2Lo := hwy.Set[T](T(trigPiOver2Lo_f32))
	one := hwy.Set[T](T(trigOne_f32))
	s1 := hwy.Set[T](T(trigS1_f32))
	s2 := hwy.Set[T](T(trigS2_f32))
	s3 := hwy.Set[T](T(trigS3_f32))
	s4 := hwy.Set[T](T(trigS4_f32))
	c1 := hwy.Set[T](T(trigC1_f32))
	c2 := hwy.Set[T](T(trigC2_f32))
	c3 := hwy.Set[T](T(trigC3_f32))
	c4 := hwy.Set[T](T(trigC4_f32))

	intOne := hwy.Set[int32](1)
	intTwo := hwy.Set[int32](2)
	intThree := hwy.Set[int32](3)

	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Range reduction
		kFloat := hwy.RoundToEven(hwy.Mul(x, twoOverPi))
		kInt := hwy.ConvertToInt32(kFloat)

		r := hwy.Sub(x, hwy.Mul(kFloat, piOver2Hi))
		r = hwy.Sub(r, hwy.Mul(kFloat, piOver2Lo))
		r2 := hwy.Mul(r, r)

		// Compute sin(r) and cos(r) polynomials
		sinPoly := hwy.MulAdd(s4, r2, s3)
		sinPoly = hwy.MulAdd(sinPoly, r2, s2)
		sinPoly = hwy.MulAdd(sinPoly, r2, s1)
		sinPoly = hwy.MulAdd(sinPoly, r2, one)
		sinR := hwy.Mul(r, sinPoly)

		cosPoly := hwy.MulAdd(c4, r2, c3)
		cosPoly = hwy.MulAdd(cosPoly, r2, c2)
		cosPoly = hwy.MulAdd(cosPoly, r2, c1)
		cosR := hwy.MulAdd(cosPoly, r2, one)

		// Octant selection for sin
		sinOctant := hwy.And(kInt, intThree)
		sinUseCosMask := hwy.Equal(hwy.And(sinOctant, intOne), intOne)
		sinNegateMask := hwy.Equal(hwy.And(sinOctant, intTwo), intTwo)

		// Octant selection for cos (shifted by 1)
		cosOctant := hwy.And(hwy.Add(kInt, intOne), intThree)
		cosUseCosMask := hwy.Equal(hwy.And(cosOctant, intOne), intOne)
		cosNegateMask := hwy.Equal(hwy.And(cosOctant, intTwo), intTwo)

		// Select and apply negation for sin
		sinRData := sinR.Data()
		cosRData := cosR.Data()
		sinResultData := make([]T, len(sinRData))
		cosResultData := make([]T, len(sinRData))

		for i := range sinRData {
			if sinUseCosMask.GetBit(i) {
				sinResultData[i] = cosRData[i]
			} else {
				sinResultData[i] = sinRData[i]
			}
			if cosUseCosMask.GetBit(i) {
				cosResultData[i] = cosRData[i]
			} else {
				cosResultData[i] = sinRData[i]
			}
		}

		// Apply negation
		for i := range sinResultData {
			if sinNegateMask.GetBit(i) {
				sinResultData[i] = -sinResultData[i]
			}
			if cosNegateMask.GetBit(i) {
				cosResultData[i] = -cosResultData[i]
			}
		}

		hwy.Store(hwy.Load(sinResultData), sinOutput[ii:])
		hwy.Store(hwy.Load(cosResultData), cosOutput[ii:])
	}
}

// =============================================================================
// Power and Exponential Functions
// =============================================================================

// Pow constants
var (
	powOne_f32  float32 = 1.0
	powZero_f32 float32 = 0.0

	powOne_f64  float64 = 1.0
	powZero_f64 float64 = 0.0
)

// BasePowPoly computes x^y = exp(y * log(x))
func BasePowPoly[T hwy.Floats](inputX, inputY, output []T) {
	size := len(inputX)
	if len(inputY) < size {
		size = len(inputY)
	}
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(powOne_f32))
	zero := hwy.Set[T](T(powZero_f32))
	posInf := hwy.Set[T](T(math.Inf(1)))
	lanes := one.NumLanes()

	// Temp buffers
	logIn := make([]T, lanes)
	logOut := make([]T, lanes)
	expIn := make([]T, lanes)
	expOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(inputX[ii:])
		y := hwy.Load(inputY[ii:])

		// x^y = exp(y * log(x))
		hwy.Store(x, logIn)
		BaseLogPoly(logIn, logOut)
		logX := hwy.Load(logOut)

		yLogX := hwy.Mul(y, logX)
		hwy.Store(yLogX, expIn)
		BaseExpPoly(expIn, expOut)
		result := hwy.Load(expOut)

		// Handle special cases
		xZeroMask := hwy.Equal(x, zero)
		xOneMask := hwy.Equal(x, one)
		xInfMask := hwy.IsInf(x, 0)
		yZeroMask := hwy.Equal(y, zero)
		yPosMask := hwy.Greater(y, zero)
		yNegMask := hwy.Less(y, zero)

		// x^0 = 1 for any x (including NaN)
		result = hwy.Merge(one, result, yZeroMask)

		// 0^y = 0 for y > 0
		result = hwy.Merge(zero, result, hwy.MaskAnd(xZeroMask, yPosMask))

		// 0^y = +Inf for y < 0
		result = hwy.Merge(posInf, result, hwy.MaskAnd(xZeroMask, yNegMask))

		// 1^y = 1 for any y
		result = hwy.Merge(one, result, xOneMask)

		// Inf^y = +Inf for y > 0
		result = hwy.Merge(posInf, result, hwy.MaskAnd(xInfMask, yPosMask))

		// Inf^y = 0 for y < 0
		result = hwy.Merge(zero, result, hwy.MaskAnd(xInfMask, yNegMask))

		// Inf^0 = 1 (already covered by x^0 = 1)

		hwy.Store(result, output[ii:])
	}
}

// Exp2/Exp10 constants
var (
	ln2_f32  float32 = 0.6931471805599453
	ln10_f32 float32 = 2.302585092994046

	ln2_f64  float64 = 0.6931471805599453
	ln10_f64 float64 = 2.302585092994046
)

// BaseExp2Poly computes 2^x = exp(x * ln(2))
func BaseExp2Poly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	ln2 := hwy.Set[T](T(ln2_f32))
	lanes := ln2.NumLanes()

	// Temp buffer
	expIn := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// 2^x = exp(x * ln(2))
		xLn2 := hwy.Mul(x, ln2)
		hwy.Store(xLn2, expIn)
		BaseExpPoly(expIn, output[ii:ii+lanes])
	}
}

// BaseExp10Poly computes 10^x = exp(x * ln(10))
func BaseExp10Poly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	ln10 := hwy.Set[T](T(ln10_f32))
	lanes := ln10.NumLanes()

	// Temp buffer
	expIn := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// 10^x = exp(x * ln(10))
		xLn10 := hwy.Mul(x, ln10)
		hwy.Store(xLn10, expIn)
		BaseExpPoly(expIn, output[ii:ii+lanes])
	}
}

// =============================================================================
// Inverse Hyperbolic Functions
// =============================================================================

// BaseAsinhPoly computes asinh(x) = ln(x + sqrt(x² + 1))
func BaseAsinhPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(1.0))
	lanes := one.NumLanes()

	// Temp buffers
	logIn := make([]T, lanes)
	logOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// asinh(x) = ln(x + sqrt(x² + 1))
		x2 := hwy.Mul(x, x)
		x2Plus1 := hwy.Add(x2, one)
		sqrtPart := hwy.Sqrt(x2Plus1)
		arg := hwy.Add(x, sqrtPart)

		hwy.Store(arg, logIn)
		BaseLogPoly(logIn, logOut)
		result := hwy.Load(logOut)

		hwy.Store(result, output[ii:])
	}
}

// BaseAcoshPoly computes acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
func BaseAcoshPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(1.0))
	zero := hwy.Set[T](T(0.0))
	lanes := one.NumLanes()

	// Temp buffers
	logIn := make([]T, lanes)
	logOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// acosh(x) = ln(x + sqrt(x² - 1))
		x2 := hwy.Mul(x, x)
		x2Minus1 := hwy.Sub(x2, one)
		sqrtPart := hwy.Sqrt(x2Minus1)
		arg := hwy.Add(x, sqrtPart)

		hwy.Store(arg, logIn)
		BaseLogPoly(logIn, logOut)
		result := hwy.Load(logOut)

		// Handle x = 1 case: acosh(1) = 0
		oneMask := hwy.Equal(x, one)
		result = hwy.Merge(zero, result, oneMask)

		// Handle x < 1: result is NaN (domain error)
		invalidMask := hwy.Less(x, one)
		nan := hwy.Div(zero, zero)
		result = hwy.Merge(nan, result, invalidMask)

		hwy.Store(result, output[ii:])
	}
}

// BaseAtanhPoly computes atanh(x) = 0.5 * ln((1+x)/(1-x)) for |x| < 1
func BaseAtanhPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(1.0))
	half := hwy.Set[T](T(0.5))
	zero := hwy.Set[T](T(0.0))
	lanes := one.NumLanes()

	// Temp buffers
	logIn := make([]T, lanes)
	logOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// atanh(x) = 0.5 * ln((1+x)/(1-x))
		onePlusX := hwy.Add(one, x)
		oneMinusX := hwy.Sub(one, x)
		ratio := hwy.Div(onePlusX, oneMinusX)

		hwy.Store(ratio, logIn)
		BaseLogPoly(logIn, logOut)
		logRatio := hwy.Load(logOut)
		result := hwy.Mul(half, logRatio)

		// Handle x = 0: atanh(0) = 0
		zeroMask := hwy.Equal(x, zero)
		result = hwy.Merge(zero, result, zeroMask)

		// Handle |x| >= 1: result is ±Inf or NaN
		inf := hwy.Div(one, zero)  // +Inf
		negInf := hwy.Neg(inf)
		oneMask := hwy.Equal(x, one)
		negOneMask := hwy.Equal(x, hwy.Neg(one))
		result = hwy.Merge(inf, result, oneMask)
		result = hwy.Merge(negInf, result, negOneMask)

		hwy.Store(result, output[ii:])
	}
}

// =============================================================================
// Additional Math Functions
// =============================================================================

// BaseCbrtPoly computes the cube root of x.
// cbrt(x) = sign(x) * |x|^(1/3) = sign(x) * exp(ln(|x|)/3)
func BaseCbrtPoly[T hwy.Floats](input, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	zero := hwy.Set[T](T(0.0))
	third := hwy.Set[T](T(1.0 / 3.0))
	posInf := hwy.Set[T](T(math.Inf(1)))
	negInf := hwy.Set[T](T(math.Inf(-1)))
	nan := hwy.Set[T](T(math.NaN()))
	lanes := zero.NumLanes()

	// Temp buffers
	logIn := make([]T, lanes)
	logOut := make([]T, lanes)
	expIn := make([]T, lanes)
	expOut := make([]T, lanes)

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Detect special cases first
		posInfMask := hwy.Equal(x, posInf)
		negInfMask := hwy.Equal(x, negInf)
		nanMask := hwy.IsNaN(x)

		// Get sign and work with absolute value
		signMask := hwy.Less(x, zero)
		absX := hwy.Abs(x)

		// cbrt(|x|) = exp(ln(|x|)/3)
		hwy.Store(absX, logIn)
		BaseLogPoly(logIn, logOut)
		logAbsX := hwy.Load(logOut)

		logAbsXOver3 := hwy.Mul(logAbsX, third)
		hwy.Store(logAbsXOver3, expIn)
		BaseExpPoly(expIn, expOut)
		cbrtAbsX := hwy.Load(expOut)

		// Restore sign
		negCbrt := hwy.Neg(cbrtAbsX)
		result := hwy.Merge(negCbrt, cbrtAbsX, signMask)

		// Handle x = 0
		zeroMask := hwy.Equal(x, zero)
		result = hwy.Merge(zero, result, zeroMask)

		// Handle special cases: Inf and NaN
		result = hwy.Merge(posInf, result, posInfMask)
		result = hwy.Merge(negInf, result, negInfMask)
		result = hwy.Merge(nan, result, nanMask)

		hwy.Store(result, output[ii:])
	}
}

// BaseHypotPoly computes sqrt(x² + y²) in a numerically stable way.
func BaseHypotPoly[T hwy.Floats](inputX, inputY, output []T) {
	size := len(inputX)
	if len(inputY) < size {
		size = len(inputY)
	}
	if len(output) < size {
		size = len(output)
	}

	one := hwy.Set[T](T(1.0))
	zero := hwy.Set[T](T(0.0))
	posInf := hwy.Set[T](T(math.Inf(1)))
	nan := hwy.Set[T](T(math.NaN()))
	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(inputX[ii:])
		y := hwy.Load(inputY[ii:])

		// Detect Inf and NaN - Inf takes precedence over NaN
		xIsInf := hwy.IsInf(x, 0)
		yIsInf := hwy.IsInf(y, 0)
		xIsNaN := hwy.IsNaN(x)
		yIsNaN := hwy.IsNaN(y)
		eitherInf := hwy.MaskOr(xIsInf, yIsInf)
		eitherNaN := hwy.MaskOr(xIsNaN, yIsNaN)

		// Use |x| and |y| for stability
		absX := hwy.Abs(x)
		absY := hwy.Abs(y)

		// Find max and min to avoid overflow
		maxXY := hwy.Max(absX, absY)
		minXY := hwy.Min(absX, absY)

		// hypot = max * sqrt(1 + (min/max)²)
		// This avoids overflow when x or y is large
		ratio := hwy.Div(minXY, maxXY)
		ratio2 := hwy.Mul(ratio, ratio)
		sqrtArg := hwy.Add(one, ratio2)
		sqrtPart := hwy.Sqrt(sqrtArg)
		result := hwy.Mul(maxXY, sqrtPart)

		// Handle case where max = 0 (both x and y are 0)
		zeroMask := hwy.Equal(maxXY, zero)
		result = hwy.Merge(zero, result, zeroMask)

		// Handle special cases: Inf takes precedence over NaN
		result = hwy.Merge(nan, result, eitherNaN)
		result = hwy.Merge(posInf, result, eitherInf)

		hwy.Store(result, output[ii:])
	}
}
