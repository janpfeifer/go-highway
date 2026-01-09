//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
)

// AVX2 vectorized constants for pow32
var (
	pow32_zero   = archsimd.BroadcastFloat32x8(0.0)
	pow32_one    = archsimd.BroadcastFloat32x8(1.0)
	pow32_negOne = archsimd.BroadcastFloat32x8(-1.0)
	pow32_inf    = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(1)))
	pow32_negInf = archsimd.BroadcastFloat32x8(float32(stdmath.Inf(-1)))
	pow32_nan    = archsimd.BroadcastFloat32x8(float32(stdmath.NaN()))
)

// Pow_AVX2_F32x8 computes x^y for Float32x8 vectors.
//
// Algorithm: x^y = exp(y * log(x))
//
// Special cases handled:
//   - x = 0, y > 0: 0
//   - x = 0, y < 0: +Inf
//   - x = 0, y = 0: 1
//   - x = 1, any y: 1
//   - any x, y = 0: 1
//   - x < 0: NaN (negative base with non-integer exponent)
//   - x = +Inf, y > 0: +Inf
//   - x = +Inf, y < 0: 0
//   - x = +Inf, y = 0: 1
func Pow_AVX2_F32x8(x, y archsimd.Float32x8) archsimd.Float32x8 {
	// Save inputs for special case handling
	origX := x
	origY := y

	// Compute log(x) - will return NaN for x < 0, -Inf for x = 0
	logX := Log_AVX2_F32x8(x)

	// Compute y * log(x)
	yLogX := y.Mul(logX)

	// Compute exp(y * log(x))
	result := Exp_AVX2_F32x8(yLogX)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)

	// Case: y = 0 -> result = 1 (for any x, including NaN)
	yIsZero := origY.Equal(pow32_zero)
	result = pow32_one.Merge(result, yIsZero)

	// Case: x = 1 -> result = 1 (for any y, including NaN)
	xIsOne := origX.Equal(pow32_one)
	result = pow32_one.Merge(result, xIsOne)

	// Case: x = 0 and y > 0 -> result = 0
	xIsZero := origX.Equal(pow32_zero)
	yIsPos := origY.Greater(pow32_zero)
	xZeroYPos := xIsZero.And(yIsPos)
	result = pow32_zero.Merge(result, xZeroYPos)

	// Case: x = 0 and y < 0 -> result = +Inf
	yIsNeg := origY.Less(pow32_zero)
	xZeroYNeg := xIsZero.And(yIsNeg)
	result = pow32_inf.Merge(result, xZeroYNeg)

	// Case: x = +Inf and y > 0 -> result = +Inf
	xIsInf := origX.Equal(pow32_inf)
	xInfYPos := xIsInf.And(yIsPos)
	result = pow32_inf.Merge(result, xInfYPos)

	// Case: x = +Inf and y < 0 -> result = 0
	xInfYNeg := xIsInf.And(yIsNeg)
	result = pow32_zero.Merge(result, xInfYNeg)

	// Case: x = +Inf and y = 0 -> result = 1
	xInfYZero := xIsInf.And(yIsZero)
	result = pow32_one.Merge(result, xInfYZero)

	// Case: x < 0 -> result = NaN (we don't handle integer exponents for simplicity)
	// Note: Log already returns NaN for negative x, but we explicitly handle this
	// to ensure correct behavior when y = 0 (already handled above with priority)
	xIsNeg := origX.Less(pow32_zero)
	// Don't apply NaN if y = 0 (already handled) - y = 0 case has higher priority
	xNegYNotZero := xIsNeg.AndNot(yIsZero)
	result = pow32_nan.Merge(result, xNegYNotZero)

	return result
}

// Pow_AVX2_F64x4 computes x^y for Float64x4 vectors.
//
// Note: Uses scalar fallback because AVX2 lacks proper 64-bit integer shift
// support required for efficient Log/Exp implementations.
func Pow_AVX2_F64x4(x, y archsimd.Float64x4) archsimd.Float64x4 {
	var xIn, yIn, out [4]float64
	x.StoreSlice(xIn[:])
	y.StoreSlice(yIn[:])
	for i := range xIn {
		out[i] = stdmath.Pow(xIn[i], yIn[i])
	}
	return archsimd.LoadFloat64x4Slice(out[:])
}
