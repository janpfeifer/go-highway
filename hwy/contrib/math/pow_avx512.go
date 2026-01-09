//go:build amd64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"sync"
)

// Lazy initialization for AVX-512 pow constants to avoid executing AVX-512
// instructions at package load time on machines without AVX-512 support.

var pow512Init sync.Once

// AVX-512 vectorized constants for pow32
var (
	pow512_32_zero   archsimd.Float32x16
	pow512_32_one    archsimd.Float32x16
	pow512_32_inf    archsimd.Float32x16
	pow512_32_nan    archsimd.Float32x16
)

// AVX-512 vectorized constants for pow64
var (
	pow512_64_zero   archsimd.Float64x8
	pow512_64_one    archsimd.Float64x8
	pow512_64_inf    archsimd.Float64x8
	pow512_64_nan    archsimd.Float64x8
)

func initPow512Constants() {
	// Float32 constants
	pow512_32_zero = archsimd.BroadcastFloat32x16(0.0)
	pow512_32_one = archsimd.BroadcastFloat32x16(1.0)
	pow512_32_inf = archsimd.BroadcastFloat32x16(float32(stdmath.Inf(1)))
	pow512_32_nan = archsimd.BroadcastFloat32x16(float32(stdmath.NaN()))

	// Float64 constants
	pow512_64_zero = archsimd.BroadcastFloat64x8(0.0)
	pow512_64_one = archsimd.BroadcastFloat64x8(1.0)
	pow512_64_inf = archsimd.BroadcastFloat64x8(stdmath.Inf(1))
	pow512_64_nan = archsimd.BroadcastFloat64x8(stdmath.NaN())
}

// Pow_AVX512_F32x16 computes x^y for Float32x16 vectors.
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
func Pow_AVX512_F32x16(x, y archsimd.Float32x16) archsimd.Float32x16 {
	pow512Init.Do(initPow512Constants)

	// Save inputs for special case handling
	origX := x
	origY := y

	// Compute log(x) - will return NaN for x < 0, -Inf for x = 0
	logX := Log_AVX512_F32x16(x)

	// Compute y * log(x)
	yLogX := y.Mul(logX)

	// Compute exp(y * log(x))
	result := Exp_AVX512_F32x16(yLogX)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)

	// Case: y = 0 -> result = 1 (for any x, including NaN)
	yIsZero := origY.Equal(pow512_32_zero)
	result = pow512_32_one.Merge(result, yIsZero)

	// Case: x = 1 -> result = 1 (for any y, including NaN)
	xIsOne := origX.Equal(pow512_32_one)
	result = pow512_32_one.Merge(result, xIsOne)

	// Case: x = 0 and y > 0 -> result = 0
	xIsZero := origX.Equal(pow512_32_zero)
	yIsPos := origY.Greater(pow512_32_zero)
	xZeroYPos := xIsZero.And(yIsPos)
	result = pow512_32_zero.Merge(result, xZeroYPos)

	// Case: x = 0 and y < 0 -> result = +Inf
	yIsNeg := origY.Less(pow512_32_zero)
	xZeroYNeg := xIsZero.And(yIsNeg)
	result = pow512_32_inf.Merge(result, xZeroYNeg)

	// Case: x = +Inf and y > 0 -> result = +Inf
	xIsInf := origX.Equal(pow512_32_inf)
	xInfYPos := xIsInf.And(yIsPos)
	result = pow512_32_inf.Merge(result, xInfYPos)

	// Case: x = +Inf and y < 0 -> result = 0
	xInfYNeg := xIsInf.And(yIsNeg)
	result = pow512_32_zero.Merge(result, xInfYNeg)

	// Case: x = +Inf and y = 0 -> result = 1
	xInfYZero := xIsInf.And(yIsZero)
	result = pow512_32_one.Merge(result, xInfYZero)

	// Case: x < 0 -> result = NaN (we don't handle integer exponents for simplicity)
	// Note: Log already returns NaN for negative x, but we explicitly handle this
	// to ensure correct behavior when y = 0 (already handled above with priority)
	xIsNeg := origX.Less(pow512_32_zero)
	// Don't apply NaN if y = 0 (already handled) - y = 0 case has higher priority
	xNegYNotZero := xIsNeg.AndNot(yIsZero)
	result = pow512_32_nan.Merge(result, xNegYNotZero)

	return result
}

// Pow_AVX512_F64x8 computes x^y for Float64x8 vectors.
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
func Pow_AVX512_F64x8(x, y archsimd.Float64x8) archsimd.Float64x8 {
	pow512Init.Do(initPow512Constants)

	// Save inputs for special case handling
	origX := x
	origY := y

	// Compute log(x) - will return NaN for x < 0, -Inf for x = 0
	logX := Log_AVX512_F64x8(x)

	// Compute y * log(x)
	yLogX := y.Mul(logX)

	// Compute exp(y * log(x))
	result := Exp_AVX512_F64x8(yLogX)

	// Handle special cases (Merge semantics: a.Merge(b, mask) returns a when TRUE, b when FALSE)

	// Case: y = 0 -> result = 1 (for any x, including NaN)
	yIsZero := origY.Equal(pow512_64_zero)
	result = pow512_64_one.Merge(result, yIsZero)

	// Case: x = 1 -> result = 1 (for any y, including NaN)
	xIsOne := origX.Equal(pow512_64_one)
	result = pow512_64_one.Merge(result, xIsOne)

	// Case: x = 0 and y > 0 -> result = 0
	xIsZero := origX.Equal(pow512_64_zero)
	yIsPos := origY.Greater(pow512_64_zero)
	xZeroYPos := xIsZero.And(yIsPos)
	result = pow512_64_zero.Merge(result, xZeroYPos)

	// Case: x = 0 and y < 0 -> result = +Inf
	yIsNeg := origY.Less(pow512_64_zero)
	xZeroYNeg := xIsZero.And(yIsNeg)
	result = pow512_64_inf.Merge(result, xZeroYNeg)

	// Case: x = +Inf and y > 0 -> result = +Inf
	xIsInf := origX.Equal(pow512_64_inf)
	xInfYPos := xIsInf.And(yIsPos)
	result = pow512_64_inf.Merge(result, xInfYPos)

	// Case: x = +Inf and y < 0 -> result = 0
	xInfYNeg := xIsInf.And(yIsNeg)
	result = pow512_64_zero.Merge(result, xInfYNeg)

	// Case: x = +Inf and y = 0 -> result = 1
	xInfYZero := xIsInf.And(yIsZero)
	result = pow512_64_one.Merge(result, xInfYZero)

	// Case: x < 0 -> result = NaN (we don't handle integer exponents for simplicity)
	// Note: Log already returns NaN for negative x, but we explicitly handle this
	// to ensure correct behavior when y = 0 (already handled above with priority)
	xIsNeg := origX.Less(pow512_64_zero)
	// Don't apply NaN if y = 0 (already handled) - y = 0 case has higher priority
	xNegYNotZero := xIsNeg.AndNot(yIsZero)
	result = pow512_64_nan.Merge(result, xNegYNotZero)

	return result
}
