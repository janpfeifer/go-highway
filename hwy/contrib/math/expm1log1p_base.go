package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Expm1 computes exp(x) - 1 for each element in the vector.
//
// This function is more accurate than computing exp(x) - 1 directly when x
// is close to zero, avoiding catastrophic cancellation that occurs when
// subtracting two nearly equal floating-point numbers.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Expm1_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Expm1 to each lane independently.
//
// Special cases:
//   - Expm1(+Inf) = +Inf
//   - Expm1(-Inf) = -1
//   - Expm1(NaN) = NaN
//   - Expm1(0) = 0
func Expm1[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Expm1(float64(x)))
	}
	return hwy.Load(result)
}

// Log1p computes log(1 + x) for each element in the vector.
//
// This function is more accurate than computing log(1 + x) directly when x
// is close to zero, avoiding catastrophic cancellation that occurs when
// adding 1 to a small number and then taking the logarithm.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Log1p_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Log1p to each lane independently.
//
// Special cases:
//   - Log1p(x) = NaN if x < -1
//   - Log1p(-1) = -Inf
//   - Log1p(+Inf) = +Inf
//   - Log1p(NaN) = NaN
//   - Log1p(0) = 0
func Log1p[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Log1p(float64(x)))
	}
	return hwy.Load(result)
}
