package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Asin computes asin(x) (arc sine) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Asin_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Asin to each lane independently.
//
// Special cases:
//   - Asin(0) = 0
//   - Asin(1) = +pi/2
//   - Asin(-1) = -pi/2
//   - Asin(x) = NaN if |x| > 1
//   - Asin(NaN) = NaN
func Asin[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Asin(float64(x)))
	}
	return hwy.Load(result)
}

// Acos computes acos(x) (arc cosine) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Acos_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Acos to each lane independently.
//
// Special cases:
//   - Acos(1) = 0
//   - Acos(-1) = pi
//   - Acos(0) = +pi/2
//   - Acos(x) = NaN if |x| > 1
//   - Acos(NaN) = NaN
func Acos[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Acos(float64(x)))
	}
	return hwy.Load(result)
}
