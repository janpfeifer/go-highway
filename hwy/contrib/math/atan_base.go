package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Atan computes atan(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Atan_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Atan to each lane independently.
//
// Special cases:
//   - Atan(±0) = ±0
//   - Atan(±Inf) = ±π/2
//   - Atan(NaN) = NaN
func Atan[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Atan(float64(x)))
	}
	return hwy.Load(result)
}

// Atan2 computes atan2(y, x) for each pair of elements in the vectors.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Atan2_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Atan2 to each pair of lanes independently.
//
// Atan2 returns the arc tangent of y/x, using the signs of both arguments
// to determine the quadrant of the return value.
//
// Special cases:
//   - Atan2(y, NaN) = NaN
//   - Atan2(NaN, x) = NaN
//   - Atan2(+0, x>=0) = +0
//   - Atan2(-0, x>=0) = -0
//   - Atan2(+0, x<0) = +π
//   - Atan2(-0, x<0) = -π
//   - Atan2(y>0, 0) = +π/2
//   - Atan2(y<0, 0) = -π/2
//   - Atan2(+Inf, +Inf) = +π/4
//   - Atan2(-Inf, +Inf) = -π/4
//   - Atan2(+Inf, -Inf) = +3π/4
//   - Atan2(-Inf, -Inf) = -3π/4
//   - Atan2(y, +Inf) = 0
//   - Atan2(y>0, -Inf) = +π
//   - Atan2(y<0, -Inf) = -π
//   - Atan2(+Inf, x) = +π/2
//   - Atan2(-Inf, x) = -π/2
func Atan2[T hwy.Floats](y, x hwy.Vec[T]) hwy.Vec[T] {
	yData := y.Data()
	xData := x.Data()
	result := make([]T, len(yData))
	for i := range yData {
		result[i] = T(stdmath.Atan2(float64(yData[i]), float64(xData[i])))
	}
	return hwy.Load(result)
}
