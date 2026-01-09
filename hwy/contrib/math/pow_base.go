package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Pow computes x^y for each corresponding pair of elements in the vectors.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Pow_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Pow to each lane independently.
//
// Special cases (following Go's math.Pow behavior):
//
//	Pow(x, ±0) = 1 for any x
//	Pow(1, y) = 1 for any y
//	Pow(x, 1) = x for any x
//	Pow(NaN, y) = NaN
//	Pow(x, NaN) = NaN
//	Pow(±0, y) = ±Inf for y an odd integer < 0
//	Pow(±0, -Inf) = +Inf
//	Pow(±0, +Inf) = +0
//	Pow(±0, y) = +Inf for finite y < 0 and not an odd integer
//	Pow(±0, y) = ±0 for y an odd integer > 0
//	Pow(±0, y) = +0 for finite y > 0 and not an odd integer
//	Pow(-1, ±Inf) = 1
//	Pow(x, +Inf) = +Inf for |x| > 1
//	Pow(x, -Inf) = +0 for |x| > 1
//	Pow(x, +Inf) = +0 for |x| < 1
//	Pow(x, -Inf) = +Inf for |x| < 1
//	Pow(+Inf, y) = +Inf for y > 0
//	Pow(+Inf, y) = +0 for y < 0
//	Pow(-Inf, y) = Pow(-0, -y)
//	Pow(x, y) = NaN for finite x < 0 and finite non-integer y
func Pow[T hwy.Floats](x, y hwy.Vec[T]) hwy.Vec[T] {
	xData := x.Data()
	yData := y.Data()
	result := make([]T, len(xData))
	for i := range xData {
		result[i] = T(stdmath.Pow(float64(xData[i]), float64(yData[i])))
	}
	return hwy.Load(result)
}
