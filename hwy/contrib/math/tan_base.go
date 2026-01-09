package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Tan computes tan(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Tan_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Tan to each lane independently.
//
// Special cases:
//   - Tan(±0) = ±0
//   - Tan(±Inf) = NaN
//   - Tan(NaN) = NaN
//   - Tan(x) approaches ±Inf as x approaches odd multiples of π/2
func Tan[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Tan(float64(x)))
	}
	return hwy.Load(result)
}
