package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Cbrt computes the cube root of each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Cbrt to each lane independently.
func Cbrt[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Cbrt(float64(x)))
	}
	return hwy.Load(result)
}
