package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Hypot computes sqrt(x^2 + y^2) for each pair of elements.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Hypot to each pair of lanes independently.
// stdmath.Hypot is numerically stable and avoids overflow/underflow.
func Hypot[T hwy.Floats](x, y hwy.Vec[T]) hwy.Vec[T] {
	xData := x.Data()
	yData := y.Data()
	n := len(xData)
	if len(yData) < n {
		n = len(yData)
	}
	result := make([]T, n)
	for i := 0; i < n; i++ {
		result[i] = T(stdmath.Hypot(float64(xData[i]), float64(yData[i])))
	}
	return hwy.Load(result)
}
