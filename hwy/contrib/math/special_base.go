//go:build !amd64 || !goexperiment.simd

package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Sigmoid computes sigmoid(x) = 1 / (1 + e^(-x)) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Sigmoid_AVX2_F32x8) will be used instead via runtime dispatch.
//
// The sigmoid function is commonly used in neural networks and logistic regression.
// It maps any real value to the range (0, 1), with sigmoid(0) = 0.5.
//
// Algorithm: Applies the sigmoid formula to each lane independently.
//
// Special cases:
//   - Sigmoid(+Inf) = 1
//   - Sigmoid(-Inf) = 0
//   - Sigmoid(NaN) = NaN
func Sigmoid[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		// sigmoid(x) = 1 / (1 + exp(-x))
		expNegX := stdmath.Exp(-float64(x))
		result[i] = T(1.0 / (1.0 + expNegX))
	}
	return hwy.Load(result)
}

// Erf computes the error function erf(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Erf_AVX2_F32x8) will be used instead via runtime dispatch.
//
// The error function is defined as:
//
//	erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
//
// It is commonly used in probability, statistics, and partial differential equations.
//
// Algorithm: Applies stdmath.Erf to each lane independently.
//
// Properties:
//   - erf(-x) = -erf(x) (odd function)
//   - erf(0) = 0
//   - erf(+Inf) = 1
//   - erf(-Inf) = -1
//   - erf(NaN) = NaN
func Erf[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Erf(float64(x)))
	}
	return hwy.Load(result)
}
