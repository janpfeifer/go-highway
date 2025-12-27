//go:build !amd64 || !goexperiment.simd

package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Exp computes e^x for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Exp_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Exp to each lane independently.
func Exp[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Exp(float64(x)))
	}
	return hwy.Load(result)
}

// Exp2 computes 2^x for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Exp2 to each lane independently.
func Exp2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Exp2(float64(x)))
	}
	return hwy.Load(result)
}

// Exp10 computes 10^x for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Pow(10, x) to each lane independently.
func Exp10[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Pow(10, float64(x)))
	}
	return hwy.Load(result)
}
