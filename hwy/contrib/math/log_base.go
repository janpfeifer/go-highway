//go:build !amd64 || !goexperiment.simd

package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Log computes ln(x) (natural logarithm) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Log_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Log to each lane independently.
//
// Special cases:
//   - Log(x) = NaN if x < 0
//   - Log(0) = -Inf
//   - Log(+Inf) = +Inf
//   - Log(NaN) = NaN
func Log[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Log(float64(x)))
	}
	return hwy.Load(result)
}

// Log2 computes log₂(x) (logarithm base 2) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Log2 to each lane independently.
//
// Special cases: Same as Log.
func Log2[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Log2(float64(x)))
	}
	return hwy.Load(result)
}

// Log10 computes log₁₀(x) (logarithm base 10) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Log10 to each lane independently.
//
// Special cases: Same as Log.
func Log10[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Log10(float64(x)))
	}
	return hwy.Load(result)
}
