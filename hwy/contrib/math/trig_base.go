//go:build !amd64 || !goexperiment.simd

package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Sin computes sin(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Sin_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Sin to each lane independently.
//
// Special cases:
//   - Sin(±0) = ±0
//   - Sin(±Inf) = NaN
//   - Sin(NaN) = NaN
func Sin[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Sin(float64(x)))
	}
	return hwy.Load(result)
}

// Cos computes cos(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Cos_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Cos to each lane independently.
//
// Special cases:
//   - Cos(±Inf) = NaN
//   - Cos(NaN) = NaN
func Cos[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Cos(float64(x)))
	}
	return hwy.Load(result)
}

// SinCos computes both sin(x) and cos(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., SinCos_AVX2_F32x8) will be used instead via runtime dispatch.
//
// This is more efficient than calling Sin and Cos separately as it shares
// the range reduction computation (in SIMD implementations) and can avoid
// redundant function calls (in scalar implementation).
//
// Algorithm: Applies stdmath.Sincos to each lane independently.
//
// Returns: (sin, cos) where each is a Vec[T] with the computed values.
func SinCos[T hwy.Floats](v hwy.Vec[T]) (sin, cos hwy.Vec[T]) {
	data := v.Data()
	sinResult := make([]T, len(data))
	cosResult := make([]T, len(data))
	for i, x := range data {
		s, c := stdmath.Sincos(float64(x))
		sinResult[i] = T(s)
		cosResult[i] = T(c)
	}
	return hwy.Load(sinResult), hwy.Load(cosResult)
}
