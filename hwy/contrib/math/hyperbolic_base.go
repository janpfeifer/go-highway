package math

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// Sinh computes sinh(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Sinh_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Sinh to each lane independently.
//
// Formula: sinh(x) = (e^x - e^(-x)) / 2
func Sinh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Sinh(float64(x)))
	}
	return hwy.Load(result)
}

// Cosh computes cosh(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Cosh_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Cosh to each lane independently.
//
// Formula: cosh(x) = (e^x + e^(-x)) / 2
func Cosh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Cosh(float64(x)))
	}
	return hwy.Load(result)
}

// Tanh computes tanh(x) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Tanh to each lane independently.
//
// Formula: tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
func Tanh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Tanh(float64(x)))
	}
	return hwy.Load(result)
}

// Asinh computes asinh(x) (inverse hyperbolic sine) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Asinh_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Asinh to each lane independently.
//
// Formula: asinh(x) = ln(x + sqrt(x² + 1))
func Asinh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Asinh(float64(x)))
	}
	return hwy.Load(result)
}

// Acosh computes acosh(x) (inverse hyperbolic cosine) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Acosh_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Acosh to each lane independently.
//
// Formula: acosh(x) = ln(x + sqrt(x² - 1))
//
// Special cases:
//   - Acosh(x) = NaN if x < 1
//   - Acosh(1) = 0
//   - Acosh(+Inf) = +Inf
func Acosh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Acosh(float64(x)))
	}
	return hwy.Load(result)
}

// Atanh computes atanh(x) (inverse hyperbolic tangent) for each element in the vector.
//
// This is the portable fallback implementation that uses scalar math
// operations. When SIMD is available, architecture-specific implementations
// (e.g., Atanh_AVX2_F32x8) will be used instead via runtime dispatch.
//
// Algorithm: Applies stdmath.Atanh to each lane independently.
//
// Formula: atanh(x) = 0.5 * ln((1 + x) / (1 - x))
//
// Special cases:
//   - Atanh(x) = NaN if x < -1 or x > 1
//   - Atanh(±1) = ±Inf
func Atanh[T hwy.Floats](v hwy.Vec[T]) hwy.Vec[T] {
	data := v.Data()
	result := make([]T, len(data))
	for i, x := range data {
		result[i] = T(stdmath.Atanh(float64(x)))
	}
	return hwy.Load(result)
}
