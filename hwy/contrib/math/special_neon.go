//go:build arm64

package math

import (
	"github.com/ajroetker/go-highway/hwy/asm"
	stdmath "math"
)

// Sigmoid_NEON_F32x4 computes sigmoid(x) = 1/(1+exp(-x)) for a Float32x4 vector.
func Sigmoid_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	var result asm.Float32x4
	asm.SigmoidF32(x[:], result[:])
	return result
}

// Sigmoid_NEON_F64x2 computes sigmoid(x) for a Float64x2 vector.
// Uses scalar fallback since we don't have F64 sigmoid in asm yet.
func Sigmoid_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		1.0 / (1.0 + stdmath.Exp(-x[0])),
		1.0 / (1.0 + stdmath.Exp(-x[1])),
	}
}

// Tanh_NEON_F32x4 computes tanh(x) for a Float32x4 vector.
func Tanh_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	var result asm.Float32x4
	asm.TanhF32(x[:], result[:])
	return result
}

// Tanh_NEON_F64x2 computes tanh(x) for a Float64x2 vector.
// Uses scalar fallback.
func Tanh_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		stdmath.Tanh(x[0]),
		stdmath.Tanh(x[1]),
	}
}

// Erf_NEON_F32x4 computes erf(x) for a Float32x4 vector.
// Uses the Abramowitz & Stegun approximation.
func Erf_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	// Use scalar implementation since we don't have vectorized erf in asm yet
	return asm.Float32x4{
		float32(stdmath.Erf(float64(x[0]))),
		float32(stdmath.Erf(float64(x[1]))),
		float32(stdmath.Erf(float64(x[2]))),
		float32(stdmath.Erf(float64(x[3]))),
	}
}

// Erf_NEON_F64x2 computes erf(x) for a Float64x2 vector.
func Erf_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		stdmath.Erf(x[0]),
		stdmath.Erf(x[1]),
	}
}

// Log_NEON_F32x4 computes natural log for a Float32x4 vector.
func Log_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	var result asm.Float32x4
	asm.LogF32(x[:], result[:])
	return result
}

// Log_NEON_F64x2 computes natural log for a Float64x2 vector.
func Log_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		stdmath.Log(x[0]),
		stdmath.Log(x[1]),
	}
}

// Sin_NEON_F32x4 computes sin(x) for a Float32x4 vector.
func Sin_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	var result asm.Float32x4
	asm.SinF32(x[:], result[:])
	return result
}

// Sin_NEON_F64x2 computes sin(x) for a Float64x2 vector.
func Sin_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		stdmath.Sin(x[0]),
		stdmath.Sin(x[1]),
	}
}

// Cos_NEON_F32x4 computes cos(x) for a Float32x4 vector.
func Cos_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	var result asm.Float32x4
	asm.CosF32(x[:], result[:])
	return result
}

// Cos_NEON_F64x2 computes cos(x) for a Float64x2 vector.
func Cos_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		stdmath.Cos(x[0]),
		stdmath.Cos(x[1]),
	}
}
