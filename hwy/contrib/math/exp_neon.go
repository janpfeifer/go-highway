//go:build arm64

package math

import (
	"github.com/ajroetker/go-highway/hwy/asm"
	stdmath "math"
)

// Exp_NEON_F32x4 computes e^x for a Float32x4 vector using NEON.
func Exp_NEON_F32x4(x asm.Float32x4) asm.Float32x4 {
	var result asm.Float32x4
	asm.ExpF32(x[:], result[:])
	return result
}

// Exp_NEON_F64x2 computes e^x for a Float64x2 vector.
// Uses scalar fallback since we don't have F64 exp in asm yet.
func Exp_NEON_F64x2(x asm.Float64x2) asm.Float64x2 {
	return asm.Float64x2{
		stdmath.Exp(x[0]),
		stdmath.Exp(x[1]),
	}
}
