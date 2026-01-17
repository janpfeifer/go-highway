//go:build !noasm && amd64

package matvec

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matvec/asm"
)

// Override F16/BF16 dispatch to use GoAT-generated AVX assembly.
// Go's archsimd doesn't support Float16/BFloat16, so we use C→assembly via GoAT.
// F32/F64 continue to use the hwygen-generated code with archsimd.

// Adapter functions to match dispatch signature: (m, rows, cols, v, result)
// The asm functions use: (m, v, result, rows, cols)

func matvecAVX512F16(m []hwy.Float16, rows, cols int, v, result []hwy.Float16) {
	asm.MatVecAVX512F16(m, v, result, rows, cols)
}

func matvecAVX2F16(m []hwy.Float16, rows, cols int, v, result []hwy.Float16) {
	asm.MatVecAVX2F16(m, v, result, rows, cols)
}

func matvecAVX512BF16(m []hwy.BFloat16, rows, cols int, v, result []hwy.BFloat16) {
	asm.MatVecAVX512BF16(m, v, result, rows, cols)
}

func matvecAVX2BF16(m []hwy.BFloat16, rows, cols int, v, result []hwy.BFloat16) {
	asm.MatVecAVX2BF16(m, v, result, rows, cols)
}

func init() {
	level := hwy.CurrentLevel()

	// Float16 dispatch
	if level == hwy.DispatchAVX512 && hwy.HasAVX512FP16() {
		// AVX-512 with native FP16 support (Sapphire Rapids+)
		MatVecFloat16 = matvecAVX512F16
	} else if level >= hwy.DispatchAVX2 && hwy.HasF16C() {
		// AVX2 with F16C for f16<->f32 conversion
		MatVecFloat16 = matvecAVX2F16
	}

	// BFloat16 dispatch
	if level == hwy.DispatchAVX512 && hwy.HasAVX512BF16() {
		// AVX-512 with native BF16 support (Cooper Lake+)
		MatVecBFloat16 = matvecAVX512BF16
	} else if level >= hwy.DispatchAVX2 {
		// AVX2 emulates bf16 via f32
		MatVecBFloat16 = matvecAVX2BF16
	}
}
