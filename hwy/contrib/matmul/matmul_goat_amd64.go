//go:build !noasm && amd64

package matmul

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// Override F16/BF16 dispatch to use GoAT-generated AVX assembly.
// Go's archsimd doesn't support Float16/BFloat16, so we use C→assembly via GoAT.
// F32/F64 continue to use the hwygen-generated code with archsimd.

func init() {
	level := hwy.CurrentLevel()

	// Float16 dispatch
	if level == hwy.DispatchAVX512 && hwy.HasAVX512FP16() {
		// AVX-512 with native FP16 support (Sapphire Rapids+)
		MatMulFloat16 = asm.MatMulAVX512F16
	} else if level >= hwy.DispatchAVX2 && hwy.HasF16C() {
		// AVX2 with F16C for f16<->f32 conversion
		MatMulFloat16 = asm.MatMulAVX2F16
	}

	// BFloat16 dispatch
	if level == hwy.DispatchAVX512 && hwy.HasAVX512BF16() {
		// AVX-512 with native BF16 support (Cooper Lake+)
		MatMulBFloat16 = asm.MatMulAVX512BF16
	} else if level >= hwy.DispatchAVX2 {
		// AVX2 emulates bf16 via f32
		MatMulBFloat16 = asm.MatMulAVX2BF16
	}
}
