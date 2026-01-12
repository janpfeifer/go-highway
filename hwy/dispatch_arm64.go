//go:build arm64

package hwy

import (
	"os"

	"golang.org/x/sys/cpu"
)

func init() {
	// Check for HWY_NO_SIMD environment variable first
	if NoSimdEnv() {
		currentLevel = DispatchScalar
		currentWidth = 16
		currentName = "scalar"
		return
	}

	// ARM64 (AArch64) always has NEON (ASIMD) available.
	// It's part of the ARMv8-A base architecture.
	// We still check the cpu package for future SVE support.

	// Note: cpu.ARM64.HasASIMD is always true for ARMv8+
	// We check it for consistency and to enable SVE detection later.
	if cpu.ARM64.HasASIMD {
		currentLevel = DispatchNEON
		currentWidth = 16 // NEON is 128-bit (16 bytes)
		currentName = "neon"
	} else {
		// Fallback to scalar (should never happen on ARMv8+)
		currentLevel = DispatchScalar
		currentWidth = 16
		currentName = "scalar"
	}

	// SME support (Apple M4+)
	// Check for HWY_NO_SME environment variable to disable SME
	if hasSME && os.Getenv("HWY_NO_SME") == "" {
		currentLevel = DispatchSME
		currentWidth = 64 // SME streaming vector length is 512-bit (64 bytes) on M4
		currentName = "sme"
	}

	// Future: SVE support (without SME streaming mode)
	// if cpu.ARM64.HasSVE {
	//     currentLevel = DispatchSVE
	//     currentWidth = ... // SVE width is variable
	//     currentName = "sve"
	// }
}
