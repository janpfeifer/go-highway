//go:build darwin && arm64

package hwy

import "syscall"

// hasSME indicates if ARM SME (Scalable Matrix Extension) is available.
// SME is available on Apple M4 and later processors.
var hasSME = detectSME()

// detectSME checks if ARM SME is available via sysctl on macOS.
func detectSME() bool {
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_SME")
	if err != nil {
		return false
	}
	return len(val) > 0 && val[0] == 1
}

// HasSME returns true if the CPU supports ARM SME instructions.
func HasSME() bool {
	return hasSME
}
