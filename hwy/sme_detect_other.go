//go:build !darwin || !arm64

package hwy

// hasSME is false on non-darwin or non-arm64 platforms.
// SME is currently only supported on Apple M4+ via macOS.
var hasSME = false

// HasSME returns true if the CPU supports ARM SME instructions.
// On non-darwin platforms, this always returns false.
func HasSME() bool {
	return false
}
