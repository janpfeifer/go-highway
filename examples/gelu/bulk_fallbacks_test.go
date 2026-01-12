//go:build !arm64 || noasm

package gelu

import "testing"

// Dummy fallbacks for non-ARM64 platforms to allow tests and benchmarks to compile.

func GELUBulkF32(input, output []float32) {
	panic("GELUBulkF32 only implemented on arm64")
}

func GELUBulkF64(input, output []float64) {
	panic("GELUBulkF64 only implemented on arm64")
}

func GELUApproxBulkF32(input, output []float32) {
	panic("GELUApproxBulkF32 only implemented on arm64")
}

func GELUApproxBulkF64(input, output []float64) {
	panic("GELUApproxBulkF64 only implemented on arm64")
}

func TestDummy(t *testing.T) {}
