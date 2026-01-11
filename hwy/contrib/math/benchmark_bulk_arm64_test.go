//go:build arm64

package math_test

import (
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/asm"
	hwymath "github.com/ajroetker/go-highway/hwy/contrib/math"
)

// BenchmarkThroughput_ExpBulk compares bulk assembly exp vs per-element vs stdlib
func BenchmarkThroughput_ExpBulk(b *testing.B) {
	size := 16384
	input := make([]float32, size)
	output := make([]float32, size)
	for i := range input {
		input[i] = float32(i%100) * 0.1
	}

	b.Run("BulkASM", func(b *testing.B) {
		b.SetBytes(int64(size * 4)) // 4 bytes per float32
		for i := 0; i < b.N; i++ {
			asm.ExpBulkF32(input, output)
		}
	})

	b.Run("PerElement", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			hwymath.ExpPoly(input, output)
		}
	})

	b.Run("Stdlib", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			for j := range input {
				output[j] = float32(stdmath.Exp(float64(input[j])))
			}
		}
	})
}

// TestExpBulkAccuracy tests that the bulk exp function produces correct results
func TestExpBulkAccuracy(t *testing.T) {
	// Test values spanning the valid range
	input := []float32{0, 1, -1, 0.5, -0.5, 2, -2, 10, -10, 50, -50}
	output := make([]float32, len(input))

	asm.ExpBulkF32(input, output)

	for i, x := range input {
		expected := float32(stdmath.Exp(float64(x)))
		got := output[i]
		// Allow ~2% relative error for float32 (accumulates for large values)
		relErr := stdmath.Abs(float64(got-expected)) / stdmath.Abs(float64(expected))
		if relErr > 0.02 && stdmath.Abs(float64(expected)) > 1e-30 {
			t.Errorf("exp(%v): got %v, expected %v, relErr %v", x, got, expected, relErr)
		}
	}
}

// TestExpBulkEdgeCases tests overflow and underflow behavior
func TestExpBulkEdgeCases(t *testing.T) {
	input := []float32{
		100,  // overflow -> inf
		-100, // underflow -> 0
		0,    // exp(0) = 1
	}
	output := make([]float32, len(input))

	asm.ExpBulkF32(input, output)

	// Check overflow
	if !stdmath.IsInf(float64(output[0]), 1) {
		t.Errorf("exp(100): expected +Inf, got %v", output[0])
	}

	// Check underflow
	if output[1] != 0 {
		t.Errorf("exp(-100): expected 0, got %v", output[1])
	}

	// Check exp(0) = 1
	if stdmath.Abs(float64(output[2]-1)) > 1e-6 {
		t.Errorf("exp(0): expected 1, got %v", output[2])
	}
}
