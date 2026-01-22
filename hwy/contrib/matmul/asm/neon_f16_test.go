// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && arm64

package asm

import (
	"fmt"
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// matmulReferenceF16Simple is a simple reference implementation for testing
func matmulReferenceF16Simple(a, b, c []hwy.Float16, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p].Float32() * b[p*n+j].Float32()
			}
			c[i*n+j] = hwy.NewFloat16(sum)
		}
	}
}

// TestMatMulNEONF16 tests the NEON F16 matrix multiplication assembly
func TestMatMulNEONF16(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	// Test various sizes - must be multiples of 8 for NEON
	sizes := []int{16, 24, 32, 48, 64}

	for _, size := range sizes {
		m, n, k := size, size, size
		t.Run(fmt.Sprintf("%dx%d", size, size), func(t *testing.T) {
			a := make([]hwy.Float16, m*k)
			b := make([]hwy.Float16, k*n)
			c := make([]hwy.Float16, m*n)
			expected := make([]hwy.Float16, m*n)

			// Fill with test values
			for i := range a {
				a[i] = hwy.NewFloat16(float32(i%7) + 0.5)
			}
			for i := range b {
				b[i] = hwy.NewFloat16(float32(i%11) + 0.25)
			}

			// Reference implementation
			matmulReferenceF16Simple(a, b, expected, m, n, k)

			// NEON implementation
			MatMulNEONF16(a, b, c, m, n, k)

			var maxErr float32
			var maxErrIdx int
			for i := range c {
				err := float32(math.Abs(float64(c[i].Float32() - expected[i].Float32())))
				if err > maxErr {
					maxErr = err
					maxErrIdx = i
				}
			}
			t.Logf("size %dx%d: max error %e at index %d", size, size, maxErr, maxErrIdx)

			// f16 has less precision, allow tolerance proportional to k
			tolerance := float32(0.1) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
				// Print some context around the error
				row := maxErrIdx / n
				col := maxErrIdx % n
				t.Logf("  at [%d,%d]: expected=%f, got=%f",
					row, col, expected[maxErrIdx].Float32(), c[maxErrIdx].Float32())
			}
		})
	}
}

// TestMatMulNEONF16Small tests with the minimum size that bypasses fallback (16)
func TestMatMulNEONF16Small(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	// Test exactly at the NEON threshold
	m, n, k := 16, 16, 16

	a := make([]hwy.Float16, m*k)
	b := make([]hwy.Float16, k*n)
	c := make([]hwy.Float16, m*n)
	expected := make([]hwy.Float16, m*n)

	// Simple identity-ish test: A = identity-like, B = simple values
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			if i == j {
				a[i*k+j] = hwy.NewFloat16(1.0)
			} else {
				a[i*k+j] = hwy.NewFloat16(0.0)
			}
		}
	}
	for i := range b {
		b[i] = hwy.NewFloat16(float32(i + 1))
	}

	// Reference
	matmulReferenceF16Simple(a, b, expected, m, n, k)

	// NEON
	MatMulNEONF16(a, b, c, m, n, k)

	// With identity A, C should equal B
	for i := range c {
		exp := expected[i].Float32()
		got := c[i].Float32()
		if math.Abs(float64(exp-got)) > 0.01 {
			row := i / n
			col := i % n
			t.Errorf("[%d,%d]: expected=%f, got=%f", row, col, exp, got)
		}
	}
}

// TestMatMulNEONF16_64 specifically tests size 64 which crashes in benchmarks
func TestMatMulNEONF16_64(t *testing.T) {
	if !hwy.HasARMFP16() {
		t.Skip("CPU does not support ARM FP16")
	}

	m, n, k := 64, 64, 64

	a := make([]hwy.Float16, m*k)
	b := make([]hwy.Float16, k*n)
	c := make([]hwy.Float16, m*n)
	expected := make([]hwy.Float16, m*n)

	// Same initialization as benchmark
	for i := range a {
		a[i] = hwy.NewFloat16(float32(i%7) + 0.5)
	}
	for i := range b {
		b[i] = hwy.NewFloat16(float32(i%11) + 0.25)
	}

	// Reference
	matmulReferenceF16Simple(a, b, expected, m, n, k)

	// This is the exact call that crashes in benchmarks
	t.Log("Calling MatMulNEONF16 with 64x64 matrices...")
	MatMulNEONF16(a, b, c, m, n, k)
	t.Log("MatMulNEONF16 returned successfully")

	var maxErr float32
	for i := range c {
		err := float32(math.Abs(float64(c[i].Float32() - expected[i].Float32())))
		if err > maxErr {
			maxErr = err
		}
	}
	t.Logf("max error: %e", maxErr)

	tolerance := float32(0.1) * float32(k)
	if maxErr > tolerance {
		t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
	}
}
