//go:build !noasm && arm64

// Multi-Tile SME FMOPA Matrix Multiplication wrappers for ARM64
package asm

import "unsafe"

// Generate assembly from C using goat
//go:generate go tool goat ../c/multitile_fmopa_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-f64f64"

// MultiTileMatMulFMOPAF32 performs multi-tile matrix multiplication using SME FMOPA: C = AT^T * B
// Uses all 4 ZA tiles (ZA0-ZA3) in a 2x2 arrangement for 32x32 output blocks,
// with single-tile fallback for 16-row/16-col remainders.
//
// AT is the transposed A matrix (K x M, row-major).
// B is K x N (row-major), C is M x N (row-major).
// Requires M, N to be multiples of 16.
func MultiTileMatMulFMOPAF32(at, b, c []float32, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	multitile_fmopa_at_f32(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}

// MultiTileMatMulFMOPAF64 performs multi-tile matrix multiplication using SME FMOPA: C = AT^T * B
// Uses all 4 ZA tiles in a 2x2 arrangement for 16x16 output blocks (8x8 per tile),
// with single-tile fallback for 8-row/8-col remainders.
//
// Requires M, N to be multiples of 8.
func MultiTileMatMulFMOPAF64(at, b, c []float64, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(at) < k*m || len(b) < k*n || len(c) < m*n {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	multitile_fmopa_at_f64(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}
