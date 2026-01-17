// SME Matrix Multiplication for go-highway
// Compile with: -march=armv9-a+sme
//
// This implements the same algorithms as the hand-written assembly in
// hwy/contrib/matmul/matmul_fmopa_at_arm64.s
//
// The generated assembly can be compared against the hand-written version
// to verify correctness and optimize.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// =============================================================================
// matmul_fmopa_at_f32: FMOPA-based matrix multiply with transposed A
// =============================================================================
// Computes C = A * B where:
//   AT is K x M (A transposed, row-major) - for contiguous column access
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses 16x16 tile processing with FMOPA outer product accumulate.
// Requires M, N to be multiples of 16.
//
// This mirrors: hwy/contrib/matmul/matmul_fmopa_at_arm64.s
//
// func matmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)
void matmul_fmopa_at_f32(float *at, float *b, float *c,
                          long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process output in 16x16 tiles
    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column from transposed AT: AT[kk, ti:ti+16]
                // This is contiguous in memory!
                svfloat32_t za_col = svld1_f32(svptrue_b32(), at + kk * m + ti);

                // Load B row: B[kk, tj:tj+16]
                svfloat32_t zb_row = svld1_f32(svptrue_b32(), b + kk * n + tj);

                // Outer product accumulate: ZA0 += za_col * zb_row^T
                // This computes a 16x16 tile contribution in one instruction
                svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za_col, zb_row);
            }

            // Store result tile to C[ti:ti+16, tj:tj+16]
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
                svst1_f32(svptrue_b32(), c + (ti + row) * n + tj, zrow);
            }
        }
    }
}

// =============================================================================
// matmul_fmopa_at_f64: FMOPA-based matrix multiply for float64
// =============================================================================
// Same algorithm but with 8x8 tiles for float64.
// Apple M4 SVL = 512 bits = 8 x float64
//
// This mirrors: hwy/contrib/matmul/matmul_fmopa_at_f64_arm64.s
//
// func matmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)
void matmul_fmopa_at_f64(double *at, double *b, double *c,
                          long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process output in 8x8 tiles (float64 uses half the lanes)
    for (long ti = 0; ti < m; ti += 8) {
        for (long tj = 0; tj < n; tj += 8) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column from transposed AT: AT[kk, ti:ti+8]
                svfloat64_t za_col = svld1_f64(svptrue_b64(), at + kk * m + ti);

                // Load B row: B[kk, tj:tj+8]
                svfloat64_t zb_row = svld1_f64(svptrue_b64(), b + kk * n + tj);

                // Outer product accumulate for float64
                svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), za_col, zb_row);
            }

            // Store result tile to C[ti:ti+8, tj:tj+8]
            for (int row = 0; row < 8; row++) {
                svfloat64_t zrow = svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), 0, row);
                svst1_f64(svptrue_b64(), c + (ti + row) * n + tj, zrow);
            }
        }
    }
}
