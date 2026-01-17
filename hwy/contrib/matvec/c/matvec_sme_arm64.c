// SME Matrix-Vector Multiplication for go-highway
// Compile with: -march=armv9-a+sme
//
// This implements the same algorithms as the hand-written assembly in
// hwy/contrib/matvec/matvec_sme_arm64.s
//
// The generated assembly can be compared against the hand-written version
// to verify correctness and optimize.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// =============================================================================
// matvec_sme_f32: FMOPA-based matrix-vector multiplication
// =============================================================================
// Computes result = M * v where:
//   MT is cols x rows (M transposed, row-major) - for contiguous column access
//   v is cols (input vector)
//   result is rows (output vector)
//
// With M transposed, loading M columns becomes contiguous!
//   M[row:row+16, k] = MT[k, row:row+16] = contiguous in memory
//
// Algorithm (using outer product like matmul):
//   For each 16-row tile:
//     zero ZA
//     for k in 0..cols:
//       z2 = MT[k, row:row+16]  (contiguous load - like A column in matmul)
//       z0 = broadcast(v[k])   (like B row, but broadcast scalar)
//       FMOPA za0 += z2 ⊗ z0
//     Extract za0 rows and store first element of each to result
//
// Apple M4 SVL = 512 bits = 16 × float32
// ZA tiles = 16×16 = 256 float32 elements
//
// This mirrors: hwy/contrib/matvec/matvec_sme_arm64.s
//
// func matvec_sme_f32(mt, v, result unsafe.Pointer, rows, cols int64)
void matvec_sme_f32(float *mt, float *v, float *result,
                     long *prows, long *pcols) __arm_streaming __arm_out("za") {
    long rows = *prows;
    long cols = *pcols;

    // Process 16 rows at a time
    for (long row = 0; row < rows; row += 16) {
        // Zero accumulator tile
        svzero_za();

        // Inner loop: accumulate over columns
        for (long k = 0; k < cols; k++) {
            // Load M column: M[row:row+16, k] from transposed MT[k, row:row+16]
            // This is contiguous at: MT_base + k * rows + row
            svfloat32_t z_col = svld1_f32(svptrue_b32(), mt + k * rows + row);

            // Broadcast v[k] to all lanes
            svfloat32_t z_v = svdup_f32(v[k]);

            // Outer product accumulate: ZA0 += z_col * z_v^T
            // Since z_v is broadcast, za[i][j] = sum_k(M[row+i, k] * v[k]) for all j
            svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), z_col, z_v);
        }

        // Extract horizontal slices from za0 and store first element of each
        // After accumulation, za[i][j] = dot(row i, v) for all j (all columns same)
        for (int i = 0; i < 16; i++) {
            svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, i);
            // Extract first element and store
            result[row + i] = svlasta_f32(svptrue_b32(), zrow);
        }
    }
}

// =============================================================================
// matvec_sme_f64: FMOPA-based matrix-vector multiplication for float64
// =============================================================================
// Same algorithm but with 8-element tiles for float64.
// Apple M4 SVL = 512 bits = 8 × float64
// ZA tiles = 8×8 = 64 float64 elements
//
// This mirrors: hwy/contrib/matvec/matvec_sme_arm64.s (f64 version)
//
// func matvec_sme_f64(mt, v, result unsafe.Pointer, rows, cols int64)
void matvec_sme_f64(double *mt, double *v, double *result,
                     long *prows, long *pcols) __arm_streaming __arm_out("za") {
    long rows = *prows;
    long cols = *pcols;

    // Process 8 rows at a time (float64 has half the lanes)
    for (long row = 0; row < rows; row += 8) {
        // Zero accumulator tile
        svzero_za();

        // Inner loop: accumulate over columns
        for (long k = 0; k < cols; k++) {
            // Load M column from transposed MT[k, row:row+8]
            svfloat64_t z_col = svld1_f64(svptrue_b64(), mt + k * rows + row);

            // Broadcast v[k] to all lanes
            svfloat64_t z_v = svdup_f64(v[k]);

            // Outer product accumulate for float64
            svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), z_col, z_v);
        }

        // Extract and store first element of each row
        for (int i = 0; i < 8; i++) {
            svfloat64_t zrow = svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), 0, i);
            // Extract first element and store
            result[row + i] = svlasta_f64(svptrue_b64(), zrow);
        }
    }
}
