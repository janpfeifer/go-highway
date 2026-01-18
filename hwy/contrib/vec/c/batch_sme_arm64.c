// SME Streaming Mode Batch Operations for go-highway
// Compile with: -march=armv9-a+sme
//
// These implementations use SME streaming mode to amortize the smstart/smstop
// overhead across multiple distance calculations. This is beneficial for
// large dimensions (3000+) where the per-vector compute time justifies the
// streaming mode transition cost.
//
// Apple M4 supports SME with SVL = 512 bits = 16 × float32 or 8 × float64
// Note: M4 does NOT support standalone SVE, only streaming SVE via SME.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sve.h>
#endif

// =============================================================================
// batch_dot_sme_f32: SME streaming mode batch dot product
// =============================================================================
// Computes dot products of one query vector with count data vectors.
// Each data vector has dims elements, stored consecutively.
//
// Parameters:
//   query: pointer to query vector (dims elements)
//   data: pointer to data matrix (count × dims elements, row-major)
//   dots: output buffer (count elements)
//   count: number of data vectors
//   dims: dimensionality of each vector
//
// For i in [0, count):
//   dots[i] = sum(query[j] * data[i*dims + j] for j in [0, dims))
//
// Uses streaming mode to amortize smstart/smstop overhead across all vectors.
// SVL=512 bits processes 16 floats per iteration.
//
// func batch_dot_sme_f32(query, data, dots unsafe.Pointer, count, dims int64)
void batch_dot_sme_f32(float *query, float *data, float *dots,
                       long *pcount, long *pdims) __arm_streaming {
    long count = *pcount;
    long dims = *pdims;

    if (count <= 0 || dims <= 0) {
        return;
    }

    svbool_t pg = svptrue_b32();
    long lanes = 16;  // SVL=512 / 32 bits = 16 lanes

    // Process each data vector
    for (long i = 0; i < count; i++) {
        float *data_vec = data + i * dims;

        // Initialize 4 accumulators for better ILP
        svfloat32_t acc0 = svdup_f32(0.0f);
        svfloat32_t acc1 = svdup_f32(0.0f);
        svfloat32_t acc2 = svdup_f32(0.0f);
        svfloat32_t acc3 = svdup_f32(0.0f);

        // Main vectorized loop with 4x unrolling
        long j = 0;
        long stride = lanes * 4;
        for (; j + stride <= dims; j += stride) {
            // Load query vectors
            svfloat32_t q0 = svld1_f32(pg, query + j);
            svfloat32_t q1 = svld1_f32(pg, query + j + lanes);
            svfloat32_t q2 = svld1_f32(pg, query + j + lanes * 2);
            svfloat32_t q3 = svld1_f32(pg, query + j + lanes * 3);

            // Load data vectors
            svfloat32_t d0 = svld1_f32(pg, data_vec + j);
            svfloat32_t d1 = svld1_f32(pg, data_vec + j + lanes);
            svfloat32_t d2 = svld1_f32(pg, data_vec + j + lanes * 2);
            svfloat32_t d3 = svld1_f32(pg, data_vec + j + lanes * 3);

            // FMA: acc += q * d
            acc0 = svmla_f32_x(pg, acc0, q0, d0);
            acc1 = svmla_f32_x(pg, acc1, q1, d1);
            acc2 = svmla_f32_x(pg, acc2, q2, d2);
            acc3 = svmla_f32_x(pg, acc3, q3, d3);
        }

        // Process remaining full vectors
        for (; j + lanes <= dims; j += lanes) {
            svfloat32_t q = svld1_f32(pg, query + j);
            svfloat32_t d = svld1_f32(pg, data_vec + j);
            acc0 = svmla_f32_x(pg, acc0, q, d);
        }

        // Combine accumulators
        acc0 = svadd_f32_x(pg, acc0, acc1);
        acc2 = svadd_f32_x(pg, acc2, acc3);
        acc0 = svadd_f32_x(pg, acc0, acc2);

        // Horizontal reduction
        float result = svaddv_f32(pg, acc0);

        // Scalar tail
        for (; j < dims; j++) {
            result += query[j] * data_vec[j];
        }

        dots[i] = result;
    }
}

// =============================================================================
// batch_dot_sme_f64: SME streaming mode batch dot product for float64
// =============================================================================
// SVL=512 bits processes 8 doubles per iteration.
//
// func batch_dot_sme_f64(query, data, dots unsafe.Pointer, count, dims int64)
void batch_dot_sme_f64(double *query, double *data, double *dots,
                       long *pcount, long *pdims) __arm_streaming {
    long count = *pcount;
    long dims = *pdims;

    if (count <= 0 || dims <= 0) {
        return;
    }

    svbool_t pg = svptrue_b64();
    long lanes = 8;  // SVL=512 / 64 bits = 8 lanes

    // Process each data vector
    for (long i = 0; i < count; i++) {
        double *data_vec = data + i * dims;

        // Initialize 4 accumulators
        svfloat64_t acc0 = svdup_f64(0.0);
        svfloat64_t acc1 = svdup_f64(0.0);
        svfloat64_t acc2 = svdup_f64(0.0);
        svfloat64_t acc3 = svdup_f64(0.0);

        // Main vectorized loop with 4x unrolling
        long j = 0;
        long stride = lanes * 4;
        for (; j + stride <= dims; j += stride) {
            svfloat64_t q0 = svld1_f64(pg, query + j);
            svfloat64_t q1 = svld1_f64(pg, query + j + lanes);
            svfloat64_t q2 = svld1_f64(pg, query + j + lanes * 2);
            svfloat64_t q3 = svld1_f64(pg, query + j + lanes * 3);

            svfloat64_t d0 = svld1_f64(pg, data_vec + j);
            svfloat64_t d1 = svld1_f64(pg, data_vec + j + lanes);
            svfloat64_t d2 = svld1_f64(pg, data_vec + j + lanes * 2);
            svfloat64_t d3 = svld1_f64(pg, data_vec + j + lanes * 3);

            acc0 = svmla_f64_x(pg, acc0, q0, d0);
            acc1 = svmla_f64_x(pg, acc1, q1, d1);
            acc2 = svmla_f64_x(pg, acc2, q2, d2);
            acc3 = svmla_f64_x(pg, acc3, q3, d3);
        }

        // Process remaining full vectors
        for (; j + lanes <= dims; j += lanes) {
            svfloat64_t q = svld1_f64(pg, query + j);
            svfloat64_t d = svld1_f64(pg, data_vec + j);
            acc0 = svmla_f64_x(pg, acc0, q, d);
        }

        // Combine accumulators
        acc0 = svadd_f64_x(pg, acc0, acc1);
        acc2 = svadd_f64_x(pg, acc2, acc3);
        acc0 = svadd_f64_x(pg, acc0, acc2);

        // Horizontal reduction
        double result = svaddv_f64(pg, acc0);

        // Scalar tail
        for (; j < dims; j++) {
            result += query[j] * data_vec[j];
        }

        dots[i] = result;
    }
}

// =============================================================================
// batch_l2_squared_sme_f32: SME streaming mode batch L2 squared distance
// =============================================================================
// Computes L2 squared distances from one query to count data vectors.
//
// For i in [0, count):
//   distances[i] = sum((query[j] - data[i*dims + j])^2 for j in [0, dims))
//
// func batch_l2_squared_sme_f32(query, data, distances unsafe.Pointer, count, dims int64)
void batch_l2_squared_sme_f32(float *query, float *data, float *distances,
                               long *pcount, long *pdims) __arm_streaming {
    long count = *pcount;
    long dims = *pdims;

    if (count <= 0 || dims <= 0) {
        return;
    }

    svbool_t pg = svptrue_b32();
    long lanes = 16;

    // Process each data vector
    for (long i = 0; i < count; i++) {
        float *data_vec = data + i * dims;

        // Initialize 4 accumulators
        svfloat32_t acc0 = svdup_f32(0.0f);
        svfloat32_t acc1 = svdup_f32(0.0f);
        svfloat32_t acc2 = svdup_f32(0.0f);
        svfloat32_t acc3 = svdup_f32(0.0f);

        // Main vectorized loop with 4x unrolling
        long j = 0;
        long stride = lanes * 4;
        for (; j + stride <= dims; j += stride) {
            svfloat32_t q0 = svld1_f32(pg, query + j);
            svfloat32_t q1 = svld1_f32(pg, query + j + lanes);
            svfloat32_t q2 = svld1_f32(pg, query + j + lanes * 2);
            svfloat32_t q3 = svld1_f32(pg, query + j + lanes * 3);

            svfloat32_t d0 = svld1_f32(pg, data_vec + j);
            svfloat32_t d1 = svld1_f32(pg, data_vec + j + lanes);
            svfloat32_t d2 = svld1_f32(pg, data_vec + j + lanes * 2);
            svfloat32_t d3 = svld1_f32(pg, data_vec + j + lanes * 3);

            // diff = q - d
            svfloat32_t diff0 = svsub_f32_x(pg, q0, d0);
            svfloat32_t diff1 = svsub_f32_x(pg, q1, d1);
            svfloat32_t diff2 = svsub_f32_x(pg, q2, d2);
            svfloat32_t diff3 = svsub_f32_x(pg, q3, d3);

            // acc += diff * diff
            acc0 = svmla_f32_x(pg, acc0, diff0, diff0);
            acc1 = svmla_f32_x(pg, acc1, diff1, diff1);
            acc2 = svmla_f32_x(pg, acc2, diff2, diff2);
            acc3 = svmla_f32_x(pg, acc3, diff3, diff3);
        }

        // Process remaining full vectors
        for (; j + lanes <= dims; j += lanes) {
            svfloat32_t q = svld1_f32(pg, query + j);
            svfloat32_t d = svld1_f32(pg, data_vec + j);
            svfloat32_t diff = svsub_f32_x(pg, q, d);
            acc0 = svmla_f32_x(pg, acc0, diff, diff);
        }

        // Combine accumulators
        acc0 = svadd_f32_x(pg, acc0, acc1);
        acc2 = svadd_f32_x(pg, acc2, acc3);
        acc0 = svadd_f32_x(pg, acc0, acc2);

        // Horizontal reduction
        float result = svaddv_f32(pg, acc0);

        // Scalar tail
        for (; j < dims; j++) {
            float diff = query[j] - data_vec[j];
            result += diff * diff;
        }

        distances[i] = result;
    }
}

// =============================================================================
// batch_l2_squared_sme_f64: SME streaming mode batch L2 squared distance for float64
// =============================================================================
// func batch_l2_squared_sme_f64(query, data, distances unsafe.Pointer, count, dims int64)
void batch_l2_squared_sme_f64(double *query, double *data, double *distances,
                               long *pcount, long *pdims) __arm_streaming {
    long count = *pcount;
    long dims = *pdims;

    if (count <= 0 || dims <= 0) {
        return;
    }

    svbool_t pg = svptrue_b64();
    long lanes = 8;

    // Process each data vector
    for (long i = 0; i < count; i++) {
        double *data_vec = data + i * dims;

        // Initialize 4 accumulators
        svfloat64_t acc0 = svdup_f64(0.0);
        svfloat64_t acc1 = svdup_f64(0.0);
        svfloat64_t acc2 = svdup_f64(0.0);
        svfloat64_t acc3 = svdup_f64(0.0);

        // Main vectorized loop with 4x unrolling
        long j = 0;
        long stride = lanes * 4;
        for (; j + stride <= dims; j += stride) {
            svfloat64_t q0 = svld1_f64(pg, query + j);
            svfloat64_t q1 = svld1_f64(pg, query + j + lanes);
            svfloat64_t q2 = svld1_f64(pg, query + j + lanes * 2);
            svfloat64_t q3 = svld1_f64(pg, query + j + lanes * 3);

            svfloat64_t d0 = svld1_f64(pg, data_vec + j);
            svfloat64_t d1 = svld1_f64(pg, data_vec + j + lanes);
            svfloat64_t d2 = svld1_f64(pg, data_vec + j + lanes * 2);
            svfloat64_t d3 = svld1_f64(pg, data_vec + j + lanes * 3);

            svfloat64_t diff0 = svsub_f64_x(pg, q0, d0);
            svfloat64_t diff1 = svsub_f64_x(pg, q1, d1);
            svfloat64_t diff2 = svsub_f64_x(pg, q2, d2);
            svfloat64_t diff3 = svsub_f64_x(pg, q3, d3);

            acc0 = svmla_f64_x(pg, acc0, diff0, diff0);
            acc1 = svmla_f64_x(pg, acc1, diff1, diff1);
            acc2 = svmla_f64_x(pg, acc2, diff2, diff2);
            acc3 = svmla_f64_x(pg, acc3, diff3, diff3);
        }

        // Process remaining full vectors
        for (; j + lanes <= dims; j += lanes) {
            svfloat64_t q = svld1_f64(pg, query + j);
            svfloat64_t d = svld1_f64(pg, data_vec + j);
            svfloat64_t diff = svsub_f64_x(pg, q, d);
            acc0 = svmla_f64_x(pg, acc0, diff, diff);
        }

        // Combine accumulators
        acc0 = svadd_f64_x(pg, acc0, acc1);
        acc2 = svadd_f64_x(pg, acc2, acc3);
        acc0 = svadd_f64_x(pg, acc0, acc2);

        // Horizontal reduction
        double result = svaddv_f64(pg, acc0);

        // Scalar tail
        for (; j < dims; j++) {
            double diff = query[j] - data_vec[j];
            result += diff * diff;
        }

        distances[i] = result;
    }
}
