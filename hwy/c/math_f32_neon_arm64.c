/*
 * Copyright 2025 go-highway Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Float32 transcendental math functions for ARM64 NEON
// Tan, Atan, Atan2, Pow, Erf, Exp2, Log2

#include <arm_neon.h>

// Tan float32: result[i] = tan(input[i])
// Uses range reduction and computes sin(x)/cos(x)
// Only processes multiples of 4 elements (SIMD only)
void tan_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants (using bit patterns for non-fmov-immediate values)
    // pi = 3.14159265f, bits: 0x40490FDB
    // half_pi = 1.5707963f, bits: 0x3FC90FDB
    // inv_pi = 0.31830989f, bits: 0x3EA2F983
    float32x4_t v_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x40490FDB));
    float32x4_t v_neg_pi = vnegq_f32(v_pi);
    float32x4_t v_half_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3FC90FDB));
    float32x4_t v_neg_half_pi = vnegq_f32(v_half_pi);
    float32x4_t v_inv_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3EA2F983));
    float32x4_t v_two = vdupq_n_f32(2.0f);
    float32x4_t v_neg_one = vdupq_n_f32(-1.0f);
    float32x4_t v_one = vdupq_n_f32(1.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float32x4_t k = vrndnq_f32(vmulq_f32(x, vmulq_f32(vdupq_n_f32(0.5f), v_inv_pi)));
        x = vfmsq_f32(x, k, vmulq_f32(v_two, v_pi));

        // For sin: reflection to [-pi/2, pi/2]
        float32x4_t sin_x = x;
        uint32x4_t need_pos_reflect = vcgtq_f32(sin_x, v_half_pi);
        uint32x4_t need_neg_reflect = vcltq_f32(sin_x, v_neg_half_pi);
        float32x4_t sin_x_pos_reflected = vsubq_f32(v_pi, sin_x);
        float32x4_t sin_x_neg_reflected = vsubq_f32(v_neg_pi, sin_x);
        sin_x = vbslq_f32(need_pos_reflect, sin_x_pos_reflected, sin_x);
        sin_x = vbslq_f32(need_neg_reflect, sin_x_neg_reflected, sin_x);

        // Compute sin(x) polynomial: s7=-0.0001984127f, s5=0.00833333f, s3=-0.16666667f, s1=1.0f
        float32x4_t sin_x2 = vmulq_f32(sin_x, sin_x);
        float32x4_t sin_p = vdupq_n_f32(-0.0001984127f);
        sin_p = vfmaq_f32(vdupq_n_f32(0.00833333f), sin_p, sin_x2);
        sin_p = vfmaq_f32(vdupq_n_f32(-0.16666667f), sin_p, sin_x2);
        sin_p = vfmaq_f32(vdupq_n_f32(1.0f), sin_p, sin_x2);
        float32x4_t sin_val = vmulq_f32(sin_p, sin_x);

        // For cos: use cos(x) = cos(|x|) and reflection
        float32x4_t cos_x = vabsq_f32(x);
        uint32x4_t cos_need_reflect = vcgtq_f32(cos_x, v_half_pi);
        float32x4_t cos_x_reflected = vsubq_f32(v_pi, cos_x);
        cos_x = vbslq_f32(cos_need_reflect, cos_x_reflected, cos_x);
        float32x4_t cos_sign = vbslq_f32(cos_need_reflect, v_neg_one, v_one);

        // Compute cos(x) polynomial: c6=-0.00138889f, c4=0.04166667f, c2=-0.5f, c0=1.0f
        float32x4_t cos_x2 = vmulq_f32(cos_x, cos_x);
        float32x4_t cos_p = vdupq_n_f32(-0.00138889f);
        cos_p = vfmaq_f32(vdupq_n_f32(0.04166667f), cos_p, cos_x2);
        cos_p = vfmaq_f32(vdupq_n_f32(-0.5f), cos_p, cos_x2);
        cos_p = vfmaq_f32(vdupq_n_f32(1.0f), cos_p, cos_x2);
        float32x4_t cos_val = vmulq_f32(cos_p, cos_sign);

        // tan(x) = sin(x) / cos(x)
        float32x4_t tan_val = vdivq_f32(sin_val, cos_val);

        vst1q_f32(result + i, tan_val);
    }
}

// Atan float32: result[i] = atan(input[i])
// Uses two-level range reduction for better accuracy:
// 1. If |x| > 1: use atan(x) = pi/2 - atan(1/x)
// 2. If |x| > tan(pi/8) ≈ 0.414: use atan(x) = pi/4 + atan((x-1)/(x+1))
void atan_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    // half_pi = 1.5707963f, quarter_pi = 0.7853982f
    float32x4_t v_half_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3FC90FDB));
    float32x4_t v_quarter_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3F490FDB));
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_tan_pi_8 = vdupq_n_f32(0.4142135623730950488f); // tan(pi/8) = sqrt(2) - 1

    // Optimized polynomial coefficients for atan(x) on [0, tan(pi/8)]
    // Higher accuracy over smaller range
    const float c1 = -0.3333333333f;
    const float c2 = 0.2f;
    const float c3 = -0.1428571429f;
    const float c4 = 0.1111111111f;
    const float c5 = -0.0909090909f;

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Get sign and work with absolute value
        uint32x4_t is_negative = vcltq_f32(x, v_zero);
        float32x4_t abs_x = vabsq_f32(x);

        // Range reduction level 1: if |x| > 1, use atan(x) = pi/2 - atan(1/x)
        uint32x4_t use_reciprocal = vcgtq_f32(abs_x, v_one);
        float32x4_t recip_x = vdivq_f32(v_one, abs_x);
        float32x4_t reduced_x = vbslq_f32(use_reciprocal, recip_x, abs_x);

        // Range reduction level 2: if reduced_x > tan(pi/8), use identity
        // atan(x) = pi/4 + atan((x-1)/(x+1))
        uint32x4_t use_identity = vcgtq_f32(reduced_x, v_tan_pi_8);
        float32x4_t x_minus_1 = vsubq_f32(reduced_x, v_one);
        float32x4_t x_plus_1 = vaddq_f32(reduced_x, v_one);
        float32x4_t transformed_x = vdivq_f32(x_minus_1, x_plus_1);
        reduced_x = vbslq_f32(use_identity, transformed_x, reduced_x);

        // Compute atan(reduced_x) using polynomial (now x is in [0, tan(pi/8)])
        // atan(x) ≈ x * (1 + x² * (c1 + x² * (c2 + x² * (c3 + x² * (c4 + x² * c5)))))
        float32x4_t x2 = vmulq_f32(reduced_x, reduced_x);

        // Evaluate polynomial using Horner's method
        float32x4_t p = vdupq_n_f32(c5);
        p = vfmaq_f32(vdupq_n_f32(c4), p, x2);
        p = vfmaq_f32(vdupq_n_f32(c3), p, x2);
        p = vfmaq_f32(vdupq_n_f32(c2), p, x2);
        p = vfmaq_f32(vdupq_n_f32(c1), p, x2);
        p = vfmaq_f32(v_one, p, x2);
        float32x4_t atan_core = vmulq_f32(p, reduced_x);

        // Adjust for identity: add pi/4 if we used the (x-1)/(x+1) transform
        float32x4_t atan_reduced = vbslq_f32(use_identity,
            vaddq_f32(v_quarter_pi, atan_core),
            atan_core);

        // Adjust for reciprocal: if |x| > 1, result = pi/2 - atan(1/x)
        float32x4_t atan_full = vbslq_f32(use_reciprocal,
            vsubq_f32(v_half_pi, atan_reduced),
            atan_reduced);

        // Apply original sign
        float32x4_t atan_val = vbslq_f32(is_negative, vnegq_f32(atan_full), atan_full);

        vst1q_f32(result + i, atan_val);
    }
}

// Atan2 float32: result[i] = atan2(y[i], x[i])
// Two-argument arctangent with quadrant handling
// Uses two-level range reduction for better accuracy
void atan2_f32_neon(float *y, float *x, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    float32x4_t v_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x40490FDB));
    float32x4_t v_neg_pi = vnegq_f32(v_pi);
    float32x4_t v_half_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3FC90FDB));
    float32x4_t v_neg_half_pi = vnegq_f32(v_half_pi);
    float32x4_t v_quarter_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3F490FDB));
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_tan_pi_8 = vdupq_n_f32(0.4142135623730950488f);

    // Polynomial coefficients for atan(x) on [0, tan(pi/8)]
    const float c1 = -0.3333333333f;
    const float c2 = 0.2f;
    const float c3 = -0.1428571429f;
    const float c4 = 0.1111111111f;
    const float c5 = -0.0909090909f;

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t yv = vld1q_f32(y + i);
        float32x4_t xv = vld1q_f32(x + i);

        // Compute ratio = y/x (handle x=0 separately)
        uint32x4_t x_is_zero = vceqq_f32(xv, v_zero);
        uint32x4_t y_is_zero = vceqq_f32(yv, v_zero);
        uint32x4_t x_is_neg = vcltq_f32(xv, v_zero);
        uint32x4_t y_is_neg = vcltq_f32(yv, v_zero);
        uint32x4_t y_is_pos = vcgtq_f32(yv, v_zero);

        // Safe division
        float32x4_t safe_x = vbslq_f32(x_is_zero, v_one, xv);
        float32x4_t ratio = vdivq_f32(yv, safe_x);

        // Get sign and absolute value of ratio
        uint32x4_t ratio_neg = vcltq_f32(ratio, v_zero);
        float32x4_t abs_ratio = vabsq_f32(ratio);

        // Range reduction level 1: if |ratio| > 1, use atan(r) = pi/2 - atan(1/r)
        uint32x4_t use_reciprocal = vcgtq_f32(abs_ratio, v_one);
        float32x4_t recip_ratio = vdivq_f32(v_one, abs_ratio);
        float32x4_t reduced = vbslq_f32(use_reciprocal, recip_ratio, abs_ratio);

        // Range reduction level 2: if reduced > tan(pi/8), use identity
        uint32x4_t use_identity = vcgtq_f32(reduced, v_tan_pi_8);
        float32x4_t r_minus_1 = vsubq_f32(reduced, v_one);
        float32x4_t r_plus_1 = vaddq_f32(reduced, v_one);
        float32x4_t transformed = vdivq_f32(r_minus_1, r_plus_1);
        reduced = vbslq_f32(use_identity, transformed, reduced);

        // Compute atan(reduced) using polynomial
        float32x4_t r2 = vmulq_f32(reduced, reduced);

        float32x4_t p = vdupq_n_f32(c5);
        p = vfmaq_f32(vdupq_n_f32(c4), p, r2);
        p = vfmaq_f32(vdupq_n_f32(c3), p, r2);
        p = vfmaq_f32(vdupq_n_f32(c2), p, r2);
        p = vfmaq_f32(vdupq_n_f32(c1), p, r2);
        p = vfmaq_f32(v_one, p, r2);
        float32x4_t atan_core = vmulq_f32(p, reduced);

        // Adjust for identity transform
        float32x4_t atan_reduced = vbslq_f32(use_identity,
            vaddq_f32(v_quarter_pi, atan_core),
            atan_core);

        // Adjust for reciprocal
        float32x4_t atan_abs = vbslq_f32(use_reciprocal,
            vsubq_f32(v_half_pi, atan_reduced),
            atan_reduced);

        // Apply ratio sign
        float32x4_t atan_val = vbslq_f32(ratio_neg, vnegq_f32(atan_abs), atan_abs);

        // Quadrant adjustment based on sign of x
        uint32x4_t need_add_pi = vandq_u32(x_is_neg, vmvnq_u32(y_is_neg));
        uint32x4_t need_sub_pi = vandq_u32(x_is_neg, y_is_neg);
        atan_val = vbslq_f32(need_add_pi, vaddq_f32(atan_val, v_pi), atan_val);
        atan_val = vbslq_f32(need_sub_pi, vaddq_f32(atan_val, v_neg_pi), atan_val);

        // Handle x = 0 cases
        uint32x4_t x_zero_y_pos = vandq_u32(x_is_zero, y_is_pos);
        uint32x4_t x_zero_y_neg = vandq_u32(x_is_zero, y_is_neg);
        uint32x4_t x_zero_y_zero = vandq_u32(x_is_zero, y_is_zero);
        atan_val = vbslq_f32(x_zero_y_pos, v_half_pi, atan_val);
        atan_val = vbslq_f32(x_zero_y_neg, v_neg_half_pi, atan_val);
        atan_val = vbslq_f32(x_zero_y_zero, v_zero, atan_val);

        vst1q_f32(result + i, atan_val);
    }
}

// Pow float32: result[i] = base[i] ^ exp[i]
// Uses pow(x, y) = exp(y * log(x))
void pow_f32_neon(float *base, float *exponent, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants using bit patterns
    // ln2 = 0.6931472f, bits: 0x3F317218
    // inv_ln2 = 1.442695f, bits: 0x3FB8AA3B
    float32x4_t v_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3F317218));
    float32x4_t v_inv_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3FB8AA3B));
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_min_clamp = vdupq_n_f32(-88.0f);
    float32x4_t v_max_clamp = vdupq_n_f32(88.0f);
    float32x4_t v_tiny = vdupq_n_f32(1e-30f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(base + i);
        float32x4_t y = vld1q_f32(exponent + i);

        // Handle special cases with masks
        uint32x4_t x_is_zero = vcleq_f32(vabsq_f32(x), v_tiny);
        uint32x4_t y_is_zero = vcleq_f32(vabsq_f32(y), v_tiny);

        // For log computation, use absolute value
        float32x4_t abs_x = vabsq_f32(x);
        abs_x = vmaxq_f32(abs_x, v_tiny);

        // Compute log(|x|) inline
        int32x4_t xi = vreinterpretq_s32_f32(abs_x);
        int32x4_t exp_bits = vshrq_n_s32(xi, 23);
        int32x4_t k = vsubq_s32(vandq_s32(exp_bits, vdupq_n_s32(0xFF)), vdupq_n_s32(127));
        int32x4_t mantissa_bits = vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000));
        float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

        // f = m - 1, compute log(1 + f)
        float32x4_t f = vsubq_f32(m, v_one);
        float32x4_t f2 = vmulq_f32(f, f);
        float32x4_t f3 = vmulq_f32(f2, f);
        float32x4_t f4 = vmulq_f32(f2, f2);
        float32x4_t f5 = vmulq_f32(f4, f);
        float32x4_t f6 = vmulq_f32(f3, f3);

        float32x4_t log_m = vmulq_f32(f, vdupq_n_f32(1.0f));
        log_m = vfmaq_f32(log_m, f2, vdupq_n_f32(-0.5f));
        log_m = vfmaq_f32(log_m, f3, vdupq_n_f32(0.3333333f));
        log_m = vfmaq_f32(log_m, f4, vdupq_n_f32(-0.25f));
        log_m = vfmaq_f32(log_m, f5, vdupq_n_f32(0.2f));
        log_m = vfmaq_f32(log_m, f6, vdupq_n_f32(-0.1666667f));

        float32x4_t kf = vcvtq_f32_s32(k);
        float32x4_t log_x = vfmaq_f32(log_m, kf, v_ln2);

        // z = y * log(|x|)
        float32x4_t z = vmulq_f32(y, log_x);
        z = vmaxq_f32(z, v_min_clamp);
        z = vminq_f32(z, v_max_clamp);

        // Compute exp(z) inline
        float32x4_t exp_k = vrndnq_f32(vmulq_f32(z, v_inv_ln2));
        float32x4_t r = vfmsq_f32(z, exp_k, v_ln2);

        float32x4_t exp_r = vdupq_n_f32(0.00138889f);
        exp_r = vfmaq_f32(vdupq_n_f32(0.00833333f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.04166667f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.16666667f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.5f), exp_r, r);
        exp_r = vfmaq_f32(v_one, exp_r, r);
        exp_r = vfmaq_f32(v_one, exp_r, r);

        int32x4_t ki = vcvtq_s32_f32(exp_k);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);

        float32x4_t pow_result = vmulq_f32(exp_r, scale);

        // Apply special case masks
        pow_result = vbslq_f32(x_is_zero, v_zero, pow_result);
        pow_result = vbslq_f32(y_is_zero, v_one, pow_result);

        vst1q_f32(result + i, pow_result);
    }
}

// Erf float32: result[i] = erf(input[i])
// Uses Abramowitz and Stegun approximation (7.1.26)
void erf_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Abramowitz and Stegun constants
    // p = 0.3275911f, bits: 0x3EA7BA27
    float32x4_t v_p = vreinterpretq_f32_s32(vdupq_n_s32(0x3EA7BA27));
    float32x4_t v_a1 = vdupq_n_f32(0.254829592f);
    float32x4_t v_a2 = vdupq_n_f32(-0.284496736f);
    float32x4_t v_a3 = vdupq_n_f32(1.421413741f);
    float32x4_t v_a4 = vdupq_n_f32(-1.453152027f);
    float32x4_t v_a5 = vdupq_n_f32(1.061405429f);

    // ln2 = 0.6931472f, bits: 0x3F317218
    // inv_ln2 = 1.442695f, bits: 0x3FB8AA3B
    float32x4_t v_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3F317218));
    float32x4_t v_inv_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3FB8AA3B));

    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_min_clamp = vdupq_n_f32(-88.0f);
    float32x4_t v_max_clamp = vdupq_n_f32(88.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Get sign and absolute value
        uint32x4_t is_negative = vcltq_f32(x, v_zero);
        float32x4_t abs_x = vabsq_f32(x);

        // t = 1 / (1 + p * |x|)
        float32x4_t t = vdivq_f32(v_one, vfmaq_f32(v_one, v_p, abs_x));

        // Compute -x^2 for exp
        float32x4_t neg_x2 = vnegq_f32(vmulq_f32(x, x));
        neg_x2 = vmaxq_f32(neg_x2, v_min_clamp);
        neg_x2 = vminq_f32(neg_x2, v_max_clamp);

        // Compute exp(-x^2) inline
        float32x4_t exp_k = vrndnq_f32(vmulq_f32(neg_x2, v_inv_ln2));
        float32x4_t r = vfmsq_f32(neg_x2, exp_k, v_ln2);

        float32x4_t exp_r = vdupq_n_f32(0.00138889f);
        exp_r = vfmaq_f32(vdupq_n_f32(0.00833333f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.04166667f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.16666667f), exp_r, r);
        exp_r = vfmaq_f32(vdupq_n_f32(0.5f), exp_r, r);
        exp_r = vfmaq_f32(v_one, exp_r, r);
        exp_r = vfmaq_f32(v_one, exp_r, r);

        int32x4_t ki = vcvtq_s32_f32(exp_k);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
        float32x4_t exp_neg_x2 = vmulq_f32(exp_r, scale);

        // Compute polynomial: t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
        float32x4_t poly = v_a5;
        poly = vfmaq_f32(v_a4, poly, t);
        poly = vfmaq_f32(v_a3, poly, t);
        poly = vfmaq_f32(v_a2, poly, t);
        poly = vfmaq_f32(v_a1, poly, t);
        poly = vmulq_f32(poly, t);

        // erf = 1 - poly * exp(-x^2)
        float32x4_t erf_abs = vfmsq_f32(v_one, poly, exp_neg_x2);

        // Apply sign
        float32x4_t erf_result = vbslq_f32(is_negative, vnegq_f32(erf_abs), erf_abs);

        vst1q_f32(result + i, erf_result);
    }
}

// Exp2 float32: result[i] = 2^input[i]
void exp2_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // ln2 = 0.6931472f, bits: 0x3F317218
    float32x4_t v_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3F317218));

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Clamp input to prevent overflow/underflow
        x = vmaxq_f32(x, vdupq_n_f32(-126.0f));
        x = vminq_f32(x, vdupq_n_f32(127.0f));

        // k = round(x)
        float32x4_t k = vrndnq_f32(x);

        // r = x - k, so r is in [-0.5, 0.5]
        float32x4_t r = vsubq_f32(x, k);

        // Compute 2^r = exp(r * ln(2)) using polynomial
        float32x4_t y = vmulq_f32(r, v_ln2);

        // exp(y) using Horner's method polynomial
        float32x4_t exp_r = vdupq_n_f32(0.00138889f);
        exp_r = vfmaq_f32(vdupq_n_f32(0.00833333f), exp_r, y);
        exp_r = vfmaq_f32(vdupq_n_f32(0.04166667f), exp_r, y);
        exp_r = vfmaq_f32(vdupq_n_f32(0.16666667f), exp_r, y);
        exp_r = vfmaq_f32(vdupq_n_f32(0.5f), exp_r, y);
        exp_r = vfmaq_f32(vdupq_n_f32(1.0f), exp_r, y);
        exp_r = vfmaq_f32(vdupq_n_f32(1.0f), exp_r, y);

        // Scale by 2^k
        int32x4_t ki = vcvtq_s32_f32(k);
        int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(exp_bits);

        vst1q_f32(result + i, vmulq_f32(exp_r, scale));
    }
}

// Log2 float32: result[i] = log2(input[i])
// Uses range reduction to improve accuracy for mantissa near 2
void log2_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // inv_ln2 = 1.442695f, bits: 0x3FB8AA3B
    float32x4_t v_inv_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3FB8AA3B));
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_sqrt2 = vdupq_n_f32(1.41421356237f); // sqrt(2)
    float32x4_t v_half = vdupq_n_f32(0.5f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Extract exponent and mantissa from IEEE float
        int32x4_t xi = vreinterpretq_s32_f32(x);
        int32x4_t exp_bits = vshrq_n_s32(xi, 23);
        int32x4_t k = vsubq_s32(vandq_s32(exp_bits, vdupq_n_s32(0xFF)), vdupq_n_s32(127));

        // Set exponent to 0 (bias 127) to get mantissa in [1, 2)
        int32x4_t mantissa_bits = vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000));
        float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

        // Range reduction: if m > sqrt(2), use m/2 and k+1 instead
        // This keeps f = m - 1 in [-0.29, 0.41] instead of [0, 1]
        uint32x4_t need_adjust = vcgtq_f32(m, v_sqrt2);
        float32x4_t m_adj = vmulq_f32(m, v_half);
        m = vbslq_f32(need_adjust, m_adj, m);
        int32x4_t k_adj = vaddq_s32(k, vdupq_n_s32(1));
        k = vbslq_s32(need_adjust, k_adj, k);

        // f = m - 1, now f is in [-0.29, 0.41]
        float32x4_t f = vsubq_f32(m, v_one);

        // Polynomial approximation for log(1+f) with better convergence
        // Using optimized coefficients for the reduced range
        float32x4_t f2 = vmulq_f32(f, f);
        float32x4_t f3 = vmulq_f32(f2, f);
        float32x4_t f4 = vmulq_f32(f2, f2);
        float32x4_t f5 = vmulq_f32(f4, f);

        // log(1+f) ≈ f - f²/2 + f³/3 - f⁴/4 + f⁵/5
        float32x4_t log_m = f;
        log_m = vfmaq_f32(log_m, f2, vdupq_n_f32(-0.5f));
        log_m = vfmaq_f32(log_m, f3, vdupq_n_f32(0.33333333f));
        log_m = vfmaq_f32(log_m, f4, vdupq_n_f32(-0.25f));
        log_m = vfmaq_f32(log_m, f5, vdupq_n_f32(0.2f));

        // log2(x) = k + log(m) / ln(2) = k + log(m) * inv_ln2
        float32x4_t kf = vcvtq_f32_s32(k);
        float32x4_t res = vfmaq_f32(kf, log_m, v_inv_ln2);

        vst1q_f32(result + i, res);
    }
}

// Log10 float32: result[i] = log10(input[i])
// Uses log10(x) = log2(x) * log10(2) = log2(x) * 0.30103
void log10_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    // log10(2) = 0.30102999566f, bits: 0x3E9A209B
    float32x4_t v_log10_2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3E9A209B));
    float32x4_t v_inv_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3FB8AA3B));
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_sqrt2 = vdupq_n_f32(1.41421356237f);
    float32x4_t v_half = vdupq_n_f32(0.5f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Extract exponent and mantissa from IEEE float
        int32x4_t xi = vreinterpretq_s32_f32(x);
        int32x4_t exp_bits = vshrq_n_s32(xi, 23);
        int32x4_t k = vsubq_s32(vandq_s32(exp_bits, vdupq_n_s32(0xFF)), vdupq_n_s32(127));

        // Set exponent to 0 to get mantissa in [1, 2)
        int32x4_t mantissa_bits = vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000));
        float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

        // Range reduction with sqrt(2)
        uint32x4_t need_adjust = vcgtq_f32(m, v_sqrt2);
        float32x4_t m_adj = vmulq_f32(m, v_half);
        m = vbslq_f32(need_adjust, m_adj, m);
        int32x4_t k_adj = vaddq_s32(k, vdupq_n_s32(1));
        k = vbslq_s32(need_adjust, k_adj, k);

        // f = m - 1
        float32x4_t f = vsubq_f32(m, v_one);

        // Polynomial for log(1+f)
        float32x4_t f2 = vmulq_f32(f, f);
        float32x4_t f3 = vmulq_f32(f2, f);
        float32x4_t f4 = vmulq_f32(f2, f2);
        float32x4_t f5 = vmulq_f32(f4, f);

        float32x4_t log_m = f;
        log_m = vfmaq_f32(log_m, f2, vdupq_n_f32(-0.5f));
        log_m = vfmaq_f32(log_m, f3, vdupq_n_f32(0.33333333f));
        log_m = vfmaq_f32(log_m, f4, vdupq_n_f32(-0.25f));
        log_m = vfmaq_f32(log_m, f5, vdupq_n_f32(0.2f));

        // log2(x) = k + log(m) * inv_ln2
        float32x4_t kf = vcvtq_f32_s32(k);
        float32x4_t log2_x = vfmaq_f32(kf, log_m, v_inv_ln2);

        // log10(x) = log2(x) * log10(2)
        float32x4_t res = vmulq_f32(log2_x, v_log10_2);

        vst1q_f32(result + i, res);
    }
}

// Exp10 float32: result[i] = 10^input[i]
// Uses 10^x = 2^(x * log2(10))
void exp10_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    // log2(10) = 3.321928095f, bits: 0x40549A78
    float32x4_t v_log2_10 = vreinterpretq_f32_s32(vdupq_n_s32(0x40549A78));
    float32x4_t v_ln2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3F317218));
    float32x4_t v_one = vdupq_n_f32(1.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Convert to base 2: y = x * log2(10)
        float32x4_t y = vmulq_f32(x, v_log2_10);

        // Range reduction: 2^y = 2^k * 2^f where k = round(y), f = y - k
        float32x4_t k = vrndnq_f32(y);
        float32x4_t f = vsubq_f32(y, k);

        // Convert f back to natural log domain: g = f * ln(2)
        float32x4_t g = vmulq_f32(f, v_ln2);

        // Polynomial for exp(g) where g is in [-ln(2)/2, ln(2)/2]
        float32x4_t g2 = vmulq_f32(g, g);
        float32x4_t g3 = vmulq_f32(g2, g);
        float32x4_t g4 = vmulq_f32(g2, g2);

        float32x4_t p = v_one;
        p = vfmaq_f32(p, g, v_one);
        p = vfmaq_f32(p, g2, vdupq_n_f32(0.5f));
        p = vfmaq_f32(p, g3, vdupq_n_f32(0.16666667f));
        p = vfmaq_f32(p, g4, vdupq_n_f32(0.04166667f));

        // Scale by 2^k using IEEE float manipulation
        int32x4_t ki = vcvtq_s32_f32(k);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);

        float32x4_t res = vmulq_f32(p, scale);
        vst1q_f32(result + i, res);
    }
}

// SinCos float32: sin_result[i] = sin(input[i]), cos_result[i] = cos(input[i])
// Computes both sin and cos together, sharing range reduction work
void sincos_f32_neon(float *input, float *sin_result, float *cos_result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    float32x4_t v_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x40490FDB));
    float32x4_t v_neg_pi = vnegq_f32(v_pi);
    float32x4_t v_half_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3FC90FDB));
    float32x4_t v_neg_half_pi = vnegq_f32(v_half_pi);
    float32x4_t v_inv_pi = vreinterpretq_f32_s32(vdupq_n_s32(0x3EA2F983));
    float32x4_t v_two = vdupq_n_f32(2.0f);
    float32x4_t v_neg_one = vdupq_n_f32(-1.0f);
    float32x4_t v_one = vdupq_n_f32(1.0f);

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        // Range reduction to [-pi, pi]
        float32x4_t k = vrndnq_f32(vmulq_f32(x, vmulq_f32(vdupq_n_f32(0.5f), v_inv_pi)));
        x = vfmsq_f32(x, k, vmulq_f32(v_two, v_pi));

        // === Sin computation ===
        float32x4_t sin_x = x;
        uint32x4_t need_pos_reflect = vcgtq_f32(sin_x, v_half_pi);
        uint32x4_t need_neg_reflect = vcltq_f32(sin_x, v_neg_half_pi);
        float32x4_t sin_x_pos_reflected = vsubq_f32(v_pi, sin_x);
        float32x4_t sin_x_neg_reflected = vsubq_f32(v_neg_pi, sin_x);
        sin_x = vbslq_f32(need_pos_reflect, sin_x_pos_reflected, sin_x);
        sin_x = vbslq_f32(need_neg_reflect, sin_x_neg_reflected, sin_x);

        float32x4_t sin_x2 = vmulq_f32(sin_x, sin_x);
        float32x4_t sin_p = vdupq_n_f32(-0.0001984127f);
        sin_p = vfmaq_f32(vdupq_n_f32(0.00833333f), sin_p, sin_x2);
        sin_p = vfmaq_f32(vdupq_n_f32(-0.16666667f), sin_p, sin_x2);
        sin_p = vfmaq_f32(vdupq_n_f32(1.0f), sin_p, sin_x2);
        float32x4_t sin_val = vmulq_f32(sin_p, sin_x);

        // === Cos computation ===
        float32x4_t cos_x = vabsq_f32(x);
        uint32x4_t cos_need_reflect = vcgtq_f32(cos_x, v_half_pi);
        float32x4_t cos_x_reflected = vsubq_f32(v_pi, cos_x);
        cos_x = vbslq_f32(cos_need_reflect, cos_x_reflected, cos_x);
        float32x4_t cos_sign = vbslq_f32(cos_need_reflect, v_neg_one, v_one);

        float32x4_t cos_x2 = vmulq_f32(cos_x, cos_x);
        float32x4_t cos_p = vdupq_n_f32(-0.00138889f);
        cos_p = vfmaq_f32(vdupq_n_f32(0.04166667f), cos_p, cos_x2);
        cos_p = vfmaq_f32(vdupq_n_f32(-0.5f), cos_p, cos_x2);
        cos_p = vfmaq_f32(vdupq_n_f32(1.0f), cos_p, cos_x2);
        float32x4_t cos_val = vmulq_f32(cos_p, cos_sign);

        vst1q_f32(sin_result + i, sin_val);
        vst1q_f32(cos_result + i, cos_val);
    }
}

// ===========================================================================
// Bulk Exp: exp(x) for entire array
// ===========================================================================
// Algorithm:
// 1. Range reduction: x = k*ln(2) + r, where |r| <= ln(2)/2
// 2. k = round(x / ln(2))
// 3. Polynomial: e^r ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6!
// 4. Scale: result = polynomial * 2^k

void exp_bulk_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants using bit patterns for precise values
    // invLn2 = 1.44269504f (bits: 0x3FB8AA3B)
    // ln2Hi = 0.693359375f (bits: 0x3F317200)
    // ln2Lo = -2.12194440e-4f (bits: 0xB95E8083)
    float32x4_t invLn2 = vreinterpretq_f32_s32(vdupq_n_s32(0x3FB8AA3B));
    float32x4_t ln2Hi = vreinterpretq_f32_s32(vdupq_n_s32(0x3F317200));
    float32x4_t ln2Lo = vreinterpretq_f32_s32(vdupq_n_s32(0xB95E8083));

    // Overflow threshold = 88.72283905f (bits: 0x42B17218)
    // Underflow threshold = -87.33654475f (bits: 0xC2AEAC50)
    float32x4_t overflow = vreinterpretq_f32_s32(vdupq_n_s32(0x42B17218));
    float32x4_t underflow = vreinterpretq_f32_s32(vdupq_n_s32(0xC2AEAC50));

    // Polynomial coefficients (Taylor series)
    // c1=1.0, c2=0.5, c3=1/6, c4=1/24, c5=1/120, c6=1/720
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);

    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);

    // Exponent bias and shift for Pow2
    int32x4_t bias = vdupq_n_s32(127);

    // Process 16 floats at a time (4 vectors)
    for (; i + 15 < n; i += 16) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t x1 = vld1q_f32(input + i + 4);
        float32x4_t x2 = vld1q_f32(input + i + 8);
        float32x4_t x3 = vld1q_f32(input + i + 12);

        // Check overflow/underflow
        uint32x4_t over0 = vcgtq_f32(x0, overflow);
        uint32x4_t over1 = vcgtq_f32(x1, overflow);
        uint32x4_t over2 = vcgtq_f32(x2, overflow);
        uint32x4_t over3 = vcgtq_f32(x3, overflow);

        uint32x4_t under0 = vcltq_f32(x0, underflow);
        uint32x4_t under1 = vcltq_f32(x1, underflow);
        uint32x4_t under2 = vcltq_f32(x2, underflow);
        uint32x4_t under3 = vcltq_f32(x3, underflow);

        // Range reduction: k = round(x * invLn2)
        float32x4_t kf0 = vrndnq_f32(vmulq_f32(x0, invLn2));
        float32x4_t kf1 = vrndnq_f32(vmulq_f32(x1, invLn2));
        float32x4_t kf2 = vrndnq_f32(vmulq_f32(x2, invLn2));
        float32x4_t kf3 = vrndnq_f32(vmulq_f32(x3, invLn2));

        // r = x - k*ln2Hi - k*ln2Lo
        float32x4_t r0 = vfmsq_f32(x0, kf0, ln2Hi);
        float32x4_t r1 = vfmsq_f32(x1, kf1, ln2Hi);
        float32x4_t r2 = vfmsq_f32(x2, kf2, ln2Hi);
        float32x4_t r3 = vfmsq_f32(x3, kf3, ln2Hi);

        r0 = vfmsq_f32(r0, kf0, ln2Lo);
        r1 = vfmsq_f32(r1, kf1, ln2Lo);
        r2 = vfmsq_f32(r2, kf2, ln2Lo);
        r3 = vfmsq_f32(r3, kf3, ln2Lo);

        // Horner's method: p = c6*r + c5
        float32x4_t p0 = vfmaq_f32(c5, c6, r0);
        float32x4_t p1 = vfmaq_f32(c5, c6, r1);
        float32x4_t p2 = vfmaq_f32(c5, c6, r2);
        float32x4_t p3 = vfmaq_f32(c5, c6, r3);

        // p = p*r + c4
        p0 = vfmaq_f32(c4, p0, r0);
        p1 = vfmaq_f32(c4, p1, r1);
        p2 = vfmaq_f32(c4, p2, r2);
        p3 = vfmaq_f32(c4, p3, r3);

        // p = p*r + c3
        p0 = vfmaq_f32(c3, p0, r0);
        p1 = vfmaq_f32(c3, p1, r1);
        p2 = vfmaq_f32(c3, p2, r2);
        p3 = vfmaq_f32(c3, p3, r3);

        // p = p*r + c2
        p0 = vfmaq_f32(c2, p0, r0);
        p1 = vfmaq_f32(c2, p1, r1);
        p2 = vfmaq_f32(c2, p2, r2);
        p3 = vfmaq_f32(c2, p3, r3);

        // p = p*r + c1
        p0 = vfmaq_f32(c1, p0, r0);
        p1 = vfmaq_f32(c1, p1, r1);
        p2 = vfmaq_f32(c1, p2, r2);
        p3 = vfmaq_f32(c1, p3, r3);

        // p = p*r + 1 (final step to complete the Taylor series)
        float32x4_t one = vdupq_n_f32(1.0f);
        p0 = vfmaq_f32(one, p0, r0);
        p1 = vfmaq_f32(one, p1, r1);
        p2 = vfmaq_f32(one, p2, r2);
        p3 = vfmaq_f32(one, p3, r3);

        // Compute 2^k: convert k to int, add bias, shift to exponent position
        int32x4_t ki0 = vcvtnq_s32_f32(kf0);
        int32x4_t ki1 = vcvtnq_s32_f32(kf1);
        int32x4_t ki2 = vcvtnq_s32_f32(kf2);
        int32x4_t ki3 = vcvtnq_s32_f32(kf3);

        int32x4_t scale_bits0 = vshlq_n_s32(vaddq_s32(ki0, bias), 23);
        int32x4_t scale_bits1 = vshlq_n_s32(vaddq_s32(ki1, bias), 23);
        int32x4_t scale_bits2 = vshlq_n_s32(vaddq_s32(ki2, bias), 23);
        int32x4_t scale_bits3 = vshlq_n_s32(vaddq_s32(ki3, bias), 23);

        float32x4_t scale0 = vreinterpretq_f32_s32(scale_bits0);
        float32x4_t scale1 = vreinterpretq_f32_s32(scale_bits1);
        float32x4_t scale2 = vreinterpretq_f32_s32(scale_bits2);
        float32x4_t scale3 = vreinterpretq_f32_s32(scale_bits3);

        // result = p * 2^k
        float32x4_t res0 = vmulq_f32(p0, scale0);
        float32x4_t res1 = vmulq_f32(p1, scale1);
        float32x4_t res2 = vmulq_f32(p2, scale2);
        float32x4_t res3 = vmulq_f32(p3, scale3);

        // Handle overflow -> inf, underflow -> 0
        res0 = vbslq_f32(over0, inf, res0);
        res1 = vbslq_f32(over1, inf, res1);
        res2 = vbslq_f32(over2, inf, res2);
        res3 = vbslq_f32(over3, inf, res3);

        res0 = vbslq_f32(under0, zero, res0);
        res1 = vbslq_f32(under1, zero, res1);
        res2 = vbslq_f32(under2, zero, res2);
        res3 = vbslq_f32(under3, zero, res3);

        vst1q_f32(result + i, res0);
        vst1q_f32(result + i + 4, res1);
        vst1q_f32(result + i + 8, res2);
        vst1q_f32(result + i + 12, res3);
    }

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);

        uint32x4_t over = vcgtq_f32(x, overflow);
        uint32x4_t under = vcltq_f32(x, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(x, invLn2));
        float32x4_t r = vfmsq_f32(x, kf, ln2Hi);
        r = vfmsq_f32(r, kf, ln2Lo);

        float32x4_t p = vfmaq_f32(c5, c6, r);
        p = vfmaq_f32(c4, p, r);
        p = vfmaq_f32(c3, p, r);
        p = vfmaq_f32(c2, p, r);
        p = vfmaq_f32(c1, p, r);
        float32x4_t one4 = vdupq_n_f32(1.0f);
        p = vfmaq_f32(one4, p, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);

        float32x4_t res = vmulq_f32(p, scale);
        res = vbslq_f32(over, inf, res);
        res = vbslq_f32(under, zero, res);

        vst1q_f32(result + i, res);
    }

    // Scalar remainder using NEON for single elements
    for (; i < n; i++) {
        float xv = input[i];
        float32x4_t x = vdupq_n_f32(xv);

        uint32x4_t over = vcgtq_f32(x, overflow);
        uint32x4_t under = vcltq_f32(x, underflow);

        float32x4_t kf = vrndnq_f32(vmulq_f32(x, invLn2));
        float32x4_t r = vfmsq_f32(x, kf, ln2Hi);
        r = vfmsq_f32(r, kf, ln2Lo);

        float32x4_t p = vfmaq_f32(c5, c6, r);
        p = vfmaq_f32(c4, p, r);
        p = vfmaq_f32(c3, p, r);
        p = vfmaq_f32(c2, p, r);
        p = vfmaq_f32(c1, p, r);
        float32x4_t one_s = vdupq_n_f32(1.0f);
        p = vfmaq_f32(one_s, p, r);

        int32x4_t ki = vcvtnq_s32_f32(kf);
        int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
        float32x4_t scale = vreinterpretq_f32_s32(scale_bits);

        float32x4_t res = vmulq_f32(p, scale);
        res = vbslq_f32(over, inf, res);
        res = vbslq_f32(under, zero, res);

        result[i] = vgetq_lane_f32(res, 0);
    }
}
