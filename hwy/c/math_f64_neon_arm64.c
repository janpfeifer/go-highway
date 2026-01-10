// Float64 transcendental math functions for ARM64 NEON
// NOTE: All F64 functions process 2 elements at a time (SIMD only, no scalar remainder)
// Callers must ensure length is a multiple of 2.

#include <arm_neon.h>

// Exp2 float64: result[i] = 2^input[i]
// Uses range reduction: 2^x = 2^k * 2^r where k = round(x) and r = x - k
void exp2_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // ln2 bits: 0x3FE62E42FEFA39EF
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp input to prevent overflow/underflow
        x = vmaxq_f64(x, vdupq_n_f64(-1022.0));
        x = vminq_f64(x, vdupq_n_f64(1023.0));

        // k = round(x)
        float64x2_t k = vrndnq_f64(x);

        // r = x - k, so r is in [-0.5, 0.5]
        float64x2_t r = vsubq_f64(x, k);

        // Compute 2^r = exp(r * ln(2)) using polynomial
        float64x2_t y = vmulq_f64(r, v_ln2);

        // exp(y) using Horner's method polynomial (more terms for double precision)
        float64x2_t exp_r = vdupq_n_f64(2.7557319223985893e-6);   // 1/9!
        exp_r = vfmaq_f64(vdupq_n_f64(2.48015873015873e-5), exp_r, y);   // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.984126984126984e-4), exp_r, y);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.388888888888889e-3), exp_r, y);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.333333333333333e-3), exp_r, y);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.166666666666667e-2), exp_r, y);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(0.16666666666666666), exp_r, y);   // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, y);                    // 1/2!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, y);                    // 1/1!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, y);                    // 1/0!

        // Scale by 2^k using IEEE double bit manipulation
        // Convert k to int64, add to exponent bias (1023), shift to exponent position
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);

        vst1q_f64(result + i, vmulq_f64(exp_r, scale));
    }
}

// Log2 float64: result[i] = log2(input[i])
// Uses range reduction: log2(x) = k + log2(m) where x = m * 2^k, 1 <= m < 2
void log2_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // inv_ln2 bits: 0x3FF71547652B82FE
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Extract exponent and mantissa from IEEE double
        int64x2_t xi = vreinterpretq_s64_f64(x);
        int64x2_t exp_bits = vshrq_n_s64(xi, 52);
        int64x2_t k = vsubq_s64(vandq_s64(exp_bits, vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));

        // Set exponent to 0 (bias 1023) to get mantissa in [1, 2)
        int64x2_t mantissa_bits = vorrq_s64(
            vandq_s64(xi, vdupq_n_s64(0x000FFFFFFFFFFFFFLL)),
            vdupq_n_s64(0x3FF0000000000000LL)
        );
        float64x2_t m = vreinterpretq_f64_s64(mantissa_bits);

        // f = m - 1, so we compute log(1 + f)
        float64x2_t f = vsubq_f64(m, v_one);

        // Polynomial approximation for log(1+f) with more terms for double precision
        float64x2_t f2 = vmulq_f64(f, f);
        float64x2_t f3 = vmulq_f64(f2, f);
        float64x2_t f4 = vmulq_f64(f2, f2);
        float64x2_t f5 = vmulq_f64(f4, f);
        float64x2_t f6 = vmulq_f64(f3, f3);
        float64x2_t f7 = vmulq_f64(f6, f);
        float64x2_t f8 = vmulq_f64(f4, f4);

        float64x2_t log_m = vmulq_f64(f, vdupq_n_f64(1.0));
        log_m = vfmaq_f64(log_m, f2, vdupq_n_f64(-0.5));
        log_m = vfmaq_f64(log_m, f3, vdupq_n_f64(0.3333333333333333));
        log_m = vfmaq_f64(log_m, f4, vdupq_n_f64(-0.25));
        log_m = vfmaq_f64(log_m, f5, vdupq_n_f64(0.2));
        log_m = vfmaq_f64(log_m, f6, vdupq_n_f64(-0.16666666666666666));
        log_m = vfmaq_f64(log_m, f7, vdupq_n_f64(0.14285714285714285));
        log_m = vfmaq_f64(log_m, f8, vdupq_n_f64(-0.125));

        // log2(x) = k + log(m) / ln(2) = k + log(m) * inv_ln2
        float64x2_t kf = vcvtq_f64_s64(k);
        float64x2_t res = vfmaq_f64(kf, log_m, v_inv_ln2);

        vst1q_f64(result + i, res);
    }
}

// ============================================================================
// Float64 Transcendental Operations (2 lanes per 128-bit vector)
// ============================================================================

// Exp float64: result[i] = exp(input[i])
// Uses range reduction: exp(x) = 2^k * exp(r), where k = round(x/ln(2)), r = x - k*ln(2)
void exp_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants for exp approximation (using bit patterns)
    // ln2 = 0.6931471805599453, bits: 0x3FE62E42FEFA39EF
    // inv_ln2 = 1.4426950408889634, bits: 0x3FF71547652B82FE
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp input to prevent overflow/underflow
        x = vmaxq_f64(x, vdupq_n_f64(-709.0));
        x = vminq_f64(x, vdupq_n_f64(709.0));

        // k = round(x / ln(2))
        float64x2_t k = vrndnq_f64(vmulq_f64(x, v_inv_ln2));

        // r = x - k * ln(2)
        float64x2_t r = vfmsq_f64(x, k, v_ln2);

        // exp(r) using polynomial (Horner's method) - more terms for double precision
        // exp(r) ≈ 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6! + r^7/7! + r^8/8!
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);                         // 1/2!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/1!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/0!

        // Scale by 2^k
        // Convert k to int, add to exponent bias (1023), shift to exponent position
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);

        vst1q_f64(result + i, vmulq_f64(exp_r, scale));
    }
}

// Log float64: result[i] = log(input[i]) (natural logarithm)
// Uses range reduction: log(x) = k*ln(2) + log(m) where x = m * 2^k, 1 <= m < 2
void log_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // ln2 bits: 0x3FE62E42FEFA39EF
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Extract exponent and mantissa
        int64x2_t xi = vreinterpretq_s64_f64(x);
        int64x2_t exp_bits = vshrq_n_s64(xi, 52);
        int64x2_t k = vsubq_s64(vandq_s64(exp_bits, vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));

        // Set exponent to 0 (bias 1023) to get mantissa in [1, 2)
        int64x2_t mantissa_bits = vorrq_s64(
            vandq_s64(xi, vdupq_n_s64(0x000FFFFFFFFFFFFFLL)),
            vdupq_n_s64(0x3FF0000000000000LL)
        );
        float64x2_t m = vreinterpretq_f64_s64(mantissa_bits);

        // f = m - 1, so we compute log(1 + f)
        float64x2_t f = vsubq_f64(m, v_one);

        // Polynomial approximation for log(1+f)
        float64x2_t f2 = vmulq_f64(f, f);
        float64x2_t f3 = vmulq_f64(f2, f);
        float64x2_t f4 = vmulq_f64(f2, f2);
        float64x2_t f5 = vmulq_f64(f4, f);
        float64x2_t f6 = vmulq_f64(f3, f3);
        float64x2_t f7 = vmulq_f64(f6, f);
        float64x2_t f8 = vmulq_f64(f4, f4);

        float64x2_t log_m = vmulq_f64(f, vdupq_n_f64(1.0));
        log_m = vfmaq_f64(log_m, f2, vdupq_n_f64(-0.5));
        log_m = vfmaq_f64(log_m, f3, vdupq_n_f64(0.3333333333333333));
        log_m = vfmaq_f64(log_m, f4, vdupq_n_f64(-0.25));
        log_m = vfmaq_f64(log_m, f5, vdupq_n_f64(0.2));
        log_m = vfmaq_f64(log_m, f6, vdupq_n_f64(-0.16666666666666666));
        log_m = vfmaq_f64(log_m, f7, vdupq_n_f64(0.14285714285714285));
        log_m = vfmaq_f64(log_m, f8, vdupq_n_f64(-0.125));

        // log(x) = k * ln(2) + log(m)
        float64x2_t kf = vcvtq_f64_s64(k);
        float64x2_t res = vfmaq_f64(log_m, kf, v_ln2);

        vst1q_f64(result + i, res);
    }
}

// Sin float64: result[i] = sin(input[i])
// Uses range reduction to [-pi, pi], reflection to [-pi/2, pi/2], and polynomial
void sin_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants (using bit patterns for non-immediate values)
    // pi = 3.14159265358979323846, bits: 0x400921FB54442D18
    // inv_pi = 0.3183098861837907, bits: 0x3FD45F306DC9C883
    // half_pi = 1.5707963267948966, bits: 0x3FF921FB54442D18
    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_neg_pi = vnegq_f64(v_pi);
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_neg_half_pi = vnegq_f64(v_half_pi);
    float64x2_t v_inv_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD45F306DC9C883LL));
    float64x2_t v_two = vdupq_n_f64(2.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float64x2_t k = vrndnq_f64(vmulq_f64(x, vmulq_f64(vdupq_n_f64(0.5), v_inv_pi)));
        x = vfmsq_f64(x, k, vmulq_f64(v_two, v_pi));

        // Reflection to [-pi/2, pi/2]:
        // if x > pi/2:  sin(x) = sin(pi - x)
        // if x < -pi/2: sin(x) = sin(-pi - x)
        uint64x2_t need_pos_reflect = vcgtq_f64(x, v_half_pi);
        uint64x2_t need_neg_reflect = vcltq_f64(x, v_neg_half_pi);
        float64x2_t x_pos_reflected = vsubq_f64(v_pi, x);
        float64x2_t x_neg_reflected = vsubq_f64(v_neg_pi, x);
        x = vbslq_f64(need_pos_reflect, x_pos_reflected, x);
        x = vbslq_f64(need_neg_reflect, x_neg_reflected, x);

        // sin(x) using polynomial
        float64x2_t x2 = vmulq_f64(x, x);

        // Coefficients: s11 = -2.5052108385441718e-8, s9 = 2.7557319223985893e-6, etc.
        float64x2_t p = vdupq_n_f64(-2.5052108385441718e-8);   // s11
        p = vfmaq_f64(vdupq_n_f64(2.7557319223985893e-6), p, x2);   // s9
        p = vfmaq_f64(vdupq_n_f64(-0.0001984126984126984), p, x2);  // s7
        p = vfmaq_f64(vdupq_n_f64(0.008333333333333333), p, x2);    // s5
        p = vfmaq_f64(vdupq_n_f64(-0.16666666666666666), p, x2);    // s3
        p = vfmaq_f64(vdupq_n_f64(1.0), p, x2);                     // s1
        p = vmulq_f64(p, x);

        vst1q_f64(result + i, p);
    }
}

// Cos float64: result[i] = cos(input[i])
// Uses range reduction to [-pi, pi], reflection to [0, pi/2], and polynomial
void cos_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_inv_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD45F306DC9C883LL));
    float64x2_t v_two = vdupq_n_f64(2.0);
    float64x2_t v_neg_one = vdupq_n_f64(-1.0);
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float64x2_t k = vrndnq_f64(vmulq_f64(x, vmulq_f64(vdupq_n_f64(0.5), v_inv_pi)));
        x = vfmsq_f64(x, k, vmulq_f64(v_two, v_pi));

        // cos(x) = cos(|x|) since cosine is even
        x = vabsq_f64(x);

        // Reflection: if |x| > pi/2, use cos(|x|) = -cos(pi - |x|)
        uint64x2_t need_reflect = vcgtq_f64(x, v_half_pi);
        float64x2_t x_reflected = vsubq_f64(v_pi, x);
        x = vbslq_f64(need_reflect, x_reflected, x);
        float64x2_t sign = vbslq_f64(need_reflect, v_neg_one, v_one);

        // cos(x) using polynomial: 1 + x^2*(c2 + x^2*(c4 + x^2*(c6 + x^2*(c8 + x^2*c10))))
        float64x2_t x2 = vmulq_f64(x, x);

        float64x2_t p = vdupq_n_f64(-2.7557319223985888e-7);   // c10
        p = vfmaq_f64(vdupq_n_f64(2.48015873015873016e-5), p, x2);   // c8
        p = vfmaq_f64(vdupq_n_f64(-0.001388888888888889), p, x2);    // c6
        p = vfmaq_f64(vdupq_n_f64(0.041666666666666664), p, x2);     // c4
        p = vfmaq_f64(vdupq_n_f64(-0.5), p, x2);                     // c2
        p = vfmaq_f64(vdupq_n_f64(1.0), p, x2);                      // c0

        // Apply sign from reflection
        p = vmulq_f64(p, sign);

        vst1q_f64(result + i, p);
    }
}

// Tanh float64: result[i] = tanh(input[i])
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
void tanh_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // For |x| > 19, tanh(x) ≈ sign(x)
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_limit = vdupq_n_f64(19.0);
    float64x2_t v_neg_limit = vdupq_n_f64(-19.0);

    // Constants (using bit patterns)
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp to prevent overflow
        float64x2_t x_clamped = vmaxq_f64(vminq_f64(x, v_limit), v_neg_limit);

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float64x2_t two_x = vmulq_f64(x_clamped, vdupq_n_f64(2.0));

        // Range reduction for exp
        float64x2_t k = vrndnq_f64(vmulq_f64(two_x, v_inv_ln2));
        float64x2_t r = vfmsq_f64(two_x, k, v_ln2);

        // exp(r) polynomial - higher precision for double
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        // Scale
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp2x = vmulq_f64(exp_r, scale);

        // tanh = (exp2x - 1) / (exp2x + 1)
        float64x2_t num = vsubq_f64(exp2x, v_one);
        float64x2_t den = vaddq_f64(exp2x, v_one);
        float64x2_t res = vdivq_f64(num, den);

        vst1q_f64(result + i, res);
    }
}

// Sigmoid float64: result[i] = 1 / (1 + exp(-input[i]))
void sigmoid_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    float64x2_t v_one = vdupq_n_f64(1.0);

    // Constants (using bit patterns)
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp to prevent overflow
        x = vmaxq_f64(x, vdupq_n_f64(-709.0));
        x = vminq_f64(x, vdupq_n_f64(709.0));

        // exp(-x)
        float64x2_t neg_x = vnegq_f64(x);

        // Range reduction for exp
        float64x2_t k = vrndnq_f64(vmulq_f64(neg_x, v_inv_ln2));
        float64x2_t r = vfmsq_f64(neg_x, k, v_ln2);

        // exp(r) polynomial - higher precision for double
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        // Scale
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_x = vmulq_f64(exp_r, scale);

        // sigmoid = 1 / (1 + exp(-x))
        float64x2_t res = vdivq_f64(v_one, vaddq_f64(v_one, exp_neg_x));

        vst1q_f64(result + i, res);
    }
}

// Tan float64: result[i] = tan(input[i])
// Uses sin(x)/cos(x)
void tan_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_neg_pi = vnegq_f64(v_pi);
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_neg_half_pi = vnegq_f64(v_half_pi);
    float64x2_t v_inv_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD45F306DC9C883LL));
    float64x2_t v_two = vdupq_n_f64(2.0);
    float64x2_t v_neg_one = vdupq_n_f64(-1.0);
    float64x2_t v_one = vdupq_n_f64(1.0);

    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Range reduction to [-pi, pi]
        float64x2_t k = vrndnq_f64(vmulq_f64(x, vmulq_f64(vdupq_n_f64(0.5), v_inv_pi)));
        x = vfmsq_f64(x, k, vmulq_f64(v_two, v_pi));

        // Sin computation with reflection
        float64x2_t sin_x = x;
        uint64x2_t need_pos_reflect = vcgtq_f64(sin_x, v_half_pi);
        uint64x2_t need_neg_reflect = vcltq_f64(sin_x, v_neg_half_pi);
        sin_x = vbslq_f64(need_pos_reflect, vsubq_f64(v_pi, sin_x), sin_x);
        sin_x = vbslq_f64(need_neg_reflect, vsubq_f64(v_neg_pi, sin_x), sin_x);

        float64x2_t sin_x2 = vmulq_f64(sin_x, sin_x);
        float64x2_t sin_p = vdupq_n_f64(-2.5052108385441718e-8);
        sin_p = vfmaq_f64(vdupq_n_f64(2.7557319223985893e-6), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(-0.0001984126984126984), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(0.008333333333333333), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(-0.16666666666666666), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(1.0), sin_p, sin_x2);
        float64x2_t sin_val = vmulq_f64(sin_p, sin_x);

        // Cos computation with reflection
        float64x2_t cos_x = vabsq_f64(x);
        uint64x2_t cos_need_reflect = vcgtq_f64(cos_x, v_half_pi);
        cos_x = vbslq_f64(cos_need_reflect, vsubq_f64(v_pi, cos_x), cos_x);
        float64x2_t cos_sign = vbslq_f64(cos_need_reflect, v_neg_one, v_one);

        float64x2_t cos_x2 = vmulq_f64(cos_x, cos_x);
        float64x2_t cos_p = vdupq_n_f64(-2.7557319223985888e-7);
        cos_p = vfmaq_f64(vdupq_n_f64(2.48015873015873016e-5), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(-0.001388888888888889), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(0.041666666666666664), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(-0.5), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(1.0), cos_p, cos_x2);
        float64x2_t cos_val = vmulq_f64(cos_p, cos_sign);

        vst1q_f64(result + i, vdivq_f64(sin_val, cos_val));
    }
}

// Atan float64: result[i] = atan(input[i])
// Uses two-level range reduction for accuracy
void atan_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // half_pi bits: 0x3FF921FB54442D18, quarter_pi bits: 0x3FE921FB54442D18
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_quarter_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE921FB54442D18LL));
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_zero = vdupq_n_f64(0.0);
    float64x2_t v_tan_pi_8 = vdupq_n_f64(0.4142135623730950488); // tan(pi/8) = sqrt(2) - 1

    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        uint64x2_t is_negative = vcltq_f64(x, v_zero);
        float64x2_t abs_x = vabsq_f64(x);

        // Level 1: if |x| > 1, use atan(x) = pi/2 - atan(1/x)
        uint64x2_t use_reciprocal = vcgtq_f64(abs_x, v_one);
        float64x2_t recip_x = vdivq_f64(v_one, abs_x);
        float64x2_t reduced_x = vbslq_f64(use_reciprocal, recip_x, abs_x);

        // Level 2: if reduced_x > tan(pi/8), use atan(x) = pi/4 + atan((x-1)/(x+1))
        uint64x2_t use_identity = vcgtq_f64(reduced_x, v_tan_pi_8);
        float64x2_t transformed_x = vdivq_f64(vsubq_f64(reduced_x, v_one), vaddq_f64(reduced_x, v_one));
        reduced_x = vbslq_f64(use_identity, transformed_x, reduced_x);

        // Polynomial for atan - more terms for double precision
        float64x2_t x2 = vmulq_f64(reduced_x, reduced_x);
        float64x2_t p = vdupq_n_f64(-0.0909090909090909);
        p = vfmaq_f64(vdupq_n_f64(0.1111111111111111), p, x2);
        p = vfmaq_f64(vdupq_n_f64(-0.1428571428571428), p, x2);
        p = vfmaq_f64(vdupq_n_f64(0.2), p, x2);
        p = vfmaq_f64(vdupq_n_f64(-0.3333333333333333), p, x2);
        p = vfmaq_f64(v_one, p, x2);
        float64x2_t atan_core = vmulq_f64(p, reduced_x);

        // Adjust for identity
        float64x2_t atan_reduced = vbslq_f64(use_identity, vaddq_f64(v_quarter_pi, atan_core), atan_core);

        // Adjust for reciprocal
        float64x2_t atan_full = vbslq_f64(use_reciprocal, vsubq_f64(v_half_pi, atan_reduced), atan_reduced);

        // Apply original sign
        float64x2_t atan_val = vbslq_f64(is_negative, vnegq_f64(atan_full), atan_full);

        vst1q_f64(result + i, atan_val);
    }
}

// Atan2 float64: result[i] = atan2(y[i], x[i])
void atan2_f64_neon(double *y_arr, double *x_arr, double *result, long *len) {
    long n = *len;
    long i = 0;

    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_quarter_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE921FB54442D18LL));
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_zero = vdupq_n_f64(0.0);
    float64x2_t v_tan_pi_8 = vdupq_n_f64(0.4142135623730950488);

    for (; i + 1 < n; i += 2) {
        float64x2_t y = vld1q_f64(y_arr + i);
        float64x2_t x = vld1q_f64(x_arr + i);

        // Compute atan(y/x)
        float64x2_t ratio = vdivq_f64(y, x);
        uint64x2_t is_negative = vcltq_f64(ratio, v_zero);
        float64x2_t abs_ratio = vabsq_f64(ratio);

        // Two-level range reduction
        uint64x2_t use_reciprocal = vcgtq_f64(abs_ratio, v_one);
        float64x2_t reduced = vbslq_f64(use_reciprocal, vdivq_f64(v_one, abs_ratio), abs_ratio);

        uint64x2_t use_identity = vcgtq_f64(reduced, v_tan_pi_8);
        float64x2_t transformed = vdivq_f64(vsubq_f64(reduced, v_one), vaddq_f64(reduced, v_one));
        reduced = vbslq_f64(use_identity, transformed, reduced);

        // Polynomial
        float64x2_t r2 = vmulq_f64(reduced, reduced);
        float64x2_t p = vdupq_n_f64(-0.0909090909090909);
        p = vfmaq_f64(vdupq_n_f64(0.1111111111111111), p, r2);
        p = vfmaq_f64(vdupq_n_f64(-0.1428571428571428), p, r2);
        p = vfmaq_f64(vdupq_n_f64(0.2), p, r2);
        p = vfmaq_f64(vdupq_n_f64(-0.3333333333333333), p, r2);
        p = vfmaq_f64(v_one, p, r2);
        float64x2_t atan_core = vmulq_f64(p, reduced);

        float64x2_t atan_reduced = vbslq_f64(use_identity, vaddq_f64(v_quarter_pi, atan_core), atan_core);
        float64x2_t atan_full = vbslq_f64(use_reciprocal, vsubq_f64(v_half_pi, atan_reduced), atan_reduced);
        float64x2_t atan_val = vbslq_f64(is_negative, vnegq_f64(atan_full), atan_full);

        // Quadrant adjustment
        uint64x2_t x_neg = vcltq_f64(x, v_zero);
        uint64x2_t y_neg = vcltq_f64(y, v_zero);
        uint64x2_t y_pos = vcgeq_f64(y, v_zero);

        float64x2_t adj_pos = vaddq_f64(atan_val, v_pi);
        float64x2_t adj_neg = vsubq_f64(atan_val, v_pi);

        uint64x2_t need_adj_pos = vandq_u64(x_neg, y_pos);
        uint64x2_t need_adj_neg = vandq_u64(x_neg, y_neg);

        atan_val = vbslq_f64(need_adj_pos, adj_pos, atan_val);
        atan_val = vbslq_f64(need_adj_neg, adj_neg, atan_val);

        vst1q_f64(result + i, atan_val);
    }
}

// Pow float64: result[i] = base[i] ^ exp[i]
// Uses pow(x, y) = exp(y * log(x)) with sqrt(2) range reduction for better accuracy
void pow_f64_neon(double *base, double *exp_arr, double *result, long *len) {
    long n = *len;
    long i = 0;

    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_sqrt2 = vdupq_n_f64(1.4142135623730951);
    float64x2_t v_half = vdupq_n_f64(0.5);

    for (; i + 1 < n; i += 2) {
        float64x2_t b = vld1q_f64(base + i);
        float64x2_t e = vld1q_f64(exp_arr + i);

        // log(b): extract exponent and mantissa
        int64x2_t bi = vreinterpretq_s64_f64(b);
        int64x2_t exp_bits = vshrq_n_s64(bi, 52);
        int64x2_t k_log = vsubq_s64(vandq_s64(exp_bits, vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));
        int64x2_t mantissa_bits = vorrq_s64(
            vandq_s64(bi, vdupq_n_s64(0x000FFFFFFFFFFFFFLL)),
            vdupq_n_s64(0x3FF0000000000000LL));
        float64x2_t m = vreinterpretq_f64_s64(mantissa_bits);

        // sqrt(2) range reduction for better log accuracy
        uint64x2_t need_adjust = vcgtq_f64(m, v_sqrt2);
        float64x2_t m_adj = vmulq_f64(m, v_half);
        m = vbslq_f64(need_adjust, m_adj, m);
        int64x2_t k_adj = vaddq_s64(k_log, vdupq_n_s64(1));
        k_log = vbslq_s64(need_adjust, k_adj, k_log);

        float64x2_t f = vsubq_f64(m, v_one);

        // log(1+f) polynomial with more terms for higher accuracy
        float64x2_t f2 = vmulq_f64(f, f);
        float64x2_t f3 = vmulq_f64(f2, f);
        float64x2_t f4 = vmulq_f64(f2, f2);
        float64x2_t f5 = vmulq_f64(f4, f);
        float64x2_t f6 = vmulq_f64(f3, f3);
        float64x2_t f7 = vmulq_f64(f6, f);
        float64x2_t log_m = f;
        log_m = vfmaq_f64(log_m, f2, vdupq_n_f64(-0.5));
        log_m = vfmaq_f64(log_m, f3, vdupq_n_f64(0.3333333333333333));
        log_m = vfmaq_f64(log_m, f4, vdupq_n_f64(-0.25));
        log_m = vfmaq_f64(log_m, f5, vdupq_n_f64(0.2));
        log_m = vfmaq_f64(log_m, f6, vdupq_n_f64(-0.16666666666666666));
        log_m = vfmaq_f64(log_m, f7, vdupq_n_f64(0.14285714285714285));

        // log(b) = k*ln(2) + log(m)
        float64x2_t kf_log = vcvtq_f64_s64(k_log);
        float64x2_t log_b = vfmaq_f64(log_m, kf_log, v_ln2);

        // y * log(b)
        float64x2_t y_log_b = vmulq_f64(e, log_b);

        // Clamp
        y_log_b = vmaxq_f64(y_log_b, vdupq_n_f64(-709.0));
        y_log_b = vminq_f64(y_log_b, vdupq_n_f64(709.0));

        // exp(y * log(b))
        float64x2_t k_exp = vrndnq_f64(vmulq_f64(y_log_b, v_inv_ln2));
        float64x2_t r = vfmsq_f64(y_log_b, k_exp, v_ln2);

        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        int64x2_t ki_exp = vcvtq_s64_f64(k_exp);
        int64x2_t scale_bits = vshlq_n_s64(vaddq_s64(ki_exp, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(scale_bits);

        vst1q_f64(result + i, vmulq_f64(exp_r, scale));
    }
}

// Erf float64: result[i] = erf(input[i])
// Uses Abramowitz & Stegun approximation 7.1.26 with full exp polynomial
void erf_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    float64x2_t v_zero = vdupq_n_f64(0.0);
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Coefficients for erf approximation (Abramowitz & Stegun 7.1.26)
    // Max error: 1.5e-7
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;
    const double p = 0.3275911;

    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        uint64x2_t is_negative = vcltq_f64(x, v_zero);
        float64x2_t abs_x = vabsq_f64(x);

        // t = 1 / (1 + p * |x|)
        float64x2_t t = vdivq_f64(v_one, vfmaq_f64(v_one, abs_x, vdupq_n_f64(p)));

        // Polynomial: a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        float64x2_t poly = vdupq_n_f64(a5);
        poly = vfmaq_f64(vdupq_n_f64(a4), poly, t);
        poly = vfmaq_f64(vdupq_n_f64(a3), poly, t);
        poly = vfmaq_f64(vdupq_n_f64(a2), poly, t);
        poly = vfmaq_f64(vdupq_n_f64(a1), poly, t);
        poly = vmulq_f64(poly, t);

        // exp(-x^2) with full polynomial for accuracy
        float64x2_t x2 = vmulq_f64(abs_x, abs_x);
        float64x2_t neg_x2 = vnegq_f64(x2);

        // Clamp for exp
        neg_x2 = vmaxq_f64(neg_x2, vdupq_n_f64(-709.0));

        // Full exp polynomial
        float64x2_t k = vrndnq_f64(vmulq_f64(neg_x2, v_inv_ln2));
        float64x2_t r = vfmsq_f64(neg_x2, k, v_ln2);

        // exp(r) polynomial - full 8 terms for double precision
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);                         // 1/2!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/1!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/0!

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_x2 = vmulq_f64(exp_r, scale);

        // erf(|x|) = 1 - poly * exp(-x^2)
        float64x2_t erf_abs = vsubq_f64(v_one, vmulq_f64(poly, exp_neg_x2));

        // Apply sign
        float64x2_t erf_val = vbslq_f64(is_negative, vnegq_f64(erf_abs), erf_abs);

        vst1q_f64(result + i, erf_val);
    }
}

// Log10 float64: result[i] = log10(input[i])
// Uses sqrt(2) range reduction for better accuracy
void log10_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // log10(2) = 0.30102999566398119, bits: 0x3FD34413509F79FF
    float64x2_t v_log10_2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD34413509F79FFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_sqrt2 = vdupq_n_f64(1.4142135623730951);
    float64x2_t v_half = vdupq_n_f64(0.5);

    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Extract exponent and mantissa
        int64x2_t xi = vreinterpretq_s64_f64(x);
        int64x2_t exp_bits = vshrq_n_s64(xi, 52);
        int64x2_t k = vsubq_s64(vandq_s64(exp_bits, vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));

        int64x2_t mantissa_bits = vorrq_s64(
            vandq_s64(xi, vdupq_n_s64(0x000FFFFFFFFFFFFFLL)),
            vdupq_n_s64(0x3FF0000000000000LL));
        float64x2_t m = vreinterpretq_f64_s64(mantissa_bits);

        // sqrt(2) range reduction: if m > sqrt(2), use m/2 and k+1
        uint64x2_t need_adjust = vcgtq_f64(m, v_sqrt2);
        float64x2_t m_adj = vmulq_f64(m, v_half);
        m = vbslq_f64(need_adjust, m_adj, m);
        int64x2_t k_adj = vaddq_s64(k, vdupq_n_s64(1));
        k = vbslq_s64(need_adjust, k_adj, k);

        float64x2_t f = vsubq_f64(m, v_one);

        // log(1+f) polynomial with more terms for better accuracy
        float64x2_t f2 = vmulq_f64(f, f);
        float64x2_t f3 = vmulq_f64(f2, f);
        float64x2_t f4 = vmulq_f64(f2, f2);
        float64x2_t f5 = vmulq_f64(f4, f);
        float64x2_t f6 = vmulq_f64(f3, f3);

        float64x2_t log_m = f;
        log_m = vfmaq_f64(log_m, f2, vdupq_n_f64(-0.5));
        log_m = vfmaq_f64(log_m, f3, vdupq_n_f64(0.3333333333333333));
        log_m = vfmaq_f64(log_m, f4, vdupq_n_f64(-0.25));
        log_m = vfmaq_f64(log_m, f5, vdupq_n_f64(0.2));
        log_m = vfmaq_f64(log_m, f6, vdupq_n_f64(-0.16666666666666666));

        // log2(x) = k + log(m) * inv_ln2
        float64x2_t kf = vcvtq_f64_s64(k);
        float64x2_t log2_x = vfmaq_f64(kf, log_m, v_inv_ln2);

        // log10(x) = log2(x) * log10(2)
        vst1q_f64(result + i, vmulq_f64(log2_x, v_log10_2));
    }
}

// Exp10 float64: result[i] = 10^input[i]
void exp10_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // log2(10) = 3.321928094887362, bits: 0x400A934F0979A371
    float64x2_t v_log2_10 = vreinterpretq_f64_s64(vdupq_n_s64(0x400A934F0979A371LL));
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));

    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // 10^x = 2^(x * log2(10))
        float64x2_t y = vmulq_f64(x, v_log2_10);

        // Clamp
        y = vmaxq_f64(y, vdupq_n_f64(-1022.0));
        y = vminq_f64(y, vdupq_n_f64(1023.0));

        float64x2_t k = vrndnq_f64(y);
        float64x2_t f = vsubq_f64(y, k);
        float64x2_t g = vmulq_f64(f, v_ln2);

        // exp(g) polynomial
        float64x2_t exp_g = vdupq_n_f64(2.48015873015873015873e-5);
        exp_g = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(0.5), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(1.0), exp_g, g);
        exp_g = vfmaq_f64(vdupq_n_f64(1.0), exp_g, g);

        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t scale_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(scale_bits);

        vst1q_f64(result + i, vmulq_f64(exp_g, scale));
    }
}

// SinCos float64: computes both sin and cos together
void sincos_f64_neon(double *input, double *sin_result, double *cos_result, long *len) {
    long n = *len;
    long i = 0;

    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_neg_pi = vnegq_f64(v_pi);
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_neg_half_pi = vnegq_f64(v_half_pi);
    float64x2_t v_inv_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD45F306DC9C883LL));
    float64x2_t v_two = vdupq_n_f64(2.0);
    float64x2_t v_neg_one = vdupq_n_f64(-1.0);
    float64x2_t v_one = vdupq_n_f64(1.0);

    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Range reduction to [-pi, pi]
        float64x2_t k = vrndnq_f64(vmulq_f64(x, vmulq_f64(vdupq_n_f64(0.5), v_inv_pi)));
        x = vfmsq_f64(x, k, vmulq_f64(v_two, v_pi));

        // === Sin computation ===
        float64x2_t sin_x = x;
        uint64x2_t need_pos_reflect = vcgtq_f64(sin_x, v_half_pi);
        uint64x2_t need_neg_reflect = vcltq_f64(sin_x, v_neg_half_pi);
        sin_x = vbslq_f64(need_pos_reflect, vsubq_f64(v_pi, sin_x), sin_x);
        sin_x = vbslq_f64(need_neg_reflect, vsubq_f64(v_neg_pi, sin_x), sin_x);

        float64x2_t sin_x2 = vmulq_f64(sin_x, sin_x);
        float64x2_t sin_p = vdupq_n_f64(-2.5052108385441718e-8);
        sin_p = vfmaq_f64(vdupq_n_f64(2.7557319223985893e-6), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(-0.0001984126984126984), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(0.008333333333333333), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(-0.16666666666666666), sin_p, sin_x2);
        sin_p = vfmaq_f64(vdupq_n_f64(1.0), sin_p, sin_x2);
        float64x2_t sin_val = vmulq_f64(sin_p, sin_x);

        // === Cos computation ===
        float64x2_t cos_x = vabsq_f64(x);
        uint64x2_t cos_need_reflect = vcgtq_f64(cos_x, v_half_pi);
        cos_x = vbslq_f64(cos_need_reflect, vsubq_f64(v_pi, cos_x), cos_x);
        float64x2_t cos_sign = vbslq_f64(cos_need_reflect, v_neg_one, v_one);

        float64x2_t cos_x2 = vmulq_f64(cos_x, cos_x);
        float64x2_t cos_p = vdupq_n_f64(-2.7557319223985888e-7);
        cos_p = vfmaq_f64(vdupq_n_f64(2.48015873015873016e-5), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(-0.001388888888888889), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(0.041666666666666664), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(-0.5), cos_p, cos_x2);
        cos_p = vfmaq_f64(vdupq_n_f64(1.0), cos_p, cos_x2);
        float64x2_t cos_val = vmulq_f64(cos_p, cos_sign);

        vst1q_f64(sin_result + i, sin_val);
        vst1q_f64(cos_result + i, cos_val);
    }
}
