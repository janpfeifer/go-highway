//go:build amd64 && goexperiment.simd

package contrib

import (
	"math"
	"simd/archsimd"
	"testing"
)

// Test different approaches for computing 2^k in float64

// Approach 1: Current scalar approach
func scale2kScalar(kFloat archsimd.Float64x4) archsimd.Float64x4 {
	var kBuf [4]float64
	kFloat.StoreSlice(kBuf[:])

	var scaleBuf [4]float64
	for i := 0; i < 4; i++ {
		k := int64(kBuf[i])
		scaleBuf[i] = math.Float64frombits(uint64(k+1023) << 52)
	}
	return archsimd.LoadFloat64x4Slice(scaleBuf[:])
}

// Approach 2: Unrolled scalar (avoid loop overhead)
func scale2kScalarUnrolled(kFloat archsimd.Float64x4) archsimd.Float64x4 {
	var kBuf [4]float64
	kFloat.StoreSlice(kBuf[:])

	k0, k1, k2, k3 := int64(kBuf[0]), int64(kBuf[1]), int64(kBuf[2]), int64(kBuf[3])
	scaleBuf := [4]float64{
		math.Float64frombits(uint64(k0+1023) << 52),
		math.Float64frombits(uint64(k1+1023) << 52),
		math.Float64frombits(uint64(k2+1023) << 52),
		math.Float64frombits(uint64(k3+1023) << 52),
	}
	return archsimd.LoadFloat64x4Slice(scaleBuf[:])
}

// Approach 3: Magic number trick - embed k in mantissa, then bit manipulate
// Uses the fact that adding 2^52 to a small integer k gives a float64
// where the mantissa bits contain k.
var (
	magic64       = archsimd.BroadcastFloat64x4(0x1.0p52)                       // 2^52
	magicBias64   = archsimd.BroadcastInt64x4(0x4330000000000000 + (1023 << 52)) // Adjustment for exponent
	magicAdjust64 = archsimd.BroadcastInt64x4(0x4330000000000000)               // To subtract the magic
)

func scale2kMagic(kFloat archsimd.Float64x4) archsimd.Float64x4 {
	// Add magic to embed k in mantissa bits
	kPlusMagic := kFloat.Add(magic64)

	// Reinterpret as int64 - the low 52 bits now contain k (for k >= 0)
	// For negative k, we get 2^52 + k which is positive
	bits := kPlusMagic.AsInt64x4()

	// We have: bits = 0x433 | (2^52 + k) in mantissa
	// We want: result = (k + 1023) << 52
	// Subtract the magic bias to adjust
	// Actually: bits - 0x4330000000000000 gives us k in the low 52 bits
	// Then add 1023<<52 and mask/shift appropriately

	// Simpler: manipulate the exponent field directly
	// The exponent of (k + 2^52) is 1075 = 1023 + 52
	// We want exponent = k + 1023

	// This is tricky... let's try a different manipulation
	// bits = 0x433XXXXXXXXXXXXX where X = k (for small positive k)

	// Alternative: extract k, add bias, construct new float
	kInt := bits.Sub(magicAdjust64) // Now low 52 bits = k (signed)
	expBits := kInt.Add(archsimd.BroadcastInt64x4(1023)).ShiftAllLeft(52)
	return expBits.AsFloat64x4()
}

// Approach 4: Two-step multiplication
// Compute 2^k = 2^k1 * 2^k2 where k1, k2 are half of k
var (
	half64 = archsimd.BroadcastFloat64x4(0.5)
)

func scale2kTwoStep(kFloat archsimd.Float64x4) archsimd.Float64x4 {
	// Split k into two halves
	k1 := kFloat.Mul(half64).RoundToEven() // floor(k/2) approximately
	k2 := kFloat.Sub(k1)                   // k - k1

	// Compute 2^k1 and 2^k2 using scalar (each is smaller range)
	scale1 := scale2kScalarUnrolled(k1)
	scale2 := scale2kScalarUnrolled(k2)

	return scale1.Mul(scale2)
}

func BenchmarkScale2k_Scalar(b *testing.B) {
	input := []float64{1, 2, 3, 4}
	k := archsimd.LoadFloat64x4Slice(input)
	var result archsimd.Float64x4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result = scale2kScalar(k)
	}
	_ = result
}

func BenchmarkScale2k_ScalarUnrolled(b *testing.B) {
	input := []float64{1, 2, 3, 4}
	k := archsimd.LoadFloat64x4Slice(input)
	var result archsimd.Float64x4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result = scale2kScalarUnrolled(k)
	}
	_ = result
}

func BenchmarkScale2k_Magic(b *testing.B) {
	input := []float64{1, 2, 3, 4}
	k := archsimd.LoadFloat64x4Slice(input)
	var result archsimd.Float64x4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result = scale2kMagic(k)
	}
	_ = result
}

func BenchmarkScale2k_TwoStep(b *testing.B) {
	input := []float64{1, 2, 3, 4}
	k := archsimd.LoadFloat64x4Slice(input)
	var result archsimd.Float64x4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result = scale2kTwoStep(k)
	}
	_ = result
}

// Correctness test
func TestScale2kApproaches(t *testing.T) {
	testCases := [][]float64{
		{0, 1, 2, 3},
		{-1, -2, -3, -4},
		{10, 20, 50, 100},
		{-10, -20, -50, -100},
	}

	for _, tc := range testCases {
		k := archsimd.LoadFloat64x4Slice(tc)

		resultScalar := scale2kScalar(k)
		resultUnrolled := scale2kScalarUnrolled(k)
		resultMagic := scale2kMagic(k)

		var scalarBuf, unrolledBuf, magicBuf [4]float64
		resultScalar.StoreSlice(scalarBuf[:])
		resultUnrolled.StoreSlice(unrolledBuf[:])
		resultMagic.StoreSlice(magicBuf[:])

		for i, kVal := range tc {
			expected := math.Pow(2, kVal)

			if math.Abs(scalarBuf[i]-expected) > 1e-10 {
				t.Errorf("Scalar: k=%v, got %v, want %v", kVal, scalarBuf[i], expected)
			}
			if math.Abs(unrolledBuf[i]-expected) > 1e-10 {
				t.Errorf("Unrolled: k=%v, got %v, want %v", kVal, unrolledBuf[i], expected)
			}
			if math.Abs(magicBuf[i]-expected) > 1e-10 {
				t.Errorf("Magic: k=%v, got %v, want %v", kVal, magicBuf[i], expected)
			}
		}
	}
}
