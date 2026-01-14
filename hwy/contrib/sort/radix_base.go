//go:build ignore

package sort

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input radix_base.go -output . -targets avx2,avx512,neon,fallback -dispatch radix

// BaseRadixPass performs one pass of LSD radix sort.
// shift specifies which byte to use for bucketing (0, 8, 16, 24 for int32).
// This is the core SIMD-accelerated function for histogram counting.
func BaseRadixPass[T hwy.SignedInts](src, dst []T, shift int) {
	n := len(src)
	if n == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	mask := T(0xFF)
	maskVec := hwy.Set(mask)

	// Count histogram for each bucket (256 buckets for 8 bits)
	var count [256]int

	// SIMD accelerated histogram counting
	i := 0
	for i+lanes <= n {
		v := hwy.Load(src[i:])
		shifted := hwy.ShiftAllRight(v, shift)
		digits := hwy.And(shifted, maskVec)

		// Extract digits and count
		var buf [16]T // Max AVX-512 lanes
		hwy.Store(digits, buf[:])
		for j := 0; j < lanes; j++ {
			digit := int(buf[j] & 0xFF)
			count[digit]++
		}
		i += lanes
	}

	// Handle tail
	for ; i < n; i++ {
		digit := int((src[i] >> shift) & 0xFF)
		count[digit]++
	}

	// Compute prefix sum to get bucket offsets
	offset := 0
	for b := 0; b < 256; b++ {
		c := count[b]
		count[b] = offset
		offset += c
	}

	// Scatter elements to destination
	for i := 0; i < n; i++ {
		digit := int((src[i] >> shift) & 0xFF)
		dst[count[digit]] = src[i]
		count[digit]++
	}
}

// BaseRadixPassSigned performs the final pass for signed integers.
// The MSB byte contains the sign bit, so negative numbers (128-255) come before positive (0-127).
func BaseRadixPassSigned[T hwy.SignedInts](src, dst []T, shift int) {
	n := len(src)
	if n == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	mask := T(0xFF)
	maskVec := hwy.Set(mask)

	var count [256]int

	// SIMD histogram counting
	i := 0
	for i+lanes <= n {
		v := hwy.Load(src[i:])
		shifted := hwy.ShiftAllRight(v, shift)
		digits := hwy.And(shifted, maskVec)

		var buf [16]T
		hwy.Store(digits, buf[:])
		for j := 0; j < lanes; j++ {
			digit := int(buf[j] & 0xFF)
			count[digit]++
		}
		i += lanes
	}

	for ; i < n; i++ {
		digit := int((src[i] >> shift) & 0xFF)
		count[digit]++
	}

	// For signed MSB: buckets 128-255 (negative) come before 0-127 (positive)
	offset := 0
	for b := 128; b < 256; b++ {
		c := count[b]
		count[b] = offset
		offset += c
	}
	for b := 0; b < 128; b++ {
		c := count[b]
		count[b] = offset
		offset += c
	}

	// Scatter
	for i := 0; i < n; i++ {
		digit := int((src[i] >> shift) & 0xFF)
		dst[count[digit]] = src[i]
		count[digit]++
	}
}
