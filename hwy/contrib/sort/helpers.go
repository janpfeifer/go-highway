package sort

import "github.com/ajroetker/go-highway/hwy"

// Helper functions for sorting that are shared across all implementations.
// These are not processed by hwygen - they remain generic.

// InsertionSortSmall is a simple insertion sort for small arrays.
func InsertionSortSmall[T hwy.Lanes](data []T) {
	for i := 1; i < len(data); i++ {
		key := data[i]
		j := i - 1
		for j >= 0 && data[j] > key {
			data[j+1] = data[j]
			j--
		}
		data[j+1] = key
	}
}

// MaxValue returns the maximum representable value for the type.
func MaxValue[T hwy.Lanes]() T {
	var maxVal T
	switch any(maxVal).(type) {
	case float32:
		maxVal = T(any(float32(3.4028235e+38)).(float32))
	case float64:
		maxVal = T(any(float64(1.7976931348623157e+308)).(float64))
	case int32:
		maxVal = T(any(int32(2147483647)).(int32))
	case int64:
		maxVal = T(any(int64(9223372036854775807)).(int64))
	}
	return maxVal
}

// BitonicMerge performs in-place bitonic merge on a slice.
func BitonicMerge[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Bitonic merge network
	for k := n / 2; k > 0; k /= 2 {
		for i := 0; i < n; i++ {
			j := i ^ k
			if j > i && data[i] > data[j] {
				data[i], data[j] = data[j], data[i]
			}
		}
	}
}

// SortSingleVector sorts elements that fit in one vector.
func SortSingleVector[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// Pad to full vector
	buf := make([]T, lanes)
	copy(buf, data)

	// Fill remaining with max value to push to end
	maxVal := MaxValue[T]()
	for i := n; i < lanes; i++ {
		buf[i] = maxVal
	}

	// Sort using insertion sort (simple for base impl)
	InsertionSortSmall(buf)

	// Copy back valid elements
	copy(data, buf[:n])
}

// PivotMedianOf3 selects pivot as median of first, middle, and last elements.
func PivotMedianOf3[T hwy.Lanes](data []T) T {
	n := len(data)
	if n <= 2 {
		return data[0]
	}

	a := data[0]
	b := data[n/2]
	c := data[n-1]

	if a > b {
		a, b = b, a
	}
	if b > c {
		b = c
		if a > b {
			b = a
		}
	}
	return b
}

// PivotSampled selects pivot by sampling elements at regular intervals.
// For larger arrays, this gives a better pivot estimate than median-of-3.
func PivotSampled[T hwy.Lanes](data []T) T {
	n := len(data)
	if n <= 8 {
		return PivotMedianOf3(data)
	}

	samples := []T{
		data[0],
		data[n/4],
		data[n/2],
		data[3*n/4],
		data[n-1],
	}

	InsertionSortSmall(samples)
	return samples[2]
}

// scalarPartition3Way performs scalar 3-way partitioning (Dutch National Flag).
// Used as fallback for small arrays.
func scalarPartition3Way[T hwy.Lanes](data []T, pivot T) (int, int) {
	lt := 0
	gt := len(data)
	i := 0

	for i < gt {
		if data[i] < pivot {
			data[lt], data[i] = data[i], data[lt]
			lt++
			i++
		} else if data[i] > pivot {
			gt--
			data[i], data[gt] = data[gt], data[i]
		} else {
			i++
		}
	}

	return lt, gt
}

// scalarPartition2Way performs scalar 2-way partitioning.
// Used as fallback for small arrays.
func scalarPartition2Way[T hwy.Lanes](data []T, pivot T) int {
	left := 0
	right := len(data)

	for left < right {
		if data[left] <= pivot {
			left++
		} else {
			right--
			data[left], data[right] = data[right], data[left]
		}
	}

	return left
}

// SortTwoVectors sorts elements that fit in two vectors using bitonic merge.
func SortTwoVectors[T hwy.Lanes](data []T) {
	n := len(data)
	if n <= 1 {
		return
	}

	lanes := hwy.MaxLanes[T]()

	// Pad both vectors
	buf1 := make([]T, lanes)
	buf2 := make([]T, lanes)

	copy(buf1, data)
	if n > lanes {
		copy(buf2, data[lanes:])
	}

	// Fill with max value
	maxVal := MaxValue[T]()

	// Pad buf1 if needed
	for i := n; i < lanes; i++ {
		buf1[i] = maxVal
	}

	// Pad buf2
	remaining := n - lanes
	if remaining < 0 {
		remaining = 0
	}
	for i := remaining; i < lanes; i++ {
		buf2[i] = maxVal
	}

	// Sort each half
	InsertionSortSmall(buf1)
	InsertionSortSmall(buf2)

	// Create bitonic sequence: first half ascending, second half descending
	// Bitonic merge requires input to be bitonic (increasing then decreasing)
	merged := make([]T, lanes*2)
	copy(merged[:lanes], buf1)
	// Reverse buf2 into the second half to create bitonic sequence
	for i := 0; i < lanes; i++ {
		merged[lanes+i] = buf2[lanes-1-i]
	}
	BitonicMerge(merged)

	// Copy back
	copy(data, merged[:n])
}
