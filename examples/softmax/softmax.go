// Package softmax demonstrates the hwygen workflow with a Softmax implementation.
//
// This example shows how to write portable SIMD code that gets transformed
// into target-specific implementations (AVX2, AVX-512, fallback).
//
// Usage:
//
//	go generate ./...
//	GOEXPERIMENT=simd go build
package softmax

//go:generate hwygen -input softmax.go -output . -targets avx2,avx512,fallback

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// BaseSoftmax computes the softmax function over the input slice.
//
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// The max subtraction provides numerical stability by preventing overflow
// in the exponential computation.
//
// This function processes the input in SIMD-width chunks. For inputs that
// aren't multiples of the SIMD width, tail elements are handled correctly.
func BaseSoftmax[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	if size == 0 {
		return
	}

	// Step 1: Find the maximum value for numerical stability
	maxVal := input[0]
	for i := 1; i < size; i++ {
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}

	// Step 2: Compute exp(x - max) and sum
	vMax := hwy.Set(maxVal)
	var expSum T

	// First pass: compute exponentials and sum
	for ii := 0; ii < size; ii += vMax.NumLanes() {
		remaining := size - ii
		if remaining >= vMax.NumLanes() {
			x := hwy.Load(input[ii:])
			xShifted := hwy.Sub(x, vMax)
			expX := math.Exp(xShifted)
			hwy.Store(expX, output[ii:])
			expSum += hwy.ReduceSum(expX)
		} else {
			// Handle tail elements scalar
			for i := ii; i < size; i++ {
				exp := T(stdmath.Exp(float64(input[i] - maxVal)))
				output[i] = exp
				expSum += exp
			}
		}
	}

	// Step 3: Normalize by dividing by sum
	vSum := hwy.Set(expSum)
	for ii := 0; ii < size; ii += vSum.NumLanes() {
		remaining := size - ii
		if remaining >= vSum.NumLanes() {
			expX := hwy.Load(output[ii:])
			normalized := hwy.Div(expX, vSum)
			hwy.Store(normalized, output[ii:])
		} else {
			// Handle tail elements scalar
			for i := ii; i < size; i++ {
				output[i] = output[i] / expSum
			}
		}
	}
}
