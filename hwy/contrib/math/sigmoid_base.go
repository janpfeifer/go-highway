// NOTE: This file is a simple test implementation. The production sigmoid is
// in math_base.go as BaseSigmoidPoly. No code generation needed here.

package math

import "github.com/ajroetker/go-highway/hwy"

// Constants for sigmoid approximation - regular Go vars, available to all generated files
var (
	sigmoidC1     float32 = 1.0
	sigmoidC2     float32 = 0.5
	sigmoidC3     float32 = 0.25
	sigmoidNegOne float32 = -1.0
)

// BaseSigmoidApprox computes an approximation of sigmoid: 1 / (1 + e^(-x))
// Uses a polynomial approximation for the core computation.
func BaseSigmoidApprox[T hwy.Floats](input []T, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	// Use package-level constants via hwy.Set
	one := hwy.Set[T](T(sigmoidC1))
	half := hwy.Set[T](T(sigmoidC2))
	quarter := hwy.Set[T](T(sigmoidC3))
	negOne := hwy.Set[T](T(sigmoidNegOne))
	lanes := one.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		x := hwy.Load(input[ii:])

		// Simple approximation: 0.5 + 0.25*tanh(x)
		// where tanh is approximated as x for small |x|
		// This is just to test constant handling, not accuracy.

		// Clamp x to [-1, 1] for tanh approximation
		xClamped := hwy.Max(hwy.Min(x, one), negOne)

		// sigmoid approx = 0.5 + 0.25 * xClamped
		result := hwy.MulAdd(quarter, xClamped, half)

		hwy.Store(result, output[ii:])
	}
}
