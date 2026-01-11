//go:generate hwygen -input poly_base.go -output . -targets avx2,avx512,neon,fallback

package math

import "github.com/ajroetker/go-highway/hwy"

// BasePoly2 evaluates a degree-2 polynomial: c0 + c1*x + c2*x^2
// Using Horner's method: c0 + x*(c1 + x*c2)
// This is a simple test case for the code generator.
func BasePoly2[T hwy.Floats](x []T, c0, c1, c2 T, result []T) {
	size := len(x)
	if len(result) < size {
		size = len(result)
	}

	// Create constant vectors
	vc0 := hwy.Set[T](c0)
	vc1 := hwy.Set[T](c1)
	vc2 := hwy.Set[T](c2)
	lanes := vc0.NumLanes()

	// Process in vector chunks
	for ii := 0; ii < size; ii += lanes {
		vx := hwy.Load(x[ii:])

		// Horner's method: c0 + x*(c1 + x*c2)
		// = c0 + x*c1 + x*x*c2
		// Using MulAdd: result = vc2.MulAdd(vx, vc1) = c2*x + c1
		//               result = result.MulAdd(vx, vc0) = (c2*x+c1)*x + c0
		p := hwy.MulAdd(vc2, vx, vc1) // c2*x + c1
		p = hwy.MulAdd(p, vx, vc0)    // (c2*x + c1)*x + c0

		hwy.Store(p, result[ii:])
	}
}

// BaseClamp clamps values to [minVal, maxVal] range.
// This tests Min/Max operations.
func BaseClamp[T hwy.Floats](input []T, minVal, maxVal T, output []T) {
	size := len(input)
	if len(output) < size {
		size = len(output)
	}

	vmin := hwy.Set[T](minVal)
	vmax := hwy.Set[T](maxVal)
	lanes := vmin.NumLanes()

	for ii := 0; ii < size; ii += lanes {
		vx := hwy.Load(input[ii:])

		// Clamp: max(min(x, maxVal), minVal)
		clamped := hwy.Max(hwy.Min(vx, vmax), vmin)

		hwy.Store(clamped, output[ii:])
	}
}
