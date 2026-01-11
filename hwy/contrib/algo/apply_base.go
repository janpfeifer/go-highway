package algo

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input apply_base.go -output . -targets avx2,avx512,neon,fallback

// BaseApply transforms input slice to output slice using the provided vector function.
// Tail elements are handled via buffer-based SIMD processing - no scalar fallback needed.
//
// This is the core primitive for zero-allocation bulk transforms.
// The fn parameter is a Vec[T] -> Vec[T] function that processes one vector at a time.
//
// Example usage:
//
//	Apply(input, output, math.BaseExpVec)
//
func BaseApply[T hwy.Floats](in, out []T, fn func(hwy.Vec[T]) hwy.Vec[T]) {
	n := min(len(in), len(out))
	lanes := hwy.MaxLanes[T]()
	i := 0

	// Process full vectors
	for ; i+lanes <= n; i += lanes {
		x := hwy.Load(in[i:])
		hwy.Store(fn(x), out[i:])
	}

	// Buffer-based tail handling
	if remaining := n - i; remaining > 0 {
		buf := make([]T, lanes)
		copy(buf, in[i:i+remaining])
		x := hwy.Load(buf)
		hwy.Store(fn(x), buf)
		copy(out[i:i+remaining], buf[:remaining])
	}
}
