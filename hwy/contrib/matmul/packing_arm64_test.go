//go:build arm64

package matmul

import (
	"os"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// TestDispatchFunctionPointer checks what the dispatch function variable is pointing to.
func TestDispatchFunctionPointer(t *testing.T) {
	t.Logf("=== Environment ===")
	t.Logf("HWY_NO_SIMD=%q", os.Getenv("HWY_NO_SIMD"))
	lanes := hwy.Zero[float32]().NumLanes()
	t.Logf("lanes=%d, CurrentName=%s", lanes, hwy.CurrentName())

	// Check if the function pointer is nil
	if PackedMicroKernelFloat32 == nil {
		t.Errorf("PackedMicroKernelFloat32 is NIL!")
		return
	}
	t.Logf("PackedMicroKernelFloat32 is not nil")

	mr, nr := 4, 8
	m, n, k := 16, 16, 16

	packedA := make([]float32, k*mr)
	for i := range packedA {
		packedA[i] = float32(i + 1)
	}
	packedB := make([]float32, k*nr)
	for i := range packedB {
		packedB[i] = float32(i + 1)
	}

	// Call the function variable directly (not via generic dispatch)
	t.Logf("=== Calling PackedMicroKernelFloat32 directly ===")
	c1 := make([]float32, m*n)
	PackedMicroKernelFloat32(packedA, packedB, c1, n, 12, 8, k, mr, nr)
	t.Logf("Direct variable call: c[200:208] = %v", c1[200:208])

	// Call via generic dispatch
	t.Logf("=== Calling via generic PackedMicroKernel ===")
	c2 := make([]float32, m*n)
	PackedMicroKernel(packedA, packedB, c2, n, 12, 8, k, mr, nr)
	t.Logf("Generic dispatch: c[200:208] = %v", c2[200:208])

	// Compare
	if c1[200] == c2[200] {
		t.Logf("Direct and generic produce same result")
		if c1[200] == 0 {
			t.Errorf("Both produce 0 - the function variable points to wrong function!")
		}
	} else {
		t.Errorf("Direct (%f) and generic (%f) produce different results!", c1[200], c2[200])
	}

	// Call fallback for comparison
	t.Logf("=== Calling fallback directly ===")
	c3 := make([]float32, m*n)
	BasePackedMicroKernel_fallback(packedA, packedB, c3, n, 12, 8, k, mr, nr)
	t.Logf("Fallback: c[200:208] = %v", c3[200:208])

	if c1[200] == c3[200] {
		t.Logf("Function variable matches fallback")
	} else {
		t.Logf("Function variable does NOT match fallback")
	}
}

// TestNeonKernelDirect tests the NEON kernel directly with ir=12, jr=8.
func TestNeonKernelDirect(t *testing.T) {
	t.Logf("=== Environment ===")
	t.Logf("HWY_NO_SIMD=%q", os.Getenv("HWY_NO_SIMD"))
	lanes := hwy.Zero[float32]().NumLanes()
	t.Logf("lanes=%d, CurrentName=%s", lanes, hwy.CurrentName())

	mr, nr := 4, 8
	m, n, k := 16, 16, 16

	packedA := make([]float32, k*mr)
	for i := range packedA {
		packedA[i] = float32(i + 1)
	}
	packedB := make([]float32, k*nr)
	for i := range packedB {
		packedB[i] = float32(i + 1)
	}

	// Call fallback directly
	cFallback := make([]float32, m*n)
	BasePackedMicroKernel_fallback(packedA, packedB, cFallback, n, 12, 8, k, mr, nr)

	// Call NEON directly
	cNeon := make([]float32, m*n)
	BasePackedMicroKernel_neon(packedA, packedB, cNeon, n, 12, 8, k, mr, nr)

	// Call via dispatch
	cDispatch := make([]float32, m*n)
	PackedMicroKernel(packedA, packedB, cDispatch, n, 12, 8, k, mr, nr)

	t.Logf("Fallback result: c[200:208] = %v", cFallback[200:208])
	t.Logf("NEON result:     c[200:208] = %v", cNeon[200:208])
	t.Logf("Dispatch result: c[200:208] = %v", cDispatch[200:208])

	// Check which one dispatch matches
	fallbackMatch := true
	neonMatch := true
	for i := 200; i < 208; i++ {
		if cDispatch[i] != cFallback[i] {
			fallbackMatch = false
		}
		if cDispatch[i] != cNeon[i] {
			neonMatch = false
		}
	}

	if fallbackMatch && !neonMatch {
		t.Logf("Dispatch is using FALLBACK kernel")
	} else if neonMatch && !fallbackMatch {
		t.Logf("Dispatch is using NEON kernel")
	} else if fallbackMatch && neonMatch {
		t.Logf("Fallback and NEON produce same result (both work or both broken)")
	} else {
		t.Logf("Dispatch matches NEITHER fallback nor NEON!")
	}

	// Check if NEON kernel has the bug
	if cNeon[200] == 0 && cFallback[200] != 0 {
		t.Errorf("NEON KERNEL BUG: NEON produces 0 but fallback produces %f", cFallback[200])
	}

	// Check if dispatch is using the wrong kernel
	if cDispatch[200] == 0 && cFallback[200] != 0 {
		if neonMatch {
			t.Errorf("Dispatch uses NEON which has bug. With HWY_NO_SIMD=%q, should use fallback!", os.Getenv("HWY_NO_SIMD"))
		} else {
			t.Errorf("Dispatch produces 0 but doesn't match NEON or fallback!")
		}
	}
}
