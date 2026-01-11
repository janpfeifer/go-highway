//go:build darwin && arm64

package matmul

import (
	"testing"
	"unsafe"
)

//go:noescape
func sme_fmopa_only_test(dst unsafe.Pointer)

// TestSMEFMOPAOnly tests just the FMOPA instruction
// without any ZA store - just to see if it runs or crashes
func TestSMEFMOPAOnly(t *testing.T) {
	// Dummy destination - we won't actually store anything
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_fmopa_only_test(unsafe.Pointer(&data[0]))

	t.Log("FMOPA test completed without crash")
	// Just check that we returned successfully
	// The data should still be -1.0 since we're not storing
	for i, v := range data {
		t.Logf("data[%d] = %f", i, v)
	}
}
