package hwy

import (
	"math"
	"testing"
)

func TestSaturatedAddUint8(t *testing.T) {
	a := Load([]uint8{250, 100, 0, 255})
	b := Load([]uint8{10, 50, 100, 1})
	result := SaturatedAdd(a, b)

	expected := []uint8{255, 150, 100, 255} // 250+10 saturates to 255
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("SaturatedAdd uint8: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestSaturatedAddInt8(t *testing.T) {
	a := Load([]int8{120, -120, 50, -50})
	b := Load([]int8{10, -10, 50, -50})
	result := SaturatedAdd(a, b)

	expected := []int8{127, -128, 100, -100} // 120+10=130 saturates to 127
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("SaturatedAdd int8: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestSaturatedAddUint16(t *testing.T) {
	a := Load([]uint16{65530, 100, 0, 65535})
	b := Load([]uint16{10, 50, 100, 1})
	result := SaturatedAdd(a, b)

	expected := []uint16{65535, 150, 100, 65535}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("SaturatedAdd uint16: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestSaturatedSubUint8(t *testing.T) {
	a := Load([]uint8{10, 100, 0, 255})
	b := Load([]uint8{20, 50, 100, 1})
	result := SaturatedSub(a, b)

	expected := []uint8{0, 50, 0, 254} // 10-20 saturates to 0
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("SaturatedSub uint8: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestSaturatedSubInt8(t *testing.T) {
	a := Load([]int8{-120, 120, 50, -50})
	b := Load([]int8{10, -10, 50, -50})
	result := SaturatedSub(a, b)

	expected := []int8{-128, 127, 0, 0} // -120-10=-130 saturates to -128
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("SaturatedSub int8: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestClamp(t *testing.T) {
	v := Load([]float32{-5, 0, 5, 15, 25})
	lo := Load([]float32{0, 0, 0, 0, 0})
	hi := Load([]float32{10, 10, 10, 10, 10})
	result := Clamp(v, lo, hi)

	expected := []float32{0, 0, 5, 10, 10}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Clamp: lane %d: got %f, want %f", i, result.data[i], expected[i])
		}
	}
}

func TestClampInt(t *testing.T) {
	v := Load([]int32{-100, -50, 0, 50, 100})
	lo := Load([]int32{-25, -25, -25, -25, -25})
	hi := Load([]int32{25, 25, 25, 25, 25})
	result := Clamp(v, lo, hi)

	expected := []int32{-25, -25, 0, 25, 25}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Clamp int: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestAbsDiff(t *testing.T) {
	a := Load([]float32{10, 5, -10, -5})
	b := Load([]float32{5, 10, -5, -10})
	result := AbsDiff(a, b)

	expected := []float32{5, 5, 5, 5}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("AbsDiff: lane %d: got %f, want %f", i, result.data[i], expected[i])
		}
	}
}

func TestAbsDiffUint(t *testing.T) {
	a := Load([]uint32{10, 5, 100, 50})
	b := Load([]uint32{5, 10, 50, 100})
	result := AbsDiff(a, b)

	expected := []uint32{5, 5, 50, 50}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("AbsDiff uint: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestAvg(t *testing.T) {
	a := Load([]uint8{10, 20, 255, 0})
	b := Load([]uint8{20, 30, 255, 1})
	result := Avg(a, b)

	// (a + b + 1) / 2, rounded
	expected := []uint8{15, 25, 255, 1}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Avg: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestAvgInt(t *testing.T) {
	a := Load([]int32{-10, 10, -5, 5})
	b := Load([]int32{-20, 20, 6, -4})
	result := Avg(a, b)

	// (a + b + 1) / 2 with integer division
	// Lane 0: (-10 + -20 + 1) / 2 = -29/2 = -14
	// Lane 1: (10 + 20 + 1) / 2 = 31/2 = 15
	// Lane 2: (-5 + 6 + 1) / 2 = 2/2 = 1
	// Lane 3: (5 + -4 + 1) / 2 = 2/2 = 1
	expected := []int32{-14, 15, 1, 1}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Avg int: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}

func TestMulHigh(t *testing.T) {
	a := Load([]uint16{0x8000, 0x4000, 0x1000, 0x0100})
	b := Load([]uint16{0x8000, 0x4000, 0x1000, 0x0100})
	result := MulHigh(a, b)

	// High 16 bits of product
	expected := []uint16{0x4000, 0x1000, 0x0100, 0x0001}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("MulHigh: lane %d: got 0x%04X, want 0x%04X", i, result.data[i], expected[i])
		}
	}
}

func TestMulHighSigned(t *testing.T) {
	a := Load([]int16{0x4000, -0x4000, 0x1000, -0x1000})
	b := Load([]int16{0x4000, 0x4000, 0x1000, -0x1000})
	result := MulHigh(a, b)

	// 0x4000 * 0x4000 = 0x10000000, high 16 bits = 0x1000
	// -0x4000 * 0x4000 = -0x10000000, high 16 bits = -0x1000
	expected := []int16{0x1000, -0x1000, 0x0100, 0x0100}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("MulHigh signed: lane %d: got 0x%04X, want 0x%04X", i, result.data[i], expected[i])
		}
	}
}

func TestNotEqual(t *testing.T) {
	a := Load([]float32{1, 2, 3, 4})
	b := Load([]float32{1, 5, 3, 7})
	mask := NotEqual(a, b)

	if mask.GetBit(0) {
		t.Error("NotEqual: expected lane 0 to be false (1 == 1)")
	}
	if !mask.GetBit(1) {
		t.Error("NotEqual: expected lane 1 to be true (2 != 5)")
	}
	if mask.GetBit(2) {
		t.Error("NotEqual: expected lane 2 to be false (3 == 3)")
	}
	if !mask.GetBit(3) {
		t.Error("NotEqual: expected lane 3 to be true (4 != 7)")
	}
}

func TestIsNaN(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	v := Load([]float32{1.0, nan, inf, 0.0})
	mask := IsNaN(v)

	if mask.GetBit(0) {
		t.Error("IsNaN: lane 0 should be false (1.0 is not NaN)")
	}
	if !mask.GetBit(1) {
		t.Error("IsNaN: lane 1 should be true (NaN)")
	}
	if mask.GetBit(2) {
		t.Error("IsNaN: lane 2 should be false (Inf is not NaN)")
	}
	if mask.GetBit(3) {
		t.Error("IsNaN: lane 3 should be false (0.0 is not NaN)")
	}
}

func TestIsInf(t *testing.T) {
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	nan := float32(math.NaN())
	v := Load([]float32{posInf, negInf, nan, 1.0})

	// Test for any infinity
	maskAny := IsInf(v, 0)
	if !maskAny.GetBit(0) {
		t.Error("IsInf(0): lane 0 should be true (+Inf)")
	}
	if !maskAny.GetBit(1) {
		t.Error("IsInf(0): lane 1 should be true (-Inf)")
	}
	if maskAny.GetBit(2) {
		t.Error("IsInf(0): lane 2 should be false (NaN)")
	}
	if maskAny.GetBit(3) {
		t.Error("IsInf(0): lane 3 should be false (1.0)")
	}

	// Test for positive infinity only
	maskPos := IsInf(v, 1)
	if !maskPos.GetBit(0) {
		t.Error("IsInf(1): lane 0 should be true (+Inf)")
	}
	if maskPos.GetBit(1) {
		t.Error("IsInf(1): lane 1 should be false (-Inf)")
	}

	// Test for negative infinity only
	maskNeg := IsInf(v, -1)
	if maskNeg.GetBit(0) {
		t.Error("IsInf(-1): lane 0 should be false (+Inf)")
	}
	if !maskNeg.GetBit(1) {
		t.Error("IsInf(-1): lane 1 should be true (-Inf)")
	}
}

func TestIsFinite(t *testing.T) {
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	nan := float32(math.NaN())
	v := Load([]float32{1.0, posInf, negInf, nan})
	mask := IsFinite(v)

	if !mask.GetBit(0) {
		t.Error("IsFinite: lane 0 should be true (1.0)")
	}
	if mask.GetBit(1) {
		t.Error("IsFinite: lane 1 should be false (+Inf)")
	}
	if mask.GetBit(2) {
		t.Error("IsFinite: lane 2 should be false (-Inf)")
	}
	if mask.GetBit(3) {
		t.Error("IsFinite: lane 3 should be false (NaN)")
	}
}

func TestTestBit(t *testing.T) {
	v := Load([]uint32{0x01, 0x02, 0x04, 0x08})

	// Test bit 0
	mask0 := TestBit(v, 0)
	if !mask0.GetBit(0) {
		t.Error("TestBit(0): lane 0 should be true (0x01 has bit 0 set)")
	}
	if mask0.GetBit(1) {
		t.Error("TestBit(0): lane 1 should be false (0x02 does not have bit 0 set)")
	}

	// Test bit 1
	mask1 := TestBit(v, 1)
	if mask1.GetBit(0) {
		t.Error("TestBit(1): lane 0 should be false (0x01 does not have bit 1 set)")
	}
	if !mask1.GetBit(1) {
		t.Error("TestBit(1): lane 1 should be true (0x02 has bit 1 set)")
	}

	// Test bit 2
	mask2 := TestBit(v, 2)
	if !mask2.GetBit(2) {
		t.Error("TestBit(2): lane 2 should be true (0x04 has bit 2 set)")
	}

	// Test bit 3
	mask3 := TestBit(v, 3)
	if !mask3.GetBit(3) {
		t.Error("TestBit(3): lane 3 should be true (0x08 has bit 3 set)")
	}
}

func TestIfThenElseZero(t *testing.T) {
	mask := Equal(
		Load([]float32{1, 2, 3, 4}),
		Load([]float32{1, 0, 3, 0}),
	)
	a := Set[float32](100.0)
	result := IfThenElseZero(mask, a)

	if result.data[0] != 100.0 {
		t.Errorf("IfThenElseZero: lane 0: got %v, want 100.0", result.data[0])
	}
	if result.data[1] != 0.0 {
		t.Errorf("IfThenElseZero: lane 1: got %v, want 0.0", result.data[1])
	}
	if result.data[2] != 100.0 {
		t.Errorf("IfThenElseZero: lane 2: got %v, want 100.0", result.data[2])
	}
	if result.data[3] != 0.0 {
		t.Errorf("IfThenElseZero: lane 3: got %v, want 0.0", result.data[3])
	}
}

func TestIfThenZeroElse(t *testing.T) {
	mask := Equal(
		Load([]float32{1, 2, 3, 4}),
		Load([]float32{1, 0, 3, 0}),
	)
	b := Set[float32](200.0)
	result := IfThenZeroElse(mask, b)

	if result.data[0] != 0.0 {
		t.Errorf("IfThenZeroElse: lane 0: got %v, want 0.0", result.data[0])
	}
	if result.data[1] != 200.0 {
		t.Errorf("IfThenZeroElse: lane 1: got %v, want 200.0", result.data[1])
	}
	if result.data[2] != 0.0 {
		t.Errorf("IfThenZeroElse: lane 2: got %v, want 0.0", result.data[2])
	}
	if result.data[3] != 200.0 {
		t.Errorf("IfThenZeroElse: lane 3: got %v, want 200.0", result.data[3])
	}
}

func TestZeroIfNegative(t *testing.T) {
	v := Load([]float32{-5, 0, 5, -10})
	result := ZeroIfNegative(v)

	expected := []float32{0, 0, 5, 0}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("ZeroIfNegative: lane %d: got %f, want %f", i, result.data[i], expected[i])
		}
	}
}

func TestZeroIfNegativeInt(t *testing.T) {
	v := Load([]int32{-100, 0, 100, -1})
	result := ZeroIfNegative(v)

	expected := []int32{0, 0, 100, 0}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("ZeroIfNegative int: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}
}
