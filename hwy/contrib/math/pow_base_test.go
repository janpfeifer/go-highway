package math

import (
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestPow_BaseF32(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
		{"e^1 = e", float32(stdmath.E), 1.0, float32(stdmath.E)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float32{tt.x, tt.x, tt.x, tt.x})
			y := hwy.Load([]float32{tt.y, tt.y, tt.y, tt.y})
			result := Pow(x, y)
			got := result.Data()[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPow_BaseF64(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		y    float64
		want float64
	}{
		{"2^3 = 8", 2.0, 3.0, 8.0},
		{"2^0 = 1", 2.0, 0.0, 1.0},
		{"2^1 = 2", 2.0, 1.0, 2.0},
		{"3^2 = 9", 3.0, 2.0, 9.0},
		{"10^2 = 100", 10.0, 2.0, 100.0},
		{"2^-1 = 0.5", 2.0, -1.0, 0.5},
		{"4^0.5 = 2", 4.0, 0.5, 2.0},
		{"8^(1/3) = 2", 8.0, 1.0 / 3.0, 2.0},
		{"1^100 = 1", 1.0, 100.0, 1.0},
		{"e^1 = e", stdmath.E, 1.0, stdmath.E},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float64{tt.x, tt.x, tt.x, tt.x})
			y := hwy.Load([]float64{tt.y, tt.y, tt.y, tt.y})
			result := Pow(x, y)
			got := result.Data()[0]

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestPow_BaseSpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float64
		y       float64
		wantInf int  // 1 for +Inf, -1 for -Inf, 0 for not inf
		wantNaN bool
		want    float64
	}{
		{"0^1 = 0", 0.0, 1.0, 0, false, 0.0},
		{"0^0 = 1", 0.0, 0.0, 0, false, 1.0},
		{"0^-1 = +Inf", 0.0, -1.0, 1, false, 0.0},
		{"1^0 = 1", 1.0, 0.0, 0, false, 1.0},
		{"NaN^0 = 1", stdmath.NaN(), 0.0, 0, false, 1.0},
		{"+Inf^1 = +Inf", stdmath.Inf(1), 1.0, 1, false, 0.0},
		{"+Inf^-1 = 0", stdmath.Inf(1), -1.0, 0, false, 0.0},
		{"+Inf^0 = 1", stdmath.Inf(1), 0.0, 0, false, 1.0},
		{"-1^2 = 1", -1.0, 2.0, 0, false, 1.0}, // integer exponent
		{"-1^2.5 = NaN", -1.0, 2.5, 0, true, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float64{tt.x})
			y := hwy.Load([]float64{tt.y})
			result := Pow(x, y)
			got := result.Data()[0]

			if tt.wantInf != 0 {
				if !stdmath.IsInf(got, tt.wantInf) {
					t.Errorf("Pow(%v, %v) = %v, want Inf(%d)", tt.x, tt.y, got, tt.wantInf)
				}
			} else if tt.wantNaN {
				if !stdmath.IsNaN(got) {
					t.Errorf("Pow(%v, %v) = %v, want NaN", tt.x, tt.y, got)
				}
			} else {
				if stdmath.Abs(got-tt.want) > 1e-10 {
					t.Errorf("Pow(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
				}
			}
		})
	}
}
