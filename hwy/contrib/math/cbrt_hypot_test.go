package math

import (
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

func TestCbrt_F32(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		want float32
	}{
		{"cbrt(8) = 2", 8.0, 2.0},
		{"cbrt(27) = 3", 27.0, 3.0},
		{"cbrt(1) = 1", 1.0, 1.0},
		{"cbrt(0) = 0", 0.0, 0.0},
		{"cbrt(-8) = -2", -8.0, -2.0},
		{"cbrt(-27) = -3", -27.0, -3.0},
		{"cbrt(64) = 4", 64.0, 4.0},
		{"cbrt(125) = 5", 125.0, 5.0},
		{"cbrt(0.001) = 0.1", 0.001, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float32{tt.x, tt.x, tt.x, tt.x})
			result := Cbrt(x)
			got := result.Data()[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Cbrt(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestCbrt_F64(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{"cbrt(8) = 2", 8.0, 2.0},
		{"cbrt(27) = 3", 27.0, 3.0},
		{"cbrt(1) = 1", 1.0, 1.0},
		{"cbrt(0) = 0", 0.0, 0.0},
		{"cbrt(-8) = -2", -8.0, -2.0},
		{"cbrt(-27) = -3", -27.0, -3.0},
		{"cbrt(64) = 4", 64.0, 4.0},
		{"cbrt(125) = 5", 125.0, 5.0},
		{"cbrt(0.001) = 0.1", 0.001, 0.1},
		{"cbrt(1000) = 10", 1000.0, 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float64{tt.x, tt.x, tt.x, tt.x})
			result := Cbrt(x)
			got := result.Data()[0]

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Cbrt(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestCbrt_SpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float64
		wantInf int
		wantNaN bool
		want    float64
	}{
		{"+Inf", stdmath.Inf(1), 1, false, 0.0},
		{"-Inf", stdmath.Inf(-1), -1, false, 0.0},
		{"NaN", stdmath.NaN(), 0, true, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float64{tt.x})
			result := Cbrt(x)
			got := result.Data()[0]

			if tt.wantInf != 0 {
				if !stdmath.IsInf(got, tt.wantInf) {
					t.Errorf("Cbrt(%v) = %v, want Inf(%d)", tt.x, got, tt.wantInf)
				}
			} else if tt.wantNaN {
				if !stdmath.IsNaN(got) {
					t.Errorf("Cbrt(%v) = %v, want NaN", tt.x, got)
				}
			}
		})
	}
}

func TestHypot_F32(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		y    float32
		want float32
	}{
		{"hypot(3, 4) = 5", 3.0, 4.0, 5.0},
		{"hypot(5, 12) = 13", 5.0, 12.0, 13.0},
		{"hypot(8, 15) = 17", 8.0, 15.0, 17.0},
		{"hypot(0, 0) = 0", 0.0, 0.0, 0.0},
		{"hypot(1, 0) = 1", 1.0, 0.0, 1.0},
		{"hypot(0, 1) = 1", 0.0, 1.0, 1.0},
		{"hypot(-3, 4) = 5", -3.0, 4.0, 5.0},
		{"hypot(3, -4) = 5", 3.0, -4.0, 5.0},
		{"hypot(-3, -4) = 5", -3.0, -4.0, 5.0},
		{"hypot(1, 1) = sqrt(2)", 1.0, 1.0, float32(stdmath.Sqrt(2))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float32{tt.x, tt.x, tt.x, tt.x})
			y := hwy.Load([]float32{tt.y, tt.y, tt.y, tt.y})
			result := Hypot(x, y)
			got := result.Data()[0]

			if stdmath.Abs(float64(got-tt.want)) > 1e-4 {
				t.Errorf("Hypot(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestHypot_F64(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		y    float64
		want float64
	}{
		{"hypot(3, 4) = 5", 3.0, 4.0, 5.0},
		{"hypot(5, 12) = 13", 5.0, 12.0, 13.0},
		{"hypot(8, 15) = 17", 8.0, 15.0, 17.0},
		{"hypot(0, 0) = 0", 0.0, 0.0, 0.0},
		{"hypot(1, 0) = 1", 1.0, 0.0, 1.0},
		{"hypot(0, 1) = 1", 0.0, 1.0, 1.0},
		{"hypot(-3, 4) = 5", -3.0, 4.0, 5.0},
		{"hypot(3, -4) = 5", 3.0, -4.0, 5.0},
		{"hypot(-3, -4) = 5", -3.0, -4.0, 5.0},
		{"hypot(1, 1) = sqrt(2)", 1.0, 1.0, stdmath.Sqrt(2)},
		{"hypot(7, 24) = 25", 7.0, 24.0, 25.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float64{tt.x, tt.x, tt.x, tt.x})
			y := hwy.Load([]float64{tt.y, tt.y, tt.y, tt.y})
			result := Hypot(x, y)
			got := result.Data()[0]

			if stdmath.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Hypot(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
			}
		})
	}
}

func TestHypot_SpecialCases(t *testing.T) {
	tests := []struct {
		name    string
		x       float64
		y       float64
		wantInf bool
	}{
		{"+Inf, any", stdmath.Inf(1), 5.0, true},
		{"any, +Inf", 5.0, stdmath.Inf(1), true},
		{"-Inf, any", stdmath.Inf(-1), 5.0, true},
		{"any, -Inf", 5.0, stdmath.Inf(-1), true},
		{"+Inf, -Inf", stdmath.Inf(1), stdmath.Inf(-1), true},
		{"+Inf, NaN", stdmath.Inf(1), stdmath.NaN(), true}, // Inf takes precedence
		{"NaN, +Inf", stdmath.NaN(), stdmath.Inf(1), true}, // Inf takes precedence
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := hwy.Load([]float64{tt.x})
			y := hwy.Load([]float64{tt.y})
			result := Hypot(x, y)
			got := result.Data()[0]

			if tt.wantInf && !stdmath.IsInf(got, 1) {
				t.Errorf("Hypot(%v, %v) = %v, want +Inf", tt.x, tt.y, got)
			}
		})
	}
}

func TestHypot_NaN(t *testing.T) {
	// NaN with finite number should produce NaN
	x := hwy.Load([]float64{stdmath.NaN()})
	y := hwy.Load([]float64{5.0})
	result := Hypot(x, y)
	got := result.Data()[0]

	if !stdmath.IsNaN(got) {
		t.Errorf("Hypot(NaN, 5) = %v, want NaN", got)
	}
}

func BenchmarkCbrt_F32(b *testing.B) {
	x := hwy.Load([]float32{27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Cbrt(x)
	}
}

func BenchmarkCbrt_F64(b *testing.B) {
	x := hwy.Load([]float64{27.0, 27.0, 27.0, 27.0})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Cbrt(x)
	}
}

func BenchmarkHypot_F32(b *testing.B) {
	x := hwy.Load([]float32{3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0})
	y := hwy.Load([]float32{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Hypot(x, y)
	}
}

func BenchmarkHypot_F64(b *testing.B) {
	x := hwy.Load([]float64{3.0, 3.0, 3.0, 3.0})
	y := hwy.Load([]float64{4.0, 4.0, 4.0, 4.0})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Hypot(x, y)
	}
}
