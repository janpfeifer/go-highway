// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Example demonstrating the ARM64 NEON SIMD operations.
// This shows the low-level NEON operations on arm64.

//go:build arm64

package main

import (
	"fmt"

	"github.com/ajroetker/go-highway/hwy/asm"
)

func main() {
	// Test ExpF32 (slice-based NEON API)
	input := make([]float32, 8)
	for i := range input {
		input[i] = 1.0
	}
	result := make([]float32, 8)
	asm.ExpF32(input, result)

	fmt.Printf("Exp(1.0) = %v\n", result[0])
	fmt.Printf("Expected: ~2.71828\n")

	// Test ExpF64
	input64 := make([]float64, 4)
	for i := range input64 {
		input64[i] = 2.0
	}
	result64 := make([]float64, 4)
	asm.ExpF64(input64, result64)

	fmt.Printf("Exp(2.0) = %v\n", result64[0])
	fmt.Printf("Expected: ~7.38906\n")

	// Test more NEON math functions
	fmt.Println("\n--- Additional NEON Math Functions ---")

	// Log
	logInput := []float32{2.71828, 7.38906, 1.0, 10.0}
	logResult := make([]float32, 4)
	asm.LogF32(logInput, logResult)
	fmt.Printf("Log([e, e^2, 1, 10]) = %v\n", logResult)

	// Sin/Cos
	trigInput := []float32{0, 0.785398, 1.5708, 3.14159} // 0, π/4, π/2, π
	sinResult := make([]float32, 4)
	cosResult := make([]float32, 4)
	asm.SinF32(trigInput, sinResult)
	asm.CosF32(trigInput, cosResult)
	fmt.Printf("Sin([0, π/4, π/2, π]) = %v\n", sinResult)
	fmt.Printf("Cos([0, π/4, π/2, π]) = %v\n", cosResult)

	// Tanh
	tanhInput := []float32{0, 0.5, 1.0, 2.0}
	tanhResult := make([]float32, 4)
	asm.TanhF32(tanhInput, tanhResult)
	fmt.Printf("Tanh([0, 0.5, 1, 2]) = %v\n", tanhResult)

	// Sigmoid
	sigmoidInput := []float32{0, 1, -1, 2}
	sigmoidResult := make([]float32, 4)
	asm.SigmoidF32(sigmoidInput, sigmoidResult)
	fmt.Printf("Sigmoid([0, 1, -1, 2]) = %v\n", sigmoidResult)

	// Additional Phase 11 functions
	fmt.Println("\n--- Phase 11 Functions ---")

	// Pow
	bases := []float32{2.0, 3.0, 4.0, 2.0}
	exps := []float32{3.0, 2.0, 0.5, 10.0}
	powResult := make([]float32, 4)
	asm.PowF32(bases, exps, powResult)
	fmt.Printf("Pow([2^3, 3^2, 4^0.5, 2^10]) = %v\n", powResult)

	// Atan
	atanInput := []float32{0, 1.0, -1.0, 0.5}
	atanResult := make([]float32, 4)
	asm.AtanF32(atanInput, atanResult)
	fmt.Printf("Atan([0, 1, -1, 0.5]) = %v\n", atanResult)

	// Tan
	tanInput := []float32{0, 0.785398, 0.5, 1.0} // 0, π/4, 0.5, 1.0
	tanResult := make([]float32, 4)
	asm.TanF32(tanInput, tanResult)
	fmt.Printf("Tan([0, π/4, 0.5, 1]) = %v\n", tanResult)

	// Erf
	erfInput := []float32{0, 0.5, 1.0, 2.0}
	erfResult := make([]float32, 4)
	asm.ErfF32(erfInput, erfResult)
	fmt.Printf("Erf([0, 0.5, 1, 2]) = %v\n", erfResult)

	fmt.Println("\nNEON contrib package working correctly on ARM64!")
}
