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

//go:build !amd64 || !goexperiment.simd

package asm

// Stub implementations for non-AMD64 or non-SIMD builds.
// These should never be called - the hwy package will use scalar fallbacks.

// PromoteF16ToF32F16C stub for non-F16C platforms.
func PromoteF16ToF32F16C(a []uint16, result []float32) {
	panic("F16C not available")
}

// DemoteF32ToF16F16C stub for non-F16C platforms.
func DemoteF32ToF16F16C(a []float32, result []uint16) {
	panic("F16C not available")
}
