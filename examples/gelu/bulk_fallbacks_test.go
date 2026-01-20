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

//go:build !arm64 || noasm

package gelu

import "testing"

// Dummy fallbacks for non-ARM64 platforms to allow tests and benchmarks to compile.

func GELUBulkF32(input, output []float32) {
	panic("GELUBulkF32 only implemented on arm64")
}

func GELUBulkF64(input, output []float64) {
	panic("GELUBulkF64 only implemented on arm64")
}

func GELUApproxBulkF32(input, output []float32) {
	panic("GELUApproxBulkF32 only implemented on arm64")
}

func GELUApproxBulkF64(input, output []float64) {
	panic("GELUApproxBulkF64 only implemented on arm64")
}

func TestDummy(t *testing.T) {}
