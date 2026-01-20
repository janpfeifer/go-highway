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

//go:build amd64 && goexperiment.simd

package varint

import "os"

// AMD64 SIMD dispatch for Stream-VByte.
//
// This file wires the hwygen-generated SIMD implementations (AVX2/AVX-512)
// to the public dispatch variables. The z_ prefix ensures this init() runs
// after dispatch_streamvbyte_amd64.gen.go, which sets up the *Float32 variables.
//
// The generated dispatch creates separate variables (DecodeStreamVByte32Float32, etc.)
// because the base functions don't use generic type parameters. This file bridges
// those to the public API.

func init() {
	// Respect HWY_NO_SIMD to allow fallback testing
	if os.Getenv("HWY_NO_SIMD") != "" {
		return
	}

	// Wire public dispatch to generated SIMD implementations.
	// The *Float32 variables are already set by dispatch_streamvbyte_amd64.gen.go
	// to the appropriate AVX2/AVX-512/fallback implementations.
	DecodeStreamVByte32 = DecodeStreamVByte32Float32
	DecodeStreamVByte32Into = DecodeStreamVByte32IntoFloat32
}
