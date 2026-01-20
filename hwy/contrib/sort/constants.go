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

package sort

// =============================================================================
// Constants for radix sort
// =============================================================================

// 8-bit mask constants for radix sort
var (
	radixMask8_i32 int32 = 0xFF
	radixMask8_i64 int64 = 0xFF
)

// 16-bit mask constants for radix sort
var (
	radixMask16_i32 int32 = 0xFFFF
	radixMask16_i64 int64 = 0xFFFF
)
