# go-highway/hwy - Phase 1: Core Types and Base Operations

This package provides portable SIMD operations with runtime CPU dispatch, inspired by Google's Highway C++ library.

## Implementation Status - Phase 1 ✓

### Core Types (`types.go`) ✓
- [x] `Lanes` interface - constraint for all SIMD-compatible types
  - [x] `Floats` - float32, float64
  - [x] `Integers` - signed and unsigned integers
- [x] `Vec[T Lanes]` - portable vector handle
  - [x] `NumLanes()` - returns number of lanes
  - [x] `Data()` - returns underlying slice (for testing)
- [x] `Mask[T Lanes]` - comparison result mask
  - [x] `AllTrue()` - all lanes active
  - [x] `AnyTrue()` - at least one lane active
  - [x] `CountTrue()` - count active lanes
  - [x] `GetBit(i)` - check individual lane

### Architecture Tags (`tags.go`) ✓
- [x] `Tag` interface - vector size control
- [x] `ScalableTag[T]` - adapts to widest available SIMD
- [x] `FixedTag128[T]` - force 128-bit SIMD (SSE, NEON)
- [x] `FixedTag256[T]` - force 256-bit SIMD (AVX2)
- [x] `FixedTag512[T]` - force 512-bit SIMD (AVX-512, SVE)

### Base Operations (`ops_base.go`) ✓

All operations implemented in pure Go (scalar fallback):

**Memory Operations:**
- [x] `Load[T](src []T) Vec[T]` - load from slice
- [x] `Store[T](v Vec[T], dst []T)` - store to slice
- [x] `Set[T](value T) Vec[T]` - all lanes same value
- [x] `Zero[T]() Vec[T]` - all lanes zero

**Arithmetic Operations:**
- [x] `Add[T](a, b Vec[T]) Vec[T]` - element-wise addition
- [x] `Sub[T](a, b Vec[T]) Vec[T]` - element-wise subtraction
- [x] `Mul[T](a, b Vec[T]) Vec[T]` - element-wise multiplication
- [x] `Div[T Floats](a, b Vec[T]) Vec[T]` - element-wise division (floats only)
- [x] `Neg[T](v Vec[T]) Vec[T]` - negate all lanes
- [x] `Abs[T](v Vec[T]) Vec[T]` - absolute value

**Min/Max Operations:**
- [x] `Min[T](a, b Vec[T]) Vec[T]` - element-wise minimum
- [x] `Max[T](a, b Vec[T]) Vec[T]` - element-wise maximum

**Special Math (Floats):**
- [x] `Sqrt[T Floats](v Vec[T]) Vec[T]` - square root
- [x] `FMA[T Floats](a, b, c Vec[T]) Vec[T]` - fused multiply-add

**Reduction:**
- [x] `ReduceSum[T](v Vec[T]) T` - sum all lanes

**Comparison Operations:**
- [x] `Equal[T](a, b Vec[T]) Mask[T]` - element-wise equality
- [x] `LessThan[T](a, b Vec[T]) Mask[T]` - element-wise less-than
- [x] `GreaterThan[T](a, b Vec[T]) Mask[T]` - element-wise greater-than

**Masked Operations:**
- [x] `IfThenElse[T](mask Mask[T], a, b Vec[T]) Vec[T]` - conditional selection
- [x] `MaskLoad[T](mask Mask[T], src []T) Vec[T]` - masked load
- [x] `MaskStore[T](mask Mask[T], v Vec[T], dst []T)` - masked store

### Runtime Dispatch (`dispatch.go`, `dispatch_*.go`) ✓
- [x] `DispatchLevel` enum - current SIMD level
  - [x] `DispatchScalar` - no SIMD
  - [x] `DispatchSSE2` - SSE2 (x86-64 baseline)
  - [x] `DispatchAVX2` - AVX2 (256-bit)
  - [x] `DispatchAVX512` - AVX-512 (512-bit)
  - [x] `DispatchNEON` - ARM NEON (128-bit)
  - [x] `DispatchSVE` - ARM SVE (scalable)
- [x] Runtime detection
  - [x] `CurrentLevel()` - get detected SIMD level
  - [x] `CurrentWidth()` - get SIMD width in bytes
  - [x] `CurrentName()` - get SIMD target name
  - [x] `NoSimdEnv()` - check HWY_NO_SIMD env var
  - [x] `MaxLanes[T]()` - max lanes for type T
- [x] AMD64 CPU detection (`dispatch_amd64.go`)
  - Currently defaults to AVX2 (TODO: use archsimd package in Go 1.26)
- [x] Fallback for other architectures (`dispatch_other.go`)

### Tail Handling (`tail.go`) ✓
- [x] `TailMask[T](count int) Mask[T]` - create mask for tail elements
- [x] `ProcessWithTail[T](size int, fullFn, tailFn func(...))` - process with tail handling
- [x] `ProcessWithTailNoMask[T](size int, fullFn func(...))` - process with overlapping vectors
- [x] `AlignedSize[T](size int) int` - round up to vector width multiple
- [x] `IsAligned[T](size int) bool` - check if size is aligned

### Testing (`ops_test.go`) ✓
- [x] Comprehensive tests for all operations
- [x] Test coverage for dispatch system
- [x] Test coverage for tail handling
- [x] All tests passing ✓

## Usage Example

```go
package main

import (
    "fmt"
    "github.com/ajroetker/go-highway/hwy"
)

func main() {
    // Simple vector addition
    a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
    b := []float32{10, 20, 30, 40, 50, 60, 70, 80}
    result := make([]float32, len(a))

    // Process with automatic tail handling
    hwy.ProcessWithTail[float32](len(a),
        func(offset int) {
            // Full vectors
            va := hwy.Load(a[offset:])
            vb := hwy.Load(b[offset:])
            vr := hwy.Add(va, vb)
            hwy.Store(vr, result[offset:])
        },
        func(offset, count int) {
            // Tail with mask
            mask := hwy.TailMask[float32](count)
            va := hwy.MaskLoad(mask, a[offset:])
            vb := hwy.MaskLoad(mask, b[offset:])
            vr := hwy.Add(va, vb)
            hwy.MaskStore(mask, vr, result[offset:])
        },
    )

    fmt.Println(result) // [11 22 33 44 55 66 77 88]
}
```

## Architecture Support

| Architecture | Status | SIMD Level | Width |
|-------------|--------|------------|-------|
| amd64 | ✓ Base (scalar) | DispatchScalar/AVX2* | 16-32 bytes |
| arm64 | ✓ Base (scalar) | DispatchScalar | 16 bytes |
| other | ✓ Base (scalar) | DispatchScalar | 16 bytes |

\* AMD64 currently defaults to AVX2 but only uses scalar implementation.
SIMD implementations coming in Phase 2.

## Next Steps - Phase 2

1. **AVX2 SIMD Implementation** (`ops_simd_avx2.go`)
   - Replace scalar operations with AVX2 intrinsics
   - Use Go 1.26's archsimd package
   - Build tag: `//go:build amd64 && simd`

2. **NEON SIMD Implementation** (`ops_simd_neon.go`)
   - ARM NEON intrinsics
   - Build tag: `//go:build arm64 && simd`

3. **Advanced Operations**
   - More math functions (exp, log, sin, cos)
   - Bitwise operations
   - Table lookups
   - Gather/scatter operations

4. **Performance Benchmarks**
   - Compare SIMD vs scalar performance
   - Benchmark common operations
   - Optimize hot paths

## Environment Variables

- `HWY_NO_SIMD=1` - Force scalar fallback (useful for testing)

## Build Tags

- `simd` - Enable SIMD implementations (Phase 2+)
- Default: scalar fallback

## Requirements

- Go 1.26+ (for generics and future archsimd package)
- Module: `github.com/ajroetker/go-highway`
