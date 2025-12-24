# Go-Highway Implementation Plan

A portable SIMD abstraction library for Go, inspired by Google's C++ Highway library.

## Overview

**Repository**: `github.com/ajroetker/go-highway` (new standalone repo)
**Target**: Go 1.26+ with `GOEXPERIMENT=simd`
**Initial Architecture**: AMD64 AVX2 + pure-Go fallback

## Package Structure

```
github.com/ajroetker/go-highway/
├── hwy/                    # Core portable API
│   ├── types.go            # Vec[T], Lanes, Floats constraints
│   ├── tags.go             # ScalableTag, FixedTag128/256/512
│   ├── ops.go              # Load, Store, Add, Mul, etc. (interface)
│   ├── ops_base.go         # Pure-Go implementations (//go:build !simd)
│   ├── dispatch.go         # Runtime dispatch infrastructure
│   ├── dispatch_amd64.go   # CPU feature detection
│   └── tail.go             # Tail handling utilities
│
├── hwy/contrib/            # Extended math operations
│   ├── exp.go              # Exp function interface
│   ├── exp_base.go         # Pure-Go exp
│   ├── exp_avx2.go         # AVX2 vectorized exp
│   ├── log.go, trig.go     # Other transcendentals
│
├── cmd/hwygen/             # Code generator tool
│   ├── main.go             # CLI entry point
│   ├── parser.go           # Go AST parsing
│   ├── transformer.go      # AST → target-specific code
│   └── emitter.go          # File generation
│
├── internal/
│   ├── archinfo/           # CPU feature detection
│   └── testutil/           # ULP comparison, test generators
│
└── examples/
    └── sigmoid/            # Complete example with go:generate
```

## Implementation Steps

### Phase 1: Core Types and Base Operations

1. **Create `hwy/types.go`**
   - `type Floats interface { ~float32 | ~float64 }`
   - `type Lanes interface { Floats | Integers }`
   - `type Vec[T Lanes] struct { data []T; count int }`

2. **Create `hwy/tags.go`**
   - `type Tag interface { Width() int; Name() string }`
   - `ScalableTag` - adapts to widest available SIMD
   - `FixedTag128/256/512` for explicit width control

3. **Create `hwy/ops.go` and `hwy/ops_base.go`**
   - Core operations: `Load`, `Store`, `Set`, `Zero`
   - Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`
   - Math: `Sqrt`, `FMA`, `Min`, `Max`
   - Reduction: `ReduceSum`, `ReduceMin`, `ReduceMax`
   - Comparison: `Equal`, `LessThan`, `GreaterThan` → `Mask[T]`
   - Masked: `IfThenElse`, `MaskLoad`, `MaskStore`

4. **Create `hwy/dispatch.go` + `hwy/dispatch_amd64.go`**
   - `DispatchLevel` enum: Scalar, SSE2, AVX2, AVX512
   - `CurrentLevel()`, `CurrentWidth()` functions
   - `init()` detection using `archsimd.HasAVX2()` etc.
   - `HWY_NO_SIMD` environment variable support

### Phase 2: Code Generator (hwygen)

5. **Create `cmd/hwygen/parser.go`**
   - Parse Go AST to find functions using `hwy.*` calls
   - Extract: function signature, type parameters, loop structure
   - Identify all `hwy.Op()` and `contrib.Op()` calls

6. **Create `cmd/hwygen/transformer.go`**
   - Define `Target` struct with TypeMap and OpMap
   - AVX2Target: `Vec[float32]` → `archsimd.Float32x8`
   - Transform `hwy.Add(a, b)` → `a.Add(b)` (method call)
   - Transform `contrib.Exp(x)` → `contrib.Exp_AVX2(x)`
   - Handle loop stride: `vOne.NumElements()` → `8` (constant)
   - Insert scalar tail handling

7. **Create `cmd/hwygen/emitter.go`**
   - Generate dispatcher file with function pointer tables
   - Generate target-specific files with build tags
   - Use `go/format` for clean output

8. **Dispatcher Template**
   ```go
   var Sigmoid func(in, out []float32)

   func init() {
       if os.Getenv("HWY_NO_SIMD") != "" { initFallback(); return }
       if archsimd.HasAVX2() { initAVX2(); return }
       initFallback()
   }
   ```

### Phase 3: Contrib Package

9. **Create `hwy/contrib/exp.go` + `exp_base.go`**
   - `var Exp func(v hwy.Vec[float32]) hwy.Vec[float32]`
   - Pure-Go: loop calling `math.Exp()`
   - Pattern for all transcendentals

10. **Create `hwy/contrib/exp_avx2.go`** (build tagged)
    - Range reduction: `x = k*ln(2) + r`
    - Polynomial approximation for `e^r`
    - Reconstruction: `2^k * e^r`
    - Uses `archsimd.Float32x8` directly

### Phase 4: Tail Handling

11. **Create `hwy/tail.go`**
    - `TailMask[T](count int) Mask[T]`
    - `ProcessWithTail[T](size, fullFn, tailFn)`

12. **Generator tail strategy**
    - Full vector loop: `for ; ii+lanes <= size; ii += lanes`
    - Scalar tail: `for ; ii < size; ii++`
    - Optionally: masked operations for AVX-512

### Phase 5: Testing

13. **Unit tests** (`hwy/ops_test.go`)
    - Table-driven tests for each operation
    - Edge cases: empty, partial, oversized slices
    - Special values: 0, ±Inf, NaN

14. **Accuracy tests** (`hwy/contrib/exp_accuracy_test.go`)
    - ULP-based comparison with `math.Exp`
    - Max 4 ULP tolerance
    - Random input fuzzing

15. **Cross-implementation tests**
    - Verify AVX2 matches fallback results
    - Run same test with `HWY_NO_SIMD=1`

16. **Benchmarks** (`hwy/*_bench_test.go`)
    - Compare SIMD vs scalar
    - Various array sizes

### Phase 6: Example and Documentation

17. **Create `examples/sigmoid/`**
    - `sigmoid.go` with `//go:generate hwygen ...`
    - Run `go generate` to produce variants
    - `main.go` demonstrating usage

18. **README.md**
    - Quick start
    - User group workflows (A/B/C/D)
    - Build instructions

## Key Design Decisions

| Decision | Choice |
|----------|--------|
| Vec abstraction | `Vec[T Lanes]` with internal slice for base, mapped to `archsimd.*` in generated code |
| Operation style | Functions `hwy.Add(a, b)` → methods `a.Add(b)` in generated code |
| Dispatch | Function pointers set in `init()`, zero per-call overhead |
| Tail handling | Scalar fallback by default (simple, portable) |
| Build tags | `//go:build amd64 && simd` for SIMD, none for fallback |
| Contrib pattern | Multi-file: `exp.go` (interface), `exp_base.go` (fallback), `exp_avx2.go` (SIMD) |

## Type Mapping Table

| Generic Type | AVX2 (256-bit) | AVX-512 (512-bit) | Fallback |
|--------------|----------------|-------------------|----------|
| `Vec[float32]` | `Float32x8` | `Float32x16` | `[]float32` |
| `Vec[float64]` | `Float64x4` | `Float64x8` | `[]float64` |
| `Vec[int32]` | `Int32x8` | `Int32x16` | `[]int32` |

## Dependencies

- `simd/archsimd` (Go 1.26+ experimental)
- `golang.org/x/tools/go/ast/astutil` (for AST manipulation)

## Build & Test Commands

```bash
# Build with SIMD
GOEXPERIMENT=simd go build ./...

# Build fallback only
go build ./...

# Test both paths
GOEXPERIMENT=simd go test ./...
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test ./...

# Generate code
go generate ./...

# Benchmark comparison
GOEXPERIMENT=simd go test -bench=. ./hwy/...
```

## Critical Files to Create First

1. `hwy/types.go` - Foundation types
2. `hwy/ops.go` + `hwy/ops_base.go` - Core API
3. `cmd/hwygen/transformer.go` - Code generation logic
4. `hwy/dispatch.go` - Runtime dispatch
5. `examples/sigmoid/sigmoid.go` - End-to-end validation
