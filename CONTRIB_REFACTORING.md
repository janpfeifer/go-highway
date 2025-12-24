# Contrib Package Refactoring for hwygen Support

## Summary

Refactored the `hwy/contrib` package to properly support hwygen-generated code by exporting archsimd-native functions that can be called directly from generated SIMD code.

## Problem

The original contrib package had functions like:
```go
func exp32AVX2(v hwy.Vec[float32]) hwy.Vec[float32]
```

But hwygen-generated code needs to call functions that work directly with `archsimd` types:
```go
// Generated code needs:
func BaseSigmoid_AVX2(in, out []float32) {
    x := archsimd.LoadFloat32x8Slice(in[ii:])
    negX := x.Neg()
    expNegX := contrib.Exp_AVX2_F32x8(negX)  // Needs this!
    // ...
}
```

## Solution

### 1. Refactored `hwy/contrib/exp_avx2.go`

**Changed:**
- Renamed `exp32x8` → `Exp_AVX2_F32x8` (exported)
- Renamed `exp64x4` → `Exp_AVX2_F64x4` (exported)

**Result:**
- Functions are now callable directly by hwygen-generated code
- Standardized naming: `<Function>_<Target>_<Type>` (e.g., `Exp_AVX2_F32x8`)

### 2. Created AVX2 stub implementations for other contrib functions

**New files:**
- `hwy/contrib/log_avx2.go` - Exports `Log_AVX2_F32x8`, `Log_AVX2_F64x4`
- `hwy/contrib/trig_avx2.go` - Exports `Sin_AVX2_F32x8`, `Cos_AVX2_F32x8`, etc.
- `hwy/contrib/special_avx2.go` - Exports `Tanh_AVX2_F32x8`, `Sigmoid_AVX2_F32x8`, `Erf_AVX2_F32x8`

**Status:**
- Exp: Fully optimized AVX2 implementation
- Log, Sin, Cos, Tanh, Sigmoid, Erf: Stub implementations using scalar fallback (marked with TODO)

### 3. Updated `cmd/hwygen/targets.go`

**Added contrib function mappings to OpMap:**

For AVX2 target:
```go
"Exp":     {Package: "contrib", Name: "Exp_AVX2", IsMethod: false},
"Log":     {Package: "contrib", Name: "Log_AVX2", IsMethod: false},
"Sin":     {Package: "contrib", Name: "Sin_AVX2", IsMethod: false},
"Cos":     {Package: "contrib", Name: "Cos_AVX2", IsMethod: false},
"Tanh":    {Package: "contrib", Name: "Tanh_AVX2", IsMethod: false},
"Sigmoid": {Package: "contrib", Name: "Sigmoid_AVX2", IsMethod: false},
"Erf":     {Package: "contrib", Name: "Erf_AVX2", IsMethod: false},
```

For Fallback target:
```go
"Exp":     {Package: "contrib", Name: "Exp", IsMethod: false},
// ... (uses generic hwy.Vec wrappers)
```

### 4. Updated `cmd/hwygen/transformer.go`

**Changes:**

1. Extended `transformCallExpr` to handle both `hwy.*` and `contrib.*` calls
2. Enhanced `transformToFunction` to:
   - Add type suffix for contrib functions (e.g., `Exp_AVX2_F32x8`)
   - Keep package as `contrib` instead of changing to `archsimd`
   - Handle fallback vs SIMD targets appropriately

**Transformation examples:**

AVX2 target (float32):
```
contrib.Exp(x) → contrib.Exp_AVX2_F32x8(x)
```

AVX2 target (float64):
```
contrib.Exp(x) → contrib.Exp_AVX2_F64x4(x)
```

Fallback target:
```
contrib.Exp(x) → contrib.Exp(x)  // No change, uses hwy.Vec wrapper
```

## File Structure

```
hwy/contrib/
├── exp.go              # Generic Exp[T] wrapper + Exp32/Exp64 vars
├── exp_base.go         # Pure Go: exp32Base, exp64Base
├── exp_avx2.go         # AVX2: Exp_AVX2_F32x8, Exp_AVX2_F64x4 (OPTIMIZED)
├── log.go              # Generic Log[T] wrapper
├── log_base.go         # Pure Go: log32Base, log64Base
├── log_avx2.go         # AVX2: Log_AVX2_F32x8, Log_AVX2_F64x4 (STUB)
├── trig.go             # Generic Sin, Cos, SinCos wrappers
├── trig_base.go        # Pure Go implementations
├── trig_avx2.go        # AVX2: Sin_AVX2_F32x8, Cos_AVX2_F32x8, etc. (STUB)
├── special.go          # Generic Tanh, Sigmoid, Erf wrappers
├── special_base.go     # Pure Go implementations
└── special_avx2.go     # AVX2: Tanh_AVX2_F32x8, Sigmoid_AVX2_F32x8, Erf_AVX2_F32x8 (STUB)
```

## Testing

All tests pass:
```bash
# Contrib tests
go1.26rc1 test ./hwy/contrib
PASS

# All hwy tests
go1.26rc1 test ./hwy/...
PASS

# Build with SIMD support
GOARCH=amd64 GOOS=linux GOEXPERIMENT=simd go1.26rc1 build ./hwy/contrib
# Success!
```

## Usage Example

### Portable code (to be transformed by hwygen):
```go
import (
    "github.com/ajroetker/go-highway/hwy"
    "github.com/ajroetker/go-highway/hwy/contrib"
)

func BaseSigmoid[T hwy.Floats](in, out []T) {
    vOne := hwy.Set(T(1))
    for ii := 0; ii < len(in); ii += vOne.NumLanes() {
        x := hwy.Load(in[ii:])
        negX := hwy.Neg(x)
        expNegX := contrib.Exp(negX)  // Will be transformed!
        y := hwy.Div(vOne, hwy.Add(vOne, expNegX))
        hwy.Store(y, out[ii:])
    }
}
```

### Generated AVX2 code (by hwygen):
```go
func BaseSigmoid_avx2(in []float32, out []float32) {
    vOne := archsimd.BroadcastFloat32x8(1.0)
    for ii := 0; ii < len(in); ii += 8 {
        x := archsimd.LoadFloat32x8Slice(in[ii:])
        negX := x.Neg()
        expNegX := contrib.Exp_AVX2_F32x8(negX)  // Direct call!
        y := vOne.Div(vOne.Add(expNegX))
        y.StoreSlice(out[ii:])
    }
}
```

### Direct usage (without hwygen):
```go
import (
    "simd/archsimd"
    "github.com/ajroetker/go-highway/hwy/contrib"
)

func example() {
    // Load 8 float32 values
    input := archsimd.BroadcastFloat32x8(1.0)

    // Call AVX2-optimized exp directly
    result := contrib.Exp_AVX2_F32x8(input)

    // Store results
    var values [8]float32
    result.StoreSlice(values[:])

    // values[0] ≈ 2.71828 (e)
}
```

## Future Work

### Immediate
- Implement optimized AVX2 versions for Log, Sin, Cos, Tanh, Sigmoid, Erf
- Add AVX512 implementations (`Exp_AVX512_F32x16`, etc.)

### Long-term
- Add more contrib functions (Pow, Atan2, etc.)
- Implement ARM NEON variants
- Benchmark and optimize polynomial coefficients

## Naming Convention

All exported contrib functions follow this pattern:
```
<Function>_<Target>_<Type>

Examples:
- Exp_AVX2_F32x8    - AVX2 exp for 8x float32
- Exp_AVX2_F64x4    - AVX2 exp for 4x float64
- Log_AVX2_F32x8    - AVX2 log for 8x float32
- Sin_AVX512_F32x16 - AVX512 sin for 16x float32 (future)
```

## Compatibility

- ✅ Backward compatible - existing code using `contrib.Exp()` continues to work
- ✅ Runtime dispatch still works - `contrib.Exp32`, `contrib.Exp64` are set at init
- ✅ Generic wrappers still work - `contrib.Exp[T]()` dispatches correctly
- ✅ New: Direct archsimd access - hwygen can call optimized functions directly

## Build Requirements

- Go 1.26rc1 or later (for GOEXPERIMENT=simd)
- Build tags: `amd64 && goexperiment.simd` for AVX2 code
- No build tag for fallback code (always available)

## References

- Original issue: hwygen needs direct access to archsimd-native contrib functions
- Design doc: ARCH_TARGETS.md (if exists)
- Related: Google Highway C++ library patterns
