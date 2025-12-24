# AVX512 Support

Add AVX512 implementations for math functions in `hwy/contrib`.

## Prerequisites

- Complete AVX2 implementations first (see avx2-math-functions.md)
- AVX512 uses same algorithms, just wider vectors (16x float32, 8x float64)

## New Files

| File | Functions |
|------|-----------|
| `hwy/contrib/exp_avx512.go` | Exp_AVX512_F32x16, Exp_AVX512_F64x8 |
| `hwy/contrib/log_avx512.go` | Log_AVX512_F32x16, Log_AVX512_F64x8 |
| `hwy/contrib/trig_avx512.go` | Sin/Cos/SinCos for F32x16 and F64x8 |
| `hwy/contrib/special_avx512.go` | Tanh/Sigmoid/Erf for F32x16 and F64x8 |

## Functions (16 total)

| Function | Notes |
|----------|-------|
| Exp_AVX512_F32x16 | Port from Exp_AVX2_F32x8 |
| Exp_AVX512_F64x8 | Port from Exp_AVX2_F64x4 |
| Log_AVX512_F32x16 | Port from Log_AVX2_F32x8 |
| Log_AVX512_F64x8 | Port from Log_AVX2_F64x4 |
| Sin_AVX512_F32x16 | Port from Sin_AVX2_F32x8 |
| Sin_AVX512_F64x8 | Port from Sin_AVX2_F64x4 |
| Cos_AVX512_F32x16 | Port from Cos_AVX2_F32x8 |
| Cos_AVX512_F64x8 | Port from Cos_AVX2_F64x4 |
| SinCos_AVX512_F32x16 | Port from SinCos_AVX2_F32x8 |
| SinCos_AVX512_F64x8 | Port from SinCos_AVX2_F64x4 |
| Tanh_AVX512_F32x16 | Port from Tanh_AVX2_F32x8 |
| Tanh_AVX512_F64x8 | Port from Tanh_AVX2_F64x4 |
| Sigmoid_AVX512_F32x16 | Port from Sigmoid_AVX2_F32x8 |
| Sigmoid_AVX512_F64x8 | Port from Sigmoid_AVX2_F64x4 |
| Erf_AVX512_F32x16 | Port from Erf_AVX2_F32x8 |
| Erf_AVX512_F64x8 | Port from Erf_AVX2_F64x4 |

## Infrastructure Changes

### cmd/hwygen/targets.go

Add contrib ops to AVX512Target OpMap:

```go
"Exp":     {Package: "contrib", Name: "Exp_AVX512", IsMethod: false},
"Log":     {Package: "contrib", Name: "Log_AVX512", IsMethod: false},
"Sin":     {Package: "contrib", Name: "Sin_AVX512", IsMethod: false},
"Cos":     {Package: "contrib", Name: "Cos_AVX512", IsMethod: false},
"Tanh":    {Package: "contrib", Name: "Tanh_AVX512", IsMethod: false},
"Sigmoid": {Package: "contrib", Name: "Sigmoid_AVX512", IsMethod: false},
"Erf":     {Package: "contrib", Name: "Erf_AVX512", IsMethod: false},
```

### init() Registration

Each AVX512 file needs init() to register when AVX512 is available:

```go
func init() {
    if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
        Exp32 = exp32AVX512
        Exp64 = exp64AVX512
    }
}
```

## Build Tags

All AVX512 files use:

```go
//go:build amd64 && goexperiment.simd
```

## Testing

Extend `hwy/contrib/contrib_test.go` to test AVX512 variants with same test cases as AVX2.
