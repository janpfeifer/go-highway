// Package matmul provides high-performance matrix multiplication operations
// using SIMD instructions.
//
// On ARM64 with SME (Apple M4+), this package uses FMOPA (floating-point
// outer product and accumulate) instructions which compute O(N²) results
// per instruction, achieving significant speedups over traditional SIMD
// approaches.
//
// Example usage:
//
//	// C = A * B where A is MxK, B is KxN, C is MxN
//	a := make([]float32, M*K)  // row-major
//	b := make([]float32, K*N)  // row-major
//	c := make([]float32, M*N)  // output, row-major
//
//	matmul.MatMul(a, b, c, M, N, K)
//
// The implementation automatically selects the best path:
//   - SME (FMOPA) on Apple M4+
//   - NEON on other ARM64
//   - Scalar fallback elsewhere
package matmul
