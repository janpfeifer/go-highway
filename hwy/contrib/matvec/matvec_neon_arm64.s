//go:build !noasm && arm64

#include "textflag.h"

// func matvec_neon_f32(m, v, result unsafe.Pointer, rows, cols int64)
//
// Computes result = M * v where:
//   M is rows x cols (row-major)
//   v is cols (input vector)
//   result is rows (output vector)
//
// Uses ARM NEON FMLA for vectorized multiply-add.
// Processes 4 floats at a time using 128-bit NEON vectors.
// Uses 4 accumulators for better instruction-level parallelism.
//
// Register usage:
//   R19: M matrix base
//   R20: v vector base
//   R21: result vector base
//   R22: rows dimension
//   R23: cols dimension
//   R24: cols * 4 (row stride in bytes)
//   R0: i (row index)
//   V0-V3: accumulators (4 lanes each = 16 parallel accumulations)
//
TEXT ·matvec_neon_f32(SB), NOSPLIT, $48-40
	// Save callee-saved registers
	MOVD	R19, 0(RSP)
	MOVD	R20, 8(RSP)
	MOVD	R21, 16(RSP)
	MOVD	R22, 24(RSP)
	MOVD	R23, 32(RSP)
	MOVD	R24, 40(RSP)

	// Load arguments
	MOVD	m+0(FP), R19       // M matrix base
	MOVD	v+8(FP), R20       // v vector base
	MOVD	result+16(FP), R21 // result vector base
	MOVD	rows+24(FP), R22   // rows dimension
	MOVD	cols+32(FP), R23   // cols dimension

	// Calculate stride (in bytes)
	LSL	$2, R23, R24       // R24 = cols * 4

	// Calculate vector elements (cols rounded down to multiple of 16)
	AND	$~15, R23, R25     // R25 = cols & ~15 (unrolled portion, 16 at a time)
	AND	$~3, R23, R26      // R26 = cols & ~3 (vector portion, 4 at a time)

	// Outer loop: for each row i
	MOVD	$0, R0

row_loop:
	CMP	R22, R0
	BGE	done

	// Calculate M row pointer: &M[i, 0] = M + i * cols * 4
	MUL	R24, R0, R1
	ADD	R19, R1, R1        // R1 = &M[i, 0]

	// Zero accumulators V0-V3
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

	// Process 16 elements at a time (4 accumulators x 4 lanes)
	MOVD	R1, R2             // R2 = M row ptr
	MOVD	R20, R3            // R3 = v ptr
	MOVD	R25, R4            // R4 = unrolled elements remaining

unrolled_loop:
	CBZ	R4, vector_loop_init

	// Load 16 elements from M row
	VLD1.P	64(R2), [V16.S4, V17.S4, V18.S4, V19.S4]

	// Load 16 elements from v
	VLD1.P	64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]

	// Multiply-accumulate: Vn += Vm * Vv
	VFMLA	V16.S4, V20.S4, V0.S4
	VFMLA	V17.S4, V21.S4, V1.S4
	VFMLA	V18.S4, V22.S4, V2.S4
	VFMLA	V19.S4, V23.S4, V3.S4

	SUB	$16, R4, R4
	B	unrolled_loop

vector_loop_init:
	// Combine accumulators: V0 = V0 + V1 + V2 + V3
	WORD	$0x4e21d400        // fadd v0.4s, v0.4s, v1.4s
	WORD	$0x4e23d442        // fadd v2.4s, v2.4s, v3.4s
	WORD	$0x4e22d400        // fadd v0.4s, v0.4s, v2.4s

	// Process remaining 4-element chunks
	LSL	$2, R25, R5        // R5 = unrolled_elements * 4 (bytes)
	ADD	R1, R5, R2         // R2 = &M[i, unrolled_elements]
	ADD	R20, R5, R3        // R3 = &v[unrolled_elements]
	SUB	R25, R26, R4       // R4 = vector_elements - unrolled_elements

vector_loop:
	CBZ	R4, reduce

	// Load 4 elements from M row
	VLD1.P	16(R2), [V4.S4]

	// Load 4 elements from v
	VLD1.P	16(R3), [V5.S4]

	// Multiply-accumulate
	VFMLA	V4.S4, V5.S4, V0.S4

	SUB	$4, R4, R4
	B	vector_loop

reduce:
	// Horizontal sum: V0.S4 -> scalar
	// V0 = [a, b, c, d]
	// Step 1: add pairs -> [a+c, b+d, ?, ?]
	WORD	$0x6e20d400        // faddp v0.4s, v0.4s, v0.4s
	// Step 2: add final pair -> [a+b+c+d, ?, ?, ?]
	WORD	$0x7e30d800        // faddp s0, v0.2s

	// V0.S[0] now contains the horizontal sum

	// Process scalar tail (cols % 4 remaining elements)
	AND	$3, R23, R4        // R4 = cols % 4
	CBZ	R4, store_result

	// Calculate tail pointers
	LSL	$2, R26, R5        // R5 = vector_elements * 4
	ADD	R1, R5, R2         // R2 = &M[i, vector_elements]
	ADD	R20, R5, R3        // R3 = &v[vector_elements]

scalar_tail:
	CBZ	R4, store_result

	FMOVS	(R2), F1           // M[i, j]
	FMOVS	(R3), F2           // v[j]
	FMADDS	F1, F2, F0, F0     // sum += M * v

	ADD	$4, R2, R2
	ADD	$4, R3, R3
	SUB	$1, R4, R4
	B	scalar_tail

store_result:
	// Calculate result pointer: &result[i]
	LSL	$2, R0, R5
	ADD	R21, R5, R5

	// Store result
	FMOVS	F0, (R5)

	ADD	$1, R0, R0
	B	row_loop

done:
	// Restore callee-saved registers
	MOVD	0(RSP), R19
	MOVD	8(RSP), R20
	MOVD	16(RSP), R21
	MOVD	24(RSP), R22
	MOVD	32(RSP), R23
	MOVD	40(RSP), R24

	RET


// func matvec_neon_f64(m, v, result unsafe.Pointer, rows, cols int64)
//
// Float64 version - processes 2 elements at a time using 128-bit NEON.
// Uses 4 accumulators for better ILP.
//
TEXT ·matvec_neon_f64(SB), NOSPLIT, $48-40
	// Save callee-saved registers
	MOVD	R19, 0(RSP)
	MOVD	R20, 8(RSP)
	MOVD	R21, 16(RSP)
	MOVD	R22, 24(RSP)
	MOVD	R23, 32(RSP)
	MOVD	R24, 40(RSP)

	// Load arguments
	MOVD	m+0(FP), R19       // M matrix base
	MOVD	v+8(FP), R20       // v vector base
	MOVD	result+16(FP), R21 // result vector base
	MOVD	rows+24(FP), R22   // rows dimension
	MOVD	cols+32(FP), R23   // cols dimension

	// Calculate stride (in bytes)
	LSL	$3, R23, R24       // R24 = cols * 8

	// Calculate vector elements (cols rounded down to multiple of 8)
	AND	$~7, R23, R25      // R25 = cols & ~7 (unrolled portion, 8 at a time)
	AND	$~1, R23, R26      // R26 = cols & ~1 (vector portion, 2 at a time)

	// Outer loop: for each row i
	MOVD	$0, R0

row_loop_f64:
	CMP	R22, R0
	BGE	done_f64

	// Calculate M row pointer: &M[i, 0] = M + i * cols * 8
	MUL	R24, R0, R1
	ADD	R19, R1, R1        // R1 = &M[i, 0]

	// Zero accumulators V0-V3
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

	// Process 8 elements at a time (4 accumulators x 2 lanes)
	MOVD	R1, R2             // R2 = M row ptr
	MOVD	R20, R3            // R3 = v ptr
	MOVD	R25, R4            // R4 = unrolled elements remaining

unrolled_loop_f64:
	CBZ	R4, vector_loop_init_f64

	// Load 8 elements from M row (4 x 2 doubles)
	VLD1.P	64(R2), [V16.D2, V17.D2, V18.D2, V19.D2]

	// Load 8 elements from v
	VLD1.P	64(R3), [V20.D2, V21.D2, V22.D2, V23.D2]

	// Multiply-accumulate
	VFMLA	V16.D2, V20.D2, V0.D2
	VFMLA	V17.D2, V21.D2, V1.D2
	VFMLA	V18.D2, V22.D2, V2.D2
	VFMLA	V19.D2, V23.D2, V3.D2

	SUB	$8, R4, R4
	B	unrolled_loop_f64

vector_loop_init_f64:
	// Combine accumulators
	WORD	$0x4e61d400        // fadd v0.2d, v0.2d, v1.2d
	WORD	$0x4e63d442        // fadd v2.2d, v2.2d, v3.2d
	WORD	$0x4e62d400        // fadd v0.2d, v0.2d, v2.2d

	// Process remaining 2-element chunks
	LSL	$3, R25, R5        // R5 = unrolled_elements * 8 (bytes)
	ADD	R1, R5, R2         // R2 = &M[i, unrolled_elements]
	ADD	R20, R5, R3        // R3 = &v[unrolled_elements]
	SUB	R25, R26, R4       // R4 = vector_elements - unrolled_elements

vector_loop_f64:
	CBZ	R4, reduce_f64

	// Load 2 elements from M row
	VLD1.P	16(R2), [V4.D2]

	// Load 2 elements from v
	VLD1.P	16(R3), [V5.D2]

	// Multiply-accumulate
	VFMLA	V4.D2, V5.D2, V0.D2

	SUB	$2, R4, R4
	B	vector_loop_f64

reduce_f64:
	// Horizontal sum: V0.D2 -> scalar
	// V0 = [a, b]
	WORD	$0x7e70d800        // faddp d0, v0.2d
	// V0.D[0] now contains a + b

	// Process scalar tail (cols % 2 remaining)
	AND	$1, R23, R4        // R4 = cols % 2
	CBZ	R4, store_result_f64

	// Calculate tail pointer
	LSL	$3, R26, R5        // R5 = vector_elements * 8
	ADD	R1, R5, R2         // R2 = &M[i, vector_elements]
	ADD	R20, R5, R3        // R3 = &v[vector_elements]

	FMOVD	(R2), F1           // M[i, j]
	FMOVD	(R3), F2           // v[j]
	FMADDD	F1, F2, F0, F0     // sum += M * v

store_result_f64:
	// Calculate result pointer: &result[i]
	LSL	$3, R0, R5
	ADD	R21, R5, R5

	// Store result
	FMOVD	F0, (R5)

	ADD	$1, R0, R0
	B	row_loop_f64

done_f64:
	// Restore callee-saved registers
	MOVD	0(RSP), R19
	MOVD	8(RSP), R20
	MOVD	16(RSP), R21
	MOVD	24(RSP), R22
	MOVD	32(RSP), R23
	MOVD	40(RSP), R24

	RET
