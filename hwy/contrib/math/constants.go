package math

// =============================================================================
// Constants for mathematical functions
// =============================================================================

// Float32 constants for Exp
var (
	expLn2Hi_f32  float32 = 0.693359375
	expLn2Lo_f32  float32 = -2.12194440e-4
	expInvLn2_f32 float32 = 1.44269504088896341

	expOverflow_f32  float32 = 88.72283905206835
	expUnderflow_f32 float32 = -87.33654475055310

	expC1_f32 float32 = 1.0
	expC2_f32 float32 = 0.5
	expC3_f32 float32 = 0.16666666666666666
	expC4_f32 float32 = 0.041666666666666664
	expC5_f32 float32 = 0.008333333333333333
	expC6_f32 float32 = 0.001388888888888889

	expOne_f32  float32 = 1.0
	expZero_f32 float32 = 0.0
)

// Float64 constants for Exp
var (
	expLn2Hi_f64  float64 = 0.6931471803691238
	expLn2Lo_f64  float64 = 1.9082149292705877e-10
	expInvLn2_f64 float64 = 1.4426950408889634

	expOverflow_f64  float64 = 709.782712893384
	expUnderflow_f64 float64 = -708.3964185322641

	expC1_f64  float64 = 1.0
	expC2_f64  float64 = 0.5
	expC3_f64  float64 = 0.16666666666666666
	expC4_f64  float64 = 0.041666666666666664
	expC5_f64  float64 = 0.008333333333333333
	expC6_f64  float64 = 0.001388888888888889
	expC7_f64  float64 = 0.0001984126984126984
	expC8_f64  float64 = 2.48015873015873e-05
	expC9_f64  float64 = 2.7557319223985893e-06
	expC10_f64 float64 = 2.755731922398589e-07

	expOne_f64  float64 = 1.0
	expZero_f64 float64 = 0.0
)

// Float32 constants for Log
var (
	logC1_f32 float32 = 1.0
	logC2_f32 float32 = 0.3333333333333367565
	logC3_f32 float32 = 0.1999999999970470954
	logC4_f32 float32 = 0.1428571437183119574
	logC5_f32 float32 = 0.1111109921607489198

	logLn2Hi_f32 float32 = 0.693359375
	logLn2Lo_f32 float32 = -2.12194440e-4
	logOne_f32   float32 = 1.0
	logTwo_f32   float32 = 2.0
)

// Float64 constants for Log
var (
	logC1_f64 float64 = 1.0
	logC2_f64 float64 = 0.3333333333333367565
	logC3_f64 float64 = 0.1999999999970470954
	logC4_f64 float64 = 0.1428571437183119574
	logC5_f64 float64 = 0.1111109921607489198
	logC6_f64 float64 = 0.0909178608080902506
	logC7_f64 float64 = 0.0765691884960468666

	logLn2Hi_f64 float64 = 0.6931471803691238
	logLn2Lo_f64 float64 = 1.9082149292705877e-10
	logOne_f64   float64 = 1.0
	logTwo_f64   float64 = 2.0
)

// Float32 constants for Trig (Sin, Cos)
var (
	trig2OverPi_f32   float32 = 0.6366197723675814
	trigPiOver2Hi_f32 float32 = 1.5707963267948966
	trigPiOver2Lo_f32 float32 = 6.123233995736766e-17

	trigS1_f32 float32 = -0.16666666641626524
	trigS2_f32 float32 = 0.008333329385889463
	trigS3_f32 float32 = -0.00019839334836096632
	trigS4_f32 float32 = 2.718311493989822e-6

	trigC1_f32 float32 = -0.4999999963229337
	trigC2_f32 float32 = 0.04166662453689337
	trigC3_f32 float32 = -0.001388731625493765
	trigC4_f32 float32 = 2.443315711809948e-5

	trigOne_f32 float32 = 1.0
)

// Float64 constants for Trig
var (
	trig2OverPi_f64   float64 = 0.6366197723675814
	trigPiOver2Hi_f64 float64 = 1.5707963267948966192313216916398
	trigPiOver2Lo_f64 float64 = 6.123233995736766035868820147292e-17

	trigS1_f64 float64 = -0.16666666666666632
	trigS2_f64 float64 = 0.008333333333332249
	trigS3_f64 float64 = -0.00019841269840885721
	trigS4_f64 float64 = 2.7557316103728803e-6

	trigC1_f64 float64 = -0.5
	trigC2_f64 float64 = 0.04166666666666621
	trigC3_f64 float64 = -0.001388888888887411
	trigC4_f64 float64 = 2.4801587288851704e-5

	trigOne_f64 float64 = 1.0
)

// Float32 constants for Tanh
var (
	tanhClamp_f32  float32 = 9.0
	tanhOne_f32    float32 = 1.0
	tanhNegOne_f32 float32 = -1.0
)

// Float64 constants for Tanh
var (
	tanhClamp_f64  float64 = 19.0
	tanhOne_f64    float64 = 1.0
	tanhNegOne_f64 float64 = -1.0
)

// Float32 constants for Sigmoid
var (
	sigmoidOne_f32 float32 = 1.0
)

// Float64 constants for Sigmoid
var (
	sigmoidOne_f64 float64 = 1.0
)

// Float32 constants for Sinh
var (
	sinhOne_f32 float32 = 1.0
	sinhC3_f32  float32 = 0.16666666666666666
	sinhC5_f32  float32 = 0.008333333333333333
	sinhC7_f32  float32 = 0.0001984126984126984
)

// Float64 constants for Sinh
var (
	sinhOne_f64 float64 = 1.0
	sinhC3_f64  float64 = 0.16666666666666666
	sinhC5_f64  float64 = 0.008333333333333333
	sinhC7_f64  float64 = 0.0001984126984126984
)

// Float32 constants for Erf
var (
	erfA1_f32   float32 = 0.254829592
	erfA2_f32   float32 = -0.284496736
	erfA3_f32   float32 = 1.421413741
	erfA4_f32   float32 = -1.453152027
	erfA5_f32   float32 = 1.061405429
	erfP_f32    float32 = 0.3275911
	erfOne_f32  float32 = 1.0
	erfZero_f32 float32 = 0.0
)

// Float64 constants for Erf
var (
	erfA1_f64   float64 = 0.254829592
	erfA2_f64   float64 = -0.284496736
	erfA3_f64   float64 = 1.421413741
	erfA4_f64   float64 = -1.453152027
	erfA5_f64   float64 = 1.061405429
	erfP_f64    float64 = 0.3275911
	erfOne_f64  float64 = 1.0
	erfZero_f64 float64 = 0.0
)

// Float32 constants for Log2, Log10, Exp2
var (
	log2E_f32  float32 = 1.4426950408889634
	log10E_f32 float32 = 0.4342944819032518
	ln2_f32    float32 = 0.6931471805599453
)

// Float64 constants for Log2, Log10, Exp2
var (
	log2E_f64  float64 = 1.4426950408889634
	log10E_f64 float64 = 0.4342944819032518
	ln2_f64    float64 = 0.6931471805599453
)
