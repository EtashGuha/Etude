/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

 
 /**
  * Original code is under
  * ${basedir}/ExtraComponents/CUDA_SDK/3.0/Linux-x86-64/C/src/BlackScholes
 **/
 
#ifdef CUDALINK_USING_DOUBLE_PRECISIONQ
#define _exp(x)			exp(static_cast<Real_t>(x))
#define _abs(x)			abs(static_cast<Real_t>(x))
#define _log(x)			log(static_cast<Real_t>(x))
#define _log10(x)		log10(static_cast<Real_t>(x))
#define _sqrt(x)		sqrt(static_cast<Real_t>(x))
#define _fmin(x)		fmin(static_cast<Real_t>(x), static_cast<Real_t>(x))
#else /* USING_DOUBLE_PRECISIONQ */
#define _exp(x)			expf(static_cast<Real_t>(x))
#define _abs(x)			fabs(static_cast<Real_t>(x))
#define _log(x)			logf(static_cast<Real_t>(x))
#define _log10(x)		log10f(static_cast<Real_t>(x))
#define _sqrt(x)		sqrtf(static_cast<Real_t>(x))
#define _fmin(x)		fminf(static_cast<Real_t>(x), static_cast<Real_t>(x))
#endif /* USING_DOUBLE_PRECISIONQ */

// Approximate cumulative normal distribution function with a polynomial
__device__ inline Real_t cndGPU(Real_t d) {
	const Real_t A1 = static_cast<Real_t>(0.31938153);
	const Real_t A2 = static_cast<Real_t>(-0.356563782);
	const Real_t A3 = static_cast<Real_t>(1.781477937);
	const Real_t A4 = static_cast<Real_t>(-1.821255978);
	const Real_t A5 = static_cast<Real_t>(1.330274429);
	const Real_t RSQRT2PI = static_cast<Real_t>(0.39894228040143267793994605993438);


	Real_t K = static_cast<Real_t>(1.0 / (1.0 + 0.2316419 * _abs(d)));
	Real_t cnd = RSQRT2PI * _exp(-static_cast<Real_t>(0.5) * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if(d > static_cast<Real_t>(0.0))
		cnd = static_cast<Real_t>(1.0) - cnd;

	return cnd;
}

//Computes CallResult and PutResult
__device__ inline void BlackScholesBody(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t V) {
	Real_t sqrtT, expRT;
	Real_t d1, d2, CNDD1, CNDD2;

	sqrtT = _sqrt(T);
	d1 = (_log(S / X) + (R + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;

	CNDD1 = cndGPU(d1);
	CNDD2 = cndGPU(d2);

	expRT = _exp(-R * T);
	CallResult = S * CNDD1 - X * expRT * CNDD2;
	PutResult = X * expRT * (static_cast<Real_t>(1.0) - CNDD2) - S * (static_cast<Real_t>(1.0) - CNDD1);
}

__global__ void BlackScholes(Real_t *d_CallResult, Real_t *d_PutResult, Real_t *d_StockPrice, Real_t *d_OptionStrike, Real_t *d_OptionYears, Real_t RiskFree, Real_t Volatility, mint optN) {

	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optN; opt += THREAD_N) {
		BlackScholesBody(d_CallResult[opt], d_PutResult[opt], d_StockPrice[opt], d_OptionStrike[opt], d_OptionYears[opt], RiskFree, Volatility);
	}
}