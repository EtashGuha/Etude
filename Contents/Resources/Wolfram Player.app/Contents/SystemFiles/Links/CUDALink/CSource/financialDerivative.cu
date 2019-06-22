/*********************************************************************//**
* @file
*
* @section LICENCE
*
*              Mathematica source file
*
*  Copyright 1986 through 2010 by Wolfram Research Inc.
*
* @section DESCRIPTION
*
*
*
* $Id$
************************************************************************/

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

#include	<wgl.h>
#include	<wgl_cuda_runtime.h>

#include	<iostream>
#include	<assert.h>

using namespace std;


#ifndef __func__
#	if defined(__FUNCTION__)
#		define __func__ __FUNCTION__
#	elif defined(__PRETTY_FUNCTION__)
#		define __func__ __PRETTY_FUNCTION__
#	else
#		define __func__ __FILE__
#	endif
#else
#	define __func__ "unknown"
#endif


#ifdef DEBUG
#if PRINT_DEBUG_LINE_NUMBERSQ
#define PRINT_DBG_LINENO                                \
std::cout << "--- On line "<< __LINE__ <<               \
" in " << __FILE__ << " ---" << std::endl
#define PRINT_DBG_END                                   \
std::cout << std::endl << "----" << std::endl
#else
#define PRINT_DBG_LINENO
#define PRINT_DBG_END                                   \
std::cout << std::endl
#endif /* PRINT_DEBUG_LINE_NUMBERSQ */

#define DEBUG_MSG(...)                                  \
PRINT_DBG_LINENO;                                       \
std::cout << "===  " << __VA_ARGS__;                    \
std::cout << "  ===";									\
PRINT_DBG_END
#else
#define DEBUG_MSG(...)
#endif

#ifndef New
#define New(to, type, n)					to = (type *) wglData->alloc(n); assert(to != NULL)
#endif /* New */

#ifndef Free
#define Free(ptr)							wglData->free(ptr); ptr = NULL
#endif /* New */

#ifdef CONFIG_USE_DOUBLE_PRECISION
#define Real_t								double
#define WGL_Real_t							WGL_Type_Double
#define CUDA_Runtime_getDeviceMemoryAsReal	CUDA_Runtime_getDeviceMemoryAsDouble
#else
#define Real_t								float
#define WGL_Real_t							WGL_Type_Float
#define CUDA_Runtime_getDeviceMemoryAsReal	CUDA_Runtime_getDeviceMemoryAsFloat
#endif

#define wglState							(wglData->state)
#define wglErr								(wglData->getError(wglData))
#define WGL_SuccessQ						(wglErr->code == WGL_Success)
#define WGL_FailQ							(!WGL_SuccessQ)
#define WGL_Type_RealQ(mem)					((mem)->type == WGL_Real_t)


#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (WGL_FailQ) { goto jmp; }

#if CONFIG_USE_DOUBLE_PRECISION
#define _exp(x)			exp(static_cast<Real_t>(x))
#define _abs(x)			abs(static_cast<Real_t>(x))
#define _log(x)			log(static_cast<Real_t>(x))
#define _log10(x)		log10(static_cast<Real_t>(x))
#define _sqrt(x)		sqrt(static_cast<Real_t>(x))
#define _fmin(x)		fmin(static_cast<Real_t>(x), static_cast<Real_t>(x))
#else /* CONFIG_USE_DOUBLE_PRECISION */
#define _exp(x)			expf(static_cast<Real_t>(x))
#define _abs(x)			fabs(static_cast<Real_t>(x))
#define _log(x)			logf(static_cast<Real_t>(x))
#define _log10(x)		log10f(static_cast<Real_t>(x))
#define _sqrt(x)		sqrtf(static_cast<Real_t>(x))
#define _fmin(x)		fminf(static_cast<Real_t>(x), static_cast<Real_t>(x))
#endif /* CONFIG_USE_DOUBLE_PRECISION */


/****************************************************/
/* Black Scholes / Analytic Options Pricing			*/
/****************************************************/

 /**
  * Original code is under
  * ${basedir}/ExtraComponents/CUDA_SDK/3.0/Linux-x86-64/C/src/BlackScholes
 **/

// toCalculate Defines:
#define VALUE				0  //*
#define DELTA				1  //*
#define VEGA				2  //*
#define THETA				3  //*
#define RHO					4  //*
#define GAMMA				5  //* These are the values calculated by FinancialDerivative, so highest priority.
#define VANNA				6  //
#define CHARM				7  //
#define VOMMA				8  //
#define DVEGADTIME			9  //
#define SPEED				10 //
#define ZOMMA				11 // Everything with a comment after is supported thus far
#define COLOR				12 //

// OptionType defines

#define EUROPEAN						100
#define AMERICAN						101
#define ASIAN							102

#define BARRIERUPIN						103
#define BARRIERDOWNIN					104
#define BARRIERUPOUT					105
#define BARRIERDOWNOUT					106

#define LOOKBACKFIXED					107
#define LOOKBACKFLOATING				108

#define ASIANGEOMETRIC					109


WolframGPULibraryData wglData = NULL;


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

	if(d > 0)
		cnd = 1 - cnd;

	return cnd;
}

__device__ inline Real_t pndGPU(Real_t d) {
	// Do something like above eventually?
	const Real_t RSQRT2PI = static_cast<Real_t>(0.39894228040143267793994605993438);
	
	const Real_t dsqby2 = d*d*static_cast<Real_t>(0.5);
	
	return _exp(-dsqby2)*RSQRT2PI;
}

//Computes CallResult and PutResult
// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesValueGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t sqrtT, expRT, expDT;
	Real_t d1, d2, CNDD1, CNDD2;

	sqrtT = _sqrt(T);
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;

	CNDD1 = cndGPU(d1);
	CNDD2 = cndGPU(d2);

	expRT = _exp(-R * T);
	expDT = _exp(-D * T);
	CallResult = S * expDT * CNDD1 - X * expRT * CNDD2;
	PutResult = X * expRT * (static_cast<Real_t>(1.0) - CNDD2) - S * expDT * (static_cast<Real_t>(1.0) - CNDD1);
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesDeltaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t expDT;
	Real_t d1, CNDD1;
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * _sqrt(T));
	
	CNDD1 = cndGPU(d1);
	
	expDT = _exp(-D*T);
	CallResult = expDT * CNDD1;
	PutResult = expDT * (CNDD1 - static_cast<Real_t>(1.0));
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesVegaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t expDT;
	Real_t d1, PNDD1;
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * _sqrt(T));
	
	PNDD1 = pndGPU(d1);
	
	expDT = _exp(-D*T);
	
	CallResult = S * _sqrt(T) * expDT * PNDD1;
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesThetaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t invSqrtT, sqrtT, expDT, expRT;
	Real_t d1, d2, CNDD1, CNDD2, CNDnD1, CNDnD2, PNDD1;
	
	invSqrtT = rsqrtf(T);
	sqrtT = static_cast<Real_t>(1.0)/invSqrtT;
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	PNDD1 = pndGPU(d1);
	CNDD1 = cndGPU(d1);
	CNDD2 = cndGPU(d2);
	CNDnD1 = cndGPU(-d1);
	CNDnD2 = cndGPU(-d2);
	
	expDT = _exp(-D*T);
	expRT = _exp(-R*T);

	CallResult = (-V * S * expDT * PNDD1 * invSqrtT * static_cast<Real_t>(0.5)) + D * S * CNDD1 * expDT - R * X * CNDD2 * expRT;
	PutResult = (-V * S * expDT * PNDD1 * invSqrtT * static_cast<Real_t>(0.5)) - D * S * CNDnD1 * expDT + R * X * CNDnD2 * expRT;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesRhoGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t expRT, sqrtT;
	Real_t d2, CNDD2, CNDnD2;

	sqrtT = _sqrt(T);
	
	d2 = (_log(S / X) + (R - D - static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	
	CNDD2 = cndGPU(d2);
	CNDnD2 = cndGPU(-d2);
	
	expRT = _exp(-R*T);

	CallResult = X * T * expRT * CNDD2;
	PutResult = -X * T * expRT * CNDnD2;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesGammaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t expDT, invVolSqrtT;
	Real_t d1, PNDD1;
	
	invVolSqrtT = rsqrtf(T) / V;
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) * invVolSqrtT;
	
	PNDD1 = pndGPU(d1);
	
	expDT = _exp(-D*T);
	CallResult = expDT * PNDD1 * invVolSqrtT / S;
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesVannaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t expDT, sqrtT;
	Real_t d1, d2, PNDD1;
	
	sqrtT = _sqrt(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	PNDD1 = pndGPU(d1);
	
	expDT = _exp(-D*T);
	CallResult = -expDT * PNDD1 * d2 / V;
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesCharmGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t expDT, sqrtT;
	Real_t d1, d2, PNDD1, CNDD1, CNDnD1;
	
	sqrtT = _sqrt(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	PNDD1 = pndGPU(d1);
	CNDD1 = cndGPU(d1);
	CNDnD1 = cndGPU(-d1);
	
	expDT = _exp(-D*T);
	CallResult = -D*expDT*CNDD1 + expDT*PNDD1*(static_cast<Real_t>(2.0)*(R-D)*T - d2*V*sqrtT) / (static_cast<Real_t>(2.0)*V*T*sqrtT);
	PutResult = CallResult + D*expDT*CNDD1 + D*expDT*CNDnD1;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesSpeedGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t invVolSqrtT, callGamma, putGamma;
	Real_t d1;
	
	invVolSqrtT = static_cast<Real_t>(1.0)/V * rsqrtf(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) * invVolSqrtT;
	
	BlackScholesGammaGPU(callGamma, putGamma, S, X, T, R, D, V);
	
	CallResult = (-callGamma / S) * (d1 * invVolSqrtT + static_cast<Real_t>(1.0));
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesZommaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	// Not terribly efficient.
	Real_t sqrtT, gammaCall, gammaPut;
	Real_t d1, d2;
	
	sqrtT = _sqrt(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	BlackScholesGammaGPU(gammaCall, gammaPut, S, X, T, R, D, V);
	
	CallResult = gammaCall * ((d1*d2 - static_cast<Real_t>(1.0))/V);
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesColorGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t sqrtT, expDT;
	Real_t d1, d2, PNDD1;
	const Real_t one = 1;
	const Real_t two = 2;
	
	sqrtT = _sqrt(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	PNDD1 = pndGPU(d1);
	
	expDT = _exp(-D*T);
	CallResult = expDT * PNDD1 / (two * S * T * V * sqrtT)  * (two * D * T + one + (two*(R-D)*T - d2*V*sqrtT) * d1 / (V * sqrtT));
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesDvegaDtimeGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t sqrtT, expDT;
	Real_t d1, d2, PNDD1;
	const Real_t one = 1;
	const Real_t two = 2;
	
	sqrtT = _sqrt(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	PNDD1 = pndGPU(d1);
	
	expDT = _exp(-D*T);
	CallResult = S*expDT*PNDD1*sqrtT* (D + ((R-D)*d1)/(V*sqrtT) - (one + d1*d2)/(two*T));
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ inline void BlackScholesVommaGPU(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V) {
	Real_t sqrtT, callVega, putVega;
	Real_t d1, d2;
	
	sqrtT = _sqrt(T);
	
	d1 = (_log(S / X) + (R - D + static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	BlackScholesVegaGPU(callVega, putVega, S, X, T, R, D, V);
	
	CallResult = callVega * (d1 * d2) / V;
	PutResult = CallResult;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility
__device__ void AsianGeometricCalculate(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V)
{
	Real_t adjVol, adjDiv;
	Real_t d1, d2, CNDD1, CNDD2, CNDnD1, CNDnD2;
	Real_t sqrtT, expRT, expBRT;
	
	sqrtT = _sqrt(T);
	adjVol = V * static_cast<Real_t>(0.577350269);  // V / sqrt(3)
	adjDiv = static_cast<Real_t>(0.5) * (R - D - V*V*static_cast<Real_t>(0.1666666666)); // (0.5 * (R - D - V*V/6))
	
	d1 = (_log(S / X) + (adjDiv + static_cast<Real_t>(0.5)*adjVol*adjVol) * T) / (adjVol * sqrtT);
	d2 = d1 - adjVol*sqrtT;
	
	CNDD1 = cndGPU(d1);
	CNDD2 = cndGPU(d2);
	CNDnD1 = cndGPU(-d1);
	CNDnD2 = cndGPU(-d2);
	
	expRT = _exp(-R*T);
	expBRT = _exp((adjDiv - R) * T);
	
	CallResult = S * expBRT * CNDD1 - X * expRT * CNDD2;
	PutResult = X * expRT * CNDnD2 - S * expBRT * CNDnD1;
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility, H: barrier, eta and phi are +/- 1 based on the type of barrier.
__device__ void BarrierCalculate(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V, Real_t H, Real_t rebate, Real_t eta, Real_t phi, unsigned int bType)
{
	Real_t sqrtT, invVolSqrtT, mu, lambda, z, x1, x2, y1, y2, expDT, expRT;
	Real_t AA, BB, CC, DD, EE, FF;
	
	const Real_t one = static_cast<Real_t>(1.0);
	const Real_t two = static_cast<Real_t>(2.0);
	sqrtT = _sqrt(T);
	invVolSqrtT = static_cast<Real_t>(1.0)/(V * sqrtT);
	mu = (R - D - static_cast<Real_t>(0.5)*V*V)/(V * V);
	lambda = (mu * mu + (static_cast<Real_t>(2.0) * R)/(V * V));
	
	z = _log(H / S) * invVolSqrtT + lambda * V * sqrtT;
	x1 = _log(S / X) * invVolSqrtT + (one + mu) * V * sqrtT;
	x2 = _log(S / H) * invVolSqrtT + (one + mu) * V * sqrtT;
	y1 = _log((H*H)/(S*X)) * invVolSqrtT + (one + mu) * V * sqrtT;
	y2 = _log(H / S) * invVolSqrtT + (one + mu) * V * sqrtT;
	
	expDT = _exp(-D*T);
	expRT = _exp(-R*T);
	
	AA = phi * S * expDT * cndGPU(phi*x1) - phi * X * expRT * cndGPU(phi * x1 - phi * V * sqrtT);
	BB = phi * S * expDT * cndGPU(phi*x2) - phi * X * expRT * cndGPU(phi * x2 - phi * V * sqrtT);
	CC = phi * S * expDT * pow(H/S, two*mu + two) * cndGPU(eta*y1) - phi * X * expRT * pow(H/S, two*mu) * cndGPU(eta * y1 - eta * V * sqrtT);
	DD = phi * S * expDT * pow(H/S, two*mu + two) * cndGPU(eta*y2) - phi * X * expRT * pow(H/S, two*mu) * cndGPU(eta * y2 - eta * V * sqrtT);
	EE = rebate * expRT * (cndGPU(eta * x2 - eta * V * sqrtT) - pow(H/S, two*mu) * cndGPU(eta * y2 - eta * V * sqrtT));
	FF = rebate * (pow(H/S, mu + lambda) * cndGPU(eta * z) + pow(H/S, mu - lambda) * cndGPU(eta * z - two * eta * lambda * V * sqrtT));
	
	switch(bType) {
		case BARRIERDOWNIN:
			if(X > H) { CallResult = CC + EE; PutResult = BB - CC + DD + EE; }
			else { CallResult = AA - BB + DD + EE; PutResult = AA + EE; }
			break;
		case BARRIERDOWNOUT:
			if(X > H) { CallResult = AA - CC + FF; PutResult = AA - BB + CC - DD + FF; }
			else { CallResult = BB - DD + FF; PutResult = FF; }
			break;
		case BARRIERUPIN:
			if(X > H) { CallResult = AA + EE; PutResult = AA - BB + DD + EE; }
			else { CallResult = BB - CC + DD + EE; PutResult = CC + EE; }
			break;
		case BARRIERUPOUT:
			if(X > H) { CallResult = FF; PutResult = BB - DD + FF; }
			else { CallResult = AA - BB + CC - DD + FF; PutResult = AA - CC + FF; }
			break;
	}		
}

// S: spot price, X: strike Price, T: Expiration, R: Risk free interest rate, D: Dividend, V: Volatility, lbType: Specifies floating or fixed lookback
__device__ void LookbackCalculate(Real_t& CallResult, Real_t& PutResult, Real_t S, Real_t X, Real_t T, Real_t R, Real_t D, Real_t V, unsigned int lbType)
{
	Real_t sqrtT, expRT, expDT, expRDT;
	Real_t d1, d2, CNDD1, CNDD2, CNDnD1, CNDnD2, K;
	
	K = max(S, X);
	sqrtT = _sqrt(T);
	
	d1 = (_log(S/X) + (R + D - static_cast<Real_t>(0.5) * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;
	
	CNDD1 = cndGPU(d1);
	CNDD2 = cndGPU(d2);
	CNDnD1 = cndGPU(-d1);
	CNDnD2 = cndGPU(-d2);
	
	expRT = _exp(-R*T);
	expDT = _exp(-D*T);
	expRDT = _exp((R-D)*T);
	
	if(lbType == LOOKBACKFIXED) {
		CallResult = expRT * max(S - X, static_cast<Real_t>(0.0)) + S * expDT * CNDD1 - K * expRT * CNDD2 + S * expRT * V * V / (static_cast<Real_t>(2.0) * (R - D)) * (-pow(S/K, static_cast<Real_t>(-2.0) * (R - D) / (V * V))) * cndGPU(d1 - static_cast<Real_t>(2.0) * (R - D) * sqrtT / V) + expRDT * CNDD1;
		PutResult = expRT * max(X - S, static_cast<Real_t>(0.0)) - S * expDT * CNDnD1 + K * expRT * CNDnD2 + S * expRT * V * V / (static_cast<Real_t>(2.0) * (R - D)) * (pow(S/K, static_cast<Real_t>(-2.0) * (R - D) / (V * V))) * cndGPU(-d1 + static_cast<Real_t>(2.0) * (R - D) * sqrtT / V) - expRDT * CNDnD1;
	}
	else {
		CallResult = S * expDT * CNDD1 - S * expRT * CNDD2 + S * expRT * V * V / (static_cast<Real_t>(2.0) * (R - D)) * cndGPU(-d1 + static_cast<Real_t>(2.0) * (R - D) * sqrtT / V) - expRDT * CNDnD1;
		PutResult = -S * expDT * CNDnD1 + S * expRT * CNDnD2 + S * expRT * V * V / (static_cast<Real_t>(2.0) * (R - D)) * cndGPU(d1 - static_cast<Real_t>(2.0) * (R - D) * sqrtT / V) + expRDT * CNDD1;
	}
}

__global__ void BlackScholesGPU(Real_t *d_CallResult, Real_t *d_PutResult, Real_t *d_CurrentPrice, Real_t *d_OptionStrike, Real_t *d_Expiration, Real_t *d_InterestRate, Real_t *d_Dividends, Real_t *d_Volatility, int optN, int toCalculate) {

	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	/*for(int opt = tid; opt < optN; opt += THREAD_N) {
		BlackScholesValueGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
	}*/
	
	switch(toCalculate) {
		case VALUE:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesValueGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case DELTA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesDeltaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case VEGA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesVegaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case THETA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesThetaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case RHO:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesRhoGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case GAMMA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesGammaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case VANNA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesVannaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case CHARM:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesCharmGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case VOMMA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesVommaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case DVEGADTIME:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesDvegaDtimeGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case SPEED:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesSpeedGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case ZOMMA:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesZommaGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		case COLOR:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				BlackScholesColorGPU(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
			}
			break;
		default:
			for(int opt = tid; opt < optN; opt += THREAD_N) {
				d_CallResult[opt] = static_cast<Real_t>(0.0);
				d_PutResult[opt] = static_cast<Real_t>(0.0);
			}
			break;
	}
}

__global__ void AsianGeometricAnalyticGPU(int optN, Real_t *d_CallResult, Real_t *d_PutResult, Real_t *d_CurrentPrice, Real_t *d_OptionStrike, Real_t *d_Expiration, Real_t *d_InterestRate, Real_t *d_Dividends, Real_t *d_Volatility) {

	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;
	
	for(int opt = tid; opt < optN; opt += THREAD_N) {
		AsianGeometricCalculate(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt]);
	}
}

__global__ void BarrierAnalyticGPU(int optN, Real_t *d_CallResult, Real_t *d_PutResult, Real_t *d_CurrentPrice, Real_t *d_OptionStrike, Real_t *d_Expiration, Real_t *d_InterestRate, Real_t *d_Dividends, Real_t *d_Volatility, Real_t *d_Barrier, Real_t *d_Rebate, unsigned int barrierType, int callOrPut)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;
	
	Real_t eta, phi;
	
	switch(barrierType) {
		case BARRIERDOWNIN:
			eta = static_cast<Real_t>(1.0);
			phi = callOrPut == 1 ? static_cast<Real_t>(1.0) : static_cast<Real_t>(-1.0);
			break;
		case BARRIERDOWNOUT:
			eta = static_cast<Real_t>(1.0);
			phi = callOrPut == 1 ? static_cast<Real_t>(1.0) : static_cast<Real_t>(-1.0);
			break;
		case BARRIERUPIN:
			eta = static_cast<Real_t>(-1.0);
			phi = callOrPut == 1 ? static_cast<Real_t>(1.0) : static_cast<Real_t>(-1.0);
			break;
		case BARRIERUPOUT:
			eta = static_cast<Real_t>(-1.0);
			phi = callOrPut == 1 ? static_cast<Real_t>(1.0) : static_cast<Real_t>(-1.0);
			break;
	}
	
	for(int opt = tid; opt < optN; opt += THREAD_N) {
		BarrierCalculate(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt], d_Barrier[opt], d_Rebate[opt], eta, phi, barrierType);
	}
}

__global__ void LookbackAnalyticGPU(int optN, Real_t *d_CallResult, Real_t *d_PutResult, Real_t *d_CurrentPrice, Real_t *d_OptionStrike, Real_t *d_Expiration, Real_t *d_InterestRate, Real_t *d_Dividends, Real_t *d_Volatility, unsigned int lookbackType)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;
	
	for(int opt = tid; opt < optN; opt += THREAD_N) {
		LookbackCalculate(d_CallResult[opt], d_PutResult[opt], d_CurrentPrice[opt], d_OptionStrike[opt], d_Expiration[opt], d_InterestRate[opt], d_Dividends[opt], d_Volatility[opt], lookbackType);
	}
}

static int iBlackScholes(WGL_Memory_t call, WGL_Memory_t put, WGL_Memory_t spot,
						 WGL_Memory_t strikePrice, WGL_Memory_t expiration, WGL_Memory_t interest,
						 WGL_Memory_t volatility, WGL_Memory_t dividend, WGL_Memory_t barrier, WGL_Memory_t rebate,
						 mint numOptions, mint calculationType, mint optionType, int callOrPut) {
	mbool barrierQ = False;
	dim3 blockDim(128);
	dim3 gridDim(512);

	if (!(WGL_Type_RealQ(call) && WGL_Type_RealQ(put) && WGL_Type_RealQ(spot) &&
		  WGL_Type_RealQ(strikePrice) && WGL_Type_RealQ(expiration) && WGL_Type_RealQ(interest) &&
		  WGL_Type_RealQ(volatility) && WGL_Type_RealQ(dividend))) {
		return LIBRARY_TYPE_ERROR;
	}

	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, call, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, put, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, spot, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, strikePrice, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, expiration, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, interest, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, dividend, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, volatility, wglErr), cleanup);

	if (calculationType >= 0) {
		BlackScholesGPU<<<gridDim, blockDim>>>(
			CUDA_Runtime_getDeviceMemoryAsReal(call),
			CUDA_Runtime_getDeviceMemoryAsReal(put),
			CUDA_Runtime_getDeviceMemoryAsReal(spot),
			CUDA_Runtime_getDeviceMemoryAsReal(strikePrice),
			CUDA_Runtime_getDeviceMemoryAsReal(expiration),
			CUDA_Runtime_getDeviceMemoryAsReal(interest),
			CUDA_Runtime_getDeviceMemoryAsReal(dividend),
			CUDA_Runtime_getDeviceMemoryAsReal(volatility),
			numOptions,
			calculationType
		);
	} else if (optionType == ASIANGEOMETRIC) {
		AsianGeometricAnalyticGPU<<<gridDim, blockDim>>>(
			numOptions,
			CUDA_Runtime_getDeviceMemoryAsReal(call),
			CUDA_Runtime_getDeviceMemoryAsReal(put),
			CUDA_Runtime_getDeviceMemoryAsReal(spot),
			CUDA_Runtime_getDeviceMemoryAsReal(strikePrice),
			CUDA_Runtime_getDeviceMemoryAsReal(expiration),
			CUDA_Runtime_getDeviceMemoryAsReal(interest),
			CUDA_Runtime_getDeviceMemoryAsReal(dividend),
			CUDA_Runtime_getDeviceMemoryAsReal(volatility)
		);
	} else if (optionType == LOOKBACKFIXED || optionType == LOOKBACKFLOATING) {
		LookbackAnalyticGPU<<<gridDim, blockDim>>>(
			numOptions,
			CUDA_Runtime_getDeviceMemoryAsReal(call),
			CUDA_Runtime_getDeviceMemoryAsReal(put),
			CUDA_Runtime_getDeviceMemoryAsReal(spot),
			CUDA_Runtime_getDeviceMemoryAsReal(strikePrice),
			CUDA_Runtime_getDeviceMemoryAsReal(expiration),
			CUDA_Runtime_getDeviceMemoryAsReal(interest),
			CUDA_Runtime_getDeviceMemoryAsReal(dividend),
			CUDA_Runtime_getDeviceMemoryAsReal(volatility),
			optionType
		);
	} else {
		barrierQ = True;
		WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, barrier, wglErr), cleanup);
		WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, rebate, wglErr), cleanup);

		BarrierAnalyticGPU<<<gridDim, blockDim>>>(
			numOptions,
			CUDA_Runtime_getDeviceMemoryAsReal(call),
			CUDA_Runtime_getDeviceMemoryAsReal(put),
			CUDA_Runtime_getDeviceMemoryAsReal(spot),
			CUDA_Runtime_getDeviceMemoryAsReal(strikePrice),
			CUDA_Runtime_getDeviceMemoryAsReal(expiration),
			CUDA_Runtime_getDeviceMemoryAsReal(interest),
			CUDA_Runtime_getDeviceMemoryAsReal(dividend),
			CUDA_Runtime_getDeviceMemoryAsReal(volatility),
			CUDA_Runtime_getDeviceMemoryAsReal(barrier),
			CUDA_Runtime_getDeviceMemoryAsReal(rebate),
			optionType,
			callOrPut
		);
	}

	CUDA_Runtime_synchronize(wglErr);


cleanup:

	if (WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, call, wglErr);
		CUDA_Runtime_setMemoryAsValidOutput(wglState, put, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, call, wglErr);
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, put, wglErr);
	}

	CUDA_Runtime_unsetMemoryAsInput(wglState, spot, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, strikePrice, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, expiration, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, interest, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, volatility, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, dividend, wglErr);

	if (barrierQ) {
		CUDA_Runtime_unsetMemoryAsInput(wglState, rebate, wglErr);
		CUDA_Runtime_unsetMemoryAsInput(wglState, barrier, wglErr);
	}

	if (WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}

EXTERN_C DLLEXPORT int oBlackScholes(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t callMemory, putMemory, currentPriceMemory, strikePriceMemory, expirationMemory, interestMemory;
	WGL_Memory_t volatilityMemory, dividendMemory, barrierMemory, rebateMemory;
	mint callMemoryId, putMemoryId, currentPriceMemoryId, strikePriceMemoryId, expirationMemoryId, interestMemoryId;
	mint volatilityMemoryId, dividendMemoryId, barrierMemoryId, rebateMemoryId;
	mint numOptions, calculationType, optionType, callOrPut;

	int err;

	assert(Argc == 16);

	callMemoryId				= MArgument_getInteger(Args[0]);
	putMemoryId					= MArgument_getInteger(Args[1]);
	currentPriceMemoryId		= MArgument_getInteger(Args[2]);
	strikePriceMemoryId			= MArgument_getInteger(Args[3]);
	expirationMemoryId			= MArgument_getInteger(Args[4]);
	interestMemoryId			= MArgument_getInteger(Args[5]);
	volatilityMemoryId			= MArgument_getInteger(Args[6]);
	dividendMemoryId			= MArgument_getInteger(Args[7]);
	barrierMemoryId				= MArgument_getInteger(Args[8]);
	rebateMemoryId				= MArgument_getInteger(Args[9]);
	numOptions					= MArgument_getInteger(Args[10]);
	calculationType				= MArgument_getInteger(Args[11]);
	optionType					= MArgument_getInteger(Args[12]);
	callOrPut					= MArgument_getInteger(Args[13]);

	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	callMemory					= wglData->findMemory(wglData, callMemoryId);
	putMemory					= wglData->findMemory(wglData, putMemoryId);
	currentPriceMemory			= wglData->findMemory(wglData, currentPriceMemoryId);
	strikePriceMemory			= wglData->findMemory(wglData, strikePriceMemoryId);
	expirationMemory			= wglData->findMemory(wglData, expirationMemoryId);
	interestMemory				= wglData->findMemory(wglData, interestMemoryId);
	volatilityMemory			= wglData->findMemory(wglData, volatilityMemoryId);
	dividendMemory				= wglData->findMemory(wglData, dividendMemoryId);
	barrierMemory				= wglData->findMemory(wglData, barrierMemoryId);
	rebateMemory				= wglData->findMemory(wglData, rebateMemoryId);

	err = iBlackScholes(callMemory, putMemory, currentPriceMemory, strikePriceMemory, expirationMemory, interestMemory,
				   	    volatilityMemory, dividendMemory, barrierMemory, rebateMemory, numOptions, calculationType, optionType,
						callOrPut);
cleanup:

	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else if (err != LIBRARY_NO_ERROR) {
		return err;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}


/************************************************/
/* Binomial Method Options Pricing				*/
/************************************************/

 /**
  * Original code is under
  * ${basedir}/ExtraComponents/CUDA_SDK/3.0/Linux-x86-64/C/src/binomialOptions
 **/
 

#define CACHE_DELTA 32
#define CACHE_SIZE 256
#define CACHE_STEP (CACHE_SIZE - CACHE_DELTA)

#define  NUM_STEPS		128

__device__ inline Real_t expiryCallValue(Real_t S, Real_t X, Real_t vDt, float callPutFactor, int i) {
	Real_t d = S * _exp(vDt * (NUM_STEPS - static_cast<Real_t>(2.0) * i)) - X;
	d *= callPutFactor;
	return (d > static_cast<Real_t>(0.0)) ? d : static_cast<Real_t>(0.0);
}

__global__ void binomialOptionsKernel(Real_t* d_CallValue, Real_t* d_CallBuffer,
									  Real_t* d_S, Real_t* d_X, Real_t* d_vDt, Real_t* d_puByDf,
									  Real_t* d_pdByDf, int optType, int call)
{
	__shared__ Real_t callA[CACHE_SIZE+1];
	__shared__ Real_t callB[CACHE_SIZE+1];
	Real_t *const d_Call = &d_CallBuffer[blockIdx.x * (NUM_STEPS + 16)];

	const int tid = threadIdx.x;
	const Real_t S = d_S[blockIdx.x];
	const Real_t X = d_X[blockIdx.x];
	const Real_t vDt = d_vDt[blockIdx.x];
	const Real_t puByDf = d_puByDf[blockIdx.x];
	const Real_t pdByDf = d_pdByDf[blockIdx.x];
	const Real_t callPutFactor = call == 1 ? static_cast<Real_t>(1.0) : static_cast<Real_t>(-1.0);
	Real_t callValue, temp, currentVal;

	for(int i = tid; i <= NUM_STEPS; i+= CACHE_SIZE)
		d_Call[i] = expiryCallValue(S, X, vDt, callPutFactor, i);

	for(int i = NUM_STEPS; i > 0; i -= CACHE_DELTA) {
		for(int c_base = 0; c_base < i; c_base += CACHE_STEP) {
			int c_start = min(CACHE_SIZE - 1, i - c_base);
			int c_end = c_start - CACHE_DELTA;

			__syncthreads();
			if(tid <= c_start)
				callA[tid] = d_Call[c_base + tid];

			currentVal = vDt * static_cast<Real_t>(i - 2*(c_base + tid) - 1);

			for(int k = c_start - 1; k >= c_end;) {
				__syncthreads();
				callValue = pdByDf * callA[tid+1] + puByDf * callA[tid];
				if(optType == AMERICAN) {
					temp = S * _exp(currentVal) - X;
					temp *= callPutFactor;
					callValue = callValue > temp ? callValue : temp;
				}
				callB[tid] = callValue;
				k--;
				currentVal -= vDt;
				
				__syncthreads();
				callValue = pdByDf * callB[tid+1] + puByDf * callB[tid];
				if(optType == AMERICAN) {
					temp = S * _exp(currentVal) - X;
					temp *= callPutFactor;
					callValue = callValue > temp ? callValue : temp;
				}
				callA[tid] = callValue;
				k--;
				currentVal -= vDt;
			}
			__syncthreads();
			if(tid <= c_end)
				d_Call[c_base + tid] = callA[tid];
		}
	}
	if(threadIdx.x == 0) d_CallValue[blockIdx.x] = static_cast<Real_t>(callA[0]);
}


static int iBinomialMethod(WGL_Memory_t priceRes, WGL_Memory_t spot, WGL_Memory_t strike, WGL_Memory_t buffer, 
						   WGL_Memory_t vDt, WGL_Memory_t puByDf, WGL_Memory_t pdByDf, mint numOptions,
						   mint optionType, mint callOrPut) {
	int err = LIBRARY_FUNCTION_ERROR;

	dim3 blockDim(256);
	dim3 gridDim(numOptions);

	if (!(WGL_Type_RealQ(priceRes) && WGL_Type_RealQ(spot) && WGL_Type_RealQ(strike) &&
		  WGL_Type_RealQ(buffer) && WGL_Type_RealQ(vDt) && WGL_Type_RealQ(puByDf) &&
		  WGL_Type_RealQ(pdByDf))) {
		return LIBRARY_TYPE_ERROR;
	}

	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, priceRes, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, spot, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, strike, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, buffer, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, vDt, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, puByDf, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, pdByDf, wglErr), cleanup);
	
	binomialOptionsKernel<<<gridDim, blockDim>>>(
		CUDA_Runtime_getDeviceMemoryAsReal(priceRes),
		CUDA_Runtime_getDeviceMemoryAsReal(buffer),
		CUDA_Runtime_getDeviceMemoryAsReal(spot),
		CUDA_Runtime_getDeviceMemoryAsReal(strike),
		CUDA_Runtime_getDeviceMemoryAsReal(vDt),
		CUDA_Runtime_getDeviceMemoryAsReal(puByDf),
		CUDA_Runtime_getDeviceMemoryAsReal(pdByDf),
		optionType,
		callOrPut
	);
	
	CUDA_Runtime_synchronize(wglErr);
	if (WGL_SuccessQ) {
		err = LIBRARY_NO_ERROR;
	}
cleanup:
	if (WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, priceRes, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, priceRes, wglErr);
	}
	
	CUDA_Runtime_unsetMemoryAsInput(wglState, spot, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, strike, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, vDt, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, puByDf, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, pdByDf, wglErr);
	
	return err;
}

EXTERN_C DLLEXPORT int oBinomialMethod(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t resMem, currentPriceMem, strikeMem, expirationMem, interestMem, volatilityMem, dividendMem;
	mint resMemId, currentPriceMemId, strikeMemId, expirationMemId, interestMemId, volatilityMemId, dividendMemId;
	mint numOptions, optionType, callOrPut;
	mint numSteps = 128;//treeDepth;
	Real_t * dt = NULL, * vDt = NULL, * rDt = NULL, * If = NULL, * df = NULL;
	Real_t * u = NULL, * d = NULL, * pu = NULL, * pd = NULL, * puByDf = NULL, * pdByDf = NULL;
	Real_t * hCallBuffer = NULL;
	WGL_Memory_t callBufferMem = NULL, vDtMem = NULL, puByDfMem = NULL, pdByDfMem = NULL;
	int err = LIBRARY_NO_ERROR;
	double * hExpiration, * hVolatility, * hInterest, * hDividend;

	assert(Argc == 12);
		
	resMemId					= MArgument_getInteger(Args[0]);
	currentPriceMemId			= MArgument_getInteger(Args[1]);
	strikeMemId					= MArgument_getInteger(Args[2]);
	expirationMemId				= MArgument_getInteger(Args[3]);
	interestMemId				= MArgument_getInteger(Args[4]);
	volatilityMemId				= MArgument_getInteger(Args[5]);
	dividendMemId				= MArgument_getInteger(Args[6]);
	numOptions 					= MArgument_getInteger(Args[7]);
	optionType 					= MArgument_getInteger(Args[8]);
	callOrPut 					= MArgument_getInteger(Args[9]);
	
	resMem						= wglData->findMemory(wglData, resMemId);
	currentPriceMem				= wglData->findMemory(wglData, currentPriceMemId);
	strikeMem					= wglData->findMemory(wglData, strikeMemId);
	expirationMem				= wglData->findMemory(wglData, expirationMemId);
	interestMem					= wglData->findMemory(wglData, interestMemId);
	volatilityMem				= wglData->findMemory(wglData, volatilityMemId);
	dividendMem					= wglData->findMemory(wglData, dividendMemId);
		
	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);
	
	New(dt, Real_t, numOptions * sizeof(Real_t));
	New(vDt, Real_t, numOptions * sizeof(Real_t));
	New(rDt, Real_t, numOptions * sizeof(Real_t));
	New(If, Real_t, numOptions * sizeof(Real_t));
	New(df, Real_t, numOptions * sizeof(Real_t));
	New(u, Real_t, numOptions * sizeof(Real_t));
	New(d, Real_t, numOptions * sizeof(Real_t));
	New(pu, Real_t, numOptions * sizeof(Real_t));
	New(pd, Real_t, numOptions * sizeof(Real_t));
	New(puByDf, Real_t, numOptions * sizeof(Real_t));
	New(pdByDf, Real_t, numOptions * sizeof(Real_t));
	

	hExpiration = wglData->MTensorMemory_getRealData(wglData, expirationMem);
	assert(hExpiration != NULL);
	
	hVolatility = wglData->MTensorMemory_getRealData(wglData, volatilityMem);
	assert(hVolatility != NULL);
	
	hInterest = wglData->MTensorMemory_getRealData(wglData, interestMem);
	assert(hInterest != NULL);
	
	hDividend = wglData->MTensorMemory_getRealData(wglData, dividendMem);
	assert(hDividend != NULL);
	
	New(hCallBuffer, Real_t, numOptions * (numSteps + 16) * sizeof(Real_t));
		
	// We need to calculate pseudoprobabilities that the price of the asset will go up or down, as well as the amount it will go up or down.
	for (mint ii = 0; ii < numOptions; ii++) {
		// Width of a time step
		dt[ii] = static_cast<Real_t>(hExpiration[ii]) / static_cast<Real_t>(numSteps);
		// Volatility multiplied by square root of the timestep -- comes up in simulating brownian motion
		vDt[ii] = static_cast<Real_t>(hVolatility[ii]) * sqrt(dt[ii]);
		// Used to account for the rate of risk free interest and the dividends of the asset
		rDt[ii] = static_cast<Real_t>(hInterest[ii] - hDividend[ii]) * dt[ii];
		// As above [these could probably be combined into one step]
		If[ii] = exp(rDt[ii]);
		// Used to account for just risk free interest
		df[ii] = exp(static_cast<Real_t>(-hInterest[ii] * dt[ii]));
		// Amount increased (u) or decreased (d) at each time step
		u[ii] = exp(vDt[ii]);
		d[ii] = exp(-vDt[ii]);
		// Pseudoprobability of increase (pu) or decrease (pd)
		pu[ii] = (If[ii] - d[ii]) / (u[ii] - d[ii]);
		pd[ii] = 1.0f - pu[ii];
		// Multiply by df to adjust for risk free interest rate.
		puByDf[ii] = pu[ii] * df[ii];
		pdByDf[ii] = pd[ii] * df[ii];
	}
	
	callBufferMem = wglData->newRawMemory(wglData, (void**)&hCallBuffer, WGL_MemoryResidence_DeviceHost, numOptions * (numSteps + 16) * sizeof(Real_t), True);
	callBufferMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);

	vDtMem = wglData->newRawMemory(wglData, (void**)&vDt, WGL_MemoryResidence_DeviceHost, numOptions * sizeof(Real_t), True);
	vDtMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);

	puByDfMem = wglData->newRawMemory(wglData, (void**)&puByDf, WGL_MemoryResidence_DeviceHost, numOptions * sizeof(Real_t), True);
	puByDfMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);

	pdByDfMem = wglData->newRawMemory(wglData, (void**)&pdByDf, WGL_MemoryResidence_DeviceHost, numOptions * sizeof(Real_t), True);
	pdByDfMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);
	
	err = iBinomialMethod(resMem, currentPriceMem, strikeMem, callBufferMem, vDtMem, puByDfMem, pdByDfMem, numOptions, optionType, callOrPut);
cleanup:	

	Free(dt);
	Free(rDt);
	Free(If);
	Free(df);
	Free(u);
	Free(d);
	Free(pu);
	Free(pd);
	
	wglData->freeMemory(wglData, callBufferMem);
	wglData->freeMemory(wglData, vDtMem);
	wglData->freeMemory(wglData, puByDfMem);
	wglData->freeMemory(wglData, pdByDfMem);
	
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else if (err != LIBRARY_NO_ERROR) {
		return err;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}


/************************************************/
/* Binomial Method Options Pricing				*/
/************************************************/


 /**
  * Original code is under
  * ${basedir}/ExtraComponents/CUDA_SDK/3.0/Linux-x86-64/C/src/binomialOptions
 **/

#define THREAD_N 256

// Barrier types (masks, eg a down and out option has type 3, up and in has type 0, etc.)
#define BARRIER_DOWN		1
#define BARRIER_OUT			2

#define LOOKBACK_FIXED		0
#define LOOKBACK_FLOATING	1

template<unsigned int blockSize>
__device__ void sumReduceSharedMem(volatile Real_t *sum, int tid)
{
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sum[tid] += sum[tid + 256]; } __syncthreads();}
    if (blockSize >= 256) { if (tid < 128) { sum[tid] += sum[tid + 128]; } __syncthreads();}
    if (blockSize >= 128) { if (tid <  64) { sum[tid] += sum[tid +  64]; } __syncthreads();}
    
    if (tid < 32)
    {
        if (blockSize >=  64) { sum[tid] += sum[tid + 32]; }
        if (blockSize >=  32) { sum[tid] += sum[tid + 16]; }
        if (blockSize >=  16) { sum[tid] += sum[tid +  8]; }
        if (blockSize >=   8) { sum[tid] += sum[tid +  4]; }
        if (blockSize >=   4) { sum[tid] += sum[tid +  2]; }
        if (blockSize >=   2) { sum[tid] += sum[tid +  1]; }
    }
}


#define UNROLL_REDUCTION
template<int SUM_N, int blockSize> 
__device__ void sumReduce(Real_t *sum)
{
#ifdef UNROLL_REDUCTION
    for(int pos = threadIdx.x; pos < SUM_N; pos += blockSize){
        __syncthreads();
        sumReduceSharedMem<blockSize>(sum, pos);
    }
#else
    for(int stride = SUM_N / 2; stride > 0; stride >>= 1){
        __syncthreads();
        for(int pos = threadIdx.x; pos < stride; pos += blockSize){
            sum[pos] += sum[pos + stride];
        }
    }
#endif
}

// S: spot price, X: strike price
// MuByT and VBySqrtT are the mean and variance of the normal random variables used to simulate brownian motion.
// d_Samples is a pool of normally distributed random samples. 
__device__ inline Real_t endCallValueBarrier(Real_t S, Real_t X, Real_t MuByT, Real_t VBySqrtT,
		unsigned int index, unsigned int pathN, Real_t *d_Samples, Real_t callPutFactor, unsigned int depthN,
		Real_t barrier, int barrierType)
{
	Real_t value = S;
	unsigned int i;
	Real_t r = 0;
	Real_t sqrtdt = rsqrt((Real_t)depthN);
	Real_t dt = sqrtdt*sqrtdt;
	Real_t exponent = 0;
	Real_t logBarrier = _log(barrier / S);  		// If exponent crosses this it is equivalent to value crossing barrier. Cuts down on computation.
	unsigned int crossed = 0;
	if((exponent < logBarrier) && (barrierType & BARRIER_DOWN)) crossed = 1;
	if((exponent > logBarrier) && !(barrierType & BARRIER_DOWN)) crossed = 1;
	for(i = 0; i < depthN; i++) {
		r = d_Samples[index + i * pathN];
		exponent += MuByT * dt + VBySqrtT * sqrtdt * r;
		if((exponent < logBarrier) && (barrierType & BARRIER_DOWN)) crossed = 1;
		if((exponent > logBarrier) && !(barrierType & BARRIER_DOWN)) crossed = 1;
	}
	value = S * _exp(exponent);
	
	Real_t callValue = value - X;
	
	callValue *= callPutFactor;
	
	if((crossed == 1) && (barrierType & BARRIER_OUT)) return static_cast<Real_t>(0.0);
	if((crossed == 0) && !(barrierType & BARRIER_OUT)) return static_cast<Real_t>(0.0);
	return (callValue > static_cast<Real_t>(0.0)) ? callValue : static_cast<Real_t>(0.0);
}

__device__ inline Real_t endCallValueLookback(Real_t S, Real_t X, Real_t MuByT, Real_t VBySqrtT,
		unsigned int index, unsigned int pathN, Real_t *d_Samples, Real_t callPutFactor, unsigned int depthN, 
		int lookbackType)
{
	Real_t maxValue = S;
	Real_t minValue = S;
	Real_t currentValue = S;
	unsigned int i;
	Real_t r = static_cast<Real_t>(0.0);
	Real_t sqrtdt = rsqrt((Real_t)depthN);
	Real_t dt = sqrtdt*sqrtdt;
	Real_t exponent = static_cast<Real_t>(0.0);
	Real_t callValue, putValue;
	for(i = 0; i < depthN; i++) {
		r = d_Samples[index + i * pathN];
		exponent += MuByT * dt + VBySqrtT * sqrtdt * r;
		currentValue = S * _exp(exponent);
		if(currentValue < minValue) minValue = currentValue;
		if(currentValue > maxValue) maxValue = currentValue;
	}

	if(lookbackType == LOOKBACK_FLOATING) {
		callValue = currentValue - minValue;
		putValue = maxValue - currentValue;
	}
	else {
		callValue = maxValue - X;
		putValue = X - minValue;
	}
	
	if(callPutFactor > static_cast<Real_t>(0.0))
		return (callValue > static_cast<Real_t>(0.0)) ? callValue : static_cast<Real_t>(0.0);
	else
		return (putValue > static_cast<Real_t>(0.0)) ? putValue : static_cast<Real_t>(0.0);
}

__device__ inline Real_t endCallValueAsian(Real_t S, Real_t X, Real_t MuByT, Real_t VBySqrtT,
        unsigned int index, unsigned int pathN, Real_t *d_Samples, Real_t callPutFactor, unsigned int depthN) 
{
    Real_t sum = S;
    unsigned int i;
    Real_t r = static_cast<Real_t>(0.0);
    Real_t sqrtdt = rsqrt((Real_t)depthN);
    Real_t dt = sqrtdt*sqrtdt;
    Real_t exponent = static_cast<Real_t>(0.0);
    for(i = 0; i < depthN; i++) {
        r = d_Samples[index + i * pathN];
        exponent += MuByT * dt + VBySqrtT * sqrtdt * r;
        sum += S * _exp(exponent);
    }

    sum /= static_cast<Real_t>(depthN + 1);
    Real_t callValue = sum - X;
    
    callValue *= callPutFactor;
    return (callValue > static_cast<Real_t>(0.0)) ? callValue : static_cast<Real_t>(0.0);
}

__device__ inline Real_t endCallValueEuropean(Real_t S, Real_t X, Real_t MuByT, Real_t VBySqrtT,
		unsigned int index, unsigned int pathN, Real_t *d_Samples, Real_t callPutFactor, unsigned int depthN) 
{
    Real_t r = d_Samples[index];
    Real_t callValue = S * _exp(MuByT + VBySqrtT * r) - X;
    
    callValue *= callPutFactor;
    
    return (callValue > static_cast<Real_t>(0.0)) ? callValue : static_cast<Real_t>(0.0);
}

__device__ inline Real_t endCallValue(Real_t S, Real_t X, Real_t MuByT, Real_t VBySqrtT, unsigned int index,
		unsigned int pathN, Real_t *d_Samples, Real_t callPutFactor, unsigned int depthN, int optType, Real_t barrier)
{
	Real_t res = static_cast<Real_t>(-1.0);
	switch(optType) {
		case EUROPEAN:
			res = endCallValueEuropean(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN);
			break;
		case ASIAN:
			res = endCallValueAsian(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN);
			break;
		case BARRIERUPIN:
			res = endCallValueBarrier(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN, barrier, 0);
			break;
		case BARRIERUPOUT:
			res = endCallValueBarrier(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN, barrier, BARRIER_OUT);
			break;
		case BARRIERDOWNIN:
			res = endCallValueBarrier(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN, barrier, BARRIER_DOWN);
			break;
		case BARRIERDOWNOUT:
			res = endCallValueBarrier(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN, barrier, BARRIER_DOWN | BARRIER_OUT);
			break;
		case LOOKBACKFIXED:
			res = endCallValueLookback(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN, LOOKBACK_FIXED);
			break;
		case LOOKBACKFLOATING:
			res = endCallValueLookback(S, X, MuByT, VBySqrtT, index, pathN, d_Samples, callPutFactor, depthN, LOOKBACK_FIXED);
			break;
		default:
			break;
	}
	return res;
}

__global__ void MonteCarloKernel(
    Real_t *d_S,
    Real_t *d_X,
    Real_t *d_MuByT,
    Real_t *d_VBySqrtT,
    Real_t *d_Barrier,
    Real_t *d_Buffer,
    Real_t *d_Samples,
    unsigned int pathN,
    unsigned int depthN,
    int optType,
    int call)
{
    const int optionIndex = blockIdx.y;

    const Real_t S = d_S[optionIndex];
    const Real_t X = d_X[optionIndex];
    const Real_t MuByT = d_MuByT[optionIndex];
    const Real_t VBySqrtT = d_VBySqrtT[optionIndex];
    const Real_t Barrier = d_Barrier[optionIndex];
    const Real_t callPutFactor = call ? static_cast<Real_t>(1.0) : static_cast<Real_t>(-1.0);
    
    //One thread per partial integral
    const unsigned int   iSum = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int accumN = blockDim.x * gridDim.x;

    //Cycle through the entire samples array:
    //derive end stock price for each path
    //accumulate into intermediate global memory array
    Real_t sumCall = static_cast<Real_t>(0.0);
    for(unsigned int i = iSum; i < pathN; i += accumN){
        Real_t      callValue = endCallValue(S, X, MuByT, VBySqrtT, i, pathN, d_Samples, callPutFactor, depthN, optType, Barrier);
        sumCall += callValue;
    }
    d_Buffer[optionIndex * accumN + iSum] = sumCall;
}

__global__ void MonteCarloOneBlockPerOption(
    Real_t *d_CallValue,
    Real_t *d_S,
    Real_t *d_X,
    Real_t *d_MuByT,
    Real_t *d_VBySqrtT,
    Real_t *d_Barrier,
    Real_t *d_Samples,
    unsigned int pathN,
    unsigned int depthN,
    int optType,
    int call)
{
    const int SUM_N = THREAD_N;
    __shared__ Real_t s_SumCall[SUM_N];

    const int optionIndex = blockIdx.x;
    const Real_t S = d_S[optionIndex];
    const Real_t X = d_X[optionIndex];
    const Real_t MuByT = d_MuByT[optionIndex];
    const Real_t VBySqrtT = d_VBySqrtT[optionIndex];
    const Real_t Barrier = d_Barrier[optionIndex];
    const Real_t callPutFactor = call ? 1 : -1;
    Real_t sumCall;
    Real_t callValue = 0;

    //Cycle through the entire samples array:
    //derive end stock price for each path
    //accumulate partial integrals into intermediate shared memory buffer
    for(unsigned int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x){
        sumCall = static_cast<Real_t>(0.0);
        for(unsigned int i = iSum; i < pathN; i += SUM_N){
            callValue = endCallValue(S, X, MuByT, VBySqrtT, i, pathN, d_Samples, callPutFactor, depthN, optType, Barrier);
            sumCall += callValue;
        }
        s_SumCall[iSum] = sumCall;
    }

    sumReduce<SUM_N, THREAD_N>(s_SumCall);

    if(threadIdx.x == 0){
        d_CallValue[optionIndex] = s_SumCall[0];
    }
}

__global__ void MonteCarloReduce(
    Real_t *d_CallValue,
    Real_t *d_Buffer,
    int accumN)
{
    const int SUM_N = THREAD_N;
    __shared__ Real_t s_SumCall[SUM_N];
    Real_t *d_SumBase = &d_Buffer[blockIdx.x * accumN];

    for(int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x){
        Real_t sumCall = 0;
        for(int pos = iSum; pos < accumN; pos += SUM_N){
            sumCall += d_SumBase[pos];
        }
        s_SumCall[iSum] = sumCall;
    }

    if(threadIdx.x == 0){
        for(int i=1; i<SUM_N; i++) s_SumCall[0] += s_SumCall[i];
        d_CallValue[blockIdx.x] = s_SumCall[0];
    }
}


__device__ inline Real_t MoroInvCNDgpu(Real_t P){
    const Real_t a1 = static_cast<Real_t>(2.50662823884);
    const Real_t a2 = static_cast<Real_t>(-18.61500062529);
    const Real_t a3 = static_cast<Real_t>(41.39119773534);
    const Real_t a4 = static_cast<Real_t>(-25.44106049637);
    const Real_t b1 = static_cast<Real_t>(-8.4735109309);
    const Real_t b2 = static_cast<Real_t>(23.08336743743);
    const Real_t b3 = static_cast<Real_t>(-21.06224101826);
    const Real_t b4 = static_cast<Real_t>(3.13082909833);
    const Real_t c1 = static_cast<Real_t>(0.337475482272615);
    const Real_t c2 = static_cast<Real_t>(0.976169019091719);
    const Real_t c3 = static_cast<Real_t>(0.160797971491821);
    const Real_t c4 = static_cast<Real_t>(2.76438810333863E-02);
    const Real_t c5 = static_cast<Real_t>(3.8405729373609E-03);
    const Real_t c6 = static_cast<Real_t>(3.951896511919E-04);
    const Real_t c7 = static_cast<Real_t>(3.21767881768E-05);
    const Real_t c8 = static_cast<Real_t>(2.888167364E-07);
    const Real_t c9 = static_cast<Real_t>(3.960315187E-07);
    Real_t y, z;


    y = P - static_cast<Real_t>(0.5);
    if(_abs(y) < static_cast<Real_t>(0.42)){
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1);
    }else{
        if(y > 0)
            z = _log(-_log(1 - P));
        else
            z = _log(-_log(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if(y < 0) z = -z;
    }

    return z;
}


// d_Samples should be filled with Uniform pseudo-random or quasi-random samples in [0,1]
// Moro Inversion is used to convert Uniform to Normal[0,1]
__global__ void inverseCNDKernel(
    Real_t *d_Samples,
    unsigned int pathN)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int threadN = blockDim.x * gridDim.x;

    for(unsigned int pos = tid; pos < pathN; pos += threadN) {
        Real_t d = d_Samples[pos];
        d_Samples[pos] = MoroInvCNDgpu(d);
    }
}


static int iMonteCarloMethod(WGL_Memory_t priceRes, WGL_Memory_t spot, WGL_Memory_t strike, WGL_Memory_t muByT, WGL_Memory_t vBySqrtT, 
						  	 WGL_Memory_t barrier, WGL_Memory_t buffer, WGL_Memory_t samples, mint pathN, mint depthN, mint numOptions,
							 mint optType, mint callOrPut, mint blocksPerOption) {
	int err = LIBRARY_FUNCTION_ERROR;

	dim3 blockDim(256);
	dim3 gridDim1(blocksPerOption, numOptions);
	dim3 gridDim2(numOptions);
	dim3 moroBlockDim(128);
	dim3 moroGridDim(1);

	if (!(WGL_Type_RealQ(priceRes) && WGL_Type_RealQ(spot) && WGL_Type_RealQ(strike) &&
		  WGL_Type_RealQ(muByT) && WGL_Type_RealQ(vBySqrtT) &&
		  WGL_Type_RealQ(barrier) && WGL_Type_RealQ(samples))) {
		return LIBRARY_TYPE_ERROR;
	} else if (callOrPut != 1.0 && callOrPut != -1.0) {
		return LIBRARY_FUNCTION_ERROR;
	}

	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, priceRes, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, spot, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, strike, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, muByT, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, vBySqrtT, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, barrier, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, samples, wglErr), cleanup);
	
	inverseCNDKernel<<<moroGridDim, moroBlockDim>>>(
		CUDA_Runtime_getDeviceMemoryAsReal(samples),
		pathN*depthN
	);
	CUDA_Runtime_synchronize(wglErr);

	if (blocksPerOption != 1) {
		if (!WGL_Type_RealQ(buffer)) {
			return LIBRARY_TYPE_ERROR;
		}
	
		WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, buffer, wglErr), cleanup);
		MonteCarloKernel<<<gridDim1, blockDim>>>(
			CUDA_Runtime_getDeviceMemoryAsReal(spot),
			CUDA_Runtime_getDeviceMemoryAsReal(strike),
			CUDA_Runtime_getDeviceMemoryAsReal(muByT),
			CUDA_Runtime_getDeviceMemoryAsReal(vBySqrtT),
			CUDA_Runtime_getDeviceMemoryAsReal(barrier),
			CUDA_Runtime_getDeviceMemoryAsReal(buffer),
			CUDA_Runtime_getDeviceMemoryAsReal(samples),
			pathN,
			depthN,
			optType,
			callOrPut
		);
		MonteCarloReduce<<<gridDim2, blockDim>>>(
			CUDA_Runtime_getDeviceMemoryAsReal(priceRes),
			CUDA_Runtime_getDeviceMemoryAsReal(buffer),
			256 * blocksPerOption
		);
	} else {
		MonteCarloOneBlockPerOption<<<gridDim2, blockDim>>>(
			CUDA_Runtime_getDeviceMemoryAsReal(priceRes),
			CUDA_Runtime_getDeviceMemoryAsReal(spot),
			CUDA_Runtime_getDeviceMemoryAsReal(strike),
			CUDA_Runtime_getDeviceMemoryAsReal(muByT),
			CUDA_Runtime_getDeviceMemoryAsReal(vBySqrtT),
			CUDA_Runtime_getDeviceMemoryAsReal(barrier),
			CUDA_Runtime_getDeviceMemoryAsReal(samples),
			pathN,
			depthN,
			optType,
			callOrPut
		);
	}
	CUDA_Runtime_synchronize(wglErr);
	if (WGL_SuccessQ) {
		err = LIBRARY_NO_ERROR;
	}
		
cleanup:
	if (WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, priceRes, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, priceRes, wglErr);
	}
	
	CUDA_Runtime_unsetMemoryAsInput(wglState, spot, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, strike, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, muByT, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, vBySqrtT, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, barrier, wglErr);
	CUDA_Runtime_unsetMemoryAsInput(wglState, samples, wglErr);
	
	if (blocksPerOption != 1) {
		CUDA_Runtime_unsetMemoryAsInput(wglState, buffer, wglErr);
	}
	
	return err;
}


EXTERN_C DLLEXPORT int oMonteCarloMethod(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t resMem, currentPriceMem, strikeMem, expirationMem, interestMem, volatilityMem, dividendMem, barrierMem;
	mint resMemId, currentPriceMemId, strikeMemId, expirationMemId, interestMemId, volatilityMemId, dividendMemId, barrierMemId;
	mint numOptions, optType, callOrPut;
	mint pathN = 10000, depthN = 500, threadN = 256, blocksPerOption = 1;
	Real_t * muByT = NULL, * vBySqrtT = NULL, * hBuffer = NULL, * hSamples = NULL;
	WGL_Memory_t callBufferMem = NULL, muByTMem = NULL, vBySqrtTMem = NULL, samplesMem = NULL;
	double * hExpiration = NULL, * hVolatility = NULL, * hInterest = NULL, * hDividend = NULL, * hRes = NULL, * RT = NULL;
	
	int err = LIBRARY_NO_ERROR;
	
	assert(Argc == 13);
	
	resMemId					= MArgument_getInteger(Args[0]);
	currentPriceMemId			= MArgument_getInteger(Args[1]);
	strikeMemId					= MArgument_getInteger(Args[2]);
	expirationMemId				= MArgument_getInteger(Args[3]);
	interestMemId				= MArgument_getInteger(Args[4]);
	volatilityMemId				= MArgument_getInteger(Args[5]);
	dividendMemId				= MArgument_getInteger(Args[6]);
	barrierMemId				= MArgument_getInteger(Args[7]);
	numOptions 					= MArgument_getInteger(Args[8]);
	optType						= MArgument_getInteger(Args[9]);
	callOrPut					= MArgument_getInteger(Args[10]);
	
	resMem						= wglData->findMemory(wglData, resMemId);
	currentPriceMem				= wglData->findMemory(wglData, currentPriceMemId);
	strikeMem					= wglData->findMemory(wglData, strikeMemId);
	expirationMem				= wglData->findMemory(wglData, expirationMemId);
	interestMem					= wglData->findMemory(wglData, interestMemId);
	volatilityMem				= wglData->findMemory(wglData, volatilityMemId);
	dividendMem					= wglData->findMemory(wglData, dividendMemId);
	barrierMem					= wglData->findMemory(wglData, barrierMemId);
	
	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	New(muByT, Real_t, numOptions * sizeof(Real_t));
	New(vBySqrtT, Real_t, numOptions * sizeof(Real_t));
	New(hSamples, Real_t, pathN * depthN * sizeof(Real_t));
	
	hExpiration = wglData->MTensorMemory_getRealData(wglData, expirationMem);
	assert(hExpiration != NULL);
	
	hVolatility = wglData->MTensorMemory_getRealData(wglData, volatilityMem);
	assert(hVolatility != NULL);
	
	hInterest = wglData->MTensorMemory_getRealData(wglData, interestMem);
	assert(hInterest != NULL);
	
	hDividend = wglData->MTensorMemory_getRealData(wglData, dividendMem);
	assert(hDividend != NULL);
	
	// The only inputs we really need for simulation are the mean (muByT) and standard deviation (vBySqrtT) for the normal random variables used to simulate brownian motion.
	for (mint ii = 0; ii < numOptions; ii++) {
		muByT[ii] = static_cast<Real_t>((hInterest[ii] - hDividend[ii] - 0.5f * hVolatility[ii]*hVolatility[ii]) * hExpiration[ii]);
		vBySqrtT[ii] = static_cast<Real_t>(hVolatility[ii] * sqrt(hExpiration[ii]));
	}
	
	// Create uniform random variables host-side, use a kernel to convert them to Normal(0,1).
	for (mint ii = 0; ii < pathN * depthN; ii++) {
		hSamples[ii] = static_cast<Real_t>(rand()) / static_cast<Real_t>(RAND_MAX);
	}
	
	// This determines how many blocks per option to use; it could probably be updated since I pulled it from the nvidia SDK 
	// which did not do "real" monte carlo method (paths), so depthN was not a factor.
	if (pathN / numOptions >= 8192) {
		blocksPerOption = numOptions < 16 ? 64 : 16;
		New(hBuffer, Real_t, blocksPerOption * threadN * numOptions * sizeof(Real_t));
		callBufferMem = wglData->newRawMemory(wglData, (void**)&hBuffer, WGL_MemoryResidence_DeviceHost, blocksPerOption * threadN * numOptions * sizeof(Real_t), True);
		callBufferMem->type = WGL_Real_t;
		assert(WGL_SuccessQ);
	}
	
	muByTMem = wglData->newRawMemory(wglData, (void**)&muByT, WGL_MemoryResidence_DeviceHost, numOptions * sizeof(Real_t), True);
	muByTMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);
	
	vBySqrtTMem = wglData->newRawMemory(wglData, (void**)&vBySqrtT, WGL_MemoryResidence_DeviceHost, numOptions * sizeof(Real_t), True);
	vBySqrtTMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);
	
	samplesMem = wglData->newRawMemory(wglData, (void**)&hSamples, WGL_MemoryResidence_DeviceHost, pathN * depthN * sizeof(Real_t), True);
	samplesMem->type = WGL_Real_t;
	assert(WGL_SuccessQ);
	
	err = iMonteCarloMethod(resMem, currentPriceMem, strikeMem, muByTMem, vBySqrtTMem, barrierMem, callBufferMem, samplesMem, pathN, depthN, numOptions, optType, callOrPut, blocksPerOption);
	
	hRes = wglData->MTensorMemory_getRealData(wglData, resMem);
	assert(hRes != NULL);
	
	New(RT, double, numOptions * sizeof(double));
	
	// The output of the kernel does not average of the number of paths or adjust for inflation; we do that now.
	for (int ii = 0; ii < numOptions; ii++) {
		RT[ii] = exp(-hInterest[ii]*hExpiration[ii]);
		hRes[ii] *= RT[ii] / static_cast<Real_t>(pathN);
	}

cleanup:	
	
	Free(RT);
	
	wglData->freeMemory(wglData, callBufferMem);
	wglData->freeMemory(wglData, muByTMem);
	wglData->freeMemory(wglData, vBySqrtTMem);
	wglData->freeMemory(wglData, samplesMem);
		
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else if (err != LIBRARY_NO_ERROR) {
		return err;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}


WGLEXPORT int WolframGPULibrary_initialize(WolframGPULibraryData wglData0) {
	wglData = wglData0;
	return LIBRARY_NO_ERROR;
}

WGLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
   return LIBRARY_NO_ERROR;
}

WGLEXPORT void WolframLibrary_uninitialize( ) {
   return;
}


