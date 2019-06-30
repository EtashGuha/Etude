
#define DBL_EPSILON    2.2204460492503131e-016

/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/

 
#ifdef USING_DOUBLE_PRECISIONQ
#define _sqrt(x)		sqrt(static_cast<Real_t>(x))
#define _log(x)			log(static_cast<Real_t>(x))
#define _sin(x)			sin(static_cast<Real_t>(x))
#define _cos(x)			cos(static_cast<Real_t>(x))
#define MUL(a, b) 		((a) * (b))
#else
#define MUL(a, b) 		__umul24(a, b)
#define _sqrt(x)		sqrtf(static_cast<Real_t>(x))
#define _log(x)			logf(static_cast<Real_t>(x))
#define _sin(x)			__sinf(static_cast<Real_t>(x))
#define _cos(x)			__cosf(static_cast<Real_t>(x))
#endif


/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/


#define MAX_ITER 		10 
#define m   			2147483647
#define A   			16807

__device__ int middleSquareMethod(unsigned int rnd) {
    int ii;
    
    #pragma unroll
    for (ii = 0; ii < MAX_ITER; ii++) {
        rnd = (rnd*rnd) >> (8*sizeof(unsigned int)/2 - 1);
    }
    return rnd; 
}

__global__ void ParkMiller(mint * seeds, mint * out, mint width) {
    __shared__ int blockSeed;
    const int tx = threadIdx.x, bx = blockIdx.x, dx = blockDim.x;
    const int index = tx + bx*dx;
    int ii;
    unsigned int seed, rnd;
    
    if (index >= width)
        return ;

    if (tx == 0)
        blockSeed = seeds[bx];
    __syncthreads();

    //seed = middleSquareMethod(blockSeed + tx);
    seed = blockSeed + tx + bx;

    #pragma unroll
    for (ii = 0, rnd = seed; ii < MAX_ITER; ii++)
        rnd = (rnd * A) % m;
    out[index] = rnd;
}


/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/

__device__ int primes[] = {
                            2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
                            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                            73, 79, 83, 89, 97,101,103,107,109,113,
                            127,131,137,139,149,151,157,163,167,173,
                            179,181,191,193,197,199,211,223,227,229,
                          };


__global__ void Halton(Real_t * out, mint dim, mint n) {
    const int tx = threadIdx.x, bx = blockIdx.x, dx = blockDim.x;
    const int index = tx + bx*dx;
    
    if (index >= n)
    	return ;
    
    int ii;
    double digit, rnd, idx, half;

    for (ii = 0, idx=index, rnd=0, digit=0; ii < dim; ii++) {
        half = 1.0/primes[ii];
        while (idx > DBL_EPSILON) {
            digit = ((int)idx)%primes[ii];
            rnd += half*digit;
            idx = (idx - digit)/primes[ii];
            half /= primes[ii];
        }
        out[index*dim + ii] = rnd;
    }
}


/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/
/* Based on:
Fahad Zafar, Aaron Curtis and Marc Olano, "GPU Random Numbers via the Tiny Encryption Algorithm",
   HPG 2010: Proceedings of the ACM SIGGRAPH/Eurographics Symposium on High Performance Graphics,
   (Saarbrucken, Germany, June 25-27, 2010).
*/

#ifndef DIM
#define DIM 	256
#endif /* DIM */

#ifndef ROUNDS
#define ROUNDS	8
#endif /* ROUNDS */

#define V0  	smem[threadIdx.x]
#define V1  	smem[threadIdx.x + DIM]

#define K0 		0xA341316C
#define K1 		0xC8013EA4
#define K2 		0xAD90777D
#define K3  	0x7E95761E
#define DELTA	0x9E3779b9

__global__ void Tea(mint * V, mint length) {
   __shared__ unsigned int smem[2*DIM + 1];
   const int xIndex = threadIdx.x + 2 * blockIdx.x * blockDim.x;
   if (xIndex < length) {
       unsigned int sum = xIndex;
       V0 = 0;
       V1 = 0; 
# pragma unroll
       for (int ii = 0; ii < ROUNDS; ii++) {
          sum += DELTA;
          V0 += ((V1 << 4) + K0)^(V1 + sum)^((V1 >> 5) + K1);
          V1 += ((V0 << 4) + K2)^(V0 + sum)^((V0 >> 5) + K3);
       }
       V[xIndex] = V0;
       V[xIndex + DIM] = V1;
   }
}


/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/

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

////////////////////////////////////////////////////////////////////////////////
// Moro's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
__device__ inline Real_t MoroInvCNDgpu(Real_t P){
    const Real_t a1 = 2.50662823884f;
    const Real_t a2 = -18.61500062529f;
    const Real_t a3 = 41.39119773534f;
    const Real_t a4 = -25.44106049637f;
    const Real_t b1 = -8.4735109309f;
    const Real_t b2 = 23.08336743743f;
    const Real_t b3 = -21.06224101826f;
    const Real_t b4 = 3.13082909833f;
    const Real_t c1 = 0.337475482272615f;
    const Real_t c2 = 0.976169019091719f;
    const Real_t c3 = 0.160797971491821f;
    const Real_t c4 = 2.76438810333863E-02f;
    const Real_t c5 = 3.8405729373609E-03f;
    const Real_t c6 = 3.951896511919E-04f;
    const Real_t c7 = 3.21767881768E-05f;
    const Real_t c8 = 2.888167364E-07f;
    const Real_t c9 = 3.960315187E-07f;
    Real_t y, z;

    if(P <= 0 || P >= static_cast<Real_t>(1.0))
        return __int_as_float(0x7FFFFFFF);

    y = P - static_cast<Real_t>(0.5);
    if(fabsf(y) < static_cast<Real_t>(0.42)){
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + static_cast<Real_t>(1.0));
    }else{
        if(y > 0)
            z = _log(-_log(static_cast<Real_t>(1.0) - P));
        else
            z = _log(-_log(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if(y < 0) z = -z;
    }

    return z;
}


////////////////////////////////////////////////////////////////////////////////
// Acklam's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
__device__ inline Real_t AcklamInvCNDgpu(Real_t P){
    const Real_t   a1 = -39.6968302866538f;
    const Real_t   a2 = 220.946098424521f;
    const Real_t   a3 = -275.928510446969f;
    const Real_t   a4 = 138.357751867269f;
    const Real_t   a5 = -30.6647980661472f;
    const Real_t   a6 = 2.50662827745924f;
    const Real_t   b1 = -54.4760987982241f;
    const Real_t   b2 = 161.585836858041f;
    const Real_t   b3 = -155.698979859887f;
    const Real_t   b4 = 66.8013118877197f;
    const Real_t   b5 = -13.2806815528857f;
    const Real_t   c1 = -7.78489400243029E-03f;
    const Real_t   c2 = -0.322396458041136f;
    const Real_t   c3 = -2.40075827716184f;
    const Real_t   c4 = -2.54973253934373f;
    const Real_t   c5 = 4.37466414146497f;
    const Real_t   c6 = 2.93816398269878f;
    const Real_t   d1 = 7.78469570904146E-03f;
    const Real_t   d2 = 0.32246712907004f;
    const Real_t   d3 = 2.445134137143f;
    const Real_t   d4 = 3.75440866190742f;
    const Real_t  low = 0.02425f;
    const Real_t high = 1.0f - low;
    Real_t z, R;

    if(P <= 0 || P >= static_cast<Real_t>(1.0)) {
#ifdef CUDALINK_USING_DOUBLE_PRECISIONQ
		return __longlong_as_double(0xFFF8000000000000ULL);
#else
        return __int_as_float(0x7FFFFFFF);
#endif
	}

    if(P < low){
        z = _sqrt(-static_cast<Real_t>(2.0) * _log(P));
        z = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) /
            ((((d1 * z + d2) * z + d3) * z + d4) * z + static_cast<Real_t>(1.0));
    }else{
        if(P > high){
            z = _sqrt(-static_cast<Real_t>(2.0) * _log(static_cast<Real_t>(1.0) - P));
            z = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) /
                 ((((d1 * z + d2) * z + d3) * z + d4) * z + static_cast<Real_t>(1.0));
        }else{
            z = P - static_cast<Real_t>(0.5);
            R = z * z;
            z = (((((a1 * R + a2) * R + a3) * R + a4) * R + a5) * R + a6) * z /
                (((((b1 * R + b2) * R + b3) * R + b4) * R + b5) * R + static_cast<Real_t>(1.0));
        }
    }

    return z;
}

////////////////////////////////////////////////////////////////////////////////
// Main kernel. Choose between transforming
// input sequence and uniform ascending (0, 1) sequence
////////////////////////////////////////////////////////////////////////////////

__global__ void InverseCND(
    Real_t *d_Output,
    mint pathN,
    mint useInput
){
    Real_t q = static_cast<Real_t>(1.0) / static_cast<Real_t>(pathN + 1);
    unsigned int     tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int threadN = MUL(blockDim.x, gridDim.x);
    
    if (useInput) {
    	for (unsigned int pos = tid; pos < pathN; pos += threadN) {
    		d_Output[pos] = static_cast<Real_t>(MoroInvCNDgpu(d_Output[pos]));
    	}
    }
    else {
	    for(unsigned int pos = tid; pos < pathN; pos += threadN){
	        Real_t d = (Real_t)(pos + 1) * q;
	        d_Output[pos] = static_cast<Real_t>(MoroInvCNDgpu(d));
	    }
	}
}

/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/


#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////
//static __constant__ unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];

__global__ void Niederreiter(
    mint *c_Table,
    Real_t *d_Output,
    mint seed,
    mint N
){
    mint *dimBase = &c_Table[threadIdx.y * QRNG_RESOLUTION + 0];
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);

    for(unsigned int pos = tid; pos < N; pos += threadN){
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for(int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if(data & 1) result ^= dimBase[bit];

        d_Output[MUL(threadIdx.y, N) + pos] = static_cast<Real_t>(result + 1) * INT_SCALE;
    }
}



/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/

// Number of direction vectors is fixed to 32
#define n_directions 32

 /*
 * Portions Copyright (c) 1993-2010 NVIDIA Corporation.  All rights reserved.
 * Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
 * Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
 *
 * Sobol Quasi-random Number Generator example
 *
 * Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
 * http://people.maths.ox.ac.uk/~gilesm/
 *
 * and C code developed by Stephen Joe, University of Waikato, New Zealand
 * and Frances Kuo, University of New South Wales, Australia
 * http://web.maths.unsw.edu.au/~fkuo/sobol/
 *
 * For theoretical background see:
 *
 * P. Bratley and B.L. Fox.
 * Implementing Sobol's quasirandom sequence generator
 * http://portal.acm.org/citation.cfm?id=42288
 * ACM Trans. on Math. Software, 14(1):88-100, 1988
 *
 * S. Joe and F. Kuo.
 * Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
 * http://portal.acm.org/citation.cfm?id=641879
 * ACM Trans. on Math. Software, 29(1):49-57, 2003
 *
 */

#ifdef CUDALINK_USING_DOUBLE_PRECISIONQ
#define k_2powneg32 2.3283064E-10 
#else
#define k_2powneg32 2.3283064E-10F
#endif

__global__ void Sobol(mint n_vectors, mint n_dimensions, mint *d_directions, Real_t *d_output)
{
    __shared__ unsigned int v[n_directions];

    // Offset into the correct dimension as specified by the
    // block y coordinate
    d_directions = d_directions + n_directions * blockIdx.y;
    d_output = d_output +  n_vectors * blockIdx.y;

    // Copy the direction numbers for this dimension into shared
    // memory - there are only 32 direction numbers so only the
    // first 32 (n_directions) threads need participate.
    if (threadIdx.x < n_directions)
    {
	    v[threadIdx.x] = d_directions[threadIdx.x];
    }
    __syncthreads();

    // Set initial index (i.e. which vector this thread is
    // computing first) and stride (i.e. step to the next vector
    // for this thread)
    int i0     = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // Get the gray code of the index
    // c.f. Numerical Recipes in C, chapter 20
    // http://www.nrbook.com/a/bookcpdf/c20-2.pdf
    unsigned int g = i0 ^ (i0 >> 1);

    // Initialisation for first point x[i0]
    // In the Bratley and Fox paper this is equation (*), where
    // we are computing the value for x[n] without knowing the
    // value of x[n-1].
    unsigned int X = 0;
    unsigned int mask;
    for (unsigned int k = 0 ; k < __ffs(stride) - 1 ; k++)
    {
        // We want X ^= g_k * v[k], where g_k is one or zero.
        // We do this by setting a mask with all bits equal to
        // g_k. In reality we keep shifting g so that g_k is the
        // LSB of g. This way we avoid multiplication.
        mask = - (g & 1);
        X ^= mask & v[k];
        g = g >> 1;
    }
    if (i0 < n_vectors)
    {
        d_output[i0] = (float)X * k_2powneg32;
    }

    // Now do rest of points, using the stride
    // Here we want to generate x[i] from x[i-stride] where we
    // don't have any of the x in between, therefore we have to
    // revisit the equation (**), this is easiest with an example
    // so assume stride is 16.
    // From x[n] to x[n+16] there will be:
    //   8 changes in the first bit
    //   4 changes in the second bit
    //   2 changes in the third bit
    //   1 change in the fourth
    //   1 change in one of the remaining bits
    //
    // What this means is that in the equation:
    //   x[n+1] = x[n] ^ v[p]
    //   x[n+2] = x[n+1] ^ v[q] = x[n] ^ v[p] ^ v[q]
    //   ...
    // We will apply xor with v[1] eight times, v[2] four times,
    // v[3] twice, v[4] once and one other direction number once.
    // Since two xors cancel out, we can skip even applications
    // and just apply xor with v[4] (i.e. log2(16)) and with
    // the current applicable direction number.
    // Note that all these indices count from 1, so we need to
    // subtract 1 from them all to account for C arrays counting
    // from zero.
    unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
    unsigned int v_stridemask = stride - 1;
    for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
    {
        // x[i] = x[i-stride] ^ v[b] ^ v[c]
        //  where b is log2(stride) minus 1 for C array indexing
        //  where c is the index of the rightmost zero bit in i,
        //  not including the bottom log2(stride) bits, minus 1
        //  for C array indexing
        // In the Bratley and Fox paper this is equation (**)
        X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
        d_output[i] = static_cast<Real_t>(X * k_2powneg32);
    }
}

/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/
#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18


////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of nPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void MersenneTwister(Real_t* d_Random, mint * ds_matrix_a, mint * ds_mask_b, mint * ds_mask_c, mint * ds_seed, mint NPerRng) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	int iState, iState1, iStateM, iOut;
	unsigned int mti, mti1, mtiM, x;
	unsigned int mt[MT_NN];

	for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N) {
		unsigned int matrix_a, mask_b, mask_c, seed;
		matrix_a = ds_matrix_a[iRng];
		mask_b = ds_mask_b[iRng];
		mask_c = ds_mask_c[iRng];
		seed = ds_seed[iRng];
		mt[0] = seed;
		for(iState = 1; iState < MT_NN; iState++) {
			mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;
		}

		iState = 0;
		mti1 = mt[0];
		for(iOut = 0; iOut < NPerRng; iOut++) {
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & mask_b;
			x ^= (x << MT_SHIFTC) & mask_c;
			x ^= (x >> MT_SHIFT1);

			d_Random[iRng + iOut * MT_RNG_COUNT] = static_cast<Real_t>(x + 1.0)/static_cast<Real_t>(4294967296.0);
		}
	}
}



/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of nPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// nPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI static_cast<Real_t>(3.14159265358979)
__device__ inline void iBoxMuller(Real_t& u1, Real_t& u2){
    Real_t   r = _sqrt(-2.0 * _log(u1));
    Real_t phi = 2 * PI * u2;
    u1 = r * _cos(phi);
    u2 = r * _sin(phi);
}

__global__ void BoxMuller(Real_t *d_Random, mint nPerRng){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int iOut = 0; iOut < nPerRng; iOut += 2)
        iBoxMuller(
                d_Random[tid + (iOut + 0) * MT_RNG_COUNT],
                d_Random[tid + (iOut + 1) * MT_RNG_COUNT]
                );
}


/***********************************************************************************************/
/***********************************************************************************************/
/***********************************************************************************************/
