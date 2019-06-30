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
 
#include <cuda.h>
 
#ifndef nIsPow2
#define nIsPow2 false
#endif

#ifndef blockSize
#define blockSize 256
#endif

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif
 
 
__device__ inline Real_t norm2(Real_t x, Real_t y) {
	return x*x + y*y;
}

__global__ void countInCircle(Real_t *g_idata, int *g_odata, unsigned int n)
{
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.

     __shared__ volatile int sdata[2 * 256];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
     unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    int mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += norm2(g_idata[2*i], g_idata[2*i + 1]) > 1 ? 0 : 1;
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
        	 mySum += norm2(g_idata[2*(i+blockSize)], g_idata[2*(i+blockSize) + 1]) > 1 ? 0 : 1;
            
        i += gridSize;
    }

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();

   // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] = mySum = mySum + sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] = mySum = mySum + sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] = mySum = mySum + sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] = mySum = mySum + sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] = mySum = mySum + sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] = mySum = mySum + sdata[tid +  1]; EMUSYNC; }
    }

       
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];

}