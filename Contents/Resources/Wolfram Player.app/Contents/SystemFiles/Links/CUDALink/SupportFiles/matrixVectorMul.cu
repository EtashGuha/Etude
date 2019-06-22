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
 
#define BLOCKDIM 16


__global__ void matrixVecMul_kernel(mint * out, mint  * mat, mint  * vec, mint  matWidth, mint  matHeight) {
    int ty = threadIdx.y, by = blockIdx.y;
    int tx = threadIdx.x, bx = blockIdx.x;

    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    int vec_index = xIndex;

#if 0  // atomic operators are only supported on compute 1.1 and above

    __shared__ mint smem[BLOCKDIM];

    int mat_index = xIndex + yIndex*matWidth;
    
    if (ty == 0)
        smem[tx] = vec[vec_index];

    __syncthreads();

    if (xIndex < matWidth && yIndex < matHeight) {
        atomicAdd(&out[vec_index], mat[mat_index]*smem[tx]);
    }
#else
    mint * mat_row = &mat[xIndex*matWidth];
    int ii;
    mint accum;
    
    if (xIndex < matHeight) {
        for (ii = 0, accum = 0; ii < matWidth; ii++)
            accum += mat_row[ii]*vec[ii];
        out[vec_index] = accum;
    }
#endif
}


