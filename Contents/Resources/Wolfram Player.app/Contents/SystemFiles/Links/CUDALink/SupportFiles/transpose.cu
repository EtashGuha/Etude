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
 
#define BLOCKDIM    16


__global__ void transpose_kernel(Real_t * idata, Real_t * odata, int width, int height) {
	__shared__ Real_t block[BLOCKDIM][BLOCKDIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCKDIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCKDIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height)) {
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCKDIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCKDIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width)) {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}


