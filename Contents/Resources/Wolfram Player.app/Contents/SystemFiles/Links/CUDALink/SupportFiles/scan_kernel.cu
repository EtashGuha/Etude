/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */
 
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

//All three kernels run 512 threads per workgroup
//Must be a power of two
#define THREADBLOCK_SIZE 256
typedef unsigned int uint;


#define T int
#define T4 int4


#ifndef IDENTITY
#define IDENTITY 0
#endif


enum {ADD_OP=1,MUL_OP,MAX_OP,MIN_OP};

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

inline __device__ T scan_sum(T t1,T t2)
{
	return t1+t2;
}

inline __device__ T scan_max(T t1,T t2)
{
	return t1>=t2?t1:t2;
}

inline __device__ T scan_op(T t1,T t2, uint oper)
{
	T res;
	switch(oper)
	{
		case ADD_OP:res = t1+t2;
					break;
					
		case MUL_OP:res = t1*t2;
					break;	
					
		case MAX_OP:res = max(t1,t2);
					break;	
					
		case MIN_OP:res = min(t1,t2);
					break;		
					
				
	}
	return res;
}

//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE
inline __device__ T warpScanInclusive(T idata, T *s_Data, uint size,T* warpEndResult, uint oper, T Identity){
    int pos = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE -1 ));
    s_Data[pos] = Identity;
    pos += WARP_SIZE;
    s_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] = scan_op(s_Data[pos],s_Data[pos - offset], oper);

*warpEndResult=s_Data[pos];
    return s_Data[pos-1];
}

inline __device__ T scan1Inclusive(T idata, T *s_Data, uint size, uint oper, T Identity){

 T warpEndResult;
    if(size > WARP_SIZE){
        //Bottom-level inclusive warp scan

   
        T warpResult = warpScanInclusive(idata, s_Data, WARP_SIZE,&warpEndResult,oper,Identity);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            s_Data[threadIdx.x >> LOG2_WARP_SIZE] = warpEndResult;

        //wait for warp scans to complete
        __syncthreads();
        if( threadIdx.x < (THREADBLOCK_SIZE / WARP_SIZE) ){
            //grab top warp elements
            T val = s_Data[threadIdx.x];
            //calculate exclsive scan and write back to shared memory
            s_Data[threadIdx.x] = warpScanInclusive(val, s_Data, size >> LOG2_WARP_SIZE,&warpEndResult,oper,Identity);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        return scan_op(warpResult , s_Data[threadIdx.x >> LOG2_WARP_SIZE], oper);
	
    }else{
        return warpScanInclusive(idata, s_Data, size,&warpEndResult,oper,Identity);
    }
}

inline __device__ T4 scan4Inclusive(T4 idata4, T *s_Data, uint size,uint oper, T Identity){
    //Level-0 inclusive scan
    idata4.y =scan_op(idata4.y, idata4.x,oper);
    idata4.z =scan_op(idata4.z, idata4.y,oper);
    idata4.w = scan_op(idata4.w,idata4.z,oper);

    //Level-1 exclusive scan
    T oval = scan1Inclusive(idata4.w, s_Data, size / 4,oper,Identity);

    idata4.x = scan_op(idata4.x,oval,oper);
    idata4.y = scan_op(idata4.y,oval,oper);
    idata4.z = scan_op(idata4.z,oval,oper);
    idata4.w = scan_op(idata4.w,oval,oper);

    return idata4;
}


////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(
    T4 *d_Dst,
    T4 *d_Src,
    uint size,
    uint oper, T Identity
){
    __shared__ T s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    T4 idata4 = d_Src[pos];

    //Calculate exclusive scan
    T4 odata4 = scan4Inclusive(idata4, s_Data, size,oper,Identity);

    //Write back
    d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(
    T *d_Buf,
    T *d_Dst,
    T *d_Src,
    uint N,
    uint arrayLength,
    uint oper, T Identity
){
    __shared__ T s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    T idata = 0;
    if(pos < N)
        idata = 
        d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] ;

    //Compute
    T odata = scan1Inclusive(idata, s_Data, arrayLength,oper,Identity);

    //Avoid out-of-bound access
    if(pos < N)
        d_Buf[pos] = odata;
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(
    T4 *d_Data,
    T *d_Buffer,
    uint oper, T Identity
){
    __shared__ T buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x == 0)
        buf = d_Buffer[blockIdx.x];
    __syncthreads();

    T4 data4 = d_Data[pos];
    data4.x = scan_op(data4.x,buf,oper);
    data4.y = scan_op(data4.y,buf,oper);
    data4.z = scan_op(data4.z,buf,oper);
    data4.w = scan_op(data4.w,buf,oper );
    d_Data[pos] = data4;
}

