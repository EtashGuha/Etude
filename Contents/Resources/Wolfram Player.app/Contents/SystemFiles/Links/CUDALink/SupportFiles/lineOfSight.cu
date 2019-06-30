 
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


#ifndef TP 
#define TP 0
#endif

#if(!TP)

#define T float
#define T4 float4


#else

#define T int
#define T4 int4

#endif

#ifndef IDENTITY
#define IDENTITY 0
#endif


//#define T float
//#define T4 float4

#ifndef SCAN_OP
#define SCAN_OP scan_sum
#endif

enum {ADD_OP=1,MUL_OP,MAX_OP,MIN_OP};

//#define IDENTITY 0

////////////////////////////////////////////////////////////////////////////////
// Basic ccan codelets
////////////////////////////////////////////////////////////////////////////////
/*
#if(0)
    //Naive inclusive scan: O(N * log2(N)) operations
    //Allocate 2 * 'size' local memory, initialize the first half
    //with 'size' zeros avoiding if(pos >= offset) condition evaluation
    //and saving instructions
    inline __device__ T scan1Inclusive(T idata, T *s_Data, T size){
        T pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
        s_Data[pos] = 0;
        pos += size;
        s_Data[pos] = idata;

        for(T offset = 1; offset < size; offset <<= 1){
            __syncthreads();
            T t = s_Data[pos] + s_Data[pos - offset];
            __syncthreads();
            s_Data[pos] = t;
        }

        return s_Data[pos];
    }

    inline __device__ T scan1Exclusive(T idata, T *s_Data, T size){
        return scan1Inclusive(idata, s_Data, size) - idata;
    }

#else
*/

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

	inline __device__ float fun_s(float x,float y)
	{
		return (float)((float)1.0 + sinf(x)*sinf(y) + .3*sinf(2*x)*sinf(2*y) + .2*sinf(5*x)*sinf(5*y));
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

/*
    inline __device__ T warpScanExclusive(T idata, T *s_Data, T size){
        return warpScanInclusive(idata, s_Data, size) - idata;
    }
*/
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
    
/*
    inline __device__ T scan1Exclusive(T idata, T *s_Data, uint size){
        return scan1Inclusive(idata, s_Data, size) - idata;
    }

#endif
*/

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

/*
//Exclusive vector scan: the array to be scanned is stored
//in local thread memory scope as T4
inline __device__ T4 scan4Exclusive(T4 idata4, T *s_Data, uint size){
    T4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}
*/

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

__global__ void scanExclusiveSharedw(
    T4 *d_Dst,
    T4 *d_Src,
    uint size,
    uint oper, T Identity,
    T angle, T ang2, int pt0,int pt1 , int numpoints,float incr,int range,int subblk,float h
){
    __shared__ T s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	uint index=4*pos+1;
	
	T4 r1;
	
	r1=d_Src[(index-1)/4];
		
	r1.x=(-500.0+index);
	r1.y=(-500.0+index+1);
	r1.z=(-500.0+index+2);
	r1.w=(-500.0+index+3);
	
	float x=0.0;
	float y=0.0;
	
	//pt0=numpoints/2-1;
	//pt1=numpoints/2-1;

		int i=0;
		x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
		y=(float)(pt1 + 1  + (index+i))*incr;

		r1.x=atan2f(fun_s(x,y)-h,sqrtf((index+i)*(index+i) + powf(roundf(((index+i)/tanf(angle))),2))*incr)*57.2884;
	
		i=1;
		x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
		y=(float)(pt1 + 1  + (index+i))*incr;
		r1.y=atan2f(fun_s(x,y)-h,sqrtf((index+i)*(index+i) + powf(roundf(((index+i)/tanf(angle))),2))*incr)*57.2884;


		i=2;
		x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
		y=(float)(pt1 + 1  + (index+i))*incr;
		
		r1.z=atan2f(fun_s(x,y)-h,sqrtf((index+i)*(index+i) + powf(roundf(((index+i)/tanf(angle))),2))*incr)*57.2884;

		i=3;
		x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
		y=(float)(pt1 + 1  + (index+i))*incr;
		
		r1.w=atan2f(fun_s(x,y)-h,sqrtf((index+i)*(index+i) + powf(roundf(((index+i)/tanf(angle))),2))*incr)*57.2884;
		
	/*	
		r1.x=(float)pt0;
		r1.y=(float)pt1;
		r1.z=(float)pt0;
		r1.w=(float)pt1;
		*/
		
    //Load data
    T4 idata4 = r1;

	d_Src[(index-1)/4]=r1;

    //Calculate exclusive scan
    T4 odata4 = scan4Inclusive(idata4, s_Data, size,oper,Identity);

    //Write back
    d_Dst[(index-1)/4] = odata4;
}


__global__ void scanExclusiveShared4(
    T4 *d_Dst,
    T4 *d_Src,
    uint size,
    uint oper, T Identity,
    T angle, T ang2, int pt0,int pt1 , int numpoints,int flag,float incr,int range,int subblk,float h
){
    __shared__ T s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	uint index=4*pos+1;
	
	T4 r1;
	
	r1=d_Src[(index-1)/4];
		
	r1.x=(-500.0+index);
	r1.y=(-500.0+index+1);
	r1.z=(-500.0+index+2);
	r1.w=(-500.0+index+3);
	
	float x=0.0;
	float y=0.0;
	
	//pt0=numpoints/2-1;
	//pt1=numpoints/2-1;
	int i=0;
	switch(flag)
	{
	
		case 1:i=0;
				x=(pt0 + 1 + index +i)*incr;
				y=(pt1 + 1 +(int)roundf((index+i)*tanf(angle)))*incr;
			
				r1.x=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;		
	
				i=1;
				x=(pt0 + 1 + index +i)*incr;
				y=(pt1 + 1 +(int)roundf((index+i)*tanf(angle)))*incr;
				
				r1.y=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;		

				i=2;
				x=(pt0 + 1 + index +i)*incr;
				y=(pt1 + 1 +(int)roundf((index+i)*tanf(angle)))*incr;
			
				r1.z=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;		

				i=3;
				x=(pt0 + 1 + index +i)*incr;
				y=(pt1 + 1 +(int)roundf((index+i)*tanf(angle)))*incr;
				
				r1.w=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;		
				break;
	
	case 2:
			i=0;
			x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
			y=(float)(pt1 + 1  + (index+i))*incr;

			r1.x=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round((index+i)/tanf(angle)),2))*incr)*57.2884;
	
			i=1;
			x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
			y=(float)(pt1 + 1  + (index+i))*incr;
			r1.y=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round((index+i)/tanf(angle)),2))*incr)*57.2884;


			i=2;
			x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
			y=(float)(pt1 + 1  + (index+i))*incr;
			
			r1.z=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round((index+i)/tanf(angle)),2))*incr)*57.2884;

			i=3;
			x=(float)(pt0 + 1  + (int)roundf((index+i)/tanf(angle)))*incr;
			y=(float)(pt1 + 1  + (index+i))*incr;
		
			r1.w=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round((index+i)/tanf(angle)),2))*incr)*57.2884;
		
			break;
			
		case 3:	i=0;
				x=(pt0 +1 -(index+i))*incr;
				y=(pt1 +1 -(int)roundf((index+i)*tanf(angle) ))*incr;

				r1.x=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;
	
				i=1;
				x=(pt0 +1 -(index+i))*incr;
				y=(pt1 +1 -(int)roundf((index+i)*tanf(angle) ))*incr;
		
				r1.y=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;


				i=2;
				x=(pt0 +1 -(index+i))*incr;
				y=(pt1 +1 -(int)roundf((index+i)*tanf(angle) ))*incr;
				
				r1.z=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;

				i=3;
				x=(pt0 +1 -(index+i))*incr;
				y=(pt1 +1 -(int)roundf((index+i)*tanf(angle) ))*incr;
				
				r1.w=atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((index+i)*tanf(angle))),2))*incr)*57.2884;
				break;
				
				
		case 4:	i=0;
				x=(pt0 + 1 -(int)roundf((index+i)/tanf(angle)) )*incr;
				y=(pt1 + 1 -( index+i))*incr;
						
				r1.x =atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr)*57.2884;
				//r1.x=sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr;
				
				i=1;
				x=(pt0 + 1 -(int)roundf((index+i)/tanf(angle)) )*incr;
				y=(pt1 + 1 -( index+i))*incr;
					
				r1.y =atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr)*57.2884;
				//r1.y=sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr;

				i=2;
				x=(pt0 + 1 -(int)roundf((index+i)/tanf(angle)) )*incr;
				y=(pt1 + 1 -( index+i))*incr;
					
				r1.z =atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr)*57.2884;
				//r1.z=sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr;

				i=3;
				x=(pt0 + 1 -(int)roundf((index+i)/tanf(angle)) )*incr;
				y=(pt1 + 1 -( index+i))*incr;
					
					
				r1.w =atan2f(fun_s(x,y)-h,sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr)*57.2884;
				//r1.w=sqrt(pow((double)(index+i),2) + pow((double)round(((double)(index+i)/tanf(angle))),2))*incr;
				break;
		
		}
	/*	
		r1.x=(float)pt0;
		r1.y=(float)pt1;
		r1.z=(float)pt0;
		r1.w=(float)pt1;
		*/
		
    //Load data
    T4 idata4 = r1;

	d_Src[(index-1)/4]=r1;

    //Calculate exclusive scan
    T4 odata4 = scan4Inclusive(idata4, s_Data, size,oper,Identity);

    //Write back
    d_Dst[(index-1)/4] = odata4;
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
// +         d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

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

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing
extern "C" const T MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const T MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const T MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const T MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const T MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;

/*
//Internal exclusive scan buffer
static T *d_Buf[(MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(T)];


extern "C" void initScan(void){
    cutilSafeCall( cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(T)) );
}

extern "C" void closeScan(void){
    cutilSafeCall( cudaFree(d_Buf) );
}


static T factorRadix2(T& log2L, T L){
    if(!L){
        log2L = 0;
        return 0;
    }else{
        for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
}

static T iDivUp(T dividend, T divisor){
    return ( (dividend % divisor) == 0 ) ? (dividend / divisor) : (dividend / divisor + 1);
}
*/
