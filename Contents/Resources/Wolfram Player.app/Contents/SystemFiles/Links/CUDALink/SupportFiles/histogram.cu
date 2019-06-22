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
  * ${basedir}/ExtraComponents/CUDA_SDK/3.0/Linux-x86-64/C/src/histogram
 **/


#define MERGE_THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////

#define HISTOGRAM256_BIN_COUNT 256
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;
typedef uint4 data_t;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////

#define LOG2_WARP_SIZE 5U

#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 16

//Threadblock size: must be a multiple of (4 * SHARED_MEMORY_BANKS)
//because of the bit permutation of threadIdx.x
#define HISTOGRAM64_THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)

//Warps ==subhistograms per threadblock
#define WARP_COUNT 6

//Threadblock size
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)

//Shared memory per threadblock
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

#define HISTOGRAM64_BIN_COUNT 64
#define HISTOGRAM64_THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)


////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////
#if __CUDA_ARCH__ <= 110
#define USE_SMEM_ATOMICS 0
#else
#define USE_SMEM_ATOMICS 1
#endif

#if(!USE_SMEM_ATOMICS)
    #define TAG_MASK ( (1U << (UINT_BITS - LOG2_WARP_SIZE)) - 1U )

inline __device__ void addByte(volatile uint *s_WarpHist, int data, uint threadTag){
    uint count;
    do{
        count = s_WarpHist[data] & TAG_MASK;
        count = threadTag | (count + 1);
        s_WarpHist[data] = count;
    }while(s_WarpHist[data] != count);
}
#else
    #ifdef CUDA_NO_SM12_ATOMIC_INTRINSICS
        #error Compilation target does not support shared-memory atomics
    #endif

    #define TAG_MASK 0xFFFFFFFFU
    inline __device__ void addByte(uint *s_WarpHist, int data, uint threadTag){
        atomicAdd(s_WarpHist + data, 1);
    }
#endif

inline __device__ void addWord(uint *s_WarpHist, int data0,int data1,int data2,int data3, uint tag){
    addByte(s_WarpHist, data0, tag);
    addByte(s_WarpHist, data1, tag);
    addByte(s_WarpHist, data2, tag);
    addByte(s_WarpHist, data3, tag);
}

inline __device__ void addWordPacked(uint *s_WarpHist, uint data, uint tag){
    addByte(s_WarpHist, (data >>  0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >>  8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}


__global__ void histogram256Kernel(uint *d_PartialHistograms, int usingPackedQ, int *d_Data, uint dataCount){
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
    #pragma unroll
    for(uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
       s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;

    //Cycle through the entire data set, update subhistograms for each warp
    
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);
    
    __syncthreads();
    for(uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)){
        if (usingPackedQ == 1) {
        	addWordPacked(s_WarpHist, d_Data[pos], tag);
        } else {
        	addWord(s_WarpHist, d_Data[4*pos],d_Data[4*pos+1],d_Data[4*pos+2],d_Data[4*pos+3], tag);
        }
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();
    for(uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE){
        uint sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////

__global__ void mergeHistogram256Kernel(
    uint *d_Histogram,
    uint *d_PartialHistograms,
    uint histogramCount
){
    uint sum = 0;
    for(uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];

    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    data[threadIdx.x] = sum;

    for(uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(threadIdx.x < stride)
            data[threadIdx.x] += data[threadIdx.x + stride];
    }

    if(threadIdx.x == 0)
        d_Histogram[blockIdx.x] = data[0];
}

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
//Count a byte into shared-memory storage
inline __device__ void addByte64(uchar *s_ThreadBase, uint data){
    s_ThreadBase[UMUL(data, HISTOGRAM64_THREADBLOCK_SIZE)]++;
}

__global__ void histogram64Kernel(uint *d_PartialHistograms, uint4 *d_Data, uint dataCount){
    //Encode thread index in order to avoid bank conflicts in s_Hist[] access:
    //each group of SHARED_MEMORY_BANKS threads accesses consecutive shared memory banks
    //and the same bytes [0..3] within the banks
    //Because of this permutation block size should be a multiple of 4 * SHARED_MEMORY_BANKS
    const uint threadPos = 
        ( (threadIdx.x & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0 ) |
        ( (threadIdx.x &  (SHARED_MEMORY_BANKS     - 1)) << 2 ) |
        ( (threadIdx.x &  (SHARED_MEMORY_BANKS * 3    )) >> 4 );

    //Per-thread histogram storage
    __shared__ uchar s_Hist[HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT];
    uchar *s_ThreadBase = s_Hist + threadPos;

    //Initialize shared memory (writing 32-bit words)
    #pragma unroll
    for(uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++)
        ((uint *)s_Hist)[threadIdx.x + i * HISTOGRAM64_THREADBLOCK_SIZE] = 0;

    //Read data from global memory and submit to the shared-memory histogram
    //Since histogram counters are byte-sized, every single thread can't do more than 255 submission
    __syncthreads();
    for(uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)){
        uint4 data = d_Data[pos];
        addByte64(s_ThreadBase, data.x);
        addByte64(s_ThreadBase, data.y);
        addByte64(s_ThreadBase, data.z);
        addByte64(s_ThreadBase, data.w);
    }

    //Accumulate per-thread histograms into per-block and write to global memory
    __syncthreads();
    if(threadIdx.x < HISTOGRAM64_BIN_COUNT){
        uchar *s_HistBase = s_Hist + UMUL(threadIdx.x, HISTOGRAM64_THREADBLOCK_SIZE);

        uint sum = 0;
        uint pos = 4 * (threadIdx.x & (SHARED_MEMORY_BANKS - 1));

        #pragma unroll
        for(uint i = 0; i < (HISTOGRAM64_THREADBLOCK_SIZE / 4); i++){
            sum += 
                s_HistBase[pos + 0] +
                s_HistBase[pos + 1] +
                s_HistBase[pos + 2] +
                s_HistBase[pos + 3];
            pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1);
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM64_BIN_COUNT + threadIdx.x] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge histogram64() output
// Run one threadblock per bin; each threadbock adds up the same bin counter 
// from every partial histogram. Reads are uncoalesced, but mergeHistogram64
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram64Kernel(
    uint *d_Histogram,
    uint *d_PartialHistograms,
    uint histogramCount
){
    __shared__ uint data[MERGE_THREADBLOCK_SIZE];

    uint sum = 0;
    for(uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM64_BIN_COUNT];
    data[threadIdx.x] = sum;

    for(uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(threadIdx.x < stride)
            data[threadIdx.x] += data[threadIdx.x + stride];
    }

    if(threadIdx.x == 0)
        d_Histogram[blockIdx.x] = data[0];
}

