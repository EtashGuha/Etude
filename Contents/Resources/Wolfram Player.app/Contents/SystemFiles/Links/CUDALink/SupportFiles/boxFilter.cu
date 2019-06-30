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


/*
    Perform a fast box filter using the sliding window method.

    As the kernel moves from left to right, we add in the contribution of the new
    sample on the right, and subtract the value of the exiting sample on the left.
    This only requires 2 adds and a mul per output value, independent of the filter radius.
    The box filter is separable, so to perform a 2D box filter we perform the filter in
    the x direction, followed by the same filter in the y direction.
    Applying multiple iterations of the box filter converges towards a Gaussian blur.
    Using CUDA, rows or columns of the image are processed in parallel.
    This version duplicates edge pixels.

    Note that the x (row) pass suffers from uncoalesced global memory reads,
    since each thread is reading from a different row. For this reason it is
    better to use texture lookups for the x pass.
    The y (column) pass is perfectly coalesced.

    Parameters
    id - pointer to input data in global memory
    od - pointer to output data in global memory
    w  - image width
    h  - image height
    r  - filter radius

    e.g. for r = 2, w = 8:

    0 1 2 3 4 5 6 7
    x - -
    - x - -
    - - x - -
      - - x - -
        - - x - -
          - - x - -
            - - x -
              - - x
*/

typedef unsigned int uint;


#include	<cutil_math.h>

// process row
__device__ void
d_boxfilter_x(float *id, float *od, mint w, mint h, mint r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;
    for (int x = 0; x < (r + 1); x++) {
        t += id[x];
    }
    od[0] = t * scale;

    for(int x = 1; x < (r + 1); x++) {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }
    
    // main loop
    for(int x = (r + 1); x < w - r; x++) {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++) {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void
d_boxfilter_y(float *id, float *od, mint w, mint h, mint r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    
    float t;
    // do left edge
    t = id[0] * r;
    for (int y = 0; y < (r + 1); y++) {
        t += id[y * w];
    }
    od[0] = t * scale;

    for(int y = 1; y < (r + 1); y++) {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }
    
    // main loop
    for(int y = (r + 1); y < (h - r); y++) {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++) {
        t += id[(h-1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void
d_boxfilter_x_global(float *id, float *od, mint w, mint h, mint r)
{
	unsigned int y = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void
d_boxfilter_y_global(float *id, float *od, mint w, mint h, mint r)
{
	unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	d_boxfilter_y(&id[x], &od[x], w, h, r);
}

// RGBA version
// reads from 32-bit uint array holding 8-bit RGBA

// convert floating point rgba color to 32-bit integer
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

// row pass using coalesced global memory reads
extern "C"
__global__ void
d_boxfilter_rgba_x(uint *id, uint *od, mint w, mint h, mint r)
{
	unsigned int y = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    id = &id[y * w];
    od = &od[y * w];

    float scale = 1.0f / (float)((r << 1) + 1);
    
    float4 t;
    // do left edge
    t = rgbaIntToFloat(id[0]) * r;
    for (int x = 0; x < (r + 1); x++) {
        t += rgbaIntToFloat(id[x]);
    }
    od[0] = rgbaFloatToInt(t * scale);

    for(int x = 1; x < (r + 1); x++) {
        t += rgbaIntToFloat(id[x + r]);
        t -= rgbaIntToFloat(id[0]);
        od[x] = rgbaFloatToInt(t * scale);
    }
    
    // main loop
    for(int x = (r + 1); x < (w - r); x++) {
        t += rgbaIntToFloat(id[x + r]);
        t -= rgbaIntToFloat(id[x - r - 1]);
        od[x] = rgbaFloatToInt(t * scale);
    }

    // do right edge
    for (int x = w - r; x < w; x++) {
        t += rgbaIntToFloat(id[w - 1]);
        t -= rgbaIntToFloat(id[x - r - 1]);
        od[x] = rgbaFloatToInt(t * scale);
    }
}

// column pass using coalesced global memory reads
extern "C"
__global__ void
d_boxfilter_rgba_y(uint *id, uint *od, mint w, mint h, mint r)
{
	unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    id = &id[x];
    od = &od[x];

    float scale = 1.0f / (float)((r << 1) + 1);
    
    float4 t;
    // do left edge
    t = rgbaIntToFloat(id[0]) * r;
    for (int y = 0; y < (r + 1); y++) {
        t += rgbaIntToFloat(id[y*w]);
    }
    od[0] = rgbaFloatToInt(t * scale);

    for(int y = 1; y < (r + 1); y++) {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[0]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }
    
    // main loop
    for(int y = (r + 1); y < (h - r); y++) {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // do right edge
    for (int y = h - r; y < h; y++) {
        t += rgbaIntToFloat(id[(h - 1) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }
}


