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
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
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
#include	<stdio.h>
#include	<stdlib.h>
#include	<wgl_cuda_runtime.h>
#include	<assert.h>

#include	<driver_functions.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

texture<uchar,  3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>     transferTex; // 1D transfer function texture

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

#define CEIL(x,y)         (((x)+(y)-1)/(y))
#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define R(p)    (((p) >> 24) & 0xFF)
#define G(p)    (((p) >> 16) & 0xFF)
#define B(p)    (((p) >>  8) & 0xFF)
#define A(p)    ((p) & 0xFF)

#ifndef __func__
#ifdef __FUNCTION__
#define __func__ __FUNCTION__
#else
#define __func__ __FILE__
#endif
#endif

#ifdef DEBUG
#define LOG(...)  \
    fprintf(stdout, "One line %d in %s:%s ----\n",\
            __LINE__, __FILE__, __func__);        \
    fprintf(stdout, __VA_ARGS__);				  \
    fprintf(stdout, "\n");
#else
#define LOG(...) 
#endif

MTensor outputPixelTensor;

cudaExtent volumeSize;

mint width, height;
dim3 blockSize(16, 16);
dim3 gridSize(CEIL(width, blockSize.x), CEIL(height, blockSize.y));

float3 viewRotation;
float3 viewTranslation;
float invViewMatrix[12];

float density;
float brightness;
float transferOffset;
float transferScale;
bool linearFiltering = true;

mint ox, oy;
mint buttonState = 0;

mint *output, *d_output;
mint *outputPixelData;

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);  // discards w
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
static __inline__ __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
static __inline__ __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// reflect
inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
inline __host__ __device__ float3 fabs(float3 v)
{
	return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// negate
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// addition
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
inline __host__ __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar) {
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


__global__ void
d_render(mint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale) {
    int maxSteps = 500;
    float tstep = 0.01f;
    float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from back to front, accumulating color
    float4 sum = make_float4(0.0f);;
    float t = tfar;
	for(int i=0; i<maxSteps; i++) {		
        float3 pos = eyeRay.o + eyeRay.d*t;
        pos = pos*0.5f+0.5f;    // map position to [0, 1] coordinates

        // read from 3D texture
        float sample = tex3D(tex, pos.x, pos.y, pos.z);

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);

        // accumulate result
        sum = lerp(sum, col, col.w*density);

        t -= tstep;
        if (t < tnear) break;
    }
    sum *= brightness;

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;
        d_output[i] = rgbaFloatToInt(sum);
    }
}

void setTextureFilterMode(bool bLinearFilter) {
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

void iSetTransferFunction(float4 * transferFunc, int transferFuncLength) {
    cudaError_t err;
    err = cudaFreeArray(d_transferFuncArray);
    assert(err == cudaSuccess);

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaUnbindTexture(transferTex);

    err = cudaMallocArray( &d_transferFuncArray, &channelDesc2, transferFuncLength, 1); 
    assert(err == cudaSuccess);
    err = cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc, transferFuncLength * sizeof(float4), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    err = cudaBindTextureToArray( transferTex, d_transferFuncArray, channelDesc2);
    assert(err == cudaSuccess);
    return ;
}

cudaError_t initCuda(uchar *h_volume, cudaExtent volumeSize, float4 * transferFunc, int transferFuncLength)
{
    cudaError_t err;

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    err = cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);
	if (err != cudaSuccess) {
		return err;
	}

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    err = cudaMemcpy3D(&copyParams);
	if (err != cudaSuccess) {
		cudaFreeArray(d_volumeArray);
		return err;
	}

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    err = cudaBindTextureToArray(tex, d_volumeArray, channelDesc);
	if (err != cudaSuccess) {
		cudaFreeArray(d_volumeArray);
		return err;
	}

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    err = cudaMallocArray( &d_transferFuncArray, &channelDesc2, transferFuncLength, 1); 
	if (err != cudaSuccess) {
		cudaFreeArray(d_volumeArray);
		return err;
	}
    err = cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc, transferFuncLength * sizeof(float4), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cudaFreeArray(d_volumeArray);
		cudaFreeArray(d_transferFuncArray);
		return err;
	}

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    err = cudaBindTextureToArray( transferTex, d_transferFuncArray, channelDesc2);
    if (err != cudaSuccess) {
		cudaFreeArray(d_volumeArray);
		cudaFreeArray(d_transferFuncArray);
		return err;
	} else {
		return cudaSuccess;
	}
}

void freeCudaBuffers() {
    cudaError_t err;
    err = cudaFreeArray(d_volumeArray);
    assert(err == cudaSuccess);
    err = cudaFreeArray(d_transferFuncArray);
    assert(err == cudaSuccess);
}

int ss = 1;

void render_kernel(dim3 gridSize, dim3 blockSize, mint *d_output, uint imageW, uint imageH, 
				   float density, float brightness, float transferOffset, float transferScale) {
	d_render<<<gridSize, blockSize>>>( d_output, imageW, imageH, density, 
										brightness, transferOffset, transferScale);
}

void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix);
    assert(err == cudaSuccess);
}


static void render(mint ** pixels) {
    cudaError_t err;
    
	copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    if (*pixels == NULL) {
        *pixels = (mint *) malloc(width*height * sizeof(mint));
        assert(*pixels != NULL);

    }

    if (d_output == NULL) {
        err = cudaMalloc((void**)&d_output, width*height * sizeof(mint));
        assert(err == cudaSuccess);
    }

    err = cudaMemset(d_output, 0, width*height*sizeof(mint));
    assert(err == cudaSuccess);

    cudaGetLastError();

    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);

    err = cudaGetLastError();
    assert(err == cudaSuccess);

    err = cudaMemcpy(*pixels, d_output, width*height*sizeof(mint), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    if (err != cudaSuccess) {
        LOG("kernel failed");
    }
#ifdef DEBUG
    printf("Rendering new Image...\n");
#endif
}

static void iNewFrame() {
    mint ii, jj;
    mint pixel;

    #define PI 3.14159265f
    float theta, psi;
    theta = -viewRotation.x*PI/180.0f;
    psi   = -viewRotation.y*PI/180.0f;

    invViewMatrix[0] = cos(psi);
    invViewMatrix[4] = sin(theta)*sin(psi);
    invViewMatrix[8] = -cos(theta)*sin(psi);

    invViewMatrix[1] = 0.0f;
    invViewMatrix[5] = cos(theta);
    invViewMatrix[9] = sin(theta);

    invViewMatrix[2] = sin(psi);
    invViewMatrix[6] = -sin(theta)*cos(psi);
    invViewMatrix[10]= cos(theta)*cos(psi);

    invViewMatrix[3] = -invViewMatrix[0]*viewTranslation.x - invViewMatrix[1]*viewTranslation.y - invViewMatrix[2]*viewTranslation.z;
    invViewMatrix[7] = -invViewMatrix[4]*viewTranslation.x - invViewMatrix[5]*viewTranslation.y - invViewMatrix[6]*viewTranslation.z;
    invViewMatrix[11] = -invViewMatrix[8]*viewTranslation.x - invViewMatrix[9]*viewTranslation.y - invViewMatrix[10]*viewTranslation.z;

    render(&output);

    for (jj = 0; jj < height; jj++) {
        for (ii = 0; ii < width; ii++) {
            pixel = output[ii + jj*width];
            outputPixelData[4*(ii + jj*width)    ] = A(pixel);
            outputPixelData[4*(ii + jj*width) + 1] = B(pixel);
            outputPixelData[4*(ii + jj*width) + 2] = G(pixel);
            outputPixelData[4*(ii + jj*width) + 3] = R(pixel);
        }
    }
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_NewFrame(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    iNewFrame();
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_MouseMovement(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    mint button, state, x, y;

    button  = MArgument_getInteger(Args[0]);
    state = MArgument_getInteger(Args[1]);
    x = MArgument_getInteger(Args[2]);
    y = MArgument_getInteger(Args[3]);

    if (state == 1) {
        buttonState = button;
#ifdef DEBUG
        printf("Enable State\n");
        printf("button state = %d\n", button);
#endif
    } else if (state == 0) {
#ifdef DEBUG
        printf("Disable State\n");
#endif
        buttonState = 0;
    }

    ox = x; oy = y;

    iNewFrame();
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_Motion(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    float dx, dy;
    mint x, y;

    x = MArgument_getInteger(Args[0]);
    y = MArgument_getInteger(Args[1]);

    dx = (float) x - ox;
    dy = (float) y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
#ifdef DEBUG
        printf("zoom\n");
#endif
        viewTranslation.z += dy / 100.0f;
    } else if (buttonState == 2) {
        // middle = translate
#ifdef DEBUG
        printf("translate dx=%f  dy =%f\n", dx, dy);
#endif
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    } else if (buttonState == 1) {
        // left = rotate
#ifdef DEBUG
        printf("rotate\n");
#endif
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x; oy = y;

	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_SetBrightness(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	float val;
	
	val = (float) MArgument_getReal(Args[0]);
	
	brightness = val;

	return LIBRARY_NO_ERROR;
}


EXTERN_C DLLEXPORT mint oVolumetricRendering_SetTransferScale(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	float val;
	
	val = (float) MArgument_getReal(Args[0]);

    transferScale = val;

	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_SetTransferOffset(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    float val;
	
	val = (float) MArgument_getReal(Args[0]);
    
    transferOffset = val;

	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_SetDensity(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	float val;
	
	val = (float) MArgument_getReal(Args[0]);
    
    density = val;
	
	return LIBRARY_NO_ERROR;
}

static void cleanup(WolframLibraryData libData) {
	freeCudaBuffers();
    free(output);
    cudaFree(d_output);
    libData->MTensor_disown(outputPixelTensor);
}

static float4 * convertTransferFunction(WolframLibraryData libData, MTensor tf) {
    double * transferFunctionData;
    float4 * transferFunctionFloatData;
    mint len, ii;

    transferFunctionData = libData->MTensor_getRealData(tf);
    len = libData->MTensor_getFlattenedLength(tf)/4;
    transferFunctionFloatData = (float4 *) malloc(len * sizeof(float4));
    assert(transferFunctionFloatData != NULL);
    for (ii = 0; ii < libData->MTensor_getDimensions(tf)[0]; ii++) {
        transferFunctionFloatData[ii].x = (float) transferFunctionData[4*ii];
        transferFunctionFloatData[ii].y = (float) transferFunctionData[4*ii + 1];
        transferFunctionFloatData[ii].z = (float) transferFunctionData[4*ii + 2];
        transferFunctionFloatData[ii].w = (float) transferFunctionData[4*ii + 3];
    }
    return transferFunctionFloatData;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_SetTransferFunction(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    float4 * transferFunctionFloatData;
    MTensor transferFunctionTensor;
    mint len;

    transferFunctionTensor = MArgument_getMTensor(Args[0]);

    transferFunctionFloatData = convertTransferFunction(libData, transferFunctionTensor);
    assert(transferFunctionFloatData != NULL);
    
    len = libData->MTensor_getFlattenedLength(transferFunctionTensor)/4;

    iSetTransferFunction(transferFunctionFloatData, len);
    
    libData->MTensor_disown(transferFunctionTensor);
    free(transferFunctionFloatData);

    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_Initialize(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    MTensor volumeTensor, transferFunctionTensor;
    mint ii, len, *volumeData;
    uchar * h_volume;
    float4 * transferFunctionFloatData;
	cudaError_t cuErr;
	int retValue = LIBRARY_FUNCTION_ERROR;

    viewTranslation.x = 0.0;
    viewTranslation.y = 0.0;
    viewTranslation.z = -4.0f;

    volumeTensor = MArgument_getMTensor(Args[0]);
    outputPixelTensor = MArgument_getMTensor(Args[1]);
    transferFunctionTensor = MArgument_getMTensor(Args[2]);

    outputPixelData = libData->MTensor_getIntegerData(outputPixelTensor);
    width = libData->MTensor_getDimensions(outputPixelTensor)[1];
    height = libData->MTensor_getDimensions(outputPixelTensor)[0];

    output = d_output = NULL;

    assert(libData->MTensor_getRank(volumeTensor) == 3);
    volumeSize = make_cudaExtent(libData->MTensor_getDimensions(volumeTensor)[1],
                                 libData->MTensor_getDimensions(volumeTensor)[0],
                                 libData->MTensor_getDimensions(volumeTensor)[2]);

    volumeData = libData->MTensor_getIntegerData(volumeTensor);

    h_volume = (uchar *) malloc(libData->MTensor_getFlattenedLength(volumeTensor) * sizeof(char));

    for (ii = 0; ii < libData->MTensor_getFlattenedLength(volumeTensor); ii++)
        h_volume[ii] = volumeData[ii] & 0xFF;

    
    transferFunctionFloatData = convertTransferFunction(libData, transferFunctionTensor);
    assert(transferFunctionFloatData != NULL);
    len = libData->MTensor_getFlattenedLength(transferFunctionTensor)/4;
    cuErr = initCuda(h_volume, volumeSize, transferFunctionFloatData, len);
	if (cuErr != cudaSuccess) {
		retValue = LIBRARY_MEMORY_ERROR;
		goto cleanup;
	}

    gridSize = dim3(CEIL(width, blockSize.x), CEIL(height, blockSize.y));

	iNewFrame();
	retValue = LIBRARY_NO_ERROR;

cleanup:
    libData->MTensor_disown(volumeTensor);
    libData->MTensor_disown(transferFunctionTensor);
    free(h_volume);
    free(transferFunctionFloatData);

    return retValue;
}

EXTERN_C DLLEXPORT mint oVolumetricRendering_Uninitialize(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	cleanup(libData);
    return LIBRARY_NO_ERROR;
}

