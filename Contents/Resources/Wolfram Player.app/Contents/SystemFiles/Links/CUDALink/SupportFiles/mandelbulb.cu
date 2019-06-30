/*
Copyright (c) 2009 David Bucciarelli (davibu@interfree.it)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#define GPU_KERNEL

#define BOUNDING_RADIUS_2 4.f

/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */

/*
    This file implements common mathematical operations on vector types
    (float3, float4 etc.) since these are not provided as standard by CUDA.

    The syntax is modelled on the Cg standard library.
*/


// float4 functions
////////////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"

typedef unsigned int uint;

#ifndef __CUDACC__
#include <math.h>
inline float fminf(float a, float b)
{
  return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
  return a > b ? a : b;
}

inline int max(int a, int b)
{
  return a > b ? a : b;
}

inline int min(int a, int b)
{
  return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
inline __device__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// int2 functions
////////////////////////////////////////////////////////////////////////////////

// negate
inline __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}

// addition
inline __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __device__ int2 operator*(int2 a, int s)
{
    return make_int2(a.x * s, a.y * s);
}
inline __device__ int2 operator*(int s, int2 a)
{
    return make_int2(a.x * s, a.y * s);
}
inline __device__ void operator*=(int2 &a, int s)
{
    a.x *= s; a.y *= s;
}
// additional constructors
inline __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// negate
inline __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// addition
inline __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
inline __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
inline __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __device__ void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
inline __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __device__ float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __device__ float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __device__ void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ float4 lerp(float4 a, float4 b, float t)
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
inline __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
inline __device__ float length(float4 r)
{
    return sqrtf(dot(r, r));
}

// normalize
inline __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __device__ float4 floor(float4 v)
{
    return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

// absolute value
inline __device__ float4 fabs(float4 v)
{
	return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

typedef struct {
	float x, y, z; // position, also color (r,g,b)
} Vec;

#define vinit(v, a, b, c) { (v).x = a; (v).y = b; (v).z = c; }
#define vclr(v) vinit(v, 0.f, 0.f, 0.f)
#define vadd(v, a, b) vinit(v, (a).x + (b).x, (a).y + (b).y, (a).z + (b).z)
#define vsub(v, a, b) vinit(v, (a).x - (b).x, (a).y - (b).y, (a).z - (b).z)
#define vsadd(v, a, b) { float k = (a); vinit(v, (b).x + k, (b).y + k, (b).z + k) }
#define vssub(v, a, b) { float k = (a); vinit(v, (b).x - k, (b).y - k, (b).z - k) }
#define vmul(v, a, b) vinit(v, (a).x * (b).x, (a).y * (b).y, (a).z * (b).z)
#define vsmul(v, a, b) { float k = (a); vinit(v, k * (b).x, k * (b).y, k * (b).z) }
#define vdot(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define vnorm(v) { float l = 1.f / sqrt(vdot(v, v)); vsmul(v, l, v); }
#define vxcross(v, a, b) vinit(v, (a).y * (b).z - (a).z * (b).y, (a).z * (b).x - (a).x * (b).z, (a).x * (b).y - (a).y * (b).x)
#define vfilter(v) ((v).x > (v).y && (v).x > (v).z ? (v).x : (v).y > (v).z ? (v).y : (v).z)
#define viszero(v) (((v).x == 0.f) && ((v).x == 0.f) && ((v).z == 0.f))

#ifndef GPU_KERNEL
#define clamp(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define max(x, y) ( (x) > (y) ? (x) : (y))
#define min(x, y) ( (x) < (y) ? (x) : (y))
#endif

#define toInt(x) ((int)(clamp(x, 0.f, 1.f) * 255.f + .5f))

typedef struct {
	/* User defined values */
	Vec orig, target;
	/* Calculated values */
	Vec dir, x, y;
} Camera;

typedef struct {
	unsigned int width, height;
	int superSamplingSize;
	int actvateFastRendering;
	int enableShadow;

	unsigned int maxIterations;
	float epsilon;
	float mu[4];
	float light[3];
	Camera camera;
} RenderingConfig;

typedef struct {
	unsigned int width, height;
	unsigned int superSamplingSize;
	unsigned int actvateFastRendering;
	unsigned int enableShadow;
	unsigned int maxIterations;
} IntRenderingConfig;

typedef struct {
	float epsilon;
	float mu[4];
	float light[3];
	Camera camera;
} FloatRenderingConfig;

// Scalar derivative approach by Enforcer:
// http://www.fractalforums.com/mandelbulb-implementation/realtime-renderingoptimisations/
__device__ float IterateIntersect(float4 z0, float4 c0, const uint maxIterations) {
	float4 z = z0;
	float4 c = z0;

	float dr = 1.0f;
	float r2 = dot(z, z);
	float r = sqrt(r2);
	for (int n = 0; (n < maxIterations) && (r < 2.f); ++n) {
		const float zo0 = asin(z.z / r);
		const float zi0 = atan2(z.y, z.x);
		float zr = r2 * r2 * r2 * r2;
		const float zo = zo0 * 8.f;
		const float zi = zi0 * 8.f;
		const float czo = cos(zo);

		dr = zr * dr * 8.f + 1.f;
		zr *= r;

		z = zr * make_float4(czo * cos(zi), czo * sin(zi), sin(zo), 0.f);
		z += c;

		r2 = dot(z, z);
		r = sqrt(r2);
	}

	return 0.5f * log(r) * r / dr;
}

__device__ float IntersectBulb(float4 eyeRayOrig, float4 eyeRayDir,
		float4 c, const uint maxIterations, const float epsilon,
		const float maxDist, float4 *hitPoint, uint *steps) {
	float dist;
	float4 r0 = eyeRayOrig;
	float distDone = 0.f;

	uint s = 0;
	do {
		dist = IterateIntersect(r0, c, maxIterations);
		distDone += dist;
		// We are inside
		if (dist <= 0.f)
			break;

		r0 += eyeRayDir * dist;
		s++;
	} while ((dist > epsilon) && (distDone < maxDist));

	*hitPoint = r0;
	*steps = s;
	return dist;
}

#define WORLD_RADIUS 1000.f
#define WORLD_CENTER (make_float4(0.f, -WORLD_RADIUS - 2.f, 0.f, 0.f))
__device__ float IntersectFloorSphere(float4 eyeRayOrig, float4 eyeRayDir) {
	float4 op = WORLD_CENTER - eyeRayOrig;
	const float b = dot(op, eyeRayDir);
	float det = b * b - dot(op, op) + WORLD_RADIUS * WORLD_RADIUS;

	if (det < 0.f)
		return -1.f;
	else
		det = sqrt(det);

	float t = b - det;
	if (t > 0.f)
		return t;
	else {
		// We are inside, avoid the hit
		return -1.f;
	}
}

__device__ int IntersectBoundingSphere(float4 eyeRayOrig, float4 eyeRayDir,
		float *tmin, float*tmax) {
	float4 op = -1 * eyeRayOrig;
	const float b = dot(op, eyeRayDir);
	float det = b * b - dot(op, op) + BOUNDING_RADIUS_2;

	if (det < 0.f)
		return 0;
	else
		det = sqrt(det);

	float t1 = b - det;
	float t2 = b + det;
	if (t1 > 0.f) {
		*tmin = t1;
		*tmax = t2;
		return 1;
	} else {
		if (t2 > 0.f) {
			// We are inside, start from the ray origin
			*tmin = 0.f;
			*tmax = t2;

			return 1;
		} else
			return 0;
	}
}

__device__ float4 NormEstimate(float4 p, float4 c, const float delta, const uint maxIterations) {
	float4 qP = p;
	float4 gx1 = qP - make_float4(delta, 0.f, 0.f, 0.f);
	float4 gx2 = qP + make_float4(delta, 0.f, 0.f, 0.f);
	float4 gy1 = qP - make_float4(0.f, delta, 0.f, 0.f);
	float4 gy2 = qP + make_float4(0.f, delta, 0.f, 0.f);
	float4 gz1 = qP - make_float4(0.f, 0.f, delta, 0.f);
	float4 gz2 = qP + make_float4(0.f, 0.f, delta, 0.f);

	const float gradX = fabs(IterateIntersect(gx2, c, maxIterations) -
		fabs(IterateIntersect(gx1, c, maxIterations)));
	const float gradY = fabs(IterateIntersect(gy2, c, maxIterations)) -
		fabs(IterateIntersect(gy1, c, maxIterations));
	const float gradZ = fabs(IterateIntersect(gz2, c, maxIterations)) -
		fabs(IterateIntersect(gz1, c, maxIterations));

	float4 N = normalize(make_float4(gradX, gradY, gradZ, 0.f));

	return N;
}

__device__ float4 Phong(float4 light, float4 eye, float4 pt, float4 N, float4 diffuse) {
	float4 ambient = make_float4(0.05f, 0.05f, 0.05f, 0.f);
	float4 L = normalize(light - pt);
	float NdotL = dot(N, L);
	if (NdotL < 0.f) {
		return diffuse * ambient;
	}

	const float specularExponent = 30.f;
	const float specularity = 0.65f;

	float4 E = normalize(eye - pt);
	float4 H = (L + E) * (float)0.5f;

	return diffuse * NdotL +
			specularity * pow(dot(N, H), specularExponent) +
			diffuse * ambient;
}

extern "C"
__global__ void MandelbulbGPU(
	float *pixels,
	//const __global RenderingConfig *config,
	const FloatRenderingConfig *config,
	const IntRenderingConfig *iconfig,
	const int enableAccumulation,
	const float sampleX,
	const float sampleY) {
		
    const int gid = threadIdx.x  + blockIdx.x * blockDim.x;
	const unsigned width = iconfig->width;
	const unsigned height = iconfig->height;

	const unsigned int x = gid % width;
	const int y = gid / width;

	// Check if we have to do something
	if (y >= height)
		return;

	const float epsilon = iconfig->actvateFastRendering ? (config->epsilon * (1.5f / 0.75f)) : config->epsilon;
	const uint maxIterations = iconfig->actvateFastRendering ? (max(3u, iconfig->maxIterations) - 2u) : iconfig->maxIterations;

	float4 mu = make_float4(config->mu[0], config->mu[1], config->mu[2], config->mu[3]);
	float4 light = make_float4(config->light[0], config->light[1], config->light[2], 0.f);
	const Camera *camera = &config->camera;

	//--------------------------------------------------------------------------
	// Calculate eye ray
	//--------------------------------------------------------------------------

	const float invWidth = 1.f / width;
	const float invHeight = 1.f / height;
	const float kcx = (x + sampleX) * invWidth - .5f;
	float4 kcx4 = make_float4(kcx);
	const float kcy = (y + sampleY) * invHeight - .5f;
	float4 kcy4 = make_float4(kcy);

	float4 cameraX = make_float4(camera->x.x, camera->x.y, camera->x.z, 0.f);
	float4 cameraY = make_float4(camera->y.x, camera->y.y, camera->y.z, 0.f);
	float4 cameraDir = make_float4(camera->dir.x, camera->dir.y, camera->dir.z, 0.f);
	float4 cameraOrig = make_float4(camera->orig.x, camera->orig.y, camera->orig.z, 0.f);

	float4 eyeRayDir =  normalize(cameraX * kcx4 + cameraY * kcy4 + cameraDir);
	float4 eyeRayOrig = eyeRayDir * make_float4(0.1f) + cameraOrig;

	//--------------------------------------------------------------------------
	// Check if we hit the bounding sphere
	//--------------------------------------------------------------------------

	int useAO = 1;
	float4 diffuse, n, color;

	float4 hitPoint;
	float dist, tmin, tmax;
	if (IntersectBoundingSphere(eyeRayOrig, eyeRayDir, &tmin, &tmax)) {
		//--------------------------------------------------------------------------
		// Find the intersection with the set
		//--------------------------------------------------------------------------

		uint steps;
		float4 rayOrig = eyeRayOrig + eyeRayDir * make_float4(tmin);
		dist = IntersectBulb(rayOrig, eyeRayDir, mu, maxIterations,
				epsilon, tmax - tmin, &hitPoint, &steps);

		if (dist <= epsilon) {
			// Set hit
			diffuse = make_float4(1.f, 0.35f, 0.15f, 0.f);
			n = NormEstimate(hitPoint, mu, dist, maxIterations);
		} else
			dist = -1.f;
	} else
		dist = -1.f;

	//--------------------------------------------------------------------------
	// Check if we hit the floor
	//--------------------------------------------------------------------------

	if (dist < 0.f) {
		dist = IntersectFloorSphere(eyeRayOrig, eyeRayDir);

		if (dist >= 0.f) {
			// Floor hit
			hitPoint = eyeRayOrig + eyeRayDir * make_float4(dist);
			n = hitPoint - WORLD_CENTER;
			n = normalize(n);
			// The most important feature in a ray tracer: a checker texture !
			const int ix = (hitPoint.x > 0.f) ? hitPoint.x : (1.f - hitPoint.x);
			const int iz = (hitPoint.z > 0.f) ? hitPoint.z : (1.f - hitPoint.z);
			if ((ix + iz) % 2)
				diffuse = make_float4(0.75f, 0.75f, 0.75f, 0.f);
			else
				diffuse = make_float4(0.75f, 0.f, 0.f, 0.f);
			useAO = 0;
		} else {
			// Sky hit
			color = make_float4(0.f, 0.1f, 0.3f, 0.f);
		}
	} else {
		// Sky hit
		color = make_float4(0.f, 0.1f, 0.3f, 0.f);
	}

	//--------------------------------------------------------------------------
	// Select the shadow pass
	//--------------------------------------------------------------------------

	if (dist >= 0.f) {
		float shadowFactor = 1.f;
		if(iconfig->enableShadow) {
			float4 L = normalize(light -  hitPoint);
			float4 rO = hitPoint + n * 1e-2f;
			float4 shadowHitPoint;

			// Check bounding sphere
			if (IntersectBoundingSphere(rO, L, &tmin, &tmax)) {
				float shadowDistSet = tmin;
				uint steps;

				rO = rO + L * make_float4(shadowDistSet);
				shadowDistSet = IntersectBulb(rO, L, mu, maxIterations, epsilon,
						tmax - tmin, &shadowHitPoint, &steps);
				if (shadowDistSet < epsilon) {
					if (useAO) {
						// Use steps count to simulate ambient occlusion
						shadowFactor = 0.6f - min(steps / 255.f, 0.5f);
					} else
						shadowFactor = 0.6f;
				}
			}
		}

		//--------------------------------------------------------------------------
		// Direct lighting of hit point
		//--------------------------------------------------------------------------

		color = Phong(light, eyeRayOrig, hitPoint, n, diffuse) * shadowFactor;
	}

	//--------------------------------------------------------------------------
	// Write pixel
	//--------------------------------------------------------------------------

	int offset = 3 * (x + (height - y - 1) * width);
	color = clamp(color, make_float4(0.f, 0.f ,0.f, 0.f), make_float4(1.f, 1.f ,1.f, 0.f));
	if (enableAccumulation) {
		pixels[offset++] += color.x;
		pixels[offset++] += color.y;
		pixels[offset] += color.z;
	} else {
		pixels[offset++] = color.x;
		pixels[offset++] = color.y;
		pixels[offset] = color.z;
	}
}




