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


#include	<wgl.h>
#include 	<stdio.h>
#include 	<stdlib.h>
#include 	<cufft.h>

#ifdef DEBUG
#include 	<assert.h>
#define Assert(x)	assert(x)
#define Expect(x)	assert(x)
#else
#define Assert(x)
#define Expect(x)   (void)(x)
#endif

#ifndef __func__
#ifdef __FUNCTION__
#define __func__ __FUNCTION__
#else
#define __func__ __FILE__
#endif
#endif

#ifdef DEBUG
#define PRINT_DBG(...)                                                  \
    printf(__VA_ARGS__)
#define LOG(...)                                                        \
    printf("One line %d in %s:%s ----", __LINE__, __FILE__, __func__);  \
    PRINT_DBG(__VA_ARGS__);                                             \
    printf("\n");
#else
#define PRINT_DBG(...)
#define LOG(...)
#endif

#ifdef DEBUG
#define cuLOG(msg)                                                      \
    if (cudaGetLastError() != cudaSuccess) {                            \
        LOG(msg);                                                       \
    }
#else
#define cuLOG(msg)
#endif

#ifndef cuSafeCall
#define cuSafeCall(stmt)                                    \
   {                                                        \
   cudaError_t cutilErr = stmt;                             \
   if (cutilErr != cudaSuccess) {                           \
        LOG(" ");                                           \
        printf("%s\n", cudaGetErrorString(cutilErr));       \
   }                                                        \
   }
#endif


// CUDA example code that implements the frequency space version of 
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the 
// CUDA FFT library (CUFFT) to perform velocity diffusion and to 
// force non-divergence in the velocity field at each time step. It uses 
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step. 
#define SetDim(dim)					\
     DIM = dim;						\
     DS = dim*dim;					\
     CPADW = dim/2+1;				\
     RPADW = 2*(dim/2+1);			\
     PDS = DIM*CPADW;				\
     wWidth = max(1024, (int) DIM);	\
     wHeight = max(1024, (int) DIM)
     
     
#define MAX_EPSILON_ERROR 1.0f

mint DIM  =  128;       // Square size of solver domain
mint DS   = (DIM*DIM);  // Total domain size
mint CPADW = (DIM/2+1);  // Padded width for real->complex in-place FFT
mint RPADW = (2*(DIM/2+1));  // Padded width for real->complex in-place FFT
mint PDS  = (DIM*CPADW); // Padded total domain size

float DT	=   0.1f;     // Delta T for interative solver
float VIS	=   0.0025f;  // Viscosity constant
float FORCE = (5.8f*DIM); // Force scale factor 
int   FR	=    4;       // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

// Vector data type used to velocity and force fields
typedef float2 cData;

// Texture reference for reading velocity field
texture<float2, 2> texref;
static cudaArray *array = NULL;

// CUFFT plan handle
static cufftHandle planr2c;
static cufftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
mint wWidth = max(1024, (int) DIM);
mint wHeight = max(1024, (int) DIM);

static int clicked = 0;

// Particle data
static cData *deviceParticles = NULL; // particle positions in host memory
static cData *hostParticles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit


static void setTimeDelta(float newDT);
static void setViscosity(float newVIS);
static void setForce(float newFORCE);
static void setForceRadius(mint newFR);
static void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
static void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
static void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
static void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy);
static void advectParticles(cData *v, int dx, int dy, float dt);
static cData * simulateFluids(mint dim);
static void resetParticles();
static void setParticles(double * data, mint n);
static void click(int button, int updown, int x, int y);
static void motion(int x, int y);
static void FluidDynamics_cleanup();
static void initialize(mint dim);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// from cutil_math.h
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}

inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float2 operator/(float s, float2 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}

// length
inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float2 floor(const float2 v)
{
    return make_float2(floor(v.x), floor(v.y));
}

// reflect
inline __host__ __device__ float2 reflect(float2 i, float2 n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
inline __host__ __device__ float2 fabs(float2 v)
{
	return make_float2(fabs(v.x), fabs(v.y));
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void setupTexture(int x, int y) {
	cudaError_t err;
    // Wrap mode appears to be the new default
    texref.filterMode = cudaFilterModeLinear;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    err = cudaMallocArray(&array, &desc, y, x);
    Expect(err == cudaSuccess);
}

void bindTexture(void) {
	if (array != NULL) {
		cudaError_t err;
		err = cudaBindTextureToArray(texref, array);
		Expect(err == cudaSuccess);
	}
}

void unbindTexture(void) {
	if (array != NULL) {
		cudaError_t err;
		err = cudaUnbindTexture(texref);
		Expect(err == cudaSuccess);
	}
}
    
void updateTexture(cData *data, size_t wib, size_t h, size_t pitch) {
	if (array != NULL) {
		cudaError_t err;
		err = cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice);
		Expect(err == cudaSuccess);
	}
}

void deleteTexture(void) {
	if (array != NULL) {
		cudaFreeArray(array);
	}
    array = NULL;
}

// Note that these kernels are designed to work with arbitrary 
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field 
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void 
addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    cData *fj = (cData*)((char*)v + (ty + spy) * pitch) + tx + spx;

    cData vterm = *fj;
    tx -= r; ty -= r;
    float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void 
advectVelocity_k(cData *v, float *vx, float *vy,
                 int dx, int pdx, int dy, float dt, int lb) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData vterm, ploc;
    float vxterm, vyterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * pdx + gtidx;
                vterm = tex2D(texref, (float)gtidx, (float)fi);
                ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
                ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
                vterm = tex2D(texref, ploc.x, ploc.y);
                vxterm = vterm.x; vyterm = vterm.y; 
                vx[fj] = vxterm;
                vy[fj] = vyterm; 
            }
        }
    }
}

// This method performs velocity diffusion and forces mass conservation 
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued 
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void 
diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt, 
                 float visc, int lb) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData xterm, yterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                xterm = vx[fj];
                yterm = vy[fj];

                // Compute the index of the wavenumber based on the
                // data order produced by a standard NN FFT.
                int iix = gtidx;
                int iiy = (fi>dy/2)?(fi-(dy)):fi;

                // Velocity diffusion
                float kk = (float)(iix * iix + iiy * iiy); // k^2 
                float diff = 1.f / (1.f + visc * dt * kk);
                xterm.x *= diff; xterm.y *= diff;
                yterm.x *= diff; yterm.y *= diff;

                // Velocity projection
                if (kk > 0.f) {
                    float rkk = 1.f / kk;
                    // Real portion of velocity projection
                    float rkp = (iix * xterm.x + iiy * yterm.x);
                    // Imaginary portion of velocity projection
                    float ikp = (iix * xterm.y + iiy * yterm.y);
                    xterm.x -= rkk * rkp * iix;
                    xterm.y -= rkk * ikp * iix;
                    yterm.x -= rkk * rkp * iiy;
                    yterm.y -= rkk * ikp * iiy;
                }
                
                vx[fj] = xterm;
                vy[fj] = yterm;
            }
        }
    }
}

// This method updates the velocity field 'v' using the two complex 
// arrays from the previous step: 'vx' and 'vy'. Here we scale the 
// real components by 1/(dx*dy) to account for an unnormalized FFT. 
__global__ void 
updateVelocity_k(cData *v, float *vx, float *vy, 
                 int dx, int pdx, int dy, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float vxterm, vyterm;
    cData nvterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fjr = fi * pdx + gtidx; 
                vxterm = vx[fjr];
                vyterm = vy[fjr];

                // Normalize the result of the inverse FFT
                float scale = 1.f / (dx * dy);
                nvterm.x = vxterm * scale;
                nvterm.y = vyterm * scale;
               
                cData *fj = (cData*)((char*)v + fi * pitch) + gtidx;
                *fj = nvterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}

// This method updates the hostParticles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).  
__global__ void 
advectParticles_k(cData *part, cData *v, int dx, int dy, 
                  float dt, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    cData pterm, vterm;
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                pterm = part[fj];
                
                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm = *((cData*)((char*)v + yvi * pitch) + xvi);   
 
                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;            
                pterm.x += 1.f; 
                pterm.x = pterm.x - (int)pterm.x;              
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;            
                pterm.y += 1.f; 
                pterm.y = pterm.y - (int)pterm.y;                  

                part[fj] = pterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}


static void setTimeDelta(float newDT) {
	DT = newDT;
	return ;
}

static void setViscosity(float newVIS) {
	VIS = newVIS;
	return ;
}

static void setForce(float newFORCE) {
	FORCE = newFORCE*DIM;
	return ;
}

static void setForceRadius(mint newFR) {
	FR = newFR;
	return ;
}

static void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r) { 

    dim3 tids(2*r+1, 2*r+1);
    
    addForces_k<<<1, tids>>>(v, dx, dy, spx, spy, fx, fy, r, tPitch);
}

static void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt) { 
    
    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);

    updateTexture(v, DIM*sizeof(cData), DIM, tPitch);
    advectVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, dt, TILEY/TIDSY);

}

static void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc) { 
    // Forward FFT
    cufftExecR2C(planr2c, (cufftReal*)vx, (cufftComplex*)vx); 
    cufftExecR2C(planr2c, (cufftReal*)vy, (cufftComplex*)vy);

    uint3 grid = make_uint3((dx/TILEX)+(!(dx%TILEX)?0:1), 
                            (dy/TILEY)+(!(dy%TILEY)?0:1), 1);

    uint3 tids = make_uint3(TIDSX, TIDSY, 1);
    
    diffuseProject_k<<<grid, tids>>>(vx, vy, dx, dy, dt, visc, TILEY/TIDSY);

    // Inverse FFT
    cufftExecC2R(planc2r, (cufftComplex*)vx, (cufftReal*)vx); 
    cufftExecC2R(planc2r, (cufftComplex*)vy, (cufftReal*)vy);
}

static void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy) { 

    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);

    updateVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, TILEY/TIDSY, tPitch);
}

static void advectParticles(cData *v, int dx, int dy, float dt) {
    cudaError_t err;

    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);
   
    advectParticles_k<<<grid, tids>>>(deviceParticles, v, dx, dy, dt, TILEY/TIDSY, tPitch);
    err = cudaMemcpy(hostParticles, deviceParticles, DS*sizeof(cData), cudaMemcpyDeviceToHost);
    Expect(err == cudaSuccess);
}

static cData * simulateFluids(mint dim) {
   if (dim != DIM && dim != -1) {
        FluidDynamics_cleanup();
        initialize(dim);
   }
   // simulate fluid
   advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM, DT);
   diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
   updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM);
   advectParticles(dvfield, DIM, DIM, DT);
   return hostParticles;
}

static void resetParticles() {
    cudaError_t err;
    int i, j;

    memset(hvfield, 0, sizeof(cData) * DS);
    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            hostParticles[i*DIM+j].x = (j+0.5f+(rand()/(float)RAND_MAX - 0.5f))/DIM;
            hostParticles[i*DIM+j].y = (i+0.5f+(rand()/(float)RAND_MAX - 0.5f))/DIM;
        }
    }
    err = cudaMemcpy(deviceParticles, hostParticles, DS*sizeof(cData), cudaMemcpyHostToDevice);
    Expect(err == cudaSuccess);
}

static void setParticles(double * data, mint n) {
    cudaError_t err;
    int i, j;

    memset(hvfield, 0, sizeof(cData) * DS);
    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            if (2*(i*DIM + j) < n) {
                hostParticles[i*DIM+j].x = static_cast<float>(data[2*(i*DIM + j)]);
                hostParticles[i*DIM+j].y = static_cast<float>(data[2*(i*DIM + j) + 1]);
            }
        }
    }
    err = cudaMemcpy(deviceParticles, hostParticles, DS*sizeof(cData), cudaMemcpyHostToDevice);
    Expect(err == cudaSuccess);
}


static void click(int button, int updown, int x, int y) {
    lastx = x; lasty = y;
    clicked = !clicked;
}

static void motion(int x, int y) {
    // Convert motion coordinates to domain
    float fx = (lastx / (float)wWidth);        
    float fy = (lasty / (float)wHeight);
    int nx = (int)(fx * DIM);        
    int ny = (int)(fy * DIM);   
    
    if (clicked && nx < DIM-FR && nx > FR-1 && ny < DIM-FR && ny > FR-1) {
        int ddx = x - lastx;
        int ddy = y - lasty;
        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny-FR;
        int spx = nx-FR;
        addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
        lastx = x; lasty = y;
    } 
}

static void FluidDynamics_cleanup() {
    unbindTexture();
    deleteTexture();

    // Free all host and device resources
    if (hvfield != NULL) {
        free(hvfield);
        hvfield = NULL;
    }
    if (hostParticles != NULL) {
        free(hostParticles); 
        hostParticles = NULL;
    }
    if (dvfield != NULL) {
        cudaFree(dvfield);
        dvfield = NULL;
    }
    if (deviceParticles != NULL) {
        cudaFree(deviceParticles);
        deviceParticles = NULL;
    }
    if (vxfield != NULL) {
        cudaFree(vxfield);
        vxfield = NULL;
    }
    if (vyfield != NULL) {
        cudaFree(vyfield);
        vyfield = NULL;
    }
    cufftDestroy(planr2c);
    cufftDestroy(planc2r);
}

static void initialize(mint dim) {
    if (dim != -1) {
        SetDim(dim);
    }

    hvfield = (cData*)malloc(sizeof(cData) * DS);
    memset(hvfield, 0, sizeof(cData) * DS);
  
    // Allocate and initialize device data
    cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData)*DIM, DIM);

    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, 
               cudaMemcpyHostToDevice); 

    // Temporary complex velocity field data     
    cudaMalloc((void**)&vxfield, sizeof(cData) * PDS);
    cudaMalloc((void**)&vyfield, sizeof(cData) * PDS);
    
    setupTexture(DIM, DIM);
    bindTexture();
    
    // Create particle array
    hostParticles = (cData*)malloc(sizeof(cData) * DS);
    cudaMalloc((void **) &deviceParticles, sizeof(cData) * DS);
    memset(hostParticles, 0, sizeof(cData) * DS);   
    
    resetParticles(); 

    // Create CUFFT transform plan configuration
    cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C);
    cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R);
}

EXTERN_C DLLEXPORT int oFluidDynamics_SetTimeDelta(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	float val;
	
	Assert(Argc == 1);
	
	val = static_cast<float>(MArgument_getReal(Args[0]));
	
	setTimeDelta(val);
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_SetViscosity(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	float val;
	
	Assert(Argc == 1);
	
	val = static_cast<float>(MArgument_getReal(Args[0]));
	
	setViscosity(val);
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_SetForce(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	float val;
	
	Assert(Argc == 1);
	
	val = static_cast<float>(MArgument_getReal(Args[0]));
	
	setForce(val);
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_SetForceRadius(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
	mint val;
	
	Assert(Argc == 1);
	
	val = MArgument_getInteger(Args[0]);
	
	setForceRadius(val);
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_Initialize(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    mint dim;
    
    dim = MArgument_getInteger(Args[0]);

    initialize(dim);

    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_Motion(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    mint x, y;
    
    x = MArgument_getInteger(Args[0]);
    y = MArgument_getInteger(Args[1]);

    motion(x, y);

    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_MouseMovement(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    mint button, updown, x, y;
    
    button = MArgument_getInteger(Args[0]);
    updown = MArgument_getInteger(Args[1]);
    x = MArgument_getInteger(Args[2]);
    y = MArgument_getInteger(Args[3]);

    click(button, updown, x, y);

    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_StepAsParticles(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    MTensor tensor;
    cData * mem;
    mint dim;

    tensor = MArgument_getMTensor(Args[0]);
    dim = MArgument_getInteger(Args[1]);

    mem = simulateFluids(dim);
    Expect(mem != NULL);

    for (mint ii = 0; ii < libData->MTensor_getFlattenedLength(tensor)/2; ii++) {
        libData->MTensor_getRealData(tensor)[2*ii] = static_cast<double>(mem[ii].x);
        libData->MTensor_getRealData(tensor)[2*ii+1] = static_cast<double>(mem[ii].y);
    }
    MArgument_setMTensor(Res, tensor);
    libData->MTensor_disown(tensor);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_StepAsPixels(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    mint dim;
    MTensor tensor;
    cData * mem;
    double * tensorData;

    tensor = MArgument_getMTensor(Args[0]);
    dim = MArgument_getInteger(Args[1]);

    mem = simulateFluids(dim);
    Expect(mem != NULL);
    tensorData = libData->MTensor_getRealData(tensor);
    memset(tensorData, 0, libData->MTensor_getFlattenedLength(tensor)*sizeof(double));
    for (mint ii = 0; ii < libData->MTensor_getFlattenedLength(tensor); ii++) {
        tensorData[(int) mem[ii].x + ((int) mem[ii].y * libData->MTensor_getDimensions(tensor)[1])] += 1;
    }

    MArgument_setMTensor(Res, tensor);

    libData->MTensor_disown(tensor);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_ResetParticles(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    resetParticles();
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oFluidDynamics_SetParticles(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    MTensor tensor;
    tensor = MArgument_getMTensor(Args[0]);
    setParticles(libData->MTensor_getRealData(tensor), libData->MTensor_getFlattenedLength(tensor));
    libData->MTensor_disown(tensor);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT mint oFluidDynamics_Uninitialize(WolframLibraryData libData, mint Argc,MArgument * Args, MArgument Res) {
    FluidDynamics_cleanup();
    return LIBRARY_NO_ERROR;
}

