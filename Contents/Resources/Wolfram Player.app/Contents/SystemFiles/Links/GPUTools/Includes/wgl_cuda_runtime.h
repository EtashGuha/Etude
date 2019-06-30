/*********************************************************************//**
* @file
*
* @section LICENCE
*
*              Mathematica source file
*
*  Copyright 1986 through 2011 by Wolfram Research Inc.
*
* This material contains trade secrets and may be registered with the
* U.S. Copyright Office as an unpublished work, pursuant to Title 17,
* U.S. Code, Section 408.  Unauthorized copying, adaptation, distribution
* or display is prohibited.
*
* @section DESCRIPTION
*
*
*
* $Id$
************************************************************************/

#include	<wgl_options.h>

#ifdef CONFIG_USE_CUDA
#ifndef __WGL_CUDA_RUNTIME_H__
#define __WGL_CUDA_RUNTIME_H__

#include	<cuda.h>
#include	<cuda_runtime.h>

#include	<math_constants.h>

#include	<wgl_types.h>


#define Ceil(x, y)											(((x) + (y) - 1)/(y))

WGLEXPORT void CUDA_Runtime_synchronize(WGL_Error_t err);

WGLEXPORT void * CUDA_Runtime_getDeviceMemory(WGL_Memory_t mem);

#define CUDA_Runtime_getDeviceMemoryAsChar(mem)              ((char *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsChar2(mem)             ((char2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsChar3(mem)             ((char3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsChar4(mem)             ((char4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsUnsignedChar(mem)      ((uchar *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedChar2(mem)     ((uchar2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedChar3(mem)     ((uchar3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedChar4(mem)     ((uchar4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsShort(mem)             ((short *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsShort2(mem)            ((short2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsShort3(mem)            ((short3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsShort4(mem)            ((short4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsUnsignedShort(mem)     ((unsigned short *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedShort2(mem)    ((ushort2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedShort3(mem)    ((ushort3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedShort4(mem)    ((ushort4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsInteger(mem)           ((int *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsInteger2(mem)          ((int2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsInteger3(mem)          ((int3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsInteger4(mem)          ((int4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsUnsignedInteger(mem)   ((unsigned int *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedInteger2(mem)  ((uint2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedInteger3(mem)  ((uint3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsUnsignedInteger4(mem)  ((uint4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsLong(mem)				 ((int64_t *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsLong2(mem)			 ((longlong2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsLong3(mem)			 ((longlong3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsLong4(mem)			 ((longlong4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsFloat(mem)             ((float *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsFloat2(mem)            ((float2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsFloat3(mem)            ((float3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsFloat4(mem)            ((float4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsDouble(mem)            ((double *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsDouble2(mem)           ((double2 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsDouble3(mem)           ((double3 *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsDouble4(mem)           ((double4 *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsComplexFloat(mem)      ((cuFloatComplex *) CUDA_Runtime_getDeviceMemory(mem))
#define CUDA_Runtime_getDeviceMemoryAsComplexDouble(mem)     ((cuDoubleComplex *) CUDA_Runtime_getDeviceMemory(mem))

#define CUDA_Runtime_getDeviceMemoryAsMInteger(mem)          ((mint *) CUDA_Runtime_getDeviceMemory(mem))


WGLEXPORT void CUDA_Runtime_setMemoryAsNeeded(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_unsetMemoryAsNeeded(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsInput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsOutput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsInputOutput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsValidOutput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsInvalidOutput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsValidInputOutput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_setMemoryAsInvalidInputOutput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_unsetMemoryAsInput(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_synchronizeMemoryToDevice(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);
WGLEXPORT void CUDA_Runtime_synchronizeMemoryToHost(WGL_State_t state, WGL_Memory_t mem, WGL_Error_t err);


#endif /* __WGL_CUDA_RUNTIME_H__ */
#endif /* CONFIG_USE_CUDA */

