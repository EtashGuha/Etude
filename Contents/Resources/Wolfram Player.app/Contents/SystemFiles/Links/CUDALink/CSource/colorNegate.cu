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
* $Id: clamp.cu,v 1.8 2010/09/29 08:40:55 abduld Exp $
************************************************************************/


#include	<wgl.h>
#include	<wgl_cuda_runtime.h>

#ifdef CONFIG_USE_DOUBLE_PRECISION
#define Real_t								double
#define WGL_Real_t							WGL_Type_Double
#define CUDA_Runtime_getDeviceMemoryAsReal	CUDA_Runtime_getDeviceMemoryAsDouble
#else
#define Real_t								float
#define WGL_Real_t							WGL_Type_Float
#define CUDA_Runtime_getDeviceMemoryAsReal	CUDA_Runtime_getDeviceMemoryAsFloat
#endif

#define wglState							(wglData->state)
#define wglErr								(wglData->getError(wglData))
#define WGL_SuccessQ						(wglErr->code == WGL_Success)
#define WGL_FailQ							(!WGL_SuccessQ)
#define WGL_Type_RealQ(mem)					((mem)->type == WGL_Real_t)

#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (WGL_FailQ) { goto jmp; }

extern WolframGPULibraryData wglData;

static int iCUDAColorNegate(WGL_Memory_t input, WGL_Memory_t output);

template <typename T>
__device__ T negate(const T & a) {
	return 255 - a;
}

template <>
__device__ float negate<float>(const float & a) {
	return 1.0f - a;
}

#ifdef CONFIG_USE_DOUBLE_PRECISION
template <>
__device__ double negate(const double & a) {
	return 1.0 - a;
}
#endif /* CONFIG_USE_DOUBLE_PRECISION */

template <typename T>
__global__ void colorNegate(T * in, T * out, int len) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < len) {
		out[index] = negate(in[index]);
	}
}

static int iCUDAColorNegate(WGL_Memory_t input, WGL_Memory_t output) {
	int err = LIBRARY_FUNCTION_ERROR;
	
	dim3 blockDim(256);
	dim3 gridDim(Ceil(input->flattenedLength, blockDim.x));

	if (input->type != output->type) {
		return LIBRARY_TYPE_ERROR;
	} else if (input->flattenedLength != output->flattenedLength) {
		return LIBRARY_DIMENSION_ERROR;
	}
	
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, input, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, output, wglErr), cleanup);


	switch (input->type) {
		case WGL_Type_Char:
			colorNegate<char><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsChar(input),
				CUDA_Runtime_getDeviceMemoryAsChar(output),
				input->flattenedLength
			);
			break ;
		case WGL_Type_UnsignedChar:
			colorNegate<unsigned char><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsUnsignedChar(input),
				CUDA_Runtime_getDeviceMemoryAsUnsignedChar(output),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Short:
			colorNegate<short><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsShort(input),
				CUDA_Runtime_getDeviceMemoryAsShort(output),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Integer:
			colorNegate<int><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsInteger(input),
				CUDA_Runtime_getDeviceMemoryAsInteger(output),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Long:
			colorNegate<int64_t><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsLong(input),
				CUDA_Runtime_getDeviceMemoryAsLong(output),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Float:
			colorNegate<float><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsFloat(input),
				CUDA_Runtime_getDeviceMemoryAsFloat(output),
				input->flattenedLength
			);
			break ;
#ifdef CONFIG_USE_DOUBLE_PRECISION
		case WGL_Type_Double:
			colorNegate<double><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsDouble(input),
				CUDA_Runtime_getDeviceMemoryAsDouble(output),
				input->flattenedLength
			);
			break ;
#endif /* CONFIG_USE_DOUBLE_PRECISION */
		default:
			err = LIBRARY_TYPE_ERROR;
	}

	CUDA_Runtime_synchronize(wglErr);
	if (WGL_SuccessQ) {
		err = LIBRARY_NO_ERROR;
	}
cleanup:
	if (WGL_SuccessQ && err == LIBRARY_NO_ERROR) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, output, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, output, wglErr);
	}
	CUDA_Runtime_unsetMemoryAsInput(wglState, input, wglErr);
	return err;
}

EXTERN_C DLLEXPORT int oCUDAColorNegate(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t input, output;
	mint inputId, outputId;
	int err = LIBRARY_FUNCTION_ERROR;
	
	inputId 					= MArgument_getInteger(Args[0]);
	outputId 					= MArgument_getInteger(Args[1]);

	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	WGL_SAFE_CALL(input = wglData->findMemory(wglData, inputId), cleanup);
	WGL_SAFE_CALL(output = wglData->findMemory(wglData, outputId), cleanup);

	err = iCUDAColorNegate(input, output);

cleanup:	
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else if (err != LIBRARY_NO_ERROR) {
		return err;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}
