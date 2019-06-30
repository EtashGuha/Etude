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

static int iCUDAClamp(WGL_Memory_t input, WGL_Memory_t output, double low, double high);

template <typename T>
__device__ T Max(const T & a, const T & b) {
	if (a > b) {
		return a;
	} else {
		return b;
	}
}

template <typename T>
__device__ T Min(const T & a, const T & b) {
	if (a > b) {
		return b;
	} else {
		return a;
	}
}


template <typename T>
__global__ void clamp(T * in, T * out, T low, T high, int len) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < len) {
		out[index] = Max(Min(in[index], high), low);
	}
}

static int iCUDAClamp(WGL_Memory_t input, WGL_Memory_t output, double low, double high) {
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
			clamp<char><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsChar(input),
				CUDA_Runtime_getDeviceMemoryAsChar(output),
				static_cast<char>(low),
				static_cast<char>(high),
				input->flattenedLength
			);
			break ;
		case WGL_Type_UnsignedChar:
			clamp<unsigned char><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsUnsignedChar(input),
				CUDA_Runtime_getDeviceMemoryAsUnsignedChar(output),
				static_cast<unsigned char>(low),
				static_cast<unsigned char>(high),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Short:
			clamp<short><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsShort(input),
				CUDA_Runtime_getDeviceMemoryAsShort(output),
				static_cast<short>(low),
				static_cast<short>(high),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Integer:
			clamp<int><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsInteger(input),
				CUDA_Runtime_getDeviceMemoryAsInteger(output),
				static_cast<int>(low),
				static_cast<int>(high),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Long:
			clamp<int64_t><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsLong(input),
				CUDA_Runtime_getDeviceMemoryAsLong(output),
				static_cast<mint>(low),
				static_cast<mint>(high),
				input->flattenedLength
			);
			break ;
		case WGL_Type_Float:
			clamp<float><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsFloat(input),
				CUDA_Runtime_getDeviceMemoryAsFloat(output),
				static_cast<float>(low),
				static_cast<float>(high),
				input->flattenedLength
			);
			break ;
#ifdef CONFIG_USE_DOUBLE_PRECISION
		case WGL_Type_Double:
			clamp<double><<<gridDim, blockDim>>>(
				CUDA_Runtime_getDeviceMemoryAsDouble(input),
				CUDA_Runtime_getDeviceMemoryAsDouble(output),
				low, high,
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

EXTERN_C DLLEXPORT int oCUDAClamp(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t input, output;
	mint inputId, outputId;
	double low, high;
	int err = LIBRARY_FUNCTION_ERROR;
	
	inputId 					= MArgument_getInteger(Args[0]);
	outputId 					= MArgument_getInteger(Args[1]);
	low							= MArgument_getReal(Args[2]);
	high						= MArgument_getReal(Args[3]);

	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	WGL_SAFE_CALL(input = wglData->findMemory(wglData, inputId), cleanup);
	WGL_SAFE_CALL(output = wglData->findMemory(wglData, outputId), cleanup);

	err = iCUDAClamp(input, output, low, high);

cleanup:	
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else if (err != LIBRARY_NO_ERROR) {
		return err;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}
