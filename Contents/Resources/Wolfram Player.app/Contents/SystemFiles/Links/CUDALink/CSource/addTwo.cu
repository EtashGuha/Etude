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

#define wglState							(wglData->state)
#define wglErr								(wglData->getError(wglData))
#define WGL_SuccessQ						(wglErr->code == WGL_Success)
#define WGL_FailQ							(!WGL_SuccessQ)
#define WGL_Type_RealQ(mem)					((mem)->type == WGL_Real_t)

#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (WGL_FailQ) { goto jmp; }

WolframGPULibraryData wglData = NULL;

__global__ void addTwo(mint * in, mint * out, mint len) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < len) {
		out[index] = in[index] + 2;
	}
}

static int iAddTwo(WGL_Memory_t input, WGL_Memory_t output) {
	int err;
	
	dim3 blockDim(256);
	dim3 gridDim(Ceil(input->flattenedLength, blockDim.x));

	err = LIBRARY_FUNCTION_ERROR;

	if (input->type != output->type) {
		return LIBRARY_TYPE_ERROR;
	} else if (input->flattenedLength != output->flattenedLength) {
		return LIBRARY_DIMENSION_ERROR;
	}
	
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInput(wglState, input, wglErr), cleanup);
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsOutput(wglState, output, wglErr), cleanup);

	addTwo<<<gridDim, blockDim>>>(
		CUDA_Runtime_getDeviceMemoryAsMInteger(input),
		CUDA_Runtime_getDeviceMemoryAsMInteger(output),
		input->flattenedLength
	);
	err = LIBRARY_NO_ERROR;
	
cleanup:
	if (WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, output, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, output, wglErr);
	}
	CUDA_Runtime_unsetMemoryAsInput(wglState, input, wglErr);
	if (WGL_SuccessQ && err == LIBRARY_NO_ERROR) {
		return LIBRARY_NO_ERROR;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}

EXTERN_C DLLEXPORT int oAddTwo(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t input, output;
	mint inputId, outputId;
	int err = LIBRARY_FUNCTION_ERROR;
	
	inputId 					= MArgument_getInteger(Args[0]);
	outputId 					= MArgument_getInteger(Args[1]);

	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	WGL_SAFE_CALL(input = wglData->findMemory(wglData, inputId), cleanup);
	WGL_SAFE_CALL(output = wglData->findMemory(wglData, outputId), cleanup);

	err = iAddTwo(input, output);

cleanup:	
	return err;
}

EXTERN_C DLLEXPORT int WolframGPULibrary_initialize(WolframGPULibraryData wglData0) {
	wglData = wglData0;
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
   return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize( ) {
   return;
}

