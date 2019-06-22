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
* $Id: addTwo.cu,v 1.6 2010/09/29 21:30:36 abduld Exp $
************************************************************************/


#include	<wgl.h>
#include	<wgl_types.h>

#define wglState							(wglData->state)

#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (!wglData->successQ(wglData)) { goto jmp; }

static WolframGPULibraryData wglData = NULL;

const char src[] =
	"__kernel void addTwo(__global mint * in, __global mint * out, mint length) {\n" \
    "	int index = get_global_id(0);\n"					 \
    "	if (index < length)\n"								 \
    "		out[index] += in[index]  + 2;\n"				 \
	"}";


static int iAddTwo(WGL_Memory_t input, WGL_Memory_t output) {
	
	wglData->newProgramFromSource(wglData, src, NULL);
	wglData->setKernel(wglData, "addTwo");
	wglData->setKernelMemoryArgument(wglData, input, WGL_Memory_Argument_Input);
	wglData->setKernelMemoryArgument(wglData, output, WGL_Memory_Argument_Output);
	wglData->setKernelLongArgument(wglData, input->flattenedLength);
	wglData->setBlockDimensions(wglData, 1, 16, 1, 1);
	wglData->setGridDimensions(wglData, 1, input->flattenedLength, 1, 1);

	wglData->launchKernel(wglData);
	
	if (wglData->successQ(wglData)) {
		return LIBRARY_NO_ERROR;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}

EXTERN_C DLLEXPORT int oAddTwo(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t input, output;
	mint inputId, outputId;
	MTensor gridDim, blockDim;
	int err = LIBRARY_FUNCTION_ERROR;
	
	inputId 					= MArgument_getInteger(Args[0]);
	outputId 					= MArgument_getInteger(Args[1]);
	gridDim						= MArgument_getMTensor(Args[2]);
	blockDim					= MArgument_getMTensor(Args[3]);

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

