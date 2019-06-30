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

#define wglState							(wglData->state)
#define wglErr								(wglData->getError(wglData))
#define WGL_SuccessQ						(wglErr->code == WGL_Success)
#define WGL_FailQ							(!WGL_SuccessQ)

#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (WGL_FailQ) { goto jmp; }

WolframGPULibraryData wglData = NULL;


__constant__ mint d_rule[8];

__device__ static int getIndex(const int & left, const int & top, const int & right) {
    return (left << 2) + (top << 1) + right;
}

__global__ void ca_kernel(mint * prevRow, mint * nextRow, mint width) {
    extern __shared__ mint smem[];
    int tx = threadIdx.x, bx = blockIdx.x;
    int dx = blockDim.x;
    int index = tx + bx*dx;

    smem[tx+1] = index < width ? prevRow[index] : 0;
    if (tx == 0)
        smem[0] = index == 0 ? 0 : prevRow[index-1];
    if (tx == dx-1)
        smem[dx+1] = index == width-1 ? 0 : prevRow[index+1];
    
    __syncthreads();
    
    if (index < width)
        nextRow[index] = d_rule[getIndex(smem[tx], smem[tx+1], smem[tx+2])];

}

static int iCellularAutomaton(mint * mem, mint width, mint steps) {
	mint ii;
	mint * prev, * next;
	
	dim3 blockDim(256);
	dim3 gridDim(Ceil(width, blockDim.x));
	
	prev = mem;
	next = &mem[width];
	
	if (steps == 1) {
		ca_kernel<<<gridDim, blockDim, (blockDim.x+2)*sizeof(mint)>>>(prev, next, width);
		CUDA_Runtime_synchronize(wglErr);
	} else {
		for (ii = 1; ii < steps; ii++) {
			ca_kernel<<<gridDim, blockDim, (blockDim.x+2)*sizeof(mint)>>>(prev, next, width);
			WGL_SAFE_CALL(CUDA_Runtime_synchronize(wglErr), cleanup);
			
			prev = &mem[ii * width];
			next = &mem[(ii+1) * width];
		}
	}
cleanup:
	if (WGL_SuccessQ) {
		return LIBRARY_NO_ERROR;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}

static int iCellularAutomaton(WGL_Memory_t mem, mint steps, mint rule) {
	int err = LIBRARY_FUNCTION_ERROR;
	mint ii, width, * ruleArray;
	WGL_Type_t type;
	mint * hostData;
	
	if (mem->rank != 2) {
		return LIBRARY_RANK_ERROR;
	}
	
	type = mem->type;
	if (type != WGL_Type_Integer) {
		return LIBRARY_DIMENSION_ERROR;
	}

	width = mem->dimensions[1];
	
	if (mem->type == WGL_Type_Integer) {
		hostData = (mint*) wglData->MTensorMemory_getIntegerData(wglData, mem);
	} else {
		hostData = (mint*) wglData->MTensorMemory_getLongData(wglData, mem);
	}
	
	memset(hostData, 0, width * mem->elementSize);
	hostData[width/2] = 1;

	ruleArray = (mint *) malloc(8 * sizeof(mint));

    for (ii = 0; ii < 8; ii++)
        ruleArray[ii] = (rule >> ii) & 1;  
	
    err = cudaMemcpyToSymbol(d_rule, ruleArray, 8 * sizeof(mint));
    if (err != cudaSuccess) {
    	goto cleanup;
    }
	
	WGL_SAFE_CALL(CUDA_Runtime_setMemoryAsInputOutput(wglState, mem, wglErr), cleanup);
	
	iCellularAutomaton((mint *) CUDA_Runtime_getDeviceMemory(mem), width, steps);
	
cleanup:
	free(ruleArray);
	if (WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, mem, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, mem, wglErr);
	}

	if (WGL_SuccessQ && err == LIBRARY_NO_ERROR) {
		return LIBRARY_NO_ERROR;
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
}

EXTERN_C DLLEXPORT int oCellularAutomaton(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t mem;
	mint memId, steps, rule;
	int err = LIBRARY_FUNCTION_ERROR;
	
	memId 						= MArgument_getInteger(Args[0]);
	steps						= MArgument_getInteger(Args[1]);
	rule						= MArgument_getInteger(Args[2]);

	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	WGL_SAFE_CALL(mem = wglData->findMemory(wglData, memId), cleanup);

	err = iCellularAutomaton(mem, steps, rule);

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



