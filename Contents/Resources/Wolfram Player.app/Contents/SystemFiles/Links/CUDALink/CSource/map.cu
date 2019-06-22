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
* $Id: apply.cu,v 1.3 2010/09/25 19:11:51 abduld Exp $
************************************************************************/


#include	<wgl.h>
#include	<wgl_cuda_runtime.h>
#include	<assert.h>

#define wglState							(wglData->state)
#define wglErr								(wglData->getError(wglData))
#define WGL_SuccessQ						(wglErr->code == WGL_Success)
#define WGL_FailQ							(!WGL_SuccessQ)

#define StringSameQ(op, str)				(strcmp(op, str) == 0)

#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (WGL_FailQ) { goto jmp; }


#ifdef CONFIG_USE_DOUBLE_PRECISION
#define Real_t								double
#else
#define Real_t								float
#endif

extern WolframGPULibraryData wglData;


#define unaryOperator(op)														\
		template <typename T>													\
		__global__ void op##_operator(T * inputData, T * outputData, mint len) {\
			const int index = threadIdx.x + blockIdx.x * blockDim.x;			\
			if (index < len) {													\
				outputData[index] = op(static_cast<Real_t>(inputData[index]));	\
			}																	\
		}

unaryOperator(cos);
unaryOperator(sin);
unaryOperator(tan);
unaryOperator(acos);
unaryOperator(asin);
unaryOperator(atan);
unaryOperator(cosh);
unaryOperator(sinh);
unaryOperator(tanh);
unaryOperator(exp);
unaryOperator(log);
unaryOperator(log10);

//unaryOperator(sqrt);
unaryOperator(ceil);
unaryOperator(floor);
unaryOperator(abs);

template <typename T>
static int iCUDAMap(T * input, T * output, mint len, const char * op) {
	dim3 blockDim(256);
	dim3 gridDim(Ceil(len, blockDim.x));

	if (StringSameQ(op, "Cos")) {
		cos_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Sin")) {
		sin_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Tan")) {
		tan_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "ArcCos")) {
		acos_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "ArcSin")) {
		asin_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "ArcTan")) {
		atan_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Cosh")) {
		cosh_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Sinh")) {
		sinh_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Tanh")) {
		tanh_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Exp")) {
		exp_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Log")) {
		log_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Log10")) {
		log10_operator<T><<<gridDim, blockDim>>>(input, output, len);
	//} else if (StringSameQ(op, "Sqrt")) {
	//	sqrt_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Ceil") || StringSameQ(op, "Ceiling")) {
		ceil_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Floor")) {
		floor_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else if (StringSameQ(op, "Abs")) {
		abs_operator<T><<<gridDim, blockDim>>>(input, output, len);
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oCUDAMap(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t inputMem, outputMem;
	char * op;
	mint inputId, outputId;
	int err;
	
	assert(Argc == 5);
	
	inputId = MArgument_getInteger(Args[0]);
	outputId = MArgument_getInteger(Args[1]);
	op = MArgument_getUTF8String(Args[2]);
	
	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	WGL_SAFE_CALL(inputMem = wglData->findMemory(wglData, inputId), cleanup);
	WGL_SAFE_CALL(outputMem = wglData->findMemory(wglData, outputId), cleanup);

	if (inputMem-> rank != 1 || outputMem->rank != 1 || outputMem->flattenedLength != inputMem->flattenedLength) {
		err = LIBRARY_DIMENSION_ERROR;
		goto cleanup;
	} else if (inputMem->type != outputMem->type) {
		err = LIBRARY_TYPE_ERROR;
		goto cleanup;
	}
	
	CUDA_Runtime_setMemoryAsInput(wglState, inputMem, wglErr);
	CUDA_Runtime_setMemoryAsOutput(wglState, outputMem, wglErr);
	
	switch (inputMem->type) {
		case WGL_Type_Char:
			err = iCUDAMap<char>(CUDA_Runtime_getDeviceMemoryAsChar(inputMem), CUDA_Runtime_getDeviceMemoryAsChar(outputMem), outputMem->flattenedLength, op);
			break ;
		case WGL_Type_UnsignedChar:
			err = iCUDAMap<uchar>(CUDA_Runtime_getDeviceMemoryAsUnsignedChar(inputMem), CUDA_Runtime_getDeviceMemoryAsUnsignedChar(outputMem), outputMem->flattenedLength, op);
			break ;
#if 0
		case WGL_Type_Short:
			err = iCUDAMap<short>(CUDA_Runtime_getDeviceMemoryAsShort(inputMem), CUDA_Runtime_getDeviceMemoryAsShort(outputMem), outputMem->flattenedLength, op);
			break ;
		case WGL_Type_UnsignedShort:
			err = iCUDAMap<unsigned short>(CUDA_Runtime_getDeviceMemoryAsUnsignedShort(inputMem), CUDA_Runtime_getDeviceMemoryAsUnsignedShort(outputMem), outputMem->flattenedLength, op);
			break ;
#endif
		case WGL_Type_Integer:
			err = iCUDAMap<int>(CUDA_Runtime_getDeviceMemoryAsInteger(inputMem), CUDA_Runtime_getDeviceMemoryAsInteger(outputMem), outputMem->flattenedLength, op);
			break ;
#if 0
		case WGL_Type_UnsignedInteger:
			err = iCUDAMap<unsigned int>(CUDA_Runtime_getDeviceMemoryAsUnsignedInteger(inputMem), CUDA_Runtime_getDeviceMemoryAsUnsignedInteger(outputMem), outputMem->flattenedLength, op);
			break ;
#endif
		case WGL_Type_Long:
			err = iCUDAMap<int64_t>(CUDA_Runtime_getDeviceMemoryAsLong(inputMem), CUDA_Runtime_getDeviceMemoryAsLong(outputMem), outputMem->flattenedLength, op);
			break ;
		case WGL_Type_Float:
			err = iCUDAMap<float>(CUDA_Runtime_getDeviceMemoryAsFloat(inputMem), CUDA_Runtime_getDeviceMemoryAsFloat(outputMem), outputMem->flattenedLength, op);
			break ;
#ifdef CONFIG_USE_DOUBLE_PRECISION
		case WGL_Type_Double:
			err = iCUDAMap<double>(CUDA_Runtime_getDeviceMemoryAsDouble(inputMem), CUDA_Runtime_getDeviceMemoryAsDouble(outputMem), outputMem->flattenedLength, op);
			break ;
#endif /* CONFIG_USE_DOUBLE_PRECISION */
		default:
			err = LIBRARY_TYPE_ERROR;
	}
cleanup:	
	if (err == LIBRARY_NO_ERROR) {
		CUDA_Runtime_synchronize(wglErr);
		if (WGL_FailQ) {
			err = LIBRARY_FUNCTION_ERROR;
		}
	}
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, outputMem, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, outputMem, wglErr);
	}
	CUDA_Runtime_unsetMemoryAsInput(wglState, inputMem, wglErr);

	libData->UTF8String_disown(op);
	
	return err;
}

