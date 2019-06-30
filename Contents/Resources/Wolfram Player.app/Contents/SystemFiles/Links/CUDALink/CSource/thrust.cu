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
* $Id: wgl.h,v 1.5 2010/08/15 11:54:25 abduld Exp $
************************************************************************/


#include	<wgl.h>
#include	<wgl_cuda_runtime.h>

#include	<thrust/host_vector.h>
#include	<thrust/device_vector.h>
#include	<thrust/device_ptr.h>
#include	<thrust/generate.h>
#include	<thrust/functional.h>
#include	<thrust/sort.h>
#include	<thrust/scan.h>

#include	<assert.h>

#define wglState							(wglData->state)
#define wglErr								(wglData->getError(wglData))
#define WGL_SuccessQ						(wglErr->code == WGL_Success)
#define WGL_FailQ							(!WGL_SuccessQ)
#define WGL_Type_RealQ(mem)					((mem)->type == WGL_Real_t)

#define StringSameQ(op, str)				(strcmp(op, str) == 0)

#define WGL_SAFE_CALL(stmt, jmp)			stmt; if (WGL_FailQ) { goto jmp; }

#if (defined(__apple) || defined(__apple__) || defined(__APPLE__))
#define APPLEQ
#endif /* (defined(__apple) || defined(__apple__) || defined(__APPLE__)) */

extern WolframGPULibraryData wglData;

#define Less(type)																								\
	template<>																									\
	__host__ __device__ bool myLess<type##2>::operator()(const type##2 & a, const type##2 & b) const {	        \
		if(a.x < b.x) {																							\
			return true;																						\
		} else if (a.x == b.x) {																				\
			if (a.y < b.y) {																					\
				return true;																					\
			}																									\
		}																										\
		return false;																							\
	}																											\
	template<>																									\
	__host__ __device__ bool myLess<type##3>::operator()(const type##3 & a, const type##3 & b) const {	        \
		if(a.x < b.x) {																							\
			return true;																						\
		} else if (a.x == b.x) {																				\
			if (a.y < b.y) {																					\
				return true;																					\
			} else if (a.y == b.y) {																			\
				return a.z < b.z;																				\
			}																									\
		}																										\
		return false;																							\
	}																											\
	template<>																									\
	__host__ __device__ bool myLess<type##4>::operator()(const type##4 & a, const type##4 & b) const {	        \
		if(a.x < b.x) {																							\
			return true;																						\
		} else if (a.x == b.x) {																				\
			if (a.y < b.y) {																					\
				return true;																					\
			} else if (a.y == b.y) {																			\
				if (a.z < b.z) {																				\
					return true;																				\
				} else {																						\
					return a.w < b.w;																			\
				}																								\
			}																									\
		}																										\
		return false;																							\
	}

#define Greater(type)																							\
	template<>																									\
	__host__ __device__ bool myGreater<type##2>::operator()(const type##2 & a, const type##2 & b) const {	    \
		if(a.x > b.x) {																							\
			return true;																						\
		} else if (a.x == b.x) {																				\
			if (a.y > b.y) {																					\
				return true;																					\
			}																									\
		}																										\
		return false;																							\
	}																											\
	template<>																									\
	__host__ __device__ bool myGreater<type##3>::operator()(const type##3 & a, const type##3 & b) const {	    \
		if(a.x > b.x) {																							\
			return true;																						\
		} else if (a.x == b.x) {																				\
			if (a.y > b.y) {																					\
				return true;																					\
			} else if (a.y == b.y) {																			\
				return a.z > b.z;																				\
			}																									\
		}																										\
		return false;																							\
	}																											\
	template<>																									\
	__host__ __device__ bool myGreater<type##4>::operator()(const type##4 & a, const type##4 & b) const {	    \
		if(a.x > b.x) {																							\
			return true;																						\
		} else if (a.x == b.x) {																				\
			if (a.y > b.y) {																					\
				return true;																					\
			} else if (a.y == b.y) {																			\
				if (a.z > b.z) {																				\
					return true;																				\
				} else {																						\
					return a.w > b.w;																			\
				}																								\
			}																									\
		}																										\
		return false;																							\
	}

template <typename T>
struct myLess : public thrust::less<T> {
    __host__ __device__ bool operator()(const T & a, const T &b) const {
        return a < b;
    }
};

template <typename T>
struct myGreater : public thrust::greater<T> {
    __host__ __device__ bool operator()(const T & a, const T &b) const {
        return a > b;
    }
};
	
#ifndef APPLEQ
Less(char);
Less(uchar);
Less(short);
Less(ushort);
Less(int);
Less(uint);
Less(float);
Less(longlong);
Less(ulonglong); 

Greater(char);
Greater(uchar);
Greater(short);
Greater(ushort);
Greater(int);
Greater(uint);
Greater(float);
Greater(longlong);
Greater(ulonglong); 

#ifdef CONFIG_USE_DOUBLE_PRECISION
Less(double);
Greater(double);
#endif /* CONFIG_USE_DOUBLE_PRECISION */

#endif /* APPLEQ */

template <typename ToT, typename FromT>													
__global__ void cast_operator(ToT * outputData, FromT * inputData, mint len) {
	const int index = threadIdx.x + blockIdx.x * blockDim.x;			
	if (index < len) {													
		outputData[index] = (ToT) (inputData[index]);	
	}																	
}

template <typename ToT, typename FromT>		
static void iCast(ToT * outputData, FromT * inputData, mint len) {
	dim3 blockDim(256);
	dim3 gridDim(Ceil(len, blockDim.x));
	cast_operator<ToT, FromT><<<gridDim, blockDim>>>(outputData, inputData, len);
}

template <typename T>
static int iCUDASort(T * data, mint len, const char * compareOperator) {
	thrust::device_ptr<T> dev_ptr(data);
	
	thrust::sort(dev_ptr, dev_ptr + len, myLess<T>());
	if (StringSameQ(compareOperator, "Greater")) {
		thrust::reverse(dev_ptr, dev_ptr + len);
	}
	
	return LIBRARY_NO_ERROR;
}

typedef unsigned short ushort;

#define doCastedSort(ToT, FromT, ew)	\
	do {\
	WGL_Memory_t tmp = wglData->newMemory(wglData, WGL_Type_Integer, 1, mem->dimensions); \
	if (tmp == NULL || !wglData->successQ(wglData)) { \
		goto cleanup; \
	} \
	CUDA_Runtime_setMemoryAsInputOutput(wglState, tmp, wglErr); \
	iCast<ToT, FromT>((ToT *) CUDA_Runtime_getDeviceMemory(tmp), (FromT *) CUDA_Runtime_getDeviceMemory(tmp), tmp->flattenedLength); \
	err = iCUDASort<ToT ## ew>((ToT ## ew *) CUDA_Runtime_getDeviceMemory(tmp), mem->flattenedLength/mem->elementWidth, compareOperator); \
	iCast<FromT, ToT>((FromT *) CUDA_Runtime_getDeviceMemory(mem), (ToT *) CUDA_Runtime_getDeviceMemory(tmp), tmp->flattenedLength); \
	wglData->freeMemory(wglData, tmp);\
	} while(0)

EXTERN_C DLLEXPORT int oCUDASort(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t mem;
	char * compareOperator;
	mint id;
	int err;
	
	assert(Argc == 4);
	
	id = MArgument_getInteger(Args[0]);
	compareOperator = MArgument_getUTF8String(Args[1]);
	
	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	mem = wglData->findMemory(wglData, id);
	if (mem == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}
	
	CUDA_Runtime_setMemoryAsInputOutput(wglState, mem, wglErr);
	
	switch (mem->type) {
		case WGL_Type_Char:
			err = iCUDASort<char>(CUDA_Runtime_getDeviceMemoryAsChar(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_Char2:
			doCastedSort(int, char, 2);
			break ;
		case WGL_Type_Char3:
			doCastedSort(int, char, 3);
			break ;
		case WGL_Type_Char4:
			doCastedSort(int, char, 4);
			break ;
#endif /* APPLEQ */
		case WGL_Type_UnsignedChar:
			err = iCUDASort<uchar>(CUDA_Runtime_getDeviceMemoryAsUnsignedChar(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_UnsignedChar2:
			doCastedSort(int, uchar, 2);
			break ;
		case WGL_Type_UnsignedChar3:
			doCastedSort(int, uchar, 3);
			break ;
		case WGL_Type_UnsignedChar4:
			doCastedSort(int, uchar, 4);
			break ;
#endif /* APPLEQ */
		case WGL_Type_Short:
			err = iCUDASort<short>(CUDA_Runtime_getDeviceMemoryAsShort(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_Short2:
			doCastedSort(int, short, 2);
			break ;
		case WGL_Type_Short3:
			doCastedSort(int, short, 3);
			break ;
		case WGL_Type_Short4:
			doCastedSort(int, short, 4);
			break ;
#endif /* APPLEQ */
		case WGL_Type_UnsignedShort:
			err = iCUDASort<unsigned short>(CUDA_Runtime_getDeviceMemoryAsUnsignedShort(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_UnsignedShort2:
			doCastedSort(int, ushort, 2);
			break ;
		case WGL_Type_UnsignedShort3:
			doCastedSort(int, ushort, 3);
			break ;
		case WGL_Type_UnsignedShort4:
			doCastedSort(int, ushort, 4);
			break ;
#endif /* APPLEQ */
		case WGL_Type_Integer:
			err = iCUDASort<int>(CUDA_Runtime_getDeviceMemoryAsInteger(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_Integer2:
			err = iCUDASort<int2>(CUDA_Runtime_getDeviceMemoryAsInteger2(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Integer3:
			err = iCUDASort<int3>(CUDA_Runtime_getDeviceMemoryAsInteger3(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Integer4:
			err = iCUDASort<int4>(CUDA_Runtime_getDeviceMemoryAsInteger4(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#endif /* APPLEQ */
		case WGL_Type_UnsignedInteger:
			err = iCUDASort<unsigned int>(CUDA_Runtime_getDeviceMemoryAsUnsignedInteger(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_UnsignedInteger2:
			err = iCUDASort<uint2>(CUDA_Runtime_getDeviceMemoryAsUnsignedInteger2(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_UnsignedInteger3:
			err = iCUDASort<uint3>(CUDA_Runtime_getDeviceMemoryAsUnsignedInteger3(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_UnsignedInteger4:
			err = iCUDASort<uint4>(CUDA_Runtime_getDeviceMemoryAsUnsignedInteger4(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#endif /* APPLEQ */
		case WGL_Type_Long:
			err = iCUDASort<int64_t>(CUDA_Runtime_getDeviceMemoryAsLong(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_Long2:
			err = iCUDASort<longlong2>(CUDA_Runtime_getDeviceMemoryAsLong2(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Long3:
			err = iCUDASort<longlong3>(CUDA_Runtime_getDeviceMemoryAsLong3(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Long4:
			err = iCUDASort<longlong4>(CUDA_Runtime_getDeviceMemoryAsLong4(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#endif /* APPLEQ */
		case WGL_Type_Float:
			err = iCUDASort<float>(CUDA_Runtime_getDeviceMemoryAsFloat(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_Float2:
			err = iCUDASort<float2>(CUDA_Runtime_getDeviceMemoryAsFloat2(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Float3:
			err = iCUDASort<float3>(CUDA_Runtime_getDeviceMemoryAsFloat3(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Float4:
			err = iCUDASort<float4>(CUDA_Runtime_getDeviceMemoryAsFloat4(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#endif /* APPLEQ */
#ifdef CONFIG_USE_DOUBLE_PRECISION
		case WGL_Type_Double:
			err = iCUDASort<double>(CUDA_Runtime_getDeviceMemoryAsDouble(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#ifndef APPLEQ
		case WGL_Type_Double2:
			err = iCUDASort<double2>(CUDA_Runtime_getDeviceMemoryAsDouble2(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Double3:
			err = iCUDASort<double3>(CUDA_Runtime_getDeviceMemoryAsDouble3(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
		case WGL_Type_Double4:
			err = iCUDASort<double4>(CUDA_Runtime_getDeviceMemoryAsDouble4(mem), mem->flattenedLength/mem->elementWidth, compareOperator);
			break ;
#endif /* APPLEQ */
#endif /* CONFIG_USE_DOUBLE_PRECISION */
		default:
			err = LIBRARY_TYPE_ERROR;
	}
cleanup:	
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, mem, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, mem, wglErr);
	}
	libData->UTF8String_disown(compareOperator);

	return err;
}


template <typename T>
static int iCUDAScan(double initialValue, T * input, T * output, int len, const char * op) {
	thrust::device_ptr<T> inputPtr(input);
	thrust::device_ptr<T> outputPtr(output);
	
	if (StringSameQ(op, "Max")) {
		thrust::exclusive_scan(inputPtr, inputPtr + len, outputPtr, static_cast<T>(initialValue), thrust::maximum<T>());
	} else if (StringSameQ(op, "Min")) {
		thrust::exclusive_scan(inputPtr, inputPtr + len, outputPtr, static_cast<T>(initialValue), thrust::minimum<T>());
	} else if (StringSameQ(op, "Plus")) {
		thrust::exclusive_scan(inputPtr, inputPtr + len, outputPtr, static_cast<T>(initialValue), thrust::plus<T>());
	} else if (StringSameQ(op, "Minus")) {
		thrust::exclusive_scan(inputPtr, inputPtr + len, outputPtr, static_cast<T>(initialValue), thrust::minus<T>());
	} else if (StringSameQ(op, "Times")) {
		thrust::exclusive_scan(inputPtr, inputPtr + len, outputPtr, static_cast<T>(initialValue), thrust::multiplies<T>());
	} else if (StringSameQ(op, "Divide")) {
		thrust::exclusive_scan(inputPtr, inputPtr + len, outputPtr, static_cast<T>(initialValue), thrust::divides<T>());
	}
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oCUDAScan(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t inputMem, outputMem;
	char * op;
	mint inputId, outputId;
	double initialValue;
	int err;
	
	assert(Argc == 6);
	
	initialValue = MArgument_getReal(Args[0]);
	inputId = MArgument_getInteger(Args[1]);
	outputId = MArgument_getInteger(Args[2]);
	op = MArgument_getUTF8String(Args[3]);
	
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
			err = iCUDAScan<char>(initialValue, CUDA_Runtime_getDeviceMemoryAsChar(inputMem), CUDA_Runtime_getDeviceMemoryAsChar(outputMem), outputMem->flattenedLength/outputMem->elementWidth, op);
			break ;
		case WGL_Type_UnsignedChar:
			err = iCUDAScan<uchar>(initialValue, CUDA_Runtime_getDeviceMemoryAsUnsignedChar(inputMem), CUDA_Runtime_getDeviceMemoryAsUnsignedChar(outputMem), outputMem->flattenedLength/outputMem->elementWidth, op);
			break ;
		case WGL_Type_Integer:
			err = iCUDAScan<int>(initialValue, CUDA_Runtime_getDeviceMemoryAsInteger(inputMem), CUDA_Runtime_getDeviceMemoryAsInteger(outputMem), outputMem->flattenedLength/outputMem->elementWidth, op);
			break ;
		case WGL_Type_Long:
			err = iCUDAScan<int64_t>(initialValue, CUDA_Runtime_getDeviceMemoryAsLong(inputMem), CUDA_Runtime_getDeviceMemoryAsLong(outputMem), outputMem->flattenedLength/outputMem->elementWidth, op);
			break ;
		case WGL_Type_Float:
			err = iCUDAScan<float>(initialValue, CUDA_Runtime_getDeviceMemoryAsFloat(inputMem), CUDA_Runtime_getDeviceMemoryAsFloat(outputMem), outputMem->flattenedLength/outputMem->elementWidth, op);
			break ;
#ifdef CONFIG_USE_DOUBLE_PRECISION
		case WGL_Type_Double:
			err = iCUDAScan<double>(initialValue, CUDA_Runtime_getDeviceMemoryAsDouble(inputMem), CUDA_Runtime_getDeviceMemoryAsDouble(outputMem), outputMem->flattenedLength/outputMem->elementWidth, op);
			break ;
#endif /* CONFIG_USE_DOUBLE_PRECISION */
		default:
			err = LIBRARY_TYPE_ERROR;
	}
cleanup:	
	if (err == LIBRARY_NO_ERROR && WGL_SuccessQ) {
		CUDA_Runtime_setMemoryAsValidOutput(wglState, outputMem, wglErr);
	} else {
		CUDA_Runtime_setMemoryAsInvalidOutput(wglState, outputMem, wglErr);
	}
	CUDA_Runtime_unsetMemoryAsInput(wglState, inputMem, wglErr);

	libData->UTF8String_disown(op);
	
	return err;
}

template <typename T>
static int iCUDAReduce(double * res, double initialValue, T * input, int len, const char * op) {
	thrust::device_ptr<T> inputPtr(input);

	if (StringSameQ(op, "Max")) {
		*res = thrust::reduce(inputPtr, inputPtr + len, static_cast<T>(initialValue), thrust::maximum<T>());
	} else if (StringSameQ(op, "Min")) {
		*res = thrust::reduce(inputPtr, inputPtr + len, static_cast<T>(initialValue), thrust::minimum<T>());
	} else if (StringSameQ(op, "Plus")) {
		*res = thrust::reduce(inputPtr, inputPtr + len, static_cast<T>(initialValue), thrust::plus<T>());
	} else if (StringSameQ(op, "Minus")) {
		*res = thrust::reduce(inputPtr, inputPtr + len, static_cast<T>(initialValue), thrust::minus<T>());
	} else if (StringSameQ(op, "Times")) {
		*res = thrust::reduce(inputPtr, inputPtr + len, static_cast<T>(initialValue), thrust::multiplies<T>());
	} else if (StringSameQ(op, "Divide")) {
		*res = thrust::reduce(inputPtr, inputPtr + len, static_cast<T>(initialValue), thrust::divides<T>());
	} else {
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oCUDAReduce(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	WGL_Memory_t inputMem;
	char * op;
	mint inputId;
	double initialValue;
	MTensor resTensor;
	double * res;
	int err;
	
	assert(Argc == 6);
	
	resTensor = MArgument_getMTensor(Args[0]);
	initialValue = MArgument_getReal(Args[1]);
	inputId = MArgument_getInteger(Args[2]);
	op = MArgument_getUTF8String(Args[3]);
	
	WGL_SAFE_CALL(wglData->setWolframLibraryData(wglData, libData), cleanup);

	res = libData->MTensor_getRealData(resTensor);

	WGL_SAFE_CALL(inputMem = wglData->findMemory(wglData, inputId), cleanup);

	if (inputMem-> rank != 1) {
		err = LIBRARY_DIMENSION_ERROR;
		goto cleanup;
	}
	
	CUDA_Runtime_setMemoryAsInput(wglState, inputMem, wglErr);
	
	switch (inputMem->type) {
		case WGL_Type_Char:
			err = iCUDAReduce<char>(res, initialValue, CUDA_Runtime_getDeviceMemoryAsChar(inputMem), inputMem->flattenedLength/inputMem->elementWidth, op);
			break ;
		case WGL_Type_UnsignedChar:
			err = iCUDAReduce<uchar>(res, initialValue, CUDA_Runtime_getDeviceMemoryAsUnsignedChar(inputMem), inputMem->flattenedLength/inputMem->elementWidth, op);
			break ;
		case WGL_Type_Integer:
			err = iCUDAReduce<int>(res, initialValue, CUDA_Runtime_getDeviceMemoryAsInteger(inputMem), inputMem->flattenedLength/inputMem->elementWidth, op);
			break ;
		case WGL_Type_Long:
			err = iCUDAReduce<int64_t>(res, initialValue, CUDA_Runtime_getDeviceMemoryAsLong(inputMem), inputMem->flattenedLength/inputMem->elementWidth, op);
			break ;
		case WGL_Type_Float:
			err = iCUDAReduce<float>(res, initialValue, CUDA_Runtime_getDeviceMemoryAsFloat(inputMem), inputMem->flattenedLength/inputMem->elementWidth, op);
			break ;
#ifdef CONFIG_USE_DOUBLE_PRECISION
		case WGL_Type_Double:
			err = iCUDAReduce<double>(res, initialValue, CUDA_Runtime_getDeviceMemoryAsDouble(inputMem), inputMem->flattenedLength/inputMem->elementWidth, op);
			break ;
#endif /* CONFIG_USE_DOUBLE_PRECISION */
		default:
			err = LIBRARY_TYPE_ERROR;
	}
cleanup:

	CUDA_Runtime_unsetMemoryAsInput(wglState, inputMem, wglErr);

	libData->UTF8String_disown(op);
	
	return err;
}


/************************************************************************
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
************************************************************************/
