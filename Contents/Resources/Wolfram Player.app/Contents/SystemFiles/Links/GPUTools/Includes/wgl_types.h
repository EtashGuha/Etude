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

#include <wgl_options.h>


#ifndef __WGL_TYPES_H__
#define __WGL_TYPES_H__

#include	<wgl.h>

#ifdef CONFIG_USE_CUDA
#include	<cuda.h>
#include	<cuda_runtime.h>
#include	<cublas_v2.h>
#include	<curand.h>
#endif /* CONFIG_USE_CUDA */

#ifdef CONFIG_USE_CUDA_FFT
#include	<cufft.h>
#endif /* CONFIG_USE_CUDA_FFT */

#ifdef CONFIG_USE_CUSPARSE
#include	<cusparse.h>
#endif /* CONFIG_USE_CUSPARSE */

#ifdef CONFIG_USE_OPENCL
#ifdef __APPLE__
#include	<OpenCL/cl.h>
#else
#include	<CL/cl.h>
#endif /* __APPLE__ */
#endif /* CONFIG_USE_OPENCL */


#define WGLEXPORT			EXTERN_C DLLEXPORT
#define WGL_Automatic		-1


#define WGL_Tree_Entry(type)															\
  struct {																				\
    struct type	*avl_left;																\
    struct type	*avl_right;																\
    int		 avl_height;																\
  }

#define WGL_Tree_Head(name, type)														\
  struct name {																			\
    struct type *th_root;																\
    mint  (*th_cmp)(struct type *lhs, struct type *rhs);								\
	mint  (*get_cmp)(struct type *lhs, mint id);										\
  }

typedef unsigned char uchar;

typedef enum {
#define WGL_Error_define(code, tag, msg)			code,
#include "wgl_error.h"
#undef WGL_Error_define
} WGL_ErrorCode_t;

#define WGL_Error_Success		WGL_Success

typedef enum en_WGL_ContextProperty_t {
	WGL_Context_Schedule_Auto = 1,
	WGL_Context_Schedule_Spin,
	WGL_Context_Schedule_Yield,
	WGL_Context_Blocking_Synchronization
} WGL_ContextProperty_t;

typedef enum en_WGL_PlatformQueryProperty_t {
	WGL_Platform_Query_NumberOfDevices = 1,
	WGL_Platform_Query_FastestDevice,
	WGL_Platform_Query_Extensions,
	WGL_Platform_Query_Vendor,
} WGL_PlatformQueryProperty_t;

typedef enum en_WGL_DeviceQueryProperty_t {
	WGL_Device_Query_ClockRate = 1,
	WGL_Device_Query_TotalMemorySize,
	WGL_Device_Query_Name,
	WGL_Device_Query_Vendor,
	WGL_Device_Query_Extensions,
	WGL_Device_Query_MultiProcessorCount,
	WGL_Device_Query_MajorComputeCapability,
	WGL_Device_Query_MinorComputeCapability,
	WGL_Device_Query_Type,
} WGL_DeviceQueryProperty_t;

typedef enum en_WGL_DeviceTypeProperty_t {
	WGL_Device_Type_GPU = 1,
	WGL_Device_Type_CPU,
	WGL_Device_Type_Accelerator
} WGL_DeviceTypeProperty_t;

typedef enum en_WGL_AddressMode_t {
	WGL_AddressMode_Wrap				= 0x00,
	WGL_AddressMode_Clamp				= 0x01,
	WGL_AddressMode_Mirror				= 0x02,
	WGL_AddressMode_Border				= 0x04
} WGL_AddressMode_t;

struct st_WGL_Error_p_t {
	WGL_ErrorCode_t code;
	char msg[WGL_SIZEOF_ERROR_MESSAGE];
	const char * fileName;
	const char * funName;
	mint lineNumber;
#ifdef CONFIG_USE_CUDA
	CUresult cudaDriver;
	cudaError_t cudaRuntime;
	enum curandStatus curandErr;
	cublasStatus_t cublasErr;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_CUDA_FFT
	cufftResult cufftErr;
#endif /* CONFIG_USE_CUDA_FFT */
#ifdef CONFIG_USE_CUSPARSE
	cusparseStatus_t cusparseErr;
#endif /* CONFIG_USE_CUSPARSE */
#ifdef CONFIG_USE_OPENCL
	cl_int ocl;
#endif /* CONFIG_USE_OPENCL */
};

typedef struct {
	WGL_API_t api;
	mint offset;
	union {
#ifdef CONFIG_USE_CUDA
		mint cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		cl_platform_id ocl;
#endif /* CONFIG_USE_OPENCL */
	};
} WGL_Platform_t;

typedef struct {
	WGL_API_t api;
	mint offset;
	union {
#ifdef CONFIG_USE_CUDA
		CUdevice cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		cl_device_id ocl;
#endif /* CONFIG_USE_OPENCL */
	};
} WGL_Device_t;

typedef struct {
	WGL_API_t api;
	WGL_Device_t * device;
	WGL_Platform_t * platform;
	union {
#ifdef CONFIG_USE_CUDA
		CUcontext cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		cl_context ocl;
#endif /* CONFIG_USE_OPENCL */
	};
} WGL_Context_t;

typedef struct {
	WGL_API_t api;
	union {
#if defined(CONFIG_USE_CUDA)
		CUstream cuda;
#endif /* defined(CONFIG_USE_CUDA) && defined(CONFIG_ENABLE_ASYNC) */
#ifdef CONFIG_USE_OPENCL
		cl_command_queue ocl;
#endif /* defined(CONFIG_USE_CUDA) && defined(CONFIG_ENABLE_ASYNC) */
	};
} WGL_Command_Queue_t;

struct st_WGL_Program_p_t {
	WGL_API_t api;
	mint id;
	char * buildOptions;
	char * programSource;
	unsigned char * programBinary;
	size_t programBinarySize;
	char * fileName;
	mbool deleteFileOnExitQ;
	union {
#ifdef CONFIG_USE_CUDA
		CUmodule cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		cl_program ocl;
#endif /* CONFIG_USE_OPENCL */
	};
	char * buildLog;
	WGL_Tree_Entry(st_WGL_Program_p_t) linkage;
};


typedef WGL_Tree_Head(st_WGL_ProgramList_t, st_WGL_Program_p_t) WGL_ProgramList_t;

struct st_WGL_Kernel_p_t {
	WGL_API_t api;
	char * name;
	union {
#ifdef CONFIG_USE_CUDA
		CUfunction cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		cl_kernel ocl;
#endif /* CONFIG_USE_OPENCL */
	};
};

typedef struct st_WGL_Kernel_p_t * WGL_Kernel_t;

struct st_WGL_Device_Memory_p_t {
	WGL_API_t api;
	size_t byteCount;
	union {
#ifdef CONFIG_USE_CUDA
		CUdeviceptr cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		cl_mem ocl;
#endif /* CONFIG_USE_OPENCL */
	} deviceMemory;
};

#ifdef CONFIG_USE_CUDA_IMAGE_PROCESSING
typedef enum {
	WGL_Border_Unknown									= -99,
	WGL_Border_Fixed									= 1,
	WGL_Border_Constant
} WGL_Border_t;
#endif /* CONFIG_USE_CUDA_IMAGE_PROCESSING */

typedef struct st_WGL_Device_Memory_p_t * WGL_Device_Memory_t;

struct st_WGL_MemoryPool_p_t {
	WGL_Device_Memory_t memories[WGL_MEMORY_POOL_SIZE];
	size_t bufferSize;
	mint length;
};

typedef struct st_WGL_MemoryPool_p_t * WGL_MemoryPool_t;


typedef char WGL_Bool_t;


struct st_WGL_Memory_p_t {
	mint id;

	WGL_API_t api;

	mint flattenedLength;
    size_t byteCount;
	size_t elementSize;
	mint elementWidth;

	mbool neededQ;
	mbool uniqueQ;
	mbool synchronizeOnLaunchStart;
	mbool synchronizeOnLaunchReturn;

	WGL_Device_Memory_t deviceMemory;

	void * hostMemory;
	mint rank;
	mint * dimensions;

	WGL_Type_t type;

	WGL_Memory_Status_t hostMemoryStatus;
	WGL_Memory_Status_t deviceMemoryStatus;
	
	WGL_MemoryResidence_t residence;
	
	MTensor tensor;

	WGL_Tensor_Sharing_t sharing;

	WGL_Tree_Entry(st_WGL_Memory_p_t) linkage;
	WGL_Bool_t transposedQ;
};

typedef WGL_Tree_Head(st_WGL_MemoryList_t, st_WGL_Memory_p_t) WGL_MemoryList_t;

struct st_WGL_BuildState_p_t {
	WGL_API_t api;

	char buildLog[WGL_BUILD_LOG_BUFFER_SIZE];
	char * programSource;
	unsigned char * programBinary;
	size_t programBinarySize;
	char * inputFile;
	char * buildOptions;
#ifdef CONFIG_USE_CUDA
	char * compilerPath;
	char * xCompilerPath;
	char * outputFile;
#endif /* CONFIG_USE_CUDA */
};

typedef struct st_WGL_BuildState_p_t * WGL_BuildState_t;

struct st_WGL_Time_p_t {
	umint startTime[2];
	umint currentTime[2];
	umint endTime[2];
	umint elapsedTime[2];
	umint granularity;
	char * category;
	char * msg;
    mbool runningQ;
};
typedef struct st_WGL_Time_p_t * WGL_Time_t;


#ifdef CONFIG_USE_OPENCL


typedef enum {
	WGL_OpenCL_DeviceVendor_Generic = 0,
	WGL_OpenCL_DeviceVendor_NVIDIA = 1,
	WGL_OpenCL_DeviceVendor_AMD = 2,
	WGL_OpenCL_DeviceVendor_INTEL = 3,
} WGL_OpenCL_DeviceVendor_t;

#define WGL_OpenCL_DeviceVendor_ATI			WGL_OpenCL_DeviceVendor_AMD

struct st_WGL_OpenCL_Device_Query_p_t {
    WGL_DeviceTypeProperty_t				type;
	char *									name;
	char *									vendor;
	WGL_OpenCL_DeviceVendor_t				vendorEnum;
	char *									deviceVersion;
	char * 									driverVersion;
	char *		 							extensions;
	char *									profile;
    mint 									vendorID;
    mint 									maxComputeUnits;
    mint 									maxWorkItemDimensions;
    size_t *								maxWorkItemSizes;
    size_t 									maxWorkGroupSize;
    mint 									preferredVectorWidthChar;
    mint 									preferredVectorWidthShort;
    mint 									preferredVectorWidthInteger;
    mint 									preferredVectorWidthLong;
    mint 									preferredVectorWidthFloat;
    mint 									preferredVectorWidthDouble;
    size_t 									maxClockFrequency;
    size_t 									addressBits;
    size_t	 								maxMemAllocSize;
    mint  									imageSupport;
    mint 									maxReadImageArgs;
    mint 									maxWriteImageArgs;
    mint 									image2DMaxWidth;
    mint 									image2DMaxHeight;
    mint 									image3DMaxWidth;
    mint 									image3DMaxHeight;
    mint 									image3DMaxDepth;
    mint 									maxSamplers;
    size_t 									maxParameterSize;
    mint 									memBaseAddrAlign;
    size_t 									minDataTypeAlignSize;
    cl_device_fp_config 					singleFpConfig;
    cl_device_mem_cache_type 				globalMemCacheType;
    size_t									globalMemCacheLineSize;
    size_t	 								globalMemCacheSize;
    size_t 									globalMemSize;
    size_t 									maxConstantBufferSize;
    mint 									maxConstantArgs;
    cl_device_local_mem_type 				localMemType;
    size_t	 								localMemSize;
    mint  									errorCorrectionSupport;
    size_t									profilingTimerResolution;
    mint  									endianLittle;
    mint  									available;
    mint  									compilerAvailable;
    cl_device_exec_capabilities 			executionCapabilities;
    cl_command_queue_properties 			queueProperties;
	mbool									extKHRFP64;
	mbool									extAMDFP64;
};

typedef struct st_WGL_OpenCL_Device_Query_p_t * WGL_OpenCL_Device_Query_t;

struct st_WGL_OpenCL_Platform_Query_p_t {
	char * profile;
	char * version;
	char * name;
	char * vendor;
	char * extensions;
	mint numDevices;
	WGL_OpenCL_Device_Query_t * devices;
};

typedef struct st_WGL_OpenCL_Platform_Query_p_t * WGL_OpenCL_Platform_Query_t;

struct st_WGL_OpenCL_Query_p_t {
	mint numPlatforms;
	WGL_OpenCL_Platform_Query_t * platforms;
};

typedef struct st_WGL_OpenCL_Query_p_t * WGL_OpenCL_Query_t;
#endif /* CONFIG_USE_OPENCL */



#ifdef CONFIG_USE_CUDA

struct st_WGL_CUDA_Device_Query_p_t {
	char * name;
	mint clockRate;
	double computeCapability;
	mint gpuOverlap;
	mint maxThreadsPerBlock;
	mint * maxBlockDims;
	mint * maxGridDims;
	mint maxSharedMemoryPerBlock;
	mint totalConstantMemory;
	mint warpSize;
	mint maxPitch;
	mint maxRegistersPerBlock;
	mint textureAlignment;
	mint multiProcessorCount;
	mint kernelExecTimeout;
	mint integrated;
	mint canMapHostMemory;
	mint computeMode;
	mint maximumTexture1DWidth;
	mint maximumTexture2DWidth;
	mint maximumTexture2DHeight;
	mint maximumTexture3DWidth;
	mint maximumTexture3DHeight;
	mint maximumTexture3DDepth;
	mint maximumTexture2DArrayWidth;
	mint maximumTexture2DArrayHeight;
	mint maximumTexture2DArraySlices;
	mint surfaceAlignment;
	mint concurrentKernels;
	mint eccEnabled;
	mint tccEnabled;
	size_t totalMemory;
};

typedef struct st_WGL_CUDA_Device_Query_p_t * WGL_CUDA_Device_Query_t;

struct st_WGL_CUDA_Query_p_t {
	mint numDevices;
	WGL_CUDA_Device_Query_t * devices;
};

typedef struct st_WGL_CUDA_Query_p_t * WGL_CUDA_Query_t;

#endif /* CONFIG_USE_CUDA */

typedef struct {
	WGL_API_t api;
	union {
#ifdef CONFIG_USE_CUDA
		WGL_CUDA_Query_t cuda;
#endif /* CONFIG_USE_CUDA */
#ifdef CONFIG_USE_OPENCL
		WGL_OpenCL_Query_t ocl;
#endif /* CONFIG_USE_OPENCL */
	};
} WGL_Query_t;

struct st_WGL_ProfileElement_p_t {
    mint id;
    const char * cfun;
    const char * file;
    mint lineno;
    WGL_Time_t time;
    struct st_WGL_ProfileElement_p_t * parent;
    struct st_WGL_ProfileElement_p_t * next, * prev;
};

struct st_WGL_ProfileTable_p_t {
    WGL_ProfileElement_t prof;
    mint len;
    double total;
};

typedef struct st_WGL_ProfileTable_p_t * WGL_ProfileTable_t;

#ifdef CONFIG_USE_CUDA
typedef struct st_WGL_CUDA_RandomState_t {
	curandGenerator_t cuRandGenerator;
	int64_t prevSeed;
	int64_t currSeed;
} WGL_CUDA_RandomState_t;
#endif /* CONFIG_USE_CUDA */

struct st_WGL_State {
	WGL_API_t api;
	WGL_Error_t err;

	void * lock;

	char stringBuffer[WGL_STATE_STRING_BUFFER_SIZE];

	const char * messageHead;

	WolframLibraryData libData;

	WGL_Query_t query;
	
	size_t argumentOffset;
	size_t localMemorySize;
	mint blockDimensionLength;
	mint gridDimensionLength;
	
	WGL_Command_Queue_t * commandQueue;
	WGL_Context_t * context;
	WGL_Platform_t * platform;
	WGL_Device_t * device;
	WGL_Program_t * program;
	WGL_Kernel_t * kernel;
	
	size_t * blockDimensions;
	size_t * gridDimensions;
	
	WGL_BuildState_t buildState;

	size_t currentDeviceMemoryUsage, maximumUsableDeviceMemory;

	WGL_MemoryList_t * registeredMemories;
	WGL_ProgramList_t * registeredPrograms;
	
	WGL_MemoryPool_t memoryBuffer;
	
    mbool profileModeQ;
    WGL_ProfileTable_t profileTable;

	void * userData[WGL_STATE_USER_DATA_SIZE];
	
#ifdef CONFIG_USE_CUDA
	cublasHandle_t cublasState;
	WGL_CUDA_RandomState_t curandState;
#endif /* CONFIG_USE_CUDA */
};

#endif /* __WGL_TYPES_H__ */


