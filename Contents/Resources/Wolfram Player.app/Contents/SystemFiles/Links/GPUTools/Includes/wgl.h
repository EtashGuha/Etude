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

#ifndef __WGL_H__
#define __WGL_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include	<WolframLibrary.h>

#define WGL_VERSION								WolframLibraryVersion
#define WGL_VERSION_MINOR						0

#ifndef CONFIG_USE_DOUBLE_PRECISION
#define CONFIG_USE_DOUBLE_PRECISION				1
#endif /* CONFIG_USE_DOUBLE_PRECISION */

#ifdef MINT_32
#define MINTBITS								32
#ifdef _WIN32
typedef __w64 unsigned int umint;
#else
typedef unsigned int umint;
#endif
#else
#define MINTBITS								64
#ifdef _WIN64
typedef unsigned long long umint;
#else
typedef unsigned long umint;
#endif
#endif

#if defined(_WIN32)
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else /*  defined(_WIN32) */ 
#include <stdint.h>
#endif /*  defined(_WIN32) */

typedef enum en_WGL_API_t {
	WGL_API_UNKNOWN		= -99,
	WGL_API_CUDA		= 1,
	WGL_API_OpenCL		= 3
} WGL_API_t;

typedef enum en_WGL_Memory_InputOutput_Argument_t {
	WGL_Memory_Argument_Unknown			= -99,
	WGL_Memory_Argument_Input			= 0x01,
	WGL_Memory_Argument_Output			= 0x10,
	WGL_Memory_Argument_InputOutput		= 0x11
} WGL_Memory_InputOutput_Argument_t;

typedef enum en_WGL_MemoryResidence_t {
	WGL_MemoryResidence_DeviceOnly = 1,
	WGL_MemoryResidence_DeviceHost,
	WGL_MemoryResidence_HostOnly
} WGL_MemoryResidence_t;

typedef enum en_WGL_Memory_Status_t {
	WGL_Memory_Uninitialized = 1,
	WGL_Memory_Unsynchronized,
	WGL_Memory_Synchronized
} WGL_Memory_Status_t;

typedef enum en_WGL_Tensor_Sharing_t {
	WGL_Tensor_NotBound = -1,
	WGL_Tensor_Cloned	= 1,
	WGL_Tensor_Shared
} WGL_Tensor_Sharing_t;

typedef enum en_WGL_Type_t {
	WGL_Type_Unknown									= -99,
	WGL_Type_MTensor									= -90,
	WGL_Type_Void										= -4,
	WGL_Type_Scalar										= -3,
	WGL_Type_Automatic									= -1,
	
	WGL_Type_Memory										= 0,
	
#ifdef MINT_32
	WGL_Type_Integer									= MType_Integer,
#else /* MINT_32 */
	WGL_Type_Long										= MType_Integer,
#endif /* MINT_32 */

	WGL_Type_Char										= 10,
	WGL_Type_Char2										= 11,
	WGL_Type_Char3										= 12,
	WGL_Type_Char4										= 13,
	WGL_Type_Char8										= 14,
	WGL_Type_Char16										= 15,
	
	WGL_Type_UnsignedChar								= 20,
	WGL_Type_UnsignedChar2								= 21,
	WGL_Type_UnsignedChar3								= 22,
	WGL_Type_UnsignedChar4								= 23,
	WGL_Type_UnsignedChar8								= 24,
	WGL_Type_UnsignedChar16								= 25,
	
	WGL_Type_Short										= 30,
	WGL_Type_Short2										= 31,
	WGL_Type_Short3										= 32,
	WGL_Type_Short4										= 33,
	WGL_Type_Short8										= 34,
	WGL_Type_Short16									= 35,

	WGL_Type_UnsignedShort								= 40,
	WGL_Type_UnsignedShort2								= 41,
	WGL_Type_UnsignedShort3								= 42,
	WGL_Type_UnsignedShort4								= 43,
	WGL_Type_UnsignedShort8								= 44,
	WGL_Type_UnsignedShort16							= 45,

#ifndef MINT_32
	WGL_Type_Integer									= 50,
#endif /* MINT_32 */
	WGL_Type_Integer2									= 51,
	WGL_Type_Integer3									= 52,
	WGL_Type_Integer4									= 53,
	WGL_Type_Integer8									= 54,
	WGL_Type_Integer16									= 55,

	WGL_Type_UnsignedInteger							= 60,
	WGL_Type_UnsignedInteger2							= 61,
	WGL_Type_UnsignedInteger3							= 62,
	WGL_Type_UnsignedInteger4							= 63,
	WGL_Type_UnsignedInteger8							= 64,
	WGL_Type_UnsignedInteger16							= 65,

#ifdef MINT_32
	WGL_Type_Long										= 70,
#endif /* MINT_32 */
	WGL_Type_Long2										= 71,
	WGL_Type_Long3										= 72,
	WGL_Type_Long4										= 73,
	WGL_Type_Long8										= 74,
	WGL_Type_Long16										= 75,
	
	WGL_Type_Float										= 90,
	WGL_Type_Float2										= 91,
	WGL_Type_Float3										= 92,
	WGL_Type_Float4										= 93,
	WGL_Type_Float8										= 94,
	WGL_Type_Float16									= 95,
	
	WGL_Type_Double = MType_Real,
	WGL_Type_Double2									= 101,
	WGL_Type_Double3									= 102,
	WGL_Type_Double4									= 103,
	WGL_Type_Double8									= 104,
	WGL_Type_Double16									= 105,
	
	WGL_Type_ComplexFloat								= 110,
	WGL_Type_ComplexDouble = MType_Complex,

	WGL_Type_MatrixFloat								= 130,
	WGL_Type_MatrixTransposedFloat						= 131,
	WGL_Type_MatrixDouble								= 140,
	WGL_Type_MatrixTransposedDouble						= 141,
	WGL_Type_MatrixComplexFloat							= 150,
	WGL_Type_MatrixTransposedComplexFloat				= 151,
	WGL_Type_MatrixComplexDouble						= 160,
	WGL_Type_MatrixTransposedComplexDouble				= 161,
	WGL_Type_Real = WGL_Type_Double,
	WGL_Type_Complex = WGL_Type_ComplexDouble,
	WGL_Type_Boolean									= 300,
	WGL_Type_Expr										= 400,
} WGL_Type_t;

#ifdef MINT_32
#define WGL_Type_MInteger			WGL_Type_Integer
#else /* MINT_32 */
#define WGL_Type_MInteger			WGL_Type_Long
#endif /* MINT_32 */

#ifdef CONFIG_USE_DOUBLE_PRECISION
typedef double						WGL_Real_t;
#define WGL_Type_MReal				WGL_Type_Double
#define WGL_Type_MComplexReal		WGL_Type_ComplexDouble
#else /* CONFIG_USE_DOUBLE_PRECISION */
typedef float						WGL_Real_t;
#define WGL_Type_MReal				WGL_Type_Float
#define WGL_Type_MComplexReal		WGL_Type_ComplexFloat
#endif /* CONFIG_USE_DOUBLE_PRECISION */

typedef struct st_WGL_Error_p_t * WGL_Error_t;
typedef struct st_WGL_Program_p_t * WGL_Program_t;
typedef struct st_WGL_Memory_p_t * WGL_Memory_t;
typedef struct st_WGL_ProfileElement_p_t * WGL_ProfileElement_t;
typedef struct st_WGL_State* WGL_State_t;

typedef struct st_WolframGPULibraryData_t * WolframGPULibraryData;
typedef struct st_WolframGPUCompileLibrary_Functions_t * WolframGPUCompileLibrary_Functions;


struct st_WolframGPULibraryData_t {
	mint VersionNumber;
	WGL_API_t API;
	WGL_State_t state;
	WolframGPUCompileLibrary_Functions compileLibraryFunctions;
	
	WGL_API_t (*getAPI)(WolframGPULibraryData);														/* get api */
	
	void (*setWolframLibraryData)(WolframGPULibraryData, WolframLibraryData);						/* sets device */

	mbool (*successQ)(WolframGPULibraryData);														/* True if the operation returned with no errors*/
	WGL_Error_t (*getError)(WolframGPULibraryData);													/* gets error */
	void (*clearError)(WolframGPULibraryData);														/* clear error */

	void (*setPlatform)(WolframGPULibraryData, mint);												/* sets platform */
	void (*setDevice)(WolframGPULibraryData, mint);													/* sets device */

	void (*setProgram)(WolframGPULibraryData, WGL_Program_t);										/* sets program from existing WGL_Program_t object */
	WGL_Program_t (*newProgramFromSource)(WolframGPULibraryData, const char *, const char *);		/* sets program from source code with specified build options */
	WGL_Program_t (*newProgramFromSourceFile)(WolframGPULibraryData, const char *, const char *);	/* sets program from source file with specified build options */
	
	WGL_Program_t (*newProgramFromBinary)(WolframGPULibraryData, const unsigned char *, size_t);	/* sets program from binary dump, along with size */
	WGL_Program_t (*newProgramFromBinaryFile)(WolframGPULibraryData, const char *);					/* sets program from binary file */

	char const * (*getProgramBuildLog)(WolframGPULibraryData, WGL_Program_t);						/* gets build log of program */

	void (*setKernel)(WolframGPULibraryData, const char *);											/* sets kernel name */

	void (*setKernelCharArgument)(WolframGPULibraryData, char);										/* sets kernel char argument */
	void (*setKernelUnsignedCharArgument)(WolframGPULibraryData, unsigned char);					/* sets kernel unsigned char argument */
	void (*setKernelShortArgument)(WolframGPULibraryData, short);									/* sets kernel short argument */
	void (*setKernelUnsignedShortArgument)(WolframGPULibraryData, unsigned short);					/* sets kernel unsigned short argument */
	void (*setKernelIntegerArgument)(WolframGPULibraryData, int);									/* sets kernel integer argument */
	void (*setKernelUnsignedIntegerArgument)(WolframGPULibraryData, unsigned int);					/* sets kernel unsigned integer argument */
	void (*setKernelLongArgument)(WolframGPULibraryData, int64_t);									/* sets kernel 64bit long argument */
	void (*setKernelFloatArgument)(WolframGPULibraryData, float);									/* sets kernel float argument */
	void (*setKernelDoubleArgument)(WolframGPULibraryData, double);									/* sets kernel double argument */
	void (*setKernelMemoryArgument)(WolframGPULibraryData, WGL_Memory_t, WGL_Memory_InputOutput_Argument_t); /* sets memory argument and whether it is input/output */

	void (*setKernelLocalMemoryArgument)(WolframGPULibraryData, size_t);							/* sets the size of the local or shared memory of the kernel */

	void (*setBlockDimensions)(WolframGPULibraryData, mint, mint, mint, mint);						/* sets block dimensions */
	void (*setGridDimensions)(WolframGPULibraryData, mint, mint, mint, mint);						/* sets grid dimensions */

	void (*launchKernel)(WolframGPULibraryData);													/* launch kernel */
	void (*synchronize)(WolframGPULibraryData);														/* synchronize */

	/* new memory, its type, rank, and dimensions */
	WGL_Memory_t (*newMemory)(WolframGPULibraryData, WGL_Type_t, mint, const mint *);

	/* new tensor memory, its type, and whether it's unique */
	WGL_Memory_t (*newMTensorMemory)(WolframGPULibraryData, MTensor, WGL_Type_t, WGL_MemoryResidence_t, WGL_Tensor_Sharing_t, mbool);

	/* new raw memory, its type, its size, and whether it's unique */
	WGL_Memory_t (*newRawMemory)(WolframGPULibraryData, void **, WGL_MemoryResidence_t, size_t, mbool);
	
	int * (*MTensorMemory_getIntegerData)(WolframGPULibraryData, WGL_Memory_t);						/* gets integer data from tensor */
	unsigned int * (*MTensorMemory_getUnsignedIntegerData)(WolframGPULibraryData, WGL_Memory_t);	/* gets unsigned integer data from tensor */
	int64_t * (*MTensorMemory_getLongData)(WolframGPULibraryData, WGL_Memory_t);					/* gets integer64 data from tensor */
	double * (*MTensorMemory_getRealData)(WolframGPULibraryData, WGL_Memory_t);						/* gets real data from tensor */
	mcomplex * (*MTensorMemory_getComplexData)(WolframGPULibraryData, WGL_Memory_t);				/* gets complex data from tensor */
	void * (*MTensorMemory_getHostData)(WolframGPULibraryData, WGL_Memory_t, size_t *);				/* gets host data from tensor */

	void * (*RawMemory_getHostData)(WolframGPULibraryData, WGL_Memory_t, size_t *);					/* gets host data from raw memory */

	void (*copyMemoryToDevice)(WolframGPULibraryData, WGL_Memory_t, mbool);							/* copies memory to device, and whether to force the copy */
	void (*copyMemoryToHost)(WolframGPULibraryData, WGL_Memory_t, mbool);							/* copies memory to host, and whether to force the copy */
#if 0
	mbool (*memoryAvailableQ)(WolframGPULibraryData, WGL_Memory_t, ...);							/* free memory */
#endif
	WGL_Memory_t * (*getMemory)(WolframGPULibraryData, mint);										/* get pointer to memory */
	WGL_Memory_t (*findMemory)(WolframGPULibraryData, mint);										/* find memory */
	void (*freeMemory)(WolframGPULibraryData, WGL_Memory_t);										/* free memory */
	
	mbool (*profileQ)(WolframGPULibraryData);
	void (*enableProfiler)(WolframGPULibraryData);
	void (*disableProfiler)(WolframGPULibraryData);
	WGL_ProfileElement_t (*startFunctionProfile)(WolframGPULibraryData, const char *, char *);
	WGL_ProfileElement_t (*startVerboseFunctionProfile)(WolframGPULibraryData, const char *, const char *, mint, const char *, char *);
	void (*stopFunctionProfile)(WolframGPULibraryData, WGL_ProfileElement_t);
	void (*clearProfilerTable)(WolframGPULibraryData);

	void * (*alloc)(size_t sz);
	void (*free)(void * mem);

};

DLLEXPORT WolframGPULibraryData WolframGPULibraryData_New(WGL_API_t api, mint version);
DLLEXPORT void WolframGPULibraryData_Free(WolframGPULibraryData * data);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __WGL_H__ */
