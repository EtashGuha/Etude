

#ifndef __WGL_COMPILE_H__
#define __WGL_COMPILE_H__

#include	<wgl.h>
#include	<WolframCompileLibrary.h>

typedef struct st_MemoryInitializationData {
	WGL_Memory_t * memories;
	mint n;
	int in_use;
} * MemoryInitializationData;

typedef union {
	mbool *boolean;
	mint *integer;
	mreal *real;
	mcomplex *cmplex;
	MTensor *tensor;
	char **utf8string;
	WGL_Memory_t * mem;
} WGLArgument;

#define WGLArgument_getBooleanAddress(marg)					((marg).boolean)
#define WGLArgument_getIntegerAddress(marg)					((marg).integer)
#define WGLArgument_getRealAddress(marg)					((marg).real)
#define WGLArgument_getComplexAddress(marg)					((marg).cmplex)
#define WGLArgument_getMTensorAddress(marg)					((marg).tensor)
#define WGLArgument_getUTF8StringAddress(marg)				((marg).utf8string)
#define WGLArgument_getMemoryAddress(marg)					((marg).mem)

#define WGLArgument_getAddress(marg)						((void *) ((marg).integer))
#define WGLArgument_setAddress(marg, add)					(((marg).integer) = ((mint *) (add)))

#define WGLArgument_getBoolean(marg)						(*WGLArgument_getBooleanAddress(marg))
#define WGLArgument_getInteger(marg)						(*WGLArgument_getIntegerAddress(marg))
#define WGLArgument_getReal(marg)							(*WGLArgument_getRealAddress(marg))
#define WGLArgument_getComplex(marg)						(*WGLArgument_getComplexAddress(marg))
#define WGLArgument_getMTensor(marg)						(*WGLArgument_getMTensorAddress(marg))
#define WGLArgument_getUTF8String(marg)						(*WGLArgument_getUTF8StringAddress(marg))
#define WGLArgument_getMemory(marg)							(*WGLArgument_getMemoryAddress(marg))

#define WGLArgument_setBoolean(marg, v)						((*WGLArgument_getBooleanAddress(marg)) = (v))
#define WGLArgument_setInteger(marg, v)						((*WGLArgument_getIntegerAddress(marg)) = (v))
#define WGLArgument_setReal(marg, v)						((*WGLArgument_getRealAddress(marg)) = (v))
#define WGLArgument_setComplex(marg, v)						((*WGLArgument_getComplexAddress(marg)) = (v))
#define WGLArgument_setMTensor(marg, v)						((*WGLArgument_getMTensorAddress(marg)) = (v))
#define WGLArgument_setUTF8String(marg, v)					((*WGLArgument_getUTF8StringAddress(marg)) = (v))
#define WGLArgument_setMemory(marg, v)						((*WGLArgument_getMemoryAddress(marg)) = (v))

#define MemoryInitializationData_getMemory(data, i)			(&((data)->memories[(i)]))

typedef int (*WGL_LibraryFunctionPointer)(WolframLibraryData libData, mint, WGLArgument *, WGLArgument);
typedef mint (*WGL_UnaryMathFunctionPointer)(void *, const void *, const mint, const mint *, const unsigned int);
typedef mint (*WGL_BinaryMathFunctionPointer)(void *, const void *, const void *, const mint, const mint *, const unsigned int);

typedef struct st_WGL_Compile_InternalState_p_t * WGL_Compile_InternalState_t;

struct st_WolframGPUCompileLibrary_Functions_t {
	mint VersionNumber;
	WolframGPULibraryData gpuData;
	WGL_Compile_InternalState_t cmpState;
	WolframCompileLibrary_Functions compileFunctions;
	void (*SetWolframLibraryData)(WolframLibraryData);
	MemoryInitializationData (*GetInitializedMemories)(WolframLibraryData, mint);
	void (*ReleaseInitializedMemories)(MemoryInitializationData);
	void (*Memory_copy)(WGL_Memory_t *, WGL_Memory_t);
	mint (*Memory_getFlattenedLength)(WGL_Memory_t);
	mint (*Memory_getRank)(WGL_Memory_t);
	mint * (*Memory_getDimensions)(WGL_Memory_t);
	double * (*Memory_getRealData)(WGL_Memory_t);
	mint * (*Memory_getIntegerData)(WGL_Memory_t);
	int (*Memory_setMemory)(WGL_Memory_t, WGL_Memory_t, mint *, mint);
	mcomplex * (*Memory_getComplexData)(WGL_Memory_t);
	int (*Memory_allocate)(WGL_Memory_t *, int type, mint rank, mint * dims);
	int (*Memory_free)(WGL_Memory_t);
	UnaryMathFunctionPointer (*getUnaryMathFunction)(int, int);
	int (*Math_V_V)(int, const unsigned int, int, void *, int, void *);
	int (*Math_T_T)(int, const unsigned int, WGL_Memory_t, int, WGL_Memory_t *);
	BinaryMathFunctionPointer (*getBinaryMathFunction)(int, int, int);
	int (*Math_TT_T)(int, const unsigned int, WGL_Memory_t, WGL_Memory_t, int, WGL_Memory_t *);
	int (*Math_VV_V)(int, const unsigned int, int, void *, int, void *, int, void *);
	void *(*getExpressionFunctionPointer)(const char *);
	int (*evaluateFunctionExpression)(void *, mint, mint, mint, int *, void **, int, mint, void *);
	LibraryFunctionPointer (*getLibraryFunctionPointer)(char *, char *);
	WGL_LibraryFunctionPointer (*getFunctionCallPointer)(const char *);
};

extern WolframGPUCompileLibrary_Functions WolframGPUCompileLibrary_Functions_new(WolframGPULibraryData data);
extern void WolframGPUCompileLibrary_Functions_free(WolframGPUCompileLibrary_Functions data);

#endif /* __WGL_COMPILE_H__ */



