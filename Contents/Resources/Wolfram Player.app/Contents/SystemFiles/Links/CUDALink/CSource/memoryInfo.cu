
#include	<wgl.h>
#include	<wgl_cuda_runtime.h>



EXTERN_C DLLEXPORT int oCUDAMemoryInfoTotal(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
  size_t freeMem, total;
  cudaMemGetInfo(&freeMem, &total);
  MArgument_setInteger(Res, total);
  return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int oCUDAMemoryInfoFree(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
  size_t freeMem, total;
  cudaMemGetInfo(&freeMem, &total);
  MArgument_setInteger(Res, freeMem);
  return LIBRARY_NO_ERROR;
}
