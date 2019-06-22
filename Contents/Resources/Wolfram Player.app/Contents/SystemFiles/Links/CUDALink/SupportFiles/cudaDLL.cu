extern "C" {

#include "WolframLibrary.h"

__global__ void vecAdd(mint * in, mint * out, mint width) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index < width)
        out[index] += in[index];
}

DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}


DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
	return LIBRARY_NO_ERROR;
}

DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
    return ;
}

DLLEXPORT int oTest(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
    MTensor inTensor, outTensor;
    mint * in, * out;
    mint * d_in, * d_out;
    mint width;

    inTensor  = MArgument_getMTensor(Args[0]);
    outTensor = MArgument_getMTensor(Args[1]);
    width = libData->MTensor_getDimensions(inTensor)[0];
    
    in = libData->MTensor_getIntegerData(inTensor);
    out = libData->MTensor_getIntegerData(outTensor);
    
    cudaMalloc((void **) &d_in, width*sizeof(mint));
    cudaMalloc((void **) &d_out, width*sizeof(mint));
    
    cudaMemcpy(d_in, in, width * sizeof(mint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, width * sizeof(mint), cudaMemcpyHostToDevice);
    
    dim3 blockDim(2);
    dim3 gridDim(width/2);
    
    vecAdd<<<blockDim, gridDim>>>(d_in, d_out, width);
    
    cudaMemcpy(out, d_out, width * sizeof(mint), cudaMemcpyDeviceToHost);
    
    MArgument_setMTensor(Res, outTensor);
    return LIBRARY_NO_ERROR;
}
}