extern "C" __global__ void vecAdd(mint * A, mint * B, mint * out, mint width) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < width)
		out[index] = A[index] + B[index];
}
