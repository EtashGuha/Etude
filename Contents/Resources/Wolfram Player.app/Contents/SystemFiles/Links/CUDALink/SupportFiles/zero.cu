__global__ void zero_kernel(int * in, int length) {
  	int idx = threadIdx.x + blockDim.x*blockIdx.x;
  	if (idx < length)
    		in[idx] = 0;
}
