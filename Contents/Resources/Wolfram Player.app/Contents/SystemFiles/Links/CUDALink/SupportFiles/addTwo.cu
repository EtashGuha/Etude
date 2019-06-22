__global__ void addTwo(mint * A, mint length) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index < length)
        A[index] += 2;
}
