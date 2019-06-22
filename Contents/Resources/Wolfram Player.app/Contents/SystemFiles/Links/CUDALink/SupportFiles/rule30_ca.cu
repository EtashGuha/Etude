#include    <stdio.h>

#ifdef DEBUG
#define BLOCKDIM    4
#define STEPS       10
#else
#define BLOCKDIM    256
#define STEPS       40000
#endif

#define USE_PINNED_MEMORY   0

extern "C"
__global__ void rule30ca_kernel(mint * prevRow, mint * nextRow, mint width) {
    __shared__ int smem[BLOCKDIM+2];
    int tx = threadIdx.x, bx = blockIdx.x;
    int index = tx + bx*BLOCKDIM;

    smem[tx+1] = index < width ? prevRow[index] : 0;
    if (tx == 0)
        smem[0] = index > 0 ? prevRow[index-1] : 0;
    else if (tx == BLOCKDIM-1)
        smem[BLOCKDIM+1] = index < width-1 ? prevRow[index+1] : 0;
    
    __syncthreads();
    
    if (index < width)
        nextRow[index] = smem[tx] ^ (smem[tx+1] | smem[tx+2]);
}

int main( ) {
    mint * ca, * d_prevRow, * d_nextRow, * d_tmp;
    int steps;
    int width;
    int ii;
#ifdef DEBUG
    int jj;
#endif
#if USE_PINNED_MEMORY
    cudaError_t err;
#endif

    steps = STEPS;
    width = 2*steps-1;
    
    /* Init CA */
#if USE_PINNED_MEMORY
    err = cudaMallocHost((void **) &ca, width*steps*sizeof(int));
    if (err != cudaSuccess) {
        printf("Could not allocate memory. Try reducing the number of steps.\n");
        exit(1);
    }
    memset((void *) ca, 0, width*steps*sizeof(int));
#else
    ca = (mint *) calloc(width*steps, sizeof(int));
    if (ca == NULL) {
        printf("Could not allocate memory. Try reducing the number of steps.\n");
        exit(1);
    }
#endif

    ca[width/2] = 1;
    
    /* allocate GPU mem */
    cudaMalloc((void **) &d_prevRow, width*sizeof(int));
    cudaMalloc((void **) &d_nextRow, width*sizeof(int));

    /* copy previous row */
    cudaMemcpy(d_prevRow, ca, width*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCKDIM);
    dim3 gridDim((width + BLOCKDIM - 1)/BLOCKDIM);

    for (ii = 1; ii < steps; ii++) {
        rule30ca_kernel<<<gridDim, blockDim>>>(d_prevRow, d_nextRow, width);                 
        cudaMemcpy(&ca[ii*width + width/2 - ii], &d_nextRow[width/2 - ii],
                   (2*ii+1)*sizeof(int), cudaMemcpyDeviceToHost);

        d_tmp = d_nextRow;
        d_nextRow = d_prevRow;
        d_prevRow = d_tmp;
    }
#ifdef DEBUG
    for (ii = 0; ii < steps; ii++) {
        for (jj = 0; jj < width; jj++) {
            printf("%d", ca[ii*width + jj]);
        }
        printf("\n");
    }
#endif

    cudaFree(d_nextRow);
    cudaFree(d_prevRow);
#if USE_PINNED_MEMORY
    cudaFreeHost(ca);
#else
    free(ca);
#endif
    return 0;
}
