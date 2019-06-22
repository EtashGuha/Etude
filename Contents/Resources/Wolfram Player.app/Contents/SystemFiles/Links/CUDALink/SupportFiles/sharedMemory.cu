#define DEBUG
#ifdef DEBUG
#include    <stdio.h>
#endif
#include    <time.h>        // for rand()

#define BLOCKDIM    8

#define WIDTH       10
#define HEIGHT      10
#define RADIUS      BLOCKDIM-2

__global__ void shared1D(int * in, int * out, int radius, int n) {
    extern __shared__ int smem[];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int dx = blockDim.x;
    int xIndex = tx + bx*dx;
    int index = xIndex;

#define SMEM(txOffset)    smem[tx + txOffset]

    if (xIndex < n)
        SMEM(radius) = in[index];
    else
        SMEM(radius) = in[n - 1];

    if (tx < radius) {
        if (xIndex - radius >= 0) {
            SMEM(0) = in[index - radius];
        } else {
            SMEM(0) = in[0];
        }

        if (xIndex + dx < n) {
            SMEM(dx + radius) = in[index + dx];
        } else {
            SMEM(dx + radius) = in[n - 1];
        }
    }

    __syncthreads();

    if (xIndex >= n)
        return ;

    tx += radius;
    out[index] = SMEM(RADIUS-1);
#undef SMEM
}

__global__ void horizShared2D(int * in, int * out, int radius, int width, int height) {
    extern __shared__ int smem[];

    int tx = threadIdx.x,   ty = threadIdx.y;
    int bx = blockIdx.x,    by = blockIdx.y;
    int dx = blockDim.x,    dy = blockDim.y;
    int xIndex = tx + bx*dx;
    int yIndex = ty + by*dy;
    int index = xIndex + yIndex*width;

    if (yIndex >= height)
        return ;

#define SMEM(txOffset, tyOffset)    smem[tx + txOffset + (ty+(tyOffset))*(dx+2*radius)]

    if (xIndex < width)
        SMEM(radius, 0) = in[index];
    else
        SMEM(radius, 0) = in[(yIndex+1)*width - 1];

    if (tx < radius) {
        if (xIndex - radius >= 0) {
            SMEM(0, 0) = in[index - radius];
        } else {
            SMEM(0, 0) = in[yIndex*width];
        }

        if (xIndex + dx < width) {
            SMEM(dx + radius, 0) = in[index + dx];
        } else {
            SMEM(dx + radius, 0) = in[(yIndex + 1)*width - 1];
        }
    }

    __syncthreads();

    if (xIndex >= width)
        return ;

    tx += radius;
    out[index] = SMEM(RADIUS-1, 0);
#undef SMEM
}

__global__ void vertShared2D(int * in, int * out, int radius, int width, int height) {
    extern __shared__ int smem[];

    int tx = threadIdx.x,   ty = threadIdx.y;
    int bx = blockIdx.x,    by = blockIdx.y;
    int dx = blockDim.x,    dy = blockDim.y;
    int xIndex = tx + bx*dx;
    int yIndex = ty + by*dy;
    int index = xIndex + yIndex*width;

    if (xIndex >= width)
        return ;

#define SMEM(txOffset, tyOffset)    smem[tx + txOffset + (ty+(tyOffset))*dx]

    if (yIndex < height)
        SMEM(0, radius) = in[index];
    else
        SMEM(0, radius) = in[(height-1)*width + xIndex];

    if (ty < radius) {
        if (yIndex - radius >= 0) {
            SMEM(0, 0) = in[index - radius*width];
        } else {
            SMEM(0, 0) = in[xIndex];
        }

        if (yIndex + dy < height) {
            SMEM(0, dy + radius) = in[index + dy*width];
        } else { 
            SMEM(0, dy + radius) = in[(height - 1)*width + xIndex];
        }
    }

    __syncthreads();

    if (yIndex >= height)
        return ;

    ty += radius;
    out[index] = SMEM(0, RADIUS-1);
#undef SMEM
}

__global__ void shared2D(int * in, int * out, int radius, int width, int height) {
    extern __shared__ int smem[];

    int tx = threadIdx.x,   ty = threadIdx.y;
    int bx = blockIdx.x,    by = blockIdx.y;
    int dx = blockDim.x,    dy = blockDim.y;
    int xIndex = tx + bx*dx;
    int yIndex = ty + by*dy;
    int index = xIndex + yIndex*width;

#define SMEM(txOffset, tyOffset)    smem[tx + txOffset + (ty+(tyOffset))*(dx+2*radius)]
    
    if (xIndex < width && yIndex < height) {
        SMEM(radius, radius) = in[index];
    } else if (xIndex < width) {
        SMEM(radius, radius) = in[(height - 1)*width + xIndex];
    } else if (yIndex < height) {
        SMEM(radius, radius) = in[(yIndex + 1)*width - 1];
    } else {
        SMEM(radius, radius) = in[height*width - 1];
    }

    if (tx < radius) {
        if (xIndex - radius >= 0) {
            if (yIndex < height) {
                SMEM(0, radius) = in[index - radius];
            } else {
                SMEM(0, radius) = in[(height - 1)*width + xIndex - radius];
            }
        } else {
            if (yIndex < height) {
                SMEM(0, radius) = in[yIndex*width];
            } else {
                SMEM(0, radius) = in[(height - 1)*width];
            }
        }

        if (xIndex + dx < width) {
            if (yIndex < height) {
                SMEM(dx+radius, radius) = in[index + dx];
            } else {
                SMEM(dx+radius, radius) = in[(height - 1)*width + xIndex + dx];
            }
        } else {
            if (yIndex < height) {
                SMEM(dx+radius, radius) = in[(yIndex + 1)*width - 1];
            } else {
                SMEM(dx+radius, radius) = in[height*width - 1];
            }
        }
    }

    if (ty < radius) {
        if (yIndex - radius >= 0) {
            if (xIndex < width) {
                SMEM(radius, 0) = in[index - width*radius];
            } else {
                SMEM(radius, 0) = in[(yIndex - radius)*width];
            }
        } else {
            if (xIndex < width) {
                SMEM(radius, 0) = in[xIndex];
            } else {
                SMEM(radius, 0) = in[0];
            }
        }

        if (yIndex + dy < width) {
            if (xIndex < width) {
                SMEM(radius, dy + radius) = in[index + dy*width];
            } else {
                SMEM(radius, dy + radius) = in[(yIndex + dy + 1)*width - 1];
            }
        } else {
            if (xIndex < width) {
                SMEM(radius, dy + radius) = in[(height - 1)*width + xIndex];
            } else {
                SMEM(radius, dy + radius) = in[height*width - 1];
            }
        }
    }

    
    if (tx < radius && ty < radius) {
        // top left corner
        if (xIndex - radius >= 0) {
            if (yIndex - radius >= 0)
                SMEM(0, 0) = in[index - radius*(width + 1)];
            else
                SMEM(0, 0) = in[xIndex - radius];
        } else {
            if (yIndex - radius >= 0)
                SMEM(0, 0) = in[(yIndex - radius)*width];
            else
                SMEM(0, 0) = in[0];
        }

        // bottom left corner
        if (xIndex - radius >= 0) {
            if (yIndex + dy < height)
                SMEM(0, dy + radius) = in[index - radius + dx*width];
            else 
                SMEM(0, dy + radius) = in[xIndex - radius + (height - 1)*width];
        } else {
            if (yIndex + dy < height)
                SMEM(0, dy + radius) = in[(yIndex + dy)*width];
            else 
                SMEM(0, dy + radius) = in[(height - 1)*width];
        }


        // top right corner
        if (xIndex + dx < width) {
            if (yIndex - radius >= 0)
                SMEM(dx + radius, 0) = in[index + dx - radius*width];
            else
                SMEM(dx + radius, 0) = in[xIndex + dx];
        } else {
            if (yIndex - radius >= 0)
                SMEM(dx + radius, 0) = in[(yIndex - radius + 1)*width - 1];
            else
                SMEM(dx + radius, 0) = in[width - 1];
        }

        // bottom right corner
        if (xIndex + dx < width) {
            if (yIndex + dy < height)
                SMEM(dx + radius, dy + radius) = in[index + dy*(width + 1)];
            else
                SMEM(dx + radius, dy + radius) = in[xIndex + dx + (height - 1)*width];
        } else {
            if (yIndex + dy < height)
                SMEM(dx + radius, dy + radius) = in[(yIndex + dy + 1)*width - 1];
            else
                SMEM(dx + radius, dy + radius) = in[height*width - 1];
        }

        SMEM(0, 0) = 11;
        SMEM(dx + radius, 0) = 33;
        SMEM(0, dy + radius) = 88;
        SMEM(dx + radius, dy + radius) = 99;
    }

    if (xIndex >= width || yIndex >= height)
        return ;

    tx += radius;
    ty += radius;
    out[index] = SMEM(4, 4);
#undef SMEM
}

__host__ void init(int * list, int width, int height) {
    int ii, jj;
    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            list[ii*width + jj] = ii;
        }
    }
}

int main( ) {
    int * in, * d_in;
    int * out, * d_out;
    int width, height, radius;
    int sharedMemSize;
#ifdef DEBUG
    int ii, jj;
#endif

    width = WIDTH;
    height = HEIGHT;
    radius = RADIUS;

    in = (int *) malloc(width*height*sizeof(int));
    out = (int *) malloc(width*height*sizeof(int));

    init(in, width, height);

    cudaMalloc((void **) &d_in, width*height*sizeof(int));
    cudaMalloc((void **) &d_out, width*height*sizeof(int));

    cudaMemcpy(d_in, in, width*height*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCKDIM,BLOCKDIM);
    dim3 gridDim((width + BLOCKDIM - 1)/BLOCKDIM, (height + BLOCKDIM -1)/BLOCKDIM);
    
    sharedMemSize = BLOCKDIM*(BLOCKDIM+2*RADIUS)*sizeof(int)*sizeof(int);

    shared2D<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, radius, width, height);
    cudaMemcpy(out, d_out, width*height*sizeof(int), cudaMemcpyDeviceToHost);

#ifdef NDEBUG
    for (jj = 0; jj < height; jj++) {
        for (ii = 0; ii < width; ii++) {
            if (out[jj*width + ii] != in[jj*width + (ii + RADIUS)%WIDTH]) {
                 printf("foo");
            }
        }
    }
#endif

#ifdef DEBUG
    for (jj = 0; jj < height; jj++) {
        for (ii = 0; ii < width; ii++) {
            printf("%d ", out[jj*width + ii]);
        }
        printf("\n");
    }
#endif

    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
