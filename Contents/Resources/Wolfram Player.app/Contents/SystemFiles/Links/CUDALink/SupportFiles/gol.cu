
#define BLOCKDIM    16

__global__ void gol_kernel(mint * in, mint * out, mint width, mint height) {
    __shared__ mint smem[BLOCKDIM+2][BLOCKDIM+3];    

    int tx = threadIdx.x,   ty = threadIdx.y;
    int bx = blockIdx.x,    by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;
    int index = xIndex + yIndex*width;
    int ii, jj, neighbrs;

    smem[tx+1][ty+1] = (xIndex < width && yIndex < height) ? in[index] : 0;
    if (tx == 0) {
        smem[0][ty+1] = (xIndex > 0 && yIndex < height) ? in[index - 1] : 0;
        smem[BLOCKDIM+1][ty+1] = (xIndex+BLOCKDIM < width && yIndex < height) ? in[index + BLOCKDIM] : 0;
    }
    if (ty == 0) {
        smem[tx+1][0] = (xIndex < width && yIndex > ty) ? in[index - width] : 0;
        smem[tx+1][BLOCKDIM+1] = (yIndex+BLOCKDIM < height && xIndex < width) ? in[index + BLOCKDIM*width] : 0;
    }

    if (tx == 0 && ty == 0) {
        if (xIndex > 0 && yIndex > 0)
            smem[0][0] = (xIndex < width && yIndex < width) ? in[index - BLOCKDIM - 1] : 0;
        else
            smem[0][0] = 0;

        if (xIndex > 0 && yIndex+BLOCKDIM < height)
            smem[0][BLOCKDIM+1] = (xIndex < width) ? in[index - 1 + BLOCKDIM*width] : 0;
        else
            smem[0][BLOCKDIM+1] = 0;

        if (xIndex+BLOCKDIM < width && yIndex > 0)
            smem[BLOCKDIM+1][0] = (yIndex < height) ? in[index + BLOCKDIM - width] : 0;
        else
            smem[BLOCKDIM+1][0] = 0;

        if (xIndex+BLOCKDIM < width && yIndex+BLOCKDIM < height)
            smem[BLOCKDIM+1][BLOCKDIM+1] = in[index + 1 + BLOCKDIM*width];
        else
            smem[BLOCKDIM+1][BLOCKDIM+1] = 0;
    }

    __syncthreads();

    tx++;
    ty++;
    if (xIndex < width && yIndex < height) {
        for (ii = -1, neighbrs = 0; ii <= 1; ii++) {
            for (jj = -1; jj <= 1; jj++) {
                if (ii != 0 || jj != 0)
                    neighbrs += smem[tx+ii][ty+jj];
            }
        }
        if (smem[tx][ty]) {
            if (neighbrs == 2 || neighbrs == 3) {
                out[index] = 1;
            } else {
                out[index] = 0;
            }
        } else {
            if (neighbrs == 3) {
                out[index] = 1;
            } else {
                out[index] = 0;
            }
        }
    }
}
