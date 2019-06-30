__global__ void imageColorNegate(mint * img, mint width, mint height, mint channels) {
    mint ii;
	int xIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y*blockDim.y;
	int index = channels*(xIndex + yIndex*width);
	if (xIndex < width && yIndex < height) {
		for (ii = 0; ii < channels; ii++)
			img[index+ii] = 255 - img[index+ii];
    }
}
