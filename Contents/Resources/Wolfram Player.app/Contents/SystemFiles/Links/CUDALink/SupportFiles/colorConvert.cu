__global__ void rgb2hsb(Real_t * in, Real_t * out, mint width, mint height, mint channels, mint pitch) {
#ifndef MIN3
#define MIN3(a,b,c) (min(min(a,b),c))
#endif

#ifndef MAX3
#define MAX3(a,b,c) (max(max(a,b),c))
#endif
	
    const int dx = blockDim.x,  dy = blockDim.y;
    const int bx = blockIdx.x,  by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    
    const int xIndex = bx*dx + tx;
    const int yIndex = by*dy + ty;
    
    if ((xIndex >= width) || (yIndex >= height)) return ;

    const int index = xIndex*channels + pitch*yIndex;
    Real_t r, g, b, h, s, v, maxValue, minValue, tmp;
    
    r = in[index];
    g = in[index+1];
    b = in[index+2];
    
    minValue = MIN3(r,g,b);
    maxValue = MAX3(r,g,b);
    tmp = maxValue-minValue;

    if (maxValue == minValue) {
        h = 0.0;
    } else if (maxValue == r) {
        h = 60.0*(g-b)/tmp + 360.0;
        if (h >= 360.0)
        	h -= 360.0;
    } else if (maxValue == g) {
        h = 60.0*(b-r)/tmp + 120.0;
    } else {        // maxValue == b
        h = 60.0*(r-g)/tmp + 240.0;
    }
    
    if (maxValue == 0)
        s = 0.0;
    else
        s = tmp/maxValue;

    v = maxValue;

    out[index]   = h/360.0;
    out[index+1] = s;
    out[index+2] = v;
}

__global__ void rgb2hsv(Real_t * in, Real_t * out, mint width, mint height, mint channels, mint pitch) {
#ifndef MIN3
#define MIN3(a,b,c) (min(min(a,b),c))
#endif

#ifndef MAX3
#define MAX3(a,b,c) (max(max(a,b),c))
#endif
	
    const int dx = blockDim.x,  dy = blockDim.y;
    const int bx = blockIdx.x,  by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    
    const int xIndex = bx*dx + tx;
    const int yIndex = by*dy + ty;
    
    if ((xIndex >= width) || (yIndex >= height)) return ;

    const int index = xIndex*channels + pitch*yIndex;
    Real_t r, g, b, h, s, l, maxValue, minValue, tmp;
    
    r = in[index];
    g = in[index+1];
    b = in[index+2];
    
    minValue = MIN3(r,g,b);
    maxValue = MAX3(r,g,b);
    tmp = maxValue-minValue;

    if (maxValue == minValue) {
        h = 0.0;
    } else if (maxValue == r) {
        h = 60.0*(g-b)/tmp + 360.0;
        if (h >= 360.0)
        	h -= 360.0;
    } else if (maxValue == g) {
        h = 60.0*(b-r)/tmp + 120.0;
    } else {        // maxValue == b
        h = 60.0*(r-g)/tmp + 240.0;
    }
    
    l = (maxValue+minValue)/2;

    if (maxValue == minValue) {
        s = 0;
    } else if (maxValue+minValue > 1) {
        s = tmp/(maxValue+minValue);
    } else {
        s = tmp/(1-(maxValue+minValue));
    }

    out[index]   = h/360.0;
    out[index+1] = s;
    out[index+2] = l;
}
