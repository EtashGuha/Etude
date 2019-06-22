# ifndef OCTAVES
# define OCTAVES     8
# endif

# define clamp(a, low, high)       (((a) >= (low) && (low) <= (high)) ? (a) : ((a) < (low) ? (low) : (high)))

__device__ int grad[12][3] = {
  {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
  {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
  {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}
};

__device__ inline Real_t dot(int gradIdx, Real_t x, Real_t y, Real_t z) {
    return grad[gradIdx][0] * x + grad[gradIdx][1] * y + grad[gradIdx][2] * z;
}

__device__ inline Real_t fade(Real_t t) {
   return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

__device__ inline Real_t lerp(Real_t x, Real_t y, Real_t t) {
   return (1.0 - t) * x + t * y;
}

__device__ inline Real_t signedNoise(int * permutations, Real_t x, Real_t y, Real_t z) {
    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    int iz = static_cast<int>(z);

    x -= ix;
    y -= iy;
    z -= iz;
    
    ix &= 255;
    iy &= 255;
    iz &= 255;
   
    int g000 = permutations[ix + permutations[iy + permutations[iz]]] % 12;
    int g001 = permutations[ix + permutations[iy + permutations[iz + 1]]] % 12;
    int g010 = permutations[ix + permutations[iy + 1 + permutations[iz]]] % 12;
    int g011 = permutations[ix + permutations[iy + 1 + permutations[iz + 1]]] % 12;
    int g100 = permutations[ix + 1 + permutations[iy + permutations[iz]]] % 12;
    int g101 = permutations[ix + 1 + permutations[iy + permutations[iz + 1]]] % 12;
    int g110 = permutations[ix + 1 + permutations[iy + 1 + permutations[iz]]] % 12;
    int g111 = permutations[ix + 1 + permutations[iy + 1 + permutations[iz + 1]]] % 12;

    Real_t n000 = dot(g000, x, y, z);
    Real_t n100 = dot(g100, x-1, y, z);
    Real_t n010 = dot(g010, x, y-1, z);
    Real_t n110 = dot(g110, x-1, y-1, z);
    Real_t n001 = dot(g001, x, y, z-1);
    Real_t n101 = dot(g101, x-1, y, z-1);
    Real_t n011 = dot(g011, x, y-1, z-1);
    Real_t n111 = dot(g111, x-1, y-1, z-1);
    
    Real_t u = fade(x);
    Real_t v = fade(y);
    Real_t w = fade(z);
    
    Real_t nx00 = lerp(n000, n100, u);
    Real_t nx01 = lerp(n001, n101, u);
    Real_t nx10 = lerp(n010, n110, u);
    Real_t nx11 = lerp(n011, n111, u);

    Real_t nxy0 = lerp(nx00, nx10, v);
    Real_t nxy1 = lerp(nx01, nx11, v);
   
    Real_t nxyz = lerp(nxy0, nxy1, w);

    return nxyz;
}

extern "C"
__global__ void monoFractal(Real_t * vals, int * permutations, Real_t amplitude, Real_t frequency, Real_t gain, Real_t lacunarity, Real_t scale, Real_t increment, int width, int height, int depth) {
    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    const int zIndex = threadIdx.z + blockIdx.z * blockDim.z;
    
	if (xIndex >= width || yIndex >= height)
		return ;
	
    const int index = xIndex + width * yIndex ;

    Real_t noiseVal = 0.0f;
    Real_t freq = frequency;
    Real_t x = xIndex * frequency / scale;
    Real_t y = yIndex * frequency / scale;
    Real_t z = zIndex * frequency / scale;
    Real_t tmp = 0.0;

    for (int ii = 0; ii < OCTAVES; ii++) {
       tmp = signedNoise(permutations, x * freq, y * freq, z * freq);
       tmp *= pow(lacunarity, -((Real_t) ii) * increment);
       noiseVal += tmp;
       freq *= lacunarity;
    }
    vals[index] = clamp(noiseVal, 0.0, 1.0);
}


extern "C"
__global__ void multiFractal(Real_t * vals, int * permutations, Real_t amplitude, Real_t frequency, Real_t gain, Real_t lacunarity, Real_t scale, Real_t increment, int width, int height, int depth) {
    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    const int zIndex = threadIdx.z + blockIdx.z * blockDim.z;
    
	if (xIndex >= width || yIndex >= height)
		return ;
	
    const int index = xIndex + width * yIndex ;

    Real_t noiseVal = 1.0f;
    Real_t freq = frequency;
    Real_t x = xIndex * frequency / scale;
    Real_t y = yIndex * frequency / scale;
    Real_t z = zIndex * frequency / scale;
    Real_t tmp = 0.0;

    increment *= 0.01;


    for (int ii = 0; ii < OCTAVES; ii++) {
       tmp = signedNoise(permutations, x * freq, y * freq, z * freq) + 1.0;
       tmp *= pow(lacunarity, -((Real_t) ii) * increment);
       noiseVal *= tmp;
       freq *= lacunarity;
    }
    vals[index] = clamp(noiseVal, 0.0, 1.0);
}


extern "C"
__global__ void turbulence(Real_t * vals, int * permutations, Real_t amplitude, Real_t frequency, Real_t gain, Real_t lacunarity, Real_t scale, Real_t increment, int width, int height, int depth) {
    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    const int zIndex = threadIdx.z + blockIdx.z * blockDim.z;
    
	if (xIndex >= width || yIndex >= height)
		return ;
	
    const int index = xIndex + width * yIndex ;

    Real_t noiseVal = 0.0f;
    Real_t freq = frequency;
    Real_t x = xIndex * frequency / scale;
    Real_t y = yIndex * frequency / scale;
    Real_t z = zIndex * frequency / scale;
    Real_t tmp = 0.0;

    for (int ii = 0; ii < OCTAVES; ii++) {
       tmp = signedNoise(permutations, x * freq, y * freq, z * freq);
       tmp *= pow(lacunarity, -((Real_t) ii) * increment);
       noiseVal += abs(tmp);
       freq *= lacunarity;
    }
    vals[index] = clamp(noiseVal, 0.0, 1.0);
}

extern "C"
__global__ void ridgeMultifractal(Real_t * vals, int * permutations, Real_t amplitude, Real_t frequency, Real_t gain, Real_t lacunarity, Real_t scale, Real_t increment, int width, int height, int depth) {
    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    const int zIndex = threadIdx.z + blockIdx.z * blockDim.z;
    
	if (xIndex >= width || yIndex >= height)
		return ;
	
    const int index = xIndex + width * yIndex ;

    Real_t noiseVal = 0.0f;
    Real_t freq = frequency;
    Real_t x = xIndex * frequency / scale;
    Real_t y = yIndex * frequency / scale;
    Real_t z = zIndex * frequency / scale;
    Real_t offset = 1.0, threshold = 0.5, a = 1.0, tmp = 0.0;

    for (int ii = 0; ii <= OCTAVES; ii++) {
       tmp = abs(signedNoise(permutations, x * freq, y * freq, z * freq));
       tmp = offset - tmp;
       tmp *= tmp * a;
       noiseVal += tmp * pow(lacunarity, -((Real_t) ii) * increment);
       a = clamp(tmp * threshold, 0.0, 1.0);
       freq *= lacunarity;
    }
    vals[index] = clamp(noiseVal, 0.0, 1.0);
}

extern "C"
__global__ void classicPerlin(Real_t * vals, int * permutations, Real_t amplitude, Real_t frequency, Real_t gain, Real_t lacunarity, Real_t scale, Real_t increment, int width, int height, int depth) {
    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    const int zIndex = threadIdx.z + blockIdx.z * blockDim.z;
    
	if (xIndex >= width || yIndex >= height)
		return ;
	
    const int index = xIndex + width * yIndex ;

    Real_t noiseVal = 0.0f;
    Real_t freq = frequency;
    Real_t amp = amplitude;
    Real_t x = xIndex * frequency / scale;
    Real_t y = yIndex * frequency / scale;
    Real_t z = zIndex * frequency / scale;

    for (int ii = 0; ii < OCTAVES; ii++) {
       noiseVal += signedNoise(permutations, x * freq, y * freq, z * freq) * amp;
       freq *= lacunarity;
       amp *= gain;
    }
    vals[index] = clamp(noiseVal, 0.0, 1.0);
}