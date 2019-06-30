

#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS 300
#endif

#ifndef BAILOUT
#define BAILOUT		   8.0
#endif

typedef struct __align__(8) complex {
   float real, img;
   __device__ complex & operator=(const float * a) {
       real = a[0];
       img = a[1];
       return *this;
   }
   __device__ complex & operator=(const complex & a) {
       real = a.real;
       img = a.img;
       return *this;
   }
} complex;
__device__ complex operator+(const complex &a, const complex &b) {
   complex res;
   res.real = a.real + b.real;
   res.img = a.img + b.img;
   return res;
}
__device__ complex operator*(const complex &a, const float &b) {
   complex res;
   res.real = a.real * b;
   res.img = a.img * b;
   return res;
}
__device__ complex operator*(const complex &a, const complex &b) {
   complex res;
   res.real = a.real * b.real - a.img * b.img;
   res.img = a.img * b.real + a.real * b.img;
   return res;
}

__device__ float abs2(const complex & z) {
      return z.real*z.real + z.img*z.img;
}
__device__ float abs(const complex & z) {
      return sqrt(abs2(z));
}
__device__ complex operator/(const complex &a, const complex &b) {
   float det = abs2(b);
   complex res = {(a.real * b.real + a.img * b.img)/det, (a.img * b.real - a.real * b.img)/det};
   return res;
}
__device__ complex log(const complex & a) {
   complex res;
   res.img = atan2((float) a.img, (float) a.real);
   res.real = log(abs(a));
   return res;
}
__device__ complex exp(const complex & a) {
   complex res;
   float ex = exp(a.real);
   res.real = ex * cos(a.img);
   res.img = ex * sin(a.img);
   return res;
}
__device__ complex pow(const complex & a, const float b) {
   float logr = log(abs(a));
   float logi = atan2(a.img, a.real);
   complex res;
   res.real = logr * b;
   res.img = logi * b;
   float modans = exp(res.real);
   res.real = modans * cos(res.img);
   res.img = modans * sin(res.img);
   return res;
}
__device__ complex pow(const complex & a, const int x) {
       complex res = a;
       for (int ii = 1; ii < x; ii++) {
           res = res * res;
       }
       return res;
}
extern "C" __global__ void mandelbrot_kernel(unsigned char * set, float zpow, float zoom, mint width, mint height) {
   int xIndex = threadIdx.x + blockIdx.x*blockDim.x;
   int yIndex = threadIdx.y + blockIdx.y*blockDim.y;
   int ii;

   float x0 = zoom*(width/2 - xIndex);
   float y0 = zoom*(height/2 - yIndex);
   complex z = {x0, y0};
   const complex c = {x0, y0};
   float cc;

   if (xIndex < width && yIndex < height) {
       for (ii = 0; (abs2(z) <= BAILOUT) && (ii < MAX_ITERATIONS); ii++) {
            z = pow(z, static_cast<float>(zpow)) + c;
        }
		cc = ii + (log(log((float) BAILOUT)) - log(log(abs(z))))/log(2.0);
        if (ii == MAX_ITERATIONS) {
            set[3*(xIndex + yIndex*width)] = 0;
            set[3*(xIndex + yIndex*width) + 1] = 0;
            set[3*(xIndex + yIndex*width) + 2] = 0;
        } else {
            set[3*(xIndex + yIndex*width)] =  zpow * ii*cc + 50;
            set[3*(xIndex + yIndex*width) + 1] = ii*cc/zpow;
            set[3*(xIndex + yIndex*width) + 2] = ii*cc + zpow;
        }
    }
}
