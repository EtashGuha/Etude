
#include<math.h>

__global__ void radial(Real_t* v, Real_t* z, Real_t* r, int *state, Real_t acc, int size, int seq)  {
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int i=ix*size+iy;
	if(ix < size && iy< size && sqrt((float)((ix-size/2)*(ix-size/2)+(iy-size/2)*(iy-size/2))) >=(float)seq ) {
	    if(v[i]<=0) {
	    	v[i]=0;
	     	state[i]=1;
		}
	    if(z[i]<=0) {
			z[i]=0;
			state[i]=-1;
		}
	    v[i]+=state[i]*acc*r[i]/.25;
	    z[i]-=state[i]*v[i];
    }
}

__global__ void sinusoidal(Real_t* v, Real_t* z, Real_t* r, int *state, Real_t acc, int size)  {
       	
	int i=threadIdx.x + blockIdx.x*blockDim.x;
	if(i < size * size ) {
		if(v[i]<=0) {
			v[i]=0;
			state[i]=1;
		}
		if(z[i]<=0) {
			z[i]=0;
			state[i]=-1;
		}
		v[i]+=state[i]*acc;
		z[i]-=state[i]*v[i]*r[i]/.25;
	}
}
