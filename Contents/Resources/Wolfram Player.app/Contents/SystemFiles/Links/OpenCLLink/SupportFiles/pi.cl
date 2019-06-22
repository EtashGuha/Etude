#ifndef DBL_EPSILON
#define DBL_EPSILON 0.0001f
#endif

#define NUM_EVAL_PER_KERNEL 8
__kernel void integratePI(__global mint * out, unsigned mint n) {
    const int index = get_global_id(0)/NUM_EVAL_PER_KERNEL;
    
    if (index >= n)
        return ;
    
    float digit, rndX, rndY, idx, div;
	int ii, accum = 0;
	
	for (ii = 0; ii < NUM_EVAL_PER_KERNEL; ii++) {
	    rndX = 0;
	    digit = 0;
	    idx = index + ii;
	    div = 1.0f/2.0f;
	    while (idx > DBL_EPSILON) {
	       digit = ((int)idx)%2;
	       rndX += div*digit;
	       idx = (idx - digit)/2.0f;
	       div /= 2.0f;
	    }
	
	    rndY = 0;
	    digit = 0;
	    idx = index + ii;
	    div = 1.0f/3.0f;
	    while (idx > DBL_EPSILON) {
	       digit = ((int)idx)%3;
	       rndY += div*digit;
	       idx = (idx - digit)/3.0f;
	       div /= 3.0f;
	    }
	    if (rndX*rndX + rndY*rndY <= 1)
	    	accum++;
	}
    out[index] = accum;
}