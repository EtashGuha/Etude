__kernel void rule30_kernel(__global mint * prev, __global mint * nxt, mint width) {
    int index = get_global_id(0);
    int p, q, r;
    
    if (index < width) {
        p = index == 0 ? 0 : prev[index-1];
        q = prev[index];
        r = index == width-1 ? 0 : prev[index+1];
        nxt[index] = p ^ (q | r); 
    }
}
