__kernel void vectorAdd(__global mint * a, __global mint * b, __global mint * c, mint n) {
    int iGID = get_global_id(0);

    if (iGID < n) {   
        c[iGID] = a[iGID] + b[iGID];
    }
}
