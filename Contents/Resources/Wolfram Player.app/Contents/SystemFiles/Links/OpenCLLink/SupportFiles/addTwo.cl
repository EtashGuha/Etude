__kernel void addTwo(__global mint * A, mint length) {
    int index = get_global_id(0);
    if (index < length)
        A[index] += 2;
}
