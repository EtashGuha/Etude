/*

Copyright (c) 2007-2009 The Regents of the University of California, Davis
campus ("The Regents") and NVIDIA Corporation ("NVIDIA"). All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, 
      this list of conditions and the following disclaimer in the documentation 
      and/or other materials provided with the distribution.
    * Neither the name of the The Regents, nor NVIDIA, nor the names of its 
      contributors may be used to endorse or promote products derived from this 
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision$
//  $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * rand_cta.cu
 * 
 * @brief CUDPP CTA-level rand routines
 */

/** \addtogroup cudpp_cta 
* @{
*/

/** @name Rand Functions
* @{
*/


//------------MD5 ROTATING FUNCTIONS------------------------

/**
 * @brief Does a GLSL-style swizzle assigning f->xyzw = f->yzwx
 * 
 *  It does the equvalent of f->xyzw = f->yzwx since this functionality is 
 *  in shading languages but not exposed in CUDA.
 *  @param[in] f the uint4 data type which will have its elements shifted.  Passed in as pointer.
 * 
**/
__device__ void swizzleShift(uint4 *f)
{
    unsigned int temp;
    temp = f->x;
    f->x = f->y;
    f->y = f->z;
    f->z = f->w;
    f->w = temp;
}
/**
 * @brief Rotates the bits in \a x over by \a n bits.
 * 
 *  This is the equivalent of the ROTATELEFT operation as described in the MD5 working memo.
 *  It takes the bits in \a x and circular shifts it over by \a n bits.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x the variable with the bits 
 *  @param[in] n the number of bits to shift left by.
**/
__device__ unsigned int leftRotate(unsigned int x, unsigned int n)
{
    unsigned int t = ( ((x) << (n)) | ((x) >> (32-n)) ) ;
    return t;
}

/**
 * @brief The F scrambling function.
 * 
 *  The F function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x & y) | ((~x) & z)
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see FF()
**/
__device__ unsigned int F(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = ( (x&y) | ((~x) & z) );
    return t;
}

/**
 * @brief The G scrambling function.
 * 
 *  The G function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x & z) | ((~z) & y)
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see GG()
**/
__device__ unsigned int G(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = ( (x&z) | ((~z) & y) );
    return t;
}

/**
 * @brief The H scrambling function.
 * 
 *  The H function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x ^ y ^ z)
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see HH()
**/
__device__ unsigned int H(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = (x ^ y ^ z );
    return t;
}

/**
 * @brief The I scrambling function.
 * 
 *  The I function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (y ^ (x | ~z))
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see II()
**/
__device__ unsigned int I(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = ( y ^ (x | ~z) );
    return t;
}

/**
 * @brief The FF scrambling function
 * 
 *  The FF function in the MD5 technical memo is a wrapper for the F scrambling function
 *  as well as performing its own rotations using LeftRotate and swizzleShift.  The variable 
 *  \a td is the current scrambled digest which is passed along and scrambled using the current 
 *  iteration \a i, the rotation information \a Fr, and the starting input \a data.  \a p is kept as a 
 *  constant of 2^32.
 *  The resulting value is stored in \a td.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Fr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see F()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void FF(uint4 * td, int i, uint4 * Fr, float p, unsigned int * data)
{
    unsigned int Ft = F(td->y, td->z, td->w);
    unsigned int r = Fr->x;
    swizzleShift(Fr);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief The GG scrambling function
 * 
 *  The GG function in the MD5 technical memo is a wrapper for the G scrambling function
 *  as well as performing its own rotations using LeftRotate() and swizzleShift().  The variable 
 *  \a td is the current scrambled digest which is passed along and scrambled using the current 
 *  iteration \a i, the rotation information \a Gr, and the starting input \a data.  \a p is kept as a 
 *  constant of 2^32.
 *  The resulting value is stored in \a td.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Gr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see G()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void GG(uint4 * td, int i, uint4 * Gr, float p, unsigned int * data)
{
    unsigned int Ft = G(td->y, td->z, td->w);
    i = (5*i+1) %16;
    unsigned int r = Gr->x;
    swizzleShift(Gr);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief The HH scrambling function
 * 
 *  The HH function in the MD5 technical memo is a wrapper for the H scrambling function
 *  as well as performing its own rotations using LeftRotate() and swizzleShift().  The variable 
 *  \a td is the current scrambled digest which is passed along and scrambled using the current 
 *  iteration \a i, the rotation information \a Hr, and the starting input \a data.  \a p is kept as a 
 *  constant of 2^32.
 *  The resulting value is stored in \a td.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Hr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see H()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void HH(uint4 * td, int i, uint4 * Hr, float p, unsigned int * data)
{
    unsigned int Ft = H(td->y, td->z, td->w);
    i = (3*i+5) %16;
    unsigned int r = Hr->x;
    swizzleShift(Hr);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief The II scrambling function
 * 
 *  The II function in the MD5 technical memo is a wrapper for the I scrambling function
 *  as well as performing its own rotations using LeftRotate() and swizzleShift().  The variable 
 *  \a td is the current scrambled digest which is passed along and scrambled using the current 
 *  iteration \a i, the rotation information \a Ir, and the starting input \a data.  \a p is kept as a 
 *  constant of 2^32.
 *  The resulting value is stored in \a td.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Ir The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see I()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void II(uint4 * td, int i, uint4 * Ir, float p, unsigned int * data)
{
    unsigned int Ft = G(td->y, td->z, td->w);
    i = (7*i) %16;
    unsigned int r = Ir->x;
    swizzleShift(Ir);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief Sets up the \a input array using information of \a seed, and \a threadIdx
 * 
 *  This function sets up the \a input array using a combination of the current thread's id and the 
 *  user supplied \a seed. 
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[out] input The array which will contain the initial values for all the scrambling functions.
 *  @param[in] seed The user supplied seed as an unsigned int.
 *
 *  @see FF()
 *  @see GG()
 *  @see HH()
 *  @see II()
 *  @see gen_randMD5()
**/
__device__ void setupInput(unsigned int * input, unsigned int seed)
{    
    //loop unroll, also do this more intelligently
    input[0] = threadIdx.x ^ seed;
    input[1] = threadIdx.y ^ seed;
    input[2] = threadIdx.z ^ seed;
    input[3] = 0x80000000 ^ seed;
    input[4] = blockIdx.x ^ seed;
    input[5] = seed;
    input[6] = seed;
    input[7] = blockDim.x ^ seed;
    input[8] = seed;
    input[9] = seed;
    input[10] = seed;
    input[11] = seed;
    input[12] = seed;
    input[13] = seed;
    input[14] = seed;
    input[15] = 128 ^ seed;
}

//-------------------END MD5 FUNCTIONS--------------------------------------

/** @} */ // end rand functions
/** @} */ // end cudpp_cta

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision$
//  $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * rand_kernel.cu
 *
 * @brief CUDPP kernel-level rand routines
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name Rand Functions
 * @{
 */

/**
 * @brief The main MD5 generation algorithm.
 *
 * This function runs the MD5 hashing random number generator.  It generates
 * MD5 hashes, and uses the output as randomized bits.  To repeatedly call this
 * function, always call cudppRandSeed() first to set a new seed or else the output
 * may be the same due to the deterministic nature of hashes.  gen_randMD5 generates
 * 128 random bits per thread.  Therefore, the parameter \a d_out is expected to be
 * an array of type uint4 with \a numElements indicies.
 *
 * @param[out] d_out the output array of type uint4.
 * @param[in] numElements the number of elements in \a d_out
 * @param[in] seed the random seed used to vary the output
 *
 * @see launchRandMD5Kernel()
 */
__global__ void gen_randMD5(uint4 *d_out, mint numElements, mint seed)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int data[16];
    setupInput(data, seed);
    
    unsigned int h0 = 0x67452301;
    unsigned int h1 = 0xEFCDAB89;
    unsigned int h2 = 0x98BADCFE;
    unsigned int h3 = 0x10325476;

    uint4 result = make_uint4(h0,h1,h2,h3);
    uint4 td = result;

    float p = pow(2.0,32.0);
    
    uint4 Fr = make_uint4(7,12,17,22);
    uint4 Gr = make_uint4(5,9,14,20);
    uint4 Hr = make_uint4(4,11,16,23);
    uint4 Ir = make_uint4(6,10,15,21);    
    
    //for optimization, this is loop unrolled
    FF(&td, 0, &Fr,p,data);
    FF(&td, 1, &Fr,p,data);
    FF(&td, 2, &Fr,p,data);
    FF(&td, 3, &Fr,p,data);
    FF(&td, 4, &Fr,p,data);
    FF(&td, 5, &Fr,p,data);
    FF(&td, 6, &Fr,p,data);
    FF(&td, 7, &Fr,p,data);
    FF(&td, 8, &Fr,p,data);
    FF(&td, 9, &Fr,p,data);
    FF(&td,10, &Fr,p,data);
    FF(&td,11, &Fr,p,data);
    FF(&td,12, &Fr,p,data);
    FF(&td,13, &Fr,p,data);
    FF(&td,14, &Fr,p,data);
    FF(&td,15, &Fr,p,data);

    GG(&td,16, &Gr,p,data);
    GG(&td,17, &Gr,p,data);
    GG(&td,18, &Gr,p,data);
    GG(&td,19, &Gr,p,data);
    GG(&td,20, &Gr,p,data);
    GG(&td,21, &Gr,p,data);
    GG(&td,22, &Gr,p,data);
    GG(&td,23, &Gr,p,data);
    GG(&td,24, &Gr,p,data);
    GG(&td,25, &Gr,p,data);
    GG(&td,26, &Gr,p,data);
    GG(&td,27, &Gr,p,data);
    GG(&td,28, &Gr,p,data);
    GG(&td,29, &Gr,p,data);
    GG(&td,30, &Gr,p,data);
    GG(&td,31, &Gr,p,data);

    HH(&td,32, &Hr,p,data);
    HH(&td,33, &Hr,p,data);
    HH(&td,34, &Hr,p,data);
    HH(&td,35, &Hr,p,data);
    HH(&td,36, &Hr,p,data);
    HH(&td,37, &Hr,p,data);
    HH(&td,38, &Hr,p,data);
    HH(&td,39, &Hr,p,data);
    HH(&td,40, &Hr,p,data);
    HH(&td,41, &Hr,p,data);
    HH(&td,42, &Hr,p,data);
    HH(&td,43, &Hr,p,data);
    HH(&td,44, &Hr,p,data);
    HH(&td,45, &Hr,p,data);
    HH(&td,46, &Hr,p,data);
    HH(&td,47, &Hr,p,data);

    II(&td,48, &Ir,p,data);
    II(&td,49, &Ir,p,data);
    II(&td,50, &Ir,p,data);
    II(&td,51, &Ir,p,data);
    II(&td,52, &Ir,p,data);
    II(&td,53, &Ir,p,data);
    II(&td,54, &Ir,p,data);
    II(&td,55, &Ir,p,data);
    II(&td,56, &Ir,p,data);
    II(&td,57, &Ir,p,data);
    II(&td,58, &Ir,p,data);
    II(&td,59, &Ir,p,data);
    II(&td,60, &Ir,p,data);
    II(&td,61, &Ir,p,data);
    II(&td,62, &Ir,p,data);
    II(&td,63, &Ir,p,data);
/*    */        
    result.x = result.x + td.x;
    result.y = result.y + td.y;
    result.z = result.z + td.z;
    result.w = result.w + td.w;

    __syncthreads();

    if (idx < numElements)
    {
        d_out[idx].x = result.x;
        d_out[idx].y = result.y;
        d_out[idx].z = result.z;
        d_out[idx].w = result.w;
    }
}
/** @} */ // end rand functions
/** @} */ // end cudpp_kernel
