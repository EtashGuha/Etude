
/* Include required header */
#include "WolframLibrary.h"


/* Return the version of Library Link */
DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}

/* Initialize Library */
DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
	return LIBRARY_NO_ERROR;
}

/* Uninitialize Library */
DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData) {
	return;
}

/* Adds one to the input, returning the result  */
DLLEXPORT int demo_I_I( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint I0;
	mint I1;
	I0 = MArgument_getInteger(Args[0]);
	I1 = I0 + 1;
	MArgument_setInteger(Res, I1);
	return LIBRARY_NO_ERROR;
}

/* Adds one to the input, returning the result */
DLLEXPORT int demo1_I_I( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint I0;
	mint I1;
	I0 = MArgument_getInteger(Args[0]);
	I1 = I0 + 1;
	MArgument_setInteger(Res, I1);
	return LIBRARY_NO_ERROR;
}

/* Multiply two reals together, returning the result */
DLLEXPORT int demo_R_R( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mreal R0;
	mreal R1;
	R0 = MArgument_getReal(Args[0]);
	R1 = R0 * R0;
	MArgument_setReal(Res, R1);
	return LIBRARY_NO_ERROR;
}

/* Multiply two reals together, returning the result */
DLLEXPORT int demo1_R_R( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mreal R0;
	mreal R1;
	R0 = MArgument_getReal(Args[0]);
	R1 = R0 * R0;
	MArgument_setReal(Res, R1);
	return LIBRARY_NO_ERROR;
}

/* Returns Sum[i*R0, {i, I0}] */
DLLEXPORT int demo_IR_R( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mreal R0;
	mint I0;
	mint i;
	mreal R1;
	I0 = MArgument_getInteger(Args[0]);
	R0 = MArgument_getReal(Args[1]);
	R1 = 0;
	for ( i = 0; i < I0; i++) {
		R1 = R1 + i*R0;
	}
	MArgument_setReal(Res, R1);
	return LIBRARY_NO_ERROR;
}

/* Returns Sum[i*R0, {i, I0}] */
DLLEXPORT int demo1_IR_R( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mreal R0;
	mint I0;
	mint i;
	mreal R1;
	I0 = MArgument_getInteger(Args[0]);
	R0 = MArgument_getReal(Args[1]);
	R1 = 0;
	for ( i = 0; i < I0; i++) {
		R1 = R1 + i*R0;
	}
	MArgument_setReal(Res, R1);
	return LIBRARY_NO_ERROR;
}

/* Gets the I0 th Real number from the rank 1 tensor T0 */
DLLEXPORT int demo_TI_R( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0;
	mreal R0;
	int err = LIBRARY_NO_ERROR;
	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	err = libData->MTensor_getReal(T0, &I0, &R0);
	if (err) return err;
	MArgument_setReal(Res, R0);
	return LIBRARY_NO_ERROR;
}

/**
 * Same as demo_TI_R, but pass in a packed array then manually free it.
 **/
DLLEXPORT int demo1_TI_R(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0;
	mreal R0;
	int err = LIBRARY_NO_ERROR;
	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	
	err = libData->MTensor_getReal( T0, &I0, &R0);
	libData->MTensor_free(T0);
	if (err) return err;

	MArgument_setReal(Res, R0);
	return LIBRARY_NO_ERROR;
}

/**
 * Same as demo_TI_R, but just to avoid copying, so we need 
 * to disown the tensor input.
 **/
DLLEXPORT int demo2_TI_R(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0;
	mreal R0;
	int err = LIBRARY_NO_ERROR;
	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	
	err = libData->MTensor_getReal( T0, &I0, &R0);
	libData->MTensor_disown( T0);
	if (err) return err;

	MArgument_setReal(Res, R0);
	return LIBRARY_NO_ERROR;
}

/**
 * Constructs a new rank 1 tensor of length I0, and sets the
 * ith element of the vector to 2*i
 **/
DLLEXPORT int demo_I_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint i, I0, dims[1];
	int err = LIBRARY_NO_ERROR;

	I0 = MArgument_getInteger(Args[0]);
	dims[0] = I0;
	
	err = libData->MTensor_new(MType_Integer, 1, dims, &T0);
	for ( i = 1; i <= I0 && !err; i++) {
		err = libData->MTensor_setInteger( T0, &i, i*2);
	}
	MArgument_setMTensor(Res, T0);
	return err;
}

/**
 * Same as demo_TI_R, but just to avoid copying
 **/
DLLEXPORT int demo_TT_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1, T2;
	mint I0;
	mreal R0;
	int err = LIBRARY_NO_ERROR;
	T0 = MArgument_getMTensor(Args[0]);
	T1 = MArgument_getMTensor(Args[1]);
	I0 = (libData->MTensor_getIntegerData(T1))[0];	
	err = libData->MTensor_getReal( T0, &I0, &R0);
	if (err) return err;

	err = libData->MTensor_new(MType_Real, 0, NULL, &T2);
	if (err) return err;
	(libData->MTensor_getRealData(T2))[0] = R0;
	
	MArgument_setMTensor(Res, T2);
	return LIBRARY_NO_ERROR;
}

/**
 * Intended to demonstrate working with rank 0 tensors.
 *
 * The arguments are three tensors.  
 * The first is a rank > 0 real tensor.
 * The second is a rank 0 integer tensor.
 * The third a rank 0 real tensor.
 *
 * The second argument is used as an index to find
 * an element of the first, which is added to the
 * third argument to form the result.
 *
 * The result is returned as a rank 0 real tensor.
 **/
DLLEXPORT int demo_TTT_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T_arg, T_I_arg, T_R_arg, T_res;
	mint I0;
	mreal R0, R1;
	int err = LIBRARY_NO_ERROR;
	T_arg = MArgument_getMTensor(Args[0]);
	T_I_arg = MArgument_getMTensor(Args[1]);
	T_R_arg = MArgument_getMTensor(Args[2]);
	I0 = (libData->MTensor_getIntegerData(T_I_arg))[0];
	R0 = (libData->MTensor_getRealData(T_R_arg))[0];	
	err = libData->MTensor_getReal( T_arg, &I0, &R1);
	if (err) return err;
	R0 = R0 + R1;

	err = libData->MTensor_new(MType_Integer, 0, NULL, &T_res);
	if (err) return err;
	(libData->MTensor_getIntegerData( T_res))[0] = (mint) R0;
	
	MArgument_setMTensor(Res, T_res);
	return LIBRARY_NO_ERROR;
}

/**
 * Constructs a copy of the input tensor with the number of elements,
 * rank, and type appended at the end.
 **/
DLLEXPORT int demo_T_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T_arg, T_res;
	mint i, rank, type, num, len;
	mint const* dims;
	mint* intData;
	mint* intDataNew;
	mreal* realData;
	mreal* realDataNew;
	int err = LIBRARY_NO_ERROR;
	
	T_arg = MArgument_getMTensor(Args[0]);


	rank = libData->MTensor_getRank(T_arg);
	type = libData->MTensor_getType(T_arg);
	dims = libData->MTensor_getDimensions(T_arg);
	num =  libData->MTensor_getFlattenedLength(T_arg);

	/*
	The result is going to have all the elements, also 
	the length of each dimension, number of elems, rank and type.
	*/
	len = num + rank + 3;

	err = libData->MTensor_new(type, 1, &len, &T_res);
	if (err) return err;

	
	if ( type == MType_Integer) {
		intData = libData->MTensor_getIntegerData(T_arg);
		intDataNew = libData->MTensor_getIntegerData(T_res);
		for ( i = 0; i < num; i++) {
			intDataNew[i] = intData[i];
		}
		for ( i = 0; i < rank; i++) {
			intDataNew[i+num] = dims[i];
		}
		intDataNew[num+rank] = num;
		intDataNew[num+rank+1] = rank;
		intDataNew[num+rank+2] = type;
	}
	else if ( type == MType_Real) {
		realData = libData->MTensor_getRealData(T_arg);
		realDataNew = libData->MTensor_getRealData(T_res);
		for ( i = 0; i < num; i++) {
			realDataNew[i] = realData[i];
		}
		for ( i = 0; i < rank; i++) {
			realDataNew[i+num] = dims[i];
		}
		realDataNew[num+rank] = num;
		realDataNew[num+rank+1] = rank;
		realDataNew[num+rank+2] = type;
	}	
	MArgument_setMTensor(Res, T_res);
	return LIBRARY_NO_ERROR;
}

/**
 * Constructs a copy of the input tensor with the number of elements,
 * rank, and type appended at the end.
 **/
DLLEXPORT int demo1_T_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T_arg, T_res;
	mint i, rank, type, num, len;
	mint const* dims;
	mint* intData;
	mint* intDataNew;
	mreal* realData;
	mreal* realDataNew;
	int err = LIBRARY_NO_ERROR;
	
	T_arg = MArgument_getMTensor(Args[0]);


	rank = libData->MTensor_getRank(T_arg);
	type = libData->MTensor_getType(T_arg);
	dims = libData->MTensor_getDimensions(T_arg);
	num = libData->MTensor_getFlattenedLength(T_arg);

	/*
	The result is going to have all the elements, also 
	the length of each dimension, number of elems, rank and type.
	*/
	len = num + rank + 3;

	err = libData->MTensor_new(type, 1, &len, &T_res);
	if (err) return err;
	
	if ( type == MType_Integer) {
		intData = libData->MTensor_getIntegerData(T_arg);
		intDataNew = libData->MTensor_getIntegerData(T_res);
		for ( i = 0; i < num; i++) {
			intDataNew[i] = intData[i];
		}
		for ( i = 0; i < rank; i++) {
			intDataNew[i+num] = dims[i];
		}
		intDataNew[num+rank] = num;
		intDataNew[num+rank+1] = rank;
		intDataNew[num+rank+2] = type;
	}
	else if ( type == MType_Real) {
		realData = libData->MTensor_getRealData(T_arg);
		realDataNew = libData->MTensor_getRealData(T_res);
		for ( i = 0; i < num; i++) {
			realDataNew[i] = realData[i];
		}
		for ( i = 0; i < rank; i++) {
			realDataNew[i+num] = dims[i];
		}
		realDataNew[num+rank] = num;
		realDataNew[num+rank+1] = rank;
		realDataNew[num+rank+2] = type;
	}	
	MArgument_setMTensor(Res, T_res);
	return LIBRARY_NO_ERROR;
}

/**
 * Constructs a new rank 1 tensor with dimension I0, and sets the
 * ith element with the value 2*i. The newly constructed tensor is
 * returned.
 **/
DLLEXPORT int demo1_I_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint i, I0, dims[1];
	int err = LIBRARY_NO_ERROR;

	I0 = MArgument_getInteger(Args[0]);
	dims[0] = I0;
	
	err = libData->MTensor_new(MType_Integer, 1, dims, &T1);
	if (err) return err;
	libData->MTensor_free(T1);
	
	err = libData->MTensor_new(MType_Integer, 1, dims, &T0);
	for ( i = 1; i <= I0 && !err; i++) {
		err = libData->MTensor_setInteger( T0, &i, i*2);
	}
	MArgument_setMTensor(Res, T0);
	return LIBRARY_NO_ERROR;
}

/* Gets the I0th element of T0, returning that value */
DLLEXPORT int demo_T_I(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0, res;
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	err = libData->MTensor_getInteger(T0, &I0, &res);
	MArgument_setInteger(Res, res);
	return err;
}

DLLEXPORT int demo_MintSize(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint res;
	int err = LIBRARY_NO_ERROR;

	res = sizeof(res);
	MArgument_setInteger(Res, res);
	return err;
}

/* Gets the I0,I1 th integer element of T0 returning that value */
DLLEXPORT int demo_TII_I(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0, I1, res;
	mint dims[2];
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	I1 = MArgument_getInteger(Args[2]);
	dims[0] = I0;
	dims[1] = I1;
	err = libData->MTensor_getInteger(T0, dims, &res);
	MArgument_setInteger(Res, res);
	return err;
}

/* Sets the I0,I1 th integer element of T0 with value, returning that position */
DLLEXPORT int demo_TIII_I(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0, I1, value, res;
	mint dims[2];
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	I1 = MArgument_getInteger(Args[2]);
	value = MArgument_getInteger(Args[3]);
	dims[0] = I0;
	dims[1] = I1;
	err = libData->MTensor_setInteger(T0, dims, value);
	if (err) return err;
	err = libData->MTensor_getInteger(T0, dims, &res);
	MArgument_setInteger(Res, res);
	return err;
}

/* Gets the I0,I1 th real element of T0 returning that value */
DLLEXPORT int demo_TII_R(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0, I1;
	mreal res;
	mint dims[2];
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	I1 = MArgument_getInteger(Args[2]);
	dims[0] = I0;
	dims[1] = I1;
	err = libData->MTensor_getReal(T0, dims, &res);
	MArgument_setReal(Res, res);
	return err;
}

/* Sets the I0,I1 th real element of T0 with value, returning that position */
DLLEXPORT int demo_TIIR_R(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0;
	mint I0, I1;
	mreal value, res;
	mint dims[2];
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	I1 = MArgument_getInteger(Args[2]);
	value = MArgument_getReal(Args[3]);
	dims[0] = I0;
	dims[1] = I1;
	err = libData->MTensor_setReal(T0, dims, value);
	if (err) return err;
	err = libData->MTensor_getReal(T0, dims, &res);
	MArgument_setReal(Res, res);
	return err;
}

/* Gets the subpart of the input tensor starting at the I0 th position */
DLLEXPORT int demo_TI_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1 = 0;
	mint I0;
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	err = libData->MTensor_getMTensor(T0, &I0, 1, &T1);
	MArgument_setMTensor(Res, T1);
	return err;
}

/* Sets the I0 th element in T0 to its value in T1 */
DLLEXPORT int demo_TTI_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint I0;
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	T1 = MArgument_getMTensor(Args[1]);
	I0 = MArgument_getInteger(Args[2]);
	err = libData->MTensor_setMTensor(T0, T1, &I0, 1);
	MArgument_setMTensor(Res, T0);
	return err;
}

/* Gets the subpart of the input tensor starting at the I0,I1 th position */
DLLEXPORT int demo_TII_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1 = 0;
	mint I0, I1;
	mint pos[2];
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	I1 = MArgument_getInteger(Args[2]);
	pos[0] = I0;
	pos[1] = I1;
	err = libData->MTensor_getMTensor(T0, pos, 2, &T1);
	MArgument_setMTensor(Res, T1);
	return err;
}

/* Sets the element in the I0,I1 position in T0 to its value in T1, returning T0 */
DLLEXPORT int demo_TTII_T(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint I0, I1;
	mint pos[2];
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	T1 = MArgument_getMTensor(Args[1]);
	I0 = MArgument_getInteger(Args[2]);
	I1 = MArgument_getInteger(Args[3]);
	pos[0] = I0;
	pos[1] = I1;
	err = libData->MTensor_setMTensor(T0, T1, pos, 2);
	MArgument_setMTensor(Res, T0);
	return err;
}

/* Accepts no inputs, but returns a result */
DLLEXPORT int demoNoArguments(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	mint b = Argc;
	MArgument_setInteger(Res, b);
	return LIBRARY_NO_ERROR;
}

/* Accepts inputs, but returns nothing */
DLLEXPORT int demoNoResult(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	MTensor T = MArgument_getMTensor(Args[0]);
	mint *p = libData->MTensor_getIntegerData(T);

	*p = MArgument_getAddress(Res) == 0? 0:1;
	return LIBRARY_NO_ERROR;
}

/* Returns BitNot[b1] of the input*/
DLLEXPORT int demoBoolean1(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	mbool b1 = MArgument_getBoolean(Args[0]);
	MArgument_setBoolean(Res, !b1);
	return LIBRARY_NO_ERROR;
}

/* Returns the complex conjugate of the complex number inputed */
DLLEXPORT int demoComplex1(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	mcomplex c1 = MArgument_getComplex(Args[0]);
	mcomplex c2;
	mcreal(c2) = mcreal(c1);
	mcimag(c2) = -mcimag(c1);
	MArgument_setComplex(Res, c2);
	return LIBRARY_NO_ERROR;
}

/* Sets the output to the number of arguments passed in to the function */
DLLEXPORT int demoNoInput(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	MArgument_setInteger(Res, Argc);
	return LIBRARY_NO_ERROR;
}

/* Accepts and returns nothing */
DLLEXPORT int demoVoid(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	return LIBRARY_NO_ERROR;
}

/* Accepts and returns nothing */
DLLEXPORT int demoNoInputVoid(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	return LIBRARY_NO_ERROR;
}



