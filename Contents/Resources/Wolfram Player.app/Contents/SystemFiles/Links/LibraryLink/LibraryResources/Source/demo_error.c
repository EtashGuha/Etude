/*
 An example that demonstrates catching errors when calling 
 a Wolfram Library function from Mathematica.
*/

#include "setjmp.h"
#include "stdio.h"  /* for printf */
#include "string.h" /* for memset */
#include "WolframLibrary.h"


DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}


DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
	return 0;
}

DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData ) {

	return;
}


DLLEXPORT int errordemo1(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint I0, I1,  res;
	mint pos[2];
	mreal R0;
	int err;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	
	err = libData->MTensor_getReal(T0, &I0, &R0);
	MArgument_setReal(Res, R0);
	return err;
}

DLLEXPORT int errordemo2(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint I0, I1,  res;
	mint pos[2];
	mreal R0 = 0.;
	int err = LIBRARY_NO_ERROR;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	
	err = libData->MTensor_getReal(T0, &I0, &R0);
	if (err) {
		printf("Encountered error");
	} 
	MArgument_setReal(Res, R0);
	return 0;
}

DLLEXPORT int errordemo3(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint I0, I1,  res;
	mint pos[2];
	mreal *data;
	mreal R0;
	int err;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);
	
	err = libData->MTensor_getReal(T0, &I0, &R0);
	if (err) {
		printf("Encountered error");
	} 
	MArgument_setReal(Res, R0);

	/* Generate a second error */
	err = libData->MTensor_getReal(T0, &I0, &R0);
	return err;
}

DLLEXPORT int errordemo4(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, T1;
	mint const *dims;
	mint I0, I1,  res;
	mint pos[2];
	mreal *data;
	mreal R0;
	int err;

	T0 = MArgument_getMTensor(Args[0]);
	I0 = MArgument_getInteger(Args[1]);

	MArgument_setReal(Res, 0.);

	if (libData->MTensor_getRank(T0) != 1) {
		libData->Message("rankerror");
		return LIBRARY_RANK_ERROR;
	}

	dims = libData->MTensor_getDimensions(T0);
	if ((I0 < 1) || (I0 > dims[0])) {
		libData->Message("outofrange");
		return LIBRARY_DIMENSION_ERROR;
	}
	
	/* Generate a second error */
	err = libData->MTensor_getReal(T0, &I0, &R0);
	MArgument_setReal(Res, R0);
	return err;
}

DLLEXPORT int errordemo5(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MTensor T0, Tres;
	mint n, type, rank, *dims;
	int err = LIBRARY_NO_ERROR;

	type = MArgument_getInteger(Args[0]);
	rank = MArgument_getInteger (Args[1]);
	T0 = MArgument_getMTensor(Args[2]);

	dims = libData->MTensor_getIntegerData(T0);

	err = libData->MTensor_new(type, rank, dims, &Tres);
	if (err) return err;
	MArgument_setMTensor(Res, Tres);

	n = libData->MTensor_getFlattenedLength(Tres);
	switch (type) {
		case MType_Integer:
			memset(libData->MTensor_getIntegerData(Tres), 0, n*sizeof(mint));
			break;
		case MType_Real:
			memset(libData->MTensor_getRealData(Tres), 0, n*sizeof(mreal));
			break;
		default:
			break;
	}
	return 0;
}

DLLEXPORT int errordemo6(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	mint i, n, *p;
	MTensor T;

	n = MArgument_getInteger(Args[0]);
	T = MArgument_getMTensor (Args[1]);
	p = libData->MTensor_getIntegerData(T);

	for (i = 0; i < n; i++) {
		if (libData->AbortQ()) {
			libData->Message("aborted");
			break;
		}
	}

	p[0] = i;
	return 0;
}




DLLEXPORT int errorTest_Return(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	mint retVal;

	retVal = MArgument_getInteger(Args[0]);

	return (int)retVal;
}


