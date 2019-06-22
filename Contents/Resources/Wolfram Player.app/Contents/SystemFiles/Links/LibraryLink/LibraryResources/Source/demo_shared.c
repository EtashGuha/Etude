/*
 An example that demonstrates using Shared memory management for 
 communicating between Mathematica and a DLL.
*/

#include "WolframLibrary.h"

static MTensor tensor;

DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}


DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
	return 0;
}

DLLEXPORT int loadArray(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {

	tensor = MArgument_getMTensor(Args[0]);
	MArgument_setInteger(Res, 0);
	return 0;
}

DLLEXPORT int setElementVector(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint pos;
	mreal value;
	
	pos = MArgument_getInteger(Args[0]);
	value = MArgument_getReal(Args[1]);
	
	libData->MTensor_setReal( tensor, &pos, value);
	
	MArgument_setInteger(Res, 0);
	return 0;
}


DLLEXPORT int getElementVector(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint pos;
	mreal value;
	int err;
	
	pos = MArgument_getInteger(Args[0]);
	err = libData->MTensor_getReal( tensor, &pos, &value);
	
	MArgument_setReal(Res, value);
	return err;
}


DLLEXPORT int setElement(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mreal value;
	mint pos[2];
	
	pos[0] = MArgument_getInteger(Args[0]);
	pos[1] = MArgument_getInteger(Args[1]);
	value = MArgument_getReal(Args[2]);
	
	libData->MTensor_setReal( tensor, pos, value);
	
	MArgument_setInteger(Res, 0);
	return 0;
}


DLLEXPORT int getElement(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint pos[2];
	mreal value;
	int err;
	
	pos[0] = MArgument_getInteger(Args[0]);
	pos[1] = MArgument_getInteger(Args[1]);
	err = libData->MTensor_getReal( tensor, pos, &value);
	
	MArgument_setReal(Res, value);
	return err;
}




DLLEXPORT int getArray(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	MArgument_setMTensor(Res, tensor);
	return 0;
}



DLLEXPORT int unloadArray(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {

	libData->MTensor_disown( tensor);
	MArgument_setInteger(Res, 0);
	return 0;
}




