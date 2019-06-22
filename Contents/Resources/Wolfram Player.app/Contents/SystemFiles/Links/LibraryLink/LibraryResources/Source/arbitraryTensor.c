/*
 An example that demonstrates using Shared memory management for 
 communicating between Mathematica and a Wolfram Library.
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
	return 0;
}

DLLEXPORT int getElementShared(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint pos[1];
	mreal* elems;
	mreal value;
	
	pos[0] = MArgument_getInteger(Args[0]);
	elems = libData->MTensor_getRealData(tensor);
	value = elems[pos[0]];
	MArgument_setReal(Res, value);
	return 0;
}

DLLEXPORT int getElementNonShared(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint pos[1];
	MTensor T0;
	mreal* elems;
	mreal value;
	
	pos[0] = MArgument_getInteger(Args[0]);
	T0 = MArgument_getMTensor(Args[1]);
	elems = libData->MTensor_getRealData(T0);
	value = elems[pos[0]];
	MArgument_setReal(Res, value);
	return 0;
}


DLLEXPORT int unloadArray(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {

	libData->MTensor_disown( tensor);
	return 0;
}




