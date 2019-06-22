/*************************************************************************

        Copyright 1986 through 2010 by Wolfram Research Inc.
        All rights reserved

*************************************************************************/


#include "tetgen.h"
#include "mathlink.h"
#include "WolframLibrary.h"

#include "tetgenWolframDLL.h"

#include <unordered_map>

EXTERN_C DLLEXPORT mint WolframLibrary_getVersion( ) ;
EXTERN_C DLLEXPORT mint WolframLibrary_initialize( WolframLibraryData libData);
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData); 
EXTERN_C DLLEXPORT int deleteTetGenInstance(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
EXTERN_C DLLEXPORT int instanceList(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
EXTERN_C DLLEXPORT void manageTetGenInstance(WolframLibraryData libData, mbool mode, mint id);
EXTERN_C DLLEXPORT mbool manageTetGenCallback(WolframLibraryData libData, mint id, MTensor argtypes);

DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}

DLLEXPORT mint WolframLibrary_initialize( WolframLibraryData libData) 
{
	mint a, b;
	a = (*libData->registerLibraryCallbackManager)("TetGenManager", manageTetGenCallback);
	b = (*libData->registerLibraryExpressionManager)("TetGenManager", manageTetGenInstance);
	return a && b;
}

DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData) 
{
	(void) (*libData->unregisterLibraryCallbackManager)("TetGenManager");
	(void) (*libData->unregisterLibraryExpressionManager)("TetGenManager");
}

static std::unordered_map< mint, tetgenio *> tetgenMap;

static mint tetgenListCnt = 0;

tetgenio* getTetGenInstance( mint num)
{
	return tetgenMap[num];
}


DLLEXPORT int deleteTetGenInstance(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	if ( tetgenMap[id] == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}
	return (*libData->releaseManagedLibraryExpression)("TetGenManager", id);
}

DLLEXPORT void manageTetGenInstance(WolframLibraryData libData, mbool mode, mint id)
{
	if (mode == 0) {
		tetgenMap[id] = new tetgenio();
	} else if (tetgenMap[id] != NULL) {
		delete tetgenMap[id];
		tetgenMap.erase(id);
	}
}

DLLEXPORT int instanceList(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint i, num = tetgenMap.size();
	mint dims[1];
	MTensor resTen;

	dims[0] = num;
	int err = libData->MTensor_new( MType_Integer, 1, dims, &resTen);
	if (err)
		return err;
	mint* elems = libData->MTensor_getIntegerData( resTen);

	std::unordered_map<mint, tetgenio *>::const_iterator iter = tetgenMap.begin();
	std::unordered_map<mint, tetgenio *>::const_iterator end = tetgenMap.end();
	
	for (i = 0; i < num; i++) {
		elems[i] = iter->first;
		if ( iter != end) {
			iter++;
		}
	}
	MArgument_setMTensor(res, resTen);
	return err;
}


mint call_id = 0;
mint call_nargs = 0;

DLLEXPORT mbool manageTetGenCallback(WolframLibraryData libData, mint id, MTensor argtypes)
{
	const mint *dims;

	if (call_id) {
		(*libData->releaseLibraryCallbackFunction)(call_id);
		call_id = 0;
	}
	call_id = id;
	dims = (*libData->MTensor_getDimensions)(argtypes);
	call_nargs = dims[0] - 1;
	return True;
}


EXTERN_C DLLEXPORT int tetUnsuitableCallback( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err = 0;

	if (Argc != call_nargs) return LIBRARY_FUNCTION_ERROR;
	err = (*libData->callLibraryCallbackFunction)(call_id, call_nargs, Args, Args[2]);
	if (err) return err;

	return 0;
}

