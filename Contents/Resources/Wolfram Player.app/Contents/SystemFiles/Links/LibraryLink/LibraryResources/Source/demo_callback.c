/* Include required header */
#include "WolframLibrary.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"


/* Return the version of Library Link */
DLLEXPORT mint WolframLibrary_getVersion()
{
	return WolframLibraryVersion;
}

static mint call_id;
static mint call_nargs;
static MArgument *cbArgs;
static mreal **tdata;

DLLEXPORT mbool manage_callback(WolframLibraryData libData, mint id, MTensor argtypes)
{
	mint i;
	mint *dims;
	mint *typerank;
	if (call_id) {
		(*libData->releaseLibraryCallbackFunction)(call_id);
		call_id = 0;
		free(cbArgs);
		free(tdata);
	}
	call_id = id;
	dims = (*libData->MTensor_getDimensions)(argtypes);
	call_nargs = dims[0] - 1;
	if (call_nargs == 0) {
		call_id = 0;
		return False;
	}
	typerank = (*libData->MTensor_getIntegerData)(argtypes);
	/* Check that the arguments and result (thus i <= call_nargs loop control) are scalar mreal */
	for (i = 0; i <= call_nargs; i++) {
		/* Each row is {type, rank} */
		if ((typerank[0] != MType_Real) || (typerank[1] != 0)) {
			call_id = 0;
			call_nargs = 0;
			return False;
		}
		typerank += 2;
	}
	cbArgs = (MArgument *) malloc((call_nargs + 1)*sizeof(MArgument));
	tdata = (mreal **) malloc(call_nargs*sizeof(mreal *));
	return True;
}

/* Initialize Library */
DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData)
{
	call_id = 0;
	return (*libData->registerLibraryCallbackManager)("demo_callback_manager", manage_callback);
}

/* Uninitialize Library */
DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData)
{
	(*libData->unregisterLibraryCallbackManager)("demo_callback_manager");
}

DLLEXPORT int apply_callback( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err = 0;
	int type;
	mint i, j, n, rank;
	mint const *dims;
	mreal *r;
	MTensor T, Tres = 0;

	if (call_id <= 0) return LIBRARY_FUNCTION_ERROR;
	if (Argc != call_nargs) return LIBRARY_FUNCTION_ERROR;

	T = MArgument_getMTensor(Args[0]);
	type = (*libData->MTensor_getType)(T);
	if (type != MType_Real) return LIBRARY_TYPE_ERROR;
	rank = (*libData->MTensor_getRank)(T);
	dims = (*libData->MTensor_getDimensions)(T);
	n = (*libData->MTensor_getFlattenedLength)(T);
	tdata[0] = (*libData->MTensor_getRealData)(T);

	for (j = 1; j < call_nargs; j++) {
		T = MArgument_getMTensor(Args[j]);
		type = (*libData->MTensor_getType)(T);
		if (type != MType_Real) return LIBRARY_TYPE_ERROR;
		if ((*libData->MTensor_getFlattenedLength)(T) != n) return LIBRARY_DIMENSION_ERROR;
		tdata[j] = (*libData->MTensor_getRealData)(T);
	}

	err = (*libData->MTensor_new)(type, rank, dims, &Tres);
	if (err) return err;
	r = (*libData->MTensor_getRealData)(Tres);

	for (i = 0; i < n; i++) {
		for (j = 0; j < call_nargs; j++) MArgument_getRealAddress(cbArgs[j]) = tdata[j] + i;
		MArgument_getRealAddress(cbArgs[call_nargs]) = r + i;
		err = (*libData->callLibraryCallbackFunction)(call_id, call_nargs, cbArgs, cbArgs[1]);
		if (err) {
			(*libData->MTensor_free)(Tres);
			return err;
		}
	}

	MArgument_setMTensor(Res, Tres);
	return 0;
}

DLLEXPORT int apply_sin( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err = 0;
	int type;
	mint i, n, rank;
	mint const *dims;
	mreal *t, *r;
	MTensor T, Tres = 0;

	if (Argc != 1) return LIBRARY_FUNCTION_ERROR;

	T = MArgument_getMTensor(Args[0]);
	type = (*libData->MTensor_getType)(T);
	rank = (*libData->MTensor_getRank)(T);
	dims = (*libData->MTensor_getDimensions)(T);
	n = (*libData->MTensor_getFlattenedLength)(T);
	t = (*libData->MTensor_getRealData)(T);

	err = (*libData->MTensor_new)(type, rank, dims, &Tres);
	if (err) return err;
	r = (*libData->MTensor_getRealData)(Tres);

	for (i = 0; i < n; i++) {
		r[i] = sin(t[i]);
	}

	MArgument_setMTensor(Res, Tres);
	return 0;
}
