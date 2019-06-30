
/* Include required header */
#include "WolframLibrary.h"
#include <unordered_map>

typedef std::unordered_map<mint, MTensor *> MTensorHash_t;
static MTensorHash_t map;

enum {
	A_ = 0,
	C_,
	M_,
	X_
};

DLLEXPORT void manage_instance(WolframLibraryData libData, mbool mode, mint id)
{
	if (mode == 0) {
		MTensor *T = new(MTensor);
		map[id] = T;
		*T = 0;
	} else {
		MTensor *T = map[id];
		if (T != 0) {
			if (*T != 0) libData->MTensor_free(*T);
			map.erase(id);
		}
	}
}

EXTERN_C DLLEXPORT int releaseInstance(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id;

	if (Argc != 1) return LIBRARY_FUNCTION_ERROR;
	id = MArgument_getInteger(Args[0]);
	return libData->releaseManagedLibraryExpression("LCG", id);
}

EXTERN_C DLLEXPORT int setInstanceState(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	int err;
	mint id;
	MTensor *T;
	MTensor Tin;

	if (Argc != 2) return LIBRARY_FUNCTION_ERROR;
	id = MArgument_getInteger(Args[0]);

	T = map[id];
	if (T == NULL) return LIBRARY_FUNCTION_ERROR;

	Tin = MArgument_getMTensor(Args[1]);
	if (libData->MTensor_getType(Tin) != MType_Integer) return LIBRARY_TYPE_ERROR;
	if (libData->MTensor_getRank(Tin) != 1) return LIBRARY_RANK_ERROR;
	if ((*libData->MTensor_getDimensions(Tin)) != 4) return LIBRARY_DIMENSION_ERROR;

	err = (*libData->MTensor_clone)(Tin, T);
	return err;
}

EXTERN_C DLLEXPORT int getInstanceState(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err;
	mint id;
	MTensor *T;
	MTensor Tres = 0;

	if (Argc != 1) return LIBRARY_FUNCTION_ERROR;
	id = MArgument_getInteger(Args[0]);

	T = map[id];
	if (T == NULL) return LIBRARY_FUNCTION_ERROR;

	err = (*libData->MTensor_clone)(*T, &Tres);
	if (!err)
		MArgument_setMTensor(Res, Tres);

	return err;
}

EXTERN_C DLLEXPORT int generateFromInstance(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err;
	mint id, *p;
	mint a, c, m, x;
	mint i, n, rank, *dims;
	mreal *d;
	MTensor *T;
	MTensor Tdims, Tres;

	if (Argc != 2) return LIBRARY_FUNCTION_ERROR;

	id = MArgument_getInteger(Args[0]);

	T = map[id];
	if ((T == NULL) || (*T == NULL)) return LIBRARY_FUNCTION_ERROR;

	p = libData->MTensor_getIntegerData(*T);
	a = p[A_];
	c = p[C_];
	m = p[M_];
	x = p[X_];

	Tdims = MArgument_getMTensor(Args[1]);

	if (libData->MTensor_getType(Tdims) != MType_Integer ||
		libData->MTensor_getRank(Tdims) != 1) {
		return LIBRARY_TYPE_ERROR;
	}

	rank = libData->MTensor_getFlattenedLength(Tdims);
	dims = libData->MTensor_getIntegerData(Tdims);

	err = (*libData->MTensor_new)(MType_Real, rank, dims, &Tres);
	if (err) return err;

	n = libData->MTensor_getFlattenedLength(Tres);
	d = libData->MTensor_getRealData(Tres);

	for (i = 0; i < n; i++) {
		x = (a*x + c) % m;
		d[i] = ((mreal)x) / ((mreal)m);
	}

	p[X_] = x;

	MArgument_setMTensor(Res, Tres);
	return err;
}

EXTERN_C DLLEXPORT int getAllInstanceIDs(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err;
	mint i, n = map.size();
	mint dims[1];
	MTensor Tres;

	dims[0] = n;
	err = libData->MTensor_new(MType_Integer, 1, dims, &Tres);
	if (err) return err;

	mint* elems = libData->MTensor_getIntegerData(Tres);
	MTensorHash_t::const_iterator iter = map.begin();
	MTensorHash_t::const_iterator end = map.end();
	for (i = 0; i < n; i++) {
		elems[i] = iter->first;
		if (iter != end) {
			iter++;
		}
	}
	MArgument_setMTensor(Res, Tres);
	return LIBRARY_NO_ERROR;
}


/* Return the version of Library Link */
EXTERN_C DLLEXPORT mint WolframLibrary_getVersion()
{
	return WolframLibraryVersion;
}

/* Initialize Library */
EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData)
{
	return libData->registerLibraryExpressionManager("LCG", manage_instance);
}

/* Uninitialize Library */
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData)
{
	int err = libData->unregisterLibraryExpressionManager("LCG");
}

