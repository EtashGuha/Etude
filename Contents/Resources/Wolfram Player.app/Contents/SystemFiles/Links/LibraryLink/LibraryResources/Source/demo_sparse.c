
/* Include required header */
#include "WolframLibrary.h"
#include "WolframSparseLibrary.h"
#include "string.h"


/* Return the version of Library Link */
DLLEXPORT mint WolframLibrary_getVersion() {
	return WolframLibraryVersion;
}

/* Initialize Library */
DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
	return LIBRARY_NO_ERROR;
}

/* Uninitialize Library */
DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
	return;
}

#define StringSameQ(a, b) (strcmp(a, b) == 0)

/* Access to sparse properties and CSR data */
DLLEXPORT int sparse_properties(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err = LIBRARY_NO_ERROR;
	char *what;
	mint len, type;
	MSparseArray S;
	MTensor *T, Tres = 0;
	WolframSparseLibrary_Functions sparseFuns = libData->sparseLibraryFunctions;

	if (Argc != 2) return LIBRARY_FUNCTION_ERROR;

	what = MArgument_getUTF8String(Args[0]);
	S = MArgument_getMSparseArray(Args[1]);

	if (StringSameQ(what, "ImplicitValue")) {
		T = sparseFuns->MSparseArray_getImplicitValue(S);
	}
	else if (StringSameQ(what, "ExplicitValues")) {
		T = sparseFuns->MSparseArray_getExplicitValues(S);
		if (*T == NULL) {
			T = (*(sparseFuns->MSparseArray_getImplicitValue))(S);
			type = libData->MTensor_getType(*T);
			len = 0;
			err = libData->MTensor_new(type, 1, &len, &Tres);
		}
	}
	else if (StringSameQ(what, "RowPointers")) {
		T = sparseFuns->MSparseArray_getRowPointers(S);
	}
	else if (StringSameQ(what, "ColumnIndices")) {
		T = sparseFuns->MSparseArray_getColumnIndices(S);
	}
	else if (StringSameQ(what, "ExplicitPositions")) {
		err = sparseFuns->MSparseArray_getExplicitPositions(S, &Tres);
	}
	else if (StringSameQ(what, "Normal")) {
		err = sparseFuns->MSparseArray_toMTensor(S, &Tres);
	}
	else {
		err = LIBRARY_FUNCTION_ERROR;
	}
	if (err) return err;
	if (!Tres) libData->MTensor_clone(*T, &Tres);

	MArgument_setMTensor(Res, Tres);
	return err;
}

DLLEXPORT int sparse_modify_values(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	int err = 0;
	mint i, nz;
	mreal *t, *v;
	MSparseArray S = 0, Sout = 0;
	MTensor T = 0, *Vp = 0;
	WolframSparseLibrary_Functions sparseFuns = libData->sparseLibraryFunctions;

	if (Argc != 2) return LIBRARY_FUNCTION_ERROR;

	S = MArgument_getMSparseArray(Args[0]);
	Vp = (*sparseFuns->MSparseArray_getExplicitValues)(S);
	if ((*libData->MTensor_getType)(*Vp) != MType_Real) return LIBRARY_TYPE_ERROR;
	nz = (*libData->MTensor_getFlattenedLength)(*Vp);
	v = (*libData->MTensor_getRealData)(*Vp);

	T = MArgument_getMTensor(Args[1]);
	if (libData->MTensor_getType(T) != MType_Real) return LIBRARY_TYPE_ERROR;
	if (libData->MTensor_getFlattenedLength(T) != nz) return LIBRARY_DIMENSION_ERROR;
	t = libData->MTensor_getRealData(T);

	for (i = 0; i < nz; i++) v[i] = t[i];

	/* Recompute explicit positions */
	err = sparseFuns->MSparseArray_resetImplicitValue(S, NULL, &Sout);

	sparseFuns->MSparseArray_disown(S);

	if (!err)
		MArgument_setMSparseArray(Res, Sout);

	return err;
}

