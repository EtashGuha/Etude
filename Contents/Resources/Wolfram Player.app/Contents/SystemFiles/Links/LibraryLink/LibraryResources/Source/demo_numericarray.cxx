/* Include required headers */
#include <cstdint>
#include <complex>
#include <cstring>
#include <fstream>
#include <vector>

#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"


/* Return the version of Library Link */
DLLEXPORT mint WolframLibrary_getVersion() { return WolframLibraryVersion; }

/* Initialize Library */
DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
	return LIBRARY_NO_ERROR;
}

/* Uninitialize Library */
DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
	return;
}

template <typename T>
static void tplNumericArrayReverse(void *out0, const void *in0, mint length) {
	T *out = static_cast<T *>(out0);
	const T *in = static_cast<const T *>(in0);

	for (mint i = 0; i < length; i++) {
		out[length - i - 1] = in[i];
	}
}

/* Reverses elements in a one-dimensional NumericArray */
EXTERN_C DLLEXPORT int numericArrayReverse(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res) {
	mint length;
	MNumericArray na_in = NULL, na_out = NULL;
	void *data_in = NULL, *data_out = NULL;
	int err = LIBRARY_FUNCTION_ERROR;
	int rank = 0;
	numericarray_data_t type = MNumericArray_Type_Undef;
	WolframNumericArrayLibrary_Functions naFuns = libData->numericarrayLibraryFunctions;

	na_in = MArgument_getMNumericArray(Args[0]);

	err = naFuns->MNumericArray_clone(na_in, &na_out);
	if (err) {
		return err;
	}
	rank = naFuns->MNumericArray_getRank(na_in);
    if(rank != 1) {
		return err;
	}	
	type = naFuns->MNumericArray_getType(na_in);
	length = naFuns->MNumericArray_getFlattenedLength(na_in);
	data_in = naFuns->MNumericArray_getData(na_in);
	data_out = naFuns->MNumericArray_getData(na_out);
	if (data_in == NULL || data_out == NULL) {
		goto cleanup;
	}

	switch (type) {
	case MNumericArray_Type_Bit8:
		tplNumericArrayReverse<std::int8_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_UBit8:
		tplNumericArrayReverse<std::uint8_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Bit16:
		tplNumericArrayReverse<std::int16_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_UBit16:
		tplNumericArrayReverse<std::uint16_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Bit32:
		tplNumericArrayReverse<std::int32_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_UBit32:
		tplNumericArrayReverse<std::uint32_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Bit64:
		tplNumericArrayReverse<std::int64_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_UBit64:
		tplNumericArrayReverse<std::uint64_t>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Real32:
		tplNumericArrayReverse<float>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Real64:
		tplNumericArrayReverse<double>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Complex_Real32:
		tplNumericArrayReverse<std::complex<float>>(data_out, data_in, length);
		break;
	case MNumericArray_Type_Complex_Real64:
		tplNumericArrayReverse<std::complex<double>>(data_out, data_in, length);
		break;
	default:
		goto cleanup;
	}

	MArgument_setMNumericArray(res, na_out);
	return LIBRARY_NO_ERROR;

cleanup:
	naFuns->MNumericArray_free(na_out);
	return err;
}

template <typename T>
static void tplNumericArrayComplexConjugate(void *inout0, mint length) {
	T *inout = static_cast<T *>(inout0);
	
	for (mint i = 0; i < length; i++) {
		inout[i] = std::conj(inout[i]);
	}
}

/* Computes the complex conjugate of each element in a NumericArray. 
   NumericArrays of non-complex types are converted to MNumericArray_Type_Complex_Real64 */
EXTERN_C DLLEXPORT int numericArrayComplexConjugate(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res) {
	mint length;
	MNumericArray na_in = NULL, na_out = NULL;
	void *data_out = NULL;
	int err = LIBRARY_FUNCTION_ERROR;
	numericarray_data_t type = MNumericArray_Type_Undef;
	WolframNumericArrayLibrary_Functions naFuns = libData->numericarrayLibraryFunctions;

	na_in = MArgument_getMNumericArray(Args[0]);
	type = naFuns->MNumericArray_getType(na_in);
	if(type != MNumericArray_Type_Complex_Real32 && type != MNumericArray_Type_Complex_Real64) {
		err = naFuns->MNumericArray_convertType(&na_out, na_in, MNumericArray_Type_Complex_Real64, MNumericArray_Convert_Coerce, 0);	
		if(err != LIBRARY_NO_ERROR) {
			return err;
		}
	}
	else {
		err = naFuns->MNumericArray_clone(na_in, &na_out);
		if (err) {
			return err;
		}
		length = naFuns->MNumericArray_getFlattenedLength(na_out);
		data_out = naFuns->MNumericArray_getData(na_out);
		if (data_out == NULL) {
			goto cleanup;
		}

		switch (type) {
		case MNumericArray_Type_Complex_Real32:
			tplNumericArrayComplexConjugate<std::complex<float>>(data_out, length);
			break;
		case MNumericArray_Type_Complex_Real64:
			tplNumericArrayComplexConjugate<std::complex<double>>(data_out, length);
			break;
		default:
			goto cleanup;
		}
	}

	MArgument_setMNumericArray(res, na_out);
	return LIBRARY_NO_ERROR;

cleanup:
	naFuns->MNumericArray_free(na_out);
	return err;
}

static mint readBytes(char const* filename, std::uint8_t** bytes) {
	std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
	if (ifs) {
		std::ifstream::pos_type pos = ifs.tellg();
		*bytes = new std::uint8_t[pos];
		ifs.seekg(0, std::ios::beg);
		ifs.read(reinterpret_cast<char*>(*bytes), pos);
		return pos;
	}
	return 0;
}

/* Reads data from a file and returns it as a ByteArray */
EXTERN_C DLLEXPORT int readBytesFromFile(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res) {
	mint length = 0;
	char *filename = NULL;
	MNumericArray barray = NULL;
	std::uint8_t *barray_data = NULL;
	std::uint8_t *bytes = NULL;
	int err = LIBRARY_FUNCTION_ERROR;
	WolframNumericArrayLibrary_Functions naFuns = libData->numericarrayLibraryFunctions;

	filename = MArgument_getUTF8String(Args[0]);
	length = readBytes(filename, &bytes);
	if (length > 0) {
		err = naFuns->MNumericArray_new(MNumericArray_Type_UBit8, 1, &length, &barray);
		if (err != 0) {
			goto cleanup;
		}
		barray_data = static_cast<std::uint8_t*>(naFuns->MNumericArray_getData(barray));
		std::memcpy(barray_data, bytes, length);
	}
	else {
		goto cleanup;
	}

cleanup:
	if (err == LIBRARY_NO_ERROR) {
		MArgument_setMNumericArray(res, barray);
	}
	delete[] bytes;
	libData->UTF8String_disown(filename);
	return err;
}
