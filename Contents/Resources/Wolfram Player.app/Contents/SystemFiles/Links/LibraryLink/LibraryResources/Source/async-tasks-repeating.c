#include <stdlib.h>
#include "WolframLibrary.h"
#include "WolframCompileLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "async-examples.h"

/**********************************************************/
typedef struct IntBackgroundArgs_st
{
	WolframIOLibrary_Functions ioLibrary;
	int pausemillis;
	mint eventdata;
}* IntBackgroundArgs;

static void IntBackgroundTask(mint asyncObjID, void* vtarg)
{
	IntBackgroundArgs targ = (IntBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	mint eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	while(ioLibrary->asynchronousTaskAliveQ(asyncObjID))
	{
		PortableSleep(pausems);
		ds = ioLibrary->createDataStore();
		ioLibrary->DataStore_addInteger(ds, eventdata);
		ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
		eventdata++;
	}
}

DLLEXPORT int start_int_repeating_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	IntBackgroundArgs threadArg = (IntBackgroundArgs)malloc(sizeof(struct IntBackgroundArgs_st));
	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->ioLibrary = ioLibrary;
	threadArg->pausemillis = MArgument_getInteger(Args[0]);
	threadArg->eventdata = MArgument_getInteger(Args[1]);
	asyncObjID = ioLibrary->createAsynchronousTaskWithThread(IntBackgroundTask, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}

/**********************************************************/
static void RealBackgroundTask(mint asyncObjID, void*);

typedef struct RealBackgroundArgs_st
{
	WolframIOLibrary_Functions ioLibrary;
	int pausemillis;
	mreal eventdata;
}* RealBackgroundArgs;

static void RealBackgroundTask(mint asyncObjID, void* vtarg)
{
	RealBackgroundArgs targ = (RealBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	mreal eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	while(ioLibrary->asynchronousTaskAliveQ(asyncObjID))
	{
		PortableSleep(pausems);
		ds = ioLibrary->createDataStore();
		ioLibrary->DataStore_addReal(ds, eventdata);
		ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
		eventdata += 1.0;
	}
}

DLLEXPORT int start_real_repeating_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	RealBackgroundArgs threadArg = (RealBackgroundArgs)malloc(sizeof(struct RealBackgroundArgs_st));
	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->ioLibrary = ioLibrary;
	threadArg->pausemillis = MArgument_getInteger(Args[0]);
	threadArg->eventdata = MArgument_getReal(Args[1]);
	asyncObjID = ioLibrary->createAsynchronousTaskWithThread(RealBackgroundTask, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}

/**********************************************************/
static void ComplexBackgroundTask(mint asyncObjID, void*);

typedef struct ComplexBackgroundArgs_st
{
	WolframIOLibrary_Functions ioLibrary;
	int pausemillis;
	mcomplex eventdata;
}* ComplexBackgroundArgs;

static void ComplexBackgroundTask(mint asyncObjID, void* vtarg)
{
	ComplexBackgroundArgs targ = (ComplexBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	mcomplex eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	while(ioLibrary->asynchronousTaskAliveQ(asyncObjID))
	{
		PortableSleep(pausems);
		ds = ioLibrary->createDataStore();
		ioLibrary->DataStore_addComplex(ds, eventdata);
		ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
		mcreal(eventdata) = mcreal(eventdata)+1.0;
		mcimag(eventdata) = mcimag(eventdata)+1.0;
	}
}

DLLEXPORT int start_complex_repeating_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	ComplexBackgroundArgs threadArg = (ComplexBackgroundArgs)malloc(sizeof(struct ComplexBackgroundArgs_st));
	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->ioLibrary = ioLibrary;
	threadArg->pausemillis = MArgument_getInteger(Args[0]);
	threadArg->eventdata = MArgument_getComplex(Args[1]);
	asyncObjID = ioLibrary->createAsynchronousTaskWithThread(ComplexBackgroundTask, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}
/**********************************************************/
union mnumber
{
	mint i;
	mreal r;
	mcomplex c;
};

typedef struct MTensorBackgroundArgs_st
{
	WolframLibraryData libData;
	int pausemillis;
	union mnumber rangemin;
	union mnumber rangemax;
	mint eltType;
	mint rank;
	mint* dims;
	mint flatlength;
}* MTensorBackgroundArgs;

static void MTensorBackgroundTask(mint asyncObjID, void* vtarg)
{
	MTensorBackgroundArgs targ = (MTensorBackgroundArgs)vtarg;
	WolframLibraryData libData = targ->libData;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	WolframCompileLibrary_Functions funStructCompile = libData->compileLibraryFunctions;
	LibraryFunctionPointer randomInteger;
	LibraryFunctionPointer randomReal;
	LibraryFunctionPointer randomComplex;
	MArgument FPA[3];
	
	mint pausems = targ->pausemillis;
	mint elttype = targ->eltType;
	mint rank = targ->rank;
	mint flatlength = targ->flatlength;
	mint* dims = targ->dims;
	union mnumber rangemin = targ->rangemin;
	union mnumber rangemax = targ->rangemax;
	mint i;
	DataStore ds;
	MTensor eventdata = NULL;
	mint* intdata;
	mreal* realdata;
	mcomplex* complexdata;
	
	free(targ);

	switch(elttype)
	{
	case MType_Integer: 
		randomInteger = funStructCompile->getFunctionCallPointer("RandomInteger");
		break;
	case MType_Real: 
		randomReal = funStructCompile->getFunctionCallPointer("RandomReal");
		break;
	case MType_Complex: 
		randomComplex = funStructCompile->getFunctionCallPointer("RandomComplex");
		break;
	}

	while(ioLibrary->asynchronousTaskAliveQ(asyncObjID))
	{
		/* create event data */
		libData->MTensor_new(elttype, rank, (mint const*)dims, &eventdata);
		switch(elttype)
		{
		case MType_Integer: 
			intdata = libData->MTensor_getIntegerData(eventdata); 
			break;
		case MType_Real: 
			realdata = libData->MTensor_getRealData(eventdata); 
			break;
		case MType_Complex: 
			complexdata = libData->MTensor_getComplexData(eventdata); 
			break;
		}

		for(i=0; i < flatlength; i++)
		{
			switch(elttype)
			{
				case MType_Integer:
					MArgument_getIntegerAddress(FPA[0]) = &rangemin.i;
					MArgument_getIntegerAddress(FPA[1]) = &rangemax.i;
					MArgument_getIntegerAddress(FPA[2]) = &intdata[i];
					randomInteger(libData, 2, FPA, FPA[2]);
					break; 
				case MType_Real: 
					MArgument_getRealAddress(FPA[0]) = &rangemin.r;
					MArgument_getRealAddress(FPA[1]) = &rangemax.r;
					MArgument_getRealAddress(FPA[2]) = &realdata[i];
					randomReal(libData, 2, FPA, FPA[2]);
					break;
				case MType_Complex:
					MArgument_getComplexAddress(FPA[0]) = &rangemin.c;
					MArgument_getComplexAddress(FPA[1]) = &rangemax.c;
					MArgument_getComplexAddress(FPA[2]) = &complexdata[i];
					randomComplex(libData, 2, FPA, FPA[2]);
					break;
			}
		}

		ds = ioLibrary->createDataStore();
		ioLibrary->DataStore_addMTensor(ds, eventdata);
		ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);

		PortableSleep(pausems);
	}

	free(dims);
}

DLLEXPORT int start_mtensor_repeating_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	MTensorBackgroundArgs threadArg = (MTensorBackgroundArgs)malloc(sizeof(struct MTensorBackgroundArgs_st));
	MTensor tdims;
	mint const* srcdims;
	mint i, dim;
	if(Argc != 5)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->libData = libData;
	threadArg->pausemillis = MArgument_getInteger(Args[0]);
	threadArg->eltType = MArgument_getInteger(Args[1]);
	switch(threadArg->eltType)
	{
	case MType_Integer:
		threadArg->rangemin.i = MArgument_getInteger(Args[2]);
		threadArg->rangemax.i = MArgument_getInteger(Args[3]);
		break;
	case MType_Real:
		threadArg->rangemin.r = MArgument_getReal(Args[2]);
		threadArg->rangemax.r = MArgument_getReal(Args[3]);
		break;
	case MType_Complex:
		threadArg->rangemin.c = MArgument_getComplex(Args[2]);
		threadArg->rangemax.c = MArgument_getComplex(Args[3]);
		break;
	}
	tdims = MArgument_getMTensor(Args[4]);
	if(libData->MTensor_getRank(tdims) != 1)
	{
		free(threadArg);
		MArgument_setInteger(Res, -1);
		return LIBRARY_FUNCTION_ERROR;
	}
	threadArg->rank = libData->MTensor_getFlattenedLength(tdims);
	threadArg->dims = (mint*)malloc(threadArg->rank*sizeof(mint));
	threadArg->flatlength = threadArg->rank > 0? 1 : 0;
	for(i=0; i < threadArg->rank; i++)
	{
		dim = libData->MTensor_getIntegerData(tdims)[i];
		threadArg->dims[i] = dim;
		threadArg->flatlength *= dim;
	}

	asyncObjID = ioLibrary->createAsynchronousTaskWithThread(MTensorBackgroundTask, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}
