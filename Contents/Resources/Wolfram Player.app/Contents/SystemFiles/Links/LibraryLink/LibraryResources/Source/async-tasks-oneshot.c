#include <stdlib.h>
#include "mathlink.h"
#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "async-examples.h"

#define TYPEDEF_THREAD_ARGS(prefix, EVENT_DATA_TYPE) \
typedef struct prefix##BackgroundArgs_st \
{ \
	WolframIOLibrary_Functions ioLibrary; \
	int pausemillis; \
	EVENT_DATA_TYPE eventdata; \
}* prefix##BackgroundArgs

/**********************************************************/
TYPEDEF_THREAD_ARGS(Int, mint);

static void IntBackgroundTask(mint asyncObjID, void* vtarg)
{
	IntBackgroundArgs targ = (IntBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	mint eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	PortableSleep(pausems);
	ds = ioLibrary->createDataStore();
	ioLibrary->DataStore_addInteger(ds, eventdata);
	ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
}

DLLEXPORT int start_int_background_task(WolframLibraryData libData,
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
TYPEDEF_THREAD_ARGS(Real, mreal);

static void RealBackgroundTask(mint asyncObjID, void* vtarg)
{
	RealBackgroundArgs targ = (RealBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	mreal eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	PortableSleep(pausems);
	ds = ioLibrary->createDataStore();
	ioLibrary->DataStore_addReal(ds, eventdata);
	ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
}

DLLEXPORT int start_real_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	RealBackgroundArgs threadArg = (RealBackgroundArgs)malloc(sizeof *threadArg);
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
TYPEDEF_THREAD_ARGS(Complex, mcomplex);

static void ComplexBackgroundTask(mint asyncObjID, void* vtarg)
{
	ComplexBackgroundArgs targ = (ComplexBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	mcomplex eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	PortableSleep(pausems);
	ds = ioLibrary->createDataStore();
	ioLibrary->DataStore_addComplex(ds, eventdata);
	ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
}

DLLEXPORT int start_complex_background_task(WolframLibraryData libData,
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
TYPEDEF_THREAD_ARGS(MTensor, MTensor);

static void MTensorBackgroundTask(mint asyncObjID, void* vtarg)
{
	MTensorBackgroundArgs targ = (MTensorBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint pausems = targ->pausemillis;
	MTensor eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	PortableSleep(pausems);
	ds = ioLibrary->createDataStore();
	ioLibrary->DataStore_addMTensor(ds, eventdata);
	ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
}

DLLEXPORT int start_mtensor_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	MTensorBackgroundArgs threadArg = (MTensorBackgroundArgs)malloc(sizeof(struct MTensorBackgroundArgs_st));
	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->ioLibrary = ioLibrary;
	threadArg->pausemillis = MArgument_getInteger(Args[0]);
	threadArg->eventdata = MArgument_getMTensor(Args[1]);
	asyncObjID = ioLibrary->createAsynchronousTaskWithThread(MTensorBackgroundTask, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}
