#include <stdlib.h>
#include <time.h>
#include "mathlink.h"
#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
/**********************************************************/
typedef struct IntBackgroundArgs_st
{
	WolframIOLibrary_Functions ioLibrary;
	mint limit;
	mint eventdata;
}* IntBackgroundArgs;

static void IntBackgroundTask(mint asyncObjID, void* vtarg)
{
	IntBackgroundArgs targ = (IntBackgroundArgs)vtarg;
	mint i;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint limit = targ->limit;
	mint eventdata = targ->eventdata;
	MLINK lp;
	DataStore ds;
	free(targ);

	for(i=0; i < limit; i++)
	{
		ds = ioLibrary->createDataStore();
		ioLibrary->DataStore_addInteger(ds, eventdata);
		ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
	}
	ds = ioLibrary->createDataStore();
	ioLibrary->DataStore_addInteger(ds, limit);
	ioLibrary->raiseAsyncEvent(asyncObjID, "done", ds);
}

DLLEXPORT int start_int_timing_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	IntBackgroundArgs threadArg = (IntBackgroundArgs)malloc(sizeof(struct IntBackgroundArgs_st));
	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->ioLibrary = ioLibrary;
	threadArg->limit = MArgument_getInteger(Args[0]);
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
	mint limit;
	mreal eventdata;
}* RealBackgroundArgs;

static void RealBackgroundTask(mint asyncObjID, void* vtarg)
{
	RealBackgroundArgs targ = (RealBackgroundArgs)vtarg;
	WolframIOLibrary_Functions ioLibrary = targ->ioLibrary;
	mint i;
	mint limit = targ->limit;
	mreal eventdata = targ->eventdata;
	DataStore ds;
	free(targ);

	for(i=0; i < limit; i++)
	{
		ds = ioLibrary->createDataStore();
		ioLibrary->DataStore_addReal(ds, eventdata);
		ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
	}
	ds = ioLibrary->createDataStore();
	ioLibrary->DataStore_addInteger(ds, limit);
	ioLibrary->raiseAsyncEvent(asyncObjID, "done", ds);
}

DLLEXPORT int start_real_timing_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	RealBackgroundArgs threadArg = (RealBackgroundArgs)malloc(sizeof(struct RealBackgroundArgs_st));
	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	threadArg->ioLibrary = ioLibrary;
	threadArg->limit = MArgument_getInteger(Args[0]);
	threadArg->eventdata = MArgument_getReal(Args[1]);
	asyncObjID = ioLibrary->createAsynchronousTaskWithThread(RealBackgroundTask, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}
