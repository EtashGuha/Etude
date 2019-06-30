#include <stdlib.h>
#include "WolframLibrary.h"
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

mint s_secondaryAsyncObjID = -1;
mint s_secondaryAsyncObjEventData;

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
		if(s_secondaryAsyncObjID < 0)
		{
			ioLibrary->DataStore_addInteger(ds, eventdata);
			ioLibrary->raiseAsyncEvent(asyncObjID, NULL, ds);
			eventdata++;
		}
		else
		{
			ioLibrary->DataStore_addInteger(ds, s_secondaryAsyncObjEventData);
			ioLibrary->raiseAsyncEvent(s_secondaryAsyncObjID, NULL, ds);
		s_secondaryAsyncObjEventData++;
		}
	}
}

DLLEXPORT int start_primary_int_background_task(WolframLibraryData libData,
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

DLLEXPORT int start_secondary_int_background_task(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibrary = libData->ioLibraryFunctions;
	if(Argc != 1)
		return LIBRARY_FUNCTION_ERROR;
	s_secondaryAsyncObjEventData = MArgument_getInteger(Args[1]);
	asyncObjID = ioLibrary->createAsynchronousTaskWithoutThread();
	s_secondaryAsyncObjID = asyncObjID;
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}
