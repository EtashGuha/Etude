/*************************************************************************
Mathematica source file

Copyright 1986 through 2015 by Wolfram Research Inc.

This material contains trade secrets and may be registered with the
U.S. Copyright Office as an unpublished work, pursuant to Title 17,
U.S. Code, Section 408.  Unauthorized copying, adaptation, distribution
or display is prohibited.

$Id$

*************************************************************************/

#ifndef WOLFRAMIOLIBRARY_H
#define WOLFRAMIOLIBRARY_H

#include "WolframLibrary.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DataStoreNode_t *DataStoreNode;

#define MType_Undef        0
#define MType_Boolean      1
#define MType_Integer      2
#define MType_Real         3
#define MType_Complex      4
#define MType_Tensor       5
#define MType_SparseArray  6 
#define MType_NumericArray 7
#define MType_Image        8
#define MType_UTF8String   9
#define MType_DataStore    10 

#define MArgument_getDataStoreAddress(marg)		((DataStore*) ((marg).tensor))
#define MArgument_getDataStore(marg)			(*MArgument_getDataStoreAddress(marg))
#define MArgument_setDataStore(marg, v)		    ((*MArgument_getDataStoreAddress(marg)) = (v))

typedef struct st_WolframIOLibrary_Functions
{
    mint (*createAsynchronousTaskWithoutThread)();
    
    mint (*createAsynchronousTaskWithThread)(
                                             void (*asyncRunner)(mint asyncTaskID, void* initData),
                                             void* initData
                                             );

	void (*raiseAsyncEvent)(mint asyncTaskID, char* eventType, DataStore);
    
    mbool (*asynchronousTaskAliveQ)(mint asyncTaskID);
    mbool (*asynchronousTaskStartedQ)(mint asyncTaskID);
    
    DataStore (*createDataStore)(void);

	void(*DataStore_addInteger)(DataStore, mint);
	void(*DataStore_addReal)(DataStore, mreal);
	void(*DataStore_addComplex)(DataStore, mcomplex);
	void(*DataStore_addString)(DataStore, char*);
	void(*DataStore_addMTensor)(DataStore, MTensor);
	void(*DataStore_addMRawArray)(DataStore, MRawArray);
	void(*DataStore_addMImage)(DataStore, MImage);
	void(*DataStore_addDataStore)(DataStore, DataStore);

	void(*DataStore_addNamedInteger)(DataStore, char*, mint);
	void(*DataStore_addNamedReal)(DataStore, char*, mreal);
	void(*DataStore_addNamedComplex)(DataStore, char*, mcomplex);
	void(*DataStore_addNamedString)(DataStore, char*, char*);
	void(*DataStore_addNamedMTensor)(DataStore, char*, MTensor);
	void(*DataStore_addNamedMRawArray)(DataStore, char*, MRawArray);
	void(*DataStore_addNamedMImage)(DataStore, char*, MImage);
	void(*DataStore_addNamedDataStore)(DataStore, char*, DataStore);
	mint(*removeAsynchronousTask)(mint asyncTaskID);

	void(*deleteDataStore)(DataStore);
	DataStore(*copyDataStore)(DataStore);
	mint (*DataStore_getLength)(DataStore);
	DataStoreNode (*DataStore_getFirstNode)(DataStore);
	DataStoreNode (*DataStore_getLastNode)(DataStore);
	DataStoreNode (*DataStoreNode_getNextNode)(DataStoreNode);
	type_t (*DataStoreNode_getDataType)(DataStoreNode);
	errcode_t (*DataStoreNode_getData)(DataStoreNode, MArgument*);
	errcode_t (*DataStoreNode_getName)(DataStoreNode, char**);
	void(*DataStore_addBoolean)(DataStore, mbool);
	void(*DataStore_addNamedBoolean)(DataStore, char*, mbool);
	void(*DataStore_addMNumericArray)(DataStore, MNumericArray);
	void(*DataStore_addNamedMNumericArray)(DataStore, char*, MNumericArray);
	void(*DataStore_addMSparseArray)(DataStore, MSparseArray);
	void(*DataStore_addNamedMSparseArray)(DataStore, char*, MSparseArray);

} *WolframIOLibrary_Functions;
#ifdef __cplusplus
}
#endif

#endif
