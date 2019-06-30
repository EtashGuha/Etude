#ifndef SOCKETINTERFACE_H
#define SOCKETINTERFACE_H

#include "WolframLibrary.h"
#include "WolframIOLibraryFunctions.h"
enum SocketStyle { TCPIP, UDP };

/* Initialize socket system, if any.  Returns true on success. */
mbool InitializeSockets(WolframLibraryData libData);

void UninitializeSockets(WolframLibraryData libData);

int NewSocket(enum SocketStyle);

int CloseSocket(int sock);

int ConnectSocket(int sock, const char* hostname, int port);

int SocketBind(int sock, int port);

int SocketListen(int sock, int backlogMax);

int SocketAccept(int serverSocket);

int LastError(void);

typedef struct SocketServerData_st
{
	int serverSocket;
	WolframIOLibrary_Functions ioLibraryFunctions;
} *SocketServerData;

void SocketServer(mint asyncObjID, void* threadArg);

extern WolframLibraryData s_libData;

extern WolframIOLibrary_Functions s_ioFunctions;

#endif /* SOCKETINTERFACE_H */
