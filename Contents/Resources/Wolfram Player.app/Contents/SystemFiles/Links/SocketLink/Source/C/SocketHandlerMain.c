#include <stdlib.h>
#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "SocketInterface.h"

DLLEXPORT int create_tcpip_socket(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint fd;
	fd = NewSocket(TCPIP);
	MArgument_setInteger(Res, fd);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int create_udp_socket(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint fd;
	fd = NewSocket(UDP);
	MArgument_setInteger(Res, fd);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int close_socket(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint result;
	mint fd = MArgument_getInteger(Args[0]);
	result = CloseSocket(fd);
	MArgument_setBoolean(Res, result == 0);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int connect_socket(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint result, sock, port;
	char* hostname;

	if(Argc != 3)
		return LIBRARY_FUNCTION_ERROR;

	sock = MArgument_getInteger(Args[0]);
	hostname = MArgument_getUTF8String(Args[1]);
	port = MArgument_getInteger(Args[2]);
	result = ConnectSocket(sock, hostname, port);
	MArgument_setInteger(Res, result);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int socket_bind(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint result, sock, port;

	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;

	sock = MArgument_getInteger(Args[0]);
	port = MArgument_getInteger(Args[1]);
	result = SocketBind(sock, port);
	MArgument_setInteger(Res, result);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int socket_listen(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint result, sock, backlogMax;

	if(Argc != 2)
		return LIBRARY_FUNCTION_ERROR;

	sock = MArgument_getInteger(Args[0]);
	backlogMax = MArgument_getInteger(Args[1]);
	result = SocketListen(sock, backlogMax);
	MArgument_setInteger(Res, result);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int socket_accept(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint clientSocket, serverSocket;

	if(Argc != 1)
		return LIBRARY_FUNCTION_ERROR;
	serverSocket = MArgument_getInteger(Args[0]);

	clientSocket = SocketAccept(serverSocket);

	MArgument_setInteger(Res, clientSocket);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int start_server_thread(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint asyncObjID;
	WolframIOLibrary_Functions ioLibraryFunctions = libData->ioLibraryFunctions;
	SocketServerData threadArg;
	if(Argc != 1)
		return LIBRARY_FUNCTION_ERROR;
	threadArg = malloc(sizeof(*threadArg));
	threadArg->serverSocket = MArgument_getInteger(Args[0]);
	threadArg->ioLibraryFunctions = ioLibraryFunctions;
	asyncObjID = ioLibraryFunctions->createAsynchronousTaskWithThread(
		SocketServer, threadArg);
	MArgument_setInteger(Res, asyncObjID);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int get_last_error(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res) 
{
	mint result = LastError();
	MArgument_setBoolean(Res, result);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int is_protected_mode(WolframLibraryData libData,
	mint Argc, MArgument *Args, MArgument Res)
{
	MArgument_setBoolean(Res, libData->protectedModeQ());
	return LIBRARY_NO_ERROR;
}

DLLEXPORT mint WolframLibrary_getVersion()
{
	return WolframLibraryVersion;
}

DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData)
{
	InitializeSockets(libData);
	return 0;
}

DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData)
{
	UninitializeSockets(libData);
	return;
}

