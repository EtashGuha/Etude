#ifndef UNICODE
#define UNICODE
#endif

#define WIN32_LEAN_AND_MEAN

#include <winsock2.h>
#include <Ws2tcpip.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// Link with ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "SocketInterface.h"
#include "SocketInterface-Common.h"

static WSADATA s_wsaData;

#if 0
void logdebug(char* msg)
{
	FILE* out = fopen("d:/sockets.log", "a");
	fprintf(out, "%s\n", msg==NULL?"NULL":msg);
	fclose(out);
}

void logdebugint(int i)
{
	char buff[16];
	sprintf(buff, "%d", i);
	logdebug(buff);
}
#else
void logdebug(char* msg)
{
}

void logdebugint(int i)
{
}
#endif

void deallocateErrorText(SocketHandlerData data)
{
	if(data->m_errText != NULL)
		LocalFree(data->m_errText);
	data->m_errText = NULL;
}

void setError(SocketHandlerData data, int error)
{
	deallocateErrorText(data);
	data->m_errno = error;
	FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
		NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPSTR)&data->m_errText, 0,	NULL);
	data->m_errText;
}

int NewSocket(enum SocketStyle style)
{
	int sock, result;
	u_long mode;

	if(s_libData->protectedModeQ())
		return INVALID_SOCKET;

	if(style == TCPIP)
		sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
	else
		sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

	/* Make the socket non-blocking */
	mode = 1; /* non-blocking */
	result = ioctlsocket(sock, FIONBIO, &mode);
	if(result != NO_ERROR)
	{
		closesocket(sock);
		sock = INVALID_SOCKET;
	}
	return sock;
}

int CloseSocket(int sock)
{
	return closesocket(sock);
}

/* Much code lifted from http://msdn.microsoft.com/en-us/library/windows/desktop/ms738613%28v=VS.85%29.aspx */

int ConnectSocket(int sock, const char* hostname, int port)
{
	struct sockaddr_in serverAddress;
	struct hostent *hp;
	unsigned int addr;
	int error, selectResult, result;
	fd_set writeset;
	struct timeval timeout;

    if (isalpha(hostname[0]))    	/* server address is a name */
    {
        hp = gethostbyname(hostname);
    }
    else /* Convert nnn.nnn address to a usable one */
    {
        addr = inet_addr(hostname);
        hp = gethostbyaddr((char *)&addr, 4, AF_INET);
    }
    if (hp == 0 )
    {
        s_lastError = WSAGetLastError();
        return -2;
    }

    memset(&serverAddress, 0, sizeof(serverAddress));
    memcpy(&(serverAddress.sin_addr),hp->h_addr, hp->h_length);
    serverAddress.sin_family = hp->h_addrtype;
    serverAddress.sin_port = htons(port);

	result = connect(sock, (struct sockaddr*)&serverAddress, 
		sizeof(serverAddress));
	/* Note: we expect SOCKET_ERROR because socket is non-blocking. */
	if(result == SOCKET_ERROR)
	{
		error = WSAGetLastError();
		if(error != WSAEWOULDBLOCK)
		{
			s_lastError = error;
			return -3;
		}
	}
	/* If result is success, assume it means connection happened already,
	 * although this seems highly unlikely.
	 */

	FD_ZERO(&writeset);
	FD_SET(sock, &writeset);
	timeout.tv_sec = 60; /* TODO pass in timeout as user-configurable parameter */
	timeout.tv_usec = 0;

	selectResult = select(1, NULL, &writeset, NULL, &timeout);
	if(selectResult == 0) /* timeout */
	{
		return -5;
	}
	else if(selectResult == SOCKET_ERROR)
	{
		s_lastError = WSAGetLastError();
		return -6;
	}

	return 0;
}

int SocketBind(int sock, int port)
{
	int result;
	struct sockaddr_in serverAddress;
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_addr.s_addr = INADDR_ANY;
	serverAddress.sin_port = htons(port);
	result = bind(sock, &serverAddress, sizeof(serverAddress));
	if(result < 0)
	{
        s_lastError = WSAGetLastError();
		return -1;
	}
	return 0;
}

int SocketListen(int sock, int backlogMax)
{
	int result = listen(sock, backlogMax);
	if(result < 0)
	{
        s_lastError = WSAGetLastError();
		return -1;
	}
	return 0;
}

int SocketAccept(int serverSocket)
{
	int clientSocket = accept(serverSocket, 0, 0);
	if(clientSocket < 0)
	{
        s_lastError = WSAGetLastError();
		return -1;
	}
	return clientSocket;
}

/*****************************************************************************/
/* SocketHandler */

int SocketHandler_fclose(MInputStream strm)
{
	int retcode;
	SocketHandlerData data = GetStreamDataWithSocket(strm);

	logdebug("fclose");
	deallocateErrorText(data);
	if(strm->isClosed)
		return 0;
	/* we don't close the socket, because writes could still happen. We use
	 * shutdown to signal eof of this direction to the other connection. When
	 * both sides of the socket are shut down, the socket is closed by the OS.
	 */
	retcode = shutdown(data->m_socket, SD_RECEIVE);
	strm->isClosed = True;
	free(strm->MSdata);

  return retcode;
}

mint SocketHandler_fread(MInputStream strm, char* buffer, size_t count)
{
	mint retcode;
	int error;
	SocketHandlerData data = GetStreamDataWithSocket(strm);
	logdebug("fread");
	retcode = recv(data->m_socket, buffer, count, 0);
	if(retcode == 0)
	{
		logdebug("fread got eof");
		data->m_eof = True;
	}
	else if(retcode < 0)
	{
		error = WSAGetLastError();
		if(error == WSAEWOULDBLOCK)
		{
			logdebug("fread would block, returning 0");
			retcode = 0; /* no bytes read, kernel will now call WaitForInput */
		}
		else
		{
			logdebug("fread got error, errno to follow");
			logdebugint(error);
			setError(data, error);
		}
	}
	else
		logdebug("fread got data");
	return retcode;
}

void SocketHandler_WaitForInput(MInputStream strm)
{
	fd_set readset;
	struct timeval timeout;
	int selectResult, waiting;
	SocketHandlerData data = GetStreamDataWithSocket(strm);

	FD_ZERO(&readset);
	FD_SET(data->m_socket, &readset);
	timeout.tv_sec = 0;
	timeout.tv_usec = 40000; /* abort check rate 25 Hz */

	logdebug("waitForInput");
	waiting = True;
	while(waiting)
	{
		selectResult = select(1, &readset, NULL, NULL, &timeout);
		if(selectResult == 0) /* timeout */
		{
			if(s_libData->AbortQ())
			{
				logdebug("waitForInput detected abort");
				return;
			}
		}
		else // Either an error or there is something to read now.
		{
			waiting = False;
			if(selectResult == SOCKET_ERROR)
			{
				logdebug("waitForInput got an error");
				setError(data, WSAGetLastError());
			}
			else
				logdebug("waitForInput has something to read");
		}
	}
}

char* SocketHandler_ferrorText(MInputStream strm)
{
	char* retval;
	SocketHandlerData data =  GetStreamData(strm, 0);
	retval = data->m_errText;
	logdebug("ferrorText, errno then errtext to follow");
	logdebugint(data->m_errno);
	logdebug(retval);
	return retval;
}

void SocketHandler_clearerror(MInputStream strm)
{
	SocketHandlerData data;
	if(strm == NULL || strm->MSdata == NULL)
		return;
	data = (SocketHandlerData)strm->MSdata;
	deallocateErrorText(data);
}

/*****************************************************************************/

mint SocketOutputHandler_fwrite(MOutputStream strm, char* buffer, size_t count)
{
	mint retcode;
	int error;
	SocketHandlerData data = GetStreamDataWithSocket(strm);
	logdebug("fwrite");
	retcode = send(data->m_socket, buffer, count, 0);
	if(retcode == 0)
	{
		logdebug("fwrite wrote zero bytes");
	}
	else if(retcode < 0)
	{
		error = WSAGetLastError();
		if(error == WSAEWOULDBLOCK)
		{
			logdebug("fwrite would block");
			retcode = 0; /* this is normal; buffer is full, so return that we
			 	 didn't write anything. Presumably we will be asked again to
			 	 write the same data. */
		}
		else /* it's really an error */
		{
			logdebug("fwrite got an error, errno and errText to follow");
			logdebugint(error);
			setError(data, error);
			logdebug(data->m_errText);
		}
	}
	else
	{
		logdebug("fwrite wrote OK, #bytes to follow");
		logdebugint(retcode);
		data->m_streamPosition += retcode;
	}
	return retcode;
}

int_64 SocketOutputHandler_ftell(MOutputStream strm)
{
	int_64 retval;
	SocketHandlerData data = GetStreamDataWithSocket(strm);
	logdebug("output ftell");
	retval = data->m_streamPosition;
	return retval;
}

int SocketOutputHandler_fclose(MOutputStream strm)
{
	int retcode;
	SocketHandlerData data = GetStreamDataWithSocket(strm);
	logdebug("output fclose");
	deallocateErrorText(data);
	if(strm->isClosed)
		return 0;
	/* we don't close the socket, because writes could still happen. We use
	 * shutdown to signal eof of this direction to the other connection. When
	 * both sides of the socket are shut down, the socket is closed by the OS.
	 */
	retcode = shutdown(data->m_socket, SD_SEND);
	strm->isClosed = True;
	free(strm->MSdata);

	return retcode;
}

char* SocketOutputHandler_ferrorText(MOutputStream strm)
{
	char* retval;
	SocketHandlerData data = GetStreamData(strm, 0);
	retval = data->m_errText;
	return retval;
}

void SocketOutputHandler_clearerror(MOutputStream strm)
{
	SocketHandlerData data;
	if(strm == NULL || strm->MSdata == NULL)
		return;
	data = (SocketHandlerData)strm->MSdata;
	deallocateErrorText(data);
}

/*****************************************************************************/
/* Server thread body */
void SocketServer(mint asyncObjID, void* pvThreadArg)
{
	SocketServerData threadArgs = (SocketServerData)pvThreadArg;
	mint serverSocket = threadArgs->serverSocket;
	WolframIOLibrary_Functions ioLibraryFunctions = 
		threadArgs->ioLibraryFunctions;
	int clientSocket, error, selectResult;
	DataStore ds;
	fd_set readset;
	struct timeval timeout;

	free(threadArgs);

	timeout.tv_sec = 0;
	timeout.tv_usec = 100000; /* check rate 10 Hz */

	while(ioLibraryFunctions->asynchronousTaskAliveQ(asyncObjID))
	{
		clientSocket = SocketAccept(serverSocket); /* TODO receive address of incoming connection */
		if(clientSocket == INVALID_SOCKET)
		{
			error = WSAGetLastError();
			if(error == WSAEWOULDBLOCK) /* no incoming connections presently */
			{
				/* go into a select call to wait, with a timeout that allows
				 * us to check whether our asynchronous task is still alive. */
				FD_ZERO(&readset);
				FD_SET(serverSocket, &readset);
				selectResult = select(1, &readset, NULL, NULL, &timeout);
				/* Now either there's a connection, a timeout, or an error. */
				if(selectResult == SOCKET_ERROR)
				{
					error = WSAGetLastError();
					/* TODO what do we need to do with a select error? */
					logdebug("server select got error, errno to follow");
					logdebugint(error);
				}
			}
			else
			{
				/* TODO what should we do with an accept error? */
				logdebug("server loop got error, errno to follow");
				logdebugint(error);
			}
		}
		else if(clientSocket >= 0)
		{
			logdebug("server got connection, socket to follow");
			logdebugint(clientSocket);
			ds = ioLibraryFunctions->createDataStore();
			ioLibraryFunctions->DataStore_addInteger(ds, clientSocket);
			ioLibraryFunctions->raiseAsyncEvent(asyncObjID, "", ds);
		}
	}
}

/*****************************************************************************/
mbool InitializeSockets(WolframLibraryData libData)
{
	mbool ok = WSAStartup(MAKEWORD(2, 0), &s_wsaData) == 0;
	libData->registerInputStreamMethod(SocketHandlerName,
		SocketHandlerConstructor, NULL, NULL, NULL);
	libData->registerOutputStreamMethod(SocketHandlerName,
		SocketOutputHandlerConstructor, NULL, NULL, NULL);
	s_libData = libData;
	s_ioFunctions = libData->ioLibraryFunctions;
	return ok;
}

void UninitializeSockets(WolframLibraryData libData)
{
	libData->unregisterInputStreamMethod(SocketHandlerName);
	libData->unregisterOutputStreamMethod(SocketHandlerName);
	WSACleanup();
}
