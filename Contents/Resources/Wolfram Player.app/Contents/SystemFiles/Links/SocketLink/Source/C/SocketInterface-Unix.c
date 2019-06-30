#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <netdb.h> /* gethostbyname */
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h> /* close */
#include <ctype.h> /* isalpha */
#include <fcntl.h>

#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "SocketInterface.h"
#include "SocketInterface-Common.h"

#if 0
void logdebug(char* msg)
{
	FILE* out = fopen("/tmp/sockets.log", "a");
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
		free(data->m_errText);
	data->m_errText = NULL;
}

void setError(SocketHandlerData data, int error)
{
	deallocateErrorText(data);
	data->m_errno = error;
	data->m_errText = strdup(strerror(error));
}

int setNonBlocking(int sock)
{
	int flags, result;
	flags = fcntl(sock, F_GETFL, 0);
	result = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
	return result;
}

int NewSocket(enum SocketStyle style)
{
	int sock, flags, result;

	if(s_libData->protectedModeQ())
		return 0;

	if(style == TCPIP)
		sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
	else
		sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

	result = setNonBlocking(sock);
	if(result < 0)
	{
		close(sock);
		sock = 0;
	}
	return sock;
}

int CloseSocket(int sock)
{
	return close(sock);
}

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
        s_lastError = errno;
        return -2;
    }

    memset(&serverAddress, 0, sizeof(serverAddress));
    memcpy(&(serverAddress.sin_addr),hp->h_addr, hp->h_length);
    serverAddress.sin_family = hp->h_addrtype;
    serverAddress.sin_port = htons(port);

    logdebug("connect, hostname then port to follow");
    logdebug(hostname);
    logdebugint(port);
	result = connect(sock, (struct sockaddr*)&serverAddress,
		sizeof(serverAddress));
	logdebug("connect returned, result to follow");
	logdebugint(result);
	if(result < 0)
	{
		error = errno;
		if(error != EINPROGRESS)
		{
			s_lastError = error;
			logdebug("connect failed, error code to follow");
			logdebugint(error);
			return -3;
		}

		logdebug("connect in progress, blocking in select");
		/* block in select for the connection */
		FD_ZERO(&writeset);
		FD_SET(sock, &writeset);
		timeout.tv_sec = 60; /* TODO pass in timeout as user-configurable parameter */
		timeout.tv_usec = 0;

		selectResult = select(sock+1, NULL, &writeset, NULL, &timeout);
		if(selectResult == 0) /* timeout */
		{
			logdebug("select timed out");
			return -5;
		}
		else if(selectResult < 0)
		{
			s_lastError = errno;
			logdebug("select failed, errno to follow");
			logdebugint(s_lastError);
			return -6;
		}
		/* TODO  use  getsockopt(2) to read the SO_ERROR option at level SOL_SOCKET to
              determine whether connect()  completed  successfully */
	}
	/* else connection is ready immediately */
	logdebug("connect succeeded");

	return 0;
}

int SocketBind(int sock, int port)
{
	int result;
	struct sockaddr_in serverAddress;
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_addr.s_addr = INADDR_ANY;
	serverAddress.sin_port = htons(port);
	result = bind(sock, (struct sockaddr*)&serverAddress,
		sizeof(serverAddress));
	if(result < 0)
	{
        s_lastError = errno;
		return -1;
	}
	return 0;
}

int SocketListen(int sock, int backlogMax)
{
	int result = listen(sock, backlogMax);
	if(result < 0)
	{
        s_lastError = errno;
		return -1;
	}
	return 0;
}

int SocketAccept(int serverSocket)
{
	int clientSocket = accept(serverSocket, 0, 0);
	if(clientSocket < 0)
	{
        s_lastError = errno;
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
	retcode = shutdown(data->m_socket, 0);

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
		error = errno;
		if(error == EWOULDBLOCK)
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
	SocketHandlerData data;

	if(strm == NULL || strm->MSdata == NULL)
		return;
	data = (SocketHandlerData)strm->MSdata;
	if(!data->m_socket)
		return;

	FD_ZERO(&readset);
	FD_SET(data->m_socket, &readset);
	timeout.tv_sec = 0;
	timeout.tv_usec = 40000; /* abort check rate 25 Hz */

	logdebug("waitForInput");
	waiting = True;
	while(waiting)
	{
		selectResult = select(data->m_socket+1, &readset, NULL, NULL, &timeout);
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
			if(selectResult < 0)
			{
				logdebug("waitForInput got an error");
				setError(data, errno);
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
		error = errno;
		if(error == EWOULDBLOCK)
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
	/* we don't close the socket, because reads could still happen. We use
	 * shutdown to signal eof of this direction to the other connection. When
	 * both sides of the socket are shut down, the socket is closed by the OS.
	 */
	retcode = shutdown(data->m_socket, 1);
	strm->isClosed = True;
	free(strm->MSdata);

	return retcode;
}

char* SocketOutputHandler_ferrorText(MOutputStream strm)
{
	char* retval;
	SocketHandlerData data =  GetStreamData(strm, 0);
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
	int clientSocket, error, selectResult, result;
	DataStore ds;
	fd_set readset;
	struct timeval timeout;

	free(threadArgs);

	timeout.tv_sec = 0;
	timeout.tv_usec = 100000; /* check rate 10 Hz */

	while(ioLibraryFunctions->asynchronousTaskAliveQ(asyncObjID))
	{
		FD_ZERO(&readset);
		FD_SET(serverSocket, &readset);

		selectResult = select(serverSocket+1, &readset, NULL, NULL, &timeout);
		if(selectResult < 0)
		{
			/* error */
			error = errno;
			/* TODO what do we need to do with a select error? */
			logdebug("server select got error, errno to follow");
			logdebugint(error);
			break;
		}
		else if(selectResult == 0)
			/* Timeout, skip to the top */
			continue;

		/* assert: selectResult > 0, we might have a connection */
		if(!FD_ISSET(serverSocket, &readset))
		{
			logdebug("select returned, but serverSocket is not set");
			continue;
		}

		clientSocket = SocketAccept(serverSocket); /* TODO receive address of incoming connection */
		if(clientSocket < 0)
		{
			error = errno;
			if(error == EWOULDBLOCK) /* no incoming connections presently */
			{
				/* go back to select */
				logdebug("select indicated ready, but accept says EWOULDBLOCK");
				continue;
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

			result = setNonBlocking(clientSocket);
			if(result < 0)
			{
				error = errno;
				logdebug("attempt to set client socket non-blocking failed, errno to follow");
				logdebugint(error);
				continue;
			}

			ds = ioLibraryFunctions->createDataStore();
			ioLibraryFunctions->DataStore_addInteger(ds, clientSocket);
			ioLibraryFunctions->raiseAsyncEvent(asyncObjID, "", ds);
		}
	}
}

/*****************************************************************************/
mbool InitializeSockets(WolframLibraryData libData)
{
	libData->registerInputStreamMethod(SocketHandlerName,
		SocketHandlerConstructor, NULL, NULL, NULL);
	libData->registerOutputStreamMethod(SocketHandlerName,
		SocketOutputHandlerConstructor, NULL, NULL, NULL);
	s_libData = libData;
	s_ioFunctions = libData->ioLibraryFunctions;
	return True;
}

void UninitializeSockets(WolframLibraryData libData)
{
	libData->unregisterInputStreamMethod(SocketHandlerName);
	libData->unregisterOutputStreamMethod(SocketHandlerName);
}
