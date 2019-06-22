#include <stdlib.h>
#include <ctype.h> /* isdigit */
#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "SocketInterface.h"
#include "SocketInterface-Common.h"

const char* SocketHandlerName = "Sockets";

int s_serverSocket = 0;
int s_clientSocket = 0;
WolframLibraryData s_libData = NULL;
WolframIOLibrary_Functions s_ioFunctions = NULL;

int LastError()
{
	return s_lastError;
}

void setIOLibraryFunctions(WolframIOLibrary_Functions);

WolframIOLibrary_Functions getIOLibraryFunctions(void);


int SocketHandler_feof(MInputStream strm)
{
  int retcode;
  SocketHandlerData data = GetStreamData(strm, 0);
  retcode = data->m_eof;
  return retcode;
}

SocketHandlerData New_SocketHandlerData(void)
{
	SocketHandlerData data = (SocketHandlerData)malloc(sizeof(*data));
	if(data == NULL)
		return NULL;

	data->m_socket = -1;
	data->m_eof = False;
	data->m_errno = 0;
	data->m_errText = NULL;
	data->m_streamPosition = 0;

	return data;
}

int parsePortNumber(char* name)
{
	int retval;
	char* p = name;
	while(*p != '\0' && !isdigit(*p))
		p++;
	if(*p == '\0')
		return -1;
	retval = atoi(p);
	return retval;
}

void SocketHandlerConstructor(MInputStream strm, const char* msgHead, 
	void* options)
{
	SocketHandlerData data  = New_SocketHandlerData();

	if(data == NULL)
	{
		strm->hasError = True;
		return;
	}

	strm->isClosed = False;
	strm->MSdata = data;

	data->m_socket = parsePortNumber(strm->name);

	strm->Mfclose = SocketHandler_fclose;
	strm->Mfread = SocketHandler_fread;
	strm->Mfeof = SocketHandler_feof;
	strm->MferrorText = SocketHandler_ferrorText;
	strm->Mclearerr = SocketHandler_clearerror;
}

/*****************************************************************************/

void SocketOutputHandlerConstructor(MOutputStream strm,
	const char* msgHead, void* options)
{
	SocketHandlerData data  = New_SocketHandlerData();

	if(data == NULL)
	{
		strm->hasError = True;
		return;
	}

	strm->isClosed = False;
	strm->MSdata = data;

	data->m_socket = parsePortNumber(strm->name);

	strm->Mfwrite = SocketOutputHandler_fwrite;
	strm->Mftell = SocketOutputHandler_ftell;
	strm->MferrorText = SocketOutputHandler_ferrorText;
	strm->Mclearerr = SocketOutputHandler_clearerror;
	strm->Mfclose = SocketOutputHandler_fclose;
}
