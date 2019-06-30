#ifndef SOCKETINTERFACE_COMMON_H
#define SOCKETINTERFACE_COMMON_H

#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "SocketInterface.h"

int s_lastError;

extern const char* SocketHandlerName;

#define GetStreamData(strm, errcode) \
  (strm==0)? 0 : (SocketHandlerData)((strm)->MSdata); \
  if(!(strm) || !(strm)->MSdata) \
     return (errcode);

#define GetStreamDataWithSocket(strm) GetStreamData(strm, 1); if(!data->m_socket) return 1;

typedef struct st_SocketHandlerData
{
  int m_socket;
  int m_errno;
  char* m_errText;
  mbool m_eof;
  int_64 m_streamPosition;
} *SocketHandlerData;

/****************************************************************************/
/* MInputStream functions */

int SocketHandler_fclose(MInputStream strm);

mint SocketHandler_fread(MInputStream strm, char* buffer, size_t count);

char* SocketHandler_ferrorText(MInputStream strm);

void SocketHandler_clearerror(MInputStream strm);

int SocketHandler_feof(MInputStream strm);

void SocketHandlerConstructor(MInputStream strm, const char* msgHead, 
	void* options);

/****************************************************************************/
/* MOutputStream functions */

mint SocketOutputHandler_fwrite(MOutputStream strm, char* buffer, size_t count);

int SocketOutputHandler_fclose(MOutputStream strm);

int_64 SocketOutputHandler_ftell(MOutputStream strm);

char* SocketOutputHandler_ferrorText(MOutputStream strm);

void SocketOutputHandler_clearerror(MOutputStream strm);

void SocketOutputHandlerConstructor(MOutputStream strm, const char* msgHead,
	void* options);

#endif
