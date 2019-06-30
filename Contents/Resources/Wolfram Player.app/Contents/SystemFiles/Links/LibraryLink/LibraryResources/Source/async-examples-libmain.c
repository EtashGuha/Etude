#include <stdlib.h>
#include "mathlink.h"
#include "WolframLibrary.h"
#include "WolframStreamsLibrary.h"
#include "async-examples.h"

#ifdef _WIN32
#include <windows.h>
void PortableSleep(int timems)
{
	Sleep(timems);
}
#else
#include <sys/time.h>
#include <sys/errno.h>
#include <time.h> 
#include <stdlib.h>
void PortableSleep(int timems)
{
	struct timespec ts;
	ts.tv_sec = timems/1000;
	ts.tv_nsec = (timems % 1000) * 1000000;
	nanosleep(&ts, NULL);
}
#endif

WolframLibraryData s_libData = 0;

/**********************************************************/
DLLEXPORT mint WolframLibrary_getVersion()
{
	return WolframLibraryVersion;
}

DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData)
{
	s_libData = libData;
	return 0;
}
