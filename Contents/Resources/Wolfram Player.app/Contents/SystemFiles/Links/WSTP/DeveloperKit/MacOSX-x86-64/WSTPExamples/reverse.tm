/* To launch this program from within Mathematica use:
 *   In[1]:= link = Install["reverse"]
 *
 * Or, launch this program from a shell and establish a
 * peer-to-peer connection.  When given the prompt Create Link:
 * type a port name. ( On Unix platforms, a port name is a
 * number less than 65536.  On Mac or Windows platforms,
 * it's an arbitrary word.)
 * Then, from within Mathematica use:
 *   In[1]:= link = Install["portname", LinkMode->Connect]
 */

#include "wstp.h"

extern int WSMain(int, char **);
#ifdef WINDOWS_WSTP
extern HWND WSInitializeIcon( HINSTANCE hinstCurrent, int nCmdShow);
#endif

void reverse_string(const unsigned short *s, int len);

:Begin:
:Function:      reverse_string
:Pattern:       reverseString[s_String]
:Arguments:     {s}
:ArgumentTypes: {UCS2String}
:ReturnType:    Manual
:End:



#define BUFCAP 64

void reverse_string(const unsigned short *s, int len)
{
	unsigned short buf[BUFCAP], *p = buf;
	long n = BUFCAP;
	
	WSPutNext( stdlink, WSTKSTR);
	while( len > 0){
		if( n-- == 0){
			WSPutUCS2Characters( stdlink, len, buf, (int)(p - buf));
			n = BUFCAP - 1;
			p = buf;
		}
		*p++ = s[--len];
	}
	WSPutUCS2Characters( stdlink, len, buf, (int)(p - buf));
}



#if WINDOWS_WSTP

int PASCAL WinMain( HINSTANCE hinstCurrent, HINSTANCE hinstPrevious, LPSTR lpszCmdLine, int nCmdShow)
{
	char  buff[512];
	char FAR * buff_start = buff;
	char FAR * argv[32];
	char FAR * FAR * argv_end = argv + 32;

    hinstPrevious = hinstPrevious; /* suppress warning */

	if( !WSInitializeIcon( hinstCurrent, nCmdShow)) return 1;
	WSScanString( argv, &argv_end, &lpszCmdLine, &buff_start);
	return WSMain( (int)(argv_end - argv), argv);
}

#else

int main(int argc, char **argv)
{
	return WSMain(argc, argv);
}

#endif
