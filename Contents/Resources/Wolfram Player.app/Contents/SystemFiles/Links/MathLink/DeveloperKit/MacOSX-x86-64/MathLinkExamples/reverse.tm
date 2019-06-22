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

#include "mathlink.h"


void reverse_string P((const unsigned short *s, int len));

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
	
	MLPutNext( stdlink, MLTKSTR);
	while( len > 0){
		if( n-- == 0){
			MLPutUCS2Characters( stdlink, len, buf, (int)(p - buf));
			n = BUFCAP - 1;
			p = buf;
		}
		*p++ = s[--len];
	}
	MLPutUCS2Characters( stdlink, len, buf, (int)(p - buf));
}



#if WINDOWS_MATHLINK

#if __BORLANDC__
#pragma argsused
#endif

int PASCAL WinMain( HINSTANCE hinstCurrent, HINSTANCE hinstPrevious, LPSTR lpszCmdLine, int nCmdShow)
{
	char  buff[512];
	char FAR * buff_start = buff;
	char FAR * argv[32];
	char FAR * FAR * argv_end = argv + 32;

    hinstPrevious = hinstPrevious; /* suppress warning */

	if( !MLInitializeIcon( hinstCurrent, nCmdShow)) return 1;
	MLScanString( argv, &argv_end, &lpszCmdLine, &buff_start);
	return MLMain( (int)(argv_end - argv), argv);
}

#else

int main(int argc, char **argv)
{
	return MLMain(argc, argv);
}

#endif
