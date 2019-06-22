/* To launch this program from within Mathematica use:
 *   In[1]:= link = Install["bitops"]
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

int  bitAnd( int x, int y);

void complements( int *px, long nx);
int  bitAnd( int x, int y)
{
	return( x & y);
}

void complements( int *px, long nx)
{
	long    i;
	for( i = 0; i < nx; i++)
		px[i] = ~ px[i] ;
	WSPutIntegerList( stdlink, px, nx);
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

int main(int argc, char* argv[])
{
	return WSMain(argc, argv);
}

#endif
