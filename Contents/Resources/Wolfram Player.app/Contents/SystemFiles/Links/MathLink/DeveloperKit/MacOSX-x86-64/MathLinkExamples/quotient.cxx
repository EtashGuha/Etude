/* To run this program use the command-line below:
 *	Unix:           quotient -linkname "math -mathlink"
 *	Mac or Windows: quotient -linkmode launch
 */

#include <stdio.h>
#include <stdlib.h>
#include "mathlink.h"

static void init_and_openlink( int argc, char* argv[]);



MLENV ep = (MLENV)0;
MLINK lp = (MLINK)0;


int main( int argc, char* argv[])
{
	int pkt, m, n, q;

	init_and_openlink( argc, argv);

	printf( "Two integers m/n: ");

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
	if( scanf_s( "%d/%d", &m, &n) != 2 && scanf_s( "%d %d", &m, &n) != 2)
#else
	if( scanf( "%d/%d", &m, &n) != 2 && scanf( "%d %d", &m, &n) != 2)
#endif
		exit(-1);

	/* Send EvaluatePacket[ Quotient[ m, n]] */
	MLPutFunction( lp, "EvaluatePacket", 1L);
		MLPutFunction( lp, "Quotient", 2L);
			MLPutInteger( lp, m);
			MLPutInteger( lp, n);
	MLEndPacket( lp);
	
	/* skip any packets before the first ReturnPacket */
	while( (pkt = MLNextPacket( lp), pkt) && pkt != RETURNPKT)
		MLNewPacket( lp);
	
	/* inside the ReturnPacket we expect an integer */
	MLGetInteger( lp, &q);
	
	printf( "quotient = %d\n", q);
	
	/* quit Mathematica */
	MLPutFunction( lp, "Exit", 0);

	return 0;
}



static void deinit( void)
{
	if( ep) MLDeinitialize( ep);
}


static void closelink( void)
{
	if( lp) MLClose( lp);
}


static void init_and_openlink( int argc, char* argv[])
{
	int err;

	ep =  MLInitialize( (MLParametersPointer)0);
	if( ep == (MLENV)0) exit(1);
	atexit( deinit);

	lp = MLOpenArgcArgv( ep, argc, argv, &err);
	if(lp == (MLINK)0) exit(2);
	atexit( closelink);
}
