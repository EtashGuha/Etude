/* To run this program use the command-line below:
 *	Unix:           factor -linkname "math -mathlink"
 *	Mac or Windows: factor -linkmode launch
 */


#include <stdio.h>
#include <stdlib.h>

#include "mathlink.h"

static void init_and_openlink( int argc, char* argv[]);
static void error( MLINK lp);


MLENV ep = (MLENV)0;
MLINK lp = (MLINK)0;


int main(int argc, char* argv[])
{
	int pkt, n, prime, expt;
#if MLINTERFACE >= 4
	int len, lenp, k;
#else
	long len, lenp, k;
#endif

	init_and_openlink( argc, argv);

	printf( "Integer to factor: ");

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
	scanf_s( "%d", &n);
#else
	scanf( "%d", &n);
#endif

	MLPutFunction( lp, "EvaluatePacket", 1L);
		MLPutFunction( lp, "FactorInteger", 1L);
			MLPutInteger( lp, n);
	MLEndPacket( lp);

	while( (pkt = MLNextPacket( lp), pkt) && pkt != RETURNPKT) {
		MLNewPacket( lp);
		if (MLError( lp)) error( lp);
	}

#if MLINTERFACE >= 4
	if ( ! MLTestHead( lp, "List", &len)) error(lp);
	for (k = 1; k <= len; k++) {
		if (MLTestHead( lp, "List", &lenp)
#else
	if ( ! MLCheckFunction( lp, "List", &len)) error(lp);
	for (k = 1; k <= len; k++) {
		if (MLCheckFunction( lp, "List", &lenp)
#endif
		&&  lenp == 2
		&&  MLGetInteger( lp, &prime)
		&&  MLGetInteger( lp, &expt)
		){
			printf( "%d ^ %d\n", prime, expt);
		}else{
			error( lp);
		}
	}

	MLPutFunction( lp, "Exit", 0);

	return 0;
}


static void error( MLINK lp)
{
	if( MLError( lp)){
		fprintf( stderr, "Error detected by MathLink: %s.\n",
			MLErrorMessage(lp));
	}else{
		fprintf( stderr, "Error detected by this program.\n");
	}
	exit(3);
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
