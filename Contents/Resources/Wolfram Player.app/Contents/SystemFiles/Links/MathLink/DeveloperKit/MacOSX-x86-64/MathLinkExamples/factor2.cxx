/* To run this program use the command-line below:
 *	Unix:           factor2 -linkname "math -mathlink"
 *	Mac or Windows: factor2 -linkmode launch
 */

#include <stdio.h>
#include <stdlib.h>
#include "mathlink.h"

static void init_and_openlink P((int argc, char* argv[]));
static void error P(( MLINK lp));
static void read_and_print_expression P(( MLINK lp));


MLENV ep = (MLENV)0;
MLINK lp = (MLINK)0;


int main( int argc, char* argv[])
{
	int n, pkt;

	init_and_openlink( argc, argv);

	printf( "Integer to factor: ");

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
	scanf_s( "%d", &n);
#else
	scanf( "%d", &n);
#endif
	/* Send EvaluatePacket[ FactorInteger[n]]. */
	MLPutFunction( lp, "EvaluatePacket", 1L);
		MLPutFunction( lp, "FactorInteger", 1L);
			MLPutInteger( lp, n);
	MLEndPacket( lp);


	/* skip any packets before the first ReturnPacket */
	while( (pkt = MLNextPacket( lp), pkt) && pkt != RETURNPKT) {
		MLNewPacket( lp);
		if( MLError( lp)) error( lp);
	}

	read_and_print_expression( lp);
	printf( "\n");

	MLPutFunction( lp, "Exit", 0L);

	return 0;
}


static void read_and_print_expression( MLINK lp)
{
	const char *s;
	int n;
	int i, len;
	double r;
	static int indent;

	switch( MLGetNext( lp)) {
	case MLTKSYM:
		MLGetSymbol( lp, &s);
		printf( "%s ", s);
#if MLINTERFACE >= 4
		MLReleaseSymbol( lp, s);
#else
		MLDisownSymbol( lp, s);
#endif
		break;
	case MLTKSTR:
		MLGetString( lp, &s);
		printf( "\"%s\" ", s);
#if MLINTERFACE >= 4
		MLReleaseString( lp, s);
#else
		MLDisownString( lp, s);
#endif
		break;
	case MLTKINT:
		MLGetInteger( lp, &n);
		printf( "%d ", n);
		break;
	case MLTKREAL:
		MLGetReal( lp, &r);
		printf( "%g ", r);
		break;
	case MLTKFUNC:
		indent += 3;
		printf( "\n %*.*s", indent, indent, "");
		if( MLGetArgCount( lp, &len) == 0){
			error( lp);
		}else{
			read_and_print_expression( lp);
			printf( "[");
			for( i = 1; i <= len; ++i){
				read_and_print_expression( lp);
				if( i != len) printf( ", ");
			}
			printf( "]");
		}
		indent -= 3;
		break;
	case MLTKERROR:
	default:
		error( lp);
	}
}


static void error( MLINK lp)
{
	if( MLError( lp)) {
		fprintf( stderr, "Error detected by MathLink: %s.\n",
		MLErrorMessage( lp));
	}else{
		fprintf( stderr, "Error detected by this program.\n");
	}
	exit( 1);
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

