/* To run this program use the command-line below:
 *	Unix:           factor3 -linkname "math -mathlink"
 *	Mac or Windows: factor3 -linkmode launch
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mathlink.h"

static void init_and_openlink P(( int argc, char* argv[]));
static void error P(( MLINK lp));
static int   read_and_print_expression P(( MLINK lp));
static int   read_and_print_atom P(( MLINK lp, int tag));
static int   read_and_print_function P(( MLINK lp));


MLENV ep = (MLENV)0;
MLINK lp = (MLINK)0;


int main( int argc, char* argv[])
{
	char buf[BUFSIZ];
	int pkt;

	init_and_openlink( argc, argv);

	printf( "Integer to factor: ");

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
	scanf_s( "%s", buf, BUFSIZ);
#else
	scanf( "%s", buf);
#endif

	/* Send EvaluatePacket[ FactorInteger[n]]. */
	MLPutFunction( lp, "EvaluatePacket", 1L);
		MLPutFunction( lp, "FactorInteger", 1L);
			MLPutNext( lp, MLTKINT);
			MLPutByteString( lp, (unsigned char *)buf, (long)strlen(buf));
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


static int read_and_print_expression( MLINK lp)
{
	int tag;

	switch (tag = MLGetNext( lp)) {
	case MLTKSYM:
	case MLTKSTR:
	case MLTKINT:
	case MLTKREAL:
		return read_and_print_atom( lp, tag);
	case MLTKFUNC:
		return (read_and_print_function( lp));
	case MLTKERROR:
	default:
		return 0;
	}
}


static int   read_and_print_atom( MLINK lp, int tag)
{
	const char *s;
	if( tag == MLTKSTR) putchar( '"');
	if( MLGetString( lp, &s)){
		printf( "%s", s);
#if MLINTERFACE >= 4
		MLReleaseString( lp, s);
#else
		MLDisownString( lp, s);
#endif
	}
	if( tag == MLTKSTR) putchar( '"');
	putchar( ' ');
	return MLError( lp) == MLEOK;
}


static int read_and_print_function( MLINK lp)
{
	int  len, i;
	static int indent;

	if( ! MLGetArgCount( lp, &len)) return 0;

	indent += 3;
	printf( "\n%*.*s", indent, indent, "");

	if( read_and_print_expression( lp) == 0) return 0;
	printf( "[");

	for( i = 1; i <= len; ++i) {
		if( read_and_print_expression( lp) == 0) return 0;
		if( i < len) printf( ", ");
	}
	printf( "]");
	indent -= 3;

	return 1;
}


static void error( MLINK lp)
{
	if (MLError( lp)) {
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
