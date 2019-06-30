/*
 An example that demonstrates calling back to Mathematica inside a Wolfram Library function.
*/

#include "mathlink.h"
#include "WolframLibrary.h"


DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}


DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
	return 0;
}

DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData) {
	return;
}


DLLEXPORT int function1(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
	mint I0, I1,  res;
	MLINK link;
	int pkt, type;

	I0 = MArgument_getInteger(Args[0]);
	I1 = MArgument_getInteger(Args[1]);
	
	link = libData->getWSLINK(libData);
	MLPutFunction( link, "EvaluatePacket", 1);
	MLPutFunction( link, "Message", 2);
	MLPutFunction( link, "MessageName", 2);
	MLPutSymbol( link, "MyFunction");
	MLPutString( link, "info");
	MLPutString( link, "Message called from within Library function.");
	libData->processWSLINK( link);
	pkt = MLNextPacket( link);
	if ( pkt == RETURNPKT) {
		MLNewPacket(link);
	}
	
	res = I0+I1;
	MArgument_setInteger(Res, res);
	return 0;
}

