
#include "WolframLibrary.h"


DLLEXPORT mint WolframLibrary_getVersion( ) {
   return WolframLibraryVersion;
}

DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
   return LIBRARY_NO_ERROR;
}

DLLEXPORT void WolframDLL_uninitialize( ) {
   return;
}

DLLEXPORT int incrementInteger(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
   mint I0;
   mint I1;
   I0 = MArgument_getInteger(Args[0]);
   I1 = I0 + 1;
   MArgument_setInteger(Res, I1);
   return LIBRARY_NO_ERROR;
}


