/*
 An example that demonstrates using Shared memory management for 
 communicating between Mathematica and a Wolfram Library.
*/

#include "string.h"
#include "WolframLibrary.h"

DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}


DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
	return 0;
}

static char *string = 0;

DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) 
{
	if (string)
		libData->UTF8String_disown(string);
}

DLLEXPORT int countSubstring(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	char *instring = MArgument_getUTF8String(Args[0]);
	char *substring = MArgument_getUTF8String(Args[1]);
	mint i, n = strlen(instring);
	mint slen = strlen(substring);
	mint c = 0;

	if (n > slen) {
		n -= slen;
		for (i = 0; i <= n; i++) {
			if (!strncmp(instring + i, substring, slen)) {
				c++;
			}
		}
	}

	MArgument_setInteger(Res, c);

	libData->UTF8String_disown(instring);
	libData->UTF8String_disown(substring);
	
	return 0;
}

DLLEXPORT int encodeString(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	mint i = 0, shift;
	
	if (string) 
		libData->UTF8String_disown(string);

	string = MArgument_getUTF8String(Args[0]);
	shift = MArgument_getInteger(Args[1]);

	/* Find shift mod 127 so we only 
	   deal with positive numbers below */
	shift = shift % 127;
	if (shift < 0) 
		shift += 127;

	shift -= 1; 
		
	while (string[i]) {
		mint c = (mint) string[i];
		/* Error for non ASCII string */
		if (c & 128) return LIBRARY_FUNCTION_ERROR;
		c = ((c + shift) % 127) + 1;
		string[i++] = (char) c;
	}
	MArgument_setUTF8String(Res, string);
	return 0;
}

DLLEXPORT int reverseString(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	mint i = 0, len = 0, n;
	
	if (string) 
		libData->UTF8String_disown(string);

	string = MArgument_getUTF8String(Args[0]);

	while (string[len]) {
		/* Error for non ASCII string */
		if (string[len] & 128) return LIBRARY_FUNCTION_ERROR;
		len++;
	}

	n = len/2;
	len--; /* For index origin 0 */
	for (i = 0; i < n; i++) {
		char ci = string[i];
		string[i] = string[len - i];
		string[len - i] = ci;
	}

	MArgument_setUTF8String(Res, string);
	return 0;
}




