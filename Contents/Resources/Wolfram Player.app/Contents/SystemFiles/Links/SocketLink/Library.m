BeginPackage["SocketLink`Library`"]

InitializeLibrary::usage = "InitializeLibrary[] "

$CreateStreamSocket
$CreateDatagramSocket
$GetLastError
$ConnectSocketToServer
$SocketBind
$SocketListen
$SocketAccept
$CloseSocket
$StartServerThread
$ProtectedModeQ

Begin["`Private`"] 

$thisdir = DirectoryName[System`Private`$InputFileName];
$LibraryFilePath = 
	FileNameJoin[{$thisdir, "LibraryResources", $SystemID, "SocketLink"}];

Options[InitializeLibrary] = {"LoadOnlyIfNeeded" -> True}

InitializeLibrary[OptionsPattern[]] := 
	If[OptionValue["LoadOnlyIfNeeded"] === False || $InitializedQ =!= True, 
		(* If we have loaded from the library previously, unload it first *)
		If[TrueQ[$InitializedQ] || SameQ[$LoadSuccessQ, False],
			LibraryUnload[$LibraryFilePath]
		];
		$LoadSuccessQ = True;
		$CreateStreamSocket = tryLibraryFunctionLoad[$LibraryFilePath, "create_tcpip_socket", {}, _Integer];
		$CreateDatagramSocket = tryLibraryFunctionLoad[$LibraryFilePath, "create_udp_socket", {}, _Integer];
		$GetLastError = tryLibraryFunctionLoad[$LibraryFilePath, "get_last_error", {}, _Integer];
		$ConnectSocketToServer = tryLibraryFunctionLoad[$LibraryFilePath, "connect_socket", {_Integer, "UTF8String", _Integer}, _Integer];
		$SocketBind = tryLibraryFunctionLoad[$LibraryFilePath, "socket_bind", {_Integer, _Integer}, _Integer];
		$SocketListen = tryLibraryFunctionLoad[$LibraryFilePath, "socket_listen", {_Integer, _Integer}, _Integer];
		$SocketAccept = tryLibraryFunctionLoad[$LibraryFilePath, "socket_accept", {_Integer}, _Integer];
		$CloseSocket = tryLibraryFunctionLoad[$LibraryFilePath, "close_socket", {_Integer}, _Integer];
		$StartServerThread = tryLibraryFunctionLoad[$LibraryFilePath, "start_server_thread", {_Integer}, _Integer];
		$ProtectedModeQ = tryLibraryFunctionLoad[$LibraryFilePath, "is_protected_mode", {}, "Boolean"];
		If[TrueQ[$LoadSuccessQ],
			$InitializedQ = True,
			$InitializedQ = False
		]
	]

InitializeLibrary::libload = "Error loading function `1` from library `2`, possibly related error message: `3`";

tryLibraryFunctionLoad[lib_, fn_, rest___] := 
	Check[LibraryFunctionLoad[lib, fn, rest],
		(* on fail *)
		$LoadSuccessQ = False;
		Message[InitializeLibrary::libload, fn, lib, LibraryLink`$LibraryError];
		$Failed
	]

End[]

EndPackage[]
