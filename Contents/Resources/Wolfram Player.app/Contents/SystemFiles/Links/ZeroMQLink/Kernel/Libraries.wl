(* Wolfram Language package *)

BeginPackage["ZeroMQLink`Libraries`"]

initializeLibraries::usage="loads the libraries for zeromqlink";

$DefaultContext::usage="the default context that always exists for usage with zeromqlink";

(*librarylink exported functions*)
iCreateZMQContext;
iCloseZMQContext;
iGetZMQContextOption;
iSetZMQContextOption;
iCreateZMQSocket;
iBindZMQSocket;
iUnbindZMQSocket;
iConnectZMQSocket;
iDisconnectZMQSocket;
iDeleteZMQSocket;
iZMQSocketConnectedQ;
(* deprecated.
iSendZMQSocket; *)
iSendBytesToSocket;
(* deprecated.
iRecvSingleMultipartMessageSocket; 
*)
iRecvSingleMultipartBinaryMessageSocket;
iPollSockets;
iStartAsyncPollingThread;
iMarkZMQSocketShadowTCPSocket;
iGetZMQSocketOption;
iSetZMQSocketOption;
iSetZMQSocketSubscriptionAll;

iSocketGetAddress;
iShadowTCPClientSocketGetAddress;

iTCPSocketConnect;
iTCPSocketClose;
iTCPSocketGetOption;
iTCPSocketClosedQ;
iTCPSocketInStreamCreate;
iTCPSocketInStreamWaitForInput;
iTCPSocketInStreamRead;
iTCPSocketInErrorText;
iTCPSocketInClearError;
iTCPSocketInEndOfFileQ;
iTCPSocketInStreamPosition;
iTCPSocketInStreamSize;
iTCPSocketInStreamClose;
iTCPSocketInStreamBufferSize;

iTCPSocketOutStreamCreate;
iTCPSocketOutStreamWrite;
iTCPSocketOutErrorText;
iTCPSocketOutClearError;
iTCPSocketOutStreamPosition;
iTCPSocketOutStreamClose;

igetservbyname;

checkRes;

Begin["Private`"]

(*module is needed so that the set functions resolve properly when evaluated later on*)
initializeLibraries[] := Module[
	{
		zeromqLib = FindLibrary["libzmq"],
		zeromqLinkLib = FindLibrary["libzeromqlink"],
		res
	},
	(*don't load the dependency libary on unix - let rpath do that for us*)
	If[$OperatingSystem =!= "Unix",
		LibraryLoad[zeromqLib];
	];
	(*args are the context uuid to use*)
	iCreateZMQContext[uuid_] := Once[Quiet[LibraryFunctionLoad[zeromqLinkLib, "CreateZMQContext", {String}, {Integer,1}], LibraryFunction::overload]][uuid];
	(*args are the context uuid to use, max number of threads to use, and max number of sockets to use*)
	iCreateZMQContext[uuid_, threads_, sockets_] := Once[Quiet[LibraryFunctionLoad[zeromqLinkLib, "CreateZMQContext", {String, Integer, Integer}, {Integer,1}], LibraryFunction::overload]][uuid, threads, sockets];
	(*args are the context uuid*)
	iCloseZMQContext = LibraryFunctionLoad[zeromqLinkLib,"CloseZMQContext",{String},{Integer, 1}];
	(*args are the context uuid, the option*)
	iGetZMQContextOption = LibraryFunctionLoad[zeromqLinkLib,"GetZMQContextOption",{String,Integer},{Integer, 1}];
	(*args are the context uuid, the option, and the option value*)
	iSetZMQContextOption = LibraryFunctionLoad[zeromqLinkLib,"SetZMQContextOption",{String,Integer,Integer},{Integer, 1}];
	(*args are the socket uuid, context uuid to use for this socket, and the type*)
	iCreateZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"CreateZMQSocket",{String,String,Integer},{Integer, 1}];
	(*args are the socket uuid, and address to bind to*)
	iBindZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"BindZMQSocket",{String,String},{Integer, 1}];
	(*args are the socket uuid, and address to unbind to*)
	iUnbindZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"UnbindZMQSocket",{String,String},{Integer, 1}];
	(*args are the socket uuid, and address to connect to*)
	iConnectZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"ConnectZMQSocket",{String,String},{Integer, 1}];
	(*args are the socket uuid, and address to disconnect from*)
	iDisconnectZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"DisconnectZMQSocket",{String,String},{Integer, 1}];
	(*args are the socket uuid*)
	iDeleteZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"DeleteZMQSocket",{String},{Integer, 1}];
	(*args are the socket uuid*)
	iZMQSocketConnectedQ = LibraryFunctionLoad[zeromqLinkLib,"ZMQSocketConnectedQ",{String},{Integer, 1}];
	(*args are the socket uuid, message of bytes to send, whether to be abortable, and flags for sending*)
	(* iSendZMQSocket = LibraryFunctionLoad[zeromqLinkLib,"SendZMQSocket",{String,{Integer,1},True|False,Integer},{Integer, 1}]; *)
	(*args are the socket uuid, message of bytes to send, whether to be abortable, and flags for sending*)
	iSendBytesToSocket = LibraryFunctionLoad[zeromqLinkLib,"SendBytesToSocket",{String,"ByteArray",True|False,Integer},{Integer, 1}];
	(* dorianb: deprecated. Use iRecvSingleMultipartBinaryMessageSocket instead. *)
	(*args are the socket uuid and flags for receiving*)
	(* iRecvSingleMultipartMessageSocket = LibraryFunctionLoad[zeromqLinkLib,"RecvSingleMultipartMessageSocket",{String,Integer},{Integer, 1}]; *)
	(* same as above but with bytearray *)
	iRecvSingleMultipartBinaryMessageSocket = LibraryFunctionLoad[zeromqLinkLib,"RecvSingleMultipartBinaryMessageSocket",{String,Integer},"ByteArray"];
	(*args are the list of uuid lengths, the list of characters making up the uuids for the sockets, the list of poll items, and the timeout to use *)
	iPollSockets = LibraryFunctionLoad[zeromqLinkLib,"PollSockets",{{Integer,1},{Integer,1},{Integer,1},Integer},{Integer, 1}];
	(*arg is just the pull socket for the thread to poll for updates from Mathematica*)
	iStartAsyncPollingThread = LibraryFunctionLoad[zeromqLinkLib,"StartAsyncPollSockets",{String,String},Integer];
	(*arg is the uuid of the socket to mark as a shadow tcp socket for special handling in SocketListen*)
	iMarkZMQSocketShadowTCPSocket = LibraryFunctionLoad[zeromqLinkLib,"MarkSocketTCPShadow",{String},{Integer,1}];
	(*arg is the socket uuid and the integer value of the option to get*)
	iGetZMQSocketOption = LibraryFunctionLoad[zeromqLinkLib,"GetZMQSocketOption",{String,Integer},{Integer,1}];
	(*arg is the socket uuid and the integer value of the option to set, and the value to set as*)
	(*note the overloading*)
	(*string form*)
	iSetZMQSocketOption[uuid_,option_?IntegerQ,val_?StringQ] := Once[Quiet[LibraryFunctionLoad[zeromqLinkLib,"SetZMQSocketOption",{String,Integer,String},{Integer,1}],LibraryFunction::overload]][uuid,option,val];
	(*integer form*)
	iSetZMQSocketOption[uuid_,option_?IntegerQ,val_?IntegerQ] := Once[Quiet[LibraryFunctionLoad[zeromqLinkLib,"SetZMQSocketOption",{String,Integer,Integer},{Integer,1}],LibraryFunction::overload]][uuid,option,val];
	(*boolean form*)
	iSetZMQSocketOption[uuid_,option_?IntegerQ,val:True|False] := Once[Quiet[LibraryFunctionLoad[zeromqLinkLib,"SetZMQSocketOption",{String,Integer,"Boolean"},{Integer,1}],LibraryFunction::overload]][uuid,option,val];
	(*array form*)
	iSetZMQSocketOption[uuid_,option_?IntegerQ,val:{_?IntegerQ ..}] := Once[Quiet[LibraryFunctionLoad[zeromqLinkLib,"SetZMQSocketOption",{String,Integer,{Integer,1}},{Integer,1}],LibraryFunction::overload]][uuid,option,val];
	
	(*arg is the socket uuid, special case for setting subscription to NULL.*)
	iSetZMQSocketSubscriptionAll = LibraryFunctionLoad[zeromqLinkLib,"SetZMQSocketSubscriptionAll",{String},{Integer,1}];

	(*socket internal address functions*)
	
	(*arg is uuid of socket to get address of, option integer representing which source/destination to use, and option integer representing tcp or ZMQ socket*)
	iSocketGetAddress = LibraryFunctionLoad[zeromqLinkLib,"GetSocketAddress",{String,Integer,Integer},{Integer, 1}];
	(*arg is uuid of underlying zmq stream socket, option integer representing which source/destination to use, and option integer representing tcp or ZMQ socket*)
	iShadowTCPClientSocketGetAddress = LibraryFunctionLoad[zeromqLinkLib,"GetZMQShadowSocketAddress",{String,Integer,{Integer,1}},{Integer, 1}];
	
	(*tcp socket functions*)
	
	(*args are uuid to use for the socket, host string to connect to, and service string to connect to (which is the port as a string)*)
	iTCPSocketConnect = LibraryFunctionLoad[zeromqLinkLib,"TCPSocketConnect",{String,String,String},{Integer, 1}];
	(*arg is uuid of socket to disconnect from*)
	iTCPSocketClose = LibraryFunctionLoad[zeromqLinkLib,"Close_SocketObject",{String},{Integer, 1}];
	(*arg is uuid of socket to get option of, option integer representing which option to get*)
	iTCPSocketGetOption = LibraryFunctionLoad[zeromqLinkLib,"SocketInformation",{String,Integer},{Integer, 1}];
	
	iTCPSocketClosedQ = LibraryFunctionLoad[zeromqLinkLib,"SocketClosedQ",{String},{Integer, 1}];
	
	(*input stream tcp functions*)
	(*args are just the uuid for the socket*)
	iTCPSocketInStreamCreate = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_Constructor",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInStreamWaitForInput = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_WaitForInput",{String},{Integer,1}];
	(*args are the uuid for the socket and how many bytes to read*)
	iTCPSocketInStreamRead = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_Read",{String,Integer},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInErrorText = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_ErrorText",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInClearError = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_ClearError",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInEndOfFileQ = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_EndOfFileQ",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInStreamPosition = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_StreamPosition",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInStreamSize = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_StreamSize",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketInStreamClose = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_Close",{String},{Integer,1}];
	
	(*args are just the uuid for the socket*)
	iTCPSocketInStreamBufferSize = LibraryFunctionLoad[zeromqLinkLib,"SocketInputHandler_CurrentBufferSize",{String},{Integer,1}];
	
	(*args are the uuid for the socket*)
	iTCPSocketOutStreamCreate = LibraryFunctionLoad[zeromqLinkLib,"SocketOutputHandler_Constructor",{String},{Integer,1}];
	(*args are the uuid for the socket and how many bytes to read*)
	iTCPSocketOutStreamWrite = LibraryFunctionLoad[zeromqLinkLib,"SocketOutputHandler_Write",{String,{Integer,1}},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketOutErrorText = LibraryFunctionLoad[zeromqLinkLib,"SocketOutputHandler_ErrorText",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketOutClearError = LibraryFunctionLoad[zeromqLinkLib,"SocketOutputHandler_ClearError",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketOutStreamPosition = LibraryFunctionLoad[zeromqLinkLib,"SocketOutputHandler_StreamPosition",{String},{Integer,1}];
	(*args are just the uuid for the socket*)
	iTCPSocketOutStreamClose = LibraryFunctionLoad[zeromqLinkLib,"SocketOutputHandler_Close",{String},{Integer,1}];
	
	igetservbyname = LibraryFunctionLoad[zeromqLinkLib,"GetServiceByName",{String,String},{Integer,1}];
	
	(*always create the default context*) 
	$DefaultContext = CreateUUID[];
	res = iCreateZMQContext[$DefaultContext];
	
	(*make sure the context was created correctly*)
	checkRes[res,Null];
];



(*helper function which confirms the result of a librarylink function and issues messages, etc. based on the caller*)
checkRes[result_,symbol_] := Which[
	MatchQ[result,LibraryFunctionError[___]], (*library function error*)
	$Failed,
	First[result] === 0, (*result is correct, no errors*)
	(*if there's more data, return the rest of it*)
	If[Length[result] > 1,
		Rest[result],
		Null
	],
	First[result] === 1, (*zmq exception, issue a message*)
	(
		With[{s=symbol},
			Message[
				MessageName[s, "zmqexception"],
				"ZeroMQ",
				result[[2]],
				stringResult[result[[3;;]]]
			]
		];
		$Failed
	),
	First[result] === 2,
	(
		With[{s=symbol},
			Message[
				MessageName[s, "zmqexception"],
				"ZeroMQLink",
				result[[2]],
				stringResult[result[[3;;]]]	
			]
		];
		$Failed
	)
];

stringResult[ba_ByteArray] := ByteArrayToString[ba, "UTF-8"];
stringResult[l_List] := FromCharacterCode[l, "UTF-8"];


End[]

EndPackage[]