BeginPackage["SocketLink`"]


(*****************************************************************************)
CreateClientSocket::usage = "CreateClientSocket[host, p] creates a client-side socket that connects to a network server at host on port p."

CreateServerSocket::usage = "CreateServerSocket[p] creates a server socket that accepts incoming connections on port p."

CreateAsynchronousServer::usage = "CreateAsynchronousServer[s, connectionHandler] uses server socket s and starts listening for incoming connection; when one is created, connectionHandler[s] is called with a connected network endpoint socket s."

OpenSocketStreams::usage = "OpenSocketStreams[socket] returns the input and output streams for communicating over socket."

General::sockerr = "Error `1` to socket: `2`."

Begin["`Private`"]
Needs["SocketLink`Library`"]
Needs["SocketLink`Sockets`"]

OpenSocketStreams[s:Socket[fd_]] := 
	(
		InitializeLibrary[];
		{
			OpenRead[ToString[fd], Method->"Sockets", BinaryFormat->True],
			OpenWrite["out:"<>ToString[fd], Method -> "Sockets",
				BinaryFormat -> True, FormatType -> OutputForm]
		}
	)

CreateClientSocket[host_String, port_Integer, OptionsPattern[]] := 
	Module[{socket, connectResult},
		InitializeLibrary[];
		socket = CreateStreamSocket[];
		If[Head[socket] =!= Socket,
			Return[$Failed]
		];
		connectResult = ConnectSocket[socket, host, port];
		If[connectResult === 0, socket, $Failed]
	]

CreateServerSocket[port_Integer, OptionsPattern[]] := 
	Module[{socket, result},
		InitializeLibrary[];
		socket = CreateStreamSocket[];
		If[Head[socket] =!= Socket, Return[$Failed]];
		
		result = SocketBind[socket, port];
		If[result =!= 0, Return[$Failed]];

		result = SocketListen[socket, 10];
		If[result =!= 0, Return[$Failed]];

		socket
	]

(*****************************************************************************)
(* Asynchronous server connection handling *)

$connectionCount = 0

CreateAsynchronousServer[Socket[fd_], connectionHandler_] := 
	Module[{asyncObj},
		InitializeLibrary[];
		asyncObj = Internal`CreateAsynchronousTask[$StartServerThread, {fd}, 
			Function[{aobj, evtType, evtData}, 
				HandleNewConnection[aobj, evtType, evtData, connectionHandler]
			],
			"TaskDetail" -> "SocketServer-"<>ToString[fd]
		];
		asyncObj
	]

(* HandleNewConnection is invoked asynchronously when a client connects to the server *)	
HandleNewConnection[asyncObj_, (*eventType*)_, {clientSocketFD_}, fn_] := 
	Module[{clientsocket, streams},
		$connectionCount++;
		If[IntegerQ[clientSocketFD],
			clientsocket = Socket[clientSocketFD];
			streams = OpenSocketStreams[clientsocket];
			$lastResult = fn[streams];
			CloseSocket[clientsocket];
			,
			Print["Error, clientsocket not an integer: ",InputForm[clientSocketFD]]
		];
	]

HandleNewConnection[args___] := 
	Print["Unhandled connection callback; args: ",ToString[args,InputForm]]

End[]

EndPackage[]
