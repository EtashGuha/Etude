(* Low-level socket functions. *)
BeginPackage["SocketLink`Sockets`"]

Socket::usage = "Socket[fd] is an expression that represents a network socket using file descriptor fd."

NewSocket::usage = "NewSocket"

CreateStreamSocket::usage = "CreateStreamSocket[] creates a connection-oriented TCP/IP socket."

CreateDatagramSocket::usage = "CreateDatagramSocket[] creates a connectionless UDP socket."

ConnectSocket::usage = "ConnectSocket[s, host, p] connects socket s to network server host at port p."

SocketBind::usage = "SocketBind[s, p] binds socket s to port p for accepting incoming connections."

SocketListen::usage = "SocketListen[s, b] puts socket s into a listening server mode with a maximum backlog of b unhandled connections, ready to accept incoming connections with SocketAccept."

SocketAccept::usage = "SocketAccept[s] waits for an incoming connection and a new socket connection to the client is returned when one arrives, or one is already in the queue."

CloseSocket::usage = "CloseSocket[s] shuts down communications on socket s and frees its system resources."

(*****************************************************************************)
(* Options *)
Options[ConnectSocket] = {"Timeout" -> 60};

(*****************************************************************************)
(* Messages *)
CreateStreamSocket::mksock = CreateDatagramSocket::mksock = "Failed to create a socket: `1`"

Begin["`Private`"] 
Needs["SocketLink`Library`"]

NewSocket[_, fd_Integer?NonNegative] := 
	If[TrueQ[$ProtectedModeQ[]],
		Message[hd::sandbox, Inactivate[hd[]]];
		$Failed,
	(* Else *)
		Socket[fd]
	]

NewSocket[hd_, _] := 
(
	InitializeLibrary[];
	If[TrueQ[$ProtectedModeQ[]],
		Message[hd::sandbox, Inactivate[hd[]]],
	(* Else *) 
		Message[hd::mksock, $GetLastError[]]
	];
	$Failed
)

SetAttributes[NewSocket, {Protected, ReadProtected, Locked}]

CreateStreamSocket[] := 
	(
		InitializeLibrary[];
		NewSocket[CreateStreamSocket, $CreateStreamSocket[]]
	)

CreateDatagramSocket[] := 
	(
		InitializeLibrary[];
		NewSocket[CreateDatagramSocket, $CreateDatagramSocket[]]
	)

ConnectSocket[Socket[fd_], host_String, port_Integer, OptionsPattern[]] := 
	Module[{(*timeout = OptionValue["Timeout"]*)},
		handleSocketResult[$ConnectSocketToServer[fd, host, port],
			ConnectSocket, "connecting"]
	]

SocketBind[Socket[fd_], port_Integer] := 
(
	InitializeLibrary[];
	handleSocketResult[$SocketBind[fd, port], SocketBind, "binding"]
)

SocketListen[Socket[fd_], backlog_Integer] := 
(
	InitializeLibrary[];
	handleSocketResult[$SocketListen[fd, backlog], SocketListen, "listening"]
)

SocketAccept[Socket[fd_]] := 
(
	InitializeLibrary[];
	$SocketAccept[fd] /. {
		n_?Negative :> handleSocketResult[n, SocketAccept, "accepting"],
		clientfd_ :> Socket[clientfd]
	}
)

handleSocketResult[n_?Negative, hd_, verb_] := 
(
	InitializeLibrary[];
	Message[hd::sockerr, verb, $GetLastError[]];
	$Failed
)

handleSocketResult[res_, hd_, verb_] := res

CloseSocket[Socket[fd_]] := 
(
	InitializeLibrary[];
	$CloseSocket[fd]
)

End[]

EndPackage[]
