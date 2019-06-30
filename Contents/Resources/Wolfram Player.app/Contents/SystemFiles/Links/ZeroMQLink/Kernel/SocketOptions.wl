(* Wolfram Language Package *)

BeginPackage["ZeroMQLink`SocketOptions`", {"ZeroMQLink`","ZeroMQLink`Libraries`"}]
(* Exported symbols added here with SymbolName::usage *)  

(*constants for manipulating ZMQ socket options - these have been auto-generated from the documentation*)
$GetSocketOptionValues;
$SetSocketOptionValues;
$SetSocketOptionTypes;
$GetSocketOptionInterpretTypes;

(*functions for manipulating ZMQ socket option names*)
toZMQOptionString;
fromZMQOptionString;

(*functions for actually setting and getting socket options*)
getSockOpts;
setSockOpt;

(*option value utility functions*)
confirmHandlerKeys;
validRecSepQ;

(*function for getting and setting socket listener options*)
setSocketListenerOpt;
getSocketListenerOpts;


Begin["`Private`"] (* Begin Private Context *) 



toZMQOptionString[str_?StringQ] := If[KeyExistsQ[str]@$SetSocketOptionTypes,
	(*THEN*)
	(*it's already the verbatim form, don't change it*)
	str,
	(*ELSE*)
	(*need t o call helper function to turn the option name into the ZMQ constant value string*)
	"ZMQ_" <> ToUpperCase@StringReplace[str,x:CharacterRange["a","z"] ~~ y:CharacterRange["A", "Z"].. :> x<>"_"<>y]
]
(*inverse of the above funciton, takes the option name and turns it into CamelCase style Mathematica option string*)
fromZMQOptionString[str_?StringQ]:=StringJoin[Capitalize/@Rest@StringSplit[ToLowerCase@str,"_"]]

(*this was auto generated - see the notebook in the repository for how this and the set options associations are generated*)
$GetSocketOptionValues = <|"ZMQ_AFFINITY" -> 4, "ZMQ_BACKLOG" -> 19, 
 "ZMQ_CONNECT_TIMEOUT" -> 79, "ZMQ_CURVE_PUBLICKEY" -> 48, 
 "ZMQ_CURVE_SECRETKEY" -> 49, "ZMQ_CURVE_SERVERKEY" -> 50, 
 "ZMQ_EVENTS" -> 15, "ZMQ_FD" -> 14, "ZMQ_GSSAPI_PLAINTEXT" -> 65, 
 "ZMQ_GSSAPI_PRINCIPAL" -> 63, "ZMQ_GSSAPI_SERVER" -> 62, 
 "ZMQ_GSSAPI_SERVICE_PRINCIPAL" -> 64, "ZMQ_HANDSHAKE_IVL" -> 66, 
 "ZMQ_IDENTITY" -> 5, "ZMQ_IMMEDIATE" -> 39, 
 "ZMQ_INVERT_MATCHING" -> 74, "ZMQ_IPV4ONLY" -> 31, "ZMQ_IPV6" -> 42, 
 "ZMQ_LAST_ENDPOINT" -> 32, "ZMQ_LINGER" -> 17, 
 "ZMQ_MAXMSGSIZE" -> 22, "ZMQ_MECHANISM" -> 43, 
 "ZMQ_MULTICAST_HOPS" -> 25, "ZMQ_MULTICAST_MAXTPDU" -> 84, 
 "ZMQ_PLAIN_PASSWORD" -> 46, "ZMQ_PLAIN_SERVER" -> 44, 
 "ZMQ_PLAIN_USERNAME" -> 45, "ZMQ_USE_FD" -> 89, "ZMQ_RATE" -> 8, 
 "ZMQ_RCVBUF" -> 12, "ZMQ_RCVHWM" -> 24, "ZMQ_RCVMORE" -> 13, 
 "ZMQ_RCVTIMEO" -> 27, "ZMQ_RECONNECT_IVL" -> 18, 
 "ZMQ_RECONNECT_IVL_MAX" -> 21, "ZMQ_RECOVERY_IVL" -> 9, 
 "ZMQ_SNDBUF" -> 11, "ZMQ_SNDHWM" -> 23, "ZMQ_SNDTIMEO" -> 28, 
 "ZMQ_SOCKS_PROXY" -> 68, "ZMQ_TCP_KEEPALIVE" -> 34, 
 "ZMQ_TCP_KEEPALIVE_CNT" -> 35, "ZMQ_TCP_KEEPALIVE_IDLE" -> 36, 
 "ZMQ_TCP_KEEPALIVE_INTVL" -> 37, "ZMQ_TCP_MAXRT" -> 80, 
 "ZMQ_THREAD_SAFE" -> 81, "ZMQ_TOS" -> 57, "ZMQ_TYPE" -> 16, 
 "ZMQ_ZAP_DOMAIN" -> 55, "ZMQ_VMCI_BUFFER_SIZE" -> 85, 
 "ZMQ_VMCI_BUFFER_MIN_SIZE" -> 86, "ZMQ_VMCI_BUFFER_MAX_SIZE" -> 87, 
 "ZMQ_VMCI_CONNECT_TIMEOUT" -> 88|>;


$SetSocketOptionValues=<|"ZMQ_AFFINITY" -> 4, "ZMQ_BACKLOG" -> 19, "ZMQ_CONNECT_RID" -> 61, 
 "ZMQ_CONFLATE" -> 54, "ZMQ_CONNECT_TIMEOUT" -> 79, 
 "ZMQ_CURVE_PUBLICKEY" -> 48, "ZMQ_CURVE_SECRETKEY" -> 49, 
 "ZMQ_CURVE_SERVER" -> 47, "ZMQ_CURVE_SERVERKEY" -> 50, 
 "ZMQ_GSSAPI_PLAINTEXT" -> 65, "ZMQ_GSSAPI_PRINCIPAL" -> 63, 
 "ZMQ_GSSAPI_SERVER" -> 62, "ZMQ_GSSAPI_SERVICE_PRINCIPAL" -> 64, 
 "ZMQ_HANDSHAKE_IVL" -> 66, "ZMQ_HEARTBEAT_IVL" -> 75, 
 "ZMQ_HEARTBEAT_TIMEOUT" -> 77, "ZMQ_HEARTBEAT_TTL" -> 76, 
 "ZMQ_IDENTITY" -> 5, "ZMQ_IMMEDIATE" -> 39, 
 "ZMQ_INVERT_MATCHING" -> 74, "ZMQ_IPV6" -> 42, "ZMQ_LINGER" -> 17, 
 "ZMQ_MAXMSGSIZE" -> 22, "ZMQ_MULTICAST_HOPS" -> 25, 
 "ZMQ_MULTICAST_MAXTPDU" -> 84, "ZMQ_PLAIN_PASSWORD" -> 46, 
 "ZMQ_PLAIN_SERVER" -> 44, "ZMQ_PLAIN_USERNAME" -> 45, 
 "ZMQ_USE_FD" -> 89, "ZMQ_PROBE_ROUTER" -> 51, "ZMQ_RATE" -> 8, 
 "ZMQ_RCVBUF" -> 12, "ZMQ_RCVHWM" -> 24, "ZMQ_RCVTIMEO" -> 27, 
 "ZMQ_RECONNECT_IVL" -> 18, "ZMQ_RECONNECT_IVL_MAX" -> 21, 
 "ZMQ_RECOVERY_IVL" -> 9, "ZMQ_REQ_CORRELATE" -> 52, 
 "ZMQ_REQ_RELAXED" -> 53, "ZMQ_ROUTER_HANDOVER" -> 56, 
 "ZMQ_ROUTER_MANDATORY" -> 33, "ZMQ_ROUTER_RAW" -> 41, 
 "ZMQ_SNDBUF" -> 11, "ZMQ_SNDHWM" -> 23, "ZMQ_SNDTIMEO" -> 28, 
 "ZMQ_SOCKS_PROXY" -> 68, "ZMQ_STREAM_NOTIFY" -> 73, 
 "ZMQ_SUBSCRIBE" -> 6, "ZMQ_TCP_KEEPALIVE" -> 34, 
 "ZMQ_TCP_KEEPALIVE_CNT" -> 35, "ZMQ_TCP_KEEPALIVE_IDLE" -> 36, 
 "ZMQ_TCP_KEEPALIVE_INTVL" -> 37, "ZMQ_TCP_MAXRT" -> 80, 
 "ZMQ_TOS" -> 57, "ZMQ_UNSUBSCRIBE" -> 7, "ZMQ_XPUB_VERBOSE" -> 40, 
 "ZMQ_XPUB_VERBOSER" -> 78, "ZMQ_XPUB_MANUAL" -> 71, 
 "ZMQ_XPUB_NODROP" -> 69, "ZMQ_XPUB_WELCOME_MSG" -> 72, 
 "ZMQ_ZAP_DOMAIN" -> 55, "ZMQ_TCP_ACCEPT_FILTER" -> 38, 
 "ZMQ_IPC_FILTER_GID" -> 60, "ZMQ_IPC_FILTER_PID" -> 58, 
 "ZMQ_IPC_FILTER_UID" -> 59, "ZMQ_IPV4ONLY" -> 31, 
 "ZMQ_VMCI_BUFFER_SIZE" -> 85, "ZMQ_VMCI_BUFFER_MIN_SIZE" -> 86, 
 "ZMQ_VMCI_BUFFER_MAX_SIZE" -> 87, "ZMQ_VMCI_CONNECT_TIMEOUT" -> 88|>;

(*for confirming the type of the option argument passed in*)
$SetSocketOptionTypes=<|"ZMQ_AFFINITY" -> (IntegerQ[#1] && #1 >= 0 && #1 <= 
      2^$SystemWordLength - 1 &), 
 "ZMQ_BACKLOG" -> Developer`MachineIntegerQ, 
 "ZMQ_CONNECT_RID" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_CONFLATE" -> Developer`MachineIntegerQ, 
 "ZMQ_CONNECT_TIMEOUT" -> Developer`MachineIntegerQ, 
 "ZMQ_CURVE_PUBLICKEY" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_CURVE_SECRETKEY" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_CURVE_SERVER" -> Developer`MachineIntegerQ, 
 "ZMQ_CURVE_SERVERKEY" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_GSSAPI_PLAINTEXT" -> Developer`MachineIntegerQ, 
 "ZMQ_GSSAPI_PRINCIPAL" -> StringQ, 
 "ZMQ_GSSAPI_SERVER" -> Developer`MachineIntegerQ, 
 "ZMQ_GSSAPI_SERVICE_PRINCIPAL" -> StringQ, 
 "ZMQ_HANDSHAKE_IVL" -> Developer`MachineIntegerQ, 
 "ZMQ_HEARTBEAT_IVL" -> Developer`MachineIntegerQ, 
 "ZMQ_HEARTBEAT_TIMEOUT" -> Developer`MachineIntegerQ, 
 "ZMQ_HEARTBEAT_TTL" -> Developer`MachineIntegerQ, 
 "ZMQ_IDENTITY" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_IMMEDIATE" -> Developer`MachineIntegerQ, 
 "ZMQ_INVERT_MATCHING" -> Developer`MachineIntegerQ, 
 "ZMQ_IPV6" -> Developer`MachineIntegerQ, 
 "ZMQ_LINGER" -> Developer`MachineIntegerQ, 
 "ZMQ_MAXMSGSIZE" -> Developer`MachineIntegerQ, 
 "ZMQ_MULTICAST_HOPS" -> Developer`MachineIntegerQ, 
 "ZMQ_MULTICAST_MAXTPDU" -> Developer`MachineIntegerQ, 
 "ZMQ_PLAIN_PASSWORD" -> StringQ, 
 "ZMQ_PLAIN_SERVER" -> Developer`MachineIntegerQ, 
 "ZMQ_PLAIN_USERNAME" -> StringQ, 
 "ZMQ_USE_FD" -> Developer`MachineIntegerQ, 
 "ZMQ_PROBE_ROUTER" -> Developer`MachineIntegerQ, 
 "ZMQ_RATE" -> Developer`MachineIntegerQ, 
 "ZMQ_RCVBUF" -> Developer`MachineIntegerQ, 
 "ZMQ_RCVHWM" -> Developer`MachineIntegerQ, 
 "ZMQ_RCVTIMEO" -> Developer`MachineIntegerQ, 
 "ZMQ_RECONNECT_IVL" -> Developer`MachineIntegerQ, 
 "ZMQ_RECONNECT_IVL_MAX" -> Developer`MachineIntegerQ, 
 "ZMQ_RECOVERY_IVL" -> Developer`MachineIntegerQ, 
 "ZMQ_REQ_CORRELATE" -> Developer`MachineIntegerQ, 
 "ZMQ_REQ_RELAXED" -> Developer`MachineIntegerQ, 
 "ZMQ_ROUTER_HANDOVER" -> Developer`MachineIntegerQ, 
 "ZMQ_ROUTER_MANDATORY" -> Developer`MachineIntegerQ, 
 "ZMQ_ROUTER_RAW" -> Developer`MachineIntegerQ, 
 "ZMQ_SNDBUF" -> Developer`MachineIntegerQ, 
 "ZMQ_SNDHWM" -> Developer`MachineIntegerQ, 
 "ZMQ_SNDTIMEO" -> Developer`MachineIntegerQ, 
 "ZMQ_SOCKS_PROXY" -> StringQ, 
 "ZMQ_STREAM_NOTIFY" -> Developer`MachineIntegerQ, 
 "ZMQ_SUBSCRIBE" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_TCP_KEEPALIVE" -> Developer`MachineIntegerQ, 
 "ZMQ_TCP_KEEPALIVE_CNT" -> Developer`MachineIntegerQ, 
 "ZMQ_TCP_KEEPALIVE_IDLE" -> Developer`MachineIntegerQ, 
 "ZMQ_TCP_KEEPALIVE_INTVL" -> Developer`MachineIntegerQ, 
 "ZMQ_TCP_MAXRT" -> Developer`MachineIntegerQ, 
 "ZMQ_TOS" -> Developer`MachineIntegerQ, 
 "ZMQ_UNSUBSCRIBE" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_XPUB_VERBOSE" -> Developer`MachineIntegerQ, 
 "ZMQ_XPUB_VERBOSER" -> Developer`MachineIntegerQ, 
 "ZMQ_XPUB_MANUAL" -> Developer`MachineIntegerQ, 
 "ZMQ_XPUB_NODROP" -> Developer`MachineIntegerQ, 
 "ZMQ_XPUB_WELCOME_MSG" -> (ListQ[#1] && ByteArrayQ[ByteArray[#1]] &),
  "ZMQ_ZAP_DOMAIN" -> StringQ, 
 "ZMQ_TCP_ACCEPT_FILTER" -> (ListQ[#1] && 
     ByteArrayQ[ByteArray[#1]] &), 
 "ZMQ_IPC_FILTER_GID" -> Developer`MachineIntegerQ, 
 "ZMQ_IPC_FILTER_PID" -> Developer`MachineIntegerQ, 
 "ZMQ_IPC_FILTER_UID" -> Developer`MachineIntegerQ, 
 "ZMQ_IPV4ONLY" -> Developer`MachineIntegerQ, 
 "ZMQ_VMCI_BUFFER_SIZE" -> (IntegerQ[#1] && #1 >= 0 && #1 <= 
      2^$SystemWordLength - 1 &), 
 "ZMQ_VMCI_BUFFER_MIN_SIZE" -> (IntegerQ[#1] && #1 >= 0 && #1 <= 
      2^$SystemWordLength - 1 &), 
 "ZMQ_VMCI_BUFFER_MAX_SIZE" -> (IntegerQ[#1] && #1 >= 0 && #1 <= 
      2^$SystemWordLength - 1 &), 
 "ZMQ_VMCI_CONNECT_TIMEOUT" -> Developer`MachineIntegerQ|>;

(*for interpreting the values we get back from the LibraryLink function*)
$GetSocketOptionInterpretTypes=<|"ZMQ_AFFINITY" -> First, "ZMQ_BACKLOG" -> First, 
 "ZMQ_CONNECT_TIMEOUT" -> First, "ZMQ_CURVE_PUBLICKEY" -> ByteArray, 
 "ZMQ_CURVE_SECRETKEY" -> ByteArray, 
 "ZMQ_CURVE_SERVERKEY" -> ByteArray, "ZMQ_EVENTS" -> First, 
 "ZMQ_FD" -> First, "ZMQ_GSSAPI_PLAINTEXT" -> First, 
 "ZMQ_GSSAPI_PRINCIPAL" -> (FromCharacterCode[DeleteCases[#1, 0], 
     "UTF-8"] &), "ZMQ_GSSAPI_SERVER" -> First, 
 "ZMQ_GSSAPI_SERVICE_PRINCIPAL" -> (FromCharacterCode[
     DeleteCases[#1, 0], "UTF-8"] &), "ZMQ_HANDSHAKE_IVL" -> First, 
 "ZMQ_IDENTITY" -> ByteArray, "ZMQ_IMMEDIATE" -> First, 
 "ZMQ_INVERT_MATCHING" -> First, "ZMQ_IPV4ONLY" -> First, 
 "ZMQ_IPV6" -> First, 
 "ZMQ_LAST_ENDPOINT" -> (FromCharacterCode[DeleteCases[#1, 0], 
     "UTF-8"] &), "ZMQ_LINGER" -> First, "ZMQ_MAXMSGSIZE" -> First, 
 "ZMQ_MECHANISM" -> First, "ZMQ_MULTICAST_HOPS" -> First, 
 "ZMQ_MULTICAST_MAXTPDU" -> First, 
 "ZMQ_PLAIN_PASSWORD" -> (FromCharacterCode[DeleteCases[#1, 0], 
     "UTF-8"] &), "ZMQ_PLAIN_SERVER" -> First, 
 "ZMQ_PLAIN_USERNAME" -> (FromCharacterCode[DeleteCases[#1, 0], 
     "UTF-8"] &), "ZMQ_USE_FD" -> First, "ZMQ_RATE" -> First, 
 "ZMQ_RCVBUF" -> First, "ZMQ_RCVHWM" -> First, "ZMQ_RCVMORE" -> First,
  "ZMQ_RCVTIMEO" -> First, "ZMQ_RECONNECT_IVL" -> First, 
 "ZMQ_RECONNECT_IVL_MAX" -> First, "ZMQ_RECOVERY_IVL" -> First, 
 "ZMQ_SNDBUF" -> First, "ZMQ_SNDHWM" -> First, 
 "ZMQ_SNDTIMEO" -> First, 
 "ZMQ_SOCKS_PROXY" -> (FromCharacterCode[DeleteCases[#1, 0], 
     "UTF-8"] &), "ZMQ_TCP_KEEPALIVE" -> First, 
 "ZMQ_TCP_KEEPALIVE_CNT" -> First, "ZMQ_TCP_KEEPALIVE_IDLE" -> First, 
 "ZMQ_TCP_KEEPALIVE_INTVL" -> First, "ZMQ_TCP_MAXRT" -> First, 
 "ZMQ_THREAD_SAFE" -> (#1 === 1 &), "ZMQ_TOS" -> First, 
 "ZMQ_TYPE" -> First, 
 "ZMQ_ZAP_DOMAIN" -> (FromCharacterCode[DeleteCases[#1, 0], 
     "UTF-8"] &), "ZMQ_VMCI_BUFFER_SIZE" -> First, 
 "ZMQ_VMCI_BUFFER_MIN_SIZE" -> First, 
 "ZMQ_VMCI_BUFFER_MAX_SIZE" -> First, 
 "ZMQ_VMCI_CONNECT_TIMEOUT" -> First|>;
 
 
 
(*zmq socket options that aren't valid to query on some types of sockets*)
$NonDefaultSockOpts = {
 	"ZMQ_CURVE_PUBLICKEY",
 	"ZMQ_CURVE_SECRETKEY",
 	"ZMQ_CURVE_SERVERKEY",
 	"ZMQ_GSSAPI_PLAINTEXT",
 	"ZMQ_GSSAPI_PRINCIPAL",
 	"ZMQ_GSSAPI_SERVER",
 	"ZMQ_GSSAPI_SERVICE_PRINCIPAL",
 	"ZMQ_IDENTITY",
 	"ZMQ_THREAD_SAFE",
 	"ZMQ_VMCI_BUFFER_MAX_SIZE",
 	"ZMQ_VMCI_BUFFER_MIN_SIZE",
 	"ZMQ_VMCI_BUFFER_SIZE",
 	"ZMQ_VMCI_CONNECT_TIMEOUT"
 };




(*get all the socket options from the socket as an association*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ]/; StringMatchQ[sock["Protocol"],"ZMQ"~~___] := Join[
	<|#->getSockOpts[sock,#]&/@$DefaultAnySockOpts|>,
	<|#->getSockOpts[sock,#]&/@fromZMQOptionString/@Keys[KeyDrop[$NonDefaultSockOpts]@$GetSocketOptionValues]|>
]

getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ]/; sock["Protocol"] === "TCP" := 
	<|#->getSockOpts[sock,#]&/@$DefaultAnySockOpts|>

(*properties returns the list of possible options - not including all the non default socket options which don't make sense for most sockets*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"Properties"]:=Keys@Options[sock]

(*standard socket options that are stored in $Sockets*)
$DefaultAnySockOpts = {"DestinationIPAddress","DestinationPort","SourceIPAddress","SourcePort"};

(*get an individual standard option*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,option_?StringQ] /; MemberQ[$DefaultAnySockOpts,option] := ZeroMQLink`Private`$Sockets[First[sock],option]

getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"Protocol"] /; ZeroMQLink`Private`$Sockets[First[sock],"Protocol"] === "TCP" := "TCP"

(*uuid is just the first element of the socket object*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"UUID"] := First[sock]

(*The protocol for ZMQ sockets is different - internally we just store the protocol as "ZMQ", but externally to users the protocol will be something like "ZMQ_Pair", etc.*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"Protocol"] /; ZeroMQLink`Private`$Sockets[First[sock],"Protocol"] === "ZMQ" := (
	ZeroMQLink`Private`$Sockets[First[sock],"Type"]
)


(*for the connected clients property, we specifically switch on what type of socket we are dealing with*)
(*for zmq sockets, the connected client is always the socket itself*)
(*however for tcp sockets, it's the list of identities that SocketListen has stored for us when SocketOpen was called on the socket*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"ConnectedClients"]:=Block[
	{
		sockUUID = First[sock]
	},
	(
		(*note that this property doesn't work for client sockets, as it's always just the server*)
		If[ZeroMQLink`Private`$Sockets[sockUUID,"DirectionType"] === "Server",
			(*THEN*)
			(*we can check the connected clients*)
			(
				(*switch on what kind of socket we have - zmq sockets just return the socket themselves*)
				If[ZeroMQLink`Private`$Sockets[sockUUID,"Protocol"] === "ZMQ",
					(*THEN*)
					(*just return this socket*)
					sock,
					(*ELSE*)
					(*we have a tcp server socket, in which case the list of client connections will be given in $AsyncState for this socket*)
					(
						System`SocketObject/@
							Select[ZeroMQLink`Private`$Sockets[#,"Active"]&]@
								Keys[ZeroMQLink`Private`$AsyncState["TCPClients",ZeroMQLink`Private`$Sockets[sockUUID,"ZMQSocket"]]/.{Missing[___]:><||>}]
					)
				]
			),
			(*ELSE*)
			(*we have a client socket so we can't also have connected clients*)
			(
				Message[SocketObject::clientSock,sock];
				$Failed
			)
		]
	)
]

(*convenience function for querying the inputstream for a socket object*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"InputStream"]:=ZeroMQLink`Private`$Sockets[First[sock],"StreamState","InputStream"]
(*same for outputstream*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,"OutputStream"]:=ZeroMQLink`Private`$Sockets[First[sock],"StreamState","OutputStream"]

(*get an individual zmq socket option*)
getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,option_?StringQ]/;!MemberQ[$DefaultAnySockOpts,option]:=Block[
	{
		(*turn the option into the form of ZMQ_opt_val*)
		optString = toZMQOptionString[option],
		res
	},
	(
		If[MemberQ[optString]@Keys[$GetSocketOptionValues],
			(*THEN*)
			(*get the function and then check the value to ensure it's valid*)
			(
				res = checkRes[iGetZMQSocketOption[First[sock],$GetSocketOptionValues[optString]],Options];
				If[res === $Failed,
					(*THEN*)
					(*error - note that checkRes would have already raised a message for us*)
					(
						$Failed
					),
					(*ELSE*)
					(*worked, interpret the result appropriately to return*)
					(
						$GetSocketOptionInterpretTypes[optString][res]
					)
				]
			),
			(*ELSE*)
			(*invalid socket option*)
			(
				Message[System`SocketObject::invalidGetOption,option];
				$Failed
			)
		]
	)
];




getSockOpts[sock_?ZeroMQLink`Private`validSocketObjectQ,invalidOption__]:=
(
	Message[System`SocketObject::getopt,invalidOption];
	$Failed
)


getSockOpts[sock_?(Not@*ZeroMQLink`Private`validSocketObjectQ),___]:=
(
	Message[System`SocketObject::invalidSock,sock];
	$Failed
)


setSockOpt[sock_?ZeroMQLink`Private`validSocketObjectQ,rules__?(MatchQ[#,(_Rule | _RuleDelayed)..]&)]:=
(
	setSockOptSingle[sock,#]&/@Flatten[{rules},1];
)

setSockOptSingle[sock_?ZeroMQLink`Private`validSocketObjectQ,option_?StringQ->val_]:=Block[
	{
		(*turn the option into the form of ZMQ_opt_val*)
		optString = toZMQOptionString[option]
	},
	(
		If[MemberQ[optString]@Keys[$SetSocketOptionValues],
			(*THEN*)
			(*check the value to ensure it's valid*)
			(
				If[optString=="ZMQ_SUBSCRIBE" && val == All,checkRes[iSetZMQSocketSubscriptionAll[First[sock]]],
				If[$SetSocketOptionTypes[optString][val],
					(*THEN*)
					(*valid option, set it*)
					(
						(*check the result of setting it*)
						checkRes[iSetZMQSocketOption[First[sock],$SetSocketOptionValues[optString],val],SetOptions];
					),
					(*ELSE*)
					(*invalid issue a message*)
					(
						Message[System`SocketObject::invalidSetOptionValue,option,val];
						$Failed
					)
				]]
			),
			(*ELSE*)
			(*invalid socket option*)
			(
				Message[System`SocketObject::invalidSetOption,option];
				$Failed
			)
		]
	)
];





$DefaultSocketListenerOptions = {HandlerFunctions,HandlerFunctionsKeys,CharacterEncoding,RecordSeparators};

getSocketListenerOpts[sock_?ZeroMQLink`Private`validSocketListenerQ,opt_] /; MemberQ[ToString@opt]@(ToString/@$DefaultSocketListenerOptions) := Block[{},
	(
		ZeroMQLink`Private`$AsyncState["Handlers",ZeroMQLink`Private`$Sockets[First[sock],"ZMQSocket"],opt/.{str_?StringQ:>Symbol[str]}]
	)
]

(*The Socket parameter returns the SocketObject associated with this SocketListener*)
getSocketListenerOpts[sock_?ZeroMQLink`Private`validSocketListenerQ,"Socket"]  := System`SocketObject[First[sock]]


getSocketListenerOpts[sock_?ZeroMQLink`Private`validSocketListenerQ,opt_] /; !MemberQ[ToString@opt]@(ToString/@$DefaultSocketListenerOptions) := (
	Message[System`SocketListener::invalidOpt,opt];
	$Failed
)

getSocketListenerOpts[sock_?(Not@*ZeroMQLink`Private`validSocketListenerQ),opt___] /; !MemberQ[ToString@opt]@(ToString/@$DefaultSocketListenerOptions) := (
	Message[System`SocketListener::invalidSock,sock];
	$Failed
)

getSocketListenerOpts[sock_?ZeroMQLink`Private`validSocketListenerQ]:=
	AssociationMap[getSocketListenerOpts[sock,#]&,Append[$DefaultSocketListenerOptions,"Socket"]]

(*this method just checks to see if the listener socket is a tcp server socket, in which case then the settings need to be propogated down to the*)
(*underlying zmq stream socket instead of the actual TCP socket the user sees*)
setSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,opt_->val_]:=Block[
	{
		sockUUID = First[sock]
	},
	If[ZeroMQLink`Private`shadowTCPServerSocketQ[System`SocketObject[sockUUID]],
		(*THEN*)
		(*then we need to update the underlying zmq stream socket*)
		(
			setRealSocketListenerOpt[System`SocketListener[ZeroMQLink`Private`$Sockets[sockUUID,"ZMQSocket"]],opt->val]
		),
		(*ELSE*)
		(*normal socket, continue with the socket as is*)
		(
			setRealSocketListenerOpt[sock,opt->val]
		)
	]
]


setSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,any__]:=Block[{},
	(
		Message[System`SocketListener::invalidOpt,any];
		$Failed
	)
]

setSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ]:=Block[{},
	(
		Message[General::args, SetOptions];
		$Failed
	)
]

setSocketListenerOpt[any_,___]:=Block[{},
	(
		Message[System`SocketListener::invalidSock, any];
		$Failed
	)
]

(*setter for the RecordSeparators option for SocketListener objects*)
setRealSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,RecordSeparators|"RecordSeparators"->val_]:=Block[
	{
		sockUUID = First[sock]
	},
	If[KeyExistsQ[sockUUID]@ZeroMQLink`Private`$AsyncState["Handlers"],
		(*THEN*)
		(*we can update the record separators here, as the socket has been submitted*)
		(
			(*check to make sure that the value for RecordSeparators is correct*)
			If[validRecSepQ[val],
				(*THEN*)
				(*we can update the recordseparators option*)
				(
					(*we don't want anything run while we are updating the association*)
					PreemptProtect[
						ZeroMQLink`Private`$AsyncState["Handlers",sockUUID,RecordSeparators] = val
					]
				),
				(*ELSE*)
				(*wrong polls*)
				(
					Message[System`SocketListener::invalidRecSep,val];
					$Failed
				)
			]
		),
		(*ELSE*)
		(*hasn't been submitted yet*)
		(
			Message[System`SocketListener::nosock,sock];
			$Failed
		)
	]
]

(*this function confirms that the value of RecordSeparators is usable*)
validRecSepQ[value_]:=Block[
	{
		emptyStream = StringToStream[""]
	},
	(
		(*None or {} means just whenever the OS feels like it for TCP sockets, or on every multipart part for ZMQ*)
		value === None || value === {} ||
		(*a positive Integer greater than 0 is the number of bytes to raise the event on*) 
		(IntegerQ[value] && value > 0) ||
		(*otherwise we need to check on an empty stream to ensure that no messages are returned*)
		Check[Quiet[Read[emptyStream, Record, RecordSeparators -> value],Read::opstlnone];Close[emptyStream];True,False]
	)
]


(*setter for the CharacterEncoding option for SocketListener objects*)
setRealSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,CharacterEncoding|"CharacterEncoding"->val_]:=Block[
	{
		sockUUID = First[sock]
	},
	If[KeyExistsQ[sockUUID]@ZeroMQLink`Private`$AsyncState["Handlers"],
		(*THEN*)
		(*we can update the record separators here, as the socket has been submitted*)
		(
			(*check to make sure that the value for CharacterEncoding is correct*)
			Which[
				MemberQ[CharacterEncoding]@$CharacterEncodings,
				(*valid setting for CharacterEncoding - use whatever was passed in*)
				(
					val = val;
				),
				CharacterEncoding === Automatic,
				(*use the default one for the system*)
				(
					val = $CharacterEncoding;
				),
				True,
				(*invalid character encoding option*)
				(
					Message[System`SocketListener::invalidRecSep,val];
					Return[$Failed]
				)
			];
			
			(*we don't want anything run while we are updating the association*)
			PreemptProtect[
				ZeroMQLink`Private`$AsyncState["Handlers",sockUUID,CharacterEncoding] = val
			]
		),
		(*ELSE*)
		(*hasn't been submitted yet*)
		(
			Message[System`SocketListener::nosock,sock];
			$Failed
		)
	]
]


(*setter for HandlerFunctions on a SocketListener object - valid keys for now are just "DataReceived"*)
setRealSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,HandlerFunctions|"HandlerFunctions"->(val_?AssociationQ)]:=Block[
	{
		sockUUID = First[sock],
		events = Keys[val],
		fixedHandlers
	},
	If[KeyExistsQ[sockUUID]@ZeroMQLink`Private`$AsyncState["Handlers"],
		(*THEN*)
		(*we can update the handler functions here, as the socket has been submitted*)
		(
			(*check to make sure that the value for HandlerFunctions is a valid Handler association*)
			If[Complement[events,{"DataReceived"}] === {},
				(*THEN*)
				(*all of the poll types are valid, turn them into the integer bitmask representation*)
				(
					(*we don't want anything run while we are updating the association*)
					PreemptProtect[
						ZeroMQLink`Private`$AsyncState["Handlers",sockUUID,HandlerFunctions] = val
					]
				),
				(*ELSE*)
				(*wrong polls*)
				(
					Message[System`SocketListener::pollError,Complement[events,{"DataReceived"}]];
					$Failed
				)
			]
		),
		(*ELSE*)
		(*hasn't been submitted yet*)
		(
			Message[System`SocketListener::nosock,sock];
			$Failed
		)
	]
];


confirmHandlerKeys[keys_]:=Block[
	{extraKeys},
	Which[
		MemberQ[{Default,Automatic},keys],
		(
			(*default keys*)
			{"Timestamp","Socket","SourceSocket","Data"}
		),
		MemberQ[{"Timestamp","Socket","SourceSocket","Data","DataBytes","DataByteArray","MultipartComplete"},keys],
		(
			(*key is a single string of one of the supported keys, so just wrap it in a list*)
			{keys}
		),
		!ListQ[keys],
		(
			(*invalid key spec - return Failure object*)
			Failure["HandlerFunctionKeys",<|"msg"->"invalidHandlerKeys","args"->{keys}|>]
		),
		(extraKeys = Complement[keys,{"Timestamp","Socket","SourceSocket","Data","DataBytes","DataByteArray","MultipartComplete"}]) =!= {},
		(
			Failure["HandlerFunctionKeys",<|"msg"->"invalidHandlerKeys","args"->extraKeys|>]
		),
		True,
		(
			(*keys is a list, and doesn't have extra keys in it - return it as is*)
			keys
		)
	]
]

(*setter for HandlerFunctionsKeys on a SocketListener object*)
setRealSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,HandlerFunctionsKeys|"HandlerFunctionsKeys"->val_]:=Block[
	{
		sockUUID = First[sock],
		handlerKeys
	},
	If[KeyExistsQ[sockUUID]@ZeroMQLink`Private`$AsyncState["Handlers"],
		(*THEN*)
		(*we can update the handler functions here, as the socket has been submitted*)
		(
		
			(*confirm the handler keys are valid*)
			handlerKeys = confirmHandlerKeys[val];
			If[FailureQ[handlerKeys],
				(*THEN*)
				(*there's something wrong with the handlerKeys, issue a message dependent on what the failure object returned*)
				(
					With[{msg = Last[handlerKeys]["msg"], args = Last[handlerKeys]["args"]}, Message[MessageName[SocketListen, msg],Sequence@@args]];
					Return[$Failed];
				)
			];
			
			(*we don't want anything run while we are updating the association*)
			PreemptProtect[
				ZeroMQLink`Private`$AsyncState["Handlers",sockUUID,HandlerFunctionsKeys] = handlerKeys
			];
		),
		(*ELSE*)
		(*hasn't been submitted yet*)
		(
			Message[System`SocketListen::nosock,sock];
			$Failed
		)
	]
];



setRealSocketListenerOpt[sock_?ZeroMQLink`Private`validSocketListenerQ,any__]:=Block[{},
	(
		Message[System`SocketListener::invalidOpt,any];
		$Failed
	)
]




End[] (* End Private Context *)

EndPackage[]