(* Wolfram Language Source File *)

BeginPackage["ZeroMQLink`",{"ZeroMQLink`SocketOptions`","ZeroMQLink`HostParser`","ZeroMQLink`Libraries`"}]

System`SocketConnect::addrspec = "The host specification `1` is wrong because `2`";
System`SocketConnect::noproto="The protocol specification `1` isn't supported.";

System`SocketOpen::addrspec = System`SocketConnect::addrspec;
System`SocketOpen::noproto=System`SocketConnect::noproto;

System`SocketReadMessage::again="Failed to receive message on socket `1` - try again (EAGAIN)";
System`SocketReadMessage::multipartOpt = "The option value for \"Multipart\", `1`, should be True or False";
System`SocketReadMessage::tcpServerNoAccess="The socket `1` is a server socket and cannot be read from or written \
to directly - use the property \"ConnectedClients\" to get client sockets that may be read from or written on."

System`SocketWaitNext::timeout="The value `1` for timeout is invalid, must be a non negative Integer in seconds, a Quantity, or Infinity";
System`SocketWaitNext::event="The event type `1` is invalid, valid event types are \"PollInput\", \"PollOutput\", and \"PollError\"";
System`SocketWaitNext::sock="SocketWaitNext only works on a list of valid SocketObjects."
System`SocketWaitNext::tcpServerSockPoll = "The socket `1` is a TCP server socket and cannot be polled - use the property \"ConnectedClients\" to get client sockets that may be polled."

System`SocketWaitAll::event = System`SocketWaitNext::event;
System`SocketWaitAll::timeout = System`SocketWaitNext::timeout;
System`SocketWaitAll::sock = "SocketWaitAll only works on a list of valid SocketObjects."
System`SocketWaitAll::tcpServerSockPoll = System`SocketWaitNext::tcpServerSockPoll

System`SocketReadyQ::event = System`SocketWaitNext::event;
System`SocketReadyQ::timeout = System`SocketWaitNext::timeout;
System`SocketReadyQ::sock = "SocketReadyQ only works on valid SocketObjects."
System`SocketReadyQ::tcpServerSockPoll = System`SocketWaitNext::tcpServerSockPoll

System`SocketListen::addrspec = System`SocketConnect::addrspec;
System`SocketListen::error = "An error occured";
System`SocketListen::pollError = "The handlers `1` are invalid, valid HandlerFunctions are {\"DataReceived\"}";
System`SocketListen::invalidHandlerKeys = "The HandlerFunctionsKeys specification `1` is invalid, \
possible HandlerFunctionKeys values are any subset of {\"Socket\",\"SourceSocket\",\"Timestamp\",\"Data\",\"DataBytes\",\"DataByteArray\",\"MultipartComplete\"}.";

System`SocketListener::invalidRecSep = "The option setting `1` for RecordSeparators is invalid - valid options are integers greater than 0 or any string or list of strings.";
System`SocketListener::nosock = "The socket `1` doesn't have any handler functions - use SocketListen to listen on the socket.";
System`SocketListener::invalidOpt = "The socket listener option `1` is invalid.";
System`SocketListener::invalidSock = "The socket listener `1` is invalid."

System`SocketObject::type = "The socket type `1` is invalid, valid types are :`2`";
System`SocketObject::invalidSetOptionValue="The value for option `1`, `2` is invalid";
System`SocketObject::invalidSetOption="The option `1` is not a valid option to set for the socket";
System`SocketObject::endpoint="Failed to determine endpoint of socket and can't close it";
System`SocketObject::invalidSock="The socket `1` is invalid and not open";
System`SocketObject::clientSock="The socket `1` is not a server socket and doesn't have clients";
System`SocketObject::tcpServerNoAccess = System`SocketReadMessage::tcpServerNoAccess;

System`Sockets::spec = "The socket specification `1` is invalid - valid types are `2`.";


OpenRead::invalidSock = System`SocketObject::invalidSock;
Read::invalidSock = System`SocketObject::invalidSock;
ReadLine::invalidSock = System`SocketObject::invalidSock;
ReadString::invalidSock = System`SocketObject::invalidSock;
ReadByteArray::invalidSock = System`SocketObject::invalidSock;
ReadList::invalidSock = System`SocketObject::invalidSock;
OpenWrite::invalidSock = System`SocketObject::invalidSock;
Write::invalidSock = System`SocketObject::invalidSock;
WriteString::invalidSock = System`SocketObject::invalidSock;
WriteLine::invalidSock = System`SocketObject::invalidSock;

System`SocketObject::zmqexception = "A `1` exception was thrown - `3` (code `2`)";
Options::zmqexception = System`SocketObject::zmqexception;
SetOptions::zmqexception = System`SocketObject::zmqexception;
Write::zmqexception = System`SocketObject::zmqexception;
System`SocketConnect::zmqexception = System`SocketObject::zmqexception;
System`SocketOpen::zmqexception = System`SocketObject::zmqexception;
Close::zmqexception = System`SocketObject::zmqexception;
System`SocketReadMessage::zmqexception = System`SocketObject::zmqexception;
System`SocketReadyQ::zmqexception = System`SocketObject::zmqexception;
System`SocketWaitNext::zmqexception = System`SocketObject::zmqexception;
System`SocketWaitAll::zmqexception = System`SocketObject::zmqexception;
System`SocketListen::zmqexception = System`SocketObject::zmqexception;

Write::zmqagain = "The socket `1` is not ready, try again later.";
DeleteObject::listenerError = "An error occured while deleting `1`."
Close::closedsocket = "The socket `1` is not open.";

Begin["`Private`"]

$MaxThreads::zmqexception = System`SocketObject::zmqexception;

$yieldingTask = False;

(*this is necessary to prevent recursive autoloading of the System` symbols that the paclet manager sets up for us*)
(Unprotect[#];
ClearAttributes[#,{Stub,Protected,ReadProtected}];
Clear[#];)&/@{
	"System`SocketConnect",
	"System`SocketObject",
	"System`SocketReadMessage",
	"System`SocketOpen",
	"System`SocketListen",
	"System`SocketListener",
	"System`SocketWaitAll",
	"System`SocketWaitNext",
	"System`SocketReadyQ",
	"System`Sockets"
}

(*summary box for SocketObject*)
System`SocketObject/:MakeBoxes[sock:System`SocketObject[uuid_String]/;AtomQ[Unevaluated @ uuid] && KeyExistsQ[uuid]@$Sockets,StandardForm|TraditionalForm]:=
	With[
		{assoc=$Sockets[uuid]},
		BoxForm`ArrangeSummaryBox[
			(*first argument is the head to use*)
			System`SocketObject,
			(*second argument is the expression*)
			System`SocketObject[uuid],
			(*third argument is the icon to use*)
			Show[$SocketIcon,ImageSize->{Automatic,Dynamic[3.5*CurrentValue["FontCapHeight"]]}],
			(*the next argument is the properties visible when the plus button isn't pressed*)
			{
				{BoxForm`SummaryItem[{"IPAddress: ",First@assoc["DestinationIPAddress"]}],BoxForm`SummaryItem[{"Port: ",assoc["DestinationPort"]}]},
				{BoxForm`SummaryItem[{"UUID: ",uuid}],BoxForm`SummaryItem[{"Protocol: ",sock["Protocol"]}]}
			},
			(*the next argument is the optional items that come down when the plus button is pressed - note this completely*)
			(*replaces the above list because we set "CompleteReplacement"->True *)
			KeyValueMap[
				BoxForm`SummaryItem[{ToString[#1]<>": ",#2}]&,
				Append[
					KeyTake[{"DestinationIPAddress","DestinationPort","UUID","SourceIPAddress","SourcePort"}]@assoc,
					"Protocol"->sock["Protocol"]
				]
			],
			(*lastly,the display form we want to display this with*)
			StandardForm,
			(*to make things look better, we push the entire thing down to a single column when the plus button is pushed*)
			"CompleteReplacement"->True
		]
	];

$AsyncState = <|
	"Status":>(
		If[Not@MissingQ@$AsyncState["Error"],
			"Error",
			MemberQ[AsynchronousTasks[],$AsyncState["Task"]]/.{True->"Running",False->"Not running"}
		]
	),
	"Handlers"-><||>,
	"ZMQSocketBuffers"-><||>,
	"TCPClients"-><||>,
	"TCPClientBuffers"-><||>
|>;

$Sockets = <||>;

wlHeuristicSocketTypeNames[type_?StringQ]:=(ToUpperCase[type]/.
	<|
		"ZMQ_PUBLISH"->"ZMQ_PUB",
		"ZMQ_SUBSCRIBE"->"ZMQ_SUB",
		"ZMQ_REQUEST"->"ZMQ_REQ",
		"ZMQ_REPLY"->"ZMQ_REP",
		"ZMQ_XPUBLISH"->"ZMQ_XPUB",
		"ZMQ_XSUBSCRIBE"->"ZMQ_XSUB"
	|>
);

wlSocketTypeToZMQSocketType[type_?StringQ]:=If[!StringMatchQ[type,StartOfString~~"ZMQ_"~~__,IgnoreCase->True],
	(*THEN*)
	(*prepent the ZMQ_ prefix to make it into a recognized socket type*)
	"ZMQ_"<>ToUpperCase[type],
	(*ELSE*)
	(*don't need to prepend the prefix, we can just make it upper case*)
	ToUpperCase[type]
]

socketTypes = <|
	(*types from zmq.h*)
	"ZMQ_PAIR"->0,
	"ZMQ_PUB"->1,
	"ZMQ_SUB"->2,
	"ZMQ_REQ"->3,
	"ZMQ_REP"->4,
	"ZMQ_DEALER"->5,
	"ZMQ_ROUTER"->6,
	"ZMQ_PULL"->7,
	"ZMQ_PUSH"->8,
	"ZMQ_XPUB"->9,
	"ZMQ_XSUB"->10,
	"ZMQ_STREAM"->11
	(*draft socket types - note disabled in release*)	
(*	"ZMQ_CLIENT"->13,
	"ZMQ_SERVER"->12,
	"ZMQ_RADIO"->14,
	"ZMQ_DISH"->15*)
|>;


(*this lets one control the number of threads by just assigning to a variable*)
$MaxThreads /: HoldPattern[Set][$MaxThreads,val_]:=checkRes[iSetZMQContextOption[$DefaultContext,1,val],$MaxThreads];
$MaxThreads := If[ListQ@#,First@#,#]&@checkRes[iGetZMQContextOption[$DefaultContext,1],$MaxThreads];

(*initialize the libraries, only if we haven't already assigned to $DefaultContext*)
If[Not@ValueQ@$DefaultContext,
	initializeLibraries[];
	
	(*define the input stream method for ZMQ that we can use for functionality with Read/ReadString/ReadList/BinaryRead/BinaryReadList*)
	DefineInputStreamMethod["ZMQSocket",
		{
			(*the constructor returns the uuid for the socket as the state*)
			(*the uuid is the name of the socket*)
			"ConstructorFunction" -> (With[{uuid=StringDrop[#1,3]},{KeyExistsQ[uuid]@$Sockets,uuid}]&),
			
			(*close function calls zmq close with the socket*)
			"CloseFunction" -> (closeZMQInStream[System`SocketObject[#]]&),
			
			(*the internal reading function will internally maintain each message as a buffer*)
			(*read function will never queue a new message from the socket, only returning bytes from the current message, the stream*)
			(*implementation knows that if this method returns an empty list it will call the wait for input function to queue up more messages*)
			"ReadFunction" -> ({zmqSocketReadStreamBytes[System`SocketObject[#1],#2],#1}&),
			
			(*the wait function will actually read a new message and updates the buffer with a new message*)
			"WaitForInputFunction" -> ({zmqSocketBufferStreamMessage[System`SocketObject[#1]],#1}&),
			
			(*we don't allow seeking through it so we don't have to keep all the socket messages in memory automatically, just the most recently queued one*)
			"SeekableQFunction" -> ({False,#1}&),
			
			(*we "never" hit the end of file on a zmq socket*)
			"EndOfFileQFunction" -> ({False,#1}&)
		}
	];
	
	(*also define the input stream method for TCP sockets that we can use for functionality with Read/ReadString/ReadList/BinaryRead/BinaryReadList*)
	DefineInputStreamMethod["TCPSocket",
		{
			(*the constructor returns the uuid for the socket as the state*)
			(*the uuid is the name of the socket*)
			(*this also calls the internal function to create the stream in C code*)
			"ConstructorFunction" -> (With[{uuid=StringDrop[#1,3]},{KeyExistsQ[uuid]@$Sockets&&checkRes[iTCPSocketInStreamCreate[uuid],OpenRead]=!=$Failed,uuid}]&),
			
			(*the close function calls the Close_SocketObject*)
			"CloseFunction" -> (checkRes[iTCPSocketInStreamClose[#1],Close]&),
			
			(*the read function doesn't block and ignores how many bytes were requested - it just returns as many as it can find*)
			"ReadFunction" -> (
				(*SocketReadMessage with "Blocking" false will read as many bytes as it can without blocking*)
				(*it does this by looking up how many bytes there are, then reading exactly that many*)
				(*it also returns a byte array, which we have to turn into a list to work with the InputStream framework*)
				{Normal[System`SocketReadMessage[System`SocketObject[#1],"Blocking"->False]],#1}&
			),
			
			(*the wait function calls the internal function*)
			"WaitForInputFunction" -> ({checkRes[iTCPSocketInStreamWaitForInput[#1],Read],#1}&),
			
			(*we don't allow seeking through the underlying stream*)
			"SeekableQFunction" -> ({False,#1}&),
			
			(*call the internal function*)
			"StreamPositionFunction"->({checkRes[iTCPSocketInStreamPosition[#1],Read],#1}&),
			
			(*call the internal function*)
			"StreamSizeFunction"->({checkRes[iTCPSocketInStreamSize[#1],Read],#1}&),
			
			(*call the internal function to get the error text function*)
			"ErrorTextFunction"->({checkRes[iTCPSocketInErrorText[#1],Read]/.{Null->{},bytes_:>FromCharacterCode[bytes]},#1}&),
			
			(*call the internal function to clear the error*)
			"ClearErrorFunction"->({checkRes[iTCPSocketInClearError[#1],Read],#1}&),
			
			(*call the internal function*)
			"EndOfFileQFunction" -> ({First[checkRes[iTCPSocketInEndOfFileQ[#1],Read]]==1,#1}&)
		}
	];

	DefineInputStreamMethod["SocketByteList", 
		{
			"ConstructorFunction" -> Function[{name, caller, opts}, 
				Block[{bytes},
					bytes = Lookup[opts, "Bytes"];
					If[!MissingQ[bytes] && 
						(ByteArrayQ[bytes] || (ListQ[bytes] && ByteArrayQ[ByteArray[bytes]])), 
						{True, {0, bytes}}, 
						{False, $Failed}
					]
				]
			],

			"ReadFunction" ->Function[{state, n},
				Block[{pos = state[[1]], bytes = state[[2]], bytesRead},
					If[pos >= Length[bytes],
						(*THEN*)
						(*at the end of the buffer can't read anything else*)
						{{}, state},
						(*ELSE*)
						(*take part of the bytes and return those*)
						(
							bytesRead = Part[bytes, pos + 1 ;; UpTo[pos + n]];
							{Normal@bytesRead, {pos + Length[bytesRead], bytes}}
						)
					]
				]
			],

			(*seek function just sets the position to be whatever was passed in*)
			"SeekFunction" -> Function[{state,newpos},
				{newpos > 0 && newpos <= Length[Last[state]], {newpos,Last[state]}}
			],

			(*stream size function is the length of the bytes*)
			"StreamSizeFunction" -> Function[{state},
				{Length[Last[state]], state}
			],

			(*to test if we're at the end of the file, we just see if the position is greater than or*)
			(*equal to the length of the bytes*)
			"EndOfFileQFunction" -> ({#[[1]] >= Length[#[[2]]], #} &)
		}
	];
	
	
	(*now we do the same method and function definition for OutputStreams*)
	DefineOutputStreamMethod["ZMQSocket",
		{
			(*the constructor returns the uuid for the socket as the state*)
			(*the uuid is the name of the socket*)
			(*note that for uniqueness - the output stream is named "out-[[UUID]]"*)
			"ConstructorFunction" -> (With[{uuid=StringDrop[#1,4]},{KeyExistsQ[uuid]@$Sockets,uuid}]&),
			
			(*close function calls zmq close with the socket*)
			"CloseFunction" -> (closeZMQOutStream[System`SocketObject[#]]&),
			
			(*the write function just takes a basic list of bytes to write out and sends them with an internal function for lists of bytes*)
			(*with zmq, we have no way of telling whether or not the write succeeded, so we just ignore that and return the length of bytes provided*)
			"WriteFunction" -> ((
						Internal`StuffBag[$Sockets[#1,"StreamState","OutMessageBuffer"],#2];
						zmqFlushBytes[System`SocketObject[#1]];
						{Length[#2],#1}
					)&),
	
			(*flush is what actually releases the bytes to the stream - note we have to get the bytes out of the bag with *)
			"FlushFunction" -> ({zmqFlushBytes[System`SocketObject[#1]],#1}&)
		}
	];
	
	(*also define the output stream method for TCP sockets*)
	DefineOutputStreamMethod["TCPSocket",
		{
			(*the constructor returns the uuid for the socket as the state*)
			(*the uuid is the name of the socket*)
			(*this also calls the internal function to create the stream in C code*)
			(*note that for uniqueness - the output stream is named "out-[[UUID]]", so we have to delete that part of the string*)
			"ConstructorFunction" -> (With[{uuid=StringDrop[#1,4]},{KeyExistsQ[uuid]@$Sockets&&checkRes[iTCPSocketOutStreamCreate[uuid],OpenWrite]=!=$Failed,uuid}]&),
			
			(*the close function calls the Close_SocketObject*)
			"CloseFunction" -> (checkRes[iTCPSocketOutStreamClose[#1],Close]&),
			
			(*the write function writes out the function *)
			"WriteFunction" -> ({checkRes[iTCPSocketOutStreamWrite[#1,#2],Write]/.{a_List:>First[a]},#1}&),
			
			(*call the internal function to get the error text function*)
			"ErrorTextFunction"->({checkRes[iTCPSocketOutErrorText[#1],System`SocketObject],#1}&),
			
			(*call the internal function to clear the error*)
			"ClearErrorFunction"->({checkRes[iTCPSocketOutClearError[#1],System`SocketObject],#1}&),
			
			(*call the internal function to get the stream position*)
			"StreamPositionFunction"->({checkRes[iTCPSocketOutStreamPosition[#1],System`SocketObject],#1}&)
		}
	];
	
]

(*this sets up the appropriate handling for OpenRead, which is a seperate function*)
System`SocketObject /: HoldPattern[OpenRead][sock:System`SocketObject[_],rest___] := socketInStreamOpen[sock,rest];


(*set upvalues on SocketObject to work with the existing stream functions for reading from the socket*)
(System`SocketObject /: HoldPattern[#][sock:System`SocketObject[_],rest___] := streamReadFuncWrapper[#,sock,rest])&/@
{
	Read,
	ReadString,
	ReadByteArray,
	ReadList,
	BinaryRead,
	BinaryReadList,
	ReadLine
};


(*this sets up the appropriate handling for OpenWrite, which is a seperate function*)
System`SocketObject /: HoldPattern[OpenWrite][sock:System`SocketObject[_],rest___] := socketOutStreamOpen[sock,rest];

(*set upvalues on SocketObject to work with the existing stream functions for reading from the socket*)
(System`SocketObject /: HoldPattern[#][sock:System`SocketObject[_],rest___] := streamWriteFuncWrapper[#,sock,rest])&/@
{
	Write,
	WriteString,
	BinaryWrite,
	WriteLine
};

(*up values to Close the socket - we need a special handler for this to close all the streams*)
System`SocketObject /: HoldPattern[Close][sock:System`SocketObject[_],rest___] := CloseSocket[sock,rest];

(*form rules for setting options*)
(*property querying*)
System`SocketObject /: HoldPattern[Set][sock:System`SocketObject[_][opt_],rest___] := setSockOpt[sock,opt->rest]
System`SocketListener /: HoldPattern[Set][sock:System`SocketListener[_][opt_],rest___] := setSocketListenerOpt[sock,opt->rest]

(*property querying directly*)
System`SocketObject /: HoldPattern[sock:System`SocketObject[_]][rest___] := getSockOpts[sock,rest]
System`SocketListener /: HoldPattern[sock:System`SocketListener[_]][rest___] := getSocketListenerOpts[sock,rest]

(*Options property querying*)
System`SocketObject /: HoldPattern[Options][sock:System`SocketObject[_],rest___] := getSockOpts[sock,rest]
System`SocketListener /: HoldPattern[Options][sock:System`SocketListener[_],rest___] := getSocketListenerOpts[sock,rest]

(*SetOptions property querying*)
System`SocketObject /: HoldPattern[SetOptions][sock:System`SocketObject[_],rest___] := setSockOpt[sock,rest]
System`SocketListener /: HoldPattern[SetOptions][sock:System`SocketListener[_],rest___] := setSocketListenerOpt[sock,rest]


(*UpValue for DeleteObject to work with SocketListener*)
Unprotect[DeleteObject]
System`SocketListener /: HoldPattern[DeleteObject][sockListener:System`SocketListener[_],rest___]:=closeSocketListener[sockListener]
Protect[DeleteObject]

(*helper function for assigning upvalues*)
(*this will first check if the input stream is open, then calls the original function on the underlying stream*)
streamReadFuncWrapper[symbol_,sock_,rest___]:=Block[{stream, uuid = First[sock]},
	(
		stream = If[KeyExistsQ[uuid]@$Sockets && !MemberQ[Streams[],$Sockets[uuid,"StreamState","InputStream"]],
			(*THEN*)
			(*we need to open up the stream with OpenRead - note we don't need any special options, just use defaults*)
			OpenRead[sock],
			(*ELSE*)
			(*already opened, just return the stream*)
			$Sockets[First[sock],"StreamState","InputStream"]
		];
		Which[
			FailureQ[stream] || MissingQ[stream],
			(
				$Failed
			),
			KeyExistsQ[uuid]@$Sockets,
			(*THEN*)
			(*socket exists and we can continue*)
			symbol[stream,rest],
			True,
			(*socket doesn't exist, raise message and issue failed*)
			(
				With[{s=symbol},Message[MessageName[s,"invalidSock"],sock]];
				$Failed
			)
		]
	)
]


(*helper function for setting upvalues for writing stream functions*)
streamWriteFuncWrapper[symbol_,sock_,rest___]:=Block[{stream, uuid = First[sock]},
	(
		stream = If[KeyExistsQ[uuid]@$Sockets && !MemberQ[Streams[],$Sockets[uuid,"StreamState","OutputStream"]],
			(*THEN*)
			(*we need to open up the stream with OpenWrite - note we don't need any special options, just use defaults*)
			OpenWrite[sock],
			(*ELSE*)
			(*already opened, just return the stream*)
			$Sockets[First[sock],"StreamState","OutputStream"]
		];
		Which[
			FailureQ[stream] || MissingQ[stream],
			(
				$Failed
			),
			KeyExistsQ[uuid]@$Sockets,
			(*THEN*)
			(*socket exists and we can continue*)
			symbol[stream,rest],
			True,
			(*socket doesn't exist, raise message and issue failed*)
			(
				With[{s=symbol},Message[MessageName[s,"invalidSock"],sock]];
				$Failed
			)
		]
	)
];

(*Sockets returns a list of socket objects that are currently in use*)

$socketFilterSpecs = {"Local","Remote","Client","Server","TCP","ZMQ"}

(*remote sockets are all shadowTCPClientSocketQ sockets*)
sockUUIDs["Remote"] := Keys@KeySelect[shadowTCPClientSocketQ[System`SocketObject[#]]&]@$Sockets

(*client sockets are all sockets that have Direction of "Client", and aren't a shadow tcp socket*)
sockUUIDs["Client"] := Complement[Keys@Select[#["DirectionType"]==="Client"&]@$Sockets,sockUUIDs["Remote"]]

(*for the main protocol type, pretty straight foward*)
sockUUIDs["ZMQ"] := Keys@Select[#["Protocol"]==="ZMQ"&]@$Sockets
sockUUIDs["TCP"] := Keys@Select[#["Protocol"]==="TCP"&]@$Sockets

(*server sockets are all sockets that have Direction of "Server"*)
sockUUIDs["Server"] := Keys@Select[#["DirectionType"]==="Server"&]@$Sockets

(*local sockets are all sockets except remote*)
sockUUIDs["Local"] := Complement[Keys@$Sockets, sockUUIDs["Remote"]]

sockUUIDs["TCPServer"] := Keys@Select[KeyExistsQ["ZMQSocket"]]@$Sockets

sockUUIDs["ZMQStreamHidden"] := Values[$Sockets[[sockUUIDs["TCPServer"],"ZMQSocket"]]]

sockUUIDs["Inactive"] := Keys@Select[TrueQ[Not[#["Active"]]] &]@$Sockets

(*remove hidden sockets will remove the sockets used for communication with the background thread, as well as the zmq_stream sockets that are used to implement tcp server sockets*)
removeHiddenSockets[sockList_] := Complement[sockList,Join[sockUUIDs["Inactive"],sockUUIDs["ZMQStreamHidden"],First/@Values@KeyTake[{"Incoming","Outgoing"}]@$AsyncState]]

System`Sockets[] := System`Sockets["Local"]

System`Sockets[All] := Join[System`Sockets["Local"],System`Sockets["Remote"]]

System`Sockets[spec_]/;MemberQ[spec]@$socketFilterSpecs := System`SocketObject /@ removeHiddenSockets[sockUUIDs[spec]]

(*special form for zmq type strings*)
System`Sockets[zmqType_]/;KeyExistsQ[wlHeuristicSocketTypeNames[zmqType]]@socketTypes := 
	System`SocketObject/@removeHiddenSockets[Keys@Select[#["Protocol"]==="ZMQ" && ToUpperCase[#["Type"]]===ToUpperCase[wlHeuristicSocketTypeNames[zmqType]]&]@$Sockets]

System`Sockets[any__]:= (Message[System`Sockets::spec,any, Join[$socketFilterSpecs,Keys[wlHeuristicSocketTypeNames/@socketTypes]]]; $Failed)

(**********************************************************************)
(***********************              *********************************)
(*********************** READ METHODS *********************************)
(***********************              *********************************)
(**********************************************************************)

(*stream destructor*)
(*this method will just empty out the buffer from memory and return*)
closeZMQInStream[sock_?validSocketObjectQ,rest___]:=Block[
	{
		sockUUID = First[sock]
	},
	(
		KeyDropFrom[$Sockets[sockUUID,"StreamState"],"InMessageBuffer"];
	)
];


(*actual read bytes function for "ReadFunction"*)
(*this will return up to the number of bytes from the current buffered message*)
zmqSocketReadStreamBytes[sock_?validSocketObjectQ,numBytes_]:=Block[
	{
		sockUUID = First[sock],
		res
	},
	(
		{res, $Sockets[sockUUID,"StreamState","InMessageBuffer"]} = TakeDrop[$Sockets[sockUUID,"StreamState","InMessageBuffer"], UpTo[numBytes]];
		res
	)
]

(*buffering function for "WaitForInputFunction"*)
(*this will replace the current message in the buffer for the stream with a new one fresh off the socket*)
zmqSocketBufferStreamMessage[sock_?validSocketObjectQ]:=Block[{},
	(
		(*this reads another message from the sock and stores it in the message buffer*)
		$Sockets[First[sock],"StreamState","InMessageBuffer"] = Normal[System`SocketReadMessage[sock]]
	)
]




(**********************************************************************)
(***********************               ********************************)
(*********************** WRITE METHODS ********************************)
(***********************               ********************************)
(**********************************************************************)




(*this method just deletes the Internal`Bag by assigning it to Null*)
closeZMQOutStream[sock_?validSocketObjectQ,rest___]:=Block[{},
	(
		$Sockets[First[sock],"StreamState","OutMessageBuffer"] = Null
	)
];


(*flush function that actually sends out the bytes on the wire*)
zmqFlushBytes[sock_?validSocketObjectQ]:=Block[
	{
		sockUUID = First[sock],
		bytes
	},
	(
		(*we join together all of the individual bag parts*)
		bytes = ByteArray[Join@@Internal`BagPart[$Sockets[sockUUID,"StreamState","OutMessageBuffer"],All]];
		(* Avoid sending an empty list accidentally, it would close the socket. *)
		If[bytes === {}, Return[]];
		(*Send the bytes*)
		ZMQWriteInternal[sock, bytes];
		(*now clear out the bag with a new one*)
		$Sockets[sockUUID,"StreamState","OutMessageBuffer"] = Internal`Bag[];
	)
];


(*actual writing bytes on the socket - called from the "FlushFunction"*)
Options[ZMQWriteInternal] = {"Asynchronous"->False,"Multipart"->False}

ZMQWriteInternal[sock_, data_List, rest___]:= ZMQWriteInternal[sock, ByteArray[data], rest];

ZMQWriteInternal[sock_?validSocketObjectQ, data_ByteArray,OptionsPattern[]] := Block[
	{
		uuid = First[sock],
		(*the flags are a bit Or'ing of the individual options*)
		flags = BitOr[
			TrueQ@OptionValue["Asynchronous"]/.{True->1,False->0},
			TrueQ@OptionValue["Multipart"]/.{True->2,False->0}
		],
		res,
		servSockUUID
	},
	(
		(*note that we need to check if this socket object is a shadow tcp socket, in which case we handle it slightly differently*)
		If[shadowTCPClientSocketQ[sock],
			(*THEN*)
			(*we need to write the identity of the client on the base socket first*)
			(
				(*get the zmq socket for this shadow client*)
				servSockUUID = $Sockets[uuid,"ZMQSocket"];
				
				If[MissingQ[servSockUUID],
					(*THEN*)
					(*this socket doesn't exist somehow but is still a valid shadow tcp socket?*)
					(
						$Failed
					),
					(*ELSE*)
					(*then the socket does exist and is specified as a Key[uuid]*)
					(
						(*first write out the identity as a multipart on the server socket*)
						(*the identity is given in the async state assoc by first the server socket - note the Part so we can use the Key[...] as is*)
						(*the the client socket uuid which then goes to the identity as a list of bytes*)
						ZMQWriteInternal[System`SocketObject[servSockUUID],$AsyncState["TCPClients",servSockUUID,uuid],"Multipart"->True];
						
						(*now write the data specified in this request, which ends the message*)
						ZMQWriteInternal[System`SocketObject[servSockUUID],data,"Multipart"->False]
					)
				]
			),
			(*ELSE*)
			(*send the message on the socket normally*)
			(
				(*the third parameter is whether we should loop waiting until the send goes through*)
				(*but if a user requests asynchronous, then we shouldn't do that and just call it once*)
				(*but the logic for whether we loop or not is inverted - if we're async, then we don't loop, and vice versa*)
				res = checkRes[iSendBytesToSocket[uuid,data,Not[TrueQ@OptionValue["Asynchronous"]],flags],Write];
				If[res === $Failed,
					(*THEN*)
					(*failed to send data - but message was already raised*)
					(
						$Failed
					),
					(*ELSE*)
					(*worked to send, check the second element to see if the data was actually sent*)
					(
						If[First[res]===0,
							(*THEN*)
							(*failed to actually send, but didn't error out*)
							(
								Message[Write::zmqagain,sock];
								$Failed
							)
							(*ELSE*)
							(*worked fine - don't return anything same as Write / WriteString*)
						]
					)
				]
			)
		]
	)
];

(* We need a way to efficiently write messages to a ZMQ socket without the heavy machinery DefineOutputStreamMethod.
It appears that as of now (02/2018), DefineOutputStreamMethod is reading the stream once and build a packed array with 
the data read, even if the input was a bytearray. The packed array is later passed to the user defined callback functions.
This is not good enough performance wise for many ZMQ application where fast response is important, e.g. ExternalEvaluate.
For now, we create an alias for ZMQWriteInternal as a PacletExport function called ZMQSocketWriteMessage.
*)
Options[ZeroMQLink`ZMQSocketWriteMessage] = {"Asynchronous"->False,"Multipart"->False};

GeneralUtilities`SetUsage[ZeroMQLink`ZMQSocketWriteMessage, "ZMQSocketWriteMessage[socket$, bytearray$] write the data in bytearray$ to the ZMQ socket socket$"];

ZeroMQLink`ZMQSocketWriteMessage[sock_?validSocketObjectQ, data_ByteArray, opts:OptionsPattern[]] := ZMQWriteInternal[sock, data, opts];


(**********************************************************************)
(***********************               ********************************)
(*********************** OPEN METHODS  ********************************)
(***********************               ********************************)
(**********************************************************************)

(*association version for options specified as an association or with rules/rule delayed*)
socketOutStreamOpen[sock_?validSocketObjectQ,opts:(OptionsPattern[] | _?AssociationQ)] := Block[
	{
		sockUUID = First[sock],
		sockStreamName="out-"<>First[sock],
		fixedOpts,
		res
	},
	(*we need to remove the Method/BinaryFormat rule if it was specified, so that we always use the appropriate input stream method of ZMQ*)
	(*same thing for BinaryFormat, because the underlying socket could in fact be of a different binary format*)
	(*additionally, for ouputting, we don't want to insert any line or page breaks, so set the page height and width to be infinity*)
	res = Which[
		shadowTCPServerSocketQ[sock],
		(*can't write out on a server socket - don't know which client to write to*)
		(*so we simply return $Failed here*)
		(
			Message[System`SocketObject::tcpServerNoAccess,sock];
			Return[$Failed]
		),
		$Sockets[sockUUID,"Protocol"] === "ZMQ" || shadowTCPClientSocketQ[sock],
		(*both normal zmq sockets and the shadow client socks write out with the zmq stream functions *)
		fixedOpts = Sequence@@Normal[KeyDrop[Association[opts], {Method, BinaryFormat, PageWidth, PageHeight}]];
		OpenWrite[
			sockStreamName,
			fixedOpts,
			Method->"ZMQSocket",
			BinaryFormat->True,
			PageWidth->Infinity,
			PageHeight->Infinity
		],
		$Sockets[sockUUID,"Protocol"] === "TCP",
		(*tcp client sockets use the tcp socket method like normal*)
		fixedOpts = Sequence@@Normal[KeyDrop[Association[opts], {Method, BinaryFormat}]];
		OpenWrite[
			sockStreamName,
			fixedOpts,
			Method->"TCPSocket",
			BinaryFormat->True
		]
	];
	
	(*only save the stream if it works*)
	If[MatchQ[res,_OutputStream],
		$Sockets[sockUUID,"StreamState","OutputStream"] = res
	];
	res
];


(*anything else that doesn't match issue message - 1 for valid socket, but invalid options, and another for invalid everything*)
socketOutStreamOpen[sock_?validSocketObjectQ,invalid__]:=(
	Message[System`SocketObject::nonopt,invalid];
	$Failed
)

socketOutStreamOpen[sock_System`SocketObject,___]:=(
	Message[System`SocketObject::invalidSock,sock];
	$Failed
)

socketOutStreamOpen[___]:=(
	Message[System`SocketObject::args,OpenWrite];
	$Failed
)

(*normal version with option spec of rules/rule delayed*)
socketInStreamOpen[sock_?validSocketObjectQ,opts:(OptionsPattern[] | _?AssociationQ)]:=Block[
	{
		sockUUID = First[sock],
		sockStreamName = "in-"<>First[sock],
		fixedOpts = Sequence@@Normal[KeyDrop[Association[opts], {Method, BinaryFormat}]],
		res
	},
	(*we need to remove the Method/BinaryFormat rule if it was specified, so that we always use the appropriate input stream method of ZMQ*)
	(*same thing for BinaryFormat, because the underlying socket could in fact be of a different binary format*)
	(*additionally, for ouputting, we don't want to insert any line or page breaks, so set the page height and width to be infinity*)
	res = Which[
		shadowTCPServerSocketQ[sock],
		(*can't write out on a server socket - don't know which client to write to*)
		(*so we simply return $Failed here*)
		(
			Message[System`SocketObject::tcpServerNoAccess,sock];
			Return[$Failed]
		),
		$Sockets[sockUUID,"Protocol"] === "ZMQ" || shadowTCPClientSocketQ[sock],
		OpenRead[sockStreamName,fixedOpts,Method->"ZMQSocket",BinaryFormat->True],
		$Sockets[sockUUID,"Protocol"] === "TCP",
		OpenRead[sockStreamName,fixedOpts,Method->"TCPSocket",BinaryFormat->True]
	];
	
	(*only save the stream if it works*)
	If[MatchQ[res,_InputStream],
		$Sockets[sockUUID,"StreamState","InputStream"] = res
	];
	res
];

(*anything else that doesn't match issue message - 1 for valid socket, but invalid options, and another for invalid everything*)
socketInStreamOpen[sock_?validSocketObjectQ,invalid__]:=(
	Message[System`SocketObject::nonopt,invalid];
	$Failed
)
socketInStreamOpen[___]:=(
	Message[System`SocketObject::args,OpenRead];
	$Failed
)

getShadowTCPClientSocketAddressDetails[zmqServerSocketUUID_?StringQ,identity_List,sym_] := Block[
	{
		assoc = <||>,
		ip,
		port,
		res
	},
	(
		(*get the source and destination addresses - 0 is the source code*)
		res = checkRes[iShadowTCPClientSocketGetAddress[zmqServerSocketUUID,0,identity],sym];
		If[!FailureQ[res],
			(
				{ip,port} = StringSplit[FromCharacterCode[DeleteCases[res,0]],","];
				AppendTo[assoc,"SourceIPAddress"->IPAddress[ip]];
				AppendTo[assoc,"SourcePort"->port];
			),
			(*ELSE*)
			(*failed, so return $Failed*)
			(
				Return[$Failed]
			)
		];
		
		(*now do same for the destination addresses - 1 is the destination code*)
		res = checkRes[iShadowTCPClientSocketGetAddress[zmqServerSocketUUID,1,identity],sym];
		If[!FailureQ[res],
			(*THEN*)
			(*got the result, parse it out and add it to the association*)
			(
				{ip,port} = StringSplit[FromCharacterCode[DeleteCases[res,0]],","];
				AppendTo[assoc,"DestinationIPAddress"->IPAddress[ip]];
				AppendTo[assoc,"DestinationPort"->port];
			),
			(*ELSE*)
			(*failed, so return $Failed*)
			(
				Return[$Failed]
			)
		];
		
		(*return association of socket address details*)
		assoc
	)
]

getSocketAddressDetails[socketUUID_?StringQ,type_?StringQ,sym_] := Block[
	{
		(*type codes for socket types are "TCP" -> 1, "ZMQ" -> 0*)
		intType = Boole[type === "TCP"],
		assoc = <||>,
		ip,
		port,
		res
	},
	(
		(*get the source and destination addresses - 0 is the source code*)
		res = checkRes[iSocketGetAddress[socketUUID,0,intType],sym];
		If[!FailureQ[res],
			(
				{ip,port} = StringSplit[FromCharacterCode[DeleteCases[res,0]],","];
				AppendTo[assoc,"SourceIPAddress"->IPAddress[ip]];
				AppendTo[assoc,"SourcePort"->port];
			),
			(*ELSE*)
			(*failed, so return $Failed*)
			(
				Return[$Failed]
			)
		];
		
		(*now do same for the destination addresses - 1 is the destination code*)
		res = checkRes[iSocketGetAddress[socketUUID,1,intType],sym];
		If[!FailureQ[res],
			(*THEN*)
			(*got the result, parse it out and add it to the association*)
			(
				{ip,port} = StringSplit[FromCharacterCode[DeleteCases[res,0]],","];
				AppendTo[assoc,"DestinationIPAddress"->IPAddress[ip]];
				AppendTo[assoc,"DestinationPort"->port];
			),
			(*ELSE*)
			(*failed, so return $Failed*)
			(
				Return[$Failed]
			)
		];
		
		(*return association of socket address details*)
		assoc
	)
]


CreateZMQSocket[type_?StringQ]:=Block[
	{
		uuid=CreateUUID[],
		typeInt = socketTypes[wlSocketTypeToZMQSocketType[wlHeuristicSocketTypeNames[type]]],
		res
	},
	(
		(*make sure that the type specified exists*)
		If[!MissingQ[typeInt],
			(*THEN*)
			(*found the type specified*)
			(
				(*attempt to create the socket, then save it in $Sockets and return*)
				If[checkRes[iCreateZMQSocket[uuid,$DefaultContext,typeInt],System`SocketConnect]===$Failed,
					$Failed,
					(
						$Sockets[uuid] = 
							<|
								"Protocol"->"ZMQ",
								"Type"->type,
								"StreamState"-><|"InMessageBuffer"->{},"OutMessageBuffer"->Internal`Bag[]|>
							|>;
						System`SocketObject[uuid]
					)
				]
			),
			(*ELSE*)
			(*wrong type of socket specified*)
			(
				Message[SocketObject::type,type,Capitalize@*ToLowerCase@*StringDelete["ZMQ_"]/@Keys[socketTypes]];
				$Failed
			)
		]
	)
]

CreateConnectTCPSocket[host_,port_]:=Block[
	{
		uuid = CreateUUID[],
		socketAddrDetails,
		portStr = ToString[port]
	},
	(
		(*attempt to connect the socket*)		
		If[checkRes[iTCPSocketConnect[uuid,host,portStr],System`SocketConnect] === $Failed,
			(*THEN*)
			(*we failed to connect - note that messages will already have been issued by checkRes*)
			(
				$Failed
			),
			(
				socketAddrDetails = getSocketAddressDetails[uuid,"TCP",System`SocketConnect];
				If[FailureQ[socketAddrDetails],
					(*THEN*)
					(*failed to get information on the socket somehow, so return $Failed*)
					(
						$Failed
					),
					(*ELSE*)
					(*success - can join the socket address details with the other metadata*)
					(
						$Sockets[uuid] = 
							Join[
								<|
									"Protocol"->"TCP",
									"DirectionType"->"Client",
									"StreamState"-><||>,
									"DestinationHostname"->host,
									"Active"->True
								|>,
								socketAddrDetails
							];
						System`SocketObject[uuid]
					)
				]
			)
		]
	)
];


CreateBindTCPServerSocket[addressAssoc_]:=Block[
	{
		(*note that we replace the scheme with tcp to use tcp as the transpost layer*)
		(*and make the association into a string for usage with binding to the zmq socket*)
		addrAssocZMQ = Append[addressAssoc,"Scheme"->"tcp"],
		addr,
		zmqSock,
		tcpSock,
		uuid = CreateUUID[],
		socketAddrDetails,
		host
	},
	(
		(*make the association into a string for usage with binding to the zmq socket*)
		addr = addressAssocString[addrAssocZMQ];
		(*first we need to create a zmq stream socket on the address specified*)
		zmqSock = CreateZMQSocket["ZMQ_STREAM"];
		If[zmqSock =!= $Failed,
			(*THEN*)
			(*now we can bind the socket to the address*)
			(
				If[BindZMQSocket[zmqSock,addrAssocZMQ] =!= $Failed,
					(*THEN*)
					(*bound successfully - mark this socket as a special kind of socket*)
					(
						If[iMarkZMQSocketShadowTCPSocket[First[zmqSock]] =!= $Failed,
							(*THEN*)
							(*we've marked the socket, now we need to listen on it with a special handler*)
							(
								(*now try to get the socket information for this opened socket*)
								
								socketAddrDetails = getSocketAddressDetails[First[zmqSock],"ZMQ",System`SocketConnect];
								If[FailureQ[socketAddrDetails],
									(*THEN*)
									(*failed to get information on the socket somehow, so return $Failed*)
									(
										$Failed
									),
									(*ELSE*)
									(*success - can join the socket address details with the other metadata*)
									(
										(*attempt to lookup the domain's ip address for DestinationIPAddress*)
										host = Quiet[HostLookup[addrAssocZMQ["Domain"], All],HostLookup::nohost];
										
										(*the special handler will take care of appending to the list of connected clients*)
										(*any clients that make connections - this is because zmq stream sockets automatically "accept"*)
										(*all connections and put the client's request onto the socket stream for reading*)
										(*it does this by prepending all message data with an identifier part of the message*)
										(*SocketListen internally knows that when it is listening on these kind of special shadow tcp sockets*)
										(*not to raise an event until it has the complete message data and identifier so it can include*)
										(*the client identifier in the event's data*)
										$Sockets[uuid] = 
											Join[
												<|
													"Protocol"->"TCP",
													"ZMQSocket"->First[zmqSock],
													"DirectionType"->"Server",
													"StreamState"-><||>,
													"TCPServerSocketSetupQ"->False,
													(*if we failed to lookup the destination IP address, then we use the domain spec given to use by the user*)
													"DestinationIPAddress"-> If[MissingQ[host],
														addrAssocZMQ["Domain"],
														First[Join @@ Values@KeyDrop["FullName"]@host]
													],
													"DestinationHostname"->addrAssocZMQ["Domain"],
													(*to get the port, ask the socket, as the port could have been specified as "*", which is any open port*)
													"DestinationPort"->URLParse[getSockOpts[zmqSock,"LastEndpoint"],"Port"],
													"Active"->True
												|>,
												KeyDrop[{"DestinationIPAddress","DestinationPort"}]@socketAddrDetails
											];
										
										tcpSock = System`SocketObject[uuid];
										
										If[Check[System`SocketListen[tcpSock];True,False],
											(*THEN*)
											(*worked - create the tcp server socket as a dummy socket*)
											(
												(*mark the socket as initialized*)
												$Sockets[uuid,"TCPServerSocketSetupQ"] = True;
												tcpSock
											),
											(*ELSE*)
											(*failed somehow*)
											(
												$Failed
											)
										]
									)
								]
								
								
							),
							(*ELSE*)
							(*marking the socket failed for some unimaginable reason*)
							(
								(*cleanup*)
								$Failed
							)
						]
					),
					(*ELSE*)
					(*failed to bind the socket*)
					(
						(*cleanup*)
						$Failed
					)
				]
			),
			(*ELSE*)
			(*failed to bind to the specified socket*)
			(
				(*messages will already have been issued, so just destroy the socket and return $Failed*)
				(*TODO implement destroy method for zmq sockets*)
				$Failed
			)
		]
	)
];


BindZMQSocket[sock_?validSocketObjectQ,addrAssoc_?AssociationQ]:=Block[
	{
		uuid = First[sock],
		addr = addressAssocString[addrAssoc],
		addressInternalDetails,
		host
	},
	(
		(*before binding to the socket, check if the addres is an ipv6 address, in which case we need to turn on*)
		(*ipv6 handling on the socket*)
		If[Socket`IPv6AddressQ[addrAssoc["Domain"]],
			(*THEN*)
			(*enable ipv6 handling on the socket*)
			setSockOpt[sock,"ZMQ_IPV6"->1]
		];
		(*bind the socket*)
		If[FailureQ[checkRes[iBindZMQSocket[uuid,addr],System`SocketOpen]],
			(*THEN*)
			(*we failed to bind, so fail*)
			$Failed,
			(*THEN*)
			(*success - mark the socket as having been bound to and save the remote address*)
			(
				(*now add all the address details*)
				addressInternalDetails = getSocketAddressDetails[uuid,"ZMQ",System`SocketOpen];
				If[FailureQ[addressInternalDetails],
					(*THEN*)
					(*failed for some reason - return $Failed*)
					(
						$Failed
					),
					(*ELSE*)
					(*no errors we got the full details*)
					(
						(*it doesn't really make sense for server sockets to have a "destination IP address" - as we refer to the destination as the endpoint*)
						(*so just include the source ip address info from the address*)
						(*the destination ip address is just whatever we get - usually this is a meaningless ipv6 multicast address information*)
						KeyValueMap[
							($Sockets[uuid,#1]=#2)&,
							KeyDrop[{"DestinationIPAddress","DestinationPort"}]@addressInternalDetails
						];
						$Sockets[uuid,"DestinationPort"] = URLParse[getSockOpts[sock,"LastEndpoint"],"Port"];
						(*if we failed to lookup the destination IP address, then we use the domain spec given to use by the user - this case is for when we connect to an IP address*)
						(*without a domain name*)
						host = Quiet[HostLookup[addrAssoc["Domain"], All],HostLookup::nohost];
						$Sockets[uuid,"DestinationIPAddress"] = If[MissingQ[host],
							addrAssoc["Domain"],
							First[Join @@ Values @ KeyDrop["FullName"] @ host]
						];
						$Sockets[uuid,"DirectionType"] = "Server";
						$Sockets[uuid,"DestinationHostname"] = addrAssoc["Domain"];
						$Sockets[uuid,"Active"] = True;
					)
				]
			)
		]
	)
];

ConnectZMQSocket[sock_?validSocketObjectQ,addrAssoc_?AssociationQ]:=Block[
	{
		uuid = First[sock],
		addr = addressAssocString[addrAssoc],
		res,
		addressInternalDetails,
		host
	},
	(
		(*bind the socket*)
		res = checkRes[iConnectZMQSocket[uuid,addr],System`SocketConnect];
		If[res === $Failed,
			(*THEN*)
			(*failed to connect, return $Failed*)
			$Failed,
			(*ELSE*)
			(*good to go, save the connection details inside $Sockets*)
			(
				(*now add all the address details*)
				addressInternalDetails = getSocketAddressDetails[uuid,"ZMQ",System`SocketConnect];
				If[FailureQ[addressInternalDetails],
					(*THEN*)
					(*failed for some reason - return $Failed*)
					(
						$Failed
					),
					(*ELSE*)
					(*no errors we got the full details*)
					(
						$Sockets[uuid,"DirectionType"] = "Client";
						$Sockets[uuid,"RemoteHostAddress"] = addr;
						$Sockets[uuid,"DestinationHostname"] = addrAssoc["Domain"];
						$Sockets[uuid,"Active"] = True;
						KeyValueMap[
							($Sockets[uuid,#1]=#2)&,
							KeyDrop[{"DestinationIPAddress","DestinationPort"}]@addressInternalDetails
						];
						$Sockets[uuid,"DestinationPort"] = addrAssoc["Port"];
						(*if we failed to lookup the destination IP address, then we use the domain spec given to use by the user - this case is for when we connect to an IP address*)
						(*without a domain name*)
						host = Quiet[HostLookup[addrAssoc["Domain"], All],HostLookup::nohost];
						$Sockets[uuid,"DestinationIPAddress"] = If[MissingQ[host],
							addrAssoc["Domain"],
							First[Join @@ Values @ KeyDrop["FullName"] @ host]
						];
					)
				]
			)
		]
	)
];

System`SocketOpen[address_]:=System`SocketOpen[address,"TCP"];

System`SocketOpen[address_,protocol_?StringQ]:=Block[
	{
		type,
		addrAssoc,
		sock,
		proto = ToUpperCase[protocol]
	},
	(
		(*first attempt to parse the address specification*)
		
		Which[
			StringMatchQ[proto,"ZMQ"~~___],
			(*use the ZMQ method*)
			(
				(*get the socket type*)
				(*note we default to Pair if no type is specified*)
				If[ToUpperCase[proto] === "ZMQ",
					(*THEN*)
					(*default zmq type is pair*)
					type = "ZMQ_PAIR",
					(*the type is probably specified in the string specified, so let CreateZMQSocket figure it out*)
					(*note that if this is invalid, CreateZMQSocket will issue the appropriate message and fail for us*)
					type = proto
				];
				(*parse the address specification, using tcp as the default transport layer, which is specified as the scheme in the url*)
				addrAssoc = addressBuildAssoc[address,"tcp",System`SocketOpen,"zmq"];
				If[addrAssoc === $Failed,
					(*THEN*)
					(*something failed in parsing out the address string specification - note that addressBuild will issue the required messages for us*)
					$Failed,
					(*ELSE*)
					(*correct address parsing - continue with internal socket creation methods*)
					(
						sock = CreateZMQSocket[type];
						If[sock === $Failed || BindZMQSocket[sock,addrAssoc] === $Failed,
							(*THEN*)
							(*one of the steps failed, but messages were already issued*)
							$Failed,
							(*ELSE*)
							(*worked, binded to the address - return the socket object*)
							(
								sock
							)
						]
					)
				]
			),
			(*ELSE*)
			proto === "TCP",
			(*need to do special handling for the tcp version of the socket - note that *)
			(*internally this still uses zmq library, but we use special functions to handle the socket, as it doesn't work quite the same *)
			(*as a normal zmq socket*)
			(
				(*parse the address specification, using tcp as the default transport layer, which is specified as the scheme in the url*)
				addrAssoc = addressBuildAssoc[address,None,System`SocketOpen,"tcp"];
				If[addrAssoc === $Failed,
					(*THEN*)
					(*something failed in parsing out the address string specification - note that addressBuild will issue the required messages for us*)
					$Failed,
					(*ELSE*)
					(*parsed okay - continue with the internal socket connect method for tcp sockets*)
					(
						(*tcp sockets that are opened are actually zmq stream socket types*)
						(*that have special handling - we use a special function for opening such sockets*)
						(*which will configure the socket appropriately, as well as start the background *)
						(*task for listening on the socket to get the connected clients and such*)
						sock = CreateBindTCPServerSocket[addrAssoc]
					)
				]
			),
			True,
			(*unknown protocol*)
			(
				Message[System`SocketOpen::noproto,protocol];
				$Failed
			)
		]
	)
];

Unprotect[System`SocketConnect]

(*form of SocketConnect that will connect to a locally opened SocketObject from SocketOpen*)
System`SocketConnect[serverSocket_?validSocketObjectQ] := Block[
	{
		serverUUID = First[serverSocket]
	},
	(
		(*make sure that the socket has been opened, and it's direction type is server*)
		If[$Sockets[serverUUID,"DirectionType"] === "Server",
			(*THEN*)
			(*we can lookup the socket connection details*)
			(
				SocketConnect[{$Sockets[serverUUID,"DestinationIPAddress"],$Sockets[serverUUID,"DestinationPort"]},$Sockets[serverUUID,"Protocol"]]
			),
			(*ELSE*)
			(*not a server socket, can't connect to it*)
			(
				Message[System`SocketConnect::notServerSock,serverSocket];
				$Failed
			)
		]
	)
]

(*the default protocol is TCP*)
System`SocketConnect[address_]:=System`SocketConnect[address,"TCP"]

System`SocketConnect[address_,protocol_?StringQ,OptionsPattern[]]:=Block[
	{
		sock,
		type,
		addrAssoc,
		uuid,
		res,
		proto = ToUpperCase[protocol]
	},
	(
		Which[
			StringMatchQ[proto,"ZMQ"~~___],
			(*use the ZMQ method*)
			(
				(*get the socket type*)
				(*note we default to Pair if no type is specified*)
				If[ToUpperCase[proto] === "ZMQ",
					(*THEN*)
					(*default zmq type is pair*)
					type = "ZMQ_PAIR",
					(*the type is probably specified in the string specified, so let CreateZMQSocket figure it out*)
					(*note that if this is invalid, CreateZMQSocket will issue the appropriate message and fail for us*)
					type = proto
				];
				(*parse the address specification, using tcp as the default transport layer, which is specified as the scheme in the url*)
				addrAssoc = addressBuildAssoc[address,"tcp",System`SocketConnect,"zmq"];
				If[addrAssoc === $Failed,
					(*THEN*)
					(*something failed in parsing out the address string specification - note that addressBuild will issue the required messages for us*)
					$Failed,
					(*ELSE*)
					(*correct address parsing - continue with internal socket creation methods*)
					(
						sock = CreateZMQSocket[type];
						If[sock === $Failed || ConnectZMQSocket[sock,addrAssoc] === $Failed,
							(*THEN*)
							(*one of the steps failed, but messages were already issued*)
							$Failed,
							(*ELSE*)
							(*worked, was successfully connected to the address - return the socket object*)
							(
								sock
							)
						]
					)
				]
			),
			proto === "TCP",
			(*use tcp socket definition*)
			(
				(*build up the components of the address as an association*)
				addrAssoc = addressBuildAssoc[address,None,System`SocketConnect,"tcp"];
				If[addrAssoc === $Failed,
					(*THEN*)
					(*something failed in parsing out the address string specification - note that addressBuild will issue the required messages for us*)
					$Failed,
					(*ELSE*)
					(*correct address parsing - continue with internal socket creation methods*)
					(
						sock = CreateConnectTCPSocket[addrAssoc["Domain"],addrAssoc["Port"]];
						If[sock === $Failed,
							(*THEN*)
							(*one of the steps failed, but messages were already issued*)
							$Failed,
							(*ELSE*)
							(*worked, was successfully connected to the address - return the socket object*)
							(
								sock
							)
						]							
					)
				]
			),
			True,
			(*invalid or unsupport protocol*)
			(
				Message[System`SocketConnect::noproto,protocol];
				$Failed
			)
		]
	)
];

Protect[System`SocketConnect]

CloseSocket[sock_?validSocketObjectQ] := Block[
	{ 
		addr,
		sockUUID = First[sock],
		closeRes,
		zmqSock,
		host
	},
	(
		If[!TrueQ[$Sockets[sockUUID,"Active"]],
			(*THEN*)
			(*the socket is already closed, so issue a message*)
			(
				Message[Close::closedsocket,sock];
				Return[$Failed]
			)
		];
		
		(*first check if this socket is being listened on, if it is then stop listening on it*)
		If[KeyExistsQ["Listener"]@$Sockets[sockUUID],
			DeleteObject[$Sockets[sockUUID,"Listener"]];
			KeyDropFrom[$Sockets[sockUUID],"Listener"]
		];
		
		(*call the Close method on the internal streams if they exist*)
		If[MemberQ[Streams[],$Sockets[sockUUID,"StreamState","InputStream"]],
			(*THEN*)
			(*an input stream was opened that we need to close*)
			Close[$Sockets[sockUUID,"StreamState","InputStream"]]
		];
		
		If[MemberQ[Streams[],$Sockets[sockUUID,"StreamState","OutputStream"]],
			(*THEN*)
			(*an output stream was opened that we need to close*)
			Close[$Sockets[sockUUID,"StreamState","OutputStream"]]
		];
		
		closeRes = Which[
			shadowTCPClientSocketQ[sock],
			(*special handling for shadow tcp client sockets*)
			(
				(*this socket doesn't really exist, but we still can disconnect it*)
				(*by sending the identity as the first part of a multipart message*)
				(*then a empty message which tells zmq to disconnect that client*)
				(*first lookup the actual zmq socket*)
				zmqSock = $Sockets[sockUUID,"ZMQSocket"];
				
				(*now write the identity bytes for this client *)
				ZMQWriteInternal[System`SocketObject[zmqSock],$AsyncState["TCPClients",zmqSock,sockUUID],"Multipart"->True];
				(*now write an empty message*)
				ZMQWriteInternal[System`SocketObject[zmqSock],{},"Multipart"->False];
				
				(*now remove this client information from the TCPClients and other locations it might have state stored*)
				KeyDropFrom[$AsyncState["TCPClients"][zmqSock],sockUUID];
				KeyDropFrom[$AsyncState["TCPClientBuffers"],sockUUID];
			),
			shadowTCPServerSocketQ[sock],
			(*for shadow tcp server sockets, we have to stop the listening on the background thread for the zmq socket*)
			(*then close the zmq server socket*)
			(
				(*get the zmq socket and stop listening on it first*)
				zmqSock = $Sockets[sockUUID,"ZMQSocket"];
				removeListenerSocket[zmqSock];
				
				(*now actually close that socket*)
				closeRes = CloseSocket[System`SocketObject[zmqSock]];
				
				(*mark all the remote clients for this socket as inactive*)
				($Sockets[#,"Active"]=False)&/@Keys[$AsyncState["TCPClients",zmqSock]/.{_Missing:>{}}];
				
				(*empty out the various extra asynchronous state for this socket*)
				KeyDropFrom[$AsyncState["TCPClients"],zmqSock];
				KeyDropFrom[$AsyncState["TCPClientBuffers"],sockUUID];
				
				(*return value is the return value of the CloseSocket function*)
				closeRes
			),
			$Sockets[sockUUID,"Protocol"] === "ZMQ",
			(*call the zmq disconnect functons link normal*)
			(
				(*zmq socket's to disconnect need to know where to disconnect from, so we use the last endpoint*) 
				(*(which currently in the api is always the ONLY endpoint that it could have connected to) *)
				addr = Options[sock,"LastEndpoint"];
				If[addr =!= $Failed,
					(*THEN*)
					(*need to disconnect from the endpoint specified*)
					(
						(*can just call the method and return what the checker does, if it's successful it'll be Null, else it will return $Failed for us*)
						checkRes[iDisconnectZMQSocket[sockUUID,addr],Close];
						
						(*now delete the socket object*)
						checkRes[iDeleteZMQSocket[sockUUID],Close];
					),
					(*ELSE*)
					(*failed to get endpoint to disconnect from*)
					(
						Message[System`SocketObject::endpoint];
						$Failed
					)
				]
			),
			$Sockets[sockUUID,"Protocol"] === "TCP" && $Sockets[sockUUID,"DirectionType"] === "Client",
			(*the socket is a tcp socket client socket (on our side), and we need to close it with a different function*)
			(
				checkRes[iTCPSocketClose[sockUUID],Close]
			)
		];
		
		(*if this socket was ever listened on with AsyncState, then we need to clear out that info as well*)
		KeyDropFrom[$AsyncState["ZMQSocketBuffers"],sockUUID];
		KeyDropFrom[$AsyncState["Handlers"],sockUUID];
				
		If[closeRes === $Failed,
			(*THEN*)
			(*something weird happened, so just return $Failed*)
			$Failed,
			(*ELSE*)
			(*worked to close it, so just return the RemoteHostAddress key to match previous behavior*)
			(
				(*finally drop the entry from the $Sockets dataset after saving the RemoteHostAddress*)
				host = $Sockets[sockUUID,"DestinationHostname"]<>":"<>ToString[$Sockets[sockUUID,"DestinationPort"]];
				
				(*the socket is now inactive*)
				$Sockets[sockUUID,"Active"] = False;
				
				(*return the host to match existing design*)
				host
			)
		]
	)
];


(*system function for always reading exactly 1 message from a zmq socket*)
(*on a normal tcp socket it just reads whatever it can without blocking*)

(*by default we want to receive entire full messages*)
Options[System`SocketReadMessage]={"Multipart"->True,"Blocking"->True};
System`SocketReadMessage[sock_?validSocketObjectQ,OptionsPattern[]]:= Block[
	{
		uuid = First[sock],
		res,
		multipart = OptionValue["Multipart"],
		done = False,
		fullData,
		numBytes,
		restData
	},
	(
		(*check what protocol this socket uses*)
		If[$Sockets[uuid,"Protocol"] === "ZMQ",
			(*THEN*)
			(*use zmq options, etc.*)
			If[MatchQ[multipart,True|False],
				(*THEN*)
				(*correct value for Multipart option*)
				(
					(*get a message, and pass -1 as the flags to allow for aborting*)
					res = checkRes[iRecvSingleMultipartBinaryMessageSocket[uuid,-1],System`SocketReadMessage];
					If[res === $Failed,
						(*THEN*)
						(*failed - return $Failed*)
						$Failed,
						(*ELSE*)
						(*got data - check if there are more and whether the user requested a full message*)
						If[res[[2]] === 1 && multipart,
							(*THEN*)
							(*loop until we get a full multipart message*)
							(
								(* assuming the number of chunks is reasonnably small so that AppendTo does not 
									ruin performances. If that weren't true we can build this function around bag. *)
								fullData = {res[[3;;]]};
								While[!done,
									res = checkRes[iRecvSingleMultipartBinaryMessageSocket[uuid,-1],System`SocketReadMessage];
									If[res === $Failed,
										(*THEN*)
										(*failed to get this message - break out*)
										Return[$Failed],
										(*ELSE*)
										(*check if we have more or not*)
										(
											(*mark whether or not we are done as whether or not the first data*)
											(*is false*)
											done = res[[2]] === 0;
											AppendTo[fullData,res[[3;;]]];
										)
									]
								];
								Join@@fullData
							),
							(*ELSE*)
							(*either no more messages or user didn't request a full message*)
							(
								res[[3;;]]
							)
						]
					]
				),
				(*ELSE*)
				(*incorrect value for Multipart option*)
				(
					Message[System`SocketReadMessage::multipartOpt,multipart];
					$Failed
				)
			],
			(*ELSE*)
			(*use tcp functions - we get the size of the buffer then read that much from the socket*)
			(
				(*first see if this socket is a shadow tcp client socket*)
				Which[
					shadowTCPClientSocketQ[sock],
					(*we need to read from this client's buffer if there's data and we need to block until there is data otherwise*)
					(
						fullData = Join@@$AsyncState["TCPClientBuffers",uuid];
						If[fullData === {},
							(*ELSE*)
							(*we have to block until the asynchronous evaluator gets a chance to hop in and*)
							(*add some data in for us here*)
							While[fullData === {},
								(*check if we're inside the zmq async callback, if so then we need to set the $yieldingTask*)
								(*True so the callback that we yield to essentially does nothing except take data that was serviced and put it*)
								(*into the appropriate buffers and return*)
								(*we DON'T want it to evaluate any HandlerFunctions, because we're already*)
								(*inside one if $AsynchronousTask is set*)
								If[PreemptProtect[Head[$AsynchronousTask] === AsynchronousTaskObject],
									$yieldingTask = True;
									(*this function will check the async task queue for any tasks waiting to eval*)
									(*(which will happen if an async task is already evaluating)*)
									(*and evaluates exactly one of them if it can*)
									(*if it evaluated one, it returns True, otherwise it returns False*)
									(*so essentially control flow from here if a task is available will jump*)
									(*to asyncPollEventHandler*)
									Internal`YieldAsynchronousTask[];
									$yieldingTask = False;
								];

								(*get all the data we have for this socket from the buffer joining it together*)
								fullData = Join@@$AsyncState["TCPClientBuffers",uuid];
							];
						];
						(*if we get here it means that we got some data we can return*)
						(*drop the amount of data we have here from the buffer and assign that back to the buffer*)
						PreemptProtect[
							(*note that the pre-emptive kernel could have fired between the previous expr and this block, so*)
							(*we need to be careful in how we remove the data from the buffer, mainly in that we take out much we had above*)
							(*so that if more was added, it stays in the buffer for next time*)
							$AsyncState["TCPClientBuffers",uuid] = {Drop[Join@@$AsyncState["TCPClientBuffers",uuid],Length[fullData]]};
						];
						ByteArray[fullData]
					),
					shadowTCPServerSocketQ[sock],
					(*then it's a server socket and we don't support reading from it because you can't *)
					(*read/write on this individual server socket, that can only happen on the connected client sockets*)
					(
						Message[System`SocketReadMessage::tcpServerNoAccess,sock];
						$Failed
					),
					True,
					(*it's not a shadow socket, so handle it normally*)
					(
						(*first see if the inputstream has been opened yet, opening it if it hasn't been opened yet*)
						If[FailureQ[checkRes[iTCPSocketInStreamCreate[uuid],System`SocketReadMessage]],
							Return[$Failed]
						];
						
						(*now we query to see how many bytes there are available to read without blocking*)
						numBytes = checkRes[iTCPSocketInStreamBufferSize[uuid],System`SocketReadMessage];
						(*check to make sure the request was successful*)
						If[FailureQ[numBytes],
							Return[$Failed],
							numBytes = First[numBytes]
						];
						(*if it returns 0, then we first check to see if the socket has been closed, if so then we issue a message and return an empty bytearray*)
						(*then we check the blocking option, and if it's true, then we block using select, which is abortable using the wait for input*)
						(*function, then after that returns we get the number of bytes readable again then read that many*)
						If[numBytes === 0,
							(*THEN*)
							(*we need to see if the socket is closed from the other side*)
							(
								res = checkRes[iTCPSocketClosedQ[uuid],System`SocketReadMessage];
								If[FailureQ[res],
									(*THEN*)
									(*some catastrophic failure*)
									Return[$Failed],
									(*ELSE*)
									(*valid, check if the socket is closed*)
									(
										If[res === {1},
											(*THEN*)
											(*socket is closed - return nothing*)
											(
												{}
											)
											(*ELSE*)
											(*not closed, continue on*)
										]
									)
								];
								
								(*if we get here we know that the socket wasn't closed, so now it's time to check if we have to block or not*)
								If[TrueQ[OptionValue["Blocking"]],
									(*THEN*)
									(*we are expected to block, so use the wait for input function*)
									(
										If[FailureQ[checkRes[iTCPSocketInStreamWaitForInput[uuid],System`SocketReadMessage]],
											Return[$Failed]
										];
										(*now that we have blocked for input, go ask for how much data to use*)
										numBytes = checkRes[iTCPSocketInStreamBufferSize[uuid],System`SocketReadMessage];
										(*check to make sure the request was successful - note that wait for input isn't allowed to return until there is at least 1 byte*)
										(*or if we got aborted, so if it returned 0, then that's an error*)
										If[FailureQ[numBytes] || numBytes === {0},
											Return[$Failed],
											numBytes = First[numBytes]
										];
									),
									(*ELSE*)
									(*not supposed to block but we don't have any data so just return an empty ByteArray*)
									(
										Return[ByteArray[{}]]
									)
								]
							)
						];
						
						(*if we get here it means that we have more than 0 bytes to read, so read that many and return it as a bytearray*)
						res = checkRes[iTCPSocketInStreamRead[uuid,numBytes],Read]/.{Null->{}};
						If[FailureQ[res],
							(*THEN*)
							(*failed*)
							$Failed,
							(*ELSE*)
							(*got something, return it as a ByteArray*)
							ByteArray[res]
						]
					)
				]
			)
		]
	)
];



normalSocketsPoll[sockPollAssoc_,timeout_,symbol_]:=Block[
	{
		polls = Flatten[{#}]&/@Values[sockPollAssoc],
		pollItems,
		socketUUIDs = First/@Keys[sockPollAssoc],
		uuidLens,
		uuids,
		pollRes,
		res
	},
	(
		(*now handle the poll items, making sure that the specified items are all correct*)
		If[AllTrue[Complement[#,{"PollInput","PollOutput","PollError"}]==={}&]@polls,
			(*THEN*)
			(*all of the poll types are valid, turn them into the integer bitmask representation*)
			(
				pollItems = Join@@(pollStateToIntegers/@polls)
			),
			(*ELSE*)
			(*invalid poll items*)
			(
				With[{s=symbol},Message[MessageName[s, "timeout"],timeout]];
				Return[$Failed]
			)
		];
		
		(*join together all of the uuids*)
		uuidLens = StringLength/@socketUUIDs;
		uuids = ToCharacterCode[StringJoin[socketUUIDs],"ASCII"];
		
		(*call the poll function*)
		pollRes = checkRes[iPollSockets[uuidLens,uuids,pollItems,timeout],symbol];
		If[pollRes === $Failed,
			(*THEN*)
			(*failed for some reason, message would already have been issued*)
			$Failed,
			(*ELSE*)
			(*didn't fail, so recreate the poll results structure*)
			(
				If[First[pollRes] === 0,
					(*THEN*)
					(*no events - can do a shortcut to determine the return association*)
					(
						AssociationThread[Keys[sockPollAssoc],Table[{},{Length[sockPollAssoc]}]]
					),
					(*ELSE*)
					(*got some events - need to build up the entire list of events*)
					(
						(*check if we have any tcp client sockets here - if we do then we have to check on it being closed*)
						(*if it's closed then it will have a "PollInput" event on it, meaning it might be "ready", but in reality*)
						(*its closed so it's not "ready"*)
						<|
							KeyValueMap[
								Function[{sock,pollItemResult},
									Which[
										$Sockets[First[sock],"Protocol"] === "TCP" && $Sockets[First[sock],"DirectionType"] === "Client",
										(*socket is a tcp client and we need to check if it's closed*)
										(
											res = checkRes[iTCPSocketClosedQ[First[sock]],symbol];
											If[FailureQ[res],
												(*THEN*)
												(*not sure what happened, but still return the result we already got*)
												sock->pollItemResult,
												(*ELSE*)
												(*worked, see if it is closed, if so no results, else return what the result was*)
												sock->If[res === {1}, {}, pollItemResult]
											]
										),
										True,
										(*normal socket - just return what we got*)
										sock->pollItemResult
									]
								],
								AssociationThread[Keys[sockPollAssoc],pollResultsReinterpret[Rest@pollRes]]
							]
						|>
					)
				]
			)
		]
	)
]

remoteTCPClientPollStatus[sockUUIDList_] := AssociationMap[(Flatten[$AsyncState["TCPClientBuffers",First[#]]] =!= {})/.{True->{"PollInput"},False->{}}&,sockUUIDList]


Options[internalPoll] = {TimeConstraint->Default}
(*symbol arg lets us issue specific messages for different System` functions*)
internalPoll[sockPollAssoc_?AssociationQ,symbol_,OptionsPattern[]]/;AllTrue[validSocketObjectQ]@Keys[sockPollAssoc]:=Block[
	{
		socks = Keys[sockPollAssoc],
		timeout = OptionValue[TimeConstraint],
		startTime,
		normalSockets,
		sockTypes,
		remoteClients,
		socketUUIDs,
		res
	},
	(
		socketUUIDs = First/@socks;
		(*first check if any of the sockets are tcp server sockets, which we can't poll*)
		If[AnyTrue[shadowTCPServerSocketQ]@socks,
			(*THEN*)
			(*we have to fail, as we can't poll on this socket*)
			(
				(*issue a message for every tcp server socket we find*)
				With[{s=symbol},Message[MessageName[s, "tcpServerSockPoll"],#]]& /@ (Keys@KeySelect[shadowTCPServerSocketQ]@sockPollAssoc);
				Return[$Failed];
			)
			(*ELSE*)
			(*don't need to do anything, can continue normally*)
		];
		
		(*now confirm the timeout value*)
		Which[
			MemberQ[{Infinity,Default,Automatic},timeout],(*no timeout*)
			timeout = -1,
			(IntegerQ[timeout]||MachineNumberQ[timeout])&&timeout>=0,
			timeout = Round@QuantityMagnitude@UnitConvert[Quantity[timeout,"Seconds"],"Milliseconds"],
			QuantityQ[timeout]&&CompatibleUnitQ[timeout,"Milliseconds"]&&timeout>=Quantity[0,"Seconds"],(*valid time spec for timeout*)
			timeout = Round@QuantityMagnitude@UnitConvert[timeout,"Milliseconds"],
			True,
			(
				With[{s=symbol},Message[MessageName[s, "timeout"],timeout]];
				Return[$Failed]
			)
		];
		
		(*now check if there's any remote tcp clients that are being requested to poll on - we have to handle those specially*)
		Which[
			NoneTrue[shadowTCPClientSocketQ]@socks,
			(*no shadow tcp sockets, so continue on as normal*)
			(
				(*call the normal poll sockets for the normal sockets*)
				normalSocketsPoll[sockPollAssoc,timeout,symbol]
			),
			AllTrue[shadowTCPClientSocketQ]@socks,
			(*all of the requested sockets are shadow tcp sockets, so we don't have to call the internal poll method*)
			(
				(*basically we sit waiting for the timeout to expire, checking if we got pre-empted and there is suddently data in the client buffers for *)
				(*any of the clients - note that we can only ever poll for input socket events*)
				startTime = AbsoluteTime[];
				While[( timeout === -1 || AbsoluteTime[] - startTime < 1000timeout ) && AnyTrue[Flatten[$AsyncState["TCPClientBuffers",#]] =!= {}&]@socketUUIDs,
					(*just pause in the while loop so we don't eat up all the cpu*)
					Pause[0.01];
				];
				
				(*if we got here, it means something happened, so assign "PollInput" states to clients that have data on them*)
				remoteTCPClientPollStatus[socks]
			),
			AnyTrue[shadowTCPClientSocketQ]@socks,
			(*we have mixed case of some normal clients and some shadow tcp clients - due to how this works the *)
			(
				(*basically what we do here is the same as the case for the normal non remote sockets, except at the end we also join together the states for the remote tcp sockets*)
				sockTypes = GroupBy[shadowTCPClientSocketQ]@socks;
				normalSockets = KeyTake[sockTypes[False]]@sockPollAssoc;
				remoteClients = KeyTake[sockTypes[True]]@sockPollAssoc;
				
				(*call the normal poll sockets for the normal sockets*)
				res = normalSocketsPoll[normalSockets,timeout,symbol];
				If[FailureQ[res],
					(*THEN*)
					(*failed, just return it - messages will already have been handled for us*)
					res,
					(*ELSE*)
					(*got the result, combine it with *)
					(
						Join[
							remoteTCPClientPollStatus[remoteClients],
							res
						]
					)
				]
			)
		]
	)
];

(*socket readyq takes a single socket and tests immediately if it has data*)
System`SocketReadyQ[sockObj_?validSocketObjectQ,opts:OptionsPattern[]]:=System`SocketReadyQ[sockObj,0,"PollInput",opts]
(*also support a timespec as second arg*)
System`SocketReadyQ[sockObj_?validSocketObjectQ,t_,opts:OptionsPattern[]]:=System`SocketReadyQ[sockObj,t,"PollInput",opts]
System`SocketReadyQ[sockObj_?validSocketObjectQ,t_,eventType_?StringQ,opts:OptionsPattern[]]:=Block[
	{
		res = internalPoll[
			<|sockObj->eventType|>,
			System`SocketReadyQ,
			TimeConstraint->t
		]
	},
	If[res===$Failed,
		(*THEN*)
		(*didn't work*)
		$Failed,
		(*ELSE*)
		(*worked, check if the result is the event we're looking for*)
		MemberQ[res[sockObj], eventType]
	]
]

(*wait all blocks for every socket to get events*)
(*by default when no events are specified, only poll for reading from the sockets*)
System`SocketWaitAll[sockObjs:{_?validSocketObjectQ..},opts:OptionsPattern[]] := System`SocketWaitAll[sockObjs,"PollInput",opts]
(*for wait all we just scan internal poll over all of the sockets*)
(*because we need data on all of them, it doesn't matter what order we poll the sockets on*)
System`SocketWaitAll[sockObjs:{_?validSocketObjectQ..},eventType_,opts:OptionsPattern[]]:=(
	If[
		Scan[
			(*if the return value of internalPoll is $Failed, then break out of the Scan*)
			If[internalPoll[
					<|#->eventType|>,
					System`SocketWaitAll,
					opts
				] === $Failed,
				Return[$Failed]
			]&,
			sockObjs
		]===$Failed,
		$Failed
	];
	(*return the list of sockets*)
	sockObjs
)

(*wait next is basically poll - it takes a list of sockets and returns as soon as an event happens on any one of them*)
(*by default when no events are specified, only poll for reading from the sockets*)
System`SocketWaitNext[sockObjs:{_?validSocketObjectQ..},opts:OptionsPattern[]]:=System`SocketWaitNext[sockObjs,"PollInput",opts];
System`SocketWaitNext[sockObjs:{_?validSocketObjectQ..},eventType_,opts:OptionsPattern[]]:=Block[
	{
		(*internal poll returns an Association of socks->events for each socket, so pick the ones that have events on them*)
		res = internalPoll[
			AssociationThread[sockObjs,Table[eventType,{Length[sockObjs]}]],
			System`SocketWaitNext,
			opts
		]
	},
	If[!FailureQ[res],
		(*THEN*)
		(*select the sockets that had events*)
		Keys@Select[#=!={}&]@res,
		(*ELSE*)
		(*failed, return it*)
		res
	]
]

System`SocketReadyQ[any___]:=(Message[System`SocketReadyQ::sock]; $Failed)
System`SocketWaitNext[any___]:= (Message[System`SocketWaitNext::sock]; $Failed)
System`SocketWaitAll[any___]:= (Message[System`SocketWaitAll::sock]; $Failed)

SocketListen[addrSpec_?(!validSocketObjectQ[#]&),opts:OptionsPattern[]]:= With[
	(*attempt to open up a socket with the address spec*)
	{sock = SocketOpen[addrSpec,"TCP"]},
	(
		If[!FailureQ[sock],
			(*THEN*)
			(*we were able to open the socket up*)
			SocketListen[sock,opts],
			(*ELSE*)
			(*we weren't able to open the socket, so fail*)
			$Failed
		]
	)
]

System`SocketListen[addrSpec_?(!validSocketObjectQ[#]&),func_?(!MatchQ[#,OptionsPattern[]]&),opts:OptionsPattern[]]:=With[
	(*attempt to open up a socket with the address spec*)
	{sock = SocketOpen[addrSpec,"TCP"]},
	(
		If[!FailureQ[sock],
			(*THEN*)
			(*we were able to open the socket up*)
			SocketListen[sock,func,opts],
			(*ELSE*)
			(*we weren't able to open the socket, so fail*)
			$Failed
		]
	)
]


(*form for specifying callback function as second arg*)
System`SocketListen[sock_?validSocketObjectQ,func_?(!MatchQ[#,OptionsPattern[]]&),opts:OptionsPattern[]]:=
	System`SocketListen[
		sock,
		(*fix the options - we remove any other HandlerFunctions option and replace it with the one specified by the second argument*)
		Sequence@@Normal[
			Append[
				KeyDrop[Association[opts], HandlerFunctions],
				HandlerFunctions-><|"DataReceived":>func|>
			]
		]
	]

(*this is the asynchronous version of SocketPoll, which allows for events to be handled asynchronously*)
Options[System`SocketListen] = {HandlerFunctions-><||>,HandlerFunctionsKeys->Default,CharacterEncoding:>$CharacterEncoding,RecordSeparators->None};
System`SocketListen[sock_?validSocketObjectQ,opts:OptionsPattern[]]:=Block[
	{
		(*the keys to apply to the handler*)
		handlerKeys = OptionValue[HandlerFunctionsKeys],
		(*the poll items are specified as the Keys in the HandlerFunctions options, i.e. HandlerFunctions-><|"PollInput":>Print,...|>*)
		polls = Keys[OptionValue[HandlerFunctions]],
		pollItems,
		sockUUID = First[sock],
		fixedHandlerKeys
	},
	(
		(*next make sure the poll states requested are valid*)
		Which[
			polls === {},
			(*no handler functions specified, so just have it generate PollInput events, which triggers async evaluation, but nothing is done with the data in WL side*)
			(
				pollItems = pollStateToIntegers[{"PollInput"}]
			),
			Complement[polls,{"DataReceived"}]==={},
			(*all of the poll types are valid, turn them into the integer bitmask representation*)
			(
				(*the "DataReceived" event is the same as a PollInput event*)
				pollItems = pollStateToIntegers[polls/.{"DataReceived"->"PollInput"}];
			),
			True,
			(*wrong format for HandlerFunctions*)
			(
				Message[System`SocketListen::pollError,Complement[polls,{"DataReceived"}]];
				Return[$Failed]
			)
		];
		
		(*confirm the handler keys are valid*)
		fixedHandlerKeys = confirmHandlerKeys[handlerKeys];
		If[FailureQ[fixedHandlerKeys],
			(*THEN*)
			(*there's something wrong with the handlerKeys, issue a message dependent on what the failure object returned*)
			(
				With[{msg = Last[fixedHandlerKeys]["msg"], args = Last[fixedHandlerKeys]["args"]}, Message[MessageName[SocketListen, msg],Sequence@@args]];
				Return[$Failed];
			)
		];
		
		(*next check on the status of the async thread*)
		If[Not[$AsyncState["Status"] == "Running"],
			(*THEN*)
			(*we need to start the thread first*)
			Switch[$AsyncState["Status"],
				Missing[___], (*some kind of error, where the assoc was modified incorrectly*)
				(
					Return[$Failed]
				),
				"Error", (*was running but encountered some kind of error*)
				(
					Return[$Failed];
				),
				"Not running", (*need to start it up*)
				(
					(*first make two sockets - one to push updates to and one to pull updates*)
					(*we don't specify the port here, letting the OS find an open one for us to bind to*)
					$AsyncState["Incoming"] = System`SocketOpen["tcp://127.0.0.1:*","ZMQ"];
					(*figure out which port we got*)
					$AsyncState["CommPort"] = URLParse[getSockOpts[$AsyncState["Incoming"],"LastEndpoint"],"Port"];
					(*now connect a push socket to that port for sending updates*)
					$AsyncState["Outgoing"] = System`SocketConnect["tcp://127.0.0.1:"<>ToString[$AsyncState["CommPort"]],"ZMQ"];
					$AsyncState["Task"] = Internal`CreateAsynchronousTask[
						iStartAsyncPollingThread,
						{First@$AsyncState["Incoming"],First@$AsyncState["Outgoing"]},
						asyncPollEventHandler
					];
					If[Head[$AsyncState["Task"]]=!=AsynchronousTaskObject,
						(*THEN*)
						(*invalid task, raise message*)
						(
							Message[System`SocketListen::error];
							Return[$Failed];
						)
					];
				)
			]
		];
		
		(*now to actaully start the polling process, we can just send on the push socket the sockets and the poll items to add*)
		(*it expects json as the string so need to export it as such*)
		(*note that the socket uuid we send for server tcp sockets is different and there are two cases here*)
		(*one is the case where we are initializing the socket and so we do send the underlying zmq socket*)
		(*and two is the case where we are done initializing and the user themselves are calling SocketListen on the socket*)
		Which[
			shadowTCPServerSocketQ[sockUUID] && !TrueQ[$Sockets[sockUUID,"TCPServerSocketSetupQ"]],
			(*this is a server tcp socket, and so when we add it, we need to intialize the listener thread by sending the zmq socket id*)
			(*SocketListen, or if it's from the initial SocketOpen call*)
			(
				WriteString[
					$AsyncState["Outgoing"],
					(*for shadow tcp server sockets, we poll for input and error events -> BitOr[1,4]=5*)
					ExportString[<|"add"-><|$Sockets[sockUUID,"ZMQSocket"]->5|>|>,"JSON","Compact"->True]
				];
			),
			shadowTCPClientSocketQ[sockUUID] || shadowTCPServerSocketQ[sockUUID],
			(*shadow tcp client sockets don't really exist to listen on, so we don't add a socket for the thread to listen on*)
			(*additionally, shadow tcp server sockets that are setup will already have been setup on the thread*)
			(*so there's nothing to do here except update the handlers to call*)
			(
				Null
			),
			True,
			(*normal case for the socket*)
			(
				WriteString[
					$AsyncState["Outgoing"],
					ExportString[<|"add"->AssociationThread[{sockUUID},pollItems]|>,"JSON","Compact"->True]
				];
			)
		];
		
		
		
		(*finally save the handler information for this socket into the dataset*)
		
		(*we also have to again check on whether or not the socket is a tcp server and the user is specified this instance, as if that's the case*)
		(*we need to cache the settings*)
		If[shadowTCPServerSocketQ[sockUUID] && TrueQ[$Sockets[sockUUID,"TCPServerSocketSetupQ"]],
			(*THEN*)
			(*this is a server tcp socket, and so when we add it, we need to intialize the listener thread by sending the zmq socket id*)
			(*SocketListen, or if it's from the initial SocketOpen call*)
			(
				(*we save it under the zmq socket handler functions so that the callback doesn't have to do a lookup for the tcp*) 
				(*socket and can simply use the one provided by the callback*)
				(*now store the handler functions*)
				$AsyncState["Handlers",$Sockets[sockUUID,"ZMQSocket"]] = <|
					HandlerFunctions:>OptionValue[HandlerFunctions],
					HandlerFunctionsKeys->fixedHandlerKeys,
					CharacterEncoding:>OptionValue[CharacterEncoding],
					RecordSeparators:>OptionValue[RecordSeparators]
				|>;
			),
			(*ELSE*)
			(*normal socket - store the sockUUID rather than the underlying zmq sock UUID*)
			(
				$AsyncState["Handlers",sockUUID] = <|
					HandlerFunctions:>OptionValue[HandlerFunctions],
					HandlerFunctionsKeys->fixedHandlerKeys,
					CharacterEncoding:>OptionValue[CharacterEncoding],
					RecordSeparators:>OptionValue[RecordSeparators]
				|>;
			)
		];
		
		(*return a SocketListener object with the uuid as the underlying socket's uuid*)
		If[shadowTCPServerSocketQ[sockUUID] && !TrueQ[$Sockets[sockUUID,"TCPServerSocketSetupQ"]],
			(*THEN*)
			(*don't save it cause this is the setup call*)
			System`SocketListener[sockUUID],
			(*ELSE*)
			(*save it so that this can be closed when Close is called on the socket*)
			$Sockets[sockUUID,"Listener"] = System`SocketListener[sockUUID]
		]
	)
];

(*error handler*)
asyncPollEventHandler[taskObject_,"POLL_ERROR",data_]:=Block[{},
	(
		Message[
			System`SocketListen::zmqexception,
			Lookup[data,"ExceptionType"]/.{1->"ZeroMQ",2->"ZeroMQLink"},
			Lookup[data,"ExceptionCode"],
			FromCharacterCode@Normal@Lookup[data,"Exception"]
		]
	)
]

asyncPollEventHandler[taskObject_,eventType_,data_]:=Block[
	{
		(*this allows there to be multiple "Data" keys in the data, and we simply merge all of them together with this*)
		dataAssoc = Merge[Association/@data,If[AllTrue[#,ListQ],Join@@#,First[#]]&],
		sockID,
		keys,
		tcpServerSockUUID,
		dataRes
	},
	(
		(*add the Task type to the dataAssoc*)
		dataAssoc["EventType"] = eventType/.{"POLL_IN"->"DataReceived","POLL_OUT"->"PollOutput"};
		
		(*first decode the socket uuid*)
		dataAssoc["SocketUUID"] = FromCharacterCode[Normal[dataAssoc["SocketUUID"]],"UTF-8"];
		sockID = dataAssoc["SocketUUID"];
		
		(*this socket ID will always be a zmq socket id for tcp server sockets, but if this is the case*)
		(*the information we want for this handler is with the actual tcp socket uuid, but we have the zmq one specified here*)
		(*so if that's the case try to find the overlying tcp socket*)
		tcpServerSockUUID = SelectFirstKey[#["ZMQSocket"]===sockID&]@$Sockets;
		If[!MissingQ[tcpServerSockUUID],
			(*THEN*)
			(*did find a tcp socket that is the overlying socket for this socket*)
			(
				(*now we have to assign the appropriate variables*)
				(*extract the uuid from the Key[...] wrapper*)
				sockID = First[tcpServerSockUUID];
			)
			(*ELSE*)
			(*didn't find it, so just continue normally using sockID*)
		];

		(*now handle if this is a shadow tcp client event to raise on or not*)
		If[KeyExistsQ[sockID]@$AsyncState["Handlers"],
			(*THEN*)
			(*check if we are on a tcp server socket that has been setup, in which case we apply a wrapper function to the user function applied*)
			If[$Sockets[sockID,"Protocol"]==="TCP" && $Sockets[sockID,"DirectionType"]==="Server" && TrueQ[$Sockets[sockID,"TCPServerSocketSetupQ"]],
				(*THEN*)
				(*execute the shadowTCPSocketHandler with the user's function - this takes care of the special sauce that *)
				(*is required by the shadow tcp sockets*)
				shadowTCPSocketHandler[dataAssoc],
				(*ELSE*)
				(*normal socket - check on any cached data that may not have been executed last time*)
				(
					(*initialize the cached data from the buffer if it doesn't exist*)
					If[MissingQ[$AsyncState["ZMQSocketBuffers",sockID]],
						$AsyncState["ZMQSocketBuffers",sockID] = {};
					];
					
					(*add this event's data to the buffer before calling into the actual handling loop*)
					$AsyncState["ZMQSocketBuffers",sockID] = {Join@@Append[$AsyncState["ZMQSocketBuffers",sockID],Normal[dataAssoc["Data"]]]};
					
					(*get the required keys*)
					keys = KeyTake[{HandlerFunctionsKeys,CharacterEncoding,RecordSeparators}]@$AsyncState["Handlers",sockID];
					
					(*now execute the function*)
					loopDataRecieved[
						dataAssoc,
						keys,
						$AsyncState["Handlers",sockID,HandlerFunctions][dataAssoc["EventType"]]
					];
				)
			]
		];
	)
];



(*the shadowTCPSocketHandler will take in events generated on a ZMQ_STREAM socket*)
shadowTCPSocketHandler[rawAssoc_]:=Block[
	{
		(*get the identity of this particular client as identified by zmq - note it's a RawArray so we have to Normal it*)
		assoc = rawAssoc,
		clientIdentity = Normal[rawAssoc["Identity"]],
		serverSockUUID = rawAssoc["SocketUUID"],
		eventMessageData = Normal[rawAssoc["Data"]],
		clientSockUUID,
		userKeys,
		recSep,
		encoding,
		rest,
		res,
		dataStream,
		userFuncAssoc,
		addressInternalDetails
	},
	(
		(*first we need to take the data passed in and store it for this client - noting if the identity key exists*)
		If[!KeyExistsQ[serverSockUUID]@$AsyncState["TCPClients"],
			(*THEN*)
			(*this server socket has never received events on it before - initialize this server socket with $AsyncState first*)
			(
				$AsyncState["TCPClients",serverSockUUID] = <||>;
			)
			(*ELSE*)
			(*already initialized - nothing to do*)
		];
		
		(*try to find the client socket id for this event, matching the identities*)
		clientSockUUID = SelectFirstKey[#===clientIdentity&]@$AsyncState["TCPClients",serverSockUUID];
		
		(*check to see if the client was found or not to put the data into the buffer*)
		If[MissingQ[clientSockUUID],
			(*THEN*)
			(*this socket has not previously recieved events before from this client*)
			(
				(*save the identity of the client as a new UUID*)
				clientSockUUID = CreateUUID[];
				$AsyncState["TCPClients",serverSockUUID,clientSockUUID] = clientIdentity;
				addressInternalDetails = getShadowTCPClientSocketAddressDetails[serverSockUUID,clientIdentity,System`SocketListen];
				If[clientIdentity === {} || FailureQ[addressInternalDetails],
					(*THEN*)
					(*failed for some reason - return $Failed*)
					(
						Message[SocketListen::error,"clientIdentity",clientIdentity];
						Return[$Failed]
					),
					(*ELSE*)
					(*no errors we got the full details*)
					(
						(*also register it as a shadow TCP socket - looking up the connection details*)
						$Sockets[clientSockUUID] = Join[
							<|
								"Protocol"->"TCP",
								"DirectionType"->"Client",
								"StreamState"-><|"OutMessageBuffer"->Internal`Bag[],"InMessageBuffer"->{}|>,
								(*the zmq socket here points directly to the underlying zmq_stream socket that the connected tcp server*)
								(*socket is using internally, but note that we don't have to lookup that socket from the tcp server socket*)
								(*because this callback was triggered by the zmq socket, so we already know what zmq socket to associate*)
								(*this shadow tcp client socket to*)
								"DestinationHostname" -> First[addressInternalDetails["DestinationIPAddress"]],
								"ZMQSocket"->serverSockUUID,
								"Active"->True
							|>,
							addressInternalDetails
						];
					)
				];
				
				(*finally initialize the data buffer for this client using the data we got in this event*)
				$AsyncState["TCPClientBuffers",clientSockUUID] = {eventMessageData};
				
				(*also save the source socket here for later*)
				assoc["SourceSocket"] = clientSockUUID;
				
				(*note that we don't check if there's a handler on this client, as in this branch we know the socket uuid doesn't exist*)
				(*and so wouldn't have been put into the "ConnectedClients" property for the user to then SocketListen on*)

				(*TODO : add a ClientConnected event to be raised here for the TCP socket that may be listened on*)
			),
			(*ELSE*)
			(*this socket HAS previously recieved events from this client*)
			(
				(*make the clientSockUUID a string spec, not the Key spec*)
				clientSockUUID = First[clientSockUUID];
				
				(*because this socket has previously recieved events, then if the data parameter is an empty list, then we know that this event is*)
				(*a disconnect event*)
				If[eventMessageData === {},
					(*THEN*)
					(*disconnect event, mark the client as inactive*)
					(
						(*TODO : add a ClientDisconnected event to be raised here*)
						$Sockets[clientSockUUID,"Active"] = False;
					),
					(*ELSE*)
					(*not disconnect event, continue normally*)
					(
						(*check if there's a listener on this socket, as if there is, that listener takes priority over the main listener*)
						(*this is the case where the following events happen :*) 
						(*
							s=SocketOpen[...,"TCP"];
							SocketListen[s,func1];
							(*client connects*)
							c=First[s["ConnectedClients"]]
							SocketListen[c,func2];
						*)
						(*here we want func2 to run on the data, not func1*)
						
						(*also save the source socket here for later in the callback functions*)
						assoc["SourceSocket"] = clientSockUUID;
						
						(*first evaluate on all the previous messages*)
						If[!MissingQ[$AsyncState["Handlers",clientSockUUID,HandlerFunctions]],
							(*THEN*)
							(*need to call that function the user specified for the second SocketListen call*)
							(
								$AsyncState["TCPClientBuffers",clientSockUUID] = Join[eventMessageData,$AsyncState["TCPClientBuffers",clientSockUUID]];

								loopDataRecieved[
									assoc,
									KeyTake[{HandlerFunctionsKeys,CharacterEncoding,RecordSeparators}]@$AsyncState["Handlers",clientSockUUID],
									$AsyncState["Handlers",clientSockUUID,HandlerFunctions][assoc["EventType"]]
								];
							),
							(*ELSE*)
							(*just append the data to the buffer - no handler function to run*)
							AppendTo[$AsyncState["TCPClientBuffers",clientSockUUID],eventMessageData]
						];
					)
				]
			)
		];
		
		(*when SocketListen was initially called for the socket by the user (not by the internal BindTCPServerSocket function)*)
		(*for simplicity's sake the corresponding ZMQSocket's uuid was used to save the handler functions and keys, so*)
		(*we just use that here cause we get the uuid of the zmq stream socket living underneath*)
		userFuncAssoc = $AsyncState["Handlers",serverSockUUID,HandlerFunctions];
		
		If[!MissingQ[userFuncAssoc],
			(*THEN*)
			(*the user func was specified for listening, so now build up that association to run*)
			(
				(*first get the keys the user specified*)
				userKeys = KeyTake[{HandlerFunctionsKeys,CharacterEncoding,RecordSeparators}]@$AsyncState["Handlers",serverSockUUID];
				
				(*handle the record separators and raise events if necessary by*)
				(*join all the individual buffer elements together*)
				loopDataRecieved[assoc,userKeys,userFuncAssoc[assoc["EventType"]]];
			)
			(*ELSE*)
			(*don't need to call anything*)
		]
	)
];


loopDataRecieved[eventAssoc_,userKeys_,userFunc_]:=
	(*if there's no more data left, or there's no difference between data and prev data we're done*)
	(
		If[!$yieldingTask,
			While[TrueQ[handleDataReceived[eventAssoc,userKeys,userFunc]]]
		]
	)

assignLeftoversToBuffer[eventAssoc_,data_]:= If[KeyExistsQ["SourceSocket"]@eventAssoc,
		(*THEN*)
		(*this socket is a tcp client socket and so we assign the rest of the data to TCPClientBuffers*)
		$AsyncState["TCPClientBuffers",eventAssoc["SourceSocket"]] = {data},
		(*ELSE*)
		(*normal zmq socket and we should put the data into the ZMQSocketBuffers*)
		$AsyncState["ZMQSocketBuffers",eventAssoc["SocketUUID"]] = {data}
	]


(*handleDataReceived will take as input the details of an event as well as a function to run if the event is to be raised*)
(*it will check on the value of RecordSeparators to determine if the event is ready to be raised, raising it with the appropriate data*)
(*if ready*)
(*it also will run the userFunc at most once, so this function should be run in a loop to take care of the fact that there could be multiple*)
(*events ready to be serviced by a user level DataRecevied HandlerFunction*)
(*this function will dynamically get all the data it needs by joining together all the individual elements of the buffer for this socket*)
(*so before calling this function the buffer for this socket needs to have been populated with all the data we have for an event*)
(*additionally, this function will update the state of the buffers and return True if it serviced an event and False if it didn't*)
handleDataReceived[eventAssoc_,userKeys_,userFunc_]:=Block[
	{
		(*populate the data for this event from either TCPClientBuffers if we're dealing with a TCP client socket*)
		(*or ZMQ socket buffers if we're dealing with a normal zeromq socket*)
		data = If[KeyExistsQ["SourceSocket"]@eventAssoc,
			$AsyncState["TCPClientBuffers",eventAssoc["SourceSocket"]],
			$AsyncState["ZMQSocketBuffers",eventAssoc["SocketUUID"]]
		],
		dataChunk,
		rest,
		dataStream,
		readRes,
		recSep = userKeys[RecordSeparators],
		encoding = userKeys[CharacterEncoding]
	},
	(
		(*check if the data is missing - this is some sort of catastrophic event, but we need to be careful not to return True here*)
		(*because that will lead to an infinite loop*)
		If[MissingQ[data],
			Return[False]
		];
		(*join the data together before continuing so it's a 1d byte array essentially*)
		data = Flatten[{data}];
		Which[
			data === {},
			(*no data available for an event - so stop servicing*)
			False,
			(IntegerQ[recSep] && recSep > 0),
			(*this is a temporary hack to allow for only issuing events every so many bytes on the stream*)
			(
				If[Length[data]>=recSep,
					(*THEN*)
					(*we're good to raise the event*)
					(
						(*only take as much data as was requested by the user to raise*)
						{dataChunk,rest} = TakeDrop[data,recSep];

						(*we need to set the state of the buffer before calling the user's func*)
						(*this is because during the user's func, they could try to read the rest of the data from it*)
						assignLeftoversToBuffer[eventAssoc,rest];
						
						(*create the data assoc, except with the new data key we just got*)
						(*now call the user's function*)
						userFunc[buildUserEventAssocWithKeys[Append[eventAssoc,"Data"->dataChunk],userKeys]];

						(*serviced an event*)
						True
					),
					(*ELSE*)
					(*didn't do anything, we don't have enough data - also don't have to modify the buffer at all*)
					(
						False
					)
				]
			),
			recSep === None || recSep === {},
			(
				(*no record separators so we write back an empty list to the buffer for this socket*)
				assignLeftoversToBuffer[eventAssoc,{}];

				(*create the data assoc, except with the new data key we just got*)
				(*now call the user's function*)
				userFunc[buildUserEventAssocWithKeys[Append[eventAssoc,"Data"->data],userKeys]];

				True
			),
			True,
			(*this just checks to see if the record separators option is valid or not for strings / streams or not*)
			If[validRecSepQ[recSep],
				(*TRUE*)
				(*normal record separator spec, so just pass this into StringToStream functions*)
				(
					(*make a stream of the string to look for the recSep option*)
					dataStream = OpenRead[CreateUUID[],Method->{"SocketByteList","Bytes"->data},BinaryFormat->True];
					
					(*now look for the RecordSeparators option, as that specifies when we raise the event*)
					(*this is a bit of a hack, but we're looking for a full record, but the stream functions*)
					(*like Read, etc. will return a record even if we don't hit a record separator before EndOfFile*)
					(*so for example Read[StringToStream["hello world"],RecordSeparators->"x"] will return "hello world", but it's not a full record yet*)
					(*so what we do is we use TokenWords, which will return the limiter if it found it*)
					(*if we get back EndOfFile, then we didn't hit the record, otherwise, if we get anything else back *)
					(*then we successfully found the record and can raise the event*)
					readRes = Read[dataStream,Record,NullRecords->True,RecordSeparators->recSep];
					wordDelimiter = Read[dataStream, Word, TokenWords -> recSep, RecordSeparators -> {}, WordSeparators -> {}];
					If[wordDelimiter =!= EndOfFile,
						(*THEN*)
						(*we did find a full valid record and can raise an event*)
						(
							(*return the rest of the stream as bytes*)
							rest = BinaryReadList[dataStream];
							
							(*close the stream we opened*)
							Close[dataStream];
							
							(*update the client buffers here before calling the user event*)
							assignLeftoversToBuffer[eventAssoc,rest];

							(*we build up the association for each chunk of data except the last one, the last one we can't determine if it*)
							(*includes the record separator or not, and so we have to wait for more data to be able to tell if it's valid or not*)
							(*note that buildUserEventAssocWithKeys excepts a list or RawArray of data, so we re-encode it to binary from the string*)
							(*form returned by ReadList*)
							userFunc[buildUserEventAssocWithKeys[Append[eventAssoc,"Data"->ToCharacterCode[readRes,encoding]],userKeys]];

							True
						),
						(*ELSE*)
						(*didn't find enough events so don't raise any events with the user's function*)
						(
							(*close the stream we opened*)
							Close[dataStream];
						
							(*we didn't modify the buffers at all, so we don't have to assign anything back*)
							False
						)
					]
				),
				(*ELSE*)
				(*invalid RecordSeparators specification - so just pretend like there wasn't enough data*)
				(
					False
				)
			]
		]
	)
];



(*this function will build up the association to return to the user's callback function*)
buildUserEventAssocWithKeys[rawDataAssoc_,keys_]:=Block[
	{
		finalAssoc=<||>,
		dataNormaled = Normal[rawDataAssoc["Data"]]
	},
	(
		(*first check the versions of data we have to deal with, there are 3 possibles,*)
		(*"Data" -> as a string encoded using the option CharacterEncoding*)
		(*"DataBytes" -> list of data integers*)
		(*"DataByteArray" -> data integers as a ByteArray*)
		
		If[MemberQ["DataBytes"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the DataBytes*)
			finalAssoc["DataBytes"] = dataNormaled
		];
		
		If[MemberQ["DataByteArray"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the Data byte array*)
			finalAssoc["DataByteArray"] = ByteArray[dataNormaled]
		];
		
		If[MemberQ["Data"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the Data byte array*)
			finalAssoc["Data"] = FromCharacterCode[dataNormaled,keys[CharacterEncoding]]
		];
		
		(*now handle the "Multipart" key, returning that as a True/False flag*)
		If[MemberQ["MultipartComplete"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the multipart flag*)
			(*the logic is flipped, as MultipartComplete means no more messages, but the flag we get is whether there are more messages or not*)
			finalAssoc["MultipartComplete"] = !rawDataAssoc["MoreMessagesQ"]
		];
		
		If[MemberQ["Timestamp"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the Data byte array*)
			finalAssoc["Timestamp"] = timeInterpret[rawDataAssoc["TimestampLow"],rawDataAssoc["TimestampHigh"]]
		];
		
		
		(*now handle if this is a shadow tcp client event to raise on or not*)
		If[KeyExistsQ["Identity"]@rawDataAssoc,
			(*THEN*)
			(*we only add the SourceSocket version if it shows up in the original rawData assoc, then we know that a shadow*)
			(
				(*note that the SourceSocket key is added from the shadowTCPSocketHandler function, where we already have to do*)
				(*the inverse lookup of the client socket id from the identity*)
				(*but that function doesn't delete the identity key, so that's how we can tell that this function is being called*)
				(*on a shadow tcp socket*)
				finalAssoc["SourceSocket"] = System`SocketObject[rawDataAssoc["SourceSocket"]];
				
				(*we also use a different Socket key here, as we want to specify the tcp server socket*)
				finalAssoc["Socket"] = System`SocketObject[First[SelectFirstKey[KeyExistsQ[rawDataAssoc["SourceSocket"]]]@$AsyncState["TCPClients"]]];
			),
			(*ELSE*)
			(*the key doesn't exist, so the source socket must be the same as the server socket*)
			(
				(*add the socket normally*)
				finalAssoc["Socket"] = System`SocketObject[rawDataAssoc["SocketUUID"]];
				finalAssoc["SourceSocket"] = System`SocketObject[rawDataAssoc["SocketUUID"]]
			)
		];
		
		(*now check if we have to drop the socket or source socket keys - this is simpler than having multiple checks for the "Identity" key*)
		If[!MemberQ["Socket"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the Data byte array*)
			KeyDropFrom[finalAssoc,"Socket"]
		];
		
		If[!MemberQ["SourceSocket"]@keys[HandlerFunctionsKeys],
			(*THEN*)
			(*need to add the Data byte array*)
			KeyDropFrom[finalAssoc,"SourceSocket"]
		];
		
		(*finally return the entire association we just built up*)
		finalAssoc
	)
]

(*closeSocketListener ensures that the socket is no longer listened on by the background thread and *) 
(*removes it's information from $AsyncState*)
closeSocketListener[sockListener_?validSocketListenerQ]:=Block[
	{
		sockUUID = First[sockListener]
	},
	(
		(*make sure that the socket has been listened on first*)
		If[KeyExistsQ[sockUUID]@$AsyncState["Handlers"],
			(*THEN*)
			(*we can stop listening as the socket is currently being listened on*)
			(
				(*only need to actually remove the socket uuid if the socket is not a shadow tcp socket type*)
				If[Nor@@Through[{shadowTCPServerSocketQ,shadowTCPClientSocketQ}[System`SocketObject[sockUUID]]],
					(*THEN*)
					(*the socket needs to be removed from the background thread*)
					(
						removeListenerSocket[sockUUID]
					)
				];
				(*remove this socket from the list of asynchronous sockets and delete the socket listener from the socket*)
				KeyDropFrom[$Sockets[sockUUID],"Listener"];
				KeyDropFrom[$AsyncState["Handlers"],sockUUID];
			),
			(*ELSE*)
			(*hasn't been submitted yet*)
			(
				Message[System`SocketListener::nosock,System`SocketObject[sockUUID]];
				$Failed
			)
		]
	)
]

removeListenerSocket[sockUUID_?StringQ]:=Block[{response},
	(
		PreemptProtect[
			(*send a message to remove this socket from the list of sockets to poll*)
			WriteString[$AsyncState["Outgoing"],ExportString[<|"remove"->{sockUUID}|>,"JSON","Compact"->True]];
			(*wait for a confirmation message from the incoming socket that it was removed successfully - this prevents us from getting into*)
			(*race conditions where the socket is used in another function while being removed from the background thread*)
			response = Check[Quiet[ImportString[ByteArrayToString[SocketReadMessage[$AsyncState["Outgoing"]]],"JSON"]],$Failed];
			If[FailureQ[response],
				(*THEN*)
				(*something went wrong in the background thread, return $Failed*)
				(
					Message[DeleteObject::listenerError,sockListener];
					Return[$Failed]
				)
			]
		]
	)
]



(*a valid socket object is one that has as it's first and only argument a uuid that is currently present inside $Sockets*)
validSocketObjectQ[System`SocketObject[sock_?StringQ]]:=validSocketObjectQ[sock];
validSocketObjectQ[sock_?StringQ]:=MemberQ[Keys[$Sockets],sock];
validSocketObjectQ[___]:=False

validSocketListenerQ[System`SocketListener[sock_]]:=validSocketListenerQ[sock]
validSocketListenerQ[sock_?StringQ]:=MemberQ[Keys[$Sockets],sock];
validSocketListenerQ[___]:=False

shadowTCPClientSocketQ[System`SocketObject[uuid_]]:=shadowTCPClientSocketQ[uuid]
shadowTCPClientSocketQ[uuid_?StringQ]:=KeyExistsQ[uuid]@(Join@@Values[$AsyncState["TCPClients"]])
shadowTCPClientSocketQ[___]:=False

shadowTCPServerSocketQ[System`SocketObject[uuid_]]:=shadowTCPServerSocketQ[uuid]
shadowTCPServerSocketQ[uuid_?StringQ]:= ( $Sockets[uuid,"Protocol"] === "TCP" && $Sockets[uuid,"DirectionType"] === "Server" ) 
shadowTCPServerSocketQ[___]:=False

(*replace the string specs like PollInput, etc. with their numeric counterparts and BitOr them together*)
pollStateToIntegers[pollItems_List]:=
	BitOr @@ AssociationThread[{"PollInput", "PollOutput", "PollError"}, {1, 2, 4}]@#& /@ pollItems;

(*with no polls, we need to give 0*)
pollStateToIntegers[{}]:=0

(*inverse of pollStateToIntegers basically, takes integers and replaces with strings of states*)
pollResultsReinterpret[results_List]:=
	ReplaceAll[
		DeleteCases[
			(*this gives us a list of the values from each bit, i.e. a poll result of 7 turns into the list {1,2,4}*)
			MapIndexed[BitShiftLeft[#1, First[#2] - 1] &, #] & /@ (Reverse[IntegerDigits[#, 2, 3]] & /@ results),
			(*delete 0 as it doesn't correspond to anything*)
			0,
			Infinity
		],
		AssociationThread[{1, 2, 4}, {"PollInput", "PollOutput", "PollError"}]
	];


(*time interpret takes the time integers from the built-in OS functions and converts them to valid DateObjects*)
timeInterpret[low_?IntegerQ,high_?IntegerQ] := Module[{},
	(
		(*this takes two integers and turns them into a proper time object with correct TimeZone and such*)
		Switch[$OperatingSystem,
			"MacOSX"|"Unix",
			(
				(*on unix systems, the two integers are the number of seconds since the unix epoch and the fractional number of microseconds*)
				TimeZoneConvert[
					FromUnixTime[0,TimeZone->"UTC"]+UnitConvert[Quantity[low,"Microseconds"]+Quantity[high,"Seconds"],"Seconds"],
					$TimeZone
				]
			),
			"Windows",
			(
				(*on windows, just take bitshift the high int up 32 and OR it with the low to get the number of 100 nanosecond intervals since Jan 1 1601 00:00 UTC*)
				TimeZoneConvert[
					Quantity[BitOr[BitShiftLeft[high,32],low]*100,"Nanoseconds"]+DateObject[{1601,1,1,0,0},TimeZone->"UTC"],
					$TimeZone
				]
			),
			_,(*if for some reason $OperatingSystem isn't one of the above...*)
			(
				$Failed
			)
		]
	)
];

	
(*helper function for selecting a key out of an association where they key's value matches a predicate*)
SelectFirstKey[assoc_,pred_,default_:Missing["NotFound"]]:=Block[
	{pos = LengthWhile[assoc,Not@*pred]},
	If[pos === Length[assoc],
		(*THEN*)
		(*we didn't find it, so return the default*)
		(default),
		(*ELSE*)
		(*we did find it, so return that key, incrementing the position, because LengthWhile counts the number of things before this element that are false for the predicate*)
		(Key[Keys[assoc][[pos+1]]])
	]
]

(*operator form*)
SelectFirstKey[pred_][assoc_]:=SelectFirstKey[assoc,pred]

(*socket icon used from : https://thenounproject.com/term/lan-cable/49978/*)
(*the base64 encoded svg was imported using Mathematica*)
$SocketIcon = Graphics[
 Scale[
  {{{FilledCurve[{{Line[{{36.558, 8.569}, {40.947, 8.569}, {40.947, 43.684000000000005}, 
          {36.558, 43.684000000000005}, {36.558, 8.569}}]}}]}, 
    {FilledCurve[{{Line[{{59.053, 8.569}, {63.443, 8.569}, {63.443, 43.684000000000005}, 
          {59.053, 43.684000000000005}, {59.053, 8.569}}]}}]}, 
    {{FilledCurve[{{Line[{{55.487, 8.569}, {56.95, 8.569}, {56.95, 21.188000000000002}, 
           {55.487, 21.188000000000002}, {55.487, 8.569}}]}}]}, 
     {FilledCurve[{{Line[{{52.562, 8.569}, {54.025, 8.569}, 
           {54.025, 21.188000000000002}, {52.562, 21.188000000000002}, 
           {52.562, 8.569}}]}}]}, 
     {FilledCurve[{{Line[{{49.636, 8.569}, {51.099000000000004, 8.569}, 
           {51.099000000000004, 21.188000000000002}, {49.636, 21.188000000000002}, 
           {49.636, 8.569}}]}}]}, 
     {FilledCurve[{{Line[{{46.709, 8.569}, {48.172000000000004, 8.569}, 
           {48.172000000000004, 21.188000000000002}, {46.709, 21.188000000000002}, 
           {46.709, 8.569}}]}}]}, 
     {FilledCurve[{{Line[{{43.783, 8.569}, {45.246, 8.569}, 
           {45.246, 21.188000000000002}, {43.783, 21.188000000000002}, 
           {43.783, 8.569}}]}}]}}, 
    {FilledCurve[{{Line[{{40.947, 4.911}, {59.787000000000006, 4.911}, 
          {59.787000000000006, 6.922}, {40.947, 6.922}, {40.947, 4.911}}]}}]}, 
    {FilledCurve[{{Line[{{44.057, 31.675}, {56.678000000000004, 31.675}, 
          {56.678000000000004, 39.051}, {44.057, 39.051}, {44.057, 31.675}}]}}]}, 
    {FilledCurve[{{Line[{{44.057, 43.685}, {56.678000000000004, 43.685}, 
    	(*originally, this was going down to around 95.089 - this was changed to 65.089*)
          {56.678000000000004, 65.089}, {44.057, 65.089}, {44.057, 43.685}}]}}]}}}, 
          (*originally the PlotRange was {{0,100},{0,100}}*)
  {1, -1}], PlotRange -> {{20, 80}, {0, 70}},
  BaseStyle -> {CacheGraphics -> False},
  ImageSize->30
  ];

(*reprotect all the System` symbols again*)
(
	SetAttributes[#,{ReadProtected}];
	Protect[#]
)&/@{
	"System`SocketConnect",
	"System`SocketObject",
	"System`SocketReadMessage",
	"System`SocketOpen",
	"System`SocketListen",
	"System`SocketListener",
	"System`SocketWaitAll",
	"System`SocketWaitNext",
	"System`SocketReadyQ",
	"System`Sockets"
}


End[]

EndPackage[]
