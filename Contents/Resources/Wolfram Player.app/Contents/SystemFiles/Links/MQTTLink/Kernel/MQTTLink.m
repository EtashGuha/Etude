(* ::Package:: *)

(* Wolfram Language Package *)

BeginPackage["MQTTLink`"]
(* Exported symbols added here with SymbolName::usage *)  

MQTTClient

CreateClient::usage="CreateClient creates a new MQTTClient";
CreateClient::exists="The client with ID `1` already exists";
CreateClient::cleanSession="The value of \"CleanSession\", `1`, is not True or False";
CreateClient::invalidHost="The host `1` is invalid";
CreateClient::clientIDInvalid="The ClientID `1` is invalid; the ClientID must be a Basic Multilingual Plane UTF-8 string between 1 and 65535 characters long (inclusive), with no \\0 characters.";
CreateClient::noUsername="The Username option must be specified as a String if the Password option is also specified";
CreateClient::clientPassword="The Password `1` is invalid, it must be specified as a string";
CreateClient::clientUsername="The Client username `1` is invalid, it must be specified as a string";

ConnectClient::usage="ConnectClient connects the given MQTTClient to a broker";
ConnectClient::alreadyConnected="The client `1` is already connected";
ConnectClient::nonexist="The client `1` doesn't exist";
ConnectClient::noPort="No port for broker specified, using 1883";
ConnectClient::noPortFound="No port for broker the broker was identified in `1`, ports should be specified as \"host:port\", with \"port\" being a valid integer port ";
ConnectClient::noIP="No IP address for broker specified, using localhost";
ConnectClient::noIPFound="No IP address was specified, specify one with ConnectClient[client, addr], where addr is a URL or IPAddress object, or a string specifying the host and port as \"host:port\", with \"port\" being a valid integer port"
ConnectClient::blocking="The value of \"Blocking\", `1`, is not True or False";
ConnectClient::timeout="The value of \"Timeout\", `1`, should be an Integer";
ConnectClient::threadFail="An unknown error occured connecting the client, try again later";
ConnectClient::connFail="`1`";
ConnectClient::unexpectDisconn="The client `1` was unexpectedly disconnected with code `2`";
ConnectClient::errno="A system error was encountered trying to connect to `1`; the connection failed";
ConnectClient::noBroker="Could not connect to the broker at  `1`";
ConnectClient::invalidHost=CreateClient::invalidHost;

DisconnectClient::unknownError="An unknown error occured while disconnecting";
DisconnectClient::nonexist=ConnectClient::nonexist;

TopicPublish::usage="TopicPublish publishs a message on a topic using the specified client";
TopicPublish::messageutf="The message `1` should be a list of integers 0-255 or a Basic Multilingual Plane UTF-8 string";
TopicPublish::nonexist=ConnectClient::nonexist;
TopicPublish::notConnected="The client `1` isn't connected";
TopicPublish::noConnection="The client `1` failed to connect to broker";
TopicPublish::protocolError="The system failed to send the message due to a protocol error";
TopicPublish::qos="The QualityOfService level `1` is invalid, valid levels are 0, 1, and 2";
TopicPublish::blocking=ConnectClient::blocking;
TopicPublish::unknownError="An unknown error occured while publishing";
TopicPublish::retain="The \"Retain\" option must be either True or False";
TopicPublish::tooLarge="The message `1` is too large, messages must be smaller than 256 MB";
TopicPublish::topicNameInvalid="The topic name `1` is invalid; topic names must be Basic Multilingual Plane UTF-8 strings with topic levels seperated by \"/\"\
 with no wildcard characters, \"+\" and \"#\", present at any level";

TopicSubscribe::usage="TopicSubscribe subscribes to the given topic";
TopicSubscribe::blocking=ConnectClient::blocking;
TopicSubscribe::qos=TopicPublish::qos;
TopicSubscribe::noConnection=TopicPublish::noConnection;
TopicSubscribe::alreadySubscribed="The client `1` is already subscribed to the topic pattern `2`";
TopicSubscribe::diffQos="The topic `1` was requested with a Quality of Service level `2`, but the broker granted level `3`";
TopicSubscribe::unknownMessageCallbackError="An unknown error occured when a message was received (`1`)";
TopicSubscribe::unknownError="An unknown error occured while subscribing (`1`)";
TopicSubscribe::subFailed="Subscribing to the topic `1` with a Quality of Service level `2` failed with error code `3`"
TopicSubscribe::invalidTopicFilter="The topic filter `1` is invalid; topic filters must be Basic Multilingual Plane UTF-8 strings with topic levels seperated by \"/\",\
 with single level wildcards specified as a \"+\", and multilevel wildcards specified as \"#\". Multilevel wildcards are only allowed as the last level, \
 while single level wildcards may be placed at any level";

TopicUnsubscribe::usage="TopicUnsubscribe unsubscribes from the specified topic";
TopicUnsubscribe::noSubscribe="The client `1` isn't subscribed to the topic `2`";
TopicUnsubscribe::blocking=ConnectClient::blocking;
TopicUnsubscribe::noConnection=TopicPublish::noConnection;
TopicUnsubscribe::unknownError="An unknown error occured while unsubscribing (`1`)";
TopicUnsubscribe::nonexist=ConnectClient::nonexist

MQTTClientSetOptions::nokey="The option `1` doesn't exist";
MQTTClientSetOptions::nonexist = ConnectClient::nonexist;
MQTTClientSetOptions::protectedKey="The option `1` is not user directly configurable with MQTTClientSetOptions";

StartBroker::usage="Starts a local MQTT broker on the specified port";
StartBroker::running="There is already a broker running on port `1`";
StartBroker::startError="The broker failed to run exiting with : `1` (exit code `2`)";
StartBroker::invalidConfigOptions="The configuration options, `1`, are invalid. Valid configuration options for the broker are Mosquitto conf options that are then written out to a temporary file"

MQTTBrokers::usage="Returns an Association of running Brokers"

Begin["`Private`"] (* Begin Private Context *) 

(*paclet manager is for accessing paclet resources*)
Needs["PacletManager`"];


(*for tracking the various brokers*)
$Brokers = <||>;


(*------RETURN CODE CONSTANTS------*)
(*MQTTLink specific return values*)
$ClientNotFound = -42;
$ClientExists = -44;
$TopicNotFound = -45;
$ThreadError = -46;
(*mosquitto return values*)
$MosquittoConnectionPending = -1;
$MosquittoSuccess = 0;
$MosquittoOutOfMemory = 1;
$MosquittoProtocolError = 2;
$MosquittoInvalidFunctionArguments = 3;
$MosquittoNoConnection = 4;
$MosquittoConnectionRefused = 5;
$MosquittoErrorNotFound = 6;
$MosquittoConnectionLost = 7;
$MosquittoTLSError = 8;
$MosquittoPayloadTooLarge = 9;
$MosquittoNotSupported = 10;
$MosquittoAuthenticationError = 11;
$MosquittoACLDenied= 12;
$MosquittoUnknownError = 13;
$MosquittoErrnoError = 14;
$MosquittoLookupError = 15;
$MosquittoProxyError = 16;
$MosquittoConnectionRefusedProtoVersion = 1;
$MosquittoConnectionRefusedIdentifierRejected = 2;
$MosquittoBrokerUnavailable = 3;
$MosquittoUsernamePasswordMalformed = 4;
$MosquittoNotAuthorized = 5;

(*lock and protect all the constants*)
SetAttributes[$ClientNotFound,{Locked,Protected}];
SetAttributes[$ClientExists,{Locked,Protected}];
SetAttributes[$TopicNotFound,{Locked,Protected}];
SetAttributes[$ThreadError,{Locked,Protected}];
SetAttributes[$MosquittoConnectionPending,{Locked,Protected}];
SetAttributes[$MosquittoSuccess,{Locked,Protected}];
SetAttributes[$MosquittoOutOfMemory,{Locked,Protected}];
SetAttributes[$MosquittoProtocolError,{Locked,Protected}];
SetAttributes[$MosquittoInvalidFunctionArguments,{Locked,Protected}];
SetAttributes[$MosquittoNoConnection,{Locked,Protected}];
SetAttributes[$MosquittoConnectionRefused,{Locked,Protected}];
SetAttributes[$MosquittoErrorNotFound,{Locked,Protected}];
SetAttributes[$MosquittoConnectionLost,{Locked,Protected}];
SetAttributes[$MosquittoTLSError,{Locked,Protected}];
SetAttributes[$MosquittoPayloadTooLarge,{Locked,Protected}];
SetAttributes[$MosquittoNotSupported,{Locked,Protected}];
SetAttributes[$MosquittoAuthenticationError,{Locked,Protected}];
SetAttributes[$MosquittoACLDenied,{Locked,Protected}];
SetAttributes[$MosquittoUnknownError,{Locked,Protected}];
SetAttributes[$MosquittoErrnoError,{Locked,Protected}];
SetAttributes[$MosquittoLookupError,{Locked,Protected}];
SetAttributes[$MosquittoProxyError,{Locked,Protected}];
SetAttributes[$MosquittoConnectionRefusedProtoVersion,{Locked,Protected}];
SetAttributes[$MosquittoConnectionRefusedIdentifierRejected,{Locked,Protected}];
SetAttributes[$MosquittoBrokerUnavailable,{Locked,Protected}];
SetAttributes[$MosquittoUsernamePasswordMalformed,{Locked,Protected}];
SetAttributes[$MosquittoNotAuthorized,{Locked,Protected}];


(*these are properties we don't want the user setting themselves using MQTTClientSetOptions, as they aren't really options*)
$ProtectedMQTTClientProperties = {"Connected","ClientID","BrokerIP","BrokerPort","SubscribedTopics"};
SetAttributes[$ProtectedMQTTClientProperties,{Locked,Protected}];


(*loads the mosquitto library and all the relevant functions from the linked library*)
loadLibrary[]:=With[
	{
		mqttlinkLib=FindLibrary["MQTTLink"]
	},
	(
		(*if we're on windows, we need to load the ssl libraries too*)
		If[$OperatingSystem === "Windows",
			(*THEN*)
			(*load the libraries, they are always deterministically in SystemFiles/Libraries/$SystemID*)
			(
				(*on windows, need to load the ssl libraries before mqttlink*)
				LibraryLoad[FileNameJoin[{$InstallationDirectory, "SystemFiles", "Libraries", $SystemID, "ssleay32.dll"}]];
				LibraryLoad[FileNameJoin[{$InstallationDirectory, "SystemFiles", "Libraries", $SystemID, "libeay32.dll"}]];
			)
		];
		
		(*now load the mqttlink dynamic library*)
		LibraryLoad[mqttlinkLib];
		(*then load all of the LibraryLink functions*)
		(*first arg is id, second is whether to make this client use a clean session or not*)
		iCreateClient = LibraryFunctionLoad[mqttlinkLib, "CreateClient", {String, "Boolean"}, Integer];
		(*first arg is id, second is host ip, third is port, fourth is timeout to use*)
		iConnectClient = LibraryFunctionLoad[mqttlinkLib, "ClientConnect", {String, String, Integer, Integer}, {Integer, 1}];
		(*only argument is the id to disconnect*)
		iDisconnectClient = LibraryFunctionLoad[mqttlinkLib,"ClientDisconnect",{String},Integer];
		(*first arg is id, second is topic pattern, third is message data, fourth is quality of service, fifth is whether this is to be a retained message*)
		iTopicPublish = LibraryFunctionLoad[mqttlinkLib, "TopicPublish", {String, String, {Integer,1}, Integer, "Boolean"}, {Integer, 1}];
		(*first arg is id, second is topic pattern to subscribe to, third is quality of service*)
		iTopicSubscribe = LibraryFunctionLoad[mqttlinkLib, "SubscribeTopic", {String, String, Integer}, {Integer, 1}];
		(*first arg is id, second is topic pattern to unsubscribe from*)
		iTopicUnsubscribe = LibraryFunctionLoad[mqttlinkLib, "UnsubscribeTopic", {String, String}, {Integer, 1}];
		(*first arg is id, second is topic pattern to unsubscribe from*)
		iCreateTopicSubscriptionTask = LibraryFunctionLoad[mqttlinkLib,"CreateSubscriptionTask",{String,String},{Integer,1}];
		(*first arg is id, second is whether there's a password or not, third is the username, and fourth is the password*)
		iSetPassword = LibraryFunctionLoad[mqttlinkLib,"SetClientUsernamePassword",{String, "Boolean", String, String},Integer];
	)
];

(*make sure to load the library whenever this package is loaded*)
loadLibrary[];


(*-----MQTTClient content box / summary box code------*)

(*set upvalues for Normal*)
MQTTClient/:Normal[client_MQTTClient]:=First[client];

(*set UpValues for MakeBoxes*)
(*StandardForm MakeBoxes*)
MQTTClient/:MakeBoxes[MQTTClient[data_Association],StandardForm]:=BoxForm`ArrangeSummaryBox[
		(*first argument is the head to use*)
		MQTTClient,
		(*second argument is the expression*)
		MQTTClient[data],
		(*third argument is the icon to use*)
		clientImage,
		(*the next argument is the always visisble properties*)
		BoxForm`SummaryItem[{ToString[#1]<>": ",#2}]&@@@Normal[KeyTake[data,{"Connected","ClientID"}]],
		(*the next argument is the optional items that come down when the plus button is pressed*)
		BoxForm`SummaryItem[{ToString[#1]<>": ",Normal@#2}]&@@@Normal[KeyDrop[data,{"Connected","ClientID"}]],
		(*lastly,the display form we want to display this with*)
		StandardForm,
		(*making it interpretable allows users to take the displayed object and query things about it directly without having to save it in a variable*)
		"Interpretable"->True
];
(*FullForm MakeBoxes*)
MQTTClient/:MakeBoxes[MQTTClient[data_Association],FullForm]:=MQTTClient[Append[data,"Connected"->$Clients[data["ClientID"],"Connected"]]];

(*when properties from the object are queried, just used the internal association*)
MQTTClient[data_Association][key_] := $Clients[data["ClientID"],key];

MQTTClient[data_Association]["Properties"]:=Keys[$Clients[data["ClientID"]]];

MQTTClient /: Set[MQTTClient[data_Association][key_], val_] := ($Clients[data["ClientID"],key] = val)	

(*can't seem to get this form to work yet*)
(*MQTTClient /: SetDelayed[MQTTClient[data_Association][key_], val_] := ($Clients[data["ClientID"],key] := val)*)

(*this is for pattern matching, so we only evaluate the arguments once if using form for Rule, and never if we are using RuleDelayed*)
SetAttributes[MQTTClientSetOptions,{HoldRest}];

(*MQTTClientSetOptions sets options for the client, mainly used for changing Callback functions after the corresponding function has been called*)
(*for example, calling 
	ClientConnect[client, "ClientConnectedFunction" :> Print["got connected"]] 
and then 
	MQTTClientSetOptions[client, "ClientConnectedFunction" :> Print["different message]]
allows the callback function for connecting to be changed after ClientConnect
*)
MQTTClientSetOptions[client_MQTTClient,settings_]:=Module[
	{
		clientID = client["ClientID"],
		(*for the option, we want to try and extract the option name, with the expectation that settings is of the form opt->val or opt:>val*)
		opt = Quiet[Check[Extract[HoldComplete[settings],{1,1}],$Failed],Extract::partd]
	},
	(
		(*make sure that the client exists*)
		If[KeyExistsQ[$Clients,clientID],
			(*THEN*)
			(*client exists, so check the option*)
			(
				If[KeyExistsQ[$Clients[clientID],opt],
					(*THEN*)
					(*key exists, check if it's settable*)
					(
						If[Not[MemberQ[$ProtectedMQTTClientProperties,opt]],
							(*THEN*)
							(*it's a user configurable option*)
							(
								(*see if we want to SetDelayed it or Set it*)
								Switch[HoldComplete[settings],
									HoldComplete[RuleDelayed[___]]|HoldComplete[List[___]],
									(*form for f[ client,"CallbackFunction" :> Print["callback evaled"] ] or for f[ client, { "opt" , val } ]*)
									(
										$Clients[clientID,opt] := Extract[HoldComplete[settings],{1,2}]
									),
									HoldComplete[Rule[___]], (*form for f[ client, opt -> val] where it's okay to eval the rhs*)
									(
										$Clients[clientID,opt] = Extract[HoldComplete[settings],{1,2}]
									),
									_, (*anything else message and exit*)
									(
										Message[General::argx,MQTTClientSetOptions];
										$Failed
									)
								]
							),
							(*ELSE*)
							(*it's a protected option, we don't want the user's messing with it*)
							(
								Message[MQTTClientSetOptions::protectedKey,opt];
								$Failed
							)
						]
					),
					(*ELSE*)
					(*key doesn't exist, can't set it*)
					(
						Message[MQTTClientSetOptions::nokey,opt];
						$Failed
					)
				]
			),
			(*ELSE*)
			(*client doesn't exist*)
			(
				Message[MQTTClientSetOptions::nonexist,clientID];
				$Failed
			)
		]
	)
];

(*convenience syntax for f[ client , opt, val ], as opposed to f[ client, { opt, val} ]*)
MQTTClientSetOptions[client_MQTTClient,opt_,val_]:=MQTTClientSetOptions[client,{opt,val}];


(*internal association for keeping track of all clients*)
$Clients=<||>;

(*client image is the image to use for the summary box / content box*)
clientImage=ImageResize[Import[PacletResource["MQTTLink","ClientImage"]],50];

Options[CreateClient]=
{
	"CleanSession"->False,
	"Timeout"->Automatic,
	"ClientID"->Automatic,
	"ClientConnectedFunction"->Null,
	"ClientDisconnectedFunction"->Null,
	"Username"->None,
	"Password"->None
};


CreateClient[host_?(MatchQ[#,_IPAddress|_URL]&),OptionsPattern[]]:=With[
	{
		stringsFromHost = Cases[host,_String,Infinity]
	},
	If[Length[stringsFromHost]>0,
		(*THEN*)
		(*we found a string from the host*)
		CreateClient[
			(*just use the first string*)
			First[stringsFromHost],
			"Timeout"->OptionValue["Timeout"],
			"ClientID"->OptionValue["ClientID"],
			"CleanSession"->OptionValue["CleanSession"],
			"ClientConnectedFunction":>OptionValue["ClientConnectedFunction"],
			"ClientDisconnectedFunction":>OptionValue["ClientDisconnectedFunction"],
			"Username"->OptionValue["Username"],
			"Password"->OptionValue["Password"]
		],
		(*ELSE*)
		(*no string inside it, error message*)
		(
			Message[CreateClient::invalidHost,Short[host]];
			$Failed
		)
	]	
];

CreateClient[host_?(MatchQ[#,_IPAddress|_URL]&),port_Integer,OptionsPattern[]]:=With[
	{
		stringsFromHost = Cases[host,_String,Infinity]
	},
	If[Length[stringsFromHost]>0,
		(*THEN*)
		(*we found a string from the host*)
		CreateClient[
			(*just use the first string and the specified port*)
			First[stringsFromHost],
			port,
			"Timeout"->OptionValue["Timeout"],
			"ClientID"->OptionValue["ClientID"],
			"CleanSession"->OptionValue["CleanSession"],
			"ClientConnectedFunction":>OptionValue["ClientConnectedFunction"],
			"ClientDisconnectedFunction":>OptionValue["ClientDisconnectedFunction"],
			"Username"->OptionValue["Username"],
			"Password"->OptionValue["Password"]
		],
		(*ELSE*)
		(*no string inside it, error message*)
		(
			Message[CreateClient::invalidHost,Short[host]];
			$Failed
		)
	]	
];


CreateClient[OptionsPattern[]]:=Block[
	{
		clientID = OptionValue["ClientID"],
		username = OptionValue["Username"],
		password = OptionValue["Password"],
		cleanSession,
		createStatus,
		passwordStatus
	},
	(
		(*first check the value of persistent session and use that for CleanSession (they are synonymous)*)
		cleanSession=If[MemberQ[{True,False},OptionValue["CleanSession"]],
			(*THEN*)
			(*we can just use the option value set*)
			OptionValue["CleanSession"],
			(*ELSE*)
			(*invalid option value, so just return $Failed with a message*)
			(
				Message[CreateClient::cleanSession,OptionValue["CleanSession"]];
				Return[$Failed];
			)
		];
		clientID = Which[
			MemberQ[{Automatic,Default},clientID], 
			(*make a compacted id with CreateUUID*)
			(
				uuidCompact[CreateUUID[]]
			),
			(*the client id needs to be a string between 1 and 65535 characters long, with just the valid alphanumeric characters in it*)
			Not@StringQ[clientID] || (# < 1 || # > 65535 & @ StringLength[clientID]) || Complement[Characters@clientID,$ValidMQTTClientIDCharacters] =!= {},
			(
				Message[CreateClient::clientIDInvalid,clientID];
				Return[$Failed];
			),
			(*if it already exists in the client association*)
			KeyExistsQ[clientID]@$Clients,
			(
				Message[CreateClient::exists,clientID];
				Return[$Failed];
			),
			(*passed all the requirements*)
			True,
			(
				clientID
			)
		];
		(*check if the username and password are valid*)
		username = Which[
			MemberQ[{Automatic,None,Default},username], 
			(*no username specified*)
			(
				None
			),
			(*the mqtt standard doesn't impose any restrictions on the username / passwords, just that they be strings*)
			Not@StringQ[username],
			(
				Message[CreateClient::clientUsername,username];
				Return[$Failed];
			),
			(*passed all the requirements*)
			True,
			(
				username
			)
		];
		
		(*check the password as well*)
		password = Which[
			username === None,
			(*the username is none, so the password should also be none/automatic/default*)
			(
				If[Not@MemberQ[{Automatic,None,Default},password],
					(*THEN*)
					(*specified password with no username, which is invalid*)
					(
						Message[CreateClient::noUsername];
						Return[$Failed];
					),
					(*ELSE*)
					(*just use None*)
					None
				]
			),
			MemberQ[{Automatic,None,Default},password], 
			(*password is default, which means we just use the username*)
			(
				None
			),
			(*the mqtt standard doesn't impose any restrictions on the username / passwords, just that they be strings*)
			Not@StringQ[password],
			(
				Message[CreateClient::clientPassword,password];
				Return[$Failed];
			),
			(*passed all the requirements*)
			True,
			(
				password
			)
		];
		(*make a client in the C library*)
		createStatus=iCreateClient[clientID,cleanSession];
		(*now we need to check the result to make sure that the client was created correctly*)
		Which[
			createStatus===$MosquittoSuccess,(*no errors, successfully made*)
			(
				(*now see if we have to set the username password*)
				If[username =!= None,
					(*THEN*)
					(*username exists, specified it*)
					(
						passwordStatus = iSetPassword[clientID, password =!=None ,username, password/.None->""];
						(*now check the result of setting the password*)
						Switch[passwordStatus,
							$MosquittoSuccess,(*nothing wrong - continue*)
							Null,
							$ClientNotFound,(*for some reason the client that was just created doesn't exist yet...*)
							(
								Return[$Failed]
							),
							$MosquittoOutOfMemory,(*not enough memory for storing the password, etc.*)
							(
								Message[General::nomem];
								Return[$Failed];
							),
							_,(*if we get here just return $Failed, somehow the function returned an illegal return value*)
							Return[$Failed];
						]
					)
				]
			),
			createStatus===$MosquittoOutOfMemory,(*not enough memory to create the client*)
			(
				Message[General::nomem];
				Return[$Failed];
			),
			createStatus===$ClientExists,(*client already exists*)
			(
				(*TODO: maybe see if we can find this client already and return that as if it exists in C library, it should also probably exist in WL*)
				Message[CreateClient::exists,clientID];
				Return[$Failed];
			),
			True,(*if we get here just return $Failed, somehow the function returned an illegal return value*)
			(
				Return[$Failed];
			)
		];
		AppendTo[
			$Clients,
			clientID->
				<|
					"Connected"->False,
					"ClientID"->clientID,
					"BrokerIP"->None,
					"BrokerPort"->None,
					"Timeout"->Automatic,
					"SubscribedTopics"-><||>,
					"CleanSession"->cleanSession,
					"MessageReceivedFunction"-><||>,
					"SubscriptionCompleteFunction"-><||>,
					"UnsubscribeCompleteFunction"-><||>,
					"PublishCompleteFunction"-><||>,
					"ClientConnectedFunction":>OptionValue["ClientConnectedFunction"],
					"ClientDisconnectedFunction":>OptionValue["ClientDisconnectedFunction"]
				|>
		];
		Return[MQTTClient[
				<|
					"Connected"->With[
							{idVal=clientID},
							Deploy@Graphics[
								{
									Dynamic[
										If[ValueQ[$Clients]&&KeyExistsQ[idVal]@$Clients&&$Clients[idVal,"Connected"],
											(*THEN*)
											(*we have an association and this client exists in that association*)
											Darker[Green],
											(*ELSE*)
											(*we don't have an association or this client doesn't exist in that association*)
											Red
										]
									],
									Disk[]
								},
								ImageSize->8
							]
						],
					"ClientID"->clientID,
					"BrokerIP"->With[{idVal=clientID},Dynamic[If[ValueQ[$Clients]&&KeyExistsQ[idVal]@$Clients,$Clients[idVal,"BrokerIP"],None]]],
					"BrokerPort"->With[{idVal=clientID},Dynamic[If[ValueQ[$Clients]&&KeyExistsQ[idVal]@$Clients,$Clients[idVal,"BrokerPort"],None]]],
					"SubscribedTopics"->With[{idVal=clientID},Dynamic[If[ValueQ[$Clients]&&KeyExistsQ[idVal]@$Clients,$Clients[idVal,"SubscribedTopics"],None]]]
				|>
			]
		];
	)
];

CreateClient[ip_String,port_Integer,OptionsPattern[]]:=
(
	If[# === $Failed,
		(*THEN*)
		(*the CreateClient call failed, return $Failed *)
		$Failed,
		(*ELSE*)
		(*the client got created successfully, now try connecting it*)
		ConnectClient[#,ip,port,"Timeout"->OptionValue["Timeout"]]
	]& @ CreateClient[
		"ClientID"->OptionValue["ClientID"],
		"CleanSession"->OptionValue["CleanSession"],
		"ClientConnectedFunction":>OptionValue["ClientConnectedFunction"],
		"ClientDisconnectedFunction":>OptionValue["ClientDisconnectedFunction"],
		"Username"->OptionValue["Username"],
		"Password"->OptionValue["Password"]
	]
)


CreateClient[ip_String,OptionsPattern[]]:=
(
	If[# === $Failed,
		(*THEN*)
		(*the CreateClient call failed, return $Failed *)
		$Failed,
		(*ELSE*)
		(*the client got created successfully, now try connecting it*)
		ConnectClient[#,ip,"Timeout"->OptionValue["Timeout"]]
	]& @ CreateClient[
		"ClientID"->OptionValue["ClientID"],
		"CleanSession"->OptionValue["CleanSession"],
		"ClientConnectedFunction":>OptionValue["ClientConnectedFunction"],
		"ClientDisconnectedFunction":>OptionValue["ClientDisconnectedFunction"],
		"Username"->OptionValue["Username"],
		"Password"->OptionValue["Password"]
	]
)

Options[ConnectClient]=
{
	"Timeout"->Automatic
};

ConnectClient[client_MQTTClient,ip_String,port_Integer,OptionsPattern[]]:=Module[
	{
		clientID=client["ClientID"],
		timeout = OptionValue["Timeout"],
		connectResult
	},
	(*check the value of the timeout*)
	Which[
		MemberQ[{Automatic,Default},timeout],
		(*default/automatic value, check if one was set for this client previously*)
		timeout = If[IntegerQ[$Clients[clientID,"Timeout"]],
			$Clients[clientID,"Timeout"],
			10
		],
		(*now check if it was specified to be something, then use that one*)
		IntegerQ[timeout],
		(*note that because we have already assigned to timeout the option value, and we just figured out that it's an integer, we don't have to reassign to it*)
		Null,
		(*ELSE*)
		(*invalid option, return $Failed*)
		True,
		(
			Message[ConnectClient::timeout,timeout];
			$Failed
		)
	];
	Which[
		Not[MemberQ[Keys[$Clients],clientID]],
		(*client doesn't exist*)
		(
			(*TODO: create the client with these properties, as this object was probably copied over or an old one from a notebook that was just opened*)
			(*so it doesn't make sense to the user that the client doesn't exist*)
			(*for now just fail and complain about the client not existing*)
			Message[ConnectClient::nonexist,clientID];
			$Failed
		),
		(*the client is already connected and exists, so don't do anything*)
		$Clients[clientID,"Connected"],
		(
			Message[ConnectClient::alreadyConnected,clientID];
			$Failed
		),
		Not[$Clients[clientID,"Connected"]],
		(*then the client isn't connected and exists, so we can connect it*)
		(
			(*set the timeout for this client in the association*)
			$Clients[clientID,"Timeout"]=timeout;
			(*connect the client using an asynchronous task*)
			Quiet[
				Internal`CreateAsynchronousTask[
					(Function[{first,last},connectResult=first;last]@@iConnectClient[clientID,ip,port,timeout])&,
					{},
					Function[{taskObject,msg,data},
						Which[
							msg === "connection",(*successfully connected*)
							(
								(*normal connection, set the appropriate property to true*)
								$Clients[clientID]["Connected"] = True;
								
								(*update the value of this client's BrokerIP and BrokerPort*)
								$Clients[clientID,"BrokerIP"]=ip;
								$Clients[clientID,"BrokerPort"]=port;

								(*finally the client's connected callback*)
								$Clients[clientID,"ClientConnectedFunction"][$AsynchronousTask,client,data];
							),
							msg === "connection_failure",
							(
								(*failed to connect properly, the data will contain reason why, so issue a message*)
								$Clients[clientID]["Connected"] = False;
								Switch[First[data],
									$MosquittoConnectionRefusedProtoVersion,
									(
										Message[ConnectClient::connFail,"The broker rejected the connection request due to an incompatible protocol version"];
									),
									$MosquittoConnectionRefusedIdentifierRejected,
									(
										Message[ConnectClient::connFail,"The broker rejected the connection request due to an invalid identifier"];
									),
									$MosquittoBrokerUnavailable,
									(
										Message[ConnectClient::connFail,"The broker specified is unavailable"];
									),
									$MosquittoUsernamePasswordMalformed,
									(
										Message[ConnectClient::connFail,"The username/password for the client was malformed"];
									),
									$MosquittoNotAuthorized,
									(
										Message[ConnectClient::connFail,"The client is not authorized to connect to the broker"];
									),
									_,
									(
										Message[ConnectClient::connFail,"Unable to connect to the specified broker (code : "<>ToString[First[data]]<>")"];
									)
								];
							),
							msg === "expected_disconnect",
							(
								(*normal expected disconnect, so just set the connected status to false*)
								$Clients[clientID]["Connected"] = False;
							),
							msg === "unexpected_disconnect",
							(
								(*unexpected disconnect, so raise a message about reason why*)
								$Clients[clientID]["Connected"] = False;
								Message[ConnectClient::unexpectDisconn,clientID,First[data]];
							),
							True,
							(
								(*unknown event type*)
								$Clients[clientID]["Connected"] = False;
								Message[ConnectClient::unexpectDisconn,clientID,data];
							)
						];
						(*now call the user's callback for disconnection as appropriate*)
						If[msg =!= "connection",
							(*THEN*)
							(*call the disconnection callback with the code*)
							(
								(*finally, because we got disconnected, check to see if this client had a clean session or not so we can update*)
								(*the subscriptions appropriately*)
								If[Not[$Clients[clientID,"CleanSession"]],
									(*THEN*)
									(*only need to remove QoS level 0 subscriptions*)
									(
										(*delete all the qos 0 subscriptions*)
										KeyDropFrom[$Clients[clientID,"SubscribedTopics"], Keys[Select[#===0&]@$Clients[clientID,"SubscribedTopics"]]];
									),
									(*ELSE*)
									(*we need to clear all subscription state*)
									(
										$Clients[clientID,"SubscribedTopics"]=<||>;
									)
								];
								
								(*now call the user's specified callback for disconnection if it exists*)
								$Clients[clientID,"ClientDisconnectedFunction"][$AsynchronousTask,client,data];
							)
						];
					],
					"TaskDetail"->clientID,
					"Visible"->False
				],
				{Internal`CreateAsynchronousTask::noid,StringForm::sfr}
			];
			Switch[connectResult,
				$MosquittoSuccess,(*no error*)
				(
					(*just return the client back to the user*)
					Return[client]
				),
				$ThreadError, (*the thread failed to start for some reason*)
				(
					Message[ConnectClient::threadFail];
					Return[$Failed];
				),
				$MosquittoLookupError, (*failed to find host or some other EAI error in getaddrinfo*)
				(
					Message[ConnectClient::noBroker,ip];
					Return[$Failed];
				),
				$MosquittoErrnoError|_, (*any other case use the errno message*)
				(
					Message[ConnectClient::errno,ip];
					Return[$Failed];
				)
			];
		)
	]
];

(*when the user just specifies the ip address and no string, check for port, if not found use 1883*)
ConnectClient[client_MQTTClient,ip_String,OptionsPattern[]]:=Block[
	{
		(*this searches through the ip string passed in by the user, looking for specifically the pattern*)
		(* :XXXXXX...  where X is any number and no other characters, and that there is also at least one number *)
		(*this is meant to match cases like localhost:8000 and test.mosquitto.org:1883 and not cases like http://localhost, or www.test.mosquitto.org*)
		(*StringCases returns back all cases where it matches in the string, but all cases include the ":", so we use StringDrop to drop that from any strings found*)
		(*we know that dropping the string will always work, because there must be at least 1 number for StringCases to match to it*)
		(*Finally, we can use FromDigits[...,10] to turn the integer string into a Integer expr for passing along to the final form of ConnectClient where it is actually used*)
		ports = FromDigits[StringDrop[#, 1], 10] & /@ StringCases[ip,":" ~~ DigitCharacter ..],
		port
	},
	(*check if a port was found*)
	(
		If[Length[ports]>0,
			(*THEN*)
			(*the port was in the ip string, use that and remove the port spec from the ip*)
			(
				port = First[ports];
				ConnectClient[client,StringReplace[ip,":"<>ToString@port->"",1],port,"Timeout"->OptionValue["Timeout"]]
			),
			(*ELSE*)
			(*port is missing from the string - use default and issue message*)
			(
				Message[ConnectClient::noPortFound,ip];
				$Failed
			)
		]
	)
];

ConnectClient[client_MQTTClient,host_?(MatchQ[#,_IPAddress|_URL]&),OptionsPattern[]]:=With[
	{
		(*extract any strings out from the IPAddress or URL object*)
		stringsFromHost = Cases[host,_String,Infinity]
	},
	If[Length[stringsFromHost]>0,
		(*THEN*)
		(*we found a string from the host*)
		ConnectClient[
			client,
			(*use the first string we found as the likely candidate for the hostname*)
			First[stringsFromHost],
			"Timeout"->OptionValue["Timeout"]
		],
		(*ELSE*)
		(*no string inside it, error message*)
		(
			Message[ConnectClient::invalidHost,Short[host]];
			$Failed
		)
	]
];

ConnectClient[client_MQTTClient,host_?(MatchQ[#,_IPAddress|_URL]&),port_Integer,OptionsPattern[]]:=With[
	{
		(*extract any strings out from the IPAddress or URL object*)
		stringsFromHost = Cases[host,_String,Infinity]
	},
	If[Length[stringsFromHost]>0,
		(*THEN*)
		(*we found a string from the host*)
		ConnectClient[
			client,
			(*use the first string we found as the likely candidate for the hostname*)
			First[stringsFromHost],
			port,
			"Timeout"->OptionValue["Timeout"]
		],
		(*ELSE*)
		(*no string inside it, error message*)
		(
			Message[ConnectClient::invalidHost,Short[host]];
			$Failed
		)
	]	
]


ConnectClient[client_MQTTClient,OptionsPattern[]]:=Module[
	{
		hostIP=client["BrokerIP"],
		hostPort=client["BrokerPort"]
	},
	Which[
		MissingQ[$Clients[client["ClientID"]]],(*then the client passed in doesn't exist*)
		(
			(*TODO: create the client with these properties, as this object was probably copied over or an old one from a notebook that was just opened*)
			(*so it doesn't make sense to the user that the client doesn't exist*)
			(*for now just fail and complain about the client not existing*)
			Message[ConnectClient::nonexist,clientID];
			$Failed
		),
		hostIP===None&&hostPort===None, (*then this one doesn't have any ports specified, assume localhost and 1883*)
		(
			Message[ConnectClient::noIP];
			Message[ConnectClient::noPort];
			ConnectClient[client,"localhost",1883,"Timeout"->OptionValue["Timeout"]]
		),
		hostIP===None&&hostPort=!=None,(*port exists, ip doesn't*)
		(
			(*note this is an extremely odd situtation, so issue a specific message with how to use ConnectClient and the spec for the ip*)
			Message[ConnectClient::noIPFound];
			$Failed
		),
		hostIP=!=None&&hostPort===None,(*ip exists, port doesn't*)
		(
			(*try calling the string version in case the hostIP has a port specified to it*)
			ConnectClient[client,hostIP,"Timeout"->OptionValue["Timeout"]]
		),
		True, (*the last case is that both have appropriate values*)
		ConnectClient[client,hostIP,hostPort,"Timeout"->OptionValue["Timeout"]]
	]	
];

(*note that in all circumstances where the client is legitimately connected before this call, the ClientDisconnectedFunction callback option will be called during this*)
(*function call, possibly after due to how the kernel ends up handling the AsynchronousTask, but this function is guaranteed to trigger the callback to be queued*)
DisconnectClient[client_MQTTClient]:=Module[
	{
		clientID = client["ClientID"],
		disconnectResult
	},
	(
		(*check to make sure the clientID is valid*)
		If[Not[MissingQ[$Clients[client["ClientID"]]]],
			(*THEN*)
			(*client id exists, we can disconnect it*)
			(
				disconnectResult = iDisconnectClient[clientID];
				Switch[disconnectResult,
					$MosquittoSuccess, (*no error - set the state to be disconnected*)
					(
						(*when we disconnect, we need to check the subscriptions.*)
						(*if this client was connected with CleanSession->False, then if it is reconnected later on, it will keep QualityOfService level 2 or 1*)
						(*subscriptions, while all QualityOfService subscriptions are lost*)
						(*also note that no subscriptions are kept if CleanSession->True, as when it disconnects now, all state is lost on the broker about*)
						(*this particular client*)
						If[Not[$Clients[clientID,"CleanSession"]],
							(*THEN*)
							(*only need to remove QoS level 0 subscriptions*)
							(
								(*delete all the qos 0 subscriptions*)
								KeyDropFrom[$Clients[clientID,"SubscribedTopics"], Keys[Select[#===0&]@$Clients[clientID,"SubscribedTopics"]]];
							),
							(*ELSE*)
							(*we need to clear all subscription state*)
							(
								$Clients[clientID,"SubscribedTopics"]=<||>;
							)
						];
						$Clients[clientID,"Connected"] = False;
						client
					),
					$ClientNotFound,(*clientID is invalid*)
					(
						Message[DisconnectClient::nonexist,clientID];
						$Failed
					),
					_,
					(
						Message[DisconnectClient::unknownError,clientID];
						$Failed
					)
				]
			),
			(*ELSE*)
			(*clientID doesn't exist, issue message and return $Failed*)
			(
				Message[DisconnectClient::nonexist,clientID];
				$Failed
			)
		]
	)
];


Options[TopicPublish]=
{
	"QualityOfService"->2,
	"Retain"->False,
	"PublishCompleteFunction"->Null
}; 

TopicPublish[client_MQTTClient,topic_String->message_ByteArray,OptionsPattern[]]:=
	TopicPublish[
		client,
		topic,
		Normal[message],
		"QualityOfService"->OptionValue["QualityOfService"],
		"Retain"->OptionValue["Retain"],
		"PublishCompleteFunction":>OptionValue["PublishCompleteFunction"]
	];
	
TopicPublish[client_MQTTClient,topic_String,message_ByteArray,OptionsPattern[]]:=
	TopicPublish[
		client,
		topic,
		Normal[message],
		"QualityOfService"->OptionValue["QualityOfService"],
		"Retain"->OptionValue["Retain"],
		"PublishCompleteFunction":>OptionValue["PublishCompleteFunction"]
	];

TopicPublish[client_MQTTClient,topic_String->message_String,OptionsPattern[]]:=
	TopicPublish[
		client,
		topic,
		ToCharacterCode[message,"UTF-8"],
		"QualityOfService"->OptionValue["QualityOfService"],
		"Retain"->OptionValue["Retain"],
		"PublishCompleteFunction":>OptionValue["PublishCompleteFunction"]
	];

TopicPublish[client_MQTTClient,topic_String,message_String,OptionsPattern[]]:=
	TopicPublish[
		client,
		topic,
		ToCharacterCode[message,"UTF-8"],
		"QualityOfService"->OptionValue["QualityOfService"],
		"Retain"->OptionValue["Retain"],
		"PublishCompleteFunction":>OptionValue["PublishCompleteFunction"]
	];

TopicPublish[client_MQTTClient,topic_String->message_List,OptionsPattern[]]:=
	TopicPublish[
		client,
		topic,
		message,
		"QualityOfService"->OptionValue["QualityOfService"],
		"Retain"->OptionValue["Retain"],
		"PublishCompleteFunction":>OptionValue["PublishCompleteFunction"]
	];
	
TopicPublish[client_MQTTClient,topic_String,message_List,OptionsPattern[]]:=Module[
	{
		clientID = client["ClientID"],
		messageLength=Length[message],
		publishResult,
		qos,
		retain,
		asyncTask,
		messageID = CreateUUID[]
	},
	(
		Which[
			(*the client doesn't exist*)
			Not[KeyExistsQ[$Clients,clientID]],
			(
				Message[TopicPublish::nonexist,clientID];
				Return[$Failed];
			),
			(*the client isn't connected to a broker*)
			(*TODO: have a check in here that if we can reconnect the client, then do so before failing*)
			Not[$Clients[clientID,"Connected"]],
			(
				Message[TopicPublish::notConnected,clientID];
				Return[$Failed];
			),
			(*qos option is valid*)
			Not[MemberQ[{0,1,2},OptionValue["QualityOfService"]]],
			(
				Message[TopicPublish::qos,OptionValue["QualityOfService"]];
				Return[$Failed];
			),
			(*retain is valid*)
			Not[MemberQ[{True,False},OptionValue["Retain"]]],
			(
				Message[TopicPublish::retain,OptionValue["Retain"]];
				Return[$Failed];
			),
			(*message isn't too long*)
			messageLength>268435455,
			(
				Message[TopicPublish::messageTooLarge,Short[message]];
				Return[$Failed];
			),
			(*the topic name is valid*)
			Not@validTopicNameQ[topic],
			(
				Message[TopicPublish::topicNameInvalid,topic];
				Return[$Failed];
			),
			(*finally, see if the message itself has non-byte  values*)
			Not[ByteArrayQ[ByteArray[message]]],
			(
				Message[TopicPublish::messageutf,Short[message]];
				Return[$Failed];
			),
			(*everything is valid*)
			True,
			(
				qos = OptionValue["QualityOfService"];
				retain = OptionValue["Retain"];
				(*also set the callback for this message*)
				$Clients[clientID,"PublishCompleteFunction",messageID] := OptionValue["PublishCompleteFunction"];
				If[qos==0,
					(*THEN*)
					(*quality of service is 0, so there won't be any response from the broker*)
					(
						(*just run the librarylink function normally*)
						publishResult=First[
							iTopicPublish[
								clientID,
								topic,
								message,
								qos,
								retain
							]
						];
						
						(*now just call the PublishCompleteFunction callback now, as there won't be any response from the broker*)
						$Clients[clientID,"PublishCompleteFunction",messageID][$AsynchronousTask,client,messageID];
						
						(*we return the task, so just set it to Null in this case*)
						Return[<|"MessageID"->messageID|>];
					),
					(*ELSE*)
					(*quality of service is more than 0, so there will be a response from the broker*)
					(
						(*run the publish via CreateAsynchronousTask*)
						asyncTask = Quiet[
							Internal`CreateAsynchronousTask[
								(Function[{first,last},publishResult=first;last]@@iTopicPublish[
									clientID,
									topic,
									message,
									qos,
									retain
								])&,
								{},
								Function[{taskObject,msg,data},
									Which[
										msg=="publish_success",(*successfully published*)
										(
											(*successfully published on the topic - call the PublishCompleteFunction callback*)
											$Clients[clientID,"PublishCompleteFunction",messageID][$AsynchronousTask,client,messageID]
										),
										True,
										(
											Message[TopicPublish::unknownError,data];
										)
									];
									
									(*finally remove this asynchronous task*)
									RemoveAsynchronousTask[$AsynchronousTask];
								],
								"TaskDetail"->topic,
								"Visible"->False
							],
							{Internal`CreateAsynchronousTask::noid,StringForm::sfr}
						]
					)
				];

				(*check the result to make sure it was sent and we didn't get an error*)
				Switch[publishResult,
					$ClientNotFound,
					(
						(*if this ever returns then the client was killed somehow*)
						Return[$Failed];
					),
					$MosquittoSuccess,
					(
						(*success*)
						Return[<|"MessageID"->messageID|>];
					),
					$MosquittoInvalidFunctionArguments,
					(
						Return[$Failed];
					),
					$MosquittoOutOfMemory,
					(
						(*not enough memory to create the message packet*)
						Message[General::nomem];
						Return[$Failed];
					),
					$MosquittoNoConnection,
					(
						Message[TopicPublish::noConnection,clientID];
						Return[$Failed];
					),
					$MosquittoProtocolError,
					(
						Message[TopicPublish::protocolError];
						Return[$Failed];
					),
					$MosquittoPayloadTooLarge,
					(
						Message[TopicPublish::messageTooLarge,Short[message]];
						Return[$Failed];
					),
					_,(*any other error*)
					(
						(*shouldn't be any other error, but just in case*)
						Return[$Failed];
					)
				]
			)
		]
	)
];


Options[TopicSubscribe]=
{
	"QualityOfService"->2,
	"SubscriptionCompleteFunction"->Null,
	"MessageReceivedFunction"->Null
};
TopicSubscribe[client_MQTTClient,topicPattern_String,OptionsPattern[]]:=Module[
	{
		clientID=client["ClientID"],
		qos,
		subscribeResult,
		asyncTask,
		taskResult
	},
	(
		(*validate the options first*)
		qos=If[MemberQ[{0,1,2},OptionValue["QualityOfService"]],
			(*THEN*)
			(*option is valid, sue whatever it's current value is*)
			OptionValue["QualityOfService"],
			(*ELSE*)
			(*option is invalid, raise message and return $Failed*)
			(
				Message[TopicSubscribe::qos,OptionValue["QualityOfService"]];
				Return[$Failed];
			)
		];
		(*now check the topic filter*)
		If[Not@validTopicFilterQ[topicPattern],
			(*THEN*)
			(*the topic name is invalid issue a message and return $Failed*)
			(
				Message[TopicSubscribe::invalidTopicFilter,topicPattern];
				Return[$Failed];
			)
		];
		(*set the options in $Clients for the MessageReceivedFunction and SubscriptionConfirmed callbacks*)
		$Clients[clientID,"SubscriptionCompleteFunction",topicPattern] := OptionValue["SubscriptionCompleteFunction"];
		$Clients[clientID,"MessageReceivedFunction",topicPattern] := OptionValue["MessageReceivedFunction"];
		(*check on whether the topic pattern is already subscribed to*)
		If[Not[KeyExistsQ[$Clients[clientID,"SubscribedTopics"],topicPattern]],
			(*THEN*)
			(*not subscribed to yet*)
			(
				asyncTask = Quiet[
					Internal`CreateAsynchronousTask[
						(Function[{first,last},subscribeResult=first;last]@@iTopicSubscribe[clientID,topicPattern,qos])&,
						{},
						Function[{taskObject,msg,data},
							Which[
								msg === "subscribe_success",(*successfully published*)
								(
									(*successfully subscribed to the topic, check the quality of service that we requested and make sure it matches what we were given*)
									If[qos=!=First[data],
										(*THEN*)
										(*different qos was granted then requested, post a message*)
										(
											(*it's not what was requested check if it's 128, which is code for an error*)
											If[First[data] === 128,
												(*THEN*)
												(*error subscribing, this client isn't subscribed*)
												(
													(*most likely case is insufficient priveledges, but could be other reasons*)
													Message[TopicSubscribe::subFailed,topicPattern,qos,First[data]];
													(*returning $Failed doesn't do anything here, so just return*)
													Return[];
												),
												(*ELSE*)
												(*just different qos level, raise message and continue*)
												(
													Message[TopicSubscribe::diffQos,topicPattern,qos,First[data]];
												)
											];
										)
									];
									(*save the level qos in the SubscribedTopics association*)
									AppendTo[$Clients[clientID,"SubscribedTopics"],topicPattern->First[data]];
									
									(*now create the topic subscription task to handle messages on this topic*)
									Quiet[
										Internal`CreateAsynchronousTask[
											(Function[{first,last},taskResult=first;last]@@iCreateTopicSubscriptionTask[clientID,topicPattern])&,
											{},
											Function[{taskObject2,msg2,data2},
												(*we got a message on the topic*)
												Which[
													msg2==="message_recieved",(*message received successfully*)
													(
														(*call the user's specified callback function*)
														$Clients[clientID,"MessageReceivedFunction",topicPattern][
															$AsynchronousTask,
															client,
															With[{dataAssoc = Association @ data2},
																<|
																	(*for the Timestamp option, use the timeInterpret function*)
																	"Timestamp"->timeInterpret[dataAssoc["TimestampLow"],dataAssoc["TimestampHigh"]],
																	(*for Topic, turn the MRawArray into a String*)
																	"Topic"->FromCharacterCode[Normal[dataAssoc["Topic"]],"UTF-8"],
																	(*for Data, make it into a ByteArray*)
																	"Data"->ByteArray[Normal[dataAssoc["Data"]]],
																	(*for Retained, make it a boolean*)
																	"Retained"->dataAssoc["Retained"]===1,
																	"MessageID"->dataAssoc["MessageID"],
																	"QualityOfService"->dataAssoc["QualityOfService"]
																|>
															]
														]
													),
													True, (*all other cases - issue message*)
													(
														Message[TopicSubscribe::unknownMessageCallbackError,data2];
													)
												];
											],
											"TaskDetail"->topicPattern,
											"Visible"->False
										],
										{Internal`CreateAsynchronousTask::noid,StringForm::sfr}
									];
									
									(*now check the taskResult*)
									If[taskResult =!= $MosquittoSuccess,
										(*THEN*)
										(*clientID is invalid somehow*)
										Message[TopicSubscribe::unknownError,taskResult];
									];
									
									(*call the user's SubscriptionCompleteFunction callback*)
									$Clients[clientID,"SubscriptionCompleteFunction",topicPattern][$AsynchronousTask,client];
								),
								True,
								(
									Message[TopicSubscribe::unknownError,data];
								)
							];
							
							(*finally remove this asynchronous task*)
							RemoveAsynchronousTask[$AsynchronousTask];
						],
						"Visible"->False
					],
					{Internal`CreateAsynchronousTask::noid,StringForm::sfr}
				];
				Switch[subscribeResult,
					$MosquittoSuccess,
					(
						(*return back the client*)
						Return[client];
					),
					$MosquittoInvalidFunctionArguments,
					(
						Return[$Failed];
					),
					$MosquittoOutOfMemory,
					(
						(*not enough memory to create the message packet*)
						Message[General::nomem];
						Return[$Failed];
					),
					$MosquittoNoConnection,
					(
						Message[TopicSubscribe::noConnection,clientID];
						Return[$Failed];
					),
					_,(*any other error*)
					(
						(*shouldn't be any other error, but just in case*)
						Return[$Failed];
					)
				]
			),
			(*ELSE*)
			(*already subscribed*)
			(
				Message[TopicSubscribe::alreadySubscribed,clientID,topicPattern];
				Return[$Failed];
			)
		];
	)
];


Options[TopicUnsubscribe]=
{
	"UnsubscribeCompleteFunction"->Null
};

TopicUnsubscribe[client_MQTTClient,topicPattern_String,OptionsPattern[]]:=Module[
	{
		clientID=client["ClientID"],
		unsubscribeResult,
		asyncTask
	},
	(
		(*make sure the client exists first*)
		If[KeyExistsQ[$Clients,clientID],
			(*THEN*)
			(*check if we are connected next*)
			(
				If[$Clients[clientID,"Connected"],
					(*THEN*)
					(*client is connected, and thus could hypothetically be disconnected*)
					(
						(*check whether we are actually subscribed to the topic*)
						If[KeyExistsQ[$Clients[clientID,"SubscribedTopics"],topicPattern],
							(*THEN*)
							(*subscribed to, so we can unsubscribe*)
							(
								(*first set the callback function for unsubscribing from this topic*)
								$Clients[clientID,"UnsubscribeCompleteFunction",topicPattern] := OptionValue["UnsubscribeCompleteFunction"];
								(*now actually perform the task*)
								asyncTask = Quiet[
									Internal`CreateAsynchronousTask[
										(Function[{first,last},unsubscribeResult=first;last]@@iTopicUnsubscribe[clientID,topicPattern])&,
										{},
										Function[{taskObject,msg,data},
											Which[
												msg=="unsubscribe_success",(*successfully unsubscribed*)
												(
													(*success, remove this topic from the list of subscribed topics*)
													$Clients[clientID,"SubscribedTopics"]=KeyDrop[$Clients[clientID,"SubscribedTopics"],topicPattern];
													
													(*now also call the callback for this unsubscribe*)
													$Clients[clientID,"UnsubscribeCompleteFunction",topicPattern][$AsynchronousTask,client,topicPattern];
												),
												True,
												(
													Message[TopicUnsubscribe::unknownError,data];
												)
											];
											
											(*finally remove this asynchronous task*)
											RemoveAsynchronousTask[$AsynchronousTask];
										]
									],
									{Internal`CreateAsynchronousTask::noid,StringForm::sfr}
								];
								Switch[unsubscribeResult,
									$MosquittoSuccess,
									(
										Return[asyncTask];
									),
									$MosquittoInvalidFunctionArguments,
									(
										$Failed
									),
									$MosquittoOutOfMemory,
									(
										(*not enough memory to create the message packet*)
										Message[General::nomem];
										$Failed
									),
									$MosquittoNoConnection,
									(
										Message[TopicUnsubscribe::noConnection,clientID];
										$Failed
									),
									_,(*any other error*)
									(
										(*shouldn't be any other error, but just in case*)
										$Failed
									)
								]
							),
							(*ELSE*)
							(*already subscribed*)
							(
								Message[TopicUnsubscribe::noSubscribe,clientID,topicPattern];
								$Failed
							)
						]
					),
					(*ELSE*)
					(*not connected*)
					(
						$Failed
					)
				]
			),
			(*ELSE*)
			(*client doesn't exist*)
			(
				Message[TopicUnsubscribe::nonexist,client];
				$Failed
			)
		]
	)
];

(*time interpret takes the time integers from the built-in OS functions and converts them to valid DateObjects*)
timeInterpret[low_Integer,high_Integer] := Module[{},
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


(*the only option to StartBroker is a list of rules/association of config rules that get copy pasted into a temporary conf file that the broker is then started with*)
Options[StartBroker]={"ConfigOptions"->Automatic};

(*function to start broker, the only argument is the port, which defaults to 1883*)
StartBroker[port_Integer:1883,OptionsPattern[]]:=Block[
	{
		(*get the mosquitto binary for this system*)
		executable=First[FileNames["mosquitto*",FileNameJoin[{PacletResource["MQTTLink","Binaries"], $SystemID}]]],
		stderr,
		proc,
		exitcode,
		configFile,
		configOptions = OptionValue["ConfigOptions"]
	},
	(
		(*check to see if there is already a broker running on this port*)
		If[Not[KeyExistsQ[$Brokers,port]],
			(*THEN*)
			(*can start up a new broker on this port*)
			(
				(*check to see if there are any config options*)
				If[Not[MemberQ[{Automatic,Default,None},configOptions]],
					(*THEN*)
					(*there are options, so we need to copy out the options to a temporary config file*)
					(
						(*turn it into a normal list, here in case it's an Association*)
						If[AssociationQ[configOptions], configOptions = Normal[configOptions]];
						If[MatchQ[configOptions, {(_String -> _String) ..}],
							(*THEN*)
							(*the configuration options are a valid list of rules, with string keys going to string values*)
							(
								(*make the config file a new temporary file*)
								configFile = CreateFile[];
								(*take the association or *)
								WriteString[configFile, StringRiffle[StringRiffle[#, " "] & /@ List @@@ configOptions,"\n"]]
							),
							(*ELSE*)
							(*the config options are invalid - raise a message*)
							(
								Message[StartBroker::invalidConfigOptions,configOptions];
								Return[$Failed];
							)
						]
						
					),
					(*ELSE*)
					(*no options, so just use the default paclet resource*)
					(
						configFile = PacletResource["MQTTLink","BrokerConfig"]
					)
				];
				(*now actually start the broker up*)
				proc = StartProcess[{executable,"-p",ToString[port],"-c",configFile}];
				If[ProcessStatus[proc]=!="Running",
					(*THEN*)
					(*error trying to start the broker*)
					(
						stderr = ReadString @ ProcessConnection[proc, "StandardError"];
						exitcode = ProcessInformation[proc, "ExitCode"];
						Message[StartBroker::startError,stderr,exitcode];
						$Failed
					),
					(*ELSE*)
					(*no error, return fine*)
					$Brokers[port] = <|"Process"->proc,"Port"->port,"ConfigFile"->configFile|>
				]
			),
			(*ELSE*)
			(*can't start it up, already have a broker on the port*)
			(
				Message[StartBroker::running,port];
				$Brokers[port]
			)
		]
	)
];

(*the With here ensures that $Brokers can't ever be modified with something like MQTTBrokers[][1883] = blah *)
MQTTBrokers[] := With[{assoc=$Brokers},assoc]

(*ensure that MQTTBrokers and StartBroker aren't modified at all*)
SetAttributes[MQTTBrokers,{Locked,Protected}]
SetAttributes[StartBroker,{Locked,Protected}]

(*these are for ensuring that topic name specifications are valid*)

(*there is a distinction between topic FILTERS and topic NAMES*)
(*when subscribing, you specify a topic FILTER*)
(*but when publishing you specify a name NAME*)
(*a topic filter can contain wildcard character, but a filter may not*)
validTopicFilterQ[pattern_String] := With[
	{
		len = StringLength[pattern],
		minMaxCharCodes = MinMax@ToCharacterCode[pattern, "Unicode"]
	},
	(
		(*first make sure that the string isn't the empty string*)
		len =!= 0 &&
		(*check to see if there's a multi-level wildcard*)
		If[StringContainsQ[pattern,"#"],
			(*THEN*)
			(*multi-level wildcards can only be the last character in a filter*)
			(*there can also only be one multi-level wildcard in a given filter*)
			(
				With[{ multilevels = StringPosition[pattern,"#"]},
					multilevels === {{len,len}} &&
					StringTake[pattern , {First[First[multilevels]] - 1}] === "/"
				]
			),
			(*ELSE*)
			(*no #, move on to the + characters*)
			True
		] &&
		If[StringContainsQ[pattern,"+"],
			(*THEN*)
			(*topic has a single level wildcard*)
			(
				(*a topic filter can have multiple single-level characters, so have to do the check for all of them*)
				Apply[And,
					(
						(*for each instance of the "+" character, make sure that it satisfies the following constraints: *)
						(* it must either be at the start of the string or have a / before it *)
						(* it must either be at the end of the string or have a / after it *)
						With[{start = First[#], end = Last[#]},
							(*this check ensures that there aren't any consecutive ++ characters*)
							start === end &&
							(*check if it's not the start that there's a / before it*)
							If[start =!= 1, StringTake[pattern, {start - 1}] === "/", True] &&
							(*check if it's not the end of the string there's a / after it*)
							If[start =!= len, StringTake[pattern, {start + 1}] === "/", True]
						]&
					)/@StringPosition[pattern, "+"]
				]
			),
			True
		] &&
		(*finally make sure that the string is a valid 3 byte UTF-8 sequence - i.e. no Unicode code points above 65536*)
		Last[minMaxCharCodes] < 65536 &&
		(*the null character (code point 0) is also disallowed in topic filters*)
		First[minMaxCharCodes] > 0
	)
];

(*a valid topic name is any UTF-8 string that *)
validTopicNameQ[pattern_String] := With[
	{minMaxCharCodes = MinMax@ToCharacterCode[pattern, "Unicode"]},
	(
		(*can't be the empty string*)
		StringLength[pattern]=!=0 &&
		(*can't include the wild-card characters*) 
		Not@StringContainsQ[pattern,"#"|"+"] &&
		(*entire string must be a valid 3 byte UTF-8 sequence - i.e. no Unicode code points above 65536*)
		Last@minMaxCharCodes < 65536 &&
		(*the null character (code point 0) is also disallowed in topic filters*)
		First[minMaxCharCodes] > 0
	)
];



(*valid client IDs can only contain alphanumeric characters, and if we just straight list them out here, we get better 
performance than say generating them with FromCharacterCode and Range*)
$ValidMQTTClientIDCharacters = {
	"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
}

(*because the MQTT standard before v3.1.1 only allowed client IDs to be 23 characters or less, if we use the
normal string created by CreateUUID, clients with automatic client IDs might be rejected, so we compress the random component
of the id to a string that's less than 23 characters long by extracting the 128-bit integer, and converting that to base 62, and using the 
digits of that base 62 number as indexes into the list of allowed characters for client IDs 
*)
uuidCompact[uuid_String] := StringJoin[
	(*remove any "-"'s from the id, then turn it into an base 10 integer then get the digits of the base 62 representation of that number*)
	(*note that if the MQTT standard changes to allow for more characters, etc dynamically getting the length prevents us from having to 
	remember to change the hard-coded base here*)
	(*then we use those digits as indexes into the list of valid characters*)
	Part[$ValidMQTTClientIDCharacters, # + 1] & /@ IntegerDigits[FromDigits[StringDelete[uuid, "-"], 16], Length[$ValidMQTTClientIDCharacters]]
]


End[] (* End Private Context *)

EndPackage[]
