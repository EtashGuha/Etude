BeginPackage["LightweightGridClient`"];

Needs["JLink`"]

General::lwgclient = "Cannot create a component needed by `1`. Remove older version of the LightweightGridClient package.";

General::lwgversion = "Conflicting versions of the LightweightGridClient package are installed.  Remove older versions.";

General::lwgconnect = "Unable to connect to `1`.  Check network connectivity and the spelling of the hostname or URL of the remote computer. Confirm that a Lightweight Grid Manager is running on the remote computer.";

General::wrsconnect = General::lwgconnect; (* backwards compatibility with version 7 servers *)

General::lwgresponse = "Received an unexpected response from `1`.  Check that the hostname or URL is correct and corresponds to a compatible Lightweight Grid Manager server.";

General::notwrsk = "`1` can only be used to connect to kernels launched with Lightweight Grid.";


(* the configuration language. A kernel is described as a LightweightGrid[{properties}] data element,
   with only a subset of the properties defined.  At a minimum the "Agent" property rule must be present,
   an optionally the "Service" and "LocalLinkMode" properties. *)
 
LightweightGrid::usage = "LightweightGrid[agent] is a description of a subkernel to be launched on agent by the Wolfram Lightweight Grid Manager
LightweightGrid[{\"Agent\" -> agent, \"KernelCount\" -> count, options}] is a description of count kernels to be launched on agent with options passed to RemoteKernelOpen..";

SetAttributes[LightweightGrid, ReadProtected];

(* the configuration language. A kernel is described as a RemoteServices[{properties}] data element,
   with only a subset of the properties defined.  At a minimum the "Agent" property rule must be present,
   an optionally the "Service" and "LocalLinkMode" properties. *)
 
RemoteServices::usage = "RemoteServices[agent] is a description of a subkernel to be launched on agent by the Wolfram Lightweight Grid Manager
RemoteServices[{\"Agent\" -> agent, \"KernelCount\" -> count, options}] is a description of count kernels to be launched on agent with options passed to RemoteKernelOpen..";

SetAttributes[RemoteServices, ReadProtected];

(* class methods and variables *)

RemoteKernelOpen::usage = "RemoteKernelOpen[spec] launches a kernel with the given specification.
RemoteKernelOpen[{spec1, spec2, ...}] launches kernels in parallel.";
SetAttributes[RemoteKernelOpen, ReadProtected];

RemoteKernelOpen::launchfailed = "Kernel could not be started on `1`.";

RemoteKernelOpen::svcstring = "Kernel could not be started on `1`. The specified value for the option \"Service\", `2`, is not a string."

RemoteKernelOpen::servicefailedinit = "The \"`2`\" service on `1` is not serving kernels because its configuration file could not be read";
(* `2` will be a user assigned name like General *)

RemoteKernelOpen::noservices = "No services could be found on `1`.";

RemoteKernelOpen::noservice = "Service \"`2`\" does not exist on `1`.  Use RemoteKernelServices to see a list of available services.";

RemoteKernelOpen::locallinkmode = 
"Kernel could not be started on `1`. \"LocalLinkMode\" `2` is not \"Create\" or \"Connect\".";

RemoteKernelOpen::linkcreatefailed = "Unable to start a kernel on `1` with \"LocalLinkMode\" -> \"Create\" because LinkCreate could not create a new TCP/IP link.";

RemoteKernelOpen::servicelaunchdisabled = "Administrators for `1` have disabled opening kernels from the \"`2`\" service.";

RemoteKernelOpen::agentlaunchdisabled = "Administrators for `1` have disabled starting kernels.";

RemoteKernelOpen::nokernels = "All kernels on `1` in the \"`2`\" service are in use.";

RemoteKernelOpen::license = "Mathematica on host `1` needs to be configured with valid license information.";

(*****************************************************)
RemoteKernelClose::usage = "RemoteKernelClose[kernel] closes a Lightweight Grid kernel.";

SetAttributes[RemoteKernelClose, ReadProtected];

RemoteKernelClose::nokernel = "Kernel could not be closed because no kernel was found for the given link \"`1`\".";

ClosedKernel::usage = "ClosedKernel[\"linkname\"] represents a Lightweight Grid kernel that was successfully closed."

(*****************************************************)

RemoteKernelCloseAll::usage = "RemoteKernelCloseAll[] closes all kernels returned by RemoteServicesLinks[].";

SetAttributes[RemoteKernelCloseAll, ReadProtected];

(*****************************************************)

RemoteServicesLinks::usage = "RemoteServicesLinks[] returns the list of all Lightweight Grid kernels that are currently open.";

SetAttributes[RemoteServicesLinks, ReadProtected];

(*****************************************************)

RemoteKernelInformation::usage = "RemoteKernelInformation[link] returns information about the Lightweight Grid kernel connected on link.
RemoteKernelInformation[] returns information about all open Lightweight Grid kernels.";

SetAttributes[RemoteKernelInformation, {ReadProtected, Listable}];

RemoteKernelInformation::nolink = "No such linkname \"`1`\" found for any
currently known remote kernel.";

RemoteKernelInformation::nokernel = "No kernel was found for the given link \"`1`\".";

RemoteServicesKernel::usage = "RemoteServicesKernel[{properties}] contains a list of properties about a currently open Lightweight Grid kernel.";

SetAttributes[RemoteServicesKernel, ReadProtected];

(*****************************************************)
RemoteServicesAgentInformation::usage = "RemoteServicesAgentInformation[\"agent\"] returns information about agent.
RemoteServicesAgentInformation[] returns information about agents on the local network.";

SetAttributes[RemoteServicesAgentInformation, ReadProtected];

RemoteServicesAgent::usage = "RemoteServicesAgent[properties] contains a list of properties for a Lightweight Grid Agent running on a networked computer."

SetAttributes[RemoteServicesAgent, ReadProtected];

(*****************************************************)
RemoteServiceInformation::usage = "RemoteServiceInformation[agent] returns information about the services provided by agent.
RemoteServiceInformation[] returns information about services provided by agents on the local network.";

SetAttributes[RemoteServiceInformation, {ReadProtected, Listable}];

RemoteServiceInformation::noservice = "No service named \"`2`\" is known for the agent `2`.";

RemoteService::usage = "RemoteService[{properties}] contains a list of properties about a service provided by a Lightweight Grid Manager.";

SetAttributes[RemoteService, ReadProtected];

(*****************************************************)

RemoteKernelServices::usage = "RemoteKernelServices[] returns a list of services provided by Lightweight Grid Managers";

SetAttributes[RemoteKernelServices, ReadProtected];

(*****************************************************)
RemoteServicesAgents::usage = "RemoteServicesAgents[] returns a list of URLs for Lightweight Grid Managers discovered on the local network.
RemoteServicesAgents[\"agent\"] returns a list of URLs for agents known to agent.";

SetAttributes[RemoteServicesAgents, ReadProtected];

RemoteServicesAgents::java = "A necessary component could not be found.";

(*****************************************************)

Options[RemoteKernelOpen] = {"LocalLinkMode" -> "Connect", "Service" -> "", 
	"Timeout" -> 30};

Options[RemoteKernelCloseAll] = { "Timeout" -> 30};

Options[RemoteKernelInformation] = { "Renew" -> False, "Timeout" -> 30 };

Options[RemoteKernelClose] = { "Timeout" -> 30 };

Options[RemoteServicesAgents] = { "Timeout" -> 30, "TemporaryPrinting" -> True,
	"CacheTimeout" -> 15*60. };

Options[RemoteKernelServices] = { "Timeout" -> 30, "CacheTimeout" -> 15*60. };

Options[RemoteServicesAgentInformation] = { "Timeout" -> 30, 
	"CacheTimeout" -> 15*60. };

Options[RemoteServiceInformation] = { "Timeout" -> 30, 
	"CacheTimeout" -> 15*60. };

Begin["`Private`"]

Needs["JLink`"];

getLogger[] := 
(
	InstallJava[];
	LoadJavaClass["com.wolfram.remoteservices.logging.LogbackFactory"];
	com`wolfram`remoteservices`logging`LogbackFactory`getLogger["LightweightGridClient`"]
)

log[DEBUG, msg_?JavaObjectQ] := (getLogger[])@debug[msg]

log[INFO, msg_?JavaObjectQ] := (getLogger[])@info[msg]

log[ERROR, msg_?JavaObjectQ] := (getLogger[])@error[msg]

log[level_, msg_] := log[level, MakeJavaObject[ToString[msg]]]

debug[msg_] := log[DEBUG, msg]

info[msg_] := log[INFO, msg]

error[msg_] := log[ERROR, msg]

JavaObjectNotNullQ[obj_] := MatchQ[obj, Except[Null, _?JavaObjectQ]];

(* $RemoteServicesClient is the singleton instance of our client class. *)
$RemoteServicesClient = Null;

$MaxConnectionsPerHost = 32;

$ClientFactory = "com.wolfram.remoteservices.client.RemoteServicesClientFactory"

createClient[] :=
	JavaBlock[
		Module[{loadresult, loader, clientresult, connmgr, params, httpclient,
				pm = PacletManager`Package`getPacletManager[]},

			info["Creating LightweightGridClient Java object"];
			InstallJava[];
			loadresult = LoadJavaClass[$ClientFactory];
			If[loadresult === $Failed,
				Return[Null]
			];
		
		(*
		    (* get JLink classloader *)
		  	obj = JavaNew[$ClientFactory];
		  	cls = obj@getClass[];
		  	loader = cls@getClassLoader[]; (* TODO make the classloader configurable *)
		*)
			(* This is the point where I considered using a custom classloader. *)
			LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
			loader = JLinkClassLoader`getInstance[];
		
			clientresult = 
				com`wolfram`remoteservices`client`RemoteServicesClientFactory`createClient[loader];
				
			If[clientresult === Null || !JavaObjectQ[clientresult],
				Return[$RemoteServicesClient]
			];

			$RemoteServicesClient = clientresult;
			KeepJavaObject[$RemoteServicesClient, Manual];
	
			With[{mathematicaVersion=LightweightGridClient`Information`$Version,
				javaVersion=Quiet[clientresult@getJarManifestInfo[]]},
	
				(* Record version if it matches, $Failed if it does not match *)
				`$RemoteServicesClientJavaVersion = 
					If[mathematicaVersion === javaVersion, javaVersion, $Failed]
			];
	
			If[JavaObjectNotNullQ[pm],
				httpclient = pm@getHttpClient[];
				If[JavaObjectNotNullQ[httpclient],
					$RemoteServicesClient@setHttpClient[httpclient];
					connmgr = httpclient@getHttpConnectionManager[];
					params = connmgr@getParams[];
					If[params@getDefaultMaxConnectionsPerHost[] < 
						$MaxConnectionsPerHost,
						params@setDefaultMaxConnectionsPerHost[
							$MaxConnectionsPerHost]
					];
					If[params@getMaxTotalConnections[] < $MaxConnectionsPerHost,
						params@setMaxTotalConnections[$MaxConnectionsPerHost]
					];
				]
			];
	
			(* Only if J/Link has a UI link, enable asynchronous DNS-SD change
			   notifications from Java back to Mathematica. *)
			If[hasFrontEndQ[],
				$RemoteServicesClient@setMathematicaDNSSDNotificationEnabled[True]
			];
	
			(* Call startServiceDiscovery[] to start DNS-SD browsing. *)
			$RemoteServicesClient@startServiceDiscovery[];
	
			(* return the client object *)
			$RemoteServicesClient
		]
	]

obtainClient[] := 
(
	Quiet[InstallJava[]]; (* Restart Java if it has shut down *)
	If[!validClientQ[$RemoteServicesClient],
		createClient[]
	];
	$RemoteServicesClient
)

validClientQ[client_] := 
	client =!= Null && JavaObjectQ[client] && !client@hasInitError[]

handleClientError[symbol_] := 
	With[{javaClientOK = $RemoteServicesClient =!= Null && 
			JavaObjectQ[$RemoteServicesClient],
		versionMatches = StringQ[`$RemoteServicesClientJavaVersion]},

		log[Which[
			!javaClientOK, "Failed to create Java object",
			!versionMatches, 
				"RemoteServices Java component's version "<>
				"does not match RemoteServices Mathematica package",
			(* Assert: Java client was created and version matches *)
			$RemoteServicesClient@hasInitError[],
				$RemoteServicesClient@getInitErrorDetails[]]];

		If[javaClientOK && !versionMatches,
			Message[symbol::lwgversion, symbol],
			Message[symbol::lwgclient, symbol]
		];
		$Failed
	]

rememberKernel[linkname_, info_RemoteServicesKernel] := 
	remoteKernelInfoMap[linkname] := info;

forgetKernel[linkname_] := 
	If[remoteKernelInfoMap[linkname] =!= 
		Unevaluated[remoteKernelInfoMap[linkname]],(* Avoid Unset::norep *)
		Unset[remoteKernelInfoMap[linkname]]
	]

forgetAllKernels[] := Clear[remoteKernelInfoMap];

lookupKernel[linkname_] := With[{info = remoteKernelInfoMap[linkname]},
	If[info === Unevaluated[remoteKernelInfoMap[linkname]],
		$Failed,
		info]
	]

createCalendar[date_] := 
	With[{cal = JavaNew["java.util.GregorianCalendar"]},
		cal@setTime[date];
		cal
	]

getMathematicaDateFromJavaDate[date_] := 
	Module[{cal}, 
		If[date === $Failed || date === Null,
			Return[Null]
		];
		cal = createCalendar[date];
		
		LoadJavaClass["java.util.Calendar"];

		{cal@get[java`util`Calendar`YEAR],
		 cal@get[java`util`Calendar`MONTH],
		 cal@get[java`util`Calendar`DAYUOFUMONTH], (* Note 'U'->'_' in JLink *)
		 cal@get[java`util`Calendar`HOURUOFUDAY],  
		 cal@get[java`util`Calendar`MINUTE],
		 cal@get[java`util`Calendar`SECOND]}
	]

(* Takes a LinkObject, such as returned by LinkLaunch, and returns the 
   linkname string, like "1021@myhost", which is passed to LinkConnect.
 *)
linknameFromLinkObject[linkobj:LinkObject[linkname_String,___]] := linkname

linknameFromLinkObject[_] := $Failed

replaceField[rec_, field_, newvalue_] := 
	ReplacePart[rec, field -> newvalue, Position[rec, field -> _ ]]

(* Creates the object returned by RemoteKernelInformation from the Java
  RemoteKernelInformation object and a value for the link field. *)
createRemoteKernelInformation[rki_?JavaObjectQ, link_, localLinkMode_] := 
	Module[{state = rki@getState[], launchdate, decorations},
			launchdate = rki@getLaunchDate[];
			decorations = rki@getDecorations[];
			RemoteServicesKernel[
		 	{"Link" -> link,
		 	 "Linkname" -> rki@getLinkname[],
		 	 "Agent" -> rki@getUrl[],
		 	 "Service" -> decorations@get[MakeJavaObject["Service"]],
		 	 "LocalLinkMode" -> localLinkMode,
		 	 "PID" -> rki@getPid[],
		 	 "Protocol" -> rki@getProtocol[],
		 	 "State" -> normalizeState[state@getText[]],
		 	 "Progress" -> state@getPercentDone[],
		 	 "Version" -> rki@getVersion[],
		 	 "VersionNumber" -> rki@getVersionNumber[],
		 	 "LaunchDate" -> getMathematicaDateFromJavaDate[launchdate]
		 	}]
	]

createRemoteKernelInformation[rki_, link_, localLinkMode_] := $Failed

normalizeState[""] := ""

normalizeState[state_?UpperCaseQ] := 
	StringTake[state, 1] <> ToLowerCase[StringDrop[state, 1]]

normalizeState["CONNECTED_BUSY"] := "ConnectedBusy"

normalizeState[state_] := state

(*****************************************************)
properties::usage = 
"properties[agent] returns the list of property rules 
for the given RemoteServicesAgent as returned by RemoteServicesAgentInformation.
properties[svc] returns the list of property rules 
for the given RemoteService as returned by RemoteServiceInformation.
properties[kernel] returns the list of property rules 
for the given RemoteServicesKernel as returned by RemoteKernelInformation.  

properties[agent, \"prop1\",...] returns the rules for the named properties 
for the given RemoteServicesAgent.
properties[svc, \"prop1\",...] returns the rules for the named properties 
for the given RemoteService.
properties[kernel, \"prop1\",...] returns the rules for the named properties
for the given RemoteServicesKernel.

agent@properties returns the list of property names for the given RemoteServicesAgent
as returned by RemoteServicesAgentInformation.
svc@properties returns the list of property names for the given RemoteService
as returned by RemoteServiceInformation.
kernel@properties returns the list of property names for the given RemoteServicesKernel
as returned by RemoteKernelInformation.

agent@\"prop\" returns the property named by prop, one of the names returned by
agent@properties, for the given RemoteServicesAgent.
svc@\"prop\" returns the property named by prop, one of the names returned by 
svc@properties, for the given RemoteService.
kernel@\"prop\" returns the property named by prop, one of the names returned by 
kernel@properties, for the given RemoteServicesKernel.
";

properties[RemoteServicesKernel[rules_List]] := rules;

properties[RemoteServicesKernel[rules:_List], props:_String..] := 
	{props} /. rules

k_RemoteServicesKernel[properties] := First[#] & /@ properties@k;

k_RemoteServicesKernel[fld_String] := fld /. properties@k;

properties[RemoteServicesAgent[rules:_List]] := rules

properties[RemoteServicesAgent[rules:_List], props:_String..] := 
	{props} /. rules

a_RemoteServicesAgent[properties] := First[#] & /@ properties@a;

a_RemoteServicesAgent[field_String] := field /. properties@a;

properties[RemoteService[rules:_List]] := rules

properties[RemoteService[rules:_List], props:_String..] := 
	{props} /. rules

svc_RemoteService[properties] := First[#] & /@ properties@svc;

svc_RemoteService[field_String] := field /. properties@svc;

(* Define a StandardForm for RemoteServicesKernel, like
   RemoteServicesKernel["1201@jfkleinwin,1202@jfkleinwin",General@http://jfkleinwin:3737/WolframLightweightGrid]
 *)
MakeBoxes[k:RemoteServicesKernel[rules:_List],StandardForm] :=
   With[{displayValue = 
   		StringJoin["RemoteServicesKernel[\"","Linkname" /. rules,"\"]"]},
   		InterpretationBox[displayValue,k]
   	]

(* A generic accessor for the object returned by createRemoteKernelInformation,
   which abstract the representation details of the object from the caller.
   getRKIField[rki,"fieldname"] returns the contents of fieldname.
*)
getRKIField[RemoteServicesKernel[rules_],field_String] := field /. rules

(* A generic setter for the object returned by createRemoteKernelInformation,
   which abstract the representation details of the object from the caller.
   updateRKIField[rki,"fieldname",newvalue] sets the value of fieldname.
*)
updateRKIField[RemoteServicesKernel[rules_],field_String,newValue_] := 
  RemoteServicesKernel[replaceField[rules, field, newValue]]

failureMessage::usage=
"failureMessage[result, agent, symbol, extraArgs] invokes Message and returns
$Failed, logging error details if needed.  The Message arguments are based on 
symbol, the result object and extraArgs.

If the Result object indicates CLIENT_ERROR or SERVER_ERROR, the message tag is
obtained from the Result object.  The Result object's MessageName symbol is 
always ignored and the symbol passed in to failureMessage is used.

If the Result object indicates an EXCEPTION, or if the Result object's message
is ill-formed, its errorCode, errorName and errorDetails are logged and generic
message is generated.

If the Result object is not a valid object a generic message is generated.";

failureMessage[result_?resultObjectQ, agent_, symbol_, extraArgs___] :=  
	failureMessage[result@getErrorCode[], result, agent, symbol, extraArgs]

failureMessage[ 2 | 3, (*com`wolfram`remoteservices`Result`CLIENTUERROR | 
	com`wolfram`remoteservices`Result`SERVERUERROR*) 
	result_?hasMessageQ, agent_, functionSymbol_, extraArgs___] := 
	With[{msgTag= StringSplit[result@getErrorName[], "::"][[2]],
		agentName = result@getAgentUrl[] /. {Null | "" -> agent}},
		With[{symbol=functionSymbol, tag=msgTag},
			Message[MessageName[symbol,tag], agentErrorTag[agentName], 
				result@getErrorDetails[]]];
		$Failed
	]

hasMessageQ[result_?resultObjectQ] := 
	StringMatchQ[result@getErrorName[], 
			(WordCharacter..) ~~ "::" ~~ (WordCharacter..)]

failureMessage[errorCode_Integer, result_?resultObjectQ, agent_, symbol_, 
	extraArgs___] :=
	(
		error[StringJoin[ToString[symbol],": Result",
			" errorCode: ", ToString[errorCode], 
			" errorName: ", ToString[result@getErrorName[]], 
			" errorDetails: ", ToString[result@getErrorDetails[]]]];
		failureMessage[symbol::lwgconnect, agent, extraArgs]
	)

failureMessage[InternalError[errorText_], agent_, symbol_, extraArgs___] := 
	(
		error["Note: "<>errorText];
		failureMessage[symbol::lwgconnect, agent, extraArgs]
	)

failureMessage[InternalError[errorText_], mn_MessageName, extraArgs___] := 
	(
		error["Note: "<>errorText];
		failureMessage[mn, None, extraArgs]
	)

failureMessage[result_, agent_, symbol_, extraArgs___] := 
	(
		error["Unrecognized result: "<>ToString[result, FormatType->InputForm]];
		failureMessage[symbol::lwgconnect, agent, extraArgs]
	)

failureMessage[mn_MessageName, args___] := 
	(Message[mn, args]; $Failed)

resultObjectQ[result_?JavaObjectQ /;
	InstanceOf[result, "com.wolfram.remoteservices.Result"]] := True

resultObjectQ[_] := False

agentErrorTag[agent_String] :=
	If[StringMatchQ[ToLowerCase@agent, "http" ~~ ___],
		Hyperlink[hostnameFromUrl@agent, agent],
		agent
	]

agentErrorTag[agents:{_String ..} /; Length[agents] < 3] := 
	Row[Riffle[agentErrorTag /@ agents, ", "]]

agentErrorTag[agents:{_String ..}] := 
	Row[
		Insert[Riffle[agentErrorTag /@ agents, ", "], 
			"and ", Length[agents]*2-1]
	]

(* infoFromLaunchedKernel
	Function to extract the RemoteKernelInformation from a given LaunchedKernel
	object.
	
	This function is shared by several kernel-opening functions and their 
	overloads.

   Returns: 
   Success[rki] on success where rki is the RemoteKernelInformation
   	Java object.
   or ClientError for failure to get a client object
   or InternalError[detail_String] for failures handled as an internal error
   or FailureResult[result] for failures handled by failureMessage
*)
infoFromLaunchedKernel[lk_] := 
	Module[{rki},
		If[lk === Null || lk === $Failed,
			Return[InternalError["LaunchedKernel is "<>ToString[lk]]]];
		If[lk@isFailed[],
			Return[FailureResult[lk]]];

		rki = lk@getKernelInfo[];
		If[rki==Null,
			Return[InternalError["RemoteKernelInformation is Null"]]];

		Success[rki]
	]

(* openKernel
   Function to open a kernel, but not connect.  
   It is shared by RemoteKernelOpenUnconnected and RemoteKernelOpen mostly so 
   these external functions can report messages and errors their own way.
   
   This function does not change the kernel information map, that is up to the
   caller.
   
   Returns: 
   Success[rki] on success where rki is the RemoteKernelInformation
   	Java object.
   or ClientError for failure to get a client object
   or InternalError[detail_String] for failures handled as an internal error
   or FailureResult[result] for failures handled by failureMessage
 *)
openKernel[agent_String, service_String, timeout_] := 
	Module[{client, lk},
		
		(* Get client object to contact agent *)
		client = obtainClient[];
		If[!validClientQ[client],
			Return[ClientError]];

		lk = client@launchKernel[agent, service, toMilliseconds[timeout]];
		infoFromLaunchedKernel[lk]
	]

toMilliseconds[timeSeconds_] := Round[timeSeconds * 1000]

"---  StartOfFunctions ---";

RemoteKernelOpenUnconnected::usage = 
"RemoteKernelOpenUnconnected[agent] launches a kernel, returning the linkname to use in 
connecting to it.

Agent can be hostname string, hostname:port string, URL, a RemoteServicesAgent expression 
returned by RemoteServicesAgentInformation, or a RemoteService expression
returned by RemoteServiceInformation.";

Options[RemoteKernelOpenUnconnected] = { "Service" -> "", "Timeout" -> 30}

RemoteKernelOpenUnconnected[agent_String, OptionsPattern[]] := 
	JavaBlock@
	Module[{service = OptionValue["Service"], timeout = OptionValue["Timeout"], 
		result, rki, linkname, rkinfo},
  		 
	  	If[!timeoutValidQ[timeout, RemoteKernelOpen],
	  		Return[$Failed]
	  	];

		result = openKernel[agent, service, timeout];

		result /.
		{
			Success[x_] :> (rki = x),
			ClientError :> 
				Return[handleClientError[RemoteKernelOpenUnconnected]],
			error_InternalError :>
				Return[failureMessage[error, agent, 
					RemoteKernelOpenUnconnected]],
			FailureResult[failresult_] :>
				Return[failureMessage[failresult, agent, 
					RemoteKernelOpenUnconnected]]
		};

		linkname = rki@getLinkname[];
		If[linkname==Null,
			Return[failureMessage[InternalError["linkname is Null"],
				agent, RemoteKernelOpenUnconnected]]
		];

		(* Construct the RemoteKernelInfo object *)
		rkinfo = createRemoteKernelInformation[rki, linkname, "Connect"];
		(* Enter it in the map. *)
		rememberKernel[linkname, rkinfo];

		linkname
	]

RemoteKernelOpenUnconnected[RemoteServicesAgent[rules:{___}], 
	opts:OptionsPattern[]] := 
	Module[{agent = OptionValue["ContactURL"]},
		RemoteKernelOpenUnconnected[agent, opts]
	]

RemoteKernelOpenUnconnected[RemoteService[rules:{___}],
	opts:OptionsPattern[]]:=
	Module[{service, agent = "Agent" /. rules},
		If[MatchQ[Hold[{opts}],Hold[{___, "Service" -> _,___}]],
			RemoteKernelOpenUnconnected[agent,opts],
		(* Else *)
			service = "Name" /. rules;
			RemoteKernelOpenUnconnected[agent, "Service" -> service, opts]
		]
	]

(*******************************************************)

RemoteKernelConnect::usage = 
"RemoteKernelConnect[linkname] connects to a kernel launched by RemoteServices and 
makes it available for monitoring and management.  RemoteKernelOpen[agent] is roughly 
equivalent to RemoteKernelConnect[RemoteKernelOpenUnconnected[agent]]";

RemoteKernelConnect::FailedRead = 
"Error reading `1` on remote kernel with link `2`.";

Options[RemoteKernelConnect] = { LinkProtocol -> "TCPIP" };

RemoteKernelConnect[linkname_String, opts___?OptionQ] :=
	Module[{linkobj, rkinfo, linkProtocolOptionValue, linkProtocolIsSpecified, 
		linkIsKnown, protocol},

		{linkProtocolOptionValue} = 
		{LinkProtocol} /. {opts} /. Options[RemoteKernelConnect];

		linkProtocolIsSpecified = MemberQ[{opts}, LinkProtocol -> _];

		rkinfo = remoteKernelInfoMap[linkname];

		linkIsKnown = rkinfo =!= Unevaluated[remoteKernelInfoMap[linkname]];

	  	(* Rules for deciding protocol:
	  		1. If a LinkProtocol option is actually specified, use that.
	  		2. If a LinkProtocol option is not specified, but the linkname is
	  			in our remoteKernelInfoMap because it was launched with
	  			RemoteKernelOpenUnconnected, use that protocol.
	  		3. Otherwise, use the default LinkProtocol value.
	  	*)
		protocol = If[linkProtocolIsSpecified, linkProtocolOptionValue,
			If[linkIsKnown, rkinfo@"Protocol", linkProtocolOptionValue]];

		linkobj = LinkConnect[linkname,LinkProtocol -> protocol];
		If[linkobj === $Failed,
			Return[$Failed]
		];

		RemoteKernelConnect[linkobj]
	]

RemoteKernelConnect[linkobj:LinkObject[linkname_, ___]] :=
	JavaBlock@
	Module[{rkinfo, linkIsKnown, connectedResult, contextIdExpr, contextId},

		rkinfo = remoteKernelInfoMap[linkname];

		linkIsKnown = rkinfo =!= Unevaluated[remoteKernelInfoMap[linkname]];

		(* Drain a possible input packet from the link *)
        LinkWriteHeld[linkobj, Hold[1]];
        If[(LinkRead[linkobj])[[0]] === InputNamePacket,
           LinkRead[linkobj]];

		(* Write fully qualified symbols on the link, so we can call code in the
			remote kernel, e.g. LightweightGridClient`Connected.  If we don't do this, 
			the remote kernel only sees a call to Connected (Global context).
		 *)
		MathLink`LinkSetPrintFullSymbols[linkobj, True];

	 	(* Notify remote manager of the connect attempt *)
	 	LinkWriteHeld[linkobj, 
	 		Hold[LightweightGridClient`Kernel`Connected[]]];

	 	(* Drain the return value *)
	 	connectedResult = LinkRead[linkobj];

		(* Check for evidence that the kernel wasn't launched by RemoteServices
		 *)
	 	If[connectedResult === 
	 		Unevaluated[ReturnPacket[LightweightGridClient`Kernel`Connected[]]],
	 		Message[RemoteKernelOpen::notwrsk, linkname];
	 		Return[$Failed]
	 	];

	 	(* TODO make the above Connected[] call synchronous *)

		(* If the info is known, update its link field, else create it. *)
	  	If[linkIsKnown,
		 	rkinfo = updateRKIField[rkinfo, "Link", linkobj],
		 (* else *)
		 	contextIdExpr = Hold[LightweightGridClient`Kernel`$ContextID];
		  	LinkWriteHeld[linkobj, contextIdExpr];
		  	contextId = LinkRead[linkobj];

		  	contextId /.
		  	{
		  		ReturnPacket[agentUrl_String] :>
		  			(
				  		(* get remote kernel info from url about linkname *)
				  		rkinfo = getRemoteKernelInfo[agentUrl, linkname, 
				  			"Connect"];

						rkinfo /.
						{
							RemoteServicesKernel[_] :> 
								(rkinfo = 
									updateRKIField[rkinfo, "Link", linkobj]),
							ClientError :> 
								Return[handleClientError[RemoteKernelConnect]],
							error_InternalError :> 
								Return[failureMessage[error, agentUrl, 
									RemoteKernelOpen]],
							FailureResult[x_] :>
								Return[failureMessage[x, agentUrl,
									RemoteKernelOpen]],
							response_ :> 
								Return[failureMessage[
									InternalError[
										"Unknown getRemoteKernelInfo response: "
										<>ToString[response]], agentUrl,
										 RemoteKernelOpen]]
						}
					),
				_ :> 
					(Message[RemoteKernelConnect::FailedRead, contextIdExpr,
						linkname];
					Return[$Failed])
			};
		];

		rememberKernel[linkname, rkinfo];

		linkobj
  ]

RemoteKernelOpen[agent_String, opts:OptionsPattern[]] := 
	JavaBlock@
	Module[{service, localLinkMode, timeout, result, rki, linkname, linkobj},

		service = OptionValue["Service"];
		localLinkMode = OptionValue["LocalLinkMode"];
		timeout = OptionValue["Timeout"];
	
		If[!StringQ[service],
			Message[RemoteKernelOpen::svcstring, agent, service];
			Return[$Failed]
		];
	  	If[!MemberQ[{"Create", "Connect"}, localLinkMode],
	  		Message[RemoteKernelOpen::locallinkmode, agent, localLinkMode];
	  		Return[$Failed]
	  	];
	  	If[!timeoutValidQ[timeout, RemoteKernelOpen],
	  		Return[$Failed]
	  	];
	
		If[localLinkMode === "Connect",
			result = openKernel[agent, service, timeout];
	
			result /.
			{
				Success[x_] :> (rki = x),
				ClientError :> Return[handleClientError[RemoteKernelOpen]],
				error_InternalError :> 
					Return[failureMessage[error, agent, RemoteKernelOpen]],
				FailureResult[result_] :>
					Return[failureMessage[result, agent, RemoteKernelOpen]]
			};
	
			linkname = rki@getLinkname[];
			If[linkname==Null,
				Return[failureMessage[InternalError["linkname is Null"], agent, 
					RemoteKernelOpen]]];
	
			linkobj = RemoteKernelConnect[linkname],
	
		(* Else *)
			(* Create/Listen case.  We create a link here and send it to the 
				remote kernel to connect back to us. *)
			Module[{client, lk, result, rkinfo},
				linkobj = createLink[];
				If[Head[linkobj] =!= LinkObject,
					Return[$Failed]];
				linkname = First@linkobj;
	
				client = obtainClient[];
				If[!validClientQ[client],
					Return[handleClientError[RemoteKernelOpen]]];
	
				(* Launch kernel, passing the linkname with which to connect back.*)
				lk = client@launchKernel[agent, service, linkname, toMilliseconds[timeout]];
				result = infoFromLaunchedKernel[lk];
				result /.
				{
					Success[x_] :> (rki = x),
					error_InternalError :> (
						LinkClose[linkobj];
						Return[failureMessage[error, agent, RemoteKernelOpen]]),
					FailureResult[result_] :> (
						LinkClose[linkobj];
						Return[failureMessage[result, agent, RemoteKernelOpen]])
				};
	
				rkinfo = createRemoteKernelInformation[rki, linkobj, 
					localLinkMode];
				rememberKernel[linkname, rkinfo];
	
				RemoteKernelConnect[linkobj]
			]
		];
		linkobj
	]

createLink[] := 
	With[{linkobj = LinkCreate[LinkProtocol -> "TCPIP"]},
		If[!MatchQ[linkobj, LinkObject[_String, ___]],
			Message[RemoteKernelOpen::linkcreatefailed, agent];
			$Failed
			,
			linkobj]
	]

RemoteKernelOpen[RemoteServicesAgent[rules:{___}],opts___?OptionQ] := 
 	RemoteKernelOpen["ContactURL" /. rules, opts]

RemoteKernelOpen[RemoteService[rules:{___}],opts___?OptionQ]:=
  Module[{service,agent},
	agent = "Agent" /. rules;
	(* If a service is specified in the options, use that *)
	If[MatchQ[Hold[{opts}],Hold[{___, "Service" -> _,___}]],
		RemoteKernelOpen[agent,opts],
	(* Else use the service named by the argument *)
		service = "Name" /. rules;
		RemoteKernelOpen[agent, "Service" -> service, opts]
	]
  ]

createOpenKernelArguments[agent_String, service_, $Failed] := Null

createOpenKernelArguments[agent_String, service_, LinkObject[linkname_,___]] :=
	createOpenKernelArguments[agent, service, linkname]

(* Note, caller is responsible to ensure InstallJava has been called *)
createOpenKernelArguments[agent_String, service_, link_] :=
	With[{obj = JavaNew["com.wolfram.remoteservices.client.OpenKernelArguments",
			agent, service]},
		If[StringQ[link],
			obj@setReverseLinkname[link]];
		obj
	]

createOpenKernelArguments[agents:{_String ..}, service_, links_] := 
(
	Quiet@InstallJava[];
	If[Length[agents] =!= Length[links],
		Message[createOpenKernelArguments::length];
		$Failed
		,
		(createOpenKernelArguments@@#)& /@ 
			Transpose[{agents, Table[service,{Length[agents]}], links}]
	]
)

(* openInParallel[args, timeout]
   Uses the client@launchKernels[args,timeout] call to launch a list of kernels
   in parallel.
   
   args is a list of OpenKernelArgument Java objects
   timeout is the number of seconds to wait for launches
     to complete before giving up on them
   
   Returns:
   Success[results] where linknames is a list, an element of which is either 
     a RemoteKernelInformation Java object or $Failed.
   InternalError[msg]
   FailureResult[lkr]
 *)
openInParallel[args_, timeout_] := 
	Module[{client = obtainClient[], lkr, lkarray, results},
		If[!validClientQ[client],
			Return[ClientError]
		];
	
		(* Launch 'em all! *)
		lkr = client@launchKernels[args, toMilliseconds[timeout]];
	
		If[lkr === Null || lkr === $Failed,
			Return[InternalError["LaunchedKernelResult is "<>ToString[lkr]]]];
		If[lkr@isFailed[],
			Return[FailureResult[lkr]]];
	
		(* Get the array of individual results *)
		lkarray = lkr@getLaunchedKernels[];
		If[!MatchQ[lkarray,{___}],
			Return[InternalError[
				"Launched kernels array is not in the expected form: "<>
				ToString[lkarray]]]];
	
		results = MapIndexed[
			Function[{lk,level},
				Module[{result, index, agent, rki, linkname},
					result = infoFromLaunchedKernel[lk];
	
					index = level[[1]];
					agent = args[[index]]@getAgent[];
	
					result /.
					{
						Success[x_] :> 
							(
								rki = x;
							 	linkname = rki@getLinkname[];
								 If[linkname === Null,
								 	failureMessage[
								 		InternalError[
								 			"RemoteKernelInformation linkname is null"],
								 		agent, RemoteKernelOpen],
								 	rki
								 ]
							),
						error_InternalError :> 
							Return[failureMessage[error, agent, RemoteKernelOpen]],
						FailureResult[failresult_] :> (
								If[failresult@getErrorName[] =!= 
									"RemoteKernelOpen::skipped",
									failureMessage[failresult, agent, 
										RemoteKernelOpen]];
								$Failed),
						_ :> Return[failureMessage[
								InternalError["Unhandled result "<>
									ToString[result]], agent, RemoteKernelOpen]]
					}
				]
			],
			lkarray];
	
	  	Success[results]
	]

RemoteKernelOpen[agents:{_String..}, opts:OptionsPattern[]] :=
	JavaBlock@
	Module[{service, timeout, localLinkMode, args, result, results, links},
		service = OptionValue["Service"];
		localLinkMode = OptionValue["LocalLinkMode"];
		timeout = OptionValue["Timeout"];

		If[!StringQ[service],
			Message[RemoteKernelOpen::svcstring, 
				If[Length[agents]===1, First@agents, "multiple agents"], 
				service];
			Return[$Failed]];
	  	If[!MemberQ[{"Create", "Connect"}, localLinkMode],
	  		Message[RemoteKernelOpen::locallinkmode, 
	  			If[Length[agents]===1, First@agents, "multiple agents"], 
	  			localLinkMode];
	  		Return[$Failed]];
	  	If[!timeoutValidQ[timeout, RemoteKernelOpen],
	  		Return[$Failed]];

		links = Table[If[localLinkMode === "Create", createLink[], Null],
			{Length[agents]}];

		(* Create OpenKernelArguments objects for each agent and link *)
		args = createOpenKernelArguments[agents, service, links];

		result = openInParallel[args, timeout];
		If[Head[result] === Success,
			results = First[result]
			,
			If[Head[#]===LinkObject, LinkClose[#], Null]& /@ links;
			Return[result /. {
				ClientError :> handleClientError[RemoteKernelOpen],
				error_InternalError :> 
					failureMessage[error, agents, RemoteKernelOpen],
				FailureResult[result_] :> 
					failureMessage[result, agents, RemoteKernelOpen],
				_ :> 
					failureMessage[InternalError[
						"Unexpected value from openInParallel: "<>
						ToString[result]], agents, RemoteKernelOpen]
			}]
		];

		finishConnectionOfParallelKernels[localLinkMode, results, links]
	]

finishConnectionOfParallelKernels["Connect", results_List, ___] := 
	Map[If[JavaObjectNotNullQ[#],
		RemoteKernelConnect[#@getLinkname[]],
		$Failed]&, 
		results
	]

finishConnectionOfParallelKernels["Create", results_List, links_List] := 
  	Map[finishConnectionOfParallelKernel, Transpose[{links, results}]]

finishConnectionOfParallelKernel[{link:LinkObject[linkname_,___], 
	rki_?JavaObjectNotNullQ}] := 
(
	rememberKernel[linkname,
  		createRemoteKernelInformation[rki, link, "Create"]];
	RemoteKernelConnect[link];
  	link);
finishConnectionOfParallelKernel[{link_, _}] := (
	If[Head[link]===LinkObject, LinkClose[link]];
	$Failed
)

RemoteKernelOpen[agents:{RemoteServicesAgent[_]..},opts___?OptionQ] := 
  Module[{agentUrls},
  	agentUrls = ("ContactURL" /. properties[#])& /@ agents;
  	RemoteKernelOpen[agentUrls, opts]
  ]

RemoteKernelOpen[services:{RemoteService[_]..}, opts___?OptionQ]:=
	JavaBlock@
	Module[{args, result, results, timeout, localLinkMode, links, 
		agents},

	  	{timeout, localLinkMode} = 
	  	{"Timeout", "LocalLinkMode"} /. {opts} /. Options[RemoteKernelOpen];
	
	  	If[!timeoutValidQ[timeout, RemoteKernelOpen],
	  		Return[$Failed]
	  	];
	  	If[!MemberQ[{"Create", "Connect"}, localLinkMode],
	  		Message[RemoteKernelOpen::locallinkmode, 
	  			If[Length[services]===1, "Agent" /. First@services, 
	  				"multiple agents"], 
	  			localLinkMode];
	  		Return[$Failed]
	  	];
	
		links = Table[If[localLinkMode === "Create", createLink[], Null],
			{Length[services]}];
	
		args = createOpenKernelArguments[("Agent" /. First[#])& /@ services, 
			("Name" /. First[#])& /@ services, links];
	
		result = openInParallel[args, timeout];
			If[Head[result] === Success,
				results = First[result]
				,
				If[Head[#]===LinkObject, LinkClose[#], Null]& /@ links;
				agents = ("Agent" /. First[#])& /@ services;
				Return[result /. {
					ClientError :> handleClientError[RemoteKernelOpen],
					error_InternalError :> 
						failureMessage[error, agents, RemoteKernelOpen],
					FailureResult[result_] :> 
						failureMessage[result, agents, RemoteKernelOpen],
					_ :> 
						failureMessage[InternalError[
							"Unexpected value from openInParallel: "<>
							ToString[result]], agents, RemoteKernelOpen]
				}]
			];
	
		finishConnectionOfParallelKernels[localLinkMode, results, links]
	]

RemoteKernelInformation[linkobj:LinkObject[linkname_String,___],
	opts___?OptionQ] :=
	RemoteKernelInformation[linkname,opts]

RemoteKernelInformation[linkname_String,opts___?OptionQ] := 
	JavaBlock@
	Module[{renew, timeout, prevInfo, url, prevLink, newInfo, info},
		{renew, timeout} = {"Renew", "Timeout"} /. {opts} /. 
			Options[RemoteKernelInformation];

	  	prevInfo = remoteKernelInfoMap[linkname];

		(* Link not found is fatal -- we have nothing to return *)
	  	If[prevInfo === Unevaluated[remoteKernelInfoMap[linkname]],
	  		Message[RemoteKernelInformation::nolink,linkname];
	  		Return[$Failed]
	  	];
	
		info = prevInfo; (* default return value *)
	
	  	If[renew,
		  	If[!timeoutValidQ[timeout, RemoteKernelInformation],
		  		Return[$Failed]
		  	];
	
			url = prevInfo@"Agent";
			newInfo = getRemoteKernelInfo[url, linkname, 
				prevInfo@"LocalLinkMode", timeout];

			newInfo /.
			{
				RemoteServicesKernel[_] :> 
					(prevLink = prevInfo@"Link";
					info = updateRKIField[newInfo, "Link", prevLink];
					remoteKernelInfoMap[linkname] := info),
				ClientError :> handleClientError[RemoteKernelInformation],
				error_InternalError :> 
					failureMessage[error, url, RemoteKernelInformation],
				FailureResult[result_] :>
					(
						(* If the failure was not a timeout, forget this link *)
						If[!(JavaObjectQ[result] && result =!= Null &&
							StringQ[result@getErrorName[]] &&
							StringMatchQ[result@getErrorName[], 
								___ ~~ "::timeout"]),
							forgetKernel[linkname]; (* TODO reconfirm first *)
						];
					 	failureMessage[result, url, RemoteKernelInformation]),
				_ :> failureMessage[
					InternalError["Unknown getRemoteKernelInfo response: "<>
						ToString[newInfo]], url, RemoteKernelInformation]
			};
	
	  	];
	
	  	info
	]

RemoteKernelInformation[opts___?OptionQ] := 
	Cases[RemoteKernelInformation[#, opts]& /@ RemoteServicesLinks[], 
		_RemoteServicesKernel];

(* Queries the agent for the remote kernel information of a given kernel.
   Return value: list of kernel properties on success,
   	  ClientError on failures handled with handleClientError,
   	  InternalError[message arguments] on failures requiring
   	  an InternalError message,
   	  FailureResult[resultobj] on failures requiring a call to
   	  failureMessage.
 *)
getRemoteKernelInfo[url_String, linkname_String, localLinkMode_, timeout_:5] :=
	Module[{client, kr, rki, info},
		client = obtainClient[];

		If[!validClientQ[client],
		 	Return[ClientError]];

		kr = client@getKernelInfo[url, linkname, toMilliseconds[timeout]];
		If[kr == Null,
			Return[InternalError["kr is Null"]]];

		If[kr@isFailed[],
			Return[FailureResult[kr]]];

		rki = kr@getKernelInfo[];
		If[rki == Null,
			Return[InternalError["rki is Null"]]];

		info = createRemoteKernelInformation[rki, linkname, localLinkMode];
		info
	]

RemoteServicesLinks[] := 
	(* Get the info record for each open kernel *)
	With[{infos = #[[2]]& /@ DownValues[remoteKernelInfoMap]},
		(* Now pick out the link element of each *)
		(* TODO add a Connected -> True option to pick out only LinkObjects *)
		(* TODO cull out down or disconnected kernels *)
		Map[#@"Link"&, infos]
	]

SetAttributes[RemoteKernelClose,Listable];
RemoteKernelClose[linkobj:LinkObject[__], opts___?OptionQ] := 
	Module[{linkname},
		linkname = linknameFromLinkObject[linkobj];
		If[linkname === $Failed,
			Message[RemoteKernelClose::nokernel, linkobj];
			$Failed,
		(* Else *)
			RemoteKernelClose[linkname, opts]
		]
	]

RemoteKernelClose[rk_RemoteServicesKernel] := RemoteKernelClose[rk@"Link"]

(* Given a KilledKernel Java object, clears its entry in the info map and
  returns the Mathematica representation of its contents.
 *)
handleClosedKernel[kk_] :=
	Module[{linkname,info},
		linkname = kk@getLinkname[];
		info = remoteKernelInfoMap[linkname];
		If[info === Unevaluated[remoteKernelInfoMap[linkname]],
	  		Message[RemoteKernelClose::nokernel, linkname];
	  		$Failed,
		(* Else *)
			info /. {
				RemoteServicesKernel[{___, "Link" -> linkobj_, ___}] :>
					LinkClose[linkobj] };
			forgetKernel[linkname];
		 	ClosedKernel[linkname]]
	]

RemoteKernelClose[linkname_String, opts___?OptionQ] := 
	JavaBlock@
	Module[{timeout, client,info,servletUrl,kk},
		{timeout} = 
		{"Timeout"} /. {opts} /. Options[RemoteKernelClose];

	  	If[!timeoutValidQ[timeout, RemoteKernelClose],
	  		Return[$Failed]];

		(* Get client object to contact agent *)
		client = obtainClient[];
		 If[!validClientQ[client],
		 	Return[handleClientError[RemoteKernelClose]]];

		(* With only the linkname given us, we can only close the kernel by
			looking up the corresponding agent in our map.
		*)
		info = remoteKernelInfoMap[linkname];
	  	If[info === Unevaluated[remoteKernelInfoMap[linkname]],
	  		Message[RemoteKernelClose::nokernel,linkname];
	  		Return[$Failed]];

		servletUrl = getRKIField[info,"Agent"];
		If[!MatchQ[servletUrl, _String],
			Return[failureMessage[
				InternalError["Unknown agent: "<>ToString[servletUrl], 
					"Unknown", RemoteKernelClose]]]];

		kk = client@killKernel[servletUrl, linkname, toMilliseconds[timeout]];
		If[kk == Null,
			Return[failureMessage[InternalError["Kernel close result is Null"],
				servletUrl]]];

		(* If kernel is closed by someone else, kk will show a failure, 
			but we should remove the kernel from our list.  
			TODO If kk is a failure, ask the agent if it knows about this
			kernel, to distinguish between "kernel call failed because
			we couldn't reach the agent" from "call failed because the kernel
			is already closed.
		*)
		If[kk@isSuccess[],
			handleClosedKernel[kk],
		(* Else *)
			forgetKernel[linkname]; (* TODO think about asking the agent to confirm the kernel doesn't exist *)
			failureMessage[kk, servletUrl, RemoteKernelClose]]
	]

RemoteKernelCloseAll[opts:OptionsPattern[]] := 
	Module[{timeout = OptionValue["Timeout"], retval},
	  	If[!timeoutValidQ[timeout, RemoteKernelCloseAll],
	  		Return[$Failed]];

		retval = Map[RemoteKernelClose[#, opts]&, RemoteServicesLinks[]];
		forgetAllKernels[];
		retval
	]

normalizeToList::usage = "normalizeToList[x] returns its argument if it is a 
list, and returns an empty list otherwise.  It is used to ensure a value becomes
a list, and can be used to automatically treat error-indicating values such as 
Null as a list of nothing to do.";

normalizeToList[l_List] := l

normalizeToList[_] := {}

(*****************************************************)

RemoteServicesLog::usage = 
"RemoteServicesLog[] returns the location of the logfile used by this package.";

RemoteServicesLog[] := 
  With[{client = obtainClient[]},
		 If[validClientQ[client],
			client@getLogfilePath[]
			,
		 	handleClientError[RemoteServicesLog]
		 ]
  ]

(* Define a StandardForm for RemoteServicesAgent, like
   RemoteServicesAgent["jfkleinwin"]
 *)
MakeBoxes[k:RemoteServicesAgent[rules:_List], StandardForm] :=
	With[{displayValue = 
		StringJoin["RemoteServicesAgent[\"", AgentShortName[k],"\"]"]}, 
		InterpretationBox[displayValue, k]
	]

SetAttributes[RemoteServicesAgentInformation,Listable];
RemoteServicesAgentInformation[agent_String, opts:OptionsPattern[]] := 
	JavaBlock@
	Module[{timeout = OptionValue["Timeout"], 
		cacheTolerance = OptionValue["CacheTimeout"], client, spr, props, 
		config, retval},

	  	If[!timeoutValidQ[timeout, RemoteServicesAgentInformation],
	  		Return[$Failed]
	  	];

		(* Get client object to contact agent *)
		client = obtainClient[];
		If[!validClientQ[client],
			Return[handleClientError[RemoteServicesAgentInformation]]
		];

		(* TODO normalize the agent spec to a full URL first - do this using the client object *)
		If[cacheTolerance > 0 && ValueQ[$AgentInformationCache[agent]],
			Block[{now = AbsoluteTime[], cachedValue, cacheTime, tdiff},
				{cacheTime, cachedValue} = $AgentInformationCache[agent];
				tdiff = now - cacheTime;
				If[tdiff <= cacheTolerance,
					Return[cachedValue]]]
		];

		If[agent === "",
			(* Construct a dummy value *)
			props = JavaNew["com.wolfram.remoteservices.ServerProperties"];
			props@setContactUrl["http://host.example.com:3737"];
			config = JavaNew["com.wolfram.remoteservices.ManagerConfiguration"];
			,
		(* Else *)
			spr = client@getServerProperties[agent, toMilliseconds[timeout]];
			If[spr == Null || !JavaObjectQ[spr],
				Return[failureMessage[InternalError["spr is "<>ToString[spr]],
					agent, RemoteServicesAgents]]];

			If[spr@isFailed[],
				Return[failureMessage[spr, agent,
					RemoteServicesAgentInformation]]];

			props = spr@getServerProperties[];
		];

		retval = RemoteServicesAgent[props@getAsExpr[]]; 

		$AgentInformationCache[agent] = {AbsoluteTime[], retval};

		retval
	]

RemoteServicesAgentInformation[opts___?OptionQ] := 
	Cases[RemoteServicesAgentInformation[#,opts]& /@ 
		RemoteServicesAgents["TemporaryPrinting" -> False],
		_RemoteServicesAgent]

(* end RemoteServicesAgentInformation *)


CPUHistory::usage = "CPUHistory[agent] returns a list of CPU usage data points.";

CPUBusyQ::usage = 
"CPUBusyQ[agent] returns True if the CPU appears to be occupied more than 30% in recent 
samples according to the CpuHistory property of the given agent, False otherwise.

The agent can be an agent specification string such as a URL or hostname, or it
can be a RemoteServicesAgent object as returned by RemoteServicesAgentInformation.";

CPUIdleQ::usage = 
"CPUIdleQ[agent] returns True if the CPU appears to be occupied less than 30% in recent 
samples according to the CpuHistory property of the given agent, False otherwise.

The agent can be an agent specification string such as a URL or hostname, or it
can be a RemoteServicesAgent object as returned by RemoteServicesAgentInformation.";

Options[CPUIdleQ] = Options[CPUBusyQ] = { "NumberOfSamples" -> 10, 
	"IdleThreshold" -> 30.0, "TrimFraction" -> 0.05, "CacheTimeout" -> 2*60.,
	"Timeout" -> 30};

CPUPlot::usage = 
"CPUPlot[agent] performs a ListPlot on the CpuHistory property of the given agent.

The agent can be an agent specification string such as a URL or hostname, or it
can be a RemoteServicesAgent object as returned by RemoteServicesAgentInformation.";

"IdleThreshold is an option to CPUBusyQ and CPUIdleQ
that specifies the highest fraction of busy CPU time to consider idle.";

"NumberOfSamples is an option to CPUBusyQ and CPUIdleQ
that specifies how many samples to examine.";

"TrimFraction is an option to CPUBusyQ and CPUIdleQ 
that specifies what fraction of largest and smallest data elements to drop from
consideration.";

CPUBusyQ[data:{{_,_}..}, opts___?OptionQ] := Module[
	{numberOfSamples, cpuUsageThreshold, trimFraction, recentData, pctBusyData},

	{numberOfSamples, cpuUsageThreshold, trimFraction} = 
	{"NumberOfSamples", "IdleThreshold", "TrimFraction"}
	/. {opts} /. Options[CPUBusyQ];
	
	If[numberOfSamples === All,
		numberOfSamples = Length[data]];

	recentData = Take[data, -Min[numberOfSamples, Length[data]]];
	pctBusyData = #[[2]]& /@ recentData;
	N[TrimmedMean[pctBusyData, trimFraction]] > cpuUsageThreshold 
	(* trimFraction needed when in 5.2 *)
]

CPUBusyQ[RemoteServicesAgent[rules_], opts___?OptionQ] := 
	CPUBusyQ["CpuHistory" /. rules, opts]

CPUBusyQ[agent_String, opts___?OptionQ] := 
	With[{result=RemoteServicesAgentInformation[agent]},
		If[MatchQ[result, _RemoteServicesAgent],
			CPUBusyQ[result, opts],
			$Failed]
	]

CPUIdleQ[data:{{_,_}..}, opts___?OptionQ] := !CPUBusyQ[data, opts]

CPUIdleQ[agent_RemoteServicesAgent, opts___?OptionQ] := !CPUBusyQ[agent, opts]

CPUIdleQ[agent_String, opts___?OptionQ] := !CPUBusyQ[agent, opts]

CPUPlotOptions = With[{fgcolor=Green},
	{Joined->True, BaseStyle->fgcolor, PlotStyle->fgcolor}]

CPUPlot[agent_String, cpuHistory_List, opts___] := 
	ListPlot[cpuHistory, opts, PlotRange -> {0, 100}, 
		Background -> Black, 
		PlotLabel -> hostnameFromUrl["ContactURL" /. rules]<>" CPU Usage", 
		Sequence@@CPUPlotOptions]

CPUPlot[agent_String, opts___] := CPUPlot[agent, CPUHistory[agent], opts]

AgentShortName::usage =
"AgentShortName[url] returns a shorter name for the given agent URL, 
generally its hostname.  This is useful for user interface purposes where a full
URL may be too long to display.";

AgentShortName[agent_String] := 
	hostnameFromUrl[agent] (* TODO In future we can look up the agent in the client's directory and handle more than URLs *)

AgentShortName[agent_RemoteServicesAgent] := 
	AgentShortName[agent@"ContactURL"]

hostnameFromUrl[url_] :=
	Module[{hostportpath, hostport, host},
		hostportpath = StringReplace[url, 
			StartOfString ~~ ("http://" | "https://") ~~ x__ -> x];
		hostport = StringReplace[hostportpath, 
			StartOfString ~~ x__ ~~ "/" ~~ ___ -> x];
		host = StringReplace[
			hostport, {StartOfString ~~ "[" ~~ x__ ~~ "]" ~~ ___ -> x, 
			StartOfString ~~ x__ ~~ ":" ~~ __ -> x, x_ -> x}];
		If[StringMatchQ[host, StartOfString ~~ LetterCharacter ~~ ___],
			First@StringSplit[host, "."],
			host
		]
	]

(* Define a StandardForm for RemoteService, like
   RemoteService["http://jfkleinwin:3737/WolframLightweightGrid"]
 *)
MakeBoxes[k:RemoteService[rules:_List],StandardForm] :=
	With[{displayValue = StringJoin["RemoteService[\"", "Name" /. rules, " on ",
		AgentShortName["Agent" /. rules],"\"]"]},
		InterpretationBox[displayValue, k]
	]

RemoteServiceInformation[agent_String, OptionsPattern[]] :=
	JavaBlock@Module[{
		timeout = OptionValue["Timeout"], 
		cacheTolerance = OptionValue["CacheTimeout"], 
		client, svcPropMapResult, map, entrySetObject, 
		entrySetIterator, acc, entry, entries, svcname, svcprop, cfg, svcinfo, 
		isEnabled, agentUrl, kernelNumber, numKernelsRunning, 
		numKernelsAvailable},

	  	If[!timeoutValidQ[timeout, RemoteServiceInformation],
	  		Return[$Failed]
	  	];

		(* TODO normalize the agent spec to a full URL first - do this using the client object *)
		If[cacheTolerance > 0 && ValueQ[$RemoteServiceInformationCache[agent]],
			Block[{now = AbsoluteTime[], cachedValue, cacheTime, tdiff},
				{cacheTime, cachedValue} = 
					$RemoteServiceInformationCache[agent];
				tdiff = now - cacheTime;
				If[tdiff <= cacheTolerance,
					Return[cachedValue]
				]
			]
		];

		(* Get client object to contact agent *)
		client = obtainClient[];
		If[!validClientQ[client],
			Return[handleClientError[RemoteServiceInformation]]
		];

		svcPropMapResult = client@getServicesWithProperties[agent, 
			toMilliseconds[timeout]];
		If[svcPropMapResult == Null || !JavaObjectQ[svcPropMapResult],
			Return[failureMessage[InternalError["svcPropMapResult is "<>
				ToString[svcPropMapResult]], agent, RemoteServiceInformation]]
		];

		If[svcPropMapResult@isFailed[],
			Return[failureMessage[svcPropMapResult, agent,
				RemoteServiceInformation]]
		];

		map = svcPropMapResult@getServicePropertiesMap[];
		If[map == Null || !JavaObjectQ[map],
			Return[failureMessage[InternalError["map is "<>
				ToString[map]], agent, RemoteServiceInformation]]
		];

		agentUrl = svcPropMapResult@getAgentUrl[];

		(* Convert the map to rules *)
		(* TODO Use the iterator to generate a Mathematica list of objects, then operate on the list *)
		acc = {};
		entrySetObject = map@entrySet[];
		entries = IteratorList[entrySetObject@iterator[]];
		entrySetIterator = entrySetObject@iterator[];
		While[entrySetIterator@hasNext[],
			entry = entrySetIterator@next[];
			svcname = entry@getKey[]; (* unused currently *)
			svcprop = entry@getValue[];
			cfg = svcprop@getConfig[];
			isEnabled = cfg@getEnabled[];

			kernelNumber = cfg@getKernelNumber[];
			numKernelsRunning = svcprop@getNumKernels[];
			numKernelsAvailable = 
				If[kernelNumber === 0,
					Infinity,
					kernelNumber - numKernelsRunning
				];

			svcinfo = 
			RemoteService[{
				"Name" -> cfg@getName[],
				"Agent" -> agentUrl,
				"Enabled" -> isEnabled,
				"KernelCommand" -> cfg@getKernelCommand[],
				"LicenseState" -> cfg@getLicenseState[],
				"VersionNumber" -> cfg@getVersionNumber[],
				"KernelNumber" -> kernelNumber,
				"KernelsAvailable" -> numKernelsAvailable,
				"KernelTimeout" -> cfg@getKernelTimeout[],
				"KernelInactiveTimeout" -> cfg@getKernelInactiveTimeout[],
				"KernelInitialization" -> cfg@getKernelInitialization[]
			}];
			If[isEnabled || showDisabled,
				acc = Append[acc,svcinfo];
			];
		];

		$RemoteServiceInformationCache[agent] = {AbsoluteTime[], acc};

		acc
	]

RemoteServiceInformation[opts___?OptionQ] :=
	Flatten[
		RemoteServiceInformation[
			RemoteServicesAgents["TemporaryPrinting"->False],
			opts],
		1
	]

AgentHyperlink::usage = "AgentHyperlink[url] or AgentHyperlink[agent] returns a hyperlink which displays as 
AgentShortName[agent] but links to the URL of the given agent.";

AgentHyperlink[url_] := Hyperlink[hostnameFromUrl[url], url]

AgentHyperlink[agent_RemoteServicesAgent] := 
	AgentHyperlink[agent@"ContactURL"]

RemoteKernelServices[opts___?OptionQ] :=
	Module[{timeout},
		{timeout} = {"Timeout"} /. {opts} /. Options[RemoteKernelServices];

	  	If[!timeoutValidQ[timeout, RemoteKernelServices],
	  		Return[$Failed]];

		Map[{"Agent" -> ("Agent" /. First[#]), "Service" -> ("Name" /. First[#])}&,
			Cases[RemoteServiceInformation[opts], _RemoteService]]
	]

$RemoteServicesAgents::usage = "$RemoteServicesAgents is the same list returned by RemoteServicesAgents[\"Timeout\"->0].";

$RemoteServicesAgents = {};

RemoteServicesAgents[opts___?OptionQ] :=
	Module[{timeout, tempPrinting, result},
	  	 {timeout, tempPrinting} = 
	  	 {"Timeout", "TemporaryPrinting"}
	  	 /. {opts} /. Options[RemoteServicesAgents];

	  	If[!timeoutValidQ[timeout, RemoteServicesAgents],
	  		Return[$Failed]];

		If[tempPrinting && hasFrontEndQ[],
			PrintTemporary[
				"Looking for computers running the Lightweight Grid Manager..."]];

		monitor[
			result = getAgentList[timeout];

			result /. 
			{
				ClientError :> Return[handleClientError[RemoteServicesAgents]],
				err_InternalError :> 
					Return[failureMessage[err, RemoteServicesAgents::java]],
				FailureResult[dslr_] :> Return[failureMessage[dslr,
					RemoteServicesAgents]],
				Success[urls_] :> updateAgentList[urls]
			},
			$RemoteServicesAgents, tempPrinting];

		$RemoteServicesAgents
	]

(*
StartBrowsing[] := 
	JavaBlock@Module[{client},
		client = obtainClient[];
		If[!validClientQ[client],
			Return[handleClientError[RemoteServicesAgents]]];
		client@startServiceDiscovery[]];
*)

hasFrontEndQ[] := Head[System`$FrontEnd] === System`FrontEndObject

SetAttributes[monitor, HoldAll];
monitor[expr_, mon_, pred_:True] := 
	If[pred && hasFrontEndQ[],
		Monitor[expr, mon],
	(* Else *)
		expr
	]

(* A wrapper to support the URL symbol *)
RemoteServicesAgents[URL[url_], opts___?OptionQ] := RemoteServicesAgents[url, opts]

RemoteServicesAgents[agent_String, opts___?OptionQ] :=
	JavaBlock@Module[{timeout, cacheTolerance, client, dslr, svcs, agenturls},
		{timeout, cacheTolerance} = 
		{"Timeout", "CacheTimeout"} /. {opts} /. Options[RemoteServicesAgents];

	  	If[!timeoutValidQ[timeout, RemoteKernelAgents],
	  		Return[$Failed]
	  	];

		client = obtainClient[];
		If[!validClientQ[client],
			Return[handleClientError[RemoteServicesAgents]]
		];

		(* TODO normalize the agent spec to a full URL first - do this using the client object *)
		If[cacheTolerance > 0 && 
			MatchQ[$AgentsCache[agent], {_, _List | $Failed}],
			Block[{now = AbsoluteTime[], cachedValue, cacheTime, tdiff},
				{cacheTime, cachedValue} = $AgentsCache[agent];
				tdiff = now - cacheTime;
				If[tdiff <= cacheTolerance,
					Return[cachedValue]]]];

		$AgentsCache[agent] = {AbsoluteTime[], "Pending"};

		dslr = client@getDiscoveredServices[agent, toMilliseconds[timeout]];
		If[!validResultQ[dslr],
			NetworkStartingPointNotResponding[agent];
			Return[failureMessage[InternalError["dslr is "<>ToString[dslr]], 
				agentErrorTag@agent, RemoteServicesAgents]]];

		If[dslr@isFailed[],
			NetworkStartingPointNotResponding[agent];
			Return[failureMessage[dslr, agentErrorTag@agent,
				RemoteServicesAgents]]];

		svcs = dslr@getDiscoveredService[];

		If[svcs == Null,
			NetworkStartingPointNotResponding[agent];
			Return[failureMessage[InternalError["svcs is "<>ToString[svcs]],
				agentErrorTag@agent, RemoteServicesAgents]]];

		agenturls = Cases[#@getUrl[]& /@ svcs, _String];
		$AgentsCache[agent] = {AbsoluteTime[], agenturls};
		agenturls
	]

NetworkStartingPointNotResponding[agent_String] := 
(
	$AgentsCache[agent] = {AbsoluteTime[], $Failed};
)

AddNetworkStartingPoint[agent_String] := 
	Module[{client = obtainClient[]},
		If[!validClientQ[client],
			Return[$Failed]
		];
		client@addNetworkStartingPoint[agent]
	]

RemoveNetworkStartingPoint[agent_String] := 
	Module[{client = obtainClient[]},
		If[!validClientQ[client],
			Return[$Failed]
		];
		client@removeNetworkStartingPoint[agent]
	]

SetNetworkStartingPointUpdateInterval[interval_] := 
	Module[{client = obtainClient[]},
		If[!validClientQ[client],
			Return[$Failed]
		];
		client@setNetworkStartingPointMonitorUpdateInterval[toMilliseconds[interval]]
	]

(* UpdateAgents is called from Java asynchronously when a network starting point
  detects new agents known by the remote agent.
 *)
UpdateAgents[agent_String, urls_List] := 
(
	$AgentsCache[agent] = {AbsoluteTime[], urls};
)

(* ServiceChange is called from Java asynchronously when a service is added or 
   removed. We could in principal have a Mathematica listener interface and
   invoke listeners here when events happen, but Dynamics let you do that. *)
ServiceChange[change_String, svcname_String, url_String, urls_List] := 
 	updateAgentList[urls]

getAgentList[timeout_] := 
	JavaBlock@
	Module[{client, dslr, svcs, agenturls},
	
		client = obtainClient[];
		If[!validClientQ[client],
			Return[ClientError]
		];
	
		dslr = client@getDiscoveredServices[toMilliseconds[timeout]];
		If[dslr === Null || dslr === $Failed,
			Return[InternalError["dslr is "<>ToString[dslr]]]
		];
		If[dslr@isFailed[],
			Return[FailureResult[dslr]]
		];
	
		svcs = dslr@getDiscoveredService[];
	
		If[svcs == Null,
			Return[InternalError["dslr is "<>ToString[dslr]]]
		];
	
		agenturls = Cases[Map[Function[svc,svc@getUrl[]], svcs],_String];
		Success[agenturls]
	]

updateAgentList[urls_List] := $RemoteServicesAgents = urls

LinksForAgent::usage =
"LinksForAgent[url] returns the members of RemoteServicesLinks[] launched from the given agent.";

SetAttributes[LinksForAgent, Listable];
LinksForAgent[agent_] := Select[RemoteServicesLinks[], 
	agent === (RemoteKernelInformation[#]@"Agent") &];

(******************************************************************************)
(* Useful abstractions *)
validResultQ[Null] := False

validResultQ[expr_?JavaObjectQ] := True

validResultQ[_] := False

normalizeLicenseState["invalid:secured"] := "Invalid"

normalizeLicenseState["invalid:unsecured"] := "Invalid"

normalizeLicenseState[state_String] := ToUpperCase[state]

(******************************************************************************)
(* Option checking *)
timeoutValidQ[t_, sym_] := 
	With[{ok=TrueQ[timeoutValidQ[t]]}, 
		If[!ok, Message[sym::iopnm, "Timeout", t]];
		ok
	]

timeoutValidQ[Infinity] := False;

timeoutValidQ[t_ /; NonNegative[t]] := True;

timeoutValidQ[_] := False;

End[];

EndPackage[];
