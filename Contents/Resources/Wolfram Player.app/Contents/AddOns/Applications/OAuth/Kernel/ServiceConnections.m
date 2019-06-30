
System`SaveConnection;

(Unprotect[#]; Clear[#])& /@ {
  SendMessage,

  ServiceConnections`ServiceInformation
}


Needs["OAuthSigning`"];
Needs["CloudObject`"];
Needs["ChannelFramework`"];

Begin["ServiceConnections`"];

ServiceConnections`ServiceInformation;

Begin["`Private`"];

$authenticatedservices={};
$savedservices={};

defaultParams[___]:={};
$oauthservices = OAuthClient`$predefinedOAuthservicelist;
$keyservices = KeyClient`$predefinedKeyservicelist;
$otherservices = OtherClient`$predefinedOtherservicelist;

$predefinedservices:=Flatten[{$oauthservices,$keyservices,$otherservices,localpacletServices[]}/.{
		HoldPattern[$oauthservices]->{},HoldPattern[$keyservices]->{},HoldPattern[$otherservices]->{}}];
(************************************** Add Service **********************************)
Services`Utilities`AddService["OAuth",
 <|

  "SubmitFunction" 					->		serviceSubmit,
  "ConnectFunction"					->		serviceConnect,
  "ExecuteFunction"					-> 		serviceExecute,
  "DisconnectFunction"				->		serviceDisconnect,
  "ServiceObjectTypesettingFunction"->		serviceObjectTypesettingFunction,
  "ServiceObjectExecuteFunction"	->		serviceObjectExecute,
  "SetOptionsFunction"				->		SetDefaultParams,
  "OptionsFunction"					->		GetDefaultParams
  |>
]
(************************************** ServiceConnect **********************************)
$ServiceConnectHiddenOptions = {"Authentication"->{}}

serviceConnect[args___]:=Module[
	{
		res,
		arguments = Check[System`Private`ArgumentsWithRules[ServiceConnect[args], {0, 3}, List, $ServiceConnectHiddenOptions],
							$Failed, {ServiceConnect::optx, ServiceConnect::nonopt}],
		opts
	},
	(
	 {arguments,opts}=arguments;
	 res = Catch[iServiceConnect[Sequence@@arguments,Association[opts]]];
	 res /; !FailureQ[res]
	) /; !FailureQ[arguments]
]

serviceConnect[___]:= False /; True

iServiceConnect[___]:=(Message[ServiceConnect::offline];$Failed)/;(!PacletManager`$AllowInternet)

iServiceConnect[service: "Services" : "Services", <||>]:= $Services

iServiceConnect[service: "Services" : "Services", <|__|>]:=(Message[ServiceConnect::sopt];$Failed)

iServiceConnect[service: Except[_String|_ServiceObject], __]:=(Message[ServiceConnect::invs, service];$Failed)

iServiceConnect[service: _String, __]:=(Message[ServiceConnect::invs, service];$Failed)/;!StringQ[service]

iServiceConnect[so: _ServiceObject, __]:=($Failed)/;!ServiceObjectQ[so]

iServiceConnect[so: _ServiceObject, defaults: (((_List | _Association)?OptionQ) | None) : {} , opts: _Association?AssociationQ]:= With[
	{name=getServiceName[so], id=getServiceID[so]},

	If[authenticatedServiceQ[id],
		SetDefaultParams[so, Normal[defaults]];
		so
		,
		iServiceConnect[name, id, defaults, opts]
	]

]

iServiceConnect[name_String, id: _?validuuidQ: Automatic, defaults: (((_List | _Association)?OptionQ) | None) : {}, opts: _Association?AssociationQ]:= Module[
	{so, authopts = Lookup[opts,"Authentication",{}], saved},

	If[MatchQ[id, Automatic | "New"] && TrueQ[opts["DeletePreviousConnections"]],
		Quiet[deleteConnection/@Union[serviceConnections[name],savedConnections[name]]]
	];

	Switch[id,
			Automatic,
				so = checkservicelist[$authenticatedservices, name, ServiceConnect];
				If[ !FailureQ[so],
					SetDefaultParams[so, Normal[defaults]];
					Throw[so]
				];
				saved = Quiet[ServiceConnections`LoadConnection[name]];
				If[ Quiet[ServiceObjectQ[saved]],
					Throw[saved]
				]
				,
			Except["New"],
				If[ authenticatedServiceQ[id],
					If[ UnsameQ[serviceName[id], name],
						Message[ServiceConnect::nameid, id, name];
						Throw[$Failed],
						so = getServiceObject[id];
						SetDefaultParams[so, Normal[defaults]];
						Throw[so]
					]
				];
				saved = Quiet[ServiceConnections`LoadConnection[name, id]];
				If[ Quiet[ServiceObjectQ[saved]],
					If[ UnsameQ[getServiceName[saved], name],
						Message[ServiceConnect::nameid, id, name];
						Throw[$Failed]
					];
					Throw[saved]
				]
	];

	If[TrueQ[opts[SaveConnection]],
		authopts = FilterRules[authopts, Except["Save"]];
		AppendTo[authopts,"Save"->True];
		so = authenticate[name, id, authopts];
		,
		so = authenticate[name, id, authopts];
	];
	so
]

iServiceConnect[___]:= $Failed

authenticate[name_, id_, authopts_]:=authenticate0[name, id, authopts]/;MemberQ[$Services,name]

authenticate0[name_, id0_, authopts0_]:= Block[

	{OAuthClient`Private`$UseChannelFramework=TrueQ[OAuthClient`Private`$ChannelBrokerAvailableQ[]],id,authopts},

	If[MatchQ[id0, Automatic | "New"], id = makeuuid[], id = id0];
	useChannel[id] = TrueQ[OAuthClient`Private`$UseChannelFramework];

	If[SameQ[id0, "New"],
		Internal`DeleteCache[{"OAuthTokens", name}]
	];

	If[TrueQ[OAuthClient`Private`$UseChannelFramework] && Lookup[authopts0,"Save",False],
		OAuthClient`ChannelSaveQ[id]=True;
		authopts = FilterRules[authopts0, Except["Save"]],
		authopts = authopts0
	];

	Quiet[Unset@Once[OAuthClient`Private`oauthservicedata[name]], {Unset::write, Unset::norep}];

	OAuthClient`oauthauthenticate[name,id,authopts]
]/;MemberQ[$oauthservices,name]

authenticate0[name_, id_, authopts_]:=KeyClient`keyauthenticate[name,id,authopts]/;MemberQ[$keyservices,name]
authenticate0[name_, id_, authopts_]:=OtherClient`otherauthenticate[name,id,authopts]/;MemberQ[$otherservices,name]
authenticate0[name_, id_, authopts_]:=pacletService[name, id, authopts]

authenticate[name_, id_, authopts_]:=pacletService[name, id, authopts]

(*************************************** $Services **************************************)
$services:=Union[$predefinedservices,findallpacletsServices[],serviceName/@$authenticatedservices]
(*************************************** ServiceSubmit **************************************)
serviceSubmit[req_ServiceRequest, opts : OptionsPattern[ServiceSubmit]] :=
 Module[
  {

	  hf, hk,
	  spec,
	  taskSchedule
   }
   ,

   hf = OptionValue[System`HandlerFunctions];
   
   hk = OptionValue[HandlerFunctionsKeys];
   
   taskSchedule = Tasks`InterpretTimeSpec[ServiceSubmit,{1}];

   spec = 
	<|
		"Caller" -> System`ServiceSubmit, 
		"TaskType" -> "Scheduled", 
   		"TaskEnvironment" -> "Session", 
   		"RealEvaluationExpression" -> serviceExecute[req],(*what to evaluate*)
   		"EvaluationExpression" -> Hold@System`ServiceSubmit[req,opts],(*what to show*)
   		"Schedule" -> taskSchedule, 
   		System`HandlerFunctions -> hf, 
   		System`HandlerFunctionsKeys -> hk, 
   		System`AutoRemove -> True, 
   		Method -> Automatic, 
   		"TaskOptions" -> {}
   	|>;
	
	Services`Utilities`ServiceSubmitTaskObject[spec]

 ]

(******************* pacletService **************************)

pacletService[name_, rest___]:=If[TrueQ[findandloadServicePaclet[name]],

		authenticate0[name, rest]
		,
		(Message[ServiceConnect::unkn,name];Throw[$Failed])
	]

findandloadServicePaclet[name_]:=With[{paclet=findservicepaclet[name]},
	If[Head[paclet]=!=PacletManager`Paclet,
		Return[$Failed]
	];
	loadServicePaclet[paclet]
]

findservicepaclet[name_]:=Block[{fullname=createPacletName[name], local},
	local=PacletManager`PacletFind[fullname];
	Switch[Length[local],
		0,findservicepacletRemote[name,fullname],
		_,
		First[local]
	]
]

findservicepacletRemote[name_,fullname_]:=Block[{remote, paclet},
	remote=PacletManager`PacletFindRemote[fullname];
	Switch[Length[remote],
		0,Return[$Failed],
		_,
		paclet = First[remote]
	];
	If[Head[paclet]===PacletManager`Paclet,
		PacletManager`PacletInstall[paclet],
		$Failed
	]
]

createPacletName[name_]:="ServiceConnection_"<>name

loadServicePaclet[paclet_]:=Block[{location, file},
	location=Lookup[PacletManager`PacletInformation[paclet], "Location", "location"];
	If[location==="location"||!StringQ[location],
		Return[$Failed]
	];
	file=FileNameJoin[{location,"Kernel","load.m"}];
	If[FileExistsQ[file],
		Get[file];
		True,
		$Failed
	]
]

localpacletServices[]:=Union[StringReplace[#["Name"]& /@
	PacletManager`PacletFind["ServiceConnection_*"], "ServiceConnection_" -> ""]]

findallpacletsServices[]:=Union[StringReplace[#["Name"]& /@
	Select[Join[PacletManager`PacletFindRemote["ServiceConnection_*"],
		PacletManager`PacletFind["ServiceConnection_*"]],
	#["Public"] =!= False &], "ServiceConnection_" -> ""]]

(**********************************Default Parameters**********************************)

SetDefaultParams[so_ServiceObject,opts__?OptionQ]:=
	Module[{dpnewlist=Flatten[List[opts]],id=getServiceID[so],newDefaultParams=Association[GetDefaultParams[so]]},
		AssociateTo[newDefaultParams,dpnewlist];
		defaultParams[id] = Normal[newDefaultParams];
		so
	]
SetDefaultParams[so_ServiceObject,None]:=
	Module[{id=getServiceID[so]},
		defaultParams[id] = {};
		so
	]
SetDefaultParams[so_ServiceObject,___]:= $Failed

GetDefaultParams[so_ServiceObject]:=defaultParams[getServiceID[so]]
GetDefaultParamsValues[so_ServiceObject]:=Values[GetDefaultParams[so]]
GetDefaultParamsNamesFormatted[so_ServiceObject]:=StringJoin[#,": "]&/@Keys[GetDefaultParams[so]]

GetDefaultParametersListFormatted[so_ServiceObject]:=
List[Dynamic[Column[BoxForm`SummaryItem/@Thread[{GetDefaultParamsNamesFormatted[so],Short[#,0.5]&/@GetDefaultParamsValues[so]}]]]]

(************************************** ServiceDisconnect *****************************)
serviceDisconnect[args___]:=With[{res=Catch[servicedisconnect[args]]},
	res/;!FailureQ[res]
]/;ArgumentCountQ[ServiceDisconnect, Length[{args}], {1}]

servicedisconnect[service_ServiceObject]:=(Message[ServiceDisconnect::nsc,service];$Failed)/;!authenticatedServiceQ[service]

servicedisconnect[service_ServiceObject]:=OAuthClient`oauthdisconnect[service]/;serviceType[getServiceID[service]]==="OAuth"

servicedisconnect[service_ServiceObject]:=KeyClient`keydisconnect[service]/;serviceType[getServiceID[service]]==="APIKey"

servicedisconnect[service_ServiceObject]:=OtherClient`otherdisconnect[service]/;serviceType[getServiceID[service]]==="Other"

servicedisconnect[___]:=$Failed

(************************************** ServiceObject **********************************)
serviceObjectExecute[service_ServiceObject,args___]:=With[{res=Catch[serviceobjectdata[service, args]]},
	res/;!FailureQ[res]
]

serviceObjectExecute[___]:= False /; True

serviceobjectdata[service_, ___]:=Message[ServiceObject::noauth]/;!authenticatedServiceQ[service]

serviceobjectdata[service_, req_String, rest___]:=Module[{},
	If[MemberQ[serviceobjectRequests[service],req],
		externalservice[service, req, rest]
		,
		Message[ServiceObject::noget,req, getServiceName[service]];Throw[$Failed]
	]
]

serviceobjectdata[___]:=$Failed

$ServiceObjectDetails = {"ID"}

ServiceObjectQ[so_ServiceObject]:= Module[{id},
	If[MatchQ[so, HoldPattern[ServiceObject[ _String?StringQ, a : Repeated[_?OptionQ]]] /; Complement[Keys[{a}], $ServiceObjectDetails] === {}],
		id = getServiceID[so];
		Switch[id,
				_Missing,
					Message[ServiceObject::noid, so];
					False,
				Except[_String?validuuidQ],
					Message[ServiceObject::invid, so];
					False,
				_,
					True
			],
		Message[ServiceObject::invs, so];
		False
	]
]

ServiceObjectQ[so___] := (Message[ServiceObject::invs, so]; False)

(* Special Requests *)
$specialRequests={"Authentication", "ID", "Information", "Name", "Requests", "RawRequests"};

(************************************** ExternalService **********************************)
serviceExecute[args___]:=With[{res=Catch[externalservice[args]]},
	res/;!FailureQ[res]
]

serviceExecute[___]:= False /; True

externalservice[___]:=(Message[ServiceExecute::offline];$Failed)/;(!PacletManager`$AllowInternet)

externalservice[name_String,rest___]:=With[{service=checkservicelist[$authenticatedservices, name, ServiceExecute]},
	If[ FailureQ[service],
		If[ MemberQ[$predefinedservices,name],
			With[{so=Catch[iServiceConnect[name, <||>]]},
				If[ !FailureQ[so],
					externalservice[so, rest],
					Message[ServiceExecute::nolink,name];
					Throw[$Failed]
				]
			],
			Message[ServiceExecute::invs, name];
			Throw[$Failed]
		]
		,
		externalservice[service,rest]
	]
]
externalservice[service_System`ServiceRequest]:= externalservice[service["Service"],Sequence@@Values[KeyTake[service[],{"Request","Parameters"}]]]
externalservice[service_ServiceObject,rest___]:=(Message[ServiceExecute::nolink,service];$Failed)/;!authenticatedServiceQ[service]
externalservice[service_ServiceObject,req_String,rest___]:=(Message[ServiceExecute::noget,req, getServiceName[service]];$Failed)/;!MemberQ[serviceobjectRequests[service],req]

externalservice[service_ServiceObject,"Name"]:=serviceinfo[service,"Name"]
externalservice[service_ServiceObject,"Requests"]:=With[{id=getServiceID[service]},
	DeleteCases[Union[serviceRequests[id],servicePosts[id],serviceDeletes[id],servicePuts[id],$specialRequests],"Requests"]]
externalservice[service_ServiceObject,"RawRequests"]:=With[{id=getServiceID[service]},
	Union[serviceRawRequests[id],serviceRawPosts[id],serviceRawDeletes[id],serviceRawPuts[id]]]
externalservice[service_ServiceObject,"ID"]:=getServiceID[service]
externalservice[service_ServiceObject,"Information"]:=serviceinfo[service,"Information"]
externalservice[service_ServiceObject,"Authentication"]:=OAuthClient`oauthdata[service,"Authentication"]/;serviceType[getServiceID[service]]==="OAuth"
externalservice[service_ServiceObject,"Authentication"]:={}

externalservice[service_ServiceObject,req_String, rules : ((_Rule | _RuleDelayed) ...) | _?AssociationQ]:=externalservice[service, req, Replace[{rules}, Association[asoc___] :> asoc, {1}]]
externalservice[service_ServiceObject,req_String, subreq : "Categories"]:=(OAuthClient`oauthdata[service, req, subreq])/;serviceType[getServiceID[service]]==="OAuth"
externalservice[service_ServiceObject,req_String, rules_List?OptionQ]:=(OAuthClient`oauthdata[service, req, rules])/;serviceType[getServiceID[service]]==="OAuth"
externalservice[service_ServiceObject,req_String, rules_List?OptionQ]:=(KeyClient`keydata[service, req, rules])/;serviceType[getServiceID[service]]==="APIKey"
externalservice[service_ServiceObject,req_String, rules_List?OptionQ]:=(OtherClient`otherdata[service, req, rules])/;serviceType[getServiceID[service]]==="Other"

externalservice[___]:=$Failed

(******************** ServiceInformation **********)
ServiceConnections`ServiceInformation[args___]:=With[{res=Catch[serviceinfo[args]]},
	res/;!FailureQ[res]
]

serviceinfo[service_ServiceObject,rest___]:=(serviceInfo[service, rest])/;(authenticatedServiceQ[service] && Quiet[ArgumentCountQ[ServiceConnections`ServiceInformation,1 + Length[{rest}], 1 ,2]])
serviceinfo[service_,___]:=(Message[ServiceInformation::nolink,service];$Failed)/;!authenticatedServiceQ[service]
serviceinfo[___]:=$Failed

serviceInfo[service_,"Information"]:=OAuthClient`OAuthServicesData[getServiceName[service],"Information"]/;serviceType[getServiceID[service]]==="OAuth"
serviceInfo[service_,"Information"]:=KeyClient`KeyServicesData[getServiceName[service],"Information"]/;serviceType[getServiceID[service]]==="APIKey"
serviceInfo[service_,"Information"]:=OtherClient`OtherServicesData[getServiceName[service],"Information"]/;serviceType[getServiceID[service]]==="Other"

serviceInfo[service_,"Name"]:=OAuthClient`OAuthServicesData[getServiceName[service],"ServiceName"]/;serviceType[getServiceID[service]]==="OAuth"
serviceInfo[service_,"Name"]:=KeyClient`KeyServicesData[getServiceName[service],"ServiceName"]/;serviceType[getServiceID[service]]==="APIKey"
serviceInfo[service_,"Name"]:=OtherClient`OtherServicesData[getServiceName[service],"ServiceName"]/;serviceType[getServiceID[service]]==="Other"

serviceInfo[id_]:=""/;!authenticatedServiceQ[id]

(****************** SendMessage *******************)

SendMessage[args___]:=With[{res=Catch[sendmessage[args]]},
	res/;!(FailureQ[res]||MatchQ[res,_MobileMessaging])]

sendmessage[___]:=(Message[SendMessage::offline];$Failed)/;(!PacletManager`$AllowInternet)

sendmessage[name_String,rest___]:=With[{service=checkservicelist[$authenticatedservices, name, SendMessage]},
	If[ FailureQ[service],
		If[ MemberQ[$predefinedservices,name],
			With[{so=Catch[iServiceConnect[name, <||>]]},
				If[ !FailureQ[so],
					sendmessage[so, rest],
					Message[SendMessage::nolink,name];
					Throw[$Failed]
				]
			],
			Message[SendMessage::invs, name];
			Throw[$Failed]
		]
		,
		sendmessage[service,rest]
	]
]

sendmessage[service_ServiceObject,rest___]:=(Message[SendMessage::nolink,service];Throw[$Failed])/;!authenticatedServiceQ[service]

sendmessage[service_ServiceObject,rest__]:=(OAuthClient`oauthsendmessage[getServiceName[service],getServiceID[service],rest])/;serviceType[getServiceID[service]]==="OAuth"
sendmessage[service_ServiceObject,rest__]:=(KeyClient`keysendmessage[getServiceName[service],getServiceID[service],rest])/;serviceType[getServiceID[service]]==="APIKey"
sendmessage[service_ServiceObject,rest__]:=(OtherClient`othersendmessage[getServiceName[service],getServiceID[service],rest])/;serviceType[getServiceID[service]]==="Other"

sendmessage["SMS",args___]:= MobileMessaging`MobileMessaging["SMS",args]
sendmessage["MMS",args___]:= MobileMessaging`MobileMessaging["MMS",args]

sendmessage["Email", rest__, o:OptionsPattern[]] := SendMail[rest, o]
sendmessage["Email" -> dest_, rest__, o:OptionsPattern[]] := SendMail[dest, rest, o]

sendmessage["Voice", rest__, o:OptionsPattern[]] := Speak[rest, o]

sendmessage[___]:=$Failed

(****************** Utilities *********************)
servicesdata[name_,property_]:=(OAuthClient`OAuthServicesData[name,property])/;MemberQ[$oauthservices,name]
servicesdata[name_,property_]:=(KeyClient`KeyServicesData[name,property])/;MemberQ[$keyservices,name]
servicesdata[name_,property_]:=(OtherClient`OtherServicesData[name,property])/;MemberQ[$otherservices,name]
servicesdata[___]:=$Failed

appendservicelist[service_ServiceObject,type_]:=appendservicelist[getServiceName[service],type]


appendservicelist[name_String,"OAuth"]:=($oauthservices=Union[Append[$oauthservices,name]])
appendservicelist[name_String,"APIKey"]:=($keyservices=Union[Append[$keyservices,name]])
appendservicelist[name_String,"Other"]:=($otherservices=Union[Append[$otherservices,name]])

appendauthservicelist[service_]:=($authenticatedservices=Union[Append[$authenticatedservices,service]])
appendsavedservicelist[service_String]:=($savedservices=Union[Append[$savedservices,service]])
appendsavedservicelist[services_List]:=($savedservices=Union[Join[$savedservices,services]])
removefromsavedservicelist[service_]:=($savedservices=DeleteCases[$savedservices,service])

makeuuid[]:=StringJoin["connection-",IntegerString[RandomInteger[{0, 16^32}], 16, 32]]
validuuidQ[id_String?StringQ]:=StringMatchQ[id, ("connection-" ~~ RegularExpression["(\\w){32}"])]
validuuidQ["New"]:=True
validuuidQ[Automatic]:=True
validuuidQ[___]:=False

createServiceObject[type_, name_, token_, id0: (_?validuuidQ) : Automatic, authQ : (True|False) : True]:=Module[{link, id},
	id=If[MatchQ[id0,Automatic|"New"], makeuuid[], id0];
	link=ServiceObject[name, "ID"->id];
	appendservicelist[link,type];
	If[authQ,appendauthservicelist[id]];

	serviceName[id]=name;
	serviceRawRequests[id]={};
	serviceRequests[id]={};
	serviceRawPosts[id]={};
	servicePosts[id]={};
	serviceRawDeletes[id]={};
	serviceDeletes[id]={};
	serviceRawPuts[id]={};
	servicePuts[id]={};
	serviceAuthentication[id]=token;
	serviceType[id]:=type;
	urlfetchFun[id]=URLFetch;
	link
]

getQueryData[id_,property_]:=With[{data=servicesdata[serviceName[id],property]},
	(* URL, method, pathparams, params, bodyparams, mpdata, headers, requiredparams, returncontentdata *)
	{Lookup[data,"URL",""],
	Lookup[data,"HTTPSMethod","GET"],
	listwrap@Lookup[data,"PathParameters",{}],
	listwrap@Lookup[data,"Parameters",{}],
	listwrap@Lookup[data,"BodyData",{}],
	listwrap@Lookup[data,"MultipartData",{}],
	listwrap@Lookup[data,"Headers",{}],
	listwrap@Lookup[data,"RequiredParameters",{}],
	listwrap@Lookup[data,"RequiredPermissions",{}],
	Lookup[data,"ReturnContentData",False],
	Lookup[data,"IncludeAuth",True]}
]

setQueryData[id_,Rule[prop_,data_]]:=getQueryData[id,prop]=data

servicedata[id_]:=Association[{
	"ServiceName"->serviceName[id],
	"ID"->id,
	"RawRequests"->serviceRawRequests[id],
	"Requests"->serviceRequests[id],
	"RawPostRequests"->serviceRawPosts[id],
	"PostRequests"->servicePosts[id],
	"RawDeleteRequests"->serviceRawDeletes[id],
	"DeleteRequests"->serviceDeletes[id],
	"RawPutRequests"->serviceRawPuts[id],
	"PutRequests"->servicePuts[id],
	"Authentication"->serviceAuthentication[id],
	"Information"->serviceInfo[id]
}]

availablequeries[id_]:=Join[serviceRequests[id],serviceRawRequests[id],servicePosts[id],serviceRawPosts[id],serviceDeletes[id],serviceRawDeletes[id],servicePuts[id],serviceRawPuts[id],
	$specialRequests]/;authenticatedServiceQ[id]
availablequeries[_]:={}

getServiceObject[id_]:=ServiceObject[serviceName[id],"ID"->id]

getServiceName[ServiceObject[name_, __]]:= name
getServiceID[ServiceObject[name_, opts__]]:=Lookup[{opts}, "ID"]
getServiceIcon[service_ServiceObject]:=With[{icon=servicesdata[getServiceName[service],"icon"]},
	If[MatchQ[icon,_Image|_Graphics],icon,defaultServiceObjectIcon]]

serviceName[id_]:=None/;!authenticatedServiceQ[id]
serviceRawRequests[id_]:={}/;!authenticatedServiceQ[id]
serviceRequests[id_]:={}/;!authenticatedServiceQ[id]
serviceRawPosts[id_]:={}/;!authenticatedServiceQ[id]
servicePosts[id_]:={}/;!authenticatedServiceQ[id]
serviceRawDeletes[id_]:={}/;!authenticatedServiceQ[id]
serviceDeletes[id_]:={}/;!authenticatedServiceQ[id]
serviceRawPuts[id_]:={}/;!authenticatedServiceQ[id]
servicePuts[id_]:={}/;!authenticatedServiceQ[id]
serviceAuthentication[id_]:={}/;!authenticatedServiceQ[id]

urlfetchFun[id_]:=URLFetch/;!authenticatedServiceQ[id]
refreshFun[id_]:=None/;!authenticatedServiceQ[id]
refreshtoken[id_]:=None/;!authenticatedServiceQ[id]
useChannel[id_]:=None/;!authenticatedServiceQ[id]
serviceType[id_]:=None/;!authenticatedServiceQ[id]
tokenread[id_]:= Identity/;!authenticatedServiceQ[id]

sortrequests[l1_,l2_]:=Sort[Select[Flatten[{l1,l2}],StringQ]]
serviceobjectRequests[service_]:=With[{id=getServiceID[service]},
	availablequeries[id]
]

authenticatedServiceQ[service_ServiceObject]:=authenticatedServiceQ[getServiceID[service]]
authenticatedServiceQ[id_]:=MemberQ[$authenticatedservices,id]

savedServiceQ[service_ServiceObject]:=savedServiceQ[getServiceID[service]]
savedServiceQ[id_]:=MemberQ[$savedservices,id]

checkservicelist[list_, name_String, fn_]:=With[{matches=Select[list,serviceName[#]===name&]},
	Switch[Length[matches],
		1,getServiceObject[First[matches]],
		0,$Failed,
		_,Message[fn::multser,name];getServiceObject[First[matches]]
	]
]

listwrap[l_List]:=l
listwrap[x_]:={x}

insertpathparameters[url0_,pathparams_,pvpairs_]:=Module[{pparams1},
		pparams1= Lookup[pvpairs, pathparams, Automatic];
		Check[url0@@(pparams1),Message[ServiceExecute::nargs];Throw[$Failed]]
]

makeServiceResponse[req_,val_]:=System`ServiceResponse[<|"ServiceRequest"->req,"Result"->val|>]
(********************************** ServiceObject Typesetting ******************************)

serviceObjectTypesettingFunction[service : ServiceObject[name0_String, Rule["ID",id0_String]] , form:StandardForm|TraditionalForm] :=
With[{below0=GetDefaultParams[service],below=GetDefaultParametersListFormatted[service], name=getServiceName[service], id=getServiceID[service], icon=getServiceIcon[service]},
	If[$VersionNumber>=10,
		BoxForm`ArrangeSummaryBox[
			(* Head *)ServiceObject,
			(* Interpretation *)service,
			(* Icon *)icon,
			(* Column or Grid *){name, (* id, *) Dynamic[If[TrueQ[authenticatedServiceQ[id]],"Connected","Not Connected"]]},
			(* Plus Box Column or Grid *)
				{(*
					BoxForm`SummaryItem[{"ID: ", id}],
					Dynamic[Switch[
						{TrueQ[authenticatedServiceQ[id]],TrueQ[savedServiceQ[id]]},
						{True, True},
						ChoiceButtons[{"Delete","Disconnect"},{ServiceConnections`DeleteConnection[service],ServiceDisconnect[service]}],
						{True, False},
						ChoiceButtons[{"Save","Disconnect"},{ServiceConnections`SaveConnection[service],ServiceDisconnect[service]}],
						{False,True},
						ChoiceButtons[{"Delete","Connect"},{ServiceConnections`DeleteConnection[service],ServiceConnect[service]}],
						_,
						""
					]
					],*)
					If[MatchQ[below0,{}],Sequence@@{},below]
				},
			form]
			,
		InterpretationBox[#,service]&@ToBoxes[Framed@Row[{"ServiceObject       ",Column[{name, id, "Authenticated"->authenticatedServiceQ[id]}]}]]
	]
]




(**)
defaultServiceObjectIcon=BoxForm`GenericIcon[LinkObject]

End[];
End[];

SetAttributes[{
  SendMessage,ServiceConnections`ServiceInformation
},
   {ReadProtected, Protected}
];


{System`SendMessage}
