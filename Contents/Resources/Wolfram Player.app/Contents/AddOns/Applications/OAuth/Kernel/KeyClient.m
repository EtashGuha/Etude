
KeyClient`saveKeyConnection;
KeyClient`loadKeyConnection;
KeyClient`deleteKeyConnection;
KeyClient`findSavedKeyConnections;
KeyClient`rawkeydata;
KeyClient`keyauthenticate;
KeyClient`keydisconnect;
KeyClient`keydata;

(Unprotect[#]; Clear[#])& /@ {
 KeyClient`rawkeydata,
 KeyClient`keyauthenticate,
 KeyClient`keydisconnect,
 KeyClient`keydata
}

Begin["KeyClient`"];

Begin["`Private`"];

keyservicesdata=KeyClient`KeyServicesData;
$KeyCloudCredentialsQ=False;
KeyClient`$CacheResults=False;
KeyClient`$SaveConnectionDefault = False;
KeyClient`$SaveConnection = False;

(* Import Functions *)
serviceName=ServiceConnections`Private`serviceName;
getServiceObject=ServiceConnections`Private`getServiceObject;
checkservicelist=ServiceConnections`Private`checkservicelist;
getServiceID=ServiceConnections`Private`getServiceID;
getServiceName=ServiceConnections`Private`getServiceName;
sortrequests = ServiceConnections`Private`sortrequests;
serviceRawRequests=ServiceConnections`Private`serviceRawRequests;
serviceRawPosts=ServiceConnections`Private`serviceRawPosts;
serviceRawDeletes=ServiceConnections`Private`serviceRawDeletes;
serviceRawPuts=ServiceConnections`Private`serviceRawPuts;
serviceRequests=ServiceConnections`Private`serviceRequests;
servicePosts=ServiceConnections`Private`servicePosts;
serviceDeletes=ServiceConnections`Private`serviceDeletes;
servicePuts=ServiceConnections`Private`servicePuts;
urlfetchFun=ServiceConnections`Private`urlfetchFun;
serviceInfo=ServiceConnections`Private`serviceInfo;
serviceAuthentication=ServiceConnections`Private`serviceAuthentication;

(************************************** API Key Authentication **********************************)

KeyClient`keyauthenticate[name_, id_, authopts_]:= With[ {service = newkeyauthenticate[name, id, FilterRules[authopts, Except["Save"]]]},
	(* KeyClient`$SaveConnection is set by the dialog window during authentication *)
	If[ Lookup[authopts, "Save", KeyClient`$SaveConnection],
		KeyClient`saveKeyConnection[service];
		KeyClient`$SaveConnection = KeyClient`$SaveConnectionDefault
	];
	service
]

newkeyauthenticate[name_, id0_, authopts_]:=Module[{service,id,key,token,urlfetchfun,
	rawgets=keyservicesdata[name, "RawGets"],
	gets=keyservicesdata[name, "Gets"],
	rawposts=keyservicesdata[name, "RawPosts"],
	posts=keyservicesdata[name, "Posts"],
	rawdeletes=keyservicesdata[name, "RawDeletes"],
	deletes=keyservicesdata[name, "Deletes"],
	rawputs=keyservicesdata[name, "RawPuts"],
	puts=keyservicesdata[name, "Puts"]
	},

	key = Lookup[authopts, "APIKey", keyservicesdata[name, "ClientInfo"]];
	If[key === $Canceled, Message[ServiceConnect::genconerr, name]; Throw[$Canceled]];
	If[!MatchQ[key, _List?OptionQ], Message[ServiceConnect::genconerr, name]; Throw[$Failed]];

	urlfetchfun = Lookup[keyservicesdata[name, "Authentication"], "URLFetchFun"];
	token = KeyClient`KeyToken[name,getapikey[name, key]];
	service=ServiceConnections`Private`createServiceObject["APIKey",name,token,id0];
	id=getServiceID[service];

	urlfetchFun[id]=urlfetchfun;
	serviceRawRequests[id]=sortrequests[serviceRawRequests[id],rawgets];
	serviceRawPosts[id]=sortrequests[serviceRawPosts[id],rawposts];
	serviceRawDeletes[id]=sortrequests[serviceRawRequests[id],rawdeletes];
	serviceRawPuts[id]=sortrequests[serviceRawPosts[id],rawputs];
	serviceRequests[id]=sortrequests[serviceRequests[id],gets];
	servicePosts[id]=sortrequests[servicePosts[id],posts];
	serviceDeletes[id]=sortrequests[serviceRequests[id],deletes];
	servicePuts[id]=sortrequests[servicePosts[id],puts];
	
	service
]/;MemberQ[KeyClient`$predefinedKeyservicelist,name]

KeyClient`keyauthenticate[___]:=$Failed

(***************************** Exchanging data ***********************************)

KeyClient`keydata[service_ServiceObject,property_,rest___]:=Module[{raw, id=getServiceID[service]},
	If[MemberQ[Join[serviceRequests[id],servicePosts[id]],property],
		KeyClient`keycookeddata[getServiceName[service],property,id,rest]
		,
		If[MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id]],property],
			raw = KeyClient`rawkeydata[id,property,rest];
			parsedata[id,property]@raw
			,
			$Failed
		]
	]
]

KeyClient`keydata[args___]:=$Failed

KeyClient`rawkeydata[id_,parameter_,rest_]:=KeyClient`rawkeydata[id,parameter,{rest}]/;!ListQ[rest]

KeyClient`rawkeydata[id_,url0_String]:=Module[{url, res},
		url=addapikey[url0,Last[serviceAuthentication[id]]];
		If[url === $Failed, Throw[$Failed]];
        (
     		res = urlfetchFun[id]@@url;
     		res /; (res =!= $Failed)
        ) /; (url =!= $Failed)
	]/;!MemberQ[ServiceConnections`Private`availablequeries[id],url0]


KeyClient`rawkeydata[id_,property_,rest___]:=Module[
		{url0,method,pathparams,params,bodyparams,mpdata,headers,reqparams,
			url, res, key, pvpairs=Flatten[{rest}], params1, bodyparams1,mpdata1,headers1, useauth,returncontentdata,querydata},
		If[KeyClient`$CacheResults,
			res = Internal`CheckCache[{"OAuth", {id, property, rest}}];
			If[res =!= $Failed, Return[res]];
		];

		querydata=ServiceConnections`Private`getQueryData[id, property];
		{url0,method,pathparams,params,bodyparams,mpdata,headers,reqparams,returncontentdata,useauth}=Drop[querydata,{-3}];

		If[!MemberQ[First/@pvpairs,#],
			Message[ServiceExecute::nparam,#];Throw[$Failed]
		]&/@reqparams;

		If[!MemberQ[Join[pathparams,params,Values[bodyparams],First/@mpdata],#],
			Message[ServiceExecute::noget,#,serviceName[id]];Throw[$Failed]
		]&/@(First/@pvpairs);


		key=Last[serviceAuthentication[id]];
		url=If[Head[url0]===Function,
			ServiceConnections`Private`insertpathparameters[url0,pathparams,Join[pvpairs,key]],
			url0
		];

		params1=FilterRules[pvpairs,params];
		If[KeyExistsQ[bodyparams,"ParameterlessBodyData"],
		bodyparams1 = Lookup[pvpairs,Lookup[bodyparams,"ParameterlessBodyData"],""],
		bodyparams1 = ""];
		mpdata1=Append[List @@ #, Lookup[pvpairs, First[#]]] & /@ FilterRules[(Rule @@ #) & /@ mpdata, Keys[pvpairs]];
		url={url,
		"Parameters"->params1,
		"BodyData"->bodyparams1,
		"MultipartData"->mpdata1,
		"Method"->method};

		If[TrueQ[useauth],
			url=addapikey[url, key]
		];

		If[!MatchQ[url,_String|{_String,___}],Throw[$Failed]];

		If[headers=!={},

			headers1=If[!KeyExistsQ[pvpairs,First[#]],#,First[#]->Lookup[pvpairs,First[#]]]&/@headers;
			url=Join[url,{"Headers"->headers1}]
		];

		(
     		res=urlfetchFun[id]@@url;
			(
				If[KeyClient`$CacheResults,Internal`SetCache[{"OAuth", {id, property, rest}}, res]];
				res
			) /; (res =!= $Failed)
		) /; (url =!= $Failed)
	]/;MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id]], property]


KeyClient`rawkeydata[___]:=Throw[$Failed]

parsedata[id_,property_]:= Lookup[keyservicesdata[serviceName[id],property],"ResultsFunction",Identity]/;MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id]], property]

parsedata[__]:=Identity

(************************************** ServiceDisconnect *****************************)
KeyClient`keydisconnect[service_ServiceObject]:=
	Quiet@Block[
	{id=getServiceID[service]},

	serviceName[id]=.;
	serviceRawRequests[id]=.;
	serviceRequests[id]=.;
	serviceRawPosts[id]=.;
	servicePosts[id]=.;
	serviceAuthentication[id]=.;
	urlfetchFun[id]=.;

	ServiceConnections`Private`$authenticatedservices=DeleteCases[ServiceConnections`Private`$authenticatedservices,id];
]

KeyClient`keydisconnect[___]:=$Failed

(******************************* Key storage *******************************)
KeyClient`saveKeyConnection[service_ServiceObject] :=
	Block[ {id = getServiceID[service], name = getServiceName[service], co, data,
	OAuthClient`Private`deobflag = True, res, current},

		If[!$CloudConnected, CloudConnect[]];
		If[!$CloudConnected,
			Message[ServiceConnections`SaveConnection::cloud];
			Return[$Failed]
		];

		co = CloudObject["connections/services/"<>name];
		current = Quiet[Import[co,"RawJSON"]];
		If[FailureQ[current], current = <||>];
		
		data = OAuthClient`Private`ob[Last[serviceAuthentication[id]]];
		res = Export[co,AssociateTo[current,id->data],"RawJSON"];
		If[FailureQ[res],Message[ServiceConnections`SaveConnection::nsave];$Failed]
	]

importSavedKeyConnection[name_, {id_, token_}]:=Block[
	{ServiceConnections`Private`makeuuid, service},
    service = KeyClient`keyauthenticate[name, id, {"APIKey"->token}];
    ServiceConnections`Private`appendauthservicelist[id];
    ServiceConnections`Private`appendsavedservicelist[id];
    service
]

KeyClient`loadKeyConnection[name_, id_, location_]:=Block[{token=loadkeyConnection[name, id, location]},
	If[ MatchQ[token, {_String,_List}],
		importSavedKeyConnection[name, token],
		Message[ServiceConnections`LoadConnection::noso,name];
		$Failed
	]
]

loadkeyConnection[name_, id_, location_] :=
	Switch[location,
		Automatic,
			If[ $CloudConnected,
				loadkeyConnectionCloud[name, id],
				Message[ServiceConnect::ncloud];
				$Failed
			],
		"Cloud" | All,
			loadkeyConnectionCloud[name, id],
		"Local",
			Message[ServiceConnect::ncloud];
			$Failed
        ]

loadkeyConnectionCloud[name_, id_] :=
    Block[ {co, data, OAuthClient`Private`deobflag = True, res, stored, fileid},

		If[ !$CloudConnected, CloudConnect[]];
		If[ !$CloudConnected,
			Message[ServiceConnections`LoadConnection::cloud];
			Return[$Failed]
		];

		co = CloudObject["connections/services/"<>name];
		stored = Quiet[Import[co,"RawJSON"]];
		If[ FailureQ[stored] || (stored === <||>), Return[$Failed]];

		If[ StringQ[id],
			data = stored[id];
			If[MissingQ[data],
				Message[ServiceConnections`LoadConnection::nost];
				Return[$Failed]
			];
			data = OAuthClient`Private`deob[data];
			If[FailureQ[data],Return[$Failed]];
			{id, data},

			If[Length[stored]>1, Message[ServiceConnections`LoadConnection::multst]];
			stored = Take[stored, -1];
			fileid = First[Keys[stored]];
			data = OAuthClient`Private`deob[First[Values[stored]]];
			If[FailureQ[data],Return[$Failed]];
			{fileid, data}
		]
    ]


KeyClient`deleteKeyConnection[so_ServiceObject]:=KeyClient`deleteKeyConnection[getServiceName[so], getServiceID[so]]

KeyClient`deleteKeyConnection[name_,id_]:=(
	deleteSavedKeyConnections[name, id];
	KeyClient`keydisconnect[name, id];
)

deleteSavedKeyConnections[name_, id_]:=Block[{co, res, current,
	OAuthClient`Private`deobflag = True},
	co = CloudObject["connections/services/"<>name];
	current = Quiet[Import[co,"RawJSON"]];
	If[AssociationQ[current],
		Export[co,KeyDrop[current,id],"RawJSON"]
	]
]/;$CloudConnected


KeyClient`findSavedKeyConnections[name_]:=Block[{co, stored,
	OAuthClient`Private`deobflag = True},
	co = CloudObject["connections/services/"<>name];
	stored = Quiet[Import[co,"RawJSON"]];
	If[AssociationQ[stored],
		Keys[stored],
		{}
	]
]/;$CloudConnected

KeyClient`findSavedKeyConnections[name_]:={}

(************************ Utilities *********************************************)
addapikey[url_String,keys_List?OptionQ]:=URLBuild[url,keys]

addapikey[url_List,fields_List?OptionQ]:=With[{params=Lookup[Rest[url],"Parameters",{}]},
	{First[url],"Parameters"->Join[fields,FilterRules[params,Except[Keys[fields]]]],
		Sequence@@FilterRules[Rest[url],Except["Parameters"]]}
]
(************************ MakeBoxes for KeyClient token **************************)
KeyClient`KeyToken /: MakeBoxes[e : KeyClient`KeyToken[name_, tok_], format_] :=  With[{head = ToString[Head[e]], str = name <> " " <> "API Key"},
	TagBox[RowBox[{head, "[", "\[LeftSkeleton]", TagBox[MakeBoxes[str, format], Editable -> False], "\[RightSkeleton]", "]"}],
		InterpretTemplate[Hold[e]], Editable -> False, Selectable -> True, SelectWithContents -> True]]

(*********************** Cloud Stored Client credentials *************************)
cloudapikeBaseURL="https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/apikey";

cloudgetapikey[name_]:=Block[{url, key},
	url=URLBuild[cloudapikeBaseURL,{"ServiceName"->name}];
	key=URLFetch[url];
	url=ToExpression[key];
	If[!StringQ[url],$Failed,url]
]

getapikey[name_,apikey_]:=cloudgetapikey[name]/;$KeyCloudCredentialsQ&&MemberQ[KeyClient`$predefinedKeyservicelist,name] (*Internal*)
getapikey[_,apikey_]:=apikey

End[];
End[];