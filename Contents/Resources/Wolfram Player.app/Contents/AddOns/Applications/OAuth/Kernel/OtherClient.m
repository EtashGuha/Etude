
OtherClient`rawotherdata;
OtherClient`otherauthenticate;
OtherClient`otherdisconnect;
OtherClient`otherdata;

(Unprotect[#]; Clear[#])& /@ {
 OtherClient`rawotherdata,
 OtherClient`otherauthenticate,
 OtherClient`otherdisconnect,
 OtherClient`otherdata
}

Begin["OtherClient`"];

Begin["`Private`"];

otherservicesdata=OtherClient`OtherServicesData;

OtherClient`$CacheResults=False;

(* Import Functions *)
serviceName=ServiceConnections`Private`serviceName;
getServiceObject=ServiceConnections`Private`getServiceObject;
checkservicelist=ServiceConnections`Private`checkservicelist;
getServiceID=ServiceConnections`Private`getServiceID;
getServiceName=ServiceConnections`Private`getServiceName;
sortrequests = ServiceConnections`Private`sortrequests;
serviceRawRequests=ServiceConnections`Private`serviceRawRequests;
serviceRawPosts=ServiceConnections`Private`serviceRawPosts;
serviceRequests=ServiceConnections`Private`serviceRequests;
servicePosts=ServiceConnections`Private`servicePosts;
urlfetchFun=ServiceConnections`Private`urlfetchFun;
serviceInfo=ServiceConnections`Private`serviceInfo;
serviceAuthentication=ServiceConnections`Private`serviceAuthentication;

(************************************** Other Authentication **********************************)
OtherClient`otherauthenticate[name_, id_String, authopts_]:=(Message[ServiceConnect::key1, name];newotherauthenticate[name,id,authopts])
OtherClient`otherauthenticate[name_, "New", authopts_]:=(Message[ServiceConnect::key2, name];newotherauthenticate[name,Automatic,authopts])
OtherClient`otherauthenticate[name_, Automatic , authopts_]:=newotherauthenticate[name,Automatic,authopts]

newotherauthenticate[name_, id0_, authopts_]:=Module[{service,id,
	rawgets=otherservicesdata[name, "RawGets"], 
	gets=otherservicesdata[name, "Gets"],
	rawposts=otherservicesdata[name, "RawPosts"], 
	posts=otherservicesdata[name, "Posts"]
	},

	service=ServiceConnections`Private`createServiceObject["Other",name,None,id0];
	id=getServiceID[service];

	serviceRawRequests[id]=sortrequests[serviceRawRequests[id],rawgets];
	serviceRawPosts[id]=sortrequests[serviceRawPosts[id],rawposts];
	serviceRequests[id]=sortrequests[serviceRequests[id],gets];
	servicePosts[id]=sortrequests[servicePosts[id],posts];

	If[Lookup[authopts,"Save",False],
		Message[ServiceConnect::key1, name]
	];
	service
]/;MemberQ[OtherClient`$predefinedOtherservicelist,name]

newotherauthenticate[___]:=$Failed

(***************************** Exchanging data ***********************************)

OtherClient`otherdata[service_ServiceObject,property_,rest_]:=Module[{id=getServiceID[service]},
	If[MemberQ[Join[serviceRequests[id],servicePosts[id]],property],
		OtherClient`othercookeddata[getServiceName[service],property,rest]
		,
		If[MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id]],property],
			OtherClient`rawotherdata[id,property,rest]
			,
			$Failed
		]
	]
]

OtherClient`otherdata[args___]:=$Failed

OtherClient`rawotherdata[id_,property_,rest___]:=Module[
		{res},	
		If[OtherClient`$CacheResults,
			res = Internal`CheckCache[{"OAuth", {id, property, rest}}];
			If[res =!= $Failed, Return[res]]
		];
			res=OtherClient`otherrawdata[serviceName[id],property,rest];
			(
				If[OtherClient`$CacheResults,Internal`SetCache[{"OAuth", {id, property, rest}}, res]];
				res
			) /; (res =!= $Failed)
	]/;MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id]], property]

OtherClient`rawotherdata[___]:=Throw[$Failed]

(************************************** ServiceDisconnect *****************************)
OtherClient`otherdisconnect[service_ServiceObject]:=Module[
	{id=getServiceID[service]},
	
	serviceName[id]=.;
	serviceRawRequests[id]=.;
	serviceRequests[id]=.;
	serviceRawPosts[id]=.;
	servicePosts[id]=.;
	serviceAuthentication[id]=.;

	ServiceConnections`Private`$authenticatedservices=DeleteCases[ServiceConnections`Private`$authenticatedservices,id];	
]

OtherClient`otherdisconnect[___]:=$Failed

End[];
End[];