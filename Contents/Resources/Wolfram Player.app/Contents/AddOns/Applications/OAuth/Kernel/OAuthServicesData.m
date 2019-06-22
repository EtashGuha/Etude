
Begin["OAuthClient`"]

OAuthClient`$predefinedOAuthservicelist;
OAuthClient`OAuthServicesData;
OAuthClient`oauthcookeddata;
OAuthClient`oauthsendmessage;
OAuthClient`addOAuthservice;

Begin["`Private`"]

(Unprotect[#]; Clear[#])& /@ {OAuthClient`OAuthServicesData,OAuthClient`oauthcookeddata,OAuthClient`oauthsendmessage,OAuthClient`addOAuthservice}
Unprotect[OAuthClient`$predefinedOAuthservicelist];

defaultOAuthParams={
					(* defaults *)
					"ServiceName"				-> Null,
				    "OAuthVersion"				-> "1.0a",
				    "RequestEndpoint"			-> "",
				    "AccessEndpoint"			-> Null,
				    "AuthorizeEndpoint"			-> Null,
				   	"RedirectURI"				-> "oob",
				   	"VerifierLabel"				-> "verifier",
				   	"AdditionalOAuthParameter"	-> None,
				   	"Scope"						-> None,
				    "AuthenticationDialog"		-> "TokenDialog",
				    "RequestFormat"				-> "URL",
				    "ResponseType"				-> "code",
				    "AccessTokenExtractor"		-> None,
				    "Information"				-> "",
				    "RefreshAccessTokenFunction"-> None
				    };

defaultOAuthLabels=First/@defaultOAuthParams;		    
(*************************** OAuthServices *************************************)

OAuthClient`$predefinedOAuthservicelist={}

OAuthClient`OAuthServicesData[args___]:=With[{res=oauthservices[args]},
	res/;!FailureQ[res]&&Head[res]=!=oauthservicedata]

oauthservices[name_,prop___]:=Module[{data=Once[oauthservicedata[name]],availableproperties},
	availableproperties=First/@data;
	Switch[{prop},
		{},	data,
		{"Requests"},availableproperties,
		{"Authentication"},
			Join[Thread[defaultOAuthLabels->(defaultOAuthLabels/.Join[data,defaultOAuthParams])],
				FilterRules[data,Except[Join[defaultOAuthLabels,{"Gets","Posts","Deletes","Puts","RawGets","RawPosts","RawDeletes","RawPuts","ClientInfo"}]]]]
		,
		{Alternatives@@availableproperties},
		prop/.data,
		_,
		oauthservicedata[name,prop]
	]
]

oauthservices[___]:=$Failed

OAuthClient`OAuthServicesData[___]:=$Failed
(*************************** Data for Services *********************************)

OAuthClient`addOAuthservice[name_, dir_: DirectoryName[System`Private`$InputFileName]]:=Module[{funs, file},
	Unprotect[OAuthClient`$predefinedOAuthservicelist,oauthservicedata,
		OAuthClient`oauthcookeddata,OAuthClient`oauthsendmessage];
	OAuthClient`$predefinedOAuthservicelist=Union[Append[OAuthClient`$predefinedOAuthservicelist,name]];
	ServiceConnections`Private`appendservicelist[name,"OAuth"];
	file=FileNameJoin[{dir,name<>".m"}];
	If[!FileExistsQ[file],Return[$Failed]];
	funs=Get[file];
	oauthservicedata[name,args___]:=funs[[1]][args];
	OAuthClient`oauthcookeddata[name,args___]:=funs[[2]][args];
	OAuthClient`oauthsendmessage[name,args___]:=funs[[3]][args];
	If[Length[funs]>4,
		OAuthClient`checkpermissions[name,args___]:=funs[[4]][args];
		OAuthClient`addpermissions[name,args___]:=funs[[5]][args];
	];
	Protect[OAuthClient`$predefinedOAuthservicelist,oauthservicedata,
		OAuthClient`oauthcookeddata,OAuthClient`oauthsendmessage];
]



Unprotect[OAuthClient`oauthcookeddata,OAuthClient`oauthsendmessage,oauthservicedata];

oauthservicedata[___]:=$Failed

(**** error handling ***)
OAuthClient`oauthcookeddata[___]:=Throw[$Failed]
OAuthClient`oauthsendmessage[___]:=Throw[$Failed]
OAuthClient`checkpermissions[___]:=All
OAuthClient`addpermissions[___]:=Throw[$Failed]

SetAttributes[{OAuthClient`$predefinedOAuthservicelist,OAuthClient`OAuthServicesData,OAuthClient`oauthcookeddata,OAuthClient`oauthsendmessage,OAuthClient`addOAuthservice},{ReadProtected, Protected}];

End[];
End[];
{}