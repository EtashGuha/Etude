
OAuthClient`saveOAuthConnection;
OAuthClient`loadOAuthConnection;
OAuthClient`deleteOAuthConnection;
OAuthClient`findSavedOAuthConnections;
OAuthClient`checkpermissions;
OAuthClient`addpermissions;
OAuthClient`oauthdata;
OAuthClient`rawoauthdata;
OAuthClient`$CacheResults;
OAuthClient`$SaveConnection;
OAuthClient`$SaveConnectionDefault;
OAuthClient`oauthauthenticate;
OAuthClient`oauthdisconnect;

(Unprotect[#]; Clear[#])& /@ {
  OAuthClient`rawoauthdata, OAuthClient`oauthdata,
  OAuthClient`saveOAuthConnection,OAuthClient`loadOAuthConnection,OAuthClient`deleteOAuthConnection,OAuthClient`findSavedOAuthConnections,
  OAuthClient`checkpermissions,OAuthClient`addpermissions
}

Begin["OAuthClient`"];

Begin["`Private`"];

oauthservicesdata = OAuthClient`OAuthServicesData;

$OAuthCloudCredentialsQ = True;
$ChannelBrokerAvailableQ[] := Quiet[MatchQ[URLFetch[$ServiceConnectChannel, "StatusCode", "Cookies" -> {}],200]]
$UseChannelFramework = False;
$Debug = False;

OAuthClient`$CacheResults = False;
OAuthClient`$SaveConnectionDefault = False;
OAuthClient`$SaveConnection = False;

$useAuthHeader = False;

(* Import Functions *)
serviceName = ServiceConnections`Private`serviceName;
getServiceObject = ServiceConnections`Private`getServiceObject;
checkservicelist = ServiceConnections`Private`checkservicelist;
getServiceID = ServiceConnections`Private`getServiceID;
getServiceName = ServiceConnections`Private`getServiceName;
sortrequests = ServiceConnections`Private`sortrequests;
serviceRawRequests = ServiceConnections`Private`serviceRawRequests;
serviceRawPosts = ServiceConnections`Private`serviceRawPosts;
serviceRawDeletes = ServiceConnections`Private`serviceRawDeletes;
serviceRawPuts = ServiceConnections`Private`serviceRawPuts;
serviceRequests = ServiceConnections`Private`serviceRequests;
servicePosts = ServiceConnections`Private`servicePosts;
serviceDeletes = ServiceConnections`Private`serviceDeletes;
servicePuts = ServiceConnections`Private`servicePuts;
urlfetchFun = ServiceConnections`Private`urlfetchFun;
refreshFun = ServiceConnections`Private`refreshFun;
refreshtoken = ServiceConnections`Private`refreshtoken;
useChannel = ServiceConnections`Private`useChannel;
serviceInfo = ServiceConnections`Private`serviceInfo;
serviceAuthentication = ServiceConnections`Private`serviceAuthentication;
tokenread = ServiceConnections`Private`tokenread;

(************************************** OAuth Authentication **********************************)

OAuthClient`oauthauthenticate[name_, id_, authopts_]:= With[ {service = newoauthauthenticate[name, id, FilterRules[authopts, Except["Save"]]]},
	(* OAuthClient`$SaveConnection is set by the dialog window during authentication *)
	If[ Lookup[authopts, "Save", OAuthClient`$SaveConnection],
		OAuthClient`saveOAuthConnection[service];
		OAuthClient`$SaveConnection = OAuthClient`$SaveConnectionDefault
	];
    service
]

OAuthClient`oauthauthenticate[___]:=$Failed

newoauthauthenticate[name_, id_, authopts_]:=Module[{service,info,
	rawgets = oauthservicesdata[name, "RawGets"],
	gets = oauthservicesdata[name, "Gets"],
	rawposts = oauthservicesdata[name, "RawPosts"],
	posts = oauthservicesdata[name, "Posts"],
	rawdeletes = oauthservicesdata[name, "RawDeletes"],
	deletes = oauthservicesdata[name, "Deletes"],
	rawputs = oauthservicesdata[name, "RawPuts"],
	puts = oauthservicesdata[name, "Puts"]
	},

	info = oauthservicesdata[name, "ClientInfo"];
	If[info === $Canceled, Message[ServiceConnect::genconerr, name]; Throw[$Canceled]];
	info = ToString/@info;
	info = If[ !MatchQ[info,{_,_}],
				Message[ServiceConnect::genconerr, name];
				Throw[$Failed],
				{"ConsumerKey"->info[[1]],"ConsumerSecret"->info[[2]]}
			];

	service = newunknownoauthauthenticate[name,id,Join[oauthservicesdata[name, "Authentication"],info, authopts]];

	serviceRawRequests[id] = sortrequests[serviceRawRequests[id],rawgets];
	serviceRawPosts[id] = sortrequests[serviceRawPosts[id],rawposts];
	serviceRawDeletes[id] = sortrequests[serviceRawDeletes[id],rawdeletes];
	serviceRawPuts[id] = sortrequests[serviceRawPuts[id],rawputs];
	serviceRequests[id] =sortrequests[serviceRequests[id],gets];
	servicePosts[id] = sortrequests[servicePosts[id],posts];
	serviceDeletes[id] = sortrequests[serviceDeletes[id],deletes];
	servicePuts[id] = sortrequests[servicePuts[id],puts];
	service
]/;MemberQ[OAuthClient`$predefinedOAuthservicelist,name]

newoauthauthenticate[___]:=$Failed

otheroauthoptions = {"AccessVerb", "CodeExtractor",
	"RequestTokenExtractor", "RequestVerb", "ScopeDomain",
	"ScopeParameter", "SignatureMethod", "URLSignService","VerifierLabel","ResponseType"};

extraoauth2opts = {"CodeExtractor", "AccessEndpoint", "AccessVerb", "ScopeDomain",
	"AuthorizationFunction","AccessTokenRequestor"};

extraoauth1opts = {"RequestVerb", "CodeExtractor", "AccessEndpoint", "AccessVerb",
	"URLSignService", "SignatureMethod","AccessTokenExtractor", "ScopeParameter"};

defaultOAuthOptions = {
    "ServiceName"				-> Null,
    "OAuthVersion"       	 	-> "1.0a",
    "RequestEndpoint"			-> "",
    "AccessEndpoint"			-> Null,
    "AuthorizeEndpoint"			-> Null,
    "ConsumerKey"				-> Null,
    "ConsumerSecret"			-> Null,
    "RedirectURI"				-> "oob",
    "AdditionalOAuthParameter"	-> None,
    "Scope"						-> None,
    "AuthenticationDialog"		-> "TokenDialog",
    "RequestFormat"				-> "URL",
    "Information"				-> "",
    "AccessTokenExtractor"		-> None,
    "Blocking"          		-> True,
    "tokenread"					-> Identity,
    "RefreshAccessTokenFunction"-> None
};

newunknownoauthauthenticate[name_,id_,authopts___]:= Module[
	{version,authurl,requrl,accessurl,key,secret,redirect,dialogfun,token,urlfetchfun,service,dialog,requestformat,
		info,extra,scope,atokenext,extraopts,params,temprefreshtoken,refreshfun,refreshatoken,blocking},

	params = {"OAuthVersion","AuthorizeEndpoint","RequestEndpoint","AccessEndpoint","ConsumerKey","ConsumerSecret",
				"RedirectURI","AuthenticationDialog","RequestFormat","Information","AdditionalOAuthParameter",
				"Scope","AccessTokenExtractor","RefreshAccessTokenFunction","Blocking"};
	{version,authurl,requrl,accessurl,key,secret,redirect,dialog,
		requestformat,info,extra,scope,atokenext,refreshatoken,blocking} = params/.Flatten[{authopts}]/.defaultOAuthOptions;
	extraopts = Append[FilterRules[Flatten[{authopts}],Except[params]],"ConnectionID"->id];

	If[ !MatchQ[version,"1.0"|"1.0a"|"2.0"|"1"|"2"|1|2|1.|2.],
		Message[ServiceConnect::oauthver,version];
		Throw[$Failed]
	];
	If[ !StringQ[#],
		Message[ServiceConnect::url,#];
		Throw[$Failed]
	]&/@{authurl, requrl, accessurl};
	If[ !StringQ[#],
		Message[ServiceConnect::skey,#];
		Throw[$Failed]
	]&/@{key, secret};
	If[ !StringQ[redirect],
		Message[ServiceConnect::url,redirect];
		Throw[$Failed]
	];

	redirect=createRedirect[redirect,id, name];
	If[!StringQ[redirect],Message[ServiceConnect::url,redirect];Throw[$Failed]];

	dialogfun = getOAuthDialogFun[dialog,{name,id}];
	urlfetchfun = getOAuthFetchFun[requestformat, version];
	refreshfun = With[{Name = name, RefAToken = refreshatoken, AccessEndpoint = accessurl, ConsumerKey = key, ConsumerSecret = secret},
					getOAuthRefreshFun[Name, RefAToken, AccessEndpoint, ConsumerKey, ConsumerSecret]
				];

	token = Which[
				MatchQ[requestformat,"Headers"|{"Headers",__}]&&version==="1.0a",
					Block[ {OAuthSigning`Private`HMACSha1SignatureService,$useAuthHeader = requestformat},
							OAuthSigning`Private`initializedQ;
							OAuthSigning`Private`HMACSha1SignatureService[args__] :=
							With[ {res = OAuthSigning`Private`oAuth10SignURL[args]},
								Sequence @@ OAuthClient`Private`fromURLtoAuthorizationHeaders[{res}, "1.0a",requestformat]
							];
							newAuthenticate[name,version,authurl,requrl,accessurl,key,secret,
							redirect,{extra,scope},dialogfun,atokenext,blocking,extraopts]
					]
				,
				atokenext === "Refresh/2.0",
					Block[{tokenobject, rawtoken},
							tokenobject = newAuthenticate[name,version,authurl, requrl,accessurl,key,secret,
											redirect,{extra,scope},dialogfun,atokenext,blocking,extraopts];
							rawtoken = Cases[tokenobject,OAuthSigning`Private`Token20[l_List]:>l,Infinity];
							If[ SameQ[rawtoken, {}],
		                     	tokenobject,
								rawtoken = First[rawtoken];
								temprefreshtoken = formatrefreshtoken[Rest[rawtoken]];
								refreshtoken[id] = temprefreshtoken;
								Replace[tokenobject,rawtoken:>rawtoken[[1]],Infinity]
							]
					]
				,
				True,
					newAuthenticate[name,version,authurl, requrl, accessurl,key,secret,
						redirect,{extra,scope},dialogfun,atokenext,blocking,extraopts]
				];

	service=ServiceConnections`Private`createServiceObject["OAuth",name, token, id, blocking];

	refreshFun[id] = refreshfun;
	urlfetchFun[id] = urlfetchfun;
	tokenread[id] = Identity;
	serviceInfo[id] = info;
	service
]


newAuthenticate[name_,version_,authurl_, requrl_, accessurl_,key_,secret_,redirect_,
    {additionalparam_,scope_},dialogfun_, accesstokenext_,blocking_,extraopts_] :=
    Module[ {token, parameters,oauthflowdef,resetflowdef = False, uuid = Lookup[extraopts,"ConnectionID"]},
        parameters =
            If[ MatchQ[version,"1"|"1.0"|"1.0a"|1|1.],
                Join[
                {
                    "ServiceName"       -> name,
                    "OAuthVersion"		-> "1.0a",
                    "RequestEndpoint"   -> requrl,
                    "AccessEndpoint"    -> accessurl,
                    "AuthorizeEndpoint" -> authurl,
                	"RedirectURI"       -> redirect,
                    "ConsumerKey"		-> key,
                    "ConsumerSecret"    ->    secret,
                    "AuthenticationDialog" -> dialogfun,
		        	"Blocking"			-> blocking,
                    If[ accesstokenext=!=None,
                        "AccessTokenExtractor"->accesstokenext,
                        Sequence@@{}
                    ]
                },
                    FilterRules[extraopts,extraoauth1opts]
                ],
                Join[
                {

					"ServiceName"			-> name,
					"OAuthVersion"			-> "2.0",
					"AuthorizeEndpoint"		-> authurl,
					"AccessEndpoint"		-> accessurl,
					"RedirectURI"			-> redirect,
					"ConsumerKey"			-> key,
					"ConsumerSecret"		-> secret,
					"AuthenticationDialog"	-> dialogfun,
					"Blocking"			-> blocking,
					If[ accesstokenext=!=None,
						"AccessTokenExtractor"->accesstokenext,
						Sequence@@{}
					]

                },
                    FilterRules[extraopts,extraoauth2opts]
                ]
            ];

   	 	If[!$UseChannelFramework,
    		parameters=FilterRules[parameters,Except["Blocking"]]];
        Switch[{additionalparam,scope},
            {_Rule,None|{}},
				If[ version==="2.0",
					Throw[$Failed]
				];
				resetflowdef = True;
				oauthflowdef = DownValues[OAuthSigning`Private`OAuthFlow];
				parameters = Join[parameters,{"ScopeParameter" -> additionalparam[[1]]}];
				DownValues[OAuthSigning`Private`OAuthFlow] =
					Join[{HoldPattern[OAuthSigning`Private`OAuthFlow][auth_] :> OAuthSigning`Private`OAuthFlow[auth, {additionalparam[[2]]}]},
					oauthflowdef],
			{None,{__}},
				If[ version=!="2.0",
					Throw[$Failed]
				];
				resetflowdef = True;
				oauthflowdef = DownValues[OAuthSigning`Private`OAuthFlow];
				DownValues[OAuthSigning`Private`OAuthFlow] =
					Join[{HoldPattern[OAuthSigning`Private`OAuthFlow][auth_] :> OAuthSigning`Private`OAuthFlow[auth, scope]},
					oauthflowdef],
			{None,None|{}},
				Null,
			_,
				Message[ServiceConnect::addparam,addparam];
				Throw[$Failed]
		];
        token = tokenread[name]@getauthtoken[parameters,uuid];
        If[ resetflowdef,
            DownValues[OAuthSigning`Private`OAuthFlow] = oauthflowdef
        ];
        If[ Head[token] =!= OAuthSigning`OAuthToken,
            Message[ServiceConnect::token, name];
            Throw[$Failed]
        ];
        token
    ]

authenticationfunction[] :=
    If[ $OAuthCloudCredentialsQ,
        oauthCloudCredentials,
        OAuthSigning`OAuthAuthentication[#1]&
    ]

getauthtoken[parameters_,uuid_] :=
    Block[ {name = "ServiceName"/.parameters, token},
        token = Internal`CheckCache[{"OAuthTokens", name}];

        If[ Head[token] =!= OAuthSigning`OAuthToken,
            token = authenticationfunction[][parameters,uuid];
            If[ token === $Canceled,
                Return[$Canceled]
            ];
            If[ Head[token] =!= OAuthSigning`OAuthToken,
                Return[$Failed]
            ];
            Internal`SetCache[{"OAuthTokens", name}, token]
         ];
        token
    ]

(*********** refresh access token *************)
refreshAccessToken[id_] :=
	Block[ {newtoken,oldtoken,oldtokenobj,expdate,newreftoken,oldcache,OAuthClient`Private`$UseChannelFramework = TrueQ[useChannel[id]],res=$Failed},
		(
		res = Switch[$OAuthCloudCredentialsQ,True,First@refreshFun[id],False,Last@refreshFun[id]][id]
		)/;ListQ[refreshtoken[id]];
    	
		If[ ListQ[res],
			oldtokenobj = serviceAuthentication[id];
			oldtoken = Cases[oldtokenobj,OAuthSigning`Private`Token20[x_]:>x,Infinity];
			If[ ListQ[oldtoken], oldtoken = First[oldtoken]];
			newtoken = First[res]; {expdate,newreftoken} = Rest[res];
			serviceAuthentication[id] = Replace[oldtokenobj,oldtoken->newtoken,Infinity];
			Internal`DeleteCache[{"OAuthTokens", serviceName[id]}];
			Internal`SetCache[{"OAuthTokens", serviceName[id]}, serviceAuthentication[id]];
			If[SameQ[newreftoken,None],
				refreshtoken[id] = formatrefreshtoken@{First@refreshtoken[id], expdate},
				refreshtoken[id] = formatrefreshtoken@{newreftoken, expdate}
			]
			,
			Message[ServiceExecute::reftok,serviceName[id]];
			refreshtoken[id]
		]
	]

$refreshTokenSafetyMargin = 30;(* seconds *)

formatrefreshtoken[{token_,time_?NumberQ}]:= {token,Floor[UnixTime[]+time]-$refreshTokenSafetyMargin}
formatrefreshtoken[expr_] := expr

jsonAccessTokenAndRefreshExtractor[body_String] :=
     Block[ {rules, tokens, state},
         rules = Quiet@ImportString[body, "RawJSON"];
         (
			tokens = Lookup[rules, {"access_token", "expires_in", "refresh_token"}];
			tokens = Switch[tokens,
					{Repeated[_String | _Integer, {3}]},
	         			tokens,
	         		{Repeated[_String | _Integer, {2}], _Missing},
	         			ReplacePart[tokens, 3 -> None],
	         		_,
	         			Return[$Failed]
	         	];
			(tokens)/; (MatchQ[tokens,{(_String|_Integer|None)..}])
        ) /; AssociationQ[rules]
     ]

jsonAccessTokenAndRefreshExtractor[___]:=$Failed

(*************************************************Automatic refresh function generator***********************************************)
cloudAutomaticRefreshfunURL = "https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/OAuth/2.0/automatic-refresh";

automaticrefreshfun[AccessEndpoint_,ConsumerKey_,ConsumerSecret_]:=(Block[{data,res,tok,time},
	res = URLFetch[AccessEndpoint,
		"Method" -> "POST",
		"Parameters" -> {"client_id" -> ConsumerKey,"client_secret" -> ConsumerSecret, "grant_type" -> "refresh_token", "refresh_token"->First@refreshtoken[#]},
		"VerifyPeer"->True];
	If[StringQ[res],
		jsonAccessTokenAndRefreshExtractor[res],
		$Failed
	]])&

automaticrefreshfun[Name_,AccessEndpoint_,ConsumerKey_,ConsumerSecret_]:=
(Block[{reftok0=First@refreshtoken[#],reftok},
	reftok = ToExpression@URLFetch[cloudAutomaticRefreshfunURL,"Parameters"->{"Name"->Name,"AccessEndpoint"->AccessEndpoint,"refreshtoken" -> ToString[reftok0,InputForm],
	Sequence@@If[useChannel[#],{"ChannelBrokerQ"->"True"},{}], Sequence@@If[$Debug,{"debug"->"True"},{}]},"VerifyPeer"->False];
	If[ListQ[reftok],reftok,$Failed]
])&

automaticrefreshfun[___] := $Failed
(*************************************************HTTPBasic refresh function generator***********************************************)
cloudHTTPBasicRefreshfunURL = "https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/OAuth/2.0/http-refresh";

httpbasicrefreshfun[AccessEndpoint_,ConsumerKey_,ConsumerSecret_]:=(Block[{data,res,tok,time},
	res = URLFetch[AccessEndpoint,
		"Method" -> "POST",
    	"Headers"->{"Authorization"-> ("Basic "<>ExportString[ConsumerKey<>":"<>ConsumerSecret,"Base64"])},
        "Parameters" -> {"grant_type" -> "refresh_token", "refresh_token"->First@refreshtoken[#]},
        "VerifyPeer"->True];
	If[StringQ[res],
		jsonAccessTokenAndRefreshExtractor[res],
		$Failed
	]])&

httpbasicrefreshfun[Name_,AccessEndpoint_,ConsumerKey_,ConsumerSecret_]:=
(Block[{reftok0=First@refreshtoken[#],reftok},
	reftok = ToExpression@URLFetch[cloudHTTPBasicRefreshfunURL,"Parameters"->{"Name"->Name,"AccessEndpoint"->AccessEndpoint,"refreshtoken" -> ToString[reftok0,InputForm],
	Sequence@@If[useChannel[#],{"ChannelBrokerQ"->"True"},{}],Sequence@@If[$Debug,{"debug"->"True"},{}]},"VerifyPeer"->False];
	If[ListQ[reftok],reftok,$Failed]
])&

httpbasicrefreshfun[___] := $Failed

(************************************** ServiceDisconnect *****************************)
OAuthClient`oauthdisconnect[service_ServiceObject] := OAuthClient`oauthdisconnect[getServiceName[service],getServiceID[service]]

OAuthClient`oauthdisconnect[name_, id_]:=
    Quiet@Block[ {},
        Internal`DeleteCache[{"OAuthTokens", name}];
        serviceName[id]=.;
        serviceRawRequests[id]=.;
        serviceRequests[id]=.;
        serviceRawPosts[id]=.;
        servicePosts[id]=.;
        serviceRawDeletes[id]=.;
        serviceDeletes[id]=.;
        serviceRawPuts[id]=.;
        servicePuts[id]=.;
        serviceAuthentication[id]=.;
        urlfetchFun[id]=.;
        refreshFun[id]=.;
        useChannel[id]=.;
        serviceInfo[id]=.;
        refreshtoken[id]=.;
        ServiceConnections`Private`$authenticatedservices = DeleteCases[ServiceConnections`Private`$authenticatedservices,id];
    ]


OAuthClient`oauthdisconnect[___] :=
    $Failed

(************************************** ServiceExecute **********************************)
OAuthClient`oauthdata[service_ServiceObject,"Authentication"] :=
    With[ {auth = serviceAuthentication[getServiceID[service]]},
        parseToken[auth,getServiceName[service]]
    ]

parseToken[token_,name_]:= parseToken0[Cases[token,(p_OAuthSigning`Private`OAuth10Parameters|p_OAuthSigning`Private`OAuth20Parameters):>p,Infinity],name]

parseToken0[{params_OAuthSigning`Private`OAuth10Parameters},name_] :=
	{
	"OAuthVersion"		->	"1.0",
    "RequestEndpoint"	->	params[[9]],
    "AuthorizeEndpoint"	->	params[[11]],
    "AccessEndpoint"	->	params[[13]]
	}

parseToken0[{params_OAuthSigning`Private`OAuth20Parameters},name_] :=
	{
	"OAuthVersion"		->	"2.0",
	"AuthorizeEndpoint"	->	params[[9]],
	"AccessEndpoint"	->	params[[11]]
	}

parseToken[___]:= Throw[$Failed]

OAuthClient`oauthdata[service_ServiceObject,property_,rest___] :=
    Module[ {raw, id = getServiceID[service], old, new},

        If[ UnsameQ[refreshFun[id], None],
	        If[ SameQ[useChannel[id], None],
        		useChannel[id] = True;
        		old = refreshtoken[id];
				If[ AbsoluteTime[]>refreshtoken[id][[2]],
					new = Quiet[refreshAccessToken[id]];
					If[ SameQ[new, old],
						useChannel[id] = False;
						new = refreshAccessToken[id];
					]
				]
				,
				old = refreshtoken[id];
				If[ UnixTime[]>refreshtoken[id][[2]],
					new = refreshAccessToken[id],
					new = old
				]
	        ];
			If[ UnsameQ[new, old] && MemberQ[ServiceConnections`Private`$savedservices, id],
				ServiceConnections`SaveConnection[service]
			]
        ];

        If[ MemberQ[Join[serviceRequests[id],servicePosts[id],serviceDeletes[id],servicePuts[id]], property],
            OAuthClient`oauthcookeddata[getServiceName[service],property,id,rest],
            raw = OAuthClient`rawoauthdata[id,property,rest];
            parsedata[id,property]@raw
        ]
    ]

OAuthClient`oauthdata[args___] :=
    $Failed

OAuthClient`rawoauthdata[id_,parameter_,rest_] :=
    OAuthClient`rawoauthdata[id,parameter,{rest}]/;!ListQ[rest]

OAuthClient`rawoauthdata[id_,url0_String] :=
    Module[ {url, res},
    	If[!StringQ[Interpreter["URL"][url0]],
    		Message[ServiceExecute::invreq,url0, serviceName[id]];
    		Throw[$Failed]
    	];
        url = getsignedurl[id,url0,serviceAuthentication[id]];
        If[ url === $Failed,
            Throw[$Failed]
        ];
        If[ url === $Canceled,
            Return[$Canceled]
        ];
        (
             res = urlfetchFun[id]@@url;
             res /; (res =!= $Failed)
        ) /; (url =!= $Failed)
    ]/;!MemberQ[ServiceConnections`Private`availablequeries[id],url0]


OAuthClient`rawoauthdata[id_,property_,rest___] :=
    Module[ {url0,method,pathparams,params,bodyparams,mpdata,headers,reqparams,
    url, res, auth, tmp, pvpairs = Flatten[{rest}], params1, bodyparams1,mpdata1,headers1
    ,reqperms,returncontentdata, missingperms, oauth1Q,querydata},
        If[ OAuthClient`$CacheResults,
            res = Internal`CheckCache[{"OAuth", {id, property, rest}}];
            If[ res =!= $Failed,
                Return[res]
            ];
        ];
        querydata = ServiceConnections`Private`getQueryData[id, property];
        {url0,method,pathparams,params,bodyparams,mpdata,headers,reqparams, reqperms, returncontentdata} = Most[querydata];
        (* Check the required permissions *)
        If[ reqperms=!={},
            missingperms = If[ grantedpermissions[id]===All,
                               {},
                               With[ {updated = updatepermissions[id]},
                                   Cases[reqperms,_?(!MemberQ[updated,#]&),1]
                               ]
                           ];
            (* Try to add any missing permissions *)
            If[ missingperms=!={},
                If[ FailureQ[requestpermissions[id,missingperms]],
                    Throw[$Failed]
                ]
            ];
        ];

        (* check for required parameters *)
        If[ !MemberQ[First/@pvpairs,#],
            Message[ServiceExecute::nparam,#];
            Throw[$Failed]
        ]&/@reqparams;

        (* Path Parameters use a StringForm Function *)
        url = If[ Head[url0]===Function,
                  ServiceConnections`Private`insertpathparameters[url0,pathparams,pvpairs],
                  url0
              ];
        params1 = Cases[params,_?(!FreeQ[pvpairs,#]&)];
        params1 = Thread[params1->(params1/.pvpairs)];
        bodyparams1 = Cases[bodyparams,_?(!FreeQ[pvpairs,#]&)];
        bodyparams1 = Thread[bodyparams1->(bodyparams1/.pvpairs)];
        mpdata1=Append[List @@ #, Lookup[pvpairs, First[#]]] & /@ FilterRules[(Rule @@ #) & /@ mpdata, Keys[pvpairs]];
        auth = serviceAuthentication[id];
        If[ FreeQ[auth,OAuthSigning`Private`Token20|OAuthSigning`Private`Token10],
            Message[ServiceExecute::nolink,id];
            Throw[$Failed]
        ];
        oauth1Q = FreeQ[auth,OAuthSigning`Private`Token20|OAuthSigning`Private`OAuth20Parameters];
        If[ oauth1Q,
            url = getsignedurl[id,url,auth,"Parameters"->Join[params1, bodyparams1], "Method"->method],
            url = getsignedurl[id,url,auth,"Parameters"->params1,"BodyData"->bodyparams1, "Method"->method]
        ];
        If[ !MatchQ[url,_String|{_String,___}],
            Throw[$Failed]
        ];

        If[!MatchQ[("Parameters"/.Rest@url),"Parameters"],
        	pvpairs = Union[pvpairs,("Parameters"/.Rest@url)]
        ];

        If[ headers=!={},
            (* Headers should have default values, check for given values *)
            headers1 = If[ FreeQ[pvpairs,First[#]],
                           #,
                           First[#]->(First[#]/.pvpairs)
                       ]&/@headers;
            url = Join[url,{"Headers"->headers1}]
        ];
        If[ method==="POST",
            If[ oauth1Q,
                tmp = cutoutparameters1[url[[1]], bodyparams];
                url[[1]] = tmp[[1]];
                url = Join[url,{"BodyData"->tmp[[2]], "MultipartData"->mpdata1}],
                tmp = cutoutparameters2[Rest[url], bodyparams];
                tmp = tmp/.HoldPattern[Rule["BodyData",bd:(_Rule|{_Rule...})]]:>Rule["BodyData",URLQueryEncode[bd]];
                url = If[ mpdata1==={},
                          Join[{url[[1]]},tmp],
                          Join[{url[[1]]},tmp,{"MultipartData"->mpdata1}]
                      ];
                url[[1]] = URLBuild[url[[1]],Normal[KeyDrop["access_token"][Lookup[Rest[url],"Parameters"]]]];
            ]
        ];
        If[ returncontentdata,
            url = Insert[url,"ContentData",2]
        ];
        url = Join[url,{"CredentialsProvider" -> None}];
        If[ url === $Canceled,
            Return[$Canceled]
        ];
        (
             res = urlfetchFun[id]@@url;
             (If[ OAuthClient`$CacheResults,
                  Internal`SetCache[{"OAuth", {id, property, rest}}, res]
              ];
              res) /; (res =!= $Failed)
        ) /; (url =!= $Failed)
    ]/;property=!="Authentication"&&MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id],serviceRawDeletes[id],serviceRawPuts[id]], property]


OAuthClient`rawoauthdata[___] :=
    Throw[$Failed]

parsedata[id_,property_] :=
    (("ResultsFunction"/.oauthservicesdata[serviceName[id],property])/."ResultsFunction"->Identity
    )/;MemberQ[Join[serviceRawRequests[id],serviceRawPosts[id],serviceRawDeletes[id],serviceRawPuts[id]], property]

parsedata[__] :=
    Identity


(**************** Manage Permissions *************)
grantedpermissions[id_] :=
    {}/;!ServiceConnections`Private`authenticatedServiceQ[id]

grantedpermissions[id_] :=
    (grantedpermissions[id] = OAuthClient`checkpermissions[serviceName[id],id])

updatepermissions[id_] :=
    (grantedpermissions[id] = OAuthClient`checkpermissions[serviceName[id],id])/;ServiceConnections`Private`authenticatedServiceQ[id]

requestpermissions[id_,p_] :=
    (OAuthClient`addpermissions[serviceName[id],id,p])/;ServiceConnections`Private`authenticatedServiceQ[id]

updatepermissions[___] :=
    $Failed
requestpermissions[___] :=
    $Failed

(****************** Utilities *********************)
hyperlink[str_String]:= Hyperlink[str];
hyperlink[___]:= Null

fromURLtoAuthorizationHeaders[url_,"2.0"|"2", header0_] :=
    Module[ {params, token,headers,
    rest = Rest[url],header,addheaders,addcontentdata= !FreeQ[url,"ContentData"],method,partialResponse},
				If[addcontentdata,rest = Rest[rest]];
				If[ ListQ[header0],
            header = header0[[2]];
            addheaders = If[ Length[header0]>2,
                             header0[[3]],
                             {}
                         ],
            header = "Oauth";
            addheaders = {}
        ];
        method = "Method"/.rest;
        params = "Parameters"/.rest/."Parameters"->{};
        headers = ("Headers"/.rest)/."Headers"->{};
        token = "access_token"/.params;
        params = FilterRules[params,Except["access_token"]];
        partialResponse = {"Headers"->Join[headers,addheaders,
            {"Authorization"->(header<>" "<>token)}],"Parameters"->params,Sequence@@FilterRules[rest,Except["Parameters"|"Headers"]]};
		If[addcontentdata,PrependTo[partialResponse,"ContentData"]];
        If[ method==="POST",
            Join[{First[url]},partialResponse],
            Join[{URLParse[First[url],"AbsolutePath"]},partialResponse]
        ]
    ]

$oauthfields = {"oauth_consumer_key", "oauth_nonce", "realm","oauth_callback",
"oauth_signature_method", "oauth_timestamp", "oauth_token", "oauth_verifier",
"oauth_version", "oauth_signature"};

fromURLtoAuthorizationHeaders[url_,__,header0_] :=
    Module[ {split,addheaders,header,
    query,auth,headers},
        If[ ListQ[header0],
            header = header0[[2]];
            addheaders = If[ Length[header0]>2,
                             header0[[3]],
                             {}
                         ],
            header = "Oauth";
            addheaders = {}
        ];
        split = URLParse[First[url]];
        query = Lookup[split,"Query",{}];
        auth = FilterRules[query,$oauthfields];
        query = FilterRules[query,Except[$oauthfields]];
        auth = URLQueryEncode[auth];
        headers = ("Headers"/.Rest[url])/."Headers"->{};
        {URLBuild@Join[KeyDrop[split, "Query"], Association["Query" -> query]],
            "Headers"->Join[headers,addheaders,{
          "Authorization"->header<>" "<>StringReplace[auth, {"=" -> "=\"", "&" -> "\","}] <> "\""}],
          Sequence@@FilterRules[Rest[url],Except["Headers"]]
        }
    ]

cutoutparameters1[str_, {}] :=
    {str,""}
cutoutparameters1[str_, params0_] :=
    Module[ {tmp, url, body, url0,params},
        params = Join[params0,URLEncode/@params0];
        tmp = StringSplit[str, {"?", "&"}];
        tmp =  GatherBy[tmp, (StringFreeQ[#, StringJoin[#, "="] & /@ params] &)];
        {url0,body} = If[ Length[tmp]===1,
                          {First[tmp],{}},
                          tmp
                      ];
        url = First[url0];
        If[ Length[url0] > 1,
            url = url <> "?" <> url0[[2]];
            If[ Length[url0] > 2,
                url = StringJoin[url,"&", ##] & @@ Riffle[Drop[url0, 2], "&"]
            ]
        ];
        StringReplace[body = StringJoin[Riffle[body, "&"]],"ParameterlessBodyData*="->""];
        {url, body}
    ]

cutoutparameters2[opts_, {}] :=
    opts
cutoutparameters2[opts_, params0_] :=
    Module[ {body,params, urlparams,body0},
        params = Join[params0,URLEncode/@params0];
        body0 = "BodyData"/.opts;
        urlparams = "Parameters"/.opts;
        body = DeleteCases[urlparams,_?(FreeQ[#,Alternatives@@params] &)];
        body = Join[body0,body]/.HoldPattern[Rule["ParameterlessBodyData",x_]]:>x;(*
        body=URLEncode[body/.({"ParameterlessBodyData"->x_}:>x)];*)
        If[ MatchQ[body,{_String}|{{_Integer..}}],
            body = First[body]
        ];
        urlparams = DeleteCases[urlparams, _?(!FreeQ[#,Alternatives@@params] &)];
        Join[{"Parameters"->urlparams, "BodyData"->body},DeleteCases[opts,Rule["Parameters"|"BodyData",_]]]
    ]

(****************************** Token Storage *************************************)
OAuthClient`saveOAuthConnection[service_ServiceObject] :=
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

		data = createConnectionTokenData[service];
		data = OAuthClient`Private`ob[data];
		res = Export[co,AssociateTo[current,id->data],"RawJSON"];
		If[FailureQ[res],Message[ServiceConnect::nsave];$Failed]
	]

createConnectionTokenData[service_] :=
    Module[ {id = getServiceID[service]},
        {Last[serviceAuthentication[id]],refreshtoken[id],useChannel[id]}
    ]


tokenpattern = (_OAuthSigning`Private`Token20|_OAuthSigning`Private`Token10)

importSavedOAuthConnection[name_, {id_,{token:tokenpattern,refresh_,channelQ_}}]:=Block[
	{OAuthSigning`Private`OAuthFlow, ServiceConnections`Private`makeuuid, Internal`CheckCache, service},
    Internal`CheckCache[___] := $Failed;
    OAuthSigning`Private`OAuthFlow[___] = token;
    service = newoauthauthenticate[name, id, {}];
    refreshtoken[getServiceID[service]] = refresh;
    useChannel[getServiceID[service]] = channelQ;
    ServiceConnections`Private`appendauthservicelist[id];
    ServiceConnections`Private`appendsavedservicelist[id];
    service
]

OAuthClient`loadOAuthConnection[name_, id_, location_]:=Block[ {token = loadoauthConnection[name, id, location]},

	If[MatchQ[token, {_,{tokenpattern}}],
		token[[2]] = AppendTo[token[[2]],None]
	];

	If[MatchQ[token, {_,{tokenpattern,_}}],
		token[[2]] = AppendTo[token[[2]],None]
	];

	If[MatchQ[token, {_,{tokenpattern,_,_}}],
		importSavedOAuthConnection[name, token],
		$Failed
	]
]

loadoauthConnection[name_, id_:Automatic, location_:Automatic] :=
	Module[ {res},
		Switch[location,
			Automatic | All,
			If[ $CloudConnected,
				res = loadoauthConnectionCloud[name, id];
				If[ FailureQ[res],
					loadoauthConnectionLocal[name, id],
					res
				],
				res = loadoauthConnectionLocal[name, id];
				If[ FailureQ[res] && (id =!= Automatic || location === All),
					Message[ServiceConnect::ncloud];
					res,
					res
				]
			],
			"Cloud",loadoauthConnectionCloud[name, id],
			"Local",loadoauthConnectionLocal[name, id]
		]
	]

loadoauthConnectionLocal[name_, id_] :=
	Module[ {tmp, dir, file, files, fileid},

		dir = FileNameJoin[{$UserBaseDirectory,"Connections","Services",name}];
		If[ DirectoryQ[dir],

			files = FileNames["connection-*.txt", dir];
			If[ files === {}, Return[$Failed]];

			If[ StringQ[id],
				file = FileNameJoin[{dir,id<>".txt"}];
				If[ !FileExistsQ[file],
					Message[ServiceConnect::nost];
					Return[$Failed]
				];
				fileid = id,

				If[Length[files]>1, Message[ServiceConnect::multst]];
				file = Last[SortBy[FileDate][files]];
				fileid = FileBaseName@file
			];
			tmp = Get[file];
			{fileid, tmp},
			$Failed
		]
    ]

loadoauthConnectionCloud[name_, id_] :=
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

OAuthClient`deleteOAuthConnection[so_ServiceObject]:=OAuthClient`deleteOAuthConnection[getServiceName[so], getServiceID[so]]

OAuthClient`deleteOAuthConnection[name_,id_]:=(
	deleteSavedOAuthConnections[name, id];
	OAuthClient`oauthdisconnect[name, id];
)

deleteSavedOAuthConnections[name_, id_]:=(
	deleteSavedOAuthConnectionsCloud[name, id];
	deleteSavedOAuthConnectionsLocal[name, id]
)

deleteSavedOAuthConnectionsCloud[name_, id_]:=Block[{co, res, current,
	OAuthClient`Private`deobflag = True},
	co = CloudObject["connections/services/"<>name];
	current = Quiet[Import[co,"RawJSON"]];
	If[AssociationQ[current],
		Export[co,KeyDrop[current,id],"RawJSON"]
	]
]/;$CloudConnected

deleteSavedOAuthConnectionsLocal[name_, id_]:=Module[ {dir, file},
	dir = FileNameJoin[{$UserBaseDirectory,"Connections","Services",name}];
	If[DirectoryQ[dir],
		file = FileNameJoin[{dir,id<>".txt"}];
		If[FileExistsQ[file],
			DeleteFile[file]
		]
	]
]
    

OAuthClient`findSavedOAuthConnections[name_String]:=Join[
	findSavedOAuthConnectionsCloud[name],
	findSavedOAuthConnectionsLocal[name]]


findSavedOAuthConnectionsCloud[name_String]:=Block[{co, stored,
	OAuthClient`Private`deobflag = True},
	co = CloudObject["connections/services/"<>name];
    stored = Quiet[Import[co,"RawJSON"]];
    If[AssociationQ[stored],
    	Keys[stored],
    	{}
    ]
]/;$CloudConnected

findSavedOAuthConnectionsCloud[_]:={}

findSavedOAuthConnectionsLocal[name_String]:=Block[{dir=FileNameJoin[{$UserBaseDirectory,"Connections","Services",name}],
	files},
	If[DirectoryQ[dir],
		files = FileNames["connection-*.txt", dir];
          If[Length[files]>0,
          	FileBaseName/@files
          	,
          	{}
          ]
		,
		{}
	]	
]


(*********************** Cloud Stored Client credentials *************************)
oauthCloudCredentials[parameters_,uuid_] :=
    Block[ {name = ("ServiceName"/.parameters)},
        If[ MemberQ[OAuthClient`$predefinedOAuthservicelist,name],
            If[ MatchQ[("OAuthVersion"/.parameters),"1.0"|"1.0a"|"1"|1],
                oauth10CloudCredentials[parameters,uuid],
                oauth20CloudCredentials[parameters,uuid]
            ],
            OAuthSigning`OAuthAuthentication[parameters]
        ]
    ]

(* OAuth 2.0 *)
cloudAuthorizationBaseURL = "https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/OAuth/2.0/authorize-url";
cloudAccessBaseURL = "https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/OAuth/2.0/access-token";

cloudassembleauthurl[id_,rules_,scope_,state_,uuid_] :=
    Block[ {url, json},
		url=URLBuild[cloudAuthorizationBaseURL,Join[rules,{"scope"->scope,"state"->state,"connectionid"->uuid,
			Sequence@@If[TrueQ[useChannel[id]],{"ChannelBrokerQ"->"True"},{}]}]];
        json = URLFetch[url,"VerifyPeer"->False];
        url = ImportString[json,"JSON"];
        If[ !StringQ[url],
            Throw[$Failed]
        ];
        url
    ]
cloudassembleauthurl[___] :=
    $Failed

cloudaccesstoken[id_,rules_,verifier_,state_] :=
    Block[ {url, stringtoken,accesstoken},
		url=URLBuild[cloudAccessBaseURL,Join[rules,{"verifier"->verifier,"state"->state,
			Sequence@@If[TrueQ[useChannel[id]],{"ChannelBrokerQ"->"True"},{}]}]];
        stringtoken = URLFetch[url,"VerifyPeer"->False];
        accesstoken = ToExpression[stringtoken];
        (accesstoken)/;MatchQ[accesstoken,_OAuthSigning`Private`Token20]
    ]

cloudaccesstoken[args___] :=
    $Failed

preparescope[{}|None] :=
    "None"
preparescope[str_String] :=
    str
preparescope[{str_String}] :=
    str
preparescope[l:{_String..}] :=
    StringJoin[Riffle[l,"+"]]

authToAuthRules20[auth_] :=
    {
    "ServiceName"->auth[[1]],
    "AuthorizationFunction"->ToString[auth[[16]],InputForm],
    "AuthorizeEndpoint"->auth[[9]],
    "RedirectURI"->auth[[13]],
    "consumerKey"->auth[[7]]
    }

authToAccRules20[auth_] :=
    {
    "ServiceName"->auth[[1]],
    "AccessEndpoint"->auth[[11]],
    "AccessVerb"->auth[[12]],
    "AccessTokenExtractor"->auth[[14]],
    "AccessTokenRequestor"->ToString[auth[[17]],InputForm],
    "RedirectURI"->auth[[13]],
    "VerifyPeer"->ToString[auth[[3]],InputForm],
    "consumerKey"->auth[[7]],
    "consumerSecret"->auth[[8]]
    }

oauth20CloudCredentials[parameters_,uuid_] :=
    Block[ {OAuthSigning`Private`assembleAuthorizationURL20, OAuthSigning`Private`getAccessToken20},
        OAuthSigning`Private`assembleAuthorizationURL20[before_,
            auth_OAuthSigning`Private`OAuth20Parameters, token_, scope_, state_] :=
            (
            cloudassembleauthurl[
            	uuid,authToAuthRules20[auth],ToString[scope,InputForm],ToString[state,InputForm],ToString[uuid,InputForm]]);
        OAuthSigning`Private`getAccessToken20[auth_OAuthSigning`Private`OAuth20Parameters, token_, verifier_, state_] :=
            cloudaccesstoken[uuid,authToAccRules20[auth],verifier,ToString[state,InputForm]];
        OAuthSigning`OAuthAuthentication[parameters]
    ]

(* OAuth 1.0 *)
cloudSignBaseURL="https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/OAuth/1.0/URLSigner";

cloudsignurl[id_,name_,unsignedURL_, signatureMethod_, accessVerb_, consumerKey_, consumerSecret_, keyStr_, secretStr_] :=
    Block[ {url, json},
		url=URLBuild[cloudSignBaseURL,{
			"name"->name,"unsignedURL"->unsignedURL,"signatureMethod"->signatureMethod,"accessVerb"->accessVerb,
			"consumerKey"->consumerKey,"consumerSecret"->consumerSecret,"keyStr"->keyStr,"secretStr"->secretStr,
			Sequence@@If[TrueQ[useChannel[id]],{"ChannelBrokerQ"->"True"},{}]}];
        json = URLFetch[url,"VerifyPeer"->False];
        url = ImportString[json,"JSON"];
        If[ !StringQ[url],
            Throw[$Failed]
        ];
        url
    ]

oauth10CloudCredentials[parameters_,uuid_] :=
    Block[ {OAuthSigning`Private`HMACSha1SignatureService},
        With[ {name = "ServiceName"/.parameters},
            OAuthSigning`Private`HMACSha1SignatureService[
                unsignedURL_, signatureMethod_, accessVerb_, consumerKey_, consumerSecret_, keyStr_, secretStr_] :=
                If[ $useAuthHeader=!=False,
                    (Sequence @@ OAuthClient`Private`fromURLtoAuthorizationHeaders[{#}, "1.0a",$useAuthHeader])&,
                    Identity
                ]
                [cloudsignurl[uuid,name,unsignedURL,signatureMethod,accessVerb,consumerKey, consumerSecret, keyStr, secretStr]]
        ];
        OAuthSigning`OAuthAuthentication[parameters]
    ]

getsignedurl[uuid_,url_,auth_, opts___] :=
    Block[ {OAuthSigning`Private`HMACSha1SignatureService, name = auth[[1,1]], version},
        OAuthSigning`Private`HMACSha1SignatureService[
            unsignedURL_, signatureMethod_, accessVerb_, consumerKey_, consumerSecret_, keyStr_, secretStr_] :=
            If[ $useAuthHeader=!=False,
                (Sequence @@ OAuthClient`Private`fromURLtoAuthorizationHeaders[{#}, "1.0a",$useAuthHeader])&,
                Identity
            ]
            [cloudsignurl[uuid,name, unsignedURL,signatureMethod,accessVerb,consumerKey, consumerSecret, keyStr, secretStr]];
        OAuthSigning`OAuthSignURL[url, "OAuthAuthentication" -> auth, opts]
    ]/;$OAuthCloudCredentialsQ&&!FreeQ[auth,_OAuthSigning`Private`OAuth10Parameters]

getsignedurl[uuid_,url_,auth_, opts___] :=
    OAuthSigning`OAuthSignURL[url, "OAuthAuthentication" -> auth, opts]



getOAuthDialogFun[dialog_,{connectionname_,connectionid_}]:=Switch[{dialog,$CloudEvaluation},
				{"TokenDialog",True},
				OAuthClient`tokenOAuthDialog[#, {connectionname,connectionid}]&,
				{"TokenDialog",_},
				OAuthClient`tokenOAuthDialog[#, connectionname]&,
		        {"WolframConnectorChannel",_},
		        OAuthClient`oauthChannelVerify[#, {connectionname,connectionid}]&,
				{Except[_String],True},
				(dialog/.HoldPattern[OAuthClient`tokenOAuthDialog][first_,second_String, rest___]:>OAuthClient`tokenOAuthDialog[first,{second, connectionid}, rest]),
				{Except[_String],_},dialog,
				_,
				Message[ServiceConnect::dialog,dialogfun];Throw[$Failed]
			]

getOAuthFetchFun[requestformat_, version_]:=
        Switch[requestformat,
            "URL",
            URLFetch,
            "Headers"|{"Headers",__},
            With[ {v = version,
                r = If[ ListQ[requestformat],
                        requestformat[[1;;2]],
                        requestformat
                    ]},
                (With[ {newurl = fromURLtoAuthorizationHeaders[{##}, v,r]},
                     URLFetch@@newurl
                 ]&)
            ],
            _Function,
            requestformat,
            _,
            Message[ServiceConnect::reqform,requestformat];
            Throw[$Failed]
        ];

getOAuthRefreshFun[Name_, RefAToken_, AccessEndpoint_, ConsumerKey_, ConsumerSecret_]:=
	Switch[RefAToken,
			None,
				None,
			Automatic,
				{automaticrefreshfun[Name,AccessEndpoint,ConsumerKey,ConsumerSecret],
					automaticrefreshfun[AccessEndpoint,ConsumerKey,ConsumerSecret]},
			"HTTPBasic",
				{httpbasicrefreshfun[Name,AccessEndpoint,ConsumerKey,ConsumerSecret],
					httpbasicrefreshfun[AccessEndpoint,ConsumerKey,ConsumerSecret]},
			{_Function,_Function},
				RefAToken,
			_,
				Message[ServiceConnect::reffun,RefAToken];
				Throw[$Failed]
	];


End[];
End[];

SetAttributes[{
  OAuthClient`saveOAuthConnection,OAuthClient`loadOAuthConnection,OAuthClient`oauthdata
},
   {ReadProtected, Protected}
];