Begin["GooglePlusOAuth`"]

ServiceExecute::gplusm="You can either use MaxItems or \"MaxResults\" but not both."

Begin["`Private`"]

(******************************* GooglePlus *************************************)

(* Authentication information *)

googleplusdata[]:=
	If[TrueQ[OAuthClient`Private`$UseChannelFramework],
		{
    	"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "Google+",
	    "AuthorizeEndpoint"		-> "https://accounts.google.com/o/oauth2/v2/auth",
	    "AccessEndpoint"		-> "https://www.googleapis.com/oauth2/v4/token",
	    "RedirectURI" 			-> "WolframConnectorChannelListen",
        "Blocking"				-> False,
        "VerifierLabel"			-> "code",
        "ClientInfo"			-> {"Wolfram","Token"},
        "AuthorizationFunction"	-> "GooglePlus",
        "RedirectURLFunction"	-> (#1&),
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"RefreshAccessTokenFunction" -> Automatic,
		"VerifyPeer"			-> True,
        "AuthenticationDialog"	:> "WolframConnectorChannel",
	 	"Gets"					-> {"UserData","ActivityList","UserSearch","ActivitySearch","ActivityData",
	 		"ActivityPlusOners","ActivityResharers","UserPosts","UserPostsTimeline","UserPostsEventSeries","CircledUsers"},
	 	"Posts"					-> {},
	 	"RawGets"				-> {"RawUserData","RawPeopleSearch","RawPeopleByActivity","RawPeopleByCollection",
	 		"RawActivity","RawActivitySearch","RawComment","RawActivityComments","RawUserMoments","RawUserActivities"},
	 	"RawPosts"				-> {"RawInsertMoment"},
	 	"Scope"					-> {"https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fplus.login&request_visible_actions=https%3A%2F%2Fschemas.google.com%2FAddActivity+https%3A%2F%2Fschemas.google.com%2FReviewActivity+http%3A%2F%2Fschemas.google.com%2FCheckInActivity"},
 		"Information"			-> "A service for receiving data from a Google+ account"
	}
    ,
    {
    	"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "Google+",
	    "AuthorizeEndpoint"		-> "https://accounts.google.com/o/oauth2/v2/auth",
	    "AccessEndpoint"		-> "https://www.googleapis.com/oauth2/v4/token",
        "RedirectURI"			-> "https://www.wolfram.com/oauthlanding/?service=GooglePlus",
        "VerifierLabel"			-> "code",
        "ClientInfo"			-> {"Wolfram","Token"},
        "AuthorizationFunction"	-> "GooglePlus",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"RefreshAccessTokenFunction" -> Automatic,
        "AuthenticationDialog" 	:> (OAuthClient`tokenOAuthDialog[#, "Google+",gpicon]&),
	 	"Gets"					-> {"UserData","ActivityList","UserSearch","ActivitySearch","ActivityData",
	 		"ActivityPlusOners","ActivityResharers","UserPosts","UserPostsTimeline","UserPostsEventSeries","CircledUsers"},
	 	"Posts"					-> {},
	 	"RawGets"				-> {"RawUserData","RawPeopleSearch","RawPeopleByActivity","RawPeopleByCollection",
	 		"RawActivity","RawActivitySearch","RawComment","RawActivityComments","RawUserMoments","RawUserActivities"},
	 	"RawPosts"				-> {"RawInsertMoment"},
	 	"Scope"					-> {"https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fplus.login&request_visible_actions=https%3A%2F%2Fschemas.google.com%2FAddActivity+https%3A%2F%2Fschemas.google.com%2FReviewActivity+http%3A%2F%2Fschemas.google.com%2FCheckInActivity"},
 		"Information"			-> "A service for receiving data from a Google+ account"
    }
]

(* a function for importing the raw data - usually json or xml - from the service *)
googleplusimport[$Failed]:=Throw[$Failed]

googleplusimport[json_String]:=With[{res = Quiet[Developer`ReadRawJSONString[json]]},
	If[ AssociationQ[res],
		If[ !KeyExistsQ[res, "error"],
			res,
			Message[ServiceExecute::apierr, res["error"]["message"]];
			Throw[$Failed]
		],
		Message[ServiceExecute::serror];
		Throw[$Failed]
	]
]

googleplusimport[raw_]:=raw

(*** Raw ***) 

(** People **)
googleplusdata["RawUserData"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/people/`1`",formatuserid[##]]&),
        "PathParameters"	-> {"userID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }

googleplusdata["RawPeopleSearch"] = {
        "URL"				-> "https://www.googleapis.com/plus/v1/people",
        "Parameters"		-> {"query","language","maxResults","pageToken"},
        "RequiredParameters"-> {"query"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }

googleplusdata["RawPeopleByActivity"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/activities/`1`/people/`2`",ToString[#1],tostring[##2,"plusoners"]]&),
        "PathParameters"	-> {"activityID","collection"},
        "Parameters"		-> {"maxResults","pageToken"},
        "RequiredParameters"-> {"activityID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }

googleplusdata["RawPeopleByCollection"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/people/`1`/people/`2`",fp[{formatuserid,tostring[##,"visible"]&},##]]&),
        "PathParameters"	-> {"userID","collection"},
        "Parameters"		-> {"maxResults","orderBy","pageToken"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }
    
(** Activities **)
googleplusdata["RawActivity"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/activities/`1`",ToString[#]]&),
        "PathParameters"	-> {"activityID"},
        "RequiredParameters"-> {"activityID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }

googleplusdata["RawActivitySearch"] = {
        "URL"				-> "https://www.googleapis.com/plus/v1/activities",
        "Parameters"		-> {"query","language","maxResults","orderBy","pageToken"},
        "RequiredParameters"-> {"query"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }

googleplusdata["RawUserActivities"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/people/`1`/activities/`2`",fp[{formatuserid,tostring[##,"public"]&},##]]&),
        "PathParameters"	-> {"userID","collection"},
        "Parameters"		-> {"maxResults","pageToken"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }    
    
(** Comments **)
googleplusdata["RawComment"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/comments/`1`",ToString[#]]&),
        "PathParameters"	-> {"commentID"},
        "RequiredParameters"-> {"commentID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }
    
googleplusdata["RawActivityComments"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/activities/`1`/comments",ToString[#]]&),
        "PathParameters"	-> {"activityID"},
        "RequiredParameters"-> {"activityID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }
(** Moments **)
googleplusdata["RawUserMoments"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/people/`1`/moments/`2`",fp[{formatuserid,tostring[##,"vault"]&},##]]&),
        "PathParameters"	-> {"userID","collection"},
        "Parameters"		-> {"maxResults","targetUrl","pageToken","type"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> googleplusimport
    }
    
googleplusdata["RawInsertMoment"] = {
        "URL"				-> (ToString@StringForm["https://www.googleapis.com/plus/v1/people/`1`/moments/`2`",fp[{formatuserid,tostring[##,"vault"]&},##]]&),
        "PathParameters"	-> {"userID","collection"},
        "Parameters"		-> {"debug"},
        "BodyData"			-> {"ParameterlessBodyData"},
        "RequiredParameters"-> {"ParameterlessBodyData"},
        "HTTPSMethod"		-> "POST",
      	"Headers" 			-> {"Content-Type" -> "application/json"},
        "ResultsFunction"	-> googleplusimport
    }

googleplusdata["icon"]=gpicon;
 
googleplusdata[___]:=$Failed
(****** Cooked Properties ******)

$googleplusactivitysearchlimit=20;
$googleplususersearchlimit=50;
$googleplusactivitylistlimit=100;

(* Cooked *)

googlepluscookeddata["UserData",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,user,params={}},

	invalidParameters = Select[Keys[args],!MemberQ[{"UserID"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"UserID"],
		user = Lookup[args,"UserID"];
		If[!(IntegerQ[user]||StringQ[user]),
			Message[ServiceExecute::nval,"UserID","GooglePlus"];
			Throw[$Failed]
		];
		AppendTo[params,"userID"->ToString@user]
	];
	
	raw = OAuthClient`rawoauthdata[id,"RawUserData",params];
	data = googleplusimport[raw];
	Replace[data, {asoc_Association :> KeyMap[camelcase, asoc]}, {0,Infinity}]
]

googlepluscookeddata["UserSearch",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,error,rqfun,query,limit=25,flag=0,start=1,count,rest,tmp,tmpdata={},params=<|"maxResults"->ToString@$googleplususersearchlimit|>,ptoken},

	invalidParameters = Select[Keys[args],!MemberQ[{"Query","MaxResults",MaxItems,"StartIndex"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Query"],
		query = Lookup[args,"Query"];
		If[!StringQ[query],
			Message[ServiceExecute::nval,"Query","GooglePlus"];
			Throw[$Failed]
		];
		AppendTo[params,"query"->query],
		Message[ServiceExecute::nparam,"Query"];
		Throw[$Failed]
	];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","GooglePlus"];
			Throw[$Failed]
		]
	];

	Which[
		KeyExistsQ[args,MaxItems] && !KeyExistsQ[args,"MaxResults"],
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,"MaxResults"] && !KeyExistsQ[args,MaxItems],
			limit = Lookup[args,"MaxResults"];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,"MaxResults","GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,MaxItems] && KeyExistsQ[args,"MaxResults"],
			Message[ServiceExecute::gplusm];
			Throw[$Failed]
	];

	count = Quotient[limit+start-1,$googleplususersearchlimit,1];
	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPeopleSearch", Normal@#]&)];

	Do[
		raw = googleplusimport@(rqfun@params);
			ptoken = raw["nextPageToken"];
			tmp = raw["items"];
			tmpdata = Join[tmpdata,tmp];
			If[Length[tmp]<$googleplususersearchlimit,flag=1;Break[]];
			If[!MissingQ[ptoken],AppendTo[params,"pageToken"->ptoken],flag=1;Break[]]
	,count];

	Switch[flag,
			0,
				raw = googleplusimport@(rqfun@params);
				tmp = raw["items"];
				tmpdata = Join[tmpdata,tmp];
				If[Length[tmpdata]>(start-1),
					formatuser/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				],
			1,
				If[Length[tmpdata]>(start-1),
					formatuser/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				]
	]
]

googlepluscookeddata["ActivitySearch",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,error,rqfun,query,limit=10,flag=0,start=1,count,rest,tmp,tmpdata={},params=<|"maxResults"->ToString@$googleplusactivitysearchlimit|>,ptoken},

	invalidParameters = Select[Keys[args],!MemberQ[{"Query","OrderBy","MaxResults",MaxItems,"StartIndex"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Query"],
		query = Lookup[args,"Query"];
		If[!StringQ[query],
			Message[ServiceExecute::nval,"Query","GooglePlus"];
			Throw[$Failed]
		];
		AppendTo[params,"query"->query],
		Message[ServiceExecute::nparam,"Query"];
		Throw[$Failed]
	];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","GooglePlus"];
			Throw[$Failed]
		]
	];

	Which[
		KeyExistsQ[args,MaxItems] && !KeyExistsQ[args,"MaxResults"],
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,"MaxResults"] && !KeyExistsQ[args,MaxItems],
			limit = Lookup[args,"MaxResults"];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,"MaxResults","GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,MaxItems] && KeyExistsQ[args,"MaxResults"],
			Message[ServiceExecute::gplusm];
			Throw[$Failed]
	];

	count = Quotient[limit+start-1,$googleplusactivitysearchlimit,1];
	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawActivitySearch", Normal@#]&)];

	Do[
		raw = googleplusimport@(rqfun@params);
			ptoken = raw["nextPageToken"];
			tmp = raw["items"];
			tmpdata = Join[tmpdata,tmp];
			If[Length[tmp]<$googleplusactivitysearchlimit,flag=1;Break[]];
			If[!MissingQ[ptoken],AppendTo[params,"pageToken"->ptoken],flag=1;Break[]]
	,count];

	Switch[flag,
			0,
				raw = googleplusimport@(rqfun@params);
				tmp = raw["items"];
				tmpdata = Join[tmpdata,tmp];
				If[Length[tmpdata]>(start-1),
					formatactivity/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				],
			1,
				If[Length[tmpdata]>(start-1),
					formatactivity/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				]
	]
]

googlepluscookeddata["ActivityData",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,activity,params={}},

	invalidParameters = Select[Keys[args],!MemberQ[{"ActivityID"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"ActivityID"],
		activity = Lookup[args,"ActivityID"];
		If[!StringQ[activity],
			Message[ServiceExecute::nval,"ActivityID","GooglePlus"];
			Throw[$Failed]
		];
		AppendTo[params,"activityID"->ToString@activity],
	Message[ServiceExecute::nparam,"ActivityID"];
	Throw[$Failed]
	];
	
	raw = OAuthClient`rawoauthdata[id,"RawActivity",params];
	data = googleplusimport[raw];
	formatactivity@data
]

googlepluscookeddata["ActivityList",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,error,rqfun,user,limit=20,flag=0,start=1,count,rest,tmp,tmpdata={},params=<|"maxResults"->ToString@$googleplusactivitylistlimit|>,ptoken},

	invalidParameters = Select[Keys[args],!MemberQ[{"UserID","MaxResults",MaxItems,"StartIndex"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"UserID"],
		user = Lookup[args,"UserID"];
		If[!(IntegerQ[user]||StringQ[user]),
			Message[ServiceExecute::nval,"UserID","GooglePlus"];
			Throw[$Failed]
		];
		AppendTo[params,"userID"->ToString@user]
	];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","GooglePlus"];
			Throw[$Failed]
		]
	];

	Which[
		KeyExistsQ[args,MaxItems] && !KeyExistsQ[args,"MaxResults"],
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,"MaxResults"] && !KeyExistsQ[args,MaxItems],
			limit = Lookup[args,"MaxResults"];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,"MaxResults","GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,MaxItems] && KeyExistsQ[args,"MaxResults"],
			Message[ServiceExecute::gplusm];
			Throw[$Failed]
	];

	count = Quotient[limit+start-1,$googleplusactivitylistlimit,1];
	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUserActivities", Normal@#]&)];

	Do[
		raw = googleplusimport@(rqfun@params);
			ptoken = raw["nextPageToken"];
			tmp = raw["items"];
			tmpdata = Join[tmpdata,tmp];
			If[Length[tmp]<$googleplusactivitylistlimit,flag=1;Break[]];
			If[!MissingQ[ptoken],AppendTo[params,"pageToken"->ptoken],flag=1;Break[]]
	,count];

	Switch[flag,
			0,
				raw = googleplusimport@(rqfun@params);
				tmp = raw["items"];
				tmpdata = Join[tmpdata,tmp];
				If[Length[tmpdata]>(start-1),
					formatactivity/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				],
			1,
				If[Length[tmpdata]>(start-1),
					formatactivity/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				]
	]
]

googlepluscookeddata["UserPosts",id_,args_]:=Module[{rawdata, data},
	rawdata=googlepluscookeddata["ActivityList",id,args];
	data=(#["Object"]["Content"]&)/@rawdata;
	fromHTML[data]
]

googlepluscookeddata["UserPostsEventSeries",id_,args_]:=Module[{rawdata, data, times},
	rawdata=googlepluscookeddata["ActivityList",id,args];
	times=Lookup[#,"Published",{}]&/@rawdata;
	data=(#["Object"]["Content"]&)/@rawdata;
	If[data=!={},
		EventSeries[MapThread[{#1,#2}&,{times,data}]],
		Missing[]
	]
]

googlepluscookeddata["UserPostsTimeline",id_,args_]:=Module[{rawdata, data, times},
	rawdata=googlepluscookeddata["ActivityList",id,args];
	times=Lookup[#,"Published",{}]&/@rawdata;
	data=(#["Object"]["Content"]&)/@rawdata;
	DateListPlot[MapThread[Tooltip[{#,1},#2]&,{times,data}],Filling->Axis,FrameTicks -> {None, {Automatic, Automatic}},Joined->False]
]

googlepluscookeddata[prop:("ActivityPlusOners"|"ActivityResharers"),id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,error,rqfun,col,activity,limit=20,flag=0,start=1,count,rest,tmp,tmpdata={},params=<|"maxResults"->ToString@$googleplusactivitylistlimit|>,ptoken},

	invalidParameters = Select[Keys[args],!MemberQ[{"ActivityID","MaxResults",MaxItems,"StartIndex"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];

	col = Switch[prop,"ActivityPlusOners","plusoners","ActivityResharers","resharers"];
	AppendTo[params,"collection"->col];

	If[KeyExistsQ[args,"ActivityID"],
		activity = Lookup[args,"ActivityID"];
		If[!StringQ[activity],
			Message[ServiceExecute::nval,"ActivityID","GooglePlus"];
			Throw[$Failed]
		];
		AppendTo[params,"activityID"->ToString@activity],
	Message[ServiceExecute::nparam,"ActivityID"];
	Throw[$Failed]
	];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","GooglePlus"];
			Throw[$Failed]
		]
	];

	Which[
		KeyExistsQ[args,MaxItems] && !KeyExistsQ[args,"MaxResults"],
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,"MaxResults"] && !KeyExistsQ[args,MaxItems],
			limit = Lookup[args,"MaxResults"];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,"MaxResults","GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,MaxItems] && KeyExistsQ[args,"MaxResults"],
			Message[ServiceExecute::gplusm];
			Throw[$Failed]
	];

	count = Quotient[limit+start-1,$googleplusactivitylistlimit,1];
	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPeopleByActivity", Normal@#]&)];

	Do[
		raw = googleplusimport@(rqfun@params);
			ptoken = raw["nextPageToken"];
			tmp = raw["items"];
			tmpdata = Join[tmpdata,tmp];
			If[Length[tmp]<$googleplusactivitylistlimit,flag=1;Break[]];
			If[!MissingQ[ptoken],AppendTo[params,"pageToken"->ptoken],flag=1;Break[]]
	,count];

	Switch[flag,
			0,
				raw = googleplusimport@(rqfun@params);
				tmp = raw["items"];
				tmpdata = Join[tmpdata,tmp];
				If[Length[tmpdata]>(start-1),
					formatuser/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				],
			1,
				If[Length[tmpdata]>(start-1),
					formatuser/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				]
	]
]

googlepluscookeddata["CircledUsers",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data,error,rqfun,col,limit=100,flag=0,start=1,count,rest,tmp,tmpdata={},params=<|"maxResults"->ToString@$googleplusactivitylistlimit|>,ptoken},

	invalidParameters = Select[Keys[args],!MemberQ[{"OrderBy","MaxResults",MaxItems,"StartIndex"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"GooglePlus"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","GooglePlus"];
			Throw[$Failed]
		]
	];

	Which[
		KeyExistsQ[args,MaxItems] && !KeyExistsQ[args,"MaxResults"],
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,"MaxResults"] && !KeyExistsQ[args,MaxItems],
			limit = Lookup[args,"MaxResults"];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,"MaxResults","GooglePlus"];
				Throw[$Failed]
			],
		KeyExistsQ[args,MaxItems] && KeyExistsQ[args,"MaxResults"],
			Message[ServiceExecute::gplusm];
			Throw[$Failed]
	];

	count = Quotient[limit+start-1,$googleplusactivitylistlimit,1];
	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPeopleByCollection", Normal@#]&)];

	Do[
		raw = googleplusimport@(rqfun@params);
			ptoken = raw["nextPageToken"];
			tmp = raw["items"];
			tmpdata = Join[tmpdata,tmp];
			If[Length[tmp]<$googleplusactivitylistlimit,flag=1;Break[]];
			If[!MissingQ[ptoken],AppendTo[params,"pageToken"->ptoken],flag=1;Break[]]
	,count];

	Switch[flag,
			0,
				raw = googleplusimport@(rqfun@params);
				tmp = raw["items"];
				tmpdata = Join[tmpdata,tmp];
				If[Length[tmpdata]>(start-1),
					formatuser/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				],
			1,
				If[Length[tmpdata]>(start-1),
					formatuser/@Take[tmpdata,{start,UpTo[limit+start-1]}],
					{}
				]
	]
]

(* Send Message *)

googleplussendmessage[___]:=$Failed

(*** Utilities ***)

camelcase[l_List, rest___]:=camelcase[#,rest]&/@l
camelcase[str_String, separators_:{"_"}]:=StringReplace[
 StringReplace[
  StringReplace[str, 
   Thread[separators -> " "]], {WordBoundary ~~ word_ :> 
    ToUpperCase[word]}], {"Id"~~WordBoundary->"ID",WhitespaceCharacter -> "","Url"~~WordBoundary->"URL","Urls"~~WordBoundary->"URLs"}]

fp[fields_,values___]:=With[{n=Length[{values}]},
	Sequence@@Join[MapThread[(#1[#2])&,{Take[fields,n],{values}}],
		Map[#1[]&,Drop[fields,n]]
	]
];

formatuser[data_Association]:= <|"DisplayName"->data["displayName"],"UserID"->data["id"]|>

formatactivity[data_Association]:= Replace[<|"Actor"->data["actor"],"URL"->data["url"],"Updated"->DateObject[DateList[data["updated"]],TimeZone->0],"Object"->data["object"],"Published"->DateObject[DateList[data["published"]],TimeZone->0],"ActivityID"->data["id"]|>, {asoc_Association :> KeyMap[camelcase, asoc]}, Infinity]
			
formatuserid[]:="me"
formatuserid[Automatic]:="me"
formatuserid[id_]:=ToString[id]

tostring[str_String,_]:=str
tostring[default_]:=default
tostring[Automatic,default_]:=default
tostring[str_,_]:=ToString[str]

fromHTML[str_String]:=ImportString[str,"HTML"]
fromHTML[l:{___String}]:=ImportString[#,"HTML"]  &/@l
fromHTML[___]:={}
    
gpicon=Image[RawArray["Byte", {{{255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 19}, {0, 0, 
  0, 34}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, 
  {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 
  38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 
  0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 34}, {0, 0, 0, 19}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 19}, {186, 186, 186, 100}, {251, 251, 251, 234}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {251, 251, 251, 234}, {186, 186, 186, 100}, {0, 0, 0, 19}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 34}, {251, 
  251, 251, 234}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {251, 251, 251, 
  234}, {0, 0, 0, 34}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 
  255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 
  255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, 
  {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {245, 201, 196, 255}, {230, 
  121, 110, 255}, {225, 93, 78, 255}, {222, 80, 63, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, 
  {221, 75, 57, 255}, {223, 85, 68, 255}, {239, 171, 163, 255}, {255, 253, 253, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 
  255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 253, 253, 255}, 
  {233, 135, 124, 255}, {224, 91, 76, 255}, {245, 202, 198, 255}, {255, 253, 253, 255}, {246, 205, 201, 255}, {225, 
  96, 81, 255}, {221, 75, 57, 255}, {222, 82, 66, 255}, {245, 199, 193, 255}, {254, 249, 249, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {234, 143, 133, 255}, {221, 75, 57, 255}, {245, 197, 192, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {251, 229, 227, 255}, {222, 81, 64, 255}, {221, 75, 57, 255}, 
  {224, 92, 77, 255}, {255, 253, 253, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 
  255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {250, 224, 221, 255}, {221, 77, 59, 255}, 
  {222, 79, 62, 255}, {255, 253, 253, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {232, 131, 120, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {241, 182, 175, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {237, 156, 147, 255}, {221, 75, 57, 255}, {222, 79, 63, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {245, 202, 198, 255}, {221, 75, 57, 255}, 
  {221, 75, 57, 255}, {235, 144, 134, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 
  255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {233, 138, 127, 255}, {221, 75, 57, 255}, 
  {221, 75, 57, 255}, {250, 224, 221, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {252, 243, 242, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {236, 153, 143, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {238, 162, 153, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {235, 143, 133, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {222, 78, 61, 255}, 
  {221, 75, 57, 255}, {244, 195, 191, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {236, 151, 141, 255}, {225, 95, 80, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 
  255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {251, 235, 233, 255}, {222, 80, 64, 255}, 
  {221, 75, 57, 255}, {223, 84, 67, 255}, {252, 237, 235, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {250, 232, 229, 255}, {221, 75, 57, 255}, {227, 105, 92, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {233, 138, 127, 
  255}, {222, 81, 64, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {241, 182, 175, 255}, {221, 77, 59, 255}, {221, 75, 57, 255}, {226, 101, 87, 
  255}, {251, 229, 226, 255}, {255, 255, 255, 255}, {253, 245, 244, 255}, {230, 118, 106, 255}, {226, 97, 83, 255}, 
  {251, 235, 233, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {233, 138, 127, 255}, {222, 81, 64, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 
  255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {245, 202, 198, 255}, {228, 111, 98, 255}, {222, 83, 66, 255}, {222, 79, 62, 255}, {225, 93, 78, 255}, {222, 78, 
  61, 255}, {224, 88, 73, 255}, {252, 239, 238, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {234, 141, 131, 255}, {224, 88, 73, 255}, {224, 88, 73, 255}, {222, 81, 64, 255}, {221, 
  76, 58, 255}, {224, 88, 73, 255}, {224, 88, 73, 255}, {224, 88, 73, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 
  255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {254, 
  251, 250, 255}, {222, 79, 62, 255}, {221, 75, 57, 255}, {231, 128, 118, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {239, 171, 164, 255}, {231, 126, 115, 255}, 
  {231, 126, 115, 255}, {225, 95, 80, 255}, {221, 77, 59, 255}, {231, 126, 115, 255}, {231, 126, 115, 255}, {231, 
  126, 115, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 
  38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {253, 245, 244, 255}, {221, 76, 60, 255}, {221, 75, 57, 255}, 
  {222, 82, 66, 255}, {248, 214, 210, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {233, 138, 127, 255}, {222, 81, 
  64, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 
  255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {231, 128, 117, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {222, 81, 64, 255}, {242, 188, 182, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {233, 138, 127, 255}, {222, 81, 64, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 
  38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {245, 201, 196, 
  255}, {233, 135, 124, 255}, {227, 105, 92, 255}, {226, 100, 85, 255}, {226, 100, 85, 255}, {223, 84, 67, 255}, 
  {221, 75, 57, 255}, {221, 75, 57, 255}, {221, 77, 59, 255}, {241, 182, 175, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {239, 171, 164, 255}, {230, 121, 109, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 
  255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {248, 
  218, 214, 255}, {226, 103, 90, 255}, {222, 81, 64, 255}, {236, 149, 139, 255}, {248, 214, 210, 255}, {255, 253, 
  253, 255}, {255, 255, 255, 255}, {253, 247, 247, 255}, {233, 133, 123, 255}, {221, 75, 57, 255}, {221, 75, 57, 
  255}, {222, 79, 63, 255}, {250, 232, 229, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 
  38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {248, 215, 211, 255}, {222, 81, 64, 255}, {222, 79, 63, 255}, {248, 217, 213, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {239, 166, 158, 255}, {221, 76, 58, 255}, {221, 75, 57, 255}, {239, 165, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 
  255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {229, 114, 102, 255}, 
  {221, 75, 57, 255}, {231, 126, 115, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {231, 123, 
  111, 255}, {221, 75, 57, 255}, {235, 143, 133, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 
  0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {222, 79, 62, 255}, {221, 75, 57, 255}, {235, 143, 133, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {238, 169, 162, 255}, {221, 75, 57, 255}, {238, 162, 153, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 
  255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {223, 84, 67, 255}, 
  {221, 75, 57, 255}, {226, 100, 86, 255}, {255, 253, 253, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {235, 146, 
  136, 255}, {222, 79, 63, 255}, {250, 232, 231, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 
  0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {233, 138, 127, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {233, 137, 126, 
  255}, {254, 251, 250, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {250, 221, 219, 255}, {223, 84, 68, 255}, {239, 165, 157, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 
  255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {254, 249, 249, 255}, 
  {229, 116, 105, 255}, {221, 75, 57, 255}, {221, 75, 57, 255}, {224, 91, 76, 255}, {236, 149, 139, 255}, {242, 185, 
  179, 255}, {245, 202, 198, 255}, {241, 182, 176, 255}, {233, 137, 126, 255}, {223, 86, 70, 255}, {238, 168, 161, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 
  38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {241, 182, 175, 255}, {229, 114, 102, 
  255}, {223, 87, 71, 255}, {222, 78, 61, 255}, {222, 78, 61, 255}, {223, 85, 69, 255}, {227, 107, 95, 255}, {236, 
  153, 143, 255}, {251, 235, 233, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 
  255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 
  38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 
  255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 38}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {0, 
  0, 0, 38}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 34}, {251, 251, 251, 234}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {251, 251, 251, 235}, {0, 0, 0, 34}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 
  {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 19}, {187, 187, 
  187, 101}, {251, 251, 251, 235}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {251, 251, 251, 232}, {187, 187, 187, 101}, 
  {0, 0, 0, 19}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 25}, {0, 0, 0, 35}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 
  38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 
  0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, 
  {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 
  38}, {0, 0, 0, 38}, {0, 0, 0, 38}, {0, 0, 0, 35}, {0, 0, 0, 24}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 4}, {0, 0, 0, 12}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, 
  {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 
  13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 
  0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 13}, {0, 0, 0, 12}, 
  {0, 0, 0, 4}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, 
  {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 
  255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}, {{255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {255, 255, 255, 0}}}], "Byte", ColorSpace -> "RGB", 
 Interleaving -> True];

End[] (* End Private Context *)
           		
End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{GooglePlusOAuth`Private`googleplusdata,GooglePlusOAuth`Private`googlepluscookeddata,GooglePlusOAuth`Private`googleplussendmessage}
