Begin["InstagramOAuth`"] (* Begin Private Context *)

ServiceConnect::disc = "Due to restrictions added by Instagram, this service is currently not available."

Begin["`Private`"](* Begin Private Context *)

(******************************* Instagram *************************************)

instagramdata[___]:=(
ServiceConnections`Private`$oauthservices=DeleteCases[ServiceConnections`Private`$oauthservices,"Instagram"];
Message[ServiceConnect::disc,"Instagram"];Throw[$Failed])

instagramcookeddata[___]:=(Message[ServiceConnect::disc,"Instagram"];Throw[$Failed])

instagramsendmessage[___]:=(Message[ServiceConnect::disc,"Instagram"];Throw[$Failed])

instagramdata[]=
    If[TrueQ[OAuthClient`Private`$AllowNonBlockingDialogsQ],{
		"OAuthVersion"		-> "2.0",
		"ServiceName" 		-> "Instagram",
	 	"AuthorizeEndpoint" -> "https://api.instagram.com/oauth/authorize/",
     	"AccessEndpoint"    -> "https://api.instagram.com/oauth/access_token",
     	"RedirectURI"       -> "WolframConnectorChannelListen",
        "Blocking"           ->False,
        "VerifierLabel"     -> "code",
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog" :> "WolframConnectorChannel",
	 	"Scope"				-> {"basic+relationships+comments+likes"}, (* a common field in OAuth 2.0 that specifies the permissions an app will receive during authentication *)
	 	"Gets"				-> {"UserID","LatestPhotos","CaptionedLatestPhotos",
	 								"Caption","CommentCount","CommentIDs","Comments","CommentAuthors","CreationDate",
	 								"FollowerIDs","Followers","FollowerCount","Followings","FollowingIDs","FollowingCount",
	 								"LatestMedia", "LatestMediaIDs","LikeCount", "Liked", "LikedIDs", "LikeIDs", "Likes","Link","LatestVideos",
	 								"Location","Media","MediaID", "MediaIDs","MediaSearch","Owner","OwnerID","Picture","PopularPhotos","PopularVideos","CaptionedPopularPhotos",
	 								"PopularMedia","PopularMediaURLs","PopularMediaIDs","Type","UserData","UserSearch","TagSearch",OAuthClient`Private`gridRequests["TagSearchGrid"],
	 								"TaggedMedia","TaggedMediaURLs","TaggedMediaIDs"}, (* cooked HTTP Get requests *)
	 	"Posts"				-> {},(* cooked HTTP Post requests *)
	 	"RawGets"			-> {"RawPopularPhotos","RawUserData","RawUserFeed","RawRecentMedia","RawRecentLikedMedia",
	 								"RawFollowings","RawFollowers","RawRelationship","RawMediaInfo","RawMediaComments",
	 								"RawPostMediaComments","RawMediaLikes","RawLikeMedia","RawMediaSearch","RawTagData",
	 								"RawRecentTaggedMedia","RawLocationData","RawLatLongLocations","RawRecentLocationMedia","RawUserSearch","RawTagSearch"},(* raw HTTP Get requests *)
	 	"RawPosts"			-> {},(* raw HTTP Post requests *)
 		"Information"		-> "A service for exchanging data with an Instagram account"
    }
    ,
    {
        "OAuthVersion"      ->"2.0",
        "ServiceName"       -> "Instagram",
        "AuthorizeEndpoint" -> "https://api.instagram.com/oauth/authorize/",
        "AccessEndpoint"    -> "https://api.instagram.com/oauth/access_token",
        "RedirectURI"       -> "https://www.wolfram.com/oauthlanding?service=Instagram",
        "VerifierLabel"     -> "code", (* specified how the authentication is returned in the callback url *)
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "Instagram",insticon]&), (* specifies the interface for receiving the authentication information *)
        "Scope"             -> {"basic+relationships+comments+likes"}, (* a common field in OAuth 2.0 that specifies the permissions an app will receive during authentication *)
        "Gets"              -> {"UserID","LatestPhotos","CaptionedLatestPhotos",
                                    "Caption","CommentCount","CommentIDs","Comments","CommentAuthors","CreationDate",
                                    "FollowerIDs","Followers","FollowerCount","Followings","FollowingIDs","FollowingCount",
                                    "LatestMedia", "LatestMediaIDs","LikeCount", "Liked", "LikedIDs", "LikeIDs", "Likes","Link","LatestVideos",
                                    "Location","Media","MediaID", "MediaIDs","MediaSearch","Owner","OwnerID","Picture","PopularPhotos","PopularVideos","CaptionedPopularPhotos",
                                    "PopularMedia","PopularMediaURLs","PopularMediaIDs","Type","UserData","UserSearch","TagSearch",OAuthClient`Private`gridRequests["TagSearchGrid"],
                                    "TaggedMedia","TaggedMediaURLs","TaggedMediaIDs"}, (* cooked HTTP Get requests *)
        "Posts"             -> {},(* cooked HTTP Post requests *)
        "RawGets"           -> {"RawPopularPhotos","RawUserData","RawUserFeed","RawRecentMedia","RawRecentLikedMedia",
                                    "RawFollowings","RawFollowers","RawRelationship","RawMediaInfo","RawMediaComments",
                                    "RawPostMediaComments","RawMediaLikes","RawLikeMedia","RawMediaSearch","RawTagData",
                                    "RawRecentTaggedMedia","RawLocationData","RawLatLongLocations","RawRecentLocationMedia","RawUserSearch","RawTagSearch"},(* raw HTTP Get requests *)
        "RawPosts"          -> {},(* raw HTTP Post requests *)
        "Information"       -> "A service for exchanging data with an Instagram account"
    }
    ]

(* a function for importing the raw data - usually json or xml - from the service *)
instagramimport[$Failed]:=Throw[$Failed]
instagramimport[rawdata_]:=With[
  {raw=ImportString[FromCharacterCode[rawdata],"RawJSON"]},
	If[("code"/.raw["meta"])===200,
    raw["data"],
    Message[ServiceExecute::apierr,"error_message"/.raw["meta"]];
    Throw[$Failed]
  ]
  (*If[FreeQ[res,_["errors",_]],
		Switch[res,
			_Rule|{_Rule...},Association@res,
			{{_Rule...}...},Association/@res,
			_,res
		],
		Message[ServiceExecute::apierr,"message"/.("errors"/.res)];
		Throw[$Failed]
	]*)
]


(****** Raw Properties ******)
(* information:
 Each entry includes the api endpoint, the HTTP method ("GET" or "POST") as well as the different types of parameters that are used
 "Parameters" - standard URL parameters that appear in the query string as ?param1=val1&param2=val2...
 "PathParameters" - parameters that are included in the url path scheme://domain/path1/`1`/`2`...  make the URL (ToString[StringForm[...,#1,#2]]&)
 "BodyData" - parameters in HTTP Post requests that are includes as body data
 "MultipartData" - parameters in HTTP Post requests that are includes as multip part data,
 		usually large files or images are multipart data, each parameter should be given as {"parametername","datatype"}
 "RequiredParameters" - all above parameters are assumed to be optional, list required parameters a second time here
 "Headers" - additional headers to be included in the HTTP request
 "RequiredPermissions" - If we support incrementally adding permission for a service, list the required permissions for the request here*)

instagramdata["RawPopularPhotos"] = {
        "URL"				->  "https://api.instagram.com/v1/media/popular",
        "Parameters" 		-> {},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport,
        "ReturnContentData"	-> True,
        "RequiredPermissions"-> {}
    }

(** User **)
instagramdata["RawUserData"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/users/`1`",pparam[##]]]&),
        "PathParameters" 		-> {"UserID"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawUserFeed"] = {
        "URL"				->  "https://api.instagram.com/v1/users/self/feed",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"count","MIN_ID","MAX_ID"},
        "ResultsFunction"	-> instagramimport,
        "RequiredPermissions"-> {}
    }

instagramdata["RawUserSearch"] = {
        "URL"				->  "https://api.instagram.com/v1/users/search",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"q","count"},
        "RequiredParameters"-> {"q"},
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport,
        "RequiredPermissions"-> {}
    }

instagramdata["RawRecentMedia"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/users/`1`/media/recent",pparam[##]]]&),
        "PathParameters" 	-> {"UserID"},
        "Parameters"		-> {"count","max_timestamp","min_timestamp","min_id","max_id"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawRecentLikedMedia"] = {
        "URL"				->  "https://api.instagram.com/v1/users/self/media/liked",
        "Parameters"		-> {"count","max_like_id"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

(** Relationships **)

instagramdata["RawFollowings"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/users/`1`/follows",pparam[##]]]&),
        "PathParameters" 		-> {"UserID"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawFollowers"] = {
        "URL"				-> (ToString[StringForm["https://api.instagram.com/v1/users/`1`/followed-by",pparam[##]]]&),
        "PathParameters" 		-> {"UserID"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawRelationship"] = {
        "URL"				-> (ToString[StringForm["https://api.instagram.com/v1/users/`1`/relationship",pparam[##]]]&),
        "PathParameters" 		-> {"UserID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

(** Media **)

instagramdata["RawMediaInfo"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/media/`1`",pparam[##]]]&),
        "PathParameters" 	-> {"MediaID"},
        "RequiredParameters"-> {"MediaID"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawMediaComments"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/media/`1`/comments",pparam[##]]]&),
        "PathParameters" 		-> {"MediaID"},
        "RequiredParameters"-> {"MediaID"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawPostMediaComments"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/media/`1`/comments",pparam[##]]]&),
        "PathParameters" 		-> {"MediaID"},
        "Parameters"		-> {"text"},
        "RequiredParameters"-> {"MediaID","text"},
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawMediaLikes"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/media/`1`/likes",pparam[##]]]&),
        "PathParameters" 		-> {"MediaID"},
        "RequiredParameters"-> {"MediaID"},
        "ReturnContentData"	-> True,
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawLikeMedia"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/media/`1`/likes",pparam[##]]]&),
        "PathParameters" 		-> {"MediaID"},
        "RequiredParameters"-> {"MediaID"},
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> instagramimport
    }
instagramdata["RawMediaSearch"] = {
        "URL"				->  "https://api.instagram.com/v1/media/search",
        "Parameters"		-> {"lat","lng","distance"},
        "RequiredParameters"-> {"lat","lng"},
        "HTTPSMethod"		-> "GET",
        "ReturnContentData"	-> True,
        "ResultsFunction"	-> instagramimport
    }

(** Tags **)
instagramdata["RawTagData"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/tags/`1`",pparam[##]]]&),
        "PathParameters" 		-> {"Tag"},
        "RequiredParameters"-> {"Tag"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawRecentTaggedMedia"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/tags/`1`/media/recent",pparam[##]]]&),
        "PathParameters" 	-> {"Tag"},
        "Parameters"		-> {"count","min_tag_id","max_tag_id"},
        "RequiredParameters"-> {"Tag"},
        "ReturnContentData"	-> True,
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawTagSearch"] = {
        "URL"				->  "https://api.instagram.com/v1/tags/search",
        "Parameters" 		-> {"q"},
        "ReturnContentData"	-> True,
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

(** Locations **)
instagramdata["RawLocationData"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/locations/`1`",pparam[##]]]&),
        "PathParameters" 		-> {"Location"},
        "RequiredParameters"-> {"Location"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawLatLongLocations"] = {
        "URL"				->  "https://api.instagram.com/v1/locations/search",
        "Parameters" 		-> {"lat","lng","distance"},
        "RequiredParameters"-> {"lat","lng"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }

instagramdata["RawRecentLocationMedia"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/locations/`1`/media/recent",pparam[##]]]&),
        "PathParameters" 		-> {"Location"},
        "RequiredParameters"-> {"Location"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }
    (*
    Requires setting up predefined geographies associated with our app
instagramdata["RawRecentGeographyMedia"] = {
        "URL"				->  (ToString[StringForm["https://api.instagram.com/v1/geographies/`1`/media/recent",pparam[##]]]&),
        "PathParameters" 		-> {"Location"},
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> instagramimport
    }
    *)
instagramdata["icon"]:=insticon

(*instagramdata[___]:=$Failed*)
(****** Cooked Properties ******)

(* cooked data queries
	Cooked queries call at least one of the raw queried underneath but add pre and post processing to make the function easier to use and the result clearer.
*)


(* SocialMediaData["Instagram", "Properties"]

{"Caption", "CommentAuthors", "CommentCount", "CommentIDs",
"Comments", "CreationDate", "FollowerIDs", "Followers",
"FollowingIDs", "Followings", "Image", "LatestMedia",
"LatestMediaIDs", "LikeCount", "Liked", "LikedIDs", "LikeIDs",
"Likes", "Link", "Location", "LowResolutionImage", "Media",
"MediaID", "MediaIDs", "Owner", "OwnerID", "Picture", "PopularMedia",
"PopularMediaIDs", "Thumbnail", "Type", "UserData"}

{"Type", "UserData"}

*)
(* User *)

$InstagramPhotoImportCount=10;
$InstagramVideoImportDefault="ImageLink"

instagramcookeddata[prop_,id_,rules___Rule]:=instagramcookeddata[prop,id,{rules}]
instagramcookeddata[prop_,id_,rule_Rule, rest___]:=instagramcookeddata[prop,id,{rule}, rest]
instagramcookeddata[prop_,id_]:=instagramcookeddata[prop,id,{}]


instagramcookeddata[prop:("UserID"|"Picture"|"UserData"),id_,args_]:=Module[
	{rawdata,as, keys, postproc,data,resolution},
	rawdata=OAuthClient`rawoauthdata[id,"RawUserData",args];
	resolution=getquality[{args},"Low"];
	as=instagramimport[rawdata];
	postproc=Switch[prop,
		"UserID",as["id"],
		"Picture",importphoto[as["profile_picture"]],
		"UserData",KeyMap[# /. {"username" -> "Username", "full_name" -> "FullName","media" -> "Media", "followed_by" -> "FollowedBy","follows" -> "Follows", "id" -> "ID"} &,
              Join[KeyTake[as, {"username", "full_name"}], as["counts"],KeyTake[as, "id"]]],
		_,Identity
	];
  postproc
]


instagramcookeddata["UserSearch",id_,args_]:=Module[
	{rawdata,as, data,params, names, ids},
	params=filterparameters[(args/.{"Query"->"q",MaxItems->"count"}),getallparameters["RawUserSearch"]];
  params=params/.{"count"->x_:>"count"->FromDigits[x]};
	rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawUserSearch",params]];
  Association[{"Username"->#1[[1]],"UserID"->#1[[2]]}]&/@({"username","id"}/.rawdata)
]

instagramcookeddata[prop:("TagSearch"|"TagSearchGrid"),id_,args_]:=Module[
	{rawdata,as, data,params, names, counts},
	params=filterparameters[(args/."Query"->"q"),getallparameters["RawTagSearch"]];
	rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawTagSearch",params]];
  Association[{"Name"->#1[[1]],"MediaCount"->#1[[2]]}]&/@({"name","media_count"}/.rawdata)
  (*
	data=StringCases[rawdata, "\"data\":[" ~~ Longest[x__] ~~ "]" :> x];

	If[Length[data]=!=1,Return[{}]];
	counts=StringCases[data[[1]], "\"media_count\":" ~~ Shortest[x__] ~~ "," :> ToExpression[x]];
	names = StringCases[data[[1]], "\"name\":\"" ~~ Shortest[x__] ~~ "\"" :> x];
	If[Length[counts]=!=Length[names],Return[{}]];
	Switch[prop,
		"TagSearch",
		MapThread[Association[{"Name"->#1,"MediaCount"->#2}]&,{names, counts}],
		"TagSearchGrid",
		OAuthClient`Private`addtitles[OAuthClient`Private`prettygrid[Transpose[{names,counts}]],{"Name","MediaCount"}]

	]*)
]

(* Media *)
instagramcookeddata[prop:("PopularMediaURLs"|"PopularMediaIDs"),id_,args_]:=Module[
	(*{rawdata,as, resolution,keys,postproc,data, videoform,urls,allids},*)
  {rawdata, resolution,data, videoform,urls,distance,params=If[MatchQ[args,_Rule],Association[List[args]],Association[args]],newParams,invalidParameters,count},
  invalidParameters = Select[Keys@params,!MemberQ[{"Tag",MaxItems,"MaxItems","MediaResolution"},#]&];
  If[Length[invalidParameters]>0,
  Message[ServiceObject::noget,#,"Instagram"]&/@invalidParameters;
  Throw[$Failed]
  ];
  If[ !KeyExistsQ[params,"Elements"],
    params=Join[params,<|"Elements"->Default|>]
  ];
  If[ !KeyExistsQ[params,"MediaResolution"],
    params=Join[params,<|"MediaResolution"->"standard_resolution"|>],
    (params=Map[#/.{"Standard"->"standard_resolution","Low"->"low_resolution"}&,params];
      If[StringMatchQ[params["MediaResolution"],"High"],
      (
        Message[ServiceExecute::nval,"MediaResolution","Instagram"];
        Throw[$Failed]
      )])
  ];
  If[ Xnor[!KeyExistsQ[params,"MaxItems"],!KeyExistsQ[params,MaxItems]],
      newParams = Join[params,<|"count"->"20"|>],
      (If[And[IntegerQ[params[MaxItems]],Positive[params[MaxItems]]],
        newParams = params,
        (Message[ServiceExecute::nval,MaxItems,"Instagram"];
        Throw[$Failed])
        ]
      )
  ];
  newParams = KeyMap[# /. MaxItems | "MaxItems" -> "count" &, newParams];
  newParams["count"]=ToString[newParams["count"]];
  count=FromDigits[newParams["count"]];
  count=If[count > 30, 30, count];

  rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawPopularPhotos",Normal[newParams]]];

  ICookedImport[rawdata,prop,count,newParams["MediaResolution"],newParams["Elements"]]
  (*
	rawdata=OAuthClient`rawoauthdata[id,"RawPopularPhotos",args];

	resolution=getquality[{args},"Low"];
  *)

	(***** workaround for JSON Import Bug  255746 ********

	as=instagramimport[rawdata];
	keys=Switch[prop,
		"PopularMedia",{},
		"PopularMediaURLs",{"images",resolution,"url"},
		"PopularMediaIDs",{"id"}
	];
	postproc=Switch[prop,
		"PopularMedia",(getphotos[#,resolution]&),
		"PopularMediaURLs",(Hyperlink/@#&),
		"PopularMediaIDs",ToExpression,
		_,Identity
	];
	If[KeyExistsQ[as,"data"],
		data=as["data"];
		postproc@gdata[data,keys]
		,{}
	]
	*)
  (*
	videoform=("VideoImport"/.args)/."VideoImport"->$InstagramVideoImportDefault;

	data=StringCases[rawdata, "\"data\":[" ~~ Longest[x__] ~~ "]" :> x];
	If[Length[data]=!=1,Return[{}]];
	data=First[data];
	Switch[prop,
		"PopularMedia",getmediajson[data,resolution, videoform],
		"PopularMediaURLs",urls=Flatten[StringCases[data,  "\"" <> resolution <> "\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}") :> x]];
			If[!MatchQ[urls,{_String..}],Return[{}]];
			urls="url" /. (ImportString[#, "JSON"] & /@urls);
			If[!MatchQ[urls,{_String..}],Return[{}]];
			Hyperlink/@urls,
		"PopularMediaIDs",
			ToExpression@getids[data,"Media"]
	]
  *)
]

instagramcookeddata[prop:("TaggedMediaURLs"|"TaggedMediaIDs"),id_,args_]:=Module[
  {rawdata, resolution,data, videoform,urls,distance,params=If[MatchQ[args,_Rule],Association[List[args]],Association[args]],newParams,invalidParameters,count},
  invalidParameters = Select[Keys@params,!MemberQ[{"Tag",MaxItems,"MaxItems","MediaResolution"},#]&];
  If[Length[invalidParameters]>0,
  Message[ServiceObject::noget,#,"Instagram"]&/@invalidParameters;
  Throw[$Failed]
  ];
  If[ !KeyExistsQ[params,"Elements"],
    params=Join[params,<|"Elements"->Default|>]
  ];
  If[ !KeyExistsQ[params,"MediaResolution"],
    params=Join[params,<|"MediaResolution"->"standard_resolution"|>],
    (params=Map[#/.{"Standard"->"standard_resolution","Low"->"low_resolution"}&,params];
      If[StringMatchQ[params["MediaResolution"],"High"],
      (
        Message[ServiceExecute::nval,"MediaResolution","Instagram"];
        Throw[$Failed]
      )])
  ];
  If[ Xnor[!KeyExistsQ[params,"MaxItems"],!KeyExistsQ[params,MaxItems]],
      newParams = Join[params,<|"count"->"20"|>],
      newParams = params;
  ];
  newParams = KeyMap[# /. MaxItems | "MaxItems" -> "count" &, newParams];
  newParams["count"]=ToString[newParams["count"]];
  count=FromDigits[newParams["count"]];
  count=If[count > 30, 30, count];
  If[StringMatchQ[newParams["Tag"],"#*"],
    newParams["Tag"]= StringDrop[newParams["Tag"], 1]
  ];
  rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawRecentTaggedMedia",Normal[newParams]]];

  ICookedImport[rawdata,prop,count,newParams["MediaResolution"],newParams["Elements"]]

  (*
	resolution=getquality[{args},"Low"];

	videoform=("VideoImport"/.args)/."VideoImport"->$InstagramVideoImportDefault;

	data=StringCases[rawdata, "\"data\":[" ~~ Longest[x__] ~~ "]" :> x];
	If[Length[data]=!=1,Return[{}]];
	data=First[data];
	Switch[prop,
		"TaggedMedia",getmediajson[data,resolution, videoform],
		"TaggedMediaURLs",urls=Flatten[StringCases[data,  "\"" <> resolution <> "\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}") :> x]];
			If[!MatchQ[urls,{_String..}],Return[{}]];
			urls="url" /. (ImportString[#, "JSON"] & /@urls);
			If[!MatchQ[urls,{_String..}],Return[{}]];
			Hyperlink/@urls,
		"TaggedMediaIDs",
			ToExpression@getids[data,"Media"]
	]*)
]
(*
instagramcookeddata[prop:("PopularPhotos"|"CaptionedPopularPhotos"|"PopularVideos"),id_,args_]:=Module[
	{rawdata,as, media, captions, data,resolution, videoform,splitdata,types},
	videoform=If[prop==="PopularVideos",{"OnlyVideo",("VideoImport"/.args)}/."VideoImport"->$InstagramVideoImportDefault,None];
	rawdata=FromCharacterCode[OAuthClient`rawoauthdata[id,"RawPopularPhotos",args]];
	resolution=getquality[{args},"Low"];
*)
	(***** workaround for JSON Import Bug  255746 ********

	as=instagramimport[rawdata];
	If[KeyExistsQ[as,"data"],
		data=as["data"];
		If[Length[data]>0,
			images=getphotos[data,resolution,"image"];
			If[prop==="CaptionedLatestPhotos",
				captions="text"/.("caption"/.data);
				MapThread[Labeled[#1,#2]&,{images,captions}],
				images
			],{}
		],{}
	]
	*)
(*
	data=StringCases[rawdata, "\"data\":[" ~~ Longest[x__] ~~ "]" :> x];
	If[Length[data]=!=1,Return[{}]];
	media=getmediajson[data[[1]],resolution, videoform];
	If[prop==="CaptionedPopularPhotos",
		getmediajson[data[[1]],resolution, videoform,True],
		getmediajson[data[[1]],resolution, videoform]
	]

]*)
instagramcookeddata[prop:("MediaSearch"|"TaggedMedia"|"PopularMedia"),id_,args_]:=Module[
	{distance,params=If[MatchQ[args,_Rule],Association[List[args]],Association[args]],newParams,invalidParameters,count},
  invalidParameters = Select[Keys@params,!MemberQ[{"Tag","Location",MaxItems,"Distance","MaxItems","Latitude","Longitude","Elements","MediaResolution"},#]&];
  If[Length[invalidParameters]>0,
  Message[ServiceObject::noget,#,"Instagram"]&/@invalidParameters;
  Throw[$Failed]
  ];
  If[ !KeyExistsQ[params,"Elements"],
    params=Join[params,<|"Elements"->Default|>]
  ];
  If[ !KeyExistsQ[params,"Distance"],
    params=Join[params,<|"Distance"->"1000 m"|>]
  ];
  If[ !KeyExistsQ[params,"MediaResolution"],
    params=Join[params,<|"MediaResolution"->"standard_resolution"|>],
    (params=Map[#/.{"Standard"->"standard_resolution","Low"->"low_resolution"}&,params];
      If[StringMatchQ[params["MediaResolution"],"High"],
      (
        Message[ServiceExecute::nval,"MediaResolution","Instagram"];
        Throw[$Failed]
      )])
  ];
  If[ Xnor[!KeyExistsQ[params,"MaxItems"],!KeyExistsQ[params,MaxItems]],
      newParams = Join[params,<|"count"->"20"|>],
      newParams = params;
  ];
  newParams = KeyMap[# /. MaxItems | "MaxItems" -> "count" &, newParams];
  newParams["count"]=ToString[newParams["count"]];
  count=FromDigits[newParams["count"]];
  count=If[count > 30, 30, count];
  If[!StringMatchQ[prop,"TaggedMedia"],(
    distance=newParams["Distance"];
    If[KeyExistsQ[newParams,"Location"],
      Switch[Head[newParams["Location"]],
        Entity,
          newParams = Join[newParams,<|"Latitude"->QuantityMagnitude[Latitude[newParams["Location"]]]|>];
          newParams = Join[newParams,<|"Longitude"->QuantityMagnitude[Longitude[newParams["Location"]]]|>],
        GeoPosition,
          newParams = Join[newParams,<|"Latitude"->QuantityMagnitude[Latitude[newParams["Location"]]]|>];
          newParams = Join[newParams,<|"Longitude"->QuantityMagnitude[Longitude[newParams["Location"]]]|>],
        List,
          center=Mean[First /@ newParams["Location"]];
          newParams = Join[newParams,<|"Latitude"->center[[1]]|>];
          newParams = Join[newParams,<|"Longitude"->center[[2]]|>],
        GeoDisk,
          Switch[newParams["Location"],
            GeoDisk[],
              center = $GeoLocation;
              distance = "1000 m",
            GeoDisk[_],
              center = newParams["Location"][[1]];
              distance = "1000 m",
            GeoDisk[_,_,___],
              center = newParams["Location"][[1]];
              distance = newParams["Location"][[2]]
            ];
            newParams = Join[newParams,<|"Latitude"->QuantityMagnitude@Latitude[center]|>];
            newParams = Join[newParams,<|"Longitude"->QuantityMagnitude@Longitude[center]|>];
            newParams = Join[newParams,<|"Distance"->QuantityMagnitude@distance|>];
        ]
    ];
    If[KeyExistsQ[newParams,"Distance"],
      (distance=Quiet[Round@QuantityMagnitude@UnitConvert[distance,"Meters"]];
        If[And[Positive[distance],!MatchQ[distance,Round[___]]],
          newParams["Distance"]=distance,
          Message[ServiceExecute::nval,"Distance","Instagram"];
          Throw[$Failed]
        ])
    ];)
  ];
  newParams = KeyMap[# /. {MaxItems | "MaxItems" -> "count","Latitude"->"lat","Longitude"->"lng","Distance"->"distance"} &, newParams];
  newParams["count"]=ToString[newParams["count"]];
  newParams["lat"]=ToString[newParams["lat"]];
  newParams["lng"]=ToString[newParams["lng"]];
  newParams["distance"]=ToString[newParams["distance"]];
  If[!StringMatchQ[prop,"TaggedMedia"],
    If[StringMatchQ[prop,"MediaSearch"],
      rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawMediaSearch",Normal[newParams]]],
      rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawPopularPhotos",Normal[newParams]]]
    ],
    (If[StringMatchQ[newParams["Tag"],"#*"],
      newParams["Tag"]= StringDrop[newParams["Tag"], 1]
    ];
    rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawRecentTaggedMedia",Normal[newParams]]])
  ];
  If[MatchQ[rawdata,{}],
    Dataset[Association[]],
    ICookedImport[rawdata,"MediaSearch",count,newParams["MediaResolution"],newParams["Elements"]]
  ]
  (*rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawMediaSearch",Normal[newParams]]]*)
]
instagramcookeddata[prop:("LatestPhotos"|"CaptionedLatestPhotos"|"LatestVideos"|"LatestMedia"|"LatestMediaIDs"|"MediaIDs"|"PopularPhotos"|"CaptionedPopularPhotos"|"PopularVideos"|"Liked"|"LikedIDs"),id_,args_]:= Module[
	{rawdata, as, media, captions,count, data,resolution, videoform,newParams,params=If[MatchQ[args,_Rule],Association[List[args]],Association[args]]},
	videoform=If[prop==="LatestVideos",{"OnlyVideo",("VideoImport"/.args)}/."VideoImport"->$InstagramVideoImportDefault,None];
  invalidParameters = Select[Keys@params,!MemberQ[{"UserID",MaxItems,"MaxItems","Elements","MediaResolution"},#]&];

  If[Or[!Positive[params[MaxItems]],!Positive[params["MaxItems"]]],
  (
    Message[ServiceExecute::nval,"MaxItems","Instagram"];
    Throw[$Failed]
  )];
  If[ !KeyExistsQ[params,"Elements"],
    params=Join[params,<|"Elements"->Default|>]
  ];
  If[ !KeyExistsQ[params,"MediaResolution"],
    params=Join[params,<|"MediaResolution"->"standard_resolution"|>],
    (params=Map[#/.{"Standard"->"standard_resolution","Low"->"low_resolution"}&,params];
      If[StringMatchQ[params["MediaResolution"],"High"],
      (
        Message[ServiceExecute::nval,"MediaResolution","Instagram"];
        Throw[$Failed]
      )])
  ];
  If[ Xnor[!KeyExistsQ[params,"MaxItems"],!KeyExistsQ[params,MaxItems]],
      newParams = Join[params,<|"count"->"20"|>],
      newParams = params;
  ];

  newParams = KeyMap[# /. MaxItems | "MaxItems" -> "count" &, newParams];
  newParams["count"]=ToString[newParams["count"]];
  count=FromDigits[newParams["count"]];
  count=If[count > 30, 30, count];
  If[Length[invalidParameters]>0,
  Message[ServiceObject::noget,#,"Instagram"]&/@invalidParameters;
  Throw[$Failed]
  ];
  If[StringMatchQ[prop,"PopularPhotos"|"CaptionedPopularPhotos"|"PopularVideos"],
    rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawPopularPhotos",Normal[newParams]]],
    If[StringMatchQ[prop,"Liked"|"LikedIDs"],
      rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawRecentLikedMedia",Normal[newParams]]],
      rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawRecentMedia",Normal[newParams]]]]
  ];

	enoughQ=If[MatchQ[prop,"LatestVideos"],Count["type"/.rawdata, "video"] >= FromDigits[newParams["count"]],Count["type"/.rawdata, "image"] >= FromDigits[newParams["count"]]];
  If[!enoughQ,
    (
    newParams["count"] = ToString[FromDigits[newParams["count"]]+20];
    If[StringMatchQ[prop,"PopularPhotos"|"CaptionedPopularPhotos"|"PopularVideos"],
      rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawPopularPhotos",Normal[newParams]]],
      If[StringMatchQ[prop,"Liked"|"LikedIDs"],
        rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawRecentLikedMedia",Normal[newParams]]],
        rawdata = instagramimport[OAuthClient`rawoauthdata[id,"RawRecentMedia",Normal[newParams]]]]
    ];
    enoughQ=If[MatchQ[prop,"LatestVideos"],Count["type" /. rawdata, "video"] >= FromDigits[newParams["count"]],Count["type" /. rawdata, "image"] >= FromDigits[newParams["count"]]])
  ];
  (*resolution=getquality[Normal[params],"Low"];*)
	(***** workaround for JSON Import Bug  255746 ********

	as=instagramimport[rawdata];
	If[KeyExistsQ[as,"data"],
		data=as["data"];
		If[Length[data]>0,
			images=getphotos[data,resolution,"image"];
			If[prop==="CaptionedLatestPhotos",
				captions="text"/.("caption"/.data);
				MapThread[Labeled[#1,#2]&,{images,captions}],
				images
			],{}
		],{}
	]
	*)
  If[MatchQ[rawdata,{}],
    Dataset[Association[]],
    ICookedImport[rawdata,prop,count,newParams["MediaResolution"],newParams["Elements"]]
  ]

	(*
  data=StringCases[rawdata, "\"data\":[" ~~ Longest[x__] ~~ "]" :> x];
	If[Length[data]=!=1,Return[{}]];

	If[prop==="CaptionedLatestPhotos",
		getmediajson[data[[1]],resolution, videoform,True],
		getmediajson[data[[1]],resolution, videoform]
	]*)
	 (* end workaround *)

]
ICookedImport[res_,request_,count_,mediaResolution_,output_:Default] :=
    Module[ {associationResult},
        associationResult = Switch[request,
            "LatestPhotos", IFormatCooked[res,{"image",output,mediaResolution},count],
            "LatestVideos", IFormatCooked[res,{"video",output,mediaResolution},count],
            "MediaSearch", IFormatCooked[res,{"All",output,mediaResolution},count],
            "CaptionedLatestPhotos",IFormatCooked[res,{"image","CaptionedLatestPhotos",mediaResolution},count],
            "LatestMedia",IFormatCooked[res,{"All",output,mediaResolution},count],
            "LatestMediaIDs",IFormatCooked[res,{"id","MediaIDs",mediaResolution},count],
            "Liked",IFormatCooked[res,{"All",output,mediaResolution},count],
            "LikedIDs",IFormatCooked[res,{"id","MediaIDs",mediaResolution},count],
            "PopularMediaIDs",IFormatCooked[res,{"id","MediaIDs",mediaResolution},count],
            "PopularMediaURLs",IFormatCooked[res,{"id","MediaURLs",mediaResolution},count],
            "PopularPhotos",IFormatCooked[res,{"image",output,mediaResolution},count],
            "PopularVideos",IFormatCooked[res,{"video",output,mediaResolution},count],
            "CaptionedPopularPhotos",IFormatCooked[res,{"image","CaptionedLatestPhotos",mediaResolution},count],
            "TaggedMediaIDs",IFormatCooked[res,{"id","MediaIDs",mediaResolution},count],
            "TaggedMediaURLs",IFormatCooked[res,{"id","MediaURLs",mediaResolution},count]
            ];

    associationResult
    ]
getThumbnail[x_] := Import["url" /. ("thumbnail" /. x)];
getDate[x_] := FromUnixTime[FromDigits[x]];
getCaption[x_] := (If[MatchQ[x, Null], Missing["NotAvailable"], "text" /. x]);

getLikes[x_] := (If[MatchQ[x, Null], Missing["NotAvailable"], "count" /. x]);

getNearestLocation[x_] := (
  If[MatchQ[x, Null],
    Missing["NotAvailable"],
    (
      geon=GeoNearest[Entity["AdministrativeDivision"],GeoPosition[{"latitude" /. x, "longitude" /. x}]];
      If[And[!MatchQ[geon,$Failed],!MatchQ[geon,{}]],
        Last[geon],
        Missing["NotAvailable"]
        ]
    )]
    );

getTaggedUsers[x_] := (If[MatchQ[x, {}], x, "username" /. ("user" /. x)]);

getGeoPosition[x_] := (If[MatchQ[x, Null], Missing["NotAvailable"],
          GeoPosition[{"latitude" /. x, "longitude" /. x}]]);

IFormatCooked[res_,options_,count_] := Module[
  {dataset=Dataset[Association/@res]
    ,countselected,total,captions
    (*photos=Select[res, ("type" /. #) == "image" &],
    videos=Select[res, ("type" /. #) == "video" &]*)},
    Switch[options[[2]],
      "Images",
                Normal[dataset[If[MatchQ[options[[1]], "All"], All, Select[#type == "image" &]]][1 ;; count, {"images"}]
                [All, {"images" -> Association}][All, "images"]
                [All, {options[[3]] -> Association}]
                [All,options[[3]]][All, {"url" -> Import}][All, "url"]],
      "Default"|Default|"Data",
                (datasetTotal=dataset[If[MatchQ[options[[1]], "All"], All, Select[#type == options[[1]] &]]];
                If[Length[datasetTotal]>0,
                  datasetTotal[1 ;; count, <|"Thumbnail" -> "images", "URL" -> "link","Date" -> "created_time"|>]
                  [All, {"Thumbnail" -> getThumbnail, "Date" -> getDate}],
                  Dataset[Association[]]]),
      "FullData",
                dataset[If[MatchQ[options[[1]], "All"], All, Select[#type == options[[1]] &]]]
                [1 ;; count, <|"Thumbnail" -> "images", "URL" -> "link", "Date" -> "created_time",
                "Type" -> "type","NearestLocation" -> "location","GeoPosition" -> "location", "TaggedUsers" -> "users_in_photo",
                "Caption" -> "caption", "Filter" -> "filter", "Likes" -> "likes", "Tags" -> "tags","MediaID" -> "id"|>]
                [All, {"Thumbnail" -> getThumbnail, "Date" -> getDate, "Caption" -> getCaption, "Likes" -> getLikes, "NearestLocation" -> getNearestLocation, "TaggedUsers" -> getTaggedUsers,"GeoPosition" -> getGeoPosition}],
      "LinkedVideos",
                (datasetTotal=dataset[Select[#type == "video" &]];
                If[Length[datasetTotal]>0,
                  Normal[datasetTotal[1 ;; count, {"images", "videos"}]
                  [All, {"images" -> Association, "videos" -> Association}]
                  [All, {"images" -> options[[3]], "videos" -> options[[3]]}]
                  [All, {"images" -> Association, "videos" -> Association}]
                  [All, {"images" -> "url", "videos" -> "url"}]
                  [All, Hyperlink[Import[#images], #videos] &]],
                  Dataset[Association[]]])
                (*[All, All, First][All, {"images" -> Association, "videos" -> Association}]
                [All, All,First][All, Hyperlink[Import[#images], #videos] &]*),
      "LinkedThumbnails",
                Normal[dataset[If[MatchQ[options[[1]], "All"], All, Select[#type == options[[1]] &]]]
                [1 ;; count, {"images", "link"}]
                [All, {"images" -> Association}][All, {"images"->First}]
                [All, {"images"->Association}][All,{"images"->First}][All,Hyperlink[Import[#images],#link]&]],
      "CaptionedLatestPhotos",
                Normal[dataset[Select[#type == options[[1]] &]][1 ;; count, {"images","caption"}]
                [All, {"images" -> options[[3]]}]
                [All, {"images" -> "url", "caption" -> "text"}]
                [All, Labeled[Import[#images], #caption] &]],
      "MediaIDs",
                Normal[dataset[1;;count,"id"]],
      "MediaURLs",
                StringDelete[Normal[dataset[1;;count, {"images" -> Association}][All, "images"]
                [All, {options[[3]] -> Association}]
                [All,options[[3]]][All, "url"]],RegularExpression["\\?ig_cache.+"]],
      __,       (Message[ServiceExecute::nval,"Elements","Instagram"];
                Throw[$Failed])
    ]
]
instagramcookeddata["Media",id_,args_]:=Module[
	{rawdata,as, images, data,resolution,videoform},
	rawdata=OAuthClient`rawoauthdata[id,"RawMediaInfo",args];
  rawdata=FromCharacterCode[rawdata];
	videoform=("VideoImport"/.args)/."VideoImport"->$InstagramVideoImportDefault;
	resolution=getquality[{args},"Low"];
	data=StringCases[rawdata, "\"data\":{" ~~ Longest[x__] ~~ "}" :> x];
	If[Length[data]=!=1,Return[{}]];
	getmediajson[data[[1]],resolution, videoform]

]

mediainfokeys=("Caption"|"CommentCount"|"CommentIDs"|"Comments"|"CommentAuthors"|"CreationDate"|"Link"|"Location"|"MediaID"|"Owner"|"OwnerID"|"Type")
instagramcookeddata[prop:mediainfokeys,id_,args_]:=Module[
	{rawdata,as, data,resolution,keys,postproc},
	rawdata=OAuthClient`rawoauthdata[id,"RawMediaInfo",args];
	as=instagramimport[rawdata];

	postproc=Switch[prop,
		"CommentAuthors",as["comments"]["data"]/.{{}->{},x_:>ReplaceAll["username",ReplaceAll["from",x]]},
    "CommentCount",ToExpression[as["comments"]["count"]],
    "CommentIDs",as["comments"]["data"]/.{{}->{},x_:>Map[ToExpression, ReplaceAll["id",x]]},
    "Caption",as["caption"]/.{Null->Missing["NotAvailable"],x_:>ReplaceAll["text", x]},
    "Comments",as["comments"]["data"] /. {{} -> {}, x_ :> ReplaceAll["text", x]},
		"CreationDate",FromUnixTime[ToExpression[as["created_time"]]],
		"Link",Hyperlink[as["link"]],
		"Location",If[MatchQ[as["location"],Null],
                Missing["NotAvailable"],
                Quiet[Map[Replace[_Missing -> Missing["NotAvailable"]],
                  Normal[Dataset[as["location"]]
                        [<|"LocationID" -> "id", "Name" -> "name","GeoPosition" -> {"latitude","longitude"}|>]
                        [{"GeoPosition" -> (Quiet[Check[GeoPosition[List[#latitude, #longitude]],  Missing["NotAvailable"]]] &)}]
                        ]
                    ]]
                ],
    "Type",as["type"],
    "Owner",as["user"]["username"],
    "OwnerID",as["user"]["id"],
    "MediaID",as["id"],
		_,Identity
	]
]

instagramcookeddata[prop:("FollowerIDs"|"Followers"|"FollowerCount"),id_,args_]:=Module[
	{rawdata,as, data,keys,postproc},
	rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawFollowers",args]];
  If[MatchQ[rawdata,{}],
    If[StringMatchQ[prop,"FollowerCount"],0,rawdata],
    Switch[prop,
    "Followers","username"/.rawdata,
    "FollowerIDs","id"/.rawdata,
    "FollowerCount",Length[rawdata]
    ]
  ]
]

instagramcookeddata[prop:("FollowingIDs"|"Followings"|"FollowingCount"),id_,args_]:=Module[
	{rawdata,as, data,keys,postproc},
	rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawFollowings",args]];
  Switch[prop,
  "FollowingIDs","id"/.rawdata,
  "Followings","username"/.rawdata,
  "FollowingCount",Length[rawdata]
  ]
]

instagramcookeddata[prop:("LikeCount"|"LikeIDs"|"Likes"),id_,args_]:=Module[
	{rawdata,as, data,keys,postproc},
	rawdata=instagramimport[OAuthClient`rawoauthdata[id,"RawMediaLikes",args]];

  Switch[prop,
		"LikeCount",Length[rawdata],
		"LikeIDs","id"/.rawdata,
		"Likes","username"/.rawdata
    ]
]


(*instagramcookeddata[___]:=$Failed*)

(******** Send message ****************)
(*instagramsendmessage[___]:=$Failed*)

(******** Service specific utilities ***********)
gdata=OAuthClient`Private`getdata;
filterparameters=OAuthClient`Private`filterParameters;

camelcase=OAuthClient`Private`camelCase;

getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.instagramdata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

getphotos[{},_]:={}
getphotos[data_List,resolution_]:=Module[{images,urls},
	images=("images"/.data);
	If[images==="images"|{"images"},Return[{}]];
	urls="url"/.(resolution/.images);
	importphoto/@urls
]


getmediajson[data_String,resolution_,videoform_:$InstagramVideoImportDefault, captionsQ_:False]:=Block[
	{splitdata,imageurls,videourls, types,captions, media},
	splitdata = Rest[StringSplit[data, "attribution"]];
	types = Flatten[StringCases[splitdata, "\"type\":\"" ~~ Shortest[x__] ~~ "\"" :> x]];

	If[Length[types]=!=Length[splitdata],Return[{}]];

	imageurls=StringCases[splitdata, "\"images\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}}") :> x];
	videourls=StringCases[splitdata, "\"videos\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}}") :> x];

	imageurls=Flatten[StringCases[#,  "\"" <> resolution <> "\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}") :> x]]&/@imageurls;
	videourls=Flatten[StringCases[#,  "\"" <> resolution <> "\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}") :> x]]&/@videourls;

	media=If[MatchQ[videourls,{({_String}|{})...}]&&MatchQ[imageurls,{{_String}...}],
		imageurls=Flatten["url"/.Map[ImportString[#, "JSON"] &, imageurls, {2}]];
		videourls=Flatten["url"/.Map[ImportString[#, "JSON"] &, videourls, {2}]];
		If[MatchQ[videourls,{_String...}]&&MatchQ[imageurls,{_String...}],
			MapThread[importmedia[#1,#2,#3,videoform]&,{imageurls, videourls,types}],
			{}
		]
		,{}
	];

	If[captionsQ,
		splitdata = splitdata[[Flatten[Position[types, "image", 1]]]];
		captions = StringCases[splitdata, "\"caption\":" ~~ x : ("{" ~~ Shortest[__] ~~ "}") :> x];
		captions = Flatten[StringCases[#, "\"text\":\"" ~~ x : Shortest[__] ~~ "\"" :> OAuthClient`Private`fromunicode[x]]]&/@captions;
		MapThread[Labeled[#1,First[#2]]&,{media,captions/.{}->{""}}],
		media
	]
]

importmedia[_,_,"image",{"OnlyVideo",_}]:=Sequence@@{}
importmedia[iurl_,_,"image",_]:=importphoto[iurl]

importmedia[iurl_,vurl_,"video",{"OnlyVideo",elem_}]:=importmedia[iurl,vurl,"video",elem]
importmedia[iurl_,vurl_,"video",Automatic]:=importmedia[iurl,vurl,"video",$InstagramVideoImportDefault]
importmedia[iurl_,vurl_,"video",True]:=importmedia[iurl,vurl,"video","Animation"]
importmedia[iurl_,_,"video","Image"]:=importmedia[iurl,"","image","Image"]
importmedia[_,_,"video",None]:=Sequence@@{}
importmedia[iurl_,vurl_,"video","ImageLink"]:=Hyperlink[importmedia[iurl,"","image","Image"],vurl]
importmedia[_,vurl_,"video",elem_]:=importvideo[vurl,elem]

importmedia[__]:=Sequence@@{}

importphoto[{str_String}]:=importphoto[str]
importphoto[str_String]:=Import[str]
importphoto[hy_Hyperlink]:=Import[Last[hy]]
importphoto[___]:=Sequence@@{}

importvideo[{str_String},elem_]:=importvideo[str,elem]
importvideo[str_String,elem_]:=Import[str,{"QuickTime",elem}]
importvideo[hy_Hyperlink,rest___]:=importvideo[Last[hy],rest]
importvideo[___]:=Sequence@@{}

readinstagramdate[date_, form_:DateObject]:=form[ToExpression[date] + AbsoluteTime[{1970, 1, 1, 0, 0, 0}, TimeZone -> 0]]

pparam[s_String]:=s
pparam[s_?IntegerQ]:=ToString[s]
pparam[___]:="self"

qualityrules={"High"|"Standard"|"Full"|"StandardResolution"->"standard_resolution","Low"|"LowResolution"->"low_resolution","Thumb"|"thumb"|"Thumbnail"->"thumbnail"};
getquality[args_,default_]:=(("MediaResolution"/.Cases[{args},_Rule,Infinity])/."MediaResolution"->default)/.qualityrules

(* TODO: remove when JSON bug is fixed *)
getids[data_,type_]:=Block[{allids},
	allids=Flatten[StringCases[data, "\"" <> "id" <> "\":\"" ~~ x : Shortest[__] ~~ "\"" :> x]];
	Switch[type,
		"User",
		Flatten[StringCases[allids,
			(DigitCharacter ..) ~~ "_" ~~ (x : (DigitCharacter ..)) :> x]],
		"Media",
		Flatten[StringCases[allids,
			(x : (DigitCharacter ..)) ~~ "_" ~~ (DigitCharacter ..) :> x]],
		"FullMedia",
		Flatten[StringCases[allids,
			(DigitCharacter ..) ~~ "_" ~~ (DigitCharacter ..)]]
	]

];

insticon=Image[RawArray["Byte", {{{253, 253, 253, 0}, {242, 242, 242, 0}, {238, 239, 239, 1}, {242, 242, 242, 2}, {238, 240,
  239, 0}, {235, 237, 239, 0}, {237, 237, 239, 0}, {239, 237, 237, 0}, {239, 238, 232, 0}, {236, 238, 238, 0}, {236,
  238, 239, 0}, {236, 238, 238, 0}, {236, 238, 238, 0}, {235, 238, 238, 0}, {235, 237, 238, 0}, {235, 237, 238, 0},
  {235, 237, 238, 0}, {235, 237, 238, 0}, {235, 238, 238, 0}, {236, 238, 238, 0}, {236, 238, 239, 0}, {236, 238, 239,
  0}, {237, 238, 239, 0}, {237, 239, 239, 0}, {237, 239, 239, 0}, {237, 239, 239, 0}, {236, 238, 238, 0}, {238, 239,
  239, 0}, {242, 242, 242, 2}, {237, 239, 239, 1}, {242, 242, 242, 0}, {252, 252, 252, 0}}, {{216, 216, 216, 0}, {16,
  16, 16, 0}, {71, 55, 52, 0}, {42, 35, 34, 0}, {43, 33, 34, 55}, {88, 67, 50, 132}, {86, 90, 68, 161}, {68, 93, 99,
  164}, {66, 88, 144, 164}, {100, 80, 78, 164}, {104, 77, 68, 164}, {106, 81, 73, 164}, {109, 85, 76, 164}, {112, 88,
  79, 164}, {115, 91, 81, 164}, {115, 92, 81, 164}, {115, 93, 82, 164}, {115, 93, 81, 164}, {112, 89, 79, 164}, {110,
  86, 77, 164}, {107, 83, 74, 164}, {103, 79, 71, 164}, {98, 75, 68, 164}, {93, 71, 65, 164}, {89, 68, 62, 164}, {90,
  68, 62, 161}, {78, 59, 56, 132}, {45, 36, 35, 55}, {49, 39, 38, 0}, {72, 56, 54, 0}, {16, 16, 16, 0}, {205, 206,
  206, 0}}, {{212, 213, 212, 1}, {0, 0, 0, 2}, {42, 30, 26, 5}, {72, 40, 45, 152}, {137, 57, 65, 255}, {171, 112, 45,
  255}, {116, 145, 79, 255}, {72, 148, 152, 255}, {59, 119, 252, 255}, {119, 91, 96, 255}, {134, 81, 64, 255}, {136,
  87, 75, 255}, {142, 92, 77, 255}, {150, 100, 83, 255}, {153, 105, 85, 255}, {156, 110, 88, 255}, {157, 113, 90,
  255}, {154, 109, 87, 255}, {149, 104, 84, 255}, {146, 99, 81, 255}, {138, 91, 75, 255}, {131, 84, 71, 255}, {124,
  79, 68, 255}, {115, 73, 62, 255}, {103, 65, 55, 255}, {98, 60, 51, 255}, {99, 61, 53, 255}, {104, 67, 60, 255},
  {73, 48, 44, 152}, {47, 32, 31, 5}, {0, 0, 0, 2}, {200, 201, 201, 1}}, {{215, 215, 215, 2}, {0, 0, 0, 0}, {55, 45,
  39, 154}, {147, 64, 76, 255}, {199, 67, 73, 252}, {229, 176, 56, 251}, {153, 204, 124, 251}, {96, 192, 192, 251},
  {72, 151, 255, 251}, {137, 108, 107, 251}, {157, 101, 79, 251}, {157, 110, 89, 251}, {162, 116, 93, 251}, {166,
  123, 99, 251}, {169, 129, 103, 251}, {171, 132, 105, 251}, {169, 133, 107, 251}, {168, 131, 104, 251}, {166, 126,
  101, 251}, {163, 121, 95, 251}, {155, 111, 87, 251}, {150, 102, 82, 251}, {118, 77, 63, 251}, {45, 30, 29, 251},
  {41, 33, 35, 251}, {44, 39, 38, 251}, {36, 30, 30, 251}, {38, 26, 25, 252}, {101, 65, 60, 255}, {67, 45, 43, 154},
  {0, 1, 1, 0}, {204, 204, 204, 2}}, {{211, 212, 212, 0}, {14, 7, 6, 58}, {88, 65, 53, 255}, {169, 56, 75, 251},
  {249, 83, 90, 254}, {255, 205, 58, 255}, {164, 213, 129, 255}, {100, 193, 194, 255}, {73, 152, 255, 255}, {143,
  112, 109, 255}, {161, 105, 80, 255}, {162, 113, 91, 255}, {166, 121, 95, 255}, {168, 124, 97, 255}, {170, 128, 100,
  255}, {173, 132, 104, 255}, {174, 135, 107, 255}, {174, 135, 107, 255}, {170, 128, 102, 255}, {167, 123, 96, 255},
  {159, 113, 88, 255}, {160, 107, 85, 255}, {72, 46, 39, 255}, {0, 7, 15, 255}, {18, 27, 31, 255}, {22, 27, 29, 255},
  {37, 50, 53, 255}, {13, 24, 28, 254}, {63, 38, 35, 251}, {99, 65, 58, 255}, {7, 2, 2, 58}, {199, 200, 200, 0}},
  {{210, 211, 212, 0}, {40, 21, 17, 141}, {88, 60, 46, 254}, {188, 58, 80, 251}, {255, 82, 89, 255}, {255, 204, 56,
  255}, {159, 209, 122, 255}, {92, 187, 187, 255}, {70, 146, 255, 255}, {140, 107, 104, 255}, {155, 95, 72, 255},
  {158, 106, 84, 255}, {162, 112, 88, 255}, {164, 116, 89, 255}, {166, 121, 92, 255}, {168, 124, 95, 255}, {168, 126,
  97, 255}, {169, 125, 96, 255}, {164, 119, 91, 255}, {161, 115, 88, 255}, {155, 107, 83, 255}, {158, 103, 80, 255},
  {74, 49, 43, 255}, {9, 18, 22, 255}, {24, 22, 23, 255}, {42, 24, 30, 255}, {69, 70, 84, 255}, {49, 67, 74, 255},
  {71, 46, 40, 251}, {91, 56, 48, 254}, {26, 12, 9, 141}, {198, 200, 200, 0}}, {{210, 212, 213, 0}, {44, 21, 16,
  170}, {91, 60, 44, 255}, {190, 58, 79, 251}, {255, 79, 84, 255}, {255, 201, 53, 255}, {154, 205, 115, 255}, {87,
  183, 181, 255}, {69, 142, 255, 255}, {131, 98, 98, 255}, {149, 87, 65, 255}, {156, 101, 81, 255}, {161, 107, 83,
  255}, {162, 111, 85, 255}, {164, 117, 89, 255}, {163, 116, 89, 255}, {163, 118, 92, 255}, {161, 114, 88, 255},
  {157, 108, 83, 255}, {156, 106, 82, 255}, {150, 100, 78, 255}, {154, 98, 76, 255}, {77, 50, 44, 255}, {9, 17, 21,
  255}, {22, 17, 17, 255}, {52, 35, 50, 255}, {58, 49, 83, 255}, {45, 64, 76, 255}, {84, 59, 50, 251}, {89, 52, 42,
  255}, {28, 11, 6, 170}, {200, 201, 202, 0}}, {{211, 212, 213, 0}, {41, 17, 12, 171}, {89, 56, 41, 255}, {189, 56,
  75, 252}, {252, 77, 81, 255}, {252, 197, 52, 255}, {154, 203, 113, 255}, {87, 180, 179, 255}, {68, 137, 253, 255},
  {126, 92, 94, 255}, {145, 82, 62, 255}, {151, 94, 76, 255}, {158, 102, 80, 255}, {154, 99, 76, 255}, {149, 95, 70,
  255}, {150, 95, 70, 255}, {148, 94, 69, 255}, {147, 93, 68, 255}, {151, 97, 75, 255}, {154, 101, 80, 255}, {147,
  93, 74, 255}, {149, 93, 73, 255}, {76, 48, 41, 255}, {5, 13, 18, 255}, {23, 23, 21, 255}, {40, 38, 38, 255}, {53,
  70, 82, 255}, {31, 56, 57, 255}, {84, 54, 47, 252}, {89, 51, 41, 255}, {25, 7, 3, 171}, {200, 201, 202, 0}}, {{211,
  213, 213, 0}, {41, 16, 11, 171}, {89, 56, 40, 255}, {192, 53, 71, 252}, {255, 73, 76, 255}, {255, 204, 47, 255},
  {156, 211, 108, 255}, {81, 183, 180, 255}, {62, 131, 255, 255}, {121, 89, 92, 255}, {140, 79, 59, 255}, {137, 80,
  63, 255}, {135, 74, 55, 255}, {158, 107, 90, 255}, {183, 145, 129, 255}, {191, 159, 143, 255}, {190, 157, 141,
  255}, {179, 139, 123, 255}, {151, 101, 82, 255}, {132, 74, 55, 255}, {138, 83, 65, 255}, {141, 87, 70, 255}, {109,
  67, 55, 255}, {28, 24, 29, 255}, {38, 42, 46, 255}, {47, 55, 53, 255}, {50, 67, 60, 255}, {46, 43, 43, 255}, {108,
  66, 55, 252}, {84, 48, 37, 255}, {26, 7, 2, 171}, {200, 201, 202, 0}}, {{211, 213, 213, 0}, {31, 7, 3, 171}, {66,
  28, 15, 255}, {149, 37, 46, 252}, {159, 48, 49, 255}, {145, 105, 27, 255}, {91, 113, 58, 255}, {58, 100, 94, 255},
  {53, 83, 171, 255}, {87, 57, 63, 255}, {90, 37, 16, 255}, {121, 76, 62, 255}, {189, 165, 157, 255}, {177, 171, 169,
  255}, {130, 131, 135, 255}, {112, 118, 135, 255}, {114, 121, 139, 255}, {136, 136, 140, 255}, {181, 172, 169, 255},
  {181, 153, 144, 255}, {119, 71, 54, 255}, {96, 46, 29, 255}, {96, 46, 30, 255}, {92, 44, 30, 255}, {86, 41, 30,
  255}, {87, 43, 32, 255}, {85, 38, 29, 255}, {90, 42, 30, 255}, {68, 29, 17, 252}, {54, 21, 11, 255}, {15, 0, 0,
  171}, {200, 202, 202, 0}}, {{209, 211, 211, 0}, {50, 35, 31, 171}, {121, 98, 90, 255}, {101, 89, 84, 252}, {140,
  149, 148, 255}, {115, 113, 119, 255}, {137, 125, 128, 255}, {158, 143, 138, 255}, {148, 132, 115, 255}, {120, 105,
  97, 255}, {145, 130, 124, 255}, {180, 170, 166, 255}, {89, 89, 91, 255}, {8, 9, 15, 255}, {22, 38, 72, 255}, {65,
  104, 152, 255}, {75, 121, 162, 255}, {32, 61, 95, 255}, {18, 23, 32, 255}, {104, 104, 105, 255}, {180, 168, 164,
  255}, {139, 122, 115, 255}, {148, 136, 131, 255}, {146, 132, 126, 255}, {145, 131, 125, 255}, {140, 127, 121, 255},
  {137, 124, 118, 255}, {130, 118, 111, 255}, {123, 110, 104, 252}, {116, 98, 91, 255}, {37, 26, 25, 171}, {198, 199,
  200, 0}}, {{207, 208, 208, 0}, {81, 65, 59, 171}, {214, 194, 181, 255}, {146, 134, 123, 252}, {143, 133, 127, 255},
  {151, 142, 135, 255}, {137, 131, 127, 255}, {136, 132, 130, 255}, {114, 109, 108, 255}, {165, 158, 152, 255}, {218,
  213, 211, 255}, {40, 38, 38, 255}, {0, 0, 4, 255}, {32, 47, 66, 255}, {65, 101, 127, 255}, {109, 148, 160, 255},
  {122, 162, 167, 255}, {80, 127, 140, 255}, {39, 67, 75, 255}, {0, 3, 8, 255}, {63, 60, 61, 255}, {210, 203, 201,
  255}, {251, 251, 249, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 254, 255}, {252, 250, 246, 255},
  {245, 240, 234, 255}, {232, 224, 217, 252}, {203, 187, 179, 255}, {67, 54, 50, 171}, {196, 197, 197, 0}}, {{207,
  208, 209, 0}, {75, 58, 52, 171}, {201, 178, 165, 255}, {199, 187, 176, 252}, {133, 121, 113, 255}, {126, 115, 108,
  255}, {127, 118, 113, 255}, {122, 115, 111, 255}, {138, 133, 129, 255}, {205, 195, 191, 255}, {67, 61, 60, 255},
  {0, 0, 0, 255}, {31, 43, 56, 255}, {37, 60, 70, 255}, {36, 69, 68, 255}, {69, 112, 114, 255}, {83, 123, 129, 255},
  {53, 86, 86, 255}, {48, 80, 79, 255}, {39, 59, 60, 255}, {0, 0, 2, 255}, {87, 78, 78, 255}, {208, 199, 194, 255},
  {243, 241, 238, 255}, {245, 243, 240, 255}, {243, 239, 235, 255}, {237, 232, 227, 255}, {229, 222, 215, 255}, {216,
  207, 198, 252}, {189, 171, 162, 255}, {61, 47, 44, 171}, {196, 197, 198, 0}}, {{207, 208, 209, 0}, {77, 60, 54,
  171}, {198, 175, 162, 255}, {222, 208, 197, 252}, {241, 232, 222, 255}, {250, 244, 234, 255}, {255, 251, 244, 255},
  {255, 255, 254, 255}, {255, 251, 245, 255}, {158, 147, 143, 255}, {2, 1, 4, 255}, {19, 26, 36, 255}, {25, 39, 46,
  255}, {12, 22, 20, 255}, {40, 61, 63, 255}, {77, 100, 116, 255}, {84, 101, 124, 255}, {68, 79, 94, 255}, {92, 113,
  128, 255}, {50, 68, 66, 255}, {34, 48, 45, 255}, {23, 20, 20, 255}, {162, 152, 147, 255}, {228, 223, 218, 255},
  {250, 248, 245, 255}, {243, 240, 235, 255}, {239, 234, 228, 255}, {231, 223, 216, 255}, {217, 207, 197, 252}, {190,
  170, 161, 255}, {61, 47, 44, 171}, {196, 197, 198, 0}}, {{207, 208, 209, 0}, {76, 58, 52, 171}, {197, 173, 159,
  255}, {215, 202, 190, 252}, {225, 215, 204, 255}, {233, 225, 216, 255}, {238, 231, 223, 255}, {243, 240, 234, 255},
  {224, 214, 208, 255}, {99, 90, 88, 255}, {2, 5, 16, 255}, {30, 42, 54, 255}, {16, 28, 29, 255}, {19, 30, 30, 255},
  {43, 55, 61, 255}, {65, 68, 85, 255}, {76, 71, 94, 255}, {59, 53, 69, 255}, {94, 99, 117, 255}, {102, 118, 127,
  255}, {55, 77, 71, 255}, {32, 46, 43, 255}, {118, 111, 107, 255}, {210, 202, 197, 255}, {246, 244, 240, 255}, {242,
  238, 233, 255}, {237, 231, 224, 255}, {228, 220, 212, 255}, {215, 203, 193, 252}, {187, 167, 157, 255}, {60, 46,
  42, 171}, {196, 197, 198, 0}}, {{207, 208, 209, 0}, {76, 58, 52, 171}, {197, 173, 158, 255}, {216, 202, 189, 252},
  {226, 216, 204, 255}, {234, 226, 215, 255}, {239, 232, 223, 255}, {244, 240, 234, 255}, {214, 202, 195, 255}, {76,
  68, 67, 255}, {7, 14, 28, 255}, {31, 45, 55, 255}, {19, 33, 32, 255}, {25, 37, 38, 255}, {31, 35, 40, 255}, {73,
  65, 92, 255}, {89, 70, 106, 255}, {54, 36, 52, 255}, {53, 46, 54, 255}, {74, 82, 88, 255}, {80, 107, 106, 255},
  {52, 81, 81, 255}, {102, 100, 99, 255}, {196, 187, 181, 255}, {239, 236, 231, 255}, {242, 238, 231, 255}, {234,
  228, 220, 255}, {226, 217, 207, 255}, {213, 200, 189, 252}, {186, 164, 153, 255}, {59, 45, 41, 171}, {196, 198,
  198, 0}}, {{207, 208, 209, 0}, {76, 59, 52, 171}, {197, 172, 157, 255}, {215, 201, 187, 252}, {226, 214, 201, 255},
  {232, 224, 212, 255}, {237, 231, 222, 255}, {243, 239, 231, 255}, {209, 196, 188, 255}, {78, 70, 68, 255}, {1, 4,
  12, 255}, {23, 34, 42, 255}, {17, 30, 29, 255}, {23, 33, 35, 255}, {26, 28, 32, 255}, {62, 53, 76, 255}, {77, 56,
  87, 255}, {50, 33, 46, 255}, {46, 39, 43, 255}, {63, 70, 74, 255}, {84, 112, 116, 255}, {49, 77, 81, 255}, {106,
  105, 106, 255}, {189, 180, 173, 255}, {234, 230, 223, 255}, {240, 235, 227, 255}, {232, 224, 214, 255}, {223, 213,
  202, 255}, {210, 196, 183, 252}, {184, 162, 148, 255}, {59, 45, 40, 171}, {196, 198, 198, 0}}, {{207, 209, 209, 0},
  {75, 57, 50, 171}, {195, 170, 154, 255}, {214, 199, 183, 252}, {225, 212, 199, 255}, {231, 222, 211, 255}, {236,
  229, 219, 255}, {243, 239, 231, 255}, {207, 194, 186, 255}, {104, 94, 91, 255}, {0, 1, 7, 255}, {19, 27, 33, 255},
  {12, 20, 20, 255}, {12, 17, 18, 255}, {17, 19, 22, 255}, {28, 24, 31, 255}, {35, 26, 35, 255}, {29, 19, 23, 255},
  {30, 26, 27, 255}, {58, 68, 70, 255}, {77, 106, 108, 255}, {45, 68, 73, 255}, {127, 124, 123, 255}, {180, 170, 162,
  255}, {234, 229, 222, 255}, {236, 230, 221, 255}, {228, 221, 209, 255}, {220, 208, 195, 255}, {206, 192, 176, 252},
  {181, 159, 144, 255}, {58, 44, 39, 171}, {196, 198, 198, 0}}, {{207, 209, 209, 0}, {74, 57, 49, 171}, {193, 167,
  149, 255}, {212, 196, 181, 252}, {223, 210, 196, 255}, {230, 220, 208, 255}, {234, 226, 215, 255}, {243, 239, 230,
  255}, {206, 195, 187, 255}, {146, 131, 125, 255}, {15, 15, 19, 255}, {12, 18, 26, 255}, {16, 26, 28, 255}, {12, 21,
  20, 255}, {16, 20, 22, 255}, {27, 28, 32, 255}, {31, 28, 34, 255}, {26, 22, 26, 255}, {37, 37, 39, 255}, {56, 68,
  65, 255}, {59, 85, 83, 255}, {46, 55, 56, 255}, {153, 144, 140, 255}, {170, 159, 150, 255}, {239, 234, 225, 255},
  {231, 224, 212, 255}, {225, 216, 203, 255}, {216, 204, 189, 255}, {203, 188, 172, 252}, {178, 155, 141, 255}, {57,
  43, 38, 171}, {197, 198, 198, 0}}, {{207, 209, 209, 0}, {73, 55, 48, 171}, {190, 164, 147, 255}, {210, 194, 179,
  252}, {221, 208, 194, 255}, {229, 218, 205, 255}, {233, 225, 213, 255}, {238, 232, 223, 255}, {214, 204, 195, 255},
  {157, 138, 131, 255}, {92, 83, 82, 255}, {0, 0, 1, 255}, {22, 31, 39, 255}, {21, 35, 37, 255}, {19, 31, 28, 255},
  {24, 31, 30, 255}, {28, 32, 33, 255}, {33, 38, 39, 255}, {45, 57, 54, 255}, {50, 68, 63, 255}, {17, 26, 23, 255},
  {100, 94, 91, 255}, {142, 129, 122, 255}, {182, 172, 162, 255}, {238, 231, 220, 255}, {227, 217, 203, 255}, {222,
  211, 195, 255}, {214, 201, 185, 255}, {200, 185, 168, 252}, {175, 152, 137, 255}, {56, 42, 37, 171}, {197, 198,
  198, 0}}, {{207, 209, 209, 0}, {72, 55, 48, 171}, {190, 164, 147, 255}, {209, 193, 177, 252}, {221, 207, 193, 255},
  {228, 217, 203, 255}, {233, 223, 211, 255}, {233, 225, 214, 255}, {227, 218, 208, 255}, {158, 141, 131, 255}, {142,
  124, 119, 255}, {70, 64, 63, 255}, {0, 6, 15, 255}, {32, 50, 65, 255}, {47, 73, 83, 255}, {50, 82, 86, 255}, {55,
  88, 90, 255}, {57, 91, 93, 255}, {51, 80, 83, 255}, {17, 26, 29, 255}, {66, 58, 56, 255}, {139, 127, 122, 255},
  {119, 104, 95, 255}, {217, 208, 194, 255}, {228, 219, 205, 255}, {224, 214, 198, 255}, {219, 207, 190, 255}, {210,
  198, 180, 255}, {197, 182, 165, 252}, {173, 150, 134, 255}, {55, 41, 36, 171}, {197, 198, 198, 0}}, {{207, 209,
  209, 0}, {72, 54, 47, 171}, {188, 162, 145, 255}, {207, 191, 175, 252}, {219, 206, 190, 255}, {226, 214, 200, 255},
  {231, 220, 208, 255}, {232, 223, 211, 255}, {225, 216, 205, 255}, {201, 187, 177, 255}, {122, 102, 95, 255}, {134,
  117, 112, 255}, {96, 87, 86, 255}, {34, 37, 47, 255}, {34, 51, 73, 255}, {48, 79, 108, 255}, {50, 86, 111, 255},
  {40, 64, 78, 255}, {37, 42, 48, 255}, {90, 80, 79, 255}, {131, 117, 114, 255}, {88, 73, 67, 255}, {177, 163, 150,
  255}, {223, 213, 197, 255}, {223, 213, 196, 255}, {221, 210, 193, 255}, {215, 203, 185, 255}, {207, 194, 175, 255},
  {195, 178, 160, 252}, {170, 147, 131, 255}, {53, 41, 36, 171}, {197, 198, 198, 0}}, {{207, 209, 210, 0}, {70, 52,
  45, 171}, {185, 159, 142, 255}, {206, 189, 172, 252}, {217, 202, 187, 255}, {224, 211, 196, 255}, {228, 217, 203,
  255}, {229, 220, 207, 255}, {225, 215, 203, 255}, {213, 200, 190, 255}, {177, 159, 148, 255}, {103, 84, 78, 255},
  {109, 92, 88, 255}, {125, 110, 106, 255}, {104, 92, 90, 255}, {90, 80, 81, 255}, {88, 79, 80, 255}, {101, 88, 86,
  255}, {120, 105, 101, 255}, {102, 89, 85, 255}, {75, 60, 55, 255}, {152, 136, 124, 255}, {211, 198, 182, 255},
  {218, 207, 190, 255}, {221, 211, 192, 255}, {217, 205, 187, 255}, {212, 199, 179, 255}, {204, 189, 170, 255}, {192,
  175, 155, 252}, {167, 144, 128, 255}, {52, 39, 34, 171}, {197, 198, 199, 0}}, {{208, 209, 210, 0}, {67, 51, 44,
  171}, {182, 155, 138, 255}, {203, 185, 168, 252}, {214, 199, 182, 255}, {220, 207, 191, 255}, {225, 213, 197, 255},
  {227, 217, 202, 255}, {226, 216, 203, 255}, {217, 205, 193, 255}, {198, 183, 171, 255}, {166, 146, 135, 255}, {108,
  89, 82, 255}, {78, 63, 58, 255}, {83, 69, 65, 255}, {89, 75, 71, 255}, {87, 73, 69, 255}, {78, 65, 61, 255}, {66,
  53, 49, 255}, {85, 69, 63, 255}, {150, 132, 121, 255}, {194, 179, 164, 255}, {210, 198, 180, 255}, {217, 207, 188,
  255}, {218, 206, 188, 255}, {214, 202, 183, 255}, {209, 195, 175, 255}, {201, 185, 166, 255}, {189, 171, 151, 252},
  {163, 139, 123, 255}, {51, 38, 33, 171}, {197, 198, 199, 0}}, {{208, 209, 210, 0}, {66, 50, 43, 171}, {177, 150,
  133, 255}, {200, 181, 163, 252}, {211, 195, 178, 255}, {218, 203, 186, 255}, {222, 209, 192, 255}, {224, 214, 197,
  255}, {225, 214, 199, 255}, {222, 210, 196, 255}, {209, 195, 182, 255}, {188, 170, 158, 255}, {164, 143, 132, 255},
  {136, 114, 105, 255}, {107, 89, 81, 255}, {91, 75, 68, 255}, {87, 71, 65, 255}, {97, 80, 73, 255}, {124, 105, 96,
  255}, {156, 137, 124, 255}, {182, 165, 150, 255}, {201, 188, 171, 255}, {213, 201, 183, 255}, {216, 205, 186, 255},
  {215, 204, 185, 255}, {212, 199, 180, 255}, {206, 192, 172, 255}, {198, 182, 161, 255}, {185, 166, 145, 252}, {160,
  135, 118, 255}, {50, 37, 32, 171}, {197, 198, 199, 0}}, {{208, 210, 210, 0}, {63, 47, 40, 171}, {169, 142, 125,
  255}, {194, 174, 155, 251}, {207, 189, 171, 255}, {214, 199, 181, 255}, {219, 204, 186, 255}, {221, 207, 190, 255},
  {222, 210, 194, 255}, {221, 209, 193, 255}, {216, 203, 187, 255}, {203, 187, 173, 255}, {184, 166, 153, 255}, {166,
  145, 133, 255}, {151, 130, 118, 255}, {143, 121, 111, 255}, {140, 119, 109, 255}, {146, 125, 113, 255}, {161, 141,
  127, 255}, {179, 161, 145, 255}, {197, 181, 164, 255}, {209, 195, 176, 255}, {213, 199, 179, 255}, {213, 200, 180,
  255}, {211, 198, 179, 255}, {208, 194, 175, 255}, {201, 186, 166, 255}, {191, 173, 151, 255}, {178, 157, 135, 251},
  {152, 127, 110, 255}, {48, 35, 29, 171}, {198, 199, 199, 0}}, {{208, 210, 210, 0}, {45, 31, 27, 146}, {155, 126,
  111, 254}, {184, 162, 143, 251}, {198, 178, 159, 255}, {206, 189, 169, 255}, {213, 197, 178, 255}, {217, 201, 183,
  255}, {219, 203, 185, 255}, {219, 204, 186, 255}, {217, 201, 184, 255}, {211, 195, 178, 255}, {202, 184, 167, 255},
  {190, 171, 155, 255}, {178, 159, 143, 255}, {170, 150, 135, 255}, {169, 149, 134, 255}, {175, 156, 140, 255}, {186,
  168, 151, 255}, {197, 180, 162, 255}, {206, 190, 171, 255}, {210, 194, 174, 255}, {210, 195, 174, 255}, {208, 193,
  173, 255}, {205, 190, 170, 255}, {200, 185, 163, 255}, {192, 175, 152, 255}, {182, 162, 139, 255}, {167, 143, 123,
  251}, {138, 111, 98, 254}, {32, 21, 17, 146}, {198, 199, 199, 0}}, {{212, 212, 213, 0}, {10, 5, 4, 67}, {125, 96,
  84, 255}, {167, 140, 123, 251}, {186, 164, 144, 254}, {196, 175, 155, 255}, {203, 184, 163, 255}, {208, 190, 170,
  255}, {210, 193, 174, 255}, {211, 195, 175, 255}, {211, 195, 175, 255}, {209, 193, 174, 255}, {206, 190, 171, 255},
  {202, 185, 166, 255}, {196, 179, 161, 255}, {192, 174, 156, 255}, {192, 173, 156, 255}, {195, 177, 158, 255}, {200,
  182, 162, 255}, {203, 186, 166, 255}, {205, 188, 168, 255}, {205, 188, 167, 255}, {202, 186, 165, 255}, {200, 182,
  160, 255}, {195, 177, 154, 255}, {189, 170, 147, 255}, {182, 161, 138, 255}, {172, 149, 127, 254}, {151, 126, 109,
  251}, {112, 87, 77, 255}, {4, 1, 1, 67}, {200, 201, 201, 0}}, {{216, 215, 215, 2}, {0, 0, 0, 1}, {53, 39, 35, 168},
  {141, 109, 95, 255}, {166, 138, 120, 251}, {181, 155, 134, 251}, {188, 165, 143, 252}, {193, 172, 150, 252}, {197,
  176, 154, 252}, {198, 178, 156, 252}, {199, 180, 159, 252}, {200, 182, 161, 252}, {200, 182, 161, 252}, {200, 181,
  161, 252}, {198, 179, 159, 252}, {196, 177, 157, 252}, {195, 176, 156, 252}, {196, 176, 155, 252}, {197, 177, 156,
  252}, {196, 177, 156, 252}, {195, 175, 154, 252}, {192, 173, 151, 252}, {188, 169, 146, 252}, {185, 164, 141, 252},
  {181, 159, 136, 252}, {174, 152, 130, 252}, {165, 141, 121, 251}, {150, 124, 107, 251}, {126, 99, 87, 255}, {49,
  37, 33, 168}, {0, 0, 0, 1}, {204, 204, 204, 2}}, {{214, 214, 215, 1}, {5, 4, 4, 1}, {18, 11, 9, 13}, {57, 43, 38,
  169}, {124, 96, 83, 255}, {153, 122, 105, 255}, {167, 136, 117, 255}, {176, 145, 125, 255}, {180, 149, 129, 255},
  {180, 151, 131, 255}, {181, 153, 133, 255}, {182, 154, 134, 255}, {183, 155, 135, 255}, {184, 156, 136, 255}, {184,
  156, 135, 255}, {182, 155, 134, 255}, {182, 153, 133, 255}, {180, 151, 131, 255}, {178, 150, 130, 255}, {176, 148,
  128, 255}, {175, 147, 126, 255}, {173, 145, 125, 255}, {168, 140, 120, 255}, {164, 136, 116, 255}, {159, 131, 113,
  255}, {151, 124, 107, 255}, {136, 111, 96, 255}, {110, 87, 76, 255}, {52, 40, 35, 169}, {18, 12, 10, 13}, {4, 4, 4,
  1}, {203, 203, 203, 1}}, {{210, 210, 210, 0}, {0, 0, 0, 1}, {11, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 76}, {27, 14, 8,
  153}, {44, 27, 20, 180}, {48, 31, 23, 182}, {49, 32, 25, 182}, {49, 33, 26, 182}, {50, 33, 26, 182}, {50, 34, 26,
  182}, {51, 34, 27, 182}, {51, 35, 27, 182}, {51, 34, 26, 182}, {51, 34, 26, 182}, {50, 33, 26, 182}, {49, 33, 25,
  182}, {48, 32, 25, 182}, {47, 31, 24, 182}, {47, 31, 24, 182}, {46, 31, 23, 182}, {44, 29, 22, 182}, {42, 27, 20,
  182}, {40, 25, 19, 182}, {36, 23, 17, 180}, {21, 10, 6, 153}, {0, 0, 0, 76}, {0, 0, 0, 0}, {10, 0, 0, 0}, {0, 0, 0,
  1}, {198, 198, 198, 0}}, {{240, 240, 240, 0}, {162, 162, 162, 0}, {160, 161, 161, 1}, {163, 163, 163, 1}, {159,
  160, 160, 0}, {156, 158, 158, 0}, {156, 157, 158, 0}, {155, 157, 158, 0}, {155, 157, 158, 0}, {155, 157, 157, 0},
  {155, 157, 157, 0}, {155, 157, 157, 0}, {155, 157, 157, 0}, {155, 157, 157, 0}, {155, 157, 157, 0}, {155, 157, 157,
  0}, {155, 157, 157, 0}, {155, 157, 157, 0}, {155, 157, 158, 0}, {156, 157, 158, 0}, {156, 157, 158, 0}, {156, 157,
  158, 0}, {156, 157, 158, 0}, {156, 157, 158, 0}, {156, 158, 158, 0}, {157, 158, 158, 0}, {157, 158, 159, 0}, {160,
  160, 161, 0}, {163, 163, 162, 1}, {160, 161, 162, 1}, {162, 162, 162, 0}, {236, 236, 236, 0}}}], "Byte",
 ColorSpace -> "RGB", Interleaving -> True];

End[] (* End Private Context *)

End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{InstagramOAuth`Private`instagramdata,InstagramOAuth`Private`instagramcookeddata,InstagramOAuth`Private`instagramsendmessage}
