Begin["LinkedIn`"]

Begin["`Private`"]

ServiceExecute::liurl="The URL field is a required parameter of `1` in the Message parameter."
SendMessage::liurl="The URL field is a required parameter of `1` in the SendMessage request."

(******************************* LinkedIn *************************************)

(* Authentication information *)

linkedindata[]:=
    If[TrueQ[OAuthClient`Private`$UseChannelFramework],{
		"OAuthVersion"		-> "2.0",
		"ServiceName" 		-> "LinkedIn", 
	 	"AuthorizeEndpoint" -> "https://www.linkedin.com/uas/oauth2/authorization", 
     	"AccessEndpoint"    -> "https://www.linkedin.com/uas/oauth2/accessToken",
     	"RedirectURI"       -> "WolframConnectorChannelListen",
        "Blocking"          -> False,
        "VerifierLabel"     -> "code",
        "ClientInfo"        -> {"Wolfram","Token"}, 
        "AuthenticationDialog" :> "WolframConnectorChannel",
        "AuthorizationFunction"-> "LinkedIn",
        "RedirectURLFunction"->(#1&),
        "Gets"              -> {"UserData"},
	 	"Posts"				-> {"Share"},
	 	"RawGets"			-> {"RawUserData"},
	 	"RawPosts"			-> {"RawShare"},
	 	"RequestFormat"		-> (Block[{params = Lookup[{##2},"Parameters",{}], method = Lookup[{##2}, "Method"],
        							body = Lookup[{##2},"BodyData",""], headers = Lookup[{##2},"Headers", {}], auth},
        							auth = Lookup[params,"access_token"];
        							Switch[method,
        									"GET",
										URLFetch[#1, "Headers" -> headers, Method -> method, "Body" -> body,
										"Parameters" -> Normal@Join[KeyDrop["access_token"][params],<|"oauth2_access_token"->auth|>],
										"CredentialsProvider" -> None]
										,
											"POST",
										URLFetch[URLDecode@URLBuild[#1, {"oauth2_access_token" -> auth}],
											"Headers" -> headers, "Method" -> method, "Body" -> body, "CredentialsProvider" -> None]
        							
        							]
        						]&),
	 	"Scope"				-> {"r_basicprofile+w_share"},
 		"Information"		-> "A service for receiving data from a LinkedIn account"

	}
	,
	{
        "OAuthVersion"      -> "2.0",
        "ServiceName"       -> "LinkedIn", 
        "AuthorizeEndpoint" -> "https://www.linkedin.com/uas/oauth2/authorization", 
        "AccessEndpoint"    -> "https://www.linkedin.com/uas/oauth2/accessToken",
        "RedirectURI"       -> "https://www.wolfram.com/oauthlanding?service=LinkedIn",
        "VerifierLabel"     -> "code",
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "LinkedIn",liicon]&),
        "Gets"              -> {"UserData","Connections","ConnectionIDs","EgoNetwork","GroupNames" (*,"UserSearch" *)},
        "Posts"             -> {"Share"},
        "RawGets"           -> {"RawUserData",(* "RawPeopleSearch", *)"RawUserGroups","RawUserGroup","RawGroups","RawGroupPosts","RawSuggestedGroups","RawConnections"},
        "RawPosts"          -> {"RawJoinGroup","RawShare"},
	 	"RequestFormat"		-> (Block[{params = Lookup[{##2},"Parameters",{}], method = Lookup[{##2}, "Method"],
        							body = Lookup[{##2},"BodyData",""], headers = Lookup[{##2},"Headers", {}], auth},
        							auth = Lookup[params,"access_token"];
        							Switch[method,
        									"GET",
										URLFetch[#1, "Headers" -> headers, Method -> method, "Body" -> body,
										"Parameters" -> Normal@Join[KeyDrop["access_token"][params],<|"oauth2_access_token"->auth|>],
										"CredentialsProvider" -> None]
										,
											"POST",
										URLFetch[URLDecode@URLBuild[#1, {"oauth2_access_token" -> auth}],
											"Headers" -> headers, "Method" -> method, "Body" -> body, "CredentialsProvider" -> None]
        							
        							]
        						]&),
	 	"Scope"				-> {"r_basicprofile+w_share"},
        "Information"       -> "A service for receiving data from a LinkedIn account"
	}
]

(* Import function *)

linkedinimport[___]:= Throw[$Failed];
linkedinimport[json_String]:=With[{res = Quiet[Developer`ReadRawJSONString[json]]},
	If[AssociationQ[res],
		If[!KeyExistsQ[res,"errorCode"],
			res,
			Message[ServiceExecute::apierr, res["message"]];
			Throw[$Failed]
		]
		,
		Message[ServiceExecute::serror];
		Throw[$Failed]
	]
]

linkedinimport[raw_]:=raw
 
(*** Raw ***) 
linkedindata["RawUserData"] = {
	"URL"				-> "https://api.linkedin.com/v1/people/~:(id,formatted-name,headline,industry,distance,current-status,current-share,num-connections,picture-url,public-profile-url)",
	"Headers" 			-> {"Accept-Language" -> "en_US", "x-li-format" -> "json"},
	"ResultsFunction"	-> linkedinimport
}

linkedindata["RawShare"] = {
	"URL"				-> "https://api.linkedin.com/v1/people/~/shares",
	"BodyData"			-> {"ParameterlessBodyData"},
	"HTTPSMethod"		-> "POST",
	"Headers" 			-> {"Content-Type" -> "application/json", "x-li-format" -> "json"},
	"ResultsFunction"	-> linkedinimport
}     
    
linkedindata["icon"]=liicon

linkedindata[___]:=$Failed

(****** Cooked Properties ******)
linkedincookeddata["UserData", id_, args_]:= Block[{invalidParameters,raw},

	invalidParameters = Keys[args];

	If[Length[invalidParameters]>0,
		Message[ServiceObject::noget,#,"LinkedIn"]&/@invalidParameters;
		Throw[$Failed]
	];

	raw = linkedinimport@OAuthClient`rawoauthdata[id,"RawUserData",{}];
	raw = MapAt[formatShare, Key["currentShare"]]@raw;
	raw = KeyMap[Capitalize[StringReplace[{"id" -> "ID", "Url" -> "URL", "num" -> "NumberOf"}][#]] &]@raw;
	Dataset[Join[raw, Map[URL]@KeySelect[raw, StringContainsQ["URL"]]]]	
]

linkedincookeddata["Share",id_,args_]:=Block[{invalidParameters,raw,message,comment,title,description,url,imageurl,visibility,jsonbody},

	invalidParameters = Select[Keys[args],!MemberQ[{"Message", "Visibility"},#]&];

	If[ Length[invalidParameters]>0,
		Message[ServiceObject::noget,#,"LinkedIn"]&/@invalidParameters;
		Throw[$Failed]
	];

	message = Lookup[args, "Message", Message[ServiceExecute::nparam, "Message", "LinkedIn"]; Throw[$Failed]];
	
	Switch[message,
				_String?StringQ,
					jsonbody = <|"comment" -> message, "visibility" -> <|"code" -> "anyone"|>|>,
				_List?OptionQ | _Association?AssociationQ,
					message = Association @ message;
					invalidParameters = Select[Keys[message],!MemberQ[{"Comment", "Title", "Description", "URL", "ThumbnailURL"},#]&];
					If[ Length[invalidParameters]>0,
						Message[ServiceExecute::nval, "Message", "LinkedIn"];
						Throw[$Failed]
					];
					comment = message["Comment"];
					title = message["Title"];
					description = message["Description"];
					url = Lookup[message, "URL", Message[ServiceExecute::liurl, message]; Throw[$Failed]];
					imageurl = message["ThumbnailURL"];
					If[ !AllTrue[{comment, title, description}, Or[StringQ[#], MissingQ[#]] &],
						Message[ServiceExecute::nval, "Message", "LinkedIn"]
					];
					If[ !AllTrue[{url, imageurl}, Or[StringQ[#], MatchQ[#, URL[_?StringQ]], MissingQ[#]] &],
						Message[ServiceExecute::nval, "Message", "LinkedIn"]
					];
					jsonbody = DeleteMissing @ <|"comment" -> comment,
								"content" -> DeleteMissing @ <|"title" -> title, "description" -> description,
									"submitted-url" -> Replace[url,URL[str_]->str],
									"submitted-image-url" -> Replace[imageurl,URL[str_]->str]|>,
								"visibility" -> <|"code" -> "anyone"|>|>
					,
				_,
					Message[ServiceExecute::nval, "Message", "LinkedIn"]; Throw[$Failed]
	];

	If[ KeyExistsQ[args, "Visibility"], 
		visibility = Lookup[args, "Visibility"];
		If[!MatchQ[visibility, "Anyone" | "ConnectionsOnly"],
			Message[ServiceExecute::nval, "Visibility", "LinkedIn"];
			Throw[$Failed]
		];
		AssociateTo[jsonbody, "visibility" -> <|"code" -> Replace[visibility, {"Anyone"->"anyone","ConnectionsOnly"->"connections-only"}]|>]
	];
	
	raw = linkedinimport@OAuthClient`rawoauthdata[id,"RawShare", "ParameterlessBodyData" -> Developer`WriteRawJSONString[jsonbody]];
	MapAt[URL, Key["UpdateURL"]]@(KeyMap[Replace[{"updateKey" -> "UpdateKey", "updateUrl" -> "UpdateURL"}]] @raw)
]

linkedincookeddata[___]:=$Failed 
(* Send Message *)

linkedinsendmessage[id_,message : _String | _List?OptionQ | _Association?AssociationQ]:= Quiet[Check[Catch[
	With[ {res = linkedincookeddata["Share", id, {"Message"->message}]},
		If[AssociationQ[res]&&FreeQ[res,"error"], res["UpdateURL"], $Failed]
	]],
	Message[SendMessage::liurl, message]; $Failed
	,
	{ServiceExecute::liurl}], {ServiceExecute::liurl}]

linkedinsendmessage[___]:=$Failed
	
(*** Service specific utilites ****)

formatShare[share_Association]:= Block[{res},
	res = MapAt[FromUnixTime[#/1000] &, Key["timestamp"]][share];
	res["visibility"] = Replace[res["visibility"]["code"], {"anyone" -> "Anyone", "connection-only" -> "ConnectionsOnly"}];
	Replace[res, asoc_?AssociationQ :> KeyMap[Capitalize[StringReplace[{"id" -> "ID"}][#]] &][asoc], {0, Infinity}]
]


(* Icon *)
liicon=Image[RawArray["Byte", {{{9, 118, 180, 16}, {9, 118, 180, 196}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 196}, {9, 119, 181, 16}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 
  119, 181, 196}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, 
  {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 
  180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 
  118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, 
  {9, 119, 181, 196}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, 
  {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 
  118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, 
  {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 32}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {32, 137, 190, 255}, {127, 186, 217, 255}, {128, 187, 218, 255}, {32, 136, 190, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}}, {{9, 119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {32, 137, 
  190, 255}, {240, 247, 250, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {240, 247, 250, 255}, {32, 136, 190, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {128, 187, 218, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {127, 186, 217, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 
  180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {128, 187, 218, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {128, 187, 218, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 
  118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, 
  {{9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {32, 136, 190, 255}, {240, 247, 
  250, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {240, 247, 250, 255}, {32, 136, 190, 255}, {9, 118, 180, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {32, 136, 190, 255}, 
  {128, 187, 218, 255}, {128, 187, 217, 255}, {32, 137, 190, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 
  118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 
  255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 
  180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {32, 
  137, 190, 255}, {65, 153, 201, 255}, {64, 153, 200, 255}, {65, 154, 201, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {193, 222, 238, 255}, {9, 118, 180, 255}, {144, 196, 223, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {224, 238, 246, 255}, {79, 
  162, 203, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 118, 180, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 118, 
  180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {160, 204, 228, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {48, 145, 195, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {160, 203, 227, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, 
  {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 118, 180, 
  255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {240, 247, 252, 255}, {9, 119, 181, 255}, {9, 118, 180, 
  255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 118, 180, 255}, 
  {9, 119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {207, 230, 241, 255}, {33, 137, 191, 255}, {9, 119, 181, 255}, {16, 128, 186, 255}, {176, 213, 231, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 
  118, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {80, 162, 204, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {64, 
  153, 200, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {65, 154, 201, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {32, 136, 190, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {9, 119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 118, 
  180, 255}, {9, 119, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 26}, {0, 0, 0, 102}, {0, 0, 0, 102}, {0, 0, 
  0, 102}, {0, 0, 0, 26}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 
  181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {0, 0, 0, 26}, {0, 0, 0, 102}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {0, 0, 0, 102}, {0, 0, 0, 26}}, {{9, 119, 181, 255}, {9, 119, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 
  119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 
  119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {0, 0, 0, 102}, {255, 255, 255, 0}, {0, 0, 0, 102}, {0, 0, 0, 51}, {0, 0, 0, 102}, {255, 255, 
  255, 0}, {0, 0, 0, 102}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 
  180, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 118, 180, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {0, 0, 0, 102}, {255, 255, 255, 0}, {0, 0, 0, 102}, {0, 0, 0, 102}, {0, 0, 0, 26}, {255, 255, 255, 0}, {0, 0, 
  0, 102}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {0, 0, 0, 102}, {255, 255, 255, 0}, {0, 0, 0, 102}, {255, 255, 255, 0}, {0, 0, 0, 102}, {255, 255, 255, 0}, {0, 0, 
  0, 102}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 118, 180, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {9, 119, 181, 255}, {9, 
  118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {0, 0, 0, 26}, {0, 0, 0, 102}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 102}, {0, 0, 
  0, 26}}, {{9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 
  180, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 
  119, 181, 255}, {9, 118, 180, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {0, 0, 0, 26}, {0, 
  0, 0, 102}, {0, 0, 0, 102}, {0, 0, 0, 102}, {0, 0, 0, 26}, {255, 255, 255, 0}}, {{9, 119, 180, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 
  119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 118, 
  180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 
  118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, 
  {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}}, {{9, 119, 181, 196}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 
  180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 196}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}}, {{9, 118, 180, 16}, {9, 119, 181, 196}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 180, 255}, {9, 119, 
  181, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 118, 180, 255}, {9, 
  119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, 
  {9, 119, 180, 255}, {9, 118, 180, 255}, {9, 119, 181, 255}, {9, 119, 181, 255}, {9, 118, 180, 255}, {9, 119, 181, 
  255}, {9, 119, 181, 196}, {9, 119, 181, 16}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}}}], 
 "Byte", ColorSpace -> "RGB", Interleaving -> True];


End[] (* End Private Context *)
           		
End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{LinkedIn`Private`linkedindata,LinkedIn`Private`linkedincookeddata,LinkedIn`Private`linkedinsendmessage}
