Begin["Reddit`"]

Get["RedditFunctions.m"];

ServiceExecute::rdtfn="`1` is not a valid Reddit GlobalID."
ServiceExecute::rdtfn2="\"GlobalIDInformation\" only supports valid t1, t3 and t5 GlobalIDs"
ServiceExecute::rdtfn3="`1` only supports valid urls or t3 GlobalIDs"
ServiceExecute::rdmod="This request requires moderator privileges"
ServiceExecute::rduflo="No additional parameter is allowed along the \"User\" parameter"
ServiceExecute::rdivsr="`1` is not a valid Subreddit"
ServiceExecute::rdivurl = "`1` is not a valid Reddit url"
ServiceExecute::rdivusr = "`1` is not a valid Reddit user"

Begin["`Private`"]

(******************************* Reddit *************************************)

redditdata[]:=
	If[TrueQ[OAuthClient`Private`$UseChannelFramework],
		{
		"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "Reddit",
		"AuthorizeEndpoint"		-> "https://www.reddit.com/api/v1/authorize",
		"AccessEndpoint"		-> "https://www.reddit.com/api/v1/access_token",
		"RedirectURI"			-> "WolframConnectorChannelListen",
		"Blocking"				-> False,
		"RedirectURLFunction"	-> (#1&),
		"AuthorizationFunction"	-> "Reddit",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"AccessTokenRequestor"	-> "HTTPBasic",
		"RefreshAccessTokenFunction" -> "HTTPBasic",
		"VerifierLabel"			-> "code",
		"VerifyPeer"			-> True,
		"AuthenticationDialog"	:> "WolframConnectorChannel",
		"ClientInfo"			-> {"Wolfram", "Token"},
		"RequestFormat"			-> (Block[{params=Lookup[{##2},"Parameters",{}],method=Lookup[{##2},"Method","GET"],body=Lookup[{##2},"BodyData",""],
									auth}, auth = Lookup[params,"access_token",""];
									URLFetch[#1, {"StatusCode", "Content"}, "Headers" -> {"Authorization" ->"bearer  "<>auth}, "Method" -> method,
										"BodyData" -> body, "Parameters" -> DeleteCases[params,("access_token" ->_)], "VerifyPeer" -> True]
									]&),
		"Gets"					-> {"AccountData","UserData","UserActivity","AccountKarma","UserTrophies","AccountFriends","GlobalIDInformation","PrivateMessages",
									"PostInformation","PostCommentsData","CommentReplyData","SubredditInformation","SubredditPosts"(*,
									"SubredditFlair","SubredditFlairAlternatives"*)},
		"Posts"					-> {},
		"RawGets"				-> {"RawAccountInformation","RawAccountPreferences","RawAccountKarma","RawAccountTrophies","RawAccountFriendsList",
									"RawFlairList",
									"RawPMInbox","RawPMUnread","RawPMSent",
									"RawCaptchaQ","RawCaptchaCreate","RawCaptchaRetrieve","RawSubredditInformation","RawFullnameInfo",
									"RawUsernameAbout","RawUsernameOverview","RawUsernameUpvoted","RawUsernameDownvoted","RawUsernameSubmitted","RawUsernameComments",
									"RawUsernameHidden","RawUsernameSaved","RawUsernameGilded","RawUsernameTrophies",
									"RawSubreddit","RawSubredditHot","RawSubredditTop","RawSubredditNew","RawSubredditControversial","RawSubredditRising",
									"RawSubredditAbout","RawSubredditSidebar","RawSubredditSticky","RawSubredditSearch","RawPostComments"},
		"RawPosts"				-> {"RawFlairSelector","RawSubredditComment"},
		"Scope"					-> {"identity,edit,flair,history,modconfig,modflair,modlog,modposts,modwiki,mysubreddits,privatemessages,read,report,save,submit,subscribe,vote,wikiedit,wikiread"},
		"Information"			-> "Import Reddit data to the Wolfram Language"
    }
    ,
		{
		"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "Reddit",
		"AuthorizeEndpoint"		-> "https://www.reddit.com/api/v1/authorize",
		"AccessEndpoint"		-> "https://www.reddit.com/api/v1/access_token",
		"RedirectURI"			-> "https://www.wolfram.com/oauthlanding/?service=Reddit",
		"AuthorizationFunction"	-> "Reddit",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"AccessTokenRequestor"	-> "HTTPBasic",
		"RefreshAccessTokenFunction" -> "HTTPBasic",
		"VerifierLabel"			-> "code",
		"VerifyPeer"			-> True,
		"AuthenticationDialog"	:> (OAuthClient`tokenOAuthDialog[#, "Reddit"]&),
		"ClientInfo"			-> {"Wolfram", "Token"},
		"RequestFormat"			-> (Block[{params=Lookup[{##2},"Parameters",{}],method=Lookup[{##2},"Method","GET"],body=Lookup[{##2},"BodyData",""],
									auth}, auth = Lookup[params,"access_token",""];
									URLFetch[#1, {"StatusCode", "Content"}, "Headers" -> {"Authorization" ->"bearer  "<>auth}, "Method" -> method,
										"BodyData" -> body, "Parameters" -> DeleteCases[params,("access_token" ->_)], "VerifyPeer" -> True]
									]&),
		"Gets"					-> {"AccountData","UserData","UserActivity","AccountKarma","UserTrophies","AccountFriends","GlobalIDInformation","PrivateMessages",
									"PostInformation","PostCommentsData","CommentReplyData","SubredditInformation","SubredditPosts"(*,
									"SubredditFlair","SubredditFlairAlternatives"*)},
		"Posts"					-> {},
		"RawGets"				-> {"RawAccountInformation","RawAccountPreferences","RawAccountKarma","RawAccountTrophies","RawAccountFriendsList",
									"RawFlairList",
									"RawPMInbox","RawPMUnread","RawPMSent",
									"RawCaptchaQ","RawCaptchaCreate","RawCaptchaRetrieve","RawSubredditInformation","RawFullnameInfo",
									"RawUsernameAbout","RawUsernameOverview","RawUsernameUpvoted","RawUsernameDownvoted","RawUsernameSubmitted","RawUsernameComments",
									"RawUsernameHidden","RawUsernameSaved","RawUsernameGilded","RawUsernameTrophies",
									"RawSubreddit","RawSubredditHot","RawSubredditTop","RawSubredditNew","RawSubredditControversial","RawSubredditRising",
									"RawSubredditAbout","RawSubredditSidebar","RawSubredditSticky","RawSubredditSearch","RawPostComments"},
		"RawPosts"				-> {"RawFlairSelector","RawSubredditComment"},
		"Scope"					-> {"identity,edit,flair,history,modconfig,modflair,modlog,modposts,modwiki,mysubreddits,privatemessages,read,report,save,submit,subscribe,vote,wikiedit,wikiread"},
		"Information"			-> "Import Reddit data to the Wolfram Language"
    }
]


(* Auxiliar Functions *)

userQ[id_,user_]:= Block[{raw},
	raw = OAuthClient`rawoauthdata[id,"RawUsernameAbout", {"username"->user}];
	!MissingQ[Lookup[Lookup[redditjson[Last@raw], "data", {}], "name"]]
]

defaultuser[id_]:=Lookup[redditjson@(Last@OAuthClient`rawoauthdata[id,"RawAccountInformation", {}]),"name"]

fullnameQ[id_,fullname_]:= !MissingQ[Lookup[Lookup[Lookup[ImportString[Last@OAuthClient`rawoauthdata[id,"RawFullnameInfo", {"id" -> fullname}],"JSON"], "data"], "children"], "name"]]

subredditQ[id_,sr_]:= (redditjson[Last@OAuthClient`rawoauthdata[id,"RawSubredditAbout", {"subreddit" -> sr}]]["kind"]==="t5")

(** Import function**)

redditimport[rawdata_]:=Module[{},
	If[rawdata=!="",rawdata,Missing["NotAvailable"]]
	]

redditjson[""]:= Missing["NotAvailable"]

redditjson[rawdata_]:= With[{res = Quiet[Developer`ReadRawJSONString[rawdata]]},
	If[ AssociationQ[res] || ListQ[res],
		res,
		Message[ServiceExecute::serror];
		Throw[$Failed]
	]
]

(* Raw *)

redditdata["RawAccountInformation"] := {
        "URL"				-> "https://oauth.reddit.com/api/v1/me",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawAccountKarma"] := {
        "URL"				-> "https://oauth.reddit.com/api/v1/me/karma",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawAccountPreferences"] := {
        "URL"				-> "https://oauth.reddit.com/api/v1/me/prefs",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"after","before","count","limit","show","sr_detail"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawAccountTrophies"] := {
        "URL"				-> "https://oauth.reddit.com/api/v1/me/trophies",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawAccountFriendsList"] := {
        "URL"				-> "https://oauth.reddit.com/api/v1/me/friends",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawPMInbox"] := {
        "URL"				-> "https://oauth.reddit.com/message/inbox",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"mark","mid","after","before","count","limit","show","sr_detail"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawPMUnread"] := {
        "URL"				-> "https://oauth.reddit.com/message/unread",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"mark","mid","after","before","count","limit","show","sr_detail"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawPMSent"] := {
        "URL"				-> "https://oauth.reddit.com/message/sent",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"mark","mid","after","before","count","limit","show","sr_detail"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawCaptchaQ"] := {
        "URL"				-> "https://oauth.reddit.com/api/needs_captcha",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawCaptchaCreate"] := {
        "URL"				-> "https://oauth.reddit.com/api/new_captcha",
        "HTTPSMethod"		-> "POST",
        "Parameters"		-> {"api_type"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawCaptchaRetrieve"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/captcha/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "PathParameters"	-> {"id"},
        "RequiredParameters"-> {"id"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawFullnameInfo"] := {
        "URL"				-> "https://oauth.reddit.com/api/info",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"id","url"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditInformation"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/api/info", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawFlairList"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/api/flairlist", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"name","limit","after","before","count","sr_detail"},
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawFlairSelector"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/api/flairselector", #]&),
        "HTTPSMethod"		-> "POST",
        "Parameters"		-> {"name","link"},
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameAbout"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/about", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameOverview"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/overview", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameSubmitted"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/submitted", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameComments"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/comments", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameUpvoted"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/upvoted", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameDownvoted"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/downvoted", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameHidden"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/hidden", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameSaved"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/saved", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameGilded"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/user/`1`/gilded", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "Parameters"		-> {"show", "sort", "t", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawUsernameTrophies"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/api/v1/user/`1`/trophies", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"username"},
        "RequiredParameters"-> {"username"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubreddit"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditRising"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/rising", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "Parameters"		-> {"show","limit","after","before","count","sr_detail"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}


redditdata["RawSubredditHot"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/hot", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "Parameters"		-> {"t","show","limit","after","before","count","sr_detail"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditTop"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/top", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "Parameters"		-> {"show","limit","after","before","count","sr_detail"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditNew"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/new", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "Parameters"		-> {"show","limit","after","before","count","sr_detail"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditControversial"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/controversial", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "Parameters"		-> {"t","show","limit","after","before","count","sr_detail"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditAbout"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/about", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditSidebar"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/sidebar", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditSticky"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/r/`1`/sticky", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"subreddit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditSearch"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/subreddits/search", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"show", "q", "after", "before", "count", "limit"},
        "RequiredParameters"-> {"subreddit"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawSubredditComment"] := {
        "URL"				-> "https://oauth.reddit.com/api/comment",
        "HTTPSMethod"		-> "POST",
        "Parameters"		-> {"parent","text","api_type"},
        "PathParameters"	-> {},
        "RequiredParameters"-> {"parent","text","api_type"},
        "ResultsFunction"	-> redditimport
}

redditdata["RawPostComments"] := {
        "URL"				-> (ToString@StringForm["https://oauth.reddit.com/comments/`1`/", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"article"},
        "Parameters"		-> {"comment", "context", "depth", "showedits", "showmore", "limit", "sort"},
        "RequiredParameters"-> {"article"},
        "ResultsFunction"	-> redditimport
}

(* Cooked functions *)

redditcookeddata["AccountData",id_, {}] := redditcookeddata["UserData",id,{}]

redditcookeddata["UserData",id_, args_?OptionQ] :=  Block[{invalidParameters,data,raw,error,user,single=False},

	invalidParameters = Select[Keys[args],!MemberQ[{"User"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	Switch[KeyExistsQ[args,"User"],
			True,
				user = Lookup[args,"User"];
				If[StringQ[user],single=True;user={user}];
				Switch[user,
						{_String..},
							data = (Block[{raw0,predata},
								raw0 = OAuthClient`rawoauthdata[id,"RawUsernameAbout", {"username"->#}];
								Switch[First@raw0,
											200,
												predata = redditjson@(Last@raw0);
												RFormatUserInformation@predata["data"],
											_,
												RFormatUserInformation["Template",#]
								]]&/@user);
							If[single,
								Dataset[First@data],
								Dataset[data]
							],
						_,
						Message[ServiceExecute::nval,"User","Reddit"];
						Throw[$Failed]
				],
					
			False,
				raw = OAuthClient`rawoauthdata[id,"RawAccountInformation", {}];
				Switch[First@raw,
							200,
								data = redditjson@(Last@raw);
								Dataset[RFormatUserInformation@data],
							_,
								error = Lookup[redditjson@(Last@raw),"error"];
								Message[ServiceExecute::serrormsg,error];
								Throw[$Failed]
				]
	]
]

redditcookeddata["AccountKarma",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,predata,data,sr,single=False},

	invalidParameters = Select[Keys[args],!MemberQ[{"Subreddit"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	Switch[KeyExistsQ[args,"Subreddit"],
			True,
				sr = Lookup[args,"Subreddit"];
				If[StringQ[sr],sr={sr};single=True];
				Switch[sr,
					{_String..},
						raw = OAuthClient`rawoauthdata[id,"RawAccountKarma", {}];
						Switch[First@raw,
									200,
										predata = redditjson@(Last@raw);
										data = RFormatKarma/@Select[predata["data"], (MemberQ[sr, Lookup[#, "sr"]])&],
									_,
										Message[ServiceExecute::serror];
										Throw[$Failed]
						],
					_,
						Message[ServiceExecute::nval,"Subreddit","Reddit"];
						Throw[$Failed]
				];
				If[single,
					Dataset[SelectFirst[data,True&,{}]],
					Dataset[Take[data,UpTo[Length[sr]]]]
				]
				,

			False,
				raw = OAuthClient`rawoauthdata[id,"RawAccountKarma", {}];
				Switch[First@raw,
							200,
								data = redditjson@(Last@raw);
								Dataset[RFormatKarma/@(data["data"])],
							_,
								Message[ServiceExecute::serror];
								Throw[$Failed]
				]
	]
]

redditcookeddata["UserTrophies",id_, args_?OptionQ] :=  Block[{invalidParameters,data,error,user,st,elems,single=False},

	invalidParameters = Select[Keys[args],!MemberQ[{"User", "Elements", "ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		],
		st = True
	];
	
	elems = Lookup[args,"Elements","FullData"];
	
	Switch[elems,
				"Icon",
					Switch[st,
						True,
							Switch[KeyExistsQ[args,"User"],
								True,
									user = Lookup[args,"User"];
									If[StringQ[user],single=True;user={user}];
									data = 
										(Block[{raw,predata,iconlist,iconrules},
										raw = OAuthClient`rawoauthdata[id,"RawUsernameTrophies", {"username"->#}];
										Switch[First@raw,
													200,
														predata = redditjson@(Last@raw);
														predata = Lookup[predata["data"]["trophies"], "data",{}];
														iconlist = Lookup[predata, "icon_70",{}];
														iconrules = (# -> If[# =!= "", Import@#, Missing["NotAvailable"]] &) /@ DeleteDuplicates[iconlist];
														Replace[iconlist,iconrules,{1}],
													_,
														error = Lookup[redditjson@(Last@raw),"explanation"];
														Message[ServiceExecute::serrormsg,error];
														Throw[$Failed]
										]]&/@user);
									If[single,
										First@data,
										Association[Thread[user->data]]
									],
								False,
									Block[{raw,predata,iconlist,iconrules},
										raw = OAuthClient`rawoauthdata[id,"RawAccountTrophies", {}];
										Switch[First@raw,
													200,
														data = redditjson@(Last@raw);
														data = Lookup[data["data"]["trophies"], "data",{}];
														iconlist = Lookup[data, "icon_70",{}];
														iconrules = (# -> If[# =!= "", Import@#, Missing["NotAvailable"]] &) /@ DeleteDuplicates[iconlist];
														Replace[iconlist,iconrules,{1}],
													_,
														Message[ServiceExecute::serror];
														Throw[$Failed]
										]
									]
								]
							,
						False,
							Switch[KeyExistsQ[args,"User"],
								True,
									user = Lookup[args,"User"];
									If[StringQ[user],single=True;user={user}];
									data = 
										(Block[{raw,predata,iconlist},
										raw = OAuthClient`rawoauthdata[id,"RawUsernameTrophies", {"username"->#}];
										Switch[First@raw,
													200,
														predata = redditjson@(Last@raw);
														predata = Lookup[predata["data"]["trophies"], "data",{}];
														iconlist = Lookup[predata, "icon_70",{}],
													_,
														error = Lookup[redditjson@(Last@raw),"explanation"];
														Message[ServiceExecute::serrormsg,error];
														Throw[$Failed]
										]]&/@user);
									If[single,
										First@data,
										Association[Thread[user->data]]
									],
								False,
								Block[{raw,predata,iconlist},
									raw = OAuthClient`rawoauthdata[id,"RawAccountTrophies", {}];
									Switch[First@raw,
												200,
													predata = redditjson@(Last@raw);
													predata = Lookup[predata["data"]["trophies"], "data",{}];
													iconlist = Lookup[predata, "icon_70",{}],
												_,
													Message[ServiceExecute::serror];
													Throw[$Failed]
									]
								]
							]
						],
				"FullData",	
					Switch[st,
							True,
								Switch[KeyExistsQ[args,"User"],
										True,
											user = Lookup[args,"User"];
											If[StringQ[user],single=True;user={user}];
											data = 
												(Block[{raw,predata,iconlist,iconrules,awardidlist},
												raw = OAuthClient`rawoauthdata[id,"RawUsernameTrophies", {"username"->#}];
												Switch[First@raw,
															200,
																predata = redditjson@(Last@raw);
																predata = Lookup[predata["data"]["trophies"], "data",{}];
																awardidlist = Lookup[predata, "award_id",{}];
																iconlist = Lookup[predata, "icon_70",{}];
																iconrules = Replace[DeleteDuplicates[Thread[Rule[awardidlist, iconlist]]], (ic_ -> icon_) :> (ic -> If[icon=!= "",Import@icon,Missing["NotAvailable"]]), 1];
																RFormatTrophy[#,iconrules]&/@predata,
															_,
																error = Lookup[redditjson@(Last@raw),"explanation"];
																Message[ServiceExecute::serrormsg,error];
																Throw[$Failed]
												]]&/@user);
											If[single,
												Dataset[First@data],
												Dataset[Association[Thread[user->data]]]
											],
										False,
											Block[{raw,predata,iconlist,iconrules,awardidlist},
											raw = OAuthClient`rawoauthdata[id,"RawAccountTrophies", {}];
											Switch[First@raw,
														200,
															predata = redditjson@(Last@raw);
															predata = Lookup[predata["data"]["trophies"], "data",{}];
															awardidlist = Lookup[predata, "award_id",{}];
															iconlist = Lookup[predata, "icon_70",{}];
															iconrules = Replace[DeleteDuplicates[Thread[Rule[awardidlist, iconlist]]], (ic_ -> icon_) :> (ic -> If[icon=!= "",Import@icon,Missing["NotAvailable"]]), 1];
															Dataset[RFormatTrophy[#,iconrules]&/@predata],
														_,
															Message[ServiceExecute::serror];
															Throw[$Failed]
											]]
								],
							False,
								Switch[KeyExistsQ[args,"User"],
										True,
											user = Lookup[args,"User"];
											If[StringQ[user],single=True;user={user}];
											data = 
												(Block[{raw,predata},
												raw = OAuthClient`rawoauthdata[id,"RawUsernameTrophies", {"username"->#}];
												Switch[First@raw,
															200,
																predata = redditjson@(Last@raw);
																RFormatTrophy/@Lookup[predata["data"]["trophies"], "data",{}],
															_,
																error = Lookup[redditjson@(Last@raw),"explanation"];
																Message[ServiceExecute::serrormsg,error];
																Throw[$Failed]
												]]&/@user);
											If[single,
												Dataset[First@data],
												Dataset[Association[Thread[user->data]]]
											],
										False,
											Block[{raw,predata},
											raw = OAuthClient`rawoauthdata[id,"RawAccountTrophies", {}];
											Switch[First@raw,
														200,
															predata = redditjson@(Last@raw);
															Dataset[RFormatTrophy/@Lookup[predata["data"]["trophies"], "data",{}]],
														_,
															Message[ServiceExecute::serror];
															Throw[$Failed]
											]]
								]
						],
			_,
				Message[ServiceExecute::nval,"Elements","Reddit"];
				Throw[$Failed]
	]
]

redditcookeddata["AccountFriends",id_, args_?OptionQ] :=  Block[{invalidParameters,raw,data},

	invalidParameters = Select[Keys[args],!MemberQ[{},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

		raw = OAuthClient`rawoauthdata[id,"RawAccountFriendsList", {}];
		Switch[First@raw,
					200,
						data = redditjson@(Last@raw);
						Dataset[RFormatFriendsList/@(data["data"]["children"])],
					_,
						Message[ServiceExecute::serror];
						Throw[$Failed]
		]
]

redditcookeddata["UserActivity",id_, args_?OptionQ] :=  Block[{params = <||>,invalidParameters,user,sort,start=1,type,tmp,tmpdata={},limit=25,count,rest,varid,rqfun,flag=0,predata,data,raw,error,single=True,st},

	invalidParameters = Select[Keys[args],!MemberQ[{"Type","User","SortBy",MaxItems,"StartIndex","Single","ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];
	
	If[KeyExistsQ[args,"SortBy"],
		sort = Lookup[args,"SortBy"];
		If[MemberQ[{"New","Hot","Top","Controversial"},sort],
			AppendTo[params,"sort"->ToLowerCase@sort],
			Message[ServiceExecute::nval,"SortBy","Reddit"];
			Throw[$Failed]
		],
		sort = "new";
		AppendTo[params,"sort"->sort];
	];

	If[KeyExistsQ[args,"Type"],
		type = Lookup[args,"Type"];
		If[!MemberQ[{"Overview","Submitted","Comments","Upvoted","Downvoted","Hidden","Saved","Gilded"},type],
			Message[ServiceExecute::nval,"Type","Reddit"];
			Throw[$Failed]
		],
		type = "Overview"
	];

	Switch[type,
			"Overview",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameOverview", Normal@#]&)],
			"Submitted",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameSubmitted", Normal@#]&)],
			"Comments",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameComments", Normal@#]&)],
			"Upvoted",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameUpvoted", Normal@#]&)],
			"Downvoted",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameDownvoted", Normal@#]&)],
			"Hidden",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameHidden", Normal@#]&)],
			"Saved",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameSaved", Normal@#]&)],
			"Gilded",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawUsernameGilded", Normal@#]&)]
	];

	If[KeyExistsQ[args,"Single"],
		single = Lookup[args,"Single"]
	];

	If[KeyExistsQ[args,"User"],
		user = Lookup[args,"User"];
		If[TrueQ[single],
			If[!MatchQ[user,_String|{_String..}],
				Message[ServiceExecute::nval,"User","Reddit"];
				Throw[$Failed]
			]
		]
		,
		user = defaultuser[id];
	];

	Switch[user,
		{_String..},
			data = Dataset[redditcookeddata["UserActivity",id, Join[FilterRules[args,Except["User"]],{"Single"->False,"User"->#}]]&/@user],
		_String,
			If[userQ[id,user],
			data = <|"User"->user,"ActivityList"->#|>&@
				(AppendTo[params,"username"->user];
				
				If[KeyExistsQ[args,"StartIndex"],
					start = Lookup[args,"StartIndex"];
					If[!(IntegerQ[start]&&start>0),
						Message[ServiceExecute::nval,"StartIndex","Reddit"];
						Throw[$Failed]
					]
				];
			
				If[KeyExistsQ[args,MaxItems],
					limit = Lookup[args,MaxItems];
					If[!(IntegerQ[limit]&&limit>0),
						Message[ServiceExecute::nval,MaxItems,"Reddit"];
						Throw[$Failed]
					];
				];
			
				count = Quotient[limit+start-1,100,1];
				rest = ToString@Mod[limit+start-1,100,1];
				AppendTo[params,"limit"->"100"];

				Do[
					raw = rqfun@params;
					Switch[First@raw,
							200,
								predata = redditjson@(Last@raw);
								tmp = RFormatFullname/@predata["data"]["children"];
								tmpdata = Join[tmpdata,tmp];
								If[Length[tmp]<100,flag=1;Break[]];
								varid = (Last@tmp)["Name"];
								If[!MissingQ[varid],AppendTo[params,"after"->varid],flag=1;Break[]],
							403,
								Message[ServiceExecute::rdmod];
								Throw[$Failed],
							_,
								error = Lookup[redditjson@(Last@raw),"error"];
								Message[ServiceExecute::serrormsg,error];
								Throw[$Failed]]
				,count];
	
				Switch[flag,
						0,
							AppendTo[params,"limit"->rest];
							raw = rqfun@params;
								Switch[First@raw,
										200,
											predata = redditjson@(Last@raw);
											tmp = RFormatFullname/@(predata["data"]["children"]);
											tmpdata = Join[tmpdata,tmp];
											If[Length[tmpdata]>(start-1),
												Take[tmpdata,{start,UpTo[limit+start-1]}],
												{}
											],
										403,
											Message[ServiceExecute::rdmod];
											Throw[$Failed],
										_,
											error = Lookup[redditjson@(Last@raw),"error"];
											Message[ServiceExecute::serrormsg,error];
											Throw[$Failed]],
						1,
							If[Length[tmpdata]>(start-1),
								Take[tmpdata,{start,UpTo[limit+start-1]}],
								{}
							]
				]),
			data = <|"User"->user,"ActivityList"->Missing["InvalidUsername"]|>
			]
	];

	If[TrueQ[single],
		Dataset[data],
		data
	]

]

redditcookeddata["GlobalIDInformation",id_, args_?OptionQ] :=  Block[{invalidParameters,fullname,fnlist,fnlist2,fnpos,datalist,placeholder,data,raw,error,st},

	invalidParameters = Select[Keys[args],!MemberQ[{"GlobalID","ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];

	If[KeyExistsQ[args,"GlobalID"],
		fnlist = Lookup[args,"GlobalID"];
		If[StringQ[fnlist],fnlist={fnlist}];
		Switch[fnlist,
				{_String..},
					Which[
							StringMatchQ[#, ("t2" | "t4" | "t6" | "t8") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn2];
								Throw[$Failed],
							!StringMatchQ[#, ("t1" | "t3" | "t5") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn,#];
								Throw[$Failed]
						]&/@fnlist;
					fnlist2 = Partition[fnlist, UpTo[50]];
					fullname = StringJoin@Riffle[#, ","]&/@fnlist2,
				_,
				Message[ServiceExecute::nval,"GlobalID","Reddit"];
				Throw[$Failed]
			],
		Message[ServiceExecute::nparam,"GlobalID","Reddit"];
		Throw[$Failed]
	];

	data = Flatten[(raw = OAuthClient`rawoauthdata[id,"RawFullnameInfo", {"id"->#}];
	
	Switch[First@raw,
						200,
							RFormatFullname/@Lookup[Lookup[redditjson@(Last@raw), "data",{}], "children",{}],
						_,
							error = Lookup[redditjson@(Last@raw),"error"];
							Message[ServiceExecute::serrormsg,error];
							Throw[$Failed]
		])&/@fullname,1];

	placeholder = DeleteDuplicates@Flatten[Keys[data]];
	datalist = Lookup[data,"GlobalID"];
	fnpos = Flatten@Position[fnlist, Except[_?(MemberQ[datalist, #] &)], Heads -> False];
	Dataset[Fold[Insert[#1, AssociationMap[(Missing["NonExistent"]) &, placeholder], #2] &, data, fnpos]]
]

redditcookeddata["PrivateMessages",id_, args_?OptionQ] :=  Block[{invalidParameters,params=<||>,source,start=1,limit=25,tmp,tmpdata={},count,rest,varid,rqfun,flag=0,data,raw,error,st},

	invalidParameters = Select[Keys[args],!MemberQ[{"Source",MaxItems,"StartIndex","ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];

	If[KeyExistsQ[args,"Source"],
		source = Lookup[args,"Source"];
		Switch[MemberQ[{"Inbox","Unread","Sent"},source],
				True,
					Null,
				False,
				Message[ServiceExecute::nval,"Source","Reddit"];
				Throw[$Failed]
			],
		Message[ServiceExecute::nparam,"Source","Reddit"];
		Throw[$Failed]
	];

	Switch[source,
			"Inbox",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPMInbox", Normal@#]&)],
			"Unread",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPMUnread", Normal@#]&)],
			"Sent",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPMSent", Normal@#]&)]
	];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","Reddit"];
			Throw[$Failed]
		]
	];

	If[KeyExistsQ[args,MaxItems],
		(
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"Reddit"];
				Throw[$Failed]
			];
		)
	];

	count = Quotient[limit+start-1,100,1];
	rest = ToString@Mod[limit+start-1,100,1];
	AppendTo[params,"limit"->"100"];

	Do[
		raw = rqfun@params;
		Switch[First@raw,
				200,
					data = redditjson@(Last@raw);
					data = MapAt[("t4") &, data["data"]["children"],Position[data["data"]["children"], "t1"]];
					tmp = RFormatFullname/@data;
					tmpdata = Join[tmpdata,tmp];
					If[Length[tmp]<100,flag=1;Break[]];
					varid = (Last@tmp)["GlobalID"];
					If[!MissingQ[varid],AppendTo[params,"after"->varid],flag=1;Break[]],
				_,
					error = Lookup[redditjson@(Last@raw),"error"];
					Message[ServiceExecute::serrormsg,error];
					Throw[$Failed]]
	,count];

	Switch[flag,
			0,
				AppendTo[params,"limit"->rest];
				raw = rqfun@params;
					Switch[First@raw,
							200,
								data = redditjson@(Last@raw);
								data = MapAt[("t4") &, data["data"]["children"],Position[data["data"]["children"], "t1"]];
								tmp = RFormatFullname/@data;
								tmpdata = Join[tmpdata,tmp];
								If[Length[tmpdata]>(start-1),
									Dataset[SortBy[(AbsoluteTime[#["CreationDate"]]&)][Take[tmpdata,{start,UpTo[limit+start-1]}]]],
									Dataset[<||>]
								],
							_,
								error = Lookup[redditjson@(Last@raw),"error"];
								Message[ServiceExecute::serrormsg,error];
								Throw[$Failed]],
			1,
				If[Length[tmpdata]>(start-1),
					Dataset[SortBy[(AbsoluteTime[#["CreationDate"]]&)][Take[tmpdata,{start,UpTo[limit+start-1]}]]],
					Dataset[<||>]
				]
	]
]

redditcookeddata["SubredditFlair",id_, args_?OptionQ] :=  Block[{invalidParameters,params=<||>,user,sr,tmp,tmpdata={},start=1,limit=25,count,rest,varid,rqfun,flag=0,data,raw,error},

	invalidParameters = Select[Keys[args],!MemberQ[{"Subreddit","User","StartIndex",MaxItems},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"User"] && (KeyExistsQ[args,"StartIndex"]||KeyExistsQ[args,MaxItems]),
		Message[ServiceExecute::rduflo];
		Throw[$Failed];
	];

	If[KeyExistsQ[args,"Subreddit"],
		sr = Lookup[args,"Subreddit"];
		If[StringQ[sr],
			AppendTo[params,"subreddit"->sr],
			Message[ServiceExecute::nval,"User","Reddit"];
			Throw[$Failed]
		],
		Message[ServiceExecute::nparam,"Subreddit","Reddit"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"User"],
			user = Lookup[args,"User"];
			If[StringQ[user] && userQ[id,user],
				AppendTo[params,"name"->user],
				Message[ServiceExecute::nval,"User","Reddit"];
				Throw[$Failed]
			]
	];
	
	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&start>0),
			Message[ServiceExecute::nval,"StartIndex","Reddit"];
			Throw[$Failed]
		]
	];

	If[KeyExistsQ[args,MaxItems],
		(
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&limit>0),
				Message[ServiceExecute::nval,MaxItems,"Reddit"];
				Throw[$Failed]
			];
		)
	];
	
	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawFlairList", Normal@#]&)];
	
	count = Quotient[limit+start-1,1000,1];
	rest = ToString@Mod[limit+start-1,1000,1];
	AppendTo[params,"limit"->"1000"];

	Do[
		raw = rqfun@params;
		Switch[First@raw,
				200,
					data = redditjson@(Last@raw);
						If[KeyExistsQ[data,"users"],
							tmp = RFormatFlair/@Lookup[data, "users"];
							tmpdata = Join[tmpdata,tmp];
							If[!KeyExistsQ[data,"next"],flag=1;Break[]];
							varid = Lookup[data, "next"]; AppendTo[params,"after"->varid],
						Message[ServiceExecute::rdivsr,sr];
						Throw[$Failed]
						],
				403,
					Message[ServiceExecute::rdmod];
					Throw[$Failed],
				_,
					error = Lookup[redditjson@(Last@raw),"error"];
					Message[ServiceExecute::serrormsg,error];
					Throw[$Failed]]
	,count];

	Switch[flag,
			0,
				AppendTo[params,"limit"->rest];
				raw = rqfun@params;
					Switch[First@raw,
							200,
								data = redditjson@(Last@raw);
								If[KeyExistsQ[data,"users"],
									tmp = RFormatFlair/@Lookup[data, "users"];
									tmpdata = Join[tmpdata,tmp];
									If[Length[tmpdata]>(start-1),
										Dataset[tmpdata[[start;;-1]]],
										Dataset[<||>]
									],
								Message[ServiceExecute::rdivsr,sr];
								Throw[$Failed]
								],
							403,
								Message[ServiceExecute::rdmod];
								Throw[$Failed],
							_,
								error = Lookup[redditjson@(Last@raw),"error"];
								Message[ServiceExecute::serrormsg,error];
								Throw[$Failed]],
			1,
				If[Length[tmpdata]>(start-1),
					Dataset[tmpdata[[start;;-1]]],
					Dataset[<||>]
				]
	]
]

redditcookeddata["SubredditFlairAlternatives",id_, args_?OptionQ] :=  Block[{invalidParameters,params={},sr,user,link,raw,data,error},

	invalidParameters = Select[Keys[args],!MemberQ[{"Subreddit","User","Fullname"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"User"] && KeyExistsQ[args,"Fullname"],
		Message[ServiceExecute::rduflo];
		Throw[$Failed];
	];

	If[KeyExistsQ[args,"Subreddit"],
		sr = Lookup[args,"Subreddit"];
		If[StringQ[sr],
			AppendTo[params,"subreddit"->sr],
			Message[ServiceExecute::nval,"User","Reddit"];
			Throw[$Failed]
		],
		Message[ServiceExecute::nparam,"Subreddit","Reddit"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"User"],
			user = Lookup[args,"User"];
			If[StringQ[user] && userQ[id,user],
				AppendTo[params,"name"->user],
				Message[ServiceExecute::nval,"User","Reddit"];
				Throw[$Failed]
			]
	];
	
	If[KeyExistsQ[args,"Fullname"],
			link = Lookup[args,"Fullname"];
			If[StringQ[link] && fullnameQ[id,link],
				AppendTo[params,"link"->link],
				Message[ServiceExecute::nval,"Fullname","Reddit"];
				Throw[$Failed]
			]
	];

	raw = OAuthClient`rawoauthdata[id,"RawFlairSelector", params];
	
	Switch[First@raw,
				200,
					data = redditjson@(Last@raw);
					Dataset[Association@ReplacePart[MapAt[RFormatFlair, data, Position[data, {__Rule}, {2, 3}]],
						{{1, 1} -> "Current", {2, 1} -> "Alternatives"}]],
				403,
					Message[ServiceExecute::rdmod];
					Throw[$Failed],
				411,
					Message[ServiceExecute::rdivsr,sr];
					Throw[$Failed],
				_,
					error = Lookup[redditjson@(Last@raw),"error"];
					Message[ServiceExecute::serrormsg,error];
					Throw[$Failed]
	]
]

redditcookeddata["PostInformation",id_, args_?OptionQ] :=  Block[{invalidParameters,st,fullname,pslist,pslist2,fnpos,nodata,datalist,placeholder,data,raw,error,single=False},

	invalidParameters = Select[Keys[args],!MemberQ[{"Post","ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];

	If[KeyExistsQ[args,"Post"],
		pslist = Lookup[args,"Post"];
		If[StringQ[pslist],single=True;pslist={pslist}];
		Switch[pslist,
				{_String..},
					Which[
							urlQ[#],
								Null,
							StringMatchQ[#, ("t1" | "t2" | "t4" | "t6" | "t5" | "t8") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn3,"PostInformation"];
								Throw[$Failed],
							!StringMatchQ[#, ("t3") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn,#];
								Throw[$Failed]
						]&/@pslist;
					pslist = RParsePostFromURL/@pslist;
					pslist2 = Partition[pslist, UpTo[50]];
					fullname = StringJoin@Riffle[#, ","]&/@pslist2,
				_,
				Message[ServiceExecute::nval,"Post","Reddit"];
				Throw[$Failed]
			],
		Message[ServiceExecute::nparam,"Post","Reddit"];
		Throw[$Failed]
	];

	data = Flatten[(raw = OAuthClient`rawoauthdata[id,"RawFullnameInfo", {"id"->#}];
	
	Switch[First@raw,
				200,
					RFormatFullname/@Lookup[Lookup[redditjson@(Last@raw), "data",{}], "children",{}],
				_,
					error = Lookup[redditjson@(Last@raw),"error"];
					Message[ServiceExecute::serrormsg,error];
					Throw[$Failed]
		])&/@fullname,1];

	placeholder = DeleteDuplicates@Flatten[Keys[data]];
	datalist = Lookup[data,"GlobalID",{}];
	nodata = Complement[pslist,datalist];
	fnpos = Flatten@Position[pslist, Except[_?(MemberQ[datalist, #] &)], Heads -> False];
	If[single,
		Dataset[First@Fold[Insert[#1, With[{pos=#2},AssociationMap[If[#=!="GlobalID",Missing["InvalidPost"],pslist[[pos]]]&, placeholder]], #2] &, data, fnpos]],
		Dataset[Fold[Insert[#1, With[{pos=#2},AssociationMap[If[#=!="GlobalID",Missing["InvalidPost"],pslist[[pos]]]&, placeholder]], #2] &, data, fnpos]]
	]
]

redditcookeddata["SubredditInformation",id_, args_?OptionQ] :=  Block[{invalidParameters,st,raw,data,sr,rqfun,single=False},

	invalidParameters = Select[Keys[args],!MemberQ[{"Subreddit","ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];

	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawSubredditAbout", "subreddit"->#]&)];

	Switch[KeyExistsQ[args,"Subreddit"],
			True,
				sr = Lookup[args,"Subreddit"];
				If[StringQ[sr],single=True;sr={sr}];
				Switch[sr,
					{_String..},
						Null,
					_,
					Message[ServiceExecute::nval,"Subreddit","Reddit"];
					Throw[$Failed]
				];
				data = (Block[{name = #},raw = redditjson@(Last@(rqfun@name));
						If[KeyExistsQ[raw[ "data"], "name"],
							RFormatFullname@raw,
							<||>
						]]&/@sr);
				If[single,
					Dataset[First@data],
					Dataset[data]
				],

			False,
				Message[ServiceExecute::nparam,"Subreddit","Reddit"];
				Throw[$Failed]
	]
]

redditcookeddata["SubredditPosts",id_, args_?OptionQ] :=  Block[{invalidParameters,params=<||>,sort,raw,count,limit=25,rest,start=1,predata,tmp,tmpdata={},data,sr,rqfun,single=True,flag=0,st,varid,missing},

	invalidParameters = Select[Keys[args],!MemberQ[{"Subreddit",MaxItems,"SortBy","StartIndex","ShowThumbnails","Single"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Single"],
		single = Lookup[args,"Single"]
	];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];
	
	If[KeyExistsQ[args,"SortBy"],
		sort = Lookup[args,"SortBy"];
		If[!MemberQ[{"Hot","Top","New","Rising","Controversial"},sort],
				Message[ServiceExecute::nval,"SortBy","Reddit"];
				Throw[$Failed]
		],
		sort = "Hot"
	];
	
	Switch[sort,
			"Hot",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawSubredditHot", Normal@#]&)],
			"Top",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawSubredditTop", Normal@#]&)],
			"New",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawSubredditNew", Normal@#]&)],
			"Rising",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawSubredditRising", Normal@#]&)],
			"Controversial",
				rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawSubredditControversial", Normal@#]&)]
	];

	Switch[KeyExistsQ[args,"Subreddit"],
			True,
				sr = Lookup[args,"Subreddit"];
				Switch[sr,
					{_String..},
						data = Dataset[redditcookeddata["SubredditPosts",id, Join[FilterRules[args,Except["Subreddit"]],{"Single"->False,"Subreddit"->#}]]&/@sr],
					_String,
						data = <|"Subreddit"->sr,"Posts"->#|>&@
						(AppendTo[params,"subreddit"->sr];
						
						If[KeyExistsQ[args,"StartIndex"],
							start = Lookup[args,"StartIndex"];
							If[!(IntegerQ[start]&&start>0),
								Message[ServiceExecute::nval,"StartIndex","Reddit"];
								Throw[$Failed]
							]
						];
					
						If[KeyExistsQ[args,MaxItems],
							(
								limit = Lookup[args,MaxItems];
								If[!(IntegerQ[limit]&&limit>0),
									Message[ServiceExecute::nval,MaxItems,"Reddit"];
									Throw[$Failed]
								];
							)
						];
					
						count = Quotient[limit+start-1,100,1];
						rest = ToString@Mod[limit+start-1,100,1];
						AppendTo[params,"limit"->"100"];
					
						Do[
							raw = redditjson@(Last@(rqfun@params));
							If[Lookup[Lookup[Lookup[raw, "data"], "children"], "kind", None]=!=None,
								tmp = RFormatFullname/@Lookup[Lookup[raw, "data"], "children"],
								missing = True, flag=1; Break[]
							];
							tmpdata = Join[tmpdata,tmp];
							If[Length[tmp]<100,flag=1;Break[]];
							varid = Lookup[raw["data"],"after"];
							If[!MissingQ[varid],AppendTo[params,"after"->varid],flag=1;Break[]]
						,count];
						
						Switch[flag,
								0,
									AppendTo[params,"limit"->rest];
									raw = redditjson@(Last@(rqfun@params));
									If[Lookup[Lookup[Lookup[raw, "data"], "children"], "kind", None]=!=None,
										tmp = RFormatFullname/@Lookup[Lookup[raw, "data"], "children"],
										missing = True
									];
									tmpdata = Join[tmpdata,tmp];
										Which[
												TrueQ[missing],
													{<||>},
												Length[tmpdata]>(start-1),
													Take[tmpdata,{start,UpTo[limit+start-1]}],
												_,
													{}
											],
								1,
									Which[
											TrueQ[missing],
												{},
											Length[tmpdata]>(start-1),
												Take[tmpdata,{start,UpTo[limit+start-1]}],
											_,
												{}
										]
						]),
					_,
					Message[ServiceExecute::nval,"Subreddit","Reddit"];
					Throw[$Failed]
				];
				If[TrueQ[single],
					Dataset[data],
					data
				],

			False,
				Message[ServiceExecute::nparam,"Subreddit","Reddit"];
				Throw[$Failed]
				]
]

redditcookeddata["PostCommentsData",id_, args_?OptionQ] :=  Block[{invalidParameters,params=<||>,depth=1,start=1,limit=25,varid,count,rest,tmpdata,tmp,post,post2,more={},morepos,rqfun,raw,raw2,raw3,pos1,pos2,predata,data,error,single=True,st,flag=0,missing},

	invalidParameters = Select[Keys[args],!MemberQ[{"Post","Depth",MaxItems,"StartIndex","ShowThumbnails","Single"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Single"],
		single = Lookup[args,"Single"]
	];

	If[KeyExistsQ[args,"Post"],
		(
			post = Lookup[args,"Post"];
			Switch[post,
					{_String..},
						Which[
							(URLParse[#,"Domain"] =!= None) && !urlQ[#],
								Message[ServiceExecute::rdivurl,#];
								Throw[$Failed],
							urlQ[#],
								Null,
							StringMatchQ[#, ("t1" | "t2" | "t4" | "t6" | "t5" | "t8") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn3,"PostCommentsData"];
								Throw[$Failed],
							!StringMatchQ[#, ("t3") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn,#];
								Throw[$Failed]
						]&/@post,
					_String,
						If[TrueQ[single],
						Which[
							(URLParse[post,"Domain"] =!= None) && !urlQ[post],
								Message[ServiceExecute::rdivurl,post];
								Throw[$Failed],
							urlQ[post],
								Null,
							StringMatchQ[post, ("t1" | "t2" | "t4" | "t6" | "t5" | "t8") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn3,"PostCommentsData"];
								Throw[$Failed],
							!StringMatchQ[post, ("t3") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn,post];
								Throw[$Failed]
						]],
					_,
						Message[ServiceExecute::nval,"User","Reddit"];
						Throw[$Failed]
				]
		),
		Message[ServiceExecute::nparam,"Post","Reddit"];
		Throw[$Failed]
	];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];

	If[KeyExistsQ[args,"Depth"],
			depth = Lookup[args,"Depth"];
			If[MemberQ[Range[10], depth],
				AppendTo[params,"depth"->ToString@depth],
				Message[ServiceExecute::nval,"Depth","Reddit"];
				Throw[$Failed]
			],
		AppendTo[params,"depth"->ToString@depth]
	];

	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPostComments", Normal@#]&)];

	Switch[post,
		{_String..},
			data = Dataset[redditcookeddata["PostCommentsData",id, Join[FilterRules[args,Except["Post"]],{"Single"->False,"Post"->#}]]&/@post],
		_String,
			data = 
			(
			post2 = RParsePostFromURL2@post;
			AppendTo[params,"article"->post2];

			If[KeyExistsQ[args,"StartIndex"],
				start = Lookup[args,"StartIndex"];
				If[!(IntegerQ[start]&&(501>start>0)),
					Message[ServiceExecute::nval,"StartIndex","Reddit"];
					Throw[$Failed]
				]
			];

			If[KeyExistsQ[args,MaxItems],
				(
					limit = Lookup[args,MaxItems];
					If[!(IntegerQ[limit]&&(501-start>=limit>0)),
						Message[ServiceExecute::nval,MaxItems,"Reddit"];
						Throw[$Failed]
					];
				)
			];

			AppendTo[params,"limit"->ToString@(limit+start-1)];
			raw = rqfun@params;
			tmp = Switch[First@raw,
						200,
							raw2 = redditjson@(Last@raw);
							If[Quiet[raw2[[2,-1,2,-1,"kind"]]==="more"],
								morepos = Last@Position[raw2, _Association?(MatchQ[#["kind"], "more"] &), {4}];
								more = RFormatFullname@Extract[raw2,morepos];
								raw2 = Delete[raw2,morepos]];
							pos1 = Position[raw2, _Association?(MatchQ[#["kind"], "more" | "listing" | "Listing"] &)];
							raw3 = MapAt[RFormatFullname, raw2, pos1];
							pos2 = Position[raw3, _Association?(MatchQ[#["kind"], "t1" | "t3"] &)];
							predata = MapAt[RFormatFullname, raw3, pos2];
							Association@Thread[Rule[{"PostID", "URL", "Post", "Comments", "MoreComments"},
								Replace[predata, {{posts_}, {comments___}} :> {posts["ID"], "http://www.reddit.com" <> posts["Permalink"], posts, 
									If[Length[#]>start-1,Take[#,{start,UpTo[start+limit-1]}],{}]&@{comments}, more}]]],
						404,
							If[post  === ("t3_" <> post2),
							<|"PostID" -> post2, "URL" -> Missing["NotExistent"], "Post" -> <||>, "Comments" -> {} , "MoreComments" -> {}|>,
							<|"PostID" -> post2, "URL" -> post, "Post" -> <||>, "Comments" -> {} , "MoreComments" -> {}|>],
						_,
							error = Lookup[redditjson@(Last@raw),"error"];
							Message[ServiceExecute::serrormsg,error];
							Throw[$Failed]
				]),
		_,
		Message[ServiceExecute::nval,"Post","Reddit"];
		Throw[$Failed]
	];
	If[TrueQ[single],
		Dataset[data],
		data
	]

]

redditcookeddata["CommentReplyData",id_, args_?OptionQ] :=  Block[{invalidParameters,params=<||>,depth=1,start=1,limit=25,pslist,pslist2,rqfun,data,error,single=False,st},

	invalidParameters = Select[Keys[args],!MemberQ[{"Comment","Depth",MaxItems,"StartIndex","ShowThumbnails"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Reddit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"ShowThumbnails"],
		st = Lookup[args,"ShowThumbnails"];
		If[!BooleanQ[st],
			Message[ServiceExecute::nval,"ShowThumbnails","Reddit"];
			Throw[$Failed]
		];
		RedditFunctions`ShowThumbnails = st,
		RedditFunctions`ShowThumbnails = False
	];

	If[KeyExistsQ[args,"Depth"],
			depth = Lookup[args,"Depth"];
			If[MemberQ[Range[9], depth],
				AppendTo[params,"depth"->ToString@(depth+1)],
				Message[ServiceExecute::nval,"Depth","Reddit"];
				Throw[$Failed]
			],
		AppendTo[params,"depth"->ToString@(depth+1)]
	];

	If[KeyExistsQ[args,"Comment"],
		pslist = Lookup[args,"Comment"];
		If[StringQ[pslist],single=True;pslist={pslist}];
		Switch[pslist,
				{_String..},
					Which[
							(URLParse[#,"Domain"] =!= None) && !urlQ[#],
								Message[ServiceExecute::rdivurl,#];
								Throw[$Failed],
							urlQ[#],
								Null,
							StringMatchQ[#, ("t3" | "t2" | "t4" | "t6" | "t5" | "t8") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn3,"CommentReplyData"];
								Throw[$Failed],
							!StringMatchQ[#, ("t1") ~~ "_" ~~ __],
								Message[ServiceExecute::rdtfn,#];
								Throw[$Failed]
						]&/@pslist;
					pslist2 = Which[
							ListQ[#],
								#,
							StringQ[#],
								{StringReplace[Lookup[(redditjson@(Last@OAuthClient`rawoauthdata[id,"RawFullnameInfo", "id"->("t1_"<>#)]))["data"]["children"],
									"data", {<|"link_id" -> "t3_Missing"|>}][[1, "link_id"]],"t3_"~~r__:>r],#}
							]&/@(RParseCommentFromURL/@pslist),
				_,
				Message[ServiceExecute::nval,"Comment","Reddit"];
				Throw[$Failed]
			],
		Message[ServiceExecute::nparam,"Comment","Reddit"];
		Throw[$Failed]
	];

	rqfun = With[{ID=id},(OAuthClient`rawoauthdata[ID,"RawPostComments", Normal@#]&)];

	If[KeyExistsQ[args,"StartIndex"],
		start = Lookup[args,"StartIndex"];
		If[!(IntegerQ[start]&&(501>start>0)),
			Message[ServiceExecute::nval,"StartIndex","Reddit"];
			Throw[$Failed]
		]
	];

	If[KeyExistsQ[args,MaxItems],
		(
			limit = Lookup[args,MaxItems];
			If[!(IntegerQ[limit]&&(501-start>=limit>0)),
				Message[ServiceExecute::nval,MaxItems,"Reddit"];
				Throw[$Failed]
			];
		)
	];

	data = (Block[{raw,raw2,raw3,predata,postid=First@#,commentid=Last@#,pos1,pos2},
		If[!(postid==="Missing"),
			AppendTo[params,"article"->postid];
			AppendTo[params,"comment"->commentid];
			AppendTo[params,"limit"->ToString@(limit+start-1)];
			raw = rqfun@params,
			raw = {404,None};
			postid = Missing["NonExistent"]];
			Switch[First@raw,
							200,
								raw2 = redditjson@(Last@raw);
								If[((raw2 // Last)["data"]["children"] // First)["data"]["id"] === commentid,
									pos1 = Position[raw2, _Association?(MatchQ[#["kind"], "more" | "listing" | "Listing"] &)];
									raw3 = MapAt[RFormatFullname, raw2, pos1];
									pos2 = Position[raw3, _Association?(MatchQ[#["kind"], "t1" | "t3"] &)];
									predata = MapAt[RFormatFullname, raw3, pos2];
									Association@Thread[Rule[{"PostID","CommentID", "URL", "Comment", "CommentReplies"},
										Replace[predata, {{post_}, {comments_}} :> {postid, comments["ID"], "http://www.reddit.com" <> post["Permalink"] <> commentid, 
											KeyDrop[comments,"Replies"], If[Length[#]>start-1,Take[#,{start,UpTo[start+limit-1]}],{}]&@comments["Replies"]}]]],
									<|"PostID" -> postid, "CommentID" -> commentid, "URL" -> Missing["NotExistent"], "Comment" -> <||>, "CommentReplies" -> {}|>
								],
							404,
								<|"PostID" -> postid, "CommentID" -> commentid, "URL" -> Missing["NotExistent"], "Comment" -> <||>, "CommentReplies" -> {}|>,
							_,
								error = Lookup[redditjson@(Last@raw),"error"];
								Message[ServiceExecute::serrormsg,error];
								Throw[$Failed]
			]
		]&/@pslist2);
	
	If[single,
		Dataset[First@data],
		Dataset[data]
	]
]

redditcookeddata[___]:=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return two functions to define oauthservicedata, oauthcookeddata  *)

{Reddit`Private`redditdata,Reddit`Private`redditcookeddata}
