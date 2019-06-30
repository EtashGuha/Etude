Begin["SurveyMonkey`"]

Get["SurveyMonkeyFunctions.m"]

Begin["`Private`"]

(******************************* SurveyMonkey *************************************)

(* Authentication information *)

surveymonkeydata[]:=
	If[TrueQ[OAuthClient`Private`$UseChannelFramework],{
		"OAuthVersion"			-> "2.0",
		"ServiceName" 			-> "SurveyMonkey", 
	 	"AuthorizeEndpoint" 	-> "https://api.surveymonkey.net/oauth/authorize", 
     	"AccessEndpoint"    	-> "https://api.surveymonkey.net/oauth/token",
     	"RedirectURI"       	-> "WolframConnectorChannelListen",
     	"Blocking"          	-> False,
        "RedirectURLFunction"	-> (#1&),
        "AuthorizationFunction"	-> "SurveyMonkey",
		"AccessTokenRequestor"	-> Automatic,
		"AccessTokenExtractor"	-> "JSON/2.0",
		"VerifierLabel"			-> "code",
		"VerifyPeer"			-> True,
	 	"AuthenticationDialog" 	:> "WolframConnectorChannel",
	 	"ClientInfo"			-> {"Wolfram","Token"},
	 	"RequestFormat"		-> (Block[{params = Lookup[{##2},"Parameters",{}], method = Lookup[{##2}, "Method"],
        							body = Lookup[{##2},"BodyData",""], headers = Lookup[{##2},"Headers", {}], auth},
        							auth = Lookup[params,"access_token"];
        							URLRead[HTTPRequest[#1,	<|"Headers" -> Join[{"Authorization" -> "Bearer " <> auth}, headers], 
										Method -> method, "Query" -> KeyDrop["access_token"][params], "Body" -> body,"ContentType" -> "application/json"|>],
										{"Body", "StatusCode"}, "CredentialsProvider" -> None]
        						]&),
	 	"Gets"					-> {"SurveyList","SurveyDetails","CollectorList","ResponseCounts","SurveyResponseList","CollectorResponseList",
	 								"Response","UserData","TemplateList","CategoryList","SurveyResults","SurveyResponseBulk"},
	 	"Posts"					-> {"CreateWeblinkCollector"},
	 	"Scope"					-> {},
	 	"RawGets"				-> {"RawSurveyList","RawSurveyDetails","RawCollectorList","RawResponseCounts","RawSurveyResponseList",
	 								"RawCollectorResponseList","RawSurveyResponse","RawCollectorResponse","RawUserDetails","RawTemplateList",
	 								"RawCategoryList","RawSurveyResponseBulk"},
	 	"RawPosts"				-> {"RawCreateCollector"},
 		"Information"			-> "A service for sending and receiving data from SurveyMonkey"
	},
	{
		"OAuthVersion"			-> "2.0",
		"ServiceName" 			-> "SurveyMonkey",
	 	"AuthorizeEndpoint"		-> "https://api.surveymonkey.net/oauth/authorize", 
     	"AccessEndpoint"   		-> "https://api.surveymonkey.net/oauth/token",
     	"RedirectURI"     	 	-> "https://www.wolfram.com/oauthlanding?service=SurveyMonkey",
		"VerifierLabel"			-> "code",
		"AuthorizationFunction"	-> Automatic,
		"AccessTokenRequestor"	-> Automatic,
	 	"ClientInfo"			-> {"Wolfram","Token"},
	 	"AuthenticationDialog" 	:> (OAuthClient`tokenOAuthDialog[#, "SurveyMonkey"]&),
	 	"RequestFormat"		-> (Block[{params = Lookup[{##2},"Parameters",{}], method = Lookup[{##2}, "Method"],
        							body = Lookup[{##2},"BodyData",""], headers = Lookup[{##2},"Headers", {}], auth},
        							auth = Lookup[params,"access_token"];
        							URLRead[HTTPRequest[#1,	<|"Headers" -> Join[{"Authorization" -> "Bearer " <> auth}, headers], 
										Method -> method, "Query" -> KeyDrop["access_token"][params], "Body" -> body,"ContentType" -> "application/json"|>],
										{"Body", "StatusCode"}, "CredentialsProvider" -> None]
        						]&),
	 	"Gets"					-> {"SurveyList","SurveyDetails","CollectorList","ResponseCounts","SurveyResponseList","CollectorResponseList",
	 								"Response","UserData","TemplateList","CategoryList","SurveyResults","SurveyResponseBulk"},
	 	"Posts"					-> {"CreateWeblinkCollector"},
	 	"Scope"					-> {},
	 	"RawGets"				-> {"RawSurveyList","RawSurveyDetails","RawCollectorList","RawResponseCounts","RawSurveyResponseList",
	 								"RawCollectorResponseList","RawSurveyResponse","RawCollectorResponse","RawUserDetails","RawTemplateList",
	 								"RawCategoryList","RawSurveyResponseBulk"},
	 	"RawPosts"				-> {"RawCreateCollector"},
 		"Information"			-> "A service for sending and receiving data from SurveyMonkey"
	}
]

(*Raw*)

surveymonkeydata["RawSurveyList"]:={
        "URL"				-> "https://api.surveymonkey.net/v3/surveys",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"page","per_page","sort_by","sort_order","include","title","start_modified_at","end_modified_at"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawSurveyDetails"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/surveys/`1`/details", ##]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"survey_id"},
        "RequiredParameters"-> {"survey_id"},
        "ResultsFunction"	-> surveymonkeyimport
}
        
surveymonkeydata["RawCollectorList"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/surveys/`1`/collectors", ##]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"page","per_page","include","start_date","end_date","name"},
        "PathParameters"	-> {"survey_id"},
        "RequiredParameters"-> {"survey_id"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawResponseCounts"]:={
        "URL"				-> "https://api.surveymonkey.net/v2/surveys/get_response_counts",
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"ParameterlessBodyData"},
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawSurveyResponseBulk"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/surveys/`1`/responses/bulk", ##]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"page","per_page"},
        "PathParameters"	-> {"survey_id"},
        "RequiredParameters"-> {"survey_id"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawCollectorResponseList"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/collectors/`1`/responses", ##]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"page","per_page","start_created_at","end_created_at","start_modified_at","end_created_at","sort_by","status"},
        "PathParameters"	-> {"collector_id"},
        "RequiredParameters"-> {"collector_id"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawSurveyResponseList"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/surveys/`1`/responses", ##]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"page","per_page","start_created_at","end_created_at","start_modified_at","end_created_at","sort_by","status"},
        "PathParameters"	-> {"survey_id"},
        "RequiredParameters"-> {"survey_id"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawSurveyResponse"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/surveys/`1`/responses/`2`", ##]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"survey_id","response_id"},
        "RequiredParameters"-> {"survey_id","response_id"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawCollectorResponse"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/collectors/`1`/responses/`2`", ##]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"collector_id","response_id"},
        "RequiredParameters"-> {"collector_id","response_id"},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawUserDetails"]:={
        "URL"				-> "https://api.surveymonkey.net/v3/users/me",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawCategoryList"]:={
        "URL"				-> "https://api.surveymonkey.net/v3/survey_categories",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"language","per_page","page"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> surveymonkeyimport
}

surveymonkeydata["RawTemplateList"]:={
        "URL"				-> "https://api.surveymonkey.net/v3/survey_templates",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"language","per_page","page","category"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> surveymonkeyimport
}
 
surveymonkeydata["RawCreateCollector"]:={
        "URL"				-> (ToString@StringForm["https://api.surveymonkey.net/v3/surveys/`1`/collectors", ##]&),
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"ParameterlessBodyData"},
       	"PathParameters"	-> {"survey_id"},
        "RequiredParameters"-> {"survey_id"},
        "ResultsFunction"	-> surveymonkeyimport
}
            
(*Cooked*) 
 (* a function for importing the raw data - usually json or xml - from the service *)

surveymonkeyimport[$Failed]:=(Message[ServiceExecute::serror];Throw[$Failed])

surveymonkeyimport[data_Association]:=Block[{status,response,errormessage,errorname},

	status = data["StatusCode"];
	response = ImportString[data["Body"],"RawJSON"];

	Switch[status,
				200 | 201,
					response
					,
				400 | 401 | 403,
					errormessage = response["error"]["message"];
					Message[ServiceExecute::serrormsg,errormessage];
					Throw[$Failed]
					,
				404,
					(*404 is gven if there is an error retrieving the requested resource, 'name' gives hint about the reason for the same *)
					errorname  = response["error"]["name"];
					Message[ServiceExecute::serrormsg,errorname];
					Throw[$Failed]
					,
				_,
					Throw[$Failed]
		]
]

surveymonkeyimport[___]:=Throw[$Failed]

surveymonkeycookeddata["SurveyResults", id_,args_]:=Block[{surveydetail,bulkresponse = {},res,questions,surveydetquesdata,
	surveydetfilterdata,responseid,questiondata,finalres = {},presponse,resppagedata,newparams,invalidParameters,surveyid,
	params={},rawdata,data,sdpagesdata,brespres,page},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	(*get all questions headings/ids for a given survey *)
	rawdata = OAuthClient`rawoauthdata[id,"RawSurveyDetails",params];
	data = surveymonkeyimport@rawdata;
	If[!(MissingQ @ Lookup[data,"pages"]),
		data["pages"] = Replace[data["pages"], r : {__Association} :> KeyDrop[r, "href"], {0, Infinity}];
		data["pages"] = capitalizekeys[#]&/@data["pages"];
		sdpagesdata = data["pages"];,
		(*else*)
		(* it is very unlikely where a survey is without questionaire, but if this
		occurs, just return empty dataset as there is no need to call survey response api *)
		Return[Dataset[<||>]]		
	];
	
	(*start retreiving response from 1st page*)
	page = 1;
	(*fetch all the answers for given survey*)
	While[Length@(brespres = surveymonkeycookeddata["SurveyResponseBulk",id,Join[args,{"MaxItems"->100,"StartIndex"->page}]]//Normal)>0,
		bulkresponse = Join[bulkresponse,brespres];
		page = page+1;
	];
	(*associate answers with question's values/headings using question ids*)
	If[Length@bulkresponse>0,
		(
			responseid = #["ResponseID"];
			resppagedata = #["Pages"];
			(*Omit unanwered responses*)
			resppagedata= Select[resppagedata, Length@#["Questions"] > 0 &];
			(*call to process response data*)
			presponse = processResponsePerRespondent[resppagedata, sdpagesdata];
			finalres = Join[finalres,{<|"ResponseID"->responseid,"Questions"->presponse|>}];
		)&/@bulkresponse;
		finalres//Dataset,
		(*else*)
		Dataset[<||>]
	]
]

surveymonkeycookeddata["SurveyResponseBulk", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,surveyid,responses,newparams,res,
	maxitems,startindex},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID","MaxItems","StartIndex"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];

	rawdata = OAuthClient`rawoauthdata[id,"RawSurveyResponseBulk",params];
	data = surveymonkeyimport@rawdata;
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		(*capitalize keys for pages data for each response*)
		data = capitalizepages/@data;
		data = Replace[Normal@data, {Rule["id", b_] :> Rule["response_id", b],
				Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
				Rule["date_modified", b_] :> Rule["date_modified", First@StringSplit[b, "+"]]}, Infinity];
		data = Association[#]& /@ data;
		res = surveymonkeyformatdata[Dataset@data];
		res[All,{"ResponseID","Pages"}]
	]		
]


surveymonkeycookeddata["SurveyList", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,fields=0,newparams,maxitems,startindex,
	smoddate,emoddate,title,newordering,res},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"StartIndex","MaxItems","StartModifiedDate","EndModifiedDate","Title","Fields"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	If[KeyExistsQ[newparams,"Fields"],
		fields = Lookup[newparams,"Fields"];
		If[AllTrue[StringMatchQ[fields, Keys@$fieldsRules],TrueQ],
			params = Append[params,Rule["include",StringRiffle[Lookup[$fieldsRules,fields],","]]],
			(*else*)
			Message[ServiceExecute::nval,"Fields","SurveyMonkey"];
			Throw[$Failed]
		]
	];
	If[ KeyExistsQ[newparams,"StartModifiedDate"],
		smoddate = Lookup[newparams,"StartModifiedDate"];
		If[!(StringQ[smoddate]||DateObjectQ[smoddate]),
			Message[ServiceExecute::nval,"StartModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
		smoddate = Quiet@DateObject[smoddate];
		If[MatchQ[smoddate,DateObject[__String]],
			Message[ServiceExecute::nval,"StartModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["start_modified_at",DateString[TimeZoneConvert[smoddate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"EndModifiedDate"],
		emoddate = Lookup[newparams,"EndModifiedDate"];
		If[!(StringQ[emoddate]||DateObjectQ[emoddate]),
			Message[ServiceExecute::nval,"EndModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
		emoddate = Quiet@DateObject[emoddate];
		If[MatchQ[emoddate,DateObject[__String]],
			Message[ServiceExecute::nval,"EndModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["end_modified_at",DateString[TimeZoneConvert[emoddate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[KeyExistsQ[newparams,"Title"],
		title = Lookup[newparams,"Title"];
		If[StringQ[title],
			params = Append[params,Rule["title",title]],
			(*else*)
			Message[ServiceExecute::nval,"Title","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawSurveyList",params];
	data = surveymonkeyimport@rawdata;
	(*get required data from response*)
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		data = Replace[Normal@data, Rule["id", b_] :> Rule["survey_id", b], Infinity];
		data = Association[#]& /@ data;
		res = surveymonkeyformatdata[Dataset@data];
		(*finally bring useful,important keys to the front of dataset*)
		If[Length@fields===0,
			newordering = Keys@FilterRules[$fieldsRules, Normal@res[1,Keys]],
			(*else*)
			newordering = Join[fields, Select[Normal@res[1,Keys], !MemberQ[fields, #] &]];
		];
		res[All,newordering]
	]
]

surveymonkeycookeddata["SurveyDetails", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,newparams,surveyid,res,
	pagesdata,questionobject,newquestionobject},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawSurveyDetails",params];
	data = surveymonkeyimport@rawdata;
	(*select valuable keys,capitalize keys for pages data*)
	If[!(MissingQ @ Lookup[data,"pages"]),
		data["pages"] = Replace[data["pages"], r : {__Association} :> KeyDrop[r, "href"], {0, Infinity}];
		data["pages"] = capitalizekeys[#]&/@data["pages"];
		pagesdata = data["pages"];
		data["pages"] = (
					questionobject = #["Questions"];
					newquestionobject = <|"PageID"->#["Id"],"QuestionCount" -> #["QuestionCount"], 
								"Answers" -> Lookup[questionobject, "Answers", Missing], 
       							"Headings" -> (Flatten[Lookup[questionobject, "Headings"]]),
       							"QuestionID" -> Lookup[questionobject, "Id"]|>
       				) & /@ pagesdata;
	];
	data = Association@Replace[Normal@data, Rule["id", b_] :> Rule["survey_id", b], Infinity];
	res = surveymonkeyformatdata[Dataset@List@data];
	(*above result is a Dataset with list of only one association, convert back to Dataset[Association[]] format*)
	Dataset@First@Normal@res
]

surveymonkeycookeddata["CollectorList", id_,args_]:=Block[{surveyid,rawdata,data,params={},invalidParameters,fields=0,newparams,maxitems,
	startindex,sdate,edate,name,validfields={"Name","URL","Type","Status","DateCreated","DateModified"},newordering,res},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID","StartIndex","MaxItems","StartDate","EndDate","Name","Fields"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	If[KeyExistsQ[newparams,"Fields"],
		fields = Lookup[newparams,"Fields"];
		If[AllTrue[StringMatchQ[fields, validfields],TrueQ],
			params = Append[params,Rule["include",StringRiffle[Lookup[$fieldsRules,fields],","]]],
			(*else*)
			Message[ServiceExecute::nval,"Fields","SurveyMonkey"];
			Throw[$Failed]
		]
	];
	If[ KeyExistsQ[newparams,"StartDate"],
		sdate = Lookup[newparams,"StartDate"];
		If[!(StringQ[sdate]||DateObjectQ[sdate]),
			Message[ServiceExecute::nval,"StartDate","SurveyMonkey"];
			Throw[$Failed]
		];
		sdate = Quiet@DateObject[sdate];
		If[MatchQ[sdate,DateObject[__String]],
			Message[ServiceExecute::nval,"StartDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["start_date",DateString[TimeZoneConvert[sdate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"EndDate"],
		edate = Lookup[newparams,"EndDate"];
		If[!(StringQ[edate]||DateObjectQ[edate]),
			Message[ServiceExecute::nval,"EndDate","SurveyMonkey"];
			Throw[$Failed]
		];
		edate = Quiet@DateObject[edate];
		If[MatchQ[edate,DateObject[__String]],
			Message[ServiceExecute::nval,"EndDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["end_date",DateString[TimeZoneConvert[edate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[KeyExistsQ[newparams,"Name"],
		name = Lookup[newparams,"Name"];
		If[StringQ[name],
			params = Append[params,Rule["name",name]],
			(*else*)
			Message[ServiceExecute::nval,"Name","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawCollectorList",params];
	data = surveymonkeyimport@rawdata;
	(*get required data from response*)
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		data = Replace[Normal@data, {Rule["id", b_] :> Rule["collector_id", b],
			Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
			Rule["date_modified", b_] :> Rule["date_modified", First@StringSplit[b, "+"]]}, Infinity];
		data = Association[#]& /@ data;
		res = surveymonkeyformatdata[Dataset@data];
		(*finally bring useful,important keys to the front of dataset*)
		If[Length@fields===0,
			newordering = Keys@FilterRules[$fieldsRules, Normal@res[1,Keys]],
			(*else*)
			newordering = Join[fields, Select[Normal@res[1,Keys], !MemberQ[fields, #] &]];
		];
		res[All,newordering]
	]
]

surveymonkeycookeddata["ResponseCounts", id_,args_]:=Block[{data,statuslist,res = <|"Completed"->0,"Started"->0|>},
	data = surveymonkeycookeddata["CollectorResponseList",id,args];
	If[Length@data===0,
		data,
		(*else*)
		statuslist = data[All,"ResponseStatus"];
		res["Completed"] = Count[statuslist,"completed"];
		res["Started"] = Count[statuslist, "started"];
		res
	]
]

surveymonkeycookeddata["CollectorResponseList", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,sdate,edate,smoddate,emoddate,
	fields,newparams,surveyid,collectorid,maxitems,startindex,sortby,res,status,newordering,responseids,responsedata = {}},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CollectorID","StartIndex","MaxItems","StartDate","EndDate",
		"StartModifiedDate","EndModifiedDate","SortBy","Status"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"CollectorID"],
		collectorid = Lookup[newparams,"CollectorID"];
		If[!(StringQ[collectorid]||IntegerQ[collectorid]),
		(	
			Message[ServiceExecute::nval,"CollectorID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["collector_id",ToString[collectorid]]],
		(*else*)
		Message[ServiceExecute::nparam,"CollectorID","SurveyMonkey"];
		Throw[$Failed]
	];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	If[ KeyExistsQ[newparams,"StartDate"],
		sdate = Lookup[newparams,"StartDate"];
		If[!(StringQ[sdate]||DateObjectQ[sdate]),
			Message[ServiceExecute::nval,"StartDate","SurveyMonkey"];
			Throw[$Failed]
		];
		sdate = Quiet@DateObject[sdate];
		If[MatchQ[sdate,DateObject[__String]],
			Message[ServiceExecute::nval,"StartDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["start_created_at",DateString[TimeZoneConvert[sdate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"EndDate"],
		edate = Lookup[newparams,"EndDate"];
		If[!(StringQ[edate]||DateObjectQ[edate]),
			Message[ServiceExecute::nval,"EndDate","SurveyMonkey"];
			Throw[$Failed]
		];
		edate = Quiet@DateObject[edate];
		If[MatchQ[edate,DateObject[__String]],
			Message[ServiceExecute::nval,"EndDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["end_created_at",DateString[TimeZoneConvert[edate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"StartModifiedDate"],
		smoddate = Lookup[newparams,"StartModifiedDate"];
		If[!(StringQ[smoddate]||DateObjectQ[smoddate]),
			Message[ServiceExecute::nval,"StartModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
		smoddate = Quiet@DateObject[smoddate];
		If[MatchQ[smoddate,DateObject[__String]],
			Message[ServiceExecute::nval,"StartModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["start_modified_at",DateString[TimeZoneConvert[smoddate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"EndModifiedDate"],
		emoddate = Lookup[newparams,"EndModifiedDate"];
		If[!(StringQ[emoddate]||DateObjectQ[emoddate]),
			Message[ServiceExecute::nval,"EndModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
		emoddate = Quiet@DateObject[emoddate];
		If[MatchQ[emoddate,DateObject[__String]],
			Message[ServiceExecute::nval,"EndModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["end_modified_at",DateString[TimeZoneConvert[emoddate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[KeyExistsQ[newparams,"SortBy"],
		sortby = Lookup[newparams,"SortBy"];
		If[!StringMatchQ[sortby, "DateModified"],
		(	
			Message[ServiceExecute::nval,"SortBy","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["sort_by","date_modified"]]
		];
	If[KeyExistsQ[newparams,"Status"],
		status = Lookup[newparams,"Status"];
		If[!StringMatchQ[status, "Completed"|"Partial"|"OverQuota"|"DisQualified"],
		(	
			Message[ServiceExecute::nval,"Status","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["status",ToLowerCase[status]]]
		];
		
	rawdata = OAuthClient`rawoauthdata[id,"RawCollectorResponseList",params];
	data = surveymonkeyimport@rawdata;
	(*get required data from response*)
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		responseids = #["id"]&/@data;
		(
			rawdata = OAuthClient`rawoauthdata[id,"RawCollectorResponse",{"collector_id"->collectorid,"response_id"->#}];
			data = surveymonkeyimport@rawdata;
			data = Association@Replace[Normal@data, {Rule["id", b_] :> Rule["response_id", b],
					Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
					Rule["date_modified", b_] :> Rule["date_modified", First@StringSplit[b, "+"]]}, Infinity];
			AppendTo[responsedata,data]
		)&/@responseids;
		res = surveymonkeyformatdata[Dataset@responsedata];
		newordering = Keys@FilterRules[$fieldsRules, DeleteCases[Normal@res[1,Keys],"CollectorID"|"RecipientID"]];
		res[All,newordering]
	]
]

surveymonkeycookeddata["SurveyResponseList", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,sdate,edate,smoddate,emoddate,
	fields,newparams,surveyid,collectorid,maxitems,startindex,sortby,res,status,newordering,responseids,responsedata = {}},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID","StartIndex","MaxItems","StartDate","EndDate",
		"StartModifiedDate","EndModifiedDate","SortBy","Status"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	If[ KeyExistsQ[newparams,"StartDate"],
		sdate = Lookup[newparams,"StartDate"];
		If[!(StringQ[sdate]||DateObjectQ[sdate]),
			Message[ServiceExecute::nval,"StartDate","SurveyMonkey"];
			Throw[$Failed]
		];
		sdate = Quiet@DateObject[sdate];
		If[MatchQ[sdate,DateObject[__String]],
			Message[ServiceExecute::nval,"StartDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["start_created_at",DateString[TimeZoneConvert[sdate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"EndDate"],
		edate = Lookup[newparams,"EndDate"];
		If[!(StringQ[edate]||DateObjectQ[edate]),
			Message[ServiceExecute::nval,"EndDate","SurveyMonkey"];
			Throw[$Failed]
		];
		edate = Quiet@DateObject[edate];
		If[MatchQ[edate,DateObject[__String]],
			Message[ServiceExecute::nval,"EndDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["end_created_at",DateString[TimeZoneConvert[edate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"StartModifiedDate"],
		smoddate = Lookup[newparams,"StartModifiedDate"];
		If[!(StringQ[smoddate]||DateObjectQ[smoddate]),
			Message[ServiceExecute::nval,"StartModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
		smoddate = Quiet@DateObject[smoddate];
		If[MatchQ[smoddate,DateObject[__String]],
			Message[ServiceExecute::nval,"StartModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["start_modified_at",DateString[TimeZoneConvert[smoddate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[ KeyExistsQ[newparams,"EndModifiedDate"],
		emoddate = Lookup[newparams,"EndModifiedDate"];
		If[!(StringQ[emoddate]||DateObjectQ[emoddate]),
			Message[ServiceExecute::nval,"EndModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
		emoddate = Quiet@DateObject[emoddate];
		If[MatchQ[emoddate,DateObject[__String]],
			Message[ServiceExecute::nval,"EndModifiedDate","SurveyMonkey"];
			Throw[$Failed]
		];
        params = Append[params, Rule["end_modified_at",DateString[TimeZoneConvert[emoddate,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]]]     
	];
	If[KeyExistsQ[newparams,"SortBy"],
		sortby = Lookup[newparams,"SortBy"];
		If[!StringMatchQ[sortby, "DateModified"],
		(	
			Message[ServiceExecute::nval,"SortBy","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["sort_by","date_modified"]]
		];
	If[KeyExistsQ[newparams,"Status"],
		status = Lookup[newparams,"Status"];
		If[!StringMatchQ[status, "Completed"|"Partial"|"OverQuota"|"DisQualified"],
		(	
			Message[ServiceExecute::nval,"Status","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["status",ToLowerCase[status]]]
		];
		
	rawdata = OAuthClient`rawoauthdata[id,"RawSurveyResponseList",params];
	data = surveymonkeyimport@rawdata;
	(*get required data from response*)
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		responseids = #["id"]&/@data;
		(
			rawdata = OAuthClient`rawoauthdata[id,"RawSurveyResponse",{"survey_id"->surveyid,"response_id"->#}];
			data = surveymonkeyimport@rawdata;
			data = Association@Replace[Normal@data, {Rule["id", b_] :> Rule["response_id", b],
					Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
					Rule["date_modified", b_] :> Rule["date_modified", First@StringSplit[b, "+"]]}, Infinity];
			AppendTo[responsedata,data]
		)&/@responseids;
		res = surveymonkeyformatdata[Dataset@responsedata];
		newordering = Keys@FilterRules[$fieldsRules, DeleteCases[Normal@res[1,Keys],"SurveyID"|"RecipientID"]];
		res[All,newordering]
	]
]


surveymonkeycookeddata["Response", id_,args_]:=Block[{rawdata,params={},invalidParameters,newparams,surveyid,responseid,data,res},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID","ResponseID"},#]&]; 
	If[Length[invalidParameters]>0,
	Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	If[KeyExistsQ[newparams,"ResponseID"],
		responseid = Lookup[newparams,"ResponseID"];
		If[!(StringQ[responseid]||IntegerQ[responseid]),
		(	
			Message[ServiceExecute::nval,"ResponseID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["response_id",ToString[responseid]]],
		(*else*)
		Message[ServiceExecute::nparam,"ResponseID","SurveyMonkey"];
		Throw[$Failed]
	];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawSurveyResponse",params];
	data = surveymonkeyimport@rawdata;
	data = Association@Replace[Normal@data, {Rule["id", b_] :> Rule["response_id", b],
			Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
			Rule["date_modified", b_] :> Rule["date_modified", First@StringSplit[b, "+"]]}, Infinity];
	res = surveymonkeyformatdata[Dataset@List@data];
	(*above result is a Dataset with list of only one association, convert back to Dataset[Association[]] format*)
	Dataset@First@Normal@res		
]

surveymonkeycookeddata["UserData", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,newparams,res},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{},#]&]; 
	If[Length[invalidParameters]>0,
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	];
		
	rawdata = OAuthClient`rawoauthdata[id,"RawUserDetails",params];
	data = surveymonkeyimport@rawdata;
	data = Association@Replace[Normal@data, {Rule["id", b_] :> Rule["user_id", b],
			Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
			Rule["date_last_login", b_] :> Rule["date_last_login", DateObject@First@StringSplit[b, "+"]]}, Infinity];
	res = surveymonkeyformatdata[Dataset@List@data];
	(*above result is a Dataset with list of only one association, convert back to Dataset[Association[]] format*)
	Dataset@First@Normal@res
]

surveymonkeycookeddata["TemplateList", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,newparams,
	maxitems,startindex,categoryid,language,res,newordering},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"Language","CategoryID","StartIndex","MaxItems"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Language"],
		language = Lookup[newparams,"Language"];
		If[!(StringQ[language]||MatchQ[language,Entity["Language", _]]),
		(	
			Message[ServiceExecute::nval,"Language","SurveyMonkey"];
			Throw[$Failed]
		)];
		language = Interpreter["Language"][language];
		If[(FailureQ[language])||(MissingQ[language = $langmap[language]]),
			(*use dafault language value*)
			language = "en";
		];
		params = Append[params,Rule["language",language]];
	];
	If[KeyExistsQ[newparams,"CategoryID"],
		categoryid = Lookup[newparams,"CategoryID"];
		If[!StringQ[categoryid],
		(	
			Message[ServiceExecute::nval,"CategoryID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["category",categoryid]]
	];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawTemplateList",params];
	data = surveymonkeyimport@rawdata;
	(*take actual data from response*)
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		data = Replace[Normal@data, Rule["id", b_] :> Rule["template_id", b], Infinity];
		data = Association[#]& /@ data;
		res = surveymonkeyformatdata[Dataset@data];
		newordering = Keys@FilterRules[$fieldsRules, DeleteCases[Normal@res[1,Keys],"Name"]];(*Name is redundant in resp as Title is present*)
		res[All,newordering]
	]
]

surveymonkeycookeddata["CategoryList", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,newparams,
	maxitems,startindex,category,language,res,newordering},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"Language","StartIndex","MaxItems"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Language"],
		language = Lookup[newparams,"Language"];
		If[!(StringQ[language]||MatchQ[language,Entity["Language", _]]),
		(	
			Message[ServiceExecute::nval,"Language","SurveyMonkey"];
			Throw[$Failed]
		)];
		language = Interpreter["Language"][language];
		If[(FailureQ[language])||(MissingQ[language = $langmap[language]]),
			(*use dafault language value*)
			language = "en";
		];
		params = Append[params,Rule["language",language]];
	];
	If[KeyExistsQ[newparams,"MaxItems"],	
		maxitems = Lookup[newparams,"MaxItems"];
		If[IntegerQ[maxitems] && 0<maxitems<=1000,
			params = Append[params,Rule["per_page",ToString[maxitems]]],
			(*else*)
			Message[ServiceExecute::nval,"MaxItems","SurveyMonkey"];
			Throw[$Failed]
		],
		(*else*)
  		params = Append[params,Rule["per_page","10"]]
  	];
	If[KeyExistsQ[newparams,"StartIndex"],
		startindex = Lookup[newparams,"StartIndex"];
		If[IntegerQ[startindex] && startindex>0,
			params = Append[params,Rule["page",ToString[startindex]]],
			(*else*)
			Message[ServiceExecute::nval,"StartIndex","SurveyMonkey"];
			Throw[$Failed]
		];
	];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawCategoryList",params];
	data = surveymonkeyimport@rawdata;
	(*take actual data from response*)
	data = data["data"];
	If[Length@data===0,
		Dataset[<||>],
		(*else*)
		data = Replace[Normal@data, Rule["id", b_] :> Rule["category_id", b], Infinity];
		data = Association[#]& /@ data;
		res = surveymonkeyformatdata[Dataset@data];
		newordering = Keys@FilterRules[$fieldsRules, Normal@res[1,Keys]];
		res[All,newordering]
	]
]

surveymonkeycookeddata["CreateWeblinkCollector", id_,args_]:=Block[{rawdata,data,params={},invalidParameters,newparams,name,surveyid,
	bodydata={},res},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SurveyID","Name"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"SurveyMonkey"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SurveyID"],
		surveyid = Lookup[newparams,"SurveyID"];
		If[!(StringQ[surveyid]||IntegerQ[surveyid]),
		(	
			Message[ServiceExecute::nval,"SurveyID","SurveyMonkey"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["survey_id",ToString[surveyid]]],
		(*else*)
		Message[ServiceExecute::nparam,"SurveyID","SurveyMonkey"];
		Throw[$Failed]
	];
	bodydata = Append[bodydata,Rule["type","weblink"]];
	If[KeyExistsQ[newparams,"Name"],
		name = Lookup[newparams,"Name"];
		If[!StringQ[name],
		(	
			Message[ServiceExecute::nval,"Name","SurveyMonkey"];
			Throw[$Failed]
		)];
		bodydata = Append[bodydata,Rule["name",name]]
		];
	params = Append[params,Rule["ParameterlessBodyData",ExportString[bodydata,"JSON"]]];
	
	rawdata = OAuthClient`rawoauthdata[id,"RawCreateCollector",params];
	data = surveymonkeyimport@rawdata;
	data = Association@Replace[Normal@data, {Rule["id", b_] :> Rule["collector_id", b],
			Rule["date_created", b_] :> Rule["date_created", First@StringSplit[b, "+"]], 
			Rule["date_modified", b_] :> Rule["date_modified", First@StringSplit[b, "+"]]}, Infinity];
	res = surveymonkeyformatdata[Dataset@List@data];
	(*above result is a Dataset with list of only one association, convert back to Dataset[Association[]] format*)
	Dataset@First@Normal@res
]

surveymonkeycookeddata[___]:=$Failed

surveymonkeyrawdata[___]:=$Failed

surveymonkeysendmessage[args_]:=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];


(* Return two functions to define oauthservicedata, oauthcookeddata  *)
{SurveyMonkey`Private`surveymonkeydata,SurveyMonkey`Private`surveymonkeycookeddata,SurveyMonkey`Private`surveymonkeysendmessage}
