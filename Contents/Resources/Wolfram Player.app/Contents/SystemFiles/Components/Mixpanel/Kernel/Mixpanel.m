Get["MixpanelFunctions.m"];

Begin["MixpanelAPI`"]

ServiceExecute::mppropv="Cannot use the parameter \"Values\" on a list of properties."
ServiceExecute::wprop="The property `1` is not applicable to the event `2`."

Begin["`Private`"]

(******************************* Mixpanel *************************************)

(* Authentication information *)

mixpaneldata[]:={

		"ServiceName" 		-> "Mixpanel",
		"URLFetchFun"		:> (Block[{params,apisecret},
						    	params = MapAt[ToString, {{All, 2}}] @ Lookup[{##2},"Parameters",{}]; apisecret=Lookup[params,"apisecret",Throw[$Failed]];
						    	URLFetch[#1, {"StatusCode","Content"}, "Parameters" -> FilterRules[params, Except["apisecret"]], "Username" -> apisecret]]&)
        	,
		"ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["Mixpanel",{"API Secret"->{"apisecret",FieldMasked->True}},"https://mixpanel.com/login/","https://mixpanel.com/terms/"],
	 	"Gets"				-> {"ListAnnotations","Events","EventTypes","TopEvents","Properties","TopProperties","PropertyValues", "Funnel", "Funnels"},
	 	"RawGets"			-> {"RawEvents","RawEventsTop","RawEventsNames","RawEventProperties", "RawEventPropertiesValues","RawEventPropertiesTop","RawFunnels","RawFunnelsList","RawSegmentation","RawSegmentationNumeric","RawSegmentationSum","RawSegmentationAverage","RawRetention","RawRetentionAddiction","RawEngage","RawAnnotations"},
	 	"Posts"				-> {"AddAnnotation","UpdateAnnotation"},
	 	"RawPosts"			-> {"RawAnnotationCreate","RawAnnotationUpdate"},
 		"Information"		-> "Import Mixpanel API data to the Wolfram Language"
 }
 
(* Auxiliar functions*)
mixpanelimportdata[""]:= Missing["NotAvailable"]
mixpanelimportdata[rawdata_]:= With[{res = Quiet[Developer`ReadRawJSONString[Last[rawdata]]]},
	If[ AssociationQ[res] || ListQ[res],
		res,
		Message[ServiceExecute::serror];
		Throw[$Failed]
	]
]

(* Raw *)
mixpaneldata["RawAnnotations"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/annotations/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"from_date","to_date"},
        "RequiredParameters"-> {"from_date","to_date"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawAnnotationCreate"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/annotations/create",
        "HTTPSMethod"		-> "POST",
        "Parameters"		-> {"date","description"},
        "RequiredParameters"-> {"date","description"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawAnnotationUpdate"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/annotations/update",
        "HTTPSMethod"		-> "POST",
        "Parameters"		-> {"id","date","description"},
        "RequiredParameters"-> {"id","date","description"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEvents"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/events/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","type","unit","interval"},
        "RequiredParameters"-> {"event","type","unit","interval"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEventsTop"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/events/top/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"type","limit"},
        "RequiredParameters"-> {"type"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEventsNames"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/events/names/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"type","limit"},
        "RequiredParameters"-> {"type"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEventProperties"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/events/properties/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","name","type","unit","interval","values","limit"},
        "RequiredParameters"-> {"event","name","type","unit","interval"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEventPropertiesTop"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/events/properties/top/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","limit"},
        "RequiredParameters"-> {"event"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEventPropertiesValues"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/events/properties/values/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","name","limit","bucket"},
        "RequiredParameters"-> {"event","name"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawFunnels"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/funnels/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"funnel_id","from_date","to_date","length","interval","unit","on","where","limit"},
        "RequiredParameters"-> {"funnel_id"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawFunnelsList"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/funnels/list/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawSegmentation"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/segmentation/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","from_date","to_date","on","unit","where","limit","type"},
        "RequiredParameters"-> {"event","from_date","to_date"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawSegmentationNumeric"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/segmentation/numeric/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","from_date","to_date","on","unit","where","type"},
        "RequiredParameters"-> {"event","from_date","to_date","on"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawSegmentationSum"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/segmentation/sum/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","from_date","to_date","on","unit","where"},
        "RequiredParameters"-> {"event","from_date","to_date","on"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawSegmentationAverage"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/segmentation/average/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"event","from_date","to_date","on","unit","where"},
        "RequiredParameters"-> {"event","from_date","to_date","on"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawRetention"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/retention/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"from_date","to_date","retention_type","born_event","event","born_where","where","interval","interval_count","unit","on","limit"},
        "RequiredParameters"-> {"from_date","to_date","born_event"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawRetentionAddiction"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/retention/addiction",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"from_date","to_date","unit","addiction_unit","event","where","on","limit"},
        "RequiredParameters"-> {"from_date","to_date","unit","addiction_unit"},
        "ResultsFunction"	-> mixpanelimportdata
	}

mixpaneldata["RawEngage"]:={
        "URL"				-> "https://mixpanel.com/api/2.0/engage/",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"where","session_id","page"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> mixpanelimportdata
	}

(* Cooked *)

mixpanelcookeddata[req_, id_, rules___Rule]:= mixpanelcookeddata[req, id, {rules}]

mixpanelcookeddata["ListAnnotations",id_, args_?OptionQ] :=  Block[{invalidParameters,window,var,startD,endD,raw,data},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"DateWindow"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"DateWindow"],
		(
			window = Lookup[args,"DateWindow"];
			Switch[window,				
					Quantity[_Integer?(#>0&),"Hours" | "Days" | "Weeks" | "Months"],
						If[Quantity[100,"Years"] > window ,
							(
								endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
								startD = DateString[Today - window, {"Year", "-", "Month", "-", "Day"}]
							),
						Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
						Throw[$Failed]
						]
					,
					
					_String,
						var = Interpreter["Quantity"][window];
						If[!FailureQ[var],
							If[MatchQ[var, Quantity[_Integer?(# > 0 &), "Hours" | "Days" | "Weeks" | "Months"]] && (Quantity[100,"Years"] > var),
								(
									endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
									startD = DateString[Today - window, {"Year", "-", "Month", "-", "Day"}]
								),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
							],
						Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
						Throw[$Failed]
						]
					,
					
					{_Integer?(#>0&),_String},
						var = ((ToUpperCase[First[#]]<>Rest[#])& @ Characters[window[[2]]]);
						If[MatchQ[var,
							"Hours" | "Days" | "Weeks" | "Months"],
							(
								endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
								startD = DateString[Today - Quantity[window[[1]],window[[2]]], {"Year", "-", "Month", "-", "Day"}]
							),
						Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
						Throw[$Failed]
						]
					,

					{_DateObject,_DateObject},
						If[window[[2]]<window[[1]],
							(
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
							)
						];
						startD = DateString[window[[1]], {"Year", "-", "Month", "-", "Day"}];
						endD = DateString[window[[2]], {"Year", "-", "Month", "-", "Day"}];
					,

					True,
					Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
					Throw[$Failed]
				]
		),
		startD = DateString[DatePlus[Today,-30], {"Year", "-", "Month", "-", "Day"}];
		endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
	];

	raw = KeyClient`rawkeydata[id,"RawAnnotations", {"from_date" -> startD, "to_date" -> endD}];
	Which[
			raw[[1]] === 200, (*status*)
				data = mixpanelimportdata@raw;
				Dataset[Replace[Lookup[data, "annotations"],
					<|Rule["date", date_], "project_id" -> pj_, "id" -> aid_, "description" -> descp_|> :>
					<|Rule["Date", DateObject[DateList[#], DateFormat -> "DateShort"]&@date], "ProjectID" -> pj, "ID" -> aid, "Description" -> descp|>, {1}]],

			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
		]
]

mixpanelcookeddata["AddAnnotation",id_, args_?OptionQ] :=  Block[{invalidParameters,dateobj,date,date2,description,raw,predata,data},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"Date","Description"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"Date"],
		(
			dateobj = Lookup[args,"Date"];
				If[!DateObjectQ[dateobj],
					Message[ServiceExecute::nval,"Date","Mixpanel"];
					Throw[$Failed]
				];
			date = DateString[dateobj, {"Year", "-", "Month", "-", "Day", " ", "Time"}];
			date2 = StringTake[date,10];
		),
		Message[ServiceExecute::nparam,"Date","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"Description"],
		(
			description = Lookup[args,"Description"];
				If[!(StringQ[description]&&StringLength[description]<1001),
					Message[ServiceExecute::nval,"Description","Mixpanel"];
					Throw[$Failed]
				];
		),
		Message[ServiceExecute::nparam,"Description","Mixpanel"];
		Throw[$Failed]
	];
	
	raw = KeyClient`rawkeydata[id,"RawAnnotationCreate",{"date"->date,"description"->description}];
	
	Which[
			raw[[1]] === 200, (*status*)
				predata = Last@SortBy[Lookup[mixpanelimportdata@KeyClient`rawkeydata[id,"RawAnnotations",{"from_date"->date2,"to_date"->date2}],"annotations"], Lookup[#,"id"]&];
				data = Replace[predata,
					<|Rule["date", rdate_], "project_id" -> pj_, "id" -> aid_, "description" -> descp_|> :>
					<|Rule["Date", DateObject[DateList[#], DateFormat -> "DateShort"]&@rdate], "ProjectID" -> pj, "ID" -> aid, "Description" -> descp|>, {0}];
				Dataset[{data}],
	
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]
]

mixpanelcookeddata["UpdateAnnotation",id_, args_?OptionQ] :=  Block[{invalidParameters,noteID,dateobj,date,date2,description,raw,predata,data,status,errmsg},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"AnnotationID","Date","Description"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"AnnotationID"],
		(
			noteID = Lookup[args,"AnnotationID"];
				If[!MatchQ[noteID,_String|_Integer],
					Message[ServiceExecute::nval,"AnnotationID","Mixpanel"];
					Throw[$Failed]
				];
		),
		Message[ServiceExecute::nparam,"Date","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"Date"],
		(
			dateobj = Lookup[args,"Date"];
				If[!(DateObjectQ[dateobj] && dateobj > DateObject[{1970,1,1,0,0,0}]),
					Message[ServiceExecute::nval,"Date","Mixpanel"];
					Throw[$Failed]
				];
			date = DateString[dateobj, {"Year", "-", "Month", "-", "Day", " ", "Time"}];
			date2 = StringTake[date,10];
		),
		Message[ServiceExecute::nparam,"Date","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"Description"],
		(
			description = Lookup[args,"Description"];
				If[!(StringQ[description]&&StringLength[description]<1001),
					Message[ServiceExecute::nval,"Description","Mixpanel"];
					Throw[$Failed]
				];
		),
		Message[ServiceExecute::nparam,"Description","Mixpanel"];
		Throw[$Failed]
	];

	raw = KeyClient`rawkeydata[id,"RawAnnotationUpdate",{"id"->noteID,"date"->date,"description"->description}];

	Which[
			raw[[1]] === 200, (*status*)
				predata = First@Cases[Lookup[mixpanelimportdata@KeyClient`rawkeydata[id,"RawAnnotations",{"from_date"->date2,"to_date"->date2}],"annotations"],_Association?(Lookup[#, "id"] === ToExpression@noteID &)];
				data = Replace[predata,
					<|Rule["date", rdate_], "project_id" -> pj_, "id" -> aid_, "description" -> descp_|> :>
					<|Rule["Date", DateObject[DateList[#], DateFormat -> "DateShort"]&@rdate], "ProjectID" -> pj, "ID" -> aid, "Description" -> descp|>, {0}];
				Dataset[{data}],
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]
]

mixpanelcookeddata["Events",id_, args_?OptionQ] :=  Block[{invalidParameters, raw, data, events, type, window, var, interval, unit, dateOrder, pvalues, pvaluesObj, svalues, setvalues, svrep},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"Event", "Type", "DateWindow"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	If[KeyExistsQ[args,"Event"],
		var = Lookup[args,"Event"];
		Switch[var,
 				_String,
 					var = Replace[var,$mprsrvnamelist,{0, 1}];
 					events = "[\"" <> var <> "\"]",
 				l : {_String ..} /; Length[l] < 101,
 					var = Replace[var,$mprsrvnamelist,{0, 1}];
 					events = "[" <> Fold[#1 <> ", " <> #2 &, ("\"" <> # <> "\"" & /@ var)] <> "]",
				_,
					Message[ServiceExecute::nval,"Event","Mixpanel"];
					Throw[$Failed]
		]
		,
		Message[ServiceExecute::nparam,"Event","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"Type"],
		(
			type = Lookup[args,"Type"];
			Switch[type,
					"general"|"unique"|"average"|"General"|"Unique"|"Average",
						type = (ToLowerCase[First[#]]<>Rest[#])& @ Characters[type],
					_,
						Message[ServiceExecute::nval,"Type","Mixpanel"];
						Throw[$Failed]
			]
		),
		type = "general";
	];
	
	If[KeyExistsQ[args,"DateWindow"],
		(
			window = Lookup[args,"DateWindow"];
			Switch[window,				
					Quantity[_Integer?(#>0&),"Minutes" | "Hours" | "Days" | "Weeks" | "Months"],
						interval = QuantityMagnitude[window]; unit = StringDrop[ToLowerCase[QuantityUnit[window]],-1];
						If[interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,
					
					_String,
						var = Quiet[Interpreter["Quantity"][window]];
						If[!MatchQ[var,Quantity[_Integer?(#>0&),"Minutes" | "Hours" | "Days" | "Weeks" | "Months"]],
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						interval = QuantityMagnitude[var]; unit = StringDrop[ToLowerCase[QuantityUnit[var]],-1];
						If[interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,
					
					{_Integer?(#>0&),_String},
						var = Quiet[Quantity@@window];
						If[!MatchQ[var, Quantity[_Integer?(#>0&),"Minutes" | "Hours" | "Days" | "Weeks" | "Months"]],
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						interval = QuantityMagnitude[var]; unit = StringDrop[ToLowerCase[QuantityUnit[var]],-1];
						If[interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,

					{_DateObject,_String},
						unit = StringDrop[ToLowerCase[Last@window],-1]; var = StringDrop[ToUpperCase[First[Characters[Last@window]]]<>Rest[Characters[Last@window]],-1];
						If[!MatchQ[var, "Minute" | "Hour" | "Day" | "Week" | "Month"],
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						interval = Ceiling[QuantityMagnitude[DateDifference[First@window,Now,var]]];
						If[interval < 0 || interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						
					,

					True,
					Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
					Throw[$Failed]
				]
		),
		interval = 30;
		unit = "day";
	];

	raw = KeyClient`rawkeydata[id,"RawEvents",{"event" -> events, "type" -> type, "unit" -> unit, "interval" -> interval}];

	Which[
			raw[[1]] === 200,
				data = mixpanelimportdata@raw; pvalues = Replace[data["data"]["values"],{asoc_Association :> KeyMap[If[StringMatchQ[#, Alternatives["mp_" ~~ ___, "$" ~~ ___]], 
						camelcase[#,{"mp_","_","$"}], #] &, asoc]}];
				If[pvalues===<||>,
					Dataset[Missing["Not Available"]]
					,
					svalues = If[MatchQ[unit,"minute"|"hour"],DateObject/@Take[data["data"]["series"],{2, -2}],DateObject/@data["data"]["series"]];
					svrep = If[MatchQ[unit,"minute"|"hour"],Thread[Rule[Take[data["data"]["series"],{2, -2}],svalues]],Thread[Rule[data["data"]["series"],svalues]]];
					pvaluesObj = Replace[pvalues,{asoc_Association :> KeyMap[Replace[#, svrep] &, asoc]},1];
					setvalues = Transpose@Replace[pvaluesObj,{x_Association :> KeySortBy[KeySelect[x, MemberQ[svalues, #] &],AbsoluteTime]},1];
					Dataset[MapThread[Prepend, {Values[setvalues], ("Date" -> #) & /@ Keys[setvalues]}]]
				],
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]
]

mixpanelcookeddata["EventTypes",id_, args_?OptionQ] :=  Block[{invalidParameters, raw, type, limit},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"Type","MaxItems",MaxItems},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"Type"],
		(
			type = Lookup[args,"Type"];
			Switch[type,
					"general"|"unique"|"average"|"General"|"Unique"|"Average",
						type = (ToLowerCase[First[#]]<>Rest[#])& @ Characters[type],
					_,
						Message[ServiceExecute::nval,"Type","Mixpanel"];
						Throw[$Failed]
			]
		),
		type = "general";
	];
	
	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		(
			limit = First@DeleteMissing[Lookup[args,{"MaxItems", MaxItems}]];
			If[!(IntegerQ[limit] && 10000>=limit>0),
			(
				Message[ServiceExecute::nval,MaxItems,"Mixpanel"];
				Throw[$Failed]
			)]
		),
		limit = 255;
	];
	
	raw = KeyClient`rawkeydata[id,"RawEventsNames",{"type" -> type, "limit" -> limit}];

	Which[
			raw[[1]] === 200,
				mixpanelimportdata@raw,
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]

]

mixpanelcookeddata["TopEvents",id_, args_?OptionQ] :=  Block[{invalidParameters, raw, type, limit, data},

	invalidParameters = Select[Keys[args],!MemberQ[{"Type","MaxItems",MaxItems},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"Type"],
		(
			type = Lookup[args,"Type"];
			Switch[type,
					"general"|"unique"|"average"|"General"|"Unique"|"Average",
						type = (ToLowerCase[First[#]]<>Rest[#])& @ Characters[type],
					_,
						Message[ServiceExecute::nval,"Type","Mixpanel"];
						Throw[$Failed]
			]
		),
		type = "general";
	];
	
	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		(
			limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
			If[!(IntegerQ[limit] && 100>=limit>0),
			(
				Message[ServiceExecute::nval,MaxItems,"Mixpanel"];
				Throw[$Failed]
			)]
		),
		limit = 100;
	];

	raw = KeyClient`rawkeydata[id,"RawEventsTop",{"type" -> type, "limit" -> limit}];

	Which[
			raw[[1]] === 200,
				data = Replace[(mixpanelimportdata@raw)["events"], asoc_Association :> KeyMap[camelcase, asoc], 1];
				data[[All, "Event"]] = If[StringMatchQ[#, Alternatives["mp_" ~~ ___, "$" ~~ ___]], camelcase[#,{"mp_","_","$"}], #] & /@ data[[All, "Event"]];
				Dataset[RotateLeft/@data],
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]
]

mixpanelcookeddata["Properties",id_, args_?OptionQ] :=  Block[{invalidParameters, raw, data, event, name, nametype, values, type, window, var, unit, interval, maxItems, dateOrder, pvalues, pvaluesObj, svalues, setvalues, svrep, datalist},

	invalidParameters = Select[Keys[args],!MemberQ[{"Event", "Property", "Type", "DateWindow", "MaxItems", MaxItems, "Values"},#]&]; 

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Event"],
		(
			event = Lookup[args,"Event"];
			If[!StringQ[event],
				Message[ServiceExecute::nval,"Event","Mixpanel"];
				Throw[$Failed],
			event = event/.$mprsrvnamelist
			]
		),
		Message[ServiceExecute::nparam,"Event","Mixpanel"];
		Throw[$Failed]
	];

	If[KeyExistsQ[args,"Property"],
		(
			name = Lookup[args,"Property"];
			Switch[name,
					_String,
					nametype = "single";
					name = name/.$mprsrvnamelist,

					{_String..},
					nametype = "multiple";
					name = name/.$mprsrvnamelist,

					_,
					Message[ServiceExecute::nval,"Property","Mixpanel"];
					Throw[$Failed]
				]
		),
		Message[ServiceExecute::nparam,"Property","Mixpanel"];
		Throw[$Failed]
	];

	If[KeyExistsQ[args,"Type"],
		(
			type = Lookup[args,"Type"];
			Switch[type,
					"general"|"unique"|"average"|"General"|"Unique"|"Average",
						type = (ToLowerCase[First[#]]<>Rest[#])& @ Characters[type],
					_,
						Message[ServiceExecute::nval,"Type","Mixpanel"];
						Throw[$Failed]
			]
		),
		type = "general";
	];

	If[KeyExistsQ[args,"DateWindow"],
		(
			window = Lookup[args,"DateWindow"];
			Switch[window,				
					Quantity[_Integer?(#>0&),"Minutes" | "Hours" | "Days" | "Weeks" | "Months"],
						interval = QuantityMagnitude[window]; unit = StringDrop[ToLowerCase[QuantityUnit[window]],-1];
						If[interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,
					
					_String,
						var = Quiet[Interpreter["Quantity"][window]];
						If[!MatchQ[var,Quantity[_Integer?(#>0&),"Minutes" | "Hours" | "Days" | "Weeks" | "Months"]],
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						interval = QuantityMagnitude[var]; unit = StringDrop[ToLowerCase[QuantityUnit[var]],-1];
						If[interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,
					
					{_Integer?(#>0&),_String},
						var = Quiet[Quantity@@window];
						If[!MatchQ[var, Quantity[_Integer?(#>0&),"Minutes" | "Hours" | "Days" | "Weeks" | "Months"]],
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						interval = QuantityMagnitude[var]; unit = StringDrop[ToLowerCase[QuantityUnit[var]],-1];
						If[interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,

					{_DateObject,_String},
						unit = StringDrop[ToLowerCase[Last@window],-1]; var = StringDrop[ToUpperCase[First[Characters[Last@window]]]<>Rest[Characters[Last@window]],-1];
						If[!MatchQ[var, "Minute" | "Hour" | "Day" | "Week" | "Month"],
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						interval = Ceiling[QuantityMagnitude[DateDifference[First@window,Now,var]]];
						If[interval < 0 || interval > (unit/.$mpunitlimit),
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						];
						
					,

					True,
					Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
					Throw[$Failed]
				]
		),
		interval = 30;
		unit = "day";
	];

	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		(
			maxItems = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
			If[!(IntegerQ[maxItems] && 10000>=maxItems>0),
			(
				Message[ServiceExecute::nval,MaxItems,"Mixpanel"];
				Throw[$Failed]
			)]
		),
		maxItems = 255;
	];

	Which[
			(nametype === "multiple") && KeyExistsQ[args,"Values"],
					Message[ServiceExecute::mppropv];
					Throw[$Failed],

			(nametype === "single"),
			If[KeyExistsQ[args,"Values"],
				(
				values = Lookup[args,"Values"];
				Switch[values,
					All,
					
						raw = KeyClient`rawkeydata[id,"RawEventProperties", {"event" -> event, "name" -> name, "type" -> type, "interval" -> interval, "unit" -> unit, "limit" -> maxItems}];

						Which[
								raw[[1]] === 200,
								data = mixpanelimportdata@raw;
								
								True,
									Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
									Throw[$Failed]
						],
				
					{(_String | _?NumberQ)..},
					values = "[" <> Fold[(ToString@#1)<>", "<>(ToString@#2)&,If[StringQ@#,"\""~~#~~"\"",#]&/@values] <> "]";
					raw = KeyClient`rawkeydata[id,"RawEventProperties", {"event" -> event, "name" -> name, "type" -> type, "interval" -> interval, "unit" -> unit, "limit" -> maxItems, "values" -> values}];

						Which[
								raw[[1]] === 200,
								data = mixpanelimportdata@raw,
								
								True,
									Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
									Throw[$Failed]
						],
					
					_,
					Message[ServiceExecute::nval,"Values","Mixpanel"];
					Throw[$Failed]
				]),
				raw = KeyClient`rawkeydata[id,"RawEventProperties", {"event" -> event, "name" -> name, "type" -> type, "interval" -> interval, "unit" -> unit, "limit" -> maxItems}];
					Which[
							raw[[1]] === 200,
							data = mixpanelimportdata@raw,
							
							True,
								Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
								Throw[$Failed]
					]
			];
			pvalues = data["data"]["values"];
				If[pvalues===<||>,
					Dataset[<||>],
					If[KeyExistsQ[pvalues, "undefined"], pvalues = KeyMap[Replace[#, "undefined" :> Missing["NotApplicable"]] &, pvalues]];
					svalues = If[MatchQ[unit,"minute"|"hour"],DateObject/@Take[data["data"]["series"],{2, -2}],DateObject/@data["data"]["series"]];
					svrep = If[MatchQ[unit,"minute"|"hour"],Thread[Rule[Take[data["data"]["series"],{2, -2}],svalues]],Thread[Rule[data["data"]["series"],svalues]]];
					pvaluesObj = Replace[pvalues,{asoc_Association :> KeyMap[Replace[#, svrep] &, asoc]},1];
					setvalues = Transpose@Replace[pvaluesObj,{x_Association :> KeySortBy[KeySelect[x, MemberQ[svalues, #] &],AbsoluteTime]},1];
					Dataset[MapThread[Prepend, {Values[setvalues], ("Date" -> #) & /@ Keys[setvalues]}]]
				]
			,

			(nametype === "multiple") && !KeyExistsQ[args,"Values"],
			datalist = Module[{},
			raw = KeyClient`rawkeydata[id,"RawEventProperties", {"event" -> event, "name" -> #, "type" -> type, "interval" -> interval, "unit" -> unit, "limit" -> maxItems}];
				Which[
						raw[[1]] === 200,
						data = mixpanelimportdata@raw,
						
						True,
							Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
							Throw[$Failed]
				];
				pvalues = data["data"]["values"];
					If[pvalues===<||>,
						Dataset[<||>],
						If[KeyExistsQ[pvalues, "undefined"], pvalues = KeyMap[Replace[#, "undefined" :> Missing["NotApplicable"]] &, pvalues]];
						svalues = If[MatchQ[unit,"minute"|"hour"],DateObject/@Take[data["data"]["series"],{2, -2}],DateObject/@data["data"]["series"]];
						svrep = If[MatchQ[unit,"minute"|"hour"],Thread[Rule[Take[data["data"]["series"],{2, -2}],svalues]],Thread[Rule[data["data"]["series"],svalues]]];
						pvaluesObj = Replace[pvalues,{asoc_Association :> KeyMap[Replace[#, svrep] &, asoc]},1];
						setvalues = Transpose@Replace[pvaluesObj,{x_Association :> KeySortBy[KeySelect[x, MemberQ[svalues, #] &],AbsoluteTime]},1];
						<|Replace[str_String :> camelcase[str, {"mp_", "_", "$"}]][#]->MapThread[Prepend, {Values[setvalues], ("Date" -> #) & /@ Keys[setvalues]}]|>
					]
			]&/@name;
			Dataset[datalist]
	]
]

mixpanelcookeddata["TopProperties",id_, args_?OptionQ] :=  Block[{invalidParameters, raw, data, event, limit},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"Event", "MaxItems", MaxItems},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"Event"],
		(
			event = Lookup[args,"Event"];
			If[!StringQ[event],
				Message[ServiceExecute::nval,"Event","Mixpanel"];
				Throw[$Failed],
				event = event/.$mprsrvnamelist
			]
		),
		Message[ServiceExecute::nparam,"Event","Mixpanel"];
		Throw[$Failed]
	];
	
	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		(
			limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
			If[!(IntegerQ[limit] && 10000>=limit>0),
			(
				Message[ServiceExecute::nval,MaxItems,"Mixpanel"];
				Throw[$Failed]
			)]
		),
		limit = 10;
	];

	raw = KeyClient`rawkeydata[id,"RawEventPropertiesTop",{"event" -> event, "limit" -> limit}];

	Which[
			raw[[1]] === 200,
				data = mixpanelimportdata@raw;
				Map[#["count"]&, Replace[data, {asoc_Association :> KeyMap[If[StringMatchQ[#, Alternatives["mp_" ~~ ___, "$" ~~ ___]],
					camelcase[#,{"mp_","_","$"}], #] &, asoc]}]],
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]
]

mixpanelcookeddata["PropertyValues",id_, args_?OptionQ] :=  Block[{invalidParameters, raw, data, event, name, nametype, bucket, limit},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"Event", "Property", "MaxItems", MaxItems, "Bucket"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"Event"],
		(
			event = Lookup[args,"Event"];
			If[!StringQ[event],
				Message[ServiceExecute::nval,"Event","Mixpanel"];
				Throw[$Failed],
			event = event/.$mprsrvnamelist
			]
		),
		Message[ServiceExecute::nparam,"Event","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"Property"],
		(
			name = Lookup[args,"Property"];
			Switch[name,
					_String,
					nametype = "single";
					name = name/.$mprsrvnamelist,

					{_String..},
					nametype = "multiple";
					name = name/.$mprsrvnamelist,

					_,
					Message[ServiceExecute::nval,"Property","Mixpanel"];
					Throw[$Failed]
				]
		),
		Message[ServiceExecute::nparam,"Property","Mixpanel"];
		Throw[$Failed]
	];

	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		(
			limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
			If[!(IntegerQ[limit] && 10000>=limit>0),
			(
				Message[ServiceExecute::nval,MaxItems,"Mixpanel"];
				Throw[$Failed]
			)]
		),
		limit = 10;
	];

	If[nametype === "multiple",
	Module[{},
		If[KeyExistsQ[args,"Bucket"],
			(
				bucket = ToString[Lookup[args,"Bucket"]];
				raw = KeyClient`rawkeydata[id,"RawEventPropertiesValues",{"event" -> event, "name" -> #, "limit"->limit, "bucket"-> bucket}];
				Which[
						raw[[1]] === 200,
							data = mixpanelimportdata@raw,
							
						True,
							Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
							Throw[$Failed]
				]
			),
		raw = KeyClient`rawkeydata[id,"RawEventPropertiesValues",{"event" -> event, "name" -> #, "limit"->limit}];
		Which[
				raw[[1]] === 200,
					data = mixpanelimportdata@raw,
					
				True,
					Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
					Throw[$Failed]
			]
		]
	]&/@name,
	If[KeyExistsQ[args,"Bucket"],
		(
			bucket = Lookup[args,"Bucket"];
			raw = KeyClient`rawkeydata[id,"RawEventPropertiesValues",{"event" -> event, "name" -> name, "limit"->limit, "bucket"-> bucket}];
			Which[
					raw[[1]] === 200,
						data = mixpanelimportdata@raw,
						
					True,
						Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
						Throw[$Failed]
			]
		),
		raw = KeyClient`rawkeydata[id,"RawEventPropertiesValues",{"event" -> event, "name" -> name, "limit"->limit}];
		Which[
				raw[[1]] === 200,
					data = mixpanelimportdata@raw,
					
				True,
					Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
					Throw[$Failed]
			]
		]
	]
]

mixpanelcookeddata["Funnels",id_, args_?OptionQ] :=  Block[{raw,data},
	
	raw = KeyClient`rawkeydata[id,"RawFunnelsList",{}];
	Which[
			raw[[1]] === 200,
				data = mixpanelimportdata@raw;
				Dataset[AssociationThread[{"FunnelID", "Name"} -> Values[#]] & /@ data],
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
	]
]

mixpanelcookeddata["Funnel",id_, args_?OptionQ] :=  Block[{invalidParameters, dateRules, raw, errmsg, data, camelData, funnelId, deadline, window, startD, endD, var, unit},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"FunnelID", "DateWindow", "CompletionDeadline"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Mixpanel"]&/@invalidParameters;
			Throw[$Failed]
		)];
		
	If[KeyExistsQ[args,"FunnelID"],
		(
			funnelId = ToString[Lookup[args,"FunnelID"]];
			If[KeyExistsQ[mixpanelimportdata@KeyClient`rawkeydata[id,"RawFunnels",{"funnel_id"->funnelId}],"error"],
			(
				Message[ServiceExecute::nval,"FunnelID","Mixpanel"];
				Throw[$Failed]
			)];
		),
		Message[ServiceExecute::nparam,"FunnelID","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"CompletionDeadline"],
		(
			deadline = Lookup[args,"CompletionDeadline"];
			If[!MatchQ[deadline,x_/;IntegerQ[x]&& x>0 && x<61],
			(
				Message[ServiceExecute::nval,"CompletionDeadline","Mixpanel"];
				Throw[$Failed]
			)]
		),
		Message[ServiceExecute::nparam,"CompletionDeadline","Mixpanel"];
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"DateWindow"],
		(
			window = Lookup[args,"DateWindow"];
			Switch[window,
					Quantity[_Integer?(9>#>0&),"Weeks"]|Quantity[_Integer?(61>#>0&),"Days"],
								unit = StringDrop[ToLowerCase[QuantityUnit[window]],-1];
								endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
								startD = DateString[Today - window, {"Year", "-", "Month", "-", "Day"}]
					,
					
					_String,
						var = Quiet@Interpreter["Quantity"][window];
						If[MatchQ[var,Quantity[_Integer?(9>#>0&),"Weeks"]|Quantity[_Integer?(61>#>0&),"Days"]],
							(
							unit = StringDrop[ToLowerCase[QuantityUnit[var]],-1];
							endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
							startD = DateString[Today - var, {"Year", "-", "Month", "-", "Day"}]
							),
						Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
						Throw[$Failed]
						]

					,
					
					{_Integer?(61>#>0&), "Days" | "Weeks" | "days" | "weeks"},
						unit = StringDrop[ToLowerCase@window[[2]],-1];
						window = UnitConvert[Quantity[window[[1]],(unit/.{"day"->"Day","week"->"Week"})],"Days"];
						endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
						startD = DateString[Today - window, {"Year", "-", "Month", "-", "Day"}]
					,

					{_DateObject,_DateObject, "Days" | "Weeks" | "days" | "weeks"},
						unit = StringDrop[ToLowerCase@window[[3]],-1];
						endD = DateString[window[[2]], {"Year", "-", "Month", "-", "Day"}];
						startD = DateString[window[[1]], {"Year", "-", "Month", "-", "Day"}];
						var = First@DateDifference[window[[1]],window[[2]],"Day"];
						If[var > 60 || var < 0,
							Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
							Throw[$Failed]
						]
					,

					_,
					Message[ServiceExecute::nval,"DateWindow","Mixpanel"];
					Throw[$Failed]
				]
		),
		startD = DateString[DatePlus[Today,-30], {"Year", "-", "Month", "-", "Day"}];
		endD = DateString[Today, {"Year", "-", "Month", "-", "Day"}];
		unit = "day";
	];

	raw = KeyClient`rawkeydata[id,"RawFunnels",{"funnel_id" -> funnelId ,"from_date" -> startD,"to_date" -> endD,"length"-> deadline,"unit" -> unit}];
	Which[
			raw[[1]] === 200,
				data = Lookup[mixpanelimportdata@raw,"data"];
				data = Replace[data, asoc_Association :> KeyMap[Replace[#, $mpfunnelcamel]&, asoc], {2, 3}];
				camelData = Replace[data, str_String?(StringMatchQ[#, Alternatives["mp_" ~~ __, "$" ~~ __]]&):> camelcase[str,{"mp_","_","$"}], {4}];
				dateRules = KeySortBy[KeyMap[DateObject, camelData], (AbsoluteTime[#] &)];
				Dataset[Map[KeyMap[Replace[#, {"steps" -> "Steps","analysis" -> "Analysis"}] &, #] &, dateRules]],
				
			True,
				Message[ServiceExecute::serrormsg,Lookup[mixpanelimportdata@raw,"error"]];
				Throw[$Failed]
		]
]

mixpanelcookeddata[___]:=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{MixpanelAPI`Private`mixpaneldata,MixpanelAPI`Private`mixpanelcookeddata}
