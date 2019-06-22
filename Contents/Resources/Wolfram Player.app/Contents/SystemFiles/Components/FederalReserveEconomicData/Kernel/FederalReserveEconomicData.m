Begin["FederalReserveEconomicData`"]

ServiceExecute::fredfl="`1` request receives up to one filter parameter."

Begin["`Private`"]

(******************************* FederalReserveEconomicData *************************************)

(* Authentication information *)

freddata[]:={
		"ServiceName" 		-> "FederalReserveEconomicData", 
        "URLFetchFun"		:> (URLRead[HTTPRequest[#1, <|"Query" -> Lookup[{##2}, "Parameters", {}], Method -> Lookup[{##2}, "Method"]|>]]&),
        "ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["FederalReserveEconomicData", {"API Key" -> "api_key"},
        								"https://research.stlouisfed.org/useraccount/register/step1","https://api.stlouisfed.org/terms_of_use.html"],
	 	"Gets"				-> {"SeriesSearch","SeriesData"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawSeriesSearch","RawSeriesObservations"},
	 	"RawPosts"			-> {},
 		"Information"		-> "Import FRED API data to the Wolfram Language"
}

(**** Raw Requests ****)

(* Documentation here: https://api.stlouisfed.org/docs/fred/series_search.html *)
freddata["RawSeriesSearch"] := {
        "URL"				-> "https://api.stlouisfed.org/fred/series/search",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"file_type","search_text","search_type","realtime_start","realtime_end","limit",
        						"offset","order_by","sort_order","filter_variable","filter_value","tag_names","exclude_tag_names"},
        "RequiredParameters"-> {"search_text"},
        "ResultsFunction"	-> fredimport
    }

(* Documentation here: https://api.stlouisfed.org/docs/fred/series_observations.html *)    
freddata["RawSeriesObservations"] := {
        "URL"				-> "https://api.stlouisfed.org/fred/series/observations",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"file_type","series_id","realtime_start","realtime_end","limit",
        						"offset","sort_order","observation_start","observation_end","units",
        						"frequency","aggregation_method","output_type","vintage_dates"},
        "RequiredParameters"-> {"series_id"},
        "ResultsFunction"	-> fredimport
    }
    
freddata[___]:=$Failed   
   
(**** Cooked Requests ****)

fredcookeddata["SeriesSearch", id_, args_?OptionQ] := Module[{invalidParameters, params={"file_type"->"json"}, filter, 
													data, totalResults, items = {}, progress = 0, residual, limit, maxPerPage=1000, startIndex, 
													calls, result, sort, sortVal, sortDir, query},
	invalidParameters = Select[Keys[args],!MemberQ[{"Query",MaxItems,"StartIndex","SortBy","Frequency","Units","SeasonalAdjustment"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"FederalReserveEconomicData"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	If[KeyExistsQ[args,"Query"],
		query = Lookup[args,"Query"];
		AppendTo[params,"search_text" -> query]			
		,
		Message[ServiceExecute::nparam,"Query"];			
		Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"SortBy"],
	(
		sort = Lookup[args,"SortBy"];
		If[StringQ[sort],
		(
			(* Default descending *)
			sort = {sort,"Ascending"}									
		)];
		If[MatchQ[sort, {_String, _String}],
		(
			Switch[sort[[2]],
					"Ascending",
						sortDir = "asc",
					"Descending",
						sortDir = "desc",
					_,
						Message[ServiceExecute::nval,"SortBy","FederalReserveEconomicData"];	
						Throw[$Failed]
			];
			Switch[sort[[1]],
				"SeriesID",
					sortVal = "series_id",
				"Title",
					sortVal = "title",
				"Units",
					sortVal = "units",
				"Frequency",
					sortVal = "frequency",
				"SeasonalAdjustment",
					sortVal = "seasonal_adjustment",
				(*"Date",
					sortVal = "realtime_start",*)
				"LastUpdated",
					sortVal = "last_updated",
				"ObservationDate",
					sortVal = "observation_start",
				"Popularity",
					sortVal = "popularity",
				_,
					Message[ServiceExecute::nval,"SortBy","FederalReserveEconomicData"];	
					Throw[$Failed]
			];			
		)];
		AppendTo[params,"order_by"->sortVal];
		AppendTo[params,"sort_order"->sortDir];	
	)];
	
	If[Plus@@Boole[{KeyExistsQ[args,"Frequency"],KeyExistsQ[args,"Units"],KeyExistsQ[args,"SeasonalAdjustment"]}]>1,
		Message[ServiceExecute::fredfl,"SeriesSearch"];			
		Throw[$Failed]];
	
	If[KeyExistsQ[args,"Frequency"],
		filter = Lookup[args,"Frequency"];
		AppendTo[params,"filter_variable"->"frequency"];
		AppendTo[params,"filter_value"->filter];			
	];
	
	If[KeyExistsQ[args,"Units"],
		filter = Lookup[args,"Units"];
		AppendTo[params,"filter_variable"->"units"];
		AppendTo[params,"filter_value"->filter];	
	];

	If[KeyExistsQ[args,"SeasonalAdjustment"],
		filter = Lookup[args,"SeasonalAdjustment"];
		Switch[filter,
				True|"Seasonally Adjusted",
					filter = "Seasonally Adjusted",
				False|"Not Seasonally Adjusted",
					filter = "Not Seasonally Adjusted",
				"Seasonally Adjusted Annual Rate",
					filter = "Seasonally Adjusted Annual Rate",
				_,
					Message[ServiceExecute::nval,"SeasonalAdjustment","FederalReserveEconomicData"];	
					Throw[$Failed]
			];
		AppendTo[params,"filter_variable"->"seasonal_adjustment"];
		AppendTo[params,"filter_value"->filter];			
	];
	
	If[KeyExistsQ[args,MaxItems],
		limit = Lookup[args,MaxItems];
		If[(!IntegerQ[limit] && limit > 0),
			Message[ServiceExecute::nval,MaxItems,"FederalReserveEconomicData"];
			Throw[$Failed]
		],
		limit = maxPerPage;
	];
	
	If[KeyExistsQ[args,"StartIndex"],
		startIndex = Lookup[args,"StartIndex"];
		If[!IntegerQ[startIndex],
			Message[ServiceExecute::nval,"StartIndex","FederalReserveEconomicData"];
			Throw[$Failed]
		],
		startIndex = 0		
	];

	(* regularized startIndex *)
	calls = Quotient[limit, maxPerPage];
	residual = limit - (calls*maxPerPage);
	
	AppendTo[params,"limit"->ToString[maxPerPage]];
	AppendTo[params,"offset"->ToString[startIndex]];

	(* this prints the progress indicator bar *)
	PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
	If[calls > 0,
	(
		(	
			params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+#*maxPerPage]]];
		
			data = fredimport@KeyClient`rawkeydata[id,"RawSeriesSearch",params];

			totalResults = Lookup[data,"count"];
			items = Join[items, If[totalResults>0,Lookup[data,"seriess"],{}]];		
			progress = progress + 1;	
		)& /@ Range[0,calls-1];		
	)];
	If[residual > 0,
	(
		params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+calls*maxPerPage]]];
		params = ReplaceAll[params,Rule["limit",_] -> Rule["limit",ToString[residual]]];
		
		data = fredimport@KeyClient`rawkeydata[id,"RawSeriesSearch",params];
	
		totalResults = Lookup[data,"count"];
		items = Join[items, If[totalResults>0,Lookup[data,"seriess"],{}]];		
	)];
	
	result = Take[items,UpTo[limit]];
	result = Block[{var},
				var = AssociationThread[Rule[{"ID", "Title", "Frequency", "Units", "SeasonalAdjustment", "StartDate", "EndDate", "LastUpdated", "Notes"},
				Lookup[#,{"id","title","frequency","units","seasonal_adjustment","observation_start",
							"observation_end","last_updated","notes"},Missing["NotAvailable"]]]];
				AssociateTo[var,Rule["StartDate", Quiet[Check[DateObject[Lookup[var,"StartDate"]], Lookup[var,"StartDate"]]]]];
				AssociateTo[var,Rule["EndDate", Quiet[Check[DateObject[Lookup[var,"EndDate"]], Lookup[var,"StartDate"]]]]];
				AssociateTo[var,Rule["LastUpdated", Quiet[Check[DateObject[formatDate[Lookup[var,"LastUpdated"]]], Lookup[var,"StartDate"]]]]];
				AssociateTo[var,Rule["Notes", If[!MissingQ[Lookup[var,"Notes"]],StringReplace[var["Notes"],{(((""|" ")~~"\r\n")..) :> "\n", (((""|" ")~~"\n")..) :> "\n"}],Missing["NotAvailable"]]]]
				]&/@result;
												
	Dataset[result]
]

fredcookeddata["SeriesData", id_, args_?OptionQ] := Module[{invalidParameters, params={"file_type"->"json"}, ids, dateRange, 
													startDate, endDate, data, totalResults, items = {}, timeSeries, result},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"ID","Date"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"FederalReserveEconomicData"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	If[KeyExistsQ[args,"ID"],
		(
			ids = Lookup[args,"ID"];
			If[StringQ[ids], ids = {ids}];
			If[!MatchQ[ids,{___String}],
			(	
				Message[ServiceExecute::nval,"ID","FederalReserveEconomicData"];
				Throw[$Failed]
			)];
		),
		(
			Message[ServiceExecute::nparam,"ID"];			
			Throw[$Failed]
		)
	];
	
	If[KeyExistsQ[args,"Date"],
	(
		dateRange = Lookup[args,"Date"];
		
		{startDate, endDate} = parseDates[dateRange];
		
		If[!(DateObjectQ[startDate]&&DateObjectQ[endDate]),
			Message[ServiceExecute::nval,"Date","FederalReserveEconomicData"];	
			Throw[$Failed]
		];
		AppendTo[params,"observation_start"->DateString[startDate,{"Year", "-", "Month", "-", "Day"}]];
		AppendTo[params,"observation_end"->DateString[endDate,{"Year", "-", "Month", "-", "Day"}]];	
	)];
	
	result = {};
	AppendTo[params, "series_id" -> ""];
	Function[i,
	(
		params = Replace[params,Rule["series_id",_]:>Rule["series_id",i],{1}];
		data = fredimport@KeyClient`rawkeydata[id,"RawSeriesObservations",params];
	
		totalResults = Lookup[data,"count"];
		items = Join[items, If[totalResults>0,Lookup[data,"observations"],{}]];
		timeSeries = ({Take[DateList[Lookup[#, "date"]], 3], If[StringMatchQ[Lookup[#, "value"], "."], Missing[], Internal`StringToDouble[Lookup[#, "value"]]]} &) /@ items;
		If[Length[timeSeries]>0,
			result = Append[result,i->TimeSeries[timeSeries]],
			result = Append[result,i-><||>]
		]
	)] /@ ids;
	
	If[Length[result] == 1,
		result[[1,2]],
		Association[result]
	]
]

fredcookeddata[___]:=$Failed

fredsendmessage[___]:=$Failed

(* Utilities *)

fredimport[raw_] := With[{res = Quiet[Developer`ReadRawJSONString@raw["Body"]]},
	If[UnsameQ[raw["StatusCode"], 200],
		If[AssociationQ[res],
			Message[ServiceExecute::serrormsg, Lookup[res,"error_message"]]
			,
			Message[ServiceExecute::serrormsg, res["StatusCodeDescription"]]
		];
		Throw[$Failed],
		If[AssociationQ[res],
			res,
			Message[ServiceExecute::serror];
			Throw[$Failed]
		]
	]
]

parseDates[param_] := Block[{startDate, endDate},
	(
   		Switch[param,
    		_String,
     			startDate = Quiet[Check[DateObject[param], ""]];
     			endDate = DateObject["9999-12-31"];
		    ,
    		_DateObject,
     			startDate = param;
     			endDate = DateObject["9999-12-31"]
     			,
    		{_DateObject|_String,_DateObject|_String},
     			startDate = Quiet[Check[DateObject[First@param], ""]];
				endDate = Quiet[Check[DateObject[Last@param], ""]];
		    ,
    		Interval[{_DateObject, _DateObject}],
		    (
     			startDate = First@@param;
     			endDate = Last@@param;
     		),
		    _,
		    (
     			startDate = "";
     			endDate = "";
		    )
		];
   		{startDate, endDate}
   	)]

formatDate[date_] := Block[{dateSplit, d, tz}, 
	(
		d = First@StringCases[date, RegularExpression["[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"]];
		tz = StringDelete[date, d];
   		If[StringLength[tz] > 0, tz = Internal`StringToDouble[tz], tz = 0];
   		DateObject[d, TimeZone -> tz]
   	)]

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{FederalReserveEconomicData`Private`freddata,FederalReserveEconomicData`Private`fredcookeddata,FederalReserveEconomicData`Private`fredsendmessage}
