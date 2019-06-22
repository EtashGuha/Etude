Begin["CrossRef`"] (* Begin Private Context *) 

Begin["`Private`"](* Begin Private Context *) 

(******************************* CrossRef *************************************)

(* Authentication information *)

crossrefdata[]:={
		"ServiceName" 		-> "CrossRef", 
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
								URLFetch[#1, {"StatusCode","Content"}, Sequence@@FilterRules[{##2},Except["Parameters"]],
								"Parameters"-> Cases[params,Except[Rule["apikey",_]]]]]&),
        "ClientInfo"		:> {},
	 	"Gets"				-> {"WorksList","WorksDataset","WorkTypes","WorkInformation","FunderInformation",
	 							"OwnerPrefixInformation","MemberInformation","TypeInformation","JournalInformation",
	 							"FunderList","FunderDataset","MemberList","MemberDataset","LicenseList","LicenseDataset",
	 							"JournalList","JournalDataset"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawWorks","RawFunders","RawMembers","RawTypes","RawLicenses","RawJournals","RawWorkInformation","RawFunderInformation",
	 							"RawOwnerPrefixInformation","RawMemberInformation","RawTypeInformation","RawJournalInformation","RawFunderWorks","RawWorksOfType",
	 							"RawWorksByOwnerPrefix","RawMemberWorks","RawJournalWorks","RawWorkAgency"},
	 	"RawPosts"			-> {},
 		"Information"		-> "Import CrossRef API data to the Wolfram Language"
}

(**** Raw Requests ****)

basicparams = {"query","filter","rows","offset","sample","sort","order","facet"}

crossrefdata["RawWorks"] := {
        "URL"				-> "http://api.crossref.org/works",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawFunders"] := {
        "URL"				-> "http://api.crossref.org/funders",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query","filter","rows","offset","sample","facet"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawMembers"] := {
        "URL"				-> "http://api.crossref.org/members",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query","filter","rows","offset","sample","facet"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawTypes"] := {
        "URL"				-> "http://api.crossref.org/types",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawLicenses"] := {
        "URL"				-> "http://api.crossref.org/licenses",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query","rows","sample","facet"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawJournals"] := {
        "URL"				-> "http://api.crossref.org/journals",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query","rows","offset","sample","facet"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawWorkInformation"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/works/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"doi"},
        "RequiredParameters"-> {"doi"},
        "ResultsFunction"	-> formatresults
    }
    
crossrefdata["RawFunderInformation"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/funders/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"funder_id"},
        "RequiredParameters"-> {"funder_id"},
        "ResultsFunction"	-> formatresults
    }
    
crossrefdata["RawOwnerPrefixInformation"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/prefixes/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"owner_prefix"},
        "RequiredParameters"-> {"owner_prefix"},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawMemberInformation"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/members/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"member_id"},
        "RequiredParameters"-> {"member_id"},
        "ResultsFunction"	-> formatresults
    }
    
crossrefdata["RawTypeInformation"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/types/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"type_id"},
        "RequiredParameters"-> {"type_id"},
        "ResultsFunction"	-> formatresults
    }
    
crossrefdata["RawJournalInformation"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/journals/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"issn"},
        "RequiredParameters"-> {"issn"},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawFunderWorks"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/funders/`1`/works", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "PathParameters"	-> {"funder_id"},
        "RequiredParameters"-> {"funder_id"},
        "ResultsFunction"	-> formatresults
    }
    
crossrefdata["RawWorksOfType"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/types/`1`/works", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "PathParameters"	-> {"type_id"},
        "RequiredParameters"-> {"type_id"},
        "ResultsFunction"	-> formatresults
    }
 
crossrefdata["RawWorksByOwnerPrefix"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/prefixes/`1`/works", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "PathParameters"	-> {"owner_prefix"},
        "RequiredParameters"-> {"owner_prefix"},
        "ResultsFunction"	-> formatresults
    }       

crossrefdata["RawMemberWorks"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/members/`1`/works", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "PathParameters"	-> {"member_id"},
        "RequiredParameters"-> {"member_id"},
        "ResultsFunction"	-> formatresults
    }
    
crossrefdata["RawJournalWorks"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/journals/`1`/works", #]&),
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> basicparams,
        "PathParameters"	-> {"issn"},
        "RequiredParameters"-> {"issn"},
        "ResultsFunction"	-> formatresults
    }

crossrefdata["RawWorkAgency"] := {
        "URL"				-> (ToString@StringForm["http://api.crossref.org/works/`1`/agency", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"doi"},
        "RequiredParameters"-> {"doi"},
        "ResultsFunction"	-> formatresults
    }
       
crossrefdata[___]:=$Failed   
   
(**** Cooked Requests ****)

crossrefcookeddata[prop:("WorksList"|"WorksDataset"), id_, args_] := Module[{license,hft,dateRange,startDate,endDate,filterParam={},reqName,invalidParameters,params={},sort,query,limit=20,maxPerPage=1000,startIndex,
																	calls,residual,progress=0,data,rawdata,totalResults,items={},result,sortDir,sortVal,
																	argsCopy,funderID,typeID,ownerPrefix,memberID,issn},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","SortBy","Query",
													"FunderID","TypeID","OwnerPrefix","MemberID","ISSN","IssuedDate",
													"UpdatedDate","DepositedDate","IndexedDate","License","HasFullText"},#]&]; 
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	(* this is the default endpoint, if other filters are specified, the last endpoint is going to be the one used *)
	reqName = "RawWorks";
	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	
	If[KeyExistsQ[args,"SortBy"],
	(
		sort = "SortBy" /. args;
		If[MatchQ[sort, {_String, _String}],
		(
			If[sort[[2]] == "Ascending", sortDir = "asc",
			(
				If[sort[[2]] == "Descending", 
					sortDir = "desc",
					(
						Message[ServiceExecute::nval,"SortBy","CrossRef"];	
						Throw[$Failed]
					)
				]
			)];		
			params = Append[params,Rule["order",sortDir]];
			sort = sort[[1]];
		)];
		Switch[sort,
			"Score",
			sortVal = "score",
			"Relevance",
			sortVal = "relevance",
			"Updated",
			sortVal = "updated",
			"Deposited",
			sortVal = "deposited",
			"Indexed",
			sortVal = "indexed",
			"Published",
			sortVal = "published",
			_,
			(
				Message[ServiceExecute::nval,"SortBy","CrossRef"];	
				Throw[$Failed]
			)
		];			
		params = Append[params,Rule["sort",sortVal]];		
	)];
	
	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"query" -> query]			
		)
	];
	
	If[KeyExistsQ[args,"FunderID"],
		(
			funderID = "FunderID" /. args;
			funderID = StringReplace[funderID,___ ~~ "/" ~~ fID___ :> fID];
			(*params = Append[params,"funder_id" -> funderID];
			reqName = "RawFunderWorks";*)
			filterParam = Append[filterParam,"funder:"~~ToString[funderID]]		
		)
	];
	
	If[KeyExistsQ[args,"TypeID"],
		(
			typeID = "TypeID" /. args;
			(*params = Append[params,"type_id" -> typeID];
			reqName = "RawWorksOfType";*)
			filterParam = Append[filterParam,"type:"~~ToString[typeID]]				
		)
	];
	
	If[KeyExistsQ[args,"OwnerPrefix"],
		(
			ownerPrefix = "OwnerPrefix" /. args;
			ownerPrefix = StringReplace[ownerPrefix,"http://id.crossref.org/prefix/" ~~ prefix_ :> prefix];
			(*params = Append[params,"owner_prefix" -> ownerPrefix];
			reqName = "RawWorksByOwnerPrefix";*)
			filterParam = Append[filterParam,"prefix:"~~ToString[ownerPrefix]]			
		)
	];
	
	If[KeyExistsQ[args,"MemberID"],
		(
			memberID = ToString["MemberID" /. args];
			memberID = StringReplace[memberID,"http://id.crossref.org/member/" ~~ m_ :> m];
			(*params = Append[params,"member_id" -> memberID];
			reqName = "RawMemberWorks";*)
			filterParam = Append[filterParam,"member:"~~ToString[memberID]]			
		)
	];
	
	If[KeyExistsQ[args,"ISSN"],
		(
			issn = "ISSN" /. args;
			(*params = Append[params,"issn" -> issn];
			reqName = "RawJournalWorks";*)
			filterParam = Append[filterParam,"issn:"~~ToString[issn]]			
		)
	];
	
	If[KeyExistsQ[args,"IssuedDate"],
	(
		dateRange = "IssuedDate" /. args;
		
		Switch[dateRange,
			_String,
			(
				startDate = DateObject[dateRange];
				endDate = startDate;
			),
			_DateObject,
			(
				startDate = dateRange;
				endDate = startDate;
			),
			List[_,_],
			(
				startDate = dateRange[[1]];
				endDate = dateRange[[2]];
		
				Switch[Head[startDate],
					String,
					startDate = DateObject[startDate],
					DateObject,
					startDate = startDate
				];
				Switch[Head[endDate],
					String,
					endDate = DateObject[endDate],
					DateObject,
					endDate = endDate
				];
			),
			Interval[{_DateObject,_DateObject}],
			(
				startDate = dateRange /. Interval[{f_,t_}]:>f;
				endDate = dateRange /. Interval[{f_,t_}]:>t;				
			),
			_,
			(
				Message[ServiceExecute::nval,"IssuedDate","CrossRef"];	
				Throw[$Failed]
			)
		];
		
		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"IssuedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"IssuedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		startDate = DateString[startDate,{"Year", "-", "Month", "-", "Day"}];
		
		endDate = DateString[endDate,{"Year", "-", "Month", "-", "Day"}];
				
		filterParam = Join[filterParam,{"from-pub-date:" ~~ startDate,"until-pub-date:" ~~ endDate}]		
		
	)];
	
	If[KeyExistsQ[args,"IndexedDate"],
	(
		dateRange = "IndexedDate" /. args;
		
		Switch[dateRange,
			_String,
			(
				startDate = DateObject[dateRange];
				endDate = startDate;
			),
			_DateObject,
			(
				startDate = dateRange;
				endDate = startDate;
			),
			List[_,_],
			(
				startDate = dateRange[[1]];
				endDate = dateRange[[2]];
		
				Switch[Head[startDate],
					String,
					startDate = DateObject[startDate],
					DateObject,
					startDate = startDate
				];
				Switch[Head[endDate],
					String,
					endDate = DateObject[endDate],
					DateObject,
					endDate = endDate
				];
			),
			Interval[{_DateObject,_DateObject}],
			(
				startDate = dateRange /. Interval[{f_,t_}]:>f;
				endDate = dateRange /. Interval[{f_,t_}]:>t;				
			),
			_,
			(
				Message[ServiceExecute::nval,"IndexedDate","CrossRef"];	
				Throw[$Failed]
			)
		];
		
		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"IndexedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"IndexedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		startDate = DateString[startDate,{"Year", "-", "Month", "-", "Day"}];
		
		endDate = DateString[endDate,{"Year", "-", "Month", "-", "Day"}];
				
		filterParam = Join[filterParam,{"from-index-date:" ~~ startDate,"until-index-date:" ~~ endDate}]		
		
	)];
	
	If[KeyExistsQ[args,"DepositedDate"],
	(
		dateRange = "DepositedDate" /. args;
		
		Switch[dateRange,
			_String,
			(
				startDate = DateObject[dateRange];
				endDate = startDate;
			),
			_DateObject,
			(
				startDate = dateRange;
				endDate = startDate;
			),
			List[_,_],
			(
				startDate = dateRange[[1]];
				endDate = dateRange[[2]];
		
				Switch[Head[startDate],
					String,
					startDate = DateObject[startDate],
					DateObject,
					startDate = startDate
				];
				Switch[Head[endDate],
					String,
					endDate = DateObject[endDate],
					DateObject,
					endDate = endDate
				];
			),
			Interval[{_DateObject,_DateObject}],
			(
				startDate = dateRange /. Interval[{f_,t_}]:>f;
				endDate = dateRange /. Interval[{f_,t_}]:>t;				
			),
			_,
			(
				Message[ServiceExecute::nval,"DepositedDate","CrossRef"];	
				Throw[$Failed]
			)
		];
		
		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"DepositedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"DepositedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		startDate = DateString[startDate,{"Year", "-", "Month", "-", "Day"}];
		
		endDate = DateString[endDate,{"Year", "-", "Month", "-", "Day"}];
				
		filterParam = Join[filterParam,{"from-deposit-date:" ~~ startDate,"until-deposit-date:" ~~ endDate}]		
		
	)];
	
	If[KeyExistsQ[args,"UpdatedDate"],
	(
		dateRange = "UpdatedDate" /. args;
		
		Switch[dateRange,
			_String,
			(
				startDate = DateObject[dateRange];
				endDate = startDate;
			),
			_DateObject,
			(
				startDate = dateRange;
				endDate = startDate;
			),
			List[_,_],
			(
				startDate = dateRange[[1]];
				endDate = dateRange[[2]];
		
				Switch[Head[startDate],
					String,
					startDate = DateObject[startDate],
					DateObject,
					startDate = startDate
				];
				Switch[Head[endDate],
					String,
					endDate = DateObject[endDate],
					DateObject,
					endDate = endDate
				];
			),
			Interval[{_DateObject,_DateObject}],
			(
				startDate = dateRange /. Interval[{f_,t_}]:>f;
				endDate = dateRange /. Interval[{f_,t_}]:>t;				
			),
			_,
			(
				Message[ServiceExecute::nval,"UpdatedDate","CrossRef"];	
				Throw[$Failed]
			)
		];
		
		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"UpdatedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"UpdatedDate","CrossRef"];	
			Throw[$Failed]
		)];
		
		startDate = DateString[startDate,{"Year", "-", "Month", "-", "Day"}];
		
		endDate = DateString[endDate,{"Year", "-", "Month", "-", "Day"}];
				
		filterParam = Join[filterParam,{"from-update-date:" ~~ startDate,"until-update-date:" ~~ endDate}]		
		
	)];
	
	If[KeyExistsQ[args,"HasFullText"],
	(
		hft = "HasFullText" /. args;
		Switch[hft,
			True,
			filterParam = Append[filterParam,"has-full-text:true"],
			False,
			filterParam = Append[filterParam,"has-full-text:false"],
			_,
			(
				Message[ServiceExecute::nval,"HasFullText","CrossRef"];	
				Throw[$Failed]
			)
		];		
	)];
	
	If[KeyExistsQ[args,"License"],
		(
			license = "License" /. args;
			filterParam = Append[filterParam,"license.url:" ~~ ToString[license]];						
		)		
	];
	
	If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(	
				Message[ServiceExecute::nval,"MaxItems","CrossRef"];
				Throw[$Failed]
			)];						
	)];
	
	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(	
				Message[ServiceExecute::nval,"StartIndex","CrossRef"];
				Throw[$Failed]
			)];
		),
		startIndex = 0		
	];
	
	calls = Quotient[limit, maxPerPage];	
	residual = limit - (calls*maxPerPage);
	
	If[Length[filterParam]>0,
	(
		filterParam = StringJoin[Riffle[filterParam,","]];
		params = Append[params,"filter"->filterParam]
	)];
	
	params = Join[params,{"rows"->ToString[maxPerPage], "offset"->ToString[startIndex]}];
	
	(* this prints the progress indicator bar *)
	PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
	If[calls > 0,
	(
		(	
			params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+#*maxPerPage]]];
			
			rawdata = KeyClient`rawkeydata[id,reqName,params];
			(*data = formatresults[rawdata];*)
			
			If[rawdata[[1]]!=200,
			(
				Message[ServiceExecute::serror];
				Throw[$Failed]
			)];
			
			data = formatresults[rawdata];
			totalResults = "total-results"/.("message"/.data);
			items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];		
			progress = progress + 1;	
		)& /@ Range[0,calls-1];		
		
	)];
	
	If[residual > 0,
	(
		params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+calls*maxPerPage]]];
		params = ReplaceAll[params,Rule["rows",_] -> Rule["rows",ToString[residual]]];
		
		rawdata = KeyClient`rawkeydata[id,reqName,params];
		(*data = formatresults[rawdata];*)
		
		If[rawdata[[1]]!=200,
		(
			Message[ServiceExecute::serror];
			Throw[$Failed]
		)];
		
		data = formatresults[rawdata];
		totalResults = "total-results"/.("message"/.data);
		items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];
	)];
	
	result = items[[1;;Min[limit,Length[items]]]];
	
	result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
	result = ReplaceAll[result,Rule["Indexed",d_]:>Rule["Indexed",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts" /. d)]];
	result = ReplaceAll[result,Rule["Issued",d_]:>Rule["Issued",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts" /. d)]];
	result = ReplaceAll[result,Rule["Deposited",d_]:>Rule["Deposited",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts" /. d)]];
	result = ReplaceAll[result,Rule["License",d_]:>Rule["License",Function[z,ReplaceAll[z,Rule["start",s_]:>Rule["start",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts"/.s)]]]/@d]];
	result = ReplaceAll[result,Rule["Author",a_]:>Rule["Author",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
	result = ReplaceAll[result,Rule["Link",a_]:>Rule["Link",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
	result = ReplaceAll[result,Rule["License",a_]:>Rule["License",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
	result = ReplaceAll[result,Rule["Funder",a_]:>Rule["Funder",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
	result = ReplaceAll[result,Rule["Assertion",a_]:>Rule["Assertion",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
	
	result = Association /@ result;
	
	If[prop=="WorksList",
		result,
		Dataset[result]
	]	
]

crossrefcookeddata["WorkTypes", id_, args_] := Module[{rawdata, invalidParameters,result},
		
		If[Length[args]>0,
		(
			Message[ServiceObject::noget,#,"WorkTypes"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		rawdata = KeyClient`rawkeydata[id,"RawTypes"];
		rawdata = formatresults[rawdata];
		result = "items" /. ("message" /. rawdata);
		
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		result = ReplaceAll[result,Rule["Id",y_]:>Rule["ID",y]];
		
		Association/@result		
]

crossrefcookeddata["WorkInformation", id_, args_] := Module[{doi,rawdata, invalidParameters,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"DOI"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"WorkInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"DOI"],
			doi = "DOI" /. args,
			(
				Message[ServiceExecute::nparam,"DOI"];			
				Throw[$Failed]
			)
		];
		
		rawdata = KeyClient`rawkeydata[id,"RawWorkInformation",{"doi"->ToString[doi]}];
		result = formatresults[rawdata];
		result = "message" /. result;
		
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		result = ReplaceAll[result,Rule["Indexed",d_]:>Rule["Indexed",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts" /. d)]];
		result = ReplaceAll[result,Rule["Issued",d_]:>Rule["Issued",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts" /. d)]];
		result = ReplaceAll[result,Rule["License",d_]:>Rule["License",Function[z,ReplaceAll[z,Rule["start",s_]:>Rule["start",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts"/.s)]]]/@d]];	
		result = ReplaceAll[result,Rule["Deposited",d_]:>Rule["Deposited",(If[DateObjectQ[DateObject[#]],DateObject[#],None])&/@("date-parts" /. d)]];
		result = ReplaceAll[result,Rule["Author",a_]:>Rule["Author",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
		result = ReplaceAll[result,Rule["Link",a_]:>Rule["Link",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
		result = ReplaceAll[result,Rule["License",a_]:>Rule["License",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
		result = ReplaceAll[result,Rule["Funder",a_]:>Rule["Funder",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
		result = ReplaceAll[result,Rule["Assertion",a_]:>Rule["Assertion",Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@a]];
		
		Association[result]		
]

crossrefcookeddata["FunderInformation", id_, args_] := Module[{rawdata, invalidParameters,funder,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"FunderID"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"FunderInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"FunderID"],
			funder ="FunderID" /. args,
			(
				Message[ServiceExecute::nparam,"FunderID"];			
				Throw[$Failed]
			)
		];
		
		rawdata = KeyClient`rawkeydata[id,"RawFunderInformation",{"funder_id"->ToString[funder]}];
		result = formatresults[rawdata];
		result = "message" /. result;
		 
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		result = ReplaceAll[result,Rule["Id",y_]:>Rule["ID",y]];
		result = ReplaceAll[result,Rule["Uri",y_]:>Rule["URI",y]];
		
		Association[result]		
]

crossrefcookeddata["OwnerPrefixInformation", id_, args_] := Module[{rawdata, invalidParameters,prefix,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"OwnerPrefix"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"OwnerPrefixInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"OwnerPrefix"],
			(
				prefix = "OwnerPrefix" /. args;
				prefix = StringReplace[prefix,"http://id.crossref.org/prefix/" ~~ p_ :> p]
			),
			(
				Message[ServiceExecute::nparam,"OwnerPrefix"];			
				Throw[$Failed]
			)
		];
		
		rawdata = KeyClient`rawkeydata[id,"RawOwnerPrefixInformation",{"owner_prefix"->ToString[prefix]}];
		result = formatresults[rawdata];
		result = "message" /. result;
		
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		Association[result]		
]

crossrefcookeddata["MemberInformation", id_, args_] := Module[{rawdata, invalidParameters,member,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"MemberID"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"MemberInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"MemberID"],
			(
				member = "MemberID" /. args;
				member = StringReplace[member,"http://id.crossref.org/member/" ~~ m_ :> m];
			),
			(
				Message[ServiceExecute::nparam,"MemberID"];			
				Throw[$Failed]
			)
		];
		
		rawdata = KeyClient`rawkeydata[id,"RawMemberInformation",{"member_id"->ToString[member]}];
		result = formatresults[rawdata];
		result = "message" /. result;
		
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		
		result = ReplaceAll[result,Rule["Counts",l_]:>Rule["Counts",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
		result = ReplaceAll[result,Rule["Breakdowns",l_]:>Rule["Breakdowns",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
		result = ReplaceAll[result,Rule["Coverage",l_]:>Rule["Coverage",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
		result = ReplaceAll[result,Rule["Prefix",l_]:>Rule["Prefix",(Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@l)]];
		result = ReplaceAll[result,Rule["Flags",l_]:>Rule["Flags",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
		result = ReplaceAll[result,Rule["Id",y_]:>Rule["ID",y]];
		result = ReplaceAll[result,Rule["LastStatusCheckTime",y_]:>Rule["LastStatusCheckTime",If[NumberQ[y],FromUnixTime[y/1000],y]]];
		
		Association[result]		
]

crossrefcookeddata["TypeInformation", id_, args_] := Module[{rawdata, invalidParameters,type,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"TypeID"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"BusinessInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"TypeID"],
			type = "TypeID" /. args,
			(
				Message[ServiceExecute::nparam,"TypeID"];			
				Throw[$Failed]
			)
		];
		
		rawdata = KeyClient`rawkeydata[id,"RawTypeInformation",{"type_id"->ToString[type]}];
		rawdata = formatresults[rawdata];
		result = "message" /. rawdata;
		
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		result = ReplaceAll[result,Rule["Id",y_]:>Rule["ID",y]];
		Association[result]		
]

crossrefcookeddata["JournalInformation", id_, args_] := Module[{rawdata, invalidParameters,issn,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"ISSN"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"JournalInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"ISSN"],
			issn = "ISSN" /. args,
			(
				Message[ServiceExecute::nparam,"ISSN"];			
				Throw[$Failed]
			)
		];
		
		rawdata = KeyClient`rawkeydata[id,"RawJournalInformation",{"issn"->ToString[issn]}];
		rawdata = formatresults[rawdata];
		result = "message" /. rawdata;
		
		result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
		result = ReplaceAll[result,Rule["LastStatusCheckTime",y_]:>Rule["LastStatusCheckTime",If[NumberQ[y],FromUnixTime[y/1000],y]]];
		Association[result]		
]

crossrefcookeddata[prop:("FunderList"|"FunderDataset"), id_, args_] := Module[{invalidParameters,params={},query,limit=20,maxPerPage=1000,startIndex,
																	argsCopy,calls,residual,progress=0,data,rawdata,totalResults,items={},result},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","Query"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];	
	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"query" -> query]			
		)
	];
	
	If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(	
				Message[ServiceExecute::nval,"MaxItems","CrossRef"];
				Throw[$Failed]
			)];						
	)];
	
	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(	
				Message[ServiceExecute::nval,"StartIndex","CrossRef"];
				Throw[$Failed]
			)];
		),
		startIndex = 0		
	];
	
	calls = Quotient[limit, maxPerPage];	
	residual = limit - (calls*maxPerPage);
	
	params = Join[params,{"rows"->ToString[maxPerPage], "offset"->ToString[startIndex]}];
	
	(* this prints the progress indicator bar *)
	PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
	If[calls > 0,
	(
		(	
			params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+#*maxPerPage]]];
			
			rawdata = KeyClient`rawkeydata[id,"RawFunders",params];
			
			If[rawdata[[1]]!=200,
			(
				Message[ServiceExecute::serror];
				Throw[$Failed]
			)];
			
			data = formatresults[rawdata];
			totalResults = "total-results"/.("message"/.data);
			items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];		
			progress = progress + 1;	
		)& /@ Range[0,calls-1];		
		
	)];
	
	If[residual > 0,
	(
		params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+calls*maxPerPage]]];
		params = ReplaceAll[params,Rule["rows",_] -> Rule["rows",ToString[residual]]];
		

		rawdata = KeyClient`rawkeydata[id,"RawFunders",params];
		If[rawdata[[1]]!=200,
		(
			Message[ServiceExecute::serror];
			Throw[$Failed]
		)];
		
		data = formatresults[rawdata];
		totalResults = "total-results"/.("message"/.data);
		items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];
	)];
	
	result = items[[1;;Min[limit,Length[items]]]];
	
	result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
	
	result = ReplaceAll[result,Rule["Id",y_]:>Rule["ID",y]];
	result = ReplaceAll[result,Rule["Uri",y_]:>Rule["URI",y]];
		
	result = Association /@ result;
	
	If[prop=="FunderList",
		result,
		Dataset[result]
	]	
]

crossrefcookeddata[prop:("MemberList"|"MemberDataset"), id_, args_] := Module[{filterParam={},invalidParameters,params={},query,limit=20,maxPerPage=1000,startIndex,
																	argsCopy,calls,residual,progress=0,data,rawdata,totalResults,items={},result,ownerPrefix},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","Query","OwnerPrefix"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"query" -> query]			
		)
	];
	
	If[KeyExistsQ[args,"OwnerPrefix"],
		(
			ownerPrefix = "OwnerPrefix" /. args;
			ownerPrefix = StringReplace[ownerPrefix,"http://id.crossref.org/prefix/" ~~ prefix_ :> prefix];
			filterParam = Append[filterParam,"prefix:"~~ToString[ownerPrefix]]			
		)
	];
	
	If[Length[filterParam]>0,
	(
		filterParam = StringJoin[Riffle[filterParam,","]];
		params = Append[params,"filter"->filterParam]
	)];
	
	If[KeyExistsQ[argsCopy,MaxItems],
	(
		limit = MaxItems /. argsCopy;
		If[!IntegerQ[limit],
		(	
			Message[ServiceExecute::nval,"MaxItems","CrossRef"];
			Throw[$Failed]
		)];						
	)];
	
	If[KeyExistsQ[args,"StartIndex"],
	(
		startIndex = "StartIndex" /. args;
		If[!IntegerQ[startIndex],
		(	
			Message[ServiceExecute::nval,"StartIndex","CrossRef"];
			Throw[$Failed]
		)];
	),
		startIndex = 0		
	];
	
	calls = Quotient[limit, maxPerPage];	
	residual = limit - (calls*maxPerPage);
	
	params = Join[params,{"rows"->ToString[maxPerPage], "offset"->ToString[startIndex]}];
	(* this prints the progress indicator bar *)
	PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
	If[calls > 0,
	(
		(	
			params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+#*maxPerPage]]];
		
			rawdata = KeyClient`rawkeydata[id,"RawMembers",params];
		
			If[rawdata[[1]]!=200,
			(
				Message[ServiceExecute::serror];
				Throw[$Failed]
			)];
			
			data = formatresults[rawdata];
			totalResults = "total-results"/.("message"/.data);
			items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];		
			progress = progress + 1;	
		)& /@ Range[0,calls-1];		
		
	)];
	
	If[residual > 0,
	(
		params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+calls*maxPerPage]]];
		params = ReplaceAll[params,Rule["rows",_] -> Rule["rows",ToString[residual]]];
	
		rawdata = KeyClient`rawkeydata[id,"RawMembers",params];
		If[rawdata[[1]]!=200,
		(
			Message[ServiceExecute::serror];
			Throw[$Failed]
		)];
		
		data = formatresults[rawdata];
		totalResults = "total-results"/.("message"/.data);
		items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];
	)];
	
	result = items[[1;;Min[limit,Length[items]]]];
	
	result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
	result = ReplaceAll[result,Rule["Counts",l_]:>Rule["Counts",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
	result = ReplaceAll[result,Rule["Breakdowns",l_]:>Rule["Breakdowns",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
	result = ReplaceAll[result,Rule["Coverage",l_]:>Rule["Coverage",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
	result = ReplaceAll[result,Rule["Prefix",l_]:>Rule["Prefix",(Association[ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]]&/@l)]];
	result = ReplaceAll[result,Rule["Flags",l_]:>Rule["Flags",Association[ReplaceAll[l,Rule[x_,y_]:>Rule[camelCase[x],y]]]]];
	result = ReplaceAll[result,Rule["Id",y_]:>Rule["ID",y]];
	result = ReplaceAll[result,Rule["LastStatusCheckTime",y_]:>Rule["LastStatusCheckTime",If[NumberQ[y],FromUnixTime[y/1000],y]]];
		
	result = Association /@ result;
	
	If[prop=="MemberList",
		result,
		Dataset[result]
	]	
]

crossrefcookeddata[prop:("LicenseList"|"LicenseDataset"), id_, args_] := Module[{invalidParameters,params={},query,limit=20,maxPerPage=1000,
																	argsCopy,data,rawdata,totalResults,items={},result},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"Query"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"query" -> query]			
		)
	];
	
	If[KeyExistsQ[argsCopy,MaxItems],
	(
		limit = MaxItems /. argsCopy;
		If[!IntegerQ[limit],
		(	
			Message[ServiceExecute::nval,"MaxItems","CrossRef"];
			Throw[$Failed]
		)];						
	),
		limit = maxPerPage;
	];
	
	params = Append[params,"rows"->ToString[limit]];
	
	rawdata = KeyClient`rawkeydata[id,"RawLicenses",params];
			
	If[rawdata[[1]]!=200,
	(
		Message[ServiceExecute::serror];
		Throw[$Failed]
	)];
			
	data = formatresults[rawdata];
	totalResults = "total-results"/.("message"/.data);
	items = If[totalResults>0,("items"/.("message"/.data))];		
	
	result = items[[1;;Min[limit,Length[items]]]];
	
	result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
	
	result = Association /@ result;
	
	If[prop=="LicenseList",
		result,
		Dataset[result]
	]	
]

crossrefcookeddata[prop:("JournalList"|"JournalDataset"), id_, args_] := Module[{invalidParameters,params={},query,limit=20,maxPerPage=1000,startIndex,
																	argsCopy,calls,residual,progress=0,data,rawdata,totalResults,items={},result},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","Query"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"query" -> query]			
		)
	];
	
	If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(	
				Message[ServiceExecute::nval,"MaxItems","CrossRef"];
				Throw[$Failed]
			)];						
	)];
	
	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(	
				Message[ServiceExecute::nval,"StartIndex","CrossRef"];
				Throw[$Failed]
			)];
		),
		startIndex = 0		
	];
	
	calls = Quotient[limit, maxPerPage];	
	residual = limit - (calls*maxPerPage);
	
	params = Join[params,{"rows"->ToString[maxPerPage], "offset"->ToString[startIndex]}];
	
	(* this prints the progress indicator bar *)
	PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
	If[calls > 0,
	(
		(	
			params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+#*maxPerPage]]];
			
			rawdata = KeyClient`rawkeydata[id,"RawJournals",params];
			
			If[rawdata[[1]]!=200,
			(
				Message[ServiceExecute::serror];
				Throw[$Failed]
			)];
			
			data = formatresults[rawdata];
			totalResults = "total-results"/.("message"/.data);
			items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];		
			progress = progress + 1;	
		)& /@ Range[0,calls-1];		
		
	)];
	
	If[residual > 0,
	(
		params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+calls*maxPerPage]]];
		params = ReplaceAll[params,Rule["rows",_] -> Rule["rows",ToString[residual]]];
		

		rawdata = KeyClient`rawkeydata[id,"RawJournals",params];
		If[rawdata[[1]]!=200,
		(
			Message[ServiceExecute::serror];
			Throw[$Failed]
		)];
		
		data = formatresults[rawdata];
		totalResults = "total-results"/.("message"/.data);
		items = Join[items, If[totalResults>0,("items"/.("message"/.data)),{}]];
	)];
	
	result = items[[1;;Min[limit,Length[items]]]];
	
	result = ReplaceAll[result,Rule[x_,y_]:>Rule[camelCase[x],y]];
	result = ReplaceAll[result,Rule["LastStatusCheckTime",y_]:>Rule["LastStatusCheckTime",If[NumberQ[y],FromUnixTime[y/1000],y]]];
	result = ReplaceAll[result,Rule["Flags",f_]:>Rule["Flags",ReplaceAll[f,Rule[x_,y_]:>Rule[camelCase[x],y]]]];
	result = ReplaceAll[result,Rule["Coverage",c_]:>Rule["Coverage",ReplaceAll[c,Rule[x_,y_]:>Rule[camelCase[x],y]]]];
	result = ReplaceAll[result,Rule["Breakdowns",f_]:>Rule["Breakdowns",ReplaceAll[f,Rule[x_,y_]:>Rule[camelCase[x],y]]]];
	result = ReplaceAll[result,Rule["Counts",c_]:>Rule["Counts",ReplaceAll[c,Rule[x_,y_]:>Rule[camelCase[x],y]]]];
	
	result = Association /@ result;
	
	If[prop=="JournalList",
		result,
		Dataset[result]
	]	
]

crossrefcookeddata[___]:=$Failed

crossrefsendmessage[___]:=$Failed

(* Utilities *)
getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.crossrefdata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]


(*formatresults[rawdata_] := ImportString[ToString[rawdata,CharacterEncoding->"UTF-8"],"JSON"]*)
formatresults[rawdata_] := ImportString[ToString[rawdata[[2]],CharacterEncoding->"UTF-8"],"JSON"]

camelCase[text_] := Module[{split, partial}, (
	(*text = ToLowerCase[text];*)
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    StringJoin[partial]
    )]
    
End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{CrossRef`Private`crossrefdata,CrossRef`Private`crossrefcookeddata,CrossRef`Private`crossrefsendmessage}
