Begin["PubMed`"]

Begin["`Private`"]

(******************************* PubMed *************************************)

(* Authentication information *)

pubmeddata[]:={
		"ServiceName" 		-> "PubMed",
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
        		URLFetch[#1 <> "?" <> StringJoin[Riffle[#[[1]]<>"="<>#[[2]]&/@params,"&"]],{"StatusCode","Content"},
        		Sequence@@FilterRules[{##2},Except["Parameters"|"Headers"]]]]&)
        	,
        "ClientInfo"		:> {},
	 	"Gets"				-> {"PublicationSearch","PublicationTypes","PublicationAbstract"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawInformation","RawSearch","RawSummary","RawFetch","RawLink","RawGeneralQuery","RawSpell","RawCitationMatch"},
	 	"RawPosts"			-> {},
 		"Information"		-> "Import PubMed API data to the Wolfram Language"
}

eutilsURL = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

(**** Raw Requests ****)

pubmeddata["RawInformation"] := {
        "URL"				-> eutilsURL <> "einfo.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","retmode"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

(*
    Provides a list of UIDs matching a text query
    Posts the results of a search on the History server
    Downloads all UIDs from a dataset stored on the History server
    Combines or limits UID datasets stored on the History server
    Sorts sets of UIDs
*)
pubmeddata["RawSearch"] := {
		"URL"				-> eutilsURL <> "esearch.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","term","usehistory","WebEnv","query_key","retstart","retmax","rettype",
        						"retmode","sort","field","datetype","reldate","mindate","maxdate"},
        "RequiredParameters"-> {"db","term"},
        "ResultsFunction"	-> formatresults
    }

(*
    Returns document summaries (DocSums) for a list of input UIDs
    Returns DocSums for a set of UIDs stored on the Entrez History server
*)
pubmeddata["RawSummary"] := {
        "URL"				-> eutilsURL <> "esummary.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","id","query_key","WebEnv","retstart","retmax","retmode"},
        "RequiredParameters"-> {"db"},
        "ResultsFunction"	-> formatresults
    }

(*
    Returns formatted data records for a list of input UIDs
    Returns formatted data records for a set of UIDs stored on the Entrez History server
*)
pubmeddata["RawFetch"] := {
        "URL"				-> eutilsURL <> "efetch.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","id","query_key","WebEnv","rettype","retstart","retmax","retmode",
        						"strand","seq_start","seq_stop","complexity"},
        "RequiredParameters"-> {"db"},
        "ResultsFunction"	-> formatresults
    }

(*
    Returns UIDs linked to an input set of UIDs in either the same or a different Entrez database
    Returns UIDs linked to other UIDs in the same Entrez database that match an Entrez query
    Checks for the existence of Entrez links for a set of UIDs within the same database
    Lists the available links for a UID
    Lists LinkOut URLs and attributes for a set of UIDs
    Lists hyperlinks to primary LinkOut providers for a set of UIDs
    Creates hyperlinks to the primary LinkOut provider for a single UID
*)
pubmeddata["RawLink"] := {
        "URL"				-> eutilsURL <> "elink.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","dbfrom","cmd","id","query_key","WebEnv","linkname","term","holding",
        						"datetype","reldate","mindate","maxdate"},
        "RequiredParameters"-> {"db","dbfrom"},
        "ResultsFunction"	-> formatresults
    }

(*
	Provides the number of records retrieved in all Entrez databases by a single text query.
*)
pubmeddata["RawGeneralQuery"] := {
        "URL"				-> eutilsURL <> "egquery.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"term"},
        "RequiredParameters"-> {"term"},
        "ResultsFunction"	-> formatresults
    }

(*
	Provides spelling suggestions for terms within a single text query in a given database.
*)
pubmeddata["RawSpell"] := {
        "URL"				-> eutilsURL <> "espell.fcgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","term"},
        "RequiredParameters"-> {"db","term"},
        "ResultsFunction"	-> formatresults
    }
(*
	Retrieves PubMed IDs (PMIDs) that correspond to a set of input citation strings.
*)
pubmeddata["RawCitationMatch"] := {
        "URL"				-> eutilsURL <> "ecitmatch.cgi",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"db","bdata","retmode"},
        "RequiredParameters"-> {"db","bdata","retmode"},
        "ResultsFunction"	-> formatresults
    }

pubmeddata[___]:=$Failed

(**** Cooked Requests ****)

pubmedcookeddata["PublicationTypes", ___] := publicationTypes

pubmedcookeddata["PublicationSearch", id_, args_] := Module[{invalidParameters,params,sort,query,limit=20,maxPerPage=10000,startIndex,
												calls,residual,progress=0,data,rawdata,totalResults,items={},sortVal,term={},
												element,argsCopy,result,ids,author,publisher,title,dateParam,startDate,endDate,isbn,type,language},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","SortBy","Query","Author",
									"Publisher","Title","CreationDate","PublicationDate","ModificationDate",
									"EntrezDate","CompletionDate","ISBN","Language","PublicationType","ID","Elements"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"PublicationSearch"]&/@invalidParameters;
			Throw[$Failed]
		)];
	argsCopy = ReplaceAll[args,Rule["MaxItems",x_]:>Rule[MaxItems,x]];

	params = {"db"->"pubmed","retmode"->"json"};

	If[KeyExistsQ[args,"ID"],
	(
		ids = "ID" /. args;
		If[MatchQ[ids,_String], ids = {ids}];
		If[!MatchQ[ids,List[___String]],
		(
			Message[ServiceExecute::nval,"ID","PubMed"];
			Throw[$Failed]
		)];
	),
	(
		If[KeyExistsQ[args,"SortBy"],
		(
			sort = "SortBy" /. args;
			Switch[sort,
				"MostRecent",
				sortVal = "",
				"Relevance",
				sortVal = "relevance",
				"PublicationDate",
				sortVal = "pub+date",
				"Author",
				sortVal = "first+author",
				"Journal",
				sortVal = "journal",
				"Title",
				sortVal = "title",
				_,
				(
					Message[ServiceExecute::nval,"SortBy","PubMed"];
					Throw[$Failed]
				)
			];
			If[sortVal != "",
				params = Append[params,Rule["sort",sortVal]];
			]
		)];

		If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			term = Append[term,query];
		)];

		If[KeyExistsQ[args,"Author"],
		(
			author = "Author" /. args;
			If[MatchQ[author,_String],
				author = "(" <> author <> "[Author])",
				author = booleanParser[author,"Author"];
			];
			term = Append[term,author];
		)];

		If[KeyExistsQ[args,"Publisher"],
		(
			publisher = "Publisher" /. args;
			If[MatchQ[publisher,_String],
				publisher = "(" <> publisher <> "[Publisher])",
				publisher = booleanParser[publisher,"Publisher"];
			];
			term = Append[term,publisher];
		)];

		If[KeyExistsQ[args,"Title"],
		(
			title = "Title" /. args;
			If[MatchQ[title,_String],
				title = "(" <> title <> "[Title])",
				title = booleanParser[title,"Title"];
			];
			term = Append[term,title];
		)];

		If[KeyExistsQ[args,"ISBN"],
		(
			isbn = "ISBN" /. args;
			If[MatchQ[isbn,_String],
				isbn = "(" <> isbn <> "[ISBN])",
				isbn = booleanParser[isbn,"ISBN"],
				(
					Message[ServiceExecute::nval,"ISBN","PubMed"];
					Throw[$Failed]
				)
			];

			term = Append[term,isbn];
		)];

		If[KeyExistsQ[args,"PublicationType"],
		(
			type = "PublicationType" /. args;
			If[MatchQ[type,_String],
				type = "(" <> type <> "[Publication Type])",
				type = booleanParser[type,"Publication Type"],
				(
					Message[ServiceExecute::nval,"PublicationType","PubMed"];
					Throw[$Failed]
				)
			];

			term = Append[term,type];
		)];

		If[KeyExistsQ[args,"Language"],
		(
			language = "Language" /. args;
			language = language /. (x_ /; MatchQ[x, Entity["Language", _]] :> EntityValue[x, "Name"]);
			If[Count[language, x_ /; !MatchQ[x, _String| _Except | _Alternatives]] > 0,
			(
				Message[ServiceExecute::nval,"Language","PubMed"];
				Throw[$Failed]
			)];
			If[MatchQ[language,_String],
				language = "(" <> language <> "[Language])",
				language = booleanParser[language,"Language"]
			];
			term = Append[term,language];
		)];

		If[KeyExistsQ[args,"CreationDate"],
		(
			dateParam = "CreationDate" /. args;
			{startDate, endDate} = parseDates[dateParam];
			If[MatchQ[{startDate, endDate},{_DateObject,_DateObject}],
			(
				term = Append[term,"(\"" <> formatDate[startDate] <> "\"[CRDT] : \"" <> formatDate[endDate] <> "\"[CRDT])"]
			),
			(
				Message[ServiceExecute::nval,"CreationDate","PubMed"];
				Throw[$Failed]
			)]
		)];

		If[KeyExistsQ[args,"CompletionDate"],
		(
			dateParam = "CompletionDate" /. args;
			{startDate, endDate} = parseDates[dateParam];
			If[MatchQ[{startDate, endDate},{_DateObject,_DateObject}],
			(
				term = Append[term,"(\"" <> formatDate[startDate] <> "\"[DCOM] : \"" <> formatDate[endDate] <> "\"[DCOM])"]
			),
			(
				Message[ServiceExecute::nval,"CompletionDate","PubMed"];
				Throw[$Failed]
			)]

		)];

		If[KeyExistsQ[args,"EntrezDate"],
		(
			dateParam = "EntrezDate" /. args;
			{startDate, endDate} = parseDates[dateParam];
			If[MatchQ[{startDate, endDate},{_DateObject,_DateObject}],
			(
				term = Append[term,"(\"" <> formatDate[startDate] <> "\"[EDAT] : \"" <> formatDate[endDate] <> "\"[EDAT])"]
			),
			(
				Message[ServiceExecute::nval,"EntrezDate","PubMed"];
				Throw[$Failed]
			)]

		)];

		If[KeyExistsQ[args,"ModificationDate"],
		(
			dateParam = "ModificationDate" /. args;
			{startDate, endDate} = parseDates[dateParam];
			If[MatchQ[{startDate, endDate},{_DateObject,_DateObject}],
			(
				term = Append[term,"(\"" <> formatDate[startDate] <> "\"[LR] : \"" <> formatDate[endDate] <> "\"[LR])"]
			),
			(
				Message[ServiceExecute::nval,"ModificationDate","PubMed"];
				Throw[$Failed]
			)]

		)];

		If[KeyExistsQ[args,"PublicationDate"],
		(
			dateParam = "PublicationDate" /. args;
			{startDate, endDate} = parseDates[dateParam];
			If[MatchQ[{startDate, endDate},{_DateObject,_DateObject}],
			(
				term = Append[term,"(\"" <> formatDate[startDate] <> "\"[DP] : \"" <> formatDate[endDate] <> "\"[DP])"]
			),
			(
				Message[ServiceExecute::nval,"PublicationDate","PubMed"];
				Throw[$Failed]
			)]

		)];

		If[Length[term] == 0,
			Throw[$Failed],
		(
			term = StringJoin[Riffle[term," AND "]];
			term = StringReplace[term, Alternatives["AND NOT", "OR NOT"] :> "NOT"];
			(*Print[term];*)

			params = Append[params,"term" -> URLEncode[term]]
		)];

		If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","PubMed"];
				Throw[$Failed]
			)];
		)];

		If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(
				Message[ServiceExecute::nval,"StartIndex","PubMed"];
				Throw[$Failed]
			)];
		),
			startIndex = 0
		];

		calls = Quotient[limit, maxPerPage];
		residual = limit - (calls*maxPerPage);

		params = Join[params,{"retmax"->ToString[maxPerPage], "retstart"->ToString[startIndex]}];

		(* this prints the progress indicator bar *)
		PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];

		If[calls > 0,
		(
			(
				params = ReplaceAll[params,Rule["retstart",_] -> Rule["retstart",ToString[startIndex+#*maxPerPage]]];

				rawdata = KeyClient`rawkeydata[id,"RawSearch",params];
				data = formatresultsJSON[rawdata];

				If[rawdata[[1]]!=200,
				(
					Message[ServiceExecute::serrormsg,"message"/.data];
					Throw[$Failed]
				)];

				totalResults = FromDigits["retmax"/.("esearchresult"/.data)];
				items = Join[items, If[totalResults>0,("idlist"/.("esearchresult"/.data)),{}]];
				progress = progress + 1;
			)& /@ Range[0,calls-1];

		)];

		If[residual > 0,
		(
			params = ReplaceAll[params,Rule["retstart",_] -> Rule["retstart",ToString[startIndex+calls*maxPerPage]]];
			params = ReplaceAll[params,Rule["retmax",_] -> Rule["retmax",ToString[residual]]];

			rawdata = KeyClient`rawkeydata[id,"RawSearch",params];

			data = formatresultsJSON[rawdata];
			If[rawdata[[1]]!=200,
			(
				Message[ServiceExecute::serrormsg,"message"/.data];
				Throw[$Failed]
			)];

			totalResults = FromDigits["retmax"/.("esearchresult"/.data)];
			items = Join[items, If[totalResults>0,("idlist"/.("esearchresult"/.data)),{}]];

		)];

		ids = items[[1;;Min[limit,Length[items]]]];

	)];

	params = {"db"->"pubmed","retmode"->"json"};
	params = Append[params,"id"->StringJoin[Riffle[ids,","]]];
	result = getPublicationSummary[id,params] //Dataset;

	element = If[KeyExistsQ[args,"Elements"],"Elements" /. args,{"Data"}];
	If[MatchQ[Head[element],String],element = {element}];


	result = Rule[#, Switch[#,
		"Data",
		(
			result[All,{"UID","ISSN","PubDate","Title","Source"}]
		),
		"FullData",
		result,
		_,
		(
			Message[ServiceExecute::nval,"Elements","PubMed"];
			Throw[$Failed]
		)
		]]& /@ element;

	If[Length[result] == 1, result[[1,2]],result]

]

(*pubmedcookeddata["PublicationSummary", id_, args_] := Module[{invalidParameters,params,ids},
	invalidParameters = Select[Keys[args],!MemberQ[{"ID"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"PublicationSummary"]&/@invalidParameters;
			Throw[$Failed]
		)];

	params = {"db"->"pubmed","retmode"->"json"};

	If[KeyExistsQ[args,"ID"],
		ids = "ID" /. args,
		(
			Message[ServiceExecute::nparam,"ID"];
			Throw[$Failed]
		)];

	Switch[ids,
		_String,
			params = Append[params,"id"->ids],
		List[___String],
			params = Append[params,"id"->StringJoin[Riffle[ids,","]]],
		_,
		(
			Message[ServiceExecute::nval,"ID","PubMed"];
			Throw[$Failed]
		)];

	getPublicationSummary[id,params]
]*)

pubmedcookeddata["PublicationAbstract", id_, args_] := Module[{result,invalidParameters,params,ids,data,rawdata},
	invalidParameters = Select[Keys[args],!MemberQ[{"ID"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"PublicationInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];

	params = {"db"->"pubmed","retmode"->"text","rettype"->"abstract"};

	If[KeyExistsQ[args,"ID"],
		ids = "ID" /. args,
		(
			Message[ServiceExecute::nparam,"ID"];
			Throw[$Failed]
		)];

	Switch[ids,
		_String,
			params = Append[params,"id"->ids],
		List[___String],
			params = Append[params,"id"->StringJoin[Riffle[ids,","]]],
		_,
		(
			Message[ServiceExecute::nval,"ID","PubMed"];
			Throw[$Failed]
		)];

	rawdata = KeyClient`rawkeydata[id,"RawFetch",params];
	data = formatresults[rawdata];

	If[rawdata[[1]]!=200,
	(
		Message[ServiceExecute::serrormsg,data];
		Throw[$Failed]
	)];

	result = StringSplit[data, RegularExpression["\n[0-9]+[.][ ]"]];
	result = StringTrim/@result;

	If[Length[result] == 0,
		"",
		If[Length[result] == 1,
			result[[1]],
			result
		]
	]
]

pubmedcookeddata[___]:=$Failed

pubmedsendmessage[___]:=$Failed

(* Utilities *)
getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.pubmeddata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

formatresults[rawdata_] := If[MatchQ[rawdata,{_,_}],ToString[rawdata[[2]],CharacterEncoding->"UTF-8"],rawdata]

formatresultsJSON[rawdata_] := ImportString[ToString[rawdata[[2]],CharacterEncoding->"UTF-8"],"JSON"]

camelCase[text_] := Module[{split, partial}, (
	split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    partial = StringJoin[partial];
    partial = StringReplace[partial,RegularExpression["[Uu][Rr][Ll]"]->"URL"];
    partial = StringReplace[partial,RegularExpression["^[Id][Dd]$"]->"ID"];
    partial
    )]

booleanParser[e_, field_] := Block[{result},
  result = e //. {Verbatim[Alternatives][x_] :> x ~~ "[" ~~ field ~~ "]",
     			Verbatim[Alternatives][x__, y_] :> "(" ~~ Alternatives[x] ~~ " OR " ~~ y ~~ "[" ~~ field ~~ "]" ~~ ")",
     			Verbatim[Except][x_] :> "NOT " ~~ x ~~ "[" ~~ field ~~ "]",
     			List[x_] :> x ~~ "[" ~~ field ~~ "]",
     			List[x__, y_] :> "(" ~~ List[x] ~~ " AND " ~~ y ~~ "[" ~~ field ~~ "]" ~~ ")"};
  result = StringReplace[result, Alternatives["AND NOT", "OR NOT"] :> "NOT"];
  result = StringReplace[result, "[" ~~ field ~~ "][" ~~ field ~~ "]" :> "[" ~~ field ~~ "]"];
  result
  ]

parseDates[param_] := Block[{startDate, endDate},
	(
   		Switch[param,
    		_String,
    		(
     			startDate = Quiet[Check[DateObject[param], ""]];
     			endDate = DateObject["3000"];
		    ),
    		_DateObject,
		    (
     			startDate = param;
     			endDate = DateObject["3000"];
     		),
    		List[_, _],
    		(
     			startDate = param[[1]];
     			endDate = param[[2]];
     			Switch[Head[startDate],
      				String, startDate = Quiet[Check[DateObject[startDate], ""]],
      				DateObject, startDate = startDate
			    ];
     			Switch[Head[endDate],
      				String, endDate = Quiet[Check[DateObject[endDate], ""]],
      				DateObject, endDate = endDate];
		    ),
    		Interval[{_DateObject, _DateObject}],
		    (
     			startDate = param /. Interval[{f_, t_}] :> f;
     			endDate = param /. Interval[{f_, t_}] :> t;
     		),
		    _,
		    (
     			startDate = "";
     			endDate = "";
		    )
		];
   		{startDate, endDate}
   	)]

formatDate[date_] := DateString[date, "ISODate"]

publicationTypes = {"Addresses", "Autobiography", "Bibliography", "Biography", "Case \
Reports", "Classical Article", "Clinical Conference", "Clinical \
Trial", "Clinical Trial, Phase I", "Clinical Trial, Phase II", \
"Clinical Trial, Phase III", "Clinical Trial, Phase IV", "Collected \
Works", "Comment", "Comparative Study", "Congresses", "Consensus \
Development Conference", "Consensus Development Conference, NIH", \
"Controlled Clinical Trial", "Corrected and Republished Article", \
"Dataset", "Dictionary", "Directory", "Duplicate Publication", \
"Editorial", "English Abstract", "Evaluation Studies", "Festschrift", \
"Government Publications", "Guideline", "Historical Article", \
"Interactive Tutorial", "Interview", "Introductory Journal Article", \
"Journal Article", "Lectures", "Legal Cases", "Legislation", \
"Letter", "Meta-Analysis", "Multicenter Study", "News", "Newspaper \
Article", "Observational Study", "Overall", "Patient Education \
Handout", "Periodical Index", "Personal Narratives", "Portraits", \
"Practice Guideline", "Pragmatic Clinical Trial", "Publication \
Components", "Publication Formats", "Publication Type Category", \
"Published Erratum", "Randomized Controlled Trial", "Research \
Support, American Recovery and Reinvestment Act", "Research Support, \
N.I.H., Extramural", "Research Support, N.I.H., Intramural", \
"Research Support, Non-U.S. Gov't Research Support, U.S. Gov't, \
Non-P.H.S.", "Research Support, U.S. Gov't, P.H.S.", "Retracted \
Publication", "Retraction of Publication", "Review", "Scientific \
Integrity Review", "Study Characteristics", "Support of Research", \
"Technical Report", "Twin Study", "Validation Studies", "Video-Audio \
Media", "Webcasts"};

getPublicationSummary[connectionID_,params_] := Block[{rawdata,data,results,uids,items,doi},
(

	rawdata = KeyClient`rawkeydata[connectionID,"RawSummary",params];
	data = formatresultsJSON[rawdata];
	If[rawdata[[1]]!=200,
	(
		(*Message[ServiceExecute::serrormsg,"message"/.data];*)
		Throw[$Failed]
	)];

	results = "result"/.data;
	If[MatchQ[results,"result"],
		Association[]
	,(
	uids = "uids" /. results;
	items = Values[FilterRules[results, uids]];
	items = ReplaceAll[items,Rule[x_,y_]:>Rule[camelCase[x],y]];

	items = ReplaceAll[items,Rule["Authors",a_]:>Rule["Authors",ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]&/@a]];
	items = ReplaceAll[items,Rule["Authors",a_]:>Rule["Authors",Association[ReplaceAll[#,{Rule["Authtype",s_]:>Rule["AuthorType",s],
																			Rule["Clusterid",s_]:>Rule["ClusterID",s]}]]&/@a]];
	items = ReplaceAll[items,Rule["History",a_]:>Rule["History",ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]&/@a]];
	items = ReplaceAll[items,Rule["History",a_]:>Rule["History",Association[ReplaceAll[#,{Rule["Pubstatus",s_]:>Rule["PubStatus",s],
																			Rule["Date",s_]:>Rule["Date",Quiet[Check[DateObject[s], s]]]}]]&/@a]];
	items = ReplaceAll[items,Rule["Articleids",a_]:>Rule["ArticleIDs",ReplaceAll[#,Rule[x_,y_]:>Rule[camelCase[x],y]]&/@a]];
	items = ReplaceAll[items,Rule["ArticleIDs",a_]:>Rule["ArticleIDs",Association[ReplaceAll[#,{Rule["Idtype",s_]:>Rule["IDType",s],
																			Rule["Idtypen",s_]:>Rule["IDTypeN",s]}]]&/@a]];
	items = (
		If[KeyExistsQ[#, "ArticleIDs"],
 		(
 			doi = Select["ArticleIDs" /. #, Function[x,MemberQ[Normal[x], Rule["IDType", "doi"]]]];
 			If[Length[doi] > 0,
  				Append[#, Rule["DOILink", "https://doi.org/" <> doi[[1]]["Value"]]],
  				#
  			]
		),
			#] &)/@ items;

	items = ReplaceAll[items,Rule["Issn",y_]:>Rule["ISSN",y]];
	items = ReplaceAll[items,Rule["AvailablefromURL",y_]:>Rule["AvailableFromURL",y]];
	items = ReplaceAll[items,Rule["Uid",y_]:>Rule["UID",y]];
	items = ReplaceAll[items,Rule["Nlmuniqueid",y_]:>Rule["NLMUniqueID",y]];
	items = ReplaceAll[items,Rule["Lastauthor",y_]:>Rule["LastAuthor",y]];
	items = ReplaceAll[items,Rule["Publishername",y_]:>Rule["PublisherName",y]];
	items = ReplaceAll[items,Rule["Bookname",y_]:>Rule["BookName",y]];
	items = ReplaceAll[items,Rule["Booktitle",y_]:>Rule["BookTitle",y]];
	items = ReplaceAll[items,Rule["Essn",y_]:>Rule["ESSN",y]];
	items = ReplaceAll[items,Rule["Sorttitle",y_]:>Rule["SortTitle",y]];
	items = ReplaceAll[items,Rule["Pubtype",y_]:>Rule["PubType",y]];
	items = ReplaceAll[items,Rule["Lang",y_]:>Rule["Language",y]];
	items = ReplaceAll[items,Rule["Recordstatus",y_]:>Rule["RecordStatus",y]];
	items = ReplaceAll[items,Rule["Locationlabel",y_]:>Rule["LocationLabel",y]];
	items = ReplaceAll[items,Rule["Pmcrefcount",y_]:>Rule["PMCRefCount",y]];
	items = ReplaceAll[items,Rule["Viewcount",y_]:>Rule["ViewCount",y]];
	items = ReplaceAll[items,Rule["Fulljournalname",y_]:>Rule["FullJournalName",y]];
	items = ReplaceAll[items,Rule["Elocationid",y_]:>Rule["ELocationID",y]];
	items = ReplaceAll[items,Rule["Doctype",y_]:>Rule["DocType",y]];
	items = ReplaceAll[items,Rule["Srccontriblist",y_]:>Rule["SrcContribList",y]];
	items = ReplaceAll[items,Rule["Vernaculartitle",y_]:>Rule["VernacularTitle",y]];
	items = ReplaceAll[items,Rule["Publisherlocation",y_]:>Rule["PublisherLocation",y]];
	items = ReplaceAll[items,Rule["Doccontriblist",y_]:>Rule["DocContribList",y]];
	items = ReplaceAll[items,Rule["Sortfirstauthor",y_]:>Rule["SortFirstAuthor",y]];
	items = ReplaceAll[items,Rule["Reportnumber",y_]:>Rule["ReportNumber",y]];
	items = ReplaceAll[items,Rule["Pubstatus",y_]:>Rule["PubStatus",y]];

	(* parse dates *)
	items = ReplaceAll[items,Rule["Epubdate",y_]:>Rule["EPubDate",Quiet[Check[DateObject[y], y]]]];
	items = ReplaceAll[items,Rule["Docdate",y_]:>Rule["DocDate",Quiet[Check[DateObject[y], y]]]];
	items = ReplaceAll[items,Rule["Pubdate",y_]:>Rule["PubDate",Quiet[Check[DateObject[y], y]]]];
	items = ReplaceAll[items,Rule["Srcdate",y_]:>Rule["SrcDate",Quiet[Check[DateObject[y], y]]]];
	items = ReplaceAll[items,Rule["Sortpubdate",y_]:>Rule["SortPubDate",Quiet[Check[DateObject[y], y]]]];

	Association /@ items)
	]
)]

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{PubMed`Private`pubmeddata,PubMed`Private`pubmedcookeddata,PubMed`Private`pubmedsendmessage}
