Get["OpenLibraryFunctions.m"]
Begin["OpenLibrary`"]
Begin["`Private`"]

(******************************* OpenLibrary *************************************)

openlibrarydata[]:={
		"ServiceName" 		-> "OpenLibrary",
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
        		URLFetch[#1,{"StatusCode","Content","ContentData"},
        		Sequence@@FilterRules[{##2},Except["Parameters"|"Headers"]],
        		"Parameters"->Cases[params, Except[Rule["apikey", _]]],
        		"Headers" -> {}]]&)
        	,
        "ClientInfo"		:> {},
	 	"Gets"				-> {"BookSummary","BookInformation","BookSearch","BookText"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawBookInfo","RawCover","RawBrief","RawSearch"},
	 	"RawPosts"			-> {},
 		"Information"		-> "Wolfram Language connection to OpenLibrary API"
}

(****** Raw Properties ******)
formatresults[data_] := ToString[data[[2]],CharacterEncoding->"UTF-8"]
formatresultsjson[data_] := ImportString[ToString[data[[2]],CharacterEncoding->"UTF-8"],"JSON"]

importcover[data_] := ImportString[FromCharacterCode[data[[3]]]]

openlibrarydata["RawBookInfo"] := {
        "URL"				-> "https://openlibrary.org/api/books",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"bibkeys","format","jscmd","callback"},
        "RequiredParameters"-> {"bibkeys"},
        "ResultsFunction"	-> formatresults
    }

openlibrarydata["RawCover"] := {
        "URL"				-> (ToString@StringForm["http://covers.openlibrary.org/b/`1`/`2`-`3`.jpg",##]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"keytype","key","size"},
        "RequiredParameters"-> {"keytype","key","size"},
        "ResultsFunction"	-> importcover
    }

openlibrarydata["RawBrief"] := {
        "URL"				-> (ToString@StringForm["http://openlibrary.org/api/volumes/brief/json/`1`",#]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"bibkeys"},
        "RequiredParameters"-> {"bibkeys"},
        "ResultsFunction"	-> formatresultsjson
    }

openlibrarydata["RawSearch"] := {
        "URL"				-> "http://openlibrary.org/search.json",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"q","author","title","subject","page"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresultsjson
    }

openlibrarydata[___]:=$Failed

(****** Cooked Properties ******)

openlibrarycookeddata["BookSummary",id_,args_]:=Block[{keysParam,bibkeys,rawdata = {},data,params,invalidParameters,maxKeys=380,calls},
		invalidParameters = Select[Keys[args],!MemberQ[{"BibKeys","ShowThumbnails"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"OpenLibrary"]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[!KeyExistsQ[args,"BibKeys"],
		(
			Message[ServiceExecute::nparam,"BibKeys"];
			Throw[$Failed]
		)];
		keysParam = Lookup[args,"BibKeys"];

		If[Head[keysParam]===Dataset,
			(
				keysParam = Normal[keysParam];
			)];

		If[!MatchQ[keysParam, List[List[___], ___]],
			(
				keysParam = {keysParam};
			)];

		If[!MatchQ[keysParam, List[List[___], ___]],
		(
			Message[ServiceExecute::nval,"BibKeys","OpenLibrary"];
			Throw[$Failed]
		)];

		calls = Ceiling[Length[keysParam]/maxKeys];
		params = {"bibkeys"->""};
		params = Append[params,Rule["format","json"]];
		data = {};
		(
			params = ReplaceAll[params,Rule["bibkeys",_]:>Rule["bibkeys",StringJoin[Riffle[#[[1]] <> ":" <> #[[2]] & /@ keysParam[[1+#*maxKeys;;Min[Length[keysParam],380*(#+1)]]], ","]]]];
			rawdata = KeyClient`rawkeydata[id,"RawBookInfo",params];
			rawdata = ImportString[formatresults[rawdata],"JSON"];

			data = Join[data,rawdata[[All,2]]];

		)&/@Range[0,calls-1];

		data = If[KeyExistsQ[args,"ShowThumbnails"] && ("ShowThumbnails" /. args) == True,
			ReplaceAll[#, Rule["thumbnail_url",u_] :> Rule["thumbnail",Import[u]]] & /@ data, data];
		data = ReplaceAll[data,Rule["bib_key",x_] :> Rule["bib_key",StringSplit[ToUpperCase[x],":"]]];
		data = ReplaceAll[data,Rule[x_,y_] :> Rule[camelCase[x],y]];

		If[Length[data] == 1,
			Dataset[Association@data[[1]]],
			Dataset[Association /@ data]]
	]

openlibrarycookeddata["BookInformation",id_,args_]:=Block[{keysParam,params,rawdata={},showT,result,rawparams,invalidParameters,data,calls,maxKeys=380},
		invalidParameters = Select[Keys[args],!MemberQ[{"BibKeys","ShowThumbnails"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"OpenLibrary"]&/@invalidParameters;
			Throw[$Failed]
		)];


		If[!KeyExistsQ[args,"BibKeys"],
		(
			Message[ServiceExecute::nparam,"BibKeys"];
			Throw[$Failed]
		)];
		keysParam = Lookup[args,"BibKeys"];

		If[Head[keysParam]===Dataset,
			(
				keysParam = Normal[keysParam];
			)];

		If[!MatchQ[keysParam, List[List[___], ___]],
			(
				keysParam = {keysParam};
			)];

		If[!MatchQ[keysParam, List[List[___], ___]],
		(
			Message[ServiceExecute::nval,"BibKeys","OpenLibrary"];
			Throw[$Failed]
		)];

		calls = Ceiling[Length[keysParam]/maxKeys];
		params = {"bibkeys"->""};
		data = {};
		(
			params = ReplaceAll[params,Rule["bibkeys",_]:>Rule["bibkeys",StringJoin[Riffle[#[[1]] <> ":" <> #[[2]] & /@ keysParam[[1+#*maxKeys;;Min[Length[keysParam],380*(#+1)]]], "|"]]]];
			rawdata = KeyClient`rawkeydata[id,"RawBrief",params];
			rawdata = formatresultsjson[rawdata];
			data = Join[data,rawdata];

		)&/@Range[0,calls-1];
		showT = If[KeyExistsQ[args,"ShowThumbnails"] && ("ShowThumbnails" /. args) == True,True,False];
		result = formatBookInformation[#[[2]],showT]& /@ data;
		result = ReplaceAll[result,Rule[x_,y_] :> Rule[camelCase[x],y]];

		If[Length[result] == 1,
			Dataset[Association@result[[1]]],
			Dataset[Association/@result]]

	]

openlibrarycookeddata["BookSearch",id_,args_]:=Block[{argsCopy,author,query,title,subject,rawparams = {},rawdata,data,valid=False,invalidParameters,
														maxPerPage=100,limit,startIndex,startPage,remainder,calls},
		invalidParameters = Select[Keys[args],!MemberQ[{"Query","Author","Title","Subject","MaxItems",MaxItems,"StartIndex"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"OpenLibrary"]&/@invalidParameters;
			Throw[$Failed]
		)];

		(* Maintaing backwards compatibility with old MaxItems form *)
		argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];

		If[KeyExistsQ[argsCopy,"Query"],
			query = "Query"/.argsCopy;
			If[Head[query]===String,
			(
				rawparams = Append[rawparams,Rule["q",query]];
				valid = True;
			),
			(
				Message[ServiceExecute::nval,"Query","OpenLibrary"];
				Throw[$Failed]
			)]
		];
		If[KeyExistsQ[argsCopy,"Author"],
			author = "Author"/.argsCopy;
			If[Head[author]===String,
			(
				rawparams = Append[rawparams,Rule["author",author]];
				valid = True;
			),
			(
				Message[ServiceExecute::nval,"Author","OpenLibrary"];
				Throw[$Failed]
			)]
		];
		If[KeyExistsQ[argsCopy,"Title"],
			title = "Title"/.argsCopy;
			If[Head[title]===String,
			(
				rawparams = Append[rawparams,Rule["title",title]];
				valid = True;
			),
			(
				Message[ServiceExecute::nval,"Title","OpenLibrary"];
				Throw[$Failed]
			)]
		];
		If[KeyExistsQ[argsCopy,"Subject"],
			subject = "Subject"/.argsCopy;
			If[Head[subject]===String,
			(
				rawparams = Append[rawparams,Rule["subject",subject]];
				valid = True;
			),
			(
				Message[ServiceExecute::nval,"Subject","OpenLibrary"];
				Throw[$Failed]
			)]
		];

		If[!valid,Throw[$Failed]];(*if no paramter is received*)

		(* pagination *)
		If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","OpenLibrary"];
				Throw[$Failed]
			)];
		),
			limit = maxPerPage;
		];

		If[KeyExistsQ[argsCopy,"StartIndex"],
		(
			startIndex = "StartIndex" /. argsCopy;
			If[!IntegerQ[startIndex],
			(
				Message[ServiceExecute::nval,"StartIndex","OpenLibrary"];
				Throw[$Failed]
			)];
		),
			startIndex = 0
		];

		startPage = 1 + Quotient[startIndex,maxPerPage];
		rawparams = Append[rawparams,Rule["page",ToString[startPage]]];

		remainder = Mod[startIndex,maxPerPage];
		calls = Ceiling[(remainder + limit) / maxPerPage*1.];
		data = {};

		(
			rawdata=KeyClient`rawkeydata[id,"RawSearch",ReplaceAll[rawparams,Rule["page",_]:>Rule["page",ToString[startPage + #]]]];

			rawdata = formatresultsjson[rawdata];

			data = Join[data,("docs" /. rawdata)];
		)& /@ Range[0,calls-1];

		If[Length[data]>0,
			data = data[[Max[remainder,1];;Min[remainder+limit,Length[data]]]];
		];

		data = FilterRules[#,{"language","isbn","publish_date","author_name","oclc","lccn","edition_key","subject","title","publisher","edition_count",
								"first_publish_year","has_fulltext"}] & /@ data;

		data = ReplaceAll[data,Rule["first_publish_year",x_]:>Rule["first_publish_year",Quiet[Check[DateObject[ToString[x]], None]]]];
		data = ReplaceAll[data,Rule["publish_date",x_]:>Rule["publish_date",(Quiet[Check[DateObject[ToString[#]],None]])&/@x]];
		data = ReplaceAll[data,Rule["publish_date",x_]:>Rule["publish_date",DeleteCases[x,None]]];
		data = ReplaceAll[data,Rule["language",x_]:>Rule["language",mapLanguage/@x]];
		data = ReplaceAll[data,Rule["edition_key",x_]:>Rule["edition_key",{"OLID",#}&/@x]];
		data = ReplaceAll[data,Rule["isbn",x_]:>Rule["isbn",{"ISBN",#}&/@x]];
		data = ReplaceAll[data,Rule["lccn",x_]:>Rule["lccn",{"LCCN",#}&/@x]];
		data = ReplaceAll[data,Rule["oclc",x_]:>Rule["oclc",{"OCLC",#}&/@x]];

		data = SortBy[#, (#[[1]] /. Thread[{"title","author_name","first_publish_year","edition_count","edition_key","publish_date",
									"publisher","language","subject","isbn","lccn","oclc","has_fulltext"} -> Range[13]]) &] & /@ data ;

		data = ReplaceAll[data,Rule[x_,y_] :> Rule[camelCase[x],y]];
		Dataset[Association/@data]
	]

openlibrarycookeddata["BookText",id_,args_]:=Block[{keysParam,params,rawdata,showT,result,invalidParameters},
		invalidParameters = Select[Keys[args],!MemberQ[{"BibKeys"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"OpenLibrary"]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[!KeyExistsQ[args,"BibKeys"],
		(
			Message[ServiceExecute::nparam,"BibKeys"];
			Throw[$Failed]
		)];

		keysParam = "BibKeys" /. args;
		params = args;
		params = ReplaceAll[params,Rule["BibKeys",x_]:>Rule["bibkeys",x]];

		If[Head[keysParam]===Dataset,
			(
				params = DeleteCases[args, Rule["bibkeys", _]];
				params = Append[params, Rule["bibkeys",Normal[keysParam]]];
				keysParam = Normal[keysParam];
			)];

		If[!MatchQ[keysParam, List[List[___], ___]],
			(
				params = DeleteCases[args, Rule["bibkeys", _]];
				keysParam = {keysParam};
				params = Append[params, Rule["bibkeys",keysParam]];
			)];

		If[!MatchQ[keysParam, List[List[___], ___]],
		(
			Message[ServiceExecute::nval,"BibKeys","OpenLibrary"];
			Throw[$Failed]
		)];

		params = ReplaceAll[params, Rule["bibkeys",bk_]:>Rule["bibkeys",StringJoin[Riffle[#[[1]] <> ":" <> #[[2]] & /@ bk, "|"]]]];
		rawdata=KeyClient`rawkeydata[id,"RawBrief",params];
		rawdata = formatresultsjson[rawdata];


		(*result = getFullText /@ rawdata;
		result = Transpose[{keysParam,result}];
		result = #[[1]]->#[[2]]&/@result;*)

		result = Rule[#[[1]],getFullText[#]]& /@ rawdata;

		Dataset[Association[result]]
		(*If[Length[result] == 1,
			Dataset[Association@result[[1]]],
			Dataset[Association/@result]]
			*)

	]


openlibrarycookeddata[___]:=$Failed

(* Utilities *)
makehyperlinks[rules_]:= (#[[1]] -> If[Head[#[[2]]] === String && StringMatchQ[#[[2]],"http" ~~ ___],Hyperlink[#[[2]]],#[[2]]]&) /@ rules

formatBookInformation[record_,showT_] := Module[{data,details,language,bibkey,covers,thumbnail,detailsData,authors},
(
	data = ("data" /. ("records" /. record)[[1,2]]);

	details = ("details" /. ("records" /. record)[[1,2]]);
	detailsData = FilterRules["details"/.details,{"publish_country","publish_places"}];
	language = "languages"/.("details"/.details);

	bibkey = "bib_key" /. details;
	bibkey = Rule["bib_key",StringSplit[ToUpperCase[bibkey],":"]];
	data = FilterRules[data,{"publishers","number_of_pages","publish_date","identifiers","title","cover","url",
								"authors","subjects","notes","identifiers","ebooks"}];
	covers = "cover" /. data;
	If[KeyExistsQ[data,"ebooks"],
	(
		data = ReplaceAll[data,Rule["ebooks",x_]:>Rule["ebooks",formatEbooksInformation[x[[1]]]]]
	)];
	data = ReplaceAll[data,Rule["publish_date",x_]:>Rule["publish_date",Quiet[Check[DateObject[ToString[x]], None]]]];
	(*data = ReplaceAll[data,Rule["cover",x_]:>Rule["cover",(#[[1]]->Hyperlink[#[[2]]])&/@x]];*)
	data = ReplaceAll[data,Rule["cover",x_]:>Rule["cover",(#[[1]]->#[[2]])&/@x]];
	(*data = ReplaceAll[data,Rule["cover",x_]:>Rule["cover",Import["small"/.x]]];*)
	(*data = ReplaceAll[data,Rule["authors",x_]:>Rule["authors",Hyperlink["name"/.#,"url"/.#]&/@x]];*)
	authors = Association/@("authors"/.("details"/.details));
	data = If[MatchQ["authors"/.data,"authors"],
					Append[data,Rule["authors",formatAuthors[authors]]],
					ReplaceAll[data,Rule["authors",x_]:>Rule["authors",{"name"/.#,"url"/.#}&/@x]]];
	(*data = ReplaceAll[data,Rule["url",x_]:>Rule["url",Hyperlink[x]]];*)
	data = ReplaceAll[data,Rule["publishers",x_]:>Rule["publishers","name" /. # & /@ x]];
	data = Join[data,detailsData];
	data = ReplaceAll[data,Rule["publish_country",x_]:>Rule["publish_country",mapPublishCountryCode[x]]];

	If[language =!= "languages",
		(
			language = Rule["languages",mapLanguage[Last@StringSplit[("key" /. #), "/"]] & /@ language];
			data = Join[data,{bibkey,language}];
		),
		data = Join[data,{bibkey}];
	];

	If[showT && KeyExistsQ[data, "cover"],
	(
		thumbnail = Import["small" /. covers];
		data = Append[data,Rule["thumbnail",thumbnail]];
	)];

	data = SortBy[data, (#[[1]] /. Thread[{"bib_key","title","authors","languages","cover","url","publish_date","publishers","number_of_pages","subjects","notes","identifiers"} -> Range[12]]) &];

	data
)]

formatAuthors[authors_List]:= Normal[{#["name"],URLBuild[{"http://openlibrary.org", #["name"], StringReplace[#["key"], " " -> "_"]}]}&/@authors]

formatEbooksInformation[ebooks_] := Module[{result = {},formats},
	(
		If[KeyExistsQ[ebooks, "preview_url"],
			(*result = Append[result,Rule["PreviewLink", Hyperlink[("preview_url" /. ebooks)]]];*)
			result = Append[result,Rule["PreviewLink", ("preview_url" /. ebooks)]];
		];
		If[KeyExistsQ[ebooks, "read_url"],
			(*result = Append[result,Rule["ReadLink", Hyperlink[("read_url" /. ebooks)]]];*)
			result = Append[result,Rule["ReadLink", ("read_url" /. ebooks)]];
		];
		If[KeyExistsQ[ebooks, "formats"],
		(
			formats = "formats" /. ebooks;
			If[KeyExistsQ[formats, "pdf"],
				(*result = Append[result,"PDFLink" -> Hyperlink["url" /. ("pdf" /. formats)]];*)
				result = Append[result,"PDFLink" -> "url" /. ("pdf" /. formats)];
			];
			If[KeyExistsQ[formats, "djvu"],
				(*result = Append[result,"DJVULink" -> Hyperlink["url" /. ("djvu" /. formats)]];*)
				result = Append[result,"DJVULink" -> "url" /. ("djvu" /. formats)];
			];
			If[KeyExistsQ[formats, "epub"],
				(*result = Append[result,"EPUBLink" -> Hyperlink["url" /. ("epub" /. formats)]];*)
				result = Append[result,"EPUBLink" -> "url" /. ("epub" /. formats)];
			];
			If[KeyExistsQ[formats, "text"],
				(*result = Append[result,"TextLink" -> Hyperlink["url" /. ("text" /. formats)]];*)
				result = Append[result,"TextLink" -> "url" /. ("text" /. formats)];
			];
		)];
		Association@@result
	)]

camelCase[text_] := Module[{split, partial}, (
	(*text = ToLowerCase[text];*)
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    partial = StringJoin[partial];
    partial = StringReplace[partial,RegularExpression["[Uu][Rr][Ll]"]->"URL"];
    partial

    )]

getFullText[info_] := Module[{data, formats, url},
  (
   data = "data" /. (Lookup[info[[2]], "records"][[1, 2]]);
   If[KeyExistsQ[data, "ebooks"],
    (
     formats = "formats" /. ("ebooks" /. data);
     formats = formats[[1]];
     If[KeyExistsQ[formats, "text"],
      (
       url = "url" /. ("text" /. formats);
       Import[url]
       ),
      Missing["NotAvailable"]]
     ),
    Missing["NotAvailable"]]
   )]

(****** Send Message ******)
openlibrarysendmessage[args_]:=$Failed
End[] (* End Private Context *)
End[]

{OpenLibrary`Private`openlibrarydata,OpenLibrary`Private`openlibrarycookeddata,OpenLibrary`Private`openlibrarysendmessage}
