Get["WikipediaFunctions.m"]

Begin["Wikipedia`"]

WikipediaData::excpar = "The following parameter(s) cannot be used at the same time: `1`.";
WikipediaData::onepar = "One of the following parameter(s) is required: `1`.";
WikipediaData::problm = "Wikipedia servers are currently under maintenance or experiencing a technical problem. Please try again in a few minutes.";
WikipediaData::date = "The service returned the following error message: `1`";

Begin["`Private`"]

(******************************* Wikipedia *************************************)

(* Authentication information *)

wikipediadata[] = {
		"ServiceName"       -> "Wikipedia",
		"URLFetchFun"       :> (
			With[
				{params = Lookup[{##2}, "Parameters", {}]},
				URLFetch[#1,
					{"StatusCode", "ContentData"},
					Sequence@@FilterRules[{##2}, Except["Parameters" | "Headers"]],
					"Parameters" -> Cases[params, Except[Rule["apikey", _]]],
					"Headers" -> {}
				]
			] & )
			,
		"ClientInfo"        :> {},
		"RawGets"           ->
			{
				"RawMainRequest",
				"RawMetricPageviewArticleRequest"
			},
		"Gets"              ->
			{
			"test",
				"ArticleContributors",
				"ArticleOpenSearch",
				"ArticlePlaintext",
				"ArticleWikicode",
				"BacklinksList",
				"BacklinksRules",
				"CategoryArticles",
				"CategoryArticleIDs",
				"CategoryLinks",
				"CategoryMembers",
				"CategoryMemberIDs",
				"CategorySearch",
				"ContentSearch",
				"ContributorArticles",
				"ExternalLinks",
				"GeoNearbyArticles",
				"GeoNearbyDataset",
				"GeoPosition",
				"ImageDataset",
				"ImageList",
				"ImageURLs",
				"LanguagesList",
				"LanguagesURLRules",
				"LanguagesURLs",
				"LinksRules",
				"LinksList",
				"PageID",
				"PageViewsArticle",
				"ParentCategories",
				"RandomArticle",
				"Revisions",
				"SeeAlsoList",
				"SeeAlsoRules",
				"SummaryPlaintext",
				"SummaryWikicode",
				"Tables",
				"Title",
				"TitleSearch",
				"TitleTranslationRules",
				"TitleTranslations",
				"WikipediaRecentChanges"
			},
		"Posts"             -> {},
		"RawPosts"          -> {},
		"Information"       -> "Import Wikipedia API data to the Wolfram Language"
	}

$requestHead;

formatresults[data_] := FromCharacterCode[data[[2]], "UTF-8"]
importresults[data_] := Block[
	{decoded, imported},

	If[SameQ[data[[1]], 200],
		decoded = Quiet[FromCharacterCode[data[[2]], "UTF-8"], {$CharacterEncoding::utf8}];
		imported = Quiet[ImportString[If[TrueQ[$VersionNumber < 11.1], decoded, ToString[decoded, CharacterEncoding -> "UTF8"]], "RawJSON"]];

		If[AssociationQ[imported] || ListQ[imported],
			If[KeyExistsQ[imported, "error"],
				Message[ServiceExecute::apierr, Lookup[imported, "code", ""] <> " | " <> Lookup[imported, "code", "info"]];
				Throw[$Failed]
			];

			imported
		,
			Throw[$Failed]
		]
	,
		Message[WikipediaData::problm];
		Throw[$Failed]
	]
]

importresultsmetrics[data_] := Block[
	{decoded, imported},

	Which[
		SameQ[data[[1]], 200],
			decoded = Quiet[FromCharacterCode[data[[2]], "UTF-8"], {$CharacterEncoding::utf8}];
			imported = Quiet[ImportString[If[TrueQ[$VersionNumber < 11.1], decoded, ToString[decoded, CharacterEncoding -> "UTF8"]], "RawJSON"]];

			If[AssociationQ[imported] || ListQ[imported],
				If[KeyExistsQ[imported, "error"],
					Message[ServiceExecute::apierr, Lookup[imported, "code", ""] <> " | " <> Lookup[imported, "code", "info"]];
					Throw[$Failed]
				];

				imported
			,
				Throw[$Failed]
			]
		,

		SameQ[data[[1]], 404],
			decoded = Quiet[FromCharacterCode[data[[2]], "UTF-8"], {$CharacterEncoding::utf8}];
			imported = Quiet[ImportString[If[TrueQ[$VersionNumber < 11.1], decoded, ToString[decoded, CharacterEncoding -> "UTF8"]], "RawJSON"]];
			Message[WikipediaData::date, Lookup[imported, "detail", "MissingDetail"]];
			Missing["NotAvailable"]
		,

		True,
			Message[WikipediaData::problm];
			Throw[$Failed]
	]
]


wikipediadata[___]:=$Failed

wikipediadata["RawMainRequest"] := {
	"URL"               -> (ToString[StringForm["http://`1`.wikipedia.org/w/api.php", ##]] &),
	"HTTPSMethod"       -> "GET",
	"Parameters"        -> {
								"accontinue",
								"aclimit",
								"acprefix",
								"acprop",
								"action",
								"blcontinue",
								"bllimit",
								"blnamespace",
								"bltitle",
								"cmcontinue",
								"cmdir",
								"cmlimit",
								"cmnamespace",
								"cmprop",
								"cmsort",
								"cmtitle",
								"clcontinue",
								"cllimit",
								"colimit",
								"continue",
								"coprimary",
								"coprop",
								"ellimit",
								"eloffset",
								"excontinue",
								"explaintext",
								"format",
								"formatversion",
								"gscoord",
								"gslimit",
								"gsradius",
								"iiprop",
								"iiurlheight",
								"iiurlwidth",
								"imcontinue",
								"imlimit",
								"incontinue",
								"list",
								"limit",
								"lllang",
								"lllimit",
								"llprop",
								"pageids",
								"pccontinue",
								"pclimit",
								"plcontinue",
								"pllimit",
								"plnamespace",
								"prop",
								"rccontinue",
								"rcend",
								"rclimit",
								"rcprop",
								"rcstart",
								"redirects",
								"rncontinue",
								"rnlimit",
								"rnnamespace",
								"rvcontinue",
								"rvend",
								"rvlimit",
								"rvprop",
								"rvsection",
								"rvstart",
								"search",
								"srlimit",
								"sroffset",
								"srsearch",
								"srwhat",
								"titles",
								"uccontinue",
								"uclimit",
								"ucprop",
								"ucuser"
							},
	"RequiredParameters"-> {"action"},
	"PathParameters"    -> {"pathlanguage"},
	"ResultsFunction"   -> importresults
}

wikipediadata["RawMetricPageviewArticleRequest"] := {
	"URL"               -> (ToString[StringForm["https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/`1`/`2`/`3`/`4`/`5`/`6`/`7`", ##]] &),
	"HTTPSMethod"       -> "GET",
	"Parameters"        -> {},
	"RequiredParameters"-> {},
	"PathParameters"    -> {"project", "access", "agent", "article", "granularity", "start", "end"},
	"ResultsFunction"   -> importresultsmetrics
}

wikipediacookeddata[prop_,id_,rules___Rule]:=wikipediacookeddata[prop,id,{rules}]

(*formats = {"dbg", "dbgfm", "dump", "dumpfm", "json", "jsonfm", "php", "phpfm", "txt", "txtfm", "wddx", "wddxfm", "xml", "xmlfm", "yaml", "yamlfm"};*)

(****************************************************************************************************)

Wikipediacookeddata[args___] := Module[
	{res = Catch[wikipediacookeddata[args]]},
	If[!FailureQ[res], res]
]

wikipediacookeddata["ArticleContributors", id_, args_] := Block[
	{invalidParameters, result, tmpResult, titles, tmpTitles, strTitles, posNonMissingTitles, continue, limit, response, tmpSize, elementsLeft, firstIteration, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];
	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	If[tmpTitles =!= {},
		(
			elementsLeft = limit;
			firstIteration = True;
			continue = "";
			tmpResult = {};

			While[firstIteration || (continue =!= "" && elementsLeft > 0),
				firstIteration = False;

				tmpSize = Min[elementsLeft, WikipediaPageSize];
				response = WikipediaGetArticleContributors[id, #, tmpSize, language, continue];
				continue = Lookup[Lookup[response, "continue", {}], "pccontinue", ""];
				response = First[Lookup[Lookup[response, "query", {}], "pages", {{}}]];
				response = Lookup[Lookup[response, "contributors", {}], "name", {}];

				(*Sometimes Wikipedia returns less results than the amount requested even though there are enough items, so we have to use "continue" to retrieve  the rest.*)
				tmpSize = Min[elementsLeft, Length[response]];
				elementsLeft -= tmpSize;
				response = Take[response, UpTo[tmpSize]];

				AppendTo[tmpResult, response]
			];

			AppendTo[result, Rule[#, Flatten[tmpResult]]];
		) & /@ tmpTitles;

		result = titles /. result
	,
		result = tmpTitles
	];

	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["ArticleOpenSearch", id_, args_] := Block[
	{invalidParameters, result, search, limit, language},
	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "Search", Language, MaxItems, Method}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args, False];

	If[!KeyExistsQ[args, "Search"],
		Message[ServiceExecute::nparam, "Search"];
		Throw[$Failed]
	];

	search = Lookup[args, "Search"];
	If[!MatchQ[search, _?ListQ], search = {search}];
	If[!(MatchQ[search, {__?StringQ}] && !MemberQ[search, ""]),
		Message[ServiceExecute::nval, "Search", "Wikipedia"];
		Throw[$Failed]
	];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 50]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];
	If[limit > 500, limit = 500];
	result = WikipediaArticleOpenSearch[id, #, limit, language["From"]] & /@ search;
	result = Take[#, {2, 3}] & /@ result;
	result = MapThread[Association["Title" -> #1, "Snippet" -> #2] &, #] & /@ result;
	If[Length[Flatten[result]]!=0,
		result = If[SameQ[language["From"], language["To"]],
			Lookup[#, "Title"]
		,
			DeleteCases[
				wikipediacookeddata["Title", id, {"Title" -> Lookup[#, "Title"], Language -> {language["From"] -> language["To"]}}],
				_Missing
			]
		] & /@ result;
	];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["ArticlePlaintext", id_, args_] := Block[
	{invalidParameters, result, titles, replacementRules, tmpTitles, strTitles, posNonMissingTitles, continue, parameters, format = "json", firstIteration, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];
	language = WikipediaValidateLanguage[args];
	titles = WikipediaTitleMapping[id, args];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];
	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];
	tmpTitles = Partition[tmpTitles, UpTo[50]];
	result = {};
	If[tmpTitles =!= {},
		(
			firstIteration = True;
			strTitles = StringJoin[Riffle[#, "|"]];

			continue = "";

			While[firstIteration || (!firstIteration && continue =!= ""),
				replacementRules = WikipediaGetPlaintext[id, strTitles, continue, language];

				continue = Lookup[Lookup[replacementRules, "continue", {}], "excontinue", ""];

				replacementRules = Cases[
					(#["title"] -> Lookup[#, "extract", Missing["NotAvailable"]]) & /@ Lookup[Lookup[replacementRules, "query", {}], "pages", {}],
					Rule[a_, b_String] :> Rule[a, b],
					1
				];

				AppendTo[result, replacementRules];
				firstIteration = False;
			]
		) & /@ tmpTitles;

		result = Flatten[result];
		result = titles /. result;
		result
	,

		result = titles
	];

	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["ArticleWikicode", id_, args_] := Block[
	{invalidParameters, result, titles, tmpTitles, posNonMissingTitles, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];
	tmpTitles = Partition[tmpTitles, UpTo[50]];

	result = WikipediaGetArticleWikicode[id, StringJoin[Riffle[#, "|"]], language] & /@ tmpTitles;
	result = Cases[
		(#["title"] -> Lookup[
			First[Lookup[#, "revisions", {{}}]],
			"content",
			Missing["NotAvailable"]
		]) & /@ Flatten[Lookup[Lookup[result, "query", {}], "pages", {}], 1],
		Rule[a_, b_String] :> Rule[a, b],
		1
	];

	result = titles /. result;
	If[Length[result] === 1, result = First[result]];
	result
]

(*Find all pages that link to the given page*)
wikipediacookeddata["BacklinksList", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["BacklinksRules", id, args];

	Which[
		MatchQ[result, {__Rule}],
			result = Union[(# /. (title_ -> _ ) :> title)  & /@ result]
		,

		MatchQ[result, {__List}],
			result = Union[((# /. (title_ -> _ ) :> title)  & /@ #)] &  /@ result
	];
	result
]

(*Find all pages that link to the given page*)
wikipediacookeddata["BacklinksRules", id_, args_] := Block[
	{invalidParameters, result, titles, limit, level, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxLevel", "MaxLevelItems", "PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, "MaxLevelItems", All];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!MatchQ[limit, _Integer] && !MatchQ[limit, All],
		Message[ServiceExecute::nval, "MaxLevelItems", "Wikipedia"];
		Throw[$Failed]
	];

	level = Lookup[args, "MaxLevel", 1];
	If[StringQ[level], level = FromDigits[level]];
	If[!MatchQ[level, _Integer],
		Message[ServiceExecute::nval, "MaxLevel", "Wikipedia"];
		Throw[$Failed]
	];

	If[titles =!= {},
		result = If[MatchQ[#, _String], WikipediaBackLinksTree[id, #, limit, level, language], #] & /@ titles;
		If[Length[result] === 1, result = First[result]]
	,
		result = titles
	];

	result
]

(*List all articles in a given category*)
wikipediacookeddata["CategoryArticles", id_, args_] := Block[
	{invalidParameters, result, category, limit, tmpSize, continue, response, showID, firstIteration = True, elementsLeft, tmpResult, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"Category", "MaxItems", "ShowID", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];
	
	If[!KeyExistsQ[args, "Category"],
		Message[ServiceExecute::nparam, "Category"];
		Throw[$Failed]
	];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	showID = Lookup[args, "ShowID", False];
	If[showID =!= True && showID =!= False,
		Message[ServiceExecute::nval, "ShowID", "Wikipedia"];
		Throw[$Failed]
	];

	category = Lookup[args, "Category"];
	If[!MatchQ[category, _String] && !MatchQ[category, {__String}],
		Message[ServiceExecute::nval, "Category", "Wikipedia"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	If[MatchQ[category, _String], category = {category}];
	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			response = WikipediaGetCategoryArticles[id, #, tmpSize, language, continue];
			continue = Lookup[Lookup[response, "continue", {}], "cmcontinue", ""];
			response = Lookup[Lookup[Lookup[response, "query", {}], "categorymembers", {}], If[showID, "pageid", "title"], {}];

			(*Sometimes Wikipedia returns less results than the amount requested even though there are enough items, so we have to use "continue" to retrieve  the rest.*)
			tmpSize = Min[elementsLeft, Length[response]];
			elementsLeft -= tmpSize;
			response = Take[response, UpTo[tmpSize]];

			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ category;

	result = Replace[category, result, 1];
	result = Replace[result, Rule[{}, Missing["NotAvailable"]], 1];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["CategoryArticleIDs", id_, args_] := Block[
	{params = args},

	wikipediacookeddata["CategoryArticles", id, AppendTo[params, "ShowID" -> True]]
]

(*extracts the category tree up to certain level*)
wikipediacookeddata["CategoryLinks", id_, args_] := Block[
	{invalidParameters, result, category, treeLevel, root, leaves, language, limit},

	invalidParameters = Select[Keys[args], !MemberQ[{"Category", "MaxLevel", "MaxLevelItems", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];
	
	If[!KeyExistsQ[args, "Category"],
		Message[ServiceExecute::nparam, "Category"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	category = Lookup[args, "Category"];
	If[!MatchQ[category, _String] && !MatchQ[category, {__String}],
		Message[ServiceExecute::nval, "Category", "Wikipedia"];
		Throw[$Failed]
	];

	limit = Lookup[args, "MaxLevelItems", All];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!MatchQ[limit, _Integer] && !MatchQ[limit, All],
		Message[ServiceExecute::nval, "MaxLevelItems", "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[category, _String],
		category = {category}
	];

	treeLevel = Lookup[args, "MaxLevel", 2];
	If[StringQ[treeLevel], treeLevel = FromDigits[treeLevel]];
	If[!(MatchQ[treeLevel, _Integer] && treeLevel > 0),
		Message[ServiceExecute::nval, "MaxLevel", "Wikipedia"];
		Throw[$Failed]
	];

	result = {};

	root = category;
	result = {}& /@category;

	Do[
		leaves = WikipediaCategoryExtraction[id, #, limit, language] & /@ root;
		result = Flatten[#] & /@ Transpose[{result, leaves}];
		root = Flatten[# /. Rule[a_,b_] :> b] & /@ leaves;
	,
		{i, 1, treeLevel}
	];

	result = Union[#] & /@ result;
	If[Length[result] === 1,result = First[result]];

	result
]

(*List all pages in a given category*)
wikipediacookeddata["CategoryMembers", id_, args_] := Block[
	{invalidParameters, result, tmpResult, response, category, limit, tmpSize, elementsLeft, continue, showID, firstIteration = True, language},
	
	invalidParameters = Select[Keys[args], !MemberQ[{"Category", "MaxItems", "ShowID", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	If[!KeyExistsQ[args, "Category"],
		Message[ServiceExecute::nparam, "Category"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];	
		Throw[$Failed]
	];

	showID = Lookup[args, "ShowID", False];
	If[showID =!= True && showID =!= False,
		Message[ServiceExecute::nval, "ShowID", "Wikipedia"];	
		Throw[$Failed]
	];

	category = Lookup[args, "Category"];
	If[!MatchQ[category, _String] && !MatchQ[category, {__String}],
		Message[ServiceExecute::nval, "Category", "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[category, _String], category = {category}];
	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			response = WikipediaGetCategoryMembers[id, #, tmpSize, language, continue];
			continue = Lookup[Lookup[response, "continue", {}], "cmcontinue", ""];
			response = Lookup[Lookup[Lookup[response, "query", {}], "categorymembers", {}], If[showID, "pageid", "title"], {}];

			(*Sometimes Wikipedia returns less results than the amount requested even though there are enough items, so we have to use "continue" to retrieve  the rest.*)
			tmpSize = Min[elementsLeft, Length[response]];
			elementsLeft -= tmpSize;
			response = Take[response, UpTo[tmpSize]];

			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ category;

	result = Replace[category, result, 1];
	result = Replace[result, Rule[{}, Missing["NotAvailable"]], 1];
	If[Length[result] === 1, result = First[result]];
	result
]

(*List all pages in a given category*)
wikipediacookeddata["CategoryMemberIDs", id_, args_] := Block[
	{params = Normal[args]},

	wikipediacookeddata["CategoryMembers", id, Join[{"ShowID" -> True}, params]]
]

(*Enumerate all categories*)
wikipediacookeddata["CategorySearch", id_, args_] := Block[
	{invalidParameters, result, tmpResult, response, search, limit, continue, firstIteration = True, elementsLeft, tmpSize, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "Search", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	If[!KeyExistsQ[args, "Search"],
		Message[ServiceExecute::nparam, "Search"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	search = Lookup[args, "Search"];
	If[!MatchQ[search, _String] && !MatchQ[search, {__String}],
		Message[ServiceExecute::nval, "Search", "Wikipedia"];
		Throw[$Failed]
	];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 50]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[search, _String], search = {search}];
	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			response = WikipediaGetCategorySearch[id, #, tmpSize, language, continue];
			continue = Lookup[Lookup[response, "continue", {}], "accontinue", ""];
			response = Lookup[Lookup[Lookup[response, "query", {}], "allcategories", {}], "category", {}];
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ search;

	result = Replace[search, result, 1];
	If[Length[result] === 1, result = First[result]];
	result
]

(*Perform a full content text search*)
wikipediacookeddata["ContentSearch", id_, args_] := Block[
	{invalidParameters, response, result, tmpResult, search, limit, continue, elementsLeft, tmpSize, exact, firstIteration = True, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"Content", "MaxItems", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	If[!KeyExistsQ[args, "Content"],
		Message[ServiceExecute::nparam, "Content"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	search = Lookup[args, "Content"];
	If[!MatchQ[search, _?ListQ], search = {search}];
	If[!(MatchQ[search, {__?StringQ}] && !MemberQ[search, ""]),
		Message[ServiceExecute::nval, "Content", "Wikipedia"];
		Throw[$Failed]
	];

	exact = False; (*TODO: validate this  parameter*)
	(*If[MatchQ[search, {__String}],
		search = StringJoin@Riffle[search, " "];
		exact = True
	];*)

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 50]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			response = WikipediaGetContentSearch[id, #, tmpSize, language, continue, exact];
			continue = Lookup[Lookup[response, "continue", {}], "sroffset", ""];
			response = Lookup[Lookup[response, "query", {}], "search", ""];
			response = (
				<|
					"Title" -> #["title"],
					"Snippet" -> (StringReplace[Lookup[#, "snippet", {}], {"<span class=\"searchmatch\">" -> "", "</span>" -> ""}])
				|>
			) & /@ response;
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ search;

	result = Replace[search, result, 1];
	If[Length[result] === 1, result = First[result]];
	result
]

(*Lists articles made by a contributor*)
wikipediacookeddata["ContributorArticles", id_, args_] := Block[
	{invalidParameters, result, tmpResult, response, contributor, limit, tmpSize, continue, firstIteration = True, elementsLeft, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"Contributor", "MaxItems", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	If[!KeyExistsQ[args, "Contributor"],
		Message[ServiceExecute::nparam, "Contributor"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	contributor = Lookup[args, "Contributor", Throw[$Failed]];
	If[!MatchQ[contributor, _String] &&!MatchQ[contributor, {__String}],
		Message[ServiceExecute::nval, "Contributor", "Wikipedia"];
		Throw[$Failed]
	];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[contributor, _String], contributor = {contributor}];
	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			response = WikipediaGetContributorArticles[id, #, tmpSize, language, continue];
			continue = Lookup[Lookup[response, "continue", {}], "uccontinue", ""];
			response = Lookup[Lookup[Lookup[response, "query", {}], "usercontribs", {}], "title", {}];
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ contributor;

	result = Replace[contributor, result, 1];
	If[Length[result] === 1, result = First[result]];
	result
]

(*Returns external links of the given page(s)*)
wikipediacookeddata["ExternalLinks", id_, args_] := Block[
	{invalidParameters, result, tmpResult, response, titles, tmpTitles, limit, tmpSize, continue, firstIteration = True, elementsLeft, posNonMissingTitles, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[limit, All], limit = Infinity];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			response = WikipediaGetExternalLinks[id, #, tmpSize, language, continue];
			continue = Lookup[Lookup[response, "continue", {}], "eloffset", ""];
			response = Lookup[Lookup[response, "query", {}], "pages", {}];
			response = If[MatchQ[response, _List] && Length[response] > 0, Lookup[Lookup[First[response], "extlinks", {}], "url", {}], {}];
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ tmpTitles;

	result = Replace[titles, result, 1];
	result = Replace[result, Rule[{}, Missing["NotAvailable"]], 1];
	If[Length[result] === 1, result = First[result]];
	result
]

(*Returns pages around the given point*)
wikipediacookeddata["GeoNearbyArticles", id_, args_] := Block[
	{invalidParameters, result, geodisk, titles, tmpTitles, gsradius, coord, geopos, geodistance, format = "json", posNonMissingTitles, resType, limit, ids, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"Geodisk", "GeoDistance", "MaxItems", "PageID", "ResultType", "Title", GeoDisk, GeoDistance, GeoLocation, GeoPosition, Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 50]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];
	If[MatchQ[limit, All] || limit > 500, limit = 500];
	limit = ToString[limit];

	geodisk = Lookup[args, GeoDisk, Lookup[args, "Geodisk"]];
	geopos = Lookup[args, GeoPosition, Lookup[args, GeoLocation]];

	resType = Lookup[args, "ResultType", "Normal"];

	titles = Lookup[args, "Title"]; (*TODO: Check exclusion geodisk-titles error message*)
	ids = Lookup[args, "PageID"];
	If[Length[DeleteCases[{titles, ids, geodisk, geopos}, _Missing]] > 1,
		Message[WikipediaData::excpar, StringRiffle[{GeoDisk, "PageID", "Title"}, ", "], "Wikipedia"];
		Throw[$Failed]
	];

	If[!MatchQ[geodisk, _Missing],
		geopos = First[geodisk];
		If[(MatchQ[GeoPosition[geopos], GeoPosition[{_?NumericQ, _?NumericQ}]] || MatchQ[GeoPosition[geopos], GeoPosition[{_?NumericQ, _?NumericQ, _?NumericQ}]]) && MatchQ[geodisk, GeoDisk[___, Quantity[a_, b_], ___]],
			geopos = GeoPosition[geopos]
		,
			Message[ServiceExecute::nval, First@Select[Keys[args], MemberQ[{GeoDisk, "Geodisk"}, #]&], "Wikipedia"];
			Throw[$Failed]
		]
	];

	geodistance = Lookup[args, GeoDistance, Lookup[args, "GeoDistance"]];

	Which[
		!MatchQ[geodisk, _Missing],
			If[!MatchQ[geodistance, _Missing],
				Message[WikipediaData::excpar, StringRiffle[{GeoDisk, "GeoDistance"}, ", "], "Wikipedia"];
				Throw[$Failed]
			];

			coord = {ToString[QuantityMagnitude[Latitude[geopos]]] <> "|" <> ToString[QuantityMagnitude[Longitude[geopos]]]};
			gsradius = Replace[geodisk, GeoDisk[___, Quantity[a_, b_], ___] :> UnitConvert[Quantity[a, b], "Meters"]];
			gsradius = N[QuantityMagnitude[gsradius]]
		,

		!MatchQ[titles, _Missing] || !MatchQ[ids, _Missing],
			titles = WikipediaTitleMapping[id, args];
			gsradius = Lookup[args, GeoDistance, Lookup[args, "GeoDistance", 10000]];

			Which[
				NumberQ[gsradius],
					gsradius = N[gsradius]
				,

				MatchQ[gsradius, Quantity[_?NumberQ, _String]],
					gsradius = N[QuantityMagnitude[UnitConvert[gsradius,  "Meters"]]]
				,

				True,
					Message[ServiceExecute::nval, "GeoDistance", "Wikipedia"];
					Throw[$Failed]
			];

			posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];
			tmpTitles = Extract[titles, posNonMissingTitles];
			tmpTitles = DeleteDuplicates[tmpTitles];

			result = importresults[
				KeyClient`rawkeydata[
					id,
					"RawMainRequest",
					{
						"pathlanguage" -> language,
						"action" -> "query",
						"format" -> "json",
						"formatversion" -> "2",
						"continue" -> "",
						"prop" -> "coordinates",
						"titles" -> StringRiffle[#, "|"]
					}
				]
			] & /@ tmpTitles;

			coord = Rule[
				#["title"],
				If[MatchQ[#, _Missing],
					#
				,
					ToString[First[#]["lat"]] <> "|" <> ToString[First[#]["lon"]]
				] & [Lookup[#, "coordinates", Missing["CoordinatesNotFound"]]]
			] & /@ Flatten[Lookup[Lookup[result, "query", {}], "pages", {}]];

			coord = Replace[titles, coord, 1];
		,

		!MatchQ[geopos, _Missing],
			geopos = GeoPosition[geopos];
			coord = {ToString[QuantityMagnitude[Latitude[geopos]]] <> "|" <> ToString[QuantityMagnitude[Longitude[geopos]]]};

			gsradius = Lookup[args, GeoDistance, Lookup[args, "GeoDistance", 10000]];

			Which[
				NumberQ[gsradius],
					gsradius = N[gsradius]
				,

				MatchQ[gsradius, Quantity[_?NumberQ, _String]],
					gsradius = N[QuantityMagnitude[UnitConvert[gsradius,  "Meters"]]]
				,

				True,
					Message[ServiceExecute::nval, "GeoDistance", "Wikipedia"];
					Throw[$Failed]
			]
		,

		True,
			Message[WikipediaData::onepar, StringRiffle[{"Geodisk", "PageID", "Title"}, ", "], "Wikipedia"];
			Throw[$Failed]

	];

	If[gsradius > 10000, gsradius = 10000];
	If[gsradius < 10, gsradius = 10];
	gsradius = ToString[gsradius];

	result = If[MatchQ[#, _Missing],
		#
	,
		<|
			"PageID" -> #["pageid"],
			"Title" -> #["title"],
			"Position" -> GeoPosition[{#["lat"], #["lon"]}],
			"Distance" -> Quantity[#["dist"], "Meters"]
		|> & /@ Lookup[
			Lookup[
				importresults[
					KeyClient`rawkeydata[
						id,
						"RawMainRequest",
						{
							"pathlanguage" -> language,
							"format" -> format,
							"formatversion" -> "2",
							"action" -> "query",
							"list" -> "geosearch",
							"gslimit" -> limit,
							"gsradius" -> gsradius,
							"gscoord" -> #,
							"continue" -> ""
						}
					]
				],
				"query",
				{}
			],
			"geosearch",
			{}
		]
	] & /@ coord;

	result = Switch[resType,
		"RawData",
			result
		,

		_,
			If[MatchQ[#, _Missing],
				#
			,
				Lookup[#, "Title", {}]
			] & /@ result
	];

	If[Length@result === 1,
		result = First[result]
	];

	result
]

(*Returns pages around the given point*)
wikipediacookeddata["GeoNearbyDataset", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["GeoNearbyArticles", id, Join[{"ResultType" -> "RawData"}, Normal[args]]];
	If[MatchQ[result, {__Association}] || MatchQ[result, _Missing],
		result = {result}
	];

	result = If[MatchQ[#, _Missing],
		#
	,
		Dataset[#]
	] & /@ result;

	If[Length@result === 1, result = First[result]];
	result
]

(*Returns coordinates of the given page(s)*)
wikipediacookeddata["GeoPosition", id_, args_] := Block[
	{invalidParameters, result, titles, tmpTitles, posNonMissingTitles, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];
	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	result = importresults[
		KeyClient`rawkeydata[
			id,
			"RawMainRequest",
			{
				"pathlanguage" -> language,
				"action" -> "query",
				"format" -> "json",
				"formatversion" -> "2",
				"continue" -> "",
				"prop" -> "coordinates",
				"coprimary"->"primary",
				"coprop"->"globe",
				"colimit"->"max",
				"titles" -> StringReplace[#, " " -> "_"]
			}
		]
	] & /@ tmpTitles;

	result = Rule[
		#["title"],
		If[MatchQ[#, _Missing],
			#
		,
			If[ToLowerCase[First[#]["globe"]] === "earth",
				GeoPosition[{First[#]["lat"], First[#]["lon"]}]
			,
				GeoPosition[{First[#]["lat"], First[#]["lon"]}, Interpreter["AstronomicalObject"][First[#]["globe"]]]
			]
		] & [Lookup[#, "coordinates", Missing["CoordinatesNotFound"]]]
	] & /@ Flatten[Lookup[Lookup[result, "query", {}], "pages", {}]];

	result = Replace[titles, result, 1];
	If[Length@result === 1,
		result = First[result]
	];

	result
]

wikipediacookeddata["ImageDataset", id_, args_] := Block[
	{invalidParameters, titles, response, result, finalResult = {}, limit, size, elements, tmpElements, keys0, values0, filter, parameters, i, valid, posNonMissingTitles, tmpResult, tmpFileNames, tmpLenght, language, validElements = {"Thumbnail", "Date", "User", "UserID", "Comment", "ParsedComment", "Title", "Size", "Dimensions", "Sha1", "Mime", "Thumbmime", "MediaType", "Metadata", "CommonMetadata", "ExternalMetadata", "ArchiveName", "BitDepth", "UploadWarning"}, styledToWikiNameRules = {
			"ArchiveName" -> "archivename",
			"BitDepth" -> "bitdepth",
			"Comment" -> "comment",
			"CommonMetadata" -> "commonmetadata",
			"Date" -> "timestamp",
			"Dimensions" -> "dimensions",
			"ExternalMetadata" -> "extmetadata",
			"MediaType" -> "mediatype",
			"Metadata" -> "metadata",
			"Mime" -> "mime",
			"ParsedComment" -> "parsedcomment",
			"Sha1" -> "sha1",
			"Size" -> "size",
			"Thumbmime" -> "thumbmime",
			"Thumbnail" -> "thumbnail",
			"Title" -> "canonicaltitle",
			"UploadWarning" -> "uploadwarning",
			"User" -> "user",
			"UserID" -> "userid"
		}
	},

	invalidParameters = Select[Keys[args], !MemberQ[{"Elements", "MaxItems", "PageID", "Size", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	size = Lookup[args, "Size", 200];
	If[!(MatchQ[size, _Integer] && size > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	,
		size = ToString[size];
	];

	elements = Lookup[args, "Elements", "Data"];
	Which[
		MatchQ[elements, _?StringQ],
			valid = MemberQ[validElements, elements] || MatchQ[elements, "Data" | "FullData"];
			If[!MatchQ[elements, "Data" | "FullData"],
				elements = {elements}
			]
		,

		MatchQ[elements, {__?StringQ}],
			valid = SubsetQ[validElements, elements]
		,

		True,
			valid = False
	];

	If[!valid,
		Message[ServiceExecute::nval, "Elements", "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[elements, {__?StringQ}],
		elements = elements /. styledToWikiNameRules
	];

	elements = Switch[elements,
		"Data",
			{"timestamp", "user", "userid", "comment", "canonicaltitle"}
		,

		"FullData",
			{"timestamp", "user", "userid", "comment", "parsedcomment", "canonicaltitle", "size", "dimensions", "sha1", "mime", "thumbmime", "mediatype", "metadata", "commonmetadata", "extmetadata", "archivename", "bitdepth", "uploadwarning"}
		,

		_,
			elements
	];

	tmpElements = elements;
	tmpElements = StringJoin@Riffle[Join[{"url"}, tmpElements], "|"];

	result = WikipediaExtractImageFilename[id, args];

	posNonMissingTitles = Position[result, Except[_?MissingQ], {1}, Heads -> False];

	tmpResult = Extract[result, posNonMissingTitles];
	tmpFileNames = DeleteDuplicates@Flatten[tmpResult];
	tmpFileNames = Partition[tmpFileNames, UpTo[50]];

	response = importresults[
		KeyClient`rawkeydata[id, "RawMainRequest", {
			"pathlanguage" -> language,
			"format" -> "json",
			"formatversion" -> "2",
			"action" -> "query",
			"prop" -> "imageinfo",
			"iiprop" -> tmpElements,
			"iiurlwidth" -> size,
			"iiurlheight" -> size,
			"titles" -> StringRiffle[#, "|"],
			"redirects" -> ""
		}]
	] & /@ tmpFileNames;

	tmpElements = DeleteDuplicates[Join[{"thumbnail"}, elements]];

	response = Flatten[Lookup[Lookup[response, "query", {}], "pages", {}]];
	keys0 = Lookup[response,"title"];
	values0 = Flatten[Lookup[response, "imageinfo", Missing["NotAvailable"]]];
	values0 = Replace[
		values0,
		asoc: _Association?AssociationQ :> KeyTake[Prepend[asoc, "thumbnail" -> Import[asoc["thumburl"]]], tmpElements], (*thumburl can be changed to url in order to retrieve the original image. Beware that the original image may have a file extension which is not currently supported by Import.*)
		{1}
	];
	values0 = Replace[
		values0,
		asoc: _Association?AssociationQ :> Replace[
			asoc,
			a_?AssociationQ :> KeyMap[(If[StringQ[#], Capitalize[#], #] &)][a],
			{0, Infinity}
		],
		{1}
	];
	values0 = (Association@KeyValueMap[WikipediaRenameAndFormat[#1, #2] &, #] &) /@ values0;
	response = Thread[Rule[keys0, values0]];

	finalResult = Replace[result, response, 2];
	finalResult = If[MatchQ[#, _Missing], #, Dataset[#]] & /@ finalResult;

	If[Length@finalResult === 1,
		finalResult = First@finalResult
	];

	finalResult
]

(*Gets image thumbnails from a given page*)
wikipediacookeddata["ImageList", id_, args_] := Block[
	{invalidParameters, limit, result, tmpResult, fileNames, tmpFileNames, posNonMissingTitles, response, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	fileNames = WikipediaExtractImageFilename[id, args]; (*Title validation is done here*)
	posNonMissingTitles = Position[fileNames, Except[_?MissingQ], {1}, Heads -> False];

	tmpResult = Extract[fileNames, posNonMissingTitles];
	tmpFileNames = DeleteDuplicates@Flatten[tmpResult];
	tmpFileNames = Partition[tmpFileNames, UpTo[50]];

	response = importresults[
		KeyClient`rawkeydata[id, "RawMainRequest", {
			"pathlanguage" -> language,
			"format" -> "json",
			"formatversion" -> "2",
			"action" -> "query",
			"prop" -> "imageinfo",
			"iiprop" -> "url",
			"iiurlwidth" -> "1000",
			"iiurlheight" -> "1000",
			"titles" -> StringRiffle[#, "|"],
			"redirects" -> ""
		}]
	] & /@ tmpFileNames;

	response = Rule[
		#["title"],
		If[MatchQ[#, _Missing],
			#
		,
			Import[First[#]["thumburl"]] (*thumburl can be changed to url in order to retrieve the original image. Beware that the original image may have a file extension which is not currently supported by Import.*)
		] & [Lookup[#, "imageinfo", Missing["NotAvailable"]]]
	] & /@ Flatten[Lookup[Lookup[response, "query", {}], "pages", {}]];

	result = Replace[fileNames, response, 2];
	If[Length@result === 1,
		result = First[result]
	];

	result
]

(*Gets image URLs from a given page*)
wikipediacookeddata["ImageURLs", id_, args_] := Block[
	{invalidParameters, limit, result, tmpResult, fileNames, tmpFileNames, posNonMissingTitles, response, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	fileNames = WikipediaExtractImageFilename[id, args]; (*Title validation is done here*)
	posNonMissingTitles = Position[fileNames, Except[_?MissingQ], {1}, Heads -> False];
	tmpResult = Extract[fileNames, posNonMissingTitles];
	tmpFileNames = DeleteDuplicates@Flatten[tmpResult];
	tmpFileNames = Partition[tmpFileNames, UpTo[50]];

	response = importresults[
		KeyClient`rawkeydata[id, "RawMainRequest", {
			"pathlanguage" -> language,
			"format" -> "json",
			"formatversion" -> "2",
			"action" -> "query",
			"prop" -> "imageinfo",
			"iiprop" -> "url",
			"titles" -> StringRiffle[#, "|"],
			"redirects" -> ""
		}]
	] & /@ tmpFileNames;

	response = Rule[
		#["title"],
		If[MatchQ[#, _Missing],
			#
		,
			Lookup[First[#], "descriptionurl", Missing["NotAvailable"]]
		] & [Lookup[#, "imageinfo", Missing["NotAvailable"]]]
	] & /@ Flatten[Lookup[Lookup[response, "query", {}], "pages", {}]];

	result = Replace[fileNames, response, 2];

	If[Length@result === 1,
		result = First[result]
	];

	result
]

wikipediacookeddata["LanguagesList", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["LanguagesURLRules", id, args];
	result = Replace[result, Rule[a_, b_] :> a, 2];
	result
]

(*Returns all interlanguage links from the given page(s)*)
wikipediacookeddata["LanguagesURLRules", id_, args_] := Block[
	{invalidParameters, result, response, titles, limit, tmpTitles, posNonMissingTitles, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 500]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];
	If[MatchQ[limit, All] || limit > 500, limit = 500];
	limit = ToString[limit];

	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	response = importresults[
		KeyClient`rawkeydata[id, "RawMainRequest", {
			"pathlanguage" -> language,
			"action" -> "query",
			"prop" -> "langlinks",
			"continue" -> "",
			"format" -> "json",
			"formatversion" -> "2",
			"lllimit" -> limit,
			"llprop" -> "url|langname|autonym",
			"titles" -> #,
			"redirects" -> ""
		}]
	] & /@ tmpTitles;

	response = Rule[
		#["title"],
		(#["langname"] -> #["url"]) & /@ Lookup[#, "langlinks", {}]
	] & /@ Flatten[Lookup[Lookup[response, "query", {}], "pages", {}]];

	result = Replace[titles, response, {1}];
	If[Length@result === 1,
		result = First[result]
	];

	result
]

wikipediacookeddata["LanguagesURLs", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["LanguagesURLRules", id, args];
	result = Replace[result, Rule[a_, b_] :> b, 2];
	result
]

wikipediacookeddata["LinksList", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["LinksRules", id, args];
	result = Replace[result, Rule[a_, b_] :> b, 2];
	result
]

wikipediacookeddata["LinksRules", id_, args_] := Block[
	{invalidParameters, result, titles, limit, level, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxLevel", "MaxLevelItems", "PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, "MaxLevelItems", All];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!MatchQ[limit, _Integer] && !MatchQ[limit, All],
		Message[ServiceExecute::nval, "MaxLevelItems", "Wikipedia"];
		Throw[$Failed]
	];

	level = Lookup[args, "MaxLevel", 1];
	If[!MatchQ[level, _Integer],
		Message[ServiceExecute::nval, "MaxLevel", "Wikipedia"];
		Throw[$Failed]
	];

	If[titles =!= {},
		result = If[MatchQ[#, _String], WikipediaLinksTree[id, #, limit, level, language], #] & /@ titles;
		If[Length[result] === 1, result = First[result]]
	,
		result = titles
	];

	result
]

(*Finds the wikipedia id of an Entity or title*)
wikipediacookeddata["PageID", id_, args_] := Block[
	{invalidParameters, param = args, result},

	invalidParameters = Select[Keys[args], !MemberQ[{"PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	AppendTo[param, "ShowID" -> True];
	result = WikipediaTitleMapping[id, param];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["PageViewsArticle", id_, args_] := Block[
	{$DateStringFormat = {"Year", "Month", "Day"}, invalidParameters, param = args, titles, response, result, posNonMissingTitles, access, agent, granularity, date, startDate, endDate, language, project},

	invalidParameters = Select[Keys[args], !MemberQ[{"Access", "Agent", "Granularity", "PageID", "Project", "Date", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	access = Lookup[param, "Access", Keys[First[WikipediaMetricsAccess]]];
	If[!StringQ[access],
		Message[ServiceExecute::nval, "Access", "Wikipedia"];
		Throw[$Failed]
	];

	access = Lookup[
		WikipediaMetricsAccess,
		ToLowerCase[access],
		(
			Message[ServiceExecute::nval, "Access", "Wikipedia"];
			Throw[$Failed]
		)
	];

	agent = Lookup[param, "Agent", Keys[First[WikipediaMetricsAgent]]];
	If[!StringQ[agent],
		Message[ServiceExecute::nval, "Agent", "Wikipedia"];
		Throw[$Failed]
	];

	agent = Lookup[
		WikipediaMetricsAgent,
		ToLowerCase[agent],
		(
			Message[ServiceExecute::nval, "Agent", "Wikipedia"];
			Throw[$Failed]
		)
	];

	granularity = Lookup[param, "Granularity", Keys[First[WikipediaMetricsGranularity]]];
	If[!StringQ[granularity],
		Message[ServiceExecute::nval, "Granularity", "Wikipedia"];
		Throw[$Failed]
	];

	granularity = Lookup[
		WikipediaMetricsGranularity,
		ToLowerCase[granularity],
		(
			Message[ServiceExecute::nval, "Granularity", "Wikipedia"];
			Throw[$Failed]
		)
	];
	
	date = Lookup[param, "Date", DateObject[{2015, 5, 1}]];
	If[!MatchQ[date, _?DateObjectQ] && !MatchQ[Head[date], Interval],
		Message[ServiceExecute::nval, "Date", "Wikipedia"];
		Throw[$Failed],
		Switch[Head[date],
			DateObject,
				startDate = DateString[date, {"Year", "", "Month", "", "Day"}, TimeZone -> 0]; (* This "" is because of a bug in DateString in M11.1.0 5694601 *)
				endDate = DateString[Today, {"Year", "", "Month", "", "Day"}, TimeZone -> 0];
				,
			Interval,
				startDate = DateString[date[[1,1]], {"Year", "", "Month", "", "Day"}, TimeZone -> 0]; (* This "" is because of a bug in DateString in M11.1.0 5694601 *)
				endDate = DateString[date[[1,2]], {"Year", "", "Month", "", "Day"}, TimeZone -> 0];
		]
	];

	language = WikipediaValidateLanguage[args, False];

	project = Lookup[param, "Project", language["To"] <> ".wikipedia"];
	If[!StringQ[project],
		Message[ServiceExecute::nval, "Project", "Wikipedia"];
		Throw[$Failed]
	];

	titles = WikipediaTitleMapping[id, param];
	titles = If[StringQ[#], StringReplace[#, {" " -> "_"}], #] & /@ titles; (* Metrics require "_" as a separator instead of " " *)

	response = If[StringQ[#],
		importresultsmetrics[
			KeyClient`rawkeydata[id, "RawMetricPageviewArticleRequest", {
				"project" -> project,
				"access" -> access,
				"agent" -> agent,
				"article" -> #,
				"granularity" -> granularity,
				"start" -> startDate,
				"end" -> endDate
			}]
		]
	,
		#
	] & /@ titles;

	result = If[!MissingQ[#],
		{
			DateObject[StringTake[#[[1]], 8]],
			#[[2]]
		} & /@ Lookup[Lookup[#, "items", {}], {"timestamp", "views"}, {}]
	,
		#
	] & /@ response;

	result = If[ListQ[#] && Length[#] > 0,
		TimeSeries[#]
	,
		#
	] & /@ result;

	If[Length[result] === 1, result = First[result]];
	result
]


(*List all categories the page(s) belong to*)
wikipediacookeddata["ParentCategories", id_, args_] := Block[
	{invalidParameters, result, tmpResult, titles, tmpTitles, limit, continue, response, posNonMissingTitles, elementsLeft, tmpSize, firstIteration, parameters, language, tmpFileNames},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[limit, All], limit = Infinity];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			parameters={
				"pathlanguage" -> language,
				"action" -> "query",
				"prop" -> "categories",
				"continue" -> "",
				"format" -> "json",
				"formatversion" -> "2",
				"cllimit" -> ToString[tmpSize],
				"titles" -> #,
				"redirects" -> ""
			};

			If[continue =!= "", AppendTo[parameters, "clcontinue" -> ToString[continue]]];

			response = importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
			continue = Lookup[Lookup[response, "continue", {}], "clcontinue", ""];
			response = Lookup[Lookup[response, "query", {}], "pages", {}];
			response = Lookup[Lookup[First[response], "categories", {}], "title", {}];
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ tmpTitles;

	result = Replace[titles, result, {1}];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["RandomArticle", id_, args_] := Block[
	{invalidParameters, result, limit, namespace, parameters, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 1]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	namespace = 0; (*Article's namespace is 0*)

	parameters = {
		MaxItems -> limit,
		"Namespace" -> namespace,
		Language -> language
	};

	result = wikipediacookeddata["RandomPage", id, parameters];
	result
]

wikipediacookeddata["RandomPage", id_, args_] := Block[
	{invalidParameters, result, response, limit, elementsLeft, namespace, parameters, format = "json", language, firstIteration, continue, tmpSize},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "Namespace", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 1]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	namespace = Lookup[args, "Namespace", 0];
	If[!MatchQ[namespace, _Integer] || namespace < 0,
		Message[ServiceExecute::nval, "Namespace", "Wikipedia"];
		Throw[$Failed]
	];

	elementsLeft = limit;
	firstIteration = True;
	continue = "";
	result = {};

	While[firstIteration || (continue =!= "" && elementsLeft > 0),
		firstIteration = False;
		tmpSize = Min[elementsLeft, WikipediaPageSize];
		parameters = {
			"pathlanguage" -> language,
			"list" -> "random",
			"action" -> "query",
			"format" -> format,
			"formatversion" -> "2",
			"rnlimit" -> ToString[tmpSize],
			"rnnamespace" -> ToString[namespace]
		};

		If[continue =!= "", AppendTo[parameters, "rncontinue" -> ToString[continue]]];

		response = importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
		continue = Lookup[Lookup[response, "continue", {}], "rncontinue", ""];
		response = Lookup[Lookup[Lookup[response, "query", {}], "random", {}], "title", {}];
		elementsLeft = elementsLeft - tmpSize;
		AppendTo[result, response]
	];

	Flatten[result]
]

(*Returns past revisions of the given page(s)*)
wikipediacookeddata["Revisions", id_, args_] := Block[
	{invalidParameters, result, tmpResult, titles, tmpTitles, date, startDate, endDate, limit, continue, response, posNonMissingTitles, elementsLeft, tmpSize, firstIteration, parameters, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"StartDate","EndDate","Date", "MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	startDate = Lookup[args, "StartDate", Now];
	If[!MatchQ[startDate, _?DateObjectQ],
		Message[ServiceExecute::nval, "StartDate", "Wikipedia"];
		Throw[$Failed]
	];
	startDate = DateString[startDate, "ISODateTime", TimeZone -> 0] <> "Z";

	endDate = Lookup[args, "EndDate", DateObject[{2000, 1, 1}]];
	If[!MatchQ[endDate, _?DateObjectQ],
		Message[ServiceExecute::nval, "EndDate", "Wikipedia"];
		Throw[$Failed]
	];
	endDate = DateString[endDate, "ISODateTime", TimeZone -> 0] <> "Z";

	If[MatchQ[limit, All], limit = Infinity];
	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			parameters={
				"pathlanguage" -> language,
				"action" -> "query",
				"prop" -> "revisions",
				"format" -> "json",
				"formatversion" -> "2",
				"rvprop" -> "timestamp|content",
				"rvlimit" -> ToString[tmpSize],
				"titles" -> #,
				"rvstart" -> startDate,
				"rvend" -> endDate,
				"continue" -> ""
			};

			If[continue =!= "", AppendTo[parameters, "rvcontinue" -> ToString[continue]]];

			response = importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
			continue = Lookup[Lookup[response, "continue", {}], "rvcontinue", ""];
			response = Lookup[Lookup[response, "query", {}], "pages", {}];
			response = {
				"Date" -> DateObject[#["timestamp"], TimeZone -> 0], (*TODO: should we convert it using TimeZoneConvert[] with $TimeZone?*)
				"Text" -> #["content"]
			} & /@ Lookup[First[response], "revisions", {}];
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult, 1]]];
	) & /@ tmpTitles;

	result = Replace[titles, result, {1}];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["SeeAlsoList", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["SeeAlsoRules", id, args];
	result = If[MatchQ[result, _?ListQ], Replace[result, Rule[a_, b_] :> b, {0, -1}], Replace[result, Rule[a_, b_] :> b, {0}]];
	result
]

wikipediacookeddata["SeeAlsoRules", id_, args_] := Block[
	{titles, result},

	titles = WikipediaTitleMapping[id, args];
	result = wikipediacookeddata["ArticleWikicode", id, args];
	If[!MatchQ[result, _?ListQ], result = {result}];

	result = If[MatchQ[#, _?StringQ],
		StringReplace[
			Flatten[
				StringCases[
					StringCases[
						StringReplace[#, "\n" -> ""],
						RegularExpression["(?i)==\\s*see also\\s*==([^=]+)(==|$)"] -> "$1" (*This only works in english. This text depends on the article's language*)
					],
					RegularExpression["\\[\\[([^\\[\\]]+)\\]\\]"] -> "$1"
				],
				1
			],
			RegularExpression["([^|]+)\\|([^|]+)"] -> "$1"
		]
	,
		#
	] & /@ result;

	result = If[MatchQ[titles[[#]], _Missing], titles[[#]], Rule[titles[[#]], result[[#]]]] & /@ Range[1, Length[titles]];

	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["SummaryPlaintext", id_, args_] := Block[
	{language, targetLanguage, result, titles, buffer, shortLanguage, shortTargetLanguage},

	result = wikipediacookeddata["ArticlePlaintext", id, args];

	If[!MatchQ[result, _List], result = List[result]];

	result = If[MatchQ[#, _String],
		buffer = Flatten[StringSplit[#, RegularExpression["==[^=]+=="]]];
		If[Length[buffer] > 0, StringTrim@First[buffer], #]
	,
		#
	] & /@ result;

	If[Length[result] === 1,
		result = First[result]
	];

	result
]

wikipediacookeddata["SummaryWikicode", id_, args_] := Block[
	{invalidParameters, result, titles, tmpTitles, strTitles, inputTitles, section, response, posNonMissingTitles, continue, firstIteration, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"PageID", "Section", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	section = Lookup[args, "Section", 0];
	If[!MatchQ[section, _Integer],
		Message[ServiceExecute::nval, "Section", "Wikipedia"];
		Throw[$Failed]
	];
	section = ToString[section];

	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];
	tmpTitles = Partition[tmpTitles, UpTo[50]];

	result = {};
	If[tmpTitles =!= {},
		(
			firstIteration = True;
			strTitles = StringJoin[Riffle[#, "|"]];
			continue = "";

			While[firstIteration || (!firstIteration && continue =!= ""),
				response = WikipediaGetSummaryWikicode[id, strTitles, section, language, continue];
				continue = Lookup[Lookup[response, "continue", {}], "rvcontinue", ""];

				response = Cases[
					(#["title"] -> Lookup[First[Lookup[#, "revisions", {{}}]], "content", Missing["NotAvailable"]]) & /@ Lookup[Lookup[response, "query", {}], "pages", {}],
					Rule[a_, b_String] :> Rule[a, b],
					1
				];

				AppendTo[result, response];
				firstIteration = False;
			]
		) & /@ tmpTitles;

		result = Flatten[result];
		result = Replace[titles, result, 1];
	,

		result = titles
	];

	If[Length[result] === 1, result = First[result]];
	result
]

(*Finds the wikipedia title of an Entity or PageID, also validates the existence of a title*)
wikipediacookeddata["Title", id_, args_] := Block[
	{invalidParameters, result},

	invalidParameters = Select[Keys[args], !MemberQ[{"PageID", "Title", Language}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	result = WikipediaTitleMapping[id, args];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["TitleSearch", id_, args_] := Block[
	{invalidParameters, result, response, search, limit, elementsLeft, firstIteration, continue, tmpResult, tmpSize, parameters, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "Search", Language, MaxItems, Method}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	If[!KeyExistsQ[args, "Search"],
		Message[ServiceExecute::nparam, "Search"];
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	search = Lookup[args, "Search"];
	If[!MatchQ[search, _?ListQ], search = {search}];
	If[!(MatchQ[search, {__?StringQ}] && !MemberQ[search, ""]),
		Message[ServiceExecute::nval, "Search", "Wikipedia"];
		Throw[$Failed]
	];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 50]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[limit, All], limit = Infinity];

	result = {};
	(
		elementsLeft = limit;
		firstIteration = True;
		continue = "";
		tmpResult = {};

		While[firstIteration || (continue =!= "" && elementsLeft > 0),
			firstIteration = False;

			tmpSize = Min[elementsLeft, WikipediaPageSize];
			parameters = {
				"pathlanguage" -> language,
				"action" -> "query",
				"list" -> "search",
				"srwhat" -> "text",
				"continue" -> "",
				"format" -> "json",
				"srsearch" -> "intitle:" <> #,
				"srlimit" -> ToString[tmpSize]
			};

			If[continue =!= "", AppendTo[parameters, "sroffset" -> ToString[continue]]];

			response = importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
			continue = Lookup[Lookup[response, "continue", {}], "sroffset", ""];
			response = Lookup[Lookup[response, "query", {}], "search", ""];
			response = (
				<|
					"Title" -> #["title"],
					"Snippet" -> (StringReplace[Lookup[#, "snippet", {}], {"<span class=\"searchmatch\">" -> "", "</span>" -> ""}])
				|>
			) & /@ response;
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];

		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ search;

	result = Replace[search, result, 1];
	If[Length[result] === 1, result = First[result]];
	result
]

wikipediacookeddata["TitleTranslationRules", id_, args_] := Block[
	{result},

	result = wikipediacookeddata["TitleTranslations", id, args];
	result = Replace[result, {"Language"->a_,"Translation"->b_} :> Rule[a, b], 2];
	result
]

(*Returns title translations for a given page(s)*)
wikipediacookeddata["TitleTranslations", id_, args_] := Block[
	{invalidParameters, result, response, titles, limit, tmpTitles, posNonMissingTitles, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"MaxItems", "PageID", "Title", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", 500]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0),
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];
	If[MatchQ[limit, All] || limit > 500, limit = 500];
	limit = ToString[limit];

	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];

	response = importresults[
		KeyClient`rawkeydata[id, "RawMainRequest", {
			"pathlanguage" -> language,
			"action" -> "query",
			"prop" -> "langlinks",
			"continue" -> "",
			"format" -> "json",
			"formatversion" -> "2",
			"lllimit" -> limit,
			"llprop" -> "url|langname|autonym",
			"titles" -> #,
			"redirects" -> ""
		}]
	] & /@ tmpTitles;

	response = Rule[
		#["title"],
		{"Language" -> #["langname"], "Translation" -> #["title"]} & /@ Lookup[#, "langlinks", {}]
	] & /@ Flatten[Lookup[Lookup[response, "query", {}], "pages", {}]];

	result = Replace[titles, response, {1}];
	If[Length[result] === 1,
		result = First[result]
	];

	result
]

(*Enumerate recent changes*)
wikipediacookeddata["WikipediaRecentChanges", id_, args_] := Block[
	{invalidParameters, result, date, startDate, endDate, limit, continue, response, posNonMissingTitles, elementsLeft, tmpSize, firstIteration, parameters, language},

	invalidParameters = Select[Keys[args], !MemberQ[{"StartDate","EndDate","Date", "MaxItems", Language, MaxItems}, #] &];
	If[Length[invalidParameters] > 0,
		Message[ServiceObject::noget, #, "Wikipedia"] & /@ invalidParameters;
		Throw[$Failed]
	];

	language = WikipediaValidateLanguage[args];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
	If[!(MatchQ[limit, _Integer] && limit > 0) && !MatchQ[limit, All],
		Message[ServiceExecute::nval, MaxItems, "Wikipedia"];
		Throw[$Failed]
	];

	If[MatchQ[limit, All], limit = Infinity];
	
	startDate = Lookup[args, "StartDate", Now];
	If[!MatchQ[startDate, _?DateObjectQ],
		Message[ServiceExecute::nval, "StartDate", "Wikipedia"];
		Throw[$Failed]
	];
	startDate = DateString[startDate, "ISODateTime", TimeZone -> 0] <> "Z";

	endDate = Lookup[args, "EndDate", DateObject[{2000, 1, 1}]];
	If[!MatchQ[endDate, _?DateObjectQ],
		Message[ServiceExecute::nval, "EndDate", "Wikipedia"];
		Throw[$Failed]
	];
	endDate = DateString[endDate, "ISODateTime", TimeZone -> 0] <> "Z";

	result = {};
	elementsLeft = limit;
	firstIteration = True;
	continue = "";
	
	While[firstIteration || (continue =!= "" && elementsLeft > 0),
		firstIteration = False;

		tmpSize = Min[elementsLeft, WikipediaPageSize];
		parameters={
			"pathlanguage" -> language,
			"action" -> "query",
			"list" -> "recentchanges",
			"continue" -> "",
			"format" -> "json",
			"formatversion" -> "2",
			"rclimit" -> ToString[tmpSize],
			"rcprop" -> "user|userid|timestamp|title|ids|redirect",
			"rcstart" -> startDate,
			"rcend" -> endDate
		};

		If[continue =!= "", AppendTo[parameters, "rccontinue" -> ToString[continue]]];

		response = importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
		continue = Lookup[Lookup[response, "continue", {}], "rccontinue", ""];
		response = Lookup[Lookup[response, "query", {}], "recentchanges", {}];
		response = {
			"OldRevisionID" -> #["old_revid"],
			"Namespace" -> #["ns"],
			"Type" -> #["type"],
			"Title" -> #["title"],
			"RecentChangeID" -> #["rcid"],
			"PageID" -> #["pageid"],
			"User" -> #["user"],
			"RevisionID" -> #["revid"],
			"Anonymous" -> Lookup[#, "anon", False],
			"UserID" -> #["userid"],
			"Date" -> DateObject[#["timestamp"], TimeZone -> 0] (*TODO: should we convert it using TimeZoneConvert[] with $TimeZone?*)
		} & /@ response;
		elementsLeft = elementsLeft - tmpSize;
		AppendTo[result, response]
	];

	Flatten[result, 1]
]

(****************************************************************************************************)

(*Experimental*)

solveTags[rawTable_List] := Block[
	{result},

	result = (StringReplace[#, {RegularExpression["<img (.*)src=\"([^\"]*)\"(.*)/>"] -> "$2"}] &) /@ rawTable;
	result = (StringReplace[#, {RegularExpression["<sup( [^>]+)*>([^<^>]*)</sup>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<span( [^>]+)*>([^<^>]*)</span>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<div( [^>]+)*>([^<^>]*)</div>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<code( [^>]+)*>([^<^>]*)</code>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<caption( [^>]+)*>([^<^>]*)</caption>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<small( [^>]*)*>([^<^>]*)</small>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<strong( [^>]*)*>([^<^>]*)</strong>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<a( [^>]*)*>([^<^>]*)</a>"] -> "$2"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<abbr( [^>]*)*>([^<^>]*)</abbr>"] -> "$2"}] &) /@ result;

	result = (StringReplace[#, {RegularExpression["<li( [^>]+)*>([^<^>]*)</li>"] -> "|$2|"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<ul( [^>]+)*>([^<^>]*)</ul>"] -> "{|$2|}"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<pre>([^<^>]*)</pre>"] -> "$1"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<p>([^<^>]*)</p>"] -> "$1"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<i>([^<^>]*)</i>"] -> "$1"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<b>([^<^>]*)</b>"] -> "$1"}] &) /@ result;
	result = (StringReplace[#, {RegularExpression["<h4>([^<^>]*)</h4>"] -> "$1"}] &) /@ result;
	result = (StringReplace[#, {"<br />" -> "\n"}] &) /@ result
]

generateRawTables[html_String] := Block[
	{result},

	result = StringReplace[
		html,
		{
			RegularExpression["<table( [^>]+)*>"] -> "<#table>",
			"</table>" -> "@-table-@<#table>",
			RegularExpression["<tr( [^>]+)*>"] -> "", "</tr>" -> "<#row>",
			RegularExpression["<td( [^>]+)*>"] -> "", "</td>" -> "<#div>",
			RegularExpression["<th( [^>]+)*>"] -> "", "</th>" -> "<#div>"
		}
	];
	result = StringSplit[result, "<#table>"];
	result = StringReplace[Select[result, ! StringFreeQ[#, "@-table-@"] &], "@-table-@" -> ""];
	result = StringSplit[#, "<#row>"] & /@ result;
	result = StringSplit[#, "<#div>"] & /@ result;
	result
]

(*Extracts tables from html code*)
wikipediacookeddata["Tables", id_, args_] := Block[
	{result, titles},

	language = WikipediaValidateLanguage[args];
	titles = WikipediaTitleMapping[id, args];

	result = If[MatchQ[#, _String], StringReplace[URLFetch["https://" <> language <> ".wikipedia.org/wiki/" <> StringReplace[#, " " -> "_"]], {"\n" -> ""}], #] & /@ titles;
	result = If[MatchQ[#, _String], generateRawTables[#], #] & /@ result;
	result = If[MatchQ[#, _List], FixedPoint[solveTags[#] &, #], #] & /@#& /@ result;

	If[Length[result] === 1,
		result = First[result]
	];

	result
]

(****************************************************************************************************)

wikipediacookeddata[___] := $Failed

wikipediasendmessage[___]:= $Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{Wikipedia`Private`wikipediadata,Wikipedia`Private`Wikipediacookeddata,Wikipedia`Private`wikipediasendmessage,Wikipedia`Private`wikipediarawdata}
