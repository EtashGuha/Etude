(*** Service specific utilites ****)

BeginPackage["WikipediaFunctions`"];

WikipediaArticleOpenSearch::usage = "";
WikipediaBackLinksTree::usage = "";
WikipediaCategoryExtraction::usage = "";
WikipediaEntityToPageID::usage = "";
WikipediaExtractImageFilename::usage = "";
WikipediaGetArticleContributors::usage = "";
WikipediaGetArticleWikicode::usage = "";
WikipediaBacklinkedArticles::usage = "";
WikipediaGetCategoryArticles::usage = "";
WikipediaGetCategoryMembers::usage = "";
WikipediaGetCategorySearch::usage = "";
WikipediaGetContentSearch::usage = "";
WikipediaGetContributorArticles::usage = "";
WikipediaGetExternalLinks::usage = "";
WikipediaGetPlaintext::usage = "";
WikipediaGetSummaryWikicode::usage = "";
WikipediaLinkedArticles::usage = "";
WikipediaLinksTree::usage = "";
WikipediaNamespaceTranslation::usage = "";
WikipediaRenameAndFormat::usage = "";
WikipediaTitleMapping::usage = "";
WikipediaTitleTranslation::usage = "";
WikipediaValidateLanguage::usage = "";
(*************************************)
WikipediaFormats::usage = "";
WikipediaMetricsAccess = "";
WikipediaMetricsAgent = "";
WikipediaMetricsGranularity = "";
WikipediaPageSize::usage = "";
WikipediaSupportedlangrules::usage = "";

Begin["`Private`"];

(*Search the wiki using the OpenSearch protocol*)
WikipediaArticleOpenSearch[id_, search_, limit_, language_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"limit" -> ToString[limit],
		"action" -> "opensearch",
		"format" -> "json",
		"formatversion" -> "2",
		"search" -> search
	};

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

WikipediaBackLinksTree[id_, title_, limit_, level_, language_] := Block[
	{result, nodes, newNodes, buffer, i},

	nodes = {title};
	newNodes = {};
	result = {};
	Do[
		newNodes = WikipediaBacklinkedArticles[id, #, limit, language] & /@ nodes;
		buffer = Thread[List[nodes, newNodes]];
		newNodes = Union[Flatten[newNodes]];
		buffer = Thread[Rule[#[[2]], #[[1]]]] & /@ buffer;
		AppendTo[result, buffer];
		nodes = newNodes;
	,
		{i, 1, level}
	];

	result = Flatten[result];
	result = Select[result, !MatchQ[#, Rule[a_, a_]] &];
	result
]

WikipediaCategoryExtraction[id_, category_String, limit_, language_] := Block[
	{result, buffer, firstIteration = True, elementsLeft, continue, tmpSize},

	elementsLeft = limit;
	firstIteration = True;
	continue = "";
	result = {};

	While[firstIteration || (continue =!= "" && elementsLeft > 0),
		firstIteration = False;

		tmpSize = Min[elementsLeft, WikipediaPageSize];

		buffer = WikipediaGetCategoryExtraction[id, category, language, continue];
		continue = Lookup[Lookup[buffer, "continue", {}], "cmcontinue", ""];
		buffer = Lookup[Lookup[Lookup[buffer, "query", {}], "categorymembers", {}], "title", {}];

		elementsLeft = elementsLeft - Length[buffer];
		result = Flatten[Append[result, buffer], 1];
	];
	
	result = Take[Union[result], If[MatchQ[limit, All], limit, UpTo[limit]]];
	result = Thread[Rule[category, result]];
	result
]

(*TODO: try to optimize this function*)
WikipediaCategoryExtraction[id_, category_List, limit_, language_] := Block[
	{result},

	result = {};
	If[MatchQ[category, {__String}],
		result = WikipediaCategoryExtraction[id, #, limit, language] & /@ category;
	];

	result
]

WikipediaEntityToPageID[element_] := WikipediaEntityToPageID[{element}]

WikipediaEntityToPageID[elements_List] := Block[
	{},
	Replace[
		elements,
		x_Entity :> Replace[
			Quiet[Check[x["WikipediaEnID"], $Failed]],
			{s_String} :> FromDigits[s],
			{0}
		],
		{1}
	]
]

WikipediaExtractImageFilename[id_, args_] := Block[
	{result, tmpResult, titles, tmpTitles, posNonMissingTitles, firstIteration, limit, elementsLeft, tmpSize, parameters, response, continue, language},

	language = WikipediaValidateLanguage[args];

	titles = WikipediaTitleMapping[id, args];

	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = DeleteDuplicates[Extract[titles, posNonMissingTitles]];

	limit = Lookup[args, MaxItems, Lookup[args, "MaxItems", All]];
	If[StringQ[limit], limit = FromDigits[limit]];
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
			parameters = {
				"pathlanguage" -> language,
				"format" -> "json",
				"formatversion" -> "2",
				"action" -> "query",
				"prop" -> "images",
				"imlimit" -> ToString[tmpSize],
				"continue" -> "",
				"titles" -> #,
				"redirects" -> ""
			};

			If[continue =!= "", AppendTo[parameters, "imcontinue" -> ToString[continue]]];

			response = Wikipedia`Private`importresults[
				KeyClient`rawkeydata[
					id,
					"RawMainRequest",
					parameters
				]
			];

			continue = Lookup[Lookup[response, "continue", {}], "imcontinue", ""];
			response = Lookup[Lookup[response, "query", {}], "pages", {}];
			response = If[MatchQ[response, _List] && Length[response] > 0, Lookup[Lookup[First[response], "images", {}], "title", {}], {}];
			elementsLeft = elementsLeft - tmpSize;
			AppendTo[tmpResult, response]
		];
		
		AppendTo[result, Rule[#, Flatten[tmpResult]]];
	) & /@ tmpTitles;

	result = Replace[titles, result, 2];
	result
]

(*Get the list of logged-in contributors and the count of anonymous contributors to a page*)
WikipediaGetArticleContributors[id_, titles_, limit_, language_, continue_] := Block[
	{parameters},
	
	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"prop" -> "contributors",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"pclimit" -> ToString[limit],
		"titles" -> titles,
		"redirects" -> ""
	};

	If[continue =!= "", AppendTo[parameters, "pccontinue" -> ToString[continue]]];
	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Get revision information - wikicode in this case*)
WikipediaGetArticleWikicode[id_, titles_, language_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"continue" -> "",
		"action" -> "query",
		"format" -> "json",
		"formatversion" -> "2",
		"prop" -> "revisions",
		"rvprop" -> "content",
		"titles" -> titles,
		"redirects" -> ""
	};

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Find all pages that link to the given page*)
WikipediaBacklinkedArticles[id_, title_,limit_, language_] := Block[
	{invalidParameters, result, response, tmpSize, continue, firstIteration = True, elementsLeft, parameters},

	result = {};
	elementsLeft = If[MatchQ[limit, _?IntegerQ], limit, Infinity];
	firstIteration = True;
	continue = "";

	While[firstIteration || (continue =!= "" && elementsLeft > 0),
		firstIteration = False;
		tmpSize = Min[elementsLeft, WikipediaPageSize];
		parameters={
			"pathlanguage" -> language,
			"action" -> "query",
			"list" -> "backlinks",
			"continue" -> "",
			"format" -> "json",
			"formatversion" -> "2",
			"bllimit" -> ToString[tmpSize],
			"blnamespace" -> "0",
			"bltitle" -> title,
			"redirects" -> ""
		};

		If[continue =!= "", AppendTo[parameters, "blcontinue" -> ToString[continue]]];

		response = Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
		continue = Lookup[Lookup[response, "continue", {}], "blcontinue", ""];
		response = Lookup[Lookup[Lookup[response,"query",{}],"backlinks",{}], "title", {} (*Missing["NotAvailable"]*)];

		(*Sometimes Wikipedia returns less results than the amount requested even though there are enough items, so we have to use "continue" to retrieve  the rest.*)
		tmpSize = Min[elementsLeft, Length[response]];
		elementsLeft -= tmpSize;
		response = Take[response, UpTo[tmpSize]];

		AppendTo[result, response]
	];

	Flatten[result]
]

(*List all articles in a given category*)
WikipediaGetCategoryArticles[id_, category_, limit_, language_, continue_] := Block[
	{tmpCategory, parameters}, 
	
	tmpCategory = category;
	If[MatchQ[tmpCategory, _String], tmpCategory = StringReplace[tmpCategory, RegularExpression["(?i)^Category:(.+)"] -> "$1"], Throw[$Failed]];
	
	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"list" -> "categorymembers",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"cmlimit" -> ToString[limit],
		"cmnamespace" -> "0",
		"cmtitle" -> "Category:" <> tmpCategory
	};

	If[continue =!= "", AppendTo[parameters, "cmcontinue" -> ToString[continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

WikipediaGetCategoryExtraction[id_, category_String, language_, continue_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"action" -> "query",
		"list" -> "categorymembers",
		"cmtitle" -> "Category:" <> StringReplace[StringTrim[category], RegularExpression["(?i)^category:(.+)"] -> "$1"],
		"cmsort" -> "timestamp",
		"cmdir" -> "desc",
		"cmlimit" -> "500", (*Wikipedia won't return the requested number of results if we specify a namespace, that's why request the max possible number and then append all the results together.*)
		"cmnamespace" -> "14" (*namespace for "Category"*)
	};

	If[continue =!= "", AppendTo[parameters, Rule["cmcontinue", continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*List all pages in a given category*)
WikipediaGetCategoryMembers[id_, category_, limit_, language_, continue_] := Block[
	{tmpCategory, parameters}, 
	
	tmpCategory = category;
	If[MatchQ[tmpCategory, _String], tmpCategory = StringReplace[tmpCategory, RegularExpression["(?i)^Category:(.+)"] -> "$1"], Throw[$Failed]];

	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"list" -> "categorymembers",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"cmlimit" -> ToString[limit],
		"cmtitle" -> "Category:" <> tmpCategory,
		"cmprop" -> "ids|title|sortkey|sortkeyprefix|type|timestamp",
		"cmsort" -> "timestamp",
		"cmdir" -> "desc"
	};

	If[continue =!= "", AppendTo[parameters, "cmcontinue" -> ToString[continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Enumerate all categories*)
WikipediaGetCategorySearch[id_, search_, limit_, language_, continue_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"list" -> "allcategories",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"acprefix" -> search,
		"acprop" -> "size",
		"aclimit" -> ToString[limit]
	};
	
	If[continue =!= "", AppendTo[parameters, "accontinue" -> ToString[continue]]];
		
	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Perform a full content text search*)
WikipediaGetContentSearch[id_, search_, limit_, language_, continue_, exactSearch_] := Block[
	{tmpSearch, parameters},
	
	(*tmpSearch = URLEncode@search;*)
	tmpSearch = search;
	
	If[exactSearch,
		tmpSearch = "\"" <> tmpSearch <> "\""
	];

	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"list" -> "search",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"srlimit" -> ToString[limit],
		"srwhat" -> "text",
		"srsearch" -> tmpSearch
	};

	If[continue =!= "", AppendTo[parameters, "sroffset" -> ToString[continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Lists articles made by a contributor*)
WikipediaGetContributorArticles[id_, contributor_, limit_, language_, continue_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"list" -> "usercontribs",
		"ucprop" -> "ids|title|timestamp|comment|parsedcomment|size|sizediff|flags|tags",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"ucuser" -> contributor,
		"uclimit" -> ToString[limit]
	};

	If[continue =!= "", AppendTo[parameters, "uccontinue" -> ToString[continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Get the list of logged-in contributors and the count of anonymous contributors to a page*)
WikipediaGetExternalLinks[id_, title_, limit_, language_, continue_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"prop" -> "extlinks",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"titles" -> title,
		"redirects" -> "",
		"ellimit" -> ToString[limit]
	};

	If[continue =!= "", AppendTo[parameters, "eloffset" -> ToString[continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Extracts wikipedia articles*)
WikipediaGetPlaintext[id_, titles_, continue_, language_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"action" -> "query",
		"prop" -> "extracts",
		"continue" -> "",
		"format" -> "json",
		"formatversion" -> "2",
		"titles" -> titles,
		"explaintext" -> "",
		"redirects" -> "",
		"excontinue" -> ToString[continue]
	};

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Get revision information - wikicode sections in this case*)
WikipediaGetSummaryWikicode[id_, titles_, section_, language_, continue_] := Block[
	{parameters},

	parameters = {
		"pathlanguage" -> language,
		"continue" -> "",
		"action" -> "query",
		"format" -> "json",
		"formatversion" -> "2",
		"prop" -> "revisions",
		"rvsection" -> section,
		"rvprop" -> "content",
		"titles" -> titles,
		"redirects" -> ""
	};

	If[continue =!= "", AppendTo[parameters, "rvcontinue" -> ToString[continue]]];

	Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]]
]

(*Find all pages that link to the given page*)
WikipediaLinkedArticles[id_, title_,limit_, language_] := Block[
	{invalidParameters, result, response, tmpSize, continue, firstIteration = True, elementsLeft, parameters},

	result = {};
	elementsLeft = If[MatchQ[limit, _?IntegerQ], limit, Infinity];
	firstIteration = True;
	continue = "";

	While[firstIteration || (continue =!= "" && elementsLeft > 0),
		firstIteration = False;
		tmpSize = Min[elementsLeft, WikipediaPageSize];
		parameters={
			"pathlanguage" -> language,
			"action" -> "query",
			"prop" -> "links",
			"plnamespace" -> "0",
			"continue" -> "",
			"format" -> "json",
			"formatversion" -> "2",
			"titles" -> title,
			"pllimit" -> ToString[tmpSize],
			"redirects" -> ""
		};

		If[continue =!= "", AppendTo[parameters, "plcontinue" -> ToString[continue]]];

		response = Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
		continue = Lookup[Lookup[response, "continue", {}], "plcontinue", ""];
		response = Lookup[Flatten[Lookup[Lookup[Lookup[response,"query",{}],"pages",{}],"links",{}]], "title", Missing["NotAvailable"]];

		elementsLeft = elementsLeft - tmpSize;
		AppendTo[result, response]
	];

	Flatten[result]
]

WikipediaLinksTree[id_, title_, limit_, level_, language_]:=Block[
	{result, nodes, newNodes, buffer, i},
	
	nodes = {title};
	newNodes = {};
	result = {};
	Do[
		newNodes = WikipediaLinkedArticles[id, #, limit, language] & /@nodes;
		buffer = Thread[List[nodes, newNodes]];
		newNodes = Union[Flatten[newNodes]];
		buffer = Thread[Rule[#[[1]], #[[2]]]] & /@buffer;
		AppendTo[result, buffer];
		nodes = newNodes;
	,
		{i, 1, level}
	];

	result = Flatten[result];
	result = Select[result, !MatchQ[#, Rule[a_, a_]] &];
	result
]

WikipediaRenameAndFormat[key_, val_] := Block[
	{nameRules = {
		"Timestamp" -> "Date",
		"Userid" -> "UserID",
		"Parsedcomment" -> "ParsedComment",
		"Canonicaltitle" -> "Title",
		"Mediatype" -> "MediaType",
		"Commonmetadata" -> "CommonMetadata",
		"Extmetadata" -> "ExternalMetadata",
		"Archivename" -> "ArchiveName",
		"Bitdepth" -> "BitDepth", 
		"Uploadwarning" -> "UploadWarning"}
	},

	Rule[key /. nameRules,
		
		Switch[key,
			"Timestamp",
				TimeZoneConvert[DateObject[val, TimeZone -> 0], $TimeZone]
			,

			_,
				val
		]
	]
]

WikipediaTitleMapping[id_, args_] := Block[
	{title, pageid, result, replacementRules, entities, strings, language, showID},

	title = Lookup[args, "Title", Missing["Title"]];
	If[StringQ[title],title = Flatten[StringSplit[title, "|"]]];
	pageid = Lookup[args, "PageID", Missing["PageID"]];
	showID = Lookup[args, "ShowID", False];

	If[MatchQ[title, _Missing] && MatchQ[pageid, _Missing],
		Message[WikipediaData::onepar, StringRiffle[{"PageID", "Title"}, ", "], Wikipedia];
		Throw[$Failed]
	];
	
	language = WikipediaValidateLanguage[args, False];

	result = {};
	Which[
		!MatchQ[title, _Missing],
			If[! MatchQ[title, _List], title = {title}];

			If[!(MatchQ[Cases[title, _?StringQ | _?MissingQ | _Entity], title] && !MemberQ[Cases[title, _?StringQ], ""]),
				Message[ServiceExecute::nval, "Title", "Wikipedia"];
				Throw[$Failed]
			];
			
			title = WikipediaEntityToPageID[title];
			
			entities = Select[title, MatchQ[#, _?NumericQ] &];
			strings = Select[title, MatchQ[#, _?StringQ] &];
				
			(*Processing pageIds found in entities*)
			replacementRules = WikipediaValidatePageID[id, entities, Lookup[WikipediaSupportedlangrules, "english", "en"], showID];		
			title = Replace[title, replacementRules, {1}];
			title = Replace[title, _?IntegerQ->Missing["NotAvailable"], {1}];
			
			(*Processing strings*)
			replacementRules = WikipediaValidateTitle[id, strings, language["From"], showID];
			title = Replace[title, replacementRules, {1}];
			result = title
		,

		!MatchQ[pageid, _Missing],
			If[! MatchQ[pageid, _List], pageid = {pageid}];
			pageid = Which[
				StringQ[#],
					If[StringMatchQ[#, RegularExpression["[0-9]+"]],
						FromDigits[#]
					,
						Message[ServiceExecute::nval, "PageID", "Wikipedia"];
						Throw[$Failed]
					]
				,

				IntegerQ[#],
					#
				,

				True,
					Message[ServiceExecute::nval, "PageID", "Wikipedia"];
					Throw[$Failed]
			] & /@ pageid;

			replacementRules = WikipediaValidatePageID[id, pageid, language["From"], showID];
			result = Replace[pageid, replacementRules, {1}];
			result = Replace[result, _?IntegerQ -> Missing["NotAvailable"], {1}]
		,

		True,
			result = Missing["NotAvailable"]
	];

	If[SameQ[language["From"], language["To"]],
		If[AssociationQ[#],
			Lookup[#, If[showID, "PageID", "Title"], Missing["NotAvailable"]]
		,
			#
			] & /@ result
	,
		result = If[AssociationQ[#],
			Lookup[#, "Title", Missing["NotAvailable"]]
		,
			#
		] & /@ result;
		result = WikipediaTitleTranslation[id, result, language["From"], language["To"]];

		If[showID,
			WikipediaTitleMapping[id, {"Title" -> result, Language -> language["To"], "ShowID" -> True}]
		,
			result
		]
	]
]

WikipediaTitleTranslation[id_, titles_List, fromLanguage_, toLanguage_, removeMissing_ : False] := Block[
	{result, response, tmpTitles, posNonMissingTitles},

	posNonMissingTitles = Position[titles, Except[_?MissingQ], {1}, Heads -> False];

	tmpTitles = Extract[titles, posNonMissingTitles];
	tmpTitles = DeleteDuplicates[tmpTitles];
	tmpTitles = Partition[tmpTitles, UpTo[50]];

	response = Wikipedia`Private`importresults[
		KeyClient`rawkeydata[id, "RawMainRequest", {
			"pathlanguage" -> fromLanguage,
			"action" -> "query",
			"prop" -> "langlinks",
			"continue" -> "",
			"format" -> "json",
			"formatversion" -> "2",
			"lllimit" -> "500",
			"lllang" -> toLanguage,
			"titles" -> StringJoin[Riffle[#, "|"]],
			"redirects" -> ""
		}]
	] & /@ tmpTitles;

	response = Rule[#["title"], Lookup[First[Lookup[#, "langlinks", {{}}], "title"], "title", Missing["NotAvailable"]]] & /@ Flatten[Lookup[Lookup[response, "query", {}], "pages", {}], 1];
	Replace[titles, response, {1}]
];

WikipediaValidateLanguageInstance[language_] := Block[
	{tmpLanguage = language},

	If[MatchQ[tmpLanguage, Entity[__]],
		If[!(SameQ[EntityTypeName[tmpLanguage], "Language"] && !MissingQ[tmpLanguage = tmpLanguage["Name"]]),
			Message[ServiceExecute::nval, Language, "Wikipedia"];
			Throw[$Failed]
		]
	];

	If[StringQ[tmpLanguage],
		tmpLanguage = ToLowerCase[tmpLanguage];
		tmpLanguage = Lookup[
			WikipediaSupportedlangrules,
			tmpLanguage,
			If[MemberQ[Values[WikipediaSupportedlangrules], tmpLanguage],
				tmpLanguage
			,
				Message[ServiceExecute::nval, Language, "Wikipedia"];
				Throw[$Failed]
			]
		]
	,
		Message[ServiceExecute::nval, Language, "Wikipedia"];
		Throw[$Failed]
	];

	tmpLanguage
]

WikipediaValidateLanguage[args_, finalLangOnly_ : True] := Block[
	{language, from, to},
	language = Lookup[args, Language, "english"];

	If[MatchQ[language, {Rule[_?StringQ | Entity[___], _?StringQ | Entity[___]]}],
		from = WikipediaValidateLanguageInstance[First[Keys[language]]];
		to = WikipediaValidateLanguageInstance[First[Values[language]]];
	,
		from = to = WikipediaValidateLanguageInstance[language]
	];

	If[finalLangOnly,
		to
	,
		<| "From" -> from, "To" -> to |>
	]
]

WikipediaValidatePageID[id_, ids_List, language_, showID_ : False] := Block[
	{posString, tmpIDs, firstIteration, continue, result, parameters, response},

	tmpIDs = ToString /@ DeleteDuplicates[ids];
	tmpIDs = Partition[tmpIDs, UpTo[50]];

	result = {};
	(
		firstIteration = True;
		continue = "";

		While[firstIteration || (continue =!= ""),
			firstIteration = False;

			parameters = {
				"pathlanguage" -> language,
				"action" -> "query",
				"prop" -> "info",
				"continue" -> "",
				"format" -> "json",
				"formatversion" -> "2",
				"pageids" -> StringJoin[Riffle[#, "|"]],
				"redirects" -> ""
			};

			If[continue =!= "", AppendTo[parameters, "incontinue" -> ToString[continue]]];

			response = Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
			continue = Lookup[Lookup[response, "continue", {}], "incontinue", ""];
			response = Lookup[Lookup[response, "query", {}], "pages", {}];
			response = Rule[#["pageid"], If[Lookup[#, "missing", False], Missing["NotAvailable"], Lookup[#, "title", Missing["NotAvailable"]]]] & /@ response;
			AppendTo[result, response]
		]
	) & /@ tmpIDs;

	result = Flatten[result, 1];
	result = (Rule[#[[1]], <|"Title" -> #[[2]], "PageID" -> If[MissingQ[#[[2]]], #[[2]], #[[1]]]|>]) & /@ result;
	result
]

WikipediaValidateTitle[id_, titles_List, language_, showID_ : False] := Block[
	{posString, tmpTitles, firstIteration, continue, result, parameters, response, normalized, redirects},
	tmpTitles = DeleteDuplicates[titles];
	tmpTitles = sortByMaxURLCharacters[tmpTitles];
	result = {};
	(
		firstIteration = True;
		continue = "";

		While[firstIteration || (continue =!= ""),
			firstIteration = False;

			parameters = {
				"pathlanguage" -> language,
				"action" -> "query",
				"prop" -> "info",
				"continue" -> "",
				"format" -> "json",
				"formatversion" -> "2",
				"titles" -> StringJoin[Riffle[#, "|"]],
				"redirects" -> ""
			};

			If[continue =!= "", AppendTo[parameters, "incontinue" -> ToString[continue]]];
			
			response = Wikipedia`Private`importresults[KeyClient`rawkeydata[id, "RawMainRequest", parameters]];
			continue = Lookup[Lookup[response, "continue", {}], "incontinue", ""];
			normalized = Flatten[Lookup[Lookup[response, "query", {}], "normalized", {}], 1];
			redirects = Flatten[Lookup[Lookup[response, "query", {}], "redirects", {}], 1];
			response = Lookup[Lookup[response, "query", {}], "pages", {}];
			response = Rule[#["title"], If[MemberQ[Keys[#],"missing"|"invalid"], Missing["NotAvailable"], If[showID, ToString[#["pageid"]], #["title"]]]] & /@ response;
			
			(
				response = Replace[response, Rule[redirects[[#]]["to"], v_] :> Rule[redirects[[#]]["from"], v], {1}]
			) & /@ Range[1, Length[redirects]];
			
			(
				response = Replace[response, Rule[normalized[[#]]["to"], v_] :> Rule[normalized[[#]]["from"], v], {1}]
			) & /@ Range[1, Length[normalized]];
			AppendTo[result, response];
		]
	) & /@ tmpTitles;

	result = Flatten[result, 1];
	If[IntegerQ@@#[[2]]&/@result,
		result = (Rule[#[[1]], <|"PageID" -> #[[2]]|>]) & /@ result,
		result = (Rule[#[[1]], <|"Title" -> #[[2]]|>]) & /@ result
	];
	result
]

sortByMaxURLCharacters[x_,max_:2000] := Module[{temp, temp2, resul},
	temp = Select[x,StringLength[#] < max &];
	resul = {};
	While[Length[temp] > 0,
		temp2 = {};
		While[Total[StringLength[temp2]] < max && Length[temp] > 0,
			temp2 = Append[temp2, First[temp]]; 
			temp = Drop[temp, 1];
		];
   		If[Length[temp] == 0,
    		resul = Append[resul, temp2]; 
   			Break,
    		temp = Prepend[temp, Last[temp2]];
    		temp2 = Drop[temp2, -1];
    		resul = Append[resul, temp2];
		];
	];
	resul
]

(*************************************************************************)

WikipediaFormats = {"json", "jsonfm", "php", "phpfm", "xml", "xmlfm"};

WikipediaMetricsAccess = {
	"allaccess" -> "all-access",
	"desktop" -> "desktop",
	"mobileapp" -> "mobile-app",
	"mobileweb" -> "mobile-web"
};

WikipediaMetricsAgent = {
	"allagents" -> "all-agents",
	"user" -> "user",
	"spider" -> "spider",
	"bot" -> "bot"	
};

WikipediaMetricsGranularity = {
	"daily" -> "daily",
	"monthly" -> "monthly"
};

WikipediaNamespaceTranslation = {
	0 -> "(Main/Article)",
	1 -> "Talk",
	2 -> "User",
	3 -> "User talk",
	4 -> "Wikipedia",
	5 -> "Wikipedia talk",
	6 -> "File",
	7 -> "File talk",
	8 -> "MediaWiki",
	9 -> "MediaWiki talk",
	10 -> "Template",
	11 -> "Template talk",
	12 -> "Help",
	13 -> "Help talk",
	14 -> "Category",
	15 -> "Category talk",
	100 -> "Portal",
	101 -> "Portal talk",
	108 -> "Book",
	109 -> "Book talk",
	118 -> "Draft",
	119 -> "Draft talk",
	446 -> "Education Program",
	447 -> "Education Program talk",
	710 -> "TimedText",
	771 -> "TimedText talk",
	828 -> "Module",
	829 -> "Module talk",
	2600 -> "Topic"
};

WikipediaPageSize = 500;

WikipediaSupportedlangrules = {
	"abkhaz" -> "ab",
	"abkhazian" -> "ab",
	"aceh" -> "ace",
	"acehnese" -> "ace",
	"adyghe" -> "ady",
	"afrikaans" -> "af",
	"akan" -> "ak",
	"alemannic" -> "als",
	"german colonia tovar" -> "als",
	"schwyzerdütsch" -> "als",
	"swabian" -> "als",
	"amharic" -> "am",
	"aragonese" -> "an",
	"anglo-saxon" -> "ang",
	"old english" -> "ang",
	"arabic" -> "ar",
	"syriac" -> "arc",
	"egyptian arabic" -> "arz",
	"assamese" -> "as",
	"asturian" -> "ast",
	"avar" -> "av",
	"aymara" -> "ay",
	"central aymara" -> "ay",
	"azerbaijani" -> "az",
	"north azerbaijani" -> "az",
	"south azerbaijani" -> "azb",
	"southern azerbaijani" -> "azb",
	"bashkir" -> "ba",
	"bavarian" -> "bar",
	"samogitian" -> "bat-smg",
	"central_bicolano" -> "bcl",
	"central bicolano" -> "bcl",
	"central bikol" -> "bcl",
	"belarusian" -> "be",
	"belarusian (taraškievica)" -> "be-x-old",
	"bulgarian" -> "bg",
	"bhojpuri" -> "bh",
	"bislama" -> "bi",
	"banjar" -> "bjn",
	"bamanankan" -> "bm",
	"bambara" -> "bm",
	"bengali" -> "bn",
	"central tibetan" -> "bo",
	"tibetan" -> "bo",
	"bishnupriya" -> "bpy",
	"bishnupriya manipuri" -> "bpy",
	"breton" -> "br",
	"bosnian" -> "bs",
	"buginese" -> "bug",
	"bugis" -> "bug",
	"buryat (russia)" -> "bxr",
	"catalan" -> "ca",
	"catalan\[Hyphen]valencian\[Hyphen]balear" -> "ca",
	"chavacano" -> "cbk-zam",
	"zamboanga chavacano" -> "cbk-zam",
	"min dong" -> "cdo",
	"min dong chinese" -> "cdo",
	"chechen" -> "ce",
	"cebuano" -> "ceb",
	"chamorro" -> "ch",
	"cherokee" -> "chr",
	"cheyenne" -> "chy",
	"central kurdish" -> "ckb",
	"sorani kurdish" -> "ckb",
	"corsican" -> "co",
	"cree" -> "cr",
	"severn ojibwa" -> "cr",
	"crimean tatar" -> "crh",
	"crimean turkish" -> "crh",
	"czech" -> "cs",
	"kashubian" -> "csb",
	"slavonic old church" -> "cu",
	"old church slavonic" -> "cu",
	"chuvash" -> "cv",
	"welsh" -> "cy",
	"danish" -> "da",
	"german" -> "de",
	"dimli" -> "diq",
	"zazaki" -> "diq",
	"lower sorbian" -> "dsb",
	"divehi" -> "dv",
	"maldivian" -> "dv",
	"dzongkha" -> "dz",
	"ewe" -> "ee",
	"éwé" -> "ee",
	"greek" -> "el",
	"emilian-romagnol" -> "eml",
	"emiliano\[Hyphen]romagnolo" -> "eml",
	"english" -> "en",
	"esperanto" -> "eo",
	"spanish" -> "es",
	"estonian" -> "et",
	"basque" -> "eu",
	"eastern farsi" -> "fa",
	"persian" -> "fa",
	"fula" -> "ff",
	"fulfulde adamawa" -> "ff",
	"finnish" -> "fi",
	"võro" -> "fiu-vro",
	"fijian" -> "fj",
	"faroese" -> "fo",
	"french" -> "fr",
	"franco\[Hyphen]provençal" -> "frp",
	"franco-provençal/arpitan" -> "frp",
	"north frisian" -> "frr",
	"northern frisian" -> "frr",
	"friulian" -> "fur",
	"west frisian" -> "fy",
	"western frisian" -> "fy",
	"irish" -> "ga",
	"irish gaelic" -> "ga",
	"gagauz" -> "gag",
	"gan chinese" -> "gan",
	"scottish gaelic" -> "gd",
	"galician" -> "gl",
	"gilaki" -> "glk",
	"chiripá" -> "gn",
	"guarani" -> "gn",
	"goan konkani" -> "gom",
	"konkani goanese" -> "gom",
	"gothic" -> "got",
	"gujarati" -> "gu",
	"manx" -> "gv",
	"hausa" -> "ha",
	"hakka" -> "hak",
	"hakka chinese" -> "hak",
	"hawaiian" -> "haw",
	"hebrew" -> "he",
	"hindi" -> "hi",
	"fiji hindi" -> "hif",
	"croatian" -> "hr",
	"upper sorbian" -> "hsb",
	"haitian" -> "ht",
	"haitian creole french" -> "ht",
	"hungarian" -> "hu",
	"armenian" -> "hy",
	"interlingua" -> "ia",
	"indonesian" -> "id",
	"interlingua" -> "ie",
	"interlingue" -> "ie",
	"igbo" -> "ig",
	"inupiak" -> "ik",
	"inupiatun north alaskan" -> "ik",
	"ilocano" -> "ilo",
	"ilokano" -> "ilo",
	"ido" -> "io",
	"icelandic" -> "is",
	"italian" -> "it",
	"inuktitut" -> "iu",
	"inuktitut eastern canadian" -> "iu",
	"japanese" -> "ja",
	"lojban" -> "jbo",
	"javanese" -> "jv",
	"georgian" -> "ka",
	"karakalpak" -> "kaa",
	"kabyle" -> "kab",
	"kabardian" -> "kbd",
	"kongo" -> "kg",
	"koongo" -> "kg",
	"gikuyu" -> "ki",
	"kikuyu" -> "ki",
	"greenlandic" -> "kl",
	"inuktitut greenlandic" -> "kl",
	"central khmer" -> "km",
	"khmer" -> "km",
	"kannada" -> "kn",
	"korean" -> "ko",
	"komi-permyak" -> "koi",
	"karachay-balkar" -> "krc",
	"kashmiri" -> "ks",
	"ripuarian" -> "ksh",
	"kurdish" -> "ku",
	"northern kurdish" -> "ku",
	"komi" -> "kv",
	"komi\[Hyphen]zyrian" -> "kv",
	"cornish" -> "kw",
	"kirghiz" -> "ky",
	"kyrgyz" -> "ky",
	"latin" -> "la",
	"ladino" -> "lad",
	"luxembourgeois" -> "lb",
	"luxembourgish" -> "lb",
	"lak" -> "lbe",
	"lezgi" -> "lez",
	"lezgian" -> "lez",
	"ganda" -> "lg",
	"luganda" -> "lg",
	"limburgisch" -> "li",
	"limburgish" -> "li",
	"ligurian" -> "lij",
	"lombard" -> "lmo",
	"lingala" -> "ln",
	"lao" -> "lo",
	"luri northern" -> "lrc",
	"northern luri" -> "lrc",
	"lithuanian" -> "lt",
	"latgalian" -> "ltg",
	"latvian" -> "lv",
	"maithili" -> "mai",
	"banyumasan" -> "map-bms",
	"moksha" -> "mdf",
	"bushi" -> "mg",
	"malagasy" -> "mg",
	"malagasy antankarana" -> "mg",
	"malagasy bara" -> "mg",
	"malagasy masikoro" -> "mg",
	"malagasy northern betsimisaraka" -> "mg",
	"malagasy plateau" -> "mg",
	"malagasy sakalava" -> "mg",
	"malagasy southern betsimisaraka" -> "mg",
	"malagasy tandroy\[Hyphen]mahafaly" -> "mg",
	"malagasy tanosy" -> "mg",
	"malagasy tsimihety" -> "mg",
	"mari eastern" -> "mhr",
	"meadow mari" -> "mhr",
	"maori" -> "mi",
	"māori" -> "mi",
	"minangkabau" -> "min",
	"macedonian" -> "mk",
	"malayalam" -> "ml",
	"mongolian" -> "mn",
	"mongolian halh" -> "mn",
	"mongolian peripheral" -> "mn",
	"marathi" -> "mr",
	"hill mari" -> "mrj",
	"western mari" -> "mrj",
	"malay" -> "ms",
	"maltese" -> "mt",
	"miranda do douro" -> "mwl",
	"mirandese" -> "mwl",
	"burmese" -> "my",
	"erzya" -> "myv",
	"mazandarani" -> "mzn",
	"mazanderani" -> "mzn",
	"nauruan" -> "na",
	"nāhuatl" -> "nah",
	"neapolitan" -> "nap",
	"low saxon" -> "nds",
	"dutch low saxon" -> "nds-nl",
	"nepali" -> "ne",
	"newar" -> "new",
	"newar / nepal bhasa" -> "new",
	"dutch" -> "nl",
	"norwegian (nynorsk)" -> "nn",
	"nynorsk norwegian" -> "nn",
	"bokmål norwegian" -> "no",
	"norwegian (bokmål)" -> "no",
	"novial" -> "nov",
	"norman" -> "nrm",
	"northern sotho" -> "nso",
	"navajo" -> "nv",
	"chichewa" -> "ny",
	"nyanja" -> "ny",
	"auvergnat" -> "oc",
	"occitan" -> "oc",
	"borana\[Hyphen]arsi\[Hyphen]guji oromo" -> "om",
	"oromo" -> "om",
	"oriya" -> "or",
	"osetin" -> "os",
	"ossetian" -> "os",
	"eastern panjabi" -> "pa",
	"eastern punjabi" -> "pa",
	"pangasinan" -> "pag",
	"kapampangan" -> "pam",
	"pampangan" -> "pam",
	"papiamentu" -> "pap",
	"picard" -> "pcd",
	"german pennsylvania" -> "pdc",
	"pennsylvania german" -> "pdc",
	"palatine german" -> "pfl",
	"pfaelzisch" -> "pfl",
	"bareli palya" -> "pi",
	"pali" -> "pi",
	"norfolk" -> "pih",
	"polish" -> "pl",
	"piedmontese" -> "pms",
	"piemontese" -> "pms",
	"western panjabi" -> "pnb",
	"western punjabi" -> "pnb",
	"pontic" -> "pnt",
	"central pashto" -> "ps",
	"northern pashto" -> "ps",
	"pashto" -> "ps",
	"southern pashto" -> "ps",
	"waneci" -> "ps",
	"portuguese" -> "pt",
	"classical quechua" -> "qu",
	"quechua" -> "qu",
	"romansch" -> "rm",
	"romansh" -> "rm",
	"romani vlax" -> "rmy",
	"vlax romani" -> "rmy",
	"kirundi" -> "rn",
	"rundi" -> "rn",
	"romanian" -> "ro",
	"aromanian" -> "roa-rup",
	"romanian macedo" -> "roa-rup",
	"tarantino" -> "roa-tara",
	"russian" -> "ru",
	"rusyn" -> "rue",
	"kinyarwanda" -> "rw",
	"rwanda" -> "rw",
	"sanskrit" -> "sa",
	"sakha" -> "sah",
	"yakut" -> "sah",
	"sardinian" -> "sc",
	"sardinian campidanese" -> "sc",
	"sardinian logudorese" -> "sc",
	"sicilian" -> "scn",
	"scots" -> "sco",
	"sindhi" -> "sd",
	"north saami" -> "se",
	"northern sami" -> "se",
	"sango" -> "sg",
	"serbo-croatian" -> "sh",
	"sinhala" -> "si",
	"sinhalese" -> "si",
	"simple english" -> "simple", (*TODO: notify this*)
	"slovak" -> "sk",
	"slovene" -> "sl",
	"slovenian" -> "sl",
	"samoan" -> "sm",
	"shona" -> "sn",
	"somali" -> "so",
	"albanian" -> "sq",
	"albanian arbëreshë" -> "sq",
	"albanian arvanitika" -> "sq",
	"albanian gheg" -> "sq",
	"albanian tosk" -> "sq",
	"serbian" -> "sr",
	"sranan" -> "srn",
	"sranan tongo" -> "srn",
	"swati" -> "ss",
	"sesotho" -> "st",
	"sotho southern" -> "st",
	"saterfriesisch" -> "stq",
	"saterland frisian" -> "stq",
	"sunda" -> "su",
	"sundanese" -> "su",
	"swedish" -> "sv",
	"swahili" -> "sw",
	"silesian" -> "szl",
	"tamil" -> "ta",
	"tulu" -> "tcy",
	"telugu" -> "te",
	"telugu(తెలుగు)" -> "te",
	"tetum" -> "tet",
	"tetun" -> "tet",
	"tajik" -> "tg",
	"tajiki" -> "tg",
	"thai" -> "th",
	"tigrigna" -> "ti",
	"tigrinya" -> "ti",
	"turkmen" -> "tk",
	"tagalog" -> "tl",
	"tswana" -> "tn",
	"tongan" -> "to",
	"tok pisin" -> "tpi",
	"turkish" -> "tr",
	"tsonga" -> "ts",
	"tatar" -> "tt",
	"tumbuka" -> "tum",
	"twi" -> "tw",
	"tahitian" -> "ty",
	"tuvan" -> "tyv",
	"tuvin" -> "tyv",
	"udmurt" -> "udm",
	"uyghur" -> "ug",
	"ukrainian" -> "uk",
	"urdu" -> "ur",
	"uzbek" -> "uz",
	"uzbek northern" -> "uz",
	"uzbek southern" -> "uz",
	"venda" -> "ve",
	"venetian" -> "vec",
	"veps" -> "vep",
	"vietnamese" -> "vi",
	"vlaams" -> "vls",
	"west flemish" -> "vls",
	"volapük" -> "vo",
	"walloon" -> "wa",
	"waray" -> "war",
	"wolof" -> "wo",
	"wu" -> "wuu",
	"wu chinese" -> "wuu",
	"kalmyk" -> "xal",
	"kalmyk\[Hyphen]oirat" -> "xal",
	"xhosa" -> "xh",
	"mingrelian" -> "xmf",
	"eastern yiddish" -> "yi",
	"yiddish" -> "yi",
	"yoruba" -> "yo",
	"zhuang" -> "za",
	"zealandic" -> "zea",
	"zeeuws" -> "zea",
	"chinese" -> "zh",
	"mandarin" -> "zh",
	"classical chinese" -> "zh-classical",
	"min nan" -> "zh-min-nan",
	"min nan chinese" -> "zh-min-nan",
	"cantonese" -> "zh-yue",
	"zulu" -> "zu"
};

End[];

EndPackage[];