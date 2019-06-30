(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["WikipediaData`"]
(* Exported symbols added here with SymbolName::usage *)  

System`WikipediaData

WikipediaData::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";
WikipediaData::unreach = "Unable to reach Wikipedia server. Please check your internet connection.";
WikipediaData::invinp0 = "Invalid input.";
WikipediaData::invinps = "`1` is not a valid input.";
WikipediaData::invinpm = "`1` are not valid inputs.";
WikipediaData::nocat = "`1` is only a property when a Category, not an article is specified.";

Begin["`Private`"] (* Begin Private Context *) 

Unprotect[WikipediaData]

Options[WikipediaData] = {
	"MaxLevel" -> Automatic,
	"MaxLevelItems" -> Automatic,
	GeoDisk -> Automatic,
	GeoDistance -> Automatic,
	Language :> $Language,
	MaxItems -> Automatic
}

titlePattern = ((_String?StringQ | Entity[__]) | {(_String?StringQ | Entity[__])..});

$wikisite = URL["http://www.wikipedia.com"];

WikipediaData[___] := (Message[WikipediaData::offline]; $Failed) /; (!PacletManager`$AllowInternet)

WikipediaData[args___] := Module[
	{res = Catch[wikipediadata0[args]]},
	res /; !FailureQ[res]
]

wikipediadata0[args___] := Block[
	{Wikipedia`Private`$requestHead = Symbol["WikipediaData"]}, 
	wikipediadata[args]
]

wikipediadata[title: titlePattern /; !MatchQ[title, "GeoNearbyArticles" | "GeoNearbyDataset" | "RandomArticle" | "WikipediaRecentChanges"], opt : OptionsPattern[]] := wikipediadata[title, "ArticlePlaintext", opt];

wikipediadata[title: titlePattern, request: "ArticlePlaintext" | "SummaryPlaintext", opt : OptionsPattern[]] := Module[
	{},
	wikipedia[request, {"Title" -> title, opt}]
];

wikipediadata[HoldPattern[Rule["PageID", pageid_]], opt : OptionsPattern[]] := wikipediadata[Rule["PageID", pageid], "ArticlePlaintext", opt];

wikipediadata[HoldPattern[Rule["PageID", pageid_]], request: "ArticlePlaintext" | "SummaryPlaintext", opt : OptionsPattern[]] := Module[
	{},
	wikipedia[request, {"PageID" -> pageid, opt}]
];

wikipediadata[HoldPattern[Rule["Category", category_String | category_List]], opt : OptionsPattern[]] := wikipedia["CategoryArticles", {"Category" -> category, opt}];

wikipediadata[HoldPattern[Rule["Title", title_]], "ArticlePlaintext", opt : OptionsPattern[]] := wikipediadata[title, "ArticlePlaintext", opt];

wikipediadata[params___] := Module[
	{list = List[params]},
	
	Switch[Length[list],
		0,
			Message[WikipediaData::invinp0];
		,

		1,
			Message[WikipediaData::invinps, First[list]];
		,

		_,

			Message[WikipediaData::invinpm, StringRiffle[list, ", "]];
	];
	
	Throw[$Failed]
]

wikipediadata[arg_, lis_List] := Module[
	{resul,params,prop, opt},
	prop = First[lis];
	opt = Rest[lis];
	params = Flatten[opt];
	If[MatchQ[prop,"DailyPageHits"|"MonthlyPageHits"],
		resul = wikipediadata[arg, prop, opt];
		resul,
		Message[WikipediaData::invinp0]; Throw[Missing["BadInput"]]
	]	
]

wikipediadata[arg_, "ArticleContributors", opt : OptionsPattern[]] := Module[
	{result, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ArticleContributors", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ArticleContributors", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[search_, "ArticleOpenSearch", opt : OptionsPattern[]] := Module[
	{result},

	If[MatchQ[search, Rule["Search", _String]],
		result = wikipedia["ArticleOpenSearch", {search, opt}],
		result = Missing["BadInput"]
	];
	result
];

(***************************************************************************)

(***************************************************************************)

wikipediadata[arg_, "ArticleWikicode", opt : OptionsPattern[]] := Module[{result, title, pageid},
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ArticleWikicode", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ArticleWikicode", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "BacklinksRules", opt : OptionsPattern[]] := Module[{result, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["BacklinksRules", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["BacklinksRules", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "BacklinksList", opt : OptionsPattern[]] := Module[
	{result, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["BacklinksList", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["BacklinksList", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "CategoryArticles", opt : OptionsPattern[]] := Module[
	{result},

	If[MatchQ[arg, Rule["Category", _String]] || MatchQ[arg, Rule["Category", {__String}]],
		result = wikipedia["CategoryArticles", {arg, opt}],
		Message[WikipediaData::nocat,"CategoryArticles"]; Throw[$Failed]
	];

	result
];

wikipediadata[arg_, "CategoryArticleIDs", opt : OptionsPattern[]] := Module[
	{result},

	If[MatchQ[arg, Rule["Category", _String]] || MatchQ[arg, Rule["Category", {__String}]],
		result = wikipedia["CategoryArticleIDs", {arg, opt}],
		Message[WikipediaData::nocat,"CategoryArticleIDs"]; Throw[$Failed]
	];

	result
];

wikipediadata[arg_, "CategoryLinks", opt : OptionsPattern[]] := Module[
	{result, level},

	If[MatchQ[arg, Rule["Category", _String]] || MatchQ[arg, Rule["Category", {__String}]],
		result = wikipedia["CategoryLinks", {arg, opt}],
		Message[WikipediaData::nocat,"CategoryLinks"]; Throw[$Failed]
	];

	result
];

wikipediadata[arg_, "CategoryMembers", opt : OptionsPattern[]] := Module[
	{result},

	If[MatchQ[arg, Rule["Category", _String]] || MatchQ[arg, Rule["Category", {__String}]],
		result = wikipedia["CategoryMembers", {arg, opt}],
		Message[WikipediaData::nocat,"CategoryMembers"]; Throw[$Failed]
	];

	result
];

wikipediadata[arg_, "CategoryMemberIDs", opt : OptionsPattern[]] := Module[
	{result},
	
	If[MatchQ[arg, Rule["Category", _String]] || MatchQ[arg, Rule["Category", {__String}]],
		result = wikipedia["CategoryMemberIDs", {arg, opt}],
		Message[WikipediaData::nocat,"CategoryMemberIDs"]; Throw[$Failed]
	];

	result
];

wikipediadata[arg_, "CategorySearch", opt : OptionsPattern[]] := Module[
	{result},
	
	If[MatchQ[arg, Rule["Search", _String]],
		result = wikipedia["CategorySearch", {arg, opt}],
		Message[WikipediaData::nocat,"CategorySearch"]; Throw[$Failed]
	];

	result
];

wikipediadata[arg_, "ContentSearch", opt : OptionsPattern[]] := Module[
	{result},
	
	If[MatchQ[arg, Rule["Content", _String]] || MatchQ[arg, Rule["Content", {__String}]],
		result = wikipedia["ContentSearch", {arg, opt}],
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "ContributorArticles", opt : OptionsPattern[]] := Module[
	{result},
	
	If[MatchQ[arg, Rule["Contributor", _String]],
		result = wikipedia["ContributorArticles", {arg, opt}],
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "DailyPageHits", opt : OptionsPattern[]] := Module[
	{result, title, pageid, params = Flatten[{opt}]},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["PageViewsArticle", Join[{"Title" -> title, "Granularity" -> "daily"}, params]],
		!MatchQ[pageid, _Missing],
		result = wikipedia["PageViewsArticle", Join[{"PageID" -> pageid, "Granularity" -> "daily"}, params]],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "ExternalLinks", opt : OptionsPattern[]] := Module[
	{result, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ExternalLinks", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ExternalLinks", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "GeoPosition", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["GeoPosition", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["GeoPosition", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[title: titlePattern, "GeoNearbyArticles", opt : OptionsPattern[]] := Module[{},
	wikipedia["GeoNearbyArticles", {"Title" -> title, opt}]
];

wikipediadata["GeoNearbyArticles", opt_List] := Module[{},
	wikipedia["GeoNearbyArticles", opt]
];

wikipediadata["GeoNearbyArticles", opt__Rule] := Module[{},
	wikipedia["GeoNearbyArticles", {opt}]
];

wikipediadata[title: titlePattern, "GeoNearbyDataset", {opt__Rule}] := wikipedia["GeoNearbyDataset", {"Title" -> title, opt}];

wikipediadata[title: titlePattern, "GeoNearbyDataset", opt : OptionsPattern[]] := Module[{},
	wikipedia["GeoNearbyDataset", {"Title" -> title, opt}]
];

wikipediadata["GeoNearbyDataset", opt__] := Module[{},
	wikipedia["GeoNearbyDataset", {opt}]
];

(*
wikipediadata[title_String, "GeoNearbyDataset", opt_List] := Module[{options, result},
	result = {};
	options = {"Title" -> title};
	options = Join[options, opt];
	result = wikipedia["GeoNearbyArticles", options];
	If[MatchQ[result, {__}],
		result = Dataset[Association@@@result]
	];
	result
];

wikipediadata["GeoNearbyDataset", opt_List] := Module[{result},
	result = wikipedia["GeoNearbyArticles", opt];
	If[MatchQ[result, {__}],
		result = Dataset[Association@@@result]
	];
	result
];
*)

wikipediadata[arg_, "ImageDataset", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ImageDataset", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ImageDataset", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "ImageList", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ImageList", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ImageList", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "ImageURLs", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ImageURLs", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ImageURLs", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "LanguagesURLRules", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["LanguagesURLRules", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["LanguagesURLRules", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "LanguagesList", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["LanguagesList", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["LanguagesList", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "LanguagesURLs", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["LanguagesURLs", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["LanguagesURLs", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "LinksRules", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["LinksRules", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["LinksRules", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "LinksList", opt : OptionsPattern[]] := Module[
	{result, limit, level, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["LinksList", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["LinksList", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "MonthlyPageHits", opt : OptionsPattern[]] := Module[
	{result, title, pageid, params = Flatten[{opt}]},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["PageViewsArticle", Join[{"Title" -> title, "Granularity" -> "monthly"}, params]],
		!MatchQ[pageid, _Missing],
		result = wikipedia["PageViewsArticle", Join[{"PageID" -> pageid, "Granularity" -> "monthly"}, params]],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "PageID", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["PageID", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["PageID", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];
	
	result
];

wikipediadata[arg_, "ParentCategories", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["ParentCategories", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["ParentCategories", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata["RandomArticle", opt : OptionsPattern[]] := Module[
	{result},
	
	result = wikipedia["RandomArticle", opt]
];

wikipediadata[arg_, "Revisions", opt : OptionsPattern[]] := Module[
	{result, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["Revisions", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["Revisions", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "SeeAlsoList", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["SeeAlsoList", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["SeeAlsoList", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "SeeAlsoRules", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["SeeAlsoRules", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["SeeAlsoRules", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "SummaryWikicode", opt : OptionsPattern[]] := Module[
	{result, section, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["SummaryWikicode", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["SummaryWikicode", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "Tables", opt : OptionsPattern[]] := Module[
	{result, title, pageid},

	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["Tables", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["Tables", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "Title", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];
	
	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["Title", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["Title", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];
	
	result
];

wikipediadata[search_, "TitleSearch", opt : OptionsPattern[]] := Module[
	{result},
	
	If[MatchQ[search, Rule["Search", _String]],
		result = wikipedia["TitleSearch", {search, opt}],
		result = Missing["BadInput"]
	];
	result
];

wikipediadata[arg_, "TitleTranslationRules", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["TitleTranslationRules", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["TitleTranslationRules", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata[arg_, "TitleTranslations", opt : OptionsPattern[]] := Module[
	{result, title, pageid},
	
	title = arg;
	pageid = If[MatchQ[arg, Rule["PageID", _]], "PageID" /. arg, Missing["NotAvailable"]];

	Which[
		MatchQ[pageid, _Missing],
		result = wikipedia["TitleTranslations", {"Title" -> title, opt}],
		!MatchQ[pageid, _Missing],
		result = wikipedia["TitleTranslations", {"PageID" -> pageid, opt}],
		True,
		result = Missing["BadInput"]
	];

	result
];

wikipediadata["WikipediaRecentChanges", opt : OptionsPattern[]] := Module[
	{result},
	
	result = wikipedia["WikipediaRecentChanges", opt]
];

Protect[WikipediaData]
End[] (* End Private Context *)

EndPackage[]
