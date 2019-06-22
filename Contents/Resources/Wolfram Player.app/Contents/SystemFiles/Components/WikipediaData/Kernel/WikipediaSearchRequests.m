(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["WikipediaData`"]
(* Exported symbols added here with SymbolName::usage *)  

System`WikipediaData

WikipediaSearch::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";
WikipediaSearch::unreach = "Unable to reach Wikipedia server. Please check your internet connection.";
WikipediaSearch::invinp0 = "Invalid input.";
WikipediaSearch::invinps = "`1` is not a valid input.";
WikipediaSearch::invinpm = "`1` are not valid inputs.";

Begin["`Private`"] (* Begin Private Context *) 

Unprotect[WikipediaSearch]

Options[WikipediaSearch] = {
	Language :> $Language,
	MaxItems -> Automatic,
	Method -> Automatic
}

WikipediaSearch[___] := (Message[WikipediaSearch::offline]; $Failed) /; (!PacletManager`$AllowInternet)

WikipediaSearch[args___] := Module[
	{res = Catch[wikipediasearch0[args]]},

	res /; !FailureQ[res]
]

wikipediasearch0[args___] := Block[
	{Wikipedia`Private`$requestHead = Symbol["WikipediaSearch"]}, 

	wikipediasearch[args]
]

wikipediasearch["Category" -> category_, opt : OptionsPattern[WikipediaSearch]] := wikipedia["CategorySearch", {"Search" -> If[ListQ[category], StringRiffle[category], category], opt}];

wikipediasearch[search_String, opt : OptionsPattern[WikipediaSearch]] := wikipediasearch["Title" -> search, opt]

wikipediasearch[search_List, opt : OptionsPattern[WikipediaSearch]] := wikipediasearch["Title" -> search, opt]

wikipediasearch[s_String | s_List | ("Title" -> (s_String | s_List)), "Dataset", opt: OptionsPattern[WikipediaSearch]] := Block[
	{result},

	result = wikipedia["ArticleOpenSearch", {"Search" -> s, opt}];
	If[ListQ[result],
		Dataset[result]
	,
		result
	]
]

wikipediasearch["Title" -> s_, opt : OptionsPattern[WikipediaSearch]] := Block[
	{result, limit, method, search = s},

	method = OptionValue[Method] /. Automatic -> "OpenSearch";
	If[ListQ[search], search = StringRiffle[search]];

	If[!MemberQ[{"OpenSearch", "MostLikely"}, method], Throw[$Failed]];
	
	Which[
		SameQ[method, "OpenSearch"],
			result = wikipedia["ArticleOpenSearch", {"Search" -> search, opt}]

		,

		SameQ[method, "MostLikely"],
			result = wikipedia["TitleSearch", {"Search" -> search, opt}]
	];

	result
];

wikipediasearch["Content" -> search_, opt : OptionsPattern[WikipediaSearch]] := Lookup[wikipedia["ContentSearch", {"Content" -> If[ListQ[search], StringRiffle[search], search], opt}], "Title", {}];

wikipediasearch["GeoLocation" -> title_String, opt___] := wikipedia["GeoNearbyArticles", {"Title" -> title, opt}];

wikipediasearch["GeoLocation" -> position_, opt___] := wikipedia["GeoNearbyArticles", {GeoPosition -> position, opt}];

wikipediasearch[geodisk_GeoDisk, opt___] := wikipedia["GeoNearbyArticles", {GeoDisk -> geodisk, opt}];

wikipediasearch[params___] := Module[
	{list = List[params]},

	Switch[Length[list],
		0,
			Message[WikipediaSearch::invinp0];
		,

		1,
			Message[WikipediaSearch::invinps, First[list]];
		,

		_,

			Message[WikipediaSearch::invinpm, StringRiffle[list, ", "]];
	];
	
	Throw[$Failed]
]

Protect[WikipediaSearch]
End[] (* End Private Context *)

EndPackage[]