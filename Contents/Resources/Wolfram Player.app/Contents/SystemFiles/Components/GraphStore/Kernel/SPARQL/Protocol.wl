BeginPackage["GraphStore`SPARQL`Protocol`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`Formats`"];

Begin["`Private`"];

SPARQLExecute[args___] := With[{res = Catch[iSPARQLExecute[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iSPARQLExecute];
iSPARQLExecute[service : (IRI | URL)[_?StringQ] | _?StringQ, query_] := Module[{
	res,
	sparqlExpr,
	post
},
	{sparqlExpr, post} = If[StringQ[query],
		{importSPARQLString[query], Identity},
		split[ToSPARQL[query]]
	];
	(* URLExecute ignores "ContentType" *)
	(* https://bugs.wolfram.com/show?number=348973 *)
	res = With[
		{response = URLRead[HTTPRequest[
			normalizeURL[service],
			<|
				"Headers" -> acceptRules[sparqlExpr],
				"Method" -> "POST",
				"Body" -> {queryRule[sparqlExpr, exportSPARQLString[sparqlExpr]]}
			|>
		]]},
		If[FailureQ[response],
			fail[];
		];
		ImportByteArray[
			response["BodyByteArray"],
			MediaTypeToFormat[response["ContentType"]]
		]
	];
	res = post[res];
	res
];
iSPARQLExecute[store_, query_] := Replace[
	If[StringQ[query], importSPARQLString[query], query][store],
	_[store] :> fail[]
];


clear[split];
split[sparql_?possibleSPARQLQ] := {sparql, Identity};
split[comp : _RightComposition | _Composition] := comp //. c_Composition :> RightComposition @@ Reverse[c] // Replace[{
	RightComposition[sparql : Longest[__]?possibleSPARQLQ, post___] :> {RightComposition[sparql], RightComposition[post]},
	_ :> fail[]
}];

clear[possibleSPARQLQ];
possibleSPARQLQ[x_] := PossibleQueryQ[x] || PossibleUpdateQ[x];

clear[exportSPARQLString];
exportSPARQLString[x_?StringQ] := x;
exportSPARQLString[query_] := Quiet[
	ExportString[query, "SPARQLQuery"],
	{Export::fmterr}
] // Replace[_?FailureQ :> Quiet[
	ExportString[query, "SPARQLUpdate"],
	{Export::fmterr}
]] // Replace[_?FailureQ :> fail[]];

clear[importSPARQLString];
importSPARQLString[query_?StringQ] := Quiet[
	ImportString[query, "SPARQLQuery"],
	{Import::fmterr}
] // Replace[_?FailureQ :> Quiet[
	ImportString[query, "SPARQLUpdate"],
	{Import::fmterr}
]] // Replace[_?FailureQ :> fail[]];

clear[normalizeURL];
normalizeURL[url_?StringQ] := url;
normalizeURL[(IRI | URL)[url_?StringQ]] := url;

clear[queryRule];
queryRule[_?PossibleQueryQ, queryStr_String] := "query" -> queryStr;
queryRule[_?PossibleUpdateQ, updateStr_String] := "update" -> updateStr;

clear[acceptRules];
acceptRules[_SPARQLAsk | _SPARQLSelect] := {"Accept" -> "application/sparql-results+json, application/sparql-results+xml"};
acceptRules[_SPARQLConstruct] := {"Accept" -> $RDFMediaTypes};
acceptRules[_] := {};

End[];
EndPackage[];
