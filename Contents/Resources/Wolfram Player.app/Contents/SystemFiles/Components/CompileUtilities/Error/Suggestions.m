
BeginPackage["CompileUtilities`Error`Suggestions`"]

Suggestions
SuggestionsString

Begin["`Private`"]

Needs["CompileUtilities`Error`Exceptions`"]

$limit = 4

nf[possibilities_] :=
	nf[possibilities] = Nearest[possibilities -> "Element", DistanceFunction -> (EditDistance[ToString[#1], ToString[#2]] &)];


Suggestions[typo_, {}] :=
	None

Suggestions[typo_, possibilities_List] :=
	Suggestions[typo, possibilities, "Limit" -> $limit] 
Suggestions[typo_, possibilities_List, "Limit" -> limit_] :=
	nf[possibilities][typo, limit]

Suggestions[args___] :=
	ThrowException[{"Unrecognized call to Suggestions", {args}}]


format[ x_String] :=
	x

format[ x_] :=
	ToString[x]
	
SuggestionsString[ typo_, possibilities_List] :=
	Module[{res},
		res = Suggestions[typo, possibilities];
		res = If[ ListQ[res], Map[ format, res], res];
		Which[
			!ListQ[res] || Length[res] === 0,
				""
			,
			Length[res] === 1,
				"Suggestion: " <> First[res] <> "."
			,
			True,
				"Suggestions: " <> Insert[ Riffle[res, ", "], "and ", -2] <> "."
		]
	]


SuggestionsString[args___] :=
	ThrowException[{"Unrecognized call to SuggestionsString", {args}}]
 
 
End[]

EndPackage[]
