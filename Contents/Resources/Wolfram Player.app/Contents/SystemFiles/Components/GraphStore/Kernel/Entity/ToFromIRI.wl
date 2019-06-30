BeginPackage["GraphStore`Entity`ToFromIRI`", {"GraphStore`", "GraphStore`Entity`"}];
Begin["`Private`"];

EntityFromIRI[args___] := With[{res = Catch[iEntityFromIRI[args], $failTag]}, res /; res =!= $failTag];
EntityToIRI[args___] := With[{res = Catch[iEntityToIRI[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[wa];
wa[s_String] := "http://www.wolframalpha.com/input/?i=" <> s;


clear[iEntityFromIRI];
iEntityFromIRI[i : IRI[s_String]] := First[StringReplace[s, {
	StringExpression[
		StartOfString,
		wa[""],
		head : "Entity" | "EntityClass" | "EntityProperty" | "EntityPropertyClass",
		URLEncode["[\""],
		type___,
		URLEncode["\",\""],
		name___,
		URLEncode["\"]"],
		EndOfString
	] :> ToExpression[head][URLDecode[type], URLDecode[name]],
	___ :> i
}, 1]];


clear[iEntityToIRI];
iEntityToIRI[HoldPattern[(head : Entity | EntityClass | EntityProperty | EntityPropertyClass)[type_, name_]]] := IRI[wa[URLEncode[StringJoin[
	ToString[head],
	"[",
	ToString[type, InputForm],
	",",
	ToString[name, InputForm],
	"]"
]]]];

End[];
EndPackage[];
