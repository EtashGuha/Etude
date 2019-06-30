
BeginPackage["CompileUtilities`Asserter`Common`"]

ToCompactLinearSyntax

SimpleTemplateQ

SimpleTemplate


FormatMessage




Begin["`Private`"]

ToCompactLinearSyntax[expr_] := ToString[compact @ expr, StandardForm];


SimpleTemplateQ[s_String] := And[
	And @@ StringMatchQ[
		StringCases[s, "`" ~~ Shortest[t___] ~~ "`" :> t], 
		DigitCharacter...]
	,
	StringFreeQ[s, "<*"],
	StringFreeQ[s, "\n"]
];

SimpleTemplate[s_String, args_] := With[
	{dummies = Table["$$" <> IntegerString[i] <> "$$", {i, Length[args]}]},
	StringReplace[
		ToString @ StringForm[s, Sequence @@ dummies],
		Thread[dummies -> args]
	]
];

FormatMessage[msg_String, {}] := msg;
FormatMessage[msg_String ? SimpleTemplateQ, args_] :=
        SimpleTemplate[msg, Map[ToCompactLinearSyntax, args]];
FormatMessage[msg_String, args_] := TemplateApply[msg, args];
FormatMessage[HoldPattern[to_TemplateObject] (* avoid autoloading *), args_] := TemplateApply[to, args];
FormatMessage[msg_MessageName, args_] := Module[
        {genmsg = MessageName[General, Last @ msg]},
        If[StringQ[genmsg], FormatMessage[genmsg, args],
                ToString @ StringForm["Undefined message `` with arguments ``", msg, args]]];
FormatMessage[msg_, args_] := TemplateApply["Invalid message `` with arguments ``", {msg, args}];

End[]

EndPackage[]
