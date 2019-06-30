Package["Macros`"]

PackageExport["$FailRHS"]
PackageExport["ConditionalRHS"]
PackageExport["UnevaluatedLHS"]

ClearAll[ConditionalRHS, UnevaluatedLHS, $FailRHS, RewriteConditionalRHS, SetupConditionalDownValue];

ConditionalRHS /: sd:SetDelayed[_, _ConditionalRHS] := RewriteConditionalRHS[sd];

SetAttributes[RewriteConditionalRHS, HoldAllComplete];

RewriteConditionalRHS[SetDelayed[lhs:head_Symbol[___], c:ConditionalRHS[checks__, body_]]] :=
Module[{held, tests, msgspecs, pairs, bodyhc},
	held = HoldComplete[checks];
	If[!EvenQ[Length[held]], 
		Message[ConditionalRHS::args, HoldForm[c]];
		Return[$Failed];
	];
	pairs = Partition[held, 2];
	tests = pairs[[All, 1]];
	msgspecs = Replace[pairs[[All, 2]], s_String :> {s}, {1}];
	bodyhc = HoldComplete[body];
	SetDelayed @@ SetupConditionalDownValue[head, HoldComplete[lhs], tests, msgspecs, bodyhc]
];

RewriteConditionalRHS[SetDelayed[_, c_ConditionalRHS]] :=
	(Message[ConditionalRHS::args, HoldForm[c]]; $Failed);

SetupConditionalDownValue[head_, HoldComplete[lhs_], HoldComplete[tests___], mc:HoldComplete[msgs___], b:HoldComplete[body_]] :=
	If[FreeQ[mc, UnevaluatedLHS], 
	HoldComplete[lhs, Internal`ConditionalValueBody[head, {tests}, {msgs}, body]],
	HoldComplete[System`Private`LHS:lhs, Internal`ConditionalValueBody[System`Private`LHS, {tests}, {msgs}, body]] /. 
		HoldForm[UnevaluatedLHS] -> Internal`ConditionalValueLHS
] /. HoldPattern[$FailRHS] -> Fail;

$FailRHS := Fail;