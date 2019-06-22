Package["Macros`"]

PackageExport["SetArgumentCount"]

SetArgumentCount[symbol_, min_Integer | {min_Integer}] := SetArgumentCount[symbol, min, min];

(* legacy usage: *)
SetArgumentCount[symbol_, {min_Integer, max_}] := SetArgumentCount[symbol, min, max];

SetArgumentCount[symbol_Symbol, min_Integer, max:(_Integer | Infinity)] := (
	symbol /: Internal`ArgumentCountRegistry[symbol] = {min, max};
	System`Private`LHS_symbol := RuleCondition[Developer`CheckArgumentCount[System`Private`LHS, min, max]; System`Fail];
);

e_SetArgumentCount := Message[SetArgumentCount::args, HoldForm[e]];

If[DownValues[Developer`CheckArgumentCount] === {},

(* this is actualy duplicated in PacletManager, but I want to guarentee there won't be any version skew issues,
so I'm duplicating it here... *)
SetAttributes[Developer`CheckArgumentCount, HoldFirst];
Developer`$PossibleRulePattern = (Rule|RuleDelayed)[_String|_Symbol, _];
Developer`CheckArgumentCount[head_Symbol[args___], min_, max_] := With[
	{hcargs = Hold[args]},
	ArgumentCountQ[head, Length @ If[Options[head] === {}, hcargs, 
		Replace[hcargs, _[a___, Developer`$PossibleRulePattern..] :> Hold[a]]], min, max]];
];