BeginPackage["NaturalLanguage`Customizations`"]

FormattingBlock

Begin["`Private`"]

SetAttributes[FormattingBlock, HoldAll]

FormattingBlock[expr_] := With[{syms = blockedSymbols},
	Internal`InheritedBlock[syms,
		Unprotect[syms];
		blockedRules;
		
		expr
	]
]

blockedRules := Map[
	Function[sym,
		FormatValues[sym] = {
			RuleDelayed[
				HoldPattern[MakeBoxes[sym[a__], TraditionalForm]], 
				MakeBoxes[custom[sym][a], TraditionalForm]
			]
		}
	],
	blockedSymbols
]

MakeBoxes[custom[Plot][a_, {x_, min_, max_}, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"plot ",
	MakeBoxes[a, TraditionalForm],
	RowBox[{"from ", MakeBoxes[x, TraditionalForm], " = ", MakeBoxes[min, TraditionalForm], " to ", MakeBoxes[max,TraditionalForm]}]
}]

MakeBoxes[custom[ReplaceAll][a_, b_], TraditionalForm] ^:= RowBox[{
	MakeBoxes[a, TraditionalForm],
	" where ",
	commaForm[toEqual @ Unevaluated @ b]
}]

MakeBoxes[custom[Series][expr_, {x_, x0_, n_}, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"series ",
	MakeBoxes[expr, TraditionalForm],
	" at ",
	MakeBoxes[x = x0, TraditionalForm],
	" to order ",
	MakeBoxes[n, TraditionalForm]
}]

MakeBoxes[custom[Minimize][expr_, x_, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"minimize ",
	MakeBoxes[expr, TraditionalForm],
	" for ",
	varForm[x]
}]

MakeBoxes[custom[Minimize][{expr_, conds_}, x_, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"minimize ",
	MakeBoxes[expr, TraditionalForm],
	" where ",
	MakeBoxes[conds, TraditionalForm],
	" for ",
	varForm[x]
}]

MakeBoxes[custom[Maximize][expr_, x_, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"maximize ",
	MakeBoxes[expr, TraditionalForm],
	" for ",
	varForm[x]
}]

MakeBoxes[custom[Maximize][{expr_, conds_}, x_, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"maximize ",
	MakeBoxes[expr, TraditionalForm],
	" where ",
	MakeBoxes[conds, TraditionalForm],
	" for ",
	varForm[x]
}]

MakeBoxes[custom[Solve][expr_, vars_, OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"solve ",
	MakeBoxes[expr, TraditionalForm],
	" for ",
	If[MatchQ[Hold[vars], Hold[_List]],
		commaForm[MakeBoxes[#, TraditionalForm]& /@ vars],
		MakeBoxes[vars, TraditionalForm]
	]
}]

MakeBoxes[custom[Solve][expr_, _, domain_:(Reals|Complexes|Integers|Rationals|Algebraics|Primes), OptionsPattern[]], TraditionalForm] ^:= RowBox[{
	"solve ",
	MakeBoxes[expr, TraditionalForm],
	" over the ",
	ToLowerCase @ SymbolName[domain] 
}]

MakeBoxes[custom[And][a__], TraditionalForm] ^:= RowBox @ BoxForm`MakeInfixForm[And[a], " and ", TraditionalForm]

blockedSymbols = DeleteDuplicates @ FormatValues[NaturalLanguage`Customizations`Private`custom][[All, 1, 1, 1, 0, 1]]

SetAttributes[varForm, HoldAllComplete]

varForm[{a__}] := MakeBoxes[And[a], TraditionalForm]
varForm[a_] := MakeBoxes[a, TraditionalForm]

commaForm[a_] := commaForm[{a}]
commaForm[{a_}] := a
commaForm[{a_, b_}] := RowBox[{a, " and ", b}]
commaForm[{a__, b_}] := RowBox[{RowBox@Riffle[{a}, ","], " and ", b}]

toEqual[(Rule|RuleDelayed)[a__]] := MakeBoxes[Equal[a], TraditionalForm]
toEqual[a_List] := toEqual /@ Unevaluated[a]

End[]

EndPackage[]