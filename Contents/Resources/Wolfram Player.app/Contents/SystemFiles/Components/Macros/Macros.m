Package["Macros`"]


PackageScope["InactivateFull"]

SetAttributes[InactivateFull,HoldFirst];
InactivateFull[x_] := 
	Inactivate @@ ReplaceAll[Hold[x], 
		s_Symbol ? ValueQ :> Inactive[s]
	];


PackageScope["InactiveSymbol"]

InactiveSymbol[str_String] :=
	ToExpression[str, InputForm, Inactive];


PackageScope["DeclareMacro"]
PackageScope["$MacroHead"]

DeclareMacro[sym_Symbol, func_] := 
	TagSetDelayed[sym,
		Verbatim[SetDelayed][lhs_, Verbatim[sym][args___]],
		Activate[
			Inactive[SetDelayed][
				Inactive[lhs], 
				ParseInactives[
					Block[{$MacroHead = gethead[lhs]},
						func @@ InactivateFull[{args}]]
				]
			]
		]
	];


PackageScope["ParseInactives"]

ParseInactives[expr_] :=
	ReplaceAll[
		expr,
		s_Symbol /; StringMatchQ[SymbolName[Unevaluated[s]], "$" ~~ ___ ~~ "$"] :> 
			ToExpression[StringTake[SymbolName[s], {2, -2}], InputForm, Inactive]
	];
	
	
SetAttributes[gethead, HoldAll];
gethead[head_[___]] := gethead[head];
gethead[head_Symbol] := head;
gethead[_] := None;