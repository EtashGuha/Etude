BeginPackage["Compile`AST`Transform`Alpha`"]

MExprSubstitute;

Begin["`Private`"] 

Needs["CompileAST`Class`Base`"]
Needs["CompileUtilities`Asserter`Expect`"]
Needs["CompileAST`Utilities`Set`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["CompileAST`Create`Fresh`"]



MExprSubstitute[mexpr_, assoc_?AssociationQ] :=
	MExprSubstitute[mexpr, Normal[assoc]]
	
(** This has terrible performance implications *)
MExprSubstitute[mexpr0_, lst_?ListQ] :=
	Module[{mexpr = mexpr0},
		Do[
			mexpr = MExprSubstitute[mexpr, rule],
			{rule, lst}
		];
		mexpr
	]

MExprSubstitute[mexpr_, rhs0_ -> lhs0_] :=
	With[
		{
			rhs = CoerceMExpr[rhs0],
			lhs = CoerceMExpr[lhs0]
		},
		sub[mexpr, rhs -> lhs]
	]
	

fresh[mexpr_] := (
	ExpectThat["the variable is a symbol", mexpr
		]["named", "mexpr"
		]["satisfies", MExprSymbolQ
	];
	With[{new = mexpr["clone"]},
		new["setName", MExprFreshVariableName[new["name"]]];
		new
	]
)
isScope[mexpr_] := MExprNormalQ[mexpr] && mexpr["isScopingConstruct"]

sub[mexpr_?isScope, y_?MExprSymbolQ -> u_] :=
	Which[
		MExprContainsQ[canberemovedMExprFreeVariables[mexpr], y],
			With[{new = mexpr["clone"]},
				new["setBoundVariables", sub[#, y -> u]& /@ mexpr["boundVariables"]];
				new["setBody", sub[mexpr["body"], y -> u]];
				new
			],
			(*
		(**< all the arguments are not in free variables of u **)
		AllTrue[canberemovedMExprFreeVariables[u], !MExprContainsQ[mexpr["boundVariables"], #]&] &&
		!MExprContainsQ[canberemovedMExprFreeVariables[mexpr["body"]], u],
			With[{new = mexpr["clone"]},
				new["setBoundVariables", sub[#, y -> u]& /@ mexpr["boundVariables"]];
				new["setBody", sub[mexpr["body"], y -> u]];
				new
			],
			*)
		True,
			(**< See FUNCTIONAL PEARLS - Alpha-Conversion is Easy *)
			(**< This technique is standard in all capture avoiding substitutions *)
			(**< Replace the arguments with a new fresh name and then apply the subsitution *)
			(**< (\x.E) [ y -> u] = (\x'.E) [x -> x'][y -> u] where x' is not free in u nor E *)
			With[{
					new = mexpr["clone"],
					vars = mexpr["boundVariables"],
					freshVars = fresh /@ mexpr["boundVariables"]
				 },
				new["setBoundVariables", freshVars];
				Do[ (**< Todo: This is very inefficient *)
					new["setBody", sub[new["body"], vars[[ii]] -> freshVars[[ii]]]],
					{ii, Length[vars]}
				];
				new["setBody", sub[new["body"], y -> u]];
				new
			]
	]

sub[mexpr_?MExprNormalQ, rhs_ -> lhs_] :=
	With[{new = mexpr["clone"]},
		new["setHead", sub[mexpr["head"], rhs -> lhs]];
		new["setArguments", sub[#, rhs -> lhs]& /@ mexpr["arguments"]];
		new
	]
	
sub[mexpr_?MExprSymbolQ, rhs_?MExprSymbolQ -> lhs_] :=
	If[mexpr["sameQ", rhs],
		lhs["clone"],
		mexpr
	]

sub[mexpr_?MExprQ, _] :=
	mexpr
	

End[]

EndPackage[]
