(* ::Package:: *)

(* ::Chapter::Closed:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}]


AreaBetweenCurves::usage="Compute the area between two plane curves";


Begin["`Private`"]


(* ::Chapter:: *)
(*main code*)


(* ::Section::Closed:: *)
(*AreaBetweenCurves*)


Options[AreaBetweenCurves] = {Assumptions :> $Assumptions};
SyntaxInformation[AreaBetweenCurves] = {"ArgumentsPattern" -> {{_, _}, {_, _, _}, OptionsPattern[]}};
AreaBetweenCurves::"endpt" = "The endpoint `1` depends on `2`.";

AreaBetweenCurves[args__] := 
	With[{res = iAreaBetweenCurves[args]},
		res /; res =!= $Failed
	]

Options[iAreaBetweenCurves] = Options[AreaBetweenCurves];

iAreaBetweenCurves[expr___] /; !ArgumentCountQ[AreaBetweenCurves, Replace[{expr}, {r___, OptionsPattern[]} :> Length[{r}]], {2}] = $Failed;

iAreaBetweenCurves[_, {x_, xmin_, xmax_}, OptionsPattern[]] /; Internal`DependsOnQ[{xmin, xmax}, x] :=
	( 
		Message[AreaBetweenCurves::"endpt", If[Internal`DependsOnQ[xmin, x], xmin, xmax], x];
		$Failed
	)

iAreaBetweenCurves[{f_, g_}, {x_, xmin_, xmax_}, opts:OptionsPattern[]] :=
	Block[{$MessageList = {}, msgs, assum, integrand, res},
		assum = OptionValue[Assumptions] && iAssum[f-g, {x, xmin, xmax}];
		integrand = absDiff[{f, g}, {x, xmin, xmax}];
		
		Quiet[
			res = Integrate[integrand, {x, xmin, xmax}, Assumptions -> assum];
			msgs = $MessageList
		];
		
		If[!FreeQ[msgs, Unevaluated[Integrate::idiv]], 
			res = \[Infinity]
		];
		
		res /; FreeQ[res, Integrate|ComplexInfinity]
	]

iAreaBetweenCurves[___] = $Failed;

iAssum[expr_, {x_, xmin_, xmax_}] := 
	With[{vars = Reduce`FreeVariables[expr, "All", False]},
		(!ListQ[vars] || vars \[Element] Reals) && xmin < x < xmax
	]


(* ::Section::Closed:: *)
(*absDiff*)


absDiff[{f_, g_}, {x_, xmin_, xmax_}] :=
	Block[{sgn, res},
		sgn = Refine[Sign[f-g], xmin < x < xmax];
		
		res = Switch[sgn,
			1|0, f-g,
			-1, g-f,
			_, $Failed
		];
		
		res /; res =!= $Failed
	]

absDiff[{f_, g_}, {x_, xmin_, xmax_}] :=
	Block[{dom},
		dom = Refine[Element[f-g, Reals], xmin < x < xmax];
		
		Abs[f-g] /; TrueQ[dom]
	]

absDiff[{f_, g_}, _] := 
	Block[{diff = f - g, re, im},
		re = Re[diff];
		im = Im[diff];
		Piecewise[{{diff, re >= 0 && im == 0}, {-diff, re < 0 && im == 0}}, 0]
	]


(* ::FunctionResourceFunctionSection:: *)
(**)


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[] (*AlphaScannerFunctions`*)
