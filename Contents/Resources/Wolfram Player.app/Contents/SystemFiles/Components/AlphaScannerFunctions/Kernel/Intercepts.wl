(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}];


Intercepts::usage = "Intercepts[expr, {var1, var2, ...}] computes the intercepts of expr with the coordinate axes {var1, ...}. Intercepts[expr, {var1, var2, ...}, {param1, param2, ...}] computes the intercepts of expr with the coordinate axes {var1, ...} as the params are varied.";


Begin["`Intercepts`Private`"];


(* ::Chapter:: *)
(*main code*)


(* ::Subsection::Closed:: *)
(*Intercepts*)


ClearAll[Intercepts]
Intercepts[expr_]:= Intercepts[expr, VarList[expr]]
Intercepts[expr_, varList_]:= Intercepts[expr, varList, varList]
Intercepts[expr_, axesVars_List, params_List] := Quiet @ Module[
	{
		e, vars, res, eqnQ = !FreeQ[{expr},Equal], 
		depVar = First[Complement[axesVars, VarList[expr]], {}]
	}, 
	(*bug 262799*)
	If[ContainsQ[Simplify[Expand[expr]],Complex[__]|I], Return[$Failed]];
	
	If[eqnQ, 
		e = expr,
		(* if only a rhs is given, create a full equation for passing to Solve *)
		If[depVar === {}, Return[$Failed], e = depVar == expr]
	];
	vars = Union[axesVars, params];
	
	res = Union @@ (Chop /@ GetIntercepts[e, vars, axesVars, #]& /@ axesVars);
	res = res /.{ConditionalExpression[a_?NumericQ,Element[C[1],Integers]]:>a}/.{c:ConditionalExpression[_,Element[C[1],Integers]] :> (If[NumericQ[#],#,c]&@(Assuming[#[[2]],Simplify[#[[1]]]]&[List@@c]))}
]


(* ::Subsection::Closed:: *)
(*GetIntercepts*)


$singlesteptimelimit = 0.1;
(* the main intercepts function *)
GetIntercepts[input_, vars_, axesvars_, axes_] :=(
	TimeConstrained[Quiet@Refine@Solve[Flatten@{input, Thread[Complement[axesvars,{axes}]==0]}, vars, Method -> "Reduce"],$singlesteptimelimit,$Failed]
)

(* This is so that we can return intercepts within some domain through CustomScannerData. *)
GetIntercepts[input_, vars_, axesvars_, axes_, {a_, lo_, b_}] := 
	TimeConstrained[Quiet@Refine@Solve[Flatten@{input, Thread[Complement[axesvars,{axes}]==0], If[lo<b,lo <= a <= b, b <= a <= lo]}, vars, Method -> "Reduce"],$singlesteptimelimit,$Failed]

GetIntercepts[input_, vars_, axesvars_, axes_, Automatic] :=
	GetIntercepts[input, vars, axesvars, axes]

GetIntercepts[___] := $Failed


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[]
