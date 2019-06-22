(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`CommonFunctions`"]; 


$AlphaScannerFunctionsDebug
myPrint
SupersetQ
guessCoordinateSystem
ContainsQ
HessianH
EliminateExtraneousRoots
polynomialDenominator
SensibleResults::usage = "Checks a Reduce result to determine if it is vacuous."
NReduce::usage = "NReduce[ineqs, {x, a, b}] numerically solves ineqs for a < x < b"
SortByPosition
EvenFnQ
VarList
usersymbolQ
preprocessUserInput
preprocessUserInputHeld
chooseVariables
ImplicitFuncQ
realFunctionQ
SortAndOr
refineCriticalPoints
OrList
chooseinterval
inequalityBoundary
inequalitiesToIntervals
chooseTestPoint
inequalityClosure
debug
niceN


Begin["`Private`"];


(* ::Chapter::Closed:: *)
(*debug helpers*)


ClearAll[myPrint]
$AlphaScannerFunctionsDebug = False;
$myPrintStylingFunction = Style[#, Darker[Green]]&;
SetAttributes[myPrint, HoldRest]
myPrint[args___]:= If[$AlphaScannerFunctionsDebug, Print[Style["XXXX BAD myPrint SYNTAX XXXXX!!!", Darker[Green]]]]
myPrint[label_String]:= Print[$myPrintStylingFunction[label]] /; $AlphaScannerFunctionsDebug
myPrint [label_String, args__]:= Module[
	{
		stylingFunction = $myPrintStylingFunction,
		line1, line2, vars= Hold[{args}]
	},
	line1 = Prepend[ReleaseHold[MapAt[HoldForm, vars, {1, All}]], ""];
	line2 = Prepend[{args}, stylingFunction[label]];
	Print[Grid[{line1, line2}, Dividers -> {{False,{True}}, Center} ]];
] /; $AlphaScannerFunctionsDebug


ClearAll[debug]
Attributes[debug] = {HoldAllComplete, SequenceHold};
debug[expr_] := Block[{$AlphaScannerFunctionsDebug = True}, expr];


(* ::Chapter::Closed:: *)
(*CalculateParse`GeneralLibrary*)


(* ::Section::Closed:: *)
(*SupersetQ*)


ClearAll[SupersetQ]
SupersetQ::usage = "SupersetQ[a, b] is Subset[b, a]";
SupersetQ[a_, b_]:= SubsetQ[b, a]


(* ::Chapter::Closed:: *)
(*CalculateScan`ApplicationsOfIntegrationScanner`*)


(* ::Section::Closed:: *)
(*guessCoordinateSystem*)


ClearAll[guessCoordinateSystem];

guessCoordinateSystem[__] = "unknown";

guessCoordinateSystem[_?NumericQ | {__?NumericQ} | (_?usersymbolQ == _?NumericQ), _] := "Cartesian"

guessCoordinateSystem[a : {Repeated[(_?usersymbolQ|Subscript[_,_])[_] == _]}, {v_, lo_, hi_}] /; 
   FreeQ[a[[All, 2]], Alternatives @@ Head /@ a[[All, 1]]] && 
    VarList[a[[All, 2]]] == {v} := "parametric";

guessCoordinateSystem[a : {Repeated[(_?usersymbolQ|Subscript[_,_]) == _]}, {v_, lo_, hi_}] /; 
   FreeQ[a[[All, 2]], Alternatives @@ a[[All, 1]]] && 
    VarList[a[[All, 2]]] == {v} := "parametric";


(*

guessCoordinateSystem[{x_?usersymbolQ == fx_, y_?usersymbolQ == fy_}, {v_, lo_, hi_}] /; 
	FreeQ[{fx, fy}, x | y] && VarList[{fx, fy}] == {v} := "parametric"

(*
In[123]:= guessCoordinateSystem[{x == t, y == 1 - t}, {t, -1, 1}]
Out[123]= "parametric"
*)

guessCoordinateSystem[{x_?usersymbolQ[v_] == fx_, y_?usersymbolQ[v_] == fy_}, {v_, lo_, hi_}] /; 
  FreeQ[{fx, fy}, x | y] && VarList[{fx, fy}] == {v} := "parametric"

(*
In[125]:= guessCoordinateSystem[{x[t] == t, y[t] == 1 - t}, {t, -1, 1}]
Out[125]= "parametric"
*)

guessCoordinateSystem[{x_?usersymbolQ == fx_, y_?usersymbolQ == fy_, z_?usersymbolQ == fz_}, {v_, lo_, hi_}] /; 
  FreeQ[{fx, fy, fz}, x | y | z] && VarList[{fx, fy, fz}] == {v} := "parametric"

guessCoordinateSystem[{x_?usersymbolQ == fx_, y_?usersymbolQ == fy_, z_?usersymbolQ == fz_, w_?usersymbolQ == fw_}, {v_, lo_, hi_}] /; 
  FreeQ[{fx, fy, fz, fw}, x | y | z | w] && VarList[{fx, fy, fz, fw}] == {v} := "parametric"

guessCoordinateSystem[{x_?usersymbolQ == fx_, y_?usersymbolQ == fy_, z_?usersymbolQ == fz_, w_?usersymbolQ == fw_, s_?usersymbolQ == fs_}, {v_, lo_, hi_}] /; 
  FreeQ[{fx, fy, fz, fw, fs}, x | y | z | w | s] && VarList[{fx, fy, fz, fw, fs}] == {v} := "parametric"
(*
In[127]:= guessCoordinateSystem[{x == t, y == 1 - t, z == 1 + t}, {t, -1, 1}]
Out[127]= "parametric"
*)

guessCoordinateSystem[{x_?usersymbolQ[v_] == fx_, y_?usersymbolQ[v_] == fy_, z_?usersymbolQ[v_] == fz_}, {v_, lo_, hi_}] /; 
  FreeQ[{fx, fy, fz}, x | y | z] && VarList[{fx, fy, fz}] == {v} := "parametric"
*)  
(* 	THIS KILLS THE OUTPUT FOR "compute the area between y=|x| and y=x^2-6"
In[128]:= guessCoordinateSystem[{x,x^2}, {x, 0, 3}]
Out[128]= "parametric"


guessCoordinateSystem[fx_List, {x_, __}] /; VarList[fx] === {x} && !MatchQ[x, Symbol["\[Theta]"] | Symbol["\[Phi]"] | Symbol["\[CurlyTheta]"]] := (Print[fx];"parametric")
*)

(*
In[129]:= guessCoordinateSystem[{x[t] == t, y[t] == 1 - t, z[t] == 1 + t}, {t, -1, 1}]
Out[129]= "parametric"
*)

guessCoordinateSystem[y_?usersymbolQ == fx_, {x_, __}] /; ! MatchQ[y, Symbol["r"] | x] && VarList[fx] === {x} := "Cartesian"

(*
In[137]:= guessCoordinateSystem[y == 1 + x - x^2, {x, 0, 3}]
Out[137]= "Cartesian"
*)

guessCoordinateSystem[fx_, {x_, __}] /; VarList[fx] === {x} && !MatchQ[x, Symbol["\[Theta]"] | Symbol["\[Phi]"] | Symbol["\[CurlyTheta]"]] := "Cartesian"

(*
In[146]:= guessCoordinateSystem[1 + x - x^2, {x, 0, 3}]
Out[146]= "Cartesian"
*)

guessCoordinateSystem[fx_, {x_, __}] /; VarList[fx] === {x} && MatchQ[x, Symbol["\[Theta]"] | Symbol["\[Phi]"] | Symbol["\[CurlyTheta]"]] := "polar"

(*
In[149]:= guessCoordinateSystem[1 + \[Theta] - \[Theta]^2, {\[Theta], 0, \[Pi]}]
Out[149]= "polar"
*)

guessCoordinateSystem[rad_ == ftheta_, {theta_, __}] /; rad === Symbol["r"] && VarList[ftheta] === {theta} := "polar"

(*
In[173]:= guessCoordinateSystem[r == 1 + t^2, {t, 0, \[Pi]}]
Out[173]= "polar"
*)

guessCoordinateSystem[rad_[theta_] == ftheta_, {theta_, __}] /; rad === Symbol["r"] && VarList[ftheta] === {theta} && 
	MatchQ[theta, Symbol["\[Theta]"] | Symbol["\[Phi]"] | Symbol["\[CurlyTheta]"]] := "polar"

(*
In[104]:= guessCoordinateSystem[r[\[Theta]] == 1 + Sin[\[Theta]], {\[Theta], 0, Pi}]
Out[104]= "polar"
*)

(* TODO: Other coordinate systems. *)


(* ::Chapter::Closed:: *)
(*CalculateScan`CommonFunctions`*)


ClearAll[ContainsQ]
ContainsQ[expr_, form_, levelSpec_: {0, Infinity}]:= !FreeQ[expr, form ,levelSpec]
ContainsQ[form_]:= Function[expr, !FreeQ[expr, form]]


(* ::Chapter::Closed:: *)
(*CalculateScan`LinearAlgebraScanner*)


(* ::Section:: *)
(*HessianH*)


HessianH[f_, x_List?VectorQ] := D[f, {x, 2}]


(* ::Chapter::Closed:: *)
(*CalculateScan`Packages`SolveReduceCommonFunctions*)


(* ::Section::Closed:: *)
(*EliminateExtraneousRoots*)


(* ::Subsection::Closed:: *)
(*EliminateExtraneousRoots*)


(* ::Text:: *)
(*For the sake of speed, EliminateExtraneousRoots only looks for extraneous roots that arise from polynomials. This accounts for the vast majority of complaints.*)


SetAttributes[EliminateExtraneousRoots, HoldFirst];


EliminateExtraneousRoots[Hold[eq_], rest__] := EliminateExtraneousRoots[eq, rest]


EliminateExtraneousRoots[eq_, x_, roots:(_Reduce|False|_Solve|{}|{{}})] := roots


EliminateExtraneousRoots[eq_, x_, rest___] /; !ListQ[x] := EliminateExtraneousRoots[eq, {x}, rest]


(* Solve case: just plug in the roots and check. *)
EliminateExtraneousRoots[eq_, _, roots : {{_Rule ..} ..}] /; FreeQ[Hold[eq], Greater | Less | GreaterEqual | LessEqual | Element | Exists | ForAll] :=
Select[roots, 
	With[{eval = Hold[eq] /. #},
		Or[
			!FreeQ[eval, ConditionalExpression | C],
			Quiet@Check[ReleaseHold[eval /. {x_?InexactNumberQ /; Abs[x] < 10^-20 -> 0.}];
				ReleaseHold[With[{r = Cases[eval, x_?InexactNumberQ /; Depth[Hold[x]] == 2 :> (HoldPattern[x] -> Rationalize[x]), Infinity]}, eval /. r]]; True, 
				False,
				{Power::"infy", Infinity::"indet"}
			]
		]
	]&
]


(* Reduce case. *)
EliminateExtraneousRoots[eq_, {x_}, roots_] /; !ListQ[roots] := 
	Block[{pden, elimsing},
		pden = polynomialDenominator[eq, x];
		elimsing = If[!NumericQ[pden], 
			TimeConstrained[Reduce[roots && pden != 0, x], $singlesteptimelimit/3, roots], 
			roots
		];
		iEliminateExtraneousRoots[eq, Replace[elimsing, Unequal[p_, e_] :> unequalToInequalities[eq, p != e, x], {0}]]
	]


EliminateExtraneousRoots[_, _, r_] := r


(* ::Subsubsection:: *)
(*iEliminateExtraneousRoots*)


(* ::Text:: *)
(*A helper function to EliminateExtraneousRoots which specializes in removing endpoint falicies in inequalities.*)


SetAttributes[iEliminateExtraneousRoots, HoldFirst];


iEliminateExtraneousRoots[Hold[eq_], r_] := iEliminateExtraneousRoots[eq, r]


iEliminateExtraneousRoots[eq_, r : (_Or | _And)] := iEliminateExtraneousRoots[eq, #]& /@ r


iEliminateExtraneousRoots[eq_, roots_] /; !FreeQ[Hold[eq, roots], Element | Exists | ForAll | Not | _?InexactNumberQ] := roots


iEliminateExtraneousRoots[eq_, i_Inequality] /; MemberQ[i, GreaterEqual | LessEqual] := 
	With[{l = List @@ i, gleqPos = Flatten[Position[i, GreaterEqual | LessEqual, {1}]]},
		ReplacePart[i, 
			# -> (Head[iEliminateExtraneousRoots[eq, #2[#1, #3]]] & @@ l[[# - 1 ;; # + 1]]) & /@ gleqPos
		]
	]


iEliminateExtraneousRoots[eq_, r : (_Equal | _GreaterEqual | _LessEqual)] := 
	With[{req = Reduce`ReduceToRules[Equal @@ r]},
		Which[
			req === $Failed || FreeQ[req, Rule], 
				r,
			And[
				FreeQ[req, ConditionalExpression | C], 
				Quiet[Check[ReleaseHold[Hold[eq] /. req];False, True, {Power::"infy", Infinity::"indet"}]]
			],
				Replace[r, {a_ == b_ :> False, a_ >= b_ :> a > b, a_ <= b_ :> a < b}, {0}],
				True,
				r
		]
	]


iEliminateExtraneousRoots[eq_, r_] := r


(* ::Subsection::Closed:: *)
(*EliminateExtraneousRoots*)


(* ::Text:: *)
(*For the sake of speed, EliminateExtraneousRoots only looks for extraneous roots that arise from polynomials. This accounts for the vast majority of complaints.*)


SetAttributes[EliminateExtraneousRoots, HoldFirst];


EliminateExtraneousRoots[Hold[eq_], rest__] := EliminateExtraneousRoots[eq, rest]


EliminateExtraneousRoots[eq_, x_, roots:(_Reduce|False|_Solve|{}|{{}})] := roots


EliminateExtraneousRoots[eq_, x_, rest___] /; !ListQ[x] := EliminateExtraneousRoots[eq, {x}, rest]


(* Solve case: just plug in the roots and check. *)
EliminateExtraneousRoots[eq_, _, roots : {{_Rule ..} ..}] /; FreeQ[Hold[eq], Greater | Less | GreaterEqual | LessEqual | Element | Exists | ForAll] :=
Select[roots, 
	With[{eval = Hold[eq] /. #},
		Or[
			!FreeQ[eval, ConditionalExpression | C],
			Quiet@Check[ReleaseHold[eval /. {x_?InexactNumberQ /; Abs[x] < 10^-20 -> 0.}];
				ReleaseHold[With[{r = Cases[eval, x_?InexactNumberQ /; Depth[Hold[x]] == 2 :> (HoldPattern[x] -> Rationalize[x]), Infinity]}, eval /. r]]; True, 
				False,
				{Power::"infy", Infinity::"indet"}
			]
		]
	]&
]


(* Reduce case. *)
EliminateExtraneousRoots[eq_, {x_}, roots_] /; !ListQ[roots] := 
	Block[{pden, elimsing},
		pden = polynomialDenominator[eq, x];
		elimsing = If[!NumericQ[pden], 
			TimeConstrained[Reduce[roots && pden != 0, x], $singlesteptimelimit/3, roots], 
			roots
		];
		iEliminateExtraneousRoots[eq, Replace[elimsing, Unequal[p_, e_] :> unequalToInequalities[eq, p != e, x], {0}]]
	]


EliminateExtraneousRoots[_, _, r_] := r


(* ::Subsubsection:: *)
(*iEliminateExtraneousRoots*)


(* ::Text:: *)
(*A helper function to EliminateExtraneousRoots which specializes in removing endpoint falicies in inequalities.*)


SetAttributes[iEliminateExtraneousRoots, HoldFirst];


iEliminateExtraneousRoots[Hold[eq_], r_] := iEliminateExtraneousRoots[eq, r]


iEliminateExtraneousRoots[eq_, r : (_Or | _And)] := iEliminateExtraneousRoots[eq, #]& /@ r


iEliminateExtraneousRoots[eq_, roots_] /; !FreeQ[Hold[eq, roots], Element | Exists | ForAll | Not | _?InexactNumberQ] := roots


iEliminateExtraneousRoots[eq_, i_Inequality] /; MemberQ[i, GreaterEqual | LessEqual] := 
	With[{l = List @@ i, gleqPos = Flatten[Position[i, GreaterEqual | LessEqual, {1}]]},
		ReplacePart[i, 
			# -> (Head[iEliminateExtraneousRoots[eq, #2[#1, #3]]] & @@ l[[# - 1 ;; # + 1]]) & /@ gleqPos
		]
	]


iEliminateExtraneousRoots[eq_, r : (_Equal | _GreaterEqual | _LessEqual)] := 
	With[{req = Reduce`ReduceToRules[Equal @@ r]},
		Which[
			req === $Failed || FreeQ[req, Rule], 
				r,
			And[
				FreeQ[req, ConditionalExpression | C], 
				Quiet[Check[ReleaseHold[Hold[eq] /. req];False, True, {Power::"infy", Infinity::"indet"}]]
			],
				Replace[r, {a_ == b_ :> False, a_ >= b_ :> a > b, a_ <= b_ :> a < b}, {0}],
				True,
				r
		]
	]


iEliminateExtraneousRoots[eq_, r_] := r


(* ::Section::Closed:: *)
(*polynomialDenominator*)


(* ::Subsubsection:: *)
(*polynomialDenominator*)


SetAttributes[polynomialDenominator, {HoldFirst}];


polynomialDenominator[expr_, {x_}] := polynomialDenominator[expr, x]


polynomialDenominator[l_List, x_] := Times @@ Function[expr, polynomialDenominator[expr, x], HoldFirst] /@ Unevaluated[l]


polynomialDenominator[(Greater | GreaterEqual | Less | LessEqual | Equal)[rats__], x_] := polynomialDenominator[{rats}, x]


polynomialDenominator[Inequality[rat1_, _, rat2_, _, rat3_], x_] := polynomialDenominator[{rat1, rat2, rat3}, x]


polynomialDenominator[(Plus|Times)[args__], x_] := polynomialDenominator[{args}, x]


polynomialDenominator[Abs[f_], x_] := polynomialDenominator[f, x]


polynomialDenominator[Sign[Abs[f_]], x_] := 
	Block[{tog, num, den},
		tog = Together[f];
		num = Numerator[tog];
		den = Denominator[tog];
		
		If[algPolynomialQ[num, x], num, 1] * polynomialDenominator[Evaluate[1/den], x]
	]


polynomialDenominator[Power[f_, _?Negative], x_] /; algPolynomialQ[f, x] := f


polynomialDenominator[Power[Abs[f_], _?Negative], x_] /; algPolynomialQ[f, x] := f


polynomialDenominator[Power[HoldPattern[Times[args__]], _?Negative], x_] := Times @@ Select[{args}, algPolynomialQ[#, x]&]


polynomialDenominator[Power[f_, _?Negative], x_] := polynomialDenominator[f, x]


polynomialDenominator[_, _] = 1;


algPolynomialQ[HoldPattern[Plus][args__], x_] := VectorQ[{args}, algPolynomialQ[#, x]&]
algPolynomialQ[HoldPattern[Times][args__], x_] := VectorQ[{args}, algPolynomialQ[#, x]&]
algPolynomialQ[f_^n_, x_] /; IntegerQ[n] || Head[n] === Rational := algPolynomialQ[f, x]
algPolynomialQ[x_, x_] = True;
algPolynomialQ[a_, x_] := FreeQ[a, x]


(* ::Section::Closed:: *)
(*SensibleResults*)


Options[SensibleResults] = {GeneratedParameters -> C};


SensibleResults[res_, unknowns_List, opts:OptionsPattern[]] := Module[{tmp, remove, param},
	param = OptionValue[SensibleResults, {opts}, GeneratedParameters];
	
	(* Fixes bug 296431 *)
	tmp = Refine[res /. Element -> Hold[Element], And @@ Cases[res, _Element, Infinity]] /. Hold[Element] -> Element;
	tmp = Reduce`ToDNF[tmp]; (* LogicalExpand sometimes reorders equalities so that the unknown is on the RHS instead of LHS. *)
	
	tmp = Replace[tmp, {HoldPattern[Or[z__]] :> {z}, z_ :> {z}}];
	tmp = Replace[tmp, {HoldPattern[And[z__]] :> {z}, z_ :> {z}}, 1];
	
	(* convert ArcTan[1+Sqrt[2]] -> 3Pi/8, etc from Weierstrass sub that Reduce defaults to *)
	tmp = SimplifyArcTan[tmp];

	(* expand domain specs *)
	tmp = tmp /. 
		HoldPattern[Alternatives][x__] \[Element] e_ :> 
		Sequence @@ (# \[Element] e &) /@ {x};
	
	(* convert C[n]\[Element]Z, a==C[n], b==f[C[n] to a\[Element]Z, b==f[a]*)
	(* The Condition fixes bug 271381 *)
	tmp = tmp //. {a___, b_param \[Element] c_, d___, e_ == b_, f___} /; !And[{a} === {d} === {f} === {}, MemberQ[unknowns, e]] :> 
		({a, e \[Element] c, d, f} /. b -> e);
	
	(* remove stupid a\[Element]Integers, a==f[b,___] results *)
	tmp = DeleteCases[tmp, {___, a_ \[Element] z_, ___, a_ == _, ___}];
	
	(* remove stupid a\[Element]Integers, a>f[b,___] results *)
	tmp = DeleteCases[tmp, 
		{___, a_?(MemberQ[unknowns,#]&) \[Element] Integers, ___, 
		(Less|LessEqual|Greater|GreaterEqual)[a_, _], ___}];
		
	(* remove stupid a\[Element]Reals domain spec combined with a>f[b,___] *)
	tmp = tmp /. 
		{a___, b_ \[Element] Reals, c___, d:(Less|LessEqual|Greater|GreaterEqual)[b_,_],e___}:>
		{a,c,d,e};

	(* split conditions from variable results *)
	tmp = {
		DeleteCases[#, x_ /; FreeQ[x, Alternatives @@ unknowns] || MatchQ[x, _Element]],
		Cases[#, x_ /; FreeQ[x, Alternatives @@ unknowns]],
		Cases[#, _?(MemberQ[unknowns, #] &) \[Element] _]
	} & /@ tmp;

	tmp = Replace[tmp, {
		{a_?(FreeQ[#, _param]&), b_?(!FreeQ[#, _param]&), c_} :> {a, DeleteCases[b, _?(!FreeQ[#, _param]&)], c}
	}, {1}];
	
	(* Combine cases like (C[1] \[Element] Integers && C[1] <= 0) || (C[1] \[Element] Integers && C[1] >= 1) into C[1] \[Element] Integers *)
	tmp = tmp //. {
		a___, 
		{sol_, {p_param \[Element] Integers, (h1:(LessEqual|GreaterEqual))[p_param, bound1_Integer]}, cond_},
		b___,
		{sol_, {p_param \[Element] Integers, (h2:(LessEqual|GreaterEqual))[p_param, bound2_Integer]}, cond_},
		c___
	} /; (h1 === LessEqual    && h2 === GreaterEqual && bound1 == bound2-1) || 
		 (h1 === GreaterEqual && h2 === LessEqual    && bound1 == bound2+1)
	:> {a, {sol, {p \[Element] Integers}, cond}, b, c};

	(* keep if 1 true variable result, or 2 false (relational) variable results *)
	tmp = Cases[tmp, {x_, r__} /; 
		Length[x] > 1 || 
		MatchQ[x, {(Equal|Greater|GreaterEqual|Less|LessEqual|Inequality|Unequal)[_, _, __]}] ||
		MatchQ[x, {(Equal|Greater|GreaterEqual|Less|LessEqual|Unequal)[_,_?(FreeQ[#, Alternatives @@ unknowns]&)]}] ||
		MatchQ[x, {(Equal|Greater|GreaterEqual|Less|LessEqual|Unequal)[_?(MemberQ[unknowns,#]&),_]}] ||
		Length[x] == 0 && MatchQ[{r}, {{}, {__Element}}]
	];
	
	(* try removing some specific duplicate results. For example:
		In[246]:= Reduce[x^2 Sin[x] == 0, x]
		Out[246]= (C[1] \[Element] Integers && (x == 2 \[Pi] C[1] || x == \[Pi] + 2 \[Pi] C[1])) || x == 0
	the solution x == 0 is unnecessary. SamB 0610 *)

	remove = Intersection[
  		Cases[tmp, r : {{_Symbol == _?NumericQ}, {}, {}} :> r[[1, 1]]], 
  		Flatten @ Cases[tmp, r : {{_Symbol == _}, {param[_] \[Element] Integers}, {}} :> Table[r[[1, 1]] /. param[_] :> i, {i, {-1, 0, 1}}]]
	];
		
	(* Remove identically true statements. See bug 299952 *)
	{#1, DeleteCases[#2, v_ /; TrueQ[N[v, 30]]], ##3}& @@@ Cases[tmp, {{sol___}, __} /; ! MemberQ[Partition[remove, 1], {sol}]]
	
	(* might still need to prune results with too many parameters *)
]

SensibleResults[res_, unknowns_] := SensibleResults[res, {unknowns}]
SensibleResults[{}, _] = {};



(* ::Section::Closed:: *)
(*SimplifyArcTan*)


SimplifyArcTan::usage = "Attempts to express ArcTan in as a rational multiple of Pi.";
(*SimplifyArcTan*)


(* ::Text:: *)
(*Attempt to eliminate ArcTan from an expression.*)
(*  e.g. convert ArcTan[1+Sqrt[2]] -> 3Pi/8, etc from Weierstrass sub that Reduce defaults to.*)
(*This helps Cos[6x+Pi/4] == Cos[3x+Pi/3] very much.*)


SimplifyArcTan[f_] := With[{strippedATan = f /. a:_ArcTan :> aTanSimp[a]},
	strippedATan //. {
		(h:Sin|Cos)[n_. a:ArcTan[_?System`Private`AlgebraicNumberQ]] /; IntegerQ[n] && n < 11 :> ToRadicals[RootReduce[FunctionExpand[h[n a]]]],
		(h:Sin|Cos)[n_. (m_. Pi + g_)] /; IntegerQ[n] && (IntegerQ[m] || Head[m] == Rational) :> h[n m Pi + n g]
	}
]


aTanSimp[a : ArcTan[r_ /; !FreeQ[r, Root|RootSum]]] := Block[{inrad = ToRadicals[r], simp},
	If[!FreeQ[inrad, Root|RootSum], 
		a,
		If[Head[(simp = SimplifyArcTan[ArcTan[inrad]])/Pi] === Rational, simp, a]
	]
]


(* Based off SimplifyDump`$FSTab[49] *)
aTanSimp[a : ArcTan[\[Eta]_?System`Private`AlgebraicNumberQ]] /; Abs[\[Eta]] > .01 := Block[{q}, 
	q = Rationalize[N[ArcTan[\[Eta]]/Pi, {20, 50}]];
	If[MemberQ[{Rational, Integer}, Head[q]] && Chop[Abs[N[ArcTan[\[Eta]] - q*Pi, {50, 50}]], 10^-48] == 0, q*Pi, a]
]


aTanSimp[a_] := a


(* ::Input:: *)
(*SimplifyArcTan[ArcTan[Tan[Pi/17]//ToRadicals]]*)


(* ::Input:: *)
(*SimplifyArcTan[ArcTan[1+Sqrt[2]]]*)


(* ::Input:: *)
(*SimplifyArcTan[ArcTan[1-Sqrt[2]]]*)


(* ::Input:: *)
(*SimplifyArcTan[(-2 ArcTan[2-Sqrt[2]+Sqrt[3]-Sqrt[6]]+2Pi C[1])/3]*)


(* ::Input:: *)
(*SimplifyArcTan[ArcTan[Root[{-3+#1^2&,-2+#2^2&,-2+#1+#2-#1 #2-3 #3+6 #3^2-3 #1 #3^2-3 #2 #3^2+3 #1 #2 #3^2+#3^3&},{2,2,1}]]]*)


(* ::Section::Closed:: *)
(*NReduce and friends*)


NReduce::usage = "NReduce[ineqs, {x, a, b}] numerically solves ineqs for a < x < b";
NReduce[expr_, {x_, a_, b_}] := 
	Catch @ Block[{pre, res},
		pre = Simplify`SimplifyInequalities[expr];
		
		res = If[literalCount[pre] > 10,
			RvachevSolve[pre, {x, a, b}],
			iNReduce[pre, {x, a, b}]
		];
		
		clipToZero[failOnEssentialSingularities[expr, {x, a, b}, res]]
	]


NReduce[___] = $Failed;


clipToZero[expr_] := expr /. x0_Real /; TrueQ[Abs[x0] < 1.*10^-300] -> 0.


failOnEssentialSingularities[expr_, {x_, a_, b_}, res_] :=
	If[!ListQ[res] || Length["IsolatedRoots" /. res] < 10,
		res,
		iFailOnEssentialSingularities[Simplify`FunctionSingularities[RvachevForm[expr], x, {"ESSENTIAL", "REAL", "IGNORE"}], {x, a, b}, res]
	]

iFailOnEssentialSingularities[Except[_List] | {_, _, _, {}, _}, _, res_] := res

iFailOnEssentialSingularities[{_, _, _, essen_, _}, {x_, a_, b_}, res_] :=
	Block[{cond, locs},
		cond = Or @@ And @@@ essen;
		locs = NReduce[cond, {x, a, b}];
		
		If[ListQ[locs] && !MatchQ[locs, {(_ -> {})..}],
			$Failed,
			res
		]
	]


(* ::Subsection:: *)
(*iNReduce*)


iNReduce[f_ == g_, {x_, a_, b_}] := oNReduce[f-g, {x, a, b}]


(* ::Text:: *)
(*Rather than plot Ramp[f-g], plot f-g and project negative y-values to the x-axis. This allows for better adaptive sampling, as it's hard to find a needle in a haystack of long stretches of 0's.*)
(**)
(*See NReduce[AiryBi[x] > x/100, {x, -30, -20}] around x == -25.15*)


iNReduce[(Less|LessEqual)[f_, g_], {x_, a_, b_}] := oNReduce[f-g, {x, a, b}, "ClipNegative" -> True]


iNReduce[(Greater|GreaterEqual)[f_, g_], {x_, a_, b_}] := oNReduce[g-f, {x, a, b}, "ClipNegative" -> True]


iNReduce[(op:Less|LessEqual|Greater|GreaterEqual|Equal)[f_, g_, h__], {x_, a_, b_}] := 
	Block[{res},
		res = iNReduce[op[##], {x, a, b}]& @@@ Partition[{f, g, h}, 2, 1];
		
		rootIntersection[res]
	]


iNReduce[HoldPattern[Inequality][args__], {x_, a_, b_}] := 
	Block[{res},
		res = iNReduce[#2[#1, #3], {x, a, b}]& @@@ Partition[{args}, 3, 2];
		
		rootIntersection[res]
	]


iNReduce[f_ != g_, {x_, a_, b_}] := rootComplement[iNReduce[f == g, {x, a, b}], {a, b}]


iNReduce[Unequal[f_, g_, h__], {x_, a_, b_}] := rootComplement[iNReduce[Or @@ Equal @@@ Subsets[{f, g, h}, {2}], {x, a, b}], {a, b}]


iNReduce[HoldPattern[And][c1___, f_ == g_, c2___], {x_, a_, b_}] := 
	Block[{res1, expr},
		res1 = iNReduce[f == g, {x, a, b}];
		(
			expr = And @@ Replace[{c1, c2}, e1_ == e2_ :> Abs[e1 - e2] <= 1.*10^-12, {1}];
			{
				"IsolatedRoots" -> Quiet[Select["IsolatedRoots" /. res1, expr /. x -> First[#]&]],
				"IntervalRoots" -> {}
			}
			
		) /; MatchQ[res1, {"IsolatedRoots" -> _, "IntervalRoots" -> {}}]
	]


iNReduce[HoldPattern[And][args__], {x_, a_, b_}] := 
	Catch[
		rootIntersection[
			With[{roots = iNReduce[#, {x, a, b}]},
				If[MatchQ[roots, {_ -> {}, _ -> {}}],
					Throw[roots, "$And"],
					roots
				]
			]& /@ {args}
		],
		"$And"
	]


iNReduce[HoldPattern[Or][args__], {x_, a_, b_}] := 
	Catch[
		rootUnion[
			With[{roots = iNReduce[#, {x, a, b}]},
				If[MatchQ[roots, {___, "IntervalRoots" -> {{aa_ /; a == aa, bb_ /; b == bb}}, ___}],
					Throw[roots, "$Or"],
					roots
				]
			]& /@ {args}
		],
		"$Or"
	]


iNReduce[Not[expr_], {x_, a_, b_}] := rootComplement[iNReduce[expr, {x, a, b}], {a, b}]


iNReduce[HoldPattern[Nand][args__], {x_, a_, b_}] := iNReduce[Or @@ Not /@ {args}, {x, a, b}]
iNReduce[HoldPattern[Nor][args__], {x_, a_, b_}] := iNReduce[And @@ Not /@ {args}, {x, a, b}]


iNReduce[expr:(_Xor|_Xnor), {x_, a_, b_}] := iNReduce[BooleanConvert[expr, "CNF"], {x, a, b}]


iNReduce[True, {_, a_, b_}] = {"IsolatedRoots" -> {}, "IntervalRoots" -> {{a, b}}};


iNReduce[False, _] = {"IsolatedRoots" -> {}, "IntervalRoots" -> {}};


iNReduce[___] := Throw[$Failed]


(* ::Subsection::Closed:: *)
(*oNReduce*)


Options[oNReduce] = {"ClipNegative" -> False};


oNReduce[f_, {x_, a_, b_}, opts:OptionsPattern[]] :=
	Block[{clippedf, plot, isolatedroots, intervalroots, nf},
		clippedf = If[TrueQ[OptionValue["ClipNegative"]], Ramp[f], f];
		
		plot = rootPlot[f, {x, a, b}, opts];
		
		isolatedroots = isolatedRoots[f, x, plot];
		intervalroots = intervalRoots[clippedf, {x, a, b}, plot];
		
		If[Length[isolatedroots] > 0 && Length[intervalroots] > 0,
			nf = Nearest[Flatten[List @@ intervalroots]];
			isolatedroots = Select[isolatedroots, !IntervalMemberQ[Interval @@ intervalroots, First[#]] && Abs[First[nf[#[[1]]]]-#[[1]]] > 64$MachineEpsilon&]
		];
		
		{
			"IsolatedRoots" -> isolatedroots,
			"IntervalRoots" -> intervalroots
		}
	]


(* ::Section::Closed:: *)
(*rootPlot and friends*)


Options[rootPlot] = {"ClipNegative" -> False};

rootPlot[fraw_, {x_, a_, b_}, OptionsPattern[]] :=
	Block[{f, df, clipQ, plot},
		clipQ = TrueQ[OptionValue["ClipNegative"]];
		
		setHighPrecisionNearZero[f, fraw, x];
		setHighPrecisionNearZero[df, realD[fraw, x], x];
		
		plot = Visualization`Core`Plot[
			f, {x, a, b},
			Exclusions -> {f == 0, {df == 0, If[clipQ, f >= 0, True] && Abs[f] < 10^-1.}},
			ExclusionsStyle -> $exclusionsStyle,
			PlotStyle -> $plotStyle,
			PlotRange -> All
		];
		
		If[MatchQ[plot, Graphics[GraphicsComplex[_, {}], ___]], Throw[$Failed]];
		
		If[clipQ,
			plot = plot /. Line[pts_] :> Line[Replace[pts, {x0_, y0_?Negative} :> {x0, 0.}, {1}]]
		];
		
		plot
	]


(* ::Text:: *)
(*Do our best to alleviate catastrophic cancellation in numerical subtraction*)
(*Hack If + Unequal to keep f unevaluated for non numeric x*)
(*Hack Or to short circuit non machine precision evaluation*)


setHighPrecisionNearZero[f_, func_, x_] := 
	With[{rand = 5.87087*10^7, eps = 10^-8., pf = Function @@ {func /. x -> #1}},
		f = If[x != rand,
			Abs[y = func] > eps || y == 0. || (y = N[pf[SetPrecision[x, 30]]]);
			y
		];
			
	]


realD[f_, x_] := 
	With[{eps = 10^-8, res = D[f, x]},
		If[!FreeQ[res, (Conjugate|Abs|Arg|Re|Im)'],
			iComplexExpand[res],
			res
		] //. {
			Sign'[g_] :> DiracDelta[g], 
			Zeta'[g_] :> (Zeta[g + eps] - Zeta[g])/eps
		}
	]


$exclusionsStyle = Black;
$plotStyle = Green;


iComplexExpand[expr_] := ComplexExpand[expr /. pw_Piecewise :> (ComplexExpand /@ pw)]


(* ::Section::Closed:: *)
(*isolatedRoots and friends*)


isolatedRoots[f_, x_, plot_] :=
	Block[{brackets, roots},
		brackets = First[Cases[plot, {___, $exclusionsStyle, l__Line} :> {l}[[All, 1, All, 1]], \[Infinity]], {}];
		roots = getRootAndBracket[f, x, #]& /@ Sort[brackets];
		
		roots = roundRoot[f, x, #]& /@ roots;
		
		validateRoots[f, x, roots]
	]
	roundRoot[f_, x_, {r_, i_}] /; Abs[r] <= 10^-7. || (Abs[r] > 0.5 && Abs[1 - r/Round[r]] <= 10^-7.) :=
	Block[{n, valr, valn},
		n = N[Round[r]];
		valr = Quiet[f /. x -> r];
		valn = Quiet[f /. x -> n];
		
		Which[
			!FreeQ[valn, DirectedInfinity|ComplexInfinity|Indeterminate|Undefined|Interval], 
				Sequence @@ {},
			TrueQ[Abs[valn] <= Abs[valr]], 
				{n, i},
			True, 
				{r, i}
		]
	]

roundRoot[_, _, r_] := r


validateRoots[f_, x_, {}] = {};

validateRoots[f_, x_, roots_] :=
	With[{sings = Simplify`FunctionSingularities[f, x, "POLES"]},
		iValidateRoots[Or @@ And @@@ Union @@ sings, x, roots] /; ListQ[sings]
	]

validateRoots[__, roots_] := roots


iValidateRoots[False, x_, roots_] := roots

iValidateRoots[sings_, x_, roots_] :=
	Block[{paddedsings, valid},
		paddedsings = sings /. Equal[f_, g_] :> Abs[f-g] <= 10^-8.;
		
		valid = !paddedsings /. ({x -> #}& /@ roots[[All, 1]]);
		
		Pick[roots, TrueQ /@ valid]
	]


intervalRoots[f_, x_, plot_Graphics] := 
	Block[{cands},
		cands = Join @@ zints /@ First[Cases[plot, {___, $plotStyle, l__Line} :> {l}[[All, 1]], \[Infinity]]];
		
		intervalRoots[f, x, cands]
	]

intervalRoots[f_, x_, {}] = {};

intervalRoots[f_, {x_, a_, b_}, ints_?MatrixQ] := 
	Block[{res, bds},
		res = BinarySearchIntervalEndpoint[f, {x, ##}]& @@@ ints;
		bds = N[{Max[#1, a], Min[#2, b]}& @@@ res];
		
		N[List @@ IntervalUnion @@ Interval /@ Rationalize[bds, 0]]
	]

intervalRoots[___] = {};


getRootAndBracket[f_^n_?Positive, args__] := getRootAndBracket[f, args]

getRootAndBracket[(Abs|ArcTan)[f_], args__] := getRootAndBracket[f, args]

getRootAndBracket[__, {b_, b_}] := {b, {b, b}}

getRootAndBracket[f_, x_, {a_, b_}] :=
	Block[{res, val},
		res = Quiet[Check[FindRoot[f, {x, a, b}, Evaluate[findRootOptions[f, x]]], $Failed, FindRoot::fddis]];
		(
			res = res /. c_Complex :> Re[c];
			val = Quiet[f /. res];
			{res, val} = refineRoot[f, {a, b}, res, val];
			
			(
				{x /. res, {a, b}}
				
			) /; Abs[val] < 10^-6.
			
		) /; Quiet[MatchQ[res, {x -> _Real|_Complex}]] && Less[a, Re[x /. res], b] && Abs[Im[x /. res]] < 10^-10.
	]

getRootAndBracket[___] := Sequence @@ {}


(* ::Text:: *)
(*Avoid bad derivatives through the secant method.*)


findRootOptions[f_, x_] /; !FreeQ[f, (Re|Im|Zeta)[v_ /; !FreeQ[v, x]]] = WorkingPrecision -> 30;


findRootOptions[___] := Sequence @@ {};


refineRoot[__, res_, val_] /; Abs[val] < 10^-6. := {res, val}


refineRoot[f_, {a_, b_}, {x_ -> x0_}, val_] := 
	Block[{res, v},
		res = Quiet[FindRoot[f, {x, a, b}, MaxIterations -> 1000, WorkingPrecision -> 30]];
		(
			v = Quiet[f /. res];
			
			N[{res, v}] /; NumericQ[v]
			
		) /; MatchQ[res, {x -> _Real}] && Less[a, x /. res, b]
	]


refineRoot[__, res_, val_] := {res, val}


zints[pts_] /; FreeQ[pts, {_, 0.}] = {};
zints[pts_] := Select[SplitBy[pts, Last[#] == 0&][[All, {1, -1}]], #[[1, -1]] == 0&][[All, All, 1]]


BinarySearchIntervalEndpoint[f_, {x_, a_, b_}] :=
	Block[{g},
		setHighPrecisionNearZero[g, f, x];
		{
			BinarySearchLeftEndpoint[g, {x, a, b}],
			BinarySearchRightEndpoint[g, {x, a, b}]
		}
	]


BinarySearchLeftEndpoint[f_, {x_, a_, b_}] := 
	-BinarySearchRightEndpoint[f /. x -> -x, {x, -b, -a}]


BinarySearchRightEndpoint[f_Symbol[x_], {x_, a_, b_}] :=
	BinarySearchRightEndpoint[f, {a, b}]

BinarySearchRightEndpoint[f_, {x_, a_, b_}] :=
	BinarySearchRightEndpoint[Function @@ {f /. x -> #}, {a, b}]

BinarySearchRightEndpoint[f_, {a_, b_}] :=
	Quiet @ Block[{\[Delta] = 0.0001, pn, cap, lo, hi, start},
		pn = b + \[Delta];
		cap = b + .25Abs[b-a];
		
		While[TrueQ[f[pn] == 0.] && pn < cap,
			\[Delta] *=2;
			pn = Min[cap, b + \[Delta]]
		];
		
		If[pn == cap, Return[pn]];
		
		{lo, hi} = {b, pn};
		start = {.01, .99}.{lo, hi};
		
		FixedPoint[(
				Which[TrueQ[Abs[#] < 1.*10^-40], lo = hi = 0., TrueQ[f[#] == 0.], lo = #, True, hi = #];
				Mean[{lo, hi}]
			)&,
			start
		]
	]


(* ::Section::Closed:: *)
(*RvachevSolve*)


RvachevSolve[expr_, {x_, a_, b_}] := 
	Block[{rvform},
		rvform = RvachevForm[expr];
		If[rvform === $Failed,
			Throw[$Failed],
			oNReduce[clipAndWarp[rvform], {x, a, b}]
		]
	]


(* ::Subsubsection::Closed:: *)
(*RvachevForm*)


RvachevForm[expr_] := Catch[iRvachevForm[expr] //. {mm:(-(h:Min|Max)[a__]) :> reverseMinMax[h] @@ Distribute[mm, h, Times]}]


iRvachevForm[HoldPattern[Or][args__]] := Max @@ iRvachevForm /@ {args}

iRvachevForm[HoldPattern[And][args__]] := Min @@ iRvachevForm /@ {args}

iRvachevForm[HoldPattern[Inequality][args__]] := Min @@ (iRvachevForm[#2[#1, #3]]& @@@ Partition[{args}, 3, 2])

iRvachevForm[(Greater|GreaterEqual)[f_, g_]] := f-g

iRvachevForm[(Less|LessEqual)[f_, g_]] := g-f

iRvachevForm[f_ == g_] := Min[f-g, g-f]

iRvachevForm[(h:Equal|Less|LessEqual|Greater|GreaterEqual)[args__]] := Min @@ (iRvachevForm[h[##]]& @@@ Partition[{args}, 2, 1])

iRvachevForm[f_ != g_] := Max[f-g, g-f]

iRvachevForm[Not[f_]] := -iRvachevForm[f]

iRvachevForm[___] := Throw[$Failed]

reverseMinMax[Min] = Max;
reverseMinMax[Max] = Min;


(* ::Section::Closed:: *)
(*literalCount*)


literalCount[expr:(_Equal|_Less|_LessEqual|_Greater|_GreaterEqual)] := Length[expr] - 1


literalCount[expr_Inequality] := (Length[expr] - 1)/2


literalCount[expr_Unequal] := Binomial[Length[expr], 2]


literalCount[HoldPattern[And][___, HoldPattern[Equal][_, _], ___]] = 1;


literalCount[HoldPattern[And|Or][args__]] := Total[literalCount /@ {args}]


literalCount[Not[expr_]] := literalCount[expr]


literalCount[HoldPattern[Nand][args__]] := literalCount[Or @@ Not /@ {args}]
literalCount[HoldPattern[Nor][args__]] := literalCount[And @@ Not /@ {args}]


literalCount[expr:(_Xor|_Xnor)] := literalCount[BooleanConvert[expr, "CNF"]]


literalCount[True|False] = 0;


literalCount[___] := Throw[$Failed]


(* ::Section::Closed:: *)
(*rootComplement, rootIntersection, and friends*)


rootComplement[rootList_List, {a_, b_}] := 
	Block[{isolatedroots, intervalroots, isofinal, intfinal},
		isolatedroots = IntervalUnion @@ Interval /@ Rationalize[("IsolatedRoots" /. rootList)[[All, 1]], 0];
		intervalroots = IntervalUnion @@ Interval /@ Rationalize["IntervalRoots" /. rootList, 0];
		
		intervalroots = IntervalUnion[isolatedroots, intervalroots];
		intfinal = Partition[Join[{a}, Flatten[List @@ intervalroots], {b}], 2];
		
		N@{
			"IsolatedRoots" -> {},
			"IntervalRoots" -> intfinal
		}
	]


rootIntersection[rootLists_List] := 
	Block[{isolatedroots, intervalroots, intersection, ptints, pcands, isonew},
		isolatedroots = Rationalize["IsolatedRoots" /. rootLists, 0];
		intervalroots = Rationalize["IntervalRoots" /. rootLists, 0];
		
		intersection = List @@ IntervalIntersection @@ MapThread[Interval @@ Join[#1[[All, 1]], #2]&, {isolatedroots, intervalroots}];
		ptints = Select[intersection, # == First[Interval[Mean[#]]]&];
		
		pcands = "IsolatedRoots" /. rootUnion[{"IsolatedRoots" -> #, "IntervalRoots" -> {}}& /@ isolatedroots];
		isonew = First[Reap[Scan[sowRootInInterval[#, pcands]&, ptints]][[-1]], {}];
		
		N@{
			"IsolatedRoots" -> Union[isonew[[All, 2]]],
			"IntervalRoots" -> Complement[intersection, isonew[[All, 1]]]
		}
	]


sowRootInInterval[_, {}] = Null;


sowRootInInterval[{a_, b_}, iso_] := 
	Block[{res},
		res = SelectFirst[iso, Between[a, Last[#]] && Between[b, Last[#]]&];
		If[ListQ[res],
			Sow[{{a, b}, res}]
		];
	]


rootUnion[rootLists_List] :=
	Block[{isolatedroots, intervalroots, nf, isofinal, intfinal},
		isolatedroots = Rationalize[Union @@ ("IsolatedRoots" /. rootLists), 0];
		intervalroots = Rationalize[Join @@ ("IntervalRoots" /. rootLists), 0];
		intervalroots = IntervalUnion @@ Interval /@ intervalroots;

		If[Length[intervalroots] > 0,
			nf = Nearest[Flatten[List @@ intervalroots]];
			isolatedroots = Select[isolatedroots, !IntervalMemberQ[intervalroots, First[#]] && Abs[First[nf[#[[1]]]]-#[[1]]] > 16$MachineEpsilon&]
		];
		
		isofinal = FixedPoint[mergeRoots, isolatedroots];
		intfinal = List @@ intervalroots;
		
		N@{
			"IsolatedRoots" -> isofinal,
			"IntervalRoots" -> intfinal
		}
	]


mergeRoots[roots_] := {Mean[#[[All, 1]]], MinMax[#[[All, 2]]]}& /@ Split[roots, Between[First[#1], Last[#2]] || Between[First[#2], Last[#1]]&]


(* ::Section::Closed:: *)
(*clipAndWarp*)


clipAndWarp[y_Real] := Clip[-y, {0, 1}]^4
clipAndWarp[y_?NumericQ] := If[TrueQ[y == 0], 0., 1.0I]
clipAndWarp[e:(_DirectedInfinity|ComplexInfinity|Indeterminate|Undefined)] := e

clipAndWarp /: realD[clipAndWarp[expr_], x_] := realD[expr, x]


(* ::Chapter::Closed:: *)
(*CalculateScan`Packages`MaxMinScanner*)


(* ::Section::Closed:: *)
(*SortPositionBy*)


SortByPosition[expr_, f_, l_List] := SortBy[{Position[l, #], #} & /@ expr, f][[All, -1]]
SortByPosition[expr_?AtomQ, _, _List] := expr


(* ::Section::Closed:: *)
(*EvenFnQ*)


EvenFnQ[expr_, {x_}] := TimeConstrained[TrueQ @ FullSimplify[expr == (expr /. x -> -x), Element[x, Reals]], $singlesteptimelimit/4, False]


(* ::Chapter::Closed:: *)
(*CalculateScan`PlotScanner`*)


(* ::Section::Closed:: *)
(*implicitPlotRange*)


(***********************************
*
*	implicitPlotRange
*
************************************)

uniteRanges[r : {{_, _, _}}] := First[r]
uniteRanges[r_List] := {r[[1, 1]], Min[r[[All, 2]]], Max[r[[All, 3]]]} 

Clear[implicitPlotRange];

implicitPlotRange[eqns_Equal, {x_, y_}] := implicitPlotRange[{eqns}, {x, y}]

implicitPlotRange[e:{__Equal}, {x_, y_}] := Block[
	{eqns, directRange, intersections, turningpts, highCurvature, pointsOfInterest, 
		boxes, xrange, yrange, xlines, ylines, lines, fnsOfx, fnsOfy, xr1D, yr1D, ranges,
		linears, linearRanges, linearPoints, xbds, sol},

	eqns = TimeConstrained[Simplify[e], $singlesteptimelimit/5, e];

	If[TimeConstrained[ 
			Resolve[Exists[{x, y}, Element[{x, y}, Reals], eqns]],
			$singlesteptimelimit/5,
			True
		] === False, Return @ $Failed];

	If[Complement[VarList[eqns], {x, y}] =!= {} || ContainsQ[eqns, _Derivative], Return[$Failed]];

	(* special case x == constant *)
	If[Length[eqns] == 1 && MatchQ[eqns, {x == rhs_} /; VarList[rhs] == {}],
		Return[ {{x, -1 - #, 1 + #}, {y, 1, 1}}& @ eqns[[1, -1]] ]
	];

	(* special case y == constant *)
	If[Length[eqns] == 1 && MatchQ[eqns, {y == rhs_} /; VarList[rhs] == {}],
		Return[ {{x, -1, 1}, {y, # - 1, # + 1}}& @ eqns[[1, -1]] ]
	];

	(* special case: y == f[x] *)
	If[Length[eqns] == 1 && MatchQ[eqns, {y == rhs_} /; FreeQ[rhs, y]],
		xrange = First @ AlphaScannerFunctions`SuggestPlotRange`Private`Get1DRange[eqns[[1, -1]]];
		yrange = AbsoluteOptions[Plot @@ {eqns[[1, -1]], xrange}, PlotRange][[1, -1, -1]];
		Return[{xrange, Prepend[yrange, y]}]
	];

	(* special case: x == f[y] *)
	If[Length[eqns] == 1 && MatchQ[eqns, {x == rhs_} /; FreeQ[rhs, x]],
		yrange = First @ AlphaScannerFunctions`SuggestPlotRange`Private`Get1DRange[eqns[[1, -1]]];
		xrange = AbsoluteOptions[Plot @@ {eqns[[1, -1]], yrange}, PlotRange][[1, -1, -1]];
		Return[{Prepend[xrange, x], yrange}]
	];
	
	(* special case: a < x < b in $Assumptions *)
	If[MatchQ[eqns, {_Equal}] && !FreeQ[$Assumptions, (Less|LessEqual|Greater|GreaterEqual)[_?NumericQ, x, _?NumericQ]|(Inequality[_?NumericQ, _, x, _, _?NumericQ])],
		xbds = First[Cases[$Assumptions, (Less|LessEqual|Greater|GreaterEqual)[_?NumericQ, x, _?NumericQ]|(Inequality[_?NumericQ, _, x, _, _?NumericQ]), {0, Infinity}]];
		xbds = {First[xbds], Last[xbds]};
		xrange = Prepend[Sort[xbds], x];
		sol = TimeConstrained[Refine[Reduce[First[eqns] && (#2 < #1 < #3& @@ xrange), y, Reals], #2 < #1 < #3& @@ xrange], .3$singlesteptimelimit];
		If[MatchQ[sol, y == _],
			yrange = Prepend[Last@PlotRange[Plot @@ {Last[sol], {x, xrange[[2]], Last[xrange]}}], y];
			Return[{xrange, yrange}]
		];
	];
	
	If[containsBivariateLinearQ[eqns, {x, y}],
		linears = Cases[eqns, eqn_ /; MatchQ[Subtract @@ eqn, a_. x + b_. y + c_. /; FreeQ[{a, b, c}, x | y]]];
		(* TODO: make my own linear rangefinder. 
			Need to improve range for implicitPlotRange[{5 x + 3 y^2 == 20, 3 x + 5 y == 100}, {x, y}]. SamB 0710 *)
		linearRanges = CalculateScan`Inequality2DScanner`Private`ProposeBivariateInequalityPlotRange[Less @@ #, {x, y}]& /@ linears;
		linearPoints = rangesToPoints[#, {x, y}]& /@ linearRanges,
		linearPoints = {}
	];
	
	(* catch special cases like "y = 1, x = - 4, y = x" to make sure we include all horizontal and vertical ranges. SamB 0510 *)
	xlines = Cases[eqns /. lhs_?NumericQ == x :> x == lhs, x == xline_?NumericQ :> {x -> xline, y -> 0}];
	ylines = Cases[eqns /. lhs_?NumericQ == y :> y == lhs, y == yline_?NumericQ :> {x -> 0, y -> yline}];
	lines = Flatten[{xlines, ylines}, 1];

	(* include intersections and turning points *)
	intersections = If[Length[eqns] == 1, {}, realSolve[{#1, #2}, {x, y}]& @@@ Tuples[eqns, 2]];
	turningpts = turningPoints[eqns, {x, y}];
	highCurvature = TimeConstrained[Flatten[findHighCurvature[#, {x, y}]& /@ eqns, 1], 4 $singlesteptimelimit, {}];
	
	printPlotScanner[{lines, intersections, turningpts, highCurvature, linearPoints}];
	
	pointsOfInterest = N @ Cases[{lines, intersections, turningpts, highCurvature, linearPoints}, {x -> _?NumericQ, y -> _?NumericQ}, Infinity];
	pointsOfInterest = extendFromOrigin[pointsOfInterest, {x, y}];

	If[MatchQ[pointsOfInterest, {{Rule[_, _?NumericQ]..}..}],
		boxes = {
				{Min[extendDown[Re[#1], 1.25], extendDown[Im[#1], 1.25]], Max[extendUp[Re[#1], 1.25], extendUp[Im[#1], 1.25]]}, 
				{Min[extendDown[Re[#2], 1.25], extendDown[Im[#2], 1.25]], Max[extendUp[Re[#2], 1.25], extendUp[Im[#2], 1.25]]}
			} & @@@ ({x, y} /. pointsOfInterest);
		xrange = {x, reMin[ boxes[[All, 1]] ], reMax[ boxes[[All, 1]] ]};
		yrange = {y, reMin[ boxes[[All, 2]] ], reMax[ boxes[[All, 2]] ]},
		directRange = CalculateScan`Inequality2DScanner`Private`ProposeBivariateInequalityPlotRange[Less @@ #, {x, y}]& /@ Flatten[{eqns}];
		directRange = DeleteCases[directRange, $Failed];
		If[MatchQ[directRange, {{{__?NumericQ}..}..}],
			ranges = (Flatten /@ Thread[{{x, y}, #}])& /@ directRange;
			{xrange, yrange} = uniteRanges1D /@ Transpose[ranges],
			directRange = TimeConstrained[Get2DRange[Subtract @@@ eqns, {x, y}], $singlesteptimelimit];
			If[MatchQ[directRange, {{_Symbol, __?NumericQ}, {_Symbol, __?NumericQ}}],
				{xrange, yrange} = directRange,
				{xrange, yrange} = {{}, {}}
			]
		]
	];

	{xrange, yrange} = {xrange, yrange} /. {v_, 0., 0.} :> {v, -1, 1};
	
	(* special case for "x = <fn of y>" or "y = <fn of x>", use AlphaScannerFunctions`SuggestPlotRange`Private`Get1DRange to find a good range *)
	fnsOfx = Cases[eqns /. lhs_ == y :> y == lhs, y == fnx_ /; FreeQ[fnx, y] && ContainsQ[fnx, x] :> {fnx, Get1DRangeExtendLinear[fnx]}];
	rangesXY = Cases[fnsOfx, {ptfn_, {xr:{x, _?NumericQ, _?NumericQ}}} :> {xr, extractRangeFromPlot[ptfn, xr, y]}, {0, Infinity}];

	fnsOfy = Cases[eqns /. lhs_ == x :> x == lhs, x == fny_ /; FreeQ[fny, x] && ContainsQ[fny, y] :> {fny, Get1DRangeExtendLinear[fny]}];
	rangesYX = Cases[fnsOfy, {ptfn_, {yr:{y, _?NumericQ, _?NumericQ}}} :> {yr, extractRangeFromPlot[ptfn, yr, x]}, {0, Infinity}];

	xr1D = Cases[{rangesXY, rangesYX}, {x, _?NumericQ, _?NumericQ}, {0, Infinity}];
	yr1D = Cases[{rangesXY, rangesYX}, {y, _?NumericQ, _?NumericQ}, {0, Infinity}];

	xrange = uniteRanges1D[N @ {xrange, Sequence @@ xr1D} /. {} -> Sequence[]];
	yrange = uniteRanges1D[N @ {yrange, Sequence @@ yr1D} /. {} -> Sequence[]];

	If[MatchQ[{xrange, yrange}, {{_Symbol, __?NumericQ}, {_Symbol, __?NumericQ}}],
		{xrange, yrange},
		$Failed
	]
]

implicitPlotRange[__] := $Failed

(*
In[335]:= implicitPlotRange[{x y - y == x^2/8 + y^2}, {x, y}]
ContourPlot[x y - y == x^2/8 + y^2, ##] & @@ %

Out[335]= {{x, -1.25, 5.}, {y, -0.441942, 1.50888}}


In[337]:= implicitPlotRange[x^2/y^2 - 1/(x y) + x y == 1/5, {x, y}]
ContourPlot[x^2/y^2 - 1/(x y) + x y == 1/5, ##] & @@ %

Out[337]= {{x, -3.32, 3.32}, {y, -3.32, 3.32}}


In[339]:= implicitPlotRange[x^2/25 - y^2/36 == 1, {x, y}]
ContourPlot[x^2/25 - y^2/36 == 1, ##] & @@ %

Out[339]= {{x, -31, 31}, {y, -37, 37}}


In[341]:= implicitPlotRange[x^2 + 3 y^2 == (x y)^2, {x, y}]
ContourPlot[x^2 + 3 y^2 == (x y)^2, ##] & @@ %

Out[341]= {{x, -3.89668, 3.89668}, {y, -3.04881, 3.04881}}


In[343]:= implicitPlotRange[x == Sqrt[y], {x, y}]
ContourPlot[x == Sqrt[y], ##] & @@ %

Out[343]= {{x, 0., 0.72111}, {y, -0.52, 0.52}}


In[345]:= implicitPlotRange[x^2 + (y + 123)^2 == 3, {x, y}]
ContourPlot[x^2 + (y + 123)^2 == 3, ##] & @@ %

Out[345]= {{x, -2, 2}, {y, -125, -121}}


In[347]:= implicitPlotRange[{x == 3 y^3, y == 1}, {x, y}]
ContourPlot[{x == 3 y^3, y == 1}, ##] & @@ %

Out[347]= {{x, -0.9375, 3.75}, {y, -0.3125, 1.25}}


In[349]:= implicitPlotRange[Log[x^2 + y^2]/(x^2 + y^2) == 0, {x, y}]
ContourPlot[Log[x^2 + y^2]/(x^2 + y^2) == 0, ##] & @@ %

Out[349]= {{x, -3., 3.}, {y, -3., 3.}}


In[351]:= implicitPlotRange[{x + y == 2, x^2 + y^2 == 4}, {x, y}]
ContourPlot[{x + y == 2, x^2 + y^2 == 4}, ##] & @@ %

Out[351]= {{x, -2.5, 2.5}, {y, -2.5, 2.5}}


In[353]:= implicitPlotRange[{y == x^3, x == 2, x == 3, y == 9}, {x, y}]
ContourPlot[{y == x^3, x == 2, x == 3, y == 9}, ##] & @@ %

Out[353]= {{x, -0.9375, 3.75}, {y, -2.8125, 11.25}}


In[355]:= implicitPlotRange[{x == 4, y == 4, y == x, x == 3 y, y^2 == x}, {x, y}]
ContourPlot[{x == 4, y == 4, y == x, x == 3 y, y^2 == x}, ##] & @@ %

Out[355]= {{x, -5., 20.}, {y, -2.5, 5.}}


In[357]:= implicitPlotRange[{Abs[x] == 1, x^2 + y^2 == 1}, {x, y}]
ContourPlot[{Abs[x] == 1, x^2 + y^2 == 1}, ##] & @@ %

Out[357]= {{x, -1.25, 1.25}, {y, -1.25, 1.25}}


In[359]:= implicitPlotRange[x y + y - 6 x == 0, {x, y}]
ContourPlot[x y + y - 6 x == 0, ##, Axes -> True, Frame -> False] & @@ %

Out[359]= {{x, -8.64199, 6.14199}, {y, 0.0648065, 14.892}}


In[361]:= implicitPlotRange[y == x Tan[Sqrt[x^2 + y^2]], {x, y}]
ContourPlot[y == x Tan[Sqrt[x^2 + y^2]], ##, Axes -> True, Frame -> False] & @@ %

Out[361]= {{x, -5.12977, 5.374}, {y, -3.9382, 4.61916}}


In[363]:= implicitPlotRange[x^2 + y^3 == (x y)^2, {x, y}]
ContourPlot[x^2 + y^3 == (x y)^2, ##] & @@ %

Out[363]= {{x, -2.67568, 2.67568}, {y, -3.32, 3.32}}


In[365]:= implicitPlotRange[x y + y == 0, {x, y}]
ContourPlot[x y + y == 0, ##] & @@ %

Out[365]= {{x, -5.56179, 2.29447}, {y, -3.59298, 2.13961}}


In[371]:= implicitPlotRange[{x == 3, y == 5 x - 3, y == 2 x + 4}, {x, y}]
ContourPlot[{x == 3, y == 5 x - 3, y == 2 x + 4}, ##] & @@ %

Out[371]= {{x, -3.0255, 3.75}, {y, -5.1275, 15.}}


In[373]:= implicitPlotRange[1/25 (x + 2)^2 - 1/16 (y - 1)^2 == 1, {x, y}]
ContourPlot[1/25 (x + 2)^2 - 1/16 (y - 1)^2 == 1, ##, Frame -> False, Axes -> True] & @@ %

Out[373]= {{x, -32, 28}, {y, -15, 17}}


In[191]:= implicitPlotRange[x^3 (x^2 - x - 4) + y^3 (y^2 - y - 4) == 6 x^2 y^2, {x, y}]

Out[191]= {{x, -2.77212, 6.21753}, {y, -2.77212, 6.21753}}


In[77]:= implicitPlotRange[(0.5` x^2 + y^2 - 1)^3 - x^2 y^3 == 0, {x, y}]

Out[77]= {{x, -2.18376, 2.18376}, {y, 0.543953, 1.70967}}

In[29]:= implicitPlotRange[{x^3 == 2 y^3 + 1, y == 0.5` x - 0.5`}, {x, y}]

Out[29]= {{x, -0.75, 3.125}, {y, -0.9375, 0.859206}}
*)

(* two lines *)
implicitPlotRange[eqns : {a_ == b_, c_ == d_}, {x_, y_}] /; 
  	PolynomialQ[a - b, {x, y}] && PolynomialQ[c - d, {x, y}] && 
  	Max[Exponent[a - b, {x, y}]] == 1 && Max[Exponent[c - d, {x, y}]] == 1 := Block[{xc, yc},
  	{xc, yc} = Flatten[{x, y} /. Solve[eqns, {x, y}]];
  	{{x, xc - 1, xc + 1}, {y, yc - 1, yc + 1}} /; VectorQ[{xc, yc}, NumericQ]
]

(* circle *)
implicitPlotRange[(x_ + a_.)^2 + (y_ + b_.)^2 == r_, {x_, y_}] /; r > 0 && NumericQ[a] && NumericQ[b] := {{x, -a - 3/2 r, -a + 3/2 r}, {y, -b - 3/2 r, -b + 3/2 r}}
  
(* hyperbola *)
implicitPlotRange[l:{__Equal}, {x_, y_}] /; VectorQ[l, hyperbolaQ[#, {x, y}]&] := uniteRanges /@ Transpose[DeleteCases[implicitPlotRange[#, {x, y}]& /@ l, $Failed]]


implicitPlotRange[a_. x_^2 + b_. y_^2 == r_, {x_, y_}] /; (b < 0 || a < 0) && r > 0 := 
	{{x, -Sqrt[r^2/Abs[a]] - Abs[1/a] - 1, Sqrt[r^2/Abs[a]] + Abs[1/a] + 1}, {y, -Abs[Numerator[a]^2/b] - 1, Abs[Numerator[a]^2/b] + 1}}

(* {Prepend[extend[#1, 1.5], x], Prepend[extend[#2, 1.5], y]}& @@ CalculateScan`Inequality2DScanner`Private`ProposeBivariateInequalityPlotRange[Less @@ expr, {x, y}] *)

implicitPlotRange[eq_Equal, {x_, y_}] /; hyperbolaQ[eq, {x, y}] := Module[
	{expr, d, xc, yc, tp, xdist, ydist},
	expr = Subtract @@ eq;
	d = Det[{{constantCoeff[expr, x^2], constantCoeff[expr, x y]}, {constantCoeff[expr, x y], constantCoeff[expr, y^2]}}];
	xc = -1/d Det[{{constantCoeff[expr, x]/2, constantCoeff[expr, x y]}, {constantCoeff[expr, y]/2, constantCoeff[expr, y^2]}}];
	yc = -1/d Det[{{constantCoeff[expr, x^2], constantCoeff[expr, x]/2}, {constantCoeff[expr, x y], constantCoeff[expr, x]/2}}];
	
	tp = turningPoints[expr == 0, {x, y}];
	If[tp === {}, Return @ $Failed];
	tp = {x, y} /. tp;
	tp = Cases[tp, {a_, b_} /; Abs[a] < 10^3 && Abs[b] < 10^3];
	
	xdist = Abs @ Apply[Subtract, {Min[#], Max[#]}& @ First[Transpose[tp]]] /. 0 -> 1;
	ydist = Abs @ Apply[Subtract, {Min[#], Max[#]}& @ Last[Transpose[tp]]] /. 0 -> 1;

	{{x, xc - 3 xdist, xc + 3 xdist}, {y, yc - 3 ydist, yc + 3 ydist}}
]
	
constantCoeff[expr_, form_] := Plus @@ Cases[expr, c_. form /; VarList[c] === {} :> c]

hyperbolaQ[expr_Equal, {x_, y_}] := hyperbolaQ[Subtract @@ expr, {x, y}]

hyperbolaQ[expr_, {x_, y_}] := PolynomialQ[expr, {x, y}] && Exponent[expr, {x, y}] === {2, 2} && 
  	TrueQ[
  		Det[{{constantCoeff[expr, x^2], constantCoeff[expr, x y]}, {constantCoeff[expr, x y], constantCoeff[expr, y^2]}}] < 0
  	]

  	
(*
In[439]:= hyperbolaQ[x^2 + 3 x y - 2 y^2 + 3 x == 0, {x, y}]
Out[439]= True
*)

implicitPlotRange[a_. (x_ + x0_.)^2 + b_. (y_ + y0_.)^2 == r_, {x_, y_}] /; 
	FreeQ[{a, b, r, x0, y0}, x | y] && (b < 0 || a < 0) && r > 0 := 
	{
		{x, -Sqrt[r/Abs[a]] - Abs[1/a] - x0, Sqrt[r/Abs[a]] + Abs[1/a] - x0}, 
		{y, -Sqrt[r/Abs[b]] - Abs[1/b] - y0, Sqrt[r/Abs[b]] + Abs[1/b] - y0}
	}

implicitPlotRange[a_. x_Symbol^n_?EvenQ + b_. y_Symbol^m_?EvenQ == 0, {x_, y_}] := $Failed

rangesToPoints[{{xlo_, xhi_}, {ylo_, yhi_}}, {x_, y_}] := {{x -> xlo, y -> ylo}, {x -> xhi, y -> yhi}}

containsBivariateLinearQ[eqns_, {x_, y_}] := 
	TrueQ @ Apply[Or, MatchQ[#, a_. x + b_. y + c_. /; FreeQ[{a, b, c}, x | y]] & /@ Subtract @@@ eqns]

(*
In[347]:= containsBivariateLinearQ[{5 x + 3 y^2 == 20, 3 x + 5 y == 100}, {x, y}]
Out[347]= True
*)


(* ::Section::Closed:: *)
(*turningPoints*)


turningPoints[l_List, {x_, y_}] := Flatten[turningPoints[#, {x, y}] & /@ l, 1]

turningPoints[eqns_, {x_, y_}] := Quiet @ Block[{yxsol, xysol, DyDx, xs = {}, DxDy, ys = {}, points},
	yxsol = Solve[Dt[eqns, x], Dt[y, x]];
	xysol = Solve[Dt[eqns, y], Dt[x, y]];
	
	(* find where dy/dx == 0 *)
	If[MatchQ[yxsol, {{__Rule}..}],
		DyDx = Dt[y, x] /. yxsol;
		xs = Solve[{eqns, DyDx == 0}, {x, y}]
	];
	
	(* find where dx/dy == 0 *)
	If[MatchQ[xysol, {{__Rule}..}],
		DxDy = Dt[x, y] /. xysol;
		ys = Solve[{eqns, DxDy == 0}, {y, x}]
	];
	
	points = Join @@ Cases[{xs, ys}, {{(_ -> _?NumericQ)..}..}];
	
	Select[points, FreeQ[#, _Complex]&] /. r:{y -> _, x -> _} :> Reverse[r]
]

(*
In[141]:= turningPoints[x y - y == x^2/8 + y^2, {x, y}]

Out[141]= {{x -> 0, y -> 0}, {x -> 4, y -> 1}, {x -> 2 - Sqrt[2], 
  y -> 1/2 (1 - Sqrt[2])}, {x -> 2 + Sqrt[2], y -> 1/2 (1 + Sqrt[2])}}
*)


(* ::Chapter::Closed:: *)
(*CalculateScan`StepByStepMath`StepByStepPlotAnalysis`*)


(* ::Section::Closed:: *)
(*chooseinterval*)


Clear[chooseinterval]
chooseinterval[expr_,x_,per_]:=If[NumericQ@Quiet[expr/.x->(per/2)],0<=x<per,-per/2<=x<per/2]


(* ::Section::Closed:: *)
(*realFunctionQ*)


(* ::Subsubsection::Closed:: *)
(*realFunctionQ*)


realFunctionQ[expr_] := VectorQ[Cases[expr, _?NumericQ, {0, \[Infinity]}], #\[Element]Reals&]



(* ::Section::Closed:: *)
(*refineCriticalPoints*)


(* ::Subsubsection:: *)
(*refineCriticalPoints*)


refineCriticalPoints[True, x_] := True
refineCriticalPoints[r_Reduce, x_] := r
(*refineCriticalPoints[roots_, x_] := Or @@ And @@@ Join @@@ CombineModuli[SensibleResults[roots, {x}, GeneratedParameters -> SBSSymbol["m"]], GeneratedParameters -> SBSSymbol["m"]]*)
refineCriticalPoints[roots_, x_]:= roots (* temp hack*)


(* ::Section::Closed:: *)
(*SortAndOr*)


SortAndOr[expr_] := Module[{$And, $Or}, iSortAndOr[expr /. {And -> $And, Or -> $Or}, $And, $Or] /. {$Or -> Or, $And -> And}]
iSortAndOr[expr_, and_, or_] := expr //. {(h:(and|or))[a__] /; FreeQ[{a}, and|or] :> If[h === and, And, Or] @@ DeleteDuplicates[SortBy[{a}, N]]}


(* ::Section::Closed:: *)
(*OrList*)


OrList[HoldPattern[Or][args__]] := {args}
OrList[e_] := {e}


(* ::Section::Closed:: *)
(*inequalityBoundary*)


inequalityBoundary[expr_, x_] :=
	Block[{ints},
		ints = inequalitiesToIntervals[expr, x];
		(
			Or @@ Thread[x == DeleteDuplicates[Select[Join @@ ints[[All, 1]], NumericQ]]]
			
		) /; MatchQ[ints, {HoldPattern[Interval[{_, _}]]..}]
	]


inequalityBoundary[___] = False;


(* ::Section::Closed:: *)
(*iInequalitiesToIntervals*)


(* ::Subsubsection::Closed:: *)
(*inequalitiesToIntervals*)


inequalitiesToIntervals[ineq_, x_] := Module[{red = lightReduce[ineq, x], res, exactNumbers},
	exactNumbers = Cases[ineq, _?(NumericQ[#] && !InexactNumberQ[#] &), {0, Infinity}];
	If[red === $Failed,
		red = ineq
	];
	red = red /. Thread[N[exactNumbers] -> exactNumbers];
	res = NumericalSort[Flatten[{iInequalitiesToIntervals[red, x]}]];
	If[res === $Failed,
		$Failed,
		res
	]
]


lightReduce[ineq_, x_] :=
	Block[{roots, res, torad},
		res = Reduce[ineq/. {r:(_Root|_RootSum) :> r, x0_Real :> Rationalize[x0, 0]}, x, Reals];
		(
			roots = Cases[ineq, _Root|_RootSum, \[Infinity]];
			torad = Append[Thread[roots -> roots], r:(_Root|_RootSum) :> ToRadicals[r]];
			
			N[Simplify`QuickSimplify[res /. torad], Precision[ineq]]
			
		) /; Head[res] =!= Reduce
	]

lightReduce[___] = $Failed;


iInequalitiesToIntervals[True, x_] := Interval[{-\[Infinity], \[Infinity]}]
iInequalitiesToIntervals[(Greater|GreaterEqual)[x_, a_], x_] /; FreeQ[a, x] := Interval[{a, \[Infinity]}]
iInequalitiesToIntervals[(Greater|GreaterEqual)[a_, x_], x_] /; FreeQ[a, x] := Interval[{-\[Infinity], a}]
iInequalitiesToIntervals[(Less|LessEqual)[x_, a_], x_] /; FreeQ[a, x] := Interval[{-\[Infinity], a}]
iInequalitiesToIntervals[(Less|LessEqual)[a_, x_], x_] /; FreeQ[a, x] := Interval[{a, \[Infinity]}]
iInequalitiesToIntervals[Equal[x_, a_], x_] /; FreeQ[a, x] := Interval[{a, a}]
iInequalitiesToIntervals[Equal[a_, x_], x_] /; FreeQ[a, x] := Interval[{a, a}]


iInequalitiesToIntervals[(Greater|GreaterEqual)[a_, x_, b_], x_] /; FreeQ[{a, b}, x] := Interval[{a, b}]
iInequalitiesToIntervals[(Less|LessEqual)[b_, x_, a_], x_] /; FreeQ[{a, b}, x] := Interval[{a, b}]


iInequalitiesToIntervals[HoldPattern[Inequality][a_, h1_, x_, h2_, b_], x_] /; FreeQ[{a, b}, x] := iInequalitiesToIntervals[h1[a, x] && h2[x, b], x]


iInequalitiesToIntervals[HoldPattern[And][e__], x_] := Apply[IntervalIntersection, Flatten[iInequalitiesToIntervals[#, x]& /@ {e}]]
iInequalitiesToIntervals[HoldPattern[Or][e__], x_] := iInequalitiesToIntervals[#, x]& /@ {e}


iInequalitiesToIntervals[__] := $Failed;


(* ::Section::Closed:: *)
(*chooseTestPoint*)


(* ::Subsubsection::Closed:: *)
(*chooseTestPoint*)


chooseTestPoint[Interval[{a_, b_}]] /; Element[{a/\[Pi], b/\[Pi]}, Rationals] := \[Pi] chooseTestPoint[Interval[{a/\[Pi], b/\[Pi]}]]


chooseTestPoint[i_Interval] := Block[{low, high, highFloor, lowCeil},

	{low, high} = First[i];
	{lowCeil, highFloor} = {Ceiling[low], Floor[high]};
	
	Quiet @ Which[
		TrueQ[low < 0 < high],
		0,

		low === -Infinity,
		If[high == highFloor, highFloor-1, highFloor],
		
		high === Infinity,
		If[low == lowCeil, lowCeil+1, lowCeil],

		TrueQ[Abs[low - high] > 1 || (Abs[low - high] == 1 && (low != lowCeil || high != highFloor))],
		If[Abs[lowCeil] < Abs[highFloor], 
			If[low == lowCeil, lowCeil+1, lowCeil],
			If[high == highFloor, highFloor-1, highFloor]
		],

		TrueQ[low < lowCeil < high],
		lowCeil,

		TrueQ[low < highFloor < high],
		highFloor,
		
		(* prefer a rational with denominator 2, then 4, 5, 10, 8, 3, 6, 9 *)
		TrueQ[low < (Floor[2low]+1)/2 < high],
		N[(Floor[2low]+1)/2, Precision[i]],
		
		TrueQ[low < (Floor[4low]+1)/4 < high],
		N[(Floor[4low]+1)/4, Precision[i]],
		
		TrueQ[low < (Floor[5low]+1)/5 < high],
		N[(Floor[5low]+1)/5, Precision[i]],
		
		TrueQ[low < (Floor[10low]+1)/10 < high],
		N[(Floor[10low]+1)/10, Precision[i]],
		
		TrueQ[low < (Floor[8low]+1)/8 < high],
		N[(Floor[8low]+1)/8, Precision[i]],
		
		TrueQ[low < (Floor[3low]+1)/3 < high],
		N[(Floor[3low]+1)/3, Precision[i]],
		
		TrueQ[low < (Floor[6low]+1)/6 < high],
		N[(Floor[6low]+1)/6, Precision[i]],
		
		TrueQ[low < (Floor[9low]+1)/9 < high],
		N[(Floor[9low]+1)/9, Precision[i]],

		True,
		(low+high)/2
	]
]


(* ::Section::Closed:: *)
(*inequalityClosure*)


inequalityClosure[expr_] := expr /. {Less -> LessEqual, Greater -> GreaterEqual}


(* ::Section::Closed:: *)
(*niceN*)


(* ::Subsubsection::Closed:: *)
(*niceN*)


SetAttributes[niceN, Listable];
niceN[n_] := Module[{niceNumbersAtoms, niceNumbersPattern},
	niceNumbersAtoms = _Integer | (Rational[num_Integer, denom_Integer] /; num < 50 && denom < 50) | (Sqrt[s_Integer] /; 0<s<100) | (HoldPattern[Power[s_Integer, Rational[1, 2|3]]] /; 0<s<100);
	niceNumbersPattern = niceNumbersAtoms | (HoldPattern[Plus[a1_, a2_]] /; VectorQ[{a1, a2}, MatchQ[#, niceNumbersAtoms|Times[-1, niceNumbersAtoms]]&] && Sign[a1]===Sign[a2]);
	If[MatchQ[n, niceNumbersPattern], n, N[n]]
]


(* ::Section::Closed:: *)
(*signSymbol*)


(* ::Subsubsection::Closed:: *)
(*signSymbol*)


SetAttributes[signSymbol, Listable];


signSymbol[Row[{x_, ___}, ___]] := signSymbol[x]


signSymbol[n_] := Sign[n] /. {1->"+", -1->"\[Dash]"}


(* ::Chapter::Closed:: *)
(*CalculateUtilities`UserVariableUtilities*)


(* ::Section::Closed:: *)
(*userSymbolQ*)


ClearAll[usersymbolQ]
usersymbolQ::usage = "Gives True for symbols in the Global` context and False otherwise.";
usersymbolQ[x_Symbol]:=Context[x]=="Global`"
usersymbolQ[_]:=False


(* ::Section::Closed:: *)
(*VarList*)


ClearAll[VarList]
VarList::usage = "VarList[expr] yields a list of all user variables in expr (returns {n, x} for Sin[n x])";
Options[VarList] = {"Expose"->False};
(*Attributes[VarList] = {HoldFirst};*)
VarList[expr_, OptionsPattern[]] := Module[{nonsubs, subs, subbases, localsymbol, subpattern},
	If[TrueQ@OptionValue["Expose"],
		subs = Union[Cases[Hold[expr], (Subscript|Superscript)[_Symbol?usersymbolQ|_String|_Integer, _], Infinity]];
		subpattern = Subscript|Superscript,
		subs = Union[Cases[Hold[expr], Subscript[_Symbol?usersymbolQ, _], Infinity]];
		subpattern = Subscript
	];
	nonsubs = Union[Cases[Hold[expr] /. subpattern[__] :> Unique[localsymbol], _Symbol?usersymbolQ, Infinity]];
	(* The non-expose version of VarList needs to treat Subscript[x, 1] and x as being dependent, not independent variables *)
	If[!TrueQ@OptionValue["Expose"],
		subbases = Union[Cases[subs, subpattern[x_Symbol,_]:>x]];
		nonsubs = Complement[nonsubs, subbases]
	];
	subs = Cases[ subs, v:subpattern[a_, _] /;
		StringQ[a] ||
		StringMatchQ[ToString[a], {_, _~~NumberString}] ||
		StringLength[ToString[a]]>1 && Count[expr, v, {0,Infinity}]>1
	];
	nonsubs = Cases[ nonsubs, a_ /; With[{n = ToString[a]},
		StringMatchQ[n, {_, _~~NumberString}] ||
		StringMatchQ[n, "QuestionMark"~~NumberString] || 
		StringMatchQ[n, "EmptySquare"~~NumberString] ||
		StringMatchQ[n, "Blank"~~NumberString] ||
		StringLength[n]>1 && Count[expr, a, {0,Infinity}]>1]
	];
	Union[subs, nonsubs]
]


(* ::Section::Closed:: *)
(*newFreeVariable*)


ClearAll[newFreeVariable];
newFreeVariable::usage = "finds a new symbol that doesn't appear in the first argument";
Options[newFreeVariable]={"Preference"->{}};
newFreeVariable[existingexprs_, opts:OptionsPattern[]]:= newFreeVariable[existingexprs,1,opts][[1]]
newFreeVariable[existingexprs_, n_Integer, opts:OptionsPattern[]]:= 
	((*Print[OptionValue["Preference"],Symbol/@Join[OptionValue["Preference"],{"x", "y", "z", "u", "v", "w", "r", "s", "t", "p", "q", "m", "n"}], Cases[{existingexprs},_Symbol,Infinity]];*) 
	Select[Symbol/@Join[OptionValue["Preference"],{"x", "y", "z", "u", "v", "w", "r", "s", "t", "p", "q", "m", "n"}],!MemberQ[Cases[{existingexprs},_Symbol,Infinity],#]&]
	)[[1;;Min[13,n]]]


(* ::Section::Closed:: *)
(*ImplicitFuncQ*)


(*TODO: make option for ignoring symbols known to be constant parameters, etc*)
ClearAll[ImplicitFuncQ]
ImplicitFuncQ::usage = "tests whether the argument is an implicit function" (*TODO: make option for ignoring symbols known to be constant parameters, etc*);
Options[ImplicitFuncQ] = {"NonVariables" -> {}};
Attributes[ImplicitFuncQ] = {HoldFirst};
ImplicitFuncQ[a_List == b_List]:= ImplicitFuncQ[Thread[a==b]]
ImplicitFuncQ[Equal[a_,b_,c_]]:= ((*Print["PLP"];*)ImplicitFuncQ[Equal[a,b]] || ImplicitFuncQ[Equal[b,c]] || ImplicitFuncQ[a,c])
ImplicitFuncQ[Equal[a_,b_]]:=!(MatchQ[a,(_?usersymbolQ)[__?usersymbolQ] | _?usersymbolQ | Subscript[_?usersymbolQ, _] | Subscript[_?usersymbolQ, _][__Symbol]] && FreeQ[b,a,Infinity]) && !(MatchQ[b,(_?usersymbolQ)[__?usersymbolQ] | _?usersymbolQ | Subscript[_?usersymbolQ, _] | Subscript[_?usersymbolQ, _][__Symbol]] && FreeQ[a,b,Infinity])
ImplicitFuncQ[__]:=False;


(* ::Section::Closed:: *)
(*chooseVariables & chooseVariablesNonParametric*)


(* ::Subsection::Closed:: *)
(*code*)


(* find the user specified variables *)
(*SetAttributes[chooseVariables, HoldFirst];*)
ClearAll[chooseVariables];
chooseVariables::usage = "chooseVariables[expr, preprocessedexpr (optional), options] returns a list of the user-specified dependent variables (returns {x} for Sin[n x]). 
							If the option \"ReturnDependent\" is set to True, it returns {\"Independents\" -> uservars_List, \"Dependents\" -> userdependentvariables_List}. 
							If \"ReturnParameters\" is set to True, it returns {\"Independents\" -> uservars_List, \"Parameters\" -> params_List}
							To be used alongside preprocessUserInput.";
chooseVariablesNonParametric::usage = "like chooseVariables for when we want to interpret lists not as parametric functions, but as a list of fns w/ same domain and range";

(* 
	In[222]:= chooseVariables[Abs[a] < 1/Abs[z]]
	Out[222]= {z} 
*)
Options[chooseVariables] = {
	"NoFail" -> True,
	"ReturnDependent" -> False, 
	"ReturnParameters" -> False, 
	"Constraints" -> {}, (*I think this is just to get the other possible variables*)
	"IgnoreArgumentsInDepVar"->False (*True is old behavior, just in case*),
	"ReturnThisManyIndeps" -> Automatic,	
	"PossibleDependents" -> {}, (*If the set of dependent variables is underspecified, Use these variables, in this order, as many as needed *)
	"WithArgsInDepVars"->Automatic  (*this is not used, nor does it make sense*)
};

(*special "option" for when we want to interpret lists not as parametric functions, but as a list of fns w/ same domain and range*)
(*TODO:? make sure variables dont appear as both ind. and dep.!*)
chooseVariablesNonParametric[expr_List, opt__] := 
		(
		{"Independents"->DeleteDuplicates@Flatten[#[[1,All,2]]],"Dependents"->DeleteCases[DeleteDuplicates@Flatten[#[[2,All,2]]],f_/;MemberQ[Flatten[#[[1,All,2]]],f]]}
		)&[Transpose[If[StringQ[#],{"Independents"->{},"Dependents"->{}},chooseVariables[#,opt]]&/@expr]]
(*for SoR: sometimes we pass just "axis"*)
(*chooseVariables[_String, opts: OptionsPattern[]] := {"Independents"->{},"Dependents"->{}}*)

(*get rid of HoldForm*)
chooseVariables[HoldForm[expr_], opts : OptionsPattern[]] := chooseVariables[expr, opts]

(*implicit funcs*)
chooseVariables[expr_?ImplicitFuncQ, opts : OptionsPattern[]] := chooseVariables[expr, preprocessUserInput[expr, OptionValue["NoFail"]], opts]

(* get the right number of arguments! *)
chooseVariables[expr_, opts : OptionsPattern[]] := chooseVariables[expr, preprocessUserInput[expr, "NoFail" -> OptionValue["NoFail"]], opts]

(* example: expr is {x, y} == {t, t^2} and fn is {t, t^2} *)
chooseVariables[expr : Equal[a_List, b_List], fn_List, opts : OptionsPattern[]] /; TrueQ[Length[a] == Length[b]] := chooseVariables[#, fn, opts] & [Thread[expr]]

(* example: expr is x^2 == f[x], fn is x^2 *)
chooseVariables[expr : Equal[rhs_, (f_?usersymbolQ)[indeps__Symbol]], fn_, opts : OptionsPattern[]] /; !MatchQ[rhs,(_?usersymbolQ)[__Symbol]] := chooseVariables[Equal[f[indeps], rhs], fn, opts]

(* example:  expr is {x == t, y == t^2} and fn is {t, t^2}*)
chooseVariables[origexpr_List, fn_List, opts : OptionsPattern[]] /; TrueQ[Length[origexpr] == Length[fn]] := 
If[TrueQ[OptionValue["ReturnDependent"]],
	Module[{
			deps,
			possdeps = Select[DeleteDuplicates[OptionValue["PossibleDependents"]], FreeQ[{origexpr},#]&],
			expr, firstrun
			},
		DebugPrint["possdeps:  ",possdeps];
		If[SupersetQ[possdeps,"Dependents" /. #], 
			DebugPrint["  w/o possdeps: ","Dependents" /. #]; {"Independents"->chooseVariables[origexpr, fn (*, DeleteCases[opts,("ReturnDependent"->_)|("PossibleDependents"->_)]*)], #}
			,
			expr = If[FreeQ[{origexpr},_Equal],
				MapIndexed[(If[Length[possdeps]>=#2[[1]],possdeps[[#2[[1]]]]==#1,#1])&,origexpr],
				origexpr
			];
			DebugPrint["expr",expr,"  w/o possdedeps: ",#];
			firstrun = {DeleteDuplicates[Flatten[#[[All, 1]]]], #[[All, -1]]} & [MapThread[chooseVariables[#1, #2, opts] &, {expr, fn}]] /. exp : {(type_ -> _)..} :> type -> DeleteDuplicates[Flatten[exp[[All, -1]]]];
			DebugPrint["firstrun",firstrun];
			If[Length[deps = "Dependents" /. firstrun] < Length[expr], deps = DeleteDuplicates[Join[deps,("Dependents" /. (getDependents[expr, #] &[Flatten["Independents" /. firstrun]]))]][[1;;Length[expr]]]];
			DebugPrint["deps",deps, getDependents[expr, #] &[Flatten["Independents" /. firstrun]],  List["Dependents" /. firstrun,("Dependents" /. (getDependents[expr, #] &[Flatten["Independents" /. firstrun]]))[[1;;Length[expr] - Length["Dependents" /. firstrun]]]]];
			firstrun /. ("Dependents" -> a_) :> ("Dependents" -> deps) /. Rule[a_, {{b__}}] :> Rule[a, {b}]
		]&[getDependents[origexpr, chooseVariables[origexpr]]] 
	],
	DeleteDuplicates[Flatten[MapThread[chooseVariables[#1, #2, opts] &, {origexpr, fn}]]]
]

chooseVariables[expr_, $Failed, opts : OptionsPattern[]] := $Failed

SetAttributes[grabIndeps, HoldFirst];
grabIndeps[expr : Equal[(f_?usersymbolQ)[indeps:(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol|Subscript[_Symbol, __]|Subscript[_CalculateParse`ExportedFunctions`CalculateSymbol, __])..], rhs_], else_, cons_] := {
			"Independents" -> If[!TrueQ@OptionValue["IgnoreArgumentsInDepVar"], 
								{indeps},
								Union @ Flatten[{Select[{indeps}, !FreeQ[Hold[rhs], #] &], pickvars[OptionValue["Constraints"], fn]}]
							]
}

grabIndeps[expr_, input_, cons_] := {
			"Independents" -> Union @ Flatten[{pickvars[expr, input], VarList[OptionValue["Constraints"]]}]
}

(* example: expr is f[x] == x^2, fn is x^2 *)
(* example: expr is y == x^2, fn is x^2 *)
chooseVariables[expr_, fn_, opts : OptionsPattern[]] := Module[{res, inds, possdep=Select[OptionValue["PossibleDependents"],FreeQ[expr,#]&]}, 
	(*Print[{opts},"IgnoreArgumentsInDepVar ", OptionValue["IgnoreArgumentsInDepVar"],"--  ",possdep];*)
	(* The line below will return {x} for inputs like f[x, y] == x *)
	res = grabIndeps[expr, fn, OptionValue["Constraints"]];
	inds = "Independents" /. res;
	
	(* This will return {x, y} for inputs like f[x, y] == x *)
	(* inds = Union @ Flatten[{{indeps}, pickvars[OptionValue["Constraints"], fn]}]; *)
	
	If[TrueQ[OptionValue["ReturnDependent"]], AppendTo[res, getDependents[If[FreeQ[{#},_Equal] && possdep=!={},possdep[[1]]==#,#]&@expr, inds]]];

	If[TrueQ[OptionValue["ReturnParameters"]], AppendTo[res, getParameters[expr, inds]]];
	
	If[Length[res] == 1, Return[inds]];
	If[TrueQ[$AlphaDevModeQ],
		postshuffle[res, OptionValue["ReturnThisManyIndeps"]],
		res
	]
]


(* ::Subsection::Closed:: *)
(*tests*)


(* ::Input:: *)
(*chooseVariables[a x^2+b x+c,"ReturnDependent"->True,"ReturnParameters"->True]*)


(* ::Input:: *)
(*chooseVariables[n^k,"ReturnDependent"->True,"ReturnParameters"->True]*)


(* ::Input:: *)
(*chooseVariables[k^n,"ReturnDependent"->True,"ReturnParameters"->True]*)


(* ::Input:: *)
(*chooseVariables[a x^2+b x+c]*)


(* ::Input:: *)
(*chooseVariables[\[Rho]==\[Theta]]*)


(* ::Input:: *)
(*chooseVariables[\[Rho]==\[Theta],"ReturnDependent"->True]*)


(* ::Input:: *)
(*chooseVariables[\[Rho]==\[Theta] x,"ReturnDependent"->True]*)


(* ::Input:: *)
(*(chooseVariables[#1,"ReturnDependent"->True,"ReturnParameters"->True]&)[Cos[m x] Sin[n x y]]*)


(* ::Section::Closed:: *)
(*preprocessUserInput and preprocessUserInputHeld*)


(* ::Subsection:: *)
(*code*)


preprocessUserInput::usage = "removes y= and f(x)= from the user-input function";
preprocessUserInputHeld::usage = "removes y= and f(x)= from the user-input function. Maintains the expr in held form.";


(* code to process user-input functions *)
ClearAll[preprocessUserInput];
Attributes[preprocessUserInput] = {Listable};
Options[preprocessUserInput] = {"NoFail"->False};

(*BUG: 
In[2673]:= CalculateUtilities`UserVariableUtilities`chooseVariables[
 Hold[y == x - x^7], "ReturnDependent" -> True]

Out[2673]= {"Independents" -> {x, y}, "Dependents" -> {z}}*)
(* example: {x, y} == {t, t^2} *)
preprocessUserInput[expr : Equal[exprlhs_List, exprrhs_List], opt:OptionsPattern[]] /; TrueQ[Length[exprlhs] == Length[exprrhs]] := 
	 preprocessUserInput[If[TrueQ[#[[1]]==#[[2]]],Symbol["repdepvar00"<>ToString[#[[1]]]]==#[[2]],#[[1]]==#[[2]]]&/@Transpose[{exprlhs,exprrhs}]]

(* example: f[x] == x *)
preprocessUserInput[expr_, opt:OptionsPattern[]] := (DebugPrint[expr]; Which[
  MatchQ[expr, Equal[(a_?usersymbolQ)[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol|Subscript[_Symbol, __]|Subscript[_CalculateParse`ExportedFunctions`CalculateSymbol, __])..] | a_Symbol | a_CalculateParse`ExportedFunctions`CalculateSymbol | a_?(MatchQ[#, Subscript[_Symbol, _]]&) | a_?(MatchQ[#, (Subscript[_Symbol, _])[__Symbol]]&) , b__?(FreeQ[{#}, (_?usersymbolQ)[__Symbol]]&)] /; ((*Print[b,a," ++ ",FreeQ[b,If[Head[Head[a]]===Subscript,Head[a],a]]," ++ ",If[Head[Head[a]]===Subscript,Head[a],a]];*) FreeQ[b,If[Head[Head[a]]===Subscript,Head[a],a]])],
  	Last[expr],
  MatchQ[expr, Equal[b__?(FreeQ[{#}, (_?usersymbolQ)[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol)..]]&), (a_?usersymbolQ)[__Symbol] | a_Symbol] /; FreeQ[b,a] && !NumericQ[a]],
  	First[expr],
  MatchQ[Head[expr], Equal | Less | Greater | GreaterEqual | LessEqual],
  	If[OptionValue["NoFail"],expr,$Failed],
  True,
  	expr
  ])

preprocessUserInput[Equal[a_, b_, c__], opt:OptionsPattern[]] := With[{pre = preprocessUserInput[a == b, opt]},
	If[pre === $Failed,
		$Failed,
		If[MemberQ[preprocessUserInput[Equal[a, #], "NoFail" -> True] & /@ {b, c}, _Equal],
			Equal[a, b, c],
			Replace[Equal[pre, c], e_Equal :> Flatten[e, 1, Equal], {0}]
		]
	]
]
preprocessUserInput[expr___, opt:OptionsPattern[]] := If[OptionValue["NoFail"], expr, $Failed]



Attributes[preprocessUserInputHeld] = {HoldFirst};
Options[preprocessUserInputHeld] = {"NoFail"->False};

(* example: Hold[{x, y} == {t, t^2}] *)
preprocessUserInputHeld[Hold[a_List == b_List], opt:OptionsPattern[]] /; Length[a] == Length[b] :=
	preprocessUserInputHeld /@ ReplaceAll[
		MapThread[Hold[#1 == #2]& , {Hold /@ Unevaluated[a],  Hold/@ Unevaluated[b] }],Equal[Hold[r_], Hold[s_]] :> Equal[r,s]
	]

(* example: Hold[f[x] == x + x] *)
preprocessUserInputHeld[expr_Hold, opt:OptionsPattern[]] := Which[
  	MatchQ[expr, Hold @ Equal[a_?usersymbolQ[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol)..] | a_Symbol | a_CalculateParse`ExportedFunctions`CalculateSymbol | a_?(MatchQ[#, Subscript[_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol, _] | (Subscript[_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol, _])[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol)..]]&), b__?(FreeQ[{#}, (_?usersymbolQ)[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol)..]]&)] /; (FreeQ[b,If[Head[Head[a]]===Subscript,Head[a],a]])],
  		Extract[expr, {1, -1}, Hold],
  	MatchQ[expr, Hold @ Equal[b__?(FreeQ[{#}, (_?usersymbolQ)[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol)..]]&), (a_?usersymbolQ)[(_Symbol|_CalculateParse`ExportedFunctions`CalculateSymbol)..] | a_CalculateParse`ExportedFunctions`CalculateSymbol | a_Symbol] /; FreeQ[b,a] && !NumericQ[a]],
  		Extract[expr, {1, 1}, Hold],
  	MatchQ[Extract[expr, {1, 0}], Equal | Less | Greater | GreaterEqual | LessEqual],
  		If[OptionValue["NoFail"], expr, $Failed],
  	True,
  		expr
]

preprocessUserInputHeld[expr___, opt:OptionsPattern[]] := If[OptionValue["NoFail"], expr, $Failed]


(* ::Subsection::Closed:: *)
(*tests*)


(* ::Input:: *)
(*preprocessUserInput[Sin[x]]*)


(* ::Input:: *)
(*preprocessUserInput[Sin[x] == y]*)


(* ::Input:: *)
(*preprocessUserInput[x == y]*)


(* ::Input:: *)
(*preprocessUserInput[y == x]*)


(* ::Input:: *)
(*preprocessUserInput[{x, y} == {t + t, t^2}]*)


(* ::Input:: *)
(*preprocessUserInputHeld[Hold[{x, y} == {t + t, t^2}]]*)


(* ::Input:: *)
(*preprocessUserInputHeld[Hold[x == t + t]]*)


(* ::Section::Closed:: *)
(*prettyFunction*)


prettyFunction::usage = "formats a math function in Row, with LongEquals, and adds a dependent variable in gray, if necessary";
(*prettyFunction is usually called after chooseVariables. It formats a math function in Row, with LongEquals, and adds a dependent variable in gray, if necessary*)
ClearAll[prettyFunction];
Clear[getYEqual];
Attributes[prettyFunction] = {HoldFirst};
Attributes[getYEqual] = {HoldFirst};
Options[prettyFunction] = { (*ONLY "WithArgsInDepVars" is hooked up, see MathTestFunc[] DebuggingUtilities.m*)
	"Wrapper" -> MatrixForm, (*by default, lists are wrapped in MatrixForm*)
	"Threaded" -> Automatic, (*by default, lists are kept in the form passed to prettyFunction, but can be changed to/from {_,_}=={_,_} or {_==_,_==_}*)
	"WithArgsInDepVars" -> Automatic, (*by default, depvars are kept in the form passed, either y or y[t]*) 
	"ReverseAcrossEquals" -> True, (*by default, lists are kept in the form passed to prettyFunction, either y = f[x] or f[x] = y*)
	"ExtraDependent"-> None,  (* "ExtraDependent"-> y adds a "y=" to the function *)
	"ReplacementDependent"-> None, (*"ReplacementDependent"-> y adds a "y=" to the function ONLY IF Head[function]=!=Equal already *)
	"PossibleDependents" -> {}
	};

prettyFunction[(originalform|HoldForm)[ex_],a___]:=prettyFunction[ex,a];
prettyFunction[Equal[f_?(!usersymbolQ),g_?usersymbolQ], a___, opts:OptionsPattern[]] := (DebugPrint["::",OptionValue["ReverseAcrossEquals"]]; prettyFunction[#,a,opts]&[If[OptionValue["ReverseAcrossEquals"],g==f,f==g]]);
prettyFunction[ex:{Equal[__]..},a___]:=(DebugPrint["Threading..",FullForm@ex]; prettyFunction[#,a]&[Thread[ex,Equal]]);

prettyFunction[expr_, "ExtraDependent"-> yy_Symbol, opts:OptionsPattern[]] := (DebugPrint["exta depvar ",yy]; 
	If[(!FreeQ[{expr},yy] || !FreeQ[yy,None|$Failed]),
		#,
		Row[{CalculateScan`CommonSymbols`GrayText[yy], CalculateScan`CommonSymbols`GrayText[" \[LongEqual] "], #}]
	]&[prettyFunction[expr,opts]]);
	
prettyFunction[expr_, depvar_?(Head[#]=!=Rule&), "ExtraDependent"-> yy_Symbol, opts:OptionsPattern[]] := (DebugPrint["exta depvar2 ",yy, depvar]; 
	If[(!FreeQ[{expr},yy] || (depvar==={yy} && FreeQ[{expr},Equal]) || !FreeQ[yy,None|$Failed]),
		#,
		Row[{CalculateScan`CommonSymbols`GrayText[yy], CalculateScan`CommonSymbols`GrayText[" \[LongEqual] "], #}]
	]&[prettyFunction[expr,depvar, opts]]);

prettyFunction[expr_, "ReplacementDependent"-> yy_Symbol, opts:OptionsPattern[]] := (DebugPrint["replacement depvar: ",yy,FreeQ[{expr},Equal]]; 
	If[(*(!FreeQ[{expr},yy] || !FreeQ[yy,None|$Failed])*)(!FreeQ[{expr},yy] || !FreeQ[{expr},Equal] || !FreeQ[yy,None|$Failed]),
		#,
		prettyFunction[expr,yy,opts](*Row[{CalculateScan`CommonSymbols`GrayText[yy], CalculateScan`CommonSymbols`GrayText[" \[LongEqual] "], #}]*)
	]&[prettyFunction[expr,opts]]);

prettyFunction[expr_, depvar_?(Head[#]=!=Rule&), "ReplacementDependent"-> yy_Symbol, opts:OptionsPattern[]] := (DebugPrint["replacement depvar2 ",yy, depvar]; 
	If[(!FreeQ[{expr},yy] || !FreeQ[#,Equal|"\[LongEqual]"|" \[LongEqual] "]),
		#,
		(*not sure if depvar or yy takes precedence*)
		Row[{CalculateScan`CommonSymbols`GrayText[yy], CalculateScan`CommonSymbols`GrayText[" \[LongEqual] "], #}]
	]&[prettyFunction[expr,depvar, opts]]);


prettyFunction[expr_, opts:OptionsPattern[]] /; FreeQ[{opts},"ExtraDependent",Infinity] := With[{ArgInDepVar = If[#===True,If[ImplicitFuncQ[expr],False,If[#==={},Select[Symbol/@{"x","t","u","v"},FreeQ[expr,#]&,1],#]]&@chooseVariables[expr,opts],#]&["WithArgsInDepVars"/.{opts}]}, DebugPrint["ArgInDepVvar",ArgInDepVar,"==",opts,"==",expr,"==="];
	Return@If[MatchQ[expr,HoldPattern[Equal[__]]|HoldPattern[{Equal[__]..}]], 
				
				If[MatchQ[ArgInDepVar,{__Symbol}] && !MatchQ[expr[[1]],(_?usersymbolQ)[__Symbol]],
					HoldForm[#]&@
					(If[Length[expr[[1]]]>1,
						(#@@ArgInDepVar)&/@expr[[1]],
						expr[[1]]@@ArgInDepVar
					]==expr[[2]]),
					HoldForm@expr
				]/.{a_List:>MatrixForm[a]}, 
				getYEqual[expr,opts]
			];
];

(*could maybe use this version to save a little time if you already ran chooseVariables*)
prettyFunction[expr_, y_?(Head[#]=!=Rule&), opts:OptionsPattern[]]/; FreeQ[opts,"ExtraDependent",Infinity] := With[{ArgInDepVar = If[#===True,If[ImplicitFuncQ[expr],False,If[#==={},Select[Symbol/@{"x","t","u","v"},FreeQ[expr,#]&,1],#]]&@chooseVariables[expr],#]&["WithArgsInDepVars"/.{opts}]}, DebugPrint[If[Head[y]===Symbol && MatchQ[ArgInDepVar,{_Symbol}|{__Symbol}], y@@ArgInDepVar, y],"  ArgInDepVarY   ",ArgInDepVar,"==",opts,"==",expr,"==",y];
	Return@If[MatchQ[Head[expr],Equal], DebugPrint["lslfff"];
		HoldForm[expr]/.{a_List:>MatrixForm[a]}, 
		Row[{CalculateScan`CommonSymbols`GrayText[#]/.{{a_}:>a}/.{a_List:>MatrixForm[a]}, CalculateScan`CommonSymbols`GrayText@" \[LongEqual] ", HoldForm[expr]/.{{a_}:>a}/.{a_List:>MatrixForm[a]}}]&
			[If[Head[y]===Symbol && MatchQ[ArgInDepVar,{_Symbol}|{__Symbol}], y@@ArgInDepVar, y]]
	]];


(* helper for prettyFuntion[expr_] *)
getYEqual[expr_, opts:OptionsPattern[]] := 
With[{
	ArgInDepVar = If[#===True,If[ImplicitFuncQ[expr],False,If[#==={},Select[Symbol/@{"x","t","u","v"},FreeQ[expr,#]&,1],#]]&@chooseVariables[expr],#]&["WithArgsInDepVars"/.{opts}], 
	y = (chooseVariables[expr, #, "ReturnDependent" -> True, opts] & [preprocessUserInput[expr]])
	}, 
	DebugPrint["getYEqual:ArgInDepVar",ArgInDepVar,"==",opts,"==",expr,"==",y];
	
	If[ArgInDepVar === True, ArgInDepVar = y[[1,1]]];
	Row[{HoldForm[CalculateScan`CommonSymbols`GrayText[#]]/.{{a_}:>a}/.{a_List:>MatrixForm[a]}, CalculateScan`CommonSymbols`GrayText@" \[LongEqual] ", HoldForm[expr]/.{{a_}:>a}/.{a_List:>MatrixForm[a]}}]&
		[
			If[MatchQ[ArgInDepVar,{_Symbol}|{__Symbol}], 
				DebugPrint["ffffs",Head[y[[-1,-1,1]]]]; If[Length[y[[-1,-1]]]==1,y[[-1,-1]]@@ArgInDepVar,(#@@ArgInDepVar)&/@y[[-1,-1]]], 
				DebugPrint["ffffsds",Head[y],FullForm@ArgInDepVar]; y[[-1,-1]]
			]
		]
];


(* ::Section::Closed:: *)
(*prettyDomain*)


(*prettyDomain::usage = "formats a math domain, such as 'x from 1 to 2'";
Attributes[prettyFunction] = {HoldAll};
prettyDomain[{x_, a_ < x_ < b_}] := RowTemplate[Localize["`1` \[LongEqual] `2`<GrayText> to </GrayText>`3`",1169],{lims[[1]] /. {z_}:>z, a, b}]
prettyDomain[lims_]:= (Which[
						ListQ[lims[[2]]] && Length[lims[[2]] /. MatrixForm[a_] :> a] <= 1, 
							RowTemplate[Localize["`1` \[LongEqual] `2`<GrayText> to </GrayText>`3`",1169],{lims[[1]] /. {a_}:>a,lims[[2]],lims[[3]]}],
						Length[lims[[2]] /. MatrixForm[a_] :> a] == 1,
							RowTemplate[Localize["`1`<GrayText> to </GrayText>`2`",1170],{lims[[2]],lims[[3]]}],
						True,
							RowTemplate["`1`",{lims[[2]]}]
			])
*)


(* ::Section::Closed:: *)
(*make1Limit*)


make1Limit::usage = "formats a single point on a curve in n dimensions";
(*
1st arg: the limit
2nd arg: what the calling scanner thinks the parameter (ind. var.) is. This is what make1Limit[] tries to solve for if necessary.
3rd arg: the curve
4th arg: the preprocessed curve - usually contains only 2nd arg as a variable*)

Clear[make1Limit]
(*TODO: option for real limits only is not completely filled out for every dv!*)
Options[make1Limit]:={"RealOnly"->False}

make1Limit[{y_ -> a_}|{{y_ -> a_}}, c__] := (DebugPrint["pre-make1lim1    ",{y, a},"===",c]; make1Limit[{y, a}, c]) 
make1Limit[limpre_,x_,expr_, op:OptionsPattern[]] := (DebugPrint["pre-make1lim2    ",expr]; make1Limit[limpre,x,expr,preprocessUserInput[expr, "NoFail"->True],op])
make1Limit[limpre_,x_,expr:{__Equal}, fn_, op:OptionsPattern[]] := (DebugPrint["pre-make1lim3    ",expr]; make1Limit[limpre,x,Thread[expr,Equal],fn])
make1Limit[limpre_,{x_},expr_, fn_, op:OptionsPattern[]] := (DebugPrint["pre-make1lim4    ",expr]; make1Limit[limpre,x,expr,fn])

(****1-D RULES****)
make1Limit[{{y1_,y2_}, {a_?NumericQ, b_?NumericQ}}, x_, expr_?(FreeQ[#,List]&), fn_, op:OptionsPattern[]]  /; !ImplicitFuncQ[fn] := 
(DebugPrint["list point but no list in expr ",(fn/.{x->a})]; Which[y1===x && TrueQ[(fn/.{x->a})==b],{{x->a}},y2===x && TrueQ[(fn/.{x->b})==a],{{x->b}},True,{"NotOnCurve",{y1->a,y2->b}}])

make1Limit[{Automatic|None, {a_?NumericQ, b_?NumericQ}}, x_, expr_?(FreeQ[#,List]&), fn_, op:OptionsPattern[]] /; !ImplicitFuncQ[fn] := 
(DebugPrint["list point but no list in expr ", ImplicitFuncQ[fn], x, {If[ListQ[x],x[[1]],x]->a}, (fn/.{x->a})]; 
	If[TrueQ[(fn/.{If[ListQ[x],x[[1]],x]->a})==b],
		{If[ListQ[x],Thread[x->{a,b}],{x->a}]}, 
		{"NotOnCurve",{x->a,"depvar"->b}}
	]
)

make1Limit[{y_?usersymbolQ, a_?NumericQ}, x_, expr_?(FreeQ[#,Equal] && FreeQ[#,List]&), fn_, op:OptionsPattern[]] /; FreeQ[expr, y] && !FreeQ[expr, x] := 
(DebugPrint["wrong var in lim! "]; solveToGetEndpoint[expr, {x, a}, op])

make1Limit[{y_, a_?NumericQ}, x_, expr : (y_ == f_)|(f_?(!usersymbolQ[#]&) == y_)|(y_[x_] == f_)|(f_?(!usersymbolQ[#]&) == y_[x_]), fn_, op:OptionsPattern[]] := 
(DebugPrint["user gave dep vars ",fn]; solveToGetEndpoint[f, {x, a}, op])

(*make1Limit[{Automatic|None, {a__?NumericQ}}, x_, expr : (y_ == f_)|(f_?(!usersymbolQ[#]&) == y_)|(y_[x_] == f_)|(f_?(!usersymbolQ[#]&) == y_[x_]), fn_] := 
(DebugPrint["unknown list point, 1-D"]; solveToGetEndpoint[fn, {x, {a}}])
*)
(***IMPLICIT - where expr===fn***)
make1Limit[{v_, {a__?NumericQ}}, x_List, expr_?ImplicitFuncQ, fn_, op:OptionsPattern[]] := 
	With[{var = If[MatchQ[v,Automatic|None],x,v]},
		(DebugPrint["implicit expr ", {var->{a}}]; 
		If[expr/.Thread[var->{a}],{Thread[var->{a}]},{"NotOnCurve",Thread[x->{a}]}]
		)]

make1Limit[{var_, a_?NumericQ}, x_List, expr_?ImplicitFuncQ, fn_Equal, op:OptionsPattern[]] /; MemberQ[x,var] := 
(DebugPrint["implicit expr, one number lim ", {var->a}, sSolve[expr/.{var->a},Complement[x,listone@var]]]; 
 DeleteDuplicates[Join[{var->a},#]&/@
 	If[TrueQ[OptionValue["RealOnly"]],
 		Solve[expr/.{var->a},Complement[x,listone@var],Reals],
 		Solve[expr/.{var->a},Complement[x,listone@var]]
 	]
 ])


(***LIST INPUTS***)
make1Limit[{y_?usersymbolQ, a_?NumericQ}, x_, (dep_List==fn_)|(fn_==dep_List), fn_List, op:OptionsPattern[]] /; FreeQ[fn, y] && MemberQ[dep, y|y[_]] := 
(DebugPrint["user gave single dep var w/ List input"]; solveToGetEndpoint[fn[[Position[dep,y|y[_],1][[1,1]]]], {x, a}, op])

make1Limit[limpre : {var_List, a_List}, x_?(Head[#]=!=List&), expr_Equal, fn_, op:OptionsPattern[]] /; (Sort@VarList[expr] === Sort[var])  := 
(DebugPrint["var is list, but not x",limpre,"==",x]; solveToGetEndpoint[expr, {var, a}, op])

make1Limit[limpre : {var_, a_List}, x_, expr : {__Equal}, fn_, op:OptionsPattern[]] /; !MemberQ[expr, If[MatchQ[var,None|Automatic],x,var]] := 
(DebugPrint["list expr endpt solve1"]; solveToGetEndpoint[fn, {If[MatchQ[var,None|Automatic] || Length@VarList[fn]==1,x,var], a}, op])

make1Limit[limpre : {var_, a_List}, x_, expr_List, fn_, op:OptionsPattern[]] /; !MemberQ[expr, If[MatchQ[var,None|Automatic],x,var]] := 
(DebugPrint["list expr endpt solve2", expr]; solveToGetEndpoint[expr, {If[Length@VarList[expr]==1,x,var], a}, op])

make1Limit[limpre : {var_, a_List}, x_, expr_List, fn_, op:OptionsPattern[]] /; MemberQ[expr, If[MatchQ[var,None|Automatic],x,var]] := 
(DebugPrint["x is in expr , use position",expr]; If[Solve[a==fn]==={},{"NotOnCurve",{var->a}},{{If[MatchQ[var,None|Automatic],x,var] -> a[[#]]}} &[Position[expr,x,1][[1,1]]]])

make1Limit[{y_?usersymbolQ, a_?NumericQ}, x_, expr_List, fn_List, op:OptionsPattern[]] /; FreeQ[{expr,fn},y] := 
(DebugPrint["user gave single dep var WHICH DOES NOT APPEAR IN List input"]; 
	Which[
		usersymbolQ[expr[[1]]], DebugPrint["1"]; make1Limit[{y,a},x,{expr[[1]],y}==expr,fn],
		usersymbolQ[expr[[2]]], DebugPrint["2"]; make1Limit[{y,a},x,{y,expr[[2]]}==expr,fn],
		Head[expr[[1]]]===List, DebugPrint["3"]; make1Limit[{y,a},x,expr[[1]],fn[[1]]],
		True, make1Limit[{y,a},x,{y,newFreeVariable[{x,y,expr,fn}]}==expr,fn]
	])

make1Limit[{var_List, a_List}, x_, expr_, fn_List, op:OptionsPattern[]] := 
With[{thevar=If[MatchQ[var, None|Automatic], x, x]}, DebugPrint["list expr endpt solve2"];
	solveToGetEndpoint[fn,{thevar, a}, op]
	]

make1Limit[{var_, a_?numericOrInfinityQ}, xx_, expr_, fn_, op:OptionsPattern[]] /; (MatchQ[var,Automatic|None|xx] || MatchQ[xx,Automatic|None]) := 
	(DebugPrint["default for numeric endpoint", MatchQ[var,Automatic|None|xx]]; 
	With[{x=If[MatchQ[xx,Automatic|None],chooseVariables[fn][[1]],xx]},
		If[ImplicitFuncQ[fn], DebugPrint["ImplicitFuncQ!",var,"---",x,"---",Solve[expr/.{x[[1]]->a},x[[2]]]];
			Join[{var->a},#]&/@ Solve[expr/.{var->a},Complement[x,listone@var]],
			{{If[MatchQ[var,None|Automatic],x,var]-> a}}
		]
	]
	)

make1Limit[limpre_, x_, expr_, fn_, op:OptionsPattern[]] := (DebugPrint["no make1Limit dv for this!",{limpre, x, expr, fn}];$Failed)

listone[a_]:=If[Head[a]===List,a,{a}];
numericOrInfinityQ[a_] := NumericQ[a] || Equal[a,-Infinity] || Equal[a,Infinity];


(* ::Section::Closed:: *)
(*pickvars*)


ClearAll[pickvars]

SetAttributes[pickvars, HoldAll];
pickvars[_, pr_] /; 
	!FreeQ[VarList[pr], Symbol["x"]] && !FreeQ[VarList[pr], Symbol["y"]] && FreeQ[VarList[pr], Symbol["z"]] && FreeQ[pr, Symbol["y"] == _ | _ == Symbol["y"]] && FreeQ[pr, Symbol["x"] == _ | _ == Symbol["x"]]  := {Symbol["x"], Symbol["y"]}

pickvars[_, pr_] /; 
	!FreeQ[VarList[pr], Symbol["x"]] && !FreeQ[VarList[pr], Symbol["y"]] && !FreeQ[VarList[pr], Symbol["z"]] && FreeQ[VarList[pr], Symbol["w"]|Symbol["v"]|Symbol["u"]] := {Symbol["x"], Symbol["y"], Symbol["z"]}

pickvars[orig_, pr_] := Module[{processed = preprocessUserInputHeld[Hold[orig]], vars, indeps, elim1, rin, rout, partitionvars},
  	If[FreeQ[processed, $Failed], vars = VarList[ processed ], vars = VarList[orig] ];
  	If[Length@vars < 2, Return[vars]];
  	
  	If[vars === Symbol /@ {"r", "t"}, Return[vars]]; (* special case *)
  	If[vars === Symbol /@ {"r", "\[Theta]"}, Return[vars]]; (* special case *)
  	If[vars === Symbol /@ {"h", "r"}, Return[vars]]; (* special case *)
  	If[vars === Symbol /@ {"u", "v", "w"}, Return[vars]]; (* special case *)
  	
  	indeps = Symbol /@ {"x", "y", "z", "u", "v", "w", "r", "s", "t", "p", "q", "m", "n"};
  
  	(* test 1 *)
  	elim1 = Select[vars, MemberQ[indeps, #] &];
  	If[0 < Length[elim1] < Length[vars],
   		Return[elim1]
   	];
  
  	(* test 2 *)
  	rin = Thread[(Symbol /@ CharacterRange["a", "z"]) -> Range[26]];
  	rout = Reverse /@ rin;
	  
  	partitionvars = Which[
  						Length[vars] == 2 && TrueQ[Abs[(vars /. rin)[[1]] - (vars /. rin)[[2]]] >= 5] && TrueQ[!VectorQ[vars, MemberQ[DeleteCases[$preferredindeps, "p"|"q"|"m"|"n"], ToString[#]] &] (*i.e. !SubsetQ[$preferredindeps,vars]*)],
  							{{vars[[1]]}, {vars[[2]]}},
  						TrueQ[Max[Differences[vars/.rin]] === 1] || !FreeQ[vars, _Subscript], 
  							{vars}, 
  						True,
  							clusterVariables[vars, rin, rout]
			  		];
  	Last @ SortBy[partitionvars, Length]
]


(* ::Section::Closed:: *)
(*clusterVariables*)


(* ::Subsection::Closed:: *)
(*clusterVariables*)


(* ::Text:: *)
(*clusterVariables is a call to FindClusters, with the goal of preserving the V9 output. See bug 331611.*)


(* ::Text:: *)
(*In V11.1, the V9 version of FindClusters lives in ClusterAnalysis`FindClusters`FindClustersOld.*)


If[TrueQ[System`Private`HasDownCodeQ[ClusterAnalysis`FindClusters`FindClustersOld]],
	clusterVariables[vars_, rin_, rout_] := Block[{res},
		res = ClusterAnalysis`FindClusters`FindClustersOld[vars /. rin];
		
		(
			res /. rout
		
		) /; ListQ[res]
	]
];


(* ::Text:: *)
(*Catch all definition. If FindClustersOld is ever taken away, we'll at least return a result.*)
(*In V11.1 Method -> {Automatic} calls the legacy FindClusters used in older versions.*)


clusterVariables[vars_, rin_, rout_] := FindClusters[vars /. rin, Method -> {Automatic}] /. rout


(* ::Section::Closed:: *)
(*getDependents*)


$preferreddependents =  {"y", "z", "w", "u", "v", "r", "s"};
$preferredlistdependents =  {"x", "y", "z", "u", "v", "w", "r", "s"};
$preferredindeps = {"x", "y", "z", "u", "v", "w", "r", "s", "t", "p", "q", "m", "n"};
$yequalspattern = ((_?usersymbolQ)[__] | _?usersymbolQ);
Clear[getDependents]
getDependents[f_?ImplicitFuncQ, indeps_List] := "Dependents" -> {} (*I hope this overrides the dv's below - pbarendse*)
getDependents[(f_?(MatchQ[#,(_?usersymbolQ)|Subscript[_Symbol, _]]&))[__]  == _, indeps_List] := "Dependents" -> {f}
getDependents[_  == (f_?(MatchQ[#,(_?usersymbolQ)|Subscript[_Symbol, _]]&))[__], indeps_List] := "Dependents" -> {f}
getDependents[y_?(MatchQ[#,(_?usersymbolQ)|Subscript[_Symbol, _]]&) == _, indeps_List]  /; FreeQ[indeps, y] := "Dependents" -> {y}
getDependents[_ == y_?(MatchQ[#,(_?usersymbolQ)|Subscript[_Symbol, _]]&), indeps_List] /; FreeQ[indeps, y] := "Dependents" -> {y}
getDependents[expr : {_Equal.. }, indeps_List] := "Dependents" -> Replace[expr[[All, 1]], a_[__?usersymbolQ]:>a, 1]
getDependents[expr_List, indeps_List] := Select[
				   							$preferredlistdependents,
				   							! MemberQ[ToString /@ indeps, #] &
				   					] /. {a : {_, ___} :> ("Dependents" -> (Symbol /@ a[[;;Min[Length[expr], Length[a]]]])), else_ :> ("Dependents" -> {$Failed})}
getDependents[expr_, indeps_List] /; MatchQ[expr, _Equal] && !MatchQ[expr, Equal[a:$yequalspattern, _] | Equal[_, a:$yequalspattern]]:= ("Dependents" -> {})
getDependents[expr_, indeps_List] := Select[
				   							$preferreddependents,
				   							! MemberQ[ToString /@ indeps, #] &
				   					] /. {a : {_, ___} :> ("Dependents" -> {Symbol[a[[1]]]}), else_ :> ("Dependents" -> {$Failed})}	


(* ::Section::Closed:: *)
(*getParameters*)


getParameters[(f_?usersymbolQ)[__]  == rhs_, indeps_List] := "Parameters" -> VarListComplement[rhs, indeps]
getParameters[rhs_  == (f_?usersymbolQ)[__], indeps_List] := "Parameters" -> VarListComplement[rhs, indeps]
getParameters[y_?usersymbolQ == rhs_, indeps_List]  := "Parameters" -> VarListComplement[rhs, indeps]
getParameters[rhs_ == y_?usersymbolQ, indeps_List] := "Parameters" -> VarListComplement[rhs, indeps]
getParameters[expr_, indeps_List] := "Parameters" -> VarListComplement[expr, indeps]

VarListComplement[expr_, indeps_] := Complement[VarList[expr], indeps]


(* ::Chapter::Closed:: *)
(*epilog*)


End[];


EndPackage[];
