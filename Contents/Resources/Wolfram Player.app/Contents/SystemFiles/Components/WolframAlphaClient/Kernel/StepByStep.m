(* ::Package:: *)

(* ::Section::Closed:: *)
(*Package header*)


Begin["WolframAlphaClient`Private`"];


(* ::Section::Closed:: *)
(*PreProcessSBS*)


(* Step-by-step support provided by GHurst *)


$sbsRegEx = "(?i)((\\s*(\
(^\\s*sbs(:|.)?\\s)|(\\s+sbs$)|\
(step(-|(\\s*))?by(-|(\\s*))?step(\\s+solution)?(:)?)|\
(show(\\s*me)?\\s*steps(\\s*(for|to))?)|\
(show\\s*(step|((me\\s*)?how(\\s*to)?)))|\
(how\\s*do\\s*i\\s)|(how\\s*does\\s*one\\s*do\\s*)|\
((give|tell|show|verify|explain|find|teach|help)?(\\s*to)?(\\s*me)?((\\s*how)|(\\s*the\\s*(sbs|(step(s)?))))(\\s*to)?(\\s*(do|perform|find|solve|calculate|compute))?(\\s*the)?(\\s*following)?(\\s*(problem|equation|expression))?)|\
((give|tell|show|verify|explain|derive|teach|help)(\\s*to)?(\\s*me)?((\\s*how)|(\\s*the\\s*(sbs|(step(s)?))))?(\\s*to\\s*(do|perform|find|calculate|compute))?((\\s*the)?\\s*(answer|solution|result|steps|work))?)|\
)((\\s*to)?\\s*the\\s*(following|((following\\s*)?(equation)))(\\s*(problem|equation|expression))?)?\\s*))";


PreProcessSBS[query_String] :=
	Block[{stripped, sbsQ = False},
		stripped = StringReplace[
			StringTrim[query],
			p: RegularExpression["^" <> $sbsRegEx <> "|" <> $sbsRegEx <> "$"] /; And[p =!= "", sbsQ = True] -> ""
		];
		
		{"QueryString" -> stripped, "StepByStepQuery" -> sbsQ}
	]


(* ::Section::Closed:: *)
(*AlphaIntegration`WolframAlphaStepByStep*)


Options[AlphaIntegration`WolframAlphaStepByStep] = {MenuView -> Automatic};


AlphaIntegration`WolframAlphaStepByStep[query_String, OptionsPattern[]]:=
	Block[{alphares, states, include, hasSBSQ, res, menuviewQ},
		alphares = WolframAlpha[query];
		
		If[noParseQ[alphares], Return[{"Result" -> $Failed, "ReturnType" -> "NoParse"}]];
		If[reinterpretedParseQ[alphares], Return[{"Result" -> $Failed, "ReturnType" -> "Reinterpreted"}]];
		
		states = Cases[alphares, {___, "input" -> in_String, "stepbystep" -> "true"} /; StringStartsQ[in, __ ~~ "__"] :> in, \[Infinity]];
		include = StringSplit[states, "__"][[All, 1]];
		
		{include, states} = relevantSBS[include, states];
		hasSBSQ = Length[include] > 0;
		
		res = If[hasSBSQ,
			WolframAlpha[query, IncludePods -> include, AppearanceElements -> {"Pods"}, PodStates -> states],
			alphares
		];
		
		(* temprorary workaround *)
		If[Count[res, "\"step\[Hyphen]by\[Hyphen]step solution unavailable\"", \[Infinity]] == Length[states],
			res = alphares /. XMLElement["state",{___,"stepbystep"->"true",___},{}]:>Sequence@@{};
			hasSBSQ = False
		];

		menuviewQ = TrueQ[Replace[OptionValue[MenuView], Automatic -> hasSBSQ, {0}]];		
		
		{
			"Result" -> postProcessAlphaSBS[res, menuviewQ],
			"ReturnType" -> If[hasSBSQ, "HasStepByStep", "NoStepByStep"]
		}
	]


noParseQ[alphares_] := !FreeQ[alphares, "success" -> "false"]


reinterpretedParseQ[alphares_] := !FreeQ[alphares, XMLElement["warnings", ___, {___, XMLElement["reinterpret", ___, {}], ___}, ___]]


relevantSBS[{f: ("Input" | "Result" | "IndefiniteIntegral" | "Limit"), ___}, {s_, ___}] := {{f}, {s}}

relevantSBS[l: {___, d: "DifferentialEquationSolution", ___}, states_] := {{d}, Extract[states, Position[l, d]]}

relevantSBS[i_, s_] := {i, s}


postProcessAlphaSBS[res_, True] := 
	With[{titles = Cases[res, XMLElement["pod", {___, "title" -> t_String, ___}, ___] :> t,\[Infinity]]},
		res /. HoldPattern[Dynamic][HoldPattern[AlphaIntegration`FormatAlphaResults][HoldPattern[Dynamic][{args__, _, o1_, o2_}]], o___] :> 
			If[Length[titles] == Length[Typeset`showpods] > 1,
				Framed[
					Manipulate[
						Dynamic[AlphaIntegration`FormatAlphaResults[Dynamic[{args, {#}, o1, o2}]], o]&[i], 
						{{i, 1, ""}, Thread[Range[Length[Typeset`showpods]] -> titles], PopupMenu},
						AppearanceElements -> None,
						FrameMargins -> 0,
						Paneled -> False
					],
					FrameStyle -> {GrayLevel[.9]},
					RoundingRadius -> 10
				],
				Dynamic[AlphaIntegration`FormatAlphaResults[Dynamic[{args, {1}, o1, o2}]], o]
			]
	]

postProcessAlphaSBS[res_, _] := res


(* ::Section::Closed:: *)
(*StepByStepQ*)


(* ::Text:: *)
(*\[Bullet] indefinite integrals*)
(*\[Bullet] definite integrals*)
(*\[Bullet] derivative*)
(*\[Bullet] limit*)
(*\[Bullet] solve*)
(*\[Bullet] DSolve*)
(*\[Bullet] expand*)
(*\[Bullet] trig proofs*)
(*\[Bullet] det, inverse, trace, eigen, rank, nullspace, rref, dot, cross*)
(*\[Bullet] GCD, LCM, PrimeQ, CoprimeQ, Divisors, FactorInteger, Divisible*)
(*\[Bullet] partial fractions*)
(*\[Bullet] factor polynomial*)
(*\[Bullet] non trig/exp/log simplification*)
(*\[Bullet] mean, median, mode, std, variance, quartile, range, geometric mean*)


(* ::Subsection::Closed:: *)
(*Main*)


SetAttributes[StepByStepQ, HoldFirst];

StepByStepQ[parse:Hold[a_. * Integrate[_, Except[_List]]], eval_] := sbsIndefiniteIntegrateQ[parse, eval]

StepByStepQ[parse:Hold[a_. * Integrate[_, _List]], eval_] := sbsDefiniteIntegrateQ[parse, eval]

StepByStepQ[parse:Hold[_D], eval_] := sbsDerivativeQ[parse, eval]

StepByStepQ[parse:Hold[_Limit], eval_] := sbsLimitQ[parse, eval]

StepByStepQ[parse:Hold[_Solve|_Reduce], eval_] := sbsSolveQ[parse, eval]

StepByStepQ[parse:Hold[_DSolve|_DSolveValue], eval_] := sbsDSolveQ[parse, eval]

StepByStepQ[parse:Hold[_Expand], eval_] := sbsExpandQ[parse, eval]

StepByStepQ[parse:Hold[_Equal|Simplify[_Equal]|TrueQ[Simplify[_Equal]]], _] := sbsTrigProofQ[parse]

StepByStepQ[Hold[(Det|Inverse|Trace|Eigenvalues|Eigensystem|MatrixRank|NullSpace|RowReduce)[M_]], _] := MatrixQ[Unevaluated[M]]

StepByStepQ[Hold[(Plus|Times|Dot|Cross)[M1_List, M2_List]], eval_] := 
	ListQ[eval] && (MatrixQ[Unevaluated[M1]] || VectorQ[Unevaluated[M1]]) && (MatrixQ[Unevaluated[M2]] || VectorQ[Unevaluated[M2]])

StepByStepQ[Hold[(GCD|LCM)[__Integer]], eval_] := IntegerQ[eval]

StepByStepQ[Hold[PrimeQ[k_]], eval_] := 
	And[
		BooleanQ[eval],
		Abs[Quiet[ReleaseHold[N[Hold[k]]]]] < 10^1000.,
		With[{n = k},
			IntegerQ[n] && ((eval && Sqrt[Abs[n]] < 100) || TimeConstrained[FactorInteger[Abs[n]][[1, 1]] < 100, .01, False])
		]
	]

StepByStepQ[Hold[CoprimeQ[_Integer, _Integer]], eval_] := IntegerQ[eval]

StepByStepQ[Hold[Divisible[_Integer, _Integer?Positive]], eval_] := BooleanQ[eval]

StepByStepQ[Hold[Divisors[_Integer]], eval_] := Length[eval] <= 50

StepByStepQ[Hold[FactorInteger[_]], eval_] := ListQ[eval] && eval =!= {{-1|1, 1}} && DeleteCases[eval, {-1|1, _}][[1, 1]] < 100 && Abs[Times @@ Power @@@ eval] < 10^1000.

StepByStepQ[Hold[_Apart], eval_] := Head[eval] === Plus && Head[Together[eval]] =!= Plus

StepByStepQ[parse:Hold[_Factor], eval_] := FreeQ[eval, Factor] && sbsFactorQ[parse, eval]

StepByStepQ[Hold[_Mean|_Mode|_Median|_StandardDeviation|_Variance|_GeometricMean], eval_] := NumericQ[eval]

StepByStepQ[Hold[(Commonest|Quartiles)[_]], eval_] := VectorQ[eval, NumericQ]

StepByStepQ[Hold[(Simplify|FullSimplify)[expr_]], _] := sbsSimpleQ[expr]

StepByStepQ[parse_Hold, eval_] := parse =!= Hold[eval] && sbsSimpleQ[parse]

StepByStepQ[___] = False;


(* ::Subsection::Closed:: *)
(*sbsIndefiniteIntegrateQ*)


sbsIndefiniteIntegrateQ[Hold[a_. * Integrate[integrand_, Except[_List]]], eval_] := elementaryQ[Hold[integrand]] && elementaryQ[eval]


elementaryQ[expr_, o___] := Complement[
	Union[Cases[Hold[expr], sym_Symbol /; Context[sym] === "System`", {0, Infinity}, Heads -> True]],
	{Plus, Times, Power, Sqrt, Surd, E, Exp, Log, Sin, Cos, Sec, Csc, Tan, Cot, Sinh, Cosh, Sech, Csch, Tanh, Coth,
		ArcSin, ArcCos, ArcSec, ArcCsc, ArcTan, ArcCot, ArcSinh, ArcCosh, ArcSech, ArcCsch, ArcTanh, ArcCoth,
		Pi, Khinchin, Derivative, Hold}, 
	{o}
] === {}


(* ::Subsection::Closed:: *)
(*sbsDefiniteIntegrateQ*)


sbsDefiniteIntegrateQ[Hold[c_. * Integrate[integrand_, {x_, a_, b_}]], eval_] := With[{f = integrand},
	And[
		FreeQ[Hold[{a, b}], x],
		FreeQ[eval, _DirectedInfinity | ComplexInfinity | Undefined | Indeterminate | ConditionalExpression],
		Or[!FreeQ[Hold[f], Abs] && elementaryQ[Hold[f], Abs], elementaryPrimitiveQ[f, x]],
		Or[
			And[Union[Cases[{f, a, b}, s_Symbol /; Context[s] =!= "System`", {0, \[Infinity]}]] =!= {x}, VectorQ[Cases[{f, a, b}, _?NumericQ, {0, \[Infinity]}], Im[#] == 0&]],
			realIntegralQ[f, {x, a, b}]
		]
	]
]


elementaryPrimitiveQ[integrand_, x_] :=
	Block[{prim},
		prim = TimeConstrained[Integrate[integrand, x], .15];
		(
			elementaryQ[prim]
			
		) /; FreeQ[prim, Integrate] && prim =!= $Aborted
	]

elementaryPrimitiveQ[___] = False;


realIntegralQ[f_, {x_, a_, b_}] := TrueQ[Im[a] == 0] && TrueQ[Im[b] == 0] && realFunctionQ[f, {x, a, b}]


realFunctionQ[f_, {x_, -\[Infinity], \[Infinity]}] := realFunctionQ[f, {x, -\[Infinity], -1}] && realFunctionQ[f, {x, -1, 1}] && realFunctionQ[f, {x, 1, \[Infinity]}]


realFunctionQ[f_, {x_, -\[Infinity], b_}] /; b >= 0 := realFunctionQ[f, {x, -\[Infinity], -1}] && realFunctionQ[f, {x, -1, b}]
realFunctionQ[f_, {x_, -\[Infinity], b_}] := realFunctionQ[f /. x -> 1/x, {x, 1/b, 0}]


realFunctionQ[f_, {x_, a_, \[Infinity]}] /; a <= 0 := realFunctionQ[f, {x, a, 1}] && realFunctionQ[f, {x, 1, \[Infinity]}]
realFunctionQ[f_, {x_, a_, \[Infinity]}] := realFunctionQ[f /. x -> 1/x, {x, 0, 1/a}]


realFunctionQ[f_, {x_, a_, b_}] := With[{mesh = Range[a + .99949862(b-a)/1000., b -(b-a)/1000., (b-a)/100.]},
	VectorQ[(Function @@ {f /. x -> #}) /@ mesh, Im[Chop[#]] == 0 &]
]

realFunctionQ[___] = False;


(* ::Subsection::Closed:: *)
(*sbsDerivativeQ*)


SetAttributes[sbsDerivativeQ, HoldAll];


sbsDerivativeQ[Hold[D[args__]], eval_] := sbsDerivativeQ[args, eval]


sbsDerivativeQ[expr_, {x_, ___}, eval_] := sbsDerivativeQ[expr, x, eval]
sbsDerivativeQ[expr_, {{x_, ___}}, eval_] := sbsDerivativeQ[expr, x, eval]
sbsDerivativeQ[expr_, x_, __, eval_] := sbsDerivativeQ[expr, x, eval]

sbsDerivativeQ[expr_, ___, eval_] /; !FreeQ[Unevaluated[{expr, eval}], Piecewise] = False;


sbsDerivativeQ[expr_, x_Symbol, _] /; specialStepsQ[expr, x] = True;


sbsDerivativeQ[_Plus|_Times|_Tan|_Sec|_Cot|_Csc|_Tanh|_Sech|_Coth|_Csch|_Abs|_Floor|_Ceiling|_IntegerPart|_TriangleWave|_SquareWave, _, _] = True;


sbsDerivativeQ[_Power|_Times|_Plus, _, _] = True;


sbsDerivativeQ[_Symbol[a___, x_, b___], x_, _] /; FreeQ[{a, b}, x] = False;


sbsDerivativeQ[y_[x_], x_, eval_] /; MatchQ[eval, HoldPattern[Derivative[1][_][x]]] = False;


sbsDerivativeQ[_, _, _] = True;

sbsDerivativeQ[___] = False;


SetAttributes[specialStepsQ, HoldFirst];


specialStepsQ[c_, x_Symbol] /; FreeQ[c, x] = True;


specialStepsQ[x_, x_Symbol] = True;


specialStepsQ[f_^g_, x_Symbol] :=
	Switch[{FreeQ[f, x], FreeQ[g, x]},
		{False, True},
			f === x && MemberQ[{-1, 2, 3, 4, 5, -1/2, 1/2, 1/3, 1/4}, g],
		{True, False},
			g === x,
		{False, False},
			FreeQ[{f, g}, Power[q_, w_] /; !FreeQ[q, x] && !FreeQ[w, x]],
		{True, True},
			True
	]


specialStepsQ[expr_, x_Symbol] := MatchQ[Unevaluated[expr], $SpecialDStepsHeads[a___, x, b___] /; FreeQ[Unevaluated[{a, b}], x]]

$SpecialDStepsHeads = Power|Sqrt|Exp|Log|Sin|Cos|Sinh|Cosh|ArcSin|ArcCos|ArcTan|ArcSec|ArcCos|ArcCot|ArcCsc|ArcSinh|ArcCosh|ArcTanh|ArcSech|ArcCosh|ArcCoth|ArcCsch|Sinc|Haversine|InverseHaversine|Gudermannian|InverseGudermannian|UnitStep|HeavisideTheta|UnitBox|HeavisidePi|HeavisideLambda|SawtoothWave|ProductLog;


(* ::Subsection::Closed:: *)
(*sbsLimitQ*)


sbsLimitQ[Hold[Limit[_, Except[_List] -> _]], eval_] := FreeQ[eval, Interval|Underflow[]|Overflow[]|Undefined|Indeterminate]

sbsLimitQ[___] = False;


(* ::Subsection::Closed:: *)
(*sbsSolveQ*)


$SBSSolveOK = {Sin, Cos, Tan, Csc, Sec, Cot, Sinh, Cosh, Tanh, Csch,
    Sech, Coth, ArcSin, ArcCos, ArcTan, ArcCsc, ArcSec, ArcCot,
    ArcSinh, ArcCosh, ArcTanh, ArcCsch, ArcSech, ArcCoth, Log, Exp,
    Rational, Complex, Sqrt, Abs, D, Surd, Subscript};

sbsSolveQ[Hold[(Solve|Reduce)[expr_, x_, rest___]], roots_] := MatchQ[{rest}, {}|{Automatic}|{Complexes}|{Reals}] && iSBSSolveQ[Hold[expr], x, roots]

sbsSolveQ[___] = False;

(* No roots *)
iSBSSolveQ[_, {_, __}, {} | {{}} | _Solve | _Reduce | False | True] := False

iSBSSolveQ[Hold[expr:Equal[_, _List]], vars_, roots_] := iSBSSolveQ[Hold[#], vars, roots]& [Thread[Flatten /@ expr]] 

iSBSSolveQ[expr_Hold, {x_}, roots_] := iSBSSolveQ[expr, x, roots]

iSBSSolveQ[expr_, vars_, roots_] := iSBSSolveQ[Hold[expr], vars, roots]

iSBSSolveQ[Hold[lhs_ == rhs_], x_, roots_] /; !ListQ[x] && FreeQ[roots, _Real|_Root|_AlgebraicNumber] :=
	With[{poly = lhs - rhs},
		(
			!IrreduciblePolynomialQ[poly]
		
		) /; PolynomialQ[poly, x] && Exponent[poly, x] == 5 && VectorQ[CoefficientList[poly, x], NumericQ]
	]

iSBSSolveQ[expr_Hold, vars_, roots_] := 
	Depth[expr] > 2 && With[{systemHeads = Cases[expr, v_Symbol /; Context[v] === "System`" && !NumericQ[v], {0, Infinity}, Heads -> True]},
      
      And[
      	Not[!ListQ[vars] && MatchQ[expr, Hold[f_ == g_] /; f === vars && FreeQ[g, vars]]], 
      	VectorQ[Cases[expr, _CalculateRatios, Infinity], Length[#] == 2&],
      	VectorQ[systemHeads,
      		MemberQ[{HoldForm, Hold, Plus, Times, Power, Equal, List, Sequence @@ $SBSSolveOK}, #] &
      	],
      	FreeQ[roots /. r_Root :> ToRadicals[r], ProductLog | Root | RootSum | Re | Im | Conjugate | HypergeometricPFQ],
      	Or[
      		And[ListQ @@ expr, ListQ[vars], Length @@ expr == Length[vars], solvableLinearSystemQ[First[expr], vars] || And[TrueQ[$AlphaDevModeQ], Length[vars] == 2, FreeQ[roots, C]]],
      		!(ListQ @@ expr) && !ListQ[vars] && !commonMixedPolyAndTrancendentalQ[Subtract @@ First[expr], vars],
      		ListQ @@ expr && Length @@ expr == 1 && !ListQ[vars]
      	],
      	FreeQ[roots, (Greater|GreaterEqual|Less|LessEqual|Inequality)[___, Alternatives @@ Flatten[{vars}], ___]]
      ]
      
]

commonMixedPolyAndTrancendentalQ[expr_, x_] := Block[{toAlg},
	
	toAlg = expr /. {_Sin|_Cos|Power[_, _?(!FreeQ[#, x]&)] :> RandomReal[]};
	
	And[
		!AlgebraicExpressionQ[expr, x],
		!FreeQ[toAlg, x] && AlgebraicExpressionQ[toAlg, x]
	]
]

AlgebraicExpressionQ[_Symbol, _] := True
AlgebraicExpressionQ[c_, x_] /; FreeQ[c, x] := True
AlgebraicExpressionQ[Power[f_, _Integer|_Rational], x_] := AlgebraicExpressionQ[f, x]
AlgebraicExpressionQ[HoldPattern[Plus|Times][args__], x_] := VectorQ[{args}, AlgebraicExpressionQ[#, x]&]
AlgebraicExpressionQ[__] := False

linearSystemQ[sys:{_Equal..}, vars_List] := 
	Apply[And, (PolynomialQ[#, vars] && Length[Select[Exponent[#, vars], #>1&]] == 0 && linnQ[#, vars])& /@ Subtract @@@ sys];
linearSystemQ[sys:HoldPattern[And][_Equal..], vars_List] := linearSystemQ[List @@ sys, vars]

linnQ[expr_, vars_] := ilinnQ[CoefficientList[expr + Total[$temp vars], vars]]

ilinnQ[_] := True
ilinnQ[{f_List, l_List}] := MatchQ[Rest[Flatten[l]], {0..}] && ilinnQ[f]

solvableLinearSystemQ[sys:{_Equal..}, vars_List] /; linearSystemQ[sys, vars] := 
	Length[sys] == Length[vars] && Det[Map[Coefficient[#, vars]&, Subtract @@@ sys]] =!= 0
solvableLinearSystemQ[sys:HoldPattern[And][_Equal..], vars_List] := solvableLinearSystemQ[List @@ sys, vars]
solvableLinearSystemQ[___] = False;


(* ::Subsection::Closed:: *)
(*sbsDSolveQ*)


$SBSDSolveBlackList = ParabolicCylinderD | AiryAi | AiryAiPrime | AiryAiZero | AiryBi | AiryBiPrime | AiryBiZero | AngerJ | 
	BesselI | BesselJ | BesselJZero | BesselK | BesselY | BesselYZero | HankelH1 | HankelH2 | KelvinBei | KelvinBer | 
	KelvinKei | KelvinKer | ScorerGi | ScorerGiPrime | ScorerHi | ScorerHiPrime | SphericalBesselJ | SphericalBesselY | 
	SphericalHankelH1 | SphericalHankelH2 | StruveH | StruveL | WeberE;


sbsDSolveQ[Hold[(DSolve|DSolveValue)[_, _, _List]], _] = False;


sbsDSolveQ[Hold[(DSolve|DSolveValue)[_, {_, __}, _]], _] = False;


sbsDSolveQ[Hold[(h:DSolve|DSolveValue)[in_, {y_}, x_]], eval_] := sbsDSolveQ[Hold[h[in, y, x]], eval]


sbsDSolveQ[Hold[(h:DSolve|DSolveValue)[in_, y_Symbol, x_Symbol]], eval_] := sbsDSolveQ[Hold[h[in, y[x], x]], eval]


sbsDSolveQ[Hold[(DSolve|DSolveValue)[_, _Symbol[_Symbol], _Symbol]], eval_] := FreeQ[eval, $SBSDSolveBlackList]


sbsDSolveQ[___] = False;


(* ::Subsection::Closed:: *)
(*sbsExpandQ*)


sbsExpandQ[Hold[Expand[input_]], res_] := And[
	heldpolyQ[Hold[input]] || !heldpolyQ[Hold[res]],
	FreeQ[Hold[input], Greater|Less|GreaterEqual|LessEqual|D|Limit|Sum]
]

sbsExpandQ[___] = False;


heldpolyQ[Hold[HoldPattern[Times | Plus][args__]]] := VectorQ[Unevaluated[{args}], Function[q, heldpolyQ[Hold[q]], HoldFirst]]
heldpolyQ[Hold[Power[f_, n_]]] /; IntegerQ[n] := heldpolyQ[Hold[f]]
heldpolyQ[Hold[_Symbol]] := True
heldpolyQ[Hold[f_]] /; FreeQ[Unevaluated[f], s_Symbol /; Context[s] === "System`"] := True
heldpolyQ[_] := False


(* ::Subsection::Closed:: *)
(*sbsTrigProofQ*)


SetAttributes[sbsTrigProofQ, HoldFirst];

sbsTrigProofQ[expr_Hold] := sbsTrigProofQ @@ expr

sbsTrigProofQ[expr_] /; FreeQ[Hold[expr], Sin|Cos|Tan|Sec|Csc|Cot] = False;

sbsTrigProofQ[expr_Equal] := Module[{modexpr, trigvars, sin, cos, tan, sec, csc, cot},
	modexpr = Hold[expr] //. {Sin[f_] :> sin[f], Cos[f_] :> cos[f], Tan[f_] :> tan[f], 
							HoldPattern[Sec][f_] :> sec[f], HoldPattern[Csc][f_] :> csc[f], HoldPattern[Cot][f_] :> cot[f]};
  	trigvars = DeleteDuplicates[
				    Cases[modexpr, (sin | cos | tan | sec | csc | cot)[_], {0, Infinity}]];
  
  	Length[trigvars] > 0 && Apply[iTrigProofQ[#, trigvars] &, modexpr]
]

sbsTrigProofQ[___] = False;


SetAttributes[iTrigProofQ, HoldFirst];

iTrigProofQ[_Symbol | (_?NumericQ), _] := True
iTrigProofQ[f_, vars_] /; MemberQ[Hold /@ vars, Hold[f]] := True
iTrigProofQ[Equal[e1_, e2_], vars_] := iTrigProofQ[e1, vars] && iTrigProofQ[e2, vars]
iTrigProofQ[HoldPattern[Plus][args___], vars_] := Apply[And, iTrigProofQ[#, vars] & /@ Unevaluated[{args}]]
iTrigProofQ[HoldPattern[Times][args___], vars_] := Apply[And, iTrigProofQ[#, vars] & /@ Unevaluated[{args}]]
iTrigProofQ[Power[f_, _Integer], vars_] := iTrigProofQ[f, vars]
iTrigProofQ[__] := False


(* ::Subsection::Closed:: *)
(*sbsFactorQ*)


sbsFactorQ[Hold[Factor[poly_]], eval_] :=
	With[{vars = Variables[eval]},
		And[
			MatchQ[Head /@ vars, {__Symbol}],
			PolynomialQ[eval, vars],
			sbsFacLookupQ[eval, vars]
		]
	]


sbsFacLookupQ[expr_^_Integer?Positive, _] := plusLength[expr] == 2

sbsFacLookupQ[expr_, {x_}] /; Exponent[expr, x] <= 4 := !IrreduciblePolynomialQ[expr]

sbsFacLookupQ[expr_, {x_, y_}] /; Max[Exponent[expr, x], Exponent[expr, y]] <= 4 := !IrreduciblePolynomialQ[expr]

sbsFacLookupQ[expr_, {x_}] := 
	With[{droppedlinear = dropLinearFactors[expr, x]},
		droppedlinear =!= expr && (Exponent[droppedlinear, x] <= 4 || IrreduciblePolynomialQ[droppedlinear])
	]


plusLength[HoldPattern[Plus][args__]] := Length[{args}]
plusLength[e_] = 1;


dropLinearFactors[expr_, x_] := Times @@ DeleteCases[timesList[expr], (a_. x + b_.)^n_. /; NumericQ[a] && NumericQ[b]]


timesList[HoldPattern[Times][args__]] := Length[{args}]
timesList[e_] := {e};


(* ::Subsection::Closed:: *)
(*sbsSimpleQ*)


SetAttributes[sbsSimpleQ, HoldFirst];


sbsSimpleQ[expr:Except[_Hold]] := sbsSimpleQ[Hold[expr]]
sbsSimpleQ[Hold[Rational[a_, b_?Positive]]] /; GCD[a, b] === 1 := False
sbsSimpleQ[Hold[Times[a_Integer, Power[b_Integer, -1]]|Times[a_Integer, Times[1, Power[b_Integer, -1]]]]] /; Positive[b] && GCD[a, b] === 1 := False
sbsSimpleQ[Hold[Power[b_Integer?Positive, -1]]] := False
sbsSimpleQ[Hold[Plus[a_Integer, b_Integer]]] /; Abs[a] < 10 && Abs[b] < 10 && Or[a+b < 0, a*b == 0] := False
sbsSimpleQ[Hold[Times[a_Integer, b_Integer]]] /; Abs[a] < 10 && Abs[b] < 10 &&  (a <= 0 || b <= 0) := False
sbsSimpleQ[Hold[_Integer|_Rational|_Real|_Complex]] := False
sbsSimpleQ[Hold[Times[a_, Power[b_, -1]]]] /; VectorQ[{a,b}, NumericQ] && (Hold[Rational[a, b]] === (Hold[#] &[Rational[a, b]])) := False
sbsSimpleQ[Hold[expr_]] /; System`Dump`HeldNumericQ[expr] && Abs[expr] > 10^10 := False
sbsSimpleQ[Hold[e1_ == e2_]] /; VectorQ[Unevaluated[{e1, e2}], NumericQ] := sbsSimpleQ[Hold[e1]] && sbsSimpleQ[Hold[e2]]
sbsSimpleQ[Hold[Times[a_?MatrixQ,b__?MatrixQ]]]:=False 
sbsSimpleQ[Hold[Dot[a_?MatrixQ,b__?MatrixQ]]]:=With[{dims=Dimensions/@ {a,b}},SameQ@@({Rest[First[#]],Most[Last[#]]}&[Transpose[dims]])]
sbsSimpleQ[expr_Hold] := FreeQ[expr, _Real] && Depth[expr] > 2 && simpleQ[expr]

simpleQ[expr_Hold] := With[
 	{systemHeads = Cases[expr, h_Symbol[___] :> h, {0, Infinity}, Heads -> True],
 	usersVars = Alternatives @@ Union[Cases[expr, s_Symbol /; Hold[s] =!= Hold[I] :> HoldPattern[s], {0, Infinity}, Heads -> False]]},
 		And[
 			(* Only algebraic number heads *)
   			VectorQ[systemHeads, MemberQ[{HoldForm, Hold, Plus, Times, Power, Complex, Abs, Rationalize, Sqrt, Surd, CubeRoot, List}, #] &],
   			(* Vars only to an integer power, or algebraic number to a rational power *)
 			VectorQ[
    			Cases[Hold[expr] /. $powPatts, Power[a_, b_] :> {Hold[a], b}, {0, Infinity}, Heads -> True], 
    			Or[IntegerQ[Cancel[Together[Last[#]]]],
    				MatchQ[Last[#], _DirectedInfinity],
    				(FreeQ[First[#], usersVars] && (rationalQ[Last[#]] || MatchQ[Last[#], _DirectedInfinity]))
    			] &
    		]
 		]
]

$powPatts = {HoldPattern[Sqrt][a_] :> Hold[a^(1/2)], HoldPattern[CubeRoot][a_] :> Hold[a^(1/3)], Surd[a_, b_] :> Hold[a^(1/b)]};
rationalQ[q_] := MemberQ[{Integer, Rational}, Head[Cancel[Together[q]]]]


(* ::Section::Closed:: *)
(*Package footer*)


End[];
