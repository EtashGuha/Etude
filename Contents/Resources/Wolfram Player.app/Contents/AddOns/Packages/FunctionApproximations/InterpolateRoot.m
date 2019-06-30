(* ::Package:: *)

(*:Name: FunctionApproximations`InterpolateRoot` *)

(*:Title:  FindRoot Optimized For Very Expensive Analytic Functions *)

(*:Author: Jerry B. Keiper *)

(*:Summary:
This package numerically finds a root of an analytic function of a 
single variable using at each iteration every bit of known information
about the function.  It is particularly useful when each evaluation of 
the function is extremely expensive and extremely high precision is
desired.
*)

(*:Context: FunctionApproximations` *)

(*:Package Version: 2.1 *)

(* :Copyright: Copyright 1991-2007,  Wolfram Research, Inc.
*)

(* :History:
	Version 2.0 by Jerry B. Keiper, October 1991.
	Version 2.1: changed PrintFlag to ShowProgress to avoid symbol
		conflicts
*)

(*:Keywords: FindRoot, root finding *)

(*:Source:
*)

(*:Mathematica Version: 2.0 *)

(*:Limitation:
	InterpolateRoot only works on a function of a single variable,
	requires two nearby starting values, and is not as robust as
	FindRoot.  It does not work at all for multiple roots, but it is
	able to give quite reliable control on the accuracy of the root.
	In fact AccuracyGoal refers to the accuracy of the root rather
	than the magnitude of the residual at the root as in FindRoot.
*)

Unprotect[InterpolateRoot];

If[Not@ValueQ[InterpolateRoot::usage],InterpolateRoot::usage =
"InterpolateRoot[lhs == rhs, {x, x0, x1}] searches for a numerical solution \
to the equation lhs == rhs, starting with x == x0 and x == x1. It assumes \
that lhs and rhs are analytic in a region containing x0, x1, and the simple \
root being sought. InterpolateRoot[f, {x, x0, x1}] searches for a numerical \
solution to the equation f == 0, starting with x == x0 and x == x1. The \
option MaxIterations is the maximum number of iterations that will be \
performed. AccuracyGoal refers to the accuracy of the root rather than the \
magnitude of the residual at the root. AccuracyGoal -> Automatic means 20 \
digits less than the value of WorkingPrecision. WorkingPrecision is a \
suggested upper bound on the precision to use in evaluating the function at \
values near the root, but greater precision may be used if necessary to \
achieve the AccuracyGoal. The option ShowProgress (True or False) allows \
progress to be monitored."];

If[Not@ValueQ[ShowProgress::usage],ShowProgress::usage =
"ShowProgress is an option to InterpolateRoot, which takes values of True \
or False and specifies whether progress is to be monitored."];

Begin["`Private`"]

Options[InterpolateRoot] = {AccuracyGoal -> Automatic,
	MaxIterations -> 15, ShowProgress -> False,  WorkingPrecision -> 40};

InterpolateRoot::maxit = "InterpolateRoot exceeded MaxIterations of `1`."

InterpolateRoot[f_ == g_, {x_, a_, b_}, options___] :=
	InterpolateRoot[f-g, {x, a, b}, options]

InterpolateRoot[fg_, {x_, a_, b_}, options___] :=
    Module[{data, prec, maxprec, extraprec, ct, maxct, x0, f0, invfx,
		ag, minprec, sp, cd, gooddigits = 0, lenfactor},
	sp = ShowProgress /. {options} /. Options[InterpolateRoot];
	If[sp =!= True, sp = False];
	maxprec = WorkingPrecision /. {options} /. Options[InterpolateRoot];
	minprec = Min[maxprec, $MachinePrecision + 5];
	prec = Precision[{a, b}];
	If[prec < Infinity, minprec = Max[prec, minprec]];
	ag = AccuracyGoal /. {options} /. Options[InterpolateRoot];
	If[ag === Automatic, ag = maxprec - 20];
	prec = Min[Max[10 Round[-Log[10, N[Abs[a-b]]]], minprec], maxprec];
	ct = 0;
	maxct = MaxIterations /. {options} /. Options[InterpolateRoot];
	x0 = SetPrecision[b, prec];
	data = {{fg /. x -> x0, x0}};
	x0 = SetPrecision[a, prec];
	PrependTo[data, {fg /. x -> x0, x0}];
	extraprec = 0;
	While[True,
	    ct++;
	    prec = N[Abs[data[[1,2]] - data[[2,2]]]];
	    prec = If[prec == 0., 10 Accuracy[prec], Round[-10 Log[10, prec]]];
	    prec = Min[Max[prec, minprec], maxprec];
	    data = SetPrecision[data, prec+extraprec];
	    invfx = InterpolatingPolynomial[data, x];
	    lenfactor = 1 + (Length[data]-2)/4;
	    (* gooddigits is the number of good digits in the previous
		value of x0.  lenfactor gooddigits is a conservative
		estimate of the number of good digits in the next x0. *)
	    gooddigits = Min[Accuracy[x0] - 5, Round[lenfactor gooddigits]];
	    x0 = SetPrecision[invfx /. x -> 0, prec+extraprec];
	    If[sp, Print[{gooddigits, x0}]];
	    If[prec == maxprec && gooddigits > ag,
		Return[answer[x, x0, Round[gooddigits]]]];
	    f0 = fg /. x -> x0;
	    cd = data[[1]] - data[[2]];
	    cd = cd[[2]]/cd[[1]];
	    invfx = f0 cd;
	    If[sp, Print[{prec, extraprec, N[invfx, 30]}]];
	    gooddigits = Accuracy[invfx] - Precision[invfx];
	    If[ct == maxct,
		Message[InterpolateRoot::maxit, ct];
		Return[answer[x, x0, gooddigits]]
	    	];
	    If[prec < maxprec,
		If[Accuracy[invfx] - prec < 10,
			extraprec = Max[extraprec, 20+prec-Accuracy[invfx]]],
		    (* else *)
		If[Accuracy[invfx] - ag < 10, 
			extraprec += Max[extraprec, 20+ag-Accuracy[invfx]]]
		];
	    PrependTo[data, {f0, x0}];
	    If[Length[data] > 4, data = Take[data, 4]];
	    ];
	];

answer[x_, x0_, acc_] := {x -> If[acc < Accuracy[x0], SetAccuracy[x0, acc], x0]}

End[] (* `Private` *)

Protect[InterpolateRoot];

(* Tests:

InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 200]
InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 200,
	ShowProgress -> True]
InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 800,
	ShowProgress -> True]
InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 70,
	ShowProgress -> True]
InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 85,
	ShowProgress -> True]
InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 30,
	AccuracyGoal -> 60, ShowProgress -> True]
InterpolateRoot[Zeta[s], {s, 1/2 + 14.1I, 1/2+14.2I}, WorkingPrecision -> 30,
	ShowProgress -> True]
InterpolateRoot[Zeta[s], {s, 1/2 + 14.1I, 1/2+14.2I}, WorkingPrecision -> 30,
	AccuracyGoal -> 60, ShowProgress -> True]
InterpolateRoot[Exp[x] == 2, {x, 0, 1}, WorkingPrecision -> 85,
	ShowProgress -> True, MaxIterations -> 5]
 *)
