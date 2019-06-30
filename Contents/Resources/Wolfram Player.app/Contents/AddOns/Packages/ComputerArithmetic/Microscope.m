(* ::Package:: *)

(* :Name: Microscope *)

(* :Title: Microscopic Examination of Roundoff Errors *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package allows functions to be plotted on a microscopic scale to
exhibit the granularity of machine arithmetic.  Alternatively the
actual error (measured in ulps (units last place)) can be plotted.
*)

(* :Context: ComputerArithmetic` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1992-2007, Wolfram Research, Inc.
*)

(* :History:
	Originally sketched by Jerry B. Keiper, January 1991.
	Revised by Jerry B. Keiper, December 1991.
*)

(* :Keywords: machine precision, roundoff error, granularity of machine
	numbers
*)

(* :Source:
*)

(* :Warnings:
*)

(* :Mathematica Version: 2.0 *)

(* :Limitations:  The size of an ulp is defined relative to the size of the
	numbers being considered.  In small neighborhoods of powers of 2
	the size of an ulp changes and it is not clear what to do about
	error plots when this happens.  This is even more of a problem
	when looking at the error near a zero of a function where the
	relative error may even be unbounded.
*)

(* :Discussion:
	This package defines four functions: Ulp, MachineError,
	MicroscopePlot and MicroscopicErrorPlot.  Ulp gives the size of an
	ulp for numbers near the argument.  MachineError gives the error
	involved in evaluating its argument using machine arithmetic.
	MicroscopePlot plots an expression at a magnification such that the
	granularity of machine arithmetic becomes obvious.  MicroscopicErrorPlot
	uses high precision (30 digits) to calculate the "true" value and plots
	the difference between the machine precision result and the "true" value
	measured in ulps (units last place).

	Both MicroscopePlot and MicroscopicErrorPlot work by evaluating the
	expression at consecutive machine numbers in a neighborhood of
	a point to come up with a set of discrete points.  The option
	Joined can be True (the points are to be simply connected with
	straight line segments), False (the points are not to be connected),
	or Real.  The choice of Real causes the resulting plot to represent
	a mapping from the set of real numbers to the set of machine numbers
	(MicroscopePlot) or the error in such a mapping (MicroscopicErrorPlot).
	Note that care must be exercised in interpreting the result when
	Joined -> Real is used.  Functions in computer libraries are
	designed to map the set of machine numbers into the set of machine
	numbers.  It is not fair to blame the function for the error incurred
	in rounding a real number to the nearest machine number prior to the
	function receiving the machine number.
*)


If[Not@ValueQ[Ulps::usage],Ulps::usage =
"Ulps is a unit of error in machine arithmetic. An ulp is the distance \
between two consecutive machine numbers and varies with the size of the \
numbers involved."];

If[Not@ValueQ[Ulp::usage],Ulp::usage =
"Ulp[x] gives the size of an ulp for numbers near x."];

If[Not@ValueQ[MachineError::usage],MachineError::usage =
"MachineError[f, x -> a] gives the error involved in evaluating f at x == a \
using machine arithmetic."];

If[Not@ValueQ[MicroscopePlot::usage],MicroscopePlot::usage =
"MicroscopePlot[f, {x, a}] plots the expression f (which is supposed to contain \
a single variable x) in a small neighborhood of a. MicroscopePlot[f, {x, a, n}] \
plots f from a - n ulps to a + n ulps."];

If[Not@ValueQ[MicroscopicErrorPlot::usage],MicroscopicErrorPlot::usage =
"MicroscopicErrorPlot[f, {x, a}] plots the error incurred by using machine \
arithmetic to evaluate the expression f (which is supposed to contain a single \
variable x) in a small neighborhood of a. MicroscopicErrorPlot[f, {x, a, n}] \
plots the error from a - n ulps to a + n ulps. The scale on the vertical axis \
represents ulps in the result."];

(*========================================================================================*)

(* Integrated in to ComputerArithmetic *)

(*========================================================================================*)

Begin["`Private`"]

Unprotect[MicroscopePlot, MicroscopicErrorPlot]
Options[MicroscopicErrorPlot] = Options[MicroscopePlot] = Joined -> True;
Attributes[MicroscopicErrorPlot] = Attributes[MachineError] = HoldFirst;



Ulp[x_] := ulp[x];

MachineError[f_, x_ -> a_] :=
    Module[{aa = N[SetPrecision[a, 30], 30], ff = Hold[f]},
	ff = ReleaseHold[SetPrecision[f, 30]];
	ff = Re[N[ff /. x -> aa, 30]];
	(SetPrecision[N[f /. x -> a], 30] - ff)/ulp[ff] Ulps
	];

MicroscopePlot[e_, {x_Symbol, a_, n_Integer:30}, opt___] :=
    Module[{joined, ans},
	ans /; (joined = Joined /. {opt} /. (Joined -> True);
		MemberQ[{True, False, Real}, joined] &&
			(ans = mic[e, x, a, n, joined]; True))
	] /; NumberQ[N[a]];

MicroscopicErrorPlot[ee_, {x_Symbol, a_, n_Integer:30}, opt___] :=
    Module[{joined, ans, e = Hold[ee]},
	ans /; (joined = Joined /. {opt} /. (Joined -> True);
		MemberQ[{True, False, Real}, joined] &&
			(ans = micer[e, x, a, n, joined]; True))
	] /; NumberQ[N[a]];

ulp[_DirectedInfinity] := DirectedInfinity[ ]

ulp[x_] := 
    Module[{t = Abs[N[x]], u},
	If[t < $MinMachineNumber,
		$MinMachineNumber,
		u = N[2^Floor[Log[2, t $MachineEpsilon] - .5]];
		t = t - ReleaseHold[t - u];
		If[t == 0. || t == 2u, 2u, u]
		]
	];
 
machpts[e_, {x_, a_}, n_] :=
    Module[{h, xtab, ytab, i, na = N[a]},
        h = ulp[na (1-$MachineEpsilon)];
        xtab = Table[i h, {i, -n, n}] + na;
        ytab = e /. x -> xtab;
        {xtab, ytab, na, e /. x -> na}]; 
         
mic[e_, x_, a_, n_, joined_] :=
    Module[{h, xtab, ytab, i, x0, y0, pts, lines, ao}, 
        {xtab, ytab, x0, y0} = machpts[e, {x, a}, n];
	xtab = (xtab - x0)/ulp[x0];
	ytab -= y0;
	ytab *= N[2^-Round[Log[2, Max[Abs[Re[ytab]]]]]];
	ao = {Min[xtab], Min[Re[ytab]]};
	pts = {PointSize[.2/n],
		Table[Point[{xtab[[i]], ytab[[i]]}], {i, Length[xtab]}]};
	If[joined === Real,
            xtab = (Drop[xtab, 1] + Drop[xtab, -1])/2; 
            xtab = Flatten[Table[{xtab[[i]], xtab[[i]]}, {i, Length[xtab]}]];
            ytab = Flatten[Table[{ytab[[i]], ytab[[i]]}, {i, Length[ytab]}]]; 
            ytab = Drop[Drop[ytab, 1], -1]];   
	If[joined === False,
	    lines = {},
	    lines = {Thickness[.001],
		Line[Table[{xtab[[i]], ytab[[i]]}, {i, Length[xtab]}]]}];
	Show[Graphics[{pts, lines}, AxesOrigin -> ao, Axes -> True,
        AspectRatio -> 1/GoldenRatio,
	    PlotRange -> All, Ticks -> {{{0,ToString[x0]}}, {{0,ToString[y0]}}}]]
	];
                 
micer[e_, x_, a_, n_, joined_] :=
    Module[{h, xtab, ytab, yytab, i, x0, y0, pts, lines, ao, sxtab, ee},
        {xtab, ytab, x0, y0} = machpts[ReleaseHold[e], {x, a}, n];
        xtab = SetPrecision[xtab, 30];
        ytab = SetPrecision[ytab, 30];
	ee = ReleaseHold[SetPrecision[e, 30]];
        yytab = N[ytab - Re[(ee /. x -> xtab)]]/ulp[y0];
	sxtab = N[xtab - SetPrecision[x0, 30]]/ulp[x0];
	ao = {Min[sxtab], 0};
	pts = {PointSize[.2/n],
		Table[Point[{sxtab[[i]], yytab[[i]]}], {i, Length[sxtab]}]};
	If[joined === Real,
            xtab = (Drop[xtab, 1] + Drop[xtab, -1])/2;
            yytab = ee /. x -> xtab; 
            xtab = Flatten[Table[{xtab[[i]], xtab[[i]]}, {i, Length[xtab]}]];
            yytab = Flatten[Table[{yytab[[i]],yytab[[i]]}, {i,Length[yytab]}]];
            ytab = Flatten[Table[{ytab[[i]], ytab[[i]]}, {i, Length[ytab]}]];
            ytab = Drop[Drop[ytab, 1], -1];
            yytab = N[ytab - yytab]/ulp[y0];
	    sxtab = N[xtab - SetPrecision[x0, 30]]/ulp[x0]
	    ];
	If[joined === False,
	    lines = {},
	    lines = {Thickness[.001],
		Line[Table[{sxtab[[i]], yytab[[i]]}, {i, Length[xtab]}]]}];
	Show[Graphics[{pts, lines}, AxesOrigin -> ao, Axes -> True,
        AspectRatio -> 1/GoldenRatio,
	    PlotRange -> All, Ticks -> {{{0,ToString[x0]}}, Automatic}]]
	];



End[]  (* Microscope`Private` *)



Protect[MicroscopePlot, MicroscopicErrorPlot];


(* NumericalMath`Microscope` *)

(* Tests:
Microscope[Log[x], {x, 7}]
Microscope[Log[x], {x, 7, 20}]
Microscope[Log[x], {x, 7}, Joined -> False]
Microscope[Log[x], {x, 7, 20}, Joined -> False]
Microscope[Log[x], {x, 7}, Joined -> True]
Microscope[Log[x], {x, 7, 20}, Joined -> True]
Microscope[Log[x], {x, 7}, Joined -> Real]
Microscope[Log[x], {x, 7, 20}, Joined -> Real]
Microscope[Sqrt[x], {x, 5}]
Microscope[Sqrt[x], {x, 5, 20}]
Microscope[Sqrt[x], {x, 5}, Joined -> False]
Microscope[Sqrt[x], {x, 5, 20}, Joined -> False]
Microscope[Sqrt[x], {x, 5}, Joined -> True]
Microscope[Sqrt[x], {x, 5, 20}, Joined -> True]
Microscope[Sqrt[x], {x, 5}, Joined -> Real]
Microscope[Sqrt[x], {x, 5, 20}, Joined -> Real]
Microscope[ArcTanh[x], {x, .5}]
Microscope[ArcTanh[x], {x, .5, 20}]
Microscope[ArcTanh[x], {x, .5}, Joined -> False]
Microscope[ArcTanh[x], {x, .5, 20}, Joined -> False]
Microscope[ArcTanh[x], {x, .5}, Joined -> True]
Microscope[ArcTanh[x], {x, .5, 20}, Joined -> True]
Microscope[ArcTanh[x], {x, .5}, Joined -> Real]
Microscope[ArcTanh[x], {x, .5, 20}, Joined -> Real]
Microscope[ArcTanh[x], {x, .05}]
Microscope[ArcTanh[x], {x, .05, 20}]
Microscope[ArcTanh[x], {x, .05}, Joined -> False]
Microscope[ArcTanh[x], {x, .05, 20}, Joined -> False]
Microscope[ArcTanh[x], {x, .05}, Joined -> True]
Microscope[ArcTanh[x], {x, .05, 20}, Joined -> True]
Microscope[ArcTanh[x], {x, .05}, Joined -> Real]
Microscope[ArcTanh[x], {x, .05, 20}, Joined -> Real]
Microscope[ArcTanh[x], {x, .00005}]
Microscope[ArcTanh[x], {x, .00005, 20}]
Microscope[ArcTanh[x], {x, .00005}, Joined -> False]
Microscope[ArcTanh[x], {x, .00005, 20}, Joined -> False]
Microscope[ArcTanh[x], {x, .00005}, Joined -> True]
Microscope[ArcTanh[x], {x, .00005, 20}, Joined -> True]
Microscope[ArcTanh[x], {x, .00005}, Joined -> Real]
Microscope[ArcTanh[x], {x, .00005, 20}, Joined -> Real]

MicroscopicError[Log[x], {x, 7}]
MicroscopicError[Log[x], {x, 7, 20}]
MicroscopicError[Log[x], {x, 7}, Joined -> False]
MicroscopicError[Log[x], {x, 7, 20}, Joined -> False]
MicroscopicError[Log[x], {x, 7}, Joined -> True]
MicroscopicError[Log[x], {x, 7, 20}, Joined -> True]
MicroscopicError[Log[x], {x, 7}, Joined -> Real]
MicroscopicError[Log[x], {x, 7, 20}, Joined -> Real]
MicroscopicError[Sqrt[x], {x, 5}]
MicroscopicError[Sqrt[x], {x, 5, 20}]
MicroscopicError[Sqrt[x], {x, 5}, Joined -> False]
MicroscopicError[Sqrt[x], {x, 5, 20}, Joined -> False]
MicroscopicError[Sqrt[x], {x, 5}, Joined -> True]
MicroscopicError[Sqrt[x], {x, 5, 20}, Joined -> True]
MicroscopicError[Sqrt[x], {x, 5}, Joined -> Real]
MicroscopicError[Sqrt[x], {x, 5, 20}, Joined -> Real]
MicroscopicError[ArcTanh[x], {x, .5}]
MicroscopicError[ArcTanh[x], {x, .5, 20}]
MicroscopicError[ArcTanh[x], {x, .5}, Joined -> False]
MicroscopicError[ArcTanh[x], {x, .5, 20}, Joined -> False]
MicroscopicError[ArcTanh[x], {x, .5}, Joined -> True]
MicroscopicError[ArcTanh[x], {x, .5, 20}, Joined -> True]
MicroscopicError[ArcTanh[x], {x, .5}, Joined -> Real]
MicroscopicError[ArcTanh[x], {x, .5, 20}, Joined -> Real]
MicroscopicError[ArcTanh[x], {x, .05}]
MicroscopicError[ArcTanh[x], {x, .05, 20}]
MicroscopicError[ArcTanh[x], {x, .05}, Joined -> False]
MicroscopicError[ArcTanh[x], {x, .05, 20}, Joined -> False]
MicroscopicError[ArcTanh[x], {x, .05}, Joined -> True]
MicroscopicError[ArcTanh[x], {x, .05, 20}, Joined -> True]
MicroscopicError[ArcTanh[x], {x, .05}, Joined -> Real]
MicroscopicError[ArcTanh[x], {x, .05, 20}, Joined -> Real]
MicroscopicError[ArcTanh[x], {x, .00005}]
MicroscopicError[ArcTanh[x], {x, .00005, 20}]
MicroscopicError[ArcTanh[x], {x, .00005}, Joined -> False]
MicroscopicError[ArcTanh[x], {x, .00005, 20}, Joined -> False]
MicroscopicError[ArcTanh[x], {x, .00005}, Joined -> True]
MicroscopicError[ArcTanh[x], {x, .00005, 20}, Joined -> True]
MicroscopicError[ArcTanh[x], {x, .00005}, Joined -> Real]
MicroscopicError[ArcTanh[x], {x, .00005, 20}, Joined -> Real]
*)
