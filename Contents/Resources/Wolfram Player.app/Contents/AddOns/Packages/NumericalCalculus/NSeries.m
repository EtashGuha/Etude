(* ::Package:: *)

(* :Name: NumericalMath`NSeries` *)

(* :Title: Numerical Derivation of a Series *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package uses the function Fourier to numerically derive
the series of an analytic function in the complex plane by sampling
the function at discrete points around a circle centered at the
center of expansion.
*)

(* :Context: NumericalMath`NSeries` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.
*)

(* :History:
	Originally written by Jerry B. Keiper, July 1994.
*)

(* :Keywords: series *)

(* :Source:
*)

(* :Warnings: None. *)

(* :Mathematica Version: 2.0 *)

(* :Limitations:
	No effort is made to justify the precision in each of the
	coefficients of the series.  NSeries is unable to recognize
	small numbers that should in fact be zero.  Chop is often
	needed to eliminate these spurious residuals.
*)




If[Not@ValueQ[NSeries::usage],NSeries::usage =
"NSeries[f, {x, x0, n}] gives a numerical approximation to the series \
expansion of f about the point x == x0 through (x-x0)^n. It does this \
by sampling f at points around on a circle centered at x0 and using \
Fourier. The option Radius specifies the radius of the circle."];

If[Not@ValueQ[Radius::usage],Radius::usage =
"Radius is an option to NSeries that specifies the radius of the circle \
around which the function is to be sampled."];

Begin["`Private`"]

Options[NSeries] = {WorkingPrecision -> MachinePrecision, Radius -> 1};

NSeries[f_, {x_, x0_, n_Integer}, opts___] :=
    Module[{p, r, ans},
	ans /; (p = WorkingPrecision /. {opts} /. Options[NSeries];
		r = Radius /. {opts} /. Options[NSeries];
		TrueQ[p > 0] && TrueQ[r > 0] && TrueQ[n > 0] &&
		    (ans = nseries0[f, {x, x0, n}, p, r]) =!= $Failed)
	];

nseries0[f_, {x_, x0_, n_}, p_, r_] :=
    Module[{data, m, k, nx0, n2pi},
	m = 2^(2 + Ceiling[Log[2, n]]);
	nx0 = N[x0, p];
	n2pi = N[2 Pi I, p]/m;
	data = Table[N[f /. x -> nx0 + r E^(k n2pi), p], {k, 0, m-1}];
	If[!VectorQ[data, NumberQ], Return[$Failed]];
	data = InverseFourier[data];
	data = Join[Take[data, -n], Take[data, n+1]] (r^Range[n, -n, -1]);
	SeriesData[x, x0, data m^(-1/2), -n, n+1, 1]
	];


End[]  (* NumericalMath`NSeries`Private` *)

(* NumericalMath`NSeries` *)

(* :Tests:
Rationalize[Chop[NSeries[Exp[x], {x, 0, 5}]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 7}]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 5}, Radius -> 1/8]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 7}, Radius -> 1/8]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 5},
	WorkingPrecision -> 40, Radius -> 1/8]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 7},
	WorkingPrecision -> 40, Radius -> 1/8]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 5}, Radius -> 3]]]
Rationalize[Chop[NSeries[Exp[x], {x, 0, 7}, Radius -> 3]]]

Chop[NSeries[Zeta[s], {s, 1, 5}]]
Chop[NSeries[Zeta[s]/(s-1)^2, {s, 1, 5}]]
Chop[%/%%]
*)

