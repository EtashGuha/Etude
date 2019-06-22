(* ::Package:: *)

(* :Name: NumericalIntegrationAnalysis`GaussianQuadrature` *)

(* :Title: Coefficients for Gaussian Quadrature *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package calculates the weights and abscissas for the elementary 
Gaussian quadrature rule with n points on the interval a to b.  
It also calculates the error in the elementary Gaussian quadrature 
formula with n points on the interval a to b.
*)

(* :Context: NumericalIntegrationAnalysis` *)

(* :Package Version: Mathematica 2.0 *)

(* :Copyright: Copyright 1990-2007,  Wolfram Research, Inc.
*)

(* :History:
	Original version by Jerry B. Keiper, May 1990.
	Revised by Jerry B. Keiper, December 1990.
	Revised by Anton Antonov, September 2005.
*)

(* :Keywords: Gaussian quadrature, abscissas, weights *)

(* :Source: Any elementary numerical analysis textbook. *)

(* :Mathematica Version: 2.0 *)

(* :Limitation: *)

If[Not@ValueQ[GaussianQuadratureWeights::usage],GaussianQuadratureWeights::usage =
"GaussianQuadratureWeights[n, a, b, prec] gives a list of the pairs \
{abscissa, weight} to prec digits precision for the elementary n-point \
Gaussian quadrature formula for quadrature on the interval a to b. The \
argument prec is optional."];

If[Not@ValueQ[GaussianQuadratureError::usage],GaussianQuadratureError::usage =
"GaussianQuadratureError[n, f, a, b, prec] gives the leading term in the \
error in the elementary n-point Gaussian quadrature formula to prec digits \
precision for the function f on an interval from a to b. The argument prec \
is optional."];

Unprotect[GaussianQuadratureWeights, GaussianQuadratureError];

Begin["`Private`"]

GaussianQuadratureWeights[n_Integer, a_, b_, prec_:MachinePrecision] :=
	GQW[n, a, b, prec] /; (n > 0 && (prec === MachinePrecision || prec > 0))

GaussianQuadratureError[n_Integer, f_, a_, b_, prec_:MachinePrecision] :=
	GQE[n, f, a, b, prec] /; (n > 0 && (prec === MachinePrecision || prec > 0))


GQW[n_, a_, b_, prec_] :=
    Module[{i, w, x, m, rhs, t, ew},
        If[n == 1, Return[N[{{(b + a)/2, b - a}}, prec]]];
        {x, w, ew} = NIntegrate`GaussKronrodRuleData[n, prec];

        x = x[[2 Range[n]]];
        w = w[[2 Range[n]]];
        ew = ew[[2 Range[n]]];

        w = ew - w;
        w *= (b - a)/Total[w];
        x = Rescale[x , {0, 1}, {a, b}];

        Transpose[{x, w}]
    ];


GQE[n_, f_, a_, b_, prec_] :=
    Module[{t, fint, fs, x, w, len},
	fs = t^(2n)/(2n)!;
	{x, w} = Transpose[GQW[n, -len, len, prec]];
	fs = Expand[fs /. t->x] . w;
	fint = 2 len^(2n+1)/(2n+1)!;
	(Expand[fs - fint] /. len -> (b-a)/2) Derivative[2n][f]
    ];

End[] (* `Private` *)

SetAttributes[{GaussianQuadratureWeights, GaussianQuadratureError}, ReadProtected]
Protect[GaussianQuadratureWeights, GaussianQuadratureError];
