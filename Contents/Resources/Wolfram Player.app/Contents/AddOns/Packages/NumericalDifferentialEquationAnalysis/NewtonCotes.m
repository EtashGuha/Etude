(* ::Package:: *)

(* :Name: NumericalIntegrationAnalysis`NewtonCotes` *)

(* :Title: Finding Open and Closed Newton-Cotes Quadrature Formulas *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package finds weights and abscissas for Newton-Cotes quadrature 
formulas of arbitrary order.  Both open and closed formulas are
supported.
*)

(* :Context: NumericalIntegrationAnalysis`NewtonCotes` *) 

(* :Package Version: Mathematica 2.0 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.
*)

(* :History:
	Original version by Jerry B. Keiper, May 1990.
	Revised by Jerry B. Keiper, January 1991.
*)

(* :Keywords:
	Newton-Cotes, quadrature, open formula, closed formula, weights,
	abscissas.
*)

(* :Source: Any elementary numerical analysis textbook. *)

(* :Mathematica Version: 2.0 *)

(* :Limitation: *)

Unprotect[NewtonCotesWeights, NewtonCotesError];

If[Not@ValueQ[NewtonCotesWeights::usage],NewtonCotesWeights::usage =
"NewtonCotesWeights[n, a, b] gives a list of the pairs {abscissa, weight} \
for the elementary n point Newton-Cotes formula for quadrature on the \
interval a to b. NewtonCotesWeights has the option \
QuadratureType, which can be Open or Closed."];

If[Not@ValueQ[NewtonCotesError::usage],NewtonCotesError::usage =
"NewtonCotesError[n, f, a, b] gives the error in the elementary n point \
Newton-Cotes quadrature formula for the function f on an interval from a \
to b. NewtonCotesError has the option \
QuadratureType, which can be Open or Closed."];

If[Not@ValueQ[QuadratureType::usage],QuadratureType::usage =
"QuadratureType is an option of NewtonCotesWeights and NewtonCotesError, \
which can be Open or Closed."];

If[Not@ValueQ[Type::usage],Type::usage =
"Type is an obsolete option of NewtonCotesWeights and NewtonCotesError, \
replaced by QuadratureType."];

If[Not@ValueQ[Open::usage],Open::usage =
"Open is a value for the QuadratureType option of NewtonCotesWeights and \
NewtonCotesError."];

If[Not@ValueQ[Closed::usage],Closed::usage =
"Closed is a value for the QuadratureType option of NewtonCotesWeights and \
NewtonCotesError."];

Options[NewtonCotesWeights] = {QuadratureType -> Closed};
Options[NewtonCotesError] = {QuadratureType -> Closed};

Begin["`Private`"]

NewtonCotesWeights[n_Integer, a_, b_, opts___] :=
	Module[{qtype, ans},
	  (
		ans
	  ) /; (
		 If[FreeQ[{opts}, QuadratureType] && !FreeQ[{opts}, Type],
		    qtype = Type /. {opts};
		    If[!(qtype===Open || qtype===Closed),
	 	       qtype = QuadratureType /. Options[NewtonCotesWeights] ];
		    Message[NewtonCotesWeights::obs, qtype],
		    qtype = QuadratureType /. {opts} /.
						Options[NewtonCotesWeights]
		 ];
		 ans = NCW[n, a, b, qtype];
		 ans =!= $Failed )
	] /; n > 0

NewtonCotesWeights::obs =
"Warning: option Type is obsolete, using QuadratureType -> ``."

NewtonCotesError[n_Integer, f_, a_, b_,  opts___] :=
	Module[{qtype, ans},
	  (
		ans
	  ) /; (
		 If[FreeQ[{opts}, QuadratureType] && !FreeQ[{opts}, Type],
		    qtype = Type /. {opts};
                    If[!(qtype===Open || qtype===Closed),       
                       qtype = QuadratureType /. Options[NewtonCotesError] ]; 
		    Message[NewtonCotesError::obs, qtype],	
                    qtype = QuadratureType /. {opts} /.
                                                Options[NewtonCotesError]
                 ];
		 ans = NCE[n, f, a, b, qtype];
		 ans =!= $Failed )
	] /; n > 0

NewtonCotesError::obs =
"Warning: option Type is obsolete, using QuadratureType -> ``."


NCW[n_, a_, b_, type_] :=
	Module[{i, w, x, h, m, rhs},
		If[type === Closed,
			If[n < 2, Return[$Failed]],
			If[n < 1, Return[$Failed]]];
		h = If[type === Closed, 2/(n-1), 2/n];
		x = Table[i h, {i, 0, n-1}] - (n-1)h/2;
		m = Table[x^i, {i,n-1}];
		PrependTo[m, Table[1, {n}]];
		rhs = Table[(1+Sign[(-1)^i])/(i+1), {i,0,n-1}];
		w = Expand[LinearSolve[m, rhs](b-a)/2];
		x = Expand[(x (b-a) + (b+a))/2];
		Transpose[{x, w}]
		]

NCE[n_, f_, a_, b_, type_] :=
	Module[{t, fint, fs, x, w, len, nn},
		If[type === Closed,
			If[n < 2, Return[$Failed]],
			If[n < 1, Return[$Failed]]];
		nn = 2 Floor[(n+1)/2]+1;
		fs = t^(nn-1)/(nn-1)!;
		{x, w} = Transpose[NCW[n, a, b, type]];
		fs = Expand[fs /. t->x] . w;
		fint = (b^nn - a^nn)/nn!;
		Expand[fs - fint] Derivative[nn-1][f]
		]

End[] (* NumericalMath`NewtonCotes`Private` *)

Protect[NewtonCotesWeights, NewtonCotesError];
