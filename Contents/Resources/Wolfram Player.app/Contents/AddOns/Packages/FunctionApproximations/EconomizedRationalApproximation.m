(* ::Package:: *)

(* :Name: FunctionApproximations`EconomizedRationalApproximation` *)

(* :Title: Economized Rational Approximations *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package finds economized rational approximations to
functions over intervals.
*)

(* :Context: FunctionApproximations` *)

(* :Package Version: 2.0 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.
*)

(* :History:
	Originally written by Jerry B. Keiper, October 1989.
	Extensively revised by Jerry B. Keiper, December 1990.
*)

(* :Keywords:
	functional approximation, Chebyshev approximation, Pade
	approximation, rational approximation
*)

(* :Source:
	Carl-Erik Froberg, Numerical Mathematics: Theory and Computer
		Applications, Benjamin/Cummings, 1985, pp. 250-266.

	A. Ralston & P. Rabinowitz, A First Course in Numerical Analysis
		(2nd. ed.) McGraw-Hill, New York, 1978.
*)

(* :Warnings: None. *)

(* :Mathematica Version: 2.0 *)

(* :Limitations: 
	The Pade` approximation returned may have lower degree than
	requested.  This results when the leading coefficient turns out
	to be 0 or when the system of linear equations specifying the
	coefficients fails to have a solution.

	The economized rational approximation can fail in similar ways
	as well as in more subtle ways.  An example is
	EconomizedRationalApproximation[Sin[x] x^2, {x, {-1, 1}, 3, 4}].
	Normally such problems can be avoided by changing the degree
	of the numerator or the denominator by 1.  For example,
	EconomizedRationalApproximation[Sin[x] x^2, {x, {-1, 1}, 4, 4}].
	If this does not succeed, breaking the symmetry of the interval
	will fix the problem.  For example,
	EconomizedRationalApproximation[Sin[x] x^2, {x, {-1, 1001/1000},
	3, 4}].
*)

(* :Discussion:
	Economized rational approximations are not unique.  The
	implementation in this package uses a sequence of Pade'
	approximations with numerator and denominator degrees as
	nearly equal as possible.  If singularities are encountered,
	a few other Pade' approximations are tried, but the search is
	not exhaustive, so failure does not imply that such an 
	economized rational approximation does not exist.

	For both Pade' and economized rational approximations, if
	there is a zero or a pole at the center (of expansion or of
	the interval in question), it is first divided out, the
	regularized function is approximated, and finally the zero
	or pole multiplied back in.  This tends to minimize the
	relative error rather than the absolute error.  For example
	era = EconomizedRationalApproximation[Sin[x] x^2, {x, {-1, 1}, 4, 4}];
	Plot[(era - Sin[x] x^2)/x^3, {x, -1,1}]
*)

If[Not@ValueQ[EconomizedRationalApproximation::usage],EconomizedRationalApproximation::usage = 
"EconomizedRationalApproximation[func, {x, {x0, x1}, m, k}] gives the \
economized rational approximation to func (a function of the variable x) where \
(x0, x1) is the interval for which the approximation is to be good and m and k \
are the degrees of the numerator and denominator, respectively."];

Unprotect[EconomizedRationalApproximation];

Begin["`Private`"]

EconomizedRationalApproximation::nser =
"A simple series expansion for `1` could not be found."

EconomizedRationalApproximation::sing =
"The series expansion of `1` has an irrational singularity at `2`."

EconomizedRationalApproximation::degnum =
"The function `1` has a zero of order `2` at `3` that is greater than the \
requested degree of the numerator."

EconomizedRationalApproximation::degden =
"The function `1` has a pole of order `2` at `3` that is greater than the \
requested degree of the denominator."

EconomizedRationalApproximation::sol = "The requested economized rational \
approximation could not be found."

EconomizedRationalApproximation[f_, {x_, {x0_, x1_}, m_Integer, k_Integer}] :=
	Module[{answer = era[f, {x, {x0, x1}, m, k}]},
		answer /; answer =!= $Failed];

seriesf[f_, {x_, x0_, m_, k_}] :=
	Module[{lack = 1, fseries, trys = 1, ordbias, clist, n = m+k+2},
		(* return both the list of coefficients in the Laurent
		    expansion and the order of the zero (positive) or the
		    order of the pole (negative) at x0. *)
		While[lack > 0 && trys++ < 4,
			fseries = Series[f, {x, x0, n+lack}];
			If[Head[fseries] =!= SeriesData,
				Message[EconomizedRationalApproximation::nser, f];
				Return[$Failed]
				];
			If[fseries[[6]] =!= 1,
				Message[EconomizedRationalApproximation::sing, f, x0];
				Return[$Failed]
				];
			ordbias = fseries[[4]];
			lack = n - Abs[ordbias] - (fseries[[5]] - fseries[[4]])
			];
		If[trys > 4, Message[EconomizedRationalApproximation::nser, f]; Return[$Failed]];
		If[ordbias > m,
			Message[EconomizedRationalApproximation::degnum, f, ordbias, x0];
			Return[0]
			];
		If[ordbias + k < 0,
			Message[EconomizedRationalApproximation::degden, f, -ordbias, x0];
			Return[DirectedInfinity[ ]]
			];
		clist = fseries[[3]];
		If[Length[clist] < n - Abs[ordbias],
			lack = Table[0, {n - Abs[ordbias] - Length[clist]}];
			clist = Join[clist, lack]
			];
		{clist, ordbias}
		]

Pade1[{cl_, ordbias_}, mm_Integer, kk_Integer] :=
	Module[{i, mk1, m, k, temp, coef, rhs},
		(* return the list of coefficients in the numerator and
		    the list of coefficients in the denominator. *)
		m = mm - Max[{0, ordbias}];
		k = kk + Min[{0, ordbias}];
		mk1 = m+k+1;
		rhs = Take[cl, mk1];
		coef = IdentityMatrix[mk1];
		temp = Join[Table[0, {k}], -rhs];
		Do[coef[[i+m+1]] = Take[temp, {k+1-i, -1-i}], {i,k}];
		temp = LinearSolve[Transpose[coef],rhs];
		While[Head[temp] === LinearSolve,
			mk1--;
			k--;
			rhs = Take[rhs, mk1];
			coef = Take[#, mk1]& /@ Take[coef, mk1];
			temp = LinearSolve[Transpose[coef],rhs];
			];
		{Take[temp, m+1], Join[{1}, Take[temp, -k]]}
		];

era[f_, {x_, {x0_, x1_}, m_, k_}] :=
	Module[{i, j, alpha, xm = (x1+x0)/2, rlist, dlist,
			mk, mk1, t, twopow, dn, dd, actdd, ordbias, cl},
		(* this is best explained by reference to Ralston and
		    Rabinowitz, pp 309 ff. *)
		If[m < 0 || k < 0, Return[$Failed]];
		t = seriesf[f, {x, xm, m, k}];
		If[Head[t] =!= List, Return[t]];
		cl = t[[1]];
		ordbias = t[[2]];
		dn = m - Max[{0, ordbias}];
		dd = k + Min[{0, ordbias}];
		mk = m + k - Max[{0, Abs[ordbias]}];
		mk1 = mk + 1;
		i = mk1;
		While[ i > 0,
			rlist[i] = Pade1[{cl, 0}, dn, dd];
			t = rlist[i][[2]];
			If[ i == mk1,
			    (* if the requested Pade' approximation does
				    not exist, the degree of the denominator
				    will be less than dd.  correct for this. *)
			    actdd = dd = Length[t] - 1;
			    mk1 = dn + dd + 1;
			    mk = mk1 - 1;
			    rlist[mk1] = rlist[i];
			    i = mk1
			    ];
			If[ Length[t] - 1 == dd,
			    dlist[i] = Sum[cl[[i+2-j]] t[[j]], {j, Length[t]}],
			    dlist[i] = 0
			    ];
			While[ dlist[i] == 0 && i < mk1,
			    (* we have a problem, try an alternative
				    Pade' approximation. *)
			    dn--; dd++;
			    If[ dn < 0 || dd > actdd,
				Message[EconomizedRationalApproximation::sol];
				Return[$Failed]
				];
			    rlist[i] = Pade1[{cl, 0}, dn, dd];
			    t = rlist[i][[2]];
			    If[ Length[t] - 1 == dd,
				dlist[i] = Sum[cl[[i+2-j]] t[[j]],
						{j, Length[t]}],
				dlist[i] = 0
				]
			    ];
			If[dn > dd, dn--, dd-- ];
			If[dn > dd, dn--, dd-- ];
			i -= 2
			];
		t = CoefficientList[ChebyshevT[mk1, x], x];
		twopow = 2^mk;
		alpha = (x1-x0)/2;
		Do[	If[dlist[mk1] === 0,
			    cl = 0,
			    cl = dlist[mk1]/dlist[i] alpha^(mk1-i) t[[i+1]];
			    ];
			rlist[i] *= cl/twopow,
			{i, mk-1, 1, -2}
			];
		t = -dlist[mk1] alpha^mk1 t[[1]]/twopow;
		cl = Table[x^i, {i, 0, Max[{m, k}]}];
		Do[	{dn, dd} = rlist[i];
			rlist[i] = {dn . Take[cl, Length[dn]],
					dd . Take[cl, Length[dd]]},
			{i, mk1, 1, -2}
			];
		dn = t + Sum[rlist[i][[1]], {i, mk1, 1, -2}];
		dd = Sum[rlist[i][[2]], {i, mk1, 1, -2}];
		(Expand[dn x^Max[{0, ordbias}]]/
			Expand[dd x^Max[{0, -ordbias}]]) /. x -> (x-xm)
		];

End[]  (* `Private` *)

Protect[EconomizedRationalApproximation];
