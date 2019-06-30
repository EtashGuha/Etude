(* ::Package:: *)

(* :Name: NumericalMath`NLimit` *)

(* :Title: Numerical Evaluation of Limits, Derivatives, and Infinite Sums *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package evaluates numerical limits by forming a short sequence of
function values corresponding to different values of the limit
variable.  This sequence is passed to SequenceLimit, or
Euler's transformation is used to find an approximation to the
limit.  Numerical sums are evaluated using Euler's transformation
rather than using NSum.  Numerical differentiation is implemented
using Richardson extrapolation via Euler's transformation.
*)

(* :Context: NumericalMath`NLimit` *)

(* :Package Version: 2.0 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.
*)

(* :History:
	Originally written by Jerry B. Keiper, February 1990.
	Revised by Jerry B. Keiper, December 1990.
	Added Cauchy integral formula, April 1994.
*)

(* :Keywords: Limit, SequenceLimit, D, Euler's transformation,
	numerical limit, numerical differentiation
*)

(* :Source:
	Carl-Erik Froberg, Numerical Mathematics: Theory and Computer
	Applications, Benjamin/Cummings, 1985, p. 309.

	M. Abramowitz & I. Stegun, Handbook of Mathematical Functions,
	Dover, 1972, p. 914.
*)

(* :Warnings: There is no guarantee that the result will be correct.  As with
	all numerical techniques for evaluating the infinite via finite
	samplings, it is possible to fool NLimit, ND, and EulerSum
	into giving an incorrect results.
*)

(* :Mathematica Version: 2.0 *)

(* :Limitations: NLimit cannot find limits whose values are any form
	of Infinity.  The workaround is to change the expression so that
	the result is a finite number.  EulerSum does not verify
	convergence, and as a result can give finite results for divergent
	sums.
*)

(* :Discussion:
	This package implements numerical summation of infinite series
	via Euler's transformation (EulerSum).  It also provides a mechanism
	for numerically evaluating limits by forming a sequence of values
	that approach the limit and then passing the sequence in the
	apropriate way to SequenceLimit or the engine for EulerSum.
	Finally numerical differentiation (ND) is implemented using
	NLimit.

	EulerSum has the following options:
	   WorkingPrecision--the number of digits of precision to be used in the
		computations.
	   Terms--the number of terms to be explicitly included in the sum
		before extrapolation.
	   ExtraTerms--the number of terms to be used in the extrapolation
		process.  ExtraTerms must be at least 2.
	   EulerRatio--the fixed ratio to be used in the transformation.
		Ideally this would be the limit of the ratio of successive
		terms in the series.  EulerRatio can also be a list of ratios
		or {ratio, degree + 1} pairs in which case the algorithm is
		applied recursively with the various ratios being used
		successively.  The default is Automatic.

	NLimit has the following options:
	   Direction--the direction of motion as the limit is approached.
	   WorkingPrecision--the number of digits of precision to be used in the
		computations.
	   Scale--for infinite limits, the sequence of values used begins
		at Scale and proceeds 10 Scale, 100 Scale, 1000 Scale, ....
		For finite limits approaching the point x0, the sequence of
		values used begins at x0+Scale and preceeds x0+Scale/10,
		x0+Scale/100, x0+Scale/1000, ....  For finite limits the
		Direction option determines the direction of approach.
		Only the magnitude of Scale is important.
	   Terms--the total number of terms generated in the sequence.
	   Method--either SequenceLimit or EulerSum.
	   WynnDegree--the value to be used for the option WynnDegree of
		SequenceLimit.

	ND has the following options:
	   WorkingPrecision--the number of digits of precision to be used in the
		computations.
	   Scale--the size of the steps taken when evaluating the
		expression at various points.  Several divided differences
		are be formed and the sequence of divided differences is
		extrapolated to a limit.  Scale is the largest size of
		steps used; smaller steps are used as the sequence progresses.
	   Terms--the total number of terms generated in the sequence.
	   Method--either EulerSum or NIntegrate.
*)



If[Not@ValueQ[EulerSum::usage],EulerSum::usage =
"EulerSum[expr, range] uses Euler's transformation to numerically \
evaluate the sum of expr over the specified range."];

If[Not@ValueQ[EulerRatio::usage],EulerRatio::usage =
"EulerRatio is an option to EulerSum, which specifies the parameter to use \
in the generalized Euler transformation. EulerRatio can be Automatic, a \
single ratio, or a list of ratios or {ratio, degree + 1} pairs. In the \
case of a list the various ratios are used successively in iterated \
Euler transformations."];

If[Not@ValueQ[NLimit::usage],NLimit::usage =
"NLimit[expr, x->x0] numerically finds the limiting value of expr as x \
approaches x0."];

If[Not@ValueQ[WynnDegree::usage],WynnDegree::usage=
"WynnDegree is an option to NLimit that specifies the degree used \
with Wynn's epsilon algorithm for approximating the limit of a sequence."];

If[Not@ValueQ[ND::usage],ND::usage =
"ND[expr, x, x0] gives a numerical approximation to the derivative of expr \
with respect to x at the point x0. ND[expr, {x, n}, x0] gives a numerical \
approximation to the n-th derivative of expr with respect to x at the \
point x0. With the method EulerSum, ND attempts to evaluate expr at x0. \
If this fails, ND fails."];

If[Not@ValueQ[ExtraTerms::usage],ExtraTerms::usage = "ExtraTerms is an option to EulerSum. It specifies \
the number of terms to be used in the extrapolation process after Terms \
terms are explicitly included. ExtraTerms must be at least 2."];

If[Not@ValueQ[Scale::usage],Scale::usage = "Scale is an option of NLimit and ND. It specifies the initial \
stepsize in the sequence of steps or the radius of the circle of integration \
for Cauchy's integral formula in ND."];

If[Not@ValueQ[Terms::usage],Terms::usage = "Terms is an option of EulerSum, NLimit, and ND. In EulerSum \
it specifies the number of terms to be included explicitly before the \
extrapolation process begins. In NLimit and ND it specifies the total \
number of terms to be used."];

Begin["`Private`"]

Unprotect[NLimit, EulerSum, ND];

Options[EulerSum] = {WorkingPrecision -> MachinePrecision, Terms -> 5,
			ExtraTerms -> 7, EulerRatio -> Automatic}
Options[NLimit] = {Direction -> Automatic,
			WorkingPrecision -> MachinePrecision, Scale -> 1,
			Terms -> 7, Method -> EulerSum, WynnDegree -> 1}
Options[ND] = {WorkingPrecision -> MachinePrecision, Scale -> 1,
		Terms -> 7, Method -> EulerSum}


EulerSum[expr_, range_, opts___] :=
    Module[{ans}, ans /; (ans = EulerSum0[expr, range, opts]) =!= $Failed]

ND[expr_, x_, x0_, opts___] := ND[expr, {x, 1}, x0, opts] /; (Head[x] =!= List);

ND[expr_, {x_, n_}, x0_, opts___] :=
    Module[{prec = WorkingPrecision  /. {opts} /. Options[ND],
	h = SetPrecision[Scale /. {opts} /. Options[ND], Infinity],
	terms = Terms /. {opts} /. Options[ND],
	meth = Method /. {opts} /. Options[ND],
	nder},
	nder /; (nder = If[meth === NIntegrate,
			ndni[expr, x, n, x0, prec, h],
			nd[expr, x, n, x0, prec, h, terms]
			];
		$Failed =!= nder)
	]

ndni[f_, x_ , n_, x0_, prec_, r_] :=
    Module[{ft, t, z, ans},
	ft = f /. x -> (x0 + z);
	ans = NIntegrate[z = r E^(I t); ft/z^n, {t, 0, 2Pi},
		Method -> Trapezoidal, WorkingPrecision -> prec];
	If[!NumberQ[ans], Return[$Failed]];
	Gamma[n+1]/(2 Pi) ans
	];

nd[expr_ , x_ , n_, x0_, prec_, h_, terms_] :=
    Module[{seq, eseq, dseq, nx0, nh, i, j},
	(* form a sequence of divided differences and use Richardson
	extrapolation to eliminate most of the truncation error. *)
	nx0 = N[x0, prec];
	nh = N[2h, prec];
	(*
	If[!NumberQ[nh] || !NumberQ[nx0],
		Message[NLimit::baddir, nx0, nh];
		Return[$Failed]];
	*)
	eseq = Table[Null, {n+1}];
	seq = Table[Null, {terms}];
	Do[eseq[[i]] = N[expr /. x -> nx0+(i-1)nh, prec], {i,n/2+1}];
	Do[	Do[eseq[[2i-1]] = eseq[[i]], {i,Floor[n/2]+1,2,-1}];
		nh /= 2;
		Do[eseq[[i]]=N[expr /. x->nx0+(i-1)nh,prec],{i,2,n+1,2}];
		dseq = eseq;
		While[Length[dseq]>1,dseq=Drop[dseq,1]-Drop[dseq,-1]];
		seq[[j]] = dseq[[1]]/(nh^n), {j, terms}];
	Do[	j = 2^i;
		seq = (j Drop[seq,1] - Drop[seq,-1])/(j-1),
		{i, terms-1}];
	seq[[1]]
	];

EulerSum::eslim = "The limit of summation `1` is incompatible with the stepsize `2`."

EulerSum0[expr_,
	{x_, DirectedInfinity[a_], DirectedInfinity[b_], step_:1}, opts___] :=
    Module[{suma, sumb},
	suma = N[Arg[-step/a]];
	If[!NumberQ[suma] || Abs[suma] > 10.^-10,
		Message[EulerSum::eslim, DirectedInfinity[a], step];
		Return[$Failed]
		];
	sumb = N[Arg[step/b]];
	If[!NumberQ[sumb] || Abs[sumb] > 10.^-10,
		Message[EulerSum::eslim, DirectedInfinity[b], step];
		Return[$Failed]
		];
	suma = EulerSumInf[expr, {x, 0, -step}, opts];
	If[suma === $Failed, Return[$Failed]];
	sumb = EulerSumInf[expr, {x, step, step}, opts];
	If[sumb === $Failed, Return[$Failed]];
	suma + sumb
	];

EulerSum0[expr_, {x_}, opts___] := EulerSumInf[expr, {x, 1, 1}, opts]

EulerSum0[expr_, {x_, a_:1, Infinity}, opts___] :=
		EulerSumInf[expr, {x, a, 1}, opts];

EulerSum0[expr_, {x_, a_:1, b_}, opts___] :=
		EulerSum0[expr, {x, a, b, 1}, opts];

EulerSum0[expr_, {x_, a_, DirectedInfinity[dir_], step_:1}, opts___] :=
    Module[{ans},
	If[Head[a] === DirectedInfinity,
		Message[EulerSum::eslim, a, step];
		Return[$Failed]
		];
	If[dir == 0,
		Message[EulerSum::eslim, DirectedInfinity[dir], step];
		Return[$Failed]
		];
	ans = N[Arg[step/dir]];
	If[!NumberQ[ans] || Abs[ans] > 10.^-10,
		Message[EulerSum::eslim, DirectedInfinity[dir], step];
		Return[$Failed]
		];
	EulerSumInf[expr, {x, a, step}, opts]
	];

EulerSum0[expr_, {x_, a_, b_, step_:1}, opts___] :=
    Module[{suma, sumb},
	If[a == b, Return[0]];
	If[Head[a] === DirectedInfinity,
		Message[EulerSum::eslim, a, step];
		Return[$Failed]
		];
	If[Head[b] === DirectedInfinity,
		Message[EulerSum::eslim, b, step];
		Return[$Failed]
		];
	suma = EulerSumInf[expr, {x, a, step}, opts];
	If[suma === $Failed, Return[$Failed]];
	sumb = EulerSumInf[expr, {x, b+step, step}, opts];
	If[sumb === $Failed, Return[$Failed]];
	suma - sumb
	];

EulerSumInf[expr_, {x_, a_, step_:1}, opts___] :=
    Module[{prec = WorkingPrecision  /. {opts} /. Options[EulerSum],
	terms = Terms /. {opts} /. Options[EulerSum],
	exterms = ExtraTerms /. {opts} /. Options[EulerSum],
	ratio = EulerRatio /. {opts} /. Options[EulerSum]},
	eulersum[expr,x,a,step,terms,exterms,prec,ratio]
	];

EulerSum::short = "The sequence `1` is too short."

EulerSum::badrat= "EulerRatio -> `1` is invalid."

eulersum[expr_,x_,a_,step_,terms_,exterms_,prec_,ratio_] :=
    Module[{b = a + terms step,sum, ratlist},
	(* send a list of terms to es for extrapolated summation and then
	include the initial terms in the series explicitly *)
	sum = Table[N[expr, prec], {x, b, b+(exterms-1)step, step}];
	ratlist = If[NumberQ[ratio], {ratio}, ratio];
	If[(ratlist =!= Automatic) && (Head[ratlist] =!= List),
		Message[EulerSum::badrat, ratio];
		Return[$Failed]];
	sum = es[sum, ratlist];
	(*
	If[!NumberQ[sum], Return[$Failed]];
	*)
	If[!FreeQ[sum, $Failed], Return[$Failed]];
	Together[sum + Sum[N[expr, prec], {x, a, b-step, step}]]
	];

EulerSum::erpair = "The EulerRatio `1` is invalid."

EulerSum::ernum = "Encountered the nonnumerical EulerRatio of `1`."

EulerSum::zrat = "Encountered an EulerRatio of 0. (This may be a result of two equal EulerRatio values.)"

newre[re_, rat_] := If[MatchQ[re, {_, _}],
				{(re[[1]] - rat)/(1-rat), re[[2]]},
				(re - rat)/(1 - rat)];
es[seq_, ratiolist_] :=
    Module[{newseq=seq, rat, dd=1, ratpow=1, r1r, i, j, l=Length[seq], tmp},
	(* Transform the sequence of terms into a new sequence of terms based
	on the given ratio.  Recursively repeat the transform with the
	various values of ratio in the ratiolist.  Finally, sum the
	transformed sequence. *)
	If[l < 2,
		Message[EulerSum::short, seq];
		Return[$Failed]];
	rat = Apply[Plus, Abs[seq]];
	If[rat == 0., Return[rat]];
	rat = If[ratiolist === Automatic, Automatic, ratiolist[[1]]];
	If[rat === Automatic,
		rat = seq[[l]]/seq[[l-1]];
		If[!NumberQ[rat],
			Message[EulerSum::ernum, rat];
			Return[$Failed]
			],
	    (* else *)
		If[Head[rat] === List && Length[rat] == 2, {rat, dd} = rat];
		If[!IntegerQ[dd],
			Message[EulerSum::erpair, ratiolist[[1]]];
			Return[$Failed]
			]
		];
	If[rat == 0,
		Message[EulerSum::zrat];
		Return[$Failed];
		];
	ratpow = 1;
	Do[newseq[[i]] /= (ratpow *= rat), {i,2,l}];
	Do[Do[newseq[[j]]  = Together[newseq[[j]]-newseq[[j-1]]], {j,l,i,-1}],
		{i,2,l}];
	r1r = rat/(1-rat);
	ratpow = 1;
	Do[newseq[[i]] *= (ratpow *= r1r), {i, 2, l}];
	newseq = Together[newseq];
	If[(ratiolist === Automatic) || (Length[ratiolist] == 1),
		Apply[Plus, newseq]/(1-rat),
	    (* else *)
		r1r = Together[newre[#, rat]& /@ Drop[ratiolist, 1]];
		tmp = es[Drop[newseq, dd], r1r];
		If[!FreeQ[tmp, $Failed], Return[$Failed]];
		(Apply[Plus, Take[newseq, dd]] + tmp)/(1-rat)
		]
	];

NLimit[e_, x_ -> x0_, opts___] :=
    Module[{prec = WorkingPrecision  /. {opts} /. Options[NLimit],
	dir = Direction /. {opts} /. Options[NLimit],
	scale = SetPrecision[Scale /. {opts} /. Options[NLimit], Infinity],
	terms = Terms /. {opts} /. Options[NLimit],
	meth = Method /. {opts} /. Options[NLimit],
	degree = WynnDegree /. {opts} /. Options[NLimit],
	limit},
	limit /; ((ToString[meth] === "SequenceLimit") || (ToString[meth] === "EulerSum")) &&
		(limit = If[Head[x0] === DirectedInfinity,
			infLimit[e,x,x0,prec,scale,terms,degree,meth],
			If[dir === Automatic, dir = -1];
			scale = -Sign[dir] Abs[scale];
			finLimit[e,x,x0,prec,scale,terms,degree,meth]];
		$Failed =!= limit)
	];

NLimit::notnum = "The expression `1` is not numerical at the point `2` == `3`."

NLimit::baddir = "Cannot approach `1` from the direction `2`."

NLimit::noise = "Cannot recognize a limiting value.  This may be due to  \
noise resulting from roundoff errors in which case higher WorkingPrecision,  \
fewer Terms, or a different Scale might help."

infLimit[e_, x_, x0_, prec_, scale_, terms_, degree_, meth_] :=
    Module[{seq, ne, nx, i, dirscale, limit, tmp},
	(* form a sequence of values that approach the limit at infinity
	and the extrapolate *)
	If[Length[x0] != 1, Return[$Failed]];
	dirscale = N[x0[[1]], prec];
	dirscale = Abs[N[scale, prec]] dirscale/Abs[dirscale];
	If[!NumberQ[dirscale],
		Message[NLimit::baddir, x0, dirscale];
		Return[$Failed]];
	seq = Table[Null, {terms}];
	tmp = Do[	nx = dirscale 10^(i-1);
		ne = N[e /. x -> nx, prec];
		If[!NumberQ[ne] || ne === Overflow[] || ne === Underflow[],
			Message[NLimit::notnum, ne, x, nx];
			Return[$Failed]];
		seq[[i]] = ne, {i,terms}];
    If[tmp === $Failed, Return[$Failed]];
	limit = If[meth === EulerSum,
			tmp = es[Drop[seq,1] - Drop[seq,-1], Automatic];
			If[!FreeQ[tmp, $Failed], Return[$Failed]];
			seq[[1]] + tmp,
		    (* else *)
			NumericalMath`NSequenceLimit[seq, Method->{"WynnEpsilon", "Degree" -> degree}]];
	(*  we must check that we have a reasonable answer *)
	If[limit == seq[[1]] || (NumberQ[limit] &&
				Abs[limit-seq[[1]]] > Abs[limit-seq[[-1]]]),
		limit, Message[NLimit::noise]; $Failed]
	];

finLimit[e_, x_, x0_, prec_, scale_, terms_, degree_, meth_] :=
    Module[{seq, ne, nx, nx0, i, dirscale, limit, tmp},
	(* form a sequence of values that approach the limit at x0 
	and the extrapolate *)
	nx0 = N[x0, prec];
	dirscale = N[scale, prec];
	If[!NumberQ[dirscale] || !NumberQ[nx0],
		Message[NLimit::baddir, nx0, dirscale];
		Return[$Failed]];
	seq = Table[Null, {terms}];
	tmp = Do[	nx = nx0 + dirscale/10^(i-1);
		ne = N[e /. x -> nx, prec];
		If[!NumberQ[ne],
			Message[NLimit::notnum, ne, x, nx];
			Return[$Failed]];
		seq[[i]] = ne, {i,terms}];
    If[tmp === $Failed, Return[$Failed]];
	limit = If[meth === EulerSum,
			tmp = es[Drop[seq,1] - Drop[seq,-1], Automatic];
			If[!FreeQ[tmp, $Failed], Return[$Failed]];
			seq[[1]] + tmp,
		    (* else *)
			NumericalMath`NSequenceLimit[seq, Method->{"WynnEpsilon", "Degree" -> degree}]];
	(*  we must check that we have a reasonable answer *)
	If[NumberQ[limit] && Abs[limit-seq[[1]]] > Abs[limit-seq[[-1]]],
		limit, Message[NLimit::noise]; $Failed]
	];

End[]  (* NumericalMath`NLimit`Private` *)

SetAttributes[{NLimit, EulerSum, ND}, ReadProtected];
Protect[NLimit, EulerSum, ND];

 (* NumericalMath`NLimit` *)

