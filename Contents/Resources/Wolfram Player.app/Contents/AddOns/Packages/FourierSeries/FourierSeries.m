(* ::Package:: *)

(*:Mathematica Version: 4.0 *)

(*:Package Version: 1.14 *)

(* :Copyright: Copyright 1990-2008, Wolfram Research, Inc.*)

(*:Name: FourierSeries` *)

(*:Context: FourierSeries` *)

(*:Title: Numerical Fourier Series *)

(*:Author: Wolfram Research, Inc. *)

(*:Summary:
This package implements numerical approximations to Fourier transforms
(exponential and trigonometric), Fourier coefficients and series (exponential
and trigonometric) and related numerical approximations, and discrete-time
Fourier transforms and related numerical approximations.
*)

(*:Keywords: Fourier, transform *)

(*:Requirements: None. *)

(*:Source:
	R. A. Roberts, C. T. Mullis, Digital Signal Processing,
	Addison-Wesley Publishing Company, 1987, Figure 4.1.1,
	"Four classes of Fourier transforms."
*)

(*:History:
	Version 1.1 by Eran Yehudai, October 1990.
	Versions 1.2-1.13 by ECM and Jeff Adams (Wolfram Research),
		January 1991-June 1997.
	Version 1.14, ECM, Jan. 1998.  Removed rules for FourierTransform,
	  InverseFourierTransform, FourierCosTransform,
	  InverseFourierCosTransform, FourierSinTransform, and
	  InverseFourierSinTransform.  Revised rules for NFourierTransform
	  and NInverseFourierTransform.  Added rules for NFourierCosTransform,
	  NInverseFourierCosTransform, NFourierSinTransform, and
	  NInverseFourierSinTransform.  Added rules for DTFourierTransform,
	  InverseDTFourierTransform, NDTFourierTransform, and
	  NInverseDTFourierTransform (discrete-time Fourier transforms).
	  Replaced FourierExpSeries with FourierSeries,
	  FourierExpSeriesCoefficient with FourierCoefficient,
	  NFourierExpSeries with NFourierSeries, and
	  NFourierExpSeriesCoefficient with NFourierCoefficient.  Added
	  rules for InverseFourierCoefficient and NInverseFourierCoefficient.
	  Revised FourierTrigSeries and NFourierTrigSeries.
	  Replaced FourierSinSeriesCoefficient with FourierSinCoefficient,
	  NFourierSinSeriesCoefficient with NFourierSinCoefficient,
	  FourierCosSeriesCoefficient with FourierCosCoefficient, and
	  NFourierCosSeriesCoefficient with NFourierCosCoefficient. 
	Sept. 2008 by Brian Van Vertloo - removed symbolic functions for inclusion into System`.
		
*)

(* :Limitations: *)

(*:Discussion:

continuous-time, continuous-frequency:
	FourierTransform & InverseFourierTransform (kernel)
	(These functions map a function in continuous-time to a
		 function in continuous-frequency and visa versa.)
	NFourierTransform & NInverseFourierTransform (package)
	(These functions are numerical approximations.) 

discrete-time, discrete-frequency:
	Fourier & InverseFourier (kernel)
	(These functions map a list to a list, not a function to a function.)

discrete-time, continuous-frequency:
    DTFourierTransform & InverseDTFourierTransform (kernel as FourierSequenceTransform and Inverse...)
	(These functions map a function in discrete-time to a
		 function in continuous-frequency and visa versa.)
	NDTFourierTransform & NInverseDTFourierTransform (package)
	(These functions are numerical approximations.)

discrete-frequency, continuous-time:
	FourierCoefficient & InverseFourierCoefficient (kernel)
	(These functions map a function in continuous-time to a
		 function in discrete-frequency and visa versa.)
	NFourierCoefficient & NInverseFourierCoefficient (package)
	(These functions are numerical approximations.)
	FourierSinCoefficient & FourierCosCoefficient (kernel)
	(These functions map a function in continuous-time to a
		function in discrete-frequency.)
	NFourierSinCoefficient & NFourierCosCoefficient (package)
	(These functions are numerical approximations.)
        FourierSeries and FourierTrigSeries (package) each take a function and
		give an approximation to that function based on a truncated
		series expansion.  NFourierSeries and NFourierTrigSeries do the
		same thing, but use numerical approximations to the
		coefficients.

*)

BeginPackage["FourierSeries`"]



(* ===================== numeric Fourier Transform ======================= *)

If[Not@ValueQ[NFourierTransform::usage],NFourierTransform::usage = 
"NFourierTransform[expr, t, w] gives the numerical value of the Fourier \
transform of expr, a function of t, at w. It is defined by \
1/Sqrt[2 Pi] NIntegrate[expr Exp[I w t], {t, -Infinity, Infinity}]. \
NFourierTransform[expr, t, w, FourierParameters -> {a, b}] gives the \
numerical value of the parameterized transform \
Sqrt[Abs[b]/(2 Pi)^(1-a)] NIntegrate[expr Exp[I b w t], \
{t, -Infinity, Infinity}]."];

If[Not@ValueQ[NInverseFourierTransform::usage],NInverseFourierTransform::usage = 
"NInverseFourierTransform[expr, w, t] gives the numerical value of the inverse \
Fourier transform of expr, a function of w, at t. It is defined by \
1/Sqrt[2 Pi] NIntegrate[expr Exp[-I w t], {w, -Infinity, Infinity}]. \
NInverseFourierTransform[expr, w, t, FourierParameters -> {a, b}] gives \
the numerical value of the parameterized inverse transform \
Sqrt[Abs[b]/(2 Pi)^(1+a)] NIntegrate[expr Exp[-I b w t], \
{w, -Infinity, Infinity}]."];

Options[NFourierTransform] = Options[NInverseFourierTransform] =
	Sort[Join[{FourierParameters -> {0, 1}}, Options[NIntegrate]]]
	
(* ==================== numeric Fourier Sin Transform ==================== *)

If[Not@ValueQ[NFourierSinTransform::usage],NFourierSinTransform::usage =
"NFourierSinTransform[expr, t, w] gives the numerical value of the Fourier sine \
transform of expr, a function of t, at w. It is defined by \
Sqrt[2/Pi] NIntegrate[expr Sin[w t], {t, 0, Infinity}]. \
NFourierSinTransform[expr, t, w, FourierParameters -> {a, b}] gives \
the numerical value of the parameterized transform \
2 Sqrt[Abs[b]/(2 Pi)^(1-a)] NIntegrate[expr Sin[b w t], {t, 0, Infinity}]."];

If[Not@ValueQ[NInverseFourierSinTransform::usage],NInverseFourierSinTransform::usage =
"NInverseFourierSinTransform[expr, w, t] gives the numerical value of the \
inverse Fourier sine transform of expr, a function of w, at t. It is defined by \
Sqrt[2/Pi] NIntegrate[expr Sin[w t], {w, 0, Infinity}]. \
NInverseFourierSinTransform[expr, w, t, FourierParameters -> {a, b}] gives \
the numerical value of the parameterized inverse transform \
2 Sqrt[Abs[b]/(2 Pi)^(1+a)] NIntegrate[expr Sin[b w t], {w, 0, Infinity}]."];

Options[NFourierSinTransform] = Options[NInverseFourierSinTransform] =
        Sort[Join[{FourierParameters -> {0, 1}}, Options[NIntegrate]]]
	

(* ==================== numeric Fourier Cos Transform ==================== *)

If[Not@ValueQ[NFourierCosTransform::usage],NFourierCosTransform::usage =
"NFourierCosTransform[expr, t, w] gives the numerical value of the Fourier \
cosine transform of expr, a function of t, at w. It is defined by \
Sqrt[2/Pi] NIntegrate[expr Cos[w t], {t, 0, Infinity}]. \
NFourierCosTransform[expr, t, w, FourierParameters -> {a, b}] gives \
the numerical value of the parameterized transform \
2 Sqrt[Abs[b]/(2 Pi)^(1-a)] NIntegrate[expr Cos[b w t], {t, 0, Infinity}]."];

If[Not@ValueQ[NInverseFourierCosTransform::usage],NInverseFourierCosTransform::usage =
"NInverseFourierCosTransform[expr, w, t] gives the numerical value of the \
inverse Fourier cosine transform of expr, a function of w, at t. It is defined \
by Sqrt[2/Pi] NIntegrate[expr Cos[w t], {w, 0, Infinity}]. \
NInverseFourierCosTransform[expr, w, t, FourierParameters -> {a, b}] gives \
the numerical value of the parameterized inverse transform \
2 Sqrt[Abs[b]/(2 Pi)^(1+a)] NIntegrate[expr Cos[b w t], {w, 0, Infinity}]."];

Options[NFourierCosTransform] = Options[NInverseFourierCosTransform] =
        Sort[Join[{FourierParameters -> {0, 1}}, Options[NIntegrate]]]
	

(* =================== discrete-time Fourier Transform ===================== *)

(* w may be symbolic or real-valued *)
(*
  DTFourierTransform[expr, n, w] gives the discrete time Fourier transform
  of expr, a function of n, at w.  It is defined by
  Sum[expr Exp[2 Pi I n w], {n, -Infinity, Infinity}], where the sum is
  periodic in w, with a period of 1.
  DTFourierTransform[expr, n, w, FourierParameters -> {a, b}] gives the
  parameterized transform Abs[b]^((1-a)/2) Sum[expr Exp[2 Pi I b n w],
  {n, -Infinity, Infinity}], where the period of the sum is 1/Abs[b]. *)
If[Not@ValueQ[DTFourierTransform::usage],DTFourierTransform::usage = 
"DTFourierTransform[expr, n, w] gives the discrete time Fourier transform \
of expr, a function of n, at w. This transform is periodic in w, with a period \
of 1. DTFourierTransform[expr, n, w, FourierParameters -> {a, b}] gives the \
discrete time Fourier transform of expr, where the transform has a period of \
1/Abs[b]."];

(* NOTE: DTFourierTransform makes use of Sum which currently does not
	support options.  So there is no need for DTFourierTransform to
	support additional Sum options.  *)
Options[DTFourierTransform] = {FourierParameters -> {1, 1}}

(* n may be symbolic or integer-valued *)
(*
  InverseDTFourierTransform[expr, w, n] gives the inverse discrete time
  Fourier transform of expr, a periodic function of w, at n.  It is defined by
  Integrate[expr Exp[-2 Pi I n w], {w, -1/2, 1/2}], where the period of expr 
  is 1.  InverseDTFourierTransform[expr, w, n, FourierParameters -> {a, b}] 
  gives the parameterized inverse transform 
  Abs[b]^((1+a)/2) Integrate[expr Exp[-2 Pi I b n w],
  {w, -1/(2 Abs[b]), 1/(2 Abs[b])}], where the period of expr is 1/Abs[b].
*)
If[Not@ValueQ[InverseDTFourierTransform::usage],InverseDTFourierTransform::usage = 
"InverseDTFourierTransform[expr, w, n] gives the inverse discrete time Fourier \
transform of a periodic function of w, at n.  The function is equal to expr for \
-1/2 <= w <= 1/2, and has a period of 1. \
InverseDTFourierTransform[expr, w, n, FourierParameters -> {a, b}] gives \
the inverse discrete time Fourier transform of a periodic function, where the \
function is equal to expr for -1/(2 Abs[b]) <= w <= 1/(2 Abs[b]), and has a \
period of 1/Abs[b]."];

Options[InverseDTFourierTransform] =
        Sort[Join[{FourierParameters -> {1, 1}}, Options[Integrate]]]


(* =============== numeric discrete-time Fourier Transform =============== *)

(* w must be real-valued *)
(*
  NDTFourierTransform[expr, n, w] gives the numerical value of the discrete time
  Fourier transform of expr, a function of n, at w.  It is defined by
  NSum[expr Exp[2 Pi I n w], {n, -Infinity, Infinity}], where the sum 
  is periodic in w, with a period of 1.
  NDTFourierTransform[expr, n, w, FourierParameters -> {a, b}] gives 
  the numerical value of the parameterized transform
  Abs[b]^((1-a)/2) NSum[expr Exp[2 Pi I b n w], {n, -Infinity, Infinity}],
  where the period of the sum is 1/Abs[b]. *)
If[Not@ValueQ[NDTFourierTransform::usage],NDTFourierTransform::usage = 
"NDTFourierTransform[expr, n, w] gives the numerical value of the discrete time \
Fourier transform of expr, a function of integer n, at real values of w.  This \
transform is periodic in w, with a period of 1. \
NDTFourierTransform[expr, n, w, FourierParameters -> {a, b}] gives \
the numerical value of the discrete-time Fourier transform of expr at real \
values of w, where the transform has a period of 1/Abs[b]."];

Options[NDTFourierTransform] =
	Sort[Join[{FourierParameters -> {1, 1}}, Options[NSum]]]

(* n must be integer-valued *)
(*
  NInverseDTFourierTransform[expr, w, n] gives the numerical value of the 
  inverse discrete time Fourier transform of expr, a periodic function of w, at
  integer n.  It is defined by NIntegrate[expr Exp[-2 Pi I n w],
  {w, -1/2, 1/2}], where the period of expr is 1.
  NInverseDTFourierTransform[expr, w, n, FourierParameters -> {a, b}] gives
  the numerical value of the parameterized inverse transform
  Abs[b]^((1+a)/2) NIntegrate[expr Exp[-2 Pi I b n w],
  {w, -1/(2 Abs[b]), 1/(2 Abs[b])}], where the period of expr is 1/Abs[b]. *)
If[Not@ValueQ[NInverseDTFourierTransform::usage],NInverseDTFourierTransform::usage =
"NInverseDTFourierTransform[expr, w, n] gives the numerical value of the \
inverse discrete time Fourier transform of a periodic function of w, at \
integer values of n. The function is equal to expr for -1/2 <= w <= 1/2, \
and has a period of 1. \
NInverseDTFourierTransform[expr, w, n, FourierParameters -> {a, b}] gives the \
numerical value of the inverse discrete time Fourier transform of a periodic \
function, where the function is equal to expr for \
-1/(2 Abs[b]) <= w <= 1/(2 Abs[b]), and has a period of 1/Abs[b]."];

Options[NInverseDTFourierTransform] =
        Sort[Join[{FourierParameters -> {1, 1}}, Options[NIntegrate]]]
	

(* t may be symbolic or real-valued *)
(*
  InverseFourierCoefficient[expr, n, t] gives the Fourier exponential series
  representation of a periodic function of t, whose coefficients are
  given by expr, a function of n.  It is defined by
  Sum[expr Exp[-2 Pi I n t], {n, -Infinity, Infinity}], where the sum has a 
  period of 1.
  InverseFourierCoefficient[expr, n, t, FourierParameters -> {a, b}] gives
  the inverse coefficient
  Abs[b]^((1+a)/2) Sum[expr Exp[-2 Pi I b n t], {n, -Infinity, Infinity}], where
  the period of the sum is 1/Abs[b]. *)
If[Not@ValueQ[InverseFourierCoefficient::usage],InverseFourierCoefficient::usage = 
"InverseFourierCoefficient[expr, n, t] gives the Fourier exponential series \
representation of a periodic function of t, where the function has a period \
of 1, and the series coefficients are given by expr, indexed by n. \
InverseFourierCoefficient[expr, n, t, FourierParameters -> {a, b}] gives the \
Fourier exponential series representation of a periodic function of t, where \
the function has a period of 1/Abs[b]."];

(* NOTE: InverseFourierCoefficient makes use of Sum which currently does not
	support options.  So there is no need for InverseFourierCoefficient to
	support additional Sum options.  *)
Options[InverseFourierCoefficient] = {FourierParameters -> {0, 1}}


(* ==== numeric  Fourier Series (discrete-frequency Fourier Transform) === *)

(* n must be integer-valued *)
(*
  NFourierCoefficient[expr, t, n] gives the numerical value of the nth 
  coefficient in the Fourier exponential series expansion of the periodic 
  function of t that is equal to expr for -1/2 <= t <= 1/2, repeating with 
  period 1, defined by NIntegrate[expr Exp[2 Pi I n t], {t, -1/2, 1/2}].
  NFourierCoefficient[expr, t, n, FourierParameters -> {a, b}] gives
  Abs[b]^((1-a)/2) NIntegrate[expr Exp[2 Pi I b n t],
  {t, -1/(2 Abs[b]), 1/(2 Abs[b])}], the numerical value of the nth coefficient
  in the Fourier exponential series expansion of the periodic function of t
  that is equal to expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), repeating with
  period 1/Abs[b]. *)
If[Not@ValueQ[NFourierCoefficient::usage],NFourierCoefficient::usage = 
"NFourierCoefficient[expr, t, n] gives the numerical value of the nth \
coefficient in the Fourier exponential series expansion of the periodic function \
of t that is equal to expr for -1/2 <= t <= 1/2, and has a period of 1. \
NFourierCoefficient[expr, t, n, FourierParameters -> {a, b}] gives \
the numerical value of the nth coefficient in the Fourier exponential series \
of the periodic function of t that is equal to expr for \
-1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), and has a period of 1/Abs[b]."];

Options[NFourierCoefficient] =
	Sort[Join[{FourierParameters -> {1, 1}}, Options[NIntegrate]]]

(* t must be real-valued *)
(*
  NInverseFourierCoefficient[expr, n, t] gives the numerical value of the
  Fourier exponential series representation of a periodic function of t, whose
  coefficients are given by expr, a function of n.  It is defined by 
  NSum[expr Exp[-2 Pi I n t], {n, -Infinity, Infinity}], where the sum
  has a period of 1.
  NInverseFourierCoefficient[expr, n, t, FourierParameters -> {a, b}]
  gives the numerical value of the inverse coefficient
  Abs[b]^((1+a)/2) NSum[expr Exp[-2 Pi I b n t], {n, -Infinity, Infinity}], 
  where the period of the sum is 1/Abs[b]. *)
If[Not@ValueQ[NInverseFourierCoefficient::usage],NInverseFourierCoefficient::usage =
"NInverseFourierCoefficient[expr, n, t] gives the numerical value of the \
Fourier exponential series representation of a periodic function of t, where the \
function has a period of 1, and the series coefficients are given by expr, \
indexed by n. \
NInverseFourierCoefficient[expr, n, t, FourierParameters -> {a, b}] gives \
the numerical value of the Fourier exponential series representation of a \
periodic function of t, where the function has a period of 1/Abs[b]."];

Options[NInverseFourierCoefficient] =
	Sort[Join[{FourierParameters -> {0, 1}}, Options[NSum]]]

(* k must be nonnegative integer-valued *)
(* Note that the numeric aspect of NFourierSeries is that NFourierCoefficient
        is used, instead of FourierCoefficient.  NFourierSeries uses Sum just
        as FourierSeries does;  it does *not* use NSum.  So "NFourierSeries"
        may be a misuse of the N* naming convention.  But the alternative, 
        subsuming NFourierSeries into FourierSeries (i.e., by picking
        FourierCoefficient or NFourierCoefficient based on an option), wouldn't
        work well either.  Note that FourierCoefficient supports Integrate
        options while NFourierCoefficient supports NIntegrate options. *)
(*
  NFourierSeries[expr, t, k] finds the numerical values of the Fourier 
  coefficients c[n], -k <= n <= k, for the periodic function of t that is equal 
  to expr for -1/2 <= t <= 1/2, repeating with period 1, and gives
  Sum[c[n] Exp[-2 Pi I n t], {n, -k, k}], the kth order Fourier exponential 
  series approximation to the function.
  NFourierSeries[expr, t, k, FourierParameters -> {a, b}] gives
  Abs[b]^((1+a)/2) Sum[c[n] Exp[-2 Pi I b n t], {n, -k, k}], the 
  kth order Fourier exponential series approximation to the periodic function of
  t that is equal to expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), repeating with
  period 1/Abs[b], where the c[n] are numerical coefficients. *)
If[Not@ValueQ[NFourierSeries::usage],NFourierSeries::usage =
"NFourierSeries[expr, t, k] uses numerical coefficients to give the kth order \
Fourier exponential series approximation to the periodic function of t that is \
equal to expr for -1/2 <= t <= 1/2, and has a period of 1. \
NFourierSeries[expr, t, k, FourierParameters -> {a, b}] uses numerical \
coefficients to give the kth order Fourier exponential series approximation to \
the periodic function of t that is equal to expr for \
-1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), and has a period of 1/Abs[b]."];

Options[NFourierSeries] = Options[NFourierCoefficient]
	

(*  ================= numeric Fourier Trigonometric Series ================ *)

(* Note: old syntax...
  NFourierSinSeriesCoefficient[expr, {x, x0, x1}, n] gives the numeric value of
  the coefficient of Sin[2Pi n x / (x1-x0)] in the Fourier trigonometric series
  expansion of expr (n > 0).
*)
(*
  NFourierSinCoefficient[expr, t, n] gives the numerical value of the 
  coefficient of Sin[2 Pi n t], n > 0, in the Fourier trigonometric series 
  expansion of the periodic function of t that is equal to expr for
  -1/2 <= t <= 1/2, repeating with period 1, defined by
  NIntegrate[expr Sin[2 Pi n t], {t, -1/2, 1/2}].
  NFourierSinCoefficient[expr, t, n, FourierParameters -> {a, b}] gives
  Abs[b]^((1-a)/2) NIntegrate[expr Sin[2 Pi b n t],
  {t, -1/(2 Abs[b]), 1/(2 Abs[b])}], the numerical value of the coefficient of
  Sin[2 Pi b n t] in the Fourier trigonometric series expansion of the periodic
  function of t that is equal to expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]),
  repeating with period 1/Abs[b]. *)
If[Not@ValueQ[NFourierSinCoefficient::usage],NFourierSinCoefficient::usage =
"NFourierSinCoefficient[expr, t, n] gives the numerical value of the \
coefficient of Sin[2 Pi n t], n > 0, in the Fourier trigonometric series \
expansion of the periodic function of t that is equal to expr for \
-1/2 <= t <= 1/2, and has a period of 1. \
NFourierSinCoefficient[expr, t, n, FourierParameters -> {a, b}] gives the \
numerical value of the coefficient of Sin[2 Pi b n t] in the Fourier \
trigonometric series expansion of the periodic function of t that is equal to \
expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), and has a period of 1/Abs[b]."];

(* Note: old syntax...
  NFourierCosSeriesCoefficient[expr, {x, x0, x1}, n] gives the numeric value of
  the coefficient of Cos[2Pi n x / (x1-x0)] in the Fourier trigometric series
  expansion of expr (n >= 0).
*)
(*
  NFourierCosCoefficient[expr, t, n] gives the numerical value of the
  coefficient of Cos[2 Pi n t], n >= 0, in the Fourier trigonometric series 
  expansion of the periodic function of t that is equal to expr for
  -1/2 <= t <= 1/2, repeating with period 1, defined by
  If[n===0, 1, 2] NIntegrate[expr Cos[2 Pi n t], {t, -1/2, 1/2}].
  NFourierCosCoefficient[expr, t, n, FourierParameters -> {a, b}] gives 
  If[n===0, 1, 2] Abs[b]^((1-a)/2) NIntegrate[expr Cos[2 Pi b n t],
  {t, -1/(2 Abs[b]), 1/(2 Abs[b])}], the numerical value of the coefficient of
  Cos[2 Pi b n t] in the Fourier trigonometric series expansion of the periodic
  function of t that is equal to expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]),
  repeating with period 1/Abs[b]. *)
If[Not@ValueQ[NFourierCosCoefficient::usage],NFourierCosCoefficient::usage =
"NFourierCosCoefficient[expr, t, n] gives the numerical value of the \
coefficient of Cos[2 Pi n t], n >= 0, in the Fourier trigonometric series \
expansion of the periodic function of t that is equal to expr for \
-1/2 <= t <= 1/2, and has a period of 1. \
NFourierCosCoefficient[expr, t, n, FourierParameters -> {a, b}] \
gives the numerical value of the coefficient of Cos[2 Pi b n t] in the Fourier \
trigonometric series expansion of the periodic function of t that is equal to \
expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), and has a period of 1/Abs[b]."];

(* Note: old syntax...
  NFourierTrigSeries[expr, {x, x0, x1}, n] gives the approximate trigonometric
  series expansion of expr to order n.
*)
(*
  NFourierTrigSeries[expr, t, k] finds the numerical values of the Fourier
  trigonometric coefficients c[0], ... c[k], d[1], ..., d[k] for the periodic
  function of t that is equal to expr for -1/2 <= t <= 1/2, repeating with 
  period 1, and gives 
  c[0] + Sum[c[n] Cos[2 Pi n t] + d[n] Sin[2 Pi n t], {n, 1, k}], the kth
  order Fourier trigonometric series approximation to the function.
  NFourierTrigSeries[expr, t, k, FourierParameters -> {a, b}] gives 
  Abs[b]^((1+a)/2) (c[0] +
  Sum[c[n] Cos[2 Pi b n t] + d[n] Sin[2 Pi b n t], {n, 1, k}]), the kth order
  Fourier trigonometric series approximation to the periodic function of
  t that is equal to expr for -1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), repeating with
  period 1/Abs[b], where the c[n] and d[n] are numerical coefficients. *)
If[Not@ValueQ[NFourierTrigSeries::usage],NFourierTrigSeries::usage =
"NFourierTrigSeries[expr, t, k] uses numerical coefficients to give the kth \
order Fourier trigonometric series approximation to the periodic function of t \
that is equal to expr for -1/2 <= t <= 1/2, and has a period of 1. \
NFourierTrigSeries[expr, t, k, FourierParameters -> {a, b}] uses numerical \
coefficients to give the kth order Fourier trigonometric series approximation \
to the periodic function of t that is equal to expr for \
-1/(2 Abs[b]) <= t <= 1/(2 Abs[b]), and has a period 1/Abs[b]."];

Options[NFourierCosCoefficient] = Options[NFourierSinCoefficient] =
   Options[NFourierTrigSeries] =
	Sort[Join[{FourierParameters -> {1, 1}}, Options[NIntegrate]]]


(************************************************************************)
Begin["`Private`"]
(************************************************************************)


(* ============== NFourierTransform & NInverseFourierTransform ============= *)


NFourierTransform[expr_, t_Symbol, w_, opts___?OptionQ] := 
Module[{ffcases, focases, a, b, optNInt},
   (
    ffcases = Cases[{opts}, FourierFrequencyConstant -> _];
    If[ffcases =!= {}, Message[NFourierTransform::ffobs]];
    focases = Cases[{opts}, FourierOverallConstant -> _];
    If[focases =!= {}, Message[NFourierTransform::foobs]];
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    (* NOTE: the pre-V4 .0 FourierOverallConstant corresponds to
	 Sqrt[Abs[b]/(2 Pi)^(1-a)] *)
    Sqrt[Abs[b]/(2 Pi)^(1-a)] NIntegrate[Evaluate[expr Exp[I b w t]],
		{t, -Infinity, Infinity}, Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts, Options[NFourierTransform]}];
	 If[NumberQ[N[b]],
	    True,
	    Message[NFourierTransform::fparm]; False]
	)
  ] /; NumberQ[N[w]] && FreeQ[N[w], Complex]

NInverseFourierTransform[expr_, w_Symbol, t_, opts___?OptionQ] := 
  Module[{ffcases, focases, a, b, optNInt},
   (
    ffcases = Cases[{opts}, FourierFrequencyConstant -> _];
    If[ffcases =!= {}, NFourierTransform::ffobs];
    focases = Cases[{opts}, FourierOverallConstant -> _];
    If[focases =!= {}, NFourierTransform::foobs];
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    Sqrt[Abs[b]/(2 Pi)^(1+a)] NIntegrate[Evaluate[expr Exp[-I b w t]],
		{w, -Infinity, Infinity}, Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts,
		 Options[NInverseFourierTransform]}];
	 If[NumberQ[N[b]],
            True,
            Message[NInverseFourierTransform::fparm]; False]
        )
  ] /; NumberQ[N[t]] && FreeQ[N[t], Complex]


(* ============== NFourierSinTransform & NInverseFourierSinTransform ======= *)

NFourierSinTransform[expr_, t_Symbol, w_, opts___?OptionQ] := 
Module[{ffcases, focases, a, b, optNInt},
   (
    ffcases = Cases[{opts}, FourierFrequencyConstant -> _];
    If[ffcases =!= {}, NFourierSinTransform::ffobs];
    focases = Cases[{opts}, FourierOverallConstant -> _];
    If[focases =!= {}, NFourierSinTransform::foobs];
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    (* NOTE: the pre-V4 .0 FourierOverallConstant corresponds to
         2 Sqrt[Abs[b]/(2 Pi)^(1-a)] *)
    2 Sqrt[Abs[b]/(2 Pi)^(1-a)] NIntegrate[Evaluate[expr Sin[b w t]],
		{t, 0, Infinity}, Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts, Options[NFourierSinTransform]}];
	 If[NumberQ[N[b]],
            True,
            Message[NFourierSinTransform::fparm]; False]
	)
  ] /; NumberQ[N[w]] && FreeQ[N[w], Complex]

NInverseFourierSinTransform[expr_, w_Symbol, t_, opts___?OptionQ] := 
Module[{ffcases, focases, a, b, optNInt},
   (
    ffcases = Cases[{opts}, FourierFrequencyConstant -> _];
    If[ffcases =!= {}, NFourierTransform::ffobs];
    focases = Cases[{opts}, FourierOverallConstant -> _];
    If[focases =!= {}, NFourierTransform::foobs];
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    2 Sqrt[Abs[b]/(2 Pi)^(1+a)] NIntegrate[Evaluate[expr Sin[b w t]],
		{w, 0, Infinity}, Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts,
         		Options[NInverseFourierSinTransform]}];
	 If[NumberQ[N[b]],
            True,
            Message[NInverseFourierSinTransform::fparm]; False]
        )
  ] /; NumberQ[N[t]] && FreeQ[N[t], Complex]


(* ============== NFourierCosTransform & NInverseFourierCosTransform ======= *)

NFourierCosTransform[expr_, t_Symbol, w_, opts___?OptionQ] := 
Module[{ffcases, focases, a, b, optNInt},
    ffcases = Cases[{opts}, FourierFrequencyConstant -> _];
    If[ffcases =!= {}, NFourierCosTransform::ffobs];
    focases = Cases[{opts}, FourierOverallConstant -> _];
    If[focases =!= {}, NFourierCosTransform::foobs];
    {a, b} = FourierParameters /. Flatten[{opts, Options[NFourierCosTransform]}];
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    2 Sqrt[Abs[b]/(2 Pi)^(1-a)] NIntegrate[Evaluate[expr Cos[b w t]],
		{t, 0, Infinity}, Evaluate[Sequence @@ optNInt]]
  ] /; NumberQ[N[w]] && FreeQ[N[w], Complex]

NInverseFourierCosTransform[expr_, w_Symbol, t_, opts___?OptionQ] := 
Module[{ffcases, focases, a, b, optNInt},
    ffcases = Cases[{opts}, FourierFrequencyConstant -> _];
    If[ffcases =!= {}, NFourierTransform::ffobs];
    focases = Cases[{opts}, FourierOverallConstant -> _];
    If[focases =!= {}, NFourierTransform::foobs];
    {a, b} = FourierParameters /. Flatten[{opts,
	 Options[NInverseFourierCosTransform]}];
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    2 Sqrt[Abs[b]/(2 Pi)^(1+a)] NIntegrate[Evaluate[expr Cos[b w t]],
		{w, 0, Infinity}, Evaluate[Sequence @@ optNInt]]
  ] /; NumberQ[N[t]] && FreeQ[N[t], Complex]


(* =============================== Messages ============================== *)
(* for NFourierTransform, NInverseFourierTransform, NFourierSinTransform,  *)
(* NInverseFourierSinTransform, NFourierCosTransform, and		   *)
(* NInverseFourierCosTransform. *)

NFourierTransform::ffobs = NInverseFourierTransform::ffobs =
NFourierSinTransform::ffobs = NInverseFourierSinTransform::ffobs =
NFourierCosTransform::ffobs = NInverseFourierCosTransform::ffobs =
"FourierFrequencyConstant is an obsolete symbol, superseded by the \
constant b in the option setting FourierParameters -> {a, b}."

NFourierTransform::foobs = NInverseFourierTransform::foobs =
"FourierOverallConstant is is an obsolete symbol, superseded by the expression \
Sqrt[Abs[b] (2 Pi)^(a-1)] given the option setting FourierParameters -> {a, b}."

NFourierSinTransform::foobs = NFourierCosTransform::foobs =
NInverseFourierSinTransform::foobs = NInverseFourierCosTransform::foobs =
"FourierOverallConstant is is an obsolete symbol, superseded by the expression \
2 Sqrt[Abs[b] (2 Pi)^(a-1)] given the option setting \
FourierParameters -> {a, b}."

NFourierTransform::fparm = NInverseFourierTransform::fparm =
NFourierSinTransform::fparm = NInverseFourierSinTransform::fparm =
NFourierCosTransform::fparm = NInverseFourierCosTransform::fparm =
"The option setting FourierParameters -> {a, b} must specify numeric b."



(* =============== numeric discrete-time Fourier Transform ================= *)
(* 			(continuous frequency)				     *)

(* w must be real *)
NDTFourierTransform[expr_, n_Symbol, w_, opts___?OptionQ] := 
Module[{a, b, optNSum},
   (
    optNSum = FilterRules[{opts}, Options[NSum]];
    Abs[b/(2Pi)]^((1-a)/2) NSum[Evaluate[expr Exp[-I b n w]],
			 {n, -Infinity, Infinity}, Evaluate[optNSum]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts, Options[NDTFourierTransform]}];
	 If[NumberQ[N[b]], 
	    True,
	    Message[NDTFourierTransform::fparm]; False]
	)
  ] /; NumberQ[N[w]] && FreeQ[N[w], Complex]

(* n must be integer-valued *)
NInverseDTFourierTransform[expr_, w_Symbol, n_?IntegerQ, opts___?OptionQ] := 
Module[{a, b, optNInt},
   (
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    Abs[b/(2Pi)]^((1+a)/2) NIntegrate[Evaluate[expr Exp[I b n w]],
		 Evaluate[{w, -Pi/(Abs[b]), Pi/(Abs[b])}], Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts,
                 Options[NInverseDTFourierTransform]}];
	 If[NumberQ[N[b]], 
            True,
            Message[NInverseDTFourierTransform::fparm]; False]
	)
  ]

NDTFourierTransform::fparm = NInverseDTFourierTransform::fparm =
"The option setting FourierParameters -> {a, b} must specify numeric b."


(* ================ (discrete-frequency) Fourier Coefficient ============== *)
(* 			(continuous time)				    *)

(* t may be symbolic or real *)
InverseFourierCoefficient[expr_, n_Symbol, t_, opts___?OptionQ] := 
Module[{a, b},
    {a, b} = FourierParameters /. Flatten[{opts, Options[InverseFourierCoefficient]}];
    (* NOTE:
	When Sum takes options, allow them to be passed from
	InverseFourierCoefficient to Sum. *)
    (* NOTE:
	When Sum takes Assumptions, add Assumptions -> {Element[w, Reals]}. *)
    Abs[b]^((1+a)/2) Sum[Evaluate[expr Exp[-2 Pi I b n t]],
						 {n, -Infinity, Infinity}]
  ]


(* ============ numeric (discrete-frequency) Fourier Coefficient ========== *)
(* 			(continuous time)				    *)

(* n must be integer-valued *)
NFourierCoefficient[expr_, t_Symbol, n_?IntegerQ, opts___?OptionQ] := 
Module[{a, b, optNInt},
   (
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    Abs[b/(2Pi)]^((a+1)/2) NIntegrate[Evaluate[expr Exp[-I b n t]],
		Evaluate[{t, -Pi/(Abs[b]), Pi/(Abs[b])}], Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts,
                         Options[NFourierCoefficient]}];
         If[NumberQ[N[b]],
            True,
            Message[NFourierCoefficient::fparm]; False]
        )
  ]

(* t must be real *)
NInverseFourierCoefficient[expr_, n_Symbol, t_, opts___?OptionQ] := 
Module[{a, b, optNSum},
   (
    optNSum = FilterRules[{opts}, Options[NSum]];
    Abs[b]^((1+a)/2) NSum[Evaluate[expr Exp[-2 Pi I b n t]],
			 {n, -Infinity, Infinity}, Evaluate[optNSum]]
   ) /; ({a, b} = FourierParameters /. Flatten[{opts,
			 Options[NInverseFourierCoefficient]}];
	 If[NumberQ[N[b]],
	    True,
            Message[NInverseFourierCoefficient::fparm]; False]
	)
  ] /; NumberQ[N[t]] && FreeQ[N[t], Complex]

NFourierCoefficient::fparm = NInverseFourierCoefficient::fparm =
"The option setting FourierParameters -> {a, b} must specify numeric b."

(* ====================== numeric Fourier series ===================== *)
(* 	based on kth-order approximation using NFourierCoefficient     *)

NFourierSeries[expr_, t_, k_?IntegerQ, opts___?OptionQ] :=
Module[{a, b, table, optNFourC},
   (
     optNFourC = FilterRules[{opts}, Options[NFourierCoefficient]];
     table = Table[Exp[I b n t] *
		NFourierCoefficient[expr, t, n, optNFourC],
							{n, -k, k}];
     (Abs[b]/(2*Pi))^((1-a)/2) Apply[Plus, table]
   ) /; (
         {a, b} = FourierParameters /. Flatten[{opts, Options[NFourierSeries]}];
	 If[NumberQ[N[b]],
            True,
            Message[NFourierSeries::fparm]; False]
        )
  ] /; k >= 0

NFourierSeries::fparm =
"The option setting FourierParameters -> {a, b} must specify numeric b."


(* ========== numeric (discrete-frequency) Fourier Cos Coefficient ======== *)
(* 			(continuous time)				    *)

(* n must be nonnegative integer-valued (i.e., n = 0, 1, 2, ...) *)
NFourierCosCoefficient[expr_, t_Symbol, n_?IntegerQ, opts___?OptionQ] := 
Module[{a, b, optNInt},
   (
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    Abs[2*b/Pi]^((a+1)/2) NIntegrate[Evaluate[expr Cos[b n t]],
		Evaluate[{t, 0, Pi/(Abs[b])}], Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /.
		 Flatten[{opts, Options[NFourierCosCoefficient]}];
	 If[NumberQ[N[b]],
            True,
            Message[NFourierCosCoefficient::fparm]; False]
	)
  ] /; n >= 0

NFourierCosCoefficient::fparm =
"The setting FourierParameters -> {a, b} must specify numeric b."




(* ========== numeric (discrete-frequency) Fourier Sin Coefficient ======== *)
(* 			(continuous time)				    *)

(* n must be positive integer-valued (i.e., n = 1, 2, ...) *)
NFourierSinCoefficient[expr_, t_Symbol, n_?IntegerQ, opts___?OptionQ] := 
Module[{a, b, optNInt},
   (
    optNInt = FilterRules[{opts}, Options[NIntegrate]];
    Abs[2 b/Pi]^((a+1)/2) NIntegrate[Evaluate[expr Sin[b*n*t]],
		Evaluate[{t, 0, Pi/(Abs[b])}], Evaluate[Sequence @@ optNInt]]
   ) /; ({a, b} = FourierParameters /.
		 Flatten[{opts, Options[NFourierSinCoefficient]}];
	 If[NumberQ[N[b]],
            True,
            Message[NFourierSinCoefficient::fparm]; False]
	)
  ] /; n >= 1

NFourierSinCoefficient::fparm =
"The setting FourierParameters -> {a, b} must specify numeric b."



NFourierTrigSeries[expr_, t_Symbol, k_?IntegerQ, opts___?OptionQ] :=
Module[{a, b, table, optNFourC},
   (
    optNFourC = FilterRules[{opts}, Options[NFourierCosCoefficient]];
    table = Table[Cos[b n t]NFourierCosCoefficient[Together[expr + (expr/.{t-> -t})], t, n, optNFourC]/2 +
		  Sin[b n t]NFourierSinCoefficient[Together[expr - (expr/.{t-> -t})], t, n, optNFourC]/2,
				{n, 1, k}];
    Abs[b/(2 Pi)]^((1-a)/2) (
       NFourierCosCoefficient[Together[expr + (expr/.{t-> -t})], t, 0, optNFourC]/4 + Apply[Plus, table]
		        )
   ) /; ({a, b} = FourierParameters /. Flatten[{opts, Options[NFourierTrigSeries]}];
	 If[NumberQ[N[b]],
            True,
            Message[NFourierTrigSeries::fparm]; False]
        )
  ] /; k >= 0

NFourierTrigSeries::fparm =
"The setting FourierParameters -> {a, b} must specify numeric b."


(* =================== discrete-time Fourier Transform ===================== *)
(* 			(continuous frequency)				     *)

(* w may be symbolic or real *)
DTFourierTransform[expr_, n_Symbol, w_, opts___?OptionQ] := 
	Module[{res},
           Message[DTFourierTransform::obsfun,DTFourierTransform,FourierSequenceTransform];Off[DTFourierTransform::obsfun];
	   res = System`FourierSequenceTransform[expr,n,w,opts];
           res /; FreeQ[res, FourierSequenceTransform]
	]

(* Note:  separate rules for InverseDTFourierTransform for n symbolic and
        n integer-valued because Assumptions -> {True} is annoying if
        it is not needed.
*)

(* n symbolic *)
InverseDTFourierTransform[expr_, w_Symbol, n_, opts___?OptionQ] := 
	Module[{res},
           Message[InverseDTFourierTransform::obsfun,InverseDTFourierTransform,InverseFourierSequenceTransform];Off[InverseDTFourierTransform::obsfun];
	   res = System`InverseFourierSequenceTransform[expr,w,n,opts];
           res /; FreeQ[res, InverseFourierSequenceTransform]
	]



(************************************************************************)
End[]             (* end `Private` Context                              *)
(************************************************************************)
 
 
 
(************************************************************************)
EndPackage[]      (* end package Context                                *)
(************************************************************************)




