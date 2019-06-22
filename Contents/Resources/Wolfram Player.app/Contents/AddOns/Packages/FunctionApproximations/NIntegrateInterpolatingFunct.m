(* ::Package:: *)

(* :Name: FunctionApproximations`NIntegrateInterpolatingFunct` *)

(* :Title:  NIntegration of Interpolating Functions *)

(* :Author: David Jacobson, Hewlett-Packard Laboratories *)

(* :Summary:
This package introduces the function NIntegrateInterpolatingFunction,
which gives a numerical approximation to an integral over a specified
interval.  NIntegrateInterpolatingFunction may be used in place of
NIntegrate to numerically integrate expressions containing
InterpolatingFunction objects.
*)

(* :Discussion:
        The algorithms for NIntegrate assume that the integrand is
        smooth to at least several orders.  InterpolatingFunction objects
        typically do not satisfy this assumption (they are however
        piecewise smooth) and the algorithms used by NIntegrate
        converge very slowly when applied to InterpolatingFunctions,
        especially in several dimensions.  NIntegrate allows the
        domain of integration to be broken up into several pieces and
        the integral evaluated over each piece.  If the pieces of the
        domain correspond to the pieces over which the
        InterpolatingFunction is smooth, NIntegrate will converge
        much more rapidly.  This package defines
        NIntegrateInterpolatingFunction, which automatically breaks
        up the domain of integration.
*)

(* :Context: FunctionApproximations` *)

(* :Mathematica Version: 3.0 *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1993-2007, Wolfram Research, Inc.  *)

(* :History:
        Original version by David Jacobson, Hewlett-Packard Laboratories,
        July, 1993.
        Revised by Jerry B. Keiper, Wolfram Research, Inc., July, 1993.
	Revised by Robert Knapp, Wolfram Research, Inc., Nov., 1994..
	Obsoleted in 2009 since NIntegrate now natively supports InterpolatingFunction objects.
*)

(* :Keywords: integration, numerical integration, interpolation *)

(* :Source:
        Robert D. Skeel and Jerry B. Keiper, Elementary Numerical
                Computing with Mathematica, McGraw-Hill, 1993.
*)

(* :Warnings:
*)

(* :Limitations:
*)

(* Usage messages *)

Unprotect[NIntegrateInterpolatingFunction];

(*If[Not@ValueQ[NIntegrateInterpolatingFunction::usage],NIntegrateInterpolatingFunction::usage =
"NIntegrateInterpolatingFunction[f, {x, xmin, xmax}] gives a numerical \
approximation to the integral of f with respect to x over the interval \
xmin to xmax. If f involves InterpolatingFunction objects the domain of \
integration is broken up at the interpolation nodes prior to being passed \
on to NIntegrate for evaluation."];*)

If[Not@ValueQ[NIntegrateInterpolatingFunction::usage],NIntegrateInterpolatingFunction::usage =
	"NIntegrateInterpolatingFunction is obsolete. NIntegrate now natively supports InterpolatingFunction objects. "];

Begin["`Private`"]

Clear[RuleOrRuleDelayedQ];
RuleOrRuleDelayedQ[x___] :=    
    And @@ ((Head[#1] === Rule || Head[#1] === RuleDelayed) & /@ {x});

SetAttributes[NIntegrateInterpolatingFunction, HoldAll]

NIntegrateInterpolatingFunction[f_, r__List, opts___] :=
    Module[{ifs, p},
        ifs = Union[Cases[{f}, InterpolatingFunction[__][__], Infinity]];
        p = WorkingPrecision /. {opts} /. Options[NIntegrate];
        NIntegrate[f, Evaluate[Sequence @@ (ein0[#, ifs, p]& /@ {r})], opts]
        ]/;RuleOrRuleDelayedQ[opts];

ein0[{x_, x0_, xi___, xn_}, fl_List, p_] := (* extract interpolating nodes *)
    Module[{r, rev, nx0 = N[x0, p], nxn = N[xn, p]},
        rev = (nx0 > nxn);
        r = If[rev, (nxn < # < nx0)&, (nx0 < # < nxn)&];
        r = Select[Union[N[{xi}, 4p], Flatten[ein1[x, #, p]& /@ fl],
                SameTest -> Equal], r];
        If[rev, r = Reverse[r]];
        {x, x0, Sequence @@ r, xn}
        ];

ein1[x_, fx_, p_] :=            (* find the x values giving nodes of f *)
    Module[{nl, dim, ifun},
        If[(dim = dimif[fx]) === $Failed, Return[{}]];
	ifun = Head[fx];
	nl = ifun["Coordinates"[]];
        Flatten[MapThread[invg[x, #1, #2]&, {List @@ fx, N[nl, p]}]]
        ];

invg[x_, g_, nl_List] :=
    Module[{y, xl},
        If[FreeQ[g, x], Return[{}]];
        If[x === g, Return[nl]];
        xl = Solve[g == y, x];
        If[!MatchQ[xl, {{_ -> _}..}], Return[{}]];
        Select[Flatten[(x /. xl) /. y -> nl], NumberQ]
        ];

dimif[fx_] :=
    With[{dim = Length[fx],ifun = Head[fx]},
    	If[Length[ifun["Coordinates"[]]] == dim,
    	    dim,
    	    $Failed]
    ];

SyntaxInformation[NIntegrateInterpolatingFunction] = {"ArgumentsPattern"->{_,_,Optional[__],OptionsPattern[]},"LocalVariables"->{"Integrate",{2,\[Infinity]}}};


End[] (* `Private` *)

Protect[NIntegrateInterpolatingFunction];
