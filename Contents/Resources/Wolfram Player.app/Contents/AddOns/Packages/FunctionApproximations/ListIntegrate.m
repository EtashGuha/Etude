(* ::Package:: *)

(* :Name: FunctionApproximations`ListIntegrate` *)

(* :Title:  Approximate Integration of Functions That Are Known Only at a Few
		Distinct Points *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This is an obsolete package for giving an approximate integral
from a list of discrete points.
*)

(* :Context: FunctionApproximations` *)

(* :Mathematica Version: 6.0 *)

(* :Package Version: 2.0 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.
*)

(* :History:
	Original version by Jerry B. Keiper, March 1989.
	Revised by Jerry B. Keiper, November 1990.
	Revised by Robert J. Knapp, January, 1995.
    Obsoleted by John M. Novak, October, 2005.
*)

Unprotect[ListIntegrate];

ListIntegrate::obslt =
"ListIntegrate is an obsolete package, superseded by the kernel \
Integrate functionality \
Integrate[Interpolation[data, InterpolationOrder -> n][var], {var, min, max}].";

Message[ListIntegrate::obslt];

If[Not@ValueQ[ListIntegrate::usage],ListIntegrate::usage =
"ListIntegrate[{y0, y1, ..., yn}, h, k] is an obsolete function that \
uses an InterpolatingFunction object of order k to give an approximation \
to the integral of a function with values equal to  y0,...,yn at points \
equally spaced a distance h apart."];

Begin["`Private`"]

ListIntegrate[cl_?VectorQ, h_, k_Integer:3] :=
	Module[{ans},
		ans /; (ans = ListIntegrate0[h, cl, k]) =!= $Failed
	] /; k > 0;

ListIntegrate[cl_?MatrixQ, k_Integer:3] :=
	Module[{ans, dim = Length[Dimensions[cl]]},
		ans /; ((dim == 2) &&
			(Length[cl[[1]]] == 2) &&
			((ans = ListIntegrate1[cl, k]) =!= $Failed))
	] /; k > 0;


ListIntegrate0[h_, cl_, k_] := 
Module[{ifun,order = If[EvenQ[k],k-1,k],x},
	ifun = Interpolation[cl,InterpolationOrder->order];
	If[Head[ifun] === Interpolation,Return[$Failed]];
	h*Integrate[ifun[x],{x,1,Length[cl]}]]

ListIntegrate1[cl_, k_] := 
Module[{a = cl[[1,1]],b = (Last[cl])[[1]],order = If[EvenQ[k],k-1,k],ifun,x},
	ifun = Interpolation[cl,InterpolationOrder->order];
	If[Head[ifun] === Interpolation,Return[$Failed]];
	Integrate[ifun[x],{x,a,b}]]

End[] (* `Private` *)

Protect[ListIntegrate];
