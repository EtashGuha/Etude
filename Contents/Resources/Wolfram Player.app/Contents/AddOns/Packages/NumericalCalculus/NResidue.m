(* ::Package:: *)

(* :Name: NumericalMath`NResidue` *)

(* :Title: Numerical Evaluation of Residues *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package uses NIntegrate to evaluate the residue of
an expression near a point in the complex plane.
*)

(* :Context: NumericalMath`NResidue` *)

(* :Package Version: 2.0 *)

(* :Copyright: Copyright 1994-2007, Wolfram Research, Inc.
*)

(* :History:
	Originally written by Jerry B. Keiper, March 1994.
*)

(* :Keywords: Residue, residue, complex analysis *)

(* :Source:
*)

(* :Warnings: None. *)

(* :Mathematica Version: 2.0 *)

(* :Limitations:
*)




If[Not@ValueQ[NResidue::usage],NResidue::usage =
"NResidue[expr, {x, x0}] uses NIntegrate to numerically find the residue \
of expr near the point x = x0."];

If[Not@ValueQ[Radius::usage],Radius::usage =
"Radius is an option to NResidue that specifies the radius of the \
circle on which the integral is evaluated."];

Begin["`Private`"]

Options[NResidue] = {Radius -> 1/100,
		     WorkingPrecision -> MachinePrecision,
		     PrecisionGoal -> Automatic};

NResidue[f_, {x_, x0_}, opts___] :=
    Module[{ans},
	ans /; (ans = nres[f, x, x0, opts]) =!= $Failed
	];

NResidue::rad = "Radius `1` is not a positive number."



nres[f_, x_, x0_, opts___] :=
    Module[{ft, r, t, z},
	r = Radius /. {opts} /. Options[NResidue];
	If[!NumericQ[r] || !TrueQ[r > 0],
		Message[NResidue::rad, r];
		Return[$Failed]
		];
	ft = f /. x -> (x0 + z);
	If[Length[{opts}]===0,NIntegrate[Evaluate[z = r E^(I t); z ft], {t, 0, 2Pi}, Method -> Trapezoidal]/(2 Pi),NIntegrate[Evaluate[z = r E^(I t); z ft], {t, 0, 2Pi}, Method -> Trapezoidal,
		Evaluate[Sequence@@FilterRules[{opts}, Options[NIntegrate]]]]/(2 Pi)]
	];

End[]  (* NumericalMath`NResidue`Private` *)

(* NumericalMath`NResidue` *)

(* :Tests:
NResidue[1/x, {x, 0}]
NResidue[1/Sin[x], {x, Pi}]
NResidue[Zeta[z], {z, 1}]
NResidue[1/Expand[((z-1.7)(z+.2+.5 I)(z+.2-.5 I))],{z,1.7}]
*)

