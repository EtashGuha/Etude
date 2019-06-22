
(* :Name: NIntegrateUtilities` *)

(* :Title: Utility functions for NIntegrate. *)

(* :Author: Anton Antonov *)

(* :Summary:
 This package provides a number of supporting functions for NIntegrate.
*)

(* :Context: DifferentialEquations`NIntegrateUtilities` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 2007, Wolfram Research, Inc. *)

(* :History:
 Version 1.0 by Anton Atnonov, March 2007.
*)

(* :Keywords:
 NIntegrate, integration, evaluations, profiling.
*)

(* :Source:
*)

(* :Mathematica Version: 6.0 *)

(* :Limitation:
*)

(* :Discussion:
*)

BeginPackage["Integration`NIntegrateUtilities`"];

NIntegrateProfile::usage = "NIntegrateProfile[NIntegrate[___], ntimes_?IntegerQ:10] gives the numerical integration result, the number of integrand evaluations, and time to compute the the integral. The integration is done ntimes."

NIntegrateProfileWithExact::usage = "NIntegrateProfile[NIntegrate[___], ntimes_?IntegerQ:10] gives the numerical integration result, the relative error compared with the exact integral, the number of integrand evaluations, and time to compute the the integral. The integration is done ntimes."

NIntegrateSamplingPoints::usage = "NIntegrateSamplingPoints[NIntegrate[___]] plots the sampling points for one-, two-, and three-dimensional integrals."

NIntegrateRangesToHyperCube::usage = "NIntegrateRangesToHyperCube[ranges, cubeSides] gives the transformation and its Jacobian of ranges to a hypercube with sides specified by cubeSides."


Unprotect[
NIntegrateProfile, NIntegrateSamplingPoints, NIntegrateProfileWithExact, NIntegrateRangesToHyperCube
];

Begin["`Private`"];

Clear[NIntegrateProfile]
SetAttributes[NIntegrateProfile, HoldFirst];
NIntegrateProfile[expr_, n_:10] :=
 Module[{k, res, t},
  k = 0;
  res = Hold[expr] /. {HoldPattern[NIntegrate[s___]] :> 
      NIntegrate[s, EvaluationMonitor :> k++], 
     HoldPattern[OldNIntegrate[s___]] :> 
      OldNIntegrate[s, EvaluationMonitor :> k++]};
  res = ReleaseHold[res];
  t = Hold[Timing[Do[expr, {i, 1, n}]]];
  t = ReleaseHold[t];
  {"IntegralEstimate"->InputForm[res],"Evaluations"->k, "Timing"->t[[1]]/n // N}
  ]


Clear[NIntegrateProfileWithExact]
SetAttributes[NIntegrateProfileWithExact, HoldFirst];
NIntegrateProfileWithExact[expr_, n_:10] :=
 Module[{k, res, t, exact},
  k = 0;
  exact = Hold[expr] /.  {HoldPattern[NIntegrate[s___]] :> Integrate[s], HoldPattern[OldNIntegrate[s___]] :> Integrate[s]};
  exact = DeleteCases[exact,HoldPattern[Rule[s___]], Infinity];
  exact = ReleaseHold[exact];
  res = Hold[expr] /. {HoldPattern[NIntegrate[s___]] :> 
      NIntegrate[s, EvaluationMonitor :> k++], 
     HoldPattern[OldNIntegrate[s___]] :> 
      OldNIntegrate[s, EvaluationMonitor :> k++]};
  res = ReleaseHold[res];
  t = Hold[Timing[Do[expr, {i, 1, n}]]];
  t = ReleaseHold[t];
  {"IntegralEstimate"->InputForm[res], "RelativeError"->Abs[res-exact]/Abs[exact], "Evaluations"->k, "Timing"->t[[1]]/n // N}
  ]



Clear[NIntegrateSamplingPoints]
SetAttributes[NIntegrateSamplingPoints, HoldFirst];
NIntegrateSamplingPoints[expr_, n_:10] :=
  Module[{vars, ranges, res, t},
   ranges = Cases[Hold[expr], _List, {2}];
   vars = First /@ ranges;
   res = Hold[
      expr] /. {HoldPattern[NIntegrate[s___]] :> 
       NIntegrate[s, EvaluationMonitor :> Sow[vars]], 
      HoldPattern[OldNIntegrate[s___]] :> 
       OldNIntegrate[s, EvaluationMonitor :> Sow[vars]]};
   res = Reap[res[[1]]];
   t = res[[2, 1]];
   Which[
    Length[ranges] == 1,
    t = Flatten[t];
    Graphics[Point /@ Transpose[{N[t], Range[Length[t]]}], Axes -> True, 
     AspectRatio -> 1.5, PlotRange -> All],
    Length[ranges] == 2,
    Graphics[Point /@ t, Axes -> True, AspectRatio -> Automatic, 
     PlotRange -> All],
    Length[ranges] == 3,
    Graphics3D[Point /@ t, Axes -> True, PlotRange -> All]
   ]
  ];

Clear[NIntegrateRangesToHyperCube]; 
NIntegrateRangesToHyperCube[ranges_, cubeSides:{{_, _}...}] := 
Module[{t, t1, jac, vars, rules={}},
   vars = First /@ ranges; 
   t = MapThread[(t1=Rescale[#1[[1]], #2, {#1[[2]], #1[[3]]}/.rules];AppendTo[rules,#1[[1]]->t1];t1)&, {ranges, cubeSides}]; 
   jac = Times @@ MapThread[D[#1, #2] & , {t, vars}]; 
   {rules, jac}
]/;Length[ranges]==Length[cubeSides]

NIntegrateRangesToHyperCube[ranges_, cubeSide:{_, _}] := 
 NIntegrateRangesToHyperCube[ranges, Table[cubeSide,{Length[ranges]}]]

NIntegrateRangesToHyperCube[ranges_] := NIntegrateRangesToHyperCube[ranges, {0,1}]





End[ ]; (* End `Private` Context. *)

SetAttributes[
{ NIntegrateProfile, NIntegrateSamplingPoints, NIntegrateProfileWithExact, NIntegrateRangesToHyperCube},
{ Protected, ReadProtected }
];

EndPackage[ ]; (* End package Context. *)

