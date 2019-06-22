(* :Name: NDSolveUtilities` *)

(* :Title: Utility functions for NDSolve. *)

(* :Author: Mark Sofroniou *)

(* :Summary:
 This package provides a number of supporting functions for NDSolve.
*)

(* :Context: DifferentialEquations`NDSolveUtilities` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 2002, Wolfram Research, Inc. *)

(* :History:
 Version 1.0 by Mark Sofroniou, June 2002.
*)

(* :Keywords:
 NDSolve, Invariants, Interpolation, InterpolatingFunction,
 ordinary differential equations, partial differential equations.
*)

(* :Source:
*)

(* :Mathematica Version: 5.0 *)

(* :Limitation:
*)

(* :Discussion:
*)

BeginPackage["DifferentialEquations`NDSolveUtilities`",
"DifferentialEquations`NDSolveProblems`"];

CompareMethods::usage = "CompareMethods[eqs, refsol, methods, opts]
Takes the equations eqs specified as an NDSolveProblem and compares
various methods with the reference solution refsol. NDSolve options
can be specified as opts."

FinalSolutions::usage =
"FinalSolutions[eqs, sols] gives the end point solutions sols for the equations
eqs specified as an NDSolveProblem."

InvariantErrorPlot::usage =
"InvariantErrorPlot[invariants, dvars, ivar, sol, opts]
plots the errors in invariants with dependent variables dvars and independent
variable ivar at the grid points of the NDSolve solution sol."

InvariantDimensions::usage = "InvariantErrorPlot option specifying the dimensions of the invariants."

InvariantErrorFunction::usage = "InvariantErrorPlot option specifying how errors should be computed."

InvariantErrorSampleRate::usage = "InvariantErrorPlot option specifying how often points from InterpolatingFunctions
are sampled."

RungeKuttaLinearStabilityFunction::usage =
"RungeKuttaLinearStabilityFunction[amat, bvec, z] gives the linear
stability function of the Runge Kutta method with coefficient matrix
amat and weight vector bvec in the variable z."

StepDataPlot::usage = "StepDataPlot[sols, opts] plots the step sizes used in
the NDSolve solutions sols."

StepDataCount::usage = "StepDataCount[sols] gives the number of accepted steps taken in
the NDSolve solutions sols."

Unprotect[
CompareMethods, FinalSolutions, InvariantDimensions, InvariantErrorPlot,
InvariantErrorSampleRate, InvariantErrorFunction,
RungeKuttaLinearStabilityFunction, StepDataPlot
];

Begin["`Private`"];

(*
 Error metric. This function finds the minimum of the maximum absolute and
 relative errors of a solution when compared with a reference solution.
*)

MaxError[approxsol_, refsol_] :=
  Min[Max[Abs[#]], Max[Abs[Divide[#, refsol]]]]& @ Subtract[refsol, approxsol];

(* Utility function for comparing various methods. *)

CompareMethods[system_NDSolveProblem, refsol_List, methods_List, opts___?OptionQ]:=
  Module[{cost, endsol, error, sol, stepa, stepr, stepm, tdata, t, t0, y},
    tdata = system["TimeData"];
    t = Part[tdata, 1];
    t0 = Part[tdata, 2];
    stepm[tnew_, _] := If[tnew - t0 == 0, stepr++, t0 = tnew; stepa++];
    Table[
      cost = 0;
      stepa = 0;
      stepr = 0;
      sol = 
        NDSolve[system,
          Method -> Part[methods, i],
          EvaluationMonitor :> (cost++),
          StepMonitor :> (stepm[t, y]),
          MaxSteps -> Infinity,
          opts
          ];
      endsol = First[FinalSolutions[system, sol]];
      error = MaxError[endsol, refsol];
      {{stepa, stepr}, cost, error},
    {i, Length[methods]}]
  ];

(* Function to evaluate the solution at the end of the integration. *)

FinalSolutions[system_NDSolveProblem, sols_List] :=
  system["DependentVariables"] /. sols /. 
    First[system["TimeData"]] -> Last[system["TimeData"]];

(* Data extraction from the interpolating functions *)
GetIFun[x_->y_]:= y;
GetIFun[x_]:= x;

GetTimeData[sol:{__}] := GetTimeData[GetIFun[Part[sol, 1]]];
GetTimeData[(ifun:InterpolatingFunction[__])[__]] := First[ifun["Coordinates"]];
GetTimeData[(ifun:InterpolatingFunction[__])] := First[ifun["Coordinates"]];
GetTimeData[___]:= Throw[$Failed];

GetGridData[sol:{__}] := Map[GetGridData[GetIFun[#]]&, sol];
GetGridData[(ifun:InterpolatingFunction[__])[__]] := ifun["ValuesOnGrid"];
GetGridData[(ifun:InterpolatingFunction[__])] := ifun["ValuesOnGrid"];
GetGridData[___]:= Throw[$Failed];

ValidInvariantsQ[{}]:= False;
ValidInvariantsQ[_?VectorQ]:= True;
ValidInvariantsQ[x_]:= !ListQ[x];

ValidNDSolveSolutionQ[{}] = False;
ValidNDSolveSolutionQ[InterpolatingFunction[__][__]]:= True;
ValidNDSolveSolutionQ[InterpolatingFunction[__]]:= True;
ValidNDSolveSolutionQ[_?VectorQ]:= True;
ValidNDSolveSolutionQ[_?MatrixQ]:= True;
ValidNDSolveSolutionQ[_]:= False;

ValidVariableQ[x_]:= !ListQ[x];
ValidVariableVectorQ[x_]:= VectorQ[x] || ValidVariableQ[x];

SetAttributes[AddTimeDependency, Listable];
AddTimeDependency[x_[t_], t_]:= x[t];
AddTimeDependency[x_, t_]:= x[t];

(* Decide how to treat multiple NDSolve solutions *)

ApplyFunction[fun_, data_?VectorQ]:=
  fun[data];

ApplyFunction[fun_, data_?MatrixQ]:=
  If[SameQ[Length[data], 1], 
    fun[First[data]],
    Map[fun, data]
  ];

ApplyFunction[fun_, data_]:=
  fun[{data}];

PMIntegerQ[n_]:= Developer`MachineIntegerQ[n] && Positive[n];

(**** StepDataCount ****)

StepDataCount[sols_?ValidNDSolveSolutionQ]:=
  Module[{res},
    res = Length[GetTimeData[sols]];
    res /; res =!= $Failed
  ];

(**** StepDataPlot ****)

StepDataPlot[sols_?ValidNDSolveSolutionQ, opts___?OptionQ]:=
  Module[{res, sdpfun, lplotopts},
    lplotopts = Flatten[{System`Utilities`FilterOptions[ListLogPlot, opts]}];
    sdpfun = sdplot[#, lplotopts]&;
    res = Catch[ ApplyFunction[sdpfun, sols] ];
    res /; res =!= $Failed
  ];

sdplot[sol_?VectorQ, {lplotopts___}] :=
  Module[{stepvals, tdata},
    tdata = GetTimeData[sol];
    stepvals = Transpose[{tdata, Join[{0}, Differences[tdata]]}];
    ListLogPlot[
      stepvals, lplotopts, Axes->False, Frame -> True,
      Joined->True, Mesh->All, PlotRange->All, RotateLabel -> False
    ]
  ];

sdplot[___]:= Throw[$Failed];

(**** InvariantErrorPlot ****)

Options[InvariantErrorPlot] = {
InvariantErrorFunction :> (Abs[Subtract[#1, #2]]&),
InvariantDimensions -> Automatic,
InvariantErrorSampleRate -> Automatic
};

(*
 Check the structure of the inputs.

 Check the values of the options.

 Check the dimensions of the interpolating functions.
 *)

InvariantErrorPlot[invariants_?ValidInvariantsQ, dvars_?ValidVariableVectorQ,
    ivar_?ValidVariableQ, sols_?ValidNDSolveSolutionQ, opts___?OptionQ] :=
  Module[{divars, iepfun, ieplotopts, ieplotuseropts, lplotopts, res},
    ieplotopts = Options[InvariantErrorPlot];
    ieplotuseropts = Flatten[{System`Utilities`FilterOptions[InvariantErrorPlot, opts]}];
    ieplotopts = Map[First, ieplotopts] /. ieplotuseropts /. ieplotopts;
    lplotopts = Flatten[{System`Utilities`FilterOptions[ListPlot, opts]}];
    divars = AddTimeDependency[Flatten[{dvars}], ivar];
    iepfun = ieplot[#, invariants, divars, ivar, ieplotopts, lplotopts]&;
    res = Catch[ ApplyFunction[iepfun, sols] ];
    res /; res =!= $Failed
  ];

ieplot[sol_?VectorQ, invariants_, dvars_, ivar_, {errfun_, errdims_, srate_}, {lplotopts___}]:=
  Module[{cfun, data, dims, errors, ierror, invts, len, prec, samples, samplerate},
    samples = GetTimeData[sol];
    data = Transpose[GetGridData[sol]];
    len = Length[samples];

    (* Take an error sample for long integrations *)
    If[SameQ[srate, Automatic],
      samplerate = If[len <= 1000, 1, Max[1, Ceiling[Log[N[len]]]]],
      If[!PMIntegerQ[srate] || (srate > len),
        (* Message *)
        Throw[$Failed];
      ];
      samplerate = srate;
    ];

    If[UnsameQ[samplerate, 1],
      samples = Take[samples, {1, len, samplerate}];
      data = Take[data, {1, len, samplerate}];
    ];
    prec = Precision[data];
    If[SameQ[errdims, Automatic],
      If[ListQ[invariants],
        invts = invariants;
        dims = Dimensions[invts];,
        invts = {invariants};
        dims = {1};
      ],
      If[SameQ[errdims, {}],
        dims = {1};
        invts = {invariants};,
        dims = errdims;
        invts = invariants;
      ]
    ];

    cfun =
      Experimental`CreateNumericalFunction[
        Join[{ivar}, dvars],
        invts,
        dims,
        {_Real, Length[dvars]}, WorkingPrecision -> prec
      ];

    (* Check for a valid NumericalFunction *)
    If[!NDSolve`ValidNumericalFunctionQ[cfun],
      (* Message *)
      Throw[$Failed];
    ];

    (* Initial values of invariants *)
    ierror = cfun[Part[samples, 1], Part[data, 1]];
    If[SameQ[Head[ierror], Experimental`NumericalFunction],
      (* Message *)
      Throw[$Failed];
    ];

    (* Compute the invariant errors *)
    errors =
      Transpose[
        MapThread[
          (errfun[ierror, cfun[#1, #2]])&,
          {samples, data}
        ]
      ];
    data = Map[Transpose[{samples, #}]&, errors];

    ListPlot[
      data, lplotopts, AspectRatio -> 1, Axes->False,
      Frame -> True, Joined->False, Mesh->All,
      PlotRange->All, RotateLabel -> False
    ]
  ];

ieplot[___]:= Throw[$Failed];

(**** Linear stability functions for Runge-Kutta methods ****)

(* Rows must have unit stride in length *)

OneVectorQ[{(1) ..}] := True;
OneVectorQ[__] := False;

ValidRowLengthsQ[{1}] = True;
ValidRowLengthsQ[lengths_?VectorQ] := 
    OneVectorQ[Subtract[Drop[lengths, 1], Drop[lengths, -1]]];
ValidRowLengthsQ[__] = False;

(* Implicit method *)

RKToMatrix[a_?MatrixQ, s_]:=
  If[UnsameQ[Dimensions[a], {s, s}],
    If[SameQ[s, 2] && SameQ[Length[Last[a]], 1],
      (* Special limiting case for a two stage explicit method *)
      ToERKMatrix[a, s],
      Throw[$Failed]
    ],
    a
  ];

(*
 Structural check for the coefficients of a diagonally implicit and an
 explicit Runge Kutta method.
 *)

ValidTriangularMatrixQ[{}] = False;
ValidTriangularMatrixQ[amat : {__?VectorQ}] :=
  ValidRowLengthsQ[Map[Length, amat]];
ValidTriangularMatrixQ[_] := False;

(* Remaining methods *)

(* Convert triangular specification to a full matrix *)

ToDIRKMatrix[a_, s_] := 
  Table[If[i < j, 0, Part[a, i, j]], {i, s}, {j, s}];

(* Convert lower triangular specification to a full matrix *)

ToERKMatrix[a_, s_] := 
  Table[If[i <= j, 0, Part[a, i - 1, j]], {i, s}, {j, s}];

RKToMatrix[a_?ValidTriangularMatrixQ, s_]:=
  Switch[Length[Last[a]],
    s, ToDIRKMatrix[a, s],
    s - 1, ToERKMatrix[a, s],
    _, Throw[$Failed]
  ];

RKToMatrix[___]:= Throw[$Failed];

RKClassify[a_]:=
  Catch[
    Module[{n, m},
      {n, m} = Dimensions[a];
      Do[If[!(TrueQ[Part[a, i, j] == 0]), Throw["Implicit", "RKClassify"]], {i, n}, {j, i + 1, m}];
      Do[If[!(TrueQ[Part[a, i, i] == 0]), Throw["DiagonallyImplicit", "RKClassify"]], {i, n}];
      "Explicit"
    ],
    "RKClassify"
  ];

(*
 Use the equivalent of ButcherPhi in tensor form for straight trees.
*)

lsf["Explicit", a_, b_, z_]:=
  Module[{lsfc, e, s},
    s = Length[b];
    e = ConstantArray[1, s];
    lsfc = 
      Map[
        Together[Dot[b, #]]&,
        NestList[Apart[Dot[a, #]]&, e, s - 1]
      ];
    1 + Dot[lsfc, Table[z^i, {i, s}]]
  ];

lsf["DiagonallyImplicit", a_, b_, z_]:=
  Module[{id, num, den, s},
    s = Length[b];
    id = IdentityMatrix[s];
    num = Det[ id - z*(a-Table[b,{s}]) ];
    den = Expand[Tr[id - z*a, Times]];
    num/den
  ];

lsf["Implicit", a_, b_, z_]:=
  Module[{id, num, den, s},
    s = Length[b];
    id = IdentityMatrix[s];
    num = Det[ id - z*(a-Table[b,{s}]) ];
    den = Det[ id - z*a ];
    num/den
  ];

lsf[___]:= Throw[$Failed];

rklsf[ark_, b_, z_]:=
  Catch[
    Module[{a, class, s},
      s = Length[b];
      a = RKToMatrix[ark, s];
      class = RKClassify[a];
      lsf[class, a, b, z]
    ]
  ];

ScalarQ[_List]:= False;
ScalarQ[_]:= True;

RungeKuttaLinearStabilityFunction[a_?ListQ, b_?VectorQ, z_?ScalarQ]:=
  Module[{res},
    res = rklsf[a, b, z];
    res /; !SameQ[res, $Failed]
  ];

End[ ]; (* End `Private` Context. *)

SetAttributes[
{ CompareMethods, FinalSolutions, InvariantDimensions, InvariantErrorPlot,
InvariantErrorSampleRate, InvariantErrorFunction,
RungeKuttaLinearStabilityFunction, StepDataPlot },
{ Protected, ReadProtected }
];

EndPackage[ ]; (* End package Context. *)

