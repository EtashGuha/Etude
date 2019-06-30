(* ::Package:: *)

(* :Title: Equation Trekker *)

(* :Context: EquationTrekker` *)

(* :Author: Rob Knapp *)

(* :Summary: An interactive tool for visualizing solutions of
             Differential Equations *)

(* :Copyright: Copyright 2004-2007, Wolfram Research, Inc. *)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 6.0 *)

(* :History:
    Version 1.0, Feb. 2004, Rob Knapp.
*)

(* :Keywords:

*)

(* :Sources:

*)

(* :Warnings: *)

(* :Limitations:  *)

(* :Discussion:

*)

(*******************************************)
BeginPackage["EquationTrekker`", "GUIKit`"];
(*******************************************)

(*******************************
   Usage Messages
 *******************************)
 
If[Not@ValueQ[EquationTrekker::usage],EquationTrekker::usage = "EquationTrekker[eqn, {x,y}, {t, tmin, tmax}] brings up a window \
and then plots the solution of the second order differential equation, \
eqn, such that {x[0], y[0]} is the point at the mouse cursor. \
EquationTrekker[eqn, x, {t, tmin, tmax}] brings up a \
window and then plots the solution of the first order differential equation, \
eqn, such that {t0, x[t0]} is the point at the mouse cursor. \
If you drag the mouse with the button down, the solution is updated continuously. \
If the equation is a second order equation, x' is considered as the y variable."];

If[Not@ValueQ[EquationTrekkerNonModal::usage],EquationTrekkerNonModal::usage = "EquationTrekkerNonModal is the non-modal dialog version of EquationTrekker."];

If[Not@ValueQ[Parameter::usage],Parameter::usage = "Parameter[name, value] specifies a variable which can be modified interactively."];

If[Not@ValueQ[TrekParameters::usage],TrekParameters::usage = "TrekParameters is an option to EquationTrekker specifying the dynamic parameter objects."];
If[Not@ValueQ[TrekGenerator::usage],TrekGenerator::usage = "TrekGenerator is an option to EquationTrekker specifying the system of trek generation."];

If[Not@ValueQ[InitializeGenerator::usage],InitializeGenerator::usage = "InitializeGenerator is a function called to initialize a generator for trek points.  To define a generator with name gen, you should define rules so that InitializeGenerator[gen, eqns, vars, {ivar, begin, end}] returns gen[data], where data includes whatever data is needed to generate trek points."];

If[Not@ValueQ[DifferentialEquationTrek::usage],DifferentialEquationTrek::usage = "DifferentialEquationTrek is a value for the option TrekGenerator which shows the solution of differential equations."];

If[Not@ValueQ[PoincareSection::usage],PoincareSection::usage = "PoincareSection is a value for the option TrekGenerator which allows you to show Poincare sections for differential equations"];

If[Not@ValueQ[EquationTrekkerState::usage],EquationTrekkerState::usage = "EquationTrekkerState[data] contains sufficient data so that if you use EquationTrekker[EquationTrekkerState[data]], the trek window will be restored to the point at which the state was saved."];

If[Not@ValueQ[TrekData::usage],TrekData::usage = "TrekData[data] represents the data which is shown as a single path or set of points in the EquationTrekker.  By default, only the initial conditions are shown."];

(*******************************)
Begin["`Private`"]
(*******************************)

(*******************************
   Options
 *******************************)

Options[EquationTrekker] = {
  PlotRange -> {Automatic,{-1,1}},
  ImageSize -> {400,400}, 
  TrekParameters -> {}, 
  TrekGenerator -> DifferentialEquationTrek
  };

Options[DifferentialEquationTrek] = Options[NDSolve];
Options[PoincareSection] = Join[{"SectionVariables"->None, "SectionCondition"->None, "SectionLocationMethod"->Automatic, "FilterFunction"->None}, Options[NDSolve]];

SetAttributes[ProcessOptionNames, Listable]; 
ProcessOptionNames[(r : (Rule | RuleDelayed))[name_Symbol, val_]] := 
    r[SymbolName[name], val];
ProcessOptionNames[opt_] := opt;

(*******************************
   Messages
 *******************************)
 
EquationTrekker::ncsol = "Could not compute the solution `1`";
EquationTrekker::prange = "Value of option PlotRange->`1` should be in the form {{xmin, xmax},{ymin, ymax}}.";
EquationTrekker::isize = "Value of option ImageSize->`1` should be in the form {horizontal pixels, vertical pixels}.";
EquationTrekker::tgen = "Value of option TrekGenerator->`1` should be the name of a trek generator or a list with a name followed by generator options."

(* What should these say? *)
DifferentialEquationTrek::only = "EquationTrekker for differential equations is for a single first order equation, single second order equation, or two first order equations.";

  
PoincareSection::dsvars = "The section variables `1` should be a length 2 subset of the dependent variables.";

PoincareSection::freal = "The value of the option SectionCondition->`1` should be a real valued function.";

(*******************************
   EquationTrekkerState Formatting

   The main idea here is to format invidual parts of the state
   expression.  This is one way to leave the graphics alone so
   that it shows appropriately in version 5 and 6.

 *******************************)

(*
    TrekData:  The idea here is to keep the data inside, but have it only 
               show the conditions in the formatting.
*)

TrekDataFormat[TrekData[disp_, cond_, ivdata_, {color_, style_}], form_] := 
    StringJoin[
        "TrekData[\"",
        ToString[disp, form],
        "\", \"<>\"]"];

Format[tdata_TrekData, OutputForm] := TrekDataFormat[tdata, OutputForm];

Format[tdata_TrekData, TextForm] := TrekDataFormat[tdata, TextForm];

TrekData /: MakeBoxes[
     tdata:TrekData[disp_, cond_, ivdata_, {color_, style_}], form_] :=
  InterpretationBox[StyleBox[#, "FontColor"->color], tdata]& [
        TrekDataFormat[tdata, form]];

EquationTrekkerStateFormat[EquationTrekkerState[indata_, parms_, trekdata_, opts_], form_] := 
    StringJoin[
        "EquationTrekkerState[\"", 
        ToString[indata, form], "\",\"",
        ToString[parms, form], "\",",
        ToString[TableForm[trekdata], form], ",\" <>\"]"];

Format[ets_EquationTrekkerState, OutputForm] := 
    EquationTrekkerStateFormat[ets, OutputForm]

Format[ets_EquationTrekkerState, TextForm] := 
    EquationTrekkerStateFormat[ets, TextForm]

EquationTrekkerState /: MakeBoxes[ets_EquationTrekkerState, form_] := 
    InterpretationBox[#, ets]&[EquationTrekkerStateFormat[ets, form]]

(*******************************
   EquationTrekker
 *******************************)
 
EquationTrekker[eqn_, dvars_, None, opts___] := 
  EquationTrekker[eqn, dvars, {None, 0, 1}, opts];

EquationTrekker[eqn_, dvars_, {ivar_, begin_, end_}, opts___] := 
  GUIRunModal["TrekFrame", {eqn, dvars, {ivar, begin, end}, opts}];

EquationTrekker[state:EquationTrekkerState[{eqns_, dvars_, iv_}, __]] := 
  EquationTrekker[eqns, dvars, iv, "State" -> state];


EquationTrekkerNonModal[eqn_, dvars_, None, opts___] := 
  EquationTrekkerNonModal[eqn, dvars, {None, 0, 1}, opts];

EquationTrekkerNonModal[eqn_, dvars_, {ivar_, begin_, end_}, opts___] := 
  GUIRun["TrekFrame", {eqn, dvars, {ivar, begin, end}, opts}];

EquationTrekkerNonModal[state:EquationTrekkerState[{eqns_, dvars_, iv_}, __]] := 
  EquationTrekkerNonModal[eqns, dvars, iv, "State" -> state];

(*****************************************
    DifferentialEquationTrek - TrekGenerator
 *****************************************)
 
DifferentialEquationTrek /: InitializeGenerator[DifferentialEquationTrek, eqns_, 
  dvarsin_, {ivar_, ivmin_, ivmax_}, opts___] := 
Module[{dvars = Flatten[{dvarsin}], order, finit, ndopts},
    (* Check form of dependent variables to be sure we have at most
       two or a first or second order equation.  *)
    If[Length[dvars] == 0 || Length[dvars] > 2,
        Message[DifferentialEquationTrek::only];
        Throw[$Failed]
    ];
    (* Convert from x[t] to x *)
    dvars = Map[If[MatchQ[#, _[ivar]], Head[#], #]&, dvars];
     
    (* Determine order and set up initial condition function *)
    order = Max[Cases[eqns, Derivative[j_][v_ /; MemberQ[dvars, v]][ivar] -> j, Infinity]];
    If[order > 2,
        Message[DifferentialEquationTrek::only];
        Throw[$Failed]
    ];

    If[Length[dvars] == 1,
        If[order == 1, 
            finit = Function[dv[#1] == First[#2]] /. dv->First[dvars],
            dvars = {First[dvars], Derivative[1][First[dvars]]}
        ];
    ];
    If[Length[dvars] == 2,
        finit = Function[Thread[Equal[{dv1[#1], dv2[#1]}, #2]]] /. Thread[{dv1, dv2}->dvars]];

    ndopts = FilterRules[{opts}, Options[NDSolve]];

    DifferentialEquationTrek[{eqns, eqns, dvars, ivar, finit, ndopts, None}]
]

DifferentialEquationTrek[{origeqns_, eqns_, dvars_, ivar_, finit_, ndopts_, state_}]["Variables"[]] := 
  {ivar, dvars}
    
DifferentialEquationTrek[{origeqns_, eqns_, dvars_, ivar_, finit_, ndopts_, state_}]["Display"[]] := 
  If[ListQ[origeqns] && (Length[origeqns] == 1), First[origeqns], origeqns]
    
DifferentialEquationTrek[___]["DisplayMode"[]] := "Line"
    
DifferentialEquationTrek[{origeqns_, eqns_, dvars_, ivar_, finit_, ndopts_, state_}]["FormatTrek"[iv0_, x0_, _]] := 
  finit[iv0, x0]
    
DifferentialEquationTrek[{origeqns_, eqns_, dvars_, ivar_, finit_, ndopts_, state_}]["ChangeParameters"[prules_]] := 
  DifferentialEquationTrek[{origeqns, origeqns /. prules, dvars, ivar, finit, ndopts, None}]
    
(de:DifferentialEquationTrek[{origeqns_, eqns_, dvars_, ivar_, finit_, ndopts_, state_}])["GenerateTrek"[x0_, {iv0_, ivmin_, ivmax_}]] := 
Module[{obj = de, newstate = state, sol, times, points, ic},
	ic = finit[iv0, x0];
    If[newstate === None,
        newstate = NDSolve`ProcessEquations[{eqns, ic},dvars, {ivar, ivmin, ivmax}, ndopts];
        If[ Head[ newstate] === NDSolve`ProcessEquations, Return[ $Failed]];
        newstate = First[newstate];
        obj[[1,-1]] = newstate;
    ];
    newstate = First[NDSolve`Reinitialize[newstate, ic]];
    sol; (* Hack to prevent extra copy due to LastValue *)
    NDSolve`Iterate[newstate, {ivmin, ivmax}];
    sol = NDSolve`ProcessSolutions[newstate];
    sol = dvars /. sol;
    times = First[sol]@"Coordinates"[];
    points = Map[(#@"ValuesOnGrid"[])&, sol];
    points = Transpose[Join[times, points]];
    {points, obj}
]
    
   
(*****************************************
    PoincareSection - TrekGenerator
 *****************************************)
 
PoincareSection /: InitializeGenerator[PoincareSection, eqns_, dvarsin_, {ivar_, ivmin_, ivmax_}, opts___] := 
Module[{dvars = dvarsin, condition, svars, flops, evmethod},
    (* Check form of dependent variables to be sure we have at most
       two or a first or second order equation.  *)
    If[Not[ListQ[dvars]], dvars = {dvars}];
    (* Convert from x[t] to x *)
    dvars = Map[If[MatchQ[#, _[ivar]], Head[#], #]&, dvars];
    order = Max[Cases[eqns, Derivative[j_][v_ /; MemberQ[dvars, v]][ivar] -> j, Infinity]];
    If[Length[dvars] < 2 && order < 2, 
        Message[PoincareSection::only];
        Throw[$Failed]
    ];

    ndopts = FilterRules[{opts}, Options[NDSolve]];
    flops = ProcessOptionNames[Flatten[{opts, Options[PoincareSection], Options[NDSolve]}]];
    
    svars = "SectionVariables" /. flops;
    condition = "SectionCondition" /. flops;
    evmethod = "SectionLocationMethod" /. flops;
    sfun = "FilterFunction" /. flops;
    method = "Method" /. flops;

    If[Not[ListQ[svars] && (Length[svars] == 2)],
        Message[PoincareSection::dsvars, svars, dvars]];
    (* Convert from x[t] to x *)
    svars = Map[If[MatchQ[#, _[ivar]], Head[#], #]&, svars];

(* 
    ###### Ideally should fix with ProcessEquations
    If[Not[MemberQ[dvars, svars[[1]]] && MemberQ[dvars, svars[[2]]]],
        Message[PoincareSection::dsvars, svars, dvars]];
*)

    finit = MakeInitFunction[Map[Function[{v}, v[#]], svars]];

    If[MatchQ[condition, Mod[arg_ /; LinearFunctionQ[ivar][arg], period_]],
        (* Optimization to use NDSolve`Iterate with NDSolve`StateData *)
        deltat = condition /. Mod[arg_, period_]:> period/D[arg, ivar];
        ndopts = {"DependentVariables"->dvars, ndopts};
        PoincareSection[{{eqns, eqns}, {deltat, deltat}, ivar, svars, sfun, finit, {ndopts, None}}],
   (* else *)
        (* Build up method option for NDSolve *)
        ndopts = Function[Evaluate[{MakeEventMethod[method, evmethod, condition, Map[Function[#[ivar]],svars], ivar, ivmin], "DependentVariables"->dvars, ndopts}]];
        PoincareSection[{{eqns, eqns}, {ndopts, ndopts}, ivar, svars, sfun, finit, EventLocator}]
    ]
]

PoincareSection[{{origeqns_, eqns_}, {origndopts_, ndopts_}, ivar_, svars_, sfun_, finit_, method_}]["Variables"[]] := 
  {ivar, svars}
    
PoincareSection[{{origeqns_, eqns_}, {origndopts_, ndopts_}, ivar_, svars_, sfun_, finit_, method_}]["Display"[]] := 
  If[ListQ[origeqns] && (Length[origeqns] == 1), First[origeqns], origeqns]
    
PoincareSection[___]["DisplayMode"[]] := "Points";
    
PoincareSection[{{origeqns_, eqns_}, {origndopts_, ndopts_}, ivar_, svars_, sfun_, finit_, method_}]["FormatTrek"[iv0_, x0_, _]] := 
  finit[iv0, x0]
    
PoincareSection[{{origeqns_, eqns_}, {origndopts_, ndopts_}, ivar_, svars_, sfun_, finit_, EventLocator}]["ChangeParameters"[prules_]] := 
  PoincareSection[{{origeqns, origeqns /. prules}, {origndopts, origndopts /. prules}, ivar, svars, sfun, finit, EventLocator}];
    
PoincareSection[{{origeqns_, eqns_}, {origdeltat_, deltat_}, ivar_, svars_, sfun_, finit_, {ndopts_, _}}]["ChangeParameters"[prules_]] := 
  PoincareSection[{{origeqns, origeqns /. prules}, {origdeltat, origdeltat /. prules}, ivar, svars, sfun, finit, {ndopts, None}}];
    
(* 
    The basic operation is a quite simple call to NDSolve since the
    work is embodied in the EventLocator method
*)
PoincareSection[{{origeqns_, eqns_}, {origndopts_, ndopts_}, ivar_, svars_, sfun_, finit_, EventLocator}]["GenerateTrek"[x0_, {iv0_, ivmin_, ivmax_}]] := 
Module[{sol, t, ts},
    points = Reap[
        Internal`DeactivateMessages[NDSolve[{eqns, finit[iv0, x0]}, {}, {ivar, ivmin, ivmax}, ndopts[ivmin, ivmax]], NDSolve::noout]];
    points = GetReapData[points];
    If[sfun =!= None && Length[points] > 0,
        points[[All,{2,3}]] = Map[sfun, points[[All, {2,3}]]]];
    points
]

(*
    This more complicated version is an optimization for the case of 
    a periodic section across the independent variable.
*)
(ps:PoincareSection[{{origeqns_, eqns_}, {origdeltat_, deltat_}, ivar_, svars_, sfun_, finit_, {ndopts_, state_}}])["GenerateTrek"[x0_, {iv0_, ivmin_, ivmax_}]] := 
Module[{obj = ps, newstate = state, sol, t, ts, mod},
    If[newstate === None,
        newstate = NDSolve`ProcessEquations[{eqns, finit[iv0, x0]},{}, ivar, ndopts];
        If[ Head[ newstate] === NDSolve`ProcessEquations, Return[ $Failed]];
        newstate = First[newstate];
        obj[[1,-1]] = newstate;
    ];
    newstate = NDSolve`ReinitializeVector[newstate, iv0, x0];
    sol; (* Hack to prevent extra copy due to LastValue *)
    If[Not[TrueQ[Positive[deltat]]],
        Message[PoincareSection::freal, deltat];
        Return[$Failed]
    ];
    points = Reap[
        t = iv0;
        If[t > ivmax, 
            t = iv0 - ivmax;
            mod = Mod[t, deltat];
            t = ivmax + mod;
            If[Developer`ZeroQ[mod], t += deltat];
            If[t < iv0, GetSolutionAt[newstate, t, svars, "Backward"]];
            t -= deltat];
        While[t >= ivmin, 
            Sow[GetSolutionAt[newstate, t, svars, "Backward"]];
            t -= deltat];
        t = iv0;
        If[t < ivmin, 
            t = ivmin - iv0;
            mod = Mod[t, deltat];
            t = ivmin - mod;
            If[Developer`ZeroQ[mod], t -= deltat];
            If[t > iv0, GetSolutionAt[newstate, t, svars, "Forward"]];
            t += deltat];
        While[t <= ivmax, 
            Sow[GetSolutionAt[newstate, t, svars, "Forward"]];
            t += deltat];
    ];
    points = GetReapData[points];
    If[sfun =!= None && Length[points] > 0,
        points[[All,{2,3}]] = Map[sfun, points[[All, {2,3}]]]];
    {points, obj}
]

SetAttributes[GetSolutionAt, HoldFirst];
GetSolutionAt[state_, t_, dvars_, direction_] := 
Module[{sol, ts},
    NDSolve`Iterate[state, t];
    sol = NDSolve`ProcessSolutions[state, direction];
    ts = state@"CurrentTime"[direction];
    Prepend[Map[#[ts]&, dvars] /. sol, t]
] 

GetReapData[{res_, {}}] := {};
GetReapData[{res_, {data_, ___}}] := data;
GetReapData[_] := {};
    
MakeEventMethod[submethod_, evmethod_, condition_, svars_, ivar_, ivmin_] := 
Rule[Method, {"EventLocator", Method->submethod, "Event"->condition, "EventAction":>If[#1 <= ivar <= #2, Sow[Prepend[svars, ivar]]], "EventLocationMethod"->If[evmethod === Automatic, "LinearInterpolation", evmethod]}];

MakeInitFunction[cl_] := Function[Thread[Equal[cl, #2]]];

LinearFunctionQ[v_][f_] := Module[{d = D[f,v]}, Not[Developer`ZeroQ[d]] && FreeQ[d, v]];


(*******************************)
End[]   (* end private context *)
(*******************************)

(*******************************)
EndPackage[];
(*******************************)
