(* :Name: BoundaryRegionPlot` *)

(* :Title: Boundary Region Plot. *)

(* :Author: Mark Sofroniou *)

(* :Summary:
	This package provides a function that shows the mesh and positions of given 
	boundary conditions for a solution Region for NDSolve.
*)

(* :Context: DifferentialEquations`BoundaryRegionPlot` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 2010, Wolfram Research, Inc. *)

(* :Keywords:
 NDSolve, finite element method, boundary condition, partial differential equations.
*)

(* :Source:
*)

(* :Mathematica Version: 8.0 *)

(* :Limitation:
*)

(* :Discussion:
*)

BeginPackage["DifferentialEquations`BoundaryRegionPlot`"];

BoundaryRegionPlot::usage = "BoundaryRegionPlot[eqns, dvars, region] shows the
meshing of region highlighting where the boundary conditions included in
eqns are applied in NDSolve[eqns, dvars, region]."

Unprotect[BoundaryRegionPlot];

Begin["`Private`"];

$ContextPath = Join[$ContextPath, {"NDSolve`FEM`", "Graphics`Mesh`"}];

ClearAll[BoundaryRegionPlot]; 

Options[BoundaryRegionPlot] = {
	"MeshOptions" -> Automatic, 
	Mesh -> Automatic, 
	PlotStyle -> Automatic, 
	"SeparateDependentVariables" -> False}; 
	
BoundaryRegionPlot[bcs_, dvars_, region_, opts : OptionsPattern[]] := 
With[{res = Catch[BoundaryRegionPlotImpl[bcs, dvars, region, opts], BoundaryRegionPlot]}, 
	res /; UnsameQ[res, $Failed]
];

Clear[throwUp];
throwUp[___] := Throw[$Failed, BoundaryRegionPlot];

ClearAll[BoundaryRegionPlotImpl]; 
BoundaryRegionPlotImpl[eqns_, dvars_, region_Region, opts:OptionsPattern[BoundaryRegionPlot]] := 
Module[{svars, dvarslist = Flatten[{dvars}], method, state, femdata},
	svars = region["Variables"];
	method = Join[{"FiniteElement"}, FilterRules[{opts}, Options[NDSolve`FiniteElement]]];
	state = NDSolve`ProcessEquations[eqns, dvarslist, region, Method -> method];
	If[Not[MatchQ[state, {__NDSolve`StateData}]], throwUp];
	femdata = First[state]["FiniteElementData"];
	If[UnsameQ[Head[femdata], NDSolve`FEM`FiniteElementData] || Not[System`Private`ValidQ[femdata]], throwUp];
	NDSolve`FEM`BoundaryConditionPlot[femdata, dvarslist, FilterRules[{opts}, Options[NDSolve`FEM`BoundaryConditionPlot]]]
];

FacePrimitive[0] = Point[Flatten[#]] &;
FacePrimitive[1] = Line;
FacePrimitive[2] = Polygon;

GraphicsType[3] = Graphics3D;
GraphicsType[_] = Graphics;

GetElementPrimitives[{}, ___] := {};
GetElementPrimitives[points_, emesh_, ename_, prim_] := 
Module[{inc = emesh[ename][[All, 1]]},
	Apply[Function[{bc, pos},
		Tooltip[
			prim[Flatten[MapThread[Function[{i, p}, i[[p]]], {inc, pos}], 1]], 
			ToString[bc]
		]],
    	points,
    	{1}
    ]
];

GetPrimitives[{points_, faces_}, emesh_, faceprim_, colors_] := 
Module[{prim},
	prim = Flatten[{
		GetElementPrimitives[points, emesh, "BoundaryPointElements", FacePrimitive[0]], 
		GetElementPrimitives[faces, emesh, "BoundaryElements", faceprim]
	}];
	Table[{colors[i], prim[[i]]}, {i, Length[prim]}]
];

ClearAll[NDSolve`FEM`BoundaryConditionPlot];
Options[NDSolve`FEM`BoundaryConditionPlot] := {
	Mesh -> Automatic, 
	Axes -> True, 
	PlotStyle -> Automatic, 
	"SeparateDependentVariables" -> False};
	
NDSolve`FEM`BoundaryConditionPlot[femdata_FiniteElementData, dvars_List, opts : OptionsPattern[]] := 
Module[{emesh, bcp, colors, bdata, n, showmesh, elems, flops, gropts, 
		pstyle, opstyle,coords, overlap, plots},
	flops = Flatten[{opts, Options[NDSolve`FEM`BoundaryConditionPlot]}];
	colors = ColorData[1];
	n = emesh["Dimension"];
	emesh = femdata["Mesh"];
	bcp = femdata["BoundaryConditionPositions"];
	bcp = bcp /. {BoundaryCondition[Natural[_, val_], pred_] :> GeneralizedNeumann[val, pred], 
		          BoundaryCondition[Essential[_, val_], pred_] :> BoundaryCondition[val, pred]};
	bcp = {bcp[[1]], Flatten[bcp[[{2, 3, 4}]], 1]};
	overlap = femdata["OverlappingBoundaryConditionPositions"];
	overlap = overlap /. {BoundaryCondition[Natural[_, val_], pred_] :> GeneralizedNeumann[val, pred], 
		          BoundaryCondition[Essential[_, val_], pred_] :> BoundaryCondition[val, pred]};
	If[TrueQ[OptionValue["SeparateDependentVariables"]],
		vpat[var_Symbol] = BoundaryCondition[
			PatternTest[_, Function[{eq},  MemberQ[eq, var, Infinity, Heads -> True]]], 
			_];
		bcp = Map[
			Function[{var}, 
				Map[Function[{pf}, Cases[pf, {vpat[var], _}]], bcp]], 
				dvars
		];
		overlap = 
			Map[Function[{var}, 
				Map[Function[{pf}, Cases[pf, {{vpat[var] ..}, _}]], overlap]], 
     		dvars
    	 ];
	(* else *),
		bcp = {bcp};
		overlap = {overlap}
	];
	bdata = Map[GetPrimitives[#, emesh, FacePrimitive[n - 1], colors] &, bcp];
	overlap = Map[GetPrimitives[#, emesh, FacePrimitive[n - 1], Red &] &, overlap];
	showmesh = OptionValue[Mesh];
	If[showmesh === Automatic, showmesh = If[n == 3, "Boundary", All]];
	Switch[showmesh,
		All, 
			elems = emesh["MeshElements"],
		"Boundary", 
			elems = emesh["BoundaryElements"],
		None,
			showmesh = Sequence[],
		_,
			throwUp[Message[BoundaryRegionPlot::mesh, showmesh]]
	];
	If[ListQ[elems], showmesh = {Gray, NDSolve`FEM`MeshElementToWireframe[elems]}];
	pstyle = OptionValue[PlotStyle];
	If[pstyle === Automatic, 
		pstyle = {PointSize[0.015], Thickness[.01]};
		opstyle = {PointSize[0.02], Thickness[0.015]}
   (* else *),
		If[! ListQ[pstyle], pstyle = {pstyle}];
		opstyle = pstyle
	];
	gropts = FilterRules[flops, Options[GraphicsType[n]]];
	coords = emesh["Coordinates"];
	If[n == 1, 
		coords = Transpose[{Flatten[coords], ConstantArray[0., Length[coords]]}];
		gropts = Append[gropts, Axes -> {True, False}]
   ];
	plots = MapThread[
		Function[
			GraphicsType[n][
				GraphicsComplex[
					coords, 
					{showmesh, Join[pstyle, #1], Join[opstyle, #2]}
				], 
			gropts]
		], 
		{bdata, overlap}
	];
	If[Not[TrueQ[OptionValue["SeparateDependentVariables"]]], plots[[1]], plots]
]
  
End[ ]; (* End `Private` Context. *)

SetAttributes[BoundaryRegionPlot, Protected];

EndPackage[ ]; (* End package Context. *)

