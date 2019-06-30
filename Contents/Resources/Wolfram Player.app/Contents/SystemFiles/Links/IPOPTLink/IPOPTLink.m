BeginPackage["IPOPTLink`"]

(* Documented functions *)
IPOPTMinimize
IPOPTArgMin
IPOPTMinValue
IPOPTData
IPOPTDataExpressions
ParametricIPOPTMinimize

IPOPTReturnCode::usage = "IPOPTReturnCode[data] gives the IPOPT solver return code \
from an IPOPTData expression data."

IPOPTStringStatus::usage = "IPOPTStringStatus[data] gives the IPOPT solver return status as a string \
from an IPOPTData expression data."

IPOPTConstraintMultipliers::usage = "IPOPTConstraintMultipliers[data] gives the values of the \
Lagrange multipliers for the function constraints from an IPOPTData expression data."

IPOPTLowerBoundMultipliers::usage = "IPOPTLowerBoundMultipliers[data] gives the values of the \
Lagrange multipliers for the lower bound variable constraints from an IPOPTData expression data."

IPOPTUpperBoundMultipliers::usage = "IPOPTUpperBoundMultipliers[data] gives the values of the \
Lagrange multipliers for the upper bound variable constraints from an IPOPTData expression data."

IPOPTDataID::usage = "IPOPTDataID[data] gives the instance id of an IPOPTData expression data."

IPOPTDataQ::usage = "IPOPTDataQ[expr] gives True if expr represents an active \
instance of an IPOPTData object."

IPOPTDataCreate::usage = "IPOPTDataCreate[] creates an instance \
of an IPOPTData expression."

IPOPTDataDelete::usage = "IPOPTDataDelete[expr] removes an instance \
of an IPOPTData expression, freeing up memory."

$IPOPTVersion::usage = "$IPOPTVersion gives the version number of the IPOPT library."

$IPOPTLinkLibrary::usage = "$IPOPTLinkLibrary is the full path to the IPOPT library loaded by IPOPTLink."

$IPOPTLinkDirectory::usage = "$IPOPTLinkDirectory gives the location of the IPOPTLink library."

LoadIPOPTLink::usage  = "LoadIPOPTLink[] loads the IPOPTLink library."

ParametricIPOPTData::usage = "ParametricIPOPTData[id] represents an instance of a \
ParametricIPOPTData object that contains information necessary for the solution of a \
parametric optimization problem as returned by ParametricIPOPTMinimize."

ParametricIPOPTDataID::usage = "ParametricIPOPTDataID[ParametricIPOPTData[id]] returns the \
instance id of the ParametricIPOPTData managed library expression"

ParametricIPOPTDataQ::usage = "ParametricIPOPTDataQ[expr] returns True if expr represents an active \
instance of an ParametricIPOPTData object."

ParametricIPOPTDataDelete::usage = "ParametricIPOPTDataDelete[expr] removes an instance \
of an ParametricIPOPTData expression, freeing up memory."

ParametricIPOPTDataExpressions::usage = "ParametricIPOPTDataExpressions[] shows all active \
ParametricIPOPTData expression instances."


Begin["`Private`"]

(*
	TODO: 
	-tests
	-add checks for convergence failure on the C++ end
	-messages if convergence failure
*)

$IPOPTVersion = "3.12.4"

$IPOPTLinkDirectory = FileNameJoin[{DirectoryName[$InputFileName], "LibraryResources", $SystemID}];

$DebugPrint = 0;
(* Set with:
IPOPTLink`Private`$DebugPrint = 1; (* prints in console *)
IPOPTLink`Private`dPrint = Print; (* to debug top level code *)
*)

(* Solver constants *)
$EFFECTIVEINFINITY = 10.^19;


(*
	Load the library functions
*)

needInitialization = True;

LoadIPOPTLink[]:=
Block[{$LibraryPath = $IPOPTLinkDirectory},
	$IPOPTLinkLibrary = FindLibrary["libIPOPTLink"];
	(* solver application data *)
	IPOPTApplicationNew0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_new", {}, {Integer}];
	IPOPTApplicationFree0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_free", {Integer}, {Integer}];
	IPOPTApplicationSetNumericValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_setNumericValue", {Integer, UTF8String, Real}, {Integer}];
	IPOPTApplicationSetStringValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_setStringValue", {Integer, UTF8String, UTF8String}, {Integer}];
	IPOPTApplicationSetIntegerValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_setIntegerValue", {Integer, UTF8String, Integer}, {Integer}];
	IPOPTApplicationPrintOptionsList0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_PrintList", {Integer}, {Integer}];
	IPOPTApplicationGetNumericValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_getNumericValue", {Integer, UTF8String}, {Real}];
	IPOPTApplicationGetStringValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_getStringValue", {Integer, UTF8String}, {UTF8String}];
	IPOPTApplicationGetIntegerValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTApplicationMap_getIntegerValue", {Integer, UTF8String}, {Integer}];
 
	(* main solver *)
	IPOPTSolve0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "LinkSolve",
		{Integer, Integer, Integer, {Integer, 2}, {Integer, 2}, {Real, 1}, {Real, 1}, {Real, 1}, {Real, 1},
			{Real, 1}, {Real, 1}, Integer}, Integer];
 
 	(*persistent solution data*)
	IPOPTDataDelete0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_delete", {Integer}, Integer];
	IPOPTDataIDList = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retIDList", {}, {Integer, 1}];
	IPOPTMinValue0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retObjVal", {Integer}, Real];
	IPOPTArgMin0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retX", {Integer}, {Real, 1}];
	IPOPTReturnCode0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retStatus", {Integer}, Integer];
	IPOPTConstraintMultipliers0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retLambda", {Integer}, {Real, 1}];
	IPOPTLowerBoundMultipliers0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retZLower", {Integer}, {Real, 1}];
	IPOPTUpperBoundMultipliers0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "SolutionMap_retZUpper", {Integer}, {Real, 1}];
 	
	(* IPOPTCallbackData *)
	IPOPTCallbackDataDelete0 = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTCallbackDataMap_delete", {Integer}, Integer];
	IPOPTCallbackDataIDList = LibraryFunctionLoad[$IPOPTLinkLibrary, "IPOPTCallbackDataMap_retIDList", {}, {Integer, 1}];

	needInitialization = False;
 ];


LoadIPOPTLink[]

(*
	Wrappers to call the LibraryLink functions
*)

(* ApplicationMap related: *)

IPOPTApplicationNew[] := Module[{}, IPOPTApplicationNew0[]];

IPOPTApplicationFree[n_Integer] := Module[{}, IPOPTApplicationFree0[n]];

(* IPOPT options setting, getting and printing: *)

IPOPTApplicationSetNumericValue[id_Integer, str_String, val_?NumericQ] := Module[{}, IPOPTApplicationSetNumericValue0[id, str, val]];

IPOPTApplicationSetStringValue[id_Integer, str_String, val_String] := Module[{}, IPOPTApplicationSetStringValue0[id, str, val]];

IPOPTApplicationSetIntegerValue[id_Integer, str_String, val_Integer] := Module[{}, IPOPTApplicationSetIntegerValue0[id, str, val]];

setIPOPTAppOption[appid_Integer, Rule[opt_String, val_String]] := IPOPTApplicationSetStringValue[appid, opt, val];
setIPOPTAppOption[appid_Integer, Rule[opt_String, val_Integer]] := IPOPTApplicationSetIntegerValue[appid, opt, val];
setIPOPTAppOption[appid_Integer, Rule[opt_String, val_?NumericQ]] := IPOPTApplicationSetNumericValue[appid, opt, N[val]];
setIPOPTAppOption[___] := Message[IPOPTMinimize::invalidopt];

getIPOPTAppNumericOption[id_Integer, str_String] := Module[{}, IPOPTApplicationGetNumericValue0[id, str]];

getIPOPTAppStringOption[id_Integer, str_String] := Module[{}, IPOPTApplicationGetStringValue0[id, str]];

getIPOPTAppIntegerOption[id_Integer, str_String] := Module[{}, IPOPTApplicationGetIntegerValue0[id, str]];

IPOPTApplicationPrintOptionsList[id_] =
Module[{},
	IPOPTApplicationPrintOptionsList0[id]
];

(* IPOPTData expression (SolutionMap) related: *)

IPOPTDataID[e_IPOPTData] := ManagedLibraryExpressionID[e, "ipopt_solution_instance_manager"];

IPOPTDataQ[e_IPOPTData] := ManagedLibraryExpressionQ[e, "ipopt_solution_instance_manager"];
IPOPTDataQ[_] := False;

testIPOPTData[][e_] := testIPOPTData[IPOPTData][e];
testIPOPTData[mhead_Symbol][e_] :=
If[TrueQ[IPOPTDataQ[e]],
	True,
	Message[MessageName[mhead, "ipoptinst"], e]; False
];
testIPOPTData[_][e_] := TrueQ[IPOPTDataQ[e]];

General::ipoptinst = "`1` does not represent an active IPOPTData object.";

IPOPTDataCreate[] :=
Module[{},
	If[needInitialization, LoadIPOPTLink[]];
	CreateManagedLibraryExpression["ipopt_solution_instance_manager", IPOPTData]
];

IPOPTDataDelete[IPOPTData[id_]?(testIPOPTData[IPOPTDataDelete])] :=
Module[{},
	IPOPTDataDelete0[id]
];
IPOPTDataDelete[l:{_IPOPTData..}] := IPOPTDataDelete /@ l;

IPOPTDataExpressions[] :=
Module[{list},
	If[needInitialization, LoadIPOPTLink[]];
	list = IPOPTDataIDList[];
	If[!ListQ[list],
	   $Failed,
	   Map[IPOPTData, list]]
]

IPOPTMinValue[IPOPTData[id_]?(testIPOPTData[IPOPTMinValue])] :=
Module[{},
	Quiet[IPOPTMinValue0[id], LibraryFunction::"rnset"]
];

IPOPTArgMin[IPOPTData[id_]?(testIPOPTData[IPOPTArgMin])] :=
Module[{},
	Quiet[IPOPTArgMin0[id], LibraryFunction::"rnset"]
];

IPOPTReturnCode[IPOPTData[id_]?(testIPOPTData[IPOPTArgMin])] :=
Module[{},
	IPOPTReturnCode0[id]
];

IPOPTConstraintMultipliers[IPOPTData[id_]?(testIPOPTData[IPOPTConstraintMultipliers])] :=
Module[{},
	IPOPTConstraintMultipliers0[id]
];

IPOPTLowerBoundMultipliers[IPOPTData[id_]?(testIPOPTData[IPOPTLowerBoundMultipliers])] :=
Module[{},
	IPOPTLowerBoundMultipliers0[id]
];

IPOPTUpperBoundMultipliers[IPOPTData[id_]?(testIPOPTData[IPOPTUpperBoundMultipliers])] :=
Module[{},
	IPOPTUpperBoundMultipliers0[id]
];

(* IPOPTData related: *)

IPOPTCallbackDataCreate[] :=
Module[{},
	If[needInitialization, LoadIPOPTLink[]];
	CreateManagedLibraryExpression["IPOPTCallbackData_instance_manager", IPOPTCallbackData]
];

IPOPTCallbackDataID[e_IPOPTCallbackData] := ManagedLibraryExpressionID[e, "IPOPTCallbackData_instance_manager"];



(* Main solver: *)

(* can put here changes to the default IPOPT options, e.g. "tol" -> .0000001 *)
Options[IPOPTSolve] = {"print_level" -> 0, "sb"-> "yes"};

IPOPTSolve[solID_, dataID_, jacgpos_, hpos_, initX_, xL_, xU_, gL_, gU_, params_, opts : OptionsPattern[]] :=
Module[{appID, options, err, errargs},

	If[!ListQ[opts], options = List[opts], options = opts];
	options = Join[Options[IPOPTSolve], options];
	(* workaround for bug 306505: *)
	If[$SystemID === "Linux", options = Join[{"linear_solver" -> "mumps"}, options]];

	(* setup an IPOPTApplication object *)
	appID = IPOPTApplicationNew[];

	(* set the associated solver options *)
	Map[(setIPOPTAppOption[appID, #]) &, options];
	(* IPOPTApplicationPrintOptionsList[appID]; *)

	(* calls LinkSolve in IPOPOTLink.cpp *)
	err = IPOPTSolve0[solID, dataID, appID, jacgpos, hpos, initX, xL, xU, gL, gU, params, $DebugPrint];

	errargs = GetIPOPTErrorArguments[appID, err];

	IPOPTApplicationFree[appID];

	{err, errargs}
];


(* 
	IPOPTMinimize
*)

Options[IPOPTMinimize] = {StepMonitor -> None, "IPOPTOptions" -> {}, "MessageHead" -> IPOPTMinimize,
	Compiled -> Automatic, "RuntimeOptions" -> Automatic};
(* For a full list of IPOPT options see
http://www.coin-or.org/Ipopt/documentation/node39.html *)

SetAttributes[IPOPTMinimize, HoldAll];

IPOPTMinimize[fx_, xvars_, x0_, opts : OptionsPattern[]]:=
Module[{xbounds, gx, gbounds},
	xbounds = Table[{-Infinity, Infinity}, Length[x0]];
	gx = {1};
	gbounds = {{-Infinity, Infinity}};
	IPOPTMinimize[fx, xvars, x0, xbounds, gx, gbounds, opts]
];

IPOPTMinimize[fx_, xvars_, x0_, xbounds_, opts : OptionsPattern[]]:=
Module[{gx, gbounds},
	gx = {1};
	gbounds = {{-Infinity, Infinity}};
	IPOPTMinimize[fx, xvars, x0, xbounds, gx, gbounds, opts]
];

IPOPTMinimize[fx_, xvars_, x00_, bounds0_, gx0_, gbounds0_, opts : OptionsPattern[]] :=
Module[{x0=x00, heldgx = Hold[gx0], gbounds, bounds, stepmonitor, ipOpts, solInstance, solID, callbackData, callbackDataID, temp, err,
	jacgpos, hpos, lbounds, ubounds, lgbounds, ugbounds, params = {}, mhead, errargs, compiled,
	errhandler, warnmsgs},

	dPrint[InputForm["IPOPTMinimize"[fx, xvars, x0, bounds0, gx0, gbounds0, opts]]];

	temp = IPOPTProcessOptions[{opts}, heldgx, gbounds0, bounds0, IPOPTMinimize];
	If[ListQ[temp] && Length[temp] === 9,
		{ipOpts, heldgx, gbounds, bounds, stepmonitor, compiled, errhandler, warnmsgs, mhead} = temp,
		Return[$Failed]
	];
	dPrint["IPOPTMinimize: processed options"];
	temp = Catch[
		IPOPTMinimizeCheckInputs[fx, heldgx, gbounds, xvars, bounds, x0],
		"IPOPTTag"];
	If[ListQ[temp] && Length[temp] === 4,
		{x0, bounds, heldgx, gbounds} = temp,
		Return[$Failed]
	];
	dPrint["IPOPTMinimize: checked inputs"];
	temp = Catch[
		IPOPTMinimizeProcessFunctions[fx, heldgx, stepmonitor, xvars, x0, compiled, Evaluate[mhead]],
		"IPOPTTag"];
	If[ListQ[temp] && Length[temp] === 2,
		{jacgpos, hpos} = temp,
		Return[$Failed]
	];
	dPrint["IPOPTMinimize: processed functions"];
	(* setup an IPOPTCallbackData object to store the connected callback (compiled) functions *)
	callbackData = IPOPTCallbackDataCreate[];
	callbackDataID = IPOPTCallbackDataID[callbackData];

	(* setup a Solution instance *)
	solInstance = IPOPTDataCreate[];
	solID = IPOPTDataID[solInstance];

	dPrint["IPOPTMinimize: set up IPOPTCallbackData and Solution instance"];
	{lbounds, ubounds} = Transpose[bounds];
	{lgbounds, ugbounds} = Transpose[gbounds];

	dPrint["IPOPTMinimize: solving..."];
	(* solve *)
	Quiet[
		temp = IPOPTSolve[solID, callbackDataID, jacgpos, hpos, x0, lbounds, ubounds, lgbounds, ugbounds, params, ipOpts],
		"Compiler"
	];
	If[ListQ[temp] && Length[temp] === 2,
		{err, errargs} = temp;
		dPrint["IPOPTMinimize: solved with return code ", err];
	,
		dPrint["IPOPTMinimize: solve failed"];
		Return[$Failed];
	];

	(* return the solution instance or $Failed if the error is critical *)
	If[TrueQ[err === 0], solInstance, IPOPTErrorCodeHandler[err, errargs, mhead, solInstance, errhandler, warnmsgs]]
];

CompiledFunctionQ[cf_CompiledFunction?System`Private`ValidQ] := True;
CompiledFunctionQ[_]:= False;
NumericalFunctionQ[nf_Experimental`NumericalFunction?System`Private`ValidQ] := True;
NumericalFunctionQ[_] := False;
SparseArrayQ[cf_SparseArray?System`Private`ValidQ] := True;
SparseArrayQ[_]:= False;

Clear[setSparseValues];
setSparseValues[HoldPattern[SparseArray[_, dims_, imp_, {v_, {rowp_, colp_}, _}]], vals_] :=
	SparseArray[Automatic, dims, imp, {v, {rowp, colp}, vals}] /; (Length[vals] == Last[rowp]);

SetAttributes[IPOPTMinimizeProcessFunctions, HoldAll];
IPOPTMinimizeProcessFunctions[fx_, heldgx_, stepmonitor_, xvars_, x0_, compiled_, mhead_] :=
Module[{gx, headf, headg, nfx, ngx, nx, glength, gradfx, jacgx, jacgpos,
	hfx, chf, hgx, chg, hpos, hfpos, hgpos,
	constantf = False, constantg = False, linearf = False, linearg = False,
	nfargs, ngargs, nfdims, ngdims, fvecvar, gvecvar,
	symsx, assocf, assocg, s, symsl, cf, cgradf, cg, cjacg, ch,
	gradfvals, gradfshape, jacgvals, jacgshape, hfvals, hfshape, hgvals, hgshape,
	gradfs, shf, shg, jacgs, hs, cmon, monitor, compile, compOpts,
	cproc, cres},

	(* load callback manager functions *)
	If[needInitialization, LoadIPOPTLink[]];

	If[ListQ[compiled],
		If[Length[compiled] >= 1,
			compile = First[compiled]];
		If[Length[compiled] >= 2,
			compOpts = Rest[compiled];
		,
			compOpts = {};
		];
	,
		compile = compiled;
		compOpts = {};
	];
	If[compile === False,
		compOpts = "RuntimeOptions" -> "Quality";
	,
		compile = True;
	];

	(* Check if fx and gx are NumericalFunctions,
		if not make NumericalFunctions  *)
	headf = Part[Hold[fx], 1, 0];
	If[NumericalFunctionQ[headf],
		nfx = headf;
	, (* else *)
		If[NumericalFunctionQ[fx],
			nfx = fx;
		, (* else *)
			With[{smon = ReleaseHold[StepMonitor:>Evaluate[stepmonitor]]},
				nfx = Experimental`CreateNumericalFunction[xvars, fx, {},
				Gradient -> {Automatic, "Sparse" -> False}, Compiled -> compiled, smon]];
		];
	];
	headg = Part[heldgx, 1, 0];
	If[NumericalFunctionQ[headg],
		ngx = headg;
		glength = ngx["ResultDimensions"][[1]];
	, (* else *)
		gx = ReleaseHold[heldgx];
		If[NumericalFunctionQ[gx],
			ngx = gx;
			glength = ngx["ResultDimensions"][[1]];
		, (* else *)
			If[ListQ[gx],
				glength = Length[gx];
			, (* else *)
				glength = Length[gx /. AssociationThread[xvars -> x0]];
			];
			ngx = Experimental`CreateNumericalFunction[xvars, gx, {glength}, Compiled -> compiled];
		];
	];

	nx = Length[x0];
	symsx = Table[Unique["x", Temporary], {nx}];
	s = Unique["s", Temporary];
	symsl = Table[Unique["l", Temporary], {glength}];

	nfargs = nfx["ArgumentNames"];
	nfdims = nfx["ArgumentDimensions"];
	ngargs = ngx["ArgumentNames"];
	ngdims = ngx["ArgumentDimensions"];
	Switch[Length[nfargs],
		0, If[nfdims === {{nx}}, fvecvar = True, fvecvar = False];,
		1, If[nfdims === {{nx}}, fvecvar = True,
			If[nx === 1 && nfdims === {{}}, fvecvar = False, Return[$Failed]]];,
		nx, If[Flatten[nfdims] === {}, fvecvar = False, Return[$Failed]];,
		_, Return[$Failed];
	];
	Switch[Length[ngargs],
		0, If[ngdims === {{nx}}, gvecvar = True, gvecvar = False];,
		1, If[ngdims === {{nx}}, gvecvar = True,
			If[nx === 1 && ngdims === {{}}, gvecvar = False, Return[$Failed]]];,
		nx, If[Flatten[ngdims] === {}, gvecvar = False, Return[$Failed]];,
		_, Return[$Failed];
	];

	If[nfargs =!= None,
		If[!TrueQ[fvecvar],
			assocf = AssociationThread[nfargs -> symsx]
		,
			assocf = (nfargs[[1]] -> symsx)
		];
	];
	If[ngargs =!= None,
		If[!TrueQ[gvecvar],
			assocg = AssociationThread[ngargs -> symsx]
		,
			assocg = (ngargs[[1]] -> symsx)
		];
	];

	constantf = NumericQ[nfx["FunctionExpression"]];
	constantg = VectorQ[ngx["FunctionExpression"], NumericQ];
	If[nfargs =!= None,
		If[fvecvar,
			linearf = Internal`LinearQ[nfx["FunctionExpression"]/.assocf, symsx];
			,
			linearf = Internal`LinearQ[nfx["FunctionExpression"], nfargs];
		];
	,
		linearf = Internal`LinearQ[nfx["FunctionExpression"][symsx], symsx];
	];
	If[ngargs =!= None,
		If[gvecvar,
			linearg = VectorQ[ngx["FunctionExpression"]/.assocg, Internal`LinearQ[#, ngargs[[1]]/.assocg] &];
		,
			linearg = VectorQ[ngx["FunctionExpression"], Internal`LinearQ[#, ngargs] &];
		];
		,
		linearg = VectorQ[ngx["FunctionExpression"][symsx], Internal`LinearQ[#, symsx] &];
	];

	(*
		Make compiled functions, derivatives and monitor
	*)

	(* objective function *)
	cf = nfx["CompiledFunction"];
	If[CompiledFunctionQ[cf],
		If[!TrueQ[fvecvar],
			cf = makeVectorCompiledObjectiveFunction[cf, compOpts][Sequence@@symsx];
		,
			cf = makeCompiledObjectiveFunction[cf, compOpts];
		]
	, (* else *)
		cf = With[{f = nfx}, Compile[{{X, _Real, 1}}, {1.*f[X]},
			{{Experimental`NumericalFunction[__][_], _Real, 0}}, Evaluate[compOpts]]];
	];

	(* gradient of objective function *)
	gradfx = nfx["GradientExpression"];
	gradfshape = nfx["GradientShape"];
	If[NumericalFunctionQ[gradfx] && compile,
		If[SparseArrayQ[gradfshape],
			If[nfargs =!= None, gradfvals = gradfx["FunctionExpression"] /. assocf,
				gradfvals = gradfx["FunctionExpression"][symsx]];
			gradfs = With[{sh = gradfshape, vals = gradfvals},
				Normal[setSparseValues[sh, vals]]];
			cgradf = With[{sx = symsx, c = 1.*gradfs},
				Compile[{{X, _Real, 1}}, Module[sx, sx = X; c], Evaluate[compOpts]]];
		, (* else *)
			cgradf = gradfx["CompiledFunction"];
			If[CompiledFunctionQ[cgradf],
				If[!TrueQ[fvecvar],
					cgradf = makeVectorCompiledFunction[cgradf, compOpts][Sequence@@symsx];
				];
			];
		];
	, (* else *)
		(* For a constant NumericalFunction "GradientExpression" gives "FiniteDifference"
		   (due to no variable dependence), but it is better to use a symbolic derivative *)
		If[constantf,
			cgradf = With[{c = ConstantArray[0., Dimensions[gradfshape]]},
				Compile[{{X, _Real, 1}}, c, Evaluate[compOpts]]];
		];
	];
	If[!CompiledFunctionQ[cgradf],
		cgradf = With[{f = nfx}, Compile[{{X, _Real, 1}}, 1.*f["Gradient"[X]],
				{{Experimental`NumericalFunction[__][_] , _Real, 1}}, Evaluate[compOpts]]];
	];

	(* constraint function(s) *)
	cg = ngx["CompiledFunction"];
	If[CompiledFunctionQ[cg],
		If[!TrueQ[gvecvar],
			cg = makeVectorCompiledFunction[cg, compOpts][Sequence@@symsx];
		];
	, (* else *)
		cg = With[{g = ngx}, Compile[{{X, _Real, 1}},  g[X],
			{{Experimental`NumericalFunction[__][_], _Real, 1}}, Evaluate[compOpts]]];
	];

	(* Jacobian of constraints *)
	jacgx = ngx["JacobianExpression"];
	jacgshape = ngx["JacobianShape"];
	If[NumericalFunctionQ[jacgx] && compile,
		If[SparseArrayQ[jacgshape],
			cjacg = jacgx["CompiledFunction"];
			If[CompiledFunctionQ[cjacg],
				If[!TrueQ[gvecvar],
					cjacg = makeVectorCompiledFunction[cjacg, compOpts][Sequence@@symsx];
				];
			, (* else *)
				If[!TrueQ[gvecvar],
					If[nfargs =!= None, jacgvals = jacgx["FunctionExpression"] /. assocg;,
						jacgvals = jacgx["FunctionExpression"][symsx]];
					cjacg = With[{sx = symsx, vals = 1.*jacgvals},
					Compile[{{X, _Real, 1}}, Module[sx, sx = X; vals],
							{{_, _Real, 1}}, Evaluate[compOpts]]];
				, (* else *)
					cjacg = With[{X = ngargs[[1]], gvals = 1.*jacgx["FunctionExpression"]},
						Compile[{{X, _Real, 1}}, gvals,
							{{_, _Real, 1}}, Evaluate[compOpts]]];
				];
			];
			jacgpos = jacgshape["NonzeroPositions"];
		, (* else *)
			If[!TrueQ[gvecvar],
				If[ListQ[jacgx["FunctionExpression"]],
					If[nfargs =!= None, jacgs = SparseArray[jacgx["FunctionExpression"] /. assocg];,
						jacgs = SparseArray[jacgx["FunctionExpression"][symsx]]];
					cjacg = With[{sx = symsx, c = jacgs["NonzeroValues"]},
						Compile[{{X, _Real, 1}}, Module[sx, sx = X; c], Evaluate[compOpts]]];
					jacgpos = jacgs["NonzeroPositions"];
				];
			, (* else *)
				If[CompiledFunctionQ[jacgx["FunctionExpression"]],
					cjacg = jacgx["FunctionExpression"];
					jacgpos = Flatten[Table[{i, j}, {i, glength}, {j, nx}], 1];
				,
					If[ListQ[jacgx["FunctionExpression"]],
						cjacg = With[{X = ngargs[[1]], c = 1.*jacgs["NonzeroValues"]},
						Compile[{{X, _Real, 1}}, c,
							{{_, _Real, 2}}, Evaluate[compOpts]]];
						jacgpos = jacgs["NonzeroPositions"];
					,
						cjacg = With[{X = ngargs[[1]], c = 1.*jacgx["FunctionExpression"]},
							Compile[{{X, _Real, 1}}, Normal[c],
								{{_Normal, _Real, 2}}, Evaluate[compOpts]]];
						jacgpos = Flatten[Table[{i, j}, {i, glength}, {j, nx}], 1];
					];
				];
			];
		];
	, (* else *)
		If[constantg,
			cjacg = Compile[{{X, _Real, 1}}, {}, Evaluate[compOpts]];
		];
		jacgpos = {};
	];
	If[!CompiledFunctionQ[cjacg],
		cjacg = With[{g = ngx}, Compile[{{X, _Real, 1}}, g["Jacobian"[X]],
			{{Experimental`NumericalFunction[__][_] , _Real, 2}}, Evaluate[compOpts]]];
		jacgpos = Flatten[Table[{i, j}, {i, glength}, {j, nx}], 1];
	];

	(* Hessian of objective function *)
	hfx = nfx["HessianExpression"];
	hfshape = nfx["HessianShape"];
	If[NumericalFunctionQ[hfx] && compile,
		If[SparseArrayQ[hfshape],
			If[nfargs =!= None, hfvals = hfx["FunctionExpression"] /. assocf,
				hfvals = hfx["FunctionExpression"][symsx]];
			shf = With[{sh = hfshape, vals = hfvals},
				setSparseValues[sh, vals]];
			hfpos = hfshape["NonzeroPositions"];
		, (* else *)
			chf = hfx["CompiledFunction"];
			If[CompiledFunctionQ[chf],
				chf = makeVectorCompiledFunction[hfx["CompiledFunction"], compOpts][Sequence@@symsx];
				hfpos = Flatten[Table[{i, j}, {i, nx}, {j, nx}], 1];
			];
		];
	, (* else *)
		If[linearf,
			shf = SparseArray[ConstantArray[0, Dimensions[hfshape]]];
		];
	];
	If[!SparseArrayQ[shf] && !CompiledFunctionQ[chf],
		chf = With[{f = nfx}, Compile[{{X, _Real, 1}}, 1.*f["Hessian"[X]],
				{{Experimental`NumericalFunction[__][_] , _Real, 2}}, Evaluate[compOpts]]];
	];
	(* Hessian of constraints *)
	hgx = ngx["HessianExpression"];
	hgshape = ngx["HessianShape"];
	If[NumericalFunctionQ[hgx] && compile,
		If[SparseArrayQ[hgshape],
			If[nfargs =!= None, hgvals = hgx["FunctionExpression"] /. assocg,
				hgvals = hgx["FunctionExpression"][symsx]];
			shg = With[{sh = hgshape, vals = hgvals},
				setSparseValues[sh, vals]];
			hgpos = Rest /@ hgshape["NonzeroPositions"];
		, (* else *)
			chg = hgx["CompiledFunction"];
			If[CompiledFunctionQ[chg],
				If[!TrueQ[gvecvar],
					chg = makeVectorCompiledFunction[chg, compOpts][Sequence@@symsx];
				];
				hgpos = Flatten[Table[{i, j}, {i, nx}, {j, nx}], 1];
			];
		];
	, (* else *)
		If[linearg,
			shg = SparseArray[ConstantArray[0, Dimensions[hgshape]]];
		];
	];
	If[!SparseArrayQ[shg] && !CompiledFunctionQ[chg],
		chg = With[{g = ngx}, Compile[{{X, _Real, 1}}, 1.*g["Hessian"[X]],
				{{Experimental`NumericalFunction[__][_] , _Real, 3}}, Evaluate[compOpts]]];
		hfpos = Flatten[Table[{i, j}, {i, nx}, {j, nx}], 1];
	];
	(* Hessian of the Lagrangian, with an extra s in front of the objective term
	   so that ipopt can obtain the hessian of the objective or the constraints independently *)
	If[SparseArrayQ[shf] && SparseArrayQ[shg],
		hs = LowerTriangularize@SparseArray[1.*(s*shf + symsl.shg)];
		hpos = hs["NonzeroPositions"];
		ch = With[{ss = {s}, sx = symsx, sl = symsl,
			allsyms = Join[{s}, symsx, symsl], c = 1.*hs["NonzeroValues"]},
			Compile[{{S, _Real, 1}, {X, _Real, 1}, {L, _Real, 1}},
				Module[allsyms, ss = S; sx = X; sl = L; c], Evaluate[compOpts]]];
	];
	If[!CompiledFunctionQ[ch],
		If[!CompiledFunctionQ[chf],
			chf = With[{sx = symsx, c = Normal[shf]},
					Compile[{{X, _Real, 1}}, Module[sx, sx = X; c], Evaluate[compOpts]]];
		];
		If[!CompiledFunctionQ[chg],
			chg = With[{sx = symsx, c = Normal[shg]},
					Compile[{{X, _Real, 1}}, Module[sx, sx = X; c], Evaluate[compOpts]]];
		];
		If[CompiledFunctionQ[chf] && CompiledFunctionQ[chg],
			ch = With[{hf = chf, hg = chg, lte = LowerTriangularElements},
				Compile[{{S, _Real, 1}, {X, _Real, 1}, {L, _Real, 1}},
				Module[{s = Compile`GetElement[S, 1]}, lte[1.*(s*hf[X] + L.hg[X])]],
					 {{LowerTriangularElements[_] , _Real, 1}}, Evaluate[compOpts]]];
			hpos = Flatten[Table[{i, j}, {i, nx}, {j, i}], 1];
		];
	];

	(* The monitor function will be called once per iteration.
	   We wrap evalmonitor to make sure there are no return values *)
	cmon = With[{
		xargs = Map[Pattern[#, Blank[]] &, symsx]},
		monitor[xargs]:= Null;
		Compile[{{X, _Real, 1}}, nfx["EvaluateStepMonitor"[X]]; monitor[X]]];

	(* Check that none of the compiled functions are complex valued *)
	Quiet[
		If[Needs["CompiledFunctionTools`"] =!= $Failed,
			cproc = CompiledFunctionTools`ToCompiledProcedure/@ {cf, cgradf, cg, cjacg, ch};
			cres = Cases[cproc, _CompiledFunctionTools`CompiledResult, {2}];
		]
	];
	If[!FreeQ[cres, Complex],
		Message[mhead::complex];
		Throw[$Failed, "IPOPTTag"];
	];

	(* Connect the callback functions *)
	If[!ConnectLibraryCallbackFunction["objective_function_callback_manager", cf],
		Message[IPOPTMinimize::connectobjfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	If[!ConnectLibraryCallbackFunction["gradient_function_callback_manager", cgradf],
		Message[IPOPTMinimize::connectgradofobjfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	If[!ConnectLibraryCallbackFunction["constraint_function_callback_manager", cg],
		Message[IPOPTMinimize::connectconstrfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	If[!ConnectLibraryCallbackFunction["jacobian_function_callback_manager", cjacg],
		Message[IPOPTMinimize::connectjacofgradofobjfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	If[!ConnectLibraryCallbackFunction["hessian_function_callback_manager", ch],
		Message[IPOPTMinimize::connecthessfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	If[!ConnectLibraryCallbackFunction["monitor_function_callback_manager", cmon],
		Message[IPOPTMinimize::connectmonitorfunfail];
		Throw[$Failed, "IPOPTTag"];
	];

	{jacgpos, hpos}
];

makeVectorCompiledFunction[f_, opts_][vars__] :=
Module[{X},
	Compile[{{X, _Real, 1}}, Module[{vars}, {vars} = X; 1.*f[vars]],
		CompilationOptions -> {"InlineCompiledFunctions" -> False}, Evaluate[opts]]
];

makeCompiledObjectiveFunction[f_, opts_] :=
 Module[{X}, Compile[{{X, _Real, 1}}, {1.*f[X]}, Evaluate[opts]]];

makeVectorCompiledObjectiveFunction[f_, opts_][vars__] :=
Module[{X},
	Compile[{{X, _Real, 1}}, Module[{vars}, {vars} = X; {1.*f[vars]}],
		CompilationOptions -> {"InlineCompiledFunctions" -> False}, Evaluate[opts]]
];

LowerTriangularElements[m_List] :=
Module[{n, pos},
	n = Length[m];
	pos = Flatten[Table[{i, j}, {i, n}, {j, i}], 1];
	Extract[m, pos]
];


IPOPTProcessOptions[opts_, gx0_, gbounds0_, bounds0_, fnname_Symbol]:=
Module[
	{ipOpts, inf, minf, gx, bounds, gbounds, stepmonitor, compiled, runtopts, errhandler, warnmsgs, mhead},

	ipOpts = OptionValue[IPOPTMinimize, opts, "IPOPTOptions"];
	If[!ListQ[ipOpts], ipOpts = List[ipOpts]];

	(* replace Infinity with IPOPT's effective infinity *)
	If[FreeQ[ipOpts, "nlp_lower_bound_inf"],
		ipOpts = Join[ipOpts, {"nlp_lower_bound_inf" -> -$EFFECTIVEINFINITY}]];
	minf = OptionValue[ipOpts, "nlp_lower_bound_inf"];
	If[FreeQ[ipOpts, "nlp_upper_bound_inf"],
		ipOpts = Join[ipOpts, {"nlp_upper_bound_inf" -> $EFFECTIVEINFINITY}]];
	inf = OptionValue[ipOpts, "nlp_upper_bound_inf"];
	gx = gx0 /. {-Infinity -> minf, Infinity -> inf};
	bounds = bounds0 /. {-Infinity -> minf, Infinity -> inf};
	gbounds = gbounds0 /. {-Infinity -> minf, Infinity -> inf};

	stepmonitor = OptionValue[IPOPTMinimize, opts, StepMonitor, Hold];

	If[SameQ[fnname, IPOPTMinimize],
		compiled = OptionValue[IPOPTMinimize, opts, "Compiled"]];

	runtopts = OptionValue[IPOPTMinimize, opts, "RuntimeOptions"];
	If[SameQ[ToString[runtopts], "Automatic"],
		errhandler = Automatic;
		warnmsgs = True;
	,
		If[ListQ[runtopts] && OptionQ[runtopts],
			If[Complement[runtopts[[All, 1]], {"RuntimeErrorHandler", "WarningMessages"}] =!= {},
				Message[fnname::"rntopts"];
			];
			errhandler = OptionValue[Append[runtopts, "RuntimeErrorHandler"-> Automatic], "RuntimeErrorHandler"];
			warnmsgs = OptionValue[Append[runtopts, "WarningMessages"-> True], "WarningMessages"];
			If[!MemberQ[{Automatic, "ReturnObject"}, errhandler],
				Message[fnname::"errhnd", errhandler];
				errhandler = Automatic;
			];
			If[!BooleanQ[warnmsgs],
				Message[fnname::"warnm", warnmsgs];
				warnmsgs = True;
			];
		,
			Message[fnname::"rntoptval"];
			errhandler = Automatic;
			warnmsgs = True;
		];
	];

	mhead = OptionValue[IPOPTMinimize, opts, "MessageHead"];

	{ipOpts, gx, gbounds, bounds, stepmonitor, compiled, errhandler, warnmsgs, mhead}
]

SetAttributes[IPOPTMinimizeCheckInputs, HoldAll];
IPOPTMinimizeCheckInputs[fx_, heldgx_, gbounds0_, xvars0_, xbounds0_, x00_]:=
Module[{x0 = Normal[x00], nvars, nx0, xdims, xbounds = xbounds0, xboundsdim,
	gx, gboundsdim, glength, g1, headg, gargs, gbounds = gbounds0,
	rule, headf, fargs, symsx, assoc, xvars = xvars0, nfargs, ngargs, nfdims, ngdims},

	If[xvars === None,
		gx = ReleaseHold[heldgx];
		If[!(NumericalFunctionQ[fx] && NumericalFunctionQ[gx]),
			Message[IPOPTMinimize::varsnone];
			Return[$Failed];
		];
		nx0 = Length[x0];
		nfargs = fx["ArgumentNames"];
		nfdims = fx["ArgumentDimensions"];
		ngargs = gx["ArgumentNames"];
		ngdims = gx["ArgumentDimensions"];
		Switch[Length[nfargs],
			0, If[nfdims =!= {{nx0}} && !(Flatten[nfdims] === {} && Length[nfdims] === nx0),
				Message[IPOPTMinimize::varsnone]; Return[$Failed]];,
			1,  If[!(nfdims === {{nx0}} || (nx0 === 1 && nfdims === {{}})),
				Message[IPOPTMinimize::varsnone]; Return[$Failed]];,
			nx0, If[Flatten[nfdims] =!= {},
				Message[IPOPTMinimize::varsnone]; Return[$Failed]];,
			_, Return[$Failed];
		];
		Switch[Length[ngargs],
			0, If[ngdims =!= {{nx0}} && !(Flatten[ngdims] === {} && Length[ngdims] === nx0),
				Message[IPOPTMinimize::varsnone]; Return[$Failed]];,
			1, If[!(ngdims === {{nx0}} || (nx0 === 1 && ngdims === {{}})),
				Message[IPOPTMinimize::varsnone]; Return[$Failed]];,
			nx0, If[Flatten[ngdims] =!= {},
				Message[IPOPTMinimize::varsnone]; Return[$Failed]];,
			_, Return[$Failed];
		];
		nvars = nx0;
	,
		xdims = Dimensions[xvars];

		If[Length[xdims]=!=1,
			Message[IPOPTMinimize::varslen, xvars];
			Throw[$Failed, "IPOPTTag"];
		];
		nvars = Length[xvars];
		If[!SameQ[nvars, Length[x0]],
			Message[IPOPTMinimize::initptlen, x0, xvars];
			Throw[$Failed, "IPOPTTag"];
		];

		rule = MapThread[Rule, {xvars, x0}];
	];

	If[
		! And @@ Map[NumericQ, x0],
		Message[IPOPTMinimize::badinitpt];
		Throw[$Failed, "IPOPTTag"];
	];

	If[xbounds === {},
		xbounds = Table[{-$EFFECTIVEINFINITY, $EFFECTIVEINFINITY}, nvars];
	];

	xboundsdim = Dimensions[xbounds];

	If[Length[xboundsdim]=!=2,
		Message[IPOPTMinimize::bounds];
		Throw[$Failed, "IPOPTTag"];
	];

	If[
		! SameQ[xboundsdim[[1]], nvars, Length[x0]],
		Message[IPOPTMinimize::boundlen];
		Throw[$Failed, "IPOPTTag"];
	];

	If[
		xboundsdim[[2]] =!= 2,
		Message[IPOPTMinimize::bounddim];
		Throw[$Failed, "IPOPTTag"];
	];

	gboundsdim = Dimensions[gbounds];

	If[Length[gboundsdim]=!=2,
		Message[IPOPTMinimize::gbounds];
		Throw[$Failed, "IPOPTTag"];
	];

	headg = Part[heldgx, 1, 0];
	If[NumericalFunctionQ[headg],
		glength = headg["ResultDimensions"][[1]];
	,(* else *)
		gx = ReleaseHold[heldgx];
		If[NumericalFunctionQ[gx],
			glength = gx["ResultDimensions"][[1]];
			, (* else *)
			If[ListQ[gx],
				If[gx === {},
					If[gbounds === {},
						gx = {1};
						heldgx = Hold[gx];
						gbounds = {{-$EFFECTIVEINFINITY, $EFFECTIVEINFINITY}};
					,
						Message[IPOPTMinimize::gboundlen];
						Throw[$Failed, "IPOPTTag"];
					];
				];
				glength = Length[gx];
			, (* else *)
				If[ListQ[gx /. rule],
					glength = Length[gx /. rule];
				, (* else *)
					Message[IPOPTMinimize::listcon];
					Throw[$Failed, "IPOPTTag"];
				];
			];
		];
	];

	If[
		gboundsdim[[1]] =!= glength,
		Message[IPOPTMinimize::gboundlen];
		Throw[$Failed, "IPOPTTag"];
	];

	If[
		gboundsdim[[2]] =!= 2,
		Message[IPOPTMinimize::gbounddim];
		Throw[$Failed, "IPOPTTag"];
	];

	If[
		! And @@ Map[NumericQ, Flatten[gbounds]],
		Message[IPOPTMinimize::gboundnum];
		Throw[$Failed, "IPOPTTag"];
	];

	If[
		! And @@ Map[NumericQ, Flatten[xbounds]],
		Message[IPOPTMinimize::boundnum];
		Throw[$Failed, "IPOPTTag"];
	];

	headf = Part[Hold[fx], 1, 0];
	If[NumericalFunctionQ[headf],
		fargs = Cases[Hold[fx], HoldPattern[_[args__]] :> args, {1}];
		If[Length[fargs] =!= 1 || fargs[[1]] =!= xvars,
			Message[IPOPTMinimize::badobjvars];
			Throw[$Failed, "IPOPTTag"];
		];
		If[!Internal`RealValuedNumericQ[headf[x0]],
			Message[IPOPTMinimize::badobj];
			Throw[$Failed, "IPOPTTag"];
		];
	, (* else *)
		If[NumericalFunctionQ[fx],
			If[!Internal`RealValuedNumericQ[fx[x0]],
				Message[IPOPTMinimize::badobj];
				Throw[$Failed, "IPOPTTag"];
			];
		, (* else *)
			If[!Internal`RealValuedNumericQ[fx /. rule],
				Message[IPOPTMinimize::badobj];
				Throw[$Failed, "IPOPTTag"];
			];
		];
	];

	If[NumericalFunctionQ[headg],
		gargs = Cases[heldgx, HoldPattern[_[args__]] :> args, {1}];
		If[Length[gargs] =!= 1 || gargs[[1]] =!= xvars,
			Message[IPOPTMinimize::badconstrvars];
			Throw[$Failed, "IPOPTTag"];
		];
		If[!(And @@ Map[NumericQ, headg[x0]]),
			Message[IPOPTMinimize::badconstr];
			Throw[$Failed, "IPOPTTag"];
		];
	, (* else *)
		If[NumericalFunctionQ[gx],
			If[!(And @@ Map[NumericQ, gx[x0]]),
				Message[IPOPTMinimize::badconstr];
				Throw[$Failed, "IPOPTTag"];
			];
		, (* else *)
			If[!FreeQ[gx, Experimental`NumericalFunction],
				(* rename any numerical function argument names that coinside with vars
				before we try gx /. rule, or it reaches inside the NF *)
				symsx = Table[Unique["x", Temporary], {nvars}];
				assoc = AssociationThread[xvars -> symsx];
				g1 = gx /. (nf_Experimental`NumericalFunction :> (nf /. assoc));
			, (* else *)
				g1 = gx;
			];
			If[!(And @@ Map[NumericQ, g1 /. rule]),
				Message[IPOPTMinimize::badconstr];
				Throw[$Failed, "IPOPTTag"];
			];
		];
	];

	{N[x0], N[xbounds], heldgx, N[gbounds]}
];


(*
ParametricIPOPTMinimize:
*)

(* ParametricIPOPTData expression (and IPOPTCallbackDataMap) related functions: *)

ParametricIPOPTDataID[e_ParametricIPOPTData] := ManagedLibraryExpressionID[e, "IPOPTCallbackData_instance_manager"];

ParametricIPOPTDataQ[e_ParametricIPOPTData] := ManagedLibraryExpressionQ[e, "IPOPTCallbackData_instance_manager"];

testParametricIPOPTData[][e_] := testParametricIPOPTData[ParametricIPOPTData][e];
testParametricIPOPTData[mhead_Symbol][e_] :=
If[TrueQ[ParametricIPOPTDataQ[e]],
	True,
	Message[MessageName[mhead, "paripoptinst"], e]; False
];
testParametricIPOPTData[_][e_] := TrueQ[ParametricIPOPTDataQ[e]];

General::paripoptinst = "`1` does not represent an active ParametricIPOPTData object.";

ParametricIPOPTDataDelete[ParametricIPOPTData[id_]?(testParametricIPOPTData[ParametricIPOPTDataDelete])] :=
Module[{},
	IPOPTCallbackDataDelete0[id]
];
ParametricIPOPTDataDelete[l:{_ParametricIPOPTData..}] := ParametricIPOPTDataDelete /@ l;

ParametricIPOPTDataExpressions[] :=
Module[{list},
	If[needInitialization, LoadIPOPTLink[]];
	list = IPOPTCallbackDataIDList[];
	If[!ListQ[list],
	   $Failed,
	   Map[ParametricIPOPTData, list]]
]

ParametricIPOPTDataCreate[soldata_, params_] :=
	Module[{ParametricDataFun},
		If[needInitialization, LoadIPOPTLink[]];
		ParametricDataFun = With[{sd = soldata, p = params}, Function[{id}, ParametricIPOPTData[id, sd, p]]];
		CreateManagedLibraryExpression["IPOPTCallbackData_instance_manager", ParametricDataFun]
	];


(* ParametricIPOPTMinimize *)

Options[ParametricIPOPTMinimize] = {StepMonitor -> None, "IPOPTOptions" -> {},
	"MessageHead" -> ParametricIPOPTMinimize, "RuntimeOptions" -> Automatic}

ParametricIPOPTMinimize[fx_, xvars_, x0_, xbounds0_, gx0_, gbounds0_, params_, opts : OptionsPattern[]] :=
Module[{gx = gx0, gbounds, xbounds, stepmonitor, ipOpts, pardata, temp,
	gradfx, jacgx, jacgpos, l, lvars, s, hx, hpos, soldata, compiled, errhandler, warnmsgs, mhead},

	temp = IPOPTProcessOptions[{opts}, gx, gbounds0, xbounds0, ParametricIPOPTMinimize];
	If[ListQ[temp] && Length[temp] === 9,
		{ipOpts, gx, gbounds, xbounds, stepmonitor, compiled, errhandler, warnmsgs, mhead} = temp,
		Return[$Failed]
	];

	temp = Catch[
		CheckInputsWithParameters[fx, gx, gbounds, xvars, xbounds, x0]
		, "IPOPTTag"];
	If[ListQ[temp] && Length[temp] === 3,
		{xbounds, gx, gbounds} = temp,
		Return[$Failed]
	];

	temp = ConstructDerivatives[fx, gx, xvars, l, s];
	If[ListQ[temp] && Length[temp] === 6,
		{gradfx, jacgx, jacgpos, lvars, hx, hpos} = temp,
		Return[$Failed]
	];

	If[Catch[
		CompileParametricFunctions[fx, gx, gradfx, jacgx, hx, stepmonitor, xvars, params, lvars, s, compiled],
		"IPOPTTag"] === $Failed,
		Return[$Failed]
	];

	soldata = {fx, gx, gbounds, xvars, xbounds, x0, jacgpos, hpos, ipOpts, errhandler, warnmsgs, mhead};

	(* In C++ creates an IPOPTCallbackData instance with the connected callback (compiled) functions
	   and here includes any other data to be used for the solver, awaiting parameter values *)
	pardata = ParametricIPOPTDataCreate[soldata, params];

	(* the 4th argument of ParametricPlugInFunction contains information necessary for the 
	   ParametricFunction blob and a 'pardata' reference to prevent it from getting deleted *)
	With[{pd = pardata},
		NDSolve`ParametricPlugInFunction[pd, params, {},
			HoldForm[IPOPTLink`ParametricIPOPTMinimize[fx, xvars, pd]]]]
];


ParametricIPOPTData[dataID_Integer, soldata_List, params_List][pvals_List]:=
Block[params, params = pvals;
	Module[{fx, gx, gbounds, xvars, bounds, x0, jacgpos, hpos, ipOpts,
		lbounds, ubounds, lgbounds, ugbounds, solInstance, solID, err, mhead, temp, errargs, errhandler, warnmsgs},

		{fx, gx, gbounds, xvars, bounds, x0, jacgpos, hpos, ipOpts, errhandler, warnmsgs, mhead} = soldata;

		If[Catch[
			CheckInputsWithParameterValues[fx, gx, gbounds, xvars, bounds, x0],
			"IPOPTTag"] === $Failed,
			Return[$Failed]
		];

		{lbounds, ubounds} = Transpose[bounds];
		{lgbounds, ugbounds} = Transpose[gbounds];

		solInstance = IPOPTDataCreate[];
		solID = IPOPTDataID[solInstance];

		Quiet[
			temp = IPOPTSolve[solID, dataID, jacgpos, hpos, x0, lbounds, ubounds, lgbounds, ugbounds, params, ipOpts],
			"Compiler"
		];
		If[ListQ[temp] && Length[temp] === 2,
			{err, errargs} = temp,
			Return[$Failed]
		];

		If[TrueQ[err === 0], solInstance, IPOPTErrorCodeHandler[err, errargs, mhead, solInstance, errhandler, warnmsgs]]
	]
];


ConstructDerivatives[fx_, gx_, xvars_, l_, s_] :=
Module[{gradfx, jacgx, jacgpos, ddfx, ddgx, lvars, hx, hpos},
	gradfx = D[fx, {xvars}];
	jacgx = SparseArray[D[gx, {xvars}]];
	jacgpos = jacgx["NonzeroPositions"];
	ddfx = SparseArray[D[gradfx, {xvars}]];
	ddgx = SparseArray[D[jacgx, {xvars}]];
	lvars = Array[l, Length[gx]];
	hx = LowerTriangularize@SparseArray[s*ddfx + lvars.ddgx];
	hpos = hx["NonzeroPositions"];
	{gradfx, jacgx, jacgpos, lvars, hx, hpos}
];


CompileParametricFunctions[fx_, gx_, gradfx_, jacgx_, hx_, stepmonitor_, xvars_, params_, lvars_, s_, compiled_] :=
Module[{cf, cg, cgradf, cjacg, ch, cmon, symsx, symsl, assoc, assoc2, symsp, symsxp, monitor, compOpts},
	(* compile and connect library callback functions *)
	If[needInitialization, LoadIPOPTLink[]];

	If[TrueQ[ListQ[compiled] && Length[compiled] >= 2],
		compOpts = Rest[compiled];
	,
		compOpts = {};
	];

	(* objective function *)
	symsx = Table[Unique["x", Temporary], {Length[xvars]}];
	symsp = Table[Unique["p", Temporary], {Length[params]}];
	symsxp = Join[symsx, symsp];
	assoc = Join[AssociationThread[xvars -> symsx], AssociationThread[params -> symsp]];
	cf = With[{sx = symsx, sp = symsp, sall = symsxp, c = {1.*fx} /. assoc},
		Compile[{{X, _Real, 1}, {P, _Real, 1}}, Module[sall, sx = X; sp = P; c], Evaluate[compOpts]]];
	If[!ConnectLibraryCallbackFunction["objective_function_callback_manager", cf],
		Message[ParametricIPOPTMinimize::connectobjfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	(* constraint function *)
	cg = With[{sx = symsx, sp = symsp, sall = symsxp, c = 1.*gx /. assoc},
		Compile[{{X, _Real, 1}, {P, _Real, 1}}, Module[sall, sx = X; sp = P; c], Evaluate[compOpts]]];
	If[!ConnectLibraryCallbackFunction["constraint_function_callback_manager", cg],
		Message[ParametricIPOPTMinimize::connectconstrfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	(* gradient of objective function *)
	cgradf = With[{sx = symsx, sp = symsp, sall = symsxp, c = 1.*gradfx /. assoc},
		Compile[{{X, _Real, 1}, {P, _Real, 1}}, Module[sall, sx = X; sp = P; c], Evaluate[compOpts]]];
	If[!ConnectLibraryCallbackFunction["gradient_function_callback_manager", cgradf],
		Message[ParametricIPOPTMinimize::connectgradofobjfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	(* jacobian of constraint function *)
	cjacg = With[{sx = symsx, sp = symsp, sall = symsxp, c = 1.*jacgx["NonzeroValues"] /. assoc},
		Compile[{{X, _Real, 1}, {P, _Real, 1}}, Module[sall, sx = X; sp = P; c], Evaluate[compOpts]]];
	If[!ConnectLibraryCallbackFunction["jacobian_function_callback_manager", cjacg],
		Message[ParametricIPOPTMinimize::connectjacofgradofobjfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	(* hessian of the lagrangian, with an extra s in front of the objective term
	   so that ipopt can obtain the hessian of the objective or the constraints independently *)
	symsl = Table[Unique["l", Temporary], {Length[lvars]}];
	assoc2 = Join[AssociationThread[xvars -> symsx], AssociationThread[lvars -> symsl],
		AssociationThread[params -> symsp]];
	ch = With[{ss = {s}, sx = symsx, sl = symsl, sp = symsp, sall = Join[{s}, symsx, symsl, symsp],
		c = 1.*hx["NonzeroValues"] /. assoc2},
		Compile[{{S, _Real, 1}, {X, _Real, 1}, {L, _Real, 1}, {P, _Real, 1}},
			Module[sall, ss = S; sx = X; sl = L; sp = P; c], Evaluate[compOpts]]];
	If[!ConnectLibraryCallbackFunction["hessian_function_callback_manager", ch],
		Message[ParametricIPOPTMinimize::connecthessfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
	(* This function will be called once per iteration.
	   We wrap evalmonitor to make sure there are no return values *)
	cmon = With[{mon = stepmonitor /. assoc,
		xargs = Map[Pattern[#, Blank[]] &, symsx],
		pargs = Map[Pattern[#, Blank[]] &, symsp]},
		monitor[xargs, pargs]:= (ReleaseHold[mon]; Null);
		Compile[{{X, _Real, 1}, {P, _Real, 1}}, monitor[X, P], Evaluate[compOpts]]];
	If[!ConnectLibraryCallbackFunction["monitor_function_callback_manager", cmon],
		Message[ParametricIPOPTMinimize::connectmonitorfunfail];
		Throw[$Failed, "IPOPTTag"];
	];
]


(* Initial checks for ParametricIPOPTMinimize inputs before we have parameter values *)
SetAttributes[CheckInputsWithParameters, HoldAll];
CheckInputsWithParameters[fx_, gx0_, gbounds0_, xvars_, xbounds0_, x0_]:=
Module[{xdim, xboundsdim, gboundsdim, nvars,
	xbounds = xbounds0, gx = gx0, gbounds = gbounds0},

	xdim = Dimensions[xvars];
	If[Length[xdim]=!=1,
		Message[IPOPTMinimize::varslen, xvars];
		Throw[$Failed, "IPOPTTag"];
	];

	nvars = Length[xvars];
	If[!SameQ[nvars, Length[x0]],
		Message[IPOPTMinimize::initptlen, x0, xvars];
		Throw[$Failed, "IPOPTTag"];
	];

	If[xbounds === {},
		xbounds = Table[{-$EFFECTIVEINFINITY, $EFFECTIVEINFINITY}, nvars];
	];
	xboundsdim = Dimensions[xbounds];

	If[Length[xboundsdim]=!=2,
		Message[ParametricIPOPTMinimize::bounds];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		! SameQ[xboundsdim[[1]], nvars, Length[x0]],
		Message[ParametricIPOPTMinimize::boundlen];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		xboundsdim[[2]] =!= 2,
		Message[ParametricIPOPTMinimize::bounddim];
		Throw[$Failed, "IPOPTTag"];
	];

	If[gx === {},
		If[gbounds === {},
			gx = {1};
			gbounds = {{-$EFFECTIVEINFINITY, $EFFECTIVEINFINITY}};
		,
			Message[IPOPTMinimize::gboundlen];
			Throw[$Failed, "IPOPTTag"];
		];
	];
	gboundsdim = Dimensions[gbounds];
	If[Length[gboundsdim]=!=2,
		Message[ParametricIPOPTMinimize::gbounds];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		gboundsdim[[1]] =!= Length[gx],
		Message[ParametricIPOPTMinimize::gboundlen];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		gboundsdim[[2]] =!= 2,
		Message[ParametricIPOPTMinimize::gbounddim];
		Throw[$Failed, "IPOPTTag"];
	];

	{xbounds, gx, gbounds}
];

(* More checks for the inputs after we get parameter values *)
SetAttributes[CheckInputsWithParameterValues, HoldAll];
CheckInputsWithParameterValues[fx_, gx_, gbounds_, xvars_, xbounds_, x0_]:=
Module[{xrule},
	If[
		! And @@ Map[NumericQ, x0],
		Message[ParametricIPOPTMinimize::badinitpt];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		! And @@ Map[NumericQ, Flatten[gbounds]],
		Message[ParametricIPOPTMinimize::gboundnum];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		! And @@ Map[NumericQ, Flatten[xbounds]],
		Message[ParametricIPOPTMinimize::boundnum];
		Throw[$Failed, "IPOPTTag"];
	];
	xrule = MapThread[Rule, {xvars, x0}];
	If[
		!Internal`RealValuedNumericQ[fx /. xrule],
		Message[ParametricIPOPTMinimize::badobj];
		Throw[$Failed, "IPOPTTag"];
	];
	If[
		! And @@ Map[NumericQ, gx /. xrule],
		Message[ParametricIPOPTMinimize::badconstr];
		Throw[$Failed, "IPOPTTag"];
	];
];


GetIPOPTErrorArguments[appID_, err_]:=
Switch[err,
	-1,
		{getIPOPTAppIntegerOption[appID, "max_iter"]},
	_,
		None
];


IPOPTErrorCodeHandler[err_, args_, mhead_Symbol, solInstance_, errhandler_, warnmsgs_]:=
	Switch[err,
	1,
		If[TrueQ[warnmsgs], Message[mhead::acceptlev]];
		solInstance,
	2,
		If[TrueQ[warnmsgs], Message[mhead::infeas]];
		solInstance,
	3,
		If[TrueQ[warnmsgs], Message[mhead::sdir]];
		solInstance,
	4,
		If[TrueQ[warnmsgs], Message[mhead::divit]];
		solInstance,
	5,
		If[TrueQ[warnmsgs], Message[mhead::ustop]];
		solInstance,
	6,
		If[TrueQ[warnmsgs], Message[mhead::feaspt]];
		solInstance,
	-1,
		If[TrueQ[warnmsgs], Message[mhead::cvmit, args[[1]]]];
		solInstance,
	-2,
		If[TrueQ[warnmsgs], Message[mhead::restor]];
		solInstance,
	-3,
		If[TrueQ[warnmsgs], Message[mhead::steperr]];
		solInstance,
	-4,
		If[TrueQ[warnmsgs], Message[mhead::maxcpu]];
		solInstance,
	-10,
		If[TrueQ[warnmsgs], Message[mhead::baddof]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-11,
		If[TrueQ[warnmsgs], Message[mhead::badprob]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-12,
		If[TrueQ[warnmsgs], Message[mhead::badopt]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-13,
		If[TrueQ[warnmsgs], Message[mhead::badnum]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-100,
		If[TrueQ[warnmsgs], Message[mhead::ipopterr]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-101,
		If[TrueQ[warnmsgs], Message[mhead::nonipopterr]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-102,
		If[TrueQ[warnmsgs], Message[mhead::outofmem]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	-199,
		If[TrueQ[warnmsgs], Message[mhead::intipopterr]];
		If[SameQ[errhandler, "ReturnObject"],
			solInstance,
			$Failed],
	_,
		$Failed
	];


IPOPTStringStatus[data_] :=
Module[{string, errcode = IPOPTReturnCode[data]},
	string = Switch[errcode,
		0, "Solved",
		1, "AcceptableToleranceReached",
		2, "Infeasible",
		3, "SmallDirection",
		4, "DivergingIterates",
		5, "Interrupted",
		6, "FeasiblePointFound",
		-1, "MaxIterationsReached",
		-2, "RestorationError",
		-3, "StepComputationError",
		-4, "MaxCPUTimeReached",
		-10, "InsufficientDegreesOfFreedom",
		-11, "InvalidProblemDefinition",
		-12, "InvalidOption",
		-13, "InvalidNumber",
		-100, "IPOPTException",
		-101, "NonIPOPTError",
		-102, "OutOfMemory",
		-199, "InternalIPOPTError",
		_, "UnknownStatus"
	]
];

(*
	Message tesxts
*)
(* Messages for inputs*)
IPOPTMinimize::varslen = ParametricIPOPTMinimize::varslen = "The variables `1` should be given in a list."
IPOPTMinimize::varsnone = "If the variables are given as None, the objective and constraint should be a NumericalFunction object that evaluates for vectors with the same length as the initial point."
IPOPTMinimize::initptlen = ParametricIPOPTMinimize::initptlen = "The initial point `1` should be a list of the same length as the variables `2`.";
IPOPTMinimize::badinitpt = "Invalid initial point. The initial point should be numeric.";
ParametricIPOPTMinimize::badinitpt = "Invalid initial point. The initial point is not numeric with the given parameter values.";
IPOPTMinimize::bounds = ParametricIPOPTMinimize::bounds = "Invalid variable bounds.";
IPOPTMinimize::boundlen = ParametricIPOPTMinimize::boundlen = "Invalid bounds, each variable should have one set of bounds.";
IPOPTMinimize::bounddim = ParametricIPOPTMinimize::bounddim = "Invalid bounds, the bounds for each variable should be a pair of numbers.";
IPOPTMinimize::gbounds = ParametricIPOPTMinimize::gbounds = "Invalid constraint bounds.";
IPOPTMinimize::gboundlen = ParametricIPOPTMinimize::gboundlen = "Invalid constraint bounds. Each constraint should have one set of bounds.";
IPOPTMinimize::gbounddim = ParametricIPOPTMinimize::gbounddim = "Invalid constraint bounds. Each constraint bound should be a pair of numbers.";
IPOPTMinimize::gboundnum = "Invalid constraint bounds. Bounds should be numeric.";
ParametricIPOPTMinimize::gboundnum = "Invalid constraint bounds. The bounds are not numeric with the given parameter values.";
IPOPTMinimize::boundnum = "Invalid bounds. Bounds should be numeric.";
ParametricIPOPTMinimize::boundnum = "Invalid bounds. Bounds of g is not numeric with the given parameter values.";
IPOPTMinimize::badobj = "Invalid objective function. The objective function doesn't evaluate to a real-valued numeric result at the initial point.";
ParametricIPOPTMinimize::badobj = "Invalid objective function. The objective function doesn't evaluate to a numeric result at the initial point with the given parameter values.";
IPOPTMinimize::badconstr = "Invalid constraints. The constraint function doesn't evaluate to a numeric result.";
ParametricIPOPTMinimize::badconstr = "Invalid constraints. The constraint function doesn't evaluate to a numeric result at the initial point with the given parameter values.";
IPOPTMinimize::badobjvars = "The variables of the objective numerical function are not the same as the problem variables.";
IPOPTMinimize::badconstrvars = "The variables of the constraints numerical function are not the same as the problem variables.";
IPOPTMinimize::listcon = "The constraint functions do not form a list."
IPOPTMinimize::invalidopt = "Possible IPOPTOptions can be found at http://www.coin-or.org/Ipopt/documentation/node39.html. The option name should be given as a string and the option value should be a string, integer or a real value according to the option documentation.";
IPOPTMinimize::rntoptval = ParametricIPOPTMinimize::rntoptval = "The value of RuntimeOptions should be Automatic or a list of options."
IPOPTMinimize::rntopts = ParametricIPOPTMinimize::rntopts = "Possible RuntimeOptions are \"RuntimeErrorHandler\" and \"WarningMessages\"."
IPOPTMinimize::errhnd = ParametricIPOPTMinimize::errhnd = "The value `1` of the option \"RuntimeErrorHandler\" is not one of Automatic and \"ReturnObject\". Switching to Automatic.";
IPOPTMinimize::warnm = ParametricIPOPTMinimize::warnm = "The value `1` of the option \"WarningMessages\" is not True or False. Switching to True.";
(* Messages for compiled functions *)
IPOPTMinimize::connectobjfunfail = ParametricIPOPTMinimize::connectobjfunfail = "Failed to connect objective function to library.";
IPOPTMinimize::connectconstrfunfail = ParametricIPOPTMinimize::connectconstrfunfail = "Failed to connect constraint function to library.";
IPOPTMinimize::connectgradofobjfunfail = ParametricIPOPTMinimize::connectgradofobjfunfail = "Failed to connect gradient of objective funciton to library.";
IPOPTMinimize::connectjacofgradofobjfunfail = ParametricIPOPTMinimize::connectjacofgradofobjfunfail = "Failed to connect jacobian function to library.";
IPOPTMinimize::connecthessfunfail = ParametricIPOPTMinimize::connecthessfunfail = "Failed to connect hessian function to library.";
IPOPTMinimize::connectmonitorfunfail = ParametricIPOPTMinimize::connectmonitorfunfail = "Failed to connect monitor function to library.";
(* Messages from IPOPT error codes *)
General::acceptlev = "Solved to acceptable level."
FindMinimum::infeas = FindMaximum::infeas = FindArgMin::infeas = FindArgMax::infeas = FindMinValue::infeas = FindMaxValue::infeas = "Possible infeasibility detected. Returning the best solution found. Setting a different initial point or Method -> InteriorPoint may lead to a better solution."
General::infeas = "Possible infeasibility detected. Returning the best solution found."
General::sdir = "Search direction has become too small."
General::divit = "Detected diverging iterates."
General::ustop = "User requested stop."
General::feaspt = "Feasible point was found."
FindMinimum::restor = FindMaximum::restor = FindArgMin::restor = FindArgMax::restor = FindMinValue::restor = FindMaxValue::restor = "Could not reach an optimal solution that satisfies the constraints. Returning the best solution found. Setting a different initial point or Method -> InteriorPoint may lead to a better solution."
General::restor = "Could not reach an optimal solution that satisfies the constraints. Returning the best solution found."
General::steperr = "Error in step computation."
General::maxcpu = "Maximum CPU time exceeded."
General::baddof = "Not enough degrees of freedom."
General::badprob = "Invalid IPOPT problem definition."
General::badopt = "Invalid IPOPT option."
General::badnum = "Invalid number detected."
General::ipopterr = "Unrecoverable IPOPT exception."
General::nonipopterr = "Non-IPOPT exception thrown."
General::outofmem = "Insufficient memory."
General::intipopterr = "Internal IPOPT error."
(* Complex valued compiled functions *)
General::complex = "All input functions should be real valued."
(* Messages from ParametricFunction *)
ParametricIPOPTMinimize::fpct = "Too many parameters in `1` to be filled from `2`."
ParametricIPOPTMinimize::prange = "Invalid non-numeric value `2` for parameter `1`."

End[]

EndPackage[]
