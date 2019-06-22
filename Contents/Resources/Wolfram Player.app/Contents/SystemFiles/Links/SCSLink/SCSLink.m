(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["SCSLink`"]

$SCSLinkLibrary::usage  = "$SCSLinkLibrary is the full path to the SCSLink library loaded."
$SCSLinkDirectory::usage = "$SCSLinkDirectory gives the location of the IPOPTLink library."
LoadSCSLink::usage  = "LoadSCSLink[] loads the SCSLink library."
SCSSolve::usage = "Solves a SCS problem."
SuperSCSSolve::usage = "Solves a SuperSCS problem."
SCSData::usage = "ManagedLibraryExpression for the result of SCSSolve"
SCSx::usage = ""
SCSy::usage = ""
SCSs::usage = ""
SCSStringStatus::usage = ""
SCSStatusVal::usage = ""
SCSPObj::usage = "Primal objective"
SCSDObj::usage = "Dual objective"
SCSResPri::usage = ""
SCSResDual::usage = ""
SCSResInfeas::usage = ""
SCSResUnbdd::usage = ""
SCSResGap::usage = ""
SCSSetupTime::usage = ""
SCSSolveTime::usage = ""
SCSLink::usage = "Message head"

Begin["`Private`"]

$SCSLinkDirectory = DirectoryName[$InputFileName];
$targetDir = FileNameJoin[{$SCSLinkDirectory, "LibraryResources", $SystemID}]
$SCSLinkLibrary = Block[{$LibraryPath = $targetDir}, FindLibrary["SCSLink"]];

(*
 Load all the functions from the SCSLink library
*)

needInitialization = True;

LoadSCSLink[] :=
Block[{$LibraryPath = $targetDir},
	SCSSolve0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSLinkSolve", {Integer, {Real, 1}, {Real, 1}, 
		LibraryDataType[SparseArray, Real, 2], Integer, Integer, {Integer, 1}, {Integer, 1}, Integer, Integer,
		{Real, 1}, Integer, Real, Real, Integer, Integer, Real, Real, Real, Integer, Integer,
		Integer, Integer, Integer, Integer, Real, Real, Real, Integer, Real, Real, UTF8String, Real,
		Integer, Integer, Integer, Integer}, Integer];
	(* SCSData ManagesLibraryExpression *)
	SCSDataDelete0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_delete", {Integer}, Integer];
	SCSDataIDList = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_retIDList", {}, {Integer, 1}];
	(* selectors *)
	SCSx0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_x", {Integer}, {Real, 1}];
	SCSy0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_y", {Integer}, {Real, 1}];
	SCSs0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_s", {Integer}, {Real, 1}];
	SCSiter0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_iter", {Integer}, Integer];
	SCSStringStatus0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_status", {Integer}, UTF8String];
	SCSStatusVal0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_statusVal", {Integer}, Integer];
	SCSPObj0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_pobj", {Integer}, Real];
	SCSDObj0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_dobj", {Integer}, Real];
	SCSResPri0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_resPri", {Integer}, Real];
	SCSResDual0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_resDual", {Integer}, Real];
	SCSResInfeas0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_resInfeas", {Integer}, Real];
	SCSResUnbdd0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_resUnbdd", {Integer}, Real];
	SCSResGap0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_relGap", {Integer}, Real];
	SCSSetupTime0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_setupTime", {Integer}, Real];
	SCSSolveTime0 = LibraryFunctionLoad[$SCSLinkLibrary, "SCSSolutionMap_ret_solveTime", {Integer}, Real];

	needInitialization = False;
]

LoadSCSLink[]



(* SCSData expression (SCSSolutionMap) related: *)

SCSDataID[e_SCSData] := ManagedLibraryExpressionID[e, "scs_solution_instance_manager"];

SCSDataQ[e_SCSData] := ManagedLibraryExpressionQ[e, "scs_solution_instance_manager"];
SCSDataQ[_] := False;

testSCSData[][e_] := testSCSData[SCSData][e];
testSCSData[mhead_Symbol][e_] :=
If[TrueQ[SCSDataQ[e]],
	True,
	Message[MessageName[mhead, "SCSinst"], e]; False
];
testSCSData[_][e_] := TrueQ[SCSDataQ[e]];

General::SCSinst = "`1` does not represent an active SCSData object.";

SCSDataCreate[] :=
Module[{},
	If[needInitialization, LoadSCSLink[]];
	CreateManagedLibraryExpression["scs_solution_instance_manager", SCSData]
];

SCSDataDelete[SCSData[id_]?(testSCSData[SCSDataDelete])] := SCSDataDelete0[id];

SCSDataDelete[l:{_SCSData..}] := SCSDataDelete /@ l;

SCSDataExpressions[] :=
Module[{list},
	If[needInitialization, LoadSCS[]];
	list = SCSDataIDList[];
	If[!ListQ[list],
	   $Failed,
	   Map[SCSData, list]]
]

(* Selectors *)

SCSx[SCSData[id_]?(testSCSData[SCSx])] = SCSx0[id];
SCSy[SCSData[id_]?(testSCSData[SCSy])] = SCSy0[id];
SCSs[SCSData[id_]?(testSCSData[SCSs])] = SCSs0[id];
SCSStringStatus[SCSData[id_]?(testSCSData[SCSStringStatus])] = SCSStringStatus0[id];
SCSStatusVal[SCSData[id_]?(testSCSData[SCSStatusVal])] = SCSStatusVal0[id];
SCSPObj[SCSData[id_]?(testSCSData[SCSPObj])] = SCSPObj0[id];
SCSDObj[SCSData[id_]?(testSCSData[SCSDObj])] = SCSDObj0[id];
SCSResPri[SCSData[id_]?(testSCSData[SCSResPri])] = SCSResPri0[id];
SCSResDual[SCSData[id_]?(testSCSData[SCSResDual])] = SCSResDual0[id];
SCSResInfeas[SCSData[id_]?(testSCSData[SCSResInfeas])] = SCSResInfeas0[id];
SCSResUnbdd[SCSData[id_]?(testSCSData[SCSResUnbdd])] = SCSResUnbdd0[id];
SCSResGap[SCSData[id_]?(testSCSData[SCSResGap])] = SCSResGap0[id];
SCSSetupTime[SCSData[id_]?(testSCSData[SCSSetupTime])] = SCSSetupTime0[id];
SCSSolveTime[SCSData[id_]?(testSCSData[SCSSolveTime])] = SCSSolveTime0[id];

scsverbose = Boole[Internal`$DebugBuild];

Options[SCSSolve] = {"normalize" -> 1, "scale" -> 5, "rho_x" -> 10.^-3,
	"max_iters" -> 2500, "eps" -> 10.^-3, "alpha" -> 1.8, "cg_rate" -> 2, "verbose" -> scsverbose, "warm_start"-> 0};

Options[SuperSCSSolve] = {"normalize" -> 1, "scale" -> 1., "rho_x" -> 0.001,
	"max_iters" -> 2500, "previous_max_iters" -> -1, "eps" -> 10.^-3, "alpha" -> 1.5, "cg_rate" -> 2.,
	"verbose" -> scsverbose, "warm_start"-> 0, "do_super_scs" -> 1, "k0" -> 0, "k1" -> 1, "k2" -> 1,
	"c_bl" -> 0.999, "c1" -> 0.9999, "sse" -> 0.999, "ls" -> 10, "beta" -> 0.5, "sigma" -> 0.01,
	"direction" -> "restarted_broyden", "thetabar" -> 0.1, "memory" -> 10, "tRule" -> 1,
	"broyden_init_scaling" -> 1, "do_record_progress" -> 0}


SuperSCSSolve[b0_?VectorQ, c0_?VectorQ, AMat0_?MatrixQ, f_Integer, l_Integer,
         q0_?VectorQ, s0_?VectorQ, ep_Integer, ed_Integer, p0_?VectorQ,
		opts : OptionsPattern[]] :=
	Module[{b = Normal[b0], c = Normal[c0], AMat = SparseArray[AMat0], q = Normal[q0],
		s = Normal[s0], p = Normal[p0], solID, solInstance, statusVal, error, args,
		settings, normalize, scale, rhox, maxiters, previousmaxiters, eps, alpha, cgrate, verbose, warmstart,
		dosuperscs, k0, k1, k2, cbl, c1, sse, ls, beta, sigma, direction, thetabar, memory, tRule,
		broydeninitscaling, dorecordprogress},

		(* check types *)
		settings = OptionValue[SuperSCSSolve, #]&/@
			{"normalize", "scale", "rho_x", "max_iters", "previous_max_iters", "eps", "alpha",
			 "cg_rate", "verbose", "warm_start", "do_super_scs", "k0", "k1", "k2", "c_bl", "c1", "sse",
			 "ls", "beta", "sigma", "direction", "thetabar", "memory", "tRule", "broyden_init_scaling",
			 "do_record_progress"};
		
		{normalize, scale, rhox, maxiters, previousmaxiters, eps, alpha, cgrate, verbose, warmstart,
		 dosuperscs, k0, k1, k2, cbl, c1, sse, ls, beta, sigma, direction, thetabar, memory, tRule,
		 broydeninitscaling, dorecordprogress} = settings;

		error = SCSCheckInput[b, c, AMat, f, l, q, s, ep, ed, p];
		If[error != 0, Return[$Failed]];

		(* setup a Solution instance *)
		solInstance = SCSDataCreate[];
		solID = SCSDataID[solInstance];

		statusVal = SCSSolve0[solID, b, c, AMat, f, l, q, s, ep, ed, p, 
			normalize, scale, rhox, maxiters, previousmaxiters, eps, alpha, cgrate, verbose, warmstart,
			dosuperscs, k0, k1, k2, cbl, c1, sse, ls, beta, sigma, direction, thetabar, memory, tRule,
			broydeninitscaling, dorecordprogress];

		args = {maxiters};
		ReportError[statusVal, args];

		solInstance
	];

SCSCheckInput[b_, c_, AMat_, f_, l_, q_, s_, ep_, ed_, p_]:=
Module[{},
 If[!(TrueQ[Length[Amat["NonzeroPositions"]]>0]),
	Message[SCSLink::"amat"];
	Return[1];
	];
	0
];

ReportError[errcode_, args_]:=
Module[{maxiters = args[[1]]},
  Switch[errcode,
	-7, Message[SCSLink::"infeasinac", maxiters],
	-6, Message[SCSLink::"unbdinac", maxiters],
	-5, Message[SCSLink::"sigint"],
	-4, Message[SCSLink::"scserr"],
	-3, Message[SCSLink::"indtr", maxiters],
	-2, Message[SCSLink::"infeas"],
	-1, Message[SCSLink::"unbdd"],
	0, Message[SCSLink::"unfin"],
	2, Message[SCSLink::"inac", maxiters]
  ];
];

SCSSolve[b0_?VectorQ, c0_?VectorQ, AMat0_?MatrixQ, f_Integer, l_Integer,
         q0_?VectorQ, s0_?VectorQ, ep_Integer, ed_Integer, p0_?VectorQ,
		opts : OptionsPattern[]] :=
Module[{},
	OptionValue[{}]; (* reports any unauthorized options names *)
	SuperSCSSolve[b0, c0, AMat0, f, l, q0, s0, ep, ed, p0,
		Join[{"do_super_scs" -> 0}, FilterRules[{opts}, Options[SCSSolve]], {"scale" -> 5, "alpha" -> 1.8}]]
]

(* SCSStringStatus from the "SCS" library:
"Infeasible/Inaccurate" SCS_INFEASIBLE_INACCURATE (-7)
"Unbounded/Inaccurate" SCS_UNBOUNDED_INACCURATE (-6)
"Interrupted" SCS_SIGINT (-5)
"Failure" SCS_FAILED (-4)
"Indeterminate" SCS_INDETERMINATE (-3)
"Infeasible" SCS_INFEASIBLE (-2) /* primal infeasible, dual unbounded /
"Unbounded" SCS_UNBOUNDED (-1) / primal unbounded, dual infeasible /
SCS_UNFINISHED (0) / never returned, used as placeholder */
"Solved" SCS_SOLVED (1)
"Solved/Inaccurate" SCS_SOLVED_INACCURATE (2)
*)

General::"infeasinac" = "The problem may be infeasible. Specifying a value for MaxIterations greater than `1` may improve the solution."
General::"unbdinac" = "The problem may be unbounded. Specifying a value for MaxIterations greater than `1` may improve the solution."
General::"sigint" = "The solver was interrupted."
General::"scserr" = "Failed to find a solution."
General::"indtr" = "The solution is indeterminate. Specifying a value for MaxIterations greater than `1` may improve the solution."
General::"infeas" = "The problem is infeasible."
General::"unbdd" = "The problem is unbounded."
General::"inac" = "The problem was solved but the requested tolerance could not be achieved. Specifying a value for MaxIterations greater than `1` may improve the solution."

End[]

EndPackage[]
