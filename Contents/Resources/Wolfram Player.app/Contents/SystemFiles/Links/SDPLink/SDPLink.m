(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["SDPLink`"]

$SDPLinkLibrary::usage  = "$SDPLinkLibrary is the full path to the SDPLink library loaded."
$SDPLinkDirectory::usage = "$SDPLinkDirectory gives the location of the SDPLink library."
LoadSDPLink::usage  = "LoadSDPLink[] loads the SDPLink library."
SDPSolve::usage = "Solve a SDP problem."

SDPData::usage = "SDPData[id] represents an instance of an SDPData expression created by SDPSolve."
SDPDataID::usage = "SDPDataID[data] gives the instance id of an SDPData expression data."
SDPDataQ::usage = "SDPDataQ[expr] gives True if expr represents an active instance of an SDPData object."
SDPDataCreate::usage = "SDPDataCreate[] creates an instance of an SDPData expression."
SDPDataExpressions::ussage = "SDPDataExpressions[] shows all active SDPData expression instances."
SDPDataDelete::usage = "SDPDataDelete[expr] removes an instance of an SDPData expression, freeing up memory."
SDPReturnCode::usage = "SDPReturnCode[data] gives a number indicating the status of the solution."
SDPStringStatus::usage = "SDPStringStatus[data, method] gives a string indicating the status of the solution."
SDPPObj::usage = "SDPPObj[data] gives the maximal value of the primal objective function from the SDPData expression data."
SDPDObj::usage = "SDPDObj[data] gives the minimal value of the dual objective function from the SDPData expression data."
SDPX::usage = "SDPX[data] gives the position matrix of the maximal value of the primal objective function from the SDPData expression data."
SDPy::usage = "SDPy[data] gives the position vector of the minimal value of the dual objective funtion from the SDPData expression data."
SDPZ::usage = "SDPZ[data] gives the final value of the positive semidenifinite slack matrix in the dual problem from the SDPData expression data."

CSDP::usage = "CSDP is used as a symbol for message heads from CSDP error codes."
DSDP::usage = "CSDP is used as a symbol for message heads from DSDP error codes."


Begin["`Private`"]
(* Implementation of the package *)

$SDPLinkDirectory = DirectoryName[$InputFileName];
$targetDir = FileNameJoin[{$SDPLinkDirectory, "LibraryResources", $SystemID}];

$SDPLinkLibrary = Block[{$LibraryPath = $targetDir}, FindLibrary["SDPLink"]];

(*
 Load all the functions from the SDPLink library
*)

$SDPPrintLevel = 0;

needInitialization = True;

LoadSDPLink[] :=
Block[{$LibraryPath = $targetDir},
	SDPSolve0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPLinkSolve", {Integer, Integer, Integer, {Integer, 1}, {Real, 1},
		LibraryDataType[SparseArray, Real, 2], LibraryDataType[SparseArray, Real, 3], UTF8String, Real, Integer, Integer}, Integer];
	SDPDataDelete0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_delete", {Integer}, Integer];
	SDPReturnCode0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retRetCode", {Integer}, Integer];
	SDPDataIDList = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retIDList", {}, {Integer, 1}];
	SDPPObj0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retPObj", {Integer}, Real];
	SDPDObj0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retDObj", {Integer}, Real];
	SDPy0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retY", {Integer}, {Real, 1}];
	SDPX0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retX", {Integer}, LibraryDataType[SparseArray, Real]];
	SDPZ0 = LibraryFunctionLoad[$SDPLinkLibrary, "SDPSolutionMap_retZ", {Integer}, LibraryDataType[SparseArray, Real]];
	needInitialization = False;
]

LoadSDPLink[]



(* SDPData expression (SDPSolMap) related: *)

SDPDataID[e_SDPData] := ManagedLibraryExpressionID[e, "sdp_solution_instance_manager"];

SDPDataQ[e_SDPData] := ManagedLibraryExpressionQ[e, "sdp_solution_instance_manager"];
SDPDataQ[_] := False;

testSDPData[][e_] := testSDPData[SDPData][e];
testSDPData[mhead_Symbol][e_] :=
If[TrueQ[SDPDataQ[e]],
	True,
	Message[MessageName[mhead, "sdpinst"], e]; False
];
testSDPData[_][e_] := TrueQ[SDPDataQ[e]];

General::sdpinst = "`1` does not represent an active SDPData object.";

SDPDataCreate[] :=
Module[{},
	If[needInitialization, LoadSDPLink[]];
	CreateManagedLibraryExpression["sdp_solution_instance_manager", SDPData]
];

SDPDataDelete[SDPData[id_]?(testSDPData[SDPDataDelete])] := SDPDataDelete0[id];

SDPDataDelete[l:{_SDPData..}] := SDPDataDelete /@ l;

SDPDataExpressions[] :=
Module[{list},
	If[needInitialization, LoadSDP[]];
	list = SDPDataIDList[];
	If[!ListQ[list],
	   $Failed,
	   Map[SDPData, list]]
]

(* Selectors *)
SDPReturnCode[SDPData[id_]?(testSDPData[SDPReturnCode])] := SDPReturnCode0[id];
SDPPObj[SDPData[id_]?(testSDPData[SDPPObj])] := SDPPObj0[id];
SDPDObj[SDPData[id_]?(testSDPData[SDPDObj])] := SDPDObj0[id];
SDPX[SDPData[id_]?(testSDPData[SDPX])] := convSymToFullMat[SDPX0[id]];
SDPy[SDPData[id_]?(testSDPData[SDPy])] := SDPy0[id];
SDPZ[SDPData[id_]?(testSDPData[SDPZ])] := convSymToFullMat[SDPZ0[id]];


(*
 convert symmetric matrix to full form
*)
convSymToFullMat[mat_] := Module[{matd, res},
   matd = DiagonalMatrix@Diagonal@mat;
   res = mat - matd;
   res + Transpose@res + matd
];


(*
	Solve an SDP problem in standard form
*)

INTMAX = 2^31-1;

SDPProcessOptions[blocksizes_, opts0___] :=
Module[{opts = {opts0}, methodoption, method, mopts = {}, tol, maxiter},
		methodoption = OptionValue[SDPSolve, opts, Method];
		If[Length[methodoption] == 0,
			method = methodoption;
		, (* else *)
			If[ListQ[methodoption],
				method = First[methodoption];
				If[Length[methodoption] >= 2,
					mopts = Rest[methodoption];
				];
			];
		];
		If[method === Automatic, method = "CSDP"];
		If[!MemberQ[{"CSDP", "DSDP"}, method],
			Message[SDPSolve::"method"];
			Return[$Failed];
		];
		If[method == "DSDP" && !(And @@ Positive[Most[blocksizes]]),
			Message[SDPSolve::"onelbl"];
			Return[$Failed];
		];
		tol = OptionValue[SDPSolve, {mopts, opts}, Tolerance];
		If[tol === Automatic, tol = 10.^-8];
		If[!TrueQ[tol >= 0], Message[SDPSolve::"tolnn", tol]; Return[$Failed]];
		maxiter = OptionValue[SDPSolve, {mopts, opts}, MaxIterations];
		If[maxiter === Automatic, maxiter = 100,
			If[maxiter === Infinity, maxiter = INTMAX]];
		If[!(IntegerQ[maxiter] && TrueQ[maxiter > 0]),
			Message[SDPSolve::"ioppfa", MaxIterations, maxiter]; Return[$Failed]];
	{method, tol, maxiter}
];

Options[SDPSolve] = {Method -> Automatic, Tolerance -> Automatic, MaxIterations -> Automatic};
SDPSolve[nmat_Integer, nblocks_Integer, blocksizes_List, objvals_List, FObj_SparseArray, FConstr_SparseArray,
		opts : OptionsPattern[]] :=
	Module[{solID, solInstance, options, method, tol, maxiter, printlevel = $SDPPrintLevel, retcode, args},

		options = SDPProcessOptions[blocksizes, opts];
		If[ListQ[options] && Length[options] == 3,
			{method, tol, maxiter} = options;
			,
			Return[$Failed]
		];

		(* setup a Solution instance *)
		solInstance = SDPDataCreate[];
		solID = SDPDataID[solInstance];

		retcode = SDPSolve0[solID, nmat, nblocks, blocksizes, objvals, FObj, FConstr, method, tol, maxiter, printlevel];

		args = {maxiter};
		If[method == "CSDP" && retcode != 0 ||
		   method == "DSDP" && retcode != 1,
		ReportError[method, retcode, args]];

		solInstance
	];

ReportError["CSDP", errcode_, args_]:=
Module[{maxiter = args[[1]]},
  Switch[errcode,
	1, Message[CSDP::"dinf"],
	2, Message[CSDP::"pinf"],
	3, Message[CSDP::"parsuc", maxiter],
	4, Message[CSDP::"maxiter", maxiter],
	5, Message[CSDP::"dfedge"],
	6, Message[CSDP::"pinfedge"],
	7, Message[CSDP::"noprog"],
	8, Message[CSDP::"singmat"],
	9, Message[CSDP::"naninf"]
  ];
];

ReportError["DSDP", errorcode_, args_]:=
Module[{maxiter = args[[1]]},
  Switch[errorcode,
	3, Message[DSDP::"dinf"],
	4, Message[DSDP::"pinf"],
	-6, Message[DSDP::"infst"],
	-2, Message[DSDP::"sstep"],
	-8, Message[DSDP::"schur"],
	-3, Message[DSDP::"maxiter", maxiter],
	-9, Message[DSDP::"numerr"],
	5, Message[DSDP::"bpobj"],
	7, Message[DSDP::"userstop"]
  ];
];

SDPStringStatus[data_, method_] := Module[{errcode = SDPReturnCode[data], string},
	Switch[method,
		"CSDP",
			string = Switch[errcode,
				0, "Solved",
				1, "Unbounded",
				2, "Infeasible",
				3, "PartialSuccess",
				4, "MaxIterationsReached",
				5, "AtDualFeasibilityEdge",
				6, "AtPrimalInfeasibilityEdge",
				7, "NoProgress",
				8, "SingularMatrix",
				9, "NonmachineNumber",
				_, "UnknownStatus"
			],
		"DSDP",
			string = Switch[errcode,
				1, "Solved",
				3, "Unbounded",
				4, "Infeasible",
				-6,"InfeasibleStart",
				-2, "ShortStep",
				-8, "IndefiniteSchur",
				-3, "MaxIterationsReached",
				-9, "NumericalError",
				5, "BigObjective",
				7, "Interrupted",
				_, "UnknownStatus"
			],
		_,
			Return[$Failed];
	];
	string
]


(* Messages from CSDP and DSDP error codes *)
General::dinf = "The primal problem is unbounded and the dual problem is infeasible."
General::pinf = "The primal problem is infeasible and the dual problem is unbounded."
General::parsuc = "Partial Success; a solution has been found, but full accuracy was not achieved. \
One or more of primal infeasibility, dual infeasibility, or relative duality gap are larger than their tolerances, \
but by a factor of less than 1000. Specifying a value for MaxIterations greater than `1` may improve the solution."
General::maxiter = "Reached the maximum number of `1` iterations."
General::dfedge = "Stuck at edge of dual feasibility."
General::pinfedge = "Stuck at edge of primal infeasibility."
General::noprog = "Lack of progress."
General::singmat = "X, Z, or O was singular."
General::naninf = "Detected NaN or Inf values."
General::userstop = "Terminated by user."
General::infst = "Infeasible start. The initial points y and r imply that S is not positive."
General::sstep = "Short step lengths created by numerical difficulties prevent progress."
General::schur = "Indefinite Schur matrix. Theoretically this matrix is positive definite."
General::numerr = "Numerical error occurred. Check the solution."
General::bpobj = "The primal objective is big enough to stop."


SDPSolve::method="The value of option Method should be Automatic, \"CSDP\" or \"DSDP\"."
SDPSolve::onelbl= "DSDP method expects at most one linear block placed last."
SDPSolve::tolnn = "Tolerance specification `1` must be a non-negative number."

End[]
EndPackage[]
