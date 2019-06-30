(* :Title: Debug/Perfmon -- performance monitoring *)

(* :Context: Parallel`Debug`Perfmon`, extends Parallel`Debug` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Basic process control for parallel evaluation of Mathematica expressions.
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 7 *)

(* :History:
   1.0 first released version.
*)

$KernelID; (* force loading *)

BeginPackage["Parallel`Debug`Perfmon`", { "Parallel`Debug`" }]

Parallel`Debug`SetPerformance::usage = "SetPerformance[None] disables collecting performance data.
	SetPerformance[Manual] allows for manual collection of performance data.
	SetPerformance[Automatic] causes data to be collected automatically for each parallel calculation."
Parallel`Debug`StartPerformance::usage = "StartPerformance[] resets performance data for a new measurement."
Parallel`Debug`LapPerformance::usage = "LapPerformance[] takes performance measurements since last start."
Parallel`Debug`ReportPerformance::usage = "ReportPerformance[] returns performance data collected by LapPerformance[]."

SetPerformance::debug = "Performance data can be collected only in debug mode."

(* for raw stats, can be used in Dynamic *)
{masterAbs,masterCPU,subIDs,subCPUs,subCPU}


Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

Needs["Parallel`Developer`"] (* access developer stuff *)
Needs["Parallel`Protected`"] (* access protected stuff *)

`$perfmode = None;
`$collectPerf = False; (* should we bother at all? *)
`$inprogress = False; (* after start was called *)

SetPerformance[key:(Manual|Automatic|True)]/;!TrueQ[Parallel`Debug`$Debug] := (
	Message[SetPerformance::debug];
	$Failed
)

SetPerformance[key:(None|False|Manual|Automatic|True)] := Switch[key,
	None|False,
		$perfmode = None;
		$collectPerf = False;
		Clear[parStart, parStop],
	Manual,
		$perfmode = Manual;
		$collectPerf = True;
		Clear[parStart, parStop],
	Automatic|True,
		$perfmode = Automatic;
		$collectPerf = True;
		parStart := StartPerformance[];
		parStop  := LapPerformance[];,
	_, $Failed
]

(* values at start *)
{subCPUs0, masterCPU0, masterAbs0}

delFailed[l_List] := DeleteCases[l,$Failed]

StartPerformance[] := If[$collectPerf && Load[]==0, (* only in this case is it safe *)
	$inprogress = True;
	Block[{Parallel`Debug`Private`trace=Parallel`Debug`Private`traceIgnore},
		subCPUs0 = delFailed[ParallelEvaluate[{$KernelID, TimeUsed[]}]]
	];
	masterCPU0 = TimeUsed[]; masterAbs0 = AbsoluteTime[];
	, (* else keep out *)
	$inprogress = False;
]

LapPerformance[] := If[ $inprogress && Load[]==0,
	masterAbs = AbsoluteTime[] - masterAbs0;
	masterCPU = TimeUsed[] - masterCPU0;
	Block[{Parallel`Debug`Private`trace=Parallel`Debug`Private`traceIgnore},
		subCPUs1 = delFailed[ParallelEvaluate[{$KernelID, TimeUsed[]}]]
	];
	subIDs = subCPUs1[[All,1]];
	(* compare kernel sets *)
	If[ subCPUs0[[All,1]] == subIDs,
		subCPUs = subCPUs1[[All,2]]-subCPUs0[[All,2]];
		subCPU = Total[subCPUs];
		, (* else kernel set changed; cannot give stats *)
		subCPUs=Table[0.0,{Length[subIDs]}]; subCPU=0.0;
	];
	, (* else nothing to collect, but keep things consistent *)
	masterAbs = AbsoluteTime[] - masterAbs0;
	masterCPU = TimeUsed[] - masterCPU0;
	subIDs={}; subCPUs={}; subCPU=0; 
	$Failed
]

{masterAbs,masterCPU,subIDs,subCPUs,subCPU} = {0,0,{},{},0}

ReportPerformance[] :=
	{masterAbs,subCPU,Transpose[{Prepend[subIDs,$KernelID],Prepend[subCPUs,masterCPU]}]}


SetPerformance[None]

End[]

EndPackage[]
