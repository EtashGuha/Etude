BeginPackage["MXNetLink`"]

Begin["`Bootstrap`Private`"]

Needs["GeneralUtilities`"];

GeneralUtilities`ClearPacletExportedSymbols["MXNetLink"];

getSources[] := (
	PreemptProtect @ Get @ FileNameJoin[{DirectoryName @ $InputFileName, "LibraryLink.m"}];
);

startLibrary[] := (
	If[FailureQ @ MXNetLink`Bootstrap`LoadLibraries[], Return @ $Failed];
	MXNetLink`Bootstrap`SetupNullHandles[];
	MXNetLink`Bootstrap`LoadOperators[];
	MXNetLink`Bootstrap`RunPostloadCode[];
);

loadMX[{defFile_, symFile_}] := (
	Get[defFile];
	Get[symFile];
	startLibrary[];
);

isWorking[] := Quiet @ Block[{nd},
	nd = MXNetLink`NDArrayCreate[{2,3,4}];
	MXNetLink`NDArrayGetNormal[nd] === {2,3,4}
]

(* we split the definitions we will save into pure code definitions and then
definitions that involve having loaded MXNetLink. This way we can save the 
version of the definitions that is free of any unsavable LibraryLink state. *)

saveMX[{defFile_, symFile_}] := (
	DumpSave[defFile, "MXNetLink`"];
	If[FailureQ @ MXNetLink`Bootstrap`LoadLibraries[], Return @ $Failed];
	MXNetLink`Bootstrap`SetupNullHandles[];
	If[!isWorking[], ReturnFailed[]]; 
	MXNetLink`Bootstrap`LoadOperators[];
	DumpSave[symFile, "MXNet`"];
	MXNetLink`Bootstrap`RunPostloadCode[];
);

If[GeneralUtilities`PacletLoadCached["MXNetLink", getSources, saveMX, loadMX, {"Definitions.mx", "Symbols.mx"}] === "Disabled",
	getSources[];
	startLibrary[];
];

End[]
EndPackage[]
