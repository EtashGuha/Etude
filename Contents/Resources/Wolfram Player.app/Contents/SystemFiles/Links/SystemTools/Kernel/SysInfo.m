(* ::Package:: *)

(* ::Section:: *)
(*SysInfo.m*)


(* ::Text:: *)
(*This file sets up this paclet's SystemInformation tab.*)


(* ::Subsection:: *)
(*Initialization and Namespace*)


Begin["System`InfoDump`"];
Unprotect[SystemInformation]

SystemTools`Private`$MemoryNames
SystemTools`Private`$SystemSpecific


(* ::Subsection:: *)
(*SystemInformation[] Definitions*)


(*This definition of SystemInformation["Machine"]. That way it can be changed from the Paclet.
	For now it is defined as getSystemMemory, but will change when details are added to the System tab.*)
SystemTools`Private`systemInformation[args___] := Normal@SystemTools`Private`getSystemMemory[args]

(*Define Component Properties*)
SystemTools`Private`systemInformation["Properties"] := Join[SystemTools`Private`$MemoryNames, SystemTools`Private`$SystemSpecific];

(* SystemInformation[Component] *)
SystemInformation["Machine"] := Internal`DeactivateMessages[SystemTools`Private`systemInformation[], FrontEndObject::notavail]

(* SystemInformation[Component, Property] for each property *)
(SystemInformation["Machine", #] := SystemTools`Private`systemInformation[#])& /@ SystemTools`Private`systemInformation["Properties"]



(* ::Subsection:: *)
(*SystemInformation Tab Formatting*)



formatRow["Machine", prop_ -> HoldPattern[val_Quantity]] := formatRow["Machine", prop -> SetPrecision[TraditionalForm[val],4]]

formatTabContent["Machine", {___, "Machine" -> lis_List, ___}] := Block[{width = $leftcolumnwidth},
  Column[{
    makeinfogrid["Machine", None, SystemTools`Private`$MemoryNames[[;;1]], lis, width],
	makeinfogrid["Machine", None, SystemTools`Private`$MemoryNames[[2;;4]], lis, width],
    makeinfogrid["Machine", None, SystemTools`Private`$MemoryNames[[5;;7]], lis, width],
    makeinfogrid["Machine", None, SystemTools`Private`$MemoryNames[[8;;11]], lis, width],
    makeclosedinfogrid[ "System Specific", "Machine", {SystemTools`Private`$SystemSpecific}, lis, width, {False}]
    },
	RowSpacings -> rowSpacings[{"loose"}],
	StripOnInput -> True]
]


Protect[SystemInformation];
End[];
