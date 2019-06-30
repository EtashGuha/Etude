(* :Title: SysInfo.m -- Parallel Tools SystemInformation tab  *)

(* :Context: System`InfoDump` (extends main SystemInformation stuff) *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   set up the "Parallel" SystemInformation[] tab, without actually loading Parallel Tools
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 7 *)

Parallel`Private`oldpath=$ContextPath;
$ContextPath={"System`", "Parallel`Developer`"}; (* the symbols should exist, but don't touch them *)

Begin["System`InfoDump`"];

`$par = "Parallel" (* name of this stuff *)
`parLoad := ($KernelID; parLoad=Null;) (* cause autoloading; needed for symbols that do not cause autoloading by themselves *)

`protected = Unprotect[ SystemInformation ]

SystemInformation[$par, "Properties"] := {

	"KernelCount",
	"RunningKernels",

	"Debugging",
	"AutomaticLaunching",
	"FailedKernelRelaunching",
	"EvaluationFailureRecovery",

	"ProcessorCount",
	"KernelConfiguration",
	"AvailableConnectionMethods",
	"LoadedConnectionMethods",

	"DistributedDefinitions",
	"ParallelPackages",
	"SharedVariables",
	"SharedFunctions",

	"ParallelToolsVersion"
}

With[{$par=$par, props=SystemInformation[$par, "Properties"]},
  SystemInformation[$par] := Internal`DeactivateMessages[
	(# -> SystemInformation[$par, #])& /@ props,
  FrontEndObject::notavail ]
]

(* TODO fill in missing ones and remove once done *)

With[{$par=$par, props=SystemInformation[$par, "Properties"]},
 	(SystemInformation[$par, #] := Missing["NotActive"])& /@ props
]

(* properties *)

SystemInformation[$par, "KernelCount"] := $KernelCount

SystemInformation[$par, "RunningKernels"] := Kernels[]

SystemInformation[$par, "ProcessorCount"] := $ProcessorCount

SystemInformation[$par, "KernelConfiguration"] := $ConfiguredKernels

SystemInformation[$par, "AvailableConnectionMethods"] :=
	(parLoad; First/@Parallel`Palette`Private`knownImplementations)
SystemInformation[$par, "LoadedConnectionMethods"] :=
	(parLoad; Parallel`Palette`Private`enabledContexts)

SystemInformation[$par, "ParallelToolsVersion"] :=
	(parLoad; NumberForm[Parallel`Information`$VersionNumber,{3,1}])

SystemInformation[$par, "Debugging"] :=
	(parLoad; Parallel`Debug`$Debug)

SystemInformation[$par, "AutomaticLaunching"] :=
	(parLoad; Parallel`Static`$autolaunch)

SystemInformation[$par, "FailedKernelRelaunching"] :=
	(parLoad; Parallel`Settings`$RelaunchFailedKernels)

SystemInformation[$par, "EvaluationFailureRecovery"] :=
	(parLoad; Parallel`Settings`$RecoveryMode)

SystemInformation[$par, "DistributedDefinitions"] :=
	(parLoad; HoldForm@@@$DistributedDefinitions)

SystemInformation[$par, "SharedVariables"] :=
	(parLoad; HoldForm@@@$SharedVariables)

SystemInformation[$par, "SharedFunctions"] :=
	(parLoad; HoldForm@@@$SharedFunctions)


(* formats *)

With[{$par=$par},
 formatTabContent[$par, {___, $par -> lis_List, ___}] := Block[{width = $leftcolumnwidth},
  Column[{
  	makeinfogrid[$par, None, {
  		"KernelCount",
  		"RunningKernels",
  		"KernelConfiguration"
  		}, lis, width],
  	makeinfogrid[$par, None, {
  		"ProcessorCount"
  		}, lis, width],
  	makeinfogrid[$par, None, {
  		"Debugging",
  		"AutomaticLaunching",
  		"FailedKernelRelaunching",
  		"EvaluationFailureRecovery"
  		}, lis, width],
  	makeclosedinfogrid["Shared Resources", $par, {{
  		"DistributedDefinitions",
  		"SharedVariables",
		"SharedFunctions"
  		}}, lis, width, {False}],
  	makeclosedinfogrid["Connection Methods", $par, {{
  		"AvailableConnectionMethods",
  		"LoadedConnectionMethods"
  		}}, lis, width, {False}],
  	makeinfogrid[$par, None, {
  		"ParallelToolsVersion"
  		}, lis, width],

  	""
  },
	RowSpacings -> rowSpacings[{"loose","loose","loose","loose","loose","tight","tight","tight","tight","tight","loose"}],
	StripOnInput -> True]
]]


Protect[Evaluate[protected]]

End[]

$ContextPath=Parallel`Private`oldpath;

Null
