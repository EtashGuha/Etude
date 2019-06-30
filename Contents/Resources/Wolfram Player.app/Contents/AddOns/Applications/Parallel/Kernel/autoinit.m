(* Master Loader of Parallel Computing Toolkit *)

(* :Package Version: 3.0  *)

(* :Summary:
   Needs["Parallel`Kernel`autoinit`"]
*)

(* :Discussion:
   this loads the required parts of PCT, and is called from autoload.m,
   which in turn is set up from sysload.m using DeclareLoad[]
*)

(* check for client side loading attempt *)

If[ TrueQ[System`Parallel`$SubKernel],
	Needs::client = "Parallel Tools cannot be loaded in a subkernel of another parallel computation.";
	Message[Needs::client];
	Abort[]
]

(* check for loading twice *)

If[ ValueQ["Parallel`Private`$PackageVersion"],
	Needs::twice = "Parallel Tools are already loaded; they cannot be read again.";
	Message[Needs::twice];
	Abort[]
]

If[!TrueQ[Parallel`Static`$sysload], Parallel`Static`$sysload=False] (* not sysinit loading *)
If[!TrueQ[Parallel`Static`$silent], Parallel`Static`$silent=False] (* no silent operation *)
If[!TrueQ[Parallel`Static`$loadDebug], Parallel`Static`$loadDebug=False] (* no debugging *)
(* do not touch Parallel`Static`$autoload *)
Parallel`Static`$loaded = True;

(* if we are debugging under WWB, force debugging mode *)

If[ Workbench`$Debug, Parallel`Static`$loadDebug=True ]


(* debug / performance monitoring mode *)

If[Parallel`Static`$loadDebug && !ValueQ[!Parallel`Debug`$Debug], Get["Parallel`Debug`Full`"]] (* load debugging if requested and possible *)

(* pull in null debug definitions, if debugging is not enabled *)

If[ !ValueQ[Parallel`Debug`$Debug], Get["Parallel`Debug`Null`"] ]


(* set up main package context and read in required parts of PCT *)

BeginPackage["Parallel`"] (* empty context *)


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

EndPackage[]


BeginPackage["Parallel`Protected`"] (* semi-hidden methods, aka protected base class *)

registerPostInit::usage = "registerPostInit[expr] registers expr for evaluation after loading all of PCT."
runPostInit::usage = "runPostInit[] runs any registered post init code."

Begin["`Private`"]

$postInit = Hold[];
SetAttributes[registerPostInit, HoldFirst]
registerPostInit[code_] := ($postInit = Append[$postInit, Unevaluated[code]])

runPostInit[] := CompoundExpression @@ $postInit

End[]

EndPackage[]

(* load remaining stuff,  but should not show up on context path *)
Get["Parallel`Kernels`"]
Get["Parallel`Parallel`"]
Get["Parallel`Palette`"]
Get["Parallel`Status`"]
Get["Parallel`VirtualShared`"]
Get["Parallel`Concurrency`"]
Get["Parallel`Combine`"]
Get["Parallel`Evaluate`"]

Begin["`Private`"] (* Parallel`Private` *)

(* identification of package *)
`$PackageVersion = 3.01;
`$thisFile = $InputFileName

End[]

(* initializations that must happen after everything has been loaded *)

Needs["Parallel`Developer`"]
Needs["Parallel`Protected`"]

runPostInit[]

EndPackage[]
$ContextPath=DeleteCases[$ContextPath, "Parallel`", 1, 1] (* no longer on context path *)


(* official version numbering for whole Parallel Tools *)
(* see also Parallel`Private`$ParallelLanguageVersion in sysload.m *)
Parallel`Information`$VersionNumber = 9.0;
Parallel`Information`$ReleaseNumber = 1; (* aka 8.0.1 *)
Parallel`Information`$Version = "Parallel Tools 9.0 (Jan 19, 2017)"


If[!Parallel`Static`$silent,
	Print[ Parallel`Information`$Version ];
	Print[ "Created by Roman E. Maeder" ]
]

(* WWB code for parallel debugging *)

If[ Workbench`$Debug&&Parallel`Debug`$Debug,
	Needs["MEETParallel`"];
	$ContextPath = DeleteCases[$ContextPath, "MEETParallel`", 1, 1]
]


(* e o f *)
