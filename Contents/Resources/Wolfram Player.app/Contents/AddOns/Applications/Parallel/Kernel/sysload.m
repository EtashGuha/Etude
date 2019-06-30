(* Autoloader of Parallel Computing Toolkit *)

(* :Package Version: 1.0  *)

(* :Summary:
   this is invoked from sysinit.m to integrate PCT into Mathematica
*)

(* :Discussion:
   this sets up autoloading of PCT and initializes the user's configuration.
   Packages are read only later when the declared symbols are first evaluated.
   This happens through Parallel/Kernel/autoload.m
*)

Begin["Parallel`Private`"]

`mainCtx= "System`";
`devCtx = "Parallel`Developer`";

`mainNames = Flatten[{
	mainCtx<>#&/@{"LaunchKernels", "Kernels", "AbortKernels", "CloseKernels", "$KernelCount", "$KernelID"},
	mainCtx<>#&/@{"ParallelSubmit","WaitAll","WaitNext", "CriticalSection"},
	mainCtx<>#&/@{"ParallelEvaluate","ParallelNeeds","DistributeDefinitions","DistributedContexts"},
	mainCtx<>#&/@{"Parallelize", "ParallelTry"},
	mainCtx<>#&/@{"SetSharedVariable", "SetSharedFunction", "$SharedVariables", "$SharedFunctions", "UnsetShared"},
	mainCtx<>#&/@{"EvaluationObject"},
	mainCtx<>"Parallel"<>#&/@{"Combine","Map","Table","Sum","Product","Do","Array"},
	{}
}];

(* when changing the symbol set above, also have a look at autoload.m *)

(* symbols that should exist, but do not cause autoloading *)

`nonLoadNames = Flatten[{
	mainCtx<>#&/@ {"$ConfiguredKernels", "$DistributedContexts"},
	{"Parallel`Protected`KernelObject"}
}];

(* developer context main functions *)

`devNames = devCtx<>#& /@ {
    "ClearKernels", "CloseError", "ConcurrentEvaluate", "ConnectKernel", "created",
    "dequeued", "DoneQ", "EvaluationCount", "finished", "KernelFromID",
    "KernelID", "KernelName", "KernelStatus", "LaunchDefaultKernels", "LaunchKernel",
    "Load", "ParallelDispatch", "ParallelPreferences", "Process",
    "ProcessID", "ProcessResult", "ProcessState", "queued", "QueueRun", "Receive",
    "ReceiveIfReady", "ResetQueues", "running", "Scheduling", "Send",
    "SendBack", "SetQueueType", "SubKernel", "$DistributedDefinitions", "ClearDistributedDefinitions",
    "$InitCode", "$LoadFactor", "$NotReady", "$ParallelCommands",
    "$Queue", "$QueueLength", "$QueueType"
};

(* user interface code *)

`uiNames = "Parallel`Palette`"<>#& /@ {
	"menuStatus", "tabPreferences", "buttonConfigure"
};

`symbolHeld[name_] := ToExpression[name, InputForm, Hold] (* create a symbol without evaluation *)

(* language version number, to detect master/subkernel version mismatch *)
`$ParallelLanguageVersion = 9;

(* remember path for later autoloading *)
Parallel`Private`tooldir = ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]]

setupMaster[]:= (
		Package`DeclareLoad[ Join[Symbol/@mainNames, Symbol/@devNames, Symbol/@uiNames],
			"Parallel`Kernel`autoload`", Package`ExportedContexts -> {}
		];
		(* create nonload symbols, but do not evaluate *)
		symbolHeld /@ nonLoadNames;
		(* to trigger conditional code inside Parallel/Kernel/init.m, whenever this is read later *)
		Parallel`Static`$sysload = True;
		Parallel`Static`$loaded = False;
)

(* defaults *)
Parallel`Static`$silent = True;
Parallel`Static`$persistentPrefs = False;
Parallel`Static`$enableLaunchFeedback = False;

Which[ (* what is our role? *)
	MemberQ[$CommandLine, "-noparallel"] || TrueQ[Parallel`Static`$noparallel],
		Parallel`Static`$noparallel=True; SetAttributes[Parallel`Static`$noparallel,{Protected,Locked}];
		Get["Parallel`Kernel`noparinit`"];
		,
	TrueQ[System`Parallel`$SubKernel] || TrueQ[Parallel`Static`$SubKernel],
		Parallel`Static`$SubKernel = True; SetAttributes[Parallel`Static`$SubKernel,{Protected,Locked}];
		Get["Parallel`Kernel`subinit`"];
		,
	$LicenseType === "PlayerPro" || TrueQ[Parallel`Static`$playerPro], (* PlayerPro product *)
		Parallel`Static`$playerPro = True; SetAttributes[Parallel`Static`$playerPro,{Protected,Locked}];
		Parallel`Static`$Profile = "PlayerPro";
		setupMaster[];
		,
	$LicenseType === "Player" && Developer`$ProtectedMode || TrueQ[Parallel`Static`$player], (* Player product *)
		Parallel`Static`$player = True; SetAttributes[Parallel`Static`$player,{Protected,Locked}];
		Parallel`Static`$Profile = "Player";
		setupMaster[];
		,
	$LicenseType === "Player" && !Developer`$ProtectedMode || TrueQ[Parallel`Static`$playerEnterprise], (* Player product *)
		Parallel`Static`$playerEnterprise = True; SetAttributes[Parallel`Static`$playerEnterprise,{Protected,Locked}];
		Parallel`Static`$Profile = "PlayerEnterprise";
		setupMaster[];
		,
	True, (* master kernel, Mathematica *)
		If[System`Parallel`$SubKernel=!=False, System`Parallel`$SubKernel=False]; (* we are NOT a subkernel *)
		Parallel`Static`$master = True; SetAttributes[Parallel`Static`$master,{Protected,Locked}];
		Parallel`Static`$persistentPrefs = True;
		Parallel`Static`$enableLaunchFeedback = True;
		setupMaster[];
		Get["Parallel`SysInfo`"]; (* Set up SystemInformation[] *)
]

Protect[Parallel`Static`$persistentPrefs]

(* $ConfiguredKernels hack; must be outside the previous Which[] *)
If[Parallel`Static`$master && !ValueQ[$ConfiguredKernels],
	$ConfiguredKernels:=(Clear[$ConfiguredKernels];$KernelID;$ConfiguredKernels)
];

End[]

Null
