(* :Title: Palette.m -- ParallelComputing palette support code *)

(* :Context: Parallel`Palette` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   this package is needed for the palette actions
   and for controlling PCT based on palette settings
 *)

(* :Package Version: 1.0 alpha  *)

(* :Mathematica Version: 7.0 *)

(* :History:
   1.0 for PCT 3.0
*)

System`$KernelID; (* force actual loading of Parallel Tools in case of sysload autoload *)


BeginPackage["Parallel`Palette`"]

(* user interface FE glue *)

`menuStatus::usage = ""
`tabPreferences::usage = ""

(* functions invoked in the palette *)

{buttonConfigure}

(* preferences *)

paletteConfig (* head for the palette-related preferences *)

(* aux *)


(* PCT config tab *)
masterConfig::usage = "masterConfig is the configuration object (compatible with the one from SubKernels`) of PCT."

Parallelize::oldpref = "User preference setting migrated from previous version; some settings may be lost."


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`ParallelPreferences::usage = "ParallelPreferences[] gives a parallel computing preference panel."
SyntaxInformation[ParallelPreferences] = { "ArgumentsPattern" -> {} }

EndPackage[]


BeginPackage["Parallel`Protected`"] (* semi-hidden methods, aka protected *)

`paletteProfiled::usage = "paletteProfiled[name] gives the profile entry specialized for the current default evaluator."
`getProfiled::usage = ""
`setProfiled::usage = ""
`clearProfiled::usage = ""

EndPackage[]


(* messages *)

General::oldpref = "Warning: User preferences version `1` is older than current version `2`."


Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

Needs["SubKernels`"]; Needs["SubKernels`Protected`"];
Needs["Parallel`Preferences`"] (* persistent storage of palette settings plus localization *)

(* master kernel config parameters *)
{`autoLaunch, `loadDebug, `tryClone, `recoveryMode, `cloudCredentials}


(* handle preferences versioning *)

prefs[load][]; (* set them up *)

(* version number of the preferences *)

oldVersion = prefs[get]["Version"]
myVersion = 3.01


(* status menu; size is dynamic *)

`menuStatusNB = Null

menuStatus[] := Module[{},
	If[menuStatusNB=!=Null, NotebookClose[menuStatusNB]; menuStatusNB=Null ];
	(* create and display *)
	menuStatusNB = CreateDialog[ Parallel`Developer`KernelStatus[],
		WindowTitle->tr["StatusMenuTitle"], WindowSize->Fit,
		WindowFrameElements->{"CloseBox"}
	];
]

firstTime=False; (* this will be set True temporarily when initializing things at load time *)


(* profiles, one-to-one correspondence with master kernels [except for default one]  *)

(* determine master to use, that is our own name; this is used everywhere except when driving the parallel prefs tab *)

defaultMaster="Local"; (* this is built in *)

If[ $FEQ,
	currentMaster = Evaluator /. Options[$FrontEnd]; If[!StringQ[currentMaster], currentMaster=defaultMaster];
	myEvaluator = CurrentValue["RunningEvaluator"]; If[!StringQ[myEvaluator], myEvaluator=currentMaster];
	(** If[myEvaluator==="System", myEvaluator = currentMaster]; (* welcome screen *) **)
,
	currentMaster = myEvaluator = "Batch"; (* best guess *)
]

(* if you change this, also look at Parallel/Preferences.m, debugQ[] *)

master = Which[
	ValueQ[Parallel`Static`$Profile] && (StringQ[Parallel`Static`$Profile] || Parallel`Static`$Profile===Automatic),
		Parallel`Static`$Profile,
	$FEQ,
		If[ myEvaluator === defaultMaster, Automatic, myEvaluator ],
	True,
		Automatic
]

If[Parallel`Static`$Profile=!=master,
	Unprotect[Parallel`Static`$Profile]; Parallel`Static`$Profile = master; Protect[Parallel`Static`$Profile]
]


(* which preference set to use; ok to use = instead of :=. the default is without the kernel name *)

paletteProfiled[name_] := If[master === Automatic, paletteConfig[name], paletteConfig[name, master]]

(* when getting prefs, we may have to fallback to the default, if the new profile has never been used before *)

getProfiled[name_] := If[prefs[exists][paletteProfiled[name]],
	prefs[get][paletteProfiled[name]],
	prefs[get][paletteConfig[name]]
]

(* when setting or clearing, we always use the profiled version *)

setProfiled[name_ -> val_] := prefs[set][paletteProfiled[name] -> val]
clearProfiled[name_] := prefs[clear][paletteProfiled[name]]


(* kernel configuration *)

(* choose implementations to load *)

`confs (* configure methods of all loaded implementations *)
`allconfs (* including parallel tools config *)

{`knownImplementations, `enabledContexts}

`enabledPrefRaw = "enabledImplementations"
`enabledPref = paletteProfiled[enabledPrefRaw] (* the preference name, profiled, good for set/clear *)

dirtyImpl := enabledContexts  =!= getProfiled[enabledPrefRaw]

implementationLoader[ctx_] :=
With[{oldctxpath=$ContextPath},
	If[StringQ[Parallel`Private`tooldir], PrependTo[$Path, Parallel`Private`tooldir]]; (* look in installation first *)
	Catch[ Needs[ctx] ];
	$ContextPath=oldctxpath; (* do not put any of the new contexts onto path *)
]

saveImpl := Module[{enabled},
	If[ enabledContexts  =!= getProfiled[enabledPrefRaw],
		(* save persistent User config *)
		prefs[set][enabledPref -> enabledContexts];
		enabledContexts = getProfiled[enabledPrefRaw];
	];
	(* load new ones *)
	implementationLoader /@ enabledContexts;
	(* note: we cannot really unload unchecked ones *)
	enabled = Select[$SubKernelTypes, MemberQ[enabledContexts, #[subContext]]&]; (* enabled ones only *)
	confs = Select[Through[enabled[subConfigure]], #[configQ]===True&]; (* config methods *)
	allconfs = Append[confs, masterConfig];
	revertConfig; (* init builtin defaults from persistent ones for all kernel configs *)
]

revertImpl := (
	knownImplementations = prefs[get][paletteConfig["knownImplementations"]]; (* not editable for now *)
	enabledContexts = Intersection[getProfiled[enabledPrefRaw], knownImplementations[[All,1]]];
	saveImpl;
)

factoryImpl := Module[{},
	prefs[clear][enabledPref];
	revertImpl;
]


(* define/get corresponding preferences *)
(* we don't do prefs[add] here, this belongs into Kernel/Preferences.m if at all;
   it works even without one *)

(* the preferences panel *)

ParallelPreferences[___] :=
Grid[{ {Invisible["M"], SpanFromLeft},
	{Style[ tr["GeneralPreferences"], Bold], SpanFromLeft},
	{Panel[masterPreferences[], BaseStyle->"ControlStyle"], SpanFromLeft},
	{Style[ tr["ParallelKernelConfiguration"], Bold], SpanFromLeft},
	{Dynamic[StringForm[tr["TotalNumberOfKernels"], Total[KernelCount/@$ConfiguredKernels]]], SpanFromLeft},
	{subkernelPreferences[], SpanFromLeft},
	{Item[Row[{Button[tr["ResetToDefaults"], factoryConfig, ImageSize->All], autoSave[]}], Alignment->Left],
	 Item[Row[{Hyperlink[Style[tr["HelpAnchor"], "ControlStyle", FontColor -> Automatic], tr["HelpURL"], Appearance -> Button],
	 " ", Button[tr["ParallelKernelStatus"], menuStatus[], ImageSize->All]}], Alignment->Right]}
  }, ItemSize->Full, Alignment->Left, Spacings->{Automatic,{0,0.8,0.6,1.7,0.2,0.5,1,0}}
]
Hyperlink[Style["Help...", "ControlStyle", FontColor -> Automatic], "paclet:ParallelTools/tutorial/ConfiguringAndMonitoring", Appearance -> Button]

masterPreferences[] :=
	masterConfig[tabConfig]

(* tab handler for enabled and other subkernel types *)
(* do not allow to unload ALL methods *)

loadunloadButton[{ctx_, name_, ___}] :=
	Button[Dynamic[If[MemberQ[enabledContexts,ctx],tr["DisableConnection"], tr["EnableConnection"]]<>name],
		(enabledContexts = If[!MemberQ[enabledContexts,ctx], Union[enabledContexts,{ctx}], Complement[enabledContexts, {ctx}]];
		 saveImpl),
		Enabled->True, ImageSize -> All
	]

`tabwidth; (* a good value for the width of the Parallel tab *)

(* tab content *)

subtab[impl:{ctx_, name_, blurb_:"", url_:None, ___}] :=
With[{subtypes = Select[$SubKernelTypes, #[subContext]===ctx&], sp = 1.3},
	Column[{
	  Style[If[StringQ[url],
				Row[{blurb, "\[NonBreakingSpace]", Hyperlink["\[RightSkeleton]", url]}],
     			Row[blurb]
     		], LineIndent -> 0],
	  If[Length[subtypes]>0 && MemberQ[enabledContexts,ctx], (* loaded and valid *)
		With[{subtype = First[subtypes]},
		If[ TrueQ[subtype[subConfigure][configQ]], (* has configure method, let it create content *)
				subtype[subConfigure][tabConfig]
			, (* else no configure method: make our own dummy tab *)
				Pane[tr["NoPreferenceSettings"]]
		]]
		, (* not (successfully) loaded, or unloaded *)
		""
	  ],
	  loadunloadButton[impl]
	},Left,sp]
]

(* some implementations may be hidden: #[[5]] == False; they can still be loaded via prefs! *)

subkernelPreferences[] := With[{visibleImplementations = Select[knownImplementations, Length[#]<5 || #[[5]]&]},
	tabwidth = CurrentValue[EvaluationNotebook[], {TaggingRules, "TabViewApproxWidth"}];
	If[!NumberQ[tabwidth], tabwidth=630];
	TabView[(Part[#,2] -> Dynamic[subtab[#]])& /@ visibleImplementations, ImageSize->tabwidth ]
]


(* open the preferences tab, e.g., from a button *)

buttonConfigure[] := FrontEndExecute[{
	FrontEnd`SetOptions[FrontEnd`$FrontEnd, FrontEnd`PreferencesSettings -> {"Page" -> "Parallel"}], 
	FrontEnd`FrontEndToken["PreferencesDialog"]
}]


(* the parallel preferences tab content; args are not used now, reserve for future *)

tabPreferences[args___] := ParallelPreferences[args]


(* kernel type preferences; ok to use paletteProfiled, the existence check is done in revertConfig *)

prefForSub[conf_] := paletteProfiled[conf[nameConfig]] (* name of kernel configuration preference *)

(* are there local unsaved changes? *)

dirty := Or @@ Map[ Function[conf, With[{name = prefForSub[conf]}, conf[getConfig] =!= prefs[get][name]]], allconfs ]

(* make it autosave after each change, needs to be in visible dynamic *)

autoSave[] := Dynamic[If[dirty, saveConfig;"",""]]

(* save and apply *)

(* the user's setting of $ConfiguredKernels is preserved on first load,
   and if the variable is Protected[], thereafter *)

`saveCounter = 0;

saveConfig := Module[{},
	(* save persistent config *)
	Scan[Function[conf,
		With[{name = prefForSub[conf]},
			If[ conf[getConfig] =!= prefs[get][name],
				prefs[set][name -> conf[getConfig]];
				If[conf===masterConfig, conf[setConfig, prefs[get][name]]]; (* trigger side effects *)
				(* conf[setConfig, prefs[get][name]]; *)
				saveCounter++ ];
		]],
	allconfs];
	(* debug prefs hack, as in Preferences.m *)
	With[{debugname = debugPreference[master]},
		If[!prefs[exists][debugname] || prefs[get][debugname] =!= loadDebug, prefs[set][debugname -> loadDebug]; saveCounter++ ]];
	If[!(MemberQ[Attributes[$ConfiguredKernels], Protected] ||
	     firstTime && ListQ[$ConfiguredKernels]),
		$ConfiguredKernels = Flatten[#[useConfig]& /@ confs,1] (* preserve user's choice otherwise *)
	]
]

revertConfig := Module[{},
	(* restore from persistent config *)
	Scan[Function[conf,
		With[{name = prefForSub[conf]},
			If[!prefs[exists][name], prefs[add][name->{}]]; (* add missing, with built-in default setting *)
			conf[setConfig, prefs[get][name]]; (* set from persistent value *)
		]],
	allconfs];
	saveConfig;   (* synchronize; 'dirty' should now be false *)
]

factoryConfig := Module[{},
	(* delete persistent config *)
	Scan[Function[conf,
		With[{name = prefForSub[conf]},
			prefs[clear][name];
		]],
	allconfs];
	InitializePreferences[]; (* just to be safe: wipe file *)
	revertConfig; (* for consistency *)
]


(* aux *)


(* master kernel config tab *)

masterConfig[configQ] = True
masterConfig[nameConfig] = "PCT"

masterConfig[setConfig] := masterConfig[setConfig, {}]
masterConfig[setConfig, {}] := masterConfig[setConfig, {Automatic, True, True, "Abandon", False}] (* same as set in Kernel/Preferences.m *)
masterConfig[setConfig, {al_, ld_, cl_, rec_, cc_:False, ___}] := Module[{},
	autoLaunch = al; loadDebug = ld; tryClone = cl; recoveryMode = rec /. "ReQueue"->"Retry";
	cloudCredentials = TrueQ[cc];
	SetSystemOptions["ParallelOptions" -> "RelaunchFailedKernels" -> tryClone];
	SetSystemOptions["ParallelOptions" -> "RecoveryMode" -> recoveryMode];
	If[!ValueQ[Parallel`Static`$autolaunch], Parallel`Static`$autolaunch = autoLaunch]; (* allow overriding *)
	Parallel`Settings`$ForwardCloudCredentials = cloudCredentials; (* no system option yet *)
]
masterConfig[setConfig, _] := masterConfig[setConfig, {}] (* unknown value *)

masterConfig[getConfig] := {autoLaunch, loadDebug, tryClone, recoveryMode, cloudCredentials}

masterConfig[useConfig] = {} (* nothing to contribute to kernel list *)

masterConfig[tabConfig] := Grid[{
	{tr["LaunchParallel"], Row[{RadioButton[Dynamic[autoLaunch], False], tr["LaunchManual"]}, " "],
		                   Row[{RadioButton[Dynamic[autoLaunch], Automatic], tr["LaunchWhenNeeded"]}, " "],
		                   Row[{RadioButton[Dynamic[autoLaunch], True], tr["LaunchAtStartup"]}, " "]},
	{tr["EvaluationFailure"], Row[{RadioButton[Dynamic[recoveryMode], "Retry"], tr["EvaluationFailureReQueue"]}, " "],
		                   Row[{RadioButton[Dynamic[recoveryMode], "Abandon"], tr["EvaluationFailureAbandon"]}, " "],
		                   SpanFromLeft},
	{Row[{Checkbox[Dynamic[tryClone]], tr["TryRelaunch"]}],
	 Row[{Checkbox[Dynamic[loadDebug]], tr["EnableParallelMonitoring"]}],
	 SpanFromLeft, SpanFromLeft },
	{Row[{Checkbox[Dynamic[cloudCredentials]], tr["ForwardCloudCredentials"]}],
	 SpanFromLeft,
	 SpanFromLeft, SpanFromLeft }
  }, Alignment->{Left,Baseline}
]


(* init *)

If[oldVersion<myVersion,
	(* Message[Parallelize::oldpref, oldVersion, myVersion]; *)
	(* Migrate user prefs *)
	(* remove pre-3.0.1 RemoteServices` *)
	If[ oldVersion<3.01,
		clearProfiled["remote services kernel"];
	];
]
prefs[set]["Version" -> myVersion]; (* record at User level *)


Block[{firstTime=True}, revertImpl]; (* this triggers everything: load kernel implementations and their preferences *)

(* a workaround for an unwanted dynamic update involving SystemOptions[] *)

Internal`SetValueNoTrack[Parallel`Settings`$RecoveryMode, True]
Internal`SetValueNoTrack[Parallel`Settings`$RelaunchFailedKernels, True]
Internal`SetValueNoTrack[Parallel`Settings`$BusyWait, True]
Internal`SetValueNoTrack[Parallel`Settings`$MathLinkTimeout, True]
Internal`SetValueNoTrack[Parallel`Settings`$AbortPause, True]


End[]

Protect[ParallelPreferences, menuStatus, tabPreferences]

EndPackage[]
