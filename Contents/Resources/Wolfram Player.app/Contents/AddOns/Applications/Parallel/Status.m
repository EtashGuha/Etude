(* :Title: Status.m -- ParallelComputing status UI support code *)

(* :Context: Parallel`Status` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   this package is needed for the status displays
 *)

(* :Package Version: 1.0 alpha  *)

(* :Mathematica Version: 7.0 *)

(* :History:
   1.0 for PCT 3.0
*)

BeginPackage["Parallel`Status`"]

(* aux *)

BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`KernelStatus::usage = "KernelStatus[] gives a dynamic parallel kernel status display."
SyntaxInformation[KernelStatus] = { "ArgumentsPattern" -> {} }

EndPackage[]

Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

Needs["SubKernels`"]; Needs["SubKernels`Protected`"];
Needs["Parallel`Preferences`"] (* persistent storage of column chooser and localization *)
Needs["Parallel`Kernels`"]   (* for kernel properties in status window *)
Needs["Parallel`Developer`"] (* access protected stuff *)
Needs["Parallel`Protected`"] (* access protected stuff *)


prefs[load][]; (* set up persistent preferences if needed *)


(* kernel status display *)

(* column sets and enabled ones *)

LocalStatus = "LocalStatus"
LocalMasterStatus = "LocalMasterStatus"
RemoteStatus = "RemoteStatus"
ControlStatus = "ControlStatus"
ControlMasterStatus = "ControlMasterStatus"

(* the column sets are not editable for now *)
localStatus = prefs[get][LocalStatus]
localMasterStatus = prefs[get][LocalMasterStatus]
remoteStatus = prefs[get][RemoteStatus]
controlStatus = prefs[get][ControlStatus]
(* controlMasterStatus = prefs[get][ControlMasterStatus] *)

{`localColumns, `remoteColumns, `controlColumns, `perfcol}

(* performance data automatic *)
`perfAuto = True;

(* tooltip options *)
tooltipOptions = {TooltipDelay->Automatic};

(* the value perfmon expects, made of perfcol and perfAuto *)
perfValue[] := Which[
	!Parallel`Debug`$Debug, False,
	!perfcol, False,
	perfAuto, Automatic,
	True,     Manual
]

(* are there local unsaved changes to columns chosen? *)

dirtyColumns := localColumns =!= prefs[get]["LocalColumns"] ||
	remoteColumns =!= prefs[get]["RemoteColumns"] ||
	perfcol =!= prefs[get]["PerformanceData"] ||
	controlColumns =!= prefs[get]["ControlColumns"]

(* save and apply *)

saveColumns := Module[{},
	(* save persistent User config *)
	If[ localColumns  =!= prefs[get]["LocalColumns"],  prefs[set]["LocalColumns"  -> localColumns] ];
	If[ remoteColumns =!= prefs[get]["RemoteColumns"], prefs[set]["RemoteColumns" -> remoteColumns] ];
	If[ perfcol =!= prefs[get]["PerformanceData"], prefs[set]["PerformanceData" -> perfcol] ];
	If[ controlColumns  =!= prefs[get]["ControlColumns"], prefs[set]["ControlColumns" -> controlColumns] ];
]

revertColumns := Module[{},
	(* restore from persistent config, and make sure stuff is valud *)
	localColumns  =  Select[prefs[get]["LocalColumns"],   1<=#<=Length[localStatus]&];
	remoteColumns =  Select[prefs[get]["RemoteColumns"],  1<=#<=Length[remoteStatus]&];
	controlColumns = Select[prefs[get]["ControlColumns"], 1<=#<=Length[controlStatus]&];
	perfcol = TrueQ[prefs[get]["PerformanceData"]];
	doPerf[perfValue[]];
]

factoryColumns := Module[{},
	(* delete persistent User config *)
	prefs[clear]["LocalColumns"];
	prefs[clear]["RemoteColumns"];
	prefs[clear]["ControlColumns"];
	prefs[clear]["PerformanceData"];
	revertColumns; (* for consistency *)
]

doPerf[yn_] := If[Parallel`Debug`$Debug, Parallel`Debug`SetPerformance[yn]]


(* kernel launching and status *)

launching = False; (* set to True during launching kernels *)

kernelSummary[] :=
	Dynamic[ StringForm[tr["StatusKernelsRunning"], Length[$kernels],
		Which[launching, Style[StringForm[tr["StatusKernelsLaunching"], Parallel`Kernels`Private`grandCount,Parallel`Kernels`Private`grandTotalCount], Brown],
			  $kernelsIdle, Style[tr["StatusKernelsIdle"], Darker[Green]],
			  True, Style[tr["StatusKernelsBusy"], Red]]] ]

buttonAbort  := AbortKernels[] (* TODO this needs some more thought *)

(* the status display *)
SetAttributes[nanFix,HoldFirst]
nanFix[arg_] := Chop[Quiet[arg] /. r_/;!NumericQ[r] :> "n/a"]

With[{w1=0.05, b=0.008, r=0.008, size=200},
	loadIndicator[per_/;!NumberQ[per]] := loadIndicator[0];
	loadIndicator[per_] :=
	  Tooltip[
		Graphics[{Black, EdgeForm[Gray], Rectangle[{0, 0}, {1, w1}, RoundingRadius -> r],
				  {EdgeForm[], Green, Rectangle[{0, b}, {per, w1 - b}]}},
				  ImageSize -> size,  BaselinePosition->Scaled[0.2]],
		StringForm["`1`%", NumberForm[100per, {4, 1}]] ]
]

SetAttributes[propGrid,HoldAll]
propGrid[enabled_,oldcpus_,oldKernelCount_,oldstate_] :=
	Which[
	$kernelsIdle && (enabled || $KernelCount != oldKernelCount || perfValue[]=!=False && oldcpus =!= Parallel`Debug`Perfmon`subCPUs), (* only if idle *)
	  enabled=False;
	  Module[{locd, remd, cond, aligns,contents,tips,errors=0, localresults, remoteresults, controlresults},
	   Catch[
		CheckAbort[Block[{Parallel`Debug`Private`trace=Parallel`Debug`Private`traceIgnore},
		  oldKernelCount = $KernelCount; (* refresh trigger *)
		  With[{localProps =localStatus[[localColumns,2]],   localNames =List@@localStatus[[localColumns,1]],
		  				 localMasterProps = localMasterStatus[[localColumns]],
			             remoteProps=remoteStatus[[remoteColumns,2]], remoteNames=List@@remoteStatus[[remoteColumns,1]],
			             controlProps=controlStatus[[controlColumns,2]], controlNames=List@@controlStatus[[controlColumns,1]],
			             kernels = Kernels[]},
			localresults = Prepend[Composition[Through,List@@localProps]/@kernels,List@@localMasterProps];
			remoteresults = Prepend[ParallelEvaluate[List@@remoteProps, kernels],List@@remoteProps];
			If[ !FreeQ[remoteresults, $Failed], Throw[$Failed, problemo]]; (* something went wrong *)
			controlresults = Prepend[Composition[Through,List@@controlProps]/@kernels,Null&/@List@@controlProps];
			contents=Join[{Join[{tr["LocalPropertiesName_ID"]},localNames,remoteNames,controlNames]},
			  Transpose[Join[
				{Prepend[KernelID/@kernels,$KernelID]},
				Transpose[localresults],
				Transpose[remoteresults],
				Transpose[controlresults]
			  ]]
			];
			aligns = Join[{Right},List@@localStatus[[localColumns,4]],List@@remoteStatus[[remoteColumns,4]],List@@controlStatus[[controlColumns,4]]];
			(* use descriptions also as tooltips *)
			tips  = Join[{tr["LocalPropertiesDescription_ID"]},List@@localStatus[[localColumns,3]],List@@remoteStatus[[remoteColumns,3]],List@@controlStatus[[controlColumns,3]]];
		    If[perfValue[]=!=False, Module[{abs,cpus,sum,subs,ids,col1,col2,col3},
				{abs,sum,subs} = Parallel`Debug`ReportPerformance[];
				oldcpus = Parallel`Debug`Perfmon`subCPUs; (* refresh trigger *)
				ids=subs[[All,1]];
				If[ids==Rest[contents][[All,1]], (*only if they match *)
					cpus = subs[[All,2]],
					cpus = Table[0,{oldKernelCount+1}]
				];
				col1=Prepend[NumberForm[#,{6,3}]& /@ cpus,tr["StatusKernelsTime"]];
				col2=Prepend[loadIndicator/@Quiet[cpus/abs],
					StringForm[tr["StatusKernelsElapsed"], NumberForm[nanFix[abs],{9,3}], NumberForm[nanFix[sum/abs],{3,2}]]];
				col3=Join[{tr["StatusKernelsAct"],Null},
						Style["\[FilledCircle]",FontColor->Dynamic[If[EvaluationCount[#]>0,Red,Darker[Green]]]]& /@ kernels];
				contents=Transpose[Join[Transpose[contents],{col3,col1,col2}]];
				aligns = Join[aligns, {Center, Right, Left}];
				tips = Join[tips, {tr["StatusKernelsDescription_Act"],tr["StatusKernelsDescription_Time"],tr["StatusKernelsDescription_Elapsed"]}];
		  ]]]],
		  Throw[$Failed, problemo] (* aborted: try again later *)
		];
		locd = Length[localColumns]+2; remd = locd + Length[remoteColumns]; cond = remd + Length[controlColumns];
		contents[[1]] = MapThread[Item[Tooltip[Style[#1,Bold],#3,tooltipOptions],Alignment->{#2,Baseline}]&,{contents[[1]], aligns /. "." -> Right, tips}];
		If[ ArrayDepth[contents] < 2, Throw[$Failed, problemo]];
		errors=0;
		oldstate = Grid[contents, ItemSize->Full, Frame->True,
								Dividers->{{2->True, locd->Gray, remd->True, cond->True}, {2->True, 3->Gray}}, Alignment->{aligns}];
	   , problemo, ( (* catch errors; dont' touch oldstate *)
			If[++errors<5, enabled = True]; (* try again *)
			#)&
	  ];
	  oldstate],
	!enabled && $kernelsIdle, (* avoid spinning *)
		enabled = True;
		oldstate,
	True,
		oldstate
]

`oldstate = Grid[{{}}]
`enabled = True; (* software flip/flop to avoid continuous updates *)
`oldcpus = Parallel`Debug`Perfmon`subCPUs; (* refresh trigger *)

(* this dynamic is tricky, we need to avoid evaluation during times when subkernels are busy *)
statusGrid[] := DynamicModule[{enabled=enabled,oldcpus={},oldKernelCount=0,oldstate=oldstate},
	Dynamic[Refresh[ doRefresh; propGrid[enabled,oldcpus,oldKernelCount,oldstate],
			TrackedSymbols:>{$kernels,$kernelsIdle,doRefresh,Parallel`Debug`Perfmon`subCPUs,localColumns,remoteColumns,controlColumns,perfcol,perfAuto}],
			SynchronousUpdating->True]
]

(* ancillary info *)

startstopButton[] :=
	Button[Dynamic[If[$KernelCount === 0, tr["StatusKernelsLaunchAll"], tr["StatusKernelsCloseAll"]]], buttonStartStop[],
		Method -> "Queued", Enabled->Dynamic[!launching], ImageSize -> All]

columnsButton[] :=
	Button[tr["StatusKernelsSelectColumns"], buttonColumns[], ImageSize->All]

configureButton[] :=
	Button[tr["StatusKernelsKernelConfiguration"], Parallel`Palette`buttonConfigure[], ImageSize->All]

(* autolaunching hacks *)

protectedLaunch[] := Block[{Parallel`Static`$launchFeedback=False},
	launching=True; Parallel`Kernels`Private`resetFeedback; CheckAbort[LaunchKernels[], Null]; launching=False]

buttonStartStop[] := If[ $KernelCount==0, protectedLaunch[], CloseKernels[] ]

`shouldAutolaunch=False;

SetAttributes[autoLaunchTrigger, HoldAll]
autoLaunchTrigger[doit_] := Dynamic[
	If[ doit && $KernelCount==0, doit=False; FinishDynamic[]; protectedLaunch[] ];"",
	TrackedSymbols :> {doit}, SynchronousUpdating->False, ImageSizeCache -> {0., {0.,7.}}
]


`doRefresh = 0; (* to trigger a refresh, change this *)

(* inline status display; do not autoload during rendering the initial contents. *)

KernelStatus[] := Module[{},
	If[ Parallel`Kernels`Private`autolaunchActive, shouldAutolaunch = True; clearAutolaunch[] ];
	revertColumns;
	Deploy[Panel[
	  Grid[{
	  	{Item[kernelSummary[], Alignment->Left], SpanFromLeft},
		{Pane[statusGrid[], Scrollbars->Automatic], SpanFromLeft},
		{Item[Row[{startstopButton[], autoLaunchTrigger[shouldAutolaunch]}], Alignment->Left], 
		 Item[Row[{columnsButton[], configureButton[]}], Alignment->Right]}
	  },Spacings->{0,0.5}, ItemSize->Full]
	]]
]


(* column chooser *)

(* subset checkboxes *)
SetAttributes[check, HoldFirst]
check[set_, i_] := Checkbox[Dynamic[MemberQ[set,i], (set = If[#, Union[set,{i}], Complement[set, {i}]])&]]


`columnNB = Null
destroyCNB := (NotebookClose[columnNB]; columnNB=Null;)
colHeading[stuff_] := {{" ", SpanFromLeft, SpanFromLeft},
	{Style[stuff, Bold], SpanFromLeft, SpanFromLeft}}

buttonColumns[] := Module[{dimperf = !Parallel`Debug`$Debug},
	If[columnNB=!=Null, destroyCNB ];
	(* create and display *)
	columnNB = CreateDialog[Column[{
		Grid[Join[
		colHeading[tr["LocalProperties"]],
		Table[{check[localColumns,i], localStatus[[i,1]], localStatus[[i,3]]}, {i, Length[localStatus]}],
		colHeading[tr["RemoteProperties"]],
		Table[{check[remoteColumns,i], remoteStatus[[i,1]], remoteStatus[[i,3]]}, {i, Length[remoteStatus]}],
		colHeading[tr["ActionsProperties"]],
		Table[{check[controlColumns,i], controlStatus[[i,1]], controlStatus[[i,3]]}, {i, Length[controlStatus]}],
		colHeading[ If[dimperf, Tooltip[tr["PerformanceDisabled"],tr["PerformanceTip"]], tr["Performance"]] ],
		 {{Checkbox[Dynamic[perfcol]], tr["PerformanceName"], tr["PerformanceDescription"]}}
		], Alignment->Left, ItemStyle->{LineBreakWithin->False}],
		" ",
		Row[{
			Button[tr["StatusOK"], saveColumns;destroyCNB;doRefresh++, Enabled->True],
			Button[tr["StatusRevert"], revertColumns, Enabled->Dynamic[dirtyColumns]],
			Button[tr["StatusDefaults"], factoryColumns, Enabled->True],
			Button[tr["StatusCancel"], revertColumns;destroyCNB, Enabled->True ]
		}]
	  }],
		WindowTitle->tr["ColumnsTitle"]
	];
]

End[]

Protect[ KernelStatus ]

EndPackage[]
