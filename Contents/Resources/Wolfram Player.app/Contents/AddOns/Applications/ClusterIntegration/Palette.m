(* :Name: Palette.m *)

(* :Title: Palettes and interface modules *)

(* :Context: ClusterIntegration`Palette` *)

(* :Author: Charles Pooh *)

(* :Summary: This package provides internal tools for CIP *)

(* :Copyright: (c) 2008 Wolfram Research, Inc. *)

(* :Sources: *)

(* :Mathematica Version: 7.0 *)

(* :History: *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: *)

(* :Discussion: *)

(* :Requirements: *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`Palette`", "ClusterIntegration`"]


(* data type *)

ComputeKernels::usage = "ComputeKernels[...] represents configurations for \
compute kernels on a cluster or grid."


(* ParallelTools plugin modules *)

CIPPalette::usage = "Integrated palette for ClusterIntegration."
CIPConfiguredClusters::usage = "Active cluster settings for ClusterIntegration."
CIPGetConfiguration::usage = "Get configuration for ClusterIntegration."
CIPSetConfiguration::usage = "Set configuration for ClusterIntegration."


Begin["`Private`"]


(* Loading resources *)

Needs["ResourceLocator`"]

$packageRoot = DirectoryName[System`Private`$InputFileName]
textFunction = TextResourceLoad["ClusterIntegration", $packageRoot]


(* ************************************************************************* **

                           Compute Kernels

   Comments:

     ComputeKernels is used internally to encapsulate launch requests
     that are passed to the ParallelTools palette. NewKernels[...]
     is called to launch kernels within the ParallelTools palette.

   ToDo:

** ************************************************************************* **)


ComputeKernels /: SubKernels`NewKernels[ComputeKernels[engine_, name_, kcount_, args__, _]] :=
    SubKernels`NewKernels[engine[name, kcount, args]]


ComputeKernels /: SubKernels`KernelCount[ComputeKernels[_, _, kcount_, __]] :=
    kcount


(* format *)

Format[ComputeKernels[_, name_String, n : 1, __]] :=
    StringForm[textFunction["SingleComputeKernel"], n, name]

Format[ComputeKernels[_, name_String, n_Integer, __]] :=
    StringForm[textFunction["MultipleComputeKernels"], n, name]


(* ************************************************************************* **

                        Plugin modules for ParallelTools

   Comments:

     data and index are used to build module type variables. This is needed to
     workaround untrackable module variables issue in Dynamic.

   ToDo:

** ************************************************************************* *)


$availableEngineNames = {
    CCS -> textFunction["CCSUsage"], HPC -> textFunction["HPCUsage"],
    LSF -> textFunction["LSFUsage"], PBS -> textFunction["PBSUsage"],
    SGE -> textFunction["SGEUsage"]
}


$selectedCluster = Null

$selectedEngine := engine[$selectedCluster]

$selectedName := descr[$selectedCluster]

$selectedMachineName := cluster[$selectedCluster]

$selectedKernels := kernelCount[$selectedCluster]

$selectedOnOff := selectedFlag[$selectedCluster]

$selectedOptions := parametersUpdate[$selectedCluster]

$selectedRemoteMaster := remoteMaster[$selectedCluster]


$configuredClusters = {}

index = 0


(* window parameters *)

$CIPTabWidth = Parallel`Palette`Private`tabwidth

$CIPMainMenuWidth = Scaled[0.97]

$CIPConfigMenuWidth = Scaled[0.40]

$CIPSettingsMenuWitdth = Scaled[0.55]

$CIPClusterNamesSize = Scaled[1]

$CIPMainFieldsSize = Scaled[0.98]

$CIPParametersInputFieldsSize = Scaled[0.9]


(* utilities *)

inputFieldType[_String] := String

inputFieldType[_?NumericQ] := Number

inputFieldType[___] := Expression


$formattedOptions =
    {
        "Duration" -> textFunction["Duration"],
        "EnginePath" -> textFunction["EnginePath"],
        "KernelProgram" -> textFunction["KernelProgram"],
        "KernelOptions" -> textFunction["KernelOptions"],
        "NativeSpecification" -> textFunction["NativeSpecification"],
        "NetworkInterface" -> textFunction["NetworkInterface"],
        "ToQueue" -> textFunction["ToQueue"]
    }


$platform = $OperatingSystem


(* ------------------------------------------------------------------------- --

                          Cluster Configuration

   Comments:

     ConfigurationCluster is used internally to store cluster configuration
     use in the palette.

     ConfigurationCluster[cl[link, descr, arglist, speed, cluster, jobID, taskID, obj]]
         link       associated link object
         descr      cluster name
         arglist    list of arguments of the form "MathCommand" -> cl[data[..]]
         cluster    host name
         engine     engine name (CCS, LSF, PBS, SGE, ...)
         jobID      job identifier
         taskID     task identifier
         obj        cluster engine object or stream


   ToDo:

-- ------------------------------------------------------------------------- **)


(* data type *)

`ConfigurationCluster
`cl
`data

SetAttributes[ConfigurationCluster, HoldAll]
SetAttributes[cl, HoldAllComplete]


(* constructor *)

NewConfigurationCluster[] :=
    NewConfigurationCluster[textFunction["NewCluster"],
        {"KernelProgram" -> "MathKernel",
         "KernelOptions" -> "-mathlink -noicon",
         "NetworkInterface" -> ""
        },
        "", Null, False, 1, False]


NewConfigurationCluster[idescr_, iarglist_, icluster_, iengine_, iflag_, ikcount_, imaster_] :=
    Block[{res, args},

        args = Table[data[++index], {Length[iarglist]}];

        res = With[{descr = data[++index], arglist = data[++index],
                    cluster = data[++index], engine = data[++index],
                    flag = data[++index], kcount = data[++index],
                    master = data[++index], args = args},
            descr = idescr;
            arglist = Thread[Rule[iarglist[[All, 1]], cl /@ args]]; (* cl will be replaced by Dynamic later *)
            args = iarglist[[All, 2]];
            cluster = icluster;
            engine = iengine;
            flag = iflag;
            kcount = ikcount;
            master = imaster;
            ConfigurationCluster[cl[descr, arglist, cluster, engine, flag, kcount, master]]
        ];

        AppendTo[$configuredClusters, res];
        res

  ]


(* basic functions *)

descr[ConfigurationCluster[cl[descr_, ___], ___]] := Dynamic[descr]

arglist[ConfigurationCluster[cl[descr_, arglist_, ___], ___]] := arglist /. cl -> Dynamic

cluster[ConfigurationCluster[cl[descr_, arglist_, cluster_, ___], ___]] := Dynamic[cluster]

engine[ConfigurationCluster[cl[descr_, arglist__, cluster_, engine_,  ___], ___]] := Dynamic[engine]

selectedFlag[ConfigurationCluster[cl[descr_, arglist_, cluster_, engine_, flag_, ___], ___]] := Dynamic[flag]

selectedQ[ConfigurationCluster[cl[descr_, arglist_, cluster_, engine_, flag_, ___], ___]] := TrueQ[flag]

kernelCount[ConfigurationCluster[cl[descr_, arglist_, cluster_, engine_, flag_, kcount_, ___], ___]] := Dynamic[kcount]

remoteMaster[ConfigurationCluster[cl[descr_, arglist_, cluster_, engine_, flag_, kcount_, master_, ___], ___]] := Dynamic[master]

decrKernel[ConfigurationCluster[cl[descr_, arglist_, cluster_, engine_, flag_, kcount_, ___], ___]] := If[kcount > 1, kcount--]

incrKernel[ConfigurationCluster[cl[descr_, arglist_, cluster_, engine_, flag_, kcount_, ___], ___]] := (kcount++)


(* methods *)

parametersUpdate[clust:ConfigurationCluster[cl[_, argl_, ___]]] :=
    Block[{res, res1, res2, copts, opts},

        copts = arglist[clust];

        res1 = engine[clust] //. Dynamic -> Identity;
        If[res1 === Null, Return[copts]];

        res = copts //. Dynamic -> Identity;
        opts = DeleteCases[Options[res1], "ScriptFile" -> _];
        res2 = opts[[All, 1]];
        If[res[[All, 1]] === res2, Return[copts]];

        res = Select[copts, MemberQ[res2, First[#]] &];
        res = Union[res, opts, SameTest -> (First[#1] === First[#2] &)];

        argl = If[MatchQ[#2, Dynamic[data[_]]],
                  #1 -> (#2 /. Dynamic -> cl),
                  With[{u = data[++index]}, u = #2; #1 -> cl[u]]
               ] & @@@ res

  ]


toComputeKernels[clust_ConfigurationCluster] /;
((engine[clust] /. Dynamic -> Identity) =!= Null) :=
    ComputeKernels[engine[clust], cluster[clust],
        kernelCount[clust], arglist[clust], {descr[clust],
            selectedFlag[clust], remoteMaster[clust]}] /. Dynamic -> Identity


toComputeKernels[___] := $Failed


(* ------------------------------------------------------------------------- --

                        Cluster Integration Palette

   Comments:

   ToDo:

-- ------------------------------------------------------------------------- **)


(* palette *)

CIPPalette[] := Panel[
    Grid[{{functionsMenu[], SpanFromLeft}, {configuredMenu[], settingsMenu[]}},
         Frame -> None, Alignment -> Left],
    Appearance-> "Frameless",
    ImageSize -> {$CIPTabWidth, All}]


(* cluster kernels to launch *)

CIPConfiguredClusters[] :=
    Block[{res},
        res = Select[$configuredClusters, selectedQ];
        res = DeleteCases[toComputeKernels /@ res, $Failed];
        res /; ListQ[res]
    ]

CIPConfiguredClusters[___] := {}


(* get configuration *)

CIPGetConfiguration[] :=
    Block[{res},
        res = DeleteCases[toComputeKernels /@ $configuredClusters, $Failed];
        res /; ListQ[res]
    ]


(* set configuration *)

CIPSetConfiguration[{}] := (
    Clear[data];
    index = 0;
    $configuredClusters = {};
    $selectedCluster = NewConfigurationCluster[];
)


CIPSetConfiguration[cfg_List] := (
    Clear[data];
    index = 0;
    $configuredClusters = {};
    CIPSetConfiguration /@ cfg;
)


CIPSetConfiguration[clust:ComputeKernels[engine_, cluster_, kcount_, args_, {descr_, flag_, master_}]] :=
    (
      $selectedCluster = NewConfigurationCluster[descr, args, cluster, engine, flag, kcount, master];
    )


(* - - - - - - - - - - - - - -  functions menu  - - - - - - - - - - - - - - *)


labelAddCluster = Style[textFunction["AddCluster"], Bold]

addClusterButton = Switch[$platform,
    "MacOSX", Button[labelAddCluster, addClusterAction[], Appearance -> "Palette"],
    _, Mouseover[Button[labelAddCluster, Null,
        Appearance -> None,  FrameMargins -> 5],
        Button[labelAddCluster, addClusterAction[]]]
]

addClusterAction[] := ($selectedCluster = NewConfigurationCluster[])


labelRemoveCluster =  Style[textFunction["RemoveCluster"], Bold]

removeClusterButton = Switch[$platform,
    "MacOSX", Button[labelRemoveCluster, removeClusterAction[], Appearance -> "Palette"],
    _, Mouseover[Button[labelRemoveCluster, Null,
    Appearance -> None, FrameMargins -> 5],
    Button[labelRemoveCluster, removeClusterAction[]]]
]

removeClusterAction[] /; ($selectedCluster =!= Null) :=
    Block[{res},

        res = Flatten[Position[$configuredClusters, $selectedCluster, 1, 1]];
        (
          $configuredClusters = DeleteCases[$configuredClusters, $selectedCluster];
          $selectedCluster = If[$configuredClusters === {}, Null,
                                res = Min[First[res], Length[$configuredClusters]];
                                $configuredClusters[[res]]]

        ) /; Length[res] == 1

   ]


labelDuplicateCluster =  Style[textFunction["DuplicateCluster"], Bold]

duplicateClusterButton = Switch[$platform,
    "MacOSX", Button[labelDuplicateCluster, duplicateClusterAction[], Appearance -> "Palette"],
    _, Mouseover[Button[labelDuplicateCluster, Null,
    Appearance -> None, FrameMargins -> 5],
    Button[labelDuplicateCluster, duplicateClusterAction[]]]
]


duplicateClusterAction[]  /; ($selectedCluster =!= Null) :=
    (
      $selectedCluster = NewConfigurationCluster[textFunction["NewCluster"],
            arglist[$selectedCluster] /. Dynamic -> Identity,
            cluster[$selectedCluster] /. Dynamic -> Identity,
            engine[$selectedCluster] /. Dynamic -> Identity,
            selectedFlag[$selectedCluster] /. Dynamic -> Identity,
            kernelCount[$selectedCluster] /. Dynamic -> Identity,
            remoteMaster[$selectedCluster] /. Dynamic -> Identity
      ];
   )


functionsMenu[] := Switch[$platform,
    "MacOSX", Panel[Row[{addClusterButton, removeClusterButton, duplicateClusterButton},
          Spacer[5]], ImageSize -> $CIPMainMenuWidth],
    _, Panel[Row[{addClusterButton, removeClusterButton, duplicateClusterButton},
                  Spacer[5]],
             FrameMargins -> 0, Alignment -> Left,
             ImageSize -> $CIPMainMenuWidth]
]


(* - - - - - - - - - - - - configured menu - - - - - - - - - - - - - - - - - *)


$configuredClusterMenu :=
    (
      Grid[{{textFunction["TitleCluster"], textFunction["TitleKernels"], textFunction["TitleEnable"]},
              Sequence @@ (
                {
                  Setter[Dynamic[$selectedCluster], #,
                        Dynamic[Style[descr[#],
                        If[selectedFlag[#] /. Dynamic -> Identity,Bold, Gray]]],
                        ImageSize -> $CIPClusterNamesSize,
                        Appearance -> "Palette"],

                  SubKernels`Protected`Spinner[kernelCount[#]],

                  Tooltip[Checkbox[selectedFlag[#]],
                         Dynamic[If[selectedFlag[#] /. Dynamic -> Identity,
                           textFunction["Disable cluster"],
                           textFunction["Enable cluster"]]]]

                } & /@ $configuredClusters)
      },
      Alignment -> {{Left, Center, Center}, Center},
      Spacings -> {1, 1/2}]

    ) /; $selectedCluster =!= Null


$configuredClusterMenu :=
    Block[{a},
        Grid[{{textFunction["Cluster"], textFunction["Kernels"], textFunction["Enable"]},
                {Invisible[Button["", Null, ImageSize -> $CIPClusterNamesSize]],
                 Invisible[SubKernels`Protected`Spinner[Dynamic[a]]],
                 Invisible[Checkbox[Dynamic[a]]]}},
              Alignment -> {Left, Center, Center}]
    ] /; $selectedCluster === Null


configuredMenu[] := Panel[
    Dynamic[$configuredClusterMenu],
    BaselinePosition -> Top, ImageSize -> $CIPConfigMenuWidth]


(*  - - - - - - - - - - - - - - settings menu  - - - - - - - - - - - - - - - *)


$parametersMenu :=
    Pane[Column[
           Insert[Column[{#1 /. $formattedOptions,
              InputField[#2, inputFieldType[#2 /. Dynamic -> Identity],
              ImageSize -> {$CIPParametersInputFieldsSize, All},
              ContinuousAction -> True]}, Left] & @@@ $selectedOptions,
            Spacer[{1, 1}], {-1, 1, -1}]
        , Left, 1],
        AppearanceElements -> {},
        ImageSize -> {Automatic, {1, 200}},
        Scrollbars -> Automatic]



$advancedSettingsOpenerState = False


$settingsClusterMenu :=
    (
        Column[{
          Grid[{
            {textFunction["ClusterName"], InputField[$selectedName, String,
                ImageSize -> $CIPMainFieldsSize, ContinuousAction -> True]},
            {textFunction["ClusterEngine"], PopupMenu[$selectedEngine,
                $availableEngineNames, textFunction["SelectEngine"],
                ImageSize -> $CIPMainFieldsSize]},
            {textFunction["HeadNode"], InputField[$selectedMachineName, String,
                ImageSize -> $CIPMainFieldsSize, ContinuousAction -> True]}
          }, Alignment -> Left],
          Invisible[""],
          OpenerView[{Style[textFunction["AdvancedSettings"], Bold], Dynamic[$parametersMenu]},
                       Dynamic[$advancedSettingsOpenerState]],
          Invisible[""]
        }]

    ) /; $selectedCluster =!= Null


$settingsClusterMenu :=
    Column[{
        textFunction["AddMessage"],
        Hyperlink[textFunction["MoreInfo"],
            "paclet:/ParallelTools/tutorial/ConnectionMethods#683809407"]
    }] /; $selectedCluster === Null


settingsMenu[] := Panel[
    Dynamic[$settingsClusterMenu],
    BaselinePosition -> Top,
    ImageSize -> $CIPSettingsMenuWitdth]


(* ************************************************************************* *)


End[]


EndPackage[]