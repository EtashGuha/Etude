(* :Title: Preferences.m -- built-in defaults of Parallel Tools preferences *)

(* :Context: none *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   declaration of all preferences. Evaluated inside a dynamic binding for addPreference[].
   The cautious user will always fully qualify symbols appearing here.
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 7.0.1 *)

(* :History:
   1.0 for PCT 3.0
   1.0.1 for PT 3.0.1
*)

(* note that during reading of this file $Context is Parallel`Preferences`
   and $ContextPath is System`; and that tr[] works for localized strings *)


addPreference["Version" -> 3.01] (* for possible migration; see Parallel`Palette` *)
addPreference["CVSRevision" -> StringReplace["$Revision$", {"$"->"", " "->"", "Revision:"->""}]]

addPreference["LocalStatus" -> Hold[
	{tr["LocalPropertiesName_KernelName"],	Parallel`Developer`KernelName,	tr["LocalPropertiesDescription_KernelName"], Left},
	{tr["LocalPropertiesName_Machine"],		System`MachineName,				tr["LocalPropertiesDescription_Machine"], Left},
	{tr["LocalPropertiesName_Speed"],			System`KernelSpeed,				tr["LocalPropertiesDescription_Speed"], "."},
	{tr["LocalPropertiesName_Subkernel"],		Parallel`Developer`SubKernel,	tr["LocalPropertiesDescription_Subkernel"], Left},
	{tr["LocalPropertiesName_Link"],			System`LinkObject,				tr["LocalPropertiesDescription_Link"], Left},
	{tr["LocalPropertiesName_Type"],			Function[SubKernels`SubKernelType[Parallel`Developer`SubKernel[#]][SubKernels`Protected`subName]], tr["LocalPropertiesDescription_Type"], Left}
]]
addPreference["LocalMasterStatus" -> Hold[
	"master", $MachineName, 1, "N/A", $ParentLink, "Master Kernel"
]]

addPreference["LocalColumns" -> {1}]

addPreference["RemoteStatus" -> Hold[
	{tr["RemotePropertiesName_ID"],			$KernelID,				tr["RemotePropertiesDescription_ID"], Right},
	{tr["RemotePropertiesName_Host"],			$MachineName,			tr["RemotePropertiesDescription_Host"], Left},
	{tr["RemotePropertiesName_Process"],		$ProcessID,				tr["RemotePropertiesDescription_Process"], Right},
	{tr["RemotePropertiesName_CPU"],			NumberForm[TimeUsed[],{9,3}],	tr["RemotePropertiesDescription_CPU"], Right},
	{tr["RemotePropertiesName_RAM"],			StringForm["`1`M", Round[MemoryInUse[]/10^6]],		tr["RemotePropertiesDescription_RAM"], Right},
	{tr["RemotePropertiesName_maxRAM"],		StringForm["`1`M", Round[MaxMemoryUsed[]/10^6]],	tr["RemotePropertiesDescription_maxRAM"], Right},
	{tr["RemotePropertiesName_Cores"],		$ProcessorCount,		tr["RemotePropertiesDescription_Cores"], Right},
	{tr["RemotePropertiesName_Priority"],		"ProcessPriority" /. SystemOptions["ProcessPriority"],	tr["RemotePropertiesDescription_Priority"], Left},
	{tr["RemotePropertiesName_Mathematica"],	$Version,				tr["RemotePropertiesDescription_Mathematica"], Left},
	{tr["RemotePropertiesName_Version"],		PaddedForm[$VersionNumber,{3,1}],	tr["RemotePropertiesDescription_Version"], Center},
	{tr["RemotePropertiesName_SystemID"],		$SystemID,				tr["RemotePropertiesDescription_SystemID"], Left},
	{tr["RemotePropertiesName_MachineID"],	$MachineID,				tr["RemotePropertiesDescription_MachineID"], Left},
	{tr["RemotePropertiesName_ParentLink"],	$ParentLink,			tr["RemotePropertiesDescription_ParentLink"], Left}
]]

addPreference["RemoteColumns" -> {2,3,4,5,10}]

(* control columns; should turn into buttons, applied to kernel object *)

addPreference["ControlStatus" -> Hold[
	{tr["ActionsName_Close"],		Button["X", CloseKernels[#], ImageSize->{Automatic,Small}]&,	tr["ActionsDescription_Close"], Center}
]]

addPreference["ControlMasterStatus" -> Hold[
	Null&
]]

addPreference["ControlColumns" -> {1}]

(* collect performance data: False|True *)

addPreference["PerformanceData" -> True]


(* known subkernel implementations; #[[5]]==False hides them *)

addPreference[Parallel`Palette`paletteConfig["knownImplementations"] -> {
	{"SubKernels`LocalKernels`",	tr["LocalKernelsName"],		tr["LocalKernelsDescription"],     "paclet:ParallelTools/tutorial/ConnectionMethods#387739223"},
	{"LightweightGridClient`",		tr["RemoteServicesName"],	tr["RemoteServicesDescription"], "paclet:ParallelTools/tutorial/ConnectionMethods#25777216"},
	{"ClusterIntegration`",			tr["ClusterIntegrationName"], tr["ClusterIntegrationDescription"], "paclet:ParallelTools/tutorial/ConnectionMethods#683809407"},
	{"SubKernels`RemoteKernels`",	tr["RemoteKernelsName"],	tr["RemoteKernelsDescription"],   "paclet:ParallelTools/tutorial/ConnectionMethods#248366312"},
	{"SubKernels`ClusterKernels`",	tr["ClusterKernelsName"],	tr["ClusterKernelsDescription"], None, False  },
Nothing}]


(* default profile *)

addPreference[Parallel`Palette`paletteConfig["enabledImplementations"] -> {
	"SubKernels`LocalKernels`"
}]

addPreference[Parallel`Palette`paletteConfig["PCT"] -> {Automatic, True, True, "Retry", True} ]

addPreference[Parallel`Palette`paletteConfig["Local Kernels"] -> {} ]

addPreference[debugPreference[Automatic] -> True ]

(* batch profile *)

addPreference[Parallel`Palette`paletteConfig["enabledImplementations", "Batch"] -> {
	"SubKernels`LocalKernels`"
}]

addPreference[Parallel`Palette`paletteConfig["PCT", "Batch"] -> {Automatic, False, False, "Retry", True} ]

addPreference[Parallel`Palette`paletteConfig["Local Kernels", "Batch"] -> {} ]

addPreference[debugPreference["Batch"] -> False ]

(* PlayerPro profile *)

addPreference[Parallel`Palette`paletteConfig["PCT", "PlayerPro"] -> {Automatic, False, True, "Retry", False} ]

addPreference[debugPreference["PlayerPro"] -> False ]

(* Player profile for EnterpriseCDF *)

addPreference[Parallel`Palette`paletteConfig["PCT", "PlayerEnterprise"] -> {Automatic, False, True, "Retry", False} ]

addPreference[debugPreference["PlayerEnterprise"] -> False ]

addPreference[Parallel`Palette`paletteConfig["Local Kernels", "PlayerEnterprise"] -> {"Limit"->8, "UseLicense"->False} ]

(* Player profile for FreeCDF *)

addPreference[Parallel`Palette`paletteConfig["PCT", "Player"] -> {Automatic, False, True, "Retry", False} ]

addPreference[debugPreference["Player"] -> False ]

addPreference[Parallel`Palette`paletteConfig["Local Kernels", "Player"] -> {"Limit"->4, "UseLicense"->False} ]

(* Private cloud localhost profile *)

addPreference[Parallel`Palette`paletteConfig["PCT", "CloudLocalhost"] -> {Automatic, False, True, "Retry", False} ]

addPreference[debugPreference["CloudLocalhost"] -> False ]

addPreference[Parallel`Palette`paletteConfig["enabledImplementations", "CloudLocalhost"] -> {
	"SubKernels`LocalKernels`"
}]

addPreference[Parallel`Palette`paletteConfig["Local Kernels", "CloudLocalhost"] -> {"UseLicense"->False, "UseLimit"->False} ]

(* Private cloud cluster profile *)

addPreference[Parallel`Palette`paletteConfig["PCT", "CloudCluster"] -> {Automatic, False, True, "Retry", False} ]

addPreference[debugPreference["CloudCluster"] -> False ]

addPreference[Parallel`Palette`paletteConfig["enabledImplementations", "CloudCluster"] -> {
	"SubKernels`ClusterKernels`"
}]
