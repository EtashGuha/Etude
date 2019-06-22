Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["$GPUCount"]
PackageExport["$HasGPU"]

(* currently there is no better way of getting the number of available devices than 
to brute force allocate a bunch of small NDArrays. please update if this ever 
changes. *)

mxlDeclare[mxlMXGetGPUCount, {}, "Integer"]

$GPUCount := With[{count = Quiet @ Replace[mxlMXGetGPUCount[], Except[_Integer] -> 0]},
	If[TrueQ[count > 0], $GPUCount = count];
	count
];

$HasGPU := With[{hasGPU = TrueQ[$GPUCount > 0]},
	If[hasGPU, $HasGPU = hasGPU];
	hasGPU
];

(******************************************************************************)

mxlDeclare[mxlMXGetGPUMemoryInformation, "Integer", "IntegerVector"]

PackageExport["GetGPUMemoryInformation"]

GetGPUMemoryInformation[n_Integer] := 
	If[1 <= n <= $GPUCount, mxlCall[mxlMXGetGPUMemoryInformation, n - 1], $Failed];

(******************************************************************************)

PackageExport["GetSMIToolInformation"]

GetSMIToolInformation[] := Scope[
	res = StreamExecute["!nvidia-smi --query-gpu=\"name,memory.total,memory.free,temperature.gpu,power.draw,utilization.gpu\" --format=csv,noheader,nounits"];
	If[!StringQ[res], GetSMIToolInformation[] := $Failed; Return[$Failed]];
	res = StringTrim[res];
	If[res === "", Return[<||>]];
	<|"Name" -> #1,
	  "TotalMemory" -> FromDigits[#2]*1000000,
	  "FreeMemory" -> FromDigits[#3]*1000000,
	  "Temperature" -> FromDigits[#4],
	  "PowerUsage" -> Quiet @ Check[ToExpression[#5], None],
	  "Utilization" -> FromDigits[#6]
	|>& @@@ Map[StringTrim, StringSplit[StringSplit[res, "\n"], ","]]
];

(******************************************************************************)

DeclarePostloadCode[
General::badtrgdevgpu = "TargetDevice -> `` could not be used. Please ensure that you have a compatible NVIDIA graphics card and have installed the latest drivers from ``.";
General::trgdevdrv = "TargetDevice -> `1` could not be used; your current NVIDIA driver version is `2`, you should update to version `3` or greater. Please download the latest NVIDIA driver for your operating system from `4`.";
General::trgdevdrvmac = "TargetDevice -> `` could not be used. Please download the latest CUDA drivers for your operating system from ``.";
General::trgdevnogpu = "TargetDevice -> `` could not be used; your system does not appear to have a supported NVIDIA GPU.";
General::trgdevgpumax = "TargetDevice -> `` could not be used; your system appears to have only `` NVIDIA GPU(s).";
General::trgdevegpu = "TargetDevice -> `` could not be used; a supported NVIDIA GPU was not found. If you are using an external GPU, ensure it is attached and recognized by your OS; restarting your system may help. Additionally, check that you are using the latest drivers from ``.";
]

(******************************************************************************)

PackageExport["ThrowTargetDeviceFailure"]

ThrowTargetDeviceFailure[tgt_] := Scope[

	If[$OperatingSystem == "MacOSX",
		version = GetSystemCUDAVersion[];
		If[FailureQ[version], ThrowFailure["trgdevnogpu", tgt]];
		If[Order[$MinimumCUDAVersion, version] == -1,
			ThrowFailure["trgdevdrvmac", tgt, $driverDownloadURL]];
	,
		version = GetNVIDIADriverVersion[];
		If[FailureQ[version], ThrowFailure["trgdevnogpu", tgt]];
		If[version =!= Indeterminate && Order[$MinimumNVIDIADriverVersion, version] == -1,
			currVersion = StringRiffle[version, "."];
			minVersion = StringRiffle[$MinimumNVIDIADriverVersion, "."];
			ThrowFailure["trgdevdrv", tgt, currVersion, minVersion, $driverDownloadURL]];
	];

	err = Quiet @ CatchFailure[General, MXGetLastError[]];
	If[StringQ[err] && StringContainsQ[err, "insufficient for CUDA runtime"],
		ThrowFailure["trgdevegpu", tgt, $driverDownloadURL]];

	(* this fallback should basically never happen *)
	ThrowFailure["badtrgdevgpu", tgt, $driverDownloadURL];
];

$driverDownloadURL := If[$OperatingSystem == "MacOSX",
	"http://www.nvidia.com/object/mac-driver-archive.html",
	"http://www.nvidia.com/Download/index.aspx"
];

(******************************************************************************)

PackageExport["GetNVIDIADriverVersion"]

(* GetNVIDIADriverVersion returns $Failed if there is definitely no GPU,
a pair of {major,minor} if it can determine the driver version, and
and Indeterminate if it can't determine the driver version. *)

GetNVIDIADriverVersion[] /; $OperatingSystem === "Windows" := Scope[
	version = getGPUToolsMajorMinorRev[GPUTools`Internal`$NVIDIADriverLibraryVersion];
	If[FailureQ[version], Return[$Failed]];
	rev = Last[version];
	float = Mod[rev * 100, 1000];
	major = IntegerPart[float];
	If[Round[float] == float,
		minor = 0,
		minor = FromDigits @ Last @ StringSplit[ToString[float], "."];
	];
	{major, minor}
];

(*
GetNVIDIADriverVersion[] /; $OperatingSystem === "Windows" := Scope[
	(* this takes about 0.25 seconds, but only occurs on failure, so its okay *)
	res = Quiet @ StreamExecute["!reg query \"HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\" /s"];
	If[!StringQ[res], Return @ Indeterminate];
	strs = Flatten @ StringCases[res, "Display.Driver/" ~~ ns : NumberString :> ns];
	If[strs === {}, Return @ Indeterminate];
	versions = Map[Map[FromDigits, StringSplit[#, "."]]&, strs];
	Last @ Sort @ versions
];
*)

GetNVIDIADriverVersion[] /; $OperatingSystem === "Unix" := 
	toMajorMinor @ getGPUToolsMajorMinorRev[GPUTools`Internal`$NVIDIADriverLibraryVersion];

GetNVIDIADriverVersion[] /; $OperatingSystem === "MacOSX" := 
	Indeterminate; (* on Mac, we have no idea *)

(******************************************************************************)

PackageExport["GetSystemCUDAVersion"]

GetSystemCUDAVersion[] /; $OperatingSystem === "MacOSX" :=
	toMajorMinor @ getGPUToolsMajorMinorRev[GPUTools`Internal`$CUDALibraryVersion];

GetSystemCUDAVersion[] /; $OperatingSystem =!= "MacOSX" :=
	Indeterminate;

SetAttributes[getGPUToolsMajorMinorRev, HoldFirst];
getGPUToolsMajorMinorRev[var_] := Scope[
	Block[{$ContextPath = {"System`"}}, Needs["GPUTools`"]; info = var];
	If[!AssociationQ[info], Return[$Failed]];
	version = Lookup[info, {"MajorVersion", "MinorVersion", "RevisionNumber"}];
	If[!MatchQ[version, {_Integer, _Integer, _Integer | _Real}], Return[$Failed]];
	version
];

toMajorMinor[{a_, b_, c_}] := {a, b};
toMajorMinor[$Failed] := $Failed;