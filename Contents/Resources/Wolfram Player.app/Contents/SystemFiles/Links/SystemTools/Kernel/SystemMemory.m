(* ::Package:: *)

(* ::Section:: *)
(*SystemMemory*)


(* ::Subsection:: *)
(*Package Namespace and Initialization*)


BeginPackage["SystemTools`"];

System`MemoryAvailable;

Begin["`Private`"];

Unprotect[MemoryAvailable]

(* ::Subsection:: *)
(*LibraryLink Loading*)


If[!StringQ[$libraryPath],

	$libraryResources = FileNameJoin[{ParentDirectory @ DirectoryName @ $InputFileName, "LibraryResources", $SystemID}];
	$libraryPath = First[FileNames["libSystemTools.*", $libraryResources], $Failed];
	(* ^ this is faster than FindLibrary, which takes 15 milliseconds *)
	If[FailureQ[$libraryPath], $libraryPath = FindLibrary["libSystemTools"]];

	If[$libraryPath === $Failed,
		Message[LibraryFunction::load, "libSystemTools"];
	,
		$RunningProcesses = {};
		If[LibraryLoad[$libraryPath] === $Failed, 
			Message[LibraryFunction::load, $libraryPath]];
	];
];

getMemoryAll = LibraryFunctionLoad[$libraryPath, "getMemoryAll", {}, {Real, 1}];
getMemoryAvailable = LibraryFunctionLoad[$libraryPath, "getMemoryAvailable", {}, Real];


MemoryAvailable[] := Refresh[checkCloudMemoryLimit@IntegerPart@getMemoryAvailable[], UpdateInterval->1]


(* ::Subsection:: *)
(*Formatting*)


findByteUnit[bytes_Integer]:=Block[{QuantityUnits`$AutomaticUnitTimes = False,i,k=1024,sizes={"Bytes","Kibibytes","Mebibytes","Gibibytes","Tebibytes","Pebibytes","Exbibytes","Zebibytes","Yobibytes"}},
	If[bytes==0,Return[Quantity[bytes, "Bytes"]];];
	If[bytes >= k^9, Return[Quantity[N[bytes/k^8], "Yobibytes"]]];
	i=Floor[Log[k,bytes]];
	Quantity[N[bytes/k^i], sizes[[i+1]]]
];

If[TrueQ[$CloudEvaluation],

	savedMemoryLimit = None;

	unitMultiplyer[units_] := Replace[StringTrim[units], {"KB" -> 10^3, "MB" -> 10^6, "GB" -> 10^9, "TB" -> 10^12}];

	findMemLimit[planInfo_] := 	Block[
		{featureInfo, feature, pos, limitInfo, limitValue},
		featureInfo = Lookup[planInfo, "planFeatureLimitsInfo"];
		pos = Position[featureInfo, {_, _ -> {___, "shortName" -> "sessionmem", ___}}];
		feature = Extract[featureInfo, pos];
		limitInfo = Association[Lookup[feature, "cloudPlanFeatureInfo"]];
		limitValue = Lookup[feature, "limitValue"];
		Interpreter["Number"][limitValue]*unitMultiplyer[limitInfo["unit"]]
	];

	(* Using same API as CloudAccountData[] *)
	getCloudMemoryLimit[] := Block[
		{rawBytes, json, data, subscriptions, plans, limit, $CloudBase = CloudObject`Private`handleCBase[Automatic]},
		If[NumberQ[savedMemoryLimit], Return[savedMemoryLimit]];
		rawBytes = CloudObject`Private`execute[$CloudBase, "GET", {"REST", "user", "subscriptions"}];
		If[MatchQ[rawBytes, {_, _List}],
			json = FromCharacterCode[Last[rawBytes]]
			,
			Return[$Failed]
		];
		data = CloudObject`Private`importFromJSON[json];
		subscriptions = Lookup[data, "subscriptions", {}];
		plans = Lookup[#, "planInfo"] & /@ subscriptions;
		limit = Max[findMemLimit[#] & /@ plans];
		If[! NumberQ[limit], Return[$Failed]];
		savedMemoryLimit = limit;
		limit
	];

	checkCloudMemoryLimit[memAvail_] := Block[{cloudLimit, cloudRemaining},
		cloudLimit = Quiet[getCloudMemoryLimit[]];
		cloudRemaining = cloudLimit - MemoryInUse[];
		If[!NumberQ[cloudRemaining], Return[memAvail]];
		If[cloudRemaining < 0, cloudRemaining = 0];
		Min[{cloudRemaining, memAvail}]
	];
,
	checkCloudMemoryLimit[memAvail_] := memAvail;
];


(* ::Subsection:: *)
(*Keys and Names*)


(*Indicates if value should be formatted as Bytes.*)
$GeneralFormatAsBytes = {True, True, True, True, True, True, True, True, True, True, True}
$MemoryNames = {
	"MemoryAvailable",
	"PhysicalUsed",
	"PhysicalFree",
	"PhysicalTotal",
	"VirtualUsed",
	"VirtualFree",
	"VirtualTotal",
	"PageSize",
	"PageUsed",
	"PageFree",
	"PageTotal"
}

$FormatAsBytes/;$OperatingSystem==="Windows" = Join[$GeneralFormatAsBytes, {True, True, True, True, True}];
$SystemSpecific/;$OperatingSystem==="Windows" = {
	"CommitPeak",
	"SystemCache",
	"WindowsKernel",
	"WindowsKernelPaged",
	"WindowsKernelTotal"
};

$FormatAsBytes/;$OperatingSystem==="MacOSX" = Join[$GeneralFormatAsBytes, {True, True, True, True, True, False, False}];
$SystemSpecific/;$OperatingSystem==="MacOSX" = {
	"AppMemory",
	"Wired",
	"Active",
	"Inactive",
	"Compressed",
	"PageIns",
	"PageOuts"
};

$FormatAsBytes/;$OperatingSystem==="Unix" = Join[$GeneralFormatAsBytes,{True, True, True, True, True}];
$SystemSpecific/;$OperatingSystem==="Unix" = {
	"Active",
	"Inactive",
	"Cached",
	"Buffers",
	"SwapReclaimable"
};



(* ::Subsection:: *)
(*SystemTools`Private`getSystemMemory[]*)


(*This prevents rerunning when several values are queried at once.*)
runRecently[] := TrueQ@((AbsoluteTime[]-$LastChange) < .1)
getSystemMemory[]:= (Block[{$CheckedCache = True},
	If[!runRecently[],
		$LastChange = AbsoluteTime[];
		$MemoryCached=getSystemMemory[]
	];
	$MemoryCached
])/;($CheckedCache =!= True)

getSystemMemory[key___]:= (Block[{$CheckedCache = True},
	If[!runRecently[],
		$LastChange = AbsoluteTime[];
		$MemoryCached=getSystemMemory[]
	];
	$MemoryCached[key]
])/;($CheckedCache =!= True)

getSystemMemory[] := Block[{info, formatted, names, amount},
	info = IntegerPart@getMemoryAll[];
	names = Join[$MemoryNames, $SystemSpecific];
	amount = Min[Length[info], Length[names], Length[$FormatAsBytes]];
	info[[1]] = checkCloudMemoryLimit[info[[1]]];
	formatted = MapThread[If[#1, findByteUnit[#2], #2, #2]&, {$FormatAsBytes[[;;amount]], info[[;;amount]]}];
	Association[
		Sequence[
			MapThread[#1->#2&, {
				names[[;;amount]],
				formatted[[;;amount]]
			}]
		]
	]
]

getSystemMemory[key__] := getSystemMemory[][key]


SetAttributes[{MemoryAvailable},{Protected, ReadProtected}];


End[];

EndPackage[];
