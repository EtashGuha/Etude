Package["NeuralNetworks`"]

PackageScope["TryPack"]

TryPack[a_ -> b_] := TryPack[a] -> TryPack[b];
TryPack[assoc_Association] := Map[TryPack, assoc];
TryPack[e:{__Rule}] := TryPack[Keys[e]] -> TryPack[Values[e]];
TryPack[e_List] := ToPackedArray[e];
TryPack[e_] := e;


PackageScope["TryPackRaw"]

TryPackRaw[assoc_Association] := Map[TryPackRaw, assoc];

TryPackRaw[arrays:{__NumericArray}] /; (SameQ @@ Map[Dimensions, arrays]) := Quiet @ Check[
	ArrayCatenate[arrays],
	arrays
];

TryPackRaw[e_] := e;


PackageScope["BroadcastingConform"]

BroadcastingConform[arrays_] := Scope[
	dims = Map[Dimensions, arrays];
	ranks = Map[Length, dims];
	deepest = MaxIndex[ranks];
	tdim = dims[[deepest]]; trank = Length[tdim];
	newArrays = MapThread[broadcastTo[trank - #1, tdim, #2]&, {ranks, arrays}];
	offset = trank - Min[ranks];
	{offset, newArrays}
];

broadcastTo[0, _, arr_] := arr;
broadcastTo[n_, td_, arr_] := CTable[arr, Take[td, n]];

	

PackageScope["FunctionArgumentCount"]

FunctionArgumentCount[f:Function[_]] := Max[0, DeepCases[f, Slot[n_Integer] :> n]];
FunctionArgumentCount[HoldPattern @ Function[var_Symbol, _]] := 1;
FunctionArgumentCount[HoldPattern @ Function[var:{__Symbol}, _]] := Length[Unevaluated[var]];
FunctionArgumentCount[f_Symbol] := Match[DownValues[f],
	{Verbatim[HoldPattern][HoldPattern[f[args___]]] :> _} :> Length[HoldComplete[args]],
	$Failed
];


PackageScope["NumberedAssociation"]

NumberedAssociation[list_] := 
	AssociationThread[
		IntegerString @ Range @ Length @ list,
		list
	];


PackageExport["TestTargetDevice"]

SetUsage @ "
TestTargetDevice[spec$] returns True if spec$ is a valid setting for TargetDevice.
TestTargetDevice[spec$, head$] issues a message on symbol head$ if the spec isn't valid."

TestTargetDevice[spec_] := 
	Not @ FailureQ @ CatchFailure @ ParseTargetDevice @ spec;

TestTargetDevice[spec_, head_] := 
	Not @ FailureQ @ CatchFailureAsMessage[head, ParseTargetDevice @ spec];


PackageExport["ContextToString"]

ContextToString[i:{__Integer}] := Scope[
	{devices, ints} = Transpose @ Map[FromContextCode, i];
	ints = Sort[ints]; {min, max} = MinMax[ints];
	ndevs = CountDistinct[devices];
	Which[
		ndevs == 1 && Length[ints] == (1 + max - min),
			StringJoin[First[devices], " ", IntegerString[min+1], "-", IntegerString[max+1]],
		ndevs == 1 && min == max, (* <- debugging *)
			StringJoin[IntegerString @ Length[ints], "\[Times]", ContextToString @ First[i]],
		True,
			StringRiffle[ContextToString /@ i, ", "]
	]
];	

ContextToString[i_Integer] := Match[FromContextCode[i],
	{"CPU", _} :> "CPU",
	{"GPU", 0} :> "GPU",
	{"GPU", n_Integer} :> "GPU " <> IntegerString[n+1]
];


PackageScope["GetGPUInformation"]

GetGPUInformation[] := Table[
	AssociationThread[{"TotalMemory", "FreeMemory"}, GetGPUMemoryInformation[i]],
	{i, $GPUCount}
];


PackageExport["MemoryUsageInfo"]
PackageExport["MemoryUsageInfoString"]

PackageScope["$reportGPUMemoryUsage"]
$reportGPUMemoryUsage = True;
(* ^ will be turned off by NetTrain if we're not doing GPU training *)

MemoryUsageInfo[] := Scope[
	MemoryAvailable;
	{free, total} = SystemTools`Private`getMemoryAll[][[{1, 4}]];
	assoc = <|"System" -> Round[total - free], "Kernel" -> MemoryInUse[], "Buckets" -> $CurrentExecutorBucketCount|>;
	If[$reportGPUMemoryUsage && $GPUCount > 0,
		gpuInfo = GetGPUInformation[];
		Do[
			{total, free} = Lookup[gpuInfo[[i]], {"TotalMemory", "FreeMemory"}];
			assoc["GPU" <> IntegerString[i]] = total - free;
		,
			{i, $GPUCount}
		]
	];
	Round[assoc]
];

MemoryUsageInfoString[] := MemoryUsageInfoString[MemoryUsageInfo[]];

MemoryUsageInfoString[info_] := 
	StringJoin @ Riffle[
		KeyValueMap[
			{Replace[#1, {"System" -> "S", "Kernel" -> "K", "Buckets" -> "B"}], ":", 
			 If[#1 === "Buckets", IntegerString[#2], fmtBytes[Round[#2]]]}&,
			info
		],
		" "
	];

fmtBytes[n_] := Scope[
	g = Floor[n / 10^9];
	m = Mod[Floor[n / 10^6], 1000];
	StringJoin[
		If[g > 0, {IntegerString[g], ","}, {}],
		IntegerString[m, 10, 3], "M"
	]
];


PackageScope["Decamel"]
PackageScope["AvoidDecamel"]

upper = RegularExpression["[A-Z]"];
lower = RegularExpression["[a-z]"];

Decamel[e_] := e;

Decamel[AvoidDecamel[e_]] := e;
Decamel[str_String] := If[StringCount[str, upper] <= 1, str,
	StringReplace[str, l:lower ~~ u:upper :> l <> " " <> ToLowerCase[u]]];


PackageScope["TransposeAssocVector"]

TransposeAssocVector[data_List] := Scope[
	keys = Keys[data];
	If[SameQ @@ keys,
		keys = First[keys];
		UnsafeQuietCheck[AssociationThread[keys, Transpose @ Values @ data], $Failed],
		$Failed
	]
];


PackageScope["WithInteractiveProgressBar"]

SetHoldAll[WithInteractiveProgressBar, makeDynamicProg];
WithInteractiveProgressBar[body_, prog_Symbol] := Scope[
	SetupTeardown[
		cell = PrintTemporary @ makeDynamicProg[prog],
		body,
		NotebookDelete[cell]
	]
];

makeDynamicProg[prog_] := ModuleScope[
	start = AbsoluteTime[];
	fraction = curr = max = 0; time = "unknown"; 
	Dynamic[
		{curr, max} = prog;
		elapsed = AbsoluteTime[] - start;
		fraction = N[curr / max];
		time = If[!TrueQ[elapsed > 0 && fraction > 0], "unknown",
			TimeString[Round[(1-fraction) / (fraction / elapsed)]]];
		ProgressPanel["Processing", Row[{curr, "/", max, " items processed, ", time, " remaining"}], fraction],
		UpdateInterval -> 0.5, TrackedSymbols :> {}
	]
];


PackageScope["GuessArgumentCount"]

GuessArgumentCount = MatchValues[
	HoldPattern @ Function[a_List, __] := Length[Unevaluated[a]];
	HoldPattern @ Function[_Symbol, __] := 1;
	HoldPattern @ Function[body_] := Max[0,
		DeepCases[HoldComplete[body] /. _Function :> Null, Slot[n_Integer] :> n]
	];
	_ := Indeterminate
];


PackageScope["ToCompressedBytes"]

ToCompressedBytes[e_] := BinarySerialize[ToPackedArray @ e, PerformanceGoal -> "Size"];


PackageScope["TimeToSamples"]

TimeToSamples[HoldPattern @ t_Quantity, sr_] := Ceiling[sr*QuantityMagnitude[UnitConvert[t, "Seconds"]]];
TimeToSamples[t_Integer, _] := t;
TimeToSamples[_, _] := $Failed;


PackageScope["FailIfExtraKeys"]

General::unreckey = "Unrecognized key `` in ``. Keys can include ``."
FailIfExtraKeys[assoc_, allowed_, msg_] := 
	Replace[
		Complement[Keys @ assoc, allowed], 
		{a_, ___} :> ThrowFailure["unreckey", a, allowed, msg]
	];


PackageScope["ArrayCatenate"]

ArrayCatenate[arrays_List] := 
	ArrayReshape[Join @@ arrays, Prepend[Dimensions @ First[arrays], Length @ arrays]];


PackageScope["ArrayUnpack"]

ArrayUnpack[na_NumericArray] := ArrayNormalToLevel[na, 1];
ArrayUnpack[e_] := e;


PackageScope["staticNiceButton"]

SetHoldRest[staticNiceButton]

staticNiceButton[label_, action_] :=
	Button[
		label, action,
		Appearance :> FEPrivate`Part[
			FEPrivate`FrontEndResource["FEExpressions", "GrayButtonNinePatchAppearance"], 
			{1, 3} (* take the "Default" -> and "Pressed" -> entries, drop the "Hover" -> entry *)
		],
		BaseStyle -> {FontSize -> 12},
		ImageSize -> 87
	];


PackageScope["makeLeftRightClickerBoxes"]

SetAttributes[makeLeftRightClickerBoxes, HoldFirst]

makeArrow[str_] := MouseAppearance[Style[str, 
	FontColor :> If[CurrentValue["MouseOver"], RGBColor[0.27, 0.54, 0.79], GrayLevel[.25]],
	FontFamily -> "MS Gothic", FontSize -> 14, FontWeight -> "Thin"
], "LinkHand"];

makeLeftRightClickerBoxes[var_, labels_] := With[
	{len = Length[labels], prev = makeArrow["\:2039"], next = makeArrow["\:203A"]},
	ToBoxes @ Dynamic[Grid[
		{{
			Button[prev, var = Mod[var - 1, len, 1], Appearance -> None],
			Dynamic[labels[[var]], TrackedSymbols :> {var}],
			Button[next, var = Mod[var + 1, len, 1], Appearance -> None]
		}}, 
		ItemSize -> {{3, 20, 3}, {2}}
	], TrackedSymbols :> {var}]
];