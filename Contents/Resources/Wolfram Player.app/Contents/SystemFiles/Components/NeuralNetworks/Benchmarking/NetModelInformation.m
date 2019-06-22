Package["NeuralNetworks`"]

PackageImport["AWSLink`"]
PackageImport["MXNetLink`"]
PackageImport["PacletManager`"]

(*----------------------------------------------------------------------------*)
(* 11.3 compatibility *)
If[$VersionNumber < 12., 
	RealDigitsString[x_, n_] := ToString[x]
]


PackageExport["NetModelRuntimeInformation"]

SetUsage @ "
NetModelRuntimeInformation[] takes a description of the \
testing system (eg 'EC2 P3.xlarge instance' or 'Macbook Pro 2016 Laptop'. See \
Benchmarking/ref/$BenchmarkSystems docs in WL for good examples) and returns an \
Association with benchmark information and output signatures for all NetModel \
models in the list NetModel[]. It uses NetModel[$$, 'ExampleInput'] for the input. \
In addition, this Association contains full information about the NeuralNetworks and related paclets \
version numbers, total available system memory, etc. The output Association is \
guaranteed to be a valid MongoDB document.
NetModelRuntimeInformation[models$] allows a model name or list of model names \
models$ to be specified. 

The following options are available:
|'BatchSizes' | Automatic | A list of batch sizes to use for benchmarking. A batch-size \
of None is always used. |
|'LogFile' | None | A file used for logging partial results. |
|TargetDevice | 'CPU' | Target device to run. Either 'CPU' or 'GPU'.  |
|'PreserveCache' | True | If True, the number of downloaded and stored NetModels will \
will be the same before and after running NetModelRuntimeInformation. If False, \
all NetModels downloaded will be kept on disk after NetModelRuntimeInformation \
has run. |
|'UpdateNetModel' | True | If True, the NetModel will automatically be updated \
if a newer version exists. |";

Options[NetModelRuntimeInformation] = {
	"BatchSizes" -> Automatic,
	TargetDevice -> "CPU",
	"PreserveCache" -> True,
	"UpdateNetModel" -> True,
	"LogFile" -> None
};

NetModelRuntimeInformation::errquery = 
	"NetModel[] failed. Check internet connection."
NetModelRuntimeInformation::invarg2 = 
	"The first argument should be All, a string name, or a list of string names, but was ``."
NetModelRuntimeInformation::invbatch = 
	"The option \"BatchSizes\" must be a list of integers."
NetModelRuntimeInformation::nosysdesc = 
	"Warning: option \"SystemDescription\" Automatic, and cannot infer machine description. It is recommended \
to set this option explicitly in this case."
NetModelRuntimeInformation::badfail = 
	"Cannot even run ElementwiseLayer[Sin][1., TargetDevice -> ``]. Please check your NeuralNetworks installation."
NetModelRuntimeInformation::skipinit = 
	"Warning: the NetModel `` is not an initialized NetModel. Skipping this model."
NetModelRuntimeInformation::skipver = 
	"Warning: the NetModel `` has required version `` but Wolfram Language $VersionNumber is ``. Skipping this model."

NetModelRuntimeInformation[opts:OptionsPattern[]] := NetModelRuntimeInformation[All, opts]

NetModelRuntimeInformation[model_, OptionsPattern[]] := 
CatchFailureAsMessage @ Scope[
	UnpackOptions[
		batchSizes, targetDevice, preserveCache, 
		updateNetModel, logFile
	];

	(* parse model spec *)
	models = Which[
		VectorQ[model, StringQ], 
			model,
		StringQ[model],
			{model},
		model === All,
			temp = NetModel[];
			If[FailureQ[temp], Return@$Failed];
			temp
			,
		True,
			ThrowFailure[NetModelRuntimeInformation::invarg2, model]
	];

	(* batchsize is usually small, otherwise causes massive slowdowns for large nets *)
	If[!VectorQ[batchSizes, IntegerQ] && batchSizes =!= Automatic, 
		ThrowFailure[NetModelRuntimeInformation::invbatch]
	];
	batchSizes = If[batchSizes === Automatic,
		{None, 1, 2},
		Prepend[batchSizes, None]
	];

	(* check that targ device is fine *)
	If[$VersionNumber < 12., 
		{devID, dev} = {1, targetDevice}
		,
		devCode = ParseTargetDevice[targetDevice];
		dev = If[devCode === 1, "CPU", "GPU"];
		devID = If[devCode === 1, 1, devCode - 1];
	];
	
	(* Warmup: don't skew timing results of first net needing to load NeuralNetworks *)
	If[FailureQ @ Quiet @ ElementwiseLayer[Sin][1., TargetDevice -> targetDevice],
		ThrowFailure[NetModelRuntimeInformation::badfail, targetDevice];
		Return[$Failed]
	];

	(* Net information *)
	numModels = ToString @ Length[models];
	(* progress info: important as very slow! *)
	prog = StringJoin["Processing model ", ToString[#2], "/", numModels, " : ", #1]&;
	netData = {};

	(* should be the same for all models, so can group them *)
	timestamp = CurrentDate[];

	(* return current set of data if abort *)
	CheckAbort[
		Do[
			name = models[[i]];
			ro = ResourceObject[name];


			(* get general system info: needs to be computed for every model
				as things like GPU + CPU mem change during run *)
			sysinfo = neuralNetEnvironmentInfo[dev, devID];

			netExistsQ = If[preserveCache,
				(ro["DownloadedVersion"] =!= None),
				False
			];
			If[updateNetModel && (ro["DownloadedVersion"] =!= ro["Version"]),
				ResourceRemove[ro]; (* forced redownload *)
				ro = ResourceObject[name];
			];

			Print @ prog[name, i];

			(* Skip netmodels that are not initialized *)
			testNet = NetModel[name];
			If[!InitializedNetQ[testNet],
				Message[NetModelRuntimeInformation::skipinit, name];
				Continue[]
			];
			(* skip netmodels that have a higher required version *)
			reqVer = NetModel[name, "WolframLanguageVersionRequired"];
			If[reqVer > $VersionNumber,
				Message[NetModelRuntimeInformation::skipver, name, reqVer, $VersionNumber];
				Continue[]
			];

			infoAssoc = <|
				"MachineInformation" -> NNMachineInformation[TargetDevice -> targetDevice],
				"PacletInformation" -> nnPacletInfo[],
				"TargetDevice" -> targetDevice,
				"Timestamp" -> timestamp
			|>;

			infoAssoc = Join[
				<|"Model" -> name|>,
				singleModelInfo[name, batchSizes, targetDevice],
				infoAssoc
			];

			(* add target device *)
			infoAssoc["TargetDevice"] = targetDevice;
			infoAssoc["PacletInformation"] = nnBenchmarkPacletInfo[];

			(* append *)
			AppendTo[netData, infoAssoc];

			(* log to file *)
			If[logFile =!= None,
				Export[logFile, netData]
			];

			(* clean net if cache invariance option is specified *)
			If[!netExistsQ && preserveCache, 
				ResourceRemove @ ro
			];
			,
			{i, Length[models]}
		]
		,
		If[!netExistsQ && preserveCache, 
			ResourceRemove @ ResourceObject[name]
		];
	];

	netData
]

netConsistencyData[net_, input_, device_String] := Scope[
	(* data1: check netencoder consistent *)
	ins = getConformedPreNetInputs[net, input];
	(* consistency 2: check non-decoder output *)
	out = net[input, All -> None, TargetDevice -> device];
	If[!AssociationQ[out],
		out = <|First @ NetInformation[net, "OutputPortNames"] -> out|>
	];
	out = 	<|
		"InputHashes" -> tensorSignatures[ins],
		"OutputHashes" -> tensorSignatures[out]
	|>;
	ReplaceAll[out, x_Real :> RuleCondition[ToExpression @ RealDigitsString[x, 4]]]
]

netInformation[netname_String] := Scope[
	net = NetModel[netname];
	<|
		"ArraysTotalByteCount" -> NetInformation[net, "ArraysTotalByteCount"],
		"LayersCount" -> NetInformation[net, "LayersCount"],
		"LayerTypeCounts" -> 
			Association@KeyMap[ToString, NetInformation[net, "LayerTypeCounts"]],
		"RecurrentStatesCount" -> NetInformation[net, "RecurrentStatesCount"]
	|>
]

resourceObjectInfo[netname_String] := Scope[
	net = NetModel[netname];
	ro = ResourceObject[netname];
	<|
		"Version" -> ro["Version"],
		"DownloadedVersion" -> ro["DownloadedVersion"],
		"WolframLanguageVersionRequired" -> NetModel[netname, "WolframLanguageVersionRequired"],
		"UUID" -> ro["UUID"]
	|>
]

singleModelInfo[netname_String, batchsizes_List, device_String] := Scope[
	net = NetModel[netname];
	in = NetModel[netname, "EvaluationExample"];
	Join[
		<|
			"ResourceObject" -> resourceObjectInfo[netname],
			"NetInformation" -> netInformation[netname],
			"BenchmarkData" -> netBenchmark[net, in, batchsizes, device]
		|>
	,
		netConsistencyData[net, in, device]
	]
]


(*----------------------------------------------------------------------------*)
(* general system information *)

(* Location is useful to know for a paclet incase user is using local checkout *)
pacletInfo[name_] := 
	AssociationThread[
		{"Version", "Location"} -> Lookup[PacletManager`PacletInformation[name], {"Version", "Location"}]
]

PackageScope["nnBenchmarkPacletInfo"]

nnBenchmarkPacletInfo[] := <|
	"MXNetLink" -> Join[pacletInfo["MXNetLink"], <|"MXNetVersion" -> GetMXNetVersion[]|>],
	"NeuralNetworks" -> pacletInfo["NeuralNetworks"],
	"NumericArrayUtilities" -> pacletInfo["NumericArrayUtilities"],
	"GeneralUtilities" -> pacletInfo["GeneralUtilities"]
|>

(*----------------------------------------------------------------------------*)
PackageExport["NNMachineInformation"]
SetUsage @ "
NNMachineInformation[] returns 

The following options are available:
|TargetDevice | 'CPU' | The target device to use. |
"

Options[NNMachineInformation] = {TargetDevice -> "CPU"}

NNMachineInformation::invtarg = "The option \'TargetDevice\' had an invalid specification ``."

NNMachineInformation[OptionsPattern[]] := CatchFailureAndMessage @ Scope[
	UnpackOptions[targetDevice];

	out = <|
		"Version" -> $Version,
		"VersionNumber" -> $VersionNumber,
		"MachineID" -> $MachineID,
		"MachineName" -> $MachineName,
		"MachineType" -> $MachineType,
		"MemoryAvailable" -> MemoryAvailable[],
		"OperatingSystem" -> $OperatingSystem,
		"ProcessorCount" -> $ProcessorCount,
		"ProcessorType" -> $ProcessorType,
		"SystemMemory" -> $SystemMemory,
		"SystemWordLength" -> $SystemWordLength,
		"ReleaseID" -> SystemInformation["Kernel", "ReleaseID"]
	|>;

	(* this needs improving *)
	{dev, ids} = Switch[targetDevice,
		{"GPU", _Integer}, {"GPU", {Last[targetDevice]}},
		{"GPU", All}, {"GPU", Range[$GPUCount]},
		{"GPU", _List}, targetDevice,
		"GPU", {"GPU", {1}},
		"CPU", {"CPU", 1},
		_, ThrowFailure[NNMachineInformation::invtarg, targetDevice]
	];

	(* todo: check GPU count *)
	If[dev === "GPU", out["GPUInformation"] = Map[gpuInfo[#]&, ids]];

	(* check whether we are on aws *)
	If[TrueQ[AWSLink`EC2InstanceQ[]], 
		out["EC2InstanceType"] = AWSLink`GetEC2InstanceMetadata["instance-type"];
		out["EC2AMI"] = AWSLink`GetEC2InstanceMetadata["ami-id"]
	];

	out
]

gpuInfo[devID_] := Scope[
Block[
	{$ContextPath = {"System`"}},
	Needs["GPUTools`"];
	(* This is not very reliable... Should use CUDALink if available *)
	out = <|"Name" -> GPUTools`Internal`VideoCardName[], "DeviceID" -> devID|>;
	res = MXNetLink`GetGPUMemoryInformation[devID];
	(* for 11.3, don't have this info *)
	If[ListQ[res],
		out["TotalMemory"] = Last[res];
		out["AvailableMemory"] = First[res];
	];
	out
]
]

(*----------------------------------------------------------------------------*)
(* array signatures: a real-valued 'hash' of a numeric array that are are close 
for numeric arrays differing by small amounts. Important, as all backends have \
slightly different numeric results
*)

SetUsage @ 
"tensorSignatures[List[$$]] returns an Association of different real-valued hashes \
of the numeric array List[$$].
tensorSignatures[Association[$$]] takes an Association with numeric array values \
and maps tensorSignatures[List[$$]] over the values."

PackageScope["tensorSignatures"]

tensorSignatures[x_List] := ApplyThrough[$SignatureFunctions, x]
tensorSignatures[x_NumericArray] := tensorSignatures @ Normal[x]

tensorSignatures[x_Association] := tensorSignatures /@ x

basis[n_] := (1 - 2 * Mod[n, 2]);

modulate1[x_] := Scope[
	flat = Flatten[x];
	Mean[basis[Range[Length[flat]]]*flat]
]

modulate2[x_] := Scope[
	flat = Flatten[x];
	Mean[basis[Floor[Range[Length[flat]]/2]] * flat]
]

$SignatureFunctions := $SignatureFunctions = <|
	"Mean" -> Mean[Flatten @ {#}]&,
	"MeanAbs" -> Total[Flatten @ Abs @ {#}, Infinity]&,
	"Norm2" -> Norm[Flatten[Abs@{#}], 2]&,
	"Norm3" -> Norm[Flatten[Abs@{#}], 3]&,
	"BasisModulation1" -> modulate1[#]&,
	"BasisModulation2" -> modulate2[#]&,
	"First" -> First[Flatten@#]&,
	"Last" -> Last[Flatten@#]&,
	"MinMax" -> MinMax[Flatten@#]&,
	"Median" -> Median[Flatten@#]&
|>

(*----------------------------------------------------------------------------*)
netBenchmark[net_, input_, batchsizes_List, device_String] := Scope[
	(* conform to assoc *)
	exampleIn = If[AssociationQ[input], 
		input,
		<|First[NetInformation[net, "InputPortNames"]] -> input|>
	];
	out = AssociationTranspose[
		netModelTimingData[net, exampleIn, #, device]& /@ batchsizes
	];
	ReplaceAll[out, x_Real :> RuleCondition[ToExpression @ RealDigitsString[x, 4]]]
]

SetAttributes[iTiming, HoldAll];
iTiming[x_] := Scope[
	{t, r} = AbsoluteTiming[x];
	If[FailureQ[r], Return@"$Failed"]; (* use None to be compatible with mongo*)
	N[t]
]

SetAttributes[iRepeatTiming, HoldAll];
iRepeatTiming[x_] := Scope[
	{t, r} = RepeatedTiming[x];
	If[FailureQ[r], Return@"$Failed"]; (* use None to be compatible with mongo*)
	N[t]
]

netModelTimingData[net_, input_Association, batchsize_, dev_] := Scope[
	(* create appropriate example. inputEx is of form <|"Input1" -> ...|> *)
	inputEx = If[batchsize === None,
		input,
		AssociationTranspose[ConstantArray[input, batchsize]]
	];

	singleInQ = (Length[inputEx] === 1);

	(* Codepath 1: Association Input + heuristic batch size *)
	ClearCache[]; 
	warmup1 = iTiming @ net[inputEx, TargetDevice -> dev];
	time1 = iTiming @ net[inputEx, TargetDevice -> dev];

	(* Codepath 2: Association Input + explicit batch size *)
	ClearCache[];
	bs2 = Replace[batchsize, None -> 1];
	warmup2 = iTiming @ net[inputEx, TargetDevice -> dev, BatchSize -> bs2];
	time2 = iTiming @ net[inputEx, TargetDevice -> dev, BatchSize -> bs2];

	(* Codepath 3: test for fast codepath. Only CPU and single input net *)
	If[(Length[inputEx] === 1) && (dev === "CPU"),
		ClearCache[];
		inputEx2 = First @ inputEx;
		warmup3 = iTiming @ net[inputEx2];
		time3 = iTiming @ net[inputEx2]
		,
		time3 = Null
	];

	(* Codepath 4: no encoders/decoder *)
	ClearCache[];
	dataStripped = getConformedPreNetInputs[net, inputEx];
	strippedNet = stripNet[net];

	strippedNet[dataStripped, TargetDevice -> dev]; (* warmup *)
	time4 = iTiming @ strippedNet[dataStripped, TargetDevice -> dev];
	
	(* return *)
	mongoBatchSizes = ToString[batchsize]; 
	<|
		"AssociationInput" -> time1,
		"AssociationInputWarmup" -> warmup1,
		"ExplicitBatchSize" -> time2,
		"FastPath" -> time3,
		"TimeWithoutCoders" -> time4,
		"BatchSize" -> Replace[batchsize, None -> "None"] (* mongo consistency *)
	|>
]

(* this returns an association of inputs, precisely what the post-encoder net will get *)
getConformedPreNetInputs[net_, in_] := Scope[
	inNames = Sort @ NetInformation[net, "InputPortNames"];
	(* either a single input or an association. Conform to assoc *)
	newin = KeySort @ If[AssociationQ[in], in, <|First[inNames] -> in|>];
	encF = AssociationMap[
		Replace[NetExtract[net, #], Except[_NetEncoder] :> Identity]&, 
		inNames
	];
	vals = MapThread[#1[#2]&, {Values[encF], Values[newin]}];
	out = AssociationThread[inNames -> vals];
	Developer`ToPackedArray /@ out
]

(* remove all encoders and decoders *)
encDecQ[x_NetDecoder] := True
encDecQ[x_NetEncoder] := True
encDecQ[_] := False

stripNet[net_] := Module[
	{ports = Join[NetInformation[net, "InputPortNames"], NetInformation[net, "OutputPortNames"]]}
	,
	ports = Select[ports, (encDecQ@NetExtract[net, #])&];
	NetReplacePart[net, Thread[ports -> None]]
]

