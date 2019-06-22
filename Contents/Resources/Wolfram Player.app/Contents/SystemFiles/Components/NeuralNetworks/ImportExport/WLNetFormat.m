Package["NeuralNetworks`"]


PackageScope["$NeuralNetworkFormatIsUnstable"]

$NeuralNetworkFormatIsUnstable = False;


PackageScope["$NeuralNetworksVersionNumber"]


PackageScope["ValidVersionQ"]

ValidVersionQ[v_String] := StringMatchQ[v, DigitCharacter.. ~~ "." ~~ DigitCharacter.. ~~ "." ~~ DigitCharacter..];
ValidVersionQ[_] := False;


PackageScope["ToWLVersion"]

ToWLVersion[v_] := StringTake[v, StringPosition[v, "."][[2, 1]] - 1];


GetIOOption[head_, defaults_, keys_, rules_] := Scope[
	rules = Flatten[{rules}];
	res = Quiet @ Check[OptionValue[defaults, rules, keys], $Failed];
	If[FailureQ[res], 
		badKey = First @ Complement[Keys[rules], keys];
		Message[head::optx, badKey, If[head === Export, "export to \"WLNet\"", "import from \"WLNet\""]];
		ThrowRawFailure[$Failed]; (* <- works better with testing for some reason *)
	];
	res
];
(* ^ this exists because OptionValue will not issue the right message when called *)


PackageExport["WLNetExport"]

Clear[WLNetExport];

General::invnet2 = "The second argument in `` is not a valid net."

$WLNetExportOptions = {
	"InternalVersion" -> Automatic (* <- allows override of what version is embedded in the exported file *)
};

WLNetExport[filename_String, head_Symbol[net_Association, meta_Association] ? ValidNetQ, opts:OptionsPattern[]] := CatchFailureAsMessage[Export, Scope[
	{internalVersion} = GetIOOption[Export, $WLNetExportOptions, {"InternalVersion"}, {opts}];
	If[internalVersion =!= Automatic,
		If[!ValidVersionQ[internalVersion], ReturnFailed[]];
		meta["Version"] = internalVersion;
	,
		internalVersion = $NeuralNetworksVersionNumber;
	];	
	CollectTo[{$tensors}, result = PrepareForExport[net]];
	Block[{$ContextPath = {"System`", "NeuralNetworks`"}, $Context = "Dummy`"}, 
		netstring = IOContextBlock @ Compress[result];
		metastring = IOContextBlock @ Compress[meta];
	];
	ExportStructuredHDF5[filename, Association[
		"Version" -> internalVersion, 
		"Unstable" -> Boole @ $NeuralNetworkFormatIsUnstable,
		"Network" -> netstring, If[$tensors === {}, {}, "Arrays" -> $tensors],
		"Metadata" -> metastring
	]];
	filename
]];

General::nonetexpelem = "No export elements are supported for \"WLNet\" format.";

WLNetExport[a_, b_, ___] := (
	If[!ValidNetQ[b] && !MatchQ[b, _RuleDelayed], 
		Message[Export::invnet2, HoldForm[Export][a, Shallow[b]]],
		Message[Export::nonetexpelem];
	];
	$Failed
);


PackageExport["ExportedArray"]


PackageScope["PrepareForExport"]

DeclareMethod[PrepareForExport, 
	PrepareLayerForExport, 
	PrepareContainerForExport,
	PrepareOperatorForExport
];

PrepareLayerForExport[assoc_] := 
	prepSharedArrays @ MapAtFields["Arrays", ExportTensor, assoc];

PrepareContainerForExport[assoc_] := 
	prepSharedArrays @ MapAtFields["Nodes", PrepareForExport, assoc];

prepSharedArrays[assoc_] := assoc;
prepSharedArrays[assoc_] /; KeyExistsQ[assoc, "SharedArrays"] := 
	MapAtFields["SharedArrays", ExportTensor, assoc];

PrepareOperatorForExport[assoc_] := 
	PrepareLayerForExport @ MapAtSubNets[PrepareForExport, assoc]

ExportTensor[raw_NumericArray] := (
	BagPush[$tensors, raw]; 
	ExportedArray[BagLength[$tensors]]
);

ExportTensor[e_] := e;


PackageExport["WLNetImport"]

Import::wlprerl = "Cannot import networks saved during the 11.1 prerelease period."

corrupt[] := ThrowFailure["wlnetcorr", file];
General::wlnetcorr = "File is corrupt or is not a WLNet file.";
General::wlbadprop = "`` is not a valid property for a WLNet file."

$WLNetImportOptions = {
	"InternalVersion" -> Automatic
};

Clear[WLNetImport];

General::wlnetimpmem = "Net in file `` is too large (``bytes) to load into memory.";

WLNetImport[file_String, retType_:"Net", opts:OptionsPattern[$WLNetImportOptions]] := CatchFailureAsMessage[Import, Scope[

	{internalVersion} = GetIOOption[Import, $WLNetImportOptions, {"InternalVersion"}, {opts}];

	Switch[retType, 
		"Net", arrayForm = "RawArrays",
		"UninitializedNet", arrayForm = "Placeholders",
		"ArrayList", arrayList = True; arrayForm = "RawArrays",
		"ArrayAssociation", arrayAssoc = True; arrayForm = "RawArrays",
		"WLVersion" | "InternalVersion" | "InternalMetadata", arrayForm = "Placeholders",
		_, ThrowFailure["wlbadprop", retType]
	];

	If[retType === "Net",
		fileSize = Quiet @ Check[FileByteCount[file], 0];
		Which[
			fileSize > $RealisticSystemMemory, ThrowFailure["wlnetimpmem", file, PositiveSIString[fileSize]],
			fileSize > ($RealisticSystemMemory / 4), ClearSystemCache[]; ClearCache[],
			fileSize > ($RealisticSystemMemory / 8), ClearCache[],
			True, Null
		];
		(* ^ if we're loading a big net, try clear some space for it *)
	];

	result = Quiet @ CatchFailure @ ImportStructuredHDF5[file, "ArrayForm" -> arrayForm];

	If[FailureQ[result], corrupt[]];

	If[!MatchQ[result, KeyValuePattern[{"Version" -> _, "Network" -> _}]], corrupt[]];

	UnpackAssociation[result, network, arrays, version, "Default" -> {}];

	If[retType === "InternalVersion", Return[version]];

	If[retType === "WLVersion", Return[version]];
	(* ^ when format registry gets InternalVersion, then change this to be Return @ ToWLVersion @ version *)

	If[arrayList, Return[arrays]];
	
	If[KeyExistsQ[result, "Metadata"],
		metadata = ReleaseHold @ toExpression @ result["Metadata"];
		If[!AssociationQ[metadata], corrupt[]];
	,
		metadata = <|"Version" -> version|>;
	];

	If[internalVersion =!= Automatic, 
		If[!ValidVersionQ[internalVersion], ReturnFailed[]];	
		metadata["Version"] = internalVersion];

	If[retType === "InternalMetadata", Return[metadata]];

	network = toExpression[network];
	If[FailureQ[network], ReturnFailed[]];

	If[arrayAssoc,
		arrpos = Position[ReleaseHold[network], _ExportedArray];
		specs = FromNetPath[arrpos /. Key[k_] :> k];
		Return[AssociationThread[specs, arrays]]
	];

	$AssumedCoderVersion = version; 
	(* ^ this is so that old nets with versionless coders in them will acquire the version of the net,
	and hence be upgraded properly *)

	If[arrayForm === "Placeholders", arrays = arrays /. H5DatasetPlaceholder -> TensorT];
	network = network /. ExportedArray[id_] :> RuleCondition @ arrays[[id]];
	network = ReleaseHold[network];

	ReconstructNet[network, metadata, "import"]
]];

$AssumedCoderVersion = Indeterminate;

WLNetImport[___] := $Failed;


SetHoldAllComplete[IOContextBlock];
IOContextBlock[expr_] := Block[
	{$ContextPath = {"System`", "NeuralNetworks`"}, $Context = "NeuralNetworks`ImportContext`"}, 
	expr
];

toExpression[str_String] := IOContextBlock @ ToExpression[str, InputForm, Hold]
toExpression[str_String] /; StringStartsQ[str, "1:"] := IOContextBlock @ Uncompress[str, Hold];
toExpression[ba_ByteArray] := BinaryDeserialize[ba, Hold];

toExpression[_] := $Failed;


PackageExport["RenameLayers"]

SetUsage @ "
RenameLayers[net$, renamer$] renames layers in net by applying renamer$ to each name to produce a \
new name. renamer$ can be a function, an association, or a (list of) string replacement."

RenameLayers[NetP[net, meta], renamingFunction_] := Scope @ CatchFailureAsMessage[Export, 
	If[MatchQ[renamingFunction, _Rule | _RuleDelayed | {Repeated[_Rule | RuleDelayed]}],
		renamingFunction = StringReplace[renamingFunction]
	];
	$renamer = checkRename[renamingFunction];
	net2 = tryRename[net];
	ConstructNet[net2, meta]
];

General::badrename = "Given renaming function applied to name `` returned the non-string ``."
General::duprename = "Given renaming function applied to name `` returned the already-produced name ``."

checkRename[renamer_][name_] := Scope[
	CacheTo[$renameCache, name, 
		newname = Replace[renamer[name], _Missing | Null | None -> name];
		If[!StringQ[newname], ThrowFailure["badrename", name, newname]];
		If[MemberQ[$renameCache, newname], ThrowFailure["duprename", name, newname]];
		newname
	]
];

tryRename[assoc_] := ReplaceAll[assoc, 
	container:<|"Type" -> ("Chain"|"Graph"), ___|> :> RuleCondition @ renameContainerLayers[container]
];

renameContainerLayers[assoc_] := Scope[
	$renameCache = Association[];
	MapAt[
		ReplaceAll[NetPath["Nodes", name_String, rest___] :> RuleCondition @ NetPath["Nodes", $renamer[name], rest]],
		MapAt[KeyMap[$renamer] /* tryRename, assoc, "Nodes"],
		"Edges"
	]
];
