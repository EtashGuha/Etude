Package["NeuralNetworks`"]


PackageScope["makeInputSetter"]


PackageScope["makeInputProcessingCode"]
PackageScope["makeDefaultSetter"]

PackageScope["$InputData"]
PackageScope["$TempData"]

(* 
makeInputProcessingCode's main job is to handle dynamic sequence lengths, producing 
code that will obtain the bucket key.

$InputData[n] is a dummy used by makeInputProcessingCode to represent an input to the net
$TempData[n] is a dummy to represent a temporarily stored result.
Callers will replace these as necessary for their purposes.
*)

makeInputProcessingCode[func_, inputs_Association, isBatched_] := Scope[
	$names = Keys[inputs];
	$idOwners = <||>; (* <- map ids to which slot set them first *)
	$isBatched = isBatched; $n = 1;
	$keyClosure = Hold[]; $lens = {};
	setter = If[isBatched, NDArraySetBatched, NDArraySetUnbatched];
	results = KeyValueMap[
		procInput[func, $n++, #1, StripCoders @ #2, ToEncoderFunction[#2, $isBatched], setter]&, 
		inputs
	];
	{$keyClosure /. Quoted[q_] :> q, $lens, results}
]

procInput[func_, n_, name_, type_, encoderf_, setter_] := 
	func[name, encoderf, $InputData[n], type, setter];

procInput[func_, n_, name_, type_ ? VarSequenceQ, encoderf_, setter_] := Scope[
	id = GetLengthVarID[type];
	idOwner = CacheTo[$idOwners, id, n]; 
	With[
	{lenf = If[$isBatched, getSeqLengths, getSeqLength],
	 lenVar = $TempData[2*n], dataVar = $TempData[2*n-1]}, 
	{input1 = If[encoderf === Identity, $InputData[n], 
		If[$isBatched, 
			Quoted[dataVar = batchApplyEnc[name, encoderf, $InputData[n]]],
			With[{encoderf2 = PropogateEncodeFailures[First @ encoderf, name]},
				Quoted[dataVar = encoderf2[$InputData[n]]]
			]
		]],
	 	input2 = If[encoderf === Identity, $InputData[n], dataVar]},
	(*  we capture the encoded data (if there is an encoder) into a $TempData, 
	and construct a keyClosure that just measures the length *)
	If[idOwner === n,
		(* if we are the first input to have seen this lengthvar, get its length *)
		keyClosure = Hold[lenVar = lenf @ input1];
		AppendTo[$lens, lenVar];
	,
		(* otherwise only check the lengths match the lengths of previous var *)
		With[{ownerName = $names[[idOwner]], ownerLen = $TempData[2*idOwner]}, 
		keyClosure = Hold[
			lenVar = lenf @ input1,
			If[ownerLen =!= lenVar, ThrowFailure["incseqlen", name, ownerName]]
			(* TODO: be more specific about where *)
		]]
	];
	AppendTo[$keyClosure, Unevaluated @@ keyClosure];
	(* the result is up to whoever called makeInputProcessingCode,
	but gets access to the encoder function, name, and a complainer *)
	func[name, Identity, input2, type, setter]
]];

General::netinvseq = "Invalid sequence `` provided to net."

PackageScope["getSeqLengths"]
PackageScope["getSeqLength"]

getSeqLength[list_List] := Length[list];
getSeqLength[{}] := ThrowFailure["netnullseq"];
getSeqLength[e_NumericArray] := Length[e];
getSeqLength[e_] := ThrowFailure["netinvseq", e];

getSeqLengths[list_] := Which[
	(NumericArrayQ[list] || PackedArrayQ[list]) && arrayDepth[list] > 1, 
		Replace[arrayDimensions[list], {d_, len_, ___} :> ConstantArray[len, d]],
	ListOfListsQ[list] || VectorQ[list, NumericArrayQ], 
		Map[getSeqLength, list],
	True, 
		ThrowFailure["netinvseq", list]
];

General::incseqlen = "Length of sequence provided to port \"``\" is inconsistent with length of sequence provided to port \"``\".";

makeDefaultSetter[name_, enc_, data_, type_, setter_] := 
	Hold @ setter[$ExecutorArrays["Inputs", name], enc @ data, InputErrorHandler[name, type]];

PackageScope["ParseNetProperty"]

PackageScope["$ExecutorArrays"]


SetUsage @ "
ParseNetProperty[net$, spec$, isBatched$] returns {odescrip$, ospec$, igrads$, wgrads$, listq$}.
* odescrip$ is a OutputDescriptor or a list of these.
* igrads$ is an assoc from prop to mxname
* wgrads$ is an assoc from prop to mxname.
* ospec$ is a list of output ports to compute.
* listq$ distinguishes {spec$} from spec$, as the returned descriptors are always a list."

ParseNetProperty[net_NetP, spec_, batchq_] := Scope[
	$net = net; $isBatched = batchq;
	$igrads = $wgrads = <||>; 
	$outputs = All; 
	$otypes = Outputs[net];
	descriptors = parseProp[spec];
	{Flatten @ List @ descriptors, $outputs, $igrads, $wgrads, ListQ[descriptors]}
];

(*****************************************************************************
parseProp parses a property spec into a held function that extracts the 
needed data from the executor's arrays. These are provided via $oarrays
(and matching $otypes). 
*)

(* None means remove the encoder, Automatic means use the 'default' decoding *)
$PropPattern = _String ? $IsPossibleDecPropQ | {_String ? $IsPossibleDecPropQ, _} | None | Automatic | 
	{_String, Repeated[_String -> (False|True|_Integer|_Real)]};
(* ^ this last` odd prop pattern is to ensure that user-defined function decoders can accept options *) 

$PortSpecPattern = _String | _NetPort | _NetPortGradient;
ListOfPortSpecsQ = MatchQ @ List @ Repeated[$PortSpecPattern | ($PortSpecPattern -> _)];

requiresOutput[out_] := If[$outputs === All, $outputs = {out}, AppendTo[$outputs, out]];

Clear[parseProp];

PackageScope["PleaseSetRSData"]
parseProp[PleaseSetRSData[spec_]] := Scope[
	res = parseProp[spec];
	MapAt[insertRSSetter, res, List @ FirstPosition[res, _OutputDescriptor]]
];

insertRSSetter[OutputDescriptor[name_, code_]] := With[
	{reader = If[$isBatched, NDArrayGetBatched, NDArrayGetUnbatchedSafe]},
	OutputDescriptor[name, 
		$RSData = RecurrentStateContainer @ Map[reader, Values @ $ExecutorArrays["OutputStates"]]; 
		code
	]
];


parseProp[Automatic] := 
	parsePropAll[Automatic, $otypes];

parseProp[All -> None] :=
	parsePropAll[None, $otypes]; (* <- needed by ClassifierMeasurements apparently *)

(* single output property *)
parseProp[prop:$PropPattern] /; MatchQ[$otypes, <|Except[prop] -> _|>] := 
	makeDecoder[First @ Keys @ $otypes, First @ $otypes, prop];

parseProp[name_String | NetPort[name_String]] :=
	If[Length[$otypes] === 1 && !KeyExistsQ[$otypes, name],
		makeDecoder[First @ Keys @ $otypes, First @ $otypes, name],
		makeDecoder[name, $otypes[name], Automatic]
	];

parseProp[name_String | NetPort[name_String] -> prop:$PropPattern] := 
	makeDecoder[name, $otypes[name], prop];

parseProp[list_List ? ListOfPortSpecsQ] :=
	parseProp /@ list;

toSpecKey[k_ -> _] := k;
toSpecKey[e_] := e;

General::invppspec = "`` is not a valid property or port specification."
parseProp[spec_] := ThrowFailure["invppspec", spec];

(*****************************************************************************
Parsing of NetPort specs
*)

parseProp[NetPort[spec_, out_]] /; out =!= "IntermediateArrays" := 
	parseProp[NetPort[ToList[spec, out]]];

parseProp[NetPort[spec_]] := Scope[
	isOutPort = KeyExistsQ[$otypes, spec];
	If[(StringQ[spec] || IntegerQ[spec]) && !isOutPort, 
		General::npambig = "There is no output port named `1`. To refer to the output of a layer identified by `1`, use NetPort[{`1`}] or NetPort[{`1`}, \"port\"].";
		ThrowFailure["npambig", MsgForm @ spec]];
	If[isOutPort, Return @ parseProp[spec]];
	If[$outputs === All, $outputs ^= {}];
	paths = EnumerateInternalOutputs[$net, spec];
	types = StripCoders[$net @@@ paths];
	ports = NetPort[FromNetPath[#]]& /@ paths;
	Scan[requiresOutput, ports];
	res = MapThread[makeArrayDecoder[#1, Identity, #2]&, {ports, types}];
	If[Length[res] === 1 && FreeQ[spec, All | Verbatim[_] | Verbatim[Alternatives]], First[res], res]
];

(*****************************************************************************
Parsing of NetPortGradient specs
*)

General::invlayerspec = "`` is not a valid specification for a layer.";
parseProp[ospec:NetPort[spec_, "IntermediateArrays"]] := Scope[
	path = ToNetPath[$net, spec];
	If[!ValidNetAssocQ[$net @@ path], ThrowFailure["invlayerspec", ospec]];
	If[$outputs === All, $outputs ^= {}];
	AppendTo[$outputs, ospec];
	With[{path = path, reader = If[$isBatched, NDArrayGetBatched, NDArrayGetUnbatchedSafe]},
		OutputDescriptor[ospec, MapAt[reader, $ExecutorArrays["Metrics", path], 1;;-2]]
	]
];

(* all grads *)
parseProp[NetPortGradient[All]] := 
	ToList[
		parseProp[NetPortGradient[All, "Inputs"]],
		parseProp[NetPortGradient[All, "Weights"]]
	];

parseProp[NetPortGradient[All, "WeightsVector"]] := Scope[
	i = 0;
	Do[
		$wgrads[++i] = path,
		{path, Keys @ NetArrays[$net, False]}
	];
	With[range = Range[i], OutputDescriptor[
		NetPortGradient[All, "WeightsVector"],
		NumericArray[arrayFlatten @ #, "Real32"]& @ Map[NDArrayGet, Lookup[$ExecutorArrays["WeightGradients"], range]]
	]]
];

General::netbadagg = "`` is not one of ``.";
parseProp[npg_NetPortGradient -> agg_String] := Scope[
	If[$isBatched, ThrowFailure["nobatchsupp", "Aggregating gradients"]];
	aggFunc = Lookup[$aggFunctions, agg, 
		ThrowFailure["netbadagg", agg, QuotedStringRow[Keys @ $aggFunctions, " or "]]];
	parseProp[npg] /. NDArrayGet -> NDArrayGet /* aggFunc
]

parseProp[NetPortGradient[All, "Weights"]] := Map[
	parseProp[NetPortGradient[FromNetPath[#]]]&,
	Keys @ NetArrays[$net, False]
	(* ^ exclude aux arrays *)
];

parseProp[NetPortGradient[All, "Inputs"]] := Map[
	parseProp[NetPortGradient[#]]&,
	Keys @ Select[StripCoders @ Inputs[$net], TType[#] === RealT&]
];

(* possibly an input gradient or weight gradient *)
parseProp[NetPortGradient[name_String | {name_String}]] := 
	If[KeyExistsQ[$net["Inputs"], name], 
		parseIGrad @ NetPortGradient[name],
		parseWGrad @ NetPortGradient[{name}]
	];

(* definitely a weight gradient *)
parseProp[npg:NetPortGradient[spec_List]] := 
	parseWGrad[npg];

parseWGrad[npg:NetPortGradient[spec_List]] := Scope[
	path = ToNetPath[$net, spec];
	If[!ArrayPathQ[path], ThrowFailure["badportd", npg]];
	If[MemberQ[NProperty[$net @@ Drop[path, -2], "AuxArrays"], Last[path]],
		General::auxnograd = "Auxiliary array `` does not possess a gradient.";
		ThrowFailure["auxnograd", npg]];
	$wgrads[npg] = path;
	(* TODO: rethink this when we support scalar arrays *)
	OutputDescriptor[npg, NDArrayGetSwitched @ $ExecutorArrays["WeightGradients", npg]]
];

parseIGrad[npg:NetPortGradient[name_String]] := Scope[
	If[$isBatched, ThrowFailure["nobatchsupp", "Obtaining gradients of input ports"]];
	type = StripCoders @ $net["Inputs", name];
	General::badportd = "`` does not correspond to an input or weight array of the net.";
	If[MissingQ[type], ThrowFailure["badportd", npg]];
	General::nointgrad = "Gradient is not available for integer-valued ``.";
	If[!FreeQ[type, _IndexIntegerT], ThrowFailure["nointgrad", NetPathString[NetPath["Inputs", name]]]];
	$igrads[npg] = ".Inputs." <> name;
	makeArrayDecoder[npg, Identity, type, "InputGradients"]
];

(*****************************************************************************
Recurrent states
*)

parseProp[NetPort[All, "States"]] := With[
	{reader = If[$isBatched, NDArrayGetBatchedSwitched, NDArrayGetUnbatchedSwitchedSafe]},
	OutputDescriptor[NetPort[All, "States"], reader /@ $ExecutorArrays["OutputStates"]]
];

(* we only use this easier way of handling scalars for state ports because its hard to get the
types out and it doesn't matter if its a bit slower *)
NDArrayGetUnbatchedSafe[nd_] := 
	If[NDArrayRank[nd] === 1, 
		First @ NDArrayGetNormal @ nd,
		NDArrayGetUnbatched @ nd
	];
NDArrayGetUnbatchedSwitchedSafe[nd_] := 
	If[NDArrayRank[nd] === 1, 
		First @ NDArrayGetNormal @ nd,
		NDArrayGetUnbatchedSwitched @ nd
	];

(*****************************************************************************
parsePropAll handles extracting all outputs, using a given property
*)

parsePropAll[prop_, <|key_ -> type_|>] :=
	makeDecoder[key, type, prop];

parsePropAll[prop_, types_] :=
	KeyValueMap[
		makeDecoder[#1, #2, prop]&,
		types
	];

(*****************************************************************************
makeDecoder looks up the decoder function to use for a given output type
*)

General::invoutport = "`` is not the name of an output port for the net."
makeDecoder[name_, _Missing, _] := ThrowFailure["invoutport", name];

General::nodeconport = "Property specification `` cannot be applied because port \"``\" does not have a NetDecoder associated with it."
makeDecoder[name_, type_, prop_] := UseMacros[
	requiresOutput[name];
	With[decoder = ToDecoderFunction[type, prop],
		If[decoder === $Failed, ThrowFailure["nodeconport", prop, name]];
		makeArrayDecoder[name, decoder, StripCoders @ type]
	]
];

PackageExport["$NNOutputHead"]

SetUsage @ "
$NNOutputHead is used to control the numerical output type (NumericArray vs packed array) \
of nets, layers and encoders/decoders. Its default value is Automatic and can be \
also set to NumericArray or List (packed array).
When it's set to Automatic, layers/containers/encoders will return packed array \
unless one or more of their inputs is a NumericArray. Notice that although undocumented, \
some encoders have a \"Bypass\" mode which activates when they are fed NumericArray. Hence, \
in those cases they will return NumericArray.
In most of the cases decoders will not output numerical data, hence they are largely \
unaffected by $NNOutputHead."

$NNOutputHead = Automatic;

NDArrayGetSwitched /; $ReturnNumericArray := NDArrayGet
NDArrayGetSwitched /; !$ReturnNumericArray := NDArrayGetNormal

NDArrayGetBatchedSwitched /; $ReturnNumericArray := NDArrayGetBatched
NDArrayGetBatchedSwitched /; !$ReturnNumericArray := NDArrayGetBatchedNormal

NDArrayGetUnbatchedSwitched /; $ReturnNumericArray := NDArrayGetUnbatched
NDArrayGetUnbatchedSwitched /; !$ReturnNumericArray := NDArrayGetUnbatchedNormal

PackageScope["SwitchNumericArrayFlag"]
PackageScope["$ReturnNumericArray"]
(* this function sets $ReturnNumericArray if it is Automatic, based on whether
a numericarray is present in the input data *)
SetAttributes[SwitchNumericArrayFlag, HoldFirst];

(* the FreeQ is somewhat hacky, as the NumericArray might go into a NetEncoder 
   that is expecting it anyway, that doesn't mean we want to return NumericArrays 
   necessarily. but then the whole Automatic behavior is a hacky transitional
   thing anyway *)
SwitchNumericArrayFlag[body_, inputData_] := Block[
	{$ReturnNumericArray = returnNumericArrayQ[inputData]}, 
	body
];

returnNumericArrayQ[inputData_] /; MatchQ[$NNOutputHead, NumericArray|List] :=  ($NNOutputHead === NumericArray);
returnNumericArrayQ[inputData_] /; MatchQ[$NNOutputHead, Automatic] := !FreeQ[inputData, _NumericArray];


PackageScope["makeArrayDecoder"]

Clear[makeArrayDecoder];

Default[makeArrayDecoder, 4] = "Outputs";

(* we must treat scalars differently as a rank-0 numeric array cannot be returned from librarylink *)
makeArrayDecoder[name_, decoder_, ScalarT, key_.] /; !$isBatched := 
	If[decoder === Identity,
		OutputDescriptor[name, First @ NDArrayGetBatchedNormal @ $ExecutorArrays[key, name]],
		OutputDescriptor[name, First @ decoder @ NDArrayGetBatched @ $ExecutorArrays[key, name]]
	]

(* case where there is no decoder function: we use the NumericArray flag to decide what to do *)
makeArrayDecoder[name_, Identity, _, key_.] := If[$isBatched,
	OutputDescriptor[name, NDArrayGetBatchedSwitched @ $ExecutorArrays[key, name]],
	OutputDescriptor[name, NDArrayGetUnbatchedSwitched @ $ExecutorArrays[key, name]]
]

makeArrayDecoder[name_, decoder_, _, key_.] := If[$isBatched,
	OutputDescriptor[name, decoder @ NDArrayGetBatched @ $ExecutorArrays[key, name]],
	OutputDescriptor[name, First @ decoder @ NDArrayGetBatched @ $ExecutorArrays[key, name]]
]



PackageScope["OutputDescriptor"]

SetHoldRest[OutputDescriptor]


(*****************************************************************************
EnumerateInternalOutputs scans net for things that match a pattern-based NetPort
spec
*)

$litNPelem = _String | _Integer;

(* fast case: we have a literal port or layer, we can just canonicalize without traversing whole net *)
EnumerateInternalOutputs[net_NetP, spec:$litNPelem | {$litNPelem..}] := Scope[
	$path = ToNetPath[net, spec];
	General::badporto = "`` does not correspond to an output of the net, or a subnet of the net, or a layer.";
	If[FailureQ[$path], ThrowFailure["badporto", NetPort[spec]]];
	General::npopinv = "It is not possible to extract the output of a net within an operator.";
	If[!FreeQ[$path, "Parameters"], ThrowFailure["npopinv"]]; 
	(* ^ a little sloppy, will false positive if there is a layer called Parameters *)
	If[MatchQ[$path, NetPath[___, "Outputs", _]], Return[{$path}]];
	layer = UnsafeQuietCheck[net @@ $path, $Failed];
	ports = OutputPaths[layer];
	If[!MatchQ[ports, {__NetPath}], ThrowFailure["badporto", NetPort[spec]]];
	ports
];

(* slow case: we have a pattern, do a slow traverse and collect up all matching outputs (outermost only) *)
EnumerateInternalOutputs[net_NetP, spec_] := Scope[
	$patt = ToNetPathPattern[net, spec];
	General::badportpo = "`` is not a valid port or port pattern, which should be a hierarchical specification using NetPort, possibly including All, or Span.";
	If[FailureQ[$patt], ThrowFailure["badportpo", NetPort[spec]]];
	res = DeleteDuplicates @ Flatten @ ReapBag @ EnumerateMatchingOutputs[net];
	General::emptportpo = "Port pattern `` did not match any ports.";
	If[res === {}, ThrowFailure["emptportpo", res]];
	res
];

DeclareMethod[EnumerateMatchingOutputs, EnumerateLayerOutputs, EnumerateContainerOutputs];

(* we deliberately don't enumerate over operator's inner nets, becuase how to even read them off his highly
complicated and requires we inject ourselves into the unrolling process to collect up the right e.g. seq nodes *)

EnumerateLayerOutputs[net_] := Scope[
	oports = OutputPaths[net];
	If[MatchQ[$path, $patt], 
		SowBag @ oports, 
		SowBag @ Cases[oports, $patt]
	]
];

EnumerateContainerOutputs[net_] := If[
	MatchQ[$path, $patt], SowBag @ OutputPaths[net], 
	ScanNodes[EnumerateMatchingOutputs, net]
];

