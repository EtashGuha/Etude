Package["NeuralNetworks`"]


PackageExport["NetChain"]

SetupGenericDispatch[NetChain, True];

(ng_NetChain ? System`Private`HoldEntryQ) := 
	UseMacros @ RuleCondition @ CatchFailureAsMessage[NetChain, make[ng]];

SetHoldAll[make];

make[nc:NetChain[<|"Type" -> "Chain", ___|>]] := 
	UpgradeAndSeal11V0Net[nc];

make[nc:NetChain[<|"Type" -> "Chain", ___|>, _Association]] :=
	SealNet[nc];

make[NetChain[layers_Association | layers_List, rules___Rule]] := 
	toNetChain[ToContainerLayers[layers], {rules}];

make[ng_NetChain] /; PatternFreeQ[Unevaluated[ng]] := 
	(CheckArgumentCount[ng, 1, 1]; $Failed);

make[_] := Fail;

PackageScope["toNetChain"]

toNetChain[layers_, rules_, sharedArrays_:<||>] := Scope[
	If[Length[layers] === 0, ThrowFailure["netempty", NetChain]];
	ins = Normal @ layers[[All, "Inputs"]];
	outs = Normal @ layers[[All, "Outputs"]];
	$currentNodes = layers; (* for error reporting *)
	$currentNodeName = "layer";
	conns = Flatten @ MapThread[glueLayers, {Prepend[outs, None], Append[ins, None]}];
	(* handle custom type specifications *)
	inputs = ins[[1,2]];
	outputs = outs[[-1,2]];
	istates = Association @ KeyValueMap[interiorStateRules["Nodes"], layers];
	Which[
		rules === {}, 
			Null,
		ListQ[rules],
			$CurrentArgumentHead = NetChain;
			kin = Keys[inputs];
			kout = Keys[outputs];
			extra = Complement[Keys[rules], kin, kout];
			If[extra =!= {} && FreeQ[rules, $Raw], ThrowFailure["netinvcport", First[extra]]];
			Set[inputs[#1],  ParseInputSpec[#1,  inputs[#1],  #2]]& @@@ FilterRules[rules, kin];
			Set[outputs[#1], ParseOutputSpec[#1, outputs[#1], #2]]& @@@ FilterRules[rules, kout],
		AssociationQ[rules], (* only used internally by NetChainTakeDrop *)
			inputs = Lookup[rules, "Inputs", inputs];
			outputs = Lookup[rules, "Outputs", outputs];
	];
	layers = StripCoders[layers];
	{layers, sarrays} = HoistSharedArrays[layers];
	assoc = Association[{
		"Type" -> "Chain",
		"Nodes" -> layers,
		"Edges" -> conns,
		"Inputs" -> inputs,
		"Outputs" -> outputs,
		If[istates === <||>, Nothing, "InteriorStates" -> istates],
		If[sarrays === <||>, Nothing, "SharedArrays" -> sarrays]
	}];
	assoc = AttachSharedArrays[assoc, sharedArrays];
	CheckForMultiports[assoc];
	ConstructWithInference[NetChain, assoc]
];

NetChain::netinvcport = "`` is neither a valid input or output port for the given NetChain.";

glueLayers[None, bname_ -> b_] :=
	NetPath["Nodes", bname, "Inputs", #] -> NetPath["Inputs", #]& /@ Keys[b]; 

glueLayers[aname_ -> a_, None] :=
	NetPath["Outputs", #] -> NetPath["Nodes", aname, "Outputs", #]& /@ Keys[a];

glueLayers[aname_ -> a_, bname_ -> b_] := Scope[
	{aport, atype} = get1port[aname, "Outputs", a];
	{bport, btype} = get1port[bname, "Inputs", b];
	res = UnifyTypes[atype, btype];
	If[FailureQ[res], edgeTypeError[aport, bport, atype, btype]]; (* <- lives in NetGraph.m *)
	bport -> aport
];

General::notuport = "Layer `` should have exactly one `` port.";
get1port[name_, ptype_, <|portname_ -> porttype_|>] := {NetPath["Nodes", name, ptype, portname], porttype};
get1port[name_, ptype_, _] := ThrowFailure["notuport", name, StringDrop[ptype, -1]];


PackageExport["$NetChainInteractivity"]

$NetChainInteractivity := $NetInteractivity;

PackageScope["netChainGrid"]

netChainGrid[assoc_] := Scope @ LengthVarScope[assoc,
	UnpackAssociation[assoc, nodes, inputs, outputs];
	$hasuninit = False; 
	rows = ToList[
		KeyValueMap[toInputRow, inputs],
		KeyValueMap[toLayerRow, nodes],
		KeyValueMap[toOutputRow, outputs]
	];
	grid = Grid[rows, Alignment -> Left, Spacings -> 1.1];
	If[$NetChainInteractivity, With[
		{gridBoxes = ToBoxes[grid], assoc2 = ReplaceArraysWithDummies[assoc]},
		RawBoxes @ DynamicModuleBox[
			{assoc3 = assoc2, opart, part, selected = Null},			
			DynamicBox[
				GridBox[
					List[
						List @ MouseClickBoxes[gridBoxes, 
							If[ListQ[part = MouseAnnotation[]],
								If[opart === part, 
									selected = Null; opart = Null,
									selected = Part[assoc3, Sequence @@ part]; opart = part;
								];
							]
						], 
						fmtSelected[selected, part, GetSharedArrays[assoc3]]
					], 
					GridBoxSpacings -> {"Columns" -> {{1}}},
					GridBoxAlignment -> {"Columns" -> {{Left}}},
					GridFrameMargins -> {{0, 0}, {0, 0}}
				], 
				TrackedSymbols :> {selected}
			],
			Initialization :> {NetChain}
		]],
		grid
	]
];

SetAttributes[MouseClickBoxes, HoldRest];
MouseClickBoxes[boxes_, code_] := TagBox[boxes, 
	EventHandlerTag[{
		"MouseClicked" :> code, Method -> "Preemptive", 
		PassEventsDown -> Automatic, PassEventsUp -> True
	}]
];


Clear[fmtSelected];

fmtSelected[Null, _, _] := 
	Nothing;

fmtSelected[type_, {"Inputs"|"Outputs", name_}, _] := 
	List @ typeInfo[name -> type];

fmtSelected[layer_, {"Nodes", name_}, sa_] := Scope[
	$AmbientSharedArrays ^= Join[$AmbientSharedArrays, sa];
	List @ itemInfo[name -> layer]
];

(* this deals with the fact that InitializedNetQ won't register uninitialized shared arrays when applied
to a subnetwork, becuase it depends on the uninitialized shared arrays showing up in the root *)
SubInitializedNetQ[assoc_] :=
	InitializedNetQ[assoc] && FreeQ[assoc, ns_NetSharedArray /; UninitializedArrayQ[ns]];

(* TODO: Handle last layer properly *)
toLayerRow[name_, assoc_] := Scope[
	output = fmtItem @ First @ assoc["Outputs"];
	summary = Style[SummaryForm[assoc], If[SubInitializedNetQ[assoc], Black, $hasuninit = True; $uninitializedColor]];
	If[!FreeQ[assoc, NetSharedArray], summary = Row[{summary, " \[UpperRightArrow]"}]];
	selector["Nodes", name] /@ {
		Style[name, Gray], 
		summary,
		output
	}
];

toInputRow[name_, type_] := 
	selector["Inputs", name] /@ {"", name, fmtInputItem @ type};

fmtInputItem[t_] := fmtItem[t];

fmtInputItem[t:TensorT[{n_}, enc_NetEncoder]] := Column[{
	fmtItem[t], 
	fmtItem[TensorT[{n}, CoderType[enc]]]}, 
	BaselinePosition -> 2
];

fmtInputItem[enc_NetEncoder] := Column[{CoderKind[enc], fmtItem[CoderType[enc]]}, BaselinePosition -> 2];


toOutputRow[name_, type_] := 
	selector["Outputs", name] /@ {"", name, fmtOutputItem @ type};

fmtOutputItem[t_] := fmtItem[t];

fmtInputItem[t:TensorT[{n_}, enc_NetEncoder]] := Column[{
	fmtItem[t], 
	fmtItem[TensorT[{n}, CoderType[enc]]]}, 
	BaselinePosition -> 2
];

fmtOutputItem[dec_NetDecoder] := CoderKind[dec];


selector[part___] := If[$NetChainInteractivity, MouseAppearance[Annotation[#, {part}, "Mouse"], "LinkHand"]&, Identity];

DefineCustomBoxes[NetChain, 
	NetChain[assoc_Association, _Association] ? System`Private`HoldNoEntryQ :> formatNetChain[assoc]
];

PackageScope["$NetChainIconBoxes"]

$NetChainIconBoxes := $NetChainIconBoxes = Uncompress @ "
1:eJxTTMoPSmNhYGAo5gYS7kWJBRmZycVO+RVpXCBBkIxPZnFJGiOIxwkkQoDy2XmpxcVF6ds/rDac2G
efxoqikhnE4wASQe5Ozvk5+UVF7WI3z30Pfm6PycBm7Ma3Cd6TlCZD5diBhH9BYnJmSWURAxh8gMqArP
DKz8xzyy/KTWNCdkIwyEG+mSWpRVAtKg4QV/FBtaSmOJcWlaWCfMmI6UtUn8B4mSCDMkH2gFloSplw86
BuMHeAMB6EOWCVt8MmH1SakwqOGLB7nXPyi1NTUG2GOGY0BgjHQIMTgRhwoyQGUK2HB49bYnIqZvBgia
F/Vypeqhk+sS96pWbIsUbmlX3RGpmoFOv79/HFAhMsSN0yc3JwBikzbh56kGKRZCROEtVKFkKRwaAPC+
xIrJHRYE6sfDD2yNTHJU9G3ECi5JZ90aqPl3yTBICR5MXDpN0udpKcuMFiGa3jhi6WoPqLcAKwgEYQQx
T2CNQjUv4AjgRggSRP2DWw5MQQgt00AyLlD2BPrnDzMeTJSI5BO+RaXwfesC+qz9pTMlnihX1Rtcg694
dVb4ZnUdFgi78oOOBErDz2lIIwnwpFxdYTZfvmS523L7r9sw4YO4/ti8CRtePZaFFBflHhTKAosCVSHl
dR4UxKUQFvRuAqCuyIlMdRVMDNp0ZR8Xjp7CMKGx7ZQ8uMB/ZFvKCK6+ar4VlUHHAlUBR4EiuPPaUgzK
dCUTFrJgjctC/6BGpURNyEtSqejhYV5BcVXviLggZXIuVxFRVeJBUVngSKAjci5XEVFZ5I8qjRQXpyPK
qwoShj4kN7TAbFRQWexElSSsFXAGELPhsCHT5LWPCF45dngOkftD51xe/TBmf8PoXLM2B0fXmADM/cxP
RU38Si9My8YrRsCVbDCVMTnFmVimp+pgqIB1LgWFqSn5tYkpmMplkAyAjIyS8JSsxLTw1ITEnJzEsHWw
IAMXiZnA==";

formatNetChain[assoc_Association] := OptimizedArrangeSummaryBox[NetChain, 
	If[InitializedNetQ[assoc], Identity, fmtUninitializedIcon] @ $NetChainIconBoxes, 
	fmtCompactContainerInfo[assoc],
	WithAmbientSharedArrays[assoc, {{ToBoxes @ netChainGrid[assoc]}}],
	True
]

RunInitializationCode[
	Format[HoldPattern[nc:NetChain[assoc_Association, _Association]] ? System`Private`HoldNoEntryQ, OutputForm] := 
		StringJoin[
			"NetChain[<", 
			IntegerString @ Length @ assoc["Nodes"],
			">]"
		];
]

PackageExport["$LowercaseTraditionalForm"]

$LowercaseTraditionalForm = True;

PackageScope["TFCase"]
TFCase[str_] := If[$LowercaseTraditionalForm, ToLowerCase[str], str];


PackageScope["makeSubNetChain"]

makeSubNetChain[old_, layers_, shared_:<||>] := Scope[
	oldKeys = Keys @ old["Nodes"];
	ioSpecs = chainIOSpecs[old, 
		KeyExistsQ[layers, First[oldKeys]], 
		KeyExistsQ[layers, Last[oldKeys]]
	];
	If[DigitStringVectorQ[oldKeys],
		layers = First @ RemapKeys[layers];
	];
	sharedArrays = JoinSharedArrays[GetSharedArrays[old], shared];
	toNetChain[layers, ioSpecs, sharedArrays]
];


PackageExport["GetNodes"]

SetUsage @ "
GetNodes[NetGraph[$$]] gives an association of the vertices within a NetGraph.
GetNodes[NetChain[$$]] gives an association of the layers within a NetChain.
GetNodes[net, True] will give a list of the assoc has integer keys."

GetNodes[HoldPattern @ NetChain[assoc_Association, _]] := WithAmbientSharedArrays[assoc,
	Map[ConstructNet, assoc["Nodes"]]
];

GetNodes[net_, True] := Scope[
	nodes = GetNodes[net];
	If[DigitStringKeysQ[nodes], Values[nodes], nodes]
];

NetChain /: Length[nc_NetChain ? ValidNetQ] := 
	Length[NData[nc]["Nodes"]];

NetChain /: Normal[nc_NetChain] := GetNodes[nc, True];


PackageExport["chainIOSpecs"]

chainIOSpecs[assoc_, in_:True, out_:True] := Scope[
	specs = KeyTake[assoc, {"Inputs", "Outputs"}];
	If[!in, KeyDropFrom[specs, "Inputs"]];
	If[!out, KeyDropFrom[specs, "Outputs"]];
	specs
];


PackageExport["ABCNetChain"]

(* this function makes tests easier to write *)

ABCNetChain[layers_, rules___Rule] := Scope[
	keys = FromLetterNumber @ Range @ Length @ layers;
	NetChain[Thread[keys -> layers], rules]
];
