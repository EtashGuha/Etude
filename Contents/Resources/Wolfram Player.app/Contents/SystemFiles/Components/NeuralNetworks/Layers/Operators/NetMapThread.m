Inputs: <||>

Outputs: <||>

Suffix: "Operator"

Parameters:
	$Net: NetT[]
	$Level: ValidatedParameterT[Identity]
	$$InputNames: ListT[SizeT, StringT]
	$$OutputNames: ListT[SizeT, StringT]
	$$MappedShape: SizeListT[]
	$$MappedDepth: SizeListT[]

ReshapeParams: {}

MinArgCount: 1
PosArgCount: 2

ArgumentRewriter: setupInputsAndOutputs

AllowDynamicDimensions: True

(* %TODO move other layers to use FailConstruction. make it auto-fill in the layer symbol in define. *)

checkRule[name_String -> level_Integer] := Which[
	!KeyExistsQ[$inputs, name], levelPanic["Level spec refers to the port ``, but net only has ports ``.", name, Keys[$inputs]],
	level < 0, levelPanic["Level `` cannot be less than 0.", level],
	True, True
];	

levelPanic[args__] := FailConstruction[NetMapThreadOperator, args];

setupInputsAndOutputs[{net_, rest___Rule}] := setupInputsAndOutputs[{net, 1, rest}];

setupInputsAndOutputs[{net_, level_, rest___}] := Scope[
	net = ToLayer[net];
	$inputs = Inputs[net]; outputs = Outputs[net];
	Switch[level,
		Automatic, level = 1,
		_Integer ? Positive, Null,
		_Integer, levelPanic["Level `` cannot be less than 1.", level],
		x_Association|x_List /; AllTrue[Normal @ x, checkRule], Null,
		_, levelPanic["Level spec `` must be a positive integer or rules mapping a subset of the ports to integers. The available ports are ``.", level, Keys @ $inputs]
	];
	inames = Keys @ $inputs;
	depths = If[IntegerQ[level], 
		Table[level, Length[inames]],
		Lookup[level, inames, 0]
	];
	maxDepth = Max[depths];
	If[maxDepth == 0, (* Can happen if $Level is an empty association or an association with 0s only *)
		levelPanic["At least one positive level to map over must be given"]
	];
	SetCurrentLayerInputs @ Map[addRank[#, Each[depths]]&, $inputs];
	SetCurrentLayerOutputs @ Map[addRank[#, maxDepth]&, outputs];
	shape = Table[SizeT, maxDepth];
	shape[[1]] = NewLengthVar[];
	SetCurrentLayerParameters @ {
		"$MappedShape" -> shape,
		"$InputNames" -> inames, 
		"$OutputNames" -> Keys @ outputs, 
		"$MappedDepth" -> depths
	};
	{$Raw[net], level, rest}
];

addRank[t_, n_] := TensorT[Table[SizeT, n], t];
addRank[t_, 0] := t; 

setupInputsAndOutputs[e_] := e;

makeIORules[io_, port_, depth_] := With[{
	interior = NetPath["Parameters", "Net", io, port],
	exterior = NetPath[io, port],
	mapshape = NetPath["Parameters", "$MappedShape"]},
	{
		interior -> RawComputedType[TensorT[Drop[TDimensions @ exterior, depth], TType[exterior]], IntegerQ @ TRank @ exterior],
		mapshape -> RawComputedType[StartsWithT @ Take[TDimensions @ exterior, depth], IntegerQ @ TRank @ exterior],
		exterior -> RawComputedType[TensorT[Take[mapshape, depth], interior], ListQ @ mapshape]
	}
	(*  NOTE: normally having both rules 1 and 3 would not be necessary, because info can flow both directions
	in a rule, but here the triggers are quite different. things would be easier if we the ranks were fixed. *)
];

RuntimeInferenceRules: Function @ Scope[
	{inames, onames, depths} = Lookup[#Parameters, {"$InputNames", "$OutputNames", "$MappedDepth"}]; 
	Join[
		MapThread[makeIORules["Inputs", #1, #2]&, {inames, depths}],
		Map[makeIORules["Outputs", #1, Max @ depths]&, onames]
	]
]

SummaryFunction: Function[
	HoldForm[NetMapThreadOperator] @ SummaryForm[#Net]
]

Writer: Function @ Scope[
	names = #$InputNames;
	inputs = GetInput[#, "Batchwise"]& /@ names;
	depths = #$MappedDepth;
	maxDepth = Max[depths];
	deepestInput = Part[inputs, ArrayMaxIndex[depths]];

	(* broadcast all inputs against the deepest input *)
	inputs = MapThread[
		SowBroadcastAgainst[#1, Range[#2+1, maxDepth], deepestInput]&, 
		{inputs, depths}
	];

	(* flatten the dims 0 through maxdepth of all inputs *)
	flattened = SowFlatten[#, maxDepth]& /@ inputs;

	(* apply the mapped net *)
	result = SowInnerNet[
		AssociationThread[names, flattened], (* inputs *)
		{"Parameters", "Net"}, #Net
	];

	(* unflatten the output *)
	KeyValueScan[
		SetOutput[#1, SowUnflatten[#2, deepestInput, maxDepth]]&,
		result
	]; 
]

Tests: {
	{Hold @ LinearLayer[2, "Input" -> 1]} -> "3*2_TqdzaQfy9+w_fPNZLgMdZc0=5.316732e+0",
	{Hold @ MeanAbsoluteLossLayer[], 1, "Input" -> {3, 2}} -> "3_Lj/eZ+QncTE_FOiV+Q2eyGg=7.552859e-1",
	{Hold @ MeanAbsoluteLossLayer[], 1, "Input" -> {"n", 2}} -> "3_Lj/eZ+QncTE_GfpDTKuZJ4M=7.552859e-1",
	{Hold @ MeanAbsoluteLossLayer[], 2, "Input" -> {4, 3, 2}} -> "4*3_XlYUowso1mg_P4M5QNXdXlo=3.571883e+0",
	{Hold @ MeanAbsoluteLossLayer[], 2, "Input" -> {"n", 3, 2}} -> "3*3_c1E2ynOzLUI_XKIQQyeOVls=2.445744e+0",
	{Hold @ MeanAbsoluteLossLayer[], {"Input" -> 1, "Target" -> 0}, "Input" -> {3, 2}, "Target" -> {2}} -> "3_VdxWdwVvork_JsVb2UqXpDA=8.412410e-1",
	{Hold @ MeanAbsoluteLossLayer[], <|"Input" -> 1, "Target" -> 0|>, "Input" -> {"n", 2}, "Target" -> {2}} -> "3_VdxWdwVvork_WseLklXy0s4=8.412410e-1",
	(* Error handling *)
	{LinearLayer[], 0} -> "Invalid argument for NetMapThreadOperator: Level 0 cannot be less than 1.",
	{LinearLayer[], <||>} -> "Invalid argument for NetMapThreadOperator: At least one positive level to map over must be given",
	{LinearLayer[], <|"Input" -> 0|>} -> "Invalid argument for NetMapThreadOperator: At least one positive level to map over must be given",
	{LinearLayer[], 1->1} -> "1 -> 1 is not a valid option for NetMapThreadOperator.",
	{LinearLayer[], Infinity} -> "Invalid argument for NetMapThreadOperator: Level spec \[Infinity] must be a positive integer or rules mapping a subset of the ports to integers. The available ports are {Input}.",
	{Hold @ MeanAbsoluteLossLayer[], <|"Input" -> 1, "Typo" -> 0|>, "Input" -> {"n", 2}, "Target" -> {2}} -> "Invalid argument for NetMapThreadOperator: Level spec refers to the port Typo, but net only has ports {Input, Target}.",
	{Hold @ MeanAbsoluteLossLayer[], <|"Input" -> 1, "Target" -> 0|>, "Input" -> {"n", 2}, "Typo" -> {2}} -> "\"Typo\" is not a known parameter for NetMapThreadOperator. Allowed parameters include: \"Net\", \"Level\", \"Input\", and \"Target\".",
	{Hold @ MeanAbsoluteLossLayer[], <|"Input" -> 1, "Target" -> -1|>, "Input" -> {"n", 2}, "Target" -> {2}} -> "Invalid argument for NetMapThreadOperator: Level -1 cannot be less than 0."
}
