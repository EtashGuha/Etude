Inputs: <||>

Outputs: <||>

Suffix: "Operator"

Parameters:
	$Net: NetT[]
	$Recurrence: ValidatedParameterT[Identity]
	$ConstantPorts: Defaulting[ListT[StringT], {}]
	$OutputPorts: ListT[StringT]
	$$InputNames: ListT[SizeT, StringT]
	$$StateNames: ListT[SizeT, StringT]
	$$OutputNames: ListT[SizeT, StringT]
	$$SequenceLength: LengthVar[]

ReshapeParams: {$$SequenceLength}

Upgraders: {
	"11.3.9" -> AddParam[{
		"Recurrence" -> ValidatedParameter[{"Output" -> "State"}],
		"ConstantPorts" -> {},
		"OutputPorts" -> {"Output"},
		"$InputNames" -> {"Input"},
		"$StateNames" -> {"State"},
		"$OutputNames" -> {"Output"}
	}]
}

MinArgCount: 2
PosArgCount: 4

ArgumentRewriter: setupInputsAndOutputs

AllowDynamicDimensions: True

Clear[setupInputsAndOutputs];

setupInputsAndOutputs[{OptionsPattern[]}] :=
	$Failed;

setupInputsAndOutputs[{net_, recurrences:Except[_Rule]:Automatic, constantPorts:Except[_Rule]:{}, outputPorts:Except[_Rule]:All, opts:OptionsPattern[]}] := Scope[
	
	net = ToLayer[net];
	
	$inputs = StripCoders @ Inputs[net]; 
	$outputs = StripCoders @ Outputs[net]; 
	
	If[recurrences === Automatic,
		If[MatchQ[$inputs, KeyValuePattern["State" -> _]] && MatchQ[$outputs, KeyValuePattern["Output" -> _]],
			recurrences = {"Output" -> "State"},
			constructFail["if no recurrences (second argument) are specified, the folded net (first argument) should have at least one input called \"State\" and one output \"Output\"."]
		]];

	If[outputPorts === All, outputPorts = Keys @ $outputs];
	testPortList[outputPorts, Keys @ $outputs, "output ports", "output"];
	testPortList[constantPorts, Keys @ $inputs, "constant ports", "input"];
	recurrences = Map[procRecRule, ToList @ recurrences];

	statePorts = Values @ recurrences;
	inputPorts = Complement[Keys @ $inputs, statePorts, constantPorts];

	If[!DuplicateFreeQ[statePorts], 
		constructFail["recurrences `` specify multiple sources for an input.", QuotedStringForm @ recurrences]];
	
	lvar = NewLengthVar[];

	SetCurrentLayerInputs @ IMap[
		If[MemberQ[constantPorts, #1], #2, ChannelT[lvar, #2]]&,
		KeyDrop[$inputs, statePorts]
	];

	If[inputPorts === {},
		constructFail["there should be at least one non-recurrent, non-constant input."]];

	SetCurrentLayerOutputs @ Map[
		ChannelT[lvar, #1]&, 
		KeyTake[$outputs, outputPorts]
	];
	
	SetCurrentLayerStates @ KeyTake[$inputs, statePorts];
	(* we choose the state names based on the input port names *)
	
	SetCurrentLayerParameters @ {
		"$InputNames" -> inputPorts,
		"$StateNames" -> statePorts,
		"$OutputNames" -> outputPorts,
		"$SequenceLength" -> lvar
	};
	
	{$Raw[net], recurrences, constantPorts, outputPorts, opts}
];

testPortList[other_, _, thing_, ptype_] :=
	constructFail["`` `` should be a list of `` ports.", thing, QuotedStringForm @ other, ptype];

testPortList[spec_List ? StringVectorQ, validSet_, thing_, ptype_] :=
	TestSubsetQ[validSet, spec, missPortFail[thing, ptype, #, validSet]&];

constructFail[args__] := FailConstruction[NetFoldOperator, args];

missPortFail[thing_, type_, port_, avail_] :=
	constructFail[ 
		"`2` specifies invalid `1` port `3`; net has `1` port`4` `5`.", 
		type, thing, QuotedStringForm @ port, If[Length[avail] > 1, "s", ""], QuotedStringRow[avail, " and "]
	];

procRecRule[from_String] := procRule[from -> from];

recForm[spec_] := StringForm["recurrence ``", QuotedStringForm @ spec];

procRecRule[spec:(oname_String -> iname_String)] := Scope[
	otype = Lookup[$outputs, oname, missPortFail[recForm[spec], "output", oname, Keys @ $outputs]];
	itype = Lookup[$inputs, iname, missPortFail[recForm[spec], "input", iname, Keys @ $inputs]];
	If[!UnifiableQ[itype, otype], 
		{ostr, istr, kind} = toTypeMismatchData[otype, itype];
		constructFail["`` of output port \"``\" (``) is not compatible with input port \"``\" (``)",
			kind, oname, ostr, iname, istr]];
	spec
];

procRecRule[e_] := constructFail["`` is not a valid recurrence.", e];

(*********************)

makeIORule[io_, port_, mapq_] := Scope[
	interior = NetPath["Parameters", "Net", io, port];
	slen = NetPath["Parameters", "$SequenceLength"];
	NetPath[io, port] -> If[mapq, ChannelT[slen, interior], interior]
];

makeRecRule[oname_, iname_] := Scope[
	input = NetPath["Parameters", "Net", "Inputs", iname];
	output = NetPath["Parameters", "Net", "Outputs", oname];
	state = NetPath["States", iname];
	{input -> output, input -> state}
];

RuntimeInferenceRules: Function @ Scope[
	{recurrences, inames, cnames, snames, onames} = Lookup[#Parameters, 
		{"Recurrence", "$InputNames", "ConstantPorts", "$StateNames", "$OutputNames"}];
	Flatten @ Join[
		Map[makeIORule["Inputs", #, True]&, inames],
		Map[makeIORule["Outputs", #, True]&, onames],
		Map[makeIORule["Inputs", #, False]&, cnames],
		makeRecRule @@@ StripVP[recurrences]
	]
]

(* TODO: fix problem with dims not unifying, e.g. 
avg = NetGraph[{ThreadingLayer[0.1*#1 + 0.9*#2 &]}, {{NetPort["Input"], NetPort["State"]} -> 1}]
NetFoldOperator[avg, {"Output" -> "State"}, "Input" -> {"n"}]
*)

SummaryFunction: Function[
	HoldForm[NetFoldOperator] @ SummaryForm[#Net]
]

Writer: Function @ Scope[

	sequenceInputs = AssociationMap[GetInput[#, "Timewise"]&, #$InputNames];
	constantInputs = AssociationMap[GetInput, #ConstantPorts];
	initialStates = AssociationMap[GetState, #$StateNames];
	recurrences = StripVP[#Recurrence];
	slen = #$SequenceLength;
	outputPorts = DeleteDuplicates @ Join[Keys @ recurrences, #$OutputNames];

	outputs = SowForEach[
		With[{c = constantInputs}, SowSubNet["Net", Join[#, c]]&],
		slen,
		sequenceInputs,
		initialStates,
		outputPorts
	];

	outputs = Map[ToMetaNode[#, slen, True]&, outputs];

	SetState[#2, SowMetaLast[outputs[#1]]]& @@@ StripVP[#Recurrence];
	Scan[SetOutput[#, outputs[#]]&, #$OutputNames];
]

Tests: {
	{Hold @ NetGraph[{2,Times}, {NetPort["Input"]->1,{1,NetPort["State"]}->2}], {"Output" -> "State"}, "Input"->SequenceOf[2]} -> "3*2_detGAPjfd7M_OXUcrmPXDXA",
	{Hold @ NetGraph[{ThreadingLayer[Plus]}, {{NetPort["Input"],NetPort["State"]}->1}], {"Output" -> "State"}, "Input"->SequenceOf[2]} -> "3*2_QlYGSoBwysk_D17I3FDKchA=4.378344e+0",
	
	{Hold[NetGraph[{ThreadingLayer[Plus]}, {{NetPort["Input"], NetPort["State"], NetPort["Constant"]} -> 1}]], {"Output" -> "State"}, {"Constant"}, "Input" -> SequenceOf[2]} -> "3*2_ZeHTLedPee4_fceio7MFFHE=1.156559e+1",
	{
		Hold[NetGraph[
			{ThreadingLayer[Plus],AttentionLayer["Dot", "Key"->{"N", 3}, "Value"->{"N", 3}, "Query"->{3}]},
			{{NetPort["Input"], NetPort["Query"]}-> 1->NetPort[2, "Query"]}
		]],
		{"Output" -> "Query"}, {"Key", "Value"}
	} -> "3*3_RB73OezICbg_XCCB6WZP/pk=4.619017e+0",
	{
		Hold[NetGraph[
			{ThreadingLayer[Plus],AttentionLayer["Key"->{"N", 2}, "Value"->{"N", 3}, "Query"->{3}]},
			{{NetPort["InputSequence"], NetPort["Query"]}-> 1->NetPort[2, "Query"]}
		]],
		{"Output" -> "Query"}, {"Key", "Value"}
	} -> "3*3_OoF6JH99Yos_fyQuf017wos=4.056626e+0",
	{
		Hold[NetGraph[
		<|"reshape" -> ReshapeLayer[{1, 12}], "lstm" -> LongShortTermMemoryLayer[10], "projection-in" ->
	LinearLayer[10], "projection-out" -> LinearLayer[8]|>,
		{
			"reshape" -> "lstm" -> "projection-out" -> NetPort["OutputState"],
			NetPort["CellState"] -> NetPort["lstm","CellState"] -> NetPort["OutputCellState"],
			NetPort["State"] -> "projection-in" -> NetPort["lstm","State"]
			},
		"Input" -> 12,
		"State" -> 8,
		"CellState" -> 10
	]], {"OutputState" -> "State", "OutputCellState" -> "CellState"}, {}, {"OutputState"}} -> "3*8_EZAQHSRRmQI_FnI3621ejN4=2.178548e+1"

}