Package["NeuralNetworks`"]



PackageExport["NetExtract"]

Clear[NetExtract];

$ambientNet = None;

NetExtract[net_NetStateObject, spec_] := CatchFailureAsMessage[
	If[spec =!= "States", ThrowFailure["netnopart", spec]];
	GetNetStateObjectCurrentStates[net]
];

NetExtract[coder:CoderP ? System`Private`HoldNoEntryQ, spec_] := CatchFailureAsMessage[
	FromInternalValue @ CoderExtract[coder, spec]
]

NetExtract[head_[assoc_Association, _] ? System`Private`HoldNoEntryQ, spec_] := Scope[
	$ambientNet = assoc; 
	$AmbientSharedArrays = GetSharedArrays[assoc];
	CatchFailureAsMessage @ If[
		ListOfListsQ[spec], 
		extract[#, assoc]& /@ spec,
		extract[spec, assoc]
	]
];

NetExtract::arg1 = "First argument `` should be a net, NetEncoder, or NetDecoder."

NetExtract[lhs_, _] /; Message[NetExtract::arg1, MsgForm @ lhs] := Null;

NetExtract /: Internal`ArgumentCountRegistry[NetExtract] = {1, 2};

Developer`Private`LHS_NetExtract /; Length[LHS] != 1 := 
	RuleCondition[Developer`CheckArgumentCount[Developer`Private`LHS, 1, 2]; Fail]

NetExtract[spec_][expr_] := NetExtract[expr, spec];


PackageExport["NetPart"]

NetPart[net_NetP, parts___] := Scope[
	$ambientNet = net;
	$AmbientSharedArrays = GetSharedArrays[net];
	extract[{parts}, net]
];

Clear[extract, extractOne, extractList];

extract[spec_, assoc_] := extractOne[spec, assoc];
extract[spec_List, assoc_] := Apply[extractList, spec] @ assoc;

extractOne[spec:(All | _List), assoc_] := FromInternalValue /@ getElement[assoc, spec];

extractOne[pos_Integer | pos_String, assoc_] := FromInternalValue @ getElement[assoc, pos];

extractOne[NetSharedArray[name_String], _] := Scope[
	res = $AmbientSharedArrays[name];
	If[MissingQ[res],
		nopart[NetSharedArray[name]],
		fromArray @ res
	]
];

extractOne[NetSharedArray[All], _] := 
	KeyMap[NetSharedArray, fromArray /@ $AmbientSharedArrays];

fromArray[na_NumericArray] := na
fromArray[other_] := Automatic;

NetExtract::invspec = "`` is not a valid specification for NetExtract."
extractOne[spec_, _] := ThrowFailure["invspec", spec];

$msgstack = {};

extractList[][data_] := FromInternalValue @ data;

extractList[ns_NetSharedArray][assoc_] := extractOne[ns, Null];

extractList[spec:(All | _List), rest___][data_] := 
	applyToElement[data, spec, Map[extractList[rest]]]

extractList[pos_, rest___][data_] := 
	applyToElement[data, pos, extractList[rest]];

applyToElement[data_, pos_, f_] := Block[
	{val = getElement[data, pos], $msgstack = Append[$msgstack, pos]},
	f @ val
]

(* TODO: make this a method, part of a container API *)
getElement[_, 0] := ThrowFailure["netnopart", 0];

getElement[coder:CoderP, spec_String] := coder[[spec]];

getElement[assoc_, spec_List] := Map[getElement[assoc, #]&, spec];	

getElement[assoc_Association, spec_] /; KeyExistsQ[assoc, "Nodes"] := 
	If[KeyExistsQ[assoc["Inputs"], spec] || KeyExistsQ[assoc["Outputs"], spec],
		getParam[assoc, spec],
		getNode[assoc[["Nodes"]], spec]
	];

getElement[assoc_Association, spec_] /; KeyExistsQ[assoc, "Type"] :=
	getParam[assoc, spec];

getElement[_, spec_] := nopart[spec];

(* bit complex: if it was a NetGraph, the nodes have been re-arranged to be in topological order,
and hence we can't say what the '5th' node IF the layers were provided as an assoc. If they were
provided as a list then we're fine, of course. See 327406.*)
getNode[nodes_, n_Integer] := If[
	1 <= Abs[n] <= Length[nodes],
		Part[
			nodes, 
			If[!DigitStringKeysQ[nodes], 
				n,
				If[Positive[n], 
					IntegerString @ n,
					IntegerString @ (n + 1 + Length[nodes])
				]
			]
		],
	nopart[n]
];

getNode[nodes_, s_String] := Lookup[nodes, s, nopart[s]]

getNode[nodes_, All] := dekey @ nodes;

getNode[_, s_] := nopart[s];

nopart[spec_] := System`Private`SetNoEntry @ Missing["NotPresent", Append[$msgstack, spec]];

(* undo the associationification that happens during construction for 
pure list versions of NetGraph and NetChain *)
dekey[assoc_Association] := 
	If[DigitStringKeysQ[assoc], Values @ assoc, assoc];

dekey[e_] := e;

General::netnopart = "Part `` does not exist.";

getParam[assoc_Association, specs_List] :=
	Map[getParam[assoc, #]&, specs];

getParam[assoc_Association, spec_] := Block[{},
	Do[	
		subassoc = assoc[key];
		If[AssociationQ[subassoc] && KeyExistsQ[subassoc, spec],
			val = subassoc[spec];
			If[key === "Arrays" && !MatchQ[val, _NumericArray | None | _NetSharedArray | _SymbolicRandomArray], 
				val = Automatic
			];
			Return[val, Block]
		],
		{key, Keys[assoc]}
	];
	Lookup[assoc, spec, nopart[spec]]
];

getParam[_, spec_] := nopart[spec];

PackageScope["FromInternalValue"]

FromInternalValue = MatchValues[
	ValidatedParameter[v_] := FromValidatedParameter[v]; 
	assoc_Association /; KeyExistsQ[assoc, "Type"] := extractLayer[assoc];
	list_List := If[PackedArrayQ[list], list, Map[FromInternalValue, list]];
	assoc_Association := Map[FromInternalValue, assoc];
	sym_Symbol := If[Context[sym] === "System`", sym, Indeterminate];
	c_NetEncoder | c_NetDecoder := despecializeCoder[c];
	ns_NetSharedArray := If[$ambientNet === None, ns, FromInternalValue @ $ambientNet["SharedArrays", First[ns]]];
	e_ ? ValidTypeQ := FromT[e];
	(* The rest is kept identical by default *)
	(*
	e_ ? AtomQ := e;
	e_Failure := e;
	na_NumericArray := na;
	m_Missing := m;
	f_Function := f;
	*)
	e_ := e
];

despecializeCoder[head_[name_, assoc_, type_]] := Scope[
	If[head === NetEncoder, 
		type = $EncoderData[name, "Output"], 
		type = $DecoderData[name, "Input"]];
	ReinferCoder[head, name, assoc, type]
];


PackageScope["FromValidatedParameter"]

FromValidatedParameter = MatchValues[
	sf_ScalarFunctionObject := ScalarFunctionToPureFunction[sf];
	ca_CharacterEncodingData := Normal[ca];
	ta_TokenEncodingData := Normal[ta];
	e_ := e
];

extractLayer[assoc_] := Scope[
	snames = DeleteDuplicates @ DeepCases[assoc, NetSharedArray[sname_] :> sname];
	If[snames =!= {}, 
		assoc["SharedArrays"] = KeyTake[$ambientNet["SharedArrays"], snames];
	];
	ConstructNet[assoc]
];


PackageExport["NetInputs"]

SetUsage @ "
NetInputs[net$] gives an association mapping input ports of net$ to their types."

NetInputs[net_NetP] := Map[FromT, net["Inputs"]];


PackageExport["NetOutputs"]

SetUsage @ "
NetOutputs[net$] gives an association mapping input ports of net$ to their types."

NetOutputs[net_NetP] := Map[FromT, net["Outputs"]];


PackageExport["NetExtractArrays"]

SetUsage @ "
NetExtractArrays[net$] gives an association mapping the positions of all initialized arrays \
in net$ to their values.
* Arrays are returned as NumericArray objects.
* The positions are in the same form as expected by NetExtract and NetReplacePart."

NetExtractArrays[net_NetP] := Scope[
	pos = Position[net, raw_NumericArray] /. Key -> Identity;
	AssociationThread[
		FromNetPath @ pos,
		Extract[net, pos]
	]
];
