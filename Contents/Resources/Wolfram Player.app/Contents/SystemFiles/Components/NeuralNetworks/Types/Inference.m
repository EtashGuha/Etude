Package["NeuralNetworks`"]



DeclareMethod[ScanInferenceRules, LayerInferenceRules, ContainerInferenceRules, OperatorInferenceRules];

ContainerInferenceRules[assoc_] := (
	SowInferenceRules[assoc["Edges"]];
	ScanNodes[ScanInferenceRules, assoc];
);

LayerInferenceRules[assoc_] := Scope[
	UnpackAssociation[$LayerData[assoc["Type"]], inferenceRules, subNets, postInferenceFunction, runtimeInferenceRules];
	SowInferenceRules[inferenceRules];
	If[runtimeInferenceRules =!= None, SowInferenceRules @ Flatten @ List @ runtimeInferenceRules[assoc]];
	KeyValueScan[sowSharedArrayRule, assoc["Arrays"]]; (* <- only looking for NetSharedArray arrays *)
	SowPostInferenceFunction[postInferenceFunction];
];

OperatorInferenceRules[assoc_] := (
	LayerInferenceRules[assoc];
	ScanSubNets[ScanInferenceRules, assoc];
);

SowInferenceRules[e_List] := 
	BagPush[$irules, PrefixPorts[e]];

SowPostInferenceFunction[None] := Null;
SowPostInferenceFunction[f_] := 
	BagPush[$pfunctions, PrefixPorts[f]];

sowSharedArrayRule[key_, NetSharedArray[name_]] :=
	BagPush[$irules, Join[$path, NetPath["Arrays", key]] -> NetPath["SharedArrays", name]];

PackageScope["ReapInferenceRules"]

ReapInferenceRules[e_] := Scope[
	CollectTo[{$irules, $pfunctions}, ScanInferenceRules[e]];
	{$irules, $pfunctions}
];


PackageScope["ConstructWithInference"]

ConstructWithInference[head_, assoc_] := 
	ConstructWithInference[head, assoc, $StandardMetadata];

ConstructWithInference[head_Symbol, assoc_Association, meta_Association] := (
	CheckPortsForBadLengthVars[assoc];
	System`Private`ConstructNoEntry[head, InferNData[pinConnectedCellStates @ assoc], meta]
);

(* 	Workaround for https://bugs.wolfram.com/show?number=348910
	Because the optimized path cannot be done for LSTM that must provide an internal state
*)
pinConnectedCellStates[asso_] := If[asso["Type"] == "Graph",
	MapAt[Function[pinConnectedCellStates[#, asso["Edges"]]], asso, "Nodes"],
	asso
];
pinConnectedCellStates[nodes_, edges_] := Scope[
	istateful = Keys @ Select[nodes,
		KeyExistsQ[Lookup[#, "Parameters", <||>], "$CellStateConnectedQ"]&
	];
	If[Length[istateful] == 0, nodes,
		istateconnected = DeleteDuplicates @ Cases[edges,
			Rule[_, NetPath["Nodes", layer_, "States", "CellState"]] :> layer
		];
		(* mark these cell states as connected *)
		nodes = MapAt[True&, nodes,
			Map[Function[{#, "Parameters", "$CellStateConnectedQ"}], istateconnected]
		];
		(* mark these cell states as NOT connected *)
		nodes = MapAt[False&, nodes,
			Map[Function[{#, "Parameters", "$CellStateConnectedQ"}], Complement[istateful, istateconnected]]
		]
	]
];
PackageScope["InferNData"]

InferNData[assoc_Association] := Scope[
	$path = NetPath[];
	{irules, pfuncs} = ReapInferenceRules[assoc];
	DoInference[assoc, irules, pfuncs]
];


PackageScope["DoInference"]

DoInference[expr_, rules_, pfuncs_] := Scope[
	paths = DeepUniqueCases[rules, _NetPath];
	pathVals = UnsafeQuietCheck[
		Extract[expr, List @@@ paths],
		findInvalidPaths[expr, paths]
	];
	$psetdirty = $restart = False; 
	(* $vals is sometimes looked at for message generation: spot origins of problems *)
	$vals = AssociationThread[paths, pathVals];
	newvals = iDoInference[$vals, Flatten @ rules, paths];
	$expr = expr; 
	KeyValueMap[setPart, newvals]; 	
	Scan[executePostFunc, pfuncs];
	If[$restart, $expr = DoInference[$expr, rules, pfuncs]];
	$expr
];


PackageScope["RestartInference"]

(* only causes re-inference if a PostSet made a change *)
RestartInference[] := If[$psetdirty, $restart = True];


PackageScope["PCFDeclareInputs"]
PackageScope["PCFGetInputs"]
PackageScope["PCFGetOutputs"]

PCFGetInputs[] := $expr["Inputs"];
PCFGetOutputs[] := $expr["Outputs"];

PCFDeclareInputs[inputs_Association] := $expr["Inputs"] = inputs;
PCFDeclareInputs[inputs_List] := $expr["Inputs"] = Association @ IMap[IntegerString[#1] -> #2&, inputs];


PackageScope["PCFExpandMultiport"]

General::incnetinpcount = "Number of inputs specified in \"Inputs\" option (``) doesn't match number of inputs required by specified function (``).";
PCFExpandMultiport[count_] := Scope[
	ins = PCFGetInputs[]; 
	If[MatchQ[ins, <|$Multiport -> _|>],
		shape = First @ PCFGetOutputs[];
		PCFDeclareInputs @ Table[shape, count];	
	,
		If[count =!= Length[ins],
			ThrowFailure["incnetinpcount", Length[ins], count]
		]
	]
]



PackageScope["PostSet"]

SetUsage @ "
PostSet[lhs$, rhs$] can be used in a PostInferenceFunction to set lhs$ to rhs$."

PostSet[a_, b_] := Scope[
	$preserveCoders = True;
	setPath = psetPath;
	{ea, eb} = {a, b} /. p_NetPath :> ($expr @@ p); 
	Catch[
		u = unify[ea, eb];
		If[ea =!= u, set[a, u]];
		If[eb =!= u && !FreeQ[b, NetPath], set[b, u]]
	,
		unify, catchUnifyFailure
	];
];


psetPath[NetPath[p___], value_] := If[$expr[p] =!= value, $psetdirty = True; $expr[p] = value];

executePostFunc[f_] := 
	Apply[f /. {
		p_PostSet :> p,
		p_NetPath :> RuleCondition[$expr @@ p]
	}, {}];

findInvalidPaths[expr_, paths_] :=
	Do[
		If[MissingQ[slice[expr]], 
			Panic["InvalidPort", "`` is not present in expression.", NetPath @@ slice]],
		{slice, Slice @@@ paths}
	];

(*setPart[NetPath[p___], val_] := If[!ConcreteParameterQ[$expr[p]], $expr[p] = val];*)
(* the above is too strong in the case of LengthVar[id], but we do need to avoid
replacing numeric arrays that we turned into types at inference time. *)

setPart[NetPath[p___], val_] := ComposeTo[$expr[p], replaceWith[val]];

replaceWith[new_][c:CoderP|SequenceT[_, CoderP]] := UnifyCoderWith[c, new];
replaceWith[new_][old_NetSharedArray] := old;
replaceWith[new_][old_NumericArray] := old;
replaceWith[new_][old:SymbolicRandomArray[dist_, dims_]] := Match[new, 
	TensorT[newdims_List, RealT] :> SymbolicRandomArray[dist, newdims],
	old];
replaceWith[new_][old_] := new;

makeIndex[list_, keys_] := Scope[
	rules = Dispatch[MapIndexed[# -> (Place @@ #2)&, keys]];
	invert = MapIndexed[Cases[#, Place[n_] :> Rule[n, First[#2]], {0, Infinity}]&, list /. rules];
	KeyMap[
		Part[keys, #]&, 
		Association @ Merge[Flatten @ invert, DeleteDuplicates]
	]
];

PackageExport["$MaxInferenceSteps"]

$MaxInferenceSteps = 40000;

$preInferenceRules = Dispatch[{
	vp_ValidatedParameter :> vp, (* avoid changing anything inside a VP *)
	r_NumericArray :> RuleCondition @ TensorT[Dimensions[r]], 
	NetSharedArray[name_] :> RuleCondition @ $types[NetPath["SharedArrays", name]],
	SymbolicRandomArray[_, dims_] :> TensorT[dims, RealT]
}];

iDoInference[types_, rules_, keys_] := Scope[
	
	$LastTypeInferenceStack ^= {};
	$LastFailedTypeInferenceData ^= None;
	$LastFailedTypeInferenceLHS ^= None;
	$LastFailedTypeInferenceRHS ^= None;

	(* shared arrays and numeric arrays are replaced with their TensorT equivalents before
	inference. separate checks when updating the original expression ensure the NumericArrays and SharedArrays
	don't get overwritten *)
	$types = types /. $preInferenceRules;
	$rules = List @@@ rules;

	$ruleindex = makeIndex[$rules, keys];
	n = Length[$rules];
	$dirty = CTable[1, n];
	range = Range[n];

	maxCount = Min[32 * Length[keys], $MaxInferenceSteps];

	count = 0;
	While[
		Total[$dirty] > 0,
		Do[
			pair = {a, b} = Part[$rules, i] /. TypeReplace[s_, rs_] :> Replace[strongeval[s], rs];
			{ea, eb} = eval @ pair; 
			u = Null;
			Catch[
				u = unify[ea, eb];
				If[ea =!= u, set[a, u]];
				If[eb =!= u && !FreeQ[b, NetPath], set[b, u]]
			,
				unify, catchUnifyFailure
			];
			Part[$dirty, i] = 0;
			If[count++ >= maxCount, 
				AppendTo[$LastTypeInferenceStack, {ea, eb} -> u];
				If[Length[$LastTypeInferenceStack] > 3, ThrowFailure["netinfexc"]];
			];
			,
			{i, Pick[range, $dirty, 1]}
		];
	];

	$types /. {
		(* condense expanded-out types that don't actually contain any info *)
		TensorT[dims:{SizeT..}] :> RuleCondition @ TensorT[SizeListT[Length[dims]]],
		ListT[0, _] :> {}
	}
];

PackageScope["$LastTypeInferenceStack"]
PackageScope["$LastFailedTypeInferenceData"]
PackageScope["$LastFailedTypeInferenceLHS"]
PackageScope["$LastFailedTypeInferenceRHS"]
PackageScope["$LastFailedTypeInferenceState"]

$LastTypeInferenceStack = {};
$LastFailedTypeInferenceState = None;
$LastFailedTypeInferenceLHS = None;
$LastFailedTypeInferenceRHS = None;

General::netinfexc = "The given net appears to be too large to correctly infer all types. Consider increasing NeuralNetworks`.`$MaxInferenceSteps from its default value."

General::tyfaildebug = "Type unification or setting failure:\n`` = ``\n`` = ``\nUnified = ``\nSet = ``";

catchUnifyFailure[_, _] := If[$DebugMode,
	$LastFailedTypeInferenceData = {a, ea, b, eb, u};
	$LastFailedTypeInferenceLHS = ea; 
	$LastFailedTypeInferenceRHS = eb;
	$LastFailedTypeInferenceState = $types;
	ThrowFailure["tyfaildebug", a, ea, b, eb, u, $setFailure],
	If[!FreeQ[{ea, eb}, _EitherT], 
		reportAmbigType[],
		reportFailure[a, b, ea, eb]
	];
];

General::tyambig2 = "Could not resolve an ambigious type within the net. The origin of the problem appears to be ``, which is ``. Please specify a type for this part manually, and try again.";
General::tyambig1 = "Could not resolve an ambigious type within the net. Try specify the input or output shapes of layers with flexible input types, such as LinearLayer, EmbeddingLayer, BatchNormalizationLayer, and DotLayer."

reportAmbigType[] := Scope[
	pos = Keys @ Select[$vals, !FreeQ[#, _EitherT]&];
	pos = SortBy[pos, -Count[#, "Input"]&];
	pos = First[pos, None];
	If[pos === None,
		ThrowFailure["tyambig1"],
		ThrowFailure["tyambig2", MsgForm[pos], MsgForm[$vals @ pos]]
	];
];

General::tyinc = "`` of `` (``) is inconsistent with `` of `` (``)."
reportFailure[p1_NetPath, p2_NetPath, t1_, t2_] := Scope[
	{str1, str2, kind} = toTypeMismatchData[t1, t2];
	ThrowFailure["tyinc", ToTitleCase @ kind, NetPathString[p1], str1, kind, NetPathString[p2], str2]
];

General::tyfail1 = "Inferred inconsistent value for ``.";
General::tyfail2 = "Inferred inconsistent ``s for `` (`` versus ``).";

reportFailure[p1_NetPath, p2_, t1_, t2_] := Scope[
	imiss = findInternalMismatch[p1, p2];
	If[!FailureQ[imiss] && useInternalMissmatchQ[Last[imiss]],
		ThrowFailure["tyfail1", NetPathString[imiss]];
	,
		{str1, str2, kind} = toTypeMismatchData[t1, t2];
		ThrowFailure["tyfail2", kind, NetPathString[p1], str1, str2]
	]
];


PackageScope["toTypeMismatchData"]

(* factored out so that multiple code paths can use this *)
toTypeMismatchData[a_, b_] := toTypeMismatchData2[StripCoders @ a, StripCoders @ b];

toTypeMismatchData2[t1_TensorT, t2_TensorT] := Scope[
	tt1 = TType[t1]; tt2 = TType[t2];
	If[!FailureQ[tt1] && !FailureQ[tt2] && FailureQ[UnifyTypes[tt1, tt2]],
		Return @ {TypeString[tt1], TypeString[tt2], "type"};
	];
	{t1, t2} = {t1, t2} /. _IndexIntegerT|AtomT -> RealT;
	(* ^ we've established that types aren't a problem, so ensure they don't appear in
	TypeString *)
	tr1 = TRank[t1]; tr2 = TRank[t2];
	If[IntegerQ[tr1] && IntegerQ[tr2] && tr1 =!= tr2, 
		Return @ {"a " <> tensorName[tr1], "a " <> tensorName[tr2], "rank"}];
	{TypeString[t1], TypeString[t2], "dimension"}
];

toTypeMismatchData2[t1_, t2_] := {TypeString[t1], TypeString[t2], "value"};

(* heuristics *)
useInternalMissmatchQ[str_] := StringStartsQ[str, "$"] || StringContainsQ[str, "Channels"];

reportFailure[p1_List, p2_, t1_List, t2_List] :=
	RuleCondition[
		ScanThread[
			If[!UnifiableQ[#2, #3], reportFailure[#1, p2, #2, #3]]&,
			{p1, t1, t2}
		];
		Fail
	];
(* ^ this case happens with the new shapefunctions and rankfunctions, which create 
{ports...} = {vals...} inference rules. Failures there need to be located among the tuple elements.
Eventually we will completely supercede this type unification based system with a big table of vars
that get explicitly created and then solved. and all this sheningans will go away.
*)

reportFailure[p1_, p2_, t1_, t2_] := Scope[
	imiss = findInternalMismatch[p1, p2];
	If[!FailureQ[imiss],
		ThrowFailure["tyfail1", NetPathString[imiss]]
	,
		General::tyufail = "Net contains inconsistent shapes.";
		ThrowFailure["tyufail"];
	]
];


(* this attempts to find an internal parameter that is responsible for the 
failure to unify. It does this by stepping through unify using a override flag,
and looking for a NetPath that doesn't unify.
*)

findInternalMismatch[a_, b_] := Block[{
	setPath = testSetPath,
	spanic := Throw[$Failed]},
	Catch[
		set[b, strongeval @ a];
		set[a, strongeval @ b];
		$Failed
	]
];

testSetPath[path_, right_] := Scope[
	left = strongeval @ path;
	If[!UnifiableQ[left, right],
		Throw[path];
	];
	Null
];


General::tyufail = "Net contains inconsistent shapes.";
reportFailure[___] := ThrowFailure["tyufail"];

fmtSlot[name_, s_, pos_, col_] := 
	Interpretation[
		Tooltip[
			Style[StringRiffle[pos, ":"], Bold, Darker[col, .2], ShowStringCharacters -> False],
			RawBoxes @ Cell[BoxData @ RowBox[{"NetPath", "[", 
				Sequence @@ Riffle[
					ToBoxes[Style[#, ShowStringCharacters -> True]]& /@ pos, 
					","], 
				"]"
			}], "Input"]
		],
		s
	];


PackageScope["RunTestInference"]

RunTestInference[types_, rules_] := CatchFailure @ Scope[
	types = KeyMap[NetPath, types];
	rules = MapAt[NetPath, rules, {All, 1}];
	result = iDoInference[types, rules, Keys[types]];
 	KeyMap[First, result]
];


PackageScope["SubTypeQ"]

SetUsage @ "
SubTypeQ[a$, b$] returns True if a$ is a subtype of b$."

SubTypeQ[a_, b_] := And[
	Not @ FailureQ @ UnifyTypes[a, b],
	Not @ FailureQ @ Catch[set[b, a], unify]
];

Clear[set];

set[p_NetPath, value_] := setPath[p, value];

setPath[p_, value_] := 
	If[$types[p] =!= value,
		Part[$dirty, $ruleindex[p]] = 1;
		$types[p] = value;
	];

setPath[p_, StartsWithT[b_List]] := Scope[
	v = $types[p];
	If[!ListQ[v], Return[Null]];
	If[Length[v] < Length[b], spanic[v, Append[b, "\[Ellipsis]"]]];
	span = 1 ;; Length[b];
	If[v[[span]] =!= b,
		set[v[[span]], b];
		$types[[Key[p], span]] = b;
		Part[$dirty, $ruleindex[p]] = 1;
	];
];

set[t1_TensorT, t2_TensorT] := setTensor[t1, t2];

set[ListT[n1_, t1_], ListT[n2_, t2_]] := (set[n1, n2]; set[t1, t2];)
set[ListT[n_, t_], e_List] := (set[n, Length[e]]; Scan[set[t, #]&, e];)
set[_, RepeatedInteger[n_]] := Null;

set[e_List, ListT[n_, t_]] := (set[Length[e], n]; Scan[set[#, t]&, e];)
set[a_List, b_List] /; Length[a] == Length[b] := MapThread[set, {a, b}];

set[a_Association, b_Association] /; Length[a] == Length[b] := (
	If[Keys[a] =!= Keys[b], spanic[a, b]];
	KeyValueScan[set[#2, b[#1]]&, a]
)

set[EnumT[t1_], EnumT[t2_]] := set[t1, t2];

set[ImageT[sz_, c_], HoldPattern @ img_Image] := 
	(set[sz, ImageDimensions[img]]; set[c, ImageColorSpace[img]];);

set[ImageT[sz1_, c1_], ImageT[sz2_, c2_]] :=
	(set[sz1, sz2]; set[c1,c2];);

set[Image3DT[sz_, c_], HoldPattern @ img_Image3D] := 
	(set[sz, ImageDimensions[img]]; set[c, ImageColorSpace[img]];);

set[Image3DT[sz1_, c1_], Image3DT[sz2_, c2_]] :=
	(set[sz1, sz2]; set[c1, c2];);

set[PosIntegerT, PosIntP] := Null;
set[NaturalT, NatP] := Null;
set[SizeT, PosIntP] := Null;
set[SizeT, SizeT] := Null;

set[SizeT, _LengthVar] := Null;

set[a_Integer, b_Integer] := If[a =!= b, spanic[a, b]];

set[_ComputedType, _] := Null;
set[_RawComputedType, _] := Null;

set[ExpressionT, _] := Null;

set[Nullable[_], None] := Null;

set[Nullable[t_], d_] := set[t, d];

set[EitherT[list_List], d_] := Block[
	{spanic := Throw[$Failed]},
	Do[ (* gah, this technically has to be transactional if types are ambigious. f that. *)
		If[!FailureQ[Catch @ set[t, d]], Break[];],
		{t, list}
	]
];

set[Defaulting[t_, _], d_] := set[t, d];

set[IndexIntegerT[n_Integer], x_Integer] := Which[x < 1, spanic[1, x], x > n, spanic[n, x]];
(* taliesb: does this make sense from a contravariance point of view? TODO: think about this very carefully
            the inference system preserve the direction of type propogation in a way that makes the asymmetry of set meaningful?
   jeromel: the propagation is bidirectional, so it seems that there is no asymmetry possible.
*)
set[IndexIntegerT[Infinity], IndexIntegerT[_]] := Null;
set[IndexIntegerT[All], IndexIntegerT[_]] := Null;

set[IndexIntegerT[a_], IndexIntegerT[b_]] := set[a, b];

set[AtomT, _] := Null;

(* comes up in using LogPerplexityLossLayer for seq2seq *)
set[i1_IndexIntegerT, TensorT[{}, i2_IndexIntegerT]] := set[i1, i2];

set[LengthVar[_], _] := Null

set[t_TensorT, coder:CoderP] :=
	set[t, CoderType[coder]];

set[_, _EitherT] := Null

set[a_, b_] := If[a =!= b, spanic[a, b]];

spanic[] := Throw[$Failed, unify];
spanic[a_, b_] := ($setFailure = {a, b}; spanic[]);

eval[e_] := e /. {
	c_ComputedType :> evalCT[strongeval @ c],
	r_RawComputedType :> evalRCT[strongeval @ r],
	p_NetPath :> Lookup[$types, p, Panic["MissingNetPath", "Could not find path `` in types ``.", p, $types]]
};

strongeval[e_] := e //. p_NetPath :> RuleCondition[$types[p]];

evalRCT[RawComputedType[expr_, trigger_]] := 
	If[TrueQ[trigger], 
		expr,
		(*
		Continue[] (* <- just move onto the next rule in iInference Do loop *)
		*)
		ExpressionT
	];

evalRCT[_] := Panic[];

evalCT[ComputedType[type_, expr_, deps_, trigger_:False]] := Scope[
	If[!VectorQ[deps, ConcreteParameterQ] && !TrueQ[trigger],
		Return[type]];
	Check[
		res = expr;
		If[!FailureQ[res] && UnifiableQ[res, type], res, type]
	, 
		type
	]
]

Clear[setTensor, setRank, length, dropRank, takeRank];

setTensor[TensorT[d_, t_], ScalarT] := (
	set[t, RealT]; 
	If[length[d] === 1, set[d, {1}], set[d, {}]];
);

setTensor[TensorT[d1_, t_], TensorT[d2_, t_]] := set[d1, d2];

setTensor[TensorT[d1_, t1:TTypeP], TensorT[d2_, t2:TTypeP]] :=
	(set[d1, d2]; set[t1, t2]);

setTensor[TensorT[d1_, t1_], TensorT[d2_, t2_]] := Scope[
	r1 = length[d1]; r2 = length[d2];
	Which[
		!IntegerQ[r1] && IntegerQ[r2], setTensor2[TensorT[d1, t1], TensorT[d2, t2]],
		!IntegerQ[r2], Null,
		r1 == r2, set[d1, d2]; setRank[t1, t2],
		r1 > 0 && r1 < r2, set[d1, takeRank[d2, r1]]; setRank[t1, TensorT[dropRank[d2, r1], t2]],
		r2 > 0 && r1 > r2, set[d2, takeRank[d1, r2]]; setRank[t2, TensorT[dropRank[d1, r2], t1]],
		r1 == 0, setTensor[t1, TensorT[d2, t2]], 
		r2 == 0, setTensor[TensorT[d1, t1], t2]
	];
]

(* ^ last two lines above are for case:
TensorT[NetPath["Parameters", "NumHeads"], TensorT[{NetPath["Parameters", "$OutputSize"]}, RealT]] = TensorT[{3}, RealT] 
where NumHeads is an empty list
*)

length[list_List] := Length[list];
length[ListT[n_Integer, _]] := n;
length[ListT[p_NetPath, _]] := intOrNull @ Lookup[$types, p];
length[p_NetPath] := length @ Lookup[$types, p];
length[_] := Null

intOrNull[i_Integer] := i;
intOrNull[_] := Null;

setRank[p_NetPath, t_TensorT] := set[p, t];
(* avoid type vars getting set to e.g. RealT, which should never be naked *)
setRank[p_NetPath, t_] := set[p, TensorT[{}, t]]; 
setRank[t1_TensorT, t2_TensorT] := setTensor[t1, t2];
setRank[t1_TensorT, t2_] := setTensor[t1, TensorT[{}, t2]];
setRank[t1_, t2_TensorT] := setTensor[TensorT[{}, t1], t2];
setRank[t1_, t2_] := set[t1, t2];

dropRank[list_List, n_] := Drop[list, n];
dropRank[ListT[n_Integer, z_], m_] := ListT[n - m, z];
dropRank[p_NetPath, n_] := dropRank[Lookup[$types, p], n];
dropRank[_, _] := Null

takeRank[list_List, n_] := Take[list, n];
takeRank[ListT[n_Integer, z_], m_] := ListT[n - m, z];
takeRank[p_NetPath, n_] := takeRank[Lookup[$types, p], n];
takeRank[_, _] := Null


setTensor[a_, b_] := spanic[a, b];

(* more complex case:
we hope that the inner array is fixed rank, and the right hand size
is fixed rank, so the outer tensor's rank can be chosen.
ex: 
RunTestInference[
 <|"A" -> TensorT[{5, 3}], "B" -> TensorT[{3}], "C" -> SizeListT[]|>,
 {"A" -> TensorT[NetPath["C"], NetPath["B"]]}
 ]
*)
setTensor2[TensorT[n_NetPath, t1_], t2_TensorT] := Scope[
	If[MatchQ[t1, TTypeP], t1 = TensorT[{}, t1]];
	r1 = TRank[strongeval @ t1];
	r2 = TRank[t2];
	If[IntegerQ[r1] && IntegerQ[r2],
		dims = TDimensions[t2];
		set[n, Take[dims, r2 - r1]];
		set[t1, TensorT[Drop[dims, r2 - r1], TType[t2]]];
	]
];

setTensor2[TensorT[SizeListT[], t1_TensorT], t2_TensorT] := Scope[
	r1 = TRank[strongeval @ t1];
	r2 = TRank[t2];
	If[IntegerQ[r1] && r1 === r2, set[t1, t2]]
];



PackageScope["TypeDependenceGraph"]

(* TODO: make this easier to use directly, e.g. takes a net directly *)

TypeDependenceGraph[rules_, index_] := Scope[
	edges = Flatten @ KeyValueMap[toRule[#1, rules[[#2]]]&, index];
	edges = DeleteDuplicatesBy[edges, Sort[First[#]]&];
	Graph[edges, VertexLabels -> Placed["Name", Tooltip]]
];

SetAttributes[toRule, Listable];
toRule[port_, rule_Rule] := Map[
	Tooltip[port <-> #, rule]&, 
	DeleteCases[DeepUniqueCases[rule, _NetPath], port]
]


PackageScope["FailValidation"]

General::valfail = "Validation failed for ``: ``";

FailValidation[layer_, reason_] := ThrowFailure["valfail", layer, fromStringForm @ reason];
FailValidation[layer_, reason_String, args__] := FailValidation[layer, StringForm[reason, args]];
_FailValidation := $Unreachable;


PackageScope["FailConstruction"]

General::cstfail = "Invalid argument for ``: ``";

FailConstruction[layer_, reason_] := ThrowFailure["cstfail", layer, fromStringForm @ reason];
FailConstruction[layer_, reason_String, args__] := FailConstruction[layer, StringForm[reason, args]];
_FailConstruction := $Unreachable;


PackageScope["FailValidationWithPath"]

FailValidationWithPath[layer_, NetPath[], rest___] := FailValidation[layer, rest];

FailValidationWithPath[layer_, path_, rest___] := FailValidation[
	StringForm["`` of net (``)", NetPathForm[$path], layer], 
	rest
]
