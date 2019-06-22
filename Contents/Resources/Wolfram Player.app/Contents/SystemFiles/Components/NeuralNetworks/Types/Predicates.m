Package["NeuralNetworks`"]



PackageScope["$AtomicTypes"]

$AtomicTypes = {
	RealT, AtomT,
	PosIntegerT, IntegerT, NaturalT,
	SizeT,
	StringT,
	BooleanT,
	TypeT, ExpressionT, FunctionT, DurationT, TypeExpressionT
};


PackageScope["ValidTypeQ"]

SetUsage @ "
ValidTypeQ[type$] gives True if type$ is a well-formed type."

ValidTypeQ = MatchValues[
	MacroEvaluate[Alternatives @@ $AtomicTypes] := True;
	IndexIntegerT[SizeP|Infinity|All] := True;
	ListT[NatP, t_] := %[t];
	EitherT[ts_] := VectorQ[ts, %];
	Defaulting[t_, _] := %[t];
	Nullable[t_] := %[t];
	TensorT[SizeListP, t_] := %[t];
	EnumT[ListT[NatP, StringT]] := True;
	EnumT[_List] := True;
	ImageT[SizePairP, ColorSpaceT | MacroEvaluate[Alternatives @@ $ColorSpaces]] := True;
	Image3DT[SizeTripleP, ColorSpaceT | MacroEvaluate[Alternatives @@ $ColorSpaces]] := True;
	MacroEvaluate[$TypeAliasPatterns] := True;
	MatchT[_] := True;
	RuleT[k_, v_] := %[k] && %[v];
	StructT[{Repeated[_String -> (_?%)]}] := True;
	AssocT[k_, v_] := %[k] && %[v];
	_NetEncoder := True;
	_NetDecoder := True;
	SequenceT[_LengthVar, t_] := %[t];
	_ValidatedParameter := True;
	SymbolicRandomArray[_, {___Integer}] := True;
	_LengthVar := True;
	IntervalScalarT[_, _] := True;
	CustomType[_, _] := True;
	_ := False;
];


PackageScope["ValidTypeExpressionQ"]

SetUsage @ "
ValidTypeExpressionQ[type$] gives True if type$ is a well-formed type that can potentially include NetPath expressions."

NatPathP = NatP | NetPathP;
SizePathP = SizeP | NetPathP;
SizeListPathP = ListT[NatPathP, SizeT] | {___LengthVar, SizePathP...} | NetPathP;

ValidTypeExpressionQ = MatchValues[
	NetPathP := True;
	IndexIntegerT[SizePathP] := True;
	ListT[NatPathP, t_] := %[t];
	EitherT[ts_] := VectorQ[ts, %];
	Defaulting[t_, _] := %[t];
	TensorT[SizeListPathP | _ComputedType, t_] := %[t];
	ImageT[SizeListPathP, _String | ColorSpaceT | NetPathP] := True;
	Image3DT[SizeListPathP, _String | ColorSpaceT | NetPathP] := True;
	_ComputedType := True;
	_RawComputedType := True;
	NetT[] := True;
	NetT[_ ? validTypeAssocQ, _ ? validTypeAssocQ] := True;
	NormalizedT[t_, _] | NormalizedT[t_, _, _] := %[t];
	ValidatedParameterT[_] | ValidatedParameterT[_, _] := True;
	Nullable[t_] := %[t];
	TypeReplace[t_, rules_] := %[t] && VectorQ[Values[rules], weakVTEQ];
	IntervalScalarT[_, _] := True;
	{}|_Integer|{__Integer}|True|False := True; (* <- allow some constants to make it easy to develop layers incrementally *)
	e_ := ValidTypeQ[e];
];

weakVTEQ[e_] := ValidTypeExpressionQ[e] || ConcreteParameterQ[e];

validTypeAssocQ[assoc_Association] := And[
	StringVectorQ[Keys[assoc]], 
	VectorQ[Values[assoc], ValidTypeExpressionQ]
];


PackageScope["ConcreteParameterQ"]

SetUsage @ "
ConcreteParameterQ[param$] gives True if param$ is a value (e.g. a NumericArray) or a \
fully-specialized type (e.g. a Tensor with known rank and dimensions)."

ConcreteParameterQ := MatchValues[
	_String := True;
	True|False|None := True;
	t_List := VectorQ[t, %];
	t_NumericArray := NumericArrayQ[t];
	TensorT[{___LengthVar, ___Integer}, RealT|IndexIntegerT[_Integer|Infinity|All]|AtomT] := True;
	ImageT[{_Integer, _Integer}, _String] := True;
	Image3DT[{_Integer, _Integer, _Integer}, _String] := True;
	_Integer | _Real := True;
	_NetEncoder := True;
	_NetDecoder := True;
	IndexIntegerT[_Integer|Infinity|All] := True;
	Nullable[t_] := %[t];
	_ScalarFunctionObject := True;
	None := True;
	Infinity := True;
	LengthVar[_Integer] := True;
	_ValidatedParameter := True;
	assoc_Association := ConcreteNetQ[assoc];
	Min|Max|Mean|Total|Times|Catenate := True;
	EuclideanDistance|CosineDistance := True;
	_ := False;
];


PackageScope["ConcreteArrayQ"]
PackageScope["ConcreteOrSharedArrayQ"]
PackageScope["InitializedOrSharedArrayQ"]

SetUsage @ "ConcreteArrayQ[param$] gives True if param$ is NumericArray, None, or a TensorT with known dimensions."
SetUsage @ "ConcreteOrSharedArrayQ[param$] gives True if param$ is ConcreteArrayQ or a NetSharedArray."
SetUsage @ "InitializedOrSharedArrayQ[param$] gives True if param$ is NumericArray, None, or a NetSharedArray."

$ConcreteTensorP = TensorT[{___Integer}, RealT|IndexIntegerT[_Integer|Infinity|All]|AtomT];
$ConcreteTensorP = $ConcreteTensorP | Nullable[$ConcreteTensorP] | _SymbolicRandomArray | _SymbolicRandomArray;
ConcreteArrayQ[e_] :=            MatchQ[e, _NumericArray ? NumericArrayQ | None | $ConcreteTensorP];
ConcreteOrSharedArrayQ[e_] :=    MatchQ[e, _NumericArray ? NumericArrayQ | None | NetSharedArray[_String] | $ConcreteTensorP];
InitializedOrSharedArrayQ[e_] := MatchQ[e, _NumericArray ? NumericArrayQ | None | NetSharedArray[_String] | _DummyArray | _SymbolicRandomArray];

InitializedOrSharedOrConcreteArrayQ[e_] := InitializedOrSharedArrayQ[e] || ConcreteArrayQ[e];

(* TODO: Explain why there is a difference betwen these two. Maybe there isn't.
It's awkward. Probably reason to go with a separate "Types" key to help separate
runtime for spec-time parametricity. Then
"Type" -> FullySpecifiedTypeQ.
"Parameter" -> FullySpecifiedParmeterQ (no higher-level types like ListT etc)
"Array" -> FullySpecifiedArrayQ (TensorT, or actual NumericArray etc)
 *)

PackageExport["FullySpecifiedTypeQ"]

SetUsage @ "
FullySpecifiedTypeQ[param$] gives True if param$ is a value (e.g. a NumericArray) or a \
fully-specialized type (e.g. a Tensor with known rank and dimensions)."

FullySpecifiedTypeQ[t_ ? System`Private`ValidQ] := True;
FullySpecifiedTypeQ[t_] := If[fullq[t], System`Private`SetValid[t]; True, False];

fullq := MatchValues[
	TensorT[{___LengthVar, ___Integer}, tt_] := fullttq[tt];
	IntegerT|PosIntegerT|BooleanT := True;
	IndexIntegerT[_Integer|Infinity|All] := True;
	_NetEncoder | _NetDecoder := True;
	t_List := VectorQ[t, %];
	_ := False
];

fullttq[RealT|_IndexIntegerT|AtomT] := True;
fullttq[c:CoderP] := fullq[CoderType[c]];
fullttq[_] := False;


PackageScope["ContainsVarSequenceQ"]

ContainsVarSequenceQ[types_] := MemberQ[types, _ ? VarSequenceQ];


PackageScope["VarSequenceQ"]

VarSequenceQ[VarSequenceP[]] := True;
VarSequenceQ[c:CoderP] := SequenceCoderQ[c];
VarSequenceQ[_] := False;


PackageScope["TestType"]

SetUsage @ "
TestType[data$, type$] returns True if data$ matches type type$."

TestType[data_, type_] :=
	ToLiteralTypeTest[type] @ data;


PackageScope["ToBatchTypeTest"]

SetUsage @ "
ToBatchTypeTest[<|'name$1'->type$1,$$|>,batchsize$] makes a predicate that \
tests whether an input association matches elementwise with the given types.
* Each vector in the association must be a batch of the given length.
* A third argument of False indicates that there are allowed to be extra keys."

ToBatchTypeTest[types_, batchsize_, strict_:True] := With[
	{len = Length[types], patt = Block[{$batchsize = batchsize}, Normal @ Map[makeBatchTest, types]]},
	If[strict,
		Function[input, And[
			Length[input] === len,
			MatchQ[input, KeyValuePattern[patt]]
		]],
		MatchQ[KeyValuePattern[patt]]
	]
];

makeBatchTest[enc_NetEncoder] := With[
	{var = Unique["NeuralNetworks`Private`TempVars`$"], bsize = $batchsize}, 
	First @ {l_List /; Length[l] === bsize} /. l$ -> var];

makeBatchTest[other_] := With[{test = makeTest[SequenceT[$batchsize, other]]}, _List ? test];


PackageScope["ToLiteralTypeTest"]

SetUsage @ "
ToLiteralTypeTest[type$] constructs a predicate function for type$."

Clear[ToLiteralTypeTest];

ToLiteralTypeTest[type_] := Memoized[makeTest[type], Method -> "Inline"];


makeTest = MatchValues[

	ScalarT := RealValuedNumericQ;

	IntervalScalarT[a_, b_] := Function[RealValuedNumericQ[#] && a <= # <= b];

	TensorT[{first_Integer, lv_LengthVar, rest___}, type_] := With[
		{itest = %[TensorT[{lv, rest}, type]]}, 
		Function[z,
			ListQ[z] && Length[z] === first && VectorQ[z, itest]
		]
	];
	
	TensorT[dims_:{___LengthVar, RepeatedNull[_Integer | SizeT]}, IndexIntegerT[max_]] := With[
		{dpatt = dims /. {SizeT -> _, _LengthVar -> _}},
		Which[
			IntegerQ[max],
				Function[MatchQ[machineArrayDimensions[#], dpatt] && arrayMin[#] >= 1 && arrayMax[#] <= max],
			max === Infinity,
				Function[MatchQ[machineArrayDimensions[#], dpatt] && arrayMin[#] >= 1],
			True, (* unbounded *)
				Function[MatchQ[machineArrayDimensions[#], dpatt]]
		]
	];

	TensorT[dims_:{___LengthVar, RepeatedNull[_Integer | SizeT]}, AtomT|RealT] := Composition[
		MatchQ[dims /. {SizeT -> _, _LengthVar -> _}],
		machineArrayDimensions
	];
	
	TensorT[SizeListT[rank_Integer], AtomT|RealT] := Composition[
		EqualTo[rank],
		MachineArrayRank
	];

	TensorT[_, _] := MachineArrayQ;

	BooleanT := BooleanQ;

	StringT := StringQ;

	IntegerT := MachineIntegerQ;

	NaturalT := NonNegativeMachineIntegerQ;

	PosIntegerT | SizeT | _LengthVar := PositiveMachineIntegerQ;

	IndexIntegerT[max_] := Switch[max,
		_?IntegerQ, Function[in, PositiveMachineIntegerQ[in] && in <= max],
		Infinity, PositiveMachineIntegerQ,
		_, (*Unbounded*) MachineIntegerQ
	];

	TypeT := ValidTypeQ;

	ExpressionT := (True&);

	FunctionT := MatchQ[_Function];

	EitherT[{a_, b_}] := With[{ap = %[a], bp = %[b]}, ap[#] || bp[#]&];

	EitherT[ts_List] := OrOperator[% /@ ts];
	
	ListT[n_Integer, t_] := With[t2 = %[t], Length[#] === n && VectorQ[#, t2]&];

	ListT[_, t_] := With[t2 = %[t], VectorQ[#, t2]&];

	StructT[rules_List] := StructPredicate[MapAt[%, rules, {All, 2}]];
	
	RuleT[k_, v_] := With[{kt = %[k], vt = %[v]}, MatchQ[Rule[_ ? kt, _ ? vt]]];

	AssocT[k_, v_] := With[{kt = %[k], vt = %[v]}, 
		AssociationQ[#] && VectorQ[Keys[#], kt] && VectorQ[Values[#], vt]&
	];

	MatchT[patt_] := MatchQ[patt];

	ImageT[size_, color_] := ReplaceAll[
		Function @ And[
			Image`ValidImageQ[#],
			%[size] @ ImageDimensions[#],
			%[color] @ ImageColorSpace[#]
		],
		m_makeTest :> RuleCondition[m]
	];

	Image3DT[size_, color_] := ReplaceAll[
		Function @ And[
			Image`ValidImage3DQ[#],
			%[size] @ ImageDimensions[#],
			%[color] @ ImageColorSpace[#]
		],
		m_makeTest :> RuleCondition[m]
	];

	Defaulting[t_, _] := With[p = %[t],
		MissingQ[#] || p[#]&
	];

	Nullable[t_] := With[p = %[t],
		If[# === None, True, p[#]]&
	];

	EnumT[list_] := MatchQ[Alternatives @@ list];

	TypeAliasP := %[ResolveAlias[type]]
];



PackageScope["StructPredicate"]

SetUsage @ "
ToLiteralTypeTest[type$] constructs a predicate function for type$."

StructPredicate[rules_][assoc_] := 
	AssociationQ[assoc] && SubsetQ[Keys[rules], Keys[assoc]] && Catch[
		MapThread[
			If[!#1[#2], Throw[False]]&,
			{Values[rules], Lookup[assoc, Keys[rules]]}
		];
		True
	];



PackageScope["VectorTypeQ"]

VectorTypeQ[t_TensorT] := TRank[t] === 1;
VectorTypeQ[_] := False;


PackageScope["DynamicDimsQ"]

DynamicDimsQ[e_] := !FreeQ[e, _LengthVar];

