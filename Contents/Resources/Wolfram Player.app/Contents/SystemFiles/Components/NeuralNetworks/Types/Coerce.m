Package["NeuralNetworks`"]



PackageScope["Coerce"]

SetUsage @ "
Coerce[data$, type$] coerces data$ to match type$, or panics if that is impossible."

Coerce[in_, type_] := ($type = None; coerce[in, type]);

Clear[coerce];

PackageScope["$smuggle"]
coerce[$smuggle[e_], t_] := e;

coerce[m_Missing, Defaulting[_, d_]] := d;

General::netreqparam = "Required parameter `1` was not specified.";
coerce[m_Missing, _] := cpanic["netreqparam", key];

coerce[in_, ct:CustomType[_, c_]] := Replace[c[in], $Failed :> tpanic[in, ct]];

coerce[in_, TypeT] := Scope[
	If[ValidTypeQ[t = ToT[in]], t, tpanic[in, TypeT]]
];

coerce[in_, t:IntervalScalarT[min_, max_]] := Scope @
	If[!MachineRealQ[n = N[in]] || n < min || n > max, tpanic[in, t], n];

coerce[in_, ScalarT] := 
	Replace[N[in], {t_ ? MachineRealQ :> t, _ :> tpanic[in, ScalarT]}];

coerce[r_RepeatedInteger, type_] := Replace[UnifyTypes[r, type], $Failed :> tpanic[r, type]]

$SimplePredicates = Association[
	RealT -> RealQ,
	BooleanT -> BooleanQ,
	StringT  -> StringQ, 
	IntegerT -> MachineIntegerQ,
	NaturalT -> NonNegativeMachineIntegerQ,
	PosIntegerT -> PositiveMachineIntegerQ,
	SizeT -> PositiveMachineIntegerQ,
	LengthVar -> PositiveMachineIntegerQ,
	TypeT -> ValidTypeQ,
	TypeExpressionT -> ValidTypeExpressionQ,
	FunctionT -> PossibleFunctionQ
];

coerce[in_, type:(Alternatives @@ Keys[$SimplePredicates])] :=
	If[TrueQ[$SimplePredicates[type] @ in],
		in, tpanic[in, type]
	];

(* to make InheritsFrom work, bit of a hack/special case *)
coerce[x_, Defaulting[_, x_]] := x;

(* currently coercing type, used for 'panic' *)
$type = None; 


PackageScope["BypassCoerce"]
coerce[BypassCoerce[e_], t_] := e;

coerce[net_, NetT[]] := ToLayer[net];

coerce[net_, type:NetT[ins_, outs_]] := Scope[
	ldata = ToLayer[net];
	$iport = True; MapAssocAssoc[checkIO, ins, Inputs[ldata], missIO["input"], extraIO["input"]];
	$iport = False; MapAssocAssoc[checkIO, outs, Outputs[ldata], missIO["output"], extraIO["output"]];
	ldata
];

General::netinvio = "`6` port \"`2`\" of net specified for `1` takes `3`, but a port `5` `4` is required."

checkIO[key_, reqt_, givent_] := 
	If[UnifyTypes[givent, reqt] === $Failed,
		cpanic["netinvio", key, TypeForm[givent], TypeForm[reqt], If[$iport, "taking", "producing"], If[$iport, "Input", "Output"]]
	];

General::netmissio = "Net specified for `` is missing required `` port ``."
General::netextraio = "Net specified for `` has unexpected `` port ``."

missIO[dir_][key_] := cpanic["netmissio", dir, key];
extraIO[dir_][key_] := cpanic["netextraio", dir, key];

General::netinvkey2 = "`2` is not a valid key for `1`. Valid keys include `3`.";
coerce[in_, StructT[rules_]] := Scope[
	If[!AssociationQ[in], 
		tpanic2[in, StringForm @ "an association"]];
	If[!SubsetQ[ruleKeys = Keys[rules], userKeys = Keys[in]], 
		ThrowFailure["netinvkey2", 
			$currParam,
			First @ Complement[userKeys, ruleKeys],
			QuotedStringRow[Select[ruleKeys, StringFreeQ["$"]], " and "]
		]];
	Association[
		#1 -> coerceStructParam[#1, Lookup[in, #1], #2]& @@@ rules
	]
];

coerceStructParam[key_, _Missing, Defaulting[_, v_]] := v;

General::reqkeymiss = "Required key `` is missing from ``."
coerceStructParam[key_, _Missing, _] := 
	ThrowFailure["reqkeymiss", QuotedStringForm @ key, $currParam];

coerceStructParam[key_, data_, type_] := 
	CoerceParam[
		StringForm["key \"``\" of ``", key, $currParam],
		data, type
	];

coerce[in_, type_] := Match[
	If[$type === None, $type = type, type]
	,
	RuleT[kt_, vt_] :>
		Match[in, 
			Rule[k_, v_] :> Rule[%[k, kt], %[v, vt]],
			tpanic[in]
		],

	(* REFACTOR: Take into account the type *)
	t_TensorT :> CoerceArray[in, TDimensions[t]],

	IndexIntegerT[max_Integer] :>
		If[TrueQ[PositiveMachineIntegerQ[in] && in <= max],
			in, tpanic[in]
		],
	IndexIntegerT[Infinity] :>
		If[PositiveMachineIntegerQ[in],
			in, tpanic[in]
		],
	IndexIntegerT[All] :>
		If[MachineIntegerQ[in],
			in, tpanic[in]
		],

	ImageT[size_, color_] :> Scope[
		If[!Image`ValidImageQ[in], tpanic[in]];
		img = in;
		If[MatchQ[size, {_Integer,_Integer}] && ImageDimensions[in] =!= size, 
			img = ImageResize[img, size]];
		If[StringQ[color] && ImageColorSpace[in] =!= color,
			img = ColorConvert[img, color]];
		img
	],

	Image3DT[size_, color_] :> Scope[
		If[!Image`ValidImage3DQ[in], tpanic[in]];
		img = in;
		If[MatchQ[size, {_Integer, _Integer, _Integer}] && ImageDimensions[in] =!= size, 
			img = ImageResize[img, size]];
		If[StringQ[color] && ImageColorSpace[in] =!= color,
			img = ColorConvert[img, color]];
		img
	],

(*  this shouldn't come up, and wouldn't TypeForm correctly on panic anyway.
	t_List :> (
		If[!ListQ[in] || Length[in] =!= Length[t], panic[in]];
		MapThread[coerce, {in, t}]
	),
*)
	AssocT[StringT, v_] :> 
		If[!AssociationQ[in] || !StringVectorQ[Keys[in]], tpanic[in],
			Association @ KeyValueMap[
				#1 -> CoerceParam[#1, #2, v]&, 
				in
			]
		],

	AssocT[k_, v_] :> 
		If[!AssociationQ[in], tpanic[in],
			Association @ KeyValueMap[
				%[#1, k] -> %[#2, v]&, 
				in
			]
		],

	EnumT[alts_] :> 
		If[!MatchQ[in, Alternatives @@ alts], tpanic[in], in],

	EitherT[ts_] :> Scope[
		Do[
			If[!FailureQ[res = CoerceSoft[in, t]], Return[res, Block]],
			{t, ts}
		];
		tpanic[in]
	],

	ListT[n_, t_] :> (
		If[!ListQ[in], tpanic[in]];
		If[IntegerQ[n] && n =!= Length[in], tpanic[in]];
		If[n === SizeT && in === {}, tpanic[in]];
		Catch @ Block[{panic := Throw[$Failed]},
			Map[coerce[#, t]&, in]
		] // OnFail[tpanic[in]]
	),

	Defaulting[t_, _] :> %[in, t],

	Nullable[t_] :> If[in === None, None, %[in, t]],

	MatchT[t_] :> If[!MatchQ[in, t], tpanic[in], in],

	TypeAliasP :> %[in, ResolveAlias @ type],

	ExpressionT :> in,

	DurationT :>
		If[
			And[
				MatchQ[in, Quantity[_ ? Positive, _]],
				MatchQ[UnitDimensions[in], {{"TimeUnit", 1}}]
			],
			in,
			tpanic[in]
		]
	,

	DistributionT :> Which[
		UnivariateDistributionQ[in], in, 
		NumericQ[in], If[in < $MachineEpsilon, 
			UniformDistribution[{0,0}], 
			NormalDistribution[0, N[in]]
		],
		True, tpanic[in]
	],

	tpanic[in]
];


PackageScope["CoerceArray"]

ClearAll[CoerceArray];
(* ^ needed for idempotency because definition order *)

CoerceArray[r_Integer|r_Real, dims:{__Integer}] := 
	CreateConstantNumericArray[dims, N[r]];

CoerceArray[arr_List, dims_] /; ArrayQ[arr, _, NumberQ] && dimsMatchQ[arr, dims] := Scope[
	res = Quiet @ toNumericArray[arr, "Real32"];
	If[!NumericArrayQ[res], cpanic["netinvtensorvals"]];
	res
];

General::netinvtensorvals = "The value specified for `` should be a real-valued tensor."

CoerceArray[arr_NumericArray ? NumericArrayQ, dims_] /; dimsMatchQ[arr, dims] := 
	If[NumericArrayType[arr] === "Real32", arr,
		toNumericArray[arr, "Real32"]
	];

dimsMatchQ[arr_, type_] := Match[type, 
	$Failed :> True,
	RankTP :> True,
	n_Integer :> arrayDepth[arr] === n,
	list_List :> And[
		arrayDepth[arr] === Length[list],
		MatchQ[arrayDimensions[arr], list /. SizeT -> _]
	]
];

CoerceArray[r_Integer|r_Real, dims_] := 
	SymbolicRandomArray[NNConstantDist[r], Replace[$Failed :> SizeListT[]] @ dims];

CoerceArray[dist:(_Symbol[___]) ? UnivariateDistributionQ, dims_] := 
	SymbolicRandomArray[dist, Replace[$Failed :> SizeListT[]] @ dims];

CoerceArray[HoldPattern @ Distributed[dims2_List, dist_ /; UnivariateDistributionQ[dist] || NumericQ[dist]], dims_] /; (dims === $Failed || MatchQ[dims2, dims /. SizeT -> _]) := 
	SymbolicRandomArray[If[UnivariateDistributionQ[dist], dist, NNConstantDist[N[dist]]], dims2];

General::netinvrank = "The value specified for `` should be a numeric array of rank ``.";
CoerceArray[_, n_Integer] := cpanic["netinvrank", n];

General::netinvdims = "The value specified for `` should be a numeric array of dimensions ``.";
CoerceArray[_, dims_List] := 
	If[MatchQ[dims, {SizeT..}],
		cpanic["netinvrank", Length[dims]],
		cpanic["netinvdims", dims]
	];

General::netinvtensor = "The value specified for `` should be a numeric array.";
CoerceArray[_, _] := cpanic["netinvtensor"];


PackageScope["CoerceSoft"]

SetUsage @ "
CoerceSoft[data$, type$] coerces data$ to match type$, or returns $Failed if that is impossible."

CoerceSoft[data_, type_] := Block[{panic := Throw[$Failed], $type}, Catch @ coerce[data, type]];

General::netinvparam = "Value given for `` should be ``, but was `` instead.";
General::netinvopt = "The value of `` should be ``.";

tpanic[in_] := tpanic2[in, $type];
tpanic[in_, t_] := tpanic2[in, If[$type === None, t, $type]];

(* for one or two custom places that don't use $type *)
$invmsgname = "netinvparam";
tpanic2[in_, type_] := 
	If[$isOpt, 
		panic["netinvopt", $currParam -> in, MsgForm[type]],
		panic["netinvparam", MsgForm[$currParam], MsgForm[type], MsgForm[in]]
	];

cpanic[msg_, args___] := panic[msg, $currParam, args];

panic[msg_, args___] := ThrowFailure[msg, args];


PackageScope["CoerceParam"]

SetUsage @ "
CoerceParam[name$, data$, type$] calls Coerce[data$, param$], but ensures error messages refer to the parameter as name$."

CoerceParam[_, Automatic, type_] /; FreeQ[type, Automatic] := 
	Replace[type, Defaulting[_, value_] :> value];

CoerceParam[name_, data_, type_] := Scope[
	$currParam = name; 
	Coerce[data, type]
];


PackageScope["CoerceOption"]

SetUsage @ "
CoerceOption[name$, data$, type$] calls Coerce[data$, param$], but ensures error messages refer to an invalid option called name$.
CoerceOption[name$, data$, type$ :> auto$] uses auto as the default if data$ is Automatic."

$isOpt = False;
CoerceOption[name_, data_, type_ :> auto_] :=
	CoerceOption[name, Replace[data, Automatic :> auto], type];

CoerceOption[name_, data_, type_] := Scope[
	$currParam = name; $isOpt = True;
	Coerce[data, type]
];


PackageScope["ToUnaryElementwiseFunction"]
PackageScope["ToNAryElementwiseFunction"]

General::invscf = "`` could not be symbolically evaluated as a `` scalar function."

ToUnaryElementwiseFunction[in_] := 
	Which[
		MemberQ[$PrimitiveUnaryElementwiseFunctions, in],
			in,
		KeyExistsQ[$namedFunctionsEquivalent, in],
			Lookup[$namedFunctionsEquivalent, in],
		True,
			OnFail[ThrowFailure["invscf", in, "unary"]] @ CompileScalarFunction[1, in]
	];

ToNAryElementwiseFunction[in_] :=
	Which[
		MemberQ[$PrimitiveBinaryElementwiseFunctions, in], in,
		MemberQ[$PrimitiveNAryElementwiseFunctions, in], in,
		True, OnFail[ThrowFailure["invscf", in, "binary"]] @ CompileScalarFunction[Automatic, in]
	];



PackageScope["CoerceUserSpec"]

SetUsage @ "
CoerceUserSpec[uspec$, type$, 'descrip$'] coerces user specification uspec$ against type$, and if a failure occurs, describes it using the term 'descrip$'.
* If type$ is a list of rules it will be interpreted as the spec for an association."

CoerceUserSpec[uspec_, list_List ? RuleVectorQ, descrip_] :=
	CoerceUserSpec[uspec, StructT[list], descrip];

CoerceUserSpec[uspec_, type_, descrip_] := 
	CoerceParam[StringForm @ descrip, uspec, type];

