Package["NeuralNetworks`"]



$SSA = <||>;
$SSACache = <||>
$NextID = 1;

PackageScope["BooleanSymbol"]
PackageScope["ScalarSymbol"]
PackageScope["NegatedSymbol"]
PackageScope["InvertedSymbol"]
PackageScope["$namedFunctionsEquivalent"]

DerivedScalar[body_] := 
	CacheTo[$SSACache, Hash[body],
		With[{new = ScalarSymbol[$NextID++]}, 
			$SSA[new] = body; 
			new
		]
	];

$namedFunctions = {
	"RectifiedLinearUnit",
	"ScaledExponentialLinearUnit",
	"ExponentialLinearUnit",
	"SoftSign",
	"SoftPlus",
	"HardTanh",
	"HardSigmoid",
	"Sigmoid"
};

$nameAliases = {
	"ReLU" -> "RectifiedLinearUnit", 
	"SELU" -> "ScaledExponentialLinearUnit",
	"ELU" -> "ExponentialLinearUnit",
	"SoftReLU" -> "SoftPlus"
};

$namedFunctionsEquivalent = {
	"RectifiedLinearUnit" -> Ramp,
	"Sigmoid" -> LogisticSigmoid
};
$namedFunctionsEquivalent = Join[
	$namedFunctionsEquivalent,
	Normal @ DeleteMissing @ Map[Lookup[$namedFunctionsEquivalent, #]&, Association[$nameAliases]]
];

PackageScope["ScalarFunctionApply"]

ScalarFunctionApply[func_][inputs_] := ScalarFunctionApply[func, inputs];
ScalarFunctionApply[func_ScalarFunctionObject, inputs_] /; VectorQ[inputs, NumericQ] :=
	Normal[
		MXLayer[<|"Input" -> "n"|>, <|"Output" -> "n"|>,
			Function @ SetOutput["Output", SowScalarFunction[func, GetInput["Input"]]]
		][inputs]
	];
ScalarFunctionApply[func_, inputs_] /; VectorQ[inputs, NumericQ] :=
	N[Lookup[$namedFunctionsEquivalent, func, func][inputs]];
(* Note:
	we can generalize to be able to call ScalarFunctionApply
	with other types of tensors than just list of values.
...
ScalarFunctionApply[func_][input_] /; NumericQ[input] := First[ScalarFunctionApply[func][{input}]]
...
	Not done because not needed yet (YAGNI).
*)

ClearAll[ScalarSymbol, NegatedSymbol, InvertedSymbol];

ScalarSymbol /: str_String[arg_ScalarSymbol] := Scope[
	name = str /. $nameAliases;
	If[!MemberQ[$namedFunctions, name], ReturnFailed[]];
	DerivedScalar[{name, arg}]
];

ScalarSymbol /: Times[-1, a_ScalarSymbol] := NegatedSymbol[a];
ScalarSymbol /: Power[a_ScalarSymbol, -1] := InvertedSymbol[a];

declareUnary[f_] := (
	ScalarSymbol /: f[a_ScalarSymbol] := DerivedScalar[{f, a}];
	NegatedSymbol /: f[n_NegatedSymbol] := f[norm[n]];
	InvertedSymbol /: f[n_InvertedSymbol] := f[norm[n]];
);

norm[NegatedSymbol[f_]] := DerivedScalar[{Minus, f}];
norm[InvertedSymbol[f_]] := DerivedScalar[{Divide, 1.0, f}];
norm[e_] := e;

NegatedSymbol /: Plus[r_ScalarSymbol, NegatedSymbol[a_]] := DerivedScalar[{Subtract, r, a}];
InvertedSymbol /: Times[r_ScalarSymbol, InvertedSymbol[a_]] := DerivedScalar[{Divide, r, a}];

NegatedSymbol /: Plus[r_ ? NumericQ, NegatedSymbol[a_]] := DerivedScalar[{Subtract, N[r], a}];
InvertedSymbol /: Times[r_ ? NumericQ, InvertedSymbol[a_]] := DerivedScalar[{Divide, N[r], a}];

(* the above rules need to go *before* the scan of declareBinary, for some reason *)

Scan[declareUnary, $PrimitiveUnaryElementwiseFunctions];

ScalarSymbol /: Clip[x_ScalarSymbol] := DerivedScalar[{Clip, x, {-1., 1.}}];
ScalarSymbol /: Clip[x_ScalarSymbol, {a_ ? NumericQ, b_ ? NumericQ}] := DerivedScalar[{Clip, x, {N[a], N[b]}}];

declareBinary[f_] := (
	ScalarSymbol /: f[a_ScalarSymbol, b_ScalarSymbol] := DerivedScalar[{f, a, b}];
	ScalarSymbol /: f[a_ScalarSymbol, b_ ? NumericQ] := DerivedScalar[{f, a, N[b]}];
	ScalarSymbol /: f[a_? NumericQ, b_ScalarSymbol] := DerivedScalar[{f, N[a], b}];
	NegatedSymbol /: f[lhs___, n_NegatedSymbol, rhs___] := f[lhs, norm[n], rhs];
	InvertedSymbol /: f[lhs___, n_InvertedSymbol, rhs___] := f[lhs, norm[n], rhs];
);

Scan[declareBinary, $PrimitiveBinaryElementwiseFunctions];

numericOrScalarQ[_ScalarSymbol] := True;
numericOrScalarQ[e_] := NumericQ[e];

$comparisonSymbols = Greater|GreaterEqual|Less|LessEqual|Equal|Unequal;

toN[s_ScalarSymbol] := s;
toN[e_ ? NumericQ] := N[e];

ScalarSymbol /: (cmp:$comparisonSymbols)[a_ScalarSymbol, b_, rest__] := And[cmp[a, b], cmp[b, rest]];
ScalarSymbol /: (cmp:$comparisonSymbols)[a_, b_ScalarSymbol, rest__] := And[cmp[a, b], cmp[b, rest]];

ScalarSymbol /: (cmp:$comparisonSymbols)[a_ScalarSymbol, b_ ? numericOrScalarQ] := 
	BooleanSymbol @ DerivedScalar[{cmp, a, toN @ b}];

ScalarSymbol /: (cmp:$comparisonSymbols)[a_ ? numericOrScalarQ, b_ScalarSymbol] := 
	BooleanSymbol @ DerivedScalar[{cmp, toN @ a, b}];

ScalarSymbol /: Inequality[lhs_ScalarSymbol, cmp:$comparisonSymbols, rhs_, rest__] :=
	And[cmp[lhs, rhs], Inequality[rhs, rest]];

ScalarSymbol /: Inequality[lhs_ScalarSymbol, cmp:$comparisonSymbols, rhs_] :=
	cmp[lhs, rhs];

BooleanSymbol /: (op:And|Or)[bools___BooleanSymbol] := 
	BooleanSymbol @ DerivedScalar[Prepend[First /@ {bools}, op]];

BooleanSymbol /: Not[bool_BooleanSymbol] := 
	BooleanSymbol @ DerivedScalar[{Not, First @ bool}];

BooleanSymbol /: Boole[BooleanSymbol[s_]] :=
	DerivedScalar[{Boole, s}];

toSource[s_ScalarSymbol] := s;
toSource[n_ ? NumericQ] := DerivedScalar[{Identity, N[n]}];
toSource[_] := $Failed;

BooleanSymbol /: If[BooleanSymbol[cond_], lhs_, rhs_] := Scope[
	lval = toSource[lhs];
	rval = toSource[rhs];
	DerivedScalar[{If, cond, lval, rval}]
]

fromBoole[BooleanSymbol[e_]] := e;
fromBoole[t:True|False] := t;
fromBoole[_] := $Failed;

BooleanSymbol /: (Which[first_BooleanSymbol, rest__]) := Scope[
	args = {first, rest};
	conditions = fromBoole /@ args[[1 ;; ;; 2]];
	values = toSource /@ args[[2 ;; ;; 2]];
	DerivedScalar[Flatten @ {Which, Transpose @ {conditions, values}}]
]

ScalarPiecewise[conds_List, value_:0] := 
	DerivedScalar[{Piecewise, piecewiseClause /@ conds, toSource @ value}];

piecewiseClause[{cond_, val_}] := {fromBoole[cond], toSource[val]};
piecewiseClause[_] := $Failed;

(* Scalar upvalues *)

PackageScope["ScalarFunctionToPureFunction"]

ScalarFunctionToPureFunction[e_, head_:None] := If[head =!= None, head[e], e];
ScalarFunctionToPureFunction[ScalarFunctionObject[in_, out_, ssa_], head_:None] := Scope[
	expr = If[head =!= None, Function[head[out]], Function[out]];
	expr = expr //. s_ScalarSymbol :> RuleCondition @ Lookup[ssa, s, Fail];
	res = expr /. ScalarSymbol[n_] :> Slot[n] //. {s_Symbol | s_String, args__} :> s[args];
	res = res /. HoldPattern[Identity[e_]] :> e;
	res /. r_Real /; (r == Round[r]) :> RuleCondition @ Round[r]
];

PackageScope["CompileScalarFunction"]
PackageScope["ScalarFunctionObject"]

CompileScalarFunction::message = "Function issued message during symbolic evaluation."
CompileScalarFunction::invret = "Function did not return a ScalarSymbol, returned `` instead."

CompileScalarFunction[Automatic, f_] := Scope[
	count = FunctionArgumentCount[f];
	If[IntegerQ[count],
		Return[CompileScalarFunction[count, f]];
	];
	Do[
		res = CompileScalarFunction[i, f];
		If[!FailureQ[res], Return[res]]
	,
		{i, 0, 5}
	];
	$Failed
];

CompileScalarFunction[ninputs_, f_] := CatchFailure @ Scope[
	$SSA = <||>;
	$SSACache = <||>;
	$NextID = 1;
	insyms = Table[ScalarSymbol[$NextID++], ninputs];
	Block[{Piecewise}, 
		Piecewise[s___] := ScalarPiecewise[s];
		result = norm @ Quiet @ Check[f @@ insyms, $Failed];
	];
	If[FailureQ[result], ReturnFailure["message"]];
	If[!MatchQ[result, _ScalarSymbol], ReturnFailure["invret", result]];
	ScalarFunctionObject[insyms, result, $SSA]
];

DefineCustomBoxes[
	ScalarFunctionObject,
	t:ScalarFunctionObject[in_List, out_, ssa_Association] :> 
		makeScalarFunctionObjectBoxes[t]
];

ScalarFunctionObject[in_, out_, ssa__Association]["SSA"] := ssa;
ScalarFunctionObject[in_, out_, _]["ArgumentCount"] := Length[in];

ScalarSymbol /: MakeBoxes[ScalarSymbol[n_Integer], TraditionalForm] := 
	SubscriptBox["x",StyleBox[IntegerString[n],5]];

nonFirstSeqToQ[from_ -> to_] := MatchQ[$SSA[to], {PackScalar, {__, Except[from]}}];
nonFirstSeqFromQ[from_ -> to_] := MatchQ[$SSA[to], {Part, _, Except[1]}];

makeScalarFunctionObjectBoxes[ScalarFunctionObject[in_, out_ScalarSymbol, <||>]] :=
	MakeBoxes[out, TraditionalForm];

makeScalarFunctionObjectBoxes[ScalarFunctionObject[in_, out_, ssa_]] := Scope[
	$SSA = ssa;
	rules = DependenceRules[ssa];
	{edges, vnames} = Labelling[rules, 2];
	inids = Flatten @ Position[vnames, Alternatives @@ in];
	outids = Flatten @ Position[vnames, out];
	plot = LayerPlot[List @@@ edges, 
		"VertexLabels" -> Placed[makeLabel /@ vnames, Right],
		"VertexSizes" -> 5,
		"Rotated" -> False,
		"ImageScale" -> 45,
		"ArrowSize" -> 6,
		"ArrowShape" -> "Chevron",
		"DuplicateInputVertices" -> True,
		"ImagePadding" -> {{5,15}, {15, 5}},
		"VertexTypes" -> {inids -> 1, outids -> 2, _ -> 3},
		"VertexTypeData" -> <|"VertexStyles" -> {RGBColor[0, 0.61, 1], RGBColor[0.83, 0.18, 0], Black}|>,
		"BaseLabelStyle" -> {FontSize -> 7},
		"MaximumImageSize" -> None
	];
	plot = Framed[plot, FrameStyle -> LightGray];
	ToBoxes[ScalarFunctionObject[plot]]
]

makeLabel[vert_] := fmtDerivation[vert, Lookup[$SSA, vert]];

DependenceRules[tsrcs_] := Flatten @ KeyValueMap[
	Thread[DeepCases[#2, _ScalarSymbol] -> #1]&,
	tsrcs
];

fmtDerivation[lhs_, _Missing] := lhs;
fmtDerivation[lhs_, {f_, rest___}] := Inactive[Set][lhs, HoldForm[f[rest]]];


PackageScope["SowScalarFunction"]

$MXBinOps = <|
	Plus     -> {"_Plus", "_PlusScalar", "_PlusScalar"},
	Subtract -> {"_Minus", "_MinusScalar", "_RMinusScalar"},
	Times    -> {"_Mul", "_MulScalar", "_MulScalar"},
	Divide   -> {"_Div", "_DivScalar", "_RDivScalar"},
	Power    -> {"_Power", "_PowerScalar", "_RPowerScalar"},
	Min      -> {"_Minimum", "_MinimumScalar", "_MinimumScalar"},
	Max      -> {"_Maximum", "_MaximumScalar", "_MaximumScalar"},
	Less      -> {"_lesser", "_lesser_scalar", "_greater_scalar"},
	LessEqual -> {"_lesser_equal", "_lesser_equal_scalar", "_greater_equal_scalar"},
	Greater   -> {"_greater", "_greater_scalar", "_lesser_scalar"},
	GreaterEqual -> {"_greater_equal", "_greater_equal_scalar", "_lesser_equal_scalar"},
	Equal -> {"_equal", "_equal_scalar", "_equal_scalar"},
	Unequal -> {"_not_equal", "_not_equal_scalar", "_not_equal_scalar"}
|>;

SowScalarFunction[ScalarFunctionObject[ins_, out_, ssa_], inIDs___] := Scope[
	inIDs = {inIDs};
	Assert[Length[inIDs] == Length[ins]];
	$SFID = AssociationThread[ins, inIDs];
	KeyValueScan[SowSSA, ssa];
	$SFID @ out
];

SowScalarFunction[f:(Alternatives @@ $PrimitiveUnaryElementwiseFunctions), id1_] :=
	SowNode[$MXUnaryFunctionNameMapping @ f, id1];

SowScalarFunction[f:(Alternatives @@ $PrimitiveBinaryElementwiseFunctions), id1_, id2_] :=
	SowNode[$MXBinaryFunctionNameMapping @ f, {id1, id2}];

SowScalarFunction[f:(Alternatives @@ $PrimitiveNAryElementwiseFunctions), ids__] := Scope[
	ids = {ids}; len = Length[ids];
	func = $MXBinaryFunctionNameMapping @ f;
	If[len > 2,
		Fold[SowNode[func, {#1, #2}]&, ids],
		SowNode[func, ids]
	]
];

SowScalarFunction[___] := Panic[];


SowSSA[sym_, op_] := Set[$SFID[sym], Apply[SSAtoOp, op]];

Clear[SSAtoOp];

SSAtoOp[Power, a_, 2.] :=  SowNode["square", $SFID[a]];

SSAtoOp[Minus, a_] := SSAtoOp[Subtract, 0.0, a];

(*----------------------------------------------------------------------------*)
(* String Arg Case *)

SSAtoOp["RectifiedLinearUnit", a_] := SowNode["relu", $SFID[a]]

(* "Self-Normalizing Neural Networks", https://arxiv.org/abs/1706.02515
Implementation taken from official TF implementation: 
	https://github.com/bioinf-jku/SNNs/blob/master/selu.py
*)
SSAtoOp["ScaledExponentialLinearUnit", a_] := 
	SowTimesScalar[
		safeLeakyRELU[$SFID[a], "elu", 1.6732632423543],
	 	1.0507009873554
	 ]

(* Definition from Torch: 
	https://github.com/torch/nn/blob/master/doc/transfer.md#elu *)
SSAtoOp["ExponentialLinearUnit", a_] := 
	safeLeakyRELU[$SFID[a], "elu", "1"];

safeLeakyRELU[in_, type_, slope_] := Scope[
	in = SowInsertDim[in, "0"];
	out = SowNode["LeakyReLU", in, "act_type" -> type, "slope" -> slope];
	SowFlatten[out]
];

(* Definition from Torch: 
	https://github.com/torch/nn/blob/master/doc/transfer.md#softsign *)
SSAtoOp["SoftSign", a_] := 
	SowDivide[
		$SFID[a], 
		SowPlusScalar[SowNode["abs", $SFID[a]], 1]
	];

(* Definition from Torch: 
	https://github.com/torch/nn/blob/master/doc/transfer.md#softplus *)
SSAtoOp["SoftPlus", a_] := SowNode["Activation", $SFID[a], "act_type" -> "softrelu"];

SSAtoOp["HardTanh", a_] := SowHardTanh[$SFID[a]];

SSAtoOp["HardSigmoid", a_] := SowHardSigmoid[$SFID[a]];

SSAtoOp["Sigmoid", a_] := SowSigmoid[$SFID[a]];

(* boolean operations *)

SSAtoOp[Boole, x_] := $SFID[x];
SSAtoOp[And, args__] := SowAnd[Sequence @@ $SFID /@ {args}];
SSAtoOp[Or, args__] := SowOr[Sequence @@ $SFID /@ {args}];
SSAtoOp[Not, x_] := SowNot[$SFID @ x];
SSAtoOp[Which, cond_, val_, rest___] := SowIf[$SFID @ cond, $SFID @ val, SSAtoOp[Which, rest]];
SSAtoOp[Piecewise, list_, else_] := SSAtoOp[Which, Sequence @@ Flatten[list], True, else];
	
SSAtoOp[Which, True, val_] := $SFID[val];

SSAtoOp[Identity, r_Real] := SowTimesScalar[$onesNode, r];
$onesNode := CacheTo[$SFID, Null, SowNode["ones_like", $SFID[ScalarSymbol[1]]]];

SSAtoOp[If, cond_, true_, false_] := SowIf[$SFID @ cond, $SFID @ true, $SFID @ false];

SowIf[cond_, true_, false_] := SowPlus[SowHad[cond, true], SowHad[SowOneMinus[cond], false]];

SSAtoOp[Clip, x_, {a_Real, b_Real}] := SowNode["clip", $SFID[x], "a_min" -> a, "a_max" -> b];
SSAtoOp[f_, a_, b_Real] := SowNode[$MXBinOps[f][[2]], $SFID[a], "scalar" -> b];
SSAtoOp[f_, a_Real, b_] := SowNode[$MXBinOps[f][[3]], $SFID[b], "scalar" -> a];
SSAtoOp[f_, a_, b_] :=     SowNode[$MXBinOps[f][[1]], {$SFID[a], $SFID[b]}];

SSAtoOp[f_, a_] := SowScalarFunction[f, $SFID[a]];


(*----------------------------------------------------------------------------*)

_SSAtoOp := $Unreachable;






