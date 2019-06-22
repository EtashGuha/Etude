Package["NeuralNetworks`"]



PackageScope["BlockDuringLoad"]
PackageScope["$IsLoading"]

$IsLoading = True;

SetHoldAll[BlockDuringLoad];
BlockDuringLoad[body_] := If[$IsLoading, Identity, body];


PackageScope["$Raw"]


PackageExport["$NetHeads"]

$NetHeads = {};


PackageScope["ReinitializeNeuralNetworks"]
PackageScope["RunInitializationCode"]

SetHoldAll[RunInitializationCode];

$InitializationCode = Bag[];
RunInitializationCode[code_] := (
	code;
	BagPush[$InitializationCode, Unevaluated[code]];
);

ReinitializeNeuralNetworks[] := 
	Quiet[BagContents @ $InitializationCode, RuleDelayed::rhs];


PackageExport["$NetInteractivity"]

If[!ValueQ[$NetInteractivity],
	$NetInteractivity = !$CloudEvaluation && !TrueQ[Developer`$DisableInteractivity];
];

PackageScope["SetCatchFailureAsMessage"]
PackageScope["TagSetCatchFailureAsMessage"]

SetHoldAllComplete[SetCatchFailureAsMessage];
SetHoldAllComplete[TagSetCatchFailureAsMessage];

(* this exists for speed purposes: avoid invoking the macro system *)
SetCatchFailureAsMessage[head_, lhs_, rhs_] := 
	SetDelayed @@ Hold[lhs, Catch[rhs, 
		GeneralUtilities`PackageScope`CatchFailureTag, 
		GeneralUtilities`PackageScope`CatchFailureAsMessageHandler[head]
	]];

TagSetCatchFailureAsMessage[head_, lhs_, rhs_] := 
	TagSetDelayed @@ Hold[head, lhs, Catch[rhs, 
		GeneralUtilities`PackageScope`CatchFailureTag, 
		GeneralUtilities`PackageScope`CatchFailureAsMessageHandler[head]
	]];

(* make sure this definition gets applied before downvalues are created *)
ScalarT = TensorT[{}, RealT];
TensorT[dims_] := TensorT[dims, RealT];


PackageExport["AnyTensorT"]
PackageExport["RealTensorT"]

AnyTensorT = TensorT[ListT[NaturalT, SizeT], AtomT];
RealTensorT = TensorT[ListT[NaturalT, SizeT], RealT];


PackageScope["VarSequenceP"]

VarSequenceP[] := TensorT[{_LengthVar, ___}, _]
Quiet[
VarSequenceP[lvar_] := TensorT[{LengthVar[lvar], ___}, _]
]


PackageExport["SequenceT"]

SequenceT[n_, t_] := TensorT[{n}, t];


PackageScope["EncoderP"]
PackageScope["DecoderP"]
PackageScope["UnevaluatedCoderP"]
PackageScope["UnevaluatedLayerP"]
PackageScope["CoderP"]

EncoderP[kind_] := HoldPattern[NetEncoder[kind, _, _]];
EncoderP[kind_, assoc_] := HoldPattern[NetEncoder[kind, assoc, _]];
EncoderP[kind_, assoc_, type_] := HoldPattern[NetEncoder[kind, assoc, type]];
DecoderP[kind_] := HoldPattern[NetDecoder[kind, _, _]];
DecoderP[kind_, assoc_] := HoldPattern[NetDecoder[kind, assoc, _]];
DecoderP[kind_, assoc_, type_] := HoldPattern[NetDecoder[kind, assoc, type]];
CoderP = _NetEncoder | _NetDecoder;

UnevaluatedCoderP = HoldPattern[(NetEncoder|NetDecoder)[_String, _Association, _?ValidTypeQ] ? System`Private`HoldEntryQ];
UnevaluatedLayerP := UnevaluatedLayerP = ((Alternatives @@ $NetHeads)[_Association, _Association]) ? System`Private`HoldEntryQ;


PackageScope["EvalNestedCodersAndLayers"]

(* evaluate any nested coders and layers *)
EvalNestedCodersAndLayers[expr_] :=  ReplaceAll[expr, $nestedRules];
$nestedRules := $nestedRules = Dispatch @ {
	coder:UnevaluatedCoderP :> RuleCondition[coder],
	layer:UnevaluatedLayerP :> RuleCondition[layer]
}



PackageScope["DurationP"]

DurationP = HoldPattern @ Quantity[_ ? Positive, "Seconds"|"Minutes"|"Hours"|"Days"|"Weeks"];


PackageScope["Self"]


PackageScope["NetP"]
PackageScope["NetGraphP"]

ClearAll[NetP];

(* This makes it easier to declare patterns that work on either heads or pure assocs *)
Quiet[
NetP[head_Symbol, data_Symbol, meta_Symbol] := head_Symbol[data_Association, meta_Association];
NetP[data_Symbol, meta_Symbol] := _Symbol[data_Association, meta_Association];
NetP /: Verbatim[Pattern][sym_Symbol, NetP] := sym_Association | _Symbol[sym_Association, _Association];
NetP /: Verbatim[Blank][NetP] := NetP;
NetGraphP /: Verbatim[Pattern][sym_Symbol, NetGraphP] := sym_Association | HoldPattern[NetGraph[sym_Association, _Association]];
NetGraphP /: Verbatim[Blank][NetGraphP] := NetGraphP;
,
RuleDelayed::rhs
];



PackageScope["NormalizePureFunction"]

SetUsage @ "
NormalizePureFunction[Function[$$]] processes a pure function to expand any macros in it if \
the body of the function calls Scope or UseMacros. 
* This is more visible kind of magic than the previous behavior, which automatically had a \
scope-like behavior in pure functions.
* It also checks whether you tried to use single-arg Return, which doesn't work as expected."

NormalizePureFunction[e_] := e;
NormalizePureFunction[f:Function[_Scope | _ModuleScope | _UseMacros]] := checkReturn @ MacroEvaluate @ f;
NormalizePureFunction[f_Function] := checkReturn @ f;

General::netbadret = "Don't use single arg Return[...] inside pure functions. It doesn't do what you think. The pure function was:\n``"
checkReturn[f_] := (If[!FreeQ[f, HoldPattern @ Return[]], Message[General::netbadreturn, f]]; f)


PackageScope["AMap"]

AMap[f_, list_] := Association[# -> f[#]& /@ list];
AMap[f_][list_] := AMap[f, list];


PackageScope["IMap"]

IMap[f_, assoc_Association] := MapIndexed[f[#2[[1,1]], #1]&, assoc];
IMap[f_, list_] := MapIndexed[f[#2[[1]], #1]&, list];
IMap[f_][list_] := IMap[f, list];


PackageScope["OnFail"]

SetAttributes[OnFail, HoldFirst];

OnFail[expr_][result_] := OnFail[expr, result];
OnFail[expr_, result_] := If[FailureQ[result], expr, result];


PackageScope["SizeP"]
PackageScope["RankTP"]
PackageScope["PosIntP"]
PackageScope["NatP"]
PackageScope["StringP"]
PackageScope["SizePairP"]
PackageScope["SizeTripleP"]
PackageScope["NetPathP"]
PackageScope["NetPathElemP"]
PackageScope["SizeListP"]
PackageScope["TTypeP"]

RankTP = SizeT | NaturalT;
PosIntP = _Integer ? Positive;
NatP = _Integer ? NonNegative | SizeT | NaturalT;
SizeP = PosIntP | SizeT;
StringP = _String | StringT;
SizePairP = ListT[2, SizeT] | {SizeP, SizeP};
SizeTripleP = ListT[3, SizeT] | {SizeP, SizeP, SizeP};
NetPathElemP = _Integer | _String;
NetPathP = NetPath[NetPathElemP..];
SizeListP = ListT[NatP, SizeT] | {___LengthVar, NatP...};
TTypeP = _IndexIntegerT|RealT|AtomT;


SetAttributes[ComputedType, HoldRest];


PackageScope["RawComputedType"]

SetAttributes[RawComputedType, HoldAll];



(* canonicalization code: this has to live here to ensure these short forms 
evaluate properly in LHS code *)


PackageScope["indexOf"]

SetAttributes[indexOf, HoldRest];
indexOf[list_List, thing_, else_] := Block[{t = thing},
	Do[If[list[[i]] === t, Return[i, Block]], {i, Length[list]}];
	else
];


PackageScope["PrefixPorts"]
PackageScope["$path"]

PrefixPorts[e_] := If[$path === NetPath[], e, e /. p_NetPath :> RuleCondition[Join[$path, p]]];

$path = NetPath[];


PackageScope["FreshPath"]

SetHoldFirst[FreshPath];

(* use this to make sure user callbacks don't stomp / get confused by $path *)
FreshPath[body_] := Block[{$path = NetPath[]}, body];


PackageScope["MapAtFields"]

MapAtFields[field_, f_, net_NetP] := Block[
	{$path = Join[$path, NetPath[field, Null]]},
	MapAt[
		MapIndexed[($path[[-1]] = #2[[1,1]]; f[#1])&],
		net,
		field
	]
];

MapAtFields[field_, f_][net_] := MapAtFields[field, f, net];


PackageScope["ScanSubNets"]

ScanSubNets[f_, assoc_] := Scan[
	Block[{$path = Join[$path, #]}, f[assoc @@ #]]&,
	SubNetPaths[assoc]
];

ScanSubNets[f_][assoc_] := ScanSubNets[f, assoc];


PackageScope["MapAtSubNets"]

MapAtSubNets[f_, assoc_] := ReplacePart[assoc,
	Map[
		Block[{$path = Join[$path, #]}, (List @@ #) -> f[assoc @@ #]]&,
		SubNetPaths[assoc]
	]
];

PackageScope["ScanFields"]

ScanFields[field_, f_, net_NetP] := Block[
	{$path = Join[$path, NetPath[field, Null]]},
	KeyValueScan[($path[[-1]] = #1; f[#2])&, net[field]];
];
ScanFields[field_, f_][net_] := ScanFields[field, f, net];


PackageScope["ScanNodes"]

ScanNodes[f_][net_] := ScanFields["Nodes", f, net];
ScanNodes[f_, net_] := ScanFields["Nodes", f, net];


PackageScope["MapFields"]

MapFields[field_, f_, net_NetP] := Block[
	{$path = Join[$path, NetPath[field, Null]]},
	KeyValueMap[($path[[-1]] = #1; f[#2])&, net[field]]
];
MapFields[field_, f_][net_] := MapFields[field, f, net];


PackageScope["OrOperator"]

OrOperator[list_][e_] := AnyTrue[list, #[e]&];


PackageScope["ComposeThread"]

ComposeThread[funcs_][args_] := ComposeThread[funcs, args];

ComposeThread[funcs_, args_] :=
	MapThread[Compose, {funcs, args}];

ComposeThread[funcs_, assoc_Association] :=
	Block[{i = 0}, Map[funcs[[++i]][#]&, assoc]];


PackageScope["MapAssocAssoc"]

SetUsage @ "
MapAssocAssoc[f, assoc$1, assoc$2, missf$, extraf$] calls f[key$, v$1, v$2] for corresponding\
values in assoc$1, assoc$2, in the order of the keys in assoc$1, but if a key in assoc$2
is missing, missf$[key$] is called, or if an extra key is present, extraf$[key$] is called."

MapAssocAssoc[f_, a_, b_, missf_, extraf_] :=
	If[Length[a] =!= Length[b],
		findBadKeys[a, b, missf, extraf],
		KeyValueMap[
			f[#1, #2, Lookup[b, #1, Return[missf[#1]]]]&,
			a
		]
	];

findBadKeys[a_, b_, missf_, extraf_] := Scope[
	ka = Keys[a]; kb = Keys[b];
	missf @ First[
		Complement[ka, kb],
		Return @ extraf @ First @ Complement[kb, ka]
	]
];


PackageScope["PartElse"]

SetHoldRest[PartElse];
PartElse[expr_, p___, else_] := 
	Replace[
		UnsafeQuietCheck[expr[[p]], Missing[]],
		_Missing :> else
	];


PackageScope["PartExistsQ"]

PartExistsQ[expr_, part___] := 
	UnsafeQuietCheck[expr[[part]]; True, False];


PackageScope["ParseWorkingPrecision"]

(* Real16 is not on this list as its not documented. Add back when it is *)
General::badwp = "Setting of WorkingPrecision should be either Automatic, \"Real32\", \"Mixed\", or \"Real64\".";

ParseWorkingPrecision[wp_] := Scope[
	If[!MatchQ[wp, "Real16"|"Real32"|"Real64"|"Mixed"|Automatic], ThrowFailure["badwp"]];
	ToDataTypeCode @ Replace[wp, {Automatic|"Mixed" -> "Real32"}]
];


PackageScope["ParseMultiContext"]

ParseMultiContext[s_String] := ParseTargetDevice[s];
ParseMultiContext[other_] := ParseTargetDevice[other];

ParseMultiContext[{"CPU", 1}] := 1;
ParseMultiContext[{"CPU", n_Integer}] := Table[1, n]; (* <- remove me, for debugging *)
ParseMultiContext[c:{_String, _List}] := ParseTargetDevice /@ Thread[c];
ParseMultiContext[{dev_String, {n_Integer ? Positive}}] := ParseMultiContext[{dev, n}]
ParseMultiContext[{"GPU", All}] := Scope[
	(* download lib if necessary / throw failure if no GPU *)
	If[$GPUCount === 0, ParseTargetDevice["GPU"]];
	(* In the future we could be able to avoid throwing failure
	   after downloading libs, so we may reach this point after 
	   a download and act immediatly by rechecking $GPUCount *)
	If[$GPUCount === 1,
		ParseTargetDevice["GPU"],
		Table[ParseTargetDevice[{"GPU", i}], {i, $GPUCount}]
	]
];


PackageScope["ParseEvalMode"]

General::invnem = "NetEvaluationMode should be set to either \"Test\" or \"Train\".";
ParseEvalMode[spec_] := Switch[spec, 
	"Train", True, "Test", 
	False, _, 
	ThrowFailure["invnem"]
]


PackageScope["UnparseContext"]

UnparseContext[code_] := Replace[FromContextCode[code], {s_, 0} :> s];


PackageScope["ParseTargetDevice"]

Clear[ParseTargetDevice];

ParseTargetDevice["CPU"] = 1; 
ParseTargetDevice["GPU"] := ParseTargetDevice[{"GPU", 1}];

ParseTargetDevice[context:{"GPU", n_Integer ? Positive}] := Scope[
	If[$GPUCount > 0 && n > $GPUCount, ThrowFailure["trgdevgpumax", context, $GPUCount]];
	code = ToContextCode[{"GPU", n - 1}];
	Which[
		$SystemWordLength === 32,
			General::trgdev32 = "TargetDevice -> \"GPU\" is not supported on 32-bit platforms.";
			ThrowFailure["trgdev32"],
		$CloudEvaluation,
			General::trgdevcloud = "TargetDevice -> \"GPU\" is not supported on the Wolfram Cloud.";
			ThrowFailure["trgdevcloud"],
		$MXRestartRequiredQ,
			General::trgrestart = "TargetDevice -> \"GPU\" requires a restart of your Wolfram Language session.";
			ThrowFailure["trgrestart"],
		!$MXGPUEnabledQ,
			MXNetResourcesInstall[First[$CatchFailureHead]]; 
			(* we always need to fail in this case *)
			ThrowFailure[],
		Quiet @ FailureQ @ CatchFailure[General, MXNetLink`NDArrayCreate[{1,2,3}, code]],
			ThrowTargetDeviceFailure @ If[n === 1, "GPU", {"GPU", n}];
		,
		True,
			ParseTargetDevice[context] = code
	]
]

ParseTargetDevice[{"GPU", {n_Integer ? Positive}}] := ParseTargetDevice[{"GPU", n}]

General::invtrgdev = "Invalid setting TargetDevice -> ``. TargetDevice should be either \"CPU\" or \"GPU\", or {\"GPU\", n} for multi-GPU systems, where n is a positive integer that specifies which system GPU to use."
ParseTargetDevice[spec_] := ThrowFailure["invtrgdev", spec];


PackageScope["ToVersionString"]

(* only 11.0 and early 11.1 era nets used reals, now we use PacletInformation. *)
ToVersionString[r_Real | r_Integer] := Match[
	Round[r, 0.001], 
	11. :> "11.0.0", 
	11.1 :> "11.1.0",
	11.05 :> ThrowFailure["wlprerl"]
];
ToVersionString[s_String] := s;
ToVersionString[_] := corrupt[];


PackageScope["FromVersionString"]

FromVersionString[s_String] := FromDigits /@ StringSplit[s, "."];
FromVersionString[_] := corrupt[];


PackageScope["VersionOrder"]

VersionOrder[a_String, b_String] /; StringLength[a] == StringLength[b] == 6 := Order[a, b];
(* ^ need special treatment when minor or point version gets to double digits *)

VersionOrder[a_String, b_String] := Order[FromVersionString[a], FromVersionString[b]]


PackageExport["$MXNetVersion"]

$MXNetVersion := $MXNetVersion = MXNetLink`GetMXNetVersion[];


PackageScope["$ExpectedMXNetLinkVersion"]

$ExpectedMXNetLinkVersion = "12.0.0";

General::mxnetlinkhi = "The version of MXNetLink being loaded from \"``\" appears to be higher than ``. This may cause problems."
General::mxnetlinklo = "The version of MXNetLink being loaded from \"``\" appears to be lower than ``. This may cause problems."

(* Note to developer: When you encounter the above message after upgrading MXNet, 
update $ExpectedMXNetVersion and related variables at the top of this file *)

$mxNetLinkVersion := Lookup[PacletManager`PacletInformation["MXNetLink"], "Version", "0.0.0"]
$mxNetLinkPath := Lookup[PacletManager`PacletInformation["MXNetLink"], "Location"];


PackageExport["$DefaultKVStoreType"]

$DefaultKVStoreType = "device";


RunInitializationCode[
	Switch[
		VersionOrder[$mxNetLinkVersion, $ExpectedMXNetLinkVersion],
		-1, Message[General::mxnetlinkhi, $mxNetLinkPath, $ExpectedMXNetLinkVersion],
		 0, Null,
		 1, Message[General::mxnetlinklo, $mxNetLinkPath, $ExpectedMXNetLinkVersion]
	];
];


PackageExport["ValidNetQ"]

ValidNetQ[sym_Symbol[_Association, _Association] ? System`Private`NoEntryQ] := MemberQ[$NetHeads, sym];
ValidNetQ[_] := False;


PackageScope["DigitStringKeysQ"]
PackageScope["DigitStringVectorQ"]

DigitStringKeysQ[assoc_] := DigitStringVectorQ[Keys[assoc]];
DigitStringVectorQ[list_] := VectorQ[list, StringMatchQ[DigitCharacter..]];

PackageScope["RemapKeys"]

RemapKeys[assoc_] := Scope[
	newkeys = IntegerString @ Range[Length[assoc]];
	remapping = AssociationThread[Keys[assoc], newkeys];
	{KeyMap[remapping, assoc], remapping}
];


PackageScope["MakeCustomHeadBox"]

MakeCustomHeadBox[head_, contents_, baseline_] := Scope[
	boxes = ToBoxes @ Panel[contents, BaselinePosition -> If[$CloudEvaluation, Automatic, baseline]];
	StyleBox[TagBox[TagBox[
		RowBox[{head, "[", boxes, "]"}],
		False
	], Deploy], LineBreakWithin -> False]
];


PackageScope["RemoveArrays"]

RemoveArrays[expr_Association] := 
	expr /. ra_NumericArray :> RuleCondition @ TensorT @ Dimensions[ra];


PackageScope["ReplaceArraysWithDummies"]
PackageScope["ReplaceCoderInfoWithDummies"]
PackageScope["DummyArray"]

(* see 331893 for idea of why this is necessary *)

ReplaceArraysWithDummies[expr_] := 
	expr /. {
		ra_NumericArray :> RuleCondition @ DummyArray @ Dimensions[ra],
		(* bug 336789 - and see typeInfo[] *)
		cod:CoderP :> RuleCondition @ Join[
			<|"Form" -> CoderType[cod], "Type" -> CoderName[cod]|>,
			ReplaceCoderInfoWithDummies[CoderData[cod]]
		]
	};

ReplaceCoderInfoWithDummies[expr_] := 
	expr /. {
		ra_NumericArray :> RuleCondition @ DummyArray @ Dimensions[ra],
		l_List  :> RuleCondition @ If[
			Length[l] <= 4, stringInputForm /@ l,
				(* Do not need to display all elements for big lists - bug 336789 *)
				Join[
					stringInputForm /@ Take[l, 2],
					{Skeleton[Length[l] - 3]},
					stringInputForm /@ Take[l, -1]
				]
		],
		(* Do not keep compressed data (ex: NetEncoder["Tokens"]) *)
		TokenEncodingData[1, _, len_Integer] :> Skeleton[Row[{len, " strings"}]],
		(* For the Function of NetEncoder["Function"] *)
		HoldPattern[Function[x_,__]] :> Function[x, "\[Ellipsis]"],
		HoldPattern[Function[_]] :> Function[x, "\[Ellipsis]"]
	};
stringInputForm[e_String] := ToString[e, InputForm];
stringInputForm[e_] := e;

PackageScope["fromStringForm"]

fromStringForm[e_] := If[$CloudEvaluation, ReplaceAll[e, s_StringForm :> ToString[s]], e];


PackageScope["CreateFunction"]
PackageScope["Eval"]
PackageScope["TempVar"]
PackageScope["SlotPart"]

CreateFunction[statements_] := Scope[
	body = BodyFunctionReplaceRepeated[
		Hold @@ DeleteCases[Flatten[List[statements]], Null],
		Eval[e_] :> RuleCondition[e]
	];
	fastBody = ReplaceAll[body, a_Association /; AssociationQ[Unevaluated[a]] :> Null];
	If[!FreeQ[fastBody, TempVar],
		rules = Map[# -> toVar[First @ #]&, DeleteDuplicates @ DeepCases[fastBody, _TempVar]];
		body = Hold[Block][Values[rules], BodyFunctionReplaceAll[body, rules]];
	];
	BodyFunctionReplaceRepeated[Compose[Function, body], {
		Hold[e_] :> e,
		Hold[ee__] :> CompoundExpression[ee],
		HoldPattern[Identity[e_]] :> e,
		SlotPart[i_] :> Part[Slot[1], i]
	}]
];

CreateFunction[body_, head_] := MapAt[head, CreateFunction[body], 1];

PackageScope["BodyFunctionReplaceAll"]
PackageScope["BodyFunctionReplaceRepeated"]
PackageScope["BodyFunctionFreeQ"]

SetUsage @ "
BodyFunctionReplaceAll[expr$, rules$] is to be used in place of ReplaceAll for body functions.
It does the same as ReplaceAll but does not look inside pre-constructed associations,
as these can be very large and hence super s_low (e.g. embedded tokens encoder).
See https://bugs.wolfram.com/show?number=369266
"
BodyFunctionReplaceAll[expr_, rule_Rule|rule_RuleDelayed] := BodyFunctionReplaceAll[expr, {rule}];
BodyFunctionReplaceAll[expr_, rules_] := ReplaceAll[expr,
	Prepend[rules,
		a_Association /; AssociationQ[Unevaluated[a]] :> a
	]
];

SetUsage @ "
BodyFunctionReplaceRepeated[expr$, rules$] is to be used in place of ReplaceRepeated for body functions.
It does the same as ReplaceRepeated but does not look inside pre-constructed associations,
as these can be very large and hence super slow (e.g. embedded tokens encoder).
See https://bugs.wolfram.com/show?number=369266
"
BodyFunctionReplaceRepeated[expr_, rule_Rule|rule_RuleDelayed] := BodyFunctionReplaceRepeated[expr, {rule}];
BodyFunctionReplaceRepeated[expr_, rules_] := Module[{patterns, out},
	patterns = Alternatives @@ Map[First, rules];
	out = expr;
	While[!BodyFunctionFreeQ[out, patterns, Infinity],
		out = BodyFunctionReplaceAll[out, rules]
	];
	out
];

SetUsage @ "
BodyFunctionFreeQ[expr$, form$] is to be used in place of FreeQ for body functions.
It does the same as FreeQ but does not look inside pre-constructed associations,
as these can be very large and hence super slow (e.g. embedded tokens encoder).
See https://bugs.wolfram.com/show?number=369266
"
BodyFunctionFreeQ[expr_, args__] := FreeQ[
	ReplaceAll[expr, a_Association /; AssociationQ[Unevaluated[a]] :> Null],
	args
];

toVar[1] := Hold @ NeuralNetworks`Private`$Temp1;
toVar[2] := Hold @ NeuralNetworks`Private`$Temp2;
toVar[3] := Hold @ NeuralNetworks`Private`$Temp3;
toVar[4] := Hold @ NeuralNetworks`Private`$Temp4;
toVar[i_Integer] := ToExpression["NeuralNetworks`Private`$Temp" <> IntegerString[i], InputForm, Hold];
toVar[sym_Symbol] := sym;

PackageScope["CreateUnpreemptableFunction"]

CreateUnpreemptableFunction[statements_] := CreateFunction[statements, PreemptProtect];


PackageScope["pluralStr"]

pluralStr[row_Row] := MapAt[pluralStr, row, {1,1}];

pluralStr[s_String] := Switch[
	StringTake[s, -1],
	"x", StringDrop[s, -1] <> "ces", 
	"s", s <> "es",
	_, s <> "s"
];

pluralStr[e_] := e;


PackageScope["$PrimitiveUnaryElementwiseFunctions"]

PackageExport["SoftRamp"]

$PrimitiveUnaryElementwiseFunctions = {
	ArcCos, ArcCosh, ArcSin, ArcSinh, ArcTan, ArcTanh, Cosh, Sinh, Gamma, LogGamma, (* <- new in 11.1 *)
	LogisticSigmoid, Ramp, SoftRamp, Tanh, Sin, Cos, Tan, Log, Exp, Sqrt, Square, Abs, Round, Ceiling, Floor, Sign, Erf
};


PackageScope["$PrimitiveBinaryElementwiseFunctions"]

$PrimitiveBinaryElementwiseFunctions = {Times, Plus, Divide, Subtract, Power, Min, Max};


PackageScope["$PrimitiveNAryElementwiseFunctions"]

$PrimitiveNAryElementwiseFunctions = {Times, Plus, Min, Max};


PackageScope["DeclareContainer"]

DeclareContainer[name_, symbol_] := (
	$SymbolToType[symbol] = name;
	$TypeToSymbol[name] = symbol;
	$VTable[name] = $VTable["GenericContainer"];
	ValidNetQ[symbol[_Association, _Association] ? System`Private`NoEntryQ] := True; 
);

jsonPathQ[path_] := StringEndsQ[path, ".json", IgnoreCase -> True]


PackageScope["$VTable"]

$VTable = Association[
	"GenericLayer" -> Association[],
	"GenericContainer" -> Association[],
	"GenericOperator" -> Association[]
];

PackageScope["DeclareMethod"]

SetUsage @ "
DeclareMethod[method$] defines a polymorphic method that when applied to a layer of type ltype$ \
will actually apply the function $VTable[ltype$, method$] to its assoc.
DeclareMethod[method$, layerf$] will call layerf$ on an ordinary layer's assoc, and when applied to \
a container will apply the method to all sub-layers recursively. 
DeclareMethod[method$, layerf$, containerf$] defines a method that calls layerf$ on layers \
and containerf$ on NetChains and NetGraphs.
DeclareMethod[method$, layerf$, containerf$, operatorf$] as above but calls operatorf$ on \
layers with subnets (e.g. NetMapOperator).
* If not supplied, the default for containerf$ is to recurse the method over the container's nodes.
* If not supplied, the default for operatorf$ is to apply layerf$."

$methodUsageTemplate = "`` is a function defined by DeclareMethod in file `` that operates on net associations, \
dispatching to different functions for layers, containers, and operators. These functions are:
For ordinary layers: ``
For containers: ``
For operators: ``
See the documentation of DeclareMethod for more information."

fastStringTemplate[str_, args___] := Block[{i = 1}, 
	StringReplace[str, "``" :> Replace[{args}[[i++]], {sym_Symbol :> SymbolName[sym], s_String :> s, _ -> "..."}]]
];

DeclareMethod[method_, layerf_:Function[Null], containerf_:Inherited, operatorf_:Inherited] := Block[{msgString},
	usageString = fastStringTemplate[$methodUsageTemplate, SymbolName @ method, $CurrentFileName, layerf, containerf, operatorf];
	method::usage = usageString;
	method[net_Association] := $VTable[net["Type"], method] @ net;
	$VTable["GenericLayer", method] = layerf;
	$VTable["GenericContainer", method] = If[containerf === Inherited, ScanNodes[method], containerf];
	$VTable["GenericOperator", method] = If[operatorf === Inherited, layerf, operatorf];
];

PackageScope["Call"]

Call[net_, method_] := $VTable[net["Type"], method] @ net;
Call[method_][net_] := Call[net, method];


PackageScope["SetupGenericDispatch"]

Clear[SetupGenericDispatch];

ClearAll[UpgradeMXSnapshotInplace];
SetHoldAllComplete[UpgradeMXSnapshotInplace];
UpgradeMXSnapshotInplace[(lhs:sym_[data_, metadata_]) /; metadata["Version"] =!= $NeuralNetworksVersionNumber] :=
	Module[{upgraded = CatchFailureAsMessage[sym, ReconstructNet[data, metadata, "upgrade"]]},
		If[!FailureQ[upgraded],
			Integrate`ResetElemNoCC[Unevaluated[lhs], 2, NetGetMetadata[upgraded]];
			Integrate`ResetElemNoCC[Unevaluated[lhs], 1, NData[upgraded]];
		];
	];
(* ^ this does an in-place update, by using a secret hack to change the elements of the expression struct in place. 
this is needed for nets that are saved in MX files in previous versions and then loaded in the future. if we 
don't use the hack, the evaluator will keep upgrading the net again and again every time it encounters the expression. *)

SetupGenericDispatch[sym_Symbol, addFastPath_] := (
	(s_sym ? System`Private`HoldNoEntryQ) ? UpgradeMXSnapshotInplace := Null;
	TagSetCatchFailureAsMessage[sym, Part[HoldPattern[sym[assoc_Association, _] ? System`Private`HoldNoEntryQ], part___], NetPart[assoc, part]];
	Language`SetMutationHandler[sym, NetMutationHandler];
	If[addFastPath, SetCatchFailureAsMessage[sym, (s_sym ? System`Private`HoldNoEntryQ)[arg_], NetApplyFast[s, arg]]];
	SetCatchFailureAsMessage[sym, (s_sym ? System`Private`HoldNoEntryQ)[args___], NetApply[s, args]];

);


PackageScope["StripCoders"]

StripCoders[e_] := ReplaceAll[e, c:CoderP :> RuleCondition[CoderType[c]]];


PackageScope["HoldSet"]

HoldSet[Hold[sym_], value_] := Set[sym, value];


PackageScope["$CloudOrPlayer"]

$CloudOrPlayer := $CloudOrPlayer = TrueQ[$CloudEvaluation] || MatchQ[$LicenseType, "Player" | "Player Pro"];


PackageScope["PrereleaseError"]

General::prerelerr = "Support for `` is not implemented in this pre-release version of Mathematica. Please try this feature again in the next pre-release."

PrereleaseError[feature_] := (
	InheritedMessage["prerelerr", feature];
	ThrowRawFailure[$Failed];
);


PackageScope["ToIntPair"]

ToIntPair[i_Integer] := {i,i};
ToIntPair[e_] := e;


PackageScope["ToIntTuple"]

ToIntTuple[i_Integer] := RepeatedInteger[i];
ToIntTuple[e_] := e;

PackageScope["ToIntMatrix"]

ToIntMatrix[i_Integer] := RepeatedInteger[i];
ToIntMatrix[l:{__Integer}] := Transpose @ {l, l}
ToIntMatrix[e_] := e;

PackageScope["$MXUnaryFunctionNameMapping"]

$MXUnaryFunctionNameMapping = Association[
	Ramp -> "relu",
	Sin -> "sin", 
	Cos -> "cos",
	Tan -> "tan",
	Log -> "log", 
	Exp -> "exp", 
	Sqrt -> "sqrt", 
	Square -> "square", (* yes i know *)
	Abs -> "abs", 
	Round -> "round", 
	Ceiling -> "ceil", 
	Floor -> "floor",
	Sign -> "sign",
	(* new in 11.1: *)
	ArcCos -> "arccos",
	ArcSin -> "arcsin",
	ArcSinh -> "arcsinh",
	ArcTan -> "arctan",
	ArcTanh -> "arctanh",
	ArcCosh -> "arccosh",
	Cosh -> "cosh",
	Sinh -> "sinh",
	Tanh -> "tanh",
	Gamma -> "gamma",
	LogGamma -> "gammaln",
	LogisticSigmoid -> "sigmoid",
	Erf -> "erf"
];


PackageScope["$MXBinaryFunctionNameMapping"]

$MXBinaryFunctionNameMapping = Association[
	Times -> "_Mul",
	Subtract -> "_Minus",
	Divide -> "_Div",
	Plus -> "_Plus",
	Power -> "_Power",
	Min -> "_Minimum",
	Max -> "_Maximum"
];


PackageScope["ObjectDefinitionT"]

ObjectDefinitionT[args___] := StructT[procLine /@ {
	args,
	"MaxArgCount" -> 	Defaulting[IntegerT, Automatic],
	"MinArgCount" -> 	Defaulting[IntegerT, Automatic],
	"PosArgCount" -> 	Defaulting[IntegerT, Automatic],
	"PosArgs" -> 		Defaulting[ListT[StringT], Automatic],
	"SourceFile" -> 	Defaulting[StringT, None]
}];

procLine[key_ -> val_Defaulting] := key -> val;
procLine[key_ -> val_List] := key -> Defaulting @ StructT @ Map[procLine, val];
procLine[key_ -> val_] := key -> Defaulting @ val;


PackageExport["DoWith"]

SetHoldFirst[DoWith];

DoWith[body_, varset___Rule] := Scope[
	syms = Keys[{varset}];
	vals = Values[{varset}];
	Scan[
		ReleaseHold[Hold[body] /. Thread[syms -> #]]&,
		Tuples[vals]
	]
];


PackageScope["TestValidNet"]

General::netarg1 = "First argument should be a valid net."

TestValidNet[net_] := If[!ValidNetQ[net], ThrowFailure["netarg1"]];


PackageScope["FastValueQ"]

DefineAlias[FastValueQ, System`Private`HasImmediateValueQ];


PackageScope["CTable"]

DefineAlias[CTable, ConstantArray];


PackageScope["$LengthVarFormatting"]

If[!FastValueQ[$LengthVarFormatting], $LengthVarFormatting = False];

DefineCustomBoxes[LengthVar,
	lv:LengthVar[id_Integer] /; $LengthVarFormatting :> ToBoxes[Interpretation[Hue[Mod[id, 26]/27.], lv]]
]


(* defined also in Multiport.m, this avoids a loading order hazard *)
$Multiport = "XXXxXXX";


PackageScope["$NNTestingMode"]

$NNTestingMode = False;
(* tweaks some details to be friendlier to PacletTools testing *)


PackageScope["ReloadNeuralNetworks"]

ReloadNeuralNetworks[] := (
	Quiet[
		ClearPacletLoadCache["NeuralNetworks"];
		ClearAll["NeuralNetworks`*"];
		ClearAll["NeuralNetworks`*`*"];
		Remove["NeuralNetworks`Private`*`*"];
		(* ^ we can only remove file-private symbols. we can't remove things in NN` like ListT, TensorT, etc,
		or even things in a NN`Private`, like ScalarFunctionObject, because this will mess up existing nets.
		so this function isn't infallable. *)
	];
	Quiet[
		Block[{$ContextPath = {"System`"}}, Get["NeuralNetworks`"]],
		General::shdw
	]
);

PackageScope["RenameDeprecatedOption"]

RenameDeprecatedOption[rules_List] :=
	ReplaceAll[
		old:Apply[Alternatives, Keys[rules]] :>
			With[{new = Replace[old, rules]},
				InheritedMessage["depropt", old, new];
				new
			]
	];
RenameDeprecatedOption[rule_Rule] := RenameDeprecatedOption[{rule}];


PackageScope["$RealisticSystemMemory"]

$RealisticSystemMemory := $RealisticSystemMemory = 
	Which[$CloudEvaluation, 1000*^6, $SystemWordLength == 32, 2000*^6, True, $SystemMemory];


PackageScope["HeldDefault"]

SetHoldAll[HeldDefault]
(* this exists to allow Quantity to be used as a default without forcing 
NN to load the quantity paclet at load time *)