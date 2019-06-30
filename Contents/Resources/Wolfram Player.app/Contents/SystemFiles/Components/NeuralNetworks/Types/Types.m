Package["NeuralNetworks`"]


(*****************************************************************************)
(********************************* SIMPLE TYPES ******************************)
(*****************************************************************************)


PackageExport["PosIntegerT"]

SetUsage @ "
PosIntegerT represents a positive integer."


PackageExport["IntegerT"]

SetUsage @ "
IntegerT represents an integer."


PackageExport["NaturalT"]

SetUsage @ "
NaturalT represents a non-negative integer."


PackageExport["RealT"]

SetUsage @ "
RealT represents an unbounded real number.
* It is usually used as the 'inner type' of a TensorT."


PackageExport["IndexIntegerT"]

SetUsage @ "
IndexIntegerT[max$] represents an integer in the range [1,max$].
IndexIntegerT[Infinity] represents a positive integer (range [1,Infinity]).
IndexIntegerT[All] represents an unbounded integer (either positive or negative).
IndexIntegerT[SizeT] represents an integer with undefined boundaries.
"

PackageExport["IntervalScalarT"]

SetUsage @ "
IntervalScalarT[min$,max$] represents a real number in the range [min$,max$]."


PackageExport["AtomT"]

SetUsage @ "
AtomT represents any of the types accepted as the element of a TensorT.
* For now, these are only RealT and IndexIntegerT."


PackageExport["StringT"]

SetUsage @ "
StringT represents a string."


PackageExport["BooleanT"]

SetUsage @ "
BooleanT represents the symbols True and False."


PackageExport["SizeT"]

SetUsage @ "
SizeT represents a positive integer used as a 'size' of some sort."


PackageExport["DurationT"]

SetUsage @ "
DurationT represents a quantity in terms of time units."

PackageScope["DefaultedType"]

SetUsage @ "
DefaultedType replace all the non-defined types that has a default by their default (e.g.: AtomT -> RealT)."

DefaultedType[t_] := ReplaceAll[t, {AtomT -> RealT}];

(*****************************************************************************)
(**************************** TENSOR-RELATED TYPES ***************************)
(*****************************************************************************)


PackageExport["TensorT"]

SetUsage @ "
TensorT[dims$] represents a real-valued array of dimensions dims$.
TensorT[dims$,type$] represents an array of dimensions dims$ whose elements are of type type$.
* The dims$ can be:
| SizeListT[n$] | n$ unknown dimensions |
| {d$1,$$,d$n} | explicit dimensions, each being an integer or SizeT |
* The type$ can be:
| RealT | floating point value |
| IndexIntegerT[n$] | integers in [1,n] |
| IndexIntegerT[Infinity] | positive integers |
| IndexIntegerT[All] | integers |
| TensorT[$$] | a sub-tensor |
* TensorT[{a$$}, TensorT[{b$$}, type$]] automatically flattens to TensorT[{a$,b$},type$]"

TensorT[{}, t:(_TensorT | _NetEncoder | _NetDecoder)] := t;
TensorT[dims1_List, TensorT[ListT[n_Integer, _], type_]] := TensorT[Join[dims1, CTable[SizeT, n]], type];
TensorT[dims1_List, TensorT[dims2_List, type_]] := TensorT[Join[dims1, dims2], type];
TensorT[ListT[n1_, SizeT], TensorT[ListT[n2_, SizeT], t_]] /; (Head[n1] === Head[n2]) := 
	TensorT[ListT[If[IntegerQ[n1], n1 + n2, NaturalT], SizeT], t];


PackageExport["ChannelT"]

SetUsage @ "
ChannelT is a shortcut for adding a single, outer dimension to a tensor."

ChannelT[n_, t_] := TensorT[{n}, t];


PackageExport["ScalarT"]

SetUsage @ "
ScalarT represents a 0-rank tensor, and is equivalent to TensorT[{}, RealT]."

ScalarT = TensorT[{}, RealT];


PackageExport["MatrixT"]

SetUsage @ "
MatrixT[rows$,cols$] evaluates to TensorT[{rows$,cols$}, RealT]."

MatrixT[] := TensorT[{SizeT, SizeT}, RealT];
MatrixT[rows_, cols_] := TensorT[{rows, cols}, RealT];


PackageExport["TensorWithMinRankT"]

SetUsage @ "
TensorWithMinRankT[n$] evaluates to TensorT that must be rank n or higher.
TensorWithMinRankT[n$, type$] sets the element type of the tensor."

TensorWithMinRankT[n_Integer] := TensorT[SizeListT[n], RealTensorT];
TensorWithMinRankT[n_Integer, t_] := TensorT[SizeListT[n], TensorT[SizeListT[], t]];


PackageExport["VectorT"]

SetUsage @ "
VectorT[n$] evaluates to TensorT[{n}, RealT].
VectorT[] evaluates to TensorT[{SizeT}, RealT]."

VectorT[] := VectorT[SizeT];
VectorT[size_] := VectorT[size, RealT];
VectorT[size_, type_] := TensorT[{size}, type];

(*****************************************************************************)
(******************************** COMPOUND TYPES *****************************)
(*****************************************************************************)


PackageExport["EitherT"]

SetUsage @ "
EitherT[{type$1,type$2,$$}] represents a value of one of the type$i."

EitherT /: TensorT[dims_, EitherT[list_List]] := EitherT[TensorT[dims, #]& /@ list];


PackageExport["ListT"]

SetUsage @ "
ListT[type$] represents a list of any length, whose values are of type type$.
ListT[n$, type$] represents a list of length n$."

ListT[type_] := ListT[NaturalT, type];


PackageExport["SizeListT"]

SetUsage @ "
SizeListT[] represents a list of positive integers.
SizeListT[n$] represents a list of n$ positive integers.
SizeListT[$$] evaluates to a ListT."

SizeListT[0] := {};
SizeListT[n_] := ListT[n, SizeT];
SizeListT[] := ListT[NaturalT, SizeT];


PackageExport["RuleT"]

SetUsage @ "
RuleT[lhs$, rhs$] represents a rule whose LHS is of type lhs$ and whose RHS is of type $rhs."


PackageExport["StructT"]

SetUsage @ "
StructT[{key$1->type$1,$$}] represents an association with keys key$i whose values match type$i."


PackageExport["AssocT"]

SetUsage @ "
AssocT[ktype$,vtype$] represents an association whose keys have type ktype$ and values have type vtype$."


(*****************************************************************************)
(***************************** 'EXTERNAL' TYPES ******************************)
(*****************************************************************************)


PackageExport["EnumT"]

SetUsage @ "
EnumT[{val$1,val$2,$$}] represents an expression whose values are one of the literal expressions val$i."


PackageExport["FunctionT"]

SetUsage @ "
FunctionT represents an expression with head Function."


PackageExport["MatchT"]

SetUsage @ "
MatchT[patt$] represents an expression which matches patt$."


PackageExport["ValidatedParameter"]

SetUsage @ "
ValidatedParameter[expr$] represents a user-provided spec that has been validated by a ValidatedParameterT."


PackageExport["ValidatedParameterT"]

SetUsage @ "
ValidatedParameterT[f$] represents an expression that should be constructed from user input by f$, then wrapped in ValidatedParameter.
ValidatedParameterT[f$, d$] specifies the default value should be ValidatedParameter[d$]."


PackageExport["NormalizedT"]

SetUsage @ "
NormalizedT[t$, f$] represents an expression of type t$ that is constructed from user input by f$.
NormalizedT[t$, f$, d$] specifies the default value should be given by f$[d$].
* NormalizedT is not actually a type, it is sugar that gets stripped off and used to populate the ParameterCoercions field."


PackageExport["ExpressionT"]

SetUsage @ "
ExpressionT represents any expression."


(*****************************************************************************)
(********************************* MODIFIERS *********************************)
(*****************************************************************************)


PackageExport["Defaulting"]

SetUsage @ "
Defaulting[type$,value$] represents an expression of type type$ that defaults to value$."

Defaulting[ValidatedParameterT[f_], v_] := ValidatedParameterT[f, v];
Defaulting[t_ListT] := Defaulting[t, {}];
Defaulting[t_AssocT] := Defaulting[t, <||>];
Defaulting[t_EnumT] := Defaulting[t, t[[1,1]]];
Defaulting[t_Nullable] := Defaulting[t, None];
Defaulting[BooleanT] := Defaulting[BooleanT, False];
Defaulting[t_] := Defaulting[t, None];


PackageExport["Nullable"]

SetUsage @ "
Nullable[type$] represents a type type$ that can also be None."


(*****************************************************************************)
(****************************** TYPE TYPES ***********************************)
(*****************************************************************************)


PackageExport["TypeT"]

SetUsage @ "
TypeT represents another type, like IntegerT or TensorT[$$]."


PackageExport["TypeExpressionT"]

SetUsage @ "
TypeExpressionT represents a type expression, which may involve types that reference ports."


PackageScope["CustomType"]

SetUsage @ "
CustomType['desc$', coercer$] represents a custom, one-off type.
* 'desc$' should descibe the type
* coercer$ is a function to coerce user data to the type, returning $Failed if it is incompatible."


PackageScope["ComputedType"]

SetUsage @ "
ComputedType[type$,expr$,{dep$1,$$}] represents a value of form type$ that is computed by expr$, and depends on ports dep$i.
ComputedType[type$,expr$,{dep$1,$$},trigger$] only attempts to compute expr$ when trigger$ is satisfied.
* ComputedType[type$,expr$] will fill in the deps$ automatically.
* The default trigger is when all of the dependancies are concrete (via ConcreteParameterQ)."

SetAttributes[ComputedType, HoldRest];


PackageScope["SwitchedType"]

SetUsage @ "
SwitchedType[port$,value$1->output$1,value$2->output$2,$$] allows a port or parameter to be one of several \
different types depending on the value of another port. Backward inference still works on the chosen type,
unlike with ComputedType (because CT strongly evaluates its body, replacing all NetPaths before unifying).
SwitchedType[port$,rules$$,basetype$] specifies that the base type of the associated port should be \
basetype$ (the default base type is TypeT)."

SwitchedType[port_, rules___Rule] := 
	TypeReplace[port, {rules, _ -> TypeT}];

SwitchedType[port_, rules___Rule, fallback:Except[_Rule]] := 
	TypeReplace[port, {rules, _ -> fallback}];

_SwitchedType := $Unreachable;


PackageScope["RawComputedType"]

SetUsage @ "
RawComputedType[expr$, trigger$] will evaluate expr$ whenever trigger$ yields True.
* Unlike ComputedType, dependencies are purely based on what is in expr$, and the condition is \
only given by trigger$, whereas with ComputedType all dependent types must first be concrete."

SetAttributes[RawComputedType, HoldAll];


PackageScope["TypeReplace"]

SetUsage @ "
TypeReplace[path$, rules$] is a 'macro' that evaluates before any other type inference rules \
evaluate, and can will expand to one of the given RHS forms based on the value of path$, which \
should be a port or parameter in the layer."


(*****************************************************************************)
(********************************** ALIASES **********************************)
(*****************************************************************************)


(* Type aliases provide a mechanism to declare new types that are really just
names for other types, e.g. ColorSpaceT = EnumT[{"RGB", "HSV", ...}]. It's up
to APIs like Coerce and so on to hook into this mechanism. *)

(* This is an underused mechanism *)

PackageScope["$TypeAliasPatterns"]

$TypeAliasPatterns = Alternatives[];
$InvalidTypeAlias = _ :> Panic["InvalidTypeAliasUsage"];
$TypeAliasRules = {$InvalidTypeAlias};

PackageScope["DeclareTypeAlias"]

DeclareTypeAlias[rule:RuleDelayed[new_, old_]] := (
	$TypeAliasRules[[-1]] = rule;
	AppendTo[$TypeAliasRules, $InvalidTypeAlias];
	AppendTo[$TypeAliasPatterns, new];
);
DeclareTypeAlias[_] := Panic["InvalidAlias"];

PackageScope["TypeAliasP"]
PackageScope["ResolveAlias"]

TypeAliasP := _ ? (MatchQ[$TypeAliasPatterns]);
ResolveAlias[p_] := Replace[p, $TypeAliasRules];


(*****************************************************************************)
(************************** CONVOLUTION-RELATED TYPES ************************)
(*****************************************************************************)


PackageScope["RepeatedInteger"]

SetUsage @ "
RepeatedInteger[n$] is an integer repeated as many times as contextually necessary."


PackageScope["ArbSizeListT"]

SetUsage @ "
ArbSizeListT[rank$, def$] is a shortcut for a fairly complex NormalizedT that handles kernel, padding, \
etc. sizes for conv and pool layers. It remains symbolic if not given as a list and will later \
expand to the right rank. 
* rank$ should be the parameter that actaully stores the rank.
* def$ is an integer to be used by default (repeated contextually), or None."

ArbSizeListT[rank_, type_, def_] := NormalizedT[ListT[rank, type], ToIntTuple, RepeatedInteger[def]];
ArbSizeListT[rank_, type_, None] := NormalizedT[ListT[rank, type], ToIntTuple];


PackageScope["PaddingSizeT"]

SetUsage @ "
PaddingSizeT[rank$, def$] is a shortcut for a NormalizedT used to represent padding specifications \
in the form {{before$1, after$1}, {before$2, after$2}, $$}. It remains symbolic if not given as a \
list and will later expand to the right rank. 
* rank$ should be the parameter that actaully stores the rank.
* def$ is an integer to be used by default (repeated contextually), or None."

PaddingSizeT[rank_, def_] := NormalizedT[ListT[rank, ListT[2, NaturalT]], ToIntMatrix, def]


PackageScope["InterleavingSwitchedT"]
Clear[InterleavingSwitchedT];
InterleavingSwitchedT[switch_, channels_, dims_, type_:RealT] :=
	SwitchedType[switch,
		False -> ChannelT[channels, TensorT[dims, type]],
		True -> TensorT[dims, VectorT[channels, type]],
		TensorT[{SizeT}, If[type === RealT, RealTensorT, AnyTensorT]]
	];


PackageScope["PoolingFunctionT"]

SetUsage @ "
PoolingFunctionT represents a pooling function for PoolingLayer (one of Mean, Max, Total)."

$PoolingFunctions = {Mean, Max, Total}
DeclareTypeAlias[PoolingFunctionT :> EnumT[$PoolingFunctions]];


(*****************************************************************************)
(************************** OTHER SPECIALIZED TYPES **************************)
(*****************************************************************************)


PackageScope["LevelSpecT"]

LevelSpecT[default_, allowEmpty_:False, enums_:{}] := ValidatedParameterT[CheckLevelSpec[General, #, allowEmpty, enums]&, default]

SetUsage @ "
LevelSpecT[default$, allowEmpty$, stringEnums$] represents an ValidatedParameter that should be a list of levels.
* default$ is the default value if the user does not specify.
* allowEmpty$ is whether the level set can be empty.
* enums$ is a list of named level specifications that are allowed."


PackageScope["EncoderDimensionsT"]

EncoderDimensionsT[] := NormalizedT[SizeListT[], toEncoderDims, {}]
toEncoderDims[dims_] := Replace[ToDimsList @ dims, {$Failed :> dims, res_ :> $smuggle[res]}];


PackageScope["StartsWithT"]

(* StartsWithT is very specialized, and just used for NetMapThreadOperator *)


PackageExport["NetT"]

SetUsage @ "
NetT[<|'inname$'->intype$,$$|>, <|'outname$'->outtype$,$$|>] represents a net that has a specific set of inputs \
and outputs of specific types."


PackageScope["StandaloneFunctionT"]

SetUsage @ "
StandaloneFunctionT represents a user-provided function, but one that is checked to ensure it is self-contained and will evaluate \
properly after being serialized and deserialized. A warning is issued if this is likely to not be the case."

StandaloneFunctionT = ValidatedParameterT[checkFunctionIsSafe]

General::extfwarn = "Specified function `` appears to require definitions of external symbols (``). Be aware that the \
definitions and values of these symbols will not be retained if the net is saved using Export, Put, or DumpSave."

checkFunctionIsSafe[expr_] := Scope[
	syms = DeepCases[expr, s_Symbol /; Context[s] =!= "System`" && unsafeSymbolQ[s] :> HoldForm[s]];
	If[syms =!= {}, InheritedMessage["extfwarn", Short @ Shallow[expr, 3], Row[DeleteDuplicates @ syms, ", "]]];
	expr
];

(* we can depend on these being loaded *)
$safeContexts = {"System`", "Developer`", "Language`", "Internal`", "GeneralUtilites`", "NeuralNetworks`", "MXNetLink`", "MachineLearning`"};
SetHoldFirst[unsafeSymbolQ];
unsafeSymbolQ[s_] := System`Private`HasAnyEvaluationsQ[s] && !System`Private`HasDownCodeQ[s] && 
	!StringStartsQ[Context[s], $safeContexts];


PackageExport["UnaryElementwiseFunctionT"]

SetUsage @ "
UnaryElementwiseFunctionT is actually a shortcut for a ValidatedParameterT that checks the input is one \
of the $PrimitiveUnaryElementwiseFunctions, or a one-input ScalarFunction."

UnaryElementwiseFunctionT = ValidatedParameterT[ToUnaryElementwiseFunction]


PackageExport["NAryElementwiseFunctionT"]

SetUsage @ "
NAryElementwiseFunctionT is actually a shortcut for a ValidatedParameterT that checks the \
input is one of the $PrimitiveBinaryElementwiseFunctions, or a two-input ScalarFunction."

NAryElementwiseFunctionT = ValidatedParameterT[ToNAryElementwiseFunction]


PackageExport["DistributionT"]

SetUsage @ "
DistributionT represents a univariate distribution."


(*****************************************************************************)
(************************** IMAGE RELATED TYPES ******************************)
(*****************************************************************************)


PackageExport["ImageT"]

SetUsage @ "
ImageT[] represents an image.
ImageT[dims$] represents an image of dimensions dims$.
ImageT[dims$,cspace$] represents an image with color space cspace$."

ImageT[] := ImageT[SizeListT[2], ColorSpaceT]; 


PackageExport["Image3DT"]

SetUsage @ "
Image3DT[] represents a 3D image.
Image3DT[dims$] represents a 3D image of dimensions dims$.
Image3DT[dims$, cspace$] represents a 3D image with color space cspace$."

Image3DT[] := Image3DT[SizeListT[3], ColorSpaceT]; 


PackageExport["ColorSpaceT"]

SetUsage @ "
ColorSpaceT represents a color space string ('RGB', 'Grayscale', etc.)."


PackageScope["$ColorSpaces"]

$ColorSpaces = {"Grayscale", "RGB", "CMYK", "HSB", "XYZ", "LAB", "LCH", "LUV", Automatic};
DeclareTypeAlias[ColorSpaceT :> EnumT[$ColorSpaces]];


(*****************************************************************************)
(***************************** AUDIO RELATED TYPES ***************************)
(*****************************************************************************)


PackageExport["AudioNormalizationT"]

SetUsage @ "
AudioNormalizationT is actually a shortcut for a ValidatedParameterT that checks the input \
is a valid $Normalization parameter for Audio encoders."

AudioNormalizationT = ValidatedParameterT[checkAudioNormalization, None]


PackageExport["WindowFunctionT"]

SetUsage @ "
WindowFunctionT represents a window function (a list, a real valued function, None or Automatic)."

WindowFunctionT = Defaulting[EitherT[{MatchT[Automatic], MatchT[None], FunctionT, ListT[ScalarT]}], Automatic]


PackageScope["AudioTargetLengthT"]

AudioTargetLengthT = Defaulting[EitherT[{MatchT[All], PosIntegerT, DurationT}], All];

SetUsage @ "
AudioTargetLengthT represents a target length (All, a positive integer or a duration."


PackageScope["InternalAudioTargetLengthT"]

SetUsage @ "
InternalAudioTargetLengthT is a switched type that is variable length if the input is all an a positive \
integer in the other cases."

InternalAudioTargetLengthT[tlenvar_] := SwitchedType[tlenvar, All -> LengthVar[], _Integer -> tlenvar, HoldPattern[_Quantity] -> PosIntegerT]


(*****************************************************************************)
(**************************** DEPRECTATED TYPES ******************************)
(*****************************************************************************)

PackageExport["EncodedType"]
PackageExport["DecodedType"]

SetUsage @ "
EncodedType[encoder$,type$] is deprecated."

SetUsage @ "
DecodedType[decoder$,type$] is deprecated."


PackageScope["SequenceOf"]


