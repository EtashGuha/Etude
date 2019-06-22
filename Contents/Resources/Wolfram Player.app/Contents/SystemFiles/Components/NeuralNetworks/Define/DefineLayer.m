Package["NeuralNetworks`"]



PackageScope["$FromMXNetName"]

$FromMXNetName = Association[];


PackageExport["$LayerData"]

$LayerData = Association[];


PackageExport["DefineLayer"]

General::netdefnoparent = "There is no layer called `` to inherit from (yet)."
General::netdefmxdollar = "Writer should not contain any $-prefixed symbols, and should use #SymbolName instead."
General::netdefruninfsf = "Can't specify both a shapefunction and runtime inference rules for a multiport layer."

$OperatorVTable := $OperatorVTable = Join[$VTable["GenericLayer"], $VTable["GenericOperator"]];

DefineLayer[name_, assoc_Association] := CatchFailure @ Scope[
	
	symbolName = name <> Lookup[assoc, "Suffix", "Layer"];

	If[NameQ["System`" <> symbolName],
		symbolName = "System`" <> symbolName,
		symbolName = "NeuralNetworks`" <> symbolName
	];
	symbol = Symbol[symbolName];
	Compose[Unprotect, symbol];

	AppendTo[$NetHeads, symbol]; 

	If[StringQ[parent = assoc["InheritsFrom"]],
		If[!AssociationQ[parentData = $LayerData[parent]],
			ThrowFailure["netdefnoparent", parent];
		];
		parentData = parentData /. $TypeToSymbol[parent] -> symbol;
		assoc = Join[KeyDrop[parentData, {"Type", "Symbol", "Tests"}], assoc];
	];

	assoc = CoerceParam[DefineLayer, assoc, LayerDefinitionT] /. 
		With[{symbol = symbol}, HoldPattern[(head:FailValidation|FailConstruction)[reason_String, rest___]] :> head[symbol, reason, rest]];

	(* turn "Input" -> foo into "Inputs" -> <|"Input" -> foo|> *)
	Do[
		If[assoc[key] =!= None,
			assoc[key <> "s"] = <|key -> assoc[key]|>;
		];
		KeyDropFrom[assoc, key];
		,
		{key, {"Input", "Output"}}
	];

	assoc["Type"] = name;
	assoc["Symbol"] = symbol;

	isMultiport = Keys[assoc["Inputs"]] === {$Multiport};
	assoc["IsMultiport"] = isMultiport;

	SetAutomatic[assoc["HasDynamicPorts"], isMultiport];

	KeyAppendTo[$LayerUpgradeData, #1, name -> #2]& @@@ assoc["Upgraders"];

	DesugarTypeDefinitions[assoc, {"Parameters", "Inputs", "Outputs", "Arrays", "States"}];
	SetupArgCounts[assoc];

	UnpackAssociation[assoc, shapeFunction, rankFunction, typeFunction];
	If[shapeFunction =!= None,
		If[assoc["HasDynamicPorts"],
			(* dynamic-port layers require the shape function rules to be produced at 
			inference time because the input paths differ from instance to instance *)
			If[assoc["RuntimeInferenceRules"] =!= None, ThrowFailure["netdefruninfsf"]]; 
			(* ^ could be lifted *)
			assoc["RuntimeInferenceRules"] = With[
				{sym = symbol, rf = rankFunction, sf = shapeFunction, tf = typeFunction}, Function[
				MakeRankAndShapeFunctionRules[
					sym, tf, rf, sf,
					LocalInputPaths[#], LocalOutputPaths[#], #
				]
			]]
		,
			extra = assoc["ExtraShapeFunctionTensors"];
			AppendTo[assoc["InferenceRules"], 
				MakeRankAndShapeFunctionRules[
					symbol, typeFunction, rankFunction, shapeFunction, 
					Join[LocalInputPaths[assoc], extra], LocalOutputPaths[assoc],
					assoc
				]
			]
		]
	];

	If[assoc["ReshapeParams"] === Automatic,
		(* jeromel: This can be source of bugs of NetReshapePart on some layers for which ReshapeParams was not defined
			see https://bugs.wolfram.com/show?number=364631
		*)
		assoc["ReshapeParams"] = Select[Keys[assoc["Parameters"]], StringStartsQ["$"]];
	,
		(* convert from NetPath to simple list of params *)
		assoc["ReshapeParams"] = assoc["ReshapeParams"] /. NetPath[_, param_] :> param;
	];

	If[assoc["MXNet"] =!= None,
		mxname = assoc["MXNet", "Name"];
		$FromMXNetName[mxname] = name;
		aliases = assoc["MXNet", "Aliases"];
		If[ListQ[aliases],
			AssociateTo[$FromMXNetName, Thread[aliases -> name]]
		];
	];

	If[!FreeQ[assoc["Writer"], NetPath], ThrowFailure["netdefmxdollar"]];

	If[assoc["SubNets"] =!= {},
		assoc["IsOperator"] = True;
		assoc["Immutable"] = True;
		$VTable[name] = $OperatorVTable,
		$VTable[name] = $VTable["GenericLayer"];
	];

	SetAutomatic[
		assoc["AllowDynamicDimensions"],
		DynamicDimsQ @ assoc["Parameters"]
	];
	If[assoc["AllowDynamicDimensions"] === False,
		assoc["FinalCheck"] = Tee[assoc["FinalCheck"], CheckNoDynamicInputs];
	];
	$VTable[name, NetFinalCheck] = assoc["FinalCheck"] = resolvePathsInLayerFunc @ assoc["FinalCheck"];

	SetAutomatic[assoc["StateExpanding"], assoc["IsOperator"]];

	If[assoc["Constraints"] =!= None,
		$VTable[name, SowNetConstraints] = resolvePathsInLayerFunc @ assoc["Constraints"];
	];

	If[assoc["HasTrainingBehaviorQ"] =!= None,
		$VTable[name, HasTrainingBehaviorQ] = Key["Parameters"] /* assoc["HasTrainingBehaviorQ"] /* Replace[True :> Throw[True]]];

	If[assoc["SummaryFunction"] =!= None,
		$VTable[name, SummaryForm] = RightComposition[Key["Parameters"], assoc["SummaryFunction"]];
	];

	$LayerData[name] ^= assoc;
	$TypeToSymbol[name] = symbol;
	$SymbolToType[symbol] = name;

	SetupLayerDispatch[
		symbol, name, 
		KeyDrop[assoc["ParameterDefaults"], assoc["PosArgs"]],
		Keys @ assoc["Inputs"], 
		Keys @ assoc["Outputs"], 
		Keys @ assoc["Arrays"]
	];

	assoc
];

LayerDefinitionT = ObjectDefinitionT[
	"Input" -> 						TypeExpressionT,
	"Output" -> 					TypeExpressionT,
	"Inputs" -> 					AssocT[StringT, TypeExpressionT],
	"Outputs" -> 					AssocT[StringT, TypeExpressionT],
	"States" -> 					AssocT[StringT, TypeExpressionT],
	"Arrays" -> 					AssocT[StringT, TypeExpressionT],
	"Parameters" -> 				AssocT[StringT, TypeExpressionT],
	"ParameterCoercions" ->			AssocT[StringT, ExpressionT],
	"ParameterDefaults" -> 			AssocT[StringT, ExpressionT],
	"InferenceRules" -> 			ListT @ RuleT[MatchT[_NetPath], TypeExpressionT],
	"RuntimeInferenceRules" -> 		FunctionT,
	"ShapeFunction" -> 				FunctionT,
	"RankFunction" -> 				FunctionT,
	"TypeFunction" -> 				FunctionT, 
	"ExtraShapeFunctionTensors" ->	ListT @ MatchT[_NetPath],
	"PostInferenceFunction" -> 		ExpressionT,
	"PostConstructionFunction" -> 	ExpressionT,
	"AuxArrays" -> 					ListT[StringT],
	"Writer" -> 					Defaulting[FunctionT, MXWriteDefault],
	"Upgraders" -> 					ListT @ RuleT[StringT, ExpressionT],
	"MXNet" -> {
		"Name" -> 					StringT, 
		"Parameters" -> 			AssocT[StringT, StringT],
		"Arrays" -> 				AssocT[StringT, StringT],
		"Reader" -> 				FunctionT,
		"Writer" -> 				FunctionT,
		"Aliases" -> 				ListT[StringT]
	},
	"SubNets" -> 					Defaulting[EitherT[{ListT @ MatchT[_NetPath], FunctionT}], {}],
	"IsLoss" -> 					BooleanT, 
	"IsOperator" ->					BooleanT,
	"IsMultiport" -> 				BooleanT,
	"InheritsFrom" -> 				StringT, 
	"Suffix" -> 					Defaulting[StringT, "Layer"],
	"SummaryFunction" -> 			FunctionT, 
	"Tests" -> 						ListT @ RuleT[ExpressionT, StringT],
	"AllowDynamicDimensions" -> 	Defaulting[BooleanT, Automatic],	
	"FinalCheck" -> 				FunctionT,
	"ReshapeParams" -> 				Defaulting[ListT[ExpressionT], Automatic],
	"ArgumentRewriter" -> 			Defaulting[FunctionT, None],
	"FusedRNNArray" -> 				Defaulting[FunctionT, None],
	"Constraints" ->				Defaulting[FunctionT, None],
	"StateExpanding" ->				Defaulting[BooleanT, Automatic],
	"Immutable" ->					Defaulting[BooleanT, False],
	"HasDynamicPorts" ->			Defaulting[BooleanT, Automatic],
	"HasTrainingBehaviorQ" ->		Defaulting[FunctionT, None]
];


SetupLayerDispatch[symbol_, name_, params_, ins_, outs_, arrays_] := Scope[
	SetCatchFailureAsMessage[symbol, s_symbol ? System`Private`HoldEntryQ, CreateNet[name, s]];
	If[ins === {$Multiport}, ins = {"Inputs"}];
	opts = Normal @ Map[toOptionValue, KeySelect[params, Not @* StringStartsQ["$"]]] /. RuleDelayed -> Rule;
	Options[symbol] = Join[opts, Thread[Join[arrays, ins, outs] -> Automatic]];
	SetupGenericDispatch[symbol, Length[ins] == 1];
	DefineCustomBoxes[symbol, s_symbol ? System`Private`HoldNoEntryQ :> MakeLayerBoxes[s]];
	Format[s_symbol ? System`Private`HoldNoEntryQ, OutputForm] := LayerOutputForm[s];
];

toOptionValue[sym_Symbol] /; Context[sym] === "System`" := sym;
toOptionValue[sym_] := If[ConcreteParameterQ[sym], sym, FromT[sym]];
toOptionValue[RepeatedInteger[n_]] := n;


resolvePathsInLayerFunc[HoldPattern @ f:Function[var_]] := f /. NetPath[a_, b_] :> Slot[a][b];
resolvePathsInLayerFunc[HoldPattern @ f:Function[var_, _]] := f /. NetPath -> var;
resolvePathsInLayerFunc[t_Tee] := Map[resolvePathsInLayerFunc, t];
resolvePathsInLayerFunc[other_] := other;


PackageScope["NetFinalCheck"]

DeclareMethod[NetFinalCheck];
(* the VTable is populated by DefineLayer based on whether there are any dynamic dims present 
in default parameters *)

PackageScope["CheckNoDynamicInputs"]

CheckNoDynamicInputs[assoc_] /; ContainsVarSequenceQ[Inputs[assoc]] := Scope[
	inputs = StripCoders @ Inputs[assoc];
	rule = First @ Normal @ Select[inputs, DynamicDimsQ];
	symbol = NSymbol[assoc];
	FailValidationWithPath[symbol, $path, 
		"`` of layer has dimensions ``, but dynamic dimensions are not currently supported by ``.",
		FmtPortName[First @ rule, "Inputs"],
		fmtDims @ Last @ rule, symbol
	]
];

PackageScope["LayerOutputForm"]

LayerOutputForm[head_[___]] := SymbolName[head] <> "[<>]";


PackageScope["SetupArgCounts"]

SetHoldFirst[SetupArgCounts];

SetupArgCounts[data_] := Scope[

	params = data["Parameters"];
	keys = Keys[params];

	n = 1; max = 0;
	Do[
		If[!StringStartsQ[key, "$"], max = n++, Break[]],
		{key, keys}
	]; 
	n = 1; min = 0; defs = data["ParameterDefaults"];
	Do[
		If[!KeyExistsQ[defs, key] && !MatchQ[params[key], _ComputedType], min = n++, Break[]],
		{key, keys}
	];
	old = Lookup[data, {"MinArgCount", "MaxArgCount"}];
	min = Min[min, max, old /. Automatic -> Infinity];
	max = Max[min, max, old /. Automatic -> 0];
	SetAutomatic[data["MaxArgCount"], max];
	SetAutomatic[data["MinArgCount"], min];
	SetAutomatic[data["PosArgCount"], data["MaxArgCount"]];
	SetAutomatic[data["PosArgs"], Take[Keys[data["Parameters"]], data["PosArgCount"]]];
];


PackageExport["LayerData"]

LayerData[name_String, parts___] := Slice[parts] @ $LayerData[name];
LayerData[nf_Association, parts___] := Slice[parts] @ nf;
LayerData[parts___] := Slice[parts] @ $LayerData;



PackageExport["LayerInfo"]

LayerInfo[parts___] := PrettyGrid[LayerData[parts]]


PackageScope["ToContainerLayers"]

ToContainerLayers[list_List] := If[
	VectorQ[list, RuleQ], ToContainerLayers[Association[list]],
	NumberedAssociation @ Map[ToLayer, list]
];

General::notstrkey = "All layer keys must be strings."

ToContainerLayers[assoc_Association] := (
	If[!StringVectorQ[Keys[assoc]], 
		If[VectorQ[Keys[assoc], IntegerQ],
			Return @ ToContainerLayers[KeyMap[IntegerString, assoc]]
		];
		(* ^ this codepath doesn't seem to be used anywhere. try remove *)
		ThrowFailure["notstrkey"]
	];
	DeleteCases[Nothing] @ Map[ToLayer, assoc]
);

PackageScope["ToLayer"]

ToLayer := MatchValues[
	i:({___Integer} | _Integer) := NData @ LinearLayer[i];
	list_List := NData @ ThrowOnFailure @ NetChain[list];
	HoldPattern[Alternatives][args___] := NData @ ThrowOnFailure @ NetParallel[args];
	Identity := NData @ ElementwiseLayer[Identity];
	sym_Symbol /; MemberQ[$PrimitiveBinaryElementwiseFunctions, sym] := NData @ ThreadingLayer[sym];
	sym_Symbol /; MemberQ[$PrimitiveUnaryElementwiseFunctions, sym] := NData @ ElementwiseLayer[sym];
	$Raw[e_] := e;
	e_ ? ValidNetQ := RemapLengthVars @ NData[e];
	e_ := ThrowFailure["netinvnodes", e];
];

General::netinvnodes = "`` is not a layer, a net, or a valid specification for one."


PackageScope["RemapLengthVars"]

(* when putting nets into other nets we have to remap the interior length vars to be totally
unique so that re-using a net several times in a bigger net does not introduce false aliasing *)

RemapLengthVars[expr_] /; FreeQ[expr, _LengthVar] := expr;

RemapLengthVars[expr_] := Scope[
	old = Select[UniqueLengthVars[expr], NonNegative[First[#]]&];
	old = Discard[old, NamedLengthVarQ];
	rules = Map[# -> NewLengthVar[]&, old];
	ReplaceAllInsideCoders[expr, rules]
]


PackageScope["GetSubNets"]
PackageScope["SubNetPaths"]

SubNetPaths[assoc_] := Replace[NProperty[assoc, "SubNets"], f_Function :> f[assoc]];
GetSubNets[assoc_] := assoc @@@ SubNetPaths[assoc];


