Package["NeuralNetworks`"]



PackageExport["$DecoderData"]

$DecoderData = Association[];


PackageExport["NetDecoder"]


PackageScope["CoderKind"]

CoderKind[HoldPattern @ NetDecoder[name_, assoc_, type_]] := 
	Replace[$DecoderData[name, "Kind"], {
		Automatic :> ToLowerCase[Decamel @ name], 
		f_Function :> f[assoc, type]
	}];


PackageScope["CoderData"]

CoderData[HoldPattern @ NetDecoder[_, data_, _]] := data;


PackageScope["CoderName"]

CoderName[HoldPattern @ NetDecoder[name_, _, _]] := name;


PackageExport["CoderType"]

CoderType[HoldPattern @ NetDecoder[_, _, type_]] := type;


PackageScope["SequenceCoderQ"]

SequenceCoderQ[HoldPattern @ NetDecoder[name_, _, type_]] := DynamicDimsQ[type];
SequenceCoderQ[_] := False;


PackageExport["DecoderDepth"]

DecoderDepth[HoldPattern @ NetDecoder[name_, assoc_, type_]] := Replace[
	$DecoderData[name, "ArrayDepth"],
	f_Function :> f[assoc, type]
];


PackageExport["DecoderDimensions"]

DecoderDimensions[HoldPattern @ NetDecoder[_, _, type_]] := 
	TDimensions[type] /. {SizeT -> _, _LengthVar -> _};


PackageExport["DefineDecoder"]

DefineDecoder[name_, assoc_] := CatchFailure @ Scope[

	assoc = CoerceParam[DefineDecoder, assoc, DecoderDefinitionT];

	DesugarTypeDefinitions[assoc, {"Parameters", "Input"}];
	SetupArgCounts[assoc];
	
	props = assoc["AvailableProperties"];
	If[ListQ[props],
		props = Replace[props, {prop_String, _} :> prop, {1}];
		AssociateTo[$IsPossibleDecPropQ, Thread[props -> True]];
	];

	ComposeTo[assoc["Upgraders"], sortUpgradeRules];

	$DecoderData[name] = assoc;
];


PackageScope["$IsPossibleDecPropQ"]

$IsPossibleDecPropQ = <||>;


DecoderDefinitionT = ObjectDefinitionT[
	"Input" -> 						TypeExpressionT,
	"Parameters" -> 				AssocT[StringT, TypeExpressionT],
	"ParameterCoercions" -> 		AssocT[StringT, ExpressionT],
	"ParameterDefaults" -> 			AssocT[StringT, ExpressionT],	
	"InferenceRules" -> 			ListT @ RuleT[MatchT[_NetPath], TypeExpressionT],
	"PostInferenceFunction" -> 		ExpressionT,
	"ArgumentRewriter" -> 			Defaulting[FunctionT, None],
	"ToDecoderFunction" -> 			FunctionT,
	"ToPropertyDecoderFunction" -> 	Defaulting[FunctionT, $Failed&],
	"AvailableProperties" -> 		Defaulting[EitherT[{ListT[ExpressionT], FunctionT}], {}],
	"ArrayDepth" -> 				Nullable[EitherT[{IntegerT, FunctionT}]],
	"Kind" -> 						Defaulting[EitherT[{StringT, FunctionT}], Automatic],
	"DecoderToEncoder" -> 			Defaulting[FunctionT, None&],
	"Upgraders" -> 					Defaulting @ ListT @ RuleT[StringT, ExpressionT]
];


(dec_NetDecoder ? System`Private`HoldEntryQ) := UseMacros @ RuleCondition @ CatchFailureAsMessage[NetDecoder, CreateDecoder[dec]];
(dec_NetDecoder ? System`Private`NoEntryQ)[input_] := CatchFailureAsMessage @ DecoderApply[dec, input, Automatic];
(dec_NetDecoder ? System`Private`NoEntryQ)[input_, prop_] := CatchFailureAsMessage @ DecoderApply[dec, input, prop];
NetDecoder /: Normal[(dec_NetDecoder ? System`Private`NoEntryQ)] := CoderExtract[dec, All];
NetDecoder /: Part[(dec_NetDecoder ? System`Private`NoEntryQ), part_] := CoderExtract[dec, part];

RunInitializationCode[
	DefineCustomBoxes[NetDecoder, dec_NetDecoder ? System`Private`HoldNoEntryQ :> MakeDecoderBoxes[dec], "UseUpValues" -> False];
	Format[dec_NetDecoder ? System`Private`HoldNoEntryQ, OutputForm] := CoderFormString[dec];
]

MakeDecoderBoxes[HoldPattern @ NetDecoder[type_, assoc_, input_]] :=
	OptimizedArrangeSummaryBox[
		NetDecoder, Nothing, 
		fmtEntries[<|"Type" -> type, "Input" -> input|>],
		fmtEntries[Prepend["Type" -> type] @ Append["Input" -> input] @ Map[ReplaceCoderInfoWithDummies] @ assoc],
		True
	]

MakeDecoderBoxes[_] := $Failed;