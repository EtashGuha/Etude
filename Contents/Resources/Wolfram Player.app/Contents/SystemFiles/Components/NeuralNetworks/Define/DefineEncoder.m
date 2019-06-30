Package["NeuralNetworks`"]



PackageExport["$EncoderData"]

$EncoderData = Association[];


PackageExport["NetEncoder"]


PackageScope["CoderKind"]

CoderKind[HoldPattern @ NetEncoder[name_, assoc_, type_]] := 
	Replace[$EncoderData[name, "Kind"], {
		Automatic :> ToLowerCase[Decamel @ name], 
		f_Function :> f[assoc, type]
	}];


PackageScope["CoderData"]

CoderData[HoldPattern @ NetEncoder[_, data_, _]] := data;


PackageScope["CoderName"]

CoderName[HoldPattern @ NetEncoder[name_, _, _]] := name;


PackageExport["CoderType"]

CoderType[HoldPattern @ NetEncoder[_, _, type_]] := type;

PackageScope["EncoderInputPattern"]

EncoderInputPattern[HoldPattern @ NetEncoder[name_, data_, _]] :=
	$EncoderData[name, "InputPattern"][data];

PackageScope["SequenceCoderQ"]

SequenceCoderQ[HoldPattern @ NetEncoder[name_, _, type_]] := DynamicDimsQ[type];


PackageScope["AcceptsListsQ"]

(* All by-passable encoders accepts lists, because bypass has to work with packed arrays *)
AcceptsListsQ[enc:HoldPattern @ NetEncoder[name_, data_, _]] := System`Private`ValidQ[enc] || $EncoderData[name]["AllowBypass"][data];


PackageScope["EncoderTypeRandomInstance"]

EncoderTypeRandomInstance[HoldPattern @ NetEncoder[name_, assoc_, type_]] := 
	$EncoderData[name, "TypeRandomInstance"][assoc, type];


PackageExport["DefineEncoder"]

DefineEncoder[name_, assoc_] := CatchFailure @ Scope[

	assoc = CoerceParam[DefineEncoder, assoc, EncoderDefinitionT];

	DesugarTypeDefinitions[assoc, {"Parameters", "Output"}];
	SetupArgCounts[assoc];

	ComposeTo[assoc["Upgraders"], sortUpgradeRules];

	$EncoderData[name] = assoc;
];

EncoderDefinitionT = ObjectDefinitionT[
	"Output" -> 					TypeExpressionT,
	"Parameters" -> 				AssocT[StringT, TypeExpressionT],
	"ParameterCoercions" -> 		AssocT[StringT, ExpressionT],
	"ParameterDefaults" -> 			AssocT[StringT, ExpressionT],
	"InferenceRules" -> 			ListT @ RuleT[MatchT[_NetPath], TypeExpressionT],
	"PostInferenceFunction" -> 		ExpressionT,
	"ArgumentRewriter" -> 			Defaulting[FunctionT, None],
	"ToEncoderFunction" -> 			FunctionT,
	"AcceptsLists" -> 				Defaulting[FunctionT, False&],
	"MLType" -> 					FunctionT,
	"InputPattern" -> 				Defaulting[FunctionT, _&],
	"TypeRandomInstance" -> 		Defaulting[FunctionT, Panic["NoTypeRandomInstanceDefined"]&],
	"Kind" -> 						Defaulting[EitherT[{StringT, FunctionT}], Automatic],
	"EncoderToDecoder" -> 			Defaulting[FunctionT, None&],
	"AllowBypass" ->				Defaulting[FunctionT, False&],
	"HiddenFields" -> 				ListT[StringT],
	"SequenceLengthFunction" ->		Defaulting[FunctionT, None],
	"Upgraders" -> 					Defaulting @ ListT @ RuleT[StringT, ExpressionT]
];

(enc_NetEncoder ? System`Private`HoldEntryQ) := UseMacros @ RuleCondition @ CatchFailureAsMessage[NetEncoder, CreateEncoder[enc]];
(enc_NetEncoder ? System`Private`NoEntryQ)[input_] := CatchFailureAsMessage @ EncoderApply[enc, input];
NetEncoder /: Normal[(enc_NetEncoder ? System`Private`NoEntryQ)] := CoderExtract[enc, All];
NetEncoder /: Part[(enc_NetEncoder ? System`Private`NoEntryQ), part_] := CoderExtract[enc, part];

RunInitializationCode[
	DefineCustomBoxes[NetEncoder, enc_NetEncoder ? System`Private`HoldNoEntryQ :> MakeEncoderBoxes[enc], "UseUpValues" -> False];
	Format[enc_NetEncoder ? System`Private`HoldNoEntryQ, OutputForm] := CoderFormString[enc];
];

MakeEncoderBoxes[HoldPattern @ NetEncoder[kind_, assoc_, type_]] := Scope[
	hidden = $EncoderData[kind, "HiddenFields"];
	entries = Append["Output" -> type] @ Prepend["Type" -> kind] @ KeyDrop[hidden] @ Map[ReplaceCoderInfoWithDummies] @ assoc;
	OptimizedArrangeSummaryBox[
		NetEncoder, Nothing,
		fmtEntries[<|"Type" -> kind, "Output" -> type|>],
		fmtEntries[entries], 
		True
	]
];

MakeEncoderBoxes[_] := $Failed;