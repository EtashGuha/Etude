Output: SequenceT[$$InternalTargetLength, IndexIntegerT[$Count]]

Parameters:
	$Tokens: ValidatedParameterT[ParseBPETokensOrVocabSpec[#, "token"]&]
	$Vocabulary: ValidatedParameterT[ParseBPETokensOrVocabSpec[#, "vocabulary"]&, None]
	$Count: ComputedType[SizeT, 
		Max[Length[First @ $Tokens], Replace[$$Padding, Automatic -> 0]],
		{$Tokens}
	]
	$IncludeTerminalTokens: Defaulting[EnumT[{False, True, {True, False}, {False, True}}], False]
	$IgnoreCase: Defaulting[BooleanT, False]
	$WhitespacePadding: Defaulting[EnumT[{Left, Right, None}], None]
	$CombineWhitespace: Defaulting[BooleanT, False]
	$UnicodeNormalization: Defaulting[EnumT[{None, "NFKC", "ModifiedNFKC"}], None]  (* Naming it $UTF8Normalization throws an error *)
	$TargetLength: Defaulting[EitherT[{MatchT[All], PosIntegerT}], All]
	$$PreMergeWhitespace: Defaulting[EnumT[{Left, Right, None}], None]
	$$InternalTargetLength: SwitchedType[$TargetLength,
		All -> LengthVar[],
		_Integer -> $TargetLength
	]
	$$Padding: Defaulting[EitherT[{MatchT[Automatic], PosIntegerT}], Automatic]

MaxArgCount: 2

ArgumentRewriter: constructor

constructor[{pretrained_Association, opts___Rule}] := Scope[
	If[!MatchQ[pretrained, KeyValuePattern[{"ModelPath" -> Alternatives[_?StringQ, File[_?StringQ]]}]],
		FailCoder["Pre-trained model specification is not an Association, or doesn't specify a Model path."]
	];
	If[!MatchQ[Lookup[pretrained, "VocabularyPath", ""], _?StringQ | File[_?StringQ]],
		FailCoder["Vocabulary path must be a string."]
	];
	If[!MatchQ[Lookup[pretrained, "VocabularyThreshold", None], _?NumberQ | None],
		FailCoder["Vocabulary threshold must be either a number or None."]
	];
	constructFromSentencePiece[
		pretrained,
		Sequence @@ DeleteCases[{opts}, HoldPattern["PreTrainedModel" -> pre_]]
	]
];
constructFromSentencePiece[modelAssoc_, opts___Rule] := Scope[ 
	(* Validate assoc *)
	modelPath = Lookup[modelAssoc, "ModelPath"];
	vocabPath = Lookup[modelAssoc, "VocabularyPath", None];
	vocabThreshold = Replace[Lookup[modelAssoc, "VocabularyThreshold", None], None -> -Infinity];
	If[FailureQ[modelPath = FindFile @ modelPath], FailCoder["Model file not found."]];
	If[vocabPath =!= None && FailureQ[vocabPath = FindFile @ vocabPath], 
		FailCoder["Vocabulary file not found."]
	]; 
	If[MatchQ[vocabPath, None] && NumberQ[vocabThreshold], 
		FailCoder["Option VocabularyThreshold was set but not vocabulary was specified."]
	];
	(* Import *)
	{proto, vocab} = NumericArrayUtilities`ImportSentencePieceModel[modelPath, vocabPath];
	If[FailureQ[proto], invalidProto[]];
	If[FailureQ[vocab], invalidVocab[]];
	(* Check model algorithm *)
	modelType = Lookup[
		Lookup[proto, "trainer_spec", invalidProto[]],
		"model_type",
		invalidProto[]
	];
	If[modelType =!= "BPE", invalidType[modelType]];
	(* Tokens and Vocabulary *)
	{pieces, normSpec} = Lookup[proto, {"pieces", "normalizer_spec"}, invalidProto[]];
	rawTokens = Lookup[pieces, "piece", invalidProto[]];
	vocab = Replace[
		StringSplit[StringSplit[vocab, "\n"], "\t"],
		{
			Condition[
				{token_, score_}, 
				StringMatchQ[score, NumberString] && TrueQ[ToExpression[score] >= vocabThreshold]
			] :> token,
			Condition[{token_, score_}, StringMatchQ[score, NumberString]] :> Nothing,
			_ :> invalidVocab[]
		},
		{1}
	];
	{tokens, vocab} = Replace[ 
		StringReplace[#, "\:2581" -> " "], 
		(* Key "Replacements" is undocumented and is exposed to control this step.
		   The replacement "<unk> -> _ is mandatory though. *)
		Lookup[modelAssoc, 
			"Replacements", 
			{"<unk>" -> _, "<s>" -> StartOfString, "</s>" -> EndOfString}
		], 
		{1} 
	]& /@ {rawTokens, vocab}; 
	(* IgnoreCase and Normalization *)
	normName = Lookup[normSpec, "name", invalidProto[]];
	caseFolding = StringContainsQ[normName, "_cf"];
	normName = Replace[StringTrim[normName, "_cf"],
		{
			"nmt_nfkc" -> "ModifiedNFKC", 
			"nfkc" -> "NFKC", 
			"identity" -> None, 
			"user_defined" :> unsuppNorm[],
			_ :> invalidProto[]
		} 	
	];
	(* Putting opts at the end allows to override settings coming from the file *)
	{
		tokens,
		"Vocabulary" -> Replace[vocab, {} -> None],
		"IgnoreCase" -> caseFolding,
		"WhitespacePadding" -> Replace[normSpec["add_dummy_prefix"], {True -> Left, False -> None}],
		"CombineWhitespace" -> normSpec["remove_extra_whitespaces"],
		"UnicodeNormalization" -> normName,
		"TargetLength" -> All,
		"$PreMergeWhitespace" -> None,
		opts
	}
];
constructor[e_] := e;

invalidProto[] := FailCoder["Invalid sentencepiece model file."];
invalidVocab[] := FailCoder["Invalid vocabulary file."];
unsuppNorm[] := FailCoder["Custom Unicode normalization rules are not currently supported."];
invalidType[type_] := FailCoder["Sentencepiece model must be of type BPE, but was `` instead.", type];

PostInferenceFunction: Function[
	If[$IgnoreCase === True && $UnicodeNormalization === None,
		FailCoder["The combination IgnoreCase -> True and UnicodeNormalization -> None is not currently supported."]	
	];
	start = $IncludeTerminalTokens === {True, False} || $IncludeTerminalTokens === True;
	If[ start && !MemberQ[StripVP[$Tokens], StartOfString],
		FailCoder["Option IncludeTerminalTokens was set to include the start of string token, but no StartOfString symbol was found in the token list."]
	];
	end = $IncludeTerminalTokens === {False, True} || $IncludeTerminalTokens === True;
	If[end && !MemberQ[StripVP[$Tokens], EndOfString],
		FailCoder["Option IncludeTerminalTokens was set to include the end of string token, but no EndOfString symbol was found in the token list."]
	];
]

ToEncoderFunction: Function @ Scope[
	processor = NumericArrayUtilities`CreateBPEProcessor[
			StripVP[#Tokens], 
			StripVP[#Vocabulary],
			"AddDummyPrefix" -> Replace[#WhitespacePadding, {Left -> True, _ -> False}],
			"AddDummyPostfix" -> Replace[#WhitespacePadding, {Right -> True, _ -> False}],
			"InsertBOS" -> MatchQ[#IncludeTerminalTokens, True | {True, False}],
			"InsertEOS" -> MatchQ[#IncludeTerminalTokens, True | {False, True}],
			"RemoveExtraWhitespaces" -> #CombineWhitespace,
			"PreMergeWhitespace" -> #$PreMergeWhitespace,
			"CaseFolding" -> #IgnoreCase,
			"UTF8Normalization" -> #UnicodeNormalization
	];
	padCode = Replace[
		#$Padding,
		Automatic -> Replace[
			Position[StripVP[#Tokens], EndOfString], 
			{{} -> #Count, p_ :> p[[1, 1]]}
		]
	];
	naType = countIntMinType[#Count];

	toNAList[naType] @* Map[ChopOrPadSequence[TFirstDim[#2], padCode]] @* encodeBPE[processor] @* checkStringList
]

TypeRandomInstance: Function[
	StringRiffle[RandomChoice[
		cachedNormal @ StripVP @ #Tokens,
		Replace[#$TargetLength, LengthVar[___] :> RandomInteger[{5,25}]]
	], " "]
]

(* otherwise "RandomData" generation is super slow *)
$randCache = <||>;
cachedNormal[e_] := CacheTo[$randCache, Hash[e], Normal[e]];

encodeFail[] := EncodeFail["input contained invalid characters"];

encodeBPE[processor_][input_] := NumericArrayUtilities`EncodeBPE[processor, input];


(* AllowBypass: Function[True] *)
Kind: "string"
MLType: Function["Text"]

EncoderToDecoder: Function @ {"BPESubwordTokens", 
	StripVP[#Tokens], 
	"Count" -> #Count, 
	"WhitespaceTrimming" -> #WhitespacePadding
}

Upgraders: {
	"12.0.6" -> Append["$Padding" -> Automatic]
}
