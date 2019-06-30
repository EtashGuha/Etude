Output: SwitchedType[$OutputForm,
	"Index" -> SequenceT[$$TargetLength, IndexIntegerT[$Count]], 
	"UnitVector" -> SequenceT[$$TargetLength, VectorT[$Count]]
]

Parameters:
	$Encoding: ValidatedParameterT[ParseCharactersSpec, Automatic]
	$OutputForm: Defaulting[EnumT[{"Index", "UnitVector"}], "Index"]
	$IgnoreCase: Defaulting[BooleanT, False]
	$Count: ComputedType[SizeT, 
		Max[
			MXNetLink`CharacterEncodingDataSize @ StripVP @ $Encoding,
			Replace[$$Padding, Automatic -> 0]
		],
		{$Encoding}
	]
	$TargetLength: Defaulting[EitherT[{MatchT[All], PosIntegerT}], All]
	$$TargetLength: SwitchedType[$TargetLength,
		All -> LengthVar[],
		_Integer -> $TargetLength
	]
	$$Padding: Defaulting[EitherT[{MatchT[Automatic], PosIntegerT}], Automatic]

MaxArgCount: 2

ToEncoderFunction: Function @ Scope[
	postFunc = If[#OutputForm === "Index", Identity, makeOneHotLookupFunction[#Count]];

	encoding = StripVP[#Encoding];
	encoderFunc = MXNetLink`ToCharacterEncodingFunction[encoding, charEncoderFail];

	padCode = Replace[
		#$Padding,
		Automatic -> Replace[MXNetLink`CharacterEncodingEOSCode[encoding], None -> #Count]
	];

	naType = If[#OutputForm === "Index", countIntMinType[#Count], "UnsignedInteger8"];

	checkStringList /* Map[encoderFunc /* ChopOrPadSequence[TFirstDim[#2], padCode] /* postFunc] /* toNAList[naType]
]

charEncoderFail[] := EncodeFail["input string contained unkown characters and the encoder doesn't contain _ in the character list"];

TypeRandomInstance: Function[
	StringJoin @ RandomChoice[
		Characters @ MXNetLink`CharacterEncodingAlphabet[StripVP @ #Encoding], 
		Replace[#$TargetLength, LengthVar[___] :> RandomInteger[{5,25}]]
	]
]

AllowBypass: Function[True]

MLType: Function["Text"]

EncoderToDecoder: Function[
	{"Characters", StripVP @ #Encoding, IgnoreCase -> #IgnoreCase}
]

Kind: "string"

Upgraders: {
	"12.0.6" -> Append["$Padding" -> Automatic]
}
