Input: SwitchedType[$InputDepth,
	1 -> VectorT[$Count], 
	2 -> SequenceT[LengthVar[], VectorT[$Count]]
]

Parameters:
	$Encoding: ValidatedParameterT[ParseCharactersSpec, Automatic]
	$IgnoreCase: Defaulting[BooleanT, False]
	$InputDepth: Defaulting[EnumT[{1, 2}], 2]
	$Count: ComputedType[SizeT, MXNetLink`CharacterEncodingDataSize @ StripVP @ $Encoding]

Upgraders: {
	"11.3.9" -> Function[Append[#, "InputDepth" -> If[TRank[$currentlyUpgradingCoderType] === 1, 1, 2]]]
}

ArrayDepth: Function[#InputDepth]

MaxArgCount: 1

ToDecoderFunction: Function @ Scope[
	enc = StripVP @ #Encoding;

	f = Switch[#InputDepth, 
		1, 
			With[{chars = Replace[Characters @ MXNetLink`CharacterEncodingAlphabet[enc], FromCharacterCode[0] -> "", {1}]},
				Function[in, Part[chars, MaxIndex[in]]]
			],
		2, 
			Function[StringReplace[#, FromCharacterCode[0] -> ""]] @* MXNetLink`ToCharacterDecodingFunction[enc]
	];
	decision[f]
]


decision[func_][probs_List] := decision[func] /@ probs;
decision[func_][probs_NumericArray] := func[Normal[probs]];

Kind: Function @ Switch[#2,
	VectorT[_], "character",
	_, "string"
]

DecoderToEncoder: Function[
	{"Characters", StripVP @ #Encoding, IgnoreCase -> #IgnoreCase}
]

