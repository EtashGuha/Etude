Input: SwitchedType[$InputDepth,
	1 -> VectorT[$Count], 
	2 -> SequenceT[LengthVar[], VectorT[$Count]]
]

Parameters:
	$Tokens: ValidatedParameterT[ParseTokensSpec, "English"]
	$IgnoreCase: Defaulting[BooleanT, True]
	$InputDepth: Defaulting[EnumT[{1, 2}], 2]
	$Count: ComputedType[SizeT, Length[StripVP @ $Tokens]+1]

Upgraders: {
	"11.3.9" -> Function[Append[#, "InputDepth" -> If[TRank[$currentlyUpgradingCoderType] === 1, 1, 2]]]
}

MaxArgCount: 1

ArrayDepth: Function[#InputDepth]

ToDecoderFunction: Function @ ModuleScope[
	tokens = Append[Normal @ StripVP @ #Tokens, ""];
	function = Switch[#InputDepth,
		2, UnsafeQuietCheck @ StringJoin @ Riffle[ExtractMaxIndex[tokens, #], " "]&,
		1, UnsafeQuietCheck @ ExtractMaxIndex[tokens, #]&
	];
	Map[function]
]

DecoderToEncoder: Function[
	{"Tokens", StripVP @ #Tokens, IgnoreCase -> #IgnoreCase}
]
