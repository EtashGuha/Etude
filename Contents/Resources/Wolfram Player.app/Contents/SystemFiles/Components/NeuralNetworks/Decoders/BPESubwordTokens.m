Input: SwitchedType[$InputDepth,
	1 -> VectorT[$Count], 
	2 -> SequenceT[LengthVar[], VectorT[$Count]]
]

Parameters:
	$Tokens: ValidatedParameterT[ParseBPETokensOrVocabSpec[#, "token"]&]
	$Count: ComputedType[SizeT, Length[First @ $Tokens], {$Tokens}]
	$WhitespaceTrimming: Defaulting[EnumT[{Left, Right, None}], None]
	$InputDepth: Defaulting[EnumT[{1, 2}], 2]

ArrayDepth: Function[#InputDepth]

MaxArgCount: 1

(* TODO: "Tokens" decoder should be able to absorb "BPESubwordTokens" decoder in the future. 
   It would need the "WhitespaceTrimming" Parameter + a parameter which specifies what to
   riffle between the tokens ("Tokens" now uses whitespace, BPE should use empty string). We
   should carefully check if this merge is really possible. If yes, EncoderToDecoder for BPE
   encoder should return a Tokens decoder, and BPE decoder should be deprecated *)

(* See the branch refactor/TokenBPEUnification *)

ToDecoderFunction: Function @ ModuleScope[
	trim = Switch[#WhitespaceTrimming,
		Left,  StringTrim[#, StartOfString ~~ " "]&,
		Right, StringTrim[#, " " ~~ EndOfString]&,
		None,  Identity
	];
	processor = NumericArrayUtilities`CreateBPEProcessor[StripVP[#Tokens], None];
	decoder = NumericArrayUtilities`DecodeBPE[processor, #]&;
	Map[Flatten[NumericArrayUtilities`PartialOrdering[Normal @ #, -1]]&] /* decoder /* trim
]

Kind: "string"

DecoderToEncoder: Function[
	{"BPESubwordTokens", StripVP[#Tokens], "Count" -> #Count, "WhitespacePadding" -> #WhitespaceTrimming}
]
