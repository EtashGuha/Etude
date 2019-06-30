Output: SequenceT[$$TargetLength, IndexIntegerT[$Count]]

Parameters:
	$Tokens: ValidatedParameterT[ParseTokensSpec, "English"]
	$SplitPattern: ValidatedParameterT[toSplitPattern, $DefaultTokenSplitPattern]
	$IgnoreCase: Defaulting[BooleanT, True]
	$TargetLength: Defaulting[EitherT[{MatchT[All], PosIntegerT}], All]
	$Count: ComputedType[SizeT, Length[StripVP @ $Tokens]+1]
	$$TargetLength: SwitchedType[$TargetLength,
		All -> LengthVar[],
		_Integer -> $TargetLength
	]

MaxArgCount: 2

AcceptsLists: Function[StripVP[#SplitPattern] === None]

AllowBypass: Function[True]

ToEncoderFunction: Function @ Scope[
	casefunc = If[#IgnoreCase, ToLowerCase, Identity];
	assoc = MakeTokenDictionary @ casefunc @ Normal @ StripVP @ #Tokens;
	func = TokenEncode[assoc, casefunc, StripVP @ #SplitPattern] /* ChopOrPadSequence[TFirstDim[#2], #Count];
	naType = countIntMinType[#Count];
	Map[func] /* toNAList[naType]
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

MLType: Function["Text"]

EncoderToDecoder: Function[
	{"Tokens", StripVP @ #Tokens, IgnoreCase -> #IgnoreCase}
]

Kind: "string"

toSplitPattern[e_] := If[StringPatternOrRulesQ[e] || e === None, e, FailCoder["`` is not a valid value for \"SplitPattern\". It should be None, a pattern or list of patterns or rules", e]];
