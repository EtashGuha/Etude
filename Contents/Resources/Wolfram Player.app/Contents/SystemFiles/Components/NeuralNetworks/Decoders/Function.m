Input: RealTensorT

Parameters:
	$Function: StandaloneFunctionT
	$Properties: Defaulting[
		Nullable @ ListT[SizeT, EitherT[{StringT, MatchT[{_String, ___}]}]], 
		None
	]

Upgraders: {
	"11.3.9" -> MapAt[ValidatedParameter[preNormal @ StripVP @ #]&, "Function"]
	(* ^ prerelease Function decoders weren't built for RawArrays *)
}

AvailableProperties: Function[#Properties]

PosArgCount: 1

ArrayDepth: None

preNormal[f_][data_, args___] := f[Normal[data], args];

PostInferenceFunction: Function @ Scope[
	argCount = GuessArgumentCount @ StripVP @ $Function;
	hasProps = $Properties =!= None;
	If[argCount =!= Indeterminate,
		If[hasProps && argCount =!= 2, FailValidation[NetDecoder, "properties were declared but specified function does not take exactly two arguments."]];
		If[!hasProps && argCount =!= 1, FailValidation[NetDecoder, "specified function does not take exactly one argument."]];	
	];
]

ToDecoderFunction: Function @ Scope[
	func = StripVP @ #Function;
	If[#Properties =!= None, func = curry[func, First @ #Properties]];
	Map[func]
]

ToPropertyDecoderFunction: Function @ Scope[
	func = StripVP @ #Function;
	If[!MatchQ[#2, Alternatives @@ #Properties], $Failed,
		Map @ curry[func, #2]
	]
]

curry[f_, a_] := f[#, a]&;

Kind: "expression"
