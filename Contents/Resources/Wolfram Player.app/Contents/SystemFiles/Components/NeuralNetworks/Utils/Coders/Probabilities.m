Package["NeuralNetworks`"]

PackageScope["CheckDecoderProbs"]

(* Utilities for sanity checks of probability vectors used by NetDecoders.
   We might want to perform all the checks (including probs summing to 1) in
   the future and disable them automatically if the decoder input comes from 
   a SoftmaxLayer *)

CheckDecoderProbs[checks_:{Min}] := Scope[
	maybeCheckMin = If[MemberQ[checks, Min],
		If[AnyTrue[#1, Negative@*Min], DecodeFail["one or more input probability values are negative"], #1]&,
		Identity
	];
	maybeCheckSum = If[MemberQ[checks, Total],
		If[AnyTrue[#1, Function[x, Min[Total[x, {-1}]] < 10^-6.]], DecodeFail["one or more input probability vectors sums to zero"], #1]&,
		Identity
	];
	maybeCheckMax = If[MemberQ[checks, Max],
		If[AnyTrue[#1, Function[x, Max[x] > 1]], DecodeFail["one or more input probability values are larger than 1"], #1]&,
		Identity
	];
	maybeCheckMax @* maybeCheckSum @* maybeCheckMin
]