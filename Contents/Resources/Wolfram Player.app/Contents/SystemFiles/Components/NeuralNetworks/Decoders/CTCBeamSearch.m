Input: SequenceT[LengthVar[], VectorT[$$InputSize]]

Parameters:
	$Labels: ListT[$$AlphabetSize, ExpressionT]
	$BeamSize: Defaulting[SizeT, 100]
	$$AlphabetSize: SizeT
	$$InputSize: ComputedType[SizeT, $$AlphabetSize + 1]

MinArgCount: 1
MaxArgCount: 1

ArrayDepth: 2

ToDecoderFunction: Function[topPath[#BeamSize, #Labels]]

ToPropertyDecoderFunction: Function @ With[
	{labels = #Labels, beamsize = #BeamSize},
	Replace[#2, {
		"Decoding" :> topPath[beamsize, labels],
		"Decodings" :> nPaths[beamsize, labels, beamsize],
		(* should TopDecodings fails if n > beamsize? *)
		{"TopDecodings", n_Integer} :> nPaths[beamsize, labels, n],
		"NegativeLogLikelihoods" :> negLogLikelihoods[beamsize, labels, beamsize],
		(* should TopNegativeLogLikelihoods fails if n > beamsize? *)
		{"TopNegativeLogLikelihoods", n_Integer} :> negLogLikelihoods[beamsize, labels, n],
		_ :> $Failed
	}]
]

AvailableProperties: {"Decoding", "Decodings", "TopDecodings", "NegativeLogLikelihoods", "TopNegativeLogLikelihoods"}

topPath[beamsize_, labels_][input_ /; (ArrayDepth[input] == 2)] := ModuleScope[
	out = NumericArrayUtilities`CTCBeamSearchDecode[toNumericArray[input], 1, beamsize, False, -1]["Paths"];
	If[out === {},
		{},
		labels[[First[out]]]
	]
];

topPath[beamsize_, labels_][input_] := topPath[beamsize, labels] /@ input;


nPaths[beamsize_, labels_, n_][input_ /; (ArrayDepth[input] == 2)] := Map[
	labels[[#]]&,
	NumericArrayUtilities`CTCBeamSearchDecode[toNumericArray[input], n, beamsize, False, -1]["Paths"]
];

nPaths[beamsize_, labels_, n_][input_] := nPaths[beamsize, labels, n] /@ input;


negLogLikelihoods[beamsize_, labels_, n_][input_ /; (ArrayDepth[input] == 2)] :=
	Function[MapAt[labels[[#]]&,Thread[#Paths -> - #LogLikelihood], {All, 1}]]@NumericArrayUtilities`CTCBeamSearchDecode[toNumericArray[input], n, beamsize, False, -1];

negLogLikelihoods[beamsize_, labels_, n_][input_] := 
	negLogLikelihoods[beamsize, labels, n] /@ input;