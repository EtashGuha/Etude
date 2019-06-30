Package["NeuralNetworks`"]

PackageScope["ParseBPETokensOrVocabSpec"]

ParseBPETokensOrVocabSpec[spec_List, type_] /; (Length[spec] > 0) := Scope[
	If[!MatchQ[spec, List @ Repeated[_?StringQ | StartOfString | EndOfString | Verbatim[_]]],
		FailCoder["Invalid `` specification.", type]
	];
	If[type === "token" && !MemberQ[spec, Verbatim[_]],
		FailCoder["Token specification must contain the unkown character _."]
	];
	If[MemberQ[spec, ""], 
		FailCoder["`` specification must not contain empty strings.", Capitalize[type]]
	];
	DeleteDuplicates[spec]
]
ParseBPETokensOrVocabSpec[None, "vocabulary"] := None;
ParseBPETokensOrVocabSpec[_, type_] := FailCoder["Invalid `` specification.", type];