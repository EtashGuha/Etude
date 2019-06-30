Output: SwitchedType[$OutputForm,
	"Index" -> SequenceT[LengthVar[], IndexIntegerT[248]], 
	"UnitVector" -> SequenceT[LengthVar[], VectorT[248]]
]

Parameters:
	$OutputForm: Defaulting[EnumT[{"Index", "UnitVector"}], "Index"]

MaxArgCount: 1

ToEncoderFunction: Function @ Scope[
	postProc = If[#OutputForm === "UnitVector", Map @ makeOneHotLookupFunction[248], Identity];
	checkStringList /* Function[ToCharacterCode[#, "UTF-8"] + 1] /* postProc /* toNAList["UnsignedInteger8"]
]

MLType: Function["Text"]

Kind: "string"

EncoderToDecoder: Function[
	{"UTF8"}
]
