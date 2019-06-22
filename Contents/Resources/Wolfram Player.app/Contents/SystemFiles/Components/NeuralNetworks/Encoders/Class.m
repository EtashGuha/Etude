Output: SwitchedType[$OutputForm,
	"Index" -> TensorT[$Dimensions, IndexIntegerT[$Count]],
	"UnitVector" -> TensorT[$Dimensions, VectorT[$Count]]
]

Parameters:
	$Labels: ListT[$Count, ExpressionT]
	$OutputForm: Defaulting[EnumT[{"Index", "UnitVector"}]]
	$Dimensions: EncoderDimensionsT[]
	$Count: SizeT

Upgraders: {
	"11.3.9" -> Append["Dimensions" -> {}]
}

MinArgCount: 1
MaxArgCount: 3

HiddenFields: {"Count"}

AcceptsLists: Function[
	MemberQ[#Labels, _List] || #Dimensions =!= {}
]

AllowBypass: Function[FreeQ[#Labels, _Integer]]

ToEncoderFunction: Function @ With[
	{
		dispatch = If[#OutputForm === "Index", makeSparseDispatch, makeOneHotDispatch][#Labels, #Count],
		depth = Length @ #Dimensions,
		isUnitVector = #OutputForm === "UnitVector"
	},
	makeReplacer[dispatch, #Dimensions, isUnitVector] /* 
	Which[
		isUnitVector, 
			toNA["UnsignedInteger8"], 
		depth > 0 && ContainsQ[#Dimensions, _LengthVar], 
			toNAList[countIntMinType @ #Count],
		depth > 0,
			toNA[countIntMinType @ #Count],
		True, 
			Identity
	]
]

TypeRandomInstance: Function[
	RandomChoice[#Labels]
]

MLType: Function["Nominal"]

EncoderToDecoder: Function[
	{"Class", #Labels, "InputDepth" -> (1 + Length[#Dimensions])}
]

makeSparseDispatch[labels_, dim_] :=
	Thread[labels -> Range[dim]] //
	Append[l_ :> EncodeFail["`` is not one of ``", l, labels]] //
	Dispatch;

makeOneHotDispatch[labels_, dim_] :=
	Thread[labels -> IdentityMatrix[dim]] //
	Append[l_ :> EncodeFail["`` is not one of ``", l, labels]] //
	Dispatch;
