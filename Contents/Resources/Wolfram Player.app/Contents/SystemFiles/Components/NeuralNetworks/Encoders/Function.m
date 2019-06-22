Output: $OutputType

Parameters:
	$Function: StandaloneFunctionT
	$OutputType: TypeT
	$Pattern: ValidatedParameterT[Identity, None]
	$Batched: BooleanT

Upgraders: {
	"11.3.6" -> Append[{
		"Pattern" -> ValidatedParameter[None],
		"Batched" -> False
	}]
}

HiddenFields: {"OutputType"}

PostInferenceFunction: Function @ Scope[
	If[!MatchQ[$Batched, True|False],
		listable = ListableFunctionQ @ StripVP @ $Function;
		PostSet[$Batched, listable]]
]

AcceptsLists: Function[
	Replace[
		StripVP @ #Pattern, {
		None :> True,
		patt_ :> Or[
			MatchQ[{}, patt],
			MatchQ[{5}, patt],
			!FreeQ[patt, List]
		]
	}] 
]

ToEncoderFunction: Function @ Scope[
	type = #OutputType;
	func = StripVP @ #Function;
	matcher = checkPattern[StripVP @ #Pattern];
	applyer = quieted @ If[#Batched, func, Map[func]];
	typeTest = ToLiteralTypeTest @ ListT[SizeT, type];
	checker = checkOutputType[typeTest, type];
	matcher /* applyer /* checker
]

checkPattern[None] := Identity;
checkPattern[p_][e_] := 
	If[MatchQ[e, {Repeated[p]}], e, EncodeFail["input to \"Function\" NetEncoder did not match pattern ``", p]];

quieted[f_][e_] := Quiet[Check[f[e], $Failed]];

Kind: Function[
	Replace[
		StripVP @ #Pattern, {
		Verbatim[Blank][s_Symbol] :> If[Context[s] === "System`", ToLowerCase, Identity] @ SymbolName[s],
		_ -> "expression"
	}]
]

checkOutputType[test_, type_][data_] :=
	Which[
		!TrueQ[test[data]], 
			fail[type],
		NumericArrayQ[data] || VectorQ[data, NumericArrayQ],
			data,
		ArrayQ[data],
			autoToNumericArray[type][data],
		True,
			Map[autoToNumericArray[type], data]
	];

autoToNumericArray[type_] := Switch[TType[type],
	IndexIntegerT[All],
		Function @ Quiet @ Check[toNumericArray[#, "Integer32"], fail[type]],
	_IndexIntegerT,
		Function @ Quiet @ Check[toNumericArray[#, "UnsignedInteger32"], fail[type]],
	_,
		Function @ toNumericArray[#]
];

fail[type_] := EncodeFail["\"Function\" encoder did not produce an output that was ``", TextString @ TypeForm[type]];

MLType: Function["Expression"]

InputPattern: Function[Replace[First[#Pattern], None -> _]]
