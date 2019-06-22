BeginPackage["Compile`Core`IR`ConstantValue`"]

ConstantValueQ;
CreateConstantValue;
ConstantValueClass;

CreateUninitialized

CreateUninitializedConstant
CreateUninitializedValue

CreateVoidValue

UninitializedConstantQ

CreateENULLConstant

DeserializeConstantValue

ProcessPackedArray

Begin["`Private`"] 


Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Internal`Utilities`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Markup`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


If[!ValueQ[nextId],
	nextId = 1
]

(*
	logger := logger = CreateLogger["ConstantValue", "TRACE"]
*)

sameQ[self_, other_] :=
	ConstantValueQ[other] && 
	With[{
		selfValue = self["value"],
		otherValue = other["value"]
	},
		Which[
			MExprQ[selfValue],
				selfValue["sameQ", otherValue],
			MExprQ[otherValue],
				otherValue["sameQ", selfValue], 
			True,
				selfValue === otherValue
		]
	]

serialize[ self_, env_] :=
	"ConstantValue"[<|"value" -> self["value"], "type" -> env["serializeType", self["type"]]|>]


RegisterCallback["DeclareCompileClass", Function[{st},
ConstantValueClass = DeclareClass[
	ConstantValue,
	<|
		"initialize" -> Function[{}, Self["setProperties", CreateReference[<||>]]],
		"serialize" -> (serialize[Self, #]&),
		"clone" -> (clone[Self, ##]&),
		"sameQ" -> Function[{other}, sameQ[Self, other]],
		"typename" -> Function[{}, Self["type"]["toString"]],
		"addUse" -> Function[{inst},
			AssertThat["The variable use must be an instruction.",
				inst]["named", "Instruction"]["satisfies", InstructionQ];
            If[!MemberQ[#["id"]& /@ Self["uses"], inst["id"]],
                Self["setUses", Append[Self["uses"], inst]]
            ];
			Self],
		"clearUses" -> Function[{},
			Self["setUses", {}];
			Self
		],
		"dispose" -> Function[{}, dispose[Self]], 
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"id" -> 0,
		"value",
		"type" -> Undefined,
		"mexpr",
		"properties",
		"uses" -> {}
	},
	Predicate -> ConstantValueQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


CreateConstantValue[mexpr_?MExprQ] :=
	Module[{var, val},
		val = getValue[mexpr];
		var = CreateObject[
			ConstantValue,
			<|
				"id" -> nextId++,
				"value" -> val,
				"type" -> Undefined,
				"mexpr" -> mexpr
			|>
		];
		If[mexpr["type"] =!= Undefined,
			var["setType", mexpr["type"]]
		];
		var
	]
	
	

CreateConstantValue[val_] :=
	Module[{var},
		var = CreateObject[
			ConstantValue,
			<|
				"id" -> nextId++,
				"value" -> val,
				"type" -> Undefined,
				"mexpr" -> None
			|>
		];
		var
	]

(*
  Note the special behaviour for Type[ Undefined].
  This is because the default type is Type[ Undefined], but if this 
  never gets Resolved then the serialized form will be Type[Undefined].
  However if we resolve the type,  which env["getType", ty] does, 
  then the serialized form will be Type["Undefined"].
  
  Perhaps the serialized form of the non-type instance should be Type["Undefined"].
*)
DeserializeConstantValue[ env_, "ConstantValue"[ data_]] :=
	deserialize[ env, data]

deserialize[ env_, data_] :=
	Module[ {var, type},
		var = CreateConstantValue[ data["value"]];
		type = data["type"];
		If[ type =!= Undefined,
			type = env["getType", data["type"]]
		];
		var["setType", type];
		var
	]
	

clone[self_, env_] :=
	deserialize[ env, self]
	
clone[self_] :=
	deserialize[ CreateCloneEnvironment[], self]

isSymbol[ head_, mexpr_] :=
	If[ mexpr["symbolQ"],  mexpr["data"] === HoldComplete[head], False]
	
isRational[ mexpr_] :=
	isSymbol[ Rational, mexpr]

isComplex[ mexpr_] :=
	isSymbol[ Complex, mexpr]

isList[ mexpr_] :=
	isSymbol[ List, mexpr]

isGlobal[ mexpr_] :=
	isSymbol[ Native`Global, mexpr]



getNormalValue[head_?isGlobal, mexpr_] :=
	Module[ {data},
		data = FromMExpr[mexpr];
		If[ !MatchQ[ data, HoldComplete[ Native`Global[_String]]],
			ThrowException[{"ConstantValue,  malformed Global expression", mexpr}]
		];
		ReleaseHold[data]
	]

getNormalValue[head_?isRational, mexpr_] :=
	Module[ {data},
		data = FromMExpr[mexpr];
		If[ !MatchQ[ data, HoldComplete[ Rational[_Integer, _Integer]]],
			ThrowException[{"ConstantValue,  malformed Rational expression", mexpr}]
		];
		ReleaseHold[data]
	]

getNormalValue[head_?isComplex, mexpr_] :=
	Module[ {data},
		data = FromMExpr[mexpr];
		If[ !MatchQ[ data, HoldComplete[ Complex[_, _]]],
			ThrowException[{"ConstantValue,  malformed Complex expression", mexpr}]
		];
		ReleaseHold[data]
	]


(*
  See if the input contains things that could be packed arrays.  This doesn't 
  check packability,  just what it contains.  If it doesn't contain the right 
  information there will be an error.
*)
isPackedArrayAble[ mexpr_] :=
	Module[{head},
		head = mexpr["head"];
		If[ 
			isList[head] || isComplex[head] || isSymbol[Plus,head] || isSymbol[Compile`ConstantValue, head],
				AllTrue[mexpr["arguments"], isPackedArrayAble],
			head["sameQ", Integer] || head["sameQ", Real]]
	]
			

getNormalValue[head_?isList, mexpr_] :=
	Module[ {data},
		If[!isPackedArrayAble[mexpr],
			ThrowException[{"ConstantValue,  malformed constant List expression", mexpr}]];
		data = ReleaseHold[FromMExpr[mexpr]];
		data = DeleteCases[data, Compile`ConstantValue, Infinity, Heads -> True];
		data
	]


ProcessPackedArray[ mexpr_] :=
	Module[{data},
		If[ isPackedArrayAble[mexpr],
			data = ReleaseHold[FromMExpr[mexpr]];
			data = DeleteCases[data, Compile`ConstantValue, Infinity, Heads -> True];
			data
			,
			Null]
	]



getNormalValue[head_, mexpr_] :=
	Module[ {data},
		data = FromMExpr[mexpr];
		data
	]



	
getValue[mexpr_] :=
	Which[
		mexpr["literalQ"],
			mexpr["data"]
		,
		mexpr["symbolQ"],
			ReleaseHold[mexpr["data"]]
		,
		mexpr["normalQ"],
			getNormalValue[mexpr["head"], mexpr]
		,
		True,
			ThrowException[{"ConstantValue,  unknown expression", mexpr}]
	]


valToString[val_?NumberQ] :=
	Which[
		Head[val] === Rational,
			StringJoin[ valToString[Numerator[val]], "/", valToString[Denominator[val]]]
		,
		IntegerQ[val],
			ToString[val]
		,
		True,
			ToString[val, CForm]]


dispose[self_] :=
	Module[{},
		self["clearUses"];
	]

	
valToString[val_] :=
	ToString[val, InputForm, PageWidth -> Infinity]
	
isLocal[var_] :=
	var["hasProperty", "localFunctionModule"]
	
toString[var_?ConstantValueQ] := (
	StringJoin[
		If[isLocal[var],
			BoldGreenText[valToString[var["value"]]],
			GrayText[valToString[var["value"]]]
		],
		If[var["type"] =!= TypeSpecifier[Undefined],
			":" <> BoldRedText[IRFormatType[var["type"]]],
			""
		]
	]
)
	
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["Const", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
    
typeName[Undefined] = Undefined
typeName[Type[args___]] = args
typeName[TypeSpecifier[args___]] = args
typeName[t_] := t["name"]
  
toBoxes[var_?ConstantValueQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"ConstantValue",
		var["toString"],
  		icon,
  		{
  		    BoxForm`SummaryItem[{"value: ", var["value"]}],
  		    BoxForm`SummaryItem[{"type: ", typeName[var["type"]]}]
  		},
  		{}, 
  		fmt
  	]
  	
makePrettyPrintBoxes[self_] := 
    With[{v = self["toString"]},
    		RowBox[{
    			StyleBox[v, Bold, $ConstantValueColor]
    		}]
    ]


uninitializedExpr := uninitializedExpr = CreateMExprSymbol[Compile`Uninitialized]

voidExpr := voidExpr = CreateMExprSymbol[Compile`Void]


CreateUninitializedValue[] :=
	CreateUninitializedConstant[]

CreateUninitializedConstant[] := 
	CreateConstantValue[uninitializedExpr]

CreateUninitializedConstant[] := 
	CreateConstantValue[uninitializedExpr]

CreateVoidValue[] :=
	CreateConstantValue[voidExpr]

nullReferenceExpr := nullReferenceExpr = CreateMExprSymbol[ Compile`NullReference]

UninitializedConstantQ[ var_] :=
	ConstantValueQ[var] && var["value"] === Compile`Uninitialized


(*
 Perhaps ENULL does not belong here.
*)
enullReferenceExpr := enullReferenceExpr = CreateMExprSymbol[ Compile`Internal`ENULLReference]

CreateENULLConstant[] :=
	CreateConstantValue[enullReferenceExpr]



End[]

EndPackage[]
