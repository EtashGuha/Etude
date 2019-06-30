BeginPackage["Compile`Core`IR`Instruction`TypeCastInstruction`"]

TypeCastInstruction;
TypeCastInstructionClass;
CreateTypeCastInstruction;
TypeCastInstructionQ;

Begin["`Private`"] 


Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
TypeCastInstructionClass = DeclareClass[
	TypeCastInstruction,
	<|
		"usedVariables" -> Function[{}, Select[{Self["source"]}, VariableQ]],
		"serialize" -> (serialize[Self, #]&),
		"definedVariable" -> Function[{}, Self["target"]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "TypeCastInstruction",
			"_visitInstructionName" -> "visitTypeCastInstruction",
			"target",
			"type",
			"source"
		}
	],
	Predicate -> TypeCastInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


Deserialize[ env_, data_] :=
	Module[ {trg, src, type},
		trg = env["getElement", data, "target"];
		src = env["getElement", data, "source"];
		type = env["getType", data["type"]];
		CreateTypeCastInstruction[trg, type, src]
	]

serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["source"] = self["source"]["serialize", env];
		data["type"] = self["serializeType", env, self["type"]];
		"Instruction"[data]
	]


CreateTypeCastInstruction[trgt0_:None, type_, source_:None, expr_:None]:=
	Module[{trgt},
		Assert[(trgt0 === None) || StringQ[trgt0] || VariableQ[trgt0]];
		Assert[source === None || VariableQ[source] || ConstantValueQ[source]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		CreateObject[
			TypeCastInstruction,
			<|
				"target" -> trgt,
				"type" -> type,
				"source" -> source,
				"mexpr" -> expr
			|>
		]
	]
CreateTypeCastInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateTypeCastInstruction ", args}]

(**************************************************)

icon := Graphics[Text[
	Style["Cast\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeCastInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	Assert[inst["source"] =!= None];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["Cast"],
		" ",
		inst["type"]["toString"],
		" ",
		inst["source"]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"TypeCastInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		"(",
		StyleBox[self["target"]["type"]["toString"], Italic, $TypeColor],
		")",
		self["source"]["makePrettyPrintBoxes"]
	}]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> TypeCastInstructionClass,
	"name" -> "TypeCastInstruction",
	"predicate" -> TypeCastInstructionQ,
	"deserialize" -> Deserialize,
	"constructor" -> CreateTypeCastInstruction,
	"instance" -> TypeCastInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
