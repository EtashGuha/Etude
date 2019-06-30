BeginPackage["Compile`Core`IR`Instruction`StoreInstruction`"]

StoreInstruction;
StoreInstructionClass;
CreateStoreInstruction;
StoreInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
StoreInstructionClass = DeclareClass[
	StoreInstruction,
	<|
		"usedVariables" -> Function[{}, If[VariableQ[Self["value"]], {Self["value"], Self["target"]}, {Self["target"]}]],
		"definedVariable" -> Function[{}, None],
		"serialize" -> (serialize[Self, #]&),
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "StoreInstruction",
			"_visitInstructionName" -> "visitStoreInstruction",
			"target",
			"value",
			"operator"
		}
	],
	Predicate -> StoreInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]

CreateStoreInstruction[value_, trgt0_, operatorIn_:None, expr_:None]:=
	Module[ {trgt = trgt0, operator = operatorIn},
		If[ operator === None,
			operator = CreateConstantValue[Native`Store]];
		Assert[(trgt === None) || VariableQ[trgt] || StringQ[trgt]];
		Assert[value === None || VariableQ[value] || ConstantValueQ[value]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0]
		];
		CreateObject[
			StoreInstruction,
			<|
				"target" -> trgt,
				"value" -> value,
				"mexpr" -> expr,
				"operator" -> operator
			|>
		]
	]
	
CreateStoreInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateStoreInstruction ", args}]


Deserialize[ env_, data_] :=
	Module[ {trg, val, oper},
		trg = env["getElement", data, "target"];
		val = env["getElement", data, "value"];
		oper = env["getElement", data, "operator"];
		CreateStoreInstruction[val, trg, oper]
	]

serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["value"] = self["value"]["serialize", env];
		data["operator"] = self["operator"]["serialize", env];
		"Instruction"[data]
	]



(**************************************************)

icon := Graphics[Text[
	Style["Store\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"StoreInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"value: ", inst["value"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	StringJoin[
		InstructionNameText["Store"],
		" ",
		inst["value"]["toString"],
		" -> ",
		inst["target"]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"StoreInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		self["value"]["makePrettyPrintBoxes"]
	}]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> StoreInstructionClass,
	"name" -> "StoreInstruction",
	"predicate" -> StoreInstructionQ,
	"deserialize" -> Deserialize,
	"constructor" -> CreateStoreInstruction,
	"instance" -> StoreInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
