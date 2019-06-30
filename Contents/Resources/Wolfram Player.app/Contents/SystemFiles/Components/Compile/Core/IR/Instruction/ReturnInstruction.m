BeginPackage["Compile`Core`IR`Instruction`ReturnInstruction`"]

ReturnInstruction;
ReturnInstructionClass;
CreateReturnInstruction;
ReturnInstructionQ;

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
ReturnInstructionClass = DeclareClass[
	ReturnInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, If[Self["hasValue"] && VariableQ[Self["value"]], {Self["value"]}, {}]],
		"definedVariable" -> Function[{}, None],
		"hasOperands" -> Function[{}, Self["hasValue"]],
		"operands" -> Function[{}, If[Self["hasValue"], {Self["value"]}, {}]],
		"getOperand" -> Function[{idx}, Assert[idx == 1]; If[Self["hasValue"], Self["value"]]],
		"setOperand" -> Function[{idx, val}, Assert[idx == 1]; If[Self["hasValue"], Self["setValue", val]]],
		"hasValue" -> Function[{}, Self["value"] =!= Undefined],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "ReturnInstruction",
			"_visitInstructionName" -> "visitReturnInstruction",
			"value"
		}
	],
	Predicate -> ReturnInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateReturnInstruction[value_:Undefined, expr_:None] :=
	(
		Assert[(value === Undefined) || VariableQ[value] || ConstantValueQ[value]];
		CreateObject[
			ReturnInstruction,
			<|
				"value" -> value,
				"mexpr" -> expr
			|>
		]
	)
CreateReturnInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateReturnInstruction ", args}]

Deserialize[ env_, data_] :=
	Module[ {value},
		value = env["getElement", data, "value"];
		CreateReturnInstruction[value]
	]
	

(**************************************************)
(**************************************************)

serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data = Delete[data, "operands"];
		data["value"] = self["value"]["serialize", env];
		"Instruction"[data]
	]


	
(**************************************************)

icon := Graphics[Text[
	Style["Return\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"ReturnInstruction",
		inst["toString"],
  		icon,
  		Join[
  			{
				BoxForm`SummaryItem[{"id: ", inst["id"]}]
  			},
  			If[inst["hasValue"],
  				{BoxForm`SummaryItem[{"value: ", inst["value"]}]},
  				{}
  			]
  		],
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		InstructionNameText["Return"],
		If[inst["hasValue"],
			" " <> inst["value"]["toString"],
			""
		]
	]

format[ self_, shortQ_:True] :=
	"ReturnInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	If[self["hasValue"],
		RowBox[{
			StyleBox["Return", Bold, $OperatorColor],
			" ",
			self["value"]["makePrettyPrintBoxes"]
		}],
		RowBox[{
    			StyleBox["Return", Bold, $OperatorColor]
		}]
	]
	

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> ReturnInstructionClass,
	"name" -> "ReturnInstruction",
	"predicate" -> ReturnInstructionQ,
	"constructor" -> CreateReturnInstruction,
	"deserialize" -> Deserialize,
	"instance" -> ReturnInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
