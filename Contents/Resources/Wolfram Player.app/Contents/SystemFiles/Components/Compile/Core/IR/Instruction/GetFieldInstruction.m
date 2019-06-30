BeginPackage["Compile`Core`IR`Instruction`GetFieldInstruction`"]

GetFieldInstruction;
GetFieldInstructionClass;
CreateGetFieldInstruction;
GetFieldInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
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
GetFieldInstructionClass = DeclareClass[
	GetFieldInstruction,
	<|
		"usedVariables" -> Function[{}, Select[{Self["source"], Self["field"]}, VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"operands" -> Function[{},
			{Self["source"], Self["field"]}
		],
		"getOperand" -> Function[{idx}, Self["operands"][[idx]]],
		"setOperand" -> Function[{idx, val},
			Which[
				idx === 1,
					Self["setSource", val],
				idx == 2,
					Self["setOffset", val],
				True,
					ThrowException[{"setOperand index ", idx, "is out of range for GetFieldInstruction"}]
			]
		],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "GetFieldInstruction",
			"_visitInstructionName" -> "visitGetFieldInstruction",
			"target",
			"field",
			"source"
		}
	],
	Predicate -> GetFieldInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateGetFieldInstruction[trgt_, source_, field_, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[source === None || VariableQ[source]];
		CreateObject[
			GetFieldInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"field" -> field,
				"mexpr" -> expr
			|>
		]
	)

CreateGetFieldInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateGetFieldInstruction ", args}]
	
(**************************************************)
(**************************************************)



icon := Graphics[Text[
	Style["SetElem\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"GetFieldInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}],
  		    BoxForm`SummaryItem[{"field: ", inst["field"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		inst["source"]["toString"],
		InstructionNameText["@>"],
		inst["field"]["toString"]
	]

format[ self_, shortQ_:True] :=
	"GetFieldInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		self["source"]["makePrettyPrintBoxes"],
		"@>",
		self["field"]["makePrettyPrintBoxes"]
	}]


(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> GetFieldInstructionClass,
	"name" -> "GetFieldInstruction",
	"predicate" -> GetFieldInstructionQ,
	"constructor" -> CreateGetFieldInstruction,
	"instance" -> GetFieldInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
