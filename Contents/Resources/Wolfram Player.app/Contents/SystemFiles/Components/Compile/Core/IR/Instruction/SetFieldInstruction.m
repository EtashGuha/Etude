BeginPackage["Compile`Core`IR`Instruction`SetFieldInstruction`"]

SetFieldInstruction;
SetFieldInstructionClass;
CreateSetFieldInstruction;
SetFieldInstructionQ;

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
SetFieldInstructionClass = DeclareClass[
	SetFieldInstruction,
	<|
		"usedVariables" -> Function[{}, Select[Self["operands"], VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"operands" -> Function[{},
			{Self["target"], Self["source"], Self["field"]}
		],
		"getOperand" -> Function[{idx}, Self["operands"][[idx]]],
		"setOperand" -> Function[{idx, val},
			Which[
				idx === 1,
					Self["setSource", val],
				idx === 2,
					Self["setSource", val],
				idx == 3,
					Self["setOffset", val],
				True,
					ThrowException[{"setOperand index ", idx, "is out of range for SetFieldInstruction"}]
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
			"_instructionName" -> "SetFieldInstruction",
			"_visitInstructionName" -> "visitSetFieldInstruction",
			"target", (** The set element is done in place, so target is modified *)
			"field",
			"source"
		}
	],
	Predicate -> SetFieldInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateSetFieldInstruction[trgt_, field_, source_, expr_:None] :=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[source === None || VariableQ[source]];
		CreateObject[
			SetFieldInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"field" -> field,
				"mexpr" -> expr
			|>
		]
	)

CreateSetFieldInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateSetFieldInstruction ", args}]
	
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
		"SetFieldInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"field: ", inst["field"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		inst["target"]["toString"],
		InstructionNameText["@>"],
		inst["field"]["toString"],
		" = ",
		inst["source"]["toString"]
	]

format[ self_, shortQ_:True] :=
	"SetFieldInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"@>",
		self["field"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		self["source"]["makePrettyPrintBoxes"]
	}]



(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> SetFieldInstructionClass,
	"name" -> "SetFieldInstruction",
	"predicate" -> SetFieldInstructionQ,
	"constructor" -> CreateSetFieldInstruction,
	"instance" -> SetFieldInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]
 	
End[]

EndPackage[]
