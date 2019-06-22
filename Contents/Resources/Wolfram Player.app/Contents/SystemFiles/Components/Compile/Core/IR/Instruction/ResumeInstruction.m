BeginPackage["Compile`Core`IR`Instruction`ResumeInstruction`"]

ResumeInstruction;
ResumeInstructionClass;
CreateResumeInstruction;
ResumeInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileClass", Function[{st},
ResumeInstructionClass = DeclareClass[
	ResumeInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, {Self["value"]}],
		"definedVariable" -> Function[{}, None],
		"hasOperands" -> Function[{}, True],
		"operands" -> Function[{},  {Self["value"]}],
		"getOperand" -> Function[{idx}, Assert[idx == 1]; Self["value"]],
		"setOperand" -> Function[{idx, val}, Assert[idx == 1]; Self["setValue", val]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "ResumeInstruction",
			"_visitInstructionName" -> "visitResumeInstruction",
			"value"
		}
	],
	Predicate -> ResumeInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateResumeInstruction[value_:Undefined, expr_:None] :=
	(
		Assert[(value === Undefined) || VariableQ[value]];
		CreateObject[
			ResumeInstruction,
			<|
				"value" -> value,
				"mexpr" -> expr
			|>
		]
	)
CreateResumeInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateResumeInstruction ", args}]

Deserialize[ env_, data_] :=
	Module[ {value},
		value = env["getElement", data, "value"];
		CreateResumeInstruction[value]
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
	Style["Resume\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"CreateResumeInstructionInstruction",
		inst["toString"],
  		icon,
  		Join[
  			{
				BoxForm`SummaryItem[{"id: ", inst["id"]}]
  			},

  			{BoxForm`SummaryItem[{"value: ", inst["value"]}]}
  		],
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		"Resume",
		" ",
		inst["value"]["toString"]
	]

format[ self_, shortQ_:True] :=
	"ResumeInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
		RowBox[{
			StyleBox["Return", Bold, $OperatorColor],
			" ",
			self["value"]["makePrettyPrintBoxes"]
		}
	]
	

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> ResumeInstructionClass,
	"name" -> "ResumeInstruction",
	"predicate" -> ResumeInstructionQ,
	"constructor" -> CreateResumeInstruction,
	"deserialize" -> Deserialize,
	"instance" -> ResumeInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
