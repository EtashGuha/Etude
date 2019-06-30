BeginPackage["Compile`Core`IR`Instruction`LabelInstruction`"]

LabelInstruction;
LabelInstructionClass;
CreateLabelInstruction;
LabelInstructionQ;

Begin["`Private`"] 

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
LabelInstructionClass = DeclareClass[
	LabelInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, {}],
		"definedVariable" -> Function[{}, None],
		"hasValue" -> Function[{}, Self["value"] =!= None],
		"makePrettyPrintBoxes" -> (makePrettyPrintBoxes[Self]&),
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "LabelInstruction",
			"_visitInstructionName" -> "visitLabelInstruction",
			"name"
		}
	],
	Predicate -> LabelInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateLabelInstruction[name_String, expr_:None] :=
	(
		CreateObject[
			LabelInstruction,
			<|
				"name" -> name,
				"mexpr" -> expr
			|>
		]
	)
CreateLabelInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateLabelInstruction ", args}]


Deserialize[ env_, data_] :=
	Module[ {name},
		name = env["getElementNoDeserialize", data, "name"];
		CreateLabelInstruction[name]
	]
	

(**************************************************)
(**************************************************)

serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data["name"] = self["name"];
		"Instruction"[data]
	]

	
(**************************************************)

icon := Graphics[Text[
	Style["Label\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LabelInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"label: ", inst["name"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		labelFullName[inst],
		":"
	]

format[ self_, shortQ_:True] :=
	"LabelInstruction " <> self["toString"]

labelFullName[inst_] :=
	BoldGrayText[
		StringJoin[
	       inst["name"], "(", ToString[inst["basicBlock"]["id"]], ")"
	   ]
	]

makePrettyPrintBoxes[self_] :=
	StyleBox[labelFullName[self], Bold, $LabelColor]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> LabelInstructionClass,
	"name" -> "LabelInstruction",
	"predicate" -> LabelInstructionQ,
	"constructor" -> CreateLabelInstruction,
	"deserialize" -> Deserialize,
	"instance" -> LabelInstruction,
	"properties" -> {
	}
|>] 
]]

End[]

EndPackage[]
