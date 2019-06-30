BeginPackage["Compile`Core`IR`Instruction`NewRecordInstruction`"]

NewRecordInstruction;
NewRecordInstructionClass;
CreateNewRecordInstruction;
NewRecordInstructionQ;

Begin["`Private`"] 


Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
NewRecordInstructionClass = DeclareClass[
	NewRecordInstruction,
	<|
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "NewRecordInstruction",
			"_visitInstructionName" -> "visitNewRecordInstruction",
			"target",
			"type"
		}
	],
	Predicate -> NewRecordInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateNewRecordInstruction[trgt_:None, type_, expr_:None] :=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		CreateObject[
			NewRecordInstruction,
			<|
				"target" -> trgt,
				"type" -> type,
				"mexpr" -> expr
			|>
		]
	)
CreateNewRecordInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateNewRecordInstruction ", args}]

(**************************************************)

icon := Graphics[Text[
	Style["StkAlloc\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"NewRecordInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
			BoxForm`SummaryItem[{"type: ", inst["type"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	AssertThat["The target needs to be initialized",
		Self["target"]][
		"named", "target"][
		"isNotEqualTo", None];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["NewRecord"],
		" ",
		inst["type"]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"NewRecordInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> NewRecordInstructionClass,
	"name" -> "NewRecordInstruction",
	"predicate" -> NewRecordInstructionQ,
	"constructor" -> CreateNewRecordInstruction,
	"instance" -> NewRecordInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
