BeginPackage["Compile`Core`IR`Instruction`UnreachableInstruction`"]

UnreachableInstruction;
UnreachableInstructionClass;
CreateUnreachableInstruction;
UnreachableInstructionQ;

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
UnreachableInstructionClass = DeclareClass[
	UnreachableInstruction,
	<|
		"usedVariables" -> Function[{}, {}],
		"definedVariable" -> Function[{}, None],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "UnreachableInstruction",
			"_visitInstructionName" -> "visitUnreachableInstruction",
			"target"
		}
	],
	Predicate -> UnreachableInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateUnreachableInstruction[expr_:None] :=
	CreateObject[UnreachableInstruction, <| "mexpr" -> expr |>]
	
CreateUnreachableInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateUnreachableInstruction ", args}]

(**************************************************)

icon := Graphics[Text[
	Style["Unreachable\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"UnreachableInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := InstructionNameText["Unreachable"]

format[ self_, shortQ_:True] :=
	"UnreachableInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		StyleBox["Unreachable", Bold, $OperatorColor]
	}]
	
(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> UnreachableInstructionClass,
	"name" -> "UnreachableInstruction",
	"predicate" -> UnreachableInstructionQ,
	"constructor" -> CreateUnreachableInstruction,
	"instance" -> UnreachableInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
