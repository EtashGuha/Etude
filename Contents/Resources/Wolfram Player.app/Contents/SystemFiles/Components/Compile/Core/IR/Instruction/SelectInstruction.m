BeginPackage["Compile`Core`IR`Instruction`SelectInstruction`"]

SelectInstruction;
SelectInstructionClass;
CreateSelectInstruction;
SelectInstructionQ;

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


RegisterCallback["DeclareCompileClass", Function[{st},
SelectInstructionClass = DeclareClass[
	SelectInstruction,
	<|
		"usedVariables" -> Function[{}, Select[Prepend[Self["operands"], Self["condition"]], VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"getOperand" -> Function[{idx}, Self["operands"][[idx]]],
		"setOperand" -> Function[{idx, val}, SetData[Self["operands"], ReplacePart[Self["operands"], idx -> val]]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "SelectInstruction",
			"_visitInstructionName" -> "visitSelectInstruction",
			"target",
			"condition",
			"operands" -> {}
		}
	],
	Predicate -> SelectInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateSelectInstruction[trgt_:None, cond_:None, operands_:{}, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[operands === {} || Length[operands] === 2];
		Assert[AllTrue[operands, (VariableQ[#] || ConstantValueQ[#])&]];
		CreateObject[
			SelectInstruction,
			<|
				"target" -> trgt,
				"condition" -> cond,
				"operands" -> operands,
				"mexpr" -> expr
			|>
		]
	)

CreateSelectInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateSelectInstruction ", args}]
	
(**************************************************)

icon := Graphics[Text[
	Style["Select\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"SelectInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"condition: ", inst["condition"]}],
  		    BoxForm`SummaryItem[{"operands: ", inst["operands"]}]
  		},
  		{}, 
  		fmt
  	]


toString[inst_] := (
	Assert[Length[inst["operands"]] === 2];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		inst["condition"]["toString"],
		" ? ",
		First[inst["operands"]]["toString"],
		" : ",
		Last[inst["operands"]]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"SelectInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[Flatten[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		StyleBox["If", Bold, $OperatorColor],
		"[",
		Riffle[
		    #["makePrettyPrintBoxes"]& /@ Prepend[self["operands"], self["condition"]],
		    ","
		],
		"]"
	}]]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> SelectInstructionClass,
	"name" -> "SelectInstruction",
	"predicate" -> SelectInstructionQ,
	"constructor" -> CreateSelectInstruction,
	"instance" -> SelectInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
