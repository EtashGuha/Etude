BeginPackage["Compile`Core`IR`Instruction`RecordRestrictInstruction`"]

RecordRestrictInstruction;
RecordRestrictInstructionClass;
CreateRecordRestrictInstruction;
RecordRestrictInstructionQ;

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
RecordRestrictInstructionClass = DeclareClass[
	RecordRestrictInstruction,
	<|
		"getArgument" -> Function[{idx}, Self["arguments"][[idx]]],
		"setArgument" -> Function[{idx, val},
			Assert[Length[Self["arguments"]] <= idx && idx > 0];
			SetData[Self["arguments"], ReplaceRecordRestrict[Self["arguments"], idx -> val]]
		],
		"getOperand" -> Function[{idx}, Self["getArgument", idx]],
		"setOperand" -> Function[{idx, val}, Self["setArgument", idx, val]],
		"operands" -> Function[{}, Self["arguments"]], (* matches the interface for binary instruction *)
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "RecordRestrictInstruction",
			"_visitInstructionName" -> "visitRecordRestrictInstruction",
			"target",
			"source",
			"arguments" -> {}
		}
	],
	Predicate -> RecordRestrictInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateRecordRestrictInstruction[trgt_:None, source_:None, arguments_:{}, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[source === None || VariableQ[source]];
		Assert[AllTrue[arguments, (VariableQ[#] || ConstantValueQ[#])&]];
		inst = CreateObject[
			RecordRestrictInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"arguments" -> arguments,
				"mexpr" -> expr
			|>
		]
	)

CreateRecordRestrictInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateRecordRestrictInstruction ", args}]
	
(**************************************************)

icon := Graphics[Text[
	Style["RCRD\nEXT\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"RecordRestrictInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}],
  		    BoxForm`SummaryItem[{"arguments: ", inst["arguments"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		inst["source"]["toString"],
		InstructionNameText["[["],
		Riffle[#["toString"]& /@ inst["arguments"], ", "],
		InstructionNameText["]]"]
	]

format[ self_, shortQ_:True] :=
	"RecordRestrictInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> RecordRestrictInstructionClass,
	"name" -> "RecordRestrictInstruction",
	"predicate" -> RecordRestrictInstructionQ,
	"constructor" -> CreateRecordRestrictInstruction,
	"instance" -> RecordRestrictInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
