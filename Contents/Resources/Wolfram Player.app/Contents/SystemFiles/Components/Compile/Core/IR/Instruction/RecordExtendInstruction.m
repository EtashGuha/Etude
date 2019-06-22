BeginPackage["Compile`Core`IR`Instruction`RecordExtendInstruction`"]

RecordExtendInstruction;
RecordExtendInstructionClass;
CreateRecordExtendInstruction;
RecordExtendInstructionQ;

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
RecordExtendInstructionClass = DeclareClass[
	RecordExtendInstruction,
	<|
		"arguments" -> Function[{}, Join[{Self["value"]}, Self["rest"]]],
		"getArgument" -> Function[{idx}, Self["arguments"][[idx]]],
		"setArgument" -> Function[{idx, val},
			Assert[Length[Self["arguments"]] <= idx && idx > 0];
			SetData[Self["arguments"], ReplaceRecordExtend[Self["arguments"], idx -> val]]
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
			"_instructionName" -> "RecordExtendInstruction",
			"_visitInstructionName" -> "visitRecordExtendInstruction",
			"target",
			"source",
			"label" -> "",
			"value",
			"rest" -> {}
		}
	],
	Predicate -> RecordExtendInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateRecordExtendInstruction[trgt_:None, label_, value_, rest_, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[source === None || VariableQ[source]];
		Assert[AllTrue[rest, (VariableQ[#] || ConstantValueQ[#])&]];
		inst = CreateObject[
			RecordExtendInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"label" -> label,
				"value" -> value,
				"rest" -> rest,
				"mexpr" -> expr
			|>
		]
	)

CreateRecordExtendInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateRecordExtendInstruction ", args}]
	
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
		"RecordExtendInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"label: ", inst["label"]}],
  		    BoxForm`SummaryItem[{"value: ", inst["value"]}],
  		    BoxForm`SummaryItem[{"rest: ", inst["rest"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["<{"],
		" ",
		inst["label"]["toString"],
		" -> ",
		inst["value"]["toString"],
		" || ",
		Riffle[#["toString"]& /@ inst["rest"], ", "],
		" ",
		InstructionNameText["}>"]
	]

format[ self_, shortQ_:True] :=
	"RecordExtendInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> RecordExtendInstructionClass,
	"name" -> "RecordExtendInstruction",
	"predicate" -> RecordExtendInstructionQ,
	"constructor" -> CreateRecordExtendInstruction,
	"instance" -> RecordExtendInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>]
]]

End[]

EndPackage[]
