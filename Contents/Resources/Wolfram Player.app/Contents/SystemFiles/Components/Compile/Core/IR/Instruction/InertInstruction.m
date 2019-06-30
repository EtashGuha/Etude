BeginPackage["Compile`Core`IR`Instruction`InertInstruction`"]

InertInstruction;
InertInstructionClass;
CreateInertInstruction;
InertInstructionQ;

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
InertInstructionClass = DeclareClass[
	InertInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, Select[Prepend[Self["arguments"], Self["head"]], VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"isRecord" -> Function[{}, Self["mexpr"] =!= None && Self["mexpr"]["hasHead", Record] && ListQ[Self["arguments"]]],
		"getArgument" -> Function[{idx}, Self["arguments"][[idx]]],
		"setArgument" -> Function[{idx, val}, Assert[Length[Self["arguments"]] >= idx && idx > 0]; SetData[Self["arguments"], ReplacePart[Self["arguments"], idx -> val]]],
		"getOperand" -> Function[{idx}, Self["getArgument", idx]],
		"setOperand" -> Function[{idx, val}, Self["setArgument", idx, val]],
		"setOperands" -> Function[{args}, Self["setArguments", args]],
		"operands" -> Function[{}, If[Self["isRecord"], Self["arguments"][[All, 2]], Self["arguments"]]], (* matches the interface for binary instruction *)
		"length" -> Function[{}, Length[Self["arguments"]]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "InertInstruction",
			"_visitInstructionName" -> "visitInertInstruction",
			"target",
			"head",
			"arguments" -> {}
		}
	],
	Predicate -> InertInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateInertInstruction[trgt_:None, head_:None, args_:{}, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[head === None || ConstantValueQ[head] || VariableQ[head]];
		Assert[AllTrue[args, (VariableQ[#] || ConstantValueQ[#])&]];
		CreateObject[
			InertInstruction,
			<|
				"target" -> trgt,
				"head" -> head,
				"arguments" -> args,
				"mexpr" -> expr
			|>
		]
	)

CreateInertInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateInertInstruction ", args}]
	

Deserialize[ env_, data_] :=
	Module[ {trg, head, args},
		trg = env["getElement", data, "target"];
		head = env["getElement", data, "head"];
		args = env["getElementList", data, "arguments"];
		CreateInertInstruction[trg, head, args]
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["head"] = self["head"]["serialize", env];
		data["arguments"] = Map[ #["serialize", env]&, self["arguments"]];
		"Instruction"[data]
	]



icon := Graphics[Text[
	Style["Inert\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"InertInstruction",
		inst["toString"],
  		icon,
		Flatten[{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"head: ", inst["head"]}],
  		    If[inst["isRecord"],
  		    		{
  		    			BoxForm`SummaryItem[{"fields: ", inst["arguments"][[All, 1]]}],
  		    			BoxForm`SummaryItem[{"values: ", inst["arguments"][[All, 2]]}]
  		    		},
  		    		BoxForm`SummaryItem[{"arguments: ", inst["arguments"]}]
  		    ]
  		}],
  		{}, 
  		fmt
  	]


toString[inst_] := (
	Assert[VariableQ[inst["head"]] || ConstantValueQ[inst["head"]]];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["Inert"],
		" ",
		instHead[inst],
		"[",
		stringArgs[inst],
		"]"
	]
)
	
format[ self_, shortQ_:True] :=
	"InertInstruction " <> self["toString"]

instHead[inst_] :=
	If[StringQ[inst["head"]],
		inst["head"],
		inst["head"]["toString"]
	]

stringArgs[inst_] :=
	If[inst["isRecord"],
		Riffle[
			Table[
				arg[[1]]["toString"] <> " -> " <> arg[[2]]["toString"],
				{arg, inst["arguments"]}
			],
			", "
		],
		Riffle[#["toString"]& /@ inst["arguments"], ", "]
	]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		StyleBox["Inert", Bold, $OperatorColor],
		" ",
		self["head"]["makePrettyPrintBoxes"],
		"[",
		RowBox[
			Riffle[Map[#["makePrettyPrintBoxes"]&, self["arguments"]], ", "]
		],
		"]"
	}]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> InertInstructionClass,
	"name" -> "InertInstruction",
	"predicate" -> InertInstructionQ,
	"constructor" -> CreateInertInstruction,
	"deserialize" -> Deserialize,
	"instance" -> InertInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
