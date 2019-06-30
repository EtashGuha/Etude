BeginPackage["Compile`Core`IR`Instruction`InvokeInstruction`"]

InvokeInstruction;
InvokeInstructionClass;
CreateInvokeInstruction;
InvokeInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileClass", Function[{st},
InvokeInstructionClass = DeclareClass[
	InvokeInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{},
		    Select[Prepend[Self["arguments"], Self["function"]], VariableQ]
		],
		"definedVariable" -> Function[{}, Self["target"]],
		"getArgument" -> Function[{idx}, Assert[0 < idx && idx <= Length[Self["arguments"]]]; Self["arguments"][[idx]]],
		"setArgument" -> Function[{idx, val}, Assert[0 < idx && idx <= Length[Self["arguments"]]]; SetData[Self["arguments"], ReplacePart[Self["arguments"], idx -> val]]],
		"getOperand" -> Function[{idx}, Self["getArgument", idx]],
		"setOperand" -> Function[{idx, val}, Self["setArgument", idx, val]],
		"setOperands" -> Function[{args}, Self["setArguments", args]],
		"operands" -> Function[{}, Self["arguments"]], (* matches the interface for binary instruction *)
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "InvokeInstruction",
			"_visitInstructionName" -> "visitInvokeInstruction",
			"target",
			"function",
			"arguments" -> {},
			"to",
			"unwind",
			"attributes" -> {}
		}
	],
	Predicate -> InvokeInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateInvokeInstruction[trgt0_, callFun_, args0_, to_, unwind_, expr_:None] :=
	Module[{trgt, args = Flatten[{args0}]},
		Assert[(trgt0 === None) || StringQ[trgt0]  || VariableQ[trgt0]];
		Assert[(callFun === None) || VariableQ[callFun] || ConstantValueQ[callFun]];
		Assert[AllTrue[args, (VariableQ[#] || ConstantValueQ[#])&]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		AssertThat["BasicBlock arguments are valid basic blocks",
			{to, unwind}]["named", "operands"]["elementsSatisfy", BasicBlockQ[#] || MatchQ[#, "BasicBlockID"[_Integer]] &];

		CreateObject[
			InvokeInstruction,
			<|
				"target" -> trgt,
				"function" -> callFun,
				"arguments" -> args,
				"to" -> to,
				"unwind" -> unwind,
				"mexpr" -> expr
			|>
		]
	]
CreateInvokeInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateInvokeInstruction ", args}]
	
Deserialize[ env_, data_] :=
	Module[ {trg, fun, args},
		trg = env["getElement", data, "target"];
		fun = env["getElement", data, "function"];
		args = env["getElementList", data, "arguments"];
		CreateInvokeInstruction[trg, fun, args]
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["function"] = self["function"]["serialize", env];
		data["arguments"] = Map[ #["serialize", env]&, self["arguments"]];
		"Instruction"[data]
	]


	
(**************************************************)

icon := Graphics[Text[
	Style["Invoke\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"InvokeInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"function: ", inst["function"]}],
  		    BoxForm`SummaryItem[{"arguments: ", inst["arguments"]}],
  		    BoxForm`SummaryItem[{"to: ", inst["to"]["fullName"]}],
  		    BoxForm`SummaryItem[{"unwind: ", inst["unwind"]["fullName"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	AssertThat[inst["target"]]["isNotEqualTo", None];
	StringJoin[
		If[inst["hasTarget"],
			inst["target"]["lhsToString"] <> " = ",
			"unknown"
		],
		"Invoke ",
		inst["function"]["toString"],
		" ",
		"[",
		Riffle[Map[#["toString"]&, inst["arguments"]], ", "],
		"]",
		" to -> ", inst["to"]["fullName"],
		" unwind -> ", inst["unwind"]["fullName"]
	]
)

format[ self_, shortQ_:True] :=
	"InvokeInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		StyleBox[self["function"]["makePrettyPrintBoxes"], Bold, $OperatorColor],
		"[",
		Sequence@@Riffle[Map[#["makePrettyPrintBoxes"]&, self["arguments"]], ", "],
		"]",
		" ",
		"to", "\[LeftArrow]", self["to"]["fullName"],
		"unwind", "\[LeftArrow]", self["unwind"]["fullName"]
	}]

	

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> InvokeInstructionClass,
	"name" -> "InvokeInstruction",
	"predicate" -> InvokeInstructionQ,
	"constructor" -> CreateInvokeInstruction,
	"deserialize" -> Deserialize,
	"instance" -> InvokeInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]


End[]

EndPackage[]
