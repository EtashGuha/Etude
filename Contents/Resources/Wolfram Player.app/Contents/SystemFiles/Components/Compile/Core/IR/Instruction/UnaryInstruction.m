BeginPackage["Compile`Core`IR`Instruction`UnaryInstruction`"]

UnaryInstruction;
UnaryInstructionClass;
CreateUnaryInstruction;
UnaryInstructionQ;

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
UnaryInstructionClass = DeclareClass[
	UnaryInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, If[VariableQ[Self["source"]], {Self["source"]}, {}]],
		"definedVariable" -> Function[{}, Self["target"]],
		"source" -> Function[{}, Self["operand"]],
		"getOperand" -> Function[{idx}, Assert[idx == 1]; Self["operand"]],
		"setOperand" -> Function[{idx, val}, Assert[idx == 1]; SetData[Self["operand"], val]],
		"setOperands" -> Function[{ops}, Assert[Length[ops] == 1]; SetData[Self["operand"], First[ops]]],
		"operands" -> Function[{}, {Self["operand"]}], (* matches the interface for binary instruction *)
		"getOperands" -> Function[{}, {Self["operand"]}], (* matches the interface for binary instruction *)
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "UnaryInstruction",
			"_visitInstructionName" -> "visitUnaryInstruction",
			"target",
			"operator",
			"operand"
		}
	],
	Predicate -> UnaryInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateUnaryInstruction[trgt_:None, operator_:None, operand_:None, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[operand === None || VariableQ[operand] || ConstantValueQ[operand]];
		CreateObject[
			UnaryInstruction,
			<|
				"target" -> trgt,
				"operator" -> operator,
				"operand" -> operand,
				"mexpr" -> expr
			|>
		]
	)
CreateUnaryInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateUnaryInstruction ", args}]


Deserialize[ env_, data_] :=
	Module[ {trg, operand, operator},
		trg = env["getElement", data, "target"];
		operand = env["getElement", data, "operand"];
		operator = env["getElementNoDeserialize", data, "operator"];
		CreateUnaryInstruction[trg, operator, operand]
	]


serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data["operand"] = self["operand"]["serialize", env];
		data["operator"] = self["operator"];
		"Instruction"[data]
	]

(**************************************************)

icon := Graphics[Text[
	Style["Unary\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"UnaryInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"operator: ", inst["operator"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	Assert[inst["source"] =!= None];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText[ToString[inst["operator"]]],
		" ",
		inst["source"]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"UnaryInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		StyleBox[self["operator"], Bold, $OperatorColor],
		"[",
		self["source"]["makePrettyPrintBoxes"],
		"]"
	}]

	
(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> UnaryInstructionClass,
	"name" -> "UnaryInstruction",
	"predicate" -> UnaryInstructionQ,
	"constructor" -> CreateUnaryInstruction,
	"deserialize" -> Deserialize,
	"instance" -> UnaryInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
