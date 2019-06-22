BeginPackage["Compile`Core`IR`Instruction`CompareInstruction`"]

CompareInstruction;
CompareInstructionClass;
CreateCompareInstruction;
CompareInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Language`ShortNames`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
CompareInstructionClass = DeclareClass[
	CompareInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, Select[Self["operands"], VariableQ]],
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
			"_instructionName" -> "CompareInstruction",
			"_visitInstructionName" -> "visitCompareInstruction",
			"target",
			"operator",
			"operands" -> {}
		}
	],
	Predicate -> CompareInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateCompareInstruction[trgt_:None, operator_:None, operands_:{}, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		Assert[operands === {} || Length[operands] === 2];
		Assert[AllTrue[operands, (VariableQ[#] || ConstantValueQ[#])&]];
		CreateObject[
			CompareInstruction,
			<|
				"target" -> trgt,
				"operator" -> operator,
				"operands" -> operands,
				"mexpr" -> expr
			|>
		]
	)
CreateCompareInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateCompareInstruction ", args}]
	
Deserialize[ env_, data_] :=
	Module[ {trg, operands, operator},
		trg = env["getElement", data, "target"];
		operands = env["getElementList", data, "operands"];
		operator = env["getElement", data, "operator"];
		CreateCompareInstruction[trg, operator, operands]
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data["operator"] = self["operator"]["serialize", env];
		"Instruction"[data]
	]



	
(**************************************************)

icon := Graphics[Text[
	Style["Compare\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"CompareInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"operator: ", inst["operator"]}],
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
		InstructionNameText[ToString[inst["operator"]]],
		" ",
		First[inst["operands"]]["toString"],
		", ",
		Last[inst["operands"]]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"CompareInstruction " <> self["toString"]

(**************************************************)



makePrettyPrintBoxes[self_] :=
	RowBox[Flatten[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		rhsFormat[self]
	}]]

rhsFormat[self_] /; KeyExistsQ[$SystemShortNames, self["operator"]] :=
	{
		self["getOperand", 1]["makePrettyPrintBoxes"],
		" ",
		StyleBox[$SystemShortNames[self["operator"]], Bold, $OperatorColor],
		" ",
		self["getOperand", 2]["makePrettyPrintBoxes"]
	}
	
rhsFormat[self_] /; !KeyExistsQ[$SystemShortNames, self["operator"]] :=
	{
		StyleBox[self["operator"], Bold, $OperatorColor],
		"[",
		self["getOperand", 1]["makePrettyPrintBoxes"],
		", ",
		self["getOperand", 2]["makePrettyPrintBoxes"],
		"]"
	}
	
(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> CompareInstructionClass,
	"name" -> "CompareInstruction",
	"predicate" -> CompareInstructionQ,
	"constructor" -> CreateCompareInstruction,
	"deserialize" -> Deserialize,
	"instance" -> CompareInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]


End[]

EndPackage[]
