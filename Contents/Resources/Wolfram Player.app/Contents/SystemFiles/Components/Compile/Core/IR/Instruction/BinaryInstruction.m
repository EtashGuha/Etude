BeginPackage["Compile`Core`IR`Instruction`BinaryInstruction`"]

BinaryInstruction;
BinaryInstructionClass;
CreateBinaryInstruction;
BinaryInstructionQ


Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileAST`Language`ShortNames`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
BinaryInstructionClass = DeclareClass[
	BinaryInstruction,
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
			"_instructionName" -> "BinaryInstruction",
			"_visitInstructionName" -> "visitBinaryInstruction",
			"target",
			"operator",
			"operands" -> {}
		}
	],
	Predicate -> BinaryInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateBinaryInstruction[trgt0_:None, operator_:None, operands0_:{}, expr_:None]:=
	Module[{inst, operands, trgt},
	    operands = Flatten[{operands0}];
		AssertThat["Target must be either None or a variable",
			trgt0]["named", "target"]["satisfiesAnyOf", {# === None&, VariableQ, StringQ}];
		AssertThat["Operands must be a list of two elements",
			operands]["named", "operands"]["satisfiesAnyOf", {# === {}&, Length[#] === 2&}];
		AssertThat["Operands must be either variables or constant values",
			operands]["named", "operands"]["elementsSatisfyAnyOf", {VariableQ, ConstantValueQ}];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		
		inst = CreateObject[
			BinaryInstruction,
			<|
				"target" -> trgt,
				"operator" -> operator,
				"operands" -> operands,
				"mexpr" -> expr
			|>
		];
		inst
	]

CreateBinaryInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateBinaryInstruction ", args}]


Deserialize[ env_, data_] :=
	Module[ {trg, operands, operator},
		trg = env["getElement", data, "target"];
		operands = env["getElementList", data, "operands"];
		operator = env["getElement", data, "operator"];
		CreateBinaryInstruction[trg, operator, operands]
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
	Style["Binary\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"BinaryInstruction",
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
	"BinaryInstruction " <> self["toString"]

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
	"class" -> BinaryInstructionClass,
	"name" -> "BinaryInstruction",
	"predicate" -> BinaryInstructionQ,
	"constructor" -> CreateBinaryInstruction,
	"deserialize" -> Deserialize,
	"instance" -> BinaryInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]


End[]

EndPackage[]
