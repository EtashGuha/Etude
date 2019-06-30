BeginPackage["Compile`Core`IR`Instruction`StackAllocateInstruction`"]

StackAllocateInstruction;
StackAllocateInstructionClass;
CreateStackAllocateInstruction;
StackAllocateInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
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
StackAllocateInstructionClass = DeclareClass[
	StackAllocateInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, If[ VariableQ[Self["size"]], {Self["size"]}, {}]],
		"definedVariable" -> Function[{}, Self["target"]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "StackAllocateInstruction",
			"_visitInstructionName" -> "visitStackAllocateInstruction",
			"operator",
			"target",
			"size"
		}
	],
	Predicate -> StackAllocateInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data["operator"] = self["operator"]["serialize", env];
		data["size"] = self["size"]["serialize", env];
		"Instruction"[data]
	]

Deserialize[ env_, data_] :=
	Module[ {trg, size, oper},
		trg = env["getElement", data, "target"];
		size = env["getElement", data, "size"];
		oper = env["getElement", data, "operator"];
		CreateStackAllocateInstruction[trg, size, oper]
	]



CreateStackAllocateInstruction[trgt0_, size_, operatorIn_:None, expr_:None] :=
	Module[ {trgt=trgt0, operator = operatorIn},
		Assert[VariableQ[trgt] || StringQ[trgt]];
		Assert[VariableQ[size] || ConstantValueQ[size]];
		If[ operator === None,
			operator = CreateConstantValue[Native`StackAllocate]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0]
		];
		CreateObject[
			StackAllocateInstruction,
			<|
				"operator" -> operator,
				"target" -> trgt,
				"mexpr" -> expr,
				"size" -> size
			|>
		]
	]
	
CreateStackAllocateInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateStackAllocateInstruction ", args}]

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
		"StackAllocateInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"size: ", inst["size"]}]
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
		InstructionNameText["StackAllocate"],
		" ",
		inst["size"]["toString"]
	]
)

format[ self_, shortQ_:True] :=
	"StackAllocateInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	FrameBox[
		OverscriptBox[
		    StyleBox[
		        self["target"]["makePrettyPrintBoxes"],
				Smaller
		    ],
		    StyleBox["Alloca", Tiny, Gray]
		]
	]
	
(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> StackAllocateInstructionClass,
	"name" -> "StackAllocateInstruction",
	"predicate" -> StackAllocateInstructionQ,
	"constructor" -> CreateStackAllocateInstruction,
	"deserialize" -> Deserialize,
	"instance" -> StackAllocateInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
