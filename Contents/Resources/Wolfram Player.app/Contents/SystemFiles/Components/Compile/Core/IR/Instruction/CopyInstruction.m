BeginPackage["Compile`Core`IR`Instruction`CopyInstruction`"]

CopyInstruction;
CopyInstructionClass;
CreateCopyInstruction;
CopyInstructionQ;

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
CopyInstructionClass = DeclareClass[
	CopyInstruction,
	<|
		"usedVariables" -> Function[{}, Select[{Self["source"]}, VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"getOperand" -> Function[{idx}, Assert[idx == 1]; Self["source"]],
		"setOperand" -> Function[{idx, val}, Assert[idx == 1]; Self["setSource", val]],
		"operands" -> Function[{}, {Self["source"]}], (* matches the interface for binary instruction *)
		"serialize" -> (serialize[Self, #]&),
		"makePrettyPrintBoxes" -> Function[{},
			RowBox[{
			    Self["target"]["makePrettyPrintBoxes"],
			    "=",
			    "Copy",
			    " ",
			    Self["source"]["makePrettyPrintBoxes"]
			}]
		],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "CopyInstruction",
			"_visitInstructionName" -> "visitCopyInstruction",
			"target",
			"source"
		}
	],
	Predicate -> CopyInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
];
]]


CreateCopyInstruction[trgt0_:None, source_:None, expr_:None]:=
	Module[{trgt},
		Assert[(trgt0 === None) || StringQ[trgt0] || VariableQ[trgt0]];
		Assert[source === None || VariableQ[source] || ConstantValueQ[source]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		CreateObject[
			CopyInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"mexpr" -> expr
			|>
		]
	];
	
CreateCopyInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateCopyInstruction ", args}]


Deserialize[ env_, data_] :=
	Module[ {trg, src},
		trg = env["getElement", data, "target"];
		src = env["getElement", data, "source"];
		CreateCopyInstruction[trg, src]
	]

	
(**************************************************)
(**************************************************)

serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["source"] = self["source"]["serialize", env];
		"Instruction"[data]
	]


	
(**************************************************)

icon := Graphics[Text[
	Style["Load\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
];
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"CopyInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}]
  		},
  		{}, 
  		fmt
  	];

toString[inst_] := (
	Assert[inst["source"] =!= None];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["Copy"],
		" ",
		inst["source"]["toString"]
	]
);

format[ self_, shortQ_:True] :=
	"CopyInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> CopyInstructionClass,
	"name" -> "CopyInstruction",
	"predicate" -> CopyInstructionQ,
	"deserialize" -> Deserialize,
	"constructor" -> CreateCopyInstruction,
	"instance" -> CopyInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>]; 
]]


End[]

EndPackage[]
