BeginPackage["Compile`Core`IR`Instruction`LambdaInstruction`"]

LambdaInstruction;
LambdaInstructionClass;
CreateLambdaInstruction;
LambdaInstructionQ;

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
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
LambdaInstructionClass = DeclareClass[
	LambdaInstruction,
	<|
		"usedVariables" -> Function[{}, {}],
		"definedVariable" -> Function[{}, Self["target"]],
		"serialize" -> (serialize[Self, #]&),
        "makePrettyPrintBoxes" -> Function[{},
            RowBox[{
                Self["target"]["makePrettyPrintBoxes"],
                "=",
                "\[Lambda]",
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
			"_instructionName" -> "LambdaInstruction",
			"_visitInstructionName" -> "visitLambdaInstruction",
			"target",
			"source"
		}
	],
	Predicate -> LambdaInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateLambdaInstruction[trgt_, source_, expr_:None]:=
	Module[{},
		Assert[VariableQ[trgt]];
		Assert[StringQ[source] || SymbolQ[source] || ConstantValueQ[ source]];
		CreateObject[
			LambdaInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"mexpr" -> expr
			|>
		]
	]
CreateLambdaInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateLambdaInstruction ", args}]


serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["source"] = self["source"]["serialize", env];
		"Instruction"[data]
	]



Deserialize[ env_, data_] :=
	Module[ {trg, src},
		trg = env["getElement", data, "target"];
		src = env["getElement", data, "source"];
		CreateLambdaInstruction[trg, src]
	]


(**************************************************)

icon := Graphics[Text[
	Style["Lambda\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LambdaInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
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
		InstructionNameText["\[Lambda]"],
		"(",
		If[ConstantValueQ[inst["source"]],
			inst["source"]["toString"],
			ToString[inst["source"]]
		],
		")"
	]
)

format[ self_, shortQ_:True] :=
	"LambdaInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> LambdaInstructionClass,
	"name" -> "LambdaInstruction",
	"predicate" -> LambdaInstructionQ,
	"deserialize" -> Deserialize,
	"constructor" -> CreateLambdaInstruction,
	"instance" -> LambdaInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
