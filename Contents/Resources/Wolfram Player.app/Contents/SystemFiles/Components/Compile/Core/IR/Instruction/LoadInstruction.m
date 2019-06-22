BeginPackage["Compile`Core`IR`Instruction`LoadInstruction`"]

LoadInstruction;
LoadInstructionClass;
CreateLoadInstruction;
LoadInstructionQ;

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



(*
  A LoadInstruction is created by 
       Module,  new variables are initialized with a LoadInstruction (an uninitializedType if no assignment)
       Set, assignments to variables use a LoadInstruction, this is often removed later (except for PackedArrays
            which stick with a CopyInstruction)
       Operation, for binary/unary instructions if the args are constants then it evals the arg and adds 
       a load instruction,
          Compile[ {}, (5+6)],  will turn into 
             V1 = Load 11      
*)

RegisterCallback["DeclareCompileClass", Function[{st},
LoadInstructionClass = DeclareClass[
	LoadInstruction,
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
			    "\[LeftArrow]",
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
			"_instructionName" -> "LoadInstruction",
			"_visitInstructionName" -> "visitLoadInstruction",
			"target",
			"source",
			"operator"
		}
	],
	Predicate -> LoadInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
];
]]


serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data["source"] = self["source"]["serialize", env];
		data["operator"] = self["operator"]["serialize", env];
		"Instruction"[data]
	]

Deserialize[ env_, data_] :=
	Module[ {trg, src, oper},
		trg = env["getElement", data, "target"];
		src = env["getElement", data, "source"];
		oper = env["getElement", data, "operator"];
		CreateLoadInstruction[trg, src, oper]
	]


CreateLoadInstruction[trgt0_, source_, operatorIn_:None, expr_:None]:=
	Module[{trgt, operator = operatorIn},
		If[ operator === None,
			operator = CreateConstantValue[Native`Load]];
		Assert[(trgt0 === None) || StringQ[trgt0] || VariableQ[trgt0]];
		Assert[source === None || VariableQ[source] || ConstantValueQ[source]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		CreateObject[
			LoadInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"mexpr" -> expr,
				"operator" -> operator
			|>
		]
	];
	
CreateLoadInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateLoadInstruction ", args}];
	
(**************************************************)
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
		"LoadInstruction",
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
		InstructionNameText["Load"],
		" ",
		inst["source"]["toString"]
	]
);

format[ self_, shortQ_:True] :=
	"LoadInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> LoadInstructionClass,
	"name" -> "LoadInstruction",
	"predicate" -> LoadInstructionQ,
	"constructor" -> CreateLoadInstruction,
	"deserialize" -> Deserialize,
	"instance" -> LoadInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>]; 
]]

End[]

EndPackage[]
