BeginPackage["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]

LoadArgumentInstruction;
LoadArgumentInstructionClass;
CreateLoadArgumentInstruction;
LoadArgumentInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
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
LoadArgumentInstructionClass = DeclareClass[
	LoadArgumentInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, Select[{Self["source"]}, VariableQ]],
		"definesVariableQ" -> Function[{}, True],
		"definedVariable" -> Function[{}, Self["target"]],
		"getOperand" -> Function[{idx}, Assert[idx == 1]; Self["source"]],
		"setOperand" -> Function[{idx, val}, Assert[idx == 1]; Self["setSource", val]],
		"operands" -> Function[{}, {Self["source"]}], (* matches the interface for binary instruction *)
		"makePrettyPrintBoxes" -> Function[{},
			RowBox[{
			    Self["target"]["makePrettyPrintBoxes"],
			    "\[LeftArrow]",
			    "Argument",
			    " ",
			    StyleBox[Self["source"]["toString"], Bold, $VariableColor]
			}]
		],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "LoadArgumentInstruction",
			"_visitInstructionName" -> "visitLoadArgumentInstruction",
			"target",
			"index",
			"source",
			"compileQ"
		}
	],
	Predicate -> LoadArgumentInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateLoadArgumentInstruction[trgt_:None, index_:None, source_:None, expr_:None, compileQ_]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		CreateObject[
			LoadArgumentInstruction,
			<|
				"target" -> trgt,
				"index" -> index,
				"source" -> source,
				"mexpr" -> expr,
				"compileQ" -> compileQ
			|>
		]
	)
	
CreateLoadArgumentInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateLoadArgumentInstruction ", args}]
	
(**************************************************)
(**************************************************)

Deserialize[ env_, data_] :=
	Module[ {trg, index, compileQ},
		trg = env["getElement", data, "target"];
		index = env["getElementMExpr", data, "index"];
		compileQ = env["getElementNoDeserialize", data, "compileQ"];
		CreateLoadArgumentInstruction[trg, index, None, None, compileQ]
	]

(*
  Need to drop the operands,  they show up because hasOperand is True for this instruction.
*)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data = Delete[data, "operands"];
		data["index"] = self["index"]["data"];
		data["compileQ"] = self["compileQ"];
		"Instruction"[data]
	]


(**************************************************)

icon := Graphics[Text[
	Style["LoadArg\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LoadArgumentInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"index: ", inst["index"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}],
  		    BoxForm`SummaryItem[{"compileQ: ", inst["compileQ"]}]
  		},
  		{}, 
  		fmt
  	];
	
toString[inst_] := (
	(*AssertThat["The input source is not null", inst["source"]
	   ]["named", "source"
	   ]["isNotEqualTo", None
	];*)
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["LoadArgument"],
		" ",
		BoldGrayText[
			If[ inst["source"] === None, 
				inst["index"]["toString"], 
				inst["source"]["toString"]
	    		]
		]
	]
);

format[ self_, shortQ_:True] :=
	"LoadArgumentInstruction " <> self["toString"]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> LoadArgumentInstructionClass,
	"name" -> "LoadArgumentInstruction",
	"predicate" -> LoadArgumentInstructionQ,
	"constructor" -> CreateLoadArgumentInstruction,
	"deserialize" -> Deserialize,
	"instance" -> LoadArgumentInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
