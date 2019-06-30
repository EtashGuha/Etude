BeginPackage["Compile`Core`IR`Instruction`BranchInstruction`"]

BranchInstruction;
BranchInstructionClass;
CreateBranchInstruction;
BranchInstructionQ;

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
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
BranchInstructionClass = DeclareClass[
	BranchInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"fixBasicBlocks" -> (fixBasicBlocks[Self, #]&),
		"usedVariables" -> Function[{},
		    If[Self["isConditional"],
		        Select[Prepend[Self["operands"], Self["condition"]], VariableQ],
		        {}
		    ]
		],
		"definedVariable" -> Function[{}, None],
		"isUnconditional" -> Function[{}, Length[Self["operands"]] === 1],
		"isConditional" -> Function[{}, Length[Self["operands"]] === 2],
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
			"_instructionName" -> "BranchInstruction",
			"_visitInstructionName" -> "visitBranchInstruction",
			"condition" -> None,
			"operands" -> {}
		}
	],
	Predicate -> BranchInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateBranchInstruction[operands_:{}, expr_:None]:=
	CreateBranchInstruction[operands, None, expr]
CreateBranchInstruction[operands0_:{}, cond_:None, expr_:None]:=
	Module[{operands = operands0, inst},
		If[BasicBlockQ[operands0],
			operands = {operands}
		]; 
		AssertThat["There should be 3 operands",
			operands]["named", "operands"]["satisfies", (Length[#] <= 3)&];
		AssertThat["The condition should be either None (for unconditional branch) or a value",
			cond]["named", "condition"]["satisfiesAnyOf", {# === None&, VariableQ, ConstantValueQ}];
		AssertThat["An unconditional branch has only one operand (and condition is None). "<>
				   "The conditional branch has 2 operands and has a condition.",
			operands]["named", "operands"]["satisfies", (
				(Length[#] === 2 && (VariableQ[cond] || ConstantValueQ[cond])) ||
			   (Length[#] === 1 && cond === None) ||
			   # === {})&
			];
		AssertThat["All operands are basic blocks",
			operands]["named", "operands"]["elementsSatisfy", BasicBlockQ[#] || MatchQ[#, "BasicBlockID"[_Integer]] &];
		
		inst = CreateObject[
			BranchInstruction,
			<|
				"condition" -> cond,
				"operands" -> operands,
				"mexpr" -> expr
			|>
		];
		inst
	]

CreateBranchInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateBranchInstruction ", args}]
	
Deserialize[ env_, data_] :=
	Module[ {cond, basicBlockTargets},
		cond = env["getElementNoDeserialize", data, "condition"];
		If[ cond =!= None,
			cond = env["deserialize", cond]];
		basicBlockTargets = 
			If[env["isClone"],
				Map[ "BasicBlockID"[#["id"]]&, data["operands"]],
				env["getElementNoDeserialize", data, "basicBlockTargets"]];
		CreateBranchInstruction[basicBlockTargets, cond, None]
	]
	

(**************************************************)
(**************************************************)

(*
  Called when deserializing and all the BasicBlocks have been fixed up.
*)
fixBasicBlocks[self_, env_] :=
	Module[ {operands, bb},
		operands = self["operands"];
		operands = Map[ (
			bb = env["getBasicBlock", #1];
			If[ !BasicBlockQ[bb],
				ThrowException[CompilerException[{"Cannot find BasicBlock ", #1}]]
			];
			bb
			)&, operands];
		self["setOperands", operands];
	]



serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["condition"] = self["condition"];
		If[ data["condition"] =!= None,
			data["condition"] = data["condition"]["serialize", env]];
		data["basicBlockTargets"] = Map[ "BasicBlockID"[#["id"]]&, self["operands"]];
		"Instruction"[data]
	]


	
(**************************************************)

icon := Graphics[Text[
	Style["Branch\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]

getBBBoxes[bb_] :=
	"BB[ " <> bb["name"] <> "(" <> ToString[bb["id"]] <> ")]"

getBBBoxes["BasicBlockID"[id_]] :=
	"BB[ID-(" <> ToString[id] <> ")]"

toBoxes[inst_, fmt_]  := 
	BoxForm`ArrangeSummaryBox[
		"BranchInstruction",
		inst["toString"],
  		icon,
  		If[inst["isConditional"],
  			{
				BoxForm`SummaryItem[{"id: ", inst["id"]}],
	  		    	BoxForm`SummaryItem[{"condition: ", inst["condition"]}],
	  		    	BoxForm`SummaryItem[{"then: ", getBBBoxes[ First[inst["operands"]]]}],
	  		    	BoxForm`SummaryItem[{"then: ", getBBBoxes[ Last[inst["operands"]]]}]
 			},
  			{
				BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    		BoxForm`SummaryItem[{"jump: ", getBBBoxes[ First[inst["operands"]]]}]
  			}
  		],
  		{}, 
  		fmt
  	]

bbLabel[bb_?BasicBlockQ] :=
	StringJoin[bb["name"], "(", ToString[bb["id"]], ")"]

bbLabel["BasicBlockID"[id_]] :=
	StringJoin["ID-", "(", ToString[id], ")"]

toString[inst_] := (
	Assert[inst["isConditional"] || inst["isUnconditional"]];
	If[inst["isConditional"],
		StringJoin[
			InstructionNameText["Branch"],
			" ",
			inst["condition"]["toString"],
			", ",
			LabelText[bbLabel[First[inst["operands"]]]],
			", ",
			LabelText[bbLabel[Last[inst["operands"]]]]
		],
		StringJoin[
			InstructionNameText["Jump"],
			" ",
			LabelText[bbLabel[First[inst["operands"]]]]
		]
	]
)

format[ self_, shortQ_:True] :=
	"BranchInstruction " <> self["toString"]

makePrettyPrintBoxes[self_ /; self["isConditional"]] :=
	RowBox[{
	    UnderscriptBox[
	        StyleBox[bbLabel[self["getOperand", 1]], Bold, $LabelColor],
	        StyleBox["\[LowerLeftArrow]", Bold, Small]
		],
		" ",
	    self["condition"]["makePrettyPrintBoxes"],
	    " ",
	    UnderscriptBox[
	        StyleBox[bbLabel[self["getOperand", 2]], Bold, $LabelColor],
	        StyleBox["\[LowerRightArrow]", Bold, Small]
		]
	}]

makePrettyPrintBoxes[self_ /; self["isUnconditional"]] :=
	RowBox[{
	    UnderscriptBox[
	        StyleBox[bbLabel[self["getOperand", 1]], Bold, $LabelColor],
	        StyleBox["\[DownArrow]", Bold]
		]
	}]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> BranchInstructionClass,
	"name" -> "BranchInstruction",
	"predicate" -> BranchInstructionQ,
	"constructor" -> CreateBranchInstruction,
	"deserialize" -> Deserialize,
	"instance" -> BranchInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]


End[]

EndPackage[]
