BeginPackage["Compile`Core`IR`Instruction`PhiInstruction`"]

PhiInstruction;
PhiInstructionClass;
CreatePhiInstruction;
PhiInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
PhiInstructionClass = DeclareClass[
	PhiInstruction,
	<|
		"usedVariables" -> Function[{},
			Select[Self["getSourceVariables"], VariableQ]
		],
		"definedVariable" -> Function[{}, Self["target"]],
		"getOperand" -> Function[{idx}, Self["source"]["get"][[idx, 2]]],
		"setOperand" -> Function[{idx, val},
			Module[{ref = Self["source"], sources = Self["source"]["get"]},
				sources[[idx, 2]] = val;
				ref["set", sources];
				val
			] 
		],
		"operands" -> Function[{}, Self["getSourceVariables"]], (* matches the interface for binary instruction *)
		"addSource" -> Function[{bb, var},
			Assert[BasicBlockQ[bb]];
			Assert[VariableQ[var] || ConstantValueQ[var]];
			Self["source"]["appendTo", {bb, var}]
		], 
		"getSourceVariables" -> Function[{},
		    With[{source = Self["source"]["get"]},
			        source[[All, 2]]
		    ]
		],
		"getSourceData" -> Function[{},
		    Self["source"]["get"]
		],
		"getSourceBasicBlocks" -> Function[{},
		    With[{source = Self["source"]["get"]},
		        source[[All, 1]]
		    ]
		],
		"setSourceBasicBlocks" -> Function[{newBBs},
		    Module[{source = Self["source"]["get"]},
		    	Assert[Length[source] === Length[newBBs]];
		        source[[All, 1]] = newBBs;
		        Self["source"]["set", source]
		    ]
		],
		"fixBasicBlocks" -> (fixBasicBlocks[Self, #]&),
		"serialize" -> (serialize[Self, #]&),
		"disposeExtra" -> Function[{}, disposeExtra[Self]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "PhiInstruction",
			"_visitInstructionName" -> "visitPhiInstruction",
			"target",
			"source"
		}
	],
	Predicate -> PhiInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreatePhiInstruction[trgt0_:None, source_:{}, expr_:None]:=
	Module[ {trgt},
		Assert[(trgt0 === None) || VariableQ[trgt0] || StringQ[trgt0]];
		Assert[ListQ[source]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		CreateObject[
			PhiInstruction,
			<|
				"target" -> trgt,
				"source" -> CreateReference[source],
				"mexpr" -> expr
			|>
		]
	]
	
CreatePhiInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreatePhiInstruction ", args}]

Deserialize[ env_, data_] :=
	Module[ {trg, source},
		trg = env["getElement", data, "target"];
		If[ env["isClone"],
			source = data["source"]["get"];
			source = Apply[ {"BasicBlockID"[#1["id"]], #2}&, source, {1}];
			,
			source = env["getElementNoDeserialize", data, "source"]];
		source = Apply[ {#1, env["deserialize", #2]}&, source, {1}];
		CreatePhiInstruction[trg, source]
	]

(*
  Called when deserializing and all the BasicBlocks have been fixed up.
*)
fixBasicBlocks[self_, env_] :=
	Module[ {src, bb},
		src = self["source"]["get"];
		src = Apply[ (
			bb = env["getBasicBlock", #1];
			If[ !BasicBlockQ[bb],
				ThrowException[CompilerException[{"Cannot find BasicBlock ", #1}]]
			];
			{bb, #2}
			)&, src, {1}];
		self["source"]["set", src];
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["source"] = Apply[ {"BasicBlockID"[#1["id"]], #2["serialize", env]}&, self["source"]["get"], {1}];
		"Instruction"[data]
	]

(**************************************************)

disposeExtra[self_] :=
	Module[{vars = self["getSourceVariables"]},
		Map[#["dispose"], vars];
		self["setSource", CreateReference[{}]];
	]


(**************************************************)

icon := Graphics[Text[
	Style["Phi\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"PhiInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"source: ", sourceToString[inst]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	Assert[inst["source"] =!= {}];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		InstructionNameText["Phi"],
		" ",
		sourceToString[inst]
	]
)

format[ self_, shortQ_:True] :=
	"PhiInstruction " <> self["toString"]

bbToString[bb_] :=
	StringJoin[bb["name"], "(",ToString[bb["id"]],")"]

bbToString["BasicBlockID"[id_]] :=
	StringJoin["ID-(",ToString[id],")"]

sourceToString[inst_] :=
	StringJoin[
	Riffle[
		Map[
			StringJoin[
				"[",
				LabelText[bbToString[#[[1]]]],
				", ",
				#[[2]]["toString"],
				"]"
			]&,
			inst["source"]["get"]
		],
		", "
	]]

formatSources[{bb_, var_}] :=
	OverscriptBox[
	    var["makePrettyPrintBoxes"],
	    StyleBox[bb["name"] <> "(" <> ToString[bb["id"]] <> ")", Small, Bold, $LabelColor]
	]

makePrettyPrintBoxes[self_] :=
	RowBox[Flatten[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		StyleBox["\[CurlyPhi]", Bold, Large, $OperatorColor],
		"[",
		Riffle[
		    formatSources /@ self["source"]["get"],
		    ","
		],
		"]"
	}]]

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> PhiInstructionClass,
	"name" -> "PhiInstruction",
	"predicate" -> PhiInstructionQ,
	"constructor" -> CreatePhiInstruction,
	"deserialize" -> Deserialize,
	"instance" -> PhiInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
