BeginPackage["Compile`Core`IR`Instruction`LoadGlobalInstruction`"]

LoadGlobalInstruction;
LoadGlobalInstructionClass;
CreateLoadGlobalInstruction;
LoadGlobalInstructionQ;

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
LoadGlobalInstructionClass = DeclareClass[
	LoadGlobalInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, Select[{Self["source"]}, VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"getOperand" -> Function[{idx}, Assert[idx == 1]; Self["source"]],
		"setOperand" -> Function[{idx, val}, Assert[idx == 1]; Self["setSource", val]],
		"operands" -> Function[{}, {Self["source"]}], (* matches the interface for binary instruction *)
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "LoadGlobalInstruction",
			"_visitInstructionName" -> "visitLoadGlobalInstruction",
			"target",
			"source"
		}
	],
	Predicate -> LoadGlobalInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateLoadGlobalInstruction[trgt_:None, source_:None, expr_:None]:=
	(
		Assert[(trgt === None) || VariableQ[trgt]];
		CreateObject[
			LoadGlobalInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"mexpr" -> expr
			|>
		]
	)

CreateLoadGlobalInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateLoadGlobalInstruction ", args}]
	
(**************************************************)
(**************************************************)




icon := Graphics[Text[
	Style["LoadGlbl\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LoadGlobalInstruction",
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
		InstructionNameText["LoadGlobal"],
		" ",
		If[ObjectInstanceQ[inst["source"]],
			inst["source"]["toString"],
			ToString[inst["source"]]
		]
	]
)

format[ self_, shortQ_:True] :=
	"LoadGlobalInstruction " <> self["toString"]

valToString[s_?StringQ] := s
valToString[s_?ObjectInstanceQ] := s["toString"]
 
makePrettyPrintBoxes[self_] :=
	RowBox[{
	    self["target"]["makePrettyPrintBoxes"],
	    "\[LeftArrow]",
	    StyleBox[
	        StringTrim[valToString[self["source"]], "Global`"],
	        Bold,
	        $GlobalVariableColor
	    ]
	}]

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBase", env];
		data["target"] = self["target"]["serialize", env];
		"Instruction"[data]
	]
		
Deserialize[ env_, data_] :=
	Module[ {trg, operands},
		trg = env["getElement", data, "target"];
		operands = env["getElementList", data, "operands"];
		Assert[ListQ[operands] && Length[operands] === 1];
		CreateLoadGlobalInstruction[trg, First[operands]]
	]
	
(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> LoadGlobalInstructionClass,
	"name" -> "LoadGlobalInstruction",
	"predicate" -> LoadGlobalInstructionQ,
	"constructor" -> CreateLoadGlobalInstruction,
	"instance" -> LoadGlobalInstruction,
	"deserialize" -> Deserialize,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
