BeginPackage["Compile`Core`IR`Instruction`LandingPadInstruction`"]

LandingPadInstruction;
LandingPadInstructionClass;
CreateLandingPadInstruction;
LandingPadInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionFields`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileClass", Function[{st},
LandingPadInstructionClass = DeclareClass[
	LandingPadInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, {}],
		"definedVariable" -> Function[{}, Self["target"]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "LandingPadInstruction",
			"_visitInstructionName" -> "visitLandingPadInstruction",
			"target"
		}
	],
	Predicate -> LandingPadInstructionQ,
	Extends -> {
		InstructionTraits,
		ClassPropertiesTrait
	}
]
]]


CreateLandingPadInstruction[trgt0_, expr_:None] :=
	Module[{trgt},
		Assert[(trgt0 === None) || StringQ[trgt0]  || VariableQ[trgt0]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];

		CreateObject[
			LandingPadInstruction,
			<|
				"target" -> trgt,
				"mexpr" -> expr
			|>
		]
	]
CreateLandingPadInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateLandingPadInstruction ", args}]
	
Deserialize[ env_, data_] :=
	Module[ {trg},
		trg = env["getElement", data, "target"];
		CreateLandingPadInstruction[trg]
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		"Instruction"[data]
	]


	
(**************************************************)

icon := Graphics[Text[
	Style["LandingPad\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LandingInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"function: ", inst["function"]}],
  		    BoxForm`SummaryItem[{"arguments: ", inst["arguments"]}],
  		    BoxForm`SummaryItem[{"to: ", inst["to"]["fullName"]}],
  		    BoxForm`SummaryItem[{"unwind: ", inst["unwind"]["fullName"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] := (
	AssertThat[inst["target"]]["isNotEqualTo", None];
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		"LandingPad[]"
	]
)

format[ self_, shortQ_:True] :=
	"LandingPadInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"=",
		"LandingPad[]"
	}]

	

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> LandingPadInstructionClass,
	"name" -> "LandingPadInstruction",
	"predicate" -> LandingPadInstructionQ,
	"constructor" -> CreateLandingPadInstruction,
	"deserialize" -> Deserialize,
	"instance" -> LandingPadInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]


End[]

EndPackage[]
