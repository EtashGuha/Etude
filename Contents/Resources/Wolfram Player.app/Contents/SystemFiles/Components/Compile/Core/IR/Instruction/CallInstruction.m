BeginPackage["Compile`Core`IR`Instruction`CallInstruction`"]

CallInstruction;
CallInstructionClass;
CreateCallInstruction;
CallInstructionQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
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
CallInstructionClass = DeclareClass[
	CallInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{},
		    Select[Prepend[Self["arguments"], Self["function"]], VariableQ]
		],
		"definedVariable" -> Function[{}, Self["target"]],
		"getArgument" -> Function[{idx}, Assert[0 < idx && idx <= Length[Self["arguments"]]]; Self["arguments"][[idx]]],
		"setArgument" -> Function[{idx, val}, Assert[0 < idx && idx <= Length[Self["arguments"]]]; SetData[Self["arguments"], ReplacePart[Self["arguments"], idx -> val]]],
		"getOperand" -> Function[{idx}, Self["getArgument", idx]],
		"setOperand" -> Function[{idx, val}, Self["setArgument", idx, val]],
		"setOperands" -> Function[{args}, Self["setArguments", args]],
		"operands" -> Function[{}, Self["arguments"]], (* matches the interface for binary instruction *)
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "CallInstruction",
			"_visitInstructionName" -> "visitCallInstruction",
			"target",
			"function",
			"arguments" -> {},
			"attributes" -> {}
		}
	],
	Predicate -> CallInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateCallInstruction[trgt0_:None, callFun_:None, args0_:{}, expr_:None] :=
	Module[{trgt, args = Flatten[{args0}]},
		Assert[(trgt0 === None) || StringQ[trgt0]  || VariableQ[trgt0]];
		Assert[(callFun === None) || VariableQ[callFun] || ConstantValueQ[callFun]];
		Assert[AllTrue[args, (VariableQ[#] || ConstantValueQ[#])&]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		CreateObject[
			CallInstruction,
			<|
				"target" -> trgt,
				"function" -> callFun,
				"arguments" -> args,
				"mexpr" -> expr
			|>
		]
	]
CreateCallInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateCallInstruction ", args}]
	
Deserialize[ env_, data_] :=
	Module[ {trg, fun, args},
		trg = env["getElement", data, "target"];
		fun = env["getElement", data, "function"];
		Which[
			ConstantValueQ[fun],
				Replace[fun["value"], {
					_String :> Null,
					Native`PrimitiveFunction[_String] :> Null,
					_Symbol :> Null,
					(* sym_Symbol :> ThrowException[{"Bad ConstantValue in CallInstruction:  " <>
                                                  Context[sym] <> ToString[sym] <>
                                                  "  (function names should be Strings)"}], *)
					_ :> ThrowException[{"Bad ConstantValue in CallInstruction function: ", fun}]
				}],
			VariableQ[fun],
				Null,
			True,
				ThrowException["Expected deserialized CallInstruction function to be a ConstantValue or Variable"]
		];
		args = env["getElementList", data, "arguments"];
		CreateCallInstruction[trg, fun, args]
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["function"] = self["function"]["serialize", env];
		data["arguments"] = Map[ #["serialize", env]&, self["arguments"]];
		"Instruction"[data]
	]


	
(**************************************************)

icon := Graphics[Text[
	Style["Call\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"CallInstruction",
		inst["toString"],
  		icon,
  		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"function: ", inst["function"]}],
  		    BoxForm`SummaryItem[{"arguments: ", inst["arguments"]}]
  		},
  		{}, 
  		fmt
  	]

loadClosureVariableCallQ[v_?ConstantValueQ] :=
	v["value"] === Native`LoadClosureVariable
loadClosureVariableCallQ[___] :=
	False
	
toString[inst_] := (
	AssertThat[inst["target"]]["isNotEqualTo", None];
	StringJoin[
		If[inst["hasTarget"],
			inst["target"]["lhsToString"] <> " = ",
			""
		],
		InstructionNameText["Call"],
		" ",
		inst["function"]["toString"],
		" ",
		"[",
		Riffle[Map[#["toString"]&, inst["arguments"]], ", "],
		"]",
		If[loadClosureVariableCallQ[inst["function"]],
			StringJoin[{
	           GrayText["\t\t  (* ClosureLoad "] ,
	           inst["target"]["getProperty", "aliasesVariable"]["toString"],
	           GrayText[" *)"] 
	        }],
			""
		],
		If[localCallQ[inst],
			GrayText["\t\t  (* Local Function Module Call *) "],
			""
		]
	]
)

localCallQ[inst_] :=
	inst["hasProperty", "localFunctionModuleCall"]

format[ self_, shortQ_:True] :=
	"CallInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		StyleBox[self["function"]["makePrettyPrintBoxes"], Bold, $OperatorColor],
		"[",
		Sequence@@Riffle[Map[#["makePrettyPrintBoxes"]&, self["arguments"]], ", "],
		"]"
	}]

	

(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> CallInstructionClass,
	"name" -> "CallInstruction",
	"predicate" -> CallInstructionQ,
	"constructor" -> CreateCallInstruction,
	"deserialize" -> Deserialize,
	"instance" -> CallInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]


End[]

EndPackage[]
