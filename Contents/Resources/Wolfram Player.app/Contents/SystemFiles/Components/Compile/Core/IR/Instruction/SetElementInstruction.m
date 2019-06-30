BeginPackage["Compile`Core`IR`Instruction`SetElementInstruction`"]

SetElementInstruction;
SetElementInstructionClass;
CreateSetElementInstruction;
SetElementInstructionQ;

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
  Target is the object which is modified
  Source is the new value to insert into target
  Offset is the list of indices in the target where value should be inserted
  
  Note that this doesn't define any variables (hence definedVariable returns None).
*)


RegisterCallback["DeclareCompileClass", Function[{st},
SetElementInstructionClass = DeclareClass[
	SetElementInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, Select[Self["operands"], VariableQ]],
		"definedVariable" -> Function[{}, None],
		"operands" -> Function[{},
			Join[{Self["target"], Self["source"]}, Self["offset"]]
		],
		"getOperand" -> Function[{idx}, Self["operands"][[idx]]],
		"setOperand" -> Function[{idx, val}, setOperand[Self, idx, val]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "SetElementInstruction",
			"_visitInstructionName" -> "visitSetElementInstruction",
			"target", (** The set element is done in place, so target is modified *)
			"offset",
			"source",
			"operator"
		}
	],
	Predicate -> SetElementInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


setOperand[self_, idx_, val_] :=
    Module[ {list},
        Which[ 
        	idx === 1,
            	self["setTarget", val],
        	idx === 2,
            	self["setSource", val],
            True,
            	list = self["offset"];
            	If[ Length[list] < idx-2,
                    ThrowException[{"setOperand index ", idx, "is out of range for GetElementInstruction"}]
            	];
            	list = ReplacePart[list, idx-2 -> val];
            	self["setOffset", list]
        ]
    ]


CreateSetElementInstruction[trgt0_, offset0_, source_, operatorIn_:None, expr_:None] :=
	Module[{offset, trgt, operator = operatorIn},
		offset = If[ListQ[offset0],
			offset0,
			{offset0}
		];
		If[ operator === None,
			operator = CreateConstantValue[Native`SetElement]
		];
		Assert[(trgt0 === None) || VariableQ[trgt0] || StringQ[trgt0]];
		Assert[source === None || VariableQ[source] || ConstantValueQ[source]];
		Assert[ConstantValueQ[operator]];
		If[StringQ[trgt0],
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0
		];
		CreateObject[
			SetElementInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"offset" -> offset,
				"mexpr" -> expr,
				"operator" -> operator
			|>
		]
	]

CreateSetElementInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateSetElementInstruction ", args}]

Deserialize[ env_, data_] :=
	Module[ {trg, src, offset, oper, inst},
		trg = env["getElement", data, "target"];
		src = env["getElement", data, "source"];
		offset = env["getElementList", data, "offset"];
		oper = env["getElement", data, "operator"];
		inst = CreateSetElementInstruction[trg, offset, src, oper]
	]
	

(**************************************************)
(**************************************************)
serialize[self_, env_] :=
	Module[ {data},
		data = self["serializeBaseNoOperands", env];
		data["source"] = self["source"]["serialize", env];
		data["offset"] = Map[ #["serialize", env]&, self["offset"]];
		data["operator"] = self["operator"]["serialize", env];
		"Instruction"[data]
	]

	
(**************************************************)

icon := Graphics[Text[
	Style["SetElem\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"SetElementInstruction",
		inst,
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"offset: ", inst["offset"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}],
  		    BoxForm`SummaryItem[{"operator: ", inst["operator"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		inst["target"]["lhsToString"],
		InstructionNameText["[["],
		StringRiffle[#["toString"]& /@ inst["offset"], ", "],
		InstructionNameText["]]"],
		" = ",
		inst["source"]["toString"],
		If[ConstantValueQ[inst["operator"]],
			GrayText["\t\t  (* ", ToString[inst["operator"]["value"]], " *)"],
			""
		]
	]

format[ self_, shortQ_:True] :=
	"SetElementInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"[[",
		RowBox[
			Riffle[
				#["makePrettyPrintBoxes"]& /@ self["offset"],
				", "
			]
		],
		"]]",
		"\[LeftArrow]",
		self["source"]["makePrettyPrintBoxes"]
	}]



(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> SetElementInstructionClass,
	"name" -> "SetElementInstruction",
	"predicate" -> SetElementInstructionQ,
	"constructor" -> CreateSetElementInstruction,
	"deserialize" -> Deserialize,
	"instance" -> SetElementInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
