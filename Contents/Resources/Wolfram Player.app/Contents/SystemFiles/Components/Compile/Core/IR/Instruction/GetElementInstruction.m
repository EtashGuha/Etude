BeginPackage["Compile`Core`IR`Instruction`GetElementInstruction`"]

GetElementInstruction;
GetElementInstructionClass;
CreateGetElementInstruction;
GetElementInstructionQ;

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
GetElementInstructionClass = DeclareClass[
	GetElementInstruction,
	<|
		"serialize" -> (serialize[Self, #]&),
		"usedVariables" -> Function[{}, Select[Self["operands"], VariableQ]],
		"definedVariable" -> Function[{}, Self["target"]],
		"operands" -> Function[{},
			Prepend[Self["offset"], Self["source"]]
		],
		"getOperand" -> Function[{idx}, Self["operands"][[idx]]],
		"setOperand" -> Function[{idx, val}, setOperand[Self, idx, val]],
		"setOperands" -> Function[{opers}, 
			If[ Length[opers] =!= 2, 
					Throw[], 
					Self["setSource", Part[opers, 1]];
					Self["setOffset", Drop[opers,1]];
			]
		],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	Join[
		InstructionFields,
		{
			"_instructionName" -> "GetElementInstruction",
			"_visitInstructionName" -> "visitGetElementInstruction",
			"target",
			"offset",
			"source",
			"operator"
		}
	],
	Predicate -> GetElementInstructionQ,
	Extends -> {
		InstructionTraits,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


setOperand[self_, idx_, val_] :=
    Module[ {list},
        If[ idx === 1,
            self["setSource", val],
            list = self["offset"];
            If[ Length[list] < idx-1,
                ThrowException[{"setOperand index ", idx, "is out of range for GetElementInstruction"}]
            ];
            list = ReplacePart[list, idx-1 -> val];
            self["setOffset", list]
        ]
    ]

CreateGetElementInstruction[trgt0_, source_, offset0_, operatorIn_:None, expr_:None]:=
	Module[{offset, trgt, operator = operatorIn},
		If[ operator === None,
			operator = CreateConstantValue[Native`GetElement]];
		offset = If[ListQ[offset0],
			offset0,
			{offset0}
		];
		Assert[(trgt0 === None) || VariableQ[trgt0] || StringQ[trgt0]];
		Assert[source === None || VariableQ[source]];
		Assert[ConstantValueQ[operator]];
		If[ StringQ[trgt0], 
			trgt = CreateVariable[];
			trgt["setName", trgt0],
			trgt = trgt0];
		CreateObject[
			GetElementInstruction,
			<|
				"target" -> trgt,
				"source" -> source,
				"offset" -> offset,
				"mexpr" -> expr,
				"operator" -> operator
			|>
		]
	]

CreateGetElementInstruction[args___] :=
	ThrowException[{"Invalid arguments to CreateGetElementInstruction ", args}]
		
Deserialize[ env_, data_] :=
	Module[ {trg, src, offset, oper},
		trg = env["getElement", data, "target"];
		src = env["getElement", data, "source"];
		offset = env["getElementList", data, "offset"];
		oper = env["getElement", data, "operator"];
		CreateGetElementInstruction[trg, src, offset, oper]
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
	Style["GetElem\nInstr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[inst_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"GetElementInstruction",
		inst["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"id: ", inst["id"]}],
  		    BoxForm`SummaryItem[{"target: ", inst["target"]}],
  		    BoxForm`SummaryItem[{"source: ", inst["source"]}],
  		    BoxForm`SummaryItem[{"offset: ", inst["offset"]}],
  		    BoxForm`SummaryItem[{"operator: ", inst["operator"]}]
  		},
  		{}, 
  		fmt
  	]

toString[inst_] :=
	StringJoin[
		inst["target"]["lhsToString"],
		" = ",
		inst["source"]["toString"],
		InstructionNameText["[["],
		StringRiffle[#["toString"]& /@ inst["offset"], ", "],
		InstructionNameText["]]"],
		If[ConstantValueQ[inst["operator"]],
			GrayText["\t\t  (* ", ToString[inst["operator"]["value"]], " *)"],
			""
		]
	]

format[ self_, shortQ_:True] :=
	"GetElementInstruction " <> self["toString"]

makePrettyPrintBoxes[self_] :=
	RowBox[{
		self["target"]["makePrettyPrintBoxes"],
		"\[LeftArrow]",
		self["source"]["makePrettyPrintBoxes"],
		"[[",
		RowBox[
			Riffle[
				#["makePrettyPrintBoxes"]& /@ self["offset"],
				", "
			]
		],
		"]]"
	}]


(**************************************************)

RegisterCallback["RegisterInstruction", Function[{st},
RegisterInstruction[<|
	"class" -> GetElementInstructionClass,
	"name" -> "GetElementInstruction",
	"predicate" -> GetElementInstructionQ,
	"constructor" -> CreateGetElementInstruction,
	"deserialize" -> Deserialize,
	"instance" -> GetElementInstruction,
	"properties" -> {
	    "visitingRequired"
	}
|>] 
]]

End[]

EndPackage[]
