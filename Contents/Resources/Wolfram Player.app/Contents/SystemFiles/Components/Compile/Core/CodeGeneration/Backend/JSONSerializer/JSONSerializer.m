
BeginPackage["Compile`Core`CodeGeneration`Backend`JSONSerializer`"]

JSONSerializerPass

Begin["`Private`"]

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Instruction`BinaryInstruction`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CompareInstruction`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Instruction`GetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`GetFieldInstruction`"]
Needs["Compile`Core`IR`Instruction`InertInstruction`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`IR`Instruction`LambdaInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadInstruction`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`IR`Instruction`NewRecordInstruction`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`IR`Instruction`RecordExtendInstruction`"]
Needs["Compile`Core`IR`Instruction`RecordRestrictInstruction`"]
Needs["Compile`Core`IR`Instruction`RecordSelectInstruction`"]
Needs["Compile`Core`IR`Instruction`ReturnInstruction`"]
Needs["Compile`Core`IR`Instruction`SelectInstruction`"]
Needs["Compile`Core`IR`Instruction`SetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`SetFieldInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`StoreInstruction`"]
Needs["Compile`Core`IR`Instruction`TypeCastInstruction`"]
Needs["Compile`Core`IR`Instruction`UnaryInstruction`"]
Needs["Compile`Core`IR`Instruction`UnreachableInstruction`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`LoadGlobalInstruction`"]


run[pm_, opts_] :=
	Module[{fms = {}},
		pm["scanFunctionModules",
			(
				AppendTo[fms,
					<|
						"kind" -> "FunctionModule",
						"id" -> #["id"],
						"name" -> #["getName"],
						"basicBlocks" -> serialize[#],
						"arguments" -> serializeFMArguments[#],
						"result" -> serializeFMResult[#]
					|>
				]
			)&
		];
		pm["setProperty", "JSON" -> ExportString[fms, "JSON"]]
	]
	
serializeFMArguments[fm_] :=
	serialize /@ fm["arguments"]
serializeFMResult[fm_] :=
	serialize[fm["result"]]
	
serialize[fm_?FunctionModuleQ] :=
	With[{bbs = fm["reversePostOrder"]},
		<|
			"kind" -> "BasicBlock",
			"name" -> #["name"],
			"fullName" -> #["fullName"],
			"id" -> #["id"],
			"instructions" -> serialize[#],
			"children" -> Map[
				Function[{child}, child["fullName"]],
				#["getChildren"]
			],
			"parents" -> Map[
				Function[{parent}, parent["fullName"]],
				#["getParents"]
			]
		|>& /@ bbs
	];

serialize[bb_?BasicBlockQ] :=
	With[{s = serialize[#]},
		Join[
			<|
				"id" -> serialize[#["id"]],
				"mexpr" -> serialize[#["mexpr"]]
			|>,
			If[s === Nothing,
				<||>,
				s
			]
		]
	]& /@ bb["getInstructions"]

serialize[inst_?BinaryInstructionQ] :=
	<|
		"kind" -> "BinaryInstruction",
		"target" -> serialize[inst["target"]],
		"operator" -> serialize[inst["operator"]],
		"operands" -> serialize /@ inst["operands"]
	|>
serialize[inst_?BranchInstructionQ] :=
	If[inst["isConditional"],
		<|
			"kind" -> "BranchInstruction",
			"id" -> serialize[inst["id"]],
			"condition" -> serialize[inst["condition"]],
			"operands" -> (#["fullName"]& /@ inst["operands"])
		|>,
		<|
			"kind" -> "BranchInstruction",
			"id" -> serialize[inst["id"]],
			"operand" -> inst["getOperand", 1]["fullName"]
		|>
	]
serialize[inst_?CallInstructionQ] :=
	<|
		"kind" -> "CallInstruction",
		"target" -> serialize[inst["target"]],
		"function" -> serialize[inst["function"]],
		"operands" -> serialize /@ inst["operands"]
	|>
serialize[inst_?TypeCastInstructionQ] :=
	<|
		"kind" -> "TypeCastInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?CompareInstructionQ] :=
	<|
		"kind" -> "CompareInstruction",
		"target" -> serialize[inst["target"]],
		"operator" -> serialize[inst["operator"]],
		"operands" -> serialize /@ inst["operands"]
	|>
serialize[inst_?CopyInstructionQ] :=
	<|
		"kind" -> "CopyInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?GetElementInstructionQ] :=
	<|
		"kind" -> "GetElementInstruction",
		"target" -> serialize[inst["target"]],
		"offset" -> serialize /@ inst["offset"],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?GetFieldInstructionQ] :=
	<|
		"kind" -> "GetFieldInstruction",
		"target" -> serialize[inst["target"]],
		"field" -> serialize[inst["field"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?InertInstructionQ] :=
	<|
		"kind" -> "InertInstruction",
		"target" -> serialize[inst["target"]],
		"head" -> serialize[inst["head"]],
		"arguments" -> serialize /@ inst["arguments"]
	|>
serialize[inst_?LambdaInstructionQ] :=
	<|
		"kind" -> "LambdaInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?LoadArgumentInstructionQ] :=
	<|
		"kind" -> "LoadArgumentInstruction",
		"target" -> serialize[inst["target"]],
		"index" -> serialize[inst["index"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?LoadGlobalInstructionQ] :=
	<|
		"kind" -> "LoadGlobalInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?LoadInstructionQ] :=
	<|
		"kind" -> "LoadInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?NewRecordInstructionQ] :=
	<|
		"kind" -> "NewRecordInstruction",
		"target" -> serialize[inst["target"]],
		"type" -> serialize[inst["type"]]
	|>
serialize[inst_?RecordExtendInstructionQ] :=
	<|
		"kind" -> "RecordExtendInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]],
		"label" -> serialize[inst["label"]],
		"value" -> serialize[inst["value"]]
	|>
serialize[inst_?RecordRestrictInstructionQ] :=
	<|
		"kind" -> "RecordRestrictInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]],
		"arguments" -> serialize /@ inst["arguments"]
	|>
serialize[inst_?RecordSelectInstructionQ] :=
	<|
		"kind" -> "RecordSelectInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]],
		"arguments" -> serialize /@ inst["arguments"]
	|>
serialize[inst_?SelectInstructionQ] :=
	<|
		"kind" -> "SelectInstruction",
		"target" -> serialize[inst["target"]],
		"condition" -> serialize[inst["condition"]],
		"operands" -> serialize /@ inst["operands"]
	|>
serialize[inst_?SetElementInstructionQ] :=
	<|
		"kind" -> "SetElementInstruction",
		"target" -> serialize[inst["target"]],
		"offset" -> serialize /@ inst["offset"],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?SetFieldInstructionQ] :=
	<|
		"kind" -> "SetFieldInstruction",
		"target" -> serialize[inst["target"]],
		"field" -> serialize[inst["field"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?StackAllocateInstructionQ] :=
	<|
		"kind" -> "StackAllocateInstruction",
		"target" -> serialize[inst["target"]]
	|>
serialize[inst_?StoreInstructionQ] :=
	<|
		"kind" -> "StoreInstruction",
		"target" -> serialize[inst["target"]],
		"source" -> serialize[inst["source"]]
	|>
serialize[inst_?UnaryInstructionQ] :=
	<|
		"kind" -> "UnaryInstruction",
		"target" -> serialize[inst["target"]],
		"operator" -> serialize[inst["operator"]],
		"operand" -> serialize[inst["operand"]]
	|>
serialize[inst_?UnreachableInstructionQ] :=
	<|
		"kind" -> "UnreachableInstruction"
	|>
serialize[inst_?PhiInstructionQ] :=
	<|
		"kind" -> "PhiInstruction",
		"target" -> serialize[inst["target"]],
		"sourceBasicBlocks" -> (#["fullName"]& /@ inst["getSourceBasicBlocks"]),
		"sourceVariables" -> (serialize /@ inst["getSourceVariables"])
	|>
serialize[inst_?ReturnInstructionQ] :=
	<|
		"kind" -> "ReturnInstruction",
		"value" -> If[inst["hasValue"],
			serialize[inst["value"]],
			Nothing
		]
	|>
serialize[_?LabelInstructionQ] :=
	<|
		"kind" -> "LabelInstruction"
	|>
	
serialize[const_?ConstantValueQ] :=
	<|
		"kind" -> "ConstantValue",
		"id" -> serialize[const["id"]],
		"value" -> With[{val = const["value"]},
			ToString[
				If[IntegerQ[val],
					val,
					SetPrecision[val, MachinePrecision]
				],
				CForm,
				PageWidth -> Infinity
			]
		],
		"type" -> serialize[const["type"]] 
	|>
serialize[var_?VariableQ] :=
	<|
		"kind" -> "Variable",
		"id" -> serialize[var["id"]],
		"name" -> var["name"],
		"type" -> serialize[var["type"]] 
	|>

serialize[typ0_?TypeQ] :=
	With[{typ = ReifyType[typ0]},
		<|
			"kind" -> "Type",
			"id" -> typ["id"],
			"name" -> typ["name"]
		|>
	]
serialize[s_?StringQ] :=
	s
serialize[s_?NumberQ] :=
	s
serialize[s_Symbol] :=
	SymbolName[s]
serialize[s_?MExprQ] :=
	s["toString"]
serialize[___] :=
	Nothing


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"JSONSerializer",
	"The pass creates a JSON serialization of the IR"
];

JSONSerializerPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

(*RegisterPass[JSONSerializerPass]*)
]]

End[]

EndPackage[]
