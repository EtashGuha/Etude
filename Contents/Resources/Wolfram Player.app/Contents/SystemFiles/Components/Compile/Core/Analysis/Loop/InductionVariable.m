
BeginPackage["Compile`Core`Analysis`Loop`InductionVariable`"]

InductionVariablePass

InductionVariable;
CreateInductionVariable;

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`Analysis`Loop`LoopNestingForest`"]
Needs["Compile`Core`IR`LoopInformation`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`Analysis`Loop`LoopInvariant`"]
Needs["Compile`Core`IR`Instruction`BinaryInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]


$InductionVariableClassifications = {
	"Basic",
	"Derived"
};	

isDefinedInLoop[loop_, var_?VariableQ] :=
	With[{
		defInst = var["def"]
	},
	With[{
		defBB = defInst["basicBlock"]
	},
		AnyTrue[loop["childrenBasicBlocks"], defBB["sameQ", #]&]
	]];
isDefinedInLoop[loop_, var_] :=
	False;

isLoopInvariant[loop_, c_?ConstantValueQ] :=
	True;
isLoopInvariant[loop_, var_] :=
	AnyTrue[loop["invariantVariables"], #["sameQ", var]&]; 

isInductionOperands[loop_, operands_] :=
	Which[
		Length[operands] =!= 2,
			False,
		isLoopInvariant[loop, First[operands]] && isLoopInvariant[loop, Last[operands]],
			False,
		isLoopInvariant[loop, First[operands]],
			True,
		isLoopInvariant[loop, Last[operands]],
			True,
		True,
			False
	];



isInductionOperator[oper_] :=
	ConstantValueQ[oper] &&
	MemberQ[{Plus, Subtract, Times, Divide}, oper["value"]];
	
isInduction[loop_, inst_?CopyInstructionQ] :=
	isInduction[loop, inst["source"]["def"]]
isInduction[loop_, inst_] :=
	(BinaryInstructionQ[inst] || CallInstructionQ[inst]) &&
	isInductionOperator[inst["operator"]] && 
	isInductionOperands[loop, inst["operands"]]
	
visitPhiInstruction[st_, inst_] :=
	With[{
		inductionVariables = st["inductionVariables"],
		loop = st["loop"]
	},
		Which[
			Length[inst["operands"]] =!= 2,
				Return[],
			isDefinedInLoop[loop, inst["getOperand", 1]],
				If[isInduction[loop, inst["getOperand", 1]["def"]],
					inductionVariables["appendTo", inst["getOperand", 1]]
				],
			isDefinedInLoop[loop, inst["getOperand", 2]],
				If[isInduction[loop, inst["getOperand", 2]["def"]],
					inductionVariables["appendTo", inst["getOperand", 2]]
				]
		]
	]
	
union[lst_?ReferenceQ] :=
	lst["set", union[lst["get"]]]
	
union[lst_?ListQ] :=
    SortBy[
    	   DeleteDuplicatesBy[lst, idOf],
    	   idOf
    ]
    
idOf[e_] := e["id"]

findInductionVariables[loop_] :=
	Module[{
		visitor,
		state = <|
			"loop" -> loop
		|>,
		changed = True,
		oldInductionVars,
		inductionVars = CreateReference[{}],
		childrenBBs = loop["childrenBasicBlocks"]
	},
		loop["scan",
       	   Function[{e},
       	   	    If[LoopInformationQ[e],
       	   	    	   findInductionVariables[e]
       	   	    ]
       	   ]
        ];
		visitor = CreateInstructionVisitor[
			state,
			<|	
				"visitPhiInstruction" -> visitPhiInstruction
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		While[changed,
			oldInductionVars = inductionVars["get"];
			visitor["setState", Append[
				visitor["getState"],
				"inductionVariables" -> inductionVars
			]];
			If[loop["header"] =!= None, (* the start node *)
				visitor["visit", loop["header"]]
			];
			visitor["visit", #]& /@ childrenBBs;
			inductionVars = union[inductionVars];
			changed = inductionVars["get"] =!= oldInductionVars;
		];
		loop["setInvariantVariables", inductionVars["get"]];
	];
	
run[fm_, opts_] :=
	With[{
		loopInfo = fm["getProperty", "loopinformation"]
	},
		findInductionVariables[loopInfo];
		fm
	]
	
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"InductionVariable",
	"This pass annotates loop information with the induction variables."
];

InductionVariablePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		DefPass,
		LoopNestingForestPass,
		LoopInvariantPass
	},
	"passClass" -> "Analysis"
|>];

RegisterPass[InductionVariablePass]
]]

End[]

EndPackage[]
