
BeginPackage["Compile`Core`Analysis`Loop`LoopInvariant`"]

LoopInvariantPass

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

isInvariantInstructionQ[inst_] :=
	With[{
		name = inst["_instructionName"]
	},
		MemberQ[
			{
				"CopyInstruction",
				"LambdaInstruction",
				"CompareInstruction",
				"BinaryInstruction",
				"CallInstruction",
				"UnaryInstruction",
				"LabelInstruction",
				"GetElementInstruction",
				"SetElementInstruction",
				"InertInstruction",
				"LambdaInstruction",
				"TypeCastInstruction",
				"SelectInstruction"
			},
			name
		]
	]

isDefinedByInvariantInstruction[loop_, var_?ConstantValueQ] :=
	True;
isDefinedByInvariantInstruction[loop_, var_?VariableQ] :=
	With[{
		defInst = var["def"]
	},
		isInvariantInstructionQ[defInst]
	];
isDefinedByInvariantInstruction[loop_, var_] :=
	False;
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

baseCaseQ[loop_, var_?ConstantValueQ] := True 
baseCaseQ[loop_, var_?VariableQ] := isDefinedByInvariantInstruction[loop, var] && !isDefinedInLoop[loop, var]


visitInstruction[st_, inst_] :=
	With[{
		invariants = st["invariants"],
		loop = st["loop"]
	},
	Module[{operands, vars},
		If[!inst["hasOperands"],
			If[inst["hasTarget"],
				invariants["appendTo", inst["target"]]
			];
			Return[]
		];
		operands = inst["operands"];
		Do[
			If[!ConstantValueQ[operand] && baseCaseQ[loop, operand],
				invariants["appendTo", operand]
			];
			,
			{operand, operands}
		];
		If[!isInvariantInstructionQ[inst],
			Return[]
		];
		vars = idOf /@ invariants["get"];
		If[inst["hasTarget"] && AllTrue[operands, (MemberQ[vars, idOf[#]] || ConstantValueQ[#])&],
			invariants["appendTo", inst["target"]]
		];
	]]


union[lst_?ReferenceQ] :=
	lst["set", union[lst["get"]]]
	
union[lst_?ListQ] :=
    SortBy[
    	   DeleteDuplicatesBy[lst, idOf],
    	   idOf
    ]
    
idOf[e_] := e["id"]

setInvariants[loop_] :=
	Module[{
		visitor,
		state = <|
			"loop" -> loop
		|>,
		changed = True,
		oldInvariantsVars,
		invariantsVars = CreateReference[{}],
		childrenBBs = loop["childrenBasicBlocks"]
	},
		visitor = CreateInstructionVisitor[
			state,
			<|	
				"visitInstruction" -> visitInstruction
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		While[changed,
			oldInvariantsVars = invariantsVars["get"];
			visitor["setState", Append[
				visitor["getState"],
				"invariants" -> invariantsVars
			]];
			If[loop["header"] =!= None, (* the start node *)
				visitor["visit", loop["header"]]
			];
			visitor["visit", #]& /@ childrenBBs;
			invariantsVars = union[invariantsVars];
			changed = invariantsVars["get"] =!= oldInvariantsVars;
		];
		loop["setInvariantVariables", invariantsVars["get"]];
		loop["scan",
       	   Function[{e},
       	   	    If[LoopInformationQ[e],
       	   	    	   setInvariants[e]
       	   	    ]
       	   ]
        ];
	];
	
run[fm_, opts_] :=
	With[{
		loopInfo = fm["getProperty", "loopinformation"]
	},
		setInvariants[loopInfo];
		fm
	]
	
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"LoopInvariant",
	"This pass computes the invariant variables for each loop.",
	"A variable used or defined within a loop is invariant with respect to the loop iff: \n" <>
	"   Base case: \n" <>
	"     - the variable is a constant \n" <>
	"     - the variable is not defined within the loop \n " <>
	"   Inductive case: \n " <>
	"     - it's a pure computation and all of its operands are loopinvariant \n " <>
	"     - it's a variable use whose single reaching def, and the rhs of that def is loop-invariant \n" <> 
	" we are conservative by making some instructions not introduce loop variance (e.g. load and store instructions)"
];

LoopInvariantPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		DefPass,
		LoopNestingForestPass
	},
	"passClass" -> "Analysis"
|>];

RegisterPass[LoopInvariantPass]
]]

End[]

EndPackage[]
