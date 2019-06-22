BeginPackage["Compile`Core`Optimization`SimplifyExpression`"]


SimplifyExpressionPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



replaceInst[inst_, newInst_] := (
	newInst["moveAfter", inst];
	newInst["setId", inst["id"]];
	inst["unlink"]
)

simplifyBinary[state_, inst_] := (
	Switch[inst["operator"],
	    Plus,
	    		Which[ (* Adding 0 is the identity of Plus, so we ignore it *)
	    		    ConstantValueQ[inst["getOperand", 1]] && inst["getOperand", 1]["value"] === 0,
	    		    		replaceInst[inst,
	    		    		    CreateCopyInstruction[
									inst["target"],
									inst["getOperand", 2],
									inst["mexpr"]
								]
							],
	    		    ConstantValueQ[inst["getOperand", 2]] && inst["getOperand", 2]["value"] === 0,
	    		    		replaceInst[inst,
	    		    		    CreateCopyInstruction[
									inst["target"],
									inst["getOperand", 1],
									inst["mexpr"]
								]
							]
	    		],
	    Times,
	    		Which[ (* Adding 1 is the identity of Times, so we ignore it *)
	    		    ConstantValueQ[inst["getOperand", 1]] && inst["getOperand", 1]["value"] === 1,
	    		    		replaceInst[inst,
	    		    		    CreateCopyInstruction[
									inst["target"],
									inst["getOperand", 2],
									inst["mexpr"]
								]
							],
	    		    ConstantValueQ[inst["getOperand", 2]] && inst["getOperand", 2]["value"] === 1,
	    		    		replaceInst[inst,
	    		    		    CreateCopyInstruction[
									inst["target"],
									inst["getOperand", 1],
									inst["mexpr"]
								]
							]
	    		],
	    	Power,
	    		Which[
	    		    (* Power by 2 is converted to multiplication *)
	    		    ConstantValueQ[inst["getOperand", 2]] && inst["getOperand", 2]["value"] === 2,
	    		    		inst["setOperand", 2, inst["getOperand", 1]],
	    		    	(* Power by -1 is converted to a division *)
	    		    ConstantValueQ[inst["getOperand", 2]] && inst["getOperand", 2]["value"] === -1,
	    		    		inst["setOperator", Divide];
	    		    		inst["setOperand", 2, inst["getOperand", 1]];
	    		    		inst["setOperand", 1, CreateConstantValue[1]]
	    		],
	    	Subtract,
	    		Which[
				(* Subtracting by the same thing is 0 *)
				inst["getOperand", 1]["sameQ", inst["getOperand", 2]],
					replaceInst[inst,
		    		    CreateCopyInstruction[
							inst["target"],
							CreateConstantValue[0],
							inst["mexpr"]
						]
					]
	    		],
	    	Divide,
	    		Which[
				(* Dividing by the same thing is 1 *)
				inst["getOperand", 1]["sameQ", inst["getOperand", 2]],
					replaceInst[inst,
		    		    CreateCopyInstruction[
							inst["target"],
							CreateConstantValue[1],
							inst["mexpr"]
						]
					]
	    		]
	    			
	]
)
	    	
	
simplifyUnary[state_, inst_] :=
	{}
	
simplifyCompare[state_, inst_] :=
	{}
	
simplifySelect[state_, inst_] :=
	{}

run[bb_?BasicBlockQ, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitBinaryInstruction" -> simplifyBinary,
			"visitUnaryInstruction" -> simplifyUnary,
			"visitCompareInstruction" -> simplifyCompare,
			"visitSelectInstruction" -> simplifySelect
		|>,
		bb,
		"IgnoreRequiredInstructions" -> True
	]
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"SimplifyExpression",
		"The pass evaluates the expression (replaces it with a load) if the operands are constant."
];

SimplifyExpressionPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[SimplifyExpressionPass]
]]


End[] 

EndPackage[]
