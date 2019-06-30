BeginPackage["Compile`Core`Optimization`EvaluateExpression`"]


EvaluateExpressionPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



evalExpr[st_, inst_] :=
	Module[{operands, newInst, expr, src},
		operands = inst["getOperands"];
		If[AllTrue[operands, ConstantValueQ] && (**< the two operands are constant *)
		   Head[inst["operator"]] === Symbol, (**< a bad test to check that the operator can be applied on the operands *)
		   With[{e = Apply[inst["operator"], #["value"]& /@ operands]},
		   	expr = CreateMExpr[e]
		   ];
		   src = CreateConstantValue[expr];
			(** we now need to replace the instruction with a load instruction.
			  * we use the same id so we do not invalidate other passes that reference
			  * the instruction id
			  *)
			newInst = CreateCopyInstruction[
				inst["target"],
				src,
				inst["mexpr"]
			];
			newInst["moveAfter", inst];
			newInst["setId", inst["id"]];
			inst["unlink"];
		]
	]
evalSelect[st_, inst_] :=
	Module[{cond, operands, newInst = Undefined},
		cond = inst["getCondition"];
		operands = inst["getOperands"];
		Which[
			ConstantValueQ[cond], (**< The condition is a known constant *)
				newInst = If[cond["value"] === True,
					CreateCopyInstruction[
						inst["target"],
						operands[[1]],
						inst["mexpr"]
					],
					CreateCopyInstruction[
						inst["target"],
						operands[[2]],
						inst["mexpr"]
					]
				],
			operands[[1]] === operands[[2]], (**< the two operands are the same *)
				newInst = CreateCopyInstruction[
					inst["target"],
					operands[[1]],
					inst["mexpr"]
				]
		];
		If[newInst =!= Undefined,
			newInst["moveAfter", inst];
			newInst["setId", inst["id"]];
			inst["unlink"];
		]
	]

run[bb_?BasicBlockQ, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitBinaryInstruction" -> evalExpr,
			"visitUnaryInstruction" -> evalExpr,
			"visitCompareInstruction" -> evalExpr,
			"visitSelectInstruction" -> evalSelect
		|>,
		bb,
		"IgnoreRequiredInstructions" -> True
	]
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"EvaluateExpression",
		"The pass evaluates the expression (replaces it with a load) if the operands are constant."
];

EvaluateExpressionPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[EvaluateExpressionPass]
]]



End[] 

EndPackage[]
