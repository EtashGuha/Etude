(**
  * Expressions that have been computed and need not be recomputed
  * An expression $x op $y at p0 is available at point pi iff every path
  * from p0 to pi does not redefine $x or $y. This is backward must dataflow.
  *)

BeginPackage["Compile`Core`Analysis`DataFlow`AvailableExpressions`"]

AvailableExpressionPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Analysis`Dominator`ImmediateDominator`"]
Needs["Compile`Core`Analysis`Utilities`ScalarExpression`"]


initialize[fm_?FunctionModuleQ, opts_] :=
	fm["scanBasicBlocks",
		Function[{bb},
			bb["setProperty", "availableExpressions[in]" -> <||>];
			bb["setProperty", "availableExpressions[out]" -> <||>]
		]
	]

	
visit[avails_, inst_] :=
	With[{
		exp = CreateScalarExpression[inst] 
	},
	With[{
		hash = exp["hash"]
	},
		avails["associateTo",
			hash -> Append[
				avails["lookup", hash, {}],
				exp
			]
		];
		True
	]]


gen[bb_, opts_] :=
	Module[{
		avails = CreateReference[<||>]
	},
		CreateInstructionVisitor[
			avails,
			<|
				"visitCompareInstruction" -> visit,
				"visitBinaryInstruction" -> visit,
				"visitCallInstruction" -> visit,
				"visitUnaryInstruction" -> visit
			|>,
			bb,
			"IgnoreRequiredInstructions" -> True
		];
		avails["get"]
	]

(*
 * Avail_in(block)  = Avail_out(dom(block))
 * Avail_out(block) = Avail_in(block) \/ Nodes(block)
 *)
run[fm_?FunctionModuleQ, opts_] :=
	Module[{idom},
		fm["topologicalOrderScan",
			Function[{bb},
				idom = bb["immediateDominator"];
				bb["setProperty", "availableExpressions[in]" ->
					If[idom === None,
						<||>,
						idom["getProperty", "availableExpressions[out]"]
					]
				];
				bb["setProperty", "availableExpressions[out]" ->
					Join[
						bb["getProperty", "availableExpressions[in]"],
						gen[bb, opts]
					]
				];
			]
		];
		fm
	]
	
(**********************************************************)
(**********************************************************)
(**********************************************************)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"AvailableExpression",
	"Computes the available expressions of each basic block in the function module",
	"An expression $x op $y at p0 is available at point pi iff every path from p0 to pi does not redefine either $x or $y."
];

AvailableExpressionPass = CreateFunctionModulePass[<|
	"information" -> info,
	"initializePass" -> initialize,
	"runPass" -> run,
	"requires" -> {
		ImmediateDominatorPass,
		UsePass
	}
|>];

RegisterPass[AvailableExpressionPass]
]]


End[] 

EndPackage[]
