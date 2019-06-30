BeginPackage["Compile`Core`Optimization`CommonSubexpressionElimination`"]

(*
https://github.com/golang/go/blob/master/src/cmd/compile/internal/ssa/zcse.go 
https://github.com/golang/go/blob/master/src/cmd/compile/internal/ssa/cse.go 
*)

CommonSubexpressionEliminationPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Instruction`UnaryInstruction`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Analysis`DataFlow`AvailableExpressions`"]
Needs["Compile`Core`IR`Instruction`BinaryInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CompareInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Analysis`Dominator`DominatorPass`"]




hash[state_, arg_Symbol] := "S" <> SymbolName[arg]
hash[state_, arg_?VariableQ] := state["lookup", arg["id"], "V" <> ToString[arg["id"]]]
hash[state_, arg_?ConstantValueQ] := "C" <> ToString[arg["value"]]
hash[state_, inst_?definesExpressionQ] := StringRiffle[hash[state, #]& /@ Prepend[inst["operands"], inst["operator"]], "_"]

definesExpressionQ[_?BinaryInstructionQ] := True
definesExpressionQ[_?CompareInstructionQ] := True
definesExpressionQ[_?UnaryInstructionQ] := True
definesExpressionQ[_?CallInstructionQ] := True
definesExpressionQ[_] := False


cse[avails_, inst_?definesExpressionQ] :=
	Module[{h, insts, bb, found, dominatedBy},
		h = hash[avails, inst];
		insts = avails["lookup", h, {}];
		found = False;
		dominatedBy = #["id"]& /@ inst["basicBlock"]["dominator"];
		Do[
			If[!found,
				bb = avail["basicBlock"];
				If[MemberQ[dominatedBy, bb["id"]],
					found = True;
					With[{load = CreateCopyInstruction[
										inst["target"],
										avail["target"],
										avail["mexpr"]
								 ]},
						load["moveAfter", inst];
						load["setId", inst["id"]];
						inst["unlink"];
	                ]
				]
			],
			{avail, insts}	
		];
		avails["associateTo",
			(* We want prepend, because we want to
			 * have the closest available expression
			 * when we perform the propagation
			 *)
			h -> Prepend[insts, inst] 
		];
	]

run[fm_, opts_] :=
	Module[{avails = CreateReference[<||>]},
		CreateInstructionVisitor[
			avails,
			<|
				"visitBinaryInstruction" -> cse,
				"visitUnaryInstruction" -> cse,
				"visitCompareInstruction" -> cse,
				"visitCallInstruction" -> cse,
				"traverse" -> "reversePostOrder"
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		]
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"CommonSubexpressionElimination",
		"The pass avoids repetitive computations of subexpressions."
];

CommonSubexpressionEliminationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		AvailableExpressionPass,
		DominatorPass
	}
|>];

RegisterPass[CommonSubexpressionEliminationPass]
]]

End[] 

EndPackage[]
