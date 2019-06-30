BeginPackage["Compile`Core`Optimization`RemoveRedundantStackAllocate`"]
RemoveRedundantStackAllocatePass;

Begin["`Private`"] 

Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Analysis`Dominator`DominatorPass`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]




run[fm_, opts_] :=
	Module[{bb, target, uses, def, dominates = True, instsToRemove},
		instsToRemove = Internal`Bag[];
		CreateInstructionVisitor[
			<|
				"visitStackAllocateInstruction" -> Function[{st, inst},
					dominates = False;
					bb = inst["basicBlock"];
					target = inst["target"];
					uses = target["uses"];
					def = target["def"];
					Do[
						If[
							!def["sameQ", inst],
								dominates = True;
								Break[],
							MemberQ[#["id"]& /@ use["basicBlock"]["dominator"], def["basicBlock"]["id"]],
								dominates = True;
								Break[]
						];
						,
						{use, uses}
					];
					If[dominates,
						Internal`StuffBag[instsToRemove, inst]
					];
				],
				"traverse" -> "reversePostOrder"
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		#["unlink"]& /@ Internal`BagPart[instsToRemove, All];
		fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"RemoveRedundantStackAllocate",
	"Deletes stack allocate instructions when all the defs dominate all the uses.",
	"TODO: The implementation is pretty stupid and terrible. Essentially we should be looking " <>
	"to see if any definition reaches all the uses"
];

RemoveRedundantStackAllocatePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		DominatorPass,
		DefPass,
		UsePass
	}
|>];

RegisterPass[RemoveRedundantStackAllocatePass]
]]

End[] 

EndPackage[]
