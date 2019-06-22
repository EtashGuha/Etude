BeginPackage["Compile`Core`Debug`InsertDebugDeclarePass`"];

InsertDebugDeclarePass;

Begin["`Private`"];

Needs["Compile`Core`IR`Variable`"];
Needs["Compile`Core`IR`ConstantValue`"];
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"];
Needs["CompileAST`Class`Base`"]; (* For MExprQ *)
Needs["CompileAST`Class`Symbol`"]; (* For MExprSymbolQ *)
Needs["CompileAST`Class`Normal`"]; (* For MExprNormalQ *)
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["CompileUtilities`Reference`"] (* For CreateReference *)
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`StoreInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]




(* Goes up throught an mexpr's parents until a scoping construct (e.g Function/Module) is found, and returns it *)
(* TODO: Add With/Block when they are implemented in the compiler.  *)
parentScopingMExpr[mexpr_?MExprQ] := Module[{current = mexpr},
	While[True,
		If[MExprQ[current["parent"]],
			current = current["parent"];,
		(* else *)
		(* TODO: Make this return value None *)
			Return[];
		];
		If[MemberQ[{Function, Module}, current["getHead"]],
			Return[current];
		];
	]
];
parentScopingMExpr[args___] := Throw[{"Bad arguments to parentScopingMExpr: ", {args}}];

createDebugDeclareCall[name_String -> var_?VariableQ, mexpr_] := Module[{debugDeclarePrimitive, returnVar, declareCall, bbMExpr},
	debugDeclarePrimitive = CreateConstantValue[Native`PrimitiveFunction["DebugDeclare"]];
	(* This variable should never actually be referenced, but CallInstruction requires a target *)
	returnVar = "debugDeclare";
	declareCall = CreateCallInstruction[returnVar, debugDeclarePrimitive, {}];

	declareCall["setMexpr", mexpr];
	declareCall["setProperty", "debug.name" -> name];
	declareCall["setProperty", "debug.value" -> var];

	declareCall
];

visit[st_, inst_] := Module[{target, mexpr, name, bb, stackAllocation, storeInst},
	(* Check `inst` represents an introduction or modification of a WL symbol in the source. *)
	If[ ! (inst["getProperty", "variableWrite", False]
	       || inst["getProperty", "variableDeclaration", False]),
		Return[];
	];

	(* Those instructions that are variableWrite or variableDeclaration MUST have a target. *)
	If[!inst["hasTarget"],
		ThrowException[{"InsertDebugDeclarePass: expected variable change to have target"}]
	];
	target = inst["target"];
	mexpr = inst["mexpr"];
	bb = inst["basicBlock"];

	(* TODO: This does not handle setting parts or subvalues *)
	Which[
		MExprSymbolQ[mexpr],
			(* Important to use name here and not sourceName; we want distinct allocations for shadowed symbols *)
			(* TODO: This may look a little weird in the debugger. Shadowed symbols will have at
			         least 2 entries in the listing of local variables, say
			         `(long) x = 10
                      (long) x$1 = 15`. Possibly not worth fixing. *)
			name = mexpr["name"],
		MExprNormalQ[mexpr] && mexpr["hasHead", Set] && Length[mexpr["arguments"]] == 2,
			Module[{firstArg = mexpr["arguments"][[1]]},
				If[!MExprSymbolQ[firstArg],
					ThrowException[{"InsertDebugDeclarePass: expected first argument of Set to be MExprSymbolQ: ", firstArg}]
				];
				name = firstArg["name"]
			],
		True,
			ThrowException[{"InsertDebugDeclarePass: expected instruction mexpr to be symbol or Set: ", mexpr}]
	];
	AssertThat["name should be StringQ", name]["named", "name"]["satisfies", StringQ];

	stackAllocation = If[st["keyExistsQ", name],
		(* We've already generated a StackAllocation for this local symbol, so use it instead *)
		st["lookup", name]
		,
		(* Create a new stack allocation to hold the value of name *)
		Module[{returnVar, firstBB, stackAllocateInst, debugDeclareInst},
			returnVar = CreateVariable[];
			stackAllocateInst = CreateStackAllocateInstruction[returnVar, CreateConstantValue[1], CreateConstantValue[Native`CreateHandle], None];
			returnVar["setDef", stackAllocateInst];

			(* This property prevents optimization in DeadCodeEliminationPass. *)
			stackAllocateInst["setProperty", "DebuggerReadable" -> True];
			stackAllocateInst["setId", Hash[Unique[name]]];

			firstBB = bb["functionModule"]["firstBasicBlock"];
			firstBB["addInstructionBefore", firstBB["firstNonLabelInstruction"], stackAllocateInst];

			(* Because `st` didn't already contain name, we know `inst` represents the first time
			   the symbol `name` is readable, so declare it after `inst` *)
			debugDeclareInst = createDebugDeclareCall[name -> stackAllocateInst["target"], mexpr];
			firstBB["addInstructionAfter", inst, debugDeclareInst];

			st["associateTo", name -> returnVar];
			returnVar
		]
	];

	(* `inst` has updated the symbol `name`, so update our mirrored value in the stack allocation *)
	storeInst = CreateStoreInstruction[target, stackAllocation, CreateConstantValue[Native`Store], mexpr];
	target["addUse", storeInst];
	bb["addInstructionAfter", inst, storeInst];
];

run[fm_, opts_] := Module[{
	(* A map from Symbols to SSA variables, where each SSA variable is a stack allocation that can be written to using Store. *)
	stackAllocations = CreateReference[<||>]
},
	CreateInstructionVisitor[
		stackAllocations,
		<|
			"visitInstruction" -> visit,
			"traverse" -> "reversePostOrder"
		|>,
		fm,
		"IgnoreRequiredInstructions" -> True
	];
];



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"InsertDebugDeclare",
	"Inserts debugging instructions to enable viewing local variables in debuggers such as GDB and LLDB.",
	"For each instruction that represents a declaration of or modification to a WL symbol in the source program, generate and cache a stack allocation on first encounter of a given symbol, and create Store instruction's on every encounter to save the value of the symbol at that point. In effect, the stack allocation for each symbol mirrors the value of that symbol through the execution of the function. These stack allocations are only ever Store'd to, never read in the WIR; they are meant to be read by the debugger. A special case exists in DeadCodeEliminationPass to avoid optimizing these allocations out."
];

InsertDebugDeclarePass = CreateFunctionModulePass[<|
	"requires" -> {DefPass},
	"information" -> info,
	"runPass" -> run,
	"postPasses" -> {TopologicalOrderRenumberPass}
|>];

RegisterPass[InsertDebugDeclarePass];
]]


End[];

EndPackage[];
