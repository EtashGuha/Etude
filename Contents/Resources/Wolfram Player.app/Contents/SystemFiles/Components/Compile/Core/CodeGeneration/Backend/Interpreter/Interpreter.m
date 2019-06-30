
BeginPackage["Compile`Core`CodeGeneration`Backend`Interpreter`"]

InterpretFunctionModule;
InterpreterException;

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]



InterpretFunctionModule[fm_?FunctionModuleQ] :=
	Module[{st, insts, env, pc, jumpTable, inst, dispatch},
		insts = fm["getInstructions"];
		jumpTable = getJumpTable[fm];
		env = CreateReference[<||>];
		pc = CreateReference[1];
		st = <|
			"jumpTable" -> jumpTable,
			"insts" -> insts,
			"env" -> env,
			"pc" -> pc,
			"getPc" :> pc["get"],
			"setPc" -> Function[{newpc}, pc["set", newpc]] 
		|>;
		dispatch = CreateInstructionVisitor[
			st,
			<|
				"visitStackAllocateInstruction" -> stackAllocateInstruction,
				"visitBinaryInstruction" -> binaryInstruction,
				"visitBranchInstruction" -> branchInstruction,
				"visitCallInstruction" -> callInstruction,
				"visitCompareInstruction" -> compareInstruction,
				"visitReturnInstruction" -> returnInstruction,
				"visitUnaryInstruction" -> unaryInstruction,
				"visitUnreachableInstruction" -> unreachableInstruction,
				"visitSelectInstruction" -> selectInstruction,
				"visitLabelInstruction" -> labelInstruction,
				"visitLoadInstruction" -> loadInstruction,
				"visitCopyInstruction" -> copyInstruction,
				"visitLoadArgumentInstruction" -> loadArgumentInstruction,
				"visitLoadGlobalInstruction" -> loadGlobalInstruction,
				"visitStoreInstruction" -> storeInstruction,
				"visitInertInstruction" -> inertInstruction,
				"visitGetElementInstruction" -> getElementInstruction,
				"visitPhiInstruction" -> phiInstruction
			|>
		];
		Catch[
			While[st["getPc"] <= Length[insts],
				inst = insts[[st["getPc"]]];
				dispatch["visit", inst];
				pc["increment"];
			]
		];
		st["env"]["lookup", fm["result"]["value"]]
	]

getJumpTable[fm_] :=
	Module[{tbl = <||>, idx = 1},
		CreateInstructionVisitor[
			<|
				"visitInstruction" -> Function[{st, inst},
					idx++
				],
				"visitLabelInstruction" -> Function[{st, inst},
					AssociateTo[tbl, inst["name"] -> idx++]
				]
			|>,
			fm
		];
		tbl
	]
	
getRegister[st_, reg_] :=
	Which[
		ConstantValueQ[reg],
			reg["value"],
		st["env"]["keyExistsQ", reg],
			With[{r = st["env"]["lookup", reg]},
				If[ConstantValueQ[r],
					r["value"],
					r
				]
			],
		True,
			ThrowException[InterpreterException[{"Cannot find " <> reg["toString"] <> " in environment"}]]
	]
setRegister[st_, reg_, val_] :=
	st["env"]["associateTo", reg -> val]

getLocation[st_, bb_] :=
	If[KeyExistsQ[st["jumpTable"], bb["name"]],
		st["jumpTable"][bb["name"]],
		ThrowException[InterpreterException[{"Cannot find PC for " <> ToString[bb["id"]] <> "(" <> bb["name"] <> ") in jump table"}]]
	]


stackAllocateInstruction[st_, inst_] :=
	setRegister[st, inst["target"], "undef"]
binaryInstruction[st_, inst_] :=
	setRegister[st, inst["target"], Apply[inst["operator"], getRegister[st, #]& /@ inst["operands"]]]
	
branchInstruction[st_, inst_] :=
	Module[{target, loc},
		target = If[inst["isUnconditional"] || getRegister[st, inst["condition"]],
			inst["getOperand", 1],
			inst["getOperand", 2]
		];
		loc = getLocation[st, target];
		st["setPc"][loc - 1] (**< we are going to add 1 in the loop
		                          so we will offset it *)
	]
callInstruction[st_, inst_] :=
	ThrowException[InterpreterException[{"unimplemented CallInstruction"}]]

compareInstruction[st_, inst_] :=
	setRegister[st, inst["target"], Apply[inst["operator"], getRegister[st, #]& /@ inst["operands"]]]
returnInstruction[st_, inst_] :=
	With[{loc = Length[st["insts"]]},
		st["setPc"][loc]
	]
unaryInstruction[st_, inst_] :=
	setRegister[st, inst["target"], inst["operator"][getRegister[st, inst["operator"]]]]
unreachableInstruction[st_, inst_] :=
	ThrowException[InterpreterException[{"instruction should not have been reached"}]]

selectInstruction[st_, inst_] :=
	setRegister[st, inst["target"],
		If[getRegister[st, inst["condition"]],
			inst["getOperand", 1],
			inst["getOperand", 2]
		]
	]
labelInstruction[st_, inst_] := {}
loadInstruction[st_, inst_] :=
	Module[{source},
		source = getRegister[st, inst["source"]];
		setRegister[st, inst["target"], source]
	]
copyInstruction[st_, inst_] :=
	Module[{source},
		source = getRegister[st, inst["source"]];
		setRegister[st, inst["target"], source]
	]
loadArgumentInstruction[st_, inst_] :=
	ThrowException[InterpreterException[{"unimplemented LoadArgumentInstruction"}]]

loadGlobalInstruction[st_, inst_] :=
	setRegister[st, inst["target"], inst["source"]]
storeInstruction[st_, inst_] :=
	setRegister[st, inst["target"], getRegister[st, inst["source"]]]
inertInstruction[st_, inst_] :=
	Module[{head, args},
		head = getRegister[st, inst["head"]];
		args = getRegister[st, #]& /@ inst["arguments"];
		setRegister[st, inst["target"], Apply[head, args]]
	]
getElementInstruction[st_, inst_] :=
	Module[{source, args},
		source = getRegister[st, inst["source"]];
		args = getRegister[st, #]& /@ inst["arguments"];
		setRegister[st, inst["target"], Apply[Part, Join[{source}, args]]]
	]
phiInstruction[st_, inst_] :=
	ThrowException[InterpreterException[{"unimplemented PhiInstruction"}]]

End[]

EndPackage[]

