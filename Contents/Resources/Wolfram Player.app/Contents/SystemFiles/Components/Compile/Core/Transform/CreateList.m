BeginPackage["Compile`Core`Transform`CreateList`"]


CreateListPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]

	
insertFixed[ state_, inst_, fun_] :=
	Module[ {cons, callInst},
		cons = CreateConstantValue[fun];
		callInst = CreateCallInstruction[ inst["target"], cons, inst["operands"], inst["mexpr"]];
		callInst["moveAfter", inst];
		inst["unlink"];
	]
	
	
insertGeneral[ state_, inst_] :=
	Module[ {args, arg, newInst, i},
		args = inst["arguments"];
		newInst = CreateCallInstruction[ "createList", CreateConstantValue[Native`CreateListView], {CreateConstantValue[Length[args]]}];
		newInst["moveBefore", inst];
		Do[
			arg = Part[args,i];
			newInst = CreateCallInstruction[ "addListElement", 
							CreateConstantValue[Native`ListViewAdd], 
							{newInst["target"], CreateConstantValue[i-1], arg}];
			newInst["moveBefore", inst];
			, {i, Length[args]}];		
		newInst = CreateCallInstruction[ inst["target"], CreateConstantValue[Native`ListViewFinalize], {newInst["target"]}, inst["mexpr"]];
		newInst["moveBefore", inst];
		inst["unlink"];
	]


visitList[ state_, inst_] :=
	Module[ {args},
		args = inst["arguments"];
		Which[
			Length[args] === 0,
			    ThrowException[{"Cannot compile a zero length list."}]
			,
			Length[args] === 1,
			    insertFixed[state, inst, Native`ListUnary]
			,
			Length[args] === 2,
			    insertFixed[state, inst, Native`ListBinary]
			,
			True,
			    insertGeneral[state, inst]
		]
	]


vistCall[state_, inst_] :=
	Module[ {fun, val},
		fun = inst["function"];
		If[ ConstantValueQ[fun],
			val = fun["value"];
			Which[
				val === List,
					visitList[state, inst],
				True,
					Null]]
			]


run[bb_, opts_] :=
	(
	CreateInstructionVisitor[
				<|
				"visitCallInstruction" -> vistCall
				|>,
				bb,
			"IgnoreRequiredInstructions" -> True
			];
	bb)
	
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"CreateList",
		"The pass replaces CallInstructions which create lists with a sequence of instructions " <>
		"that actually do the work. It can't be done with macros because List is used so often as a structual component."
];

CreateListPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[CreateListPass]
]]

End[] 

EndPackage[]
