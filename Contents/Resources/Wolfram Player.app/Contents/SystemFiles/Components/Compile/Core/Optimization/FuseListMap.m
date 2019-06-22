BeginPackage["Compile`Core`Optimization`FuseListMap`"]


FuseListMapPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["CompileUtilities`Callback`"]



mapConstantValue := mapConstantValue =
	CreateConstantValue[CreateMExpr[Map]]
parmapConstantValue := parmapConstantValue =
	CreateConstantValue[CreateMExpr[ParallelMap]]
compositionConstantValue := compositionConstantValue =
	CreateConstantValue[CreateMExpr[Composition]]

isMap[f_] :=
	f["sameQ", mapConstantValue] ||
	f["sameQ", parmapConstantValue]

fuseMap[fm_, inst_] /; isMap[inst["function"]]:=
	Module[{lst, defInst, funComp},
		lst = inst["getArgument", 2];
		defInst = lst["def"];
		If[CallInstructionQ[defInst] && (* the list def is a map and is only used once *)
		   isMap[defInst["function"]] &&
		   Length[defInst["definedVariable"]["uses"]] === 1,
			inst["setArgument", 2, defInst["getArgument", 2]];
			funComp = CreateCallInstruction[
				defInst["definedVariable"],
				compositionConstantValue,
				{
					defInst["getArgument", 1],
					inst["getArgument", 1]	
				},
				defInst["mexpr"]
			];
			inst["setArgument", 1, funComp["target"]];
			funComp["moveBefore", inst];
			defInst["unlink"]
		]
	]

run[bb_?BasicBlockQ, opts_] :=
	Module[{vst},
		vst = CreateInstructionVisitor[
			bb["functionModule"],
			<|
				"visitCallInstruction" -> fuseMap
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		vst["traverse", "reversePostOrder", bb]
	]
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"FuseListMap",
		"The pass fuses list maps (this assumes that the functions are not side effecting)."
];

FuseListMapPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		UsePass,
		DefPass
	}
|>];

RegisterPass[FuseListMapPass]
]]


End[] 

EndPackage[]
