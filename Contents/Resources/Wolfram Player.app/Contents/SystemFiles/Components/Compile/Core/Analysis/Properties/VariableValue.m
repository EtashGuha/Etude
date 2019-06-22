
BeginPackage["Compile`Core`Analysis`Properties`VariableValue`"]

VariableValuePass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]


visitInst[state_, inst_] :=
	If[ CopyInstructionQ[inst], visitCopy[state, inst]]

visitCopy[ state_, inst_] :=
	Module[ {trg, src, trgDef, srcDef},
		trg = inst["target"];
		src = inst["source"];
		trgDef = trg["getProperty", "variableValue", Null];
		srcDef = src["getProperty", "variableValue", Null];
		Which[
			trgDef === srcDef,
				Null,
			trgDef === Null,
				trg["setProperty", "variableValue" -> srcDef];
				Scan[ visitInst[state, #]&, trg["uses"]]
			,
			srcDef === Null,
				src["setProperty", "variableValue" -> trgDef];
				visitInst[state,  trg["def"]]
		]
	]


createVisitor[state_] :=
	CreateInstructionVisitor[
		state,
		<|
			"visitCopyInstruction" -> visitCopy
		|>,"IgnoreRequiredInstructions" -> True]



(*
	Scan through all basic blocks and annotates the ones that are interior to a loop with
	the property "isPartOfLoop" -> True.
*)

run[fm_, opts_] := 
	Module[{state, visitor},
		state = <||>;
		visitor = createVisitor[state];
		visitor["traverse", fm];
		
		fm
	]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"VariableValue",
	"This pass fills out the VariableValue property across Copy instructions."
];

VariableValuePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		UsePass,
		DefPass
	},
	"passClass" -> "Analysis"
|>];


]]

End[]

EndPackage[]
