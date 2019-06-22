(*
   Search start to end recording whether variables can be reached from FixedMTensor 
   instructions,  such as Constant MTensor or variables that have a MTensor value.
   
   Based on LiveVariables pass.
*)

BeginPackage["Compile`Core`Optimization`RecordFixedMTensor`"]

RecordFixedMTensorPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`LoadInstruction`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["CompileUtilities`Callback`"]



finalize[fm_] :=
	Module[{},
		fm["scanBasicBlocks",
			Function[{bb},
				Module[ {mods},
					mods = bb["getProperty", "fixedMTensors"];
					mods = mods["values"];
					bb["setProperty", "fixedMTensors" -> mods];
				]
			]
		]
	]
	
initialize[fm_, opts_] :=
	Module[{},
		fm["scanBasicBlocks",
			Function[{bb},
				bb["setProperty", "fixedMTensors" -> CreateReference[<||>]];
			]
		]
	]


(*
 Pass through the basic blocks in reversePostOrder.  
 For each basic block get the constant MTensors for the parent basic blocks. 
 Then pass through the basic block collection an updated list, and add this 
 to the basic block as a property.
 
*)

run[fm_?FunctionModuleQ, opts_] :=
	Module[{worklist, bb, mods, old},
		worklist = fm["reversePostOrder"];
		While[worklist =!= {},
			
			{bb, worklist} = {First[worklist], Rest[worklist]}; 
			mods = bb["getProperty", "fixedMTensors"];
			old = mods["keys"];
			Scan[ mods["join", #["getProperty", "fixedMTensors"]]&, bb["getParents"]];
			bb["scanInstructions",
				Function[{inst},
					computeFixed[ inst, mods]]];
			bb["setProperty", "fixedMTensors" -> mods];
			(*
			  If we modified the elementsModified of this bb, we have to revisit any children 
			  because these use the info for this bb.
			*)
			If[old =!= bb["getProperty", "fixedMTensors"]["keys"],
				worklist = DeleteDuplicates[Join[worklist, bb["getChildren"]]];
			]
		];
		fm
	]
	
SetAttributes[timeIt, HoldAllComplete]
accum = 0;
timeIt[e_] :=
	With[{t = AbsoluteTiming[e;][[1]]},
		accum += t;
		Print[StringTake[ToString[Unevaluated[e]], 10], "  t = ", t, "  accum = ", accum]
	]


(*
  If the instruction has a contantMTensor property then add the target to the list
  and set the property on the target.
  
  If the instruction is one of Copy/Load/Phi then we look at its used variables. If any of these 
  are on the list we add the target to the list and set the property on the target.   
     
*)
computeFixed[ inst_, mods_] :=
	Module[ {vars},
		If[TrueQ[inst["getProperty", "variableDeclaration"]],
			addVariable[inst["target"], mods]]; 
		If[ TrueQ[inst["getProperty", "constantMTensor"]],
			addVariable[inst["target"], mods]]; 
		If[passInstructionQ[inst],
			vars = inst["usedVariables"];
			If[AnyTrue[vars, mods["keyExistsQ", #["id"]]&],
				addVariable[inst["target"], mods]]];
	]
	
addVariable[ var_?VariableQ, mods_] :=
	(
	var["setProperty", "fixedMTensor" -> True];
	mods["associateTo", var["id"] -> var];
	)

passInstructionQ[inst_] :=
	CopyInstructionQ[inst] || LoadInstructionQ[inst] || PhiInstructionQ[inst]


(**********************************************************)
(**********************************************************)
(**********************************************************)


RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["RecordFixedMTensor", "INFO"];

info = CreatePassInformation[
	"RecordFixedMTensor",
	"Records whether MTensors are fixed and need copy for instructions that modify.",
	"The fixedMTensors property on a BasicBlock records any variables that this BasicBlock (or it's parents) " <>
	"derive from fixed MTensors.  This is needed to keep immutability of MTensors."
];

RecordFixedMTensorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"finalizePass" -> finalize,
	"requires" -> {},
	"passClass" -> "Analysis"
|>];

RegisterPass[RecordFixedMTensorPass]
]]


End[] 

EndPackage[]
