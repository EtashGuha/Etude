BeginPackage["Compile`Core`Transform`MoveMutability`"]


MoveMutabilityPass

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`Analysis`DataFlow`AliasSets`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]

(*
  Try to move MutabilityClone instructions closer to their source.
  This follows the def chain along via Copy and Phi instructions.
  It only moves to one source and doesn't bother moving to the same 
  BasicBlock.  If it finds a Phi instruction it follows both branches, 
  so if we get back to the clone instruction we started with we can 
  drop.  This means that we can move a clone out of a loop, ie for
  
  Function[{Typed[d, "PackedArray"]["Integer64", 1]}, 
      Module[{dd = d}, Do[dd[[1]]++, {10^5}];
        dd]]
  
  but we don't move for 
  
  Function[{Typed[d, "PackedArray"]["Integer64", 1]}, 
  	Module[{dd = d}, dd[[1]]++; 
  	dd]]
  
  or
  
  func = Function[{Typed[d, "PackedArray"["Integer64", 1]], 
   Typed[cond, "Boolean"]},
  		Module[{dd = d}, If[cond, dd = {1}]; dd[[1]]++;
   dd]]
   (this could be optimized to move the clone into the branch)
 
  Note also we don't move the Clone inserted at the end of a Function
  to prevent argument aliasing.  
*)



(*
  Find the instruction to insert after.  Basically at 
  the end of any Phi and LoadArgument instructions.
*)
findAfterInstruction[inst_] :=
	Module[{next = inst["next"]},
		If[PhiInstructionQ[next] || LoadArgumentInstructionQ[next],
			findAfterInstruction[next]
			,
			inst]
	]

(*
  we want to move the mutability clone inst to just after origin.
  If the inst is in the same BB as origin then don't move,  
  there is no point.

  inst is 
      var1 = Native`MutabilityClone[var2] 
  origin is the source of the data ie
      var3 = something()
      
  add var4 = Native`MutabilityClone[var3] after origin
  and var1 = Copy(var4) to replace inst
*)
fixClone[state_, inst_, origin_] :=
	Module[ {fun, cpyInst, cloneInst, findAfter},
		If[ inst["basicBlock"]["id"] === origin["basicBlock"]["id"],
			Return[]];
		fun = CreateConstantValue[Native`MutabilityClone];
		cloneInst = CreateCallInstruction[ "movedClone", fun, {origin["target"]}];
		findAfter = findAfterInstruction[origin];
		cloneInst["moveAfter", findAfter];
		cpyInst = CreateCopyInstruction[inst["target"], cloneInst["target"]];
		cpyInst["moveAfter", inst];
		inst["unlink"];
	] 

isMutabilityClone[inst_] :=
	CallInstructionQ[inst] && ConstantValueQ[inst["function"]] && inst["function"]["value"] === Native`MutabilityClone


(*
 CallInstruction,  if this is the MutabilityClone instruction we are 
 trying to remove, do nothing.
*)
processDef[state_, inst_?CallInstructionQ] :=
	Module[{},
		state["sources"]["appendTo", inst];
	]

(*
 PhiInstruction,  just follow the def of each source.
 But stop if there is more than 1 source.
*)
processDef[state_, inst_?PhiInstructionQ] :=
	Module[{srcs},
		srcs = inst["getSourceVariables"];
		Scan[
			If[state["sources"]["length"] < 2,	
				processVar[state,#]]&, srcs];
	]

(*
 CopyInstruction.  If we have a constant source then make 
 this the site of the source,  else just follow the def of the source.
*)
processDef[state_, inst_?CopyInstructionQ] :=
	Module[{src = inst["source"]},
		If[ConstantValueQ[src],
			state["sources"]["appendTo", inst]
			,
			processVar[state, inst["source"]]]
	]

(*
 We have come to an unknown instruction. 
 Treat this as the source.
*)
processDef[state_, inst_] :=
	state["sources"]["appendTo", inst];

processVar[state_, var_] :=
	Module[ {def},
		If[!state["visited"]["keyExistsQ", var["id"]],
			state["visited"]["associateTo", var["id"] -> var];
			def = var["def"];
			processDef[state, def];
		]
	]
	
(*
  inst is a clone instruction.  We want to see if we 
  can move it to the original source.  This is beneficial 
  if we are moving the clone out of a loop.  We do this by 
  following the def chain.   We only move if there is one
  and only one source.
  
  If this is a clone introduced at the return 
  (to avoid aliasing of arguments) don't do this.
*)
processClone[state_, inst_] :=
	Module[{src, sources},
		If[ TrueQ[inst["getProperty", "cloneForReturn", False]],
			Return[]];
		state["visited"]["set", <|inst["target"]["id"] -> inst["target"]|>];		
		state["sources"]["set", {}];
		src = inst["getOperand",1];
		processVar[state, src];
		sources = state["sources"]["get"];
		If[ Length[sources] === 1 && InstructionQ[First[sources]],
			fixClone[state, inst, First[sources]]]
	] 


visitCall[ state_, inst_] :=
	Module[{},
		If[ !isMutabilityClone[inst],
			Return[]];
		state["cloneInstructions"]["associateTo", inst["target"]["id"] -> inst];
	]


createState[fm_] :=
	<|"functionModule" -> fm, 
		"cloneInstructions" -> CreateReference[<||>], 
		"visited" -> CreateReference[<||>], 
		"sources" -> CreateReference[{}]|>

createVisitor[state_] :=
	CreateInstructionVisitor[
		state,
		<|
			"visitCallInstruction" -> visitCall
		|>,
		"IgnoreRequiredInstructions" -> True
	]



(*
  Collect the mutability clone instructions.
  If none return.
  If there are collect LoopNestingForest information and 
  process each Clone to see if it can be moved.
*)
run[fm_, opts_] :=
	Module[ {state, visitor, cloneInsts},
		state = createState[fm];
		visitor = createVisitor[state];
		visitor["traverse", fm];
		cloneInsts = state["cloneInstructions"]["values"];
		If[ Length[cloneInsts] === 0,
			Return[fm]];
		Scan[ processClone[state,#]&, state["cloneInstructions"]["values"]];
		fm
	]
	
run[args___] :=
	ThrowException[{"Invalid MoveMutabilityPass argument to run ", args}]	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"MoveMutability",
		"The pass moves MutabilityClone instructions closer to their source."
];

MoveMutabilityPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
	    UsePass
	}
|>];

RegisterPass[MoveMutabilityPass]
]]

End[] 

EndPackage[]
