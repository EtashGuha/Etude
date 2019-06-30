BeginPackage["Compile`Core`Transform`ProcessMutability`"]


ProcessMutabilityPass

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`Analysis`DataFlow`AliasSets`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]

mutabilityCloneConstant[] := CreateConstantValue[Native`MutabilityClone]

manageVariableQ[var_] :=
	(TrueQ[var["type"]["isNamedApplication", "PackedArray"]])


(*
  Convert the Native`MutabilityClone call into a copy instruction
  since the variable is not aliased.
*)
convertInstruction[state_, inst_, var_] :=
	Module[{newInst},
		newInst = CreateCopyInstruction[ inst["target"], var];
		newInst["moveBefore", inst];
		inst["unlink"]; 
	]


(*
 Return a list of all the aliases containing var.
*)
aliasesContaining[ state_, aliasesIn_, var_] :=
	Module[ {varId = var["id"], aliases = aliasesIn},
		
		aliases = Map[ If[MemberQ[#, varId], DeleteCases[#, varId], {}]&, aliases];
		aliases = DeleteCases[aliases, {}];
		(*
		  The length should really be 0 or 1 because there should not be more 
		  than one alias set containing a given variable (else they should be 
		  merged).
		*)
		Switch[Length[aliases],
			0,
				{}
			,
			1,
				First[aliases]
			,
			_,
			    (*  TODO make this an error *)
				Null
		]
	]

(*
 aliases are aliased with var,  if any are not closure variables
 return False,  else add the variables to the list of closure variables and 
 return True
*)
checkClosureAliased[ state_, aliases_, var_] :=
	Module[ {closures},
		closures = Map[ state["closureVariablesConsumed"]["lookup", #, Null]&, aliases];
		If[MemberQ[closures, Null],
			Return[False]
		];
		Scan[ state["closureVariablesAliased"]["associateTo", #["id"] -> #]&, closures];
		True
	]



(*
  If this a clone instruction then look at the aliases.
  If the var is not aliased (aliases of {}) then convert
  If the var is only aliased by closure variables, then convert the def of the closure
  Otherwise leave the instruction.
*)
visitCall[state_, inst_] :=
	Module[ {aliases, var, fun},
		fun = inst["function"];
		If[ ConstantValueQ[fun] && fun["sameQ", mutabilityCloneConstant[]],
			var = inst["getArgument", 1];
			aliases = inst["getProperty", "aliasSetsIn", Null];
			If[ aliases === Null,
				Return[]
			];
			aliases = aliasesContaining[ state, aliases, var];
			Which[
				aliases === {},
					convertInstruction[state, inst, var]
				,
				ListQ[aliases] && checkClosureAliased[state, aliases, var],
					convertInstruction[state, inst, var]
				,
				True,
					Null
			]
		];
	]

(*
  if the return variable is managed then
  if the return var is an argument then add a clone
  if the return var aliases nothing then leave
  if the return var aliases a closure then add a clone
  else leave
*)
visitReturn[state_, inst_] :=
	Module[ {retVar = inst["value"], aliases},
		If[!manageVariableQ[retVar],
			Return[]];
		If[ 
			state["arguments"]["keyExistsQ", retVar["id"]],
				addCloneForReturn[state, inst]
			,
			aliases = inst["getProperty", "aliasSetsIn", {}];
			If[ aliases === Null,
				Return[]
			];
			aliases = aliasesContaining[ state, aliases, retVar];
			Which[
				aliases === {},
					Null
				,
				ListQ[aliases] && checkClosureAliased[state, aliases, retVar],
					Null
				,
				True,
					addCloneForReturn[state, inst]
			]
		];
	]


(*
  If the argument is managed then add it to the arguments list.
  This is used to avoid aliasing arguments in the output.  
  If the function is going to be inlined, then don't do this.
*)
visitLoadArgument[ state_, inst_] :=
	Module[{trgt = inst["target"], meta = inst["basicBlock"]["functionModule"]["getMetaData"]},
		If[meta === Null,
			If[ manageVariableQ[trgt],
				state["arguments"]["associateTo", trgt["id"] -> trgt]]
			,
			If[ !TrueQ[meta["getData", "Inline", False]] && manageVariableQ[trgt],
				state["arguments"]["associateTo", trgt["id"] -> trgt]
			]
		]
	]

(*
 Add a clone instruction for Return
 If the ArgumentAlias information is set don't add a clone,  the caller of 
 the function will add an alias if necessary.  Later we could use this code 
 to set the ArgumentAlias information.
*)
addCloneForReturn[state_, inst_] :=
	Module[ {fm, cloneInst, fun},
		fm = inst["basicBlock"]["functionModule"];
		If[TrueQ[fm["information"]["ArgumentAlias"]],
			Return[]];
		state["typeValid"]["set", False];
		fun = mutabilityCloneConstant[];
		cloneInst = CreateCallInstruction[ "returnClone", fun, {inst["value"]}, inst["mexpr"]];
		cloneInst["moveBefore", inst];
		cloneInst["setProperty", "cloneForReturn" -> True];
		inst["setValue", cloneInst["target"]];
	]

createVisitor[state_] :=
	CreateInstructionVisitor[
		state,
		<|
			"visitLoadArgumentInstruction" -> visitLoadArgument,
			"visitCallInstruction" -> visitCall,
			"visitReturnInstruction" -> visitReturn
		|>,
		"IgnoreRequiredInstructions" -> True
	]

createState[pm_] :=
	<|
		"programModule" -> pm, 
		"arguments" -> CreateReference[<||>], 
		"typeValid" -> CreateReference[True],
		"closureVariablesAliased" -> CreateReference[<||>],
		"closureVariablesConsumed" -> CreateReference[<||>],
		"closureVariablesCloned" -> CreateReference[{}]
	|>


(*
  inst is a LoadClosureVariable call the target of which is mutated.
  It has an aliased variable and this has a definition.
  
  aliasedVar = instruction[];
  
  we should change this to
  
  newVar = instruction[];
  aliasedVar = Call Native`MutabilityClone[ newVar];
  
  instruction should never be a LoadArgumentInstruction, because we should not 
  be mutating an argument.
  
  Also, we store the aliasedVar to use for output aliasing.
  
*)
fixClosureVariable[state_, aliasedVar_] :=
	Module[{def, fun, newVar, newInst},
		state["closureVariablesCloned"]["appendTo", aliasedVar];
		def = aliasedVar["def"];
		newVar = CreateVariable[];
		newVar["setName", "closureFix"];
		def["setTarget", newVar];
		If[ !InstructionQ[def],
			ThrowException[{"Cannot find definition of aliased variable", aliasedVar}]
		];
		fun = mutabilityCloneConstant[];
		newInst = CreateCallInstruction[ aliasedVar, fun, {newVar}];
		newInst["moveAfter", def];
		aliasedVar["setDef", newInst];
		newVar["setDef", def];
		newVar["setUses", {newInst}];
	]

(*
 Set the closureVariablesConsumed field in the state and then call on the instructions
*)
visitFunction[ state_, visitor_, fm_] :=
	Module[{vars = fm["getProperty", "closureVariablesConsumed", Null]},
		state["arguments"]["set", <||>];
		vars = If[ vars === Null, {}, vars["get"]];
		state["closureVariablesConsumed"]["set", <||>];
		Scan[state["closureVariablesConsumed"]["associateTo", #["id"] -> #]&, vars];
		visitor["traverse", fm]
	]

run[pm_, opts_] :=
	Module[ {state, visitor, closures},
		state = createState[pm];
		visitor = createVisitor[state];
		pm["scanFunctionModules",
			visitFunction[state, visitor, #]&
		];
		closures = state["closureVariablesAliased"]["get"];
		Scan[
			fixClosureVariable[state, #]&,
			closures
		];
		If[!TrueQ[state["typeValid"]["get"]],
			RunPasses[{InferencePass}, pm]
		];
		pm["setProperty", "processMutabilityRun" -> True];
		pm
	]
	
run[args___] :=
	ThrowException[{"Invalid ProcessMutabilityPass argument to run ", args}]	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ProcessMutability",
		"The pass removes unecessary MutabilityClone instructions and inserts them when necessary."
];

ProcessMutabilityPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		AliasSetsPass
	}
|>];

RegisterPass[ProcessMutabilityPass]
]]

End[] 

EndPackage[]
