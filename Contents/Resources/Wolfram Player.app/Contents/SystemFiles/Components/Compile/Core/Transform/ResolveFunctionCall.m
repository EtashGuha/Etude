BeginPackage["Compile`Core`Transform`ResolveFunctionCall`"]


ResolveFunctionCallPass

ResolveFunctionCallInstruction
ResolveFunctionCallInstructionWithInferencing

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Utilities`Serialization`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["Compile`Core`Transform`ResolveConstants`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`LambdaInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadGlobalInstruction`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`ResolveTypes`"]
Needs["Compile`Core`PassManager`PassRunner`"]



    
SetAttributes[timeIt, HoldAllComplete]
accum = 0;
timeIt[e_] :=
    With[{t0 = AbsoluteTiming[e]},
    With[{t = First[t0], r = Last[t0]},
        accum += t;
        Print[StringTake[ToString[Unevaluated[e]], 10], "  t = ", t, "  accum = ", accum];
        r
    ]]
    
(*
 Inline the function fmCall in place of the call instruction.
 
 Split the current BB after the instruction, get prevBB and afterBB.
 
 Go through the new BBs,  convert LoadArgument to Copy from the arguments.
 Change the return instruction to a Copy to the target.
 Add a branch to the afterBB.
 Remove the Call Instruction.
 Change the branch in the prevBB to point to the start BB.
*)
inlineFunction[ pm_, inst_, fun_, funFM_] :=
	Module[ {fm, inline,  prevBB, afterBB, firstBB, args, trgt, branchInst},
		fm = inst["basicBlock"]["functionModule"];
		inline = fm["getProperty", "inlineCall", 0];
		fm["setProperty", "inlineCall" -> inline +1];
		(*
		  Need to process the FM being inlined to make sure that other functions are added 
		  to the PM.   Not sure this comment is relevant.
		*)
		funFM["setProgramModule", pm];
		funFM["setTypeEnvironment", pm["typeEnvironment"]];
		resolveFunctionModule[funFM, Null];	
			
		prevBB = inst["basicBlock"];
		afterBB = prevBB[ "splitAfter", inst];
		firstBB = funFM["firstBasicBlock"];
		args = inst["operands"];
		trgt = inst["target"];
		funFM["scanInstructions", 
				Function[{inst1},
					inst1["setMexpr", inst["mexpr"]];
					Switch[inst1["_instructionName"],
						"LoadArgumentInstruction", fixLoadArgument[inst1, args],
						"ReturnInstruction", fixReturn[inst1, trgt, afterBB]]]];
		branchInst = inst["next"];
		Assert[ BranchInstructionQ[branchInst]];
		inst["unlink"];
		branchInst["setOperand", 1, firstBB];
		prevBB["removeChild", afterBB];
		prevBB["addChild", firstBB];
		linkBasicBlocks[prevBB["functionModule"], firstBB];
		funFM["disposeNotElements"]
	]


linkBasicBlocks[ fm_, bb_] :=
	Module[ {children},
		If[ fm["id"] === bb["functionModule"]["id"],
			Return[]];
		fm["linkBasicBlock", bb];	
		children = bb["getChildren"];	
		Scan[linkBasicBlocks[fm, #]&, children]
	]



(*
  replace the LoadArgumentInstruction with a CopyInstruction.
*)
fixLoadArgument[ inst_, args_] :=
	Module[ {index, copyInst},
		index = inst["index"]["data"];
		copyInst = CreateCopyInstruction[inst["target"], Part[args, index], inst["mexpr"]];
		copyInst["moveAfter", inst];
		inst["unlink"]
	]

(*
  replace the ReturnInstruction with a CopyInstruction and add a Branch
*)
fixReturn[ inst_, trgt_, afterBB_] :=
	Module[ {copyInst, branchInst},
		copyInst = CreateCopyInstruction[trgt, inst["value"], inst["mexpr"]];
		copyInst["moveAfter", inst];
		inst["unlink"];
		branchInst = CreateBranchInstruction[ {afterBB}, inst["mexpr"]];
		branchInst["moveAfter", copyInst];
		branchInst["basicBlock"]["addChild", afterBB];
	]




makeName[ pm_, fun_, argTys_] :=
	Module[ {name},
		name = 
			Which[ 
				SymbolQ[fun],
					SymbolName[fun],
				StringQ[fun],
					fun,
				True,
					ThrowException[{"Cannot form function name ", fun}]
			];
		pm["typeEnvironment"]["functionLookup"]["specializeName", name, argTys]	 
	]


(*
 Goes with the support eg in TypeInferencing for function variables,  eg f = Abs.
 We should look for the function in the PM either with modified name or not.
 If found use it.  If not then find the definition in the type environment and 
 add to the PM.
*)
visitLambda[pm_, inst_] :=
	Module[ {src, ty, name, fmData, funFM},
		src = inst["source"];
		ty = inst["target"]["type"];
		If[ MissingQ[ pm["getFunctionModule", src]] && TypeArrowQ[ty],
			name = makeName[ pm, src, ty["arguments"]];
			If[ MissingQ[pm["getFunctionModule", name]],
				fmData = pm["typeEnvironment"]["findDefinition", src, ty["arguments"], ty["result"]];
				If[ fmData =!= Null,
					funFM = WIRDeserialize[ fmData, "UniqueID" -> True];
					funFM["programModule"] = pm;
					funFM["setName", name];
					funFM["setTypeEnvironment", pm["typeEnvironment"]];
					pm["addFunctionModule", funFM];
				]
			];
			(*
			 Fix the name even if the function is already in the PM
			*)
			inst["setSource", name]
		];
	]
	

(*
   We have a definition to add. 
   Insert into the PM.
   
   Any local definitions have to be added to the PM,  these can't be inlined.
*)

processDefinition[ pm_, inst_, funName_, funTy_, data_] :=
	Module[ {mainDef, inline, localDefs, fm},
		mainDef = Lookup[data, "Definition", Null];
		If[ mainDef === Null,
			ThrowException[{"Cannot find definition", funName, data}]
		];
		localDefs = Lookup[data, "LocalDefinitions", {}];
		Map[
			addToProgramModule[pm, #]&, localDefs];
		fm = WIRDeserialize[ pm["typeEnvironment"], mainDef, "UniqueID" -> True];
		If[
			!FunctionModuleQ[fm],
				ThrowException[CompilerException["Cannot deserialize to a single function ", funName]]
		];
		inline = fm["information"]["inlineInformation"]["shouldInline", pm];
		If[inline,
			inlineFunction[ pm, inst, funName, fm],
			processCallFunction[ pm, inst, funName, funTy, fm]]
	]


addToProgramModule[ pm_, serialDef_WIRSerialization] :=
	Module[ {fm},
		fm = WIRDeserialize[ pm["typeEnvironment"], serialDef, "UniqueID" -> True];
		addToProgramModule[pm, fm]
	]

addToProgramModule[ pm_, fm_] :=
	Module[ {visitor},
		If[ pm["addFunctionModuleIfNotPresent", fm],
			visitor = createInstructionVisitor[pm];
			visitor["visit", fm]
		];
		fm
	]

(*
  Don't replace a CallInstruction with another CallInstruction of the same form.
*)
processCallFunction[pm_, inst_, funName_, funTy_, fmIn_] :=
	Module[ {fm, newFunName, cons, newInst},
		fm = addToProgramModule[pm, fmIn];
		newFunName = fm["name"];
		If[ !(CallInstructionQ[inst] && ConstantValueQ[inst["function"]] && inst["function"]["value"] === newFunName),
			cons = CreateConstantValue[newFunName];
			cons["setType", funTy];
			newInst = CreateCallInstruction[ inst["target"], cons, inst["operands"], inst["mexpr"]];
			newInst["moveAfter", inst];
			inst["unlink"]
		];
	]

(*
 lookupData is Linked,  add to the externalDefinitions.
 If we redirect to the same function, we can stop...  
 this happens when we are processing a function that calls itself
*)
processRedirect[pm_, inst_, funName_, funTy_, lookupData_] :=
	Module[ {newFunName, newConst, visitor, newInst},
		If[!KeyExistsQ[lookupData, "Redirect"],
			ThrowException[{"Cannot find redirected function", funName, lookupData}]
		];
		newFunName = Lookup[lookupData, "Redirect"];
		If[ newFunName === funName,
			Return[]
		];
		newConst = CreateConstantValue[newFunName];
		newConst["setType", funTy];
		If[ CallInstructionQ[inst],
			inst["setFunction", newConst];
			newInst = inst
			,
			newInst = CreateCallInstruction[inst["target"], newConst, inst["operands"], inst["mexpr"]];
			newInst["moveAfter", inst];
			inst["unlink"]
		];
		visitor = createInstructionVisitor[pm];
		visitor["visit", newInst];
		newInst
	]



(*
 lookupData is Linked,  add to the externalDefinitions.
*)
processLinked[pm_, inst_, funName_, funTy_, lookupData_] :=
	Module[ {},
		pm["externalDeclarations"]["addFunction", funName, lookupData]
	]

(*
 Erase the function,  actually replace with a copy instruction from the zero of the type 
 to the target,  but this should really be optimized away.  Maybe later use a constant that
 cannot be instantiated.
*)
processErasure[pm_, inst_, funName_, funTy_, lookupData_] :=
	Module[ {newCons, newInst},
		newCons = CreateConstantValue[ 0];
		newCons["setType", inst["target"]["type"]];
		newInst = CreateCopyInstruction[inst["target"], newCons];
		newInst["moveAfter", inst];
		inst["unlink"]
	]


throwError[pm_, inst_, funName_, funTy_, lookupData_] :=
	ThrowException[{"Unknown function resolution class", funName, lookupData}]


$classFunction =
<|
	"Linked" -> processLinked,
	"Redirect" -> processRedirect,
	"Definition" -> processDefinition,
	"Erasure" -> processErasure
|>


(*
   process a function where the call is a ConstantValue.
   
   Lookup function in ProgramModule,  if found exit.
   Lookup function in externalDeclarations,  if found exit.
   Get definition property from value (added by Inference System),
      process in functionDefinitionLookup.
   result might be 
      linked,  easy,  add to externalDefinitions and leave call
      function name,  switch call to function name and re-process
      implementation add to PM (if not present) and call or inline
*)
visitConstantValueFunction[ pm_, inst_, fun_] :=
	Module[ {funName, funTy, data, workFun, def},
		funName = fun["value"];
		funTy = fun["type"];
		
		data = pm["externalDeclarations"]["lookupFunction", funName];
		If[ !MissingQ[data],
			(* If the function being called is declared externally, we don't need to do anything *)
			Return[]
		];
		def = fun["getProperty", "definition"];
		def = pm["typeEnvironment"]["functionDefinitionLookup"]["process", pm, funName, funTy, def];
		If[ AssociationQ[def] && KeyExistsQ[def, "Class"],
			workFun = Lookup[$classFunction, def["Class"], throwError];
			workFun[ pm, inst, funName, funTy, def]
		]
	]


(*
  External function designed to be called from elsewhere that resolves a function call
  where the function has a single monomorphic definition.  This is useful for inserting 
  code later down the generation,  eg for exception or abort handling.
  
  It checks there is a single monomorphic function,  that the function and the type 
  have the same numbers of arguments.  It sets the types in the instructions and the 
  definition property, then calls the other ResolveFunctionCall tools to deal with 
  the call.
  
  An improvement for this would be to call type inferencing,  just for this function.
*)
ResolveFunctionCallInstruction[ pm_, inst_?CallInstructionQ] :=
	Module[{tyEnv = pm["typeEnvironment"], fun = inst["function"], funName, ty, tyArgs, len, def},
		If[ !ConstantValueQ[fun],
			ThrowException[{"ResolveFunctionCallInstruction expects a ConstantValue function"}]];
		funName = fun["value"];
		ty = tyEnv["functionTypeLookup"]["getMonomorphicList", funName];
		If[Length[ty] =!= 1,
			ThrowException[{"ResolveFunctionCallInstruction can only resolve a single definition."}]];
		ty = First[ty];
		If[!TypeArrowQ[ty] || Length[ty["arguments"]] =!= Length[inst["arguments"]],
			ThrowException[{"Resolved function type does not match CallInstruction."}]];
		fun["setType", ty];
		def = ty["getProperty", "definition", Null];
		If[ def === Null,
			ThrowException[{"ResolveFunctionCallInstruction cannot find definition for function."}]];
		fun["setProperty", "definition" -> def];
		inst["target"]["setType", ty["result"]];
		tyArgs = ty["arguments"];
		len = Length[inst["arguments"]];
		Do[ inst["getArgument", i]["setType", Part[tyArgs,i]], {i, len}];
		visitCall[pm, inst];
	]


ResolveFunctionCallInstruction[ args___] :=
	ThrowException[{"Illegal arguments to ResolveFunctionCallInstruction"}];


(*
  Resolves a function call where inferencing has been done.
*)
ResolveFunctionCallInstructionWithInferencing[ pm_, inst_?CallInstructionQ] :=
	Module[{fun = inst["function"],  def},
		def = fun["getProperty", "definition", Null];
		If[ def === Null,
			ThrowException[{"ResolveFunctionCallInstruction cannot find definition for function."}]];
		fun["setProperty", "definition" -> def];
		visitCall[pm, inst];
	]



visitCall[pm_, inst_] :=
	Module[ {fun},
		fun = inst["function"];
		If[ ConstantValueQ[fun],
			visitConstantValueFunction[pm, inst, fun]
		]
	]


visitNAry[pm_, inst_] :=
	visitConstantValueFunction[ pm, inst, inst["operator"]]


(*
  Convert into a Call instruction.
*)
visitInert[pm_, inst_] :=
	Module[ {newInst},
		newInst = CreateCallInstruction[inst["target"], inst["head"], inst["arguments"], inst["mexpr"]];
		newInst["moveAfter", inst];
		inst["unlink"];
	]






$globalClassFunction =
<|
	"Definition" -> processGlobalDefinition,
	"Redirect" -> processGlobalRedirect
|>


(*
  A load global function must have a constant value as its source.  
  This should have a definition property, added by type inferencing.
  At present we are only working with functions that have an actual definition,  though it should be easy to extend.
  
*)

visitLoadGlobal[pm_, inst_] :=
	Module[ {src, def, ty, workFun, name},
		src = inst["source"];
		If[!ConstantValueQ[src],
			ThrowException[{"LoadGlobalInstruction source must be a Constant Value", inst}]
		];
		name = src["value"];
		ty = src["type"];
		def = src["getProperty", "definition"];
		def = pm["typeEnvironment"]["functionDefinitionLookup"]["process", pm, name, ty, def];
		If[ AssociationQ[def] && KeyExistsQ[def, "Class"],
			workFun = Lookup[$globalClassFunction, def["Class"], throwError];
			workFun[ pm, inst, name, ty, def]
		];
	]


(*
 Process the various definitions,  and return the new name.
 
 The main function is added as a localFunction, but maybe local definitions 
 should as well,  not sure about those.
*)
processGlobalDefinition[ pm_, inst_, funName_, ty_, data_] :=
	Module[ {mainDef, localDefs, fm, newName, newCons, newInst},
		mainDef = Lookup[data, "Definition", Null];
		If[ mainDef === Null,
			ThrowException[{"Cannot find definition", funName, data}]
		];
		localDefs = Lookup[data, "LocalDefinitions", {}];
		Scan[
			addToProgramModule[pm, #]&,
			localDefs
		];
		fm = addToProgramModule[pm, mainDef];
		fm["setProperty", "localFunction" -> True];
		newName = fm["name"];
		newCons = CreateConstantValue[newName];
		newCons["setType", ty];
		newInst = CreateLambdaInstruction[inst["target"], newCons, inst["mexpr"]];
		newInst["moveBefore", inst];
		inst["unlink"]
	]


(*
  If we are redirecting to the same name this means we are already processing a function 
  and came back to it.  We just do nothing.  The function will be added later.
*)
processGlobalRedirect[pm_, inst_, funName_, funTy_, lookupData_] :=
	Module[ {newName, newConst, visitor, newInst},
		newName = Lookup[lookupData, "Redirect", Null];
		If[ newName === Null,
			ThrowException[{"Cannot find redirected function", funName, lookupData}]
		];
		If[ newName === funName,
			Null,
			newConst = CreateConstantValue[newName];
			newConst["setType", funTy];
			newInst = CreateLoadGlobalInstruction[inst["target"], newConst, inst["mexpr"]];
			newInst["moveAfter", inst];
			inst["unlink"];
			visitor = createInstructionVisitor[pm];
			visitor["visit", newInst]];
	]

createInstructionVisitor[pm_] :=
	CreateInstructionVisitor[
		pm,
		<|
			"visitBinaryInstruction" -> visitNAry,
			"visitCallInstruction" -> visitCall,
			"visitCompareInstruction" -> visitNAry,
			"visitInertInstruction" -> visitInert,
			"visitLoadGlobalInstruction" -> visitLoadGlobal,
			"visitUnaryInstruction" -> visitNAry
		|>,
		"IgnoreRequiredInstructions" -> True
	]

resolveFunctionModule[ fm_, logger_] :=
	Module[ {visitor, pm = fm["programModule"], oLogger},
		(*
		  Record the old pass logger.  Then if we have a logger set and it 
		  is a recursing logger we add this as a property on the program 
		  module.
		*)
		oLogger = pm["getProperty","PassLogger", Null];
		If[ logger =!= Null && logger =!= Automatic && logger["hasField", "shouldRecurse"] && TrueQ[logger["shouldRecurse"]],
			pm["setProperty", "PassLogger" -> logger]
			,
			oLogger = False];
		visitor = createInstructionVisitor[pm];
		visitor["visit", fm];
		(*
		oLogger of False means the logger was never altered
		if it is Null that means a logger was added but one was not already set
		so we should remove
		otherwise we should just restore
		*)
		Which[
			oLogger === False,
				Null,
			oLogger === Null,
				pm["removeProperty", "PassLogger"],
			True,
				pm["setProperty", "PassLogger" -> oLogger]];
		fm
	]

(*
  If we inline any number of calls then it is useful to run the FuseBasicBlocks pass immediately.
  The mechanism for determining the number is done with a property on the FM.  It would be better 
  to pass a state around, but this would be more intrusive on the code, which I want to keep stable.
  
  It might also be good to have the running of the FuseBasicBlock pass controlled by the PassRunner 
  that invoked this.  But that is definitely more work.
  
  TODO,  improve this to use a state to pass around so that the number of inline calls can be kept.
*)
run[fm_, opts_] :=
	Module[{logger = Lookup[opts, "PassLogger", Null], ef, inline},
		fm["getProperty", "inlineCall" -> 0];
		ef = resolveFunctionModule[fm, logger];
		inline = ef["getProperty", "inlineCall",0];
		If[ inline > 5,
			ef = RunPasses[{"FuseBasicBlocks"}, ef]];
		ef["removeProperty", "inlineCall"];
		ef
	]

run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveFunctionCall",
	"The pass replaces CallInstructions which create lists with a sequence of instructions " <>
	"that actually do the work."
];

ResolveFunctionCallPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		InferencePass,
		ResolveConstantsPass
	},
	"postPasses" -> {
		ResolveTypesPass
	}
|>];

RegisterPass[ResolveFunctionCallPass]
]]

End[] 

EndPackage[]
