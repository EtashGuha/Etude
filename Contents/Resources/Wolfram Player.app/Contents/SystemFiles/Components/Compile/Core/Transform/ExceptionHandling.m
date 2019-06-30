BeginPackage["Compile`Core`Transform`ExceptionHandling`"]


ExceptionHandlingPass

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRegistry`"] (* for RegisterPass *)

Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]

Needs["Compile`Core`IR`Instruction`LandingPadInstruction`"]
Needs["Compile`Core`IR`Instruction`ResumeInstruction`"]
Needs["Compile`Core`IR`Instruction`InvokeInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`IR`Instruction`ReturnInstruction`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["LLVMCompileTools`Exceptions`"]

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`Analysis`Function`FunctionThrows`"]

pushStackFun[] := CreateConstantValue[ Native`PrimitiveFunction["SetJumpStack_Push"]]
popStackFun[] := CreateConstantValue[ Native`PrimitiveFunction["SetJumpStack_Pop"]]
setjmpFun[] := CreateConstantValue[ Native`PrimitiveFunction["setjmp"]]
throwFun[] := CreateConstantValue[ Native`PrimitiveFunction["throwWolframException"]]
compFun[] := CreateConstantValue[ Native`PrimitiveFunction["binary_sameq_SignedInteger"]]

makeConstant[val_] := CreateConstantValue[ val]


setFunctionCallType[state_, inst_, funTy_] :=
	Module[ {
				instArgs = inst["arguments"], 
				argTys = funTy["arguments"], 
				resTy = funTy["result"] 
		},
		If[ Length[instArgs] =!= Length[argTys],
			ThrowException[{"Number of arguments does not match argument types", instArgs, argTys}]];
		inst["target"]["setType", resTy];
		Apply[ #1["setType", #2]&, Transpose[{instArgs, argTys}], {1}];
		inst["function"]["setType", funTy]
	]


$throwable =
<|
	"_Native`ThrowWolframException_Integer32_Void" -> True,
	Native`PrimitiveFunction["throwWolframException"] -> True
|>

throws[ state_, inst_] :=
	inst["getProperty", "Throws", False]


(*
  The last instruction of bb should be a jump instruction, this should also be removed.
  Also add the resume BB as a child of this bb.
*)

fixBasicBlock[state_, bb_] :=
	Module[ {lastinst},
		lastinst = bb["lastInstruction"];
		If[ !BranchInstructionQ[lastinst],
			ThrowException[{"Last instruction should be a branch", lastinst}]];
		lastinst["unlink"];
		bb["addChild", state["resumeBasicBlock"]];
	]

(*
 We have res = Call[ fun, args];
 
 Convert to
 
 except = Call[ SetJumpPush, {}]
 br except resumeBB splitBB
 
 splitBB
   res = Call[ fun, args]
   Call[ SetJumpPop, {}]
*)

visitCallModel[state_, inst_, "SetJumpLongJump"] :=
	Module[ {pushInst, bb, tobb, setjmpInst, compInst, brInst, popInst},
		
		pushInst = CreateCallInstruction["pushTrg", pushStackFun[], {}];
		setFunctionCallType[state, pushInst, state["pushType"]];
		pushInst["moveBefore", inst];
		
(*
   Introduce BB just before instruction
*)
		bb = inst["basicBlock"];
		tobb = bb["splitAfter", pushInst];
		fixBasicBlock[state, bb];

		setjmpInst = CreateCallInstruction["setjmpTrg", setjmpFun[], {pushInst["target"]}];
		setFunctionCallType[state, setjmpInst, state["setjmpType"]];
		setjmpInst["moveAfter", pushInst];
		
		compInst = CreateCallInstruction["compTrg", compFun[], {setjmpInst["target"], makeConstant[0]}];
		setFunctionCallType[state, compInst, state["compareType"]];
		compInst["moveAfter", setjmpInst];

		brInst = CreateBranchInstruction[ {tobb, state["resumeBasicBlock"]}, compInst["target"], None];
		brInst["moveAfter", compInst];
		
		popInst = CreateCallInstruction["popTrg", popStackFun[], {}];
		setFunctionCallType[state, popInst, state["popType"]];
		popInst["moveAfter", inst];
	]

visitCallModel[state_, inst_, model_] :=
	Module[ {bb, tobb, invokeInst},
(*
   Introduce BB after instruction and change Call to Invoke.
*)
		bb = inst["basicBlock"];
		tobb = bb["splitAfter", inst];
		invokeInst = CreateInvokeInstruction[inst["target"], inst["function"], inst["arguments"], tobb, state["resumeBasicBlock"]];
		invokeInst["moveBefore", inst];
		inst["unlink"];
		
		fixBasicBlock[state, bb];
	]


	
visitCall[state_, inst_] :=
	Module[ {model = state["ExceptionsModel"]},
		If[ !throws[state, inst],
			Return[]];
		If[ model === "Basic" && ReturnInstructionQ[inst["next"]],
			Return[]];
		visitCallModel[state, inst, state["ExceptionsModel"]]
	]


(*
  If this instruction is tracking the return instruction,  marked by the property, 
  then return True.  This will cause the instruction not be included in the exception
  block.  Since we are in an exception, we are going to need to collect the return object.
*)
fromReturnInstruction[inst_] :=
	inst["getProperty", "TrackedMemoryReturnInstruction", False]

createResumeBasicBlock[state_, fm_, mmBB_, model_] :=
	Module[ {newBB, landTrgt, inst, newInst},
		newBB = CreateBasicBlock["ExceptionReturn"];
		newBB["initId"];
		newInst = CreateLabelInstruction[ "ExceptionResume"];
		newBB[ "addInstruction", newInst];
		newInst = CreateLandingPadInstruction["cleanup"];
		newBB[ "addInstruction", newInst];
		landTrgt = newInst["target"];
		landTrgt["setType", state["landingTargetType"]];
		(* move instructions *)
		inst = mmBB["firstNonLabelInstruction"];
		While[ !BranchInstructionQ[inst] && inst =!= None,
			If[CallInstructionQ[inst] && !fromReturnInstruction[inst],
				newInst = CreateCallInstruction["mem", inst["function"], inst["arguments"]];
				newInst["cloneProperties", inst];
				newBB["addInstruction", newInst]];
			inst = inst["next"]];
		If[ model === "SetJumpLongJump",
			newInst = CreateCallInstruction["popTrg", popStackFun[], {}];
			setFunctionCallType[state, newInst, state["popType"]];
			newBB[ "addInstruction", newInst];
			(*
			  Should really use the same argument that was used earlier.
			*)
			newInst = CreateCallInstruction["throwTrg", throwFun[], {makeConstant[1]}];
			setFunctionCallType[state, newInst, state["throwType"]];
			newBB[ "addInstruction", newInst]];
		newInst = CreateResumeInstruction[landTrgt];
		newBB[ "addInstruction", newInst];
		state["functionModule"]["linkBasicBlock", newBB];
		newBB
	]

createResumeBasicBlock[state_, fm_, Null, "Basic"] :=
	Module[ {newBB, landTrgt, newInst},
		newBB = CreateBasicBlock["ExceptionReturn"];
		newBB["initId"];
		newInst = CreateLabelInstruction[ "ExceptionResume"];
		newBB[ "addInstruction", newInst];
		newInst = CreateLandingPadInstruction["cleanup"];
		newBB[ "addInstruction", newInst];
		landTrgt = newInst["target"];
		landTrgt["setType", state["landingTargetType"]];
		newInst = CreateResumeInstruction[landTrgt];
		newBB[ "addInstruction", newInst];
		state["functionModule"]["linkBasicBlock", newBB];
		newBB
	]


createState[fm_, model_] :=
	Module[ {tyEnv = fm["programModule"]["typeEnvironment"]},
		<|
			"functionModule" -> fm,
			"landingTargetType" -> tyEnv["resolve", TypeSpecifier["MachineInteger"]],
			"compareType" -> tyEnv["resolve", TypeSpecifier[{"Integer32", "Integer32"} -> "Boolean"]],
			"pushType" -> tyEnv["resolve",TypeSpecifier[{} -> "Handle"["Integer32"]]],
			"popType" -> tyEnv["resolve",TypeSpecifier[{} -> "Void"]],
			"setjmpType" -> tyEnv["resolve",TypeSpecifier[{"Handle"["Integer32"]} -> "Integer32"]],
			"throwType" -> tyEnv["resolve",TypeSpecifier[{"Integer32"} -> "Void"]],
			"ExceptionsModel" -> model|>
	]


postProcess[state_, model_, fm_] :=
	Module[ {},
		Null
	]


fixLinkage[ pm_, fun_, linkage_] :=
	pm["externalDeclarations"]["addFunction", fun, 
			<|"class" -> "Function", "Linkage" -> linkage, "Class" -> "Linked"|>]


postProcess[state_, "SetJumpLongJump", fm_] :=
	Module[ {pm = fm["programModule"]},
		fixLinkage[ pm, Native`PrimitiveFunction["SetJumpStack_Push"], "Runtime"];
		fixLinkage[ pm, Native`PrimitiveFunction["SetJumpStack_Pop"], "Runtime"];
		fixLinkage[ pm, Native`PrimitiveFunction["setjmp"], "Runtime"];
		fixLinkage[ pm, Native`PrimitiveFunction["binary_sameq_SignedInteger"], "LLVMCompareFunction"];
	]


run[fm_, opts_] :=
	Module[ {
			mmBB = fm["getProperty", "memoryManageBasicBlock", Null], 
			throws = fm["information"]["Throws"], 
			state, 
			model
		},
		model = Lookup[opts, "ExceptionsModel", Automatic];
		model = ResolveExceptionsModel[model];
        (*
          If exception model is none, then we do not need to do anything.
        *)
        If[ model === None,
            Return[fm]];
		(*
		  If no exceptions are thrown then don't need to do anything.
		  If there is no clean up block and the model is not Basic,
		  then don't need to do anything.
		*)
		If[ !throws,
			Return[fm]];
		If[ mmBB === Null && model =!= "Basic",
			Return[fm]];
		state = createState[fm, model];
		state["resumeBasicBlock"] = createResumeBasicBlock[ state, fm, mmBB, model];
		CreateInstructionVisitor[
			state,
			<|
				"visitCallInstruction" -> visitCall
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		postProcess[state, model, fm];
		fm
	]
	
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ExceptionHandling",
		"The pass adds code to deal with exceptions."
];

ExceptionHandlingPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		FunctionThrowsPass	
	}
|>];

RegisterPass[ExceptionHandlingPass]
]]

End[] 

EndPackage[]
