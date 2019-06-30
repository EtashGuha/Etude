BeginPackage["Compile`Core`Transform`AbortHandling`"]


AbortHandlingPass

Begin["`Private`"] 

Needs["Compile`Core`Analysis`Loop`LoopNestingForest`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRegistry`"] (* for RegisterPass *)
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]

Needs["Compile`Core`IR`Instruction`LandingPadInstruction`"]
Needs["Compile`Core`IR`Instruction`ResumeInstruction`"]
Needs["Compile`Core`IR`Instruction`InvokeInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`IR`Instruction`ReturnInstruction`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["LLVMCompileTools`Exceptions`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`LoopInformation`"]
Needs["Compile`Core`Transform`ResolveFunctionCall`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`Transform`BasicBlockPhiReorder`"]


(*
 Add support for abort handling.  
 The current strategy finds the header for loops and inserts an Native`GetCheckAbort call.
*)



(*
 bb is a basic block that is the head of a loop.
 Add Native`GetCheckAbort call after the last Phi Instruction.
 Typically ResolveFunctionCallInstruction will inline the code.
*)
processHeader[ state_, fm_, bb_] :=
	Module[{inst = bb["lastPhiInstruction"], fun, newInst},
		fun = CreateConstantValue[ Native`GetCheckAbort];
		newInst = CreateCallInstruction["abort", fun, {}];
		newInst["moveAfter", inst];
		ResolveFunctionCallInstruction[ state["programModule"], newInst];
		state["exceptionAdded"]["set", True];
	]


checkLocal[state_, func_?ConstantValueQ] :=
	Module[{val = func["value"]},
		If[ FunctionModuleQ[state["programModule"]["getFunctionModule", val]],
			state["localFunction"]["set", True]];
	]
	
checkLocal[state_, func_] :=
	Null

visitLambda[state_, inst_] :=
	checkLocal[state, inst["source"]]

visitCall[state_, inst_] :=
	checkLocal[state, inst["function"]]

noLocalFunctionCallQ[state_, fm_] :=
	Module[{vst},
		vst = CreateInstructionVisitor[
				state,
					<|
					"visitLambdaInstruction" -> visitLambda,
					"visitCallInstruction" -> visitCall,
					"visitCallInstruction" -> visitCall
					|>,
				"IgnoreRequiredInstructions" -> True
		];
		vst["traverse", fm];
		TrueQ[state["localFunction"]["get"]]
	]


(*
  There is no loop in fm,  we should add a  Native`GetCheckAbort
  at the start.  Only do this if the function calls other local functions
  (eg functions in the same PM).
*)
processNoLoop[ state_, fm_] :=
	Module[{bb = fm["firstBasicBlock"], inst, fun, newInst},
		If[!noLocalFunctionCallQ[state, fm],
			Return[]];
		inst = bb["firstNonLabelInstruction"];
		While[LoadArgumentInstructionQ[inst],
			inst = inst["next"]];
		fun = CreateConstantValue[ Native`GetCheckAbort];
		newInst = CreateCallInstruction["abort", fun, {}];
		newInst["moveBefore", inst];
		ResolveFunctionCallInstruction[ state["programModule"], newInst];
		state["exceptionAdded"]["set", True];
	]



(*
 Find the loopinformation for the FunctionModule.  
 Scan over over these and any loop headers get processed.
*)
processLoops[state_, fm_] :=
	Module[{loopInfo, workList = CreateReference[{}], elem, header},
		loopInfo = fm["getProperty", "loopinformation", Null];
		If[loopInfo === Null,
			ThrowException[{"Cannot find loop information", fm}]];
		workList["pushBack", loopInfo];
		While[!workList["isEmpty"],
			elem = workList["popFront"];
			header = elem["header"];
			If[BasicBlockQ[header],
				processHeader[ state, fm, header]];
			Scan[
				If[LoopInformationQ[#], workList["pushBack", #]]&, elem["children"]["get"]]
		];
		If[ !state["exceptionAdded"]["get"],
			processNoLoop[ state, fm]];
	]


createState[ fm_, abortHand_] :=
	Module[ {},
		<|"functionModule" -> fm, "programModule" -> fm["programModule"], 
			"localFunction" -> CreateReference[False], "exceptionAdded" -> CreateReference[False],
			"AbortHandling" -> abortHand|>
	]


(*

*)
run[fm_, opts_] :=
	Module[ {
			state, abortHand, passOpts = Lookup[opts, "PassOptions", {}]
		},
		abortHand = TrueQ[
		    Lookup[ passOpts, "AbortHandling", True] &&
		    Lookup[opts, "AbortHandling", True] =!= False
		];
		If[ TrueQ[abortHand],
			state = createState[fm, abortHand];
			processLoops[state, fm]];
		fm
	]
	
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"AbortHandling",
		"The pass adds code to respond to external aborts."
];

AbortHandlingPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		LoopNestingForestPass,
		BasicBlockPhiReorderPass
	}
|>];

RegisterPass[AbortHandlingPass]
]]

End[] 

EndPackage[]
