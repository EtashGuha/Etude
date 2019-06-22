
BeginPackage["Compile`Core`CodeGeneration`Backend`LLVM`MObjectCollect`"]


MObjectCollectPass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRegistry`"] (* for RegisterPass *)
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`SetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]






getSetElementOperator[tyEnv_] :=
	Module[ {setOper},
		setOper = CreateConstantValue[Native`SetElement];
		setOper["setType", tyEnv["resolve", TypeSpecifier[{"CArray"["MObject"], "MachineInteger", "MObject"} -> "Void"]]];
		setOper
	]

getGetElementOperator[tyEnv_] :=
	Module[ {getOper},
		getOper = CreateConstantValue[Native`GetElement];
		getOper["setType", tyEnv["resolve", TypeSpecifier[{"CArray"["MObject"], "MachineInteger"} -> "MObject"]]];
		getOper
	]

getStackAllocateOperator[tyEnv_] :=
	Module[ {setOper},
		setOper = CreateConstantValue[Native`StackAllocate];
		setOper["setType", tyEnv["resolve", TypeSpecifier[{ "MachineInteger"} -> "CArray"["MObject"]]]];
		setOper
	]
	
getRuntimePushFunction[tyEnv_] :=
	Module[ {setOper},
		setOper = CreateConstantValue[Native`PrimitiveFunction["Runtime_PushStackFrame"]];
		setOper["setType", tyEnv["resolve", TypeSpecifier[{ "CArray"["MObject"], "MachineInteger"} -> "Void"]]];
		setOper
	]

getRuntimePopFunction[tyEnv_] :=
	Module[ {setOper},
		setOper = CreateConstantValue[Native`PrimitiveFunction["Runtime_PopStackFrame"]];
		setOper["setType", tyEnv["resolve", TypeSpecifier[{} -> "Void"]]];
		setOper
	]


createTypedConstant[ data_, val_, ty_] :=
	Module[{cons = CreateConstantValue[val]},
		cons["setType", data["typeEnvironment"]["resolve", TypeSpecifier[ ty]]];
		cons
	]

createBitCastTo[data_, var_, ty_] :=
	createCallInstruction[data, Native`PrimitiveFunction["BitCast"], {var}, ty]

createCallInstruction[data_, fun_, vars_List, ty_] :=
	Module[ {conFun, funTy, newInst},
		conFun = CreateConstantValue[ fun];
		funTy = data["typeEnvironment"]["resolve", TypeSpecifier[ Map[#["type"]&, vars] -> ty]];
		conFun["setType", funTy];
		newInst = CreateCallInstruction[ "callres", conFun, vars];
		newInst["target"]["setType", data["typeEnvironment"]["resolve", TypeSpecifier[ ty]]];
		newInst
	]


(*
   Do not need a Cast here since this stack only has MObjects.
*)
setElementProcessor[data_, inst_] :=
	Module[ {tenCon, setOper},
		data["active"]["set", True];	
		tenCon = createTypedConstant[ data, Compile`NullReference, "MObject"];
		inst["setSource", tenCon];
		setOper = getSetElementOperator[data["typeEnvironment"]];
		inst["setOperator", setOper]
	]


(*
  "void" = Call Native`MemoryStoreStack[ stackArray, index, value]
  convert to
   
   mobjTrgt = Call BitCast * -> MObject
   SetElement[ stackArray, index, mobjTrgt]
*)
storeStackProcessor[data_, inst_] :=
	Module[ {tyEnv = data["typeEnvironment"], setOper, castInst, setInst},
		data["active"]["set", True];
		castInst = createBitCastTo[data, inst["getArgument", 3], "MObject"];
		castInst["moveBefore", inst];
		setOper = getSetElementOperator[tyEnv];
		setInst = CreateSetElementInstruction[inst["getArgument", 1], inst["getArgument", 2], castInst["target"], setOper];
		setInst["moveBefore", inst];
		inst["unlink"];
	]


(*
 Initialize the stack by calling Native`PrimitiveFunction["Runtime_PushStackFrame"].
*)
stackInitialize[ data_, inst_] :=
	Module[ {tyEnv = data["typeEnvironment"], fun},
		fun = getRuntimePushFunction[tyEnv];
		inst["setFunction", fun];
		inst["target"]["setType", tyEnv["resolve", TypeSpecifier["Void"]]];
	]

(*
 Free the stack by calling Native`PrimitiveFunction["Runtime_PopStackFrame"].
 Also,  drop the arguments.
*)
stackFree[ data_, inst_] :=
	Module[ {tyEnv = data["typeEnvironment"], fun},
		fun = getRuntimePopFunction[tyEnv];
		inst["setFunction", fun];
		inst["setArguments", {}];
		inst["target"]["setType", tyEnv["resolve", TypeSpecifier["Void"]]];
	]


passThroughProcessor[ data_, inst_] :=
	Null

removeProcessor[ data_, inst_] :=
	inst["unlink"]


$callprocessors =
<|
	Native`MemoryReleaseStack -> removeProcessor,
	Native`MemoryStoreStack -> storeStackProcessor,
	Native`MemoryAcquireStack -> removeProcessor,
	Native`MemoryAcquire -> removeProcessor,
	Native`MemoryStackInitialize -> stackInitialize,
	Native`MemoryStackFree -> stackFree
|>


getFunctionProcessor[data_, inst_] :=
	Module[ {fun = inst["function"], funValue},
		If[ inst["getProperty", "TrackedMemoryType"] === "MObject",
			funValue = If[ ConstantValueQ[fun], fun["value"], Null];
			Lookup[ $callprocessors, funValue, passThroughProcessor]
			,
			passThroughProcessor
		]
	]

setupCallInstruction[data_, inst_] :=
	Module[ {funProcessor = getFunctionProcessor[data, inst]},
		funProcessor[ data, inst]
	]

setupSetElementInstruction[data_, inst_] :=
	Module[ {},
		If[ inst["getProperty", "TrackedMemoryType"] === "MObject",
			setElementProcessor[data, inst]]
	]

setupStackAllocateInstruction[data_, inst_] :=
	Module[ {stackOper, tyEnv = data["typeEnvironment"]},
		If[ inst["getProperty", "TrackedMemoryType"] === "MObject",
			stackOper = getStackAllocateOperator[tyEnv];
			inst["setOperator", stackOper];
			inst["target"]["setType", tyEnv["resolve", TypeSpecifier[ "CArray"["MObject"]]]]]
	]



(*
 We are setting the type of the Stack to be Reference[ PackedVector[ Integer]], 
 but this is quite arbitrary.
*)
setupData[fm_] :=
	Module[ {data, tyEnv = fm["programModule"]["typeEnvironment"]},
		data = <|
				"active" -> CreateReference[ False],
				"typeEnvironment" -> tyEnv
				|>;
		data
	]




(*
  Add linkage information for the external definitions. 
  Normally this is set up by the ResolveFunctionCallPass,  but here it is done 
  directly by calling to the function/type mechanism.  At the least this gets the 
  current values, but it relies on the details which are somewhat internal.
*)
addLinkage[extDecls_, tyLookup_, funName_, isPoly_] :=
	Module[ {tys, defs},
		tys = If[ isPoly, tyLookup["getPolymorphicList", funName], tyLookup["getMonomorphicList", funName]];
		If[ Length[tys] < 1,
				ThrowException[CompilerException["Cannot find a single function type", funName,  tys]]];
		defs = First[ tys]["getProperty", "definition", Null];
		If[ defs === Null || !AssociationQ[defs],
			ThrowException[CompilerException["Cannot find function definition", funName]]];
		extDecls["addFunction", funName, defs]
	]

postProcess[ data_, fm_] :=
	Module[ {tyLookup, extDecls},
		If[ TrueQ[data["active"]["get"]],
			tyLookup = data["typeEnvironment"]["functionTypeLookup"];
			extDecls = fm["programModule"]["externalDeclarations"];
			addLinkage[extDecls, tyLookup, Native`PrimitiveFunction["BitCast"], True];
			addLinkage[extDecls, tyLookup, Native`PrimitiveFunction["Runtime_PushStackFrame"], False];
			addLinkage[extDecls, tyLookup, Native`PrimitiveFunction["Runtime_PopStackFrame"], False]];
	]


run[fm_, opts_] :=
	Module[{data},
		data = setupData[fm];
		CreateInstructionVisitor[
			data,
			<|
				"visitCallInstruction" -> setupCallInstruction,
				"visitSetElementInstruction" -> setupSetElementInstruction,
				"visitStackAllocateInstruction" -> setupStackAllocateInstruction
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		postProcess[data, fm];
	    fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"MObjectCollect",
		"The pass works on the result from MemoryManagePass to deal with garbage collection of MObjects."
];

MObjectCollectPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"postPasses" -> {}
|>];

RegisterPass[MObjectCollectPass];
]]

End[]

EndPackage[]


