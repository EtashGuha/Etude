
BeginPackage["Compile`Core`CodeGeneration`Backend`LLVM`ExpressionRefCount`"]


ExpressionRefCountPass

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
Needs["Compile`Core`IR`Instruction`GetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]






getSetElementOperator[tyEnv_] :=
	Module[ {setOper},
		setOper = CreateConstantValue[Native`SetElement];
		setOper["setType", tyEnv["resolve", TypeSpecifier[{"CArray"["VoidHandle"], "MachineInteger", "VoidHandle"} -> "Void"]]];
		setOper
	]

getGetElementOperator[tyEnv_] :=
	Module[ {getOper},
		getOper = CreateConstantValue[Native`GetElement];
		getOper["setType", tyEnv["resolve", TypeSpecifier[{"CArray"["VoidHandle"], "MachineInteger"} -> "VoidHandle"]]];
		getOper
	]

getStackAllocateOperator[tyEnv_] :=
	Module[ {setOper},
		setOper = CreateConstantValue[Native`StackAllocate];
		setOper["setType", tyEnv["resolve", TypeSpecifier[{ "MachineInteger"} -> "CArray"["VoidHandle"]]]];
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
   Cast the Native`EternalMTensor to VoidHandle and set as the element.
*)
setElementProcessor[data_, inst_] :=
	Module[ {tenCon, castInst, setOper},
		data["active"]["set", True];	
		tenCon = createTypedConstant[ data, Compile`Internal`ENULLReference, "Expression"];
		castInst = createBitCastTo[data, tenCon, "VoidHandle"];
		castInst["moveBefore", inst];
		inst["setSource", castInst["target"]];
		setOper = getSetElementOperator[data["typeEnvironment"]];
		inst["setOperator", setOper];

	]


setStackAllocate[data_, inst_] :=
	Module[ {oper},
		oper = getStackAllocateOperator[data["typeEnvironment"]];
		inst["setOperator", oper];
	]


(*
  "void" = Call Native`MemoryReleaseStack[ stackArray, index]
  convert to
  
   voidTrgt = GetElement[ stackArray, index];
   expr = Call BitCast voidTrgt -> Expression
   Call Native`PrimitiveFunction["DecrementReferenceCount"][expr]
*)
(*
  Add linkage info for external calls
*)
releaseStackProcessor[data_, inst_] :=
	Module[ {tyEnv = data["typeEnvironment"], getOper, getInst, castInst, refInst},
		data["active"]["set", True];
		getOper = getGetElementOperator[ tyEnv];
		getInst = CreateGetElementInstruction[ "release", inst["getArgument", 1], inst["getArgument", 2], getOper];
		getInst["target"]["setType", tyEnv["resolve", TypeSpecifier["VoidHandle"]]];
		getInst["moveBefore", inst];
		castInst = createBitCastTo[data, getInst["target"], "Expression"];
		castInst["moveBefore", inst];
		refInst = createCallInstruction[ data, Native`PrimitiveFunction["DecrementReferenceCount"], {castInst["target"]}, "UnsignedInteger64"];
		refInst["moveBefore", inst];
		inst["unlink"];
	]


(*
  "void" = Call Native`MemoryStoreStack[ stackArray, index, value]
  convert to
   
   voidTrgt = Call BitCast * -> VoidHandle
   SetElement[ stackArray, index, voidTrgt]
*)
storeStackProcessor[data_, inst_] :=
	Module[ {tyEnv = data["typeEnvironment"], setOper, castInst, setInst},
		data["active"]["set", True];
		castInst = createBitCastTo[data, inst["getArgument", 3], "VoidHandle"];
		castInst["moveBefore", inst];
		setOper = getSetElementOperator[tyEnv];
		setInst = CreateSetElementInstruction[inst["getArgument", 1], inst["getArgument", 2], castInst["target"], setOper];
		setInst["moveBefore", inst];
		inst["unlink"];
	]

(*
  "void" = Call Native`MemoryAcquireStack[ stackArray, index, value]
  convert to
   mten = Call BitCast * -> MTensor
   Call Native`PrimitiveFunction["MTensorRefCountIncrement"][mten]
   voidTrgt = Call BitCast * -> VoidHandle
   SetElement[ stackArray, index, voidTrgt]
*)
acquireStackProcessor[data_, inst_] :=
	Module[ {},
		data["active"]["set", True];
		acquireWorker[data, inst, inst["getArgument", 3]];
		storeStackProcessor[data, inst];
	]

(*
  "void" = Call Native`MemoryAcquire[ value]
  convert to
   mten = Call BitCast * -> MTensor
   Call Native`PrimitiveFunction["MTensorRefCountIncrement"][mten]
*)
acquireProcessor[data_, inst_] :=
	Module[ {},
		data["active"]["set", True];
		acquireWorker[data, inst, inst["getArgument", 1]];
		inst["unlink"];
	]

acquireWorker[ data_, inst_, value_] :=
	Module[ {castInst, refInst},
		castInst = createBitCastTo[data, value, "Expression"];
		castInst["moveBefore", inst];
		refInst = createCallInstruction[ data, Native`PrimitiveFunction["IncrementReferenceCount"], {castInst["target"]}, "UnsignedInteger64"];
		refInst["moveBefore", inst];
	]


passThroughProcessor[ data_, inst_] :=
	Null

removeProcessor[ data_, inst_] :=
	inst["unlink"]


$callprocessors =
<|
	Native`MemoryReleaseStack -> releaseStackProcessor,
	Native`MemoryStoreStack -> storeStackProcessor,
	Native`MemoryAcquireStack -> acquireStackProcessor,
	Native`MemoryAcquire -> acquireProcessor,
	Native`MemoryStackInitialize -> removeProcessor,
	Native`MemoryStackFree -> removeProcessor
|>


getFunctionProcessor[data_, inst_] :=
	Module[ {fun = inst["function"], funValue},
		If[ inst["getProperty", "TrackedMemoryType"] === "Expression",
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
		If[ inst["getProperty", "TrackedMemoryType"] === "Expression",
			setElementProcessor[data, inst]]
	]

(*
  General means it shares a stack with similar,  the stack itself has a type of CArray[VoidHandle]
*)
setupStackAllocateInstruction[data_, inst_] :=
	Module[ {},
		If[ inst["getProperty", "TrackedMemoryType"] === "General",
			setStackAllocate[data, inst]]
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
			addLinkage[extDecls, tyLookup, Native`PrimitiveFunction["DecrementReferenceCount"], False];
			addLinkage[extDecls, tyLookup, Native`PrimitiveFunction["IncrementReferenceCount"], False];
			addLinkage[extDecls, tyLookup, Native`PrimitiveFunction["BitCast"], True]];
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
		"ExpressionRefCount",
		"The pass works on the result from MemoryManagePass to deal with Expression refcounting."
];

ExpressionRefCountPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"postPasses" -> {}
|>];

RegisterPass[ExpressionRefCountPass];
]]

End[]

EndPackage[]


