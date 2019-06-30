
BeginPackage["Compile`Core`CodeGeneration`Backend`LLVM`MemoryManage`"]

MemoryManagePass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRegistry`"] (* for RegisterPass *)
Needs["Compile`Core`Transform`BasicBlockPhiReorder`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`IR`Instruction`ReturnInstruction`"]
Needs["Compile`Core`IR`Instruction`SetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]




acquireFun[] := CreateConstantValue[ Native`MemoryAcquire]
acquireStackFun[] := CreateConstantValue[ Native`MemoryAcquireStack]
releaseStackFun[] := CreateConstantValue[ Native`MemoryReleaseStack]
storeStackFun[] := CreateConstantValue[ Native`MemoryStoreStack]
initializeStack[] := CreateConstantValue[ Native`MemoryStackInitialize]
freeStack[] := CreateConstantValue[ Native`MemoryStackFree]


(*
  Add generic instructions for managing memory.  These work for refcounting,  MTensors and Expressions,  and for 
  MObject garbage collection.
  
  It is generic in that it doesn't specify the details of the functions added, and these are resolved in a later pass.
  
  The generic functions are
  
  stackarray = StackAllocate[size]
  stackarray[[pos]] = NULL
  ...
  Native`MemoryStackInitialize[stackarray, size] called after it is initialized.
  
  Native`MemoryStackFree[stackarray, size] called at the end.
  
  Native`MemoryAcquireStack[stackarray, pos, var] called when a varible needs to be acquired and should be stored on the stack.
  Native`MemoryStoreStack[stackarray, pos, var] called when a varible needs to be stored on the stack,  ie it is acquired elsewhere.
  Native`MemoryReleaseStack[stackarray, pos]  called when a position is passing out of scope,  maybe something else is being written
                                              or the stack is passing out of scope
                                              
  Native`MemoryAcquire[var]  called when a variable needs to be acquired,  this is done on the return var which is 
                             also stored in the stack.
  
  
  There are further potential optimizations such as clearing at the end of a BasicBlock.
  
  If the variable doesn't live beyond the BasicBlock we could just clear it at the end.
  
  There are some details about whether the refcount should be incremented on storing (needed 
  for Phi) or it already has an increment (for Call).
  
  We don't need to worry about LoadArgument or Copy instruction, because these will be counted 
  elsewhere.  Since we only free at the end that's fine,  but if we were more sophisticated in
  freeing we would handle free here.
  
  Typically everything is stored on one general stack,  but it is possible to have specific stacks.
  This is useful if the stack is to be used as a storage array to be passed elsewhere.
  
  The initial size can larger than 0,  to give extra storage locations which might be useful.
  
  Every instruction added has a TrackedMemory property of True and a TrackedMemoryType of the memory type.
  Except the StackAllocate which has a TrackedMemoryType of General for the common stack and the specific 
  memory type for unique stacks.
*)


$stackTypes=
<|
	"MTensor" -> "General",
	"Expression" -> "General",
	"MObject" -> "MObject"
|>


getInitialSize[data_, "MObject"] :=
	3

getInitialSize[data_, stackType_] :=
	0


(*
  Return the stack memory type.
*)
getMemoryStackType[ data_, type_] :=
	Module[ {stackTy},
		stackTy = Lookup[$stackTypes, type, Null];
		If[ stackTy === Null,
			ThrowException[CompilerException["Cannot find memory stack type ", type]]];
		stackTy
	]

isExpression[ var_] :=
	var["type"]["isConstructor", "Expression"]

isMTensor[var_] :=
	var["type"]["isNamedApplication", "PackedArray"] || 
	var["type"]["isConstructor", "MTensor"] || 
	(var["type"]["isNamedApplication", "Handle"] && First[var["type"]["arguments"]]["isConstructor", "SpanBase"])

(*
 Return the memory type.
*)
getTrackMemoryClass[ var_] :=
	Which[
		isMTensor[var],
			"MTensor"
		,
		isExpression[var],
			"Expression"
		,
		var["type"]["implementsQ", "MObjectManaged"],
			"MObject"
		,
		True,
			Null]

shouldTrackMemory[inst_, var_] :=
	!inst["hasProperty", "TrackedMemory"] && 
	isMTensor[var] || isExpression[var] || var["type"]["implementsQ", "MObjectManaged"]

setMemoryTracked[elem_] :=
	elem["setProperty", "TrackedMemory" -> True]

setInstructionData[data_, elem_, memType_] :=
	Module[ {},
		setMemoryTracked[elem];
		elem["setProperty", "TrackedMemoryType" -> memType]
	]	


createReleaseCall[data_, stackVar_, indexCons_, memType_] :=
	Module[{releaseInst},
		releaseInst = CreateCallInstruction[ "refdecr", releaseStackFun[], {stackVar, indexCons}];
		setInstructionData[data, releaseInst, memType];
		releaseInst
	]



$rawList =
<|
	Native`PrimitiveFunction["BitCast"] -> True,
	Native`LoadClosureVariable -> True
|>

(*
   If the inst doesn't come back with the result counted, then we need an acquire.
*)
shouldAcquire[inst_] :=
	Module[ {fun = inst["function"]},
		ConstantValueQ[fun] && KeyExistsQ[ $rawList, fun["value"]]
	]


(*
    We have
       v = something
       where v is a PackedArray (or some other object managed by refcounts)
       
     add
       ReleaseStack[ stack, ind]
       
     and
       StoreStack[ stack, ind, v]
     or
       AcquireStack[ stack, ind, v]
       
     AcquireStack is used if acquireQ is True. 

	 acquireQ is True if the something added a refcount for v and False otherwise.
	 So it is False for function calls (except for pass throughs like BitCast), the function
	 will increment the refcount of the result.
	 It is True for things like Phi or Copy instructions.
*)

fixTrackedTarget[data_, memType_, inst_, acquireQ_] :=
	Module[{trgt, index, indexCons, acquireInst, storeInst, releaseInst, stackVar, insertInst},
		setMemoryTracked[inst];
		trgt = inst["target"];
		index = GetStackIndex[ data, trgt, memType];
		stackVar = GetStackVariable[ data, memType];
		indexCons = CreateConstantValue[index];
		insertInst = If[ PhiInstructionQ[inst], 
							inst["basicBlock"][ "getProperty", "lastPhiInstruction", inst], inst];
		releaseInst = createReleaseCall[data, stackVar, indexCons, memType];		
		releaseInst["moveAfter", insertInst];

		If[ acquireQ,
			acquireInst = CreateCallInstruction[ "incr", acquireStackFun[], {stackVar, indexCons, trgt}];
			setInstructionData[data, acquireInst, memType];
			acquireInst["moveAfter", releaseInst];
			,
			storeInst = CreateCallInstruction[ "incr", storeStackFun[], {stackVar, indexCons, trgt}];
			setInstructionData[data, storeInst, memType];
			storeInst["moveAfter", releaseInst]];
	]
	

setupCallInstruction[data_, inst_] :=
	Module[ {memType = getTrackMemoryClass[inst["target"]]},
		If[shouldTrackMemory[inst, inst["target"]] && memType =!= Null,
			fixTrackedTarget[data, memType, inst, shouldAcquire[inst]]];
	]


testTrackedMemory[ data_, inst_, acquire_] :=
	Module[ {memType = getTrackMemoryClass[inst["target"]]},
		If[ shouldTrackMemory[inst, inst["target"]] && memType =!= Null,
			fixTrackedTarget[data, memType, inst, acquire]];
	]


setupPhiInstruction[data_, inst_] :=
	testTrackedMemory[data, inst, True]

setupLoadInstruction[data_, inst_] :=
	testTrackedMemory[data, inst, False]


createTypedConstant[ val_, ty_] :=
	Module[{cons = CreateConstantValue[val]},
		cons["setType", ty];
		cons
	]

createIntegerConstant[tyEnv_, val_] :=
	createTypedConstant[val, tyEnv["resolve", TypeSpecifier["MachineInteger"]]]

(*
  Finish the process.
  This adds the stack arrays with StackAllocate with postProcessDataTop, deals with the Return instruction 
  and then frees the contents of the stack arrays.
*)
postProcessData[ fm_, data_] :=
	Module[ {vals = data["memoryStackData"]["values"], bb, memBB, retBB, lastInst, shouldTrackReturnedMemory},
		Scan[
			postProcessDataTop[fm, data, #["initialSize"], #["size"]["get"], #["stackVariable"], #["memoryTypeMap"], #["memoryStackType"]]&,
			vals];
			
		(*
		 Now acquire the result if it needs tracking.
		 If it has not been added to the tracking system, then we still add an 
		 acquireFun call.
		*)
		bb = fm["lastBasicBlock"];
		lastInst = bb["lastInstruction"];
		Assert[InstructionQ[lastInst]];

		shouldTrackReturnedMemory = ReturnInstructionQ[lastInst]
		                            && shouldTrackMemory[lastInst, lastInst["value"]];
		
		If[ !shouldTrackReturnedMemory && Length[vals] === 0,
			(* There's no memory to track, so return *)
			Return[]
		];

		Which[
			InstructionQ[lastInst["previous"]],
				memBB = bb["splitAfter", lastInst["previous"]];
				memBB["setProperty", "memoryManage" -> True];
				fm["setLastBasicBlock", memBB];
				fm["setProperty", "memoryManageBasicBlock" -> memBB];
				,
			lastInst["previous"] === None,
				Null
				,
			True,
				ThrowException[{"Expected Instruction/previous to return another " <>
								"instruction or None. Got: ", lastInst["previous"]}]
		];
				
		If[ shouldTrackReturnedMemory,
			Module[ {
					val = lastInst["value"], 
					memType = getTrackMemoryClass[lastInst["value"]], 
					incrInst},
				setMemoryTracked[lastInst];
				incrInst = CreateCallInstruction[ "incr", acquireFun[], {val}];
				incrInst["setProperty", "TrackedMemoryReturnInstruction" -> True];
				setInstructionData[data, incrInst, memType];
				incrInst["moveBefore", lastInst];
			]];
			
		retBB = memBB["splitAfter", lastInst["previous"]];
		fm["setLastBasicBlock", retBB];
		lastInst = memBB["lastInstruction"];

		Scan[
			postProcessDataBottom[fm, data, lastInst, #["initialSize"], #["size"]["get"], #["stackVariable"], #["memoryTypeMap"], #["memoryStackType"]]&,
			vals];
	]

	
postProcessDataTop[ fm_, data_, initSize_, size_, stackVar_, memTypeMap_, memStackType_] :=
	Module[ {bb, inst, stackInst, i, nullConst, setInst, tyEnv, voidHandle, voidHandleArray, memType},
		tyEnv = data["typeEnvironment"];
		voidHandle = tyEnv["resolve", TypeSpecifier["VoidHandle"]];
		voidHandleArray = tyEnv["resolve", TypeSpecifier[ "CArray"[ "VoidHandle"]]];
		If[ initSize =!= size,
			bb = fm["firstBasicBlock"];
			inst = bb["firstNonLabelInstruction"];
			stackVar["setType", voidHandleArray];
			stackInst = CreateStackAllocateInstruction[stackVar, createIntegerConstant[tyEnv, size]];
			setInstructionData[data, stackInst, memStackType];
			stackInst["moveBefore", inst];
			inst = stackInst;
			(*
			 Assign NULL to every element
			 perhaps do this in a loop if very large?
			*)
			Do[
				memType = memTypeMap["lookup", i, Null];
				nullConst = createTypedConstant[Compile`NullReference, voidHandle];
				setInst = CreateSetElementInstruction[stackVar, createIntegerConstant[tyEnv, i], nullConst];
				setInstructionData[data, setInst, memType];
				setInst["moveAfter", inst];
				inst = setInst;
				,
				{i,initSize, size-1}];
			stackInst = CreateCallInstruction[ "incr", initializeStack[], {stackVar, createIntegerConstant[tyEnv, size]}];
			setInstructionData[data, stackInst, memType];
			stackInst["moveAfter", inst];
			inst = stackInst;
		];

	]

postProcessDataBottom[ fm_, data_, lastInst_, initSize_, size_, stackVar_, memTypeMap_, memStackType_] :=
	Module[ {stackInst, i, relInst, tyEnv, memType},
		tyEnv = data["typeEnvironment"];
		If[ initSize =!= size,
			Do[
				memType = memTypeMap["lookup", i, Null];
				relInst = createReleaseCall[data, stackVar, CreateConstantValue[i], memType];
				relInst["moveBefore", lastInst];
				,
				{i,initSize,size-1}];
			stackInst = CreateCallInstruction[ "incr", freeStack[], {stackVar, createIntegerConstant[tyEnv, size]}];
			setInstructionData[data, stackInst, memType];
			stackInst["moveBefore", lastInst];
		];
	]

(*
 Return the stack array variable for a given memType.
*)
GetStackVariable[ data_, memType_] :=
	Module[ {memStackType, ent},
		memStackType = getMemoryStackType[data, memType];
		ent = data["memoryStackData"]["lookup", memStackType, Null];
		If[ ent === Null,
			ThrowException[CompilerException["Cannot find StackVariabler", memType]]];
		ent["stackVariable"]
	]

(*
 Return the index in the stack array for variable var of memtype
*)
GetStackIndex[ data_, var_, memType_] :=
	Module[{index, memStackType, ent},
		index = data["indexMap"]["lookup", var["id"], Null];
		If[ index === Null,	
			memStackType = getMemoryStackType[data, memType];
			ent = data["memoryStackData"]["lookup", memStackType, Null];
			If[ ent === Null,
				Module[ {stackVar = CreateVariable[], initSize},
					stackVar["setName", memStackType];
					initSize = getInitialSize[data, memStackType];
					ent = <|"memoryStackType" -> memStackType,
							"memoryTypeMap" -> CreateReference[<||>],
							"stackVariable" -> stackVar, 
							"initialSize" -> initSize,
							"size" -> CreateReference[initSize]|>;
					data["memoryStackData"]["associateTo", memStackType -> ent];
				]];
			
			index = ent["size"]["increment"];
			data["indexMap"]["associateTo", var["id"] -> index];
			ent["memoryTypeMap"]["associateTo", index -> memType]
			];
		index
	]

setupData[fm_] :=
	Module[ {data, stackVar, tyEnv = fm["programModule"]["typeEnvironment"]},
		stackVar = CreateVariable[];
		stackVar["setName", "MTensorArray"];
		data = <|
					"memoryStackData" -> CreateReference[<||>],
					"indexMap" -> CreateReference[<||>], 
					"variableMap" -> CreateReference[<||>], 
					"typeEnvironment" -> tyEnv|>;
		data
	]


run[fm_, opts_] :=
	Module[{data, manageMemory, passOpts = Lookup[opts, "PassOptions", {}]},
		manageMemory = Lookup[ passOpts, "ManageMemory", True];
		If[ TrueQ[manageMemory],
			data = setupData[fm];
			CreateInstructionVisitor[
				data,
				<|
					"visitLoadInstruction" -> setupLoadInstruction,
					"visitCallInstruction" -> setupCallInstruction,
					"visitPhiInstruction" -> setupPhiInstruction
				|>,
				fm,
				"IgnoreRequiredInstructions" -> True
			];
			postProcessData[ fm, data]];
	    fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"MemoryManage",
		"The pass sets up memory management insertions."
];

MemoryManagePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"postPasses" -> {},
	"requires" -> {
		BasicBlockPhiReorderPass
	}
|>];

RegisterPass[MemoryManagePass]
]]

End[]

EndPackage[]
