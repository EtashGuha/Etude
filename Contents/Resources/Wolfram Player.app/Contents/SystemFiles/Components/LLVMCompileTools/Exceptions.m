

BeginPackage["LLVMCompileTools`Exceptions`"]

InitializeLandingPad
AddLandingPad
AddResume
AddRuntimeFunctionInvokeModel
AddBuildInvokeModel
ExceptionsStyleValue
ResolveExceptionsModel

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMTools`"]
Needs["LLVMCompileTools`Globals`"]

(*
  This function should really live in a code gen repo to be shared between 
  the Compiler and any code generation repos.  Here is fine for now.
*)
ResolveExceptionsModel[model_] :=
	If[model === Automatic, 
		If[$OperatingSystem === "Windows" || $SystemID === "Linux-ARM", "Basic", "Itanium"],
		model]
	
	
getModel[data_] :=
	Module[ {model = Lookup[data, "exceptionsModel", Automatic]},
		model = ResolveExceptionsModel[model];
		(*
		 Model should be StringQ or None
		*)
		If[ !StringQ[model] && model =!= None,
			ThrowException[{"Unexpected value for ExceptionsModel.", model}]];
		model
		]	  

InitializeLandingPad[data_] :=
	Module[ { model = getModel[data]},
		InitializeLandingPad[data, model]
	]



InitializeLandingPad[data_, "Itanium"] :=
	Module[ {persFunTy, persFunId, currFunId},
		persFunId = data["landingPadPersonality"]["get"];
		If[ persFunId === Null,
			persFunTy = WrapIntegerArray[LLVMLibraryFunction["LLVMFunctionType"][GetIntegerType[data, 32], #, 0, 1]&, {}];
        	persFunId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], "__gxx_personality_v0", persFunTy];
        	persFunId = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], persFunId, GetVoidHandleType[data], ""];
        	data["landingPadPersonality"]["set", persFunId]];
        currFunId = data["functionId"]["get"];
        LLVMLibraryFunction["LLVMSetPersonalityFn"][currFunId, persFunId];
        LLVMAddFunctionAttribute[ data, currFunId, "uwtable"];
	]

InitializeLandingPad[data_, "Windows"] :=
	Module[ {persFunTy, persFunId, currFunId},
		persFunId = data["landingPadPersonality"]["get"];
		If[ persFunId === Null,
			persFunTy = WrapIntegerArray[LLVMLibraryFunction["LLVMFunctionType"][GetIntegerType[data, 32], #, 0, 1]&, {}];
        	persFunId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], "__CxxFrameHandler3", persFunTy];
        	persFunId = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], persFunId, GetVoidHandleType[data], ""];
        	data["landingPadPersonality"]["set", persFunId]];
		currFunId = data["functionId"]["get"];
		LLVMLibraryFunction["LLVMSetPersonalityFn"][currFunId, persFunId];
		LLVMAddFunctionAttribute[ data, currFunId, "uwtable"];	
	]

InitializeLandingPad[data_, "Basic"] :=
	Module[ {},
		0
	]

InitializeLandingPad[data_, "SetJumpLongJump"] :=
	Module[ {},
		0
	]

InitializeLandingPad[data_, None] :=
    Module[ {},
        0
    ]

InitializeLandingPad[data_, model_] :=
	Module[ {},
		ThrowException[{"Unknown ExceptionsModel", model}]
	]

AddLandingPad[data_] :=
	Module[ { model = getModel[data]},
		AddLandingPad[data, model]
	]

AddLandingPad[data_, "Itanium"] :=
	Module[ {ty, id},
		ty = GetStructureType[data, {GetVoidHandleType[data], GetIntegerType[data, 32]}];
		id = LLVMLibraryFunction["LLVMBuildLandingPad"][data["builderId"], ty, 0, 0, ""];
		LLVMLibraryFunction["LLVMSetCleanup"][id, 1];
		id
	]

AddLandingPad[data_, "Windows"] :=
	Module[ {ty, id},
		ty = GetStructureType[data, {GetVoidHandleType[data], GetIntegerType[data, 32]}];
		id = LLVMLibraryFunction["LLVMBuildCleanupPad"][data["builderId"], 0, 0, 0, ""];
		id
	]

AddLandingPad[data_, "Basic"] :=
	Module[ {},
		0
	]

AddLandingPad[data_, "SetJumpLongJump"] :=
	Module[ {},
		0
	]
	
AddLandingPad[data_, None] :=
    Module[ {},
        0
    ]

AddLandingPad[data_, model_] :=
	Module[ {},
		ThrowException[{"Unknown ExceptionsModel", model}]
	]



AddResume[data_, val_] :=
	Module[ { model = getModel[data]},
		AddResume[data, val, model]
	]

AddResume[data_, val_, "Itanium"] :=
	LLVMLibraryFunction["LLVMBuildResume"][data["builderId"], val]

AddResume[data_, val_, "Windows"] :=
	LLVMLibraryFunction["LLVMBuildCleanupRet"][data["builderId"], val, 0]

AddResume[data_, val_, "Basic"] :=
	Module[ {funId, funTy, retTy, zeroId},
		If[ data["functionReturnVoid"],
			LLVMLibraryFunction["LLVMBuildRetVoid"][data["builderId"]]
			,
			funId = data["functionId"]["get"];
			funTy = LLVMLibraryFunction["LLVMTypeOf"][funId];
			funTy = LLVMLibraryFunction["LLVMGetElementType"][funTy];
			retTy = LLVMLibraryFunction["LLVMGetReturnType"][funTy];
			zeroId = LLVMLibraryFunction["LLVMConstNull"][retTy];
			LLVMLibraryFunction["LLVMBuildRet"][data["builderId"], zeroId]]
	]

AddResume[data_, val_, "SetJumpLongJump"] :=
	Module[ {},
		LLVMLibraryFunction["LLVMBuildUnreachable"][data["builderId"]];
	]
	
AddResume[data_, val_, None] :=
    Module[{}, 0]


AddResume[data_, val_, model_] :=
	Module[ {},
		ThrowException[{"Unknown ExceptionsModel", model}]
	]


AddRuntimeFunctionInvokeModel[ data_?AssociationQ, name_, inputs_, toBB_, unwindBB_] :=
	Module[ {model = getModel[data]},
		Which[
            model === None,
                0
            ,
			model === "Basic",
				AddRuntimeFunctionInvokeBasic[data, name, inputs, toBB, unwindBB]
			,
			model === "SetJumpLongJump",
				AddRuntimeFunctionInvokeSetJumpLongJump[data, name, inputs, toBB, unwindBB]
			,
			True,
				AddRuntimeFunctionInvoke[data, name, inputs, toBB, unwindBB]]
	]

AddRuntimeFunctionInvokeBasic[data_?AssociationQ, name_, inputs_, toBB_, unwindBB_] :=
	Module[ {exceptionActive, resId, id, comp, eqOp, test},
		resId = AddRuntimeFunctionCall[ data, name, inputs];
		exceptionActive = AddRuntimeFunctionCall[ data, "getExceptionActive", {}];
		comp = AddConstantInteger[data, 32, 0];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, exceptionActive, comp, ""];	
		id = LLVMLibraryFunction["LLVMBuildCondBr"][data["builderId"], test, toBB, unwindBB];
		resId
	]

(*
  Should not be reached.
*)
AddRuntimeFunctionInvokeSetJumpLongJump[data_?AssociationQ, name_, inputs_, toBB_, unwindBB_] :=
	Module[ {},
		ThrowException[{"An invoke instruction should never be created for SetJump LongJump exceptions"}]
	]

AddBuildInvokeModel[ data_?AssociationQ, name_, inputs_, toBB_, unwindBB_] :=
	Module[ {model = getModel[data]},
		Which[
            model === None,
                0
            ,
			model === "Basic",
				AddBuildInvokeBasic[data, name, inputs, toBB, unwindBB]
			,
			model === "SetJumpLongJump",
				AddBuildInvokeSetJumpLongJump[data, name, inputs, toBB, unwindBB]
			,
			True,
				AddBuildInvoke[data, name, inputs, toBB, unwindBB]]
	]


AddBuildInvokeBasic[data_?AssociationQ, name_, inputs_, toBB_, unwindBB_] :=
	Module[ {exceptionActive, resId, id, comp, eqOp, test},
		resId = AddBuildCall[ data, name, inputs];
		exceptionActive = AddRuntimeFunctionCall[ data, "getExceptionActive", {}];
		comp = AddConstantInteger[data, 32, 0];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, exceptionActive, comp, ""];		
		id = LLVMLibraryFunction["LLVMBuildCondBr"][data["builderId"], test, toBB, unwindBB];
		resId
	]


(*
 return the value for the setExceptionStyle RTL method based on the ExceptionsModel setting.
*)
ExceptionsStyleValue[data_] :=
	Module[ {model = getModel[data], val},
		val = Switch[model,
				"Basic", 1
				,
				"SetJumpLongJump", 2
				,
				(*
				  Everything else including None will use the default exception mechanism.
				*)
				_, 0
		];
		AddConstantInteger[data, 32, val]	
	]


End[]


EndPackage[]

