

BeginPackage["LLVMCompileTools`CreateWrapper`"]

CreateWrapper
InitializeWrapper
FinalizeWrapper


CreateWrapperFunction
CreateInitialization


InitializeInitialization
CompleteInitialization


Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"];
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`Globals`"]
Needs["LLVMCompileTools`Types`"];
Needs["LLVMCompileTools`ExprFunctions`"];
Needs["LLVMCompileTools`PackedArrayExprFunctions`"]
Needs["LLVMCompileTools`ExprExprFunctions`"]
Needs["LLVMCompileTools`MTensor`"]
Needs["LLVMCompileTools`NumericArray`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]
Needs["LLVMCompileTools`Exceptions`"]

(*
 CreateWrapper is used to create the wrapper function with calls to 
 the LLVM API (and not by generating and function and compiling it).
 
 Returns the names of the wrapper and the initialization function.
*)

createWrapperCall[ data_, funName_, argTypes_List, retType_] :=
	Module[ {name, argTys, resTy, funTy, funId, funArg1Id, bbId, argIds, resId, resFun, resVoidId, callFunId, resRefId},
		name = funName <> "_Wrapper_Call";
		argTys = GetPointerType[data, GetVoidHandleType[data]];
        resTy = GetVoidType[data];
        funTy = WrapIntegerArray[LLVMLibraryFunction["LLVMFunctionType"][resTy, #, 2, 0]&, {GetIntegerType[data, 32], argTys}];
        funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], name, funTy];
        LLVMAddFunctionParameterAttribute[data, funId, "noalias", 2];
        internalizeFunction[data, funId];
        AddFunctionSystemMetadata[data, funId];
        funArg1Id = LLVMLibraryFunction["LLVMGetParam"][funId, 1];
        bbId = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], funId, data["getBBName"]["main"]];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbId];
        argIds =
        	MapIndexed[
        		Module[{argType, index, funs, argVoidId, off, argTy, argRefId, argId},
					argType = #1;
					index = First[#2];
					funs = getTypeFunctions[argType];
					off = AddConstantInteger[data, 64, index];
					argVoidId = AddGetArray[data, funArg1Id, off];
					argTy = GetPointerType[data, funs["getType"][data, argType]];
					argRefId = AddTypeCast[data, argVoidId, argTy];
					argId = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], argRefId, "arg_" <> ToString[index]];
					argId
        		]&,
        		argTypes
        	];
		callFunId = LLVMLibraryFunction["LLVMGetNamedFunction"][data["moduleId"], funName];
		resId = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildCall"][data["builderId"], callFunId, #, Length[argIds], ""]&, argIds];
		If[ retType =!= "Void",
			resFun = getTypeFunctions[retType];
			resVoidId = AddGetArray[data, funArg1Id, AddConstantInteger[data, 64, Length[argTypes]+1]];
			resTy = GetPointerType[data, resFun["getType"][data, retType]];
			resRefId = AddTypeCast[data, resVoidId, resTy];
			LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], resId, resRefId]];
		LLVMLibraryFunction["LLVMBuildRetVoid"][data["builderId"]];
		{funId, funTy}
	]


buildStackSpaceTy[data_, argTy_, argArray_, index_] :=
	Module[ {argRefId, off, argRef},
		argRefId = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], argTy, AddConstantInteger[data, 32,1],""];
		off = AddConstantInteger[data, 64, index];
		argRef = AddTypeCast[data, argRefId, GetVoidHandleType[data]];
		AddSetArray[ data, argArray, off, argRef];
		argRefId
	]

buildStackSpace[data_, inpTy_, funs_, argArray_, index_] :=
	Module[ {argTy, argRefId},
		argTy = funs["getType"][data, inpTy];
		argRefId = buildStackSpaceTy[ data, argTy, argArray, index];
		argRefId
	]

InitializeWrapper[ data_, mainName_] :=
	Module[{initName, initId},
		CreateInitializeRTL[data];
		initName = mainName <> "_Initialization";
		initId = InitializeInitialization[ data, initName];
		CreateGlobal[data, "initializationDone", GetBooleanType[data], AddConstantBoolean[data, False]];
		{initName, initId}
	]

FinalizeWrapper[ data_, initId_] :=
	Module[{},
		CompleteInitialization[data, initId];
		FinalizeModule[data];
	]
	
getExceptionModel := getExceptionModel = LLVMCompileTools`Exceptions`Private`getModel

CreateWrapper[ dataIn_?AssociationQ, funName_, argTypes_List, retType_] :=
	Module[ {wrapperName, data = dataIn, initDoneId, wrapFunId,
				id1, id2, arg1Id, arg2Id, eqOp, argData, index, 
				funs, funId, funTy, resId, resExprId, argArray, argArrayTy, 
				resRefId, callFunRefId, retId, errTy, mainFunId},
		errTy = GetCompilerErrorType[data];
		wrapperName = funName <> "_Wrapper";
		{funId, funTy} = createWrapperCall[ data, funName, argTypes, retType];
		wrapFunId = CreateExprWrapperFunction[data, wrapperName];
        data["functionId"]["set", wrapFunId];
        initDoneId = GetExistingGlobal[ data, "initializationDone"];
        AddIfFalseBasicBlock[data, initDoneId, 
        	Function[ {dataArg, bbAfter},
        		Module[ {id},
        			id = CreateCompilerError[dataArg, 1, 0];
        			LLVMLibraryFunction["LLVMBuildRet"][dataArg["builderId"], id];
        		]]];
        If[getExceptionModel[data] =!= None,
            AddRuntimeFunctionCall[ data, "setExceptionStyle", {ExceptionsStyleValue[data]}]
        ];
		arg1Id = LLVMLibraryFunction["LLVMGetParam"][wrapFunId, 0];
		arg2Id = LLVMLibraryFunction["LLVMGetParam"][wrapFunId, 1];
		id1 = AddExprLength[data, Null, {arg1Id}];
		id2 = LLVMLibraryFunction["LLVMConstInt"][GetMIntType[data], Length[argTypes], 0];
		eqOp = data["LLVMIntPredicate"][SameQ];
		id1 = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id1, id2, ""];
		AddIfFalseBasicBlock[data, id1, 
        	Function[ {dataArg, bbAfter},
        		Module[ {id},
					id = CreateCompilerError[dataArg, 2, Length[argTypes]];
        			LLVMLibraryFunction["LLVMBuildRet"][dataArg["builderId"], id];
        		]]];
        argArrayTy = GetVoidHandleType[data];
        
        (*
          Create an array for function to call, arguments and result.
        *)
        argArray = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], argArrayTy, AddConstantInteger[data, 32,Length[argTypes]+2],""];
        argData =
            MapIndexed[
                Module[ {argType, argId, testId, argRefId},
                    argType = #1;
                    index = First[#2];
                    funs = getTypeFunctions[argType];
                    id1 = AddExprPart[data, arg1Id, index, False];
                    argRefId = buildStackSpace[data, argType, funs, argArray, index];
                    testId = funs["testGet"][data, argType, argRefId, id1];
                    AddIfFalseBasicBlock[data, testId, 
                        Function[ {dataArg, bbAfter},
                            Module[ {id},
                                id = CreateCompilerError[dataArg, 3, index];
                                LLVMLibraryFunction["LLVMBuildRet"][dataArg["builderId"], id];
                        ]]];
                    argId = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], argRefId, ""];
                    <| "argId" -> argId, "argType" -> argType |>
                ]&
                ,
                argTypes];
                
	            
		funs = getTypeFunctions[retType];
		resRefId = buildStackSpace[data, retType, funs, argArray, Length[argTypes]+1];
		
		callFunRefId = buildStackSpaceTy[data, GetPointerType[ data,funTy], argArray, 0];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], funId, callFunRefId];

		retId = AddRuntimeFunctionCall[ data, "catchExceptionHandler", {AddConstantInteger[data, 32,Length[argTypes]], argArray}];
        eqOp = data["LLVMIntPredicate"][SameQ];
        id1 = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, AddNull[data, errTy], retId, ""];
        AddIfTrueBasicBlock[data, id1, 
            Function[ {dataArg, bbAfter},
                resId = LLVMLibraryFunction["LLVMBuildLoad"][dataArg["builderId"], resRefId, ""];
                resExprId = funs["toExpr"][dataArg, retType, resId];
                funs["collect"][dataArg, retType, resId];
                LLVMLibraryFunction["LLVMBuildStore"][dataArg["builderId"], resExprId, arg2Id];
                LLVMLibraryFunction["LLVMBuildBr"][dataArg["builderId"], bbAfter];
            ]]; 
		
		Scan[ getTypeFunctions[#["argType"]]["collect"][data, #["argType"], #["argId"]]&,  argData];
		AddFunctionCall[data, "Runtime_RunGarbageCollect", {}];
		LLVMLibraryFunction["LLVMBuildRet"][data["builderId"], retId];
		wrapperName
	]


CreateExprWrapperFunction[dataIn_?AssociationQ, name_] :=
    Module[ {resTy, t2, t3, ty, funId, bbId, data = dataIn},
        resTy = GetCompilerErrorType[data];
        t2 = GetBaseExprType[data];
        t3 = GetHandleType[ data, GetBaseExprType[data]];
        ty = WrapIntegerArray[LLVMLibraryFunction["LLVMFunctionType"][resTy, #, 2, 0]&, {t2, t3}];
        funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], name, ty];
        LLVMAddFunctionParameterAttribute[data, funId, "noalias", 0]; (* Return Type *)
        LLVMAddFunctionParameterAttribute[data, funId, "noalias", 1];
        LLVMAddFunctionParameterAttribute[data, funId, "noalias", 2];
        externalizeFunction[data, funId];
        bbId = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], funId, data["getBBName"]["main"]];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbId];
        funId
    ]

CreateCompilerError[data_?AssociationQ, val_, subval_] :=
	Module[{retId},
		retId = AddRuntimeFunctionCall[ data, "New_CompilerError", {AddConstantInteger[data, 32, val], AddConstantInteger[data, 32, subval]}];
		retId
	]

(* makes a function internal to the module *)
internalizeFunction[data_, funId_] :=
    Module[{},
	    AddLLVMGlobalLinkageAttribute[data, funId, "LLVMPrivateLinkage"];
        (*
            private linkage requires default visibility
            bug 367178
        *)
        AddLLVMGlobalVisibilityAttribute[data, funId, "LLVMDefaultVisibility"];
	    AddLLVMGlobalUnnamedAddressAttribute[data, funId, "LLVMGlobalUnnamedAddr"];
        AddLLVMDLLStorageClass[data, funId, "LLVMDefaultStorageClass"];
	    0
    ];
(* makes a function external to the module *)
externalizeFunction[data_, funId_] :=
    Module[{},
        AddLLVMGlobalLinkageAttribute[data, funId, "LLVMExternalLinkage"];
        AddLLVMGlobalVisibilityAttribute[data, funId, "LLVMDefaultVisibility"];
        AddLLVMGlobalUnnamedAddressAttribute[data, funId, "LLVMNoUnnamedAddr"];
        AddLLVMDLLStorageClass[data, funId, "LLVMDLLExportStorageClass"];
        0
    ];

(*
  Create a function that initializes the RTL used by the Compiler. 
  See the comments in LLVMCompileTools that discusses this more.
*)
CreateInitializeRTL[ data_?AssociationQ] :=
	Module[ {argTy, resTy, funTy, name, funId, bbId, arg1Id, ptrId, args},
		name = "initializeRTL";
		resTy = GetMIntType[data];
		argTy = GetMIntType[data];
        funTy = WrapIntegerArray[LLVMLibraryFunction["LLVMFunctionType"][resTy, #, 3, 0]&, {argTy, argTy, argTy}];
        funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], name, funTy];
        LLVMAddFunctionAttribute[data, funId, "cold"];
        LLVMAddFunctionParameterAttribute[data, funId, "noalias", #]& /@ Range[Length[args]];
        externalizeFunction[data, funId];
        AddFunctionSystemMetadata[data, funId];
        data["functionId"]["set", funId];
        bbId = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], funId, data["getBBName"]["main"]];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbId];
        args =
        	Table[
				arg1Id = LLVMLibraryFunction["LLVMGetParam"][funId, i];
				ptrId = LLVMLibraryFunction["LLVMBuildIntToPtr"][data["builderId"], arg1Id, 
							GetVoidHandleType[data], "Cast"];
				ptrId, {i, 0, 2}];
		AddRuntimeFunctionCall[ data, "InitializeWolframRTL", args];	
		LLVMLibraryFunction["LLVMBuildRet"][data["builderId"], AddConstantMInt[data, 0]];
	]



nullHandler = Null&
ClearAll[$handlers]
$handlers := $handlers = <|
 "Boolean" -> <| "testGet" -> AddTestGetBooleanExpr, "getType" -> (GetBooleanType[#]&),  "toExpr" -> AddMBooleanToExpr, "collect" -> nullHandler|>,
 "Integer64" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,64]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "Integer32" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,32]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "Integer16" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,16]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "Integer8" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,8]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "UnsignedInteger64" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,64]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "UnsignedInteger32" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,32]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "UnsignedInteger16" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,16]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 "UnsignedInteger8" ->  <|"testGet" -> AddTestGetIntegerExpr, "getType" -> (GetIntegerType[#,8]&), "toExpr" -> AddMIntegerToExpr, "collect" -> nullHandler |>,
 (*"Real16" ->  <|"testGet" -> AddTestGetRealExpr, "getType" -> (GetRealType[#, 16]&), "toExpr" -> AddRealToExpr["Real16"], "collect" -> nullHandler |>,*)
 "Real32" ->  <|"testGet" -> AddTestGetRealExpr, "getType" -> (GetRealType[#, 32]&), "toExpr" -> AddRealToExpr["Real32"], "collect" -> nullHandler |>,
 "Real64" ->  <|"testGet" -> AddTestGetRealExpr, "getType" -> (GetRealType[#, 64]&), "toExpr" -> AddRealToExpr["Real64"], "collect" -> nullHandler |>,
 (*"Real128" ->  <|"testGet" -> AddTestGetRealExpr, "getType" -> (GetRealType[#, 128]&), "toExpr" -> AddRealToExpr["Real128"], "collect" -> nullHandler |>,*)
 "Complex" ->  <|"testGet" -> AddTestGetComplexExpr, "getType" -> (GetMComplexType[#1]&),  "toExpr" -> AddMComplexToExpr, "fromExpr" -> AddMComplexFromExpr,  "collect" -> nullHandler |>,
 "String" -> <|"testGet" -> AddTestGetStringExpr, "getType" -> (GetMStringType[#1]&),  "toExpr" -> AddMStringToExpr,  "collect" -> nullHandler |>,
 "PackedArray" ->  <|"testGet" -> AddTestGetMTensorExpr, "getType" -> (GetMTensorType[#1]&),"toExpr" -> AddPackedArrayToExpr,  "collect" -> AddMTensorRelease |>,
 "NumericArray" ->  <|"testGet" -> AddTestGetNumericArrayExpr, "getType" -> (GetMNumericArrayType[#1]&),"toExpr" -> AddNumericArrayToExpr,  "collect" -> nullHandler |>,
 "Expression" -> <|"testGet" -> AddTestGetExprExpr, "getType" -> (GetBaseExprType[#1]&),   "toExpr" -> AddExprToExpr,  "collect" -> nullHandler |>,
 "Function" -> <|"testGet" -> AddTestGetFunctionExpr, "getType" -> (GetBaseFunctionType[#1, #2]&),   "toExpr" -> AddFunctionToExpr,  "collect" -> nullHandler |>,
 "Void" -> <|"testGet" -> AddTestGetVoidExpr, "getType" -> (GetBaseExprType[#1]&),   "toExpr" -> AddVoidToExpr,  "collect" -> nullHandler |>
|>


getHead[args_ -> body_] :=
	"Function"


getHead[ty_[___]] :=
	ty
	
getHead[ ty_] :=
	ty

getTypeFunctions[ type_] :=
	Module[ {typeHead, handler},
		typeHead = getHead[type];
		handler = Lookup[ $handlers, typeHead, Null];
		If[ handler === Null,
			ThrowException[{"Type not handled ", type}]
		];
		handler
	]


(*
 InitializeInitialization is called to setup the initialization function.
*)
InitializeInitialization[data_?AssociationQ, name_] :=
	Module[ {funRes, funTy, funId, voidHandleArrayTy},
    	funRes = GetIntegerType[data, 32];
		voidHandleArrayTy = GetPointerType[ data, GetVoidHandleType[data]];
		funTy = WrapIntegerArray[ LLVMLibraryFunction["LLVMFunctionType"][funRes, #, 1, 0]&, {voidHandleArrayTy}];
        funId = 
			LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], name, funTy];
		LLVMAddFunctionParameterAttribute[data, funId, "noalias", 1];
        externalizeFunction[data, funId];
        AddFunctionSystemMetadata[data, funId];
		funId
	]	
	
(*
 CompleteInitialization is called to fill out the initialization function, when 
 all the requirements are known.
*)
CompleteInitialization[data_?AssociationQ, funId_] :=
    Module[ {argId, bbId, initDoneId, id1},
    	argId = LLVMLibraryFunction["LLVMGetParam"][funId, 0];
    	data["initializationArgumentId"]["set", argId];
 		data["functionId"]["set", funId];
        bbId = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], funId, data["getBBName"]["main"]];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbId];
		initDoneId = GetExistingGlobal[ data, "initializationDone"];
        AddIfTrueBasicBlock[data, initDoneId, 
        	Function[ {dataArg, bbAfter},
        		Module[ {id},
        			id = AddConstantInteger[dataArg, 32,1];
        			LLVMLibraryFunction["LLVMBuildRet"][dataArg["builderId"], id];
        		]]];
        (*
          Initialize the watchKernelHandle with the value from the Kernel.
          Also set initializationDone to True.
        *)
        If[Lookup[data, "abortHandling", Automatic] =!= False,
	        id1 = AddRuntimeFunctionCall[ data, "getAbortWatchHandle", {}];
			SetExistingGlobal[ data, "abortWatchHandle", id1]
        ];
		SetExistingGlobal[ data, "initializationDone", AddConstantBoolean[data, True]];
        ProcessConstants[data];
        id1 = AddConstantInteger[data, 32,0];
        LLVMLibraryFunction["LLVMBuildRet"][data["builderId"], id1];
        funId
    ]




End[]

EndPackage[]

