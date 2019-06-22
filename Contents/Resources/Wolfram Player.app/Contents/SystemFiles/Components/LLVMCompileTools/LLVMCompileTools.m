BeginPackage["LLVMCompileTools`"]

InitializeSystem

CreateModule::usage = "CreateModule  "
CreateModule::notriple = "Invalid target triple '`1`' (defaulted to not setting the module triple)."
CreateModule::nodl = "Invalid data layout '`1`' (defaulted to not setting the module data layout)."

FinalizeModule::usage = "FinalizeModule  "

GetRuntimeFunction

GetExternalLibraryFunction

GetLLVMModuleFunction

LLVMCodeTools

CreateExternalFunction
InitializeCodeGeneration
VerifyModule

DisposeOrcData

DisposeBuilders

$CompileErrorString

AddCodeFunction

FinalizeModule

ExternalAddresses

ExternalNameIndices

$RuntimeLibraryDirectory
$RuntimeBitcodeDirectory

$LLVMCompilerResources

GetLLVMContext

ShutdownLLVM

FindWolframRTLStub

SetAllowedExternalNames

Begin["`Private`"]



(*
  TODO,  add an initialization mechanism to pick these up
*)
If[!AssociationQ[$codeFunctions],
    $codeFunctions = <||>
]

AddCodeFunction[ name_, func_] :=
	$codeFunctions[name] = func

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["LLVMLink`"]
Needs["LLVMLink`LLVMInformation`"]
Needs["LLVMCompileTools`ExprFunctions`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`Globals`"]
Needs["LLVMCompileTools`MTensor`"]
Needs["LLVMCompileTools`NumericArray`"]
Needs["LLVMCompileTools`Debugging`"]
Needs["LLVMCompileTools`BitOperations`"]
Needs["LLVMCompileTools`Complex`"]
Needs["LLVMTools`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Platform`Platform`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Intrinsics`"]
Needs["LLVMCompileTools`Unchecked`"]
Needs["LLVMCompileTools`ReinterpretCast`"]
Needs["LLVMCompileTools`String`"]



LLVMCodeToolsDir = FileNameDrop[ $InputFileName, -1]

If[ !ValueQ[$RuntimeLibraryDirectory],
		$RuntimeLibraryDirectory = FileNameJoin[ {$InstallationDirectory, "SystemFiles", "Libraries", $SystemID}]]

If[ !ValueQ[$RuntimeBitcodeDirectory],
		$RuntimeBitcodeDirectory = FileNameJoin[ {$InstallationDirectory, "SystemFiles", "Libraries", $SystemID}]]

$LLVMCompilerResources = FileNameJoin[ {LLVMCodeToolsDir, "LLVMResources"}]

isDebugVersion[] := StringContainsQ[$Version, "Debugging"]



Options[ CreateModule] = {
	"LLVMPasses" -> None,
	"LazyJIT" -> True,
    "AddRuntime" -> "SharedLibrary",
    "DataLayout" -> Automatic,
    "TargetArchitecture" -> Automatic,
    "TargetTriple" -> Automatic,
    "TargetSystemID" -> Automatic
}

Options[ InitializeSystem] = {
    "AddRuntime" -> "SharedLibrary",
    "DataLayout" -> Automatic,
    "TargetTriple" -> Automatic,
    "TargetSystemID" -> Automatic
}

(*
 Maybe this should return LLVMContext[ $LLVMContext].  If so we'd have 
 to think through the chain of imports.  Maybe this whole functionality 
 should be moved into LLVMTools.
*)
GetLLVMContext[] :=
	Module[ {},
		If[!ValueQ[ $LLVMContext],
			Module[{contextId, diagHandlerRef},
				contextId = LLVMLibraryFunction["LLVMContextCreate"][];
				diagHandlerRef = LLVMLibraryFunction["getDiagnosticHandlerAddress"][];
				LLVMLibraryFunction["LLVMContextSetDiagnosticHandler"][contextId, diagHandlerRef, 0];
				$LLVMContext = LLVMContext[contextId];
			];
		];
		$LLVMContext
	]


InitializeSystem[opts:OptionsPattern[]] :=
	InitializeSystem[<| opts |>]
	
InitializeSystem[opts_?AssociationQ] :=
	Module[ {context, runtimeData},
		initializeStatic[];
		context = $LLVMContext;
		runtimeData = initializeRuntime[context, opts];
	]

(*
  Create data for an LLVM Module.  Note the use of CreateReference, which assumes that 
  this has been loaded.
*)

CreateModule[name_, opts:OptionsPattern[]] :=
	CreateModule[name, <| opts |>]
CreateModule[name_, opts_?AssociationQ] :=
	Module[ {contextId, moduleId, builderId, diBuilderId = None, externalData, data, runtimeData, addRuntime, 
		  	 expressionInterface, lazyJIT, llvmPasses, targetTriple, targetSystemID, externalNameIndices, 
		  	 dataLayout, systemID, machineIntegerSize, exceptionModel, abortHandling, llvmOptLevel},
		initializeStatic[];
		externalNameIndices = ExternalNameIndices[];
		externalData = Association @@ (Rule @@@ Compile`GetFunctionAddresses[]);
		contextId = First[$LLVMContext];
		moduleId = Module[{modId, triple},
			modId = LLVMLibraryFunction["LLVMModuleCreateWithNameInContext"][name, contextId];
			If[$LLVMInformation["LLVM_VERSION"] >= 7.0,
				If[KeyExistsQ[opts, "sourceFilePath"],
					LLVMLibraryFunction["LLVMSetSourceFileName"][modId, opts["sourceFilePath"], StringLength[opts["sourceFilePath"]]]
				]
			]; 
			systemID = Lookup[opts, "SystemID", $SystemID];
			triple = ResolveLLVMTargetTriple[Lookup[opts, "TargetTriple", Automatic], Lookup[opts, "TargetSystemID", Automatic]];
			If[!FailureQ[triple],
				LLVMLibraryFunction["LLVMSetTarget"][modId, triple],
				Message[CreateModule::notriple, systemID]
			];
			dataLayout = ResolveLLVMDataLayout[Lookup[opts, "DataLayout", Automatic], Lookup[opts, "TargetSystemID", Automatic]];
            If[!FailureQ[dataLayout],
                LLVMLibraryFunction["LLVMSetDataLayout"][modId, dataLayout],
                Message[CreateModule::nodl, systemID]
            ];
			modId
		];
		
		builderId = LLVMLibraryFunction["LLVMCreateBuilderInContext"][contextId];
		If[TrueQ[Lookup[opts, "LLVMDebug", False]],
			diBuilderId = LLVMLibraryFunction["LLVMWLDIBuilderCreate"][moduleId];
		];
		runtimeData = initializeRuntime[$LLVMContext, opts];
		addRuntime = Lookup[opts, "AddRuntime", "SharedLibrary"];
		lazyJIT = Lookup[opts, "LazyJIT", True];
		llvmPasses = Lookup[opts, "LLVMPasses", None];
		targetTriple = Lookup[opts, "TargetTriple", Automatic];
		targetSystemID = Lookup[opts, "TargetSystemID", Automatic];
		exceptionModel = Lookup[opts, "ExceptionsModel", Automatic];
        abortHandling = Lookup[opts, "AbortHandling", Automatic];
        llvmOptLevel = Lookup[opts, "LLVMOptimization", 0];
		If[targetSystemID === Automatic,
				machineIntegerSize = $SystemWordLength,
				machineIntegerSize = MachineIntegerSizeFromSystemID[targetSystemID]
		];
		expressionInterface = Lookup[opts, "ExpressionInterface", Automatic];
		(*
		 Should determine if this is a C++ build.
		*)
		If[expressionInterface === Automatic,
				expressionInterface = "Inlined"
		];
		data = <|
			"contextId" -> contextId,
			"moduleId" -> moduleId,
			"builderId" -> builderId,
			"diBuilderId" -> diBuilderId,
			"isDebug" -> isDebugVersion[],
			"globalFunctionTypes" -> getGlobalFunctionTypes[],
			"getBBName" -> Function[{base}, getBBName[base]],
			"LLVMIntPredicate" -> Function[ {pred}, getLLVMIntPredicates[pred]],
			"externalNameIndices" -> externalNameIndices,
			"externalData" -> externalData,
			"LLVMModules" -> CreateReference[<||>],
			"globalData" -> CreateReference[<||>],
			"constantData" -> CreateReference[<||>],
			"constantDataList" -> CreateReference[{}],
			"functionCache" -> CreateReference[<||>],
			"orcData" -> CreateReference[],
			"codeFunctions" -> initializeCodeFunctions[],
			"runtimeData" -> runtimeData,
			"addRuntime" -> addRuntime,
			"lazyJIT" -> lazyJIT,
			"expressionInterface" -> expressionInterface,
			"llvmPasses" -> llvmPasses,
			"typeResolutionModule" -> If[ addRuntime === "Clone", moduleId, runtimeData["moduleId"]],
			"expressionVersion" -> Compile`ExpressionVersion[],
			"targetTriple" -> targetTriple,
			"targetSystemID" -> targetSystemID,
			"dataLayout" -> dataLayout,
			"machineIntegerSize" -> machineIntegerSize,
			"exceptionsModel" -> exceptionModel,
			"abortHandling" -> abortHandling,
			"llvmOptimization" -> llvmOptLevel,
			"exceptionsInitialized" -> CreateReference[False]
		|>;
		data = Which[
			addRuntime === "Clone",
				cloneRuntime[data],
			addRuntime === "Link",
				linkRuntime[data],
			True,
				data
		];
		(*
		  Add globals needed by compiled code,  so far just the watchKernelAddress
		  which is needed for abort handling.
		*)
        If[abortHandling =!= False,
		  With[{ty = GetHandleType[data, GetIntegerType[data, 32]]},
			CreateGlobal[data, "abortWatchHandle", ty, AddNull[data, ty]]
		  ]
        ];
		data
	]

getTargetRef[triple_] :=
    Module[{str, targetRef, res},
        ScopedAllocation["LLVMTargetObjectPointer"][Function[{targetRefPtr},
        ScopedAllocation["CharObjectPointer"][Function[{strRef},
            res = LLVMLibraryFunction["LLVMGetTargetFromTriple"][triple, targetRefPtr, strRef];
            If[res =!= 0,
                str = LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][strRef, 0];
                ThrowException[{"Could not get target from triple", triple, str}]
            ];
            targetRef = LLVMLibraryFunction["LLVMLink_getLLVMTargetObjectPointer"][targetRefPtr, 0];
        ], 1];
        ], 1];
        targetRef
    ]

createTargetMachine[data_, targetRef_, triple_] :=
    Module[{cpu, features, level, reloc, codeModel, tmId},
        cpu = "";
        features = "";
        If[data["targetSystemID"] === Automatic || data["targetSystemID"] === $SystemID,
						cpu = LLVMLibraryFunction["LLVMGetHostCPUName"][];
						features = LLVMLibraryFunction["LLVMGetHostCPUFeatures"][]
				];
        (*
          We optimize with direct calls to the LLVM Pass Manager.
        *)
        level = LLVMEnumeration["LLVMCodeGenOptLevel", "LLVMCodeGenLevelNone"];
        reloc = LLVMEnumeration["LLVMRelocMode", "LLVMRelocDefault"];
        codeModel = LLVMEnumeration["LLVMCodeModel", "LLVMCodeModelJITDefault"];

        tmId = LLVMLibraryFunction["LLVMCreateTargetMachine"][targetRef, triple, cpu, features, level, reloc, codeModel];   
        
        tmId
    ]

createOrcJITStackForModule[data_?AssociationQ, modId_] :=
	Module[ {triple, targetRef, tmId, orcInstance, resolverId, resolveDataId, sharedModRef, orcModuleId},

		triple = ResolveLLVMTargetTriple[data["targetTriple"], data["targetSystemID"]];
        targetRef = getTargetRef[triple];

		tmId = createTargetMachine[data, targetRef, triple];
		
		orcInstance = LLVMLibraryFunction["LLVMOrcCreateInstance"][ tmId];
		resolverId = LLVMLibraryFunction["getSymbolResolverAddress"][];
		resolveDataId = 0;

		If[ TrueQ[data["lazyJIT"]],
			If[$LLVMInformation["LLVM_VERSION"] >= 7.0,
				ScopedAllocation["Uint64_tObject"][Function[{orcModuleIdRef},
					LLVMLibraryFunction["LLVMOrcAddLazilyCompiledIR"][orcInstance, orcModuleIdRef, modId, resolverId, resolveDataId];
					orcModuleId = LLVMLibraryFunction["LLVMLink_getUint64_tObject"][orcModuleIdRef, 0];
				], 1];
				,
				(*else*)
				ScopedAllocation["Uint32_tObject"][Function[{orcModuleIdRef},
					sharedModRef = LLVMLibraryFunction["LLVMOrcMakeSharedModule"][modId];
					LLVMLibraryFunction["LLVMOrcAddLazilyCompiledIR"][orcInstance, orcModuleIdRef, sharedModRef, resolverId, resolveDataId];
					LLVMLibraryFunction["LLVMOrcDisposeSharedModuleRef"][sharedModRef];
					orcModuleId = LLVMLibraryFunction["LLVMLink_getUint32_tObject"][orcModuleIdRef, 0];
				], 1];
			]
			,
			If[$LLVMInformation["LLVM_VERSION"] >= 7.0,
				ScopedAllocation["Uint64_tObject"][Function[{orcModuleIdRef},
					LLVMLibraryFunction["LLVMOrcAddEagerlyCompiledIR"][orcInstance, orcModuleIdRef, modId, resolverId, resolveDataId];
					orcModuleId = LLVMLibraryFunction["LLVMLink_getUint64_tObject"][orcModuleIdRef, 0];
				], 1];
				,
				(*else*)
				ScopedAllocation["Uint32_tObject"][Function[{orcModuleIdRef},
					sharedModRef = LLVMLibraryFunction["LLVMOrcMakeSharedModule"][modId];
					LLVMLibraryFunction["LLVMOrcAddEagerlyCompiledIR"][orcInstance, orcModuleIdRef, sharedModRef, resolverId, resolveDataId];
					LLVMLibraryFunction["LLVMOrcDisposeSharedModuleRef"][sharedModRef];
					orcModuleId = LLVMLibraryFunction["LLVMLink_getUint32_tObject"][orcModuleIdRef, 0];
				], 1];
			]
		];

		<|"orcInstance" -> orcInstance, "orcModuleId" -> orcModuleId, "targetMachineId" -> tmId |>
	]

intPreds = <|
			SameQ -> LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"],
			Greater -> LLVMEnumeration["LLVMIntPredicate", "LLVMIntSGT"],
			GreaterEqual -> LLVMEnumeration["LLVMIntPredicate", "LLVMIntSGE"],
			Less -> LLVMEnumeration["LLVMIntPredicate", "LLVMIntSLT"],
			LessEqual -> LLVMEnumeration["LLVMIntPredicate", "LLVMIntSLE"]
			|>

getLLVMIntPredicates[pred_] :=
	Module[ {ef},
		ef = Lookup[intPreds, pred, Null];
		If[ ef === Null,
			ThrowException[{"Cannot find integer predicate"}]];
		ef
	]
	


cnt = -1;

getBBName[base_] :=
	(
	cnt++;
	base <> "_" <> ToString[cnt]
	)

getGlobalFunctionTypes[] :=
(
	If[!ValueQ[$GlobalFunctionTypes],
		$GlobalFunctionTypes = SetupGlobalFunctionTypes[];
	];
	$GlobalFunctionTypes
)

(*
   initializeStatic is called each time a module is created,  but it only executes the 
   first time.
*)

initializeStatic[] :=
	Module[ {res, passReg},
		res = InstallLLVM[];
		If[FailureQ[res],
			ThrowException[{"Error in InstallLLVM", res}]
		];
		LLVMLibraryFunction["LLVMInitializeAllTargetInfos"][];
		LLVMLibraryFunction["LLVMInitializeAllTargets"][];
		LLVMLibraryFunction["LLVMInitializeAllTargetMCs"][];
		LLVMLibraryFunction["LLVMInitializeAllAsmPrinters"][];
		LLVMLibraryFunction["LLVMInitializeAllAsmParsers"][];
		LLVMLibraryFunction["LLVMInitializeAllDisassemblers"][];
		res = LLVMLibraryFunction["LLVMInitializeNativeTarget"][];
		If[res =!= 0,
			ThrowException[{"Error in LLVMInitializeNativeTarget", res}]
		];
		res = LLVMLibraryFunction["LLVMInitializeNativeAsmPrinter"][];
		If[res =!= 0,
			ThrowException[{"Error in LLVMInitializeNativeAsmPrinter", res}]
		];
		passReg = LLVMLibraryFunction["LLVMGetGlobalPassRegistry"][];
		LLVMLibraryFunction["LLVMInitializeCore"][passReg];
		LLVMLibraryFunction["LLVMInitializeScalarOpts"][passReg];
		LLVMLibraryFunction["LLVMInitializeVectorization"][passReg];
		LLVMLibraryFunction["LLVMInitializeIPO"][passReg];
		LLVMLibraryFunction["LLVMInitializeAnalysis"][passReg];
		LLVMLibraryFunction["LLVMInitializeAssumptionCacheTracker"][passReg];
		LLVMLibraryFunction["LLVMInitializeIPA"][passReg];
		LLVMLibraryFunction["LLVMInitializeTransformUtils"][passReg];
		LLVMLibraryFunction["LLVMInitializeInstCombine"][passReg];
		LLVMLibraryFunction["LLVMInitializeCodeGen"][passReg];
		LLVMLibraryFunction["LLVMInitializeInstrumentation"][passReg];
		LLVMLibraryFunction["LLVMInitializeTarget"][passReg];
		
		GetLLVMContext[];  (* has the side effect of initializing $LLVMContext *)
	]

FindWolframRTLStub[] :=
	getLibraryName[ "WolframRTL", True]

getLibraryName[ name_, stub_] :=
	Module[{ext},
		FileNameJoin[ {$RuntimeLibraryDirectory, 
		
			Switch[$SystemID,
				"MacOSX-x86-64", 
					"lib" <> name <> "." <>"dylib",
				"Linux" | "Linux-ARM" | "Linux-x86-64",
					"lib" <> name <> "." <>"so",
				"Windows" | "Windows-x86-64",
					ext = If[stub, "lib", "dll"];
					name <> "." <> ext,
				_,
					ThrowException[{"Cannot expand name"}]
			]}]
	]


$RuntimeData
$RuntimeOrcData

initializeRuntime[context_, opts:OptionsPattern[]] :=
    initializeRuntime[context,  <| opts |>]
    
initializeRuntime[context_, opts_?AssociationQ] :=
	Module[{runtimeName, runtimeExports, addRuntime, libName},

		addRuntime = Lookup[opts, "AddRuntime", "SharedLibrary"];
		runtimeName = Switch[Lookup[opts, "TargetArchitecture", Automatic],
			"CUDA",
				"runtime_cuda",
			_,
				"WolframRTL"
		];
		runtimeExports = Switch[Lookup[opts, "TargetArchitecture", Automatic],
			"CUDA",
				"runtime_cuda",
			_,
				"WolframRTL_Exports"
		];
		
		If[!AssociationQ[$RuntimeData],
			$RuntimeData = <|
				(* addresses is only used by AddRuntime->Link *)
				"addresses" -> CreateReference[<||>],
				"moduleId" -> loadRuntimeModule[ context, runtimeExports, opts]
			|>;
		];
		If[ addRuntime === "Link",
			If[!ReferenceQ[$RuntimeOrcData],
				Module[{lazyJIT, targetTriple, targetSystemID, data, orcData},
					$RuntimeOrcData = CreateReference[];
					lazyJIT = Lookup[opts, "LazyJIT", True];
					targetTriple = Lookup[opts, "TargetTriple", Automatic];
					targetSystemID = Lookup[opts, "TargetSystemID", Automatic];
					data = <|
						"lazyJIT" -> lazyJIT,
						"targetTriple" -> targetTriple,
						"targetSystemID" -> targetSystemID
					|>;
					orcData = createOrcJITStackForModule[data, $RuntimeData["moduleId"]];
					$RuntimeOrcData["set", orcData]]]];
			
		If[ addRuntime === "SharedLibrary",
			Module[{ret},
				libName = getLibraryName[runtimeName, False];
				ret = LLVMLibraryFunction["LLVMLoadLibraryPermanently"][libName];
				If[ret != 0,
					ThrowException[{"Cannot load library ", libName, ret}]];
			]];

		$RuntimeData
	]

RegisterCallback["ShutdownLLVMCompileTools", Function[{st},
If[ReferenceQ[$RuntimeOrcData],
	If[AssociationQ[$RuntimeOrcData["get"]],
		
		DisposeOrcData[$RuntimeOrcData["get"]];
		Clear[$RuntimeOrcData];
	]
];

(* destroying Runtime OrcJITStack also destroys Runtime Module via LLVM refcounting *)
Clear[$RuntimeData];

Clear[$GlobalFunctionTypes];

disposeLLVMContext[];

]]


disposeLLVMContext[] :=
If[IntegerQ[$LLVMContext],
	LLVMLibraryFunction["LLVMContextDispose"][$LLVMContext];
	Clear[$LLVMContext];
]





(*
  Look for a function in the runtime, used when in AddRuntime -> "Link".
  Check to see if the function has already been processed, if so exit.
  Otherwise,  get the execution and jit engine for the runtime module, 
  find the address of the function and make it available via AddSymbol.
*)
findRuntimeFunction[data_?AssociationQ, "Link", funName_] :=
	Module[ {rtData, orcData, orcInstance, addrRef, addr},
		rtData = data["runtimeData"];
		If[rtData["addresses"]["keyExistsQ", funName],
			Return[]];
		orcData = $RuntimeOrcData["get"];
		orcInstance = orcData["orcInstance"];

		ScopedAllocation["Uint64_tObject"][Function[{addrRef},
			If[$OperatingSystem != "Windows",
				errCode = LLVMLibraryFunction["LLVMOrcGetSymbolAddress"][orcInstance, addrRef, funName];
				,
				errCode = LLVMLibraryFunction["LLVMOrcGetSymbolAddress2"][orcInstance, addrRef, funName, False];
			];
			If[ errCode =!= 0,
				ThrowException[{"Cannot find runtime function ", funName, errCode}]];
			
			addr = LLVMLibraryFunction["LLVMLink_getUint64_tObject"][addrRef, 0];
		], 1];
		
		LLVMLibraryFunction["LLVMAddSymbol"][funName, addr];
		rtData["addresses"]["associateTo", funName -> addr];
	]
	

(*
  Used when in AddRuntime -> "SharedLibrary". It is not necessary for this function to do anything
  because when the runtime is used in SharedLibrary mode, functions in the runtime are found by
  OrcJIT using standard os-level shared library lookup calls (i.e., dlsym()/GetProcAddress()). 
*)
findRuntimeFunction[data_?AssociationQ, "SharedLibrary", funName_] = Null


(*
  Doesn't need to do anything right now.
*)
linkRuntime[ data_] :=
	data


LLVMCodeTools::load = "Cannot load the LLVMCode resources `1`."


loadRuntimeModule[ LLVMContext[contextId_], name_String, opts_?AssociationQ] :=
	Module[ {file, memBuffRef, memBuffId, strRef, modRef, modId, res, triple, dataLayout},
		file = FileNameJoin[ {$RuntimeBitcodeDirectory, name <> ".bc"}];
		If[ !FileExistsQ[file],
			Message[ LLVMCodeTools::load,  file];
			ThrowException[{"Cannot load LLVMCode resources ", file}]];
		
		(* Create MemoryBuffer and String references *)
		ScopedAllocation["LLVMOpaqueMemoryBufferObjectPointer"][Function[{memBuffRef},
		ScopedAllocation["CharObjectPointer"][Function[{strRef},
			(* Load file into memory buffer *)
			If[ LLVMLibraryFunction["LLVMCreateMemoryBufferWithContentsOfFile"][ file, memBuffRef, strRef] =!= 0,
				str = LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][strRef, 0];
				ThrowException[{"Cannot create memory buffer for file", name,  strRef, str}]];
			(* get the MemoryBuffer,  deallocate the reference, create a Module reference *)
			memBuffId = LLVMLibraryFunction["LLVMLink_getLLVMOpaqueMemoryBufferObjectPointer"][memBuffRef, 0];
		], 1];
		], 1];

		ScopedAllocation["LLVMOpaqueModuleObjectPointer"][Function[{modRef},
		ScopedAllocation["CharObjectPointer"][Function[{strRef},
			(* parse the memory buffer into a module *)
			If[ LLVMLibraryFunction["LLVMParseIRInContext"][contextId, memBuffId, modRef, strRef] =!= 0,
				str = LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][strRef, 0];
				ThrowException[{"Cannot parse IR from file", name, str}]];
			modId = LLVMLibraryFunction["LLVMLink_getLLVMOpaqueModuleObjectPointer"][modRef, 0];
		], 1];
		], 1];

		triple = ResolveLLVMTargetTriple[Lookup[opts, "TargetTriple", Automatic], Lookup[opts, "TargetSystemID", Automatic]];
		LLVMLibraryFunction["LLVMSetTarget"][modId, triple];
		
        dataLayout = ResolveLLVMDataLayout[Lookup[opts, "DataLayout", Automatic], Lookup[opts, "TargetSystemID", Automatic]];
        LLVMLibraryFunction["LLVMSetDataLayout"][modId, dataLayout];

		(* Verify the module *)
		res = LLVMVerifyModule[LLVMModule[modId]];
		If[!TrueQ[res["valid"]],
			ThrowException[{"Cannot verify module", name, res["errorString"]}]];
		modId
	]
	


(*
  cloneRuntime is called each time compilation is done if AddRuntime is True.  
  Its purpose is to make the runtime functions available.  It does this 
  by cloning the runtime and adding that to the data module.  It might be 
  better to do this the other way round,  but before any of that it would be
  good to avoid this copying of modules.  It seems necessary to do this for 
  linking,  an alternative would be to get a function pointer and work with that.
*)

cloneRuntime[data_?AssociationQ] :=
	cloneModuleInto[data, data["runtimeData"]["moduleId"]]

cloneModuleInto[ data_?AssociationQ, sourceModuleId_] :=
	Module[{newId},
		newId = LLVMLibraryFunction["LLVMCloneModule"][sourceModuleId];
		If[ LLVMLibraryFunction["LLVMLinkModules2"][data["moduleId"], newId] =!= 0,
			ThrowException[{"Cannot link modules"}]];
		data
	]


(*
 Hookup a call to a runtime function.  
 If we cloned the runtime we just look up the function. 
 If we are linking to the runtime we have to make sure the function address 
 from the runtime module has been added as a global symbol, then we add the 
 function and type as a global.
*)
GetRuntimeFunction[ data_?AssociationQ, funName_] :=
	Module[ {funId, modId, tyId},
		funId = data["functionCache"]["lookup", funName, Null];
		If[ funId === Null,
			Which[ 
				data["addRuntime"] === "Clone",
					modId = data["moduleId"];
					funId = LLVMLibraryFunction["LLVMGetNamedFunction"][modId, funName];
					If[ !(IntegerQ[funId] && funId > 0),
						ThrowException[{"Cannot load function from runtime ", funName}]];
				,
				
				data["addRuntime"] === "Link" || data["addRuntime"] === "SharedLibrary",					
					findRuntimeFunction[data, data["addRuntime"], funName];
					(*
					  Lookup the function in the Runtime Module, this is purely done
					  to get the type.
					*)
					modId = data["runtimeData"]["moduleId"];
					funId = LLVMLibraryFunction["LLVMGetNamedFunction"][modId, funName];
					If[ !(IntegerQ[funId] && funId > 0),
						ThrowException[{"Cannot load function from runtime ", funName}]];
					tyId = LLVMLibraryFunction["LLVMTypeOf"][funId];
					funId = GetLinkedFunctionFromName[data, funName, tyId];
				,
				True,
				 ThrowException[{"Unknown setting for AddRuntime ", data["addRuntime"]}]
				];
			data["functionCache"]["associateTo", funName -> funId]];
		funId 
	]

GetExternalLibraryFunction[ data_?AssociationQ, funName_] :=
	Module[ {funId, tyId},
		funId = data["functionCache"]["lookup", funName, Null];
		If[ funId === Null,
			tyId = data["externalLibraryFunctionTypes"]["lookup", funName, Null];
			If[ tyId === Null,
				ThrowException[{"Cannot find a type definition for external library function", funName}]
			];
			funId = GetLinkedFunctionFromName[data, funName, tyId];
			data["functionCache"]["associateTo", funName -> funId]
		];
		funId 
	]
	
(*
  If the module has not been added then add it.
  Then lookup function name.
*)
GetLLVMModuleFunction[data_, LLVMModule[id_], funName_] :=
	Module[ {funId},
		funId = data["functionCache"]["lookup", funName, Null];
		If[ funId === Null,
			If[ !data["LLVMModules"]["keyExistsQ", id],
				cloneModuleInto[ data, id];
				data["LLVMModules"]["associateTo", id -> True]];
			funId = LLVMLibraryFunction["LLVMGetNamedFunction"][data["moduleId"], funName];
			If[ funId === Null,
				ThrowException[{"Cannot find a function definition for an LLVMModule function", funName}]];
			data["functionCache"]["associateTo", funName -> funId]];
		funId 

	]

initializeCodeFunctions[] :=
	Module[ {},
		Join[
			$codeFunctions,
		<|
		"Expr`NormalLength" -> AddExprLength,
		"not_Boolean" -> AddNotBoolean
		|>
		]
	]



CodeFunction::verification  = "Cannot verify the program module.\n`ErrorString`"
CodeFunction::nofunction  = "Cannot find address for function `FunctionName`. "
CodeFunction::errcode  = "Error while getting symbol address for function `FunctionName`. \
MainAddressError: `MainAddressError`. InitializationAddressError: `InitializationAddressError`. "

VerifyModule[ data_] :=
	Module[{modId, res},
		modId = data["moduleId"];
		res = LLVMVerifyModule[LLVMModule[modId]];
		If[!TrueQ[res["valid"]],
			$CompileErrorString = res["errorString"];
			ThrowException[ 
				<|"Exception" -> "VerificationError", "MessageParameters" -> <|"ErrorString" -> res["errorString"]|>,
					"ModuleID" -> modId, "MessageTemplate" -> CodeFunction::verification|>]];
	]

InitializeCodeGeneration[ data_] :=
	Module[ {modId, orcData},
		modId = data["moduleId"];
		orcData = createOrcJITStackForModule[ data, modId];
		data["orcData"]["set", orcData];
	]


CreateExternalFunction[ data_?AssociationQ, {name_, initName_, rawName_}, funData_] :=
	Module[ {addr, initAddr,  utilityAddr, errCode, initErrCode, utilityErrCode,
			rawAddr, rawErrCode, orcInstance, orcData, extLibs,
			utilityName = "initializeRTL",signature = Lookup[funData, "Signature", Null]},
		If[ !MatchQ[signature, TypeSpecifier[_]],
			ThrowException["Cannot find function signature "]];	
		orcData = data["orcData"]["get"];			
		orcInstance = orcData["orcInstance"];			
		{addr, errCode} = getAddress[ orcInstance, name];
		{initAddr, initErrCode} = getAddress[ orcInstance, initName];
		{utilityAddr, utilityErrCode} = getAddress[ orcInstance, utilityName];
		{rawAddr, rawErrCode} = getAddress[ orcInstance, rawName];
		extLibs = data["externalLibraries"];
		If[ MatchQ[extLibs, {__String}],
			Map[Module[{ret},
				
				If[TrueQ[Compile`$CompilerSandbox] && Developer`CheckFileAccess["Execute" -> #] =!= True,
					ThrowException[{"Cannot load library ", #}]];
	
				ret = LLVMLibraryFunction["LLVMLoadLibraryPermanently"][#];
				If[ret != 0,
					ThrowException[{"Cannot load library ", #, ret}]];
				]&, extLibs]];
		Which[
			addr == 0 || initAddr == 0 || utilityAddr == 0 || rawAddr == 0,
			Failure["FunctionAddress",
					<|"MessageParameters" -> <|
    					"FunctionName" -> Which[
					       addr === 0, name,
                           initAddr === 0, initName,
                           utilityAddr === 0, utilityName,
                           rawAddr === 0, rawName,
                           True, "Unknown"
					   ]
					|>, "MessageTemplate" -> CodeFunction::nofunction|>]
			,
			errCode =!= 0 || initErrCode =!= 0 || utilityErrCode =!= 0,
			ThrowException[ 
				<|"Exception" -> "FunctionAddress", 
					"MessageParameters" -> <|"FunctionName" -> name|>,
					"MessageTemplate" -> "MessageTemplate" -> CodeFunction::errcode,
					"MainAddressError" -> errCode, "InitializationAddressError" -> initErrCode, "UtilityAddressError" -> utilityErrCode
					|>];
			,
			True,
			callUtility[ data, utilityAddr];
			Compile`CreateCompiledCodeFunction[
				{Join[funData, orcData], addr, initAddr, rawAddr, ToString[First[signature], InputForm]}]
		]
	]

getAddress[orcInstance_, name_] :=
	Module[ {addr, errCode},
		ScopedAllocation["Uint64_tObject"][Function[{addrRef},
			If[$OperatingSystem != "Windows",
				errCode = LLVMLibraryFunction["LLVMOrcGetSymbolAddress"][orcInstance, addrRef, name];
				,
				errCode = LLVMLibraryFunction["LLVMOrcGetSymbolAddress2"][orcInstance, addrRef, name, False];
			];
			addr = LLVMLibraryFunction["LLVMLink_getUint64_tObject"][addrRef, 0];
		], 1];
		{addr, errCode}
	]




(*
  This initializes the WolframRTL with the addresses of various Kernel parameters.
  It uses the address of a function that was created in the by CreateWrapper
  When the Kernel uses just one instance of the RTL (by dynamic linking) 
  this can be simplified.
*)
getAddressFromKernel[ addrList_, name_] :=
	Module[{addr},
		addr = SelectFirst[ addrList, MatchQ[#, {name, _}]&, 0];
		If[addr === 0,
			ThrowException[{"Cannot find " <> name}]];
		Last[addr]
	]


callUtility[ data_, utilityFunAddr_] :=
	Module[ {addrList = Compile`GetFunctionAddresses[], allocListAddr, watchAddr, watchProcess},
		allocListAddr = getAddressFromKernel[addrList, "AllocatorList"];
		watchAddr = getAddressFromKernel[addrList, "WatchAddress"];
		watchProcess = getAddressFromKernel[addrList, "WatchProcess"];
		Compile`InitializeCompileUtilities[utilityFunAddr, allocListAddr, watchAddr, watchProcess];
	]


DisposeOrcData[data_?AssociationQ] :=
	Module[{orcInstance, orcModuleId, targetMachineId},

		If[KeyExistsQ[data, "targetMachineId"],
			targetMachineId = data["targetMachineId"];
			LLVMLibraryFunction["LLVMDisposeTargetMachine"][targetMachineId];
			,
			ThrowException[{"Key \"targetMachineId\" was not found"}, data]
		];

		If[KeyExistsQ[data, "orcInstance"],
			orcInstance = data["orcInstance"];

			If[KeyExistsQ[data, "orcModuleId"],
				orcModuleId = data["orcModuleId"];
				LLVMLibraryFunction["LLVMOrcRemoveModule"][orcInstance, orcModuleId];
				,
				ThrowException[{"Key \"orcModuleId\" was not found"}, data]
			];

			LLVMLibraryFunction["LLVMOrcDisposeInstance"][orcInstance];
			,
			ThrowException[{"Key \"orcInstance\" was not found"}, data]
		]
	]

DisposeOrcData[args___] :=
	ThrowException[{"Unrecognized call to DisposeOrcData", {args}}]



DisposeBuilders[data_?AssociationQ] :=
	Module[{diBuilderId, builderId},
		If[KeyExistsQ[data, "diBuilderId"],
			diBuilderId = data["diBuilderId"];
			If[diBuilderId =!= None,
				LLVMLibraryFunction["LLVMWLDIBuilderDispose"][diBuilderId]
			]
		];
		If[KeyExistsQ[data, "builderId"],
			builderId = data["builderId"];
			LLVMLibraryFunction["LLVMDisposeBuilder"][builderId];
			,
			ThrowException[{"Key \"builderId\" was not found", data}]
		];
	]

DisposeBuilders[args___] :=
	ThrowException[{"Unrecognized call to DisposeBuilders", {args}}]


(*
  Does nothing.  Called from LLVM pass and also from CreateWrapper, so 
  could be called twice.  Probably this should be cleaned up.
*)
FinalizeModule[data_?AssociationQ] :=
	Null


ShutdownLLVM[] :=
	LLVMLibraryFunction["LLVMShutdown"][]


getExternalData[] :=
	Module[ {data},
		data = Compile`GetFunctionAddresses[];
		{AssociationThread[Part[data, All, 1] -> Range[ Length[data]]-1], Part[data, All, 2]}
	]


ExternalNameIndices[] :=
	First[getExternalData[]]
	
ExternalAddresses[] :=
	Last[getExternalData[]]


SetAllowedExternalNames[ names:{__String}] :=
	Module[{fixNames},
		fixNames = If[ $OperatingSystem === "MacOSX",  Map[ "_" <> #&, names], names];
		LLVMLibraryFunction["addAllowedExternalNames"][ fixNames]
	]



End[]



EndPackage[]

