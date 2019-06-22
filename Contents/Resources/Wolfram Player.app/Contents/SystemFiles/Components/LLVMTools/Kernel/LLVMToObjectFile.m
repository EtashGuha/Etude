BeginPackage["LLVMTools`LLVMToObjectFile`"]


(* TODO: It would be nice to at least create a summary box representation of LLVMModule, but I don't
         see anything in LLVMLink, or the LLVM C API, to get anything useful out of a ModuleRef. *)


Begin["`Private`"]

Needs["LLVMTools`"]
Needs["LLVMLink`"]
Needs["LLVMTools`LLVMComponentUtilities`"] (* For LLVMTripleFromSystemID *)
Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileUtilities`Callback`"]



General::llvmerr = "Error returned by LLVM function `1`: `2`"  (* Note no period at end, as the
    period is typically supplied by LLVM *)

General::cont = "`1` is not a recognised form for the LLVM context.  These are specified as LLVMContext[id] or Automatic."



(*
$LLVMToModuleContext is the LLVM Context that is used by LLVMToModule[]
*)

$LLVMToModuleContext



processContext[ h_, LLVMContext[contextID_Integer]] :=
	contextID

(*
  return the global context
*)
processContext[ h_, Automatic] :=
	Module[ {diagHandlerRef},
        (* lazy creation of $LLVMToModuleContext *)
        If[!IntegerQ[$LLVMToModuleContext],
            $LLVMToModuleContext = LLVMLibraryFunction["LLVMContextCreate"][];
            diagHandlerRef = LLVMLibraryFunction["getDiagnosticHandlerAddress"][];
            LLVMLibraryFunction["LLVMContextSetDiagnosticHandler"][$LLVMToModuleContext, diagHandlerRef, 0];
        ];
		$LLVMToModuleContext
	]
	
processContext[ h_, x_] :=
	Module[ {},
		Message[h::cont, x];
		processContext[ Automatic]	
	]	
	


(**********************************  LLVMToModule  **************************************)

Options[LLVMToModule] = {
	"LLVMContext" -> Automatic
};


(* TODO: Should I call VerifyModule or not? *)

LLVMToModule[input:(_String | _File), opts:OptionsPattern[]] :=
    Module[{inputFile, moduleID, moduleRef, memBufRef, memBufID, errorStrRef, res, valid, contextID},

		contextID = processContext[ LLVMToModule, OptionValue["LLVMContext"]];

        If[StringQ[input],
            inputFile = CreateFile[];
            WriteString[inputFile, input];
            Close[inputFile],
        (* else *)
            (* input argument was given as a File expression. *)
            inputFile = First[input];
            (* Because we will be passing the file path to external code that knows nothing about Mathematica's current directory,
               convert file to a full pathname if it was specified without any path elements.
            *)
            If[FileNameDepth[inputFile] === 1,
                inputFile = FileNameJoin[{Directory[], inputFile}]
            ]
        ];

		If[TrueQ[Compile`$CompilerSandbox] && Developer`CheckFileAccess["Execute" -> inputFile] =!= True,
			Message[LLVMToModule::compsand];
			Return[$Failed]];

        (*****  Create memory buffer holding the IR.  *****)
        ScopedAllocation["LLVMOpaqueMemoryBufferObjectPointer"][Function[{memBufRef},
        ScopedAllocation["CharObjectPointer"][Function[{errorStrRef},
            res = LLVMLibraryFunction["LLVMCreateMemoryBufferWithContentsOfFile"][inputFile, memBufRef, errorStrRef];
            If[res =!= 0,
                Message[LLVMToModule::llvmerr, "LLVMCreateMemoryBufferWithContentsOfFile",
                    LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][errorStrRef, 0]];
                Return[$Failed]
            ];
            memBufID = LLVMLibraryFunction["LLVMLink_getLLVMOpaqueMemoryBufferObjectPointer"][memBufRef, 0];

            (*****  Create a module by parsing the IR.  *****)
            ScopedAllocation["LLVMOpaqueModuleObjectPointer"][Function[{moduleRef},
                res = LLVMLibraryFunction["LLVMParseIRInContext"][contextID, memBufID, moduleRef, errorStrRef];
                If[res =!= 0,
                    Message[LLVMToModule::llvmerr, "LLVMParseIRInContext",
                        LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][errorStrRef, 0]];
                    Return[$Failed]
                ];
                moduleID = LLVMLibraryFunction["LLVMLink_getLLVMOpaqueModuleObjectPointer"][moduleRef, 0];
            ], 1];
        ], 1];
        ], 1];

        (*****  Verify the module  *****)
        res = LLVMVerifyModule[LLVMModule[moduleID]];
        valid = res["valid"];
        If[!TrueQ[valid],
            Message[LLVMToModule::llvmerr, "LLVMVerifyModule", res["errorString"]];
            Return[$Failed]
        ];

        LLVMModule[moduleID]
    ]

LLVMToModule[args___] :=
    ThrowException[{"Unrecognized call to LLVMToModule", {args}}]


RegisterCallback["ShutdownLLVMToObjectFile", Function[{st},
disposeLLVMToModuleContext[]
]]

disposeLLVMToModuleContext[] :=
If[IntegerQ[$LLVMToModuleContext],
    LLVMLibraryFunction["LLVMContextDispose"][$LLVMToModuleContext];
    Clear[$LLVMToModuleContext];
]


(********************************  LLVMToMachineCodeFile  ************************************)

$options = {
	"TargetSystemID" -> Automatic,
	"TargetTriple" -> Automatic,
	"TargetCPU" -> Automatic,
	"TargetFeatures" -> Automatic,
	"OutputType" -> "ObjectCode",
	"CommandLineOptions" -> {},
	"OptimizationLevel" -> 2
};

Options[LLVMToObjectFile] = $options;
Options[LLVMToMachineCodeFile] = $options;
Options[LLVMToMachineCodeString] = $options;

(*Options[driver] = $options;*)

LLVMToMachineCodeString::sysid = LLVMToObjectFile::sysid = LLVMToMachineCodeFile::sysid = "Invalid value for TargetSystemID option: `1`. Must be a valid Wolfram $SystemID."
LLVMToMachineCodeString::outfile = LLVMToObjectFile::outfile = LLVMToMachineCodeFile::outfile = "An output file name must be specified when the input is a string of code."



LLVMToMachineCodeString[ input:(_String | _File), opts:OptionsPattern[]] :=
	Module[ {outputFile, contextID, diagHandlerRef, text},
		outputFile = CreateFile[];
        contextID = LLVMLibraryFunction["LLVMContextCreate"][];
        diagHandlerRef = LLVMLibraryFunction["getDiagnosticHandlerAddress"][];
        LLVMLibraryFunction["LLVMContextSetDiagnosticHandler"][contextID, diagHandlerRef, 0];

		outputFile = driver[ LLVMToMachineCodeString, input, outputFile, LLVMContext[contextID], opts];
		text = Import[ outputFile, "Text"];
        LLVMLibraryFunction["LLVMContextDispose"][contextID];
		DeleteFile[outputFile];
		text
	]

LLVMToMachineCodeString[ input:LLVMModule[moduleID_Integer], opts:OptionsPattern[]] :=
    Module[ {outputFile, text, contextID},
        contextID = LLVMLibraryFunction["LLVMGetModuleContext"][moduleID];
        outputFile = CreateFile[];
        outputFile = driver[ LLVMToMachineCodeString, input, outputFile, LLVMContext[contextID], opts];
        text = Import[ outputFile, "Text"];
        DeleteFile[outputFile];
        text
    ]

LLVMToMachineCodeString[args___] :=
    ThrowException[{"Unrecognized call to LLVMToMachineCodeString", {args}}]



LLVMToMachineCodeFile[input:(_String | _File),
                      outputFile:(_String | _File | Automatic):Automatic, opts:OptionsPattern[]] :=
    Module[{contextID, diagHandlerRef, ret},
        contextID = LLVMLibraryFunction["LLVMContextCreate"][];
        diagHandlerRef = LLVMLibraryFunction["getDiagnosticHandlerAddress"][];
        LLVMLibraryFunction["LLVMContextSetDiagnosticHandler"][contextID, diagHandlerRef, 0];

    	ret = driver[ LLVMToMachineCodeFile, input, outputFile, LLVMContext[contextID], opts];
        LLVMLibraryFunction["LLVMContextDispose"][contextID];
        ret
    ]

LLVMToMachineCodeFile[input:LLVMModule[moduleID_Integer],
                      outputFile:(_String | _File | Automatic):Automatic, opts:OptionsPattern[]] :=
    Module[{contextID},
        contextID = LLVMLibraryFunction["LLVMGetModuleContext"][moduleID];
        driver[ LLVMToMachineCodeFile, input, outputFile, LLVMContext[contextID], opts]
    ]

LLVMToMachineCodeFile[args___] :=
    ThrowException[{"Unrecognized call to LLVMToMachineCodeFile", {args}}]



LLVMToObjectFile[input:(_String | _File),
                 outputFile:(_String | _File | Automatic):Automatic, opts:OptionsPattern[]] :=
    Module[{ret, contextID, diagHandlerRef},
        contextID = LLVMLibraryFunction["LLVMContextCreate"][];
        diagHandlerRef = LLVMLibraryFunction["getDiagnosticHandlerAddress"][];
        LLVMLibraryFunction["LLVMContextSetDiagnosticHandler"][contextID, diagHandlerRef, 0];

    	ret = driver[ LLVMToObjectFile, input, outputFile, LLVMContext[contextID], opts];
        LLVMLibraryFunction["LLVMContextDispose"][contextID];
        ret
    ]

LLVMToObjectFile[input:LLVMModule[moduleID_Integer],
                 outputFile:(_String | _File | Automatic):Automatic, opts:OptionsPattern[]] :=
    Module[{contextID},
        contextID = LLVMLibraryFunction["LLVMGetModuleContext"][moduleID];
        driver[ LLVMToObjectFile, input, outputFile, LLVMContext[contextID], opts]
    ]

LLVMToObjectFile[args___] :=
    ThrowException[{"Unrecognized call to LLVMToObjectFile", {args}}]


    
driver[head_, input:(_String | _File | _LLVMModule),
       outputFile:(_String | _File | Automatic):Automatic, LLVMContext[contextID_Integer], opts:OptionsPattern[]] :=
    Module[{systemID, triple, cpu, features, outputtype, commandLineOptions, outputSpec, inputFile, outFile, llvmModule,
            moduleID, errorStrRef, errorStr, targetID, targetMachineRef, res, level, reloc, codeModel},

		{systemID, triple, cpu, features, outputtype, commandLineOptions} = OptionValue[head, {opts},
			{"TargetSystemID","TargetTriple","TargetCPU","TargetFeatures", "OutputType", "CommandLineOptions"}];
         (* SystemID is a way to specify the target in a way that is convenient to Wolfram users. If
            specified, it supplies the triple values. But only if the TargetTriple option is not
            specified. *)

        triple = ResolveLLVMTargetTriple[triple, systemID];

        (* Empty string for this means use current CPU *)
        If[!StringQ[cpu],
            cpu = ""
        ];
        (* Empty string for this means no features *)
        If[!StringQ[features],
            features = ""
        ];

		outputSpec = getOutputTypeSpec[outputtype];

        If[outputFile === Automatic,
            (* Deduce the output filename, if possible, from the input file. *)
            If[!MatchQ[input, File[_]],
                (* Automatic as an output file is only meaningful if the input is a File expression
                   (so the output filename can be deduced from the input file). *)
                Message[head::outfile];
                Return[$Failed]
            ];
            inputFile = First[input];
            (* Convert file to a full pathname if it was specified without any path elements. *)
            If[FileNameDepth[inputFile] === 1,
                inputFile = FileNameJoin[{Directory[], inputFile}]
            ];
            outFile = FileNameJoin[{FileNameDrop[inputFile, -1], FileBaseName[inputFile] <>
                    If[StringMatchQ[triple, "*win32*" | "*windows*"],
                    	outputSpec["windowsExtension"],
                    	outputSpec["otherExtension"]
					]}]
			,
        (* else *)
            (* output file specified explicitly as a File or String. *)
            outFile = If[StringQ[outputFile], outputFile, First[outputFile]];
			(* Because we will be passing the file path to external code that knows nothing about
               Mathematica's current directory, convert file to a full pathname if it was specified
               without any path elements. *)
            If[FileNameDepth[outFile] === 1,
                outFile = FileNameJoin[{Directory[], outFile}]
            ]
        ];

        LLVMLibraryFunction["LLVMInitializeAllTargetInfos"][];
        LLVMLibraryFunction["LLVMInitializeAllTargets"][];
        LLVMLibraryFunction["LLVMInitializeAllTargetMCs"][];
        LLVMLibraryFunction["LLVMInitializeAllAsmParsers"][];
        LLVMLibraryFunction["LLVMInitializeAllAsmPrinters"][];
        (* Don't understand why this init is necessary, after previously calling InitializeAllTargets. *)
        LLVMLibraryFunction["LLVMInitializeNativeTarget"][];
        
        If[commandLineOptions =!= {},
            With[{delimiter = ";"},
                LLVMLibraryFunction["LLVMParseCommandLineOptions2"][
                    StringRiffle[commandLineOptions, delimiter],
                    delimiter,
                    "wolfram compiler"
                ]
            ]
        ];

        (*****  Create a module from the input, if input was not already an LLVMModule  *****)
        If[Head[input] =!= LLVMModule,
            llvmModule = LLVMToModule[input, "LLVMContext" -> LLVMContext[contextID]];
            If[FailureQ[llvmModule],
                (* Rely on messages already issued by LLVMToModule. *)
                Return[llvmModule]
            ],
        (* else *)
            (* Input was already a module. *)
            llvmModule = input
        ];
        moduleID = First[llvmModule];

        (*****  Create a Target from the triple spec  *****)
        targetID = createLLVMTarget[triple];

        level = codeGenOptLevel[OptionValue[head, {opts}, "OptimizationLevel"]];
        If[StringContainsQ[triple, "linux"],
            (*
            The default on Linux is to not generate PIC.

            This resulted in errors, e.g., :
                /usr/bin/x86_64-linux-gnu-ld: /tmp/m000003312561.obj: relocation
                R_X86_64_32S against symbol `Main_Wrapper_Call' can not be used when
                making a shared object; recompile with -fPIC

            So on Linux, make sure to specify PIC
            *)
            reloc = LLVMEnumeration["LLVMRelocMode", "LLVMRelocPIC"];
            ,
            reloc = LLVMEnumeration["LLVMRelocMode","LLVMRelocDefault"];
        ];
        codeModel = LLVMEnumeration["LLVMCodeModel","LLVMCodeModelDefault"];

        (*****  Create a TargetMachine from the Target  *****)
        targetMachineRef = LLVMLibraryFunction["LLVMCreateTargetMachine"][targetID, triple, cpu, features, level, reloc, codeModel];

        ScopedAllocation["CharObjectPointer"][Function[{errorStrRef},
            (****  Generate and emit the object code  ****)
            res = LLVMLibraryFunction["LLVMTargetMachineEmitToFile"][targetMachineRef, moduleID, outFile,
            LLVMEnumeration["LLVMCodeGenFileType", outputSpec["enumeration"]], errorStrRef];

            If[res =!= 0,
                errorStr = LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][errorStrRef, 0];
            ];
        ], 1];

        If[res =!= 0,
            Message[head::llvmerr, "LLVMTargetMachineEmitToFile", errorStr];
            Return[$Failed]
        ];

        LLVMLibraryFunction["LLVMDisposeTargetMachine"][targetMachineRef];

        (* dispose module that was created earlier *)
        If[Head[input] =!= LLVMModule,
            LLVMLibraryFunction["LLVMDisposeModule"][moduleID];
        ];
        
        If[commandLineOptions =!= {},
            LLVMLibraryFunction["LLVMResetCommandLineOptions"][];
        ];

        outFile
    ]

codeGenOptLevel[level:(None | _Integer)] := Switch[level,
	None | 0,
        LLVMEnumeration["LLVMCodeGenOptLevel", "LLVMCodeGenLevelNone"],
	1,
    	LLVMEnumeration["LLVMCodeGenOptLevel", "LLVMCodeGenLevelLess"],
	2,
        LLVMEnumeration["LLVMCodeGenOptLevel", "LLVMCodeGenLevelDefault"],
    3,
        LLVMEnumeration["LLVMCodeGenOptLevel", "LLVMCodeGenLevelAggressive"],
    _,
        Throw[{"Bad code gen level: ", level}]
];

createLLVMTarget[triple_String] := Module[{targetRef, errorStrRef, res, target},
	(* LLVMGetTargetFromTriple takes an LLVMTargetRef* and populates the pointer
	   with an LLVMTargetRef *)
    ScopedAllocation["LLVMTargetObjectPointer"][Function[{targetRef},
    ScopedAllocation["CharObjectPointer"][Function[{errorStrRef},
        res = LLVMLibraryFunction["LLVMGetTargetFromTriple"][triple, targetRef, errorStrRef];
        If[res =!= 0,
            Message[createLLVMTarget::llvmerr, "LLVMGetTargetFromTriple",
                    LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][errorStrRef, 0]];
            Return[$Failed]
        ];
        (* Dereference the populated LLVMTargetRef* *)
        target = LLVMLibraryFunction["LLVMLink_getLLVMTargetObjectPointer"][targetRef, 0];
    ], 1];
    ], 1];
    target
];

getOutputTypeSpec[ "Assembly"] :=
	<|"windowsExtension" -> ".s", "otherExtension" -> ".s", "enumeration" -> "LLVMAssemblyFile"|>

getOutputTypeSpec[ _] :=
	<|"windowsExtension" -> ".obj", "otherExtension" -> ".o", "enumeration" -> "LLVMObjectFile"|>

End[] (* Private *)

EndPackage[]

