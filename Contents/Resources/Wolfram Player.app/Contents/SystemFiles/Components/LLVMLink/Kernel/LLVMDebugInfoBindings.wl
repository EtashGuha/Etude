
LLVMLibraryFunction["LLVMWLDIBuilderCreate"] :=
LLVMLibraryFunction["LLVMWLDIBuilderCreate"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLDIBuilderCreate_Wrapper",
			{Integer},
			Integer
		];

LLVMLibraryFunction["LLVMWLDIBuilderDispose"] :=
LLVMLibraryFunction["LLVMWLDIBuilderDispose"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLDIBuilderDispose_Wrapper",
			{Integer},
			"Void"
		];

LLVMLibraryFunction["LLVMWLDIBuilder_insertDeclare_atEndOfBB"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_insertDeclare_atEndOfBB"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLDIBuilder_insertDeclare_atEndOfBB_Wrapper",
			{
				Integer, (* DIBuilder* *)
				Integer, (* llvm::Value* Storage *)
				Integer, (* DILocalVariable* VarInfo *)
				Integer, (* DIExpression* Expr *)
				Integer, (* const DILocation* DL *)
				Integer  (* BasicBlock* InsertAtEnd *)
			},
			Integer (* Instruction* *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_insertDbgValueIntrinsic_atEndOfBB"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_insertDbgValueIntrinsic_atEndOfBB"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLDIBuilder_insertDbgValueIntrinsic_atEndOfBB_Wrapper",
			{
				Integer, (* DIBuilder *)
				Integer, (* llvm::Value* *)
				Integer, (* DILocalVariable* *)
				Integer, (* DIExpression* *)
				Integer, (* DILocation* *)
				Integer (* BasicBlock* InsertAtEnd *)
			},
			Integer (* Instruction *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_finalize"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_finalize"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLDIBuilder_finalize_Wrapper",
			{Integer}, (* DIBuilder* *)
			"Void"
		];

LLVMLibraryFunction["LLVMWLDIBuilder_createParameterVariable"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createParameterVariable"] =
	LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMWLDIBuilder_createParameterVariable_Wrapper",
		{
			Integer, (* DIBuilder* *)
			Integer, (* DIScope* Scope *)
			"UTF8String", (* StringRef Name *)
			Integer, (* unsigned ArgNo *)
			Integer, (* DIFile* File *)
			Integer, (* unsigned LineNo *)
			Integer (* DIType* Ty *)
		},
		Integer (* DILocalVariable* *)
	];

LLVMLibraryFunction["LLVMWLDIBuilder_createFunction"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createFunction"] =
	LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createFunction_Wrapper",
		{
			Integer, (* DIBuilder* *)
			Integer, (* DIScope* *)
			"UTF8String", (* Name *)
			"UTF8String", (* LinkageName *)
			Integer, (* DIFile* *)
			Integer, (* unsigned LineNo *)
			Integer, (* DISubroutineType* *)
			True|False, (* bool isLocalToUnit *)
			True|False, (* bool isDefinition *)
			Integer  (* unsigned ScopeLine *)
		},
		Integer (* DISubprogram* *)
];

LLVMLibraryFunction["LLVMWLDIBuilder_createLexicalBlock"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createLexicalBlock"] =
        LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createLexicalBlock_Wrapper",
			{
				Integer, (* DIBuilder* *)
				Integer, (* DIScope* *)
				Integer, (* DIFile* *)
				Integer, (* unsigned Line *)
				Integer  (* unsigned Col *)
			},
			Integer (* DILexicalBlock* *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_createCompileUnit"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createCompileUnit"] =
        LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createCompileUnit_Wrapper",
			{
				Integer, (* DIBuilder* *)
				Integer, (* unsigned Lang *)
				Integer, (* DIFile* *)
				"UTF8String", (* StringRef Producer *)
				True|False, (* bool isOptimized *)
				"UTF8String", (* StringRef Flags *)
				Integer (* unsigned RV (runtime version) *)
			},
			Integer (* DICompileUnit* *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_createFile"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createFile"] =
	LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createFile_Wrapper",
					{
						Integer, (* DIBuilder* *)
						"UTF8String", (* StringRef File  *)
						"UTF8String" (* StringRef Dir *)
					},
					Integer (* DIFile* *)
				];

LLVMLibraryFunction["LLVMWLDIBuilder_createExpression"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createExpression"] =
        LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createExpression_Wrapper",
			{
				Integer  (* DIBuilder* *)
			},
			Integer (* DIExpression * *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_createUnspecifiedType"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createUnspecifiedType"] =
        LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createUnspecifiedType_Wrapper",
			{
				Integer, (* DIBuilder* *)
				"UTF8String" (* Name *)
			},
			Integer (* DIBasicType* *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_createBasicType"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createBasicType"] =
        LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createBasicType_Wrapper",
			{
				Integer, (* DIBuilder *)
				"UTF8String", (* StringRef Name *)
				Integer, (* uint64_t SizeInBits *)
				Integer (* unsigned Encoding *)
			},
			Integer (* DIBasicType* *)
		];

LLVMLibraryFunction["LLVMWLDIBuilder_createPointerType"] :=
LLVMLibraryFunction["LLVMWLDIBuilder_createPointerType"] =
		LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLDIBuilder_createPointerType_Wrapper",
			{
				Integer, (* DIBuilder *)
				Integer (* DIType* *)
			},
			Integer (* DIDerivedType* *)
		];

LLVMLibraryFunction["LLVMWLCreateDebugLocation"] :=
LLVMLibraryFunction["LLVMWLCreateDebugLocation"] =
				LibraryFunctionLoad[LLVMLibraryName[],
				"LLVMLink_LLVMWLCreateDebugLocation",
					{
						Integer, (* DIBuilder* *)
						Integer, (* unsigned int line *)
						Integer, (* unsigned int column *)
						Integer (* DIScope* scope *)
					},
					Integer (* DILocation* *)
				];


LLVMLibraryFunction["LLVMWLIRBuilder_SetCurrentDebugLocation"] :=
LLVMLibraryFunction["LLVMWLIRBuilder_SetCurrentDebugLocation"] =
				LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLIRBuilder_SetCurrentDebugLocation_Wrapper",
					{
						Integer, (* IRBuilder* *)
						Integer, (* unsigned int line *)
						Integer, (* unsigned int column *)
						Integer (* DIScope* scope *)
					},
					"Void"
				];

LLVMLibraryFunction["LLVMWLIRBuilder_SetInstDebugLocation"] :=
LLVMLibraryFunction["LLVMWLIRBuilder_SetInstDebugLocation"] =
				LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLIRBuilder_SetInstDebugLocation_Wrapper",
					{
						Integer, (* IRBuilder* *)
						Integer (* Instruction* *)
					},
					"Void"
				];

LLVMLibraryFunction["LLVMWLFunction_setSubprogram"] :=
LLVMLibraryFunction["LLVMWLFunction_setSubprogram"] =
        		LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMWLFunction_setSubprogram_Wrapper",
					{
						Integer, (* llvm::Function* *)
						Integer (* DISubprogram* *)
					},
					"Void"
				];

LLVMLibraryFunction["LLVMWLCreateLocalVariable"] :=
LLVMLibraryFunction["LLVMWLCreateLocalVariable"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLCreateLocalVariable",
			{
				Integer, (* DIBuilder* *)
				Integer, (* DIScope* *)
				"UTF8String", (* StringRef name *)
				Integer, (* DIFile* *)
				Integer, (* unsigned LineNo *)
				Integer (* DIType* *)
			},
			Integer
		];

LLVMLibraryFunction["LLVMWLCreateBreakpointInst"] :=
LLVMLibraryFunction["LLVMWLCreateBreakpointInst"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLCreateBreakpointInst",
			{
				Integer, (* Module* *)
				Integer (* IRBuilder *)
			},
			Integer (* The resulting LLVMValueRef CallInst *)
		];

LLVMLibraryFunction["LLVMWLCreateInstrProfIncrement"] :=
LLVMLibraryFunction["LLVMWLCreateInstrProfIncrement"] =
        LibraryFunctionLoad[LLVMLibraryName[],
			"LLVMLink_LLVMWLCreateInstrProfIncrement",
			{
				Integer, (* Module* *)
				Integer, (* IRBuilder *)
				(* See http://llvm.org/docs/LangRef.html#llvm-instrprof-increment-intrinsic *)
				"UTF8String", (* Profiling name symbol (function name/branch name, etc.) *)
				Integer, (* i64 Hash *)
				Integer, (* i32 Number of counters *)
				Integer (* i32 counter index *)
			},
			Integer (* LLVMValueRef of the CallInst *)
		];
