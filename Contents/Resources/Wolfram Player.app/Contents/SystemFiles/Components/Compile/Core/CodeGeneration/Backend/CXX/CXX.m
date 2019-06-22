
BeginPackage["Compile`Core`CodeGeneration`Backend`CXX`"]

CreateSymbolicCXXFunctionModulePass
CreateCXXStringFunctionModulePass

Begin["`Private`"]


Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Class`Symbol`"]; (* For MExprSymbolQ *)

Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]

Needs["Compile`Core`PassManager`ProgramModulePass`"]

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]

Needs["CompileUtilities`Markup`"] (* For $UseANSI *)

Needs["Compile`Core`CodeGeneration`Backend`LLVM`MemoryManage`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`MTensorMemoryFinalize`"]


Needs["Compile`Core`CodeGeneration`Backend`LLVM`ExpressionRefCount`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`MObjectCollect`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`GenerateWrapper`"]

Needs["Compile`Core`CodeGeneration`Passes`ConstantPhiCopy`"]
Needs["Compile`Core`CodeGeneration`Passes`DropNoTypeFunction`"]

Needs["SymbolicC`"]
Needs["SymbolicC`SymbolicCXX`"]

Needs["LLVMCompileTools`Types`"]
Needs["Compile`Core`Transform`Closure`ResolveClosure`"]


ClearAll[toCName]
toCName[Native`PrimitiveFunction[f_]] :=
	toCName[f]
toCName[name_?StringQ] :=
	StringReplace[
		name,
		{
			"$" -> "_",
			"." -> "__"
		}
	];
toCName[c_?MExprSymbolQ] :=
	toCName[c["name"]]
toCName[c_?ConstantValueQ] :=
	toCName[c["value"]]
toCName[v_] /; StringQ[getName[v]] :=
	toCName[getName[v]]

ClearAll[getName]
getName[fm_?FunctionModuleQ] :=
	fm["name"]
getName[bb_?BasicBlockQ] :=
	StringRiffle[{"bb", bb["name"], bb["id"]}, "_"]
getName[var_?VariableQ] :=
	If[var["mexpr"] =!= None && var["mexpr"]["symbolQ"],
		toCName[var["mexpr"]["name"]],
		toCName[var["name"]]
	]
	
ClearAll[toCValue]
toCValue[var_?VariableQ] :=
	toCName[getName[var]]
toCValue[c_?ConstantValueQ] :=
	c["value"]

ClearAll[genType];

genType[data_, var_?VariableQ] :=
	genType[data, var["type"]]
genType[data_, c_?ConstantValueQ] :=
	genType[data, c["type"]]
genType[data_, ty_?TypeConstructorQ] :=
	With[{
		name = ty["typename"]
	},
		Switch[name,
				"Void",
					"void",
				"Boolean",
					"bool",
				"Integer8",
					"int8_t",
				"UnsignedInteger8",
					"uint8_t",
				"Integer16",
					"int16_t",
				"UnsignedInteger16",
					"uint16_t",
				"Integer32",
					"int32_t",
				"UnsignedInteger32",
					"uint32_t",
				"Integer64",
					"int64_t",
				"UnsignedInteger64",
					"uint64_t",
				"Real16",
					"half",
				"Real32",
					"float",
				"Real64",
					"double",
				"Complex" | "Complex[Real64]",
					With[{
						realTy = genType[data, TypeSpecifier[Real[64]]]
					},
						SymbolicCXXTemplateInstance["std::complex", realTy]
					],
				"Complex[Real32]",
					With[{
						realTy = genType[data, TypeSpecifier[Real[32]]]
					},
						SymbolicCXXTemplateInstance["std::complex", realTy]
					],
				"String",
					"std::string",		
				"MTensor",
				   	"MTensor",
				"VoidHandle",
				   	CPointerType["void"],  (* LLVM uses the term Pointer *)
				"MObject",
				   	"MObject",
				"Expression",
					"expr",
				_,
					ThrowException[CompilerException[{"Cannot convert type ", ty}]]
			]
	]
	


genType[data_, ty_?TypeApplicationQ] :=
	Which[
		ty["type"]["sameQ", TypeSpecifier["CArray"]] || ty["type"]["sameQ", TypeSpecifier["CUDABaseArray"]] || ty["isNamedApplication", "MIterator"],
			AssertThat[Length[ty["arguments"]] == 1];
			
			With[{
				argTy = genType[data, First[ty["arguments"]]]
			},
				CPointerType[argTy]
			]
			,
		ty["type"]["isConstructor", "CArray"] || ty["type"]["isConstructor", "Handle"],
			With[{
				argTy = genType[data, First[ty["arguments"]]]
			}, 
				CPointerType[argTy]
			]
		,
		
		(*
		  Really should be a struct
		*)
		ty["type"]["isConstructor", "ArrayPartProcessor"],
			CPointerType[CPointerType["void"]]
		,
		ty["type"]["isConstructor", "PackedArray"],
			"MTensor"
		,
		(*
		  Should look at the argument first
		*)
		ty["type"]["isConstructor", "Complex"],
			With[{
				argTy = genType[data, First[ty["arguments"]]]
			}, 
				SymbolicCXXTemplateInstance["std::complex", argTy]
			]
		,
		isStructureType[ty],
			genTypeStructure[data, ty]
		,
		True,
			ThrowException[CompilerException[{"Cannot convert type ", ty}]]
	]



genTypeStructure[data_, ty_] :=
	Module[ {args},
		args = Map[ genType[data,#]&, ty["arguments"]];
		GetStructureType[data, args]
	]


ClearAll[initializeFunction];
ClearAll[functionDecl]
functionDecl[data_, fm_?FunctionModuleQ, body_:None] :=
		With[{
			name = getName[fm],
			resTy = genType[data, fm["result"]["type"]]
		},
		With[{
			res = CFunction[
				{(*If[body === None, "extern", Nothing],*) resTy},
				toCName[name],
				Map[
					{
						genType[data, #["type"]],
						toCName[#["name"]]
					}&,
					fm["arguments"] 
				]
			]
		},
			If[body === None,
				res,
				Append[res, body]
			]
		]]
initializeFunction[data_, fm_] :=
	data["topLevelDecls"]["appendTo", functionDecl[data, fm]]
	
initializeModule[pm_, opts_] :=
	<|
		"topLevelDecls" -> CreateReference[{}],
		"functions" -> CreateReference[<||>],
		"currentFunction" -> CreateReference[<|
			"__decls" -> CreateReference[{}]
		|>]
	|>
	
initializeDataForFunction[pm_, fm_, data_, opts_] :=
	data["currentFunction"]["set", <|
		"__decls" -> CreateReference[{}]
	|>]

initBasicBlock[data_, bb_] :=
	With[{
		name = getName[bb]
	},
		data["currentFunction"]["associateTo", name -> CreateReference[{}]]
	]

addDecl[data_, inst_, line_] :=
	data["currentFunction"]["lookup", "__decls"]["appendTo", line]

addDecl[data_, inst_][line_] :=
	addDecl[data, inst, line]
	
addLine[data_, inst_, line_] :=
	With[{
		bb = getName[inst["basicBlock"]]
	},
		data["currentFunction"]["lookup", bb]["appendTo", line]
	]
	
addLine[data_, inst_][line_] :=
	addLine[data, inst, line]

addDebugLine[data_, inst_] := 
	With[{
		debugValue = inst["getProperty", "debug.value"]
	},
		If[!MissingQ[debugValue],
			With[{
				bb = getName[inst["basicBlock"]]
			},
				Block[{$FrontEnd = Null, $UseANSI = False},
					data["currentFunction"]["lookup", bb]["appendTo", CLine[debugValue["toString"]]]
				]
			]
		]
	]
	
genLabelInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	addLine[data, inst]@
	CLabel[toCName[inst["basicBlock"]]]
)

genStackAllocateInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	addDecl[data, inst]@
	CDeclare[
		First[genType[data, inst["target"]]], (* Get the first to remove the pointer type *)
		CArray[
			toCName[inst["target"]],
			toCValue[inst["size"]]
		]
	];
)
	
genCallInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	addDecl[data, inst]@
	CDeclare[genType[data, inst["target"]], toCName[inst["target"]]];
	
	addLine[data, inst]@
	CAssign[
		toCName[inst["target"]],
		CCall[toCName[inst["function"]], toCValue /@ inst["operands"]]
	]
)
	
genLoadArgumentInstruction[data_, inst_] := 
	If[inst["source"] =!= None,
		addDebugLine[data, inst];
		
		addDecl[data, inst]@
		CDeclare[genType[data, inst["target"]], toCName[inst["target"]]];
		
		addLine[data, inst]@
		CAssign[
			toCName[inst["target"]],
			toCName[inst["source"]]
		]
	]
genLoadInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	addDecl[data, inst]@
	CDeclare[genType[data, inst["target"]], toCName[inst["target"]]];
	
	addLine[data, inst]@
	CAssign[
		toCName[inst["target"]],
		CDereference[toCName[inst["source"]]]
	]
)
genStoreInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	addLine[data, inst]@
	CAssign[
		CDereference[toCName[inst["source"]]],
		toCName[inst["target"]]
	]
)
genBinaryInstruction[data_, inst_] := (
	
	addDebugLine[data, inst];
	
	addDecl[data, inst]@
	CDeclare[genType[data, inst["target"]], toCName[inst["target"]]];
	
	addLine[data, inst]@
	CAssign[
		CDereference[toCName[inst["source"]]],
		COperator[inst["operator"], toCName /@ inst["operands"]]
	]
)

genUnaryInstruction[st_, inst_] := 
	genBinaryInstruction[st, inst]
	
genCompareInstruction[st_, inst_] :=
	genBinaryInstruction[st, inst]
 
genPhiInstruction[data_, inst_] := (
	
	addDebugLine[data, inst];
	
	addDecl[data, inst]@
	CDeclare[genType[data, inst["target"]], toCName[inst["target"]]];
	
	
	MapThread[
		Function[{bb, opr},
			Module[{
				insts, firstJumpPos, cinst
			},
				cinst = CAssign[toCName[inst["target"]], toCName[opr]];
				insts = data["currentFunction"]["lookup", getName[bb]];
				firstJumpPos = FirstPosition[insts["get"], _CGoto | _CIf];
				If[MissingQ[firstJumpPos], 
					insts["appendTo", cinst],
					insts["set",
						Insert[
							insts["get"],
							cinst, 
							firstJumpPos
						]
					]
				]
			]
		],
		Transpose[inst["getSourceData"]]
	]
)

genBranchInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	
	If[inst["isConditional"],
		addLine[data, inst]@
		CIf[toCName[inst["condition"]],
			CGoto[toCName[inst["getOperand", 1]]]
		];
		addLine[data, inst]@
		CGoto[toCName[inst["getOperand", 2]]]
		, (* Else *)
		CGoto[toCName[inst["getOperand", 1]]]
	]
	
)

genReturnInstruction[data_, inst_] := (
	addDebugLine[data, inst];
	
	addLine[data, inst]@
	If[inst["hasValue"],
		CReturn[toCName[inst["value"]]],
		CReturn[]
	]
)

run[pm_, opts_] :=
	Module[{data},
		data = initializeModule[pm, opts];
		

		pm["scanFunctionModules",
			Function[{fm},
	        	initializeFunction[ data, fm]
	        ]
	    ];
	        	
	    pm["scanFunctionModules", Function[{fm},
	    		initializeDataForFunction[pm, fm, data, opts];

	        	fm["topologicalOrderScan",
	        		Function[{bb},
	        			initBasicBlock[data, bb]
	        		]
	        	];
				CreateInstructionVisitor[
					data,
					<|
						"visitLabelInstruction" -> genLabelInstruction,
						"visitStackAllocateInstruction" -> genStackAllocateInstruction,
						"visitLoadArgumentInstruction" -> genLoadArgumentInstruction,
						"visitLoadInstruction" -> genLoadInstruction,
						"visitStoreInstruction" -> genStoreInstruction,
						"visitBinaryInstruction" -> genBinaryInstruction,
						"visitReturnInstruction" -> genReturnInstruction,
						"visitBranchInstruction" -> genBranchInstruction,
						"visitCompareInstruction" -> genCompareInstruction,
						"visitCopyInstruction" -> genCopyInstruction,
						"visitUnaryInstruction" -> genUnaryInstruction,
						"visitCallInstruction" -> genCallInstruction,
						"visitGetElementInstruction" -> genGetElementInstruction,
						"visitSetElementInstruction" -> genSetElementInstruction,
						"visitInertInstruction" -> genInertInstruction,
						"visitLambdaInstruction" -> genLambdaInstruction,
						"visitTypeCastInstruction" -> genTypeCastInstruction,
						"visitLoadGlobalInstruction" -> genLoadGlobalInstruction,
						"visitSelectInstruction" -> genSelectInstruction,
						"" -> Null
					|>,
					fm,
					"IgnoreRequiredInstructions" -> True,
					"TraversalOrder" -> "topologicalOrder"
				];
				CreateInstructionVisitor[
					data,
					<|
						"visitPhiInstruction" -> genPhiInstruction,
						"" -> Null
					|>,
					fm,
					"IgnoreRequiredInstructions" -> True,
					"TraversalOrder" -> "topologicalOrder"
				];

				data["functions"]["associateTo",
					getName[fm] -> functionDecl[data, fm, CBlock[
						ReplaceAll[
							{
								data["currentFunction"]["lookup", "__decls"],
								Reverse[data["currentFunction"]["keyDropFrom", "__decls"]["values"]]
							},
							x_?ReferenceQ -> x["get"] 
						]
					]]
				]
	    ]];
		
	   pm["setProperty", "SymbolicC" ->
	   		CProgram[{
 				data["topLevelDecls"]["get"],
 				data["functions"]["values"]
 			}]
	   ];
	   pm
	]


runCString[pm_, opts_] :=
	pm["setProperty", "CString" ->
		ToCCodeString[pm["getProperty", "SymbolicC"], "Indent" -> Automatic]
	]

RegisterCallback["RegisterPass", Function[{st},
	info = CreatePassInformation[
			"CreateSymbolicCFunctionModule",
			"The pass creates a symbolicC form of the instructions."
	];
	
	CreateSymbolicCXXFunctionModulePass = CreateProgramModulePass[<|
		"information" -> info,
		"runPass" -> run,
		"requires" -> {
			DropNoTypeFunctionPass,
			MemoryManagePass,
			MTensorMemoryFinalizePass,
			MObjectCollectPass,
			ExpressionRefCountPass,
			InferencePass, 
			ResolveClosurePass,
			GenerateWrapperPass,
			ConstantPhiCopyPass
		}
	|>];
	
	RegisterPass[CreateSymbolicCXXFunctionModulePass]
]]


RegisterCallback["RegisterPass", Function[{st},
	info = CreatePassInformation[
			"CreateCStringFunctionModule",
			"The pass creates a symbolicC form of the instructions."
	];
	
	CreateCXXStringFunctionModulePass = CreateProgramModulePass[<|
		"information" -> info,
		"runPass" -> runCString,
		"requires" -> {
			CreateSymbolicCXXFunctionModulePass
		}
	|>];
	
	RegisterPass[CreateCXXStringFunctionModulePass]
]]


End[]

EndPackage[]
