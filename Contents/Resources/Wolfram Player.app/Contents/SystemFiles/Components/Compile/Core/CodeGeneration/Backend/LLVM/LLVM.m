BeginPackage["Compile`Core`CodeGeneration`Backend`LLVM`"]

CreateLLVMIRPass
CreateLLVMIRPreprocessPass
CreateLLVMIROnlyPass


Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileAST`Class`Base`"]; (* For MExprQ *)
Needs["CompileAST`Class`Symbol`"]; (* For MExprSymbolQ *)
Needs["CompileAST`Class`Literal`"]; (* For MExprLiteralQ *)
Needs["LLVMLink`"]
Needs["LLVMLink`LLVMInformation`"]
Needs["LLVMCompileTools`"]
Needs["LLVMTools`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`Globals`"]
Needs["LLVMCompileTools`CreateWrapper`"]
Needs["LLVMCompileTools`Comparisons`"]
Needs["LLVMCompileTools`Complex`"]
Needs["LLVMCompileTools`Exceptions`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["CompileUtilities`Markup`"] (* For $UseANSI *)

Needs["Compile`Core`CodeGeneration`Backend`LLVM`MemoryManage`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`MTensorMemoryFinalize`"]
Needs["Compile`Core`Transform`ExceptionHandling`"]
Needs["Compile`Core`Transform`AbortHandling`"]

Needs["Compile`Core`CodeGeneration`Passes`ConstantPhiCopy`"]
Needs["Compile`Core`CodeGeneration`Passes`DropNoTypeFunction`"]

Needs["Compile`Core`CodeGeneration`Backend`LLVM`ExpressionRefCount`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`MObjectCollect`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`GenerateWrapper`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["TypeFramework`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["Compile`Core`Transform`Closure`ResolveClosure`"]
Needs["Compile`Core`Analysis`Loop`LoopNestingForest`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`IR`LoopInformation`"]

(* private imports *)
Needs["Compile`Core`Debug`InsertDebugDeclarePass`"]



genLabelInstruction[data_, inst_] :=
	Null


(*
  These instructions should never be left in the code, they should always be 
  turned into Call instructions.
*)

genBinaryInstruction[data_, inst_] := 
	ThrowException[CompilerException[{"Unexpected Binary instruction", inst}]]

genUnaryInstruction[data_, inst_] := 
	ThrowException[CompilerException[{"Unexpected Unary instruction", inst}]]


genCompareInstruction[data_, inst_] :=
	ThrowException[CompilerException[{"Unexpected Compare instruction", inst}]]


genInertInstruction[data_, inst_] :=
	ThrowException[CompilerException[{"Unexpected Inert instruction", inst}]]

genTypeCastInstruction[data_, inst_] :=
	ThrowException[CompilerException[{"Unexpected TypeCast instruction", inst}]]

genLoadGlobalInstruction[data_, inst_] :=
	ThrowException[CompilerException[{"Unexpected LoadGlobal instruction", inst}]]

genSelectInstruction[data_, inst_] :=
	ThrowException[CompilerException[{"Unexpected Select instruction", inst}]]


(*
  Return the external function type.  First look in a cache.
  If it is not found then look in the type environment,  the type should be in the 
  monomorphic list and there should be one and only one definition.
*)
getExternalFunctionType[ data_, funName_] :=
	Module[ {ty, funs, tyId},
		tyId = data["externalLibraryFunctionTypes"]["lookup", funName, Null];
		If[ tyId === Null,
			funs = data["typeEnvironment"]["functionTypeLookup"]["getMonomorphicList", Native`PrimitiveFunction[funName]];
			If[ !ListQ[funs] || Length[funs] =!= 1,
				ThrowException[{"The type environment does not contain a single definition for the function", funName}]];
			ty = First[funs]["unresolve"];
			tyId = GetLLVMType[data, ty];
			data["externalLibraryFunctionTypes"]["associateTo", funName -> tyId]];
		tyId
	]

addExternalLibraryFunctionType[data_, funName_ -> alias_] :=
	Module[ {funs, ty, tyId},
		funs = data["typeEnvironment"]["functionTypeLookup"]["getMonomorphicList", Native`PrimitiveFunction[funName]];
		If[ !ListQ[funs] || Length[funs] =!= 1,
			ThrowException[{"The type environment does not contain a single definition for the function", funName}]
		];
		ty = First[funs]["unresolve"];
		tyId = GetLLVMType[data, ty];
		data["externalLibraryFunctionTypes"]["associateTo", funName -> tyId];
		data["externalLibraryFunctionTypes"]["associateTo", alias -> tyId];
		tyId
	]
	
addExternalLibraryFunctionType[data_, funName_] :=
	addExternalLibraryFunctionType[data, funName -> funName]

isNamedConstructor[names_, ty_?TypeConstructorQ] :=
	MemberQ[ names, ty["typename"]]

isNamedConstructor[names_, ty_] :=
	False


$integerTypes = {
	"Integer64", "C`int64",
	"Integer32", "C`int16",
	"Integer16", "C`int16",
	"Integer8",
	"C`int"
}

isIntegerType[ty_] := isNamedConstructor[$integerTypes , ty] || isUnsignedIntegerType[ty]

$unsignedIntegerTypes = {
	"UnsignedInteger64", "C`uint64",
	"UnsignedInteger32", "C`uint32",
	"UnsignedInteger16", "C`uint16",
	"UnsignedInteger8",  "C`uint8",
	 "C`uint"
}
isUnsignedIntegerType[ty_] := isNamedConstructor[$unsignedIntegerTypes , ty]

$realTypes = {"C`float", "C`double", "Real16", "Real64", "Real32"}
isRealType[ty_] := isNamedConstructor[ $realTypes, ty]

isComplexType[ty_] := ty["isNamedApplication", "Complex"] && 
						Length[ty["arguments"]] === 1 && 
						isRealType[First[ty["arguments"]]]

isCStringType[ty_] := ty["isNamedApplication", "CArray"] && 
						Length[ty["arguments"]] === 1 && 
						First[ty["arguments"]]["isConstructor", "UnsignedInteger8"]

isHandleType[_, ty_?TypeApplicationQ] :=
	ty["type"]["unresolve"] === TypeSpecifier["Handle"] ||
	ty["type"]["unresolve"] === TypeSpecifier["CArray"] ||
	ty["type"]["unresolve"] === TypeSpecifier["C`ConstantArray"]
	
isHandleType[data_, ty_] :=
	With[{
		tyEnv = data["typeEnvironment"]
	},
		isHandleType[data, tyEnv["resolve", ty]]
	]
	

isHandleType[___] := False

isVoidType[ty_?TypeConstructorQ] :=
	ty["unresolve"] === TypeSpecifier["Void"]
isVoidType[___] := False

isStructureType[_, ty_?TypeApplicationQ] :=
	TrueQ[ty["getProperty", "structQ", False]] || 
	Module[ {metadata = ty["type"]["getProperty", "metadata", Null]},
		metadata =!= Null && KeyExistsQ[metadata, "Fields"]
	]

isStructureType[data_, ty_] :=
	With[{
		tyEnv = data["typeEnvironment"]
	},
		isStructureType[data, tyEnv["resolve", ty]]
	]

isStructureType[_, ty_] :=
	False
	
ClearAll[genType];


genType[data_, ty_?TypeConstructorQ] :=
	With[{
		name = ty["typename"],
		llvmCtxId = data["contextId"],
		typeCache = data["typeCache"]
	},
		If[typeCache["keyExistsQ", name],
			typeCache["lookup", name],
			With[{
				tyId = Switch[name,
					"Void",
						LLVMLibraryFunction["LLVMVoidTypeInContext"][llvmCtxId],
					"Boolean" | "C`bool",
						LLVMLibraryFunction["LLVMInt1TypeInContext"][llvmCtxId],
					"Integer8" | "UnsignedInteger8" | "C`int8" | "C`uint8" | "C`char",
						LLVMLibraryFunction["LLVMInt8TypeInContext"][llvmCtxId],
					"Integer16" | "UnsignedInteger16" | "C`int16" | "C`uint16" | "C`short",
						LLVMLibraryFunction["LLVMInt16TypeInContext"][llvmCtxId],
					"Integer32" | "UnsignedInteger32" | "C`int32" | "C`uint32" | "C`int",
						LLVMLibraryFunction["LLVMInt32TypeInContext"][llvmCtxId],
					"Integer64" | "UnsignedInteger64" | "C`int64" | "C`uint64",
						LLVMLibraryFunction["LLVMInt64TypeInContext"][llvmCtxId],
					"Real64" | "C`double",
						LLVMLibraryFunction["LLVMDoubleTypeInContext"][llvmCtxId],
					"Real16" | "C`half",
						LLVMLibraryFunction["LLVMHalfTypeInContext"][llvmCtxId],
					"Real32" | "C`float",
						LLVMLibraryFunction["LLVMFloatTypeInContext"][llvmCtxId],
					"String",
						GetMStringType[data],		
					"MTensor",
					   	GetMTensorType[data],
					"MNumericArray",
					   	GetMNumericArrayType[data],
					"VoidHandle",
					   	GetVoidPointerType[data],  (* LLVM uses the term Pointer *)
					"MObject",
					   	GetMObjectType[data],
					"Expression",
						GetBaseExprType[data],
					_,
						GetOpaqueStructureType[data, name]
				]
			},
				typeCache["associateTo", name -> tyId];
				tyId
			]
		]
	]
	
genType[data_, ty_?TypeApplicationQ] :=
	With[ {
		key = ty["unresolve"],
		typeCache = data["typeCache"]
	},
		If[typeCache["keyExistsQ", key],
			typeCache["lookup", key],
			With[ {
				val = genTypeApplication[ data, ty]
			},
				typeCache["associateTo", key -> val];
				val
			]]
	]	


genTypeApplication[data_, ty_?TypeApplicationQ] :=
	Which[
		ty["isNamedApplication", "CArray"] || 
		ty["isNamedApplication", "CUDAArray"] || 
		ty["isNamedApplication", "MIterator"],
			AssertThat[Length[ty["arguments"]] == 1];
			
			With[{
				argTy = genType[data, First[ty["arguments"]]]
			},
			With[{
				val = LLVMLibraryFunction["LLVMPointerType"][argTy, 0]
			},
				val
			]]
			,
		ty["isNamedApplication", "CArray"] ||
		ty["isNamedApplication", "Handle"],
			With[{
				argTy = genType[data, First[ty["arguments"]]]
			}, 
				LLVMLibraryFunction["LLVMPointerType"][argTy, 0]
			]
		,
		
		(*
		  Really should be a struct
		*)
		ty["isNamedApplication", "ArrayPartProcessor"],
			LLVMLibraryFunction["LLVMPointerType"][GetVoidPointerType[data], 0]
		,
		ty["isNamedApplication", "PackedArray"],
			GetMTensorType[ data]
		,
		ty["isNamedApplication", "NumericArray"],
			GetMNumericArrayType[ data]
		,
		(*
		  Should look at the argument first
		*)
		ty["isNamedApplication", "Complex"],
			With[{
				argTy = genType[data, First[ty["arguments"]]]
			}, 
				GetVectorComplexType[data, argTy]
			]
		,
		isStructureType[data, ty],
			genTypeStructure[data, ty]
		,
		True,
			ThrowException[CompilerException[{"Cannot convert type ", ty}]]
	]



genTypeStructure[data_, ty_] :=
	Module[ {args},
		args = Map[ genTypeFirstClass[data,#]&, ty["arguments"]];
		GetStructureType[data, args]
	]


genTypeFunctionArgument[ data_, arg_] :=
	Module[ {},
		If[isVoidType[arg],
			ThrowException[{"Cannot generate a function with an argument of Void type."}]];
		genTypeFirstClass[data, arg]
	]

(*
  At present the only object which is known not to be first class is a FunctionType.
  Perhaps this should dig down more deeply,  eg into constructors.
*)
genTypeFirstClass[data_, arg_] :=
	Module[ {ty},
		ty = genType[data,arg];
		If[TypeArrowQ[arg],
			GetPointerType[data,ty]
			,
			ty
		]
	]
	
(*
	If we are generating the type of a function, we need to make sure that all the 
	arguments are themselves first class.  I'm not sure if this should be handled 
	elsewhere,  but this is a mode that works.
	
	TODO revisit this,  there might be better ways to do this.  Maybe the LambdaInstruction
	should return a pointer type.
*)

genType[data_, ty_?TypeArrowQ] :=
	With[{
		argsTy = genTypeFunctionArgument[data, #]& /@ ty["arguments"],
		resultTy = genTypeFirstClass[data, ty["result"]]	
	},
		WrapIntegerArray[ LLVMLibraryFunction["LLVMFunctionType"][resultTy, #, Length[argsTy], 0]&, argsTy]
	]

genType[data_, ty_?TypeLiteralQ] :=
	genType[data, ty["type"]]


genType[data_, ty_?TypeObjectQ] :=
	ThrowException[CompilerException[{"Unknown type ", ty["unresolve"], " when generating LLVM"}]]

genType[data_, ty_] :=
	With[{
		tyEnv = data["typeEnvironment"]
	},
		If[tyEnv["resolvableQ", TypeSpecifier[ty]],
			genType[data, tyEnv["resolve", TypeSpecifier[ty]]],
			ThrowException[CompilerException[{"Invalid type ", ty, " when generating LLVM"}]]
		]
	]



getBasicBlockId[data_, bb_?BasicBlockQ] :=
	With[{
		map = data["basicBlockMap"],
		id = bb["id"]
	},
		If[map["keyExistsQ", id],
			First[map["lookup", bb["id"]]],
			ThrowException[CompilerException[{"basic block Id is not found in the BasicBlock map", map, id}]]
		]
	]

getBasicBlockIdMapped[data_, bb_?BasicBlockQ] :=
	With[{
		map = data["basicBlockMap"],
		id = bb["id"]
	},
		If[map["keyExistsQ", id],
			Last[map["lookup", bb["id"]]],
			ThrowException[CompilerException[{"basic block Id is not found in the BasicBlock map", map, id}]]
		]
	]

getBasicBlockId[data_, arg_] :=
	ThrowException[CompilerException[{"getBasicBlockId called on unknown type", arg}]]


getId[data_, var_?VariableQ] :=
	With[{
		map = data["valueMap"],
		id = var["id"]
	},
		If[map["keyExistsQ", id],
			map["lookup", var["id"]],
			ThrowException[CompilerException[{"variable ", var, " is not found in the value id map ", map, id}]]
		]
	]


getId[data_, const_?ConstantValueQ] :=
	Module[{
		ty = const["type"],
		value = const["value"]
	},
		getConstant[data, value, ty]
	]

getConstant[data_, valueIn_, ty0_] :=
	Module[{
		ty,
		tyEnv = data["typeEnvironment"],
		value = valueIn,
		tyId
	},
		ty = tyEnv["resolve", ty0];
		tyId = genType[data, ty];
		Which[
			MatchQ[value, Native`Global[_String]],
					GetExistingGlobal[ data, First[ value]]
			,
			ty["isConstructor", "Boolean"],
				value = Which[
					value === Compile`Uninitialized,
						0,
					value === Undefined,
						LLVMLibraryFunction["LLVMGetUndef"][tyId],
					True,
					   (*
					    Not sure this is good, maybe should be only good for True/False
					   *)
						value = If[TrueQ[value], 1, 0]
				];
				LLVMLibraryFunction["LLVMConstInt"][tyId, value, 0],
			isIntegerType[ty],
				value = Which[
					value === Compile`Uninitialized,
						0,
					value === Undefined,
						LLVMLibraryFunction["LLVMGetUndef"][tyId],
					True,
						value
				];
				If[value === Compile`Uninitialized || value === Undefined,
					value = LLVMLibraryFunction["LLVMGetUndef"][tyId]
				];
				LLVMLibraryFunction["LLVMConstInt"][tyId, value, 0],
			isRealType[ty],
				value = Which[
					value === Compile`Uninitialized,
						0,
					value === Undefined,
						LLVMLibraryFunction["LLVMGetUndef"][tyId],
					True,
						value
				];
				LLVMLibraryFunction["LLVMConstReal"][tyId, value],
			isComplexType[ty],
				AddCreateVectorComplex[data, _, {
							getConstant[data, Re[value], First[ty["arguments"]]],
							getConstant[data, Im[value], First[ty["arguments"]]]
				}],
			(* Should check that this is some type of reference type *)
			value === Compile`NullReference,
				LLVMLibraryFunction["LLVMConstNull"][tyId],
			ty["isConstructor", "Expression"],
			    getExprConstantId[data, value, ty],
			ty["isNamedApplication", "PackedArray"],
			    GetPackedArrayConstant[data, value],
			ty["isConstructor", "MTensor"],
			    setupExternalConstant[data, value],
			ty["isConstructor", "String"],
			    GetStringConstant[data, value],
			isCStringType[ty],
			    GetCStringConstant[data, value],
			TypeArrowQ[ty],
				getConstantFunction[data, value],
			TypeLiteralQ[ty],
				getConstant[data, ty["value"], ty["type"]],
			True,
				ThrowException[CompilerException[{"Constructor for constant is not implemented.", ty, value}]]
		]
	]

(*
  The name mapping should not be implemented here.
*)
setupExternalConstant[ data_, value_] :=
	Switch[ value,
		Native`EternalMTensor,
			GetExternalConstant[data, "EternalMTensor"]
		,
		_,
			ThrowException[CompilerException[{"Unknown External Constant", value}]]
	]

getId[data_, arg_] :=
	ThrowException[CompilerException[{"getId called on unknown type", arg}]]


getExprConstantId[data_, value_, ty_] :=
	If[ 
		value === Compile`Internal`ENULLReference,
		GetExternalConstant[data, "ENULL"],
		GetExprConstant[data, value]
	]


	
getName[fm_?FunctionModuleQ] :=
	fm["name"]
getName[bb_?BasicBlockQ] :=
	StringRiffle[{bb["name"], bb["id"]}, "_"]
getName[var_?VariableQ] :=
	var["name"]

toPlainString[s_] :=
	Block[{$FrontEnd = Null, $UseANSI = False},
		s["toString"]
	]
	
getValueName[var_?VariableQ] :=
	"var"<>StringTrim[toPlainString[var], "\"" | "%"]

ClearAll[setValueName];
setValueName[id_, var_?VariableQ] :=
	With[{name = getValueName[var]},
		If[!StringQ[name] || isVoidType[var["type"]],
			Return[]
		];
		If[$LLVMInformation["LLVM_VERSION"] >= 7.0,
			LLVMLibraryFunction["LLVMSetValueName2"][id, name, Length[name]],
			LLVMLibraryFunction["LLVMSetValueName"][id, name]
		];
		id
	]; 
	
initBasicBlock[data_, bb_] :=
	With[{
		funId = data["functionId"]["get"],
		name = getName[bb]
	},
		addBasicBlockMap[data,
			bb["id"],
			LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], funId, name]
		]
	]
	
genBasicBlock[data_, bb_] :=
	With[{
		bbId = getBasicBlockId[data, bb]
	},
		If[bbId === Null,
			ThrowException[CompilerException[{"Cannot find basic block in Map ", bb["id"]}]]
		];
		LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbId];
		data["currentW_BBId"]["set", bb["id"]];
		bbId
	]
	

findLoop[loopInfo_, bb_] :=
    Module[{
        header = loopInfo["header"],
        children = loopInfo["children"]["get"]
    },
        If[header =!= None && bb["sameQ", header],
            Return[loopInfo]
        ];
        Do[
            If[LoopInformationQ[child],
                Return[findLoop[child, bb], Module],
                If[bb["sameQ", child],
                    Return[loopInfo, Module]
                ]
            ],
            {child, children}
        ];
        None
    ]

(* We are trying to check if the inst is a backedge 
 * and it has no child loops
 *)
isLoopLeafBackedgeQ[fmLoopInfo_, inst_] :=
    Module[{
        bbLoopInfo,
        bb = inst["basicBlock"]
    },
        bbLoopInfo = findLoop[fmLoopInfo, bb];
        If[bbLoopInfo === None,
            Return[False]
        ];
        If[Count[bbLoopInfo["children"]["get"], _?LoopInformationQ] === 0 && !bbLoopInfo["header"]["sameQ", bb],
            Return[True]
        ];
        False
    ]
  
mkMDNode[data_, name_?StringQ] :=
    mkMDNode[data, {name}]
mkMDNode[data_, names_?ListQ] :=
    Module[{id},
        WrapIntegerArray[LLVMLibraryFunction["LLVMMDNodeInContext"][data["contextId"], #, Length[names]]&,
            Table[
                Which[
                    IntegerQ[name],
                        name,
                    StringQ[name],
                        id = LLVMLibraryFunction["LLVMMDStringInContext"][data["contextId"], name, StringLength[name]];
                        id
                ],
                {name, names}
            ]
        ]
    ]
    

ClearAll[addLoopHint]
addLoopHint["Unroll"][data_, val_] :=
    With[{
        enable = AddConstantBoolean[data, val]
    },
        mkMDNode[data, {"llvm.loop.unroll.enable", enable}]
    ]
addLoopHint["UnrollFactor"][data_, factor_?IntegerQ] :=
    With[{
        factorId = AddConstantInteger[data, 32, factor]
    },
        If[factor <= 1,
            Nothing,
            mkMDNode[data, {"llvm.loop.unroll.count", factorId}]
        ]
    ]
addLoopHint["UnrollFactor"][data_, "Full"] :=
    mkMDNode[data, "llvm.loop.unroll.full"]
addLoopHint["Interleave"][data_, val_] :=
    With[{
        enable = AddConstantBoolean[data, val]
    },
        mkMDNode[data, {"llvm.loop.interleave.enable", enable}]
    ]
addLoopHint["InterleaveFactor"][data_, factor_] :=
    With[{
        factorId = AddConstantInteger[data, 32, factor]
    },
        If[factor <= 1,
            Nothing,
            mkMDNode[data, {"llvm.loop.interleave.count", factorId}]
        ]
    ]
addLoopHint["UnrollAndJam"][data_, val_] :=
    If[TrueQ[val],
        mkMDNode[data, {"llvm.loop.unroll_and_jam.enable"}],
        mkMDNode[data, {"llvm.loop.unroll_and_jam.disable"}]
    ]
addLoopHint["UnrollAndJamFactor"][data_, factor_] :=
    With[{
        factorId = AddConstantInteger[data, 32, factor]
    },
        If[factor <= 1,
            Nothing,
            mkMDNode[data, {"llvm.loop.unroll_and_jam.count", factorId}]
        ]
    ]
addLoopHint["Vectorize"][data_, val_] :=
    With[{
        enable = AddConstantBoolean[data, val]
    },
        mkMDNode[data, {"llvm.loop.vectorize.enable", enable}]
    ]
addLoopHint["VectorizeWidth"][data_, val_] :=
    With[{
        id = AddConstantInteger[data, 32, val]
    },
        If[val <= 1,
            Nothing,
            mkMDNode[data, {"llvm.loop.vectorize.width", id}]
        ]
    ]
addLoopHint[_][data_, factor_] :=
    Nothing
  

addLoopHints[data_, id_] :=
    Module[{
        root = "llvm.loop",
        hints = {},
        dummy, 
        rootId,
        loopHints = data["LoopHints"]
    },
        If[TrueQ[loopHints["Vectorize"]],
            AppendTo[hints, {
                addLoopHint["Vectorize"][data, loopHints["Vectorize"]],
                addLoopHint["VectorizeWidth"][data, loopHints["VectorizeWidth"]]
            }];
        ];
        If[TrueQ[loopHints["UnrollAndJam"]],
            AppendTo[hints, {
                addLoopHint["UnrollAndJam"][data, loopHints["UnrollAndJam"]],
                addLoopHint["UnrollAndJamFactor"][data, loopHints["UnrollAndJamFactor"]]
            }];
        ];
        If[TrueQ[loopHints["Unroll"]],
            AppendTo[hints, {
                addLoopHint["Unroll"][data, loopHints["Unroll"]],
                addLoopHint["UnrollFactor"][data, loopHints["UnrollFactor"]]
            }];
        ];
        If[TrueQ[loopHints["Interleave"]],
            AppendTo[hints, {
                addLoopHint["Interleave"][data, loopHints["Interleave"]],
                addLoopHint["InterleaveFactor"][data, loopHints["InterleaveFactor"]]
            }];
        ];
        hints = Flatten[hints];
        If[hints === {},
            Return[]
        ];
        rootId = LLVMLibraryFunction["LLVMGetMDKindIDInContext"][data["contextId"], root, StringLength[root]];
        dummy = mkMDNode[data, "dummy"];
        hints = mkMDNode[data, Prepend[hints, dummy]];
        LLVMLibraryFunction["LLVMSetMetadata"][id, rootId, hints];
        LLVMLibraryFunction["LLVMSetMDNodeOperand"][hints, 0, LLVMLibraryFunction["LLVMValueAsMetadata"][hints]];
    ]


$DefaultUnrollHints = <|
    "Unroll" -> True,
    "UnrollFactor" -> 4,
    "UnrollAndJam" -> False,
    "UnrollAndJamFactor" -> 4,
    "Interleave" -> True,
    "InterleaveFactor" -> 4,
    "Vectorize" -> False,
    "VectorizeWidth" -> 4
|>


processLoopHints[ llvmOpt_, loopHints:OptionsPattern[]] :=
	processLoopHints[ llvmOpt, Association @@ {Flatten[loopHints]}]

processLoopHints[ llvmOpt_, loopHints_?AssociationQ] :=
	Join[ $DefaultUnrollHints, loopHints]

processLoopHints[ "ClangOptimization"[lev_], Automatic] :=
	If[lev > 1, $DefaultUnrollHints, None]

processLoopHints[ ___] :=
	None

genUnconditionalBranchInstruction[data_, inst_] :=
	With[{
        fm = inst["basicBlock"]["functionModule"],
		target = getBasicBlockId[data, First[inst["operands"]]]
	},
	Module[{
	    loopInfo,
	    res
	},
        res = LLVMLibraryFunction["LLVMBuildBr"][
        	data["builderId"],
			target
        ];
        
        loopInfo = fm["getProperty", "loopinformation", Null];
        If[AssociationQ[data["LoopHints"]] && loopInfo =!= None && isLoopLeafBackedgeQ[loopInfo, inst],
            addLoopHints[data, res];
        ];
        
        res
	]]
	
genConditionalBranchInstruction[data_, inst_] :=
	With[{
		condId = getId[data, inst["condition"]],
		targets = Map[getBasicBlockId[data, #]&, inst["operands"]]
	},
	Module[{
	    res
	},
		If[ Length[targets] =!= 2,
			ThrowException[CompilerException[{"Branch Error"}]]
		];
		If[ MemberQ[targets, Null],
			ThrowException[CompilerException[{"Cannot find basic block in Map ", inst["operands"]}]]
		];
        res = LLVMLibraryFunction["LLVMBuildCondBr"][
        	data["builderId"],
        	condId,
        	First[targets],
        	Last[targets]
        ];
        res
	]]


genBranchInstruction[data_, inst_] :=
	If[inst["isConditional"],
		genConditionalBranchInstruction[data, inst],
		genUnconditionalBranchInstruction[data, inst]
	]

(*
 Check the source of the Phi instruction to see if it is a Constant with type Void.
 If so give a special error message.
*)
checkPhiSource[data_, inst_, src_] :=
	Module[{mexpr, expr},
		If[ !ConstantValueQ[src] || !isVoidType[src["type"]],
			Return[]];
		mexpr = inst["mexpr"];
		If[ MExprQ[mexpr],
			 mexpr = If[mexpr["hasProperty", "originalExpr"],
						mexpr["getProperty", "originalExpr"]];
			expr = HoldForm @@ mexpr["toExpression"];
			ThrowException[{"Cannot process the Null result of `1`.", expr}]
			,
			ThrowException[{"Cannot process the Null result of branch."}]];
	]

processPhi[data_] :=
	Module[ {keys},
		keys = data["phiMap"]["keys"];
		Scan[
			Module[ {inst, valIds, bbIds, tmp, srcVars},
				inst = data["phiMap"]["lookup", #, Null];
				If[ inst === Null,
					ThrowException[CompilerException[{"Phi Map process error ", #}]]
				];
				tmp = Map[#["id"]&, inst["getSourceBasicBlocks"]];
				bbIds = Map[getBasicBlockIdMapped[data, #]&, inst["getSourceBasicBlocks"]];
				srcVars = inst["getSourceVariables"];
				Scan[ checkPhiSource[data, inst, #]&, srcVars];
				valIds = Map[ getId[data,#]&, srcVars];
				WrapIntegerArray[ Function[{vals, bbs},
						LLVMLibraryFunction["LLVMAddIncoming"][ #, vals, bbs, Length[valIds]]], valIds, bbIds];
			]&,
			keys];
	]
	
genPhiInstruction[data_, inst_] :=
	Module[ {srcVars, tyId, phiId},
		srcVars = inst["getSourceVariables"];
		tyId = genTypeFirstClass[data, First[srcVars]["type"]];
		phiId = LLVMLibraryFunction["LLVMBuildPhi"][ data["builderId"], tyId,  ""];
		setValueName[phiId, inst["target"]];
		(*setInstDebugLocation[data, phiId, inst];*)
		addValueMap[data, inst["target"]["id"], phiId];
		data["phiMap"]["associateTo", phiId -> inst];
	]
	

getConstantFunction[data_, value_] :=
	Module[{funId},
		funId = data["functionMap"]["lookup", value, Null];
		If[ funId === Null,
			ThrowException[CompilerException[{"Local function cannot be found ", value}]]
		];
		funId
	]


genLambdaInstruction[data_, inst_] :=
	Module[ {source, funId},
		source = inst["source"];
		If[ !ConstantValueQ[source],
			ThrowException[{"Lambda instruction must have a constant value for a source", source, inst}]
		];
		funId = data["functionMap"]["lookup", source["value"], Null];
		If[ funId === Null,
			ThrowException[CompilerException[{"Local function cannot be found ", source}]]
		];
		addValueMap[data, inst["target"]["id"], funId]
	]




genDIType[data_, ty_?TypeConstructorQ] :=
		With[{
			name = ty["name"],
			diTypeCache = data["diTypeCache"],
			(* Defined in the DWARF 4 specification. Not all values defiend there are reproduced here. *)
			DWATEBoolean = 2, DWATEFloat = 4, DWATESigned = 5, DWATEUnsigned = 7
		},
			If[diTypeCache["keyExistsQ", name],
				diTypeCache["lookup", name],
				With[{
					typeData = Switch[name,
						"Void",
							ThrowException[{"No DIType for Void"}]
						,
						"Boolean",
							{8, DWATEBoolean},
						"Integer[8]",
							{8, DWATESigned},
						"UnsignedInteger8",
							{8, DWATEUnsigned},
						"Integer16",
							{16, DWATESigned},
						"UnsignedInteger16",
							{16, DWATEUnsigned},
						"Integer32",
							{32, DWATESigned},
						"UnsignedInteger32",
							{32, DWATEUnsigned},
						"Integer64" ,
							{64, DWATESigned},
						"UnsignedInteger64" | "UnsignedInteger",
							{64, DWATEUnsigned},
						"Real" | "Real64",
							{64, DWATEFloat},
						"Real16",
							{16, DWATEFloat},
						"Real32",
							{32, DWATEFloat},
						"String" | "MTensor" | "MObject" | "Expr",
							ThrowException[{"genDIType is not implemented for all types: ", ty}]
						,
						_,
							ThrowException[{"genDIType is not implemented for all types: ", ty}]
					        (* LLVMLibraryFunction["LLVMWLDIBuilder_createUnspecifiedType"][data["diBuilderId"], "Unspecified"];*)
					]
				},
					With[{tyId = LLVMLibraryFunction["LLVMWLDIBuilder_createBasicType"][data["diBuilderId"],
						name, typeData[[1]], typeData[[2]]]
						(*                   ^^^^^^^^^^^^^^
						TODO Question?: Make sure this   
						is handled correctly for more complex/compound types.
						The 4th argument above is alignment in bits. For basic types I believe this
						is the same as size, but for compound types we'll need to make a choice.
						See: http://www.catb.org/esr/structure-packing/#_structure_alignment_and_padding *)
					},
						diTypeCache["associateTo", name -> tyId];
						tyId
					]
				]
			]
		];

isPointerType[TypeSpecifier["Handle"[x_]]] := True
isPointerType[TypeSpecifier["CArray"[x_]]] := True
isPointerType[___] := False

genDIType[data_, ty_?TypeApplicationQ] := Module[{handlePointeeType, typeId},
	If[isPointerType[ty["unresolve"]],
		handlePointeeType = genDIType[data, ty["arguments"][[1]]];
		AssertThat["handlePointeeType should be an integer", handlePointeeType]["named", "handlePointeeType"]["satisfies", IntegerQ];
		typeId = LLVMLibraryFunction["LLVMWLDIBuilder_createPointerType"][data["diBuilderId"], handlePointeeType];
		typeId
		,
		Throw[{"genDIType cannot handle general TypeApplications: ", ty}];
	]
];
genDIType[data_, Null] := LLVMLibraryFunction["LLVMWLDIBuilder_createUnspecifiedType"][data["diBuilderId"], "Unspecified"];
genDIType[data_, args___] := Throw @@ {{"genDIType called with bad arguments: ", args}};


processDebugDeclare[data_, inst_, funName_, funData_] :=
	Module[{ value,
		mexpr, symbolName, diScopeId, diFileRef, diLocalVar, valueId, diExprId, debugLocationId, bbId, diInstId},

	value = inst["getProperty", "debug.value"];
	mexpr = inst["mexpr"];

	AssertThat["PrimitiveFunction[DebugDeclare] mexpr should be MExprQ", mexpr]["named", "mexpr"]["satisfies", MExprQ];
	symbolName = inst["getProperty", "debug.name"];
		
	diScopeId = Module[{parentScope},
		parentScope = parentScopingMExpr[mexpr];
		If[parentScope === Null,
			ThrowException[{"Got Null parent scope for: ", mexpr}]
		];
		genScope[data, parentScope]
	];

	diFileRef = data["sourceDIFileRef"];
	AssertThat["diFileRef should be an integer", diFileRef]["named", "diFileRef"]["satisfies", IntegerQ];

	diLocalVar = Module[{diVariableCache = data["diVariableCache"], argNum = None, diTypeId = None, newDIVariable},
		If[diVariableCache["keyExistsQ", symbolName],
			diVariableCache["lookup", symbolName]
			,
			Which[
				VariableQ[value],
					argNum = value["getProperty", "argNum", None];
					diTypeId = genDIType[data, value["type"]],
				ConstantValueQ[value],
					diTypeId = genDIType[data, value["type"]],
				True,
					Throw["processDebugDeclare: expected value to be VariableQ or ConstantValueQ"]
					(*diTypeId = genDIType[data, Null];*)
			];

			AssertThat["diTypeId should be an integer", diTypeId]["named", "diTypeId"]["satisfies", IntegerQ];

			(* TODO: Cache DIVariables; otherwise a new one is allocated for every Variable in the WIR. *)
			newDIVariable = If[argNum === None,
			(*Print["creating local `", symbolName, "` variable with diScopeId: ", BaseForm[diScopeId, 16]];*)
				LLVMLibraryFunction["LLVMWLCreateLocalVariable"][
					data["diBuilderId"], diScopeId, symbolName, diFileRef, 1, diTypeId],
			(* else *)
			(*Print["creating param `", symbolName, "` variable with diScopeId: ", BaseForm[diScopeId, 16]];*)
				LLVMLibraryFunction["LLVMWLDIBuilder_createParameterVariable"][
					data["diBuilderId"], diScopeId, symbolName, argNum, diFileRef, 1, diTypeId]
			];
			diVariableCache["associateTo", symbolName -> newDIVariable];
			newDIVariable
		]
	];
	AssertThat["diLocalVar should be an integer", diLocalVar]["named", "diLocalVar"]["satisfies", IntegerQ];

	valueId = Which[
		VariableQ[value],
			(* An instance of an llvm::Value *)
			data["valueMap"]["lookup", value["id"], None],
		ConstantValueQ[value],
			getId[data, value],
		True,
			ThrowException[{"Cannot get value id for value: ", value}]
	];
	AssertThat["valueId should be an integer", valueId]["named", "valueId"]["satisfies", IntegerQ];

	diExprId = LLVMLibraryFunction["LLVMWLDIBuilder_createExpression"][data["diBuilderId"]];
	AssertThat["diExprId should be an integer", diExprId]["named", "diExprId"]["satisfies", IntegerQ];

	debugLocationId = Module[{line, column = 1},
		line = Which[
			mexpr["hasProperty", "sourceLine"],
			    mexpr["getProperty", "sourceLine"],
			MExprSymbolQ[mexpr] || MExprLiteralQ[mexpr],
			    AssertThat["mexpr parent should exist", mexpr["parent"]]["named", "mexpr[parent]"]["satisfies", MExprQ];
			    mexpr["parent"]["getProperty", "sourceLine", 1],
			True,
				1
		];
		LLVMLibraryFunction["LLVMWLCreateDebugLocation"][data["contextId"], line, column, diScopeId]
	];
	AssertThat["debugLocationId should be an integer", debugLocationId]["named", "debugLocationId"]["satisfies", IntegerQ];

	bbId = Module[{currBBId = data["currentW_BBId"]["get"], val},
		val = data["basicBlockMap"]["lookup", currBBId, None];
		AssertThat["val should be a pair", val]["named", "val"]["satisfies", (Length[#] === 2 && IntegerQ[Part[#,2]])&];
		(* TODO: Ensure this is sound; I don't know the situtations in which this map is invalidated *)
		Part[val, 2]
	];

	diInstId = LLVMLibraryFunction["LLVMWLDIBuilder_insertDeclare_atEndOfBB"][data["diBuilderId"],
		valueId, diLocalVar, diExprId, debugLocationId, bbId
	];
	AssertThat["diInstId should be an integer", diInstId]["named", "diInstId"]["satisfies", IntegerQ];
	diInstId
];

processDebugDeclare[data_, args___] :=
	ThrowException[{"processDebugDeclare called with bad arguments: ", args}]








genLLVMInternal[data_, inst_, Native`PrimitiveFunction["BitCast"], funData_] :=
	Module[ {target, args, arg, sourceId, tyIn, tyOut, trgtTy, valId},
		target = inst["target"];
		tyOut = target["type"];
		trgtTy = genTypeFirstClass[data, tyOut];
		args = inst["arguments"];
		If[ Length[args] =!= 1,
			ThrowException[{"BitCast has incorrect number of instructions", Length[args], inst}]
		];
		arg = First[ args];
		tyIn = arg["type"];
		sourceId = getId[data, arg];
		valId = Which[ 
			tyOut["isConstructor", "VoidHandle"] && isIntegerType[tyIn],
				AddCastIntegerToPointer[data, sourceId, trgtTy]
			,
			tyIn["isConstructor", "VoidHandle"] && isIntegerType[tyOut],
				AddCastPointerToInteger[data, sourceId, trgtTy],
			tyOut["hasProperty", "ByteCount"] && tyIn["hasProperty", "ByteCount"] &&
			tyOut["getProperty", "ByteCount"] < tyIn["getProperty", "ByteCount"],
				AddTypeDownCast[data, sourceId, trgtTy],
			True,
				AddTypeCast[data, sourceId, trgtTy]
		];
		addValueMap[ data, target["id"], valId]
	]




genExternalCall[ data_, inst_, funName_, funData_, setupCallFun_, setupInvokeFun_] :=
	With[{
		operands = inst["operands"],
		target = inst["target"]
	},
	With[{
		args = getId[data, #]& /@ operands
	},
		Module[ {valId},
                        data["callFunctionType"]["set", inst["function"]["type"]["unresolve"]];
			valId = If[CallInstructionQ[inst], 
				setupCallFun[data, funName, args], 
				setupInvokeFun[data, funName, args, getBasicBlockId[data, inst["to"]], getBasicBlockId[data, inst["unwind"]]]
			];
                        data["callFunctionType"]["set", Null];
			If[!isVoidType[target["type"]],
				addValueMap[ data, target["id"], valId]
			];
			valId
		]
	]]


genRuntimeCall[ data_, inst_, Native`PrimitiveFunction[funName_], funData_] :=
	genExternalCall[data, inst, funName, funData, AddRuntimeFunctionCall, AddRuntimeFunctionInvokeModel]
genRuntimeCall[args___] :=
    ThrowException[{"Unexpected arguments to genRuntimeCall", args}]

genLLVMCompileTools[ data_, inst_, Native`PrimitiveFunction[funName_], funData_] :=
	genExternalCall[data, inst, funName, funData, AddLLVMCodeCall, Null]
genLLVMCompileTools[_, args___] :=
    ThrowException[{"Unexpected arguments to genLLVMCompileTools", args}]

genExternal[ data_, inst_, Native`PrimitiveFunction[funName0_], funData_] :=
	Module[{ty, funName},
		funName = If[KeyExistsQ[funData, "ExternalFunctionName"],
			funData["ExternalFunctionName"],
			funName0
		];
		ty = data["externalLibraryFunctionTypes"]["lookup", funName, Null];
		If[ ty === Null,
			addExternalLibraryFunctionType[data, funName0 -> funName]
		];
		genExternalCall[data, inst, funName, funData, AddFunctionCall, Null]
	]
genExternal[args___] :=
    ThrowException[{"Unexpected arguments to genExternal", args}]


(*
 Add a definition with linkage of ExternalLibrary, need to check to see if the type 
 has been added and if not add it.
*)
genExternalLibrary[ data_, inst_, Native`PrimitiveFunction[ funName0_], funData_] :=
	Module[ {ty, funName},
		funName = Lookup[funData, "ExternalFunctionName", funName0];
		ty = data["externalLibraryFunctionTypes"]["lookup", funName, Null];
		If[ ty === Null,
			addExternalLibraryFunctionType[data, funName0 -> funName]
		];
		genExternalCall[data, inst, funName, funData, AddExternalLibraryCall, Null]
	]


(*
 Add a definition with linkage of LLVMModule.
*)
genLLVMModule[ data_, inst_, Native`PrimitiveFunction[ funName_], funData_] :=
	Module[ {mod, funId, valId, target = inst["target"]},
		mod = Lookup[funData, "LLVMModule", Null];
		If[!MatchQ[mod, LLVMModule[_]],
			ThrowException[{"Cannot find LLVMModule specification.", funName, funData}]
		];
		funId = GetLLVMModuleFunction[data, mod, funName];
		valId = addLocalFunctionCall[data, inst, funId];
		addValueMap[ data, target["id"], valId];
		valId
	]


genLLVMCompareFunction[ data_, inst_, Native`PrimitiveFunction[funName_], funData_] :=
	genExternalCall[data, inst, funName, funData, AddLLVMCompareCall, Null]

genLLVMDebug[data_, inst_, Native`PrimitiveFunction[funName_], funData_] := Module[{},
	(* TODO: Assuming that if diBuilderId is IntegerQ, then LLVMDebug is True *)
	If[!IntegerQ[data["diBuilderId"]],
		ThrowException[{"Native`PrimitiveFunction[" <> funName <> "] is illegal when \"LLVMDebug\" is not True"}]
	];
    Switch[funName,
		"DebugDeclare",
			processDebugDeclare[data, inst, Native`PrimitiveFunction[funName], funData],
		_,
			ThrowException[{"PrimitiveFunction[" <> funName <> "] is not implemented"}]
	]
];



addIntrinsicFunctionType[data_, funName_, funData_] :=
	Module[ {mod, intrinsicName, intrinsicId, funs, ty, argTyIds, funId},
		
		intrinsicName = funData["LLVMIntrinsicName"];
		If[!StringQ[intrinsicName],
			ThrowException[{"Cannot find LLVMModule intrinsic for " <> funName, intrinsicName}]
		];
		
		mod = data["moduleId"];
		If[!IntegerQ[mod],
			ThrowException[{"Cannot find LLVMModule specification while creating intrinsic function.", funName, funData}]
		];
		intrinsicId = LLVMLibraryFunction["LLVMLookupIntrinsicID"][intrinsicName];
		If[!IntegerQ[intrinsicId],
			ThrowException[{"Native`PrimitiveFunction[" <> funName <> "] resolved to " <> intrinsicName <> " which is not a known LLVM intrinsic "}]
		];
		
		
		funs = data["typeEnvironment"]["functionTypeLookup"]["getMonomorphicList", Native`PrimitiveFunction[funName]];
		If[ !ListQ[funs] || Length[funs] =!= 1,
			ThrowException[{"The type environment does not contain a single definition for the function", funName}]
		];
		
		ty = First[funs];
		argTyIds = GetLLVMType[data, #["unresolve"]]& /@ ty["arguments"];
		
		funId = LLVMLibraryFunction["LLVMIntrinsicDeclaration"][mod, intrinsicId, argTyIds];
		If[!IntegerQ[intrinsicId],
			ThrowException[{"Unable to get function ID Native`PrimitiveFunction[" <> funName <> "] resolved to " <> intrinsicName <> " which is not a known LLVM intrinsic."}]
		];
		
		data["intrinsicFunctionIds"]["associateTo", funName -> funId];
		funId
	]
	
lookupOrAddIntrinsicFunctionType[data_, funName_, funData_] :=
	Module[{id},
		id = data["intrinsicFunctionIds"]["lookup", funName, Null];
		If[id =!= Null,
			id,
			addIntrinsicFunctionType[data, funName, funData]
		]
	];

genIntrinsic[data_, inst_, Native`PrimitiveFunction[funName_], funData_] :=
	Module[{
		funId, valId, target = inst["target"]
	},
		funId = lookupOrAddIntrinsicFunctionType[data, funName, funData];
		valId = addLocalFunctionCall[data, inst, funId];
		addValueMap[ data, target["id"], valId];
		valId
	]

$linkageFunction =
<|
	"Runtime" -> genRuntimeCall,
	"LLVMCompileTools" -> genLLVMCompileTools,
	"LLVMDebug" -> genLLVMDebug,
	"LLVMCompareFunction" -> genLLVMCompareFunction,
	"LLVMInternal" -> genLLVMInternal,
	"External" -> genExternal,
	"ExternalLibrary" -> genExternalLibrary,
	"LLVMModule" -> genLLVMModule,
	"Intrinsic" -> genIntrinsic

|>

throwError[data_, inst_, funName_, funData_] :=
	ThrowException[{"Unknown linkage", funName, funData}]


(*
  Utility functions useful for printing LLVM type 
*)
printType[data_, id_] :=
	Module[{ty},
		ty = LLVMLibraryFunction["LLVMTypeOf"][id];
		Print[ LLVMToString[LLVMType[ty]]];
	]

printType[ data_, list_List] :=
	Scan[printType[data,#]&, list]

addLocalFunctionCall[data_, inst_, funId_] :=
		Module[ {trgt, args, argIds, valId},
			trgt = inst["target"];
			args = inst["arguments"];
			argIds = Map[ getId[data,#]&, args];
			valId = If[CallInstructionQ[inst], 
				AddBuildCall[data, funId, argIds], 
				AddBuildInvokeModel[data, funId, argIds, getBasicBlockId[data, inst["to"]], getBasicBlockId[data, inst["unwind"]]]];
			If[!isVoidType[trgt["type"]],
				setValueName[valId, inst["target"]]
			];
			valId
		]


genInvokeInstruction[data_, inst_] :=
	genCallInvokeInstruction[data, inst]

genCallInstruction[data_, inst_] :=
	genCallInvokeInstruction[data, inst]

genCallInvokeInstruction[data_, inst_] := 
	Module[ {pm, fun, funName, funId, valId, funData, linkFun},
		pm = inst["basicBlock"]["functionModule"]["programModule"];
		fun = inst["function"];
		If[ ConstantValueQ[fun],
			funName = fun["value"];
			funData = pm["externalDeclarations"]["lookupFunction", funName];
			Which[
				AssociationQ[funData] && KeyExistsQ[funData, "Linkage"],
					linkFun = Lookup[ $linkageFunction, funData["Linkage"], throwError];
					valId = linkFun[data, inst, funName, funData];
					addMetadata[data, valId, "linkage" -> funData["Linkage"]];
					addMetadata[data, valId, "name" -> funName];
				,
				data["functionMap"]["keyExistsQ", funName],
					funId = data["functionMap"]["lookup", funName];
					valId = addLocalFunctionCall[data, inst, funId];
					addMetadata[data, valId, "linkage" -> "local"];
					addValueMap[data, inst["target"]["id"], valId];
				,
				True,
					ThrowException[{"Cannot resolve function call ", funName, " during LLVM code generation."}]
			];
			,
			funId = getId[data,fun];
			valId = addLocalFunctionCall[data, inst, funId];
			addMetadata[data, valId, "linkage" -> "local"];
			addValueMap[data, inst["target"]["id"], valId];
		];
		AssertThat["valId should be an integer", valId]["named", "valId"]["satisfies", IntegerQ];
		If[!isVoidType[inst["target"]["type"]],
			setValueName[valId, inst["target"]]
		];
		setInstDebugLocation[data, valId, inst];
		valId
	]

isThrowCallQ[funName_] :=
    funName === "_Native`ThrowWolframException_Integer32_Void" ||
    funName === Native`PrimitiveFunction["throwWolframException"];

genLoadArgumentInstruction[data_, inst_] :=
	With[{
		argumentIndex = inst["index"]["data"],
		funId = data["functionId"]["get"]
	},
	With[{
		valId = LLVMLibraryFunction["LLVMGetParam"][
			funId,
			argumentIndex-1
		]
	},
		setValueName[valId, inst["target"]];
		addValueMap[
			data,
			inst["target"]["id"],
			valId
		]
	]]
	
genLoadInstruction[data_, inst_] :=
	With[{
		source = inst["source"],
		target = inst["target"]
	},
	With[{
		valId = LLVMLibraryFunction["LLVMBuildLoad"][
			data["builderId"],
			getId[data, source],
			getName[target]
		]
	},
		setValueName[valId, inst["target"]];
		addValueMap[
			data,
			target["id"],
			valId
		]
	]]


(*
  At one time we bothered if the target was a PackedArray and tried to avoid a copy.
  This is not needed anymore.
*) 	
genCopyInstruction[data_, inst_] :=
	With[{
		source = inst["source"],
		target = inst["target"]
	},
		addValueMap[
			data,
			target["id"],
			getId[data, source]
		]
	]


genGetElementStructure[data_, target_, src_, {fieldIndex_}] :=
	Module[{fieldVal, valId},
		fieldVal = getIndexValue[data, src, fieldIndex];
		valId = AddExtractValue[data, 
					getId[data, src],
					fieldVal
		];
		setValueName[valId, target];
		addValueMap[ data, target["id"], valId]
	]



genGetElementStructureHandle[data_, target_, src_, {fieldIndex_}] :=
	Module[{fieldVal, valId},
		fieldVal = getIndexValue[data, src, fieldIndex];
		valId = AddGetStructureElement[data, 
					getId[data, src],
					fieldVal
		];
		setValueName[valId, target];
		addValueMap[ data, target["id"], valId]
	]


genGetElementInstruction[data_, inst_] :=
	With[{

		target = inst["target"],
		src = inst["source"],
		offsets = inst["offset"]
	},
	With[ {
		ty = src["type"]
	},
		Which[
			isStructureType[data, ty],
				genGetElementStructure[data, target, src, offsets]
			,
			isHandleType[data, ty] && isStructureType[data, First[ty["arguments"]]],
				genGetElementStructureHandle[data, target, src, offsets]
			,
			True,
				With[{
					valId = AddGetElement[data, 
						getId[data, src],
						Map[ getId[data,#]&, offsets]
					]
				},
					setValueName[valId, target];
					addValueMap[
						data,
						target["id"],
						valId
					]
				]
		]
	]]

getIndexValue[data_, var_?ConstantValueQ, val_?ConstantValueQ] :=
	getIndexValue[data, var["type"], val]
getIndexValue[data_, var_?VariableQ, val_?ConstantValueQ] :=
	getIndexValue[data, var["type"], val]
	
getIndexValue[data_, ty_?TypeObjectQ, val_?ConstantValueQ] :=
	Which[
		IntegerQ[val["value"]],
			val["value"],
		StringQ[val["value"]],
			If[isHandleType[data, ty],
				Return[getIndexValue[data, First[ty["arguments"]], val]]
			];
			Module[ {
				metadata = ty["type"]["getProperty", "metadata", Null],
				fields
			},
				If[metadata === Null,
					ThrowException[{"Unable to find metadata for type structure ", ty}]
				];
				If[!KeyExistsQ[metadata, "Fields"],
					ThrowException[{"Unable to find metadata fields for type structure ", ty}]
				];
				fields = metadata["Fields"];
				If[!KeyExistsQ[fields, val["value"]],
					ThrowException[{"Unable to find field index ", val["value"], " in fields ", fields, " for type structure ", ty}]
				];
				fields[val["value"]] - 1
			],
		True,
			ThrowException[{"Unhandled type for field value", val}]
	]


getIndexValue[data_, ty_, val_?ConstantValueQ] :=
	With[{
		tyEnv = data["typeEnvironment"]
	},
		getIndexValue[data, tyEnv["resolve", ty], val]
	]
getIndexValue[args___] :=
	ThrowException[{"Unhandled type for field index", args}]


genSetElementStructureHandle[data_, target_, {fieldOffset_}, src_] :=
	Module[{fieldVal, valId},
		fieldVal = getIndexValue[data, target, fieldOffset];
		valId = AddSetStructureElement[data, 
			getId[data, target],
			fieldVal,
			getId[data, src]
		];
		valId
	]

genSetElementInstruction[data_, inst_] :=
	With[{
		target = inst["target"],
		offsets = inst["offset"],
		src = inst["source"]
	},
	With[ {
		ty = target["type"]
	},
		Which[
			isStructureType[data, ty],
				ThrowException[{"Cannot call SetElementInstruction on a structure."}]
			,
			isHandleType[data, ty] && isStructureType[data, First[ty["arguments"]]],
				genSetElementStructureHandle[data, target, offsets, src]
			,
			True,
				AddSetElement[data, 
					getId[data, target],
					Map[ getId[data,#]&, offsets],
					getId[data, src]
				]
		]
	]]


genLandingPadInstruction[data_, inst_] := 
	Module[{valId},
		If[!data["landingPadInitialized"]["get"],
			InitializeLandingPad[data];
			data["landingPadInitialized"]["set", True]
		];
		valId = AddLandingPad[data];
		If[ valId =!= 0,
			setValueName[valId, inst["target"]]];
		addValueMap[data, inst["target"]["id"], valId];
		valId
	]

genResumeInstruction[data_, inst_] :=
	With[{
		valId = AddResume[data, getId[data, inst["value"]]]
	},
		valId
	]


genReturnInstruction[data_, inst_] := Module[{llvmInstRef},
	llvmInstRef = If[inst["hasValue"] && !data["functionReturnVoid"],
		LLVMLibraryFunction["LLVMBuildRet"][
			data["builderId"],
			getId[data, inst["value"]]
		],
		LLVMLibraryFunction["LLVMBuildRetVoid"][
			data["builderId"]
		]
	];
	setInstDebugLocation[data, llvmInstRef, inst];
	llvmInstRef
];


removeFirstReference[ty_] :=
	First[ty["arguments"]]




genBuildArray[ data_, target_, size_] :=
    With[{
        ty = genType[
            data,
            removeFirstReference[target["type"]]
        ]
    },
	With[{
		valId = If[ConstantValueQ[size] && size["value"] === 1,
		    LLVMLibraryFunction["LLVMBuildAlloca"][
	            data["builderId"],
	            ty,
	            getName[target]
		    ],
            LLVMLibraryFunction["LLVMBuildArrayAlloca"][
	            data["builderId"],
	            ty,
                getId[data,size],
	            getName[target]
            ]
		]
	},
		setValueName[valId, target];
		addValueMap[
			data,
			target["id"],
			valId
		]
	]]


genBuildStructure[data_, target_] :=
	With[{
		valId = LLVMLibraryFunction["LLVMBuildAlloca"][
			data["builderId"],
			genType[
				data,
				target["type"]
			],
			getName[target]
		]
	},
		setValueName[valId, target];
		addValueMap[
			data,
			target["id"],
			valId
		]
	]	

genStackAllocateInstruction[data_, inst_] :=
	With[{
		target = inst["target"],
		size = inst["size"]
	},
	With[ {
		ty = target["type"]
	},
		Which[
			isStructureType[data, ty],
				genBuildStructure[data, target]
			,
			ty["isNamedApplication", "Handle"] || ty["isNamedApplication", "CArray"] || ty["isNamedApplication", "MIterator"],
				genBuildArray[data, target, size]
			,
			True,
				ThrowException[{"Unhandled StackAllocateInstruction ", inst}]
		]
	]]


genStoreInstruction[data_, inst_] :=
	Module[{value = inst["value"], target = inst["target"], valId},
		valId = LLVMLibraryFunction["LLVMBuildStore"][
			data["builderId"],
			getId[data,value],
			getId[data,target]
		];
		setInstDebugLocation[data, valId, inst];
		valId
	];




setInstDebugLocation[data_, llvmInstId_Integer, inst_?InstructionQ] := Module[{mexpr = None, scopingMExpr,
                                                                           diScopeId, line, column = 1},
	(* TODO: Assuming that if diBuilderId is IntegerQ, then LLVMDebug is True *)
	If[!IntegerQ[data["diBuilderId"]],
		Return[];
	];

	mexpr = Which[
		MExprQ[inst["mexpr"]],
			inst["mexpr"],
		True,
			Return[]
	];

	scopingMExpr = parentScopingMExpr[mexpr];
	If[!MExprQ[scopingMExpr],
		ThrowException[{"setInstDebugLocation: scopingMExpr should not be ", scopingMExpr, " mexpr: ", mexpr}]
	];

	line = Which[
		mexpr["hasProperty", "sourceLine"],
		    mexpr["getProperty", "sourceLine"],
		MExprSymbolQ[mexpr] || MExprLiteralQ[mexpr],
		    AssertThat["mexpr parent should exist", mexpr["parent"]]["named", "mexpr[parent]"]["satisfies", MExprQ];
		    mexpr["parent"]["getProperty", "sourceLine", 1],
		True,
			1
	];
	AssertThat["line is IntegerQ", line]["named", "line"]["satisfies", IntegerQ];

	diScopeId = genScope[data, scopingMExpr];
	LLVMLibraryFunction["LLVMWLIRBuilder_SetCurrentDebugLocation"][data["builderId"], line, column, diScopeId];
	LLVMLibraryFunction["LLVMWLIRBuilder_SetInstDebugLocation"][data["builderId"], llvmInstId];
	LLVMLibraryFunction["LLVMSetCurrentDebugLocation"][data["builderId"], 0];
];

setInstDebugLocation[data_, args___] := 
	ThrowException[{"Bad arguments for setInstDebugLocation: ", args}]

parentScopingMExpr := Compile`Core`Debug`InsertDebugDeclarePass`Private`parentScopingMExpr;

genScope[data_, scopingMExpr_?MExprQ] := Module[{scopeCache},
	(* Maps mexpr id's to DISubprogram* *)
	scopeCache = data["scopeCache"];
	If[scopeCache["keyExistsQ", scopingMExpr["id"]],
		Return[scopeCache["lookup", scopingMExpr["id"]]],
	  (* else *)
		Module[{parentMExpr, parentDiScopeId, diFileRef, lineNo},
			parentMExpr = parentScopingMExpr[scopingMExpr];

			parentDiScopeId = Which[
				MExprQ[parentMExpr],
					genScope[data, parentMExpr],
				parentMExpr === Null,
					genScope[data, Null],
				True,
					ThrowException[{parentMExpr, " is not MExprQ"}]
			];
			AssertThat["parentDiScopeId should be an integer", parentDiScopeId]["named", "parentDiScopeId"]["satisfies", IntegerQ];

			diFileRef = data["sourceDIFileRef"];
			AssertThat["diFileRef should be an integer", diFileRef]["named", "diFileRef"]["satisfies", IntegerQ];

			lineNo = scopingMExpr["getProperty", "sourceLine", 1];

			(* Update this as support for With/Block/other scoping constructs is added *)
			Switch[
				scopingMExpr["getHead"],
				Function,
					Module[{name, linkageName, diSubroutineType, isLocalToUnit, isDefinition, scopeLine, diSubprogram},
						(*name = scopingMExpr["getName"];*)
						name = data["functionName"];
						linkageName = "";
						diSubroutineType = 0;
						isLocalToUnit = True;
						isDefinition = True;
						scopeLine = lineNo;

						(*Print["      genScope Function parent: ", "(",parentMExpr["id"],") ", BaseForm[parentDiScopeId, 16]];*)
						diSubprogram = LLVMLibraryFunction["LLVMWLDIBuilder_createFunction"][data["diBuilderId"],
							parentDiScopeId, name, linkageName, diFileRef, lineNo, diSubroutineType,
							isLocalToUnit, isDefinition, scopeLine
						];
						(*Print["                  new function: ", BaseForm[diSubprogram, 16]];*)

						(* TODO: This will be incorrect for anonymous/local functions *)
						LLVMLibraryFunction["LLVMWLFunction_setSubprogram"][data["functionId"]["get"], diSubprogram];

						scopeCache["associateTo", scopingMExpr["id"] -> diSubprogram];
						Return[diSubprogram];
					],
				Module,
					Module[{colNo, diLexicalBlock},
						colNo = 1;

						(*Print["      genScope Module parent: ", "(",parentMExpr["id"],") ", BaseForm[parentDiScopeId, 16]];*)
						diLexicalBlock = LLVMLibraryFunction["LLVMWLDIBuilder_createLexicalBlock"][data["diBuilderId"],
							parentDiScopeId, diFileRef, lineNo, colNo
						];
						(*Print["                    new module: ", BaseForm[newDiScopeId, 16]];*)

						scopeCache["associateTo", scopingMExpr["id"] -> diLexicalBlock];
						Return[diLexicalBlock];
					],
				_,
					ThrowException[{"Cannot generate LLVM scope for expr with head: ", scopingMExpr["getHead"]}]
			]
		]
	];
];
genScope[data_, Null] := Module[{langId, producer, fileRef, isOptimized, cmdLineFlags, runtimeVersion, compileUnit},
	If[IntegerQ[data["diCompileUnitId"]["get"]],
		Return[data["diCompileUnitId"]["get"]];
	];
	langId = 2; (* Arbitrary; Rust pretends to be C++, not sure yet of the importance for us *)
	fileRef = data["sourceDIFileRef"];
	AssertThat["fileRef should be an integer", fileRef]["named", "fileRef"]["satisfies", IntegerQ];
	producer = "Wolfram Language Compiler <Compiler Version " <> ToString[Compiler`Internal`$Version] <> " >";
	isOptimized = False;
	cmdLineFlags = "";
	runtimeVersion = 1; (* TODO: Make use of this at some point? Similar to Obj-C runtime version for example *)
	compileUnit = LLVMLibraryFunction["LLVMWLDIBuilder_createCompileUnit"][data["diBuilderId"],
		langId, fileRef, producer, isOptimized, cmdLineFlags, runtimeVersion];
	AssertThat["compileUnit should be an integer", compileUnit]["named", "compileUnit"]["satisfies", IntegerQ];
	(*Print["diCompileUnitId: ", BaseForm[compileUnit, 16]];*)
	data["diCompileUnitId"]["set", compileUnit];
	compileUnit
];
genScope[args___] :=
	ThrowException[{"Invalid arguments to genScope ", Rest[{args}]}]

addValueMap[ data_, insID_, valID_] :=
	Module[{},
		If[!IntegerQ[valID],
			ThrowException[CompilerException[{"value Id is not an integer", valID}]]
		];
		data["valueMap"]["associateTo", insID -> valID];
		valID
	]


(*
  basicBlockMap is a map from the WIR BB Id to the LLVMIR BB Id.
  We need to store the original LLVMIR BB Id,  used for branching,
  and any updates to the Id caused by introducing a new BB, used for
  Phi instructions.
*)

addBasicBlockMap[ data_, insID_, bbID_] :=
	data["basicBlockMap"]["associateTo", insID -> {bbID, bbID}];

updateBasicBlockMap[data_, bbID_] :=
	Module[ {currBBId, val},
		currBBId = data["currentW_BBId"]["get"];
		val = data["basicBlockMap"]["lookup", currBBId, Null];
		If[ val === Null,
			ThrowException[CompilerException[{"updateBasicBlockMap cannot find basic block ", currBBId}]]
		];
		val = {First[val], bbID};
		data["basicBlockMap"]["associateTo", currBBId -> val];
	]


getCurrentBasicBlock[ data_] :=
	Module[ {currBBId, val},
		currBBId = data["currentW_BBId"]["get"];
		val = data["basicBlockMap"]["lookup", currBBId, Null];
		If[ val === Null,
			ThrowException[CompilerException["getCurrentBasicBlock cannot find basic block ", currBBId]]
		];
		Last[val]
	]

sanitizeASM[s_] :=
	"#" <> StringReplace[s, {"$" -> "", "\n" -> "\n\t#"}]

ClearAll[initializeFunction];
initializeFunction[data_, fm_?FunctionModuleQ] :=
	With[ {
		funTy = fm["type"]
	},
	If[ !TypeArrowQ[funTy],
		ThrowException[CompilerException[{"Type is not a function type", funTy, fm["name"]}]]];

	With[{
		name = getName[fm],
		tyId = genType[data, funTy],
		unresTy = funTy["unresolve"],
		exported = TrueQ[fm["getProperty", "exported", False]]
	},
	With[{
	    funId = LLVMCreateFunction[data, name, tyId, unresTy,
		    "LLVMLinkage" -> If[exported, "LLVMExternalLinkage", "LLVMPrivateLinkage"],
		    (*
				private linkage requires default visibility
				bug 367178
		    *)
            "LLVMVisibility" -> If[exported, "LLVMDefaultVisibility", "LLVMDefaultVisibility"],
		    "LLVMUnnamedAddress" -> If[exported, "LLVMNoUnnamedAddr", "LLVMGlobalUnnamedAddr"],
		    "LLVMDLLStorageClass" -> If[exported, "LLVMDLLExportStorageClass", "LLVMDefaultStorageClass"],
		    "LLVMInline" -> "Hint"
		]
	},
		If[!IntegerQ[funId],
			ThrowException[CompilerException[{"function Id is not an integer", funId}]]
		];
		addFunctionInformationAttributes[data, funId, fm["information"]];
		(* This is needed to create the wrapper function for the main function. *)
		fm["setProperty", "typeSignature" -> unresTy];
		data["functionMap"]["associateTo", name -> funId];
	]]]


addFunctionInformationAttributes[data_, funId_, info_?FunctionInformationQ] := 
    (
        If[TrueQ[info["Throws"]],
            LLVMAddFunctionAttribute[data, funId, "uwtable"],
            LLVMAddFunctionAttribute[data, funId, "nounwind"] 
        ];
        
        If[info["inlineInformation"]["inlineValue"] === "Always",
            LLVMAddFunctionAttribute[data, funId, "alwaysinline"]
        ];
    )


cnt = 0;
defaultFilePath := defaultFilePath = "WMCompiler " <> $Version <> ".wl"

initializeModule[pm_, opts_] :=
	Module[ {name, fileName, data, llvmOpts, loopHints, machArch},
		cnt++;
		name = "WolframCompiledFunction$" <> ToString[cnt];
		fileName = Which[
			pm["hasProperty", "sourceFilePath"],
				pm["getProperty", "sourceFilePath"],
			pm["mexpr"] =!= None && pm["mexpr"]["hasProperty", "sourceFilePath"],
				pm["mexpr"]["getProperty", "sourceFilePath"],
			True,
				defaultFilePath
		];
		llvmOpts = Lookup[opts, "LLVMOptimization", None];
		machArch = Lookup[opts, "MachineArchitecture", Automatic];
		loopHints = processLoopHints[llvmOpts, Lookup[opts, "LoopHints", None]];
		data = CreateModule[name, Append[opts, "sourceFilePath" -> fileName]];
		AssociateTo[data, {
			"LLVMOptimization" -> llvmOpts,
			"LoopHints" -> loopHints,			
			"MachineArchitecture" -> machArch,			
			"dummyFile" -> CreateReference[],
			"diCompileUnitId" -> CreateReference[],
			"scopeCache" -> CreateReference[<||>],
			"valueMap" -> CreateReference[<||>],
			"basicBlockMap" -> CreateReference[<||>],
			"phiMap" -> CreateReference[<||>],
			"typeCache" -> CreateReference[<||>],
			"diTypeCache" -> CreateReference[<||>],
			"currentW_BBId" -> CreateReference[],
			"functionMap" -> CreateReference[<||>],
			"functionId" -> CreateReference[],
			"callFunctionType" -> CreateReference[],
			"initializationArgumentId" -> CreateReference[],
			"updateBasicBlockMap" -> updateBasicBlockMap,
			"getCurrentBasicBlock" -> getCurrentBasicBlock,
			"landingPadInitialized" -> CreateReference[False],
			"landingPadPersonality" -> CreateReference[],
			"typeEnvironment" -> pm["typeEnvironment"],
			"externalLibraryFunctionTypes"  -> CreateReference[<||>],
			"intrinsicFunctionIds"  -> CreateReference[<||>],
			"externalLibraries" -> pm["getProperty", "externalLibraries", {}],
			"getExternalFunctionType" -> Function[ {dataArg, nameArg}, getExternalFunctionType[dataArg, nameArg]]
		}]
	]

initializeDataForFunction[pm_, fm_, data_, opts_] :=
		(
		data["landingPadInitialized"]["set", False];
		Join[
			data,
			<|
				"valueMap" -> CreateReference[<||>],
				"basicBlockMap" -> CreateReference[<||>],
				"phiMap" -> CreateReference[<||>],
				"functionReturnVoid" -> fm["type"]["result"]["isConstructor", "Void"],
				"functionName" -> fm["name"],
				(* TODO: Remove this when setInstDebugLocation is fixed *)
				"functionMExpr" -> fm["mexpr"],
				"sourceDIFileRef" -> Module[{path = fm["getProperty", "sourceFilePath", defaultFilePath]},
					If[data["diBuilderId"] =!= None,
						LLVMLibraryFunction["LLVMWLDIBuilder_createFile"][data["diBuilderId"], Last[FileNameSplit[path]], DirectoryName[path]]
					]
				],
				"diVariableCache" -> CreateReference[<||>]
			|>]
			)


(*
 This is necessary because addMetadata might be called with a valRef which is not 
 an instruction.  The builders nearly always return an instruction,  which turns into 
 the target ID (typical of LLVM usage).  However,  some builders,  eg the BitCast returns 
 its argument if the types are equal.  If the argument is then eg a parameter or a 
 constant,  the call to addMetadata would not be an instruction.
*)
isInstructionQ[value_] :=
	LLVMLibraryFunction["LLVMIsAInstruction"][value] =!= 0
	
addMetadata[data_, valRefId_?isInstructionQ, key_?StringQ -> val0_] :=
	With[{
		val = ToString[val0]
	},
	With[{
		keyId = LLVMLibraryFunction["LLVMGetMDKindIDInContext"][data["contextId"], key, StringLength[key]],
		valId = LLVMLibraryFunction["LLVMMDStringInContext"][data["contextId"], val,  StringLength[val]]
	},
	With[{
		mdNode = WrapIntegerArray[LLVMLibraryFunction["LLVMMDNodeInContext"][data["contextId"], #, 1]&, {valId}]
	},
		LLVMLibraryFunction["LLVMSetMetadata"][valRefId, keyId, mdNode]
	]]]

(*
 Fall through when the argument is not an Instruction
*)
addMetadata[data_, _, key_?StringQ -> val0_] :=
    Null
    
addNamedMetadataOperand[ data_, key_?StringQ -> val_?StringQ] :=
	With[ {
		valId = LLVMLibraryFunction["LLVMMDStringInContext"][data["contextId"], val,  StringLength[val]]

	},
	With[ {
		mdNode = WrapIntegerArray[LLVMLibraryFunction["LLVMMDNodeInContext"][data["contextId"], #, 1]&, {valId}]
	},
		LLVMLibraryFunction["LLVMAddNamedMetadataOperand"][data["moduleId"], key, mdNode]
	]]

	
addInstructionMetadata[data_, inst_?InstructionQ][val_Integer] :=
	With[{
		str = toPlainString[inst]
	},
		addMetadata[data, val, "wir.instruction" -> str];
		val
	]
	
addInstructionMetadata[data_, bb_?BasicBlockQ][val_] :=
	val
addInstructionMetadata[data_, _][val_] := val

setDataForAPICodeGen[ data_] :=
	Prepend[ data, "updateBasicBlockMap" -> (Null&)]

setDataForLLVMPassCodeGen[ data_] :=
	Prepend[ data, "updateBasicBlockMap" -> updateBasicBlockMap]

run[pm_, opts_] :=
	Module[{
		data, funId, compilerName,
		addInstructionMetadataInDebug, compilerOptions, timeNow
	},
		data = initializeModule[pm, opts];
		addInstructionMetadataInDebug = 
			If[Lookup[opts, "LLVMDebug", False],
				Function[{visitor},
					Function[{st, inst},
						addInstructionMetadata[st, inst]@
						visitor[st, inst]
					]
				],
				Identity
			];

		If[ TrueQ[Lookup[opts, "CreateWrapper", True]],
			funId = InitializeInitialization[ data, "ProgramInitialization"];
			data["functionMap"]["associateTo", "ProgramInitialization" -> funId]];
		pm["scanFunctionModules", Function[{fm},
	        initializeFunction[ data, fm]]
	     ];
	    pm["scanFunctionModules", Function[{fm},
	    		data = initializeDataForFunction[pm, fm, data, opts];
	    		funId = data["functionMap"]["lookup", getName[fm], Null];
	    		If[ funId === Null,
	    			ThrowException[CompilerException[{"Cannot find function ID", fm["name"]}]]
	    		];
	    		data["functionId"]["set", funId];
	        	fm["topologicalOrderScan",
	        		Function[{bb},
	        			initBasicBlock[data, bb]
	        		]
	        	];
				CreateInstructionVisitor[
					data,
					addInstructionMetadataInDebug /@ <|
						"visitBasicBlock" -> genBasicBlock,
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
						"visitInvokeInstruction" -> genInvokeInstruction,
						"visitLambdaInstruction" -> genLambdaInstruction,
						"visitLandingPadInstruction" -> genLandingPadInstruction,
						"visitPhiInstruction" -> genPhiInstruction,
						"visitTypeCastInstruction" -> genTypeCastInstruction,
						"visitLoadGlobalInstruction" -> genLoadGlobalInstruction,
						"visitResumeInstruction" -> genResumeInstruction,
						"visitSelectInstruction" -> genSelectInstruction,
						"" -> Null
					|>,
					fm,
					"IgnoreRequiredInstructions" -> True
				];
				processPhi[data];
	    ]];
		If[KeyExistsQ[data, "diBuilderId"],
			If[data["diBuilderId"] =!= None,
				LLVMLibraryFunction["LLVMWLDIBuilder_finalize"][data["diBuilderId"]];
			]
		];
		
		compilerName = "Wolfram Compiler " <> ToString[Compiler`Internal`$Version] <> "";
		addNamedMetadataOperand[data, "llvm.ident" -> compilerName];
		
		addNamedMetadataOperand[data, "wolfram.version" -> $Version];
		
		timeNow = DateString[];
		addNamedMetadataOperand[data, "wolfram.build_time" -> timeNow];
		
		compilerOptions = ToString[opts];
		addNamedMetadataOperand[data, "wolfram.options" -> compilerOptions];
		
		If[pm["hasProperty", "mexpr"] =!= None && pm["mexpr"] =!= None,
			Block[{$FrontEnd = Null},
			Module[{
				prog = pm["mexpr"]["toString"]
			},
				addNamedMetadataOperand[data, "wolfram.expr" -> prog];
			]]
		];

		
		FinalizeModule[data];
		pm["setProperty", "LLVMLinkData" -> data];
		(* TODO: Perhaps do something like:
		   pm["setProperty", "LLVMLinkData" -> Association[FilterRules[data, {"moduleId"}]]];
		*)
		pm
	]


(*
  This should NOT call the GenerateWrapper function.
*)


RegisterCallback["RegisterPass", Function[{st},
CreateLLVMIRPreprocessPass = CreateProgramModulePass[<|
	"information" -> 
		CreatePassInformation[
			"CreateLLVMIRPreprocess",
			"The pass runs all the passes used by LLVM IR generation, but does not actually generate LLVMIR."],
	"runPass" -> Function[{pm, opts}, pm],
	"requires" -> {
		AbortHandlingPass,
		DropNoTypeFunctionPass,
		MemoryManagePass,
		ExceptionHandlingPass,
		MTensorMemoryFinalizePass,
		MObjectCollectPass,
		ExpressionRefCountPass,
		InferencePass, 
		ResolveClosurePass,
		GenerateWrapperPass,
		ConstantPhiCopyPass
	}
|>];
RegisterPass[CreateLLVMIRPreprocessPass]
]]

RegisterCallback["RegisterPass", Function[{st},
CreateLLVMIRPass = CreateProgramModulePass[<|
	"information" -> 
		CreatePassInformation[
			"CreateLLVMIR",
			"The pass creates LLVM IR for the functions in a module."],
	"runPass" -> run,
	"requires" -> {
		AbortHandlingPass,
		DropNoTypeFunctionPass,
		MemoryManagePass,
		ExceptionHandlingPass,
		MTensorMemoryFinalizePass,
		MObjectCollectPass,
		ExpressionRefCountPass,
		InferencePass, 
		ResolveClosurePass,
		GenerateWrapperPass,
		ConstantPhiCopyPass,
		LoopNestingForestPass
	}

|>];
RegisterPass[CreateLLVMIRPass]
]]

RegisterCallback["RegisterPass", Function[{st},
CreateLLVMIROnlyPass = CreateProgramModulePass[<|
	"information" -> 
		CreatePassInformation[
			"CreateLLVMIROnly",
			"The pass creates LLVM IR for the functions in a module, but does not run any pre-passes."],
	"runPass" -> run,
	"requires" -> {DropNoTypeFunctionPass, GenerateWrapperPass, ConstantPhiCopyPass}
|>];
RegisterPass[CreateLLVMIROnlyPass]
]]

End[]
EndPackage[]
