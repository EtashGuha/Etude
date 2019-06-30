

BeginPackage["LLVMCompileTools`Intrinsics`"]

AddLLVMExpectIntrinsic

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]
Needs["LLVMCompileTools`Basic`"]

(* Arithmetic Overflow functions *)
AddCodeFunction["SignedPlusWithOverflowIntrinsic", arithmeticIntrinsic["llvm.sadd.with.overflow", getArithmeticOverFlowReturnType]]
AddCodeFunction["UnsignedPlusWithOverflowIntrinsic", arithmeticIntrinsic["llvm.uadd.with.overflow", getArithmeticOverFlowReturnType]]

AddCodeFunction["SignedSubtractWithOverflowIntrinsic", arithmeticIntrinsic["llvm.ssub.with.overflow", getArithmeticOverFlowReturnType]]
AddCodeFunction["UnsignedSubtractWithOverflowIntrinsic", arithmeticIntrinsic["llvm.usub.with.overflow", getArithmeticOverFlowReturnType]]

AddCodeFunction["SignedTimesWithOverflowIntrinsic", arithmeticIntrinsic["llvm.smul.with.overflow", getArithmeticOverFlowReturnType]]
AddCodeFunction["UnsignedTimesWithOverflowIntrinsic", arithmeticIntrinsic["llvm.umul.with.overflow", getArithmeticOverFlowReturnType]]


(* Elementary functions *)
AddCodeFunction["UncheckedCos", arithmeticIntrinsic["llvm.cos", getReturnType]]
AddCodeFunction["UncheckedSin", arithmeticIntrinsic["llvm.sin", getReturnType]]
AddCodeFunction["UncheckedPowerFI", arithmeticIntrinsic["llvm.powi", getReturnType]]
AddCodeFunction["UncheckedPower", arithmeticIntrinsic["llvm.pow", getReturnType]]
AddCodeFunction["UncheckedExp", arithmeticIntrinsic["llvm.exp", getReturnType]]
AddCodeFunction["UncheckedSqrt", arithmeticIntrinsic["llvm.sqrt", getReturnType]]
AddCodeFunction["UncheckedExp2", arithmeticIntrinsic["llvm.exp2", getReturnType]]
AddCodeFunction["UncheckedLog", arithmeticIntrinsic["llvm.log", getReturnType]]
AddCodeFunction["UncheckedLog10", arithmeticIntrinsic["llvm.log10", getReturnType]]
AddCodeFunction["UncheckedLog2", arithmeticIntrinsic["llvm.log2", getReturnType]]


(* Utility functions *)
AddCodeFunction["UncheckedFMA", arithmeticIntrinsic["llvm.fma", getReturnType]]
AddCodeFunction["UncheckedAbs", arithmeticIntrinsic["llvm.fabs", getReturnType]]
AddCodeFunction["UncheckedMin", arithmeticIntrinsic["llvm.minimum", getReturnType]]
AddCodeFunction["UncheckedMax", arithmeticIntrinsic["llvm.maximum", getReturnType]]
AddCodeFunction["UncheckedFloor", arithmeticIntrinsic["llvm.floor", getReturnType, castResultToInteger]]
AddCodeFunction["UncheckedCeiling", arithmeticIntrinsic["llvm.ceil", getReturnType, castResultToInteger]]
AddCodeFunction["UncheckedBitLength", arithmeticIntrinsic["llvm.ctpop", getReturnType]]

(* Timer functions *)
AddCodeFunction["ReadCycleCounter", AddLLVMReadCycleCounter]

AddLLVMReadCycleCounter := AddLLVMReadCycleCounter = 
    arithmeticIntrinsic["llvm.readcyclecounter", Function[{data, arg}, GetIntegerType[data, 64]]];
    

(* Expect functions *)
AddCodeFunction["Expect", AddLLVMExpectIntrinsic]

AddLLVMExpectIntrinsic := AddLLVMExpectIntrinsic = expectIntrinsic["llvm.expect"]

expectIntrinsic[baseName_][data_?AssociationQ, _, {val_, expected_}] :=
    Module[ {funName, funId, argTy, resTy, funTy, id},
        funName = getIntrinsicName[data["callFunctionType"]["get"], baseName];
        funId = data["functionCache"]["lookup", funName, Null];
        If[funId === Null,
            argTy = LLVMLibraryFunction["LLVMTypeOf"][val];
            resTy = argTy;
            funTy = WrapIntegerArray[
                LLVMLibraryFunction["LLVMFunctionType"][resTy, #, 2, 0]&,
                {argTy,argTy}
            ];
            funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], funName, funTy];
            data["functionCache"]["associateTo", funName -> funId]
        ];
        id = WrapIntegerArray[
            LLVMLibraryFunction["LLVMBuildCall"][
                data["builderId"], 
                funId,
                #,
                2,
                ""
            ]&,
            {val, expected}
        ];
        id
    ] 
    

(* Utilities *)

$intSizes = <|
	"Integer8" -> "8",
	"Integer16" -> "16",
	"Integer32" -> "32",
	"Integer64" -> "64",
	"UnsignedInteger8" -> "8",
	"UnsignedInteger16" -> "16",
	"UnsignedInteger32" -> "32",
	"UnsignedInteger64" -> "64"
|>

$realSizes = <|
	"Real16" -> "16",
	"Real32" -> "32",
	"Real64" -> "64",
	"Real128" -> "128"
|>

firstArgumentType[s_String] := s
argumentType[TypeSpecifier[args_ -> _]] := args
firstArgumentType[TypeSpecifier[args_ -> _]] := If[args === {}, "Void", First[args]]
resultType[s_String] := s
resultType[TypeSpecifier[_ -> res_]] := res
isVoidFunctionType[ty_] := argumentType[ty] === {}
isBooleanFunctionType[ty_] := firstArgumentType[ty] === "Boolean"
isIntegerFunctionType[ty_] :=
	MemberQ[Keys[$intSizes], firstArgumentType[ty]]
isIntegerFunctionResultType[ty_] :=
	MemberQ[Keys[$intSizes], resultType[ty]]
isRealFunctionType[ty_] :=
	MemberQ[Keys[$realSizes], firstArgumentType[ty]]
isRealFunctionResultType[ty_] :=
	MemberQ[Keys[$realSizes], resultType[ty]]

getArithmeticOverFlowReturnType[data_, argTy_] :=
	GetStructureType[data, {argTy, GetBooleanType[data]}]
	
getReturnType[data_, argTy_] :=
	argTy

arithmeticIntrinsic[baseName_, getReturnType_, postProcess_:noOpPostProcess][data_?AssociationQ, _, args_List] :=
	Module[ {funName, argTy, argsTy, arg1, resTy, funTy, funId, id},
		arg1 = If[Length[args] === 0, "Void", First[args]];
		funName = getIntrinsicName[data["callFunctionType"]["get"], baseName];
		funId = data["functionCache"]["lookup", funName, Null];
		If[funId === Null,
		    If[arg1 === "Void",
			    argTy = "Void";
			    argsTy = {};
			    , (* Else *)
			    argTy = LLVMLibraryFunction["LLVMTypeOf"][arg1];
                argsTy = Map[ LLVMLibraryFunction["LLVMTypeOf"][#]&, args];
			];
			resTy = getReturnType[data, argTy];
			funTy = WrapIntegerArray[
				LLVMLibraryFunction["LLVMFunctionType"][resTy, #, Length[argsTy], 0]&,
				argsTy
			];
			funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], funName, funTy];
			data["functionCache"]["associateTo", funName -> funId]
		];
		id = WrapIntegerArray[
			LLVMLibraryFunction["LLVMBuildCall"][
				data["builderId"], 
				funId,
				#,
				Length[args],
				""
			]&,
			args
		];
		postProcess[data, args, id]
	]
	
noOpPostProcess[data_, args_, id_] := id
castResultToInteger[data_, args_, id_] := 
	With[{
		fun = data["callFunctionType"]["get"]
	},
	With[{
		idTy = LLVMLibraryFunction["LLVMTypeOf"][id],
		resultTy = resultType[fun]
	},
		Which[
			LLVMLibraryFunction["LLVMGetTypeKind"][idTy] === LLVMEnumeration["LLVMTypeKind", "LLVMIntegerTypeKind"],
				id,
			StringStartsQ[resultTy, "Unsigned"],
				LLVMLibraryFunction["LLVMBuildFPToUI"][
					data["builderId"],
					id,
					GetIntegerType[data, ToExpression[$intSizes[resultTy]]],
					""
				],
			True,
				LLVMLibraryFunction["LLVMBuildFPToSI"][
					data["builderId"],
					id,
					GetIntegerType[data, ToExpression[$intSizes[resultTy]]],
					""
				]
		]
	]];
	
getIntrinsicName[ty_?isVoidFunctionType, baseName_] :=
    baseName
    
getIntrinsicName[ty_?isBooleanFunctionType, baseName_] :=
    baseName <> ".i1"
    
getIntrinsicName[ty_?isIntegerFunctionType, baseName_] :=
	Module[ {argTy, ext},
		argTy = firstArgumentType[ty];
		ext = Lookup[ $intSizes, argTy];
		If[MissingQ[ext],
			ThrowException[{"Integer intrinsic size not found:", argTy}]
		];
		baseName <> ".i" <> ext
	]

getIntrinsicName[ty_?isRealFunctionType, baseName_] :=
	Module[ {argTy, ext},
		argTy = firstArgumentType[ty];
		ext = Lookup[ $realSizes, argTy];
		If[MissingQ[ext],
			ThrowException[{"Real intrinsic size not found:", argTy}]
		];
		baseName <> ".f" <> ext
	]
	
getIntrinsicName[args___] :=
    ThrowException[{"Unrecognized call to getIntrinsicName when creating intrinsics ", {args}}];


(* Memory functions *)
AddCodeFunction["memcpyIntrinsicAligned32", memcpyIntrinsicAligned["llvm.memcpy.p0i8.p0i8.i32"]]
AddCodeFunction["memcpyIntrinsicAligned64", memcpyIntrinsicAligned["llvm.memcpy.p0i8.p0i8.i64"]]

memcpyIntrinsicAligned[funName_][data_?AssociationQ, _, {dest_, src_, len_}] :=
	Module[ {funId, funTyPointer, funTy, align, volatile, args, id},
		funId = data["functionCache"]["lookup", funName, Null];
		If[funId === Null,
			funTyPointer = GetLLVMType[data, {"VoidHandle", "VoidHandle", "MachineInteger", "Integer32", "Boolean"} -> "Void"];
			funTy = LLVMLibraryFunction["LLVMGetElementType"][ funTyPointer];
			funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], funName, funTy];
			data["functionCache"]["associateTo", funName -> funId]
		];
		align = AddConstantInteger[data, 32, 0];
		volatile = AddConstantBoolean[data, False];
		args = {dest, src, len, align, volatile};
		id = WrapIntegerArray[
			LLVMLibraryFunction["LLVMBuildCall"][
				data["builderId"], 
				funId,
				#,
				Length[args],
				""
			]&,
			args
		];
		id
	] 
	
	
(* Not an intrinsic,  but fits here *)
AddCodeFunction["AddressShift", AddressShift]

AddressShift[data_?AssociationQ, _, {src_, index_}] :=
	Module[ {id},
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], src, #, 1, ""]&, {index}];
        id
	]

End[]


EndPackage[]

