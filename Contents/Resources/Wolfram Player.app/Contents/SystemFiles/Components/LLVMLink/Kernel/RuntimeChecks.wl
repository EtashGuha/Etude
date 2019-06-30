
BeginPackage["LLVMLink`RuntimeChecks`"]


EnableLLVMRuntimeChecks


Begin["`Private`"]

Needs["LLVMLink`"]


(*
The intention with this package is to reproduce any asserts that may
happen inside of LLVM, and write them in WL

This has several benefits.

By running these checks first, then the assert in LLVM may be bypassed,
and abort() will not be called and bring down the whole program

These checks can provide more information than the LLVM asserts.
Names of values, if something is wrong, then what its value is, etc.

These checks may be run on a Release build of LLVM and still catch
the same problems (that would normally silently allow bad values without
these checks)

These checks can be controlled and do not have to be fatal
*)

EnableLLVMRuntimeChecks[] :=
Module[{res},
	res = InstallLLVM[];
	If[FailureQ[res],
		Assert[False || "InstallLLVM failed. LLVM Runtime checks NOT enabled."]
	];
	EnableBuildCallChecks[];
	EnableBuildStoreChecks[];
	EnableBuildICmpChecks[];
	EnableBuildTruncChecks[];
	EnableSetMetadataChecks[];
	EnableSetValueNameChecks[];
	EnableFunctionTypeChecks[];
	EnableSetVisibilityChecks[];
	Print["LLVM Runtime checks enabled."];
]




EnableBuildCallChecks[] := 
Module[{loadedBuildCall},

	(*
	Redefine LLVMLibraryFunction["LLVMBuildCall"] to always call checking
	 functions before doing the call
	*)

	Unset[LLVMLibraryFunction["LLVMBuildCall"]];

	(* copied from llvmc60.wl *)
	loadedBuildCall = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMBuildCall_Wrapper",
		{
			(* Type[LLVMBuilderRef -> struct LLVMOpaqueBuilder *] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[LLVMValueRef * -> struct LLVMOpaqueValue * *] *)
			Integer,
			(* Type[unsigned int] *)
			Integer,
			(* Type[const char *] *)
			"UTF8String"
		},
		(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
		Integer
	];

	LLVMLibraryFunction["LLVMBuildCall"] := (
		instructionsCPP299[##];
		loadedBuildCall[##]
	)&;
]

EnableBuildStoreChecks[] :=
Module[{loadedBuildStore},

	(*
	Redefine LLVMLibraryFunction["LLVMBuildStore"] to always call checking
	 functions before doing the call
	*)

	Unset[LLVMLibraryFunction["LLVMBuildStore"]];

	(* copied from llvmc60.wl *)
	loadedBuildStore = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMBuildStore_Wrapper",
		{
			(* Type[LLVMBuilderRef -> struct LLVMOpaqueBuilder *] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer
		},
		(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
		Integer
	];

	LLVMLibraryFunction["LLVMBuildStore"] := (
		instructionsCPP1400[##];
		loadedBuildStore[##]
	)&;
]

EnableBuildICmpChecks[] :=
Module[{loadedBuildICmp},

	(*
	Redefine LLVMLibraryFunction["LLVMBuildICmp"] to always call checking
	 functions before doing the call
	*)

	Unset[LLVMLibraryFunction["LLVMBuildICmp"]];

	(* copied from llvmc60.wl *)
	loadedBuildICmp = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMBuildICmp_Wrapper",
		{
			(* Type[LLVMBuilderRef -> struct LLVMOpaqueBuilder *] *)
			Integer,
			(* Type[LLVMIntPredicate -> enum LLVMIntPredicate] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[const char *] *)
			"UTF8String"
		},
		(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
		Integer
	];

	LLVMLibraryFunction["LLVMBuildICmp"] := (
		instructionsH1113[##];
		loadedBuildICmp[##]
	)&;
]

EnableBuildTruncChecks[] :=
Module[{loadedBuildTrunc},

	Unset[LLVMLibraryFunction["LLVMBuildTrunc"]];

	loadedBuildTrunc = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMBuildTrunc_Wrapper",
		{
			(* Type[LLVMBuilderRef -> struct LLVMOpaqueBuilder *] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[LLVMTypeRef -> struct LLVMOpaqueType *] *)
			Integer,
			(* Type[const char *] *)
			"UTF8String"
		},
		(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
		Integer
	];

	LLVMLibraryFunction["LLVMBuildTrunc"] := (
		instructionsCPP2565[LLVMEnumeration["LLVMOpcode", "LLVMTrunc"], #2, #3];
		loadedBuildTrunc[##]
	)&;
]

EnableSetMetadataChecks[] :=
Module[{loadedSetMetadata},

	Unset[LLVMLibraryFunction["LLVMSetMetadata"]];

	loadedSetMetadata = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMSetMetadata_Wrapper",
		{
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[unsigned int] *)
			Integer,
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer
		},
		(* Type[void] *)
		"Void"
	];

	LLVMLibraryFunction["LLVMSetMetadata"] := (
		castingH255[#1, LLVMEnumeration["LLVMValueKind", "LLVMInstructionValueKind"]];
		loadedSetMetadata[##]
	)&;
]

EnableFunctionTypeChecks[] := 
Module[{loadedFunctionType},
	(*
	Redefine LLVMLibraryFunction["LLVMFunctionType"] to always call checking
	 functions before doing the call
	*)

	Unset[LLVMLibraryFunction["LLVMFunctionType"]];

	(* copied from llvmc60.wl *)
	loadedFunctionType = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMFunctionType_Wrapper",
		{
			(* Type[LLVMTypeRef -> struct LLVMOpaqueType *] *)
			Integer,
			(* Type[LLVMTypeRef * -> struct LLVMOpaqueType * *] *)
			Integer,
			(* Type[unsigned int] *)
			Integer,
			(* Type[LLVMBool -> int] *)
			Integer
		},
		(* Type[LLVMTypeRef -> struct LLVMOpaqueType *] *)
		Integer
	];

	LLVMLibraryFunction["LLVMFunctionType"] := (
		typeCPP288[##];
		loadedFunctionType[##]
	)&;
]

EnableSetValueNameChecks[] := 
Module[{loadedSetValueName},

	(*
	Redefine LLVMLibraryFunction["LLVMSetValueName"] to always call checking
	 functions before doing the call
	*)

	Unset[LLVMLibraryFunction["LLVMSetValueName"]];

	(* copied from llvmc60.wl *)
	loadedSetValueName = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMSetValueName_Wrapper",
		{
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[const char *] *)
			"UTF8String"
		},
		(* Type[void] *)
		"Void"
	];

	LLVMLibraryFunction["LLVMSetValueName"] := (
		valueCPP247[##];
		loadedSetValueName[##]
	)&;
]

EnableSetVisibilityChecks[] := 
Module[{loadedSetVisibility},

	(*
	Redefine LLVMLibraryFunction["LLVMSetVisibility"] to always call checking
	 functions before doing the call
	*)

	Unset[LLVMLibraryFunction["LLVMSetVisibility"]];

	(* copied from llvmc60.wl *)
	loadedSetVisibility = LibraryFunctionLoad[LLVMLibraryName[],
		"LLVMLink_LLVMSetVisibility_Wrapper",
		{
			(* Type[LLVMValueRef -> struct LLVMOpaqueValue *] *)
			Integer,
			(* Type[LLVMVisibility -> enum LLVMVisibility] *)
			Integer
		},
		(* Type[void] *)
		"Void"
	];

	LLVMLibraryFunction["LLVMSetVisibility"] := (
		globalValueH233[##];
		loadedSetVisibility[##]
	)&;
]






instructionsCPP299[builder_, fn_, args_, numArgs_, name_] :=
Module[{fName, argsList, fTy, fParamCount, fParamTys, argTys},

	(*
		This reproduces the assertion that looks like this in LLVM 6.0.0:
		Assertion failed: (i >= FTy->getNumParams() || FTy->getParamType(i) ==
		 Args[i]->getType()) && "Calling a function with a bad signature!",
		  file e:\llvm-development\llvm600\llvm\lib\ir\instructions.cpp, line
		   299
	*)

	fName = LLVMLibraryFunction["LLVMGetValueName"][fn];

	argsList = LLVMLibraryFunction["LLVMLink_fromIntegerArray_Wrapper"][args, numArgs];

	fTy = LLVMLibraryFunction["LLVMTypeOf"][fn];
	fTy = LLVMLibraryFunction["LLVMGetElementType"][fTy];
	fParamCount = LLVMLibraryFunction["LLVMCountParamTypes"][fTy];
	fParamTys = getParameterTypes[fTy];

	argTys = (LLVMLibraryFunction["LLVMTypeOf"][#])& /@ argsList;

	With[{text = StringTemplate["Calling a function with a bad signature! Function: `function`, Expected arg count: `expected`, Actual arg count: `actual`"][<|
	"function" -> fName, "expected" -> fParamCount, "actual" -> Length[argTys]
	|>]},
	Assert[TrueQ[(Length[argTys] == fParamCount ||
		LLVMLibraryFunction["LLVMIsFunctionVarArg"][fTy] && Length[argTys] > fParamCount) || text]];
	];

	MapIndexed[
		Module[{i = #2[[1]]},
			strRef1 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][fParamTys[[i]]];
			str1 = LLVMLibraryFunction["setUTF8String"][strRef1];
			LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef1];
			
			strRef2 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][#];
			str2 = LLVMLibraryFunction["setUTF8String"][strRef2];
			LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef2];

			With[{text = StringTemplate["Calling a function with a bad signature! Function: `function`, i: `i`, Expected type: `expected`: Actual type: `actual`"][<|
			"function" -> fName, "i" -> i, "expected" -> str1, "actual" -> str2
			|>]},
			Assert[TrueQ[((i-1) >= fParamCount || fParamTys[[i]] == #) || text]];
			];
		]&
		,
		argTys
	]
]

instructionsCPP1400[builder_, val_, ptr_] :=
Module[{valTy, ptrTy, elementTy},

	(*
	Reproduces assertion that looks like this in LLVM 6.0.0:
	Assertion failed: getOperand(0)->getType() ==
	cast<PointerType>(getOperand(1)->getType())->getElementType() &&
	"Ptr must be a pointer to Val type!",
	file e:\llvm-development\llvm600\llvm\lib\ir\instructions.cpp, line 1400
	*)
	
	valTy = LLVMLibraryFunction["LLVMTypeOf"][val];
	strRef1 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][valTy];
	str1 = LLVMLibraryFunction["setUTF8String"][strRef1];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef1];

	ptrTy = LLVMLibraryFunction["LLVMTypeOf"][ptr];
	elementTy = LLVMLibraryFunction["LLVMGetElementType"][ptrTy];
	strRef2 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][elementTy];
	str2 = LLVMLibraryFunction["setUTF8String"][strRef2];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef2];

	With[{text = StringTemplate["Ptr must be a pointer to Val type! Value type: `value`, Pointer type: `pointer`"][<|
	"value" -> str1, "pointer" -> str2
	|>]},
	Assert[TrueQ[(valTy == elementTy) || text]];
	];
]

instructionsH1113[builder_, predicate_, a_, b_, name_] :=
Module[{aTy, bTy},

	(*
	Reproduces assertion that looks like this in LLVM 6.0.0:
	Assertion failed: getOperand(0)->getType() == getOperand(1)->getType() &&
	"Both operands to ICmp instruction are not of the same type!",
	file e:\llvm-development\llvm600\llvm\include\llvm\ir\instructions.h, line 1113
	*)
	
	aTy = LLVMLibraryFunction["LLVMTypeOf"][a];
	strRef1 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][aTy];
	str1 = LLVMLibraryFunction["setUTF8String"][strRef1];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef1];

	bTy = LLVMLibraryFunction["LLVMTypeOf"][b];
	strRef2 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][bTy];
	str2 = LLVMLibraryFunction["setUTF8String"][strRef2];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef2];

	With[{text = StringTemplate["Both operands to ICmp instruction are not of the same type! Operand 1: `a`, Operand 2: `b`"][<|
	"a" -> str1, "b" -> str2
	|>]},
	Assert[TrueQ[(aTy == bTy) || text]];
	];
]

instructionsCPP2565[op_, val_, ty_] :=
Module[{castIsValid},
	
	(*
	Reproduces assertion that looks like this in LLVM 6.0.0:
	Assertion failed: castIsValid(op, S, Ty) &&
	"Invalid cast!", file e:\llvm-development\llvm600\llvm\lib\ir\instructions.cpp, line 2565
	*)

	castIsValid = LLVMLibraryFunction["castIsValid"][op, val, ty];

	opStr = FirstCase[DownValues[LLVMEnumeration], Verbatim[RuleDelayed][Verbatim[HoldPattern][HoldPattern[LLVMEnumeration]["LLVMOpcode", name_]], op] :> name];

	strRef1 = LLVMLibraryFunction["LLVMPrintValueToString_toPointer"][val];
	str1 = LLVMLibraryFunction["setUTF8String"][strRef1];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef1];

	strRef2 = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][ty];
	str2 = LLVMLibraryFunction["setUTF8String"][strRef2];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef2];

	With[{text = StringTemplate["Invalid cast! Operand: `operand`, Value: `value`, Type: `type`"][<|
	"operand" -> opStr, "value" -> str1, "type" -> str2
	|>]},
	Assert[TrueQ[(castIsValid) || text]];
	];
]

castingH255[value_, valueKind_] :=
Module[{actualValueKind},

	(*
		This reproduces the assertion that looks like this in LLVM 6.0.0:
		Assertion failed: (isa<X>(Val) && "cast<Ty>() argument of incompatible type!"),
			function cast,
			file /Users/brenton/llvm-development/llvm60/llvm/include/llvm/Support/Casting.h, line 255.
	*)
	
	actualValueKind = LLVMLibraryFunction["LLVMGetValueKind"][value];

	With[{text = StringTemplate["cast<Ty>() argument of incompatible type! Expected ValueKind: `expected`, Actual ValueKind: `actual`"][<|
	"expected" -> valueKind, "actual" -> actualValueKind
	|>]},
	Assert[TrueQ[(actualValueKind === valueKind) || text]];
	]
]


typeCPP288[returnType_, paramTypes_, paramCount_, isVarArg_] :=
Module[{paramsList, isFirstClassType, isValidArgumentType, i, str, strRef},

	(*
		This reproduces the assertion that looks like this in LLVM 6.0.0:
		Assertion failed: (isValidArgumentType(Params[i]) && "Not a valid type for function argument!"),
			function FunctionType,
			file /Users/brenton/llvm-development/llvm601/llvm/lib/IR/Type.cpp, line 288.
	*)
	
	paramsList = LLVMLibraryFunction["LLVMLink_fromIntegerArray_Wrapper"][paramTypes, paramCount];
	
	MapIndexed[
		(
			i = #2[[1]];
			isFirstClassType = LLVMLibraryFunction["LLVMGetTypeKind"][#] != LLVMEnumeration["LLVMTypeKind", "LLVMFunctionTypeKind"] &&
										LLVMLibraryFunction["LLVMGetTypeKind"][#] != LLVMEnumeration["LLVMTypeKind", "LLVMVoidTypeKind"];
			isValidArgumentType = isFirstClassType;

			strRef = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][#];
			str = LLVMLibraryFunction["setUTF8String"][strRef];
			LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];

			With[{text = StringTemplate["Not a valid type for function argument! i: `i`, Type: `type`"][<|
			"i" -> i, "type" -> str
			|>]},
			Assert[TrueQ[(isValidArgumentType) || text]];
			]
		)&
		,
		paramsList
	]
]


valueCPP247[val_, name_] :=
Module[{ty, isVoidTy, strRef, str},

	(*
		This reproduces the assertion that looks like this in LLVM 6.0.0:
		Assertion failed: (!getType()->isVoidTy() && "Cannot assign a name to void values!"),
			function setNameImpl,
			file /Users/brenton/llvm-development/llvm601/llvm/lib/IR/Value.cpp, line 247.
	*)

	ty = LLVMLibraryFunction["LLVMTypeOf"][val];
	isVoidTy = LLVMLibraryFunction["LLVMGetTypeKind"][ty] == LLVMEnumeration["LLVMTypeKind", "LLVMVoidTypeKind"];

	strRef = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][ty];
	str = LLVMLibraryFunction["setUTF8String"][strRef];
	LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];

	With[{text = StringTemplate["Cannot assign a name to void values! Type: `type`"][<|
	"type" -> str
	|>]},
	Assert[TrueQ[(!isVoidTy) || text]];
	]
]



globalValueH233[global_, visibility_] :=
Module[{defaultVisibility, linkage, isInternalLinkage, isPrivateLinkage, isLocalLinkage, hasLocalLinkage},

	(*
		This reproduces the assertion that looks like this in LLVM 6.0.0:
		Assertion failed: ((!hasLocalLinkage() || V == DefaultVisibility) && "local linkage requires default visibility"),
			function setVisibility,
			file /Users/brenton/llvm-development/llvm601/llvm/include/llvm/IR/GlobalValue.h, line 233.
	*)

	defaultVisibility = LLVMEnumeration["LLVMVisibility", "LLVMDefaultVisibility"];

	linkage = LLVMLibraryFunction["LLVMGetLinkage"][global];
	isInternalLinkage = (linkage == LLVMEnumeration["LLVMLinkage", "LLVMInternalLinkage"]);
	isPrivateLinkage = (linkage == LLVMEnumeration["LLVMLinkage", "LLVMPrivateLinkage"]);
	isLocalLinkage = isInternalLinkage || isPrivateLinkage;
	hasLocalLinkage = isLocalLinkage;
	With[{text = StringTemplate["local linkage requires default visibility! Linkage: `linkage`, Visibility: `visibility`"][<|
	"linkage" -> linkage, "visibility" -> visibility
	|>]},
	Assert[TrueQ[(!hasLocalLinkage || visibility == defaultVisibility) || text]];
	]
]












getParameterTypes[fTy_] :=
	Module[{num, args, argsArray},
		num = LLVMLibraryFunction["LLVMCountParamTypes"][fTy];
		argsArray = LLVMLibraryFunction["LLVMLink_allocateLLVMOpaqueTypeObjectPointer"][num];
		LLVMLibraryFunction["LLVMGetParamTypes"][fTy, argsArray];
		args = Table[LLVMLibraryFunction["LLVMLink_getLLVMOpaqueTypeObjectPointer"][argsArray, i], {i, 0, num-1}];
		LLVMLibraryFunction["LLVMLink_deallocateLLVMOpaqueTypeObjectPointer"];
		args
	]





End[]

EndPackage[]
