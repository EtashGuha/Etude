
BeginPackage["LLVMLink`Unsafe`"]


DisableUnsafeLLVMLinkFunctions


Begin["`Private`"]

Needs["LLVMLink`"]

(*

This files overrides definitions of LLVMLibraryFunction and LLVMEnumeration that are considered unsafe.

*)


DisableUnsafeLLVMLinkFunctions[] :=
Module[{globalContextFunctions, deprecatedFunctions, leakingFunctions},

(*

GLOBAL CONTEXT

Functions that use the global LLVM Context

*)

globalContextFunctions = {
"LLVMModuleCreateWithName",
"LLVMInt1Type",
"LLVMInt8Type",
"LLVMInt16Type",
"LLVMInt32Type",
"LLVMInt64Type",
"LLVMInt128Type",
"LLVMIntType",
"LLVMHalfType",
"LLVMFloatType",
"LLVMDoubleType",
"LLVMX86FP80Type",
"LLVMFP128Type",
"LLVMPPCFP128Type",
"LLVMStructType",
"LLVMVoidType",
"LLVMLabelType",
"LLVMX86MMXType",
"LLVMConstString",
"LLVMMDString",
"LLVMMDNode",
"LLVMInsertBasicBlock",
"LLVMConstStruct",
"LLVMAppendBasicBlock",
"LLVMCreateBuilder",
"LLVMParseBitcode",
"LLVMParseBitcode2",
"LLVMGetMDKindID",
"LLVMIntPtrType",
"LLVMIntPtrTypeForAS",

"LLVMGetGlobalContext"
};

Map[(
	LLVMLibraryFunction[#] :=
		Throw["Function " <> # <> " is unsafe. It uses the global LLVM Context. \
Prefer to use the definition that takes an LLVM Context."]
)&, globalContextFunctions];





(*

DEPRECATED

*)

deprecatedFunctions = {
"LLVMParseBitcode",
"LLVMParseBitcodeInContext",
"LLVMGetBitcodeModuleInContext",
"LLVMGetBitcodeModule"
};

Map[(
	LLVMLibraryFunction[#] :=
		Throw["Function " <> # <> " is unsafe. It is deprecated. \
Refer to LLVM documentation for an alternative."]
)&, deprecatedFunctions];

LLVMEnumeration["LLVMLinkerMode", "LLVMLinkerPreserveSource_Removed"] :=
	Throw["Enum " <> ("LLVMLinkerPreserveSource_Removed") <> " is unsafe. It is deprecated. \
Refer to LLVM documentation for an alternative."];






(*

LEAKS MEMORY

*)



leakingFunctions = {
"LLVMDisposeMessage",
"LLVMPrintModuleToString",
"LLVMPrintValueToString",
"LLVMPrintTypeToString",
"LLVMGetDefaultTargetTriple"
};



Map[(
	LLVMLibraryFunction[#] :=
		Throw["Function " <> # <> " is unsafe. It leaks memory in LLVMLink. \
Refer to LLVMLink documentation for an alternative."]
)&, leakingFunctions];


]



(*

DEFAULTS

*)
defaultFunctions = {
"LLVMGetDefaultTargetTriple",
"LLVMGetDefaultTargetTriple_toPointer"
}

Map[(
	LLVMLibraryFunction[#] :=
		Throw["Function " <> # <> " is unsafe. It uses system defaults. \
Construct the value explicitly or use a provided function."]
)&, defaultFunctions];






(*

Unsafe for miscellaneous reasons

*)

If[$OperatingSystem == "Windows",

LLVMLibraryFunction["LLVMOrcGetSymbolAddress"] :=
		Throw["Function LLVMOrcGetSymbolAddress is unsafe on Windows. It assumes ExportedSymbolsOnly=true. \
Use LLVMOrcGetSymbolAddress2 instead."]

]


End[]

EndPackage[]




