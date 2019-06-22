BeginPackage["LLVMTools`Allocation`"]

Begin["`Private`"]

Needs["LLVMTools`"]
Needs["LLVMLink`"]
Needs["LLVMLink`LLVMInformation`"]


WrapIntegerArray[ fun_, args_] :=
	Module[ {argsId},
		Internal`WithLocalSettings[
			argsId = LLVMLibraryFunction["LLVMLink_getIntegerArray_Wrapper"][args];
			,
			fun[ argsId]
			,
			LLVMLibraryFunction["LLVMLink_freeIntegerArray_Wrapper"][argsId];
		]
	]
	
WrapIntegerArray[fun_, args1_, args2_] :=
	Module[ {args1Id, args2Id},
		Internal`WithLocalSettings[
			args1Id = LLVMLibraryFunction["LLVMLink_getIntegerArray_Wrapper"][args1];
			args2Id = LLVMLibraryFunction["LLVMLink_getIntegerArray_Wrapper"][args2];
			,
			fun[ args1Id, args2Id]
			,
			LLVMLibraryFunction["LLVMLink_freeIntegerArray_Wrapper"][args1Id];
			LLVMLibraryFunction["LLVMLink_freeIntegerArray_Wrapper"][args2Id];
		]
	]




(*
use memoization to prevent LLVMLibraryFunction from being called at package load time

We also want to keep this foo := foo = bar memoization pattern because it can be easily
recognized by the sandbox system
*)
$allocators := $allocators = <|
"CharObjectPointer" -> LLVMLibraryFunction["LLVMLink_allocateCharObjectPointer"],
"LLVMOpaqueMemoryBufferObjectPointer" -> LLVMLibraryFunction["LLVMLink_allocateLLVMOpaqueMemoryBufferObjectPointer"],
"LLVMOpaqueModuleObjectPointer" -> LLVMLibraryFunction["LLVMLink_allocateLLVMOpaqueModuleObjectPointer"],
"LLVMTargetObjectPointer" -> LLVMLibraryFunction["LLVMLink_allocateLLVMTargetObjectPointer"],
"LLVMOpaqueTypeObjectPointer" -> LLVMLibraryFunction["LLVMLink_allocateLLVMOpaqueTypeObjectPointer"],
"Uint64_tObject" -> LLVMLibraryFunction["LLVMLink_allocateUint64_tObject"],
If[$LLVMInformation["LLVM_VERSION"] >= 7.0, Nothing, "Uint32_tObject" -> LLVMLibraryFunction["LLVMLink_allocateUint32_tObject"]]
|>

$deallocators := $deallocators = <|
"CharObjectPointer" -> LLVMLibraryFunction["LLVMLink_deallocateCharObjectPointer"],
"LLVMOpaqueMemoryBufferObjectPointer" -> LLVMLibraryFunction["LLVMLink_deallocateLLVMOpaqueMemoryBufferObjectPointer"],
"LLVMOpaqueModuleObjectPointer" -> LLVMLibraryFunction["LLVMLink_deallocateLLVMOpaqueModuleObjectPointer"],
"LLVMTargetObjectPointer" -> LLVMLibraryFunction["LLVMLink_deallocateLLVMTargetObjectPointer"],
"LLVMOpaqueTypeObjectPointer" -> LLVMLibraryFunction["LLVMLink_deallocateLLVMOpaqueTypeObjectPointer"],
"Uint64_tObject" -> LLVMLibraryFunction["LLVMLink_deallocateUint64_tObject"],
If[$LLVMInformation["LLVM_VERSION"] >= 7.0, Nothing, "Uint32_tObject" -> LLVMLibraryFunction["LLVMLink_deallocateUint32_tObject"]]
|>



ScopedAllocation[type_][func_, args___] :=
	Module[{res, allocate, deallocate},
		allocate = $allocators[type];
		deallocate = $deallocators[type];
		Internal`WithLocalSettings[
			res = allocate[args];
			,
			func[res];
			,
			deallocate[res];

		]
	]

End[]

EndPackage[]
