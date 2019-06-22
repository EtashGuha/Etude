BeginPackage["LLVMCompileTools`Finalize`"]

(*
The sense of the word "finalize" as used in this file is very specific.

When the last instance of a registered expression is going out of scope, then it is ready for finalizing.

This is an opportunity to release resources and free up memory.

The word "finalize" may be used in other places in the code, but its usage is not the same.
*)

EnableCompiledCodeFunctionFinalization

EnableLLVMModuleFinalization



Begin["`Private`"]


Needs["LLVMCompileTools`"]
Needs["LLVMLink`"]
Needs["LLVMTools`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"]


(*
Garbage Collection for LLVM objects

Use Language`FinalizeExpressionInitialize to hook up functions to run when the last instance (associated with a given name) goes out of scope.

These are paired with Language`FinalizeExpressionAdd calls, where the instances are added.
*)


Language`FinalizeExpressionInitialize::unhandled = "Unhandled call to finalizer: `1`"



disposeLLVMModule[{"Normal", LLVMModule, {id_Integer}}] :=
(
	LLVMLibraryFunction["LLVMDisposeModule"][id];
	$activeCompileCount--;

	checkRestartLLVM[];
)

disposeLLVMModule[args___] := (
	(*
	Throw cannot be used here
	*)
	Message[Language`FinalizeExpressionInitialize::unhandled, {args}];
	$Failed
)

disposeCompiledCodeFunction[{"Normal", CompiledCodeFunction, {_Association, _Integer, _Integer, _Integer, _String}}] :=
(
	DisposeOrcData[data];
	$activeCompileCount--;

	checkRestartLLVM[];
)

disposeCompiledCodeFunction[args___] := (
	(*
	Throw cannot be used here
	*)
	Message[Language`FinalizeExpressionInitialize::unhandled, {args}];
	$Failed
)


(*
The number of times LLVMModule[] and CompiledCodeFunction[] objects may be created before
restarting LLVM.

Note, this is a naive attempt at memory management. A better approach would be to ask
LLVM directly for its memory usage, but LLVM does not seem to publicly expose its memory allocator information.
*)
$llvmCompileLimit = 100


checkRestartLLVM[] :=
(
	If[$activeCompileCount == 0 && $totalCompileCount > $llvmCompileLimit,
		
		RunCallback["LLVMMemoryLimitExceeded", {}];
		
		$totalCompileCount = 0;
	]
)

RegisterCallback["FinalizeExpressionInitialize", Function[{st},
Language`FinalizeExpressionInitialize["LLVMModule", disposeLLVMModule];
Language`FinalizeExpressionInitialize["CompiledCodeFunction", disposeCompiledCodeFunction];
$totalCompileCount = 0;
$activeCompileCount = 0;
]]






EnableCompiledCodeFunctionFinalization[fun_CompiledCodeFunction] :=
(
	Language`FinalizeExpressionAdd["CompiledCodeFunction", fun];
	$totalCompileCount++;
	$activeCompileCount++;
)

EnableCompiledCodeFunctionFinalization[args___] :=
	ThrowException[{"Unrecognized call to EnableCompiledCodeFunctionFinalization", {args}}]


EnableLLVMModuleFinalization[mod_LLVMModule] :=
(
	Language`FinalizeExpressionAdd["LLVMModule", mod];
	$totalCompileCount++;
	$activeCompileCount++;
)

EnableLLVMModuleFinalization[args___] :=
	ThrowException[{"Unrecognized call to EnableLLVMModuleFinalization", {args}}]

	
End[]


EndPackage[]
