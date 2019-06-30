BeginPackage["LLVMCompileTools`Debugging`"]

AddSetBreakpoint

Begin["`Private`"]

Needs["LLVMCompileTools`"]; (* For AddCodeFunction *)
Needs["LLVMLink`"]; (* For LLVMLibraryFunction *)

AddCodeFunction["SetBreakpoint", AddSetBreakpoint];

AddSetBreakpoint[data_, _, {}] := Module[{voidTy, funTyId, funId, id},
	funId = LLVMLibraryFunction["LLVMWLCreateBreakpointInst"][data["moduleId"], data["builderId"]];
	If[!IntegerQ[funId], Throw["Expected funId to be IntegerQ, got: ", funId]];
	funId
];

End[]

EndPackage[]

