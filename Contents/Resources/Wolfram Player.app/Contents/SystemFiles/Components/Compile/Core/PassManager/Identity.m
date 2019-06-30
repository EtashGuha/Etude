
BeginPackage["Compile`Core`PassManager`Identity`"]

IdentityPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["CompileUtilities`Callback`"]



run[__] := True


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"Identity",
		"The pass does nothing to the basic blocks, function modules, or program modules.",
		"The pass's main purpose is to allow one to place a placeholder in the pass manager."
];

IdentityPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[IdentityPass]
]]

	
End[]
EndPackage[]