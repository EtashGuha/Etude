


BeginPackage["Compile`Core`Analysis`Function`FunctionInlineInformation`"]


FunctionInlineInformationPass



Begin["`Private`"] 

Needs["Compile`Core`IR`BasicBlock`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Utilities`Serialization`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]

Needs["Compile`Core`Analysis`Function`FunctionIsTrivialCall`"]
Needs["Compile`Core`Analysis`Function`FunctionAlwaysInline`"]
    
$passes = {
    FunctionIsTrivialCallPass,
    FunctionAlwaysInlinePass
}


run[fm_, opts_] :=
    fm
    

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "FunctionInlineInformation",
        "This pass computes inlining information for each function."
];

FunctionInlineInformationPass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "postPasses" -> $passes,
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionInlineInformationPass]
]]
    
End[]
EndPackage[]

