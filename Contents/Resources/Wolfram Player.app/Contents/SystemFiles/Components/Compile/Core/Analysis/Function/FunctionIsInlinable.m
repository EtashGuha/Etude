
BeginPackage["Compile`Core`Analysis`Function`FunctionIsInlinable`"]

FunctionIsInlinablePass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`InvokeInstruction`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Callback`"]
    
run[fm_?FunctionModuleQ, opts_] := 0

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "FunctionIsInlinable",
        "This pass computes whether the function can be inlined."
];

FunctionIsInlinablePass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionIsInlinablePass]
]]

End[]

EndPackage[]