BeginPackage["LLVMTools`"]

LLVMToObjectFile
LLVMToModule
LLVMToMachineCodeFile
LLVMToMachineCodeString
LLVMToObjectFile
LLVMToBitcodeFile

LLVMModule
LLVMFunction
LLVMType
LLVMValue
LLVMContext

LLVMToString
LLVMToLLFile
LLVMVerifyModule
LLVMTripleFromSystemID
ResolveLLVMTargetTriple

LLVMDataLayoutFromSystemID
ResolveLLVMDataLayout

LLVMToWebAssembly


LLVMRunPasses::usage = ""

LLVMRunOptimizationPasses::usage = ""

LLVMRunPassManagerOptPasses::usage = "";

LLVMRunInstructionProfilingPass::usage = "Lowers llvm.instrprof_increment instructions into the functions and data structures responsible for managing runtime profiling."



WrapIntegerArray

ScopedAllocation



Begin["`Private`"]

General::compsand = "The operation is not permitted when running the compiler in sandbox mode."


Needs["LLVMTools`LLVMToObjectFile`"]
Needs["LLVMTools`LLVMComponentUtilities`"]
Needs["LLVMTools`LLVMPasses`"]
Needs["LLVMTools`WebAssembly`"]
Needs["LLVMTools`Allocation`"]

End[]

EndPackage[]

