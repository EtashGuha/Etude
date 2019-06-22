BeginPackage["LLVMCompileTools`BitOperations`"]


Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"]


AddCodeFunction["BitAnd", AddBinaryBitFunction[#1, #2, #3, LLVMLibraryFunction["LLVMBuildAnd"]]&]
AddCodeFunction["BitOr", AddBinaryBitFunction[#1, #2, #3, LLVMLibraryFunction["LLVMBuildOr"]]&]
AddCodeFunction["BitXor", AddBinaryBitFunction[#1, #2, #3, LLVMLibraryFunction["LLVMBuildXor"]]&]
AddCodeFunction["BitShiftLeft", AddBinaryBitFunction[#1, #2, #3, LLVMLibraryFunction["LLVMBuildShl"]]&]
AddCodeFunction["BitShiftRight", AddBinaryBitFunction[#1, #2, #3, LLVMLibraryFunction["LLVMBuildAShr"]]&]

AddBinaryBitFunction[ data_, _, {s1_, s2_}, libFun_] :=
	Module[ {id},
		id = libFun[ data["builderId"], s1, s2, ""];
		id
	]



End[]


EndPackage[]

