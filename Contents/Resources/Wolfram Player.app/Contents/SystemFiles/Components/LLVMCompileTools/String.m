

BeginPackage["LLVMCompileTools`String`"]


Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMTools`"]

AddCodeFunction["AddMStringToCString", AddMStringToCString]

AddMStringToCString[data_, _, {src_}] :=
	Module[ {t1, t2, id},
		t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], src, #, 2, ""]&, {t1, t1}];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
       	id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], id, Getutf8strType[data], ""];
       	t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 3, 0];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]



End[]


EndPackage[]

