
BeginPackage["LLVMCompileTools`Structures`"]

GetStructureField

Begin["`Private`"]
Needs["LLVMLink`"]
Needs["LLVMCompileTools`Types`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]
Needs["LLVMCompileTools`Basic`"]

GetStructureField[ data_?AssociationQ, src_, index_] :=
	Module[ {id},
		id = getStructureFieldAddress[data, src, index];
        (* Load this from memory *)
        id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
        id
	]

getStructureFieldAddress[ data_?AssociationQ, src_, index_] :=
	Module[ {t1, t2, id},
		t1 = AddConstantInteger[data, 32, 0];
		t2 = AddConstantInteger[data, 32, index];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], src, #, 2, ""]&, {t1, t2}];
        id
	]

	
	
End[]

EndPackage[]

