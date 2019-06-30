

BeginPackage["LLVMCompileTools`Complex`"]

AddCreateVectorComplex

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Basic`"]


AddCodeFunction["CreateComplex", AddCreateVectorComplex]

AddCreateVectorComplex[data_, _, {re_, im_}] :=
	Module[ {tyId, idComp},
		tyId = LLVMLibraryFunction["LLVMTypeOf"][re];
		idComp = AddNull[data, GetVectorComplexType[data, tyId]];
		idComp = AddInsertElement[data, idComp, re, AddConstantInteger[data, 32,0]];
		idComp = AddInsertElement[data, idComp, im, AddConstantInteger[data, 32,1]];
		idComp
	]



End[]


EndPackage[]

