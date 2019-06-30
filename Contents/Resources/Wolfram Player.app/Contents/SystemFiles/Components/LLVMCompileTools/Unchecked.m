

BeginPackage["LLVMCompileTools`Unchecked`"]

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]

(*
use mkBinOp[...][##]& to prevent LLVMLibraryFunction from evaluating at package load time
*)
AddCodeFunction["UncheckedPlus", mkBinOp[LLVMLibraryFunction["LLVMBuildAdd"], LLVMLibraryFunction["LLVMBuildAdd"], LLVMLibraryFunction["LLVMBuildFAdd"]][##]&]
AddCodeFunction["UncheckedSubtract", mkBinOp[LLVMLibraryFunction["LLVMBuildSub"], LLVMLibraryFunction["LLVMBuildSub"], LLVMLibraryFunction["LLVMBuildFSub"]][##]&]
AddCodeFunction["UncheckedTimes", mkBinOp[LLVMLibraryFunction["LLVMBuildMul"], LLVMLibraryFunction["LLVMBuildMul"], LLVMLibraryFunction["LLVMBuildFMul"]][##]&]
AddCodeFunction["UncheckedDivide", mkBinOp[LLVMLibraryFunction["LLVMBuildSDiv"], LLVMLibraryFunction["LLVMBuildUDiv"], LLVMLibraryFunction["LLVMBuildFDiv"]][##]&]

mkBinOp[signedInteger_, unsignedInteger_, float_][data_?AssociationQ, f_, {arg1_, arg2_}] :=
	Module[ {id},
		id = Switch[data["callFunctionType"]["get"],
			TypeSpecifier[{"Integer8",  "Integer8"} -> _] |
			TypeSpecifier[{"Integer16", "Integer16"} -> _] |
			TypeSpecifier[{"Integer32", "Integer32"} -> _] |
			TypeSpecifier[{"Integer64", "Integer64"} -> _],
				signedInteger[data["builderId"], arg1, arg2, ""],
			TypeSpecifier[{"UnsignedInteger8",  "UnsignedInteger8"} -> _] |
			TypeSpecifier[{"UnsignedInteger16", "UnsignedInteger16"} -> _] |
			TypeSpecifier[{"UnsignedInteger32", "UnsignedInteger32"} -> _] |
			TypeSpecifier[{"UnsignedInteger64", "UnsignedInteger64"} -> _],
				unsignedInteger[data["builderId"], arg1, arg2, ""],
			TypeSpecifier[{"Real16",  "Real16"} -> _] |
			TypeSpecifier[{"Real32", "Real32"} -> _] |
			TypeSpecifier[{"Real64", "Real64"} -> _],
				float[data["builderId"], arg1, arg2, ""],
			Null,
			     ThrowException[{"Unhandled type when creating unchecked arithmetic function ", data["callFunctionType"]["get"]}]
		];
		id
	];
	

End[]


EndPackage[]

