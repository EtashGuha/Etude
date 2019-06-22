

BeginPackage["LLVMCompileTools`Comparisons`"]



AddLLVMCompareCall


Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["CompileUtilities`Error`Exceptions`"]


$compareData := $compareData =
<|
	"binary_sameq_Boolean" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"]},
	"binary_unsameq_Boolean" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntNE"]},
	"binary_equal_Boolean" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"]},
	"binary_unequal_Boolean" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntNE"]},

	"binary_sameq_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"]},
	"binary_unsameq_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntNE"]},
	"binary_equal_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"]},
	"binary_unequal_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntNE"]},
	"binary_greater_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntSGT"]},
	"binary_greaterequal_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntSGE"]},
	"binary_less_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntSLT"]},
	"binary_lessequal_SignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntSLE"]},

	"binary_sameq_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"]},
	"binary_unsameq_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntNE"]},
	"binary_equal_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"]},
	"binary_unequal_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntNE"]},
	"binary_greater_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntUGT"]},
	"binary_greaterequal_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntUGE"]},
	"binary_less_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntULT"]},
	"binary_lessequal_UnsignedInteger" -> {LLVMLibraryFunction["LLVMBuildICmp"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntULE"]},

	"binary_sameq_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealUEQ"]},
	"binary_unsameq_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealUNE"]},
	"binary_equal_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealUEQ"]},
	"binary_unequal_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealUNE"]},
	"binary_greater_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealUGT"]},
	"binary_greaterequal_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealUGE"]},
	"binary_less_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealULT"]},
	"binary_lessequal_Real" -> {LLVMLibraryFunction["LLVMBuildFCmp"], LLVMEnumeration["LLVMRealPredicate", "LLVMRealULE"]}


|>



AddLLVMCompareCall[ state_, name_,inputs_] :=
	AddLLVMCompareCall[state, name, inputs, ""]

AddLLVMCompareCall[ state_, name_, {inputs___}, varName_] :=
	Module[{data, fun, opCode, id},
		data = Lookup[ $compareData, name, Null];
		If[data === Null,
			ThrowException[{"Cannot found comparison data", name}]];
		
		fun = First[data];
		opCode = Last[data];
		id = fun[ state["builderId"], opCode, inputs, varName];
		id
	]



AddCodeFunction["Native`SameInstanceQ", AddSameInstance]

AddSameInstance[ data_, _, {s1_, s2__}] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildICmp"][ data["builderId"], LLVMEnumeration["LLVMIntPredicate", "LLVMIntEQ"], s1, s2, ""];
		id
	]


End[]


EndPackage[]

