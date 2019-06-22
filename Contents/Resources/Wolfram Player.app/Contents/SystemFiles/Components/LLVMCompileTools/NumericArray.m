

BeginPackage["LLVMCompileTools`NumericArray`"]

AddTestGetNumericArrayExpr

AddNumericArrayToExpr

Begin["`Private`"]

Needs[ "LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`ExprFunctions`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`MTensor`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMCompileTools`Structures`"]


$typeBase =
<|
	"Integer8" -> 1,
	"UnsignedInteger8" -> 2,
	"Integer16" -> 3,
	"UnsignedInteger16" -> 4,
	"Integer32" -> 5,
	"UnsignedInteger32" -> 6,
	"Integer64" -> 7,
	"UnsignedInteger64" -> 8,
	"Real32" -> 9,
	"Real64" -> 10,
	"Complex"["Real32"] -> 11,
	"Complex"["Real64"] -> 12
|>

NumericArrayTypeForm[ data_, "NumericArray"[ ty_, TypeFramework`TypeLiteral[ rank_Integer, _]]] :=
	Module[ {base = Lookup[$typeBase, ty, Null]},
		{base, rank}
	]

NumericArrayTypeForm[ __] :=
	{Null, Null}


(*
  Check base, rank,  coerce list to data.
*)
AddTestGetNumericArrayExpr[data_, ty_, ref_, src_] :=
	Module[ {base, rank, baseVal, rankVal, id, comp, eqOp, test},
		{base, rank} = NumericArrayTypeForm[data, ty];
		If[base === Null || rank === Null,
			ThrowException[{"Type not handled ", ty}]];
		baseVal = AddConstantInteger[data, 32, base];
		rankVal = AddConstantInteger[data, data["machineIntegerSize"], rank];
		id = AddFunctionCall[ data, "TestGet_MNumericArray", {src, baseVal, rankVal, ref}];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]


AddNumericArrayToExpr[data_?AssociationQ, ty_, src_] :=
	Module[ {id},
		id = AddFunctionCall[data, "CreateMNumericArrayExpr", {src}];
		id
	]



AddCodeFunction["AddGetMNumericArrayDimensions", AddGetMNumericArrayDimensions]

AddGetMNumericArrayDimensions[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		(*  Get Dimensions *)
		id = GetStructureField[data, src, 1];
		id
	]

AddCodeFunction["MNumericArrayNumberOfElements", MNumericArrayNumberOfElements]

MNumericArrayNumberOfElements[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		(*  Get Dimensions *)
		id = GetStructureField[data, src, 6];
		id
	]

AddCodeFunction["MNumericArrayData", MNumericArrayData]

MNumericArrayData[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		(*  Get Dimensions *)
		id = GetStructureField[data, src, 7];
		id
	]




End[]


EndPackage[]