

BeginPackage["LLVMCompileTools`PackedArrayExprFunctions`"]


AddPackedArrayToExpr
AddPackedArrayFromExpr
AddTestPackedArray
AddTestGetMTensorExpr


Begin["`Private`"]

Needs[ "LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`ExprFunctions`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`MTensor`"]
Needs["CompileUtilities`Error`Exceptions`"]


(*
 Packed Array Expr Functions
*)	

(*
  TODO,  if not a PackedArray, then coerce.  
*)

AddCodeFunction["Expr`PackedArrayQ", AddTestPackedArray]

AddTestPackedArray[data_?AssociationQ, _, {src_}] :=
	Module[ {id},
     	id = AddRawTypeQ[data, src, "RPACKED"];
     	id
	]


AddCodeFunction["Expr`PackedArrayToExpr", AddPackedArrayToExpr1]

AddPackedArrayToExpr1[data_, _, {src_}] :=
	AddPackedArrayToExpr[data, Null, src]
	
AddPackedArrayToExpr[data_?AssociationQ, ty_, src_] :=
	Module[ {id},
		id = AddFunctionCall[data, "CreateMTensorExpr", {src}];
		id
	]


PackedTypeBase[ data_, arg_] :=
	Which[
		arg === "Integer32" && data["machineIntegerSize"] === 32,
			2,
		arg === "Integer64" && data["machineIntegerSize"] === 64,
			2,
		True,
			Null]	

PackedTypeBase[ data_, "Real64"] :=
	3
	
PackedTypeBase[ data_, "Complex"["Real64"]] :=
	4
	

GetPackedTypeForm[ data_, "PackedArray"[ ty_, TypeFramework`TypeLiteral[ rank_Integer, _]]] :=
	{PackedTypeBase[data, ty], rank}

GetPackedTypeForm[ __] :=
	{Null, Null}

(*
  Check base, rank,  coerce list to data.
*)
AddTestGetMTensorExpr[data_, ty_, ref_, src_] :=
	Module[ {base, rank, baseVal, rankVal, id, comp, eqOp, test},
		{base, rank} = GetPackedTypeForm[data, ty];
		If[base === Null || rank === Null,
			ThrowException[{"Type not handled ", ty}]];
		baseVal = AddConstantInteger[data, 32, base];
		rankVal = AddConstantInteger[data, data["machineIntegerSize"], rank];
		id = AddFunctionCall[ data, "TestGet_MTensor", {src, baseVal, rankVal, ref}];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]


AddCodeFunction["ExprToMTensor", TestGetExprToMTensor]

TestGetExprToMTensor[data_, y_, {ref_, src_, base_, rank_}] :=
	Module[ {id, comp, eqOp, test},
		id = AddFunctionCall[ data, "TestGet_MTensor", {src, base, rank, ref}];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]

AddPackedArrayFromExpr[data_?AssociationQ, arg_, {src_}] :=
	Module[ {id},
		AddTestPackedArray[data, arg, {src}];
 		id = AddGetRawContents[data, src];
        id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], id, GetMTensorType[data], ""];
        AddMTensorRefCountIncrement[data, Null, {id}];
       	id
	]



End[]


EndPackage[]