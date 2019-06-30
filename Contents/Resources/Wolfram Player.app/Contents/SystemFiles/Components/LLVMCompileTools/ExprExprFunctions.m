

BeginPackage["LLVMCompileTools`ExprExprFunctions`"]


AddExprToExpr::usage = "AddExprToExpr  "

AddExprFromExpr::usage = "AddExprFromExpr  "

AddTestExpr::usage = "AddTestExpr  "

AddTestGetExprExpr


Begin["`Private`"]

Needs[ "LLVMLink`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`MTensor`"]


(*
 Expr Expr Functions
*)	

(*
   Really it is going to be an Expr, so just add True
*)
AddTestExpr[data_, _, {src_}] :=
	Module[ {id},
     	id = AddConstantBoolean[data, True];
     	id
	]


(*
  Doesn't need refcounting because the wrapper is refcount neutral.
*)

AddCodeFunction["Expr`ExprToExpr", AddExprToExpr1]

AddExprToExpr1[data_, _, {src_}] :=
	AddExprToExpr[data, "Expression", src]
	
AddExprToExpr[data_, ty_, src_] :=
	Module[ {},
		src
	]

AddCodeFunction["Expr`ExprFromExpr", AddExprFromExprTest]

AddExprFromExprTest[data_, _, {ref_, src_, _, _}] :=
	Module[ {off, test},
		off = AddConstantInteger[data, 64, 0];
		AddSetArray[ data, ref, off, src];
		test = AddConstantBoolean[data, True];
		test

	]


AddExprFromExpr[data_, _, {src_}] :=
	Module[ {},
		src
	]


(*
 Pretty simple
*)
AddTestGetExprExpr[data_, ty_, succRef_, src_] :=
	Module[ {},
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], src, succRef];
		AddConstantBoolean[data, True]
	]




End[]


EndPackage[]