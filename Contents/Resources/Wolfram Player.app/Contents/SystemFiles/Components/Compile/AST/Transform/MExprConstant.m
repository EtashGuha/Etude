BeginPackage["Compile`AST`Transform`MExprConstant`"]

MExprConstantPass

Begin["`Private`"] 


Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Literal`"]

(*

*)

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"MExprConstant",
	"Finds List structures that can be transformed into constants and wraps them in Compile`ConstantValue."
];

MExprConstantPass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[MExprConstantPass]
]]


run[mexpr_, opts:OptionsPattern[]] :=
	run[mexpr, <| opts |>]
	
run[mexpr_, opts_?AssociationQ] :=
	Module[ {visitor},
		visitor = CreateObject[ ConstantListVisitor, <||>];
		mexpr[ "accept", visitor];
		visitor["result"]
	]
	

(*
  Look at a List calling the code in ConstantValue to check if 
  it contains the right contents. If so then try to pack the contents.
  If it is packable then make a Compile`ConstantValue.
*)
processList[visitor_, mexpr_] :=
	Module[{data, newMexpr},
		data = ProcessPackedArray[mexpr];
		If[
			data === Null,
				Return[visitor["processNormal", mexpr]]];
			data = Developer`ToPackedArray[data];
			If[Developer`PackedArrayQ[data],
				newMexpr = CreateMExpr @@ {data};
				newMexpr = CreateMExprNormal[ Compile`ConstantValue, {newMexpr}];
				newMexpr["joinProperties", mexpr];
				visitor["setResult", newMexpr];
				False
				,
				visitor["processNormal", mexpr]]
	]


checkRationalComplex[mexpr_] :=
	mexpr["length"] === 2 && MExprLiteralQ[mexpr["part",1]]  && MExprLiteralQ[mexpr["part",2]]

(*
  
*)
processRationalComplex[visitor_, mexpr_] :=
	Module[{newMexpr},
		If[ checkRationalComplex[mexpr],
			newMexpr = CreateMExprNormal[ Compile`ConstantValue, {mexpr}];
			newMexpr["joinProperties", mexpr];
			visitor["setResult", newMexpr];
			False
			,
			visitor["processNormal", mexpr]]
	]


(*
  Don't go inside a Compile`ConstantValue,  and process a list.
*)
visitNormal[ visitor_, mexpr_] :=
	Which[
		mexpr["hasHead", Compile`ConstantValue],
			visitor["setResult", mexpr];
			False
		,
		mexpr["hasHead", List],
			processList[visitor, mexpr]
		,
		mexpr["hasHead", Complex],
			processRationalComplex[visitor, mexpr]
		,
		mexpr["hasHead", Rational],
			processRationalComplex[visitor, mexpr]
		,
		True,
			visitor["processNormal", mexpr]
	]


RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	ConstantListVisitor,
	<|
		"visitNormal" -> Function[{mexpr}, visitNormal[Self, mexpr]]
	|>,
	{
  	},
  	Extends -> {MExprMapVisitorClass}
 ]
]]


End[]

EndPackage[]
