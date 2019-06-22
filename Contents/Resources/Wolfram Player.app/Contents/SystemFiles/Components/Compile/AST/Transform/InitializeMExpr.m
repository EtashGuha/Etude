BeginPackage["Compile`AST`Transform`InitializeMExpr`"]


InitializeMExprPass
FinalizeMExprPass

Begin["`Private`"] 


Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]

(*

*)

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"InitializeMExpr",
	"Initialize the MExpr "<>
	"this involves removing the contents of KernelFunction"
];

InitializeMExprPass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> runInitialize
|>];

RegisterPass[InitializeMExprPass];


]]

RegisterCallback["RegisterPass", Function[{st},

info = CreatePassInformation[
	"FinalizeMExpr",
	"Initialize the MExpr "<>
	"this involves restoring the contents of KernelFunction"
];

FinalizeMExprPass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> runFinalize
|>];

RegisterPass[FinalizeMExprPass];

]]

runInitialize[mexpr_, opts:OptionsPattern[]] :=
	runInitialize[mexpr, <| opts |>]
	
runInitialize[mexpr_, opts_?AssociationQ] :=
	Module[ {visitor},
		visitor = CreateObject[ InitializeMExprVisitor, <||>];
		mexpr[ "accept", visitor];
		visitor["result"]
	]
	

(*
  Return True if has the form[ Typed[ KernelFunction[ body], ty]
*)
isTypedKernelFunction[ mexpr_] :=
	Module[ {arg},
		If[ mexpr["hasHead", Typed] && mexpr["length"] === 2,
			arg = mexpr["part",1];
			arg["hasHead", KernelFunction] && arg["length"] === 1,
			False]
	]
	
	
(*
  Typed[KernelFunction[body], ty][args] ->
     TypedKernelFunctionCall[body, ty][args]
     
  Typed[KernelFunction[body], ty] ->
     TypedKernelFunction[body, ty]
*)

visitInitializeNormal[ visitor_, mexpr_] :=
	Module[ {body, ty, nexpr, head, nArgs, i},
		Which[
			isTypedKernelFunction[mexpr],
				body = mexpr["part",1]["part",1];
				ty = mexpr["part",2];
				nexpr = Apply[ CreateMExprNormal, {Compile`Utilities`TypedKernelFunction, {body, ty}}];
				visitor["setResult", nexpr];
			,
			isTypedKernelFunction[ mexpr["head"]],
				head = mexpr["head"];
				body = head["part",1]["part",1];
				ty = head["part",2];
				nexpr = Apply[ CreateMExprNormal, {Compile`Utilities`TypedKernelFunctionCall, {body, ty}}];
				nArgs = Table[
							mexpr["part",i]["accept", visitor];
							visitor["getResult"], {i, mexpr["length"]}];
				nexpr = Apply[ CreateMExprNormal, {nexpr, nArgs}];
				visitor["setResult", nexpr];
			,
			True,
				visitor["processNormal", mexpr]];
		False
	]
		
runFinalize[mexpr_, opts:OptionsPattern[]] :=
	runFinalize[mexpr, <| opts |>]
	
runFinalize[mexpr_, opts_?AssociationQ] :=
	Module[ {visitor},
		visitor = CreateObject[ FinalizeMExprVisitor, <||>];
		mexpr[ "accept", visitor];
		mexpr
	]
	
	


visitFinalizeNormal[ visitor_, mexpr_] :=
	Module[ {nexpr},
		If[ mexpr["hasHead", Compile`ConstantValue] && mexpr["length"] === 0 && 
				mexpr["hasProperty", "constantValueArgument"],
			nexpr = mexpr["getProperty", "constantValueArgument"];
			mexpr["removeProperty", "constantValueArgument"];
			mexpr["appendArgument", nexpr];
			False
			,
			True
		]
	]



RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	InitializeMExprVisitor,
	<|
		"visitNormal" -> Function[{mexpr}, visitInitializeNormal[Self, mexpr]]
	|>,
	{

  	},
  	Extends -> {MExprMapVisitorClass}
 ];
]]

RegisterCallback["DeclareCompileClass", Function[{st},
 DeclareClass[
	FinalizeMExprVisitor,
	<|
		"visitNormal" -> Function[{mexpr}, visitFinalizeNormal[Self, mexpr]]
	|>,
	{

  	},
  	Extends -> {MExprVisitorClass}
 ];
]]


End[]

EndPackage[]
