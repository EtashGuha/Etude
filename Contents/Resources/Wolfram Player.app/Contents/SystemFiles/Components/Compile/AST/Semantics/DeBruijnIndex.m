(**
  * The DeBruijnIndexWriter augments all MExpr symbols with their deBruijn
  * index. Global symbols (unbound symbols) have an index of 0.
  *********************************************************************** 
  * From wikipedia:
  * In mathematical logic, the De Bruijn index is a notation invented
  * by the Dutch mathematician Nicolaas Govert de Bruijn for representing 
  * terms in the lam calculus with the purpose of eliminating the names of 
  * the variable from the notation. Terms written using these indices are 
  * invariant with respect to alpha conversion, so the check for alpha-equivalence 
  * is the same as that for syntactic equality. Each De Bruijn index is a 
  * natural number that represents an occurrence of a variable in a lam-term, 
  * and denotes the number of binders that are in scope between that occurrence 
  * and its corresponding binder. The following are some examples:
  *  - The term lam x. lam y. x, sometimes called the K combinator, is written
  *    as lam lam 2 with De Bruijn indices. The binder for the occurrence x is
  *    the second lam in scope.
  *  - The term lam x. lam y. lam z. x z (y z) (the S combinator), with De Bruijn
  *    indices, is lam lam lam 3 1 (2 1).
  *  - The term lam z. (lamy. y (lam x. x)) (lam x. z x) is
  *     lam (lam 1 (lam 1)) (lam 2 1).
  ***********************************************************************
  *) 

BeginPackage["Compile`AST`Semantics`DeBruijnIndex`"]

DeBruijnIndexWriter;
DeBruijnIndexWriterPass;

Begin["`Private`"]

Needs["CompileAST`Utilities`Visitor`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"DeBruijnIndexWriter",
	"The DeBruijnIndexWriter augments all MExpr symbols with their deBruijn " <>
	"index. Global symbols (unbound symbols) have an index of 0."
];

DeBruijnIndexWriterPass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> DeBruijnIndexWriter
|>];

RegisterPass[DeBruijnIndexWriterPass]
]]


(** TODO: A way to fix the reference problem is to not have all variables
  alias, but rather have them maintain their use-def chain and during rename
  one just traverses the chain and renames. This way one can set attributes
  at each variable level *)
DeBruijnIndexWriter[mexpr_, opts_:<||>] :=
	Module[{vst, st, deBruijnVistior},
		st = CreateReference[1];
		deBruijnVistior = <|
			"postVisitFunction" -> postVisitFunction,
			"postVisitScope" -> postVisitScope,
			"postVisitSymbol" -> postVisitSymbol
		|>;
		vst = CreateRecursiveMExprVisitor[st, deBruijnVistior];
		MExprVisit[vst, mexpr];
		mexpr
	]

postVisitSymbol[vst_, expr_] := (
	If[!expr["hasProperty", "deBruijnIndex"],
		expr["setProperty", "deBruijnIndex" -> -1]
	];
	True
)

postVisitScope[vst_, expr_] :=
	Module[{st, idx, var, args},
		st = vst["state"];
		idx = st["get"];
		args = expr["part", 1];
		Do[
			var = If[arg["hasHead", Set] && arg["length"]=== 2,
				arg["part", 1], 
				arg
			];
			Assert[MExprSymbolQ[var]];
			var["setProperty", "deBruijnIndex" -> idx],
			{arg, args["arguments"]}
		];
		True
	]

postVisitFunction[vst_, expr_] :=
	Module[{st, var, arg, idx, args},
		st = vst["state"];
		idx = st["increment"];
		args = expr["part", 1];
		Do[
			var = If[MExprNormalQ[arg] && arg["hasHead", Typed] && arg["length"] == 2, 
				arg["part", 1], 
				arg
			];
			Assert[MExprSymbolQ[var]];
			var["setProperty", "deBruijnIndex" -> idx],
			{arg, args["arguments"]}
		];
		expr["setProperty", "deBruijnIndex" -> idx];
		True
	]


	

	
	
End[]
EndPackage[]
