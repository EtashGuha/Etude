
BeginPackage["CompileAST`Utilities`MExprVisitor`"]

MExprVisitor
MExprVisitorClass
MExprMapVisitor
MExprMapVisitorClass;

CreateMExprScanner

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Create`Construct`"]


RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprVisitorClass =
DeclareClass[
	MExprVisitor,
	<|
		"visitNormal" -> Function[{}, True],
		"visitSymbol" -> Function[{}, Null],
		"visitInteger" -> Function[{}, Null],
		"visitReal" -> Function[{}, Null],
		"visitBoolean" -> Function[{}, Null],
		"visitString" -> Function[{}, Null]
	|>,
	{
		"properties"	,
        "visitHeadsQ" -> True	
	},
	Extends -> {
		ClassPropertiesTrait
	},
    Predicate -> MExprVisitorQ
]
]]

RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprMapVisitorClass = DeclareClass[
	MExprMapVisitor,
 	<|
 	    "processNormal" -> Function[{mexpr}, processNormal[Self, mexpr]],
		"visitNormal" -> Function[{mexpr}, processNormal[Self, mexpr]],
		"visitSymbol" -> Function[{mexpr}, Self["setResult", mexpr]],
		"visitInteger" -> Function[{mexpr}, Self["setResult", mexpr]],
		"visitReal" ->   Function[{mexpr}, Self["setResult", mexpr]],
		"visitBoolean" -> Function[{mexpr}, Self["setResult", mexpr]],
		"visitString" -> Function[{mexpr}, Self["setResult", mexpr]]
  	|>,
  	{
  		"result"
  	},
  	Extends -> {MExprVisitorClass}
]
]]



(*
  TODO add an assert that mexpr is a normal
*)

sameInstanceQ[e1_Integer, e2_Integer] := e1 === e2
sameInstanceQ[e1_Real, e2_Real] := e1 === e2
sameInstanceQ[e1_Symbol, e2_Symbol] := e1 === e2
sameInstanceQ[e1_, e2_] :=
	e2 =!= Null && e1["sameQ", e2]
	
processNormal[ self_, mexpr_] :=
	Module[ {hd, args, tmp, ef = {}, changed = False},
	    AssertThat["The input mexpr is a normal", mexpr
			]["satisfies", MExprNormalQ
		];
		self["setResult", Null];
		hd = mexpr["head"];
		args = mexpr["arguments"];
		If[TrueQ[self["getVisitHeadsQ"]], (* Process heads of normals *)
		  hd["accept", self]
		];
		tmp = self["result"];
		If[ !sameInstanceQ[hd, tmp], 
		    changed = True;
		];
		ef = {ef, tmp};
		Do[ 
			self["setResult", Null];
			elem["accept", self];
			tmp = self["result"];
			If[tmp === Null,
			   tmp = elem; 
			];
			If[!sameInstanceQ[elem, tmp],
			   changed = True;
			];
			ef = {ef, tmp};
			,
			{elem, args}
		];
	    If[changed === True,
			ef = Flatten[ef];
			ef = CreateMExprNormal @@{ First[ef], Rest[ef]};
			self["setResult", ef]
			, (* Else *)
			self["setResult", mexpr]
	    ];
		False
	]
 
 
 
RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprScannerClass = DeclareClass[
	MExprScanner,
 	<|
 	    "visitNormal" -> Function[{mexpr}, Self["normal"][Self, mexpr]],
		"visitSymbol" -> Function[{mexpr}, Self["atom"][Self, mexpr]],
		"visitInteger" -> Function[{mexpr}, Self["atom"][Self, mexpr]],
		"visitReal" ->   Function[{mexpr}, Self["atom"][Self, mexpr]],
		"visitBoolean" -> Function[{mexpr}, Self["atom"][Self, mexpr]],
		"visitString" -> Function[{mexpr}, Self["atom"][Self, mexpr]]
  	|>,
  	{
  		"state",
  		"normal",
  		"atom"
  	},
  	Extends -> {MExprVisitorClass}
]
]]

CreateMExprScanner[ state_, normalFun_, atomFun_] :=
	CreateObject[ MExprScanner, <|"state" -> state, "normal" -> normalFun, "atom" -> atomFun|>]
	
End[]

EndPackage[]
