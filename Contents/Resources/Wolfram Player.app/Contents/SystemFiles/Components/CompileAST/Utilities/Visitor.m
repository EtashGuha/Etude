
BeginPackage["CompileAST`Utilities`Visitor`"]

CreateBuilderMExprVisitor;
CreateRecursiveMExprVisitor;
RecursiveMExprVisitor;
MExprVisit;

Begin["`Private`"] 

Needs["CompileAST`Utilities`Node`"]
Needs["CompileUtilities`Reference`"]

nodeTypes = Join[
	Union[ASTNodeType /@ $ASTNodes],
	{"Program", Normal}
];

allnodes = Join[$ASTNodes, nodeTypes]

defaultTraverseVisitors = <||>
defaultPreVisitors = <||>
defaultPostVisitors = <||>
defaultVisitors = <||>

nodeType[nd_] :=
	Lookup[ASTNodeType, nd, nd]
	
(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)

(*
  st is the visitor,  we are getting an accept method for the node.
  
  head will be Program,  head of a Normal expr, or Normal

  symNameF is a function to convert head into the name of the method (note that if head was compound 
  then the typical symNameF will not be happy
  
  default is an Association that might contain head -> fun,  it really needs to include Normal -> fun.
*)

iGetVisitor[st_, head_, symNameF_, default_] :=
	Module[{name = symNameF[head]},
		Which[
			KeyExistsQ[st, name],
				st[name],
			KeyExistsQ[st, symNameF[nodeType[head]]],
				st[symNameF[nodeType[head]]],
			nodeType[head] =!= "Program" &&
			nodeType[head] =!= "Atom" &&
			KeyExistsQ[st, symNameF[Normal]],
				st[symNameF[Normal]],
			KeyExistsQ[default, head],
				default[head],
			True,
				iGetVisitor[st, Normal, symNameF, default]
		]
	]

getPreVisitor[st_, head_] :=
	iGetVisitor[st, head, makePreVisitSymbolName, defaultPreVisitors]
getVisitor[st_, head_] :=
	iGetVisitor[st, head, makeVisitSymbolName, defaultVisitors]
getPostVisitor[st_, head_] :=
	iGetVisitor[st, head, makePostVisitSymbolName, defaultPostVisitors]
getTraverseVisitor[st_, head_] :=
	iGetVisitor[st, head, makeTraverseSymbolName, defaultTraverseVisitors]
	

(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)

previsitCommon[vst_, data_] := True
postVisitCommon[vst_, data_] := True
visitCommon[vst_, data_] := True

traverseAtom[vst_, data_] := True
traverseExpr[vst_, data_] := vst["traverseNormal"][vst, data]
traverseBinaryOp[vst_, data_] := vst["traverseNormal"][vst, data]
traverseScope[vst_, data_] := vst["traverseNormal"][vst, data]
traverseProgram[vst_, data_] := traverseStatement[vst, data]
traverseNormal[vst_, expr_] :=
	Module[{tovisit, bag, tmp, continue = True},
		tovisit = Join[
			{expr["head"]},
			expr["arguments"]
		];
		bag = Internal`Bag[];
		Do[
			If[continue,
				tmp = vst["traverseStatement"][vst, arg];
				If[continueTraversalQ[tmp],
					Internal`StuffBag[bag, tmp],
					continue = False
				];
			],
			{arg, tovisit}
		];
		If[continue,
			Internal`BagPart[bag, All],
			stopTraversalSymbol
		]
	]


getHead[ mexpr_] :=
	With[ {hd = mexpr["_head"]},
		Which[
			TrueQ[hd["symbolQ"]], hd["symbol"],
			TrueQ[hd["literalQ"]], hd["data"],
			True, hd]
	]

traverseStatement[vst_, stmt_] :=
	Module[{next, res = True, hd, rev = {}, prevQueue,
			previsitF, visitF, postVisitF, traverseF},
		prevQueue = vst["statementQueue"]["toList"];
		vst["statementQueue"]["set", {stmt}];
		While[vst["statementQueue"]["length"] =!= 0 && continueTraversalQ[res],
			next = vst["statementQueue"]["popFront"];
			hd = getHead[next];
			previsitF = getPreVisitor[vst, hd];
			visitF = getVisitor[vst, hd];
			traverseF = getTraverseVisitor[vst, hd];
			res = previsitF[vst, next];
			If[continueTraversalQ[res],
				res = traverseF[vst, next]
			];
			If[continueTraversalQ[res],
				res = visitF[vst, next]
			];
			AppendTo[rev, next];
		];

		Do[
			hd = getHead[next];
			postVisitF = getPostVisitor[vst, hd];
	        res = postVisitF[vst, next],
	        {next, Reverse[rev]}
		];
		vst["statementQueue"]["set", prevQueue];
		res
	]

(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)
	
symName[s_String] := s
symName[s_Symbol] := SymbolName[s]

makePreVisitSymbolName[node_] := makePreVisitSymbolName[node] =
	"preVisit" <> symName[node]
makeVisitSymbolName[node_] := makeVisitSymbolName[node] =
	"visit" <> symName[node]
makePostVisitSymbolName[node_] := makePostVisitSymbolName[node] =
	"postVisit" <> symName[node]
makeTraverseSymbolName[node_] := makeTraverseSymbolName[node] =
	"traverse" <> symName[node]
	
makePreVisitSymbol[node_] := makePreVisitSymbol[node] =
	Symbol[$Context <> makePreVisitSymbolName[node]]
makeVisitSymbol[node_] := makeVisitSymbol[node] =
	Symbol[$Context <> makeVisitSymbolName[node]]
makePostVisitSymbol[node_] := makePostVisitSymbol[node] =
	Symbol[$Context <> makePostVisitSymbolName[node]]
makeTraverseSymbol[node_] := makeTraverseSymbol[node] =
	Symbol[$Context <> makeTraverseSymbolName[node]]

(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)

Do[
	makePreVisitSymbol[nodeType[node]];
	makeVisitSymbol[nodeType[node]];
	makePostVisitSymbol[nodeType[node]];
	makeTraverseSymbol[nodeType[node]];
	AssociateTo[defaultPreVisitors, node -> previsitCommon];
	AssociateTo[defaultPostVisitors, node -> postVisitCommon];
	AssociateTo[defaultVisitors, node -> visitCommon];
	AssociateTo[defaultTraverseVisitors, node -> makeTraverseSymbol[nodeType[node]]];
	,
	{node, allnodes}
]

(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)
(*******************************************************************************)

CreateRecursiveMExprVisitor[st_, derived_:<||>] :=
	Module[{vst},
		vst = <||>;
		vst["state"] = st;
		vst["getState"] = Function[{}, st];
		vst["statementQueue"] = CreateReference[{}];
		vst["traverseAtom"] = traverseAtom;
		vst["traverseExpr"] = traverseExpr;
		vst["traverseBinaryOp"] = traverseBinaryOp;
		vst["traverseNormal"] = traverseNormal;
		vst["traverseStatement"] = traverseStatement;
		vst["traverseScope"] = traverseScope;
		vst["traverseProgram"] = traverseProgram;
		vst = Join[vst, derived];
		RecursiveMExprVisitor[vst]
	]

RecursiveMExprVisitor[vst_][key_] :=
	vst[key]

MExprVisit[RecursiveMExprVisitor[vst_], prog_] :=
	Module[{hd, preF, visitF, traverseF, postF, res},
		hd = "Program";
		preF = getPreVisitor[vst, hd];
		visitF = getVisitor[vst, hd];
		traverseF = getTraverseVisitor[vst, hd];
		postF = getPostVisitor[vst, hd];
		res = preF[vst, prog];
		If[continueTraversalQ[res],
			res = traverseF[vst, prog]
		];
		If[continueTraversalQ[res],
			res = visitF[vst, prog]
		];
		postF[vst, prog];
	]

CreateBuilderMExprVisitor[visitorIn_] :=
	Module[{visitor, stateArg, normalArgs},
		normalArgs = CreateReference[{}];
		stateArg = CreateReference[<|
			"result" -> "",
			"normalArgs" -> normalArgs
		|>];
		visitor = Join[
			visitorIn,
			<|
				"state" -> stateArg,
				"getResult" -> Function[{}, stateArg["lookup", "result"]],
				"pushExpr" -> Function[{}, normalArgs["appendTo", CreateReference[{}]]],
				"popExpr" -> Function[{}, normalArgs["popBack"]],
				"appendToExpr" -> Function[{val}, normalArgs["last"]["appendTo", val]],
				"setResult" -> Function[{val}, stateArg["associateTo", "result" -> val]]
			|>
		];
		CreateRecursiveMExprVisitor[stateArg, visitor]
	]
	
stopTraversalSymbol = $Failed
continueTraversalQ[stopTraversalSymbol] = False
continueTraversalQ[___] := True
	
End[]

EndPackage[]
