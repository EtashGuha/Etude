
BeginPackage["CompileAST`PatternMatching`ComparePattern`"]

MExprComparePattern;
MExprSortPatterns;

Begin["`Private`"] 


Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]

$classifications = 	<|
	"Identical" -> 0,
	"Equivalent" -> 0,
	"General" -> 0,
	"Specific" -> 0,
	"Disjoint" -> 0,
	"Incomparable" -> 0,
	"Error" -> 0
|>

compareBlankBlank[st_, a_, b_] :=
	Which[
		a["length"] === 0 && b["length"] === 0,
			"Identical",
		a["length"] === 1 && b["length"] === 1,
			compare[st, a["part", 1], b["part", 1]],
		a["length"] === 0 && b["length"] === 1,
			"General",
		a["length"] === 1 && b["length"] === 0,
			"Specific",
		a["sameQ", b],
			"Identical"
	]

compareBlank[st_, a_, b_] :=
	Which[
		a["hasHead", Blank] && b["normalQ"] && b["hasHead", Blank],
			compareBlankBlank[st, a, b],
		True,
			0
	]
compareNamedPattern[st_, a_, b_] :=
	compare[st, a["part", 2], b]
	
compareBlank[st_, a_, b_] :=
	0
	

compare[st_, a_, b_] :=
	0;


RegisterCallback["DeclareCompileASTClass", Function[{st},
DeclareClass[
	ComparePatternsVisitor,
	<|
		"visitNormal" -> Function[{obj}, compare[Self, obj, Self["other"]]]
	|>,
	{
		"matcher",
		"other",
		"result"
	},
	Extends -> {MExprVisitorClass}]
]]


MExprComparePattern[st_, a_, b_] :=
	With[{
		visitor = CreateObject[ComparePatternsVisitor, <|
			"matcher" -> obj,
			"other" -> b,
			"result" -> CreateReference[]
		|>]
	},
		a["accept", visitor];
		visitor["result"]
	]

MExprSortPatterns[pats_] :=
	pats

End[]

EndPackage[]

