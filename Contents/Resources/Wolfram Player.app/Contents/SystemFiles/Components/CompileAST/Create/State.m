

BeginPackage["CompileAST`Create`State`"]

MExprState;
MExprStateQ;
CreateMExprState;
ResetMExprState;



Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileAST`Create`Fresh`"]


getId := getId = CreateReference[1]
symbolCache := symbolCache = CreateReference[<||>]
literalCache := literalCache = CreateReference[<||>]
ResetMExprState[] := (
	MExprFreshVariableNameReset[];
	getId = CreateReference[1];
	symbolCache = CreateReference[<||>];
	literalCache = CreateReference[<||>]
)
CreateMExprState[] :=
	MExprState[<|
		"getId" -> getId,
		"symbolCache" -> symbolCache,
		"literalCache" -> literalCache
	|>]
MExprState[st_][key_] := st[key]

MExprStateQ[MExprState[_Association]] := True
MExprStateQ[___] := False


End[]
EndPackage[]