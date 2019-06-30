


BeginPackage["CompileAST`Create`Fresh`"]

MExprFreshVariableName;
MExprFreshVariableNameReset;

Begin["`Private`"]

If[!AssociationQ[$freshNameTable],
	$freshNameTable = <||>
]

MExprFreshVariableNameReset[] :=
	$freshNameTable = <||>

MExprFreshVariableName[] :=
	MExprFreshVariableName["var"]
	
MExprFreshVariableName[s_String] :=
	Module[{id = Lookup[$freshNameTable, s, 1] + 1},
		AssociateTo[$freshNameTable, s -> id];
		StringJoin[s, "$$", ToString[id]]
	]

End[]
EndPackage[]