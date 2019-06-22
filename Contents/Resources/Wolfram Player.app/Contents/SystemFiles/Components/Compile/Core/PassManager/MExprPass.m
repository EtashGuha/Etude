
BeginPackage["Compile`Core`PassManager`MExprPass`"]

MExprPass;
MExprPassQ;
CreateMExprPass;

Begin["`Private`"]

Needs["Compile`Core`PassManager`Pass`"]

CreateMExprPass[data_?AssociationQ] :=
	CreatePass[MExprPass, data]

MExprPassQ[obj_?PassQ] := 
	(obj["_passtype"] === MExprPass)
MExprPassQ[___] := False


End[]
EndPackage[]

