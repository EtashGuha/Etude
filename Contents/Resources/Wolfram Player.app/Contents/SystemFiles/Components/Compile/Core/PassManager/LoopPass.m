
BeginPackage["Compile`Core`PassManager`LoopPass`"]

LoopPass;
LoopPassQ;
CreateLoopPass;

Begin["`Private`"]

Needs["Compile`Core`PassManager`Pass`"]


CreateLoopPass[data_?AssociationQ] :=
	CreatePass[LoopPass, data]

LoopPassQ[obj_?PassQ] := 
	(obj["_passtype"] === LoopPass)
LoopPassQ[___] := False


End[]
EndPackage[]