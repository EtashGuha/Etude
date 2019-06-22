
BeginPackage["Compile`Core`PassManager`BasicBlockPass`"]

BasicBlockPass;
BasicBlockPassQ;
CreateBasicBlockPass;

Begin["`Private`"]

Needs["Compile`Core`PassManager`Pass`"]


CreateBasicBlockPass[data_?AssociationQ] :=
	CreatePass[BasicBlockPass, data]

BasicBlockPassQ[obj_?PassQ] := 
	(obj["_passtype"] === BasicBlockPass)
BasicBlockPassQ[___] := False


End[]
EndPackage[]