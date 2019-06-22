
BeginPackage["Compile`Core`PassManager`ProgramModulePass`"]

ProgramModulePass;
ProgramModulePassQ;
CreateProgramModulePass;

Begin["`Private`"]

Needs["Compile`Core`PassManager`Pass`"]

CreateProgramModulePass[data_?AssociationQ] :=
	CreatePass[ProgramModulePass, data]

ProgramModulePassQ[obj_?PassQ] := 
	(obj["_passtype"] === ProgramModulePass)
ProgramModulePassQ[___] := False


End[] 
EndPackage[]