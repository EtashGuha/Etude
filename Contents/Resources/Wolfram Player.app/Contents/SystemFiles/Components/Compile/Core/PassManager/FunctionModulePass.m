
BeginPackage["Compile`Core`PassManager`FunctionModulePass`"]

FunctionModulePass;
FunctionModulePassQ;
CreateFunctionModulePass;

Begin["`Private`"]

Needs["Compile`Core`PassManager`Pass`"]

CreateFunctionModulePass[data_?AssociationQ] :=
	CreatePass[FunctionModulePass, data]

FunctionModulePassQ[obj_?PassQ] := 
	(obj["_passtype"] === FunctionModulePass)
FunctionModulePassQ[___] := False


End[]
EndPackage[]