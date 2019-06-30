
BeginPackage["Compile`Core`PassManager`CompiledProgramPass`"]

CompiledProgramPass
CompiledProgramPassQ
CreateCompiledProgramPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`Pass`"]

CreateCompiledProgramPass[data_?AssociationQ] :=
	CreatePass[CompiledProgramPass, data]

CompiledProgramPassQ[obj_?PassQ] := 
	(obj["_passtype"] === CompiledProgramPass)

CompiledProgramPassQ[___] := False


End[]
EndPackage[]