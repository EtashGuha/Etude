BeginPackage["Compile`Core`IR`Instruction`InstructionQ`"]

InstructionQ

Begin["`Private`"]

Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]

InstructionQ[inst_] := 
	TrueQ[inst["hasField", "_instructionName"]] && KeyExistsQ[$RegisteredInstructions, inst["_instructionName"]]


End[]

EndPackage[]
