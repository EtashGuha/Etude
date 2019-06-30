
BeginPackage["Compile`Core`Analysis`Properties`BasicBlockLastPhi`"]

BasicBlockLastPhiPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



(*
  Pass that computes the last PhiInstruction in each BasicBlock.
*)


run[bb_, opts_] :=
	Module[{inst, lastPhiInst},
		inst = bb["firstNonLabelInstruction"];
		If[ !PhiInstructionQ[inst],
			bb[ "removeProperty", "lastPhiInstruction"];
			Return[]];
		lastPhiInst = NestWhile[#["next"] &, inst, (PhiInstructionQ[#["next"]] && #["next"] =!= None) &];
		bb[ "setProperty", "lastPhiInstruction" -> lastPhiInst];
		bb
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"BasicBlockLastPhi",
		"This pass computes the last PhiInstruction in each Basic Block."
];

BasicBlockLastPhiPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[BasicBlockLastPhiPass]
]]

End[]

EndPackage[]
