
BeginPackage["Compile`Core`Transform`BasicBlockPhiReorder`"]

BasicBlockPhiReorderPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


(*
  Pass that moves PhiInstructions to the start of each BasicBlock.
*)

run[bb_, opts_] :=
	Module[{inst, lastPhi},
		inst = bb["firstInstruction"];
		lastPhi = None;
		Assert[InstructionQ[inst]];

		(* Set `lastPhi` (could be None) *)
		While[inst =!= None,
			If[PhiInstructionQ[inst],
				lastPhi = inst
				, (* else *)
				Break[] (* We've reached the first non Phi *)
			];
			inst = inst["next"];
		];

		While[inst =!= None,
			If[PhiInstructionQ[inst],
				If[lastPhi === None,
					(* If there was no leading Phi, make this phi the first instruction *)
					inst["moveBefore", bb["firstInstruction"]]
					,
					(* Otherwise if there was a leading Phi, move this Phi after that one *)
					inst["moveAfter", lastPhi]
				];
				lastPhi = inst;
			];
			inst = inst["next"];
		];

		bb["setProperty", "lastPhiInstruction" -> lastPhi]; (* Could be None *)
		bb
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"BasicBlockPhiReorder",
		"This pass moves all PhiInstruction to the start of each Basic Block it also adds the lastPhiInstruction property to the BasicBlock."
];

BasicBlockPhiReorderPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[BasicBlockPhiReorderPass]
]]

End[]

EndPackage[]
