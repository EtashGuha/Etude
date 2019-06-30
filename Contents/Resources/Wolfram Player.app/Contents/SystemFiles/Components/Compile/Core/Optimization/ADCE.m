(*

BeginPackage["Compile`Passes`ADCEPass`"] (* Aggressive Dead code elimination *)

ADCEPass;

Begin["`Private`"] 


Needs["CompileUtilities`Callback`"]


ADCEPass[fun_FunctionInstruction] :=
	Module[{st},
		st = <|
			"LiveSet" -> Ref[<||>],
			"WorkList" -> Ref[{}]
		|>;
		Do[
			Do[
				Which[
					isTriviallyLive[inst],
						markLive[st, inst],
					emptyUses[inst],
						removeInstruction[bb, inst]
				],	
				{inst, instructions[bb]}
			],
			{bb, depthOrder[fun]}
		];
		While[notEmpty[st["WorkList"]],
			inst = pop[st["WorkList"]];
			If[reachable[basicBlock[inst]],
				Do[
					If[isInstruction[op],
						markLive[st, op]
					],
					{op, operands[inst]}
				]
			]
		];
		Do[
			If[reachable[bb],
				Do[
					If[!live[inst],
						drop[inst]
					],
					{inst, instructions[bb]}
				]
			],
			{bb, basicBlocks[fun]}
		]
	]
	
markLive[st_, inst_] :=
	If[TrueQ[Lookup[st["LiveSet"]["get"], inst, False]],
		st["LiveSet"]["set", True];
		st["WorkList"]["set", Append[st["WorkList"], inst]]
	]

isTriviallyLive[inst_] :=
	hasSideEffect[inst] || isTerminator[inst]
	
	
End[] 

EndPackage[]

*)