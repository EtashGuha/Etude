BeginPackage["Compile`Core`IR`Instruction`Utilities`InstructionMatch`"]

InstructionMatchQ


Begin["`Private`"]

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`SelectInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`"]



ClearAll[InstructionMatchQ]

instructionInfo[s_Symbol] :=
	instructionInfo[ToString[s]]
instructionInfo[s_String] :=
	$RegisteredInstructions[s]
	
instructionPredicate[s_] :=
	With[{info = instructionInfo[s]},
		info["predicate"]
	]

SetAttributes[InstructionMatchQ, HoldRest]

InstructionMatchQ[inst_, Set[trgt_, hd_[argsCheck___]]] :=
	inst["hasTarget"] &&
	matchArg[inst["target"], trgt] &&
	InstructionMatchQ[inst, hd[argsCheck]]

InstructionMatchQ[inst_, hd_[argsCheck___]] :=
	instructionPredicate[hd][inst] &&
	matchInstrArgs[inst, {argsCheck}]

SetAttributes[matchInstrArgs, HoldRest]
matchInstrArgs[inst_, argsCheck_] /; inst["hasOperands"] :=
	Which[
		SelectInstructionQ[inst],
			matchArgs[Prepend[inst["operands"], inst["condition"]], argsCheck],
		Length[inst["operands"]] === Length[argsCheck],
			matchArgs[inst["operands"], argsCheck],
		True,
			False
	]

SetAttributes[matchArgs, HoldAllComplete]
matchArgs[args_, argsCheck_] := (
	Which[
		MatchQ[argsCheck, Verbatim[{___}]],
			True,
		MatchQ[argsCheck, Verbatim[{__}]],
			Length[args] > 0,
		True,
			AllTrue[
				Transpose[{args, argsCheck}],
				Apply[matchArg, #]&
			]
	]
)

SetAttributes[matchArg, HoldRest]
matchArg[arg_, hd_] := (
	Which[
		MatchQ[hd, _Blank],
			If[Length[hd] === 0,
				True,
				matchArg[arg, First[hd]]
			],
		MatchQ[hd, _Pattern],
			matchArg[arg, Last[hd]],
		MatchQ[hd, _PatternTest],
			Last[hd][arg], 
		True,
			Head[arg] === hd ||
			AnyTrue[
				{
					hd === ConstantValue && ConstantValueQ[arg],
					hd === canberemovedVariable && VariableQ[arg],
					hd === BasicBlock && BasicBlockQ[arg]
				},
				TrueQ
			]
	]
)

End[]
EndPackage[]
