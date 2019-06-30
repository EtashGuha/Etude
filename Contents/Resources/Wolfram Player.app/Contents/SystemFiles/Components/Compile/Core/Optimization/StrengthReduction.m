
(*

StrengthReduce[inst_Instruction] /; inst["op"] === Times && inst["arg1"] === 2 :=
	Instruction[inst["arg2"] + inst["arg2"]]

*)