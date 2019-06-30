
BeginPackage["Compile`Core`Transform`ConstantArrayPromote`"]

ConstantArrayPromotePass

Begin["`Private`"]

Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["CompileUtilities`Callback`"]


(*
 Introduce a new copy for constant packed arrays.  This makes the
 alias code detect this as an alias.
*)
visitCopy[state_, inst_] :=
	Module[{src = inst["source"], newInst, newVar},
		If[ConstantValueQ[src] && src["type"]["isNamedApplication", "PackedArray"],
			newInst = CreateCopyInstruction["cons", src];
			newVar = newInst["target"];
			newVar["setType", src["type"]];
			inst["setSource", newVar];
			newInst["moveBefore", inst];
			]
	]


(*
  If the var is a constant it will have a PackedArrayType introduce a Copy from 
  the constant into the srcBB and return the Copy var as the result.
*)
newPhiData[ state_, srcBB_, var_] :=
	Module[ {newVar, newInst},
		If[ !ConstantValueQ[var],
			Return[ {srcBB, var}]];
		newInst = CreateCopyInstruction["phiMove", var];
		newVar = newInst["target"];
		newVar["setType", var["type"]];
		newInst["moveBefore", srcBB["lastInstruction"]];
		{srcBB, newVar}
	]

visitPhi[ data_, inst_] :=
	Module[ {trgt = inst["target"], srcData},
		If[trgt["type"]["isNamedApplication", "PackedArray"] && AnyTrue[inst["getSourceVariables"], ConstantValueQ],
			srcData = inst["source"]["get"];
			srcData =
				Apply[ newPhiData[data, #1, #2]&, srcData, {1}];
			inst["source"]["set", srcData];
		]
	]


run[bb_, opts_] :=
	Module[{data},
		data = <||>;
		CreateInstructionVisitor[
				data,
				<|
				"visitCopyInstruction" -> visitCopy,
				"visitPhiInstruction" -> visitPhi
				|>,
				bb,
			"IgnoreRequiredInstructions" -> True
			];		
		bb
	]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ConstantArrayPromote",
		"The pass moves array constants from Phi into their Basic Blocks and Copy Instructions into a new Copy."
];

ConstantArrayPromotePass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ConstantArrayPromotePass]
]]

End[]

EndPackage[]
