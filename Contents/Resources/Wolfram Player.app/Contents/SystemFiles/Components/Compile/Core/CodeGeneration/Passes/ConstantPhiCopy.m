
BeginPackage["Compile`Core`CodeGeneration`Passes`ConstantPhiCopy`"]

ConstantPhiCopyPass

Begin["`Private`"]

Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["CompileUtilities`Callback`"]

(*
 return True if the constant should move.
 Do this for Expression,  MObject,  CString, PackedArray
*)
shouldMove[ty_] :=
	ty["isConstructor","Expression"] || ty["implementsQ", "MObjectManaged"] || 
		isCStringType[ty] || ty["isNamedApplication", "PackedArray"]

isCStringType[ty_] := ty["isNamedApplication", "CArray"] && 
						Length[ty["arguments"]] === 1 && 
						First[ty["arguments"]]["isConstructor", "UnsignedInteger8"]

(*
  If the var is a constant with an Expression type (should really be 
  a type that is not a native type).  Then introduce a Copy from 
  the Constant into the srcBB and return the Copy var as the result.
*)
checkPhiData[ data_, srcBB_, var_] :=
	Module[ {newVar, newInst},
		If[ !(ConstantValueQ[var] && shouldMove[var["type"]]),
			Return[ {srcBB, var}]];
		newInst = CreateCopyInstruction["ConstantPhiCopy$" <> ToString[var["id"]], var];
		newVar = newInst["target"];
		newVar["setType", var["type"]];
		newInst["moveBefore", srcBB["lastInstruction"]];
		{srcBB, newVar}
	]

visitPhi[ data_, inst_] :=
	Module[ {srcData},
		srcData = inst["source"]["get"];
		srcData =
			Apply[ checkPhiData[data, #1, #2]&, srcData, {1}];
		inst["source"]["set", srcData];
	]


run[bb_, opts_] :=
	Module[{data},
		data = <||>;
		CreateInstructionVisitor[
				data,
				<|
				"visitPhiInstruction" -> visitPhi
				|>,
				bb,
			"IgnoreRequiredInstructions" -> True
			];		
		bb
	]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ConstantPhiCopy",
		"The pass moves non-atomic constants from Phi instructions into their Basic Blocks."
];

ConstantPhiCopyPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ConstantPhiCopyPass]
]]

End[]

EndPackage[]
