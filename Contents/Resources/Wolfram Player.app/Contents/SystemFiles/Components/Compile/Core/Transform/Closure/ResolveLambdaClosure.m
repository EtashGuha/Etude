BeginPackage["Compile`Core`Transform`Closure`ResolveLambdaClosure`"]

ResolveLambdaClosurePass

Begin["`Private`"] 

Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`LambdaInstruction`"]

run[fm_, opts_] :=
	Module[{ visitor, tyEnv, pm = fm["programModule"]},
		tyEnv = pm["typeEnvironment"];
		visitor =  CreateInstructionVisitor[
			<|
				"visitLambdaInstruction" -> Function[{st, inst},
					Module[{
						target = inst["target"],
						source = inst["source"],
						capturedBy
						},
						If[!ConstantValueQ[source] || !TrueQ[target["getProperty", "isCapturedVariable", False]],
							Return[]
					    ];
						capturedBy = target["getProperty", "capturedByVariables", {}];
						Scan[ fixCapturedBy[source, #, target]&, capturedBy];
						RemoveClosureCapteeProperties[fm, target];
					]
				]
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		visitor["traverse", fm];
	    fm
	]

(*
  var gets a value from a capture of a lambda instruction of src
  replace the def with a LambdaInstruction of src.
  remove the capture properties from var
  remove the capture properties from fm
*)
fixCapturedBy[ src_, var_, varMain_] :=
	Module[ {def = var["def"], fm, newInst},
		fm = def["basicBlock"]["functionModule"];
		If[ !CallInstructionQ[def] || !ConstantValueQ[def["function"]] || def["function"]["value"] =!= Native`LoadClosureVariable,
				ThrowException[{"Unrecognized form of closure capture", def}]];
		RemoveClosureCapturerProperties[fm, var, varMain];
		newInst = CreateLambdaInstruction[var, src];
		var["setDef", newInst];
		newInst["moveAfter", def];
		def["unlink"];
	]
	


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveLambdaClosure",
	"The pass looks at all Lambda Instructions which have a target which is a captured closure variable." <>
	"If it finds any, it converts the LoadClosureVariable call into a lambda instruction." <>
	"It then tidies up any properties to turn this out of a closure variable."
];

ResolveLambdaClosurePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ResolveLambdaClosurePass]
]]

End[]
	
EndPackage[]
