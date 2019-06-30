BeginPackage["Compile`Core`Optimization`CopyPropagation`"]
CopyPropagationPass;

Begin["`Private`"] 

Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`InertInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`InvokeInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]




(*
 Don't delete CopyInstructions that are setting the value of a symbol.
 This is important because there is semantic information in this, ie for 
 mutability testing.  Once the mutability pass has run we can go ahead and
 delete them.
*)

manageVariableQ[var_] :=
	var["type"] =!= Undefined && (TrueQ[var["type"]["isNamedApplication", "PackedArray"]])

preserveVariableWrite[ fm_, inst_] :=
	manageVariableQ[inst["target"]] && !TrueQ[fm["programModule"]["getProperty", "processMutabilityRun", False]]





run[fm_, opts_] :=
	Module[{
		varsToCheck = CreateReference[{}],
		instsToRemove = {}
	},
		CreateInstructionVisitor[
			<|
				"visitCopyInstruction" -> Function[{st, inst},
					Module[{
						source,
						target,
						def,
						numRemoved,
						use,
						idx
					},
					source = inst["source"];
					target = inst["target"];
					Assert[ConstantValueQ[source] ||
						VariableQ[source] && Length[source["defs"]] === 1];
					Assert[Length[target["defs"]] === 1];
					Which[
						(*
						  If the target is captured and the source is constant don't remove
						*)
						target["getProperty", "isCapturedVariable", False] && ConstantValueQ[source] ||
						preserveVariableWrite[fm, inst],
						Null
						,
						
						(*
						  If the source is only used once (here in this instruction) and 
						  the source is only defined in one instruction, then we can replace 
						  the target of the instruction that defines the source with the target 
						  of this instruction and remove this instruction.
						*)
						VariableQ[source] &&
						Length[source["uses"]] === 1 &&
						CopyInstructionQ[source["def"]],
							def = source["def"];
							AssertThat["The def should be InstructionQ", def]["named", "def"]["satisfies", InstructionQ];
							def["setTarget", target];
							def["target"]["setDef", def];
							AppendTo[instsToRemove, inst];
							(*
							 source is being removed and being replaced by target
							*)
							checkClosure[ varsToCheck, source, target],
						(*
						  The target of the Load is only defined once (since we are in SSA form),
						  we can use the source everywhere the target was used.
						  Then we can remove this instruction.

						  We don't do this if the target of the copy is the target of the instruction.
						  This can happen for Phi instructions.
						*)
						True,
							numRemoved = 0;
							Do[
								AssertThat["The use should be InstructionQ", use]["named", "use"]["satisfies", InstructionQ];
								Which[
									use["id"] === inst["id"],
										Continue[],
									BranchInstructionQ[use] && use["isConditional"],
										numRemoved++;
										use["setCondition", source],
									use["hasOperands"] &&
									AllTrue[use["operands"], VariableQ[#] || ConstantValueQ[#] &] &&
									(!use["hasTarget"] || use["target"]["id"] =!= inst["source"]["id"]),
										Module[{operands, operand},
											operands = use["operands"];
											Table[
												operand = use["getOperand", idx];
												If[VariableQ[operand] && operand["sameQ", target],
													numRemoved++;
													use["setOperand", idx, inst["source"]]
												],
												{idx, Length[operands]}
											];
											If[ InertInstructionQ[use],
												operand = use["head"];
												If[VariableQ[operand] && operand["sameQ", target],
													numRemoved++;
													use["setHead", inst["source"]]
												]];
											If[ CallInstructionQ[use] || InvokeInstructionQ[use],
												operand = use["function"];
												If[VariableQ[operand] && operand["sameQ", target],
													numRemoved++;
													use["setFunction", inst["source"]]
												]];
										];
								],	
								{use, target["uses"]}
							];
							If[ numRemoved === Length[target["uses"]],
									(*
							 			target is being removed and being replaced by source
									*)
									checkClosure[ varsToCheck, target, source];
									AppendTo[instsToRemove, inst]];
					]
				]],
				"traverse" -> "reversePostOrder"
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		Map[ processClosure[fm, #]&, varsToCheck["get"]];
		#["unlink"]& /@ instsToRemove;
		fm
	]

(*
  If oldVar is captured then add to list
*)
checkClosure[ list_, oldVar_, newVar_] :=
	If[ oldVar["hasProperty", "capturedByVariables"],
			list["appendTo", {oldVar, newVar}]]


(*
 oldVar is a captured closure variable and is being removed for newVar
 	add capturedByVariables and isCapturedVariable properties to newVar
 	fix fm to have closureVariablesProvided of newVar not OldVar
 	fix capturedByVariables to point to newVar not to oldVar
 	fix fm of capatured variables to have closureVariablesConsumed of newVar not oldVar
*)
processClosure[ fm_, {oldVar_, newVar_}] :=
	Module[ {captured = oldVar["getProperty", "capturedByVariables", {}]},
		RemoveClosureCapteeProperties[ fm, oldVar];
		AddClosureCapteeProperties[ fm, newVar, captured];
		Map[ fixAliased[ oldVar, newVar, #]&, captured];
	]


(*
 oldVar is a captured closure variable and is being removed for newVar, aliased used to alias oldVar
 	fix aliasesVariable of alias to point to newVar not to oldVar
 	fix fm of alias to have capturedScopeVariables of newVar not oldVar
*)

fixAliased[ oldVar_, newVar_, aliased_] :=
	Module[ {fm},
		fm = aliased["def"]["basicBlock"]["functionModule"];
		RemoveClosureCapturerProperties[ fm, aliased, oldVar];
		AddClosureCapturerProperties[ fm, aliased, newVar];
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"CopyPropagation",
	"Propagates the loads to the uses if there is only one definition of the load."
];

CopyPropagationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		UsePass,
		DefPass
	}
|>];

RegisterPass[CopyPropagationPass]
]]

End[] 

EndPackage[]
