(*
   Search back to start recording whether variables can be reached by SetElement instructions.
   
   Based on LiveVariables pass.
*)

BeginPackage["Compile`Core`Optimization`RecordSetElement`"]

RecordSetElementPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`LoadInstruction`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Instruction`SetElementInstruction`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	Module[{},
		fm["scanBasicBlocks",
			Function[{bb},
				bb["setProperty", "elementsModified" -> CreateReference[<||>]];
			]
		]
	]

finalize[fm_, opts_] :=
	Module[{},
		fm["scanBasicBlocks",
			Function[{bb},
				Module[ {mods},
					mods = bb["getProperty", "elementsModified"];
					mods = mods["values"];
					bb["setProperty", "elementsModified" -> mods];
				]
			]
		]
	]
	


(*
 Pass through the basic blocks in postOrder.  
 For each basic block get the elements modified for the child basic blocks. 
 Then pass through the basic block collection an updated element modification list, and 
 add this to the basic block as a property.
 
*)

run[fm_?FunctionModuleQ, opts_] :=
	Module[{worklist, bb, mods, old},
		worklist = fm["postOrder"];
		While[worklist =!= {},
			
			{bb, worklist} = {First[worklist], Rest[worklist]}; 
			mods = bb["getProperty", "elementsModified"];
			old = mods["keys"];
			Scan[ mods["join", #["getProperty", "elementsModified"]]&, bb["getChildren"]];
			bb["reverseScanInstructions",
				Function[{inst},
					computeMods[ inst, mods]]];
			bb["setProperty", "elementsModified" -> mods];
			(*
			  If we modified the elementsModified of this bb, we have to revisit any parents 
			  because these use the info for this bb.
			*)
			If[old =!= bb["getProperty", "elementsModified"]["keys"],
				worklist = DeleteDuplicates[Join[worklist, bb["getParents"]]];
			]

		];
		fm
	]
	
SetAttributes[timeIt, HoldAllComplete]
accum = 0;
timeIt[e_] :=
	With[{t = AbsoluteTiming[e;][[1]]},
		accum += t;
		Print[StringTake[ToString[Unevaluated[e]], 10], "  t = ", t, "  accum = ", accum]
	]


(*
  Add the current set of mods to the instruction as an elementsModified property.
  Then compute the updated mods.

  If the instruction is a SetElementInstruction then add the target to the mods and
  set the target to have elementsModified property.
  
  If the instruction is a passInstruction (Copy, Load or Phi) then if the target is modified 
  we add the used variables to the mods and give them the elementsModified.  Or if one of the
  used vars is modified, then we add the target to the mods and give it the elementsModified.
  
  We do not do anything for a Store,  because this is dealt with by the IntroducePackedArrayCopy. 
  Perhaps we should check a property for this.
  
  Note the test for copyCheck,  this is introduced in CreateList and is to prevent a copy for thing like
  Compile[ {{n, _Integer}}, Table[ i, {i, n}]],  when the table is created the first element is set,
  this was causing a copy.  Not very satisfactory and the whole mechanism needs rethinking.
*)
computeMods[ inst_, mods_] :=
	Module[ {trgt, vars},
		If[ SetElementInstructionQ[inst] && inst["getProperty", "copyCheck", True],
			addVariable[inst["target"], mods]];
		If[passInstructionQ[inst],
			trgt = inst["target"];
			vars = inst["usedVariables"];
			If[ mods["keyExistsQ", trgt["id"]],
				Scan[addVariable[#,mods]&, vars]];
			Scan[
				If[mods["keyExistsQ", #["id"]],
					addVariable[trgt, mods]]&, vars]];
	]

addVariable[ var_?VariableQ, mods_] :=
	(
	var["setProperty", "elementsModified" -> True];
	mods["associateTo", var["id"] -> var];
	)

passInstructionQ[inst_] :=
	CopyInstructionQ[inst] || LoadInstructionQ[inst] || PhiInstructionQ[inst]


(**********************************************************)
(**********************************************************)
(**********************************************************)

RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["RecordSetElement", "INFO"];

info = CreatePassInformation[
	"RecordSetElement",
	"Eliminates any unecessary MTensor Copy instructions.",
	"The elementsModified property on a BasicBlock records any variables that this BasicBlock (or it's children) " <>
	"modify elements of with SetElementInstruction.  This is needed to keep immutability of MTensors."
];

RecordSetElementPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"finalizePass" -> finalize,
	"requires" -> {},
	"passClass" -> "Analysis"
|>];

RegisterPass[RecordSetElementPass]
]]

End[] 

EndPackage[]
