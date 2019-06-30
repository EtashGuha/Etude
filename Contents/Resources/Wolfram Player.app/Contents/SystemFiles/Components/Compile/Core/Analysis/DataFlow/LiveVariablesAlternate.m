
(**
  * One use is that if one traces a sequence of statements and records the types
  * then one only needs to include guards for variables that are live at the start
  * of the program. For example, global variables would are alive before the start
  * of the program.
  *)

BeginPackage["Compile`Core`Analysis`DataFlow`LiveVariablesAlternate`"]

LiveVariablesAlternatePass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["Compile`Core`Analysis`Dominator`StrictDominator`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]



fillUsesFrom[ usesFrom_, inst_] :=
	Module[ {bbs, vars},
		bbs = inst["getSourceBasicBlocks"];
		vars = inst["getSourceVariables"];
		Scan[
			Module[{var = First[#], bb = Last[#], uses},
				If[VariableQ[var],
					uses = usesFrom["lookup", bb["id"], Null];
					If[uses === Null,
						uses = CreateReference[<||>];
						usesFrom["associateTo", bb["id"] -> uses]];
					uses["associateTo", var["id"] -> var]];
			]&, Transpose[{vars,bbs}]]
	]


(*
 Compute the Gen and Kill sets for each BasicBlock.
 Set up the live[in] and live[out] sets for each BasicBlock. 
 All of these are AssociationReferences[ bbId -> AssociationReference[ varId -> var]].
 
 bbGenFrom records uses of variables in phi instructions.  These are recorded separately so 
 that we only add out to a BasicBlock for variables that were imported from that BB.
*)	
initializeData[ fm_] :=
	Module[ {bbGen, bbGenFrom, bbKill, liveIn, liveOut, var, instUsed},
		bbGen = CreateReference[<||>];
		bbGenFrom = CreateReference[<||>];
		bbKill = CreateReference[<||>];
		liveIn = CreateReference[<||>];
		liveOut = CreateReference[<||>];
		fm["reversePostOrderScan",
			Function[{bb},
				Module[ {uses, usesFrom, defs},
				uses = CreateReference[<||>];
				defs = CreateReference[<||>];
				usesFrom = CreateReference[<||>];
				bb[ "reverseScanInstructions",
					Function[ {inst},
						If[ PhiInstructionQ[inst],
							fillUsesFrom[usesFrom, inst];
							,
							instUsed = Association[(#["id"] -> #)& /@ inst["usedVariables"]];
							uses["join", instUsed]];
						If[ inst["definesVariableQ"],
							var = inst["definedVariable"];
							defs["associateTo", var["id"] -> var];
							(*
							 Drop variable from uses if it is defined,  it won't be live at the top.
							 I am going to assume that this can't happen in a Phi Instruction so we 
							 don't need to fix usesFrom.   Ie we can't have a definition for a variable 
							 for which there is an earlier PhiInstruction.
							*)
							uses["keyDropFrom", var["id"]];
						];
					]];
				bbGen["associateTo", bb["id"] -> uses];
				bbGenFrom["associateTo", bb["id"] -> usesFrom];
				bbKill["associateTo", bb["id"] -> defs];
				liveIn["associateTo", bb["id"] -> CreateReference[<||>]];
				liveOut["associateTo", bb["id"] -> CreateReference[<||>]];
				]
			]
		];
		<|
			"bbGen" -> bbGen, "bbGenFrom" -> bbGenFrom, "bbKill" -> bbKill, 
			"liveIn" -> liveIn, "liveOut" -> liveOut
		|>
	]	

(*
 Set the BasicBlock live[in]/live[out] properties and compute and store 
 properties for each instruction.  There is a lint test that these match.
*)	
finalizeResult[ fm_, data_] :=
	Module[ {},
		fm["reversePostOrderScan",
			Function[ {bb},
				Module[ {liveIn, liveOut, extraArgs,  liveVars, instOut},
					liveIn = data["liveIn"]["lookup", bb["id"]];
					extraArgs = data["bbGenFrom"]["lookup", bb["id"]]["values"];
					Scan[liveIn["join", #]&, extraArgs];
					bb[ "setProperty", "live[in]" -> liveIn["values"]];
					liveOut = data["liveOut"]["lookup", bb["id"]];
					bb[ "setProperty", "live[out]" -> liveOut["values"]];
					liveVars = liveOut;
					instOut = liveOut["clone"];
					bb["reverseScanInstructions", 
						addInstructionLive[instOut,#]&]; 
				];
		]];
	]


addInstructionLive[liveVars_, inst_] :=
	Module[ {usedVars},
		inst["setProperty", "live[out]" -> liveVars["values"]];
		usedVars = Association[(#["id"] -> #)& /@ inst["usedVariables"]];
		liveVars["join", usedVars];
		If[inst["definesVariableQ"],
			liveVars["keyDropFrom", inst["definedVariable"]["id"]]];
		inst["setProperty", 
			"live[in]" -> liveVars["values"]];

	]

joinOut[ data_, out_, bb_, bbChild_] :=
	Module[{},
		(*
		 This is the dominated list.
		*)
		out["join",data["liveIn"]["lookup",bbChild["id"]]];
		joinNonDominatedOut[data, out, bb, bbChild];
	]

joinNonDominatedOut[ data_, out_, bb_, bbChild_] :=
	Module[{args, vars},
		(*
		 This is the non-dominated list.
		*)
		args = data["bbGenFrom"]["lookup", bbChild["id"]];
		If[args["keyExistsQ", bb["id"]],
			vars = args["lookup", bb["id"]];
			out["join", vars];
		]
	]

run[fm_?FunctionModuleQ, opts_] :=
	Module[{worklist, bbId, out, comp, outOld, outClone, inOld, in, bb, data, gen, kill},
		data = initializeData[ fm];
		worklist = fm["postOrder"];
		While[worklist =!= {},
			{bb, worklist} = {First[worklist], Rest[worklist]}; 
			bbId = bb["id"];
			inOld = data["liveIn"]["lookup", bbId];
			outOld = data["liveOut"]["lookup", bbId];
			outClone = outOld["clone"];
			Scan[
				joinNonDominatedOut[data,outClone,bb,#]&,
				bb["getChildren"]];			
			gen = data["bbGen"]["lookup", bbId];
			kill = data["bbKill"]["lookup", bbId];
			comp = outClone["keyDropFrom", kill["keys"]];
			in = comp["join", gen];
			out = CreateReference[<||>];
			Scan[
				joinOut[data,out,bb,#]&,
				bb["getChildren"]];
			If[!refKeySameQ[out, outOld] || !refKeySameQ[in, inOld],
				data["liveIn"]["associateTo", bbId -> in];
				data["liveOut"]["associateTo", bbId -> out];
				worklist = DeleteDuplicates[Join[bb["getParents"], worklist]]
			];
		];
		finalizeResult[fm, data];
		fm
	]

(*
 Return True if the two Association Refs have identical Keys. 
 Avoids Sorting the keys.
*)
refKeySameQ[ refIn1_, refIn2_] :=
	Module[ {ref1},
		If[ Length[refIn1["keys"]] =!= Length[refIn2["keys"]],
			Return[False]];
		ref1 = refIn1["clone"];
		ref1["keyDropFrom", refIn2["keys"]];
		Length[ref1["keys"]] === 0
	]

	
SetAttributes[timeIt, HoldAllComplete]
accum = 0;
timeIt[e_] :=
	With[{t = AbsoluteTiming[e;][[1]]},
		accum += t;
		Print[StringTake[ToString[Unevaluated[e]], 10], "  t = ", t, "  accum = ", accum]
	]




(**********************************************************)
(**********************************************************)
(**********************************************************)

RegisterCallback["RegisterPass", Function[{st},

logger = CreateLogger["LiveVariablesAlternate", "INFO"];

info = CreatePassInformation[
	"LiveVariablesAlternate",
	"Computes the live variables of each instruction and basic block in the function module",
	"A definition `a` is live at point `b` if it is used after `b` (or subsequent instructions) "<>
	"and there is no intervening dominating definitions between a and b."
];

LiveVariablesAlternatePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		TopologicalOrderRenumberPass,
		DefPass,
		UsePass,
		StrictDominatorPass
	},
	"passClass" -> "Analysis"
|>];

RegisterPass[LiveVariablesAlternatePass];
]]


End[] 

EndPackage[]

