


BeginPackage["Compile`Core`Analysis`DataFlow`AliasSets`"]

AliasSetsPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Analysis`DataFlow`LiveVariables`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`Transform`ConstantArrayPromote`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`Analysis`Properties`ClosureVariablesProvided`"]

(*
  Computation of AliasSets,  following on from 
  
  Efficient Alias Set Analysis Using SSA Form
  Nomair A. Naeem and Ondrej Lhotak
  D. R. Cheriton School of Computer Science University of Waterloo, Canada
  
  
*)

(*
  If the variable is managed return True and False otherwise.
  This could be set to work in a more general way.
*)
manageVariableQ[var_] :=
	var["type"] =!= Undefined && 
		(TrueQ[var["type"]["isNamedApplication", "PackedArray"]])

(*
  v1 is being added,  remove it from any alias sets.
  Prune any empty sets.
*)
visitNew[state_, in_, v1Id_] :=
	Module[ {data},
		data = Map[ DeleteCases[#, v1Id]&, in];
		data = DeleteCases[data, {}];
		data
	]

(*
 Return true if any of the aliases contain vId
*)
aliasPresentQ[ state_, aliases_, vId_] :=
	AnyTrue[ aliases, MemberQ[#, vId]&]


(*
   v1 is assigned to v2.
   If v2 is present in an alias set then add v1,
   else if v2 is not present in an alias set then remove v1.
   Prune any empty sets.
*)
visitCopyPair[state_, in_, v1Id_, v2Id_] :=
	Module[ {data},
		data = DeleteCases[Map[ If[MemberQ[ #, v2Id], DeleteDuplicates[Append[#, v1Id]], DeleteCases[#, v1Id]]&, in], {}];
		data
	]


visitSpecific[ state_, in_, inst_?LoadArgumentInstructionQ] :=
	visitNewVariable[state, in, inst]

(*
  New variable, if it is managed add a new set for the target Var.
*)
visitNewVariable[ state_, in_, inst_] :=
	Module[ {trgt, trgtId, out},
		trgt = inst["target"];
		If[!manageVariableQ[trgt],
			Return[in]];
		addVariable[state, trgt];
		trgtId = trgt["id"];
		out = Map[ visitNew[ state, #, trgtId]&, in];
		out = Join[ {{trgtId}}, out];
		out
	]



(*
  New variable, if it is managed add a new set for the target Var.
  Merge implementation with LoadArgumentInstructionQ
*)
visitSpecific[ state_, in_, inst_?CallInstructionQ] :=
	Module[ {trgt, trgtId, out},
		trgt = inst["target"];
		If[!manageVariableQ[trgt],
			Return[in]];
		Which[ 
			LoadClosureVariableCallQ[ inst["function"]],
				visitClosure[state, in, inst]
			,
			functionCanAliasArgument[ inst["function"]],
				visitFunctionAliasArgument[state, in, inst]				
			,
			True,
				addVariable[state, trgt];
				trgtId = trgt["id"];
				out = Map[ visitNew[ state, #, trgtId]&, in];
				out = Join[ {{trgtId}}, out];
				out]
	]


(*
 inst is a function that might alias it's arguments
*)
visitFunctionAliasArgument[state_, in_, inst_] :=
	Module[{args = Select[inst["arguments"], manageVariableQ], trgID = inst["target"]["id"]},
		Fold[ visitCopyPair[state, #1, trgID, #2["id"]]&, in, args]
	]

(*
  Look in the definition to see if "ArgumentAlias" is True. 
  Later we can probably  generate this automatically.
*)
functionCanAliasArgument[ function_] :=
	Module[{def = function["getProperty", "definition", <||>]},
		TrueQ[Lookup[def, "ArgumentAlias", False]]
	]


(*
  inst is a LoadClosureVariable call
*)
visitClosure[state_, inIn_, inst_] :=
	Module[{trgt = inst["target"], aliasedVar, trgId, aliasedVarId, out, in = inIn},
		aliasedVar = trgt["getProperty", "aliasesVariable", Null];
		If[ aliasedVar === Null,
			ThrowException[{"aliasesVariable property has not been set in LoadClosureVariable call", {inst}}]];
		trgId = trgt["id"];
		aliasedVarId = aliasedVar["id"];
		addVariable[state, trgt];
		addVariable[state, aliasedVar];
		If[ !aliasPresentQ[state, in, aliasedVarId],
			in = Join[ {{aliasedVarId}}, in]];
		out = visitCopyPair[ state, in, trgId, aliasedVarId];
		out
	]

(*
  Copy instruction,  if the source is a constant then treat as a new variable
  to prevent modifications of the constant.  Otherwise call visitCopyPair.
*)
visitSpecific[state_, in_, inst_?CopyInstructionQ] :=
	Module[{trgt, src, out, srcId, trgId},
		trgt = inst["target"];
		If[!manageVariableQ[trgt],
			Return[in]];
		src = inst["source"];
		If[ ConstantValueQ[src],
			out = visitNewVariable[state, in, inst];
			,
			trgId = trgt["id"];
			srcId = src["id"];
			addVariable[state, trgt];
			addVariable[state, src];
			out = visitCopyPair[ state, in, trgId, srcId]];
		out
	]


(*
  Other instruction,  do nothing.
*)
visitSpecific[state_, in_, inst_] :=
	in

(*
  Remove any variables that are not live on output.
  We also include the bbFlowThrough variables,  these 
  are not live in any way in this BB,  but are live in the 
  output of one of the parents.  The worst that could result
  from this is an overestimate of the alias set.
  Prune any empty sets.
*)
removeDeadVariables[state_, in_, inst_] :=
	Module[{liveOut, out, flowThroughs = state["bbFlowThrough"]["lookup", inst["basicBlock"]["id"], {}]},
		liveOut = inst["getProperty", "live[out]", Null];
		If[liveOut === Null,
			Return[in]];
		liveOut = Map[#["id"]&, liveOut];
		liveOut = Join[liveOut, flowThroughs, state["arguments"]["get"], state["closureVariablesConsumed"]];
		out = Map[ Intersection[#, liveOut]&, in];
		out = DeleteCases[out, {}];
		out
	]


(*
  visiting a Phi instruction.  Get the aliasSetsOut of the source block
  and fold in the target/source variable as a copy instruction.
*)
fixPhi[ state_, inst_, {lastInst_, src_}] :=
	Module[ {in = lastInst["getProperty", "aliasSetsOut"]},
		If[!manageVariableQ[src],
			Return[in]];
		visitCopyPair[state, in, inst["target"]["id"], src["id"]]
	]


mergeLists[ list1_, list2_] :=
 	Fold[mergeElement, list1, list2]

mergeElement[s1_, s2_] :=
	Append[Last[#], First[#]] &[Fold[mergeFun, {s2, {}}, s1]]

mergeFun[{test_, extra_}, candidate_] :=
	If[ContainsAny[test, candidate], 
  		{DeleteDuplicates[Join[test, candidate]], extra},
  		{test, Append[extra, candidate]}]

(*
 Run the computation for one instruction.  
 We take the in state from the out of previous instruction.
 If it is a PhiInstruction we need also to take the out from 
 the last instruction of the source BB.  This is held in the phiMap.
 
 If the alias set doesn't change then we have finished with this instruction.
*)
runAnalysisIterate[state_] :=
	Module[ {inst = state["instructionList"]["popFront"], prev, in, out, outNew, phiData},
		prev = state["predecessorMap"]["lookup", inst["id"], Null];
		in = If[ prev === Null, {}, prev["getProperty", "aliasSetsOut"]];
		If[ PhiInstructionQ[inst],
			phiData = state["phiMap"]["lookup", inst["id"]];
			phiData = Map[ fixPhi[state, inst, #]&, phiData]; 
			outNew = Fold[ mergeLists, in, phiData];
			,
			outNew = visitSpecific[ state, in, inst]];	
		out = inst["getProperty", "aliasSetsOut"];
		If[state["removeDeadVariables"],
			outNew = removeDeadVariables[state, outNew, inst]];
		inst["setProperty", "aliasSetsOut" -> outNew];
		If[ !ContainsExactly[out, outNew, SameTest -> ContainsExactly],
			state["instructionList"]["pushBack", inst]]
	]

(*
 As long as there are instructions that can change keep running.
*)
runAnalysis[state_] :=
	Module[ {},
		While[ state["instructionList"]["length"] > 0,
			runAnalysisIterate[state]]
	]

(*
 Return a pair of last instruction, variable for each source of the PhiInstruction.
*)
initPhi[state_, {bb_, var_}] :=
	Module[ {lastInst = bb["lastInstruction"]},
		{lastInst, var}
	]
	
(*
 Set up the phiMap for each Phi instruction.
*)
initializePhiInstruction[ state_, inst_] :=
	Module[ {instId, phiData = inst["getSourceData"], initData},
		instId = inst["id"];
		initData = Map[ initPhi[state, #]&, phiData];
		state["phiMap"]["associateTo", instId -> initData];
	]


constantPackedArray[inst_] :=
	ConstantValueQ[inst["source"]] && inst["source"]["type"]["isNamedApplication", "PackedArray"]

(*
  Initialize the alias sets properties. Also set up the phiMap.
  Also add argument data,  we want to make sure we find aliases 
  for any arguments,  this makes sure that arguments are not aliased. 
*)
initializeInstruction[ state_, inst_, prevInst_] :=
	Module[ {instId},
		instId = inst["id"];
		addInstruction[state, inst];
		inst["setProperty", "aliasSetsOut" -> {}];
		state["instructionList"]["appendTo", inst];
		state["predecessorMap"]["associateTo", instId -> prevInst];
		If[ PhiInstructionQ[inst],
			initializePhiInstruction[state, inst]];
		(*
		  Treat constant packed arrays as arguments
		*)
		If[ LoadArgumentInstructionQ[inst],
			state["arguments"]["appendTo", inst["target"]["id"]]];
		If[ CopyInstructionQ[inst] && constantPackedArray[inst],
			state["arguments"]["appendTo", inst["target"]["id"]]];
		inst
	]

(*
  Compute any variables that are in live[out] of parents that are not in the 
  live[in] and live[out] of bb.  These are the FlowThrough variables.
  They need to be included in the computation for Aliasing in loops.
*)
setBBFlowThrough[ state_, bb_, parents_] :=
	Module[ {liveOuts, liveOut, liveIn},
		liveOuts = Flatten[Map[ #["getProperty", "live[out]", {}]&, parents]];
		liveIn = bb["getProperty", "live[in]", {}];
		liveOut = bb["getProperty", "live[out]", {}];
		liveOuts = Map[#["id"]&, liveOuts];
		liveIn = Map[#["id"]&, liveIn];
		liveOut = Map[#["id"]&, liveOut];
		liveOuts = Complement[liveOuts, liveIn, liveOut];
		state["bbFlowThrough"]["associateTo", bb["id"] -> liveOuts];
	]


(*
 initialize a basic block.  Pick out all of the phi instructions first.
*)
initializeBasicBlock[ state_, bb_] :=
	Module[ {parents = bb["parents"]["get"], instList, instGrp, phiInsts, otherInsts, phiData, newInst, prevInst = Null},
		
		setBBFlowThrough[ state, bb, parents];
		instList = bb["getInstructions"];
		If[ LabelInstructionQ[First[ instList]],
			First[ instList]["setProperty", "aliasSetsOut" -> {}];
			instList = Rest[instList]];
			
		instGrp = GroupBy[ instList, PhiInstructionQ];
		phiInsts = Lookup[instGrp, True, {}];
		otherInsts = Lookup[instGrp, False, {}];
		
		Which[
			Length[parents] === 0,
				phiInsts = {}
			,
			Length[phiInsts] === 0,
				phiData =  Table[{Part[parents,i], CreateConstantValue[1]}, {i,Length[parents]}];
				newInst = CreatePhiInstruction[ "newPhi", phiData];
				newInst["setId", state["counter"]["increment"]];
				phiInsts = {newInst}
			,
			True,
				Null
			];
		Scan[
			(prevInst = initializeInstruction[state, #, prevInst])&, phiInsts];	
		Scan[
			(prevInst = initializeInstruction[state, #, prevInst])&, otherInsts];
	]


isClosureConsumed[var_] :=
	var["getProperty", "isClosureVariable", False]
	
isClosureProvided[var_] :=
	var["getProperty", "isCapturedVariable", False]


(*
  Run a filter that the alias doesn't alias the same symbol taking care of 
  closure info.  Perhaps this would be better done in the ProcessMutability 
  pass, but for now,  let's leave here since we'd have to convert to var 
  from the varId.
*)

filterSameSymbol[state_, set_List]:=
	Module[{vars, var, varValue},
		If[Length[set] < 2,
			Return[set]];
		vars = Map[state["variableMap"]["lookup",#]&, set];
		var = First[vars];
		varValue = var["getProperty", "variableValue", Null];
		If[ varValue === Null || isClosureConsumed[var] || isClosureProvided[var],
			Return[set]];
		If[AllTrue[vars, (#["getProperty", "variableValue"] === varValue && 
					!isClosureConsumed[var] && !isClosureProvided[var]) &],
			{First[set]},
			set]
	]


filterAlias[ state_, setsIn_] :=
	Module[{sets},
		sets = Map[ filterSameSymbol[state, #]&, setsIn];
		DeleteCases[sets, {_}]
	]

(*
  Set the alias sets properties.
*)
finalizeInstruction[state_, inst_] :=
	Module[ {prev = state["predecessorMap"]["lookup", inst["id"], Null], in, tmp},
		tmp = inst["getProperty", "aliasSetsOut"];
		tmp = filterAlias[ state, tmp];
		inst["setProperty", "aliasSetsOut" -> tmp];
		tmp = inst["getProperty", "aliasSetsIn", Null];
		If[ListQ[tmp],
			tmp = filterAlias[state, tmp];
			inst["setProperty", "aliasSetsIn" -> tmp]];
		If[prev =!= Null,
			in = prev["getProperty", "aliasSetsOut"];
			If[ListQ[in],
				in = filterAlias[state, in];
				inst["setProperty", "aliasSetsIn" -> in]]]
	]

finalizeBasicBlock[ state_, bb_] :=
	Module[ {},
		bb["scanInstructions",
			finalizeInstruction[state, #]&];
	]



addVariable[state_, var_] :=
	state["variableMap"]["associateTo", var["id"] -> var]

addInstruction[state_, inst_] :=
	state["instructionMap"]["associateTo", inst["id"] -> inst]

createState[fm_, opts_] :=
	Module[ {consumed = fm["getProperty", "closureVariablesConsumed", Null]},
		consumed = Map[#["id"]&, If[ consumed === Null, {}, consumed["get"]]];
		<|
		"counter" -> CreateReference[1],
		"arguments" -> CreateReference[{}],
		"closureVariablesConsumed" -> consumed,
		"pointsData" -> CreateReference[], 
		"variableMap" -> CreateReference[<||>],
		"instructionList" -> CreateReference[{}],
		"predecessorMap" -> CreateReference[<||>],
		"phiMap" -> CreateReference[<||>],
		"instructionMap" -> CreateReference[<||>],
		"removeDeadVariables" -> Lookup[opts, "RemoveDeadVariables", True],
		"bbFlowThrough"  -> CreateReference[<||>]
		|>
	]
	
setState[state_, inst_] :=
	Module[{id = inst["id"], cnt = state["counter"]["get"]},
		If[ id >= cnt,
			state["counter"]["set", id+1]]
	]

run[fm_?FunctionModuleQ, opts_] :=
	Module[{state, visitor},	
		state = createState[ fm, opts];
		visitor = CreateInstructionVisitor[ state, <|"visitInstruction" -> setState|>];
		visitor["traverse", fm];
		fm[ "topologicalOrderScan",
			initializeBasicBlock[state, #]&];
		runAnalysis[ state];
		fm[ "topologicalOrderScan",
			finalizeBasicBlock[state, #]&];
		fm
	]
	



(**********************************************************)
(**********************************************************)
(**********************************************************)



RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["AliasSets", "INFO"];

info = CreatePassInformation[
	"AliasSets",
	"Computes alias sets information for SSA variables."
];

AliasSetsPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		ConstantArrayPromotePass,
		LiveVariablesPass,
		TopologicalOrderRenumberPass,
		ClosureVariablesProvidedPass
	},
	"passClass" -> "Analysis"
|>];

RegisterPass[AliasSetsPass];
]]


End[] 

EndPackage[]
