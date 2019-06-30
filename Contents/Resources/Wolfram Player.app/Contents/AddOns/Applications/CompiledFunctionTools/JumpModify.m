(* Mathematica Package *)

BeginPackage["CompiledFunctionTools`JumpModify`", {"CompiledFunctionTools`"}]
(* Exported symbols added here with SymbolName::usage *)  



Begin["`Private`"] (* Begin Private Context *) 


initInstructions[] :=
	Module[ {instStore},
		instStore[_] = Null;
		instStore
	]
	
deleteInstructions[ objData_]:=
	Apply[ Clear, {objData[ "instructions"]}]

initObjData[ objData_, inst_] :=
	Module[ {instStore},
		instStore[_] = Null;
		objData[ "instructions"] = initInstructions[];
		objData[ "done"][_] = False;
		objData[ "oldInstructions"] = inst;
		objData[ "instructionStack"] := {}
	]

	
saveInstructions[ objData_] :=
	(
	objData[ "instructionStack"] = {objData[ "instructionStack"], objData[ "instructions"]};
	objData[ "instructions"] = initInstructions[];
	)

restoreInstructions[ objData_, startIndex_, endIndex_] :=
	Module[ {insts, oldInsts, oldStore},
		insts = getInstructions[ objData, startIndex, endIndex];
		deleteInstructions[ objData];
		{oldStore, oldInsts} = objData[ "instructionStack"];
		objData[ "instructionStack"] = oldStore;
		objData[ "instructions"] = oldInsts;
		insts
	]
	
saveInstruction[ objData_, num_, inst_] :=
	(
	objData[ "done"][num] = True;
	objData[ "instructions"][num] = inst
	)

clearInstructions[ objData_, start_, end_] :=
	Do[ objData[ "instructions"][i] = Null, {i, start, end}]
	
	
getInstructions[ objData_, start_, end_] :=
	DeleteCases[ Table[ objData[ "instructions"][i], {i, start, end}], Null]

JumpModifyErrorHandler[ _, JumpModifyException[ txt_, line_]] :=
	Module[ {text},
		text = txt <> " at instruction " <> ToString[ line];
		Message[ JumpModify::err, txt]
	]

JumpModify[ cp_CompiledProcedure] :=
	Module[ {},
		Catch[ JumpModifyWorker[ cp], _JumpModifyException, JumpModifyErrorHandler]
	]

JumpModifyWorker[ CompiledProcedure[ci_CompiledInfo, cs_CompiledSetup, 
   cons_CompiledConstants,
   cr_CompiledResult, cInst_List, code_List]] :=
	Module[ {len, objData, newInst},
		len = Length[ cInst];
		initObjData[ objData, cInst];
		Do[ scanInstruction[ objData, i, cInst[[i]]], {i,len}];
		newInst = getInstructions[ objData, 1, len];
		CompiledProcedure[ ci, cs, cons, cr, Evaluate[newInst], code]
	]
	
	
scanInstruction[ objData_, num_ , inst_] /; !objData[ "done"][num] :=
	saveInstruction[ objData, num, inst]
	
	
getJumpTarget[ Instruction[ "Jump", Line[ lineNum_]]] :=
	lineNum
	
scanInstruction[ objData_, num_, Instruction[ "Branch", reg_, Line[ falseJumpNum_]]] /; !objData[ "done"][num] :=
	Module[ {startIndex, endIndex, endTrueInst, trueJumpNum, origInsts, trueInsts, falseInsts, condInsts},
		objData[ "done"][num] = True;
		startIndex = num+1;
		endIndex = falseJumpNum-2;
		endTrueInst = Part[ objData[ "oldInstructions"], endIndex+1];
		trueJumpNum = getJumpTarget[ endTrueInst];
		saveInstructions[ objData];
		origInsts = objData[ "oldInstructions"];
		Do[ scanInstruction[ objData, i, origInsts[[i]]], {i,startIndex, endIndex}];
		objData[ "done"][endIndex+1] = True;
		trueInsts = restoreInstructions[ objData, startIndex, endIndex];
		If[ trueJumpNum < num,  (*  This is a While *)
			condInsts = getInstructions[ objData, trueJumpNum, num-1];
			clearInstructions[ objData, trueJumpNum, num-1];
			saveInstruction[ objData, num, InstructionWhile[ condInsts, reg, trueInsts]]
			,
			If[ trueJumpNum === falseJumpNum,  (*  if no false clause *)
				saveInstruction[ objData, num, InstructionIf[ reg, trueInsts]],
				startIndex = falseJumpNum;
				endIndex = trueJumpNum-1;
				saveInstructions[ objData];
				Do[ scanInstruction[ objData, i, origInsts[[i]]], {i,startIndex, endIndex}];
				falseInsts = restoreInstructions[ objData, startIndex, endIndex];
				saveInstruction[ objData, num, InstructionIf[ reg, trueInsts, falseInsts]]
			]
		]
	]


(*
 We could probably sweep up the init for the loop as well.
*)

scanInstruction[ objData_, num_, inst:Instruction[ "Jump", Line[ incrLineNum_]]] /; !objData[ "done"][num] :=
	Module[ {targetInst, dummy, regCount, regLim, origInsts, start, end, bodyInsts, regInit},
		targetInst = Part[ objData[ "oldInstructions"], incrLineNum];
		If[ !MatchQ[ targetInst, Instruction[ "LoopIncr", __]], 
				saveInstruction[ objData, num, inst];   (* It's not a loop,  perhaps this should be an error *)
				Return[]];
		{dummy, {regCount, regLim}, start} = Apply[ List, targetInst];
		regInit = getLoopInit[ objData, regCount, num];
		end = incrLineNum-1;
		saveInstructions[ objData];
		origInsts = objData[ "oldInstructions"];
		Do[ scanInstruction[ objData, i, origInsts[[i]]], {i,start, end}];
		objData[ "done"][num] = True;
		objData[ "done"][incrLineNum] = True;
		bodyInsts = restoreInstructions[ objData, start, end];
		saveInstruction[ objData, num, InstructionFor[ regCount, regInit, regLim, bodyInsts]];
	]

(*
 Return init register
*)
getLoopInit[ objData_, regCount_, num_] :=
	Module[ {initInst, setPos},
		setPos = num-1;
		initInst = objData[ "oldInstructions"][[setPos]];
		If[ !MatchQ[ initInst, Instruction[ Set, regCount, Register[ Integer, _]]],
			Throw[ Null, JumpModifyException["LoopIncr missing Set", num]]];
		clearInstructions[ objData, setPos, setPos];
		initInst[[ 3]]
	]





End[] (* End Private Context *)

EndPackage[]