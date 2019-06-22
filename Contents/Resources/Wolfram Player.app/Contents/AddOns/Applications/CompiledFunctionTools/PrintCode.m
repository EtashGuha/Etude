(* Mathematica Package *)

BeginPackage["CompiledFunctionTools`PrintCode`", {"CompiledFunctionTools`"}]
(* Exported symbols added here with SymbolName::usage *)  



Begin["`Private`"] (* Begin Private Context *) 


returnChar = "\n"

indentChar = "\t"


(*Utilities for output of characters.*)
printReturn[] := printExpr[returnChar]
printExpr[Rule[a_,b_]]:=(printExpr[a]; printExpr[" -> "]; printExpr[b];)
printExpr[x_List /; First[Union[Map[NumericQ, x]]]] := 
	Sow[ToString[x], "PrintCode"]
printExpr[x_List] := Scan[printExpr, x]
printExpr[x_Register]:=printRegister[x]
printExpr[x_] := Sow[ToString[x], "PrintCode"]

getRegisterType[ Register[ type_, num_]] := type
getRegisterType[ Register[ MTensor[type_,rank_], num_]] := type

(*
 TODO,  get this to print with *10^ notation
*)
printExpr[x_Real] := Sow[ToString[x, InputForm], "PrintCode"]




(*Main entry point,we recurse doing printElement on the \
CompiledProcedure,we Reap the result and call StringJoin to return \
the result.*)
Options[ CompiledProcedureToString] = {ShowInstructions -> False}
CompiledProcedureToString[cp_CompiledProcedure, opts___Rule] := 
	CompiledProcedureToString[cp, ReplaceAll[ReplaceAll[ShowInstructions,{opts}], Options[ CompiledProcedureToString]]]

Options[ CompiledFunctionToString] = {ShowInstructions -> False}
CompiledFunctionToString[cf_CompiledFunction, opts___Rule] :=
	CompiledProcedureToString[ToCompiledProcedure[cf],opts]
	
CompiledProcedureToString[cp_CompiledProcedure, bool_] := 
  Apply[StringJoin, Reap[printElement[cp, bool], "PrintCode"][[2, 1]]]

Options[ CompilePrint] = {ShowInstructions -> False}
CompilePrint[ cf_CompiledFunction, opts___Rule] :=
	CompilePrint[ToCompiledProcedure[cf],opts]
	
CompilePrint[cp_CompiledProcedure, opts___] := 
	Module[ {bool},
		bool = ShowInstructions /. {opts} /. Options[ CompilePrint];
		If[ !MatchQ[ bool, True|False], bool = False];
		Apply[StringJoin, Reap[printElement[cp, bool], "PrintCode"][[2, 1]]]
	]


nestDepth = 0;


printIndent[ ] :=
	Module[ {indentNum = nestDepth+1},
		Do[
			printExpr[indentChar], {indentNum}]
	]

(*printElement on the CompiledProcedure body.Process the \
info,setup,result,then do the instructions.*)


printElement[
  CompiledProcedure[ci_CompiledInfo, cs_CompiledSetup, 
   cons_CompiledConstants,
   cr_CompiledResult, cInst_List, code_List], bool_] := 
 	Module[{long=Max[Map[StringLength[ToString[#]]&,code]]},
		nestDepth = 0;
  		printElement[ci];
  		printElement[cs];
  		printElement[cons];
  		printElement[cr];
  		printReturn[];  	
  		Do[printExpr[i];
    		printIndent[];
    		If[bool, printExpr[{ToString[code[[i]]], Apply[StringJoin,Table[" ",{long-StringLength[ToString[code[[i]]]]+10}]]}]];
    		printInstruction[cInst[[i]]];
    		printReturn[], {i, Length[cInst]}]
    ]

(*printElement on the CompiledInfo.*)
printElement[cinfo:CompiledInfo[arg_List, list_List, "RuntimeFlags"->rtFlags_]] := 
 Block[{liste,indent, rets, aflags},
  indent = Table[indentChar, {2}];
  rets = Table[returnChar, {1}];
  liste = Delete[list, 
    Position[Map[Not[FreeQ[#, 0]] &, list], True]];
  printExpr[rets];
  printExpr[indent];  	
  printExpr[StringReplace[ToString[Length[arg]]<>StringDrop[" arguments",-(1-Sign[Length[arg]-1])],StringExpression[StartOfString,"0",__]->"No argument"]];
  Scan[(
	 printExpr[rets];
     printExpr[indent];
     printExpr[ToString[#[[2]]]<>" "<>typeToString[#[[1]]]<>StringDrop[" registers",-(1-Sign[#[[2]]-1])]]) &, liste];
  aflags = cinfo["ArithmeticFlags"];   
  MapIndexed[Function[
  		printExpr[rets];
  		printExpr[indent];
  		printExpr[#1<>" "<>If[BitAnd[aflags, 2^(#2[[1]] - 1)] === 0, "off", "on"]]],
  	{"Underflow checking", "Overflow checking", "Integer overflow checking"}
  ];
  printExpr[rets];
  printExpr[indent];
  printExpr[RuntimeAttributes->ToString[cinfo["RuntimeAttributes"]]];
  printReturn[];]

typeToString[MTensor] := "Tensor"
typeToString[type_] := ToString[type]

(*printElement on the CompiledSetup.*)
printElement[CompiledSetup[setlist_List]] := 
	Module[{indent, rets}, 
		rets = Table[returnChar, {1}];
  		indent = Table[indentChar, {2}];
  		Scan[(printExpr[rets];
     			printExpr[indent];
     			printInstruction[#]) &, setlist];
  	]
  
(*printElement on the CompiledConstants.*)
printElement[CompiledConstants[setlist_List]] := 
	Module[{indent, rets}, 
		rets = Table[returnChar, {1}];
  		indent = Table[indentChar, {2}];
  		Scan[(printExpr[rets];
     			printExpr[indent];
     			printInstruction[#]) &, setlist];
  		printReturn[];
  	]
  
  
  

(*printElement on the CompiledResult.*)
printElement[CompiledResult[reg_]] := 
 Block[{}, printExpr[Table[indentChar, {2}]];
  printExpr["Result = "];
  printRegister[reg];
  printReturn[];]

(*printElement on something unknown.*)
(*printElement[x___] := printExpr["Unknown"[x]]*)


(*Register,argument,and type printing utilities*)
registerString[Register[type_ /; Not[MemberQ[{MTensor}, type]], num_]] := toShortName[type]<>ToString[num];
registerString[Register[MTensor[type_, rank_], num_]] := "T("<>toShortName[getTypeName[type]]<>ToString[rank]<> ")";

printRegister[op_List]:=Scan[printRegister,op]
printRegister[Register[type_ /; Not[MemberQ[{MTensor}, type]], num_]] :=
  printExpr[{toShortName[type], num}]
printRegister[Register[MTensor[type_, rank_], num_]] := 
 printExpr[{"T(", toShortName[getTypeName[type]], ToString[rank], ")",
    num}]
printRegister[Null]:=printExpr[""]

printRegister[Argument[num_]] := printExpr[{"A", num}]

toShortName[Boolean] := "B"
toShortName[Integer] := "I"
toShortName[Real] := "R"
toShortName[Complex] := "C"
toShortName[MTensor[type_Symbol, rank_]] := 
 "T(" <> toShortName[type] <> ToString[rank] <> ")"
toShortName[VoidType] := "V"

toInfixForm[Plus] = "+"
toInfixForm[Minus] = "-"
toInfixForm[Times] = "*"
toInfixForm[Equal] = "=="
toInfixForm[Less] = "<"
toInfixForm[LessEqual] = "<="
toInfixForm[Not] = "!"
toInfixForm[Unequal] = "!="
toInfixForm[Greater] = ">"
toInfixForm[GreaterEqual] = ">="

toInfixForm[_] := None



(*Actually print out the various instructions*)
printInstruction[i : Instruction[___]] := 
 printExpr["UnknownInstruction"[i]]

printInstruction[Instruction["Version", num_Integer]] := 
 printExpr[{"CF Version ", num}]
 
printInstruction[Instruction["SetElement", regTen_, regPos_, regValue_]]:=
	(
 	printExpr["Element[ "]; 
 	printRegister[regTen]; 
 	printExpr[", "]; 
 	printRegister[regPos]; 
 	printExpr["] = "];
 	printRegister[regValue];
 	)





printInstruction[Instruction[version_], num_Integer] := 
 printExpr[{"CF Version ", num}]

printInstruction[Instruction["SetPart", regint_, {tens_, int__}]]:=
	(
	printExpr["Part[ "]; 
	printRegister[tens]; 
	Do[(printExpr[", "]; 
		printRegister[{int}[[i]]];),{i,Length[{int}]}];
	printExpr["] = "]; 
	printRegister[regint];
	)

printInstruction[Instruction["Jump", Line[line_]]]:=(printExpr["goto "];printExpr[line];)

printInstruction[Instruction["StuffBag", reg_List]]:=(printExpr["StuffBag[ "]; Do[(printRegister[reg[[i]]]; printExpr[", "];),
	{i,1,Length[reg]-1}]; printRegister[Last[reg]]; printExpr["]"];)

printInstruction[Instruction["LoopIncr", {reg1_, reg2_}, goto_]]:=(printExpr["if[ ++ "];printRegister[reg1];
	printExpr[" <= "]; printRegister[reg2]; printExpr[{"] goto ",ToString[goto]}];)

printInstruction[Instruction[AllOuter[type_String], rego_, regs_]]:=(printRegister[rego]; printExpr["= Outer ["<>type]; 
    Do[(printExpr[", "]; printRegister[regs[[i]]]), {i,1,Length[regs]}]; printExpr[" ]"];)

printInstruction[
  Instruction["SetComplex", 
   rego_Register, {reg1_Register, reg2_Register}]] := (printRegister[
   rego]; printExpr[" = "]; printRegister[reg1]; printExpr[" + "];
  printRegister[reg2]; printExpr[" I"];)

printInstruction[
  Instruction[Set, lhs_Register, 
   rhs_ConstantTensor]] := (printRegister[lhs]; printExpr[" = "];
  printExpr[ToString[First[rhs]]];)

printInstruction[
  Instruction[Set, lhs_Register, 
   rhs_ /; Not[
     MemberQ[{Register, Argument}, Head[rhs]]]]] := (printRegister[
   lhs]; printExpr[" = "]; printExpr[rhs];)

printInstruction[
  Instruction[Set, lhs_Register, rhs_Argument]] := (printRegister[
   lhs]; printExpr[" = "]; printRegister[rhs];)

printInstruction[
  Instruction[Set, lhs_Register, rhs_Register]] := (printRegister[
   lhs]; printExpr[" = "]; printRegister[rhs];)

printInstruction[Instruction["Branch", regbool_, line_Line]] := 
	(
	printExpr["if[ !B"];
	printExpr[Last[regbool]];
	printExpr["] goto "];
	printExpr[Last[line]];
	)


printInstruction[InstructionIf[regbool_, trueInst_, falseInst_:Null]] := 
	Module[ {depth},
		depth = nestDepth;
		printExpr["if[ !B"];
		printExpr[Last[regbool]];
		printExpr["] { "];
		printReturn[];
		nestDepth = nestDepth + 1;
		Do[ 
			printIndent[];
			printInstruction[ trueInst[[i]]];
			printReturn[], {i, Length[ trueInst]}];
		If[ falseInst =!= Null,
			nestDepth = depth;
			printIndent[];
			printExpr["} else {"];
			printReturn[];
			nestDepth = nestDepth + 1;
			Do[
				printIndent[];
				printInstruction[ falseInst[[i]]];
				printReturn[], {i, Length[ falseInst]}];
		];
		nestDepth = depth;
		printIndent[];
		printExpr["}"];
	]

printInstruction[InstructionWhile[condInst_, regbool_, bodyInst_]] := 
	Module[ {depth, insts},
		Do[ 
			printIndent[];
			printInstruction[ condInst[[i]]];
			printReturn[], {i, Length[ condInst]}];
		depth = nestDepth;
		insts = Join[ bodyInst, condInst];
		printIndent[];
		printExpr["while[ B"];
		printExpr[Last[regbool]];
		printExpr["] { "];
		printReturn[];
		nestDepth = nestDepth + 1;
		Do[ 
			printIndent[];
			printInstruction[ insts[[i]]];
			printReturn[], {i, Length[ insts]}];
		nestDepth = depth;
		printIndent[];
		printExpr["}"];
	]

printInstruction[InstructionFor[regIncr_, regInit_, regLimit_, bodyInst_]] := 
	Module[ {depth},
		depth = nestDepth;
		printExpr["for[ "];
		printRegister[regIncr];
		printExpr[" = "];
		printRegister[regInit];
		printExpr["; "];
		printRegister[regIncr];
		printExpr[" < "];
		printRegister[regLimit];
		printExpr["; "];
		printRegister[regIncr];
		printExpr["++"];
		printExpr[" ] {"];
		printReturn[];
		nestDepth = nestDepth + 1;
		Do[ 
			printIndent[];
			printInstruction[ bodyInst[[i]]];
			printReturn[], {i, Length[ bodyInst]}];
		nestDepth = depth;
		printIndent[];
		printExpr["}"];

	]


printInstruction[Instruction[BlockRandom, regOut_, {be_Integer}]]:=
	If[be == 0,
		printExpr["BlockRandomEnd"],
		printExpr["BlockRandomBegin"]
	]

printInstruction[Instruction[SeedRandom, regOut_, args_]]:=
(
	printExpr["SeedRandom["];
	If[Length[args] > 0, 
		Do[printRegister[arg], {arg, args}]
	];
	printExpr["]"]
)

printInstruction[Instruction[op_List, rego_, regs_List]]:=
	Module[
	{tab=Insert[op,"[ ",(Range[Length[op]]+1)/.x_Integer->{x}],
	 reg=Insert[regs,", ",(Range[Length[regs]-1]+1)/.x_Integer->{x}]},
		printRegister[rego];
		printExpr[" = "];
		printExpr[tab];
		printExpr[reg];
		Do[printExpr["]"],{Length[op]}];
	]


printInstruction[Instruction["MainEvaluate", function_, regres_, reg_Register]] :=
	(
	printRegister[regres];
	printExpr[" = MainEvaluate[ "];
	printExpr[ToString[function]]; 
	printExpr["["]; 
	printRegister[reg]; 
	printExpr["]]"];
	)
	
printInstruction[Instruction["MainEvaluate", function_, regres_, reg_List]]:=
	(
	printRegister[regres];
	printExpr[" = MainEvaluate[ "];
	printExpr[ToString[function]]; 
	printExpr["[ "]; 
	Do[(printRegister[reg[[i]]]; printExpr[", "];), {i,Length[reg]-1}]; 
	If[reg=!={},printRegister[Last[reg]]];
	printExpr["]]"];
	)

printInstruction[Instruction["CompiledFunctionCall", function_, regres_, reg_List]]:=
	(
	printRegister[regres];
	printExpr[" = CompiledFunctionCall[ "];
	printExpr[ToString[function]]; 
	printExpr["[ "]; 
	Do[(printRegister[reg[[i]]]; printExpr[", "];), {i,Length[reg]-1}]; 
	If[reg=!={},printRegister[Last[reg]]];
	printExpr["]]"];
	)

printInstruction[Instruction["FunctionCall", function_, regOut_, reg_List]]:=
	(
	printRegister[ regOut];
	printExpr[" = " <> ToString[function]];	
	printExpr["[ "]; 
	Do[(printRegister[reg[[i]]]; printExpr[", "];), {i,Length[reg]-1}]; 
	If[reg=!={},printRegister[Last[reg]]];
	printExpr["]]"];
	)
	
printInstruction[Instruction[List, dims_, regOut_, regArgs_List]] :=
Module[{pargs = Map[registerString, regArgs]},
	printRegister[regOut];
	printExpr[" = "];
	printExpr[ToString[ArrayReshape[pargs, dims]]];
];

printInstruction[Instruction[op_/;Not[StringFreeQ[ToString[op],ToString[Random]]], reg_, {{reg1_,reg2_}, regtens___}]]:=(
	printRegister[reg]; printExpr[" = "<>ToString[op]<>"[ {"]; printRegister[reg1]; printExpr[", "];
	printRegister[reg2]; printExpr["}"]; If[Length[{regtens}]=!=0, printExpr[", "]]; printRegister[regtens]; printExpr["]"];)

printInstruction[Instruction[op_, lhs_, rhs_]] := 
 	printInstruction[Instruction[op, lhs, {rhs}]]

printInstruction[Instruction[op_, lhs_, args_List]] := 
	printInstruction[Instruction[op, lhs, args, 0]]

(*
  TODO,  think of a better way to spread info about the CompareFunctionQ
*)
printInstruction[Instruction[op_?CompiledFunctionTools`Opcodes`CompareFunctionQ, lhs_, {tol_, argRegs__}, flags_]] :=
    Module[ {strOp, args, type},
    	args = {argRegs};
    	type = getRegisterType[ Last[ args]];
    	If[ type === Integer,
    		args = {tol,argRegs}];
        strOp = toInfixForm[op];
        printRegister[lhs];
        printExpr[" ="];
        Do[
        	printExpr[" "];
			If[ (i =!= 1), printExpr[strOp];printExpr[" "]];
			printExpr[args[[i]]]
			, 
			{i, Length[args]}];
		If[ type =!= Integer,
			printExpr[ " (tol "]; printExpr[tol]printExpr[ ")"]];
	]


printInstruction[Instruction[op_, lhs_, args_List, flags_]] :=
    Module[ {strOp, useFullForm},
        strOp = toInfixForm[op];
        useFullForm = ! StringQ[strOp];
        printRegister[lhs];
        printExpr[" ="];
        If[ FreeQ[op, List],
            If[ useFullForm,
                printExpr[{" ",op, "["}];
                strOp = ", ",
                strOp = {" ", strOp, " "}
            ];
            If[ Not[MemberQ[{Minus,Not},op]],
                printExpr[" "]
            ];
            Do[
             If[ (i =!= 1) || (MemberQ[{Minus,Not},op]),
                 printExpr[strOp]
             ];
             printExpr[args[[i]]], {i, Length[args]}];
            If[ useFullForm,
                printExpr["]"]
            ],
            printExpr["{"];
            Do[printExpr[" "];
               printRegister[args[[i]]];
               If[ i =!= Length[args],
                   printExpr[","]
               ], {i, Length[args]}];
            printExpr[" }"]
        ]
    ]
    
    

printInstruction[Instruction[Return]] := printExpr[Return];

printInstruction[Instruction["RuntimeError"]] := printExpr["Return Error"];


End[] (* End Private Context *)

EndPackage[]