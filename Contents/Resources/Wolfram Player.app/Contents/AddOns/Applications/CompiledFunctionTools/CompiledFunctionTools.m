(* Mathematica Package *)

(* Created by the Wolfram Workbench 15-Jun-2007 *)

BeginPackage["CompiledFunctionTools`"]

CompiledProcedure::usage = "CompiledProcedure[ args...] holds a rich expression form of a CompiledFunction."

CompiledInfo::usage = "CompiledInfo[ args] holds information in a CompiledProcedure describing a CompiledFunction."

CompiledSetup::usage = "CompiledSetup[args] holds setup code for a CompiledProcedure describing a CompiledFunction."

CompiledResult::usage = "CompiledResult[register] holds the register for the result of a CompiledFunction."

Instruction::usage = "Instruction[ fun, res, args] represents an instruction of a CompiledFunction."

Register::usage = "Register[type] represents a register for a CompiledFunction."

Argument::usage = "Argument[i] represents the ith argument of a CompiledFunction."

Boolean::usage = "Boolean represents a boolean type for a CompiledFunction."

MTensor::usage = "MTensor[type, rank] represents a tensor of given type and rank for a CompiledFunction."

VoidType::usage = "VoidType represents an uninitialized type, or a function that returns no result."

ConstantTensor::usage ="ConstantTensor represents a constant tensor."

ToCompiledProcedure::usage = "ToCompiledProcedure[ cf] generates a CompiledProcedure for a CompiledFunction."

CompiledProcedureToString::usage = "CompiledProcedureToString[comp] returns a string representation for a CompiledProcedure."

CompiledFunctionToString::usage = "CompiledFunctionToString[comp] returns a string representation for a CompiledFunction."

CompilePrint::usage = "CompilePrint[ cfun] prints a human readable form of a compiled function."

ShowInstructions::usage = "ShowInstructions is an option of CompilePrint that displays the opcodes for instructions."

FromTypeEnum::usage = "FromTypeEnum[code] gives the name of the type associated with enumerated code."

ToTypeEnum::usage = "ToTypeEnum[name] gives the enumerated code for the type name."

CompiledConstants::usage = "CompiledConstants[cons] holds constants used by CompiledFunction."

JumpModify::usage = "JumpModify[ proc] modifies the procedure to remove gotos."

InstructionIf::usage = "InstructionIf[ reg, branch1, branch2] represents an If statement."

InstructionWhile::usage = "InstructionWhile[ conditionEval, reg, bodyEval] represents a While statement."

InstructionFor::usage = "InstructionFor[ regCount, regInit, regLimit, body] represents a For loop."

Begin["`Private`"]
(* Implementation of the package *)

Get[ "CompiledFunctionTools`Opcodes`"]
Get[ "CompiledFunctionTools`PrintCode`"]
Get[ "CompiledFunctionTools`JumpModify`"]

(*
 Basic constructs
*)

CompiledProcedure
SetAttributes[CompiledProcedure, HoldAll]

CompiledInfo
CompiledSetup
CompiledResult
Instruction
Register
Argument

(*
 Global to keep track of tensor register reuse.
*)

$RegisterReuse

ToCompiledProcedure[
  HoldPattern[CompiledFunction][args_, setupCode_,  regs_ /; Dimensions[regs] === {5}, codeIn_, ___]] := 
  	(
  		Message[ToCompileProcedure::oldver, Part[Hold[codeIn], 1, 1, 2]];
  		$Failed
  	)
  
  
(* The definitions for getInstruction needs to be changed to avoid this *)
OpcodeInitialize[10, 10.];
  	
ToCompiledProcedure[
  cf:HoldPattern[CompiledFunction][{codeVer_, mathVer_, rtFlags_}, args_, setupCode_, consSpecs_, regs_ /; Dimensions[regs] === {5}, codeIn_, HoldPattern[Function[{vars__}, _]], ___]] := 
Block[{vars},
	ToCompiledProcedureLocalized[cf]
]
 
ToCompiledProcedure[
  cf:HoldPattern[CompiledFunction][{codeVer_, mathVer_, rtFlags_}, args_, setupCode_, consSpecs_, regs_ /; Dimensions[regs] === {5}, codeIn_, ___]] := 
	ToCompiledProcedureLocalized[cf]

ToCompiledProcedureLocalized[
  HoldPattern[CompiledFunction][{codeVer_, mathVer_, rtFlags_}, args_, setupCode_, consSpecs_, regs_ /; Dimensions[regs] === {5}, codeIn_, ___]] := 
 Module[{info, setup, result, insts, code, cons, runtimeFlags = rtFlags},
 	If[!TrueQ[OpcodeInitialize[codeVer, mathVer]],
 		Message[ToCompiledProcedure::badver, codeVer, mathVer]
 	];
 	code = Hold[codeIn];
 	code = Replace[code, 
 			{op:(CCEVALREG | CCCOMPILEDFUNCTION), fun_, rest___} :> 
 				{op, Hold[fun], rest}, {2}];
 	If[codeVer < 10, (*CCCREATETENSORFROMLIST change in codeVer 10 *)
 		code = Replace[code, {CCCREATETENSORFROMLIST, stuff___} :> {CCCREATETENSORFROMLIST, 1, Length[{stuff}] - 2, stuff}, {2}];
 	];
	If[mathVer < 11.3,
		(* clear the CatchMachineUnderflow bit which might be set, but is no longer supported *)
		runtimeFlags = BitAnd[rtFlags, BitNot[1]]
	];
 	clearTensorTable[];
	info = getCompiledInfo[args, regs, Drop[setupCode,-1], runtimeFlags];
  	setup = getCompiledSetup[Drop[setupCode, -1]];
  	cons = getCompiledConstants[ consSpecs];
  	result = CompiledResult[If[Part[Last[setupCode],2]==0,
  				toRegister[{First[#],Last[#]}&[Last[setupCode]]],
  				toRegister[Last[setupCode]]]];
 	insts = MapIndexed[getInstructionMI, code, {2}];
 	insts = ReleaseHold[insts];
  	Apply[CompiledProcedure[##, codeIn]&, {info, setup, cons, result, insts}]
]

(*
	Way to get the Runtime options data out
*)
CompiledInfo[__, "RuntimeFlags"->flags_]["RuntimeFlags"] := flags;

(cinfo_CompiledInfo)["ArithmeticFlags"] := 
	BitAnd[cinfo["RuntimeFlags"], ArithmeticFlags];

(cinfo_CompiledInfo)["RuntimeAttributes"] := 
Module[{listable = BitAnd[cinfo["RuntimeFlags"], ListableFlag]},
	If[TrueQ[listable > 0], 
		{Listable},
		{}
	]
];

(*Create CompiledSetup*)
getCompiledSetup[regs_List] := (
 Map[setTensor[Last[#], Most[#]]&,
  Select[regs, Part[#,2]=!=0&]];
 CompiledSetup[
   Table[Instruction[Set, toRegister[regs[[i]]], Argument[i]], {i, 
     Length[regs]}]])



(* Create CompiledConstants *)
getCompiledConstants[ spec_] :=
	Module[ { con, reg, no, type, rank},
		CompiledConstants[ 
			Map[ 
				(con = #[[1]]; reg = #[[2]];
				{type, rank, no} = reg;
				If[rank > 0, 
					setTensor[no, {type, rank}];
					con = ConstantTensor[ con]];
				Instruction[ Set, toRegister[reg], con])&, spec]
		]
	]


(*Create CompiledInfo*)
getCompiledInfo[args_, regs_, setup_, rtFlags_] :=(
	CompiledInfo[
		Table[First[toRegister[setup[[i]]]], {i, Length[setup]}], 
		Table[getRegisterInfo[i, regs[[i]]], {i, Length[regs]}],
		"RuntimeFlags"->rtFlags
	])

getRegisterInfo[type_Integer, num_] := {FromTypeEnum[type],num}


(*Register formatting tools*)
toRegister[{type_?NumericQ, 0, reg_}] := toRegister[{type, reg}]

toTensorRegister[reg_] := toTensorRegister[Append[getTensor[reg], reg]]

toTensorRegister[{type_?NumericQ, rank_, reg_}] := 
	Register[MTensor[FromTypeEnum[type], rank], reg]

toRegister[{type_?NumericQ, rank_, reg_}] := 
 	Register[MTensor[FromTypeEnum[type], rank], reg]

toRegister[{type_?NumericQ, reg_}] := 
 	Register[FromTypeEnum[type], reg]

(*toRegister[{type_?Symbol, 0.1, reg_}]:=
	toRegister[{ToTypeEnum[type], 0.1, reg}]*)

toRegister[{type_Symbol, rank_, reg_}] := 
 	toRegister[{ToTypeEnum[type], rank, reg}]

toRegister[{type_Symbol, reg_}] := 
 	toRegister[{ToTypeEnum[type], reg}]

toRegister[{type_Symbol, 0, reg_}] := 
 	toRegister[{ToTypeEnum[type], 0, reg}]

(* For convenience in going from something like, e.g. 
   toRegister[getTensor[reg], reg]    
*)
toRegister[{{type_, rank_}, reg_}] := toRegister[{type, rank, reg}]

(*Type processing tools*)

FromTypeEnum[1] = Boolean
FromTypeEnum[2] = Integer
FromTypeEnum[3] = Real
FromTypeEnum[4] = Complex
FromTypeEnum[5] = MTensor
FromTypeEnum[6] = VoidType

FromTypeEnum[x_/;Negative[x]] := FromTypeEnum[-x]

ToTypeEnum[Boolean] = 1
ToTypeEnum[True | False] = 1
ToTypeEnum[Integer] = 2
ToTypeEnum[Real] = 3
ToTypeEnum[Complex] = 4
ToTypeEnum[MTensor] = 5
ToTypeEnum[VoidType] = 6

toShortName[Boolean] := "B"
toShortName[Integer] := "I"
toShortName[Real] := "R"
toShortName[Complex] := "C"
toShortName[MTensor[type_Symbol, rank_]] := 
 "T(" <> toShortName[type] <> ToString[rank] <> ")"
toShortName[VoidType] := "V"


(*Formatting tool*)
toString[x___] := Apply[StringJoin, Map[ToString, {x}]]


(*Utilities for the tensor table*)
clearTensorTable[] := ($RegisterReuse = {};
  Clear[myTensorTable];)

setTensor[i_Integer, data_List] :=(
 If[Head[myTensorTable[i]] === List, 
 	$RegisterReuse = Append[$RegisterReuse, {i, myTensorTable[i], data}]];
  myTensorTable[i] = data;)

getTensor[t_] := myTensorTable[t];

(* This is used to go from the form of MapIndexed to
   the order needed by getInstruction without evaluation *)
SetAttributes[getInstructionMI, HoldAll];
getInstructionMI[code_, {__, i_}] := getInstruction[i, code];

(*Create the Instruction expressions*)

getInstruction[line_, {CCVERSION, num_}] := 
 Instruction["Version", num]

getInstruction[l_, {CCRET}] := Instruction[Return]


(*Plus operator*)
getInstruction[line_, {CCPLUSREAL, regs__, rego_}] := 
 Instruction[Plus, toRegister[{Real, rego}], 
  Map[toRegister[{Real, #}] &, {regs}]]

getInstruction[line_, {CCPLUSINT, regs__, rego_}] := 
 Instruction[Plus, toRegister[{Integer, rego}], 
  Map[toRegister[{Integer, #}] &, {regs}]]

getInstruction[line_, {CCPLUSCX, regs__, rego_}] := 
 Instruction[Plus, toRegister[{Complex, rego}], 
  Map[toRegister[{Complex, #}] &, {regs}]]

getInstruction[line_, {CCPLUSTENSOR, regs__, rego_}] := 
 Module[{temp,tab = getTensor[First[{regs}]], type, rank}, 
  type = First[tab];
  rank = Last[tab];
  temp=Map[toRegister[{type, rank, #}] &, {regs}];
  setTensor[rego, {type, rank}];
  Instruction[Plus, toRegister[{type, rank, rego}], 
    temp]]



(*Times operator*)
getInstruction[line_, {CCTIMESINT, regs__, rego_}] := 
 Instruction[Times, toRegister[{Integer, rego}], 
  Map[toRegister[{Integer, #}] &, {regs}]]

getInstruction[line_, {CCTIMESREAL, regs__, rego_}] := 
 Instruction[Times, toRegister[{Real, rego}], 
  Map[toRegister[{Real, #}] &, {regs}]]

getInstruction[line_, {CCTIMESCX, regs__, rego_}] := 
 Instruction[Times, toRegister[{Complex, rego}], 
  Map[toRegister[{Complex, #}] &, {regs}]]

getInstruction[line_, {CCTIMESTENSOR, regs__, rego_}] := 
 Module[{temp,tab = getTensor[First[{regs}]], type, rank}, 
  type = First[tab];
  rank = Last[tab];
  temp=Map[toRegister[{type, rank, #}] &, {regs}];
  setTensor[rego, {type, rank}];
  Instruction[Times, toRegister[{type, rank, rego}], 
    temp]]




(*Minus operator*)
getInstruction[line_, {CCMINUSINT, regin_, rego_}] := 
 Instruction[Minus, toRegister[{Integer, rego}], {toRegister[{Integer, regin}]}]

getInstruction[line_, {CCMINUSREAL, regin_, rego_}] := 
 Instruction[Minus, toRegister[{Real, rego}], {toRegister[{Real, regin}]}]

getInstruction[line_, {CCMINUSCX, regin_, rego_}] := 
 Instruction[Minus, toRegister[{Complex, rego}], {toRegister[{Complex, regin}]}]


(*Constants*)
getInstruction[line_, {CCLDCONSTINT, regs_, rego_}] := 
 Instruction[Set, toRegister[{Integer, rego}], regs]

getInstruction[line_, {CCLDCONSTREAL, regs_, rego_}] := 
 Instruction[Set, toRegister[{Real, rego}], regs]

getInstruction[line_, {CCLDCONSTCX, regre_, regim_, rego_}] := 
 Instruction[Set, toRegister[{Complex, rego}], 
  regre + regim*I]

getInstruction[line_, {CCLDCONSTBOOL, regs_, rego_}] := 
 Instruction[Set, toRegister[{Boolean, rego}], regs]




(*Conversions*)
getInstruction[line_, {CCCVTBOOLTOINT, reg_, rego_}] := 
 Instruction[Set, toRegister[{Integer, rego}], 
  toRegister[{Boolean, reg}]]

getInstruction[line_, {CCCVTINTTOREAL, reg_, rego_}] := 
Module[{res = toRegister[{Real, rego}], arg = toRegister[{Integer, reg}]},
	Instruction[Set, res, arg]
]

getInstruction[line_, {CCCVTREALTOCX, regs__, rego_}] := 
 Instruction["SetComplex", toRegister[{Complex, rego}], 
  Map[toRegister[{Real, #}] &, {regs}]]

(* 
	Equalities (and SameQ) 
*)
getInstruction[line_,{CCEQBOOL, regs__, rego_}]:=
 Instruction[Equal, toRegister[{Boolean, rego}], 
  Map[toRegister[{Boolean, #}] &, {regs}]]
  
getInstruction[line_, {CCEQINT, regs__, rego_}] := 
 Instruction[Equal, toRegister[{Boolean, rego}], 
  Map[toRegister[{Integer, #}] &, {regs}]]

ComparisonHead[1] = SameQ
ComparisonHead[0] = Equal;

getInstruction[line_, {CCEQREAL, regs__, what_, rego_}] := 
 Instruction[ComparisonHead[what], toRegister[{Boolean, rego}], 
  Map[toRegister[{Real, #}] &, {regs}]]

getInstruction[line_, {CCEQCOMPLEX, regs__, what_, rego_}] := 
 Instruction[ComparisonHead[what], toRegister[{Boolean, rego}], 
  Map[toRegister[{Complex, #}] &, {regs}]]

getInstruction[line_, {CCEQTENSOR, regs__, rego_}] :=
Module[{type, what = 0, args = {regs}},
	type = FromTypeEnum[First[getTensor[First[args]]]];
	If[type =!= Integer, 
		what = Last[args];
		args = Drop[args, -1];
	];
	Instruction[ComparisonHead[what], toRegister[{Boolean, rego}], Map[toRegister[{getTensor[#], #}]&, args]]
]

(*Inequalities*)
getInstruction[line_, {CCLTINT, regs__, rego_}] := 
 Instruction[Less, toRegister[{Boolean, rego}], 
  Map[toRegister[{Integer, #}] &, {regs}]]

getInstruction[line_, {CCLEINT, regs__, rego_}] := 
 Instruction[LessEqual, toRegister[{Boolean, rego}], 
  Map[toRegister[{Integer, #}] &, {regs}]]

getInstruction[line_, {CCCOMPAREREAL, type_, tolreg_, regs__, rego_}] := 
	Instruction[CompareName[type], 
		toRegister[{Boolean, rego}], 
		Join[
			{toRegister[{Real, tolreg}]},
			Map[toRegister[{Real, #}] &, {regs}]
		]
	]

getInstruction[line_, {CCCOMPARECOMPLEX, type_, tolreg_, regs__, rego_}] := 
	Instruction[CompareName[type], 
		toRegister[{Boolean, rego}], 
		Join[
			{toRegister[{Real, tolreg}]},
			Map[toRegister[{Complex, #}] &, {regs}]
		]
	]

(*Copy*)
getInstruction[line_, {CCCOPYINT, regs_, rego_}] := 
 Instruction[Set, toRegister[{Integer, rego}], 
  toRegister[{Integer, regs}]]

getInstruction[line_, {CCCOPYREAL, regs_, rego_}] := 
 Instruction[Set, toRegister[{Real, rego}], toRegister[{Real, regs}]]

getInstruction[line_, {CCCOPYCX, regs_, rego_}] := 
 Instruction[Set, toRegister[{Complex, rego}], 
  toRegister[{Complex, regs}]]

getInstruction[line_, {CCCOPYBOOL, regs_, rego_}] := 
 Instruction[Set, toRegister[{Boolean, rego}], 
  toRegister[{Boolean, regs}]]

(*Some mathematical operators*)
getInstruction[
  line_, {CCMATH1ARG, fun_, type_, rank_, reg_, typeres_, rankres_, 
   regres_}] := 
	(
	If[rankres =!= 0, setTensor[regres, {typeres, rankres}]];
  	Instruction[ Arg1Name[fun], toRegister[{typeres, rankres, regres}], 
   		{toRegister[{type, rank, reg}]}]
   	)

getInstruction[
  line_, {CCMATH2ARG, fun_, type1_, rank1_, reg1_, type2_, rank2_, 
   reg2_, typeres_, rankres_, regres_}] := 
	(
	If[rankres =!= 0, 
   		setTensor[regres, {typeres, rankres}];];
  	Instruction[ Arg2Name[fun], 
   		toRegister[{typeres, rankres, 
     		regres}], {toRegister[{type1, rank1, reg1}], 
    		toRegister[{type2, rank2, reg2}]}]
    )

(*MTensor construction*)
getInstruction[line_, {CCCREATETENSORFROMLIST, rank_, args__, type_, rego_}] := 
Module[{dims, regs, tensor, outreg, argregs},
	dims = Take[{args}, rank];
	regs = Drop[{args}, rank];
	If[type == 0,
		tensor = getTensor[First[regs]];
		setTensor[rego, MapAt[# + rank &, tensor, 2]];
		outreg = toRegister[{Apply[Sequence, getTensor[rego]], rego}];
		argregs = Map[toRegister[{Apply[Sequence, getTensor[#]], #}] &, regs]
	(* else *),
		setTensor[rego, {type, rank}];
		outreg = toRegister[{type, rank, rego}];
		argregs = Map[toRegister[{type, #}] &, regs];
	];
	Instruction[List, dims, outreg, argregs]
]

(*
getInstruction[line_, {CCCREATETENSORFROMLIST, reg__, 0, rego_}] := 
Module[{tensor = getTensor[First[{reg}]], temp=Map[toRegister[{Apply[Sequence, getTensor[#]], #}] &, {reg}]}, 
  setTensor[rego, MapAt[# + 1 &, tensor, 2]];
  Instruction[List, 
   toRegister[{Apply[Sequence, getTensor[rego]], rego}], temp]]

getInstruction[line_, {CCCREATETENSORFROMLIST, reg___, type_, rego_}] := 
Module[{temp = Map[toRegister[{type, #}] &, {reg}]},
  setTensor[rego, {type, 1}];
  Instruction[List, toRegister[{type, 1, rego}], 
   temp]]
*) 
getInstruction[line_, {CCLDCONSTTENSOR, tensor_, type_, rank__, rego_}] := (
  setTensor[rego, {type, Depth[tensor] - 1}];
  Instruction[Set, toRegister[{type, Length[{rank}], rego}], 
   ConstantTensor[tensor]])

getInstruction[line_, {CCCREATETENSORFORTABLE, regs__Integer, type_, rego_}]:=(
 setTensor[rego, {type, Length[{regs}]}];
 Instruction[Table, toRegister[{type, Length[{regs}], rego}],
 	Map[toRegister[{2,#}]&,{regs}]])



getInstruction[line_, {CCINSERTTOTABLE, regPos_, regValue_, 0, regTensor_}]:=
	Module[{temp},
		temp = toRegister[Append[getTensor[regValue],regValue]];
 		setTensor[regTensor, MapAt[ # + 1&, getTensor[regValue],2]];
  		Instruction["SetElement", 
 			toRegister[Append[getTensor[regTensor],regTensor]],
 			toRegister[{Integer,regPos}], temp]
 	]

getInstruction[line_,{CCINSERTTOTABLE, regPos_, regVal_, tcode_, regTen_}]:=
Module[{type = FromTypeEnum[tcode]},
	Instruction["SetElement", 
		toRegister[{getTensor[regTen],regTen}],
		toRegister[{Integer,regPos}],
		toRegister[{type,regVal}]
		]
]

(*MTensor manipulations*)
PartSpecification[{0, reg_}] := toRegister[{Integer, reg}];
PartSpecification[{1, reg_}] := {toRegister[{getTensor[reg], reg}], List};
PartSpecification[{2, _}] := All;
PartSpecification[{3, reg_}] := {toRegister[{getTensor[reg], reg}], Span};

getInstruction[line_, {sp:(CCPART | CCSETPART), regin_, ptypereg__, rankout_, rego_}] := 
Module[{tin = getTensor[regin], specs = Partition[{ptypereg}, 2], out},
	If[rankout > 0,
		setTensor[rego, {First[tin], rankout}];
		out = toRegister[{getTensor[rego], rego}],
		out = toRegister[{First[tin], rego}]
	];
	Instruction[If[sp === CCPART, Part, "SetPart"],
  		out, 
  		Join[
  			{toRegister[{tin, regin}]},
  			Table[PartSpecification[spec], {spec, specs}]
  		]
	]
]

getInstruction[line_, {CCGETELEMENT, arg_, indices__, 0, res_}] :=
Module[{type, argrank, rank, out, parts = {indices}},
	{type, argrank} = getTensor[arg];
	rank = argrank - Length[parts];
	setTensor[res, {type, rank}];
	out = toRegister[{type, rank, res}];
	parts = Map[toRegister[{Integer, #}]&, parts];
	Instruction["GetElement", out, Flatten[{toRegister[{type, argrank, arg}], parts}]]
]

getInstruction[line_, {CCGETELEMENT, arg_, indices__, type_, res_}] :=
Module[{argtype, argrank, rank, out, parts = {indices}},
	{argtype, argrank} = getTensor[arg];
	rank = argrank - Length[parts];
	out = toRegister[{type, res}];
	parts = Map[toRegister[{Integer, #}]&, parts];
	Instruction["GetElement", out, Flatten[{toRegister[{type, argrank, arg}], parts}]]
]

getInstruction[line_, {CCLENGTH, regtens_, rego_}] := 
 Instruction[Length, toRegister[{Integer, rego}], 
  toRegister[{Apply[Sequence, getTensor[regtens]], regtens}]]

getInstruction[line_, {CCDIMENSIONS, regtens_, regint_, rank_, rego_}]:=Module[{temp=toRegister[Append[getTensor[regtens],regtens]]},
 setTensor[rego, {2,1}];
 Instruction[Dimensions, toRegister[Append[getTensor[rego],rego]], {temp,
 	toRegister[{2,regint}]}]]

getInstruction[line_, {CCERR}]:=
 	Instruction["RuntimeError"]



(*
  TODo  not correct for 6.0
*)


(* Branch, jump, evaluate operators *)
getInstruction[line_, {CCBRANCH, reg_, branch_}]:=Instruction["Branch", toRegister[{1,reg}], Line[line+branch]]

getInstruction[line_, {CCJUMP, jump_}]:=Instruction["Jump", Line[line+jump]]



getInstruction[line_,{CCLOOPINCR, reg1_, reg2_, jump_}]:=Instruction["LoopIncr", {toRegister[{2,reg1}],toRegister[{2,reg2}]},
	line+jump]
	





getInstruction[line_, {CCXOR, regs__, rego_}]:=
 Instruction[Xor, toRegister[{Boolean,rego}],
 	Map[toRegister[{Boolean,#}]& ,{regs}]]
 	
getInstruction[line_, {CCNOT, regs_, rego_}]:=
 Instruction[Not, toRegister[{Boolean,rego}], toRegister[{Boolean,regs}]]
 
 
 

getInstruction[line_, {CCEVALREG, op_, reg__}] :=
	Module[{regs=Partition[ {reg},3], rego},
		rego = Part[ regs, -1];
		If[rego[[2]] > 0, setTensor[rego[[-1]], rego[[{1,2}]]]];
		regs = Drop[ regs, -1];
		Instruction["MainEvaluate", op, toRegister[rego], Map[toRegister,regs]]
	]

getInstruction[line_, {CCCOMPILEDFUNCTION, op_, reg__}] :=
	Module[{regs=Partition[ {reg},3], rego},
		rego = Part[ regs, -1];
		If[rego[[2]] > 0, setTensor[rego[[-1]], rego[[{1,2}]]]];
		regs = Drop[ regs, -1];
		Instruction["CompiledFunctionCall", op, toRegister[rego], Map[toRegister,regs]]
	]

SetAttributes[UniqueCompileSym, HoldAll];

cnt = 0;
UniqueCompileSym[ x_Symbol] :=
Block[{x}, (* Prevent evaluation in case x is defined *)
	cnt = cnt + 1;
	ToExpression[ ToString[ FullForm[x]] <> "Compile$" <>  ToString[ cnt]]
]

(*
  Try to fix any Function[ args, body] to incorporate the new settings 
  for dynamic local variables.   For Function[{a,b}, f[a,b]] and 
  vv of {r,s}, we return,   
      Function[ {a,b,r$1,s$2}, Block[{r = r$1, s = s$2}, {f[a,b], r, s}].
*)

fixFunction[ HoldPattern[Function[{varsFun__}, body_]], Hold[vv__]] :=
	Module[ {varsFix, ef, fun, setVars, varsNew, vars, varsHeld, newBody},
		vars = Hold[vv];
		varsHeld = Apply[List, Map[Hold, vars]];
		varsNew = Apply[List, Map[ UniqueCompileSym, vars]];
		varsFix = Join[ Hold[varsFun], Apply[Hold, varsNew]];
		setVars = HoldComplete @@ Transpose[{varsHeld, varsNew}];
		setVars = Replace[setVars, {Hold[a_], b_} :> Set[a, b], {1}];
		setVars = HoldComplete @@ {setVars};
		setVars = Apply[ List, setVars, {1}];
		newBody = HoldComplete @@ {Join[HoldComplete[body], HoldComplete[vv]]};
		newBody = ReplacePart[newBody, {1, 0} -> List];
		ef = Join[ setVars, newBody];
		ef = HoldComplete @@ {ef};
		ef = ReplacePart[ef, {1, 0} -> Block];
		ef = Insert[ ef, varsFix, 1];
		fun = formFunction @@ ef;
		fun
	]

(* To form the function without unwanted messages or evaluation *)
SetAttributes[formFunction, HoldAll];
formFunction[Hold[vl___], body_] := Function[{vl}, body];

(*
 If there are no Block local variables then don't modify the function.
*)
fixFunction[ fun_, {}] := fun

(*
 If the function does not match Function[ vars, body] don't modify it.
*)
fixFunction[ fun_, _] := fun

(*
	This last def'n used for CCEVALDYNAMIC
*)

fixFunction[Hold[op_], v_] := fixFunction[Function[{}, op], v]

getInstruction[line_, {CCEVALARG, funcIn_, varsIn___, reg:Except[_List]..}]:=
	Module[{regs=Partition[ {reg},3], regsout, settings, inputs, func, vars, varRegs},
		vars = Hold[varsIn];
		varRegs = Apply[List, vars[[All, 2;;-2]]];
		regsout = Join[Part[regs, {-1}], varRegs];
		settings = Map[
			Function[
				If[#[[2]] > 0, setTensor[#[[-1]], #[[{1,2}]]]];
				toRegister[#]
			],
			regsout
		];
		regs = Drop[ regs, -1];
		regs = Join[ regs, varRegs];
		inputs = Map[toRegister,regs];
		vars = vars[[All, 1]];
		func = fixFunction[ funcIn, vars];
		Instruction["MainEvaluate", func, settings, inputs]
	]
	
	
getInstruction[line_, {CCEVALDYNAMIC, op_, regsIn_, {dvars_, dregs_}, {locals___}, rego_}] :=
Module[{func, regs = regsIn, regsout, settings, vars, varRegs, hl},
	hl = Hold[locals];
	vars = hl[[All,1]];
	func = fixFunction[op, vars];
	varRegs = Apply[List, hl[[All, 2;;-2]]];
	regsout = Join[{rego}, varRegs];
	settings = Map[
		Function[
			If[#[[2]] > 0, setTensor[#[[-1]], #[[{1,2}]]]];
			toRegister[#]
		],
		regsout
	];
	regs = Join[regs, varRegs];
	Instruction["MainEvaluate", func, settings, Map[toRegister, regs], {dvars, Map[toRegister, dregs]}]
]

RegisterNumberQ[x_Integer] := Not[Negative[x]];
RegisterNumberQ[x_] := False;

FunctionCallArgumentToRegister[{param_, 0, -1}] := param;
FunctionCallArgumentToRegister[{type_?Positive, 0, num_?RegisterNumberQ}] := toRegister[{type, num}];
FunctionCallArgumentToRegister[{type_?Negative, 0, num_?RegisterNumberQ}] := toTensorRegister[{type, 0, num}];
FunctionCallArgumentToRegister[{type_, rank_?Positive, num_?RegisterNumberQ}] := toTensorRegister[{type, rank, num}];

FunctionCallResultToRegister[{type_, 0, num_?RegisterNumberQ}] := toRegister[{type, num}];
FunctionCallResultToRegister[{type_?Negative, 0, num_?RegisterNumberQ}] := (setTensor[num, {type, 0}]; toTensorRegister[num]);
FunctionCallResultToRegister[{type_, rank_?Positive, num_?RegisterNumberQ}] := (setTensor[num, {type, rank}]; toTensorRegister[num]);

getInstruction[line_, {CCFUNCTIONCALL, funname_, regs__Integer}] := 
Module[{pargs},
	pargs = Partition[{regs}, 3];
	SystemAssert[Length[pargs] === Length[{regs}]/3];
	Instruction[
		"FunctionCall", 
		funname, 
		FunctionCallResultToRegister[Last[pargs]], 
		Map[FunctionCallArgumentToRegister, Drop[pargs, -1]]
	]
]
	
(*
    The processing below is for instructions 
    that were eliminated (replaced by 
    using CC_FUNCTION_CALL for version 8)
*)

(* Random operators *)
getInstruction[line_,{CCRANDOM, type_, rego_}]:=
	Instruction[Random, FromTypeEnum[type], toRegister[{type, rego}], {}]
	 
getInstruction[line_,{CCRANDOM, type_, arg1_, arg2_, rego_}]:=
	Instruction[Random, FromTypeEnum[type], toRegister[{type, rego}], {toRegister[{type, arg1}], toRegister[{type, arg2}]}]

RandomType[CCRANDOMINTEGER] = ToTypeEnum[Integer];
RandomType[CCRANDOMREAL] = ToTypeEnum[Real];
RandomType[CCRANDOMCOMPLEX] = ToTypeEnum[Complex];

RandomCommand[opcode_] := ToExpression["Random" <> ToString[FromTypeEnum[RandomType[opcode]]]];

getInstruction[line_, {opcode:(CCRANDOMINTEGER | CCRANDOMREAL | CCRANDOMCOMPLEX), dist_, argseq__, rego_}] :=
Module[{args = {argseq}, len, range, dims = Sequence[], type = RandomType[opcode], res},
	len = Length[args] + 3;
	range = If[EvenQ[len],
		toTensorRegister[First[args]],
		{toRegister[{type, args[[1]]}], toRegister[{type, args[[2]]}]}
	];
	If[len < 6,
		res = toRegister[{type, rego}];
		args = {range},
	(* else *)
		setTensor[rego, {type, args[[-1]]}];
		res = toTensorRegister[rego];
		dims = toTensorRegister[args[[-2]]];
		args = {range, dims}
	];
	Instruction[RandomCommand[opcode], res, args]
] 

getInstruction[line_, {CCRANDOMCHOICE, weights_, regin_, regout_}] := 
Module[{type, rank, out, arg},
	{type, rank} = getTensor[regin];
	If[rank > 1,
		setTensor[regout, {type, rank - 1}];
		out = toTensorRegister[regout],
	(* else *)
		out = toRegister[{type, regout}]
	];
	arg = toTensorRegister[regin];
	If[weights != -1, arg = toTensorRegister[weights]->arg];
	Instruction[RandomChoice, out, {arg}]
]

getInstruction[line_, {CCRANDOMCHOICE, weights_, regin_, dims_, rankout_, regout_}] := 
Module[{type, rank, out, arg},
	{type, rank} = getTensor[regin];
	setTensor[regout, {type, rankout}];
	out = toTensorRegister[regout];
	arg = toTensorRegister[regin];
	If[weights != -1, arg = toTensorRegister[weights]->arg];
	Instruction[RandomChoice, out, {arg, toTensorRegister[dims]}]
]

getInstruction[line_, {CCRANDOMSAMPLE, weights_, regin_, nreg___, regout_}] := 
Module[{type, rank, out, arg, n = nreg},
	{type, rank} = getTensor[regin];
	setTensor[regout, {type, rank}];
	out = toTensorRegister[regout];
	arg = toTensorRegister[regin];
	If[weights != -1, arg = toTensorRegister[weights]->arg];
	If[Length[{n}] > 0, n = toRegister[{Integer, First[{n}]}]];
	Instruction[RandomSample, out, {arg, n}]
]

getInstruction[line_, {CCINSERT, regs__, rego_}] := 
Module[{tr = getTensor[First[{regs}]], args},
	args = Map[toTensorRegister, {regs}];
	setTensor[rego, tr];
	Instruction[Insert, toTensorRegister[rego], args]]

getInstruction[line_, {CCDELETE, regs__, rego_}]:=Module[{tab=getTensor[First[{regs}]],temp},
 temp=Map[toRegister[Append[getTensor[#],#]]&,{regs}];
 setTensor[rego,tab];
 Instruction[Delete, toRegister[Append[tab,rego]], temp]]

getInstruction[line_, {CCSORT, reg_, rego_}]:=Module[{temp=toRegister[Append[getTensor[reg],reg]]},
	setTensor[rego,getTensor[reg]];
	Instruction[Sort, toRegister[Append[getTensor[rego],rego]], temp]]

getInstruction[line_, {CCREVERSE, regs_, rego_}]:=
 Module[{tab=getTensor[regs]},
 	setTensor[rego, tab];
 	Instruction[Reverse, toRegister[Append[tab,rego]],toRegister[Append[tab,regs]]]]

RotateHead[1] = RotateRight;
RotateHead[-1] = RotateLeft;

getInstruction[line_, {CCROTATE, arg_, shift_, rl_, rego_}] :=
Module[{tr = getTensor[arg]},
	setTensor[rego, tr];
	Instruction[RotateHead[rl], toTensorRegister[rego], {toTensorRegister[arg], toTensorRegister[shift]}]
]

getInstruction[line_, {CCPARTITION, regs__, rego_}]:=Module[{temp=Map[toRegister[Append[getTensor[#],#]]&,{regs}]},
	setTensor[rego,MapAt[#+1&,getTensor[First[{regs}]],2]];
	Instruction[Partition, toRegister[Append[getTensor[rego],rego]], temp]]

getInstruction[line_, {CCPOSITION, regs__, rego_}]:=
	Module[{temp=Map[toTensorRegister[Append[getTensor[#],#]]&,{regs}]},
		setTensor[rego, {2,2}];
		Instruction[Position, toRegister[Append[getTensor[rego],rego]], temp]
	]

getInstruction[line_, {CCCOUNT, regVec_, regThing_, rego_}]:=
	Instruction[Count, toRegister[{Integer, rego}], 
			{toTensorRegister[Append[getTensor[regVec],regVec]], toTensorRegister[Append[getTensor[regThing],regThing]]}]

getInstruction[line_, {CCMEMBERQ, regVec_, regThing_, rego_}]:=
	Instruction[MemberQ, toRegister[{Boolean, rego}], 
			{toTensorRegister[Append[getTensor[regVec],regVec]], toTensorRegister[Append[getTensor[regThing],regThing]]}]
	
 
getInstruction[line_, {CCFREEQ, regContainer_, regPart_, rego_}]:=
	Instruction[FreeQ, 
		toRegister[{Boolean, rego}], 
		{toRegister[Append[getTensor[regContainer],regContainer]],
		toTensorRegister[Append[getTensor[regPart],regPart]]}]
 
 	
getInstruction[line_, {CCORDEREDQ, regt_, rego_}]:=Instruction[OrderedQ, toRegister[{Boolean, rego}],
	toRegister[Append[getTensor[regt],regt]]]
 
getInstruction[line_, {CCFLATTEN, tens_, sh__, rego_}]:=Module[{tab=MapAt[#-sh&, getTensor[tens], 2],temp},
	temp=toRegister[Append[getTensor[tens],tens]];
	setTensor[rego,tab];
	Instruction[Flatten, toRegister[Append[tab,rego]], {temp, sh}]]
	
OuterFunction[CCOUTERLIST] = List;
OuterFunction[CCOUTERTIMES] = Times;
OuterFunction[CCOUTERPLUS] = Plus;
	
getInstruction[line_, {opcode:(CCOUTERLIST | CCOUTERPLUS | CCOUTERTIMES), regs__, rank_, rego_}] :=
Module[{tin, levels, type}, 
	{tin, levels} = Partition[{regs}, Length[{regs}]/2];
	type = First[getTensor[First[tin]]];
	setTensor[rego, {type, rank}];
	Instruction[Outer, OuterFunction[opcode], toTensorRegister[rego], 
		Flatten[{Map[toTensorRegister, tin], levels}]]
]	 
 
getInstruction[line_, {CCTRANSPOSE, flag_, regs__, rank_, rego_}]:=Module[{tab=getTensor[First[{regs}]],temp},
 temp=Map[toRegister[Append[getTensor[#],#]]&, {regs}];
 setTensor[rego, MapAt[rank &, tab, 2]];
 Instruction[ToExpression[Apply[StringJoin,{flag*"Conjugate"+(1-flag)*"","Transpose"}]],
 	toRegister[Append[getTensor[rego],rego]], temp]]

getInstruction[line_, {CCDOT, regs__, rego_}]:=Module[{tab=MapAt[#-1&,getTensor[First[{regs}]],2],temp},
 temp=Map[toRegister[Append[getTensor[#],#]]&,{regs}];
 setTensor[rego,tab];
 Instruction[Dot, toRegister[Append[getTensor[rego],rego]], temp]]

getInstruction[line_, {CCTENSORFUNNARG, op_, regs__, rego_}] :=
 Module[{tab = getTensor[First[{regs}]],temp},
  temp=Map[toRegister[Append[tab, #]] &, {regs}];
  setTensor[rego, tab];
  Instruction[ArgNName[op], toRegister[Append[tab, rego]], 
   temp]]

(* Divers operators *)
getInstruction[line_, {CCINTEGERDIGITS, regs__, rego_}] := 
 Instruction[IntegerDigits, toRegister[{Integer, 1, rego}], 
  Map[toRegister[{Integer, #}] &, {regs}]]
  

(* This must be done after all of the rules for
   getInstruction have been done so that that the
   symbolic instruction identifiers get evaluated
   to the numbers during rule definition *)
   
SetAttributes[getInstruction, HoldRest];


End[]

EndPackage[]