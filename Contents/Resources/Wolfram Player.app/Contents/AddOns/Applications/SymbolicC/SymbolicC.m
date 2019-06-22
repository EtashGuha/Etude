(* Mathematica Package *)

(* Created by the Wolfram Workbench 11-Jan-2010 *)

(*
  TODO
    Way to pass down options with GenerateCode
    Support for unknown expressions passed into CForm
*)

BeginPackage["SymbolicC`"]

ToCCodeString::usage = "ToCCodeString[ symbolicC] generates a string of C code from a symbolic C expression."

GenerateCode::usage = "GenerateCode[ symbolicC] generates a string of C code from a symbolic C expression."

COperator::usage = "COperator[ oper, arg1, ...] is a symbolic representation of an operator."

CAssign::usage = "CAssign[lhs, rhs] is a symbolic representation of an assignment statement."

CFunction::usage = "CFunction[ type, name, args, body] is a symbolic representation of a function definition. CFunction[ type, name, args] is a symbolic representation of a function declaration."

CArray::usage = "CArray[ name, args] is a symbolic representation of an array."

CMember::usage = "CMember[ obj, mem] is a symbolic representation of access from a struct."

CPointerMember::usage = "CPointerMember[ obj, mem] is a symbolic representation of access from a pointer to a struct."

CCast::usage = "CCast[ type, obj] is a symbolic representation of a cast of obj to type."

CDereference::usage = "CDereference[ obj] is a symbolic representation of the dereferencing of a pointer."

CPointerType::usage = "CPointerType[ type] is a symbolic representation of a type that is a pointer to a type."

CFunctionPointerType::usage = "CFunctionPointerType[ type, name, args] is a symbolic representation of a type that is a pointer to a function that returns type and has arguments args."

CAddress::usage = "CAddress[ obj] is a symbolic representation of the address of an object."

CSizeOf::usage = "CSizeOf[ obj] is a symbolic representation of the sizeof expression."

CStatement::usage = "CStatement[ obj] is a symbolic representation of a statement. "

CBlock::usage = "CBlock[ args] is a symbolic representation of a block of statements.  "

CDeclare::usage = "CDeclare[ type, var] is a symbolic representation of a variable declaration. CDeclare[ type, {vars...}] declares a number of variables."

CTypedef::usage = "CTypedef[ type, var] is a symbolic representation of a type declaration."

CCall::usage = "CCall[ fname, args] is a symbolic representation of a call to a function."

CReturn::usage = "CReturn[ ] is a symbolic representation of return from a function. CReturn[ arg] returns the argument arg."

CConditional::usage = "CConditional[ test, trueArg, falseArg] is a symbolic representation of an inline conditional expression."

CIf::usage = "CIf[ test, trueArg, falseArg] is a symbolic representation of a conditional statement. CIf[ test, trueArg] only has a branch if test is True."

CGoto::usage = "CGoto[ label] is a symbolic representation of a goto statement."

CLabel::usage = "CLabel[ label] is a symbolic representation of a label."

CProgram::usage = "CProgram[ args] is a symbolic representation of an entire program."

CInclude::usage = "CInclude[ header] is a symbolic representation of a preprocessor include statement."

CString::usage = "CString[ string] is a symbolic representation of a string expression."

CDo::usage = "CDo[ body, test] is a symbolic representation of a do/while statement."

CWhile::usage = "CWhile[ test, body] is a symbolic representation of a while statement."

CFor::usage = "CFor[ init, test, incr, body] is a symbolic representation of a for loop."

CContinue::usage = "CContinue[] is a symbolic representation of a continue statement."

CBreak::usage = "CBreak[] is a symbolic representation of a break statement."

CDefault::usage = "CDefault[] is a symbolic representation of a default statement."

CSwitch::usage = "CSwitch[ cond, statements] is a symbolic representation of a switch statement."

CEnum::usage = "CEnum[ members] is a symbolic representation of an enum definition."

CUnion::usage = "CUnion[ name, members] is a symbolic representation of a union. CUnion[ name] declares a union without specifying the contents. CUnion[ None, members] does not give the union a name."

CStruct::usage = "CStruct[ name, members] is a symbolic representation of a struct. CStruct[ name] declares a struct without specifying the contents. CStruct[ None, members] does not give the struct a name."

CComment::usage = "CComment[ text] is a symbolic representation of a comment. CComment[ text, {pre, post}] includes text to add before and after the comment."

CPrecedence::usage = "CPrecedence[ oper] returns the precedence of an operator."

CLine::usage = "CLine[ line] is a symbolic representation of a preprocessor line directive."

CPragma::usage = "CPragma[ line] is a symbolic representation of a preprocessor pragma directive."

CError::usage = "CError[ line] is a symbolic representation of a preprocessor error directive."
	
CPreprocessorIfdef::usage = "CPreprocessorIfdef[ cond] is a symbolic representation of a preprocessor ifdef conditional. CPreprocessorIfdef[ cond, true, false] represents the true and false cases."

CPreprocessorIfndef::usage = "CPreprocessorIfndef[ cond] is a symbolic representation of a preprocessor ifndef conditional. CPreprocessorIfndef[ cond, true, false] represents the true and false cases."

CPreprocessorElse::usage = "CPreprocessorElse[ ] is a symbolic representation of a preprocessor else conditional." 

CPreprocessorElif::usage = "CPreprocessorElif[ cond] is a symbolic representation of a preprocessor elif conditional."

CPreprocessorEndif::usage = "CPreprocessorEndif[ ] is a symbolic representation of a preprocessor endif conditional."

CDefine::usage = "CDefine[ def] is a symbolic representation of a preprocessor define."

CUndef::usage = "CUndef[ def] is a symbolic representation of a preprocessor undef."

CPreprocessorIf::usage = "CPreprocessorIf[ cond] is a symbolic representation of a preprocessor if conditional. CPreprocessorIf[ cond, true, false] represents the true and false cases."

CParentheses::usage = "CParentheses[ symb] adds parentheses around an expression."

CStandardMathOperator::usage = "CStandardMathOperator[ oper, args] is a symbolic representation of a standard math operator."

CExpression::usage = "CExpression[ arg] is a symbolic representation of code that will format using CForm[arg]."

CConstant::usage = "CConstant[ val, lab] is a symbolic representation of a constant."

Begin["`Private`"]
(* Implementation of the package *)

Needs["SymbolicC`Utilities`"]

ToCCodeString::unk = "An unknown element, `1`, was found when generating code";



(* ::Section:: *)
(* Precedence *)





CPrecedence[_?AtomQ] = 50;

CPrecedence[CString] = 29
CPrecedence[CConstant] = 29

CPrecedence[CExpression] = 29

CPrecedence[CFunction] = 28
CPrecedence[CArray] = 28
CPrecedence[Decrement] = 28
CPrecedence[Increment] = 28
CPrecedence[CMember] = 28
CPrecedence[CPointerMember] = 28

CPrecedence[PreDecrement] = 26
CPrecedence[PreIncrement] = 26
CPrecedence[Minus] = 26
CPrecedence[BitNot] = 26
CPrecedence[Not] = 26
CPrecedence[CCast] = 26
CPrecedence[CDereference] = 26
CPrecedence[CPointerType] = 26
CPrecedence[CFunctionPointerType] = 26
CPrecedence[CAddress] = 26
CPrecedence[CSizeOf] = 26

CPrecedence[Mod] = 24
CPrecedence[Divide] = 24
CPrecedence[Times] = 24

CPrecedence[Subtract] = 22
CPrecedence[Plus] = 22

CPrecedence[BitShiftRight] = 20
CPrecedence[BitShiftLeft] = 20

CPrecedence[LessEqual] = 18
CPrecedence[GreaterEqual] = 18
CPrecedence[Less] = 18
CPrecedence[Greater] = 18

CPrecedence[Unequal] = 16
CPrecedence[Equal] = 16

CPrecedence[BitAnd] = 14

CPrecedence[BitXor] = 12

CPrecedence[BitOr] = 10

CPrecedence[And] = 8

CPrecedence[Or] = 6

CPrecedence[CConditional] = 4

CPrecedence[AddTo] = 2
CPrecedence[SubtractFrom] = 2
CPrecedence[TimesBy] = 2
CPrecedence[DivideBy] = 2
CPrecedence["ModBy"] = 2
CPrecedence["BitXorAssign"] = 2
CPrecedence["BitOrAssign"] = 2
CPrecedence["BitAndAssign"] = 2
CPrecedence["BitShiftLeftAssign"] = 2
CPrecedence["BitShiftRightAssign"] = 2
CPrecedence[Set] = 2

CPrecedence[_CParentheses] = 0
CPrecedence["Comma"] = 0

CPrecedence[COperator[oper_, _]] := CPrecedence[oper]
CPrecedence[CAssign[oper_, _, _]] := CPrecedence[oper]
CPrecedence[_CCall] := CPrecedence[CCall]
CPrecedence[_CStandardMathOperator] := CPrecedence[CStandardMathOperator] 
CPrecedence[CAssign[ _,_]] := CPrecedence[Set]
CPrecedence[_CConditional] := CPrecedence[CConditional]
CPrecedence[_CArray] := CPrecedence[CArray]
CPrecedence[_CCast] := CPrecedence[CCast]
CPrecedence[_CDereference] := CPrecedence[CDereference]
CPrecedence[_CPointerType] := CPrecedence[CPointerType]
CPrecedence[_CFunctionPointerType] := CPrecedence[CFunctionPointerType]
CPrecedence[_CPointerMember] := CPrecedence[CPointerMember]
CPrecedence[_CAddress] := CPrecedence[CAddress]
CPrecedence[_CMember] := CPrecedence[ CMember]
CPrecedence[_CSizeOf] := CPrecedence[ CSizeOf]
CPrecedence[_CExpression] := CPrecedence[ CExpression]
CPrecedence[_CConstant] := CPrecedence[ CConstant]
CPrecedence[_CString] := CPrecedence[ CString]


IsCExpression[ arg_] := False
IsCExpression[ x_?AtomQ] := True
IsCExpression[ _CString] := True
IsCExpression[ _COperator] := True
IsCExpression[ _CCall] := True
IsCExpression[ _CAssign] := True
IsCExpression[ _CConditional] := True
IsCExpression[ _CPointerMember] := True
IsCExpression[ _CMember] := True
IsCExpression[ _CDereference] := True
IsCExpression[ _CPointerType] := True
IsCExpression[ _CFunctionPointerType] := True
IsCExpression[ _CAddress] := True
IsCExpression[ _CArray] := True
IsCExpression[ _CCast] := True
IsCExpression[ _CParentheses] := True
IsCExpression[ _CSizeOf] := True
IsCExpression[ _CStandardMathOperator] := True
IsCExpression[ _CStruct] := True
IsCExpression[ _CUnion] := True
IsCExpression[ _CExpression] := True
IsCExpression[ _CConstant] := True

(* ::Section:: *)
(* Name Specification *)

CAssignName[ Set] = "="
CAssignName[ DivideBy] = "/="
CAssignName[ TimesBy] = "*="
CAssignName[ AddTo] = "+="
CAssignName[ SubtractFrom] = "-="
CAssignName[ "ModBy"] = "%="
CAssignName[ "BitShiftRightAssign"] = ">>="
CAssignName[ "BitShiftLeftAssign"] = "<<="
CAssignName[ "BitAndAssign"] = "&="
CAssignName[ "BitOrAssign"] = "|="
CAssignName[ "BitXorAssign"] = "^="
CAssignName[ arg_] :=
	(Message[ ToCCodeString::unk, arg];ToString[ "CUnknownElement"[ arg]])


COperatorName[ Minus] = "-"
COperatorName[ PreDecrement] = "--"
COperatorName[ Decrement] = "--"
COperatorName[ PreIncrement] = "++"
COperatorName[ Increment] = "++"

COperatorName[ Greater] = ">"
COperatorName[ Less] = "<"
COperatorName[ GreaterEqual] = ">="
COperatorName[ LessEqual] = "<="
COperatorName[ Equal] = "=="
COperatorName[ Unequal] = "!="

COperatorName[ Mod] = "%"
COperatorName[ Divide] = "/"
COperatorName[ Times] = "*"
COperatorName[ Subtract] = "-"
COperatorName[ Plus] = "+"
COperatorName[ BitShiftLeft] = "<<"
COperatorName[ BitShiftRight] = ">>"
COperatorName[ BitAnd] = "&"
COperatorName[ BitNot] = "~"
COperatorName[ BitXor] = "^"
COperatorName[ BitOr] = "|"
COperatorName[ And] = "&&"
COperatorName[ Or] = "||"
COperatorName[ Not] = "!"

COperatorName[ arg_] :=
	(Message[ ToCCodeString::unk,  arg];ToString[ "CUnknownElement"[ arg]])

(* ::Section:: *)
(* Code Formating *)
Options[Tabify] = {Embedded -> False, "Indent" -> None}
Tabify[str_String, opts:OptionsPattern[]] :=
	Module[{tabOpt = OptionValue[Tabify, {opts}, "Indent"], tabs},
		If[tabOpt === Automatic,
			tabOpt = 1;
		];
		If[StringQ[tabOpt] || (IntegerQ[tabOpt] && Positive[tabOpt]),
			tabs = If[StringQ[tabOpt],
				tabOpt,
				StringJoin[ConstantArray["\t", tabOpt]]
			]; 
			StringJoin[
				Riffle[
					Riffle[StringSplit[str, "\n"], "\n"],
					tabs,
					{1, -1, 3}
				]
			] <> "\n",
			str
		]
	]
Tabify[x__, opts:OptionsPattern[]] := Tabify[GenerateCode[x], opts]

(* ::Section:: *)
(* Code Generation *)

(*
 Toplevel function
*)
Options[ToCCodeString] = Options[GenerateCode] = {Embedded -> False, "Indent" -> Automatic}
ToCCodeString[ arg_, opts:OptionsPattern[]] := 
	With[{
		str = If[ IsCExpression[arg] || ListQ[ arg] || MatchQ[ arg, _CComment ], 
			GenerateCode[ arg, opts],
			GenerateCode[ {arg}, opts]
		],
		indent = Lookup[<|opts|>, "Indent", OptionValue[ToCCodeString, "Indent"]]
	},
		Which[
			indent === Automatic || indent === "ClangFormat",
				ClangFormatString[str],
			True,
				str
		] /; StringQ[str]
	]



GenerateCode[CParentheses[ ], opts:OptionsPattern[]] := 
	"( )"
GenerateCode[CParentheses[CParentheses[x_]], opts:OptionsPattern[]] := 
	GenerateCode[CParentheses[x], opts]
	
GenerateCode[CParentheses[args__], opts:OptionsPattern[]] := 
	("(" <> GenerateCode[args, opts] <> ")")
	
GenerateCode[COperator[ oper_, arg_], opts:OptionsPattern[]] := 
	GenerateCode[ COperator[ oper, {arg}], opts]

GenerateCode[COperator[ oper:(Decrement|Increment), {arg_}], opts:OptionsPattern[]] := 
	(GenerateCodeWithParenthesis[ arg, CPrecedence[oper], opts] <> COperatorName[ oper])

GenerateCode[COperator[ oper:(Minus|Plus|PreIncrement|PreDecrement|Not|BitNot), {arg_}], opts:OptionsPattern[]] := 
	(COperatorName[ oper] <> GenerateCodeWithParenthesis[ arg, CPrecedence[oper], opts])

GenerateCode[COperator[ oper_, args_List], opts:OptionsPattern[]] := 
	If[Length[args] == 1,
		GenerateCode[First@args, opts],
	 	InfixFormat[CPrecedence[oper], COperatorName[oper], args]
	]




GenerateCode[a_Symbol, opts:OptionsPattern[]] := 
	ToString[a]

GenerateCode[a_String, opts:OptionsPattern[]] := 
	a

GenerateCode[a_Rational, opts:OptionsPattern[]] := 
	GenerateCode[COperator[Divide, {GenerateCode[Numerator[a], opts], GenerateCode[Denominator[a], opts]}], opts]
	
GenerateCode[a_Integer, opts:OptionsPattern[]] := 
	ToString[CForm[a]]

GenerateCode[a_Real, opts:OptionsPattern[]] := 
	ToString[CForm[a]]

GenerateCode[CExpression[arg_Rational], opts:OptionsPattern[]] := 
	GenerateCode[arg, opts]
	
GenerateCode[CExpression[arg_Integer], opts:OptionsPattern[]] := 
	GenerateCode[arg, opts]
	
GenerateCode[CExpression[arg_Real], opts:OptionsPattern[]] := 
	GenerateCode[arg, opts]
	
GenerateCode[CExpression[arg_], opts:OptionsPattern[]] := 
	ToString[ CForm[HoldForm[arg]]]
	
GenerateCode[ CConstant[ atom_, label_String], opts:OptionsPattern[]] :=
	GenerateCode[ atom, opts] <> label

		

GenerateCode[arg___, opts:OptionsPattern[]] := 
	(Message[ ToCCodeString::unk, arg];ToString[ "CUnknownElement"[ arg]])


GenerateCode[CString[ a_CString], opts:OptionsPattern[]] := 
	GenerateCode[a, opts]
	
GenerateCode[CString[ a_String], opts:OptionsPattern[]] := 
	("\"" <> StringReplace[a, {"\\"->"\\\\","\"" -> "\\\""}] <> "\"")
	
GenerateCode[CString[ a_], opts:OptionsPattern[]] := 
	GenerateCode[CString[GenerateCode[a, Embedded->True, opts]], opts]



GenerateCode[CDeclare[ typeArg_, idArg_], opts:OptionsPattern[]] := 
	Module[ {type, id},
		type = Flatten[ {typeArg}];
		id = Flatten[ {idArg}];
		type = Riffle[ Map[ GenerateCode[#,opts]&, type], " "];
		id = Riffle[ Map[ GenerateCode[#,opts]&, id], ", "];
		type <> " " <> id <> ";"
	]

	
GenerateCode[CTypedef[ typeArg_, idArg_], opts:OptionsPattern[]] := 
	GenerateCode[CDeclare[Join[{"typedef"}, List[typeArg]], idArg], opts]

GenerateCode[CAssign[ id_, rhs_], opts:OptionsPattern[]] := 
	GenerateCode[ CAssign[ Set, id, rhs], opts]
	
GenerateCode[CAssign[ oper_, id_, rhsArg_], opts:OptionsPattern[]] := 
	Module[ {prec, rhs = rhsArg},
		prec = CPrecedence[ oper];
		rhs = If[ oper === Set && ListQ[ rhs], 
				"{" <> StringJoin[Riffle[ Map[ GenerateCode[#, opts]&, rhs], ", "]] <> "}",
				GenerateCode[ rhs, opts]
		];
		If[ needsParens[ rhs, CPrecedence[ oper]],
			rhs = "(" <> rhs <> ")"];
		GenerateCodeWithParenthesis[ id, prec, opts] <> " " <>
			CAssignName[ oper] <> " " <>
			rhs
	]
	
	
GenerateCode[CProgram[ args___], opts:OptionsPattern[]] := 
	(Riffle[ Map[ GenerateCode[#, opts]&, Flatten[{args}]], "\n\n"] <> "\n")



GenerateCode[CStatement[""], opts:OptionsPattern[]] := 
	""
	
GenerateCode[CStatement[ ], opts:OptionsPattern[]] := 
	""
	
GenerateCode[CStatement[ arg_], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin[
			GenerateCode[ arg, opts],
				 If[ IsCExpression[arg], ";", ""]]
	]


GenerateCode[CBlock[ arg_CBlock], opts:OptionsPattern[]] := 
	GenerateCode[ arg, opts]

GenerateCode[CBlock[ arg___], opts:OptionsPattern[]] := 
	GenerateCode[ CBlock[ {arg}], opts]


GenerateCode[CBlock[ args_List], opts:OptionsPattern[]] := 
	("{\n" <> Tabify[GenerateCode[ args, opts], opts] <> "}") 
	
	
GenerateCode[{comment_CComment, args__}, opts:OptionsPattern[]] := 
	StringJoin[ GenerateCode[ comment, opts], GenerateCode[ {args}, opts]]


(*
 Format lists, making sure to add ret at the end.
 
*)

GenerateCode[ args_List, opts:OptionsPattern[]] := 
	Module[ {elem, len = Length[ args]},
		StringJoin[ 
			Table[ 
				elem = Part[args,i];
				If[ i < len && MatchQ[ Part[args, i+1], _CComment],
					{
						GenerateCode[ elem, Embedded -> True, opts], 
						If[ IsCExpression[elem], ";", ""]
					}
					,
					{
						GenerateCode[ CStatement[elem],opts], 
						If[ ListQ[ elem] || MatchQ[ elem, CStatement[]], "", "\n"]
					}], {i, len}]]
	]



GenerateCode[CComment[ CComment[comment_], format_:{}], opts:OptionsPattern[]] := 
	GenerateCode[CComment[comment, format], opts]	
	
GenerateCode[CComment[ comment_, formatIn_:{}], opts:OptionsPattern[]] := 
	Module[ {pre = "", post = "", format},
		format = If[ ListQ[ formatIn], formatIn, {formatIn}];
		Switch[Length[format], 
			1, pre = GenerateCode[First[format], opts],
			2, {pre, post} = GenerateCode[#, opts]& /@ format
		];
		pre <> "/*  " <> GenerateCode[comment, opts] <> "  */" <> post
	]





GenerateCode[CReturn[ arg_], opts:OptionsPattern[]] := 
	StringJoin[ "return ", GenerateCode[arg, opts], ";"]

GenerateCode[CReturn[ ], opts:OptionsPattern[]] := 
	"return;"

GenerateCode[CCall[ id_, args_], opts:OptionsPattern[]] := 
	(GenerateCode[ id, opts] <> "(" <> Riffle[ Map[ GenerateCode[#, opts]&, Flatten[ {args}]], ", "] <> ")")



GenerateCode[CFunction[typeArg_, id_, args_List,body_], opts:OptionsPattern[]] := 
	Module[ {type},
		type = Flatten[ {typeArg}];
		type = Riffle[ Map[ GenerateCode[#, opts]&, type], " "];
		type <> " " <> GenerateCode[ id, opts] <> 
		"(" <> Riffle[ Map[ formatArgument, args], ", "] <> ")\n" <> GenerateCode[CBlock[ body], opts]
	]

GenerateCode[CFunction[typeArg_, id_, args_List], opts:OptionsPattern[]] := 
	Module[ {type},
		type = Flatten[ {typeArg}];
		type = Riffle[ Map[ GenerateCode[#, opts]&, type], " "];
		type <> " " <> GenerateCode[ id, opts] <> 
		"(" <> Riffle[ Map[ formatArgument[#, opts]&, args], ", "] <> ");"
	]

formatArgument[ args_, opts:OptionsPattern[]] :=
	Riffle[ Map[ GenerateCode[#, opts]&, Flatten[ {args}]], " "]

	
GenerateCode[CConditional[ test_, true_, false_], opts:OptionsPattern[]] := 
	(GenerateCodeWithParenthesis[ test, CPrecedence[ CConditional], opts] <>
	" ? " <> GenerateCodeWithParenthesis[ true, CPrecedence[ CConditional], opts] <>
	" : " <> GenerateCodeWithParenthesis[ false, CPrecedence[ CConditional], opts])

GenerateCode[CIf[ test_, true_], opts:OptionsPattern[]] := 
	Module[ {},
		("if( "<> GenerateCode[ test, opts] <> ")\n" <> GenerateCode[ CBlock[true], opts])
	]

GenerateCode[CIf[ test_, true_, false_], opts:OptionsPattern[]] := 
	Module[ {},
		("if( "<> GenerateCode[ test, opts] <> ")\n" <> GenerateCode[ CBlock[true], opts] <> 
			"\nelse\n" <>  GenerateCode[ CBlock[false], opts])
	]

GenerateCode[CIf[ test_, true_, false_CIf], opts:OptionsPattern[]] := 
	Module[ {},
		("if( "<> GenerateCode[ test, opts] <> ")\n" <> GenerateCode[ CBlock[true], opts] <> 
		"\nelse " <> GenerateCode[ false, opts])
	]

GenerateCode[CDo[ body_, test_], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin[
			"do", trimIfString[GenerateCode[CBlock[body], opts]], " while( ", GenerateCode[test, opts], ");"
		]
	]
	
GenerateCode[CWhile[ test_, body_], opts:OptionsPattern[]] := 
	Module[ {},
		("while( "<> GenerateCode[ test, opts] <> ")\n" <> GenerateCode[ CBlock[body], opts])
	]

GenerateCode[CFor[ init_CDeclare, test_, incr_, body_], opts:OptionsPattern[]] := 
	GenerateCode[CFor[{init}, test, incr, body], opts]
	
GenerateCode[CFor[ init_, test_, incr_, body_], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin[
			"for( ",
			If[ListQ[init],
				Riffle[(trimIfString[GenerateCode[#, opts], (Whitespace | ";") ...])& /@ init, ", "], 
				GenerateCode[ init, opts]
			],
			"; ",
			GenerateCode[ test, opts],
			"; ",
			If[ListQ[incr],
				Riffle[(trimIfString[GenerateCode[#, opts], (Whitespace | ";") ...])& /@ incr, ", "],
				GenerateCode[ incr, opts]
			],
			")\n",
			GenerateCode[ CBlock[body], opts]
		]
	]



GenerateCode[CPointerMember[ obj_, val_], opts:OptionsPattern[]] := 
	InfixFormat[CPrecedence[CPointerMember], "->", {obj, val}, False, opts]


GenerateCode[CMember[ obj_, val_], opts:OptionsPattern[]] := 
	InfixFormat[CPrecedence[CMember], ".", {obj, val}, False, opts]

GenerateCode[CDereference[ obj_], opts:OptionsPattern[]] := 
	("*" <> GenerateCodeWithParenthesis[ obj, CPrecedence[ CDereference], opts])

GenerateCode[CPointerType[ typeArg_], opts:OptionsPattern[]] := 
	Module[ {type},
		type = Flatten[ {typeArg}];
		If[ MatchQ[ typeArg, _CPointerType],
			GenerateCode[ typeArg, opts],
			type = Flatten[ {typeArg}];
			type = Riffle[ Map[ GenerateCode[#, opts]&, type], " "];
			type] <> "*"
	]


GenerateCode[CFunctionPointerType[ type_, name_, args_], opts:OptionsPattern[]] := Module[
	{value},
	value = Flatten[{type}];
	value = Riffle[ Map[ GenerateCode[#, opts]&, value], " "];
	value <> " " <> "(*" <> GenerateCode[ name, opts] <> ")" <> 
		"(" <> Riffle[ Map[ formatArgument[#, opts]&, args], ", "] <> ")"	
];

GenerateCode[CAddress[ obj_], opts:OptionsPattern[]] := 
	("&" <> GenerateCodeWithParenthesis[ obj, CPrecedence[ CAddress], opts])

GenerateCode[CCast[ typeArg_, obj_], opts:OptionsPattern[]] := 
	Module[ {type},
		type = Flatten[ {typeArg}];
		type = Riffle[ Map[ GenerateCode[#, opts]&, type], " "];
		"(" <> type <> ")" <> " " <> 
			GenerateCodeWithParenthesis[ obj, CPrecedence[ CCast], opts]
	]


GenerateCode[CGoto[ label_Symbol | label_String], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin[ "goto ", GenerateCode[label, opts], ";"]
	]

GenerateCode[CLabel[ label_Symbol | label_String], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin[ GenerateCode[label, opts], ":"]
	]

GenerateCode[CArray[ args_List], opts:OptionsPattern[]] := 
	"{" <> StringJoin[Riffle[GenerateCode[#, opts]& /@ args, ", "]] <> "}"

GenerateCode[CArray[ id_, arg_], opts:OptionsPattern[]] := 
	GenerateCode[ CArray[ id, {arg}], opts]

GenerateCode[CArray[ id_, args_List], opts:OptionsPattern[]] := 
	Module[ {},
		(GenerateCodeWithParenthesis[id, CPrecedence[ CArray], All, opts] <> 
			"[" <> Riffle[ Map[ GenerateCode[#, opts]&, args], "]["] <> "]")
	]


GenerateCode[CArray[id_], opts:OptionsPattern[]] := Module[
	{},
	(GenerateCodeWithParenthesis[id, CPrecedence[CArray], All, opts] <> "[]")
];

GenerateCode[CSizeOf[ arg_], opts:OptionsPattern[]] := 
	("sizeof( " <> GenerateCode[ arg, opts] <> ")")
	

GenerateCode[CContinue[], opts:OptionsPattern[]] := 
	"continue;"

GenerateCode[CBreak[], opts:OptionsPattern[]] := 
	"break;"
	
	
(*
 Adds the : but no return
*)

GenerateCode[CDefault[], opts:OptionsPattern[]] := 
	"default:"


(*
 Format Switch,  add the list around the statements to get nesting and \n correct
*)

formatSwitchArgument[ {label_, args__}, opts:OptionsPattern[]] :=
	StringJoin[
		Which[
			trimIfString[label] === "default", "default:",
			label === CDefault[], GenerateCode[ label, opts],
			True, {"case ", GenerateCode[ label, opts], ":"}]
		,
		"\n"
		,
		GenerateCode[ {args}, opts]
	]

GenerateCode[ CSwitch[cond_, stmtsIn__], opts:OptionsPattern[]] /; EvenQ[Length[{stmtsIn}]] :=
	Module[ {stms},
		stms = Partition[ {stmtsIn}, 2];
		StringJoin[
			"switch ( ", GenerateCode[cond, opts], ")\n"
			,
			"{\n",
				Tabify[StringJoin[Riffle[ Map[formatSwitchArgument[ #, opts]&, stms], "\n"]], opts],
			"}"
		]
	]


GenerateCode[CEnum[members_List], opts:OptionsPattern[]] := 
	GenerateCode[ CEnum["", members], opts]
	
GenerateCode[CEnum[tag_, members_List], opts:OptionsPattern[]] := 
	GenerateCode[ CEnum[tag, members, ""], opts]	
	
GenerateCode[CEnum[tag_, members_List, name_], opts:OptionsPattern[]] := 
	Module[ {},
		("enum" <> If[tag =!= "", " " <> GenerateCode[tag, opts], ""] <> " {\n" <>
			GenerateCode[
				StringJoin[Riffle[ GenerateCode[#, opts]& /@ members, ",\n"]]
			] <>
		"\n}" <> If[name =!= "", " " <> GenerateCode[name, opts], ""] <> ";" )
	]


	
GenerateCode[CUnion[name_, members_:None], opts:OptionsPattern[]] := 
	cContainer["union", name, members, opts]

GenerateCode[CStruct[name_, members_:None], opts:OptionsPattern[]] := 
	cContainer["struct", name, members, opts]


		
cContainer[tag_, name_, membersIn_, opts:OptionsPattern[]] :=
	Module[ {members},
		StringJoin[ 
			tag
			,
			If[ name =!= None, {" ", GenerateCode[name, opts]}, ""]
			,
			If[ ListQ[membersIn], 
				members = Map[
					If[ListQ[#],
						Apply[
							If[tag === "struct",
								CDeclareBitCode[##, opts],
								CDeclare[##]
							]&,
							#
						],
						#
					]&,
					membersIn
				];
				{
				"\n",
				GenerateCode[ CBlock[ members], opts]
				}
				,
				""
				]
		]
	]

CDeclareBitCode[varType_, varName_, bits_ /; !MatchQ[bits, OptionsPattern[]], opts:OptionsPattern[]] :=
	CDeclare[varType, GenerateCode[varName, opts] <> ":" <> GenerateCode[bits, opts]]

CDeclareBitCode[varType_, varName_, opts:OptionsPattern[]] :=
	CDeclare[varType, varName]			
		

(* ::Section:: *)
(* CStandardMathOperator Section *)


CStandardMathFunction[ ArcCos, 1] := "acos"
CStandardMathFunction[ ArcSin, 1] := "asin"
CStandardMathFunction[ ArcTan, 1] := "atan"
CStandardMathFunction[ ArcTan, 2] := "atan2"
CStandardMathFunction[ Ceiling, 1] := "ceil"
CStandardMathFunction[ Cos, 1] := "cos"
CStandardMathFunction[ Cosh, 1] := "cosh"
CStandardMathFunction[ Exp, 1] := "exp"
CStandardMathFunction[ Abs, 1] := "fabs"
CStandardMathFunction[ Floor, 1] := "floor"
CStandardMathFunction[ "frexp", 1] := "frexp"
CStandardMathFunction[ "ldexp", 1] := "ldexp"
CStandardMathFunction[ Log, 1] := "log"
CStandardMathFunction[ Log10, 1] := "log10"
CStandardMathFunction[ "log10", 1] := "log10"
CStandardMathFunction[ "modf", 2] := "modf"
CStandardMathFunction[ Power, 2] := "pow"
CStandardMathFunction[ Sin, 1] := "sin"
CStandardMathFunction[ Sinh, 1] := "sinh"
CStandardMathFunction[ Sqrt, 1] := "sqrt"
CStandardMathFunction[ Tan, 1] := "tan"
CStandardMathFunction[ Tanh, 1] := "tanh"

GenerateCode[CStandardMathOperator[ fun_, arg:Except[_List]], opts:OptionsPattern[]] := 
	GenerateCode[ CStandardMathOperator[ fun, {arg}], opts]

GenerateCode[CStandardMathOperator[ fun_, args_List] /; 
            StringQ[CStandardMathFunction[ fun, Length[ args]]], opts:OptionsPattern[]] := 
	Module[ {oper},
		oper = CStandardMathFunction[ fun, Length[ args]];
		GenerateCode[ CCall[ oper, args], opts]	
	]
	






(* ::Section:: *)
(* Preprocessor Section *)

GenerateCode[CLine[ln_, comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["line", ln, comment, opts]
	
GenerateCode[CPragma[pragma_, comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["pragma", pragma, comment, opts]
	
GenerateCode[CError[err_, comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["error", err, comment, opts]


GenerateCode[CUndef[def_, comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["undef", def, comment, opts]
	

	
formatPreprocessorDirective[directive_String, stmt_, comment_:Null, opts:OptionsPattern[]] :=
	Module[ {},
		StringJoin[
			"#", directive,
				If[ stmt =!= Null,
					{" ", GenerateCode[stmt, opts]}, {}],
			If[comment =!= Null && Head[comment] === CComment,
				GenerateCode[comment, opts],
				{}
			]
		]
	]
	
GenerateCode[CDefine[def_, val_:Null], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin["#define ", GenerateCode[def, opts],  
			If[val =!= Null, {"  ", GenerateCode[val, opts]}, ""]]
	]


GenerateCode[CInclude[{}], opts:OptionsPattern[]] := 
	""


GenerateCode[CInclude[{args___, arg_}], opts:OptionsPattern[]] := 
	Module[ {},
		StringJoin[ 
			GenerateCode[ Map[ CInclude, {args}], opts],
			GenerateCode[ CInclude[arg], opts]
			]
	]
	
	
GenerateCode[CInclude[fn_], opts:OptionsPattern[]] := 
	Module[ {},
		("#include " <> If[ StringQ[fn] && StringMatchQ[fn,"<*>"], fn, 
			"\"" <> GenerateCode[fn, opts] <> "\""])
	]

GenerateCode[CPreprocessorIf[cond_], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["if", cond, Null, opts]
	
GenerateCode[CPreprocessorIf[cond_, true_, false_:Null], opts:OptionsPattern[]] := 
	GeneratePreprocessorCode[CPreprocessorIf, cond, true, false, opts]

GenerateCode[CPreprocessorIfdef[cond_], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["ifdef", cond, Null, opts]
	
GenerateCode[CPreprocessorIfdef[cond_, true_, false_:Null], opts:OptionsPattern[]] := 
	GeneratePreprocessorCode[CPreprocessorIfdef, cond, true, false, opts]

GenerateCode[CPreprocessorIfndef[cond_], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["ifndef", cond, Null, opts]
	
GenerateCode[CPreprocessorIfndef[cond_, true_, false_:Null], opts:OptionsPattern[]] := 
	GeneratePreprocessorCode[CPreprocessorIfndef, cond, true, false, opts]

GenerateCode[CPreprocessorElse[ comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["else", Null, Null, opts]

GenerateCode[CPreprocessorElif[ cond_, comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["elif", cond, Null, opts]

GenerateCode[CPreprocessorEndif[ comment_:Null], opts:OptionsPattern[]] := 
	formatPreprocessorDirective["endif", Null, Null, opts]


GeneratePreprocessorCode[directive_, condIn_, trueIn_, falseIn_, opts:OptionsPattern[]] :=
	Module[ {cond, true, false},
		cond = If[ ListQ[ condIn], condIn, {condIn}];
		true = If[ ListQ[ trueIn], trueIn, {trueIn}];
		false = If[ ListQ[ falseIn], falseIn, {falseIn}];
		StringJoin[{
			"#if",
			Switch[directive,
				CPreprocessorIf, " ",
				CPreprocessorIfdef, "def ",
				CPreprocessorIfndef, "ndef ",
				_, "Unknown"
			],
			Map[ GenerateCode[#, opts]&, cond],
			"\n",
			GenerateCode[ true, opts],
			If[ falseIn =!= Null, 
				StringJoin[
					"#else\n",
					GenerateCode[ false, opts]],
					""],
			"#endif"
		}]
	]


(* ::Section:: *)
(* Formatting *)

trimIfString[s_] := If[StringQ[s], StringTrim[s], s]

trimIfString[s_, pat_] := If[StringQ[s], StringTrim[s, pat], s]


InfixFormat[ prec_, oper_, args_List, addSpace_:True, opts:OptionsPattern[]] :=
    Module[ {formArgs, space},
    	space = If[ addSpace, " ", ""];
        formArgs = Map[GenerateCodeWithParenthesis[#, prec, opts] &, args];
        StringJoin[Riffle[formArgs, space <> oper <> space]]
    ]



needsParens[ arg_, precParent_, grouping_:None] :=
    Module[ {compareFun},
    	compareFun = If[ grouping === All, LessEqual, Less];
        !compareFun[precParent, CPrecedence[arg]] && Head[arg] =!= CParentheses
    ]

GenerateCodeWithParenthesis[arg_, precParent_, grouping_:None, opts:OptionsPattern[]] :=
    Module[ {res},
    	res = GenerateCode[arg, opts];
    	If[ needsParens[ arg, precParent, grouping],
             "(" <> res <> ")",
             res]
    ]
 

(* ::Section:: *)
(* Package End *)

End[]

EndPackage[]

