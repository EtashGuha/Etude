(* ::Package:: *)

(* Mathematica Package *)

(* Created by the Wolfram Workbench Dec 13, 2010 *)

BeginPackage["LibraryLink`LibraryLinkTemplate`", {"SymbolicC`"}]
(* Exported symbols added here with SymbolName::usage *) 

LibraryTemplate::usage="LibraryTemplate[outfile,{arg1, arg2, ...}] generates a C template from a list of SymbolicC arguments, and writes them to the outfile."
LibraryTemplate::argmu="LibraryTemplate called with `1` argument, 1 or 2 arguments are expected.";
LibraryTemplate::argtype="Argument `1` is an invalid SymbolicC argument. 
						  Valid SymbolicC arguments are CFunction, CInclude, CDefine, CError, CLine, CPragma, and any CPreprocessor call."
LibraryTemplate::filename="Outfile `1` is an invalid outfile. A valid outfile argument must be a string."
LibraryTemplate::arglist="SymbolicC argument list `1` is not of the form {arg1, arg2, ...}."

LibraryFunctionTemplate::usage = "LibraryFunctionTemplate[fname,{arg1, arg2, ...},result] returns a SymbolicC template for a Wolfram library function named fname, which takes arguments of type {arg1, arg2, ...} and returns a value of type result.";
LibraryFunctionTemplate::argtype="Argument `1` is not a valid argument type. 
								  Valid types are _Integer, _Real, _Complex, boolean (True or False), or Compile's Tensor."
LibraryFunctionTemplate::funcname="The name `1` is not a valid function name. Valid function names must be strings and start with a character."
LibraryFunctionTemplate::argmu="LibraryFunctionTemplate called with `1` argument, 3 arguments are expected."
LibraryFunctionTemplate::arglist="Argument list `1` is not of the form {arg1, arg2, ...}"

Options[LibraryTemplate]= {"FullLibrary"->True,"Indented"->True,"Header"->True}

Begin["`Private`"]
(* Implementation of the package *)

Needs["SymbolicC`"]

LibraryTemplate[args_,OptionsPattern[]] :=
	If[validTemplateArgsQ[createFunctions[args]],
		Template[args,OptionValue["FullLibrary"]], 
	 	(* else, *) $Failed]


LibraryTemplate[filename_,args_,OptionsPattern[]] :=
(	If[validTemplateArgsQ[createFunctions[args]] && StringQ[filename],
		With[{path=sourceFilePath[filename]},Export[ path,
		    	CIndent[ToCCodeString[Template[args,OptionValue["FullLibrary"]]],OptionValue["Indented"]], 
		    "String"]; path],
	 	(* else, *) $Failed])

(*LibraryTemplate[args_List]:=Template[args,True]*)

LibraryTemplate[x___]:=(Message[LibraryTemplate::argmu,Length[{x}]]; $Failed)

Template[args_List,fullLibraryQ_] :=
(	If[validTemplateArgsQ[createFunctions[args]],
		If[fullLibraryQ, 
			CProgram[
				Flatten[
			  		Prepend[Join[DeleteCases[createFunctions[args],$Failed],LibraryFunctions[]],
			  		{HeaderComment[],CInclude["WolframLibrary.h"]}
				]
			]], 
			CProgram[
				Flatten[
					DeleteCases[createFunctions[args],$Failed]
			]]
		],
	(*else *) $Failed
])

LibraryFunctionTemplate[fname_, args_, result_] :=
	If[Not[MemberQ[{validFunctionArgsQ[args], validFunctionArgQ[result], validFunctionNameQ[fname]},False]],
		FunctionTemplate[fname, args, result],
		$Failed
	]

LibraryFunctionTemplate[___] := 
	(Message[LibraryFunctionTemplate::argmu]; $Failed)

FunctionTemplate[funcName_String, args_List, result_] :=
	CFunction[
  		(*type*) {"DLLEXPORT", "mint"},
  		(*name*) ToString[funcName],
  		(*args*) {{"WolframLibraryData", "libData"}, {"mint", "Argc"}, {"MArgument", "*Args"}, {"MArgument", "Res"}},
  		(*body*) {
        	varDeclare[args],
        	CStatement[CDeclare[CType[result], CName[result, "Res"]]],
        	varAssign[args],
        	CComment["Insert implementation here", {"\n","\n"}],
        	CStatement[StringJoin["MArgument_set", ToString[CLiteral[result]], "(Res,", ToString[CName[result, "Res"]], ")"]],
        	CReturn["LIBRARY_NO_ERROR"]
    	}
	]

varDeclare[args_List] := Map[declareVariable, namedNumberedArgList[args]]
	
declareVariable[arg_List] := CStatement[CDeclare[CType[First[arg]], Last[arg]]]

varAssign[args_List] := Map[assignVariable, Sort[namedNumberedArgList[args], #1[[2]] < #2[[2]] &]]

assignVariable[arg_List] := 
 CStatement[
  CAssign[Last[arg], 
   StringJoin["MArgument_get", CLiteral[First[arg]], "(Args[", 
    ToString[Part[arg, 2] - 1], "])"]]
  ]

(* Converts Mathematica types to Symbolic C types *)

		 CType[Verbatim[_Integer] | Verbatim[Integer]] := "mint"
CType[Verbatim[True | False] | Verbatim[False|True] 
				|Verbatim[Boolean]|Verbatim[_Boolean]] := "mbool"
			   CType[Verbatim[_Real] | Verbatim[Real]] := "mreal"
		 CType[Verbatim[_Complex] | Verbatim[Complex]] := "mcomplex"
		   CType[Verbatim[_String] | Verbatim[String]] := "char"
					      CType[{t_?CTypeQ, _Integer}] := "MTensor"
															   
CTypeNumberedPair[arg_List] := CType[First[arg]]

CName[arg_, pos_] := Switch[arg,
		Verbatim[String] | Verbatim[_String], "*S"<>ToString[pos],
		_,ToUpperCase[StringTake[CType[arg], {2}]] <> ToString[pos]
	]

		CLiteral[Verbatim[_Integer] | Verbatim[Integer]] := "Integer"
 CLiteral[Verbatim[True | False] | Verbatim[False|True]] := "Boolean"
		  CLiteral[Verbatim[String] | Verbatim[_String]] := "UTF8String"
			  CLiteral[Verbatim[_Real] | Verbatim[Real]] := "Real"
		CLiteral[Verbatim[_Complex] | Verbatim[Complex]] := "Complex"
						 CLiteral[{t_?CTypeQ, _Integer}] := "Tensor"

(* optional implementation in varDeclare, where each variable can have an initial value *)

		CInitialValue[Verbatim[_Integer] | Verbatim[Integer]] := "0"
 CInitialValue[Verbatim[True | False] | Verbatim[False|True]] := "0"
		  CInitialValue[Verbatim[String] | Verbatim[_String]] := "NULL"
			  CInitialValue[Verbatim[_Real] | Verbatim[Real]] := "0"
		CInitialValue[Verbatim[_Complex] | Verbatim[Complex]] := "0"
						 CInitialValue[{t_?CTypeQ, _Integer}] := "NULL"

(* Parses through a template argument list, and replaces with a LibraryFunctionTemplate if a list is found *)

createFunctions[args_List] := Map[createFunction,args]

createFunction[arg:{_,_?VectorQ,_,___}] := LibraryFunctionTemplate[arg[[1]],arg[[2]],arg[[3]]] 
createFunction[arg:{_,arg_,_,___}] := (Message[LibraryFunctionTemplate::arglist,arg]; $Failed)
createFunction[arg_] := arg

(* Ensures that the source file is given the correct extension *)

sourceFilePath[fname_String /; StringDrop[fname,{1,Length[fname]-3}]!=".c"] := fname<>".c"
sourceFilePath[fname_String /; StringDrop[fname,{1,Length[fname]-3}]==".c"] := fname

(* Converts list of arguments into a list of:
{argType,argNumber,argCName} *)

namedNumberedArgList[arg_List] := Flatten[Map[namedArgList, separatedArgList[numberedArgList[arg]]], 1]

numberedArgList[arg_List] := MapIndexed[numberedArg, arg]
numberedArg[arg_, pos_List] := List[arg, First[pos]]

namedArgList[arg_List] := MapIndexed[namedArg, arg]
namedArg[arg_, pos_List] := List[First[arg], Last[arg], CName[First[arg], First[pos] - 1]]

separatedArgList[arg_List] := GatherBy[arg, CTypeNumberedPair]

(* Input type checking *)

CTypeQ[Verbatim[_Integer] | Verbatim[Integer] | Verbatim[_Real] | Verbatim[Real]
	| Verbatim[_Complex] | Verbatim[Complex] | Verbatim[True | False] | Verbatim[False | True]
	| Verbatim[String] | Verbatim[_String] | {tensor_?CTypeQ, _Integer}] := True
CTypeQ[_] := False

(* These functions are responsible for calling most of the error messages *)

validTemplateArgsQ[args_?VectorQ] := Not[MemberQ[Map[validTemplateArgQ,DeleteCases[args,$Failed]],False]]
validTemplateArgsQ[args_] := (Message[LibraryTemplate::arglist,args]; False)

validTemplateArgQ[arg_/;MemberQ[validSymbolicC,Head[arg]]] := True
validTemplateArgQ[arg_/;Not[MemberQ[validSymbolicC,Head[arg]]]] := Message[LibraryTemplate::argtype,arg]; False

validFunctionArgsQ[args_?VectorQ] := Not[MemberQ[Map[validFunctionArgQ,args],False]]
validFunctionArgsQ[args_] := (Message[LibraryFunctionTemplate::arglist,args]; False) 

validFunctionArgQ[arg_?CTypeQ] := True
validFunctionArgQ[arg_] := (Message[LibraryFunctionTemplate::argtype,arg]; False)


validFunctionNameQ[fname_/;(StringQ[fname] && LetterQ[StringTake[fname,1]])] := True
validFunctionNameQ[fname_]:=(Message[LibraryFunctionTemplate::funcname,fname]; False)

validFileNameQ[fname_/;StringQ[fname]] := True
validFileNameQ[fname_] := (Message[LibraryTemplate::filename,fname]; False)

validSymbolicC={CFunction,CComment,CInclude,CDefine,CError,CLine,CPragma,CPreprocessorElif,CPreprocessorElse,
	CPreprocessorEndif,CPreprocessorIf,CPreprocessorIfdef,CPreprocessorIfndef,CUndef}

(* CTabbed inserts tabs into the generated CCodeString based on occurrences of { and } *)

CIndent[cstring_String,indentOption_] := 
	(indent = 0;
   		ExportString[
    		Map[ (indent = indent - StringCount[#, "}"];
       			  temp = CIndentedLine[#, indent];
       			  indent = indent + StringCount[#, "{"];
       			  temp ) &, 
       			  StringSplit[cstring, "\n"]], "List"]
   ) /; indentOption==True

CIndent[cstring_String,indentOption_]:=cstring /; indentOption==False

CIndentedLine[line_String, indent_Integer] := 
 ToString[Table["\t", {indent}] <> line]

(* Various constant functions *)

LibraryFunctions[] := {LibraryInitializationFunction[],
					   LibraryGetVersion[],
					   LibraryUnitializationFunction[]}
					   
LibraryInitializationFunction[] := 
	CFunction[{"DLLEXPORT", "mint"}, "WolframLibraryInitialize", {{"WolframLibraryData", "libdata"}}, {
		CComment["Insert initialization implementation here", {"","\n"}],
		CReturn["LIBRARY_NO_ERROR"] } 
	]
LibraryGetVersion[] := 
	CFunction[{"DLLEXPORT", "mint"}, "getWolframLibraryVersion", {},
		{ CReturn["WolframLibraryVersion"] }
	]
LibraryUnitializationFunction[] := 
	CFunction[{"DLLEXPORT", "void"}, "WolframLibraryUninitialize", {}, { 
		CComment["Insert uninitialization implementation here", {"","\n"}],
		CReturn[] } 
	]
	
HeaderComment[] := CComment[StringJoin["\n************************************************************\n",
									   "* Wolfram Library template generated by LibraryTemplate on \n",
									   "* ",DateString[],
									   "\n************************************************************"], {"","\n"}]

End[]

EndPackage[]
