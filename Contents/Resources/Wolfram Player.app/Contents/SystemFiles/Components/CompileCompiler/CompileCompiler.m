(* Wolfram Language Package *)

(* Created by the Wolfram Workbench 06-Dec-2018 *)


BeginPackage["CompileCompiler`"]
(* Exported symbols added here with SymbolName::usage *) 

CompilePredicates

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`"]

$compiled = False

CompilePredicates[] :=
	Module[{classes = $Classes, data},
		If[ $compiled,
			Return[]];
		$compiled = True;
		data = Map[ compilePredicate, classes];
		data = DeleteCases[data, Null];
		Apply[Set, data, {1}];
	]

cnt = 0;

compilePredicate[ className_] :=
	Module[{info = Compile`Utilities`Class`Impl`ClassInformation[className], pred, funName},
		Print[cnt++];
		pred = Lookup[Lookup[info, "Methods"], "_predicate", Null];
		If[!MatchQ[pred, Function[{}, _Symbol]],
			Return[Null]];
		funName = Part[pred, 2];
		If[Head[funName] =!= Symbol,
			Return[Null]];
		(*
		
		*)
		{funName, compilePredicate1[ Context@@{className} <> SymbolName[className]]}
	]

compilePredicate1[className_] :=
	Module[{func = Function[{Typed[arg, "Expression"]}, 
				CompileClassSystem`ObjectInstanceQ[arg] &&
 					Native`Equal[ CompileClassSystem`MClassName[arg], 
  						Typed[className, "CString"]]]},
  			CompileToCodeFunction[func]
	]
					

End[]

EndPackage[]

