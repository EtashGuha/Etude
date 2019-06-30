


(*
	A way to handle initialization for the Compiler
	
	TODO,  
	    add dependencies,  can be done with a simple topological sort
	    automatic invocations
	    ensuring that each init is only run once,  not sure about this
	    really we might want to run multiple times
	
	But this is good enough to get started
*)



BeginPackage["CompileUtilities`Callback`"]


RunCallback

RegisterCallback

SortCallbacks

SortClassesFunction

FinalizeCallbacks

Begin["`Private`"]

Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileUtilities`ClassSystem`"]


$CallbackFunctions = <||>

$executedCallbacks = 0

RegisterCallback[ name_String, fun_] :=
	Module[ {funs},
		funs = Lookup[$CallbackFunctions, name, {}];
		funs = Append[ funs, fun];
		$CallbackFunctions[name] = funs;
	]
RegisterCallback[args___] :=
	ThrowException[{"Unrecognized call to RegisterCallback", args}]

SortCallbacks[name_String, sortFunction_] :=
	Module[{funs},
		funs = Lookup[$CallbackFunctions, name, {}];
		funs = sortFunction[funs];
		$CallbackFunctions[name] = funs;
	]
SortCallbacks[args___] :=
	ThrowException[{"Unrecognized call to SortCallbacks", args}]

RunCallback[ name_String, arg___] :=
	Module[ {funs},
		funs = Lookup[$CallbackFunctions, name, {}];
		Scan[#[arg]&, funs];
		$executedCallbacks += Length[funs];
	]
	
RunCallback[args___] :=
	ThrowException[{"Unrecognized call to RunCallback", args}]

FinalizeCallbacks[] := (
	Null
)


FinalizeCallbacks[args___] :=
	ThrowException[{"Unrecognized call to FinalizeCallbacks", args}]
	


SortClassesFunction[funs_] :=
	Module[{names, extends, g, sort, newFuns, edges, vertices, tags},
		(*
		verify that DeclareClass[] occurs in the function before proceeding
		*)
		If[FreeQ[#, HoldPattern[DeclareClass][tag_, ___]],
			ThrowException[{"Improper class declaration. The pattern DeclareClass does not occur inside the function", #}]
		] & /@ funs;

		(* all classes have a 1st arg tag *)
		tags = Association[Flatten[Cases[#, HoldPattern[DeclareClass][tag_, ___] :> (tag -> #), Infinity] & /@ funs]];

		names = Association[Flatten[Cases[#, HoldPattern[Set][name_, HoldPattern[DeclareClass][tag_, ___]] :> (name -> tag), Infinity] & /@ funs]];

		extends = Association[Flatten[Cases[#, HoldPattern[DeclareClass][tag_, _, _, opts___] :> (tag -> <|opts|>), Infinity] & /@ funs]];
		extends = DeleteCases[Flatten[{Lookup[#, Extends, {}]}], _ClassTrait]& /@ extends;

		edges = Association[Flatten[Map[Thread, Normal[extends]]]];
		edges = KeyValueMap[
					(#1 -> Lookup[names, #2,
								If[MatchQ[#2, _Class],
									(* class is already loaded *)
									#2[[1]],
									ThrowException[{"Cannot sort classes. Cannot lookup class name.", #1, #2}]]])&, edges];
		edges = Normal[edges];

		vertices = Keys[tags];

		g = Graph[vertices, edges];
		sort = TopologicalSort[g];

		(* The original order was Class1 -> Class2, with Class1 Extends Class2, but we want the reverse *)
		sort = Reverse[sort];

		(* Erase tags that are not in the list given to us right now *)
		newFuns = Lookup[tags, #, Nothing]& /@ sort;

		If[Length[newFuns] =!= Length[funs],
			ThrowException[{"Cannot sort classes", Length[funs], Length[newFuns], Complement[newFuns, funs], Complement[funs,newFuns]}]
		];
		newFuns
	]


End[]





EndPackage[]

