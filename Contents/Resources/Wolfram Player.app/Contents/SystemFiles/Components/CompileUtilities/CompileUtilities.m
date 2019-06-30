BeginPackage["CompileUtilities`"]

InitializeCompileUtilitiesClasses

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]

InitializeCompileUtilitiesClasses[] := (
	If[TrueQ[$classesInitialized],
		Return[]
	];

	(*
	DeclareCompileUtilitiesClassProfile is special because it is just a call to initialize[], and
	does not have DeclareClass appear lexically within the callback
	*)
	RunCallback["DeclareCompileUtilitiesClassProfile", {}];
	
	SortCallbacks["DeclareCompileUtilitiesClass", SortClassesFunction];
	RunCallback["DeclareCompileUtilitiesClass", {}];
	
	$classesInitialized = True;
)

End[]

EndPackage[]

