BeginPackage["CompileAST`"]

InitializeCompileASTClasses

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]


InitializeCompileASTClasses[] := (
	If[TrueQ[$classesInitialized],
		Return[]
	];
	SortCallbacks["DeclareCompileASTClass", SortClassesFunction];
	RunCallback["DeclareCompileASTClass", {}];
	$classesInitialized = True;
)


End[]

EndPackage[]
