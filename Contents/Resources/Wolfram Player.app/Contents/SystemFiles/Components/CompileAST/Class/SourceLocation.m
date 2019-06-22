
BeginPackage["CompileAST`Class`SourceLocation`"]

SourceLocation;
SourceLocationClass;
CreateSourceLocation;
SourceLocationQ;


Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
SourceLocationClass = DeclareClass[
	SourceLocation,
	<|
		"toString" -> Function[{},
			StringJoin[
				"[",
				ToString[Self["line"]],
				", ",
				ToString[Self["column"]],
				"]"
			]
		]
	|>,
	<|
		"line" -> 0,
		"column" -> 0
	|>,
	Predicate -> SourceLocationQ
]
]]

CreateSourceLocation[line_:Undefined, column_:Undefined] :=
	Module[{cls = CreateObject[SourceLocation]},
		If[line =!= Undefined,
			cls["setLine", line]
		];
		If[column =!= Undefined,
			cls["setColumn", column]
		];
		cls
	] 
	
End[]

EndPackage[]
