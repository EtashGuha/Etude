
BeginPackage["CompileAST`Class`SourceSpan`"]

SourceSpan;
SourceSpanClass;
CreateSourceSpan;
SourceSpanQ;


Begin["`Private`"] 

Needs["CompileAST`Class`SourceLocation`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
SourceSpanClass = DeclareClass[
	SourceSpan,
	<|
		"initialize" -> Function[{},
			Self["setStart", CreateObject[SourceLocation]];
			Self["setEnd", CreateObject[SourceLocation]];
		],
		"toString" -> Function[{},
			StringJoin[
				FileBaseName[Self["file"]],
				"  ",
				Self["start"]["toString"],
				"->",
				Self["end"]["toString"]
			]
		]
	|>,
	{
		"file",
		"start",
		"end"
	},
	Predicate -> SourceSpanQ
]
]]

CreateSourceSpan[file_:Undefined, start_:Undefined, end_:Undefined] :=
	Module[{cls = CreateObject[SourceSpan]},
		If[file =!= Undefined,
			cls["setFile", file]
		];
		If[start =!= Undefined,
			cls["setStart", start]
		];
		If[end =!= Undefined,
			cls["setEnd", end]
		];
		cls
	] 

End[]

EndPackage[]
