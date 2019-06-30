
BeginPackage["CompileAST`Create`Read`"]

ReadMExpr;

Begin["`Private`"] 

Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`SourceLocation`"]
Needs["CompileAST`Class`SourceSpan`"]
Needs["CompileAST`Class`Normal`"]


getFileInformation[file_] := 
	Module[{tags, info},
		tags = RuntimeTools`GetTags[];
		info = Map[{#, RuntimeTools`GetTagInformation[#]} &, tags];
		info = Select[info, (("FileName" /. #[[2]]) === file) &];
	  Map[
	  	Join[
	  		<|
	  			"Tag" -> #[[1]],
	  			"Expr" -> RuntimeTools`TagToExpr[#[[1]]]
	  		|>,
	  		Association[#[[2]]]
	  	] &,
	  	info
	  ]
	]

getTaggedFiles[] := Module[{tags},
	tags = RuntimeTools`GetTags[];
	tags = Map[RuntimeTools`GetTagInformation, tags];
	tags = Part[tags, All, 1, 2];
	Union[tags]
]
startTagRecord[] := (
	RuntimeTools`DebugOn[True];
	RuntimeTools`ToolsInitialize[];
)
stopTagRecord[] := (
	RuntimeTools`DebugOff[];
	RuntimeTools`ToolsUninitialize[];
)

ReadMExpr::syntx = "The input `1` could not be parsed as Mathematica syntax."
ReadMExpr::nofile = "The input file `1` could not be located."
ReadMExpr[s_String] :=
	If[SyntaxQ[s],
		Module[{expr, tags, files},
			startTagRecord[];
			expr = ImportString[s, {"Package", "HeldExpressions"}];
			files = getTaggedFiles[];
			tags = getFileInformation[Last[files]];
			stopTagRecord[];
			With[{e = First[expr]},
				expr = CreateMExpr[e]["part", 1]; (**< we remove the hold // there has to be a better way *)
			];
			propagateTags[expr, tags];
			expr
		],
		Message[ReadMExpr::syntx, s];
		$Failed
	]
ReadMExpr[{fileName_String}] :=
	If[FileExistsQ[fileName],
		Module[{expr, tags, files},
			startTagRecord[];
			expr = Import[fileName, {"Package", "HeldExpressions"}];
			files = getTaggedFiles[];
			tags = getFileInformation[fileName];
			stopTagRecord[];
			With[{e = First[expr]},
				expr = CreateMExpr[e]["part", 1]; (**< we remove the hold // there has to be a better way *)
			];
			propagateTags[expr, tags];
			expr
		],
		Message[ReadMExpr::nofile, fileName];
		$Failed
	]
	
	
propagateEnd[mexpr_] :=
	Module[{lastArg},
		Do[
			propagateEnd[arg],
			{elem, mexpr["arguments"]}
		];
		If[MExprNormalQ[mexpr],
			lastArg = Last[mexpr["arguments"]];
			mexpr["span"]["setEnd",
				CreateSourceLocation[lastArg["span"]["end"]["line"]]
			],
			mexpr["span"]["setEnd",
				CreateSourceLocation[mexpr["span"]["start"]["line"]]
			]
		]
	]
propagateTag[mexpr_, tag_] := (
	mexpr["setSpan", 
		CreateSourceSpan[
			tag["FileName"],
			CreateSourceLocation[tag["LineCount"]]
		]
	]
)


propagateNormalTags[mexpr_, tags0_] :=
	Module[{tags = tags0, hd, args},
		If[MExprNormalQ[mexpr],
			propagateTag[mexpr, First[tags]];
			tags = Rest[tags];
			args = mexpr["arguments"];
			If[mexpr["hasHead", CompoundExpression],
				args = Select[args, !#["sameQ", Null]&]
			];
			Do[
				tags = propagateNormalTags[arg, tags],
				{arg, Reverse[args]}
			];
			
			hd = mexpr["_head"];
			If[hd["normalQ"],
				tags = propagateNormalTags[hd, tags]
			];
			If[mexpr["hasHead", CompoundExpression],
				Do[
					nullExpr["setSpan",
						CreateSourceSpan[
							Last[args]["span"]["file"],
							CreateSourceLocation[Last[args]["span"]["start"]["line"]]
						]
					],
					{nullExpr, Select[mexpr["arguments"], #["sameQ", Null]&]}
				]
			]
		];
		tags
	]
propagateSymbolTags[mexpr_, tags0_] :=
	Module[{tags = tags0, args},
		If[MExprNormalQ[mexpr],
			args = mexpr["arguments"];
			If[mexpr["hasHead", CompoundExpression],
				args = Select[args, !#["sameQ", Null]&]
			];
			Do[
				tags = propagateSymbolTags[arg, tags],
				{arg, Reverse[args]}
			];
			If[NoneTrue[{List, Equal, CompoundExpression, Set, Plus, Greater}, mexpr["head"]["sameQ",#]&], 
				tags = propagateSymbolTags[mexpr["_head"], tags]
			],
			propagateTag[mexpr, First[tags]];
			tags = Rest[tags]
		];
		tags
	]
propagateTags[mexpr_, tags_] := (
	propagateSymbolTags[mexpr,
		propagateNormalTags[mexpr, tags]
	];
	propagateEnd[mexpr]
)
	
			


End[]

EndPackage[]
