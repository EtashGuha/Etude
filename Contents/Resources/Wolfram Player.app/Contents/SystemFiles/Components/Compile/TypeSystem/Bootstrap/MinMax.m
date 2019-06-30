
BeginPackage["Compile`TypeSystem`Bootstrap`MinMax`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
			inline = MetaData[<|"Inline" -> "Hint"|>]},


		env["declareFunction", Min, 
			Typed[
				TypeForAll[ {"elem"}, {Element["elem", "Ordered"]}, {"elem"} -> "elem"]
			]@Function[{elem}, 
				elem
			]
		];

		env["declareFunction", Max, 
			Typed[
				TypeForAll[ {"elem"}, {Element["elem", "Ordered"]}, {"elem"} -> "elem"]
			]@Function[{elem}, 
				elem
			]
		];

		env["declareFunction", Min, 
			inline@Typed[
				TypeForAll[ {"elem"}, {Element["elem", "Ordered"]}, {"elem", "elem"} -> "elem"]
			]@Function[{e1, e2}, 
				If[e1 < e2, e1, e2]
			]
		];

		env["declareFunction", Max, 
			inline@Typed[
				TypeForAll[ {"elem"}, {Element["elem", "Ordered"]}, {"elem", "elem"} -> "elem"]
			]@Function[{e1, e2}, 
				If[e1 > e2, e1, e2]
			]
		];

		env["declareFunction", Min, 
			Typed[
				TypeForAll[ {"elem", "rank", "container"}, {Element["elem", "Ordered"], Element["container", "ArrayContainer"]}, 
									{"container"["elem", "rank"]} -> "elem"]
			]@Function[{array}, 
				Module[{stm},
					stm = Native`CreateElementReadStream[ array];
					Fold[ Min, stm]
				]
			]
		];

		env["declareFunction", Max, 
			Typed[
				TypeForAll[ {"elem", "rank", "container"}, {Element["elem", "Ordered"], Element["container", "ArrayContainer"]}, 
								{"container"["elem", "rank"]} -> "elem"]
			]@Function[{array}, 
				Module[{stm},
					stm = Native`CreateElementReadStream[ array];
					Fold[ Max, stm]
				]
			]
		];


	]

] (* StaticAnalysisIgnore *)


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		
		Scan[
			With[ {fun = #},
				RegisterMacro[env, fun,
					fun[a: Except[_fun], inds__] -> 
						 fun[fun[a], fun[inds]]
				]]&, {Min, Max}];


	];



RegisterCallback["SetupMacros", setupMacros]
RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
