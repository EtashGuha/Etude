
BeginPackage["Compile`TypeSystem`Bootstrap`Structural`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Create`Construct`"]



"StaticAnalysisIgnore"[

setupTypes[st_] :=
	With[{
		env = st["typeEnvironment"],
		inline = MetaData[<|"Inline" -> "Hint"|>]
	},
		    
		env["declareFunction", Native`GetPartUnary,
			inline@
			Typed[
				TypeForAll[ {"a"}, {"CArray"["a"], "MachineInteger"} -> "a"]
			]@Function[{arg1, arg2},
				Native`GetElement[arg1, arg2]
			]
		];
	
		
		env["declareFunction", Native`Unchecked[Native`GetPartUnary],
			inline@
			Typed[
				TypeForAll[ {"a"}, {"CArray"["a"], "MachineInteger"} -> "a"]
			]@Function[{arg1, arg2},
				Native`GetElement[arg1, arg2]
			]
		];

        env["declareFunction", Native`SetPartUnary,
            inline@
            Typed[
                TypeForAll[ {"a"}, {"CArray"["a"], "MachineInteger", "a"} -> "Void"]
            ]@Function[{arg1, arg2, arg3},
               Native`SetElement[arg1, arg2, arg3];
            ]
        ];

		env["declareFunction", Native`Unchecked[Native`SetPartUnary],
			inline@
			Typed[
				TypeForAll[ {"a"}, {"CArray"["a"], "MachineInteger", "a"} -> "Void"]
			]@Function[{arg1, arg2, arg3},
				Native`SetElement[arg1, arg2, arg3];
			]
		];

	(*
	 Span
	*)
	
		env["declareType", TypeConstructor["SpanBase"]];
		env["declareType", TypeAlias["Span", "Handle"["SpanBase"]]];

		env["declareFunction", Span,
			Typed[
				{"MachineInteger", "MachineInteger", "MachineInteger"} -> "Span"
			]@Function[{arg1, arg2, arg3},
				Native`BitCast[{arg1, arg2, arg3}, "Span"]
			]
		];


	]

] (* StaticAnalysisIgnore *)



getLength[ args___] :=
	CreateMExprLiteral[ Length[{args}]]


(*
TODO
  Process Set of a List here.
  This should really look at meta-data on arg to verify this is not just a symbol but that it is writable.
  Should have custom error handling here for any version of Set of a symbol or list.
*)

invalidLHS[arg_] :=
	!validLHS[arg]

validLHS[ arg_] :=
	arg["symbolQ"] && checkWritable[arg]


checkWritable[arg_] :=
	Module[ {binder = arg["getProperty", "scopeBinder", Null]},
		If[
			MExprQ[binder],
			!(binder["hasHead", Function] || binder["hasHead", With])
			,
			True]
	]


setupMacros[st_] := With[{env = st["macroEnvironment"]},
		
	RegisterMacro[env, Set,

		Set[Part[a_, Native`Field[b_]], c_] -> 
			Native`SetElement[a,Native`Field[b],c]
			,
		(*Set[Native`Unchecked[Part][a_, Native`Field[b_]], c_] ->
			Native`Unchecked[Native`SetElement][a,Native`Field[b],c]
			,*)
		Set[Part[a_?validLHS, b_], c_] -> 
			Module[ {elem = c},
				a = Native`MutabilityClone[a];
			 	Native`SetPartUnary[a,b,elem];
			 	elem
			]
			,
		(*Set[Native`Unchecked[Part][a_?validLHS, b_], c_] -> 
			Module[ {elem = c},
			 	Native`SetPartUnary[a,b,c];
			 	elem
			]
			,*)
		Set[Part[a_?validLHS, b_, c_], d_] -> 
			Module[ {elem = d},
				a = Native`MutabilityClone[a];
				Native`SetPartBinary[a,b,c,elem];
				elem
			]
			,
		(*Set[Native`Unchecked[Part][a_?validLHS, b_, c_], d_] -> 
			Module[ {elem = d},
				Native`SetPartBinary[a,b,c,d];
				elem
			]
			,*)
		Set[Part[a_?validLHS, b_, c_, inds__], d_] ->
			Module[ {elem = d},
				a = Native`MutabilityClone[a];
				Native`SetPartNary[a,b,c,inds,elem];
				elem
			]
			,
		(*Set[Native`Unchecked[Part][a_?validLHS, b_, c_, inds__], d_] ->
			Module[ {elem = d},
				Native`SetPartNary[a,b,c,inds,d];
				elem
			]
			,*)
		Set[Part[a_?invalidLHS, inds__], d_] -> 
			Compile`Error[{"Invalid part assignment to ", a}]
			(*,
		Set[Native`Unchecked[Part][a_?invalidLHS, inds__], d_] -> 
			Compile`Error[Hold[ Set, Native`Unchecked[Part], a, inds, d], "Invalid LHS"]*)
			
		,
		Set[syms_List, data_] ->
			Compile`Internal`MacroEvaluate[ makeListAssign[syms, data]]		

	];

	RegisterMacro[env, Native`SetPartNary,
		Native`SetPartNary[a_, inds__, d_] -> 
			 Native`PartViewFinalizeSet[
				Native`PartViewAdd[ 
					Native`CreatePartView[a, Compile`Internal`MacroEvaluate[ getLength[inds]]],
					0,
					inds
				],
				d
			 ]
    ];
			

	
	RegisterMacro[env, Part,
		
		Part[a_, Native`Field[b_]] -> 
				Native`GetElement[a,Native`Field[b]]
			,	
		Part[a_, b_] -> 
			Native`GetPartUnary[a,b]
			,
		Part[a_, b_, c_] -> 
			Native`GetPartBinary[a,b,c]
			,
		Part[a_, b_, c_, inds__] -> 
			Native`GetPartNary[a,b,c,inds]
	];
	
	RegisterMacro[env, Span,
		
		Span[a_] -> 
				Span[1,a,1]
			,	
		Span[a_, b_] -> 
				Span[a,b,1]
	];
	
	RegisterMacro[env, Native`GetPartNary,
		Native`GetPartNary[a_, inds__] -> 
			 Native`PartViewFinalizeGet[
				Native`PartViewAdd[ 
					Native`CreatePartView[a, Compile`Internal`MacroEvaluate[ getLength[inds]]],
					0,
					inds
				]
			]
	];
	
	RegisterMacro[env, Native`PartViewAdd,
		Native`PartViewAdd[a_, indNum_, ind1_, indr__] -> 
			Compile`Internal`MacroRecursionExpand[Native`PartViewAdd[ Native`PartViewAdd[ a, indNum, ind1], indNum+1, indr]]
	];


	RegisterMacro[env, Join,
		
		Join[a_] -> 
			a
		,
		Join[a_, b_] -> 
			Native`JoinBinary[a,b]
		,
		Join[a_, b_, ext__] -> 
			 Native`JoinNary[a,b,ext]
	];

	RegisterMacro[env, Native`JoinNary,
		Native`JoinNary[args__] -> 
			 Native`JoinViewFinalize[
				Native`ListViewAdd[ 
					Native`CreateListView[Compile`Internal`MacroEvaluate[ getLength[args]]],
					0,
					args
				]
			]
	];

	RegisterMacro[env, Native`ListViewAdd,
		Native`ListViewAdd[a_, indNum_, ind1_, indr__] ->
			Compile`Internal`MacroRecursionExpand[Native`ListViewAdd[ Native`ListViewAdd[ a, indNum, ind1], indNum+1, indr]]
	];

	
]


(*
  We have Set[ symsList, data] where syms is a list of symbols and data is a packed array.
  Compile this into
        Compile`AssertTypeApplication[data, "PackedArray"];
     	If[lenSymList =!= Length[data],
     		Native`ThrowWolframException[Typed[Native`ErrorCode["DimensionError"], "Integer32"]]];
     	symsList1 = Native`GetPartUnary[data,1];
     	symsList2 = Native`GetPartUnary[data,2];
     	symsList3 = Native`GetPartUnary[data,3];
     	...

*)
makeListAssign[symsList_, data_] :=
	Module[{
			syms = symsList["arguments"], 
			assertArg, tmp1, tmp2, lenCheck, assArgs
		},
		assertArg = buildMExpr[ Compile`AssertTypeApplication, {data, buildLiteral["PackedArray"]}];
		tmp1 = buildMExpr[ Length, {data}];
		tmp1 = buildMExpr[ UnsameQ, {buildLiteral[Length[syms]], tmp1}];
		tmp2 = buildMExpr[Native`ErrorCode, {buildLiteral["DimensionError"]}];
		tmp2 = buildMExpr[Typed, {tmp2, buildLiteral["Integer32"]}];
		tmp2 = buildMExpr[Native`ThrowWolframException, {tmp2}];
		lenCheck = buildMExpr[ If, {tmp1, tmp2}];
		assArgs = MapIndexed[
					Function[{sym, indList},
						Module[{ind = First[indList], tmp},
							tmp = buildMExpr[ Native`GetArrayElement, {data, buildLiteral[ind-1]}];
							tmp = buildMExpr[ Native`UncheckedBlock, {tmp}];
							buildMExpr[ Set, {sym, tmp}]
						]], syms];
		buildMExpr[ CompoundExpression, Join[ {assertArg, lenCheck}, assArgs]]
	]

buildMExpr[ h_, args_] :=
	CreateMExpr[h, args]

buildLiteral[ x_] :=
	CreateMExprLiteral[x]



RegisterCallback["SetupMacros", setupMacros]
RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
