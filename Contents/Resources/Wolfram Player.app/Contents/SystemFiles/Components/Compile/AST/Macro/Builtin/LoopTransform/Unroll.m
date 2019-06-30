BeginPackage["Compile`AST`Macro`Builtin`LoopTransform`Unroll`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Native`LoopTransform,
			Native`LoopTransform["Unroll", 1][
				For[iter_ = start_, cond_, Null,
					body_
				]
			] ->
				For[iter = start, cond, Null,
					body
				],
			Native`LoopTransform["Unroll", 1][
				For[iter_ = start_, cond_, iter_ = op_[iter_, incr_],
					body_
				]
			] ->
				For[iter = start, cond, iter = op[iter, incr],
					body
				],
			Native`LoopTransform["Unroll", 2][
				For[iter_ = start_, cond_, Null,
					body_
				]
			] ->
				For[iter = start, cond, Null ,
					body;
					body;
				],
			Native`LoopTransform["Unroll", 2][
				For[iter_ = start_, cond_, iter_ = op_[iter_, incr_],
					body_
				]
			] ->
				For[iter = start, cond, iter = op[iter, incr],
					body;
					iter = op[iter, incr];
					body
				],
			Native`LoopTransform["Unroll", 3][
				For[iter_ = start_, cond_, Null,
					body_
				]
			] ->
				For[iter = start, cond, Null,
					body;
					body;
					body;
				],
			Native`LoopTransform["Unroll", 3][
				For[iter_ = start_, cond_, iter_ = op_[iter_, incr_],
					body_
				]
			] ->
				For[iter = start, cond, iter = op[iter, incr],
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body
				],
			Native`LoopTransform["Unroll", 4][
				For[iter_ = start_, cond_, Null,
					body_
				]
			] ->
				For[iter = start, cond, Null,
					body;
					body;
					body;
					body;
				],
			Native`LoopTransform["Unroll", 4][
				For[iter_ = start_, cond_, iter_ = op_[iter_, incr_],
					body_
				]
			] ->
				For[iter = start, cond, iter = op[iter, incr],
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body
				],
			Native`LoopTransform["Unroll", 8][
				For[iter_ = start_, cond_, Null,
					body_
				]
			] ->
				For[iter = start, cond, Null,
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body
				],
			Native`LoopTransform["Unroll", 8][
				For[iter_ = start_, cond_, iter_ = op_[iter_, incr_],
					body_
				]
			] ->
				For[iter = start, cond, iter = op[iter, incr],
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body
				],
			Native`LoopTransform["Unroll", 16][
				For[iter_ = start_, cond_, Null,
					body_
				]
			] ->
				For[iter = start, cond, Null,
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body;
					body
				],
			Native`LoopTransform["Unroll", 16][
				For[iter_ = start_, cond_, iter_ = op_[iter_, incr_],
					body_
				]
			] ->
				For[iter = start, cond, iter = op[iter, incr],
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body;
					iter = op[iter, incr];
					body
				]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
