BeginPackage["Compile`AST`Macro`Builtin`Do`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] := Module[{
	env = st["macroEnvironment"]
},
		RegisterMacro[env, Do,
			Do[body_, end:Except[_List]] ->
				(
				Compile`AssertType[end, "MachineInteger"];
				Do[body, {ii, 1, end, 1}]
				),
			Do[body_, {end_}] ->
				Do[body, {ii, 1, end, 1}],
			Do[body_, {ii_, end_}] ->
				Do[body, {ii, 1, end}],
			Do[body_, {ii_, start_, end_, inc_:1}] ->
				Native`UncheckedBlock[
				Module[{
					iterator = Native`CreateIterator[ Native`CheckedBlockRestore[start], Native`CheckedBlockRestore[end], Native`CheckedBlockRestore[inc]],
					len,
					val,
					ii,
					icnt = 0
				},
					len = Native`IteratorLength[iterator];
					While[icnt < len,
						val = Native`IteratorValue[iterator, icnt];
						ii = val;
						Native`CheckedBlockRestore[body];
						icnt = icnt + 1];
				]],
			Do[body_, a_, b__] ->
				Do[Do[body, b], a]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
