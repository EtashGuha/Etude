BeginPackage["Compile`AST`Macro`Builtin`String`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		
		RegisterMacro[env, StringTake,
			StringTake[a_, {i_}] -> Native`SubString[a,i,i],
			StringTake[a_, {i_, j_}] -> Native`SubString[a,i,j],
			StringTake[a_, i_] -> Native`SubString[a,1,i]
		];
		
		RegisterMacro[env, StringDrop,
            StringDrop[a_, {i_}] -> StringJoin[StringTake[a, i-1], StringTake[a, {i+1,StringLength[a]}]],
            StringDrop[a_, {i_, j_}] -> StringJoin[StringTake[a, i-1], StringTake[a, {j+1,StringLength[a]}]],
            StringDrop[a_, i_] -> StringTake[a,{i+1,StringLength[a]}]
        ];
        
		RegisterMacro[env, StringJoin,
			StringJoin[] -> "",
			StringJoin[a_] -> a,
			StringJoin[a_, ""] -> a,
			StringJoin["", a_] -> a,
			StringJoin[a_, b_, c__] -> StringJoin[a, StringJoin[b, c]]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
