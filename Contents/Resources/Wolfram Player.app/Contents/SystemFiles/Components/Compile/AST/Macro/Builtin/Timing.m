
BeginPackage["Compile`AST`Macro`Builtin`Timing`"]
Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]

setupMacros[st_] :=
    With[ {
        env = st["macroEnvironment"]
    },
        (* 

        RegisterMacro[env, AbsoluteTiming,
            AbsoluteTiming[e_] -> Module[{tic$$, toc$$, ret$$}, 
                tic$$ = Native`ReadCycleCounter[];
                ret$$ = e;
                toc$$ = Native`ReadCycleCounter[];
                {Native`UncheckedBlock[(toc$$-tic$$) / cpufreq], ret$$}
            ]
        ];
        *)
        RegisterMacro[env, Native`Internal`AbsoluteCycles,
            Native`Internal`AbsoluteCycles[e_] -> Module[{tic$$, toc$$}, 
                tic$$ = Native`ReadCycleCounter[];
                e;
                toc$$ = Native`ReadCycleCounter[];
                Native`UncheckedBlock[toc$$-tic$$]
            ]
        ];
    ]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]