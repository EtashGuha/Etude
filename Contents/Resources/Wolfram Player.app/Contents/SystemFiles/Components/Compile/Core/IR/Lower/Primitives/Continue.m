BeginPackage["Compile`Core`IR`Lower`Primitives`Continue`"]

Begin["`Private`"]


Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`BasicBlock`"]



lower[state_, mexpr_, opts_] :=
	Module[{builder, loopHeader},
		If[mexpr["length"] =!= 0,
			ThrowException[{"Invalid number for the Continue statement in " <> mexpr["toString"]}]
		];
		builder = state["builder"];
		loopHeader = builder["getProperty", "currentLoopHeader"];
		If[MissingQ[loopHeader] || !BasicBlockQ[loopHeader],
			ThrowException[{"Expecting the loop header to be set and to be a basic block while lowering the Continue statement. " <> mexpr["toString"]}]
		];
		builder["createBranchInstruction",
			loopHeader,
        	mexpr
	    ];
	    state["lower", CreateMExprSymbol[Null], opts]
	];


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Continue], lower]
]]

End[]

EndPackage[]
