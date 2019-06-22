BeginPackage["Compile`Core`IR`Lower`Primitives`Break`"]

Begin["`Private`"]


Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`BasicBlock`"]



lower[state_, mexpr_, opts_] :=
	Module[{builder, loopTerminator},
		If[mexpr["length"] =!= 0,
			ThrowException[{"Invalid number for the Break statement in " <> mexpr["toString"]}]
		];
		builder = state["builder"];
		loopTerminator = builder["getProperty", "currentLoopTerminator"];
		If[MissingQ[loopTerminator] || !BasicBlockQ[loopTerminator],
			ThrowException[{"Expecting the loop terminator to be set and to be a basic block while lowering the Break statement. " <> mexpr["toString"]}]
		];
		builder["createBranchInstruction",
			loopTerminator,
        	mexpr
	    ];
	    state["lower", CreateMExprSymbol[Null], opts]
	];


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Break], lower]
]]

End[]

EndPackage[]
