BeginPackage["Compile`Core`IR`Lower`Primitives`While`"]

Begin["`Private`"]

Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]



(**
 * Creates 3 basic blocks where we have
 *
 * [start]
 *    cond = check
 *    [branch cond, then, endBB]
 * [thenBB]
 *    body
 *    goto [start]
 *  [endBB]
 *
 * Supports Return in the body,  but not in the condition,  this could be added.
 *
 *)
lower[state_, mexpr_, opts_] :=
    Module[{
    	builder, whileCondition, whileEntry,
    	whileBody, whileEnd, condVar, fmBuilder, lastBB,
    	prevLoopHeader, prevLoopTerminator
    },
		builder = state["builder"];
		
		prevLoopHeader = builder["getProperty", "currentLoopHeader"];
		prevLoopTerminator = builder["getProperty", "currentLoopTerminator"];
		
		fmBuilder = builder["currentFunctionModuleBuilder"];
  		whileEntry = builder["currentBasicBlock"];
		whileCondition = builder["addBasicBlock", "whileCondition", mexpr];

        builder["setCurrentBasicBlock", whileEntry];
        builder["createBranchInstruction",
			{
	    		whileCondition
	    	},
	        mexpr
	    ];
    	whileEntry["addChild", whileCondition];

        builder["setCurrentBasicBlock", whileCondition];

		whileBody = builder["addBasicBlock", "whileBody", mexpr];
		whileEnd = builder["addBasicBlock", "whileExit", mexpr];

    	builder["setCurrentBasicBlock", whileCondition];
    	
    	builder["setProperty", "currentLoopHeader" -> whileCondition];
    	builder["setProperty", "currentLoopTerminator" -> whileEnd];

		condVar = state["lower", mexpr["part", 1], opts];
		(*
		  TODO add support for Return. If a return then return.
		*)
        builder["createBranchInstruction",
			{
    			whileBody,
    			whileEnd
    		},
    		condVar,
        	mexpr
	    ];
		builder["currentBasicBlock"]["addChild", whileBody];
		builder["currentBasicBlock"]["addChild", whileEnd];

		builder["setCurrentBasicBlock", whileBody];
  	 	builder["sealBasicBlock", whileBody];
		state["lower", mexpr["part", 2], opts];
		(*
		  If lowering the condition hit a Return then it is going to 
		  jump to the lastBB.  In this case we add a branch to the 
		  lastBB and make the lastBB a child.
		  Otherwise we add a branch to the condition and make this a 
		  child of the condition.
		*)
		If[ fmBuilder["returnMode"],
	        fmBuilder["setReturnMode", False];
	        lastBB = fmBuilder["lastBasicBlock"];	
	        builder["createBranchInstruction", lastBB, mexpr];
	        builder["currentBasicBlock"]["addChild", lastBB];
			, (* Else *)
			builder["createBranchInstruction", whileCondition, mexpr];
			builder["currentBasicBlock"]["addChild", whileCondition]
		];
  	 	
  	 	builder["sealBasicBlock", whileCondition];
        builder["setCurrentBasicBlock", whileEnd];
  	 	builder["sealBasicBlock", whileEnd];	
  	 	
  	 	If[MissingQ[prevLoopHeader],
  	 		builder["removeProperty", "currentLoopHeader"],
  	 		builder["setProperty", "currentLoopHeader" -> prevLoopHeader]
  	 	];
  	 	If[MissingQ[prevLoopTerminator],
  	 		builder["removeProperty", "currentLoopTerminator"],
  	 		builder["setProperty", "currentLoopTerminator" -> prevLoopHeader]
  	 	];
    	
		state["lower", CreateMExprSymbol[Null], opts]
    ]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[While], lower]
]]

End[]

EndPackage[]
