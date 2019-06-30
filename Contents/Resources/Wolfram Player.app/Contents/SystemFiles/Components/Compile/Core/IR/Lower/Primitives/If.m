BeginPackage["Compile`Core`IR`Lower`Primitives`If`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]



(**
 * Creates 3 basic blocks where we have
 *
 *  [branch cond, then, else]
 *     /              \
 * [thenBB]          [elseBB]
 *     \               /
 *      \            /
 *       [  endBB   ]
 *
 * if the else branch does not exist, then we generate the
 * mexpr If[cond, then, Null] and call the lowerIfThenElse.
 * the boolean conditional expression is placed in the 
 * incoming basic block
 * Desugar a missing else to a If[cond, then, Null]
 *
 * The construction also does dead branch optimization
 *) 

(*
  Prepare branches for the If.
  The length must be 2 or 3.
  If length is 3,  then the if and else branches are just parts 2 and 3.
  If length is 2,  then the else branch is Compile`Void, and the ifBranch 
  is turned into CompoundExpression[ args, Compile`Void].
  
*)
prepareBranches[ state_, mexpr_, opts_] :=
	Module[ {cond, ifBranch, elseBranch},
		If[ mexpr["length"] =!= 2 && mexpr["length"] =!= 3,
			ThrowException[LanguageException[{"If is expected to have 2 or 3 arguments ", mexpr["toString"]}]]
		];
		cond = mexpr["part", 1];
		ifBranch = mexpr["part", 2];
		If[ mexpr["length"] === 3,
			elseBranch = mexpr["part", 3]
			,
			ifBranch = CreateMExpr[ CompoundExpression, {ifBranch, CreateMExprSymbol[Compile`Void]}];
			elseBranch = CreateMExprSymbol[Compile`Void];
		];
		{cond, ifBranch, elseBranch}
	]


ClearAll[lower]
lower[state_, mexpr_, opts_] :=
     Module[{builder, cond, ifBranch, elseBranch, thenBB, startBB, 
    	     elseBB, endBB, condvar, resThen, resElse, res, lastBB, retThen = False, retElse = False, fmBuilder},
		{cond, ifBranch, elseBranch} = prepareBranches[ state, mexpr, opts];
		builder = state["builder"];
		fmBuilder = builder["currentFunctionModuleBuilder"];
		(*
		 TODO deal with return in the condition,  really would just return at this point.
		*)
  		condvar = state["lower", cond, opts];
		If[
		    (* We do not want to introduce any new Phi instructions, so
			 * we read the variable with the IR in read only mode
			 *)
		    ConstantValueQ[condvar],
				res = If[condvar["value"] === True,
				    state["lower", ifBranch, opts],
				    state["lower", elseBranch, opts]
				]
			,  (* Else *)
		  		startBB = builder["currentBasicBlock"];

				res = state["createFreshVariable", mexpr];

				thenBB = builder["addBasicBlock", "thenif", mexpr];
				elseBB = builder["addBasicBlock", "elseif", mexpr];
				(*
				  Only create the endBB if it is needed,  ie if one of 
				  the branches does not have a return.
				*)
				endBB = Null;
				
				builder["setCurrentBasicBlock", startBB];
				builder["createBranchInstruction",
					{
		    			thenBB,
		    			elseBB
		    		},
		    		condvar,
		        	mexpr
			    ];

				startBB["addChild", thenBB];
		  	 	builder["sealBasicBlock", thenBB];
		        builder["setCurrentBasicBlock", thenBB];
		        resThen = state["lower", ifBranch, opts];
		        (*
		          If there was a return we add the jump to the last BB
		          otherwise we add a jump to the endBB after creating it.
		        *)
		        If[ fmBuilder["returnMode"],
		        	fmBuilder["setReturnMode", False];
		        	lastBB = fmBuilder["lastBasicBlock"];	
		        	builder["createBranchInstruction", lastBB, mexpr];
		        	builder["currentBasicBlock"]["addChild", lastBB];
		        	retThen = True;
		        	,
		        	thenBB = builder["currentBasicBlock"];
		        	endBB = builder["addBasicBlock", "endif", mexpr];
		        	builder["setCurrentBasicBlock", thenBB];
					builder["createBranchInstruction", endBB, mexpr];
					builder["currentBasicBlock"]["addChild", endBB]
				];
			
				startBB["addChild", elseBB];
		  	 	builder["sealBasicBlock", elseBB];
				builder["setCurrentBasicBlock", elseBB];
				resElse = state["lower", elseBranch, opts];
				(*
		          If there was a return 
		              if there was also a return in the other branch then don't add a jump
		              because we are just going to propagate the return onwards,  the jump 
		              will be added later when the returnMode is finally caught
		              
		              if there was not a return in the other branch we add a jump to the lastBB
		              
		          if there is no return we add a jump to the endBB after creating it if necessary.
				*)
		        If[ fmBuilder["returnMode"],
		        	fmBuilder["setReturnMode", False];
		        	lastBB = fmBuilder["lastBasicBlock"];
		        	If[!retThen,
		        		builder["createBranchInstruction", lastBB, mexpr];
		        		builder["currentBasicBlock"]["addChild", lastBB]];
		        	retElse = True;
		        	,
		        	elseBB = builder["currentBasicBlock"];
		        	If[ endBB === Null,
		        		endBB = builder["addBasicBlock", "endif", mexpr];
		        		builder["setCurrentBasicBlock", elseBB]];
					builder["createBranchInstruction", endBB, mexpr];
					builder["currentBasicBlock"]["addChild", endBB]];
		
				Which[
					(*
					  Both branches returned,  so the end is unreachable, 
					  propagate the return.
					*)
					retThen && retElse,
					   fmBuilder["setReturnMode", True];
					   res = Null;
					,
					(*
					  If one branch returned make the endBB current.
					  The result will be the result of the branch that 
					  did not return.
					*)
					retThen,
						builder["setCurrentBasicBlock", endBB];
						builder["sealBasicBlock", endBB];
						res = resElse;
					,
					retElse,
						builder["setCurrentBasicBlock", endBB];
						builder["sealBasicBlock", endBB];
						res = resThen;
					,
					(*
					 There was no return.  Make the endBB current.
					 Add a phi instruction to get the result value.
					*)
					True,
		        		builder["setCurrentBasicBlock", endBB];
	       				builder["createPhiInstruction", res, {{thenBB, resThen}, {elseBB, resElse}}, mexpr];
		        		builder["sealBasicBlock", endBB]
		        ];
		];
		res
    ]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[If], lower]
]]


End[]

EndPackage[]
