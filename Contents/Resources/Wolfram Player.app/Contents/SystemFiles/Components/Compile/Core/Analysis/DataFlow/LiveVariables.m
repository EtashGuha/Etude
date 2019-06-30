(**
  * One use is that if one traces a sequence of statements and records the types
  * then one only needs to include guards for variables that are live at the start
  * of the program. For example, global variables would are alive before the start
  * of the program.
  *
  * The code references the algorithms defined in
  * Boissinot, Benoit, Florian Brandner, Alain Darte, Benoit Dupont de Dinechin, 
  * and Fabrice Rastello. \"A non-iterative data-flow algorithm for computing 
  * liveness sets in strict SSA programs.\" In Asian Symposium on Programming 
  * Languages and Systems, pp. 137-154. Springer, Berlin, Heidelberg, 2011.
  *
  *)

BeginPackage["Compile`Core`Analysis`DataFlow`LiveVariables`"]

LiveVariablesPass;

Begin["`Private`"] 

Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`Analysis`Dominator`DominatorPass`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Analysis`Loop`LoopNestingForest`"]
Needs["Compile`Core`IR`LoopInformation`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


ClearAll[run]

idOf[e_] := e["id"]
idOf[e_?ListQ] := idOf /@ e

(*
 Set up the live[in] and live[out] sets for each BasicBlock. 
 All of these are AssociationReferences[ bbId -> AssociationReference[ varId -> var]].
*)	
initialize[ fm_, opts_] :=
	fm["scanBasicBlocks",
		Function[{bb},
			bb["clearVisited"];
			bb["setProperty", "live[in]" -> None];
			bb["setProperty", "live[out]" -> None];
		]
	];
finalize[ fm_, opts_] := 
    fm["topologicalOrderScan",
		Function[{bb},
            bb["clearVisited"];
	        bb["setProperty", "live[out]" -> SortBy[bb["getProperty", "live[out]"], idOf]];
			computeLiveVariablesForInstructions[bb];
            bb["setProperty", "live[in]" -> SortBy[bb["firstInstruction"]["getProperty", "live[in]"], idOf]];
		]
	];
	
computeLiveVariablesForInstructions0[bb_] :=
    Module[{
        live =  bb["getProperty", "live[in]"],
        liveOutIds =  idOf[bb["getProperty", "live[out]"]]
    },
        bb["scanInstructions",
           Function[{inst},
               inst["setProperty", "live[in]" -> SortBy[live, idOf]];
               If[inst["definesVariableQ"] && MemberQ[liveOutIds, idOf[inst["definedVariable"]]],
                    AppendTo[live, inst["definedVariable"]]
               ];
               live = DeleteDuplicates[live, sameQ];
               inst["setProperty", "live[out]" -> SortBy[live, idOf]];
           ]
        ]
    ];
    
computeLiveVariablesForInstructions[bb_] :=
    Module[{
        live =  bb["getProperty", "live[out]"]
    },
        bb["reverseScanInstructions",
           Function[{inst},
               inst["setProperty", "live[out]" -> live];
               If[inst["definesVariableQ"],
                    live = Select[live, idOf[#] =!= idOf[inst["definedVariable"]]&]
               ];
               With[{
                    operands = inst["usedVariables"]
               },
                    Do[
                       If[includeQ[operand],
                           AppendTo[live, operand]
                       ],
                       {operand, operands}
                    ]
               ];
               live = DeleteDuplicates[live, sameQ];
               inst["setProperty", "live[in]" -> live];
           ]
        ]
    ];
	
sameQ[a_, b_] := idOf[a] === idOf[b]
includeQ[e_] := VariableQ[e]	

(* 
 * Algorithm 4: First pass of the loop-based liveness analysis
 * 
 *)
dagDFS[fm_?FunctionModuleQ] :=
    Module[{
    	   backedges = getBackedges[fm]
    	},
    	   fm["topologicalOrderScan", dagDFS[#, backedges]&]
    ];
(* Calculate partial liveness sets for CFG nodes. *)
dagDFS[bb_?BasicBlockQ, backedges_] :=
    Module[{
       live = {},
       liveIn = {},
       liveOut = {},
       phiDefinedIds,
       children = bb["getChildren"]
    },
       If[bb["visitedQ"],
            Return[]
       ];
       AssertThat["the basic block must not have been visited. i.e. the live[in] of the bb should be None", bb["getProperty", "live[in]"]
           ]["named", "bb"
           ]["isEqualTo", None
       ];
       (* Calculate live-out set (lines 4-7 of Algorithm 2). *)
       Do[
           (* skip this edge if it is a loop edge *)
           If[idOf[child] =!= idOf[bb] && !KeyExistsQ[backedges, Key[idOf[bb] -> idOf[child]]],
               If[!child["visitedQ"],
                  dagDFS[child, backedges]
               ];
           ],
           {child, children}
       ];
       Do[
           live = Join[live, phiUses[bb, child]];
           (* skip this edge if it is a loop edge *)
           If[!KeyExistsQ[backedges, Key[idOf[bb] -> idOf[child]]],
               AssertThat["the child must has been visited. i.e. the live[in] of the child should not be None", child["getProperty", "live[in]"]
                   ]["named", child["fullName"] <> " as a child of " <> bb["fullName"]
                   ]["isNotEqualTo", None
               ];
               If[idOf[child] === idOf[bb],
                   Continue[]
               ]; 
               phiDefinedIds = idOf[phiDefs[child]];
               Table[
                   (* Live = Live U (LiveIn(S) \ PhiDefs(S)) *)
                   If[FreeQ[phiDefinedIds, idOf[childLiveIn]] && includeQ[childLiveIn],
                       AppendTo[live, childLiveIn] 
                   ],
                   {childLiveIn, child["getProperty", "live[in]"]}
               ];
           ],
           {child, children}
       ];
       live = DeleteDuplicates[live, sameQ];
       liveOut = live;
       (* Calculate live-in set (lines 8-11 of Algorithm 2). *)
       bb["reverseScanInstructions",
           Function[{inst},
                If[inst["definesVariableQ"],
                  live = Select[live, idOf[#] =!= idOf[inst["definedVariable"]]&]
                ];
           	    If[!PhiInstructionQ[inst], (* a phi is not a program point *)
                   With[{
                       operands = inst["usedVariables"]
                   },
                       Do[
                           If[includeQ[operand],
                               AppendTo[live, operand]
                           ],
                           {operand, operands}
                       ]
                  ];
           	    ];
           ]
       ];
       (*live = With[{
       	 ds = idOf[phiDefs[bb]]
       },
       	 Select[live, FreeQ[ds, idOf[#]]&]
       ];*)
       live = Join[live, phiDefs[bb]];
       liveIn = DeleteDuplicates[live, sameQ];
           
       bb["setProperty", "live[in]" -> liveIn];
       bb["setProperty", "live[out]" -> liveOut];
       
       bb["setVisited", True];
    ]
propagateValues[loopInfo_?LoopInformationQ] :=
    With[{
    	   header = loopInfo["header"]
    },
    With[{
    	   liveLoop = With[{
    	   	 liveIn = header["getProperty", "live[in]"],
    	   	 defs = idOf[phiDefs[header]]
    	   },
    	     Select[liveIn, FreeQ[defs, idOf[#]]&]
    	   ]
    },
        loopInfo["scan",
        	   Function[{m},
        	   	   With[{
        	   	       bb = If[BasicBlockQ[m],
        	   	           m,
        	   	           m["header"]
        	   	       ]
        	   	   },
        	   	   With[{
        	   	       liveLoopIn = Select[liveLoop,
        	   	       	   idOf[#["def"]["basicBlock"]] =!= idOf[bb]&
        	   	       ]
        	   	   },  
        	   	       (*Print[{bb, "  liveLoop = ", liveLoop, "  liveLoopIn  = ", liveLoopIn, " phi defs = ", phiDefs[header]}]; *)
        	   	       bb["setProperty", "live[in]" ->
        	   	       	   DeleteDuplicates[Join[bb["getProperty", "live[in]"], liveLoopIn], sameQ]
        	   	       ];
                   bb["setProperty", "live[out]" ->
                       DeleteDuplicates[Join[bb["getProperty", "live[out]"], liveLoop], sameQ]
                   ];
        	           loopTreeDFS[m]
        	   	   ]]
        	   ]
        ];
        header["setProperty", "live[out]" ->
           DeleteDuplicates[Join[header["getProperty", "live[out]"], liveLoop], sameQ]
        ];
    ]];
propagateValues[bb_?BasicBlockQ] := "do nothing";
propagateValues[args___] :=
    ThrowException[{"Unrecognized call to propagateValues in live variables pass", {args}}];
    
(* Propagate live variables within loop bodies. *)
loopTreeDFS[bb_?BasicBlockQ] := Nothing
loopTreeDFS[loopInfo_?LoopInformationQ] :=
    loopInfo["scan",
    	   propagateValues
    ];
    
    
run[fm_, opts_] :=
    With[{
	   loopInfo = fm["getProperty", "loopinformation"]
	},
   	   (* Compute partial liveness sets using a postorder traversal. *)
	   dagDFS[fm];
	   (* Propagate live variables within loop bodies. *)
	   loopTreeDFS[loopInfo];
	   
	   fm
    ];
    
(* Find loop edges and convert to an association for existence checking.
 * The loop information is computed by the LoopNestingForestPass.
 * The returned association is of the form <| Key[source -> dest] -> True, ... |>
 *)
getBackedges[fm_?FunctionModuleQ] :=
    With[{
    	   loopInfo = fm["getProperty", "loopinformation"]
    },
       getBackedges[loopInfo]
    ]
getBackedges[loopInfo_?LoopInformationQ] :=
    Module[{
    	   assoc = Association[Key[#] -> True& /@ loopInfo["edges"]]
    },
       loopInfo["scan",
       	   Function[{e},
       	   	    If[LoopInformationQ[e],
       	   	    	   assoc = Join[assoc, getBackedges[e]]
       	   	    ]
       	   ]
       ];
       assoc
    ]

(* are the set of phi operands (variables) in basic block B 
 * where the source is the basic block S. Usually S is a 
 * successor to (i.e. a child of) B
 *)
phiUses[s_, b_] :=
    Module[{
    	   sid = idOf[s],
    	   res = {}
    },
	    b["scanInstructions",
	    	   Function[{inst},
	    	      If[PhiInstructionQ[inst],
	    	      	  With[{
                      ops = inst["source"]["get"]
                  },
                      MapThread[
                         Function[{incomingBB, val},
                            If[includeQ[val] && sid === idOf[incomingBB],
                                AppendTo[res, val]
                            ]
                         ],
                         Transpose[ops]
                      ]
                  ]
	    	      ]
	    	   ]
	    ];
	    res
    ]
phiUses[b_] :=
    Module[{
        res = {}
    },
        b["scanInstructions",
               Function[{inst},
                  If[PhiInstructionQ[inst],
	                  With[{
	                      ops = inst["source"]["get"]
	                  },
	                      MapThread[
	                         Function[{incomingBB, val},
	                            If[includeQ[val],
	                                AppendTo[res, val]
	                            ]
	                         ],
	                         Transpose[ops]
	                      ]
	                  ]
                  ]
               ]
        ];
        res
    ]

(* are the set of phi targets in basic block B *)
ClearAll[phiDefs]
phiDefs[bb_] :=
    Module[{
        res = {}
    },
        bb["scanInstructions",
               Function[{inst},
                  If[PhiInstructionQ[inst],
                      AppendTo[res, inst["definedVariable"]]
                  ]
               ]
        ];
        res
    ];

(*
 Return True if the two Association have identical Keys. 
 Avoids Sorting the keys.
*)
keySameQ[ x_, y_] :=
	With[{xKeys = Keys[x], yKeys = Keys[y]},
		Length[xKeys] === Length[yKeys] &&
		KeyDrop[x, yKeys] === <||>
	]

	
SetAttributes[timeIt, HoldAllComplete]
accum = 0;
timeIt[e_] :=
	With[{t = AbsoluteTiming[e;][[1]]},
		accum += t;
		Print[StringTake[ToString[Unevaluated[e]], 10], "  t = ", t, "  accum = ", accum]
	]




(**********************************************************)
(**********************************************************)
(**********************************************************)



RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["LiveVariables", "INFO"];

info = CreatePassInformation[
	"LiveVariables",
	"Computes the live variables of each instruction and basic block in the function module",
	"A definition `a` is live at point `b` if it is used after `b` (or subsequent instructions) "<>
	"and there is no intervening dominating definitions between a and b.",
	"References" -> {
        "Boissinot, Benoit, Florian Brandner, Alain Darte, Benoît Dupont de Dinechin, and Fabrice Rastello. \"A non-iterative data-flow algorithm for computing liveness sets in strict SSA programs.\" In Asian Symposium on Programming Languages and Systems, pp. 137-154. Springer, Berlin, Heidelberg, 2011.",
		"Brandner, Florian, Benoit Boissinot, Alain Darte, Benoît Dupont De Dinechin, and Fabrice Rastello. \"Computing Liveness Sets for SSA-Form Programs.\" PhD diss., INRIA, 2011.",
		"Boissinot, Benoit, Sebastian Hack, and Daniel Grund. \"Fast liveness checking for SSA-form programs.\" In Proceedings of the 6th annual IEEE/ACM international symposium on Code generation and optimization, pp. 35-44. ACM, 2008."
	}
];

LiveVariablesPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
    "initializePass" -> initialize,
    "finalizePass" -> finalize,
	"requires" -> {
		DefPass,
		DominatorPass,
		LoopNestingForestPass
	},
	"passClass" -> "Analysis"
|>];

RegisterPass[LiveVariablesPass];
]]


End[] 

EndPackage[]
