BeginPackage["Compile`Core`Analysis`Dominator`DominanceFrontier`"]

DominanceFrontierPass;

Begin["`Private`"]

Needs["Compile`Core`Analysis`Dominator`ImmediateDominator`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	fm["scanBasicBlocks", #["setProperty", "dominanceFrontier" -> {}]&]


run[fm_, opts_] := Module[{bbMap, domChildren = <||>},
    bbMap = fm["basicBlockMap"];
    fm["postOrderScan",
       Function[{bb},
          With[{idom = bb["immediateDominator"]},
            If[idom =!= None,
               AssociateTo[domChildren,
                  idom["id"] -> Append[Lookup[domChildren, idom["id"], {}], bb]
               ]
            ]
          ]
       ]
    ];
    fm["postOrderScan",
       (**
        * DF_local(X) = {Y \in Succ(X) | X does not strictly dominate Y}
        * DF_up(X) = { Y \in DF(Z) | Z \in children(X) and X does not strictly dominate Y}
        * DF(X) = Union( DF_local(X), DF_up(X))
        * where Succ = immediate successors in the CFG
        *       Children = descendents in the dominator tree
        **)
       Function[{bb},
          Module[{df = {}}, 
            Do[ (* DF_local *)
               If[yy["immediateDominator"]["id"] =!= bb["id"],
                  AppendTo[df, yy["id"]] 
               ],
               {yy, bb["getChildren"]}
            ];
            Do[ (* DF_up *)
               (*Print["bb = ", bb, "  zz = ", zz, "  dominanceFrontier", zz["getProperty", "dominanceFrontier"]];*)
               Do[
                   If[yy["immediateDominator"]["id"] =!= bb["id"],
                      AppendTo[df, yy["id"]] 
                   ],
                   {yy, zz["getProperty", "dominanceFrontier"]}
               ],
               {zz, Lookup[domChildren, bb["id"], {}]}
            ];
            bb["setProperty", "dominanceFrontier" -> Map[bbMap["lookup", #]&, DeleteDuplicates[df]]];
            (*Print["result = ", {bb, bb["getProperty", "dominanceFrontier"]}];*)
          ];
       ]
    ];
    fm
];

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"DominanceFrontier",
	"The pass computes the dominance frontier for each basic block in the function module.",
	"The dominance frontier of x is the set of all nodes w such that x dominates the predecessor " <>
	"of w and x does not strictly dominate w . This algorithm is Figure 5 in " <> 
	"A Simple, Fast Dominance Algorithm (http://www.cs.rice.edu/~keith/EMBED/dom.pdf)"
];

DominanceFrontierPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"preserves" -> {ImmediateDominatorPass},
	"traversalOrder" -> "reversePostOrder"
|>];

RegisterPass[DominanceFrontierPass]
]]


End[] 

EndPackage[]
