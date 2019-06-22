

BeginPackage["Compile`Core`IR`Internal`FunctionModuleTraversal`"]

FunctionModuleTraversalTrait;


Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Core`IR`FunctionModule`"]


FunctionModuleTraversalTrait =
	ClassTrait[<|
		"scanBasicBlocks" -> Function[{fun}, scanBasicBlocks[Self, fun]],
		"scanInstructions" -> Function[{fun}, scanInstructions[Self, fun]],
		"breadthFirstScan" -> Function[{fun}, breadthFirstScan[Self, fun]],
		"reverseBreadthFirstScan" -> Function[{fun}, reverseBreadthFirstScan[Self, fun]],
		"depthFirstScan" -> Function[{fun}, depthFirstScan[Self, fun]],
		"topologicalOrder" -> Function[{}, topologicalOrder[Self]],
		"topologicalOrderScan" -> Function[{fun}, topologicalOrderScan[Self, fun]],
		"postTopologicalOrder" -> Function[{}, postTopologicalOrder[Self]],
		"postTopologicalOrderScan" -> Function[{fun}, postTopologicalOrderScan[Self, fun]],
		"preOrderScan" -> Function[{fun}, preOrderScan[Self, fun]],
		"preOrder" -> Function[{}, preOrder[Self]],
		"reversePreOrder" -> Function[{}, reversePreOrder[Self]],
		"reversePreOrderScan" -> Function[{fun}, reversePreOrderScan[Self, fun]],
		"postOrderScan" -> Function[{fun}, postOrderScan[Self, fun]],
		"postOrder" -> Function[{}, postOrder[Self]],
		"reversePostOrder" -> Function[{}, reversePostOrder[Self]],
		"reversePostOrderScan" -> Function[{fun}, reversePostOrderScan[Self, fun]],
		"invalidateOrder" -> Function[{}, invalidateOrder[Self]; Self]
	|>]





(*
Scan over BasicBlocks in a FunctionModule in an unspecified order.

"associationMap" operates on the key->value pairs, e.g., Rule[1, 2]

"basicBlockMap" is constructed with bbId->bb pairs, so make sure to use #[[2]] to grab the bb

NICE TO HAVE: Implement "associationScan" for association references in Kernel and use here
*)
scanBasicBlocks[fm_, fun_] :=
	fm["basicBlockMap"]["associationMap", fun[#[[2]]]&]

(*
Scan over Instructions in a FunctionModule in an unspecified order.
*)
scanInstructions[fm_, fun_] :=
	fm["scanBasicBlocks", #["scanInstructions", fun]&]








(**************************************************
 * http://www.cs.rice.edu/~keith/EMBED/dom.pdf
 * http://www.cs.nyu.edu/leunga/MLRISC/Doc/html/compiler-graphs.html
 **************************************************)


breadthFirstScan0[fm_, fun_] := 
	Module[{currBB, stack = {}, visited = <||>, children},
		Assert[FunctionModuleQ[fm]];
		currBB = fm["firstBasicBlock"];
		AppendTo[stack, currBB];
		While[stack =!= {},
			{currBB, stack} = {First[stack], Rest[stack]};
			If[!KeyExistsQ[visited, currBB["id"]],
				children = currBB["getChildren"];
				stack = Join[children, stack];
				AssociateTo[visited, currBB["id"] -> True];
				fun[currBB]
			]
		];
	]
		
(** We get the basic block list so that we can modify the 
  * basicblock within the scan loop
  *)  	
breadthFirstScan[fm_, fun_] :=
	Module[{bbs = Internal`Bag[]},
		breadthFirstScan0[fm, Internal`StuffBag[bbs, #]&];
		Scan[fun, Internal`BagPart[bbs, All]];
		fm
	]
	
reverseBreadthFirstScan[fm_, fun_] :=
	Module[{bbs = Internal`Bag[]},
		breadthFirstScan0[fm, Internal`StuffBag[bbs, #]&];
		Scan[fun, Reverse[Internal`BagPart[bbs, All]]];
		fm
	]

topologicalOrder[fm_] :=
Module[{ref, visited, toVisit, next},
	ref = CreateReference[{}];
	visited = CreateReference[<||>];
	toVisit = CreateReference[{}];
	toVisit["pushBack", fm["firstBasicBlock"]];
	While[!toVisit["isEmpty"],
		next = toVisit["popFront"];
		If[!visited["keyExistsQ", next["id"]],
			visited["associateTo", next["id"] -> True];
			Do[
				toVisit["pushBack", m];
				,
				{m, next["getChildren"]}
			];
			ref["pushBack", next]
		]
	];
	ref["toList"]
]
	
topologicalOrderScan[fm_, fun_] :=
	Scan[fun, topologicalOrder[fm]]

postTopologicalOrder[fm_] :=
	Module[{ref, visited, toVisit, next},
	ref = CreateReference[{}];
	visited = CreateReference[<||>];
	toVisit = CreateReference[{}];
	toVisit["pushBack", fm["firstBasicBlock"]];
	While[!toVisit["isEmpty"],
		next = toVisit["popFront"];
		If[!visited["keyExistsQ", next["id"]],
			visited["associateTo", next["id"] -> True];
			Do[
				toVisit["pushBack", m];
				,
				{m, next["getChildren"]}
			];
			ref["pushFront", next]
		]
	];
	ref["toList"]
]
	
	
postTopologicalOrderScan[fm_, fun_] :=
	Scan[fun, postTopologicalOrder[fm]]
	
invalidateOrder[fm_] :=
	Module[{},
	    fm["removeProperty", "topologicalOrder"];
	    fm["removeProperty", "postTopologicalOrder"];
	    fm["removeProperty", "preOrder"];
	    fm["removeProperty", "reversePostOrder"];
	]

depthFirstScan[fm_, fun_] := 
	Module[{currBB, stack = {}, visited = <||>, children},
		Assert[FunctionModuleQ[fm]];
		currBB = fm["firstBasicBlock"];
		AppendTo[stack, currBB];
		While[stack =!= {},
			{currBB, stack} = {First[stack], Rest[stack]};
			If[!KeyExistsQ[visited, currBB["id"]],
				children = SortBy[currBB["getChildren"], #["id"]&];
				stack = Join[stack, children];
				AssociateTo[visited, currBB["id"] -> True];
				fun[currBB]
			]
		];
	];
(* Preorder traversal: visit the parent before its children *)
preOrder[fm_] :=
	Module[{currBB, stack = {}, visited = <||>, order, children},
		Assert[FunctionModuleQ[fm]];
		currBB = fm["firstBasicBlock"];
		order = Internal`Bag[];
		AppendTo[stack, currBB];
		While[stack =!= {},
			{currBB, stack} = {First[stack], Rest[stack]};
			If[!KeyExistsQ[visited, currBB["id"]],
				children = currBB["getChildren"];
				stack = Join[stack, children];
				AssociateTo[visited, currBB["id"] -> True];
				Internal`StuffBag[order, currBB];
			]
		];
		Internal`BagPart[order, All]
	]
	

preOrderScan[fm_, fun_] :=
	Scan[fun, fm["preOrder"]]

reversePreOrderScan[fm_, fun_] :=
	Scan[fun, fm["reversePreOrder"]]
	
reversePreOrder[fm_] :=
	Reverse[fm["preOrder"]]
	
(*
Postorder traversal: visit the children before its children  
1. Push root to first stack.
2. Loop while first stack is not empty
   2.1 Pop a node from first stack and push it to second stack
   2.2 Push left and right children of the popped node to first stack
3. Print contents of second stack
*)
ClearAll[postOrder];
postOrder[fm_] :=
  Module[{currBB, s1 = {}, s2 = {}, visited = <||>, nd, children},
    Assert[FunctionModuleQ[fm]];
    currBB = fm["firstBasicBlock"];
    PrependTo[s1, currBB];
    While[s1 =!= {},
      {nd, s1} = {First[s1], Rest[s1]};
      If[!KeyExistsQ[visited, nd["id"]],
        PrependTo[s2, nd];
        children = nd["getChildren"];
        s1 = Join[s1, children];
        AssociateTo[visited, nd["id"] -> True];
      ];
    ];
    s2
  ];

postOrderScan[fm_, fun_] :=
	Scan[fun, fm["postOrder"]]
	
reversePostOrder[fm_] :=
	Reverse[fm["postOrder"]]
	
reversePostOrderScan[fm_, fun_] :=
	Scan[fun, fm["reversePostOrder"]]

	
End[]
EndPackage[]