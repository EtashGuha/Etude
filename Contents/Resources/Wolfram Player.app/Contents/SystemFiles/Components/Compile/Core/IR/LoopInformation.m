
BeginPackage["Compile`Core`IR`LoopInformation`"]

(**
a type declaration
*)
LoopInformation;
CreateLoopInformation;
LoopInformationQ;

Begin["`Private`"] 

Needs["Compile`Core`IR`BasicBlock`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]

(*
 * The loop information class is a nested datastructure that includes other loop informations 
 * or basic blocks. It represents a loop nesting graph with the roon (with the header === None)
 * and children which may be other loop informations or basic blocks. With the exception to the root
 * the header is the basic block that dominates all of its children.
 * 
 * The loop information is computed using the LoopNestingForestPass which implements the 
 * Ramalingam algorithm. 
 *)
RegisterCallback["DeclareCompileClass", Function[{st},
LoopInformationClass = DeclareClass[
    LoopInformation,
    <|
        "scan" -> (scan[Self, #]&),
        "addChild" -> (addChild[Self, ##]&),
        "childrenBasicBlocks" -> (allChildren[Self]&), 
        "toGraph" -> (toGraph[Self, ##]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id",
        "header",
        "edges", (* edges that lead to one of the loop-headers *)
        "children",
        "inductionVariables",
        "invariantVariables"
    },
    Predicate -> LoopInformationQ
]
]]

basicBlockOrNone[_?BasicBlockQ] := True
basicBlockOrNone[None] := True
basicBlockOrNone[___] := False

CreateLoopInformation[id_?IntegerQ, header_?basicBlockOrNone, children_:{}] :=
    Module[ {lookup},
        lookup = CreateObject[LoopInformation, <|
    	    "id" -> id,
    	    "header" -> header,
    	    "edges" -> {},
            "children" -> CreateReference[children],
            "inductionVariables" -> Missing["use the InductionVariablesPass to compute the loops's induction variables"],
            "invariantVariables" -> Missing["use the LoopInvariantPass to compute the loop invariants"]
        |>];
        lookup
    ]
    
CreateLoopInformation[args___] :=
    ThrowException[{"Unknown call to CreateLoopInformation", {args}}]

addChild[ self_, child_?BasicBlockQ] :=
    Module[ {},
        self["children"]["appendTo", child]
    ]
    
addChild[ self_, child_?LoopInformationQ] :=
    Module[ {},
        self["children"]["appendTo", child]
    ]
    
scan[self_, f_] := (
	If[self["children"] =!= Null,
        Scan[f, self["children"]["get"]];
	];
    self
)

isRoot[self_] :=
    self["header"] === None;
    
allChildren[self_] :=
    Join[
       self["children"]["select", BasicBlockQ],
       Catenate[allChildren /@ self["children"]["select", LoopInformationQ]]
    ]

headerName[self_] := If[isRoot[self], "Root", self["header"]["fullName"]]

toGraph0[self_] :=
    With[{
    	   header = headerName[self],
    	   bbs = self["children"]["select", BasicBlockQ]
    },
    With[{
    	   verts = #["fullName"]& /@ bbs
    },
        Graph[
        	   verts,
        	   Map[DirectedEdge[header, #]&, verts],
        	   VertexLabels -> Automatic
        ]
    ]]
toGraph[self_] :=
    With[{
    	   header = headerName[self]
    },
    With[{
    	   subgraphs = Table[
    	   	   Module[{
    	   	   	  g = child["toGraph"]
    	   	   },
    	   	      EdgeAdd[g, DirectedEdge[header, headerName[child]]]
    	   	   ],
    	   	   {child, self["children"]["select", LoopInformationQ]}
    	   ]
    },
    	With[{
    		g = If[subgraphs === {},
    			toGraph0[self],
    			Apply[GraphUnion,
	    	       Prepend[
	    	   	       subgraphs,
	    	   	       toGraph0[self]
	    	       ]
    			]
        ]
    },
        Graph[
           g,
           VertexShapeFunction -> "Square",
           VertexSize -> {.4, .1},
           VertexStyle -> Hue[0.125, 0.7, 0.9],
           VertexLabelStyle -> Directive[FontFamily -> "Arial", 8],
           VertexLabels -> Placed["Name", Center],
           ImageSize -> Scaled[0.5],
           EdgeShapeFunction -> "Arrow",
           GraphLayout ->  {"LayeredEmbedding", "RootVertex" -> header},
           ImagePadding -> 15,
           PlotRangePadding -> 0.2
        ]
    ]]]
        

(**************************************************)






icon := Graphics[Text[
    Style["LOOP\nINFO",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]
      
toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "LoopInformation",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["id: ", {90, Automatic}], self["id"]}],
            BoxForm`SummaryItem[{Pane["header: ", {90, Automatic}], self["header"]}],
            BoxForm`SummaryItem[{Pane["edges: ", {90, Automatic}], self["edges"]}],
            If[MissingQ[self["inductionVariables"]],
            	Nothing,
            	BoxForm`SummaryItem[{Pane["inductionVariables: ", {90, Automatic}], self["inductionVariables"]}]
            ],
            If[MissingQ[self["invariantVariables"]],
            	Nothing,
            	BoxForm`SummaryItem[{Pane["invariantVariables: ", {90, Automatic}], self["invariantVariables"]}]
            ]
        },
        {
            BoxForm`SummaryItem[{Pane["children: ", {90, Automatic}], self["children"]}]
        }, 
        fmt
    ]


toString[env_] := "LoopInformation[<>]"



End[]
EndPackage[]
