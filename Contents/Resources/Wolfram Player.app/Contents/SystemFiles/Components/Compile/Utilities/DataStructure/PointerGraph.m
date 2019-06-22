
BeginPackage["Compile`Utilities`DataStructure`PointerGraph`"]

CreatePointerGraph;
PointerGraph;
AddEdge;
AddVertex;
ToGraph;
Predecessors;
Successors;
AddVertexProperty;
AddEdgeProperty;

InDegree;
OutDegree;

OutEdges;

HasSink;
GetSink;
HasSource;
GetSource;

DeleteEdge;
DeleteVertex;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]

CreatePointerGraph[] :=
	Module[{res},
		res = <|
			    "vertexList" -> CreateReference[{}],
(* Successors *)"edges" -> CreateReference[<||>], (* a s to targets map <| v -> { t1, t2, ..., tn }, ... |> *)
			    "predecessors" -> CreateReference[<||>],
			    "vertexProperties" -> CreateReference[<||>],
			    "edgeProperties" -> CreateReference[<||>],
			    "properties" -> CreateReference[<||>]
		|>;
		PointerGraph[res]
	]
	
CreatePointerGraph[verts_, edges_] :=
	Module[{grph},
		grph = CreatePointerGraph[];
		Scan[AddVertex[grph, #]&, verts];
		Scan[AddEdge[grph, #]&, edges];
		grph
	]


PointerGraph[rgrph_]["getVertexList"] := rgrph["vertexList"]
PointerGraph[rgrph_]["getEdges" | "getSuccessors"] := rgrph["edges"]
PointerGraph[rgrph_]["getPredecessors"] := rgrph["predecessors"]
PointerGraph[rgrph_][key_, args___] := rgrph[key, args]


DeleteVertex[pg_PointerGraph, vert_] :=
	Module[{verts, edges, preds},
		verts = pg["vertexList"];
		If[verts["memberQ", vert],
			edges = pg["edges"];
			preds = pg["predecessors"];
			(* need to delete the vertex from all the predecessors and successor lists *) 
			Do[
			    edges["associateTo", p -> Complement[edges["lookup", p, {}], {vert}]],
			    {p, preds["lookup", vert]}
			];
			Do[
				preds["associateTo", p -> Complement[preds["lookup", p, {}], {vert}]],
			    {p, preds["lookup", vert]}
			];
			edges["keyDropFrom", vert];
			preds["keyDropFrom", vert];
			verts["deleteCases", vert];
			pg["vertexProperties"]["keyDropFrom", vert];
		];
		pg
	]
	
DeleteEdge[pg_PointerGraph, src_ -> trgt_] :=
	DeleteEdge[pg, UndirectedEdge[src, trgt]]
DeleteEdge[pg_PointerGraph, UndirectedEdge[src_, trgt_]] := (
	DeleteEdge[pg, DirectedEdge[src, trgt]];
	DeleteEdge[pg, DirectedEdge[trgt, src]];
	pg
)
DeleteEdge[pg_PointerGraph, DirectedEdge[src_, trgt_]] :=
	Module[{edges, preds},
		edges = pg["edges"];
		preds = pg["predecessors"];
		(* need to delete the vert from the predecessors and successor lists *) 
		edges["associateTo", src -> Complement[edges["lookup", src, {}], {trgt}]];
		preds["associateTo", trgt -> Complement[preds["lookup", trgt, {}], {src}]];
		pg
	]

InDegree[pg_PointerGraph, vert_] :=
	Length[pg["predecessors"]["lookup", vert, {}]]
	
OutDegree[pg_PointerGraph, vert_] :=
	Length[Successors[pg, vert]]
	
OutEdges[pg_PointerGraph, vert_] :=
	Map[DirectedEdge[vert, #]&, Successors[pg, vert]]

HasSink[pg_PointerGraph] :=
	AnyTrue[pg["getVertexList"], (OutDegree[pg, #] === 0)&]
	
GetSink[pg_PointerGraph] :=
	SelectFirst[pg["getVertexList"], (OutDegree[pg, #] === 0)&]
	
HasSource[pg_PointerGraph] :=
	AnyTrue[pg["getVertexList"], (InDegree[pg, #] === 0)&]
	
GetSource[pg_PointerGraph] :=
	SelectFirst[pg["getVertexList"], (InDegree[pg, #] === 0)&]
	
PointerGraph /: VertexList[pg_PointerGraph] :=
	pg["vertexList"]["get"]
	
PointerGraph /: EdgeList[pg_PointerGraph] :=
	Module[{edges, verts},
		verts = pg["vertexList"]["get"];
		edges = pg["edges"]["get"];
		Flatten[
			Table[
				DirectedEdge[vert, #]& /@ Lookup[edges, vert, {}],
				{vert, verts}
			]
		]
	]

PointerGraph /: AddVertex[pg_PointerGraph, s_] :=
	(
		pg["vertexList"]["appendTo", s];
		pg
	)
	
PointerGraph /: AddEdge[pg_PointerGraph, Rule[s_, t_]] :=
	AddEdge[pg, UndirectedEdge[s, t]]
	
PointerGraph /: AddEdge[pg_PointerGraph, UndirectedEdge[s_, t_]] :=
	(
		AddEdge[pg, DirectedEdge[s, t]];
		AddEdge[pg, DirectedEdge[t, s]];
		pg
	)

PointerGraph /: AddEdge[pg_PointerGraph, DirectedEdge[s_, t_]] :=
	Module[{verts, edges, preds},
		verts = pg["vertexList"];
		edges = pg["edges"];
		preds = pg["predecessors"];
		If[verts["freeQ", s],
			verts["appendTo", s]
		];
		If[verts["freeQ", t],
			verts["appendTo", t]
		];
		edges["associateTo", s -> DeleteDuplicates[Join[edges["lookup", s, {}], {t}]]];
		preds["associateTo", t -> DeleteDuplicates[Join[preds["lookup", t, {}], {s}]]];
		pg
	]
	
PointerGraph /: Predecessors[pg_PointerGraph, vert_] :=
	pg["predecessors"]["lookup", vert, {}]
	
PointerGraph /: Successors[pg_PointerGraph, vert_] :=
	pg["edges"]["lookup", vert, {}]

(**********************************************************)
(**********************************************************)
(**********************************************************)

PointerGraph /: AddVertexProperty[pg_PointerGraph, vert_, key_ -> prop_ ] :=
	Module[{props},
		props = pg["vertexProperties"];
		props["associateTo", 
			vert -> Join[
				props["lookup", vert, <||>],
				<| key -> prop |>
			]
		]
	]

PointerGraph /: AddEdgeProperty[pg_PointerGraph, edge_, key_ -> prop_ ] :=
	Module[{props},
		props = pg["edgeProperties"];
		props["associateTo", 
			edge -> Join[
				props["lookup", edge, <||>],
				<| key -> prop |>
			]
		]
	]
	
(**********************************************************)
(**********************************************************)
(**********************************************************)

(pg_PointerGraph)["toGraph", opts___] := ToGraph[pg, opts]

PointerGraph /: ToGraph[pg_PointerGraph, root_:None, opts___] :=
	Module[{verts, edges, vertProps, edgeProps, makeTooltip, layout},
		verts = VertexList[pg];
		edges = EdgeList[pg];
		vertProps = pg["vertexProperties"];
		edgeProps = pg["edgeProperties"];
		makeTooltip[vert_] := Module[{prop},
			prop = vertProps["lookup", vert, None];
			If[prop =!= None,
				prop = Lookup[prop, "Tooltip", None]
			];
			If[prop === None,
				vert,
				Tooltip[vert, prop]
			]
		];
		layout = If[root === None,
			"LayeredDigraphEmbedding",
			{"LayeredDigraphEmbedding", "RootVertex" -> root}
		];
		Graph[
			Map[makeTooltip, verts],
			edges,
			opts,
			VertexLabels -> "Name",
			GraphLayout -> layout,
			ImagePadding -> 15
		]
	]
(**********************************************************)
(**********************************************************)
(**********************************************************)


End[]

EndPackage[]
