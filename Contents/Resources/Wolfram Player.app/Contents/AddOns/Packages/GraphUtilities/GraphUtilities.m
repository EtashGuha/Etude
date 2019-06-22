(* ::Package:: *)

(* :Title: GraphUtilities *)

(* :Context: GraphUtilities` *)

(* :Author: Yifan Hu *)

(* :Summary: A collection of graph theory related function,
    including functions for aesthetic plotting of graphs *)

(* :Copyright: Copyright 2004-2007, Wolfram Research, Inc. *)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 5.1, 6.0 *)

(* :History:
    Version 1.0, Oct. 2004, Yifan Hu.
*)

(* :Keywords:
Graph theory.
*)

(* :Sources:

*)

(* :Warnings: *)

(* :Limitations:  *)

(* :Discussion:

*)

{GraphUtilities`Parent, Combinatorica`Parent} (* Special declaration for a symbol in common *)

BeginPackage["GraphUtilities`"];

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"GraphUtilities`"],
(StringMatchQ[#,StartOfString~~"GraphUtilities`*"]&&
!StringMatchQ[#,"GraphUtilities`"~~("AdjacencyMatrix"|"ClosenessCentrality"|"EdgeList"|"FindHamiltonianCycle"|"GraphDistanceMatrix"|"GraphDistance"|"VertexList")~~__])&]//ToExpression;
];

(* 
If[Not@ValueQ[GraphCoordinates1D::usage],
GraphCoordinates1D::usage = "GraphCoordinates1D[g, options] calculates a 1D layout of the vertices of a graph g and returns the coordinates of the vertices.";
];
*)

If[Not@ValueQ[GraphCoordinates::usage],
GraphCoordinates::usage = "GraphCoordinates[g, options] calculates a visually appealing 2D layout of the vertices of a graph g and returns the coordinates of the vertices.";
];

If[Not@ValueQ[GraphCoordinates3D::usage],
GraphCoordinates3D::usage = "GraphCoordinates3D[g, options] calculates a visually appealing 3D layout of the vertices of a graph g and returns the coordinates of the vertices.";
];

(*
If[Not@ValueQ[GraphCoordinateRules1D::usage],
GraphCoordinateRules1D::usage = "GraphCoordinateRules1D[g, options] calculates a 1D layout of the vertices of a graph g and returns the coordinates of the vertices as a list of rules.";
];

If[Not@ValueQ[GraphCoordinateRules::usage],
GraphCoordinateRules::usage = "GraphCoordinateRules[g, options] calculates a 2D layout of the vertices of a graph g and returns the coordinates of the vertices as a rule list.";
];

If[Not@ValueQ[GraphCoordinateRules3D::usage],
GraphCoordinateRules3D::usage = "GraphCoordinateRules3D[g, options] calculates a 3D layout of the vertices of a graph g and returns the coordinates of the vertices as a list of rules.";
];
*)

(*
If[Not@ValueQ[GraphDistance::usage],
GraphDistance::usage="GraphDistance[g, i, j] calculates the distance between vertices i and j in a graph g. If there is no path from i to j, the distance is Infinity";
];
*)

If[Not@ValueQ[MaximalBipartiteMatching::usage],
MaximalBipartiteMatching::usage="MaximalBipartiteMatching[g] returns the maximal matching of the bipartite graph as represented by a matrix g. For a m \[Times] n matrix, the output is a list of index pairs, of form {{i_1, j_1}, ..., {i_k, j_k},...} where 1 <= i_k <= m and 1 <= j_k <= n";
];

If[Not@ValueQ[MaximalIndependentVertexSet::usage],
MaximalIndependentVertexSet::usage="MaximalIndependentVertexSet[g] gives a maximal independent vertex set of an undirected graph g. If g is a directed graph, it is treated as if it is undirected.";
];

If[Not@ValueQ[MaximalIndependentEdgeSet::usage],
MaximalIndependentEdgeSet::usage="MaximalIndepndentEdgeSet[g] returns a maximal independent edge set, also known as maximal matching, of an undirected graph g. If g is a directed graph, it is treated as if it is undirected.";
];

If[Not@ValueQ[MinCut::usage],
MinCut::usage="MinCut[g,k] gives a partition an undriected graph g into k parts. The partition groups the vertices into k groups, such that the number of edges between these groups are approximately minimized. A directed graph is treatd as if it is undirected.";
];

If[Not@ValueQ[PseudoDiameter::usage],
PseudoDiameter::usage="PseudoDiameter[g] gives the pseudo diameter of the undirected graph g, and the two vertices that achieve this diameter. If the graph is disconnected, then the diameter for each of the component, together with the indices of the two vertices that achieves this diameter, are returned. If g is a directed graph, it is treated as if it is undirected.";
];

If[Not@ValueQ[StrongComponents::usage],
StrongComponents::usage="StrongComponents[g] returns a list of all strongly connected components in the directed graph as represented by a matrix g.";
];

If[Not@ValueQ[WeakComponents::usage],
WeakComponents::usage="WeakComponents[g] returns a list of all weakly connected components in the directed graph as represented by a matrix g.";
];

(*
If[Not@ValueQ[VertexList::usage],
VertexList::usage="VertexList[g] returns a list of all vertices.";
];
*)

(*
If[Not@ValueQ[EdgeList::usage],
EdgeList::usage="EdgeList[g] returns a list of all edges.";
];
*)

(*
If[Not@ValueQ[AdjacencyMatrix::usage],
AdjacencyMatrix::usage="AdjacencyMatrix[g] returns the sparse array representation of the graph g. If the graph g has m vertices, AdjacencyMatrix[g, n] returns the sparse array representing of the graph g, with the last n-m rows and columns padded by zeros. If n <= m, AdjacencyMatrix[g, n] is the same as AdjacencyMatrix[g].";
];
*)

If[Not@ValueQ[Bicomponents::usage],
Bicomponents::usage="Bicomponents[g] returns the biconnected components of the undirected graph g.";
];

If[Not@ValueQ[MinimumBandwidthOrdering::usage],
MinimumBandwidthOrdering::usage="MinimumBandwidthOrdering[g] returns the vertex ordering r that attempts to minimize the bandwidth. MinimumBandwidthOrdering[m], for a matrix m, returns a pair of ordering {r, c} so that the bandwidth of the matrix m[[r,c]] is minimized.";
];

If[Not@ValueQ[PageRankVector::usage],
PageRankVector::usage="PageRankVector[g] returns the PageRank of graph g as a vector.";
];

If[Not@ValueQ[PageRanks::usage],
PageRanks::usage="PageRanks[g] returns a list of rules giving the PageRank of each vertices of graph g.";
];

If[Not@ValueQ[LinkRankMatrix::usage],
LinkRankMatrix::usage="LinkRankMatrix[g] returns the LinkRank of graph g, in the form of a sparse matrix. The LinkRank of an edge u->v is defined as the PageRank of u, divided by the outer-degree of u.";
];

If[Not@ValueQ[LinkRanks::usage],
LinkRanks::usage="LinkRanks[g] returns a list of rules giving the LinkRank of each edges of graph g.";
];

If[Not@ValueQ[CommunityStructurePartition::usage],
CommunityStructurePartition::usage="CommunityStructurePartition[g] gives the partition of a graph g into communities. The partition groups the vertices into communities, such that there is a higher density of edges within communities than between them.";
];

If[Not@ValueQ[CommunityStructureAssignment::usage],
CommunityStructureAssignment::usage="CommunityStructureAssignment[g] gives the assignment of vertices of a graph g into communities. The assignment groups the vertices into communities, such that there is a higher density of edges within communities than between them.";
];

If[Not@ValueQ[CommunityModularity::usage],
CommunityModularity::usage="CommunityModularity[g, partition] gives community modularity of a partition. CommunityModularity[g, assignment] gives community modularity of an assignment.";
];

(*
If[Not@ValueQ[GraphDistanceMatrix::usage],
GraphDistanceMatrix::usage="GraphDistanceMatrix[g] gives a matrix, where the (i, j)th entry is the length of a shortest path in g between vertices i and j. GraphDistanceMatrix[g, Parent] returns a three-dimensional matrix, in which the (1, i, j)th entry is the length of a shortest path from i to j and the (2, i, j)th entry is the predecessor of j in a shortest path from i to j.";
];
*)

If[Not@ValueQ[GraphPath::usage],
GraphPath::usage = "GraphPath[g, start, end] finds a shortest path between vertices start and end in graph g";
];

(*
If[Not@ValueQ[ClosenessCentrality::usage],
ClosenessCentrality::usage = "ClosenessCentrality[g] finds the closensss centrality. Closeness centrality of a vertex u is defined as the inverse of the sum of distance from u to all other vertices. Closeness centrality of a vertex in a disconnected graph is based on the closeness centrality of the component where this vertex belongs.";
];
*)

If[Not@ValueQ[ToCombinatoricaGraph::usage],
ToCombinatoricaGraph::usage="Giving a graph g in any of the format acceptable by the GraphUtilities package, ToCombinatoricaGraph[g] returns a Combinatoric representation of this graph. ToCombinatoricaGraph[g, n] returns a Combinatorica representation of g with at least n vertices, adding additional unconnected vertices when necessary.";
];

If[Not@ValueQ[Aggressive::usage],
Aggressive::usage = "Aggressive is an option of PseudoDiameter, if option Aggressive->True is used, when the distance between a start and an end vertices u and v no longer increases, the algorithm will be carried out for one extra step by starting from each vertex w that is farthest away from u. The pseudo-diameter is the farthest distance possible from all such w. Possible values are False (the default) or True.";
];

If[Not@ValueQ[RecursionMethod::usage],
RecursionMethod::usage = "RecursionMethod is an option of MinimumBandwidthOrdering. This specifies whether to employ a multilevel process to find a minimal bandwidth ordering. Possible values for this option are None (the default) and Multilevel.";
];

If[Not@ValueQ[RefinementMethod::usage],
RefinementMethod::usage = "RefinementMethod is an option of MinimumBandwidthOrdering. This specifies the refinement method used to further improve the bandwidth following the application of one of the above methods. Possible values for this option are Automatic (the default), None, HillClimbing, and NodeCentroidHillClimbing.";
];

If[Not@ValueQ[TeleportProbability::usage],
TeleportProbability::usage = "TeleportProbability is an option of PageRanks and LinkRanks. It specified the probability that the Internet surfer may choose to visit nodes randomly, instead of following the out-links of a vertex. The default value is 0.15.";
];

If[Not@ValueQ[RemoveSinks::usage],
RemoveSinks::usage = "RemoveSinks is an option of PageRanks and LinkRanks. It specified whether sinks (a sink is a vertex with no outer links) are removed by link with all vertices. The default value is True.";
];

If[Not@ValueQ[Weighted::usage],
Weighted::usage = "Weighted is an option of MaximalIndependentEdgeSet, it specifies whether edges with higher weights are preferred during matching. Possible values are True or False (the default). \n Weighted is an option of CommunityStructurePartition and CommunityModularity, it specifies whether vertices linked by edges with higher weights are preferrablly kept in the same community. Possible values are True or False (the default). \n Weighted is also an option of ShortestedPath, GraphDistance, GraphDistanceMatrix and ClosenessCentrality, it specifies whether edge weight is to be used in calculating distance. Possible values are True (the default except for GraphDistance) or False (the default for GraphDistance).";
];

If[Not@ValueQ[LineScaledCoordinate::usage],
LineScaledCoordinate::usage = "LineScaledCoordinate[{{x1,y1},{x2,y2},...,{xk,yk}}, r] gives the coordinate of a point in the polyline {{x1,y1},{x2,y2},...,{xk,yk}}, at a scaled distance of r from point {x1,y1}. LineScaledCoordinate[{{x1,y1},{x2,y2},...,{xk,yk}}] is the same as LineScaledCoordinate[{{x1,y1},{x2,y2},...,{xk,yk}}, 0.5]";
];

If[Not@ValueQ[GraphEdit::usage],
GraphEdit::usage = "GraphEdit[] opens a graph editor. GraphEdit[g] edits the graph g.";
];

If[Not@ValueQ[ExpressionTreePlot::usage],
ExpressionTreePlot::usage = "ExpressionTreePlot[e] plots an expression tree of the given expression e, with head of e as the the root, and heads of subexpressions as other vertices. Tooltip for each vertex is given as the subexpression under than vertex. ExpressionTreePlot[e, pos] plot the expression tree with the root placed at position pos. ExpressionTreePlot[e, pos, lev] plot the expression tree with the root placed at position pos, up to level lev.";
];

If[Not@ValueQ[NeighborhoodVertices::usage],
NeighborhoodVertices::usage = "NeighborhoodVertices[g, v] gives a list of all vertices reachable from vertex v, including v itself; NeighborhoodVertices[g, v, n] gives a list of all vertices reachable from vertex v within n hops, including v itself.";
];

If[Not@ValueQ[NeighborhoodSubgraph::usage],
NeighborhoodSubgraph::usage = "NeighborhoodSubgraph[g, v] gives a subgraph consists of vertices reachable from vertex v; NeighborhoodSubgraph[g, v, n] gives a subgraph consists of all vertices reachable from vertex v within n hops.";
];

(*
If[Not@ValueQ[FindHamiltonianCycle::usage],
FindHamiltonianCycle::usage = "FindHamiltonianCycle[g] attempt to find a Hamiltonian cycle using heuristics.";
];
*)

If[Not@ValueQ[HamiltonianCycles::usage],
HamiltonianCycles::usage = "HamiltonianCycles[g] finds a Hamiltonian cycle. HamiltonianCycles[g, n] finds up to n Hamiltonian cycles.";
];

GraphUtilities`Parent = Combinatorica`Parent;

Begin["`Private`"];


(*================ Error messages =========== *)

LinkRanks::sqma = LinkRankMatrix::sqma = PageRanks::sqma = PageRankVector::sqma = Bicomponents::sqma = WeakComponents::sqma = StrongComponents::sqma = GraphDistance::sqma = GraphPath::sqma = GraphDistanceMatrix::sqma = NeighborhoodVertices::sqma = FindHamiltonianCycle::sqma = HamiltonianCycles::sqma = "`1` is not a square matrix";

MaximalIndependentVertexSet::vtxwgt = "`1` is not a list of length equal to the number of vertices.";

ClosenessCentrality::grp = GraphDistanceMatrix::grp = GraphPath::grp = CommunityStructureAssignment::grp = CommunityStructurePartition::grp = CommunityModularity::grp = "Argument `1` at position 1 does not represent a graph.";

ClosenessCentrality::negc = GraphDistanceMatrix::negc = GraphDistance::negc = GraphPath::negc = "Negative cycle found.";

ClosenessCentrality::negw = GraphDistanceMatrix::negw = GraphPath::negw = "Method->Dijkstra can not be applied to a graph containing negative edge weight.";

ClosenessCentrality::wgh1 = GraphDistanceMatrix::wgh1 = GraphDistance::wgh1 = GraphPath::wgh1 = "Warning: some of the edge weights are not machine real numbers, it is assumed all edge weights are 1.";

ClosenessCentrality::zdis = "Close centrality can not be calculated because the distance of one vertex to others adds up to zero.";

MaximalIndependentVertexSet::rug = MinCut::rug = PseudoDiameter::rug = MaximalIndependentEdgeSet::rug = "Argument `1` at position 1 does not represent an undirected graph.";

GraphDistance::wgt = GraphDistanceMatrix::wgt = GraphPath::wgt = CommunityStructureAssignment::wgt = CommunityStructurePartition::wgt = CommunityModularity::wgt = MaximalIndependentEdgeSet::wgt = "The value of option Weighted->`1` must be either True or False.";

CommunityStructureAssignment::wgtm = CommunityStructurePartition::wgtm = CommunityModularity::wgtm = "When Weighted->True is specified, the matrix, `1`, must have all nonnegative entries.";
CommunityModularity::part = "`1` is not a valid assignment or partition.";

MaximalIndependentEdgeSet::symat = "`1` is not a symmetric matrix.";

MinCut::kgtwo = "The value of the second argument, `1`, must be an integer >= 1";

NeighborhoodSubgraph::rind = NeighborhoodVertices::rind = GraphDistance::rind = "Argument `1` at position `2` is not a valid vertex.";

GraphCoordinates1D::gc1dm = GraphCoordinateRules1D::gc1dm = "The value of option Method -> `1` should be Automatic, \"Spectral\", \"SpectralOrdering\", or \"TwoNormApproximate\".";

LinkRanks::tol = LinkRankMatrix::tol = PageRanks::tol = PageRankVector::tol = "The value of option Tolerance -> `1` should be Automatic, or a positive machine sized number.";

LinkRanks::tel = LinkRankMatrix::tel = PageRanks::tel = PageRankVector::tel = "The value of option TeleportProbability -> `1` should be a positive machine real number less than 1.";

LinkRanks::rms = LinkRankMatrix::rms = PageRanks::rms = PageRankVector::rms = "The value of option RemoveSinks -> `1` should be True or False.";

GraphPath::wmthd = "Value of option Method->`1` must be Automatic, BellmanFord, or Dijkstra.";
GraphDistanceMatrix::wmthd = "Value of option Method->`1` must be Automatic, Dijkstra, Johnson, or FloydWarshall.";

FindHamiltonianCycle::maxit = "Warning: value of option MaxIterations->`1` must be either Automatic, or a positive integer. Proceed with MaxIterations->100."

FindHamiltonianCycle::seed = "Warning: value of option RandomSeed->`1` must be either Automatic, or an integer. Proceed with RandomSeed->0."

HamiltonianCycles::nham = "The second argument `1` must be a positive integer or All."

NeighborhoodSubgraph::und = NeighborhoodVertices::und = "Warning: value of option Undirected->`1` must be either True or False."

 (*  option processing *)
stringName[sym_Symbol] := SymbolName[sym];
stringName[name_] := name;
SetAttributes[processOptionNames, Listable];
processOptionNames[(r : (Rule | RuleDelayed))[name_Symbol, val_]] := 
    r[SymbolName[name], val];
processOptionNames[opt___] := opt;
filterOptions[hiddenOpts_List, command_Symbol, options___?OptionQ] := 
  (filterOptions[
   First /@ processOptionNames[Flatten[{Options[command], hiddenOpts}]], 
   options]);
filterOptions[command_Symbol, options___?OptionQ] := 
  (filterOptions[{}, command, options]);
filterOptions[optnames_List, options___?OptionQ] := 
  (Select[Flatten[{options}], 
   MemberQ[optnames, stringName[First[#]]] &]);

leftOverOptions[command_Symbol, options___?OptionQ]:=
Complement[Flatten[{options}], filterOptions[hiddenOptions[command],command, options]];


SetAttributes[argLength, HoldFirst];
argLength[b:(f_[args___, opts:((_Rule|_RuleDelayed)...)])]:=
 Block[{},
   If [args==={}, Return[0]];
   Length[{args}]
];

IssueArgsMessage[caller_, n_, nmin_, nmax_]:= Module[
  {},
  (* do not issue message if args ok, this happens
     say for GraphDistance[{1 -> 2}, 1, 3] *)
  If [n >= nmin && n <= nmax, Return[$Failed]];
  If [nmin === nmax, 
     If [nmin == 1,
        Message[caller::argx, caller, n, nmax],
        Message[caller::argrx, caller, n, nmax]
     ],
     Message[caller::argb, caller, n, nmin, nmax]
  ];
  $Failed;
];

SetAttributes[checkopt, HoldFirst];
checkopt[e:(f_[args___, opts:((_Rule | _RuleDelayed) ...)]), nmin_, nmax_]:=
Module[
  {n = argLength[e]},
  If [n < nmin || n > nmax,
    If [nmin === nmax, 
       If [nmin == 1,
          Message[f::argx, f, n, nmax],
          Message[f::argrx, f, n, nmax]
       ],
       Message[f::argb, f, n, nmin, nmax]
    ];
	 Return[False];
   ];
   res = leftOverOptions[f, opts];
   If [Length[res] > 0,
      Message[f::optx, res[[1,1]], Unevaluated[e]]; False,
      True
   ]   
];
checkopt[e : (f_[args___, opts : ((_Rule | _RuleDelayed) ...)])] := Module[
 {res},
 res = leftOverOptions[f, opts];
 If [Length[res] > 0, 
    Message[f::optx, res[[1,1]], Unevaluated[e]]; False,
    True
 ]
]

 (*  option processing *)
StringName[sym_Symbol] := SymbolName[sym];
StringName[name_] := name;
SetAttributes[processOptionNames, Listable];
processOptionNames[(r : (Rule | RuleDelayed))[name_Symbol, val_]] := 
    r[SymbolName[name], val];
processOptionNames[opt___] := opt;
filterOptions[hiddenOpts_List, command_Symbol, options___?OptionQ] := 
  (filterOptions[
   First /@ processOptionNames[Flatten[{Options[command], hiddenOpts}]], 
   options]);
filterOptions[command_Symbol, options___?OptionQ] := 
  (filterOptions[{}, command, options]);
filterOptions[optnames_List, options___?OptionQ] := 
  (Select[Flatten[{options}], 
   MemberQ[optnames, StringName[First[#]]] &]);

DeleteOptions[optnames_List, opts___] := 
 Select[
   Flatten[{opts}], !MemberQ[optnames, StringName[First[#]]]&]
DeleteOptions[optnames_, opts___] := DeleteOptions[{optnames}, opts];

orderOptions[opts_]:= Module[{x}, x=Ordering[Map[StringName[#[[1]]]&, opts]]; opts[[x]]];

defaultOptions[GraphCoordinates1D] = Options[GraphCoordinates1D] = {Method->Automatic};
defaultOptions[GraphCoordinates] = Options[GraphCoordinates] = Options[GraphPlot];
defaultOptions[GraphCoordinates3D] = Options[GraphCoordinates3D] = Options[GraphPlot3D];

defaultOptions[GraphCoordinateRules1D] = Options[GraphCoordinateRules1D] = {Method->Automatic};
defaultOptions[GraphCoordinateRules] = Options[GraphCoordinateRules] = Options[GraphPlot];
defaultOptions[GraphCoordinateRules3D] = Options[GraphCoordinateRules3D] = Options[GraphPlot3D];
defaultOptions[ToCombinatoricaGraph] = Options[ToCombinatoricaGraph] = {Method->Automatic};
defaultOptions[MaximalIndependentEdgeSet] = Options[MaximalIndependentEdgeSet] = {GraphUtilities`Weighted->False};
defaultOptions[PseudoDiameter] = Options[PseudoDiameter] = {GraphUtilities`Aggressive->False};
defaultOptions[MinimumBandwidthOrdering] = Options[MinimumBandwidthOrdering] = {Method->Automatic, GraphUtilities`RefinementMethod->Automatic, GraphUtilities`RecursionMethod->None};
defaultOptions[PageRanks] = defaultOptions[PageRankVector] = defaultOptions[LinkRanks] = defaultOptions[LinkRankMatrix] = Options[PageRankVector] = Options[PageRanks] = Options[LinkRanks] = Options[LinkRankMatrix] = {Tolerance->Automatic, TeleportProbability->.15, RemoveSinks->True};
defaultOptions[CommunityModularity] = defaultOptions[CommunityStructurePartition] = defaultOptions[CommunityStructureAssignment] = Options[CommunityModularity] = Options[CommunityStructurePartition] = Options[CommunityStructureAssignment] = {GraphUtilities`Weighted->False};

defaultOptions[ExpressionTreePlot] = Options[ExpressionTreePlot] = Options[TreePlot];


defaultOptions[NeighborhoodVertices] = Options[NeighborhoodVertices] = NondefaultOptions[NeighborhoodVertices] = {"Undirected"->False};

defaultOptions[NeighborhoodSubgraph] = Options[NeighborhoodSubgraph] = NondefaultOptions[NeighborhoodSubgraph] = {"Undirected"->False};

defaultOptions[HamiltonianCycles] = Options[HamiltonianCycles] = NondefaultOptions[HamiltonianCycles] = {};

(* Options on overloaded functions *)
u = Unprotect[{ClosenessCentrality, FindHamiltonianCycle, GraphDistance, GraphDistanceMatrix}];

defaultOptions[ClosenessCentrality] = Options[ClosenessCentrality] = {GraphUtilities`Weighted->True};

If[FreeQ[Options[FindHamiltonianCycle], MaxIterations],
Options[FindHamiltonianCycle] = Append[Options[FindHamiltonianCycle],{MaxIterations->100, RandomSeed->0}];
]
defaultOptions[FindHamiltonianCycle] = NondefaultOptions[FindHamiltonianCycle] = Options[FindHamiltonianCycle];

defaultOptions[GraphDistanceMatrix] = defaultOptions[GraphPath] = Options[GraphDistanceMatrix] = Options[GraphPath] = {Method->Automatic, GraphUtilities`Weighted->True};

If[FreeQ[Options[GraphDistance], GraphUtilities`Weighted],
Options[GraphDistance] = Append[Options[GraphDistance],{GraphUtilities`Weighted->False}];
defaultOptions[GraphDistance] = Options[GraphDistance];
]

Protect @@ u;


hiddenOptions[_] := {};

NondefaultOptions[symbol_]:= Module[
  {res={}, dfo, opt},
  (* get the nondefault options set by user
    with SetOptions[symbol] *)

  dfo = processOptionNames[defaultOptions[symbol]];
  opt = processOptionNames[Flatten[{Options[symbol]}]];
  ophash[x_]:= nothing;

  Map[(ophash[#[[1]]] = #[[2]])&,dfo];
  Map[If [ophash[#[[1]]] =!= #[[2]], 
         res = {res, #}]&, opt];
  ophash = .;
  Clear[ophash];
  Sequence@@Flatten[res]
] 

u = Unprotect[{AdjacencyMatrix, EdgeList, VertexList}];

AdjacencyMatrix[G_?GraphQ, Automatic] := AdjacencyMatrix[G]
AdjacencyMatrix[G_?GraphQ, n_] := Module[
   {A},
   A = SparseArray[AdjacencyMatrix[G]];
   If [n > Dimensions[A][[1]], SparseArray[A, {n,n}], A]
]

AdjacencyMatrix[x_?InternalGraphQ, r___] := With [{res = Catch[Network`GraphPlot`AdjacencyMatrix[x, r]]},
  res/;(Head[res] === SparseArray)
]

G2S := AdjacencyMatrix;

VertexList[x_?InternalGraphQ, r___] := With[{res = Network`GraphPlot`VertexList[x, r]},
 res/;ListQ[res]];

EdgeListInternal[g_]:= Module[
  {vtxnames, A},
  vtxnames = VertexList[g];
  If [!ListQ[vtxnames], Return[$Failed]];
  A = AdjacencyMatrix[g];
  Map[({vtxnames[[#[[1,1]]]], vtxnames[[#[[1,2]]]]}&), Drop[ArrayRules[A],-1]]
];

EdgeList[{}]:= {};
EdgeList[Combinatorica`Graph[{},{}]]:= {};

EdgeList[g_?InternalGraphQ]:= With[
  {res = EdgeListInternal[g]},
  res/;res =!= $Failed
];

Protect @@ u;

RuleListGraphQ := Network`GraphPlot`RuleListGraphQ;

InternalGraphQ := Network`GraphPlot`GraphQ;

(e:ToCombinatoricaGraph[x___, opts___?OptionQ])/;checkopt[e, 1, 2]:= (
Needs["Combinatorica`"];
With[
  {res = ToCombinatoricaGraphInternal[x,opts]},
  res/;(res =!= $Failed)
]
);

ToCombinatoricaGraphInternal[g_, opts___?OptionQ] := Module[
  {res},
  If [g==={}, Return[Combinatorica`EmptyGraph[0]]];
  res = Catch[ToCombinatoricaGraphInternalCore[g, Automatic, opts]];
  res
];

ToCombinatoricaGraphInternal[g_, n_, opts___?OptionQ] := Module[
  {res},
  res = Catch[ToCombinatoricaGraphInternalCore[g, n, opts]];
  res
];

NumericMatrixQ[B_?MatrixQ]:= Module[
  {res = True},
  If [SparsePatternArrayQ[B], Return[False]];
  If [Developer`PackedArrayQ[B], Return[True]];
  MatrixQ[B,NumericQ]
];

RuleListToCombinatorica[G_, AG_, m_: 0] := 
  Module[{ind, f, a, b, nz = 1, s, vtxnames, edgelabels,
     v, n = 0, x, mthd, i, coord, gr, uniqueverts, needsreindexq, ord},
  (* from a rule list object G, and its adjacency matrix
   AdjacencyMatrix[G,m], get a Combinatorica object layed
   out with circular embedding *)
   f[Rule[x_, y_]] := {{v[x], v[y]}};
   f[Rule[Tooltip[x_, tp_], y_]] := f[Rule[x, y]];
   f[Rule[x_, Tooltip[y_, tp_]]] := f[Rule[x, y]];
   f[{Rule[x_, y_], elabel_}] := {{v[x], v[y]}, 
     Combinatorica`EdgeLabel -> elabel};
   f[{Rule[Tooltip[x_, tp_], y_], elabel_}] := 
    f[{Rule[x, y], elabel}];
   f[{Rule[x_, Tooltip[y_, tp_]], elabel_}] := 
    f[{Rule[x, y], elabel}];
   f[Tooltip[Rule[x_, y_], tp_]] := f[Rule[x, y]];
   ind = Map[f, G];
   uniqueverts = DeleteDuplicates[Cases[ind, v[___], Infinity]];
   n = Length[uniqueverts];
   ord = Ordering[uniqueverts];
   uniqueverts = uniqueverts[[ord]];
   ind = ind/.MapThread[Rule, {uniqueverts, Range[n]}];
   vtxnames = First /@ uniqueverts;
   coord = GraphCoordinates[AG, Method -> "CircularEmbedding"];
   gr = Combinatorica`Graph[ind, Map[List, coord], Combinatorica`EdgeDirection -> True];
   If[n < m,
       vtxnames = Join[vtxnames, Range[n + 1, m]];
       ord = Join[ord, Range[n + 1, m]]
   ];
   {Combinatorica`SetVertexLabels[gr, vtxnames], ord}
   ];

ToCombinatoricaGraphInternalCore[g_, n_, opts___?OptionQ] := 
  Module[
   {a, gr = g, m, coord, mthd, caller = ToCombinatoricaGraph, labels, ar, ord = Automatic},

	mthd = "Method"/.processOptionNames[Flatten[{opts}]]
         /.processOptionNames[Options[ToCombinatoricaGraph]];
   a = AdjacencyMatrix[g, n];
   If [Head[a] =!= SparseArray, Throw[$Failed]];
   If [Head[g] =!= Combinatorica`Graph,
     (* This does not take care of edge labels, so for rulelist we
        use RuleListToCombinatorica instead*)
     If [RuleListGraphQ[g],
            {gr, ord} = RuleListToCombinatorica[g, a, n],
  	    gr = Combinatorica`FromAdjacencyLists[a@AdjacencyLists[],
         Combinatorica`Type -> Combinatorica`Directed];
       labels = VertexList[g];
       m = Length[labels];
       If [n > m, labels = Join[labels, Range[m+1, n]]];
       gr = Combinatorica`SetVertexLabels[gr, labels]
     ],
     (* add extra vertices. If method = Automatic, we 
        do not touch coordinates *)
     If [StringName[mthd] === "Automatic", mthd = None];
     m = Length[Combinatorica`GetVertexLabels[g]];
     gr = g;
     If [n - m > 0, Do[gr = Combinatorica`AddVertex[gr],{i,n - m}]];
   ];

   (* add edge weight if the input is a matrix *)
	If [NumericMatrixQ[g],
      ar = Drop[ArrayRules[g], -1];
	   edges = Combinatorica`Edges[gr];
      gr = Combinatorica`SetEdgeWeights[gr, edges, edges/.ar]
   ];

   (* find coordinate *)
   Switch[StringName[mthd],
     "None",
      coord = None,
     _,
      coord = SparseArray`Private`GraphPlot[caller, a, Method->mthd];
      If [!ListQ[coord], Throw[$Failed]];
      coord = Take[coord[[1]],Dimensions[a][[1]]];
      If[ord =!= Automatic && Length[ord] === Length[coord],
          coord = coord[[ord]]
	];
   ]; 

   (* set coord *)
   If [coord =!= None, gr = Combinatorica`ChangeVertices[gr, coord]];
	
   gr
];

extractCoordinate[res_]:=
  VertexCoordinateRules /. Cases[res, Rule[x_, y_] -> (x -> y), {0, Infinity}];

GraphCoordinates[_?((GraphQ[#] && VertexCount[#] === 0)&), ___] := {}
GraphCoordinates[G_?GraphQ, opts___?OptionQ] :=
    GraphCoordinates[AdjacencyMatrix[G], opts]

(e: GraphCoordinates[G_?InternalGraphQ, opts___?OptionQ])/;checkopt[e] := With[
   {res = GraphPlot[G, opts]},
   extractCoordinate[res]/;(Head[res] =!= GraphPlot)
];

GraphCoordinateRules[{}, opts___?OptionQ]:={};
GraphCoordinateRules[Combinatorica`Graph[{},{}], opts___?OptionQ]:={};

(e: GraphCoordinateRules[G_Combinatorica`Graph, opts___?OptionQ])/;checkopt[e] := With[
   {res = GraphPlot[G, opts], vtx = VertexList[G]},
   Thread[vtx -> extractCoordinate[res]]/;(Head[res] =!= GraphPlot)
];

GraphCoordinates[{}, opts___?OptionQ] := {};
GraphCoordinates[Combinatorica`Graph[{},{}], opts___?OptionQ] := {};

(e: GraphCoordinates3D[G_?InternalGraphQ, opts___?OptionQ])/;checkopt[e] := With[
   {res = GraphPlot3D[G, opts]},
   extractCoordinate[res]/;(Head[res] =!= GraphPlot)
];

(e: GraphCoordinateRules3D[G_Combinatorica`Graph, opts___?OptionQ])/;checkopt[e] := With[
   {res = GraphPlot3D[G, opts], vtx = VertexList[G]},
   Thread[vtx -> extractCoordinate[res]]/;(Head[res] =!= GraphPlot)
];

GraphCoordinates3D[{}, opts___?OptionQ] := {};
GraphCoordinates3D[_?((GraphQ[#] && VertexCount[#] === 0)&), ___] := {}
GraphCoordinates3D[Combinatorica`Graph[{},{}], opts___?OptionQ] := {};

GraphCoordinates1D[{}, opts___?OptionQ] := {};

GraphCoordinates1D[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};


(e: GraphCoordinates1D[g_?InternalGraphQ, opts___?OptionsQ])/;checkopt[e] := With[
   {res = Network`GraphPlot`GraphLayout1D[GraphCoordinates1D, AdjacencyMatrix[g], 
           opts]},
   res/;(res =!= $Failed)
];


GraphCoordinateRules1D[{}, opts___?OptionQ] := {};
(e: GraphCoordinateRules1D[g_?InternalGraphQ, opts___?OptionsQ])/;checkopt[e] := With[
   {res = Network`GraphPlot`GraphLayout1D[GraphCoordinates1D, AdjacencyMatrix[g], 
           opts]},
   Thread[VertexList[g]->res]/;(res =!= $Failed)
];

(* end 1D stuff *)

StructurallySymmetricMatrixQ[A_]:= Module[
  {B, ind, val},
  If [!MatrixQ[A], Return[False]];
  If [Head[A] =!= SparseArray, B = SparseArray[A], B = A];
  SparseArray`SymmetricQ[B, "Test"->"Structural"]
];

squareMatrixQ[A_]:= Module[
  {},
  If [!MatrixQ[A], Return[False]];
  {m,n}=Dimensions[A];
  If [m =!= n, False, True]
];

GraphDistance2[G_?RuleListGraphQ, i_, j_, opts___?OptionQ]:= Module[{},
      GraphDistanceInternal[checkGraph[GraphDistance, G], CheckVertex[GraphDistance, G, i, 2], CheckVertex[GraphDistance, G, j, 3], opts, NondefaultOptions[GraphPath]]
     
];
GraphDistance2[G_Combinatorica`Graph, i_Integer, j_Integer, opts___?OptionQ]:= Module[
   {},
   GraphDistanceInternal[checkGraph[GraphDistance, G], CheckVertex[GraphDistance, G, i, 2], CheckVertex[GraphDistance, G, j, 3], opts, NondefaultOptions[GraphPath]]
];
GraphDistance2[A_?MatrixQ, i_Integer, j_Integer, opts___?OptionQ]:= Module[
      {},
      GraphDistanceInternal[checkGraph[GraphDistance, A], CheckVertex[GraphDistance, A, i, 2], CheckVertex[GraphDistance, A, j, 3], opts, NondefaultOptions[GraphPath]]
];

u = Unprotect[GraphDistance];
(e:GraphDistance[x_?InternalGraphQ, r___, opts___?OptionQ])/;checkopt[e, 3, 3]:= With[
  {res = Catch[GraphDistance2[x, r, opts]]},
  res/;(res =!= $Failed && Head[res] =!= GraphDistance2)
];
Protect @@ u;

GraphDistanceInternal[A_?MatrixQ, i_Integer, j_Integer, opts___?OptionQ]:= Module[
   {res, n = Dimensions[A][[1]], weighted},
   If [!squareMatrixQ[A],
       Message[GraphDistance::sqma, A];
       Return[$Failed]
   ];
   weighted = "Weighted"/.processOptionNames[Flatten[{opts}]]
         /.processOptionNames[Options[GraphDistance]];
   If [StringName[weighted] =!= "False" && StringName[weighted] =!= "True",
      Message[GraphDistance::wgt, weighted];
      Return[$Failed];
   ];
	If [StringName[weighted] === "False",
      res = SparseArray`GraphDistance[A, i, j];
      If [res < 0, res = Infinity],
      res = SparseArray`ShortestPath[A, i, j];
      If [!ListQ[res], Throw[$Failed]];
      res = ShortestPathMessage[GraphDistance, res];
      If [ListQ[res], res = res[[1]]/.$MaxMachineNumber->Infinity];
   ];
   res
];

MaximalBipartiteMatching[{}]:= {};
MaximalBipartiteMatching[_?((GraphQ[#] && VertexCount[#] === 0)&)] := {}
MaximalBipartiteMatching[Combinatorica`Graph[{},{}]]:= {};

MaximalBipartiteMatching[G_?GraphQ] := With[
      {res = MaximalBipartiteMatchingInternal[G2S[G]],vtx=VertexList[G]},
      (res/.Thread[Range[Length[vtx]]->vtx]) /;(res =!= $Failed)
];
MaximalBipartiteMatching[G_?RuleListGraphQ]:= With[
      {res = MaximalBipartiteMatchingInternal[G2S[G]],vtx=VertexList[G]},
      (res/.Thread[Range[Length[vtx]]->vtx]) /;(res =!= $Failed)
];

MaximalBipartiteMatching[G_Combinatorica`Graph]:= With[
      {res = MaximalBipartiteMatchingInternal[G2S[G]]},
      res /; (res =!= $Failed)
];

MaximalBipartiteMatching[A_?MatrixQ]:= With[
      {res = MaximalBipartiteMatchingInternal[A]},
      res /; (res =!= $Failed)
];

(e:MaximalBipartiteMatching[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[MaximalBipartiteMatching, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

MaximalBipartiteMatchingInternal[A_?MatrixQ]:= Module[
   {res, n = Dimensions[A][[1]]},
   res = SparseArray`MaximalBipartiteMatching[A];
   If [!List[res], Return[$Failed]];
   res
];

getUndirectedGraph[caller_, G_?GraphQ] :=
    getUndirectedGraphInternal[caller, G]

getUndirectedGraph[caller_, G_?RuleListGraphQ]:= Module[{},
   getUndirectedGraphInternal[caller, G]
];

getUndirectedGraph[caller_, G_Combinatorica`Graph]:= 
  Module[{},
   getUndirectedGraphInternal[caller, G]
];

getUndirectedGraph[caller_, A_?MatrixQ]:= 
  Module[{},
   getUndirectedGraphInternal[caller, A]
];

getUndirectedGraphInternal[caller_, G_]:= Module[{A, lst, m, n},
   (* check that the matrix from data is symmetric,
      if not, convert to symmetric by A+Transpose[A] *)
	A = G2S[G];
   If [!MatrixQ[A], Throw[$Failed]];
   {m, n} = Dimensions[A];
	If [m != n,
       Message[caller::rug, G];
       Throw[$Failed]
   ];
   If [!StructurallySymmetricMatrixQ[A],
      A = A + Transpose[A];
   ];
   A
];


checkGraph[caller_, G_?GraphQ] := checkGraphInternal[caller, G]

checkGraph[caller_, G_?RuleListGraphQ]:= Module[{},
   checkGraphInternal[caller, G]
];

checkGraph[caller_, G_Combinatorica`Graph]:= 
  Module[{},
   checkGraphInternal[caller, G]
];

checkGraph[caller_, A_?MatrixQ]:= 
  Module[{},
   checkGraphInternal[caller, A]
];

checkGraphInternal[caller_, G_]:= Module[{A, lst, m, n},
   (* check that the matrix from data is square
       *)
	A = G2S[G];
	If[!MatrixQ[A], Throw[$Failed]];
   {m, n} = Dimensions[A];
	If [m != n,
       Message[caller::grp, G];
       Throw[$Failed]
   ];
   A
];

MaximalIndependentVertexSet[{},___]:={};
MaximalIndependentVertexSet[_?((GraphQ[#] && VertexCount[#] === 0)&),___] := {}
MaximalIndependentVertexSet[Combinatorica`Graph[{},{}],___]:={};


MaximalIndependentVertexSet[G_?GraphQ] := With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, G], Automatic]], vtx = VertexList[G]},
   vtx[[res]] /; (res =!= $Failed)
]
MaximalIndependentVertexSet[G_?RuleListGraphQ]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, G], Automatic]], vtx = VertexList[G]},
   vtx[[res]] /; (res =!= $Failed)
];
MaximalIndependentVertexSet[G_Combinatorica`Graph]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, G], Automatic]]},
   res /; (res =!= $Failed)
];
MaximalIndependentVertexSet[G_?GraphQ, vtxwgt_]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, G], vtxwgt]], vtx = VertexList[G]},
   vtx[[res]] /; (res =!= $Failed)
];
MaximalIndependentVertexSet[G_?RuleListGraphQ, vtxwgt_]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, G], vtxwgt]], vtx = VertexList[G]},
   vtx[[res]] /; (res =!= $Failed)
];
MaximalIndependentVertexSet[G_Combinatorica`Graph, vtxwgt_]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, G], vtxwgt]]},
   res /; (res =!= $Failed)
];

MaximalIndependentVertexSet[A_?MatrixQ]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, A], Automatic]]},
   res /; (res =!= $Failed)
];
MaximalIndependentVertexSet[A_?MatrixQ, vtxwgt_]:= With[
   {res = Catch[MaximalIndependentVertexSetInternal[getUndirectedGraph[MaximalIndependentVertexSet, A], vtxwgt]]},
   res /; (res =!= $Failed)
];

(e:MaximalIndependentVertexSet[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[MaximalIndependentVertexSet, argLength[e], 1, 2]},
  res/;(res =!= $Failed)
];

MaximalIndependentVertexSetInternal[A_?MatrixQ, vtxwgt_]:= Module[
   {n=Dimensions[A][[1]], dims = Dimensions[vtxwgt]},

   (* this should already be checked *)
   If [!StructurallySymmetricMatrixQ[A],
      Return[$Failed]
   ];
   If [vtxwgt =!= Automatic && (Length[dims] =!= 1 ||
         dims[[1]] =!= n),
      Message[MaximalIndependentVertexSet::vtxwgt, vtxwgt];
      Return[$Failed]
   ];
   SparseArray`MaximalIndependentVertexSet[A, vtxwgt]
];

MaximalIndependentEdgeSet[{},___]:={};
MaximalIndependentEdgeSet[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={}
MaximalIndependentEdgeSet[Combinatorica`Graph[{},{}],___]:={};

MaximalIndependentEdgeSet[G_?GraphQ, opts___?OptionQ] := With[
   {res = Catch[MaximalIndependentEdgeSetInternal[getUndirectedGraph[MaximalIndependentEdgeSet, G], opts, NondefaultOptions[MaximalIndependentEdgeSet]]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
]
MaximalIndependentEdgeSet[G_?RuleListGraphQ, opts___?OptionQ]:= With[
   {res = Catch[MaximalIndependentEdgeSetInternal[getUndirectedGraph[MaximalIndependentEdgeSet, G], opts, NondefaultOptions[MaximalIndependentEdgeSet]]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
MaximalIndependentEdgeSet[G_Combinatorica`Graph, opts___?OptionQ]:= With[
   {res = Catch[MaximalIndependentEdgeSetInternal[getUndirectedGraph[MaximalIndependentEdgeSet, G], opts, NondefaultOptions[MaximalIndependentEdgeSet]]]},
   res /; (res =!= $Failed)
];
MaximalIndependentEdgeSet[A_?MatrixQ, opts___?OptionQ]:= With[
   {res = Catch[MaximalIndependentEdgeSetInternal[getUndirectedGraph[MaximalIndependentEdgeSet, A], opts, NondefaultOptions[MaximalIndependentEdgeSet]]]},
   res /; (res =!= $Failed)
];

(e:MaximalIndependentEdgeSet[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[MaximalIndependentEdgeSet, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

MaximalIndependentEdgeSetInternal[A_?MatrixQ, opts___?OptionQ]:= Module[
   {weighted},
   (* this should be already checked with getUndirectedGraph*)
	If [!StructurallySymmetricMatrixQ[A],
      Return[$Failed]
   ];

   weighted = 
        "Weighted" /. processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[MaximalIndependentEdgeSet]];

   If [StringName[weighted] =!= "True" && StringName[weighted] =!= "False",
      Message[MaximalIndependentEdgeSet::wgt, weighted];
      Return[$Failed]
   ];
   If [StringName[weighted] === "True",
  	   If [!SymmetricMatrixQ[A],
         Message[MaximalIndependentEdgeSet::symat, A];
         Return[$Failed]
      ]
   ];

   SparseArray`MaximalMatching[A, "Weighted"->weighted]
];


MinCut[G_?((GraphQ[#] && VertexCount[#] > 0) &), k_Integer] := With[
   {res = Catch[MinCutInternal[getUndirectedGraph[MinCut, G], k]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
MinCut[G_?RuleListGraphQ, k_Integer]:= With[
   {res = Catch[MinCutInternal[getUndirectedGraph[MinCut, G], k]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
MinCut[G_Combinatorica`Graph, k_Integer]:= With[
   {res = Catch[MinCutInternal[getUndirectedGraph[MinCut, G], k]]},
   res /; (res =!= $Failed)
];
MinCut[A_?MatrixQ, k_Integer]:= With[
   {res = Catch[MinCutInternal[getUndirectedGraph[MinCut, A], k]]},
   res /; (res =!= $Failed)
];

(e:MinCut[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[MinCut, argLength[e], 2, 2]},
  res/;(res =!= $Failed)
];

MinCutInternal[A_?MatrixQ, k_Integer]:= Module[
   {},

   (* this should already be checked by getUndirectedGraph*)
	If [!StructurallySymmetricMatrixQ[A],
      Return[$Failed]
   ];
   If [k < 1,
      Message[MinCut::kgtwo, k];
      Return[$Failed]
   ];
   If [k == 1, Return[{Table[i,{i,Dimensions[A][[1]]}]}]];
   SparseArray`MinCut[A, k]
];

CommunityStructurePartition[{},___]:= {};
CommunityStructurePartition[_?((GraphQ[#] && VertexCount[#] === 0)&), ___] := {};
CommunityStructurePartition[Combinatorica`Graph[{},{}],___]:= {};
CommunityStructurePartition[G_?GraphQ, opts___?OptionQ] := With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructurePartition, checkGraph[CommunityStructurePartition, G], opts, NondefaultOptions[CommunityStructurePartition]]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
CommunityStructurePartition[G_?RuleListGraphQ, opts___?OptionQ]:= With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructurePartition, checkGraph[CommunityStructurePartition, G], opts, NondefaultOptions[CommunityStructurePartition]]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];


CommunityStructurePartition[G_Combinatorica`Graph, opts___?OptionQ]:= With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructurePartition, checkGraph[CommunityStructurePartition, G], opts, NondefaultOptions[CommunityStructurePartition]]]},
   res /; (res =!= $Failed)
];

CommunityStructurePartition[A_?MatrixQ, opts___?OptionQ]:= With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructurePartition, checkGraph[CommunityStructurePartition, A], opts, NondefaultOptions[CommunityStructurePartition]]]},
   res /; (res =!= $Failed)
];

(e:CommunityStructurePartition[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[CommunityStructurePartition, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

CommunityStructureAssignment[{},___]:={};
CommunityStructureAssignment[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
CommunityStructureAssignment[Combinatorica`Graph[{},{}],___]:={};

CommunityStructureAssignment[G_?GraphQ, opts___?OptionQ] := With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructureAssignment, checkGraph[CommunityStructureAssignment, G], opts, NondefaultOptions[CommunityStructureAssignment]]]},
   res /; (res =!= $Failed)
];

CommunityStructureAssignment[G_?RuleListGraphQ, opts___?OptionQ]:= With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructureAssignment, checkGraph[CommunityStructureAssignment, G], opts, NondefaultOptions[CommunityStructureAssignment]]]},
   res /; (res =!= $Failed)
];


CommunityStructureAssignment[G_Combinatorica`Graph, opts___?OptionQ]:= With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructureAssignment, checkGraph[CommunityStructureAssignment, G], opts, NondefaultOptions[CommunityStructureAssignment]]]},
   res /; (res =!= $Failed)
];

CommunityStructureAssignment[A_?MatrixQ, opts___?OptionQ]:= With[
   {res = Catch[CommunityStructurePartitionInternal[CommunityStructureAssignment, checkGraph[CommunityStructureAssignment, A], opts, NondefaultOptions[CommunityStructureAssignment]]]},
   res /; (res =!= $Failed)
];

(e:CommunityStructureAssignment[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[CommunityStructureAssignment, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

NumericNonnegativeMatrixQ[A_]:= MatrixQ[A, (NumericQ[#] && # >= 0)&];

CommunityStructurePartitionInternal[caller_, A_?MatrixQ, opts___?OptionQ]:= Module[
   {weighted, res},
   weighted = 
        "Weighted" /. processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[CommunityStructurePartition]];

   If [StringName[weighted] =!= "True" && StringName[weighted] =!= "False",
      Message[caller::wgt, weighted];
      Return[$Failed]
   ];

   If [StringName[weighted] === "True",
      If [!NumericNonnegativeMatrixQ[A],
         Message[caller::wgtm, A];
         Return[$Failed];
      ];
   ];
   If [caller === CommunityStructurePartition,
     res = SparseArray`CommunityStructurePartition[CommunityStructurePartition, A, opts],
     res = SparseArray`CommunityStructureAssignment[CommunityStructureAssignment, A, opts]
   ];
   If [!ListQ[res], $Failed, res]
];

GraphPath[{},___]:={};
GraphPath[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
GraphPath[Combinatorica`Graph[{},{}],___]:={};

GraphPath[G_?GraphQ, i_, j_, opts___?OptionQ] := With[
   {vtx = VertexList[G], res = Catch[GraphPathInternal[checkGraph[GraphPath, G], i, j, VertexList[G], opts, NondefaultOptions[GraphPath]]]},
   If [j =!= All,
     Map[vtx[[#]]&, res],
     {Round[res[[1]]], Map[vtx[[#]]&, res[[2]]]}
   ] /; (res =!= $Failed)
];

GraphPath[G_?RuleListGraphQ, i_, j_, opts___?OptionQ]:= With[
   {vtx = VertexList[G], res = Catch[GraphPathInternal[checkGraph[GraphPath, G], i, j, VertexList[G], opts, NondefaultOptions[GraphPath]]]},
   If [j =!= All,
     Map[vtx[[#]]&, res],
     {Round[res[[1]]], Map[vtx[[#]]&, res[[2]]]}
   ] /; (res =!= $Failed)
];

GraphPath[G_Combinatorica`Graph, i_, j_, opts___?OptionQ]:= With[
   {res = Catch[GraphPathInternal[checkGraph[GraphPath, G], i, j, None, opts, NondefaultOptions[GraphPath]]]},
   res /; (res =!= $Failed)
];

GraphPath[A_?MatrixQ, i_, j_, opts___?OptionQ]:= With[
   {res = Catch[GraphPathInternal[checkGraph[GraphPath, A], i, j, None, opts, NondefaultOptions[GraphPath]]]},
   res /; (res =!= $Failed)
];

(e:GraphPath[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[GraphPath, argLength[e], 2, 3]},
  res/;(res =!= $Failed)
];

ShortestPathMessage[caller_, res0_]:= Module[
  {success, res},


   {success, res} = res0;
   Switch[success,
    "NegativeCycle",
       Message[caller::negc];
       res = $Failed,
    "NegativeWeight",
       Message[caller::negw];
       res = $Failed,
    "WrongWeightedOption",
      Message[caller::wgt, res];
      res = $Failed,
    "WrongMethodOption",
      Message[caller::wmthd, res];
      res = $Failed,
    "NotSquareMatrix",
      Message[caller::sqma, res];
      res = $Failed,
    "SecondArgWrong",
      (* this should have been caught before *)
      res = $Failed,
    "ThirdArgWrong",
      (* this should have been caught before *)
      res = $Failed,
    "AssumeWeightOne",
      Message[caller::wgh1],
     _,
       nothing
   ];
   res
]

GraphPathInternal[A_?MatrixQ, i0_, j0_, vtxlst_, opts___?OptionQ]:= Module[
   {mthd, vtxrule, n = Dimensions[A][[1]], i = i0, j = j0, 
    res, dist, predecessors, success},

   (* rule list specified graph, convert vertex name to indices *)
   If [vtxlst =!= None,
      vtxrule = Thread[Rule[vtxlst, Range[n]]];
      i = i/.vtxrule;
      j = j/.vtxrule;
   ];

   If [i > n || i <= 0, Return[$Failed]];
   If [j =!= All && (j > n || j <= 0), Return[$Failed]];

   (* if people use Combinatorica, they can do
     Algorithm->.., which is the same as our Method->.. *)
   mthd = "Algorithm"/.processOptionNames[Flatten[{opts}]]
         /.{"Algorithm"->Automatic};

   If [j =!= All,
      res = SparseArray`ShortestPath[A, i, j, opts, Method->mthd],
      res = SparseArray`ShortestPath[A, i, opts, Method->mthd]
   ];
   If [!ListQ[res], Throw[$Failed]];
   res = ShortestPathMessage[GraphPath, res];
  (* res = {dist,path}. replace $MaxMachineNumber by Infinity *)
  If [ListQ[res], res = res[[2]]/.$MaxMachineNumber->Infinity];
   res

];

u = Unprotect[GraphDistanceMatrix];
GraphDistanceMatrix[{},___]:= {{},{}};
GraphDistanceMatrix[Combinatorica`Graph[{},{}],___]:= {{},{}};
GraphDistanceMatrix[G_?RuleListGraphQ, opts___?OptionQ]:= With[
   {res = Catch[GraphDistanceMatrixInternal[checkGraph[GraphDistanceMatrix, G], False, opts, NondefaultOptions[GraphDistanceMatrix]]]},
   Round[res] /; (res =!= $Failed)
];

GraphDistanceMatrix[G_?RuleListGraphQ, Combinatorica`Parent, opts___?OptionQ]:= With[
   {res = Catch[GraphDistanceMatrixInternal[checkGraph[GraphDistanceMatrix, G], True, opts, NondefaultOptions[GraphDistanceMatrix]]], vtx = VertexList[G]},
   {Round[res[[1]]], Map[vtx[[#]]&, res[[2]],{2}]} /; (res =!= $Failed)
];

GraphDistanceMatrix[G_Combinatorica`Graph, opts___?OptionQ]:= With[
   {res = Catch[GraphDistanceMatrixInternal[checkGraph[GraphDistanceMatrix, G], False, opts, NondefaultOptions[GraphDistanceMatrix]]]},
   res /; (res =!= $Failed)
];
GraphDistanceMatrix[G_Combinatorica`Graph, Combinatorica`Parent, opts___?OptionQ]:= With[
   {res = Catch[GraphDistanceMatrixInternal[checkGraph[GraphDistanceMatrix, G], True, opts, NondefaultOptions[GraphDistanceMatrix]]]},
   res /; (res =!= $Failed)
];

GraphDistanceMatrix[A_?MatrixQ, opts___?OptionQ]:= With[
   {res = Catch[GraphDistanceMatrixInternal[checkGraph[GraphDistanceMatrix, A], False, opts, NondefaultOptions[GraphDistanceMatrix]]]},
   res /; (res =!= $Failed)
];
GraphDistanceMatrix[A_?MatrixQ, Combinatorica`Parent, opts___?OptionQ]:= With[
   {res = Catch[GraphDistanceMatrixInternal[checkGraph[GraphDistanceMatrix, A], True, opts, NondefaultOptions[GraphDistanceMatrix]]]},
   res /; (res =!= $Failed)
];

Protect @@ u;

GraphDistanceMatrixInternal[A_?MatrixQ, parent_, opts___?OptionQ]:= Module[
   {success, res},

  res = SparseArray`AllPairsShortestPath[A, opts];
  If [!ListQ[res], Throw[$Failed]];
  {success, res} = res;

  Switch[success,
    "NegativeCycle",
       Message[GraphDistanceMatrix::negc];
       res = $Failed,
    "NegativeWeight",
       Message[GraphDistanceMatrix::negw];
       res = $Failed,
    "WrongWeightedOption",
      Message[GraphDistanceMatrix::wgt, res];
      res = $Failed,
    "WrongMethodOption",
      Message[GraphDistanceMatrix::wmthd, res];
      res = $Failed,
    "NotSquareMatrix",
      Message[GraphDistanceMatrix::sqma, res];
      res = $Failed,
    "AssumeWeightOne",
      (* need just distance matrix*)
      If [!parent, res = res[[1]]]; 
      Message[GraphDistanceMatrix::wgh1],
    _,
      (* need just distance matrix*)
      If [!parent, res = res[[1]]]; 
  ];

  If[res === $Failed, Throw[$Failed]];

  (* replace $MaxMachineNumber by Infinity *)
  If [ListQ[res], res = res/.$MaxMachineNumber->Infinity];
  If [MatrixQ[A,IntegerQ], 
    If [!parent, 
       res = Round[res], 
       res[[1]] = Round[res[[1]]]
    ]
  ];
  res
];


u = Unprotect[{ClosenessCentrality}];
ClosenessCentrality[{},___]:={};
ClosenessCentrality[Combinatorica`Graph[{},{}],___]:={};

ClosenessCentrality[G_?RuleListGraphQ, opts___?OptionQ]:= With[
   {res = Catch[ClosenessCentralityInternal[checkGraph[ClosenessCentrality, G], opts]]},
   res /; (res =!= $Failed)
];

ClosenessCentrality[G_Combinatorica`Graph, opts___?OptionQ]:= With[
   {res = Catch[ClosenessCentralityInternal[checkGraph[ClosenessCentrality, G], opts]]},
   res /; (res =!= $Failed)
];

ClosenessCentrality[A_?MatrixQ, opts___?OptionQ]:= With[
   {res = Catch[ClosenessCentralityInternal[checkGraph[ClosenessCentrality, A], opts]]},
   res /; (res =!= $Failed)
];

(e:ClosenessCentrality[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[ClosenessCentrality, argLength[e], 1, 2]},
  res/;(res =!= $Failed)
];

Protect @@ u;

ClosenessCentralityInternal[A_?MatrixQ, opts___?OptionQ]:= Module[
   {success, res},

  res = SparseArray`Centrality[A, opts];
  If [!ListQ[res], Throw[$Failed]];

  {success, res} = res;

  Switch[success,
    "NegativeCycle",
       Message[ClosenessCentrality::negc];
       res = $Failed, 
    "NegativeWeight",
       Message[ClosenessCentrality::negw];
       res = $Failed,
    "ZeroDistance",
       (*Centrality can not be calculated because the distance of one vertex to others sum to zero*)
       Message[ClosenessCentrality::zdis];
       res = $Failed,
   "AssumeWeightOne",
      (* need just distance matrix*)
      If [!parent, res = res[[1]]]; 
      Message[ClosenessCentrality::wgh1],
    _,
      (* need just distance matrix*)
      If [!parent, res = res[[1]]]; 
  ];

  (* replace $MaxMachineNumber by Infinity *)
  If [ListQ[res], res = res/.$MaxMachineNumber->Infinity];
  res
];

CommunityModularity[G_?((GraphQ[#] && VertexCount[#] > 0) &), part_, opts___?OptionQ] := With[
   {res = Catch[CommunityModularityInternal[checkGraph[CommunityModularity, G], part, VertexList[G], opts, NondefaultOptions[CommunityModularity]]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];

CommunityModularity[G_?RuleListGraphQ, part_, opts___?OptionQ]:= With[
   {res = Catch[CommunityModularityInternal[checkGraph[CommunityModularity, G], part, VertexList[G], opts, NondefaultOptions[CommunityModularity]]], vtx = VertexList[G]},
   Map[vtx[[#]]&, res] /; (res =!= $Failed)
];

CommunityModularity[G_Combinatorica`Graph, part_, opts___?OptionQ]:= With[
   {res = Catch[CommunityModularityInternal[checkGraph[CommunityModularity, G], part, VertexList[G], opts, NondefaultOptions[CommunityModularity]]]},
   res /; (res =!= $Failed)
];

CommunityModularity[A_?MatrixQ, part_, opts___?OptionQ]:= With[
   {res = Catch[CommunityModularityInternal[checkGraph[CommunityModularity, A], part, VertexList[A], opts, NondefaultOptions[CommunityModularity]]]},
   res /; (res =!= $Failed)
];

(e:CommunityModularity[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[CommunityModularity, argLength[e], 2, 2]},
  res/;(res =!= $Failed)
];

(* assignment is a vector of length n of the form
 {0,1,0,0,1,1}*)
isAssignment[v_, n_] := (VectorQ[v, IntegerQ] && Length[v] == n);

(* partition is a list of vectors, after
   flattening and sorting it should equal to
   Range[n] *)
isPartition[v_, n_] := Module[{},
  If[! VectorQ[v, VectorQ] || Length[v] > n, Return[False]];
  (Union[Map[VectorQ[#, IntegerQ] &, v]] === {True}) && (Sort[
      Union[Flatten[v]]] === Range[n])
  ];




CommunityModularityInternal[A_?MatrixQ, part0_, vtxlst_,
         opts___?OptionQ]:= Module[
   {weighted, n = Dimensions[A][[1]], part = part0},
   weighted = 
        "Weighted" /. processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[CommunityModularity]];

   If [StringName[weighted] =!= "True" && StringName[weighted] =!= "False",
      Message[CommunityModularity::wgt, weighted];
      Return[$Failed]
   ];

   If [StringName[weighted] === "True",
      If [Min[A] < 0,
         Message[CommunityModularity::wgtm, A];
         Return[$Failed];
      ];
   ];

   (* is this an assignment?*)
   If [!isAssignment[part,n],
      (* partition could be of the form {{a,b},{c,d}}, so
         we need to convert to {{1,2},{3,4}} etc *)
      part = part/.Thread[Rule[vtxlst, Range[n]]];
      If [!isPartition[part, n],
          Message[CommunityModularity::part, part0];
          Return[$Failed]
      ];
   ];
   SparseArray`CommunityModularity[A, part, opts]
];

PseudoDiameterToVertex[res_, vtxlist_]:= Module[
  (* convert results from a rule list, {{diam,{i,j}},...}
     to {{diam,{v_i, v_j}}, ...} *)
  {},
  Map[{#[[1]], vtxlist[[#[[2]]]]} &, res]
];

PseudoDiameter[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
PseudoDiameter[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

PseudoDiameter[A_?MatrixQ, opts___?OptionQ]:= With[
   {res = Catch[PseudoDiameterInternal[getUndirectedGraph[PseudoDiameter, A], opts, NondefaultOptions[PseudoDiameter]]]},
   res /; (res =!= $Failed)
];

PseudoDiameter[G_?GraphQ, opts___?OptionQ] := With[
   {res = Catch[PseudoDiameterInternal[getUndirectedGraph[PseudoDiameter, G], opts, NondefaultOptions[PseudoDiameter]]], vtx = VertexList[G]},
   PseudoDiameterToVertex[res, vtx] /; (res =!= $Failed)
];

PseudoDiameter[G_?RuleListGraphQ, opts___?OptionQ]:= With[
   {res = Catch[PseudoDiameterInternal[getUndirectedGraph[PseudoDiameter, G], opts, NondefaultOptions[PseudoDiameter]]], vtx = VertexList[G]},
   PseudoDiameterToVertex[res, vtx] /; (res =!= $Failed)
];

PseudoDiameter[G_Combinatorica`Graph, opts___?OptionQ]:= With[
   {res = Catch[PseudoDiameterInternal[getUndirectedGraph[PseudoDiameter, G], opts, NondefaultOptions[PseudoDiameter]]]},
   res /; (res =!= $Failed)
];

(e:PseudoDiameter[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[PseudoDiameter, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

PseudoDiameterInternal[A_, opts___?OptionQ]:= Module[
   {agg, res},

   (* this should already be checked by getUndirectedGraph *)
	If [!StructurallySymmetricMatrixQ[A],
      Return[$Failed]
   ];
   agg = 
        "Aggressive" /. processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[PseudoDiameter]];

   If [StringName[agg] =!= "True" && StringName[agg] =!= "False",
      Message[PseudoDiameter::agg, agg];
      Return[$Failed]
   ];
 
   res = SparseArray`PseudoDiameter[A, Aggressive->agg];

   (* converting from {{d, i, j},...} to {{d,{i,j}},...} *)
	If [!ListQ[res], Return[$Failed]];
   Map[{#[[1]], {#[[2]], #[[3]]}} &, res]
];
 


StrongComponents[{}] := {};
StrongComponents[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
StrongComponents[Combinatorica`Graph[{},{}]] := {};

StrongComponents[G_?GraphQ] := With[
      {res = StrongComponentsInternal[G2S[G]], vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];

StrongComponents[G_?RuleListGraphQ]:= With[
      {res = StrongComponentsInternal[G2S[G]], vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
StrongComponents[G_Combinatorica`Graph]:= With[
      {res = StrongComponentsInternal[G2S[G]]},
      res /; (res =!= $Failed)
];
StrongComponents[A_?MatrixQ]:= With[
      {res = StrongComponentsInternal[A]},
      res /; (res =!= $Failed)
];


(e:StrongComponents[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[StrongComponents, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

StrongComponentsInternal[A_?MatrixQ]:= Module[
   {res},
   If [!squareMatrixQ[A],
       Message[StrongComponents::sqma, A];
       Return[$Failed]
   ];
   SparseArray`StronglyConnectedComponents[A]
];


WeakComponents[{}] := {};
WeakComponents[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
WeakComponents[Combinatorica`Graph[{},{}]] := {};

WeakComponents[G_?GraphQ] := With[
      {res = WeakComponentsInternal[G2S[G]], vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];

WeakComponents[G_?RuleListGraphQ]:= With[
      {res = WeakComponentsInternal[G2S[G]], vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
WeakComponents[G_Combinatorica`Graph]:= With[
      {res = WeakComponentsInternal[G2S[G]]},
      res /; (res =!= $Failed)
];
WeakComponents[A_?MatrixQ]:= With[
      {res = WeakComponentsInternal[A]},
      res /; (res =!= $Failed)
];


(e:WeakComponents[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[WeakComponents, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];
WeakComponentsInternal[A_?MatrixQ]:= Module[
   {res},
   If [!squareMatrixQ[A],
       Message[WeakComponents::sqma, A];
       Return[$Failed]
   ];
   SparseArray`StronglyConnectedComponents[A+Transpose[A]]
];

Bicomponents[{}] := {};
Bicomponents[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
Bicomponents[Combinatorica`Graph[{},{}]] := {};

Bicomponents[G_?GraphQ] := With[
      {res = BicomponentsInternal[G2S[G]], vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];

Bicomponents[G_?RuleListGraphQ]:= With[
      {res = BicomponentsInternal[G2S[G]], vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
Bicomponents[G_Combinatorica`Graph]:= With[
      {res = BicomponentsInternal[G2S[G]]},
      res /; (res =!= $Failed)
];
Bicomponents[A_?MatrixQ]:= With[
      {res = BicomponentsInternal[A]},
      res /; (res =!= $Failed)
];
(e:Bicomponents[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[Bicomponents, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

BicomponentsInternal[A_?MatrixQ]:= Module[
   {res},
   If [!squareMatrixQ[A],
       Message[Bicomponents::sqma, A];
       Return[$Failed]
   ];
   SparseArray`BiconnectedComponents[A]
];

PageRanks[{}, opts___?OptionQ]:= {};
PageRanks[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
PageRanks[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

PageRanks[G_?GraphQ, opts___?OptionQ] := With[
      {res = PageRankInternal[PageRanks, G2S[G], "PageRank", opts, NondefaultOptions[PageRanks]], vtx = VertexList[G]},
      Thread[Rule[vtx, res]] /; (res =!= $Failed)
];
PageRanks[G_?RuleListGraphQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[PageRanks, G2S[G], "PageRank", opts, NondefaultOptions[PageRanks]], vtx = VertexList[G]},
      Thread[Rule[vtx, res]] /; (res =!= $Failed)
];
PageRanks[G_Combinatorica`Graph, opts___?OptionQ]:= With[
      {res = PageRankInternal[PageRanks, G2S[G], "PageRank", opts, NondefaultOptions[PageRanks]], vtx = VertexList[G]},
      Thread[Rule[vtx, res]] /; (res =!= $Failed)
];
PageRanks[A_?MatrixQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[PageRanks, A, "PageRank", opts, NondefaultOptions[PageRanks]], vtx = VertexList[A]},
      Thread[Rule[vtx, res]] /; (res =!= $Failed)
];

(e:PageRanks[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[PageRanks, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

PageRankVector[{}, opts___?OptionQ]:= {};
PageRankVector[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
PageRankVector[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

PageRankVector[G_?GraphQ, opts___?OptionQ] := With[
      {res = PageRankInternal[PageRankVector, G2S[G], "PageRank", opts, NondefaultOptions[PageRankVector]]},
      res /; (res =!= $Failed)
];
PageRankVector[G_?RuleListGraphQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[PageRankVector, G2S[G], "PageRank", opts, NondefaultOptions[PageRankVector]]},
      res /; (res =!= $Failed)
];
PageRankVector[G_Combinatorica`Graph, opts___?OptionQ]:= With[
      {res = PageRankInternal[PageRankVector, G2S[G], "PageRank", opts, NondefaultOptions[PageRankVector]]},
      res /; (res =!= $Failed)
];
PageRankVector[A_?MatrixQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[PageRankVector, A, "PageRank", opts, NondefaultOptions[PageRankVector]]},
      res /; (res =!= $Failed)
];

(e:PageRankVector[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[PageRankVector, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

LinkRanks[{}, opts___?OptionQ]:= {};
LinkRanks[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
LinkRanks[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

LinkRanks[G_?GraphQ, opts___?OptionQ] := With[
      {res = PageRankInternal[LinkRanks, G2S[G], "LinkRank", opts], vtx = VertexList[G]},
      Map[vtx[[#[[1]]]]->#[[2]]&, Drop[ArrayRules[res], -1]] /; (res =!= $Failed)
];
LinkRanks[G_?RuleListGraphQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[LinkRanks, G2S[G], "LinkRank", opts], vtx = VertexList[G]},
      Map[vtx[[#[[1]]]]->#[[2]]&, Drop[ArrayRules[res], -1]] /; (res =!= $Failed)
];
LinkRanks[G_Combinatorica`Graph, opts___?OptionQ]:= With[
      {res = PageRankInternal[LinkRanks, G2S[G], "LinkRank", opts], vtx = VertexList[G]},
      Map[vtx[[#[[1]]]]->#[[2]]&, Drop[ArrayRules[res], -1]] /; (res =!= $Failed)
];
LinkRanks[A_?MatrixQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[LinkRanks, A, "LinkRank", opts], vtx = VertexList[A]},
      Map[vtx[[#[[1]]]]->#[[2]]&, Drop[ArrayRules[res], -1]] /; (res =!= $Failed)
];

(e:LinkRanks[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[LinkRanks, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

LinkRankMatrix[{}, opts___?OptionQ]:= {};
LinkRankMatrix[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
LinkRankMatrix[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

LinkRankMatrix[G_?GraphQ, opts___?OptionQ] := With[
      {res = PageRankInternal[LinkRankMatrix, G2S[G], "LinkRank", opts]},
      res /; (res =!= $Failed)
];
LinkRankMatrix[G_?RuleListGraphQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[LinkRankMatrix, G2S[G], "LinkRank", opts]},
      res /; (res =!= $Failed)
];
LinkRankMatrix[G_Combinatorica`Graph, opts___?OptionQ]:= With[
      {res = PageRankInternal[LinkRankMatrix, G2S[G], "LinkRank", opts]},
      res /; (res =!= $Failed)
];
LinkRankMatrix[A_?MatrixQ, opts___?OptionQ]:= With[
      {res = PageRankInternal[LinkRankMatrix, A, "LinkRank", opts]},
      res /; (res =!= $Failed)
];
(e:LinkRankMatrix[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[LinkRankMatrix, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

PositiveMachineReal[x_]:= (NumericQ[x] && Head[x] =!= Complex && x > 0);

PageRankInternal[caller_, A_?MatrixQ, task_, opts___?OptionQ]:= Module[
   {res, tol, tel, rmsink},
   If [!squareMatrixQ[A],
       Message[caller::sqma, A];
       Return[$Failed]
   ];

	tol = "Tolerance"/.processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[caller]];
   tol = N[tol];

	If [StringName[tol] =!= "Automatic" && !PositiveMachineReal[tol],
       Message[caller::tol, tol];
       Return[$Failed]
   ];

	tel = "TeleportProbability"/.processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[caller]];

	If [!PositiveMachineReal[tel] || tel >= 1,
       Message[caller::tel, tel];
       Return[$Failed]
   ];

	rmsink = "RemoveSinks"/.processOptionNames[Flatten[{opts}]] /. 
          processOptionNames[Options[caller]];

	If [StringName[rmsink] =!= "True" && StringName[rmsink] =!= "False",
       Message[caller::rms, rmsink];
       Return[$Failed]
   ];

	Switch[task,
     "PageRank",
      SparseArray`PageRank[A, Tolerance->tol, TeleportProbability->tel,
          RemoveSinks->rmsink],
     "LinkRank",
      SparseArray`LinkRank[A, Tolerance->tol, TeleportProbability->tel,
          RemoveSinks->rmsink],
     _,
      Throw[$Failed] (* should not happen *)
   ]
   
];

MinimumBandwidthOrdering[{}, opts___?OptionQ]:= {};
MinimumBandwidthOrdering[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
MinimumBandwidthOrdering[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

MinimumBandwidthOrdering[G_?GraphQ, opts___?OptionQ] := With[
    {res = MinimumBandwidthOrderingInternal["GraphInput", AdjacencyMatrix[G],
                           opts, NondefaultOptions[MinimumBandwidthOrdering]]},
    res/;(res =!= $Failed && Head[res] =!= MinimumBandwidthOrdering)
];

MinimumBandwidthOrdering[G_?RuleListGraphQ, opts___?OptionQ]:= With[
      {res = MinimumBandwidthOrderingInternal["GraphInput", G2S[G],
         opts, NondefaultOptions[MinimumBandwidthOrdering]], 
         vtx = VertexList[G]},
      Map[vtx[[#]]&, res] /; (res =!= $Failed)
];
MinimumBandwidthOrdering[G_Combinatorica`Graph, opts___?OptionQ]:= With[
      {res = MinimumBandwidthOrderingInternal["GraphInput", G2S[G], opts, NondefaultOptions[MinimumBandwidthOrdering]]},
      res /; (res =!= $Failed)
];
MinimumBandwidthOrdering[A_?MatrixQ, opts___?OptionQ]:= With[
      {res = MinimumBandwidthOrderingInternal["MatrixInput", A, opts, NondefaultOptions[MinimumBandwidthOrdering]]},
      res /; (res =!= $Failed)
];
(e:MinimumBandwidthOrdering[___, opts___?OptionQ]):= With[
  {res = IssueArgsMessage[MinimumBandwidthOrdering, argLength[e], 1, 1]},
  res/;(res =!= $Failed)
];

MinimumBandwidthOrderingInternal[inputform_String, A0_?MatrixQ, opts___?OptionQ]:= Module[
   {res, refinement, mlv, A = A0},

   If [inputform === "GraphInput",
      (* if the input is a graph, we work with undirected graph
         (symmetric matrix) *)
       A = A + Transpose[A];
   ];

   refinement = "RefinementMethod"/.processOptionNames[Flatten[{opts}]] /. 
            processOptionNames[Options[MinimumBandwidthOrdering]];
   mlv = "RecursionMethod"/.processOptionNames[Flatten[{opts}]] /. 
            processOptionNames[Options[MinimumBandwidthOrdering]];
   res = SparseArray`MinimumBandwidthOrdering[A, "RefinementMethod"->refinement, "RecursionMethod"->mlv,
      Sequence@@filterOptions[SparseArray`MinimumBandwidthOrdering, Flatten[{opts}]]
   ];
   If [ListQ[res], 
        Switch [inputform,
          "MatrixInput", res[[2]], 
          "GraphInput", res[[2,1]],
          _, res[[2]] (* this should not happen *)
        ],
   $Failed]
];

SubgraphInternal[G_?GraphQ, vertices_List] :=
    Subgraph[G, VertexList[G][[vertices]]]

SubgraphInternal[G_?RuleListGraphQ, vertices_List]:= Module[
  {hash},
  hash = .;
  Clear[hash];
  hash[_]:= False;
  Map[(hash[#]=True)&,vertices];
  Select[G, (hash[#[[1]]] && hash[#[[2]]])&]

]

(* vertices is a list of vertex indices (integers) *)
SubgraphInternal[G_Combinatorica`Graph, vertices_List]:= Module[
  {res, vlabels = VertexList[G][[vertices]]},
  res = InduceSubgraph[G,vertices];
  Combinatorica`SetVertexLabels[res, vlabels[[Ordering[vertices]]]]
]


SubgraphInternal[A_?MatrixQ, vertices_List]:= Module[
  {vs, m, n},
  {m, n} = Dimensions[A];
  n = Min[m, n];
  vs = Select[vertices, ((IntegerQ[#] && 1<=#<=n)&)];
  A[[vs,vs]]
]


NeighborhoodVertices[{}, v_, opts___?OptionQ]:= {};
NeighborhoodVertices[{}, v_, n_, opts___?OptionQ]:= {};
NeighborhoodVertices[_?((GraphQ[#] && VertexCount[#] === 0)&),v_, opts___?OptionQ]:={};
NeighborhoodVertices[_?((GraphQ[#] && VertexCount[#] === 0)&),v_, n_, opts___?OptionQ]:={};
NeighborhoodVertices[Combinatorica`Graph[{},{}], v_, opts___?OptionQ]:= {};
NeighborhoodVertices[Combinatorica`Graph[{},{}], v_, n_, opts___?OptionQ]:= {};

NeighborhoodVertices2[G_?GraphQ, i_Integer, n_:Infinity, opts___?OptionQ]:= Module[
   {},
   NeighborhoodVerticesInternal[NeighborhoodVertices, checkGraph[NeighborhoodVertices, G], CheckVertex[NeighborhoodVertices, G, i, 2], n, opts, NondefaultOptions[NeighborhoodVertices]]
];
NeighborhoodVertices2[G_?RuleListGraphQ, i_, n_:Infinity, 
                    opts___?OptionQ]:= Module[{res},
    NeighborhoodVerticesInternal[NeighborhoodVertices, checkGraph[NeighborhoodVertices, G], CheckVertex[NeighborhoodVertices, G, i, 2], n, opts, NondefaultOptions[NeighborhoodVertices]]
     
];
NeighborhoodVertices2[G_Combinatorica`Graph, i_Integer, n_:Infinity, opts___?OptionQ]:= Module[
   {},
   NeighborhoodVerticesInternal[NeighborhoodVertices, checkGraph[NeighborhoodVertices, G], CheckVertex[NeighborhoodVertices, G, i, 2], n, opts, NondefaultOptions[NeighborhoodVertices]]
];

NeighborhoodVertices2[A_?MatrixQ, i_Integer, n_:Infinity, opts___?OptionQ]:= Module[
      {},
      NeighborhoodVerticesInternal[NeighborhoodVertices, checkGraph[NeighborhoodVertices, A], CheckVertex[NeighborhoodVertices, A, i, 2], n, opts, NondefaultOptions[NeighborhoodVertices]]
];

(e:NeighborhoodVertices[x___, opts___?OptionQ])/;checkopt[e, 2, 3]:= With[
  {res = Catch[NeighborhoodVertices2[x, opts]]},
  VertexList[{x}[[1]]][[res]]/;(res =!= $Failed && ListQ[res])
];

NeighborhoodVerticesInternal[caller_, A0_?MatrixQ, i_Integer, n_, opts___?OptionQ]:= Module[
   {v, undirected, A = A0},
   If [!squareMatrixQ[A],
       Message[NeighborhoodVertices::sqma, A];
       Return[$Failed]
   ];
	undirected = "Undirected"/.processOptionNames[Flatten[{opts}]]
         /.processOptionNames[Options[caller]];
   Switch[StringName[undirected],
     "True", A = A + Transpose[A],
     "False", nothing,
     _, Message[caller::und]
   ];
   res = SparseArray`BreadthFirstSearch[A, i];
   pos = Position[res[[2]], _?(0 <= # <= n &)];
   If[pos === {}, Return[{}]];
   Take[res[[1]], Last[pos][[1]]]
];



(e:NeighborhoodSubgraph[x___, opts___?OptionQ])/;checkopt[e, 2, 3]:= With[
  {res = Catch[NeighborhoodSubgraph2[x, opts]]},
  res/;(res =!= $Failed && Head[res] =!= NeighborhoodSubgraph2)
];

NeighborhoodSubgraph2[G_?GraphQ, i_, n_:Infinity, opts___?OptionQ] :=
    NeighborhoodSubgraphInternal[G, i, n, opts, NondefaultOptions[NeighborhoodSubgraph]]

NeighborhoodSubgraph2[G_?RuleListGraphQ, i_, n_:Infinity, 
                    opts___?OptionQ]:= Module[{},
    NeighborhoodSubgraphInternal[G, i, n, opts, NondefaultOptions[NeighborhoodSubgraph]]
        
];
NeighborhoodSubgraph2[G_Combinatorica`Graph, i_Integer, n_:Infinity, opts___?OptionQ]:= Module[
   {},
   NeighborhoodSubgraphInternal[G, i, n, opts, NondefaultOptions[NeighborhoodSubgraph]]
];

NeighborhoodSubgraph2[A_?MatrixQ, i_Integer, n_:Infinity, opts___?OptionQ]:= Module[
      {},
      NeighborhoodSubgraphInternal[A, i, n, opts, NondefaultOptions[NeighborhoodSubgraph]]
];

NeighborhoodSubgraphInternal[G_, i0_, n_, opts___?OptionQ]:= Module[
   {res, A},
   A = checkGraph[NeighborhoodSubgraph, G];
	i = CheckVertex[NeighborhoodSubgraph, G, i0, 2];
   res = NeighborhoodVerticesInternal[NeighborhoodSubgraph, A, i, n, Sequence@@Flatten[{opts}]];
   If [!ListQ[res], Throw[$Failed]];
   res = VertexList[G][[res]];
   SubgraphInternal[G, res]
];

u = Unprotect[{FindHamiltonianCycle}];
FindHamiltonianCycle[{}, opts___?OptionQ]:= {};
FindHamiltonianCycle[Combinatorica`Graph[{},{}], opts___?OptionQ]:= {};

FindHamiltonianCycle2[G_?RuleListGraphQ, 
                    opts___?OptionQ]:= 
    FindHamiltonianCycleInternal[checkGraph[FindHamiltonianCycle, G], opts, NondefaultOptions[FindHamiltonianCycle]];

FindHamiltonianCycle2[G_Combinatorica`Graph, opts___?OptionQ]:= 
   FindHamiltonianCycleInternal[checkGraph[FindHamiltonianCycle, G], opts, NondefaultOptions[FindHamiltonianCycle]];

FindHamiltonianCycle2[A_?MatrixQ, opts___?OptionQ]:= 
      FindHamiltonianCycleInternal[checkGraph[FindHamiltonianCycle, A], opts, NondefaultOptions[FindHamiltonianCycle]];

(* not V8-style graph, but all others *)
(e:FindHamiltonianCycle[x_?((!GraphQ[#])&), opts___?OptionQ])/;checkopt[e, 1, 1]:= With[
  {res = Catch[FindHamiltonianCycle2[x, opts]]},
  VertexList[x][[res]]/;(res =!= $Failed && ListQ[res])
];
Protect @@ u;

FindHamiltonianCycleInternal[A_?MatrixQ, opts___?OptionQ]:= Module[
   {v, maxit, seed},
   If [!squareMatrixQ[A],
       Message[FindHamiltonianCycle::sqma, A];
       Return[$Failed]
   ];
	maxit = "MaxIterations"/.processOptionNames[Flatten[{opts}]]
         /.processOptionNames[Options[FindHamiltonianCycle]];
   If [maxit =!= "Automatic" && (!IntegerQ[maxit] || maxit <= 0),
      Message[FindHamiltonianCycle::maxit, maxit];
      maxit = 100;
   ];

	seed = "RandomSeed"/.processOptionNames[Flatten[{opts}]]
         /.processOptionNames[Options[FindHamiltonianCycle]];
   If [seed =!= "Automatic" && !IntegerQ[maxit],
      Message[FindHamiltonianCycle::seed, seed];
      seed = 0;
   ];


   res = SparseArray`FindHamiltonianCycle[A, RandomSeed->seed, MaxIterations->maxit];
	If [ListQ[res], res, {}]
];




HamiltonianCycles[{}, n___, opts___?OptionQ]:= {};
HamiltonianCycles[_?((GraphQ[#] && VertexCount[#] === 0)&),___]:={};
HamiltonianCycles[Combinatorica`Graph[{},{}], n___, opts___?OptionQ]:= {};

HamiltonianCycle2[G_?GraphQ, n_:1, opts___?OptionQ] :=
    HamiltonianCycleInternal[checkGraph[HamiltonianCycles, G], n, opts, NondefaultOptions[HamiltonianCycles]];

HamiltonianCycle2[G_?RuleListGraphQ, n_:1,
                    opts___?OptionQ]:= 
    HamiltonianCycleInternal[checkGraph[HamiltonianCycles, G], n, opts, NondefaultOptions[HamiltonianCycles]];

HamiltonianCycle2[G_Combinatorica`Graph, n_:1, opts___?OptionQ]:= 
   HamiltonianCycleInternal[checkGraph[HamiltonianCycles, G], n, opts, NondefaultOptions[HamiltonianCycles]];

HamiltonianCycle2[A_?MatrixQ, n_:1, opts___?OptionQ]:= 
      HamiltonianCycleInternal[checkGraph[HamiltonianCycles, A], n, opts, NondefaultOptions[HamiltonianCycles]];

(e:HamiltonianCycles[x_, n___, opts___?OptionQ])/;checkopt[e, 1, 2]:= With[
  {res = Catch[HamiltonianCycle2[x, n, opts]]},
  Map[(VertexList[x][[#]])&,res]/;(res =!= $Failed && ListQ[res])


];

HamiltonianCycleInternal[A_?MatrixQ, n_:1, opts___?OptionQ]:= Module[
   {v},
   If [!squareMatrixQ[A],
       Message[HamiltonianCycles::sqma, A];
       Return[$Failed]
   ];

	If [n =!= All && (!IntegerQ[n] || n <= 0),
      Message[HamiltonianCycles::nham, n];
   ];

   res = SparseArray`HamiltonianCycle[A, n];
	If [ListQ[res], res, {}]
];

CheckVertex[caller_, g_, vertex_, argnum_]:= Module[{vtxnames},
   vtxnames = VertexList[g];
   If [!MemberQ[vtxnames, vertex], Message[caller::rind, vertex, argnum]; 
        Throw[$Failed]];
   Position[vtxnames, vertex][[1,1]]
];

machineRealNumberQ[rp_]:= ((MachineNumberQ[rp] || (IntegerQ[rp] && rp >= -$MaxMachineNumber && rp <= $MaxMachineNumber))
	    && Head[rp] =!= Complex);

LineScaledCoordinate[coord_?MatrixQ]:= LineScaledCoordinate[coord, 0.5];

LineScaledCoordinate[coord_?MatrixQ, rr0_]/;machineRealNumberQ[rr0] := Module[
      {dist, dist2, sta, sto, newpos, total, rr=rr0},
      If [coord === {}, Return[{}]];
      If [Length[coord] == 1, Return[coord[[1]]]];
      If [rr > 1, rr = 1];
      If [rr < 0, rr = 0];
      dist = Map[Norm, Drop[coord - RotateLeft[coord], -1]];
      total = Total[dist];
      If [total <= $MachineEpsilon, Return[coord[[1]]]];
      dist /= total;
      dist2 = FoldList[Plus, 0, dist];(* distance from start *)
      dist2[[Length[dist2]]] = 1.; (* make sure the last one is 1 *)
      sto = Flatten[Position[dist2, (_?(# >= rr &))]][[
        1]];
      If [sto == 1, Return[coord[[1]]]];
      sta = sto - 1;
      newpos = 
       coord[[sta]] + (rr - dist2[[sta]])/(dist2[[sto]] - dist2[[sta]])(coord[[
            sto]] - coord[[sta]]);
      newpos
    ];



(*============ ExpressionTreePlot ============= *)
ExpressionTreePlot[]=Graphics[];

(ee:ExpressionTreePlot[e_, opts___?OptionQ])/;checkopt[ee, 0, 3] :=
 With[{res = Catch[Network`GraphPlot`ExprTreePlot[e,Automatic,Infinity,StandardForm, opts]]},
  res/;res =!= $Failed
];
 
(ee:ExpressionTreePlot[e_, orientation_, opts___?OptionQ])/;checkopt[ee, 0, 3] := With[{res = Catch[Network`GraphPlot`ExprTreePlot[e,orientation,Infinity,StandardForm, opts]]},
   res/;res =!= $Failed
];
 
(ee:ExpressionTreePlot[e_, orientation_, lev_, opts___?OptionQ])/;checkopt[ee, 0, 3] := With[{res = Catch[Network`GraphPlot`ExprTreePlot[e,orientation,lev,StandardForm, opts]]},
   res/;res =!= $Failed
];
 

(*============ end ExpressionTreePlot =============== *)


(*============== begin  GraphEdit ============== *)

Needs[ "GUIKit`"]


{XSIZE, YSIZE} = {600, 600};

GraphEdit[d_?GraphQ, opts___?OptionQ] :=
    GraphEdit[EdgeList[d]/.{UndirectedEdge[{a_,b_},___] :> (a->b),
                            DirectedEdge[{a_,b_},___] :> (a->b)},
              opts];

GraphEdit[d_Combinatorica`Graph, opts___?OptionQ] :=
	Module[ {},
      $VertexNames = Map[ToString,VertexList[d]];
		GraphEditImpl[d, {}, opts]
	];


GraphEdit[d_?MatrixQ, opts___?OptionQ] :=
	Module[ {rules},
      GraphEdit[SparseArray[d]]
	];

GraphEdit[d_?SparseArrayGraphQ, opts___?OptionQ] :=
	Module[ {rules},
      rules = Map[(Rule@@#)&,EdgeList[d]];
      $VertexNames = Map[ToString,VertexList[d]];
		GraphEditImpl[rules, {}, opts]
	];

GraphEdit[d_?RuleListQ, opts___?OptionQ] :=
	Module[ {rules},
      rules = Map[(Rule@@#)&,EdgeList[d]];
      $VertexNames = Map[ToString,VertexList[d]];
		GraphEditImpl[rules, {}, opts]
	];

GraphEdit[opts___?OptionQ] := Module[
 {},
 $VertexNames = {};
 GraphEditImpl[{}, {}, opts]
];


GraphEditImpl[ d_, opts___?OptionQ] :=
  Module[{edges = {}, xSize, ySize, GUI, mlv = True,vnames,pts,verts},
    $opts = opts;
    { xSize, ySize} = {XSIZE, YSIZE};
    {pts, edges} = PointsAndEdges[d];
    pts = ScalePoints[pts, {xSize, ySize}];
    $GraphInitial = {pts, edges};
    {pts, verts, vnames} = GUIRunModal[GUINew];
    {Graphics[ {RGBColor[0.5, 0., 0.],
    	Map[ Line[ Part[pts, #]]&, verts],
    	RGBColor[0., 0., 0.7], PointSize[0.02], Map[ Point, pts]}, AspectRatio->Automatic],
      "Graph"->Apply[Rule, verts, {1}], "Coordinates"->pts, "VertexLabels"->vnames
     }
    ];


SparseArrayGraphQ[ x_] := (Head[x] === SparseArray && ArrayDepth[x] === 2);
SparseArrayGraphQ[___] := False;


RuleListQ[a_] := MatchQ[a, {(_ -> _) ...}];
RuleListQ[___] := False;


PointsAndEdges[d0_] :=
  Module[{d=d0,verts, list, hash, pts, ef},
    If [Head[d]===Combinatorica`Graph,
      (* for combinatorica, uses original coordinates *)
      pts = GraphCoordinates[d, Method->None, $opts];
      d = Map[(Rule@@#)&,EdgeList[d]];
      pts=pts[[VertexList[d]]],
      pts = GraphCoordinates[d, $opts];
    ];
    verts = VertexList[d];
    list = Apply[List, d, {1}];
    MapIndexed[(hash[#1] = First[#2]) &, verts];
    ef = {pts, Map[hash, list, {2}]};
    Clear[hash];
    ef
    
];

ScalePoints[pts_, {maxX_, maxY_}] :=
  Module[ {xP, yP, xd, yd},
    xD = maxX/25.;
    yD = maxY/25.;
    xP = Part[pts, All, 1];
    yP = Part[pts, All, 2];
    xP = Rescale[xP, {Min[xP], Max[xP]}, {xD, maxX - xD}];
    yP = Rescale[yP, {Min[yP], Max[yP]}, {yD, maxY - yD}];
	 Floor[Transpose[{xP, yP}]]
    ];


GraphEdit`Private`getInitialVertices[] :=
		If[MatchQ[$GraphInitial, 
    	{{{_Integer, _Integer} ..}, {{_Integer, _Integer} ..}}], 
    		First[$GraphInitial], {}];

GraphEdit`Private`getInitialEdges[] :=
    If[MatchQ[$GraphInitial, {{{_Integer, _Integer} ..}, 
    	{{_Integer, _Integer} ..}}], Last[$GraphInitial], {}];









(*================= GraphGUINew.m ============== *)
GUINew := Widget["Frame",{Widget["Panel", {
      {
        {WidgetGroup[{
              Widget["RadioButton", {"text" -> "Add", 
                  "selected" -> True, 
                  BindEvent["action", Script[addButton[]]]},
                	Name -> "addButton"],
              Widget["RadioButton", {"text" -> "Move", 
                  "selected" -> False,
                  BindEvent["action", Script[moveButton[]]]},
                Name -> "moveButton"],
              Widget["RadioButton", {"text" -> "Delete",
                   "selected" -> False,
                  BindEvent["action", Script[deleteButton[]]]},
                Name -> "deleteButton"],
              Widget["ButtonGroup", {
                  WidgetReference["addButton"],
                  WidgetReference["moveButton"],
                  WidgetReference["deleteButton"]}],
              Widget["Button", {"text" -> "Delete All",
                  BindEvent["action", Script[deleteAllButton[]]]}]
              }, WidgetLayout -> {"Border" -> "Drawing mode"}]},
        
        {WidgetGroup[{
			  Widget["ComboBox", {
                  "items" -> 
                    Script[{"None", "Name", "Number", "Text"}],
              	BindEvent["action", Script[showVertexLabels[]]]
                  }, Name -> "setVertexLabel"]},WidgetLayout -> {"Border" -> "Vertex Labels"}]}, 
        {WidgetGroup[{
			  Widget["ComboBox", {
                  "items" -> 
                    Script[{"None", "Number", "Text"}],
              	BindEvent["action", Script[showEdgeLabels[]]]
                  }, Name -> "setEdgeLabel"]},WidgetLayout -> {"Border" -> "Edge Labels"}]}, 
			
        {
          WidgetGroup[{
              Widget["ComboBox", {
                  
                  "items" -> 
                    Script[{"Automatic", "Spring-Electrical", "Spring", 
                        "High-D Embedding",  "Tree Drawing", "Layered Drawing",   
                        "Radial Drawing", "Randomize"}],
              	BindEvent["action", Script[redrawButton[False]], InvokeThread->"New"]
                  }, Name -> "methodBox"] ,

               Widget["CheckBox", {"text" -> "Multilevel",  "selected" -> True,
                  BindEvent["action", Script[setMultilevel[]]]},
                	Name -> "mlvcheckbox"],

              Widget["Button", {"text" -> "Rescale",
                  	BindEvent["action", Script[rescaleButton[]]]}],
              Widget["Button", {"text" -> "Redraw",
                  	BindEvent["action", Script[redrawButton[False]]]
                  }],
              Widget["Button", {"text" -> "Random",
                  	BindEvent["action", Script[randomButton[]]]
                  }],
              Widget["Button", {"text" -> "Animate",
                  	BindEvent["action", Script[animateButton[]],
						InvokeThread->"New"]
                  }],
              Widget["Button", {"text" -> "Undo",
                  	"enabled" -> False,
                  	BindEvent["action", Script[undoButton[]]]
                  }, Name -> "undobutton"],
              Widget["Button", {"text" -> "Done",
                  	"enabled" -> True,
                  	BindEvent["action", Script[doneButton[]]]
                  }, Name -> "donebutton"]
              }, WidgetLayout -> {"Border" -> "Layout"}]}        
        },      
      Widget["MathPanel",
        {"preferredSize" -> 
            Widget["Dimension", {"width" -> 800, "height" -> 800}],
          BindEvent["mousePressed", Script[mousePressed[]]],
          BindEvent["mouseDragged", Script[mouseDragged[]]],
          BindEvent["mouseReleased", Script[mouseReleased[]]],
          BindEvent["componentResized", Script[componentResized[]]], 
          BindEvent["componentShown", Script[focusGained[]]]
          }, Name -> "canvas", WidgetLayout->{ "Stretching"->{Maximize, Maximize}}] ,
      
		Widget["Dialog", 
              {"title" -> "Edit label","modal" -> True,
                  Widget["TextField", {"text" -> ""}, Name -> "dialogTextField"],
                  Widget["Button", {"text" -> "Done",
                      BindEvent["action", Script[getVLabel[];InvokeMethod[{"myDialog", "dispose"}]] ]}],
                  InvokeMethod["center"]}, 
              Name -> "myDialog"],
        

      Script[
        makeColor[{r_, g_, b_}] := JavaNew["java.awt.Color", r, g, b];
        vlabel:="";
        black := makeColor[{0, 0, 0}];
        white := makeColor[{255, 255, 255}];
        red := makeColor[{255, 0, 0}];
        blue := makeColor[{0, 0, 255}];
		green := makeColor[{0, 255, 0}];
        frameback := makeColor[{255, 255, 204}];
        frameedge := makeColor[{240, 217, 92}];
        edgecolor := makeColor[{128, 0, 0}];
        mlv = True;
        mindist = 20;
        verts = GraphEdit`Private`getInitialVertices[]; 
        edges = GraphEdit`Private`getInitialEdges[];
        vtxnames = $VertexNames;
		vertStack = {};
        If[! ListQ[verts], verts = {}];
        If[! ListQ[edges], edges = {}];
        Map[(edgeHash[#] = 1) &, edges];
        currPos = Null;
        currIndex = 0;
        needInit = True;
        xSize = 800;
        ySize = 800;
        state = addNodes;
        showVertexLabelsState = "None";
        showEdgeLabelsState = "None";
        
        setMultilevel[]:= Module[{},
           mlv = PropertyValue[{"mlvcheckbox", "selected"}];
        ];
        getVLabel[]:=Module[{},vlabel:=PropertyValue[{"dialogTextField", "text"}]];
        deleteAllButton[] :=
          Module[{},
            clearVertStack[];
            verts = {};
            edges = {};
            Clear[edgeHash];
            redrawAll[];
            ];
        componentResized[] :=
          Module[{},
            xSize = PropertyValue[{"canvas", "width"}];
            ySize = PropertyValue[{"canvas", "height"}];
            needInit = True;
            makeGraphics[];
            redrawAll[];
            ];
        makeGraphics[] :=
          If[needInit,
            needInit = False;
            InvokeMethod[{WidgetReference["canvas"], "createImage"}, xSize, 
              ySize, Name -> "offscreen"];
            PropertyValue[{"offscreen", "graphics"}, Name -> "gObj"]];
 
        getMin[pts_, {x_, y_}] :=
          Module[{min = Infinity, imin = Length[pts] + 1}, 
            MapIndexed[
              If[Norm[N[#] - {x, y}] < min, min = Norm[N[#] - {x, y}]; 
                  imin = First[#2]] &, pts];
            {imin, min}];
        
        addVert[pt_] :=
          Module[ {imin, min},
            {imin, min} = getMin[verts, pt];
            If[min > mindist, pushVerts[verts];AppendTo[verts, pt]; imin = Length[verts]; vtxnames=Append[vtxnames,imin]];
            imin
            ];
        
        undoButton[] :=
          Module[{},
            popVerts[];
            redrawAll[];
            ];
        
        doneButton[] :=
          Module[{},
             CloseGUIObject[GUIObject[]]
            ];
        clearVertStack[] :=
          Module[{},
            vertStack = {};
            SetPropertyValue[{"undobutton", "enabled"}, False];
            ];
        pushVerts[vertSave_] :=
          Module[{},
            SetPropertyValue[{"undobutton", "enabled"}, True];
            vertStack = {vertSave,edges,vtxnames, vertStack};
            ];
        
        popVerts[] :=
          Module[{},
            If[vertStack =!= {},
              {verts,edges,vtxnames,vertStack} = vertStack];
            If[vertStack === {},SetPropertyValue[{"undobutton", "enabled"}, False]];
            ];
        randomButton[] :=
        	Module[{},
        		pushVerts[verts];
        		verts = Table[ {Random[],Random[]},{Length[verts]}];
         		rescaleFunction[];
        	];       
        rescaleButton[] :=
          rescaleFunction[verts];
        
        scaleCoord[coord0_, xD_, yD_]:= Module[
          {coord = coord0,min,max,diff,xmax },
            (* in the applet, y going from boptton to top *)
            coord = Map[{#[[1]],#[[2]]}&,coord];
            min=Map[Min,Transpose[coord]];
            max=Map[Max,Transpose[coord]];
            coord = Map[(#-min)&, coord];
            (* convert to between {0,0} and {1,1}*)
            diff = Map[Max[#,1]&,(max-min)];
            coord = Map[(#/diff)&, coord];
            coord = Map[(Round[{Min[xSize-xD,ySize-yD],Min[xSize-xD,ySize-yD]}*#] + {xD,yD}/2)&, coord];
            Round[coord]

        ];

        rescaleFunction[vertSave_:Null] :=
          Module[ {xP, yP, xd, yd},
            xD = xSize/25.;
            yD = ySize/25.;
            verts = scaleCoord[verts, xD, yD];
            If[verts =!= vertSave && vertSave =!= Null, pushVerts[vertSave]];
            redrawAll[];
            ];
        compLayout[GraphPlotMethod_String, maxit_Integer] := Module[
            {graph,nverts},
            graph = Apply[Rule, edges, {1}];
            If [GraphPlotMethod === "SpringModel" ||  
                 GraphPlotMethod === "SpringElectricalModel",
                 verts = -GraphCoordinates[graph, Method->{SpringElectricalModel,Octree->False,Rotation->False,RecursionMethod->None,MaxIterations->maxit},$opts];          
            ];
       		rescaleFunction[];
        ];

        animateButton[] :=
          Module[{oldsysopt, graph, nverts, method, vertSave = verts},
            method = PropertyValue[{"methodBox", "selectedItem"}];
            method = method /.
                {"Spring-Electrical" -> "SpringElectricalModel", 
                  "Spring" -> "SpringModel", 
                  "High-D Embedding" -> "HighDimensionalEmbedding",  
                  "Tree Drawing" -> "LayeredDrawing",  
                  "Layered Drawing" -> "LayeredDigraphDrawing",  
                  "Radial Drawing" -> "RadialDrawing"};
            oldsysopt = SystemOptions["GraphPlotOptions"];
            SetSystemOptions[
				"GraphPlotOptions"->{
						ScaleCoordinates->False,
						FineSteps->True}]; 

            If [method =!= "SpringModel" && 
                method =!= "SpringElectricalModel", 
                  method = "SpringElectricalModel"
            ];
                
            Do[
               t=AbsoluteTiming[compLayout[method, MAXITERATIONS]][[1]]/1000;
               Pause[Max[0,.1-t]];
              ,
              {MAXITERATIONS,1,1000,40}
            ];
            Do[
               t=AbsoluteTiming[compLayout[method, MAXITERATIONS]][[1]]/1000;
               Pause[Max[0,.1-t]];
               ,
              {MAXITERATIONS,1000,10000,400}
            ];
            SetSystemOptions@@oldsysopt;
        ];
  
        redrawButton[firsttime_] :=
          Module[{graph, nverts, method, vertSave = verts, vertold, step, t = 0},
            method = PropertyValue[{"methodBox", "selectedItem"}];
            method = method /.
                {"Spring-Electrical" -> "SpringElectricalModel", 
                  "Spring" -> "SpringModel", 
                  "High-D Embedding" -> "HighDimensionalEmbedding",  
                  "Layered Drawing" -> "LayeredDigraphDrawing",  
                  "Tree Drawing" -> "LayeredDrawing",  
                  "Radial Drawing" -> "RadialDrawing"};
            If[ method === "Randomize",
            	randomButton[],
            	graph = Apply[Rule, edges, {1}];
               If [method === "SpringElectricalModel" || 
                   method === "SpringModel",method = {method, RecursionMethod->mlv}];
            	If [firsttime,
                  nverts = Floor[10000.* verts],
                  nverts = Floor[10000.* (-GraphCoordinates[graph, $opts, Method -> method])];
               ];
            	If[MatchQ[nverts, {{_Integer, _Integer} ..}],
                 If [vertStack =!= {},
                   vertold = verts;
           		    t = AbsoluteTiming[rescaleFunction[vertold];][[1]];
                   step = 20;
                   If [t> 0.1 && t < 1.5, 
                       step = Round[20*0.1/t], step = 1];
                   Do[
                      verts = (1-i^3/step^3) * vertold + (i^3/step^3)* nverts;
                      verts = Floor[10000.*verts];
                      Pause[0.01];
              		    rescaleFunction[vertSave]
                  ,{i,0,step}],
                   verts = nverts;
                   rescaleFunction[vertSave]
                  ]]];
            ];       
        locateVert[] :=
          Module[{i},
            i = addVert[getXY[]];
            getVert[i]
            ];

        getVert[i_] :=
          Part[verts, i];

        findVertIndex[] :=
          Module[{imin, min},
            {imin, min} = getMin[verts, getXY[]];
            If[min < mindist, imin, 0]
            ];
        setVertPosition[ind_, {x_, y_}] :=
          Part[verts, ind] = {x, y};
        
        removeVert[ind_] :=
          Module[{},
            pushVerts[verts];
            edges = DeleteCases[edges, {ind, _} | {_, ind}];
            edges = Map[If[# > ind, # - 1, #] &, edges, {2}];
            verts = Drop[verts, {ind}];
            Clear[edgeHash];
            Delete[vtxnames,ind];
            Map[edgeHash[#] = 1, edges];
            ];
        
        addButton[] :=
          Module[{},
            state = addNodes;
            ];
        
        moveButton[] :=
          Module[{},
            state = moveNodes;
            ];
        
        deleteButton[] :=
          Module[{},
            state = deleteNodes;
            ];
        
        mousePressed[] :=
          Module[{ind, coords = Null,txt},
			If[PropertyValue[{"#", "modifiers"}]==18,ind = findVertIndex[];
            If[ind>0,InvokeMethod[{"myDialog", "show"}];vtxnames[[ind]]=vlabel;redrawAll[],Null],
			makeGraphics[];
            print[verts];
            Switch[state,
              moveNodes,
              	ind = findVertIndex[];
              	If[ind > 0, pushVerts[verts]; currIndex = ind; coords = getVert[ind]],
              addNodes,
              	coords = locateVert[],
              deleteNodes,
              	ind = findVertIndex[];
              	If[ind > 0, removeVert[ind];redrawAll[]];
              	];
            currPos = coords];
            ];
        mouseDragged[] :=
          Module[{xP, yP},
            If[! MatchQ[currPos, {_, _}], Return[{}]];
            {xP, yP} = getXY[];
            If[state === moveNodes,
              setVertPosition[currIndex, {xP, yP}];
              redrawAll[];Return[]];
            If[state === addNodes,redrawAll[False];
              SetPropertyValue[{"gObj", "color"}, black];
              InvokeMethod[{"gObj", "drawLine"}, currPos[[1]], currPos[[2]], 
                xP, yP];
              SetPropertyValue[{"gObj", "color"}, frameback];
              InvokeMethod[{"gObj", "fillOval"}, currPos[[1]] - 5, 
                currPos[[2]] - 5, 10, 10];
              repaint[]];
            ];
        mouseReleased[] :=
          Module[{imin1, imin2},
            If[! MatchQ[currPos, {_, _}] || state =!= addNodes, Return[]];
            imin1 = addVert[currPos];
            imin2 = addVert[getXY[]];
            If[edgeHash[{imin1, imin2}] =!= 1,pushVerts[verts]; AppendTo[edges, {imin1, imin2}];];
            edgeHash[{imin1, imin1}] = 1;
            redrawAll[]
            ];
		showVertexLabels[] := Module[{},
            showVertexLabelsState = PropertyValue[{"setVertexLabel", "selectedItem"}];
            redrawAll[];
            ];
 	   showEdgeLabels[] := Module[{},
            showEdgeLabelsState = PropertyValue[{"setEdgeLabel", "selectedItem"}];
            redrawAll[];
            ];
        getXY[] :=
          {PropertyValue[{"#", "x"}], PropertyValue[{"#", "y"}]};
        
        redrawAll[ repaintQ_:True] :=
          Module[{fm,str,s},
            print[{xSize, ySize}];
            obj = WidgetReference["gObj"];

            obj@setColor[white];
            obj@fillRect[0, 0, xSize, ySize];
            obj@setColor[edgecolor];
            
            Map[obj@drawLine[verts[[#[[1]], 1]], verts[[#[[1]], 2]], 
                    verts[[#[[2]], 1]], verts[[#[[2]], 2]]] &, edges];
            
            obj@setColor[black];

            If[showEdgeLabelsState != "None", MapIndexed[
					obj@drawString[
               If[showEdgeLabelsState === "Number", 
                 ToString[First[#2]],
                 FromCharacterCode[65+IntegerDigits[First[#2]-1,26]]
               ],( verts[[#1[[1]], 1]]/2.0 + verts[[#1[[2]], 1]]/2.0 + 10*Normalize[verts[[#1[[1]], 2]]-verts[[#1[[2]], 2]]]), 
                    ( verts[[#1[[1]], 2]]/2.0+ verts[[#1[[2]], 2]]/2.0+10*Normalize[verts[[#1[[2]], 1]]-verts[[#1[[1]], 1]]])] &, edges]];
            obj@setColor[red];

            fm=obj@getFontMetrics[obj@getFont[]];
            getBox[cord_,i_,xx_,yy_]:= Module[{x,y},
              {x,y}=cord;
              obj@setColor[frameback];
              obj@fillRect[x-Round[xx/2], y-Round[yy*0.5],xx, yy];
              obj@setColor[frameedge];
              obj@drawRect[x-Round[xx/2], y-Round[yy*0.5],xx, yy];
            ];
            getLabeledBox[cord_, ind_]:= Module[
               {str = None,xx,yy},
              obj@setColor[black];
              If[showVertexLabelsState =!= "None",
               str=If[showVertexLabelsState === "Number", 
                ToString[First[ind]],
                If [showVertexLabelsState === "Name",
	                 ToString[vtxnames[[First[ind]]]],
                    FromCharacterCode[65+IntegerDigits[First[ind]-1,26]]
                ]
               ];
               ];
              If [str =!= None,
                xx=fm@stringWidth[str];
                yy=fm@getHeight[];
                getBox[cord,ind[[1]],xx,yy];
                obj@setColor[black];
                obj@drawString[
                  str, cord[[1]]-Round[xx/2], cord[[2]]+Round[yy/4]],

                obj@setColor[frameback];
                obj@fillOval[cord[[1]] - 5, cord[[2]] - 5, 10, 10];
                obj@setColor[frameedge];
                obj@drawOval[cord[[1]] - 5, cord[[2]] - 5, 10, 10];
              ]
             ];

            MapIndexed[getLabeledBox,verts];

            If[repaintQ, repaint[]];
            ];
        
        
        repaint[] :=
          (
            print[repaint];
            makeGraphics[];
            SetPropertyValue[{"canvas", "image"}, 
              WidgetReference["offscreen"],InvokeMethod->"Dispatch"];
            );
        
        focusGained[] :=
          (makeGraphics[]; redrawAll[]);
        
        ],
        
		BindEvent["endModal", 
    		Script[
        		{Apply[ {#1, ySize - #2} &,verts,{1}], edges, vtxnames}
     		]
   		]
   	}, Name -> "panel"],
   	"title" -> "Graph Editor"
   	}];



(* end GraphGUINew.m *)


(*============== end  GraphEdit ============== *)


End[];(*end private*)
EndPackage[];
