(* ::Package:: *)

(* :Title: Combinatorica *)
(* :Authors: Sriram V. Pemmaraju and Steven S. Skiena*)
(* :Summary:
This package contains all the programs from the book, "Computational 
Discrete Mathematics: Combinatorics and Graph Theory in Mathematica",
by Sriram V. Pemmaraju and Steven S. Skiena, Cambridge University Press,
2003.
*)
(* :Discussion:
The programs from the book "Computational Discrete Mathematics: Combinatorics
and Graph Theory with Mathematica" are available at www.combinatorica.com.
Any comments or bug reports should be forwarded to one of the following:
	Sriram Pemmaraju
	Department of Computer Science
	University of Iowa
	Iowa City, IA 52242
        sriram@cs.uiowa.edu
	(319)-353-2956
	Steven Skiena
	Department of Computer Science
	State University of New York
	Stony Brook, NY 11794-4400
	skiena@cs.sunysb.edu
	(631)-632-9026
*)
(* :Context: Combinatorica`  
*)
(* :Package Version: 2.1.0
*)
(* :Copyright: Copyright 2000-2007 by Sriram V. Pemmaraju and Steven S. Skiena
*)
(*
This package may be copied in its entirety for nonprofit purposes only.
Sale, other than for the direct cost of the media, is prohibited.  This
copyright notice must accompany all copies.
The authors, Wolfram Research, and Cambridge University Press
make no representations, express or implied, with respect to this
documentation, or the software it describes and contains, including
without limitations, any implied warranties of mechantability or fitness
for a particular purpose, all of which are expressly disclaimed.  The
authors, Wolfram Research, or Cambridge University Press, their licensees,
distributors and dealers shall in no event be liable for any indirect,
incidental, or consequential damages.
*)
(* :History:
    Version 2.1 updated to Mathematica 6 by John M. Novak, 2006.
    Version 2.0 most code rewritten Sriram V. Pemmaraju, 2000-2002
            Too many changes to describe here. Read the book!
	Version 1.1 modification by ECM, March 1996.  
		Replaced K with CompleteGraph because K is now the
			default generic name for the summation index in
			symbolic sum.
		Added CombinatorialFunctions.m and Permutations.m to
			BeginPackage, and commented out CatalanNumber,
			PermutationQ, ToCycles, FromCycles, and
			RandomPermutation so there would be no shadowing of
			symbols among the DiscreteMath packages.	
		Replaced old BinarySearch with new code by Paul Abbott
			correctly implementing binary search.
        Version 1.0 by Steven S. Skiena, April 1995.
        Version .9 by Steven S. Skiena, February 1992.
	Version .8 by Steven S. Skiena, July 1991.
	Version .7 by Steven S. Skiena, January 1991. 
	Version .6 by Steven S. Skiena, June 1990.
*)
(*
Acknowledgements: 
        WRI people who helped: John Novak, Eric Weisstein, Arnoud Buzing,
        Shiral Devmal, Anna Pakin, Andy Shiekh, Darren Glosemeyer, Ranjani
        Krishnan, Daniel Lichtblau, 
        Robby Villegas, Stephen Wolfram
	Others who helped: Eugene Curtin, Levon LLoyd, Joan Trias, Kaushal 
        Kurapati, students at Iowa, students at IIT Bombay
*)
(* :Keywords:
	adjacency, automorphism, chromatic, clique, coloring,
	combination, composition, connected components, connectivity, cycle,
	de Bruijn, degree, derangement, Dijkstra, Durfee,
	embedding, equivalence, Eulerian, Ferrers,
	geodesic, graph, Gray code, group, Hamiltonian cycle, Harary, Hasse,
	heap, hypercube, interval, inversion, involution, isomorphism,
	Josephus, network,
	partition, perfect, permutation, planar graph, pseudograph,
	self-loop, sequence, signature, simple, spanning tree,
	stable marriage, star, Stirling,
	transitive closure, traveling salesman tour, tree, Turan,
	vertex cover, wheel, Young tableau
*)
(* :Source:
	Sriram V. Pemmaraju and Steven S. Skiena
        Computational Discrete Mathematics: Combinatorics
	and Graph Theory in Mathematica, 
*)
(* :Mathematica Version: 6.0
*)
(* Force preloading of System` *)
{System`Path,System`Star,System`Thick,System`Thin};
(* Following declarations suppress shadow messages for overridden System` functionality. *)
{
Combinatorica`AlternatingGroup,
Combinatorica`ButterflyGraph,
Combinatorica`ChromaticPolynomial,
Combinatorica`CirculantGraph,
Combinatorica`CompleteGraph,
Combinatorica`CompleteKaryTree,
Combinatorica`ConnectedComponents,
Combinatorica`Cycles,
Combinatorica`CyclicGroup,
Combinatorica`DeBruijnGraph,
Combinatorica`DeBruijnSequence,
Combinatorica`DeleteEdge,
Combinatorica`DihedralGroup,
Combinatorica`EdgeColor,
Combinatorica`EdgeConnectivity,
Combinatorica`EdgeLabel,
Combinatorica`EdgeStyle,
Combinatorica`EdgeWeight,
Combinatorica`FindCycle,
Combinatorica`FromCycles,
Combinatorica`Graph,
Combinatorica`GraphCenter,
Combinatorica`GraphComplement,
Combinatorica`GraphDifference,
Combinatorica`GraphIntersection,
Combinatorica`GraphPower,
Combinatorica`GraphUnion,
Combinatorica`GridGraph,
Combinatorica`IncidenceMatrix,
Combinatorica`InversePermutation,
Combinatorica`LineGraph,
Combinatorica`PermutationQ,
Combinatorica`Permute,
Combinatorica`PetersenGraph,
Combinatorica`RandomGraph,
Combinatorica`RandomPermutation,
Combinatorica`SymmetricGroup,
Combinatorica`ToCycles,
Combinatorica`TopologicalSort,
Combinatorica`VertexConnectivity,
Combinatorica`VertexCoverQ,
Combinatorica`VertexLabel,
Combinatorica`VertexStyle,
Combinatorica`VertexWeight,
Combinatorica`WeaklyConnectedComponents
};
(* Attach following to General for convenience; easier for users to deactivate. *)
General::compat = "Combinatorica Graph and Permutations functionality has been superseded by preloaded functionality. The package now being loaded may conflict with this. Please see the Compatibility Guide for details.";
Message[General::compat];

BeginPackage["Combinatorica`"]
Unprotect[
AcyclicQ, 
Combinatorica`AddEdge,
AddEdges, 
Combinatorica`AddVertex,
AddVertices,
Algorithm,
Combinatorica`AlternatingGroup,
AlternatingGroupIndex,
AlternatingPaths,
AnimateGraph,
AntiSymmetricQ,
Approximate,
ApproximateVertexCover,
ArticulationVertices,
Automorphisms,
Backtrack, 
BellmanFord,
BiconnectedComponents, 
BiconnectedQ,
BinarySearch, 
BinarySubsets, 
BipartiteMatching, 
BipartiteMatchingAndCover, 
BipartiteQ,
BooleanAlgebra,
Box, 
Combinatorica`BreadthFirstTraversal, 
Brelaz,
BrelazColoring,
Bridges, 
Combinatorica`ButterflyGraph,
ToCanonicalSetPartition,
CageGraph,
CartesianProduct, 
Center, 
ChangeEdges, 
ChangeVertices,
ChromaticNumber, 
Combinatorica`ChromaticPolynomial,
ChvatalGraph,
Combinatorica`CirculantGraph, 
CircularEmbedding, 
CircularVertices, 
CliqueQ, 
CoarserSetPartitionQ,
CodeToLabeledTree, 
Cofactor,
CompleteBinaryTree,
Combinatorica`CompleteKaryTree,
CompleteKPartiteGraph,
CompleteGraph,
CompleteQ,
Compositions, 
Combinatorica`ConnectedComponents, 
ConnectedQ, 
ConstructTableau,
Combinatorica`Contract, 
CostOfPath,
CoxeterGraph,
CubeConnectedCycle,
CubicalGraph,
Cut,
Cycle, 
Combinatorica`Cycles, 
CycleIndex,
CycleStructure,
Cyclic,
Combinatorica`CyclicGroup,
CyclicGroupIndex,
Combinatorica`DeBruijnGraph, 
Combinatorica`DeBruijnSequence, 
Degrees,
DegreesOf2Neighborhood,
DegreeSequence,
DeleteCycle, 
Combinatorica`DeleteEdge, 
DeleteEdges, 
DeleteFromTableau, 
Combinatorica`DeleteVertex,
DeleteVertices, 
Combinatorica`DepthFirstTraversal,
DerangementQ, 
Derangements, 
Diameter, 
Dihedral,
Combinatorica`DihedralGroup,
DihedralGroupIndex,
Dijkstra, 
DilateVertices,
Directed, 
Distances,
DistinctPermutations, 
Distribution, 
DodecahedralGraph,
DominatingIntegerPartitionQ,
DominationLattice,
DurfeeSquare, 
Eccentricity,
Edge,
EdgeChromaticNumber, 
Combinatorica`EdgeColor,
EdgeColoring, 
Combinatorica`EdgeConnectivity, 
EdgeDirection, 
Combinatorica`EdgeLabel, 
EdgeLabelColor, 
EdgeLabelPosition, 
Edges, 
Combinatorica`EdgeStyle, 
Combinatorica`EdgeWeight, 
Element,
EmptyGraph, 
EmptyQ, 
EncroachingListSet, 
EquivalenceClasses, 
EquivalenceRelationQ, 
Equivalences, 
Euclidean,
Eulerian,
EulerianCycle,
EulerianQ, 
ExactRandomGraph, 
ExpandGraph, 
ExtractCycles, 
FerrersDiagram, 
Combinatorica`FindCycle,
FindSet, 
FiniteGraphs,
FirstLexicographicTableau, 
FolkmanGraph,
FranklinGraph,
FruchtGraph,
FromAdjacencyLists,
FromAdjacencyMatrix,
Combinatorica`FromCycles,
FromInversionVector, 
FromOrderedPairs,
FromUnorderedPairs, 
FunctionalGraph,
GeneralizedPetersenGraph,
GetEdgeLabels,
GetEdgeWeights,
GetVertexLabels,
GetVertexWeights,
Girth, 
Combinatorica`GraphCenter, 
Combinatorica`GraphComplement, 
Combinatorica`GraphDifference, 
GraphicQ,
Combinatorica`GraphIntersection,
Combinatorica`GraphJoin, 
GraphOptions, 
GraphPolynomial,
Combinatorica`GraphPower, 
Combinatorica`GraphProduct, 
Combinatorica`GraphSum, 
Combinatorica`GraphUnion, 
GrayCode, 
GrayCodeSubsets, 
GrayCodeKSubsets, 
GrayGraph,
Greedy,
GreedyVertexCover,
Combinatorica`GridGraph, 
GroetzschGraph,
GrotztschGraph,
HamiltonianCycle, 
HamiltonianPath, 
HamiltonianQ, 
Harary,
HasseDiagram, 
Heapify, 
HeapSort, 
HeawoodGraph,
HerschelGraph,
HideCycles, 
HighlightedEdgeColors,
HighlightedEdgeStyle,
HighlightedVertexColors,
HighlightedVertexStyle,
Highlight,
Hypercube, 
IcosahedralGraph,
IdenticalQ,
IdentityPermutation,
Combinatorica`IncidenceMatrix,
InDegree,
IndependentSetQ, 
Index, 
InduceSubgraph,
InitializeUnionFind,
InsertIntoTableau, 
IntervalGraph, 
Invariants,
InversePermutation, 
InversionPoset,
Inversions,
InvolutionQ, 
Involutions, 
IsomorphicQ, 
Isomorphism, 
IsomorphismQ, 
Josephus, 
KnightsTourGraph,
KSetPartitions,
KSubsetGroup,
KSubsetGroupIndex,
KSubsets,
LNorm,
LabeledTreeToCode, 
LastLexicographicTableau,
LexicographicPermutations, 
LexicographicSubsets, 
LeviGraph,
Combinatorica`LineGraph,
ListGraphs,
ListNecklaces,
LongestIncreasingSubsequence, 
LoopPosition,
LowerLeft, 
LowerRight, 
M,
MakeDirected,
MakeGraph, 
MakeSimple, 
MakeUndirected,
MaximalMatching,
MaximumAntichain, 
MaximumClique, 
MaximumIndependentSet,
MaximumSpanningTree, 
McGeeGraph,
MeredithGraph,
MinimumChainPartition, 
MinimumChangePermutations,
MinimumSpanningTree, 
MinimumVertexColoring, 
MinimumVertexCover, 
MultipleEdgesQ,
MultiplicationTable,
MycielskiGraph,
NecklacePolynomial,
Neighborhood,
NetworkFlow, 
NetworkFlowEdges, 
NextBinarySubset, 
NextComposition, 
NextGrayCodeSubset,
NextKSubset,
NextLexicographicSubset,
NextPartition, 
NextPermutation, 
NextSubset, 
NextTableau, 
NoMultipleEdges, 
NonLineGraphs,
NoPerfectMatchingGraph,
Normal, 
NormalDashed, 
NormalizeVertices,
NoSelfLoops, 
NthPair,
NthPermutation, 
NthSubset, 
NumberOfCompositions,
NumberOfDerangements, 
NumberOfDirectedGraphs, 
NumberOfGraphs,
NumberOfInvolutions, 
NumberOf2Paths,
NumberOfKPaths,
NumberOfNecklaces,
NumberOfPartitions,
NumberOfPermutationsByCycles, 
NumberOfPermutationsByInversions, 
NumberOfPermutationsByType,
NumberOfSpanningTrees, 
NumberOfTableaux,
OctahedralGraph,
OddGraph,
One,
Optimum,
OrbitInventory,
OrbitRepresentatives,
Combinatorica`Orbits,
Ordered,
OrientGraph, 
OutDegree,
PairGroup,
PairGroupIndex,
Parent,
ParentsToPaths,
PartialOrderQ, 
PartitionLattice,
PartitionQ, 
Partitions, 
Path, 
PerfectQ,
PermutationGraph, 
PermutationGroupQ, 
Combinatorica`PermutationQ,
PermutationToTableaux,
PermutationType,
PermutationWithCycle,
Combinatorica`Permute, 
PermuteSubgraph, 
Combinatorica`PetersenGraph,
PlanarQ,
PlotRange, 
Polya,
PseudographQ, 
RadialEmbedding, 
Radius,
RandomComposition, 
Combinatorica`RandomGraph, 
RandomHeap, 
RandomInteger,
RandomKSetPartition,
RandomKSubset,
RandomPartition, 
Combinatorica`RandomPermutation,
RandomRGF,
RandomSetPartition,
RandomSubset, 
RandomTableau, 
RandomTree, 
RandomVertices, 
RankBinarySubset, 
RankedEmbedding, 
RankGraph,
RankGrayCodeSubset, 
RankKSetPartition, 
RankKSubset, 
RankPermutation, 
RankRGF,
RankSetPartition,
RankSubset, 
ReadGraph,
RealizeDegreeSequence, 
ReflexiveQ,
RegularGraph, 
RegularQ, 
RemoveMultipleEdges, 
RemoveSelfLoops, 
ResidualFlowGraph,
RevealCycles, 
ReverseEdges,
RGFQ,
RGFs,
RGFToSetPartition,
RobertsonGraph,
RootedEmbedding, 
RotateVertices, 
Runs, 
SamenessRelation,
SelectionSort, 
SelfComplementaryQ, 
SelfLoopsQ,
SetEdgeWeights,
SetGraphOptions, 
SetPartitions,
SetPartitionListViaRGF,
SetPartitionQ,
SetPartitionToRGF,
SetEdgeLabels,
SetVertexLabels,
SetVertexWeights,
ShakeGraph,
ShortestPathSpanningTree,
ShowLabeledGraph,
ShowGraph, 
ShowGraphArray, 
ShuffleExchangeGraph,
SignaturePermutation,
Simple, 
SimpleQ,
SmallestCyclicGroupGraph,
Spectrum, 
SpringEmbedding, 
StableMarriage, 
Star,
StirlingFirst, 
StirlingSecond,
Strings, 
Strong,
StronglyConnectedComponents,
Combinatorica`SymmetricGroup,
SymmetricGroupIndex,
SymmetricQ,
TableauClasses, 
Combinatorica`TableauQ, 
Tableaux,
TableauxToPermutation, 
TetrahedralGraph,
ThickDashed, 
ThinDashed, 
ThomassenGraph,
ToAdjacencyLists,
ToAdjacencyMatrix, 
Combinatorica`ToCycles,
ToInversionVector, 
ToOrderedPairs,
Combinatorica`TopologicalSort, 
ToUnorderedPairs, 
TransitiveClosure, 
TransitiveQ,
TransitiveReduction, 
TranslateVertices, 
TransposePartition, 
Combinatorica`TransposeTableau,
TravelingSalesmanBounds, 
TravelingSalesman, 
TreeIsomorphismQ,
TreeQ, 
TreeToCertificate,
TriangleInequalityQ,
Turan, 
TutteGraph,
TwoColoring, 
Type,
Undirected,
UndirectedQ, 
UnionSet, 
Uniquely3ColorableGraph,
UnitransitiveGraph,
UnrankBinarySubset,
UnrankGrayCodeSubset,
UnrankKSetPartition,
UnrankKSubset,
UnrankPermutation,
UnrankRGF,
UnrankSetPartition,
UnrankSubset,
UnweightedQ,
UpperLeft, 
UpperRight, 
V, 
VertexColor, 
VertexColoring, 
Combinatorica`VertexConnectivity, 
VertexConnectivityGraph, 
VertexCover,
Combinatorica`VertexCoverQ, 
Combinatorica`VertexLabel, 
VertexLabelColor, 
VertexNumber, 
VertexNumberColor,
Combinatorica`VertexStyle, 
Combinatorica`VertexWeight, 
Vertices,
WaltherGraph,
Weak, 
Combinatorica`WeaklyConnectedComponents, 
WeightingFunction,
WeightRange,
Wheel, 
WriteGraph,
Zoom
];

(* force load of regular usage messages to be appended *)
Map[MessageName[#, "usage"]&, {All, Box, Center, Disk, Element, K, Large, Normal, Path, Small, Star}];
(*Thick::usage;Thin::usage; (* because these two evaluate *) *)
(* get formatted Combinatorica messages, except for special cases *)
If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"Combinatorica`"],
StringMatchQ[#,StartOfString~~"Combinatorica`*"]&&
!StringMatchQ[#,"Combinatorica`"~~("EdgeColor"|"Path"|"Thin"|"Thick"|"Star"|"RandomInteger")~~__]&]//ToExpression;
]

Block[{$NewMessage},
If[Not[ValueQ[AcyclicQ::usage]],
AcyclicQ::usage = "AcyclicQ[g] yields True if graph g is acyclic."
];
If[Not[ValueQ[Combinatorica`AddEdge::usage]],
Combinatorica`AddEdge::usage = "AddEdge[g, e] returns a graph g with the new edge e added. e can have the form {a, b} or the form {{a, b}, options}."
];
If[Not[ValueQ[AddEdges::usage]],
AddEdges::usage = "AddEdges[g, edgeList] gives graph g with the new edges in edgeList added. edgeList can have the form {a, b} to add a single edge {a, b} or the form {{a, b}, {c, d}, ...}, to add edges {a, b}, {c, d}, ... or the form { {{a, b}, x}, {{c, d}, y}, ...}, where x and y can specify graphics information associated with {a, b} and {c, d}, respectively."
];
If[Not[ValueQ[Combinatorica`AddVertex::usage]],
Combinatorica`AddVertex::usage = "AddVertex[g] adds one disconnected vertex to graph g. AddVertex[g, v] adds to g a vertex with coordinates specified by v."
];
If[Not[ValueQ[AddVertices::usage]],
AddVertices::usage = "AddVertices[g, n] adds n disconnected vertices to graph g. AddVertices[g, vList] adds vertices in vList to g. vList contains embedding and graphics information and can have the form {x, y} or {{x1, y1}, {x2, y2}...} or the form {{{x1, y1}, g1}, {{x2, y2}, g2},...}, where {x, y}, {x1, y1}, and {x2, y2} are point coordinates and g1 and g2 are graphics information associated with vertices."
];
If[Not[ValueQ[Algorithm::usage]],
Algorithm::usage = "Algorithm is an option that informs functions such as ShortestPath, VertexColoring, and VertexCover about which algorithm to use."
];
If[Not[ValueQ[AllPairsShortestPath::usage]],
AllPairsShortestPath::usage = "AllPairsShortestPath[g] gives a matrix, where the (i, j)th entry is the length of a shortest path in g between vertices i and j. AllPairsShortestPath[g, Parent] returns a three-dimensional matrix with dimensions 2 * V[g] * V[g], in which the (1, i, j)th entry is the length of a shortest path from i to j and the (2, i, j)th entry is the predecessor of j in a shortest path from i to j."
];
$NewMessage[ All, "usage"]; (* reset the usage of All to the System usage *)
If[StringQ[All::usage], All::usage = StringJoin[ All::usage, " All is also an option to certain Combinatorica functions specifying that all solutions should be returned, instead of just the first one."]];
If[Not[ValueQ[Combinatorica`AlternatingGroup::usage]],
Combinatorica`AlternatingGroup::usage = "AlternatingGroup[n] generates the set of even-size n permutations, the alternating group on n symbols. AlternatingGroup[l] generates the set of even permutations of the list l."
];
If[Not[ValueQ[AlternatingGroupIndex::usage]],
AlternatingGroupIndex::usage = "AlternatingGroupIndex[n, x] gives the cycle index of the alternating group of size n permutations as a polynomial in the symbols x[1], x[2], ..., x[n]."
];
If[Not[ValueQ[AlternatingPaths::usage]],
AlternatingPaths::usage = "AlternatingPaths[g, start, ME] returns the alternating paths in graph g with respect to the matching ME, starting at the vertices in the list start. The paths are returned in the form of a forest containing trees rooted at vertices in start."
];
If[Not[ValueQ[AnimateGraph::usage]],
AnimateGraph::usage = "AnimateGraph[g, l] displays graph g with each element in the list l successively highlighted. Here l is a list containing vertices and edges of g. An optional flag, which takes on the values All and One, can be used to inform the function about whether objects highlighted earlier will continue to be highlighted or not. The default value of flag is All. All the options allowed by the function Highlight are permitted by AnimateGraph, as well. See the usage message of Highlight for more details."
];
If[Not[ValueQ[AntiSymmetricQ::usage]],
AntiSymmetricQ::usage = "AntiSymmetricQ[g] yields True if the adjacency matrix of g represents an anti-symmetric binary relation."
];
If[Not[ValueQ[Approximate::usage]],
Approximate::usage = "Approximate is a value that the option Algorithm can take in calls to functions such as VertexCover, telling it to use an approximation algorithm."
];
If[Not[ValueQ[ApproximateVertexCover::usage]],
ApproximateVertexCover::usage = "ApproximateVertexCover[g] produces a vertex cover of graph g whose size is guaranteed to be within twice the optimal size."
];
If[Not[ValueQ[ArticulationVertices::usage]],
ArticulationVertices::usage = "ArticulationVertices[g] gives a list of all articulation vertices in graph g. These are vertices whose removal will disconnect the graph."
];
If[Not[ValueQ[Automorphisms::usage]],
Automorphisms::usage = "Automorphisms[g] gives the automorphism group of the graph g."
];
If[Not[ValueQ[Backtrack::usage]],
Backtrack::usage = "Backtrack[s, partialQ, solutionQ] performs a backtrack search of the state space s, expanding a partial solution so long as partialQ is True and returning the first complete solution, as identified by solutionQ."
];
If[Not[ValueQ[BellmanFord::usage]],
BellmanFord::usage = "BellmanFord[g, v] gives a shortest-path spanning tree and associated distances from vertex v of graph g. The shortest-path spanning tree is given by a list in which element i is the predecessor of vertex i in the shortest-path spanning tree. BellmanFord works correctly even when the edge weights are negative, provided there are no negative cycles."
];
If[Not[ValueQ[BiconnectedComponents::usage]],
BiconnectedComponents::usage = "BiconnectedComponents[g] gives a list of the biconnected components of graph g. If g is directed, the underlying undirected graph is used."
];
If[Not[ValueQ[BiconnectedQ::usage]],
BiconnectedQ::usage = "BiconnectedQ[g] yields True if graph g is biconnected. If g is directed, the underlying undirected graph is used."
];
If[Not[ValueQ[BinarySearch::usage]],
BinarySearch::usage = "BinarySearch[l, k] searches sorted list l for key k and gives the position of l containing k, if k is present in l. Otherwise, if k is absent in l, the function returns (p + 1/2) where k falls between the elements of l in positions p and p+1. BinarySearch[l, k, f] gives the position of k in the list obtained from l by applying f to each element in l."
];
If[Not[ValueQ[BinarySubsets::usage]],
BinarySubsets::usage = "BinarySubsets[l] gives all subsets of l ordered according to the binary string defining each subset. For any positive integer n, BinarySubsets[n] gives all subsets of {1, 2,.., n} ordered according to the binary string defining each subset."
];
If[Not[ValueQ[BipartiteMatching::usage]],
BipartiteMatching::usage = "BipartiteMatching[g] gives the list of edges associated with a maximum matching in bipartite graph g. If the graph is edge weighted, then the function returns a matching with maximum total weight."
];
If[Not[ValueQ[BipartiteMatchingAndCover::usage]],
BipartiteMatchingAndCover::usage = "BipartiteMatchingAndCover[g] takes a bipartite graph g and returns a matching with maximum weight along with the dual vertex cover. If the graph is not weighted, it is assumed that all edge weights are 1."
];
If[Not[ValueQ[BipartiteQ::usage]],
BipartiteQ::usage = "BipartiteQ[g] yields True if graph g is bipartite."
];
If[Not[ValueQ[BooleanAlgebra::usage]],
BooleanAlgebra::usage = "BooleanAlgebra[n] gives a Hasse diagram for the Boolean algebra on n elements. The function takes two options: Type and VertexLabel, with default values Undirected and False, respectively. When Type is set to Directed, the function produces the underlying directed acyclic graph. When VertexLabel is set to True, labels are produced for the vertices."
];
$NewMessage[Box,"usage"];
If[StringQ[Box::usage],
Box::usage = StringJoin[Box::usage,"   Box is a value that the option VertexStyle, used in ShowGraph, can be set to.",
Box::usage = "Box is a value that the option VertexStyle, used in ShowGraph, can be set to."]
];
If[Not[ValueQ[Combinatorica`BreadthFirstTraversal::usage]],
Combinatorica`BreadthFirstTraversal::usage = "BreadthFirstTraversal[g, v] performs a breadth-first traversal of graph g starting from vertex v, and gives the breadth-first numbers of the vertices. BreadthFirstTraversal[g, v, Edge] returns the edges of the graph that are traversed by breadth-first traversal. BreadthFirstTraversal[g, v, Tree] returns the breadth-first search tree. BreadthFirstTraversal[g, v, Level] returns the level number of the vertices."
];
If[Not[ValueQ[Brelaz::usage]],
Brelaz::usage = "Brelaz is a value that the option Algorithm can take when used in the function VertexColoring."
];
If[Not[ValueQ[BrelazColoring::usage]],
BrelazColoring::usage = "BrelazColoring[g] returns a vertex coloring in which vertices are greedily colored with the smallest available color in decreasing order of vertex degree."
];
If[Not[ValueQ[Bridges::usage]],
Bridges::usage = "Bridges[g] gives a list of the bridges of graph g, where each bridge is an edge whose removal disconnects the graph."
];
If[Not[ValueQ[Combinatorica`ButterflyGraph::usage]],
Combinatorica`ButterflyGraph::usage = "ButterflyGraph[n] returns the n-dimensional butterfly graph, a directed graph whose vertices are pairs (w, i), where w is a binary string of length n and i is an integer in the range 0 through n and whose edges go from vertex (w, i) to (w', i+1), if w' is identical to w in all bits with the possible exception of the (i+1)th bit. Here bits are counted left to right. An option VertexLabel, with default setting False, is allowed. When this option is set to True, vertices are labeled with strings (w, i)."
];
If[Not[ValueQ[CageGraph::usage]],
CageGraph::usage = "CageGraph[k, r] gives a smallest k-regular graph of girth r for certain small values of k and r. CageGraph[r] gives CageGraph[3, r]. For k = 3, r can be 3, 4, 5, 6, 7, 8, or 10. For k = 4 or 5, r can be 3, 4, 5, or 6."
];
If[Not[ValueQ[CartesianProduct::usage]],
CartesianProduct::usage = "CartesianProduct[l1, l2] gives the Cartesian product of lists l1 and l2."
];
$NewMessage[Center,"usage"];
If[StringQ[Center::usage],
Center::usage = StringJoin[Center::usage,"   Center is a value that options VertexNumberPosition, VertexLabelPosition, and EdgeLabelPosition can take on in ShowGraph."]
];
If[Not[ValueQ[ChangeEdges::usage]],
ChangeEdges::usage = "ChangeEdges[g, e] replaces the edges of graph g with the edges in e. e can have the form {{s1, t1}, {s2, t2}, ...} or the form { {{s1, t1}, gr1}, {{s2, t2}, gr2}, ...}, where {s1, t1}, {s2, t2}, ... are endpoints of edges and gr1, gr2, ... are graphics information associated with edges."
];
If[Not[ValueQ[ChangeVertices::usage]],
ChangeVertices::usage = "ChangeVertices[g, v] replaces the vertices of graph g with the vertices in the given list v. v can have the form {{x1, y1}, {x2, y2}, ...} or the form {{{x1, y1}, gr1}, {{x2, y2}, gr2}, ...}, where {x1, y1}, {x2, y2}, ... are coordinates of points and gr1, gr2, ... are graphics information associated with vertices."
];
If[Not[ValueQ[ChromaticNumber::usage]],
ChromaticNumber::usage = "ChromaticNumber[g] gives the chromatic number of the graph, which is the fewest number of colors necessary to color the graph."
];
If[Not[ValueQ[ChromaticPolynomial::usage]],
ChromaticPolynomial::usage = "ChromaticPolynomial[g, z] gives the chromatic polynomial P(z) of graph g, which counts the number of ways to color g with, at most, z colors."
];
If[Not[ValueQ[ChvatalGraph::usage]],
ChvatalGraph::usage = "ChvatalGraph returns a smallest triangle-free, 4-regular, 4-chromatic graph."
];
If[Not[ValueQ[Combinatorica`CirculantGraph::usage]],
Combinatorica`CirculantGraph::usage = "CirculantGraph[n, l] constructs a circulant graph on n vertices, meaning the ith vertex is adjacent to the (i+j)th and (i-j)th vertices, for each j in list l. CirculantGraph[n, l], where l is an integer, returns the graph with n vertices in which each i is adjacent to (i+l) and (i-l)."
];
If[Not[ValueQ[CircularEmbedding::usage]],
CircularEmbedding::usage = "CircularEmbedding[n] constructs a list of n points equally spaced on a circle. CircularEmbedding[g] embeds the vertices of g equally spaced on a circle."
];
If[Not[ValueQ[CircularVertices::usage]],
CircularVertices::usage = "CircularVertices[n] constructs a list of n points equally spaced on a circle. CircularVertices[g] embeds the vertices of g equally spaced on a circle. This function is obsolete; use CircularEmbedding instead."
];
If[Not[ValueQ[CliqueQ::usage]],
CliqueQ::usage = "CliqueQ[g, c] yields True if the list of vertices c defines a clique in graph g."
];
If[Not[ValueQ[CoarserSetPartitionQ::usage]],
CoarserSetPartitionQ::usage = "CoarserSetPartitionQ[a, b] yields True if set partition b is coarser than set partition a, that is, every block in a is contained in some block in b."
];
If[Not[ValueQ[CodeToLabeledTree::usage]],
CodeToLabeledTree::usage = "CodeToLabeledTree[l] constructs the unique labeled tree on n vertices from the Prufer code l, which consists of a list of n-2 integers between 1 and n."
];
If[Not[ValueQ[Cofactor::usage]],
Cofactor::usage = "Cofactor[m, {i, j}] calculates the (i, j)th cofactor of matrix m."
];
If[Not[ValueQ[CompleteBinaryTree::usage]],
CompleteBinaryTree::usage = "CompleteBinaryTree[n] returns a complete binary tree on n vertices."
];
If[Not[ValueQ[CompleteGraph::usage]],
CompleteGraph::usage = "CompleteGraph[n] creates a complete graph on n vertices. An option Type that takes on the values Directed or Undirected is allowed. The default setting for this option is Type -> Undirected. CompleteGraph[a, b, c,...] creates a complete k-partite graph of the prescribed shape. The use of CompleteGraph to create a complete k-partite graph is obsolete; use CompleteKPartiteGraph instead."
];
If[Not[ValueQ[Combinatorica`CompleteKaryTree::usage]],
Combinatorica`CompleteKaryTree::usage = "CompleteKaryTree[n, k] returns a complete k-ary tree on n vertices."
];
If[Not[ValueQ[CompleteKPartiteGraph::usage]],
CompleteKPartiteGraph::usage = "CompleteKPartiteGraph[a, b, c, ...] creates a complete k-partite graph of the prescribed shape, provided the k arguments a, b, c, ... are positive integers. An option Type that takes on the values Directed or Undirected is allowed. The default setting for this option is Type -> Undirected."
];
If[Not[ValueQ[CompleteQ::usage]],
CompleteQ::usage = "CompleteQ[g] yields True if graph g is complete. This means that between any pair of vertices there is an undirected edge or two directed edges going in opposite directions."
];
If[Not[ValueQ[Compositions::usage]],
Compositions::usage = "Compositions[n, k] gives a list of all compositions of integer n into k parts."
];
If[Not[ValueQ[Combinatorica`ConnectedComponents::usage]],
Combinatorica`ConnectedComponents::usage = "ConnectedComponents[g] gives the vertices of graph g partitioned into connected components."
];
If[Not[ValueQ[ConnectedQ::usage]],
ConnectedQ::usage = "ConnectedQ[g] yields True if undirected graph g is connected. If g is directed, the function returns True if the underlying undirected graph is connected. ConnectedQ[g, Strong] and ConnectedQ[g, Weak] yield True if the directed graph g is strongly or weakly connected, respectively."
];
If[Not[ValueQ[ConstructTableau::usage]],
ConstructTableau::usage = "ConstructTableau[p] performs the bumping algorithm repeatedly on each element of permutation p, resulting in a distinct Young tableau."
];
If[Not[ValueQ[Combinatorica`Contract::usage]],
Combinatorica`Contract::usage = "Contract[g, {x, y}] gives the graph resulting from contracting the pair of vertices {x, y} of graph g."
];
If[Not[ValueQ[CostOfPath::usage]],
CostOfPath::usage = "CostOfPath[g, p] sums up the weights of the edges in graph g defined by the path p."
];
If[Not[ValueQ[CoxeterGraph::usage]],
CoxeterGraph::usage = "CoxeterGraph gives a non-Hamiltonian graph with a high degree of symmetry such that there is a graph automorphism taking any path of length 3 to any other."
];
If[Not[ValueQ[CubeConnectedCycle::usage]],
CubeConnectedCycle::usage = "CubeConnectedCycle[d] returns the graph obtained by replacing each vertex in a d-dimensional hypercube by a cycle of length d. Cube-connected cycles share many properties with hypercubes but have the additional desirable property that for d > 1 every vertex has degree 3."
];
If[Not[ValueQ[CubicalGraph::usage]],
CubicalGraph::usage = "CubicalGraph returns the graph corresponding to the cube, a Platonic solid."
];
If[Not[ValueQ[Cut::usage]],
Cut::usage = "Cut is a tag that can be used in a call to NetworkFlow to tell it to return the minimum cut."
];
If[Not[ValueQ[CycleIndex::usage]],
CycleIndex::usage = "CycleIndex[pg, x] returns the polynomial in x[1], x[2], ..., x[index[g]] that is the cycle index of the permutation group pg. Here index[pg] refers to the length of each permutation in pg."
];
If[Not[ValueQ[Cycle::usage]],
Cycle::usage = "Cycle[n] constructs the cycle on n vertices, the 2-regular connected graph. An option Type that takes on values Directed or Undirected is allowed. The default setting is Type -> Undirected."
];
If[Not[ValueQ[Combinatorica`Cycles::usage]],
Combinatorica`Cycles::usage = "Cycles is an optional argument for the function Involutions."
];
If[Not[ValueQ[CycleStructure::usage]],
CycleStructure::usage = "CycleStructure[p, x] returns the monomial in x[1], x[2], ..., x[Length[p]] that is the cycle structure of the permutation p."
];
If[Not[ValueQ[Cyclic::usage]],
Cyclic::usage = "Cyclic is an argument to the Polya-theoretic functions ListNecklaces, NumberOfNecklace, and NecklacePolynomial, which count or enumerate distinct necklaces. Cyclic refers to the cyclic group acting on necklaces to make equivalent necklaces that can be obtained from each other by rotation."
];
If[Not[ValueQ[Combinatorica`CyclicGroup::usage]],
Combinatorica`CyclicGroup::usage = "CyclicGroup[n] returns the cyclic group of permutations on n symbols."
];
If[Not[ValueQ[CyclicGroupIndex::usage]],
CyclicGroupIndex::usage = "CyclicGroupIndex[n, x] returns the cycle index of the cyclic group on n symbols, expressed as a polynomial in x[1], x[2], ..., x[n]."
];
If[Not[ValueQ[Combinatorica`DeBruijnGraph::usage]],
Combinatorica`DeBruijnGraph::usage = "DeBruijnGraph[m, n] constructs the n-dimensional De Bruijn graph with m symbols for integers m > 0 and n > 1. DeBruijnGraph[alph, n] constructs the n-dimensional De Bruijn graph with symbols from alph. Here alph is nonempty and n > 1 is an integer. In the latter form, the function accepts an option VertexLabel, with default value False, which can be set to True, if users want to associate strings on alph to the vertices as labels."
];
If[Not[ValueQ[Combinatorica`DeBruijnSequence::usage]],
Combinatorica`DeBruijnSequence::usage = "DeBruijnSequence[a, n] returns a De Bruijn sequence on the alphabet a, a shortest sequence in which every string of length n on alphabet a occurs as a contiguous subsequence."
];
If[Not[ValueQ[DegreeSequence::usage]],
DegreeSequence::usage = "DegreeSequence[g] gives the sorted degree sequence of graph g."
];
If[Not[ValueQ[Degrees::usage]],
Degrees::usage = "Degrees[g] returns the degrees of vertex 1, 2, 3, ... in that order."
];
If[Not[ValueQ[DegreesOf2Neighborhood::usage]],
DegreesOf2Neighborhood::usage = "DegreesOf2Neighborhood[g, v] returns the sorted list of degrees of vertices of graph g within a distance of 2 from v."
];
If[Not[ValueQ[DeleteCycle::usage]],
DeleteCycle::usage = "DeleteCycle[g, c] deletes a simple cycle c from graph g. c is specified as a sequence of vertices in which the first and last vertices are identical. g can be directed or undirected. If g does not contain c, it is returned unchanged; otherwise g is returned with c deleted." 
];
If[Not[ValueQ[Combinatorica`DeleteEdge::usage]],
Combinatorica`DeleteEdge::usage = "DeleteEdge[g, e] gives graph g minus e. If g is undirected, then e is treated as an undirected edge, otherwise it is treated as a directed edge. If there are multiple edges between the specified vertices, only one edge is deleted. DeleteEdge[g, e, All] will delete all edges between the specified pair of vertices. Using the tag Directed as a third argument in DeleteEdge is now obsolete."
];
If[Not[ValueQ[DeleteEdges::usage]],
DeleteEdges::usage = "DeleteEdges[g, edgeList] gives graph g minus the list of edges edgeList. If g is undirected, then the edges in edgeList are treated as undirected edges, or otherwise they are treated as directed edges. If there are multiple edges that qualify, then only one edge is deleted. DeleteEdges[g, edgeList, All] will delete all edges that qualify. If only one edge is to be deleted, then edgeList can have the form {s, t}, or otherwise it has the form {{s1, t1}, {s2, t2}, ...}."
];
If[Not[ValueQ[DeleteFromTableau::usage]],
DeleteFromTableau::usage = "DeleteFromTableau[t, r] deletes the last element of row r from Young tableaux t."
];
If[Not[ValueQ[Combinatorica`DeleteVertex::usage]],
Combinatorica`DeleteVertex::usage = "DeleteVertex[g, v] deletes a single vertex v from graph g. Here v is a vertex number."
];
If[Not[ValueQ[DeleteVertices::usage]],
DeleteVertices::usage = "DeleteVertices[g, vList] deletes vertices in vList from graph g. vList has the form {i, j, ...}, where i, j, ... are vertex numbers."
];
If[Not[ValueQ[Combinatorica`DepthFirstTraversal::usage]],
Combinatorica`DepthFirstTraversal::usage = "DepthFirstTraversal[g, v] performs a depth-first traversal of graph g starting from vertex v, and gives a list of vertices in the order in which they were encountered. DepthFirstTraversal[g, v, Edge] returns the edges of the graph that are traversed by the depth-first traversal in the order in which they are traversed. DepthFirstTraversal[g, v, Tree] returns the depth-first tree of the graph."
];
If[Not[ValueQ[DerangementQ::usage]],
DerangementQ::usage = "DerangementQ[p] tests whether permutation p is a derangement, that is, a permutation without a fixed point."
];
If[Not[ValueQ[Derangements::usage]],
Derangements::usage = "Derangements[p] constructs all derangements of permutation p."
];
If[Not[ValueQ[Diameter::usage]],
Diameter::usage = "Diameter[g] gives the diameter of graph g, the maximum length, among all pairs of vertices in g, of a shortest path between each pair."
];
If[Not[ValueQ[Dihedral::usage]],
Dihedral::usage = "Dihedral is an argument to the Polya-theoretic functions ListNecklaces, NumberOfNecklace, and NecklacePolynomial, which count or enumerate distinct necklaces. Dihedral refers to the dihedral group acting on necklaces to make equivalent necklaces that can be obtained from each other by a rotation or a flip."
];
If[Not[ValueQ[Combinatorica`DihedralGroup::usage]],
Combinatorica`DihedralGroup::usage = "DihedralGroup[n] returns the dihedral group on n symbols. Note that the order of this group is 2n."
];
If[Not[ValueQ[DihedralGroupIndex::usage]],
DihedralGroupIndex::usage = "DihedralGroupIndex[n, x] returns the cycle index of the dihedral group on n symbols, expressed as a polynomial in x[1], x[2], ..., x[n]."
];
If[Not[ValueQ[Dijkstra::usage]],
Dijkstra::usage = "Dijkstra[g, v] gives a shortest-path spanning tree and associated distances from vertex v of graph g. The shortest-path spanning tree is given by a list in which element i is the predecessor of vertex i in the shortest-path spanning tree. Dijkstra does not work correctly when the edge weights are negative; BellmanFord should be used in this case."
];
If[Not[ValueQ[DilateVertices::usage]],
DilateVertices::usage = "DilateVertices[v, d] multiplies each coordinate of each vertex position in list v by d, thus dilating the embedding. DilateVertices[g, d] dilates the embedding of graph g by the factor d."
];
If[Not[ValueQ[Directed::usage]],
Directed::usage = "Directed is an option value for Type."
];
$NewMessage[Disk, "usage"]; (* reset the usage of Disk to the system usage *)
If[StringQ[Disk::usage], Disk::usage = StringJoin[Disk::usage, " Disk is also a value taken by the VertexStyle option in ShowGraph."]];
If[Not[ValueQ[Distances::usage]],
Distances::usage = "Distances[g, v] returns the distances in nondecreasing order from vertex v to all vertices in g, treating g as an unweighted graph."
];
If[Not[ValueQ[DistinctPermutations::usage]],
DistinctPermutations::usage = "DistinctPermutations[l] gives all permutations of the multiset described by list l."
];
If[Not[ValueQ[Distribution::usage]],
Distribution::usage = "Distribution[l, set] lists the frequency of each element of set in list l."
];
If[Not[ValueQ[DodecahedralGraph::usage]],
DodecahedralGraph::usage = "DodecahedralGraph returns the graph corresponding to the dodecahedron, a Platonic solid."
];
If[Not[ValueQ[DominatingIntegerPartitionQ::usage]],
DominatingIntegerPartitionQ::usage = "DominatingIntegerPartitionQ[a, b] yields True if integer partition a dominates integer partition b, that is, the sum of a size-t prefix of a is no smaller than the sum of a size-t prefix of b for every t."
];
If[Not[ValueQ[DominationLattice::usage]],
DominationLattice::usage = "DominationLattice[n] returns a Hasse diagram of the partially ordered set on integer partitions of n in which p < q if q dominates p. The function takes two options: Type and VertexLabel, with default values Undirected and False, respectively. When Type is set to Directed, the function produces the underlying directed acyclic graph. When VertexLabel is set to True, labels are produced for the vertices."
];
If[Not[ValueQ[DurfeeSquare::usage]],
DurfeeSquare::usage = "DurfeeSquare[p] gives the number of rows involved in the Durfee square of partition p, the side of the largest-sized square contained within the Ferrers diagram of p."
];
If[Not[ValueQ[Eccentricity::usage]],
Eccentricity::usage = "Eccentricity[g] gives the eccentricity of each vertex v of graph g, the maximum length among all shortest paths from v."
];
If[Not[ValueQ[Edge::usage]],
Edge::usage = "Edge is an optional argument to inform certain functions to work with edges instead of vertices."
];
If[Not[ValueQ[EdgeChromaticNumber::usage]],
EdgeChromaticNumber::usage = "EdgeChromaticNumber[g] gives the fewest number of colors necessary to color each edge of graph g, so that no two edges incident on the same vertex have the same color."
];
If[Not[ValueQ[Combinatorica`EdgeColor::usage]],
Combinatorica`EdgeColor::usage = "EdgeColor is an option that allows the user to associate colors with edges. Black is the default color. EdgeColor can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[EdgeColoring::usage]],
EdgeColoring::usage = "EdgeColoring[g] uses Brelaz's heuristic to find a good, but not necessarily minimal, edge coloring of graph g."
];
If[Not[ValueQ[Combinatorica`EdgeConnectivity::usage]],
Combinatorica`EdgeConnectivity::usage = "EdgeConnectivity[g] gives the minimum number of edges whose deletion from graph g disconnects it. EdgeConnectivity[g, Cut] gives a set of edges of minimum size whose deletion disconnects the graph."
];
If[Not[ValueQ[EdgeDirection::usage]],
EdgeDirection::usage = "EdgeDirection is an option that takes on values True or False allowing the user to specify whether the graph is directed or not. EdgeDirection can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[Combinatorica`EdgeLabel::usage]],
Combinatorica`EdgeLabel::usage = "EdgeLabel is an option that can take on values True or False, allowing the user to associate labels to edges. By default, there are no edge labels. The EdgeLabel option can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[EdgeLabelColor::usage]],
EdgeLabelColor::usage = "EdgeLabelColor is an option that allows the user to associate different colors to edge labels. Black is the default color. EdgeLabelColor can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[EdgeLabelPosition::usage]],
EdgeLabelPosition::usage = "EdgeLabelPosition is an option that allows the user to place an edge label in a certain position relative to the midpoint of the edge. LowerLeft is the default value of this option. EdgeLabelPosition can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[Edges::usage]],
Edges::usage = "Edges[g] gives the list of edges in g. Edges[g, All] gives the edges of g along with the graphics options associated with each edge. Edges[g, EdgeWeight] returns the list of edges in g along with their edge weights."
];
If[Not[ValueQ[Combinatorica`EdgeStyle::usage]],
Combinatorica`EdgeStyle::usage = "EdgeStyle is an option that allows the user to associate different sizes and shapes to edges. A line segment is the default edge. EdgeStyle can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[Combinatorica`EdgeWeight::usage]],
Combinatorica`EdgeWeight::usage = "EdgeWeight is an option that allows the user to associate weights with edges. 1 is the default weight. EdgeWeight can be set as part of the graph data structure."
];
$NewMessage[ Element, "usage"]; (* reset the usage of Element to the System usage *)
If[StringQ[Element::usage], Element::usage = StringJoin[ Element::usage, " The use of the function Element in Combinatorica is now obsolete, though the function call Element[a, p] still gives the pth element of nested list a, where p is a list of indices."]];
If[Not[ValueQ[EmptyGraph::usage]],
EmptyGraph::usage = "EmptyGraph[n] generates an empty graph on n vertices. An option Type that can take on values Directed or Undirected is provided. The default setting is Type -> Undirected." 
];
If[Not[ValueQ[EmptyQ::usage]],
EmptyQ::usage = "EmptyQ[g] yields True if graph g contains no edges."
];
If[Not[ValueQ[EncroachingListSet::usage]],
EncroachingListSet::usage = "EncroachingListSet[p] constructs the encroaching list set associated with permutation p."
];
If[Not[ValueQ[EquivalenceClasses::usage]],
EquivalenceClasses::usage = "EquivalenceClasses[r] identifies the equivalence classes among the elements of matrix r."
];
If[Not[ValueQ[EquivalenceRelationQ::usage]],
EquivalenceRelationQ::usage = "EquivalenceRelationQ[r] yields True if the matrix r defines an equivalence relation. EquivalenceRelationQ[g] tests whether the adjacency matrix of graph g defines an equivalence relation."
];
If[Not[ValueQ[Equivalences::usage]],
Equivalences::usage = "Equivalences[g, h] lists the vertex equivalence classes between graphs g and h defined by their vertex degrees. Equivalences[g] lists the vertex equivalences for graph g defined by the vertex degrees. Equivalences[g, h, f1, f2, ...] and Equivalences[g, f1, f2, ...] can also be used, where f1, f2, ... are functions that compute other vertex invariants. It is expected that for each function fi, the call fi[g, v] returns the corresponding invariant at vertex v in graph g. The functions f1, f2, ... are evaluated in order, and the evaluation stops either when all functions have been evaluated or when an empty equivalence class is found. Three vertex invariants, DegreesOf2Neighborhood, NumberOf2Paths, and Distances are Combinatorica functions and can be used to refine the equivalences."
];
If[Not[ValueQ[Euclidean::usage]],
Euclidean::usage = "Euclidean is an option for SetEdgeWeights."
];
If[Not[ValueQ[Eulerian::usage]],
Eulerian::usage = "Eulerian[n, k] gives the number of permutations of length n with k runs."
];
If[Not[ValueQ[EulerianCycle::usage]],
EulerianCycle::usage = "EulerianCycle[g] finds an Eulerian cycle of g if one exists."
];
If[Not[ValueQ[EulerianQ::usage]],
EulerianQ::usage = "EulerianQ[g] yields True if graph g is Eulerian, meaning there exists a tour that includes each edge exactly once."
];
If[Not[ValueQ[ExactRandomGraph::usage]],
ExactRandomGraph::usage = "ExactRandomGraph[n, e] constructs a random labeled graph of exactly e edges and n vertices."
];
If[Not[ValueQ[ExpandGraph::usage]],
ExpandGraph::usage = "ExpandGraph[g, n] expands graph g to n vertices by adding disconnected vertices. This is obsolete; use AddVertices[g, n] instead."
];
If[Not[ValueQ[ExtractCycles::usage]],
ExtractCycles::usage = "ExtractCycles[g] gives a maximal list of edge-disjoint cycles in graph g."
];
If[Not[ValueQ[FerrersDiagram::usage]],
FerrersDiagram::usage = "FerrersDiagram[p] draws a Ferrers diagram of integer partition p."
];
If[Not[ValueQ[Combinatorica`FindCycle::usage]],
Combinatorica`FindCycle::usage = "FindCycle[g] finds a list of vertices that define a cycle in graph g."
];
If[Not[ValueQ[FindSet::usage]],
FindSet::usage = "FindSet[n, s] gives the root of the set containing n in union-find data structure s."
];
If[Not[ValueQ[FiniteGraphs::usage]],
FiniteGraphs::usage = "FiniteGraphs produces a convenient list of all the interesting, finite, parameterless graphs built into Combinatorica."
];
If[Not[ValueQ[FirstLexicographicTableau::usage]],
FirstLexicographicTableau::usage = "FirstLexicographicTableau[p] constructs the first Young tableau with shape described by partition p."
];
If[Not[ValueQ[FolkmanGraph::usage]],
FolkmanGraph::usage = "FolkmanGraph returns a smallest graph that is edge-transitive but not vertex-transitive."
];
If[Not[ValueQ[FranklinGraph::usage]],
FranklinGraph::usage = "FranklinGraph returns a 12-vertex graph that represents a 6-chromatic map on the Klein bottle. It is the sole counterexample to Heawood's map coloring conjecture."
];
If[Not[ValueQ[FromAdjacencyLists::usage]],
FromAdjacencyLists::usage = "FromAdjacencyLists[l] constructs an edge list representation for a graph from the given adjacency lists l, using a circular embedding. FromAdjacencyLists[l, v] uses v as the embedding for the resulting graph. An option called Type that takes on the values Directed or Undirected can be used to affect the type of graph produced. The default value of Type is Undirected."
];
If[Not[ValueQ[FromAdjacencyMatrix::usage]],
FromAdjacencyMatrix::usage = "FromAdjacencyMatrix[m] constructs a graph from a given adjacency matrix m, using a circular embedding. FromAdjacencyMatrix[m, v] uses v as the embedding for the resulting graph. An option Type that takes on the values Directed or Undirected can be used to affect the type of graph produced. The default value of Type is Undirected. FromAdjacencyMatrix[m, EdgeWeight] interprets the entries in m as edge weights, with infinity representing missing edges, and from this constructs a weighted graph using a circular embedding. FromAdjacencyMatrix[m, v, EdgeWeight] uses v as the embedding for the resulting graph. The option Type can be used along with the EdgeWeight tag."
];
If[Not[ValueQ[Combinatorica`FromCycles::usage]],
Combinatorica`FromCycles::usage = "FromCycles[{c1, c2, ...}] gives the permutation that has the given cycle structure."
];
If[Not[ValueQ[FromInversionVector::usage]],
FromInversionVector::usage = "FromInversionVector[v] reconstructs the unique permutation with inversion vector v."
];
If[Not[ValueQ[FromOrderedPairs::usage]],
FromOrderedPairs::usage = "FromOrderedPairs[l] constructs an edge list representation from a list of ordered pairs l, using a circular embedding. FromOrderedPairs[l, v] uses v as the embedding for the resulting graph. The option Type that takes on values Undirected or Directed can be used to affect the kind of graph produced. The default value of Type is Directed. Type -> Undirected results in the underlying undirected graph."
];
If[Not[ValueQ[FromUnorderedPairs::usage]],
FromUnorderedPairs::usage = "FromUnorderedPairs[l] constructs an edge list representation from a list of unordered pairs l, using a circular embedding. FromUnorderedPairs[l, v] uses v as the embedding for the resulting graph. The option Type that takes on values Undirected or Directed can be used to affect the kind of graph produced."
];
If[Not[ValueQ[FruchtGraph::usage]],
FruchtGraph::usage = "FruchtGraph returns the smallest 3-regular graph whose automorphism group consists of only the identity."
];
If[Not[ValueQ[FunctionalGraph::usage]],
FunctionalGraph::usage = "FunctionalGraph[f, v] takes a set v and a function f from v to v and constructs a directed graph with vertex set v and edges (x, f(x)) for each x in v. FunctionalGraph[f, v], where f is a list of functions, constructs a graph with vertex set v and edge set (x, fi(x)) for every fi in f. An option called Type that takes on the values Directed and Undirected is allowed. Type -> Directed is the default, while Type -> Undirected returns the corresponding underlying undirected graph. FunctionalGraph[f, n] takes a nonnegative integer n and a function f from {0,1,..., n-1} onto itself and produces the directed graph with vertex set {0, 1,..., n-1} and edge set {x, f(x)} for each vertex x." 
];
If[Not[ValueQ[GeneralizedPetersenGraph::usage]],
GeneralizedPetersenGraph::usage = "GeneralizedPetersenGraph[n, k] returns the generalized Petersen graph, for integers n > 1 and k > 0, which is the graph with vertices {u1, u2, ..., un} and {v1, v2, ..., vn} and edges {ui, u(i+1)}, {vi, v(i+k)}, and {ui, vi}. The Petersen graph is identical to the generalized Petersen graph with n = 5 and k = 2."
];
If[Not[ValueQ[GetEdgeLabels::usage]],
GetEdgeLabels::usage = "GetEdgeLabels[g] returns the list of labels of the edges of g. GetEdgeLabels[g, es] returns the list of labels in graph g of the edges in es." 
];
If[Not[ValueQ[GetEdgeWeights::usage]],
GetEdgeWeights::usage = "GetEdgeWeights[g] returns the list of weights of the edges of g. GetEdgeWeights[g, es] returns the list of weights in graph g of the edges in es."
];
If[Not[ValueQ[GetVertexLabels::usage]],
GetVertexLabels::usage = "GetVertexLabels[g] returns the list of labels of vertices of g. GetVertexLabels[g, vs] returns the list of labels in graph g of the vertices specified in list vs."
];
If[Not[ValueQ[GetVertexWeights::usage]],
GetVertexWeights::usage = "GetVertexWeights[g] returns the list of weights of vertices of g. GetVertexWeights[g, vs] returns the list of weights in graph g of the vertices in vs."
];
If[Not[ValueQ[Girth::usage]],
Girth::usage = "Girth[g] gives the length of a shortest cycle in a simple graph g."
];
If[Not[ValueQ[Combinatorica`Graph::usage]],
Combinatorica`Graph::usage = "Graph[e, v, opts] represents a graph object where e is the list of edges annotated with graphics options, v is a list of vertices annotated with graphics options, and opts is a set of global graph options. e has the form {{{i1, j1}, opts1}, {{i2, j2}, opts2},...}, where {i1, j1}, {i2, j2},... are edges of the graph and opts1, opts2,... are options that respectively apply to these edges. v has the form {{{x1, y1}, opts1}, {{x2, y2}, opts2},...}, where {x1, y1}, {x2, y2},... respectively denote the coordinates in the plane of vertex 1, vertex 2,... and opts1, opts2,... are options that respectively apply to these vertices. Permitted edge options are EdgeWeight, EdgeColor, EdgeStyle, EdgeLabel, EdgeLabelColor, and EdgeLabelPosition. Permitted vertex options are VertexWeight, VertexColor, VertexStyle, VertexNumber, VertexNumberColor, VertexNumberPosition, VertexLabel, VertexLabelColor, and VertexLabelPosition. The third item in a Graph object is opts, a sequence of zero or more global options that apply to all vertices or all edges or to the graph as a whole. All of the edge options and vertex options can be used as global options also. If a global option and a local edge option or vertex option differ, then the local edge or vertex option is used for that particular edge or vertex. In addition to these options, the following two options can also be specified as part of the global options: LoopPosition and EdgeDirection. Furthermore, all the options of the Mathematica function Plot can be used as global options in a Graph object. These can be used to specify how the graph looks when it is drawn. These can be used to affect the look of arrows that represent directed edges. See the usage message of individual options to find out more about values these options can take on. Whether a graph is undirected or directed is given by the option EdgeDirection. This has default value False. For undirected graphs, the edges {i1, j1}, {i2, j2},... have to satisfy i1 <= j1, i2 <= j2,... and for directed graphs the edges {i1, j1}, {i2, j2},... are treated as ordered pairs, each specifying the direction of the edge as well. Graph[name] constructs a graph from GraphData information for the string name."
];
If[Not[ValueQ[Combinatorica`GraphCenter::usage]],
Combinatorica`GraphCenter::usage = "GraphCenter[g] gives a list of the vertices of graph g with minimum eccentricity."
];
If[Not[ValueQ[Combinatorica`GraphComplement::usage]],
Combinatorica`GraphComplement::usage = "GraphComplement[g] gives the complement of graph g."
];
If[Not[ValueQ[Combinatorica`GraphDifference::usage]],
Combinatorica`GraphDifference::usage = "GraphDifference[g, h] constructs the graph resulting from subtracting the edges of graph h from the edges of graph g."
];
If[Not[ValueQ[GraphicQ::usage]],
GraphicQ::usage = "GraphicQ[s] yields True if the list of integers s is a graphic sequence, and thus represents a degree sequence of some graph."
];
If[Not[ValueQ[Combinatorica`GraphIntersection::usage]],
Combinatorica`GraphIntersection::usage = "GraphIntersection[g1, g2, ...] constructs the graph defined by the edges that are in all the graphs g1, g2, ...."
];
If[Not[ValueQ[Combinatorica`GraphJoin::usage]],
Combinatorica`GraphJoin::usage = "GraphJoin[g1, g2, ...] constructs the join of graphs g1, g2, and so on. This is the graph obtained by adding all possible edges between different graphs to the graph union of g1, g2, ...."
];
If[Not[ValueQ[GraphOptions::usage]],
GraphOptions::usage = "GraphOptions[g] returns the display options associated with g. GraphOptions[g, v] returns the display options associated with vertex v in g. GraphOptions[g, {u, v}] returns the display options associated with edge {u, v} in g."
];
If[Not[ValueQ[GraphPolynomial::usage]],
GraphPolynomial::usage = "GraphPolynomial[n, x] returns a polynomial in x in which the coefficient of x^m is the number of nonisomorphic graphs with n vertices and m edges. GraphPolynomial[n, x, Directed] returns a polynomial in x in which the coefficient of x^m is the number of nonisomorphic directed graphs with n vertices and m edges."
];
If[Not[ValueQ[Combinatorica`GraphPower::usage]],
Combinatorica`GraphPower::usage = "GraphPower[g, k] gives the kth power of graph g. This is the graph whose vertex set is identical to the vertex set of g and that contains an edge between vertices i and j for each path in g between vertices i and j of length at most k."
];
If[Not[ValueQ[Combinatorica`GraphProduct::usage]],
Combinatorica`GraphProduct::usage = "GraphProduct[g1, g2, ...] constructs the product of graphs g1, g2, and so forth."
];
If[Not[ValueQ[Combinatorica`GraphSum::usage]],
Combinatorica`GraphSum::usage = "GraphSum[g1, g2, ...] constructs the graph resulting from joining the edge lists of graphs g1, g2, and so forth."
];
If[Not[ValueQ[Combinatorica`GraphUnion::usage]],
Combinatorica`GraphUnion::usage = "GraphUnion[g1, g2, ...] constructs the union of graphs g1, g2, and so forth. GraphUnion[n, g] constructs n copies of graph g, for any nonnegative integer n."
];
If[Not[ValueQ[GrayCode::usage]],
GrayCode::usage = "GrayCode[l] constructs a binary reflected Gray code on set l. GrayCode is obsolete, so use GrayCodeSubsets instead."
];
If[Not[ValueQ[GrayCodeKSubsets::usage]],
GrayCodeKSubsets::usage = "GrayCodeKSubsets[l, k] generates k-subsets of l in Gray code order."
];
If[Not[ValueQ[GrayCodeSubsets::usage]],
GrayCodeSubsets::usage = "GrayCodeSubsets[l] constructs a binary reflected Gray code on set l."
];
If[Not[ValueQ[GrayGraph::usage]],
GrayGraph::usage = "GrayGraph returns a 3-regular, 54-vertex graph that is edge-transitive but not vertex-transitive; the smallest known such example." 
];
If[Not[ValueQ[Greedy::usage]],
Greedy::usage = "Greedy is a value that the option Algorithm can take in calls to functions such as VertexCover, telling the function to use a greedy algorithm."
];
If[Not[ValueQ[GreedyVertexCover::usage]],
GreedyVertexCover::usage = "GreedyVertexCover[g] returns a vertex cover of graph g constructed using the greedy algorithm. This is a natural heuristic for constructing a vertex cover, but it can produce poor vertex covers."
];
If[Not[ValueQ[Combinatorica`GridGraph::usage]],
Combinatorica`GridGraph::usage = "GridGraph[n, m] constructs an n*m grid graph, the product of paths on n and m vertices. GridGraph[p, q, r] constructs a p*q*r grid graph, the product of GridGraph[p, q] and a path of length r."
];
If[Not[ValueQ[GroetzschGraph::usage]],
GroetzschGraph::usage = "GroetzschGraph returns the smallest triangle-free graph with chromatic number 4. This is identical to MycielskiGraph[4]."
];
If[Not[ValueQ[GrotztschGraph::usage]],
GrotztschGraph::usage = "GrotztschGraph is an obsolete name for GroetzschGraph, present for compatability purposes.";
];
If[Not[ValueQ[HamiltonianCycle::usage]],
HamiltonianCycle::usage = "HamiltonianCycle[g] finds a Hamiltonian cycle in graph g if one exists. HamiltonianCycle[g, All] gives all Hamiltonian cycles of graph g."
];
If[Not[ValueQ[HamiltonianPath::usage]],
HamiltonianPath::usage = "HamiltonianPath[g] finds a Hamiltonian path in graph g if one exists. HamiltonianPath[g, All] gives all Hamiltonian paths of graph g."
];
If[Not[ValueQ[HamiltonianQ::usage]],
HamiltonianQ::usage = "HamiltonianQ[g] yields True if there exists a Hamiltonian cycle in graph g, or in other words, if there exists a cycle that visits each vertex exactly once."
];
If[Not[ValueQ[Harary::usage]],
Harary::usage = "Harary[k, n] constructs the minimal k-connected graph on n vertices."
];
If[Not[ValueQ[HasseDiagram::usage]],
HasseDiagram::usage = "HasseDiagram[g] constructs a Hasse diagram of the relation defined by directed acyclic graph g."
];
If[Not[ValueQ[Heapify::usage]],
Heapify::usage = "Heapify[p] builds a heap from permutation p."
];
If[Not[ValueQ[HeapSort::usage]],
HeapSort::usage = "HeapSort[l] performs a heap sort on the items of list l."
];
If[Not[ValueQ[HeawoodGraph::usage]],
HeawoodGraph::usage = "HeawoodGraph returns a smallest (6, 3)-cage, a 3-regular graph with girth 6."
];
If[Not[ValueQ[HerschelGraph::usage]],
HerschelGraph::usage = "HerschelGraph returns a graph object that represents a Herschel graph."
];
If[Not[ValueQ[HideCycles::usage]],
HideCycles::usage = "HideCycles[c] canonically encodes the cycle structure c into a unique permutation."
];
If[Not[ValueQ[Highlight::usage]],
Highlight::usage = "Highlight[g, p] displays g with elements in p highlighted. The second argument p has the form {s1, s2,...}, where the sis are disjoint subsets of vertices and edges of g. The options, HighlightedVertexStyle, HighlightedEdgeStyle, HighlightedVertexColors, and HighlightedEdgeColors are used to determine the appearance of the highlighted elements of the graph. The default settings of the style options are HighlightedVertexStyle->Disk[Large] and HighlightedEdgeStyle->Thick. The options HighlightedVertexColors and HighlightedEdgeColors are both set to {Black, Red, Blue, Green, Yellow, Purple, Brown, Orange, Olive, Pink, DeepPink, DarkGreen, Maroon, Navy}. The colors are chosen from the palette of colors with color 1 used for s1, color 2 used for s2, and so on. If there are more parts than colors, then the colors are used cyclically. The function permits all the options that SetGraphOptions permits, for example, VertexColor, VertexStyle, EdgeColor, and EdgeStyle. These options can be used to control the appearance of the non-highlighted vertices and edges."
];
If[Not[ValueQ[HighlightedEdgeColors::usage]],
HighlightedEdgeColors::usage = "HighlightedEdgeColors is an option to Highlight that determines which colors are used for the highlighted edges."
];
If[Not[ValueQ[HighlightedEdgeStyle::usage]],
HighlightedEdgeStyle::usage = "HighlightedEdgeStyle is an option to Highlight that determines how the highlighted edges are drawn."
];
If[Not[ValueQ[HighlightedVertexColors::usage]],
HighlightedVertexColors::usage = "HighlightedVertexColors is an option to Highlight that determines which colors are used for the highlighted vertices."
];
If[Not[ValueQ[HighlightedVertexStyle::usage]],
HighlightedVertexStyle::usage = "HighlightedVertexStyle is an option to Highlight that determines how the highlighted vertices are drawn."
];
If[Not[ValueQ[Hypercube::usage]],
Hypercube::usage = "Hypercube[n] constructs an n-dimensional hypercube."
];
If[Not[ValueQ[IcosahedralGraph::usage]],
IcosahedralGraph::usage = "IcosahedralGraph returns the graph corresponding to the icosahedron, a Platonic solid."
];
If[Not[ValueQ[IdenticalQ::usage]],
IdenticalQ::usage = "IdenticalQ[g, h] yields True if graphs g and h have identical edge lists, even though the associated graphics information need not be the same."
];
If[Not[ValueQ[IdentityPermutation::usage]],
IdentityPermutation::usage = "IdentityPermutation[n] gives the size-n identity permutation."
];
If[Not[ValueQ[Combinatorica`IncidenceMatrix::usage]],
Combinatorica`IncidenceMatrix::usage = "IncidenceMatrix[g] returns the (0, 1)-matrix of graph g, which has a row for each vertex and a column for each edge and (v, e) = 1 if and only if vertex v is incident upon edge e. For a directed graph, (v, e) = 1 if edge e is outgoing from v."
];
If[Not[ValueQ[InDegree::usage]],
InDegree::usage = "InDegree[g, n] returns the in-degree of vertex n in directed graph g. InDegree[g] returns the sequence of in-degrees of the vertices in directed graph g."
];
If[Not[ValueQ[IndependentSetQ::usage]],
IndependentSetQ::usage = "IndependentSetQ[g, i] yields True if the vertices in list i define an independent set in graph g."
];
If[Not[ValueQ[Index::usage]],
Index::usage = "Index[p] gives the index of permutation p, the sum of all subscripts j such that p[j] is greater than p[j+1]."
];
If[Not[ValueQ[InduceSubgraph::usage]],
InduceSubgraph::usage = "InduceSubgraph[g, s] constructs the subgraph of graph g induced by the list of vertices s."
];
If[Not[ValueQ[InitializeUnionFind::usage]],
InitializeUnionFind::usage = "InitializeUnionFind[n] initializes a union-find data structure for n elements."
];
If[Not[ValueQ[InsertIntoTableau::usage]],
InsertIntoTableau::usage = "InsertIntoTableau[e, t] inserts integer e into Young tableau t using the bumping algorithm. InsertIntoTableau[e, t, All] inserts e into Young tableau t and returns the new tableau as well as the row whose size is expanded as a result of the insertion."
];
If[Not[ValueQ[IntervalGraph::usage]],
IntervalGraph::usage = "IntervalGraph[l] constructs the interval graph defined by the list of intervals l."
];
If[Not[ValueQ[Invariants::usage]],
Invariants::usage = "Invariants is an option to the functions Isomorphism and IsomorphicQ that informs these functions about which vertex invariants to use in computing equivalences between vertices."
];
If[Not[ValueQ[InversePermutation::usage]],
InversePermutation::usage = "InversePermutation[p] yields the multiplicative inverse of permutation p."
];
If[Not[ValueQ[InversionPoset::usage]],
InversionPoset::usage = "InversionPoset[n] returns a Hasse diagram of the partially ordered set on size-n permutations in which p < q if q can be obtained from p by an adjacent transposition that places the larger element before the smaller. The function takes two options: Type and VertexLabel, with default values Undirected and False, respectively. When Type is set to Directed, the function produces the underlying directed acyclic graph. When VertexLabel is set to True, labels are produced for the vertices."
];
If[Not[ValueQ[Inversions::usage]],
Inversions::usage = "Inversions[p] counts the number of inversions in permutation p."
];
If[Not[ValueQ[InvolutionQ::usage]],
InvolutionQ::usage = "InvolutionQ[p] yields True if permutation p is its own inverse."
];
If[Not[ValueQ[Involutions::usage]],
Involutions::usage = "Involutions[l] gives the list of involutions of the elements in the list l. Involutions[l, Cycles] gives the involutions in their cycle representation. Involution[n] gives size-n involutions. Involutions[n, Cycles] gives size-n involutions in their cycle representation."
];
If[Not[ValueQ[IsomorphicQ::usage]],
IsomorphicQ::usage = "IsomorphicQ[g, h] yields True if graphs g and h are isomorphic. This function takes an option Invariants -> {f1, f2, ...}, where f1, f2, ... are functions that are used to compute vertex invariants. These functions are used in the order in which they are specified. The default value of Invariants is {DegreesOf2Neighborhood, NumberOf2Paths, Distances}."
];
If[Not[ValueQ[Isomorphism::usage]],
Isomorphism::usage = "Isomorphism[g, h] gives an isomorphism between graphs g and h if one exists. Isomorphism[g, h, All] gives all isomorphisms between graphs g and h. Isomorphism[g] gives the automorphism group of g. This function takes an option Invariants -> {f1, f2, ...}, where f1, f2, ... are functions that are used to compute vertex invariants. These functions are used in the order in which they are specified. The default value of Invariants is {DegreesOf2Neighborhood, NumberOf2Paths, Distances}."
];
If[Not[ValueQ[IsomorphismQ::usage]],
IsomorphismQ::usage = "IsomorphismQ[g, h, p] tests if permutation p defines an isomorphism between graphs g and h."
];
If[Not[ValueQ[Josephus::usage]],
Josephus::usage = "Josephus[n, m] generates the inverse of the permutation defined by executing every mth member in a circle of n members."
];
If[Not[ValueQ[KnightsTourGraph::usage]],
KnightsTourGraph::usage = "KnightsTourGraph[m, n] returns a graph with m*n vertices in which each vertex represents a square in an m x n chessboard and each edge corresponds to a legal move by a knight from one square to another."
];
If[Not[ValueQ[KSetPartitions::usage]],
KSetPartitions::usage = "KSetPartitions[set, k] returns the list of set partitions of set with k blocks. KSetPartitions[n, k] returns the list of set partitions of {1, 2, ..., n} with k blocks. If all set partitions of a set are needed, use the function SetPartitions."
];
If[Not[ValueQ[KSubsetGroup::usage]],
KSubsetGroup::usage = "KSubsetGroup[pg, s] returns the group induced by a permutation group pg on the set s of k-subsets of [n], where n is the index of pg. The optional argument Type can be Ordered or Unordered and depending on the value of Type s is treated as a set of k-subsets or k-tuples."
];
If[Not[ValueQ[KSubsetGroupIndex::usage]],
KSubsetGroupIndex::usage = "KSubsetGroupIndex[g, s, x] returns the cycle index of the k-subset group on s expressed as a polynomial in x[1], x[2], .... This function also takes the optional argument Type that tells the function whether the elements of s should be treated as sets or tuples."
];
If[Not[ValueQ[KSubsets::usage]],
KSubsets::usage = "KSubsets[l, k] gives all subsets of set l containing exactly k elements, ordered lexicographically."
];
$NewMessage[ K, "usage"]; (* reset the usage of K to the System usage *)
If[StringQ[K::usage], K::usage = StringJoin[ K::usage, " The use of K to create a complete graph is obsolete. Use CompleteGraph to create a complete graph."]];
If[Not[ValueQ[LabeledTreeToCode::usage]],
LabeledTreeToCode::usage = "LabeledTreeToCode[g] reduces the tree g to its Prufer code."
];
$NewMessage[Large, "usage"];
If[StringQ[Large::usage],
Large::usage = Large::usage <> " Large is also a symbol in Combinatorica used to denote the size of the object that represents a vertex. The option VertexStyle can be set to Disk[Large] or Box[Large] either inside the graph data structure or in ShowGraph."
];
If[Not[ValueQ[LastLexicographicTableau::usage]],
LastLexicographicTableau::usage = "LastLexicographicTableau[p] constructs the last Young tableau with shape described by partition p."
];
If[Not[ValueQ[Level::usage]],
Level::usage = "Level is an option for the function BreadthFirstTraversal that makes the function return levels of vertices."
];
If[Not[ValueQ[LeviGraph::usage]],
LeviGraph::usage = "LeviGraph returns the unique (8, 3)-cage, a 3-regular graph whose girth is 8."
];
If[Not[ValueQ[LexicographicPermutations::usage]],
LexicographicPermutations::usage = "LexicographicPermutations[l] constructs all permutations of list l in lexicographic order."
];
If[Not[ValueQ[LexicographicSubsets::usage]],
LexicographicSubsets::usage = "LexicographicSubsets[l] gives all subsets of set l in lexicographic order. LexicographicSubsets[n] returns all subsets of {1, 2,..., n} in lexicographic order."
];
If[Not[ValueQ[Combinatorica`LineGraph::usage]],
Combinatorica`LineGraph::usage = "LineGraph[g] constructs the line graph of graph g."
];
If[Not[ValueQ[ListGraphs::usage]],
ListGraphs::usage = "ListGraphs[n, m] returns all nonisomorphic undirected graphs with n vertices and m edges. ListGraphs[n, m, Directed] returns all nonisomorphic directed graphs with n vertices and m edges. ListGraphs[n] returns all nonisomorphic undirected graphs with n vertices. ListGraphs[n, Directed] returns all nonisomorphic directed graphs with n vertices."
];
If[Not[ValueQ[ListNecklaces::usage]],
ListNecklaces::usage = "ListNecklaces[n, c, Cyclic] returns all distinct necklaces whose beads are colored by colors from c. Here c is a list of n, not necessarily distinct colors, and two colored necklaces are considered equivalent if one can be obtained by rotating the other. ListNecklaces[n, c, Dihedral] is similar except that two necklaces are considered equivalent if one can be obtained from the other by a rotation or a flip."
];
If[Not[ValueQ[LNorm::usage]],
LNorm::usage = "LNorm[p] is a value that the option WeightingFunction, used in the function SetEdgeWeights, can take. Here p can be any integer or Infinity."
];
If[Not[ValueQ[LongestIncreasingSubsequence::usage]],
LongestIncreasingSubsequence::usage = "LongestIncreasingSubsequence[p] finds the longest increasing scattered subsequence of permutation p."
];
If[Not[ValueQ[LoopPosition::usage]],
LoopPosition::usage = "LoopPosition is an option to ShowGraph whose values tell ShowGraph where to position a loop around a vertex. This option can take on values UpperLeft, UpperRight, LowerLeft, and LowerRight."
];
If[Not[ValueQ[LowerLeft::usage]],
LowerLeft::usage = "LowerLeft is a value that options VertexNumberPosition, VertexLabelPosition, and EdgeLabelPosition can take on in ShowGraph."
];
If[Not[ValueQ[LowerRight::usage]],
LowerRight::usage = "LowerRight is a value that options VertexNumberPosition, VertexLabelPosition, and EdgeLabelPosition can take on in ShowGraph."
];
If[Not[ValueQ[M::usage]],
M::usage = "M[g] gives the number of edges in the graph g. M[g, Directed] is obsolete because M[g] works for directed as well as undirected graphs."
];
If[Not[ValueQ[MakeDirected::usage]],
MakeDirected::usage = "MakeDirected[g] constructs a directed graph from a given undirected graph g by replacing each undirected edge in g by two directed edges pointing in opposite directions. The local options associated with edges are not inherited by the corresponding directed edges. Calling the function with the tag All, as MakeDirected[g, All], ensures that local options associated with each edge are inherited by both corresponding directed edges."
];
If[Not[ValueQ[MakeGraph::usage]],
MakeGraph::usage = "MakeGraph[v, f] constructs the graph whose vertices correspond to v and edges between pairs of vertices x and y in v for which the binary relation defined by the Boolean function f is True. MakeGraph takes two options, Type and VertexLabel. Type can be set to Directed or Undirected and this tells MakeGraph whether to construct a directed or an undirected graph. The default setting is Directed. VertexLabel can be set to True or False, with False being the default setting. Using VertexLabel -> True assigns labels derived from v to the vertices of the graph."
];
If[Not[ValueQ[MakeSimple::usage]],
MakeSimple::usage = "MakeSimple[g] gives the undirected graph, free of multiple edges and self-loops derived from graph g."
];
If[Not[ValueQ[MakeUndirected::usage]],
MakeUndirected::usage = "MakeUndirected[g] gives the underlying undirected graph of the given directed graph g."
];
If[Not[ValueQ[MaximalMatching::usage]],
MaximalMatching::usage = "MaximalMatching[g] gives the list of edges associated with a maximal matching of graph g."
];
If[Not[ValueQ[MaximumAntichain::usage]],
MaximumAntichain::usage = "MaximumAntichain[g] gives a largest set of unrelated vertices in partial order g."
];
If[Not[ValueQ[MaximumClique::usage]],
MaximumClique::usage = "MaximumClique[g] finds a largest clique in graph g. MaximumClique[g, k] returns a k-clique, if such a thing exists in g; otherwise it returns {}."
];
If[Not[ValueQ[MaximumIndependentSet::usage]],
MaximumIndependentSet::usage = "MaximumIndependentSet[g] finds a largest independent set of graph g."
];
If[Not[ValueQ[MaximumSpanningTree::usage]],
MaximumSpanningTree::usage = "MaximumSpanningTree[g] uses Kruskal's algorithm to find a maximum spanning tree of graph g."
];
If[Not[ValueQ[McGeeGraph::usage]],
McGeeGraph::usage = "McGeeGraph returns the unique (7, 3)-cage, a 3-regular graph with girth 7."
];
If[Not[ValueQ[MeredithGraph::usage]],
MeredithGraph::usage = "MeredithGraph returns a 4-regular, 4-connected graph that is not Hamiltonian, providing a counterexample to a conjecture by C. St. J. A. Nash-Williams."
];
If[Not[ValueQ[MinimumChainPartition::usage]],
MinimumChainPartition::usage = "MinimumChainPartition[g] partitions partial order g into a minimum number of chains."
];
If[Not[ValueQ[MinimumChangePermutations::usage]],
MinimumChangePermutations::usage = "MinimumChangePermutations[l] constructs all permutations of list l such that adjacent permutations differ by only one transposition."
];
If[Not[ValueQ[MinimumSpanningTree::usage]],
MinimumSpanningTree::usage = "MinimumSpanningTree[g] uses Kruskal's algorithm to find a minimum spanning tree of graph g."
];
If[Not[ValueQ[MinimumVertexColoring::usage]],
MinimumVertexColoring::usage = "MinimumVertexColoring[g] returns a minimum vertex coloring of g. MinimumVertexColoring[g, k] returns a k-coloring of g, if one exists."
];
If[Not[ValueQ[MinimumVertexCover::usage]],
MinimumVertexCover::usage = "MinimumVertexCover[g] finds a minimum vertex cover of graph g. For bipartite graphs, the function uses the polynomial-time Hungarian algorithm. For everything else, the function uses brute force."
];
If[Not[ValueQ[MultipleEdgesQ::usage]],
MultipleEdgesQ::usage = "MultipleEdgesQ[g] yields True if g has multiple edges between pairs of vertices. It yields False otherwise."
];
If[Not[ValueQ[MultiplicationTable::usage]],
MultiplicationTable::usage = "MultiplicationTable[l, f] constructs the complete transition table defined by the binary relation function f on the elements of list l."
];
If[Not[ValueQ[MycielskiGraph::usage]],
MycielskiGraph::usage = "MycielskiGraph[k] returns a triangle-free graph with chromatic number k, for any positive integer k."
];
If[Not[ValueQ[NecklacePolynomial::usage]],
NecklacePolynomial::usage = "NecklacePolynomial[n, c, Cyclic] returns a polynomial in the colors in c whose coefficients represent numbers of ways of coloring an n-bead necklace with colors chosen from c, assuming that two colorings are equivalent if one can be obtained from the other by a rotation. NecklacePolynomial[n, c, Dihedral] is different in that it considers two colorings equivalent if one can be obtained from the other by a rotation or a flip or both."
];
If[Not[ValueQ[Neighborhood::usage]],
Neighborhood::usage = "Neighborhood[g, v, k] returns the subset of vertices in g that are at a distance of k or less from vertex v. Neighborhood[al, v, k] behaves identically, except that it takes as input an adjacency list al."
];
If[Not[ValueQ[NetworkFlow::usage]],
NetworkFlow::usage = "NetworkFlow[g, source, sink] returns the value of a maximum flow through graph g from source to sink. NetworkFlow[g, source, sink, Edge] returns the edges in g that have positive flow along with their flows in a maximum flow from source to sink. NetworkFlow[g, source, sink, Cut] returns a minimum cut between source and sink. NetworkFlow[g, source, sink, All] returns the adjacency list of g along with flows on each edge in a maximum flow from source to sink. g can be a directed or an undirected graph."
];
If[Not[ValueQ[NetworkFlowEdges::usage]],
NetworkFlowEdges::usage = "NetworkFlowEdges[g, source, sink] returns the edges of the graph with positive flow, showing the distribution of a maximum flow from source to sink in graph g. This is obsolete, and NetworkFlow[g, source, sink, Edge] should be used instead."
];
If[Not[ValueQ[NextBinarySubset::usage]],
NextBinarySubset::usage = "NextBinarySubset[l, s] constructs the subset of l following subset s in the order obtained by interpreting subsets as binary string representations of integers."
];
If[Not[ValueQ[NextComposition::usage]],
NextComposition::usage = "NextComposition[l] constructs the integer composition that follows l in a canonical order."
];
If[Not[ValueQ[NextGrayCodeSubset::usage]],
NextGrayCodeSubset::usage = "NextGrayCodeSubset[l, s] constructs the successor of s in the Gray code of set l."
];
If[Not[ValueQ[NextKSubset::usage]],
NextKSubset::usage = "NextKSubset[l, s] gives the k-subset of list l, following the k-subset s in lexicographic order."
];
If[Not[ValueQ[NextLexicographicSubset::usage]],
NextLexicographicSubset::usage = "NextLexicographicSubset[l, s] gives the lexicographic successor of subset s of set l."
];
If[Not[ValueQ[NextPartition::usage]],
NextPartition::usage = "NextPartition[p] gives the integer partition following p in reverse lexicographic order."
];
If[Not[ValueQ[NextPermutation::usage]],
NextPermutation::usage = "NextPermutation[p] gives the permutation following p in lexicographic order."
];
If[Not[ValueQ[NextSubset::usage]],
NextSubset::usage = "NextSubset[l, s] constructs the subset of l following subset s in canonical order."
];
If[Not[ValueQ[NextTableau::usage]],
NextTableau::usage = "NextTableau[t] gives the tableau of shape t, following t in lexicographic order."
];
If[Not[ValueQ[NoMultipleEdges::usage]],
NoMultipleEdges::usage = "NoMultipleEdges is an option value for Type."
];
If[Not[ValueQ[NonLineGraphs::usage]],
NonLineGraphs::usage = "NonLineGraphs returns a graph whose connected components are the 9 graphs whose presence as a vertex-induced subgraph in a graph g makes g a nonline graph."
];
If[Not[ValueQ[NoPerfectMatchingGraph::usage]],
NoPerfectMatchingGraph::usage = "NoPerfectMatchingGraph returns a connected graph with 16 vertices that contains no perfect matching."
];
$NewMessage[Normal, "usage"]; (* reset the usage of Normal to the system usage *)
If[StringQ[Normal::usage], Normal::usage = StringJoin[Normal::usage, " Normal is also a value that options VertexStyle, EdgeStyle, and PlotRange can take on in ShowGraph."]];
If[Not[ValueQ[NormalDashed::usage]],
NormalDashed::usage = "NormalDashed is a value that the option EdgeStyle can take on in the graph data structure or in ShowGraph."
];
If[Not[ValueQ[NormalizeVertices::usage]],
NormalizeVertices::usage = "NormalizeVertices[v] gives a list of vertices with a similar embedding as v but with all coordinates of all points scaled to be between 0 and 1."
];
If[Not[ValueQ[NoSelfLoops::usage]],
NoSelfLoops::usage = "NoSelfLoops is an option value for Type."
];
If[Not[ValueQ[NthPair::usage]],
NthPair::usage = "NthPair[n] returns the nth unordered pair of distinct positive integers, when sequenced to minimize the size of the larger integer. Pairs that have the same larger integer are sequenced in increasing order of their smaller integer."
];
If[Not[ValueQ[NthPermutation::usage]],
NthPermutation::usage = "NthPermutation[n, l] gives the nth lexicographic permutation of list l. This function is obsolete; use UnrankPermutation instead."
];
If[Not[ValueQ[NthSubset::usage]],
NthSubset::usage = "NthSubset[n, l] gives the nth subset of list l in canonical order, with the empty set indexed as set 0."
];
If[Not[ValueQ[NumberOf2Paths::usage]],
NumberOf2Paths::usage = "NumberOf2Paths[g, v] returns a sorted list that contains the number of paths of length 2 to different vertices of g from v."
];
If[Not[ValueQ[NumberOfCompositions::usage]],
NumberOfCompositions::usage = "NumberOfCompositions[n, k] counts the number of distinct compositions of integer n into k parts."
];
If[Not[ValueQ[NumberOfDerangements::usage]],
NumberOfDerangements::usage = "NumberOfDerangements[n] counts the derangements on n elements, that is, the permutations without any fixed points."
];
If[Not[ValueQ[NumberOfDirectedGraphs::usage]],
NumberOfDirectedGraphs::usage = "NumberOfDirectedGraphs[n] returns the number of nonisomorphic directed graphs with n vertices. NumberOfDirectedGraphs[n, m] returns the number of nonisomorphic directed graphs with n vertices and m edges."
];
If[Not[ValueQ[NumberOfGraphs::usage]],
NumberOfGraphs::usage = "NumberOfGraphs[n] returns the number of nonisomorphic undirected graphs with n vertices. NumberOfGraphs[n, m] returns the number of nonisomorphic undirected graphs with n vertices and m edges."
];
If[Not[ValueQ[NumberOfInvolutions::usage]],
NumberOfInvolutions::usage = "NumberOfInvolutions[n] counts the number of involutions on n elements."
];
If[Not[ValueQ[NumberOfKPaths::usage]],
NumberOfKPaths::usage = "NumberOfKPaths[g, v, k] returns a sorted list that contains the number of paths of length k to different vertices of g from v. NumberOfKPaths[al, v, k] behaves identically, except that it takes an adjacency list al as input."
];
If[Not[ValueQ[NumberOfNecklaces::usage]],
NumberOfNecklaces::usage = "NumberOfNecklaces[n, nc, Cyclic] returns the number of distinct ways in which an n-bead necklace can be colored with nc colors, assuming that two colorings are equivalent if one can be obtained from the other by a rotation. NumberOfNecklaces[n, nc, Dihedral] returns the number of distinct ways in which an n-bead necklace can be colored with nc colors, assuming that two colorings are equivalent if one can be obtained from the other by a rotation or a flip."
];
If[Not[ValueQ[NumberOfPartitions::usage]],
NumberOfPartitions::usage = "NumberOfPartitions[n] counts the number of integer partitions of n."
];
If[Not[ValueQ[NumberOfPermutationsByCycles::usage]],
NumberOfPermutationsByCycles::usage = "NumberOfPermutationsByCycles[n, m] gives the number of permutations of length n with exactly m cycles."
];
If[Not[ValueQ[NumberOfPermutationsByInversions::usage]],
NumberOfPermutationsByInversions::usage = "NumberOfPermutationsByInversions[n, k] gives the number of permutations of length n with exactly k inversions. NumberOfPermutationsByInversions[n] gives a table of the number of length-n permutations with k inversions, for all k."
];
If[Not[ValueQ[NumberOfPermutationsByType::usage]],
NumberOfPermutationsByType::usage = "NumberOfPermutationsByTypes[l] gives the number of permutations of type l."
];
If[Not[ValueQ[NumberOfSpanningTrees::usage]],
NumberOfSpanningTrees::usage = "NumberOfSpanningTrees[g] gives the number of labeled spanning trees of graph g."
];
If[Not[ValueQ[NumberOfTableaux::usage]],
NumberOfTableaux::usage = "NumberOfTableaux[p] uses the hook length formula to count the number of Young tableaux with shape defined by partition p."
];
If[Not[ValueQ[OctahedralGraph::usage]],
OctahedralGraph::usage = "OctahedralGraph returns the graph corresponding to the octahedron, a Platonic solid."
];
If[Not[ValueQ[OddGraph::usage]],
OddGraph::usage = "OddGraph[n] returns the graph whose vertices are the size-(n-1) subsets of a size-(2n-1) set and whose edges connect pairs of vertices that correspond to disjoint subsets. OddGraph[3] is the Petersen graph."
];
If[Not[ValueQ[One::usage]],
One::usage = "One is a tag used in several functions to inform the functions that only one object need be considered or only one solution be produced, as opposed to all objects or all solutions."
];
If[Not[ValueQ[Optimum::usage]],
Optimum::usage = "Optimum is a value that the option Algorithm can take on when used in functions VertexColoring and VertexCover."
];
If[Not[ValueQ[OrbitInventory::usage]],
OrbitInventory::usage = "OrbitInventory[ci, x, w] returns the value of the cycle index ci when each formal variable x[i] is replaced by w. OrbitInventory[ci, x, weights] returns the inventory of orbits induced on a set of functions by the action of a group with cycle index ci. It is assumed that each element in the range of the functions is assigned a weight in list weights."
];
If[Not[ValueQ[OrbitRepresentatives::usage]],
OrbitRepresentatives::usage = "OrbitRepresentatives[pg, x] returns a representative of each orbit of x induced by the action of the group pg on x. pg is assumed to be a set of permutations on the first n natural numbers and x is a set of functions whose domain is the first n natural numbers. Each function in x is specified as an n-tuple."
];
If[Not[ValueQ[Combinatorica`Orbits::usage]],
Combinatorica`Orbits::usage = "Orbits[pg, x] returns the orbits of x induced by the action of the group pg on x. pg is assumed to be a set of permutations on the first n natural numbers and x is a set of functions whose domain is the first n natural numbers. Each function in x is specified as an n-tuple." 
];
If[Not[ValueQ[Ordered::usage]],
Ordered::usage = "Ordered is an option to the functions KSubsetGroup and KSubsetGroupIndex that tells the functions whether they should treat the input as sets or tuples."
];
If[Not[ValueQ[OrientGraph::usage]],
OrientGraph::usage = "OrientGraph[g] assigns a direction to each edge of a bridgeless, undirected graph g, so that the graph is strongly connected."
];
If[Not[ValueQ[OutDegree::usage]],
OutDegree::usage = "OutDegree[g, n] returns the out-degree of vertex n in directed graph g. OutDegree[g] returns the sequence of out-degrees of the vertices in directed graph g."
];
If[Not[ValueQ[PairGroup::usage]],
PairGroup::usage = "PairGroup[g] returns the group induced on 2-sets by the permutation group g. PairGroup[g, Ordered] returns the group induced on ordered pairs with distinct elements by the permutation group g."
];
If[Not[ValueQ[PairGroupIndex::usage]],
PairGroupIndex::usage = "PairGroupIndex[g, x] returns the cycle index of the pair group induced by g as a polynomial in x[1], x[2], .... PairGroupIndex[ci, x] takes the cycle index ci of a group g with formal variables x[1], x[2], ..., and returns the cycle index of the pair group induced by g. PairGroupIndex[g, x, Ordered] returns the cycle index of the ordered pair group induced by g as a polynomial in x[1], x[2], .... PairGroupIndex[ci, x, Ordered] takes the cycle index ci of a group g with formal variables x[1], x[2], ..., and returns the cycle index of the ordered pair group induced by g."
];
If[Not[ValueQ[Parent::usage]],
Parent::usage = "Parent is a tag used as an argument to the function AllPairsShortestPath in order to inform this function that information about parents in the shortest paths is also wanted."
];
If[Not[ValueQ[ParentsToPaths::usage]],
ParentsToPaths::usage = "ParentsToPaths[l, i, j] takes a list of parent lists l and returns the path from i to j encoded in the parent matrix (as returned by the second element of AllPairsShortestPath[g, Parent]). ParentsToPaths[l, i] returns the paths from i to all vertices."
];
If[Not[ValueQ[PartialOrderQ::usage]],
PartialOrderQ::usage = "PartialOrderQ[g] yields True if the binary relation defined by edges of the graph g is a partial order, meaning it is transitive, reflexive, and antisymmetric. PartialOrderQ[r] yields True if the binary relation defined by the square matrix r is a partial order."
];
If[Not[ValueQ[PartitionLattice::usage]],
PartitionLattice::usage = "PartitionLattice[n] returns a Hasse diagram of the partially ordered set on set partitions of 1 through n in which p < q if q is finer than p, that is, each block in q is contained in some block in p. The function takes two options: Type and VertexLabel, with default values Undirected and False, respectively. When Type is set to Directed, the function produces the underlying directed acyclic graph. When VertexLabel is set to True, labels are produced for the vertices."
];
If[Not[ValueQ[PartitionQ::usage]],
PartitionQ::usage = "PartitionQ[p] yields True if p is an integer partition. PartitionQ[n, p] yields True if p is a partition of n."
];
If[Not[ValueQ[Partitions::usage]],
Partitions::usage = "Partitions[n] constructs all partitions of integer n in reverse lexicographic order. Partitions[n, k] constructs all partitions of the integer n with maximum part at most k, in reverse lexicographic order."
];
If[$VersionNumber < 4, Path::usage=" "];
$NewMessage[Path, "usage"]; (* reset the usage of Path to the system usage *)
If[StringQ[Path::usage], Path::usage = StringJoin[Path::usage, " Path[n] constructs a tree consisting only of a path on n vertices. Path[n] permits an option Type that takes on the values Directed and Undirected. The default setting is Type -> Undirected."]];
If[Not[ValueQ[PathConditionGraph::usage]],
PathConditionGraph::usage = "The usage of PathConditionGraph is obsolete. This functionality is no longer supported in Combinatorica."
];
If[Not[ValueQ[PerfectQ::usage]],
PerfectQ::usage = "PerfectQ[g] yields True if g is a perfect graph, meaning that for every induced subgraph of g the size of a largest clique equals the chromatic number."
];
If[Not[ValueQ[PermutationGraph::usage]],
PermutationGraph::usage = "PermutationGraph[p] gives the permutation graph for the permutation p."
];
If[Not[ValueQ[PermutationGroupQ::usage]],
PermutationGroupQ::usage = "PermutationGroupQ[l] yields True if the list of permutations l forms a permutation group."
];
If[Not[ValueQ[Combinatorica`PermutationQ::usage]],
Combinatorica`PermutationQ::usage = "PermutationQ[p] yields True if p is a list representing a permutation and False otherwise."
];
If[Not[ValueQ[PermutationToTableaux::usage]],
PermutationToTableaux::usage = "PermutationToTableaux[p] returns the tableaux pair that can be constructed from p using the Robinson-Schensted-Knuth correspondence."
];
If[Not[ValueQ[PermutationType::usage]],
PermutationType::usage = "PermutationType[p] returns the type of permutation p."
];
If[Not[ValueQ[PermutationWithCycle::usage]],
PermutationWithCycle::usage = "PermutationWithCycle[n, {i, j, ...}] gives a size-n permutation in which {i, j, ...} is a cycle and all other elements are fixed points."
];
If[Not[ValueQ[Combinatorica`Permute::usage]],
Combinatorica`Permute::usage = "Permute[l, p] permutes list l according to permutation p."
];
If[Not[ValueQ[PermuteSubgraph::usage]],
PermuteSubgraph::usage = "PermuteSubgraph[g, p] permutes the vertices of a subgraph of g induced by p according to p."
];
If[Not[ValueQ[Combinatorica`PetersenGraph::usage]],
Combinatorica`PetersenGraph::usage = "PetersenGraph returns the Petersen graph, a graph whose vertices can be viewed as the size-2 subsets of a size-5 set with edges connecting disjoint subsets."
];
If[Not[ValueQ[PlanarQ::usage]],
PlanarQ::usage = "PlanarQ[g] yields True if graph g is planar, meaning it can be drawn in the plane so no two edges cross."
];
If[Not[ValueQ[PointsAndLines::usage]],
PointsAndLines::usage = "PointsAndLines is now obsolete."
];
If[Not[ValueQ[Polya::usage]],
Polya::usage = "Polya[g, m] returns the polynomial giving the number of colorings, with m colors, of a structure defined by the permutation group g. Polya is obsolete; use OrbitInventory instead."
];
If[Not[ValueQ[PseudographQ::usage]],
PseudographQ::usage = "PseudographQ[g] yields True if graph g is a pseudograph, meaning it contains self-loops."
];
If[Not[ValueQ[RadialEmbedding::usage]],
RadialEmbedding::usage = "RadialEmbedding[g, v] constructs a radial embedding of the graph g in which vertices are placed on concentric circles around v depending on their distance from v. RadialEmbedding[g] constructs a radial embedding of graph g, radiating from the center of the graph."
];
If[Not[ValueQ[Radius::usage]],
Radius::usage = "Radius[g] gives the radius of graph g, the minimum eccentricity of any vertex of g."
];
If[Not[ValueQ[RandomComposition::usage]],
RandomComposition::usage = "RandomComposition[n, k] constructs a random composition of integer n into k parts."
];
If[Not[ValueQ[Combinatorica`RandomGraph::usage]],
Combinatorica`RandomGraph::usage = "RandomGraph[n, p] constructs a random labeled graph on n vertices with an edge probability of p. An option Type is provided, which can take on values Directed and Undirected, and whose default value is Undirected. Type->Directed produces a corresponding random directed graph. The usages Random[n, p, Directed], Random[n, p, range], and Random[n, p, range, Directed] are all obsolete. Use SetEdgeWeights to set random edge weights."
];
If[Not[ValueQ[RandomHeap::usage]],
RandomHeap::usage = "RandomHeap[n] constructs a random heap on n elements."
];
If[StringQ[RandomInteger::usage]&&StringPosition[RandomInteger::usage,"SetEdgeWeights"]==={},
	RandomInteger::usage = StringJoin[RandomInteger::usage, 
	"   RandomInteger is a value that the WeightingFunction option of the function SetEdgeWeights can take."]];
If[Not[ValueQ[RandomKSetPartition::usage]],
RandomKSetPartition::usage = "RandomKSetPartition[set, k] returns a random set partition of set with k blocks. RandomKSetPartition[n, k] returns a random set partition of the first n natural numbers into k blocks."
];
If[Not[ValueQ[RandomKSubset::usage]],
RandomKSubset::usage = "RandomKSubset[l, k] gives a random subset of set l with exactly k elements."
];
If[Not[ValueQ[RandomPartition::usage]],
RandomPartition::usage = "RandomPartition[n] constructs a random partition of integer n."
];
If[Not[ValueQ[Combinatorica`RandomPermutation::usage]],
Combinatorica`RandomPermutation::usage = "RandomPermutation[n] generates a random permutation of the first n natural numbers."
];
If[Not[ValueQ[Combinatorica`RandomPermutation1::usage]],
Combinatorica`RandomPermutation1::usage = "RandomPermutation1 is now obsolete. Use RandomPermutation instead."
];
If[Not[ValueQ[Combinatorica`RandomPermutation2::usage]],
Combinatorica`RandomPermutation2::usage = "RandomPermutation2 is now obsolete. Use RandomPermutation instead."
];
If[Not[ValueQ[RandomRGF::usage]],
RandomRGF::usage = "RandomRGF[n] returns a random restricted growth function (RGF) defined on the first n natural numbers. RandomRGF[n, k] returns a random RGF defined on the first n natural numbers having maximum element equal to k."
];
If[Not[ValueQ[RandomSetPartition::usage]],
RandomSetPartition::usage = "RandomSetPartition[set] returns a random set partition of set. RandomSetPartition[n] returns a random set partition of the first n natural numbers."
];
If[Not[ValueQ[RandomSubset::usage]],
RandomSubset::usage = "RandomSubset[l] creates a random subset of set l."
];
If[Not[ValueQ[RandomTableau::usage]],
RandomTableau::usage = "RandomTableau[p] constructs a random Young tableau of shape p."
];
If[Not[ValueQ[RandomTree::usage]],
RandomTree::usage = "RandomTree[n] constructs a random labeled tree on n vertices."
];
If[Not[ValueQ[RandomVertices::usage]],
RandomVertices::usage = "RandomVertices[g] assigns a random embedding to graph g."
];
If[Not[ValueQ[RankBinarySubset::usage]],
RankBinarySubset::usage = "RankBinarySubset[l, s] gives the rank of subset s of set l in the ordering of subsets of l, obtained by interpreting these subsets as binary string representations of integers."
];
If[Not[ValueQ[RankedEmbedding::usage]],
RankedEmbedding::usage = "RankedEmbedding[l] takes a set partition l of vertices {1, 2,..., n} and returns an embedding of the vertices in the plane such that the vertices in each block occur on a vertical line with block 1 vertices on the leftmost line, block 2 vertices in the next line, and so on. RankedEmbedding[g, l] takes a graph g and a set partition l of the vertices of g and returns the graph g with vertices embedded according to RankedEmbedding[l]. RankedEmbedding[g, s] takes a graph g and a set s of vertices of g and returns a ranked embedding of g in which vertices in s are in block 1, vertices at distance 1 from any vertex in block 1 are in block 2, and so on."
];
If[Not[ValueQ[RankGraph::usage]],
RankGraph::usage = "RankGraph[g, l] partitions the vertices into classes based on the shortest geodesic distance to a member of list l."
];
If[Not[ValueQ[RankGrayCodeSubset::usage]],
RankGrayCodeSubset::usage = "RankGrayCodeSubset[l, s] gives the rank of subset s of set l in the Gray code ordering of the subsets of l." 
];
If[Not[ValueQ[RankKSetPartition::usage]],
RankKSetPartition::usage = "RankKSetPartition[sp, s] ranks sp in the list of all k-block set partitions of s. RankSetPartition[sp] ranks sp in the list of all k-block set partitions of the set of elements that appear in any subset in sp."
];
If[Not[ValueQ[RankKSubset::usage]],
RankKSubset::usage = "RankKSubset[s, l] gives the rank of k-subset s of set l in the lexicographic ordering of the k-subsets of l." 
];
If[Not[ValueQ[RankPermutation::usage]],
RankPermutation::usage = "RankPermutation[p] gives the rank of permutation p in lexicographic order."
];
If[Not[ValueQ[RankRGF::usage]],
RankRGF::usage = "RankRGF[f] returns the rank of a restricted growth function (RGF) f in the lexicographic order of all RGFs."
];
If[Not[ValueQ[RankSetPartition::usage]],
RankSetPartition::usage = "RankSetPartition[sp, s] ranks sp in the list of all set partitions of set s. RankSetPartition[sp] ranks sp in the list of all set partitions of the set of elements that appear in any subset in sp."
];
If[Not[ValueQ[RankSubset::usage]],
RankSubset::usage = "RankSubset[l, s] gives the rank, in canonical order, of subset s of set l."
];
If[Not[ValueQ[ReadGraph::usage]],
ReadGraph::usage = "ReadGraph[f] reads a graph represented as edge lists from file f and returns a graph object."
];
If[Not[ValueQ[RealizeDegreeSequence::usage]],
RealizeDegreeSequence::usage = "RealizeDegreeSequence[s] constructs a semirandom graph with degree sequence s."
];
If[Not[ValueQ[ReflexiveQ::usage]],
ReflexiveQ::usage = "ReflexiveQ[g] yields True if the adjacency matrix of g represents a reflexive binary relation."
];
If[Not[ValueQ[RegularGraph::usage]],
RegularGraph::usage = "RegularGraph[k, n] constructs a semirandom k-regular graph on n vertices, if such a graph exists."
];
If[Not[ValueQ[RegularQ::usage]],
RegularQ::usage = "RegularQ[g] yields True if g is a regular graph."
];
If[Not[ValueQ[RemoveMultipleEdges::usage]],
RemoveMultipleEdges::usage = "RemoveMultipleEdges[g] returns the graph obtained by deleting multiple edges from g."
];
If[Not[ValueQ[RemoveSelfLoops::usage]],
RemoveSelfLoops::usage = "RemoveSelfLoops[g] returns the graph obtained by deleting self-loops in g."
];
If[Not[ValueQ[ResidualFlowGraph::usage]],
ResidualFlowGraph::usage = "ResidualFlowGraph[g, flow] returns the directed residual flow graph for graph g with respect to flow."
];
If[Not[ValueQ[RevealCycles::usage]],
RevealCycles::usage = "RevealCycles[p] unveils the canonical hidden cycle structure of permutation p."
];
If[Not[ValueQ[ReverseEdges::usage]],
ReverseEdges::usage = "ReverseEdges[g] flips the directions of all edges in a directed graph."
];
If[Not[ValueQ[RGFQ::usage]],
RGFQ::usage = "RGFQ[l] yields True if l is a restricted growth function. It yields False otherwise."
];
If[Not[ValueQ[RGFs::usage]],
RGFs::usage = "RGFs[n] lists all restricted growth functions on the first n natural numbers in lexicographic order."
];
If[Not[ValueQ[RGFToSetPartition::usage]],
RGFToSetPartition::usage = "RGFToSetPartition[rgf, set] converts the restricted growth function rgf into the corresponding set partition of set. If the optional second argument, set, is not supplied, then rgf is converted into a set partition of {1, 2, ..., Length[rgf]}."
];
If[Not[ValueQ[RobertsonGraph::usage]],
RobertsonGraph::usage = "RobertsonGraph returns a 19-vertex graph that is the unique (4, 5)-cage graph."
];
If[Not[ValueQ[RootedEmbedding::usage]],
RootedEmbedding::usage = "RootedEmbedding[g, v] constructs a rooted embedding of graph g with vertex v as the root. RootedEmbedding[g] constructs a rooted embedding with a center of g as the root."
];
If[Not[ValueQ[RotateVertices::usage]],
RotateVertices::usage = "RotateVertices[v, theta] rotates each vertex position in list v by theta radians about the origin (0, 0). RotateVertices[g, theta] rotates the embedding of the graph g by theta radians about the origin (0, 0)."
];
If[Not[ValueQ[Runs::usage]],
Runs::usage = "Runs[p] partitions p into contiguous increasing subsequences."
];
If[Not[ValueQ[SamenessRelation::usage]],
SamenessRelation::usage = "SamenessRelation[l] constructs a binary relation from a list l of permutations, which is an equivalence relation if l is a permutation group."
];
If[Not[ValueQ[SelectionSort::usage]],
SelectionSort::usage = "SelectionSort[l, f] sorts list l using ordering function f."
];
If[Not[ValueQ[SelfComplementaryQ::usage]],
SelfComplementaryQ::usage = "SelfComplementaryQ[g] yields True if graph g is self-complementary, meaning it is isomorphic to its complement."
];
If[Not[ValueQ[SelfLoopsQ::usage]],
SelfLoopsQ::usage = "SelfLoopsQ[g] yields True if graph g has self-loops."
];
If[Not[ValueQ[SetEdgeLabels::usage]],
SetEdgeLabels::usage = "SetEdgeLabels[g, l] assigns the labels in l to edges of g. If l is shorter than the number of edges in g, then labels get assigned cyclically. If l is longer than the number of edges in g, then the extra labels are ignored."
];
If[Not[ValueQ[SetEdgeWeights::usage]],
SetEdgeWeights::usage = "SetEdgeWeights[g] assigns random real weights in the range [0, 1] to edges in g. SetEdgeWeights accepts options WeightingFunction and WeightRange. WeightingFunction can take values Random, RandomInteger, Euclidean, or LNorm[n] for nonnegative n, or any pure function that takes two arguments, each argument having the form {Integer, {Number, Number}}. WeightRange can be an integer range or a real range. The default value for WeightingFunction is Random and the default value for WeightRange is [0, 1]. SetEdgeWeights[g, e] assigns edge weights to the edges in the edge list e. SetEdgeWeights[g, w] assigns the weights in the weight list w to the edges of g. SetEdgeWeights[g, e, w] assigns the weights in the weight list w to the edges in edge list e."
];
If[Not[ValueQ[SetGraphOptions::usage]],
SetGraphOptions::usage = "SetGraphOptions[g, opts] returns g with the options opts set. SetGraphOptions[g, {v1, v2, ..., vopts}, gopts] returns the graph with the options vopts set for vertices v1, v2, ... and the options gopts set for the graph g. SetGraphOptions[g, {e1, e2,..., eopts}, gopts], with edges e1, e2,..., works similarly. SetGraphOptions[g, {{elements1, opts1}, {elements2, opts2},...}, opts] returns g with the options opts1 set for the elements in the sequence elements1, the options opts2 set for the elements in the sequence elements2, and so on. Here, elements can be a sequence of edges or a sequence of vertices. A tag that takes on values One or All can also be passed in as an argument before any options. The default value of the tag is All and it is useful if the graph has multiple edges. It informs the function about whether all edges that connect a pair of vertices are to be affected or only one edge is affected."
];
If[Not[ValueQ[SetPartitionListViaRGF::usage]],
SetPartitionListViaRGF::usage = "SetPartitionListViaRGF[n] lists all set partitions of the first n natural numbers, by first listing all restricted growth functions (RGFs) on these and then mapping the RGFs to corresponding set partitions. SetPartitionListViaRGF[n, k] lists all RGFs on the first n natural numbers whose maximum element is k and then maps these RGFs into the corresponding set partitions, all of which contain exactly k blocks." 
];
If[Not[ValueQ[SetPartitionQ::usage]],
SetPartitionQ::usage = "SetPartitionQ[sp, s] determines if sp is a set partition of set s. SetPartitionQ[sp] tests if sp is a set of disjoint sets."
];
If[Not[ValueQ[SetPartitions::usage]],
SetPartitions::usage = "SetPartitions[set] returns the list of set partitions of set. SetPartitions[n] returns the list of set partitions of {1, 2, ..., n}. If all set partitions with a fixed number of subsets are needed use KSetPartitions."
];
If[Not[ValueQ[SetPartitionToRGF::usage]],
SetPartitionToRGF::usage = "SetPartitionToRGF[sp, set] converts the set partition sp of set into the corresponding restricted growth function. If the optional argument set is not specified, then it is assumed that Mathematica knows the underlying order on the set for which sp is a set partition."
];
If[Not[ValueQ[SetVertexLabels::usage]],
SetVertexLabels::usage = "SetVertexLabels[g, l] assigns the labels in l to vertices of g. If l is shorter than the number of vertices in g, then labels get assigned cyclically. If l is longer than the number of vertices in g, then the extra labels are ignored." 
];
If[Not[ValueQ[SetVertexWeights::usage]],
SetVertexWeights::usage = "SetVertexWeights[g] assigns random real weights in the range [0, 1] to vertices in g. SetVertexWeights accepts options WeightingFunction and WeightRange. WeightingFunction can take values Random, RandomInteger, or any pure function that takes two arguments, an integer as the first argument and a pair {number, number} as the second argument. WeightRange can be an integer range or a real range. The default value for WeightingFunction is Random and the default value for WeightRange is [0, 1]. SetVertexWeights[g, w] assigns the weights in the weight list w to the vertices of g. SetVertexWeights[g, vs, w] assigns the weights in the weight list w to the vertices in the vertex list vs."
];
If[Not[ValueQ[ShakeGraph::usage]],
ShakeGraph::usage = "ShakeGraph[g, d] performs a random perturbation of the vertices of graph g, with each vertex moving, at most, a distance d from its original position."
];
If[Not[ValueQ[ShortestPath::usage]],
ShortestPath::usage = "ShortestPath[g, start, end] finds a shortest path between vertices start and end in graph g. An option Algorithm that takes on the values Automatic, Dijkstra, or BellmanFord is provided. This allows a choice between using Dijkstra's algorithm and the Bellman-Ford algorithm. The default is Algorithm -> Automatic. In this case, depending on whether edges have negative weights and depending on the density of the graph, the algorithm chooses between Bellman-Ford and Dijkstra."
];
If[Not[ValueQ[ShortestPathSpanningTree::usage]],
ShortestPathSpanningTree::usage = "ShortestPathSpanningTree[g, v] constructs a shortest-path spanning tree rooted at v, so that a shortest path in graph g from v to any other vertex is a path in the tree. An option Algorithm that takes on the values Automatic, Dijkstra, or BellmanFord is provided. This allows a choice between Dijkstra's algorithm and the Bellman-Ford algorithm. The default is Algorithm -> Automatic. In this case, depending on whether edges have negative weights and depending on the density of the graph, the algorithm chooses between Bellman-Ford and Dijkstra."
];
If[Not[ValueQ[ShowGraph::usage]],
ShowGraph::usage = "ShowGraph[g] displays the graph g. ShowGraph[g, options] modifies the display using the given options. ShowGraph[g, Directed] is obsolete and it is currently identical to ShowGraph[g]. All options that affect the look of a graph can be specified as options in ShowGraph. The list of options is: VertexColor, VertexStyle, VertexNumber, VertexNumberColor, VertexNumberPosition, VertexLabel, VertexLabelColor, VertexLabelPosition, EdgeColor, EdgeStyle, EdgeLabel, EdgeLabelColor, EdgeLabelPosition, LoopPosition, and EdgeDirection. In addition, options of the Mathematica function Plot can also be specified here. If an option specified in ShowGraph differ from options explicitly set within a graph object, then options specified inside the graph object are used." 
];
If[Not[ValueQ[ShowGraphArray::usage]],
ShowGraphArray::usage = "ShowGraphArray[{g1, g2, ...}] displays a row of graphs. ShowGraphArray[{ {g1, ...}, {g2, ...}, ...}] displays a two-dimensional table of graphs. ShowGraphArray accepts all the options accepted by ShowGraph, and the user can also provide the option GraphicsSpacing -> d."
];
If[Not[ValueQ[ShowLabeledGraph::usage]],
ShowLabeledGraph::usage = "ShowLabeledGraph[g] displays graph g according to its embedding, with each vertex labeled with its vertex number. ShowLabeledGraph[g, l] uses the ith element of list l as the label for vertex i."
];
If[Not[ValueQ[ShuffleExchangeGraph::usage]],
ShuffleExchangeGraph::usage = "ShuffleExchangeGraph[n] returns the n-dimensional shuffle-exchange graph whose vertices are length n binary strings with an edge from w to w' if (i) w' differs from w in its last bit or (ii) w' is obtained from w by a cyclic shift left or a cyclic shift right. An option VertexLabel is provided, with default setting False, which can be set to True, if the user wants to associate the binary strings to the vertices as labels." 
];
If[Not[ValueQ[SignaturePermutation::usage]],
SignaturePermutation::usage = "SignaturePermutation[p] gives the signature of permutation p."
];
If[Not[ValueQ[Simple::usage]],
Simple::usage = "Simple is an option value for Type."
];
If[Not[ValueQ[SimpleQ::usage]],
SimpleQ::usage = "SimpleQ[g] yields True if g is a simple graph, meaning it has no multiple edges and contains no self-loops."
];
$NewMessage[Small, "usage"];
If[StringQ[Small::usage],
Small::usage = Small::usage<> " Small is also a symbol in Combinatorica used to denote the size of the object that represents a vertex. The option VertexStyle can be set to Disk[Small] or Box[Small] either inside the graph data structure or in ShowGraph."
];
If[Not[ValueQ[SmallestCyclicGroupGraph::usage]],
SmallestCyclicGroupGraph::usage = "SmallestCyclicGroupGraph returns a smallest nontrivia al graph whose automorphism group is cyclic."
];
If[Not[ValueQ[Spectrum::usage]],
Spectrum::usage = "Spectrum[g] gives the eigenvalues of graph g."
];
If[Not[ValueQ[SpringEmbedding::usage]],
SpringEmbedding::usage = "SpringEmbedding[g] beautifies the embedding of graph g by modeling the embedding as a system of springs. SpringEmbedding[g, step, increment] can be used to refine the algorithm. The value of step tells the function for how many iterations to run the algorithm. The value of increment tells the function the distance to move the vertices at each step. The default values are 10 and 0.15 for step and increment, respectively."
];
If[Not[ValueQ[StableMarriage::usage]],
StableMarriage::usage = "StableMarriage[mpref, fpref] finds the male optimal stable marriage defined by lists of permutations describing male and female preferences."
];
$NewMessage[Star,"usage"];
If[StringQ[Star::usage],
Star::usage = StringJoin[Star::usage,"   Star[n] constructs a star on n vertices, which is a tree with one vertex of degree n-1."]
];
If[Not[ValueQ[StirlingFirst::usage]],
StirlingFirst::usage = "StirlingFirst[n, k] returns a Stirling number of the first kind. This is obsolete. Use the built-in Mathematica function StirlingS1 instead."
];
If[Not[ValueQ[StirlingSecond::usage]],
StirlingSecond::usage = "StirlingSecond[n, k] returns a Stirling number of the second kind."
];
If[Not[ValueQ[Strings::usage]],
Strings::usage = "Strings[l, n] constructs all possible strings of length n from the elements of list l."
];
If[Not[ValueQ[StronglyConnectedComponents::usage]],
StronglyConnectedComponents::usage = "StronglyConnectedComponents[g] gives the strongly connected components of directed graph g as lists of vertices."
];
If[Not[ValueQ[Strong::usage]],
Strong::usage = "Strong is an option to ConnectedQ that seeks to determine if a directed graph is strongly connected."
];
If[Not[ValueQ[Combinatorica`SymmetricGroup::usage]],
Combinatorica`SymmetricGroup::usage = "SymmetricGroup[n] returns the symmetric group on n symbols."
];
If[Not[ValueQ[SymmetricGroupIndex::usage]],
SymmetricGroupIndex::usage = "SymmetricGroupIndex[n, x] returns the cycle index of the symmetric group on n symbols, expressed as a polynomial in x[1], x[2], ..., x[n]."
];
If[Not[ValueQ[SymmetricQ::usage]],
SymmetricQ::usage = "SymmetricQ[r] tests if a given square matrix r represents a symmetric relation. SymmetricQ[g] tests if the edges of a given graph represent a symmetric relation."
];
If[Not[ValueQ[TableauClasses::usage]],
TableauClasses::usage = "TableauClasses[p] partitions the elements of permutation p into classes according to their initial columns during Young tableaux construction."
];
If[Not[ValueQ[Combinatorica`TableauQ::usage]],
Combinatorica`TableauQ::usage = "TableauQ[t] yields True if and only if t represents a Young tableau."
];
If[Not[ValueQ[Tableaux::usage]],
Tableaux::usage = "Tableaux[p] constructs all tableaux having a shape given by integer partition p."
];
If[Not[ValueQ[TableauxToPermutation::usage]],
TableauxToPermutation::usage = "TableauxToPermutation[t1, t2] constructs the unique permutation associated with Young tableaux t1 and t2, where both tableaux have the same shape."
];
If[Not[ValueQ[TetrahedralGraph::usage]],
TetrahedralGraph::usage = "TetrahedralGraph returns the graph corresponding to the the Tetrahedron, a Platonic solid."
];
(*
$NewMessage[Thick,"usage"];
If[StringQ[Thick::usage],
Thick::usage = StringJoin[Thick::usage,"   Thick is a value that the option EdgeStyle can take on in the graph data structure or in ShowGraph."]
];
*)
If[Not[ValueQ[ThickDashed::usage]],
ThickDashed::usage = "ThickDashed is a value that the option EdgeStyle can take on in the graph data structure or in ShowGraph."
];
(*
$NewMessage[Thin,"usage"];
If[StringQ[Thin::usage],
Thin::usage = StringJoin[Thin::usage,"   Thin is a value that the option EdgeStyle can take on in the graph data structure or in ShowGraph."]
];
*)
If[Not[ValueQ[ThinDashed::usage]],
ThinDashed::usage = "ThinDashed is a value that the option EdgeStyle can take on in the graph data structure or in ShowGraph."
];
If[Not[ValueQ[ThomassenGraph::usage]],
ThomassenGraph::usage = "ThomassenGraph returns a hypotraceable graph, a graph G that has no Hamiltonian path but whose subgraph G-v for every vertex v has a Hamiltonian path."
];
If[Not[ValueQ[ToAdjacencyLists::usage]],
ToAdjacencyLists::usage = "ToAdjacencyLists[g] constructs an adjacency list representation for graph g. It allows an option called Type that takes on values All or Simple. Type -> All is the default setting of the option, and this permits self-loops and multiple edges to be reported in the adjacency lists. Type -> Simple deletes self-loops and multiple edges from the constructed adjacency lists. ToAdjacencyLists[g, EdgeWeight] returns an adjacency list representation along with edge weights."
];
If[Not[ValueQ[ToAdjacencyMatrix::usage]],
ToAdjacencyMatrix::usage = "ToAdjacencyMatrix[g] constructs an adjacency matrix representation for graph g. An option Type that takes on values All or Simple can be used to affect the matrix constructed. Type -> All is the default, and Type -> Simple ignores any self-loops or multiple edges g may have. ToAdjacencyMatrix[g, EdgeWeight] returns edge weights as entries of the adjacency matrix with Infinity representing missing edges."
];
If[Not[ValueQ[ToCanonicalSetPartition::usage]],
ToCanonicalSetPartition::usage = "ToCanonicalSetPartition[sp, set] reorders sp into a canonical order with respect to set. In the canonical order, the elements of each subset of the set partition are ordered as they appear in set, and the subsets themselves are ordered by their first elements. ToCanonicalSetPartition[sp] reorders sp into canonical order, assuming that Mathematica knows the underlying order on the set for which sp is a set partition."
];
If[Not[ValueQ[Combinatorica`ToCycles::usage]],
Combinatorica`ToCycles::usage = "ToCycles[p] gives the cycle structure of permutation p as a list of cyclic permutations."
];
If[Not[ValueQ[ToInversionVector::usage]],
ToInversionVector::usage = "ToInversionVector[p] gives the inversion vector associated with permutation p."
];
If[Not[ValueQ[ToOrderedPairs::usage]],
ToOrderedPairs::usage = "ToOrderedPairs[g] constructs a list of ordered pairs representing the edges of the graph g. If g is undirected each edge is interpreted as two ordered pairs. An option called Type that takes on values Simple or All can be used to affect the constructed representation. Type -> Simple forces the removal of multiple edges and self-loops. Type -> All keeps all information and is the default option."
];
If[Not[ValueQ[Combinatorica`TopologicalSort::usage]],
Combinatorica`TopologicalSort::usage = "TopologicalSort[g] gives a permutation of the vertices of directed acyclic graph g such that an edge (i, j) implies that vertex i appears before vertex j."
];
If[Not[ValueQ[ToUnorderedPairs::usage]],
ToUnorderedPairs::usage = "ToUnorderedPairs[g] constructs a list of unordered pairs representing the edges of graph g. Each edge, directed or undirected, results in a pair in which the smaller vertex appears first. An option called Type that takes on values All or Simple can be used, and All is the default value. Type -> Simple ignores multiple edges and self-loops in g."
];
If[Not[ValueQ[TransitiveClosure::usage]],
TransitiveClosure::usage = "TransitiveClosure[g] finds the transitive closure of graph g, the supergraph of g that contains edge {x, y} if and only if there is a path from x to y."
];
If[Not[ValueQ[TransitiveQ::usage]],
TransitiveQ::usage = "TransitiveQ[g] yields True if graph g defines a transitive relation."
];
If[Not[ValueQ[TransitiveReduction::usage]],
TransitiveReduction::usage = "TransitiveReduction[g] finds a smallest graph that has the same transitive closure as g."
];
If[Not[ValueQ[TranslateVertices::usage]],
TranslateVertices::usage = "TranslateVertices[v, {x, y}] adds the vector {x, y} to the vertex embedding location of each vertex in list v. TranslateVertices[g, {x, y}] translates the embedding of the graph g by the vector {x, y}."
];
If[Not[ValueQ[TransposePartition::usage]],
TransposePartition::usage = "TransposePartition[p] reflects a partition p of k parts along the main diagonal, creating a partition with maximum part k."
];
If[Not[ValueQ[Combinatorica`TransposeTableau::usage]],
Combinatorica`TransposeTableau::usage = "TransposeTableau[t] reflects a Young tableau t along the main diagonal, creating a different tableau."
];
If[Not[ValueQ[TravelingSalesman::usage]],
TravelingSalesman::usage = "TravelingSalesman[g] finds an optimal traveling salesman tour in a Hamiltonian graph g."
];
If[Not[ValueQ[TravelingSalesmanBounds::usage]],
TravelingSalesmanBounds::usage = "TravelingSalesmanBounds[g] gives upper and lower bounds on a minimum cost traveling salesman tour of graph g."
];
If[Not[ValueQ[Tree::usage]],
Tree::usage = "Tree is an option that informs certain functions for which the user wants the output to be a tree."
];
If[Not[ValueQ[TreeIsomorphismQ::usage]],
TreeIsomorphismQ::usage = "TreeIsomorphismQ[t1, t2] yields True if the trees t1 and t2 are isomorphic. It yields False otherwise."
];
If[Not[ValueQ[TreeQ::usage]],
TreeQ::usage = "TreeQ[g] yields True if graph g is a tree."
];
If[Not[ValueQ[TreeToCertificate::usage]],
TreeToCertificate::usage = "TreeToCertificate[t] returns a binary string that is a certificate for the tree t such that trees have the same certificate if and only if they are isomorphic."
];
If[Not[ValueQ[TriangleInequalityQ::usage]],
TriangleInequalityQ::usage = "TriangleInequalityQ[g] yields True if the weights assigned to the edges of graph g satisfy the triangle inequality."
];
If[Not[ValueQ[Turan::usage]],
Turan::usage = "Turan[n, p] constructs the Turan graph, the extremal graph on n vertices that does not contain CompleteGraph[p]."
];
If[Not[ValueQ[TutteGraph::usage]],
TutteGraph::usage = "TutteGraph returns the Tutte graph, the first known example of a 3-connected, 3-regular, planar graph that is non-Hamiltonian."
];
If[Not[ValueQ[TwoColoring::usage]],
TwoColoring::usage = "TwoColoring[g] finds a two-coloring of graph g if g is bipartite. It returns a list of the labels 1 and 2 corresponding to the vertices. This labeling is a valid coloring if and only the graph is bipartite."
];
If[Not[ValueQ[Type::usage]],
Type::usage = "Type is an option for many functions that transform graphs. Depending on the functions it is being used in, it can take on values such as Directed, Undirected, Simple, etc."
];
If[Not[ValueQ[UndirectedQ::usage]],
UndirectedQ::usage = "UndirectedQ[g] yields True if graph g is undirected."
];
If[Not[ValueQ[Undirected::usage]],
Undirected::usage = "Undirected is an option to inform certain functions that the graph is undirected."
];
If[Not[ValueQ[UnionSet::usage]],
UnionSet::usage = "UnionSet[a, b, s] merges the sets containing a and b in union-find data structure s."
];
If[Not[ValueQ[Uniquely3ColorableGraph::usage]],
Uniquely3ColorableGraph::usage = "Uniquely3ColorableGraph returns a 12-vertex, triangle-free graph with chromatic number 3 that is uniquely 3-colorable."
];
If[Not[ValueQ[UnitransitiveGraph::usage]],
UnitransitiveGraph::usage = "UnitransitiveGraph returns a 20-vertex, 3-unitransitive graph discovered by Coxeter, that is not isomorphic to a 4-cage or a 5-cage."
];
If[Not[ValueQ[UnrankBinarySubset::usage]],
UnrankBinarySubset::usage = "UnrankBinarySubset[n, l] gives the nth subset of list l, listed in increasing order of integers corresponding to the binary representations of the subsets."
];
If[Not[ValueQ[UnrankGrayCodeSubset::usage]],
UnrankGrayCodeSubset::usage = "UnrankGrayCodeSubset[n, l] gives the nth subset of list l, listed in Gray code order."
];
If[Not[ValueQ[UnrankKSetPartition::usage]],
UnrankKSetPartition::usage = "UnrankSetPartition[r, s, k] finds a k-block set partition of s with rank r. UnrankSetPartition[r, n, k] finds a k-block set partition of {1, 2,..., n} with rank r." 
];
If[Not[ValueQ[UnrankKSubset::usage]],
UnrankKSubset::usage = "UnrankKSubset[m, k, l] gives the mth k-subset of set l, listed in lexicographic order."
];
If[Not[ValueQ[UnrankPermutation::usage]],
UnrankPermutation::usage = "UnrankPermutation[r, l] gives the rth permutation in the lexicographic list of permutations of list l. UnrankPermutation[r, n] gives the rth permutation in the lexicographic list of permutations of {1, 2,..., n}."
];
If[Not[ValueQ[UnrankRGF::usage]],
UnrankRGF::usage = "UnrankRGF[r, n] returns a restricted growth function defined on the first n natural numbers whose rank is r."
];
If[Not[ValueQ[UnrankSetPartition::usage]],
UnrankSetPartition::usage = "UnrankSetPartition[r, set] finds a set partition of set with rank r. UnrankSetPartition[r, n] finds a set partition of {1, 2, ..., n} with rank r." 
];
If[Not[ValueQ[UnrankSubset::usage]],
UnrankSubset::usage = "UnrankSubset[n, l] gives the nth subset of list l, listed in some canonical order."
];
If[Not[ValueQ[UnweightedQ::usage]],
UnweightedQ::usage = "UnweightedQ[g] yields True if all edge weights are 1 and False otherwise."
];
If[Not[ValueQ[UpperLeft::usage]],
UpperLeft::usage = "UpperLeft is a value that options VertexNumberPosition, VertexLabelPosition, and EdgeLabelPosition can take on in ShowGraph."
];
If[Not[ValueQ[UpperRight::usage]],
UpperRight::usage = "UpperRight is a value that options VertexNumberPosition, VertexLabelPosition, and EdgeLabelPosition can take on in ShowGraph."
];
If[Not[ValueQ[V::usage]],
V::usage = "V[g] gives the order or number of vertices of the graph g."
];
If[Not[ValueQ[Value::usage]],
Value::usage = "Value is an option for the function NetworkFlow that makes the function return the value of the maximum flow."
];
If[Not[ValueQ[VertexColor::usage]],
VertexColor::usage = "VertexColor is an option that allows the user to associate colors with vertices. Black is the default color. VertexColor can be set as part of the graph data structure and it can be used in ShowGraph."
];
If[Not[ValueQ[VertexColoring::usage]],
VertexColoring::usage = "VertexColoring[g] uses Brelaz's heuristic to find a good, but not necessarily minimal, vertex coloring of graph g. An option Algorithm that can take on the values Brelaz or Optimum is allowed. The setting Algorithm -> Brelaz is the default, while the setting Algorithm -> Optimum forces the algorithm to do an exhaustive search to find an optimum vertex coloring."
];
If[Not[ValueQ[Combinatorica`VertexConnectivity::usage]],
Combinatorica`VertexConnectivity::usage = "VertexConnectivity[g] gives the minimum number of vertices whose deletion from graph g disconnects it. VertexConnectivity[g, Cut] gives a set of vertices of minimum size, whose removal disconnects the graph."
];
If[Not[ValueQ[VertexConnectivityGraph::usage]],
VertexConnectivityGraph::usage = "VertexConnectivityGraph[g] returns a directed graph that contains an edge corresponding to each vertex in g and in which edge disjoint paths correspond to vertex disjoint paths in g."
];
If[Not[ValueQ[VertexCover::usage]],
VertexCover::usage = "VertexCover[g] returns a vertex cover of the graph g. An option Algorithm that can take on values Greedy, Approximate, or Optimum is allowed. The default setting is Algorithm -> Approximate. Different algorithms are used to compute a vertex cover depending on the setting of the option Algorithm."
];
If[Not[ValueQ[Combinatorica`VertexCoverQ::usage]],
Combinatorica`VertexCoverQ::usage = "VertexCoverQ[g, c] yields True if the vertices in list c define a vertex cover of graph g."
];
If[Not[ValueQ[Combinatorica`VertexLabel::usage]],
Combinatorica`VertexLabel::usage = "VertexLabel is an option that can take on values True or False, allowing the user to set and display vertex labels. By default, there are no vertex labels. VertexLabel can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[VertexLabelColor::usage]],
VertexLabelColor::usage = "VertexLabelColor is an option that allows the user to associate different colors to vertex labels. Black is the default color. VertexLabelColor can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[VertexLabelPosition::usage]],
VertexLabelPosition::usage = "VertexLabelPosition is an option that allows the user to place a vertex label in a certain position relative to the vertex. The default position is upper right. VertexLabelPosition can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[VertexNumber::usage]],
VertexNumber::usage = "VertexNumber is an option that can take on values True or False. This can be used in ShowGraph to display or suppress vertex numbers. By default, the vertex numbers are hidden. VertexNumber can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[VertexNumberColor::usage]],
VertexNumberColor::usage = "VertexNumberColor is an option that can be used in ShowGraph to associate different colors to vertex numbers. Black is the default color. VertexNumberColor can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[VertexNumberPosition::usage]],
VertexNumberPosition::usage = "VertexNumberPosition is an option that can be used in ShowGraph to display a vertex number in a certain position relative to the vertex. By default, vertex numbers are positioned to the lower left of vertices. VertexNumberPosition can be set as part of the graph data structure or in ShowGraph."
];
If[Not[ValueQ[Combinatorica`VertexStyle::usage]],
Combinatorica`VertexStyle::usage = "VertexStyle is an option that allows the user to associate different sizes and shapes to vertices. A disk is the default shape. VertexStyle can be set as part of the graph data structure and it can be used in ShowGraph."
];
If[Not[ValueQ[Combinatorica`VertexWeight::usage]],
Combinatorica`VertexWeight::usage = "VertexWeight is an option that allows the user to associate weights with vertices. 0 is the default weight. VertexWeight can be set as part of the graph data structure."
];
If[Not[ValueQ[Vertices::usage]],
Vertices::usage = "Vertices[g] gives the embedding of graph g, that is, the coordinates of each vertex in the plane. Vertices[g, All] gives the embedding of the graph along with graphics options associated with each vertex."
];
If[Not[ValueQ[WaltherGraph::usage]],
WaltherGraph::usage = "WaltherGraph returns the Walther graph."
];
If[Not[ValueQ[Weak::usage]],
Weak::usage = "Weak is an option to ConnectedQ that seeks to determine if a directed graph is weakly connected."
];
If[Not[ValueQ[Combinatorica`WeaklyConnectedComponents::usage]],
Combinatorica`WeaklyConnectedComponents::usage = "WeaklyConnectedComponents[g] gives the weakly connected components of directed graph g as lists of vertices."
];
If[Not[ValueQ[WeightingFunction::usage]],
WeightingFunction::usage = "WeightingFunction is an option to the functions SetEdgeWeights and SetVertexWeights and it tells the functions how to compute edge weights and vertex weights, respectively. The default value for this option is Random."
];
If[Not[ValueQ[WeightRange::usage]],
WeightRange::usage = "WeightRange is an option to the functions SetEdgeWeights and SetVertexWeights that gives the range for these weights. The default range is [0, 1] for real as well as integer weights." 
];
If[Not[ValueQ[Wheel::usage]],
Wheel::usage = "Wheel[n] constructs a wheel on n vertices, which is the join of CompleteGraph[1] and Cycle[n-1]."
];
If[Not[ValueQ[WriteGraph::usage]],
WriteGraph::usage = "WriteGraph[g, f] writes graph g to file f using an edge list representation."
];
If[Not[ValueQ[Zoom::usage]],
Zoom::usage = "Zoom[{i, j, k, ...}] is a value that the PlotRange option can take on in ShowGraph. Setting PlotRange to this value zooms the display to contain the specified subset of vertices, i, j, k, ...."
];];

Begin["`Private`"]
(* Internal cache of graph options; referred to more than once,
   so kept here for convenience *)
$GraphEdgeStyleOptions = {
    Combinatorica`EdgeColor -> Black,
    EdgeDirection -> False, (* really semantic, but used in rendering *)
    Combinatorica`EdgeLabel -> False,
    EdgeLabelColor -> Black,
    EdgeLabelPosition -> LowerLeft,
    Combinatorica`EdgeStyle -> Normal,
    LoopPosition -> UpperRight
};
$GraphVertexStyleOptions = {
    VertexColor -> Black,
    Combinatorica`VertexLabel -> False,
    VertexLabelColor -> Black,
    VertexLabelPosition -> UpperRight,
    VertexNumber -> False,
    VertexNumberColor -> Black,
    VertexNumberPosition -> LowerLeft,
    Combinatorica`VertexStyle -> Disk[Normal]
};
$GraphSemanticOptions = {
    Combinatorica`EdgeWeight -> 1,
    Combinatorica`VertexWeight -> 0
};
AcyclicQ[g_Combinatorica`Graph] := SameQ[Combinatorica`FindCycle[g],{}]
Combinatorica`AddEdge::obsolete = "Usage of Directed as a second argument to AddEdge is obsolete."
Combinatorica`AddEdge[g_Combinatorica`Graph, edge:{_Integer,_Integer}, Directed] := (Message[Combinatorica`AddEdge::obsolete]; AddEdges[g, {{edge}}])
Combinatorica`AddEdge[g_Combinatorica`Graph, edge:{_Integer,_Integer}] := AddEdges[g, {{edge}}]
Combinatorica`AddEdge[g_Combinatorica`Graph, edge:{{_Integer,_Integer}, ___?OptionQ}] := AddEdges[g, {edge}]
AddEdges[g_Combinatorica`Graph, edgeList:{{{_Integer, _Integer},___?OptionQ}...}] := 
        Module[{ne = If[UndirectedQ[g], 
                        Map[Prepend[Rest[#], Sort[First[#]]]&, edgeList],
                        edgeList
                     ]
               },
               ChangeEdges[g, Join[Edges[g, All], ne]]
        ] 
AddEdges[g_Combinatorica`Graph, edgeList:{{_Integer,_Integer}...}] := 
        AddEdges[g, Map[{#}&, edgeList] ] 
AddEdges[g_Combinatorica`Graph, edge:{_Integer,_Integer}] := AddEdges[g, {edge}]
AddToEncroachingLists[k_Integer,{}] := {{k}}
AddToEncroachingLists[k_Integer,l_List] :=
	Append[l,{k}]  /; (k > First[Last[l]]) && (k < Last[Last[l]])
AddToEncroachingLists[k_Integer,l1_List] :=
	Module[{i,l=l1},
		If [k <= First[Last[l]],
			i = Ceiling[ BinarySearch[l,k,First] ];
			PrependTo[l[[i]],k],
			i = Ceiling[ BinarySearch[l,-k,(-Last[#])&] ];
			AppendTo[l[[i]],k]
		];
		l
	]
 
Combinatorica`AddVertex[g_Combinatorica`Graph] := AddVertices[g, 1]
Combinatorica`AddVertex[g_Combinatorica`Graph, p:{_?NumericQ, _?NumericQ}] := AddVertices[g, {{p}}]
AddVertices[g_Combinatorica`Graph, n_Integer?Positive] := Combinatorica`GraphUnion[g, EmptyGraph[n]]
AddVertices[Combinatorica`Graph[e_List, v_List, opts___?OptionQ], 
            p:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}] := Combinatorica`Graph[e, Join[v, p], opts]
AddVertices[g_Combinatorica`Graph, p:{{_?NumericQ, _?NumericQ}...}] := AddVertices[g, Map[{#}&, p]]
AddVertices[g_Combinatorica`Graph, p:{_?NumericQ, _?NumericQ}] := AddVertices[g, {p}]
AllPairsShortestPath[g_Combinatorica`Graph] := {} /; (V[g] == 0)
AllPairsShortestPath[g_Combinatorica`Graph] :=
        Module[{p = ToAdjacencyMatrix[g, Combinatorica`EdgeWeight, Type->Simple], m},
               m = V[g]*Ceiling[Max[Cases[Flatten[p], _Real | _Integer | _Rational]]]+1;
               Zap[DP1[p /. {0 -> 0.0, x_Rational -> 1.0 x, x_Integer -> 1.0 x, Infinity -> m*1.0}, m]] /. m -> Infinity
        ]
DP1 = Compile[{{p, _Real, 2}, {m, _Integer}},
        Module[{np = p, k, n = Length[p]},
               Do[np = Table[If[(np[[i, k]] == 1.0*m) || (np[[k, j]] == 1.0*m), 
                                np[[i,j]],
                                Min[np[[i,k]]+ np[[k,j]], np[[i,j]]]
                             ], {i,n},{j,n}
                       ], {k, n}];
               np
        ]
      ]
AllPairsShortestPath[g_Combinatorica`Graph, Parent] := {} /; (V[g] == 0)
AllPairsShortestPath[g_Combinatorica`Graph, Parent] :=
        Module[{q, p = ToAdjacencyMatrix[g,Combinatorica`EdgeWeight,Type->Simple], n=V[g], m},
               Do[p[[i, i]] = 0, {i, n}];
               m = V[g]*Ceiling[Max[Cases[Flatten[p], _Real | _Integer | _Rational]]]+1;
               q = Table[If[(p[[i, j]]===Infinity), j, i], 
                         {i, n}, {j, n}
                   ];
               p = p /. {x_Rational-> 1.0 x, x_Integer->1.0 x, Infinity -> m*1.0};
               {p, q} = Zap[DP2[p, q, m]];
               {p /. m -> Infinity, q} 
        ]
DP2 =Compile[{{p, _Real, 2}, {q, _Integer, 2}, {m, _Real}}, 
        Module[{np = p, nq = q, k = 0, n = Length[p]}, 
               Do[
                  Do[If[(np[[i, k]] != m*1.0) && (np[[k, j]] != m*1.0),
                        np[[i, j]] = Min[np[[i, k]] + np[[k, j]], np[[i, j]]];
                        If[(np[[i, j]] == np[[i, k]] + np[[k, j]]) && (k != j) && (i != j), 
                           nq[[i, j]] = nq[[k, j]]
                        ]
                     ], 
                     {i, n}, {j, n}
                  ], {k, n}
               ]; 
               {np, nq}
        ]
     ]
Combinatorica`AlternatingGroup[l_List] := Select[Permutations[l], (SignaturePermutation[#]===1)&] /; (Length[l] > 0)
Combinatorica`AlternatingGroup[n_Integer?Positive] := Select[Permutations[n], (SignaturePermutation[#]===1)&]
AlternatingGroupIndex[n_Integer?Positive, x_Symbol] := 
       Module[{p, y},
              p = SymmetricGroupIndex[n, y]; 
              (p /. Table[y[i] -> x[i], {i, n}]) + (p /. Table[y[i] -> (-1)^(i + 1)x[i], {i, n}])
       ]
AlternatingPaths[g_Combinatorica`Graph, start_List, ME_List] := 
       Module[{MV = Table[0, {V[g]}], e = ToAdjacencyLists[g], 
               lvl = Table[Infinity, {V[g]}], cnt = 1, u, v,
               queue = start, parent = Table[i, {i, V[g]}]},
               Scan[(MV[[#[[1]]]] = #[[2]]; MV[[#[[2]]]] = #[[1]]) &, ME]; 
               Scan[(lvl[[#]] = 0) &, start];
               While[queue != {},
                     {v, queue} = {First[queue], Rest[queue]};
                     If[EvenQ[lvl[[v]]],
                        Scan[(If[lvl[[#]] == Infinity, lvl[[#]] = lvl[[v]] + 1;
                              parent[[#]] = v; AppendTo[queue, #]]) &, e[[v]]
                        ],
                        If[MV[[v]] != 0,
                           u = MV[[v]];
                           If[lvl[[u]] == Infinity, 
                              lvl[[u]] = lvl[[v]] + 1; 
                              parent[[u]] = v; 
                              AppendTo[queue, u]
                           ]
                        ]
                     ]
               ];
               parent
       ]
Options[AnimateGraph] = Sort[
    $GraphEdgeStyleOptions ~Join~
    $GraphVertexStyleOptions ~Join~
    {HighlightedVertexStyle -> Disk[Large],
     HighlightedEdgeStyle -> Thick,
     HighlightedVertexColors -> ScreenColors,
     HighlightedEdgeColors -> ScreenColors} ~Join~
     Options[ListAnimate]
]
AnimateGraph[g_Combinatorica`Graph, l_List, flag_Symbol:All, opts___?OptionQ] := 
       ListAnimate[Map[ShowGraph, 
           If[flag === One,
              Table[Highlight[g, {{ l[[ i ]] }},
                        Sequence @@ FilterRules[{opts, Complement[Options[AnimateGraph], Options[Highlight]]}, Except[Options[ListAnimate]]]],
                    {i, Length[l]}],
              Table[Highlight[g, {l[[ Range[i] ]]},
                        Sequence @@ FilterRules[{opts, Complement[Options[AnimateGraph], Options[Highlight]]}, Except[Options[ListAnimate]]]],
                    {i, Length[l]}]
           ]
       ], Sequence @@ FilterRules[{opts, Options[AnimateGraph]}, Options[ListAnimate]]]
AntiSymmetricQ[r_?squareMatrixQ] := 
       Apply[And,
             Flatten[Table[r[[i, j]] != r[[j, i]], {i, Length[r]}, {j, i-1}], 1]
       ]
AntiSymmetricQ[g_Combinatorica`Graph] := 
	Module[{e = Edges[RemoveSelfLoops[g]]},
		Apply[And, Map[!MemberQ[e, Reverse[#]]&, e] ]
	] /; !UndirectedQ[g]
AntiSymmetricQ[g_Combinatorica`Graph] := M[RemoveSelfLoops[g]] == 0 
ApproximateVertexCover[g_Combinatorica`Graph] := ApproximateVertexCover[g] /; (!SimpleQ[g] || !UndirectedQ[g])
ApproximateVertexCover[g_Combinatorica`Graph] := 
       GreedyVertexCover[g, Apply[Join, MaximalMatching[g]]]
Arctan[{x_,y_}] := Arctan1[Chop[{x,y}]]
Arctan1[{0,0}] := 0
Arctan1[{x_,y_}] := ArcTan[x,y]
Arrows[pointPair_, mel_?NumericQ] :=
          Block[{size, triangle},
               (*size = Min[0.05, mel/3];*)
               size = 0.05;
               triangle={ {0,0}, {-size,size/2}, {-size,-size/2} };
               Polygon[TranslateVertices[
                           RotateVertices[
                               triangle, Arctan[Apply[Subtract, pointPair]]+Pi
                           ], 
                           pointPair[[2]]
                       ]
               ]
          ]
ArticulationVertices[g_Combinatorica`Graph]  := Union[Last[FindBiconnectedComponents[g]]];
AugmentFlow[f_List, m_, p_List] :=
        Module[{i, j, pf, pb},
               Scan[({i,j} = {#[[1]], #[[2]]};
                    pf = Position[f[[i]], {j, _}];
                    pb = Position[f[[j]], {i, _}];
                    If[(pb != {}) && (f[[j, pb[[1,1]], 2]] >= m),
                       f[[j, pb[[1,1]],2]]-=m,
                       f[[i, pf[[1,1]],2]]+=m
                    ])&,
                    p
               ]
        ]
(* TODO: Follownig code appears to be incomplete *)
AugmentingPath[g_Combinatorica`Graph,src_Integer,sink_Integer] :=
	Block[{l={src},lab=Table[0,{V[g]}],v,c=Edges[g,All],e=ToAdjacencyLists[g]},
		lab[[src]] = start;
		While[l != {} && (lab[[sink]]==0),
			{v,l} = {First[l],Rest[l]};
			Scan[ (If[ c[[v,#]] - flow[[v,#]] > 0 && lab[[#]] == 0,
				lab[[#]] = {v,f}; AppendTo[l,#]])&,
				e[[v]]
			];
			Scan[ (If[ flow[[#,v]] > 0 && lab[[#]] == 0,
				lab[[#]] = {v,b}; AppendTo[l,#]] )&,
				Select[Range[V[g]],(c[[#,v]] > 0)&]
			];
		];
		Combinatorica`Private`FindPath[lab,src,sink]
	]
Automorphisms[g_Combinatorica`Graph] := Isomorphism[g]
BFS[g_Combinatorica`Graph, start_Integer] :=
	Module[{e,bfi=Table[0,{V[g]}],cnt=1,queue={start}, v,
                parent=Table[i, {i, V[g]}],lvl=Table[Infinity,{V[g]}]},
		e = ToAdjacencyLists[g];
		bfi[[start]] = cnt++; lvl[[start]]=0;
		While[ queue != {},
			{v,queue} = {First[queue],Rest[queue]};
			Scan[(If[bfi[[#]] == 0,
                                 bfi[[#]] = cnt++; parent[[#]]=v; 
                                 lvl[[#]] = lvl[[v]]+1; AppendTo[queue,#]
                              ])&,
                              e[[v]]
                        ];
		];
		{bfi, parent, lvl}
	] /; (1 <= start) && (start <= V[g])
BFS[g_Combinatorica`Graph,s_Integer,t_Integer] :=
	Module[{queue={s}, parent=Table[Infinity, {i, V[g]}], e, v},
                If[s==t, Return[{s}]];
		e = ToAdjacencyLists[g];
                parent[[s]] = s;
		While[ queue != {},
			{v,queue} = {First[queue],Rest[queue]};
			Scan[(If[parent[[#]] == Infinity,
                                 parent[[#]] = v;
                                 AppendTo[queue,#];
                                 If[# == t, queue={};Return[] ];
                              ])&,
                             e[[v]]
			];
		];
                If[parent[[t]] == Infinity,
                   {},
                   Rest[Reverse[FixedPointList[Function[x, parent[[x]] ], t]]]
                ]
	] /; (1 <= s) && (s <= V[g]) && (1 <= t) && (t <= V[g])
				
Backtrack[space_List,partialQ_,solutionQ_,flag_:One] :=
	Module[{n=Length[space],all={},done,index, v=2, solution},
		index=Prepend[ Table[0,{n-1}],1];
		While[v > 0,
			done = False;
			While[!done && (index[[v]] < Length[space[[v]]]),
				index[[v]]++;
				done = Apply[partialQ,{Solution[space,index,v]}];
			];
			If [done, v++, index[[v--]]=0 ];
			If [v > n,
				solution = Solution[space,index,n];
				If [Apply[solutionQ,{solution}],
					If [SameQ[flag,All],
						AppendTo[all,solution],
						all = solution; v=0
					]
				];
				v--
			]
		];
		all
	]
BeforeQ[l_List,a_,b_] :=
	If [First[l]==a, True, If [First[l]==b, False, BeforeQ[Rest[l],a,b] ] ]
BellmanFord[g_Combinatorica`Graph, s_Integer?Positive] := 
         Module[{p = Table[i, {i, V[g]}], d = Table[Infinity, {V[g]}]},
                d[[s]] = 0;
                {p, d}
         ] /; EmptyQ[g] && (s <= V[g])
BF = Compile[{{n, _Integer}, {s, _Integer}, {e1, _Integer, 2}, 
              {w1, _Real, 1}, {e2, _Integer, 2}, {w2, _Real, 1}}, 
             Module[{d, dist, parent = Range[n], 
                     m = (Length[e1] + Length[e2])*Max[Abs[w1], Abs[w2]] + 1}, 
                     dist = Table[m, {n}]; dist[[s]] = 0; 
                     Do[
                         Do[d = dist[[ e1[[j, 1]] ]] + w1[[j]];
                            If[dist[[ e1[[j, 2]] ]] > d, 
                               dist[[ e1[[j, 2]] ]] = d; parent[[ e1[[j, 2]] ]] = e1[[j, 1]]], 
                            {j, Length[e1]}
                         ]; 
                         Do[d = dist[[ e2[[j, 1]] ]] + w2[[j]];
                            If[dist[[ e2[[j, 2]] ]] > d, 
                               dist[[ e2[[j, 2]] ]] = d; parent[[ e2[[j, 2]] ]] = e2[[j, 1]]], 
                            {j, Length[e2]}
                         ], 
                         {i, Ceiling[n/2]}
                     ]; 
                     {parent, dist}
             ]
     ]
BellmanFord[g_Combinatorica`Graph, s_Integer?Positive] := 
         Module[{e = Sort[Edges[g, Combinatorica`EdgeWeight]], n = V[g], e1 = {}, w1 = {}, 
                 e2 = {}, w2 = {}, b}, 
                If[UndirectedQ[g], 
                   {e1, w1} = Transpose[e]; 
                   {e2, w2} = Transpose[Reverse[Sort[{Reverse[#1], #2}& @@@ e]]];
                   b = Zap[BF[n, s, e1, 1.0*w1, e2, 1.0*w2]],
                   e1 = Select[e, #[[1, 1]] < #[[1, 2]] &];
                   e2 = Select[e, #[[1, 1]] > #[[1, 2]] &];
                   {e1, w1} = If[e1 != {}, Transpose[e1], {{{1, 1}}, {0.0}}]; 
                   {e2, w2} = If[e2 != {}, Transpose[e2], {{{1, 1}}, {0.0}}];
                   b = Zap[BF[n, s, e1, 1.0*w1, e2, 1.0*w2]]
                ];
                {b[[1]], 
                 Table[If[(i==b[[1, i]]) && (i != s), Infinity, b[[2, i]]], 
                       {i, Length[b[[2]]]}
                 ]}
         ] /; (s <= V[g])
BiconnectedComponents[g_Combinatorica`Graph]:= Map[{#}&, Range[V[g]] ]/; (EmptyQ[g])
BiconnectedComponents[g_Combinatorica`Graph] := First[FindBiconnectedComponents[g]] /; UndirectedQ[g]
BiconnectedComponents[g_Combinatorica`Graph] := First[FindBiconnectedComponents[MakeUndirected[g]]] 
BiconnectedQ[g_Combinatorica`Graph] := (Length[ BiconnectedComponents[g] ] == 1)
BinarySearch::error = "The input list is non-numeric."
BinarySearch[l_?(Length[#] > 0&), k_?NumericQ, f_:Identity]:= 
        With[{res = binarysearchchore[l, k, f]},
             res/; res =!= $Failed
        ]
binarysearchchore[l_, k_, f_]:=
        Module[{lo = 1, mid, hi = Length[l], el},
                    While[lo <= hi,
                        If[(el=f[l[[mid =
                                    Floor[ (lo + hi)/2 ]]]])===k,
                           Return[mid]
                        ];
            If[!NumericQ[el], (Message[BinarySearch::error]; Return[$Failed])];
                        If[el > k, hi = mid-1, lo = mid+1]
                    ];
                    Return[lo-1/2]
        ];
BinarySubsets[l_List] := Map[(l[[Flatten[Position[#, 1], 1]]])&, Strings[{0, 1}, Length[l]]]
BinarySubsets[0] := {{}}
BinarySubsets[n_Integer?Positive] := BinarySubsets[Range[n]]
BipartiteMatching[g_Combinatorica`Graph] :=
	Module[{p,v1,v2,coloring=TwoColoring[g],n=V[g]},
		v1 = Flatten[Position[coloring,1]];
		v2 = Flatten[Position[coloring,2]];
		p = BipartiteMatchingFlowGraph[MakeSimple[g],v1,v2];
		Complement[
                   Map[Sort[First[#]]&, NetworkFlow[p, n+1, n+2, Edge]],
                   Map[{#,n+1}&, v1], 
                   Map[{#,n+2}&, v2]
                ]
	] /; BipartiteQ[g] && UnweightedQ[g]
BipartiteMatching[g_Combinatorica`Graph] := First[BipartiteMatchingAndCover[g]] /; BipartiteQ[g]
BipartiteMatchingAndCover[g_Combinatorica`Graph] := 
       Module[{ng = g, MV, UV, U, S, T, epsilon, u , v, cover, diff, MM, WM,
               c = TwoColoring[g], OV1, OV2, V1, V2, r, ip, jp, Tbar, currentMatching},
              V1=OV1=Flatten[Position[c,1]]; V2=OV2=Flatten[Position[c, 2]];
              If[Length[V2] < Length[V1], 
                 {OV1, OV2} = {V1, V2} = {OV2, OV1};
                 ng = AddVertices[g, Length[V2]-Length[V1]];
                 V1 = Join[V1, Range[Length[V1]+Length[V2]+1, 2 Length[V2]]]
              ];
              MM = ToAdjacencyMatrix[ng, Combinatorica`EdgeWeight];
              WM = Table[MM[[ V1[[i]], V2[[j]]]], 
                         {i, Length[V1]}, {j, Length[V2]}] /. Infinity -> 0;
              u = Table[Max[WM[[i]]], {i, Length[V1]}];
              v = Table[0, {Length[V2]}];
              While[True,
                    cover = Table[u[[i]] + v[[j]], {i, Length[V1]}, {j, Length[V2]}];
                    ng = ChangeEdges[ng, 
                    Map[Sort[{V1[[#[[1]]]], V2[[#[[2]]]]}]&, Position[diff = cover - WM, 0]]];
                    currentMatching = BipartiteMatching[ng];
                    If[Length[currentMatching]==Length[V1], 
                       Return[{Intersection[currentMatching, Edges[g]], 
                               Transpose[
                                  Sort[Join[Transpose[{OV1, u[[Range[Length[OV1]]]]}],
                                            Transpose[{OV2, v}]
                                       ]
                                  ]
                               ][[2]]}
                       ]
                    ];
                    MV = Apply[Union, currentMatching];
                    U = Complement[V1, MV];
                    r = ReachableVertices[ng, U, currentMatching];
                    S = Complement[r, V2]; T = Complement[r, V1];
                    Tbar = Complement[V2, T];
                    epsilon = Min[Table[ip = Position[V1, S[[i]]][[1, 1]];
                                        jp = Position[V2, Tbar[[j]]][[1, 1]];
                                        diff[[ip, jp]], 
                                        {i, Length[S]}, {j, Length[Tbar]}
                                  ]
                              ];
                    Do[ip=Position[V1, S[[i]]][[1, 1]]; 
                       u[[ip]]=u[[ip]]-epsilon, 
                       {i, Length[S]}
                    ];
                    Do[jp  = Position[V2, T[[j]]][[1, 1]]; 
                       v[[jp]] = v[[jp]]+epsilon, 
                       {j, Length[T]}]
                    ]
              ]
BipartiteMatchingFlowGraph[g_Combinatorica`Graph, v1_List, v2_List] := 
       Module[{n = V[g], ng},
              ng = ChangeEdges[
                       SetGraphOptions[g, EdgeDirection -> True],
                       Map[If[MemberQ[v1, #[[1]]], #, Reverse[#]] &, Edges[g]]
                   ];
              AddEdges[AddVertices[ng, 2], 
                       Join[Map[{{n + 1, #}} &, v1], Map[{{#, n + 2}} &, v2]]
              ]
       ]
BipartiteQ[g_Combinatorica`Graph] := 
        Module[{c = TwoColoring[g]}, Apply[And, Map[c[[ #[[1]] ]] != c[[ #[[2]] ]]&, Edges[g]]]]
Options[BooleanAlgebra] = {Type -> Undirected, Combinatorica`VertexLabel->False}
BooleanAlgebra[n_Integer?Positive, opts___?OptionQ]:=
       Module[{type, label, s = Subsets[n], br},
              {type, label} = {Type, Combinatorica`VertexLabel} /. Flatten[{opts, Options[BooleanAlgebra]}];
              br = ((Intersection[#2,#1]===#1)&&(#1!=#2))&;
              If[type === Directed,
                 MakeGraph[s, br, Combinatorica`VertexLabel->label],
                 HasseDiagram[MakeGraph[s, br, Combinatorica`VertexLabel->label]]
              ]
       ]
Combinatorica`BreadthFirstTraversal[g_Combinatorica`Graph, s_Integer?Positive] := 
        First[BFS[g,s]] /; (s <= V[g])
Combinatorica`BreadthFirstTraversal[g_Combinatorica`Graph, s_Integer?Positive, t_Integer?Positive] := 
        BFS[g,s,t] /; (s <= V[g]) && (t <= V[g])
Combinatorica`BreadthFirstTraversal[g_Combinatorica`Graph, s_Integer?Positive, Edge] := 
        Module[{b  = BFS[g,s], v, p},
               v = InversePermutation[Cases[b[[1]], _?Positive]];
               p = b[[2]];
               Table[{p[[ v[[i]] ]], v[[i]]}, {i, 2, Length[v]}]
        ] /; (s <= V[g])
Combinatorica`BreadthFirstTraversal[g_Combinatorica`Graph, s_Integer?Positive, Tree] := 
        Module[{p = BFS[g,s][[2]]},
               ChangeEdges[
                   g,
                   Flatten[
                       Table[If[i!=p[[i]], {{{p[[i]],i}}}, {}], 
                             {i, Length[p]}
                       ], 1
                   ]
               ]
        ] /; (s <= V[g])
        
Combinatorica`BreadthFirstTraversal[g_Combinatorica`Graph, s_Integer?Positive, Level] := 
        Last[BFS[g,s]] /; (s <= V[g])
BrelazColoring[g_Combinatorica`Graph] := BrelazColoring[MakeSimple[g]] /; !UndirectedQ[g]
BrelazColoring[g_Combinatorica`Graph] := {} /; (V[g] == 0)
BrelazColoring[g_Combinatorica`Graph] :=
        Module[{cd = Table[0, {V[g]}], color = Table[0, {V[g]}], m = 0, p, nc,
                e = ToAdjacencyLists[g]},
               While[ m >= 0,
                      p = Position[cd, m][[1, 1]];
                      nc = Append[color[[ e[[p]] ]], 0];
                      color[[ p ]] = Min[Complement[ Range[Max[nc] + 1], nc]];
                      cd[[ p ]] = -2 V[g];
                      Scan[(cd[[ # ]]++)&, e[[ p ]] ];
                      m = Max[cd]
               ];
               color
        ]
Bridges[g_Combinatorica`Graph] := Select[BiconnectedComponents[g],(Length[#] == 2)&]
Options[Combinatorica`ButterflyGraph] = {Combinatorica`VertexLabel->False}
Combinatorica`ButterflyGraph[n_Integer?Positive, opts___?OptionQ] := 
        Module[{v = Map[Flatten, CartesianProduct[Strings[{0, 1}, n], Range[0, n]]], label},
               label = Combinatorica`VertexLabel /. Flatten[{opts, Options[Combinatorica`ButterflyGraph]}];
               RankedEmbedding[ 
                   MakeUndirected[
                       MakeGraph[v, 
                                 (#1[[n+1]]+1 == #2[[n+1]]) && 
                                 (#1[[Range[#2[[n+1]]-1]]] == #2[[Range[#2[[n+1]]-1]]]) && 
                                 (#1[[Range[#2[[n+1]]+1, n]]] == #2[[Range[#2[[n+1]]+1, n]]])&,
                                 Combinatorica`VertexLabel -> label
                       ]
                   ],
                   Flatten[Position[v, {__, 0}]]
               ]
        ]
CageGraph[g_Integer?Positive] := CageGraph[3,g]
CageGraph[3,3] := CompleteGraph[4]
CageGraph[3,4] := CompleteGraph[3,3]
CageGraph[3,5] := Combinatorica`PetersenGraph
CageGraph[3,6] := HeawoodGraph
CageGraph[3,7] := McGeeGraph
CageGraph[3,8] := LeviGraph
CageGraph[3,10] := 
        Module[{p, i, j},
               p = Combinatorica`GraphUnion[Combinatorica`CirculantGraph[20,{6}],Cycle[50]];
               AddEdges[p,
                        Join[Table[{7+5i,32+5i},{i,3,7}],
                             Table[{20+10i,10i+29},{i,4}],
                             Table[{15+10i,10i+24},{i,4}],
                             {{21,20},{1,23},{70,29},{24,65}},
                             Flatten[
                                     Table[{2i+j,21+5i+2j},{j,0,1},{i,9}], 1
                             ]
                         ]
                ]/. Combinatorica`Graph[l_List, v_List]:>
                    Combinatorica`Graph[l, 
                          Join[Table[{{Cos[#], Sin[#]}}&[2.Pi (i+1/2)/20],
                                     {i,20}
                               ],
                               2 Table[{{Cos[#],Sin[#]}}&[2.Pi (i+1/2)/50],
                                       {i,50}
                                 ]
                          ]
                    ]
        ]
CageGraph[4, 3] := CompleteGraph[5]
CageGraph[4, 4] := Combinatorica`CirculantGraph[8,{1,3}]
CageGraph[4, 5] := RobertsonGraph
CageGraph[4,6] := MakeUndirected[
                      MakeGraph[Range[26],
                            Mod[#1-#2, 26] == 1 ||
                                (-1)^#1Mod[#1-#2, 26] == 11 ||
                                (-1)^#1Mod[#1-#2, 26] == 7&,
                            Type -> Directed
                      ]
                  ]
CageGraph[5,3] := CompleteGraph[6]
CageGraph[5,4] := Combinatorica`CirculantGraph[10,{1,3,5}]
CageGraph[5,5] := MakeSimple[
                      AddEdges[
                          MakeGraph[Range[30],
                                    (Mod[#1-#2,30]==1 ||
                                    (Mod[#1-#2,30]==15 && Mod[#1,3]==0) ||
                                    (Mod[#1-#2,30]==4 && Mod[#1,2]==0))&
                          ],
                          {{1,9},{1,13},{1,19},{2,16},{3,11},{3,18},{3,25},
                           {4,20},{5,17},{5,23},{5,27},{6,21},{7,15},{7,19},
                           {7,25},{8,22},{9,17},{9,24},{10,26},{11,23},{11,29},
                           {12,27},{13,21},{13,25},{14,28},{15,23},{15,30},
                           {17,29}, {19,27},{21,29}}
                      ]
                  ]
CageGraph[5,6] := MakeUndirected[
                     MakeGraph[Range[42],
                            (Mod[#1-#2,42]==1 ||
                            (MemberQ[{7,27,31},Mod[#1-#2,42]] && Mod[#1,2]==1)||
                            (MemberQ[{11,15,35},Mod[#1-#2,42]]&&Mod[#1,2]==0))&
                     ]
                  ]
CalculateForce[u_Integer,g_Combinatorica`Graph,em_List] :=
	Module[{n=V[g],stc=0.25,gr=10.0,e=ToAdjacencyMatrix[g],
                f={0.0,0.0},spl=1.0,v,dsquared},
		Do [
			dsquared = Max[0.001, Apply[Plus,(em[[u]]-em[[v]])^2] ];
			f += (1-e[[u,v]]) (gr/dsquared) (em[[u]]-em[[v]])
				- e[[u,v]] stc Log[dsquared/spl] (em[[u]]-em[[v]]),
			{v,n}
		];
		f
	]
CartesianProduct[a_List, b_List] := Flatten[Outer[List, a, b, 1, 1], 1]
ChangeEdges[Combinatorica`Graph[_List, v_List, dopts___?OptionQ], 
            newE:{{{_Integer, _Integer},___?OptionQ}...}] := 
        Combinatorica`Graph[newE, v, dopts] /; (Max[Map[First, newE]] <= Length[v]) && (Min[Map[First, newE]] >= 1)
ChangeEdges[Combinatorica`Graph[_, v_, dopts___], newE:{{_Integer, _Integer}...}] :=
        Combinatorica`Graph[Map[{#}&, newE], v, dopts] /; (Max[newE] <= Length[v]) && (Min[newE] >= 1)
ChangeVertices[Combinatorica`Graph[e_List, v_List, dopts___?OptionQ], 
               newV:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}]  := 
        Combinatorica`Graph[e, newV, dopts] /; (Length[newV] >= Length[v])
ChangeVertices[Combinatorica`Graph[e_List, v_List, dopts___?OptionQ], 
               newV:{{_?NumericQ, _?NumericQ}...}]  := 
        Combinatorica`Graph[e, 
              Table[If[i <= Length[v], {newV[[i]], Apply[Sequence, Rest[v[[i]]]]}, {newV[[i]]}],{i, Length[newV]}], 
              dopts
        ] /; (Length[newV] >= Length[v])
ChooseShortestPathAlgorithm[g_Combinatorica`Graph, s_Integer, algorithm_Symbol] := 
         If[algorithm === Automatic,
            If[(MemberQ[Negative[GetEdgeWeights[g]],True]) || (M[g] <= 10 V[g]),
               First[BellmanFord[g, s]], First[Dijkstra[g, s]]],
            If[algorithm === BellmanFord, First[BellmanFord[g, s]], First[Dijkstra[g, s]]]
         ]
ChromaticDense[g_Combinatorica`Graph, z_] := ChromaticPolynomial[g,z] /; CompleteQ[g]
ChromaticDense[g_Combinatorica`Graph, z_] :=
        Block[{el = Edges[Combinatorica`GraphComplement[g]]}, 
              ChromaticDense[AddEdges[g,{{First[el]}}], z] 
              + ChromaticDense[MakeSimple[Combinatorica`Contract[g,First[el]]], z]
        ]
ChromaticNumber[g_Combinatorica`Graph] := 0 /; (V[g] == 0)
ChromaticNumber[g_Combinatorica`Graph] := 1 /; EmptyQ[MakeSimple[g]]
ChromaticNumber[g_Combinatorica`Graph] := 2 /; BipartiteQ[MakeSimple[g]]
ChromaticNumber[g_Combinatorica`Graph] := V[g] /; CompleteQ[MakeSimple[g]]
ChromaticNumber[g_Combinatorica`Graph] := Max[MinimumVertexColoring[g]]
ChromaticPolynomial[g_Combinatorica`Graph,z_]:= 
        ChromaticPolynomial[MakeSimple[g], z]/; !SimpleQ[g]
ChromaticPolynomial[g_Combinatorica`Graph,z_]:=0/;IdenticalQ[g,CompleteGraph[0]]
ChromaticPolynomial[g_Combinatorica`Graph,z_] :=
	Module[{i}, Product[z-i, {i,0,V[g]-1}] ] /; CompleteQ[g]
ChromaticPolynomial[g_Combinatorica`Graph,z_]:=z ( z - 1 ) ^ (V[g]-1) /; TreeQ[g]
ChromaticPolynomial[g_Combinatorica`Graph,z_] :=
	If[M[g]>Binomial[V[g],2]/2, ChromaticDense[g,z], ChromaticSparse[g,z]]
ChromaticSparse[g_Combinatorica`Graph,z_] := z^V[g] /; EmptyQ[g]
ChromaticSparse[g_Combinatorica`Graph,z_] :=
	Block[{e = Edges[g]},
		ChromaticSparse[ DeleteEdges[g,{First[e]}], z] -
		ChromaticSparse[MakeSimple[Combinatorica`Contract[g,First[e]]], z]
	]
ChvatalGraph := 
 Combinatorica`Graph[{{{6, 7}}, {{7, 8}}, {{8, 9}}, {{9, 10}}, {{6, 10}}, {{5, 6}},
  {{5, 9}}, {{3, 9}}, {{3, 7}}, {{1, 7}}, {{1, 10}}, {{4, 10}}, {{4, 8}},
  {{2, 8}}, {{2, 6}}, {{2, 11}}, {{5, 11}}, {{5, 12}}, {{3, 12}}, {{1, 11}},
  {{1, 12}}, {{4, 12}}, {{4, 11}}, {{2, 3}}},
 {{{-0.9510565162951535, 0.3090169943749475}},
  {{-0.5877852522924732, -0.8090169943749473}},
  {{0.5877852522924729, -0.8090169943749476}},
  {{0.9510565162951536, 0.3090169943749472}}, {{3.061515884555943*^-16, 1.}},
  {{-1.902113032590307, 0.618033988749895}},
  {{-1.1755705045849465, -1.6180339887498947}},
  {{1.1755705045849458, -1.6180339887498951}},
  {{1.9021130325903073, 0.6180339887498943}}, {{6.123031769111886*^-16, 2.}},
  {{-0.3, 0}}, {{0.3, 0}}}]
Combinatorica`CirculantGraph[n_Integer?Positive, l_Integer] := Combinatorica`CirculantGraph[n, {l}]
Combinatorica`CirculantGraph[n_Integer?Positive, l:{_Integer...}] :=
        Combinatorica`Graph[Union[
                  Flatten[
                      Table[Map[{Sort[{i, Mod[i+#, n]}]+1}&, l], 
                            {i,0,n-1}
                      ], 1
                  ],
                  Flatten[
                      Table[Map[{Sort[{i, Mod[i-#, n]}]+1}&, l], 
                            {i,0,n-1}
                      ], 1
                  ]
              ],
	      CircularEmbedding[n] 
        ]
CircularEmbedding[0] := {}
CircularEmbedding[n_Integer] :=
	Module[{i,x = N[2 Pi / n]},
		Chop[ Table[ N[{{ (Cos[x i]), (Sin[x i]) }}], {i,n} ] ]
	]
CircularEmbedding[g_Combinatorica`Graph] := 
        ChangeVertices[g, CircularEmbedding[ V[g] ] ]
CircularVertices[0] := {}
CircularVertices[n_Integer] :=
	Module[{i,x = N[2 Pi / n]},
		Chop[ Table[ N[{ (Cos[x i]), (Sin[x i]) }], {i,n} ] ]
	]
CircularVertices[g_Combinatorica`Graph] := ChangeVertices[g, CircularVertices[ V[g] ] ]
CliqueQ[g_Combinatorica`Graph, clique_List] :=
	IdenticalQ[CompleteGraph[Length[clique]], 
                      InduceSubgraph[MakeSimple[g],clique] 
        ]
CoarserSetPartitionQ[a_?SetPartitionQ, b_?SetPartitionQ] := 
        Apply[And, Map[Apply[Or, Map[Function[x, (Intersection[x, #] === #)], b] ]&, a ]]
CodeToLabeledTree[l_List] :=
	Module[{m=Range[Length[l]+2],x,i},
		FromUnorderedPairs[
			Append[
				Table[
					x = Min[Complement[m,Drop[l,i-1]]];
					m = Complement[m,{x}];
					Sort[{x,l[[i]]}],
					{i,Length[l]}
				],
				Sort[m]
			]
		]
	] /; (Complement[l, Range[Length[l]+2]] == {})
Cofactor[m_?MatrixQ, {i_Integer?Positive , j_Integer?Positive}] :=
	(-1)^(i+j) * Det[ Drop[ Transpose[ Drop[Transpose[m],{j,j}]], {i,i}]] /; (i <= Length[m]) && 
                                                                                 (j <= Length[m[[1]]]) 
CompleteBinaryTree[n_Integer?Positive] := 
       RootedEmbedding[Combinatorica`Graph[Join[Table[{{i, 2i}}, {i, Floor[n/2]}], 
                                  Table[{{i, 2i + 1}}, {i, Ceiling[n/2 - 1]}]],
                              CircularEmbedding[n]
                       ], 1
       ]
Options[CompleteGraph] = {Type -> Undirected};
CompleteGraph[n_Integer, opts___?OptionQ] := 
        Module[{type = Type /. Flatten[{opts, Options[CompleteGraph]}]},
              If[type === Undirected, CG[n], CDG[n]]
        ] /; (n >= 0)
CG[0] := Combinatorica`Graph[{},{}]
CG[1] := Combinatorica`Graph[{},{{{0,0}}}]
CG[n_Integer?Positive] := 
        Combinatorica`Graph[
                 Flatten[
                         Table[{{i, j}}, {i, n-1}, {j, i+1, n}], 1
                 ],
                 CircularEmbedding[n]
        ]
CDG[0]  := Combinatorica`Graph[{},{}, EdgeDirection -> True]
CDG[1] := Combinatorica`Graph[{},{{{0,0}}}, EdgeDirection -> True]
CDG[n_Integer?Positive] := 
        Combinatorica`Graph[Map[{#}&, 
                     Double[Flatten[
                                Table[{i, j}, {i, n-1}, {j, i+1, n}],
                                1
                            ]
                     ]
                 ],
                 CircularEmbedding[n],
                 EdgeDirection -> True
        ]
CompleteGraph[l__] :=
        CompleteKPartiteGraph[l] /; TrueQ[Apply[And, Map[Positive,List[l]]]] && (Length[List[l]]>1)
Options[CompleteKPartiteGraph] = {Type -> Undirected};
CompleteKPartiteGraph[l__, opts___?OptionQ] := 
        Module[{type = Type /. Flatten[{opts, Options[CompleteKPartiteGraph]}]},
              If[type === Undirected, 
                 CKPG[l], 
                 SetGraphOptions[CKPG[l], EdgeDirection -> True]
              ]
        ] /; TrueQ[Apply[And, Map[Positive,List[l]]]] && (Length[List[l]] > 0)
CKPG[l__] :=
        Module[{ll=List[l],t,i,x,row,stages=Length[List[l]]},
                t = FoldList[Plus,0,ll];
                AddEdges[Combinatorica`Graph[{},
                               Apply[Join,
                                     Table[Table[{{x,i-1+(1-ll[[x]])/2}}//N, 
                                                 {i,ll[[x]]}], 
                                           {x,stages}]
                               ]
                         ],
                         Flatten[
                            Table[
                                CartesianProduct[Range[t[[i-1]]+1, t[[i]]], 
                                                 Range[t[[i]]+1, t[[stages+1]]]],
                                {i, 2, stages}
                            ], 1
                         ]
                ]
        ] 
Combinatorica`CompleteKaryTree[n_Integer?Positive, k_Integer?Positive]:=
         RootedEmbedding[Combinatorica`Graph[
                            Join[
                               Flatten[
                                  Table[
                                     Table[{{i, j}}, {j,  k i-(k-2), k i+1}], 
                                     {i, Floor[(n-2)/k]} 
                                  ], 1
                               ], 
                               Table[{{Floor[(n-2)/k]+1, j}}, 
                                     {j, k Floor[(n-2)/k]+2 , n}
                               ]
                            ], 
                            CircularEmbedding[n]
                         ], 1
         ]
CompleteQ[g_Combinatorica`Graph] := 
        Block[{n = V[g], m = M[g]}, 
              (SimpleQ[g] && ((UndirectedQ[g] && (m == n (n-1)/2)) || (!UndirectedQ[g] && (m == n (n-1))))) 
        ]
Compositions[n_Integer,k_Integer] :=
	Map[
		(Map[(#[[2]]-#[[1]]-1)&, Partition[Join[{0},#,{n+k}],2,1] ])&,
		KSubsets[Range[n+k-1],k-1]
	]
Combinatorica`ConnectedComponents[g_Combinatorica`Graph] :=
        Block[{untraversed=Range[V[g]], visit, edges, comps={}, e=ToAdjacencyLists[g],
              parent=Table[0,{V[g]}], cnt=1, $RecursionLimit = Infinity, start},
              While[untraversed != {},
                    visit = {}; edges = {};
                    start = First[untraversed];
                    parent[[start]] = start;
                    DFS[start];
                    AppendTo[comps,visit];
                    untraversed = Complement[untraversed,visit]        
              ];
              ToCanonicalSetPartition[comps]
        ] /; UndirectedQ[g]
Combinatorica`ConnectedComponents[g_Combinatorica`Graph] := Combinatorica`ConnectedComponents[MakeUndirected[g]]
ConnectedQ[g_Combinatorica`Graph] := True /; (V[g] == 0)
ConnectedQ[g_Combinatorica`Graph, _] := True /; (V[g] == 0)
ConnectedQ[g_Combinatorica`Graph] := Length[Combinatorica`DepthFirstTraversal[g,1]]==V[g] /; UndirectedQ[g]
ConnectedQ[g_Combinatorica`Graph] := Length[Combinatorica`DepthFirstTraversal[MakeUndirected[g],1]]==V[g]
ConnectedQ[g_Combinatorica`Graph, Weak] := 
        Length[ Combinatorica`WeaklyConnectedComponents[g] ] == 1 /; !UndirectedQ[g]
 
ConnectedQ[g_Combinatorica`Graph, Strong] := 
        Length[ StronglyConnectedComponents[g] ] == 1 /; !UndirectedQ[g]
ConstructTableau[p_List] := 
       Module[{t = {}}, Map[(t = InsertIntoTableau[#, t]) &, p]; t]
Combinatorica`Contract[g_Combinatorica`Graph, l_List]  := 
       Module[{v = Vertices[g, All], t = Table[0, {V[g]}], 
               cnt = 0, last = V[g] - Length[l] + 1,
	       undirected = UndirectedQ[g]}, 
              Do[If[MemberQ[l, k], cnt++; t[[k]] = last, t[[k]] = k - cnt], {k, V[g]}]; 
              Combinatorica`Graph[
                 DeleteCases[Edges[g, All] /. {{x_Integer, y_Integer}, opts___?OptionQ} 
                                           :> {If[undirected, Sort, Identity][{t[[x]], t[[y]]}], opts}, {{last, last}, ___?OptionQ}
                 ],
                 Append[v[[Complement[Range[Length[v]], l]]],
                       {Apply[Plus, Map[First, v[[l]]]]/Length[l]}
                 ],
                 Apply[Sequence, GraphOptions[g]]
              ]
       ]
Convert[l_List] := 
	Module[{ch,num,edge={},i=1},
		While[i <= Length[l],
			If[ DigitQ[ l[[i]] ], 
				num = 0;
				While[ ((i <= Length[l]) && (DigitQ[l[[i]]])),
					num = 10 num + Toascii[l[[i++]]] - Toascii["0"]
				];
				AppendTo[edge,num],
				i++
			];
		];
		edge
	]
CostOfPath[g_Combinatorica`Graph, p_List] :=
        Block[{w = GetEdgeWeights[g], pos},
              If[UndirectedQ[g],
                  pos = Map[Position[Sort /@ Edges[g], #]&,
                         Map[Sort, Partition[p, 2, 1]]
                     ],
                   pos = Map[Position[Edges[g], #]&,
                         Partition[p, 2, 1]
                     ]
              ];
              If[MemberQ[pos, {}], 
                 Infinity,
                 Apply[Plus, w[[ Map[#[[1, 1]]&, pos] ]]
                 ]
              ]
        ]
CoxeterGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 8}}, {{2, 5}}, {{2, 14}}, {{3, 4}}, {{3, 9}}, 
  {{4, 7}}, {{4, 10}}, {{5, 6}}, {{5, 13}}, {{6, 7}}, {{6, 12}}, {{7, 11}}, 
  {{8, 20}}, {{8, 25}}, {{9, 21}}, {{9, 24}}, {{10, 15}}, {{10, 23}}, 
  {{11, 16}}, {{11, 22}}, {{12, 17}}, {{12, 28}}, {{13, 18}}, {{13, 27}}, 
  {{14, 19}}, {{14, 26}}, {{15, 18}}, {{15, 19}}, {{16, 19}}, {{16, 20}}, 
  {{17, 20}}, {{17, 21}}, {{18, 21}}, {{22, 24}}, {{22, 27}}, {{23, 25}}, 
  {{23, 28}}, {{24, 26}}, {{25, 27}}, {{26, 28}}}, 
 {{{0.412, 0.984}}, {{0.494, 0.984}}, {{0.366, 0.926}}, {{0.388, 0.862}}, 
  {{0.546, 0.926}}, {{0.518, 0.86}}, {{0.458, 0.818}}, {{0.152, 0.684}}, 
  {{0.264, 0.682}}, {{0.354, 0.68}}, {{0.458, 0.67}}, {{0.554, 0.672}}, 
  {{0.658, 0.668}}, {{0.774, 0.692}}, {{0.164, 0.45}}, {{0.228, 0.448}}, 
  {{0.274, 0.39}}, {{0.242, 0.33}}, {{0.194, 0.278}}, {{0.146, 0.328}}, 
  {{0.102, 0.39}}, {{0.668, 0.472}}, {{0.638, 0.416}}, {{0.656, 0.334}}, 
  {{0.714, 0.27}}, {{0.798, 0.326}}, {{0.83, 0.408}}, {{0.754, 0.466}}}]
CubeConnectedCycle[d_Integer] := 
        Module[{g = Hypercube[d], al, n, v }, 
               al = ToAdjacencyLists[g]; 
               n = V[g]; 
               v = Vertices[g]; 
               InduceSubgraph[
                   AddEdges[
                       AddEdges[
                           AddVertices[g, 
                               Flatten[Table[Map[(.3 #+.7 v[[i]])&,  
                                                 v[[al[[i]]]]
                                             ] , {i, n}
                                       ], 1
                               ]
                           ], 
                           Flatten[
                               Table[Append[
                                         Partition[
                                              Range[n+(i-1)d+1,n+d i],2,1
                                         ],
                                         {n + (i - 1)d + 1, n + d i}
                                     ], 
                                     {i, n}
                               ], 1
                           ]
                       ], 
                       Union[
                           Flatten[
                               Table[
                                   MapIndexed[
                                       Sort[{n+d (i-1)+#2[[1]], 
                                             n+d (#1-1)+
                                             Position[al[[#1]],i][[1,1]]
                                            }
                                       ]&, 
                                       al[[i]] 
                                   ], 
                                   {i, n}
                               ], 1
                           ]
                       ]
                   ], 
                   Range[n+1, n (d+1)]
               ]
        ]
CubicalGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{2, 3}}, {{3, 4}}, {{1, 4}}, {{5, 6}}, {{6, 7}}, {{7, 8}}, 
  {{5, 8}}, {{1, 5}}, {{2, 6}}, {{3, 7}}, {{4, 8}}}, 
 {{{0, 1.}}, {{-1., 0}}, {{0, -1.}}, {{1., 0}}, {{0, 2.}}, {{-2., 0}}, 
  {{0, -2.}}, {{2., 0}}}]
Options[Cycle] = {Type -> Undirected};
Cycle[n_Integer, opts___?OptionQ] := 
        Module[{type = Type /. Flatten[{opts, Options[Cycle]}],
                e = Table[{{i, i+1}}, {i, n-1}], c = CircularEmbedding[n]},
               If[type === Undirected,
                  Combinatorica`Graph[Append[e, {{1, n}}], c],
                  Combinatorica`Graph[Append[e, {{n, 1}}], c, EdgeDirection -> True]
               ]
        ] /; n>=2
CycleIndex[g_List, x_Symbol] := 
        Expand[Apply[Plus, Map[CycleStructure[#, x]&, g]]/Length[g]] /; (Length[g] > 0)
CycleStructure[p_?Combinatorica`PermutationQ, x_Symbol] := Apply[Times, Map[x[Length[#]]&, Combinatorica`ToCycles[p]]]
Combinatorica`CyclicGroup[0] := {{}}
Combinatorica`CyclicGroup[n_Integer] := Table[RotateRight[Range[n], i], {i, 0, n-1}]
CyclicGroupIndex[n_Integer?Positive, x_Symbol] :=
        Expand[Apply[Plus, Map[x[#]^(n/#) EulerPhi[#]&, Divisors[n]] ]/n ]
DFS[v_Integer] :=
	( AppendTo[visit,v];
	  Scan[ (If[parent[[#]]==0, AppendTo[edges,{v,#}]; parent[[#]] = v; DFS[#]])&, e[[v]] ])
DValues[0, m_Integer?Positive] := 1
DValues[t_Integer?Positive, m_Integer?Positive] :=
        Block[{$RecursionLimit = Infinity},
              DValues[t, m] = DValues[t - 1, m + 1] + m DValues[t - 1, m]
        ]
Options[Combinatorica`DeBruijnGraph] = {Combinatorica`VertexLabel->False}
Combinatorica`DeBruijnGraph[m_Integer?Positive, n_Integer?Positive] := 
        Block[{alph, states, s, i, j},
              alph = Table[i, {i, 0, m-1}];
              states = Strings[alph, n];
              s = Length[states];
              Combinatorica`Graph[
                       Flatten[ 
                               Table[
                                     Table[{{i, 
                                           Position[states, 
                                                    Append[Rest[ states[[i]] ], 
                                                           alph[[j]] 
                                                    ]
                                           ][[1, 1]]
                                           }},
                                           {j, Length[alph]}
                                     ],
                                     {i, s}
                               ], 1
                       ],
                       CircularEmbedding[s],
                       EdgeDirection->True
              ]
        ] 
Combinatorica`DeBruijnGraph[alph_List, n_Integer?Positive, opts___?OptionQ] := 
        Module[{label, nalpha = Union[alph]},
               label = Combinatorica`VertexLabel /. Flatten[{opts, Options[Combinatorica`DeBruijnGraph]}];
               If[label === True || label === On,
                  SetVertexLabels[
                     Combinatorica`DeBruijnGraph[Length[nalpha], n], 
                     Map[Apply[StringJoin, Map[ToString, #]] &, Strings[nalpha, n]]
                  ],
                  Combinatorica`DeBruijnGraph[Length[nalpha], n]
               ]
        ] /; (alph != {})
Combinatorica`DeBruijnSequence[{}, n_Integer?Positive] := {}
Combinatorica`DeBruijnSequence[alph_List, 1] := Union[alph]
Combinatorica`DeBruijnSequence[alph_List, n_Integer?Positive] := 
              Rest[Strings[Union[alph], n-1]
                   [[ EulerianCycle[Combinatorica`DeBruijnGraph[Union[alph], n-1]], 1]]
              ] 
DegreeSequence[g_Combinatorica`Graph] := Reverse[ Sort[ Degrees[g] ] ]
Degrees[g_Combinatorica`Graph] := Map[Length, ToAdjacencyLists[g]]
DegreesOf2Neighborhood[g_Combinatorica`Graph, v_Integer?Positive] := 
        Module[{al = ToAdjacencyLists[g], degrees = Degrees[g]},
               Sort[degrees[[ Neighborhood[al, v, 2] ]]]
        ]
DeleteCycle::obsolete = "Usage of Directed as a second argument to DeleteCycle is obsolete."
DeleteCycle[g_Combinatorica`Graph, c_List, Directed] := (Message[DeleteCycle::obsolete]; DeleteCycle[g, c])
DeleteCycle[g_Combinatorica`Graph, {}] := g
DeleteCycle[g_Combinatorica`Graph, c_List] :=
        Module[{e = If[UndirectedQ[g],
                       Map[Sort, Partition[c, 2, 1] ],
                       Partition[c, 2, 1]
                    ]
               },
               If[Complement[e, Edges[g]] == {}, DeleteEdges[g, e], g]
        ] /; (Last[c] == First[c])
Combinatorica`DeleteEdge::obsolete = "Usage of Directed as a second argument to DeleteEdge is obsolete."
Combinatorica`DeleteEdge[g_Combinatorica`Graph, ne:{_Integer, _Integer}] := DeleteEdges[g, {ne}]
Combinatorica`DeleteEdge[g_Combinatorica`Graph, ne:{_Integer, _Integer}, All] := DeleteEdges[g, {ne}, All]
Combinatorica`DeleteEdge[g_Combinatorica`Graph, ne:{_Integer, _Integer}, Directed] := (Message[Combinatorica`DeleteEdge::obsolete]; DeleteEdges[g, {ne}])
DeleteEdges[g_Combinatorica`Graph, ne:{_Integer, _Integer}, All] := DeleteEdges[g, {ne}, All]
DeleteEdges[g_Combinatorica`Graph, ne:{{_Integer, _Integer}...}, All] :=
        Module[{nne},
               nne = If[UndirectedQ[g], Join[ne, Reverse /@ ne], ne];
               ChangeEdges[g, 
                           Select[Edges[g, All], 
                                  (!MemberQ[nne, First[#]])& 
                           ]
               ]
        ]
DeleteEdges[g_Combinatorica`Graph, ne:{_Integer, _Integer}] := DeleteEdges[g, {ne}]
DeleteEdges[g_Combinatorica`Graph, ne:{{_Integer, _Integer}...}] :=
        Module[{el = Edges[g, All], nne, p},
               nne = If[UndirectedQ[g], Join[ne, Reverse /@ ne], ne];
               ChangeEdges[g, 
                           DeleteCases[
                               Table[If[(p = Position[ nne, el[[i,1]] ]) != {},
                                        nne = MapAt[Infinity&, nne, p[[1]] ];
                                        {},
                                        el[[ i ]]
                                     ],
                                     {i, M[g]}
                               ],
                               {} 
                           ] 
               ] 
        ]
DeleteFromTableau[t1_?Combinatorica`TableauQ,r_Integer]:=
	Module [{t=t1, col, row, item=Last[t1[[r]]]},
		col = Length[t[[r]]];
		If[col == 1, t = Drop[t,-1], t[[r]] = Drop[t[[r]],-1]];
		Do [
			While [t[[row,col]]<=item && Length[t[[row]]]>col, col++];
			If [item < t[[row,col]], col--];
			{item,t[[row,col]]} = {t[[row,col]],item},
			{row,r-1,1,-1}
		];
		t
	]
 
Combinatorica`DeleteVertex[g_Combinatorica`Graph,v_Integer] := InduceSubgraph[g, Complement[Range[V[g]],{v}]]
DeleteVertices[g_Combinatorica`Graph,vl_List] := InduceSubgraph[g, Complement[Range[V[g]],vl]]
Combinatorica`DepthFirstTraversal[g_Combinatorica`Graph, start_Integer, flag_:Vertex] :=
	Block[{visit={},e=ToAdjacencyLists[g],edges={},
               parent=Table[0,{V[g]}], cnt=1,
               $RecursionLimit = Infinity},
              parent[[start]] = start;
	      DFS[start];
	      Switch[flag, Edge, edges, 
                           Tree, ChangeEdges[g, edges],
                           Vertex, visit
              ]
	] /; (1 <= start) && (start <= V[g])
DerangementQ[p_?Combinatorica`PermutationQ] := !(Apply[ Or, Map[( # === p[[#]] )&, Range[Length[p]]] ])
Derangements[0] := { {} }
Derangements[n_Integer] := Derangements[Range[n]]
Derangements[p_?Combinatorica`PermutationQ] := Select[ Permutations[p], DerangementQ ]
Diameter[g_Combinatorica`Graph] := Max[ Eccentricity[g] ]
Combinatorica`DihedralGroup[0] := {{}}
Combinatorica`DihedralGroup[1] := {{1}}
Combinatorica`DihedralGroup[2] := {{1, 2}, {2, 1}}
Combinatorica`DihedralGroup[n_Integer?Positive] := Module[{c = Combinatorica`CyclicGroup[n]}, Join[c, Map[Reverse, c]]]
DihedralGroupIndex[n_Integer?Positive , x_Symbol] :=
        Expand[Simplify[CyclicGroupIndex[n, x]/2 + 
                        If[EvenQ[n], 
                           (x[2]^(n/2) + x[1]^2x[2]^(n/2-1))/4,
                           (x[1]x[2]^((n-1)/2))/2
                        ]
               ]
        ]
Dijkstra[al_List, start_Integer] :=
	Module[{dist = Table[Infinity,{i, Length[al]}],
                parent = Table[i, {i, Length[al]}],
                untraversed = Range[Length[al]],
                m, n, v},
               dist[[start]] = 0;
               While[untraversed != {},
                     m = Infinity;
                     Scan[(If[dist[[#]]<=m, v=#;m=dist[[#]]])&, untraversed];
                     untraversed = Complement[untraversed, {v}];
                     n = Table[{al[[v, i, 1]],  m + al[[v, i, 2]]}, {i, Length[ al[[v]] ]}];
                     Scan[If[dist[[ #[[1]] ]] > #[[2]], dist[[ #[[1]] ]] = #[[2]]; parent[[#[[1]]]] = v]&, 
                          n
                     ];
               ];
               {parent, dist}
        ]
Dijkstra[g_Combinatorica`Graph, start_Integer] :=
        Dijkstra[ToAdjacencyLists[g, Combinatorica`EdgeWeight], start]
Dijkstra[g_Combinatorica`Graph, start_List] :=
        Module[{al = ToAdjacencyLists[g, Combinatorica`EdgeWeight]},
               Map[Dijkstra[ToAdjacencyLists[g, Combinatorica`EdgeWeight], #]&, start]
        ]
DilateVertices[v:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}, d_] := 
       Module[{p = Map[First, v], np},
              np = DilateVertices[p, d];
              Table[{np[[i]], Apply[Sequence, Rest[v[[i]]]]}, {i, Length[np]}]
       ]
DilateVertices[v:{{_?NumericQ, _?NumericQ}...}, d_] := Map[(d * #)&, v]
DilateVertices[g_Combinatorica`Graph, d_] := ChangeVertices[g, DilateVertices[Vertices[g, All], d]]
DilateVertices[g_Combinatorica`Graph, s_List, t_] :=
       Module[{v = Vertices[g, All]},
              ChangeVertices[g, v[[s]] = DilateVertices[v[[s]], t]; v]
       ]
DilworthGraph[g_Combinatorica`Graph] :=
	FromUnorderedPairs[
		Map[
			(#+{0,V[g]})&,
			ToOrderedPairs[RemoveSelfLoops[TransitiveReduction[g]]]
		]
	]
Distance[{p1_List, p2_List}] := 
        Distance[ {p1, p2}, LNorm[2] ]
Distance[{p1_List, p2_List}, Euclidean] := 
        Distance[ {p1, p2}, LNorm[2] ]
Distance[{p1:{(_Integer|_Real), (_Integer|_Real)}, 
          p2:{(_Integer|_Real),(_Integer|_Real)}}, LNorm[Infinity]] := 
        N[Max[ Abs[p1[[1]] - p2[[1]] ], Abs[p1[[2]] - p2[[2]] ] ] ]
Distance[{p1:{(_Integer|_Real), (_Integer|_Real)}, 
          p2:{(_Integer|_Real),(_Integer|_Real)}},LNorm[x_Integer?Positive]] := 
        N[(Abs[p1[[1]] - p2[[1]] ]^x + Abs[p1[[2]] - p2[[2]] ]^x)^(1/x)]
Distances[g_Combinatorica`Graph, v_Integer?Positive] := Sort[Combinatorica`BreadthFirstTraversal[g, v, Level]]
DistinctPermutations[s_List] := Permutations[s] /; (Length[s] <= 1)
DistinctPermutations[s_List] :=
	Module[{freq,alph=Union[s],n=Length[s]},
		freq = Map[ (Count[s,#])&, alph];
		Map[
			(alph[[#]])&,
			Backtrack[
				Table[Range[Length[alph]],{n}],
				(Count[#,Last[#]] <= freq[[Last[#]]])&,
				(Count[#,Last[#]] <= freq[[Last[#]]])&,
				All
			]
		]
	]
 
Distribution[l_List] := Distribution[l, Union[l]]
Distribution[l_List, set_List] := Map[(Count[l,#])&, set]
DodecahedralGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{1, 5}}, {{1, 6}}, {{2, 3}}, {{2, 7}}, {{3, 4}}, {{3, 8}}, 
  {{4, 5}}, {{4, 9}}, {{5, 10}}, {{6, 11}}, {{6, 12}}, {{7, 11}}, {{7, 15}}, 
  {{8, 14}}, {{8, 15}}, {{9, 13}}, {{9, 14}}, {{10, 12}}, {{10, 13}}, 
  {{11, 16}}, {{12, 17}}, {{13, 18}}, {{14, 19}}, {{15, 20}}, {{16, 17}}, 
  {{16, 20}}, {{17, 18}}, {{18, 19}}, {{19, 20}}}, 
 {{{0.546, 0.956}}, {{0.144, 0.65}}, {{0.326, 0.188}}, {{0.796, 0.188}}, 
  {{0.988, 0.646}}, {{0.552, 0.814}}, {{0.264, 0.616}}, {{0.404, 0.296}}, 
  {{0.752, 0.298}}, {{0.846, 0.624}}, {{0.43, 0.692}}, {{0.682, 0.692}}, 
  {{0.758, 0.492}}, {{0.566, 0.358}}, {{0.364, 0.484}}, {{0.504, 0.602}}, 
  {{0.608, 0.602}}, {{0.634, 0.51}}, {{0.566, 0.444}}, {{0.48, 0.51}}}]
DominatingIntegerPartitionQ[a_List, b_List] := 
        Module[{aa  = Table[0, {Length[a]}], 
                bb = Table[0, {Length[b]}]}, 
               (Length[a] <= Length[b]) && 
               (aa[[1]] = a[[1]]; 
                Do[aa[[i]] = aa[[i - 1]] + a[[i]], {i, 2, Length[a]}]; 
                bb[[1]] = b[[1]]; 
                Do[bb[[i]] = bb[[i - 1]] + b[[i]], {i, 2, Length[b]}]; 
                Apply[And, Table[aa[[i]] >= bb[[i]], {i, Length[a]}]]
               )
        ]
Options[DominationLattice] = {Type -> Undirected, Combinatorica`VertexLabel->False}
DominationLattice[n_Integer?Positive, opts___?OptionQ] :=
         Module[{type, label, s = Partitions[n], br},
                {type, label} = {Type, Combinatorica`VertexLabel} /. Flatten[{opts, Options[DominationLattice]}];
                br = DominatingIntegerPartitionQ[#2, #1]&;
                If[type === Directed,
                   MakeGraph[s, br, Combinatorica`VertexLabel->label],
                   HasseDiagram[MakeGraph[s, br, Combinatorica`VertexLabel->label]]
                ]
         ]
Double[e_List] := Join[Map[Reverse, Select[e,(#[[1]]!=#[[2]])&]], e ]
Double[e_List, Combinatorica`EdgeWeight] :=
        Join[Map[Prepend[Rest[#],Reverse[ #[[1]] ]]&,
                 Select[e,(#[[1,1]]!=#[[1,2]])&]
             ],
             e
        ]
Double[e_List, All] := Double[e, Combinatorica`EdgeWeight]
DurfeeSquare[s_List] :=
	Module[{i,max=1},
		Do [
			If [s[[i]] >= i, max=i],
			{i,2,Min[Length[s],First[s]]}
		];
		max
	]
DurfeeSquare[{}] := 0
Eccentricity[g_Combinatorica`Graph, start_Integer, NoEdgeWeights] := Max[ Combinatorica`BreadthFirstTraversal[g, start, Level] ] 
Eccentricity[g_Combinatorica`Graph, start_Integer] := Eccentricity[g, start, NoEdgeWeights] /; UnweightedQ[g]
Eccentricity[g_Combinatorica`Graph, start_Integer] := Map[Max, Last[BellmanFord[g, start]]]
Eccentricity[g_Combinatorica`Graph] := Table[Eccentricity[g, i, NoEdgeWeights], {i, V[g]}] /; UnweightedQ[g]
Eccentricity[g_Combinatorica`Graph] := Map[ Max, AllPairsShortestPath[g] ]
EdgeChromaticNumber[g_Combinatorica`Graph] := ChromaticNumber[ Combinatorica`LineGraph[g] ]
EdgeColoring[g_Combinatorica`Graph] :=
        Module[{c = VertexColoring[Combinatorica`LineGraph[g]], e = Edges[g], se},
               se = Sort[ Table[{e[[i]], i}, {i, Length[e]}]];
               Map[Last, Sort[Map[Reverse, Table[Prepend[se[[i]], c[[i]]], {i, Length[se]}]]]]
        ]
Combinatorica`EdgeConnectivity[gin_Combinatorica`Graph] :=
    Module[{i, g = gin},
        If[MultipleEdgesQ[g],
	    Message[Combinatorica`EdgeConnectivity::multedge];
            g=RemoveMultipleEdges[g, True]
        ];
	Apply[Min, Table[NetworkFlow[g,1,i], {i, 2, V[g]}]]
    ]
Combinatorica`EdgeConnectivity[gin_Combinatorica`Graph, Cut] := 
        Module[{i, c, g = gin}, 
	    If[MultipleEdgesQ[g],
                Message[Combinatorica`EdgeConnectivity::multedge];
                g=RemoveMultipleEdges[g, True]
            ];
            Last[First[Sort[Table[{Length[c = NetworkFlow[g,1,i,Cut]], c}, 
                                     {i, 2, V[g]}
                               ]
                          ]
                    ]
               ]
        ]
EdgeGroup[g_] := KSubsetGroup[g, KSubsets[Range[Max[g[[1]]]], 2]]
EdgeGroupIndex[g_, x_Symbol] := EdgeGroup[CycleIndex[g, x], x] /; Combinatorica`PermutationQ[g[[1]]]
EdgeGroupIndex[ci_?PolynomialQ, x_Symbol]:=
        Module[{f1,f2,f3,i,PairCycles},
               f1[x[i1_]^(j1_)] := 1;
               f1[x[i1_]] := 1;
               f1[x[i1_]*x[(i2_)^(j2_)]] :=
                       x[LCM[i1, i2]]^(j2*GCD[i1, i2]);
               f1[x[i1_]^(j1_)*x[i2_]] :=
                       x[LCM[i1, i2]]^(j1*GCD[i1, i2]);
               f1[x[i1_]*x[i2_]] := x[LCM[i1, i2]]^GCD[i1, i2];
               f1[x[i1_]^(j1_)*x[i2_]^(j2_)] :=
                       x[LCM[i1, i2]]^(j1*j2*GCD[i1, i2]);
               f1[(a_)*(t__)] :=
                       Product[f1[a*{t}[[i]]], {i, Length[{t}]}]*
                       f1[Apply[Times, {t}]];
               f2[x[i1_]^j1_]:=x[i1]^(i1 Binomial[j1,2]);
               f2[x[i1_]]:=1;
               f2[a_  b_ ]:=f2[a] f2[b];
               f3[x[i1_]]:=If[OddQ[i1],x[i1]^( (i1-1)/2),
                       x[i1]^( (i1-2)/2) * x[i1/2]];
               f3[x[i1_]^j1_]:=If[OddQ[i1],x[i1]^(j1 (i1-1)/2),
                       x[i1]^(j1 (i1-2)/2) * x[i1/2]^j1];
               f3[a_ b_]:=f3[a] f3[b];
               PairCycles[u_ + v_]:=PairCycles[u]+ PairCycles[v];
               PairCycles[a_?NumericQ b_]:=a PairCycles[b];
               PairCycles[a_]:=f1[a] f2[a] f3[a];
               Expand[PairCycles[ci]]
        ]
Edges[Combinatorica`Graph[e_List, _List, ___?OptionQ]] := Map[First[#]&, e]
Edges[Combinatorica`Graph[e_List, _List, ___?OptionQ], All] := e
Edges[g_Combinatorica`Graph, Combinatorica`EdgeWeight] :=
         Map[{First[#], Combinatorica`EdgeWeight} /. 
             Flatten[{Rest[#], GraphOptions[g], Options[Combinatorica`Graph]}]&, 
             Edges[g, All]
         ]
Element[a_List,{index___}] := a[[ index ]]
Options[EmptyGraph] = {Type -> Undirected};
EmptyGraph[n_Integer, opts___?OptionQ] := 
        Module[{type = Type /. Flatten[{opts, Options[EmptyGraph]}]},
               If[type === Undirected, EG[n], EDG[n]]
        ] /; (n >= 0)
EG[0] := Combinatorica`Graph[{}, {}]
EDG[0] := Combinatorica`Graph[{}, {}, EdgeDirection -> True]
EG[n_Integer?Positive] := Combinatorica`Graph[{}, CircularEmbedding[n]]
EDG[n_Integer?Positive] := Combinatorica`Graph[{}, CircularEmbedding[n], EdgeDirection -> True]
EmptyQ[g_Combinatorica`Graph] := (Length[Edges[g]]==0)
EncroachingListSet[l_List?Combinatorica`PermutationQ] := EncroachingListSet[l,{}]
EncroachingListSet[{},e_List] := e
EncroachingListSet[l_List,e_List] :=
        Block[{$RecursionLimit = Infinity},
	      EncroachingListSet[Rest[l], AddToEncroachingLists[First[l],e] ]
        ]
EquivalenceClasses[r_List?EquivalenceRelationQ] := Combinatorica`ConnectedComponents[ FromAdjacencyMatrix[r] ]
EquivalenceClasses[g_Combinatorica`Graph?EquivalenceRelationQ] := Combinatorica`ConnectedComponents[g]
EquivalenceRelationQ[r_?squareMatrixQ] :=
	ReflexiveQ[r] && SymmetricQ[r] && TransitiveQ[r]
EquivalenceRelationQ[g_Combinatorica`Graph] := EquivalenceRelationQ[ToAdjacencyMatrix[g]]
Equivalences[g_Combinatorica`Graph, h_Combinatorica`Graph, f___] := 
        Module[{dg = Degrees[g], dh = Degrees[h], eq}, 
               eq = Table[Flatten[Position[dh, dg[[i]]], 1], {i, Length[dg]}];
               EQ[g, h, eq, f]
        ]
EQ[g_Combinatorica`Graph, h_Combinatorica`Graph, eq_List] := eq
EQ[g_Combinatorica`Graph, h_Combinatorica`Graph, eq_List, f1_, f___] := 
               If[Position[eq, {}] == {},
                  EQ[g, h, RefineEquivalences[eq, g, h, f1], f],
                  eq
               ]
Equivalences[g_Combinatorica`Graph, f___] := Equivalences[g, g, f]
Eulerian[n_Integer,k_Integer] := Block[{$RecursionLimit = Infinity}, Eulerian1[n, k]]
Eulerian1[0,k_Integer] := If [k==0, 1, 0]
Eulerian1[n_Integer, k_Integer] := 0 /; (k >= n)
Eulerian1[n_Integer, 0] := 1
Eulerian1[n_Integer,k_Integer] := Eulerian1[n,k] = (k+1) Eulerian1[n-1,k] + (n-k) Eulerian1[n-1,k-1] 
EulerianCycle::obsolete = "Usage of Directed as a second argument to EulerianCycle is obsolete."
EulerianCycle[g_Combinatorica`Graph, Directed] := (Message[EulerianCycle::obsolete]; EulerianCycle[g])
EulerianCycle[g_Combinatorica`Graph]/;EmptyQ[g] := {}
EulerianCycle[g_Combinatorica`Graph] :=
	Module[{euler,c,cycles,v},
		cycles = Map[(Drop[#,-1])&, ExtractCycles[g]];
		{euler, cycles} = {First[cycles], Rest[cycles]};
		Do [
			c = First[ Select[cycles, (Intersection[euler,#]=!={})&] ];
			v = First[Intersection[euler,c]];
			euler = Join[
				RotateLeft[c, Position[c,v] [[1,1]] ],
				RotateLeft[euler, Position[euler,v] [[1,1]] ]
			];
			(* cycles = Complement[cycles,{c}] *)
                        cycles = Sort[DeleteCases[cycles, c, {1}, 1]],
			{Length[cycles]}
		];
		Append[euler, First[euler]]
	] /; EulerianQ[g]
EulerianCycle[g_Combinatorica`Graph] := {} (* fall through *)
EulerianQ::obsolete = "Usage of Directed as a second argument to EulerianQ is obsolete."
EulerianQ[g_Combinatorica`Graph, Directed] := (Message[EulerianQ::obsolete]; EulerianQ[g])
EulerianQ[g_Combinatorica`Graph] := ConnectedQ[g] && (InDegree[g] === OutDegree[g]) /; !UndirectedQ[g]
EulerianQ[g_Combinatorica`Graph] := ConnectedQ[g] && Apply[And,Map[EvenQ, Degrees[g]]] /; UndirectedQ[g]
ExactRandomGraph[n_Integer,e_Integer] :=
	Combinatorica`Graph[
		Map[{NthPair[#]}&, Take[ Combinatorica`RandomPermutation[n (n-1)/2], e] ],
		CircularEmbedding[n]
	]
ExpandEdgeOptions[opts_List, i_Integer?Positive, fvp_List, svp_List, aopts_List] :=
        Module[{ec, es, ed, elc, el, elp, lp, nel},
               {ec, es, ed, elc, el, elp, lp} =
               {Combinatorica`EdgeColor, Combinatorica`EdgeStyle, EdgeDirection,
                EdgeLabelColor, Combinatorica`EdgeLabel, EdgeLabelPosition, LoopPosition} /. opts;
               {ec,
                ExpandEdgeOptions[Combinatorica`EdgeStyle, es],
                If[SameQ[fvp, svp],
                   ExpandEdgeOptions[Loop, ed, fvp, lp],
                   ExpandEdgeOptions[EdgeDirection, ed, fvp, svp, aopts]
                ],
                If[(el =!= False) && (el =!= Off) && (el =!= None),
                   {elc, 
                    nel = If[el === True || el === Automatic, i, el];
                    ExpandEdgeOptions[Combinatorica`EdgeLabel, nel, elp, fvp, svp]
                   },
                   {Black}
                ]
               }
        ]
ExpandEdgeOptions[EdgeDirection, ed_, fvp_List, svp_List, aopts_List] :=
        Module[{mp},
               Switch[ed, {True|On, _Integer},
                          mp = PathMidpoint[fvp, svp, ed[[2]]*Distance[{fvp,svp}//N]/30];
                          Apply[Sequence, {Line[{fvp, mp}], Arrow[{mp, svp}, Apply[Sequence, aopts]]}],
                          {False|Off, _Integer},
                          mp = PathMidpoint[fvp, svp, ed[[2]]*Distance[{fvp,svp}//N]/30];
                          Line[{fvp, mp, svp}],
                          True|On, Arrow[{fvp, svp}, Apply[Sequence, aopts]],
                          False|Off, Line[{fvp, svp}]
               ]
        ]
ExpandEdgeOptions[Loop, ed_, fvp_, lp_] :=
        Module[{offset, radius = 0.02, direction, center},
               If[Length[ed] === 0, offset = 0, offset = ed[[2]]*0.005];
               Switch[lp, UpperRight, direction = {1, 1},
                          UpperLeft, direction = {-1, 1},
                          LowerLeft, direction = {-1, -1},
                          LowerRight, direction = {1, -1},
			  _,  direction = {-1, 1}
               ];
               center = fvp + (radius + offset)*direction;
               Circle[center, Sqrt[Apply[Plus,(center-fvp)^2]]]
        ]
ExpandEdgeOptions[Combinatorica`EdgeStyle, es_] :=
        Apply[Sequence,
              Switch[es, Thick,        {Thickness[0.02]},
                         Normal,       {Thickness[0.005]},
                         Thin,         {Thickness[0.0005]},
                         ThickDashed,  {Thickness[0.02], Dashing[{0.05, 0.03}]},
                         NormalDashed, {Thickness[0.005], Dashing[{0.05,0.03}]},
                         ThinDashed,   {Thickness[0.005], Dashing[{0.05, 0.03}]},
                         _,            If[!ListQ[es], {Thickness[0.005], es},
                                                      Join[{Thickness[0.005]}, es]]
              ]
        ]
ExpandEdgeOptions[Combinatorica`EdgeLabel, el_, elp_, fvp_List, svp_List] :=
        Switch[elp, Center, Text[el, (fvp+svp)/2],
                    LowerLeft, Text[el, Scaled[{-.02,-.02},(fvp+svp)/2],{1,0}],
                    UpperRight, Text[el, Scaled[{.02,.02}, (fvp+svp)/2],{-1,0}],
                    LowerRight,Text[el, Scaled[{.02,-.02}, (fvp+svp)/2],{-1,0}],
                    UpperLeft, Text[el, Scaled[{-.02,.02}, (fvp+svp)/2],{1,0}],
                    {_, _}, Text[el, Scaled[{elp[[1]], elp[[2]]}, (fvp+svp)/2],{1,0}],
		    _, Text[el, Scaled[{-.02,-.02},(fvp+svp)/2],{1,0}]
        ]
ExpandGraph[g_Combinatorica`Graph, n_] := Combinatorica`GraphUnion[g, EmptyGraph[n - V[g]] ] /; V[g] <= n
ExpandPlotOptions[PlotRange -> (Normal | Full), v_List] := PlotRange -> FindPlotRange[v]
ExpandPlotOptions[PlotRange -> x_Real, v_List] :=
        Module[{r = FindPlotRange[v], xd, yd},
               xd = (r[[1,2]]-r[[1,1]])Abs[x]/2;
               yd = (r[[2,2]]-r[[2,1]])Abs[x]/2;
               PlotRange -> {{r[[1,1]]-Sign[x] xd, r[[1,2]]+Sign[x] xd},
                             {r[[2,1]]-Sign[x] yd, r[[2,2]]+Sign[x] yd}
                            }
        ]
ExpandPlotOptions[PlotRange -> Zoom[x_], v_List] := PlotRange -> FindPlotRange[ v[[x]] ]
ExpandPlotOptions[PlotRange -> r_, v_List] := PlotRange -> r
ExpandPlotOptions[x_, _] := x
ExpandVertexOptions[vertexOptions_List, p_List, i_Integer?Positive] :=
        Module[{vc, vs, vn, vnc, vnp, nvl, vl, vlc, vlp},
               {vc, vs, vn, vnc, vnp, vl, vlc, vlp} =
                {VertexColor,
                 Combinatorica`VertexStyle,
                 VertexNumber,
                 VertexNumberColor,
                 VertexNumberPosition,
                 Combinatorica`VertexLabel,
                 VertexLabelColor,
                 VertexLabelPosition
                } /. vertexOptions;
               {vc,
                ExpandVertexOptions[Combinatorica`VertexStyle, vs, p],
                If[(vn === True) || (vn === On),
                   {vnc, ExpandVertexOptions[VertexNumber, vnp, p, i]},
                   {Black}
                ],
                If[(vl =!= False) && (vl =!= Off) && (vl =!= None),
                   {vlc, 
                    nvl = If[vl === True || vl === Automatic, i, vl];
                    ExpandVertexOptions[ Combinatorica`VertexLabel, nvl, vlp, p]
                   },
                   {Black}
                ]
               }
        ]
ExpandVertexOptions[Combinatorica`VertexStyle, vs_, p_List] :=
        Module[{x, y, d},
	       If[MatchQ[vs, (Disk | Box)[_]],
	          x = vs[[1]]; y = Head[vs];
                  Switch[x, Small, d = 0.012,
                         Normal, d = 0.025,
                         Large, d = 0.07,
                         _, d = x
                  ];
                  Switch[y, Disk, {PointSize[d], Point[p]},
                         Box, Rectangle[p-d/2, p+d/2]
                  ],
		  {vs, Point[p]}
	      ]
        ]
ExpandVertexOptions[VertexNumber, vnp_, p_List, i_Integer] :=
        Switch[vnp, Center, Text[i, p],
                    LowerLeft, Text[i, Scaled[{-0.02,-0.02},p], {1, 0}],
                    UpperRight, Text[i, Scaled[{0.02,0.02},p], {-1, 0}],
                    LowerRight, Text[i, Scaled[{0.02,-0.02},p], {-1, 0}],
                    UpperLeft, Text[i, Scaled[{-0.02, 0.02},p], {1, 0}],
                    {_, _}, Text[i, Scaled[{vnp[[1]], vnp[[2]]}, p], {1, 0}],
		    _, Text[i, Scaled[{-0.02,-0.02},p], {1, 0}]
        ]
ExpandVertexOptions[Combinatorica`VertexLabel, vl_, vlp_, p_List] :=
        Switch[vlp, Center, Text[vl, p],
                    LowerLeft, Text[vl,Scaled[{-.02,-.02},p],{1,0}],
                    UpperRight, Text[vl,Scaled[{.02,.02},p],{-1,0}],
                    LowerRight,Text[vl,Scaled[{.02,-.02},p],{-1,0}],
                    UpperLeft, Text[vl,Scaled[{-.02,.02},p],{1,0}],
                    {_, _}, Text[vl,Scaled[{vlp[[1]], vlp[[2]]}, p], {1,0}],
		    _, Text[vl,Scaled[{.02,.02},p],{-1,0}]
        ]
ExtractCycles[gi_Combinatorica`Graph] := 
	Module[{g=gi,cycles={},c},
		While[!SameQ[{}, c=Combinatorica`FindCycle[g]],
			PrependTo[cycles,c];
			g = DeleteCycle[g,c];
		];
		cycles
	]
FerrersDiagram[p1_List] :=
	Module[{i,j,n=Length[p1],p=Sort[p1]},
		Show[
			Graphics[
				Join[
					{PointSize[ Min[0.05,1/(2 Max[p])] ]},
					Table[Point[{i,j}], {j,n}, {i,p[[j]]}]
				],
				{AspectRatio -> 1, PlotRange -> All}
			]
		]
	]
(* note: by design, variables not localized in the following; they are localized
   by Block in the function that calls FindBackEdge. *)
FindBackEdge[v_Integer, tag_:True] := 
        (s[[v]] = cnt++; 
         Scan[(If[parent[[#]] == 0, 
                  parent[[#]] = v; FindBackEdge[#, tag], 
                  If[tag === True,
                     If[parent[[v]] != #, Throw[{v, #}]],
                     If[(s[[#]] < s[[v]]) && (f[[#]] == 0), Throw[{v, #}]]
                  ]
               ])&, 
               e[[v]]
         ]; 
         f[[v]] = cnt++;)
FindBiconnectedComponents[g_Combinatorica`Graph] := { {}, {} } /; EmptyQ[g]
FindBiconnectedComponents[g_Combinatorica`Graph] :=
	Block[{e=ToAdjacencyLists[g],n=V[g],par,c=0,cc, act={},back, dfs,ap={}, bcc={}},
		back=dfs=Table[0,{n}];
		par = Table[n+1,{n}]; 
                Map[(c=0; act={}; SearchBiConComp[First[#]]; ap = Drop[ap,-1]) &,
                    Select[cc = Combinatorica`ConnectedComponents[g], Length[#] > 1 &]];
                {Join[bcc, Select[cc, Length[#] == 1 &]], ap}
        ]
FindBridge[g_Combinatorica`Graph, cycle_List] := 
        Module[{rg = RemoveCycleEdges[g, cycle], b, bridge, j}, 
               b = Map[(IsolateSubgraph[rg, g, cycle, #]) &, 
                       Select[Combinatorica`ConnectedComponents[rg], (Intersection[#, cycle] == {})&]
                   ];
               b = Select[b, (! EmptyQ[#]) &];
               j = Join[ 
                      Map[Function[
                             bridge, 
                             Select[cycle, ToAdjacencyLists[bridge][[#]] !={}&]
                          ], b
                      ],
                      Complement[
                         Select[Edges[g], 
                                (Length[Intersection[#, cycle]] == 2)&
                         ], 
                         Join[#, Reverse /@ #]&[Partition[Append[cycle,First[cycle]], 2, 1]]
                      ]
                   ];
               {b, j}
        ]
Combinatorica`FindCycle::obsolete = "Usage of Directed as a second argument to FindCycle is obsolete."
Combinatorica`FindCycle[g_Combinatorica`Graph, Directed] := (Message[Combinatorica`FindCycle::obsolete]; Combinatorica`FindCycle[g])
Combinatorica`FindCycle[g_Combinatorica`Graph] := First[Select[Edges[g], #[[1]]==#[[2]]&]] /; SelfLoopsQ[g]
Combinatorica`FindCycle[g_Combinatorica`Graph] :=
     Module[{e = Cases[Split[Sort[Edges[g]]], {x_List, x_List, ___}][[1, 1]]}, 
             Append[e, e[[1]]]
     ] /; (UndirectedQ[g] && MultipleEdgesQ[g])
Combinatorica`FindCycle[g_Combinatorica`Graph] := Combinatorica`FindCycle[ToAdjacencyLists[g], UndirectedQ[g]]
Combinatorica`FindCycle[al_List, tag_:True] := 
       Block[{e = al, c, parent = Table[0, {Length[al]}], s=Table[0, {Length[al]}],
              f = Table[0, {Length[al]}], cnt = 1, start, edge, $RecursionLimit = Infinity},
              While[Count[parent, 0] > 0, 
                    start = Position[parent, 0][[1, 1]];
                    parent[[start]] = start;
                    edge = Catch[FindBackEdge[start, tag]];
                    If[edge =!= Null,
                       c = Reverse[NestWhileList[parent[[#]] &, parent[[edge[[1]]]], (# != edge[[2]])&]];
                       Return[Prepend[Append[c, edge[[1]]], edge[[1]]]]
                    ]
              ];
              {}
       ]
Combinatorica`Private`FindPath[l_List,v1_Integer,v2_Integer] :=
	Block[{x=l[[v2]],y,z=v2,lst={}},
		If[SameQ[x,0], Return[{}]];
		While[!SameQ[x, start],
			If[ SameQ[x[[2]],f],
				PrependTo[lst,{{ x[[1]], z }, f}],
				PrependTo[lst,{{ z, x[[1]] }, b}]
			];
			z = x[[1]]; x = l[[z]];
		];
		lst
	]
FindPlotRange[v_List] :=
        Block[{xmax, xmin, ymax, ymin, xr, yr, ave},
              xmin=Min[Map[First[First[#]]&, v]];
              xmax=Max[Map[First[First[#]]&, v]];
              ymin=Min[Map[Last [First[#]]&, v]]; 
              ymax=Max[Map[Last [First[#]]&, v]];
	      If[Chop[xmax - xmin] == 0.0, xmax += 0.05; xmin -= 0.05];
	      If[Chop[ymax - ymin] == 0.0, ymax += 0.05; ymin -= 0.05];
	      xr = xmax - xmin; yr = ymax - ymin;
	      Which[xr/yr > 3,
	               ave = (ymax + ymin)/2;
		       ymax = ave + (1/3) xr;
		       ymin = ave - (1/3) xr,
		   yr/xr > 3,
		       ave = (xmax + xmin)/2;
		       xmax = ave + (1/3) yr;
		       xmin = ave - (1/3) yr
              ];
              {{xmin - 0.05 Max[1,xmax-xmin], xmax + 0.05 Max[1,xmax-xmin]},
               {ymin - 0.05 Max[1,ymax-ymin], ymax + 0.05 Max[1,ymax-ymin]}
              }
        ]
FindSet[n_Integer,s_List] := 
        Block[{$RecursionLimit = Infinity}, 
              If [n == s[[n,1]], n, FindSet[s[[n,1]],s]]
        ]
FiniteGraphs := 
        {ChvatalGraph, CoxeterGraph, CubicalGraph, DodecahedralGraph, 
         FolkmanGraph, FranklinGraph, FruchtGraph, GroetzschGraph, 
         HeawoodGraph, HerschelGraph, LeviGraph, McGeeGraph, MeredithGraph, 
         NonLineGraphs, NoPerfectMatchingGraph, OctahedralGraph, Combinatorica`PetersenGraph,
         RobertsonGraph, SmallestCyclicGroupGraph, TetrahedralGraph, 
         ThomassenGraph, TutteGraph, Uniquely3ColorableGraph, 
         UnitransitiveGraph, WaltherGraph}
FirstExample[list_List, predicate_] := Scan[(If [predicate[#],Return[#]])&,list]
FirstLexicographicTableau[s_List] :=
	Combinatorica`TransposeTableau[ LastLexicographicTableau[ TransposePartition[s] ] ]
FolkmanGraph := 
 Combinatorica`Graph[{{{1, 6}}, {{1, 9}}, {{1, 11}}, {{1, 14}}, {{2, 8}}, {{2, 10}}, 
  {{2, 13}}, {{2, 15}}, {{3, 7}}, {{3, 9}}, {{3, 12}}, {{3, 14}}, {{4, 6}}, 
  {{4, 8}}, {{4, 11}}, {{4, 13}}, {{5, 7}}, {{5, 10}}, {{5, 12}}, {{5, 15}}, 
  {{6, 16}}, {{6, 20}}, {{7, 16}}, {{7, 17}}, {{8, 17}}, {{8, 18}}, 
  {{9, 18}}, {{9, 19}}, {{10, 19}}, {{10, 20}}, {{11, 16}}, {{11, 20}}, 
  {{12, 16}}, {{12, 17}}, {{13, 17}}, {{13, 18}}, {{14, 18}}, {{14, 19}}, 
  {{15, 19}}, {{15, 20}}}, {{{0.474, 0.922}}, {{0.472, 0.844}}, 
  {{0.472, 0.77}}, {{0.478, 0.69}}, {{0.472, 0.998}}, {{0.576, 0.596}}, 
  {{0.68, 0.6}}, {{0.786, 0.596}}, {{0.866, 0.6}}, {{0.946, 0.598}}, 
  {{0.39, 0.598}}, {{0.32, 0.596}}, {{0.214, 0.6}}, {{0.118, 0.598}}, 
  {{0.026, 0.586}}, {{0.484, 0.494}}, {{0.478, 0.388}}, {{0.482, 0.306}}, 
  {{0.478, 0.222}}, {{0.484, 0.15}}}]
If[$VersionNumber < 6.0,
Format[Combinatorica`Graph[e_, v_, o___]] := 
       SequenceForm["\[SkeletonIndicator]Graph:<", 
                    Length[e], ", ", Length[v], ", ", 
                    If[MemberQ[{o}, EdgeDirection -> On] ||
                       MemberQ[{o}, EdgeDirection -> True], 
                       "Directed", 
                       "Undirected"
                    ], 
                    ">\[SkeletonIndicator]"
       ]
,
Combinatorica`Graph /: MakeBoxes[g : Combinatorica`Graph[e_List, v_List, opts___?OptionQ], fmt_] :=
  BoxForm`MakeInterpretationBox[
    StyleBox[RowBox[{
        "\[SkeletonIndicator]", "Graph:<", Length[e], ",", Length[v], ",",
        If[TrueQ[EdgeDirection/.Flatten[{opts, EdgeDirection -> False}]],
	      "Directed", "Undirected"],
        ">\[SkeletonIndicator]"
     }], ShowAutoStyles -> False, AutoSpacing -> False],
     g
   ]
]
FranklinGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 7}}, {{2, 4}}, {{2, 8}}, {{3, 5}}, {{3, 11}}, 
  {{4, 6}}, {{4, 12}}, {{5, 6}}, {{5, 7}}, {{6, 8}}, {{7, 9}}, {{8, 10}}, 
  {{9, 10}}, {{9, 12}}, {{10, 11}}, {{11, 12}}}, 
 {{{0.394, 0.924}}, {{0.562, 0.924}}, {{0.394, 0.762}}, {{0.562, 0.76}}, 
  {{0.256, 0.564}}, {{0.726, 0.564}}, {{0.104, 0.48}}, {{0.872, 0.47}}, 
  {{0.192, 0.332}}, {{0.782, 0.332}}, {{0.65, 0.436}}, {{0.324, 0.436}}}]
Options[FromAdjacencyLists] = {Type -> Undirected}
FromAdjacencyLists[al_List, v_List, opts___?OptionQ] :=
        ChangeVertices[FromAdjacencyLists[al, opts], v]
FromAdjacencyLists[al:{{_Integer...}...}, opts___?OptionQ] := 
        Module[{type, g, nal, i},
               type = Type /. Flatten[{opts, Options[FromAdjacencyLists]}];
               nal = If[type === Undirected, 
                        Table[Select[al[[i]], (# >= i) &], {i, Length[al]}], 
                        al
                     ];
               g = Combinatorica`Graph[Flatten[Table[Map[{{i,#}}&,  nal[[i]] ],
                                          {i, Length[al]}
                                 ], 1
                         ],
                         CircularEmbedding[Length[al]]
                   ];
               If[type === Directed, SetGraphOptions[g, EdgeDirection->True], g]
        ]
FromAdjacencyLists[al:{{{_Integer,(_Integer|_Real)}...}...}, Combinatorica`EdgeWeight, 
                   opts___?OptionQ] :=
        Module[{type, g, nal, i},
               type = Type /. Flatten[{opts,Options[FromAdjacencyLists]}]; 
               nal = If[type === Undirected,
                        Table[Select[al[[i]],(#[[1]] >= i) &],{i, Length[al]}],
                        al
                     ];
               g = Combinatorica`Graph[Flatten[
                            Table[Map[{{i,#[[1]]}, Combinatorica`EdgeWeight->#[[2]]}&, 
                                      nal[[i]]
                                  ],
                                  {i, Length[al]}
                            ], 1
                         ],
                         CircularEmbedding[Length[al]]
                   ];
               If[type === Directed, SetGraphOptions[g, EdgeDirection->True], g]
        ]
Options[FromAdjacencyMatrix] = {Type -> Undirected};
FromAdjacencyMatrix[m:{{_Integer...}...}, v_List, opts___?OptionQ] := 
        ChangeVertices[FromAdjacencyMatrix[m, opts], v]
FromAdjacencyMatrix[m:{{_Integer...}...}, opts___?OptionQ] := 
        Module[{type, p},
               type = Type /. Flatten[{opts, Options[FromAdjacencyMatrix]}];
               If[type === Undirected, 
                  p=Union[Map[Sort, Position[m, _Integer?Positive]]]; AM[p, m],
                  p=Position[m, _Integer?Positive]; 
                  SetGraphOptions[AM[p, m], EdgeDirection -> True]
               ]
        ]
               
AM[p_List, m_List] := 
        Combinatorica`Graph[Flatten[
                  Map[Table[{#}, {i, m[[Apply[Sequence, #]]]}] &, p], 1
              ], 
              CircularEmbedding[Length[m]]
        ]
FromAdjacencyMatrix[m_List, v_List, Combinatorica`EdgeWeight, opts___?OptionQ] := 
        ChangeVertices[FromAdjacencyMatrix[m, Combinatorica`EdgeWeight, opts], v]
FromAdjacencyMatrix[m_List, Combinatorica`EdgeWeight, opts___?OptionQ] := 
        Module[{type, p}, type = Type /. Flatten[{opts, Options[FromAdjacencyMatrix]}];
               If[type === Undirected, 
                  p = Union[Map[Sort, Position[m, _?NumericQ, 2]]]; 
                  AM[p, m, Combinatorica`EdgeWeight], 
                  p = Position[m, _?NumericQ, 2];
                  SetGraphOptions[AM[p, m, Combinatorica`EdgeWeight], EdgeDirection -> True]
               ]
        ]
AM[p_List, m_List, Combinatorica`EdgeWeight] := 
        Combinatorica`Graph[Map[{#, Combinatorica`EdgeWeight -> m[[Apply[Sequence, #]]]} &, p], 
              CircularEmbedding[Length[m]]
        ]
Combinatorica`FromCycles[cyc_List] := Map[Last, Sort[Transpose[Map[Flatten, {Map[RotateRight, cyc], cyc}]]]]
FromInversionVector[vec_List] :=
	Module[{n=Length[vec]+1,i,p},
		p={n};
		Do [
			p = Insert[p, i, vec[[i]]+1],
			{i,n-1,1,-1}
		];
		p
	]
Options[FromOrderedPairs] = {Type -> Directed}
FromOrderedPairs[el_List, v_List, opts___?OptionQ] := ChangeVertices[FromOrderedPairs[el, opts], v]
FromOrderedPairs[el_List, opts___?OptionQ] := 
       Module[{type},
              type = Type /. Flatten[{opts, Options[FromOrderedPairs]}];
              If[type === Directed, FOPD[el], FOP[el]]
       ]
FOPD[{}] := Combinatorica`Graph[{}, {{0,0}}, EdgeDirection->True]
FOPD[el_List] := Combinatorica`Graph[Map[{#}&, el], CircularEmbedding[Max[el]], EdgeDirection -> True]
FOP[{}] := Combinatorica`Graph[{}, {{0,0}}]
FOP[el_List] := Combinatorica`Graph[Map[{#}&, Union[Map[Sort, el]]], CircularEmbedding[Max[el]]]
FromParent[parent_List,s_Integer] :=
	Module[{i=s,lst={s}},
		While[!MemberQ[lst,(i=parent[[i]])], PrependTo[lst,i] ];
		PrependTo[lst,i];
		Take[lst, Flatten[Position[lst,i]]]
	]
 
Options[FromUnorderedPairs] = {Type -> Undirected};
FromUnorderedPairs[el_List, v_List, opts___?OptionQ] := ChangeVertices[FromUnorderedPairs[el, opts], v]
FromUnorderedPairs[el_List, opts___?OptionQ] := 
       Module[{type},
              type = Type /. Flatten[{opts, Options[FromUnorderedPairs]}];
              If[type === Undirected, FUP[el], FUPD[el]]
       ]
FUP[{}] := Combinatorica`Graph[{}, {{{0,0}}}]
FUP[el_List] := Combinatorica`Graph[Map[{Sort[#]}&, el], CircularEmbedding[Max[el]] ]
FUPD[{}] := Combinatorica`Graph[{}, {{{0,0}}}, EdgeDirection->True]
FUPD[el_List] := Combinatorica`Graph[Map[{#}&, Double[el]], CircularEmbedding[Max[el]], EdgeDirection -> True]
FruchtGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 12}}, {{2, 4}}, {{2, 7}}, {{3, 6}}, 
  {{3, 11}}, {{4, 5}}, {{4, 7}}, {{5, 9}}, {{5, 12}}, {{6, 10}}, {{6, 11}}, 
  {{7, 8}}, {{8, 9}}, {{8, 10}}, {{9, 10}}, {{11, 12}}}, 
 {{{0.474, 0.916}}, {{0.264, 0.832}}, {{0.72, 0.8}}, {{0.388, 0.734}}, 
  {{0.46, 0.7}}, {{0.71, 0.672}}, {{0.246, 0.628}}, {{0.336, 0.46}}, 
  {{0.46, 0.588}}, {{0.656, 0.464}}, {{0.598, 0.686}}, {{0.51, 0.79}}}]
Options[FunctionalGraph] = {Type -> Directed};
FunctionalGraph[f_, n_Integer] := 
        FromOrderedPairs[
            Table[{i, Mod[Apply[f, {i}], n]}+1, {i, 0, n-1}]
        ]
FunctionalGraph[f_List, v_List, opts___?OptionQ] :=
        Module[{type, t, i, j},
               type = Type /. Flatten[{opts, Options[FunctionalGraph]}];
               t = Flatten[
                      Table[Table[{i, Position[v, Apply[f[[j]], {v[[i]]}]][[1, 1]]},
                                  {j, Length[f]}
                            ],
                            {i, Length[v]}
                      ], 1
                   ];
               If[type === Directed,
                  FromOrderedPairs[t, Type -> Directed],
                  FromOrderedPairs[t, Type -> Undirected]
               ]
        ]
FunctionalGraph[f_, v_List, opts___?OptionQ] := FunctionalGraph[{f}, v, opts]
GeneralizedPetersenGraph[n_Integer?Positive, k_Integer?Positive] := 
        Module[{c = CircularEmbedding[n], i}, 
               AddEdges[
                  ChangeVertices[
                     Combinatorica`GraphUnion[Combinatorica`CirculantGraph[n, k], Cycle[n]], 
                     Join[c, 2c]
                  ], 
                  Table[{i, n + i}, {i, n}]
               ]
        ] /; (n > 1)
GetEdgeLabels[g_Combinatorica`Graph, el:{{{_Integer, _Integer},___?OptionQ}...}] := 
         Map[Combinatorica`EdgeLabel/.  
             Flatten[{Rest[#], GraphOptions[g], Options[Combinatorica`Graph]}]&,  
             el
         ]
GetEdgeLabels[g_Combinatorica`Graph, el:{{_Integer, _Integer}...}] := 
         Module[{nel = If[UndirectedQ[g], Map[Sort, el], el]},
                GetEdgeLabels[g, Select[Edges[g, All], MemberQ[nel, #[[1]]]] ]
         ]
                    
GetEdgeLabels[g_Combinatorica`Graph] := GetEdgeLabels[g, Edges[g, All]]
GetEdgeWeights[g_Combinatorica`Graph, el:{{{_Integer, _Integer},___?OptionQ}...}] := 
         Map[Combinatorica`EdgeWeight /.  
             Flatten[{Rest[#], GraphOptions[g], Options[Combinatorica`Graph]}]&,  
             el
         ]
GetEdgeWeights[g_Combinatorica`Graph, el:{{_Integer, _Integer}...}] := 
         Module[{nel = If[UndirectedQ[g], Map[Sort, el], el]},
                GetEdgeWeights[g, Select[Edges[g, All], MemberQ[nel, #[[1]]]& ]]
         ]
                    
GetEdgeWeights[g_Combinatorica`Graph] := GetEdgeWeights[g, Edges[g, All]]
GetVertexLabels[g_Combinatorica`Graph, el:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}] := 
         Map[Combinatorica`VertexLabel/.  
             Flatten[{Rest[#], GraphOptions[g], Options[Combinatorica`Graph]}]&,  
             el
         ]
GetVertexLabels[g_Combinatorica`Graph, el:{_Integer...}] := 
         GetVertexLabels[g, Vertices[g, All][[ el ]] ]
                    
GetVertexLabels[g_Combinatorica`Graph] := GetVertexLabels[g, Vertices[g, All]]
GetVertexWeights[g_Combinatorica`Graph, el:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}] := 
         Map[Combinatorica`VertexWeight /.  
             Flatten[{Rest[#], GraphOptions[g], Options[Combinatorica`Graph]}]&,  
             el
         ]
GetVertexWeights[g_Combinatorica`Graph, el:{_Integer...}] := 
         GetVertexWeights[g, Vertices[g, All][[ el ]] ]
                    
GetVertexWeights[g_Combinatorica`Graph] := GetVertexWeights[g, Vertices[g, All]]
Girth[g_Combinatorica`Graph] := 1 /; SelfLoopsQ[g]
Girth[g_Combinatorica`Graph] := 2 /; MultipleEdgesQ[g]
Girth[g_Combinatorica`Graph] := 
	Module[{v,dist,queue,n=V[g],girth=Infinity,
                parent,e=ToAdjacencyLists[g],x},
		Do [
			dist = parent = Table[Infinity, {n}];
			dist[[v]] = parent[[v]] = 0;
			queue = {v};
			While [queue != {},
				{x,queue} = {First[queue],Rest[queue]};
				Scan[
					(If [ (dist[[#]]+dist[[x]]<girth) &&
				     	      (parent[[x]] != #),
						girth=dist[[#]]+dist[[x]] + 1,
				 	 If [dist[[#]]==Infinity,
						dist[[#]] = dist[[x]] + 1;
						parent[[#]] = x;
						If [2 dist[[#]] < girth-1,
							AppendTo[queue,#] ]
					]])&,
					e[[ x ]]
				];
			],
			{v,n}
		];
		girth
	] 
Combinatorica`GraphCenter[g_Combinatorica`Graph] := 
	Module[{eccentricity = Eccentricity[g]},
		Flatten[ Position[eccentricity, Min[eccentricity]] ]
	]
Combinatorica`GraphComplement[g_Combinatorica`Graph] :=
        ChangeVertices[
             DeleteEdges[If[UndirectedQ[g],
                            CompleteGraph[V[g]],
                            CompleteGraph[V[g], Type->Directed]
                         ],
                         Edges[g]
             ],
             Vertices[g, All]
        ]
Combinatorica`GraphDifference[g_Combinatorica`Graph,h_Combinatorica`Graph] :=
        Module[{e = Complement[Edges[g], Edges[h]]}, 
               ChangeEdges[g, Select[Edges[g, All], (MemberQ[e, First[#]]) &]]
	] /; (V[g] == V[h]) && 
             ((UndirectedQ[g] && UndirectedQ[h]) ||
              (!UndirectedQ[g] && !UndirectedQ[h]))
                                                          
Combinatorica`GraphIntersection[g_Combinatorica`Graph] := g
Combinatorica`GraphIntersection[g_Combinatorica`Graph,h_Combinatorica`Graph] :=
       Module[{e = Intersection[Edges[g], Edges[h]]}, 
              ChangeEdges[g, Select[Edges[g, All], (MemberQ[e, First[#]]) &]]
       ] /; (V[g] == V[h]) && (UndirectedQ[g] == UndirectedQ[h])
Combinatorica`GraphIntersection[g_Combinatorica`Graph,h_Combinatorica`Graph, l__Combinatorica`Graph] := Combinatorica`GraphIntersection[Combinatorica`GraphIntersection[g, h], l]
Combinatorica`GraphJoin[g_Combinatorica`Graph] := g
Combinatorica`GraphJoin[g_Combinatorica`Graph,h_Combinatorica`Graph] :=
        AddEdges[Combinatorica`GraphUnion[g, h], 
                 CartesianProduct[Range[V[g]],Range[V[h]]+V[g]] 
        ] /; (UndirectedQ[g] == UndirectedQ[h])
Combinatorica`GraphJoin[g_Combinatorica`Graph,h_Combinatorica`Graph, l__Combinatorica`Graph] := Combinatorica`GraphJoin[Combinatorica`GraphJoin[g, h], l]
GraphLabels[v_List,l_List] :=
	Module[{i},
		Table[ Text[ l[[i]],v[[i]]-{0.03,0.03},{0,1} ],{i,Length[v]}]
	]
GraphOptions[Combinatorica`Graph[e_, v_, opts___?OptionQ]] := {opts}
GraphOptions[Combinatorica`Graph[e_, v_, opts___?OptionQ], sv_?NumberQ] :=
        Combinatorica`Private`Merge[Rest[v[[sv]]], {opts}]
GraphOptions[Combinatorica`Graph[e_, v_, opts___?OptionQ], se_List] :=
        Module[{sse = Sort[se]},
               Combinatorica`Private`Merge[Rest[Select[e, First[#]==sse&][[1]]], {opts}]
        ]
GraphPolynomial[0, _] := 1
GraphPolynomial[n_Integer?Positive, x_] :=
        OrbitInventory[PairGroupIndex[SymmetricGroupIndex[n, x], x], x, {1, x}] 
GraphPolynomial[0, _, Directed] := 1
GraphPolynomial[n_Integer?Positive, x_, Directed] :=
        OrbitInventory[PairGroupIndex[SymmetricGroupIndex[n, x], x, Ordered], 
                       x, 
                       {1, x}
        ] 
Combinatorica`GraphPower[g_Combinatorica`Graph,1] := g
Combinatorica`GraphPower[g_Combinatorica`Graph, k_Integer] :=
        Module[{prod, power, p = ToAdjacencyMatrix[g]},
               power = prod = p;
               FromAdjacencyMatrix[
                      Do[prod = prod.p; power = power + prod, {k-1}]; 
                      power, 
                      Vertices[g, All],
                      Type -> If[UndirectedQ[g], Undirected, Directed]
               ]
        ]
Combinatorica`GraphProduct[g_Combinatorica`Graph] := g
Combinatorica`GraphProduct[g_Combinatorica`Graph, h_Combinatorica`Graph] := Combinatorica`Graph[{}, {}] /; (V[g] == 0) || (V[h] == 0)
Combinatorica`GraphProduct[g_Combinatorica`Graph, h_Combinatorica`Graph] :=
	Module[{k, i, eg=Edges[g,All],eh=Edges[h,All],leng=V[g],lenh=V[h]},
               Combinatorica`Graph[Flatten[
                         Join[Table[Map[Prepend[
                                           Rest[#], #[[1]] + (i-1)*leng
                                        ]&, eg
                                    ], {i,lenh}
                              ],
                              Map[(Table[Prepend[Rest[#], leng*(#[[1]]-1)+k],
                                         {k, leng}
                                   ])&,
				   eh
                              ]
                         ], 1
                     ],
		     Map[{#}&, ProductVertices[Vertices[g], Vertices[h]]],
                     Apply[Sequence, Options[g]]
               ]
	] /; (UndirectedQ[g] == UndirectedQ[h])
Combinatorica`GraphProduct[g_Combinatorica`Graph, h_Combinatorica`Graph, l__Combinatorica`Graph] := Combinatorica`GraphProduct[Combinatorica`GraphProduct[g, h],l]
Combinatorica`GraphSum[g_Combinatorica`Graph] := g
Combinatorica`GraphSum[g_Combinatorica`Graph, h_Combinatorica`Graph] :=
	ChangeEdges[g, Join[Edges[g,All], Edges[h,All]]
        ] /; (V[g]==V[h]) && (UndirectedQ[g] == UndirectedQ[h])
Combinatorica`GraphSum[g_Combinatorica`Graph, h_Combinatorica`Graph, l__Combinatorica`Graph] := Combinatorica`GraphSum[Combinatorica`GraphSum[g, h], l]
Combinatorica`GraphUnion[g_Combinatorica`Graph] := g 
Combinatorica`GraphUnion[g_Combinatorica`Graph, h_Combinatorica`Graph] := g /; (V[h] == 0)
Combinatorica`GraphUnion[g_Combinatorica`Graph, h_Combinatorica`Graph] := h /; (V[g] == 0)
Combinatorica`GraphUnion[g_Combinatorica`Graph, h_Combinatorica`Graph] := 
        If[UndirectedQ[h], 
           Combinatorica`GraphUnion[MakeUndirected[g], h], 
           Combinatorica`GraphUnion[SetGraphOptions[g, EdgeDirection -> True], h]
        ] /; EmptyQ[g] && (UndirectedQ[g] != UndirectedQ[h])
Combinatorica`GraphUnion[g_Combinatorica`Graph, h_Combinatorica`Graph] := 
        If[UndirectedQ[g], 
           Combinatorica`GraphUnion[g, MakeUndirected[h]], 
           Combinatorica`GraphUnion[g, SetGraphOptions[h, EdgeDirection -> True]]
        ] /; EmptyQ[h] && (UndirectedQ[g] != UndirectedQ[h])
Combinatorica`GraphUnion[lg__Combinatorica`Graph] := 
        PutTogether[Map[NormalizeVertices, {lg}]] /; (Count[Map[UndirectedQ, {lg}], True] == 0) ||
                                                     (Count[Map[UndirectedQ, {lg}], True] == Length[{lg}])
PutTogether[{g_Combinatorica`Graph,h_Combinatorica`Graph}] :=
	Module[{maxg=Max[ Map[First, Vertices[g]]], 
                minh=Min[ Map[First, Vertices[h]]],
                n = V[g], s},
                s = maxg - minh + 1;
                ChangeEdges[
                    ChangeVertices[g,
                                   Join[Vertices[g, All],
                                        Map[Prepend[Rest[#], {s, 0}+ First[#]]&, 
                                            Vertices[h, All]
                                        ]
                                   ]
                    ],
                    Join[Edges[g,All], 
                         Map[Prepend[Rest[#], First[#]+ n ]&,
                             Edges[h,All]
                         ]
                    ] 
                ]
	] 
PutTogether[{g_Combinatorica`Graph, h_Combinatorica`Graph, l__Combinatorica`Graph}] := PutTogether[{PutTogether[{g, h}], l}]
Combinatorica`GraphUnion[0,g_Combinatorica`Graph] := EmptyGraph[0];
Combinatorica`GraphUnion[1,g_Combinatorica`Graph] := g
Combinatorica`GraphUnion[k_Integer,g_Combinatorica`Graph] := Combinatorica`GraphUnion[Apply[Sequence, Table[g, {k}]] ]
GraphicQ[s_List] := False /; (Min[s] < 0) || (Max[s] >= Length[s])
GraphicQ[s_List] := (First[s] == 0) /; (Length[s] == 1)
GraphicQ[s_List] :=
	Module[{m,sorted = Reverse[Sort[s]]},
		m = First[sorted];
		GraphicQ[ Join[ Take[sorted,{2,m+1}]-1, Drop[sorted,m+1] ] ]
	]
GrayCode[l_List] := GrayCodeSubsets[l]
GrayCodeKSubsets[n_Integer?Positive, k_Integer] := GrayCodeKSubsets[Range[n], k]
GrayCodeKSubsets[l_List, 0] := {{}}
GrayCodeKSubsets[l_List, 1] := Partition[l, 1]
GrayCodeKSubsets[l_List, k_Integer?Positive] := {l} /; (k == Length[l])
GrayCodeKSubsets[l_List, k_Integer?Positive] := {} /; (k > Length[l])
GrayCodeKSubsets[l_List, k_Integer] := 
       Block[{$RecursionLimit = Infinity},
             Join[GrayCodeKSubsets[Drop[l, -1], k], 
                  Map[Append[#, Last[l]]&, 
                      Reverse[GrayCodeKSubsets[Drop[l,-1], k-1]]
                  ]
             ]
       ]
GrayCodeSubsets[n_Integer?Positive] := GrayCodeSubsets[Range[n]]
GrayCodeSubsets[ { } ] := { {} }
GrayCodeSubsets[l_List] := 
       Block[{s, $RecursionLimit = Infinity}, 
              s = GrayCodeSubsets[Take[l, 1-Length[l]]];
              Join[s,  Map[Prepend[#, First[l]] &, Reverse[s]]]
       ]
GrayGraph := 
    MakeUndirected[
        MakeGraph[Range[54],
            Mod[#2 - #1, 54] == 1 ||
                (Mod[#1, 6] == 1 && Mod[#2 - #1, 54] == 25) ||
                (Mod[#1, 6] == 2 && Mod[#2 - #1, 54] == 29) ||
                (Mod[#1, 6] == 5 && Mod[#2 - #1, 54] == 7) ||
                (Mod[#1, 6] == 3 && Mod[#2 - #1, 54] == 13) &
        ]
    ]
GreedyVertexCover[g_Combinatorica`Graph, l_List] := GreedyVertexCover[MakeSimple[g]] /; (!SimpleQ[g] || !UndirectedQ[g])
GreedyVertexCover[g_Combinatorica`Graph, l_List] := 
       Module[{ng = g, d, m, s = {}, v, al, i}, 
              While[M[ng] != 0,  
                    d = Degrees[ng][[l]]; 
                    m  = Max[d]; 
                    If[m == 0, Return[s]]; 
                    v = l[[Position[d, m][[1, 1]]]]; 
                    AppendTo[s, v]; 
                    al = ToAdjacencyLists[ng]; 
                    ng = DeleteEdges[
                             ng, 
                             Table[{v, al[[v, i]]}, {i, Length[al[[v]]]}]
                         ]
              ]; 
              s
       ]
GreedyVertexCover[g_Combinatorica`Graph] := GreedyVertexCover[g, Range[V[g]]]
Combinatorica`GridGraph[n_Integer?Positive, m_Integer?Positive] :=
	Combinatorica`GraphProduct[
		ChangeVertices[Path[n],Map[({{Max[n,m]*#,0}})&,Range[n]]],
		Path[m]
	]
Combinatorica`GridGraph[p_Integer?Positive, q_Integer?Positive, r_Integer?Positive] := 
        Combinatorica`GraphProduct[Combinatorica`GridGraph[p, q], 
                     DilateVertices[RotateVertices[Path[r], 1], 1/(10*r)]
        ]
GroetzschGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 8}}, {{1, 11}}, {{2, 4}}, {{2, 7}}, 
  {{2, 10}}, {{3, 5}}, {{3, 7}}, {{3, 9}}, {{4, 5}}, {{4, 9}}, {{4, 11}}, 
  {{5, 8}}, {{5, 10}}, {{6, 7}}, {{6, 8}}, {{6, 9}}, {{6, 10}}, {{6, 11}}}, 
 {{{0.468, 0.908}}, {{0.242, 0.736}}, {{0.722, 0.728}}, {{0.332, 0.462}}, 
  {{0.612, 0.462}}, {{0.466, 0.684}}, {{0.466, 0.816}}, {{0.604, 0.71}}, 
  {{0.552, 0.556}}, {{0.388, 0.558}}, {{0.336, 0.716}}}]
GrotztschGraph = GroetzschGraph;
GroupEdgePositions[e_List, n_Integer] := 
        Map[Map[Last, #]&,
            Split[Union[Transpose[{e, Range[Length[e]]}], Table[{ {i, 0}, 0}, {i, n}]],
                  (#1[[1,1]]=== #2[[1,1]])&
            ]
        ]
HamiltonianCycle[g_Combinatorica`Graph,flag_:One] :=
	Module[{s={1},all={},done,adj=Edges[g],
                e=ToAdjacencyLists[g],x,v,ind,n=V[g]},
        (* by definition, a graph must have three or more vertices to have
  	      a Hamiltonian cycle *)
        If[n < 3, Return[{}]];
		ind=Table[1,{n}];
		While[ Length[s] > 0,
			v = Last[s];
			done = False;
			While[ ind[[v]] <= Length[e[[v]]] && !done,
                               x = e[[v,ind[[v]]++]];
                               done = !MemberQ[s, x] && 
                                       (Length[s] == 1 ||
                                        BiconnectedQ[DeleteVertices[AddEdges[g, {{1, x}}], Rest[s]]])
			];
			If[done, AppendTo[s,x], s=Drop[s,-1]; ind[[v]] = 1];
			If[(Length[s] == n),
				If [MemberQ[adj, Sort[{x, 1}]],
                                    AppendTo[all,Append[s,First[s]]];
                                    If [SameQ[flag,All],
                                        s=Drop[s,-1],
					all = Flatten[all]; s={}
			            ],
			            s = Drop[s,-1]
				]
			]
		];
		all
	]
HamiltonianPath[g_Combinatorica`Graph] := HamiltonianPath[g, One]
HamiltonianPath[g_Combinatorica`Graph, One]/;V[g] === 2 && ConnectedQ[g] := {{1,2}}
HamiltonianPath[g_Combinatorica`Graph, One] := 
        Module[{c = HamiltonianCycle[g], nonEdges, p, q, i, j, h},
               If[c != {}, 
                  Drop[c, -1],
                  nonEdges = Complement[Flatten[Table[{i, j}, {i, V[g]-1}, {j, i+1, V[g]}],1], Edges[g]];
                  Do[h = AddEdges[g, nonEdges[[i]]]; 
                     If[((BiconnectedQ[h]) && ((c = HamiltonianCycle[h]) != {})),
                        p = Position[c = Drop[c,-1], nonEdges[[i, 1]]][[1, 1]];
                        c = RotateLeft[c, p-1];
                        If[nonEdges[[i, 2]] == c[[2]], c = RotateLeft[c, 1]];
                        Break[]
                     ],
                     {i, Length[nonEdges]}
                  ];
                  c
               ]
        ]
HamiltonianPath[g_Combinatorica`Graph, All]/;V[g] === 2 && ConnectedQ[g] := {{1,2}, {2,1}}
HamiltonianPath[g_Combinatorica`Graph, All] := 
        Module[{c = HamiltonianCycle[g, All], nonEdges, edgesA, edgesB, p, q, h, i, j, k, a, b, 
                al = ToAdjacencyLists[g], s}, 
            Union[
               Flatten[
                   Map[Table[RotateLeft[Drop[#, -1], i - 1], {i, Length[#] - 1}] &, c], 1],
               Flatten[
                   nonEdges = Complement[Flatten[Table[{i, j}, {i, V[g] - 1}, {j, i + 1, V[g]}], 1], Edges[g]];
                   Table[{a, b} = nonEdges[[i]]; 
                         edgesA = Map[{a, #} &, al[[a]]];
                         edgesB = Map[{b, #} &, al[[b]]];
                         h = AddEdges[DeleteEdges[g, Join[edgesA, edgesB]], {a, b}];
                         Table[h = AddEdges[h, {edgesA[[j]], edgesB[[k]]}];
                               If[BiconnectedQ[h],
                                  c = Map[Drop[#, -1] &, HamiltonianCycle[h, All]];
                                  Map[({p, q} = {Position[#, a][[1, 1]], Position[#, b][[1, 1]]};
                                       s = RotateLeft[#, q - 1];
                                       If[s[[2]] == a, RotateLeft[s, 1], s])&, 
                                       c
                                  ], 
                                  {}
                               ], 
                               {j, Length[edgesA]}, {k, Length[edgesB]}
                         ],
                         {i, Length[nonEdges]}
                   ], 3
               ]
            ]
        ]
HamiltonianQ[g_Combinatorica`Graph] := False /; !BiconnectedQ[g]
HamiltonianQ[g_Combinatorica`Graph] := HamiltonianCycle[g] != {}
Harary[k_?EvenQ, n_Integer] := Combinatorica`CirculantGraph[n,Range[k/2]] /; (k > 1) && (n > k)
Harary[k_?OddQ, n_?EvenQ] := Combinatorica`CirculantGraph[n,Append[Range[k/2],n/2]] /; (k > 1)&& (n > k)
Harary[k_?OddQ, n_?OddQ] :=
        AddEdges[Harary[k-1, n],
                    Join[{{{1,(n+1)/2}}, {{1,(n+3)/2}}}, Table[{{i,i+(n+1)/2}}, {i,2,(n-1)/2}]]
        ] /; (k > 1) && (n > k)
HasseDiagram[g_Combinatorica`Graph, fak_:1] :=
	Module[{r, rank, m, stages, freq=Table[0,{V[g]}], 
                adjm, first, i},
	       r = TransitiveReduction[RemoveSelfLoops[g]];
               adjm = ToAdjacencyLists[r];
               rank = Table[ 0,{ V[g]} ];                              
               first = Select[ Range[ V[g]], InDegree[r,#]==0& ];
               rank = MakeLevel[ first, 1, adjm, rank];          
               first = Max[rank];
               stages = Distribution[ rank ];                       
               MakeUndirected[
                   ChangeVertices[r,
		          Table[
                            m = ++ freq[[ rank[[i]] ]];
                            {((m-1) + (1-stages[[rank[[i]] ]])/2)*
                            fak^(first-rank[[i]]), rank[[i]]}//N,
                            {i, V[g]}
                          ]
                   ]
               ]
        ] /; AcyclicQ[RemoveSelfLoops[g]] && !UndirectedQ[g]
HeapSort[p_List] :=
	Module[{heap=Heapify[p],min,n},
		Append[
			Table[
				min = First[heap];
				heap[[1]] = heap[[n]];
				heap = Heapify[Drop[heap,-1],1];
				min,
				{n,Length[p],2,-1}
			],
			Max[heap]
		]
	] /; (Length[p] > 0)
 
HeapSort[{}] := {}
Heapify[p_List] :=
	Module[{j,heap=p},
		Do [
			heap = Heapify[heap,j],
			{j,Quotient[Length[p],2],1,-1}
		];
		heap
	]
Heapify[p_List, k_Integer] :=
	Module[{hp=p, i=k, l, n=Length[p]},
		While[ (l = 2 i) <= n,
			If[ (l < n) && (hp[[l]] > hp[[l+1]]), l++ ];
			If[ hp[[i]] > hp[[l]],
				{hp[[i]],hp[[l]]}={hp[[l]],hp[[i]]};
				i = l,
				i = n+1
			];
		];
		hp
	]
HeawoodGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 6}}, {{1, 14}}, {{2, 3}}, {{2, 11}}, {{3, 4}}, 
  {{3, 8}}, {{4, 5}}, {{4, 13}}, {{5, 6}}, {{5, 10}}, {{6, 7}}, {{7, 8}}, 
  {{7, 12}}, {{8, 9}}, {{9, 10}}, {{9, 14}}, {{10, 11}}, {{11, 12}}, 
  {{12, 13}}, {{13, 14}}}, {{{0.8262387743159949, 0.563320058063622}}, 
  {{0.6234898018587336, 0.7818314824680297}}, 
  {{0.0747300935864246, 0.9972037971811801}}, 
  {{-0.22252093395631412, 0.9749279121818236}}, 
  {{-0.7330518718298261, 0.6801727377709197}}, 
  {{-0.9009688679024189, 0.4338837391175586}}, 
  {{-0.9888308262251286, -0.14904226617617403}}, 
  {{-0.9009688679024194, -0.4338837391175576}}, 
  {{-0.5000000000000004, -0.8660254037844384}}, 
  {{-0.22252093395631545, -0.9749279121818234}}, 
  {{0.36534102436639454, -0.9308737486442045}}, 
  {{0.6234898018587327, -0.7818314824680305}}, 
  {{0.9555728057861403, -0.2947551744109056}}, {{1., 0}}}]
HerschelGraph :=
 Combinatorica`Graph[{{{1, 3}}, {{1, 4}}, {{1, 5}}, {{1, 6}}, {{2, 3}}, {{2, 4}}, {{2, 7}}, 
  {{2, 8}}, {{3, 11}}, {{4, 10}}, {{5, 9}}, {{5, 10}}, {{6, 9}}, {{6, 11}}, 
  {{7, 9}}, {{7, 10}}, {{8, 9}}, {{8, 11}}}, 
  {{{0.002, 0.244}}, {{0., -0.26}}, {{0.41, 0.008}}, {{-0.396, 0.006}}, 
  {{-0.044, 0.07}}, {{0.042, 0.076}}, {{-0.038, -0.066}}, {{0.048, -0.07}}, 
  {{0., 0.}}, {{-0.204, 0.006}}, {{0.2, 0.008}}}]
HideCycles[c_List] := 
	Flatten[
		Sort[
			Map[(RotateLeft[#,Position[#,Min[#]] [[1,1]] - 1])&, c],
			(#1[[1]] > #2[[1]])&
		]
	]
Options[Highlight] = Sort[
    $GraphEdgeStyleOptions ~Join~
    $GraphVertexStyleOptions ~Join~
    {HighlightedVertexStyle -> Disk[Large], 
     HighlightedEdgeStyle -> Thick, 
     HighlightedVertexColors -> ScreenColors,
     HighlightedEdgeColors -> ScreenColors}
];
Highlight[g_Combinatorica`Graph, {}] := g
Highlight[g_Combinatorica`Graph, {}, _] := g
Highlight[g_Combinatorica`Graph, l : {___, _Integer, ___}, opts___?OptionQ] := 
        Highlight[g, {l}, opts]
Highlight[g_Combinatorica`Graph, l_List, opts___?OptionQ] := 
        Module[{vnc, enc, vs, es, vc, ec, i,
                vl = Map[Cases[#, _Integer] &, l], 
                el = Map[Cases[#, {_Integer, _Integer}] &, l], gopts}, 
               {vs, es, vc, ec} = 
               {HighlightedVertexStyle, 
                HighlightedEdgeStyle, 
                HighlightedVertexColors, 
                HighlightedEdgeColors} /. Flatten[{opts, Options[Highlight]}];
	       gopts = FilterRules[{opts,
	                            Complement[Options[Highlight],
				               Options[Combinatorica`Graph]]
                                   }, Options[Combinatorica`Graph]];
               el = If[UndirectedQ[g], Map[Sort, el, 2], el];
               vnc = Length[vl]; 
               enc = Length[el];
               If[!ListQ[vs], vs = {vs}];
               vs = PadRight[vs, vnc, vs];
               If[!ListQ[vc], vc = {vc}];
               vc = PadRight[vc, vnc, vc];
               If[!ListQ[es], es = {es}];
               es = PadRight[es, enc, es];
               If[!ListQ[ec], ec = {ec}];
               ec = PadRight[ec, enc, ec];
               SetGraphOptions[
                  g, 
                  Join[Table[{Apply[Sequence, vl[[i]]], 
                              Combinatorica`VertexStyle -> vs[[i]], 
                              VertexColor -> vc[[i]]}, 
                             {i, Length[vl]}
                       ], 
                       Table[{Apply[Sequence, el[[i]]], 
                              Combinatorica`EdgeStyle -> es[[i]], 
                              Combinatorica`EdgeColor -> ec[[i]]}, 
                             {i, Length[el]}
                       ]
                  ], gopts
               ]
        ]
Hypercube[n_Integer] := Hypercube1[n]
Hypercube1[0] := CompleteGraph[1]
Hypercube1[1] := Path[2]
Hypercube1[2] := Cycle[4]
Hypercube1[n_Integer] := Hypercube1[n] =
	Combinatorica`GraphProduct[
		RotateVertices[Hypercube1[Floor[n/2]], 2Pi/5],
		Hypercube1[Ceiling[n/2]]
	]
Hypercube1[0] := CompleteGraph[1]
Hypercube1[1] := Path[2]
Hypercube1[2] := Cycle[4]
Hypercube1[n_Integer] := Hypercube1[n] =
	Combinatorica`GraphProduct[
		RotateVertices[ Hypercube1[Floor[n/2]], 2Pi/5],
		Hypercube1[Ceiling[n/2]]
	]
IcosahedralGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 4}}, {{1, 5}}, {{1, 9}}, {{2, 3}}, {{2, 7}}, 
  {{2, 8}}, {{2, 9}}, {{3, 5}}, {{3, 6}}, {{3, 7}}, {{4, 5}}, {{4, 9}}, 
  {{4, 10}}, {{4, 12}}, {{5, 6}}, {{5, 12}}, {{6, 7}}, {{6, 11}}, {{6, 12}}, 
  {{7, 8}}, {{7, 11}}, {{8, 9}}, {{8, 10}}, {{8, 11}}, {{9, 10}}, {{10, 11}}, 
  {{10, 12}}, {{11, 12}}}, {{{0.452, 0.956}}, {{0.046, 0.242}}, 
  {{0.832, 0.246}}, {{0.444, 0.67}}, {{0.566, 0.56}}, {{0.544, 0.458}}, 
  {{0.45, 0.396}}, {{0.35, 0.452}}, {{0.32, 0.564}}, {{0.404, 0.558}}, 
  {{0.442, 0.49}}, {{0.48, 0.558}}}]
IdenticalQ[g_Combinatorica`Graph, h_Combinatorica`Graph] := True /; (EmptyQ[g])&&(EmptyQ[h])&&(V[g] == V[h])
IdenticalQ[g_Combinatorica`Graph, h_Combinatorica`Graph] := False /; (UndirectedQ[g] != UndirectedQ[h])
IdenticalQ[g_Combinatorica`Graph, h_Combinatorica`Graph] := (V[g]==V[h])&&
    If[UndirectedQ[g],
        Sort[Sort /@ Edges[g]] === Sort[Sort /@ Edges[h]],
        Sort[Edges[g]]===Sort[Edges[h]]
    ]
IdentityPermutation[n_Integer] := Range[n]
InDegree[g_Combinatorica`Graph, n_Integer] := 
        Length[Select[Edges[g], (Last[#]==n)&]]
InDegree[g_Combinatorica`Graph] := 
        OutDegree[ReverseEdges[g]]
Combinatorica`IncidenceMatrix[g_Combinatorica`Graph] := 
        Module[{e = Edges[g],i,j}, 
               If[UndirectedQ[g], 
                  Table[If[MemberQ[e[[j]], i], 1, 0], {i, V[g]}, {j, M[g]}], 
                  Table[If[i === First[e[[j]]], 1, 0], {i, V[g]}, {j, M[g]}]
               ]
        ]
IndependentSetQ[g_Combinatorica`Graph, indep_List] :=
        (Complement[indep, Range[V[g]]] == {}) && Combinatorica`VertexCoverQ[ g, Complement[ Range[V[g]], indep] ] 
Index[p_?Combinatorica`PermutationQ]:=
	Module[{i},
		Sum[ If [p[[i]] > p[[i+1]], i, 0], {i,Length[p]-1} ]
	]
InduceSubgraph[g_Combinatorica`Graph,{}] := Combinatorica`Graph[{},{}]
InduceSubgraph[g_Combinatorica`Graph, s_List] := PermuteSubgraph[g, Union[s]] /; Complement[s, Range[V[g]]] == {}
InitializeUnionFind[n_Integer] := Module[{i}, Table[{i,1},{i,n}] ]
InsertIntoTableau[e_Integer, t_?Combinatorica`TableauQ] := First[InsertIntoTableau[e, t, All]]
InsertIntoTableau[e_Integer,{}, All] := {{{e}}, 1}
InsertIntoTableau[e_Integer, t1_?Combinatorica`TableauQ, All] :=
	Module[{item=e,row=0,col,t=t1},
		While [row < Length[t],
			row++;
			If [Last[t[[row]]] <= item,
				AppendTo[t[[row]],item];
				Return[{t, row}]
			];
			col = Ceiling[ BinarySearch[t[[row]],item] ];
			{item, t[[row,col]]} = {t[[row,col]], item};
		];
		{Append[t, {item}], row+1}
	]
 
InterlockQ[juncs_, c_] :=
    Module[{pieces = Map[Sort, juncs/.MapThread[Rule, {c, Range[Length[c]]}]]},
            !BipartiteQ[MakeSimple[MakeGraph[pieces, LockQ, Type->Undirected]]]
    ]
IntervalGraph[l_List] :=
        MakeSimple[
	     MakeGraph[l,
		       (((First[#1] <= First[#2]) && (Last[#1] >= First[#2])) ||
		       ((First[#2] <= First[#1]) && (Last[#2] >= First[#1])) )&
	     ]
        ]
InversePermutation[p_?Combinatorica`PermutationQ] := Ordering[p]
Options[InversionPoset] = {Type -> Undirected, Combinatorica`VertexLabel->False}
InversionPoset[n_Integer?Positive, opts___?OptionQ] := 
        Module[{type, label, p = Permutations[n], br},
                {type, label} = {Type, Combinatorica`VertexLabel} /. Flatten[{opts, Options[InversionPoset]}];
                br = (Count[p = #1 - #2, 0] == n-2) &&
                (Position[p, _?Negative][[1,1]]+1 == Position[p, _?Positive][[1, 1]])&;
                If[type === Directed,
                   MakeGraph[p, br, Combinatorica`VertexLabel->label],
                   HasseDiagram[MakeGraph[p, br, Combinatorica`VertexLabel->label]]
                ]
        ]
Inversions[{}] := 0
Inversions[p_?Combinatorica`PermutationQ] := Apply[Plus,ToInversionVector[p]]
InvolutionQ[p_?Combinatorica`PermutationQ] := p[[p]] == Range[Length[p]]
Involutions[l_List, Combinatorica`Cycles] := {{ l }} /; (Length[l] === 1)
Involutions[l_List, Combinatorica`Cycles] := {{}} /; (Length[l] === 0)
Involutions[l_List, Combinatorica`Cycles] := 
       Block[{$RecursionLimit = Infinity,i},
             Join[Flatten[Table[Map[Append[#, {l[[1]], l[[i]]}] &, 
                                    Involutions[Drop[Rest[l], {i - 1}], Combinatorica`Cycles]
                                ], 
                                {i, 2, Length[l]}], 
                          1
                  ],
                  Map[Append[#, {l[[1]]}] & , Involutions[Rest[l], Combinatorica`Cycles]]
             ]
       ]
Involutions[l_List] := Map[Combinatorica`FromCycles, Involutions[l, Combinatorica`Cycles]]
Involutions[n_Integer] := Involutions[Range[n]]
Involutions[n_Integer, Combinatorica`Cycles] := Involutions[Range[n], Combinatorica`Cycles]
IsolateSubgraph[g_Combinatorica`Graph, orig_Combinatorica`Graph, cycle_List, cc_List] := 
        ChangeEdges[g,
                    Join[Select[Edges[g], 
                                (Length[Intersection[cc, #]] == 2)&
                         ],
                         Select[Edges[orig], 
                                (Intersection[#, cycle] != {} && 
                                 Intersection[#, cc] != {})&
                         ]
                    ]
        ]
Options[IsomorphicQ] = {Invariants -> {DegreesOf2Neighborhood, NumberOf2Paths, Distances}};
IsomorphicQ[g_Combinatorica`Graph, h_Combinatorica`Graph, opts___?OptionQ] := True /; IdenticalQ[g,h]
IsomorphicQ[g_Combinatorica`Graph, h_Combinatorica`Graph, opts___?OptionQ] := 
        Module[{invariants=Invariants /. Flatten[{opts, Options[IsomorphicQ]}]},
	       ! SameQ[ Isomorphism[g, h, Invariants -> invariants], {}]
        ]
Options[Isomorphism] = 
       {Invariants -> {DegreesOf2Neighborhood, NumberOf2Paths, Distances}};
Isomorphism[g_Combinatorica`Graph, opts___?OptionQ] := {} /; (V[g] == 0)
Isomorphism[g_Combinatorica`Graph, opts___?OptionQ] := {{1}} /; (V[g] == 1)
Isomorphism[g_Combinatorica`Graph, opts___?OptionQ] :=
        Module[{invariants=Invariants /. Flatten[{opts, Options[Isomorphism]}]},
               Isomorphism[g, g, Equivalences[g, Apply[Sequence, invariants]], All]
        ] 
Isomorphism[g_Combinatorica`Graph, h_Combinatorica`Graph, flag_Symbol:One, opts___?OptionQ] := {} /; (V[g] != V[h]) 
Isomorphism[g_Combinatorica`Graph, h_Combinatorica`Graph, flag_Symbol:One, opts___?OptionQ] := {{1}} /; ((V[g] == 1) && (V[h] == 1) && (M[g] == M[h]))
Isomorphism[g_Combinatorica`Graph, h_Combinatorica`Graph, flag_Symbol:One, opts___?OptionQ] :=
        Module[{invariants=Invariants /. Flatten[{opts, Options[Isomorphism]}]},
               Isomorphism[g,h,Equivalences[g,h, Apply[Sequence, invariants]], flag]
        ]
Isomorphism[g_Combinatorica`Graph, h_Combinatorica`Graph, equiv_List, flag_Symbol:One] :=
	If[!MemberQ[equiv,{}],
           Backtrack[equiv,
                     (IdenticalQ[
                          PermuteSubgraph[g,Range[Length[#]]],
		          PermuteSubgraph[h,#] 
                      ] &&
                      !MemberQ[Drop[#,-1],Last[#]]
                     )&,
                     (IsomorphismQ[g,h,#])&,
                     flag
            ],
            {}
        ]
IsomorphismQ[g_Combinatorica`Graph, h_Combinatorica`Graph, p_List] := False	/;
        (V[g]!= V[h]) || !Combinatorica`PermutationQ[p] || (Length[p] != V[g])
IsomorphismQ[g_Combinatorica`Graph,h_Combinatorica`Graph,p_List] := 
        IdenticalQ[g, PermuteSubgraph[h,p]]
JoinCycle[g1_Combinatorica`Graph, cycle_List] :=
	Module[{g=g1},
		Scan[(g = Combinatorica`AddEdge[g,#])&, Partition[cycle,2,1] ];
		Combinatorica`AddEdge[g,{First[cycle],Last[cycle]}]
	]
Josephus[n_Integer,m_Integer] :=
	Module[{live=Range[n],next},
		InversePermutation[
			Table[
				next = RotateLeft[live,m-1];
				live = Rest[next];
				First[next],
				{n}
			]
		]
	]
KSetPartitions[{}, 0] := {{}}
KSetPartitions[s_List, 0] := {}
KSetPartitions[s_List, k_Integer] := {} /; (k > Length[s])
KSetPartitions[s_List, k_Integer] := {Map[{#} &, s]} /; (k === Length[s])
KSetPartitions[s_List, k_Integer] :=
       Block[{$RecursionLimit = Infinity,j},
             Join[Map[Prepend[#, {First[s]}] &, KSetPartitions[Rest[s], k - 1]],
                  Flatten[
                     Map[Table[Prepend[Delete[#, j], Prepend[#[[j]], s[[1]]]],
                              {j, Length[#]}
                         ]&, 
                         KSetPartitions[Rest[s], k]
                     ], 1
                  ]
             ]
       ] /; (k > 0) && (k < Length[s])
KSetPartitions[0, 0] := {{}}
KSetPartitions[0, k_Integer?Positive] := {}
KSetPartitions[n_Integer?Positive, 0] := {}
KSetPartitions[n_Integer?Positive, k_Integer?Positive] := KSetPartitions[Range[n], k]
KSubsetGroup[g_List, s:{{_Integer..}...}, type_:Unordered] :=
        Table[Flatten[
              Map[Position[s, If[SameQ[type, Ordered], #,Sort[#]]]&,
                  Map[g[[i]][[#]]&, s, {2}]]
              ],
              {i, Length[g]}
        ]
KSubsetGroupIndex[g_, s_, x_Symbol, type_:Unordered] :=
        CycleIndex[KSubsetGroup[g, s, type],x]
KS = Compile[{{n, _Integer}, {k, _Integer}}, 
             Module[{h, ss = Range[k], x},  
                    Table[(h = Length[ss]; x = n;
                           While[x === ss[[h]], h--; x--];
                           ss = Join[Take[ss, h - 1], 
                                     Range[ss[[h]]+1, ss[[h]]+Length[ss]-h+1] 
                                ]), 
                          {Binomial[n, k]-1}
                    ] 
             ]
     ]
KSubsets[l_List,0] := { {} }
KSubsets[l_List,1] := Partition[l,1]
KSubsets[l_List,2] := Flatten[Table[{l[[i]], l[[j]]}, 
                                    {i, Length[l]-1}, 
                                    {j, i+1, Length[l]}
                              ], 
                              1
                      ]
KSubsets[l_List,k_Integer?Positive] := {l} /; (k == Length[l])
KSubsets[l_List,k_Integer?Positive] := {}  /; (k > Length[l])
KSubsets[s_List, k_Integer] := Prepend[Map[s[[#]] &, KS[Length[s], k]], s[[Range[k] ]] ]
KnightsTourGraph[m_Integer?Positive, n_Integer?Positive] := 
        Module[{p = Flatten[Table[{i, j}, {i, m}, {j, n}], 1]}, 
               Combinatorica`Graph[Union[
                         Map[{Sort[{n (#[[1, 1]] - 1) + #[[1, 2]], 
                                n (#[[2, 1]] - 1) + #[[2, 2]]}
                              ]
                             }&, 
                             Select[
                                 Flatten[
                                     Map[{{#, #+{2,1}}, {#, #+{2,-1}}, 
                                          {#, #+{-2,1}},{#, #+{-2,-1}}, 
                                          {#, #+{1,2}}, {#, #+{1,-2}}, 
                                          {#, #+{-1,2}},{#, #+{-1,-2}}}&, 
                                         p
                                     ], 1
                                 ],  
                                 (Min[#] >= 1) && (Max[#[[1,1]],#[[2, 1]]]<= m) && 
                                 (Max[#[[1,2]], #[[2,2]]]<=n)&
                             ]
                         ]
                     ],
                     Map[{#} &, p]
               ]
        ]
LabeledTreeToCode[g_Combinatorica`Graph] :=
	Module[{e=ToAdjacencyLists[g],i,code},
		Table [
			{i} = First[ Position[ Map[Length,e], 1 ] ];
			code = e[[i,1]];
			e[[code]] = Complement[ e[[code]], {i} ];
			e[[i]] = {};
			code,
			{V[g]-2}
		]
	]
LastLexicographicTableau[s_List] :=
	Module[{c=0},
		Map[(c+=#; Range[c-#+1,c])&, s]
	]
LeviGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{1, 8}}, {{1, 30}}, {{2, 3}}, {{2, 25}}, {{3, 4}}, 
  {{3, 12}}, {{4, 5}}, {{4, 17}}, {{5, 6}}, {{5, 22}}, {{6, 7}}, {{6, 27}}, 
  {{7, 8}}, {{7, 14}}, {{8, 9}}, {{9, 10}}, {{9, 18}}, {{10, 11}}, 
  {{10, 23}}, {{11, 12}}, {{11, 28}}, {{12, 13}}, {{13, 14}}, {{13, 20}}, 
  {{14, 15}}, {{15, 16}}, {{15, 24}}, {{16, 17}}, {{16, 29}}, {{17, 18}}, 
  {{18, 19}}, {{19, 20}}, {{19, 26}}, {{20, 21}}, {{21, 22}}, {{21, 30}}, 
  {{22, 23}}, {{23, 24}}, {{24, 25}}, {{25, 26}}, {{26, 27}}, {{27, 28}}, 
  {{28, 29}}, {{29, 30}}}, {{{0.9781476007338057, 0.20791169081775931}}, 
  {{0.9135454576426009, 0.40673664307580015}}, 
  {{0.8090169943749475, 0.5877852522924731}}, 
  {{0.6691306063588582, 0.7431448254773941}}, 
  {{0.5000000000000001, 0.8660254037844386}}, 
  {{0.30901699437494745, 0.9510565162951535}}, 
  {{0.10452846326765368, 0.9945218953682733}}, 
  {{-0.10452846326765333, 0.9945218953682734}}, 
  {{-0.30901699437494734, 0.9510565162951536}}, 
  {{-0.4999999999999998, 0.8660254037844387}}, 
  {{-0.6691306063588579, 0.7431448254773945}}, 
  {{-0.8090169943749473, 0.5877852522924732}}, 
  {{-0.9135454576426008, 0.40673664307580043}}, 
  {{-0.9781476007338056, 0.20791169081775973}}, {{-1., 0}}, 
  {{-0.9781476007338057, -0.20791169081775907}}, 
  {{-0.9135454576426011, -0.4067366430757998}}, 
  {{-0.8090169943749476, -0.587785252292473}}, 
  {{-0.6691306063588585, -0.743144825477394}}, 
  {{-0.5000000000000004, -0.8660254037844384}}, 
  {{-0.30901699437494756, -0.9510565162951535}}, 
  {{-0.10452846326765423, -0.9945218953682733}}, 
  {{0.10452846326765299, -0.9945218953682734}}, 
  {{0.30901699437494723, -0.9510565162951536}}, 
  {{0.49999999999999933, -0.866025403784439}}, 
  {{0.6691306063588578, -0.7431448254773946}}, 
  {{0.8090169943749473, -0.5877852522924734}}, 
  {{0.9135454576426005, -0.40673664307580093}}, 
  {{0.9781476007338056, -0.20791169081775987}}, {{1., 0}}}]
LexicographicPermutations[0] := {{}}
LexicographicPermutations[1] := {{1}}
LexicographicPermutations[n_Integer?Positive] := LP[n]
LexicographicPermutations[l_List] := Combinatorica`Permute[l, LexicographicPermutations[Length[l]] ]
LP = Compile[{{n, _Integer}},
             Module[{l = Range[n], i, j, t},
                    NestList[(i = n-1; While[ #[[i]] > #[[i+1]], i--];
                              j = n; While[ #[[j]] < #[[i]], j--];
                              t = #[[i]]; #[[i]] = #[[j]]; #[[j]] = t;
                              Join[ Take[#,i], Reverse[Drop[#,i]] ])&,
                              l, n!-1
                    ]
             ]
     ]
LexicographicSubsets[{}] := {{ }} 
LexicographicSubsets[l_List] := 
       Block[{$RecursionLimit = Infinity, s = LexicographicSubsets[Rest[l]]}, 
             Join[{{}}, Map[Prepend[#, l[[1]]] &, s], Rest[s]]
       ]
LexicographicSubsets[0] := {{ }} 
LexicographicSubsets[n_Integer] := LexicographicSubsets[Range[n]]
Combinatorica`LineGraph[g_Combinatorica`Graph] :=
        Module[{e = Sort[Edges[g]], ef, eb, c, i, j,
                v = Vertices[g]},
               ef = GroupEdgePositions[e, V[g]];
               eb = GroupEdgePositions[ Map[Reverse,e], V[g]];
               c  = Table[Rest[Union[ef[[i]], eb[[i]]]], {i, V[g]}];
               Combinatorica`Graph[Union[
                       Flatten[
                           Map[Table[{{#[[i]], #[[j]]}}, {i, Length[#]-1}, {j, i+1, Length[#]}]&, c],  
                           2
                       ]
                     ],
                   Map[({(v[[ #[[1]] ]] + v[[ #[[2]] ]]) / 2})&, e]
               ]
        ]
ListGraphs[n_Integer?Positive, m_Integer] := 
    Module[{allBitVectors, distinctBitVectors, graphs, s = KSubsets[Range[n], 2]}, 
           allBitVectors = Permutations[Join[Table[1, {m}], Table[0, {n (n - 1)/2 - m}]]];
           distinctBitVectors = OrbitRepresentatives[KSubsetGroup[
                                    Combinatorica`SymmetricGroup[n], s], allBitVectors
                                ];
           Map[FromUnorderedPairs[#, CircularEmbedding[n]]&,
               Map[s[[#]] &, Map[Flatten[Position[#, 1]] &, distinctBitVectors]]
           ]
    ]
ListGraphs[n_Integer?Positive] :=
        Flatten[Table[ListGraphs[n, m], {m, 0, n (n-1)/2}], 1]
ListGraphs[n_Integer?Positive, m_Integer, Directed] := 
    Module[{allBitVectors, distinctBitVectors, graphs, 
            s = Complement[Flatten[Table[{i, j}, {i, n}, {j, n}], 1], Table[{i, i}, {i, n}]
                ]
           }, 
            allBitVectors = Permutations[Join[Table[1, {m}], Table[0, {n (n - 1) - m}]]];
            distinctBitVectors = OrbitRepresentatives[
                                     KSubsetGroup[Combinatorica`SymmetricGroup[n],s,Ordered], 
                                     allBitVectors
                                 ];
            Map[FromOrderedPairs[#, CircularEmbedding[n]]&,
                Map[s[[#]] &, Map[Flatten[Position[#, 1]] &, distinctBitVectors]]
            ]
    ]
ListGraphs[n_Integer?Positive, Directed] :=
        Flatten[Table[ListGraphs[n, m, Directed], {m, 0, n (n-1)}], 1]
ListNecklaces[n_Integer?Positive, c_List, Cyclic]/;Length[c] === n :=
        OrbitRepresentatives[Combinatorica`CyclicGroup[n], Permutations[c]]
ListNecklaces[n_Integer?Positive, c_List, Dihedral]/;Length[c] === n :=
        OrbitRepresentatives[Combinatorica`DihedralGroup[n], Permutations[c]]
ListNecklaces[n_Integer?Positive, c_List, any_]/;Length[c] > 1 :=
    ListNecklaces[n, PadRight[c, n, c], any]
Lock1Q[a_List,b_List] :=
	Module[{bk, aj},
		bk = Min[ Select[Drop[b,-1], (#>First[a])&] ];
		aj = Min[ Select[a, (# > bk)&] ];
		(aj < Max[b])
	]
LockQ[a_List,b_List] := Lock1Q[a,b] || Lock1Q[b,a]
LockQ[a_List, b_List]/;(a === b && Length[a] > 2) := True
LongestIncreasingSubsequence[p_?Combinatorica`PermutationQ] :=
	Module[{c,x,xlast},
		c = TableauClasses[p];
		xlast = x = First[ Last[c] ];
		Append[
			Reverse[
				Map[
					(x = First[ Intersection[#,
					       Take[p, Position[p,x][[1,1]] ] ] ])&,
					Reverse[ Drop[c,-1] ]
				]
			],
			xlast
		]
	]
LongestIncreasingSubsequence[{}] := {}
LowerBoundTSP[g_Combinatorica`Graph] := Apply[Plus, Map[Min, ToAdjacencyLists[g, Combinatorica`EdgeWeight] /. {_Integer, x_Integer} -> x]]
M::obsolete = "Usage of Directed as a second argument to M is obsolete."
M[Combinatorica`Graph[e_List, _List, ___?OptionQ]] := Length[e]
M[g_Combinatorica`Graph, Directed] := (Message[M::obsolete]; M[g])
MakeDirected[g_Combinatorica`Graph?UndirectedQ] := 
        SetGraphOptions[ChangeEdges[g, Double[Edges[g]]], EdgeDirection->True]
MakeDirected[g_Combinatorica`Graph?UndirectedQ, All] := 
        SetGraphOptions[ChangeEdges[g, Double[Edges[g, All], All]], EdgeDirection->True]
Options[MakeGraph] = {Type -> Directed, Combinatorica`VertexLabel->False}
MakeGraph1[v_List, f_] :=
        Table[If [Apply[f,{v[[i]],v[[j]]}]===True, 1, 0],
                  {i,Length[v]},
                  {j,Length[v]}
        ]
MakeGraph[v_List, f_, opts___?OptionQ] :=
        Module[{type, label, g, l},
               {type, label} = {Type, Combinatorica`VertexLabel} /. Flatten[{opts, Options[MakeGraph]}];
               g = If[type === Directed,
                      FromAdjacencyMatrix[MakeGraph1[v, f], Type -> Directed],
                      FromAdjacencyMatrix[MakeGraph1[v, f], Type -> Undirected]
                   ];
               If[(label === On) || (label === True) || (label === Automatic),
                  l = v;
	       ];
	       If[ListQ[l],
                  SetVertexLabels[g, l],
                  g
               ]
        ]
MakeLevel[{},_,_,rank_] := rank
MakeLevel[l_List,lvl_,adjm_List,r_List] :=
  Module[ {rank=r, v, lst=l }, 
          rank = SetLevel[lst,lvl,rank];  (* make this level ready *)
          While[ lst != {},
                 v = First[lst];
                 rank = MakeLevel[adjm[[v]], lvl+1,adjm,rank]; 
                 lst = Rest[lst];
          ];                            
          rank
  ]
MakeSimple[g_Combinatorica`Graph] := MakeUndirected[RemoveMultipleEdges[RemoveSelfLoops[g]]]
MakeUndirected[g_Combinatorica`Graph] := 
        If[UndirectedQ[g],
           g,
           RemoveMultipleEdges[
              SetGraphOptions[ 
                   ChangeEdges[g, Edges[g,All] /. {x_Integer, y_Integer} :> Sort[{x, y}]], 
                   EdgeDirection -> False
              ]
           ]
        ]
MaximalMatching[g_Combinatorica`Graph] :=
	Module[{match={}},
		Scan[
			(If [Intersection[#,match]=={}, match=Join[match,#]])&,
			Edges[g]
		];
		Partition[match,2]
	]
 
MaximumAntichain[g_Combinatorica`Graph] := MaximumIndependentSet[MakeUndirected[TransitiveClosure[g]]]
MaximumClique[g_Combinatorica`Graph, k_Integer] := {} /; (V[g] == 0)
MaximumClique[g_Combinatorica`Graph, k_Integer] := {1} /; ((V[g] == 1) && (k == 1))
MaximumClique[g_Combinatorica`Graph, k_Integer] := {} /; ((V[g] == 1) && (k != 1))
MaximumClique[g_Combinatorica`Graph, k_Integer] := MaximumClique[MakeSimple[g], k] /; (!SimpleQ[g] || !UndirectedQ[g])
MaximumClique[g_Combinatorica`Graph,k_Integer] :=
	Module[{e = ToAdjacencyLists[g]},
               Flatten[
                  Position[
                     Backtrack[
                        Table[{True,False}, {V[g]}],
                        ((Last[#] == False)||
                         ((Count[#,True]<=k) &&
                          (Length[
                              Intersection[Flatten[Position[#,True]], e[[ Length[#]]]]
                           ] == (Count[#,True]-1)
                          )
                         )
                        )&,
                        ((Count[#,True]==k) && 
                         ((Last[#] == False) ||
                          (Length[Intersection[Flatten[Position[#,True]], e[[ Length[#]]]]
                           ] == (Count[#,True]-1)
                          )
                         )
                        )&,
                        First
                     ],
                     True
                  ]
               ]
	]
MaximumClique[g_Combinatorica`Graph] := {} /; (V[g] == 0)
MaximumClique[g_Combinatorica`Graph] := {1} /; (V[g] == 1)
MaximumClique[g_Combinatorica`Graph] := {1} /; EmptyQ[g]
MaximumClique[g_Combinatorica`Graph] :=  Range[V[g]] /; IdenticalQ[MakeSimple[g], CompleteGraph[V[g]]] 
MaximumClique[g_Combinatorica`Graph] := MaximumClique[MakeSimple[g]] /; (!SimpleQ[g] || !UndirectedQ[g])
MaximumClique[g_Combinatorica`Graph] := 
        Module[{oldClique = First[Edges[g]], c, d = Max[Degrees[g]], i}, 
               Do[c = MaximumClique[g, i]; If[c == {}, Break[], oldClique = c ], {i, 3, d+1}];
               oldClique 
        ]
MaximumColorDegreeVertices[e_List,color_List] :=
	Module[{n=Length[color],l,i,x},
		l = Table[ Count[e[[i]], _?(Function[x,color[[x]]!=0])], {i,n}];
		Do [ 
			If [color[[i]]!=0, l[[i]] = -1],
			{i,n}
		];
		Flatten[ Position[ l, Max[l] ] ]
	]
MaximumIndependentSet[g_Combinatorica`Graph] := Complement[Range[V[g]], MinimumVertexCover[g]]
MaximumSpanningTree[g_Combinatorica`Graph] :=
        MinimumSpanningTree[Map[{First[#], -Last[#]}&, Edges[g, Combinatorica`EdgeWeight]], g ]
McGeeGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 8}}, {{1, 24}}, {{2, 3}}, {{2, 19}}, {{3, 4}}, 
  {{3, 15}}, {{4, 5}}, {{4, 11}}, {{5, 6}}, {{5, 22}}, {{6, 7}}, {{6, 18}}, 
  {{7, 8}}, {{7, 14}}, {{8, 9}}, {{9, 10}}, {{9, 21}}, {{10, 11}}, 
  {{10, 17}}, {{11, 12}}, {{12, 13}}, {{12, 24}}, {{13, 14}}, {{13, 20}}, 
  {{14, 15}}, {{15, 16}}, {{16, 17}}, {{16, 23}}, {{17, 18}}, {{18, 19}}, 
  {{19, 20}}, {{20, 21}}, {{21, 22}}, {{22, 23}}, {{23, 24}}}, 
 {{{0.9659258262890683, 0.25881904510252074}}, 
  {{0.8660254037844387, 0.49999999999999994}}, 
  {{0.7071067811865476, 0.7071067811865475}}, 
  {{0.5000000000000001, 0.8660254037844386}}, 
  {{0.25881904510252096, 0.9659258262890682}}, {{0, 1.}}, 
  {{-0.25881904510252063, 0.9659258262890683}}, 
  {{-0.4999999999999998, 0.8660254037844387}}, 
  {{-0.7071067811865475, 0.7071067811865476}}, 
  {{-0.8660254037844385, 0.5000000000000003}}, 
  {{-0.9659258262890682, 0.258819045102521}}, {{-1., 0}}, 
  {{-0.9659258262890684, -0.25881904510252035}}, 
  {{-0.8660254037844388, -0.4999999999999997}}, 
  {{-0.7071067811865479, -0.7071067811865471}}, 
  {{-0.5000000000000004, -0.8660254037844384}}, 
  {{-0.2588190451025215, -0.9659258262890681}}, {{0, -1.}}, 
  {{0.2588190451025203, -0.9659258262890684}}, 
  {{0.49999999999999933, -0.866025403784439}}, 
  {{0.7071067811865474, -0.7071067811865477}}, 
  {{0.8660254037844384, -0.5000000000000004}}, 
  {{0.9659258262890681, -0.25881904510252157}}, {{1., 0}}}]
MeredithGraph := 
 Combinatorica`Graph[{{{1, 5}}, {{1, 6}}, {{1, 7}}, {{2, 5}}, {{2, 6}}, {{2, 7}}, {{3, 5}}, 
  {{3, 6}}, {{3, 7}}, {{4, 5}}, {{4, 6}}, {{4, 7}}, {{8, 12}}, {{8, 13}}, 
  {{8, 14}}, {{9, 12}}, {{9, 13}}, {{9, 14}}, {{10, 12}}, {{10, 13}}, 
  {{10, 14}}, {{11, 12}}, {{11, 13}}, {{11, 14}}, {{15, 19}}, {{15, 20}}, 
  {{15, 21}}, {{16, 19}}, {{16, 20}}, {{16, 21}}, {{17, 19}}, {{17, 20}}, 
  {{17, 21}}, {{18, 19}}, {{18, 20}}, {{18, 21}}, {{22, 26}}, {{22, 27}}, 
  {{22, 28}}, {{23, 26}}, {{23, 27}}, {{23, 28}}, {{24, 26}}, {{24, 27}}, 
  {{24, 28}}, {{25, 26}}, {{25, 27}}, {{25, 28}}, {{29, 33}}, {{29, 34}}, 
  {{29, 35}}, {{30, 33}}, {{30, 34}}, {{30, 35}}, {{31, 33}}, {{31, 34}}, 
  {{31, 35}}, {{32, 33}}, {{32, 34}}, {{32, 35}}, {{36, 40}}, {{36, 41}}, 
  {{36, 42}}, {{37, 40}}, {{37, 41}}, {{37, 42}}, {{38, 40}}, {{38, 41}}, 
  {{38, 42}}, {{39, 40}}, {{39, 41}}, {{39, 42}}, {{43, 47}}, {{43, 48}}, 
  {{43, 49}}, {{44, 47}}, {{44, 48}}, {{44, 49}}, {{45, 47}}, {{45, 48}}, 
  {{45, 49}}, {{46, 47}}, {{46, 48}}, {{46, 49}}, {{50, 54}}, {{50, 55}}, 
  {{50, 56}}, {{51, 54}}, {{51, 55}}, {{51, 56}}, {{52, 54}}, {{52, 55}}, 
  {{52, 56}}, {{53, 54}}, {{53, 55}}, {{53, 56}}, {{57, 61}}, {{57, 62}}, 
  {{57, 63}}, {{58, 61}}, {{58, 62}}, {{58, 63}}, {{59, 61}}, {{59, 62}}, 
  {{59, 63}}, {{60, 61}}, {{60, 62}}, {{60, 63}}, {{64, 68}}, {{64, 69}}, 
  {{64, 70}}, {{65, 68}}, {{65, 69}}, {{65, 70}}, {{66, 68}}, {{66, 69}}, 
  {{66, 70}}, {{67, 68}}, {{67, 69}}, {{67, 70}}, {{3, 51}}, {{2, 52}}, 
  {{10, 58}}, {{9, 59}}, {{17, 65}}, {{16, 66}}, {{24, 37}}, {{23, 38}}, 
  {{31, 44}}, {{30, 45}}, {{4, 22}}, {{8, 25}}, {{15, 32}}, {{1, 18}}, 
  {{11, 29}}, {{39, 43}}, {{36, 67}}, {{60, 64}}, {{53, 57}}, {{46, 50}}}, 
 {{{-0.25093605216412773, -5.626403722878969}}, 
  {{-1.2019925684592812, -5.3173867285040215}}, 
  {{-2.153049084754435, -5.008369734129074}}, 
  {{-3.1041056010495884, -4.699352739754127}}, 
  {{-0.4174473159367571, -4.520838709396341}}, 
  {{-1.3685038322319105, -4.211821715021394}}, 
  {{-2.319560348527064, -3.9028047206464462}}, 
  {{5.273484419331281, -1.9773087351681486}}, 
  {{4.685699167038807, -2.786325729543096}}, 
  {{4.097913914746334, -3.5953427239180433}}, 
  {{3.510128662453861, -4.404359718292991}}, 
  {{4.1705747988100965, -1.794031980063149}}, 
  {{3.5827895465176236, -2.6030489744380962}}, 
  {{2.9950042942251502, -3.4120659688130437}}, 
  {{3.510128662453863, 4.404359718292988}}, 
  {{4.097913914746336, 3.595342723918041}}, 
  {{4.685699167038809, 2.7863257295430937}}, 
  {{5.273484419331282, 1.9773087351681466}}, 
  {{2.995004294225152, 3.4120659688130415}}, 
  {{3.5827895465176254, 2.603048974438094}}, 
  {{4.170574798810099, 1.7940319800631468}}, 
  {{-3.104105601049586, 4.6993527397541275}}, 
  {{-2.1530490847544326, 5.008369734129075}}, 
  {{-1.201992568459279, 5.317386728504022}}, 
  {{-0.2509360521641253, 5.626403722878969}}, 
  {{-2.3195603485270624, 3.9028047206464476}}, 
  {{-1.3685038322319085, 4.211821715021395}}, 
  {{-0.4174473159367549, 4.520838709396342}}, 
  {{-5.428571428571429, -1.4999999999999982}}, 
  {{-5.428571428571429, -0.49999999999999806}}, 
  {{-5.428571428571429, 0.500000000000002}}, 
  {{-5.428571428571428, 1.5000000000000018}}, 
  {{-4.428571428571429, -0.9999999999999983}}, 
  {{-4.428571428571429, 1.6969545188681513*^-15}}, 
  {{-4.428571428571429, 1.0000000000000018}}, 
  {{-1.5311493145746229, 9.566495004673177}}, 
  {{-2.4822058308697765, 9.25747801029823}}, 
  {{-3.43326234716493, 8.948461015923282}}, 
  {{-4.384318863460084, 8.639444021548336}}, 
  {{-2.315694567097147, 10.363043023780858}}, 
  {{-3.266751083392301, 10.05402602940591}}, 
  {{-4.217807599687454, 9.745009035030963}}, 
  {{-9.571428571428571, 1.5000000000000013}}, 
  {{-9.571428571428571, 0.5000000000000012}}, 
  {{-9.571428571428571, -0.4999999999999988}}, 
  {{-9.571428571428571, -1.4999999999999987}}, 
  {{-10.571428571428571, 1.0000000000000013}}, 
  {{-10.571428571428571, 1.2945838597550844*^-15}}, 
  {{-10.571428571428571, -0.9999999999999987}}, 
  {{-4.384318863460085, -8.639444021548334}}, 
  {{-3.433262347164932, -8.94846101592328}}, 
  {{-2.4822058308697783, -9.257478010298229}}, 
  {{-1.5311493145746247, -9.566495004673175}}, 
  {{-4.217807599687456, -9.74500903503096}}, 
  {{-3.2667510833923026, -10.054026029405907}}, 
  {{-2.315694567097149, -10.363043023780856}}, 
  {{6.8617704962929285, -6.839470049218952}}, 
  {{7.449555748585402, -6.0304530548440045}}, 
  {{8.037341000877875, -5.221436060469057}}, 
  {{8.625126253170349, -4.41241906609411}}, 
  {{7.964680116814112, -7.022746804323951}}, 
  {{8.552465369106585, -6.213729809949004}}, 
  {{9.140250621399058, -5.404712815574057}}, 
  {{8.62512625317035, 4.412419066094105}}, 
  {{8.037341000877879, 5.221436060469053}}, 
  {{7.449555748585405, 6.030453054844}}, 
  {{6.861770496292932, 6.8394700492189475}}, 
  {{9.140250621399062, 5.404712815574052}}, 
  {{8.552465369106589, 6.213729809948999}}, 
  {{7.964680116814116, 7.022746804323947}}}]
Combinatorica`Private`Merge[igd_List, ild_List] :=
    Module[{i, j, gd = Flatten[igd], ld = Flatten[ild]},
        Join[Table[If[Position[ld,  Rule[gd[[i]][[1]], _]]=={},
                      gd[[i]],
                      {j} = First[Position[ld,  Rule[gd[[i]][[1]], _]]];
                      ld[[j]]
                   ],
                   {i, Length[gd]}
             ],
             Flatten[Table[If[MemberQ[gd, Rule[ld[[i]][[1]], _]],
                              {},
                              {ld[[i]]}
                           ],
                           {i, Length[ld]}
                     ],
                     1
             ]
        ]
    ]
Combinatorica`Private`Merge[rl1_List, rl2_List, rls__] := Combinatorica`Private`Merge[Combinatorica`Private`Merge[rl1, rl2], rls]
MinOp[l_List,f_] :=
	Module[{min=First[l]},
		Scan[ (If[ Apply[f,{#,min}], min = #])&, l];
		Return[min];
	]
MinimumChainPartition[g_Combinatorica`Graph] :=
	Combinatorica`ConnectedComponents[
		FromUnorderedPairs[
			Map[(#-{0,V[g]})&, BipartiteMatching[DilworthGraph[g]]],
			Vertices[g, All]
		]
	]
MinimumChangePermutations[l_List] := LexicographicPermutations[l] /; (Length[l] < 2)
MinimumChangePermutations[l_List] :=
	Module[{i=1,c,p=l,n=Length[l],k},
		c = Table[1,{n}];
		Join[
			{l},
			Table[
				While [ c[[i]] >= i, c[[i]] = 1; i++];
				If[OddQ[i], k=1, k=c[[i]] ];
				{p[[i]],p[[k]]} = {p[[k]],p[[i]]};
				c[[i]]++;
				i = 2;
				p,
				{n!-1}
			]
		]
	] 
 
MinimumChangePermutations[n_Integer] := MinimumChangePermutations[Range[n]]
MinimumEdgeLength[v_List,pairs_List] :=
        Max[ Select[
                Chop[ Map[(Sqrt[ N[(v[[#[[1]]]]-v[[#[[2]]]]) .
                        (v[[#[[1]]]]-v[[#[[2]]]])] ])&,pairs] ],
                (# > 0)&
        ], 0.001 ]
MinimumSpanningTree[e_List, g_Combinatorica`Graph] :=
	Module[{ne=Sort[e, (#1[[2]] <= #2[[2]])&],
                s=InitializeUnionFind[V[g]]},
		ChangeEdges[g,
			Select[Map[First, ne],
                               (If[FindSet[#[[1]],s]!=FindSet[#[[2]], s],
                                    s=UnionSet[#[[1]],#[[2]], s]; True,
                                    False
				])&
			]
		]
	] 
MinimumSpanningTree[g_Combinatorica`Graph] := MinimumSpanningTree[ Edges[g, Combinatorica`EdgeWeight], g ] /; UndirectedQ[g]
MinimumVertexColoring[g_Combinatorica`Graph] := MinimumVertexColoring[MakeSimple[g]] /; !UndirectedQ[g]
MinimumVertexColoring[g_Combinatorica`Graph] := Table[1, {V[g]}] /; EmptyQ[g]
MinimumVertexColoring[g_Combinatorica`Graph] := TwoColoring[g] /; BipartiteQ[g]
MinimumVertexColoring[g_Combinatorica`Graph] := {} /; (V[g] == 0)
MinimumVertexColoring[g_Combinatorica`Graph] := {1} /; (V[g] == 1)
MinimumVertexColoring[g_Combinatorica`Graph] := 
       Module[{col, oldCol, c, i},
              c = Max[oldCol = VertexColoring[g, Algorithm->Brelaz]];
              col = oldCol;
              For[i = c-1, i >= 3, i--, 
                  col = MinimumVertexColoring[g, i];
                  If[col == {}, Return[oldCol], oldCol = col]
              ];
              col
       ]
MinimumVertexColoring[g_Combinatorica`Graph, k_Integer, number_Symbol:One] := {} /; (V[g] == 0)
MinimumVertexColoring[g_Combinatorica`Graph, k_Integer, number_Symbol:One] := {1} /; ((V[g] == 1) && (k == 1))
MinimumVertexColoring[g_Combinatorica`Graph, k_Integer, number_Symbol:One] := {} /; ((V[g] == 1) && (k != 1))
MinimumVertexColoring[g_Combinatorica`Graph, k_Integer, number_Symbol:One] := 
       Module[{e}, 
              e = ToAdjacencyLists[g, Type->Simple];
              Backtrack[
                  Join[{{1}}, Table[Range[k], {V[g] - 1}]], 
                  (!MemberQ[#[[PriorEdges[e[[Length[#]]], Length[#]]]], Last[#]]) &, 
                  (!MemberQ[#[[PriorEdges[e[[Length[#]]], Length[#]]]], Last[#]]) &, 
                  number
              ]
       ]
PriorEdges[l_List, k_Integer] := Select[l, (# <= k) &]
MinimumVertexCover[g_Combinatorica`Graph] := {} /; EmptyQ[g]
MinimumVertexCover[g_Combinatorica`Graph] := Flatten[Position[Last[BipartiteMatchingAndCover[g]], 1]]/; BipartiteQ[g]
MinimumVertexCover[g_Combinatorica`Graph] := Complement[ Range[V[g]], MaximumClique[ Combinatorica`GraphComplement[g] ] ]
MultipleEdgesQ[g_Combinatorica`Graph] := Module[{e = Edges[g]}, Length[e] != Length[Union[e]]]
MultiplicationTable[elems_List, op_] :=
        With[{rules = Append[Thread[elems -> Range[Length[elems]]], _ -> 0]},
             Outer[Replace[op[##], rules] &, elems, elems, 1]]
MycielskiGraph[1] := CompleteGraph[1]
MycielskiGraph[2] := CompleteGraph[2]
MycielskiGraph[3] := Cycle[5]
MycielskiGraph[4] := GroetzschGraph
MycielskiGraph[k_Integer] := 
        Module[{g = MycielskiGraph[k - 1], al, n, i}, 
               n = V[g]; 
               al = ToAdjacencyLists[g]; 
               FromAdjacencyLists[
                   Join[Map[Join[#, n + #] &, al], 
                        Table[Append[al[[i]], 2n + 1], {i, n}], 
                        {Range[n + 1, 2n]}
                   ] 
               ]
        ] /; (k > 4)
Nary[0,b_]:={}
Nary[n_,b_]:=
    Module[{d = IntegerDigits[n,b], p},
           While[(p = Flatten[Position[d,_?NonPositive]]) != {},
                 d[[Last[p]]]+= b; 
                 d[[Last[p]-1]]-= 1; 
                 If[First[d] == 0, d = Rest[d]];
           ];
           d
    ]
NecklacePolynomial[n_Integer?Positive, c_, Cyclic] :=
        OrbitInventory[CyclicGroupIndex[n, x], x, c]
NecklacePolynomial[n_Integer?Positive, c_, Dihedral] :=
        OrbitInventory[DihedralGroupIndex[n, x], x, c]
Neighborhood[g_Combinatorica`Graph, v_Integer?Positive, 0] := {v} /; (1 <= v) && (v <= V[g])
Neighborhood[g_Combinatorica`Graph, v_Integer?Positive, k_Integer?Positive] := 
       Neighborhood[ToAdjacencyLists[g], v, k] /; (1 <= v) && (v <= V[g])
Neighborhood[al_List, v_Integer?Positive, 0] := {v} /; (1 <= v)&&(v<=Length[al])
Neighborhood[al_List, v_Integer?Positive, k_Integer?Positive] := 
       Module[{n = {v}, i},
              Do[n = Union[n, Flatten[al[[ n ]], 1]], {i, k}]; 
              n
       ] /; (1 <= v) && (v <= Length[al])
General::multedge = "The input graph has multiple edges between some vertices. These will be collapsed into single edges with weight equal to the sum of the individual edges.";
NetworkFlow[gin_Combinatorica`Graph, s_Integer, t_Integer, All] :=
	Block[{g = gin,al,f,rg,p,pe,pf,pb,i,j,m},
	      If[MultipleEdgesQ[g],
	          Message[NetworkFlow::multedge];
		  g=RemoveMultipleEdges[g, True]
	      ];
	      al=ToAdjacencyLists[g,Combinatorica`EdgeWeight];
              f = Map[ Map[Function[x, {First[x], 0}], #]&, al ];
              rg = ResidualFlowGraph[g,al,f]; 
              While[(p = Combinatorica`BreadthFirstTraversal[rg,s,t]) != {},
                    m = Min[ GetEdgeWeights[rg, pe = Partition[p,2,1]] ];
                    Scan[({i,j} = {#[[1]], #[[2]]};
                         pf = Position[f[[i]], {j, _}];
                         pb = Position[f[[j]], {i, _}];
                         If[(pb != {}) && (f[[j, pb[[1,1]], 2]] >= m),
                            f[[j, pb[[1,1]],2]]-=m,
                            f[[i, pf[[1,1]],2]]+=m
                         ])&,
                         pe
                    ];
                    rg = ResidualFlowGraph[g,al,f]; 
              ];
              f
	] /; (1 <= s) && (s <= V[gin]) && (1 <= t) && (t <= V[gin])
NetworkFlow[g_Combinatorica`Graph, s_Integer, t_Integer, Edge] :=
        Module[{f = NetworkFlow[g,s,t, All],i},
               Flatten[Table[
                             Map[{{i, First[#]}, Last[#]}&,
                                 Select[f[[i]], (Last[#] > 0)&]
                             ],
                             {i, Length[f]}
                       ], 1
               ]
        ] /; (1 <= s) && (s <= V[g]) && (1 <= t) && (t <= V[g])
NetworkFlow[gin_Combinatorica`Graph, s_Integer, t_Integer, Cut] := 
        Module[{g = gin, e, rg, u, v, cut},
	       If[MultipleEdgesQ[g],
	            Message[NetworkFlow::multedge];
		    g=RemoveMultipleEdges[g, True]
	       ];
	       e = Edges[g];
	       rg = ResidualFlowGraph[g, NetworkFlow[g, s, t, All]];
               u = Combinatorica`DepthFirstTraversal[rg, s]; v = Complement[Range[V[g]], u];
               cut = Select[e, MemberQ[u, #[[1]]] && MemberQ[v, #[[2]]] &];
               If[UndirectedQ[g], 
                  cut = Join[cut, Select[e, MemberQ[v, #[[1]]] && MemberQ[u, #[[2]]] &]]
               ];
               cut
        ]
NetworkFlow[g_Combinatorica`Graph, s_Integer, t_Integer] :=
        Module[{f = NetworkFlow[g,s,t, All]},
               Apply[Plus, Map[Last, f[[s]]]] -
               Apply[Plus, Map[Last, Select[Flatten[f, 1], 
                                            (First[#]==s)&
                                     ]
                           ]
               ]
        ] /; (1 <= s) && (s <= V[g]) && (1 <= t) && (t <= V[g])
NetworkFlowEdges[g_Combinatorica`Graph, source_Integer, sink_Integer] := NetworkFlow[g, source, sink, Edge]
NextBinarySubset[set_List,subset_List] := UnrankBinarySubset[RankBinarySubset[set,subset]+1, set]
NextComposition[l_List] :=
	Append[Table[0,{Length[l]-1}], Apply[Plus, l]] /; First[l]==Apply[Plus,l]
NextComposition[l_List] := NC[l]
NC = Compile[{{l, _Integer, 1}},  
             Module[{n = Apply[Plus, l], nl = l, t = Length[l],i}, 
                    While[l[[t]] == 0, t--];
                    nl[[t-1]]++;
                    Do[nl[[i]] = 0, {i, t, Length[l]}];
                    nl[[Length[l]]] = Apply[Plus, Take[l, -(Length[l] - t + 1)]] - 1; nl
             ]
     ]
NextGrayCodeSubset[l_List, s_List] := 
        If[ MemberQ[s,l[[1]]], Rest[s], Prepend[s,l[[1]] ] ] /; EvenQ[Length[s]]
NextGrayCodeSubset[l_List, s_List] := 
        Module[{i = 1}, 
               While[ ! MemberQ[s, l[[i]] ], i++]; 
               If[MemberQ[s, l[[i+1]] ], Rest[s], Insert[s, l[[i+1]], 2 ] ]]
NKS = Compile[{{n, _Integer}, {ss, _Integer, 1}}, 
              Module[{h = Length[ss], x = n},
                     While[x === ss[[h]], h--; x--];
                     Join[Take[ss, h - 1], 
                          Range[ss[[h]]+1, ss[[h]]+Length[ss]-h+1] 
                     ]
              ]
      ]
NextKSubset[s_List,ss_List] := Take[s,Length[ss]] /; (Take[s,-Length[ss]] === ss)
NextKSubset[s_List,ss_List] :=
        Map[s[[#]] &, 
            NKS[Length[s], Table[Position[s, ss[[i]]][[1, 1]], 
                                 {i, Length[ss]}
                           ]
            ]
        ]
NextLexicographicSubset[n_Integer, s_List] := NextLexicographicSubset[Range[n], s]
NextLexicographicSubset[l_List, {}] := {First[l]}
NextLexicographicSubset[l_List, s_List] := 
       Module[{elem}, 
              If[Last[s] === Last[l], 
                 (elem = s[[Length[s] - 1]]; 
                 Append[Drop[s,-2], l[[Position[l, elem][[1, 1]] + 1]]]), 
                 Append[s, l[[Position[l, Last[s]][[1, 1]] + 1]] ]
              ]
       ]
NextPartition[p_List] := Join[Drop[p,-1],{Last[p]-1,1}]  /; (Last[p] > 1)
NextPartition[p_List] := {Apply[Plus,p]}  /; (Max[p] == 1)
NextPartition[p_List] := NPT[p];
NPT = Compile[{{p, _Integer, 1}}, 
              Module[{k = Length[p], q = Table[0, {Length[p] + 2}], i, j, m, r}, 
                     j = Position[p, 1][[1, 1]] - 1;
                     Do[q[[i]] = p[[i]], {i, j - 1}];
                     m = Quotient[p[[j]] + (k - j), p[[j]] - 1];
                     Do[q[[i]] = p[[j]] - 1, {i, j, j + m - 1}];
                     r = Mod[p[[j]] + (k - j), p[[j]] - 1];
                     q[[j + m]] = r;
                     DeleteCases[q, 0]
              ]
      ]
NextPermutation[l_List] := Sort[l] /; (l === Reverse[Sort[l]])
NextPermutation[l_List] := 
        Module[{n = Length[l], i, j, t, nl = l},
               i = n-1; While[ Order[nl[[i]], nl[[i+1]]] == -1, i--];
               j = n; While[ Order[nl[[j]], nl[[i]]] == 1, j--];
               {nl[[i]], nl[[j]]} = {nl[[j]], nl[[i]]};
               Join[ Take[nl,i], Reverse[Drop[nl,i]] ]
        ]
NextSubset[set_List,subset_List] := 
       UnrankSubset[RankSubset[set,subset]+1, set]
NextTableau[t_?Combinatorica`TableauQ] :=
	Module[{s,y,row,j,count=0,tj,i,n=Max[t]},
		y = TableauToYVector[t];
		For [j=2, (j<n)  && (y[[j]]>=y[[j-1]]), j++];
		If [y[[j]] >= y[[j-1]],
			Return[ FirstLexicographicTableau[ ShapeOfTableau[t] ] ]
		];
		s = ShapeOfTableau[ Table[Select[t[[i]],(#<=j)&], {i,Length[t]}] ];
		{row} = Last[ Position[ s, s[[ Position[t,j] [[1,1]] + 1 ]] ] ];
		s[[row]] --;
		tj = FirstLexicographicTableau[s];
		If[ Length[tj] < row,
			tj = Append[tj,{j}],
			tj[[row]] = Append[tj[[row]],j]
		];
		Join[
			Table[
				Join[tj[[i]],Select[t[[i]],(#>j)&]],
				{i,Length[tj]}
			],
			Table[t[[i]],{i,Length[tj]+1,Length[t]}]
		]
	]
NoPerfectMatchingGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 4}}, {{2, 3}}, {{2, 4}}, {{3, 4}}, {{3, 5}}, 
  {{4, 5}}, {{5, 6}}, {{6, 7}}, {{6, 8}}, {{7, 13}}, {{7, 14}}, {{8, 9}}, 
  {{8, 10}}, {{9, 10}}, {{9, 11}}, {{9, 12}}, {{10, 11}}, {{10, 12}}, 
  {{11, 12}}, {{13, 14}}, {{13, 15}}, {{13, 16}}, {{14, 15}}, {{14, 16}}, 
  {{15, 16}}}, {{{0.3, 0.902}}, {{0.582, 0.902}}, {{0.3, 0.73}}, 
  {{0.582, 0.73}}, {{0.44, 0.664}}, {{0.44, 0.528}}, {{0.534, 0.434}}, 
  {{0.334, 0.434}}, {{0.178, 0.434}}, {{0.342, 0.288}}, {{0.216, 0.182}}, 
  {{0.046, 0.342}}, {{0.716, 0.434}}, {{0.534, 0.28}}, {{0.65, 0.168}}, 
  {{0.834, 0.33}}}]
NonLineGraphs := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 4}}, {{5, 7}}, {{5, 8}}, {{6, 7}}, {{6, 8}}, 
  {{7, 8}}, {{6, 9}}, {{5,9}}, {{10, 12}}, {{10, 13}}, {{10, 14}}, {{11, 12}}, 
  {{11, 13}}, {{11, 14}}, {{12, 13}}, {{12, 14}}, {{13, 14}}, {{15, 16}}, 
  {{16, 17}}, {{16, 18}}, {{17, 18}}, {{17, 19}}, {{18, 19}}, {{19, 20}}, 
  {{21, 22}}, {{21, 23}}, {{21, 24}}, {{22, 23}}, {{22, 24}}, {{22, 25}}, 
  {{23, 24}}, {{23, 25}}, {{25, 26}}, {{27, 28}}, {{27, 29}}, {{27, 30}}, 
  {{28, 29}}, {{28, 30}}, {{28, 31}}, {{28, 32}}, {{29, 30}}, {{29, 31}}, 
  {{29, 32}}, {{31, 32}}, {{33, 35}}, {{33, 36}}, {{33, 37}}, {{34, 35}}, 
  {{34, 36}}, {{34, 38}}, {{35, 36}}, {{37, 38}}, {{39, 40}}, {{39, 41}}, 
  {{39, 44}}, {{40, 41}}, {{40, 42}}, {{40, 43}}, {{40, 44}}, {{41, 42}}, 
  {{42, 43}}, {{43, 44}}, {{45, 46}}, {{45, 47}}, {{46, 47}}, {{46, 48}}, 
  {{47, 48}}, {{47, 49}}, {{48, 49}}, {{48, 50}}, {{49, 50}}}, 
 {{{0.104, 0.906}}, {{0.25, 0.982}}, {{0.25, 0.906}}, {{0.244, 0.83}}, 
  {{0.476, 0.974}}, {{0.474, 0.802}}, {{0.366, 0.894}}, {{0.476, 0.894}}, 
  {{0.582, 0.894}}, {{0.798, 0.986}}, {{0.77, 0.828}}, {{0.664, 0.79}}, 
  {{0.808, 0.908}}, {{0.946, 0.79}}, {{0.248, 0.762}}, {{0.08, 0.762}}, 
  {{0.238, 0.664}}, {{0.144, 0.664}}, {{0.076, 0.554}}, {{0.244, 0.554}}, 
  {{0.444, 0.738}}, {{0.356, 0.63}}, {{0.534, 0.63}}, {{0.444, 0.676}}, 
  {{0.442, 0.54}}, {{0.532, 0.54}}, {{0.968, 0.598}}, {{0.828, 0.718}}, 
  {{0.828, 0.474}}, {{0.89, 0.598}}, {{0.77, 0.606}}, {{0.682, 0.606}}, 
  {{0.158, 0.444}}, {{0.15, 0.206}}, {{0.096, 0.328}}, {{0.216, 0.328}}, 
  {{0.288, 0.444}}, {{0.288, 0.204}}, {{0.838, 0.398}}, {{0.838, 0.282}}, 
  {{0.944, 0.32}}, {{0.902, 0.182}}, {{0.77, 0.182}}, {{0.728, 0.318}}, 
  {{0.51, 0.464}}, {{0.422, 0.376}}, {{0.598, 0.376}}, {{0.422, 0.242}}, 
  {{0.598, 0.242}}, {{0.51, 0.154}}}]
NormalizeVertices[v:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}] := 
        Module[{nv = NormalizeVertices[Map[First, v]],i},
                Table[{nv[[i]], Apply[Sequence, Rest[v[[i]]]]}, {i, Length[nv]}]
        ]
NormalizeVertices[v:{{_?NumericQ, _?NumericQ}...}] := 
	Module[{v1 = TranslateVertices[v, {-Min[v], -Min[v]}]},
		DilateVertices[v1, 1/Max[v1, 0.01]]
	]
NormalizeVertices[g_Combinatorica`Graph] := 
        ChangeVertices[g, NormalizeVertices[Vertices[g]]]
NthPair = Compile[{{n, _Integer}}, 
             Module[{j}, 
                    j = Ceiling[(1 + Sqrt[1 + 8n])/2];
                    {Round[n - (j-1)(j-2)/2], j}
             ]
          ]
NthPermutation[r_Integer, l_List] := UnrankPermutation[r, l]
NthSubset[x_Integer, y_List] := First[Subsets[y, All, {Mod[x, 2^Length[y]]+1}]]
NumberOfPermutationsByInversions[n_Integer?Positive] := 
       Module[{p,z,i},
              p = Expand[Product[Cancel[ (z^i -1)/(z-1)], {i, 1, n}]];
              CoefficientList[p, z]
       ]
NumberOfPermutationsByInversions[n_Integer, k_Integer] := 0 /; (k > Binomial[n,2])
NumberOfPermutationsByInversions[n_Integer, 0] := 1 
NumberOfPermutationsByInversions[n_Integer, k_Integer?Positive] := 
        NumberOfPermutationsByInversions[n][[k+1]]
NumberOf2Paths[g_Combinatorica`Graph, v_Integer?Positive] := NumberOfKPaths[g, v, 2]
NumberOfCompositions[n_,k_] := Binomial[ n+k-1, n ]
NumberOfDerangements[0] := 1
NumberOfDerangements[n_Integer?Positive] := 
       Block[{$RecursionLimit = Infinity}, n * NumberOfDerangements[n-1] + (-1)^n]
NumberOfDirectedGraphs[0] := 1
NumberOfDirectedGraphs[n_Integer?Positive] :=
    Module[{x},
        OrbitInventory[PairGroupIndex[SymmetricGroupIndex[n, x], x, Ordered], 
                       x, 2
        ]
    ]
NumberOfDirectedGraphs[n_Integer, 0] := 1 /; (n >= 0)
NumberOfDirectedGraphs[n_Integer?Positive, m_Integer] := 
    Module[{x},
        Coefficient[GraphPolynomial[n, x, Directed], x^m]
    ]
NumberOfGraphs[0] := 1
NumberOfGraphs[n_Integer?Positive] :=
    Module[{x},
        OrbitInventory[PairGroupIndex[SymmetricGroupIndex[n, x], x], x, 2 ]
    ]
NumberOfGraphs[n_Integer, 0] := 1 /; (n >= 0)
NumberOfGraphs[n_Integer?Positive, m_Integer] :=
    Module[{x},Coefficient[GraphPolynomial[n, x], x^m]]
NumberOfInvolutions[n_Integer] := Module[{k}, n! Sum[1/((n - 2k)! 2^k k!), {k, 0, Quotient[n, 2]}]]
NumberOfKPaths[g_Combinatorica`Graph, v_Integer?Positive, 0] := 1 /; (1 <= v) && (v <= V[g])
NumberOfKPaths[g_Combinatorica`Graph, v_Integer?Positive, k_Integer?Positive] := 
       NumberOfKPaths[ToAdjacencyLists[g], v, k] 
NumberOfKPaths[al_List, v_Integer?Positive, 0] := 1 /; (1<=v)&&(v<=Length[al])
NumberOfKPaths[al_List, v_Integer?Positive, k_Integer?Positive] := 
       Module[{n = {v},i}, 
              Do[n = Flatten[al[[ n ]], 1]  , {i, k}]; 
              Sort[Map[Length, Split[Sort[n]]]] 
       ] /; (1 <= v) && (v <= Length[al]) 
NumberOfNecklaces[n_Integer?Positive, nc_Integer?Positive, Cyclic] :=
        Module[{x},OrbitInventory[CyclicGroupIndex[n, x], x, nc]]
NumberOfNecklaces[n_Integer?Positive, nc_Integer?Positive, Dihedral] :=
        Module[{x},OrbitInventory[DihedralGroupIndex[n, x], x, nc]]
NumberOfPartitions[n_Integer] := NumberOfPartitions1[n]
NumberOfPartitions1[n_Integer] := 0  /; (n < 0)
NumberOfPartitions1[n_Integer] := 1  /; (n == 0)
NumberOfPartitions1[n_Integer] := 
	Block[{$RecursionLimit = Infinity, k},
              NumberOfPartitions1[n] =
              Sum[(-1)^(k+1) NumberOfPartitions1[n - k (3k-1)/2] +
                  (-1)^(k+1) NumberOfPartitions1[n - k (3k+1)/2],
                  {k, Ceiling[ (1+Sqrt[1.0 + 24n])/6 ], 1, -1}
              ]
	]
NumberOfPartitions[n_Integer, k_Integer] := NumberOfPartitions2[n, k] /; ((n >= 0) && (k >= 0))
NumberOfPartitions2[n_Integer?Positive, 0] := 0 
NumberOfPartitions2[0, k_Integer] := 1 
NumberOfPartitions2[n_Integer?Positive, 1] := 1
NumberOfPartitions2[n_Integer?Positive, k_Integer?Positive] := NumberOfPartitions[n] /; (k >= n)
NumberOfPartitions2[n_Integer, k_Integer] := 
        Block[{$RecursionLimit = Infinity},
               NumberOfPartitions2[n, k] = NumberOfPartitions2[n, k-1] + NumberOfPartitions2[n-k, k]
        ]
NumberOfPermutationsByCycles[n_Integer,m_Integer] := Abs[StirlingS1[n,m]]
NumberOfPermutationsByType[l_List] := (Length[l]!)/Apply[Times, Table[l[[i]]!i^(l[[i]]), {i, Length[l]}]]
NumberOfSpanningTrees[g_Combinatorica`Graph] := 0 /; (V[g] == 0)
NumberOfSpanningTrees[g_Combinatorica`Graph] := 1 /; (V[g] == 1)
NumberOfSpanningTrees[g_Combinatorica`Graph] :=
        Module[{m = ToAdjacencyMatrix[g]},
	       Cofactor[ DiagonalMatrix[Map[(Apply[Plus,#])&,m]] - m, {1,1}]
        ]
NumberOfTableaux[{}] := 1
NumberOfTableaux[s_List] := 
	Module[{row,col,transpose=TransposePartition[s]},
		(Apply[Plus,s])! /
		Product [
			(transpose[[col]]-row+s[[row]]-col+1),
			{row,Length[s]}, {col,s[[row]]}
		]
	]
NumberOfTableaux[n_Integer] := Apply[Plus, Map[NumberOfTableaux, Partitions[n]]]
OctahedralGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{2, 3}}, {{4, 5}}, {{4, 6}}, {{5, 6}}, {{1, 4}}, 
  {{1, 6}}, {{2, 4}}, {{2, 5}}, {{3, 5}}, {{3, 6}}}, 
 {{{-0.4999999999999998, 0.8660254037844387}}, 
  {{-0.5000000000000004, -0.8660254037844384}}, {{1., 0}}, 
  {{-0.3333333333333333, 4.082021179407924*^-17}}, 
  {{0.16666666666666644, -0.28867513459481303}}, 
  {{0.16666666666666666, 0.28867513459481287}}}]
OddGraph[n_Integer] := MakeGraph[KSubsets[Range[2n-1],n-1], 
                                 (SameQ[Intersection[#1,#2],{}])&,
                                 Type -> Undirected
                       ] /; (n > 1)
Options[Combinatorica`Graph] = Sort[Join[
    $GraphSemanticOptions,
    $GraphEdgeStyleOptions,
    $GraphVertexStyleOptions
]];
Combinatorica`Graph[s:(_String | {_String, ___} | {_Integer,_Integer}), a___, o___?OptionQ] :=
    Module[{edges = GraphData[s, a, "EdgeIndices"],
          vertices = GraphData[s, a, "VertexCoordinates"]},
        If[MatrixQ[vertices, NumericQ],
            vertices = List /@ vertices,
            If[IntegerQ[GraphData[s, a, "VertexCount"]],
                vertices = CircularEmbedding[GraphData[s, a, "VertexCount"]],
                vertices = $Failed
            ]
        ];
        Combinatorica`Graph[List /@ edges, vertices, o]/;
            MatchQ[edges, {{_Integer,_Integer}...}] &&
                vertices =!= $Failed
    ]/;ListQ[GraphData[s, a, "EdgeIndices"]]
OrbitInventory[ci_?PolynomialQ, x_Symbol, weights_List] :=
        Expand[ci /. Table[x[i] ->  Apply[Plus, Map[#^i&, weights]],
                                  {i, Exponent[ci, x[1]]}
              ]
        ]
OrbitInventory[ci_?PolynomialQ, x_Symbol, r_] :=
        Expand[ci /. Table[x[i] -> r, {i, Exponent[ci, x[1]]} ]]
OrbitRepresentatives[g_List, x_List, f_:Combinatorica`Permute] :=
        Module[{y = Combinatorica`Orbits[g, x, f]},
               Table[y[[i]][[1]], {i, Length[y]}]
        ] /; ((Length[g] > 0) && (Length[x] > 0)) && (Length[ g[[1]] ] == Length[ x[[1]] ]) 
Combinatorica`Orbits[g_List, x_List, f_:Combinatorica`Permute] :=
        Module[{y = x, n = Length[g], orbit, out = {}, i},
               While[y != {},
                     orbit = Table[Apply[f, {y[[1]], g[[i]]}], {i, 1, n}];
                     y = Complement[y, orbit];
                     AppendTo[out, orbit];
                     Length[y]
               ];
               out
        ]
OrientGraph[g_Combinatorica`Graph] :=
	Module[{pairs,newg,rest,cc,c,i,e},
		pairs = Flatten[Map[(Partition[#,2,1])&,ExtractCycles[g]],1];
		newg = FromUnorderedPairs[pairs,Vertices[g, All]];
		rest = ToOrderedPairs[ Combinatorica`GraphDifference[ g, newg ] ];
		cc = Sort[Combinatorica`ConnectedComponents[newg], 
                          (Length[#1]>=Length[#2])&];
		c = First[cc];
		Do[
			e = Select[rest,(MemberQ[c,#[[1]]] &&
					 MemberQ[cc[[i]],#[[2]]])&];
			rest = Complement[rest,e,Map[Reverse,e]];
			c = Union[c,cc[[i]]];
			pairs = Join[pairs, Prepend[ Rest[e],Reverse[e[[1]]] ] ],
			{i,2,Length[cc]}
		];
		FromOrderedPairs[
			Join[pairs, Select[rest,(#[[1]] > #[[2]])&] ],
			Vertices[g, All]
		]
	] /; SameQ[Bridges[g],{}]
OutDegree[g_Combinatorica`Graph, n_Integer] := Length[Select[Edges[g], (First[#]==n)&]]
OutDegree[g_Combinatorica`Graph] := Map[Length, ToAdjacencyLists[g]]
PairGroup[g_List] := KSubsetGroup[g, KSubsets[Range[Max[g[[1]]]], 2]] /; (Length[g] > 0)
PairGroup[g_List, Ordered] :=
        KSubsetGroup[g,
                     Complement[Flatten[Table[{i, j}, {i, Max[g[[1]]]}, 
                                              {j, Max[g[[1]]]}
                                        ],
                                        1
                                ],
                                Table[{i, i}, {i, Max[g[[1]]]}]
                     ],
                     Ordered
        ] /; (Length[g] > 0)
PairGroupIndex[g_, x_Symbol] := PairGroupIndex[CycleIndex[g, x], x] /; Combinatorica`PermutationQ[g[[1]]]
PairGroupIndex[ci_?PolynomialQ, x_Symbol]:=
        Module[{f1,f2,f3,PairCycles,i},
               f1[x[i1_]^(j1_)] := 1;
               f1[x[i1_]] := 1;
               f1[x[i1_]*x[(i2_)^(j2_)]] := x[LCM[i1, i2]]^(j2*GCD[i1, i2]);
               f1[x[i1_]^(j1_)*x[i2_]] := x[LCM[i1, i2]]^(j1*GCD[i1, i2]);
               f1[x[i1_]*x[i2_]] := x[LCM[i1, i2]]^GCD[i1, i2];
               f1[x[i1_]^(j1_)*x[i2_]^(j2_)] := x[LCM[i1, i2]]^(j1*j2*GCD[i1, i2]);
               f1[(a_)*(t__)] := Block[{$RecursionLimit = Infinity},
                                       Product[f1[a*{t}[[i]]], {i, Length[{t}]}]*
                                       f1[Apply[Times, {t}]]
                                 ];
               f2[x[i1_]^j1_]:=x[i1]^(i1 Binomial[j1,2]);
               f2[x[i1_]]:=1;
               f2[a_  b_ ]:= Block[{$RecursionLimit = Infinity}, f2[a] f2[b]];
               f3[x[i1_]]:=If[OddQ[i1],x[i1]^( (i1-1)/2),
                       x[i1]^( (i1-2)/2) * x[i1/2]];
               f3[x[i1_]^j1_]:=If[OddQ[i1],x[i1]^(j1 (i1-1)/2),
                       x[i1]^(j1 (i1-2)/2) * x[i1/2]^j1];
               f3[a_ b_]:= Block[{$RecursionLimit = Infinity}, f3[a] f3[b]];
               PairCycles[u_ + v_] := Block[{$RecursionLimit = Infinity},
                                            PairCycles[u] + PairCycles[v]
                                      ];
               PairCycles[a_?NumericQ b_]:= Block[{$RecursionLimit = Infinity},
                                                 a PairCycles[b]
                                           ];
               PairCycles[a_]:=f1[a] f2[a] f3[a];
               Expand[PairCycles[Expand[ci]]]
        ]
PairGroupIndex[g_, x_, Ordered] := PairGroupIndex[CycleIndex[g, x], x, Ordered] /; Combinatorica`PermutationQ[g[[1]]]
PairGroupIndex[ci_?PolynomialQ, x_, Ordered]:=
        Module[{f1,f2,f3,PairCycles,i},
               f1[x[i1_]^(j1_)] := 1;
               f1[x[i1_]] := 1;
               f1[x[i1_]*x[(i2_)^(j2_)]] := x[LCM[i1, i2]]^(2*j2*GCD[i1, i2]);
               f1[x[i1_]^(j1_)*x[i2_]] := x[LCM[i1, i2]]^(2*j1*GCD[i1, i2]);
               f1[x[i1_]*x[i2_]] := x[LCM[i1, i2]]^(2*GCD[i1, i2]);
               f1[x[i1_]^(j1_)*x[i2_]^(j2_)] :=
                       x[LCM[i1, i2]]^(2*j1*j2*GCD[i1, i2]);
               f1[(a_)*(t__)] := Block[{$RecursionLimit = Infinity},
                                        Product[f1[a*{t}[[i]]], {i, Length[{t}]}]*
                                        f1[Apply[Times, {t}]]   
                                 ];
               f2[x[i1_]^j1_]:=x[i1]^(i1 j1 (j1-1));
               f2[x[i1_]]:=1;
               f2[a_  b_ ]:= Block[{$RecursionLimit = Infinity}, f2[a] f2[b]];
               f3[x[i1_]]:= x[i1]^(i1-1);
               f3[x[i1_]^j1_]:= x[i1]^(j1 (i1-1));
               f3[a_ b_]:= Block[{$RecursionLimit = Infinity}, f3[a] f3[b]];
               PairCycles[u_ + v_] := Block[{$RecursionLimit = Infinity},
                                           PairCycles[u]+ PairCycles[v]
                                     ];
               PairCycles[a_?NumericQ b_] := Block[{$RecursionLimit = Infinity},
                                                  a PairCycles[b]
                                            ];
               PairCycles[a_]:=f1[a] f2[a] f3[a];
               Expand[PairCycles[Expand[ci]]]
        ]
ParentsToPaths[l_?MatrixQ, i_Integer, j_Integer]/;
    (i >= 1 && i <= Length[l] && j >= 1 && j <= Length[l]) := 
        If[First[#] =!= i, {}, #]&[
	    Rest[Reverse[FixedPointList[l[[i, #]] &, j]]]
	]
ParentsToPaths[l_?MatrixQ, i_Integer]/;(i >= 1 && i <= Length[l]) := 
        Table[ParentsToPaths[l, i, j], {j, Length[l]}]
PartialOrderQ[r_?SquareMatrix] := ReflexiveQ[r] && AntiSymmetricQ[r] && TransitiveQ[r]
PartialOrderQ[g_Combinatorica`Graph] := ReflexiveQ[g] && AntiSymmetricQ[g] && TransitiveQ[g]
Options[PartitionLattice] = {Type -> Undirected, Combinatorica`VertexLabel -> False}
PartitionLattice[n_Integer?Positive, opts___?OptionQ] := 
        Module[{type, label, s = SetPartitions[n], br, g},
               {type, label} = {Type, Combinatorica`VertexLabel} /. Flatten[{opts, Options[PartitionLattice]}];
               br = CoarserSetPartitionQ[#2, #1]&;
               g = MakeGraph[s, br];
               If[(label === On) || (label === True),
                  g = SetVertexLabels[MakeGraph[s, br], Map[SetPartitionToLabel, s]];
               ];
               If[type === Undirected, HasseDiagram[g], g]
        ]
PartitionQ[p_List] := (Min[p]>0) && Apply[And, Map[IntegerQ,p]]
PartitionQ[n_Integer, p_List] := (Apply[Plus, p] === n) && (Min[p]>0) && Apply[And, Map[IntegerQ,p]]
Partitions[n_Integer] := Partitions[n,n]
Partitions[n_Integer,_] := {} /; (n<0)
Partitions[0,_] := { {} }
Partitions[n_Integer,1] := { Table[1,{n}] }
Partitions[_,0] := {}
Partitions[n_Integer, maxpart_Integer] :=
        Block[{$RecursionLimit = Infinity},
	      Join[Map[(Prepend[#,maxpart])&, Partitions[n-maxpart,maxpart]],
                   Partitions[n,maxpart-1]
              ]
	]
Options[Path] = {Type->Undirected};
Path[1, opts___?OptionQ] := 
         Module[{type = Type /. Flatten[{opts, Options[Path]}]},
                CompleteGraph[1, Type -> type]
         ]
Path[n_Integer?Positive, opts___?OptionQ] := 
         Module[{type = Type /. Flatten[{opts, Options[Path]}],i},
                If[type === Undirected,
	           Combinatorica`Graph[Table[{{i, i+1}}, {i, n-1}], Map[({{#,0}})&,Range[n]]],
	           Combinatorica`Graph[Table[{{i, i+1}}, {i, n-1}], Map[({{#,0}})&,Range[n]], EdgeDirection -> True]
                ]
         ]
PathMidpoint[p1_List, p2_List, epsilon_] := 
        ((p1 + p2)/2 + {0,epsilon}) /; (p1[[2]] == p2[[2]])
PathMidpoint[p1_List, p2_List, epsilon_] := 
        Block[{pmid = (p1+p2)/2, yvalue, s,x,y},
              yvalue  = pmid[[2]] + 
                        ((p1 -p2)[[1]])/((p2 - p1)[[2]])(x - pmid[[1]]);
	      s = Solve[(y-pmid[[2]])^2 + (x - pmid[[1]])^2 == epsilon^2 /.
		         y -> yvalue, x];
              If[epsilon>0, {x, yvalue}/.s[[1]], {x, yvalue}/.s[[2]]]
	]
PathQ[g_Combinatorica`Graph] := 
         Module[{d = Degrees[g]}, 
                ConnectedQ[g] && Count[d, 1] == 2 && Count[d, 2] == V[g]-2
         ]
PerfectQ::largegraph =
"The graph has `1` vertices, which is too large for the current algorithm.";
PerfectQ[g_Combinatorica`Graph] :=
    Module[{s = Catch[Quiet[Subsets[Range[V[g]]], {Subsets::toomany}],
                      SystemException[___], $Failed&]},
        If[s === $Failed || Head[s] =!= List,
            Message[PerfectQ::largegraph, V[g]]];
        Apply[
                And,
                Map[(ChromaticNumber[#] == Length[MaximumClique[#]])&,
                    Map[(InduceSubgraph[g,#])&, Subsets[Range[V[g]]] ] ]
        ]/; Head[s] === List
    ]
PermutationGraph[p_?Combinatorica`PermutationQ] :=
        Module[{q = InversePermutation[p]},
                MakeGraph[Range[Length[q]], 
                          ((#1 < #2 && q[[#1]] > q[[#2]]) ||
                          (#1 > #2 && q[[#1]] < q[[#2]]))&, 
                          Type -> Undirected
                ]
        ]
PermutationGroupQ[perms_List] :=
	FreeQ[ MultiplicationTable[perms,Combinatorica`Permute], 0] &&
		EquivalenceRelationQ[SamenessRelation[perms]]
Combinatorica`PermutationQ[e_List] := (Sort[e] === Range[Length[e]])
PermutationToTableaux[{}] := {{}, {}}
PermutationToTableaux[p_?Combinatorica`PermutationQ] := 
       Module[{pt = {{p[[1]]}}, qt = {{1}}, r, i}, 
              Do[{pt, r} = InsertIntoTableau[p[[i]], pt, All];
                 If[r <= Length[qt], AppendTo[qt[[r]], i], AppendTo[qt, {i}]],
                 {i, 2, Length[p]}
              ]; 
              {pt, qt}
       ] 
PermutationType[p_?Combinatorica`PermutationQ] := 
        Module[{m = Map[Length, Combinatorica`ToCycles[p]], c = Table[0, {Length[p]}], i},
               Do[c[[ m[[i]] ]]++, {i, Length[m]}];
               c
        ]
PermutationWithCycle[n_Integer, l_List] := 
        Combinatorica`FromCycles[Append[Map[{#} &, Complement[Range[n], l]], l]]
Unprotect[Permutations]
Permutations[n_Integer] := Permutations[Range[n]]
Protect[Permutations]
Combinatorica`Permute[l_List,p_?Combinatorica`PermutationQ] := l [[ p ]]
Combinatorica`Permute[l_List,p_List] := Map[ (Combinatorica`Permute[l,#])&, p] /; (Apply[And, Map[Combinatorica`PermutationQ, p]])
PermuteEdges[e_List, s_List, n_Integer] :=
        Module[{t = Table[0, {n}], i},
               Do[t[[ s[[i]] ]] = i, {i, Length[s]}];
               Map[Prepend[Rest[#], t[[ First[#] ]]]&,
                   Select[e, (MemberQ[s, First[#][[1]]] &&
                             MemberQ[s, First[#][[2]]])&
                   ]
               ]
        ]
PermuteSubgraph[g_Combinatorica`Graph,{}] := Combinatorica`Graph[{},{}]
PermuteSubgraph[g_Combinatorica`Graph, s_List] :=
        Combinatorica`Graph[
              If[UndirectedQ[g], 
                 Map[Prepend[Rest[#], Sort[ First[#] ] ]&, 
                     PermuteEdges[Edges[g,All], s, V[g]]
                 ],
                 PermuteEdges[Edges[g,All], s, V[g]]
              ],
              Vertices[g, All][[ Sort[s] ]],
              Apply[Sequence, GraphOptions[g]]
        ] /; (Length[s] <= V[g]) && (Length[Union[s]]==Length[s])
Combinatorica`PetersenGraph :=  GeneralizedPetersenGraph[5, 2]
PlanarGivenCycleQ[g_Combinatorica`Graph, cycle_List] :=
        Module[{b, j, i},
               {b, j} = FindBridge[g, cycle];
               If[Length[j] === 1,
                  If[b === {}, True, SingleBridgeQ[First[b], First[j]]],
                  If[InterlockQ[j, cycle],
                     False,
                     Apply[And, Table[SingleBridgeQ[b[[i]],j[[i]]], {i,Length[b]}]]
                  ]
               ]
        ]
PlanarQ[g_Combinatorica`Graph] :=
        Block[{simpleGraph = MakeSimple[g], 
               $RecursionLimit = Infinity},
          SimplePlanarQ[simpleGraph]
        ]
Polya[g_List, m_] := OrbitInventory[CycleIndex[g, x], x, m]
ProductVertices[vg_List, vh_List] :=
	Flatten[
		Map[(TranslateVertices[
                         DilateVertices[vg, 1/(Max[Length[vg],Length[vh]])], #
                     ])&,
                     RotateVertices[vh,Pi/2]
		],
		1
	] /; (vg != {}) && (vh != {})
PseudographQ[g_Combinatorica`Graph] := MemberQ[Edges[g], _?(Function[l, l[[1]] == l[[2]]])]
RGFQ[{}]:= True;
RGFQ[r_]:= 
        Module[{m = Table[1, {Length[r]}],i},
               ListQ[r] && (Length[r] > 0) && (Depth[r]==2) && (r[[1]]==1) && 
               (m[[1]] = 1; 
                Do[m[[i]]=Max[m[[i-1]], r[[i]]], {i, 2, Length[r]}];
                Apply[And, Table[r[[i]] <= (m[[i-1]] + 1), {i, 2, Length[r]}]]
               )
        ]
RGFToSetPartition[{}] := {{}}
RGFToSetPartition[rgf_?RGFQ] := RGFToSetPartition[rgf, Range[Length[rgf]]]
RGFToSetPartition[{}, {}] := {{}}
RGFToSetPartition[rgf_?RGFQ, set_List] :=
        Table[set[[Flatten[Position[rgf, i]]]],
              {i, Max[rgf]}
        ] /; (Length[rgf] === Length[set]) 
RGFs[0] := {}
RGFs[1] := {{1}}
RGFs[n_Integer?Positive] := RGFs1[n]
RGFs1 = Compile[{{n, _Integer}}, 
               Module[{r = Table[1, {n}], c = Prepend[Table[2, {n - 1}], 1], i},
                      Transpose[
                         NestList[(i = n;
                                   While[#[[1, i]] === #[[2, i]], i--]; 
                                   {Join[Take[#[[1]], i - 1], {#[[1, i]] + 1}, 
                                         Table[1, {n - i}]
                                    ],
                                    Join[Take[#[[2]], i], 
                                         Table[Max[#[[1, i]] + 2, #[[2, i]]], 
                                               {n - i}
                                         ]
                                    ]
                                   }
                                  )&, 
                                  {r, c},  
                                  BellB[n] - 1
                         ]
                      ][[1]]
               ]
       ]
RadialEmbedding[g_Combinatorica`Graph, ct_Integer] := 
        ChangeVertices[g, Vertices[RadialEmbedding[MakeUndirected[g], ct]]] /; !UndirectedQ[g] && (1 <= ct) && (ct <= V[g])
RadialEmbedding[g_Combinatorica`Graph, ct_Integer] :=
	Module[{center=ct,ang,i,da,theta,n,v,positioned,done,next,new,
                e = ToAdjacencyLists[g]},
		ang = Table[{0,2 Pi},{n=V[g]}];
		v = Table[{0,0},{n}];
		positioned = next = done = {center};
		While [next != {},
			center = First[next];
			new = Complement[e[[center]], positioned];
			Do [
				da = (ang[[center,2]]-ang[[center,1]])/Length[new];
				ang[[ new[[i]] ]] = {ang[[center,1]] + (i-1)*da, ang[[center,1]] + i*da};
				theta = Apply[Plus,ang[[ new[[i]] ]] ]/2;
				v[[ new[[i]] ]] = v[[center]] + N[{Cos[theta],Sin[theta]}],
				{i,Length[new]}
			];
			next = Join[Rest[next],new];
			positioned = Union[positioned,new];
			AppendTo[done,center]
		];
		ChangeVertices[g, Map[{#}&,v]]
	] /; (1 <= ct) && (ct <= V[g])
RadialEmbedding[g_Combinatorica`Graph] := RadialEmbedding[g, First[Combinatorica`GraphCenter[g]]];
Radius[g_Combinatorica`Graph] := Min[ Eccentricity[g] ]
RandomComposition[n_Integer,k_Integer] :=
	Map[
		(#[[2]] - #[[1]] - 1)&,
		Partition[Join[{0},RandomKSubset[Range[n+k-1],k-1],{n+k}], 2, 1]
	]
Options[Combinatorica`RandomGraph] = {Type -> Undirected};
Combinatorica`RandomGraph[n_Integer, p_?NumericQ, {x_Integer, y_Integer}] := 
        SetEdgeWeights[Combinatorica`RandomGraph[n, p], 
                       WeightingFunction -> RandomInteger,
                       WeightRange -> {x, y}
        ]
Combinatorica`RandomGraph[n_Integer, p_?NumericQ, Directed] := Combinatorica`RandomGraph[n, p, Type->Directed]
Combinatorica`RandomGraph[0, p_?NumericQ, opts___?OptionQ] := Combinatorica`Graph[{}, {}]
Combinatorica`RandomGraph[n_Integer?Positive, p_?NumericQ, opts___?OptionQ] :=
        Module[{type},
               type = Type /. Flatten[{opts, Options[Combinatorica`RandomGraph]}];
               If[type === Directed, RDG[n, p], RG[n, p] ]
        ]
RG[1, p_] := Combinatorica`Graph[{}, {{{1.0,0}}}]
RG[n_, p_] := 
        Module[{d = BinomialDistribution[Binomial[n, 2], p]}, 
               Combinatorica`Graph[Map[{NthPair[#]}&, RandomKSubset[Range[Binomial[n, 2]], RandomInteger[d]]],
                     CircularEmbedding[n]
               ]
        ]
RDG[1, p_] := Combinatorica`Graph[{}, {{{1.0,0}}}, EdgeDirection -> True]
RDG[n_, p_] :=
        Module[{d = BinomialDistribution[n (n-1), p], i, j}, 
               Combinatorica`Graph[Map[(j = Mod[#, n-1]; i = Quotient[#, n-1];
                         If[j!=0, i++, j=n-1]; 
                         If[j >= i, {{i, j+1}}, {{i, j}}])&, 
                         RandomKSubset[Range[n (n-1)], RandomInteger[d]]
                     ],
                     CircularEmbedding[n],
                     EdgeDirection -> True
               ]
        ]
RandomHeap[n_Integer] := Heapify[Combinatorica`RandomPermutation[n]]
RandomKSetPartition[{}, 0] := {}
RandomKSetPartition[set_List, k_Integer?Positive] :=
        UnrankKSetPartition [
           RandomInteger[StirlingSecond[Length[set], k]-1], set, k
        ] /; ((Length[set] > 0) && (k <= Length[set]))
RandomKSetPartition[0, 0] := {}
RandomKSetPartition[n_Integer?Positive, k_Integer?Positive] := RandomKSetPartition [Range[n], k] /; (k <= n)
RandomKSubset[n_Integer,k_Integer] := RandomKSubset[Range[n],k]
RandomKSubset[s_List, k_Integer] := s[[Sort[Combinatorica`RandomPermutation[Length[s]][[Range[k] ]]]]]
RandomPartition[n_Integer?Positive] :=
  Module[{mult = Table[0, {n}], j, d, r=n, z},
    While[ (r > 0),
      d = 1;  j = 0;
      z = RandomReal[] r PartitionsP[r];
      While [z >= 0,
         j++;
         If [r-j*d < 0, {j=1; d++;}];
         z -= j*PartitionsP[r-j*d];
      ];
      r -= j d;
      mult[[j]] += d;
    ];
    Reverse[Flatten[Table[Table[j, {mult[[j]]}], {j, Length[mult]}]]]
  ]
Combinatorica`RandomPermutation[n_Integer]/; n < 1 := {}
Combinatorica`RandomPermutation[n_Integer] := RandomSample[Range[n]]
Combinatorica`RandomPermutation[{}] := {}
Combinatorica`RandomPermutation[l_List] := RandomSample[l]
RandomRGF[0] := {}
RandomRGF[n_Integer?Positive] := UnrankRGF[RandomInteger[BellB[n]-1], n]
RandomSetPartition[{}] := {}
RandomSetPartition [set_List] :=
        UnrankSetPartition [RandomInteger[BellB[Length[set]]-1], set] /; (Length[set] > 0)
RandomSetPartition [n_Integer] := RandomSetPartition [ Range[n] ]
RandomSquare[y_List,p_List] :=
	Module[{i=RandomInteger[{1,First[y]}], j=RandomInteger[{1,First[p]}]},
		While[(i > y[[j]]) || (j > p[[i]]), 
			i = RandomInteger[{1,First[y]}];
			j = RandomInteger[{1,First[p]}]
		];
		{i,j}
	]
RandomSubset[set_List] := UnrankSubset[RandomInteger[2^(Length[set])-1],set]
RandomSubset[0] := {}
RandomSubset[n_Integer] := UnrankSubset[RandomInteger[2^(n)-1], Range[n]]
RandomTableau[shape:{__Integer}] :=
	Module[{i,j,n=Apply[Plus,shape],done,l,m,h=1,k,y,p=shape},
	    i = j = n;
		y= Join[TransposePartition[shape],Table[0,{n - Max[shape]}]];
		Do[
			{i,j} = RandomSquare[y,p]; done = False;
			While [!done,
				h = y[[j]] + p[[i]] - i - j;
				If[ h != 0,
					If[ RandomReal[] < 0.5,
						j = RandomInteger[{j,p[[i]]}],
						i = RandomInteger[{i,y[[j]]}]
					],
					done = True
				];
			];
			p[[i]]--; y[[j]]--;
			y[[m]] = i,
			{m,n,1,-1}
		];
		YVectorToTableau[y]
	]/; (GreaterEqual @@ shape) && (Last[shape] > 0)
    
RandomTableau::inform = "RandomTableau only accepts a list of integers greater than zero in decreasing order, giving the shape of the tableau.";
RandomTableau[shape_List] := ($$dummy/;(Message[RandomTableau::inform]/;False))
RandomTree[1] := Combinatorica`Graph[{}, {{{0, 0}}}]
RandomTree[n_Integer?Positive] :=
	RadialEmbedding[CodeToLabeledTree[Table[RandomInteger[{1,n}], {n-2}] ], 1]
RandomVertices[n_Integer?Positive] := Table[{{RandomReal[], RandomReal[]}}, {n}]
RandomVertices[g_Combinatorica`Graph] := ChangeVertices[g, RandomVertices[V[g]] ]
RankBinarySubset[set_List,subset_List] :=
	Module[{i,n=Length[set]},
		Sum[ 2^(n-i) * If[ MemberQ[subset,set[[i]]], 1, 0], {i,n}]
	]
RankGraph[g_Combinatorica`Graph, start_List] :=
	Module[ {rank = Table[0,{V[g]}],edges = ToAdjacencyLists[g],v,
                 queue,new},
		Scan[ (rank[[#]] = 1)&, start];
		queue = start;
		While [queue != {},
			v = First[queue];
			new = Select[ edges[[v]], (rank[[#]] == 0)&];
			Scan[ (rank[[#]] = rank[[v]]+1)&, new];
			queue = Join[ Rest[queue], new];
		];
		rank
	]
 
RankGrayCodeSubset[l_List, s_List] := 
       Module[{c = Table[If[MemberQ[s, l[[i]]], 1, 0], {i, Length[l]}], b = Table[0, {Length[l]}], n = Length[l], i}, 
              b[[ 1 ]] = c[[ 1 ]]; 
              Do[b[[i]] = Mod[b[[i - 1]] + c[[i]], 2], {i, 2, n}]; 
              FromDigits[b, 2]
       ]
RankKSetPartition[sp_?SetPartitionQ] := 
        Module[{s = Sort[Flatten[sp, 1]]}, 
               RankKSetPartition1[ToCanonicalSetPartition[sp, s], s]
        ] 
RankKSetPartition[sp_List, s_List] := 
        RankKSetPartition1[ToCanonicalSetPartition[sp, s], s] /; SetPartitionQ[sp, s]
RankKSetPartition1[sp_List, s_List] := 0 /; (Length[sp] === Length[s])
                  
RankKSetPartition1[sp_List, s_List] :=
        Block[{k = Length[sp], n = Length[s], t, orderedT, j,
               $RecursionLimit = Infinity}, 
              If[First[sp] === {s[[1]]}, 
                 RankKSetPartition1[Rest[sp], Rest[s]],
                 (t = Prepend[Rest[sp], Rest[First[sp]]];
                 orderedT = ToCanonicalSetPartition[t, Rest[s]];
                 {j} = First[Position[orderedT, First[t]]];
                 StirlingSecond[n-1, k-1]+k*RankKSetPartition1[orderedT, Rest[s]]+j-1)            
              ]
        ]
RankKSubset[{},s_List] := 0
RankKSubset[ss_List, s_List] := 0 /; (Length[ss] === Length[s])
RankKSubset[ss_List, s_List] := Position[s, ss[[1]]][[1, 1]] - 1 /; (Length[ss] === 1)
RankKSubset[ss_List, s_List] := 
       Block[{n = Length[s], k = Length[ss], 
              x = Position[s, ss[[1]]][[1, 1]], $RecursionLimit = Infinity},
              Binomial[n, k] - Binomial[n-x+1, k] + RankKSubset[Rest[ss], Drop[s, x]]
       ]
RankPermutation[{1}] := 0
RankPermutation[{}] := 0
RankPermutation[p_?Combinatorica`PermutationQ] := 
        Block[{$RecursionLimit = Infinity},
              (p[[1]]-1) (Length[Rest[p]]!) + 
              RankPermutation[ Map[(If[#>p[[1]], #-1, #])&, Rest[p]]]
        ]
RankRGF[r_List] := 
        Module[{u = 1, n = Length[r], i}, DValues[n, 1];
               Sum[u = Max[u, r[[i - 1]]]; DValues[n - i, u]*(r[[i]] - 1), 
                   {i, 2, n}
               ]
        ]
RankSetPartition[sp_?SetPartitionQ] :=
       Module[{s = Sort[Flatten[sp, 1]], n, k = Length[sp], i},
              n = Length[s];
              Sum[StirlingSecond[n, i], {i, 1, k-1}] + RankKSetPartition [sp, s]
       ]
RankSetPartition[sp_List, s_List] :=
       Module[{n = Length[s], k = Length[sp],i},
              Sum[StirlingSecond[n, i], {i, 1, k-1}] + RankKSetPartition [sp, s]
       ] /; SetPartitionQ[sp, s]
(* following are utility functions used in RankSubset *)
(* nthsubgroup indexes into Subset's sort order; for set size m, subset
   size x starts here. Defined by Sum[Binomial[m, n], {n, 1, x - 1}] - 1 *)
nthsubgroup[m_, x_] := 
    2^m - Binomial[m, x]*Hypergeometric2F1[x - m, 1, 1 + x, -1]
(* setindices gives the indices of elements of a subset of a set. Note
   that this version is limited to sets of unique elements; my only
   implementation so far for non-unique sets is much too slow. *)
setindices[set_, subs_] :=
    Map[If[# === {}, 0, First[#]] &,
        Position[set, Alternatives @@ subs, {1}]
    ]
RankSubset[set_List, {}] := 0
RankSubset[set_List,subset_List] :=
    Module[{order = setindices[set, subset]},
        If[! FreeQ[order, 0], Return[0]];
        nthsubgroup[Length[set], Length[subset]] + 
            RankKSubset[order, Range[Length[set]]]
    ]
RankedEmbedding[stages_List] := 
        Module[{m, rank, stageSizes, freq = Table[0, {Length[stages]}],i}, 
               rank = Table[Position[stages,i][[1,1]], {i, Max[stages]}];
               stageSizes = Distribution[rank]; 
               Table[m = ++freq[[rank[[i]]]];
                     {rank[[i]], (m-1)+(1 - stageSizes[[rank[[i]]]])/2} // N, 
                     {i, Max[stages]}
               ]
        ] 
RankedEmbedding[g_Combinatorica`Graph, stages_List] := 
        ChangeVertices[g, RankedEmbedding[stages]] /; SetPartitionQ[stages, Range[ V[g] ]]
RankedEmbedding[g_Combinatorica`Graph, start:{_Integer?Positive..}] := 
        Module[{l = RankGraph[g, start],i},
               RankedEmbedding[g, Table[Flatten[Position[l, i]], {i, Max[l]}]] 
        ] /; (Max[start] <= V[g])
RankedVertices[g_Combinatorica`Graph,start_List] :=
	Module[{i,m,stages,rank,freq = Table[0,{V[g]}]},
		rank = RankGraph[g,start];
		stages = Distribution[ rank ];
		Table[
			m = ++ freq[[ rank[[i]] ]];
			{{rank[[i]], (m-1) + (1 - stages[[ rank[[i]] ]])/2 }}//N,
			{i,V[g]}
		]
	]
ReachableVertices[g_Combinatorica`Graph, start_List, ME_List] := 
       Module[{r = AlternatingPaths[g, start, ME]}, 
              Join[start, Flatten[Position[Range[V[g]]-r, _Integer?(# != 0 &)]]]
       ]
ReadGraph[file_] :=
        Module[{edgelist={}, v={},x},
                If[Head[OpenRead[file]] =!= InputStream,
                   EmptyGraph[0], 
                   While[!SameQ[(x = Read[file,Number]), EndOfFile],
                         AppendTo[v,Read[file,{{Number,Number}}]];
                         AppendTo[edgelist,
                                  Convert[Characters[Read[file,String]]]
                         ];
                   ];
                   Close[file];
                   FromAdjacencyLists[edgelist,v]
                ]
        ]
RealizeDegreeSequence[d_List] :=
	Module[{i,j,k,v,set,seq,n=Length[d],e},
		seq = Reverse[ Sort[ Table[{d[[i]],i},{i,n}]] ];
		Combinatorica`Graph[
			Flatten[ Table[
				{{k,v},seq} = {First[seq],Rest[seq]};
				While[ !GraphicQ[
					MapAt[
						(# - 1)&,
						Map[First,seq],
						set = RandomKSubset[Table[{i},{i,n-j}],k] 
					] ]
				];
				e = Map[{Sort[(Prepend[seq[[#,2]],v])]}&,set];
				seq = Reverse[ Sort[
					MapAt[({#[[1]]-1,#[[2]]})&,seq,set]
				] ];
				e,
				{j,Length[d]-1}
			], 1],
			CircularEmbedding[n]
		]
	] /; GraphicQ[d]
RealizeDegreeSequence[d_List,seed_Integer] :=
	(SeedRandom[seed]; RealizeDegreeSequence[d])
 
RefineEquivalences[eq_List, g_Combinatorica`Graph, h_Combinatorica`Graph, f_] := 
        Module[{dg = Table[Apply[f, {g, i}], {i, V[g]}], 
                dh = Table[Apply[f, {h, i}], {i, V[h]}], eq1}, 
               eq1 = Table[Flatten[Position[dh, dg[[i]]], 1], {i, Length[dg]}]; 
               Table[Intersection[eq[[i]], eq1[[i]]], {i, Length[eq]}]
        ]
ReflexiveQ[r_?squareMatrixQ] := 
	Module[{i}, Apply[And, Table[(r[[i,i]]!=0), {i, Length[r]}] ] ]
ReflexiveQ[g_Combinatorica`Graph] := False /; (V[g] == 0)
ReflexiveQ[g_Combinatorica`Graph] := 
	Module[{e=Edges[g],i},
		Apply[And, Table[MemberQ[e,{i,i}],{i, V[g]}] ]
	]
RegularGraph[k_Integer, n_Integer] := RealizeDegreeSequence[Table[k,{n}]]
RegularQ[g_Combinatorica`Graph] := Apply[ Equal, Degrees[g] ]
RemoveCycleEdges[g_Combinatorica`Graph, c_List] := 
        ChangeEdges[g, Select[Edges[g], (Intersection[#, c] == {}) &]]
RemoveMultipleEdges[g_Combinatorica`Graph] := rme[g, First]
RemoveMultipleEdges[g_Combinatorica`Graph, True] := rme[g, mergeedges]
RemoveMultipleEdges[g_Combinatorica`Graph, False] := rme[g, First]
RemoveMultipleEdges[g_Combinatorica`Graph, f_] := rme[g, f]
rme[g_Combinatorica`Graph, mergefunc_] :=
        ChangeEdges[g,
                       Map[mergefunc,
                           Split[
                                 Sort[Edges[g,All], 
                                      OrderedQ[{First[#1], First[#2]}]&
                                 ],
                                 (First[#1] == First[#2])&
                           ]
                       ]
        ]
mergeedges[e_] :=
     {e[[1,1]], Combinatorica`EdgeWeight -> Plus @@ Map[(Combinatorica`EdgeWeight/.Flatten[{Rest[#], Combinatorica`EdgeWeight->1}]) &, e], Sequence @@ DeleteCases[Rest[First[e]], Combinatorica`EdgeWeight->_]}
RemoveSelfLoops[g_Combinatorica`Graph] :=
        ChangeEdges[g, Select[Edges[g, All], (First[#][[1]] != First[#][[2]])& ]]
Options[RenderEdges] := Options[ShowGraph,
    {Combinatorica`EdgeColor, Combinatorica`EdgeStyle, Combinatorica`EdgeLabel, EdgeLabelColor, 
     EdgeLabelPosition, LoopPosition, EdgeDirection}]
RenderEdges[v_List, e_List, aopts_List, eopts_List] :=
        Module[{i,fvp, svp,
                ne = RenderMultipleEdges[e, Cases[eopts, _[EdgeDirection,_]][[1,2]]]
               },
               Table[({fvp, svp} = v[[First[ ne[[i]] ]]];
                     ExpandEdgeOptions[Combinatorica`Private`Merge[Flatten[Map[selectval[#, i]&, eopts]], Rest[ ne[[i]] ]], i, fvp, svp, aopts]),
                     {i, Length[ne]}                  
               ]
        ]
RenderGraph[Combinatorica`Graph[e_, v_, gopts___], PlotRange -> pr_, ropts___] :=
        Block[{defaults = Combinatorica`Private`Merge[Options[ShowGraph], {gopts}],
               nv = NormalizeVertices[Map[First[#]&, v]]},
               Graphics[{RenderVertices[nv, defaults],
                         RenderEdges[Map[First[#]&, nv], e, defaults]},
                         ExpandOptions[PlotRange -> pr, nv, defaults],
                         ropts
               ]
        ]
RenderMultipleEdges[e_List, flag_] :=
        Module[{ord = Ordering[e, All, OrderedQ[{First[#1], First[#2]}]&],
                se, r, nf,p,i},
               se = Split[e[[ord]], (First[#1] == First[#2])&];
               Flatten[Map[If[(Length[#]==1),
                              #,
                              r = Join[ Range[ Floor[Length[#]/2] ],
                                        -Range[ Floor[Length[#]/2] ],
                                         If[OddQ[Length[#]], {0}, {}]
                                  ];
                              Table[p = Position[#[[i]], EdgeDirection->_];
                                    nf = flag;
                                    If[p != {}, nf = #[[i, p[[1,1]], 2]] ];
                                    Prepend[Combinatorica`Private`Merge[Rest[#[[i]]],
                                          {EdgeDirection ->{nf, r[[i]]}}
                                    ], First[#[[i]]]],
                                    {i, Length[#]}
                              ]
                           ]&,
                           se
                       ], 1
               ][[Ordering[ord]]]
        ]
Options[RenderVertices] := Options[ShowGraph,
         {VertexColor, Combinatorica`VertexStyle, VertexNumber, VertexNumberColor,
          VertexNumberPosition, Combinatorica`VertexLabel, VertexLabelColor, VertexLabelPosition}]
RenderVertices[v_List, opts_List] :=
        Table[ExpandVertexOptions[Combinatorica`Private`Merge[Flatten[Map[selectval[#, i]&, opts]], Rest[v[[i]]]], First[v[[i]]], i],
              {i, Length[v]}
        ]
(* distribution of values for list options varies a bit *)
(* EdgeDirection is a special case, List not allowed (1 value per graph) *)
selectval[EdgeDirection -> _List, _] := EdgeDirection -> None
(* Empty list is a forced 'no' default for some, standard default for others *)
selectval[(a:(VertexNumber | Combinatorica`VertexLabel | Combinatorica`EdgeLabel)) -> {}, _] := a -> False
selectval[any_ -> {}, _] :=
    FilterRules[{Options[RenderVertices], Options[RenderEdges]}, any]
(* certain options take lists, only list-of-lists is mapped *)
selectval[(n:(VertexLabelPosition | VertexNumberPosition | EdgeLabelPosition)) -> a_?MatrixQ, i_] :=
     n-> a[[Mod[i - 1, Length[a]] + 1]]
selectval[(n:(VertexLabelPosition | VertexNumberPosition | EdgeLabelPosition)) -> a_List, ___] :=
     n -> a
(* all other options map across lists (modulo length) *)
selectval[any_ -> a_List, i_] := any -> a[[Mod[i - 1, Length[a]] + 1]]
selectval[any:(_ -> _), ___] := any
ResidualFlowGraph[g_Combinatorica`Graph, f_List] := ResidualFlowGraph[g,ToAdjacencyLists[g,Combinatorica`EdgeWeight], f]
ResidualFlowGraph[g_Combinatorica`Graph, al_List, f_List] := 
        Module[{r, e, i, j},
               e = Flatten[
                       Table[Join[
                                  If[(r = f[[i,j,2]])>0, {{{al[[i,j,1]],i}, r}},{}],
                                  If[(r = al[[i,j,2]]-f[[i,j,2]])>0, {{{i,al[[i,j,1]]}, r}},{}]
                             ],
                             {i, Length[f]}, {j, Length[f[[i]]]}
                       ], 2 
                   ];
               e = Map[{#[[1, 1]], Combinatorica`EdgeWeight->Apply[Plus, Transpose[#][[2]]]}&,
                       Split[Sort[e], First[#1] == First[#2]&]
                   ];
               SetGraphOptions[ChangeEdges[g, e], EdgeDirection -> True]
        ]
RevealCycles[p_?Combinatorica`PermutationQ] := 
      Module[{m = Infinity, i},
             Map[Take[p, {#[[1]], #[[2]] - 1}]&, 
                 Partition[
                    Join[DeleteCases[
                             Table[If[ p[[i]] < m, m = p[[i]]; i, 0], 
                                   {i, Length[p]}
                             ], 
                             0
                         ],  
                         {Length[p] + 1}
                    ], 2, 1
                 ]
             ]
      ]
ReverseEdges[g_Combinatorica`Graph] := 
        ChangeEdges[g,
                       Map[Prepend[Rest[#], Reverse[First[#]]]&,
                           Edges[g,All]
                       ]
        ] /; !UndirectedQ[g]
ReverseEdges[g_Combinatorica`Graph] := g
RobertsonGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{2, 3}}, {{3, 4}}, {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 8}}, 
  {{8, 9}}, {{9, 10}}, {{10, 11}}, {{11, 12}}, {{12, 13}}, {{13, 14}}, 
  {{14, 15}}, {{15, 16}}, {{16, 17}}, {{17, 18}}, {{18, 19}}, {{1, 19}}, 
  {{1, 5}}, {{5, 10}}, {{10, 14}}, {{14, 18}}, {{3, 18}}, {{3, 7}}, 
  {{7, 11}}, {{11, 16}}, {{1, 16}}, {{2, 9}}, {{9, 17}}, {{6, 17}}, 
  {{6, 13}}, {{2, 13}}, {{8, 19}}, {{8, 15}}, {{4, 15}}, {{4, 12}}, 
  {{12, 19}}}, {{{0.9458172417006346, 0.32469946920468346}}, 
  {{0.7891405093963936, 0.6142127126896678}}, 
  {{0.546948158122427, 0.8371664782625285}}, 
  {{0.24548548714079924, 0.9694002659393304}}, 
  {{-0.08257934547233227, 0.9965844930066698}}, 
  {{-0.40169542465296926, 0.9157733266550575}}, 
  {{-0.6772815716257409, 0.7357239106731318}}, 
  {{-0.879473751206489, 0.4759473930370737}}, 
  {{-0.9863613034027223, 0.16459459028073403}}, 
  {{-0.9863613034027224, -0.16459459028073378}}, 
  {{-0.8794737512064891, -0.4759473930370735}}, 
  {{-0.6772815716257414, -0.7357239106731313}}, 
  {{-0.40169542465296987, -0.9157733266550573}}, 
  {{-0.08257934547233274, -0.9965844930066698}}, 
  {{0.2454854871407988, -0.9694002659393305}}, 
  {{0.5469481581224266, -0.8371664782625288}}, 
  {{0.7891405093963934, -0.614212712689668}}, 
  {{0.9458172417006346, -0.32469946920468373}}, {{1., 0}}}]
RootedEmbedding[g_Combinatorica`Graph] := RootedEmbedding[g, First[Combinatorica`GraphCenter[g]]]
RootedEmbedding[g_Combinatorica`Graph,rt_Integer] :=
	Module[{root=rt,pos,i,x,dx,new,n=V[g],v,done,next,
                e=ToAdjacencyLists[g]},
		pos = Table[{-Ceiling[Sqrt[n]],Ceiling[Sqrt[n]]},{n}];
		v = Table[{0,0},{n}];
		next = done = {root};
		While [next != {},
			root = First[next];
			new = Complement[e[[root]], done];
			Do [
				dx = (pos[[root,2]]-pos[[root,1]])/Length[new];
				pos[[ new[[i]] ]] = {pos[[root,1]] + (i-1)*dx,
					pos[[root,1]] + i*dx};
				x = Apply[Plus,pos[[ new[[i]] ]] ]/2;
				v[[ new[[i]] ]] = {x,v[[root,2]]-1},
				{i,Length[new]}
			];
			next = Join[Rest[next],new];
			done = Join[done,new]
		];
		ChangeVertices[g,Map[{#}&, N[v]]]
	]
RotateVertices[v:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}, t_] := 
        Module[{p = Map[First, v], np, i},
               np = RotateVertices[p, t];
               Table[{np[[i]], Apply[Sequence, Rest[v[[i]]]]}, {i, Length[np]}]
        ]
RotateVertices[v:{{_?NumericQ, _?NumericQ}...}, t_] := 
	Module[{d, theta},
		Map[
			(If[# == {0,0}, 
                            #, 
                            d=Sqrt[#[[1]]^2 + #[[2]]^2];
                            theta = t + Arctan[#];
                            If[!FreeQ[theta, Complex],
                               (* hack to get around new ArcTan behavior *)
                               theta = Chop[theta]
                            ];
                            N[{d Cos[theta], d Sin[theta]}]
			 ])&,
			v
		]
	]
RotateVertices[g_Combinatorica`Graph, t_] := 
        ChangeVertices[g, RotateVertices[Vertices[g, All], t]]
RotateVertices[g_Combinatorica`Graph, s_List, t_] :=
        Module[{v = Vertices[g, All]},
               ChangeVertices[g, v[[s]] = RotateVertices[v[[s]], t]; v]
        ]
Runs[p_?Combinatorica`PermutationQ] :=
	Map[
		(Apply[Take,{p,{#[[1]]+1,#[[2]]}}])&,
		Partition[
			Join[
				{0},
				Select[Range[Length[p]-1], (p[[#]]>p[[#+1]])&],
				{Length[p]}
			],
			2,
			1
		]
	] /; (Length[p] > 0)
Runs[{}] := {}
SamenessRelation[perms_List] :=
	Module[{positions = Transpose[perms], i, j, n=Length[First[perms]]},
		Table[
			If[ MemberQ[positions[[i]],j], 1, 0],
			{i,n}, {j,n}
		]
	] /; perms != {}
ScreenColorNames = {"Black", "Red", "Blue", "Green", "Yellow", "Purple", 
                    "Brown", "Orange", "Olive", "Pink", "DeepPink", 
                    "DarkGreen", "Maroon", "Navy"}
ScreenColors =
    {GrayLevel[0], RGBColor[1, 0, 0], RGBColor[0, 0, 1], RGBColor[0, 1, 0],
     RGBColor[1, 1, 0], RGBColor[0.5, 0, 0.5], RGBColor[0.6, 0.4, 0.2],
     RGBColor[1, 0.5, 0], RGBColor[0.230003, 0.370006, 0.170003], 
     RGBColor[1, 0.5, 0.5], RGBColor[1., 0.078402, 0.576495],
     RGBColor[0., 0.392193, 0.], RGBColor[0.690207, 0.188192, 0.376507],
     RGBColor[0., 0., 0.501999]};
(* note that many of the variables in the following belong to the calling function *)
SearchBiConComp[v_Integer] :=
	Block[{r, $RecursionLimit = Infinity},
              back[[v]]=dfs[[v]]=++c;
              Scan[
                   (If[dfs[[#]] == 0, 
                       If[!MemberQ[act,{v,#}], PrependTo[act,{v,#}]];
                       par[[#]] = v;
                       SearchBiConComp[#];
                       If[back[[#]] >= dfs[[v]],
                          {r} = Flatten[Position[act,{v,#}]];
                          AppendTo[bcc,Union[Flatten[Take[act,r]]]];
                          AppendTo[ap,v];
                          act = Drop[act,r]
                       ];
                       back[[v]] = Min[ back[[v]],back[[#]] ],
                       If[# != par[[v]],back[[v]]=Min[dfs[[#]],back[[v]]]]
                    ])&,
                    e[[v]]
              ];
        ]
SearchStrongComp[v_Integer] :=
	Block[{r, $RecursionLimit = Infinity},
		low[[v]]=dfs[[v]]=c++;
		PrependTo[cur,v];
		Scan[
			(If[dfs[[#]] == 0,
				SearchStrongComp[#];
				low[[v]]=Min[low[[v]],low[[#]]],
				If[(dfs[[#]] < dfs[[v]]) && MemberQ[cur,#],
					low[[v]]=Min[low[[v]],dfs[[#]] ]
				];
			])&,
			e[[v]]
		];
		If[low[[v]] == dfs[[v]],
			{r} = Flatten[Position[cur,v]];
			AppendTo[scc,Take[cur,r]];
			cur = Drop[cur,r];
		];
	]
SelectionSort[l_List,f_] :=
	Module[{where,item,unsorted=l},
		Table[
			item = MinOp[unsorted, f];
			{where} = First[ Position[unsorted,item] ];
			unsorted = Drop[unsorted,{where,where}];
			item,
			{Length[l]}
		]
	]
SelfComplementaryQ[g_Combinatorica`Graph] := IsomorphicQ[g, Combinatorica`GraphComplement[g]]
SelfLoopsQ[g_Combinatorica`Graph] := MemberQ[Edges[g], {x_, x_}]
SetEdgeLabels[g_Combinatorica`Graph, labels_List] :=
         Module[{el = Edges[g],i},
                 SetGraphOptions[g, Table[{el[[i]], 
                                           Combinatorica`EdgeLabel -> labels[[Mod[i-1,Length[labels]]+1]]
                                          },
                                          {i, M[g]}
                                    ]
                 ]
         ]
Options[SetEdgeWeights] = {WeightingFunction -> Random, WeightRange -> {0, 1}}
SetEdgeWeights[g_Combinatorica`Graph, e : {{_Integer, _Integer} ...}, opts___?OptionQ] := 
        Module[{ v = Vertices[g], myfn, myrange}, 
               {myfn, myrange} = {WeightingFunction, WeightRange} /. 
               Flatten[{opts, Options[SetEdgeWeights]}];
               Switch[myfn, 
                      Random,
                      SetGraphOptions[g, 
                          Map[{#, Combinatorica`EdgeWeight->RandomReal[myrange]} &, e]
                      ],
                      RandomInteger,
                      SetGraphOptions[g, 
                          Map[{#, Combinatorica`EdgeWeight->RandomInteger[myrange]}&,e]
                      ],
                      Euclidean | LNorm[_],
                      SetGraphOptions[g, 
                          Map[{#, Combinatorica`EdgeWeight->Distance[v[[#]], myfn]} &, e]
                      ],
                      _,
                      SetGraphOptions[g, 
                          Map[{#, 
                               Combinatorica`EdgeWeight->Apply[myfn,Transpose[{#, v[[#]]}]]}&,
                               e
                          ]
                      ] 
               ]
        ]
  
SetEdgeWeights[g_Combinatorica`Graph, opts___?OptionQ] := SetEdgeWeights[g, Edges[g], opts]
SetEdgeWeights[g_Combinatorica`Graph, e:{{_Integer, _Integer}...}, weights:{_?NumericQ...}] := 
        SetGraphOptions[g, 
            MapIndexed[{#1, Combinatorica`EdgeWeight->weights[[First[#2]]]}&, 
                       If[UndirectedQ[g], Map[Sort, e], e]
            ]
        ] /; (Length[weights] == Length[e])
SetEdgeWeights[g_Combinatorica`Graph, weights : {_?NumericQ...}] := 
        SetEdgeWeights[g, Edges[g], weights]
SetGraphOptions[{}, {}] := {}
SetGraphOptions[l_List, {}] := l
SetGraphOptions[l_List, {}, _] := l
SetGraphOptions[vl_List, ol : {{_Integer.., __?OptionQ}..}] := 
        Module[{o=Transpose[ol /. {x__Integer, y__?OptionQ} :> {{x}, {y}}], p, i},
               Table[p = Position[o[[1]], i]; 
                     If[p == {},
                        vl[[i]],
                        Prepend[Combinatorica`Private`Merge[Rest[vl[[i]]], o[[2, p[[1, 1]] ]] ], 
                                First[ vl[[i]] ] 
                        ]
                     ],
                     {i, Length[vl]}
               ]
        ]
SetGraphOptions[el_List, ol_List, All] := 
        Module[{o=Transpose[ol /. {x : {_Integer, _Integer}.., y__?OptionQ} :> 
                                  {{x}, {y}}
                  ], p, i},
               Table[p = Position[o[[1]], First[ el[[i]] ] ]; 
                     If[p == {}, 
                        el[[i]], 
                        Prepend[Combinatorica`Private`Merge[Rest[el[[i]]], o[[2, p[[1, 1]]  ]]  ], 
                                First[ el[[ i ]] ]
                        ]
                     ], 
                     {i, Length[el]}
               ]
        ]
SetGraphOptions[el_List, ol_List, One] := 
        Module[{no=Transpose[ol /. {x:{_Integer, _Integer} .., y__?OptionQ} :> 
                                   {{x}, {y}}
                   ], p, e, i},
               Table[p = Position[no[[1]], First[ el[[i]] ] ]; 
                     If[p == {}, 
                        el[[i]], 
                        e = Prepend[Combinatorica`Private`Merge[Rest[el[[i]]], no[[2, p[[1, 1]] ]] ], 
                                    First[ el[[i]] ]
                        ];
                        no = MapAt[Infinity &, no, p];
                        e
                     ], 
                     {i, Length[el]}
               ]
        ]
SetGraphOptions[g_Combinatorica`Graph, l:{_Integer.., ___?OptionQ}, 
                flag_Symbol:All, opts___?OptionQ] := 
        SetGraphOptions[g, {l}, flag, opts]
SetGraphOptions[g_Combinatorica`Graph, l:{{_Integer,_Integer}.., ___?OptionQ}, 
                flag_Symbol:All, opts___?OptionQ] := 
        SetGraphOptions[g, {l}, flag, opts]
SetGraphOptions[Combinatorica`Graph[e_, v_, dopts___], l_List:{}, flag_Symbol:All, opts___?OptionQ] :=
        Module[{lv, le, ne, nv, dopt},
	      dopt = EdgeDirection/.Flatten[{opts}]/.EdgeDirection->False;
              lv = Cases[l, {_Integer.., __?OptionQ}];
              If[UndirectedQ[Combinatorica`Graph[e, v, dopts]] && !dopt,
	         ne = {Sort[#1], ##2} & @@@ e;
                 le = Cases[l, 
                            {{_Integer, _Integer}.., __?OptionQ}
                      ] /. {x : {_Integer, _Integer}.., y___?OptionQ} :> 
                           {Apply[Sequence, Map[Sort[#] &, {x}]], y},
	         ne = e;
                 le = Cases[l, {{_Integer, _Integer}.., __?OptionQ}]
              ];
              If[flag===One, 
                 ne = SetGraphOptions[ne, le, One],
                 ne = SetGraphOptions[ne, le, All]
              ];
              nv = SetGraphOptions[v, lv];
              Apply[Combinatorica`Graph, Join[{ne, nv}, Combinatorica`Private`Merge[{dopts}, {opts}]]]
        ]
SetLevel[l_List,lvl_,rank_List] :=  
    Module[ {r=rank},
            If[ r[[#]] < lvl, r[[#]] = lvl ] & /@ l;
            r
    ]
SetPartitionListViaRGF[n_Integer?Positive] := 
                Map[RGFToSetPartition, RGFs[n]]
SetPartitionListViaRGF[n_Integer?Positive, k_Integer?Positive] :=
                Map[RGFToSetPartition, RGFs[n, k]]
SetPartitionQ[sp_] := (ListQ[sp]) && (Depth[sp] > 2) && SetPartitionQ[sp, Apply[Union, sp]]
SetPartitionQ[sp_, s_List] := (ListQ[sp]) && (Depth[sp] > 2) &&
                              (Apply[And, Map[ListQ, sp]]) && (Sort[Flatten[sp, 1]] === Sort[s])
SetPartitionToLabel[s_?SetPartitionQ] := 
       StringDrop[Apply[StringJoin, 
                        Map[StringJoin[Apply[StringJoin, Map[ToString, #]], "|"] &, s]
                  ], -1
       ]
SetPartitionToRGF[{{}}] := {}
SetPartitionToRGF[sp_?SetPartitionQ] := 
       SetPartitionToRGF[sp, Sort[Flatten[sp, 1]]]
SetPartitionToRGF[sp_?SetPartitionQ, set_List] :=
       Module[{i, rgf = Table[1, {Length[set]}], 
               nsp = ToCanonicalSetPartition[sp, set]
              },
              Table[rgf[[ Map[Position[set, #][[1, 1]]&, nsp[[i]] ] ]] = i,
                    {i, Length[sp]}
              ];
              rgf
       ]
SetPartitions[{}] := {{}}
SetPartitions[s_List] := Flatten[Table[KSetPartitions[s, i], {i, Length[s]}], 1]
SetPartitions[0] := {{}}
SetPartitions[n_Integer?Positive] := SetPartitions[Range[n]]
SetVertexLabels[g_Combinatorica`Graph, labels_List] :=
         SetGraphOptions[g, Table[{i, 
                                   Combinatorica`VertexLabel-> labels[[ Mod[i-1,Length[labels]]+1 ]]
                                  },
                                  {i, V[g]}
                            ]
         ]
Options[SetVertexWeights] = {WeightingFunction -> Random, 
                             WeightRange -> {0, 1} }
SetVertexWeights[g_Combinatorica`Graph, opts___?OptionQ] :=
        Module[{v = Vertices[g], myfn, myrange, i},
                {myfn, myrange} = {WeightingFunction, WeightRange} /.
                Flatten[{opts, Options[SetVertexWeights]}];
               Switch[myfn,
                      Random,
                      SetGraphOptions[g,
                          Table[{i, Combinatorica`VertexWeight->RandomReal[myrange]}, 
                                {i, V[g]}
                          ]
                      ],
                      RandomInteger,
                      SetGraphOptions[g,
                          Table[{i, Combinatorica`VertexWeight->RandomInteger[myrange]}, 
                                {i, V[g]}
                          ]
                      ],
                      _,
                      SetGraphOptions[g,
                          Table[{i, Combinatorica`VertexWeight -> Apply[myfn, {i, v[[i]]}]},
                                {i, V[g]}
                          ]
                      ]
               ]
        ]
SetVertexWeights[g_Combinatorica`Graph, vs:{_Integer ...}, weights:{_?NumericQ ...}] :=
        SetGraphOptions[g,
            MapIndexed[{#1, Combinatorica`VertexWeight->weights[[First[#2]]]}&, vs]
        ] /; (Length[weights] == Length[vs])
SetVertexWeights[g_Combinatorica`Graph, weights : {_?NumericQ ...}] :=
        SetVertexWeights[g, Range[V[g]], weights]
ShakeGraph[g_Combinatorica`Graph, s_List] := ShakeGraph[g, s, 0.1]
ShakeGraph[g_Combinatorica`Graph] := ShakeGraph[g, 0.1]
ShakeGraph[g_Combinatorica`Graph, s_List, fract_?NumericQ] :=
        Module[{i, d, a, v = Vertices[g, All]},
               v[[s]] = Map[(d = RandomReal[fract];
                             a = RandomReal[2 N[Pi]];
                             Prepend[Rest[#], 
                                     First[#] + {N[d Cos[a]], N[d Sin[a]]}
                             ])&,
                             v[[s]]
                        ];
               ChangeVertices[g, v]
        ]
ShakeGraph[g_Combinatorica`Graph, fract_?NumericQ] := ShakeGraph[g, Range[V[g]], fract]
ShapeOfTableau[t_List] := Map[Length,t]
Options[ShortestPath] = {Algorithm -> Automatic};
ShortestPath[g_Combinatorica`Graph, s_Integer, e_Integer, opts___?OptionQ] := 
       Module[{algorithm, parent}, 
              algorithm = Algorithm /. Flatten[{opts, Options[ShortestPath]}];
              parent = ChooseShortestPathAlgorithm[g, s, algorithm];
              Rest[Reverse[FixedPointList[Function[x, parent[[x]]], e, V[g]] ]]
       ] /; (1 <= s) && (s <= V[g]) && (1 <= e) && (e <= V[g])
Options[ShortestPathSpanningTree] = {Algorithm -> Automatic}
ShortestPathSpanningTree[g_Combinatorica`Graph, s_Integer, opts___?OptionQ] :=
	Module[{algorithm, parent},
               algorithm = Algorithm /. Flatten[{opts, Options[ShortestPathSpanningTree]}];
               parent = ChooseShortestPathAlgorithm[g, s, algorithm];
               Combinatorica`Graph[Map[({Sort[{#,parent[[#]]}]})&, 
                            Complement[Range[V[g]],{s}]
                     ],
                     Vertices[g, All]
	       ]
	]
ShowGraph::obsolete = "Usage of Directed as a second argument to ShowGraph is obsolete."
Options[ShowGraph] = Sort[
     $GraphVertexStyleOptions ~Join~
     $GraphEdgeStyleOptions ~Join~
     Developer`GraphicsOptions[]
     ];
SetOptions[ShowGraph, AspectRatio -> Automatic, PlotRange -> Normal]
SelectOptions[options_List, name_] :=
        Select[Flatten[options], (MemberQ[Options[name], First[#], 2])&]
ShowGraph[g_Combinatorica`Graph, Directed] := (Message[ShowGraph::obsolete]; ShowGraph[g])
ShowGraph[g_Combinatorica`Graph, lopts_List, opts___?OptionQ] := 
        ShowGraph[SetGraphOptions[g, lopts], opts] /; (V[g] > 0)
ShowGraph[g_Combinatorica`Graph, opts___?OptionQ]/;V[g] === 0 :=
    Graphics[{}, FilterRules[{opts}, Options[Graphics]]]
ShowGraph[g_Combinatorica`Graph, opts___?OptionQ] :=
        Module[{i, VertexOptions = Combinatorica`Private`Merge[Options[RenderVertices],
                                      SelectOptions[{opts}, RenderVertices],
                                      SelectOptions[Options[g], RenderVertices]
                                ],
                EdgeOptions  =  Combinatorica`Private`Merge[Options[RenderEdges],
                                      SelectOptions[{opts}, RenderEdges],
                                      SelectOptions[Options[g], RenderEdges]
                                ],
                ArrowOptions =  Combinatorica`Private`Merge[SelectOptions[{opts}, Arrow],
                                      SelectOptions[Options[g], Arrow]
                                ],
                PlotOptions =   Combinatorica`Private`Merge[FilterRules[Options[ShowGraph],Options[Graphics]],
                                      SelectOptions[{opts}, Graphics],
                                      SelectOptions[Options[g], Graphics]
                                ],
                v = Vertices[g, All],
                nv = NormalizeVertices[Vertices[g]], nnv
               },
               nnv = Table[Prepend[Rest[v[[i]]], nv[[i]] ], {i, Length[v]}];
               Graphics[{RenderEdges[nv, Edges[g, All], ArrowOptions, EdgeOptions],
                         RenderVertices[nnv, VertexOptions]},
                         Apply[Sequence, Map[ExpandPlotOptions[#, nnv]&, PlotOptions]]
               ]
        ] /; (V[g] > 0)
Options[ShowGraphArray] = Sort[
    $GraphVertexStyleOptions ~Join~
    $GraphEdgeStyleOptions ~Join~
    Options[GraphicsGrid] (* includes graphics options *)
    ];
ShowGraphArray[gs_List, opts___] := 
        ShowGraphArray[{gs}, opts] /; (Head[First[gs]] == Combinatorica`Graph)
ShowGraphArray[gs_List, opts___] := 
        Block[{s, nopts},
              s = FilterRules[{opts, Spacings -> Scaled[0.1],
	                       Complement[Options[ShowGraphArray],
		                          Options[GraphicsGrid]]},
			      Options[GraphicsGrid]];
              nopts = FilterRules[{opts}, Options[ShowGraph]];
	    (* only pass through options unique to ShowGraph *)
	      nopts = Complement[nopts, s];
              Show[Apply[GraphicsGrid, 
                   {Map[ShowGraph[#, Sequence @@ nopts]&, gs, {2}], Sequence @@ s}
              ]]
        ]
Options[ShowLabeledGraph] = DeleteCases[Options[ShowGraph], VertexNumber -> _];
ShowLabeledGraph[g_Combinatorica`Graph, o___?OptionQ] := ShowGraph[g, VertexNumber -> True, o]
ShowLabeledGraph[g_Combinatorica`Graph, l_List, o___?OptionQ] := ShowGraph[ SetVertexLabels[g, l], o]
Options[ShuffleExchangeGraph] = {Combinatorica`VertexLabel -> False}
ShuffleExchangeGraph[n_Integer?Positive, opts___?OptionQ] := 
        Module[{label},
               label = Combinatorica`VertexLabel /. Flatten[{opts, Options[MakeGraph]}];
               MakeGraph[Strings[{0, 1}, n],
                        (Last[#1] != Last[#2]) && (Take[#1,(n-1)]==Take[#2,(n-1)]) || 
                        (RotateRight[#1,1] == #2) || (RotateLeft[#1, 1] == #2)&,
                        Type -> Undirected,
                        Combinatorica`VertexLabel -> label
               ]
        ]
SignaturePermutation[p_?Combinatorica`PermutationQ] := (-1) ^ (Length[p]-Length[Combinatorica`ToCycles[p]])
SimplePlanarQ[g_Combinatorica`Graph] :=
  Block[{components, $RecursionLimit = Infinity},
        components = BiconnectedComponents[g];
        Apply[  And,
                Map[(PlanarQ[InduceSubgraph[g,#]])&, components]
        ]
  ] /; !(ConnectedQ[g] && BiconnectedQ[g])
SimplePlanarQ[g_Combinatorica`Graph] := False /;  (M[g] > 3 V[g]-6) && (V[g] > 2)
SimplePlanarQ[g_Combinatorica`Graph] := True /;   (M[g] < V[g] + 3)
SimplePlanarQ[g_Combinatorica`Graph] := (PlanarGivenCycleQ[ g, Rest[Combinatorica`FindCycle[g]] ])
SimpleQ[g_Combinatorica`Graph] := (!PseudographQ[g]) && (!MultipleEdgesQ[g])
SingleBridgeQ[b_Combinatorica`Graph, {_}] := PlanarQ[b]
SingleBridgeQ[b_?PathQ, {_,_}] := True
SingleBridgeQ[b_Combinatorica`Graph, j_List] :=
    Module[{sp = ShortestPath[b, j[[1]], j[[2]]]},
           If[Intersection[sp, Drop[j, 2]] =!= {},
              SingleBridgeQ[b, RotateLeft[j]],
              PlanarGivenCycleQ[JoinCycle[b,j], Join[sp, Drop[j,2]]]
           ]
    ]
SmallestCyclicGroupGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{1, 3}}, {{1, 4}}, {{1, 5}}, {{1, 6}}, {{2, 3}}, {{2, 4}}, 
  {{2, 8}}, {{2, 9}}, {{3, 6}}, {{3, 7}}, {{3, 8}}, {{4, 9}}, {{5, 6}}, 
  {{7, 8}}}, {{{0.508, 0.81}}, {{0.302, 0.44}}, {{0.704, 0.44}}, 
  {{0.426, 0.852}}, {{0.592, 0.854}}, {{0.786, 0.49}}, {{0.704, 0.344}}, 
  {{0.302, 0.344}}, {{0.23, 0.484}}}]
Solution[space_List,index_List,count_Integer] :=
	Module[{i}, Table[space[[ i,index[[i]] ]], {i,count}] ]
Spectrum[g_Combinatorica`Graph]/;EmptyQ[g] := Table[0, {V[g]}]
Spectrum[g_Combinatorica`Graph] := Eigenvalues[ToAdjacencyMatrix[g]]
SpringEmbedding[g_Combinatorica`Graph, step_:10, inc_:0.15] := g /; EmptyQ[g]
SpringEmbedding[g_Combinatorica`Graph, step_:10, inc_:0.15] := 
       Module[{verts=Vertices[g], new, m=ToAdjacencyMatrix[MakeUndirected[g]]},
              new = UV[step, inc, m, verts];
              ChangeVertices[g, new]
       ]
UV = Compile[{{step, _Real}, {inc, _Real}, {m, _Integer, 2}, 
              {verts, _Real, 2}},
       Module[{u, i, new = verts, old = verts, n = Length[verts]},
              Do[ Do[new[[u]] = old[[u]] + inc*CF[u, m, old], {u, n}];
                  old = new, {i, step}
              ];
              new
       ],
       {{CF[___], _Real, 1}} 
     ]
CF = Compile[{{u, _Integer}, {m, _Integer, 2}, {em, _Real, 2}}, 
       Module[{n = Length[m], stc = 0.25, gr = 10.0, f = {0.0, 0.0}, 
               spl = 1.0, v, dsquared}, 
              Do[dsquared = Max[0.001, Apply[Plus, (em[[u]] - em[[v]])^2]];
                 f += (1 - m[[u, v]]) (gr/dsquared) (em[[u]] - em[[v]]) - 
                      m[[u, v]] stc Log[dsquared/spl] (em[[u]] - em[[v]]), 
                 {v, n}
              ];
              f
       ]
     ]
squareMatrixQ[{}] = True
squareMatrixQ[r_] := MatrixQ[r] && (Length[r] == Length[r[[1]]])
StableMarriage[mpref_List,fpref_List] :=
	Module[{n=Length[mpref],freemen,cur,i,w,husband},
		freemen = Range[n];
		cur = Table[1,{n}];
		husband = Table[n+1,{n}];
		While[ freemen != {},
			{i,freemen}={First[freemen],Rest[freemen]};
			w = mpref[[ i,cur[[i]] ]];
			If[BeforeQ[ fpref[[w]], i, husband[[w]] ], 
				If[husband[[w]] != n+1,
					AppendTo[freemen,husband[[w]] ]
				];
				husband[[w]] = i,
				cur[[i]]++;
				AppendTo[freemen,i]
			];
		];
		InversePermutation[ husband ]
	] /; Length[mpref] == Length[fpref]
Star[n_Integer?Positive] :=
        Combinatorica`Graph[Table[{{i, n}}, {i, n-1}], 
                 Append[CircularEmbedding[n-1], {{0, 0}}]
        ]
StirlingFirst[n_Integer,m_Integer] := StirlingFirst1[n,m] /; ((n>=0)&&(m>=0))
StirlingFirst1[n_Integer,0] := If [n == 0, 1, 0] 
StirlingFirst1[0,m_Integer] := If [m == 0, 1, 0]
StirlingFirst1[n_Integer,m_Integer] := 
        Block[{$RecursionLimit = Infinity},
               StirlingFirst1[n,m] = (n-1) StirlingFirst1[n-1,m] + StirlingFirst1[n-1, m-1] 
        ]
StirlingSecond[n_Integer,0] := If [n == 0, 1, 0]
StirlingSecond[0,k_Integer] := If [k == 0, 1, 0]
StirlingSecond[n_Integer?Positive, k_Integer?Positive] := 
        Sum [ (-1)^(k-i)*Binomial [k, i]*(i^n), {i, 1, k}]/k!
Strings[l_List,0] := { {} }
Strings[l_List, k_Integer] := Strings[Union[l], k] /; (Length[l] =!= Length[Union[l]])
Strings[l_List,k_Integer] := Distribute[Table[l, {k}], List]
StronglyConnectedComponents[g_Combinatorica`Graph] :=
	Block[{e=ToAdjacencyLists[g],s,c=1,i,cur={},
               low=Table[0,{V[g]}], dfs=Table[0,{V[g]}],scc={}
              },
	      While[(s=Select[Range[V[g]],(dfs[[#]]==0)&]) != {},
	            SearchStrongComp[First[s]];
              ];
              ToCanonicalSetPartition[scc]
	] /; !UndirectedQ[g]
(* Note: Subsets is now kernel functionality; the following definition
   is provided for compatibility. *)
prot = Unprotect[Subsets];
Subsets[n_Integer] := Subsets[Range[n]]
Protect @@ prot;
Combinatorica`SymmetricGroup[n_Integer] := Permutations[n] /; (n >= 0)
SymmetricCoefficient[y_[i_], y_]  := i;
SymmetricCoefficient[y_[i_]^k_, y_]  := i^k k!;
SymmetricCoefficient[u_ v_, y_]  :=
        Block[{$RecursionLimit = Infinity}, SymmetricCoefficient[u, y]*SymmetricCoefficient[v, y]]
SymmetricGroupIndex[n_Integer?Positive, x_Symbol] :=
        Apply[Plus,
              Map[#/SymmetricCoefficient[#, x]&,
                  Map[Apply[Times, #]&, Map[x, Partitions[n], {2}]]
              ]
        ]
SymmetricQ[r_?squareMatrixQ] := (r === Transpose[r])
SymmetricQ[g_Combinatorica`Graph] := 
        Module[{e = Edges[g]},
               Apply[And, Map[MemberQ[e, Reverse[#]]&, e]]
        ]  /; !UndirectedQ[g]
SymmetricQ[g_Combinatorica`Graph] := True
TableauClasses[p_?Combinatorica`PermutationQ] :=
	Module[{classes=Table[{},{Length[p]}],t={}},
		Scan [
			(t = InsertIntoTableau[#,t];
			 PrependTo[classes[[Position[First[t],#] [[1,1]] ]], #])&,
			p
		];
		Select[classes, (# != {})&]
	]
Combinatorica`TableauQ[{}] = True
Combinatorica`TableauQ[t_List] :=
	And [
		Apply[And, Map[(Apply[LessEqual,#])&, t] ],
		Apply[And, Map[(Apply[LessEqual,#])&, Combinatorica`TransposeTableau[t]] ],
		Apply[GreaterEqual, ShapeOfTableau[t] ],
		Apply[GreaterEqual, Map[Length,Combinatorica`TransposeTableau[t]] ]
	]
TableauToYVector[t_?Combinatorica`TableauQ] :=
	Module[{i,y=Table[1,{Length[Flatten[t]]}]},
		Do [ Scan[ (y[[#]]=i)&, t[[i]] ], {i,2,Length[t]} ];
		y
	]
Tableaux[s_List] :=
	Module[{t = LastLexicographicTableau[s]},
		Table[ t = NextTableau[t], {NumberOfTableaux[s]} ]
	]
Tableaux[n_Integer?Positive] := Apply[ Join, Map[ Tableaux, Partitions[n] ] ]
TableauxToPermutation[p1_?Combinatorica`TableauQ,q1_?Combinatorica`TableauQ] :=
	Module[{p=p1, q=q1, row, firstrow},
		Reverse[
			Table[
				firstrow = First[p];
				row = Position[q, Max[q]] [[1,1]];
				p = DeleteFromTableau[p,row];
				q[[row]] = Drop[ q[[row]], -1];
				If[ p == {},
					First[firstrow],
					First[Complement[firstrow,First[p]]]
				],
				{Apply[Plus,ShapeOfTableau[p1]]}
			]
		]
	] /; ShapeOfTableau[p1] === ShapeOfTableau[q1]
TetrahedralGraph :=
 Combinatorica`Graph[{{{1, 4}}, {{2, 4}}, {{3, 4}}, {{1, 2}}, {{2, 3}}, {{1, 3}}}, 
 {{{-0.4999999999999998, 0.8660254037844387}}, 
  {{-0.5000000000000004, -0.8660254037844384}}, {{1., 0}}, {{0, 0}}}]
ThomassenGraph :=
 Combinatorica`Graph[{{{1, 3}}, {{1, 4}}, {{2, 4}}, {{2, 5}}, {{3, 5}}, {{6, 8}}, {{6, 9}}, 
  {{7, 9}}, {{7, 10}}, {{8, 10}}, {{11, 13}}, {{11, 14}}, {{12, 14}}, 
  {{12, 15}}, {{13, 15}}, {{16, 18}}, {{16, 19}}, {{17, 19}}, {{17, 20}}, 
  {{18, 20}}, {{10, 20}}, {{5, 15}}, {{25, 26}}, {{26, 27}}, {{21, 27}}, 
  {{21, 22}}, {{22, 23}}, {{23, 24}}, {{24, 28}}, {{28, 29}}, {{29, 30}}, 
  {{30, 31}}, {{31, 32}}, {{32, 33}}, {{33, 34}}, {{25, 34}}, {{6, 25}}, 
  {{7, 26}}, {{8, 27}}, {{9, 21}}, {{1, 21}}, {{2, 22}}, {{3, 23}}, 
  {{4, 24}}, {{11, 28}}, {{12, 29}}, {{13, 30}}, {{14, 31}}, {{16, 31}}, 
  {{17, 32}}, {{18, 33}}, {{19, 34}}}, 
 {{{-1.6909830056250525, -1.0489434837048464}}, 
  {{-2.8090169943749475, -1.4122147477075266}}, 
  {{-2.8090169943749475, -2.5877852522924734}}, 
  {{-1.6909830056250528, -2.9510565162951536}}, {{-1., -2.0000000000000004}}, 
  {{-1.6909830056250525, 2.9510565162951536}}, 
  {{-2.8090169943749475, 2.5877852522924734}}, 
  {{-2.8090169943749475, 1.4122147477075266}}, 
  {{-1.6909830056250528, 1.0489434837048464}}, {{-1., 1.9999999999999998}}, 
  {{1.6909830056250525, -2.9510565162951536}}, 
  {{2.8090169943749475, -2.5877852522924734}}, 
  {{2.8090169943749475, -1.412214747707527}}, 
  {{1.690983005625053, -1.0489434837048464}}, {{1., -1.9999999999999996}}, 
  {{1.6909830056250525, 1.0489434837048464}}, 
  {{2.8090169943749475, 1.4122147477075266}}, 
  {{2.8090169943749475, 2.587785252292473}}, 
  {{1.690983005625053, 2.9510565162951536}}, {{1., 2.0000000000000004}}, 
  {{-1.3510643118126104, -0.0027813157801774846}}, 
  {{-3.6989356881873894, -0.7656509701858061}}, 
  {{-3.6989356881873894, -3.234349029814194}}, 
  {{-1.3510643118126109, -3.9972186842198227}}, 
  {{-1.3510643118126104, 3.9972186842198223}}, 
  {{-3.6989356881873894, 3.2343490298141937}}, 
  {{-3.6989356881873894, 0.7656509701858059}}, 
  {{1.35106431181261, -3.9972186842198223}}, 
  {{3.6989356881873894, -3.234349029814194}}, 
  {{3.6989356881873903, -0.7656509701858067}}, 
  {{1.3510643118126109, -0.0027813157801772626}}, 
  {{3.6989356881873894, 0.7656509701858059}}, 
  {{3.6989356881873903, 3.2343490298141933}}, 
  {{1.3510643118126109, 3.9972186842198227}}}]
Options[ToAdjacencyLists] = {Type -> All};
ToAdjacencyLists[g_Combinatorica`Graph, opts___?OptionQ] :=
       Module[{type, s, al, e = Edges[g], n, i},
              type = Type /. Flatten[{opts, Options[ToAdjacencyLists]}];
              s = Join[If[UndirectedQ[g], Double[e], e],
                       Table[{i, -1}, {i, V[g]}]
                  ];
              al = Map[Rest,
                       Split[Sort[s], (#1[[1]] === #2[[1]]) &] /.
                       {_Integer, n_Integer} :> n
                   ];
              If[type === Simple,
                 Map[Union, Table[Select[al[[i]], (# != i) &], {i,Length[al]}]],
                 al
              ]
       ]
ToAdjacencyLists[g_Combinatorica`Graph, Combinatorica`EdgeWeight, opts___?OptionQ] :=
       Module[{type, s, al, e = Edges[g, Combinatorica`EdgeWeight], i},
              type = Type /. Flatten[{opts, Options[ToAdjacencyLists]}];
              s = Join[If[UndirectedQ[g], Double[e, Combinatorica`EdgeWeight], e],
                       Table[{{i, -1}, 1}, {i, V[g]}]
                  ];
              al = Map[Rest[Map[Rest, Partition[Flatten[#], 3, 3]]] &,
                       Split[Sort[s], (#1[[1, 1]] == #2[[1, 1]]) &]
                   ];
              If[type === Simple,
                 Map[Union,
                     Table[Select[al[[i]],(#[[1]] != i) &], {i, Length[al]}]
                 ],
                 al
              ]
       ]
Options[ToAdjacencyMatrix] = {Type -> All};
ToAdjacencyMatrix[g_Combinatorica`Graph, opts___?OptionQ] :=
        Module[{e = ToAdjacencyLists[g], blanks = Table[0, {V[g]}], type, am, nb, i},
               type = Type /. Flatten[{opts, Options[ToAdjacencyMatrix]}]; 
               am = Table[nb = blanks; Scan[nb[[#]]++ &, e[[i]]]; nb, 
                          {i, Length[e]}
                    ];
               If[type === Simple, 
                  Do[am[[i, i]] = 0, {i, V[g]}]; am /. _Integer?Positive -> 1, 
                  am
               ]
        ]
ToAdjacencyMatrix[g_Combinatorica`Graph, Combinatorica`EdgeWeight, opts___?OptionQ] :=
        Module[{adjList = ToAdjacencyLists[g, Combinatorica`EdgeWeight],
                freshRow = Table[Infinity, {V[g]}], row, am, type, i},
               type = Type /. Flatten[{opts, Options[ToAdjacencyMatrix]}]; 
               am = Map[(row = freshRow;
                        If[#==={}, 
                           row,
                           Apply[Set[Part[row, #1], #2]&, Transpose[#]]; row
                        ])&,
                        adjList
                    ]; 
               If[type === Simple,
                  Do[am[[i, i]] = 0, {i, V[g]}]; am,
                  am
               ]
        ]
ToCanonicalSetPartition[{}] :=  {}
ToCanonicalSetPartition[sp_?SetPartitionQ] := 
      Transpose[Sort[Map[{First[#], #} &, Map[Sort, sp]]]] [[2]]
ToCanonicalSetPartition[sp_List, X_List] := 
      Map[Last, 
          Sort[Map[{First[#], #}&, 
                   Map[Sort[#, 
                            (Position[X,#1][[1,1]]<Position[X,#2][[1,1]])& 
                       ]&,
                       sp
                   ]
               ], 
               (Position[X,#1[[1]]][[1,1]] < Position[X,#2[[1]]][[1,1]])&
          ]
      ] /; SetPartitionQ[sp, X]
(* alas, the definitions are subtly different, and this introduces some test
   failures... keeping the old version for now.
Combinatorica`ToCycles[p_?Combinatorica`PermutationQ] := System`ToCycles[p, Singletons -> False];
*)
Combinatorica`ToCycles[p_?Combinatorica`PermutationQ] :=
        Module[{k, j, first, np = p, q = Table[0, {Length[p]}], i},
               DeleteCases[
                   Table[If[np[[i]] == 0,
                            {},
                            j = 1; first = np[[i]]; np[[i]] = 0; 
                            k = q[[j++]] = first;
                            While[np[[k]] != 0, q[[j++]] = np[[k]]; np[[k]] = 0; k = q[[j-1]]];
                            Take[q, j-1]
                         ],
                         {i, Length[p]}
                   ],
                   _?(#==={}&)
               ]
        ]
ToInversionVector[p_?Combinatorica`PermutationQ] :=
	Module[{i,inverse=InversePermutation[p]},
		Table[
			Length[ Select[Take[p,inverse[[i]]], (# > i)&] ],
			{i,Length[p]-1}
		]
	] /; (Length[p] > 0)
Options[ToOrderedPairs] = {Type -> All}
ToOrderedPairs[g_Combinatorica`Graph, opts___?OptionQ] := 
        Module[{type, op},
               type = Type /. Flatten[{opts, Options[ToOrderedPairs]}];
               op = If[UndirectedQ[g], Double[Edges[g]], Edges[g]];
               If[type === Simple, Union[Select[op, (#[[1]] != #[[2]])&]], op] 
        ]
Options[ToUnorderedPairs] = {Type -> All};
ToUnorderedPairs[g_Combinatorica`Graph, opts___?OptionQ] := 
        Module[{type, el},
               type = Type /. Flatten[{opts, Options[ToUnorderedPairs]}];
               el = If[UndirectedQ[g], Edges[g], Map[Sort, Edges[g]]];
               If[type === All, el, Union[Select[el, (#[[1]] != #[[2]])&]]]
        ]
Toascii[s_String] := First[ ToCharacterCode[s] ]
Combinatorica`TopologicalSort[g_Combinatorica`Graph] := Range[V[g]] /; EmptyQ[g]
Combinatorica`TopologicalSort[g_Combinatorica`Graph] :=
	Module[{g1 = RemoveSelfLoops[g],e,indeg,zeros,v},
		e=ToAdjacencyLists[g1];
		indeg=InDegree[g1];
		zeros = Flatten[ Position[indeg, 0] ];
		Table [
			{v,zeros}={First[zeros],Rest[zeros]};
			Scan[
				( indeg[[#]]--;
				  If[indeg[[#]]==0, AppendTo[zeros,#]] )&,
				e[[ v ]]
			];
			v,
			{V[g]}
		]
	] /; AcyclicQ[RemoveSelfLoops[g]] && !UndirectedQ[g]
 
TransitiveClosure[g_Combinatorica`Graph] := g /; EmptyQ[g]
TransitiveClosure[g_Combinatorica`Graph] := 
        Module[{e = ToAdjacencyMatrix[g]},
               If[UndirectedQ[g], 
                  FromAdjacencyMatrix[TC[e], Vertices[g, All]],
                  FromAdjacencyMatrix[TC[e], Vertices[g, All], Type -> Directed]
               ]
        ]
TC = Compile[{{e, _Integer, 2}},
             Module[{ne = e, n = Length[e], i, j, k},
                    Do[If[ne[[j, i]] != 0, 
                          Do[If[ne[[i, k]] != 0, ne[[j, k]] = 1], {k, n}]
                       ], {i, n}, {j, n}
                    ];
                    ne
             ]
     ]
TransitiveQ[r_?squareMatrixQ] := 
        TransitiveQ[FromAdjacencyMatrix[r, Type->Directed]]
TransitiveQ[g_Combinatorica`Graph] := IdenticalQ[g,TransitiveClosure[g]]
TransitiveReduction[g_Combinatorica`Graph] := g /; EmptyQ[g]
TransitiveReduction[g_Combinatorica`Graph] :=
	Module[{closure = ToAdjacencyMatrix[g]},
               If[UndirectedQ[g],
                  FromAdjacencyMatrix[TR[closure], Vertices[g, All]],
                  If[AcyclicQ[RemoveSelfLoops[g]],
		     FromAdjacencyMatrix[TRAcyclic[closure], Vertices[g, All], Type->Directed],
		     FromAdjacencyMatrix[TR[closure], Vertices[g, All], Type->Directed]
                  ] 
               ]
	] 
TR = Compile[{{closure, _Integer, 2}},
        Module[{reduction = closure, n = Length[closure], i, j, k},
               Do[
                  If[reduction[[i,j]]!=0 && reduction[[j,k]]!=0 &&
                     reduction[[i,k]]!=0 && (i!=j) && (j!=k) && (i!=k),
                     reduction[[i,k]] = 0
                  ],
                  {i,n},{j,n},{k,n}
                ]; 
                reduction
        ]
     ]
TRAcyclic = Compile[{{closure, _Integer, 2}},
               Module[{n = Length[closure], reduction = closure, i, j, k},
                      Do[
                         If[closure[[i,j]]!=0 && closure[[j,k]]!=0 &&
                            reduction[[i,k]]!=0 && (i!=j) && (j!=k) && (i!=k),
                            reduction[[i,k]] = 0
                         ],
                         {i,n},{j,n},{k,n}
                      ]; 
                      reduction
               ]
            ]
TranslateVertices[v:{{{_?NumericQ, _?NumericQ},___?OptionQ}...}, 
                  {x_?NumericQ, y_?NumericQ}] := 
        Module[{p = Map[First, v], np, i},
               np = TranslateVertices[p, {x, y}];
               Table[{np[[i]], Apply[Sequence, Rest[v[[i]]]]}, {i, Length[np]}]
        ]
TranslateVertices[v:{{_?NumericQ, _?NumericQ}...}, {x_?NumericQ, y_?NumericQ}] := 
        Map[(# + {x,y})&, v]//N
TranslateVertices[g_Combinatorica`Graph, {x_?NumericQ, y_?NumericQ}] := 
        ChangeVertices[g, TranslateVertices[Vertices[g, All], {x, y}] ]
TranslateVertices[g_Combinatorica`Graph, s_List, t_] :=
        Module[{v = Vertices[g, All]},
               ChangeVertices[g, v[[s]] = TranslateVertices[v[[s]], t]; v]
        ]
TransposeGraph[Combinatorica`Graph[g_List,v_List]] := Combinatorica`Graph[ Transpose[g], v ]
TransposePartition[{}] := {}
TransposePartition[p_List] :=
	Module[{s=Select[p,(#>0)&], i, row, r},
		row = Length[s];
		Table [
			r = row;
			While [s[[row]]<=i, row--];
			r,
			{i,First[s]}
		]
	]
Combinatorica`TransposeTableau[tb_List] :=
	Module[{t=Select[tb,(Length[#]>=1)&],row},
		Table[
			row = Map[First,t];
			t = Map[ Rest, Select[t,(Length[#]>1)&] ];
			row,
			{Length[First[tb]]}
		]
	]
TravelingSalesman::ham = "The graph must contain a Hamiltonian cycle for a traveling salesman tour to be found.";
TravelingSalesman[g_Combinatorica`Graph] :=
	Module[{v, s={1}, sol={}, done, cost, e=ToAdjacencyLists[g],
                x, ind, best, n=V[g]},
		ind=Table[1,{n}];
		best = Infinity;
		While[ Length[s] > 0,
			v = Last[s];
			done = False;
			While[ ind[[v]] <= Length[e[[v]]] && !done,
				x = e[[v,ind[[v]]++]];
				done = (best > CostOfPath[g,Append[s,x]]) &&
					!MemberQ[s,x]
			];
			If[done, AppendTo[s,x], s=Drop[s,-1]; ind[[v]] = 1];
			If[(Length[s] == n),
				cost = CostOfPath[g, Append[s,First[s]]];
				If [(cost < best), sol = s; best = cost ];
				s = Drop[s,-1]
			]
		];
		Append[sol,First[sol]]
	] /; HamiltonianQ[g]
TravelingSalesman[g_Combinatorica`Graph] := $$dummy/;(Message[TravelingSalesman::ham]; False)
TravelingSalesmanBounds[g_Combinatorica`Graph] := {LowerBoundTSP[g], UpperBoundTSP[g]}
TreeIsomorphismQ[t1_Combinatorica`Graph?TreeQ, t2_Combinatorica`Graph?TreeQ] := 
        (V[t1] == V[t2]) &&
        (IdenticalQ[t1, t2] || (TreeToCertificate[t1]==TreeToCertificate[t2]))
TreeQ[g_Combinatorica`Graph] := ConnectedQ[g] && (M[g] == V[g]-1)
TreeToCertificate[t_Combinatorica`Graph?TreeQ] := 
        Module[{codes = Table["01", {V[t]}], al, leaves, nbrLeaves, nt = t, i}, 
               While[V[nt] > 2, 
                     al = ToAdjacencyLists[nt]; 
                     leaves  = Flatten[Position[al, _?(Length[#] == 1 &)], 1]; 
                     nbrLeaves = Apply[Union, Map[al[[#]] &, leaves]]; 
                     Do[codes[[ nbrLeaves[[ i ]] ]] = 
                        StringInsert[
                            StringInsert[
                                Apply[StringJoin, 
                                      Sort[Append[codes[[Intersection[al[[nbrLeaves[[i]]]], leaves]]], 
                                                  StringDrop[StringDrop[codes[[nbrLeaves[[i]]]], 1],-1]
                                           ] 
                                      ]
                                ], "0", 1
                            ], "1", -1
                        ],      
                        {i, Length[nbrLeaves]}
                     ]; 
                     codes = codes[[Complement[Range[V[nt]], leaves]]]; 
                     nt = DeleteVertices[nt, leaves]
               ]; 
               Apply[StringJoin, Sort[codes]] 
        ]
TriangleInequalityQ[e_?squareMatrixQ] :=
	Module[{i,j,k,n=Length[e],flag=True},
		Do [
                        If[(e[[i, k]]!=0)&&(e[[k, j]]!=0)&&(e[[i,j]] !=0),
				If[ e[[i,k]]+ e[[k,j]] < e[[i,j]],
					flag = False;
				]
			],
			{i,n},{j,n},{k,n}
		];
		flag
	]
TriangleInequalityQ[g_Combinatorica`Graph] := 
        Block[{e = Edges[g], w = GetEdgeWeights[g], 
               m = Table[0, {V[g]}, {V[g]} ], i},
              If[UndirectedQ[g], e = Double[e]; w = Join[w, w]]; 
              Do[m[[ e[[i,1]], e[[i, 2]] ]] = w[[ i ]], {i, Length[e]}
              ];
              TriangleInequalityQ[m]
        ] /; SimpleQ[g] 
 
Turan[n_Integer, 2] := Combinatorica`GraphUnion[n, CompleteGraph[1]] /; (n > 0)
Turan[n_Integer,p_Integer] :=
	Module[{k = Floor[ n / (p-1) ], r},
		r = n - k (p-1);
		Apply[CompleteGraph, Join[Table[k,{p-1-r}], Table[k+1,{r}]]]
	] /; (n >= p) && (p > 2)
 
Turan[n_Integer, p_Integer] := CompleteGraph[n] /; (n < p) && (p > 2)
TutteGraph :=
 Combinatorica`Graph[{{{1, 11}}, {{1, 12}}, {{1, 13}}, {{2, 3}}, {{2, 8}}, {{2, 20}}, 
  {{3, 4}}, {{3, 42}}, {{4, 5}}, {{4, 28}}, {{5, 6}}, {{5, 34}}, {{6, 7}}, 
  {{6, 46}}, {{7, 10}}, {{7, 30}}, {{8, 9}}, {{8, 22}}, {{9, 10}}, {{9, 23}}, 
  {{10, 25}}, {{11, 14}}, {{11, 15}}, {{12, 27}}, {{12, 29}}, {{13, 31}}, 
  {{13, 32}}, {{14, 16}}, {{14, 22}}, {{15, 16}}, {{15, 19}}, {{16, 17}}, 
  {{17, 18}}, {{17, 21}}, {{18, 19}}, {{18, 24}}, {{19, 25}}, {{20, 26}}, 
  {{20, 41}}, {{21, 22}}, {{21, 23}}, {{23, 24}}, {{24, 25}}, {{26, 27}}, 
  {{26, 39}}, {{27, 35}}, {{28, 29}}, {{28, 40}}, {{29, 35}}, {{30, 31}}, 
  {{30, 45}}, {{31, 36}}, {{32, 33}}, {{32, 36}}, {{33, 34}}, {{33, 43}}, 
  {{34, 44}}, {{35, 37}}, {{36, 38}}, {{37, 39}}, {{37, 40}}, {{38, 43}}, 
  {{38, 45}}, {{39, 41}}, {{40, 42}}, {{41, 42}}, {{43, 44}}, {{44, 46}}, 
  {{45, 46}}}, {{{0.518, 0.586}}, {{0.294, 0.986}}, {{0.504, 0.99}}, 
  {{0.69, 0.99}}, {{0.998, 0.616}}, {{0.872, 0.374}}, {{0.746, 0.152}}, 
  {{0.024, 0.558}}, {{0.17, 0.382}}, {{0.334, 0.15}}, {{0.454, 0.54}}, 
  {{0.518, 0.67}}, {{0.592, 0.53}}, {{0.35, 0.548}}, {{0.436, 0.484}}, 
  {{0.342, 0.502}}, {{0.296, 0.478}}, {{0.336, 0.418}}, {{0.408, 0.404}}, 
  {{0.332, 0.93}}, {{0.214, 0.502}}, {{0.138, 0.558}}, {{0.226, 0.43}}, 
  {{0.282, 0.38}}, {{0.368, 0.272}}, {{0.394, 0.822}}, {{0.464, 0.732}}, 
  {{0.638, 0.894}}, {{0.55, 0.734}}, {{0.696, 0.274}}, {{0.62, 0.482}}, 
  {{0.658, 0.55}}, {{0.768, 0.568}}, {{0.906, 0.6}}, {{0.508, 0.774}}, 
  {{0.674, 0.5}}, {{0.508, 0.83}}, {{0.728, 0.482}}, {{0.424, 0.864}}, 
  {{0.556, 0.894}}, {{0.414, 0.922}}, {{0.506, 0.934}}, {{0.784, 0.506}}, 
  {{0.842, 0.482}}, {{0.76, 0.376}}, {{0.824, 0.412}}}]
TwoColoring[g_Combinatorica`Graph] := {} /; (V[g] == 0)
TwoColoring[g_Combinatorica`Graph] := TwoColoring[MakeSimple[g]] /; (!SimpleQ[g]) || (!UndirectedQ[g])
TwoColoring[g_Combinatorica`Graph] := 
        Module[{c = Combinatorica`ConnectedComponents[g], p, b}, 
                Mod[
                    Flatten[
                       Map[Cases[#, _Integer] &, 
                           Transpose[Map[Combinatorica`BreadthFirstTraversal[g, #[[1]], Level]&, c]]
                       ], 1
                    ], 2
                ] + 1
        ]
UndirectedQ[g_Combinatorica`Graph] := (!MemberQ[GraphOptions[g], EdgeDirection->On]) &&
                        (!MemberQ[GraphOptions[g], EdgeDirection->True])
UnionSet[a_Integer,b_Integer,s_List] :=
	Module[{sa=FindSet[a,s], sb=FindSet[b,s], set=s},
		If[ set[[sa,2]] < set[[sb,2]], {sa,sb} = {sb,sa} ];
		set[[sa]] = {sa, Max[ set[[sa,2]], set[[sb,2]]+1 ]};
		set[[sb]] = {sa, set[[sb,2]]};
		set
	]
Uniquely3ColorableGraph :=
 Combinatorica`Graph[{{{1, 2}}, {{1, 5}}, {{1, 8}}, {{1, 12}}, {{2, 3}}, {{2, 6}},
     {{2, 11}}, {{3, 4}}, {{3, 7}}, {{3, 12}}, {{4, 5}}, {{4, 8}}, {{4, 10}},
     {{5, 6}}, {{5, 9}}, {{6, 7}}, {{6, 10}}, {{7, 8}}, {{7, 9}}, {{8, 11}},
     {{9, 11}}, {{10, 12}}},
  {{{-0.383, -0.924}}, {{0.383, -0.924}}, {{0.924, -0.383}}, {{0.924, 0.383}},
     {{0.383, 0.924}}, {{-0.383, 0.924}}, {{-0.924, 0.383}}, {{-0.924, -0.383}},
     {{-1.2, 2.2}}, {{1.2, 2.2}}, {{-1.2, -2.2}}, {{1.2, -2.2}}}]
UnitransitiveGraph := 
        Module[{i, c = CircularEmbedding[10]}, 
               AddEdges[
                  Combinatorica`GraphUnion[
                     Cycle[10], 
                     MakeUndirected[MakeGraph[Range[10], (Mod[#1 - #2, 10] == 3)&]]
                  ], 
                  Table[{i, i + 10}, {i, 10}]
               ] /. Combinatorica`Graph[a_List, v_List] :> Combinatorica`Graph[a, Join[1.5c, c]]
        ]
UnrankBinarySubset[n_Integer, 0] := {}
UnrankBinarySubset[n_Integer, m_Integer?Positive] := UnrankBinarySubset[Mod[n, 2^m], Range[m]] 
UnrankBinarySubset[n_Integer, l_List] := 
        l[[Flatten[Position[IntegerDigits[Mod[n, 2^Length[l]], 2, Length[l]], 1], 1]]]
UnrankGrayCodeSubset[0, {}] := {}
UnrankGrayCodeSubset[m_Integer, s_List] := 
       Module[{c = Table[0, {Length[s]}], n = Length[s], b, nm, i}, 
              nm = Mod[m, 2^n];
              b = IntegerDigits[nm, 2, Length[s]];
              c[[ 1 ]] = b[[1]]; 
              Do[c[[i]] = Mod[b[[i]] + b[[i-1]], 2], {i, 2, n}]; 
              s[[ Flatten[Position[c, 1], 1] ]]
       ] 
UnrankKSetPartition[r_Integer, {}, 0] := {} 
UnrankKSetPartition[0, set_List, k_Integer?Positive] :=
       Append[Table[{set[[i]]}, {i, 1, k-1}],
              Take[set, -(Length[set]-k+1)]
       ] /; (k <= Length[set])
UnrankKSetPartition[r_Integer?Positive, set_List, k_Integer?Positive] :=
       Block[{n = Length[set], t, j, tempSP, $RecursionLimit = Infinity},
             If[r < StirlingSecond[n-1, k-1],
                Prepend[UnrankKSetPartition[r, Rest[set], k-1],
                        {First[set]}
                ],
                t = r - StirlingSecond[n-1, k-1];
                j = 1 + Mod[t, k];
                tempSP = UnrankKSetPartition[Quotient[t, k], Rest[set], k];
                Prepend[Delete[tempSP, j], Prepend[tempSP[[j]], First[set]]]
             ]
       ] /; (k <= Length[set])
UnrankKSetPartition[r_Integer, n_Integer, k_Integer] := 
       UnrankKSetPartition[r, Range[n], k] /; (k <= n) && (k >= 0)
UnrankKSubset[m_Integer, 1, s_List] := {s[[m + 1]]}
UnrankKSubset[0, k_Integer, s_List] := Take[s, k]
UnrankKSubset[m_Integer, k_Integer, s_List] := 
       Block[{i = 1, n = Length[s], x1, u, $RecursionLimit = Infinity}, 
             u = Binomial[n, k]; 
             While[Binomial[i, k] < u - m, i++]; 
             x1 = n - (i - 1); 
             Prepend[UnrankKSubset[m-u+Binomial[n-x1+1, k], k-1, Drop[s, x1]], s[[x1]]]
       ]
UP[r_Integer, n_Integer] := 
        Module[{r1 = r, q = n!, i}, 
               Table[r1 = Mod[r1, q]; 
                     q = q/(n - i + 1); 
                     Quotient[r1, q] + 1, 
                     {i, n}
               ]
        ]
UnrankPermutation[r_Integer, {}] := {}
UnrankPermutation[r_Integer, l_List] := 
        Module[{s = l, k, t, p = UP[Mod[r, Length[l]!], Length[l]], i}, 
               Table[k = s[[t = p[[i]] ]];  
                     s = Delete[s, t]; 
                     k, 
                     {i, Length[ p ]}
               ] 
        ]
UnrankPermutation[r_Integer, n_Integer?Positive] := 
        UnrankPermutation[r, Range[n]] 
UnrankRGF[0, n_Integer?Positive] := Table[1, {n}]
UnrankRGF[r_Integer?Positive, n_Integer?Positive] := 
        Module[{f = Table[1, {n}], m = 1, tr, i}, tr = r; DValues[n, 1];
               Do[If[tr >= m DValues[n - i, m], 
                     (f[[i]] = m + 1; tr = tr - m DValues[n - i, m]; m++), 
                     (f[[i]] = Quotient[tr, DValues[n - i, m]] + 1; 
                      tr = Mod[tr, DValues[n - i, m]])
                  ], {i, 2, n}
               ]; 
               f
        ]
UnrankSetPartition[0, set_List] := {set}
UnrankSetPartition[r_Integer?Positive, set_List] :=
       Block[{n = Length[set], k = 0, sum = 0, $RecursionLimit = Infinity},
             While[sum <= r, k++; sum = sum + StirlingSecond[n, k]];
             UnrankKSetPartition[r - (sum - StirlingSecond[n, k]), set, k]
       ] /; (r < BellB[ Length[set] ])
UnrankSetPartition[0, 0] = {{}}
UnrankSetPartition[r_Integer, n_Integer?Positive] := UnrankSetPartition[r, Range[n]] /; (r >= 0)
UnrankSubset = NthSubset
UnweightedQ[g_Combinatorica`Graph] := (Count[GetEdgeWeights[g],1] === M[g])
UpperBoundTSP[g_Combinatorica`Graph] :=
	CostOfPath[g, Append[Combinatorica`DepthFirstTraversal[MinimumSpanningTree[g],1],1]]
V[Combinatorica`Graph[_List, v_List, ___?OptionQ]] := Length[v]
Options[VertexColoring] = {Algorithm -> Brelaz};
VertexColoring[g_Combinatorica`Graph, opts___?OptionQ] :=
        Module[{algo = Algorithm /. Flatten[{opts, Options[VertexColoring]}]},
               If[algo === Brelaz, BrelazColoring[g], MinimumVertexColoring[g] ]
        ]
Combinatorica`VertexConnectivity[g_Combinatorica`Graph] := 0 /; EmptyQ[g]
Combinatorica`VertexConnectivity[g_Combinatorica`Graph] := V[g] - 1 /; CompleteQ[g]
Combinatorica`VertexConnectivity[gin_Combinatorica`Graph] :=
	Module[{g=gin,p,k=V[gin],i=0,notedges},
	        If[MultipleEdgesQ[g],
	            Message[Combinatorica`VertexConnectivity::multedge];
	            g=RemoveMultipleEdges[g, True]
		];
		p=VertexConnectivityGraph[g];
		notedges = ToUnorderedPairs[ Combinatorica`GraphComplement[g] ];
		While[i++ <= k,
		      k=Min[
		            Map[
		                (NetworkFlow[p,2 #[[1]],2 #[[2]]-1])&,
			        Select[notedges,(First[#]==i)&]
                            ],
                            k
                        ]
                ];
                k
        ]
Combinatorica`VertexConnectivity[g_Combinatorica`Graph, Cut] := Range[V[g]-1] /; CompleteQ[g]
Combinatorica`VertexConnectivity[gin_Combinatorica`Graph, Cut] :=
	Module[{g = gin, p,k=V[gin],i=0,notedges,c = {}, tc},
		If[MultipleEdgesQ[g],
		    Message[Combinatorica`VertexConnectivity::multedge];
		    g=RemoveMultipleEdges[g, True]
                ];
		p=VertexConnectivityGraph[g];
		notedges = ToUnorderedPairs[ Combinatorica`GraphComplement[g] ];
		While[i++ <= k,
		      {k, c} =First[Sort[
                                         Append[
		                         Map[({Length[tc = NetworkFlow[p,2 #[[1]],2 #[[2]]-1,Cut]], tc})&,
			                 Select[notedges,(First[#]==i)&]],
                                         {k, c}
                                         ]
                                    ]
                              ]
                ];
               Map[Ceiling[First[#]/2]&, c] 
        ]
VertexConnectivityGraph[g_Combinatorica`Graph] :=
	Module[{n=V[g],e=Edges[g,All], v=Vertices[g, All], epsilon=0.05, 
                ne, nv},
               ne = Join[Map[{{2First[#][[1]], 2First[#][[2]]-1}, Combinatorica`EdgeWeight->2}& ,e],
                         Table[{{2i-1, 2i}}, {i, n}]
                    ];
               If[UndirectedQ[g], 
                  ne = Join[ne, 
                            Map[{{2First[#][[2]], 2First[#][[1]]-1}, Combinatorica`EdgeWeight->2}&, e]
                       ]
               ];
               nv = Flatten[
                             Map[
                                 {Prepend[Rest[#], First[#]-{epsilon, 0}],
                                  Prepend[Rest[#], First[#]+{epsilon, 0}]}&,
                                 v
                             ], 1
                    ];
               SetGraphOptions[ChangeEdges[ChangeVertices[g, nv], ne], 
                               EdgeDirection -> True
               ]
	]
Options[VertexCover] = {Algorithm -> Approximate};
VertexCover[g_Combinatorica`Graph, opts___?OptionQ] := 
       Module[{algo = Algorithm /. Flatten[{opts, Options[VertexCover]}]},
              Switch[algo, Approximate, ApproximateVertexCover[g], 
                           Greedy, GreedyVertexCover[g], 
                           Optimum, MinimumVertexCover[g]
              ]
       ]
Combinatorica`VertexCoverQ[g_Combinatorica`Graph, vc_List] := CliqueQ[ Combinatorica`GraphComplement[g], Complement[Range[V[g]], vc] ]
Vertices[Combinatorica`Graph[_List, v_List, ___?OptionQ]] := Map[First[#]&, v]
Vertices[Combinatorica`Graph[_List, v_List, ___?OptionQ], All] := v
WaltherGraph := 
 Combinatorica`Graph[{{{1, 2}}, {{2, 3}}, {{2, 9}}, {{3, 4}}, {{3, 14}}, {{4, 5}}, 
  {{4, 17}}, {{5, 6}}, {{6, 7}}, {{6, 20}}, {{7, 8}}, {{7, 21}}, {{8, 22}}, 
  {{9, 10}}, {{9, 14}}, {{10, 11}}, {{10, 23}}, {{11, 12}}, {{11, 21}}, 
  {{12, 13}}, {{14, 15}}, {{15, 16}}, {{15, 24}}, {{16, 17}}, {{16, 18}}, 
  {{18, 19}}, {{19, 20}}, {{19, 25}}, {{21, 25}}, {{23, 24}}, {{24, 25}}}, 
 {{{0.526, 0.958}}, {{0.526, 0.854}}, {{0.594, 0.768}}, {{0.652, 0.686}}, 
  {{0.724, 0.562}}, {{0.79, 0.44}}, {{0.87, 0.304}}, {{0.91, 0.22}}, 
  {{0.4, 0.704}}, {{0.298, 0.53}}, {{0.162, 0.304}}, {{0.114, 0.24}}, 
  {{0.066, 0.17}}, {{0.508, 0.68}}, {{0.508, 0.552}}, {{0.596, 0.56}}, 
  {{0.628, 0.622}}, {{0.63, 0.488}}, {{0.662, 0.412}}, {{0.726, 0.42}}, 
  {{0.544, 0.304}}, {{0.962, 0.14}}, {{0.37, 0.482}}, {{0.45, 0.422}}, 
  {{0.544, 0.398}}}]
Combinatorica`WeaklyConnectedComponents[g_Combinatorica`Graph] := Combinatorica`ConnectedComponents[MakeUndirected[g]]
Wheel[n_Integer] :=
        Combinatorica`Graph[Join[Table[{{i, n}}, {i, n-1}], Table[{{i, i+1}}, {i, n-2}], 
                      {{{1, n-1}}}
                 ], 
                 Append[CircularEmbedding[n-1], {{0, 0}}]
	] /; (n >= 3)
WriteGraph[g_Combinatorica`Graph, file_] :=
        Module[{edges=ToAdjacencyLists[g],v=N[NormalizeVertices[Vertices[g]]],i,x,y},
                OpenWrite[file];
                Do[
                        WriteString[file,"      ",ToString[i]];
                        {x,y} = Chop[ v [[i]] ];
                        WriteString[file,"      ",ToString[x]," ",ToString[y]];
                        Scan[
                                (WriteString[file,"     ",ToString[ # ]])&,
                                edges[[i]]
                        ];
                        Write[file],
                        {i,V[g]}
                ];
                Close[file];
        ]
YVectorToTableau[y_List] :=
	Module[{k},
		Table[ Flatten[Position[y,k]], {k,Length[Union[y]]}]
	]
Attributes[Zap] = {Listable}
Zap[n_Integer] := n
Zap[n_Real] := If[Chop[Round[n] - n] == 0, Round[n], n]
End[]
Protect[
AcyclicQ, 
Combinatorica`AddEdge,
AddEdges, 
Combinatorica`AddVertex,
AddVertices,
Algorithm,
Combinatorica`AlternatingGroup,
AlternatingGroupIndex,
AlternatingPaths,
AnimateGraph,
AntiSymmetricQ,
Approximate,
ApproximateVertexCover,
ArticulationVertices,
Automorphisms,
Backtrack, 
BellmanFord,
BiconnectedComponents, 
BiconnectedQ,
BinarySearch, 
BinarySubsets, 
BipartiteMatching, 
BipartiteMatchingAndCover, 
BipartiteQ,
Box, 
BooleanAlgebra,
Combinatorica`BreadthFirstTraversal, 
Brelaz,
BrelazColoring,
Bridges, 
Combinatorica`ButterflyGraph,
ToCanonicalSetPartition,
CageGraph,
CartesianProduct, 
Center, 
ChangeEdges, 
ChangeVertices,
ChromaticNumber, 
ChromaticPolynomial,
ChvatalGraph,
Combinatorica`CirculantGraph, 
CircularEmbedding, 
CircularVertices, 
CliqueQ, 
CoarserSetPartitionQ,
CodeToLabeledTree, 
Cofactor,
CompleteBinaryTree,
Combinatorica`CompleteKaryTree,
CompleteKPartiteGraph,
CompleteGraph,
CompleteQ,
Compositions, 
Combinatorica`ConnectedComponents, 
ConnectedQ, 
ConstructTableau,
Combinatorica`Contract, 
CostOfPath,
CoxeterGraph,
CubeConnectedCycle,
CubicalGraph,
Cut,
Cycle, 
Combinatorica`Cycles, 
CycleIndex,
CycleStructure,
Cyclic,
Combinatorica`CyclicGroup,
CyclicGroupIndex,
Combinatorica`DeBruijnGraph, 
Combinatorica`DeBruijnSequence, 
Degrees,
DegreesOf2Neighborhood,
DegreeSequence,
DeleteCycle, 
Combinatorica`DeleteEdge, 
DeleteEdges, 
DeleteFromTableau, 
Combinatorica`DeleteVertex,
DeleteVertices, 
Combinatorica`DepthFirstTraversal,
DerangementQ, 
Derangements, 
Diameter, 
Dihedral,
Combinatorica`DihedralGroup,
DihedralGroupIndex,
Dijkstra, 
DilateVertices,
Directed, 
Disk, 
Distances,
DistinctPermutations, 
Distribution, 
DodecahedralGraph,
DominatingIntegerPartitionQ,
DominationLattice,
DurfeeSquare, 
Eccentricity,
Edge,
EdgeChromaticNumber, 
Combinatorica`EdgeColor,
EdgeColoring, 
Combinatorica`EdgeConnectivity, 
EdgeDirection, 
Combinatorica`EdgeLabel, 
EdgeLabelColor, 
EdgeLabelPosition,
LoopPosition, 
Edges, 
Combinatorica`EdgeStyle, 
Combinatorica`EdgeWeight, 
Element,
EmptyGraph, 
EmptyQ, 
EncroachingListSet, 
EquivalenceClasses, 
EquivalenceRelationQ, 
Equivalences, 
Euclidean,
Eulerian,
EulerianCycle,
EulerianQ, 
ExactRandomGraph, 
ExpandGraph, 
ExtractCycles, 
FerrersDiagram, 
Combinatorica`FindCycle,
FindSet, 
FiniteGraphs,
FirstLexicographicTableau, 
FolkmanGraph,
FranklinGraph,
FruchtGraph,
FromAdjacencyLists,
FromAdjacencyMatrix,
Combinatorica`FromCycles,
FromInversionVector, 
FromOrderedPairs,
FromUnorderedPairs, 
FunctionalGraph,
GeneralizedPetersenGraph,
GetEdgeLabels,
GetEdgeWeights,
GetVertexLabels,
GetVertexWeights,
Girth, 
Combinatorica`GraphCenter, 
Combinatorica`GraphComplement, 
Combinatorica`GraphDifference, 
GraphicQ,
Combinatorica`GraphIntersection,
Combinatorica`GraphJoin, 
GraphOptions, 
GraphPolynomial,
Combinatorica`GraphPower, 
Combinatorica`GraphProduct, 
Combinatorica`GraphSum, 
Combinatorica`GraphUnion, 
GrayCode,
GrayCodeSubsets, 
GrayCodeKSubsets, 
GrayGraph,
Greedy,
GreedyVertexCover,
Combinatorica`GridGraph, 
GroetzschGraph,
GroetzschGraph,
HamiltonianCycle, 
HamiltonianPath, 
HamiltonianQ, 
Harary,
HasseDiagram, 
Heapify, 
HeapSort, 
HeawoodGraph,
HerschelGraph,
HideCycles, 
HighlightedEdgeColors,
HighlightedEdgeStyle,
HighlightedVertexColors,
HighlightedVertexStyle,
Highlight,
Hypercube, 
IcosahedralGraph,
IdenticalQ,
IdentityPermutation,
Combinatorica`IncidenceMatrix,
InDegree,
IndependentSetQ, 
Index, 
InduceSubgraph,
InitializeUnionFind,
InsertIntoTableau, 
IntervalGraph, 
Invariants,
InversePermutation, 
InversionPoset,
Inversions,
InvolutionQ, 
Involutions, 
IsomorphicQ, 
Isomorphism, 
IsomorphismQ, 
Josephus, 
KnightsTourGraph,
KSetPartitions,
KSubsetGroup,
KSubsetGroupIndex,
KSubsets,
LNorm,
LabeledTreeToCode, 
LastLexicographicTableau,
LexicographicPermutations, 
LexicographicSubsets, 
LeviGraph,
Combinatorica`LineGraph,
ListGraphs,
ListNecklaces,
LongestIncreasingSubsequence, 
LowerLeft, 
LowerRight, 
M,
MakeDirected,
MakeGraph, 
MakeSimple, 
MakeUndirected,
MaximalMatching,
MaximumAntichain, 
MaximumClique, 
MaximumIndependentSet,
MaximumSpanningTree, 
McGeeGraph,
MeredithGraph,
MinimumChainPartition, 
MinimumChangePermutations,
MinimumSpanningTree, 
MinimumVertexColoring, 
MinimumVertexCover, 
MultipleEdgesQ,
MultiplicationTable,
MycielskiGraph,
NecklacePolynomial,
Neighborhood,
NetworkFlow, 
NetworkFlowEdges, 
NextBinarySubset, 
NextComposition, 
NextGrayCodeSubset, 
NextKSubset,
NextLexicographicSubset,
NextPartition, 
NextPermutation, 
NextSubset, 
NextTableau, 
NoMultipleEdges, 
NonLineGraphs,
Normal, 
NormalDashed, 
NormalizeVertices,
NoPerfectMatchingGraph,
NoSelfLoops, 
NthPair,
NthPermutation, 
NthSubset, 
NumberOfCompositions,
NumberOfDerangements, 
NumberOfDirectedGraphs,
NumberOfGraphs,
NumberOfInvolutions, 
NumberOf2Paths,
NumberOfKPaths,
NumberOfNecklaces,
NumberOfPartitions,
NumberOfPermutationsByCycles, 
NumberOfPermutationsByInversions, 
NumberOfPermutationsByType, 
NumberOfSpanningTrees, 
NumberOfTableaux,
OctahedralGraph,
OddGraph,
One,
Optimum,
OrbitInventory,
OrbitRepresentatives,
Combinatorica`Orbits,
Ordered,
OrientGraph, 
OutDegree,
PairGroup,
PairGroupIndex,
Parent,
ParentsToPaths,
PartialOrderQ, 
PartitionLattice,
PartitionQ, 
Partitions, 
Path, 
PerfectQ,
PermutationGraph,
PermutationGroupQ, 
Combinatorica`PermutationQ, 
PermutationToTableaux, 
PermutationType, 
PermutationWithCycle,
Combinatorica`Permute, 
PermuteSubgraph, 
Combinatorica`PetersenGraph,
PlanarQ,
PlotRange, 
Polya,
PseudographQ, 
RadialEmbedding, 
Radius,
RandomComposition, 
Combinatorica`RandomGraph, 
RandomHeap, 
RandomInteger,
RandomKSetPartition,
RandomKSubset,
RandomPartition, 
Combinatorica`RandomPermutation, 
RandomRGF,
RandomSetPartition,
RandomSubset, 
RandomTableau, 
RandomTree, 
RandomVertices, 
RankBinarySubset, 
RankedEmbedding, 
RankGraph,
RankGrayCodeSubset,
RankKSetPartition,
RankKSubset,
RankPermutation, 
RankRGF,
RankSetPartition,
RankSubset, 
ReadGraph,
RealizeDegreeSequence, 
ReflexiveQ,
RegularGraph, 
RegularQ, 
RemoveMultipleEdges,
RemoveSelfLoops, 
ResidualFlowGraph,
RevealCycles, 
ReverseEdges,
RGFQ,
RGFs,
RGFToSetPartition,
RobertsonGraph,
RootedEmbedding, 
RotateVertices, 
Runs, 
SamenessRelation,
SelectionSort, 
SelfComplementaryQ, 
SelfLoopsQ,
SetEdgeWeights,
SetGraphOptions, 
SetPartitions,
SetPartitionListViaRGF,
SetPartitionQ,
SetPartitionToRGF,
SetEdgeLabels,
SetVertexLabels,
SetVertexWeights,
ShakeGraph,
ShortestPathSpanningTree,
ShowLabeledGraph,
ShowGraph, 
ShowGraphArray, 
ShuffleExchangeGraph,
SignaturePermutation,
Simple, 
SimpleQ,
SmallestCyclicGroupGraph,
Spectrum, 
SpringEmbedding, 
StableMarriage, 
Star,
StirlingFirst, 
StirlingSecond, 
Strings, 
Strong,
StronglyConnectedComponents,
Combinatorica`SymmetricGroup,
SymmetricGroupIndex,
SymmetricQ,
TableauClasses, 
Combinatorica`TableauQ, 
Tableaux,
TableauxToPermutation, 
TetrahedralGraph,
ThickDashed, 
ThinDashed, 
ThomassenGraph,
ToAdjacencyLists,
ToAdjacencyMatrix, 
Combinatorica`ToCycles,
ToInversionVector, 
ToOrderedPairs,
Combinatorica`TopologicalSort, 
ToUnorderedPairs, 
TransitiveClosure, 
TransitiveQ,
TransitiveReduction, 
TranslateVertices, 
TransposePartition, 
Combinatorica`TransposeTableau,
TravelingSalesmanBounds, 
TravelingSalesman, 
TreeIsomorphismQ,
TreeQ, 
TreeToCertificate,
TriangleInequalityQ,
Turan, 
TutteGraph,
TwoColoring, 
Type,
Undirected,
UndirectedQ, 
Uniquely3ColorableGraph,
UnionSet, 
UnitransitiveGraph,
UnrankBinarySubset,
UnrankGrayCodeSubset,
UnrankKSetPartition,
UnrankKSubset,
UnrankPermutation,
UnrankRGF,
UnrankSetPartition,
UnrankSubset,
UnweightedQ,
UpperLeft, 
UpperRight, 
V, 
VertexColor, 
VertexColoring, 
Combinatorica`VertexConnectivity, 
VertexConnectivityGraph, 
VertexCover,
Combinatorica`VertexCoverQ, 
Combinatorica`VertexLabel, 
VertexLabelColor, 
VertexNumber, 
VertexNumberColor,
Combinatorica`VertexStyle, 
Combinatorica`VertexWeight, 
Vertices,
WaltherGraph,
Weak, 
Combinatorica`WeaklyConnectedComponents, 
WeightingFunction,
WeightRange,
Wheel, 
WriteGraph,
Zoom
]
EndPackage[ ]
