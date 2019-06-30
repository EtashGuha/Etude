


Begin["Chemistry`Private`MoleculeGraphDump`"]



ConnectedMoleculeQ[ mol_?MoleculeQ] := With[
	{imol = getCachedMol @ mol},
	imol["nFragments"] < 2
];
ConnectedMoleculeQ[___] := False



Options[MoleculeGraph] = Join[
	{IncludeHydrogens -> Automatic,
	Method -> Automatic},
	Options[Graph]
]


MoleculeGraph[ mol_?MoleculeQ, opts:OptionsPattern[]] := With[ 
	{res = iMoleculeGraph[mol, opts]},
	res /; GraphQ[res]
]

MoleculeGraph[ arg1_ /; !MoleculeQ[arg1], ___] := (messageNoMol[MoleculeGraph,arg1]; Null /; False)

iMoleculeGraph[ mol_, opts:OptionsPattern[]] := Module[ 
	{atoms, bonds, edges, imol, graph,
		bondOrders, edgeShapeFunctions, vc, labels, atomSymbols, stereo, method},
		
	method = Replace[
		OptionValue[ MoleculeGraph, {opts}, Method],
		"Quick" :> Return[ getMolGraph[mol], Module]
	];
		
	GraphComputation`ChemicalDataExtension;
	imol = getCachedMol[ mol, 
		Flatten @ {opts}
	];
	
	vc = Replace[
		OptionValue[ MoleculeGraph, {opts}, VertexCoordinates],
		{
			Inherited :> 100 * getiMolProperty[ imol, "coords2D"]
		}
	];
	
	atoms = getiMolProperty[imol,"AtomList"];
	atoms = MapIndexed[ 
		Replace[
			#1,
			Atom[symbol_, atomOptions___Rule] :> Property[
				#2[[1]],
				{
					"AtomicNumber" -> AtomicNumber[symbol],
					atomOptions
				}
			]
		]&,
		atoms
	];
	
	bonds = getiMolProperty[imol,"BondList", (*"Kekulized"*) method === "Multigraph"];
	edges = Switch[ method,
		Automatic | "SimpleGraph" | "Simplegraph",
			Property[ UndirectedEdge @@ #1, {"BondOrder" ->#2}]&,
		"Multigraph" | "MultiGraph",
			(Sequence @@ ConstantArray[ UndirectedEdge @@ #1, Replace[#2, {"Double" ->2, "Triple"->3, _ -> 1}]])&,
		_,
			Message[MoleculeGraph::method, method];
			Return[ $Failed, Module]
	] @@@ bonds;
	
	bondOrders = Replace[ bonds[[All,2]], {"Double" ->2, "Triple"->3, _ -> 1}, 1];
	

	
	graph = Graph[ atoms, edges,
		FilterRules[ {opts}, Options[Graph]],
		VertexCoordinates -> vc];
	(
		If[ 
			(stereo = mol["StereochemistryElements"]) =!= {},
			graph = SetProperty[ graph, "StereochemistryElements" -> stereo]
		];
		graph
	) /; GraphQ[graph]
	
]

makeEntityDownValue[iMoleculeGraph]

iMoleculeGraph[ ___] := $Failed


bondEdgeShape[bond_String][{{x1_, y1_}, {x2_, y2_}}, _] := 
	GraphComputation`GraphBuilderDump`ef[ {{x1, y1}, {x2, y2}}, bond /. {"Double" ->2, "Triple"->3, "Single" -> 1}] 
  
bondEdgeShape[bond_String][coords:{{x1_, y1_, z1_}, {x2_, y2_, z2_}},_] := Cylinder[coords,5] 


atomVertexShape[sym_String][coords:{_, _},n_, dim:{_, _}] := {
	{White, EdgeForm[None],Disk[coords, (Max @ dim)/1.25]}, {Black,Text[sym, coords]}
}


atomVertexShape[sym_String][coords:{_, _, _ },n_, dim:{_, _, _}] := {
	{Sphere[coords, (Max @ dim)/1.25]}
}

Options[ GraphMolecule] = {"GraphValenceRules" -> { 2|3|4 -> "C", 1 -> "H"  }}
GraphMolecule[g_?GraphQ, opts:OptionsPattern[]] := Block[
	{vertices, edges, atoms, bonds, stereo},
	vertices = VertexList[ g];
	edges = EdgeList[ g];
	atoms = atomFromVertex[ {g, #}, Normal @ OptionValue["GraphValenceRules"] ] & /@ vertices;
	bonds = bondFromEdge[ {g, #}] & /@ edges;
	stereo = Replace[
		PropertyValue[g, "StereochemistryElements"],
		{
			x:{__Association} :> Rule[StereochemistryElements, x],
			_ :> Nothing
		}
	];

	Molecule @@ {atoms, bonds, stereo}
]

atomFromVertex[vertex:{g_,v_}, gvr_] := Module[{anum, props},
	anum = Replace[
		PropertyValue[ vertex, "AtomicNumber"],
		$Failed :> Return[ defaultAtomForValence[{g,v}, gvr], Module]
	];
	anum = Replace[
		numberElementMap[anum], 
		_Missing :> Return[ Atom["C"], Module]
	];
	props = PropertyList[ vertex];
	props = Intersection[ props, Keys @ Options @ Atom];
	props = Rule[ #, PropertyValue[vertex, #]] & /@ props;
	Atom[ anum,  Sequence @@ props]
]

defaultAtomForValence[{g_,v_},gvr_] := Replace[
	VertexDegree[g,v],
	gvr(*{
		1 -> Atom["H"],
		2 -> Atom["O"],
		3 -> Atom["N"],
		4 -> Atom["C"],
		_ -> Atom["C"]
	}*)
]

bondFromEdge[edge:{g_,e_}] := Module[{order},
	order = Replace[
		PropertyValue[ edge, "BondOrder"],
		Except[_String] :> "Single"
	];
	Bond[ List @@ e, order]
]


ConnectedMoleculeComponents[mol_ /; ConnectedMoleculeQ[mol] ] := {mol}

ConnectedMoleculeComponents[ mol_?MoleculeQ] := Block[
	{im, components},
	im = getCachedMol[mol];
	components = im["connectedComponents"];
	MoleculeModify[mol, {"ExtractParts", components}, ValenceErrorHandling -> False]
];


(* ::Section:: *)
(*WL Graph descriptors*)


Options[BalabanIndex] = {IncludeHydrogens -> False}

BalabanIndex[m_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{graph},
	graph = getMolGraph[m,
		If[ OptionValue[BalabanIndex, {opts}, IncludeHydrogens], "AllAtoms", "NoHydrogens"]
	];
	BalabanIndex[ graph]
	
]
			

BalabanIndex[g0_Graph] := With[{g = IndexGraph[g0]}, Module[
	{
		den = EdgeCount[g] - VertexCount[g] + Length @ WeaklyConnectedComponents[g] + 1,
		d = GraphDistanceMatrix[g]
	},
	EdgeCount[g]/den Total[(Total[d[[#1]]] Total[d[[#2]]])^(-1/2) & @@@ EdgeList[g]]
]]

HosoyaIndex[g_System`Graph] := Block[{x}, 
	Total[
		CoefficientList[MatchingGeneratingPolynomial[g, x], x]
	]
]

MatchingGeneratingPolynomial[g_System`Graph?UndirectedGraphQ, x_] := Internal`InheritedBlock[{rp},
	rp[System`LineGraph[g], x]
	(* bug(272429): Implement AcyclicPolynomial, IndependencePolynomial, MatchingGeneratingPolynomial, MatchingPolynomial, RankPolynomial, and SpanningPolynomial *)
]

rp[g_, x_] := rp[g, x] = Replace[
	VertexList[g],
	{
		{} -> 1,
		{v_, ___} :> Expand[rp[CanonicalGraph @ VertexDelete[g, v], x] + x rp[CanonicalGraph @ VertexDelete[g, Append[AdjacencyList[g, v], v]], x]]
	}
]



End[] (* End Private Context *)

