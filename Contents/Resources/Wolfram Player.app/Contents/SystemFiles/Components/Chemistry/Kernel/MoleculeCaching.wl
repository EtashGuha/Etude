


Begin["Chemistry`Private`MoleculeCachingDump`"]


(* ************************************************************************* **

                        $MoleculeStore

** ************************************************************************* *)


$Initialized = False;

InitializeMoleculeStore[] := Module[
	{flag},
	flag = newInstance["CacheManager"];
	flag["setFlag"];
	Internal`DeleteMoleculeCache[0];
	Internal`SetMoleculeCache[0,flag];
	Unprotect @ $MoleculeStore;
	$MoleculeStore = Language`NewExpressionStore["Molecule"];
	$Initialized = True;
	Protect @ $MoleculeStore;
]


Chemistry`MoleculeCaching`ClearMoleculeExpressionCache[] := (
	iClearMoleculeExpressionCache[];
	$Initialized = False;
	Internal`DeleteMoleculeCache[0];
)


cacheMolecule[args___] /; !TrueQ[$Initialized] := Module[
	{},
	InitializeMoleculeStore[];
	cacheMolecule[args]
]


getCachedMol[args___] /; !TrueQ[$Initialized] := Module[
	{},
	InitializeMoleculeStore[];
	getCachedMol[args]
]

getMolGraph[args___] /; !TrueQ[$Initialized] := Module[
	{},
	InitializeMoleculeStore[];
	getMolGraph[args]
]

cachedEvaluation[args___] /; !TrueQ[$Initialized] := Module[
	{},
	InitializeMoleculeStore[];
	cachedEvaluation[args]
]

(* :cacheMolecule: *)


cacheMolecule[mol_, imol_, graph_] := (
	$MoleculeStore["put"[mol,$moleculeGraph,graph]];
	
	If[
		Head @ mol === Molecule &&
		!MemberQ[ mol[[1]], Atom["H",___]],
		$MoleculeStore["put"[mol, {$moleculeGraph, "NoHydrogens"},graph]]
	];
	
	If[
		!imol[ "hasImplicitHydrogens"],
		$MoleculeStore["put"[mol, {$moleculeGraph, "AllAtoms"},graph]]
	];
	
	cacheMolecule[ mol, imol];
)


$ReuseLibraryObjects = False;
Protect[$LibraryID];

cachedLibraryID[head_[args___]] /; TrueQ[ $ReuseLibraryObjects] := Replace[
	Internal`CheckMoleculeCache[ molecule[args]],
	Except[ n_Integer /; ManagedLibraryExpressionQ[ iMolecule[n]]] :> $Failed
]

cachedLibraryID[args___] := $Failed

cacheMolecule[mol_, molData_Symbol] := cacheMolecule[ mol, molData["imol"]];
cacheMolecule[mol_, libID_Integer] := cacheMolecule[ mol, iMolecule[ libID] ]; 
cacheMolecule[mol_, imol_iMolecule] := (
	$MoleculeStore["put"[mol,$iMolecule,imol]];
	If[
		Head @ mol === Molecule && !MemberQ[ mol[[1]], Atom["H",___]]
		,
		$MoleculeStore["put"[mol, {$iMolecule, "NoHydrogens"},imol]]
	];
	
	If[
		TrueQ[$ReuseLibraryObjects]
		,
		Replace[
			mol,
			HoldPattern[ Molecule[args___] ] :> If[
				!IntegerQ[ cachedLibraryID[args]]
				,
				Internal`SetMoleculeCache[
					molecule[args], 
					ManagedLibraryExpressionID @ imol
				]
			]
		]
	];
	
	If[
		Head@mol === Molecule && !imol[ "hasImplicitHydrogens"]
		,
		$MoleculeStore["put"[mol, {$iMolecule, "AllAtoms"},imol]]
	];
)




Attributes[MoleculeBlock] = HoldRest;
MoleculeBlock[vars_List, arg_] := Module[ {res},
	res = arg;
	$MoleculeStore[ "remove"[#] ] & /@ vars;
	res
]

(* 	:getCachedMol: *)

$CachedMolPattern = _Molecule | _Atom | _Alternatives

getCachedMol[mols:{__},opts___] := Map[ getCachedMol[ #, opts]&, mols];

getCachedMol[mol_(*$CachedMolPattern*)] := Module[
	{res},
	Which[
		res = $MoleculeStore["get"[ mol, $iMolecule]];
		ManagedLibraryExpressionQ @ res,
			res,
		IntegerQ[ res = cachedLibraryID @ mol],
			iMolecule[res],
		res = Replace[mol, HoldPattern[Molecule[args__]] :> LoadMoleculeData[args] ];	
		ManagedLibraryExpressionQ @ res,
			cacheMolecule[ mol, res];
			res,	
		True,
			Null
	]
	
];

getCachedMol[mol_(*$CachedMolPattern*), "AllAtoms"] := Replace[
	$MoleculeStore["get"[ mol, {$iMolecule, "AllAtoms"}]],
	Null :> Replace[
		getiMolWithExplicitHydrogens[mol],
		res_?checkMolecule :> (
			$MoleculeStore["put"[mol, {$iMolecule, "AllAtoms"},res]];
			res
		)
	]
]

getCachedMol[mol_(*$CachedMolPattern*), "NoHydrogens"] := Replace[
	$MoleculeStore["get"[ mol, {$iMolecule, "NoHydrogens"}]],
	Null :> Replace[
		getiMolWithSuppressedHydrogens[mol],
		res_?checkMolecule :> (
			$MoleculeStore["put"[mol, {$iMolecule, "NoHydrogens"},res]];
			res
		)
	]
]

getCachedMol[mol_(*$CachedMolPattern*), "MoleculePlotAutomatic"] := 
	getiMolWithSuppressedHydrogens[mol, False]
	
getCachedMol[mol_(*$CachedMolPattern*), {"KeepTheseHydrogens", atoms:{__Integer}}] := 
	getiMolWithSuppressedHydrogens[mol, atoms]
	
getCachedMol[mol_(*$CachedMolPattern*), {"DeleteTheseHydrogens", atoms:{__Integer}}] := Module[
	{keepAtoms},
	keepAtoms = Complement[
		AtomList[mol, "H", "AtomIndex"],
		atoms
	];
	getiMolWithSuppressedHydrogens[mol, keepAtoms]
]
getCachedMol[mol_(*$CachedMolPattern*), opts:{___Rule}] :=Switch[
	Lookup[ opts, IncludeHydrogens, Automatic],
	True, getCachedMol[ mol, "AllAtoms"],
	False, getCachedMol[ mol, "NoHydrogens"],
	_, getCachedMol[ mol]
];



getiMolWithSuppressedHydrogens[mol_, removeAll_:True] := Block[
	{id, toCopy, res, coords, imol, keepCoordinates, makeExplicit},
	toCopy = getCachedMol @ mol;
	If[
		TrueQ[ toCopy["hasQuery"]] || !toCopy["hasAnyHydrogens"]
		,
		Return[ toCopy, Block]
		,
		id = ManagedLibraryExpressionID @ getCachedMol @ mol;
		imol = newInstance[];
		coords = Lookup[ Options[ mol], AtomDiagramCoordinates, {}];
		
		
		keepCoordinates = And[
			MatrixQ[coords, NumericQ],
			(Length @ coords) === (Length @ First @ mol)
		];
		makeExplicit = True;
		Switch[ removeAll,
			True,
				imol[ "createCopyWithNoHydrogens", id, keepCoordinates, makeExplicit],
			False,
				imol[ "createCopyWithAutoHydrogens", id, keepCoordinates, makeExplicit],
			{__Integer},
				imol[ "createCopyKeepingHydrogens", id, removeAll, keepCoordinates, makeExplicit]
		];
		If[
			imol["atomCount", True] == Length @ coords,
			imol["addCoordinates", coords]
		];
		imol
	]
]

getiMolWithExplicitHydrogens[mol_] := Block[
	{toCopy, res, coords, imol, keepCoordinates},
	toCopy = getCachedMol @ mol;
	If[
		TrueQ[ toCopy["hasQuery"]]
		,
		Return[ toCopy, Block]
		,
		imol = newInstance[];
	
		keepCoordinates = True;
		imol[ "createCopyWithAddedHydrogens", ManagedLibraryExpressionID @ toCopy, keepCoordinates];
		imol
	]
]


(* 	:getMolGraph: *)

getMolGraph[mol_] := Replace[
	$MoleculeStore["get"[ mol, $moleculeGraph]],
	Null :> Replace[
		Graph[ Range @ Length @ mol[[1]], UndirectedEdge @@@ mol[[2,All,1]] ],
		res_?GraphQ :> (
			$MoleculeStore[ "put"[mol, $moleculeGraph, res]];
			res
		)
	]
]


(* 	:getMolGraph: *)

getMolGraph[mol_, "AllAtoms"] := Replace[
	$MoleculeStore["get"[ mol, {$moleculeGraph, "AllAtoms"}]],
	Null :> Replace[
		simpleGraphFromLibraryMol[ getCachedMol[ mol, "AllAtoms"] ],
		res_?GraphQ :> (
			$MoleculeStore[ "put"[mol, {$moleculeGraph, "AllAtoms"}, res]];
			res
		)
	]
]

getMolGraph[mol_, "NoHydrogens"] := Replace[
	$MoleculeStore["get"[ mol, {$moleculeGraph, "NoHydrogens"}]],
	Null :> Replace[
		simpleGraphFromLibraryMol[ getCachedMol[ mol, "NoHydrogens"] ],
		res_?GraphQ :> (
			$MoleculeStore[ "put"[mol, {$moleculeGraph, "NoHydrogens"}, res]];
			res
		)
	]
]

simpleGraphFromLibraryMol[imol_] := With[
	{natoms = imol["atomCount",True], bonds = imol["getBondsList"]},
	Graph[
		Range @ natoms,
		UndirectedEdge @@@ bonds[[All,;;2]]
	]
];

cachedEvaluation[func_, expr_, args___] := 
	Replace[$MoleculeStore["get"[expr, {func, args}]], Null :>
		Replace[func[expr, args],
			res:Except[_ ? FailureQ] :> 
			($MoleculeStore["put"[expr, {func, args}, res]]; res)
		]
	];


(* 	:iClearMoleculeExpressionCache: *)
(*	This is called by ClearSystemCache["Molecule"] 	*)

iClearMoleculeExpressionCache[] /; $Initialized := 	(
	(* if everything has been done correctly, there should be no references
		left to the managed library expressions outside of the expression store, 
		so clearing the store should delete the library objects.  *)
	$MoleculeStore["remove"[First@#]] & /@ $MoleculeStore["listTable"[]];
	(*but we call the deletion function regardless, for any stray library objects*)
	deleteiMolecule /@ iMoleculeList[];
)
iClearMoleculeExpressionCache[];

	

SetAttributes[
    {
        $iMolecule,
        $moleculeGraph,
        $MoleculeStore,
		cacheMolecule, 
		getCachedMol,
		getMolGraph,
		cachedEvaluation
	},
    {ReadProtected, Protected}
]


End[] (* End Private Context *)

