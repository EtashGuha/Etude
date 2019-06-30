(* :Title: ResonanceStructures.m *)

(* :Context: Chemistry` *)



Begin["Chemistry`Private`ResonanceStructuresDump`"]


(* ************************************************************************* **

                        ResonanceStructures

** ************************************************************************* *)


Options[ResonanceStructures] = {
	"KekuleAll" -> True,
	"AllowChargeSeparation" -> False,
	"AllowIncompleteOctets" -> False,
	"UnconstrainedCations" -> False,
	"UnconstrainedAnions" -> False
}

ResonanceStructures[args___] :=
    Block[{res, argCheck = System`Private`Arguments[ResonanceStructures[args],{1,2}]},
        (
	        res = Catch[
	        		iResonanceStructures[Sequence @@ argCheck],
	        		$tag
	        ];
	        res /; res =!= $Failed
        ) /; argCheck =!= {}
    ]

(* :getResonanceStructures: *)


iResonanceStructures[{mol_?MoleculeQ, nreturn_Integer:10}, opts:OptionsPattern[]] := Block[
	{imol, structureHolder=newInstance[], res, nstructs, libOpts},
	imol = getCachedMol[mol];
	libOpts = Association[
		FilterRules[opts, {"KekuleAll", "AllowChargeSeparation", "AllowIncompleteOctets", 
				"UnconstrainedCations", "UnconstrainedAnions"}
		]
	];
	libOpts["MaxStructs"] = nreturn;
	coords = mol["AtomDiagramCoordinates"];
	
	nstructs = structureHolder[ "generateResonanceStructures", ManagedLibraryExpressionID @ imol, libOpts];		
	
	res = Table[
		Block[{expr, pickle, data, newMol = newInstance[]},
			
			newMol[ "takeOwnershipOfResonanceStructure", ManagedLibraryExpressionID @ structureHolder, n];
			If[
				checkMolecule[newMol],
				(* This needs to roundtrip for some reason *)
				expr = Molecule @@ List @@ Molecule[newMol,IncludeAromaticBonds -> False];
				
(*				Append[
					Molecule @@ List @@ Molecule[newMol],
					AtomDiagramCoordinates -> coords
				];*)
				expr,
				Nothing
			]		
		],
		{n, 0, nstructs - 1}];
		(*Union @ *)res
]

iResonanceStructures[___] := $Failed;


SetAttributes[
    {
        ResonanceStructures
    },
    {ReadProtected, Protected}
]


End[] (* End Private Context *)


