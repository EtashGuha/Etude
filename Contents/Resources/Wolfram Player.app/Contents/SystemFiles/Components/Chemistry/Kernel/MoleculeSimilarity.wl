




Begin["Chemistry`Private`MoleculeSimilarityDump`"]




$MoleculeNearestFunctionStore := (
	Unprotect @ $MoleculeNearestFunctionStore;
	$MoleculeNearestFunctionStore = Language`NewExpressionStore["MoleculeNearestFunction"];
	Protect @ $MoleculeNearestFunctionStore;
	$MoleculeNearestFunctionStore
)


(* ************************************************************************* **

                        MoleculeSimilarity

** ************************************************************************* *)

Options[ MoleculeSimilarity ] = {IncludeHydrogens -> False, "FingerprintType" -> Automatic, "SimilarityMeasure" -> "Tanimoto"}

MoleculeSimilarity[mol1_?MoleculeQ, mol2_?MoleculeQ , opts:OptionsPattern[]] :=
    Block[{res},
        res = iMoleculeSimilarity[mol1, mol2, opts];
        res /; NumericQ[res]
    ]



(* :innerFunction: *)

iMoleculeSimilarity[mol1_, mol2_, opts___?OptionQ] := Block[
	{
		im1 = getCachedMol @ mol1,
		im2 = getCachedMol @ mol2,
		fp = newInstance["Fingerprint"], fptype, smeasure
	},
	
	fptype = fingerprintTypeLookup[ OptionValue[MoleculeSimilarity, {opts}, "FingerprintType"] ] ;
	fptype = fptype /. x_String :> fingerprintTypeLookup[0];
	
	smeasure = similarityMeasureLookup[ OptionValue[MoleculeSimilarity, {opts}, SimilarityMeasure] ];
	smeasure = smeasure /. x_String :> similarityMeasureLookup[0];
    (	
		fp["setFingerprintFunction",fptype];
		fp["setSimilarityMeasure",smeasure];
		
		fp[ "fingerprintSimilarity", mol1["LibraryID"], mol2["LibraryID"]]
    )/; AllTrue[ {im1, im2, fp}, ManagedLibraryExpressionQ]

]

iMoleculeSimilarity[___] := $Failed


Options[MoleculeNearest] = Options[MoleculeSimilarity];
MoleculeNearest[mols:{__Molecule} /; And @@ Map[MoleculeQ, mols], 
	queryMol_?MoleculeQ, opts:OptionsPattern[] ] :=
	MoleculeNearest[mols, queryMol, 1, opts];
	
MoleculeNearest[mols:{__?MoleculeQ}, 
	queryMol_?MoleculeQ, numReturn_Integer, opts:OptionsPattern[] ] :=
	Block[{fp = newInstance["Fingerprint"], fptype, smeasure, idlist, ret},
		idlist = #["LibraryID"] & /@ mols;
		(
			fptype = fingerprintTypeLookup[ OptionValue["FingerprintType"] ];
			fptype = fptype /. x_String :> fingerprintTypeLookup[0];
			
			smeasure = similarityMeasureLookup[ OptionValue["SimilarityMeasure"] ];
			smeasure = smeasure /. x_String :> similarityMeasureLookup[0];
			
			fp["setFingerprintFunction",fptype];
			fp["setSimilarityMeasure",smeasure];
			
			fp[ "buildNearestFunction", idlist, Range @ Length@ idlist];
			ret = fp[ "queryNearestFunction", queryMol["LibraryID"] ,numReturn];
			mols[[ret]]
		) /; MatchQ[ idlist, {__Integer}]
	]
	
MoleculeNearest[mols:{__?MoleculeQ}, opts:OptionsPattern[] ] :=
	Block[{fp = newInstance["Fingerprint"], fptype, smeasure, idlist, ret},
		idlist = #["LibraryID"] & /@ mols;
		(
			fptype = fingerprintTypeLookup[ OptionValue["FingerprintType"] ];
			fptype = fptype /. x_String :> fingerprintTypeLookup[0];
			
			smeasure = similarityMeasureLookup[ OptionValue["SimilarityMeasure"] ];
			smeasure = smeasure /. x_String :> similarityMeasureLookup[0];
			
			fp["setFingerprintFunction",fptype];
			fp["setSimilarityMeasure",smeasure];
			
			fp[ "buildNearestFunction", idlist, Range @ Length@ idlist];
			ret = MoleculeNearestFunction[fp];
			$MoleculeNearestFunctionStore["put"[ret, 0, mols] ];
			ret
		) /; MatchQ[ idlist, {__Integer}]
	]
	
Options[MoleculeFingerprint] = Options[MoleculeNearest];
	
MoleculeFingerprint[ mol_?MoleculeQ, opts:OptionsPattern[]] := Block[
	{fp = newInstance["Fingerprint"], fptype, numbits, onbits, res, sa},
	
	fptype = fingerprintTypeLookup[ OptionValue["FingerprintType"] ];
	fptype = fptype /. x_String :> fingerprintTypeLookup[0];
	fp["setFingerprintFunction",fptype];
	fp[ "buildNearestFunction", {mol["LibraryID"]}, {1}];		
	
	numbits = fp["getNumBits"];
	onbits = fp["getOnBits", 0] + 1;
	
	res = ConstantArray[0, 2048];
	res[[onbits]] = 1;
	SparseArray@res
	
]
	
	
	
validNearestFunction[MoleculeNearestFunction[ fp_ ] ]:= ManagedLibraryExpressionQ[ fp ];
 
(mnf:MoleculeNearestFunction[fp_?ManagedLibraryExpressionQ])[mol_?MoleculeQ, numReturn_Integer:1] /; validNearestFunction[mnf ] := Module[
	{mols = $MoleculeNearestFunctionStore["get"[mnf, 0] ], molID = mol["LibraryID"], ret },
	ret = fp[ "queryNearestFunction", molID ,numReturn];
	mols[[ret]]
]


MoleculeNearestFunction /: MakeBoxes[mnf:MoleculeNearestFunction[fp_?ManagedLibraryExpressionQ], fmt_] /; validNearestFunction[Unevaluated @ mnf] := 
Module[{alwaysGrid, sometimesGrid, fpcount, fptype},
	fptype = fingerprintTypesMap[ fp[ "getFPType"]];
	fpcount = Length @ $MoleculeNearestFunctionStore["get"[mnf, 0] ];
	alwaysGrid = {
		BoxForm`SummaryItem[{"Fingerprint type: ", fptype}],
		BoxForm`SummaryItem[{"Molecules: ", fpcount}]
	};
	sometimesGrid = { };
	BoxForm`ArrangeSummaryBox[MoleculeNearestFunction, mnf, BoxForm`GenericIcon[NearestFunction], 
		alwaysGrid, sometimesGrid, fmt, "Interpretable" -> False ]
]

	 
(*	typedef enum  {
		Tanimoto,
		Cosine,
		Dice,
		Kulczynski,
		BraunBlanquet,
		Sokal,
		McConnaughey,
		Asymmetric,
		Russel
	} SimilarityMeasure;*)
	
similaritymeasuresMap = <|
	0 -> "Tanimoto", 1 -> "Cosine", 2 -> "Dice", 3 -> "Kulczynski", 4 -> "BraunBlanquet",
	5 -> "Sokal", 6 -> "McConnaughey", 7 -> "Asymmetric", 8 -> "Russel"
|>;
similarityMeasureLookup = <|
	Automatic -> 0,
	"Tanimoto" -> 0, "Cosine" -> 1, "Dice" -> 2, "Kulczynski" -> 3, "BraunBlanquet" -> 4,
	"Sokal" -> 5, "McConnaughey" -> 6, "Asymmetric" -> 7, "Russel" -> 8
|>;

(*	typedef enum  {
		RDKit,
		AtomPairs,
		TopologicalTorsions,
		MACCSKeys,
		MorganConnectivity,
		MorganFeature
	} FingerprintFlavor ;	 *)

fingerprintTypesMap = <|
		0 -> "RDKit", 1 -> "AtomPairs", 2 -> "TopologicalTorsions", 
		3 -> "MACCSKeys", 4 -> "MorganConnectivity", 5 -> "MorganFeature"
	|> ;
fingerprintTypeLookup = <|
	Automatic -> 0,
	"RDKit" -> 0, 
	"AtomPairs" -> 1, 
	"TopologicalTorsions" -> 2, 
	"MACCSKeys" -> 3, 
	"MorganConnectivity" -> 4,
	"MorganFeature" -> 5
|> ;



SetAttributes[
    {
        "similaritymeasuresMap",
        "similarityMeasureLookup",
        "fingerprintTypesMap",
        "fingerprintTypeLookup",
        iMoleculeSimilarity,
        iMoleculeNearest,
		MoleculeSimilarity,
		MoleculeNearest,
		MoleculeNearestFunction
    },
    {ReadProtected, Protected}
]


End[] (* End Private Context *)


