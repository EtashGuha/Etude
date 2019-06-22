
BeginPackage["Chemistry`"]

(* Exported system symbols *)
Chemistry`Private`$exportedSymbols = Function[Unprotect[#]; Clear[#]; #] /@ {
	System`Atom,
	System`AtomCoordinates,
	System`Bond,
	System`BondList,
	System`BondQ,
	System`ConnectedMoleculeQ,
	System`ConnectedMoleculeComponents,
	System`FindMoleculeSubstructure,
	System`IncludeAromaticBonds,
	System`IncludeHydrogens,
	System`Molecule,
	System`MoleculeContainsQ,
	System`MoleculeEquivalentQ,
	System`MoleculeGraph,
	System`MoleculeModify,
	System`MoleculePattern,
	System`MoleculePlot,
	System`MoleculePlot3D,
	System`MoleculeProperty,
	System`MoleculeQ,
	System`MoleculeValue,
	System`StereochemistryElements,
	System`AtomDiagramCoordinates,
	System`ValenceErrorHandling
}


(* symbols used by the ImportExport code*)
Chemistry`Common`ReadMoleculeFile
Chemistry`Common`ReadMoleculeFileString
Chemistry`Common`WriteMoleculeFile
Chemistry`Common`XYZMolecule


(* Used in ResourceFunction["NameToSMILES"] *)
Chemistry`NameToSMILES


(*
 Useful functions that will possibly be system symbols in the future 	
 or possibly resource functions 										
*)
Unprotect["Chemistry`*"];
ClearAll["Chemistry`*"];
Chemistry`AtomicNumber::usage = "AtomicNumber[element] returns the atomic number for element."
Chemistry`FromAtomicNumber::usage = "FromAtomicNumber[num] returns the atomic symbol for the element with atomic number num."
Chemistry`MaximumCommonSubstructure::usage = "MaximumCommonSubstructure[ mol1,mol2,..] gives a MoleculePattern representing the maximum common substructure for the given molecules."
Chemistry`MoleculeFingerprint
Chemistry`MoleculeNearest
Chemistry`MoleculeNearestFunction
Chemistry`MoleculeRelationQ::usage = "MoleculeRelationQ[\"relation\", mol1, mol2] returns True if mol1 and mol2 have the given relationship."
Chemistry`MoleculeSimilarity
Chemistry`MoleculeSubstructureCount::usage = "MoleculeSubstructureCount[mol, patt] gives the number of times patt appears in mol."
Chemistry`ResonanceStructures::usage = "ResonanceStructures[mol] gives a molecules created from mol by rearranging pi electrons and formal charges."
Chemistry`SequenceMolecule
Chemistry`MoleculePatternQ
Chemistry`SmilesToCanonicalSmiles::usage = "SmilesToCanonicalSmiles[\"smiles\"] returns a canonical version of the given SMILES string."

(* Toggles whether to use EntityValue or ChemicalData for Molecule[_Entity] *)
Chemistry`$UseEntityValue = True

Begin["`Private`"]

Unprotect["Chemistry`RDKitLink`*"];
ClearAll["Chemistry`RDKitLink`*"];
Chemistry`RDKitLink`deleteiMolecule
Chemistry`RDKitLink`iMolecule
Chemistry`RDKitLink`iMoleculeList
Chemistry`RDKitLink`newInstance
Chemistry`RDKitLink`$chemLibraryFunctions
Chemistry`RDKitLink`$MolTemplateFile


Unprotect["Chemistry`Common`*"];
ClearAll["Chemistry`Common`*"];
Chemistry`Common`addCompletion
Chemistry`Common`alignConnectedComponents
Chemistry`Common`associationInvert
Chemistry`Common`atomColorRules
Chemistry`Common`atomNameMap
Chemistry`Common`atomPropertyLookup
Chemistry`Common`bondDirMap
Chemistry`Common`bondOrderLookup
Chemistry`Common`bondOrderMap
Chemistry`Common`bondStereoLookup
Chemistry`Common`bondStereoMap
Chemistry`Common`cachedEvaluation
Chemistry`Common`cachedLibraryID
Chemistry`Common`cacheMolecule
Chemistry`Common`canonicalizeAtoms
Chemistry`Common`canonicalizeConformer
Chemistry`Common`checkMolecule
Chemistry`Common`chiralTagLookup
Chemistry`Common`chiralTypeMap
Chemistry`Common`chiralTypeLookup
Chemistry`Common`elementCommonName
Chemistry`Common`embedMol
Chemistry`Common`firstQuantity
Chemistry`Common`generateDefault3DCoordinates
Chemistry`Common`getCachedMol
Chemistry`Common`getEmbedParameters
Chemistry`Common`getiMolProperty
Chemistry`Common`getMolGraph
Chemistry`Common`getMolOption
Chemistry`Common`GraphMolecule
Chemistry`Common`has2DCoordinates
Chemistry`Common`has3DCoordinates
Chemistry`Common`hasImplicitHydrogens
Chemistry`Common`hybridizationLookup
Chemistry`Common`hybridizationMap
Chemistry`Common`iAtomValue
Chemistry`Common`InertiaTensor
Chemistry`Common`iGeometricProperty
Chemistry`Common`LoadMoleculeData
Chemistry`Common`makeEntityDownValue
Chemistry`Common`messageNoMol
Chemistry`Common`messageOnAtomsDimension
Chemistry`Common`messageOnBadAtomReference
Chemistry`Common`messageOnNotEmbedded
Chemistry`Common`MessageValidQuery
Chemistry`Common`molDataFromLibrary
Chemistry`Common`MoleculeBlock
Chemistry`Common`moleculeBox
Chemistry`Common`MoleculeSubstructureIndices
Chemistry`Common`molIcon
Chemistry`Common`numberElementLookup
Chemistry`Common`numberElementMap
Chemistry`Common`parseBonds
Chemistry`Common`ReadMoleculeFile
Chemistry`Common`replaceMoleculeOption
Chemistry`Common`SetMoleculeValueFunction
Chemistry`Common`SymmetryInformation
Chemistry`Common`validAtomicNumber
Chemistry`Common`ValidMol
Chemistry`Common`ValidQuery
Chemistry`Common`ValidQueryAtom
Chemistry`Common`ValidQueryBond
Chemistry`Common`wrapWithTargetUnit
Chemistry`Common`XYZMolecule
Chemistry`Common`$AtomInMoleculePattern
Chemistry`Common`$AtomInputPattern
Chemistry`Common`$AtomPattern
Chemistry`Common`$AtomReferencePattern
Chemistry`Common`$AtomStereoTypes
Chemistry`Common`$AtomsPattern
Chemistry`Common`$BondInputPattern
Chemistry`Common`$BondPattern
Chemistry`Common`$BondStereoTypes
Chemistry`Common`$BondTypes
Chemistry`Common`$ElementsPattern
Chemistry`Common`$EmptyMolecule
Chemistry`Common`$ExposedProperties
Chemistry`Common`$PatternBondHeads
Chemistry`Common`$SequenceTypes

Chemistry`Common`quantity
Chemistry`Common`quantityArray
Chemistry`Common`quantityArrayQ

$UseInterpreter = False;

Block[
	{
		loadingPackage = FileNameJoin[{DirectoryName[$InputFileName], #} ],
		packageSymbols = StringJoin["Chemistry`Private`",FileBaseName@#,"Dump`*"],
		$ContextPath = {"System`","Chemistry`Common`","Chemistry`RDKitLink`","Chemistry`OPSINLink`","Chemistry`"}
	},

	With[{packageSymbols=packageSymbols},
		Unprotect[packageSymbols];
		ClearAll[packageSymbols];
		Get @ loadingPackage;
	];
	
] & /@ {
	"RDKitLink.wl",
	"OPSINLink.wl",
	"Common.wl",
	"MoleculeValue.wl",(* Should be read in before PatternMatching so property lists are defined *)
	"Formatting.wl",
	"Molecule.wl",
	"MoleculeSubstructure.wl",
	"PatternMatching.wl",
	"MoleculeModify.wl",
	"MoleculeCaching.wl",
	"MoleculePlot.wl",
	"MoleculeSimilarity.wl",
	"ResonanceStructures.wl",
	"SymmetrizerLink.wl",
	"MoleculeGeometry.wl",
	"MoleculeGraph.wl",
	"MoleculeFileParse.wl"
}	

With[{opts=Options[#]}, Options[#] = SortBy[First] @ DeleteDuplicatesBy[First] @ opts]& /@ Chemistry`Private`$exportedSymbols;

SetAttributes[ #, {Protected, ReadProtected}] & /@ Chemistry`Private`$exportedSymbols;
SetAttributes[ #, {Protected, ReadProtected}] & /@ Names["Chemistry`*"];


Chemistry`Private`MoleculeValueDump`updateMolValueAutoComplete[]

(* TODO: move these into the proper file *)
Molecule::discon = "Warning: coordinate generation for disconnected structures is experimental."


End[] (*private*)
EndPackage[]
