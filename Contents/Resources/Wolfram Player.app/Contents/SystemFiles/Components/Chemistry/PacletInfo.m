(* ::Package:: *)

(* Paclet Info File *)

Paclet[
    Name -> "Chemistry",
    Description -> "Chemistry tools for the Wolfram Language",
    Version -> "0.0.62",
    MathematicaVersion -> "12.0+",
    Loading -> Automatic,
    Updating -> Automatic,
    Extensions -> {
        {"LibraryLink"},
        {"JLink"},
        {"Documentation", Language -> "English"},
        {
        	"Resource",
        	Root -> "Resources",
        	Resources -> {"labelGraphicsCache"}
        },
        {
            "Kernel",
		    HiddenImport -> None,
            Root->"Kernel",
            Context->{"Chemistry`"},
            Symbols -> 
            		{
					"System`Molecule",
					"System`MoleculeQ",
					"System`Atom",
					"System`Bond",
					"System`BondQ",
					
					"System`MoleculePattern",
					"System`AtomDiagramCoordinates",
					"System`AtomCoordinates",
					"System`StereochemistryElements",
					"System`IncludeHydrogens",
					"System`IncludeAromaticBonds",
					
					
					"System`MoleculePlot", 
					"System`MoleculePlot3D", 
					"System`MoleculeValue", 
					"System`MoleculeProperty",
					"System`BondCount",
					"System`BondList",
					"System`AtomList",
					"System`AtomCount",
									     
					"System`MoleculeEquivalentQ",
					"System`MoleculeContainsQ",
					"System`FindMoleculeSubstructure",
						
					
					"System`ConnectedMoleculeComponents",
					"System`ConnectedMoleculeQ",
					
					"System`MoleculeGraph",
					"Chemistry`MoleculeRelationQ",
					
					
					"System`MoleculeModify",
					"System`ValenceErrorHandling",
					"Chemistry`NameToSMILES"
			    		
				}
            
		}
	}
]
