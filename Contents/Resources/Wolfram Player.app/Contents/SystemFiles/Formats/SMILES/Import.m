(* ::Package:: *)

Begin["System`Convert`SMILESDump`"]

$SMILESAvailableElements = {
	"Molecule", "Metadata", "MoleculeCount", 
	"EdgeRules", "EdgeTypes", "FormalCharges", "VertexTypes"
};

$SMILESHiddenElements = {};

GetSMILESElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				$SMILESAvailableElements
				,
				$SMILESHiddenElements
			]
		];
		
partialForms = System`ConvertersDump`Utilities`$PartialAccessForms;

ImportExport`RegisterImport[
	"SMILES",
	{
		elem : Alternatives @@ $SMILESAvailableElements :> ImportSMILES[elem][All],
		{elem : Alternatives @@ $SMILESAvailableElements, part_} :> ImportSMILES[elem][part],
		"Elements" :> GetSMILESElements
	},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> $SMILESAvailableElements,
	"DefaultElement" -> "Molecule",
	"Sources" -> ImportExport`DefaultSources["SMILES"],
	"SkipPostImport" -> AssociationMap[
		{partialForms}&,
		DeleteCases[$SMILESAvailableElements,"MoleculeCount"]
	]
];

End[]
