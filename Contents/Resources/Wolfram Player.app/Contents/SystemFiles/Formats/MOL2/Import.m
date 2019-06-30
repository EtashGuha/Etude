(* ::Package:: *)

Begin["System`Convert`MOL2Dump`"]


$Mol2AvailableElements = {
	"Molecule", "MoleculeCount", "PartialCharges","ResidueAtoms", "ResidueCharges",
	"ResidueCoordinates", "Residues","Sequence","StructureDiagram", 
	"VertexCoordinates", "VertexTypes","EdgeRules", "EdgeTypes", "Graphics3D"
};

$MolHiddenElements = {};

GetMol2Elements[___] :=
	"Elements" ->
		Sort[
			Complement[
				$Mol2AvailableElements
				,
				$MolHiddenElements
			]
		];
		
partialForms = System`ConvertersDump`Utilities`$PartialAccessForms;

ImportExport`RegisterImport[
	"MOL2",
	{
		elem : Alternatives @@ $Mol2AvailableElements :> ImportMol2[elem][All],
		{elem : Alternatives @@ $Mol2AvailableElements , part_} :> ImportMol2[elem][part],
		"Elements" :> GetMol2Elements
	},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> $Mol2AvailableElements,
	"DefaultElement" -> "Molecule",
	"SkipPostImport" -> <|
		"ResidueAtoms" -> {partialForms},
		"ResidueCharges" -> {partialForms},
		"ResidueCoordinates" -> {partialForms},
		"Residues" -> {partialForms},
		"Sequence" -> {partialForms},
		"StructureDiagram" -> {partialForms},
		"VertexCoordinates" -> {partialForms},
		"VertexTypes" -> {partialForms},
		"EdgeRules" -> {partialForms},
		"EdgeTypes" -> {partialForms},
		"Graphics3D" -> {partialForms},
		"Molecule" -> {partialForms},
		"PartialCharges" -> {partialForms}
	|>
];

End[]
