(* ::Package:: *)

Begin["System`Convert`MolDump`"]

$MolAvailableElements = {
	"Molecule", "EdgeRules", "EdgeTypes", "FormalCharges", "Graphics3D", "Header",
	"MassNumbers", "StructureDiagram", "VertexCoordinates", "VertexTypes"
};

$MolHiddenElements = {};

GetMolElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				$MolAvailableElements
				,
				$MolHiddenElements
			]
		];

ImportExport`RegisterImport[
	"MOL",
	{
		elem : Alternatives @@ $MolAvailableElements :> ImportMol[elem],
		"Elements" :> GetMolElements
	},
	"FunctionChannels" -> {"FileNames"},
	"AvailableElements" -> $MolAvailableElements,
	"DefaultElement" -> "Molecule",
	"Sources" -> ImportExport`DefaultSources["Mol"]
];

End[]
