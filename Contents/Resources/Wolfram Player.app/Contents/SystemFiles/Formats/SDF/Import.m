(* ::Package:: *)

Begin["System`Convert`MolDump`"]


$SdfAvailableElements = {"EdgeRules", "EdgeTypes", "FormalCharges",
 				"Graphics3D", "Header", "MassNumbers", "Metadata", "Molecule", "MoleculeCount",
 				"StructureDiagram", "VertexCoordinates", "VertexTypes"};

$SdfHiddenElements = {};

GetSDFElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				$SdfAvailableElements
				,
				$SdfHiddenElements
			]
		];

partialForms = System`ConvertersDump`Utilities`$PartialAccessForms;

ImportExport`RegisterImport[
	"SDF",
	{
		elem : Alternatives @@ $SdfAvailableElements :> ImportSDF[elem][All],
		{elem : Alternatives @@ $SdfAvailableElements , part_} :> ImportSDF[elem][part],
		"Elements" :> GetSDFElements
	},
	"FunctionChannels" -> {"FileNames"},
	"Sources" -> ImportExport`DefaultSources["Mol"],
	"AvailableElements" -> $SdfAvailableElements,
	"DefaultElement" -> "Molecule",
	"SkipPostImport" -> <|
		"StructureDiagram" -> {partialForms},
		"VertexCoordinates" -> {partialForms},
		"VertexTypes" -> {partialForms},
		"EdgeRules" -> {partialForms},
		"EdgeTypes" -> {partialForms},
		"Graphics3D" -> {partialForms},
		"Molecule" -> {partialForms},
		"Metadata" -> {partialForms}
	|>
]

End[]

