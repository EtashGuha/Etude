(* ::Package:: *)

Begin["System`Convert`XYZDump`"]

$XYZAvailableElements = {
	"Graphics3D", "VertexCoordinates", "VertexTypes", "Molecule"
};

$XYZHiddenElements = {};

GetXYZElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				$XYZAvailableElements
				,
				$XYZHiddenElements
			]
		];

ImportExport`RegisterImport[
	"XYZ",
	{
		elem : Alternatives @@ $XYZAvailableElements :> ImportXYZ[elem],
		"Elements" :> GetXYZElements
	},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> $XYZAvailableElements,
	"DefaultElement" -> "Molecule",
	"Sources" -> ImportExport`DefaultSources["XYZ"]
];

End[]