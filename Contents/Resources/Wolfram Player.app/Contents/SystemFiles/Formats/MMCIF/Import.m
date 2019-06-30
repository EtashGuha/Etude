(* ::Package:: *)

Begin["System`Convert`MMCIFDump`"]


ImportExport`RegisterImport[
 "MMCIF",
 {
	"Elements" :> getMMCIFElements,
	ImportMMCIF
  },
 {
    "Graphics3D":> ImportExport`MoleculePlot3D
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"AdditionalAtoms", "AdditionalCoordinates",
		"AdditionalCoordinatesList", "AdditionalResidues", "Authors",
		"DepositionDate", "Graphics3D", "PDBClassification", "PDBID",
		"References", "ResidueAtoms", "ResidueChainLabels",
		"ResidueCoordinates", "ResidueCoordinatesList", "ResidueRoles",
		"Residues", "Resolution", "SecondaryStructure", "Sequence", "Title",
		"VertexCoordinates", "VertexCoordinatesList", "VertexTypes"},
 "DefaultElement" -> "Graphics3D"
]


End[]
