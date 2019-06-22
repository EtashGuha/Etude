(* ::Package:: *)

Begin["System`Convert`PDBDump`"]


ImportExport`RegisterImport[
 "PDB",
 {
	"Elements" :> getPDBElements,
	ImportPDB
 },
 {
    "Graphics3D":> ImportExport`MoleculePlot3D
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"AdditionalAtoms", "AdditionalCoordinates",
			"AdditionalCoordinatesList", "AdditionalIndex", "AdditionalResidues",
			"Authors", "Comments", "DepositionDate", "Graphics3D", "Organism",
			"PDBClassification", "PDBID", "References", "ResidueAtoms",
			"ResidueChainLabels", "ResidueCoordinates", "ResidueCoordinatesList",
			"ResidueIndex", "ResidueRoles", "Residues", "Resolution",
			"SecondaryStructure", "Sequence", "Title", "VertexCoordinates",
			"VertexCoordinatesList", "VertexTypes"},
 "DefaultElement" -> "Graphics3D"
]


End[]
