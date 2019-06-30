(* ::Package:: *)

Paclet[
	Name -> "GeneralUtilities",
	Version -> "12.0.0",
	MathematicaVersion -> "12.0+",
	Description -> "General utilities",
	Loading -> Automatic,
	Extensions -> {
		{"Kernel", 	
			HiddenImport -> None,
			Context -> {"GeneralUtilitiesLoader`", "GeneralUtilities`"}, Symbols -> {
			"System`DeleteMissing",
			"System`DisjointQ", 
			"System`IntersectingQ", 
			"System`SubsetQ",
			"System`Failure",
			"System`PowerRange",
			"System`CountDistinct",
			"System`CountDistinctBy",
			"System`DeleteDuplicatesBy",
			"System`TextString",
			"System`AssociationMap",
			"System`InsertLinebreaks",
			"System`StringPadLeft",
			"System`StringPadRight",
			"System`StringExtract",
			"System`Decapitalize",
			"System`StringRepeat",
			"System`StringRiffle",
			"System`PrintableASCIIQ",
			(* Options for TextString -- I wish these didn't have autoload stubs, but there is no
			other way of getting PacletManager to make these system symbols *)
			"System`AssociationFormat","System`ListFormat",
			"System`BooleanStrings", "System`MissingString",
			"System`TimeFormat", "System`ElidedForms",

			(*
			"System`Softmax",
			"System`MeanSquaredDistance",
			"System`MeanAbsoluteDistance",
			"System`CrossEntropy",
			*)
	
			"GeneralUtilities`MLExport",
			"GeneralUtilities`MLImport",
			"GeneralUtilities`ValidPropertyQ",
			"GeneralUtilities`InformationPanel",
			"GeneralUtilities`ComputeWithProgress",
			"GeneralUtilities`TightLabeled"

		}}
	}
]
