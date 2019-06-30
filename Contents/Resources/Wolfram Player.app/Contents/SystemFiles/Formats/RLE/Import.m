(* ::Package:: *)

Begin["System`Convert`RLEDump`"]

$RLEAvailableElements = SortBy[{"Comments", "Data", "Graphics", "Image", "ImageSize", "Rule"}, ToString];

(*Returns the list of documented elements*)
GetRLEElements[___] := "Elements" -> $RLEAvailableElements

ImportExport`RegisterImport[
	"RLE"
	,
	{
		elem : (
			(* Data representation elements *)
			"Data" 			  |
			"Graphics" 		  |
			"Image"     	  |
			(* Meta-data elements *)
			"Rule" 			  | (* Added in 12.0 *)
			"Comments" 		  |
			"ImageSize"
		) :> ImportRLEElem[elem],

		(* List of documented elements *)
		"Elements"  :> GetRLEElements
	}
	,
	"AvailableElements" -> $RLEAvailableElements,
	"DefaultElement"    -> "Image",
	"Sources"           -> { "Convert`RLE`" },
	"BinaryFormat"      -> True
]


End[]
