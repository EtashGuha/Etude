(* ::Package:: *)

If[ StringMatchQ[$SystemID, "Windows*" | "MacOSX*"],

	(****************** QuickTime for Windows & OSX ******************)
    ImportExport`RegisterExport[
    	"QuickTime",
		System`Convert`MovieDump`ExportQuickTime,
		"Sources" -> {"Convert`CommonGraphics`", "Convert`QuickTime`"},
		"FunctionChannels" -> {"FileNames"},
		"SystemID" -> ("Windows*" | "Mac*"),
		"BinaryFormat" -> True
		(* no "DefaultElement" explicitly. Converter handles default element parsing *)
	]
	(* No RegisterExport[] for all other platforms. *)
]
