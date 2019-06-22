(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Developer maintains this list of symbols.
   SemanticImport does its autoloading defs via sysinit.m, so this list must match the DeclareLoad call in that file.
*)

PacletManager`Package`loadWolframLanguageCode[
	"SemanticImport", 
	"SemanticImport`", 
	ParentDirectory[DirectoryName[$InputFileName]], 
	"Kernel/SemanticImport.m",
	"AutoUpdate" -> True,
	"AutoloadSymbols" -> {
		"System`SemanticImport", 
		"System`SemanticImportString", 
		"System`MissingDataRules",
		"System`HeaderLines", 
		"System`ExcludedLines", 
		"System`ColumnSpans"
	},
	"HiddenImports" -> {
		"Dataset`", 
		"PacletManager`", 
		"Interpreter`"
	}
]