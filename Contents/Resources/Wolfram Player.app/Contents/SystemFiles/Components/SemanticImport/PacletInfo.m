(* Paclet Info File *)

Paclet[
    Name -> "SemanticImport",
    Version -> "0.1.0",
    MathematicaVersion -> "10+",
    Description->"SemanticImport transforms ordinary tabular data into a typed representation",
    Loading -> Automatic, 
    Extensions -> {
		{"Resource",
			SystemID->"MacOSX-x86-64",
			Resources -> {{"server","Binaries/MacOSX-x86-64/server"}}
		},
		{"Resource",
			SystemID->"Windows",
			Resources -> {{"server","Binaries/Windows/server.exe"}}
		},
		{"Resource",
			SystemID->"Windows-x86-64",
			Resources -> {{"server","Binaries/Windows-x86-64/server.exe"}}
		},
		{"Resource",
			SystemID->"Linux", 
			Resources -> {{"server","Binaries/Linux/server"}}
		},
		{"Resource",
			SystemID->"Linux-x86-64",
			Resources -> {{"server","Binaries/Linux-x86-64/server"}}
		},
		{"Resource",
			SystemID->"Linux-ARM",
			Resources -> {{"server","Binaries/Linux-ARM/server"}}
		},	
		{"Resource", 
			Root -> "resources",
			Resources -> {"spellings","config"}	
		},
		{"Kernel", Root-> "Kernel", Context -> {"SemanticImportLoader`", {"SemanticImport`", "SemanticImportMain.m"}},
			Symbols -> {
				"System`SemanticImport",
				"System`SemanticImportString", 
				"System`MissingDataRules", 
				"System`HeaderLines", 
				"System`ExcludedLines"
			}
		} 
	}
]


	
