Paclet[
	Name -> "HTTPHandling",
	Version -> "1.0.0",
	MathematicaVersion -> "10+",
	Description -> "HTTPHandling Library",
	Creator -> "Riccardo Di Virgilio <riccardod@wolfram.com>, Ian Johnson <ijohnson@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{"Resource", 
			Root -> "Resources", 
			Resources -> {{"Server", "Binaries"}}
		},
		{"Kernel", 
			Context      -> {"HTTPHandling`"}, 
            HiddenImport -> True, 
            Symbols      -> {
				"HTTPHandling`WebServer",
				"HTTPHandling`StartWebServer"
			}
		}
	}
]
