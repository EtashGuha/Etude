(* Paclet Info File *)

Paclet[
	Name -> "ExternalEvaluate_NodeJS",
	Version -> "20.1.1",
	MathematicaVersion -> "11.3+",
	Extensions -> {
		{
			"Resource",
			Root -> "ExternalEvaluate",
			Resources -> {
		    	{"System","NodeJS.wl"}
	    	}
		},
		{
			"Resource",
			Root -> "Resources",
			Resources -> {
		    	{"NodeJSREPL","eval.js"},
		    	{"NodeJSZMQTest","test.js"},
		    	{"VariableNameRegex","regex.txt"}
	    	}
		}
	}
]
