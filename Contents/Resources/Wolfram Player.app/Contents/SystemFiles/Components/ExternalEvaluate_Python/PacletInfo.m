(* Paclet Info File *)

Paclet[
	Name -> "ExternalEvaluate_Python",
	Version -> "20.1.1",
	MathematicaVersion -> "11.3+",
	Extensions -> {
		{
			"Kernel",
			Root -> "ExternalEvaluate",
			Context -> {"ExternalEvaluatePython`"}
		},
		{
			"Resource",
			Root -> "ExternalEvaluate",
			Resources -> {
		    	{"System","ExternalEvaluatePython.wl"}
	    	}
		},
		{
			"Resource",
			Root -> "Resources",
			Resources -> {
		    	{"PythonREPL",        "cli.py"},
		    	{"PythonZMQTest",     "capabilities/WLTest.py"},
		    	{"PythonNumpyZMQTest","capabilities/WLNumPy.py"},
		    	{"PythonPILZMQTest",  "capabilities/WLPIL.py"}
	    	}
		}	
	}
]
