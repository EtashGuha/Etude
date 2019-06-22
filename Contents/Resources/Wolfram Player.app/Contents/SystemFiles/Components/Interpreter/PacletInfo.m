(* Paclet Info File *)

(* created 2014/10/21*)

Paclet[
	Name -> "Interpreter",
Version -> "1.3.70.0",
	MathematicaVersion -> "11.3+",
	Description -> "Interpretation of strings",
	Creator -> "Riccardo Di Virgilio <riccardod@wolfram.com>, Carlo Barbieri <carlob@wolfram.com>, Chiara Basile <chiarab@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{"Kernel", 
			HiddenImport -> True,
			Context -> {"Interpreter`"}, 
			Symbols -> {
				"System`$InterpreterTypes",
				"System`AnySubset",
				"System`CompoundElement",
				"System`DelimitedArray",
				"System`DelimitedSequence", 
				"System`GeoLocation", 
				"System`ImportOptions",
				"System`Interpreter", 
				"System`RectangularRepeatingElement",
				"System`RepeatingElement",
				"System`Restricted", 
				"System`SquareRepeatingElement",
				"Interpreter`ArrayRepeatingElement",
				"Interpreter`InterpreterObject"
			}
		},
		{"JLink"},
		{"Resource", Root -> ".", Resources -> {"MetaData"}}
	}
]