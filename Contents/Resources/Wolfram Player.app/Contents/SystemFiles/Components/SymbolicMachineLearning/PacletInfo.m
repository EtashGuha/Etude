Paclet[
	Name -> "SymbolicMachineLearning",
	Version -> "1.1.0",
	MathematicaVersion -> "10.2+",
	Description -> "FindSimpleFit and FindSimpleDistribution function",
	Creator -> "Giorgia Fortuna, <giorgiaf@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{
			"Kernel", 
			Symbols -> {
				"System`FindDistribution",
				"System`FindFormula"
				},
			Context -> {
				"SymbolicMachineLearningLoader`", 
				"SymbolicMachineLearning`"
			}
		}
	(*, Root-> "Kernel" (*to specify where the .m files are*) *)
	}
]
