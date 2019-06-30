Paclet[
	Name -> "CloudExpression",
	Version -> "0.1.0",
	MathematicaVersion -> "10+",
	Description -> "Efficiently store expressions in the Wolfram Cloud",
	Loading -> Automatic,
	Extensions -> {
		{	
			"Kernel", 
			Context -> {"CloudExpressionLoader`", "CloudExpression`"}, 
			Symbols -> {
				"System`CloudExpression",
				"System`CreateCloudExpression",
				"System`DeleteCloudExpression",
				"System`CloudExpressions",
				"System`PartProtection",
				"System`$CloudExpressionBase"
			}
		}
	}
]
