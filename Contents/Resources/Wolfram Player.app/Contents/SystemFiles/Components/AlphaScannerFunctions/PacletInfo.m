(* ::Package:: *)

(* Paclet Info File *)
Paclet[
	Name -> "AlphaScannerFunctions",
	Description -> "Some Wolfram|Alpha scanner utility functions ported to paclet",
	Version -> "0.2",
	MathematicaVersion -> "12.0+",
	Loading -> Automatic,
	Updating -> Automatic,
	Extensions -> {
		{
			"Kernel",
			HiddenImport -> True,
			Root->"Kernel",
			Context->{"AlphaScannerFunctions`"},
			Symbols -> {
				"AlphaScannerFunctions`AreaBetweenCurves",
				"AlphaScannerFunctions`Asymptotes",
				"AlphaScannerFunctions`CompleteSquare",
				"AlphaScannerFunctions`FunctionDiscontinuities",
				"AlphaScannerFunctions`FunctionAmplitude",
				"AlphaScannerFunctions`InflectionPoints",
				"AlphaScannerFunctions`Intercepts",
				"AlphaScannerFunctions`RepeatingDecimalToRational",
				"AlphaScannerFunctions`StationaryPoints",
				"AlphaScannerFunctions`SuggestPlotRange"
			}
		}
	}
]
