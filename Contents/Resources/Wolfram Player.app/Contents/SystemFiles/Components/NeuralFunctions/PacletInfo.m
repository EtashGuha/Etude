Paclet[
	Name -> "NeuralFunctions",
	Version -> "12.0.0",
	WolframVersion -> "12+",
	Description -> "Functions built out of neural networks",
	Creator -> "Giulio Alessandrini <giulioa@wolfram.com>; Carlo Giacometti <carlog@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{
			"Kernel",
			Root -> "Kernel",
			Context -> {"NeuralFunctions`"},
			HiddenImport -> True,
			Symbols -> {
				"System`ImageRestyle",

				"System`ImageIdentify",
				"System`ImageInstanceQ",

				"System`ImageContents",
				"System`ImageBoundingBoxes",
				"System`ImageCases",
				"System`ImagePosition",
				"System`ImageContainsQ",

				"System`SpeechRecognize",

				"System`FacialFeatures",

				"System`AudioIdentify",

				"NeuralFunctions`FindText"
			}
		},
		{
			"Resource",
			Root -> "Resources",
			Resources -> {
				"ObjectDetection",
				"AudioClassification"
			}
		}
	}
]
