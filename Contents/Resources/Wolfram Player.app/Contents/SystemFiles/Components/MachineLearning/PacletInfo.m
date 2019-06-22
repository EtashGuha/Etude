Paclet[
	Name -> "MachineLearning",
	Version -> "1.1.0",
	MathematicaVersion -> "11.3+",
	Description -> "Automatic machine learning functionality",
	Creator -> "Etienne Bernard <etienneb@wolfram.com>, Sebastian Bodenstein <sebastianb@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{
			"Resource", 
			Root -> "Resources", 
			Resources -> {"Binaries", "Unicode", "SentenceSplitter", "Libraries"}
		},
		{
			"Resource",
			Root -> "Resources",
			SystemID -> "Windows",
			Resources -> {{"KenLM", "Binaries/Windows/lmplz.exe"}}
		},
		{
			"Resource",
			Root -> "Resources",
			SystemID -> "Windows-x86-64",
			Resources -> {{"KenLM", "Binaries/Windows-x86-64/lmplz.exe"}}
		},
		{
			"Resource",
			Root -> "Resources",
			SystemID -> "Linux",
			Resources -> {{"KenLM", "Binaries/Linux/lmplz"}}
		},
		{
			"Resource",
			Root -> "Resources",
			SystemID -> "Linux-x86-64",
			Resources -> {{"KenLM", "Binaries/Linux-x86-64/lmplz"}}
		},
		{
			"Resource",
			Root -> "Resources",
			SystemID -> "Linux-ARM",
			Resources -> {{"KenLM", "Binaries/Linux-ARM/lmplz"}}
		},
		{
			"Resource",
			Root -> "Resources",
			SystemID -> "MacOSX-x86-64",
			Resources -> {{"KenLM", "Binaries/MacOSX-x86-64/lmplz"}}
		},
		{
			"Kernel", 
			Context -> {"MachineLearningLoader`", "MachineLearning`"}, 
			Symbols -> {
				"System`Classify", 
				"System`ClassifierFunction", 
				"System`ClassifierMeasurements",
				"System`ClassifierMeasurementsObject",
				"System`ClassifierInformation",  (* obsolete since 12 *)

				
				"System`Predict",
				"System`PredictorFunction",
				"System`PredictorMeasurements",
				"System`PredictorMeasurementsObject",
				"System`PredictorInformation", (* obsolete since 12 *)
				
				
				"System`DimensionReduction",
				"System`DimensionReduce",
				"System`DimensionReducerFunction",

				"System`DistanceMatrix",
				"System`FeatureNearest",
				
				"System`ClusterClassify",
				
				"System`Dendrogram",
				"System`ClusteringTree",
				
				"System`FeatureExtract",
				"System`FeatureExtraction",
				"System`FeatureExtractorFunction",
				"System`FeatureDistance",

				"System`FindClusters",
				"System`ClusteringComponents",
				
				"System`BayesianMinimization",
				"System`BayesianMaximization",
				"System`BayesianMinimizationObject",
				"System`BayesianMaximizationObject",
				
				"System`ActiveClassification",
				"System`ActiveClassificationObject",
				"System`ActivePrediction",
				"System`ActivePredictionObject",
				
				"System`LearnDistribution",
				"System`LearnedDistribution",
				"System`SynthesizeMissingValues",
				"System`RarerProbability",

				"System`AnomalyDetection",
				"System`DeleteAnomalies",
				"System`AnomalyDetectorFunction",
				"System`FindAnomalies",

				"System`SequencePredict",
				"System`SequencePredictorFunction",
				
				"System`ComputeUncertainty"
			}
		}
	}
]


