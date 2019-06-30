(* 
 *
 *	ImageRestyle
 *
 *  Slow method
 *
 *	Created by Matteo Salvarezza
 *
 *)

$contentLayers = {"relu4_2"};
$styleLayers = {"relu1_2", "relu3_1", "relu4_1"};

$contentLayersWeights = {1};
$styleLayersWeights = {0.35, 0.1, 1.3};
$tvWeight = 6*10^5;

$toYIQ = {{0.299,0.587,0.114}, {0.596,-0.274,-0.322}, {0.211,-0.523,0.312}};
$toRGB = {{1,0.956,0.621}, {1,-0.272,-0.647}, {1,-1.106,1.703}};

(* mean image for VGG-16 *)
$mean = {0.4850196078431373`, 0.457956862745098`, 0.4076039215686274`};

PackageScope["slowStyleTransfer"]

slowStyleTransfer[content_, styleBlend_, styleWeight_] := Scope[

	iContent = toMinSize[content, 17];

	weights = Join[
		(1 - styleWeight)*AssociationThread["rescale_content_"<>#&/@$contentLayers, $contentLayersWeights],
		styleWeight   *   AssociationThread["rescale_style_"<>#&/@$styleLayers,     $styleLayersWeights],
		<|"rescale_tv" -> $tvWeight|>
	];		

	(* set up feature extraction net & content/style features *)
	contentFeatures = extractContentFeatures[
		ImageData[iContent, Interleaving -> False]
	];
	styleFeatures = extractStyleGrams[
		ImageData[toMinSize[#, 17]  , Interleaving -> False]& /@ styleBlend[[All,2]], 
		styleBlend[[All,1]]
	]; 

	styleTransferNet = createStyleTransferNet[
		Prepend[Reverse@ImageDimensions@iContent, 3],
		contentFeatures,
		styleFeatures,
		weights
	];

	output = optimize[styleTransferNet, iContent];

	ImageResize[output, ImageDimensions@content]
]

toMinSize[img_, min_] := Scope[
	dims = ImageDimensions[img];
	oldMin = Min[dims];
	newMin = Max[oldMin, min];
	newDims = Ceiling[dims / oldMin * newMin];
	ImageResize[img, newDims]
]

toMaxSize[array_, max_] := Scope[
	dims = Rest@Dimensions[array];
	oldMax = Max[dims];
	newMax = Min[oldMax, max];
	newDims = Ceiling[dims / oldMax * newMax];
	ArrayResample[array, Prepend[newDims, 3]]
]

optimize[net_, content_] := Scope[

	progressFraction = 0;
	startTime = SessionTime[];
	timeElapsed = 0;
	loss = oldLoss = None;
	lossGradient = None;
	stop = False;

	lumTransfer = GetOption[PreserveColor] && GetOption["LuminanceTransfer"];

	contentIQ = None;
	If[lumTransfer,
		contentIQ = Map[$toYIQ.# &, ImageData[content], {2}][[All, All, 2;;]];
		contentIQ = Transpose[contentIQ, {2, 3, 1}]
	];
    monitorImage = makeMonitorImage[
    	Normal @ NetExtract[net, {"target", "Array"}],
    	contentIQ
    ];

	makeProgressBox[] := InformationPanel[
		"Restyling Progress",
		{
			Center :> ProgressIndicator[progressFraction],
			"Progress" :> Row[{Round[progressFraction * 100], "%"}],
			"Time elapsed" :> TimeString[timeElapsed],
			"Loss" :> SetPrecision[loss, 3],
			"Loss gradient" :> SetPrecision[lossGradient, 3],
			Center :> Image[monitorImage, ImageSize -> All],
			Right -> NiceButton["Stop", stop = True]
		},
		UpdateInterval -> If[TrueQ@lumTransfer, 0.75, 0.5]
	];

	updateProgress = Function[ 
    	monitorImage = makeMonitorImage[
    		Normal @ NetExtract[#Net, {"target", "Array"}],
    		contentIQ
    	];
    	timeElapsed = SessionTime[] - startTime;
    	progressFraction = N[#Round/GetOption["Iterations"]];  
    	oldLoss = loss;
    	loss = #RoundLoss; 
    	lossGradient = loss - oldLoss;
    	If[stop, "StopTraining"]
    ];

	If[
		$Notebooks && $LicenseType =!= "Player" && 
			$LicenseType =!= "Player Pro" && !$CloudEvaluation, 
		progressBox = PrintTemporary@makeProgressBox[];
	];

	trained = Quiet @ NetTrain[
 		net, 
 		<||> &,
 		LossFunction -> "Output",
 		BatchSize -> 1,
 		MaxTrainingRounds -> GetOption["Iterations"],
 		LearningRateMultipliers -> {"target" -> 1, _ -> None},
 		Method -> {"ADAM", "LearningRate" -> GetOption["StepSize"], "LearningRateSchedule" -> (1 &)},
    	TrainingProgressReporting -> None,
    	TrainingProgressFunction -> updateProgress,
 		TargetDevice -> GetOption[TargetDevice]
 	]; 

 	If[$Notebooks, 
 		NotebookDelete[progressBox]
 	];

 	If[trained === $Aborted, Abort[]];

	If[Head[trained] =!= NetGraph,
		DBPrint[StringTemplate["`1`: optimization failed"][ToString[$FunctionName]]];
		$AllowFailure ^= True;
		ThrowFailure["interr2"]
	];

	Image[Normal @ NetExtract[trained, {"target", "Array"}] + $mean, Interleaving -> False]
]

makeMonitorImage[array_, contentIQ_] := Scope[
	conformed = Which[
		Max@Dimensions[array] < 500, array,
		True, toMaxSize[array, 500]
	];
	If[contentIQ =!= None,
		Image[ 
			$toRGB . Prepend[contentIQ, Total[(conformed + $mean) * First@$toYIQ]], 
			Interleaving -> False
		],
		Image[conformed + $mean, Interleaving -> False]
	]
]

(* ------------ Tools ------------ *)

createStyleTransferNet[targetSize_, contentFeatures_, styleFeatures_, finalWeights_] := Scope[

	allLayers = Union[$contentLayers, $styleLayers];

	extractor = NetReplacePart[
		Take[GetNetModel["ImageRestyleChoppedVGG16"], endLayer[allLayers]], 
		"Input" -> targetSize
	];
	extractor = openOutPorts[extractor, allLayers];
	layerSizes = NeuralNetworks`NetOutputs@extractor;

	contentLosses = Association@@Map[
		Rule["content_"<>#, contentLoss@contentFeatures@#]&, 
		$contentLayers
	];
	styleLosses = Association@@Map[
		Rule[ "style_"<>#, styleLoss[Normal @ styleFeatures[#], layerSizes["out_"<>#]] ]&, 
		$styleLayers
	];
	allLosses = Union[contentLosses, styleLosses, <|"tv" -> tvLoss[targetSize]|>];

	allRescales = Map[ 
		Function[ factor, ElementwiseLayer[factor*#&] ],
		finalWeights
	];
	targetInitial = RandomReal[{-0.1, 0.1}, targetSize];

	styleTransferNet = NetGraph[
		Join[
			<|"target" -> ConstantArrayLayer["Array" -> targetInitial]|>,
			<|"extractor" -> extractor|>,
			allLosses,
			allRescales,
			<|"tot_loss" -> TotalLayer[]|>
		],
		Join[
			{"target" -> "extractor"},
			Map[NetPort["extractor", "out_"<>#] -> "content_"<># -> "rescale_content_"<># &, $contentLayers],
			Map[NetPort["extractor", "out_"<>#] -> "style_"<>#   -> "rescale_style_"<># &,    $styleLayers],
			{"target" -> "tv" -> "rescale_tv"},
			{Keys[allRescales] -> "tot_loss"}
		]
	];

	styleTransferNet
]

extractContentFeatures[data_] := Scope[

	extractor = NetReplacePart[
		Take[GetNetModel["ImageRestyleChoppedVGG16"], endLayer[$contentLayers]], 
		"Input" -> Dimensions@data
	];
	extractor = openOutPorts[extractor, $contentLayers];

	features = SafeNetEvaluate@extractor[data - $mean, TargetDevice -> GetOption[TargetDevice]];
	(* we want ot output an association, no matter what *)
	If[!AssociationQ[features], 
		features = <|"out_"<>First@$contentLayers -> features|>
	];
	features = KeyMap[ StringDrop[#, 4]&, features ];

	features
]


extractStyleGrams[stylesData_, blendWeights_] := Scope[

	extractor = openOutPorts[
		Take[GetNetModel["ImageRestyleChoppedVGG16"], endLayer[$styleLayers]],
		$styleLayers
	];

	allGrams = {};
	Do[
		extractor = NetReplacePart[extractor, "Input" -> Dimensions@stylesData[[i]]];
		features = SafeNetEvaluate@extractor[stylesData[[i]] - $mean, TargetDevice -> GetOption[TargetDevice]];
		If[!AssociationQ[features], 
			features = <|"out_"<>First@$styleLayers -> features|>
		];
		grams = gramMatrix[Dimensions@#][#]& /@ features;
		AppendTo[allGrams, grams * blendWeights[[i]]]
		,	
		{i, Length@stylesData}
	];
	wSum = Total[blendWeights];
	If[wSum == 0, wSum = 1];
	grams = Total[allGrams] / wSum;

	(* we want ot output an association, no matter what *)
	If[!AssociationQ[grams], 
		grams = <|"out_"<>First@$styleLayers -> First@grams|>
	];
	grams = KeyMap[ StringDrop[#, 4]&, grams ];

	grams
]

endLayer[layers_] := Part[
	layers,
	First@Ordering[Position[Keys@Normal[GetNetModel["ImageRestyleChoppedVGG16"]], #] & /@ layers, -1]
];

openOutPorts[net_, layers_] := NetGraph[
	Normal@net, 
	Join[
		Rule @@@ Partition[Keys@Normal@net, 2, 1],
		Thread[ layers -> Map[NetPort["out_"<>#]&, layers] ]
	]
]

(* ------------ Losses ------------ *)

(* This needs to propapagate the gradient back 
   to both input ports for tvLoss, hence the 
   built-in MeanSquaredLossLayer can't be used *)
meanSquaredLoss := NetGraph[
	{
		ThreadingLayer[(#1 - #2)^2&], 
		AggregationLayer[Mean, "Levels" -> All]
	},
 	{NetPort["1"] -> 1, NetPort["2"] -> 1 -> 2}
]

gramMatrix[size_] := NetGraph[
	{
		FlattenLayer[-1], 
		TransposeLayer[1 -> 2], 
		DotLayer[], 
		ElementwiseLayer[# / Times@@Rest@size &]
	},
	{1 -> 3, 1 -> 2 -> 3 -> 4}
]

contentLoss[contentFeatures_] := NetGraph[
	{
		ConstantArrayLayer["Array" -> contentFeatures],
		meanSquaredLoss
	},
	{1 -> NetPort[2, "1"]}
]

styleLoss[styleFeatures_, targetSize_] := NetGraph[
   {
		ConstantArrayLayer["Array" -> styleFeatures], 
		gramMatrix[targetSize], 
		meanSquaredLoss,
		ElementwiseLayer[Sqrt]
	}
   ,
   {1 -> NetPort[3, "1"], 2 -> NetPort[3, "2"], 3 -> 4}
]

tvLoss[size_] := NetGraph[
	<|
		"dx1" -> PaddingLayer[{{0,0}, {0,0}, {1,0}}, "Padding" -> "Fixed"],
		"dx2" -> PaddingLayer[{{0,0}, {0,0}, {0,1}}, "Padding" -> "Fixed"],
		"dy1" -> PaddingLayer[{{0,0}, {1,0}, {0,0}}, "Padding" -> "Fixed"],
		"dy2" -> PaddingLayer[{{0,0}, {0,1}, {0,0}}, "Padding" -> "Fixed"],
		"tot_x" -> meanSquaredLoss,
		"tot_y" -> meanSquaredLoss,
		"tot" -> ThreadingLayer[(#1 + #2)/2.&]				
	|>,
	{
		"dx1" -> NetPort["tot_x", "1"], "dx2" -> NetPort["tot_x", "2"],
		"dy1" -> NetPort["tot_y", "1"], "dy2" -> NetPort["tot_y", "2"],
		{"tot_x", "tot_y"} -> "tot"	
	}

]
