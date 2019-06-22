(* 
 *
 *	ImageRestyle
 *
 *  Fast method
 *
 *	Created by Matteo Salvarezza
 *
 *)

PackageScope["fastStyleTransfer"]

fastStyleTransfer[content_, styleBlend_, styleWeight_] := Scope[

	adaInModel = GetNetModel["AdaIN-Style Trained on MS-COCO and Painter by Numbers Data"];
	mean = Normal[NetExtract[adaInModel, "Content"]]["MeanImage"];
	adainEncoder = NetExtract[adaInModel, "encoder_content"];
	adainDecoder = NetExtract[adaInModel, "decoder"];

	cData = ImageData[toMinSize[content, 16], Interleaving -> False] - mean;
	sData = Map[ ImageData[toMinSize[#, 16],   Interleaving -> False] - mean &, styleBlend[[All, 2]] ];

	(* drown content in noise to fake more style presence *)
	If[styleWeight > 0.5,
		BlockRandom[
			SeedRandom[1];
			noise = RandomReal[1, Dimensions@cData];
		];
		noiseAlpha = (styleWeight - 0.5)*2;
		cData = noiseAlpha*noise + (1 - noiseAlpha)*cData;
	];

	resized = NetReplacePart[adainEncoder, "Input" -> Dimensions@cData];
	cFeats  = Normal @ SafeNetEvaluate @ resized[cData, TargetDevice -> GetOption[TargetDevice]];

	sFeats = {};
	Do[
		resized = NetReplacePart[adainEncoder, "Input" -> Dimensions@sData[[i]]];
		AppendTo[sFeats, 
			Normal @ SafeNetEvaluate @ resized[sData[[i]], TargetDevice -> GetOption[TargetDevice]]
		];
		,	
		{i, Length@styleBlend}
	];

	encoded = adaIN[cFeats, #]& /@ sFeats;  (* nested List now *)
	encoded = Total[encoded * styleBlend[[All, 1]]];
	
	(* mix encoded with cFeats for less style presence *)
	If[ styleWeight < 0.5,
		contentAlpha = 2*styleWeight;
		encoded = contentAlpha*encoded + (1 - contentAlpha)*cFeats;
	];

	resized = NetReplacePart[adainDecoder, "Input" -> Dimensions@encoded];
	netOutput = SafeNetEvaluate@resized[encoded, TargetDevice -> GetOption[TargetDevice]];
	output = Image[netOutput, Interleaving -> False];

	(* Final image dimensions may differ by some pixel *)
	ImageResize[output, ImageDimensions@content]
]

toMinSize[img_, min_] := ImageResize[img, Max[#, min]& /@ ImageDimensions[img]]

adaIN[cFeats_, sFeats_] := Scope[
	
	(* 1 -> content, 2 -> style *)

	flat = Flatten[#, {{1}, {2, 3}}]& /@ {cFeats, sFeats};

	mean = Map[Mean, flat, {2}];
	std  = Sqrt@Map[Variance, flat, {2}] + 10^-6.;

	final = (flat[[1]] - mean[[1]])/std[[1]] * std[[2]] + mean[[2]];

	ArrayReshape[final, Dimensions@cFeats]
]