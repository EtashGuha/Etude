(* Functions *)
PackageExport["FindText"]

Options[FindText] = {
	PerformanceGoal -> Automatic
} // SortBy[ToString];

hiddenOptions = {
	"InputDimensions" -> Automatic,
	"BinarizationThreshold" -> Automatic
} // SortBy[ToString];

DefineFunction[FindText, detectionRouter, 1, "ExtraOptions" -> hiddenOptions];

detectionRouter[args_, opts_] :=
	Block[
		{
			img, res, binarize = Automatic, performance = Automatic, inDims = Automatic
		},

		(* image parsing *)
		If[!Image`Utilities`SetImage2D[img, args[[1]]],
			ThrowFailure["imginv", args[[1]]]
		];
		If[ImageColorSpace[img] =!= "RGB",
			img = ColorConvert[img, "RGB"]
		];
		If[Image`ImageInformation[img, "Transparency"],
			img = RemoveAlphaChannel[img, White]
		];

		(* option parsing *)
		performance = GetOption[PerformanceGoal];
		If[
			And[!MatchQ[performance, Automatic],
				Or[!StringQ[performance],
					!StringMatchQ[performance, "Speed" | "Quality"]
				]
			],
			ThrowFailure["mlbdpg", performance]
		];

		binarize = GetOption["BinarizationThreshold"];
		binarize =
			Replace[binarize,
				{
					Automatic :> .5,
					n_ /; TrueQ[0 <= n <= 1] :> n,
					_ :> ThrowFailure["binthrs", binarize]
				}
			];

		inDims = GetOption["InputDimensions"];
		inDims =
			Replace[
				inDims,
				{
					Automatic :> {500, 500},
					n_?Internal`PositiveMachineIntegerQ :> {n, n},
					list : {n_?Internal`PositiveMachineIntegerQ, n_?Internal`PositiveMachineIntegerQ} :> list,
					_ :> ThrowFailure["inptdmns", inDims]
				}
			];

		Switch[performance,
			"Speed",
			inDims = {300, 300}
			,
			"Quality",
			inDims = {1300, 1300}
		];

		(* network initialization *)
		res = intializeNetworks[inDims];

		If[!FailureQ[res],
			(* detection *)
			res = detectText[img, inDims, binarize];
		];

		(* output creation *)
		detectionResult[$FunctionName]
	]

detectionResult[FindText] :=
	Which[
		MatchQ[res, {_Rectangle ..}],
		res
		,
		MatchQ[res, {}],
		Missing["NotRecognized"]
		,
		True,
		ThrowFailure["interr2"]
	]

(****************************************************************************
    						Detection framework
****************************************************************************)
$networkCombined; (* This contains the combination of 2 independent parts of the network in a graph structure *)
$netP1;    		  (* This is for storing the first part of uninitialized  network *)
$netP2;           (* This is for storing the second part of uninitialized  network *)
$netSlice;        (* This is for storing the MXNet crop layer interpretation in WL *)

intializeNetworks[inDims_] :=
	Module[
		{
			deconvDims
		},

		$networkCombined = Quiet[GetNetModel["FCN Trained on ICDAR and MSRA Data", "EvaluationNet"]];
		If[!MatchQ[Head[$networkCombined], NetGraph],
			DBPrint["Failed to load the network."];
			Return[$Failed];
		];

		$netP1 = NetExtract[$networkCombined, "Network1"];
		$netP2 = NetExtract[$networkCombined, "Network2"];

		$netP1 = NetReplacePart[$netP1, "Input" -> NetEncoder[{"Image", inDims, ColorSpace -> "RGB"}]];
		deconvDims = NetExtract[$netP1, {"upsample_16", "Output"}];

		$netSlice = sliceLayer[inDims, deconvDims];

		$netP2 = NetReplacePart[$netP2, "Input" -> Prepend[inDims, 1]];
	];

sliceLayer[inDims_, deconvDims_] :=
	Module[
		{
			start, end
		},

		start = (deconvDims[[2]] - inDims[[2]]) / 2;
		end = deconvDims[[2]] - start - 1;

		PartLayer[{
			;; ,
			start ;; end,
			start ;; end
		}]
	]

detectText[img_, dims_, binarize_] :=
	Module[
		{
			inDims = ImageDimensions[img], tmpImg, netRes, netOutput
		},

		tmpImg = ImageResize[img, dims];

		netRes = Internal`CheckImageCache[{"FCNN", Hash[{tmpImg, binarize}]}];
		If[FailureQ[netRes],
			netOutput =
				SafeNetEvaluate[
					netOutput = $netP1[tmpImg];
					netOutput = $netSlice[netOutput];
					netOutput = $netP2[netOutput];

					netOutput = First[netOutput]
					,
					NumericArrayQ
				];

			netOutput = Binarize[Image[netOutput], binarize];
			netOutput = ImageResize[netOutput, inDims];
			netOutput = regionToBB[netOutput];
			netOutput = expandRectangle[netOutput, 5, inDims];

			netRes = netOutput;

			Internal`SetImageCache[{"FCNN", Hash[{tmpImg, binarize}]}, netRes];
		];

		Return[netRes];
	];

(****************************************************************************
    						Helper functions
****************************************************************************)

expandRectangle[recs_List, pad_Integer, dims_List] := expandRectangle[#, pad, dims] & /@ recs
expandRectangle[Rectangle[{xmin_, ymin_}, {xmax_, ymax_}], pad_, {maxW_, maxH_}] :=
	Rectangle @@ Transpose[{Clip[{xmin - pad, xmax + pad}, {0, maxW}], Clip[{ymin - pad, ymax + pad}, {0, maxH}]}]

regionToBB[reg_] := Rectangle @@@ Values[ComponentMeasurements[MorphologicalComponents[reg], "BoundingBox"]]