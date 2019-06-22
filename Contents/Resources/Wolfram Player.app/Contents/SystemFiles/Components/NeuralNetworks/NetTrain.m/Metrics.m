Package["NeuralNetworks`"]


metricSpecT = CustomType[
	"plural[port], plural[string], or plural[function]", 
	If[MatchQ[#, NetPort[__] | _String | Rule[_String, _Integer | _?Positive] | _Function], #, $Failed]&
]

(* TODO:
add GroupBy, e.g. mean activation of hidden units but grouped by mnist digit.
TODO: suppor histogramming over specific range
*)

MetricDefinionT = StructT[{
	"Measurement" -> metricSpecT,
	"Key" -> Defaulting[StringT, Automatic],
	"ShortName" -> Defaulting[StringT, Automatic],
	"ClassAveraging" -> Defaulting[EnumT[{"Micro", "Macro", None}], None],
	"Aggregation" -> Defaulting[Nullable[EnumT[{"L1Norm", "L2Norm", "Total", "Mean", "RootMeanSquare", "StandardDeviation", "Min", "Max"}]], None],
	"Source" -> Defaulting[Nullable[StringT], None],
	"Direction" -> Defaulting[EnumT[{"Increasing", "Decreasing"}], "Decreasing"],
	"PlotType" -> Defaulting[EnumT[{"Log", "Linear", Automatic}], Automatic],
	"PlotRange" -> Defaulting[ListT[2, RealT], {-Infinity, Infinity}],
	"Interval" -> Defaulting[EnumT[{"Round", "Batch", Automatic}], Automatic],
	"$IsAutomatic" -> Defaulting[BooleanT, False]
}];


PackageScope["ParseMeasurementSpecs"]

General::dupmetrickey = "Two measurements use the same key, ``. Please specify unique keys for each measurement using \"Key\" -> key."

Clear[ParseMeasurementSpecs];

ParseMeasurementSpecs[net_NetP, specs_List, hasPrefix_] := Scope[
	
	$net = net;
	$hasPrefix = hasPrefix;

	specs = Map[parseMetric, specs];
	
	$metricDirections ^= Association @ Map[#Key -> #Direction&, specs];
	(* ^ see StoppingCriterion.m for the usage of this *)

	keys = specs[[All, "Key"]];
	If[!DuplicateFreeQ[keys], ThrowFailure["dupmetrickey", QuotedStringForm @ First @ FindDuplicates[keys]]];

    NetResolveMetrics[net, specs]
];

ParseMeasurementSpecs[net_, None, _] := 
	{};

ParseMeasurementSpecs[net_, spec:(_String | _NetPort | _Association | _Rule), hasPrefix_] :=
	ParseMeasurementSpecs[net, {spec}, hasPrefix];

ParseMeasurementSpecs[net_, Automatic, hasPrefix_] := 
	ParseMeasurementSpecs[net, {<|"Measurement" -> "ErrorRate", "$IsAutomatic" -> True|>}, hasPrefix]

General::invmetrspec = "Setting of TrainingProgressMeasurements -> `` should be Automatic, None, or a list of associations, strings, or NetPorts."
ParseMeasurementSpecs[net_, other_, _] :=
	ThrowFailure["invmetrspec", QuotedStringForm @ other];

General::invmetricoutput = "The output `` specified for measurement `` does not exist. Please choose one of ``."

(* this does shallow parsing of the user spec to find the output ports being requested as measurements *)
PackageScope["GetUserMeasurementPorts"]

GetUserMeasurementPorts[outputNames_, spec_] := Block[{$onames = outputNames},
	DeleteDuplicates @ Flatten @ Map[getUserPort, ToList @ spec]
]

getUserPort[port:NetPort[name_String | {name_String}]] := 
	name;

getUserPort[assoc_Association] := 
	getUserPort @ assoc["Measurement"];

getUserPort[_] := Nothing;

parseMetric[spec_] := Scope[
	$ospec = spec;
	spec = If[!AssociationQ[spec], <|"Measurement" -> spec|>, spec];
	assoc = CoerceUserSpec[spec, MetricDefinionT, "the measurement specification"];
	metric = assoc["Measurement"];
	If[MatchQ[metric, _String | Rule[_String, _Integer | _?Positive]], parseStringMetric, parseUserMetric][metric, assoc]
];

parseStringMetric[metric_, assoc_] := Scope[
	
	assoc = assoc; (* :-( *)

	classAvg = assoc["ClassAveraging"];

	metric = Lookup[$builtinMetricSynonyms, metric, metric];

	metricData = Lookup[$builtinMetricData, metric];

	(* if the metric is absent from the list of built-ins that means that either the metric is not valid or it is a TopK metric which wouldn't have a fixed name *)
	If[MissingQ[metricData],
		Switch[metric, 
			Rule["Accuracy" | "ErrorRate", _Integer],
				k = Last @ metric;
				If[k <= 1, failInvalidMetric[]];

				prefix = First @ metric;
				metricData = <|
					"Path" -> $CrossEntropyMetric$["ErrorRate", k],
					"Finalizer" -> If[prefix === "ErrorRate", errorRate, accuracy],
					"PlotRange" -> {0, 1}, "IsPercent" -> True,
					"ShortName" -> "top " <> IntegerString[k] <> If[prefix === "ErrorRate", " error", " acc."],
					"LongName" -> "top " <> IntegerString[k] <> If[prefix === "ErrorRate", " error rate", " accuracy"],
					"Direction" -> If[prefix === "ErrorRate", "Decreasing", "Increasing"],
					"TextFormat" -> errForm,
					"$PerBatch" -> True
				|>,
			Rule["FScore", _?Positive], 
				beta = Last @ metric;
				metricData = <|
					"Path" -> $CrossEntropyMetric$["Counts"],
					"Finalizer" -> macroMicro[classFbetaScore[beta]],
					"PlotRange" -> {0, 1}, "IsPercent" -> True,
					"ShortName" -> "F" <> TextString[beta],
					"LongName" -> "F" <> TextString[beta] <> " score",
					"Direction" -> "Increasing",
					"TextFormat" -> errForm,
					"PreFormat" -> replaceIndeterminate[classFbetaScore[beta][1,1,1,1]]
				|>,
			_, 
				failInvalidMetric[]
		];		
		metricData = Join[$defaultMetricData, metricData];
	];

	(* user can only loosen interval *)	
	If[assoc["Interval"] === "Round" && metricData["$PerBatch"], 
		metricData["$PerBatch"] = False];

	SetAutomatic[classAvg, Lookup[metricData, "ClassAveraging", None]];

	If[classAvg === None && ContainsQ[metricData["Finalizer"], macroMicro], 
		(* reporting should take the mean, if necessary *)
		metricData["PreFormat"] = metricData["PreFormat"] /* maybeMean
		(* ^ this needs to use maybeMean to deal with the Binary classification case *)
		(* ^ also, we can't simply DeleteaCases[Indeterminate] here because that would remove our data points from the plot etc.
		rather we should replace the Indeterminate data points with some value which is what the pre-formatting function replaceIndeterminate does.
		The value to be replaced is the limit of the measure as the variables in the denominator tend to 0.
		*)
	];

	metricData["Finalizer"] = metricData["Finalizer"] /. macroMicro -> 
		Switch[classAvg, None, classThread, "Micro", classMicro, "Macro", classMacro];

	metricData["Aggregation"] = None; (* <-- Doesn't make sense to do this for built-in metrics? Maybe throw error if user tries this? *)
	AssociateTo[assoc, metricData];

	(* If the user specified a path then we can resolve the $CrossEntropyMetric$/$DistanceMetric$ and $CrossEntropyClasses$ now: *)
	output = assoc["Source"];
	If[output =!= None,
		outputLayerPositions = GetOutputLayerPositions[$net];
		
		metricPath = Lookup[
			outputLayerPositions,
			output,
			ThrowFailure["invmetricoutput", output, metric, QuotedStringRow[Append[Keys @ outputLayerPositions, None], " and "]]
		];
		
		layerType = ($net @@ metricPath)["Type"];

		Switch[
			assoc["Path"],
			_$CrossEntropyMetric$,
				If[!MatchQ[layerType, "CrossEntropyLoss"], 
					failMetricLayerType[output, layerType, metric, "CrossEntropyLossLayer"]];

				classes = getCEClasses[metricPath, $net];

				assoc = assoc /. {
					$CrossEntropyMetric$[tag_] :> RuleCondition[Append[metricPath, tag]],
					$CrossEntropyMetric$[tag_, k_] :> RuleCondition[Join[metricPath, NetPath[tag, k]]], (* For the TopK metrics*)
					$CrossEntropyClasses$ -> classes
				};
				assoc = checkKLessThanNumClasses[assoc, classes]
			,
			_$DistanceMetric$,
				If[!MatchQ[layerType, "MeanSquaredLoss"|"MeanAbsoluteLoss"], 
					failMetricLayerType[output, layerType, metric, "MeanSquaredLossLayer or MeanAbsoluteLossLayer"]];

				If[assoc["Measurement"] == "IntersectionOverUnion", checkLastDimIsFour[metricPath, $net]];
				
				assoc = assoc /. {$DistanceMetric$[tag_] :> RuleCondition[Append[metricPath, tag]]}
			,
			_$AttentionMetric$,
				If[!MatchQ[layerType, "Attention"], 
					failMetricLayerType[output, layerType, metric, "AttentionLayer"]];

				If[assoc["Measurement"] == "AttentionWeights", checkFixedLength[metricPath, $net]]; 
				
				assoc = assoc /. {$AttentionMetric$[tag_] :> RuleCondition[Append[metricPath, tag]]}
			(* ^ TODO: fix this case - it doesn't make sense to find an output port for an AttentionLayer! Need some other way of specifying this! *)
		]
	];

	SetAutomatic[assoc["Key"], Replace[metric, (a_ -> b_) -> a]];
	
	assoc
]

General::varlenmetricarr = "Cannot measure `` as it is a variable-length array."

General::lfmesambig = "Cannot measure output port `1` when specifying LossFunction -> Automatic, as it is ambigious whether `1` should be treated as a loss in this case. \
Please provide an explicit set of output ports for LossFunction to remove the ambiguity."

PackageScope["$aggFunctions"]

$aggFunctions = <|
	None -> Identity,
	"L1Norm" -> Function[Norm[toFlat @ #, 1]], 
	"L2Norm" -> Function[Norm[toFlat @ #, 2]], 
	"Total" -> Function[Total[#, {1, Infinity}]],
	"Mean" -> toFlat /* Mean,
	"RootMeanSquare" -> toFlat /* RootMeanSquare,
	"StandardDeviation" -> toFlat /* StandardDeviation,
	"Min" -> arrayMin, "Max" -> arrayMax
|>

toFlat[na_NumericArray] := Flatten @ Normal @ na;
toFlat[a_List] := Flatten @ a;
toFlat[a_] := N @ List @ a;

$defaultUserMetricData = Sequence[
	"IsPercent" -> False,
	"PlotFormat" -> None,
	"PreFormat" -> arrayMean,
	"TextFormat" -> Identity
]

General::misscustmeas = "The requested measurement NetPort[\"``\"] was not an input or output of the net, and was not found in the training data. The available keys in the training data are: ``."
lookupMeasure[name_][] := 
	If[!AssociationQ[$LastGeneratorData], 0., (* <- should never happen *)
		Lookup[$LastGeneratorData, name, 
			ThrowFailure["misscustmeas", name, QuotedStringRow[Keys @ $LastGeneratorData, " and "]]]]

General::custmeaskey = "The measurement employing the custom function `` must specify a value for \"Key\"."
parseUserMetric[f_Function, assoc_] := 
	If[assoc["Key"] === Automatic,
		ThrowFailure["custmeaskey", Shallow[f]],
		callbackMetric[f[$livePropertyData]&, Automatic, assoc]
	]

callbackMetric[callback_, name_, assoc_] := Scope[
	aggFunc = $aggFunctions @ assoc["Aggregation"];
	name = Replace[assoc["Key"], Automatic -> name];
	Append[assoc, {
		"Path" -> CallbackMetric[callback /* aggFunc],
		"Finalizer" -> Identity,
		"Key" -> name, "LongName" -> name, "ShortName" -> name,
		"$PerBatch" -> (assoc["Interval"] =!= "Round"),
		$defaultUserMetricData
	}]
]

PackageScope["CallbackMetric"]
ResolveCallbackMetric[CallbackMetric[func_]] := 
	checkReal @ func[];

(* this could be generalized between numeric values, but then it gets harder to decide how to plot
etc the measurement in advance *)
General::badcustmeas = "Custom measurement gave ``, which is not a real number.";
checkReal[e_] := Replace[N @ e, Except[_Real] :> ThrowFailure["badcustmeas", Shallow[e]]];

parseUserMetric[NetPort[a_, b_], assoc_] := 
	parseUserMetric[NetPort @ ToList[a, b], assoc]

General::invnetportmetric = "`` does not correspond to an input, output, or internal result of the net, and so cannot be used as a measurement.";
parseUserMetric[metric_, assoc_] := Scope[
	assoc = assoc; (* so its gets Blocked *)
	path = ToNetPath[If[$hasPrefix, $net["Nodes", "Net"], $net], First @ metric];
	If[FailureQ[path] && MatchQ[metric, NetPort[_String]],
		(* assume this is an input that will be provided by a user data generator *)
		name = First @ metric;
		Return @ callbackMetric[lookupMeasure @ name, name, assoc]];
	type = Apply[$net, path];
	If[DynamicDimsQ[type], ThrowFailure["varlenmetricarr", metric]];
	agg = assoc["Aggregation"];
	shortName = StringRiffle[FromNetPath @ path, "/"];
	SetAutomatic[assoc["Key"], shortName];
	If[FailureQ[path], ThrowFailure["invnetportmetric", QuotedStringForm @ metric]];
	If[StringQ[agg], AppendTo[path, "$" <> agg]];
	truePath = If[$hasPrefix, addPrefix[path], path];
	If[$isInNetTrain && MatchQ[truePath, NetPath["Outputs", _]] && lossFunction === Automatic, 
		ThrowFailure["lfmesambig", QuotedStringForm @ metric]];
	AssociateTo[assoc, {
		"Path" -> truePath,
		"Finalizer" -> If[!ArrayPathQ[path], normalizeLength, normalizeBatch],
		"LongName" -> shortName, "ShortName" -> shortName,
		"$PerBatch" -> (assoc["Interval"] =!= "Round"),
		$defaultUserMetricData
	}];
	assoc
];

General::invoutputlayer = "The specified output `` was generated by ``. Metric `` requires a ``."
failMetricLayerType[output_, type_, metric_, alts_] := ThrowFailure["invoutputlayer", output, type, metric, alts];

General::invbuiltinmetric = "`` is not a supported measurement. Valid measurements can be NetPort expressions, Function[...] expressions, or built-in measurements including ``.";
failInvalidMetric[] := ThrowFailure["invbuiltinmetric", QuotedStringForm @ $ospec, QuotedStringRow[$topBuiltinMetrics, " and "]];


PackageScope["$UnresolvedPath$"]

PackageScope["$CrossEntropyMetric$"]
PackageScope["$CrossEntropyClasses$"]

$ceCounts = $CrossEntropyMetric$["Counts"];
$ceROCCounts = $CrossEntropyMetric$["ROCCounts"];
$ceError = $CrossEntropyMetric$["ErrorRate"];
$cePairs = $CrossEntropyMetric$["Pairs"];
(*  ^ when ParseMeasurementSpecs is called, we may not yet have the classes, or know the NetPath of the 
final CE layer,  because that might only be added later by NetAttachLoss. so we use a placeholder. 
it will be resolved in NetResolveMetrics later (which also ensures it is unambigious etc) *)

PackageScope["$DistanceMetric$"]
$distanceVarRatio = $DistanceMetric$["GoodnessOfFit"];
$distanceIOU = $DistanceMetric$["IOU"];

PackageScope["$AttentionMetric$"]

Clear[classMicro, classMacro, classThread, maybeMean]

classMicro[func_] := Function[Quiet[func @@ Total[#]]];
classMacro[func_] := classThread[func] /* DeleteCases[Indeterminate] /* Mean;

classThread[func_] := classThread[$CrossEntropyClasses$, func];
classThread[classes_List, func_] := Function[AssociationThread[classes, Quiet[func @@@ #]]];
classThread[None, func_] := Function[Quiet[func @@ #1]];
(* ^ TODO: for the binary case we may want to even skip emitting the 1 dimension in the first place, so we don't need to flatten *)

maybeMean[x_List] := Mean @ x;
maybeMean[x_Association] := Mean @ x;
maybeMean[x_] := x;

$topBuiltinMetrics = {"ErrorRate", "F1Score", "Precision", "Recall", "ConfusionMatrix"};

$defaultMetricData = <|
	"$PerBatch" -> False,
	"TextFormat" -> Identity,
	"PlotRange" -> {-Infinity, Infinity},
	"PlotFormat" -> None,
	"IsPercent" -> False,
	"Direction" -> "Decreasing",
	"PreFormat" -> Identity
|>;

PackageScope["makeLossMetricData"]

(* this is used for the measurement assocs that correspond to losses rather than metrics *)
makeLossMetricData[key_, path_] := Scope[
	hasLoss = StringContainsQ[ToLowerCase @ key, "loss"];
	longName = If[hasLoss, If[key == "Loss", "loss", key], "\"" <> key <> "\" loss"];
	shortName = If[hasLoss, ToLowerCase @ Decamel @ key, key <> " loss"];
	If[!hasLoss, key = key <> "Loss"];
	Join[$defaultMetricData,
	<|"Key" -> key, "Path" -> path, "$PerBatch" -> True, "Finalizer" -> Identity, 
	"PlotType" -> "Log", "LongName" -> longName, "ShortName" -> shortName, "Measurement" -> longName|>
	]
];

procMDentry[metric_, entry_] := Scope[
	decamelName := decamelName = ToLowerCase @ Decamel @ metric;
	entry = entry;
	SetMissing[entry["LongName"], decamelName];
	SetMissing[entry["ShortName"], decamelName];
	If[entry["IsPercent"], 
		entry["TextFormat"] = errForm;
		entry["PlotType"] = "Linear";
	];
	Join[$defaultMetricData, entry]
];

replaceIndeterminate[v_] := ReplaceAll[Indeterminate -> v]; 

classf1score[tp_, fp_, fn_, tn_] := (2 * tp) / (2 * tp + fp + fn);
classFbetaScore[beta_][tp_, fp_, fn_, tn_] := ((1 + beta^2) * tp) / ((1 + beta^2) * tp + fp + beta^2 *fn);
classPrecision[tp_, fp_, fn_, tn_] := tp / (tp + fp);
classRecall[tp_, fp_, fn_, tn_] := tp / (tp + fn);
classSpecificity[tp_, fp_, fn_, tn_] := tn / (tn + fp);
classFPR[tp_, fp_, fn_, tn_] := fp / (fp + tn);
classFNR[tp_, fp_, fn_, tn_] := fn / (fn + tp);
classFDR[tp_, fp_, fn_, tn_] := fp / (fp + tp);
classNPV[tp_, fp_, fn_, tn_] := tn / (tn + fn);
classFOR[tp_, fp_, fn_, tn_] := fn / (fn + tn);
classMCC[tp_, fp_, fn_, tn_] := ((tp * tn) - (fp * fn)) / Sqrt[(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)];
classInformedness[tp_, fp_, fn_, tn_] := (tp*tn - fn*fp)/(tp*tn + fn*tn + fp*tp + fn*fp);
classMarkedness[tp_, fp_, fn_, tn_] := (tp*tn - fn*fp)/(tp*tn + fp*tn + tp*fn + fn*fp);
classBalancedAccuracy[tp_, fp_, fn_, tn_] := (tp*tn + tp*fp + tn*tp + tn*fn)/(2*(tp*tn + fn*tn + fp*tp + fn*fp));
classPPCR[tp_, fp_, fn_, tn_] := (tp + fp)/(tp + fp + fn + tn);
classTruePositive[tp_, fp_, fn_, tn_] := tp;
classFalsePositive[tp_, fp_, fn_, tn_] := fp;
classFalseNegative[tp_, fp_, fn_, tn_] := fn;
classTrueNegative[tp_, fp_, fn_, tn_] := tn;

$builtinMetricData = procMDentry ~IMap~ Association[
	"RawCounts" -> <|
		"Path" -> $ceCounts,
		"Finalizer" -> Identity,
		"ShortName" -> "counts", 
		"$PerBatch" -> True
	|>,
	"F1Score" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classf1score],
		"ShortName" -> "F1", "LongName" -> "F1 score",
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classf1score[1,1,1,1]]
	|>,
	"Precision" -> <|
		"Path" -> $ceCounts, 
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classPrecision],
		"ShortName" -> "prec.", 
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classPrecision[1,1,1,1]]
	|>,
	"Recall" -> <|
		"Path" -> $ceCounts, 
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classRecall],
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classRecall[1,1,1,1]]
	|>,
	"Specificity" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classSpecificity],
		"ShortName" -> "spec.", 
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classSpecificity[1,1,1,1]]
	|>,
	"FalsePositiveRate" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classFPR],
		"ShortName" -> "F.P. rate",
		"PreFormat" -> replaceIndeterminate[classFPR[1,1,1,1]]
	|>,
	"FalseNegativeRate" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classFNR],
		"ShortName" -> "F.N. rate",
		"PreFormat" -> replaceIndeterminate[classFNR[1,1,1,1]]
	|>,
	"FalseDiscoveryRate" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classFDR],
		"ShortName" -> "F.D. rate",
		"PreFormat" -> replaceIndeterminate[classFDR[1,1,1,1]]
	|>,
	"NegativePredictiveValue" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classNPV],
		"ShortName" -> "N.P.V.",
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classNPV[1,1,1,1]]
	|>,
	"FalseOmissionRate" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> macroMicro[classFOR],
		"ShortName" -> "F.O. rate",
		"PreFormat" -> replaceIndeterminate[classFOR[1,1,1,1]]
	|>,
	"MatthewsCorrelationCoefficient" -> <|
		"Path" -> $ceCounts, 
		"PlotRange" -> {-1, 1},
		"Finalizer" -> macroMicro[classMCC],
		"ShortName" -> "M.C.C.", "LongName" -> "Matthews correlation coefficient",
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classMCC[1,1,1,1]]
	|>,
	"Informedness" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {-1, 1}, 
		"Finalizer" -> macroMicro[classInformedness],
		"ShortName" -> "inf.",
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classInformedness[1,1,1,1]]
	|>,
	"Markedness" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {-1, 1},
		"Finalizer" -> macroMicro[classMarkedness],
		"ShortName" -> "mark.",
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classMarkedness[1,1,1,1]]
	|>,
	"BalancedAccuracy" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1},
		"Finalizer" -> macroMicro[classBalancedAccuracy],
		"ShortName" -> "bal. acc.",
		"Direction" -> "Increasing",
		"PreFormat" -> replaceIndeterminate[classBalancedAccuracy[1,1,1,1]]
	|>,
	"PredictedPositiveConditionRate" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, 1},
		"Finalizer" -> macroMicro[classPPCR],
		"ShortName" -> "P.P.C.R"
	|>,
	"TruePositiveNumber" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> macroMicro[classTruePositive],
		"ShortName" -> "T.P."
	|>,
	"FalsePositiveNumber" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> macroMicro[classFalsePositive],
		"ShortName" -> "F.P.",
		"Direction" -> "Decreasing"
	|>,
	"TrueNegativeNumber" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> macroMicro[classTrueNegative],
		"ShortName" -> "T.N."
	|>,
	"FalseNegativeNumber" -> <|
		"Path" -> $ceCounts,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> macroMicro[classFalseNegative],
		"ShortName" -> "F.N.",
		"Direction" -> "Decreasing"
	|>,
	"ErrorRate" -> <|
		"Path" -> $ceError, 
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> errorRate,
		"ShortName" -> "error",
		"$PerBatch" -> True
	|>,
	"Accuracy" -> <|
		"Path" -> $ceError, 
		"PlotRange" -> {0, 1}, "IsPercent" -> True,
		"Finalizer" -> accuracy,
		"ShortName" -> "acc.",
		"$PerBatch" -> True,
		"Direction" -> "Increasing"
	|>,
	"ConfusionMatrix" -> <|
		"Path" -> $cePairs,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> Identity,
		"ShortName" -> "conf. mat.",
		"PlotFormat" -> Function[combinedConfusionMatrixPlot[#1, #2, #3, #4, $CrossEntropyClasses$]],
		"TextFormat" -> None
	|>,
	"ConfusionMatrixPlot" -> <|
		"Path" -> $cePairs,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> Function[confusionMatrixPlot[#1, 300, True, $validationFinalize, $CrossEntropyClasses$]],
		"ShortName" -> "conf. mat. plot",
		"PlotFormat" -> Function[stackOrFlipPlots[#1, #2, #4]],
		"TextFormat" -> None
	|>,
	"CohenKappa" -> <|
		"Path" -> $cePairs,
		"PlotRange" -> {-Infinity, 1},
		"Finalizer" -> cohenKappa,
		"ShortName" -> "\[Kappa]", "LongName" -> "Cohen Kappa",
		"Direction" -> "Increasing"
	|>,
	"ScottPi" -> <|
		"Path" -> $cePairs,
		"PlotRange" -> {-Infinity, 1},
		"Finalizer" -> scottPi,
		"ShortName" -> "\[Pi]", "LongName" -> "Scott Pi",
		"Direction" -> "Increasing"
	|>,
	"ROCCurve" -> <|
		"Path" -> $ceROCCounts,
		"PlotRange" -> {0, 1},
		"Finalizer" -> macroMicro[classROCCurve],
		"PlotFormat" -> combinedRocCurvePlot, 
		"TextFormat" -> None,
		"ShortName" -> "R.O.C.", "LongName" -> "R.O.C. curve"
	|>,
	"ROCCurvePlot" -> <|
		"Path" -> $ceROCCounts,
		"PlotRange" -> {0, 1},
		"Finalizer" -> macroMicro[classROCCurve] /* Function[rocCurvePlot[#, 300, True, $validationFinalize]],
		"PlotFormat" -> Function[overlayPlots[#1, #2]], 
		"TextFormat" -> None,
		"ShortName" -> "R.O.C. plot", "LongName" -> "R.O.C. curve plot"
	|>,
	"AreaUnderROCCurve" -> <|
		"Path" -> $ceROCCounts,
		"PlotRange" -> {0, 1},
		"Finalizer" -> macroMicro[classROCCurve] /* areaUnderCurve,
		"ShortName" -> "A.U.C.", "LongName" -> "Area under R.O.C. curve",
		"Direction" -> "Increasing"
	|>,
	"Entropy" -> <|
		"Path" -> $CrossEntropyMetric$["Entropy"],
		"PlotRange" -> {0, 1}, 
		"Finalizer" -> normalizeLength,
		"$PerBatch" -> True
	|>,
	"Perplexity" -> <|
		"Path" -> $CrossEntropyMetric$["Entropy"],
		"PlotRange" -> {1, E}, 
		"Finalizer" -> Function[Exp[normalizeLength[#]]],
		"$PerBatch" -> True
	|>,
	"FractionVarianceUnexplained" -> <|
		"Path" -> $distanceVarRatio,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> Apply[fracVarUnexplained],
		"ShortName" -> "var. unex.", "LongName" -> "fraction of variance unexplained"
	|>,
	"RSquared" -> <|
		"Path" -> $distanceVarRatio,
		"PlotRange" -> {-Infinity, 1},
		"Finalizer" -> Apply[rSquared],
		"ShortName" -> "res. r\.b2", "LongName" -> "R-squared",
		"Direction" -> "Increasing"
	|>,
	"StandardDeviation" -> <|
		"Path" -> $distanceVarRatio,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> Apply[residualStandardDeviation],
		"ShortName" -> "res. s.d."
	|>,
	"MeanSquare" -> <|
		"Path" -> $distanceVarRatio,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> Apply[residualMeanSquare],
		"ShortName" -> "res. mean sqr.", "LongName" -> "residual mean square"
	|>,
	"MeanDeviation" -> <|
		"Path" -> $distanceVarRatio,
		"PlotRange" -> {0, Infinity},
		"Finalizer" -> Apply[residualMeanAbsolute],
		"ShortName" -> "res. mean dev.", "LongName" -> "residual mean absolute"
	|>,
	"IntersectionOverUnion" -> <|
		"Path" -> $distanceIOU,
		"PlotRange" -> {0, 1},
		"Finalizer" -> normalizeLength,
		"ShortName" -> "I.O.U.",
		"Direction" -> "Increasing",
		"$PerBatch" -> True
	|>,
	(* TODO: should we make entropy and perplexity use bits rather than nats? Simple as divinding by log_2(E) *)
	"AttentionWeights" -> <|
		"Path" -> $AttentionMetric$["AttentionWeights"],
		"PlotRange" -> {0, 1},
		"Finalizer" -> normalizeLength
	|>
	
];

$builtinMetricSynonyms = <|
	"Recall" -> {"Sensitivity", "HitRate", "TruePositiveRate", "TPR", "ClassAccuracy"},
	"Precision" -> {"PositivePredictiveValue", "TruePositiveRate", "TPR", "PPV", "Confidence", "TruePositiveAccuracy"},
	"Specificity" -> {"Selectivity", "TrueNegativeRate", "TNR", "InverseRecall"},
	"NegativePredictiveValue" -> {"InversePrecision", "NPV", "TrueNegativeAccuracy"},
	"FalsePositiveRate" -> {"FallOut", "FPR"},
	"FalseDiscoveryRate" -> {"FDR"},
	"FalseOmissionRate" -> {"FOR"},
	"FalseNegativeRate" -> {"MissRate", "FNR"},
	"MatthewsCorrelationCoefficient" -> {"MCC"},
	"FScore" -> {"FMeasure"},
	"RSquared" -> {"R2", "CoefficientOfDetermination"},
	"FractionVarianceUnexplained" -> {"FVU"},
	"AreaUnderROCCurve" -> {"AUC"},
	"ROCCurve" -> {"ROC"},
	"IntersectionOverUnion" -> {"JaccardIndex", "JaccardSimilarityCoefficient", "IOU"},
	"Informedness" -> {"YoudenIndex", "YoudenJStatistic", "BookmakerInformedness", "BM"},
	"Markedness" -> {"MK"},
	"CohenKappa" -> {"Kappa"},
	"ScottPi" -> {"Pi"},
	"PredictedPositiveConditionRate" -> {"PPCR"}
|> // KeyValueMap[Function[key = #1; value = #2; Map[Function[#1 -> key], value]]] // Flatten;
(* ^ Convert this association into a more usable form: synonym -> measurement *)

normalizeLength[e_] := e / $metricInputLen;
normalizeBatch[e_] := e / $metricNumBatches;

classROCCurve[tp_, fp_, fn_, tn_] := {
	Quiet[classFPR[tp, fp, fn, tn]],
	Quiet[classRecall[tp, fp, fn, tn]]
} /. Indeterminate -> 0.5;

areaUnderCurve[{xs_List, ys_List}] := Module[{points},
	points = Prepend[Reverse @ Prepend[Transpose[{xs, ys}], {1., 1.}], {0., 0.}];
   	Total[iarea @@@ Partition[points, 2, 1]]
   ];

areaUnderCurve[data_] := Map[areaUnderCurve, data];
iarea[{x1_, y1_}, {x2_, y2_}] := 0.5*(y1 + y2)*(x2 - x1);

errorRate = normalizeLength;
accuracy = Function[1 - errorRate[#1]];

cohenKappa[pairs_] := Scope[
	tot = Total[pairs, Infinity];
	po = Total[Tr@pairs]/tot;
	pe = Total[pairs, {2}].Total[pairs, {1}] / (tot^2);
	If[pe == 1 && po == 1, pe = 0]; (* deal with an edge case that would cause Kappa to be Indeterminate *)
	(po - pe)/(1 - pe) // N
];

scottPi[pairs_] := Scope[
   tot = Total[pairs, Infinity];
   pra = Total[Tr@pairs]/tot;
   pre = (Total[pairs, {2}] + Total[pairs, {1}] )/ (tot*2);
   pre = Total[pre^2];
   If[pra == 1 && pre == 1, pre = 0]; (* deal with an edge case that would cause Pi to be Indeterminate *)
   (pra - pre)/(1 - pre) // N
];

fracVarUnexplained[absoluteResidual_, squaredResidual_, squaredTarget_, target_] := 
	squaredResidual/(squaredTarget - target^2 / $metricInputLen + $MachineEpsilon);

rSquared[absoluteResidual_, squaredResidual_, squaredTarget_, target_] := 
	1 - fracVarUnexplained[absoluteResidual, squaredResidual, squaredTarget, target];

residualStandardDeviation[absoluteResidual_, squaredResidual_, squaredTarget_, target_] := 
	Sqrt @ residualMeanSquare[absoluteResidual, squaredResidual, squaredTarget, target];

residualMeanSquare[absoluteResidual_, squaredResidual_, squaredTarget_, target_] := 
	squaredResidual / $metricInputLen;

residualMeanAbsolute[absoluteResidual_, squaredResidual_, squaredTarget_, target_] := 
	absoluteResidual / $metricInputLen;

(* a very similar pattern repeats her with 
combinedRocCurvePlot, rocCurvePlot, and overlayPlots
vs 
combinedConfusionMatrixPlot, confusionMatrixPlot, and stackOrFlipPlots.
*)

combinedRocCurvePlot[trainPoints_, None, width_, final_] :=
	rocCurvePlot[trainPoints, width, final, False]

combinedRocCurvePlot[trainPoints_, validPoints_, width_, final_] := 
	overlayPlots[
		rocCurvePlot[trainPoints, width, final, False],
		rocCurvePlot[validPoints, width, final, True]
	]

overlayPlots[TightLabeled[a_, l_], TightLabeled[b_, _]] := TightLabeled[Show[a, b], l];
overlayPlots[a_, b_] := Show[a, b];
overlayPlots[a_, None] := a;

rocCurvePlot[points_Association, width_, final_, valid_] := 
	rocCurvePlot[Mean @ points, width, final, valid]

rocCurvePlot[points_List, width_, final_, valid_] := Scope[
	plot = ListPlot[
		{Reverse @ Join[{{1,1}}, Transpose @ points, {{0,0}}], {{0, 0}, {1, 1}}}, 
		(* ^ this reverse is required or else the plot isn't filled in as desired *)
		PlotRange -> {{0., 1.}, {0., 1.}}, PlotRangePadding -> 0.003,
		Frame -> True, Joined -> True, 
		ImageSize -> {width, width},
		Background -> White,
		AspectRatio -> 1,
		PlotStyle -> {If[valid === False, $LossLineStyle, $ValidationLossLineStyle], GrayLevel[0.7]}, 
		GridLines -> Automatic,	GridLinesStyle -> Directive[GrayLevel[0.5, 0.5], AbsoluteThickness[1], AbsoluteDashing[{1, 2}]],
		PlotRangeClipping -> True
	];
	TightLabeled[plot, {If[final, "false positive rate", None], "recall", If[!final, "false positive rate", None]}]
]

stackOrFlipPlots[a_, b_, False] := clickFlip[a, b];
stackOrFlipPlots[a_, b_, True] := Column[{a, b}];
stackOrFlipPlots[a_, None, False|True] := a;

$flipState = False;
clickFlip[a_, None] := a;
clickFlip[a_, b_] := MouseAppearance[
	EventHandler[
		PaneSelector[{False -> a, True -> b}, Dynamic[$flipState]],
		"MouseClicked" :> ($flipState = !$flipState)
	],
	"LinkHand"
];

combinedConfusionMatrixPlot[trainConfMat_, None, width_, final_, classes_] :=
	confusionMatrixPlot[trainConfMat, width, final, False, classes]

combinedConfusionMatrixPlot[trainConfMat_, validConfMat_, width_, final_, classes_] :=
	stackOrFlipPlots[
		confusionMatrixPlot[trainConfMat, width, final, False, classes],
		confusionMatrixPlot[validConfMat, width, final, True, classes],
		final
	]

numDigits[n_] := Ceiling[$MachineEpsilon + Log10[n]];
labelReqPadding[n_] := 6. + 6. * numDigits[n];

confusionMatrixPlot[confMat_, width_, final_, valid_, classes_:None] := Scope[
	confMat = Floor @ confMat;
	If[classes === None, classes = Range@Length[confMat]];
	nclass = Length[confMat];
	bticks = rticks = None;
	rpadding = bpadding = All;
	If[nclass > 30, 
		lticks = tticks = All;
		doLabels = False;
	,
		doLabels = numDigits[Max[confMat]] <= Floor[20/nclass]*2;
		(* ^ don't plot labels if annotations wont fit in the sequres of the MatrixPlot *)
		lticks = Transpose[{Range[nclass], Rotate[#, 0.] & /@ classes}];
		tticks = Transpose[{Range[nclass], Rotate[#, Pi/2] & /@ classes}];
		If[final,
			ctotals = Total[confMat]; rtotals = Total /@ confMat;
			bticks = Transpose[{Range[nclass], Rotate[#, Pi/2] & /@ Total[confMat]}];
			rticks = Transpose[{Range[nclass], Total /@ confMat}];
			rpadding = labelReqPadding @ Total @ rtotals;
			bpadding = labelReqPadding @ Total @ ctotals;
		];
	];
	plot = MatrixPlot[
		confMat,
		ImageSize -> width,	
		Background -> White,
		ColorFunction -> Function[Blend[{White, If[valid === False, $LossLineStyle, $ValidationLossLineStyle]}, 2*(#-0.5)]],
		FrameTicks -> {{lticks, rticks}, {bticks, tticks}}, 
		Epilog -> If[final && doLabels,
			MapIndexed[
				If[#1 === 0, Nothing,
					Text[#1, {#2[[2]] - .5, nclass - #2[[1]] + .5}, {0, 0}]
				]&,
				confMat,
				{2}
			], {}
		],
		ImagePadding -> {{All, rpadding}, {bpadding, All}},
		BaseStyle -> Directive[FontSize -> 7, FontFamily -> "Verdana", FontWeight -> "Thin", FontTracking -> "Condensed"]
	];
	TightLabeled[plot, {If[final, "predicted class", None], "actual class", If[!final, "predicted class", None]}]
]

$builtinMetrics = Keys[$builtinMetricData];


PackageScope["NetResolveMetrics"]

General::nometriclayer = "One or more measurements require a `` to be present in the training net, but there are none."
General::multimetriclayers = "There should be exactly one `` present in the training net, but there is more than one."

Clear[NetResolveMetrics];

SetUsage @ "
NetResolveMetrics[net$, metrics$] attempts to resolve various details in the list of metric associations metrics$.
* $CrossEntropyMetric:0036, which represents hidden metric outputs generated by a (unique) CrossEntropy layer in net$.
* $CrossEntropyClasses:0036, which represents a list of classes obtained by following the CE target port back to a 'Class' NetEncoder."

NetResolveMetrics[net_NetP, metrics_] := NetResolveMetrics[net, metrics, True]

NetResolveMetrics[net_NetP, metrics_] /; ContainsQ[metrics, $CrossEntropyMetric$] := Scope[
	cePaths = FindSubNets[net, "CrossEntropyLoss"];
	ceCount = Length[cePaths];

	If[TrueQ[metrics[[1, "$IsAutomatic"]]] && ceCount =!= 1, Return[{}]];
	(* ^ if we are in the Automatic case, and there isn't exactly one CE layer,
	we just give up without issuing a message *)

	If[ceCount == 0, ThrowFailure["nometriclayer", "CrossEntropyLossLayer"]];
	If[ceCount > 1, ThrowFailure["multimetriclayers", "CrossEntropyLossLayer"]];

	cePath = First[cePaths];
	
	classes = getCEClasses[cePath, net];

	(* TODO: we don't need to defer the lookup of these things anymore *)
	metrics = metrics /. {
		$CrossEntropyMetric$[tag_] :> RuleCondition[Append[cePath, tag]],
		$CrossEntropyMetric$[tag_, k_] :> RuleCondition[Join[cePath, NetPath[tag, k]]], (* For the TopK metrics*)
		$CrossEntropyClasses$ -> classes
	};

	metrics = Map[checkKLessThanNumClasses[#, classes]&, metrics];

	NetResolveMetrics[net, metrics]
]

NetResolveMetrics[net_NetP, metrics_] /; ContainsQ[metrics, $DistanceMetric$] := Scope[
	distPaths = FindSubNets[net, "MeanSquaredLoss" | "MeanAbsoluteLoss"];
	distCount = Length[distPaths];

	If[distCount == 0, ThrowFailure["nometriclayer", "MeanSquaredLossLayer or MeanAbsoluteLossLayer"]];
	If[distCount > 1, ThrowFailure["multimetriclayers", "MeanSquaredLossLayer or MeanAbsoluteLossLayer"]]; 

	distPath = First[distPaths];
	graph = PortConnectivityGraph[net];

	metrics = metrics /. $DistanceMetric$[tag_] :> RuleCondition[Append[distPath, tag]];

	Map[If[#Measurement === "IntersectionOverUnion", checkLastDimIsFour[#Path // Most, net]]&, metrics];

	NetResolveMetrics[net, metrics]
]

NetResolveMetrics[net_NetP, metrics_] /; ContainsQ[metrics, $AttentionMetric$] := Scope[
	attPaths = FindSubNets[net, "Attention"];
	attCount = Length[attPaths];

	If[attCount == 0, ThrowFailure["nometriclayer", "Attention"]];
	If[attCount > 1, ThrowFailure["multimetriclayers", "Attention"]]; 

	attPath = First[attPaths];
	graph = PortConnectivityGraph[net];

	metrics = metrics /. $AttentionMetric$[tag_] :> RuleCondition[Append[attPath, tag]];

	Map[If[#Measurement === "AttentionWeights", checkFixedLength[#Path // Most, net]]&, metrics];

	NetResolveMetrics[net, metrics]
]

NetResolveMetrics[_, metrics_, _] := metrics

pushAssocToBags[<||>, _] := Null;
pushAssocToBags[assoc_, bags_] := MapThread[BagPush, {bags, Values @ assoc}];

unpackBagsToAssoc[keys_, bags_] := AssociationThread[keys, Internal`BagPart[#, All]& /@ bags];

$metricStateSetup := Quoted[

	$batchMeasurements = $roundMeasurements = $validationMeasurements = <||>;
	$batchMeasurementsLists = $roundMeasurementsLists = $validationMeasurementsLists = None;

	$metricInputLen = batchSize;
	$metricNumBatches = $batchesPerRound;

	metricObjects = $measurementsInfo;

	$batchMeasurementsKeys = $batchMeasurementsValues = $batchMeasurementsBags = {};
	Scan[setupBatchMetricState, Select[metricObjects, Key["$PerBatch"]]];
	$batchMeasurements := AssociationThread[$batchMeasurementsKeys, $batchMeasurementsValues];
	$batchMeasurementsLists := unpackBagsToAssoc[$batchMeasurementsKeys, $batchMeasurementsBags];

	$roundMeasurementsKeys = metricObjects[[All, "Key"]];
	$roundMeasurementsBags = Table[Bag[], Length[metricObjects]];
	$validationMeasurementsBags = Table[Bag[], Length[metricObjects]];
	
	setter = With[
		observer = CreateMetricObserver[metricObjects, batchSize], 
		makeCollector[1, $roundMeasurements = observer[]; pushAssocToBags[$roundMeasurements, $roundMeasurementsBags]]
	];
	$roundMeasurementsLists := unpackBagsToAssoc[$roundMeasurementsKeys, $roundMeasurementsBags];
	$validationMeasurementsLists := unpackBagsToAssoc[$roundMeasurementsKeys, $validationMeasurementsBags];

	(* we have a single observer that will pull all metrics *)
	If[!MatchQ[trainingProgressReporting, "ProgessIndicator" | None],
		If[!MatchQ[trainingProgressReporting, "Panel" | "Window"],
			metricObjects = Discard[metricObjects, #PlotFormat =!= None&]];
		(* ^ if the mode is textual, discard metrics that do require a custom plotter *)
		$reportingObserver = CreateMetricObserver[metricObjects, batchSize];
		$lastReportingObservation = <||>;
	];

	$roundCollectors ^= Join[$roundCollectors, setter];
];

(* ^ we could use an observer function to create the batch measurements value,
but because it runs every batch, we want it to be as fast as possible, so we
have custom code here *)

(*
back of the envelope calculation: 15k batches, 1 hour / round, 30 days of 
training will yield 300 megabytes of batch metrics. acceptable *)

setupBatchMetricState[assoc_] := ModuleScope[

	UnpackAssociation[assoc, path, finalizer, key];

	AppendTo[$batchMeasurementsKeys, key];
	AppendTo[$batchMeasurementsValues, $value]; (* $value symbol has no value initially *)
		
	$bag = Bag[];

	setter = With[
		{finalizer = finalizer, path = path},
		Which[
			path === "TotalLoss",
				makeCollector[1, $value = Total @ $rawBatchLosses; BagPush[$bag, $value]],
			Head[path] === CallbackMetric,
				makeCollector[1, $value = finalizer @ ResolveCallbackMetric @ path; BagPush[$bag, $value]],
			StringQ[path],
				makeCollector[1, $value = finalizer @ $rawBatchLosses @ path; BagPush[$bag, $value]],
			True,
				makeCollector[1, $value = finalizer @ $rawBatchMetrics @ path; BagPush[$bag, $value]]
		]
	];

	AppendTo[$batchMeasurementsBags, $bag];
	$batchCollectors ^= Join[$batchCollectors, setter];
];




PackageScope["RoundMetricFinalize"]

RoundMetricFinalize[metricObjects_, pathTotals_, lossTotals_, length_, numBatches_, valid_] := Scope[
	$metricInputLen = length;
	$metricNumBatches = numBatches;
	$validationFinalize = valid;
	(* ^ to tell ROCCurvePLot and ConfusionMatrixPlot which color scheme to use *)
	Association @ Map[
		#Key -> #Finalizer[resolvePath[#Path, pathTotals, lossTotals]]&, 
		metricObjects
	]
]

resolvePath[metric_CallbackMetric, _, _] := ResolveCallbackMetric[metric];
resolvePath[path_, pathTotals_, lossTotals_] := Lookup[pathTotals, path, lossTotals[path]];

getCEClasses[cePath_, net_NetP] := Scope[
	ceTargetPath = Join[cePath, NetPath["Inputs", "Target"]];
	graph = PortConnectivityGraph[net];

	Quiet[
		(* this graph stuff could fail, in which case we'll just
		use integer class labels *)
		classes = $Failed;

		upstream = VertexInComponent[graph, ceTargetPath];
		inpath = FirstCase[upstream, NetPath["Inputs", _]];
		(* ^ look up the input that feeds the target of the CE layer *)

		input = net @@ inpath;
		classes = Match[input, 
			EncoderP["Class", data_] :> data["Labels"],
			$Failed
		];
	];

	If[FailureQ[classes], 
		classes = net @@ Join[cePath, NetPath["Parameters", "$Classes"]];
		(* ^ the $Classes parameter is either an integer or None for binary *)
		If[IntegerQ[classes], classes = Range[classes]];
	];

	classes
]

General::hightopkchoice = "The choice of k for top-k measurements should be less than the number of classes, ``, but was ``."

checkKLessThanNumClasses[assoc_, classes_] := Scope[
	path = assoc[["Path"]];
	k = If[MatchQ[path, NetPath[___, "ErrorRate", _Integer]], Last @ path, -Infinity];

	numClasses = Length[classes];

	If[k >= numClasses, ThrowFailure["hightopkchoice", numClasses, k]];

	If[assoc["Measurement"] === "ConfusionMatrix" && numClasses > 100,
		(* turn off live plotting for large numbers of classes *)
		ReplacePart[assoc, "PlotFormat" -> "Disabled"]
	,
		assoc
	]
]

General::invioulayer = "The IntersectionOverUnion metric requires that the last dimmension of the input be 4, but it was ``."

checkLastDimIsFour[path_, net_] := Scope[
	sz = getFinalDimSize[path, net];
	If[sz =!= 4,
		ThrowFailure["invioulayer", sz];
	];
];

getFinalDimSize[path_, net_NetP] := Scope[
	inputPath = Join[path, NetPath["Inputs", "Input"]];
	input = net @@ inputPath;

	input /. TensorT[{___, n_}, _] -> n
]

General::invattenmetriclayer = "The AttentionWeights metric requires that the dimmensions of the AttentionLayer are all fixed, but some were \"Varying\"."

checkFixedLength[path_, net_NetP] := Scope[
	att = net @@ path;

	count = Count[{att[["Parameters", "$QueryShape"]], att[["Parameters", "$InputShape"]]}, {LengthVar[_], ___}];

	If[count =!= 0, ThrowFailure["invattenmetriclayer"]];
]

PackageScope["$metricInputLen"] 
PackageScope["$metricNumBatches"]
PackageScope["$validationFinalize"]
(* ^ to make debugging a little cleaner *)


PackageScope["CreateMetricObserver"]

CreateMetricObserver[metricObjects_, batchSize_] := ModuleScope[
	
	paths = DeleteDuplicates @ metricObjects[[All, "Path"]];
	(* this is slightly more complex, but has the benefit that if several of the metric objects
	share a single path (e.g. counts for Recall + Precision), then we only have one partial sum
	for the paths array *)

	{lossPaths, metricPaths} = SelectDiscard[paths, StringQ];
	
	$n = 0; $partialMetricTotal = 0; $partialLossTotal = 0;
	adder = With[path = path, makeCollector[1, $n++; 
		$partialMetricTotal += Lookup[$rawBatchMetrics, metricPaths];
		$partialLossTotal += Lookup[$rawBatchLosses, lossPaths];
	]];
	(* the added will increment n, and add on the raw metric value to the partial total *)

	$batchCollectors ^= Join[$batchCollectors, adder];

	$metricResults := AssociationThread[metricPaths, $partialMetricTotal];
	$lossResults := AssociationThread[lossPaths, $partialLossTotal / $n];
	
	$emptyMetrics = AssociationThread[metricObjects[[All, "Key"]], Missing["NotAvailable"]];

	Function @ Then[
		If[$n === 0, $emptyMetrics, 
			RoundMetricFinalize[metricObjects, $metricResults, $lossResults, $n * batchSize, $n, False]],
		$n = $partialMetricTotal = $partialLossTotal = 0
	]
];
