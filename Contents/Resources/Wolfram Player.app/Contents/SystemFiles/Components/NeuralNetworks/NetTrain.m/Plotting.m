Package["NeuralNetworks`"]


PackageExport["$ValidationLossPointStyle"]
PackageExport["$ValidationLossLineStyle"]
PackageExport["$LossPointStyle"]
PackageExport["$LossLineStyle"]

$ValidationLossPointStyle = Hue[.59, .7, .65];
$ValidationLossLineStyle = Hue[.59, .7, .75];
$LossPointStyle = Hue[0.083, 1., 0.76]; 
$LossLineStyle = Hue[0.083, 1., 1.];

PackageExport["$ProgressPanelPlotWidth"]
PackageExport["$ProgressPanelPlotAspectRatio"]

$ProgressPanelPlotWidth = 300;
$ProgressPanelPlotAspectRatio = 0.6;
	

makeFinalLossPlot[] := 
	makeMetricPlot[1, $ProgressPanelPlotWidth, True];

makeFinalPlots[] := 
	Association @ Table[
		$measurementsInfo[[i, "Key"]] -> makeMetricPlot[i, $ProgressPanelPlotWidth, True],
		{i, Length[$measurementsInfo]}
	]

updatePlotBoxes[] := (
	$lastLossPlotBoxes = ToBoxes @ makeMetricPlot[1, $ProgressPanelPlotWidth, False];
	If[$currentMetric > 0, $lastMetricPlotBoxes = ToBoxes @ makeMetricPlot[$currentMetric + 1, $ProgressPanelPlotWidth, False]];
);

makeMetricPlot[metricIdx_, width_, final_] := Scope[

	info = $measurementsInfo[[metricIdx]];

	UnpackAssociation[info, plotFormat, perBatch:"$PerBatch", preFormat, longName, key];

	If[plotFormat =!= None,

		If[plotFormat === "Disabled", Return @ "(disabled)"];

		trainValue = If[final && Length[$roundMeasurements] > 0,
			If[Length[$roundMeasurements] >= metricIdx, $roundMeasurements[[metricIdx]], None],
			(* ^ this length check is required because if training is stopped before the first round is finished
			$roundMeasurements will be empty*)
			$lastReportingObservation[key]
		];

		validValue = If[Length[$validationMeasurements] >= metricIdx, $validationMeasurements[[metricIdx]], None];

		Return @ If[MissingQ[trainValue] || trainValue === None, 
			TightLabeled[makeEmptyPlot[{1, 1} * width], {If[final, "  ", None], "  ", If[!final, "  ", None]}], 
			plot = plotFormat[trainValue, validValue, width, final];
			If[final && validValue =!= None, addValidationLegend @ plot, plot]
		]
	];

	If[$round > 125 || !perBatch,
		If[$round < 2, 
			tdata = None
		,
			data = Map[preFormat] @ BagContents @ $roundMeasurementsBags[[metricIdx]];
			tdata = RoundPlotData[{Range @ Length[data], data}]
		],
		(* because some measurements aren't batched, the indices don't line up necessarily *)
		index = IndexOf[$batchMeasurementsKeys, key];
		tdata = BatchPlotData[Map[preFormat] @ BagContents @ $batchMeasurementsBags[[index]]]
	];
	
	If[$doValidation,
		vdata = RoundPlotData[{BagContents @ $validationTimes, Map[preFormat] @ BagContents @ $validationMeasurementsBags[[metricIdx]]}],
		vdata = None
	];

	isIncreasing = info["Direction"] === "Increasing";
	plotRange = info["PlotRange"];
	isPercent = info["IsPercent"];
	plotType = info["PlotType"];

	MetricPlot[
		tdata, vdata, 
		longName, plotType, 
		$batchesPerRound, width, 
		isIncreasing, plotRange, isPercent,
		final
	]
]

PackageExport["RoundPlotData"]
PackageExport["BatchPlotData"]

SetUsage @ "
BatchPlotData[vector$, min$, max$] represents data for plotting by MetricPlot.
* vector$ is a vector of values, one per batch."

BatchPlotData[vector_] := BatchPlotData[vector, Min[vector], Max[vector]];
BatchPlotData[{}] := None;

SetUsage @ "
RoundPlotData[{xcoords$, ycoords$}, min$, max$] represents data for plotting by MetricPlot.
* matrix$ is a list of {x, y} pairs, where x is a round value, and y is a value."

RoundPlotData[{{}, _}] := None;
RoundPlotData[{xcoords_, ycoords_}] := RoundPlotData[{xcoords, ycoords}, Min[ycoords], Max[ycoords]];
(* this is for debugging *)

SetUsage @ "
MetricPlot[tdata$, vdata$, name$, axis$, bpr$, width$] plots the data.
* tdata$ and vdata$ can be RoundPlotData or BatchPlotData objects, or None.
* axis$ is one of Automatic, 'LogWhenPositive', 'Log', 'Linear'.
* bpr$ is batches per round.
* width$ is target width of the plot, in pixels."


poolStats[means_, vars_] := Mean[means^2 + vars^2] - Mean[means]^2;

(* TODO: strip off indeterminate / non numeric values before they get here *)

getMin[None] := Infinity; 
getMax[None] := -Infinity;
getMin[e_] := e[[2]];
getMax[e_] := e[[3]];

getRecentMinMax[None | {}] := {Infinity, -Infinity};
getRecentMinMax[e_] /; e[[-1, 1]] < 0.5 := MinMax[e]; (* don't start ignoring beginning until we're through 0.5 rounds *)
getRecentMinMax[e_] := MinMax @ Part[e, Span[Ceiling[Length[e] / 5], All], 2];

PackageExport["MetricPlot"]

$commonPlotOptions = Sequence[
	Frame -> True, Axes -> None, AspectRatio -> Full,
	BaseStyle -> {FontFamily -> "Verdana", FontSize -> 8, FontColor -> Gray, ScriptSizeMultipliers -> 0.2, ScriptMinSize -> 6},
	PlotRangePadding -> None, PlotRangeClipping -> True,
	Background -> White, FrameStyle -> Gray,
	ImagePadding -> 1, FrameTicks -> None
];

makeEmptyPlot[imageSize_] := Graphics[
	{{Rectangle[{0, 0}, {0, 0}]}},			
	PlotRange -> {{0, 1}, {0, 1}},
	ImageSize -> imageSize, 
	$commonPlotOptions
];

Clear[MetricPlot]
MetricPlot[training_, validation_, name_, type_, bpr_, width_, goingUp_:False, plotRange_:{-Infinity, Infinity}, isPercent_:False, final_:False] := Scope[
	
	min = Min[getMin @ training, getMin @ validation];
	max = Max[getMax @ training, getMax @ validation];

	useLog = Switch[type, 
		Automatic, min > 0 && max > 99 * min,
		"Log", min > 0,
		"Linear", False
	];
	tcoords = discretize[training, bpr, width, useLog];
	vcoords = discretize[validation, bpr, width, useLog];
	imageSize = {width, width * $ProgressPanelPlotAspectRatio} + 2;

	If[tcoords === {}, Return @ TightLabeled[makeEmptyPlot[imageSize], {"rounds", name, None}]];
	(* ^ if there is no data yet then display a placeholder.
	this will most likely be the case for non-batchwise metrics *)

	{bottom, top} = plotRange;

	(* if the measurements are asymptoting to a particular value, then 'zoom in' as we get closer to it. there are fewer
	benefits to zooming for log measurements, as log gives us high dynamic range. also, linear but unbounded measurements
	don't have a particular spatial scale that we can know apriori *)
	If[!useLog && goingUp && NumberQ[top] && !final, 
		min = Min[First @ getRecentMinMax @ tcoords, First @ getRecentMinMax @ vcoords]];
	If[!useLog && !goingUp && NumberQ[bottom] && !final, 
		max = Max[Last @ getRecentMinMax @ tcoords, Last @ getRecentMinMax @ vcoords]];

	If[!goingUp && NumberQ[bottom] && !(useLog && bottom == 0), min = Min[min, bottom]];
	If[goingUp && NumberQ[top], max = Max[max, top]];
	If[useLog, {min, max} = Log10 @ {min, max}];
	diff = (max - min) + $MachineEpsilon; If[isPercent, diff = Max[diff, 0.000001]]; 
	If[useLog, 
		If[max - min < 1.0, {min, max} = Mean[{min, max}] + {-0.5, 0.5}];
		min = Floor[min-diff/25., 0.1]; max = Ceiling[max+diff/25., 0.1];
	,
		{m, e} = MantissaExponent[diff]; scale = Round[m, .05] * Power[10, e]; 
		mean = Mean[{min, max}];
		min = Min[min, mean - scale / 2]; max = Max[max, mean + scale / 2];
		min = Floor[min, scale / 20]; max = Ceiling[max, scale / 20];
	];
	xmax = Max[tcoords[[-1, 1]], If[vcoords =!= {}, vcoords[[-1, 1]], 0]];
	(* basic summary of above code:
	we want to enlarge the plot bounds to 'nice' values. how much we enlarge them
	depends on the dynamic range of the original data. *)
	isBatch = xmax < 1;
	If[useLog,
		{ylabels, ylines} = makeLogTicks[Floor @ min, Ceiling @ max],
		min = Max[min, bottom]; max = Min[max, top];
		{ylabels, ylines, tmp, tmp} = makeLinearTicks[min, max, If[isPercent, PercentForm, ScientificForm]]
	];
	dy = (max - min)/100.; min -= dy; max += dy;
	{xlabels, xlines} = makeTimeTicks[xmax, If[isBatch, bpr, 1], {min, max}, goingUp];
	graphics = Graphics[
		{
			AbsoluteThickness[1.25], 
			(*If[$compactPlot, Nothing, 
			{Text[Round[#/dx], Offset[{-2,0}, {#, timeLabelHeight}], {1.,1.}]& /@ dividers, lossLabels}],*)
			makeLines[tcoords, $LossPointStyle, $LossLineStyle],
			makeLines[vcoords, $ValidationLossPointStyle, $ValidationLossLineStyle]
		},
		Prolog -> {xlabels, ylabels},
		PlotRange -> {{0, xmax + 0.5/bpr}, {min, max}},
		ImageSize -> imageSize, 
		GridLines -> {xlines, ylines}, 
		$commonPlotOptions
	];
	
	timeLabel = If[isBatch, "batches", "rounds"];
	If[final,
		graphics = TightLabeled[graphics, {None, name, timeLabel}];
		If[vcoords === {}, graphics, addValidationLegend @ graphics]
	,
		TightLabeled[graphics, {timeLabel, name, None}]
	]
];

addValidationLegend[graphics_] := 
	Legended[graphics, LineLegend[{$ValidationLossLineStyle, $LossLineStyle}, {"validation", "training"}]];


(* TODO: take min and max values during the binning/averaging process
or mean + sd. also, for min and max, we need to store seperately the
min, mean, and max from the begging of the metric tracking process, so that
when we aggregate in the plot, we take the min of the mins, the mean of the means,
the max of the maxes. *)

discretize[None, _, _, _] :=
	{};

(* discretize will try to achieve one point per pixel, roughly.
x axis is scaled such than one unit = one round
  *)
discretize[dense_BatchPlotData, bpr_, width_, useLog_] := 
	denseToSparseMean[First @ dense, bpr, width, useLog];

denseToSparseMean := denseToSparseMean = Compile[
	{{data, _Real, 1}, {bpr, _Integer}, {width, _Integer}, {useLog, True|False}}, 
	Module[{n, val, window, x, ibpr = 1.0 / bpr, dx}, 
	n = Length[data];
	(* size of averaging window to accomplish <= width points *)
	window = Ceiling[n / width];
	If[window <= 1, (* we can skip averaging *)
		Transpose[{
			Range[ibpr, n * ibpr, ibpr],
			If[useLog, Log10[data], data]
		}]
	,
		x = 0.; dx = window * ibpr;
		Table[
			val = Mean[data[[i ;; Min[n, i+window-1]]]];
			If[useLog, val = Log10[val]]; (* ??? or mean of logs *)
			x += dx;
			{x, val}
		, 
			{i, 1, Max[n-1, 1], window}
		]
	]
]];

discretize[sparse_RoundPlotData, bpr_, width_, useLog_] :=
	DeleteCases[{0., 0.}] @ sparseMean[sparse[[1,1]], sparse[[1, 2]], width, useLog];

sparseMean := sparseMean = Compile[
	{{xdata, _Real, 1}, {ydata, _Real, 1}, {width, _Integer}, {useLog, True|False}},
	Module[{n, bins, ytotals, ycounts, bin, lastx, ihscale},
	n = Length[xdata];
	lastx = xdata[[-1]];
	ihscale = width / lastx;
	ytotals = ConstantArray[0., width];
	ycounts = ConstantArray[0., width];
	bins = Round[xdata * ihscale];
	Do[
		bin = Max[bins[[i]], 1];
		ytotals[[bin]] += ydata[[i]];
		ycounts[[bin]] += 1;
	,
		{i, 1, n}
	];

	Module[{yvals, pairs},
		yvals = ytotals / (Min[Max[#, 1], n]& /@ ycounts);
		If[useLog, yvals = Map[Function[If[# == 0, #, Log10[#]]], yvals]];
		(* ^ it is okay not to log the 0s here because the 0s will all get filtered out anyway *)
		pairs = Transpose[{
			Range[0., lastx, lastx / (width - 1.)],
			yvals
		}];
		pairs * (Min[Max[#, -1.], 1.]& /@ ycounts)
	]
]];

(* TODO: extend to make the s.d. envelopes *)
makeLines[coords_, pointStyle_, lineStyle_] := Scope[
	n = Length[coords];
	lines = {lineStyle, Line @ coords};
	points = If[n < 120, {AbsolutePointSize[1.5 + 3/(1 + n/35)], pointStyle, Point @ coords}, {}];
	{lines, points}
];

makeTimeTicks[n_, xscale_, {ymin_, ymax_}, labelOnBottom_] := Scope[
	{man, exp} = MantissaExponent[N[n * xscale]];
	If[labelOnBottom, 
		ypos = ymin; yoffset = -1.0; ynudge = 1,
		ypos = ymax; yoffset = 1.0; ynudge = -2
	];
	positions = Which[
		man > 0.5, {0.2, 0.4, 0.6, 0.8, 1.0},
		man > 0.3, {0.1, 0.2, 0.3, 0.4, 0.5},
		man > 0.15, {0.05, 0.1, 0.15, 0.2, 0.25, 0.3},
		man == 0.10, {0.02, 0.04, 0.06, 0.08, 0.1},
		True, {0.025, 0.05, 0.075, 0.100, 0.125, 0.150}
	] * Power[10, exp];
	positions = Select[positions, # == Round[#]&];
	labels = Text[Round[#], Offset[{-2, ynudge}, {# / xscale, ypos}], {1., yoffset}]& /@ positions;
	{labels, positions / xscale}
];
	
errtxt[0|0.] := "";
errtxt[r_] := Scope[
	txt = TextString[r];
	If[StringEndsQ[txt, ".0"] && $doTrim, 
		StringDrop[txt, -2],
		txt
	] <> "%"
];

makeLegend[] := Scope[
	items = MapThread[
		{GraphicsBox[{}, Background -> #2, ImageSize -> {13,1}, 
			ImagePadding -> {{0, 0}, {5,2}}, Axes -> None],"  ", #1}&,
		{{"\"training set\"", "\"validation set\""},
		{$LossLineStyle, $ValidationLossLineStyle}}
	];
	StyleBox[RowBox @ Flatten @ Riffle[items, "\t"], FontSize -> 8, FontFamily -> "Verdana", FontColor -> GrayLevel[0.4]]
];

$lightGridLine = GrayLevel[0.8501];
$darkGridLine = GrayLevel[0.3001];

Clear[makeLinearTicks];
makeLinearTicks[min_, max_, form_:EngineeringForm] := Scope[
	{major, minor} = N@FindDivisions[{min, max}, {4,5}];
	diff = (max - min)/100.0; 
	labels = If[min <= # + diff <= max, Text[form[#], Offset[{3,0}, {0, #}], {-1,-1.}], Nothing]& /@ major;
	{min, max} = MinMax[major];
	{labels, Join[{#, $lightGridLine}& /@ Flatten[minor], {#, $darkGridLine}& /@ major], min, max}
];

Clear[makeLogTicks];
makeLogTicks[min_, max_] := makeLogTicks[min, max] = Scope[
	major = Range[N[min], max-1]; 
	minor = Which[
		max > min + 4, {}, 
		max > min + 2, makeMinorLogTicks[min, max, 2], 
		True, makeMinorLogTicks[min, max, 1]
	];
	diff = (max - min)/100.0; 
	labels = If[min <= # - diff <= max, Text[power10Label @ Round @ #, Offset[{3,0}, {0, #}], {-1,-1.}], Nothing]& /@ major;
	{labels, Join[{#, $darkGridLine}& /@ major, {#, $lightGridLine}& /@ minor]}
];

power10Label[1] := "10";
power10Label[0] := "1";
power10Label[n_] := Superscript[10, TextString @ n];

makeMinorLogTicks[min_, max_, sub_] := 
	Log[10, Flatten @ Table[Range[2*(10.^i),10^(i+0.999),sub*10^i],{i,min,max-1}]];

