Package["NeuralNetworks`"]

NetTrain::invhistprop = "\"Property\" key of specification `` should give a pure function or valid property name. See the reference page for TrainingProgressFunction for the available properties."
NetTrain::invhistform = "Value of \"Form\" key of specification `` should be one of ``."
NetTrain::invhistkey = "Key `` in property specification should be one of one of ``."
NetTrain::invhistplotopts = "Value for \"PlotOptions\" should be an association or list of rules."

$allowedHistoryKeys = {"Property", "Form", "Interval", "Event", "PlotOptions"};

parseHistory[assoc_] := Scope[
	FailIfExtraKeys[assoc, $allowedHistoryKeys, "property specification association"];
	UnpackAssociation[assoc, property, form, interval, event, plotOptions, "Default" -> None];
	If[StringQ[interval], interval = {1, interval}];
	periodOpt = If[interval === None, If[event === None, "Interval" -> "Round", "Event" -> event], "Interval" -> interval];
	With[prop = property, expr = Switch[prop,
		_Function,
			checkCallbackKeyUsage[prop];
			Hold @ prop[$livePropertyData],
		$livePropertyP | {$livePropertyP..},
			Hold @ Lookup[$livePropertyData, prop],
		_,
			ThrowFailure["invhistprop", assoc]
	]];
	bag = addPeriodicCollector[expr, property, periodOpt];
	resultSpec = "$Bag"[bag];
	form = Lookup[assoc, "Form", "List"];
	Switch[form,
		"List", 
			Null,
		"TransposedList", 
			AppendTo[resultSpec, assocTranspose],
		"Plot" | "EvolutionPlot", 
			If[plotOptions === None, plotOptions = <||>];
			If[!AssociationQ[plotOptions] && !VectorQ[plotOptions, RuleQ], ThrowFailure["invhistplotopts", plotOptions]];
			plotter = superPlotWithOpts @ Append[$spopts, plotOptions];
			AppendTo[resultSpec, UnpackNArrays /* If[ListQ[property], multiPlot[plotter], plotter]],
		Null, 
			ThrowFailure["invhistform", property, {"List", "TransposedList", "Plot"}]
	];
	resultSpec
]

assocTranspose[{}] := Missing["NoData"];
assocTranspose[list_List ? AssociationVectorQ] := AssociationTranspose[list];
assocTranspose[list_List] := Quiet @ Check[Transpose @ list, Missing["NotRectangular"]];
assocTranspose[e_] := e;

UnpackNArrays[e_] := ReplaceAll[e, na_NumericArray :> RuleCondition @ Normal @ na];
	
addPeriodicCollector[body_, name_, "Interval" -> s_String] := 
	addPeriodicCollector[body, name, "Interval" -> {1, s}];

addPeriodicCollector[Hold[body_], name_, "Interval" -> HoldPattern @ q:(Quantity[n_, type_] | {n_, type_String})] := Module[
	{bag = Internal`Bag[], hold, callback},
	hold = If[n === 1 && MatchQ[type, "Round"|"Rounds"|"Batch"|"Batches"], 
		makeCollector[2, BagPush[bag, body]] 
		(* if we are doing it exactly once every round/batch, then we can just use the round/batch collector mechanism unchanged *)
	, 
		callback = makePeriodicFunction[name, BagPush[bag, body]&, "Interval" -> q, "ForceZero" -> MatchQ[type, "Round"|"Rounds"]];
		makeCollector[2, callback[$clock]]
		(* if we are doing it at a different frequency, then we must make a periodic function to decide whether to fire or not,
		but we can still use the associated collector to drive the periodic function *)
	];
	Switch[type,
		"Round"|"Rounds" /; IntegerQ[n], JoinTo[$roundCollectors, hold],
		"Round"|"Rounds"|"Batch"|"Batches", JoinTo[$batchCollectors, hold],
		"Second"|"Seconds"|"Minute"|"Minutes"|"Hour"|"Hours"|"Percent", JoinTo[$batchCollectors, hold],
		_, ThrowFailure[NetTrain::invcollperiod, "Interval" -> q, name]
	];
	bag
];

addPeriodicCollector[Hold[body_], name_, "Event" -> ev_] := Module[
	{bag = Internal`Bag[]},
	makePeriodicFunction[name, BagPush[bag, body]&, "Event" -> ev];
];

NetTrain::invcollperiod = "Invalid option `` specified for property ``.";
addPeriodicCollector[_, name_, rule_] := ThrowFailure[NetTrain::invcollperiod, rule, name];

multiPlot[plotter_][data_] := Column[Map[plotter, Transpose[data]]];

$spopts = <|"Smoothing" -> None, ScalingFunctions -> Automatic, "PlotLabels" -> False|>;

superPlotWithOpts[opts_][data_] := Scope[
	$spopts = opts;
	superPlot @ data
];

Clear[superPlot, autoPlot];

superPlot[{}] := Missing["NoData"];

(* if its a series of associations, transpose it to be an association of series *)
superPlot[data_List ? AssociationVectorQ] := Scope[
	series = Map[ToPackedArray, AssociationTranspose[data]];
	series = toScalarSeries /@ series;
	autoPlot @ series
];

toLabel[a_List] := Text @ Row[a, Style[" \[RightPointer] ", Gray]];
toLabel[e_] := Text[e];

(* if it looks like it was a timeseries of a small number of values, 
plot those individual values separately *) 
superPlot[data_List /; ArrayQ[data, 2, NumericQ] && Last[Dimensions[data]] < 8] := 
	autoPlot @ Transpose @ data;

(* if its a vector of numbers, just plot it as-is *)
superPlot[data_List /; ArrayQ[data, 1, NumericQ]] := 
	autoPlot @ data;

(* if its a higher-order tensor, take the mean of sub-arrays *)
superPlot[data_List] := 
	autoPlot @ toScalarSeries @ data

superPlot[_] := Missing["NonNumericData"]

(* this tries to reduce series of arrays to be series of scalars so that 
can form the individual y-values of a plot *)
toScalarSeries[series_ ? VectorQ] := 
	series;

toScalarSeries[series_] := Quiet @ Check[
	Mean[deleteNonNumeric @ Flatten @ #]& /@ series,
	series
];

deleteNonNumeric[a_ ? PackedArrayQ] := a;
deleteNonNumeric[a_] := DeleteCases[a, _Missing | None | Indeterminate, Infinity];

NetTrain::badhistplotvals = "Collected property data appeared to contain unplottable values. Plot may appear empty."

(* autoplot accepts either lists of individual series, or associations of individual series *)
autoPlot[data_] := Scope[
	{min, max} = MinMax @ deleteNonNumeric @ If[AssociationQ[data], Values[data], data];
	scaling = $spopts[ScalingFunctions] /. {Log -> "Log", Log10 -> "Log10"};
	If[MatchQ[scaling, {_, _}], {hscaling, vscaling} = scaling, hscaling = None; vscaling = scaling];
	SetAutomatic[hscaling, None];
	SetAutomatic[vscaling, If[min > 0 && max > 0 && Log10[max] - Log10[min] > 3, "Log10", None]];
	isMulti = MatrixQ[data] || AssociationQ[data];
	mapper = If[isMulti, Map, Compose];
	lnq = looksNumericQ[data];
	If[!lnq, Message[NetTrain::badhistplotvals]];
	smoothing = $spopts["Smoothing"];
	If[lnq, 
		odata = data;
		Quiet @ Check[
			If[smoothing =!= None, data = mapper[smooth[#, smoothing]&, data]];
			len = If[isMulti, Length @ First @ data, Length @ data];
			If[len > 800, data = mapper[blockMean] @ data];
		,
			data = odata (* give up on post-processing *)
		]; 
	];
	opts = KeyDrop[$spopts, {ScalingFunctions, "Smoothing", "PlotLabels"}];
	ListLinePlot[data, 
		Sequence @@ (Normal @ opts),
		AspectRatio -> 1/Sqrt[2.], 
		PlotRangePadding -> {0, Automatic},
		Sequence @@ If[!$spopts["PlotLabels"],
			{
				PlotLegends -> If[AssociationQ[data], LineLegend[Map[toLabel, Keys @ data]], None],
				ImageSize -> 400, 
				ImagePadding -> {{30, 10}, {15, 5}}	
			}
		,
			{
				PlotLabels -> If[AssociationQ[data], Keys @ data, None],
				ImageSize -> 500, 
				PlotLegends -> None,
				ImagePadding -> All	
			}
		],
		PlotRange -> {All, All}, Frame -> True, 
		ScalingFunctions -> {hscaling, vscaling}
	]
];

looksNumericQ[<||> | {}] := False;
looksNumericQ[data_Association | data_List] := AnyTrue[data, looksNumericQ]; 
looksNumericQ[HoldPattern @ q_Quantity] := True;
looksNumericQ[a_] := NumericQ[a];

smooth[data_, None] := data;
smooth[data_, n_] := If[Length[data] < n, data, MovingAverage[data, n]];

blockMean[data_] := Scope[
	n = Floor[Length[data] / 100]; i = 0;
	BlockMap[{1 + (i++ * n), Mean[#]}&, data, n]
];


