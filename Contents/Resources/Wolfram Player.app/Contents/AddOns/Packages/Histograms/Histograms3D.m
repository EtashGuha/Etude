Quiet[BeginPackage["Histograms`", "BarCharts`"], {General::obspkg}]

(* Histogram3D *)

(* Histogram3D does not have the BarChart3D options XSpacing or YSpacing. 
	The option HistogramBins functions like the option
        PlotPoints;  HistogramRange functions like PlotRange.
*)


Options[Histograms`Histogram3D] =
	Sort[{
	ApproximateIntervals -> Automatic,
	FrequencyData -> False,
	HistogramCategories -> Automatic,
        HistogramRange -> Automatic,
	HistogramScale -> Automatic,
	BarEdges -> True,		 (* opt of GeneralizedBarChart3D *)
	BarEdgeStyle -> Black, (* opt of GeneralizedBarChart3D *)	
	BarStyle -> White	 (* opt of GeneralizedBarChart3D *)
	}~Join~Options[Graphics3D]];

SetOptions[Histograms`Histogram3D, PlotRange -> All, BoxRatios -> {1,1,1},
	Axes -> Automatic, Ticks -> Automatic, Lighting->Automatic]



Begin["`Private`"]


(* Note: Histogram3D calls RangeCounts to compute frequencies
        and GeneralizedBarChart3D for plotting. *)
(* Note:
rc = RangeCounts[data2D, {0, .2, .4, .6, .8, 1.0}, {0, .5, 1.0}];
Dimensions[rc] -> {7, 4}
Dimensions[ Map[#[[{2, 3}]]&, rc[[{2, 3, 4, 5, 6}]] ] ] ->  {5, 2}
*)

Histograms`Histogram3D[mat_?MatrixQ, opts___?OptionQ] :=
 Module[{approximate, fdata, hcat, range, scale, sbedges, sbedgestyle, sbstyle,
	 ticks, dim,
	 rangeX, rangeY, (* x and y components of HistogramRange *)	 
	 countdata, numberOfBinsX, numberOfBinsY, countdataX, countdataY,
	 hcatX, hcatY, (* x and y components of HistogramCategories *)
	dataminX, datamaxX, (* min and max as determined by x-component of
				data and option HistogramRange *)
	dataminY, datamaxY, (* min and max as determined by y-component of
                                data and option HistogramRange *)
	cutoffsX, cutoffsY, 
	binminX, binmaxX, (* min and max of x coordinate bin boundaries *)
	binminY, binmaxY, (* min and max of y coordinate bin boundaries *)
	listX, listY, totalcount, cutoffs,
	leftTailCountX, rightTailCountX, leftTailCountY, rightTailCountY,
	binwidthsX, binwidthsY, binareas, binareasFlatList,
	bincentersX, bincentersY, axisticksX, axisticksY,
	bincenters, binwidths,
	phwdata, (* position-height-width data for GeneralizedBarChart3D *)
	groptslist,
	volume	(* volume of histogram; used for scaling z axis so that
			histogram has unit volume *)
	},
  (

     (* Error check for HistogramRange setting. *)
     If[range === Automatic, range = {Automatic, Automatic}];
     If[!( ListQ[range] && Length[range] == 2 ),
	   Message[Histogram3D::badrg, range];
           range = {Automatic, Automatic}   ];
     {rangeX, rangeY} = range;		

     If[TrueQ[fdata],

        (* PROCESS MATRIX assuming that it represents FREQUENCIES
	   on a 2D grid.  That is,
	   Histogram3D[{{7, 4}, {1, 4}, {3, 2}, {1, 3}, {2, 3}},
		FrequencyData -> True,
		HistogramCategories -> {{cx0, cx1, cx2, cx3, cx4, cx5},
			{cy0, cy1, cy2}}]
	   implies that there are 3 data in the range 
		cx2 <= x < cx3 && cy0 <= y <= cy1 .
	*)
        (* Define countdata, numberOfBins, xmin, xmax, cutoffs. *)
         

        countdata = mat;
	{numberOfBinsX, numberOfBinsY} = Dimensions[countdata]; 
	countdataX = Map[Apply[Plus, #]&, countdata];
	countdataY = Map[Apply[Plus, #]&, Transpose[countdata]];

	(* Error check for HistogramCategories setting. *)
	If[hcat === Automatic, hcat = {Automatic, Automatic}];
	If[!( ListQ[hcat] && Length[hcat] == 2 &&
	     (hcat[[1]]===Automatic || monotoneIncreasingVectorQ[hcat[[1]]]) &&
	     (hcat[[2]]===Automatic || monotoneIncreasingVectorQ[hcat[[2]]]) ),
           Message[Histogram3D::fdhc, hcat];
           hcat = {Automatic, Automatic}   ];
	{hcatX, hcatY} = hcat;

	{dataminX, datamaxX} = findRange[rangeX,
		If[hcatX === Automatic, Range[0, numberOfBinsX], hcatX]
		];
	{dataminY, datamaxY} = findRange[rangeY,
		If[hcatY === Automatic, Range[0, numberOfBinsY], hcatY]
                ];

	If[hcatX === Automatic,
                cutoffsX = dataminX + (datamaxX-dataminX)/numberOfBinsX*
			Range[0, numberOfBinsX],
                cutoffsX = findCutoffs1[hcatX, dataminX, datamaxX, countdataX];
                numberOfBinsX = Length[cutoffsX]-1
           ];
        {binminX, binmaxX} = {First[cutoffsX], Last[cutoffsX]};
	If[hcatY === Automatic,
                cutoffsY = dataminY + (datamaxY-dataminY)/numberOfBinsY*
			Range[0, numberOfBinsY],
                cutoffsY = findCutoffs1[hcatY, dataminY, datamaxY, countdataY];
                numberOfBinsY = Length[cutoffsY]-1
           ];
        {binminY, binmaxY} = {First[cutoffsY], Last[cutoffsY]},

	
	(* ===================================================== *)
        (* PROCESS LIST assuming that it represents RAW DATA. *)
        (* Define countdata, numberOfBins, binmin, binmax, cutoffs. *)
        (* ===================================================== *)

        (* === Define min and max of rangeX, and count x data in range. ==== *)
	listX = Map[First, mat];
        {dataminX, datamaxX} = findRange[rangeX, listX];
        (* === Define min and max of rangeY, and count y data in range. ==== *)
	listY = Map[Last, mat];
        {dataminY, datamaxY} = findRange[rangeY, listY];

	(* ================== Count data in range. ===================== *)
	totalcount = Count[mat, {x_, y_} /; dataminX <= x <= datamaxX &&
						dataminY <= y <= datamaxY];

	(* =========== Define category cutoffs for raw data. =========== *)
	(* Error check for HistogramCategories setting. *)
        (* Note that RangeCounts considers intervals of the form
                {{binminX <= x < etc, binminY <= y < etc}, ..., 
		 {etc <= x < binmaxX, etc <= y < binmaxY}}. *)
	(* Define {cutoffsX, cutoffsY. *)
	If[hcat === Automatic || PositiveIntegerQ[hcat],
	   (* look at the distribution as a whole *)
	   cutoffs = findCutoffs3[hcat,
		 {dataminX, datamaxX}, {dataminY, datamaxY}, 
		totalcount, approximate];
	   If[Head[cutoffs] === findCutoffs3, Return[$Failed]];
	   {cutoffsX, cutoffsY} = cutoffs,	
	   (* look at the x and y components separately *)
	   If[ListQ[hcat] && Length[hcat] == 2 &&
	      (hcat[[1]]===Automatic || monotoneIncreasingVectorQ[hcat[[1]]] ||
		PositiveIntegerQ[hcat[[1]]]) &&
	      (hcat[[2]]===Automatic || monotoneIncreasingVectorQ[hcat[[2]]] ||
		PositiveIntegerQ[hcat[[2]]]) ,
	      (* = HistogramCategories points to a valid list of two items. = *)
	      {hcatX, hcatY} = hcat;
              cutoffsX = findCutoffs4[hcatX, dataminX, datamaxX,
		 If[IntegerQ[hcatY], totalcount/hcatY, (totalcount)^(2/3)],
                 approximate];
              cutoffsY = findCutoffs4[hcatY, dataminY, datamaxY,
		 If[IntegerQ[hcatX], totalcount/hcatX, (totalcount)^(2/3)],
                 approximate],
	      (* ==== HistogramCategories points to an invalid setting. ==== *)	
              Message[Histogram3D::rdhc, hcat];
	      cutoffs = findCutoffs3[hcat,
		 {dataminX, datamaxX}, {dataminY, datamaxY}, 
		totalcount, approximate];
              If[Head[cutoffs] === findCutoffs3, Return[$Failed]];
              {cutoffsX, cutoffsY} = cutoffs	
	   ]
	];
        numberOfBinsX = Length[cutoffsX]-1;
        {binminX, binmaxX} = {First[cutoffsX], Last[cutoffsX]};
        numberOfBinsY = Length[cutoffsY]-1;
        {binminY, binmaxY} = {First[cutoffsY], Last[cutoffsY]};


	(* ========== Warning messages for points not plotted. ======== *)
	(* If histogram range is to be determined automatically, *)
        (* presumably because the user wants all points to be plotted, *)
        (* warn user if some points will be excluded from histogram. *)
        If[ (rangeX === Automatic ||
             (VectorQ[rangeX] && Length[rangeX] == 2 &&
                 rangeX[[1]] === Automatic) ) &&
                        dataminX < binminX,
             leftTailCountX = Count[mat, z_ /; z[[1]] < binminX];
             If[leftTailCountX == 1,
                Message[Histogram3D::lt1, leftTailCountX, "x", binminX] ];
             If[leftTailCountX > 1,
                Message[Histogram3D::lt, leftTailCountX, "x", binminX] ]
	];
        If[ (rangeX === Automatic ||
             (VectorQ[rangeX] && Length[rangeX] == 2 &&
                 rangeX[[2]] === Automatic) ) &&
                        datamaxX >= binmaxX,
             rightTailCountX = Count[mat, z_ /; z[[1]] >= binmaxX];
             If[rightTailCountX == 1,
                Message[Histogram3D::gtet1, rightTailCountX, "x", binmaxX] ];
             If[rightTailCountX > 1,
                Message[Histogram3D::gtet, rightTailCountX, "x", binmaxX] ]
	];
        If[ (rangeY === Automatic ||
             (VectorQ[rangeY] && Length[rangeY] == 2 &&
                 rangeY[[1]] === Automatic) ) &&
                        dataminY < binminY,
             leftTailCountY = Count[mat, z_ /; z[[2]] < binminY];
             If[leftTailCountY == 1,
                Message[Histogram3D::lt1, leftTailCountY, "y", binminY] ];
             If[leftTailCountY > 1,
                Message[Histogram3D::lt, leftTailCountY, "y", binminY] ];
	];
        If[ (rangeY === Automatic ||
             (VectorQ[rangeY] && Length[rangeY] == 2 &&
                 rangeY[[2]] === Automatic) ) &&
                        datamaxY >= binmaxY,
             rightTailCountY = Count[mat, z_ /; z[[2]] >= binmaxY];
             If[rightTailCountY == 1,
                Message[Histogram3D::gtet1, rightTailCountY, "y", binmaxY] ];
             If[rightTailCountY > 1,
                Message[Histogram3D::gtet, rightTailCountY, "y", binmaxY] ]
	];

	(* =========== Compute category counts for raw data. =========== *)
        countdata = BinCounts[mat, {cutoffsX}, {cutoffsY}];
        If[Head[countdata] === BinCounts,
                Message[Histogram3D::rcount];
                Return[$Failed] ];
        ]; (* end If TrueQ[fdata] *)
	
     (* ============================================================= *)
     (* ============================================================= *)
     (* Use countdata, cutoffsX, cutoffsY, numberOfBinsX, numberOfBinsY *)
     (* 	 binminX, binmaxX, binminY, and binmaxY to   *)
     (*              generate histogram.                              *)
     (* ============================================================= *)
     (* ============================================================= *)

     (* ================= Scale category counts. ================ *)
     binwidthsX = Drop[cutoffsX, 1] - Drop[cutoffsX, -1];
     binwidthsY = Drop[cutoffsY, 1] - Drop[cutoffsY, -1];
     binareas = Outer[Times, binwidthsX, binwidthsY];
     binareasFlatList = Flatten[binareas];
     If[TrueQ[scale] || !Apply[Equal, binareasFlatList],
           (* Make the volume of the solid bar proportional to the frequency
		associated with the bar. *)
	   countdata = countdata/binareas
     ];
     bincentersX = Drop[FoldList[Plus, binminX, binwidthsX], -1] +
                1/2 binwidthsX;
     bincentersY = Drop[FoldList[Plus, binminY, binwidthsY], -1] +
                1/2 binwidthsY;

     (* =============================================================== *)
     (*  Define category axis ticks from                                *)
     (*      bincenters, countdata, and ticks.                          *)
     (* =============================================================== *)
     (* Note that it is not possible to figure out what Automatic setting of
	   category axis ticks would be for, say, ScatterPlot3D of
	   data3D = Flatten[MapThread[Append, {bincenters, countdata}, 2], 1],
	   where bincenters = Outer[List, bincentersX, bincentersY],
	   because FullOptions does not work for extracting the Ticks setting
           of a Graphics3D object.  So "autoticks" cannot be defined as in
	   the case of Graphics`Graphics`Histogram. ECM '97. *)	

     (* Process the Ticks setting. *)
     If[MatchQ[ticks, Automatic | IntervalCenters | IntervalBoundaries],
                ticks = {ticks, ticks, Automatic}];
     If[ticks === None, ticks = {None, None, None}];
     (* Check the Ticks setting, and reset to Automatic if the setting is
                illegal. *)
     If[!(ListQ[ticks] && Length[ticks] == 3 && ticksCheckQ[ticks]),
           Message[Histogram3D::ticks, ticks];
           ticks = {Automatic, Automatic, Automatic}];
     axisticksX = Switch[ticks[[1]],
                _?ListQ, (* ticksCheckQ has already checked for
                                monotoneIncreasingVectorQ *)
                   Map[neatTick, ticks[[1]] ],
                IntervalBoundaries,
                   (
                     Map[neatTick, cutoffsX]
                   ),
                IntervalCenters,
                   (
                     Map[neatTick, bincentersX]
                   ),
                None, (* no category axis ticks *)
                        None,
                _, (* place x axis ticks automatically *)
                        Automatic
        ];
     axisticksY = Switch[ticks[[2]],
                _?ListQ, (* ticksCheckQ has already checked for
                                monotoneIncreasingVectorQ *)
                        Map[neatTick, ticks[[2]] ],
                IntervalBoundaries,
                   (
                     Map[neatTick, cutoffsY]
                   ),
                IntervalCenters,
                   (
                     Map[neatTick, bincentersY]
                   ),
                None, (* no category axis ticks *)
                        None,
                _, (* place y axis ticks automatically *)
                        Automatic
        ];
     ticks = {axisticksX, axisticksY, ticks[[3]]};

     (* ==================================================================== *)
     (* ======= Define phwdata (position, height, width). ================== *)
     (* ======= { {{xpos1, ypos1}, height1, {xwidth1, ywidth1}}, ...} ====== *)
     (* ==================================================================== *)
     (* Note that BarSpacing is assumed to be 0 here.  *)
     (* If you want to add that option to Histogram3D, the option should be *)
     (* processed here.  (Some would say that histograms of discrete data *)
     (* ought to have columns separated from each other, i.e., with  *)
     (* BarSpacing greater than zero, depending on whether the *)
     (* x or y variable is discrete.) *)

     bincenters = Outer[List, bincentersX, bincentersY]; (* position *)
     binwidths = Outer[List, binwidthsX, binwidthsY]; (* widths *)
     phwdata = Flatten[
	MapThread[Join,
		 {bincenters, Map[List, countdata, {2}], binwidths},
		 2], 1];
     (* Now phwdata has the form {{xpos1, ypos1, height1, xwidth1, ywidth1},
	{xpos2, ypos2, height2, xwidth2, ywidth2}...}. *)
     phwdata = Map[{#[[{1, 2}]], #[[3]], #[[{4, 5}]]}&, phwdata];


     (* ============= Extract any other options relevent to Graphics3D. ==== *)
     groptslist = FilterRules[{opts}, Options[Graphics3D]];

     (* ====== Scale solid bar heights according to HistogramScale -> k ==== *)
     (* NOTE that phwdata has the form...
	{  {{xpos1, ypos1}, height1, {xwidth1, ywidth1}},
	   {{xpos2, ypos2}, height2, {xwidth2, ywidth2}}, ...} *)
     If[NumberQ[scale] && FreeQ[scale, Complex] && scale > 0,
        volume = Apply[Plus, Map[(#[[2]]#[[3, 1]]#[[3, 2]])&, phwdata]];
	phwdata = Map[{#[[1]], #[[2]] scale/volume, #[[3]]}&, phwdata]
     ];

     (* =================== GeneralizedBarChart3D =================== *)
     BarCharts`GeneralizedBarChart3D[phwdata,
	BarEdges->sbedges,	(* option of GeneralizedBarChart3D *)
   	BarEdgeStyle->sbedgestyle, (* option of GeneralizedBarChart3D *)
   	BarStyle->sbstyle, (* option of GeneralizedBarChart3D *)
	Ticks -> ticks,		(* option of Graphics3D *)
	(* groptslist includes any other options relevent to Graphics3D *)
	Apply[Sequence, groptslist]
     ]
  ) /;  (
	{approximate, fdata, hcat, range, scale, sbedges, sbedgestyle, sbstyle, 
	 ticks} =
	 {ApproximateIntervals, FrequencyData, HistogramCategories,
	  HistogramRange, HistogramScale,
		BarEdges, BarEdgeStyle, BarStyle,
	  	Ticks} /. {opts} /. Options[Histograms`Histogram3D];
	dim = Dimensions[mat];
	If[TrueQ[fdata] && ListQ[hcat] && Length[hcat] == 2,
	   If[VectorQ[hcat[[1]]],
	      If[dim[[1]]+1 == Length[hcat[[1]]],
		 True,
		 Message[Histogram3D::fdfail];  False],
	      True] &&
	   If[VectorQ[hcat[[2]]],
	      If[dim[[2]]+1 == Length[hcat[[2]]],
	         True,
		 Message[Histogram3D::fdfail];  False],
	      True],
           True] &&
	If[!TrueQ[fdata],
	   If[Length[mat[[1]]] != 2,
	      Message[Histogram3D::rd2d];  False,
	      True],
	   True]	
        )
	
 ] (* end Histogram3D *)
 

(* Interpret the HistogramCategories option when the
        data is frequency data. *)
findCutoffs1[hcat_, datamin_, datamax_, data_] :=
        Module[{countdata = data, cutoffs = hcat, n},
                (* If range specifies something more restrictive than
                        the given categories, then trim cutoffs. *)
                If[datamin >= First[cutoffs],
                   While[!(cutoffs[[1]] <= datamin < cutoffs[[2]]),
                        countdata = Drop[countdata, 1];
                        cutoffs = Drop[cutoffs, 1]]
                ];
                If[datamax < Last[cutoffs],
                   While[!(n = Length[cutoffs];
                       cutoffs[[n-1]] <= datamax < cutoffs[[n]]),
                        countdata = Drop[countdata, -1];
                        cutoffs = Drop[cutoffs, -1]]
                ];
                cutoffs
        ] (* end findCutoffs1 *)


(* Interpret the HistogramCategories option when the
        data is raw data, FOR A SINGLE COMPONENT x or y. *)
(* Example:
       findCutoffs4[hcatX, dataminX, datamaxX, totalcount, approximate]
*)
findCutoffs4[hcat_, datamin_, datamax_, totalcount_, approximate_] :=
   Module[{numberOfBins, cutoffs, binmin, binmax, bindelta},
           If[monotoneIncreasingVectorQ[hcat],
              (* Intervals are NOT approximated when they are
                        specifically requested using HistogramCategories. *)
              If[!(approximate===Automatic || approximate===False),
                         Message[Histogram3D::noapprox, approximate]];
              cutoffs = hcat;
              (* If range specifies something more restrictive than
                                the given categories, then trim cutoffs. *)
              If[datamin >= First[cutoffs],
                 While[!(cutoffs[[1]] <= datamin < cutoffs[[2]]),
                                cutoffs = Drop[cutoffs, 1]]
              ];
              If[datamax < Last[cutoffs],    
                 While[!(n = Length[cutoffs];
                       cutoffs[[n-1]] <= datamax < cutoffs[[n]]),
                        cutoffs = Drop[cutoffs, -1]]
              ],
	      (* ==================================================== *)
              (* hcat === Automatic || PositiveIntegerQ[hcat] *)
              If[PositiveIntegerQ[hcat],
                 numberOfBins = hcat,
                 (* hcat === Automatic *)
                 numberOfBins = Sqrt[totalcount]
	      ];
              If[approximate === Automatic || TrueQ[approximate],
		 (* make the intervals approximate and make them neat *)
                 {binmin, binmax, bindelta} =
                        approximateIntervals[datamin, datamax, numberOfBins];
                 numberOfBins = Round[(binmax-binmin)/bindelta],
		 (* make the cutoffs exact, ignore neatness *)
		 numberOfBins = Round[numberOfBins];
                 {binmin, binmax, bindelta} =
                         {datamin, datamax, (datamax-datamin)/numberOfBins}
              ];
              cutoffs = binmin + bindelta Range[0, numberOfBins]
           ]; (* end If monotoneIncreasingVectorQ[hcat] *)
           cutoffs
   ] (* end findCutoffs4 *)



(* Interpret the HistogramCategories option when the
        data is raw data, FOR BOTH COMPONENTS x and y. *)
(* Example:
       findCutoffs3[hcat, {dataminX, datamaxX}, {dataminY, datamaxY},
		 totalcount, approximate]
		returns
	{cutoffsX, cutoffsY}
*)
findCutoffs3[hcat_, {dataminX_, datamaxX_}, {dataminY_, datamaxY_},
	 totalcount_, approximate_] :=
   Module[{numberOfBinsOnAnAxis, binminX, binmaxX, bindeltaX,
		binminY, binmaxY, bindeltaY, numberOfBinsX, numberOfBinsY},
	   If[PositiveIntegerQ[hcat],
	      (* hcat gives the total number of bins desired by the user *)
	      numberOfBinsOnAnAxis = Sqrt[hcat],
	      (* hcat === Automatic *)
	      (* NOTE formerly used
                    numberOfBins = Ceiling[(totalcount)^(1/3)]; *)	
	      numberOfBinsOnAnAxis = (totalcount)^(1/3)
	   ];
       If[numberOfBinsOnAnAxis < 2, numberOfBinsOnAnAxis = 2];
	   If[approximate === Automatic || TrueQ[approximate],
	      (* make the cutoffs approximate and make them neat *)
	      {binminX, binmaxX, bindeltaX} =
                   approximateIntervals[dataminX, datamaxX,
			 numberOfBinsOnAnAxis];
              numberOfBinsX = Round[(binmaxX-binminX)/bindeltaX];
	      {binminY, binmaxY, bindeltaY} =
                   approximateIntervals[dataminY, datamaxY,
			 numberOfBinsOnAnAxis];
              numberOfBinsY = Round[(binmaxY-binminY)/bindeltaY],
	      (* make the cutoffs exact, ignore neatness *)	
	      numberOfBinsX = numberOfBinsY = Round[numberOfBinsOnAnAxis];	
              {binminX, binmaxX, bindeltaX} =
                   {dataminX, datamaxX,
			 (datamaxX-dataminX)/numberOfBinsX};
              {binminY, binmaxY, bindeltaY} =
                   {dataminY, datamaxY,
			 (datamaxY-dataminY)/numberOfBinsY}
           ];
	   (* returning {cutoffsX, cutoffsY} *)
	   {binminX + bindeltaX Range[0, numberOfBinsX],
	    binminY + bindeltaY Range[0, numberOfBinsY]}
   ] (* end findCutoffs3 *)



neatTick[t_] := If[TrueQ[Round[t]==t], Round[t],
                   If[Head[t] === Rational, N[t], t]]


(* interpret the HistogramRange option *)
findRange[range_, list_] :=
  Module[{min, max},
   (
        {min, max} = {Min[list], Max[list]};
        max += 10 $MachineEpsilon; (* this is done so that the maximum data 
                                point is included in an interval that is closed 
                                on the left and open on the right *)
        Switch[range,
                Automatic | {Automatic, Automatic}, {min, max},
                {l_?NumberQ, u_?NumberQ} /; FreeQ[{l, u}, Complex] && l < u,
                                range,
                {l_?NumberQ, Automatic} /; FreeQ[l, Complex] && l < max,
                                {range[[1]], max},
                {Automatic, u_?NumberQ} /; FreeQ[u, Complex] && min < u,
                                {min, range[[2]]},
                _, (Message[Histogram::range, range];
                    {min, max})
        ]
   )
  ]


ticksCheckQ[{x_, y_, z_}] :=
        (x === None || x === Automatic || monotoneIncreasingVectorQ[x] ||
         x === IntervalBoundaries || x === IntervalCenters) &&
        (y === None || y === Automatic || monotoneIncreasingVectorQ[y] ||
         y === IntervalBoundaries || y === IntervalCenters) &&
        (z === None || z === Automatic || monotoneIncreasingVectorQ[z])

monotoneIncreasingVectorQ[x_] :=
   Module[{positions},
        positions = If[VectorQ[x], x, Map[If[ListQ[#], First[#], #]&, x] ];
        VectorQ[positions, NumberQ] && FreeQ[positions, Complex] &&
        Apply[Less, positions]
   ]

(* RangeCounts expects cutoffs {c0, c1, c2, ..., cm, cn}, specifying
        intervals {c0 <= x < c1, c1 <= x < c2, ..., cm <= x < cn}.
   Thus the {first, last, delta} returned by approximateIntervals specifies
        {first <= x < first + delta, ..., last - delta <= x < last}.
*)
(* approximateIntervals[min, max, numOfInt] defines a set of
        approximately numOfInt intervals, covering the range {min, max},
        and having boundaries expressible in terms of simple numbers. *)
approximateIntervals[min_, max_, numOfInt_] :=
    Module[ {nmin = N[min], nmax = N[max], spacing, t,
                  nicebins, first, last, delta},
         (* start with handling the bad case of min and max being so
            close together that having multiple bins doesn't make sense;
            user can override with specific bins if this exceptional case
            is actually desired. *)
        If[ numOfInt===1,
            spacing = If[ # == 0.,
                          1,
                          #
                      ]&[max-min];
            Return[{min - 0.2 spacing, max + 0.2 spacing, 1.5 spacing}]
        ];
        If[ Abs[(max - min)/(spacing = If[ # == 0.,
                                           1,
                                           #
                                       ]&[Max[Abs[{min, max}]]])] < 10^-5,
            spacing = 0.2 spacing;
            Return[{min - 1.5 spacing, min + 1.5 spacing, spacing}]
        ];
            (* ======= The following code is similar to LinearScale. ===== *)
            (* It uses TickSpacing[, Nearest], rather than the default
                     TickSpacing[, GreaterEqual]. *)
        spacing = TickSpacing[nmax-nmin, numOfInt,
                     {1, 2, 2.5, 5, 10}, Nearest];
        t = Range[Ceiling[nmin/spacing - 0.05] spacing, max, spacing];
        If[ Length[t]==1,
            t = Join[t, t+spacing]
        ];        nicebins = Map[{#, If[ Round[#]==#,
                               Round[#],
                               #
                           ]}&, t];
            (* =========================================================== *)
        {first, last} = {First[nicebins][[1]], Last[nicebins][[1]]};
        delta = nicebins[[2, 1]]-first;
            (* If x < first, then x will not be counted in an interval
                    {first <= x < first + delta}.
               If x >= last, then x will not be counted in an interval.
                    {last - delta <= x < last.
               Keep adding intervals until all points min <= x <= max are
                    counted. *)
        While[min < first || max >= last,
        (* Make sure that min and max are included in default categories. *)
             If[ min < first,
                 nicebins = Join[
                      Map[{#, If[ Round[#]==#,
                                  Round[#],
                                  #
                              ]}&, {first-delta}],
                      nicebins]
             ];
             If[ max >= last,
                 nicebins = Join[
                      nicebins,
                      Map[{#, If[ Round[#]==#,
                                  Round[#],
                                  #
                              ]}&, {last+delta}]]
             ];
             {first, last} = {First[nicebins][[1]], Last[nicebins][[1]]}
            ];
        {first, last, delta}
    ]

TickSpacing[dx_, n_, prefs_List, method_:GreaterEqual] :=
    Module[ { dist=N[dx/n], scale, prefsdelta, min, pos } ,
        scale = 10.^Floor[Log[10., dist]] ;
        dist /= scale ;
        If[dist < 1, dist *= 10 ; scale /= 10] ;
        If[dist >= 10, dist /= 10 ; scale *= 10] ;
        scale * Switch[method,
            GreaterEqual,
            (* "nice" tick spacing is greater than or equal to
                requested tick spacing *)
            First[Select[prefs, (dist <= #)&]],
            LessEqual,
            (* "nice" tick spacing is less than or equal to
                                requested tick spacing *)
            First[Select[Reverse[prefs], (dist >= #)&]],
            Nearest,
            (* "nice" tick spacing is the taken from the
                element of "prefs" nearest to "dist" *)
            prefsdelta = Map[Abs[#-dist]&, prefs];
            min = Min[prefsdelta];
            pos = Position[prefsdelta, min][[1, 1]];
            prefs[[pos]]
        ]
    ]

End[]

EndPackage[]
