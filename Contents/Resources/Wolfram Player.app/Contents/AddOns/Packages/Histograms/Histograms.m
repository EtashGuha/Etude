Message[General::obspkg, "Histograms`"]

Quiet[BeginPackage["Histograms`", "BarCharts`"], {General::obspkg}]

Options[Histograms`Histogram] =
    {
    ApproximateIntervals -> Automatic,
    BarEdges -> True,            (* opt of GeneralizedBarChart *)
    BarEdgeStyle -> GrayLevel[0],         (* opt of GeneralizedBarChart *)
    BarOrientation -> Vertical,         (* opt of GeneralizedBarChart *)
    BarStyle -> Automatic,             (* opt of GeneralizedBarChart *)
    FrequencyData -> False,
    HistogramCategories -> Automatic,
    HistogramRange -> Automatic,
    HistogramScale -> Automatic
    } ~Join~ Developer`GraphicsOptions[];

Begin["`Private`"]

(* Histogram *)

(* Histogram does not have the BarChart options BarSpacing, BarGroupSpacing,
   and BarValues.  The option HistogramCategories functions like the option
   PlotPoints (except that it also allows category boundaries to be specified);
  HistogramRange functions like PlotRange.
*)



SetOptions[Histograms`Histogram,
           Ticks -> Automatic,
           Axes -> True,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];


(* Note: Histogram calls an internal RangeCounts variant to compute
    frequencies and GeneralizedBarChart for plotting. *)
Histograms`Histogram[list_?VectorQ, opts___?OptionQ] :=
  (* use of numericalization here is somewhat questionable; I don't
     *think* it will break any practical use, but keep an eye on it.
     It's necessary for efficient computation later on, though. *)
    With[{res = histogram[N[list], opts]},
        res/; res =!= $Failed
    ]

histogram[list_, opts___] :=
   Module[{approximate, bedges, bedgestyle, borien, bstyle, fdata, hcat,
           range, scale, ticks, padding,
           countdata, numberOfBins,
           dmin, dmax,
           datamin, datamax, (* min and max as determined by the data and the
                                option HistogramRange *)
           cutoffs, fixedbins = False,
           binmin, binmax, (* min and max as determined by bin boundaries *)
           totalcount,
           leftTailCount, rightTailCount,
           binwidths, bincenters,
           autoticks, autolength, (* automatic setting for ticks *)
           caxisticks, (* category axis ticks ... can be x or y axis
                          depending on BarOrientation *)
           phwdata, (* position-height-width data for GeneralizedBarChart *)
           orig, rng, (* settings for AxesOrigin, PlotRange *)
           gropts, groptslist,
           area (* area of histogram; used for scaling non-category axis so
                   that histogram has unit area *)
           },
      (* Histogram only handles real numeric data, so issue a message and 
      	 return unevaluated if non-real values are present *)
      If[Not[TrueQ[Element[list, Reals]]], Message[Histogram::realvec];Return[$Failed]];
      {approximate, bedges, bedgestyle, borien, bstyle,
           fdata, hcat, range, scale, ticks, padding} =
         {ApproximateIntervals, BarEdges, BarEdgeStyle,
             BarOrientation, BarStyle, FrequencyData,
             HistogramCategories, HistogramRange, HistogramScale, Ticks, PlotRangePadding} /.
          Flatten[{opts,Options[Histograms`Histogram]}];
     (* sanity check: if this is frequency data, and HistogramCategories
        gives explicit bins, then the number of bins must match the number
        of data quantities. *)
       If[TrueQ[fdata] && VectorQ[hcat] && (Length[list] + 1 != Length[hcat]),
           Message[Histogram::fdfail];  Return[$Failed]
       ];
     (* check value of 'range' *)
       If[range =!= Automatic &&
              !MatchQ[range, {_?NumberQ | Automatic, _?NumberQ | Automatic}],
           range = Automatic
       ];
     (* Define countdata, numberOfBins, binmin, binmax, cutoffs. *)
       If[TrueQ[fdata],
         (* ===================================================== *)    
         (* PROCESS LIST assuming that it represents FREQUENCIES. *)
         (* ===================================================== *)    
           countdata = list;
           numberOfBins = Length[countdata];

         (* Error check for HistogramCategories setting. *)
           If[!(hcat === Automatic || BarCharts`Private`monotoneIncreasingVectorQ[hcat]),
               Message[Histogram::fdhc, hcat];
               hcat = Automatic];

          {datamin, datamax} = findRange[range,
               If[hcat === Automatic, {0, numberOfBins}, {Min[hcat], Max[hcat]}]
          ];

          If[hcat === Automatic,
              cutoffs = datamin + (datamax-datamin)/numberOfBins *
                  Range[0, numberOfBins],
              cutoffs = findCutoffs1[hcat, datamin, datamax, countdata];
              numberOfBins = Length[cutoffs]-1
          ];
          {binmin, binmax} = {First[cutoffs], Last[cutoffs]},
        (* ===================================================== *)    
        (* PROCESS LIST assuming that it represents RAW DATA.    *)
        (* ===================================================== *)    
        (* Define min and max of range, and count data in range. *)
          {dmin, dmax} = {Min[list], Max[list]};
          {datamin, datamax} = findRange[range, {dmin, dmax}];
          If[datamin <= dmin && datamax >= dmax,
              totalcount = Length[list],
              totalcount = With[{d1 = datamin, d2 = datamax},
                              Compile[{{l, _Real, 1}},
                                 Module[{count = 0, n},
                                     Do[If[d1 <= l[[n]] <= d2, count++],
                                        {n, Length[l]}];
                                     count
                                 ]
                              ][list]
                            ]
          ];
        (* Define category cutoffs for raw data. *)
          cutoffs = findCutoffs2[hcat, datamin, datamax, totalcount, approximate];
        (* Note: the following is a bit of a hack, used in preference to
           doing a major rewrite of the code. It is useful for some later
           efficiency hacks to know whether we have evenly-sized bins or not;
           this could be determined by point changes in findCutoffs2. *)
          If[Head[cutoffs] === BarCharts`Private`binrange,
              fixedbins = cutoffs;
              cutoffs = cutoffs[[1]] + cutoffs[[3]] *
                  Range[0, Round[(cutoffs[[2]] - cutoffs[[1]])/cutoffs[[3]]]]
          ];
          numberOfBins = Length[cutoffs]-1;
        (* Note that RangeCounts considers intervals of the form
           {binmin <= x < etc, ..., etc <= x < binmax}. *)
          {binmin, binmax} = {First[cutoffs], Last[cutoffs]};
        (* Compute category counts for raw data. *)
          countdata = 
              If[Head[fixedbins] === BarCharts`Private`binrange,
                  BarCharts`Private`bincounts[list, fixedbins],
                  BinCounts[list, {Join[{-Infinity},cutoffs,{Infinity}]}]
              ];
          If[!ListQ[countdata], Message[Histogram::rcount];Return[$Failed] ];
        (* Warning messages for points not plotted, if histogram range
           was determined automatically. *)
          If[(range === Automatic || First[range] === Automatic) &&
                  First[countdata] > 0,
              If[First[countdata] === 1,
                  Message[Histogram::ltail1, binmin],
                  Message[Histogram::ltail, First[countdata], binmin]
              ]
          ];
          If[(range === Automatic || Last[range] === Automatic) &&
                  Last[countdata] > 0,
              If[Last[countdata] === 1,
                  Message[Histogram::rtail1, binmax],
                  Message[Histogram::rtail, Last[countdata], binmax]
              ]
          ];
        If[!ListQ[padding], padding = {padding, padding}];  
		padding = padding /. Automatic -> Scaled[0.02];
        (* Length of data should be numberOfBins+2.
           Eliminate first and last elements
           of data corresponding to the ranges x < binmin and x >= binmax. *)
          countdata = Take[countdata, {2, -2}]
      ]; (* end If TrueQ[fdata] *)

    (* ============================================================= *)
    (* ============================================================= *)
    (* Use countdata, cutoffs, numberOfBins, binmin, and binmax to     *)
    (*        generate histogram.                 *)
    (* ============================================================= *)
    (* ============================================================= *)

    (* ================= Scale category counts. ================ *)
    (* Here we choose to normalize so that the height of the tallest *)
    (* bar is unchanged.  To normalize to get unit area, you need to *)
    (* set HistogramScale -> 1. *)
    binwidths = Drop[cutoffs, 1] - Drop[cutoffs, -1];
    If[TrueQ[scale] || ((scale === Automatic) &&
            !(hcat === Automatic || IntegerQ[hcat] ||
              (0.0001 > Abs[Max[binwidths]/Min[binwidths] - 1]))),
       (* Make the area of the bar proportional to the frequency
        associated with the bar. *)
       countdata = countdata/binwidths
    ];
    bincenters = Drop[FoldList[Plus, binmin, binwidths], -1] +
        1/2 binwidths;

    (* =============================================================== *)
    (*  Define category axis ticks from                    *)
    (*     bincenters, countdata, and ticks.               *)
    (* =============================================================== *)
    autoticks = LinearScale[binmin, binmax, 7];
    autolength = Length[autoticks];
    (* Process the Ticks setting. *)
      If[MatchQ[ticks, Automatic | IntervalCenters | IntervalBoundaries],
        ticks = {ticks, Automatic}];
    If[ticks === None, ticks = {None, None}];
    (* Check the Ticks setting, and reset to Automatic if the setting is
        illegal. *)
    If[!(ListQ[ticks] && Length[ticks] == 2 && ticksCheckQ[ticks]),
       Message[Histogram::ticks, ticks];
       ticks = {Automatic, Automatic}];
    caxisticks = Switch[ticks[[1]],
        _?ListQ, (* ticksCheckQ has already checked for
                BarCharts`Private`monotoneIncreasingVectorQ *)
            Map[neatTick, ticks[[1]] ],
        IntervalBoundaries,
           (
           trim[ Map[neatTick, cutoffs], autolength]
           ),
        IntervalCenters,
           (
           trim[ Map[neatTick, bincenters], autolength]
           ),    
        None, (* no category axis ticks *)
            None,
        _, (* place category axis ticks automatically *)
            autoticks
    ];
    ticks = {caxisticks, ticks[[2]]};

    (* =============================================================== *)
    (* ======= Define phwdata (position, height, width). ============= *)
    (* =============================================================== *)
    (* Note that BarGroupSpacing is assumed to be 0 here.  If you want *)
    (* to add that option to Histogram, the option should be    *)
    (* processed here.  (Some would say that histograms of discrete data *)
    (* ought to have columns separated from each other, i.e., with  *)
    (* BarGroupSpacing greater than zero.) *)
    phwdata = Transpose[{bincenters, countdata, binwidths}];

    (* =========== Define settings for AxesOrigin & PlotRange. ======== *)
    (* First category is from 
        bincenters[[1]]-1/2 binwidths[[1]] (= First[cutoffs])
           to
                bincenters[[1]]+1/2 binwidths[[1]]...
           Adjust origin so that first category lines up with vertical axis. *)
        orig = {First[cutoffs], 0};
        rng = {{First[cutoffs], Last[cutoffs]}, All};
        If[borien === Horizontal,
       ticks = Reverse[ticks]; orig = Reverse[orig];
       rng = Reverse[rng]];

        (* =========== Extract any other options relevent to Graphics. ==== *)
        gropts = FilterRules[{opts, Options[Histograms`Histogram]}, Options[Graphics]];
    groptslist = DeleteCases[{gropts}, _[Ticks,_]];

    (* ======= Scale bar heights according to HistogramScale -> k ====== *)
    (* NOTE that phwdata has the form...
        {  {pos1, height1, width1}, {pos2, height2, width2}, ...} *)
    If[NumberQ[scale] && FreeQ[scale, Complex] && scale > 0,
           area = Total[phwdata[[All,2]]];
           phwdata = Map[{#[[1]], #[[2]]/area * scale/#[[3]], #[[3]]}&, phwdata]
        ];

	If[bstyle === Automatic, 
		bstyle = RGBColor[0.7771114671549554`, 0.7981689173723965`, 0.92304875257496`]];

    (* ================== GeneralizedBarChart call ===================== *)
        BarCharts`GeneralizedBarChart[phwdata, 
            AxesOrigin -> orig,         (* option of Graphics *)
            BarEdges -> bedges,         (* option of GeneralizedBarChart *)
               BarEdgeStyle -> bedgestyle, (* option of GeneralizedBarChart *)
            BarOrientation -> borien,   (* option of GeneralizedBarChart *)
            BarStyle -> bstyle,         (* option of GeneralizedBarChart *)
            PlotRange -> rng,           (* option of Graphics *)
            Ticks -> ticks,             (* option of Graphics *)
            PlotRangePadding -> padding, 
          (* groptslist includes any other options relevent to Graphics *)
            Apply[Sequence, groptslist]
    ]
 ] (* end Histogram *)


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
    data is raw data. *)
findCutoffs2[hcat_, datamin_, datamax_, totalcount_, approximate_] :=
   Module[{numberOfBins, cutoffs, binmin, binmax, bindelta},
       If[BarCharts`Private`monotoneIncreasingVectorQ[hcat],
          (* Intervals are NOT approximated when they are
            specifically requested using HistogramCategories. *)
          If[!(approximate===Automatic || approximate===False),
             Message[Histogram::noapprox, approximate]];
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
          (* ====================================================== *)
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
            (*  cutoffs = binmin + bindelta Range[0, numberOfBins] *)
            cutoffs = BarCharts`Private`binrange[binmin, binmax, bindelta];
       ]; (* end If BarCharts`Private`monotoneIncreasingVectorQ[hcat] *)
       cutoffs
   ] (* end findCutoffs2 *)


neatTick[t_] := If[TrueQ[Round[t]==t], Round[t],
                   If[Head[t] === Rational, N[t], t]]


(* interpret the HistogramRange option *)
findRange[range_, {imin_, imax_}] :=
  Module[{min = imin, max = imax},
   (
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
 
 LinearScale[min_, max_, n_Integer:8] :=
    Module[{spacing, t, nmin=N[min], nmax=N[max]},
        (spacing = TickSpacing[nmax-nmin, n, {1, 2, 2.5, 5, 10}] ;
        t = N[spacing * Range[Ceiling[nmin/spacing - 0.05],
                              Floor[max/spacing + 0.05]]] ;
        Map[{#, If[Round[#]==#, Round[#], #]}&, t])
    /; nmin <= nmax
    ]
 
PositiveIntegerQ[a_Integer?Positive] := True
PositiveIntegerQ[_]:=False

trim[tlist_, n_] :=
  Module[{l = Length[tlist], delta, k, result = {}},
    delta = Round[l/n];
    If[l <= n || delta == 1,
    tlist,
    If[EvenQ[l],
       (* simply pick ticks starting from leftmost tick *)
       k = 1;
       While[k <= l,
        If[Mod[k-1, delta] == 0,
           AppendTo[result, tlist[[k]]],
           AppendTo[result, {tlist[[k]], ""}]
        ];
        k++
       ],
       (* pick ticks such that the center tick of tlist is included *)
       k = (l+1)/2;
       While[k <= l,
        If[Mod[k-(l+1)/2+delta, delta] == 0,
           AppendTo[result, tlist[[k]]],
           AppendTo[result, {tlist[[k]], ""}]
            ];
        k+=delta
       ];
      k = (l+1)/2 - delta;
       While[k >= 1,
        If[Mod[k-((l+1)/2-delta), delta] == 0,
           PrependTo[result, tlist[[k]]],
           PrependTo[result, {tlist[[k]], ""}]
        ];
        k-=delta
       ]
    ];
    result
    ]
  ] (* end trim *)
  
 End[]
  
 EndPackage[]
