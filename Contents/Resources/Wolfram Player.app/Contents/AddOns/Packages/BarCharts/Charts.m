Message[General::obspkg, "BarCharts`"]

BeginPackage["BarCharts`"]


Options[BarCharts`BarChart] =
Sort[
    {BarStyle -> Automatic,
    BarSpacing -> Automatic,
    BarGroupSpacing -> Automatic,
    BarLabels -> Automatic,
    BarValues -> False,
    BarEdges -> True,
    BarEdgeStyle -> Opacity[0.5],
    BarOrientation -> Vertical} ~Join~ Developer`GraphicsOptions[]
];

SetOptions[BarCharts`BarChart,
           Axes -> True,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];


Options[GeneralizedBarChart] =
Sort[
    {BarStyle -> Automatic,
    BarValues -> False,
    BarEdges -> True,
    BarEdgeStyle -> Opacity[0.5],
    BarOrientation -> Vertical} ~Join~ Developer`GraphicsOptions[]
];

SetOptions[GeneralizedBarChart,
           Axes -> True,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];


Options[StackedBarChart] =
Sort[
    {BarStyle -> Automatic,
    BarSpacing -> Automatic,
    BarLabels -> Automatic,
    BarEdges -> True,
    BarEdgeStyle -> Opacity[0.5],
    BarOrientation -> Vertical} ~Join~ Developer`GraphicsOptions[]
];

SetOptions[StackedBarChart,
           Axes -> True,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];


Options[PercentileBarChart] =
Sort[
    {BarStyle -> Automatic,
    BarSpacing -> Automatic,
    BarLabels -> Automatic,
    BarEdges -> True,
    BarEdgeStyle -> Opacity[0.5],
    BarOrientation -> Vertical} ~Join~ Developer`GraphicsOptions[]
];

SetOptions[PercentileBarChart,
           Axes -> True,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];


Begin["`Private`"]


numberQ[x_] := NumberQ[N[x]]

(* The following is a useful internal utility function to be
used when you have a list of values that need to be cycled to
some length (as PlotStyle works in assigning styles to lines
in a plot).  The list is the list of values to be cycled, the
integer is the number of elements you want in the final list. *)

CycleValues[{},_] := {}

CycleValues[list_List, n_Integer] :=
    Module[{hold = list},
        While[Length[hold] < n,hold = Join[hold,hold]];
        Take[hold,n] /. None -> {}
    ]

CycleValues[item_,n_] := CycleValues[{item},n]

(* BarCharts -
    BarChart, GeneralizedBarChart, StackedBarChart, PercentileBarChart.
    with the internal RectanglePlot and small utilities *)

(* RectanglePlot *)

Options[RectanglePlot] =
    {RectangleStyle -> Automatic,
    EdgeStyle -> Automatic,
    ObscuredFront -> False} ~Join~ Developer`GraphicsOptions[];
SetOptions[RectanglePlot,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];

RectanglePlot[boxes:{{{_?numberQ,_?numberQ},{_?numberQ,_?numberQ}}..},
        opts___?OptionQ] :=
    Module[{ln = Length[boxes], bstyle, estyle, gopts, sort},
    (* Handle options and defaults *)
        {bstyle, estyle,sort} = {RectangleStyle, EdgeStyle,
            ObscuredFront}/.Flatten[{opts, Options[RectanglePlot]}];
        gopts = FilterRules[{opts, Options[RectanglePlot]}, Options[Graphics]];

        If[bstyle === Automatic,
            bstyle = Map[Hue,.6 Range[0, ln - 1]/(ln - 1)]];
        If[bstyle === {}, bstyle = {{}}];
        If[bstyle === None, bstyle = {{}}];
        If[estyle === Automatic, estyle = {GrayLevel[0]}];
        If[estyle === {}, estyle = {{}}];
		If[estyle === None, estyle = {{}}];
        bstyle = CycleValues[bstyle,ln];
        estyle = CycleValues[estyle,ln];
    (* generate shapes *)
(*        recs = If[bstyle === {},
            Table[{},{ln}],
            Transpose[{bstyle, Apply[Rectangle, boxes,{1}]}]];
        lrecs = If[estyle === {},
            Table[{},{ln}],
            Transpose[{estyle, Map[LineRectangle, boxes]}]];
*)    
	recs = Table[{bstyle[[i]], EdgeForm[estyle[[i]]], Rectangle @@ boxes[[i]]}, 
		{i, Length[boxes]}];

	(* sort 'em 
        recs = Map[Flatten,
            If[TrueQ[sort],
                Sort[Transpose[{recs,lrecs}], coversQ],
                Transpose[{recs, lrecs}]
            ],
            {2}
        ];*)
    (* show 'em *)
        Show[Graphics[recs],gopts]
    ]

RectanglePlot[boxes:{{_?numberQ,_?numberQ}..}, opts___] :=
    RectanglePlot[Map[{#, # + 1}&,boxes],opts]

LineRectangle[pts:{{x1_,y1_}, {x2_,y2_}}] :=
    Line[{{x1,y1},{x1,y2},{x2,y2},{x2,y1},{x1,y1}}]

coversQ[{{___,Rectangle[{x11_,y11_}, {x12_,y12_}]},___},
        {{___,Rectangle[{x21_,y21_}, {x22_,y22_}]},___}] :=
    N[And[x11 <= x21 <= x12,
        x11 <= x22 <= x12,
        y11 <= y21 <= y12,
        y11 <= y22 <= y12]]

coversQ[___] := True


(* Modify a tick list {t1, t2, ...} so that approximately n or fewer ticks
    have labels attached.  This is done so that labels do not overlap in
    a plot. *)
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
        If[Mod[k-(l+1)/2, delta] == 0,
           AppendTo[result, tlist[[k]]],
           AppendTo[result, {tlist[[k]], ""}]
            ];
        k++
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


ticksCheckQ[{x_, y_}] :=
    (x === None || x === Automatic || monotoneIncreasingVectorQ[x] ||
     x === IntervalBoundaries || x === IntervalCenters) &&
    (y === None || y === Automatic || monotoneIncreasingVectorQ[y])

monotoneIncreasingVectorQ[x_] :=
   Module[{positions},
    positions = If[VectorQ[x], x, Map[If[ListQ[#], First[#], #]&, x] ];
    VectorQ[positions, NumberQ] && FreeQ[positions, Complex] &&
        Apply[Less, positions]
   ]


(* the following does the equivalent of BinCounts, albeit with less 
   error-checking, since that is taken care of in the function call above. 
   Also, this variant returns counts for data less than and greater than
   the range. A counting function is separated out and compiled
   for the 'rangecount' equivalent; the whole of the function is
   compiled for bincounts, except for a redirection which strips out
   the 'binrange' header that is used above to identify uniformly-sized
   bins in a range-like syntax. *)
countfunc = Compile[{{dat, _Integer, 1}, {bincount, _Integer}},
      Module[{bins = Table[0, {bincount}], i},
          Do[bins[[dat[[i]]]] += 1, {i, Length[dat]}];
          bins
      ]];

(* note use of Round in nbin computation assumes that range limits
   are on integer bounds within numerical error. If this function is
   called generically, then the assumption may not be quite right, and
   Ceiling would be better. *)
bincountfunc = 
    Compile[{{dat, _Real, 1}, {min, _Real}, {max, _Real}, {incr, _Real}}, 
        Module[{nbin = Round[(max - min)/incr], 
                vals = Floor[(dat - min)/incr] + 2,
                bins, thisval = 0},
            bins = Table[0, {nbin + 2}]; 
            Do[thisval = vals[[i]];
               Which[thisval < 1,        bins[[1]] += 1,
                     thisval > nbin + 1, bins[[nbin + 2]] += 1,
                     True,               bins[[thisval]] += 1
               ],
               {i, Length[dat]}
            ];
            bins
        ]
    ];

bincounts[dat_, binrange[min_, max_, incr_]] :=
    bincountfunc[dat, min, max, incr]

(* approximateIntervals[min, max, numOfInt] defines a set of
    approximately numOfInt intervals, covering the range {min, max},
    and having boundaries expressible in terms of simple numbers. *)
approximateIntervals[min_, max_, numOfInt_] :=
   Module[{nmin = N[min], nmax = N[max], spacing, t,
         nicebins, first, last, delta},
(* start with handling the cases of only one interval desired, or
       min and max being so close together that having multiple bins
       doesn't make sense; user can override with specific bins if this
       exceptional case is actually desired. *)
    If[numOfInt===1,
    	spacing=If[# == 0., 1, #]&[max-min];
    	Return[{min - 0.2 spacing, max + 0.2 spacing, 1.5 spacing}]
    ];
    If[Abs[(max - min)/(spacing = If[# == 0., 1, #]&[Max[Abs[{min, max}]]])] < 10^-5,
        spacing = 0.2 spacing;
        Return[{min - 1.5 spacing, min + 1.5 spacing, spacing}]
    ];
    (* ======= The following code is similar to LinearScale. ===== *)
    (* It uses TickSpacing[, Nearest], rather than the default
         TickSpacing[, GreaterEqual]. *)
    spacing = TickSpacing[nmax-nmin, numOfInt,
         {1, 2, 2.5, 5, 10}, Nearest];
    t = Range[Ceiling[nmin/spacing - 0.05] spacing, max, spacing] ;
    (* need at least two bins *)
    If[Length[t]==1, t=Join[t, t+spacing]];
    nicebins = Map[{#, If[Round[#]==#, Round[#], #]}&, t];
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
     If[min < first,
       nicebins = Join[
        Map[{#, If[Round[#]==#, Round[#], #]}&, {first-delta}],
        nicebins]
     ];
     If[max >= last,
       nicebins = Join[
        nicebins,
        Map[{#, If[Round[#]==#, Round[#], #]}&, {last+delta}]]
     ];
     {first, last} = {First[nicebins][[1]], Last[nicebins][[1]]}
    ];
    {first, last, delta}
   ]

PositiveIntegerQ[n_] := IntegerQ[n] && n > 0


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


(* Bar Chart *)

Clear[BarCharts`BarChart]



BarCharts`BarChart[idata:{_?numberQ..}.., opts:OptionsPattern[]] := 
	BarCharts`BarChart[{idata}, opts]

BarCharts`BarChart[idata_?(VectorQ[#, VectorQ[#, numberQ]&]&), opts:OptionsPattern[]]:=
    Module[{data=idata, ln = Length[idata], ticks, orig,rng,
            lns = Map[Length,idata], bs, bgs, labels, width,gbopts},
        {bs,bgs,labels,orient} = {BarSpacing, BarGroupSpacing,
            BarLabels, BarOrientation}/.
            Flatten[{opts, Options[BarCharts`BarChart]}];
        gbopts = FilterRules[Options[BarCharts`BarChart], Options[GeneralizedBarChart]];
        bs = N[bs]; bgs = N[bgs];
        If[bs === Automatic, bs = 0.03];
        If[bgs === Automatic, bgs = .2];
        Which[labels === Automatic,
                labels = Range[Max[lns]],
            labels === None,
                Null,
            labels === {},
                labels = None,
            True,
                labels = CycleValues[labels,Max[lns]]
        ];
        width = (1 - bgs)/ln;
        data = MapIndexed[
            {#2[[2]] + width (#2[[1]] - 1), #1, width - bs}&,
            idata,{2}];
        If[labels =!= None,
            ticks = {Transpose[{
                        Range[Max[lns]] + (ln - 1)/2 width,
                        labels,
                        Table[0, {Max[lns]}]}],
                    Automatic},
        (* else *)
            ticks = {None, Automatic};
        ];
        orig = {1 - width/2 - bgs,0};
        rng = {{1 - width/2 - bgs,
                    Max[lns] + (ln - 1/2) width + bgs},
                All};
        If[orient === Horizontal,
            ticks = Reverse[ticks]; orig = Reverse[orig];
            rng = Reverse[rng]];
        GeneralizedBarChart[data, opts,
            Ticks -> ticks,
            AxesOrigin -> orig,
            PlotRange -> rng,
            FrameTicks -> ticks,
            gbopts]
    ]

(* For compatability only... *)

BarCharts`BarChart[list:{{_?numberQ, _}..},
        opts___?OptionQ] :=
    Module[{lab,dat},
        {dat, lab} = Transpose[list];
        BarCharts`BarChart[dat, opts, BarLabels -> lab]
    ]

BarCharts`BarChart[list:{{_?numberQ, _, _}..},
        opts___?OptionQ] :=
    Module[{lab, sty, dat},
        {dat, lab, sty} = Transpose[list];
        BarCharts`BarChart[dat, opts, BarLabels -> lab, BarStyle -> sty]
    ]

(* GeneralizedBarChart *)


GeneralizedBarChart::badorient =
"The value given for BarOrientation is invalid; please use \
Horizontal or Vertical. The chart will be generated with \
Vertical.";

GeneralizedBarChart[idata:{{_?numberQ,_?numberQ,_?numberQ}..}.., opts___?OptionQ] :=
	GeneralizedBarChart[{idata}, opts]

GeneralizedBarChart[idata:{{{_?numberQ,_?numberQ,_?numberQ}..}..},
        opts___?OptionQ] :=
    Module[{data = idata, bsty, val, vpos, unob, edge, esty, bsf,
            orient, ln = Length[idata],
            lns = Map[Length,idata], bars, disp, pr, origopts},
    (* Get options *)
        {bsty, val, edge, esty, orient, pr} =
            {BarStyle, BarValues, BarEdges, BarEdgeStyle,
            BarOrientation, PlotRange}/.
                Flatten[{opts, Options[GeneralizedBarChart]}];
        origopts = FilterRules[Flatten[{opts, Options[GeneralizedBarChart]}], {DisplayFunction}];
        gopts = FilterRules[{opts, Options[GeneralizedBarChart]}, Options[Graphics]];
    (* Handle defaults and error check options *)
        If[bsty =!= Automatic && bsty =!= None && Head[bsty] =!= List &&
                !MatchQ[bsty,
                   (Hue | RGBColor | GrayLevel | CMYKColor | Opacity| Directive)[__]
                 ],
            bsty = Join @@ Map[bsty[#[[2]]]&,data,{2}],
            bsty = barcoloring[bsty, ln, lns]
        ];
        If[TrueQ[edge],
            If[ln === 1,
                esty = CycleValues[esty, Length[First[data]]],
                esty = Join @@ MapThread[Table[#1,{#2}]&,
                    {CycleValues[esty,ln], lns}]
            ],
            esty = None
        ];
        If[!MemberQ[{Horizontal, Vertical},orient],
            Message[GeneralizedBarChart::badorient,orient];
                orient = Vertical
        ];
        val = TrueQ[val];
        vpos = .05;   (* was an option, position of value label; now hardcoded at
                        swolf recommendation. *)
    (* generate bars and labels, call RectanglePlot *)
        data = Flatten[data,1];
        bars = Map[barcoords[orient],data];
        If[val,
            Show[RectanglePlot[bars,
                    RectangleStyle -> bsty,
                    EdgeStyle -> esty,
                    DisplayFunction -> Identity],
                Graphics[Map[varcoords[orient,vpos,(#&)],data]],
                If[pr === Automatic,
                    PlotRange -> All,
                    PlotRange -> pr
                ],
                origopts,
                gopts
            ],
        (* else *)
            RectanglePlot[bars,
                RectangleStyle -> bsty,
                    EdgeStyle -> esty,
                    ObscuredFront -> unob,
                    gopts]
        ]
    ]

(* fallthrough for empty data set *)
GeneralizedBarChart[{}, opts___] :=
    Show[Graphics[{},
        FilterRules[{opts, Options[GeneralizedBarChart]}, Options[Graphics]]]
    ]

barcoords[Horizontal][{pos_,len_,wid_}] :=
    {{0,pos - wid/2},{len,pos + wid/2}}

barcoords[Vertical][{pos_,len_,wid_}] :=
    {{pos - wid/2, 0},{pos + wid/2, len}}

varcoords[Horizontal,offset_,format_][{pos_,len_,wid_}] :=
    Text[format[len], Scaled[{(Sign[len]/. (0 ->1)) offset, 0}, {len, pos}]]

varcoords[Vertical,offset_,format_][{pos_,len_,wid_}] :=
    Text[format[len], Scaled[{0,(Sign[len]/.(0 -> 1)) offset}, {pos,len}]]

barcoloring[Automatic, 1, _] := {Hue[0.67, 0.45, 0.65]}

barcoloring[Automatic, ln_, lns_] :=
    Join @@ MapThread[Table[#1,{#2}]&,
        {Table[Hue[FractionalPart[0.67 + 2.0 (i - 1)/GoldenRatio], 0.45, 0.65],
                {i, 1, ln}], lns}]

barcoloring[bsty_, 1, lns_] :=
    CycleValues[bsty, First[lns]]

barcoloring[bsty_, ln_, lns_] :=
    Join @@ MapThread[Table[#1,{#2}]&,
                {CycleValues[bsty, ln], lns}]

(* StackedBarChart *)



StackedBarChart::badorient =
"The value given for BarOrientation is invalid; please use \
Horizontal or Vertical. The chart will be generated with \
Vertical.";

StackedBarChart::badspace =
"The value `1` given for the BarSpacing option is invalid; \
please enter a number or Automatic.";

StackedBarChart[idata:{_?numberQ..}.., opts___?OptionQ] := 
	StackedBarChart[{idata}, opts]

StackedBarChart[idata_?(VectorQ[#, VectorQ[#, numberQ]&]&), opts:OptionsPattern[]]:=
    Module[{data = idata, sty, space, labels, bv, bvp, edge,
            esty, orient, ln = Length[idata], add, tmp,
            lns = Map[Length, idata], ticks, fticks, orig, rng},
    (* process options *)
        {sty, space, labels, edge, esty, orient, orig, rng, ticks, fticks} =
            {BarStyle, BarSpacing, BarLabels, BarEdges, BarEdgeStyle,
             BarOrientation, AxesOrigin, PlotRange, Ticks, FrameTicks}/.
                Flatten[{opts, Options[StackedBarChart]}];
        sty = barcoloring[sty, ln, lns];
        If[TrueQ[edge],
            If[ln === 1,
                esty = CycleValues[esty, First[lns]],
                esty = Join @@ MapThread[Table[#1,{#2}]&,
                    {CycleValues[esty,ln], lns}]
            ],
            esty = None
        ];
        If[!MemberQ[{Horizontal, Vertical},orient],
            Message[StackedBarChart::badorient,orient];
                orient = Vertical
        ];
        Which[labels === Automatic,
                labels = Range[Max[lns]],
            labels === None,
                Null,
            True,
                labels = CycleValues[labels,Max[lns]]
        ];
        If[!(numberQ[space] || (space === Automatic)),
            Message[StackedBarChart::badspace, space];
            space = Automatic];
        If[space === Automatic, space = .2];
        If[ticks === Automatic,
            If[labels =!= None,
                ticks = {Transpose[{
                            Range[Max[lns]],
                            labels,
                            Table[0, {Max[lns]}]}
                         ],
                         Automatic},
              (* else *)
                ticks = {None, Automatic};
            ];
            If[orient === Horizontal, ticks = Reverse[ticks]];
        ];
        If[fticks === Automatic, fticks = ticks];
        If[!MatchQ[N[orig], {_?NumberQ, _?NumberQ}],
            If[orient === Horizontal,
               orig = {0, 1/2},
               orig = {1/2, 0}
            ]
        ];
        If[rng === Automatic,
            rng = {{1/2,Max[lns] + 1/2}, All};
            If[orient === Horizontal, rng = Reverse[rng]]
        ];
            (* data to rectangles *)
        halfwidth = (1 - space)/2; width = (1 - space);
        ends = Table[{0,0},{Max[lns]}];
        data = Map[
            MapIndexed[
                (If[Negative[N[#1]],
                    add = {0, #1};
                    tmp = {First[#2] - halfwidth,
                        Last[ends[[ First[#2] ]] ]},
                    (* else *)
                    add = {#1, 0};
                    tmp = {First[#2] - halfwidth,
                        First[ends[[ First[#2] ]] ]}
                ];
                ends[[ First[#2] ]] += add;
                {tmp, tmp + {width, N[#1]}})&,
            #]&,
            data
        ];
        If[orient === Horizontal, data = Map[Reverse,data,{3}]];
    (* plot 'em! *)
        RectanglePlot[Flatten[data,1],
            RectangleStyle -> sty,
            EdgeStyle -> esty,
            AxesOrigin -> orig,
            PlotRange -> rng,
            Ticks -> ticks,
            FrameTicks -> fticks,
            FilterRules[{opts, Options[StackedBarChart]}, Options[RectanglePlot]]]

    ]

(* PercentileBarChart *)

Options[PercentileBarChart] =
Sort[
    {BarStyle -> Automatic,
    BarSpacing -> Automatic,
    BarLabels -> Automatic,
    BarEdges -> True,
    BarEdgeStyle -> GrayLevel[0],
    BarOrientation -> Vertical} ~Join~ Developer`GraphicsOptions[]
];

SetOptions[PercentileBarChart,
           Axes -> True,
           AspectRatio -> 1/GoldenRatio,
           PlotRangeClipping -> True
];


PercentileBarChart[idata:{_?numberQ..}.., opts___?OptionQ] := 
	PercentileBarChart[{idata}, opts]

PercentileBarChart[idata_?(VectorQ[#, VectorQ[#, numberQ]&]&), opts:OptionsPattern[]]:=
    Module[{data = idata, labels, orient, ln = Length[idata],
            lns = Map[Length,idata],xticks, yticks, ticks},
    (* options and default processing *)
        {labels, orient} = {BarLabels, BarOrientation}/.
            Flatten[{opts, Options[PercentileBarChart]}];
        Which[labels === Automatic,
                labels = Range[Max[lns]],
            labels === None,
                Null,
            True,
                labels = CycleValues[labels,Max[lns]]
        ];
        If[labels =!= None,
            xticks = Transpose[{Range[Max[lns]],labels,Table[0, {Max[lns]}]}],
            xticks = Automatic
        ];
        If[MemberQ[ Flatten[Sign[N[data]]], -1],
            yticks = Transpose[{
                Range[-1,1,.2],
                Map[ToString[#] <> "%"&,Range[-100,100,20]]}],
            yticks = Transpose[{
                Range[0,1,.1],
                Map[ToString[#] <> "%"&, Range[0,100,10]]}]
        ];
        If[orient === Horizontal,
            ticks = {yticks, xticks},
            ticks = {xticks, yticks}
        ];
    (* process data - convert to percentiles *)
        data = Map[pad[#,Max[lns]]&, data];
        maxs = Apply[Plus, Transpose[Abs[data]],{1}];
        data = Map[MapThread[If[#2 == 0, 0, #1/#2]&,{#,maxs}]&,
            data];
    (* plot it! *)
        StackedBarChart[data,
            opts,
            Ticks -> ticks,
            FrameTicks -> ticks,
            Sequence @@ Options[PercentileBarChart]
        ]
    ]

pad[list_, length_] := list/; Length[list] === length

pad[list_,length_] :=
    Join[list, Table[0,{length - Length[list]}]]

 
 End[]
 
 EndPackage[]
