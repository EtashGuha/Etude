(* :Title: Statistical Plots *)

(* :Context: StatisticalPlots` *)

(* :Author: John M. Novak and Darren Glosemeyer *)

(* :Summary: plotting functions for statistics *)

(* :Copyright: Copyright 2002-2010, Wolfram Research, Inc. *)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.0 *)

(* :History:
    V1.0 Initial development, September 2002 by John M. Novak.
    V1.1 StemLeafPlot added, May 2005 by Darren Glosemeyer.
    V1.2 Added PlotDirection option to PairwiseScatterPlot and 
    	 changed the default orientation of PairwiseScatterPlot 
    	 to plot data column i in the ith row and column of the 
    	 plot matrix, matching the most common usage of pairwise 
    	 scatter plots, 2005 by Darren Glosemeyer.
   	V2.0 Moved from Statistics`StatisticsPlots` standard add-on to
   	     StatisticalPlots` package, 2006 by Darren Glosemeyer
   	V3.0 Removed QuantilePlot which moved to the kernel in Mathematica 
   		 8.0, 2010 by Darren Glosemeyer
*)

(* :Keywords:
Statistics, Box Plot, Box-Whisker Plot, Pareto Plot, Pairs Scatter Plot, Matrix Scatter Plot, Stem-and-Leaf Plot
*)

(* :Sources:
Wickham-Jones, Tom, Mathematica Graphics: Techniques and Applications,
    Springer-Verlag, 1994.
For Stem-and-Leaf plot:
	J. W. Tukey, Exploratory Data Analysis, 1977, 
		Addison-Wesley Publishing Co., Inc., Reading, MA.
*)

(* :Discussion:
This package implements a variety of plots used primarily in the
context of statistics and exploratory data analysis. It borrows
some ideas from Emily Martin's StatisticsPlots.m package, but no
code.
*)

BeginPackage["StatisticalPlots`"]

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"StatisticalPlots`"],
StringMatchQ[#,StartOfString~~"StatisticalPlots`*"]&]//ToExpression;
];

If[ Not@ValueQ[BoxWhiskerPlot::usage],
BoxWhiskerPlot::usage =
"BoxWhiskerPlot[data] creates a box-and-whisker plot of the given vector of \
data. BoxWhiskerPlot[data1, data2, ...] creates a multiple box-and-whisker \
plot. BoxWhiskerPlot[matrix] also creates a multiple box-and-whisker \
plot, with each column of the matrix used for a separate box."]

If[ Not@ValueQ[BoxQuantile::usage],
BoxQuantile::usage =
"BoxQuantile is an option to BoxWhiskerPlot denoting how far the \
box extends from the median. It is given as a value between 0 and 0.5, \
indicating a +/- value relative to the median (which is a quantile of 0.5). \
The default is 0.25."]

If[ Not@ValueQ[BoxOutliers::usage],
BoxOutliers::usage =
"BoxOutliers is an option to BoxWhiskerPlot indicating how to handle \
outliers. If None, no handling is done, and the whiskers extend over \
the entire data set. If Automatic, the outliers are partitioned into \
near and far sets. If All, all outliers are drawn the same."]

If[ Not@ValueQ[BoxOrientation::usage],
BoxOrientation::usage =
"BoxOrientation is an option to BoxWhiskerPlot specifying the orientation \
of the boxes. If Vertical, the boxes are drawn with the whiskers extending \
up and down; if Horizontal, the whiskers are drawn left-to-right."]

If[ Not@ValueQ[BoxLabels::usage],
BoxLabels::usage =
"BoxLabels is an option to BoxWhiskerPlot specifying labels to be \
given for each of the data sets."]

If[ Not@ValueQ[BoxFillingStyle::usage],
BoxFillingStyle::usage =
"BoxFillingStyle is an option to BoxWhiskerPlot specifying a color to be \
used in drawing the box. It can be None, indicating a transparent box. \
It can be a list, in which case the values are applied cyclically to \
each data set being displayed."]

If[ Not@ValueQ[BoxLineStyle::usage],
BoxLineStyle::usage =
"BoxLineStyle is an option to BoxWhiskerPlot giving styles to be \
applied to the lines drawn in the plot."]

If[ Not@ValueQ[BoxMedianStyle::usage],
BoxMedianStyle::usage =
"BoxMedianStyle is an option to BoxWhiskerPlot specifying styles to \
be applied specifically to the median lines in the plot."]

If[ Not@ValueQ[BoxOutlierMarkers::usage],
BoxOutlierMarkers::usage =
"BoxOutlierMarkers is an option to BoxWhiskerPlot specifying markers \
to be used when drawing outliers. Specifications are the same as for PlotMarkers."]

If[ Not@ValueQ[BoxExtraSpacing::usage],
BoxExtraSpacing::usage =
"BoxExtraSpacing is an option to BoxWhiskerPlot giving spacing adjustments \
to be applied when plotting multiple boxes. This allows you to create \
groups of boxes by placing extra space where needed."]

If[ Not@ValueQ[ParetoPlot::usage],
ParetoPlot::usage =
"ParetoPlot[list] creates a Pareto plot from the given list. \
ParetoPlot[{{cat1, freq1}, {cat2, freq2}, ...}] creates a Pareto plot from \
the categories cat1, cat2, \[Ellipsis] and associated frequencies freq1, \
freq2, \[Ellipsis]."]

If[ Not@ValueQ[PairwiseScatterPlot::usage],
PairwiseScatterPlot::usage =
"PairwiseScatterPlot[matrix] creates a matrix of scatter plots of each \
column of the matrix against every other column."]

If[ Not@ValueQ[DataSpacing::usage],
DataSpacing::usage =
"DataSpacing is an option for PairwiseScatterPlot, specifying \
spacing between the data graphs. It takes the form of {horiz, vert} \
or just a number (in which case the horizontal and vertical space are \
the same) indicating the amount of extra space to place between \
grid elements, where 1 is the size of an individual grid element."]

If[ Not@ValueQ[DataRanges::usage],
DataRanges::usage =
"DataRanges is an option for PairwiseScatterPlot, specifying range \
limits for the data to be displayed. It takes the form of a list of \
ranges of the form {min, max} or All, one for each column of the data."]

If[ Not@ValueQ[DataLabels::usage],
DataLabels::usage =
"DataLabels is an option for PairwiseScatterPlot, specifying labels to \
place on the graph for each column of data. If None, no labels are \
used; if Automatic, the columns are numbered sequentially."]

If[ Not@ValueQ[DataTicks::usage],
DataTicks::usage =
"DataTicks is an option for PairwiseScatterPlot, specifying ticks \
to place on the graph for each column of data."]

If[ Not@ValueQ[PlotDirection::usage],
PlotDirection::usage =
"PlotDirection is an option for PairwiseScatterPlot, specifying the direction \
in which scatter plots are generated. Possible values are {xdir, ydir}, where \
xdir is either Left or Right and ydir is either Up or Down."]

If[ Not@ValueQ[StemLeafPlot::usage],
StemLeafPlot::usage = "StemLeafPlot[data] creates a stem-and-leaf plot \
for real-valued vector data. StemLeafPlot[data1, data2] creates a  \
side-by-side stem-and-leaf plot for real-valued vectors data1 and data2."]

If[ Not@ValueQ[StemExponent::usage],
StemExponent::usage = "StemExponent is an option to StemLeafPlot that \
specifies the units as a power of 10 for stems in the plot. StemExponent must \
be an integer or Automatic.  Units can be subdivided by using the \
\"UnitDivisions\" suboption to StemExponent. Unit subdivisions can be labelled \
using the \"DivisionLabels\" suboption."]

If[ Not@ValueQ[IncludeEmptyStems::usage],
IncludeEmptyStems::usage = "IncludeEmptyStems is an option to StemLeafPlot \
that specifies whether stems with no leaves should be included in the plot. \
Possible values are True and False, with False being the default setting."]

If[ Not@ValueQ[ColumnLabels::usage],
ColumnLabels::usage = "ColumnLabels is an option to StemLeafPlot that \
specifies the labels for columns in the StemLeafPlot.  ColumnLabels must be \
Automatic or a list of the same length as the number of columns in the plot."]

If[ Not@ValueQ[Leaves::usage],
Leaves::usage = "Leaves is an option to StemLeafPlot that specifies how \
leaves should be displayed.  Possible values are Digits, TallySymbol, and None. \
Leaves also takes additional options which depend on the Leaves option values."]

If[ Not@ValueQ[IncludeStemUnits::usage],
IncludeStemUnits::usage = "IncludeStemUnits is an option to StemLeafPlot that \
specifies whether the units of the stems should be included with the plot. \
The value must be True or False."]

If[ Not@ValueQ[IncludeStemCounts::usage],
IncludeStemCounts::usage = "IncludeStemCounts is an option to StemLeafPlot \
that specifies whether a counts for each stem should be included along with \
the leaves."]

Begin["`Private`"]

(* V4.x hack *)
If[$VersionNumber < 5, MachinePrecision = $MachinePrecision];

(* default graphics options; this is given as an explicit list rather
   than in-code references to Options[Graphics] because if a user does
   SetOptions[Graphics,...] then loads this package, we don't actually
   want the package to pick up the changes to Options[Graphics]; the
   package should behave the same whether loaded before or after the
   user plays with Options[Graphics]. Note that some defaults are modified
   to plot-type default rather than raw graphics (e.g., AspectRatio) *)
$defaultgraphicsoptions =Developer`GraphicsOptions[];

(* cyclevalues utility *)
cyclevalues[l_List, n_] := PadRight[l, n, l]
cyclevalues[val_, n_] := cyclevalues[{val}, n]

(* test NumericQ vector or matrix without unpacking *)
nvectorQ[vec_] := (Developer`PackedArrayQ[vec] && VectorQ[vec]) ||
                      VectorQ[vec, NumericQ]

nmatrixQ[mat_] := (Developer`PackedArrayQ[mat] && MatrixQ[mat]) ||
                      MatrixQ[mat, NumericQ]
                      
removeOldOpts[optlist_, oldvals_, caller_] := 
 Block[{oldoptpos, oldopts, optstrings},
  optstrings = Map[ToString, optlist[[All, 1]]];
  oldoptpos = Position[optstrings, Apply[Alternatives, oldvals]];
  optstrings = Extract[optstrings, oldoptpos];
  Map[oldoptmessage[caller, #] &, optstrings];
  If[# === {}, #, optlist[[#]]] &[
   Complement[Range[Length[optlist]], Flatten[oldoptpos]]]]


oldoptmessage[BoxWhiskerPlot, "BoxOutlierShapes"]:=Message[BoxWhiskerPlot::shapes]

oldoptmessage[ParetoPlot, "SymbolShape"]:=Message[ParetoPlot::shape]

oldoptmessage[ParetoPlot, "SymbolStyle"]:=Message[ParetoPlot::style]

BoxWhiskerPlot::shapes="The option BoxOutlierShapes is obsolete and will be ignored. \
Use BoxOutlierMarkers instead."

ParetoPlot::shape="The option SymbolShape is obsolete and will be ignored. \
Use PlotMarkers instead."

ParetoPlot::style="The option SymbolStyle is obsolete and will be ignored. \
Use PlotMarkers instead."


(************************ BoxWhiskerPlot **************************)

(* For efficiency, this effectively reimplements the stats code for
   median, quantile, etc., to specifically suit the purposes of a
   box-whisker plot. This could be implemented with the separate stats
   functions, but that would mean sorting the data multiple times, bad
   for large data sets. It may be worth considering folding this code
   into a Developer`-type function for getting multiple stats at once;
   alternately, a design based on caching the data might be considered
   in the standard functions, though that increases memory usage... *)
boxwhiskerstatistics[{}, ___] := {}
boxwhiskerstatistics[data_, quantile_, outliers_] :=
    Module[{sorted, len = Length[data], median, qlow, qhigh, outs},
        sorted = sort[data];
        If[OddQ[len],
             median = sorted[[(len + 1)/2]],
             median = (sorted[[len/2]] + sorted[[len/2 + 1]])/2
        ];
        qlow = sorted[[(-Floor[-(0.5 - quantile) len])/.(0 -> 1)]];
        qhigh = sorted[[(-Floor[-(0.5 + quantile) len])/.(0 -> 1)]];
        Which[
          (***** note: these values for outliers option are not final *)
             outliers === None || quantile == 0.5,
                 outs = {{},{},{},{}, First[sorted], Last[sorted]},
             outliers === Automatic,
                 outs = getoutliers[sorted, qlow, qhigh],
             outliers === All,
                 outs = {{}, Join[#1, #2], Join[#3, #4], {}, #5, #6}& @@
                                      getoutliers[sorted, qlow, qhigh]
        ];
        {median, qlow, qhigh, outs}
    ]

(* outlier extraction is done under the assumption that there are few
   outliers, so a linear search is OK. Returns:
   {{vlow outliers}, {low outliers}, {high outlier}, {vhigh outliers},
     lowest non-outlier, highest non-outlier} *)
getoutliers[data_, qlow_, qhigh_] :=
    Module[{qlen = qhigh - qlow, vlow, low, vhigh, high, count},
         vlow = qlow - 3 qlen; low = qlow - 3/2 qlen;
         vhigh = qhigh + 3 qlen; high = qhigh + 3/2 qlen;
         count = 1;
         While[data[[count]] < vlow, count++];
         vlow = count - 1;
         While[data[[count]] < low, count++];
         low = count - 1;
         count = -1;
         While[data[[count]] > vhigh, count--];
         vhigh = count + 1;
         While[data[[count]] > high, count--];
         high = count + 1;
         {Take[data, vlow],
          If[low > vlow, Take[data, {vlow + 1, low}], {}],
          If[high < vhigh, Take[data, {high, vhigh - 1}], {}],
          Take[data, vhigh],
          data[[low + 1]],
          data[[count]]}
    ]

(* also borrowed from DescriptiveStatistics.m; I'm not sure this is
   really necessary for a graphics package, it might be better to just
   coerce the data to machine-precision for non-NumberQ... *)
sort[s_?(VectorQ[#, NumberQ]&)] := Sort[s]

sort[s_] := Sort[s, OrderedQ[N[{#1, #2},
          Precision[s] /. Infinity -> MachinePrecision]] &]

(* create the graphics primitives *)
boxwhiskergraphic[{}, ___] := {}

boxwhiskergraphic[{median_, qlow_, qhigh_,
                   {vlow_, low_, high_, vhigh_, wlow_, whigh_}},
                  pos_, orient_,
                  boxstyle_, linestyle_, medlinestyle_,
                  {nearshape_, farshape_}] :=
    Module[{boxline, wlines, bw = 0.25, ww = 0.12,medline,outlierposns},
        boxline = reorient[{{pos - bw, qlow}, {pos + bw, qlow},
                            {pos + bw, qhigh}, {pos - bw, qhigh},
                            {pos - bw, qlow}}, orient];
        wlines = Map[reorient[#, orient]&,
                       {{{pos - ww, wlow}, {pos + ww, wlow}},
                        {{pos, wlow}, {pos, qlow}},
                        {{pos, qhigh}, {pos, whigh}},
                        {{pos - ww, whigh}, {pos + ww, whigh}}}
                 ];
        medline = reorient[{{pos - bw, median}, {pos + bw, median}}, orient];
        outlierposns = Map[reorient[{pos, #}& /@ #, orient]&,
                           {vlow, low, high, vhigh}];
      (* assemble components *)
        {If[boxstyle =!= None,
              {boxstyle, Polygon[boxline]},
              {}
         ],
         Flatten[{linestyle, Line[boxline], Line /@ wlines,
          If[nearshape =!= None,
          		If[#=!={},ListPlot[#,PlotMarkers->nearshape][[1]],{}]&[Apply[Join,outlierposns[[{2,3}]]]], 
          		{}],
          If[farshape =!= None,
          		If[#=!={},ListPlot[#,PlotMarkers->farshape][[1]],{}]&[Apply[Join,outlierposns[[{1,4}]]]], 
          		{}],
          {If[medlinestyle === Automatic || medlinestyle === None,
                {}, medlinestyle],
           Line[medline]}}]}
    ]

reorient[coords_, Vertical] := coords
reorient[coords_, Horizontal] := {#2, -#1}& @@@ coords

(* options processing *)
boxwhiskeroptions[dlen_, opts___] :=
    Module[{bq, bouts, orient, labels, boxsty, linesty, medsty,
            outliershape, space, badopts,newopts},
        newopts=removeOldOpts[{opts}, {"BoxOutlierShapes"}, BoxWhiskerPlot];
        {bq, bouts, orient, labels, boxsty, linesty, medsty,
            outliershape, space} =
            {BoxQuantile, BoxOutliers, BoxOrientation, BoxLabels,
             BoxFillingStyle, BoxLineStyle, BoxMedianStyle,
             BoxOutlierMarkers, BoxExtraSpacing}/.
            Flatten[{newopts, Options[BoxWhiskerPlot]}];
      (* check quantile size option *)
        If[!TrueQ[0 <= bq <= 0.5],
            Message[BoxWhiskerPlot::quant, bq]; bq = 0.25
        ];
      (* check orientation option *)
        If[orient =!= Vertical && orient =!= Horizontal,
            Message[BoxWhiskerPlot::orient, orient]; orient = Vertical
        ];
      (* check outlier handling option -- allow True/False in a
         'do the right thing' spirit. *)
        If[!MemberQ[{All, Automatic, None, True, False}, bouts],
            Message[BoxWhiskerPlot::bouts, bouts]; bouts = None
        ];
        Which[bouts === True, bouts = Automatic,
              bouts === False, bouts = None];
      (* outlier shape should end up as two functions (or None) *)
        outliershape = cyclevalues[outliershape, 2];
      (* check style options *)
      	If[boxsty===Automatic, boxsty=Hue[0.67, 0.4, 0.8]];
        boxsty = cyclevalues[boxsty, dlen];
        If[!MatchQ[boxsty, {(None | RGBColor[___] | Hue[___] |
                             GrayLevel[_] | CMYKColor[___])..}],
            Message[BoxWhiskerPlot::boxsty, boxsty];
            boxsty = cyclevalues[Hue[0.67, 0.4, 0.8], dlen]
        ];
      (* these options will treat a list of styles as a single style if
         there is only one data set to be displayed *)
        If[dlen =!= 1, linesty = cyclevalues[linesty, dlen], linesty = {linesty}];
        linesty = linesty/.Automatic -> GrayLevel[0];
        If[dlen =!= 1, medsty = cyclevalues[medsty, dlen], medsty = {medsty}];
      (* handle extra space *)
        If[dlen === 1, 
          (* we can ignore the extra space option *)
            space = {0},
          (* else handle it *)
            If[!NumberQ[space] && !MatchQ[space, {_?NumberQ..}],
                Message[BoxWhiskerPlot::extraspace, space];
                space = 0
            ];
            space = cyclevalues[space, dlen - 1];
            (* extra spacing is in terms of box width and boxes have width 1/2, 
               so divide the values by 2 *)
            space = FoldList[Plus, 0, space/2];
        ];
      (* handle labels *)
        If[dlen === 1,
            If[labels === Automatic || labels === None,
                 labels = None,
                 labels = {{1,labels}}
            ],
          (* else more than one data set *)
            If[labels === Automatic, labels = Range[dlen]];
            If[labels =!= None,
                labels = Transpose[{Range[dlen] + space,
                                    cyclevalues[labels, dlen]}]
            ]
        ];
        If[labels =!= None && orient === Horizontal,
            labels = Map[{-1, 1} * # &, labels]
        ];
      (* tell user about unknown options *)
        If[(badopts = Complement[First /@ Flatten[{newopts}],
                                 First /@ Flatten[{Options[BoxWhiskerPlot],
                                                   Options[Graphics]}]]) =!=
                 {},
            Message[BoxWhiskerPlot::badopts, badopts]
        ];
      (* output final option values *)
        {bq, bouts, orient, labels, boxsty, linesty, medsty,
            outliershape, space}
    ]

boxwhiskercore[data_, n_,
              {quantile_, outliers_, orient_, _, boxsty_, linesty_,
               medsty_, outshape_, extraspace_, ___}] :=
    boxwhiskergraphic[
        boxwhiskerstatistics[data, quantile, outliers],
        n + extraspace[[n]],
        orient,
        boxsty[[n]], linesty[[n]], medsty[[n]], outshape
    ]

Options[BoxWhiskerPlot] =
    {BoxQuantile -> 0.25,
     BoxOutliers -> None,
     BoxOrientation -> Vertical,
     BoxLabels -> Automatic,
     BoxFillingStyle -> Automatic,
     BoxLineStyle -> Automatic,
     BoxMedianStyle -> Automatic,
     BoxOutlierMarkers -> Automatic,
     BoxExtraSpacing -> 0} ~Join~
    $defaultgraphicsoptions;

SetOptions[BoxWhiskerPlot,
    Axes -> False,
    Frame -> True,
    AspectRatio -> Automatic];

BoxWhiskerPlot::quant =
"Value for BoxQuantile option is `1`; must be between 0 and 0.5. Setting \
to 0.25.";

BoxWhiskerPlot::orient =
"Value for BoxOrientation option is `1`; should be Vertical or Horizontal. \
Setting to Vertical.";

BoxWhiskerPlot::bouts =
"Value for BoxOutliers option is `1`; should be None, Automatic, or All. \
Setting to None.";

BoxWhiskerPlot::badopts =
"The options `1` are not valid for BoxWhiskerPlot, and will be ignored.";

BoxWhiskerPlot::boxsty =
"Value for BoxFillingStyle option is `1`; should be None or a color directive, \
or a list of these values. Setting to the default.";

obsoleteBoxWhiskerMessageFlag = True;

BoxWhiskerPlot[d1_?nvectorQ, d2__?nvectorQ, Shortest[opts___?OptionQ]] :=
    Module[{allopts, alldat = {d1, d2}, gr, dlen},
        If[TrueQ[obsoleteBoxWhiskerMessageFlag],
        	Message[General::obsfun, StatisticalPlots`BoxWhiskerPlot, BoxWhiskerChart];
        	obsoleteBoxWhiskerMessageFlag = False];
        dlen = Length[alldat];
        allopts = boxwhiskeroptions[dlen, opts];
        gr = MapIndexed[boxwhiskercore[#1, First[#2], allopts]&,
                   alldat
        ];
        Show[Graphics[gr,
            FilterRules[Flatten[{opts}],Options[Graphics]],
            If[allopts[[3]] === Vertical,
                {PlotRange -> {{0.5, dlen + Last[allopts[[-1]]] + 0.5}, All},
                 FrameTicks -> {allopts[[4]], Automatic, None, None},
                 AspectRatio -> (OptionValue[BoxWhiskerPlot, Flatten[{opts}], AspectRatio] /. 
 					Automatic -> 1/(0.6 dlen))},
              (* else *)
                {PlotRange -> {All, {-0.5, -(dlen + Last[allopts[[-1]]] + 0.5)}},
                 FrameTicks -> {Automatic, allopts[[4]], None, None},
                 AspectRatio -> (OptionValue[BoxWhiskerPlot, Flatten[{opts}], AspectRatio] /. 
 					Automatic -> (0.6 dlen))}
            ],
            FilterRules[Flatten[Join[{opts},Options[BoxWhiskerPlot]]],Options[Graphics]]
        ]]
    ]

BoxWhiskerPlot[data_?nvectorQ, opts___?OptionQ] :=
    Module[{allopts = boxwhiskeroptions[1, opts]},
        If[TrueQ[obsoleteBoxWhiskerMessageFlag],
        	Message[General::obsfun, StatisticalPlots`BoxWhiskerPlot, BoxWhiskerChart];
        	obsoleteBoxWhiskerMessageFlag = False];
        Show[Graphics[boxwhiskercore[data, 1, allopts],
            FilterRules[Flatten[{opts}],Options[Graphics]],
            AspectRatio -> (OptionValue[BoxWhiskerPlot, Flatten[{opts}], AspectRatio] /. 
 				Automatic -> 1),
            If[allopts[[3]] === Vertical,
                {PlotRange -> {{0.5, 1.5}, All},
                 FrameTicks -> {allopts[[4]], Automatic, None, None}},
              (* else *)
                {PlotRange -> {All, {-0.5, -1.5}},
                 FrameTicks -> {Automatic, allopts[[4]], None, None}}
            ],
            FilterRules[Flatten[Join[{opts},Options[BoxWhiskerPlot]]],Options[Graphics]]
        ]]
    ]
    

(* handle matrix data sets too *)
BoxWhiskerPlot[dat:((_?nvectorQ | _?nmatrixQ)..), opts___?OptionQ] :=
    BoxWhiskerPlot[##, opts]& @@
        Map[If[MatrixQ[#], Sequence @@ Transpose[#], #]&, {dat}]

(************************** ParetoPlot ****************************)

ParetoPlot::badopts =
"The options `1` are not valid for ParetoPlot, and will be ignored.";

ParetoPlot::dpopt = "The option `1` is deprecated. Use `2` instead."

(* for best appearance, Automatic handling of various options dictating
   the appearance is best dealt with in the Pareto routines rather than
   the called plotting functions. The remaining good options will just
   be passed through to the applicable routines. *)
paretooptionhandling[cats_, caller_, opts___] :=
    Module[{orient, labels, optsWithStringNames,
            goodopts, badopts, optnameStrings, len = Length[cats], newopts,
            style, edge, edgestyle},
      (* options that need special handling *)
        newopts=removeOldOpts[{opts}, {"SymbolShape","SymbolStyle"}, ParetoPlot];
      (* switch to strings to avoid creating symbols for deprecated options *)
      	optnameStrings = allParetoOptNames;
      (* tell user about unknown options *)
        goodopts = Select[Flatten[{newopts}], MemberQ[optnameStrings, ToString[First[#]]]&];
        If[(badopts = Complement[Flatten[{newopts}], goodopts]) =!= {},
            Message[MessageName[caller,"badopts"], First /@ badopts]
        ];
      (* tell user about deprecated options and tack new versions of the option settings 
         onto the end of the option list. that way option values from old options will 
         still be used, but only if option values are not explicitly given for new options *)
       optnameStrings=Map[ToString,goodopts[[All,1]]];
       optsWithStringNames=Map[ReplacePart[#, ToString[#[[1]]], 1] &,goodopts];
       If[MemberQ[optnameStrings, "BarLabels"],
       		Message[ParetoPlot::dpopt, "BarLabels", ChartLabels];
       		goodopts=Join[goodopts,{ChartLabels->("BarLabels"/.optsWithStringNames)}]];
       If[MemberQ[optnameStrings, "BarOrientation"],
       		Message[ParetoPlot::dpopt, "BarOrientation", BarOrigin];
       		goodopts=Join[goodopts,{BarOrigin->barOrientationToOrigin[("BarOrientation"/.optsWithStringNames)]}]];
       If[Intersection[optnameStrings,{"BarEdgeStyle","BarEdges","BarStyle"}]=!={},
       	edgestyle=If[MemberQ[optnameStrings, "BarEdgeStyle"],
       		Message[ParetoPlot::dpopt, "BarEdgeStyle", "ChartStyle with an EdgeForm specification"];
       		"BarEdgeStyle"/.optsWithStringNames,
       		Opacity[.5]];
       	edge=If[MemberQ[optnameStrings, "BarEdges"],
       		Message[ParetoPlot::dpopt, "BarEdges", "ChartStyle with an EdgeForm specification"];
       		"BarEdges"/.optsWithStringNames,
       		True];
       	style=If[MemberQ[optnameStrings, "BarStyle"],
       		Message[ParetoPlot::dpopt, "BarStyle", ChartStyle];
       		"BarStyle"/.optsWithStringNames,
       		Automatic];
      	goodopts=Join[goodopts,{ChartStyle->barStylesToChartStyle[style, edge, edgestyle]}]];
      (* remove any of the old options that appear in the list of options *)
      goodopts=removeOldOpts[goodopts, oldParetoOptNames, ParetoPlot];
      (* combine BarEdge, BarEdgeStyle, and BarStyle into a ChartStyle option *)
      {orient, labels} =
            {BarOrigin, ChartLabels}/.Flatten[
             	{goodopts, Options[caller]}];
      
      (* more option handling, for opts that require computed values *)
        If[labels === Automatic,
           labels = cats, 
           If[labels =!= None,
               labels = cyclevalues[labels, len]
           ]
        ];
        {BarOrigin -> orient, ChartLabels->labels, goodopts}
    ]

(* functions for changing BarOrientation, BarEdges, BarEdgeStyle, and BarStyle to charting options added in version 7 *)
barOrientationToOrigin[orient_] := 
 If[orient === Horizontal, Left, Bottom]

barStylesToChartStyle[bar_, edge_, edgestyle_] := 
 Directive[
  If[bar === Automatic, 
   RGBColor[0.798413061722744, 0.824719615472648, 0.968322270542458], 
   bar], If[MemberQ[{False, None}, edge], EdgeForm[None], 
   EdgeForm[edgestyle]]]

(* following is based on spacing definitions from BarChart.m; 
   these are needed to get the position of the points in the Pareto plot correct;
   in case of a bad spec, leave it to BarChart to catch and issue message *)
spacingrules = {
	Automatic|Small-> 0.1,
	None->0,
	Tiny->0.05,
	Medium->0.25,
	Large->0.5,
	All->1.0}

(* do the computations and display *)
paretoplotcore[dat_, caller_, opts___] :=
    Module[{sum, cats, vals, cumulative, fixopts, space, bc},
        {vals, cats} = Transpose[Reverse[Sort[dat]]];
        sum = Tr[vals];
        vals = vals/sum;
        cumulative = FoldList[Plus, First[vals], Rest[vals]];
      (* handle options *)
        fixopts = paretooptionhandling[cats, caller, opts];
        space=(BarSpacing/.Flatten[fixopts]/.Options[ParetoPlot])/.spacingrules;
        If[Head[space]===List&&Length[space]>0,
        	(* for BarChart, BarSpacing->{barspace, groupspace} can be given, 
        	   but only barspace is relevant to ParetoPlot *)
        	space=space[[1]]];
      (* display *)
      bc=Block[{$DisplayFunction=Identity},
       			BarChart[vals, BarSpacing->space,
       				DeleteCases[FilterRules[Join[Flatten[fixopts],
                	Options[caller]],Options[BarChart]],
                	_[Ticks | FrameTicks, Automatic]| _[DisplayFunction, _]]]];
      If[Not[FreeQ[bc,BarChart]&&NumberQ[N[space]]],
      	$Failed,
      	Apply[Show,{bc,
            Block[{$DisplayFunction=Identity},
       			ListPlot[
                   Switch[(BarOrigin/.Flatten[fixopts]),
                   		Left,
                        Transpose[{cumulative, Range[Length[cats]]}],
                        Right,
                        Transpose[{-cumulative, Range[Length[cats]]}],
                        Top,
                        Transpose[{Range[Length[cats]], -cumulative}],
                        _,
                        Transpose[{Range[Length[cats]], cumulative}]
                   ],DeleteCases[FilterRules[Flatten[{Last[fixopts], Options[caller]}],
                   		Options[ListPlot]],_[DisplayFunction|ColorFunction, _]]]
            ],
            FilterRules[Join[{opts},Options[caller]],{DisplayFunction}]}
        ]]
    ]

(* main function setup *)
Options[ParetoPlot] =
    {BarOrigin->Bottom,
   	 BarSpacing -> Automatic,
   	 ChartBaseStyle -> Automatic,
   	 ChartElementFunction -> Automatic, 
   	 ChartElements -> Automatic,
     ChartLabels->Automatic,
     ChartLegends->None,
     ChartStyle->Automatic,
     ColorFunction -> Automatic, 
     ColorFunctionScaling -> True,
     Joined->{True, False},
     LabelingFunction -> Automatic,
     PerformanceGoal :> $PerformanceGoal,
     PlotMarkers -> Automatic,
     PlotStyle -> Automatic} ~Join~
    $defaultgraphicsoptions;

SetOptions[ParetoPlot, Axes -> True, AspectRatio->1/GoldenRatio];

Options[ParetoPlot]=Sort[Options[ParetoPlot]]

(* use strings to avoid creating the symbols *)
oldParetoOptNames={
	"BarStyle",
     "BarLabels",
     "BarEdges",
     "BarEdgeStyle",
     "BarOrientation"}
     
allParetoOptNames=Join[oldParetoOptNames, Map[ToString,Options[ParetoPlot][[All,1]]]]

ParetoPlot[freqs:{{_, _?NonNegative}..}, opts___?OptionQ] :=Block[{res=
    paretoplotcore[
        Reverse /@
            (freqs//.{a___, {b_, c_}, d___, {b_, e_}, f___} :>
                         {a,{b, c + e}, d, f}), ParetoPlot, opts]},
    res/;res=!=$Failed]

ParetoPlot[data_List, opts___?OptionQ]/;Length[data] > 0 :=Block[{res=
    paretoplotcore[Tally[data][[All,{2,1}]], ParetoPlot, opts]},
    res/;res=!=$Failed]


(********************** PairwiseScatterPlot ***********************)

Options[PairwiseScatterPlot] =
    {DataSpacing -> 0,
     DataRanges -> All,
     DataLabels -> None,
     DataTicks -> None,
     PlotDirection -> {Right, Down},
     PlotStyle -> Automatic} ~Join~
    $defaultgraphicsoptions;

SetOptions[PairwiseScatterPlot,
   Axes -> False, Frame -> False, AspectRatio -> Automatic, PlotRange -> All
];

PairwiseScatterPlot::badrng =
"One or more values for the DataRanges option `1` is not \
All, or Automatic, or of the form {min, max}. Setting to All.";

PairwiseScatterPlot::badspace =
"The value for the DataSpacing option `1` is not numeric or a pair \
of numbers.";

PairwiseScatterPlot::badopts =
"The options `1` are not valid for PairwiseScatterPlot, and will be ignored.";

pspoptions[cols_, opts___] :=
    Module[{gridspace, ranges, stys, goodoptnames, goodopts, badopts, labels, plotdir,ticks},
        {gridspace, ranges, stys, labels, ticks, plotdir} =
           {DataSpacing, DataRanges, PlotStyle, DataLabels, DataTicks, PlotDirection}/.
            Flatten[{opts, Options[PairwiseScatterPlot]}];
      (* check and fix ranges option *)
        If[MatchQ[ranges, {_?NumericQ, _?NumericQ} | All | Automatic],
            ranges = {ranges}
        ];
        If[!MatchQ[ranges, {(All | Automatic |
                              ({_?NumericQ, _?NumericQ}?(Apply[Less, #] &)))..}],
            Message[PairwiseScatterPlot::badrng, ranges];
            ranges = {All}
        ];
        ranges = PadRight[ranges, cols, ranges];
      (* check spacing option *)
        If[!MatchQ[gridspace, _?NumericQ | {_?NumericQ, _?NumericQ}],
            Message[PairwiseScatterPlot::badspace, gridspace];
            gridspace = {0,0}
        ];
        If[!ListQ[gridspace], gridspace = {gridspace, gridspace}];
      (* arrange one style per subgraph *)
        If[!MatrixQ[stys], stys = {{stys}}];
        stys = PadRight[stys, {cols, cols}, stys]/.Automatic -> GrayLevel[0];
      (* check labels *)
        If[labels =!= None,
            labels = Which[labels === Automatic, Range[cols],
                           !ListQ[labels], Table[labels, {cols}],
                           True, PadRight[labels, cols, ""]
                     ];
        ];
      (* check ticks -- note that handling for individual columns will
         be done later. Note that use of False instead of None is
         undocumented, but supported in the spirit of 'do the right thing'. *)
        Which[ticks === False, ticks = None,
              ListQ[ticks], ticks = Map[If[# === False, None, #]&, ticks]
        ];
        If[ticks =!= None,
            ticks = cyclevalues[ticks, cols];
          (* shortcut for no ticks at all *)
            If[MatchQ[ticks, {None..}], ticks = None];
        ];
      (* check direction *)
      If[!MatchQ[plotdir, {Left|Right, Up|Down}],
      		Message[PairwiseScatterPlot::badpldr, gridspace];
            plotdir={Right,Down}];
      (* notify user of bad options *)
        goodoptnames = First /@ Options[PairwiseScatterPlot];
        goodopts = Select[Flatten[{opts}], MemberQ[goodoptnames, First[#]]&];
        If[(badopts = Complement[Flatten[{opts}], goodopts]) =!= {},
            Message[PairwiseScatterPlot::badopts, First /@ badopts]
        ];
        {gridspace, ranges, stys, labels, ticks, plotdir}
    ]

PairwiseScatterPlot[idata_?nmatrixQ, opts___?OptionQ] :=
    Module[{cols = Length[First[idata]], horizspace, vertspace, ranges,
            stys, data, gr, adjustedranges, ticks, labels, plotdir, pos},
      {{horizspace, vertspace}, ranges, stys, labels, ticks, plotdir} =
            pspoptions[cols, opts];
      pos = MapThread[pointPositions[idata[[All, #1]], #2] &, {Range[cols], ranges}];
	  ranges = MapThread[If[! ListQ[#1], {Min[#], Max[#]} &[idata[[All, #2]]], #1] &, 
	  		{ranges, Range[cols]}];
      data = Table[idata[[myIntersection[pos[[i]], pos[[j]]], {i, j}]], {j, cols}, {i, cols}];
	  (* tweak ranges for display and scale data to match *)
	  adjustedranges = Map[tweakrange, ranges];
      data = Table[scaledata[data[[j, i]], adjustedranges[[i]], adjustedranges[[j]]], 
      	{j, cols}, {i, cols}];
      (* get tick setup *)
      If[ticks =!= None,
            ticks = MapThread[generateticklocations,
                              {ticks, ranges, adjustedranges}];
            If[MatchQ[ticks, {{}..}], ticks = None]
        ];
      (* transform data to points used in graphics *)
	  data=Transpose[data];
	  If[plotdir[[1]]===Right,
	  	Do[data[[i + 1, All, All, 1]] = data[[i + 1, All, All, 1]] + i + horizspace * i;, 
  	  		{i, 0, cols - 1}],
  	  	Do[data[[i + 1, All, All, 1]] = data[[i + 1, All, All, 1]] + 
  	  		cols-i-1+horizspace(cols-i+1);, 
  	  		{i, 0, cols - 1}]];
  	  If[plotdir[[2]] === Up,
	  	Do[data[[All, i + 1, All, 2]] = data[[All, i + 1, All, 2]] + i + vertspace * i;, 
  	  		{i, 0, cols - 1}],
  	  	Do[data[[All, i + 1, All, 2]] = data[[All, i + 1, All, 2]] + 
  	  		cols-i-1 + vertspace(cols-i-1);, {i, 0, cols - 1}]];
      (* construct graphics *)
        gr = Flatten[
          Table[{Line[Transpose[{m + {0, 1, 1, 0, 0} + horizspace * m,
					n + {0, 0, 1, 1, 0} + vertspace * n}]],
          		 Flatten[{stys[[n + 1, m + 1]],Map[Point , data[[n + 1, m + 1]]]}]},
                 {m, 0, cols - 1}, {n, 0, cols - 1}], 
                1];
        If[labels =!= None,
            labels = makepsplabels[labels, horizspace, vertspace,
                                   cols, ticks =!= None, plotdir],
            labels = {}
        ];
        If[ticks =!= None,
            ticks = Table[makepspticks[
  					If[plotdir[[1]] === Right, {ticks[[n + 1]], adjustedranges[[n + 1]]}, 
  						{ticks[[cols - n]], adjustedranges[[cols - n]]}], 
  					If[plotdir[[2]] === Up, {ticks[[n + 1]], adjustedranges[[n + 1]]}, 
  						{ticks[[cols - n]], adjustedranges[[cols - n]]}],
                	cols, n, horizspace, vertspace,plotdir],
                 {n, 0, cols - 1}
            ],
            ticks = {}
        ];
        Show[Graphics[{gr, labels, ticks}],
            FilterRules[Join[Flatten[{opts}], Options[PairwiseScatterPlot]],
            	Options[Graphics]]
        ]
    ]


(* pointPositions and myIntersection are used by PairwiseScatterPlot to select
   points based on the given DataRanges *)    
pointPositions[dat_, {min_, max_}] := 
 Flatten[Position[dat, _?(min <= # <= max &), 1, Heads -> False]]

pointPositions[dat_, All] := All

pointPositions[dat_, Automatic] := All

myIntersection[lis1_List, All] := lis1

myIntersection[All, lis2_List] := lis2

myIntersection[All, All] := All

myIntersection[lis1_, lis2_] := Intersection[lis1, lis2]

tweakrange[{min_, max_}] :=
    {-1, 1} * 0.05 * (max - min) + {min, max}

scaledata[data_, {min_, max_}] :=If[min==max,
	data-min,
    (data - min)/(max - min)]
    
scaledata[data_, {minx_, maxx_}, {miny_, maxy_}] := 
 Transpose[{
 	If[minx==maxx, data[[All, 1]] - minx, (data[[All, 1]] - minx)/(maxx - minx)], 
 	If[miny==maxy, data[[All, 2]] - miny, (data[[All, 2]] - miny)/(maxy - miny)]
 	}]

(* labels to graphics primitives *)
(***** I really need a better way to handle determination of offset
       for label placement. Unfortunately, there isn't yet a *good*
       way to do so. So currently the offsets are fixed 'magic' numbers
       used when ticks are present. Sigh. *)
makepsplabels[labels_, hspace_, vspace_, cols_, tickflag_,{xdir_,ydir_}] :=
    Module[{vpos, hpos, voff, hoff, xlabs, ylabs, p},
        If[tickflag,
           voff = 12; hoff = 30,
           voff = 0; hoff = 0
        ];
        If[vspace < -1, vpos = cols - 1 + (cols - 1) * vspace, vpos = 0];
        If[hspace < -1, hpos = cols - 1 + (cols - 1) * hspace, hpos = 0];
        xlabs = If[xdir === Right, labels, Reverse[labels]];
		ylabs = If[ydir === Up, labels, Reverse[labels]];
		{MapIndexed[(p = First[#2];
			Text[#, Offset[{0, -voff}, {0.5 + (p - 1) * (1 + hspace), vpos}],
                           {0, 1.3}]) &,
            xlabs],
         MapIndexed[(p = First[#2];
    		Text[#, Offset[{-hoff, 0}, {hpos, 0.5 + (p - 1) * (1 + vspace)}],
                           {1.3, 0}]) &,
               ylabs]}
    ]

(* tick handling. Something like this should probably be made into
   a generic Developer` functionality; or similiary, the internal System`
   functionality should be externally accessible... Note that as written,
   this is fairly generic; it doesn't handle the location rescaling
   required by the PSP routines... *)
generateticklocations[None, _, _] := {}
    (***** should the number of ticks adjust to the grid density? *)
generateticklocations[Automatic | All | True, {min_, max_}, boundrng_ ] :=
    generateticklocations[LinearScale[min, max, 5], {min, max}, boundrng]
generateticklocations[locs_List, genrng_, {min_, max_}] :=
    Select[Map[fixsingletick, locs], min <= First[#] <= max &]
(* fallthrough to arbitrary function; clean up function's result if
   it is a valid tick spec, otherwise no ticks from this one *)
generateticklocations[any_, {min_, max_}, br_] :=
   If[ListQ[#] || # === None || # === Automatic,
          generateticklocations[#, {min, max}, br], {}]&[
       any[min, max]
   ]

(*LinearScale and TickSpacing inlined from obsolete Graphics`Graphics` standard add-on*)
LinearScale[min_, max_, n_Integer:8] :=
    Module[{spacing, t, nmin=N[min], nmax=N[max]},
        (spacing = TickSpacing[nmax-nmin, n, {1, 2, 2.5, 5, 10}] ;
        t = N[spacing * Range[Ceiling[nmin/spacing - 0.05],
                              Floor[max/spacing + 0.05]]] ;
        Map[{#, If[Round[#]==#, Round[#], #]}&, t])
    /; nmin < nmax
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

fixsingletick[n_?NumericQ] := defaulttick[{n, n}]
fixsingletick[{n_?NumericQ}] := defaulttick[{n, n}]
fixsingletick[{n_?NumericQ, l_}] := defaulttick[{n, l}]
fixsingletick[{n_?NumericQ, l_, d_?NumericQ}] := defaulttick[{n, l, {d, 0}}]
fixsingletick[{n_?NumericQ, l_, d:{_?NumericQ, _?NumericQ}}] :=
    defaulttick[{n, l, d}]
fixsingletick[{n_?NumericQ, l_, d_?NumericQ, s_}] :=
    {n, l, {d, 0}, s}
fixsingletick[{n_?NumericQ, l_, d:{_?NumericQ, _?NumericQ}, s_}] :=
    {n, l, d, s}  (* sigh, should arg-check the style... *)
fixsingletick[any_] := {-Infinity} (* bad tick -- I'm letting these get
                                      filtered by the Select above *)

$defaulttick = {0, 0, {0.01, 0}, GrayLevel[0]}
defaulttick[l_] :=
    Join[l, Drop[$defaulttick, Length[l]]]

(****** the following two functions seem more complicated than they
        need to be to me; rethink them at some point... --JMN 09/02 *)
(* create lines from tick specs; this handles the location rescaling
   and graphics primitive generation. *)
makepspticks[{xticks_, {minx_, maxx_}}, {yticks_, {miny_, maxy_}}, cols_, offset_, 
	horiz_, vert_, {xdir_, ydir_}] :=
    Module[{hlocadjust = (1 + vert)*offset,
			vlocadjust = (1 + horiz)*offset,
			hmaxposn = cols + (cols - 1)*horiz,
			vmaxposn = cols + (cols - 1)*vert},
        {Map[maketick[#, {minx, maxx}, vlocadjust, vmaxposn, EvenQ[offset], True, 
    		vert < -1] &, xticks], 
 		Map[maketick[#, {miny, maxy}, hlocadjust, hmaxposn, EvenQ[offset], False, 
    		horiz < -1] &, yticks]}
    ]

maketick[{t_, l_, {ipos_, ineg_}, s_},
         rng_, offset_, imaxposn_,
         textbottomflag_, horizorvertflag_, swapflag_] :=
    Module[{loc = scaledata[t, rng] + offset, pos = ipos, neg = ineg,
            minposn = 0, maxposn = imaxposn, textpos, textoffset},
        If[swapflag, minposn = imaxposn - 1; maxposn = 1];
        If[Xor[textbottomflag, swapflag],
            textpos = minposn;
            textoffset = 1.1,
            textpos = maxposn;
            textoffset = -1.1
        ];
        If[horizorvertflag,
             textpos = {loc, textpos};
             textoffset = {0, textoffset};
             pos = {0, pos}; neg = {0, -neg};
             minposn = {loc, minposn}; maxposn = {loc, maxposn},
             textpos = {textpos, loc};
             textoffset = {textoffset, 0};
             pos = {pos, 0}; neg = {-neg, 0};
             minposn = {minposn, loc}; maxposn = {maxposn, loc}
        ];
        {Text[l, textpos, textoffset],
         Flatten[{s,
             Line[{Scaled[pos, minposn], Scaled[neg, minposn]}],
             Line[{Scaled[-pos, maxposn], Scaled[-neg, maxposn]}]
         }]}
    ]


(********************** StemLeafPlot ******************************)


Options[StemLeafPlot] = {StemExponent -> Automatic, 
	IncludeEmptyStems -> False, ColumnLabels -> Automatic, 
	Leaves -> "Digits", IncludeStemUnits ->True, IncludeStemCounts -> False};
         
StemLeafPlot::collab = "`1` is not a valid ColumnLabels specification. \
ColumnLabels must be a length `2` list of labels or Automatic. Automatic \
column labelling will be used.";

StemLeafPlot::lvs = "The value `1` for the Leaves option is not \"Digits\", \
\"Tallies\", None or a list containing one of these values and additional \
options. The default value will be used.";

StemLeafPlot::stdivs = "The value `1` for the \"UnitDivisions\" option is not \
a positive integer. The value 1 will be used.";

StemLeafPlot::stexp = "The value `1` for the StemExponent option is not an \
integer or Automatic or a list containing an integer and additional options. \
Automatic will be used.";

StemLeafPlot::lfrnd = "The value `1` for the \"RoundLeaves\" option is neither \
True nor False. True will be used.";

StemLeafPlot::lfdgpr = "The precision of the data is insufficient to obtain  \
the requested number of \"LeafDigits\" `1`. `2` digits will be used for the leaves.";

StemLeafPlot::lfpos="The value `1` for the \"LeafDigits\" option is not a positive \
integer. The value 1 will be used.";

StemLeafPlot::lfsp = "The value `1` for the \"LeafSpacing\" option is not a  \
positive integer or Automatic.  The default value of Automatic will be used.";

StemLeafPlot::stun = "The value `1` for the IncludeStemUnits option is \
neither True nor False.  The default value of True will be used.";

StemLeafPlot::stcts = "The value `1` for the IncludeStemCounts option is \
neither True nor False.  The default value of False will be used.";

StemLeafPlot::divlab = "The value `1` for the \"DivisionLabels\" option is \
neither None nor a list of length equal to the \"UnitDivisions\" value `2`. \
Default division labelling will be used.";
    
StemLeafPlot::lfwrp = "The value `1` for the \"LeafWrapping\" option is neither \
None nor a postive integer. No leaf wrapping will be done.";

StemLeafPlot::stmtst="The value `1` for the IncludeEmptyStems option is neither \
True nor False. False will be used.";

StemLeafPlot::ndata ="The argmuent at position `1` is not a list of real numbers.";
    
StemLeafPlot::badopts ="The options `1` are not valid for StemLeafPlot, \
and will be ignored.";         

StemLeafPlot::rndlvs="Due to rounding for display, one or more leaves has more \
than the requested number of leaf digits `1`. These leaves will be replaced by a \
string of `1` tally symbols.";
  
         
(* ProcessOptionNames is used to convert symbol suboptions to string suboptions *)
SetAttributes[ProcessOptionNames, Listable];

ProcessOptionNames[(r : (Rule | RuleDelayed))[name_Symbol, val_]] := 
	r[SymbolName[name], val];

ProcessOptionNames[opt_] := opt;

(* function for processing StemLeafPlot options *)
slOptions[data_, datasets_, opts___] := 
  Block[{stemlevel, stemdivs, stemlabs, emptystems, collabels, leafdigs, rounddig, 
  	leafspace, deflabels, units, stemcts, leavesopt, leafwrap, tallies, talsymb, includeleaves},
    {stemlevel, emptystems, collabels, leavesopt, units, stemcts} = {StemExponent, 
       IncludeEmptyStems, ColumnLabels, Leaves, IncludeStemUnits, IncludeStemCounts
       } /. {opts} /. Options[StemLeafPlot];
    (* identify bad options and issue message *)
    With[{bopts = Complement[{opts}[[All, 1]], Join[Options[GridBox], Options[StemLeafPlot]][[All, 1]]]},
        If[bopts =!= {}, Message[StemLeafPlot::badopts, bopts]]];
    (* set StemExponent and its suboption values *)
    {stemlevel, stemdivs, stemlabs} = 
       With[{stemexpsubopts = If[Head[stemlevel] === List, ProcessOptionNames[Rest[stemlevel]]]},
           Which[stemlevel === Automatic
           	,
            {If[PossibleZeroQ[#], 0, Floor[Log[10, #]]] &[Mean[Abs[data]]], 1, None}
            ,
            Head[stemlevel] === List && stemlevel[[1]] === Automatic
            ,
            {If[PossibleZeroQ[#], 0, Floor[Log[10, #]]] &[Mean[Abs[data]]], 
             "UnitDivisions" /. stemexpsubopts /. {"UnitDivisions" -> 1}, 
             "DivisionLabels" /. stemexpsubopts /. {"DivisionLabels" -> None}}
            ,
            Head[stemlevel] === List && IntegerQ[stemlevel[[1]]]
            ,
            {stemlevel[[1]], "UnitDivisions" /. stemexpsubopts /. {"UnitDivisions" -> 1}, 
             "DivisionLabels" /. stemexpsubopts /. {"DivisionLabels" -> None}}
            ,
            IntegerQ[stemlevel]
            ,
            {stemlevel, 1, None}
            ,
            True
            ,
            Message[StemLeafPlot::stexp, If[Head[stemlevel] === List, stemlevel[[1]], stemlevel]]; 
            {If[PossibleZeroQ[#], 0, Floor[Log[10, #]]] &[Mean[Abs[data]]], 1, None}
          ]];
      If[! (IntegerQ[stemdivs] && Positive[stemdivs]), 
      	Message[StemLeafPlot::stdivs, stemdivs]; stemdivs = 1];
      If[Length[stemlabs] =!= stemdivs && stemlabs =!= None, 
      	Message[StemLeafPlot::divlab, stemlabs, stemdivs]; stemlabs = None];
      If[!MemberQ[{True, False}, stemcts], 
      	Message[StemLeafPlot::stcts, stemcts]; stemcts = False];
      {leafdigs, rounddig, leafwrap, tallies, talsymb, includeleaves} = 
      	With[{leavessubopts = If[Head[leavesopt] ===List, ProcessOptionNames[Rest[leavesopt]]]}, 
      		Which[ToString[leavesopt]==="Digits"
      		,
            {1, True, None, False, None, True}
            ,
            leavesopt === None
            ,
            {1, True, None, False, None, False}
            ,
            ToString[leavesopt] === "Tallies"
            ,
            {1, True, None, True, "X", True}
            ,
            Head[leavesopt] ===List && ToString[leavesopt[[1]]]==="Digits"
            ,
            (* extract TallySymbol here in case it is needed for unrounded leaves *)
            {"LeafDigits" /. leavessubopts /. {"LeafDigits" -> 1}, 
             "RoundLeaves" /. leavessubopts /. {"RoundLeaves" -> True}, 
             "LeafWrapping" /. leavessubopts /. {"LeafWrapping" -> None}, 
             False, "TallySymbol" /. leavessubopts /. {"TallySymbol" -> "X"}, True}
            ,
            Head[leavesopt] === List && leavesopt[[1]] === None
            ,
            {"LeafDigits" /. leavessubopts /. {"LeafDigits" -> 1}, 
             "RoundLeaves" /. leavessubopts /. {"RoundLeaves" -> True}, 
             None, False, None, False}
            ,
            Head[leavesopt] === List && ToString[leavesopt[[1]]] === "Tallies"
            ,
            {"LeafDigits" /. leavessubopts /. {"LeafDigits" -> 1}, 
             "RoundLeaves" /. leavessubopts /. {"RoundLeaves" -> True}, 
             "LeafWrapping" /.leavessubopts /. {"LeafWrapping" -> None}, 
             True, "TallySymbol" /. leavessubopts /. {"TallySymbol" -> "X"}, True}
            ,
            True
            ,
			Message[StemLeafPlot::lvs, leavesopt]; 
			{1, True, None, False, None, True}
          ]];
      (* adjust leafdigs and issue a message if the precision of the data 
            is not sufficient to get the requested number of digits *)
      If[Not[TrueQ[IntegerQ[leafdigs]&&leafdigs>0]],
      	Message[StemLeafPlot::lfpos, leafdigs];leafdigs=1,
      	With[{olddigs = leafdigs, 
      		newdigs = Ceiling[Max[0, Min[leafdigs, Precision[data] - 
      			If[PossibleZeroQ[#], 0, Floor[Log[10, #]]] &[Max[Abs[data]]] + stemlevel - 1]]]},
          	  If[olddigs =!= newdigs, 
        		Message[StemLeafPlot::lfdgpr, olddigs, newdigs]; leafdigs = newdigs]]
        ];
      (* check suboption values; issue messages and re-set if necessary *)
      If[! MemberQ[{True, False}, rounddig], 
      	Message[StemLeafPlot::lfrnd, rounddig]; rounddig = True];
      If[Not[MemberQ[{None, Infinity}, leafwrap] || (IntegerQ[leafwrap] && Positive[leafwrap])], 
      	Message[StemLeafPlot::lfwrp, leafwrap]; leafwrap = None];
      leafspace = If[Head[leavesopt] === List, 
      	"LeafSpacing" /. ProcessOptionNames[Rest[leavesopt]] /. {"LeafSpacing" -> Automatic}, 
      	Automatic];
      leafspace = Which[leafdigs <= 1 && leafspace === Automatic,
          0,
          leafdigs > 1 && leafspace === Automatic,
          1,
          IntegerQ[leafspace] && leafspace>=0,
          leafspace,
          True,
          Message[StemLeafPlot::lfsp, leafspace]; If[leafdigs > 1, 1, 0]];
      deflabels = Which[datasets == 1 && stemcts && includeleaves, 
      	{"Stem", "Leaves", "Counts"},
        datasets == 1 && stemcts && Not[includeleaves], 
        {"Stem", "Counts"},
        datasets == 1 && Not[stemcts] && Not[includeleaves], 
        {"Stem"},
        datasets == 1 && Not[stemcts] && includeleaves, 
        {"Stem", "Leaves"},
        stemcts && includeleaves, 
        {"Counts", "Leaves", "Stem", "Leaves", "Counts"},
        stemcts && Not[includeleaves], 
        {"Counts", "Stem", "Counts"},
        Not[stemcts] && Not[includeleaves], 
        {"Stem"},
        True, 
        {"Leaves", "Stem", "Leaves"}];
      collabels = Which[collabels === Automatic,
          deflabels,
          VectorQ[collabels] && stemcts && includeleaves && 
          	Length[collabels] === 2datasets + 1,
          collabels,
          VectorQ[collabels] && Not[stemcts] && includeleaves && 
            Length[collabels] === datasets + 1,
          collabels,
          VectorQ[collabels] && stemcts && Not[includeleaves] && 
          	Length[collabels] === datasets + 1,
          collabels,
          VectorQ[collabels] && stemcts && Not[includeleaves] && 
          	Length[collabels] === datasets + 1,
          collabels,
          VectorQ[collabels] && Not[stemcts] && Not[includeleaves] && 
          	Length[collabels] === 1,
          collabels,
          True,
          Message[StemLeafPlot::collab, collabels, datasets + 1]; deflabels];
      If[! MemberQ[{True, False}, units], 
      	Message[StemLeafPlot::stun, units]; units = True];
      If[! MemberQ[{True, False}, emptystems], 
      	Message[StemLeafPlot::stmtst, emptystems]; emptystems = False];
      {stemlevel, stemdivs, stemlabs, emptystems, collabels, leafdigs, rounddig, leafspace, 
       leafwrap, tallies, talsymb, includeleaves, units, stemcts}];


(* this function computes stems and leaves from a data set; the optional sortfun option is 
   used to reverse leaf order for negative data or side-by-side plots *)
getStemsLeaves[data_, leafdigs_, stemdivs_, stemlevel_, stemcts_, rounddig_, negative_:False] := 
	Block[{sls, stems, leaves},
    	digs = If[rounddig,
			Round[data*10^-(stemlevel - leafdigs)]/10^(leafdigs),
        	data*10^-stemlevel];
    	sls = Transpose[{IntegerPart[digs], Floor[stemdivs*FractionalPart[Rationalize[digs, 0]]],
            	FractionalPart[digs]*10^leafdigs}];
       	If[negative, sls=Map[If[#[[{2,3}]]==={0,0},{#[[1]],stemdivs-1,0},#]&, sls]]; 
     	sls=Sort[sls];
        (* the following replace is needed for cases where leaves are not rounded, 
           and a negative value falls on the lower stem boundary *)
    	If[stemdivs===1,sls[[All,2]]=sls[[All,2]]/.{-1->0}];
    	sls = Split[sls, Most[#1] === Most[#2] &];
    	stems = sls[[All, 1, {1, 2}]];
    	stems[[All, 2]] = Mod[stems[[All, 2]], stemdivs];
    	leaves = Map[Round[Sort[#[[All, 3]]]] &, sls];
    	If[Length[stems] > 1 && stems[[-1]] === stems[[-2]],
      		stems = Most[stems];
      		leaves[[-2]] = Join[leaves[[-2]], leaves[[-1]]];
      		leaves = Most[leaves]];
    	{stems, leaves, If[stemcts, Map[Length, leaves], stemcts]}]


(* this function allows for strings to be passed through like integers in 
   the case where a long unrounded leaf must be replaced by a tally string *)
spacingIntegerDigits[xx_String]:=Characters[xx]

spacingIntegerDigits[xx_]:=IntegerDigits[xx]

(* function for spacing leaves and turning them into strings *)
spaceLeaves[leafdigs_, leafspace_, leaves_, leafwrap_, tallies_, talsymb_,wrapdir_:Right] := 
  Block[{spacefun, ii, jj, j, xx, spacefunsymb, leafstrings, dir},
	spacefun[ii_, jj_, xx_,dir_:Right] := With[{newstring=If[dir===Left,StringReverse[#],#]&[Apply[StringJoin, 
		Map[Join[With[{len = Length[spacingIntegerDigits[#]]}, If[len < jj, Table["0", {jj - len}], {}]], 
			{ToString[#]},Table[" ", {ii}]] &, xx]]]},
			Which[StringLength[newstring] > 0 && dir === Left,
  				StringDrop[newstring, ii],
  				StringLength[newstring] > 0,
  				StringDrop[newstring, -ii], 
  				True, newstring]];
	spacefunsymb[ii_,jj_, xx_] := Apply[StringJoin, 
		Table[StringJoin[ToString[xx], If[j<jj,Apply[StringJoin, Table[" ", {ii}]],""]], {j, jj}]];
	leafstrings=If[wrapdir===Left&&Not[leafspace === 0&&leafdigs===1],
    	Map[Reverse,leaves],
    	leaves];
    leafstrings = Which[leafspace === 0 && tallies,
          Map[Apply[StringJoin, Map[ToString, 
                  Table[talsymb, {#}]]] &, Map[Length, leafstrings]],
          leafspace === 0&&leafdigs===1,
          Map[Apply[StringJoin, Map[ToString, #]] &, leafstrings],
          leafspace =!= 0 && tallies,
          Map[spacefunsymb[leafspace, #, talsymb] &, Map[Length, leafstrings]],
          True,
          Map[If[Length[#]>0,spacefun[leafspace, leafdigs, #,wrapdir],""] &, leafstrings]];
    (* wrap leaves if necessary *)
    leafstrings=With[{wrapdigs=If[tallies,StringLength[ToString[talsymb]],leafdigs]},
    	leafstrings=If[IntegerQ[leafwrap], 
    	  Map[With[{newstring = StringReplace[StringInsert[#, "\n", 1 + (leafwrap*(wrapdigs + 
    		leafspace))*Union[DeleteCases[Floor[Range[StringLength[#]]/(leafwrap*(wrapdigs + 
    		leafspace))], 0]]], 
    		{Apply[StringJoin, Flatten[{Table[" ", {leafspace}],"\n"}]] -> "\n",
    		Apply[StringJoin, Flatten[{"\n", Table[" ", {leafspace}]}]] -> "\n"}]},
            If[StringLength[newstring]>0&&StringTake[newstring, -1] === "\n", StringDrop[newstring, -1], newstring]] &, leafstrings]
        	, 
            leafstrings];
    	If[wrapdir===Left,
        	Map[reversewrap[#,leafwrap,wrapdigs,leafspace]&,leafstrings],
        	leafstrings]];
     If[Not[tallies]&&ToString[talsymb/.None->"X"]=!="X",
    	Map[StringReplace[#,"X"->ToString[talsymb]]&,leafstrings],leafstrings]]

(* still need to pad last row with spaces to get desired justification *)
reversewrap[str_,lfwrap_,lfdigs_,lfspace_]:=Module[{
    pos=Transpose[{#-(lfwrap*(lfdigs+lfspace))+lfspace,#-1}]&[StringPosition[str,"\n"][[All,1]]]},
    pos=If[pos==={},
    	{{1,StringLength[str]}},
    	Join[pos,{{pos[[-1,-1]]+2,StringLength[str]}}]];
    StringReplacePart[str,Map[StringReverse[StringTake[str,#]]&,pos],pos]]           

(* function for adding empty stems to plot if IncludeEmptyStems->True *)            
fillEmpty[stems_, leaves_, stemdivs_, stemcts_, leafcts_] :=
  If[stems === {}, 
  	{{}, {}, {}},
    If[stemcts
    	, 
    	Transpose[Sort[Join[Transpose[{stems, leaves, leafcts}], 
    	 Map[{#, {}, 0} &, 
    	  Complement[
    	   DeleteCases[Flatten[Outer[List, Range[stems[[1, 1]], stems[[-1, 1]]], Range[stemdivs] - 1], 1], 
    	    _?(FromDigits[#] < FromDigits[stems[[1]]] || FromDigits[#] > FromDigits[stems[[-1]]] &)], 
    	   stems]]]]]
    	,
      Join[Transpose[Sort[Join[Transpose[{stems, leaves}], 
        Map[{#, {}} &, Complement[
        	DeleteCases[Flatten[Outer[List, Range[stems[[1, 1]], stems[[-1, 1]]], Range[stemdivs] - 1], 1], 
        	  _?(FromDigits[#] < FromDigits[stems[[1]]] || FromDigits[#] > FromDigits[stems[[-1]]] &)], 
        	stems]]]]], {False}]]]
          
(* positive and negative values are handled separately; this function fills in gaps between 
   positive and negative stems if IncludeEmptyStems->True *)         
fillPosNegGap[stemsneg_, stemsnonneg_, stemdivs_, stemcts_] := 
  Module[{fillstemsneg, fillstemsnonneg, allfillstems},
    If[stemsneg === {} || stemsnonneg === {},
      {{}, {}, {}},
      fillstemsneg = Flatten[Join[If[stemsneg[[-1, 2]] + 2 <= stemdivs, 
      	Outer[List, {stemsneg[[-1, 1]]}, Range[stemsneg[[-1, 2]] + 1, stemdivs - 1]], {}], 
      	If[stemsneg[[-1, 1]] + 1 <= 0, Outer[List, Range[stemsneg[[-1, 1]] + 1, 0], 
      	  Range[stemdivs] - 1], {}]], 1];
      fillstemsneg[[All, 1]] = Map[If[# == 0, "-0", ToString[#]] &, fillstemsneg[[All, 1]]];
      fillstemsnonneg = Flatten[Join[If[stemsnonneg[[1, 1]] + 1 >= 0, 
      	Outer[List, Range[0, stemsnonneg[[1, 1]] - 1], Range[stemdivs] - 1], {}], 
      	If[stemsnonneg[[1, 2]] <= stemdivs, Outer[List, {stemsnonneg[[1, 1]]}, 
      		Range[0, stemsnonneg[[1, 2]] - 1]], {}]], 1];
      fillstemsnonneg[[All, 1]] = Map[ToString, fillstemsnonneg[[All, 1]]];
      {allfillstems = Join[fillstemsneg, fillstemsnonneg], Table[{}, {Length[allfillstems]}], 
      	If[stemcts, Table[0, {Length[allfillstems]}], {False}]}
      ]]
      

(* function for adding empty stems to side-by-side plot if IncludeEmptyStems->True *)         
fillEmptySidebySide[leaves1_, stems_, leaves2_, stemdivs_, stemcts_, leafcts1_, leafcts2_] :=
  If[stemcts
  	,
    Transpose[Sort[Join[Transpose[{leafcts1, leaves1, stems, leaves2, leafcts2}], 
   	  Map[{0, {}, #,{}, 0} &, 
    	Complement[DeleteCases[Flatten[Outer[List, Range[stems[[1, 1]], stems[[-1, 1]]], 
    		Range[stemdivs] - 1], 1], 
    		_?(FromDigits[#] < FromDigits[stems[[1]]] || FromDigits[#] > FromDigits[stems[[-1]]] &)], 
    	  stems]]], FromDigits[#1[[3]]] < FromDigits[#2[[3]]] &]]
   	,
    Join[{False}, Transpose[Sort[Join[Transpose[{leaves1, stems, leaves2}], 
      Map[{{}, #, {}} &, Complement[DeleteCases[Flatten[Outer[List, Range[stems[[1, 1]], 
          stems[[-1, 1]]], Range[stemdivs] - 1], 1], 
          _?(FromDigits[#] < FromDigits[stems[[1]]] || FromDigits[#] > FromDigits[stems[[-1]]] &)], 
         stems]]], FromDigits[#1[[2]]] < FromDigits[#2[[2]]] &]], {False}]]
         
         
realQ[val_] := Element[val, Reals];

(* function for splitting data into negative and non-negative values *)
negativeSplit[data_List] := 
  Module[{groups = Reap[Scan[Sow[#, Negative[#]] &, data], _, List][[2]], 
          negvals, posvals}, 
          negvals = Select[groups, (#[[1]] === True &)];
    	posvals = Select[groups, (#[[1]] === False &)];
    	If[Length[negvals] > 0, negvals = negvals[[1, 2]]];
    	If[Length[posvals] > 0, posvals = posvals[[1, 2]]];
    	{negvals, posvals}
    	]
    	
fixNegativeZero[{stems_, leaves_,leafcts_}, stempos_,stemdivs_, side_:Right] := 
  Module[{pos = Position[stems, {stems[[stempos,1]], 1}], stemsnew = stems, 
  		leavesnew = leaves, leafctsnew = leafcts},
    If[pos === {}, stemsnew[[stempos]] = {stems[[stempos,1]], stemdivs - 1},
      leavesnew[[pos[[1, 1]]]] = If[side === Right,
          Join[leavesnew[[pos[[1, 1]]]], leavesnew[[stempos]]],
          Join[leavesnew[[stempos]], leavesnew[[pos[[1, 1]]]]]];
      stemsnew = Drop[stemsnew, {stempos}];
      leavesnew = Drop[leavesnew, {stempos}];
      If[VectorQ[leafcts], 
      	leafctsnew[[pos[[1, 1]]]] = leafctsnew[[pos[[1, 1]]]] + leafctsnew[[stempos]];
      	leafctsnew=Drop[leafctsnew, {stempos}]]]; 
    {stemsnew, leavesnew, leafctsnew}]
    

(* valid input interface to internal code for a single data set *)    
StemLeafPlot[data_?(VectorQ[#, realQ] &), opts___?OptionQ] := 
	Module[{result = iSLPlot[data, opts]}, result /; (result =!= $Failed)]
	
(* valid input interface to internal code for two data sets *)	
StemLeafPlot[data1_?(VectorQ[#, realQ] &), data2_?(VectorQ[#, realQ] &), opts___?OptionQ] := 
	Module[{result = iSLPlot2[data1, data2, opts]}, result /; (result =!= $Failed)]
	
	
(* function for constructing stem-and-leaf plot for one data set *)	
iSLPlot[data_, opts___] := Block[
	{stemlevel, stemdivs, dataneg, datanonneg, stemsneg, leavesneg, leafctsneg, stemsnonneg, 
     leavesnonneg, leafctsnonneg, stemsfill, leavesfill, leafctsfill, emptystems, stems, 
     leaves, leafcts, collabels, leafdigs, rounddig, leafspace, units, stemcts, res, stemlabs, 
     tallies, talsymb, includeleaves, leafwrap},
    {stemlevel, stemdivs, stemlabs, emptystems, collabels, leafdigs, rounddig, leafspace, 
     leafwrap, tallies, talsymb, includeleaves, units, stemcts} = slOptions[data, 1, opts];
    (* split data into negative and non-negative parts; this allows for differentiating 
       between positive and negative 0 stems *)
    {dataneg, datanonneg} = negativeSplit[data];
    (* get stems and leaves for negative and non-negative parts *)
    {stemsneg, leavesneg, leafctsneg} = getStemsLeaves[dataneg, leafdigs, stemdivs, stemlevel, 
    	stemcts, rounddig, True];
    (* splitting data into negative and non-negative parts can result in separation of 
       stems with 0 leaves into their own category; need to check for and fix this if it occurs *)
    If[stemdivs>1&&Length[stemsneg]>0,
    	{stemsneg, leavesneg, leafctsneg}=
    		Fold[fixNegativeZero[#1, #2, stemdivs] &, 
    			{stemsneg, leavesneg, leafctsneg}, 
    			Reverse[Flatten[
  				  With[{pos = Position[stemsneg, {_, 0}]},
           			Extract[pos, Position[Map[Union, Extract[leavesneg, pos]], {0}]]]]]]
      ];
    {stemsnonneg, leavesnonneg, leafctsnonneg} = getStemsLeaves[datanonneg, leafdigs, 
    	stemdivs, stemlevel, stemcts, rounddig];
   	(* add empty stems if necessary *)
    If[TrueQ[emptystems],
      (* fill in empty stems for negative and nonnegative portions of data *)
      {stemsneg, leavesneg, leafctsneg} = 
      	fillEmpty[stemsneg, leavesneg, stemdivs, stemcts, leafctsneg];
      {stemsnonneg, leavesnonneg, leafctsnonneg} = 
      	fillEmpty[stemsnonneg, leavesnonneg, stemdivs, stemcts, leafctsnonneg];
      (* fill in gap between positive and negative stems *)
      {stemsfill, leavesfill, leafctsfill} = 
      	fillPosNegGap[stemsneg, stemsnonneg, stemdivs, stemcts];
      (* convert first part of stems to strings, replacing 0 with -0 for negative values *)
      stemsneg[[All, 1]] = Map[If[# == 0, "-0", ToString[#]] &, stemsneg[[All, 1]]];
      stemsnonneg[[All, 1]] = Map[ToString, stemsnonneg[[All, 1]]];
      {stems, leaves, leafcts} = {Join[stemsneg, stemsfill, stemsnonneg], 
      	Join[leavesneg, leavesfill, leavesnonneg], 
      	If[stemcts, Join[leafctsneg, leafctsfill, leafctsnonneg], False]}
      ,
      (* if IncludeEmptyStems->False, just convert first parts of stems to strings, 
         replacing 0 with -0 for negative 0 stems *)
      stemsneg[[All, 1]] = Map[If[# == 0, "-0", ToString[#]] &, stemsneg[[
          All, 1]]];
      stemsnonneg[[All, 1]] = Map[ToString, stemsnonneg[[All, 1]]];
      {stems, leaves, leafcts} = {Join[stemsneg, stemsnonneg], Join[leavesneg, leavesnonneg], 
      	If[stemcts, Join[leafctsneg, leafctsnonneg], False]}
      ];
    (* add division labels to stems if necessary; the last parts of stems are elements of 
       Range[0, stemdivs-1] corresponding to the stemdivs divisions in each stem *)
    stems = If[stemlabs === None,
        stems[[All, 1]],
        stems[[All, -1]] = 
        	stems[[All, -1]] /. Thread[Rule[Range[stemdivs] - 1, Map[ToString, stemlabs]]];
        stems = MapThread[StringJoin, Transpose[stems]]];
    (* if rounding is not used, it is possible to get leaves with one more digit 
	   than requested. for instance, with one leaf digit and "RoundLeaves"->False,
	   9.99 would be a stem of 9 and leaf 10 because the "RoundLeaves" only 
	   determines whether rounding is used to match stems and leaves; it doesn't 
	   change the display of leaves *)
    If[Not[rounddig]&&includeleaves&&Not[tallies]&&Not[FreeQ[leaves,10^leafdigs|-10^leafdigs]],
    		Message[StemLeafPlot::rndlvs,leafdigs];
    		(* use "X" as tally mark placeholder to get spacing right;
    		   spaceLeaves will replace the "X"s with talsymb *)
    		leaves=leaves/.{(10^leafdigs|-10^leafdigs)->
    			Apply[StringJoin,Table["X",{leafdigs}]]}];
    (* space the leaves if necessary *)
    leaves = If[includeleaves, 
    	Map[With[{spl = StringSplit[#, "\n"]}, 
    		  Switch[Length[spl], 1, spl[[1]], 0, "", _, Grid[Transpose[{spl}]]]] &,
    		  spaceLeaves[leafdigs, leafspace, Abs[leaves]/.Abs[xx_String]->xx, leafwrap, tallies, talsymb]], None];
    With[{colalign=(ColumnAlignments/.{opts}/.{ColumnAlignments -> Left})},
      leaves=leaves/.Grid->(Grid[#,
    	ColumnAlignments->If[Length[colalign]>1&&stemcts,colalign[[2]],colalign],
    	RowAlignments->(RowAlignments/.{opts}/.{RowAlignments ->Top})]&)];
    (* construct the columns of the plot *)
    res = Transpose[Which[stemcts && includeleaves, {stems, leaves, leafcts},
          stemcts, {stems, leafcts},
          includeleaves, {stems, leaves},
          True, {stems}]];
    (* create the formatted plot *)
    res = Apply[Grid[Join[{collabels}, res], ##]&,Join[FilterRules[Flatten[{opts}],Options[GridBox]], 
    	{ColumnLines -> {True, False}, ColumnSpacings -> 1.5, 
    	(* by default, center stems and counts if leaves are not present *)
    	ColumnAlignments -> If[includeleaves, {Right, Left, Center}, Center], 
    	RowLines -> {True, False},RowAlignments->Top}]];
    (* add stem units if necessary *)
    If[TrueQ[units], 
    	Grid[{{res}, {"Stem units: " <> ToString[StandardForm[10^stemlevel]]}}, 
              ColumnAlignments -> Left, RowSpacings -> {2}], 
        res]]
              

(* function for constructing side-by-side stem-and-leaf plot *)              
iSLPlot2[data1_, data2_, opts___] := 
  Block[{stemlevel, stemdivs, stems, leaves1, leafcts1, leaves2, leafcts2, dataneg1, 
      datanonneg1, stemsneg1, leavesneg1, leafctsneg1, stemsnonneg1, leavesnonneg1, 
      leafctsnonneg1, dataneg2, datanonneg2, stemsneg2, leavesneg2, leafctsneg2, 
      stemsnonneg2, leavesnonneg2, leafctsnonneg2, stemsfill, leavesfill, leafctsfill, 
      stemsneg, stemsnonneg, tabentries, emptystems, collabels, leafdigs, rounddig, 
      leafspace, units, stemcts, stemlabs, signs1, signs2, tallies, talsymb, 
      includeleaves, leafwrap, res},
    {stemlevel, stemdivs, stemlabs, emptystems, collabels, leafdigs, rounddig, leafspace, 
     leafwrap, tallies, talsymb, includeleaves, units, stemcts} = slOptions[Join[data1, data2], 2, opts];
    (* split data1 into negative and non-negative parts and compute the stems and leaves *)
    {dataneg1, datanonneg1} = negativeSplit[data1];
    {stemsneg1, leavesneg1, leafctsneg1} = 
    	getStemsLeaves[dataneg1, leafdigs, stemdivs, stemlevel, stemcts, rounddig, True];
    {stemsnonneg1, leavesnonneg1, leafctsnonneg1} = 
    	getStemsLeaves[datanonneg1, leafdigs, stemdivs, stemlevel, stemcts, rounddig];
    (* split data1 into negative and non-negative parts and compute the stems and leaves *)
    {dataneg2, datanonneg2} = negativeSplit[data2];
    {stemsneg2, leavesneg2, leafctsneg2} = 
    	getStemsLeaves[dataneg2, leafdigs, stemdivs, stemlevel, stemcts, rounddig, True];
    {stemsnonneg2, leavesnonneg2, leafctsnonneg2} = 
    	getStemsLeaves[datanonneg2, leafdigs, stemdivs, stemlevel, stemcts, rounddig];
    (* splitting data into negative and non-negative parts can result in separation of 
       stems with 0 leaves into their own category; need to check for and fix this if it occurs *)
    If[stemdivs>1&&Length[stemsneg1]>0,
    	{stemsneg1, leavesneg1, leafctsneg1}=
    		Fold[fixNegativeZero[#1, #2, stemdivs,Left] &, 
    			{stemsneg1, leavesneg1, leafctsneg1}, 
    			Reverse[Flatten[
  				  With[{pos = Position[stemsneg1, {_, 0}]},
           			Extract[pos, Position[Map[Union, Extract[leavesneg1, pos]], {0}]]]]]]
      ];
    If[stemdivs>1&&Length[stemsneg2]>0,
    	{stemsneg2, leavesneg2, leafctsneg2}=
    		Fold[fixNegativeZero[#1, #2, stemdivs] &, 
    			{stemsneg2, leavesneg2, leafctsneg2}, 
    			Reverse[Flatten[
  				  With[{pos = Position[stemsneg2, {_, 0}]},
           			Extract[pos, Position[Map[Union, Extract[leavesneg2, pos]], {0}]]]]]]
      ];    
    (* if IncludeStemCounts->True add {} leaves and 0 stem counts in cases where there 
       are leaves for only one of the data sets; otherwise just add the {} leaves *)
    If[stemcts,
      If[Not[stemsneg1 === {} === stemsneg2],
        {stemsneg1, leavesneg1, leafctsneg1} = 
        	Transpose[Sort[Join[Transpose[{stemsneg1, leavesneg1, leafctsneg1}], 
        		Map[{#, {}, 0} &, Complement[stemsneg2, stemsneg1]]]]];
        {stemsneg2, leavesneg2, leafctsneg2} = 
        	Transpose[Sort[Join[Transpose[{stemsneg2, leavesneg2, leafctsneg2}], 
        		Map[{#, {}, 0} &, Complement[stemsneg1, stemsneg2]]]]]];
      If[Not[stemsnonneg1 === {} === stemsnonneg2],
        {stemsnonneg1, leavesnonneg1, leafctsnonneg1} = 
        	Transpose[Sort[Join[Transpose[{stemsnonneg1, leavesnonneg1, leafctsnonneg1}], 
        		Map[{#, {}, 0} &, Complement[stemsnonneg2, stemsnonneg1]]]]];
        {stemsnonneg2, leavesnonneg2, leafctsnonneg2} = 
        	Transpose[Sort[Join[Transpose[{stemsnonneg2, leavesnonneg2, leafctsnonneg2}], 
        		Map[{#, {}, 0} &, Complement[stemsnonneg1, stemsnonneg2]]]]]];
      ,
      If[Not[stemsneg1 === {} === stemsneg2],
        {stemsneg1, leavesneg1} = Transpose[Sort[Join[Transpose[{stemsneg1, leavesneg1}], 
        	Map[{#, {}} &, Complement[stemsneg2, stemsneg1]]]]];
        {stemsneg2, leavesneg2} = Transpose[Sort[Join[Transpose[{stemsneg2, leavesneg2}], 
        	Map[{#, {}} &, Complement[stemsneg1, stemsneg2]]]]]];
      If[Not[stemsnonneg1 === {} === stemsnonneg2],
        {stemsnonneg1, leavesnonneg1} = 
        	Transpose[Sort[Join[Transpose[{stemsnonneg1, leavesnonneg1}], 
        		Map[{#, {}} &, Complement[stemsnonneg2, stemsnonneg1]]]]];
        {stemsnonneg2, leavesnonneg2} = 
        	Transpose[Sort[Join[Transpose[{stemsnonneg2, leavesnonneg2}], 
        		Map[{#, {}} &, Complement[stemsnonneg1, stemsnonneg2]]]]]];];
    (* add empty stems if necessary *)
    If[TrueQ[emptystems],
      If[Length[stemsneg1]>0,
      	{leafctsneg1, leavesneg1, stemsneg, leavesneg2, leafctsneg2} =
        	fillEmptySidebySide[leavesneg1, stemsneg1, leavesneg2, stemdivs, stemcts, 
        		leafctsneg1, leafctsneg2],
       	stemsneg={}];
      If[Length[stemsnonneg1]>0,
      	{leafctsnonneg1, leavesnonneg1, stemsnonneg, leavesnonneg2, leafctsnonneg2} =
        	fillEmptySidebySide[leavesnonneg1, stemsnonneg1, leavesnonneg2, stemdivs, stemcts, 
        		leafctsnonneg1, leafctsnonneg2],
        stemsnonneg={}];
      (* fill gaps between positive and negative stems if necessary *)
      {stemsfill, leavesfill, leafctsfill} = 
      	fillPosNegGap[stemsneg, stemsnonneg, stemdivs, stemcts];
      (* convert first parts of stems to strings, replacing negative 0s with -0 *)
      stemsneg[[All, 1]] = Map[If[# == 0, "-0", ToString[#]] &, stemsneg[[All, 1]]];
      stemsnonneg[[All, 1]] = Map[ToString, stemsnonneg[[All, 1]]];
      {stems, leaves1, leaves2, leafcts1, leafcts2} = {
      	Join[stemsneg, stemsfill, stemsnonneg], 
      	Join[leavesneg1, leavesfill, leavesnonneg1],
        Join[leavesneg2, leavesfill, leavesnonneg2], 
        If[stemcts, Join[leafctsneg1, leafctsfill, leafctsnonneg1], False], 
        If[stemcts, Join[leafctsneg2, leafctsfill, leafctsnonneg2], False]},
      stemsneg1[[All, 1]] = Map[If[# == 0, "-0", ToString[#]] &, stemsneg1[[All, 1]]];
      stemsneg2[[All, 1]] = Map[If[# == 0, "-0", ToString[#]] &, stemsneg2[[All, 1]]];
      stemsnonneg1[[All, 1]] = Map[ToString, stemsnonneg1[[All, 1]]];
      stemsnonneg2[[All, 1]] = Map[ToString, stemsnonneg2[[All, 1]]];
      {stemsneg, stemsnonneg} = {stemsneg1, stemsnonneg1};
      {stems, leaves1, leaves2, leafcts1, leafcts2} = {Join[stemsneg, stemsnonneg], 
      	Join[leavesneg1, leavesnonneg1],
        Join[leavesneg2, leavesnonneg2], 
        If[stemcts, Join[leafctsneg1, leafctsnonneg1], False], 
        If[stemcts, Join[leafctsneg2, leafctsnonneg2], False]}
      ];
    (* add stem labels if necessary *)
    stems = If[stemlabs === None,
        stems[[All, 1]],
        stems[[All, -1]] = 
        	stems[[All, -1]] /. Thread[Rule[Range[stemdivs] - 1, Map[ToString, stemlabs]]];
        stems = MapThread[StringJoin, Transpose[stems]]];
    (* if rounding is not used, it is possible to get leaves with one more digit 
	   than requested. for instance, with one leaf digit and "RoundLeaves"->False,
	   9.99 would be a stem of 9 and leaf 10 because the "RoundLeaves" only 
	   determines whether rounding is used to match stems and leaves; it doesn't 
	   change the display of leaves *)
	If[Not[rounddig]&&includeleaves&&Not[tallies]&&Not[FreeQ[{leaves1,leaves2},10^leafdigs|-10^leafdigs]],
    		Message[StemLeafPlot::rndlvs,leafdigs];
    		{leaves1,leaves2}={leaves1,leaves2}/.{(10^leafdigs|-10^leafdigs)->
    			Apply[StringJoin,Table["X",{leafdigs}]]}];
    (* space and wrap leaves *)
    leaves1 = If[includeleaves, 
    		Map[With[{spl = StringSplit[#, "\n"]}, 
    		  Switch[Length[spl], 1, spl[[1]], 0, "", _, Grid[Transpose[{spl}]]]] &,
    		    spaceLeaves[leafdigs, leafspace, Abs[leaves1]/.Abs[xx_String]->xx, leafwrap, tallies, talsymb, Left]], 
    		None];
    leaves2 = If[includeleaves, 
    		Map[With[{spl = StringSplit[#, "\n"]}, 
    		  Switch[Length[spl], 1, spl[[1]], 0, "", _, Grid[Transpose[{spl}]]]] &,
    		    spaceLeaves[leafdigs, leafspace, Abs[leaves2]/.Abs[xx_String]->xx, leafwrap, tallies, talsymb]], None];
    With[{colalign1=(ColumnAlignments/.{opts}/.{ColumnAlignments -> Right}),
    	  colalign2=(ColumnAlignments/.{opts}/.{ColumnAlignments -> Left})},
      leaves1=leaves1/.Grid->(Grid[#,
    	ColumnAlignments->If[Length[colalign1]>1&&stemcts,colalign1[[2]],colalign1],
    	RowAlignments->(RowAlignments/.{opts}/.{RowAlignments ->Top})]&);
      leaves2=leaves2/.Grid->(Grid[#,
    	ColumnAlignments->If[Length[colalign2]>1,colalign2[[Max[If[stemcts,4,3],Length[colalign2]]]],colalign2],
    	RowAlignments->(RowAlignments/.{opts}/.{RowAlignments ->Top})]&)];
    (* construct the plot columns *)
    tabentries = Join[{collabels}, 
    	Which[stemcts&&includeleaves, Transpose[{leafcts1, leaves1, stems, leaves2, leafcts2}], 
    		stemcts, Transpose[{leafcts1, stems, leafcts2}],
    		includeleaves, Transpose[{leaves1, stems, leaves2}],
    		True, Transpose[{stems}]]];
   	(* construct the formatted plot *)
    res = Apply[Grid[tabentries, ##]&,Join[FilterRules[Flatten[{opts}],Options[GridBox]], 
    	(* by default, only add column lines between stems and leaves *)
    	{ColumnLines -> Which[stemcts&&includeleaves, {False, True, True, False}, 
    		stemcts, False,
    		True, True], 
    	ColumnSpacings -> 1.5, 
    	(* by default center stems and counts if no leaves are present *)
    	ColumnAlignments -> Which[
    		stemcts&&includeleaves, {Center, Right, Center, Left, Center}, 
    		includeleaves, {Right, Center, Left},
    		True, Center], 
    		RowLines -> {True, False},RowAlignments->Top}]];
    (* add stem units if necessary *)
    If[TrueQ[units], Grid[{{res}, {"Stem units: " <> ToString[StandardForm[10^stemlevel]]}}, 
    	ColumnAlignments ->Left, RowSpacings -> {2}], res]]
    	
(* the following cases catch bad input and issue error messages *)
    	
(* data is not a vector of real values *)    	
StemLeafPlot[data_, opts___?OptionQ] := 
  Module[{result = iSLPlotfail[data, opts]}, result /; (result =!= $Failed)]

iSLPlotfail[data_, opts___] := (Message[StemLeafPlot::ndata, 1]; $Failed)


(* data1 or data2 is not a vector of real values, or opts are not options *)
StemLeafPlot[data1_, data2_, opts___] := 
  Module[{result = iSLPlotfail2[data1, data2, opts]}, result /; (result =!= $Failed)]

iSLPlotfail2[data1_, data2_, opts___] := (Which[
      Not[VectorQ[data1, realQ]], Message[StemLeafPlot::ndata, 1],
      Not[VectorQ[data2, realQ]], Message[StemLeafPlot::ndata, 2],
      Not[VectorQ[{opts}, OptionQ]], 
      	Message[StemLeafPlot::nonopt, Select[{opts},Not[OptionQ[#]]&][[1]], 2, 
		HoldForm[StemLeafPlot[data1, data2, opts]]],
      True, Null]; $Failed)


(* case of no arguments *)
StemLeafPlot[] := Module[{result = iSLPlotfailnoargs[]}, 
	result /; (result =!= $Failed)]

iSLPlotfailnoargs[] := (Message[StemLeafPlot::argt, StemLeafPlot, 0, 1, 2]; $Failed)


End[]

EndPackage[]
