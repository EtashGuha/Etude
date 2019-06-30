
(* :Title: Graphics Legends *)

(* :Author: John M. Novak *)

(* :Summary: 
This package provides functions for placing a legend box on a graphic.
It includes numerous options for specifying the legend and its
placement.
*)

(* :Context: Graphics`Legends` *)

(* :Package Version: 1.1 *)

(* :History:
    Version 1.0 by John M. Novak, February 1991
    Version 1.1 by John M. Novak, May 1999 -- Improved option handling
         a bit, allowing certain graphics options to be properly applied to
         the sub- or super- graphic as necessary.
    Version 1.2 by Mark Sofroniou, September 2007. Rewritten to consolidate
    common code and functionality. Extended to include additional Log
    variants of Plot and ListPlot functions. 
*)

(* :Keywords: graphics, legends, key *)

(* :Mathematica Version: 4.0 *)

(* :Copyright: Copyright 1991-2007, Wolfram Research, Inc.*)

(* :Warning: Adds rules to Plot[]. *)

(* :Limitation: This does not yet deal with scaled coordinates. *)

(* :Limitation: Automatic placing of legend boxes is not very
    good; tweaking by hand is likely to be required (and is definitely
    needed if there is more than one box being placed). *)

(* :Limitation: Graphics options affect the entire graphic (with
    legend box emplaced). Because the boxes generally contain
    graphics in rectangles, if AspectRatio is changed, unexpected
    changes may occur.  This applies even more so to Legend (as
    opposed to ShowLegend). *)

(* :Limitation: Error checking is somewhat limited at this time. *)

Message[General::obspkg, "PlotLegends`"]

BeginPackage["PlotLegends`"]

Legend::badsize =
"The LegendSize option must be a number or a pair of numbers greater than zero, or it must be Automatic. Using the default value instead.";

Begin["`Private`"]

Options[ShadowBox] = {
  ShadowBorder -> {Thickness[.001], GrayLevel[0]},
  ShadowForeground -> GrayLevel[1],
  ShadowBackground -> GrayLevel[0],
  ShadowOffset -> {.1,-.1}
  };

ShadowBoxOptionNames = Map[First, Options[ShadowBox]];

ShadowBox[pos:{_, _}, size:{_, _}, opts___] :=
  Module[{bordsty, foresty, backsty, offset, forebox, backbox, border},
    {bordsty, foresty, backsty, offset} =
      ShadowBoxOptionNames /. Flatten[{opts, Options[ShadowBox]}];
    If[foresty === Automatic, foresty = GrayLevel[1]];
    If[bordsty === Automatic, bordsty = {Thickness[.001], GrayLevel[0]}];
    If[backsty === Automatic, backsty = GrayLevel[0]];
    forebox = Rectangle[pos, pos + size];
    backbox = Rectangle[pos + offset, pos + size + offset];
    Flatten[{backsty, backbox, foresty, EdgeForm[bordsty], forebox}]
  ];

Options[Legend] = {
  LegendPosition -> {-1, -1},
  LegendSize -> Automatic,
  LegendShadow -> Automatic,
  LegendTextSpace -> Automatic,
  LegendTextDirection -> Automatic,
  LegendTextOffset -> Automatic,
  LegendLabel -> None,
  LegendLabelSpace -> Automatic,
  LegendOrientation -> Vertical,
  LegendSpacing -> Automatic,
  LegendBorder -> Automatic,
  LegendBorderSpace -> Automatic,
  LegendBackground -> Automatic
  };

LegendOptionNames = Map[First, Options[Legend]];

PlotLegendOptionNames = Union[{PlotLegend}, LegendOptionNames, ShadowBoxOptionNames];

Legend[fn:(_Function | _Symbol), boxes_?NumberQ, minstr_String:"",
    maxstr_String:"", opts___] :=
  Module[{its, strs},
    its = Map[fn, Range[0, 1, 1/(boxes - 1)]];
    strs = Table["", {Length[its]}];
    strs[[1]] = minstr;
    strs[[Length[strs]]] = maxstr;
    Legend[Transpose[{its, strs}], opts, LegendSpacing -> 0]
  ];

Legend[items:{{_, _}..}, opts___] :=
  Module[{ln = Length[items], boxes, lb, n, inc, rn, as, gr, sbox,
      pos, size, shadow, tspace, lspace, bspace, tdir, toff,
      label, orient, sopts, space, back, bord, pt, tmp},
    {pos, size, shadow, tspace, tdir, toff, label, lspace,
      orient, space, bord, bspace, back} =
        LegendOptionNames /. Flatten[{opts, Options[Legend]}];

    sopts = FilterRules[Flatten[{opts}], ShadowBoxOptionNames];

    If[Not[NumberQ[space]],
      inc = .08,
      inc = space
    ];
    If[tspace === Automatic,
      If[Count[Transpose[items][[2]], ""] === ln,
        tspace = 0,
        If[orient === Vertical,
          tspace = 2,
          tspace = 1
        ]
      ]
    ];
    If[lspace === Automatic,
      If[(label =!= None) && (label =!= ""),
        lspace = 1,
        lspace = 0
      ]
    ];
    If[bspace === Automatic, bspace = .1];
    If[toff === Automatic,
      If[orient === Vertical,
        toff = {-1, 0},
        toff = {0, -1}
      ]
    ];
    If[tdir === Automatic, tdir = {1, 0}];
    boxes =
      If[orient === Vertical,
        Table[
          pt = {inc, inc (2 n - 1) + (n - 1)};
          {rec[pt, {1, 1}, items[[ln - n + 1, 1]]],
           Text[items[[ln - n + 1, 2]], pt + {1 + inc + .05, 1/2}, toff, tdir]},
        {n, ln}],
        Table[
          pt = {inc (2 n - 1) + (n - 1), inc};
          {rec[pt, {1, 1}, items[[n, 1]]],
           Text[items[[n,2]], pt + {1/2, 1 + inc}, toff, tdir]},
        {n, ln}]
      ];
    lb =
      If[lspace != 0,
        Text[label,
          If[orient === Vertical,
            {(2 inc + 1 + tspace)/2, (2 inc + 1) ln + lspace/2},
            {(2 inc + 1) ln /2, 2 inc + 1 + tspace + lspace/2}
          ],
          {0, 0}
        ],
        {}
      ];
    rn =
      If[orient === Vertical,
        {{-bspace, 2 inc + 1 + tspace + bspace},
          {-bspace, (2 inc + 1) ln + lspace + bspace}},
        {{-bspace, (2 inc + 1) ln + bspace},
          {-bspace, 2 inc + 1 + tspace + lspace + bspace}}
      ];
    If[Min[size] <= 0 || !MatchQ[size, {_?NumericQ, _?NumericQ}],
      If[Not[NumberQ[size]],
        If[size =!= Automatic, Message[Legend::badsize]];
        size = .8
      ];
      tmp = Map[#[[2]] - #[[1]] &, rn];
      size = tmp (size/Max[tmp])
    ];
    as = size[[2]]/size[[1]];
    gr = Graphics[{boxes, lb}, AspectRatio -> as, PlotRange -> rn,
         Apply[Sequence, FilterRules[Flatten[{opts}], Options[Graphics]] ] ];
    If[bord === False, bord = None];
    If[shadow === False || shadow === None, shadow = {0,0}];
    If[shadow === Automatic, shadow = {.05,-.05}];
    sbox = ShadowBox[pos, size, Apply[Sequence, sopts], ShadowForeground -> back,
        	 ShadowBorder -> bord, ShadowOffset -> shadow, opts];
    GraphicsGroup[Flatten[{sbox, Inset[gr, pos, {Left, Bottom}, size]}]]
  ];
            

rec[start:{_,_}, size:{_,_}, style_] :=
  If[MemberQ[{RGBColor, Hue, CMYKColor, GrayLevel, Directive}, Head[style]],
    {style, Rectangle[start, start + size]},
    Inset[style, start, {Left, Bottom}, size]
  ];

ShowLegend[agr_, largs:({__}..), opts___?OptionQ] :=
  Module[{as, ls = {largs}, rec, ap, bubbleupopts, bubbledownopts},
    (* options that 'bubble up' from the central graphic to the
         containing graphic *)
    bubbleupopts =
      Quiet[AbsoluteOptions[agr, {ImageSize, Background, ColorOutput}]];
    as = Quiet[AspectRatio /. AbsoluteOptions[agr, AspectRatio]];
    (* options that 'bubble down' from ShowLegend to the subgraphics --
       doesn't override if they are explicitly set in the subgraphics,
       but does override defaults for the subgraphics *)
    bubbledownopts = FilterRules[Flatten[{opts}], {FormatType}];
    (* aspect ratio of input graphic is used to compute the default position
       of the legend keys and the size of the containing rectangle *)         
    If[!NumberQ[as], as = 1];
    If[as > 1,
      rec = Inset[Append[agr, bubbledownopts], {-1/as, -1}, {Left, Bottom}, {2/as, 2}];
      ap = {-1/as - .2, -1.2},
      rec = Inset[Append[agr, bubbledownopts], {-1, -as}, {Left, Bottom}, {2, 2as}];
      ap = {-1.2, -as - .2}
    ];

    ls =
      Apply[
        Legend[##, LegendPosition -> ap, Apply[Sequence, bubbledownopts]]&,
        ls,
        {1}
      ];
    Show[
      Graphics[{rec, ls}, Apply[Sequence, FilterRules[Flatten[{opts}], Graphics]],
        Apply[Sequence, bubbleupopts], AspectRatio -> Automatic, PlotRange -> All]
    ]
  ];

(* update syntax coloring templates to allow the PlotLegend options *)

UpdateSyntax[fun_]:=
  If[FreeQ[SyntaxInformation[fun], "OptionNames"],
    SyntaxInformation[fun] =
      Join[
        SyntaxInformation[fun],
        {"OptionNames" -> Map[ToString, Union[Map[First, Options[fun]], PlotLegendOptionNames]]}
      ],
    SyntaxInformation[fun] =
      Join[
        DeleteCases[SyntaxInformation[fun], "OptionNames"->_],
        {"OptionNames" -> Prepend[Map[ToString, Union[{CoordinatesToolOptions}, PlotLegendOptionNames]],Automatic]}
      ]
  ];
  
ProcessPlotStyles[n_, styles_]:=
  Module[{defaultcolors, ps = styles},
    defaultcolors =
      Table[
        Hue[FractionalPart[0.67 + 2.0 (i - 1)/GoldenRatio], 0.6, 0.6],
      {i, 1, n}];
    If[ps === Automatic, ps = defaultcolors];
    If[Head[ps] =!= List, ps = {ps},
    If[Length[ps] === 0, ps = {{}}]];
    ps = ps /. Dashing[x_] :> Dashing[scale[x]]; (* scale dashes *)
    If[Length[ps] =!= n, ps = PadRight[ps, n, ps]];
    Transpose[{defaultcolors, ps}]
  ];

scale[x_] := (x /. {a_?NumericQ :> 2/0.3a})

ProcessText[n_, lg_]:=
  Module[{txt},
    txt = lg;
    If[Head[txt] =!= List, txt = {txt}];
    PadRight[txt, n, ""]
  ];

ProcessOptions[plotfn_, popts__]:=
  Module[{fnopts, fngopts, gopts, lopts, opts, sopts},
    opts = Flatten[{popts}];
    fnopts = Options[plotfn];
    gopts = FilterRules[opts, Map[First, fnopts]];
    fngopts = Flatten[{gopts, fnopts}];
    sopts = FilterRules[opts, ShadowBoxOptionNames];
    lopts = FilterRules[opts, LegendOptionNames];
    {fngopts, gopts, lopts, sopts}
  ];

(**** Plot functions ****)

plotnames = {Plot, LogPlot, LogLinearPlot, LogLogPlot, PolarPlot, ParametricPlot};

Unprotect[Evaluate[plotnames]];

Map[
  (* Only insert a rule once if the package is loaded multiple times *)
  If[FreeQ[DownValues[#], PlotLegend],
    DownValues[#] =
      Prepend[
        DownValues[#, Sort->False],
        HoldPattern[#[a : PatternSequence[___, Except[_?BoxForm`HeldOptionQ]] | PatternSequence[], opts__?BoxForm`HeldOptionQ]]:>
          legendPlot[#, a, PlotLegend /. Flatten[{opts}], opts] /; !FreeQ[Flatten[{opts}], PlotLegend]
      ]
  ]&,
  plotnames
];

SetAttributes[legendPlot, HoldAll];

legendPlot[plotfn_, fn_, r_, None, opts__] :=
  plotfn[fn, r, Evaluate[FilterRules[Flatten[{opts}], Map[First, Options[plotfn]]]]];

legendPlot[plotfn_, fn_, r_, lg_, opts__] :=
  Module[{bubbledownopts, fngopts, gopts, lopts, sopts, disp, ln, gr, tb, ps, txt},
    {fngopts, gopts, lopts, sopts} = ProcessOptions[plotfn, opts];

    {ps} = {PlotStyle} /. fngopts;
    disp = First[FilterRules[fngopts, DisplayFunction]];

    ln = If[Head[Unevaluated[fn]] === List, Total[Dimensions[Unevaluated[fn]]], 1];

    txt = ProcessText[ln, lg];
    ps = ProcessPlotStyles[ln, ps];

    tb =
      Table[
        {Graphics[Flatten[{ps[[n]], Line[{{0, 0}, {1, 0}}]}]], txt[[n]]},
      {n, ln}];

    gr = Insert[ plotfn[fn, r, DisplayFunction->Identity, Evaluate[gopts]], disp, 2];

    bubbledownopts = Quiet[AbsoluteOptions[gr, {FormatType}]];

    ShowLegend[gr, {tb, sopts, lopts}, disp, Apply[Sequence, bubbledownopts]]
  ];

Map[UpdateSyntax, plotnames];
  
Protect[Evaluate[plotnames]];

(* Utilities for ListPlot *)

stripAnnotations[d_ /; VectorQ[d, NumericQ]]:= d;

stripAnnotations[d_] := 
  d //. {
         Annotation[a_, ___]:>a,
         Button[a_, ___]:>a,
         EventHandler[a_, ___]:>a,
         Hyperlink[a_, ___]:>a,
         Mouseover[a_, ___]:>a,
         PopupWindow[a_, ___]:>a,
         StatusArea[a_, ___]:>a,
         Style[a_, ___]:>a,
         Tooltip[a_, ___]:>a
        };

(* following utility functions used for date list validation *)
realQ[x_] := TrueQ[(NumberQ[#] && FreeQ[#, Complex]) &[N[x]]];

realorMissingQ[x_] := TrueQ[Head[x] === Missing || realQ[x]];

datelistQ[x_] := 
 VectorQ[x, realQ] && 0 < Length[x] <= 6 && 
  If[Length[x] > 1, VectorQ[x[[{1, 2}]], IntegerQ], IntegerQ[x[[1]]]];
  

datasetLength[l_?(MatrixQ[#] && MatchQ[Dimensions[#], {_,2}]&), All] := Length[l];

datasetLength[l_?(MatrixQ[#] && MatchQ[Dimensions[#], {_,2}]&), _] := 1;

datasetLength[l_?VectorQ, _] := 1;

(* next case catches date plotting data specified as {datelist_i, y_i} *)  
datasetLength[l_?(VectorQ[#, MatchQ[#, {_?datelistQ, _?realorMissingQ}] &] &), _] := 1;

datasetLength[l_List, mode_] := 
  Total[Map[datasetLength[#,mode]&, l]];

(**** ListPlot functions ****)

listplotnames = {ListPlot, ListLinePlot, ListLogPlot, ListLogLinearPlot, ListLogLogPlot, DateListPlot, DateListLogPlot, ListPolarPlot};

Unprotect[Evaluate[listplotnames]];

Map[
  (* Only insert a rule once if the package is loaded multiple times *)
  If[FreeQ[DownValues[#], PlotLegend],
    DownValues[#] =
      Prepend[
        DownValues[#, Sort->False],
        HoldPattern[#[a : PatternSequence[___, Except[_?OptionQ]] | PatternSequence[], opts__?OptionQ]]:>
          legendListPlot[#, a, PlotLegend /. Flatten[{opts}], opts] /; !FreeQ[Flatten[{opts}], PlotLegend]
      ]
  ]&,
  listplotnames
];

legendListPlot[plotfn_, d_, None, opts___]:=
  plotfn[d, FilterRules[Flatten[{opts}], Map[First, Options[plotfn]]]];

legendListPlot[plotfn_, d_, lg_, opts___] :=
  Module[{bubbledownopts, fngopts, gopts, lopts, sopts, disp, ln, gr, tb,
      pj, pm, ps, txt},
    {fngopts, gopts, lopts, sopts} = ProcessOptions[plotfn, opts];
    {ps, pm, pj} = {PlotStyle, PlotMarkers, Joined} /. fngopts;
    disp = First[FilterRules[fngopts, DisplayFunction]];

    datarange = DataRange /. fngopts;
    ln = datasetLength[stripAnnotations[d], datarange];
    txt = ProcessText[ln, lg];
    ps = ProcessPlotStyles[ln, ps];

    If[pm === Automatic, pm = Graphics`PlotMarkers[]];
    If[MatchQ[pm, {Automatic, _}], pm = {#,Last[pm]}& /@ (Graphics`PlotMarkers[][[All,1]])];
    If[Head[pm] =!= List, pm = {pm}];
	If[MatchQ[pm, {_, (_?NumberQ)}], pm = Table[pm, {ln}]];
	If[MatchQ[pm, {_, Tiny|Small|Medium|Large}], pm = Table[pm, {ln}]];
    If[Length[pm] < ln, pm = PadRight[pm, ln, pm]];
					

    If[Head[pj] =!= List, pj = {pj}];
    If[Length[pj] < ln, pj = PadRight[pj, ln, pj]];

    tb =
      Table[{Graphics[{
        If[pj[[n]],
          Flatten[{ps[[n]], Line[{{0,0.5},{1,0.5}}]}], {}],
          Switch[pm[[n]],
            _Graphics3D, 
            	Inset[pm[[n]],{0.5,0.5}],
            {_Graphics3D, _}, 
            	Inset[pm[[n,1]],{0.5,0.5},Center,5pm[[n,2]]],
            _Graphics, 
            	Inset[Graphics[{ps[[n]],First[pm[[n]]]},Options[pm[[n]]]], {0.5,0.5}],
            {_Graphics, _}, 
            	Inset[Graphics[{ps[[n,1]],First[First[pm[[n]]]]},Options[pm[[n,1]]]], {0.5,0.5},Center,5pm[[n,2]]],
            None|False, 
            	If[pj[[n]], {}, Flatten[{ps[[n]],Point[{0.5,0.5}]}]],
			{_String, _}, 
				Inset[Style[pm[[n,1]], ps[[n]], FontSize->pm[[n,2]]], {0.5,0.5}],
			{_,_}, 
				Inset[Style[pm[[n,1]], ps[[n]]], Center, Center,pm[[n,2]]],
            _, 
            	Inset[Style[pm[[n]], ps[[n]]], {0.5,0.5}]
          ]
        },
        PlotRange->{{0,1},{0,1}}],
        txt[[n]]},
      {n, ln}];

    gr = Insert[plotfn[d, DisplayFunction->Identity, Evaluate[gopts]], disp, 2];

    bubbledownopts = Quiet[AbsoluteOptions[gr, {FormatType}]];

    ShowLegend[gr, {tb, sopts, lopts}, disp, Apply[Sequence, bubbledownopts]]
  ];

(* update syntax coloring templates to allow the PlotLegend options *)

Map[UpdateSyntax, listplotnames];
  
Protect[Evaluate[listplotnames]];

End[]

EndPackage[]

