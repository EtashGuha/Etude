(* ::Package:: *)

(* :Name: FunctionApproximations`OrderStar` *)

(*
   :Title: Order Stars For Approximants To Functions.
*)

(* :Author: Mark Sofroniou *)

(* :Summary:
This package plots the order star of an approximating function,
to an essentially analytic function.  It is common to consider
rational approximants to functions such as Pade approximants.
Various information about a numerical scheme (such as order and stability)
may be ascertained from its order star. For example, Runge-Kutta methods
may be considered as rational approximants to the exponential,
where relative and absolute stability regions are considered in
terms of the linear scalar test problem of Dahlquist.
The zeros, poles and interpolation points convey important additional
information and may also be displayed.
*)

(* :Context: FunctionApproximations` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1993-2007, Wolfram Research, Inc.
*)

(* :History:
 Original Version by Mark Sofroniou, January, 1993.
 Updated with suggestions from Jerry Keiper, December 1993.
 Revised for release, July 1995.
*)

(* :Keywords:
 Numerical Integration, Runge-Kutta, ODE, Order Star, Stability.
*)

(* :Source:
 Mark Sofroniou, Ph.D. Thesis (1994), Loughborough University,
 Loughborough, Leicestershire LE11 3TU, England.

 For a comprehensive study see Order Stars, A. Iserles &
 S. P. Norsett, Chapman & Hall, 1991.
*)

(* :Mathematica Version: 6.0 *)

(* :Limitations:
 The package relies on the code for ContourPlot to draw
 the order star. ContourPlot may not produce a smooth contour
 unless the options MaxRecursion or PlotPoints are set to a
 sufficiently high number.
 The values for the poles and zeros of the function and
 approximant and the interpolation points are found using
 NSolve and may not always give full solutions (or indeed any
 at all). Therefore, values may be specified using options.
*)

If[Not@ValueQ[OrderStarPlot::usage],OrderStarPlot::usage = "OrderStarPlot[r, f] plots the order star \
of the approximating function r to the function f. \
OrderStarPlot[r, f, var] may be used to specify the variable explicitly."];

If[Not@ValueQ[OrderStarInterpolation::usage],OrderStarInterpolation::usage = "OrderStarInterpolation is an \
option to OrderStarPlot specifying whether interpolation points of \
an approximant to a function should be displayed. \
OrderStarInterpolation may evaluate to True, False or a list \
of {x,y} coordinates (useful if NSolve fails to detect solutions)."];

If[Not@ValueQ[OrderStarKind::usage],OrderStarKind::usage = "OrderStarKind is an option to \
OrderStarPlot specifying the type of order star to be displayed. \
OrderStarKind may be set to First or to Second. \
Order stars of the first kind trace out the level curve Abs[R/F]==1. \
Order stars of the second kind trace out the level curve Re[R-F]==0."];

If[Not@ValueQ[OrderStarLegend::usage],OrderStarLegend::usage = "OrderStarLegend is an option to OrderStarPlot \
specifying whether (or where) to display the legend of symbols \
used to represent zeros, poles and interpolation points. \
OrderStarLegend may evaluate to True, False or {{xmin,ymin},{xmax,ymax}} \
where the x,y values are scaled coordinates ranging from 0 to 1."];

If[Not@ValueQ[OrderStarPoles::usage],OrderStarPoles::usage = "OrderStarPoles is an option to OrderStarPlot \
specifying whether poles of an approximant and a function should be \
displayed. OrderStarPoles may evaluate to any pair consisting of \
True, False or a list of {x,y} coordinates (useful if NSolve fails \
to detect solutions)."];

If[Not@ValueQ[OrderStarZeros::usage],OrderStarZeros::usage = "OrderStarZeros is an option to OrderStarPlot \
specifying whether zeros of an approximant and a function should be \
displayed. OrderStarZeros may evaluate to any pair consisting of \
True, False or a list of {x,y} coordinates (useful if NSolve fails \
to detect solutions)."];

If[Not@ValueQ[OrderStarSymbolSize::usage],OrderStarSymbolSize::usage = "OrderStarSymbolSize is an option \
of OrderStarPlot specifying the size of the symbols used to represent \
poles, zeros and interpolation points."];

If[Not@ValueQ[OrderStarSymbolThickness::usage],OrderStarSymbolThickness::usage = "OrderStarSymbolThickness is \
an option of OrderStarPlot specifying the thickness of the outline \
of the symbols used to represent poles and zeros."];

Unprotect[OrderStarPlot, OrderStarInterpolation, OrderStarKind,
OrderStarLegend, OrderStarPoles, OrderStarSymbolSize,
OrderStarSymbolThickness, OrderStarZeros];


(* Set default options. *)

Options[OrderStarPlot] =
  {Axes -> True, AxesOrigin -> {0, 0}, ColorFunction -> Function[GrayLevel[1-#/2]],
   FrameTicks -> None, ClippingStyle -> Automatic, MaxRecursion-> 5, PlotPoints -> 100,
   PlotRange -> Automatic, Ticks -> None, AspectRatio -> Automatic,
   OrderStarInterpolation -> False, OrderStarKind -> First, OrderStarLegend -> False,
   OrderStarPoles -> {True, True}, OrderStarZeros -> {True, True},
   OrderStarSymbolSize -> 0.05, OrderStarSymbolThickness -> 0.05
  };


Begin["`Private`"]

(* Generic Error message. *)

OrderStarPlot::opts = "The option `1` in OrderStarPlot did not evaluate to `2`";

TFLQ[opt_] := (opt === True || opt === False || ListQ[opt]);
TFLString = "a pair consisting of True, False, or a List of {x,y} coordinates.";

ScaledNumberQ[x_]:= (NumberQ[x] && TrueQ[0 <= x <= 1]);

LegendCoordsQ[{{xmin_, ymin_},{xmax_, ymax_}}]:=
  ScaledNumberQ[xmin] && ScaledNumberQ[ymin] && ScaledNumberQ[xmax] && ScaledNumberQ[ymax];
LegendCoordsQ[___]:= False;

opttest[bool_, mess_] := 
	If[bool, True, Message[OrderStarPlot::opts, Apply[Sequence, mess]]; False];

optmessages = {
	{"OrderStarInterpolation", TFLString},
	{"OrderStarKind", "First or Second."},
	{"OrderStarLegend", "True, False, or a list of scaled coordinates {{xmin,ymin},{xmax,ymax}}."},
	{"PlotRange", "Automatic, or a list {{xmin,xmax},{ymin,ymax}}."},
	{"ClippingStyle", "None or Automatic."},
	{"MaxRecursion", "a non-negative machine integer."},
	{"PlotPoints", "an integer>=2 or a list of two such integers."},
	{"OrderStarSymbolSize", "a positive number."},
	{"OrderStarSymbolThickness", "a positive number."},
	{"OrderStarZeros", TFLString},
	{"OrderStarPoles", TFLString}
  };

OptionTest[opts___]:= 
  Module[{datatypes, optlist},
    optlist = {
      OrderStarInterpolation, OrderStarKind, OrderStarLegend,
      PlotRange, ClippingStyle, MaxRecursion, PlotPoints, OrderStarSymbolSize,
      OrderStarSymbolThickness, OrderStarZeros, OrderStarPoles} /.
        {opts} /. Options[OrderStarPlot];
    If[Head[optlist[[2]]] === Symbol, optlist[[2]] = SymbolName[optlist[[2]]]];
    datatypes = {
      TFLQ[Part[optlist, 1]], (* interpolation *)
      MemberQ[{"First", "Second"}, Part[optlist, 2]], (* kind *)
      TFLQ[Part[optlist, 3]] && (* legendcoords *)
        If[ListQ[Part[optlist, 3]],
          LegendCoordsQ[Part[optlist, 3]],
	      True],
      MatchQ[ Map[Union,N[Part[optlist, 4]]], (* plotrange *)
        Automatic|{{_?NumberQ,_?NumberQ},{_?NumberQ,_?NumberQ}}],
      MatchQ[Part[optlist, 5],None|Automatic], (* clipstyle *)
      MatchQ[Part[optlist, 6],_?Internal`NonNegativeMachineIntegerQ], (* maxrec *)
      MatchQ[Part[optlist, 7],_?(Internal`PositiveMachineIntegerQ[#] && (# > 1)&) | (* plotpoints *)
        {_?(Internal`PositiveMachineIntegerQ[#] && (# > 1)&),_?(Internal`PositiveMachineIntegerQ[#] && (# > 1)&)}],
      TrueQ[Positive[Part[optlist, 8]]], (* symbolsize *)
      TrueQ[Positive[Part[optlist, 9]]], (* symbolthickness *)
      MatchQ[Part[optlist, 10],{_?TFLQ,_?TFLQ}], (* zeros *)
      MatchQ[Part[optlist, 11],{_?TFLQ,_?TFLQ}] (* poles *)
    };
    If[Apply[And, MapThread[opttest,{datatypes,optmessages}]], optlist, $Failed]
  ]; (* End of OptionTest. *)


(* Valid variables are non-numeric symbols and integer indexed functions (arrays). *)

OrderStarPlot::var = "The expressions `1` and `2` are not univariate functions \
of the same variable.";

varQ[_?NumericQ] = False;
varQ[_Symbol] = True;
varQ[_[(_Integer)..]] = True;
varQ[_] = False;

findz[f_, g_] :=
  Module[{nf},
    nf =
      Select[
        Union[ Join[ Level[{f,g},{-1}], Level[{f,g},{-2}] ] ],
        varQ
      ];
    If[Length[nf] == 1,
      Part[nf, 1],
      Message[OrderStarPlot::var, f, g]; $Failed
    ]
  ];


(* makeCP[ ] draws the main plot and sub plots *)

makeCP[f_, {plotrange_, clipstyle_, maxrec_, plotpoints_}, plotoptions___] :=
  Module[{x, y, func},
    func[n_?NumberQ] := f[n];
    ContourPlot[func[x + I y],
      Evaluate[Prepend[Part[plotrange, 1], x]],
      Evaluate[Prepend[Part[plotrange, 2], y]],
      DisplayFunction :> Identity, 
      Evaluate[ClippingStyle -> clipstyle], 
      Evaluate[MaxRecursion -> maxrec],
      Evaluate[PlotPoints -> plotpoints],
      Evaluate[plotoptions]
    ]
  ];


(* automaticPR[ ] extracts a plot range from the list of zeros and poles
 using a scaling factor and an offset. The range of the plot aspect ratio
 is also restricted. *)

automaticPR[points_List] :=
  Module[{armax, scale, offset, xdiff, xmid, xmin, xmax, xrange,
      ydiff, ymid, ymin, ymax, yrange},
    {armax,scale,offset} = {2.5, 1.3, 1.};
    {xrange,yrange} = Thread[points];
    xmin = Min[xrange]; xmax = Max[xrange];
    ymin = Min[yrange]; ymax = Max[yrange];
    xmid = (xmin+xmax)/2.; ymid = (ymin+ymax)/2.;
    xdiff = Abs[xmax-xmin]/2.; ydiff = Abs[ymax-ymin]/2.;
    xdiff = Max[offset + scale xdiff, ydiff/armax];
    ydiff = Max[offset + scale ydiff, xdiff/armax];
    {xmid + {-xdiff, xdiff}, ymid + {-ydiff, ydiff}}
  ];

OrderStarPlot::cvar = "The variables `1` and `2` in the expressions \
`3` and `4` are not the same.";

OrderStarPlot[R_, F_, opts___?OptionQ] :=
  Module[{ans, var = findz[R,F]},
    ans /; (var =!= $Failed &&
            (ans = OrderStarPlot[R, F, var, opts];
             True))
  ];

OrderStarPlot[R_, F_, var:(_Symbol | _Symbol[(_?IntegerQ)..]), opts___?OptionQ] :=
  Module[{ans},
    ans /; ((ans = OptionTest[opts]) =!= $Failed &&
            (ans = mainOrderStar[R, F, var, ans, opts];
             True))
  ];

(* Start of main routine. *)

mainOrderStar[R_, F_, var_, optslist_List, opts___?OptionQ] :=
  Module[{aspect, clipstyle, comp, contour, funct, glfontinfo, groptions, interpolation,
      kind, legendcoords, maxrec, orderstar, plotinterp, plotlegend, plotpolesf, plotpolesr,
      plotzerosf, plotzerosr, plotoptions, plotrange, polesf, polesr, ppoints,
      range, symb, symbdata, symbsize, symbthick, zerosf, zerosr},

    {interpolation, kind, legendcoords, range, clipstyle, maxrec, ppoints, symbsize,
     symbthick, {zerosr, zerosf}, {polesr, polesf}} = optslist;

    plotoptions = Flatten[{opts,Contours->{contour}, Options[OrderStarPlot]}];
    plotoptions = DeleteCases[plotoptions, (MaxRecursion | PlotRange | PlotPoints) -> _, 1];
    plotoptions = Sequence @@ FilterRules[plotoptions, Options[ContourPlot]];
    {aspect,comp} = {AspectRatio, Compiled} /. {plotoptions} /. Options[ContourPlot];

    groptions = Sequence @@ FilterRules[{plotoptions}, Options[Graphics]];

(* Get any font information for the legend window. The head of the pattern used
 * could be either Rule or RuleDelayed *)

	glfontinfo =
		Apply[
			Sequence,
			Cases[{groptions}, _[DefaultFont,_] | _[FormatType,_] | _[TextStyle,_]]
		];

(* Order star of first or second kind as a function of a symbol. *)

    funct =
      If[kind==="First",
        contour = 1; Abs[R/F],
        contour = 0; Re[R-F]
      ] /. var->symb;

(* Compiled function definition for efficient graphics rendering. *)

    funct =
      If[TrueQ[comp],
        Compile[Evaluate[{{symb,_Complex}}], Evaluate[funct], Evaluate[{{_, _Complex}}] ],
        Function[Evaluate[{symb}, funct] ]
      ];

(* Used to calculate automatic plot range. *)

    plotinterp = findsolution[Numerator[R] - F Denominator[R], var,
			"interpolation points", "approximant", interpolation];
    plotpolesf = findsolution[1/F, var, "poles", "function", polesf];
    plotpolesr = findsolution[Denominator[R], var, "poles", "approximant", polesr];
    plotzerosf = findsolution[F, var, "zeros", "function", zerosf];
    plotzerosr = findsolution[Numerator[R], var, "zeros", "approximant", zerosr];

(* Calculate plot range. *)

    plotrange = Union[plotinterp, plotpolesf, plotpolesr, plotzerosf,
			plotzerosr, {{0, 0}}, SameTest -> Equal];

    plotrange = 
      If[SameQ[range, Automatic],
        If[plotrange == {{0, 0}},
          {{-10, 10}, {-10, 10}},
        (* else *)
          automaticPR[plotrange]
        ],
      (* else *)
        range
      ];

    aspect =
      Abs[N[#]]& @
        If[aspect === Automatic,
          Apply[Subtract, Part[plotrange, 2]]/Apply[Subtract, Part[plotrange, 1]],
        (* else *)
          aspect
        ];

(* Symbol style data *)

    symbdata = {symbsize, symbthick, aspect};

    plotinterp = makeshape[plotinterp, symbdata, interpsymbol];
    plotpolesf = makeshape[plotpolesf, symbdata, polefsymbol];
    plotpolesr = makeshape[plotpolesr, symbdata, polersymbol];
    plotzerosf = makeshape[plotzerosf, symbdata, zerofsymbol];
    plotzerosr = makeshape[plotzerosr, symbdata, zerorsymbol];

(* Information window for the symbols used. *)

    plotlegend =
      If[legendcoords,
        legendwindow[{{0.01, 1 - 0.25/aspect}, {0.35, 1 - 0.01/aspect}},
          symbdata, glfontinfo ],
        {},
        legendwindow[legendcoords, symbdata, glfontinfo ]
      ];

(* Make the main plot. *)

    ppoints = Min[ Max[ 15, Round[.5 ppoints] ], 50 ];

    orderstar = makeCP[funct, {plotrange, clipstyle, maxrec, ppoints}, plotoptions];

(* graphics for the symbols *)

    symbdata = Graphics[
        {plotinterp, plotpolesf, plotpolesr, plotzerosf, plotzerosr}
    ];

(* Combine the Graphics rendering symbol outline after the
 symbol background shape.  *)

    Show[Flatten[{orderstar, symbdata, plotlegend}],
      PlotRange -> plotrange, groptions,
      Method -> {"AxesInFront" -> True},
      DisplayFunction:>$DisplayFunction ]

  ]; (* End of mainOrderStar. *)


(* Define functions used in OrderStarPlot. *)

(* Generate a list of solution points. *)

(* Avoid possible division by zero when poles not required. *)

SetAttributes[findsolution,HoldFirst];

findsolution[__,False] = {};

findsolution[eqn_, var_, info_, func_, True]:=
  extractsolutions[eqn,var,info,func];

findsolution[eqn_, var_, info_, func_, points_List]:=
  Union[
    Join[
      extractsolutions[eqn,var,info,func],
      SetPrecision[points,6] /. 0 -> 0.0
    ],
    SameTest -> Equal
  ];

(* Remove infinite solutions and generate solution messages. *)

OrderStarPlot::sols = "Warning: No `1` of `2` found using NSolve. Either inverse \
functions or transcendental dependencies were involved. Try specifying omitted \
points using options.";

(* No solutions *)

finitesolutions[{},_,_,False]:= {};

(* No solutions, but Solve used inverse fuctions etc *)

finitesolutions[{},info_,func_,True]:=
  (Message[OrderStarPlot::sols,info,func]; {});

(* Finite solutions *)

finitesolrules = {Complex[x_,y_]->{x,y}, x_?NumberQ->{x,0}};

finitesolutions[solutions_,info_,func_,False]:=
  Module[{fsols},
    fsols = DeleteCases[solutions,_DirectedInfinity];
    fsols /. finitesolrules
  ];

(* Generate a message if there were no finite solutions, but Solve
 used inverse fuctions etc *)

finitesolutions[solutions_,info_,func_,True]:=
  Module[{fsols},
    fsols = DeleteCases[solutions,_DirectedInfinity];
    If[fsols === {},
      Message[OrderStarPlot::sols,"finite "<>info,func]
    ];
    fsols /. finitesolrules
  ];

(* NSolve eqn in terms of var. Suppress Solve messages, but
 set a flag if messages were generated. *)

extractsolutions[eqn_,var_,info_,func_]:=
  Module[{sol, msgs=False},
    Block[{$Messages},
		Check[ sol = NSolve[eqn==0, var], msgs = True; sol]
    ];
    sol = If[MatchQ[sol,_NSolve|{}|{{}}], {}, var /. sol];
    SetPrecision[finitesolutions[sol,info,func,msgs], 6] /. 0 -> 0.0
  ];


(* General graphics primitive for the symbfuncs
	interpsymbol, polefsymbol, polersymbol, zerofsymbol,
	and zerorsymbol. *)

makeshape[{}, __] :=  {{},{}}

makeshape[coords_List, {s_, t_, r___}, symbfunc_] :=
    Map[Inset[symbfunc[t,r], #, Automatic, Scaled[{s, s}]]&, coords]

(* Graphics primitives for symbols with graphicsprims Line and Polygon. *)

interpsymbol[thick_, ___] :=
   Graphics[{EdgeForm[{Thickness[thick], GrayLevel[0]}],
            GrayLevel[1], Disk[{0, 0}]},
            PlotRange -> {{-1.05, 1.05},{-1.05, 1.05}}]

polefsymbol[thick_, ___] :=
    Graphics[{GrayLevel[0], Line[{{-1, -1},{1,1}}], Line[{{-1, 1}, {1, -1}}]},
             PlotRange -> {{-1.05, 1.05}, {-1.05, 1.05}}]

polersymbol[thick_, ___] :=
    Graphics[{EdgeForm[{Thickness[thick], GrayLevel[0]}], GrayLevel[1],
             Polygon[{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}]},
             PlotRange -> {{-1.05, 1.05}, {-1.05, 1.05}}]

zerofsymbol[thick_, ___] :=
    Graphics[{GrayLevel[0], Line[{{0, -1}, {0, 1}}], Line[{{-1, 0}, {1, 0}}]},
             PlotRange -> {{-1.05, 1.05}, {-1.05, 1.05}}] 

zerorsymbol[thick_, ___] :=
    Graphics[{EdgeForm[{Thickness[thick], GrayLevel[0]}], GrayLevel[1],
             Polygon[{{-0.7, -0.7}, {0.7, -0.7}, {0.7, 0.7}, {-0.7, 0.7}}]},
             PlotRange -> {{-1.05, 1.05}, {-1.05, 1.05}}]


(* Primitives for symbol information window. *)

showsymbols[sd:{size_, thick_, ___}, opts___]:=
      {
       makeshape[{{0.12, 0.9}},sd ,polersymbol ],
       Text[" Poles of approximant",{0.25, 0.9},{-1,0}],
       makeshape[{{0.12, 0.7}},sd ,zerorsymbol ],
       Text[" Zeros of approximant",{0.25, 0.7},{-1,0}],
       makeshape[{{0.12, 0.5}},sd ,polefsymbol],
       Text[" Poles of function",{0.25, 0.5},{-1,0}],
       makeshape[{{0.12, 0.3}},sd ,zerofsymbol],
       Text[" Zeros of function",{0.25, 0.3},{-1,0}],
       makeshape[{{0.12, 0.1}},sd ,interpsymbol],
       Text[" Interpolation points",{0.25, 0.1},{-1,0}]
      }

legendwindow[{pt1_List, pt2_List}, {size_, thick_, ar_}, opts___]:=
  Module[{mid},
      mid = (pt1 + pt2)/2;
    Graphics[{{Hue[0], PointSize[0.1], Point[Scaled[mid]]},Inset[
    Graphics[
      {{EdgeForm[{GrayLevel[0], Thickness[0.005]}],
       GrayLevel[1], Rectangle[{0,0}, {1,1}]},
        showsymbols[{0.16, thick}]},
     PlotRange -> {{-0.05, 1.05}, {-0.05, 1.05}},
     AspectRatio -> Full,
     opts
    ],
    Scaled[mid], Center, Scaled[pt2 - pt1]]}]
  ];

End[];    (* End `Private` Context. *)

(* Protect exported symbols. *)

SetAttributes[
{OrderStarPlot},
ReadProtected];

Protect[OrderStarPlot, OrderStarInterpolation, OrderStarKind,
OrderStarLegend, OrderStarPoles, OrderStarSymbolSize,
OrderStarSymbolThickness, OrderStarZeros];
