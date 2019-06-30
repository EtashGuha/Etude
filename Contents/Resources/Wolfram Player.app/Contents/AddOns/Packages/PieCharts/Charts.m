Message[General::obspkg, "PieCharts`"]

BeginPackage["PieCharts`"]

Options[PieCharts`PieChart] =
Sort[
    {PieLabels -> Automatic,
    PieStyle -> Automatic,
    PieEdgeStyle -> Automatic,
    PieExploded -> None,
    PieOrientation -> Automatic
    } ~Join~ Developer`GraphicsOptions[]
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

(* Pie Chart *)


PieChart::badexplode =
"The PieExploded option was given an invalid value ``. PieExploded takes \
a list of distances or a list of {wedgenumber, distance} pairs.";

PieChart::"pchornt" = "Value of option PieOrientation -> `` should be an angle, a direction, or a pair of an angle and a direction.";
(* The following line is for compatability purposes only... *)

PieCharts`PieChart[list:{{_?((numberQ[#] && NonNegative[N[#]])&), _}..}, opts___?OptionQ] :=
    PieCharts`PieChart[First[Transpose[list]],
        PieLabels->Last[Transpose[list]],opts]

PieCharts`PieChart[list:{_?((numberQ[#] && NonNegative[N[#]])&) ..}, opts___?OptionQ]/;
        (!(And @@ (# == 0 & /@ list))) :=
    Module[ {labels, styles, linestyle, tlist, thalf, text,offsets,halfpos,
                len = Length[list],exploded,wedges,angles1,angles2,lines,
                tmp, gopts, orientation, angle=0, ccw=True},
    (* Get options *)
        {labels, styles, linestyle,exploded, orientation} =
            {PieLabels, PieStyle, PieEdgeStyle,PieExploded, PieOrientation}/.
            Flatten[{opts, Options[PieCharts`PieChart]}];
        gopts = FilterRules[{opts, Options[PieCharts`PieChart]}, Options[Graphics]];
    (* Error handling on options, set defaults *)
        If[Head[labels] =!= List || Length[labels] === 0,
            If[labels =!= None, labels = Range[len]],
            labels = CycleValues[labels, len]
        ];
        If[Head[styles] =!= List || Length[styles] === 0,
           styles = Table[Hue[FractionalPart[0.67 + 2.0 (i - 1)/GoldenRatio], 0.35, 0.75], {i, 1, len}],
           styles = CycleValues[styles, len]
        ];
        If[linestyle === Automatic, linestyle = GrayLevel[0]];
        linestyle = CycleValues[linestyle, len];
        If[MatchQ[exploded,{_Integer,_Real}],exploded = {exploded}];
        If[exploded === None, exploded = {}];
        If[exploded === All,
            exploded = Range[len]];
        If[(tmp = DeleteCases[exploded,
                (_Integer | {_Integer,_?(NumberQ[N[#]]&)})]) =!= {},
            Message[PieChart::"badexplode",tmp];
            exploded = Cases[exploded,
                (_Integer | {_Integer,_?(NumberQ[N[#]]&)})]
        ];
        exploded = Map[If[IntegerQ[#], {#,.1},#]&,exploded];
        offsets = Map[If[(tmp = Cases[exploded,{#,_}]) =!= {},
                Last[First[tmp]],
                0]&,
            Range[len]
        ];

		Switch[N[orientation],
			Automatic, {angle, ccw} = {0,"CounterClockwise"},
			_Real, {angle, ccw} = {orientation, "CounterClockwise"},
			"Clockwise"|"CounterClockwise", {angle, ccw} = {0, orientation},
			{__Real, "Clockwise"|"CounterClockwise"}, {angle, ccw} = orientation,
			{"Clockwise"|"CounterClockwise",_Real}, {ccw,angle} = orientation,
			_, 
				Message[PieChart::"pchornt", orientation];
				{angle, ccw} = {0,"CounterClockwise"}
			];
		ccw = If[ccw == "CounterClockwise", 1, -1];

    (* Get range of values, set up list of thetas *)
        tlist = N[ 2Pi+ angle + ccw 2 Pi FoldList[Plus,0,list]/(Plus @@ list)];
    (* Get pairs of angles *)
        angles1 = Drop[tlist,-1];angles2 = Drop[tlist,1];
    (* bisect pairs (for text placement and offsets) *)
        thalf = 1/2 (angles1 + angles2);
        halfpos = Map[{Cos[#],Sin[#]}&,thalf];
    (* generate lines, text, and wedges *)
        text = If[labels =!= None,
            MapThread[Text[#3,(#1 + .6) #2]&,
                    {offsets,halfpos,labels}],
                {}];
        wedges = MapThread[
                Flatten[{#5, EdgeForm[#6], Disk[#1 #2, 1, {#3,#4}]}]&,
            {offsets,halfpos,angles1,angles2,styles, linestyle}];
     	wedges = wedges /. Disk[{0,0},1,a_]:>Disk[{0,0},1,Sort[a]];
    (* show it all... *)
        Show[Graphics[
            {wedges,
            text},
            gopts]]
    ]


 
 End[]
 
 EndPackage[]
