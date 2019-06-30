BeginPackage["ErrorBarPlots`"]

If[!ValueQ[ErrorBarFunction::usage], ErrorBarFunction::usage = "ErrorBarFunction is an option for ErrorListPlot that specifies a function to apply to determine the shape of error bars. "];
If[!ValueQ[ErrorBar::usage], ErrorBar::usage = "\!\(\*RowBox[{RowBox[{\"ErrorBar\", \"[\", \"{\"}], StyleBox[\"negerror\", \"TI\"], \",\", StyleBox[\"poserror\", \"TI\"], RowBox[{\"}\", \"]\"}]}]\) error in the positive and negative directions.\n\!\(\*RowBox[{\"ErrorBar\", \"[\", StyleBox[\"yerr\", \"TI\"], \"]\"}]\) error \!\(\*StyleBox[\"yerr\", \"TI\"]\) in both the positive and negative directions\n\!\(\*RowBox[{\"ErrorBar\", \"[\", RowBox[{StyleBox[\"xerr\", \"TI\"], \",\", StyleBox[\"yerr\", \"TI\"]}], \"]\"}]\) errors specified for both the x and the y coordinates"];
If[!ValueQ[ErrorBarPlot::usage], ErrorListPlot::usage = "\!\(\*RowBox[{\"ErrorListPlot\", \"[\", RowBox[{\"{\", RowBox[{RowBox[{\"{\", RowBox[{SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"1\", \"TR\"]], \",\", SubscriptBox[StyleBox[\"dy\", \"TI\"], StyleBox[\"1\", \"TR\"]]}], \"}\"}], \",\", RowBox[{\"{\", RowBox[{SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"2\", \"TR\"]], \",\", SubscriptBox[StyleBox[\"dy\", \"TI\"], StyleBox[\"2\", \"TR\"]]}], \"}\"}], \",\", StyleBox[\"\[Ellipsis]\", \"TI\"]}], \"}\"}], \"]\"}]\) plots points corresponding to a list of values \!\(\*RowBox[{SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"1\", \"TR\"]], \",\", \" \", SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"2\", \"TR\"]], \",\", \" \", StyleBox[\"\[Ellipsis]\", \"TR\"]}]\), with corresponding error bars. The errors have magnitudes \!\(\*RowBox[{SubscriptBox[StyleBox[\"dy\", \"TI\"], StyleBox[\"1\", \"TR\"]], \",\", SubscriptBox[StyleBox[\"dy\", \"TI\"], StyleBox[\"2\", \"TR\"]], \",\", StyleBox[\"\[Ellipsis]\", \"TR\"]}]\).\n\!\(\*RowBox[{\"ErrorListPlot\", \"[\", RowBox[{\"{\", RowBox[{RowBox[{\"{\", RowBox[{RowBox[{\"{\", RowBox[{SubscriptBox[StyleBox[\"x\", \"TI\"], StyleBox[\"1\", \"TR\"]], \",\", SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"1\", \"TR\"]]}], \"}\"}], \",\", RowBox[{\"ErrorBar\", \"[\", SubscriptBox[StyleBox[\"err\", \"TI\"], \"1\"], \"]\"}]}], \"}\"}], \",\", RowBox[{\"{\", RowBox[{RowBox[{\"{\", RowBox[{SubscriptBox[StyleBox[\"x\", \"TI\"], StyleBox[\"2\", \"TR\"]], \",\", SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"2\", \"TR\"]]}], \"}\"}], \",\", RowBox[{\"ErrorBar\", \"[\", SubscriptBox[StyleBox[\"err\", \"TI\"], StyleBox[\"2\", \"TR\"]], \"]\"}]}], \"}\"}], \",\", StyleBox[\"\[Ellipsis]\", \"TR\"]}], \"}\"}], \"]\"}]\) plots points with specified \!\(\*StyleBox[\"x\", \"TI\"]\) and \!\(\*StyleBox[\"y\", \"TI\"]\) coordinates and error magnitudes."];

Message[General::"obspkg", "ErrorBarPlots`"];

Begin["`Private`"]

(* ErrorListPlot *)
Options[ErrorListPlot] = Append[
	Complement[
		Options[ListLinePlot],
		{MaxPlotPoints->Infinity}
		],
	ErrorBarFunction -> Automatic
	];


ErrorListPlot[dat_, opts:OptionsPattern[]] :=
	Block[{newdat, error, newopts, ebfunc, range},

		{ebfunc, range} = OptionValue[{ErrorBarFunction, DataRange}];
	
		If[ebfunc === Automatic, ebfunc = ebarfun];
	
		If[range === All, 
			newdat = dat /. {a_/;VectorQ[a, (NumericQ[#]||Head[#]===PlusMinus||Head[#]===ErrorBar)&] :>
				Transpose[{Range[Length[a]], a}]},
			newdat = dat /. {	
				a_/;VectorQ[a, MatchQ[#1, {_?NumericQ, _?NumericQ}] &] :>
					MapIndexed[Prepend[#,First[#2]]&, a],
				a_/;(VectorQ[a] && Length[a]>3) :> Transpose[{Range[Length[a]], a}]
				}	
			]; 
	
		newdat = newdat /. {
			{x_?NumericQ, y_?NumericQ, e_?NumericQ} :> (error[N[{x,y}]] = ErrorBar[{0,0},{-e,e}]; {x,y}),
			{{x_,y_}, e_ErrorBar} :> (error[N[{x,y}]] = makeError[e]; {x,y}),
			{x_/;Head[x]=!=List,y_/;Head[y]=!=List} :> handlePlusMinus[{x,y}]
			};
	
		newopts = FilterRules[{opts}, Options[ListLinePlot]];	
			
		p =	ListPlot[newdat, newopts, Method -> {"OptimizePlotMarkers" -> False}];

		(*Needed to handle points introduced by clipping when Joined->True*)
		error[_] := ErrorBar[{0,0},{0,0}];

		p[[1]] = p[[1]] /. {
			g_GraphicsComplex :> markErrors[g, ebfunc],
			l_Line :> markErrors[l,ebfunc],
			p_Point :> markErrors[p, ebfunc],
			i_Inset :> markErrors[i, ebfunc]
		};
		
		p
			
	]

makeError[ErrorBar[y_]] := ErrorBar[{0,0}, eb[y]]
makeError[ErrorBar[x_,y_]] := ErrorBar[eb[x],eb[y]]
eb[n_?Positive] := {-n,n}
eb[{n_?NumericQ, p_?NumericQ}] := {n,p}
eb[_]:={0,0}

handlePlusMinus[{x_?NumericQ, y_?NumericQ}] := 
	(error[N[{x,y}]] = ErrorBar[{0,0},{0,0}]; {x,y})
handlePlusMinus[{PlusMinus[x_,e_], y_?NumericQ}] := 
	(error[N[{x,y}]] = ErrorBar[{-e,e},{0,0}]; {x,y})
handlePlusMinus[{PlusMinus[x_,ex_], PlusMinus[y_,ey_]}] := 
	(error[N[{x,y}]] = ErrorBar[{-ex,ex},{-ey,ey}]; {x,y})
handlePlusMinus[{x_?NumericQ, PlusMinus[y_,ey_]}] := 
	(error[N[{x,y}]] = ErrorBar[{0,0},{-ey,ey}]; {x,y})
handlePlusMinus[a_] := a

markErrors[GraphicsComplex[pts_, prims_, opts___], ebfunc_] := 
	GraphicsComplex[pts, prims /. {
		Line[l:{__Integer}] :> {Line[l], ebfunc[pts[[#]], error[pts[[#]]]]& /@ l},
		Line[l:{{__Integer}..}] :> {Line[l], ebfunc[pts[[#]], error[pts[[#]]]]& /@ Flatten[l]},
		Point[l:{__Integer}] :> {Point[l], ebfunc[pts[[#]], error[pts[[#]]]]& /@ l},
		Point[l:{{__Integer}..}] :> {Point[l], ebfunc[pts[[#]], error[pts[[#]]]]& /@ Flatten[l]},
		(l:Inset[obj_, pos_, a___]) :> {l, ebfunc[pts[[pos]], error[pts[[pos]]]]}		
	}, opts]

markErrors[l_Line, ebfunc_] := 
	{l, ebfunc[#, error[#]]& /@ Cases[l, {_?NumericQ, _?NumericQ}, Infinity]}

markErrors[l_Point, ebfunc_] := 
	{l, ebfunc[#, error[#]]& /@ Cases[l, {_?NumericQ, _?NumericQ}, Infinity]}

markErrors[l:Inset[obj_, pos_, a___], ebfunc_] := 
	{l, ebfunc[pos, error[pos]]}

(* default error bar function *)
ebarfun[pt:{x_, y_}, ErrorBar[{xmin_, xmax_}, {ymin_, ymax_}]] :=
    Module[ {xline, yline},
        If[ xmin === 0 && xmax === 0,
            xline = {},
            xline = {Line[{{x + xmax, y}, {x + xmin, y}}],
                           Line[{Offset[{0,1.5}, {x + xmax, y}],
                               Offset[{0,-1.5}, {x + xmax, y}]}],
                           Line[{Offset[{0,1.5}, {x + xmin, y}],
                               Offset[{0,-1.5}, {x + xmin, y}]}]}
        ];
        If[ ymin === 0 && ymax === 0,
            yline = {},
            yline = {Line[{{x, y + ymax}, {x, y + ymin}}],
                           Line[{Offset[{1.5,0}, {x, y + ymax}],
                               Offset[{-1.5,0}, {x, y + ymax}]}],
                           Line[{Offset[{1.5,0}, {x, y + ymin}],
                               Offset[{-1.5,0}, {x, y + ymin}]}]}
        ];
        Join[xline, yline]
    ]

                
End[]

EndPackage[]
