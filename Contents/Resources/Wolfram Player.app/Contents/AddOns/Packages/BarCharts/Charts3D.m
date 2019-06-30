BeginPackage["BarCharts`"]

(* BarChart3D *)



BarChart3D::bspzo = "BarSpacing -> `` must be a number between 0 and 1, or a pair of such numbers.";

Options[BarCharts`BarChart3D] =  Sort[{BarSpacing -> 0, BarEdges->True,
	BarEdgeStyle->GrayLevel[0], BarStyle -> GrayLevel[1]} ~Join~
	Options[Graphics3D]];

SetOptions[BarCharts`BarChart3D, PlotRange -> All, BoxRatios -> {1,1,1},
	Axes -> Automatic, Ticks -> Automatic, Lighting->Automatic]


Options[GeneralizedBarChart3D] =
  Sort[{
   BarEdges -> True,
   BarEdgeStyle -> Black,
   BarStyle -> White
  }~Join~Options[Graphics3D]];

SetOptions[GeneralizedBarChart3D, PlotRange -> All, BoxRatios -> {1,1,1},
	Axes -> Automatic, Ticks -> Automatic, Lighting->Automatic]




Begin["`Private`"]

BarCharts`BarChart3D[list:{{_?numberQ..}..}, opts___] :=
	BarCharts`BarChart3D[Flatten[Table[{{x,y,list[[x,y]]},
				  BarStyle/.{opts}/.Options[BarCharts`BarChart3D]},
				 {x,Length[list]},
				 {y,Length[Transpose[list]]}
				],1],opts]

BarCharts`BarChart3D[list:{{{_?numberQ,_}..}..}, opts___] :=
	BarCharts`BarChart3D[Flatten[Table[{{x,y,list[[x,y,1]]},
				  list[[x,y,2]]},
				 {x,Length[list]},
				 {y,Length[Transpose[list]]}
				],1],opts]

BarCharts`BarChart3D[list:{{{_?numberQ,_?numberQ,_?numberQ},_}...},opts___] :=
  Module[{x,y,xs,ys,xspacing,yspacing,boxopts,g3dopts,list1},

	xspacing = BarSpacing /. {opts} /. Options[BarCharts`BarChart3D];
	
	Which[
		NumberQ[xspacing] && 0<=xspacing<=1, 
			yspacing = xspacing,
		Head[xspacing] === List && Length[xspacing] === 2,
			{xspacing, yspacing} = xspacing,
		True, 
			Message[BarChart3D::bspzo, xspacing];
			{xspacing, yspacing} = {0,0}
		];
	If[xspacing>1 || xspacing<0 || yspacing>1 || yspacing<0,
		Message[BarChart3D::bspzo];
		{xspacing, yspacing} = {0,0}
		];

	If[TrueQ[BarEdges/.{opts}/.Options[BarCharts`BarChart3D]],
	   edges = EdgeForm[BarEdgeStyle/.{opts}/.Options[BarCharts`BarChart3D]],
	   edges = EdgeForm[]];
  	g3dopts = FilterRules[Flatten[{opts, Options[BarCharts`BarChart3D]}], Options[Graphics3D]];
  	xs = (1-xspacing)/2;
  	ys = (1-yspacing)/2;
  	list1 = Transpose[Map[#[[1]]&,list]];
        Show[
   		Graphics3D[Map[Flatten[{#[[2]],edges,
   				Cuboid[{#[[1,1]]-xs, #[[1,2]]-ys, 0},
				   	{#[[1,1]]+xs, #[[1,2]]+ys, #[[1,3]]}]
				      }]&,
			    list]],
		   Flatten[{g3dopts}]
	]
  ]                                        


(* GeneralizedBarChart3D *)

(* NOTE that BarSpacing is NOT an option of
	Graphics`Graphics`GeneralizedBarChart,
	so BarSpacing is also NOT an option of
	Graphics`Graphics3D`GeneralizedBarChart3D. *)
(* NOTE that the data in "list" are of the form...
	{  {{xpos1, ypos1}, height1, {xwidth1, ywidth1}},
	   {{xpos2, ypos2}, height2, {xwidth2, ywidth2}}, ...}
*)
GeneralizedBarChart3D[
	list:{ {{_?numberQ,_?numberQ}, _?numberQ, {_?numberQ,_?numberQ}}... },
	opts___] :=
  Module[{edges, g3dopts, barstyle},
   (
    If[TrueQ[BarEdges/.Flatten[{opts, Options[GeneralizedBarChart3D]}]],
	    edges = EdgeForm[BarEdgeStyle/.Flatten[{opts, Options[GeneralizedBarChart3D]}]],
	    edges = EdgeForm[]
    ];
    barstyle = BarStyle/.Flatten[{opts, Options[GeneralizedBarChart3D]}];
    g3dopts = FilterRules[Flatten[{opts, Options[GeneralizedBarChart3D]}], Options[Graphics3D]];
    Show[
   	Graphics3D[{If[ListQ[barstyle], Sequence @@ barstyle, barstyle], Map[Flatten[{edges,
   		Cuboid[{#[[1,1]]-#[[3, 1]]/2, #[[1,2]]-#[[3, 2]]/2, 0},
		       {#[[1,1]]+#[[3, 1]]/2, #[[1,2]]+#[[3, 2]]/2, #[[2]]}]
		      }]&,
		   list]}],
	Flatten[Join[
	   {g3dopts},
	   {Axes -> Automatic, BoxRatios -> {1,1,1}, PlotRange -> All} 
		]  ]
    ]
   ) 
  ] (* end GeneralizedBarChart3D *)                                        


End[]

EndPackage[]
