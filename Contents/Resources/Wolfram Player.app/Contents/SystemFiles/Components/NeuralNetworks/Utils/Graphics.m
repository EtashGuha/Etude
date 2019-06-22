Package["NeuralNetworks`"]


$activationIconRange = <|
	Sin -> {-Pi, Pi},
	Cos -> {-Pi, Pi},
	Tan -> {-Pi, Pi},
	Log -> {0.1, 2},
	Sqrt -> {0, 2},
	"SELU" -> {-5, 5},
	"SoftSign" -> {-5, 5},
	"Sigmoid" -> {-5, 5},
	LogisticSigmoid -> {-5, 5}
	(*
	"ReLU" -> {-2, 2},
	"ELU" -> {-2, 2},
	"SoftPlus" -> {-2, 2},
	"HardTanh" -> {-2, 2},
	"HardSigmoid" -> {-2,2},
	Tanh -> {-2, 2},
	Ramp -> {-2, 2},
	Exp -> {-2, 2},
	Abs -> {-2, 2},
	Round -> {-2, 2},
	Floor -> {-2, 2},
	Ceiling -> {-2, 2},
	Sign -> {-2, 2},
	Erf -> {-2, 2}
	*)
|>;

plotAct[act_, {x1_, x2_}] := Module[
	{xs, values, frame, size, plot},
	xs = Range[x1, x2, (x2 - x1)/50.];
	values = ScalarFunctionApply[act, xs];
	If[Apply[Equal, MinMax[values]], $Failed, (* Do not plot the act if the output is constant on the range (can happen with custom function on default range) *)
		values = Transpose[{xs, values}];
		(* Remove possible Infinity and NaN *)
		(* TODO values = Select[values, numericQ @* EchoFunction[# -> {Export["~/tmp.mx", Echo@ #];Abort[];numericQ[#], numericQ["NaN"], numericQ[NaN]}&] @* Last ]; *)
		frame = GrayLevel[0, 0.3];
		size = 20;
		plot = ListLinePlot[
			values,
			AxesLabel -> None, AxesOrigin -> {0, 0}, 
			PlotRange -> All, FrameStyle -> frame, AxesStyle -> frame,
			Frame -> False, FrameTicks -> None, ImageSize -> size, 
			Ticks -> None,
			ImagePadding -> {{0,1},{1,0}}, PlotRangePadding -> {0,{0.1,0.1}},
			PlotStyle -> Directive[RGBColor[0.4, 0.396, 0.659],AbsoluteThickness[2]],
			Background -> Transparent
		];
		plot = Rasterize[plot, "Image", ImageResolution -> 144, ImageSize -> 2*size, Background -> None];
		Image[plot, ImageSize -> size]
	]
];
numericQ[x_] := Internal`RealValuedNumericQ[x];

PackageScope["ActivationIcon"]

$activationIconCache = <||>;

ActivationIcon[act_] := 
	CacheTo[$activationIconCache, act,
		Quiet @ Check[
			plotAct[act, Lookup[$activationIconRange, act, {-2,2}]],
			$Failed
		]
	];