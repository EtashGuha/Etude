(* 
 *
 *	ImageRestyle
 *
 *  Top-level interface
 *
 *	Created by Matteo Salvarezza
 *
 *)

Options[ImageRestyle] = {
	PerformanceGoal -> "Speed",
	TargetDevice -> "CPU",
	PreserveColor -> False,
	RandomSeeding -> 1234
}//Sort;

$ImageRestyleHiddenOptions = {
 	"LuminanceTransfer" -> True, (* method for PreserveColor, False uses color stats matching *)
 	"Iterations" -> 1000, (* options for PerformanceGoal -> Quality *)
	"StepSize" -> 0.04	
}

DefineFunction[ImageRestyle, iImageRestyle, 2, 
	"ExtraOptions" -> $ImageRestyleHiddenOptions
];

iImageRestyle[args_, opts_] := BlockRandom[iiImageRestyle[args, opts]]

iiImageRestyle[args_, opts_] := Scope[

	{content, styleBlend} = parseInput[args];

	(* Not supported on Raspberry Pi due to memory limitations 
		(Fast method only works with very small images, slow method always crashes) 
	*)
	If[$SystemID == "Linux-ARM", ThrowFailure["piuns"]];

	Switch[GetOption[RandomSeeding],
		Automatic, SeedRandom[],
		Inherited, Null,
		_, SeedRandom[GetOption[RandomSeeding]]
	];

	normStyleBlend = Thread[ N@Normalize[styleBlend[[All,1]], Total] -> styleBlend[[All,2]] ];
	styleWeight = Max[styleBlend[[All,1]]];

	If[GetOption[PreserveColor] && !GetOption["LuminanceTransfer"], 
		colorMatchedStyles = colorMatch[content, #]& /@ normStyleBlend[[All, 1]];
		normStyleBlend = Thread[colorMatchedStyles -> normStyleBlend[[All, 2]]];
	];
	result = Switch[GetOption[PerformanceGoal],
		"Speed",
			fastStyleTransfer[content, normStyleBlend, styleWeight],
		"Quality",
			slowStyleTransfer[content, normStyleBlend, styleWeight],
		_,
		DBPrint["Post-validation of PerformanceGoal failed."];
		ThrowFailure["perfgoal"];
	];
	If[GetOption[PreserveColor] && GetOption["LuminanceTransfer"],
		result = colorRestore[result, content]
	];

	Image`CopyOptions[content, result, "Type" -> ImageType[content]]
]

parseInput[args_List] := Scope[

	(* Argument parsing *)
	content = args[[1]];
	content = If[Image`Utilities`SetImage2D[content, content],
		ColorConvert[RemoveAlphaChannel@content, "RGB"],
		ThrowFailure["imginv", content]
	];

	styleBlend = args[[2]];
	styleBlend = Switch[styleBlend,
		img_ /; Image`Utilities`SetImage2D[styleImg, img],
			styleImg = ColorConvert[RemoveAlphaChannel@styleImg, "RGB"];
			{0.5 -> styleImg},
		Rule[weight_, img_] /; Image`Utilities`SetImage2D[styleImg, img],
			styleImg = ColorConvert[RemoveAlphaChannel@styleImg, "RGB"];
			{N@styleBlend[[1]] -> styleImg},
		{imgs__} /; Image`Utilities`SetImageList2D[styleImgs, {imgs}],
			styleImgs = ColorConvert[RemoveAlphaChannel@#, "RGB"]& /@ styleImgs;
			Thread[0.5 -> styleImgs],
		{rules__Rule} /; Image`Utilities`SetImageList2D[styleImgs, {rules}[[All,2]]],
			styleImgs = ColorConvert[RemoveAlphaChannel@#, "RGB"]& /@ styleImgs;
			Thread[N@styleBlend[[All,1]] -> styleImgs],
		_,
			ThrowFailure["styleinv"]
	];
	weights = styleBlend[[All, 1]];
	If[!(VectorQ[weights, Internal`RealValuedNumericQ]) || !(And@@Thread[0 <= weights <= 1]),
		ThrowFailure["styleinv"]
	];

	(* Options checking *)
	If[!MatchQ[GetOption[PerformanceGoal], "Speed" | "Quality"],
		ThrowFailure["perfg", GetOption[PerformanceGoal]]
	];
	If[!TestTargetDevice[GetOption[TargetDevice], ImageRestyle],
		ThrowFailure[]
	];
	If[!MatchQ[GetOption[PreserveColor], True | False],
		ThrowFailure["opttf", "PreserveColor", GetOption[PreserveColor]];
	];
	If[!MatchQ[GetOption[RandomSeeding], _?IntegerQ | _?StringQ | Automatic | Inherited],
		Message[ImageRestyle::seeding, GetOption[RandomSeeding], ImageRestyle];
		GetOption[RandomSeeding] = Automatic;
	];

	{content, styleBlend}
]

colorMatch[content_, style_] := Scope[
	(* 1 -> content, 2 -> style *)
	data = Flatten[ImageData[#], {{1,2},{3}}]& /@ {content, style};	

	mean = Mean /@ data;
	cov  = Covariance /@ data;	

	svd = SingularValueDecomposition /@ cov;
	covSqrt = #[[1]] . Sqrt[ #[[2]] ] . #[[3]]& /@ svd; 

	weights = covSqrt[[1]].Inverse[covSqrt[[2]]];
	bias = mean[[1]] - weights.mean[[2]];

	ImageApply[ weights.# + bias&, style ]
]

colorRestore[result_, content_] := Scope[

	toYIQ = {{0.299,0.587,0.114},{0.596,-0.274,-0.322},{0.211,-0.523,0.312}};
	toRGB = {{1,0.956,0.621},{1,-0.272,-0.647},{1,-1.106,1.703}};

	rYIQ = ImageData[result] . Transpose[toYIQ];
	cYIQ = ImageData[content] . Transpose[toYIQ];

	cYIQ[[All, All, 1]] = rYIQ[[All, All, 1]];
	out = cYIQ . Transpose[toRGB];
	Image[out]
]
