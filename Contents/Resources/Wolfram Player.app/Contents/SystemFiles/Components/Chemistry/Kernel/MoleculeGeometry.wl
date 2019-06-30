


Begin["Chemistry`Private`MoleculeGeometryDump`"]


RectangleBinPack::usage = "RectangleBinPack[ boundingBox, {rect_1, rect_2,..}] attempts to arrange the rect_i inside the bounds of boundingBox."



optionValue[opt_, pattern_, head_, opts_] := Replace[
  OptionValue[head, opts, opt] ,
  Except[pattern] -> OptionValue[head, Options[head], opt],
  Heads -> False]

$embedMolOptionsPattern = $chemLibraryFunctions["embedMol", "Parameters"] // 
	Normal // Map[ HoldPattern] //Apply[Alternatives];


messageBadOption[ x_ ] := Message[ MoleculeModify::method, x];


generateDefault3DCoordinates[imol_?ManagedLibraryExpressionQ, coords_?MatrixQ] := (
	imol[ "addCoordinates", QuantityMagnitude @ coords];
	has3DCoordinates[imol]
) /; Dimensions[ coords] === {imol["atomCount", True], 3}

$DefaultGraphEmbeddingMethod = "SpringEmbedding";
generateDefault3DCoordinates[imol_?ManagedLibraryExpressionQ /; TrueQ[imol["atomCount",True] > 0], atomCoordinateOption_ ] := Module[
	{params, succeeded, init},

	params = getEmbedParameters[ atomCoordinateOption];
	
	succeeded = Internal`NonNegativeIntegerQ @ embedMol[imol, params];
	
	If[!succeeded && tryGraphEmbedding3D[imol, $DefaultGraphEmbeddingMethod] =!= $Failed
		,
		Replace[
			tryGraphEmbedding3D[imol, $DefaultGraphEmbeddingMethod],
			$Failed :> (Message[Molecule::nocoord];Return[False, Module])
		];
		init = MatchQ[
			Quiet @ imol[ "initializeMMFF"],
			Except[ _LibraryFunctionError]
		];
		If[init
			,
			imol["minimize",200];
			,
			Message[Molecule::nocoord];
			imol["clearConformers"]
		];
		imol["clearMMFF"];
	];
	
	has3DCoordinates[imol]

];

generateDefault3DCoordinates[___] := False


$largestSeedValue = 2^32/2 - 1;
getEmbedParameters[ Automatic, args___] := getEmbedParameters[ {}, args]
getEmbedParameters[ opts_, issueMessagesAs_Symbol:Molecule ] := Module[
	{seed, method, methodOptions, options},
	
	{method,methodOptions} = Replace[
		opts,
		{
			x_String :> {x,{}},
			Automatic | {} :> {"ETKDGv2",{}},
			rule:{___Rule} :> {
				Lookup[rule, Method, "ETKDGv2"],
				rule
			},
			{x_String, rule___?OptionQ} :> {x,{rule}},
			_  :> {"ETKDGv2",{}}
		}
	];
	
	methodOptions = DeleteCases[methodOptions,HoldPattern[Method -> _]] /. {RandomSeeding -> "RandomSeeding", MaxIterations -> "MaxIterations"};
	
	seed = Replace[
		Lookup[ methodOptions, "RandomSeeding", OptionValue[ Chemistry`Private`MoleculeModifyDump`iComputeAtomCoordinates, RandomSeeding]],
		{
			Automatic :> -1,
			x_Integer :> If[
				0 <= x <= $largestSeedValue,
				x,
				BlockRandom[SeedRandom[x]; RandomInteger[{0, $largestSeedValue}]]
			],
			x_String :> BlockRandom[(SeedRandom[x]; RandomInteger[{0, $largestSeedValue}])],
			Inherited :> BlockRandom[RandomInteger[{0, $largestSeedValue}]],
			other_ :> (
				Message[MessageName[issueMessagesAs,"seeding"], other]; 
				-1
			)
		}
	];
	methodOptions = DeleteCases[methodOptions,HoldPattern["RandomSeeding" -> _]];
	
	Join[ {method}, {"RandomSeeding" -> seed, Sequence @@ methodOptions}]
]



embedMol[mol_iMolecule] := embedMol[mol, {"ETKDGv2"}]
embedMol[mol_iMolecule,{opts__Rule}] := embedMol[mol, {"ETKDGv2",opts}]

embedMol[imol_?ManagedLibraryExpressionQ, {method_String, opts___Rule}]:= Catch[Module[
	{methodInt, methodOpts, res, canonicalize, coords, disconnected},
	
	methodInt = Switch[method,
		"ExperimentalTorsionKnowledgeDistanceGeometryVersion1" | "ETKDGv1",
			1,
		"ExperimentalTorsionDistanceGeometry" | "ETDG",
			2,
		"DistanceGeometry" | "KnowledgeDistanceGeometry" | "KDG",
			3,
		"ExperimentalTorsionKnowledgeDistanceGeometry" | "ETKDG" | "ETKDGv2", 
			0,
		x_String /; (tryGraphEmbedding3D[ imol, x] =!= $Failed),
			-1,
		_,
			(Message[ Molecule::method, method]; Throw[$Failed, $tag]);
	];
	
	canonicalize = TrueQ @ Lookup[ {opts}, "Canonicalize", True];
	
	methodOpts = FilterRules[{opts}, $chemLibraryFunctions["embedMol", "Parameters"]];
	
	methodOpts = Replace[
		methodOpts,
		rule:Except[$embedMolOptionsPattern] :> (
			messageBadOption[rule];
			Throw[$Failed, $tag]
		),
		1
	];

	methodOpts = Replace[
		methodOpts,
		{
			rules:{HoldPattern[_String -> _]..} :> Association[ rules], 
			_ :> Null
		}
	];
	res = Replace[
		If[
			Internal`NonNegativeIntegerQ[methodInt]
			,
			imol[ "embedMol", methodInt, methodOpts]
			,
			imol["get3DConformerIndex"]
		],
		Except[_?Internal`NonNegativeIntegerQ] :> Throw[$Failed,$tag]
	];
	
	disconnected = imol["nFragments"] > 1;
	
	If[
		disconnected
		,
		Message[Molecule::discon];
		alignConnectedComponents[imol, 3];
	];
	
	If[
		canonicalize
		,
		coords = canonicalizeConformer[imol];
		imol["clearConformers"];
		imol["addCoordinates",coords, <|"CanonicalizeConformer" -> False|>];
	];
	
	res
		
	]
,$tag]


tryGraphEmbedding3D[ imol_, method_] := Module[
	{coords},
	coords = GraphEmbedding @ Graph3D[
		Range @ imol["atomCount", True],
		UndirectedEdge[#1,#2] & @@@ imol["getBondsList"],
		GraphLayout -> method
	];
	
	imol["addCoordinates", coords] /; MatrixQ[coords, NumericQ]
 
]

tryGraphEmbedding3D[___] := $Failed;

InertiaTensor[coords_, masses_] := Module[
	{x,y,z,res},
	{x, y, z} = Transpose @ coords;
	{x, y, z} = Map[Sqrt[masses] * #&, {x, y, z}];
	res = ConstantArray[0, {3, 3}];
	res[[1,1]] = (y.y + z.z);
	res[[2,2]] = (x.x + z.z);
	res[[3,3]] = (x.x + y.y);
	res[[1,2]] = res[[2,1]] = -x.y;
	res[[2,3]] = res[[3,2]] = -y.z;
	res[[1,3]] = res[[3,1]] = -x.z;
	res
]

InertiaTensor[mol_] := Module[{masses, coords},
	masses = QuantityMagnitude @ mol @ "AtomicMass";
	coords = QuantityMagnitude @ mol @ "AtomCoordinates";
	InertiaTensor[coords, masses]
] 

translateCOM[coords_, masses_] := With[
    	{com = Total[coords masses]/Total[masses]},
    	# - com & /@ coords
    ]

canonicalizeConformer[ coords_, masses_] := Module[
	{x = coords, m = masses, res, tensor, vectors, t},
	res = translateCOM[x, m];
	tensor = InertiaTensor[res, m];
	vectors = Reverse @ Eigenvectors[tensor];
	(* If the third eigenvector is not equal to the cross product 
    		of the first two, it will introduce a reflection along with rotation. *)
	vectors[[3]] = Cross[ vectors[[1]], vectors[[2]]];
	Developer`ToPackedArray[vectors.# & /@ res]
]


canonicalizeConformer[imol_] := Module[
	{
		masses = imol["realAtomProperty", "AtomicMass", Range @ imol["atomCount",True]],
		coords = imol["get3DCoordinates"]
	},
	canonicalizeConformer[coords, masses]

]

alignConnectedComponents[ imol_ /; (imol["nFragments"] > 1), dim_:3 ] := Module[
	{coords, fragments, newcoords},
	
	coords = If[
		dim === 3, 
		imol["get3DCoordinates"],
		imol["get2DCoordinates"]
	];
	fragments = imol["connectedComponents"];
	
	newcoords = iAlignConnectedComponents[ coords, fragments];
	
	imol["clearConformers"];
	imol["addCoordinates",newcoords, <|"CanonicalizeConformer" -> True|>];
];

alignConnectedComponents[ imol_, 2] := Module[
	{coords},
	coords = imol["get2DCoordinates"];
	coords = alignAndTranslate[coords];
	coords = With[ 
		{mean = Mean[coords]},
		# - mean & /@ coords
	];
	imol["clearConformers"];
	imol["addCoordinates",coords, <|"CanonicalizeConformer" -> False|>];
]

iAlignConnectedComponents[ coords_, components_] := Module[
	{res, padding, displacement, rectangles, width, height, newRectangles, boxData, rotatedData},
	res = coords;

	Scan[
		(res[[#]] = alignAndTranslate[res[[#]]]) &,
		components
	];

	padding = .5;
	rectangles = { Max[ res[[#, 1]] ] + padding, Max[ res[[#, 2]] ] + padding} & /@ components;
	
	rectangles = Ceiling @ rectangles;
	
	{width, height} = getBoxParameters[ rectangles];
	
	newRectangles = packRectangles[ width, height, rectangles];
	
	Do[
		boxData = res[[ components[[n]] ]];
		rotatedData = rotateTranslateComponent[ boxData, rectangles[[n]], newRectangles[[n]]]; 
		res[[components[[n]] ]] = rotatedData;
		,
		{n, Length @ rectangles}
	];

	res
	
]



getBoxParameters[ rectangles_]:= Module[
	{longest, totalArea, width, height, nRectangles},
	
	nRectangles = Length @ rectangles;
	longest = rectangles[[1,1]];
	totalArea = Total[ Times @@@ rectangles ];

	width = Round[ 3/2 longest];
	height = Round[ 3 (totalArea/width)];
	{width, height}	
]

packRectangles[ width_, height_, data_] := Block[
	{nm = newInstance[]},
	nm["insertRectangles", width, height, data, 1]
	
]

rotateTranslateComponent[ coords_, originalRectangle_ , newRectangle_] := Block[
	{rotated, width, height, res, dim = Length @ First @ coords, translation},
	{width, height} = EuclideanDistance @@@ Thread[newRectangle];
	rotated = {height, width} === originalRectangle;
	
	If[
		rotated,
		res = Dot[
			coords,
			RotationMatrix@@If[dim === 3, {Pi/2, {0, 0, 1}}, {Pi/2}]
		],
		res = coords
	];
	translation = -(Min /@ Thread[res]);
	translation += If[
		dim === 3,
		Append[ First @ newRectangle, 0],
		First @ newRectangle
	];
	TranslationTransform[translation ] @ res
	
	
]


$MaxRectFreeChoiceMethods = {"BestShortSideFit", "BestLongSideFit", "BestAreaFit",
		"BottomLeftRule", "ContactPointRule"}
		
$GuillotineFreeChoiceMethods = {"BestAreaFit", "BestShortSideFit", "BestLongSideFit",
		"WorstAreaFit", "WorstShortSideFit", "WorstLongSideFit"}

$GuillotineSplitMethods = {
	"ShorterLeftoverAxis", "LongerLeftoverAxis", "MinimizeArea",
	"MaximizeArea", "ShorterAxis", "LongerAxis"
}
		


$RectangleMethods = {
		"MaximalRectangles", 
		
		"Guillotine"
	};
	
rpSubOptions["MaximalRectangles"] = <|"FreeChoiceRule" -> "BestAreaFit", "AllowFlip" -> True|>
	
rpSubOptions["Guillotine"] = <|"FreeChoiceRule" -> "BestAreaFit", "SplitRule" -> "MaximizeArea", "MergeFreeRectangles" -> True|>


rpMethodPick = With[
	{assoc = AssociationThread[$RectangleMethods -> Range[Length[$RectangleMethods]]]},
	Lookup[assoc, #]&
]

subMethodEnumWithMessage[ submethod_, value_, valid_ ] := Replace[
	FirstPosition[ valid, value], 
	{
		{x_Integer} :> x - 1 (* these library enums start at 0 *),
		_Missing :> (
			Message[RectangleBinPack::submtd, value, submethod, valid];
			Throw[ $Failed, "rectanglePackError"];
		) 
	}
]

(* ************************************************************************* **

                        RectangleBinPack

** ************************************************************************* *)


Options[RectangleBinPack] = {Method -> "MaximalRectangles"}

RectangleBinPack[args___] :=
    Block[
    		{argCheck = System`Private`Arguments[ RectangleBinPack[args], {1,2}], res},
        (
        		res = iRectanglePack @@ argCheck;
        		res /; res =!= $Failed
        ) /; argCheck =!= {}
    ]
    

$rectanglePattern = _Rectangle | _Cuboid | {_,_}

iRectanglePack[ {rectangle:$rectanglePattern, 
		rectangles:{$rectanglePattern..} }, opts_ ] := Block[ 
	{caught, canvasDims, rectDims, res, method, options, obj, unplacedRectanglePositions},
	caught = Catch[
		canvasDims = rectangleDims @ rectangle;
		If[ !MatchQ[ canvasDims, {_Integer, _Integer}], 
			Message[RectangleBinPack::invld, rectangle];
			Throw[ $Failed, "rectanglePackError"]
		];
		
		rectDims = rectangleDims /@ rectangles;
		If[ !MatchQ[ rectDims, {{_Integer, _Integer}..}], 
			Message[RectangleBinPack::invld, rectangles];
			Throw[ $Failed, "rectanglePackError"]
		];
		
		{method,options} = Replace[rpMethod[ OptionValue[ RectangleBinPack, opts, Method] ], x_Integer :> {x,<||>}];
		
		obj = newInstance[];
		
		res = obj["insertRectangles", Sequence @@ canvasDims, rectDims, method, options];
		
		unplacedRectanglePositions = Flatten @ Position[ res, {{0, 0}, {0, 0}}, {1}];
		
		If[ 
			unplacedRectanglePositions =!= {},
			Message[ RectangleBinPack::nopac, unplacedRectanglePositions]
		];
		
		Switch[
			Head @ First @ rectangles,
			Rectangle,
			Rectangle @@@ res,
			Cuboid,
			Cuboid @@@ res,
			List,
			res
		]
		
		,
		"rectanglePackError"];
	caught
]

(* 	:rectangleDims: *)

rectangleDims[ dims:{_Integer, _Integer} ] := dims; 
rectangleDims[ Rectangle[ {_Integer, _Integer} ]  ] := {1,1}
rectangleDims[ (Rectangle|Cuboid)[{xmin_Integer, ymin_Integer},{xmax_Integer, ymax_Integer}] ] := 
	{ Abs[ xmax - xmin], Abs[ymax - ymin]}
	
rectangleDims[ rect:$rectanglePattern /; !FreeQ[rect, _Real] ] := (Message[RectangleBinPack::nopac,rect];
	rectangleDims[ rect /. x_Real :> Ceiling[x] ])



(* 	:rpMethod: *)

rpMethod[ method:(Alternatives @@ $RectangleMethods)] := rpMethod[{method, rpSubOptions[method]}];

rpMethod[{method:(Alternatives @@ $RectangleMethods), subOptions__Rule} ] := rpMethod[{method, Association[{subOptions}]}];

rpMethod[{method:(Alternatives @@ $RectangleMethods), subOptions_Association}] :=
	With[
		{res = { rpMethodPick[method], processSubOptions[method][subOptions] } },
		res /; MatchQ[res, {_Integer,_Association}]
	]
	 
rpMethod[ method_] := (Message[ RectangleBinPack::mtd, method]; Throw[$Failed,"rectanglePackError"];)


(* 	:onlyHasOptions: *)

onlyHasOptions[opts_, method_] := With[ 
	{badOptions = Complement[Keys @ opts ,Keys @ rpSubOptions[method]]},
	Replace[ badOptions, { 
		{} -> True, 
			{opt_,____} :> (Message[RectangleBinPack::moptrs, opt, method]; Throw[$Failed,"rectanglePackError"])}]
	
]
	

(* 	:processSubOptions: *)

processSubOptions["MaximalRectangles"][opts_ /; onlyHasOptions[ opts, "MaximalRectangles"] ] := 
	Block[{res},
		res = Join[ rpSubOptions["MaximalRectangles"],  opts];
		res = MapAt[subMethodEnumWithMessage[ "FreeChoiceRule" , #, $MaxRectFreeChoiceMethods ]&,
				res, "FreeChoiceRule"];
		res
	]

processSubOptions["Guillotine"][opts_ /; onlyHasOptions[ opts, "Guillotine"] ] := 
	Block[{res},
		res = Join[ rpSubOptions["Guillotine"],  opts];
		res = MapAt[subMethodEnumWithMessage[ "FreeChoiceRule" , #, $GuillotineFreeChoiceMethods ]&,
				res, "FreeChoiceRule"];
		res = MapAt[subMethodEnumWithMessage[ "SplitRule" , #, $GuillotineSplitMethods ]&,
				res, "SplitRule"];
		res
	]


alignAndTranslate[pts_] := Module[
	{br = boundingBox @ pts, x, y, z,
	  p1, p2, p3, tfunc, res, dimension},
	
	If[ 
		MatchQ[ br, Except[ _Parallelepiped | _Parallelogram]],
		Return[ translateToOrigin[pts], Module]
	];
	dimension = Switch[
		br,
		_Parallelepiped,
		3,
		_Parallelogram,
		2
	];
	{p2, p1} = Normalize /@ SortBy[ br[[2, ;;2]], Norm];
	If[
		dimension === 3,
		p3 = Cross[ p1, p2]
	];
	(* create an empty tranformation function *)
	tfunc = TranslationTransform[
		If[dimension === 3, {0.0, 0.0, 0.0}, {0.0, 0.0}]
	];
	If[dimension === 3,tfunc[[1, ;;3, ;;3]] = {p1, p2, p3}, tfunc[[1, ;;2, ;;2]] = {p1, p2}];
	
	res = tfunc @ pts;
	
	{x,y} = Min /@ Thread[res[[All, {1,2}]] ];
	If[dimension === 3, z = Mean[ MinMax @ res[[All,3]]] ];
	
	
	tfunc = If[dimension === 3, TranslationTransform[ -{x,y,z} ], TranslationTransform[ -{x,y} ] ];
	 
	tfunc @ res 
]

boundingBox[ pts_] := Quiet[
	(* FastOrientedCuboid is unreliable : https://bugs.wolfram.com/show?number=348802 *)
	If[
		Length @ First @ pts === 3,
		BoundingRegion[ pts, "MinOrientedCuboid"],
		BoundingRegion[ pts, "MinOrientedRectangle"]
	],
	{
		(* TODO: remove Divide::indet when bug(364983) is fixed *)
		BoundingRegion::degbr,Divide::indet 
	}
]
	
translateToOrigin[ pts_ ]:= Module[ 
	{center = RegionCentroid @ Quiet[BoundingRegion @ pts, BoundingRegion::degbr]},
	TranslationTransform[ -center ] @ pts
]


iGeometricProperty[prop:("BondLength" | "InteratomicDistance"), {mol_, b_Bond}, tag_] := 
	firstQuantity @ iGeometricProperty[prop, {mol, {b}}, tag]
	
iGeometricProperty[prop:("BondAngle" | "TorsionAngle" | "OutOfPlaneAngle"), {mol_, b:{__Bond}}, tag_] := 
	firstQuantity @ iGeometricProperty[prop, {mol, {b}}, tag]
	
iGeometricProperty[prop_, {mol_, bond:{__Integer}}, tag_ ] := 
	firstQuantity @ iGeometricProperty[prop, {mol, {bond}}, tag]


iGeometricProperty[prop_, {mol_, atoms_} /; !FreeQ[atoms, Bond], tag_] :=
	Module[{atms,bond},
		
		atms = ReplaceAll[
			atoms, 
			{
				x:Bond[a_,_] /; BondQ[mol, x] :> bond[a],
				x_Bond :> (Message[ Molecule::bond, x]; Throw[$Failed, $tag])
			}
		];
		atms = Replace[
			atms,
			{
				{
					bond[OrderlessPatternSequence[{a_,b_}]],
					bond[OrderlessPatternSequence[{b_,c_}]],
					bond[OrderlessPatternSequence[{c_,d_}]]
				} :> {a,b,c,d},
				{
					bond[OrderlessPatternSequence[{a_,b_}]],
					bond[OrderlessPatternSequence[{b_,c_}]]
				} :> {a,b,c},
				bond[{a_,b_}] :> {a,b}
			},
			{1}
		];
		(
		iGeometricProperty[prop, {mol, atms}, tag]
		) /; FreeQ[ atms, Bond]
	]

iGeometricProperty[prop_, {mol_, atoms:{{__Integer}...}}, tag_ ] := Module[
	{res, validated,im, unit},
	
	validated = atoms /. x_?Negative :> (mol["FullAtomCount"] + x + 1);
	
	MapThread[
		messageOnBadAtomReference[ mol["AtomIndex"], #1, #2, $tag]&,
		{atoms, validated}
	];
	Replace[
		messageOnAtomsDimension[ prop, atoms, validated, tag],
		$Failed :> Return[$Failed, Module]
	];
	
	im = getCachedMol[ mol, "AllAtoms"];
	messageOnNotEmbedded[ mol, im, $tag];
	
	
	
	res = Switch[ prop,
		"InteratomicDistance", 
			im["getBondLengths", validated],
		"BondAngle", 
			im["getBondAngles", validated],
		"TorsionAngle", 
			im["getDihedralAngles", validated],
		"OutOfPlaneAngle", 
			im["getWilsonAngles", validated]
	];
	unit= Switch[ prop,
		"InteratomicDistance", 
			"Angstroms",
		_, 
			"AngularDegrees"
	];
	
	With[
		{head = If[ Length[res] >= 10, quantityArray, quantity]}, 
		head[ res, unit]
	] /; VectorQ[ res, NumericQ] 
]

iGeometricProperty[ prop_, {mol_, All}, tag_ ] := Module[
	{atoms},
	atoms = Switch[ prop,
		"InteratomicDistance", 
			BondList[ mol],
		"BondAngle", 
			MoleculeSubstructureIndices[mol, MoleculePattern["[*]~[*]~[*]"]],
		"TorsionAngle", 
			MoleculeSubstructureIndices[mol, MoleculePattern["[*]~[*]~[*]~[*]"] ],
		_,
			{}
	];
	iGeometricProperty[ prop, {mol, atoms}, tag] /; ListQ[atoms]
]

iGeometricProperty[___] := $Failed


End[] (* End Private Context *)

