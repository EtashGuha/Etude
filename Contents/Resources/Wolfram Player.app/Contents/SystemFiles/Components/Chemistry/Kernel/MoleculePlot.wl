


Begin["Chemistry`Private`MoleculePlotDump`"]



System`MoleculeQ
System`Atom
System`Bond
System`Molecule

Chemistry`MaximumCommonSubstructure






(* ::Section::Closed:: *)
(*MoleculePlot*)


(* ************************************************************************* **

                        MoleculePlot

** ************************************************************************* *)
Options[MoleculePlotMethod] = {
	"ShowAromaticBonds" -> False, 
	"FontScaleFactor" -> Automatic,
	"LineOffsetFactor" -> 1.4,
	"DoubleBondOffset" -> 0.15, 
	"DoubleBondOffsetInRing" -> 0.15, 
	"DoubleBondLengthFraction" -> 0.8, 
	"PhenylRings" -> Automatic,
	"AromaticBondStyle" -> Dashed,
	"HighlightedAtomStyle" -> White,
	"HighlightedBondStyle" -> White,
	"LabeledCarbons" -> Automatic,
	"ImageSizeScaled" -> True,
	"LabelIndices" -> False
};

Options[ MoleculePlot ] = {
	PlotLegends -> None,
	ColorRules -> Automatic,
	PlotTheme :> $PlotTheme,
	Method -> {},
	PerformanceGoal -> Automatic,
	Sequence@@Options[Graphics],
	IncludeHydrogens -> Automatic}
	

MoleculePlot[args___] := Module[
	{argCheck = System`Private`Arguments[ MoleculePlot[args], {1,2} ],res},
	(
		res = Catch[
				iMoleculePlot[Sequence @@ argCheck],
				$tag
		];
		res /; graphicsQ[res] 
	) /; argCheck =!= {}
]

MoleculePlot[ arg1_ /; !MoleculeQ[arg1], ___] := (messageNoMol[MoleculePlot,arg1]; Null /; False)



(* ::Subsection::Closed:: *)
(*iMoleculePlot*)


(* :iMoleculePlot: *)

iMoleculePlot[ {$EmptyMolecule,___},opts:OptionsPattern[]] := Graphics[
	{},
	FilterRules[ {opts}, Options[Graphics]]
]


iMoleculePlot[ {molecule_?MoleculeQ , hl_?AssociationQ} , opts_ ] := 	
	iMoleculePlot[ {molecule, Values @ hl}, Join[ FilterRules[opts,Except[PlotLegends]], 
		{PlotLegends -> mergePlotLegendOption[Keys[hl],OptionValue[MoleculePlot, opts, PlotLegends]]}]]

iMoleculePlot[ args_, opts_  /; OptionValue[MoleculePlot, opts, PlotLegends] =!= None ]:= 
	legendedMoleculePlot[ {MoleculePlot, iMoleculePlot}, args, opts]


mergePlotLegendOption[labels_, None|Automatic] := labels
mergePlotLegendOption[labels_, leg_] := leg /. fun_[Automatic,args___] :> fun[labels,args]

iMoleculePlot[ {molecule_?MoleculeQ , hl_:{}} , opts_ ] := Module[
	{
		util = newInstance @ "PlotUtility",
		aromaticBondStyle, atomicSymbols, autoPlotSize, circles, colorDrawing, colorList,
		colorRules, coordMap, coords, dotsPerAngstrom, doubleBondLengthFraction, doubleBondOffset,
		doubleBondOffsetInRing, fontScaleFactor, graphicsComplex, graphicsElements,
		graphicsOptions, highlightedAtomIndices, highlightedAtoms, highlightedBonds,
		highlightedBondStyle, highlightGraphics, highlights, hydrogenIndices, imageSize,
		imol, kekulize, labelAtoms, labeledVertices, labelIndices, labels, labelSize,
		lineOffsetFactor, lines, linesAndLabels, options, phenylRings, plotData, plotRange,
		plotTheme, rotate, showHydrogens, subOptions, subOptionValue, theme,
		tooltips, wedges, atomCount, labeledCarbons, plotLegends, embeddingOption
	}, 
	
	(* process themes before anything else so that options set by a theme are picked up correctly*)
	Quiet[plotTheme = OptionValue[MoleculePlot, opts, PlotTheme]];
	theme = fixHydrogenOption @ fixPlotThemeOptions @ Charting`ResolvePlotTheme[plotTheme, MoleculePlot];
	Quiet[ subOptions = OptionValue[MoleculePlot, theme, Method]];
	subOptions = Charting`ConstructMethod[ subOptions];
	theme = FilterRules[ theme, Except[Method]];
	theme = FilterRules[ theme, Options[MoleculePlot] ];
	options = Join[opts, theme];
	subOptions = Charting`parseMethod[Charting`ConstructMethod[ OptionValue[ MoleculePlot, options, Method]], subOptions];
	plotLegends = OptionValue[ MoleculePlot, options, PlotLegends];
	
	subOptionValue = Quiet[OptionValue[ MoleculePlotMethod, subOptions, #]]&;
	
	highlights = preprocessHighlights[MoleculePlot, molecule, hl];
	atomCount = AtomCount[molecule];
	
	
	highlightedAtomIndices = Flatten @ Cases[
		highlights,
		Atom[x:{_Integer}]|Bond[x:{__Integer},___] :> x,
		Infinity
	];
	
	Replace[
		highlightedAtomIndices,
		x_ /; !Between[x,{1,atomCount}] :> (
			Message[ Molecule::atom, x]; Throw[ $Failed, $tag]
		),
		{1}
	];
	
	showHydrogens = If[!FreeQ[plotTheme, "AllAtom" | "HeavyAtom"]
		,
		OptionValue[ MoleculePlot, theme, IncludeHydrogens]
		,
		OptionValue[ MoleculePlot, fixHydrogenOption @ opts, IncludeHydrogens]
	];
	
	atomicSymbols = AtomList[molecule, _, "AtomicSymbol"];
	hydrogenIndices = Flatten[ Position[ atomicSymbols, "H"]];
	
	hydrogenIndices = Intersection[ highlightedAtomIndices, hydrogenIndices];
	
	If[
		hydrogenIndices =!= {} && !MatchQ[ showHydrogens, All|True|{__Integer}],
		showHydrogens = hydrogenIndices
	];

	imol = Switch[ showHydrogens,
		Automatic,
			getCachedMol[ molecule, "MoleculePlotAutomatic"],
		Inherited | "ExplicitOnly",
			getCachedMol[molecule],
		True | All,
			getCachedMol[ molecule, "AllAtoms"],
		False | None,
			getCachedMol[ molecule, "NoHydrogens"],
		x:{__Integer} /; ( 
				AllTrue[ x, Between[{1,atomCount}]] && 
				MatchQ[ molecule[{"AtomicSymbol", x}], {"H"..}]
			),
			getCachedMol[ molecule,  {"KeepTheseHydrogens", showHydrogens}],
		_,
			Throw[$Failed, $tag]
	];
	
	If[ !ManagedLibraryExpressionQ[imol], Throw[$Failed, $tag]];
	
	embeddingOption = Replace[
		getMolOption[molecule, AtomDiagramCoordinates],
		Automatic :> Switch[ OptionValue[ MoleculePlot, options, PerformanceGoal],
			Automatic,
				Automatic,
			"Speed",
				"RDKit",
			"Quality",
				"CoordGen",
			_,
				Message[MoleculePlot::perfg, OptionValue[ MoleculePlot, options, PerformanceGoal]];
				Throw[$Failed, $tag]
		]
	]; 
	coords = Check[
		getiMolProperty[ imol, "coords2D", embeddingOption],
		Throw[$Failed, $tag]
	];
	
	highlights = adjustAtomIndices[imol] @ highlights;
	
	highlightedBonds = Cases[
		highlights ,
		Bond[{a_,b_},___] :> {a,b}, Infinity
	];
	
	highlightedAtoms = Cases[
		highlights,
		Atom[{x_}] :> x, Infinity
	];
	
	highlightedAtoms = DeleteDuplicates[
		Flatten[{highlightedAtoms, 
		 highlightedBonds}]
	];
	
	fontScaleFactor = subOptionValue["FontScaleFactor"];
	lineOffsetFactor = subOptionValue["LineOffsetFactor"];
	dotsPerAngstrom = 1 (* This can possibly be exposed as a coordinate scaling option or something similar*);
	doubleBondOffset = subOptionValue["DoubleBondOffset"];
	doubleBondOffsetInRing = subOptionValue["DoubleBondOffsetInRing"];
	doubleBondLengthFraction = subOptionValue["DoubleBondLengthFraction"];
	
	
	kekulize = !subOptionValue["ShowAromaticBonds"];
	phenylRings = !kekulize && (False =!= subOptionValue["PhenylRings"]);
	aromaticBondStyle = subOptionValue["AromaticBondStyle"];
	labeledCarbons = Replace[
		subOptionValue["LabeledCarbons"],
		{
			Automatic :> {0},
			All :> adjustAtomIndices[imol] @ AtomList[ molecule, "C", "AtomIndex"],
			None :> {},
			x:{__Integer} /; MatchQ[ molecule[[1,x,1]], {"C"..}] :> adjustAtomIndices[imol][x],
			_ :> Throw[$Failed, $tag]
		}
	];
	
	colorRules = Replace[
		OptionValue[ MoleculePlot, options, ColorRules],
		{
			Automatic -> $atomColorRules,
			cr:Except[ {Rule[_,_]...} | _?AssociationQ | _?System`DispatchQ] :> (
				Message[ MoleculePlot::invrl, cr];
				Throw[$Failed, $tag]
			)
		}
	];
	
	
	colorDrawing = True;

	labelAtoms = fontScaleFactor =!= 0;
	labelIndices = subOptionValue["LabelIndices"];

	rotate = False;
	
	util[ "setOptions", dotsPerAngstrom, doubleBondOffset, doubleBondOffsetInRing, doubleBondLengthFraction, colorDrawing, 
		labelAtoms, labelIndices, kekulize, rotate, phenylRings, labeledCarbons];

	util["setHighlights", highlightedBonds, highlightedAtoms, ManagedLibraryExpressionID @ imol];
	plotData = Replace[
		util[ "drawMolecule", ManagedLibraryExpressionID @ imol],
		Except[_?AssociationQ] :> Throw[$Failed, $tag]
	];
	
	
	(* 
		The ColorRules can be of the form index -> color, or "symbol" -> color
		replacements are done on the indices first, then the symbols
	*)
	colorList = Replace[
		plotData["AtomIndices"],
		Cases[ colorRules, HoldPattern[ Rule[ _Integer, _]]],
		{1}
	];
	
	With[{indices = plotData["AtomIndices"]},
		Sow[ elementCommonName[#], "AtomNames"]& /@ PadRight[
			atomicSymbols, Max @ indices, "H"][[indices]] 
	];

	colorList = Replace[
		colorList,
		{
			x_Integer /; (x <= Length[atomicSymbols]) :> Replace[
				atomicSymbols[[x]],
				colorRules
			],
			x_Integer :> ("H" /. colorRules)
		},
		{1}
	];
	(* one final replacement is needed since the user may have given a partial list of rules *)
	colorList = Replace[ colorList, $atomColorRules, {1}];
	Sow[ #, "ColorList"]&/@colorList;
	
	coords =  dotsPerAngstrom plotData[ "Coordinates"][[ All, ;; 2]];
	
	
	(* index, atomicNum, formalCharge, isotope, orientation, numchars, chars__ *)	  
	
	labels = plotData["AtomLabels"];
	
	plotRange = plotData["Bounds"] ;
		

	
	{labels, labeledVertices} =  If[
		subOptionValue["HighlightedAtomStyle"] === Automatic ||
			!MatchQ[highlightedAtoms, {__Integer}] 
		,
		createLabels[ labels, colorList]
		,
		createLabels[ labels, 
			ReplacePart[
				colorList,
				Thread[{highlightedAtoms}] -> subOptionValue["HighlightedAtomStyle"]
			]
		]
	];
			
	(* turn the list of labels into an association of label -> {indices} *)
	labels = gatherLabels[ {labels, labeledVertices}];
	(* now call the front end to make the label graphics *)
	labels = KeyMap[ getGraphicsLabel, labels];
	
	fontScaleFactor = dotsPerAngstrom Switch[
		fontScaleFactor,
		Automatic,
		.025 + .005 Tanh[.4 (AtomCount[molecule] - 10)],
		_,
		fontScaleFactor
	];
	
	KeyValueMap[
		(labelSize[Alternatives @@ #2] = lineOffsetFactor fontScaleFactor Last[minDiskFromGraphics[#1]] )&,
		labels
	]; 
	labelSize[_] := 0;
	
	labels = Map[
		Part[ coords, #]&,
		labels
	];
	labels = KeyValueMap[
		rescaleAndTranslateGraphics[fontScaleFactor],
		labels
	];
	

 	lines = plotData["Lines"];
	
	circles = plotData["AromaticCircles"];
	circles =  Switch[circles,
		{{{__Real},_Real}..},
			{ "C" /. colorRules, Circle[ Most @ #1, .7 #2] & @@@ circles},
		_,
			Nothing
	];
			
	highlightedBondStyle = subOptionValue["HighlightedBondStyle"];
		
	
	lines =  createLines[ lines, labeledVertices, colorList, labelSize, 
		aromaticBondStyle, highlightedBondStyle];
	
	(* direction, a1Ids, a2Idx, x1, y1, x2, y2 *)
	wedges = plotData["Wedges"];
	
	wedges = If[
		wedges === {},
		Nothing,
		{Black, createWedges[ wedges, labeledVertices, labelSize]}
	];
	
	
	
	graphicsOptions = Replace[
		FilterRules[options,Options[Graphics]],
		HoldPattern[ Method -> rules:{__Rule}] :> If[
			TrueQ[Length @ FilterRules[ rules, Except @ Options[MoleculePlotMethod]] > 0]
			,
			Method -> FilterRules[ rules, Except @ Options[MoleculePlotMethod]]
			,
			Sequence @@ {}
		],
		{1}
	];
	
	
	If[
		FreeQ[options, HoldPattern[ImageSize -> _]] &&
		TrueQ[subOptionValue["ImageSizeScaled"]]
		,
		Replace[
			Quiet[imageSizeFromPlotRange @@ Append[plotRange, Length @ coords]],
			x:(_?NumericQ | {__?NumericQ}) :> AppendTo[graphicsOptions, System`ImageSizeRaw -> x]
		]
	];
	
	
	
	Clear @ labelSize;
	labelSize[_] := $HighlightScalingFactor * fontScaleFactor^.98 * dotsPerAngstrom ;
	
	
	
	highlightGraphics = highlightRulestoGraphics[
		highlights,
		labelSize
	];
	
	tooltips = {
		Opacity[0], EdgeForm[None], 
		MapThread[ 
			Tooltip[ Disk[#1, .20 dotsPerAngstrom],  #2] &, 
			{coords, plotData["AtomIndices"]}
		] 
	};
	linesAndLabels = gatherLines @ lines;
	graphicsElements = {
		Sequence @@ linesAndLabels,
		wedges,
		tooltips,
		circles
	};

	(* TODO:  Should they fix bug https://bugs.wolfram.com/show?number=335882 then remove this hack *)
	highlightGraphics = highlightGraphics /. {
		StadiumShape[ indices:{_Integer, _Integer}, radius_]  :> 
			StadiumShape[ coords[[indices]] , radius],
		Disk[x_Integer, radius_] :> Disk[coords[[x]], radius]
	};

	
	coords = deleteDuplicatesForAssociation @ Cases[ graphicsElements, {_Real,_Real}, Infinity];
	coordMap = positionIndex @ coords;
	
	
	graphicsComplex = GraphicsComplex[
		Developer`ToPackedArray @ coords,
		Join[
			{highlightGraphics} /. _Missing -> {},
			graphicsElements /. coordMap
		]
	];
	labels = labels /. FilledCurve[ind_, c_] :> FilledCurve[ ind, NumericArray /@ c];
	Graphics[ 
		{
			graphicsComplex,
			labels
		},
		Sequence @@ graphicsOptions
    ]
]


makeEntityDownValue[{iMoleculePlot}]
iMoleculePlot[___] := $Failed



(* ::Subsection::Closed:: *)
(*preprocessHighlights*)


preprocessHighlights[symbol_, mol_, highlights_] := Module[
	{res = atomNameMap[mol] @ highlights, defaultStyleIterator, expandedPattern, literalPattern},
	
	expandedPattern = False;
	literalPattern = _Integer | _Style | _Rule | _UndirectedEdge;
	
	res = Replace[
		res,
		{
			x_List :> x,
			x:literalPattern :> {x},
			x_ :> (expandedPattern = True; expandPattern[symbol,mol,x])
		}
	];
	
	
	defaultStyleIterator = 1;
	
	res = Replace[
		res,
		{
			Style[expr_, style_] :> Style[ 
				Sow[ expr, "InputHighlights"],
				Sow[ style, "HighlightStyles"]
			],
			Style[expr_, style__] :> Style[ 
				Sow[ expr, "InputHighlights"],
				Sow[ Directive[style], "HighlightStyles"]
			],
			expr_ :> Style[ 
				Sow[ expr, "InputHighlights"],
				Sow[ defaultPlotStyle[defaultStyleIterator++], "HighlightStyles"]
			]
		},
		1
	];

	If[expandedPattern,
		res
		,
		normalizeHighlight[symbol,  mol, #] & /@ res
	]
	
	
]


normalizeHighlight[symbol_,  mol_, Style[highlight_, style___] ] := Module[
	{res},
	
	res = Flatten[ {highlight}];
	
	res = Replace[
		res,
		{
			atm_Integer :> Atom[{atm}],
			pattern_MoleculePattern /; MessageValidQuery[symbol][pattern] :>
				pattMatchesForPlot[getCachedMol[mol], pattern],
			query_?ValidQuery :>  pattMatchesForPlot[getCachedMol[mol], query],
			bnd_ :> BondList[mol, bnd]
		},
		{1}
	];
	Style[ Sow[Flatten[{res}],"EvaluatedHighlights"], style]
]




(* ::Subsection::Closed:: *)
(*pattMatchesForPlot*)


pattMatchesForPlot[ imol_, patt_] := Module[
	{matches,pattim},
	pattim = getCachedMol[patt];
	If[ !ManagedLibraryExpressionQ[pattim], Return[$Failed, Module]];
	
	matches = Replace[
		imol["atomAndBondMatches", ManagedLibraryExpressionID @ pattim], 
		 {
		 	Null | {Null} | "InvalidSMARTS" -> {}
		 }
	];
	
	Replace[
		matches,
		{
			bnd : {_Integer, _Integer} :> Bond[bnd], 
			a_Integer :> Atom[{a}]
		}, 
		{2}
	]
	
]


expandPattern[ symbol_, mol_, patt_] := Replace[
	patt,
	{
		pattern_MoleculePattern /; MessageValidQuery[symbol][pattern] :>
			pattMatchesForPlot[getCachedMol[mol], pattern],
		query_?ValidQuery :>  pattMatchesForPlot[getCachedMol[mol], query],
		bnd_ :> BondList[mol, bnd]
	}
]





(* ::Subsection::Closed:: *)
(*adjustAtomIndices*)


adjustAtomIndices[imol_iMolecule /; imol["hasImplicitHydrogens"] ] := Module[
	{atomMap},
	atomMap = Rule[imol["getAtomBookmark", #], #] & /@ Range[imol["atomCount",True]];
	ReplaceAll[ atomMap](*Identity*)
]

adjustAtomIndices[imol_] := Identity



(* ::Subsection::Closed:: *)
(*imageSizeFromPlotRange*)


	
(*
	The basics of this function were taken from ChemicalData, but modified many times.
	I arrived at the values of a,b,c, and d via the following function:
	
	mols = Molecule /@ {
		"N", "O=CCS(=O)(=O)O", "C1=CNC=CC1", "Oc1cccc(Cl)c1O", "NS(=O)(=O)c1nc2ccc(O)cc2s1",
		"NC(=O)C(=O)c1c[nH]c2ccccc12", "CC[C@H](C)C=O.COC", "c1ccc2c(SC3=NCCN3)c[nH]c2c1",
		"CCCC(C)(C)OC(C)C", "CC1(C)C=CC2=C(O1)c1ccccc1[NH+]C2=O"
	};
	{afactor = 1., bfactor = 1.2, dfactor = 2.09, cfactor = 37};(* starting values *)
	f[a_, b_, c_, d_, args__] := Block[
		{afactor = a, bfactor = b, cfactor = c, dfactor = d},
		MoleculePlot @ args
	];
	Manipulate[
		Style[
			Grid[
				Partition[Map[f[a, b, c, d, #]&, mols], 2]
			],
			ImageSizeMultipliers -> {1, 1}
		],
		{{a, afactor}, 0.1, 10, 0.1},
		{{b, bfactor}, 0.1, 10, 0.1},
		{{c, cfactor}, 1, 100, 1},
		{{d, dfactor}, 0.1, 10, 0.01}
	]  

*)

imageSizeFromPlotRange[__, 1] := 80;
imageSizeFromPlotRange[minx_, maxx_, miny_, maxy_, plotRangeDiagonal_, longestBond_,nAtoms_] := Module[
	{afactor = 1.,
	bfactor = 1.2,
	dfactor = 2.09,
	cfactor = 50,
	(* uncomment this line to test different scaling factors *)
	(*afactor = Global`afactor, bfactor = Global`bfactor, cfactor = Global`cfactor, dfactor = Global`dfactor,*) (* This is the overall scaling factor *)
	aRatio (* Upper limit for image size *),
	imagesize, newWidth, max},
	
	If[
		(Round[longestBond] != 0)
		,
		(*2D Coordinates*)
		imagesize = {{minx, maxx},{miny, maxy}};
		
		aRatio = Quiet[Check[
			Abs[Max[{maxx, maxy}] / Min[{maxx, maxy}]],
			100
		]];
			
		imagesize = Abs[ Subtract @@@ (imagesize)];
		(*imagesize is rescaled so that small and large molecules are scaled up, and the image size increases slowly as the aspect ratio increases*)
		(*Note also:the image size is renormalized by the bond length, so that the image size should not depend on the absolute scale used in the.mol file*)    
		newWidth = (afactor*Log[aRatio]/Power[aRatio, bfactor] + 1.)*imagesize[[1]]*(cfactor/longestBond)*
                    (2.125 - 1. Erf[(plotRangeDiagonal/longestBond - 1)/dfactor] + .125 Erf[(plotRangeDiagonal/longestBond - 12.5)/dfactor]);
        imagesize = imagesize * newWidth/imagesize[[1]];
        ,
        (*If there are no bonds listed, then choose a small image size that increases slowly as the number of atoms is increased*)
		imagesize = 150.*(nAtoms^.5) * {1, (maxy-miny)/(maxx -minx)};
	];
	First @ If[ 
		(max = Max @ imagesize) > 400
		,
		imagesize (400 + 0.1 (max - 400))/max
		,
		imagesize
	]
]



(* ::Subsection::Closed:: *)
(*createLabels*)


	

(* labels = {{index, atomicNum, formalCharge, isotope, {coords},orientation,characters}..} *)
createLabels[labels_, colorList_] := Module[{res, labeledVerts, colors},
	res = labels;
	labeledVerts = res[[All,1]];
	colors = colorList[[ res[[All,1]] ]];
	res[[All,-1]] = Characters /@ res[[All, -1]];
	res = makeSubscripts @ res[[All,{3,4,7}]];
	res = labelsymbol @@@ res;
	{Thread[ {colors, res} ], labeledVerts}
	]
createLabels[ {}, _]:= {{}, {}}


(* ::Subsection::Closed:: *)
(*makeSubscripts*)


makeSubscripts = ReplaceRepeated[
	RuleDelayed[
		{
			bef___,
			PatternSequence[
				PatternTest[letter_String, StringMatchQ[LetterCharacter]],
				"_", Shortest[n__String], "_"
			],
			aft___
		},
		{bef, Subscript[letter, StringJoin @ n], aft}
	]
] 
	
	


(* ::Subsection::Closed:: *)
(*stringRadius*)


stringRadius[str:Row[{s_String}], fontsize_] :=
	stringRadius[ s, fontsize]
	
stringRadius[str:Row[{__}], fontsize_] :=
	stringRadius[ ToString[str,StandardForm], fontsize]


stringRadius[label_String, fontsize_] :=1.2 * With[
	{str=label, fs = fontsize},
	0.8 * MathLink`CallFrontEndHeld[
		FrontEnd`Value[
			FEPrivate`StringRectangle[str,"Graphics",fs, FontFamily->"Times"]
		]
	][[2]]
];


(* ::Subsection::Closed:: *)
(*createLines*)


createLines[lines_, labeledVertices_, colorList_, labelSize_, dashingStyle_, highlightedBondStyle_] := Module[{res},
	(* highlightQ, dashed (0 False, 1 True), coloringIdx, a1Idx, a2Idx, x1, y1, x2, y2 *)
	res = SortBy[lines,#[[2]]&]; (* get the dashed lines separate *)
		
	res = With[
		{line = shortenLine[{{#6,#7},{#8,#9}},{labelSize[#4], labelSize[#5]}] },
		{
			If[ #1(*highlightQ*) && highlightedBondStyle =!= Automatic
				,
				highlightedBondStyle
				,
				colorList[[#3]]
			],
			If[ #2 ===1,
				{dashingStyle, Line[line]},
				Line[ line ]
			]
		}
	]& @@@ res
]



(* ::Subsection::Closed:: *)
(*createWedges*)


createWedges[ {}, __] := {};
createWedges[ wedges_, labeledVertices_, labelSize_] := Block[
	{res},
	(* direction, a1Ids, a2Idx, x1, y1, x2, y2 *)
	res = SortBy[First] @ wedges;
	With[
		{direction = #1 /. bondDirMap,
		coord = shortenLine[{{#4,#5},{#6,#7}},{labelSize[#2], labelSize[#3]}]
		},
		
		If[ direction === "BeginWedge",
			{Black, makeWedge[ coord]},
			{Black, makeDashed[coord]}
		]
		
	]  & @@@ res
]


(* ::Subsection::Closed:: *)
(*gatherLines*)


gatherLines[lines_] := Module[{res},
	res = GroupBy[lines,
		First, (* group by the color *)
		Flatten[Rest/@#,1]& (* then combine the primitives for colors *)
	];
	res = res // Normal // Replace[ #, Rule->List, 2,Heads->True] & ; (* Keep only one copy of the color directive *)
	res = ReplaceAll[
		res,
		{before___,dashed:Longest[{Dashed,_Line}..],after___} :>
		{
			before, 
			{Dashed, Line[{dashed}[[All, 2,1]]]},
			after
		}
	];(* contract the dashed lines *)
	res = ReplaceAll[
		res,
		{
			{before___, lin:Longest[PatternSequence[Line[_]..]], after___} :> 
			{before,Line[First/@{lin}],after}
		}
	]; (* contract the other lines *)
	res
]


(* ::Subsection::Closed:: *)
(*shortenLine*)


(* Shortens the two points forming a line by dist1, and dist2 (to leave room for the VertexLabel) *)
shortenLine[{p1_,p2_}, {dist1_, dist2_}] := 
 Module[{v= Normalize[ p2 - p1]},
  {p1 + dist1*v, p2 - dist2*v}
  ]
  
 


(* ::Subsection::Closed:: *)
(*makeWedge*)


makeWedge[{from_, to_}] := With[
	
	{perp = .10 * Normalize[({{0, 1}, {-1, 0}}).(to - from)]},
	
	Polygon[
		{from, 
		perp + to, 
		-perp + to}
	]
]

makeWedge[{from_, to_}, {0, 0}] := makeWedge[{from, to}];

makeWedge[{from_, to_}, {r1_,r2_}] := With[
	{
		perp = Times[
			.10,
			Normalize[
				Dot[{{0, 1}, {-1, 0}}, to + -from]
			]
		]
	},
	Polygon[{Offset[r1 Normalize[to-from],from], Offset[r2 Normalize[from-to],to + perp], Offset[r2 Normalize[from-to],to - perp]}]
];


(* ::Subsection::Closed:: *)
(*makeDashed*)


Clear[makeDashed];
Options[makeDashed] = {
	"OffsetInterval" -> .10(*15*),
	"Thickness" -> 1,
	"Width" -> .10
};




makeDashed[{to_, from_}, opts:OptionsPattern[]] := makeDashed[ {to, from}, {False, False}, opts];
makeDashed[{to_, from_},{ offsetA_, offsetB_}, OptionsPattern[]] := Module[{normalvector, unitvector, distance, lines, centerpoint, currentwidth, r},
  normalvector = Normalize[{{0, 1}, {-1, 0}} . (to - from)];
  unitvector = Normalize[to - from];
  distance = EuclideanDistance[to, from];
  r = OptionValue["OffsetInterval"];
  lines = Table[
    (
     centerpoint = from + unitvector*offset;
     currentwidth = 
      OptionValue["Width"]*(distance - offset)/distance;
     {
      TranslationTransform[currentwidth*normalvector][centerpoint],
      TranslationTransform[-currentwidth*normalvector][centerpoint]
      }
     ),
    {offset, If[ offsetB, 3 r, 0] , distance - If[ offsetA, 3 r, 0], r}
    ];
  
  {Thickness[Small], Line[lines]}
  ]
  


(* ::Subsection::Closed:: *)
(*fontScaleFactor*)


(* This seems to work pretty well.  The old code had a convoluted method of arriving at a Scaled font size,
	but a visual inspection showed it was inversely proportional to the width of the plotted area
	*)
fontScaleFactor[xRange_] := 40./xRange;


(* ::Subsection::Closed:: *)
(*labelsymbol*)


labelsymbol[formalCharge_, isotope_, chars_] := Module[
	{res, symbol, symbolPosition, isotopePos, hpat = Alternatives["H" | Subscript["H", _]]},
	res = Replace[
		chars,
		{
			
			{sym__ /; FreeQ[{sym}, hpat], hh:hpat} :> (
				symbolPosition = 1;
				{StringJoin[sym],hh}
			),
			{hh:hpat, sym__ /; FreeQ[{sym}, hpat]} :> (
				symbolPosition = -1;
				{hh, StringJoin[sym]}
			),
			{sym__} :> (
				symbolPosition = 1;
				{StringJoin[sym]}
			)
		}
		
	];
	symbol = res[[symbolPosition]];
	If[formalCharge=!=0,
		Part[res, symbolPosition] = Superscript[symbol,
			Replace[formalCharge,
				{
					1 -> "+", 
					-1 -> "-", 
					x_ /; x>1 :> StringJoin[ToString[x], "+"],
					y_ /; y<-1 :> StringJoin[ToString[Abs[y]], "-"]
				}
			]
		];
	];
	If[isotope=!=0,
		isotopePos = If[MatchQ[res, {"H" | Subscript["H", _], __}],
			2, 1
		];
		res = Insert[res, Superscript["\[InvisiblePrefixScriptBase]", ToString @ isotope], isotopePos];
	];
	Row @ res
]
 	



(* ::Subsection::Closed:: *)
(*highlightRulestoGraphics*)

$HighlightScalingFactor = 7;

highlightRulestoGraphics[highlightRules_List, labelSize_] := ReplaceAll[
	highlightRules,
	{
		Atom[ {x_Integer} ] :> Disk[ x, labelSize[x]],
		Bond[{x_Integer, y_Integer}, ___] :> StadiumShape[
			{x,y},
			Max[ labelSize[x], labelSize[y]]
		],
		style:defaultPlotStyle[_Integer] :> getDefaultPlotStyle[style]
	}
]


getDefaultPlotStyle[defaultPlotStyle[x_Integer]] := substructureColors[[Mod[ x, Length @ substructureColors, 1] ]]
getDefaultPlotStyle[x_] := x;
	
(*	{substructRules},
	
	
	substructRules =formatSubstructureRules @ highlightRules;
	Replace[
		substructRules, 
		{
			atoms:{__Integer} :> (Disk[ #, labelSize[#]] & /@ atoms),
			bonds:{__Bond} :> (
				StadiumShape[
					#[[1]] , 
					Max[labelSize /@ (#[[1]])]] & /@ bonds)
		},
		2
	]
];*)


(* ::Subsection::Closed:: *)
(*directiveQ*)


directiveQ[dir_] := System`Dump`ValidDirective @@ {dir};


(* ::Subsection::Closed:: *)
(*formatSubstructureRules*)


formatSubstructureRules[highlights_List]:= Module[{},
	
	
	If[!MatchQ[highlights, {(_?directiveQ | {(_Integer|_Rule|_Bond)..}) ..}], Return @ Missing["InvalidSubstructureRules"]];
	
	If[
		MatchQ[ highlights, {{(_Integer|_Rule|_Bond)...}..}],
		Flatten[
			Thread[
				{ substructureColors[[;;Length[highlights]]],
					highlights}
			],
		1],
		highlights
	]
];



(* ::Subsection::Closed:: *)
(*substructureColors*)


substructureColors = RGBColor @@@ {{0.985248,0.676238,0.0398315},{0.21099,0.531208,0.953188},{0.519913,0.338384,0.950217},{0.0358167,0.691123,0.698773},{0.68343,0.28,0.602415},{1.,0.4,0.},{0.655728,0.8,0.},{0.,0.742291,0.873126},{1.,0.0231022,0.144121},{0.893126,0.4,0.767184},{0.295048,0.8,0.286932}};
substructureColors = Join[#, Darker/@#] &@ substructureColors;



(* ::Subsection::Closed:: *)
(*deleteDuplicatesForAssociation*)


deleteDuplicatesForAssociation = Keys @ positionIndex @ # &;


(* ::Subsection::Closed:: *)
(*positionIndex*)


positionIndex = AssociationThread[#, Range @ Length @ #] &;


(* ::Subsection::Closed:: *)
(*graphicsQ*)


graphicsQ[_Graphics] := True	
graphicsQ[_Pane] := True
graphicsQ[HoldPattern[Legended[_Graphics,__]]] := True

graphicsQ[___] := False



(* ::Subsection::Closed:: *)
(*coordinatesQ*)


coordinatesQ[x_] := MatrixQ[ x , NumberQ]

(* ::Subsection::Closed:: *)
(*getGraphicsLabel*)

getGraphicsLabel[{color_, row_}] := {color, labelGraphics[row]}

labelGraphics := With[
	{file = FileNameJoin[
			{
				PacletManager`PacletResource["Chemistry", "labelGraphicsCache"],
				StringJoin[ IntegerString[$SystemWordLength], "Bit"],
				"labelGraphicsCache.mx"
			}
		]
	},
	ClearAll @ labelGraphics;
	Quiet[Get[file]];
	labelGraphics[label_] := labelGraphics[label] = UsingFrontEnd @ graphicsFromBox @ labelToBox @ label;
	labelGraphics
	
]


(* ::Subsection::Closed:: *)
(*graphicsFromBox*)


ClearAll@graphicsFromBox;
graphicsFromBox[box_] := Cases[
			
		ToExpression[
			First @ FrontEndExecute[
				FrontEnd`ExportPacket[
					Cell[ BoxData @ box, "Output","Graphics","GraphicsLabel"], 
					"InputForm", "Outlines" -> True
				]
			]
		],
		_FilledCurve,
		Infinity
	] 


(* ::Subsection::Closed:: *)
(*labelToBox*)


labelToBox[row_] := Replace[
	ToBoxes[
		Style[
			ReplaceAll[x_String :> RawBoxes[x]] @ row,
		 	CurrentValue @ {FrontEnd`GeneratedCellStyles, "Output"},
		 	Apply[Sequence,
		  		Flatten @ {CurrentValue @ {GraphicsBoxOptions, DefaultBaseStyle}}
		  	],
		 	Apply[Sequence,
		  		Flatten @ {CurrentValue @ {GraphicsBoxOptions, 
		      DefaultLabelStyle}}
		  	],
		 	ToString[
		  		CurrentValue[{GraphicsBoxOptions, FormatType},
		   			Sequence @@ 
		    Flatten[{CurrentValue[{InsetBoxOptions, DefaultStyle}]}]
		   		]
		  	]
		]
	],
	Except[_StyleBox] :> Throw[$Failed, $tag]
]

 


(* ::Subsection::Closed:: *)
(*gatherLabels*)


gatherLabels[labels_] := GroupBy[ Thread[labels], First -> Last]


(* ::Subsection::Closed:: *)
(*minDiskFromGraphics*)


minDiskFromGraphics[graphics_] := Module[
	{pts},
	pts = Cases[ 
		graphics, 
		coords:{{_?NumericQ,_?NumericQ}..} :> Sequence @@ coords,
		Infinity
	];
	BoundingRegion[ pts, "MinDisk"]
]	 


(* ::Subsection::Closed:: *)
(*rescaleAndTranslateGraphics*)


rescaleAndTranslateGraphics[scaleFactor_][graphics_, coords:{__List}] := Module[
	{center, radius, transforms},
	{center,radius} = List @@ minDiskFromGraphics[graphics];
	transforms = Map[
		Function[ coord,
			Composition[
				TranslationTransform[ coord], 
				ScalingTransform[{scaleFactor, scaleFactor}],
				TranslationTransform[-center]
			]
		],
		coords
	];
	GeometricTransformation[
		graphics,
		transforms
	]
]


(* ::Section::Closed:: *)
(*molIcon*)


(* ************************************************************************* **

                        molIcon

** ************************************************************************* *)


$molIconMaxAtomCount = 200

$maxIconWidth = 100;


$molIconOptions = {
	Axes -> False, 
	Frame -> True, FrameTicks -> None, PlotRangePadding -> None, FrameStyle -> Directive[
		Opacity[0.5], Thin, GrayLevel[0.7]
	],
	ImageSize -> Dynamic[{UpTo[8.1 * CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]], 
		UpTo[4.1 * CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]]}]
};


molIcon[mol_?MoleculeQ /; AtomCount[mol] < $molIconMaxAtomCount] :=
    Block[{res},
        res = getMolIcon[mol];
        res /; graphicsQ[res]
    ]

molIcon[expr___] := $Failed


(* :getMolIcon: *)

getMolIcon[ $EmptyMolecule] := Pane[Graphics[Sequence @@ $molIconOptions], ImageSize -> {30, 30}, Alignment -> Center];

getMolIcon[ mol_ ] := MoleculePlot[ mol, Sequence @@ $molIconOptions]
	
getMolIcon[___] := $Failed




(* ::Section::Closed:: *)
(*MoleculePlot3D*)


 
Options[MoleculePlot3DMethod] = {
	(*"ShowAromaticBonds" -> False, *)
	"DrawMultiBonds"-> True,
	"AtomScaling" -> Automatic,
	"DrawBonds" -> True,
	"DrawAtoms" -> True, 
	"AtomSizeRules" -> Automatic,	
	"WireframeBonds" -> False, 
	"HighlightScalingFactor" -> 1.2,
	"HighlightOpacity" -> 0.7,
	"BondSpacingRules" -> Automatic,
	"BondSizeRules" -> Automatic,
	"Directives" -> {},
	"LabelIndices" -> False
};

Options[ MoleculePlot3D ] = {
	(*"LabelIndices" -> False,*)
	(*AtomCoordinates -> Automatic,*)
	PlotTheme :> $PlotTheme,
	PlotLegends -> None,
	ColorRules -> Automatic,
	Boxed -> False,
	Method -> {},
	IncludeHydrogens -> True,
	Sequence@@Options[Graphics3D]
} 
	



MoleculePlot3D[args___] :=
    Module[{argtest, res},
    		argtest = System`Private`Arguments[MoleculePlot3D[args], {1,2}, List, {iMoleculePlot3D,MoleculePlot3DMethod}];
    		(
    			res = Catch[ iMoleculePlot3D @@ argtest, $tag];
    			res /; MatchQ[res, _Graphics3D | _Legended]
    		) /; (argtest =!= {})
        
    ]


MoleculePlot3D[ arg1_ /; !MoleculeQ[arg1], ___] := (messageNoMol[MoleculePlot3D,arg1]; Null /; False)



(* ::Subsection::Closed:: *)
(*iMoleculePlot3D*)


iMoleculePlot3D[ {$EmptyMolecule,___},opts:OptionsPattern[]] := Graphics3D[
	{},
	Boxed -> False,
	FilterRules[ {opts}, Options[Graphics3D]]
]

iMoleculePlot3D[ {molecule_?MoleculeQ , hl_?AssociationQ} , opts_ ] := 	
	iMoleculePlot3D[ {molecule, Values @ hl}, Join[ FilterRules[opts,Except[PlotLegends]], 
		{PlotLegends -> mergePlotLegendOption[Keys[hl],OptionValue[MoleculePlot3D, opts, PlotLegends]]}]]


iMoleculePlot3D[ args_, opts_  /; OptionValue[MoleculePlot3D, opts, PlotLegends] =!= None ]:= 
	legendedMoleculePlot[ {MoleculePlot3D, iMoleculePlot3D}, args, opts]
	
	
iMoleculePlot3D[{ent_?MoleculeQ, hl_:{}}, opts_] := Module[
	{
		plotData, coords, highlights, options, bonds, atoms, imol, res, kekulize, 
		theme, subOptions, showHydrogens, highlightedAtomIndices, hydrogenIndices, 
		atomIndices, toKeep, atomCount, plotTheme
	},
	
	
	imol = getCachedMol[ ent, "AllAtoms"];
	messageOnNotEmbedded[ent, imol, $tag];
	
	
	(* process themes before anything else so that options set by a theme are picked up correctly*)
	Quiet[plotTheme = OptionValue[MoleculePlot3D, opts, PlotTheme]];
	theme = Replace[
		plotTheme,
		{"AllAtoms" -> "AllAtom", "NoHydrogens"|"HeavyAtoms" -> "HeavyAtom", "SpaceFilling" -> "Spacefilling"},
		{0,1}
	];
	theme = fixHydrogenOption @ fix3DPlotThemeOptions @ Charting`ResolvePlotTheme[theme, MoleculePlot3D];
	Quiet[ subOptions = OptionValue[MoleculePlot3D, theme, Method]];
	subOptions = Charting`ConstructMethod[ subOptions];
	theme = FilterRules[ theme, Except[Method]];
	theme = FilterRules[ theme, Options[MoleculePlot3D] ];
	options = Join[opts, theme];
	subOptions = Charting`parseMethod[Charting`ConstructMethod[ OptionValue[ MoleculePlot3D, options, Method]], subOptions];
	subOptions = FilterRules[ subOptions, Options @ MoleculePlot3DMethod];
	
	
	showHydrogens = If[!FreeQ[plotTheme, "AllAtom" | "HeavyAtom"]
		,
		OptionValue[ MoleculePlot3D, theme, IncludeHydrogens]
		,
		OptionValue[ MoleculePlot3D, fixHydrogenOption @ opts, IncludeHydrogens]
	];
	
	highlights = preprocessHighlights[MoleculePlot3D, ent, hl];
	highlightedAtomIndices = Cases[
		highlights,
		Atom[{x_Integer}] | Bond[ {y__Integer},___] :> Sequence[x,y],
		Infinity
	];
	
	atomCount = AtomCount[ent];
	Replace[
		highlightedAtomIndices,
		x_ /; !Between[x,{1,atomCount}] :> (
			Message[ Molecule::atom, x]; Throw[ $Failed, $tag]
		),
		{1}
	];
	
	hydrogenIndices = AtomList[ ent, Atom["H"], "AtomIndex"];
	
	If[
		ContainsAny[
			highlightedAtomIndices,
			hydrogenIndices
		],
		showHydrogens = True
	];
	
	
	coords = ent["AtomCoordinates"];
	
	If[ !has3DCoordinates[imol],Return[ $Failed, Module]];
	
	toKeep = Switch[showHydrogens,
		True | All,
			All,
		False | None,
			None,
		x:{__Integer} /; ( 
				AllTrue[ x, Between[{1,AtomCount[ent]}]] && 
				MatchQ[ ent[{"AtomicSymbol", x}], {"H"..}]
			),
			showHydrogens,
		Inherited | "ExplicitOnly",
			Replace[
				AtomList[ ent, "H", "AtomIndex"],
				Except[{__Integer}] :> None
			],
		_,
			Throw[$Failed, $tag]
	];
	
	If[
		toKeep =!= All
		,
		
		With[{imol2 = newInstance[]},
			Switch[toKeep,
				{__Integer},
					imol2[ "createCopyKeepingHydrogens", ManagedLibraryExpressionID @ imol, toKeep, True, False],
				None,
					imol2["createCopyWithNoHydrogens", ManagedLibraryExpressionID @ imol, True, False];,
				_,
					Throw[$Failed, $tag]
			];
			imol = imol2;
		];
		highlights = adjustAtomIndices[imol][highlights]; 
		subOptions = subOptions /. HoldPattern[Rule["LabelIndices",lab_]]:> Rule["LabelIndices", adjustAtomIndices[imol]@lab];

	];
	
	atomIndices = adjustAtomIndices[imol][Range @ imol["atomCount", True]];
	

	
	(* TODO: add support for aromatic bond display in 3D *)
	kekulize = True;
	plotData = imol[ "getPlotData", kekulize ];
	bonds = Replace[
		plotData["Bonds"],
		Null -> {}
	] /. "Unspecified" -> "Single";
	
	highlights = ReplaceAll[
		highlights,
		{
			Bond[{a_, b_}, "Aromatic"] :> FirstCase[
				bonds,
				{OrderlessPatternSequence[a,b],type_} :>
					Bond[{a,b},type]
			]
		}
	];
	
	atoms = plotData["AtomicNumbers"];
	coords = plotData["Coordinates"];
	coords = N[coords];
	
	If[
		!MatrixQ[coords,NumericQ]
		,
		Message[MoleculePlot3D::nocoord];
		Throw[ $Failed, $tag]
	];

	
	res /; MatchQ[ 
		res = oMoleculePlot3D[{atoms, bonds, coords, highlights, atomIndices, subOptions, options}],
		_Graphics3D
	] 
]
	
makeEntityDownValue[{iMoleculePlot3D}]
iMoleculePlot3D[___] := $Failed


(* ::Subsection::Closed:: *)
(*oMoleculePlot3D*)


oMoleculePlot3D[{atms_, bnds_, crds_, hl_, indices_, subOptions_, options_}] := Module[
	{
		atomcount, atomdata, atomlist, atoms, atomScaling, atomSizeRules, bonddata,
		bondRenderer, bonds, bondsize, bondSize, bondSpacing, bondTypes, colorList,
		colorRules, coords, cylinderHelper, cylinders, dir, drawAtoms, drawBonds,
		drawMultiBonds, getNeighborBonds, graphicsOptions, highlightedAtoms, highlightedBonds, highlightGraphics,
		highlights, highlightScaling, imagesizeMax, labels, lineHelper,
		makeSpheres, mcolor, msize, multibonds, points, shift,
		showHydrogens, spacing, spheres, tubeHelper,
		vec, wireframeBonds, subOptionValue
	},
	subOptionValue = OptionValue[ MoleculePlot3DMethod, subOptions, #]&;
	
	
	colorRules = Replace[
		OptionValue[ MoleculePlot3D, options, ColorRules],
		{
			Automatic -> $atomColorRules3D,
			cr:Except[ {Rule[_,_]...} | _?AssociationQ | _?System`DispatchQ ] :> (
				Message[ MoleculePlot3D::invrl, cr];
				Throw[$Failed, $tag]
			)
		}
	];
  		
	atomScaling = subOptionValue["AtomScaling"];
	drawBonds = subOptionValue["DrawBonds"];
	drawAtoms = subOptionValue["DrawAtoms"];
	wireframeBonds = subOptionValue["WireframeBonds"];
	drawMultiBonds = subOptionValue["DrawMultiBonds"];
	showHydrogens = OptionValue[ MoleculePlot3D, options, IncludeHydrogens];
	bondSpacing = subOptionValue["BondSpacingRules"];
	bondSize = subOptionValue["BondSizeRules"];
  	
  	
  	highlightedBonds = Cases[
		hl,
		{x : __Bond} :> x, {1}
	];
		
	highlightedAtoms = Cases[
		hl,
		{x : __Integer} :> x, {1}
	];
  	
  	{spacing["Double" | "Aromatic"], spacing["Triple"]} = Replace[ {"Double", "Triple"}, bondSpacing, {1}];
	{
		bondsize["Single"],
		bondsize["Double" | "Aromatic"],
		bondsize["Triple"]
	} = Replace[ {"Single", "Double", "Triple"}, bondSize, {1}]; 
	bondsize[_] := bondsize["Single"];
		
	atomSizeRules = Replace[
		subOptionValue["AtomSizeRules"],
		Automatic :> $atomSizeRules3D
	];
	
	
	colorList = Replace[
		Range @ Length @ atms,
		Cases[colorRules, HoldPattern[ _Integer -> _]],
		{1}
	];
	colorList = Replace[
		colorList,
		x_Integer :> Replace[
			FromAtomicNumber[atms[[x]]],
			colorRules
		],
		{1}
	];
	(* one final replacement is needed since the user may have given a partial list of rules *)
	colorList = Replace[ colorList, $atomColorRules3D, {1}];
	Sow[ #, "ColorList"]&/@colorList;
	
	atoms = atms;
	coords = crds;
	bonds = bnds;
	
	bondTypes = bonds[[All,3]];
	
	bonds = Most /@ bonds;
	

	
	(* Number atoms *)
	atomlist = FromAtomicNumber[atoms];
	atomdata = Transpose[{atomlist, Range[Length[atoms]], indices}] ;
	atomcount = Length[atomlist];
	(* Add padding for 2D data *)
	points = coords;
	
	Sow[ elementCommonName[#], "AtomNames"]& /@ atomlist;
	
	makeSpheres = Function[sdata, 
		mcolor = colorList[[ sdata[[ 1, 2]] ]];
		msize = atomScaling*(sdata[[1, 1]] /. atomSizeRules);
		{mcolor, Sphere[sdata[[All, 2]], msize] }
		(*{mcolor, Tooltip[Sphere[#2,  msize],#3] & @@@ sdata  }*)
	];
	
	(* Helper functions to draw bonds *)
	
	lineHelper = Function[{a, color},
		If[ListQ[a]&&(Length[a]===1), {color, Line[a[[1,All,1]]]}, Sequence[]]
	];

	cylinderHelper[a_,color_] := If[
		ListQ[a]&&(Length[a]===1), 
		{color, 
			Cylinder[#1, bondsize[#2]]& @@@ a[[1]]}, 
		Sequence[]
	];
	
	tubeHelper[a_,color_] := 
	If[
		ListQ[a]&&(Length[a]===1), 
		With[
			{splitBySize = SplitBy[ SortBy[a[[1]], Last], Last]},
			{color, 
			Cylinder[ #[[All, 1]], bondsize @ #[[1, 2]] ] & /@ splitBySize}], 
		Sequence[]
	];
	

	
	bondRenderer = If[TrueQ @ wireframeBonds, lineHelper, tubeHelper];
	
	(* Helper functions to determine the plane in which to draw the extra cylinders *)
	
	bonddata = If[ListQ[bonds], Thread[{List @@@ bonds,bondTypes}], {}]; 
	
 	If[drawMultiBonds,
		multibonds = Pick[ Range@Length@bonddata, bondTypes, "Double"|"Triple"|"Aromatic"];
		
		vec[a_,b_]:=Subtract@@coords[[{a,b}]];
		vec[{a_,b_}]:=vec[a,b];
		
		getNeighborBonds[bondindex_]:=getNeighborBonds[bondindex]=Select[First/@Delete[bonddata,bondindex],(Intersection[#, First@bonddata[[bondindex]]]=!={} &)];
		
		shift[bondindex_]:= shift[First@bonddata[[bondindex]]] = With[
			{thisbond = First@bonddata[[bondindex]],
			neighbors=getNeighborBonds[bondindex], 
			default = Normalize[ViewPoint /. options /. ViewPoint->{1.3,-2.4,-2.}]},
			
			dir = If[
				neighbors==={},
				default,
				Scan[(If[sufficientlyNonParallelQ[vec[thisbond],vec[#]],Return[Normalize[Cross[vec[thisbond],vec[#]]]]])&,neighbors]/.Null->default
				];
			spacing[bondTypes[[bondindex]]] Normalize[Cross[vec[thisbond],dir]]
		];
		
		Scan[shift,multibonds];, 
		
		(* If no multi bonds are to be shown, set all the bond types to "Single" *)
		
		bonddata = Cases[ bonddata, {{a_,b_},c_}:>{{a,b},"Single"}];
	];

	highlightedBonds = Map[
		Function[hlBond,
			FirstCase[
				bonddata,
				{ First[ hlBond] | Reverse[First[hlBond]] , _},
				Nothing
			]
		],
		highlightedBonds
	];

	bonddata = {#1, #2, colorList[[ #1]]} & @@@ bonddata ;
	highlights = hl /. Bond[{a_,b_}] :> Bond[{a,b},"Single"];
	
	If[ 
		!MissingQ[ highlights]
		,
		highlightScaling = subOptionValue["HighlightScalingFactor"];
		
		highlightGraphics = ReplaceAll[
			highlights,
			{
				style:defaultPlotStyle[_Integer] :> getDefaultPlotStyle[style],
				Atom[{x_Integer}] :> Sphere[x, highlightScaling atomScaling(atomdata[[x, 1]] /. atomSizeRules)],
				Bond[ {a_,b_}, order_] :> {
					Sphere[ a, highlightScaling atomScaling(atomdata[[a, 1]] /. atomSizeRules)],
					Sphere[ b, highlightScaling atomScaling(atomdata[[b, 1]] /. atomSizeRules)],
					Switch[
						order,
						"Double",
						Cylinder[coords[[{a,b}]] + # {shift[{a,b}],shift[{a,b}]}, highlightScaling bondsize[order] ] & /@ {-1,1},
						"Triple",
						Cylinder[coords[[{a,b}]] + # {shift[{a,b}],shift[{a,b}]}, highlightScaling bondsize[order] ] & /@ {-1,0,1},
						_(*"Single"*), 
						Cylinder[coords[[{a,b}]], highlightScaling bondsize[order]]
					]
				}
			}
		];		
		
		highlights = {
			Opacity @ subOptionValue["HighlightOpacity"], 
			highlightGraphics
		};
		,
		highlights = Nothing
	];
	
	(* Draw the spheres and cylinders *)
	
	spheres = If[drawAtoms,
		makeSpheres /@ Split[Sort@ atomdata, colorList[[ #1[[2]] ]] === colorList[[ #2[[2]] ]] &],
		{}
	];
	
	cylinders = If[
		Length[bonddata]>0 && drawBonds
		,
		{points,bonddata} = Reap[
			sowBonds3D[ coords, bonddata, shift], 
			Union @ colorList
		];
		bonddata = bonddata /. points;
		MapThread[
			bondRenderer,
			{bonddata, Union @ colorList }
		]
		,
		points = AssociationThread[ coords, Range @ Length @ coords];
		Unevaluated@Sequence[]
	];
	
	points = Keys @ points;
	
	graphicsOptions = Join[
		Replace[
			FilterRules[options,Options[Graphics3D]],
			HoldPattern[ Method -> rules:{__Rule}] :> If[
				TrueQ[Length @ FilterRules[ rules, Except @ Options[MoleculePlot3DMethod]] > 0]
				,
				Method -> FilterRules[ rules, Except @ Options[MoleculePlot3DMethod]]
				,
				Sequence @@ {}
			],
			{1}
		],
		{ViewPoint -> viewPointFromCoords[coords], Boxed->False, Lighting ->"Neutral"}
	];

	
	If[
		And[
			OptionValue[ MoleculePlot3D, graphicsOptions, SphericalRegion] === Automatic,
			!MatchQ[
				OptionValue[ MoleculePlot3D, graphicsOptions, ViewVector],
				{{_,_,_},{_,_,_}}
			]
		]
		,
		graphicsOptions = Join[
			DeleteCases[graphicsOptions, SphericalRegion -> Automatic] ,
			{Rule[ SphericalRegion, boundingSphere[spheres,coords]]}
		]
	];
	
	
	labels = Switch[ labels = subOptionValue["LabelIndices"],
		{__Integer},
			getCallouts2[ coords[[labels]], labels, graphicsOptions  ],
		All | True,
			getCallouts2[ coords, Range @ Length @ coords, graphicsOptions  ],
		_,
			Sequence@@{}
	];

	 
	Graphics3D[{
		Specularity[White,100],
		Sequence @@ subOptionValue["Directives"],
  		GraphicsComplex[points, {labels, spheres, cylinders, highlights}]}, 
		Sequence @@ graphicsOptions
	]
  ];

Attributes[associateTo] = {HoldFirst}
associateTo[pts_, pt_] /; KeyFreeQ[pts,pt] := AssociateTo[ pts, pt -> (Length @pts +1)]

sowBonds3D[coords:{{__Real}..}, bondData_, shiftVals_] := Module[
	{ extrapoint, groupBonds , 
	points = AssociationThread[ coords, Range @ Length @ coords]},
	groupBonds[{{a:{__Real},b:{__Real}},order_String, {colorA_, colorB_}}] := If[
		
		colorA === colorB,
		Sow[{{a, b},order}, colorA];,
		
		extrapoint = a + (b - a)/2.;
		associateTo[ points, extrapoint];
		Sow[{{a, extrapoint},order}, colorA];
		Sow[{{extrapoint, b},order}, colorB];
	];
	
	groupBonds[{{a_Integer,b_Integer},"Single", {colorA_, colorB_}}] :=
		groupBonds[ {coords[[{a,b}]], "Single", {colorA, colorB}} ];
		
	groupBonds[{{a_Integer,b_Integer}, "Double", {colorA_, colorB_}}] :=Module[
		{shift, newcoords},
		shift = shiftVals[{a,b}];
		newcoords = Join[coords[[{a,b}]] + {shift,shift},coords[[{a,b}]] - {shift,shift}];
		groupBonds[{
			newcoords[[;;2]],
			"Double", {colorA, colorB}}];
		groupBonds[{
			newcoords[[3;;]],
			"Double", {colorA, colorB}}];
		Scan[ associateTo[points,#]&, newcoords];
	];
	
	groupBonds[{{a_Integer,b_Integer}, "Aromatic", {colorA_, colorB_}}] :=Module[
		{shift, newcoords},
		shift = shiftVals[{a,b}];
		newcoords = Join[coords[[{a,b}]]  + {shift,shift},coords[[{a,b}]] - {shift,shift}];
		groupBonds[{
			newcoords[[;;2]],
			"Double", {colorA, colorB}}];
		groupBonds[{
			newcoords[[3;;]],
			"Double", {colorA, colorB}}];
		Scan[ associateTo[points,#]&, newcoords];
	];
	
	groupBonds[{{a_Integer,b_Integer}, "Triple", {colorA_, colorB_}}] :=Module[
		{shift, newcoords},
		shift = shiftVals[{a,b}];
		newcoords = Join[coords[[{a,b}]] + {shift,shift},coords[[{a,b}]] - {shift,shift}];
		groupBonds[ { coords[[{a,b}]], "Triple", {colorA, colorB}} ];
		groupBonds[{
			newcoords[[;;2]],
			"Triple", {colorA, colorB}}];
		groupBonds[{
			newcoords[[3;;]],
			"Triple", {colorA, colorB}}];
		Scan[ associateTo[points,#]&, newcoords];
	];
	
	Scan[groupBonds, bondData];
	points
	
]	


(* ::Subsection::Closed:: *)
(*imageSizeFromPlotRange3D*)


imageSizeFromPlotRange3D[plotrange_, eRules_, coords_, nAtoms_] := Module[
	{afactor = .53,
	bfactor = 1.8,
	dfactor = 1.66,
	cfactor = 350, (* This is the overall scaling factor *)
	longestBond, imagesize, plotRangeDiagonal},
	plotrange;
	If[(Length[Union @ Flatten[ List @@@ eRules]] != 0),
		longestBond = Max[Norm[coords[[#[[1]]]] - coords[[#[[2]]]]] & /@ eRules];
		
		(*3D Coordinates*)
		imagesize = {plotrange[[1]] - plotrange[[1, 1]], plotrange[[2]] - plotrange[[2, 1]], plotrange[[3]] - plotrange[[3, 1]]};
	  	If[Abs[imagesize[[1, 2]] - imagesize[[1, 1]]] < (1.*10^-6)*longestBond, imagesize[[1, 2]] = longestBond, Null];
	  	If[Abs[imagesize[[2, 2]] - imagesize[[2, 1]]] < (1.*10^-6)*longestBond, imagesize[[2, 2]] = longestBond, Null];
	  	If[Abs[imagesize[[3, 2]] - imagesize[[3, 1]]] < (1.*10^-6)*longestBond, imagesize[[3, 2]] = longestBond, Null];
	  	(*lp is the diagonal length of the plot range*)    
	  	plotRangeDiagonal = Sqrt[imagesize[[1, 2]]^2 + imagesize[[2, 2]]^2 + imagesize[[3, 2]]^2];
			
		(*imagesize is rescaled so that small and large molecules are scaled up, and the image size increases slowly as the aspect ratio increases*)
		(*Note also:the image size is renormalized by the bond length, so that the image size should not depend on the absolute scale used in the.mol file*)    
		imagesize = (1 + Log[nAtoms, 10]/2)*cfactor*(afactor + (1 - afactor)*Erf[Power[(plotRangeDiagonal/longestBond), .6]/bfactor - dfactor]),
	  (*If there are no bonds listed, then choose a small image size that increases slowly as the number of atoms is increased*)
		imagesize = 150.*(nAtoms^.5);
	];
	imagesize
]


(* ::Subsection::Closed:: *)
(*InferBonds*)


  

InferBonds[elementNames:{__String},coords_, min_, tol_] := 
	InferBonds[ AtomicNumber[ elementNames], coords, min, tol];
	
InferBonds[atomicNumbers:{__Integer}, incoords_List, min_, tol_] :=
    Module[ {iatoms, icoords, max, atomR, atomP, nearR, near, nearD, curAtom, bondingNeighbors, nearFunc},
        icoords = Developer`ToPackedArray@N@incoords;
        iatoms = Developer`ToPackedArray@N@elementCovalentRadii[[atomicNumbers]];
        max = Max[iatoms] + tol;
        nearFunc = Nearest[icoords -> Automatic];
        bondingNeighbors = 
         Function[atomNum, 
          curAtom = iatoms[[atomNum]];
          near = 
           Select[nearFunc[
             icoords[[atomNum]], {8, curAtom + max}], (# > atomNum) &];
          atomR = curAtom + tol;
          nearR = iatoms[[near]];
          atomP = icoords[[atomNum]];
          nearD = Total[(atomP - #)^2] & /@ icoords[[near]];
          Pick[near, 
           MapThread[#1 < #2 && #1 > min &, {nearD, (atomR + nearR)^2}]]];
        Flatten[
         Thread[# -> bondingNeighbors[#]] & /@ Range[Length[iatoms]]]
    ]
   
NumberRules[list_] := MapThread[Rule, {list, Range@Length@list}];
ElementNamesToNumbers[list_List] := Module[{disp},	
	disp = Association @@ Join[
   		NumberRules[elementShortNames],
   		{
   			(* The non-element VertexTypes that we have are {"R", "R1", "R2", "R3", "R4", "R5", "X"}
   			These can be found in ChemicalIntermediateData. Biomolecule EP table also has some "R" VertexTypes *)
   			"X" -> 899,
   			s_String /; StringTake[s, 1] === "R" :> (900 + If[# === "", 0, ToExpression@#] &@ StringTake[s, {2, -1}])
   		}
   	];
   	
	Developer`ToPackedArray[Replace[list, disp, {1}]]
]


(* ::Subsection::Closed:: *)
(*getCallouts*)



getCallouts2[pts_?MatrixQ, labels_, graphicsOpts_:{},
		calloutOpts_:{LeaderSize->25,Background->None}] := Block[
	{lpp3d, callouts},
	callouts = MapThread[
		Callout[ #1, #2, calloutOpts]&,
		{pts, labels}
	];
	ListPointPlot3D[ callouts, graphicsOpts ] // Cases[#,_Inset,Infinity] &
]


elementShortNames = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", 
	"Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", 
	"Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", 
	"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", 
	"Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", 
	"Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", 
	"Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
	"Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md","No", "Lr", 
	"Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", 
	"Lv", "Ts", "Og"};

elementCovalentRadii = Developer`ToPackedArray[
	{37, 32, 134, 90, 82, 77, 75, 73, 71, 69, 154, 130, 118, 111, 106, 
	102, 99, 97, 196, 174, 144, 136, 125, 127, 139, 125, 126, 121, 138, 
	131, 126, 122, 119, 116, 114, 110, 211, 192, 162, 148, 137, 145, 156, 
	126, 135, 131, 153, 148, 144, 141, 138, 135, 133, 130, 225, 198, 169, 
	185, 185, 185, 185, 185, 185, 180, 175, 175, 175, 175, 175, 175, 160,
	150, 138, 146, 159, 128, 137, 128, 144, 149, 148, 147, 146, 190, 150,
	145, 175, 215, 195, 180, 180, 175, 175, 175, 175, 200, 200, 200, 200,
	200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
	200, 200, 200, 200, 200, 37}];
	
	
NumbersToElementShortNames[list_List] := Module[{disp},
	disp = Association @@ Join[
		Reverse/@NumberRules[elementShortNames],
		{
			899->"X", 900 -> "R",
			x_Integer /; x > 899 :> ("R" <> StringJoin[Table["'", {x - 900}]])
			(*895->"A", 896->"Z", 897->"Z1", 898->"Z2", 899->"X", 900 -> "R",
			901 -> "OH", 902 -> "HO", 903 -> "CHO", 904 -> "CH2OH", 905 -> "HOCH2",
			x_Integer /; x > 899 :> ("R" <> StringJoin[Table["'", {x - 900}]])*)
		}
	];
	list/.disp
];

viewPointFromCoords[x_] := Module[{x0, x2, m, v},
   x0 = Map[Mean, Transpose[x]];
   x2 = Map[# - x0 &, x];
   m = Transpose[x2].x2;
   v = Eigenvectors[m, 3];
   If[! ListQ[v], v = DiagonalMatrix[{1, 1, 1}]];
   v[[3]]*3.38];
   
   


(* ::Subsection::Closed:: *)
(*sufficientlyNonParallelQ*)


sufficientlyNonParallelQ[v1_,v2_]:=Abs[v1.v2/(Norm[v1] Norm[v2])] < 0.9;


(* ::Subsection::Closed:: *)
(*boundingSphere*)


boundingSphere[{{_,Sphere[{1},rad_]}},{coord_}] := Sphere[ coord, rad * 1.2]

boundingSphere[spheres_, coords_] := Module[
	{margin, res},
	margin = If[
		Length @ spheres > 0,
		Max[Cases[ spheres, Sphere[ _,rad_]:> rad, Infinity]],
		0
	];
	res = BoundingRegion[coords, "MinBall"];
	Replace[
		res,
		{
			Ball[center_, radius_] :> Sphere[ center, radius + margin],
			_ :> False
		}
	]
]


(* ::Subsection::Closed:: *)
(*fixPlotThemeOptions*)


fixTix = Function[t, 
  Charting`FindScaledTicks[t[##], {Identity, Identity}] &
];


fix3DPlotThemeOptions[opts_] := fix3DPlotThemeOptions[opts] = ReplaceAll[
	opts,
	{
		HoldPattern[ Ticks -> ticks_?Visualization`Utilities`TicksQ] :>
			Rule[ Ticks, fixTix /@ Visualization`Utilities`TicksExpand3D[ticks] ],
		x:Function[_Charting`SimpleTicks] :> fixTix @ x
	}
]

fixPlotThemeOptions[opts_] := fixPlotThemeOptions[opts] = ReplaceAll[
	opts,
	{
		HoldPattern[ Ticks -> ticks_?Visualization`Utilities`TicksQ] :>
			Rule[ Ticks, fixTix /@ Visualization`Utilities`TicksExpand[ticks] ],
		x:Function[_Charting`SimpleTicks] :> fixTix @ x
	}
]

fixHydrogenOption[opts_] := ReplaceAll[
	opts,
	HoldPattern[Rule[Method, method_List]] :> Sequence[
		Apply[Sequence,
			FilterRules[method, "ShowHydrogens"] /. {
				"ShowHydrogens" -> IncludeHydrogens
			}
		],
		Method -> FilterRules[method, Except["ShowHydrogens"]]
	]
]



(* ::Section::Closed:: *)
(*PlotLegends*)



legendedMoleculePlot[ {plotFunc_,iplotFunc_}, args_, opts_] := Module[
	{plot,inputHighlights, evaluatedHighlightss, highlightStyles, legendsInput,
		 legend, place, atomNames, colorList, customLegend, lgFuncs, legendFunction},
	
	
	{plot,{inputHighlights,highlightStyles,evaluatedHighlightss,colorList,atomNames}} = 
		getPlotAndReapLegendData[ iplotFunc, args, opts];
	
	legendsInput = OptionValue[plotFunc, opts, PlotLegends];
	
	If[ 
		Or[
			UnsameQ @@ (Length /@ {inputHighlights,highlightStyles,evaluatedHighlightss}), 
			MatchQ[
				legendsInput,
				None | _[None,___]
			]
		]
		,
		Return[plot, Module]
	];
	highlightStyles = getDefaultPlotStyle /@ highlightStyles;
	
	
	
	
	place = After;
	legend = Replace[ 
		legendsInput, 
		Placed[ leg_, pl__] :> (place = pl;leg)
	];


	lgFuncs = LineLegend | SwatchLegend | PointLegend;
	customLegend = False;
	legend	= Replace[
		legend,
		(lg:lgFuncs)[arg_,other___?OptionQ] :> (
			customLegend = True;
			legendFunction = lg[#1,#2, other]&;
			arg
		)
	];

	
	legend = Replace[
		legend,
		{
			"Expressions" :> getMolPlotLegend[ plotFunc, HoldForm /@ inputHighlights, highlightStyles],
			"Atoms" :> getAutoPlotLegend[ plotFunc, colorList, atomNames],
			Automatic :> If[
				Length @ inputHighlights === 0
				,
				getAutoPlotLegend[ plotFunc, colorList, atomNames]
				,
				getMolPlotLegend[ plotFunc, Automatic, highlightStyles]
			],
			leg_List :> getMolPlotLegend[ plotFunc, HoldForm /@ leg, highlightStyles],
			else_ :> else
		}
	];
	
	If[
		customLegend
		,
		legend = legendFunction @@ legend
	];
	
	legend = Replace[
		legend,
		lgFuncs[{},___] :> Null
	];
	
	Legended[ 
		plot,
		Replace[
			legend,
			Except[ Null] :> Placed[
				legend,
				place
			]
		]
	]
]

getMolPlotLegend[ _, {}, {}] := Null

getMolPlotLegend[ plotFunc_, labels_, highlightStyles_] := Module[
	{marker = Switch[ plotFunc, MoleculePlot, "Bubble", _, "SphereBubble"]},
	
	SwatchLegend[
		highlightStyles, labels, LegendMarkers -> marker
	]
]


getAutoPlotLegend[ MoleculePlot, colorList_, atomNames_ ] := LineLegend @@ Thread[sortAtomsList @ DeleteDuplicates[Thread[{colorList,atomNames}]]]
getAutoPlotLegend[ MoleculePlot3D, colorList_, atomNames_ ] := SwatchLegend[ 
	Sequence @@ Thread[sortAtomsList @ DeleteDuplicates[Thread[{colorList,atomNames}]]],
	LegendMarkers -> "SphereBubble"
]

sortAtomsList[ list_] := SortBy[
	list,
	AtomicNumber[Capitalize[ #[[2]] ] ]&
]
	


getPlotAndReapLegendData[ plotFunc_, args_, opts_] := Module[
	{res},
	res = Reap[
		plotFunc[ args, Append[FilterRules[ opts, Except[ PlotLegends]], PlotLegends -> None ]],
		{"InputHighlights", "HighlightStyles", "EvaluatedHighlights", "ColorList", "AtomNames"}
	];
	
	(* 2-argument Reap with a list second argument adds an extra list to all returns *)
	res[[2]] = Replace[
		res[[2]],
		{x_List} :> x,
		{1}
	];
	res
	
]



(* ::Section::Closed:: *)
(*PlotThemes*)


(* ::Subsection::Closed:: *)
(*$atomColorRules*)


(* 
	The 2D diagrams use different colors and I don't know if we
	want to unify them or not.  For 2D we need significantly darker 
	colors since they are used for text.
*)
$atomColorRules := $atomColorRules = Dispatch[{
	"H" -> RGBColor[0.433333, 0.466667, 0.466667],
	"He" -> RGBColor[0.371872, 0.444444, 0.444444],
	"Li" -> RGBColor[0.532957, 0.362381, 0.665039],
	"Be" -> RGBColor[0.51371, 0.642873, 0.0294906],
	"B" -> RGBColor[0.666667, 0.473203, 0.473203],
	"C" -> RGBColor[0.4, 0.4, 0.4],
	"N" -> RGBColor[0.291989, 0.437977, 0.888609],
	"O" -> RGBColor[0.800498, 0.201504, 0.192061],
	"F" -> RGBColor[0.385641, 0.57026, 0.27257],
	"Ne" -> RGBColor[0.451509, 0.618949, 0.636858],
	"Na" -> RGBColor[0.658708, 0.492173, 0.842842],
	"Mg" -> RGBColor[0.418849, 0.567035, 0.0521821],
	"Al" -> RGBColor[0.5942, 0.421269, 0.418266],
	"Si" -> RGBColor[0.627451, 0.522876, 0.418301],
	"P" -> RGBColor[1., 0.501961, 0],
	"S" -> RGBColor[0.602953, 0.646767, 0.0900267],
	"Cl" -> RGBColor[0.275132, 0.621793, 0.110932],
	"Ar" -> RGBColor[0.364092, 0.562829, 0.594728],
	"K" -> RGBColor[0.534026, 0.420729, 0.705621],
	"Ca" -> RGBColor[0.480072, 0.744591, 0.0955222],
	"Sc" -> RGBColor[0.400872, 0.400872, 0.400872],
	"Ti" -> RGBColor[0.499347, 0.507189, 0.520261],
	"V" -> RGBColor[0.433987, 0.433987, 0.447059],
	"Cr" -> RGBColor[0.360784, 0.4, 0.520261],
	"Mn" -> RGBColor[0.611765, 0.478431, 0.780392],
	"Fe" -> RGBColor[0.878431, 0.4, 0.2],
	"Co" -> RGBColor[0.627451, 0.376471, 0.418301],
	"Ni" -> RGBColor[0.20915, 0.543791, 0.20915],
	"Cu" -> RGBColor[0.784314, 0.501961, 0.2],
	"Zn" -> RGBColor[0.490196, 0.501961, 0.690196],
	"Ga" -> RGBColor[0.533838, 0.361777, 0.355675],
	"Ge" -> RGBColor[0.403387, 0.421643, 0.384326],
	"As" -> RGBColor[0.494117, 0.334641, 0.593464],
	"Se" -> RGBColor[0.611499, 0.438555, 0.0471085],
	"Br" -> RGBColor[0.58847, 0.22163, 0.16064],
	"Kr" -> RGBColor[0.284013, 0.498308, 0.540275],
	"Rb" -> RGBColor[0.425391, 0.329242, 0.585895],
	"Sr" -> RGBColor[0.325959, 0.646423, 0.095983],
	"Y" -> RGBColor[0.354009, 0.666667, 0.666667],
	"Zr" -> RGBColor[0.305733, 0.611644, 0.612382],
	"Nb" -> RGBColor[0.256691, 0.556569, 0.561121],
	"Mo" -> RGBColor[0.206883, 0.501442, 0.512882],
	"Tc" -> RGBColor[0.234466, 0.669394, 0.701499],
	"Ru" -> RGBColor[0.157459, 0.586546, 0.638209],
	"Rh" -> RGBColor[0.0793033, 0.50362, 0.579453],
	"Pd" -> RGBColor[0., 0.420615, 0.525231],
	"Ag" -> RGBColor[0.501961, 0.501961, 0.501961],
	"Cd" -> RGBColor[0.666667, 0.56732, 0.373856],
	"In" -> RGBColor[0.728371, 0.440594, 0.422196],
	"Sn" -> RGBColor[0.39799, 0.491477, 0.495586],
	"Sb" -> RGBColor[0.619608, 0.388235, 0.709804],
	"Te" -> RGBColor[0.816706, 0.451332, 0.0100947],
	"I" -> RGBColor[0.580392, 0, 0.580392],
	"Xe" -> RGBColor[0.316906, 0.638078, 0.710252],
	"Cs" -> RGBColor[0.332803, 0.217712, 0.483666],
	"Ba" -> RGBColor[0.165935, 0.55605, 0.0796556],
	"La" -> RGBColor[0.618723, 0.477383, 0.219618],
	"Ce" -> RGBColor[0.596549, 0.487616, 0.216754],
	"Pr" -> RGBColor[0.57682, 0.471999, 0.210174],
	"Nd" -> RGBColor[0.558557, 0.441983, 0.20109],
	"Pm" -> RGBColor[0.541328, 0.405239, 0.190417],
	"Sm" -> RGBColor[0.787563, 0.549894, 0.268279],
	"Eu" -> RGBColor[0.764628, 0.493261, 0.250405],
	"Gd" -> RGBColor[0.743177, 0.440115, 0.23269],
	"Tb" -> RGBColor[0.72281, 0.39143, 0.215783],
	"Dy" -> RGBColor[0.702434, 0.347663, 0.200392],
	"Ho" -> RGBColor[0.679962, 0.309234, 0.187368],
	"Er" -> RGBColor[0.652012, 0.276823, 0.17779],
	"Tm" -> RGBColor[0.613603, 0.251489, 0.173042],
	"Yb" -> RGBColor[0.557855, 0.234598, 0.17489],
	"Lu" -> RGBColor[0.475685, 0.227573, 0.18555],
	"Hf" -> RGBColor[0.521025, 0.478259, 0.477719],
	"Ta" -> RGBColor[0.489629, 0.362993, 0.455647],
	"W" -> RGBColor[0.681179, 0.360409, 0.63675],
	"Re" -> RGBColor[0.605181, 0.367584, 0.556343],
	"Os" -> RGBColor[0.521806, 0.382125, 0.469204],
	"Ir" -> RGBColor[0.445624, 0.373159, 0.399069],
	"Pt" -> RGBColor[0.543791, 0.543791, 0.585621],
	"Au" -> RGBColor[0.666667, 0.546405, 0.0915033],
	"Hg" -> RGBColor[0.481046, 0.481046, 0.543791],
	"Tl" -> RGBColor[0.65098, 0.329412, 0.301961],
	"Pb" -> RGBColor[0.341176, 0.34902, 0.380392],
	"Bi" -> RGBColor[0.619608, 0.309804, 0.709804],
	"Po" -> RGBColor[0.670588, 0.360784, 0],
	"At" -> RGBColor[0.458824, 0.309804, 0.270588],
	"Rn" -> RGBColor[0.218799, 0.516091, 0.591608],
	"Fr" -> RGBColor[0.25626, 0.0861372, 0.398932],
	"Ra" -> RGBColor[0., 0.473472, 0.04654],
	"Ac" -> RGBColor[0.214695, 0.477953, 0.658986],
	"Th" -> RGBColor[0.240533, 0.447773, 0.628669],
	"Pa" -> RGBColor[0.397469, 0.628, 0.898853],
	"U" -> RGBColor[0.43205, 0.58595, 0.856029],
	"Np" -> RGBColor[0.464542, 0.54551, 0.814532],
	"Pu" -> RGBColor[0.494945, 0.506679, 0.774361],
	"Am" -> RGBColor[0.52326, 0.469458, 0.735517],
	"Cm" -> RGBColor[0.549486, 0.433847, 0.697999],
	"Bk" -> RGBColor[0.573624, 0.399845, 0.661808],
	"Cf" -> RGBColor[0.595673, 0.367454, 0.626942],
	"Es" -> RGBColor[0.615633, 0.336672, 0.593404],
	"Fm" -> RGBColor[0.633505, 0.307499, 0.561191],
	"Md" -> RGBColor[0.649288, 0.279937, 0.530305],
	"No" -> RGBColor[0.662982, 0.253984, 0.500746],
	"Lr" -> RGBColor[0.674588, 0.22964, 0.472513],
	"Rf" -> RGBColor[0.684106, 0.206907, 0.445606],
	"Db" -> RGBColor[0.691534, 0.185783, 0.420025],
	"Sg" -> RGBColor[0.696874, 0.166269, 0.395772],
	"Bh" -> RGBColor[0.700126, 0.148365, 0.372844],
	"Hs" -> RGBColor[0.701289, 0.13207, 0.351243],
	"Mt" -> RGBColor[0.700363, 0.117385, 0.330968],
	"Ds" -> RGBColor[0.697348, 0.10431, 0.31202],
	"Rg" -> RGBColor[0.692245, 0.0928444, 0.294398],
	"Cn" -> RGBColor[0.685054, 0.0829886, 0.278102],
	"Nh" -> RGBColor[0.675773, 0.0747426, 0.263133],
	"Fl" -> RGBColor[0.664405, 0.0681063, 0.249491],
	"Mc" -> RGBColor[0.650947, 0.0630797, 0.237174],
	"Lv" -> RGBColor[0.635401, 0.0596628, 0.226184],
	"Ts" -> RGBColor[0.635401, 0.056628, 0.226184],
	"Og" -> RGBColor[0.635401, 0.0528, 0.226184],
	"D" -> RGBColor[0.433333, 0.466667, 0.466667],
	_String -> RGBColor[0.4, 0.4, 0.4]
} ];

(* Maybe ColorData is fast enough I shouldn't be caching these?*)

$atomColorRules3D := $atomColorRules3D = Dispatch @ Join[
	# -> System`ColorData["Atoms"][#] & /@ elementShortNames,
	{_String -> RGBColor[0.4, 0.4, 0.4]}
]



(* ::Subsection::Closed:: *)
(*$atomSizeRules*)


$atomSizeRules3D := $atomSizeRules = Dispatch[{
	"H" -> 1.2, "He" -> 1.4, "Li" -> 2.2, "Be" -> 1.9,
	"B" -> 1.8, "C" -> 1.7, "N" -> 1.6, "O" -> 1.55, "F" -> 1.5, "Ne" -> 1.54, "Na" -> 2.4,
	"Mg" -> 2.2, "Al" -> 2.1, "Si" -> 2.1, "P" -> 1.95, "S" -> 1.8, "Cl" -> 1.8, "Ar" -> 1.88,
	"K" -> 2.8, "Ca" -> 2.4, "Sc" -> 2.3, "Ti" -> 2.15,
	"V" -> 2.05, "Cr" -> 2.05, "Mn" -> 2.05, "Fe" -> 2.05, "Co" -> 2., "Ni" -> 2., "Cu" -> 2.,
	"Zn" -> 2.1, "Ga" -> 2.1, "Ge" -> 2.1, "As" -> 2.05, "Se" -> 1.9,
	"Br" -> 1.9, "Kr" -> 2.02, "Rb" -> 2.9, "Sr" -> 2.55,
	"Y" -> 2.4, "Zr" -> 2.3, "Nb" -> 2.15, "Mo" -> 2.1, "Tc" -> 2.05,
	"Ru" -> 2.05, "Rh" -> 2., "Pd" -> 2.05, "Ag" -> 2.1, "Cd" -> 2.2, "In" -> 2.2, "Sn" -> 2.25,
	"Sb" -> 2.2, "Te" -> 2.1, "I" -> 2.1, "Xe" -> 2.16, "Cs" -> 3., "Ba" -> 2.7, "La" -> 2.5,
	"Ce" -> 2.48, "Pr" -> 2.47, "Nd" -> 2.45, "Pm" -> 2.43, "Sm" -> 2.42, "Eu" -> 2.4,
	"Gd" -> 2.38, "Tb" -> 2.37, "Dy" -> 2.35, "Ho" -> 2.33, "Er" -> 2.32, "Tm" -> 2.3,
	"Yb" -> 2.28, "Lu" -> 2.27, "Hf" -> 2.25, "Ta" -> 2.2, "W" -> 2.1,
	"Re" -> 2.05, "Os" -> 2., "Ir" -> 2., "Pt" -> 2.05, "Au" -> 2.1, "Hg" -> 2.05, "Tl" -> 2.2,
	"Pb" -> 2.3, "Bi" -> 2.3, "Po" -> 2., "At" -> 2., "Rn" -> 2.,
	"Fr" -> 2., "Ra" -> 2., "Ac" -> 2., "Th" -> 2.4, "Pa" -> 2., "U" -> 2.3,
	"Np" -> 2., "Pu" -> 2., "Am" -> 2., "Cm" -> 2., "Bk" -> 2., "Cf" -> 2., "Es" -> 2.,
	"Fm" -> 2., "Md" -> 2., "No" -> 2., "Lr" -> 2., "Rf" -> 2., "Db" -> 2., "Sg" -> 2.,
	"Bh" -> 2., "Hs" -> 2., "Mt" -> 2., "Ds" -> 2., "Rg" -> 2., "Cn" -> 2., "Nh" -> 2.,
	"Fl" -> 2., "Mc" -> 2., "Lv" -> 2., "Ts" -> 2., "Og" -> 2., "Uue" -> 1.2, _String -> 1.2
}];


End[] (* End Private Context *)

