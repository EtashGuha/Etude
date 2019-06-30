


Begin["Chemistry`Private`FormattingDump`"]


$moleculeBoxProperties = {"MolecularFormula", "AtomCount", "FullAtomCount", "BondCount", "FullBondCount","SMILES","InChIKey"};

expandablePane2[x_] = Pane[x, ImageSize -> {{1, 300}, Automatic}, BaselinePosition->Baseline, ContentPadding->False, FrameMargins->0, StripOnInput->True]

moleculeBox[mol_, fmt_] := Module[
	{
		formula, atomCount, fullAtomCount, bondCount, fullBondCount, smiles, inchikey,
		alwaysGrid, sometimesGrid, icon, format, properties
	},
	format = Lookup[ 
		Lookup[ Options[mol], Method, {}],
		"Format",
		True
	];
	
	icon = If[ 
		format
		,
		Replace[
			TimeConstrained[ molIcon[mol], 1.5],
			Except[ _Graphics | _Pane] :> genericIcon[]
		]
		,
		genericIcon[]
	];
	properties = Quiet[ MoleculeValue[ mol, $moleculeBoxProperties]];
	If[ 
		Length @ properties =!= Length @ $moleculeBoxProperties
		, 
		Return[ $Failed, Module]
		,
		{formula, atomCount, fullAtomCount, bondCount, fullBondCount, smiles, inchikey} = properties;
	];
	formula = If[
		MatchQ[ formula, _Row | _String | _Subscript | _Superscript]
		,
		{ BoxForm`SummaryItem[{"Formula: ", formula}](*, SpanFromLeft*)}
		,
		Nothing
	];
	
	If[
		fullAtomCount =!= atomCount
		,
		atomCount = Sequence[ atomCount, StringJoin["\[ThinSpace]","(",IntegerString @ fullAtomCount,")"]];
	];
	atomCount = BoxForm`SummaryItem[{"Atoms: ", atomCount}];
	
	If[
		fullBondCount =!= bondCount
		,
		bondCount = Sequence[ bondCount, Row[{"\[ThinSpace]","(",fullBondCount,")"}]]; (*Row[{ bondCount, "\[ThinSpace]", Style[ ,GrayLevel[.5]] }];*)
	];
	bondCount = BoxForm`SummaryItem[{"Bonds: ", bondCount}];
	
	alwaysGrid =  {formula, {Row[ {atomCount,"  ", bondCount} ]}};

	smiles = If[ 
		StringQ[smiles] && StringLength[smiles] > 0
		, 
		{BoxForm`SummaryItem[{"SMILES: ",expandablePane2[smiles]}](* , SpanFromLeft*)}
		,
		Nothing
	];
	
	inchikey = If[ 
		StringQ[inchikey] && StringLength[inchikey] > 0
		, 
		{BoxForm`SummaryItem[{"InChIKey: ",inchikey}](* , SpanFromLeft*)}
		,
		Nothing
	];
	
	sometimesGrid = {smiles, inchikey};
	BoxForm`ArrangeSummaryBox[Molecule, mol, icon, alwaysGrid, sometimesGrid, fmt, "CompleteReplacement" -> False]

	]
	
moleculeBox[___]:= $Failed

$molIconPlotFactor = 4.1 (* just trial and error *)

genericIcon[] := Graphics[
	{
		GraphicsComplex[
			{
				{-100., -1.1102230246251565*^-14},
				{-50.00000000000004, -86.60254037844388},
				{-50.00000000000002, 86.60254037844385},
				{50.00000000000004, -86.60254037844388},
				{100., -1.1102230246251565*^-14},
				{50., 86.60254037844385}
			},
			{
				{
					GrayLevel @ 0.5,
					Disk[{-100., -1.1102230246251565*^-14}, 25.],
					Disk[{-50.00000000000004, -86.60254037844388}, 25.],
					Disk[{50.00000000000004, -86.60254037844388}, 25.],
					Disk[{100., -1.1102230246251565*^-14}, 25.],
					Disk[{50., 86.60254037844385}, 25.],
					Disk[{-50.00000000000002, 86.60254037844385}, 25.]
				},
				{
					CapForm @ "Round",
					{GrayLevel @ 0.5, Line @ {{1, 2}}},
					{GrayLevel @ 0.5, Line @ {1, 3}},
					{GrayLevel @ 0.5, Line @ {2, 4}},
					{GrayLevel @ 0.5, Line @ {4, 5}},
					{GrayLevel @ 0.5, Line @ {5, 6}},
					{GrayLevel @ 0.5, Line @ {6, 3}}
				}
			}
		],
		{}
	},
	Axes -> False,
	Background -> GrayLevel[0.93],
	Frame -> True,
	FrameStyle -> Directive[
		Opacity[0.5], Thickness @ Tiny, RGBColor[0.368417, 0.506779, 0.709798]
	],
	FrameTicks -> None,
	ImageSize -> {
		Automatic,
		$molIconPlotFactor * Dynamic[ CurrentValue["FontCapHeight"]]
	},
	PlotRangePadding -> 10
]  


SetAttributes[
    {
        moleculeBox
    },
    {ReadProtected, Protected}
]

End[] (* End Private Context *)

