


Begin["Chemistry`Private`MoleculeDump`"]



(* ::Subsection::Closed:: *)
(*Patterns*)


$AdjustHydrogens = True;

$NonEvaluatingOptionsPattern = 
	opts:OptionsPattern[] /; FreeQ[
		{opts}, 
		"R"|"S"| Rule[ IncludeHydrogens, _]
	];


$MoleculePattern = With[
	{ap = $AtomsPattern, bp = {$BondPattern...}, op = $NonEvaluatingOptionsPattern},
	HoldPattern[ Molecule[ap, bp, op ] ]
]

$StereoPattern1 = KeyValuePattern[
	{
		"StereoType" -> "Tetrahedral", 
		"ChiralCenter" -> _Integer, "Direction" -> _String, 
		"FiducialAtom" -> _Integer, "Ligands" -> {__Integer}
	}
]

$StereoPattern2 = st:KeyValuePattern[
	{
		"StereoType" -> "Tetrahedral", 
		"ChiralCenter" -> _Integer, 
		"Direction" -> "Clockwise"|"Counterclockwise"
	}
] /; Length[st] === 3

$StereoPattern3 = KeyValuePattern[
	{
		"StereoType" -> "DoubleBond", 
		"StereoBond" -> {_Integer, _Integer},
		"Ligands" -> {_Integer, _Integer},
		"Value" -> _String
	}
]

$StereoPattern = $StereoPattern1 | $StereoPattern2 | $StereoPattern3


$ValidMoleculeLayouts = {Automatic, _String, "RDKit", "CoordGen"}


addCompletion[ "Bond" -> {0,{"Single","Double","Triple","Aromatic"}}];

(* ::Section::Closed:: *)
(*MoleculeQ*)


Protect[$MoleculePickle];

(*
	a valid Molecule will have passed through the single Molecule down value
	and a byte array will have been created and cached
*)

MoleculeQ[  mol:Molecule[atoms_,bonds_,opts___] ] := System`Private`ValidQ[Unevaluated[mol]]

MoleculeQ[___] := False


(* ::Section::Closed:: *)
(*Molecule DownValue*)


Options[Molecule] = Options[ LoadMol ] = {
	IncludeHydrogens -> Automatic,
	IncludeAromaticBonds -> True,
	StereochemistryElements -> None,
	AtomDiagramCoordinates -> Automatic,
	AtomCoordinates -> Automatic(*{RandomSeeding -> 1234, Method -> Automatic}*),
	MetaInformation -> <||>
}


$hiddenMoleculeOptions = {
	"SanitizationOptions" -> All,
	"AllowValenceErrors" -> False
	(*"IncludeCoordinates" -> False*)
}


mol:Molecule[args___] /; !System`Private`ValidQ[Unevaluated[mol]] := Module[
	{canonical = Catch[CanonicalizeMolecule[ args], $tag]},
	canonical /; System`Private`ValidQ[canonical]
]


(* ::Subsection::Closed:: *)
(*Molecule SubValue*)


(molecule_Molecule)[prop_String, val__Rule] /; MoleculeQ[ molecule ] := With[
	{return = MoleculeValue[ molecule, MoleculeProperty[ prop, {val} ] ]},
	
	return /; !MatchQ[return, _MoleculeValue]
]

(molecule_Molecule)[args__] /; MoleculeQ[ molecule ] := With[
	{return = MoleculeValue[ molecule, args ]},
	return /; !MatchQ[return, _MoleculeValue]
]


(* ::Subsection::Closed:: *)
(*Molecule format value*)



Molecule /: MakeBoxes[mol_Molecule, fmt_] /; MoleculeQ[Unevaluated[mol]] := Module[
	{res},
	res = moleculeBox[mol, fmt];
	res /; (res =!= $Failed)
]





(* ::Subsection::Closed:: *)
(*Auxilliary Functions*)


validMolecule[args___] := System`Private`SetValid[Unevaluated[ Molecule[ args] ] ]

moleculeOptionValue[ opts_, opt_] := OptionValue[ Molecule, FilterRules[opts, Options[Molecule]], opt]
moleculeOptionValue[ opts_, opt:(Alternatives @@ Keys[$hiddenMoleculeOptions])] := Lookup[ $hiddenMoleculeOptions, opt]


(* ::Section::Closed:: *)
(*Load Molecule*)


CanonicalizeMolecule[args:PatternSequence[atms:$AtomsPattern, bnds:{$BondPattern...}, opts:OptionsPattern[]]] := Module[
	{res, libraryID, molData},
	
	If[
		IntegerQ[libraryID = cachedLibraryID[args]],
		res = validMolecule[args];
		cacheMolecule[ res, libraryID];
		Throw[res, $tag]
	];
	
	molData["ValidQ"] = False;
	molData["Modified"] = False;
	
	parseMolecule[ {args}, molData];
	validateMolecule[{args}, molData];
	loadMoleculeData[molData];
	
	Which[ 
		!molData["ValidQ"], 
			$Failed,
		molData["Modified"] && correctStereoAssignments[molData],
			CanonicalizeMolecule[molData["imol"], molData["Options"]],
		True,
			res = validMolecule[args];
			cacheMolecule[ res, molData];
			res
	]
	
]

correctStereoAssignments[molData_ /; AssociationQ[ molData["CheckStereo"]] ] := Module[
	{check, centers = Keys @ molData["CheckStereo"], vals = Values @ molData["CheckStereo"]},
	check = molData["imol"][ "rdkitAtomProperty", #, "_CIPCode"] & /@ centers;
	If[
		check === vals
		,
		True
		,
		Message[ Molecule::inpt];
		Throw[$Failed, $tag]
	]
	
]

correctStereoAssignments[___] := True;

LoadMoleculeData[args:PatternSequence[atms:$AtomsPattern, bnds:{$BondPattern...}, opts:OptionsPattern[]]] := Catch[Module[
	{res, molData},
	
	
	parseMolecule[ {atms,bnds,opts}, molData];
	validateMolecule[{atms,bnds,opts}, molData];
	loadMoleculeData[molData];
	
	If[
		TrueQ @ molData["ValidQ"]
		,
		molData["imol"]
		,
		res = $Failed
	]
], $tag]


(* ::Subsection::Closed:: *)
(*parseMolecule*)


parseMolecule[ {atms_, bnds_, opts:OptionsPattern[]}, molData_Symbol ] := Module[
	{options},
	options = Flatten[{opts}];
	parseAtoms[ atms, molData];
	parseBonds[ bnds, molData];
	parseOptions[ options, molData];
	parseCoordinates[ options, molData];
	parseStereo[ options, molData];
	parseMetaInformation[ options, molData];
	
]


(* ::Subsubsection::Closed:: *)
(*parseAtoms*)


parseAtoms[atms_List, molData_Symbol] := Module[
	{atomData},
	molData["AtomCount"] = Length @ atms;
	molData["AtomIndices"] = Range @ molData["AtomCount"];
	molData["AtomicNumbers"] = AtomicNumber[ atms ];
	molData["HasExplicitHydrogens"] = MemberQ[ molData["AtomicNumbers"], 1];
	
	atomData = getAtomData /@ atms;
	
	molData["AtomArray"] = atomData;
	molData["AtomNames"] = If[
		FreeQ[atms, "Name"],
		{},
		Replace[
			Thread[ { molData["AtomIndices"], #["Name"] &/@ atms}],
			{ _, None} -> Nothing,
			1
		]
	];
]

atomDataLookup[rul_,val_]:= Lookup[rul,val,0]
getAtomData[Atom[c_String]] := {AtomicNumber[c], 0, 0, 0, 0};
getAtomData[atm:Atom[c_String, rules__Rule]] := {
	AtomicNumber[c],
	getFormalCharge[atm],
	getMassNumber[atm],
	getUnpairedElectronCount[atm],
	getHydrogenCount[atm]
};

getFormalCharge[atm:Atom[c_String, rules__Rule]] := Replace[
	atomDataLookup[{rules},"FormalCharge"],
	Except[ x_Integer /; Between[x, {-20,20}]] :> (Message[Molecule::atom,atm];Throw[$Failed,$tag])
];

getMassNumber[atm:Atom[c_String, rules__Rule]] := Replace[
	atomDataLookup[{rules},"MassNumber"],
	{
		Automatic -> 0,
		Except[  x_Integer /; x >= 0] :> (Message[Molecule::atom,atm];Throw[$Failed,$tag])
	}
];

getUnpairedElectronCount[atm:Atom[c_String, rules__Rule]] := Replace[
	atomDataLookup[{rules},"UnpairedElectronCount"],
	Except[ x_Integer /; Between[x, {0,20}]] :> (Message[Molecule::atom,atm];Throw[$Failed,$tag])
];

getHydrogenCount[atm:Atom[c_String, rules__Rule]] := Replace[
	atomDataLookup[{rules},"HydrogenCount"],
	Except[ x_Integer /; Between[x, {0,10}]] :> (Message[Molecule::atom,atm];Throw[$Failed,$tag])
];



(* ::Subsubsection::Closed:: *)
(*parseBonds*)


parseBonds[ bonds_List, molData_Symbol] := Module[
	{},
	molData["BondsInput"] = bonds;
	molData["Bonds"] = MapThread[
		Append,
 		{bonds[[All, 1]], bondOrderLookup @ bonds[[All, 2]]}
	 ];
	 molData["HasAromaticBonds"] = MemberQ[ bonds[[All,2]], "Aromatic"];
	
]



(* ::Subsubsection::Closed:: *)
(*parseCoordinates*)


parseCoordinates[ options_List, molData_Symbol] := Module[
	{coords2D, coords3D, modCoord},
	molData["HasCoordinates"] = False;
	coords2D = moleculeOptionValue[options, AtomDiagramCoordinates];
	Switch[coords2D,
		_?MatrixQ,
			molData["HasCoordinates"] = True;
			molData["coords2D"] = Replace[
				coords2D,
				x:_?quantityArrayQ | {__quantity} :> QuantityMagnitude[x]
			],
		Except[Automatic | _?StringQ], 
			Message[ Molecule::coord, AtomDiagramCoordinates, AtomDiagramCoordinates /. options];
			Throw[$Failed,$tag]
	];
			
	
	coords3D =  moleculeOptionValue[ options, AtomCoordinates];
	modCoord = False;
	Which[
		MatchQ[coords3D, Automatic | _?StringQ | {_String, ___Rule} | {__Rule}],
			Null,
		quantityArrayQ[coords3D],
			molData["HasCoordinates"] = True;
			molData["coords3D"] = If[getUnit[coords3D] === "Angstroms",
				QuantityMagnitude[coords3D],
				modCoord = True;
				QuantityMagnitude[UnitConvert[coords3D,"Angstroms"]]
			],
		MatrixQ[coords3D,QuantityQ],
			molData["HasCoordinates"] = True;
			modCoord = True;
			molData["coords3D"] = If[getUnit[coords3D] === "Angstroms",
				QuantityMagnitude[coords3D],
				QuantityMagnitude[UnitConvert[coords3D,"Angstroms"]]
			],
		MatrixQ[coords3D],
			molData["HasCoordinates"] = True;
			modCoord = True;
			molData["coords3D"] = coords3D,
		
		True, 
			Message[ Molecule::coord, AtomCoordinates, AtomCoordinates /. options];
			Throw[$Failed,$tag]
	];
	If[TrueQ[modCoord] && MatrixQ[molData["coords3D"]] && UnsameQ[ modCoord = QuantityArray[molData["coords3D"], "Angstroms"], coords3D]
		,
		molData["Options"] = Replace[
			molData["Options"],
			HoldPattern[AtomCoordinates -> _] :> AtomCoordinates -> modCoord,
			{1}
		]
	]
];

getUnit[HoldPattern[StructuredArray[QuantityArray,dims_,StructuredData[QuantityArray, numbers_, units_, levels_]]]] := units 
getUnit[HoldPattern[{{Quantity[_,unit_], ___},___}]] := unit; 

(* ::Subsubsection::Closed:: *)
(*parseOptions*)


parseOptions[options_, molData_] := Module[ 
	{(*hasUnspecifiedBonds = MemberQ[molData["Bonds"][[All,3]],0]*)},
	molData["Options"] = options;
	molData["IncludeHydrogens"] = moleculeOptionValue[ molData["Options"], IncludeHydrogens];
	molData["IncludeAromaticBonds"] = moleculeOptionValue[ molData["Options"], IncludeAromaticBonds];
	molData["SanitizationOptions"] = Lookup[ 
		options, "SanitizationOptions", None];
(*		If[
			hasUnspecifiedBonds,
			<| "Properties"->False|>,
			None
		]
	];*)
]



(* ::Subsubsection::Closed:: *)
(*parseStereo*)


parseStereo[ options:KeyValuePattern[StereochemistryElements -> elements_], molData_Symbol] := Module[
	{parsed},
	
	parsed = If[ ListQ[elements], elements, {elements}];
	
	parsed = With[{ast = $AtomStereoTypes, bst = $BondStereoTypes}, 
		Replace[
			parsed, 
			{
				assoc_Association?AssociationQ /; KeyExistsQ[assoc, "StereoType"] :> assoc,
				HoldPattern[n_Integer -> val:ast] :> 
					<| "StereoType" -> "Tetrahedral", "Direction" -> val, "ChiralCenter" -> n |>,
				HoldPattern[{n_Integer,m_Integer} -> val:bst] :> 
					<| "StereoType" -> "DoubleBond", "Value" -> val, "StereoBond" -> {n,m} |>,
				(* Normal[molecule] will destroy an association *)
				rules:{__Rule} /; KeyExistsQ[rules, "StereoType"] :> (
					molData["Modified"] = True;
					Association[rules]
				),
				invalid_ :> (Message[ Molecule::stereo, invalid]; Throw[$Failed, $tag])	
			},
			{1}
		]
	];

	molData["ChiralCenters"] = Internal`Bag[];
	molData["BondStereo"] = Internal`Bag[];
	Scan[
		Switch[ #StereoType,
			"Tetrahedral",
				parseChiralCenter[ #, molData],
			"DoubleBond",
				parseBondStereo[ #, molData],
			_,
				Message[Molecule::strtype, #StereoType];
				Throw[$Failed,$tag];	
		]&,
		parsed
	];
];

parseChiralCenter[
	data:KeyValuePattern[{"ChiralCenter" -> center_Integer, "Direction" -> val_String}], 
	molData_Symbol] := Module[
	{dir, ligands, res},
	
	
	dir = Replace[
		chiralTagLookup @ val,
		bad:Except[ _Integer] :> messageBadStereochemistry[molData, bad]
	];
	
	If[
		MatchQ[ val, "R" | "S"],
		molData["Modified"] = True;
		If[
			AssociationQ[ molData["CheckStereo"]]
			,
			AssociateTo[ molData["CheckStereo"], center -> val]
			,
			molData["CheckStereo"] = <|center -> val|>
		];
		molData["Options"] = FilterRules[ molData["Options"], Except[StereochemistryElements]]
	];
	
	
	res = If[
		KeyFreeQ[ data, "FiducialAtom" | "Ligands"]
		,
		molData["Modified"] = True;
		molData["Options"] = FilterRules[ molData["Options"], Except[StereochemistryElements]];
		{center, dir, data}
		,
		ligands = Replace[
			Flatten @ Lookup[ data,  {"FiducialAtom", "Ligands"} ],
			Except[ {__Integer}] :> messageBadStereochemistry[molData, data]
		];
		Flatten[{center, dir, ligands, data}]
	];

	Internal`StuffBag[
		molData["ChiralCenters"],
		res
	]
]

parseChiralCenter[data_,molData_] := messageBadStereochemistry[molData, data]

parseBondStereo[
	data:KeyValuePattern[{"StereoBond" -> bondedAtoms:{_Integer,_Integer},"Value" -> val_String}], 
	molData_Symbol] := Module[
	{dir, ligands},
	ligands = Replace[
		data["Ligands"], 
		{
			_Missing :> (
				molData["Modified"] = True;
				molData["Options"] = FilterRules[ molData["Options"], Except[StereochemistryElements]];
				Sequence @@ {}
			),
			Except[{_Integer,_Integer}] :> (
			messageBadStereochemistry[molData, data];
				Throw[$Failed, $tag]
			)
		}
	];
	
	dir = Replace[
		bondStereoLookup @ val,
		Except[ _Integer] :> messageBadStereochemistry[molData, data]
	];
	
	Internal`StuffBag[
		molData["BondStereo"],
		Flatten[ {bondedAtoms, dir, ligands, data}]
	]
];

parseBondStereo[data_,molData_] := messageBadStereochemistry[molData, data]




(* ::Subsubsection:: *)
(*parseMetaInformation*)


parseMetaInformation[ options:KeyValuePattern[MetaInformation -> meta:Except[_?AssociationQ]], molData_Symbol] := (
	molData["Options"] = fixMetaInformation[options];
	molData["Modified"] = True;
)

fixMetaInformation[options:KeyValuePattern[MetaInformation -> meta:Except[_?AssociationQ]] ] := Module[
	{res},
	res = Replace[
		meta,
		{
			rule_Rule :> Association[rule],
			rules:{__Rule} :> Association[rules],
			other_ :> Association[ "Untagged" -> other]
		}
	];
	Join[
		FilterRules[ options, Except[ MetaInformation] ],
		{MetaInformation -> res}
	]
]

fixMetaInformation[arg_] := arg;

(* ::Subsection::Closed:: *)
(*validateMolecule*)


validateMolecule[ { atms_, bnds_, opts:OptionsPattern[]}, molData_Symbol ] := Module[
	{options},
	options = Flatten[{opts}];
	validateAtoms[ atms, molData];
	validateBonds[ bnds, molData];
	validateStereo[ options, molData];
	validateCoordinates[ options, molData];
	molData["ValidQ"]=True;
]



(* ::Subsubsection::Closed:: *)
(*validateAtoms*)


validateAtoms[ atms_List, molData_Symbol] := Module[
	{atoms = molData["Atoms"]},
	If[
		MatrixQ[{atms,atoms}] && validAtomicNumber[ molData["AtomicNumbers"] ]
		,
		True,
		Replace[ 
			Thread[{atms, molData["AtomicNumbers"]}],
			{in_,badAtom_?(!validAtomicNumber[#]&)} :> (
				Message[Molecule::atom, in];
				Throw[ $Failed, $tag]
			),
			1
		];
		False
	]
]



(* ::Subsubsection::Closed:: *)
(*validateBonds*)


	
validateBonds[ bnds_List, molData_Symbol] := Module[
	{bonds = molData["Bonds"],res, noSelfBonds, dupFree, ficticiousAtoms, valid},
	res = And[
		valid = MatrixQ[ bonds, Internal`NonNegativeIntegerQ],
		noSelfBonds = FreeQ[ bonds, {x_, x_, _}],
		dupFree = DuplicateFreeQ[List @@@ bonds],
		ficticiousAtoms = VectorQ[ 
			Union @ Flatten[ bonds[[All,;;2]] ], 
			Between[{1,molData["AtomCount"]}]
		]
	];
	
	If[ res, Return[ True, Module]];
	

	If[
		!dupFree,
		Message[Molecule::dupbnd];
		Throw[$Failed, $tag]
	];
	If[
		!noSelfBonds,
		Replace[
			bnds,
			bnd:_[x_,x_] :> (
				Message[Molecule::bond,bnd];
				Throw[$Failed, $tag]
			),
			1
		]
	];
	
	If[!ficticiousAtoms,
		Replace[molData @ "BondsInput",
			RuleDelayed[
				bond:Bond[{a_, b_}, _] /; !ContainsAll[Range @ molData["AtomCount"], {a, b}],
				Message[Molecule::bndvi, molData["AtomCount"], bond];
				Throw[$Failed, $tag]
			],
			1
		];
		Throw[$Failed, $tag];
	]; 
				
	False
]


(* ::Subsubsection::Closed:: *)
(*validateStereo*)


(* TODO: hook this up *)


(* ::Subsubsection::Closed:: *)
(*validateCoordinates*)


validateCoordinates[ options_List, molData_Symbol /; !ValueQ[ {molData["coords2D"], molData["coords3D"]} ] ] := True;
	
validateCoordinates[ options_List, molData_Symbol] := Module[
	{coords2D,coords3D},
	If[
		MatrixQ[coords2D = molData["coords2D"]] && Dimensions[coords2D] =!= {molData["AtomCount"],2}
		,
		Message[ Molecule::coord, AtomDiagramCoordinates, AtomDiagramCoordinates /. options];
		Throw[$Failed, $tag]
	];
	If[
		MatrixQ[coords3D = molData["coords3D"]] && Dimensions[coords3D] =!= {molData["AtomCount"],3}
		,
		Message[ Molecule::coord, AtomCoordinates, AtomCoordinates /. options];
		Throw[$Failed, $tag]
	];
	True
]



(* ::Subsection::Closed:: *)
(*loadMoleculeData*)


loadMoleculeData[ molData_Symbol , kekulize_:True] := Module[
	{imol = newInstance[]},
	molData["imol"] = imol;
	loadAtomsAndBonds[ molData, kekulize];
	loadStereo[ molData["imol"], molData];
	loadCoordinates[ molData["imol"], molData];
	adjustHydrogens[ molData];
	
]



(* ::Subsubsection::Closed:: *)
(*loadAtomsAndBonds*)


loadAtomsAndBonds[ molData_Symbol, kekulize_:True] := Module[
	{},
	
	(*imol["setFlags", OptionValue[ Molecule, Flatten @ {opts},"AllowBadValence"]];*) 
	
	molData["imol"][ "createMolWithAtomsAndBonds", molData["AtomArray"], molData["Bonds"]];

	sanitizeMol[molData["imol"],molData];

	If[
		molData["Bonds"] =!= molData["imol"]["getBondsList"]
		,
		molData["Modified"] = True
	];
	
		
	If[
		MatchQ[ molData["AtomNames"], {{_Integer, _String}..}],
		molData["imol"][ "setNames", Developer`WriteRawJSONString[ molData["AtomNames"]] ]
	];

]



(* ::Subsubsection::Closed:: *)
(*sanitizeMol*)


sanitizeMol[imol_, molData_] := Module[
	{res, flags, nAttempts, messages, toFix, maxAttempts},

	flags = Replace[
		molData["SanitizationOptions"],
		Except[_?AssociationQ] :>  <||>
	];
	Which[
		!TrueQ @ molData["IncludeAromaticBonds"] && TrueQ @ molData["HasAromaticBonds"],
			flags["SetAromaticity"] = False;
		,
		!TrueQ @ molData["IncludeAromaticBonds"],
			flags["SetAromaticity"] = False;
			flags["SetAromaticity"] = False;,
		Count[
			molData["Bonds"],
			{_Integer,_Integer,0}
		] > 0,
			flags["SetAromaticity"] = False;
			flags["FindRadicals"] = False;
			flags["Properties"] = False;	
	];
	
	toFix = True;


	nAttempts = 0;
	messages = Replace[
		molData["ParsingMessage"],
		{
			x_String /; StringMatchQ[x,"non-ring atom"~~__~~"aromatic"] :> nonAromaticMessage[x],
			x_String :> {x},
			_ :> {}
		}
	];
	
	
	maxAttempts = If[ toFix, 4, 1];
	messages = Join[
		messages,
		Flatten @ Last @ Reap @ While[ 
			res =!= Null && nAttempts++ < maxAttempts,
			
			If[
				Length @ flags > 0,
				imol["setFlags", flags]
			];
			res = imol[ "sanitize"];
			Replace[
				res,
				{
					{8, x_String /; StringMatchQ[x,"non-ring atom"~~__~~"aromatic"]} :> nonAromaticMessage[x],
					{8, x_String /; StringMatchQ[x,"non-ring atom"~~__~~"aromatic"]} :> nonAromaticMessage[x],
					{flag_Integer, message_String} :> (
						If[ toFix,updateFlags[ flag, flags]];
						Sow[message]
					)
				}
			]
		]
	];
	
	If[
		MatchQ[ flags, KeyValuePattern[ "Properties" -> False]],
		molData["BadValence"] = True
	];
	
	If[
		res === Null
		,
		If[ Length[messages] > 0, issueValenceMessage[First @ messages] ];
		Return[ Null, Module]
		,
		issueValenceMessage[First @ messages];
		Throw[$Failed, $tag]
	];
	
]

Attributes[updateFlags] = HoldRest;
updateFlags[ 2, flags_] := AssociateTo[ flags, "Properties" -> False];
updateFlags[ 8, flags_] := AssociateTo[ flags, "Kekulize" -> False];

issueValenceMessage[ msg_String] := Replace[
	StringCases[msg,
		{
			RuleDelayed[
				StringExpression["Explicit valence for atom # ",
					index:NumberString, " ", element__, ", ", valence:NumberString, ", is greater than permitted"
				],
				Message[Molecule::valenc,
					FromDigits[index] + 1,
					elementCommonName[element],
					valence
				]
			],
			RuleDelayed[
				StringExpression["Explicit valence for atom # ",
					index:NumberString, " ", element__, " greater than permitted"
				],
				Message[Molecule::valenc,
					FromDigits[index] + 1,
					elementCommonName[element],
					""
				]
			],
			RuleDelayed[
				"Can't kekulize mol.  Unkekulized atoms: " ~~ atoms__ ~~ "\n",
				Message[Molecule::aromat,
					1 + Map[FromDigits, StringSplit[atoms]]
				]
			]
		}
	],
	{
		{} :> Message[Molecule::inpt],
		{{m__}} :> Message[m]
	}
]

nonAromaticMessage[msg_] := (Message[Molecule::aromat2]; Throw[$Failed,$tag]);



(* ::Subsubsection::Closed:: *)
(*adjustHydrogens*)

adjustHydrogens[ molData_Symbol] /; ($AdjustHydrogens && !molData["HasCoordinates"]) := Module[
	{nm},
	molData["HasImplicitHydrogens"] = molData["imol"]["hasImplicitHydrogens"];
	Switch[molData["IncludeHydrogens"],
		Automatic,
			Return[Null, Module],
		True,
			If[
				molData["HasImplicitHydrogens"]
				,
				nm = newInstance[];
				nm[ "createCopyWithAddedHydrogens", ManagedLibraryExpressionID @ molData["imol"], False];
				molData["imol"] =.;
				molData["imol"] = nm;
				molData["Modified"] = True;
				Return[Null, Module]
			],
		False,
			If[
				molData["HasExplicitHydrogens"]
				,
				nm = newInstance[];
				nm[ "createCopyWithNoHydrogens", ManagedLibraryExpressionID @ molData["imol"], True, False];
				molData["imol"] =.;
				molData["imol"] = nm;
				molData["Modified"] = True;
				Return[Null, Module]
			],
		None,
			If[
				molData["HasImplicitHydrogens"]
				,
				molData["imol"][ "convertHydrogensToHairs"];
				molData["Modified"] = True;
				,
				Return[Null, Module]
			]
	];
]



(* ::Subsubsection::Closed:: *)
(*loadStereo*)


loadStereo[ imol_, molData_Symbol] := (
	If[
		MatchQ[ molData["ChiralCenters"], _Internal`Bag ],
		loadChiralCenters[imol, molData];
	];
	If[
		MatchQ[ molData["BondStereo"], _Internal`Bag ],
		loadStereoBonds[imol, molData];
	];
);

(*
	setAtomChirality and setBondStereo are both void return type,
	any non-null return is a caught exception - index out of range,
	no bonds between ligands and central atom, etc.  It's faster to 
	check validity on the library side and return a LibraryFunctionError
	than to check in top level.
*)
loadChiralCenters[imol_, molData_Symbol ] := (
	Do[
		Replace[
			imol[ "setAtomChirality", Most @ center]
			,
			Except[Null] :> messageBadStereochemistry[ molData, Last @ center]
		];
		,
		{center, Internal`BagPart[molData["ChiralCenters"],All]}
	];
	imol[ "assignStereochemistry", (*clean*)False, (*force*)True, (*flagUnknown*)True];
)

loadStereoBonds[imol_, molData_Symbol ] := Do[
	Replace[
		imol[ "setBondStereo", Most @ bondStereo]
		,
		Except[Null] :> messageBadStereochemistry[ molData, Last @ bondStereo]
	];
	,
	{bondStereo, Internal`BagPart[molData["BondStereo"],All]}
];
	
	
(* ::Subsubsection::Closed:: *)
(*messageBadStereochemistry*)


messageBadStereochemistry[ molData_Symbol, stereoData_] := (
	Message[ Molecule::stereo, stereoData];
	Throw[ $Failed, $tag];
)


(* ::Subsubsection::Closed:: *)
(*loadCoordinates*)


loadCoordinates[ imol_, molData_Symbol] := (
	loadcoords2d[imol, molData];
	loadcoords3d[imol, molData];
)

loadcoords2d[imol_, molData_Symbol /; ValueQ[ molData["coords2D"] ] ] := 
	imol[ "addCoordinates", molData["coords2D"]];
	
loadcoords3d[imol_, molData_Symbol /; ValueQ[ molData["coords3D"] ] ] := 
	imol[ "addCoordinates", molData["coords3D"]]



(* ::Section::Closed:: *)
(*Constructors*)


(* ::Subsection::Closed:: *)
(*Nearly canonical form*)


CanonicalizeMolecule[atoms:{$AtomInputPattern..}, bonds:{$BondInputPattern...}, opts:OptionsPattern[]] (*/; FreeQ[{opts},IncludeHydrogens]*):=
	Module[{res},
		res = iCanonicalizeMolecule[atoms, bonds, opts] ;
        res /; MoleculeQ[ res]
	]


(* ::Subsubsection::Closed:: *)
(*iCanonicalizeMolecule*)


(* This takes a molecule from the normal input form, which can contain 
	properties in the atoms list, can have atoms referred to by their string-valued
	keys, and in general could have other options in the atom list that need
	to be evaluated before coming to the canonical evaluated form
*)
iCanonicalizeMolecule[ atomsInput_, bondsInput_, opts:OptionsPattern[] ] := Module[
	{usesKeys, hasStereo, stereo, atomNames, atoms, bonds, needsStereoFix, 
		fixedStereo, hadProperties, options, originalProperties},
	
	
	options = fixAtomCoordinateOption @ Flatten @ {opts};
	
	atomNames = Check[
		Association[ getNamesFromAtoms[atomsInput] ],
		Return[ $Failed, Block]
	];
			
	atoms = canonicalizeAtoms[atomsInput];
	(* what's a better way to merge options?  For now, deleting existing
	name options and adding them back in *)
	atoms = DeleteCases[atoms, Rule["Name",_], {2}];
	
	KeyValueMap[
		AppendTo[ atoms[[#2]], "Name" -> #1]&,
		atomNames
	];
	
	bonds = ReplaceAll[ bondsInput, atomNames];
	
	bonds = canonicalizeBonds[bonds];
	
	Replace[
		bonds,
		bond:Except[$BondPattern] :> (
		badBondMessage[{atomsInput,bondsInput,bonds,bond}];
		; Throw[$Failed, $tag]),
		1
	];
		
	hasStereo = MatchQ[
		stereo = StereochemistryElements /. options,
		{__Association}
	];
	If[
		hasStereo,
		needsStereoFix = !AllTrue[
			stereo,
			MatchQ[$StereoPattern]
		],
		needsStereoFix = False
	];
	
	If[ needsStereoFix, 
		fixedStereo = fixStereo[atomNames] /@ stereo ;
		Replace[
			fixedStereo,
			str:Except[$StereoPattern] :> (Message[
				Molecule::stereo,
				str
			];Return[$Failed, Block]),
			1
		];, 
		fixedStereo = stereo 
	];
	
	If[
		hasStereo,
		options = options /. stereo -> fixedStereo
	];
	

	(
		Molecule[ atoms, bonds, Sequence @@ options]
	) /; And[
		MatchQ[ atoms, $AtomsPattern],
		MatchQ[ bonds, {$BondPattern...}],
		MatchQ[ options, OptionsPattern[]]
	] 
		
	
]



(* ::Subsubsubsection::Closed:: *)
(*getNamesFromAtoms*)


getNamesFromAtoms[ atoms_] := Block[{key, keys},
	keys = MapIndexed[
		(key = getNameFromAtom[ #1];
		If[ StringQ[key],
			key -> #2[[1]],
			Nothing
		])&,
		atoms
	];
	keys
]

(* 	:getKeyFromAtom: *)

getNameFromAtom[Rule[key_String, atom_] ] := key;
getNameFromAtom[Rule[key_, atom_] ] := (Message[Molecule::atnam, key]; $Failed)
(*getKeyFromAtom[atom:Atom[args_Association] /; KeyExistsQ[args,"Name"] ] := Lookup[args, "Name"];*)
getNameFromAtom[atom:$AtomPattern] := atom["Name"]
getNameFromAtom[___] := Nothing;


(* ::Subsubsubsection::Closed:: *)
(*canonicalizeAtoms*)


canonicalizeAtoms[atoms_List] := canonicalizeAtom /@ atoms;

canonicalizeAtom[HoldPattern[Rule[name_String,atom_]]] := canonicalizeAtom @ atom

canonicalizeAtom[ element:HoldPattern[Entity[ "Element", _]] ] := Atom @ EntityValue[ element, "Abbreviation"];

canonicalizeAtom[ isotope: HoldPattern[Entity[ "Isotope", _]] ] := With[
	{data = EntityValue[ isotope, {"AtomicNumber", "MassNumber"}]},
	Atom[
		numberElementMap[ data[[1]] ],
		"MassNumber" -> data[[2]]
	]
]


canonicalizeAtom[ n_Integer ] := Atom[ numberElementMap @ n]
canonicalizeAtom[ atom:Atom[ $ElementsPattern, ___Rule] ] := atom
canonicalizeAtom[ Atom[ atom_String] ] := atomFromString[atom]
canonicalizeAtom[ atom_Atom ] := atom
canonicalizeAtom[ atom:$ElementsPattern ] := Atom[atom]

canonicalizeAtom[alts:Verbatim[Alternatives][atoms__]] := canonicalizeAtom /@ alts
canonicalizeAtom[except:Verbatim[Except][Except[_Atom]]] := Atom @ except
canonicalizeAtom[Verbatim[Blank[]] ]:= Atom[_]

canonicalizeAtom[atom_String] := atomFromString[atom]

canonicalizeAtom[___] := $Failed

(* ::Subsubsection::Closed:: *)
(*atomFromString*)

atomFromString[atom_] := Catch[ StringReplace[
	atom,
	{
		iso:NumberString ~~ elem:$ElementsPattern ~~ chrg:NumberString ~~ sign:"+"|"-" :>
			Throw[ Atom[ elem, "FormalCharge" -> ToExpression[ sign<>chrg], "MassNumber" -> ToExpression[iso] ] ],
		iso:NumberString ~~ elem:$ElementsPattern ~~ sign:"+"|"-" :>
			Throw[ Atom[ elem, "FormalCharge" -> ToExpression[ sign<>"1"], "MassNumber" -> ToExpression[iso] ] ],
		iso:NumberString ~~ elem:$ElementsPattern ~~ chrg:NumberString :>
			Throw[ Atom[ elem, "FormalCharge" -> ToExpression[ chrg], "MassNumber" -> ToExpression[iso] ] ],
		iso:NumberString ~~ elem:$ElementsPattern :>
			Throw[ Atom[ elem, "MassNumber" -> ToExpression[iso] ] ],
		elem:$ElementsPattern ~~ chrg:NumberString ~~ sign:"+"|"-" :>
			Throw[ Atom[ elem, "FormalCharge" -> ToExpression[ sign<>chrg] ] ],
		elem:$ElementsPattern ~~ sign:"+"|"-" :>
			Throw[ Atom[ elem, "FormalCharge" -> ToExpression[ sign<>"1"] ] ],
		elem:$ElementsPattern ~~ chrg:NumberString ~~ sign:"+"|"-" :>
			Throw[ Atom[ elem, "FormalCharge" -> ToExpression[ chrg] ] ]
	}
]; $Failed]


(* ::Subsubsection::Closed:: *)
(*canonicalizeBonds*)


canonicalizeBonds[bonds_List] := canonicalizeBond /@ bonds;

canonicalizeBond[ Bond[ a_Integer, b_Integer, c: _String | PatternSequence[] ] ] :=
	canonicalizeBond @ Bond[ {a, b}, c]
	
canonicalizeBond[  Bond[ {a_Integer, b_Integer}] ] := Bond[ {a, b}, "Single"]

canonicalizeBond[ {a_Integer, b_Integer, c: _String | PatternSequence[] } ] := 
	canonicalizeBond @ Bond[ {a, b}, c]
	
canonicalizeBond[ {{a_Integer, b_Integer}, c: _String | PatternSequence[] } ] := 
	canonicalizeBond @ Bond[ {a, b}, c]
	
canonicalizeBond[ bond:$BondPattern] := bond


badBondMessage[{atomsInput_,bondsInput_,bonds_,bond_}] := Module[
	{index = First @ Flatten @ Position[ bonds, bond]},
	Replace[
		bond,
		{
			Bond[{OrderlessPatternSequence[a:Except[_Integer], _]},___] :> 
				Message[Molecule::noatom, a],
			_ :> Message[ Molecule::bond, bondsInput[[index]]]
		}
	];
	
]



(* ::Subsubsection::Closed:: *)
(*fixStereo*)


fixStereo[keys_Association][val:KeyValuePattern[{"ChiralCenter"->_, "FiducialAtom"->_, "Ligands"->_}] ] := 
	MapAt[ ReplaceAll[keys], val, {{"ChiralCenter"},{"FiducialAtom"},{"Ligands"}}];
	
fixStereo[keys_Association][val:KeyValuePattern[{"ChiralCenter"->_}] ] := 
	MapAt[ ReplaceAll[keys], val, {{"ChiralCenter"}}];

fixStereo[keys_Association][val:KeyValuePattern[{"StereoBond"->_, "Ligands"->_, "Value"->_}] ] := 
	MapAt[ ReplaceAll[keys], val, {{"StereoBond"},{"Ligands"},{"Value"}}];
	


(* ::Subsection::Closed:: *)
(*Molecule[{atoms}]*)


CanonicalizeMolecule[atoms:{$AtomInputPattern..}, opts:OptionsPattern[]]  /; opts =!= {} := 
	CanonicalizeMolecule[ atoms, {}, opts]
	
CanonicalizeMolecule[atom_Atom, opts:OptionsPattern[]]  /; opts =!= {} := 
	CanonicalizeMolecule[ {atom}, {}, opts]


(* ::Subsection::Closed:: *)
(*Molecule[_String]*)


(* ::Subsubsection::Closed:: *)
(*createMoleculeFromString*)
With[
	{map = Dispatch[{
		"AllExplicit" | All | True -> 0,
		Automatic -> 1,
		"AllImplicit" | False -> 2,
		None -> 3}]},
	ReplaceAll[map]
]

CanonicalizeMolecule[input_String, opts:OptionsPattern[] ] := Module[ 
	{validQ, valid, molData, inputType, stereo},
	molData["HasCoordinates"] = False;
	
	(*	Molecule["C(Br)(F)CCC", StereochemistryElements -> {1 -> "S"}] *)
	stereo = OptionValue[ Molecule, {opts}, StereochemistryElements];
	If[
		stereo =!= None
		,
		Return[ 
			Molecule[ 
				CanonicalizeMolecule[input, FilterRules[{opts}, Except[StereochemistryElements]]],
				StereochemistryElements -> stereo
			],
			Module
		]
	];
		
	parseOptions[ Flatten @ {opts}, molData];
	
	molData["HydrogenLevel"] = Switch[ molData["IncludeHydrogens"],
		"AllExplicit" | All | True,
			0,
		"AllImplicit" | False,
			2,
		None,
			3,
		_,(* for a SMILES string, take Automatic to mean True *)
			molData["IncludeHydrogens"] = True;
			0
	];
	
	molData["InputString"] = input;
	
	
	validQ[] := checkMolecule[molData["imol"]];
	inputType = Which[
		loadSmiles[molData]; validQ[],
		"SMILES",
		loadInchi[molData]; validQ[],
		"InChI",
		loadName[molData]; validQ[],
		"Name"
		(* just let it return Null at this point *)
	];
	
	valid = validQ[];
	
	If[
		valid
		,
		CanonicalizeMolecule[molData["imol"],
			opts
		]
		,
		(Message[Molecule::nintrp,input]; $Failed)
	]
]



possiblyValidSMILES[s_String] := StringMatchQ[s, RegularExpression["([-+\\(\\)0-9=@#.:/%A-z\\[\\]]{1,})"]];
possiblyValidSMILES[___] := False
possiblyValidInChI = StringStartsQ["InChI="];


(* ::Subsubsection::Closed:: *)
(*loadSmiles*)


getHydrogenLevel[molData_Symbol] := molData["HydrogenLevel"](*Replace[
		molData["HydrogenLevel"],
		{
			3 -> 2,
			Except[ 0 | 1 | 2] :> 1
		}
	]*);

loadSmiles[ molData_Symbol] := Module[
	{smi = molData["InputString"], res, imol},
	If[
		!possiblyValidSMILES[smi],
		Return[Null,Module]
	];
	imol = newInstance[];
	If[
		!TrueQ[molData["IncludeAromaticBonds"]]
		,
		imol["setFlags", <|"SetAromaticity" -> False|>]
	];
	res = imol[ "createMolFromSmiles", smi, getHydrogenLevel[molData]];
	Switch[
		res,
		_LibraryFunctionError,
			Return[Null,Module],
		{_Integer, _String},
			molData["ParsingMessage"] = Last @ res;
			res = sanitizeMol[ imol, molData];
			molData["imol"] = imol;
			adjustHydrogens[molData];
			Return[Null,Module]
	];
	molData["imol"] = imol;
]

loadSmiles[___]:=False


(* ::Subsubsection::Closed:: *)
(*loadInchi*)


loadInchi[ molData_Symbol] := Module[
	{inch = molData["InputString"], res, imol, smi},
	If[
		!possiblyValidInChI[inch],
		Return[Null,Module]
	];
	imol = newInstance[];
	If[
		!TrueQ[molData["IncludeAromaticBonds"]]
		,
		imol["setFlags", <|"SetAromaticity" -> False|>]
	];
	
	res = imol[ "createMolFromInChI", inch, getHydrogenLevel[molData]];
	
	Switch[
		res,
		_LibraryFunctionError,
			Return[Null,Module],
		{_Integer, _String},
			
			If[
				StringQ[ smi = imol["jsonProperty","SMILES"]]
				,
				imol =.;
				molData["InputString"] = smi;
				Return[ loadSmiles[molData], Module]
				,
				Return[Null,Module]
			]
	];
	molData["imol"] = imol;
]



(* ::Subsubsection::Closed:: *)
(*loadName*)


loadName[ molData_Symbol] := Module[
	{name = molData["InputString"], smi},
	Which[
		StringQ[smi = quickCheckName[name]],
			molData["InputString"] = smi;
			loadSmiles[molData];,
		StringQ[smi = NameToSMILES[name]],
			molData["InputString"] = smi;
			loadSmiles[molData];,
		StringQ[ smi = getInterpreterSMILES[name]],
			molData["InputString"] = smi;
			loadSmiles[molData];
	]

]

With[
	{
		specialNames = Dispatch[
			{
				"nitrogen" | "dinitrogen" -> "N#N",
				"hydrogen" | "dihydrogen" -> "[H][H]",
				"chlorine" | "dichlorine" -> "[Cl][Cl]",
				"oxygen" | "dioxygen" -> "O=O",
				"iodine" | "diiodine" -> "[I][I]",
				"bromine" | "dibromine" -> "[Br][Br]",
				"fluorine" | "difluorine" -> "[F][F]",
				"sulfur" -> "S1SSSSSSS1",
				_ -> Null
			}
		]
	},
	quickCheckName[name_] := ToLowerCase[name] /. specialNames
];

getInterpreterSMILES[name_] /; (Chemistry`Private`$UseInterpreter =!= False) := Module[
	{entity,smi},
	entity = Replace[
		Interpreter["Chemical"][name],
		{
			chem:Entity["Chemical", _] :> chem,
			_ :> Return[$Failed, Module]
		}
	];
	smi = Replace[
		EntityValue[ entity, "IsomericSMILES"],
		Except[_?StringQ] :> EntityValue[ entity, "SMILES"]
	];
	smi /; StringQ[smi]
]

getInterpreterSMILES[___] := $Failed

(* ::Subsection::Closed:: *)
(*Molecule[_Entity]*)


CanonicalizeMolecule[ent:HoldPattern[_Entity] , opts:OptionsPattern[]] := Module[
	{res},
    res =  createMoleculeFromEntity[ent, opts];
    res /; MoleculeQ[ res]
]

CanonicalizeMolecule[ents:HoldPattern[_EntityClass|{__Entity}] , opts:OptionsPattern[]] := Module[
	{res},
    res =  createMoleculeFromEntities[ents, opts] ;
    System`Private`SetValid[res] /; ListQ[res]
]

(* ::Subsubsection::Closed:: *)
(*createMoleculeFromEntity*)

createMoleculeFromEntity[input:HoldPattern[Entity["Element",_]],opts___] := With[
	{sym = input["Abbreviation"]},
	Molecule["["<>sym<>"]"] /; StringQ[sym]
]
createMoleculeFromEntity[input:HoldPattern[Entity["Chemical", name_]], opts:OptionsPattern[] ] := Module[
	{rules, options, res},
	rules = getEntityRules[input];
	
	If[
		!MatchQ[
			rules,
			KeyValuePattern[ {"VertexTypes"->_List,"EdgeTypes"->_List}]
		]
		,
		Message[ Molecule::noinfo, input]; 
		Throw[$Failed,$tag]
	];
	
	options = Flatten[ {opts}];
	createMoleculeFromRules[ rules, options]
];

$entityProps = {"VertexTypes", "EdgeRules", "EdgeTypes", "FormalCharges", "IsomericSMILES", 
		"NonStandardIsotopeNumbers", "VertexCoordinates", "AtomPositions"};

getEntityRules[ent_] /; TrueQ[$UseEntityValue] := EntityValue[ ent, $entityProps, "PropertyAssociation"];
	
getEntityRules[ent:HoldPattern[_Entity]] := AssociationThread[
	$entityProps,
	ChemicalData[ent,#]& /@ $entityProps
]

getEntityRules[ent:HoldPattern[_EntityClass]] := Map[
	AssociationThread[$entityProps -> #] &,
	Transpose[Map[ChemicalData[ent, #]&, $entityProps]]
]

getEntityRules[ent:HoldPattern[{__Entity}]]:= getEntityRules /@ ent;

stereoSmilesQ[smi_] := StringQ[smi] && StringContainsQ[smi, "@" | "/" | "\\"]

createMoleculeFromEntities[input:HoldPattern[_EntityClass|{__Entity}], opts:OptionsPattern[] ] := Module[
	{rules, options},
	rules =  getEntityRules[input];
	
	options = Flatten[ {opts}];
	Map[
		Catch[createMoleculeFromRules[ #, options],$tag] &,
		rules
	] /; ListQ[rules]
];


(* ::Subsection::Closed:: *)
(*Molecule[__Rule]*)


CanonicalizeMolecule[rules:{__Rule | __RuleDelayed}, opts:OptionsPattern[] ] := Module[
	{res},
	res =  createMoleculeFromRules[Association @ rules, opts] ;
    res /; MoleculeQ[ res]
]

CanonicalizeMolecule[rules_?AssociationQ, opts:OptionsPattern[] ] := Module[
	{res},
	res =  createMoleculeFromRules[rules, opts] ;
    res /; MoleculeQ[ res]
]

(* ::Subsubsection::Closed:: *)
(*createMoleculeFromRules*)


rord = Rule | RuleDelayed;


createMoleculeFromRules[rules:KeyValuePattern[rord["IsomericSMILES", smi_?stereoSmilesQ]],opts:OptionsPattern[]] := Module[
	{mol, mol2, map},
	
	mol = Replace[
		createMoleculeFromRules[KeyDrop[rules, "IsomericSMILES"]],
		Except[_?MoleculeQ] :> Throw[$Failed, $tag]
	];
	
	mol2 = Replace[ 
		Molecule[ smi],
		Except[_?MoleculeQ] :> Return[mol, Module]
	];
	
	map = Replace[
		FindMoleculeSubstructure[ mol2, mol],
		{
			{} :> Return[mol, Module],
			{ass_Association,___} :> Values[ass]
		}
	];
	
	mol2 = MoleculeModify[
		mol2,
		{"RenumberAtoms", map}
	];
	
	Replace[
		Molecule[ mol, StereochemistryElements -> mol2["StereochemistryElements"]],
		Except[_?MoleculeQ] :> Return[mol, Module]
	]
]

createMoleculeFromRules[rules:KeyValuePattern[ {rord["VertexTypes",_List],rord["EdgeRules",_List]}], opts:OptionsPattern[] ] := Module[
	{options, assoc = rules,molData, valid, hydrogenOption},
	If[
		MatchQ[Values @ assoc, {{_List} ..}],
		assoc = First /@ assoc
	];
	assoc = KeyMap[Replace["NonStandardIsotopeNumbers" -> "MassNumbers"], DeleteMissing @ assoc];
	
	parseRules[ assoc , molData];
	molData["AtomCoordinates"];
	valid = validateMolData[molData];
	molData["Options"] = Flatten[{opts}];
	molData["SanitizationOptions"] = <| 
		"Kekulize" -> False,
		"SetAromaticity" -> False 
	|>;
	molData["KeepKekulized"] = True;
	loadMoleculeData[molData];
	
	options = Join[
		molData["Options"],
		{
			If[ 
				ValueQ[molData["AtomDiagramCoordinates"]],
				AtomDiagramCoordinates -> molData["AtomDiagramCoordinates"],
				Nothing
			],
			If[ 
				ValueQ[molData["AtomCoordinates"]],
				AtomCoordinates -> molData["AtomCoordinates"],
				Nothing
			]
		}
	];
	
	hydrogenOption = If[MatchQ[(hydrogenOption = IncludeHydrogens /. options), True|False]
		,
		IncludeHydrogens -> hydrogenOption
		,
		Sequence @@ {}
	];
	
	
	Molecule[ Molecule[molData["imol"],Sequence @@ options], hydrogenOption]
	

]



createMoleculeFromRules[rules_,__] := (Message[ Molecule::noinfo, rules]; Throw[$Failed,$tag]);


(* ::Subsubsubsection::Closed:: *)
(*parseRules*)


$fullMolRules = KeyValuePattern[{rord["VertexTypes",_List],rord["EdgeRules",_List], rord["EdgeTypes",_List]}];
$atomsAndCoordRules = KeyValuePattern[{"VertexTypes"->_List,"AtomPositions"|"AtomCoordinates"->(_List|_?quantityArrayQ)}]

parseRules[rules:$fullMolRules, molData_Symbol] := Module[ 
	{atoms, bonds, bondTypes, charges, isotopes, vertexCoordinates, atomPositions, options = {}, res, im, graph, valid, identifiers, nAtoms, atomsArray},
	
	{atoms, bonds, bondTypes, charges, isotopes, vertexCoordinates, atomPositions} = 
		{"VertexTypes", "EdgeRules", "EdgeTypes", "FormalCharges", 
			"MassNumbers","VertexCoordinates","AtomPositions"} /. rules ;
			
	If[
		!And[
			Length[bonds] === Length[bondTypes],
			AllTrue[Union @ Flatten[List @@@ bonds], Between[{1,Length @ atoms}]]
		],
		Throw[$Failed, $tag]
	];
	
	molData["AtomsInput"] = Lookup[rules,"VertexTypes"];
	
	molData["AtomicNumbers"] = AtomicNumber[atoms];
	nAtoms = Length @ atoms;
	If[
		!VectorQ[molData["AtomicNumbers"], IntegerQ]
		,
		Message[Molecule::invsys1,atoms, " list of atoms"];
		Throw[$Failed,$tag];
	];
	
		
	
	atomsArray = ConstantArray[0, {nAtoms, 5}];
	isotopes = Replace[
		isotopes,
		None -> 0,
		1
	];
	isotopes = Replace[
		Replace[isotopes,None->0,1],
		{
			"MassNumbers" | None | {None..} -> ConstantArray[ 0, nAtoms],
			 x:{__?NonNegative} /; Length[x]===nAtoms :>  x,
			 _ :> (
				Message[Molecule::invsys1,isotopes, " list of mass numbers"];
				Throw[$Failed,$tag];
			)
		}
	];
	
	charges = Replace[
		Replace[charges,None->0,1],
		{
			"FormalCharges" | None | {None..} -> ConstantArray[ 0, nAtoms],
			 x:{__?IntegerQ} /; Length[x]===nAtoms :>  x,
			 _ :> (
				Message[Molecule::invsys1,charges, " list of formal charges"];
				Throw[$Failed,$tag];
			)
		}
	];
	
	atomsArray[[All, 1]] = molData["AtomicNumbers"];
	atomsArray[[All,2]] = charges;
	atomsArray[[All,3]] = isotopes;
	
	molData["AtomArray"] = atomsArray;
	
	bonds = Replace[
		List @@@ bonds,
		{
			"EdgeRules" -> {},
			Except[{{_Integer,_Integer}...}] :> (
				Message[Molecule::invsys1,bonds, " list of bonds"];
				Throw[$Failed,$tag];
			)
		}
	];
	
	
	bondTypes = Switch[
		bondTypes,
		"EdgeTypes",
			molData["BondTypes"] = Missing["NotProvided"];
			ConstantArray[0, Length @ bonds],
		{___String},
			Replace[
				bondOrderLookup[bondTypes],
				Except[{___Integer}] :> (
					Message[Molecule::invsys1,bondTypes, " list of bond types"];
					Throw[$Failed,$tag];
				)
			],
		_,
			Message[Molecule::invsys1,bondTypes, " list of bond types"];
			Throw[$Failed,$tag];
	];
	
	molData["Bonds"] = MapThread[
		Append,
		{bonds, bondTypes}
	];
	
	(* some of the import/export functions return both 2D and 3D coordinates as "VertexCoordinates" *)
	If[
		Dimensions[ vertexCoordinates] === {Length @ atoms, 3} && !MatrixQ[atomPositions],
		atomPositions = vertexCoordinates;
		vertexCoordinates = Null;
	];
	
	If[!MatrixQ[atomPositions, QuantityQ]
		,
		atomPositions = atomPositions / 100; 		
	];
	
	If[!MatrixQ[vertexCoordinates, QuantityQ]
		,
		vertexCoordinates = vertexCoordinates / 100;
	];

	
	If[ 
		MatchQ[vertexCoordinates, {{_Real, _Real}..}],
		molData["AtomDiagramCoordinates"] = vertexCoordinates;
	];
	
	If[ 
		MatrixQ[atomPositions],
		molData["AtomCoordinates"] = atomPositions
	];
	
	molData["HasAtomsAndBonds"] = True;
	
	identifiers = Normal[
		rules[[{"IUPACName", "Name", "InChI", "StdInChI", "SMILES", "IsomericSMILES"}]]
	];
	Replace[
		identifiers,
		Rule[key_String,val_String] :> (molData[key] = val),
		1
	];
	
]

parseRules[___] := Throw[$Failed,$tag];


validateMolData[molData_Symbol] := And[
	VectorQ[molData["AtomicNumbers"]],
	VectorQ[molData["Bonds"]]
]


hasCoordsButNoBonds[molData_] := And[
	MatchQ[ molData[ "Bonds"], _Missing],
	VectorQ[ molData[ "AtomicNumbers"], IntegerQ],
	MatrixQ[ molData["AtomCoordinates"] ]
];



(* ::Subsection::Closed:: *)
(*Molecule[_Graph]*)


CanonicalizeMolecule[g_?GraphQ , opts:OptionsPattern[]] := Module[
	{res},
    res =  GraphMolecule[g, opts] ;
    res /; MoleculeQ[ res]
]



(* ::Subsection::Closed:: *)
(*Molecule[_?ByteArrayQ]*)


CanonicalizeMolecule[ba_?ByteArrayQ , opts:OptionsPattern[]] := Module[
	{nm = newInstance[]},
    nm["loadPickle", ba];
    CanonicalizeMolecule[nm, opts]
]



(* ::Subsection::Closed:: *)
(*Molecule[_?ManagedLibraryExpressionQ]*)


CanonicalizeMolecule[imol_iMolecule /; checkMolecule[imol], opts:OptionsPattern[]] := Module[
	{data, expr, atoms, bonds, options, molData},
	data = molDataFromLibrary[ imol];
    (
    	{atoms, bonds} = data[[{1,2}]];
		options = fixMetaInformation @ mergeOptions[ data, imol, Flatten @ {opts}];
		valid = FreeQ[ bonds, "Unspecified"];
		If[
			FreeQ[ bonds, "Unspecified"]
			,
			expr = validMolecule[atoms, bonds, Sequence @@ options];
			If[
    			!has2DCoordinates[imol] || !has3DCoordinates[imol],
    			parseCoordinates[options, molData];
    			loadCoordinates[imol, molData];
    		];
			cacheMolecule[ expr, imol];
			,
			expr = Molecule[atoms, bonds, Sequence @@ options];
		];
    	expr    
    ) /; MatchQ[ data, {$AtomsPattern, {$BondPattern...}, OptionsPattern[]}]
	
]



(* ::Subsubsection::Closed:: *)
(*molDataFromLibrary*)


molDataFromLibrary[ mol_?ManagedLibraryExpressionQ ] := Block[
	{atoms, bonds, stereoElements},
	(
		atoms = getiMolProperty[ mol, "AtomList"];
		(
			bonds =getiMolProperty[ mol, "BondList"];
			stereoElements = getiMolProperty[ mol, "StereochemistryElements"];
			
			If[ 
				stereoElements === {},
				stereoElements = Nothing,
				stereoElements = StereochemistryElements -> stereoElements
			];
			
			{atoms, bonds, stereoElements}
			
		) /; MatchQ[atoms, $AtomsPattern]
	) /; checkMolecule[ mol]
]

molDataFromLibrary[___] := $Failed



(* ::Subsubsection::Closed:: *)
(*mergeOptions*)


mergeOptions[ molData:{$AtomsPattern, {$BondPattern...}, opts:OptionsPattern[]}, imol_, {} ] := Flatten @ {opts};

mergeOptions[molData:{$AtomsPattern, {$BondPattern...}, opts:OptionsPattern[]}, imol_, newOpts:{__Rule}] := Block[
	{coords3D, coords2D, nAtoms, returnOptions, stereo, fixedStereo},
	
	returnOptions = Normal @ Merge[
		{
			Flatten[{opts}],
			newOpts
		},
		Last
	];

	returnOptions = fixAtomCoordinateOption[returnOptions];
	
	
	coords2D = N @ Lookup[ returnOptions, AtomDiagramCoordinates ];
	
	If[
		MatrixQ[ coords2D, Internal`RealValuedNumericQ] && Length[coords2D] === imol["atomCount", True],
		imol[ "addCoordinates", coords2D]
	];
	
	
	
	coords3D = Lookup[
		newOpts,
		AtomCoordinates
	];
	If[
		quantityArrayQ[coords3D],
		imol[ "addCoordinates", QuantityMagnitude[coords3D, "Angstroms"]]
	];
	
	If[
		!MatchQ[ stereo = StereochemistryElements /. returnOptions, StereochemistryElements | None | {__?AssociationQ} | {{__Rule}..}]
		,
		Message[Molecule::stereo, stereo];
		Throw[$Failed, $tag]
	];
	If[
		stereo =!= (fixedStereo = getiMolProperty[imol,"StereochemistryElements"])
		,
		returnOptions = ReplaceAll[
			returnOptions,
			stereo -> fixedStereo
		]
	
	];
	
	returnOptions
]



(* ::Subsubsection::Closed:: *)
(*fixAtomCoordinateOption*)


fixAtomCoordinateOption[options:{before___,HoldPattern[AtomCoordinates | AtomCoordinates -> ac_], after___}] := {before,
	AtomCoordinates -> Replace[
		ac,
		{
			x_?quantityArrayQ :> UnitConvert[ x, "Angstroms"],
			x_ /; MatrixQ[x, QuantityQ] :> quantityArray[ UnitConvert[ x, "Angstroms"]],
			x_?MatrixQ :> quantityArray[ x, "Angstroms"]
		}
	], after}
fixAtomCoordinateOption[x___] := x;



(* ::Subsection::Closed:: *)
(*Molecule[_?MoleculeQ, opts]*)

CanonicalizeMolecule[ mol_?MoleculeQ, opts:OptionsPattern[] /; !FreeQ[{opts}, IncludeHydrogens -> True|False]] := Module[
	{h,other},
	h = OptionValue[Molecule, {opts}, IncludeHydrogens];
	other = FilterRules[{opts},Except[IncludeHydrogens]];
	MoleculeModify[
		Switch[other,
			{},
				mol,
			_,
				Molecule[mol,other]
		],
		If[
			TrueQ @ h
			,
			"AddHydrogens"
			,
			"RemoveHydrogens"
		]
	]
]

CanonicalizeMolecule[ mol_?MoleculeQ, opts:OptionsPattern[] ] := Block[ 
	{atoms, bonds, options},
	{atoms, bonds} = (List @@ mol)[[;;2]];
	options = Normal @ Merge[
		{
			Flatten[{opts}],
			Options[mol]
		},
		First
	];
	Molecule[ atoms, bonds, Sequence @@ options]
]


(* ::Subsubsection::Closed:: *)
(*adjustHydrogens*)


adjustHydrogens[atoms:$AtomsPattern, bonds:{$BondPattern...}, {before___, IncludeHydrogens -> ih_, after___}] := 
	Block[ {mol = Molecule[ atoms, bonds, before, after]},
		Switch[ih,
			True | "AllExplicit",
				MoleculeModify["AddHydrogens"][ mol],
			False | "AllImplicit",
				MoleculeModify["RemoveHydrogens"][mol],
			None,
				MoleculeModify["RemoveHydrogens","MakeImplicit" -> False][mol]
		]
	]


(* ::Subsection::Closed:: *)
(*Molecule[_?Molecule, {id1,id2,..}]*)


CanonicalizeMolecule[mol_?MoleculeQ, ids:{__Integer}, opts:OptionsPattern[] ] := Block[
	{res},
	res = MoleculeModify[ mol, {"ExtractParts", ids}];
	If[ !FreeQ[{opts}, IncludeHydrogens],
		res = Molecule[ res, FilterRules[{opts}, IncludeHydrogens] ]
	];
	res /; MoleculeQ[res]
]



(* ::Subsection::Closed:: *)
(*fallthrough*)


CanonicalizeMolecule[___] := $Failed



(* ::Section::Closed:: *)
(*SequenceMolecule*)


SequenceMolecule[args___] := Module[
	{res = Catch[ iSequenceMolecule[args], $tag]},
	res /; res =!= $Failed
]

addCompletion[ "Chemistry`SequenceMolecule" -> {0, Keys @ $SequenceTypes}]

Options[SequenceMolecule] = Options[iSequenceMolecule] = {IncludeHydrogens -> True}

iSequenceMolecule[seq_String, type_String:"LAminoAcid", OptionsPattern[]] := Module[
	{imol = newInstance[], flavor, res},
	flavor = Replace[
		$SequenceTypes[type],
		Except[_Integer] :> (
			Message[SequenceMolecule::badtyp, type];
			Throw[$Failed, $tag]
		)
	];
	res = imol["createMolFromSequence", seq, flavor];
	(
		res = Molecule[ imol];
		If[
			OptionValue[IncludeHydrogens],
			Molecule[ res, IncludeHydrogens -> True],
			res
		]
	) /; !MatchQ[ res, _LibraryFunctionError]
	
]

iSequenceMolecule[___] := $Failed



(atom:$AtomPattern)[val_String] := iAtomValue[atom, val]


(* ::Section::Closed:: *)
(*End package*)


End[] (* End Private Context *)

