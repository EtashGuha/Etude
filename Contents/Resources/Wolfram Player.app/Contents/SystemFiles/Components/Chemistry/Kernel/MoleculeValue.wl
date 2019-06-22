


Begin["Chemistry`Private`MoleculeValueDump`"]


(* ::Section::Closed:: *)
(*Property lists*)



exposedProperties[] := Flatten[Values @ $ExposedProperties];

updateMolValueAutoComplete[] := (
	addCompletion[ "MoleculeValue" -> {0,exposedProperties[]}];
	addCompletion[ "MoleculeProperty" -> {exposedProperties[]}];
	addCompletion[ "AtomList" -> {0, 0, $ExposedProperties["AtomProperties"]}];
	addCompletion[ "BondList" -> {0, 0, $ExposedProperties["BondProperties"]}];
)


$PropertyAliases = <|
		"VertexCoordinates" -> "AtomDiagramCoordinates",
		"AtomPositions" -> "AtomCoordinates"
	|>
	
findPropertyAlias = Lookup[ $PropertyAliases, #, #] &;
addPropertyAlias[x_, y_] := AssociateTo[$PropertyAliases, y -> x];


$ExposedProperties = <||>
addExposedProperty[prop_, category_] := If[
	KeyExistsQ[$ExposedProperties, category],
	AppendTo[ $ExposedProperties[category], prop],
	AssociateTo[$ExposedProperties, category -> {prop}]
]



(* ::Section::Closed:: *)
(*iAtomValue*)


AtomValue[ args___] := Block[ {res},
	res = Catch[ iAtomValue[ args], $tag];
	res /; res =!= $Failed
]


iAtomValue[atom:($AtomPattern | $AtomInMoleculePattern), "Properties"] := $AtomProperties;


iAtomValue[ atom:($AtomPattern | $AtomInMoleculePattern), "FormalCharge"] := With[
	{res = Lookup[ Options @ atom, "FormalCharge", 0]},
	res /; IntegerQ[res]
]
iAtomValue[ atom_, "FormalCharge"] := (Message[Atom::chrg, Lookup[ Options @ atom, "FormalCharge", 0] ]; $Failed)


iAtomValue[ atom:($AtomPattern | $AtomInMoleculePattern), "MassNumber"] := With[
	{res = Lookup[ Options @ atom, "MassNumber", Automatic]},
	res /; MatchQ[ res, Automatic | _Integer] 
](* Is Automatic the right word here?  I think so, but how
		do we handle "Commonest"? Do we evaluate it? *)
iAtomValue[ atom_, "MassNumber"] := (Message[Atom::mass,  Lookup[ Options @ atom, "MassNumber", Automatic] ]; $Failed)


iAtomValue[ atom:($AtomPattern | $AtomInMoleculePattern), "UnpairedElectronCount"] := With[
	{res = Lookup[ Options @ atom, "UnpairedElectronCount", 0]},
	res /; IntegerQ[res] && Positive[res]
]
iAtomValue[ atom_, "UnpairedElectronCount"] := (Message[Atom::elec, Lookup[ Options @ atom, "UnpairedElectronCount", 0] ]; $Failed)


iAtomValue[ atom:($AtomPattern | $AtomInMoleculePattern), "HydrogenCount"] := With[
	{res = Lookup[ Options @ atom, "HydrogenCount", Automatic]},
	res /; IntegerQ[res] && Positive[res]
]
iAtomValue[ atom_, "HydrogenCount"] := 
	(Message[Atom::mass, "HydrogenCount", Lookup[ Options @ atom, "HydrogenCount", Automatic] ]; $Failed)


iAtomValue[ atom:($AtomPattern | $AtomInMoleculePattern), "Name"] := With[
	{res = Lookup[ Options @ atom, "Name", None]},
	res /; MatchQ[ res, None | _String ]
]
iAtomValue[ atom_, "Name"] := (Message[Atom::atnam, Lookup[ Options @ atom, "Name", None] ]; $Failed)


iAtomValue[ atom:($AtomPattern | $AtomInMoleculePattern), "AtomicNumber"] := AtomicNumber @ First @ atom;




(* ::Section::Closed:: *)
(*MoleculeValue*)


$elementProperty = HoldPattern[EntityProperty[ "Element", _]];

$BasicPropertyPattern = _String | $elementProperty
$AtomSpecPattern = _Bond | {__Bond} | _Integer | {__Integer} | {{__Integer}..} | {{__Bond}..} | All
$PropertyPattern = $BasicPropertyPattern | {$BasicPropertyPattern, $AtomSpecPattern} | _MoleculeProperty



MoleculeValue[ args___] := Block[ {res},
	res = Catch[ iMoleculeValue[ args], $tag];
	res /; res =!= $Failed
]



(* ::Subsection:: *)
(*iMoleculeValue*)


iMoleculeValue[molecule_?MoleculeQ, prop:$PropertyPattern] := With[
	{property = ReplaceAll[
		normalizeMolProperty @ prop,
		atoms:HoldPattern["Atoms" -> _String|{ ___,_String,___}] :> atomNameMap[molecule][atoms]
	]
	},
	cachedEvaluation[molValue, molecule, property ]
];

iMoleculeValue[ molecules:{__?MoleculeQ},  
	prop:$PropertyPattern, resultStyle_String:"List"] := With[
	{property = normalizeMolProperty @ prop},
	{res = cachedEvaluation[molValue, #, property] & /@ molecules},
	 Switch[resultStyle,
		"MoleculeAssociation",
			AssociationThread[  molecules, res ],
		"Dataset",
			Dataset @ AssociationThread[  molecules, res ],
		_,
			res
    ]
]
	
iMoleculeValue[ molecule_?MoleculeQ, 
		props:{$PropertyPattern..},resultStyle_String:"List", opts___?OptionQ] := With[
	{properties = normalizeMolProperty[#, opts] & /@ props},
	{res = cachedEvaluation[molValue, molecule, # ] & /@ properties},
	Switch[resultStyle,
		"PropertyAssociation",
			AssociationThread[  props, res ],
		"Dataset",
			Dataset @ AssociationThread[ props, res ],
		_,
			res
    ]
]

iMoleculeValue[ molecules:{__?MoleculeQ},  props:{$PropertyPattern..},resultStyle_String:"List"] := With[
	{properties = normalizeMolProperty @ props},
	{res = Outer[ cachedEvaluation[molValue, #1, #2] & , molecules, properties]},
	Switch[resultStyle,
		"PropertyAssociation",
			AssociationThread[props -> #] & /@ res,
		"Dataset",
			Dataset[AssociationThread[props -> #] & /@ res],
		"MoleculePropertyAssociation",
			AssociationThread[ molecules, AssociationThread[props -> #] & /@ res],
		_,
			res
	]
]

	
iMoleculeValue[ mp_MoleculeProperty, "CanonicalName"] :=  Replace[
	normalizeMolProperty @ mp,
	{
		MoleculeProperty[x_String, opts:{___Rule}] :> x,
		MoleculeProperty[ {x_String, All|_Integer|_Bond|{__Integer}|{__Bond}}] :> x,
		_ :> Missing["NotAvailable"]
	}
]

iMoleculeValue[ mp_MoleculeProperty, "Qualifiers"] := molQualifiers[ mp[ "CanonicalName"] ]

iMoleculeValue[ mp_MoleculeProperty, "Description"] := molDescription[ mp[ "CanonicalName"] ]

iMoleculeValue[ mp_MoleculeProperty, "LongDescription"] := Replace[
	molLongDescription[ mp[ "CanonicalName"] ],
	_molLongDescription :> molDescription[ mp[ "CanonicalName"] ]
]

iMoleculeValue[ mp_MoleculeProperty, "Source"] := Replace[
	molPropertyCitation[ mp[ "CanonicalName"] ],
	{
		_molPropertyCitation :> Missing["NotAvailable"],
		x_String /; StringStartsQ[x, "http"] :> Hyperlink[x]
	}
]

iMoleculeValue[ mp_MoleculeProperty, "MetaInformation"] := Module[
	{propertyProperties = {"CanonicalName","Description","LongDescription","Source"}, values},
	values = AssociationThread[ propertyProperties -> (mp /@ propertyProperties)];
	Dataset @ If[#Description === #LongDescription, KeyDrop["LongDescription"] @ #, #]& @ DeleteMissing @ values
]

iMoleculeValue["Properties"] := $ExposedProperties
iMoleculeValue["AllProperties"] := Flatten[ Values @ $ExposedProperties]
iMoleculeValue["PropertyClasses"] := Keys @ $ExposedProperties

makeEntityDownValue[iMoleculeValue]
iMoleculeValue[m_ ,___] /; !MatchQ[m, _?MoleculeQ | {__?MoleculeQ}] := (messageNoMol[MoleculeValue,m]; Null /; False)

iMoleculeValue[___] := $Failed


(* ::Section::Closed:: *)
(*MoleculeProperty*)


(*
	SubValues and UpValues for MoleculeProperty
*)

(prop:MoleculeProperty[__])[subprop_String] := MoleculeValue[ prop, subprop]

(prop:MoleculeProperty[__])[m_] := Module[{res = MoleculeValue[ m, prop]}, res /; !MatchQ[res, _MoleculeValue]]

MoleculeProperty /: Set[ MoleculeProperty[prop_String], func:(_Function | _Symbol)] := SetMoleculeValueFunction[ prop, func]

MoleculeProperty /: Set[ MoleculeProperty[prop_String]["Qualifiers"], rules:({__Rule}|_Rule)] := Module[
	{opts = Replace[ rules, Except[{__Rule}] :> {rules}] },
	opts = Join[ FilterRules[opts, molQualifiers[ prop]], molQualifiers[ prop]];
	opts = Reverse @ Normal @ Association @ Reverse @ opts;
	molQualifiers[ prop] = opts;
]



(* ::Subsection::Closed:: *)
(*normalizeMoleculeProperty*)


normalizeMolProperty[ MoleculeProperty[x:{"MetaInformation",___}] ] := MoleculeProperty[x,{}]
normalizeMolProperty[ x_String, y___Rule ] := normalizeMolProperty[ MoleculeProperty[ findPropertyAlias @ x, {y}] ];
normalizeMolProperty[ MoleculeProperty[x:(_String | $elementProperty),y__Rule] ] :=  normalizeMolProperty @ MoleculeProperty[ x, {y}];

normalizeMolProperty[ MoleculeProperty[x_String,y___], opts__Rule] :=  
	normalizeMolProperty[ 
		MoleculeProperty[x, Sequence @@ Flatten[{y,opts}]] 
	]

normalizeMolProperty[ MoleculeProperty[ x_String] ] := MoleculeProperty[findPropertyAlias @ x, {}]
normalizeMolProperty[ MoleculeProperty[x_String, opts:{___Rule}] ] := With[
	{prop = findPropertyAlias @ x, options = fixIHRule @ opts},
	MoleculeProperty[ prop, options] /; validatePropertyQualifiers[ prop, options]
]
(* remove when the ImportExport code is updated *)
fixIHRule = DeleteCases[HoldPattern[IncludeHydrogens -> _]]
iMoleculeValue[bef___, IncludeHydrogens -> Automatic] := iMoleculeValue[bef]
(***********************)



normalizeMolProperty[ MoleculeProperty[prop:$elementProperty, opts:{___Rule}] ] := MoleculeProperty[ prop, opts]
normalizeMolProperty[ prop:$elementProperty, opts_] := normalizeMolProperty[ MoleculeProperty[ prop, opts] ]

normalizeMolProperty[ prop:MoleculeProperty[{_String, $AtomSpecPattern}, {}] ] := prop;
normalizeMolProperty[ {x_String,y:$AtomSpecPattern} | MoleculeProperty[{x_String,y:$AtomSpecPattern}] ] := MoleculeProperty[ {findPropertyAlias @ x,y}, {}];
normalizeMolProperty[ {x:$elementProperty,y:$AtomSpecPattern} ] := MoleculeProperty[ {x,y}, {}];
normalizeMolProperty[ x:$elementProperty ] := MoleculeProperty[ x, {}];

normalizeMolProperty[ x_List] := normalizeMolProperty /@ x;
normalizeMolProperty[___] := $Failed

validatePropertyQualifiers[prop_String, quals:{___Rule}] := Which[
	ContainsAll[Keys @ molQualifiers[prop], Keys @ quals],
		True,
	MemberQ[ElementData["Properties"],prop],
		True,
	!MemberQ[exposedProperties[], prop],
		Message[ MoleculeValue::notprop, prop, MoleculeValue];
		Throw[$Failed, $tag];,
	True,
		Message[MoleculeValue::qname, First @ Complement[Keys @ quals, Keys @ molQualifiers[prop]], prop];
		Throw[$Failed, $tag]
];


(* ::Section::Closed:: *)
(*molValue*)


(* ::Subsection::Closed:: *)
(*SetMoleculeValueFunction*)


SetMoleculeValueFunction[ prop_String, func_] := (
	Unprotect[ molValue];
	molValue[ mol_, mp[prop] ] := func[mol];
	AppendTo[$UserProperties,prop];
	Protect[molValue];
);

SetMoleculeValueFunction[ prop_String, func_Symbol, optionsSymbol_Symbol:Automatic] := Module[
	{symbol = Replace[ optionsSymbol, Automatic -> func]},
	Unprotect[ molValue];
	molValue[ mol_, mp[prop, opts_] ] := func[mol, Sequence @@ opts];

	Replace[
		Options[symbol],
		x:{__Rule} :> Set[molQualifiers[prop], x]
	];
	Replace[
		symbol["Description"],
		x_String :> Set[molDescription[prop], x]
	];
	Replace[
		symbol["PropertyAlias"],
		x:(_String|{__String}) :> AssociateTo[
			$PropertyAliases,
			AssociationThread[ Flatten[ {x} ] -> prop ]
		]
	];

	AppendTo[$UserProperties,prop];
	Protect[molValue];
];



(* ::Subsection::Closed:: *)
(*molValue*)


mp[x_] := MoleculeProperty[x,{}]
mp[x__] := MoleculeProperty[x]



molValue[_, mp["Properties"]] := iMoleculeValue["Properties"];
molValue[mol_, mp["PropertyValueAssociation"]] := Map[
	iMoleculeValue[mol, #, "PropertyAssociation"]&,
	iMoleculeValue["Properties"]
]


(*molValue[ $EmptyMolecule, ___] := Missing[ "NotAvailable"]*)


(* ::Subsection::Closed:: *)
(*utilities*)

(* ::Subsubsection::Closed:: *)
(*makeQA*)

makeQA[] := Replace[
	{
		x:({__quantity} | {{__quantity}..}) /; Length[x] > 1 :> quantityArray[x],
		x_?quantityArrayQ /; Length[x] === 1 :> Normal[x, quantityArray]
	}
] 

(* ::Subsubsection::Closed:: *)
(*getLibraryMolForProperty*)


getLibraryMolForProperty[prop:(
	"SMILES" | "CrippenClogP" | "CrippenMR" | "HeavyAtomCount" | "HBondAcceptorCount" |
	"SlogP_VSA" | "SMILES" | "SMR_VSA" | "PEOE_VSA" | "LabuteApproximateSurfaceArea" | "RotatableBondCount" | "MolecularQuantumNumbers"
	) ] = "NoHydrogens"
	
getLibraryMolForProperty[ 
	prop:((*"InChI" | "InChIKey" | *)"PossibleStereocenters")
	] := "AllAtoms"

getLibraryMolForProperty[___] := Sequence @@ {};


(* ::Subsubsection::Closed:: *)
(*setMolPropertyValue*)

With[
	{
		props = {
			"Asphericity", "Autocorrelation3D", "BondLength", "Eccentricity", "GETAWAY", "InertialShapeFactor",
			"MORSE", "PlaneOfBestFitDistance", "RadiusOfGyration", "NormalizedPrincipalMomentRatios",
			"RDF", "SpherosityIndex", "WHIM", "GeometricStericEffectIndex", "SymmetryElements", 
			"SymmetryEquivalentAtoms", "PointGroup", "CoulombMatrix", "MMFFEnergy", "MMFFsEnergy",
			"CenterOfMass", "PrincipalAxes", "PrincipalMoments"
		}
	},
	check3DProperty[prop:(Alternatives @@ props)][args__] := messageOnNotEmbedded @ args(*Check[ Catch[messageOnNotEmbedded @ args,$tag], Return[Missing["NotAvailable"],Module]]*);
] 

setPerAtomProperty[prop_String] := (
	molValue[mol_, mp[prop,opts___] ] := molValue[mol, mp[{prop, All}, opts] ]
)
setMolPropertyValue[prop_, type_, rejectionPattern_:Blank[LibraryFunctionError]] := With[
	{im = getLibraryMolForProperty[prop]},
	molValue[mol_, mp[prop]] := Module[
		{imol = getCachedMol[mol, im]},
		check3DProperty[prop][mol, imol, $tag];
		Replace[
			imol[type,prop],
			{
				rejectionPattern :> Missing["NotAvailable"],
				x_ :> postProcessMolProperty[prop][x]
			}
		]
	]
]


(* ::Subsubsection::Closed:: *)
(*setPerAtomPropertyValue*)


setPerAtomPropertyValue[prop_String, type_String] := (
	molValue[mol_, mp[prop,opts___] ] := molValue[mol, mp[{prop, All}, opts] ];
	molValue[mol_, mp[{prop, x_Integer}, o_]] := With[
		{res = molValue[ mol, mp[{prop, {x}}, o]]},
		First @ res /; MatchQ[res, {_}]
	];
	molValue[mol_, mp[{prop, Bond[{a_,b_},___]}, o_]] := molValue[ mol, mp[ {prop, {a,b}},o]];
	molValue[mol_, mp[{prop, b:{__Bond}}, o_]] := molValue[ mol, mp[ {prop, #},o]] & /@ b;
	molValue[mol_, mp[{prop, atoms_:All}, opts_]] := Module[
		{imol = getCachedMol[ mol, opts], processedAtoms},
		processedAtoms = getAtomsList[ imol, atoms];
		If[
			!VectorQ[ processedAtoms, IntegerQ ]
			,
			Return[ $Failed, Module]
		];
		check3DProperty[prop][ mol, imol, $tag];		
		(* make sure that the relevant postProcessMolProperty definition is listable *)
		makeQA[] @ postProcessMolProperty[prop] @ imol[type, prop, processedAtoms]
	]
)

setPerAtomPropertyValue[prop_String, func_Function] := (
	molValue[mol_, mp[prop,opts___] ] := molValue[mol, mp[{prop, All}, opts] ];
	molValue[mol_, mp[{prop, x_Integer}, o_]] := With[
		{res = molValue[ mol, mp[{prop, {x}}, o]]},
		First @ res /; MatchQ[res, {_}]
	];
	molValue[mol_, mp[{prop, Bond[{a_,b_},___]}, o_]] := molValue[ mol, mp[ {prop, {a,b}},o]];
	molValue[mol_, mp[{prop, b:{__Bond}}, o_]] := molValue[ mol, mp[ {prop, #},o]] & /@ b;
	molValue[mol_, mp[{prop, atoms_:All}, opts_]] := Module[
		{imol = getCachedMol[ mol, opts], processedAtoms},
		processedAtoms = getAtomsList[ imol, atoms];
		
		makeQA[] @ func[mol, imol, processedAtoms] /; VectorQ[ processedAtoms, IntegerQ ]
	]
)



(* ::Subsubsection::Closed:: *)
(*setPerBondPropertyValue*)



setPerBondPropertyValue[prop_String, func_Function:None] := (

	(*
		MoleculeValue[ mol, "BondLength"] should evaluate to 
		MoleculeValue[ mol, {"BondLength",All}]
		
		MoleculeValue[ mol, {"BondLength", {__Bond} }] will verify that
		all of the given bonds are valid before converting to the canonical
		form of MoleculeValue[ mol, {"BondLength", {{i1,i2},{j1,j2}..} }].
		
		BondList[ mol, patt, prop] will call the canonical form so as to bypass
		checking.
	
	*)
	molValue[mol_, mp[prop] ] := molValue[mol, mp[{prop, All}] ];
	molValue[mol_, mp[{prop, {}}]] := {};
	molValue[mol_, mp[{prop, All}] ] := With[
		{bonds = BondList[mol]},
		molValue[mol, mp[{prop, First /@ bonds}] ]
	];
	molValue[mol_, mp[{prop, x:(_Bond | {_Integer,_Integer})}]] := With[
		{res = molValue[ mol, mp[{prop, {x}}]]},
		First @ res /; MatchQ[res, {_}]
	];
	molValue[mol_, mp[{prop, bonds:{__Bond}}]] := Module[
		{processedBonds},
		processedBonds = Replace[
			bonds,
			{
				b:Bond[atoms_,_] /; BondQ[mol,b] :> atoms, 
				x_ :> (
					Message[Molecule::bond, x];
					Return[$Failed, Module]
				)
			},
			{1}
		];
		molValue[ mol, mp[{prop, processedBonds}]]
	];
	If[
		func === None
		,
		molValue[mol_, mp[{prop, bonds:{{_Integer,_Integer}..}}]] := Module[
			{imol = getCachedMol[mol]},
			check3DProperty[prop][ mol, imol, $tag];
			makeQA[] @ postProcessMolProperty[prop] @ Replace[
				imol["getBondJSONProperty", prop, bonds],
				{
					{"noBond",x_} :> (
						Message[Molecule::bond, x];
						Return[$Failed, Module]
					),
					_LibraryFunctionError :> Return[$Failed, Module]
				}
			]
		]
		,
		molValue[mol_, mp[{prop, bonds:{{_Integer,_Integer}..}}]] := With[
			{imol = getCachedMol[mol]},
			makeQA[] @ func[ mol, imol, bonds]
		]
	]
	
)



(* ::Subsubsection::Closed:: *)
(*getAtomsList*)


getAtomsList[ _, Bond[{a_,b_},___]] := {a,b}
getAtomsList[ _, b:{Bond[{_,_},___]..}] := b[[All,1]]

getAtomsList[ imol_, atoms_] := getAtomsList[ imol, imol["atomCount", True], atoms];
getAtomsList[ imol_, natoms_, atoms_] := Module[ 
	{ res},
	Switch[atoms,
		
		All | "All",
			Range @  natoms,
		
		_Integer | _String,
			getAtomsList[ imol, natoms, {atoms}],
		
		(* a list of integers, all greater than zero *)
		list:{__?Internal`PositiveIntegerQ} /; Max[list] <= natoms,
			atoms,
		
		(* a list of integers, containing some negative values *)
		_List,
			Replace[
				res = atomNameMap[imol][atoms] /. x_?Negative :> (natoms + x + 1),
				Except[ x_?Internal`PositiveIntegerQ /; x <= natoms] :>  (
					Message[ Molecule::noatom, Pick[atoms,res,x]]; 
					Return[$Failed,Module]
				),
				{1}
			],
		_,
		Message[ Molecule::atom, atoms]; Return[$Failed,Module]
	]
]



(* ::Subsubsection::Closed:: *)
(*postProcessMolProperty*)


postProcessMolProperty["MolecularMass" | "MonoIsotopicMolecularMass" | "AtomicMass"] = quantity[#, "AtomicMassUnit"]&;
postProcessMolProperty["PossibleStereocenters"] = Replace[ Null -> {}];
postProcessMolProperty["GraphDistanceMatrix"] = 
	ReplaceAll[ Rationalize @ #, 10^8 -> Infinity]& ;

postProcessMolProperty["OrbitalHybridization"] = Lookup[hybridizationMap,#]&;
postProcessMolProperty["BondType"] = Lookup[bondOrderMap,#]&;
postProcessMolProperty["MassNumber"] = Replace[ #, 0 -> None, {1}]&

postProcessMolProperty["BondLength"] = quantity[#, "Angstroms"]&
postProcessMolProperty["BondStereo"] = Replace[#, "None" -> None, {1}]&;

postProcessMolProperty[
	Alternatives["SlogPVSA",
		"SMRVSA", "PEOEVSA", "Autocorrelation2D", "GETAWAY", "WHIM", "RDF", "MORSE",
		"Autocorrelation3D"
	]
] = Replace[x_ /; VectorQ[x, NumericQ] :> NumericArray[x]]

postProcessMolProperty["AtomChirality"] = Lookup[ chiralTypeMap, #, None]&


postProcessMolProperty[x__] := Identity;



(* ::Subsubsection::Closed:: *)
(*findFormulaDisplay*)


findFormulaDisplay = Row[
	FixedPoint[
		Replace[#1,
			{
				{nose___,element:PatternSequence[_String?UpperCaseQ,_String?LowerCaseQ],tail:_String...}:>{nose,StringJoin[element],tail},
				{nose___,coeff:PatternSequence[_String?(StringMatchQ[#1,DigitCharacter]&),_String?(StringMatchQ[#1,DigitCharacter]&),_String?(StringMatchQ[#1,DigitCharacter]&)],tail:_String...}:>{nose,StringJoin[coeff],tail},
				{nose___,coeff:PatternSequence[_String?(StringMatchQ[#1,DigitCharacter]&),_String?(StringMatchQ[#1,DigitCharacter]&)],tail:_String...}:>{nose,StringJoin[coeff],tail},
				{nose___,PatternSequence[s_String/;StringMatchQ[s,LetterCharacter..],d_String/;StringMatchQ[d,NumberString]],tail:_String...}:>{nose,Subscript[s,ToExpression[d]],tail},
				{nose___,Shortest[PatternSequence[leftcap:"("|"[",comp:Except["("|")"|"["|"]"]..,rightcap:")"|"]"]],d_String/;StringMatchQ[d,NumberString],tail___}/;(leftcap/.{"("->")","["->"]"})===rightcap:>{nose,Subscript[Row[{leftcap,comp,rightcap}],ToExpression[d]],tail},
				{nose___,PatternSequence[leftcap:"("|"[",comp:Except["("|")"|"["|"]"]..,rightcap:")"|"]"],var:"m"|"n"|"x"|"y",tail___}/;(leftcap/.{"("->")","["->"]"})===rightcap:>{nose,Subscript[Row[{leftcap,comp,rightcap}],var],tail},
				{nose___,PatternSequence[leftcap:"[",comp:Except["["|"]"]..,rightcap:"]"],var:"m"|"n"|"x"|"y"|"z",tail___}/;(leftcap/.{"("->")","["->"]"})===rightcap:>{nose,Subscript[Row[{leftcap,comp,rightcap}],var],tail},
				{nose___,Shortest[PatternSequence[leftcap:"(",comp:Except["["|"]",_String|_Subscript]..,rightcap:")"]],d:Except["m"|"n"|"x"|"y"|"z"|NumberString,_String|_Subscript],tail___}/;(leftcap/.{"("->")","["->"]"})===rightcap:>{nose,Row[{leftcap,comp,rightcap}],d,tail}
			}
		]&,
		DeleteCases[
			StringSplit[#1,{(cap_)?UpperCaseQ:>cap,d:DigitCharacter:>d,"\[CenterDot]"->"\[CenterDot]","["->"[","]"->"]","("->"(",")"->")","+"->"+","-"->"-",","->",","^"->"^"}],
			""
		]
	]
]&;


(* ::Subsubsection::Closed:: *)
(*iSMILESString*)


Options[iSMILESString] = {"Atoms" -> All, "IncludedAtoms" -> All, "Isomeric" -> True, "Kekulized" -> False, 
	"Canonical" -> True, "AllBondsExplicit" -> False, IncludeHydrogens -> False, "RootedAtom" -> -1};


iSMILESString[imol_iMolecule, OptionsPattern[]] := Module[{isomeric, kekulize, rootedAtom, 
	canonical, explicitbonds, explicitHs, includedAtoms, numAtoms},
	isomeric = TrueQ @ OptionValue @ "Isomeric";
	kekulize = TrueQ @ OptionValue @ "Kekulized";
	canonical = TrueQ @ OptionValue @ "Canonical";
	explicitbonds = TrueQ @ OptionValue @ "AllBondsExplicit";
	explicitHs = TrueQ @ OptionValue @ IncludeHydrogens;
	
	numAtoms = imol["atomCount", True];
	rootedAtom = OptionValue["RootedAtom"] /. Except[_Integer?(0 < # <= numAtoms &)] :> 0;
	
	includedAtoms = Replace[
		OptionValue["IncludedAtoms"],
		All -> {}
	];
	imol[ "getSmiles", 
			isomeric,
			kekulize,
			rootedAtom,
			canonical,
			explicitbonds,
			explicitHs,
			includedAtoms
		]
]


(* ::Subsubsection::Closed:: *)
(*copyDescription*)


copyChemicalDataDescription[prop_] := (
	molDescription[prop] := ChemicalData[_, prop, "Description"];
	molLongDescription[prop] := ChemicalData[_, prop, "Description"];
)



(* ::Subsection::Closed:: *)
(*getiMolProperty*)


getiMolProperty[imol_, "coords2D", method_:Automatic] := If[
	has2DCoordinates[imol],
	imol[ "get2DCoordinates"],
	
	(* "bool canonicalize, mint nFlipsPerSample, mint nSamples, mint randomseed, bool permuteDegree4Nodes, bool heavyAtomsFirst" *)
	Switch[method,
		Automatic,
			embedDefault[imol];
			If[ !TrueQ[$ControlActiveSetting] && 
				has2DCoordinates[imol] && 
				imol["minDistance2D"] < 0.2
				,
				
				imol["clearConformers"];
				embedCoordGen[imol]
			],
		"CoordGen",
			embedCoordGen[imol],
		"RDKit",
			embedDefault[imol],
		_String,
			Replace[
				 tryGraphEmbedding2D[ imol, method],
				$Failed :> Message[MoleculeModify::method, method]
			]
	];
	If[
		has2DCoordinates[imol],
		alignConnectedComponents[imol,2];
		imol[ "get2DCoordinates"],
		Missing["NotAvailable"]
	]
]

embedCoordGen[imol_] := 
	Replace[
		imol[ "generate2DCoordinatesCoordgen", <| "TemplateFile" -> Chemistry`RDKitLink`$MolTemplateFile |>],
		_LibraryFunctionError :> (
			embedDefault[imol]
		)
	]
	
embedDefault[imol_] := 
	(* "bool canonicalize, mint nFlipsPerSample, mint nSamples, mint randomseed, bool permuteDegree4Nodes, bool heavyAtomsFirst" *)
	imol[ "generate2DCoordinates", False, 0, 0, 0, False, False];

tryGraphEmbedding2D[ imol_, method_] := Module[
	{coords},
	coords = GraphEmbedding @ Graph[
		Range @ imol["atomCount", True],
		UndirectedEdge[#1,#2] & @@@ imol["getBondsList"],
		GraphLayout -> method
	];
	
	imol["addCoordinates", coords] /; MatrixQ[coords, NumericQ]
 
]

tryGraphEmbedding2D[___] := $Failed;

getiMolProperty[imol_, "coords3D", coordinateOption_:{}] := Block[
	{},
	If[
		has3DCoordinates[imol],
		imol[ "get3DCoordinates"],
		If[ 
			generateDefault3DCoordinates[ imol, coordinateOption],
			imol[ "get3DCoordinates"],
			Missing["NotAvailable"]
		]
	]
]


getiMolProperty[imol_, "BondList", kekule_:False] := Module[{bonds, orders},
	bonds = imol["getBondsList"];
	If[ 
		kekule,
		orders = imol[ "kekuleBonds"];
		If[ 
			SameQ @@ Map[Length,{bonds,orders}],
			bonds[[All, -1]] = orders
		] 
	];
	bonds = Bond[{#1, #2}, bondOrderMap[#3]] & @@@ bonds;
	bonds
]

getiMolProperty[imol_, "AtomList" ] := Block[
	{atoms,properties},
	atoms = Atom@*numberElementMap /@ imol["getAtomsList"];
	properties = getiMolProperty[imol, "_atomproperties"];
	If[
		Length @ properties === 0,
		Return[ atoms, Block]
	];
	
	KeyValueMap[
		(atoms[[#1]] =  Atom[ atoms[[#1,1]], Sequence @@ Normal @ #2 ])&,
		properties
	];
	atoms
]



getiMolProperty[imol_, "_atomproperties"] := Module[{props},
	props = imol[ "atomPropertyAssociation"];
	props = props // Replace[Null -> <||>] // 
				KeyMap[FromDigits] // KeySort;
	props
]



getiMolProperty[ imol_, "StereochemistryElements"] := Block[ 
	{keys, chiral, bonds},
	bonds = imol[ "getStereoBonds"] /. Null -> {};
	bonds = Association[
		"StereoType" -> "DoubleBond", 
		"StereoBond" -> {#1, #2}, 
		"Ligands" -> {#4,#5}, 
		"Value" ->  Lookup[ bondStereoMap, #3]
	] & @@@ bonds;
	chiral = imol[ "getChiralElements"] /. Null -> {};
	chiral = Association[
		"StereoType" -> "Tetrahedral",
		"ChiralCenter" -> #1,
		"Direction" -> #2,
		"FiducialAtom" -> #3,
		"Ligands" -> {##4} 
	]& @@@ chiral;
	Join[
		chiral,
		bonds
	]
]

getiMolProperty[___] := $Failed


(* ::Subsection::Closed:: *)
(*Fix me*)

$chi1234 = "\!\(\*FormBox[SubscriptBox[\"\[Chi]\", \"1234\"], TraditionalForm]\)";




addExposedProperty["TorsionAngle", "GeometricProperties"]
molDescription["TorsionAngle"] = "torsion angle"
molLongDescription["TorsionAngle"] = "torsion angle Chi1234 between the plane containing atoms {1,2,3} and the plane containing atoms {2,3,4}"
addPropertyAlias["TorsionAngle",  "DihedralAngle"]

addExposedProperty["OutOfPlaneAngle", "GeometricProperties"]
molDescription["OutOfPlaneAngle"] = "out-of-plane angle"
molLongDescription["OutOfPlaneAngle"] = "Wilson angle "<>$chi1234<>" between the plane containing atoms {1,2,3} and the bond containing atoms 2 and 4"


addExposedProperty["BondAngle", "GeometricProperties"]
molDescription["BondAngle"] = "bond angle"
molLongDescription["BondAngle"] = "bond angle"

addExposedProperty["InteratomicDistance", "GeometricProperties"]
molDescription["InteratomicDistance"] = "interatomic distance"
molLongDescription["InteratomicDistance"] = "Euclidean distance between atoms"

setPerAtomProperty /@ {"OutOfPlaneAngle","BondAngle","InteratomicDistance","TorsionAngle"}
molValue[ mol_, mp[{prop:("OutOfPlaneAngle"|"InteratomicDistance"|"TorsionAngle"|"BondAngle"), atoms_:All}, opts_]] := 
	iGeometricProperty[prop, {mol, atoms},MoleculeValue]






molValue[args___] := (
	Replace[
		{args},
		{_, mp[prop_String,___]|mp[{prop_String,___},___] ,___} /; !MemberQ[exposedProperties[], prop] :> 
			Message[ MoleculeValue::notprop, prop, MoleculeValue]
		];
	$Failed
)	
		
(*molValue[___] := $Failed*)
molQualifiers[___] := {};


(* ::Subsection::Closed:: *)
(*3D Descriptors*)


setMolPropertyValue["NormalizedPrincipalMomentRatios", "jsonProperty3D"]
addExposedProperty["NormalizedPrincipalMomentRatios", "3DDescriptors"]
molDescription["NormalizedPrincipalMomentRatios"] = "normalized principal moment ratios"
molLongDescription["NormalizedPrincipalMomentRatios"] = "normalized principal moment ratios, equal to {I1/I3, I2/I3} where Ij is the jth principal moment of inertia."
molPropertyCitation["NormalizedPrincipalMomentRatios"] = "https://dx.doi.org/10.1021/ci025599w"
addPropertyAlias["NormalizedPrincipalMomentRatios",  "NPR1"]


setMolPropertyValue["PlaneOfBestFitDistance", "jsonProperty3D"]
addExposedProperty["PlaneOfBestFitDistance", "3DDescriptors"]
molDescription["PlaneOfBestFitDistance"] = "plane of best fit distance"
molLongDescription["PlaneOfBestFitDistance"] = "average distance from heavy atoms to the plane of best fit"
molPropertyCitation["PlaneOfBestFitDistance"] = "https://dx.doi.org/10.1021%2Fci300293f"
addPropertyAlias["PlaneOfBestFitDistance",  "PBF"]


setMolPropertyValue["InertialShapeFactor", "jsonProperty3D"]
addExposedProperty["InertialShapeFactor", "3DDescriptors"]
molDescription["InertialShapeFactor"] = "inertial shape factor"
molPropertyCitation["InertialShapeFactor"] = "http://dx.doi.org/10.1002/9783527618279.ch37"
molLongDescription["InertialShapeFactor"] = "inertial shape factor defined as the ratio of the second principal moment divided by the product of the first and third principal moments"


setMolPropertyValue["RadiusOfGyration", "jsonProperty3D"]
addExposedProperty["RadiusOfGyration", "3DDescriptors"]
molDescription["RadiusOfGyration"] = "radius of gyration"
molLongDescription["RadiusOfGyration"] = "radius of gyration"
molPropertyCitation["RadiusOfGyration"] = "http://dx.doi.org/10.1002/9780470125861.ch5"


setMolPropertyValue["Eccentricity", "jsonProperty3D"]
addExposedProperty["Eccentricity", "3DDescriptors"]
molDescription["Eccentricity"] = "eccentricity"
molPropertyCitation["Eccentricity"] = "http://dx.doi.org/10.1002/9780470125861.ch5"


setMolPropertyValue["Asphericity", "jsonProperty3D"]
addExposedProperty["Asphericity", "3DDescriptors"]
molDescription["Asphericity"] = "asphericity"
molPropertyCitation["Asphericity"] = "http://dx.doi.org/10.1063/1.464689"
(*molLongDescription["Asphericity"] = "asphericity"*)


setMolPropertyValue["SpherosityIndex", "jsonProperty3D"]
addExposedProperty["SpherosityIndex", "3DDescriptors"]
molDescription["SpherosityIndex"] = "spherosity index"
molPropertyCitation["SpherosityIndex"] = "http://dx.doi.org/10.1002/9783527618279.ch37"
molLongDescription["SpherosityIndex"] = "spherosity index"


setMolPropertyValue["WHIM", "realVectorProperty"]
addExposedProperty["WHIM", "3DDescriptors"]
molDescription["WHIM"] = "weighted holistic invariant molecular descriptors"
molPropertyCitation["WHIM"] = "https://doi.org/10.1016/0169-7439(95)80026-6"


setMolPropertyValue["RDF", "realVectorProperty"]
addExposedProperty["RDF", "3DDescriptors"]
molDescription["RDF"] = "radial distribution function"
molLongDescription["RDF"] = "radial distribution function descriptors"
addPropertyAlias["RDF",  "RadialDistributionFunction"]
molPropertyCitation["RDF"] = "https://doi.org/10.1016/S0924-2031(99)00014-4"


setMolPropertyValue["MORSE", "realVectorProperty"]
addExposedProperty["MORSE", "3DDescriptors"]
molDescription["MORSE"] = "MORSE descriptors"
molLongDescription["MORSE"] = "3D-Molecule Representation of Structures based on Electron diffraction"
molPropertyCitation["MORSE"] = "http://dx.doi.org/10.1021/ci950164c"

setMolPropertyValue["GETAWAY", "realVectorProperty"]
addExposedProperty["GETAWAY", "3DDescriptors"]
molDescription["GETAWAY"] = "GETAWAY descriptor"
molLongDescription["GETAWAY"] = "geometry, topology, and atom-weights assembly descriptors"
molPropertyCitation["GETAWAY"] = "http://dx.doi.org/10.1021/ci015504a"

setMolPropertyValue["Autocorrelation3D", "realVectorProperty"]
addExposedProperty["Autocorrelation3D", "3DDescriptors"]
molDescription["Autocorrelation3D"] = "3D autocorrelation"
molPropertyCitation["Autocorrelation3D"] = "http://dx.doi.org/10.1002/9783527618279.ch37"


(* ::Subsection::Closed:: *)
(*Identifiers*)


molDescription["InChI"] = "international chemical identifier"
addExposedProperty["InChI", "Identifiers"]
setMolPropertyValue["InChI", "jsonProperty"]
molPropertyCitation["InChI"] = "https://doi.org/10.1186%2F1758-2946-5-7"


molDescription["InChIKey"] = "international chemical identifier key"
addExposedProperty["InChIKey", "Identifiers"]
setMolPropertyValue["InChIKey", "jsonProperty"]
molPropertyCitation["InChIKey"] = "https://doi.org/10.1186%2F1758-2946-5-7"



molQualifiers["SMILES"] = Options[iSMILESString]
molDescription["SMILES"] = "SMILES string"
addPropertyAlias["SMILES","CanonicalSMILES"]
molLongDescription["SMILES"] = "canonicalized SMILES string"
addExposedProperty["SMILES", "Identifiers"]
molValue[mol_, mp["SMILES" , opts_]] := iSMILESString[ getCachedMol[mol,"NoHydrogens"] , "Canonical" -> True]

molValue[mol_, mp[{"SMILES",atoms:(All|{__Integer})}] ] := Module[
	{processedAtoms, imol = getCachedMol[mol]},
	processedAtoms = getAtomsList[ imol, atoms];
	If[
		!VectorQ[ processedAtoms, IntegerQ ]
		,
		Return[ $Failed, Module]
	];
	
	iSMILESString[
		imol, "IncludedAtoms" -> processedAtoms, 
		"RootedAtom" -> First[processedAtoms,0], "Canonical" -> False
	]
]



molPropertyCitation["SMILES"] = "https://doi.org/10.1021/ci00057a005"

(* ::Subsection::Closed:: *)
(*TopologicalDescriptors*)


$jsonProperties = {
	"MolecularFormula", "InChI", "InChIKey", "HeterocycleCount", "AromaticHeterocycleCount",
	"AromaticCarbocycleCount", "SaturatedHeterocycleCount", "SaturatedCarbocycleCount",
	"AliphaticHeterocycleCount", "AliphaticCarbocycleCount", "RingCount", "AromaticRingCount",
	"AliphaticRingCount", "SaturatedRingCount", "SpiroAtomCount", "BridgeheadAtomCount",
	"StereocenterCount", "UnspecifiedStereocenterCount", "LipinskiHBondAcceptorCount",
	"LipinskiHBondDonorCount", "RotatableBondCount", "HBondDonorCount", "HBondAcceptorCount",
	"HeteroatomCount", "AmideBondCount", "MolecularMass", "MonoIsotopicMolecularMass",
	"LabuteApproximateSurfaceArea", "TopologicalPolarSurfaceArea", "CrippenClogP",
	"CrippenMR", "FractionCarbonSP3", "KierHallAlphaShape","Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n", 
	"Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "Kappa1", "Kappa2", "Kappa3"
} 

molDescription["HeterocycleCount"] = "heterocycle count"
addExposedProperty["HeterocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["HeterocycleCount", "jsonProperty"]


molDescription["AromaticHeterocycleCount"] = "aromatic heterocycle count"
addExposedProperty["AromaticHeterocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["AromaticHeterocycleCount", "jsonProperty"]


molDescription["AromaticCarbocycleCount"] = "aromatic carbocycle count"
addExposedProperty["AromaticCarbocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["AromaticCarbocycleCount", "jsonProperty"]


molDescription["SaturatedHeterocycleCount"] = "saturated heterocycle count"
addExposedProperty["SaturatedHeterocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["SaturatedHeterocycleCount", "jsonProperty"]


molDescription["SaturatedCarbocycleCount"] = "saturated carbocycle count"
addExposedProperty["SaturatedCarbocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["SaturatedCarbocycleCount", "jsonProperty"]


molDescription["AliphaticHeterocycleCount"] = "aliphatic heterocycle count"
addExposedProperty["AliphaticHeterocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["AliphaticHeterocycleCount", "jsonProperty"]


molDescription["AliphaticCarbocycleCount"] = "aliphatic carbocycle count"
addExposedProperty["AliphaticCarbocycleCount", "TopologicalDescriptors"]
setMolPropertyValue["AliphaticCarbocycleCount", "jsonProperty"]


molDescription["RingCount"] = "ring count"
addExposedProperty["RingCount", "TopologicalDescriptors"]
setMolPropertyValue["RingCount", "jsonProperty"]


molDescription["AromaticRingCount"] = "aromatic ring count"
addExposedProperty["AromaticRingCount", "TopologicalDescriptors"]
setMolPropertyValue["AromaticRingCount", "jsonProperty"]


molDescription["AliphaticRingCount"] = "aliphatic ring count"
addExposedProperty["AliphaticRingCount", "TopologicalDescriptors"]
setMolPropertyValue["AliphaticRingCount", "jsonProperty"]


molDescription["SaturatedRingCount"] = "saturated ring count"
addExposedProperty["SaturatedRingCount", "TopologicalDescriptors"]
setMolPropertyValue["SaturatedRingCount", "jsonProperty"]


molDescription["SpiroAtomCount"] = "spiro atom count"
addExposedProperty["SpiroAtomCount", "TopologicalDescriptors"]
setMolPropertyValue["SpiroAtomCount", "jsonProperty"]


molDescription["BridgeheadAtomCount"] = "bridgehead atom count"
addExposedProperty["BridgeheadAtomCount", "TopologicalDescriptors"]
setMolPropertyValue["BridgeheadAtomCount", "jsonProperty"]


molDescription["StereocenterCount"] = "stereocenter count"
molLongDescription["StereocenterCount"] = "the number of possible stereocenters"
addExposedProperty["StereocenterCount", "TopologicalDescriptors"]
setMolPropertyValue["StereocenterCount", "jsonProperty"]


molDescription["UnspecifiedStereocenterCount"] = "unspecified stereocenter count"
addExposedProperty["UnspecifiedStereocenterCount", "TopologicalDescriptors"]
setMolPropertyValue["UnspecifiedStereocenterCount", "jsonProperty"]


molDescription["LipinskiHBondAcceptorCount"] = "Lipinski H-bond acceptor count"
addExposedProperty["LipinskiHBondAcceptorCount", "TopologicalDescriptors"]
setMolPropertyValue["LipinskiHBondAcceptorCount", "jsonProperty"]


molDescription["LipinskiHBondDonorCount"] = "Lipinski H-bond donor count"
addExposedProperty["LipinskiHBondDonorCount", "TopologicalDescriptors"]
setMolPropertyValue["LipinskiHBondDonorCount", "jsonProperty"]


molDescription["RotatableBondCount"] = "rotatable bond count"
addExposedProperty["RotatableBondCount", "TopologicalDescriptors"]
setMolPropertyValue["RotatableBondCount", "jsonProperty"]


molDescription["HBondDonorCount"] = "H-bond donor count"
molLongDescription["HBondDonorCount"] = "number of hydrogen bond donors"
addExposedProperty["HBondDonorCount", "TopologicalDescriptors"]
setMolPropertyValue["HBondDonorCount", "jsonProperty"]


molDescription["HBondAcceptorCount"] = "H-bond acceptor count"
molLongDescription["HBondAcceptorCount"] = "number of hydrogen bond acceptors"
addExposedProperty["HBondAcceptorCount", "TopologicalDescriptors"]
setMolPropertyValue["HBondAcceptorCount", "jsonProperty"]


molDescription["HeteroatomCount"] = "heteroatom count"
addExposedProperty["HeteroatomCount", "TopologicalDescriptors"]
setMolPropertyValue["HeteroatomCount", "jsonProperty"]


molDescription["AmideBondCount"] = "amide bond count"
addExposedProperty["AmideBondCount", "TopologicalDescriptors"]
setMolPropertyValue["AmideBondCount", "jsonProperty"]


molDescription["LabuteApproximateSurfaceArea"] = "Labute approximate surface area"
addExposedProperty["LabuteApproximateSurfaceArea", "TopologicalDescriptors"]
setMolPropertyValue["LabuteApproximateSurfaceArea", "jsonProperty"]


molDescription["TopologicalPolarSurfaceArea"] = "topological polar surface area"
addExposedProperty["TopologicalPolarSurfaceArea", "TopologicalDescriptors"]
setMolPropertyValue["TopologicalPolarSurfaceArea", "jsonProperty"]


molDescription["CrippenClogP"] = "Crippen calculated log P"
addExposedProperty["CrippenClogP", "TopologicalDescriptors"]
setMolPropertyValue["CrippenClogP", "jsonProperty"]


molDescription["CrippenMR"] = "Crippen molar refactivity"
addExposedProperty["CrippenMR", "TopologicalDescriptors"]
setMolPropertyValue["CrippenMR", "jsonProperty"]


molDescription["FractionCarbonSP3"] = "fraction carbon sp3"
molLongDescription["FractionCarbonSP3"] = "fraction of carbon atoms with sp3 hybridization"
addExposedProperty["FractionCarbonSP3", "TopologicalDescriptors"]
setMolPropertyValue["FractionCarbonSP3", "jsonProperty"]


molDescription["KierHallAlphaShape"] = "Kier & Hall modified shape index"
addExposedProperty["KierHallAlphaShape", "TopologicalDescriptors"]
setMolPropertyValue["KierHallAlphaShape", "jsonProperty"]


molDescription["MolecularQuantumNumbers"] = "molecular quantum numbers"
molLongDescription["MolecularQuantumNumbers"] = "molecular quantum numbers"
addExposedProperty["MolecularQuantumNumbers", "TopologicalDescriptors"]
setMolPropertyValue["MolecularQuantumNumbers", "intVectorProperty"]
molPropertyCitation["MolecularQuantumNumbers"] = "https://doi.org/10.1002/cmdc.200900317"

molDescription["SlogPVSA"] = "Crippen log P van der Waals surface area"
addExposedProperty["SlogPVSA", "TopologicalDescriptors"]
addPropertyAlias["SlogPVSA","SlogP_VSA"]
setMolPropertyValue["SlogPVSA", "realVectorProperty"]


molDescription["SMRVSA"] = "Crippen molar refractivity van der Waals surface area"
addExposedProperty["SMRVSA", "TopologicalDescriptors"]
addPropertyAlias["SMRVSA","SMR_VSA"]
setMolPropertyValue["SMRVSA", "realVectorProperty"]


molDescription["PEOEVSA"] = "partial equalization of orbital electronegativity van der Waals surface area"
addExposedProperty["PEOEVSA", "TopologicalDescriptors"]
addPropertyAlias["PEOEVSA","PEOE_VSA"]
setMolPropertyValue["PEOEVSA", "realVectorProperty"]


molDescription["Autocorrelation2D"] = "2D autocorrelation"
addExposedProperty["Autocorrelation2D", "TopologicalDescriptors"]
setMolPropertyValue["Autocorrelation2D", "realVectorProperty"]

molDescription["Chi0n"] = "zeroth-order chi index"
addExposedProperty["Chi0n", "TopologicalDescriptors"]
setMolPropertyValue["Chi0n", "jsonProperty"]


molDescription["Chi1n"] = "first-order chi index"
addExposedProperty["Chi1n", "TopologicalDescriptors"]
setMolPropertyValue["Chi1n", "jsonProperty"]


molDescription["Chi2n"] = "second-order chi index"
addExposedProperty["Chi2n", "TopologicalDescriptors"]
setMolPropertyValue["Chi2n", "jsonProperty"]


molDescription["Chi3n"] = "third-order chi index"
addExposedProperty["Chi3n", "TopologicalDescriptors"]
setMolPropertyValue["Chi3n", "jsonProperty"]


molDescription["Chi4n"] = "fourth-order chi index"
molLongDescription["Chi4n"] = "Kier and Hall connectivity indices"
addExposedProperty["Chi4n", "TopologicalDescriptors"]
setMolPropertyValue["Chi4n", "jsonProperty"]


molDescription["Chi0v"] = "zeroth-order valence chi index"
addExposedProperty["Chi0v", "TopologicalDescriptors"]
setMolPropertyValue["Chi0v", "jsonProperty"]


molDescription["Chi1v"] = "first-order valence chi index"
addExposedProperty["Chi1v", "TopologicalDescriptors"]
setMolPropertyValue["Chi1v", "jsonProperty"]


molDescription["Chi2v"] = "second-order valence chi index"
addExposedProperty["Chi2v", "TopologicalDescriptors"]
setMolPropertyValue["Chi2v", "jsonProperty"]


molDescription["Chi3v"] = "third-order valence chi index"
addExposedProperty["Chi3v", "TopologicalDescriptors"]
setMolPropertyValue["Chi3v", "jsonProperty"]


molDescription["Chi4v"] = "fourth-order valence chi index"
molLongDescription["Chi4v"] = "Kier and Hall valence connectivity indices"
addExposedProperty["Chi4v", "TopologicalDescriptors"]
setMolPropertyValue["Chi4v", "jsonProperty"]


molDescription["Kappa1"] = "first-order shape index"
addExposedProperty["Kappa1", "TopologicalDescriptors"]
setMolPropertyValue["Kappa1", "jsonProperty"]


molDescription["Kappa2"] = "second-order shape index"
addExposedProperty["Kappa2", "TopologicalDescriptors"]
setMolPropertyValue["Kappa2", "jsonProperty"]


molDescription["Kappa3"] = "third-order shape index"
addExposedProperty["Kappa3", "TopologicalDescriptors"]
setMolPropertyValue["Kappa3", "jsonProperty"]



molDescription["AromaticMoleculeQ"] = "aromatic molecule Q"
addExposedProperty["AromaticMoleculeQ", "TopologicalDescriptors"]
molValue[ mol_, mp["AromaticMoleculeQ",{}]] := Module[
	{im = getCachedMol[mol]},
	im["aromaticMoleculeQ"]
]; 


(* ::Subsection::Closed:: *)
(*GeneralProperties*)


molValue[mol_, mp["inactive"]] := MapAt[Inactive, mol, {0} ]


molQualifiers["LibraryID"] = {IncludeHydrogens -> Automatic};
molValue[mol_, mp["LibraryID", opts_]] := ManagedLibraryExpressionID @ getCachedMol[ mol, opts];

molQualifiers["ManagedLibraryExpression"] = {IncludeHydrogens -> Automatic};
molValue[mol_, mp["ManagedLibraryExpression", opts_]] := getCachedMol[ mol, opts]

setMolPropertyValue["_debugString", "jsonProperty"]



molDescription["MetaInformation"] = "meta-information"
molLongDescription["MetaInformation"] = "any meta-information contained in the molecule expression"
addExposedProperty["MetaInformation","GeneralProperties"]
molValue[mol_, mp["MetaInformation"|{"MetaInformation",keys___}]] := With[
	{res = OptionValue[Molecule, Options @ mol, MetaInformation]},
	res[keys] /; AssociationQ[res]
]



molDescription["AtomCount"] = "atom count"
molLongDescription["AtomCount"] = "number of atoms in the molecule expression"
addExposedProperty["AtomCount","GeneralProperties"]
molValue[mol_, mp["AtomCount"]] := getCachedMol[mol]["atomCount", True]


molDescription["FullAtomCount"] = "full atom count"
molLongDescription["FullAtomCount"] = "number of atoms in the molecule, including implicit hydrogens"
addExposedProperty["FullAtomCount","GeneralProperties"]
molValue[mol_, mp["FullAtomCount"]] := getCachedMol[mol]["atomCount", False]


molDescription["AtomList"] = "atom list"
molLongDescription["AtomList"] = "list of atoms in the molecule expression"
addExposedProperty["AtomList","GeneralProperties"]
molValue[mol_, mp["AtomList"]] := getiMolProperty[ getCachedMol[mol], "AtomList"]


molDescription["FullAtomList"] = "full atom list"
molLongDescription["FullAtomList"] = "list of atoms in the molecule, including implicit hydrogens"
addExposedProperty["FullAtomList","GeneralProperties"]
molValue[mol_, mp["FullAtomList"]] := getiMolProperty[ getCachedMol[mol, "AllAtoms"], "AtomList"]


molDescription["BondCount"] = "bond count"
molLongDescription["BondCount"] = "number of bonds in the molecule expression"
addExposedProperty["BondCount","GeneralProperties"]
molValue[mol_, mp["BondCount"]] := getCachedMol[mol]["bondCount", True]


molDescription["FullBondCount"] = "full bond count"
molLongDescription["FullBondCount"] = "number of bonds in the molecule, including implicit hydrogens"
addExposedProperty["FullBondCount","GeneralProperties"]
molValue[mol_, mp["FullBondCount"]] := getCachedMol[mol]["bondCount", False]


molDescription["BondList"] = "bond list"
molLongDescription["BondList"] = "list of bonds in the molecule expression"
addExposedProperty["BondList","GeneralProperties"]
molValue[mol_, mp["BondList"]] := getiMolProperty[ getCachedMol[mol], "BondList"]


molDescription["FullBondList"] = "full bond list"
molLongDescription["FullBondList"] = "list of bonds in the molecule, including implicit hydrogens"
addExposedProperty["FullBondList","GeneralProperties"]
molValue[mol_, mp["FullBondList"]] := getiMolProperty[ getCachedMol[mol, "AllAtoms"], "BondList"]



molDescription["StereochemistryElements"] = "stereochemistry elements"
molLongDescription["StereochemistryElements"] = "list of associations detailing any defined local stereochemistry as defined by StereochemistryElements"
addExposedProperty["StereochemistryElements", "GeneralProperties"]
molValue[ mol_, mp["StereochemistryElements"]] := Module[
	{im},
	If[
		hasImplicitHydrogens[mol]
		,
		im = getCachedMol[mol,"AllAtoms"];
		getiMolProperty[im,"StereochemistryElements"]
		,
		Lookup[
			Options[mol],
			StereochemistryElements,
			{}
		]
	]
]


molDescription["MolecularFormulaString"] = "molecular formula string"
addPropertyAlias["MolecularFormulaString", "FormulaString"]
addExposedProperty["MolecularFormulaString", "Identifiers"]
setMolPropertyValue["MolecularFormulaString", "jsonProperty"]

molDescription["MolecularFormula"] = "molecular formula"
addPropertyAlias["MolecularFormula", "Formula"]
addExposedProperty["MolecularFormula", "Identifiers"]
molValue[ mol_, mp["MolecularFormula"] ] := Module[
	{formulaString = molValue[ mol, mp["MolecularFormulaString"]]},
	Replace[StringExpression[x_] :> x] @ StringReplace[
		formulaString,
		{
			StartOfString ~~ formula__ ~~ sign : ("+" | "-") ~~ chrg : NumberString  ~~ EndOfString :>
				Replace[
					findFormulaDisplay[formula],
					Row[{f__}] :> Row[{f,Superscript["\[InvisiblePrefixScriptBase]",Row[{chrg,sign}]]}]
				],
			StartOfString ~~ formula__ ~~ sign : ("+" | "-") ~~ EndOfString :>
				Replace[
					findFormulaDisplay[formula],
					Row[{f__}] :> Row[{f,Superscript["\[InvisiblePrefixScriptBase]",sign]}]
				],
			StartOfString ~~ f__ ~~ EndOfString :> Chemistry`Private`MoleculeValueDump`findFormulaDisplay[f]
		}
	] /; StringQ[formulaString]
]

molDescription["MolecularMass"] = "molecular mass"
molLongDescription["MolecularMass"] = "molecular mass, using the average atomic mass for atoms with no specified mass number"
addExposedProperty["MolecularMass", "GeneralProperties"]
setMolPropertyValue["MolecularMass", "jsonProperty"]

molDescription["MonoIsotopicMolecularMass"] = "mono-isotopic molecular mass"
molLongDescription["MonoIsotopicMolecularMass"] = "molecular mass, using the most abundant isotope for all atoms"
addExposedProperty["MonoIsotopicMolecularMass", "GeneralProperties"]
setMolPropertyValue["MonoIsotopicMolecularMass", "jsonProperty"]

molDescription["PossibleStereocenters"] = "possible stereocenters"
molLongDescription["PossibleStereocenters"] = "list of atom indices for chiral atoms"
addExposedProperty["PossibleStereocenters", "GeneralProperties"]
setMolPropertyValue["PossibleStereocenters", "intVectorProperty"]

molDescription["TotalCharge"] = "total charge"
molLongDescription["TotalCharge"] = "sum of all formal charges"
addExposedProperty["TotalCharge","GeneralProperties"]
molValue[mol_, mp["TotalCharge"]] := With[
	{chrg = molValue[mol,mp["FormalCharge"]]},
	Total @ chrg
]

copyChemicalDataDescription["ElementTally"]
molLongDescription["ElementTally"] = "list {{sym,n}...} of atomic symbols and the corresponding number of atoms"
addExposedProperty["ElementTally","GeneralProperties"]
molValue[mol_, mp["ElementTally"]] := With[
	{sym = molValue[mol,mp["AtomicSymbol"]]},
	Tally @ sym
]

molLongDescription["ElementCounts"] = "association <|sym->n,...|> of atomic symbols and the corresponding number of atoms"
addExposedProperty["ElementCounts","GeneralProperties"]
molValue[mol_, mp["ElementCounts"]] := With[
	{sym = molValue[mol,mp["AtomicSymbol"]]},
	Counts @ sym
]

copyChemicalDataDescription["ElementMassFraction"]
molLongDescription["ElementMassFraction"] = "list {{sym,n}...} of atomic symbols and the corresponding mass fraction as a percentage"
addExposedProperty["ElementMassFraction","GeneralProperties"]
molValue[mol_, mp["ElementMassFraction"]] := Module[
	{syms = molValue[mol,mp["ElementTally"]], masses},
	masses = {#1, #2 ElementData[#1,"AtomicMass"]} & @@@ syms;
	With[{total = Total[masses[[All,2]]]},
		{#1, quantity[ 100 #2/total, "Percent"]} & @@@ masses
	]
			
	
]

molDescription["AtomDiagramCoordinates"] = "structure diagram coordinates"
addExposedProperty["AtomDiagramCoordinates", "GeneralProperties"]
setPerAtomProperty @ "AtomDiagramCoordinates";
molValue[$EmptyMolecule, mp["AtomDiagramCoordinates",___]] := {{}}
molValue[mol_, mp[{"AtomDiagramCoordinates", atoms_:All}, opts_] ] := Module[
	{imol, coords, method, processedAtoms},
	imol = getCachedMol[mol, opts];
	
	method = Lookup[
		Options[ mol],
		AtomDiagramCoordinates,
		Automatic
	];
	coords = getiMolProperty[ imol, "coords2D", method];
	
	If[
		atoms =!= All
		,
		processedAtoms = Replace[
			getAtomsList[ imol, imol["atomCount", True], atoms],
			Except[_List] :> Return[$Failed, Module]
		];
		coords[[processedAtoms]]
		,
		coords
	] /; MatrixQ[ coords]
]


molDescription["MMFFEnergy"] = "MMFF energy"
molLongDescription["MMFFEnergy"] = "energy computed using the MMFF94 force field"
addExposedProperty["MMFFEnergy", "GeneralProperties"]

molDescription["MMFFsEnergy"] = "MMFFs energy"
molLongDescription["MMFFsEnergy"] = "energy computed using the MMFF94/MMFF94s force field"
addExposedProperty["MMFFsEnergy", "GeneralProperties"]


molQualifiers[ "MMFFEnergy"] = { "AtomCoordinates" -> Automatic}
molValue[ mol_, mp[prop:("MMFFEnergy"|"MMFFsEnergy"), opts_] ] := Module[
	{im = getCachedMol[ mol, "AllAtoms"], res, coords, variant},
	coords = Replace[
		OptionValue[ molQualifiers[ "MMFFEnergy"], {opts}, "AtomCoordinates"],
		{
			Automatic :> ( check3DProperty["MMFFEnergy"][ mol, im, $tag]; im["get3DCoordinates"]),
			qa_?quantityArrayQ :> QuantityMagnitude[ qa]
		}
	];
	If[
		Dimensions[coords] =!= { im["atomCount",True], 3},
		Message[ Molecule::coord, "AtomCoordinates", coords];
		Return[ Missing["NotAvailable"], Module]
	];
	
	variant = Switch[prop,
		"MMFFEnergy",
			<|"MMFFVariant" -> "MMFF94" |>,
		_,
			<|"MMFFVariant" -> "MMFF94s" |>
	];
	
	res = If[
		MatchQ[ im["initializeMMFF", variant], _LibraryFunctionError]
		,
		Missing["NotAvailable"] 
		,
		quantity[
			im["calcEnergy", coords ],
			"KilocaloriesThermochemical"/"Moles"
		]
	];
	im["clearMMFF"];
	res
]


molDescription["pickle"] = "pickle"
molLongDescription["pickle"] = "serialized pickle string"
molQualifiers[ "pickle"] = { IncludeHydrogens -> Automatic}
molValue[mol_, mp["pickle", opts_]] := Module[
	{im = getCachedMol[mol, opts] },
	im["getPickle"]
]


molValue[mol_, mp["Embedded"]] := MatrixQ[Lookup[ Options[ mol], AtomCoordinates] ];
molValue[mol_, mp["HasImplicitHydrogens"]] := TrueQ @ getCachedMol[mol]["hasImplicitHydrogens"]

molValue[mol_, mp["HasValenceErrors"]] := !TrueQ @ getCachedMol[mol]["validProperties"]

(* ::Subsection::Closed:: *)
(*GeometricProperties*)


molDescription["DistanceMatrix"] = "distance matrix"
molLongDescription["DistanceMatrix"] = "matrix of Euclidean distances between atom centers"
addExposedProperty["DistanceMatrix", "GeometricProperties"]
molQualifiers[ "DistanceMatrix"] = {TargetUnits -> "Angstroms"}
molValue[ mol_, mp["DistanceMatrix", opts_]] := Module[
	{coords},
	coords = Replace[
		molValue[mol, mp["AtomCoordinates",opts] ],
		Except[x_?MatrixQ /; Length[x[[1]]] === 3] :> Return[Missing["NotAvailable"], Module]
	];
	
	If[
		quantityArrayQ[coords]
		,
		quantityArray[
			DistanceMatrix @ QuantityMagnitude @ coords,
			"Angstroms"
		]
		,
		DistanceMatrix @ coords
	]
]


molDescription["CoulombMatrix"] = "Coulomb matrix"
molLongDescription["CoulombMatrix"] = "Coulomb matrix"
addExposedProperty["CoulombMatrix", "GeometricProperties"]
molValue[ mol_, mp["CoulombMatrix"]] := Module[
	{im = getCachedMol[mol, "AllAtoms"], natoms, res},
	check3DProperty["CoulombMatrix"][ mol, im, $tag];
	res = im["realMatrixProperty", "CoulombMatrix"];
	natoms = Length @ First @ mol;
	If[ 
		TrueQ[ Length @ res > natoms], 
		Take[res,natoms,natoms], 
		res
	]
]

molDescription["AtomCoordinates"] = "atom coordinates"
molLongDescription["AtomCoordinates"] = "list of Cartesian atomic coordinates"
addExposedProperty["AtomCoordinates", "GeometricProperties"]
setPerAtomProperty @ "AtomCoordinates"
molQualifiers["AtomCoordinates"] = {TargetUnits -> "Angstroms"}
molValue[$EmptyMolecule, mp[{"AtomCoordinates",_},___]] := {{}}


molValue[mol_, mp[{"AtomCoordinates", atoms_:All}, opts_]] := Module[
	{imol = getCachedMol[mol, "AllAtoms"],coords, processedAtoms, coordinateOption, unit},
	
	coordinateOption = getMolOption[mol, AtomCoordinates];
	coords = getiMolProperty[ imol, "coords3D", coordinateOption];
	
	If[
		!MatrixQ[ coords, NumberQ],
		Return[Missing["NotAvailable"], Module]
	];	
	
	If[
		atoms =!= All
		,
		processedAtoms = Replace[
			getAtomsList[ imol, imol["atomCount", True], atoms],
			Except[_List] :> Return[$Failed, Module]
		];
		coords = coords[[processedAtoms]];
	];
	unit = OptionValue[ molQualifiers["AtomCoordinates"], {opts}, TargetUnits];
	Which[ 
		IntegerQ[atoms], 
			coords = First @ coords;,
		atoms === All && (atomCount = getCachedMol[mol]["atomCount",True]) =!= Length[coords],
			coords = Take[coords, atomCount];
	];
	Switch[unit,
		None,
			coords,
		"Angstroms",
			quantityArray[coords, unit],
		_String,
			UnitConvert[ quantityArray[coords, "Angstroms"], unit]
	]
		 
]



addExposedProperty["CenterOfMass", "GeometricProperties"]
molDescription["CenterOfMass"] = "center of mass"
setPerAtomProperty @ "CenterOfMass"
molValue[mol_, mp[{"CenterOfMass", atoms_:All}, opts_]] := Module[
	{res,im = getCachedMol[mol, "AllAtoms"],processedAtoms},
	check3DProperty["CenterOfMass"][ mol, im, $tag];
	processedAtoms = getAtomsList[ im, im["atomCount", True], atoms];
	res = im["getCenterOfMass", processedAtoms];
	quantity[res, "Angstroms"]
		 
]


addExposedProperty["PrincipalAxes", "GeometricProperties"]
molDescription["PrincipalAxes"] = "principal axes"
molLongDescription["PrincipalAxes"] = "principal axes, eigenvectors of the inertia tensor for the molecule"
molValue[mol_, mp[prop:"PrincipalAxes"|"PrincipalMoments"]] := Module[
	{im = getCachedMol[mol,"AllAtoms"], qm = QuantityMagnitude, coords, masses, com},
	check3DProperty["PrincipalAxes"][ mol, im, $tag];
	masses = qm @ mol["AtomicMass"];
	coords = qm @ mol["AtomCoordinates"];
	com = qm @ mol["CenterOfMass"];
	coords = TranslationTransform[-com][coords];
	Switch[prop,
		"PrincipalAxes",
			N @ Reverse @ Eigenvectors @ InertiaTensor[ SetPrecision[coords, $MachinePrecision], SetPrecision[masses, 6]],
		_,(*PrincipalMoments*)
			quantity[N @ Reverse @ Eigenvalues[InertiaTensor[ coords, masses]], "AtomicMassUnit" "Angstroms"^2]
	]
		 
]

addExposedProperty["PrincipalMoments", "GeometricProperties"]
molDescription["PrincipalMoments"] = "principal moments of inertia"




molDescription["PointGroupDisplay"] = "molecule point group display"
molDescription["PointGroupString"] = "molecule point group string"
addExposedProperty["PointGroupDisplay", "GeometricProperties"]
addExposedProperty["PointGroupString", "GeometricProperties"]
addPropertyAlias["PointGroupString", "PointGroup"]
$symmetryProps = Alternatives["PointGroupDisplay","PointGroup","PointGroupString","SymmetryEquivalentAtoms","SymmetryElements"];
molQualifiers[ $symmetryProps] := Options[SymmetryInformation]
molValue[ mol_, mp[prop:$symmetryProps, opts_]] := molValue[ mol, mp[{prop,All}, opts]];

molValue[ mol_, mp[{prop:$symmetryProps,atoms:(All|{__Integer})}, opts_] ]:= Module[
	{im = getCachedMol[mol, "AllAtoms"], natoms, elements, res},
	natoms = im["atomCount",False];
	check3DProperty["PointGroupString"][ mol, im, $tag];
	If[
		natoms === 1
		,
		Return[
			Switch[prop,
				"PointGroupDisplay",
					Style["K",Italic],
				"PointGroupString",
					"K",
				"SymmetryEquivalentAtoms",
					{{1}},
				_,
					{}
			]
		]
	];
	elements = Quiet[
		cachedEvaluation[SymmetryInformation, mol, atoms, opts],
		JLink`Java::excptn
	];
	If[
		!MatchQ[elements, KeyValuePattern[prop -> _]]
		,
		Return[Missing["NotAvailable"]]
	];
	elements[prop]
]


molDescription["SymmetryEquivalentAtoms"] = "topologically equivalent atoms"
addExposedProperty["SymmetryEquivalentAtoms", "GeometricProperties"]


molDescription["SymmetryElements"] = "symmetry elements"
molLongDescription["SymmetryElements"] = "list of the symmetry elements including rotation axes, planes of symmetry and inversion centers"
addExposedProperty["SymmetryElements", "GeometricProperties"]


(* ::Subsection::Closed:: *)
(*GraphProperties*)


copyChemicalDataDescription["VertexTypes"]
(*addExposedProperty["VertexTypes", "GraphProperties"]*)
addPropertyAlias["AtomicSymbol", "VertexTypes"]

copyChemicalDataDescription["EdgeRules"]
(*addExposedProperty["EdgeRules", "GraphProperties"]*)
setPerBondPropertyValue[ "EdgeRules",
	Function[{mol, imol, bonds},
		Rule @@@ bonds
	]
]


copyChemicalDataDescription["EdgeTypes"]
(*addExposedProperty["EdgeTypes", "GraphProperties"]*)
addPropertyAlias["BondType", "EdgeTypes"]


copyChemicalDataDescription["AdjacencyMatrix"]
molLongDefinition["AdjacencyMatrix"] = "sparse matrix representation of chemical bonding between atomic centers"
addExposedProperty["AdjacencyMatrix", "GraphProperties"]
molValue[ mol_, mp["AdjacencyMatrix", _] ] := Module[
	{imol, bondweighted = False, mat},
	imol = getCachedMol[ mol];
	mat = imol[ "getAdjacencyMatrix", bondweighted];
	
	If[
		MatrixQ[mat]
		,
		SparseArray @ Rationalize @ mat
		,
		Missing["NotAvailable"]
	]
]

molDescription["BondWeightedAdjacencyMatrix"] = "bond-weighted adjacency matrix"
molLongDescription["BondWeightedAdjacencyMatrix"] = "bond-weighted, sparse matrix representation of chemical bonding between atomic centers"
addExposedProperty["BondWeightedAdjacencyMatrix", "GraphProperties"]
molValue[ mol_, mp["BondWeightedAdjacencyMatrix", _] ] := Module[
	{imol, bondweighted = True, mat},
	imol = getCachedMol[ mol];
	mat = imol[ "getAdjacencyMatrix", bondweighted];
	
	If[
		MatrixQ[mat]
		,
		SparseArray @ mat
		,
		Missing["NotAvailable"]
	]
]

molDescription["SmallestSetOfSmallestRings"] = "smallest set of smallest rings"
molLongDescription["SmallestSetOfSmallestRings"] = "minimal cycle basis of the molecular graph, returned as a nested list of atom indices"
addExposedProperty["SmallestSetOfSmallestRings", "GraphProperties"]
addPropertyAlias["SmallestSetOfSmallestRings", "SSSR"]
molValue[ mol_, mp["SmallestSetOfSmallestRings", _] ] := Module[
	{imol,res},
	imol = getCachedMol[ mol];
	res /; MatchQ[ res = imol["atomRings"],{{__Integer}...}] 
]
molPropertyCitation["SmallestSetOfSmallestRings"] = "https://dx.doi.org/10.1021/ci960013p"



molDescription["GraphDistanceMatrix"] = "graph distance matrix"
molLongDescription["GraphDistanceMatrix"] = "matrix whose elements are the shortest path between atoms"
addExposedProperty["GraphDistanceMatrix", "GraphProperties"]
molValue[ mol_, mp["GraphDistanceMatrix"]] := Module[
	{graph = getMolGraph[mol]},
	GraphDistanceMatrix @ graph
]
	
(* ::Subsection::Closed:: *)
(*AtomProperties*)


$atomProps = Join[ $AtomIntegerProps, $AtomRealProps, $AtomCustomProps, $AtomBooleanProps];


(* ::Subsubsection::Closed:: *)
(*integer properties*)


molDescription["AtomicNumber"] = "atomic number"
addExposedProperty["AtomicNumber", "AtomProperties"]
addPropertyAlias["AtomicNumber", "AtomicNumbers"]
setPerAtomPropertyValue["AtomicNumber", "intAtomProperty"]


molDescription["OrbitalHybridization"] = "orbital hybridization"
molLongDescription["OrbitalHybridization"] = "computed orbital hybridization"
addExposedProperty["OrbitalHybridization", "AtomProperties"]
addPropertyAlias["OrbitalHybridization", "OrbitalHybridizations"]
setPerAtomPropertyValue["OrbitalHybridization", "intAtomProperty"]


molDescription["FormalCharge"] = "formal charge"
addExposedProperty["FormalCharge", "AtomProperties"]
addPropertyAlias["FormalCharge", "FormalCharges"]
setPerAtomPropertyValue["FormalCharge", "intAtomProperty"]

(*molDescription["FormalCharge"] = "formal charge"
addExposedProperty["FormalCharge", "AtomProperties"]
addPropertyAlias["FormalCharge", "FormalCharges"]*)
setPerAtomPropertyValue["OuterShellElectronCount", "intAtomProperty"]


molDescription["MassNumber"] = "mass number"
molLongDescription["MassNumber"] = "mass number; returns None if not specified"
addExposedProperty["MassNumber", "AtomProperties"]
addPropertyAlias["MassNumber", "MassNumbers"]
setPerAtomPropertyValue["MassNumber", "intAtomProperty"]


molDescription["HydrogenCount"] = "hydrogen count"
molLongDescription["HydrogenCount"] = "number of hydrogens bonded to the atom"
addExposedProperty["HydrogenCount", "AtomProperties"]
addPropertyAlias["HydrogenCount", "HydrogenCounts"]
setPerAtomPropertyValue["HydrogenCount", "intAtomProperty"]


molDescription["ImplicitHydrogenCount"] = "implicit hydrogen count"
molLongDescription["ImplicitHydrogenCount"] = "number of implicit hydrogens bonded to the atom"
addExposedProperty["ImplicitHydrogenCount", "AtomProperties"]
addPropertyAlias["ImplicitHydrogenCount", "ImplicitHydrogenCounts"]
setPerAtomPropertyValue["ImplicitHydrogenCount", "intAtomProperty"]


molDescription["ChiralTag"] = "chiral tag"
(*addExposedProperty["ChiralTag", "AtomProperties"]*)
addPropertyAlias["ChiralTag", "ChiralTags"]
setPerAtomPropertyValue["ChiralTag", "intAtomProperty"]


molDescription["ImplicitValence"] = "implicit valence"
molLongDescription["ImplicitValence"] = "atomic valence using only implicit atoms"
addExposedProperty["ImplicitValence", "AtomProperties"]
addPropertyAlias["ImplicitValence", "ImplicitValences"]
setPerAtomPropertyValue["ImplicitValence", "intAtomProperty"]


molDescription["ExplicitValence"] = "explicit valence"
molLongDescription["ExplicitValence"] = "atomic valence excluding any implicit hydrogens"
(*addExposedProperty["ExplicitValence", "AtomProperties"]*)
addPropertyAlias["ExplicitValence", "ExplicitValences"]
setPerAtomPropertyValue["ExplicitValence", "intAtomProperty"]


molDescription["UnpairedElectronCount"] = "unpaired electron count"
molLongDescription["UnpairedElectronCount"] = "number of unpaired electrons"
addExposedProperty["UnpairedElectronCount", "AtomProperties"]
addPropertyAlias["UnpairedElectronCount", "UnpairedElectronCounts"]
setPerAtomPropertyValue["UnpairedElectronCount", "intAtomProperty"]


molDescription["CIPRank"] = "CIP rank"
molLongDescription["CIPRank"] = "atom rank using the Cahn-Ingold-Prelog priority rules"
addExposedProperty["CIPRank", "AtomProperties"]
addPropertyAlias["CIPRank", "CIPRanks"]
setPerAtomPropertyValue["CIPRank", "intAtomProperty"]


molDescription["Valence"] = "valence"
molLongDescription["Valence"] = "number of valence electrons"
addExposedProperty["Valence", "AtomProperties"]
addPropertyAlias["Valence", "Valences"]
setPerAtomPropertyValue["Valence", "intAtomProperty"]


molDescription["AvailableElectrons"] = "available electrons"
molLongDescription["AvailableElectrons"] = "number of electrons available for aromaticity"
(*addExposedProperty["AvailableElectrons", "AtomProperties"]*)
setPerAtomPropertyValue["AvailableElectrons", "intAtomProperty"]


molDescription["CoordinationNumber"] = "coordination number"
molLongDescription["CoordinationNumber"] = "number of explicit bonds for an atom"
addExposedProperty["CoordinationNumber", "AtomProperties"]
addPropertyAlias["CoordinationNumber", "CoordinationNumbers"]
setPerAtomPropertyValue["CoordinationNumber", "intAtomProperty"]


molDescription["HeavyAtomCoordinationNumber"] = "heavy atom coordination number"
molLongDescription["HeavyAtomCoordinationNumber"] = "number of bonds to heavy atoms"
addExposedProperty["HeavyAtomCoordinationNumber", "AtomProperties"]
addPropertyAlias["HeavyAtomCoordinationNumber", "HeavyAtomCoordinationNumbers"]
setPerAtomPropertyValue["HeavyAtomCoordinationNumber", "intAtomProperty"]


molDescription["MostAbundantMassNumber"] = "commonest mass number"
molLongDescription["MostAbundantMassNumber"] = "mass number for the most commonly occurring isotope"
addExposedProperty["MostAbundantMassNumber", "AtomProperties"]
addPropertyAlias["MostAbundantMassNumber", "CommonestIsotopes"]
setPerAtomPropertyValue["MostAbundantMassNumber", "intAtomProperty"]


molDescription["PiElectronCount"] = "\[Pi] electron count"
molLongDescription["PiElectronCount"] = "number of \[Pi] electrons"
addExposedProperty["PiElectronCount", "AtomProperties"]
addPropertyAlias["PiElectronCount", "PiElectronCounts"]
setPerAtomPropertyValue["PiElectronCount", "intAtomProperty"]


molDescription["AtomChirality"] = "atom chirality"
molLongDescription["AtomChirality"] = "absolute atomic chirality; returns R, S or Undefined for chiral atoms and None otherwise"
addExposedProperty["AtomChirality", "AtomProperties"]
setPerAtomPropertyValue["AtomChirality", "intAtomProperty"]
(*setPerAtomPropertyValue[
	"AtomChirality",
	Function[
		{mol, imol, atomIndices},
		With[
			{possible = Alternatives @@ mol["PossibleStereocenters"]},
			Replace[
				atomIndices, 
				{
					n:possible :> Replace[
						imol[ "rdkitAtomProperty", n, "_CIPCode"],
						Except["R" | "S"] :> "Undefined"
					],
					_ :> None
				},
				1
			]
		]
		
	]
]*)

(* ::Subsubsection::Closed:: *)
(*Real properties*)


molDescription["GasteigerPartialCharge"] = "gasteiger partial charge"
molLongDescription["GasteigerPartialCharge"] = "atomic charge using the Gasteiger charge model"
addExposedProperty["GasteigerPartialCharge", "AtomProperties"]
addPropertyAlias["GasteigerPartialCharge", "GasteigerPartialCharges"]
setPerAtomPropertyValue["GasteigerPartialCharge", "realAtomProperty"]


molDescription["MMFFPartialCharge"] = "MMFF partial charge"
molLongDescription["MMFFPartialCharge"] = "atomic charge computed using the MMFF force field"
addExposedProperty["MMFFPartialCharge", "AtomProperties"]
addPropertyAlias["MMFFPartialCharge", "MMFFPartialCharges"]
setPerAtomPropertyValue["MMFFPartialCharge", "realAtomProperty"]


molDescription["TopologicalStericEffectIndex"] = "topological steric effect index"
addExposedProperty["TopologicalStericEffectIndex", "AtomProperties"]
addPropertyAlias["TopologicalStericEffectIndex", "TopologicalStericIndices"]
addPropertyAlias["TopologicalStericEffectIndex", "TSEI"]
setPerAtomPropertyValue["TopologicalStericEffectIndex", "realAtomProperty"]
molPropertyCitation["TopologicalStericEffectIndex"] = "https://dx.doi.org/10.1021/ci034266b"


molDescription["GeometricStericEffectIndex"] = "geometric steric effect index"
molLongDescription["GeometricStericEffectIndex"] = "geometric steric index; requires atomic coordinates"
addExposedProperty["GeometricStericEffectIndex", "AtomProperties"]
addPropertyAlias["GeometricStericEffectIndex", "GeometricStericIndices"]
setPerAtomPropertyValue["GeometricStericEffectIndex", "realAtomProperty"]
molPropertyCitation["GeometricStericEffectIndex"] = "https://dx.doi.org/10.1021/ci034266b"

molDescription["AtomicMass"] = "atomic mass"
addExposedProperty["AtomicMass", "AtomProperties"]
setPerAtomPropertyValue["AtomicMass", "realAtomProperty"]

(* ::Subsubsection::Closed:: *)
(*boolean properties*)


molDescription["AromaticAtomQ"] = "aromatic atom q"
molLongDescription["AromaticAtomQ"] = "gives True if an atom is aromatic"
addExposedProperty["AromaticAtomQ", "AtomProperties"]
setPerAtomPropertyValue["AromaticAtomQ", "boolAtomProperty"]


molDescription["RingMemberQ"] = "ring member q"
molLongDescription["RingMemberQ"] = "gives True if an atom is part of a ring"
addExposedProperty["RingMemberQ", "AtomProperties"]
setPerAtomPropertyValue["RingMemberQ", "boolAtomProperty"]


molDescription["UnsaturatedAtomQ"] = "unsaturated atom q"
molLongDescription["UnsaturatedAtomQ"] = "gives True if an atom is unsaturated"
addExposedProperty["UnsaturatedAtomQ", "AtomProperties"]
setPerAtomPropertyValue["UnsaturatedAtomQ", "boolAtomProperty"]



(* ::Subsubsection::Closed:: *)
(*custom atom properties*)


molDescription["Element"] = "element"
molLongDescription["Element"] = "the corresponding Element entity"
addExposedProperty["Element", "AtomProperties"]
addPropertyAlias["Element", "Elements"]
setPerAtomPropertyValue["Element",
	Function[
		{mol, imol, atoms},
		With[{atnums = imol["intAtomProperty","AtomicNumber", atoms]},
			ElementData /@ atnums
		]
	]
]

molDescription["AtomicSymbol"] = "atomic symbol"
molLongDescription["AtomicSymbol"] = "IUPAC atomic symbol"
addExposedProperty["AtomicSymbol", "AtomProperties"]
addPropertyAlias["AtomicSymbol", "Abbreviation"]
addPropertyAlias["AtomicSymbol", "AtomicSymbols"]
setPerAtomPropertyValue["AtomicSymbol",
	Function[
		{mol, imol, atoms},
		With[{atnums = imol["intAtomProperty","AtomicNumber", atoms]},
			FromAtomicNumber @ atnums
		]
	]
]

molDescription["Isotope"] = "isotope"
molLongDescription["Isotope"] = "the corresponding Isotope entity"
addExposedProperty["Isotope", "AtomProperties"]
addPropertyAlias["Isotope", "Isotopes"]
setPerAtomPropertyValue["Isotope",
	Function[
		{mol, imol, atoms},
		Module[
			{massNums,atnums, res},
			massNums = imol["intAtomProperty","MassNumber", atoms];
			atnums = imol["intAtomProperty","AtomicNumber", atoms];
			res = Replace[
				Thread[ {atoms, atnums, massNums} ],
				{id_, an_, 0} :> {id, an, First[ imol["intAtomProperty","MostAbundantMassNumber", {id}]]},
				1
			];
			IsotopeData /@ res[[All, {2,3}]]
		]
	]
] 	

molDescription["AtomIndex"] = "index"
molLongDescription["AtomIndex"] = "atom index"
addExposedProperty["AtomIndex", "AtomProperties"]
addPropertyAlias["AtomIndex", "AtomIndices"]
(* making this property marginally faster than other properties *)
molValue[mol_, mp["AtomIndex"] | mp[{"AtomIndex",All}] ] := Range @ getCachedMol[mol]["atomCount",True];
molValue[mol_, mp["AtomIndex"] | mp[{"AtomIndex",atoms_}] ] := With[
	{atms = getAtomsList[getCachedMol[mol], atoms]},
	If[
		IntegerQ[atms] && MatchQ[atms,{_}]
		,
		First @ atms
		,
		atms
	]
]


molValue[mol_, mp[{prop:$elementProperty, Bond[{a_,b_},___]}, o_]] := molValue[ mol, mp[ {prop, {a,b}},o]];
molValue[mol_, mp[{prop:$elementProperty, b:{__Bond}}, o_]] := molValue[ mol, mp[ {prop, #},o]] & /@ b;
molValue[ mol_, mp[prop:$elementProperty, o_] ] := molValue[ mol, mp[{prop, All},o]]
molValue[ mol_, mp[{prop:$elementProperty, atoms_:All}, _] ] := 
	Module[
		{atomicNumbers},
		atomicNumbers = molValue[mol,mp[{"AtomicNumber",atoms}]];
		atomicNumbers = Entity["Element",#] & /@ atomicNumbers;
		EntityValue[ atomicNumbers, prop] 
	]



(* ::Subsection::Closed:: *)
(*BondProperties*)

molDescription["BondType"] = "bond type"
molLongDescription["BondType"] = "bond type, e.g. Single, Double, etc."
addExposedProperty["BondType", "BondProperties"]
addPropertyAlias["BondType", "BondTypes"]
setPerBondPropertyValue["BondType"]

molDescription["BondIndex"] = "bond index"
molLongDescription["BondIndex"] = "position of a bond in the bond list for a molecule"
addExposedProperty["BondIndex", "BondProperties"]
addPropertyAlias["BondIndex","BondIndices"]
setPerBondPropertyValue["BondIndex"]

molDescription["BondLength"] = "bond length"
molLongDescription["BondLength"] = "Euclidean distance between the given atoms"
addExposedProperty["BondLength", "BondProperties"]
addPropertyAlias["BondLength","BondLengths"]
setPerBondPropertyValue["BondLength"]


molDescription["ConjugatedBondQ"] = "conjugated bond Q"
molLongDescription["ConjugatedBondQ"] = "gives True if the bond is part of a conjugated system"
addExposedProperty["ConjugatedBondQ", "BondProperties"]
setPerBondPropertyValue["ConjugatedBondQ"]


molDescription["BondOrder"] = "bond order"
molLongDescription["BondOrder"] = "number of bonding electron pairs"
addExposedProperty["BondOrder", "BondProperties"]
setPerBondPropertyValue["BondOrder"]


molDescription["BondAtomIndices"] = "bond atom indices"
setPerBondPropertyValue["BondAtomIndices", #3&]


molDescription["BondStereo"] = "bond stereo"
molLongDescription["BondStereo"] = "absolute bond stereo as determined using the Cahn-Ingold-Prelog priority rules"
addExposedProperty["BondStereo", "BondProperties"]
setPerBondPropertyValue["BondStereo"]

molDescription["BondedAtomIndices"] = "bonded atom indices"
molLongDescription["BondedAtomIndices"] = "indices of atoms forming bonds"
addExposedProperty["BondedAtomIndices", "BondProperties"]
setPerBondPropertyValue["BondedAtomIndices", #3&] 

(* ::Section::Closed:: *)
(*Remaining references*)
molPropertyCitation["GasteigerPartialCharge"] = "https://doi.org/10.1016/0040-4020(80)80168-2"
molPropertyCitation["MMFFPartialCharge"] = "https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P"
molPropertyCitation["MMFFEnergy"] = "https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P"
molPropertyCitation["MMFFsEnergy"] = "https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P"
molPropertyCitation["Autocorrelation2D"] = "https://doi.org/10.1002/9783527618279.ch37"
molPropertyCitation[
	"Chi0n"|"Chi0v"|"Chi1n"|"Chi1v"|"Chi2n"|"Chi2v"|"Chi3n"|"Chi3v"|"Chi4n"|"Chi4v"|"Kappa1"|"Kappa2"|"Kappa3" 
] = "https://doi.org/10.1002/9780470125793.ch9" 
molPropertyCitation["CrippenClogP"|"CrippenMR"] = "https://doi.org/10.1021/ci990307l"
molPropertyCitation["KierHallAlphaShape"] = "https://doi.org/10.1002/9780470125793.ch9"
molPropertyCitation["LabuteApproximateSurfaceArea"] = "https://doi.org/10.1016/S1093-3263(00)00068-1"
molPropertyCitation["LipinskiHBondAcceptorCount"] = "https://doi.org/10.1016/S0169-409X(00)00129-0"
molPropertyCitation["LipinskiHBondDonorCount"] = "https://doi.org/10.1016/S0169-409X(00)00129-0"
molPropertyCitation["PEOEVSA"] = "https://doi.org/10.1016/S1093-3263(00)00068-1"
molPropertyCitation["SlogPVSA"] = "https://doi.org/10.1016/S1093-3263(00)00068-1"
molPropertyCitation["SMRVSA"] = "https://doi.org/10.1016/S1093-3263(00)00068-1"
molPropertyCitation["TopologicalPolarSurfaceArea"] = "https://doi.org/10.1021/jm000942e"
molPropertyCitation[$symmetryProps] = "https://doi.org/10.1002/jcc.22995"

 


(* ::Section::Closed:: *)
(*Cleanup*)


$ExposedProperties = KeySort[ Sort /@ $ExposedProperties]

KeyValueMap[
	SetDelayed[
		molValue[ mol_, mp[ #1]],
		iMoleculeValue[mol, #2, "PropertyAssociation"]
	]&,
	$ExposedProperties
]




End[] (* End Private Context *)

