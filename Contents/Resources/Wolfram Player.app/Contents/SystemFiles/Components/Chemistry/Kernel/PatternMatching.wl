


Begin["Chemistry`Private`PatternMatchingDump`"]


(* ::Section::Closed:: *)
(*AtomCount*)


Options[AtomCount] = Options[AtomList] = Options[BondCount] ={IncludeHydrogens -> True};


AtomCount[mol_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{im = getCachedMol[mol,Flatten@{opts}]},
	im["atomCount",True]
]


AtomCount[mol_?MoleculeQ, patt_, opts:OptionsPattern[]] := Module[
	{list = AtomList[ mol, patt, opts]},
	Length[list] /; ListQ[list]
]

AtomCount[ mol_?MoleculeQ, Verbatim[Blank[]] | All, opts:OptionsPattern[]] := AtomCount[ mol, opts]

makeEntityDownValue[AtomCount]

AtomCount[ arg1_, ___] := (messageNoMol[AtomCount,arg1]; Null /; False)

(* ::Section::Closed:: *)
(*AtomList*)


AtomList[args___] := Module[
	{argCheck, res},
	argCheck = System`Private`ArgumentsWithRules[ AtomList[ args], {1,3}];
	(
		(* {} is OptionsPattern, but ArgumentsWithRules will categorize it as an
		argument and not an empty list of options *)
		argCheck = Replace[
			argCheck,
				{ {a_, {}}, b_} :> {{a},b}
		];
	 	res = Catch[ iAtomList @@ argCheck, $tag];
	 	res /; res =!= $Failed
	)/; MatchQ[ argCheck, {_, _}] 
]


iAtomList[{mol_?MoleculeQ}, opts_] := Module[
	{imol},
	imol = getCachedMol[mol,opts];
	
	If[
		Length[mol[[1]]] === imol["atomCount",True],
		mol[[1]],
		getiMolProperty[imol,"AtomList"]
	]

]


iAtomList[ {mol_?MoleculeQ, Verbatim[Blank[]]}, opts_] := iAtomList[ {mol}, opts]


iAtomList[{mol_?MoleculeQ, patt_}, opts_] := Module[
	{indices},
	(* TODO: replace this with a single optimized library call *)
	indices = Switch[ patt,
		Verbatim[Blank[]] | All,
			Range @ AtomCount[mol, opts],
		_?ValidQueryAtom,
			Flatten[ MoleculeSubstructureIndices[ mol, patt] ],
		x:{__Integer} /; AllTrue[ x, Between[{1,AtomCount[mol,opts]}]],
			patt,
		_,
			Message[ AtomList::atmpat, patt];
			Throw[$Failed, $tag]
	];
	iAtomList[{mol},{}][[ indices]]
]


$possibleProperty = _String | _MoleculeProperty | HoldPattern[_EntityProperty];

iAtomList[{mol_?MoleculeQ, patt_,prop_}, opts_] := Module[
	{indices},
	indices = Switch[ patt,
		Verbatim[Blank[]] | All,
			Range @ AtomCount[mol, opts],
		$ElementsPattern,
			Flatten[Position[mol[[1,All,1]], patt, 1, Heads -> False]],
		(
			Verbatim[Alternatives][$ElementsPattern..] | 
			Verbatim[Except][$ElementsPattern] | Verbatim[Except][Verbatim[Alternatives][$ElementsPattern..]]
		),
		
			Flatten[Position[mol[[1,All,1]], patt, 1, Heads -> False]],
		_?ValidQueryAtom,
			Flatten[ MoleculeSubstructureIndices[ mol, patt] ],
		x:{__Integer} /; AllTrue[ x, Between[{1,AtomCount[mol,opts]}]],
			patt,
		_,
			Message[ AtomList::atmpat, patt];
			Throw[$Failed, $tag]
	];
	If[
		Length @ indices === 0
		,
		Return[{}, Module]
	];
	Switch[ prop,
		"AtomIndices"|"AtomIndex",
			indices,
		$possibleProperty,
			Replace[
				MoleculeValue[ mol, {prop,indices}],
				_MoleculeValue :> (
					Message[AtomList::notprop, prop, Molecule];
					Throw[$Failed, $tag]
				)
			],
		{$possibleProperty..},
			Thread @ Replace[
				MoleculeValue[ mol, {#,indices}& /@ prop ],
				_MoleculeValue :> (
					Message[AtomList::notprop, prop, Molecule];
					Throw[$Failed, $tag]
				)
			],
		_,
			Message[AtomList::notprop, prop, Molecule];
			Throw[$Failed, $tag]
	]
];

makeEntityDownValue[{iAtomList}]

iAtomList[ {arg1_ /; !MoleculeQ[arg1], ___}, ___] := (messageNoMol[AtomList,arg1]; $Failed)
iAtomList[___] := $Failed


(* ::Section::Closed:: *)
(*BondCount*)



Options[BondList] = Options[BondCount] = {"Kekulized" -> False, IncludeHydrogens -> True}


BondCount[mol_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{im = getCachedMol[mol,Flatten@{opts}]},
	im["bondCount",True]
]


BondCount[mol_?MoleculeQ, patt_, opts:OptionsPattern[]] := Module[
	{list = BondList[ mol, patt, opts]},
	Length[list] /; ListQ[list]
]

makeEntityDownValue[BondCount]

BondCount[ arg1_, ___] := (messageNoMol[BondCount,arg1]; Null /; False)

(* ::Section::Closed:: *)
(*BondList*)



BondList[args___] := Module[
	{res, argCheck = System`Private`ArgumentsWithRules[ BondList[ args], {1,3}]},
	(
		res /; (res = Catch[ iBondList @@ argCheck, $tag]) =!= $Failed
	)/; MatchQ[ argCheck, {_, _}] 
]


iBondList[ {mol_, Verbatim[Blank[]]}, opts_] := iBondList[ {mol}, opts]


iBondList[ {mol_?MoleculeQ}, opts_] := Module[
	{imol, bonds, kekule},
	imol = getCachedMol[mol, Flatten @ {opts}];
	bonds = mol[[2]];
	kekule = TrueQ @ OptionValue[BondList, opts, "Kekulized"];
	kekule = kekule && MemberQ[bonds[[All,2]], "Aromatic"];
	
	If[
		kekule || Length[bonds] =!= imol["bondCount",True],
		getiMolProperty[imol, "BondList", kekule],
		bonds
	]
]


$VerbatimAtomInputs = _Integer | Verbatim[Blank[]] | Verbatim[Alternatives][__Integer] 
$VerbatimAtomInputs = $VerbatimAtomInputs | Verbatim[Except][$VerbatimAtomInputs]
$bondTypePattern = (_String | Verbatim[Blank[]] | Verbatim[Alternatives][$BondTypes..]);


(* BondList[ mol, "Single" ] *)
iBondList[ {mol_?MoleculeQ, pattern:$BondTypes}, opts_ ] :=  
	Cases[ iBondList[ {mol}, opts], Bond[ _, pattern]];
	
(* BondList[ mol, Bond["BondStereo" -> 4]] *)
iBondList[ {mol_?MoleculeQ, Bond[Rule[prop_, val_]]}, opts_] := With[
	{bondlist = iBondList[{mol},opts], values = mol[prop]},
	Cases[
		Thread[{bondlist,values}],
		{bond_, val} :> bond
	]
]


(* Take Bond[ _List] to mean an unspecified bond type *)
iBondList[ {a_, Bond[ b_List]}, opts_] := iBondList[ {a, Bond[ b, _]}, opts]


(*
	 BondList[ mol, Bond[ {a, b_}, "Single"]
*)
iBondList[ {mol_?MoleculeQ, Bond[{atm1:$VerbatimAtomInputs, atm2:$VerbatimAtomInputs},type:$bondTypePattern] }, opts_] := 
	Cases[ iBondList[ {mol}, opts], Bond[{OrderlessPatternSequence[atm1,atm2]},type]];
	
iBondList[ {mol_?MoleculeQ, (Rule|RuleDelayed|DirectedEdge|UndirectedEdge)[a_,b_]}, opts_] := iBondList[ {mol, Bond[{a,b}]},opts]
		
(*
	BondList[ mol, {1,2,5}] - returns all bonds between these atoms
*)
iBondList[ {mol_?MoleculeQ, atomsList:{__Integer}}, opts_] := With[
	{atom = Alternatives @@ atomsList},
	
	iBondList[ {mol, Bond[{atom,atom},_]}, opts]
]


(* BondList[ mol, Bond[ Atom[_], x_]] *)
iBondList[ {mol_?MoleculeQ, 
	Bond[{OrderlessPatternSequence[Verbatim[Atom[_]],atom1_]},b___]}, opts_ ] := 
		iBondList[ {mol, Bond[{_,atom1},b]}, opts];


iBondList[ {mol_?MoleculeQ, 
	Bond[{OrderlessPatternSequence[atm1_Integer, atm2:_?ValidQueryAtom]},type:$bondTypePattern]}, opts_ ] := 
	Module[
		{atomlist,bondlist},
		atomlist = AtomList[ mol, atm2 , "AtomIndex"];
		bondlist = iBondList[ {mol}, opts];
		Cases[ bondlist, Bond[{OrderlessPatternSequence[atm1,Alternatives @@ atomlist]},type]]
	];


	
iBondList[ {mol_?MoleculeQ, pattern_?ValidQueryBond} ,opts_] := Module[
	{indices = MoleculeSubstructureIndices[mol, pattern],
	bonds = iBondList[{mol},  opts]},
	Replace[indices,
		{
			{id1_, id2_} :> FirstCase[
				bonds,
				Bond[{OrderlessPatternSequence[id1, id2]},_],
				Nothing
			]
		},
		1
	]
];


iBondList[ {mol_?MoleculeQ, pattern_}, opts_] := Cases[
	iBondList[{mol},  opts],
	Replace[ pattern, All :> Blank[]]
]

iBondList[{mol_?MoleculeQ, patt_,prop_}, opts_] := Module[
	{bonds},
	bonds = Replace[
		iBondList[{mol, patt},opts],
		Except[{__Bond}] :> Return[{}, Module]
	];
	Switch[ prop,
		$possibleProperty,
			Replace[
				MoleculeValue[ mol, {prop,bonds}],
				_MoleculeValue :> (
					Message[BondList::notprop, prop, Molecule];
					Throw[$Failed, $tag]
				)
			],
		{$possibleProperty...},
			Thread @ Replace[
				MoleculeValue[ mol, {#,bonds}& /@ prop ],
				_MoleculeValue :> (
					Message[BondList::notprop, prop, Molecule];
					Throw[$Failed, $tag]
				)
			],
		_,
			Message[BondList::notprop, prop, Molecule];
			Throw[$Failed, $tag]
	]
];


makeEntityDownValue[{iBondList}]

iBondList[ {arg1_ /; !MoleculeQ[arg1], ___}, ___] := (messageNoMol[BondList,arg1]; Null /; False)
iBondList[___] := $Failed


(* ::Section::Closed:: *)
(*MoleculeEquivalentQ*)

$EquivalenceProperty = "SMILES";

MoleculeEquivalentQ[ mol1_?MoleculeQ, mol1_] := True

MoleculeEquivalentQ[ mol1_?MoleculeQ, mols__?MoleculeQ] := With[
	{vals = MoleculeValue[ {mol1,mols}, $EquivalenceProperty]},
	SameQ @@ vals /; ListQ[vals]
]

makeEntityDownValue[MoleculeEquivalentQ]

MoleculeEquivalentQ[args___] := False


(* ::Section::Closed:: *)
(*MoleculeRelationQ*)

MoleculeRelationQ[ test_, mol1_?MoleculeQ, mol2_?MoleculeQ] := Catch[ Module[
	{sametest},
	sametest = Switch[ test,
		"ConstitutionalIsomers",
			iConstitutionalIsomerQ,
		"GraphIsomorphic",
			iIsomorphicMoleculeQ,
		"Isotopologues",
			iIsotopologueQ,
		"Stereoisomers",
			iStereoisomerQ,
		"Enantiomers",
			iEnantiomerQ,
		"Diastereomers",
			iDiastereomerQ,
		"Epimers",
			iEpimerQ
	];
	
	sametest[mol1,mol2]
	
], $tag]

MoleculeRelationQ[f_][mol1_?MoleculeQ, mol2_?MoleculeQ] := MoleculeRelationQ[ f, mol1, mol2]


addCompletion[ "MoleculeRelationQ" -> {
	Sort @ {"ConstitutionalIsomers","GraphIsomorphic","Isotopologues","Stereoisomers","Enantiomers","Diastereomers","Epimers"},0,0}]


(* ::Subsection::Closed:: *)
(*iIsomorphicMoleculeQ*)



iIsomorphicMoleculeQ[mol1_?MoleculeQ, mol2_?MoleculeQ ] :=  Module[
	{graph1 = getMolGraph[mol1,"AllAtoms"], graph2 = getMolGraph[mol2,"AllAtoms"]},
	IsomorphicGraphQ[ graph1, graph2] /; (GraphQ[graph1] && GraphQ[graph2])
]

iIsomorphicMoleculeQ[___] := False


(* ::Subsection::Closed:: *)
(*iConstitutionalIsomerQ*)



iConstitutionalIsomerQ[mol1_?MoleculeQ, mol2_?MoleculeQ ] := Module[
	{atomLists = MoleculeValue[{mol1,mol2},"FullAtomList"]},
	And[
		SameQ @@ Sort /@ atomLists,
		!iIsomorphicMoleculeQ[mol1, mol2 ]
	]
]

iConstitutionalIsomerQ[___] := False


(* ::Subsection::Closed:: *)
(*iIsotopologueQ*)



iIsotopologueQ[mol1_?MoleculeQ, mol2_?MoleculeQ ] := Module[
	{mols,mapping},
	(* TODO: if this function is useful it could be done more elegantly *)
	mols = DeleteCases[{mol1, mol2},  HoldPattern["MassNumber" -> _], Infinity];
	mapping = FindMoleculeSubstructure @@ mols;
	And[
		MoleculeEquivalentQ @@ mols,
		UnsameQ[
			mol1["MassNumber"],
			(mol2["MassNumber"])[[Values @ First @ mapping]]
		]	
	] /; Length[mapping] > 0
] 

iIsotopologueQ[___] := False


(* ::Subsection::Closed:: *)
(*iStereoisomerQ*)


iStereoisomerQ[mol1_?MoleculeQ, mol2_?MoleculeQ ] := And[
	iIsomorphicMoleculeQ[mol1, mol2],
	!MoleculeEquivalentQ[mol1,mol2],
	MoleculeEquivalentQ[ 
		MoleculeModify[ mol1, "RemoveStereochemistry"],
		MoleculeModify[ mol2, "RemoveStereochemistry"]
	]
]

iStereoisomerQ[___] := False

(* ::Subsection::Closed:: *)
(*iEnantiomerQ*)

pairChiralityList[mol1_, mol2_] := Module[
	{st1,st2,mapping},
	{st1,st2} = MoleculeValue[{mol1, mol2}, "AtomChirality"];
	mapping = FindMoleculeSubstructure[mol2,mol1];
	If[ Length[mapping] === 0, Return[False, Module]];
	st2 = st2[[ Values @ First @ mapping]];
	DeleteCases[ Thread[{st1,st2}], {None,None} | {"Unspecified","Unspecified"}]
]

(* ::Subsection::Closed:: *)
(*iEnantiomerQ*)


iEnantiomerQ[ mol1_?MoleculeQ, mol2_?MoleculeQ] /; iIsomorphicMoleculeQ[mol1, mol2] := Module[
	{vals},
	vals = pairChiralityList[mol1,mol2];
	If[ Length[vals] === 0, Return[False, Module]];
	AllTrue[
		vals,
		MatchQ[{None,None}|{"R","S"}|{"S","R"}]
	]
]


iEnantiomerQ[___] := False



(* ::Subsection::Closed:: *)
(*iDiastereomerQ*)


iDiastereomerQ[mol1_?MoleculeQ, mol2_?MoleculeQ] /; iIsomorphicMoleculeQ[mol1, mol2] := Module[
	{vals},
	vals = pairChiralityList[mol1,mol2];
	If[ Length[vals] === 0, Return[False, Module]];
	TrueQ @ And[
		Count[ vals,{"R","R"}|{"S","S"}] > 0, 
		Count[ vals, {"R","S"}|{"S","R"}] > 0
	]
]


iDiastereomerQ[___]:=False


(* ::Subsection::Closed:: *)
(*iEpimerQ*)


iEpimerQ[mol1_?MoleculeQ, mol2_?MoleculeQ] /; iIsomorphicMoleculeQ[mol1, mol2] := Module[
	{vals},
	vals = pairChiralityList[mol1,mol2];
	If[ Length[vals] < 2, Return[False, Module]];
	Count[ vals, {"R","S"}|{"S","R"}] === 1
]


iEpimerQ[___]:=False


(* ::Section::Closed:: *)
(*MoleculeContainsQ*)


MoleculeContainsQ[ mol1_?MoleculeQ, 
	Verbatim[Alternatives][patts__?(MessageValidQuery[MoleculeContainsQ])]] :=  
	AnyTrue[
		{patts},
		MoleculeContainsQ[mol1, #]&
	]

MoleculeContainsQ[ mol1_?MoleculeQ, mol2_?(MessageValidQuery[MoleculeContainsQ])  ] :=  Module[
	{im1,im2id, useChirality = False},
	im1 = getCachedMol @ mol1;
	im2id = ManagedLibraryExpressionID[ getCachedMol[ mol2] ];
	im1["moleculeContainsQ", im2id, useChirality]

]

makeEntityDownValue[MoleculeContainsQ]

MoleculeContainsQ[patt_][mol_?MoleculeQ] := MoleculeContainsQ[ mol, patt]
MoleculeContainsQ[patt_][mol_ /; !MoleculeQ[mol]] := (messageNoMol[MoleculeContainsQ,mol]; Null /; False)

MoleculeContainsQ[ arg1_ /; !MoleculeQ[arg1], __] := (messageNoMol[MoleculeContainsQ,arg1]; Null /; False)

(* ::Section::Closed:: *)
(*FindMoleculeSubstructure*)



Options[FindMoleculeSubstructure] = {Overlaps -> False}


FindMoleculeSubstructure[mol1_?MoleculeQ, mol2_?(MessageValidQuery[FindMoleculeSubstructure]) , n:(_Integer|All):1, opts:OptionsPattern[]  ] := 
	Module[ {im1, match, im2id,useChirality,unique},
		useChirality = False;
		unique = !TrueQ[OptionValue[Overlaps]];
		im1 = getCachedMol @ mol1;
		im2id = ManagedLibraryExpressionID[ getCachedMol[ mol2(*, "NoHydrogens"*)] ];
		(match = im1[ "findSubstructureMatches", im2id , useChirality, unique, n /. All -> 10^6];
		Map[Association[(Rule @@@ #)]&] @ match) 
	];
	
	
FindMoleculeSubstructure[ arg1_ /; !MoleculeQ[arg1], ___] := (messageNoMol[FindMoleculeSubstructure,arg1]; Null /; False)


(* ::Section::Closed:: *)
(*MoleculeSubstructureCount*)


MoleculeSubstructureCount[mol_?MoleculeQ, {patts__?(MessageValidQuery[MoleculeSubstructureCount])}] := Module[
	{im1 = getCachedMol[mol], queryIDs, unique = !TrueQ[OptionValue[Overlaps]], useChirality = False},
	queryIDs = ManagedLibraryExpressionID@*getCachedMol /@ {patts};
	im1["moleculeSubstructureCounts", queryIDs, useChirality, unique]
]

MoleculeSubstructureCount[mol_?MoleculeQ, patt_?(MessageValidQuery[MoleculeSubstructureCount])] :=  Module[
	{res},
	First[res] /; MatchQ[ res = MoleculeSubstructureCount[mol, {patt}], {_Integer}]
]


(* ::Section::Closed:: *)
(*MoleculeSubstructureIndices*)

(* An internal function*)

Options[MoleculeSubstructureIndices] = {IncludeHydrogens -> True}

	
MoleculeSubstructureIndices[mol1_?MoleculeQ, mol2_?ValidQuery, opts:OptionsPattern[] ] := Values /@ FindMoleculeSubstructure[ mol1, mol2, 1000]


(* ::Section::Closed:: *)
(*End package*)



End[] (* End Private Context *)


