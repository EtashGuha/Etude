(* ::Package:: *)

Begin["Chemistry`Private`MoleculeModifyDump`"]

(* ::Section::Closed:: *)
(*Patterns and definitions*)


$ModifierFunctions[] = <|
	"AddAtom" -> iAddAtom,
	"DeleteAtom" -> iDeleteAtom,
	"ReplaceAtom" -> iReplaceAtom,
	"AddBond" -> iAddBond,
	"Canonicalize" -> iCanonicalize,
	"DeleteBond" -> iDeleteBond,
	"ExtractParts" -> iExtractParts,
	"RemoveHydrogens" -> iRemoveHydrogens,
	"AddHydrogens" -> iAddHydrogens,
	"Kekulize" -> iKekulize,
	"SetAromaticity" -> iSetAromaticity,
	"RenumberAtoms" -> iRenumberAtoms,
	"RemoveStereochemistry" -> iRemoveStereochemistry,
	"AssignStereochemistryFromCoordinates" -> iAssignStereochemistryFromCoordinates,
	"CanonicalizeAtomCoordinates" -> iCanonicalizeAtomCoordinates,
	"ComputeAtomCoordinates" -> iComputeAtomCoordinates,
	"GenerateConformers" -> iGenerateConformers,
	"EnergyMinimizeAtomCoordinates" -> iEnergyMinimizeAtomCoordinates,
	"ReplaceSubstructure" -> iReplaceSubstructure,
	"ReplaceSubstructureList" -> iReplaceSubstructureList,
	"SetAtomChirality" -> iSetAtomChirality,
	"SetBondStereo" -> iSetBondStereo,
	"SetFormalCharge" -> iSetAtomProperty["FormalCharge"],
	"SetMassNumber" -> iSetAtomProperty["MassNumber"],
	"SetUnpairedElectronCount" -> iSetAtomProperty["UnpairedElectronCount"],
	"SetBondLength" -> iSetBondLength,
	"SetBondAngle" -> iSetBondAngle,
	"SetTorsionAngle" -> iSetTorsionAngle,
	"TransformAtomCoordinates" -> iTransformAtomCoordinates,
	"SetMetaInformation" -> iSetMetaInformation
|>



$ValidModifications = Alternatives @@ Keys[ $ModifierFunctions[]]
$ModificationPattern = $ValidModifications | {$ValidModifications,___}

$UndocumentedModifications = {"ExtractParts", "Canonicalize"}

addCompletion[ "MoleculeModify" -> {0,Complement[Keys[ $ModifierFunctions[]], $UndocumentedModifications] }];

MoleculeModify[ mod:$ModificationPattern][mol_ /; MoleculeQ[mol] ] := MoleculeModify[ mol, mod]




(* ::Section::Closed:: *)
(*MoleculeModify*)


Options[ MoleculeModify ] = {ValenceErrorHandling -> Automatic, Method -> Automatic}

modifyMoleculeMethodOptions[] := Catenate[
	Options /@ Values[$ModifierFunctions[]]
];
	

MoleculeModify[ mod:$ValidModifications, args___][mol_?MoleculeQ ] := MoleculeModify[ mol, mod, args]

MoleculeModify[args___] :=
    Module[
    		{argCheck = System`Private`Arguments[ MoleculeModify[args], {1,2}, List, modifyMoleculeMethodOptions[] ], res},
        (
        		res = Catch[ iMoleculeModify @@ argCheck, $tag];
        		res /; res =!= $Failed
        ) /; argCheck =!= {}
    ]


(* ::Subsection::Closed:: *)
(*iMoleculeModify*)


(* :iMoleculeModify: *)
iMoleculeModify[{mol_Molecule, mod:$ValidModifications}, opts_] := 
	iMoleculeModify[ {mol, {mod}}, opts];

iMoleculeModify[{mol_?MoleculeQ, arguments:{what:$ValidModifications, rest___}}, opts_] := Block[
	{res, modFunc, options, explicitH},
	
	If[
		!StringQ[ what ],
		Message[ MoleculeModify::strx, what];
		Throw[$Failed,$tag]
	];
	
	modFunc = Lookup[
		$ModifierFunctions[],
		what
	];
	
	If[
		MissingQ[modFunc],
		Message[ MoleculeModify::invld, what];
		Throw[$Failed,$tag]
	];
	
	options = opts /. HoldPattern[ Method -> rul:{__Rule}] :> Sequence @@ rul;
	
	explicitH = And[
		handleValencesOption[opts],
		StringFreeQ[what,"Hydrogen"],
		!mol["HasImplicitHydrogens"],
		FreeQ[mol, IncludeHydrogens]
	];
	
	Block[{fixHydrogens = True},
		res = modFunc[mol, rest, options];
		If[
			fixHydrogens
			,
			res = fixH[explicitH] @ res;
		]
	];
	
    res /; AllTrue[ Flatten[{res}], MoleculeQ]

]

makeEntityDownValue[{iMoleculeModify}]

iMoleculeModify[ {m_ /; !MoleculeQ[m] && !MatchQ[m,$ValidModifications], args__}, _] := 
	(messageNoMol[MoleculeModify,m]; $Failed)

iMoleculeModify[___] := $Failed


(* ::Section::Closed:: *)
(*AddAtom*)


iAddAtom[ mol_, atom:$AtomInputPattern, opts:OptionsPattern[]] := iAddAtom[ mol, {atom}, opts];


iAddAtom[mol_, inputAtoms:{$AtomInputPattern...}, opts:OptionsPattern[]] := Block[
	{atoms = First @ mol, atomsToAdd, bonds, options},
	atomsToAdd = canonicalizeAtoms[inputAtoms];
	messageOnBadAtomInput[ inputAtoms, atomsToAdd, $tag];
	atoms = Join[ atoms, atomsToAdd];
	bonds = mol[[2]];
	options = includeHydrogens[opts] @ removeCoordinates @ Options[mol];
	Molecule[atoms, bonds, Sequence @@ options]
]

iAddAtom[expr___] :=
    (messageAddAtom[expr]; Null /; False)
    
messageAddAtom[ mol_, atm_,___] := Message[ Molecule::atom, atm];



(* ::Section::Closed:: *)
(*DeleteAtom*)


(* 	:DeleteAtom: *)

Options[iDeleteAtom] = {
	"RemoveHydrogens" -> Automatic, 
	System`ValenceErrorHandling -> Automatic,
	"FixStereochemistry" -> True 
};
	

iDeleteAtom[ mol_, atom:(_Integer|_String), opts:OptionsPattern[] ] := iDeleteAtom[ mol, {atom}, opts];

(*
	* remove any stereoBonds where the deleted atom is
	either one of the bonded atoms or one of the ligands.
	* remove any chiral elements where the deleted atom is
	one of the four ligands.
	* when we remove an atom, how do we split up 
	the bonding electrons?
	* what do we do with hydrogen atoms that were 
	attached to the removed atom?
*)


iDeleteAtom[ mol_?MoleculeQ, inputAtoms:{(_Integer|_String)...}, opts:OptionsPattern[]] := Block[
	{libID, copy = newInstance[], coords2D,coords3D,
		atomsToDelete, bonds, removeHs, fillValences, molData, notDeleted, params,
		stereo,atoms, options},
		
	atomsToDelete = atomNameMap[mol][inputAtoms];	
	
	options = removeCoordinatesAndStereo @ Options @ mol;
	
	messageOnBadAtomReference[ mol["AtomIndex"], inputAtoms, atomsToDelete, $tag];
	
	If[
		MemberQ[mol[[1,atomsToDelete,1]], "H"]
		,
		fixHydrogens = False;
		options = FilterRules[ options, Except[ IncludeHydrogens]]
	];
	
	libID = Which[
		TrueQ @ AllTrue[ atomsToDelete,( 0 < # <= Length @ First @ mol )&],
		mol["LibraryID"],
		TrueQ @ AllTrue[  atomsToDelete, (0 < # <= AtomCount[mol]) &],
		mol["LibraryID", IncludeHydrogens -> True],
		True,
		Throw[$Failed, $tag]
	];
	
		
	params = $chemLibraryFunctions["deleteAtoms", "Parameters"];
	
	params["FillValences"] = handleValencesOption[opts];
	params["RemoveOrphanedHs"] = params["FillValences"];

	copy["createCopy", libID];
	copy["deleteAtoms", atomsToDelete, params];
	
	
	{atoms,bonds,stereo} = {getiMolProperty[ copy, "AtomList"], getiMolProperty[ copy, "BondList"],
		getiMolProperty[ copy, "StereochemistryElements"]};
	
	
	
	(* for now we will keep the coordinates only if we aren't filling valences *)
	If[
		!params["FillValences"],
		notDeleted = Complement[ Range @ Length @ mol[[1]], atomsToDelete];
		coords3D = Replace[
			coordsFromExpression[ mol] ,
			{
				x_?MatrixQ :> Rule[AtomCoordinates, x[[ notDeleted]] ],
				_ :> Nothing
			}
		];
		AppendTo[options, coords3D];
	];
	If[
		TrueQ @ optionValue[iDeleteAtom, {opts}, "FixStereochemistry"],
		stereo = quietlyRemoveBadStereocenters[atoms, bonds, stereo];
	];
	If[
		Length @ stereo > 0,
		AppendTo[options, StereochemistryElements -> stereo]
	];

	Molecule[ atoms, bonds, Sequence @@ options]
]

iExtractParts[ mol_?MoleculeQ, atomToKeep_Integer, opts:OptionsPattern[] ] := 
	iExtractParts[mol, {atomToKeep}, opts]

iExtractParts[ mol_?MoleculeQ, atomsToKeep:{{___Integer}..}, opts:OptionsPattern[] ] := Map[
	iExtractParts[mol, #, opts]&, atomsToKeep]
	
iExtractParts[ mol_?MoleculeQ, atomsToKeep:{___Integer}, opts:OptionsPattern[] ] := Module[
	{atoms,atomsToDelete},
	atoms = Range @ Length @ First @ mol;
	atomsToDelete = Complement[ atoms, atomsToKeep];
	iDeleteAtom[mol, atomsToDelete, opts]
]


(* ::Section::Closed:: *)
(*ReplaceAtom*)


Options[ iReplaceAtom ] = {}

iReplaceAtom[ mol_?MoleculeQ, rule:Rule[ $AtomReferencePattern, $AtomInputPattern], opts:OptionsPattern[] ] := iReplaceAtom[ mol, {rule}, opts]

iReplaceAtom[ mol_?MoleculeQ, 
	rules:{Rule[ $AtomReferencePattern, $AtomInputPattern]...}, opts:OptionsPattern[] ] := Module[
	{atomsToDelete, atomsToAdd, copy, inputAtoms, molData, coords, res, params},
	
	inputAtoms = Keys @ rules;
	atomsToDelete = atomNameMap[mol][inputAtoms];	
	Check[Catch[
		messageOnBadAtomReference[ mol["AtomIndex"], inputAtoms, atomsToDelete, "badatom"],
		"badatom"], Return[$Failed, Module];
	];
	
	inputAtoms = Values @ rules;
	atomsToAdd = canonicalizeAtoms[inputAtoms];	
	
	messageOnBadAtomInput[ inputAtoms, atomsToAdd, $tag];
	
	If[
		!handleValencesOption[opts]
		,
		Return[
			ReplacePart[
				mol,
				Thread[ Thread[{1,atomsToDelete}] -> atomsToAdd]
			],
			Module
		]
	]; 
		
	atomsToAdd = Replace[
		atomsToAdd,
		a_Atom :> With[{atomOpts=Options[a]},
			{AtomicNumber[a[[1]]], Lookup[ atomOpts, "FormalCharge", 0], Lookup[ atomOpts, "MassNumber", 0], Lookup[ atomOpts, "UnpairedElectronCount", 0]}
		],
		1
	];
	
	copy = newInstance[];
	copy["createCopy", ManagedLibraryExpressionID @ getCachedMol @ mol];
	params = If[
		MemberQ[atomsToAdd[[All,1]], 1]
		,
		<|"AdjustHydrogens" -> False|>
		,
		Null
	];
	copy["replaceAtoms", atomsToDelete, atomsToAdd, params];
	
	coords = getMolOption[mol,AtomCoordinates];
	
	molData = molDataFixStereo[copy, 
		removeCoordinatesAndStereo @ Options @ mol,
		Lookup[Flatten @ {opts},"FixStereochemistry",True]
	];
	
	If[
		copy["atomCount",True] === Length[coords]
		,
		AppendTo[molData, AtomCoordinates -> coords]
	];
	Molecule @@ molData
]

molDataFixStereo[imol_iMolecule, options_List, fixStereo_] := Module[
	{atoms, bonds, stereo, molOpts = options},
	{atoms,bonds,stereo} = {
		getiMolProperty[ imol, "AtomList"], 
		getiMolProperty[ imol, "BondList"],
		getiMolProperty[ imol, "StereochemistryElements"]};
	If[
		fixStereo,
		stereo = quietlyRemoveBadStereocenters[atoms, bonds, stereo];
	];
	
	If[
		Length @ stereo > 0,
		AppendTo[molOpts, StereochemistryElements -> stereo]
	];
	{atoms, bonds, Sequence @@ molOpts}
]


(* ::Section::Closed:: *)
(*AddBond*)


Options[ iAddBond ] = {"DeleteAtoms"-> Automatic}

iAddBond[mol_ /; MoleculeQ[mol] , inputBonds_, opts:OptionsPattern[]] :=Module[
    {args, bondsToAdd, atomsToDelete, 
	imol, copy, startingMol, molData, deleteAtoms},
	
	messageOnDuplicateBonds[ mol, inputBonds];
	
	args = Replace[
		processAddBonds[ mol, inputBonds /. Bond[a:{_,_}]:>Bond[a,"Single"] ],
		Except[{_,_}] :> Throw[ $Failed, $tag]
	];

	{startingMol, bondsToAdd} = args;
	parseBonds[ bondsToAdd, molData];
	bondsToAdd = molData["Bonds"];
	deleteAtoms = handleValencesOption[opts];
	
	atomsToDelete = processAtomsToDelete[ startingMol, bondsToAdd, deleteAtoms];
	(
		imol = getCachedMol @ startingMol;
		copy = newInstance[];
		
		copy["createCopy", ManagedLibraryExpressionID @ imol];
		
		copy["addBondsDeleteAtoms", bondsToAdd, atomsToDelete];
		
		molData = molDataFixStereo[copy, 
			FilterRules[Options[mol], Except[StereochemistryElements]], 
			Lookup[Flatten @ {opts},"FixStereochemistry",True]
		] ;
		
		Molecule @@ molData
		
	) /; (atomsToDelete =!= $Failed) 
    
]



(* ::Subsection::Closed:: *)
(*processAddBonds*)


(* 
	parses inputs, replacing string-valued atom identifiers with their vertex number,
	if atoms are added implicitly in the bonds list then this will return a new molecule
*)

processAddBonds[ mol_, bonds:{$BondPattern...}] := With[
	{atoms = mol["AtomIndex"], edges = bonds[[All,1]]},
	If[
		ContainsAll[ atoms, Flatten @ edges] && FreeQ[ edges, {x_, x_}],
		{mol, bonds},
		Message[MoleculeModify::atmlst, bonds]; Throw[$Failed, $tag]
	]	
]

processAddBonds[ mol_, bond:$BondInputPattern] := processAddBonds[ mol, {bond}];

processAddBonds[ mol_, bonds:{$BondInputPattern..}] := Module[
	{molecule, bondList, atomCount, newAtoms},
	
	(* take care of any bonds given as simple index lists - AddBond[ mol, {1,2}] *)
	bondList = Replace[bonds, 
		{
			bond_ /; BondQ[mol, bond] :> (
				Message[Molecule::dupbnd];
				Throw[$Failed, $tag]
			),
			{x_, y_} :> Bond[ {x, y}, "Single"]
		}, 
		1 
	];
	
	(* handle named atoms *)
	bondList = atomNameMap[mol] @ bondList;
	
	(* now check for any new atoms added in the list *)
	atomCount = Length @ mol[[1]];
	newAtoms = Last @ Reap[
		bondList[[All, 1]] = ReplaceAll[
			bondList[[All, 1]],
			x:$AtomInputPattern :> (Sow[ x]; ++atomCount)
		];
	];

	(* if we have any new atoms, create a new Molecule exrpession to modify *)
	molecule = If[
		newAtoms != {},
		With[{new = iAddAtom[ mol, First @ newAtoms]},
			If[
				MoleculeQ[new],
				new,
				Throw[$Failed, $tag]
			]
		],
		mol
	];
	If[
		MatchQ[ bondList, {$BondPattern..}],
		processAddBonds[ molecule, bondList],
		$Failed
	]
]

processAddBonds[___] := $Failed;

messageOnDuplicateBonds[mol_, bonds_List] := Replace[
	List @@@ bonds,
	bond_ /; BondQ[mol, bond] :> (
		Message[Molecule::dupbnd];
		Throw[$Failed, $tag]
	), 
	{0,1}
]

messageOnDuplicateBonds[mol_, bonds_] := messageOnDuplicateBonds[mol, {bonds}]



(* ::Subsection::Closed:: *)
(*processAtomsToDelete*)


(* 	:processAtomsToDelete: *)

(*
	find any hydrogens we can delete to fix valences
*)

processAtomsToDelete[ mol_, bonds_, atoms:{__Integer} ] := Which[
	
	(* the atoms to delete are all valid atoms *)
	!ContainsAll[ mol["AtomIndex"], atoms],
	Message[ MoleculeModify::atmlst, atoms];
	$Failed,
	
	(* but aren't any of the atoms we are creating a new bond to *)
	ContainsNone[
		Flatten[ List @@@ bonds[[All, ;;2]] ],
		atoms
	],
	Message[ MoleculeModify::atmlst, atoms];
	$Failed,
	
	True,
	atoms
	
]

processAtomsToDelete[ mol_, bonds_, atoms_List] := Module[
	{atomList = atomNameMap[mol] @ atoms},
	atomList /; VectorQ[atomList, IntegerQ]
]

processAtomsToDelete[ mol_, bonds_, True] := Block[
	{atoms, graph, valences, hydrogenVerts, toDelete},
	atoms = First @ mol;
	graph = getMolGraph @ mol;
	
	

	(* Adding a single, double, or triple bond needs 1,2, or 3 electrons, adding an 
		aromatic bond means it's up to the user. *)
	valences = ConstantArray[0, VertexCount @ graph];
	
	Do[
		With[{atom1 = First @ bond, atom2 = bond[[2]],order = Last @ bond},
			valences[[ atom1 ]] += order;
			valences[[ atom2 ]] += order
		],
		{bond, bonds}
	];
	
	hydrogenVerts = connectedHs[graph,atoms,#] &/@ VertexList[graph];
	
	toDelete = Flatten @ MapThread[
		#1[[ ;; UpTo[#2] ]]&,
		{hydrogenVerts,valences}
	];
	
	toDelete /; VectorQ[ toDelete, IntegerQ]
]

processAtomsToDelete[ mol_, bonds_, _] := {}



(* ::Section::Closed:: *)
(*DeleteBond*)


(*	:DeleteBond:		*)

$BondPattern2 = Bond[ {$AtomReferencePattern, $AtomReferencePattern}, _String] | {$AtomReferencePattern, $AtomReferencePattern}

iDeleteBond[mol_ /; MoleculeQ[mol], bnd:$BondPattern2, opts:OptionsPattern[]] := 
	iDeleteBond[mol, {bnd}, opts]

iDeleteBond[mol_ /; MoleculeQ[mol], bnds:{$BondPattern2...},opts:OptionsPattern[]] := Block[
	{caught, graph = getMolGraph @ mol, im1 = getCachedMol[mol], 
		copy = newInstance[],molData, res, bonds, fixValences},
		
	bonds = atomNameMap[mol][bnds];
		
	bonds = Replace[
		bnds,
		{
			x_ /; !BondQ[mol, x] :> (
				Message[MoleculeModify::bond, x];
				Throw[ $Failed, $tag]
			),
			Bond[ {a_,b_},_] :> {a,b}
		},
		1
	];
	
	fixValences = handleValencesOption[opts];
	
	
	copy["createCopy", ManagedLibraryExpressionID @ im1];
	copy["deleteBonds", bonds];
	If[
		!fixValences,
		copy["convertHydrogensToHairs"]
	];
	
	molData = molDataFixStereo[copy, 
		FilterRules[Options[mol], Except[StereochemistryElements]],
		Lookup[Flatten @ {opts},"FixStereochemistry",True]
	] ;
	
	Molecule @@ molData

	
]



(* ::Section::Closed:: *)
(*ReplaceSubstructure*)


Options[iReplaceSubstructure] = {}

Options[iReplaceSubstructureList] =	{"DeleteDuplicates" -> True}


iReplaceSubstructure[args___] := oReplaceSubstructure["All", args ]

iReplaceSubstructureList[args___] := oReplaceSubstructure["List", args ]

oReplaceSubstructure[type:("List"|"All"), mol_?MoleculeQ, rep:HoldPattern[patt_ -> replacement_], 
	attchmts:(_Association | {Rule[_Integer, _Integer]..}|Automatic):Automatic, opts:OptionsPattern[] ] := Module[
	{
		util = newInstance[], queryiMol, replacementiMol, nmatches, explicitHydrogens,
		attachments, explicitOnly, replaceAll, res, part, pattern, nMatch, nReplace, 
		imol, molData, fixValences, deleteDuplicates
	},
	
	explicitHydrogens = MemberQ[ mol[[1]], Atom["H"]];

	Switch[type,
		"All",
			replaceAll = True;
			deleteDuplicates = False,
		"List",
			replaceAll = False;
			deleteDuplicates = TrueQ @ OptionValue[ iReplaceSubstructureList, {opts}, "DeleteDuplicates"];
	];

	pattern = True;
	queryiMol = Switch[ patt,
		MoleculePattern[_String],
			loadSMARTSString[First @ patt],
		
		_?MoleculeQ,
			getCachedMol[patt, "NoHydrogens"],
		
		_?ValidQuery,
			getCachedMol[patt, "NoHydrogens"],
		
		_String,
			loadSMILESString[ patt],
		
		{_String | _Integer..},
			pattern = False;
			{patt},
		
		{{_String | _Integer..}..},
			pattern = False;
			patt,
		
		_String | _Integer,
			pattern = False;
			{{patt}},
		
		_, 
			Message[MoleculeModify::atmlst, patt];
			Throw[$Failed,$tag];
	];
	
	replacementiMol = Switch[ replacement,
		_?MoleculeQ,
		getCachedMol[replacement, "NoHydrogens"],
		_String,
		loadSMILESString[ replacement], 
		_,
		Message[MoleculeModify::subst, rep];
		Throw[$Failed, $tag]
	];
	
	
	If[
		pattern,
		nMatch = queryiMol["atomCount", explicitOnly = True],
		nMatch = Length @ First @ queryiMol
	];		
	
	attachments = Replace[
		attchmts,
		{
			Automatic :> Which[
				nMatch === (nReplace = replacementiMol["atomCount", explicitOnly = True]),
					Thread[{Range[nMatch],Range[nMatch]}],
				nMatch > 1 && nReplace > 1,
					{{1,1},{nMatch,nReplace}},
				True,
					{{1,1}}
			],
			x_?AssociationQ :> List @@@ Normal[x],
			x:{HoldPattern[Rule[_Integer,_Integer]]..} :> List @@@ x,
			_ :> (
				Message[MoleculeModify::atchpt, "AttachmentPoints", attchmts];
				Throw[$Failed, $tag]
			)
		}
	];
	
	
	messageOnBadAtomReference[
		Range @ nMatch,
		attachments[[All, 1]],
		attachments[[All, 1]],
		$tag
	];
	
	messageOnBadAtomReference[
		Range @ replacementiMol["atomCount", explicitOnly = True],
		attachments[[All, 2]],
		attachments[[All, 2]],
		$tag
	];
	fixValences = handleValencesOption[opts];
	
	If[ 
		pattern,
		nmatches = util["replaceSubstructs",
			ManagedLibraryExpressionID @ getCachedMol[ mol],
			ManagedLibraryExpressionID @ queryiMol,
			ManagedLibraryExpressionID @ replacementiMol,
			attachments,
			replaceAll,
			fixValences,
			deleteDuplicates
		];
		,
		MapThread[
			messageOnBadAtomReference[ mol["AtomIndex"], ##, $tag]&,
			{ queryiMol, queryiMol = atomNameMap[mol][queryiMol] } 
		];
		nmatches = util[ "replacePart",
			ManagedLibraryExpressionID @ getCachedMol[ mol(*, "AllAtoms"*)],
			queryiMol,
			ManagedLibraryExpressionID @ replacementiMol,
			attachments,
			replaceAll,
			fixValences
		];
	];
	(
		
		res = Table[
			util["getMolAtIdx", n];
			imol = If[ 
				fixValences && explicitHydrogens && util["hasImplicitHydrogens"] && SameQ[ Quiet[ util["sanitize"]], 0]
				,
				replacementiMol = newInstance[];
				replacementiMol[ "createCopyWithAddedHydrogens", ManagedLibraryExpressionID @ util,False];
				replacementiMol
				,
				util
			];
			molData = molDataFixStereo[imol, 
				removeCoordinatesAndStereo @ Options[mol],
				Lookup[Flatten @ {opts},"FixStereochemistry",True]
			] ;
			
			Molecule @@ molData
			,
			{n,nmatches}
		];
		Switch[ type,
			"List", res,
			"All", res[[1]]
		] 
		
	) /; IntegerQ[nmatches] (*&& Positive[nmatches]*)
				
]

oReplaceSubstructure[___] := $Failed



(* ::Section::Closed:: *)
(*iSetGeometricProperty*)


stripBonds[mol_, input_ /; !FreeQ[input, Bond]] := Module[
	{res,bond},
	res = ReplaceAll[
		input, 
		{
			x:Bond[atms_,_] /; BondQ[mol, x] :> bond[atms],
			x_Bond :> (Message[ MoleculeModify::bond, x]; Throw[$Failed, $tag])
		}
	];
	res = Replace[
		res,
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
	
	If[
		!FreeQ[res, bond]
		, 
		Message[ MoleculeModify::modspc, input];
		Throw[$Failed, $tag]
	];
	res
]

stripBonds[mol_, x_] := x


iSetBondLength[ mol_, rule_Rule,opts:OptionsPattern[] ] := iSetBondLength[mol, {rule}, opts];

iSetBondLength[ mol_?MoleculeQ, rules:{HoldPattern[_ -> (_?NumericQ | _Quantity)]..}, opts:OptionsPattern[] ]  /; checkMessage3D[mol] := Module[
	{coords, nm = newInstance[],im, val, a1, a2, res, atoms, lengths, validated},
	coords = mol["AtomCoordinates", TargetUnits->None];
	(
	atoms = rules[[All, 1]];
	lengths = rules[[All, 2]];
	lengths = getMagnitude["distance"] /@ lengths;
	If[ !VectorQ[lengths, NumericQ], Throw[$Failed, $tag]];
	
	
	validated = atoms /. x_?Negative :> (mol["FullAtomCount"] + x + 1);
	validated = stripBonds[mol, validated];
	
	messageOnBadAtomReference[ mol["AtomIndex"], atoms, validated, $tag];
	Replace[
		messageOnAtomsDimension[ "BondLength", validated, atoms, MoleculeModify],
		$Failed :> Throw[$Failed,$tag]
	];
	
	
	im = getCachedMol[mol,"AllAtoms"];
	nm["createCopy", ManagedLibraryExpressionID @ im, <|"QuickCopy" -> False, "LeaveEditable" -> False |> ];
	
	If[
		!has3DCoordinates[ nm],
		nm["addCoordinates",coords]
	];
	
	Do[
		Replace[
			nm[ "setBondLength", Sequence @@ validated[[n]], lengths[[n]] ],
			msg:Except[ Null] :> (geometryMessage[ msg, validated[[n]], rules]; Throw[$Failed, $tag])
		]
		
		,{n, Length @ lengths}
	];
	
	coords = nm["get3DCoordinates"];
	
	res = replaceMoleculeOption[mol, AtomCoordinates, quantityArray[coords, "Angstroms"]];
	res
	) /; MatrixQ[coords, NumberQ]
]

iSetBondLength[___] := $Failed


iSetBondAngle[ mol_, rule_Rule,opts:OptionsPattern[] ] := iSetBondAngle[mol, {rule}, opts];

iSetBondAngle[ mol_?MoleculeQ, rules:{HoldPattern[_ -> (_?NumericQ | _Quantity)]..}, opts:OptionsPattern[] ]  /; checkMessage3D[mol] := Module[
	{coords, nm = newInstance[],im, val, a1, a2, res, atoms, angles, validated},
	coords = mol["AtomCoordinates", TargetUnits->None];
	(
	atoms = rules[[All, 1]];
	
	angles = rules[[All, 2]];
	angles = getMagnitude["angle"] /@ angles;
	If[ !VectorQ[angles, NumericQ], Throw[$Failed, $tag]];
	
	
	validated = atoms /. x_?Negative :> (mol["FullAtomCount"] + x + 1);
	
	validated = stripBonds[mol,validated];
	
	messageOnBadAtomReference[ mol["AtomIndex"], atoms, validated, $tag];
	Replace[
		messageOnAtomsDimension[ "BondAngle", validated, atoms, MoleculeModify],
		$Failed :> Throw[$Failed,$tag]
	];
	
	
	im = getCachedMol[mol,"AllAtoms"];
	nm["createCopy", ManagedLibraryExpressionID @ im, <|"QuickCopy" -> False, "LeaveEditable" -> False |> ];
	
	If[
		!has3DCoordinates[ nm],
		nm["addCoordinates",coords]
	];
	
	Do[
		Replace[
			nm[ "setBondAngle", Sequence @@ validated[[n]], angles[[n]] ],
			msg:Except[ Null] :> (geometryMessage[ msg, validated[[n]], rules]; Throw[$Failed, $tag])
		]
		
		,{n, Length @ angles}
	];
	
	coords = nm["get3DCoordinates"];
	
	res = replaceMoleculeOption[mol, AtomCoordinates, quantityArray[coords, "Angstroms"]];
	res
	) /; MatrixQ[coords, NumberQ]
]

iSetBondAngle[___] := $Failed



iSetTorsionAngle[ mol_, rule_Rule,opts:OptionsPattern[] ] := iSetTorsionAngle[mol, {rule}, opts];

iSetTorsionAngle[ mol_?MoleculeQ, rules:{HoldPattern[_ -> (_?NumericQ | _Quantity)]..}, opts:OptionsPattern[] ]  /; checkMessage3D[mol] := Module[
	{coords, nm = newInstance[],im, val, a1, a2, res, atoms, angles, validated},
	coords = mol["AtomCoordinates", TargetUnits->None];
	(
	atoms = rules[[All, 1]];
	
	angles = rules[[All, 2]];
	angles = getMagnitude["angle"] /@ angles;
	If[ !VectorQ[angles, NumericQ], Throw[$Failed, $tag]];
	
	
	validated = atoms /. x_?Negative :> (mol["FullAtomCount"] + x + 1);
	
	validated = stripBonds[mol,validated];
	
	messageOnBadAtomReference[ mol["AtomIndex"], atoms, validated, $tag];
	Replace[
		messageOnAtomsDimension[ "TorsionAngle", validated, atoms, MoleculeModify],
		$Failed :> Throw[$Failed,$tag]
	];
	
	
	im = getCachedMol[mol,"AllAtoms"];
	nm["createCopy", ManagedLibraryExpressionID @ im, <|"QuickCopy" -> False, "LeaveEditable" -> False |> ];
	
	If[
		!has3DCoordinates[ nm],
		nm["addCoordinates",coords]
	];
	
	Do[
		Replace[
			nm[ "setTorsionAngle", Sequence @@ validated[[n]], angles[[n]] ],
			msg:Except[ Null] :> (geometryMessage[ msg, validated[[n]], rules]; Throw[$Failed, $tag])
		]
		
		,{n, Length @ angles}
	];
	
	coords = nm["get3DCoordinates"];
	
	res = replaceMoleculeOption[mol, AtomCoordinates, quantityArray[coords, "Angstroms"]];
	res
	) /; MatrixQ[coords, NumberQ]
]

iSetTorsionAngle[___] := $Failed



(* ::Section::Closed:: *)
(*Kekulize*)


(* 	:Kekulize: *)

iKekulize[mol_?MoleculeQ, opts:OptionsPattern[]] := Which[
	FreeQ[mol[[2]], "Aromatic"],
		mol,
	Length[ getCachedMol[mol]["atomRings"]] === 0,
		mol,
	True,
		Molecule[ mol, IncludeAromaticBonds -> False]
]

(* 	:SetAromaticity: *)

iSetAromaticity[mol_?MoleculeQ, opts:OptionsPattern[]] := If[
	Length[ getCachedMol[mol]["atomRings"]] === 0
	,
	mol,
	Molecule[ mol, IncludeAromaticBonds -> True]
]


(* ::Section::Closed:: *)
(*RenumberAtoms*)


(* 	:RenumberAtoms: *)

Options[ iRenumberAtoms ] = {"FixStereochemistry" -> False}

iRenumberAtoms[mol_ /; MoleculeQ[mol] , newIndices:({__Integer}|{Rule[_Integer,_Integer]..}), opts:OptionsPattern[]] := Module[
	{
    		nm = newInstance[], imol = getCachedMol @ mol, 
    		indices = atomNameMap[mol] @ newIndices, natoms, explicitOnly, res,
    		molData, molOpts
    	},
	natoms = imol[ "atomCount", explicitOnly = True];
	
	If[
		MatchQ[indices, {__Rule}]
		,
		indices = Replace[ Range[natoms], indices, {1}]
	];
	
	If[
		Sort[indices] =!= Range @ natoms,
		Message[ MoleculeModify::atmlst, newIndices];
		Return[ $Failed, Block]
	];
	
	nm[
		"createCopy", 
		ManagedLibraryExpressionID @ imol, 
		<|"MakeEditable" -> False|>
	];
	
	nm["renumberAtoms", indices];
	
	molOpts = ReplaceAll[
		FilterRules[Options[mol], Except[StereochemistryElements]],
		{
			HoldPattern[AtomCoordinates -> x_?MatrixQ] :> Rule[ AtomCoordinates, x[[indices]] ],
			HoldPattern[AtomDiagramCoordinates -> x_?MatrixQ] :> Rule[ AtomDiagramCoordinates, x[[indices]] ]
		}
	];
	
	
	molData = molDataFixStereo[nm, 
		molOpts,
		Lookup[{opts},"FixStereochemistry",False]
	] ;
	
	Molecule @@ molData

]

iRenumberAtoms[mol_?MoleculeQ, indices_, ___] := (Message[ MoleculeModify::atmlst, indices]; $Failed)


(* ::Section::Closed:: *)
(*RemoveStereoChemistry*)


Options[iRemoveStereochemistry] = Options[iRemoveStereochemistry] = {}

iRemoveStereochemistry[mol_, atom:$AtomReferencePattern, rest___] := iRemoveStereochemistry[ mol, {atom}, rest]

iRemoveStereochemistry[ mol_, atomList:($AtomReferencePattern | {$AtomReferencePattern..}):All, opts:OptionsPattern[]] := Block[
	{stereo,newMol, atoms, newStereo},
	stereo = mol["StereochemistryElements"];
	
	Switch[ atomList,
		All,	
		replaceMoleculeOption[ mol, StereochemistryElements],
		
		x_List /; MatchQ[ atoms = atomNameMap[mol] @ x, {__Integer}],
		newStereo = removeStereoFromAtoms[ stereo, atoms];
		replaceMoleculeOption[ mol, StereochemistryElements, newStereo],
		_,
		mol
	]	
]

iRemoveStereochemistry[___] := $Failed



(* ::Section::Closed:: *)
(*AssignStereochemistryFromCoordinates*)


iAssignStereochemistryFromCoordinates[ mol_ /; MoleculeQ[mol], opts:OptionsPattern[] ]  /; checkMessage3D[mol] := Block[
	{imol = getCachedMol[mol], im2 = newInstance[], coords = mol["AtomCoordinates",TargetUnits -> None], stereo},
	im2["createCopy", ManagedLibraryExpressionID @ imol, <| "QuickCopy" -> True, "MakeEditable" -> False |>];
	im2["addCoordinates", coords];
	im2["setStereoFrom3D"];
	stereo = getiMolProperty[ im2, "StereochemistryElements"];
	
	replaceMoleculeOption[ mol, StereochemistryElements, stereo]
]




(* ::Section::Closed:: *)
(*ComputeAtomCoordinates*)

Options[iGenerateConformers] = Options[iGenerateConformers] = {
	Method -> Automatic,
	RandomSeeding -> Automatic,
	"EnergyMinimize" -> False,
	"Canonicalize" -> False,
	"EnforceChirality" -> True,
	"MaxAttempts" -> Automatic,
	"AlignConformers" -> True
}

iGenerateConformers[ mol_, nConformers_, opts:OptionsPattern[]] := Module[
	{imol, returnMol, coords, options},
	options = With[ {names = Keys @ Options @ iGenerateConformers},
		Thread[ names -> System`Utilities`GetOptionValues[iGenerateConformers, names , Flatten@{opts}] ]
	];
	AppendTo[options, "ConformerCount" -> nConformers];
	
	{imol, returnMol} = Replace[
		Block[{$nConformers = nConformers},
			iComputeAtomCoordinates[ mol, options]
		],
		Except[{_iMolecule, _Molecule}] :> Throw[$Failed, $tag]
	];
	
	coords = Replace[
		imol["getAll3DCoordinates"],
		Except[ x_?ArrayQ /; Dimensions[x] === {nConformers, AtomCount[returnMol], 3}] :> Throw[$Failed, $tag]
	];
	
	
	Molecule[ returnMol, AtomCoordinates -> quantityArray[ #, "Angstroms"] ] & /@ coords
	
]


Options[iComputeAtomCoordinates] = {
	Method -> Automatic,
	RandomSeeding -> 1234,
	"EnergyMinimize" -> True,
	"EnforceChirality" -> True,
	"MaxAttempts" -> Automatic}
iComputeAtomCoordinates[ molecule_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{imol, returnMol, res, method, seed, embedParameters, options = {opts}},
	
	returnMol = If[
		molecule["HasImplicitHydrogens"],
		iAddHydrogens[molecule],
		molecule
	];
		
	If[ !MoleculeQ[returnMol], returnMol = molecule];
	
	method = Replace[
		OptionValue[ iComputeAtomCoordinates, FilterRules[options, Method], Method],
		{
			x_String :> {x},
			x:{_String, ___Rule} :> x,
			x:{__Rule} :> x,
			_ :> {}
		}
	];
	options = Join[
		method,
		FilterRules[ {opts}, Except[ Method ] ]
	];

	(* we now have AtomCoordinates -> Automatic, and Method -> Except[Automatic] *)
	
	embedParameters = getEmbedParameters[
		options,
		MoleculeModify
	];

	
	imol = newInstance[];
	imol["createCopy", 
		ManagedLibraryExpressionID @ getCachedMol[returnMol, "AllAtoms"]
	];
	
	
	If[ IntegerQ @ $nConformers, 
		res = embedMol[imol, embedParameters];
		Return[ {imol,returnMol}, Module]
		,
		res = embedMol[imol, embedParameters]
	];
	
	If[ 
		!Negative[ res ],
		alignConnectedComponents @ imol
	];
	
	If[ 
		!(IntegerQ[res] && res > -1),
		Message[MoleculeModify::nocoord];
		Throw[$Failed, $tag]
	];
	
	res = imol["get3DCoordinates"];
	(
		Molecule[ returnMol, AtomCoordinates -> quantityArray[ res, "Angstroms"] ]
	) /; MatrixQ[res]
]

iComputeAtomCoordinates[___] := $Failed



(* ::Section::Closed:: *)
(*CanonicalizeAtomCoordinates*)

iCanonicalizeAtomCoordinates[ mol_?MoleculeQ, OptionsPattern[]] := Module[
	{coords = mol["AtomCoordinates", TargetUnits->None], newCoords, im = getCachedMol[ mol, "AllAtoms"]},
	(
		newCoords = canonicalizeConformer[im];
		replaceMoleculeOption[mol, AtomCoordinates, quantityArray[newCoords, "Angstroms"]] /; MatrixQ[newCoords]
	
	) /; MatrixQ[coords]
]

iCanonicalizeAtomCoordinates[___] := $Failed

(* ::Section::Closed:: *)
(*EnergyMinimizeAtomCoordinates*)


Options[iEnergyMinimizeAtomCoordinates] = {
	MaxIterations -> Automatic,
	"ForceField" -> "MMFF94"
}

$constraintsPattern = Rule[ _Integer | {__Integer}, _] | {Rule[ _Integer | {__Integer}, _] ..} | _Association

iEnergyMinimizeAtomCoordinates[ mol_?MoleculeQ , constraintsIn:$constraintsPattern:<||>, opts:OptionsPattern[] ] := Block[
	{coords, nm = newInstance[], im = getCachedMol[ mol, "AllAtoms"], 
		return, init, res, attempts, constraints, params},
	
	If[ 
		hasImplicitHydrogens[mol], 
		Throw[ iEnergyMinimizeAtomCoordinates[ iAddHydrogens[mol]], $tag] 
	];
	coords = mol["AtomCoordinates", TargetUnits->None];
	(nm["createCopy", ManagedLibraryExpressionID @ im, <|"QuickCopy" -> False, "LeaveEditable" -> False |> ];
	
	params = <| "MMFFVariant" -> 
		Switch[ OptionValue[ iEnergyMinimizeAtomCoordinates, {opts}, "ForceField"],
			"MMFF94","MMFF94",
			_,"MMFF94s"
		] 
	|>;
	
	init = MatchQ[
		nm[ "initializeMMFF", params],
		Except[ _LibraryFunctionError]
	];
	(
	constraints = Replace[
		constraintsIn,
		Except[_?AssociationQ] :> parseConstraints[constraintsIn]
	];
	
	If[
		MatchQ[
			constraints["DistanceConstraints"],
			{KeyValuePattern[
				{
					"Atoms" -> {_Integer, _Integer},
					"Value" -> _?NumericQ,
					"ForceConstant" -> _?NumericQ
				}
			]..}
		],
		nm["addDistanceConstraint", Sequence @@ #Atoms, N @ #Value, N @ #ForceConstant] & /@ constraints["DistanceConstraints"];
	];
	
	If[
		MatchQ[
			constraints["AngleConstraints"],
			{KeyValuePattern[
				{
					"Atoms" -> {_Integer, _Integer, _Integer},
					"Value" -> _?NumericQ,
					"ForceConstant" -> _?NumericQ
				}
			]..}
		],
		nm["addAngleConstraint", Sequence @@ #Atoms, N @ #Value, N @ #ForceConstant] & /@ constraints["AngleConstraints"];
	];

	If[
		MatchQ[
			constraints["TorsionConstraints"],
			{KeyValuePattern[
				{
					"Atoms" -> {_Integer, _Integer, _Integer, _Integer},
					"Value" -> _?NumericQ,
					"ForceConstant" -> _?NumericQ
				}
			]..}
		],
		nm["addTorsionConstraint", Sequence @@ #Atoms, N @ #Value, N @ #ForceConstant] & /@ constraints["TorsionConstraints"];
	];

	res = 1;
	attempts = 0;
	While[
		res =!= 0 && attempts < 25,
		res = nm["minimize", 2000];
		attempts++;
	];
	nm["clearMMFF"];
	coords = nm["get3DCoordinates"];
	
	res = replaceMoleculeOption[mol, AtomCoordinates, quantityArray[coords, "Angstroms"]];
	nm[ "clearMMFF"];
	res
	) /; init
	) /; MatrixQ[coords, NumberQ]
	
]


parseConstraints[constraint_Rule] := parseConstraints[{constraint}]
parseConstraints[constraints:{__Rule}] := Module[
	{res = <|"TorsionConstraints"->{},"AngleConstraints"->{},"DistanceConstraints"->{}|> },
	
	Replace[
		constraints,
		{
			HoldPattern[Rule[atms:{_Integer, _Integer}, x_]] :> AppendTo[ res["DistanceConstraints"],
				<|"Atoms" -> atms, "Value" -> getMagnitude["distance"][x], "ForceConstant" -> 10^4 |>],
			HoldPattern[Rule[atms:{_Integer, _Integer, _Integer }, x_]] :> AppendTo[ res["AngleConstraints"],
				<|"Atoms" -> atms, "Value" -> getMagnitude["angle"][x], "ForceConstant" -> 10^4 |>],
			HoldPattern[Rule[atms:{_Integer, _Integer, _Integer, _Integer }, x_]] :> AppendTo[ res["TorsionConstraints"],
				<|"Atoms" -> atms, "Value" -> getMagnitude["angle"][x], "ForceConstant" -> 10^3 |>]
		},
		{1}
	];
	res
]

parseConstraints[___] := {}

iEnergyMinimizeAtomCoordinates[___]:= $Failed

getMagnitude["distance"][HoldPattern[x_Quantity]] := QuantityMagnitude[ UnitConvert[ x, "Angstroms"]];
getMagnitude["angle"][HoldPattern[x_Quantity]] := QuantityMagnitude[ UnitConvert[ x, "AngularDegrees"]];
getMagnitude[_][x_] := x


(* ::Section::Closed:: *)
(*RemoveHydrogens*)


Options[iRemoveHydrogens] = {"MakeImplicit" -> True, "FixStereochemistry" -> True}
iRemoveHydrogens[mol_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{imol, copy, molData, molOptions, coords},
	
	If[
		FreeQ[ mol[[1]], "H"]
		,
		Return[mol, Module]
	];
	copy = newInstance[];
	imol = getCachedMol[mol];
	copy[ "createCopyWithNoHydrogens", ManagedLibraryExpressionID @ imol, True, False];
	
	If[
		!TrueQ[ optionValue[ iRemoveHydrogens, {opts}, "MakeImplicit"]],
		copy["convertHydrogensToHairs"]
	];
	
	molOptions = FilterRules[
		Options[mol], 
		Except[StereochemistryElements | AtomCoordinates | AtomDiagramCoordinates | IncludeHydrogens]
	];
	
	AppendTo[molOptions, IncludeHydrogens -> False ];
	
	If[
		has2DCoordinates[copy] && MatrixQ[coords = copy["get2DCoordinates"]],
		AppendTo[molOptions, AtomDiagramCoordinates -> coords ]
	];
	
	If[
		has3DCoordinates[copy] && MatrixQ[coords = copy["get3DCoordinates"]],
		AppendTo[molOptions, AtomCoordinates -> quantityArray[coords, "Angstroms"] ]
	];
	
	molData = molDataFixStereo[copy, 
		molOptions,
		Lookup[Flatten @ {opts},"FixStereochemistry",True]
	];
	
			
	Molecule @@ molData
]

iRemoveHydrogens[___] := $Failed


(* ::Section::Closed:: *)
(*AddHydrogens*)


iAddHydrogens[mol_?MoleculeQ,opts:OptionsPattern[]] := If[
	!hasImplicitHydrogens[mol]
	,
	mol
	,
	Module[
		{imol = getCachedMol[mol, "AllAtoms"], molData},
		
		molData = molDataFixStereo[imol, 
			FilterRules[Options[mol], Except[StereochemistryElements | IncludeHydrogens | AtomDiagramCoordinates | AtomCoordinates]],
			Lookup[Flatten @ {opts},"FixStereochemistry",True]
		];
				
		Molecule @@ molData
	]
]

iAddHydrogens[___] := $Failed


(* ::Section::Closed:: *)
(*Canonicalize*)


iCanonicalize[$EmptyMolecule, opts:OptionsPattern[]] := Molecule[{},{}]

iCanonicalize[mol_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{smiles = mol["SMILES"],res},
	res = Molecule[ smiles ];
	If[
		smiles === res[ "SMILES" ]
		,
		res
		,
		Molecule[res["SMILES"]]
	]
]



(* ::Section::Closed:: *)
(*TransformAtomCoordinates*)


Options[iTransformAtomCoordinates] = {"FixStereochemistry" -> True}

iTransformAtomCoordinates[mol_?MoleculeQ, t_, opts:OptionsPattern[]] := Module[
	{coords = mol["AtomCoordinates"], tfunc, newCoords,res},
	tfunc = quantifyTransformationFunction @ t;
	If[ !quantityArrayQ[coords], Throw[$Failed, $tag]];
	
	newCoords = tfunc[coords];
	
	res = If[
		quantityArrayQ[
			newCoords = Quiet[quantityArray[ newCoords] ]
		]
		,
		Molecule[mol, AtomCoordinates -> newCoords]
		,
		$Failed
	];
	If[ And[
			MoleculeQ[res],
			reflectionQ[tfunc],
			MemberQ[ mol["StereochemistryElements"],KeyValuePattern["StereoType" -> "Tetrahedral"]],
			TrueQ @ OptionValue["FixStereochemistry"]
		],
		res = DeleteCases[
			iAssignStereochemistryFromCoordinates[res],
			KeyValuePattern[
				"ChiralCenter" -> Except[Alternatives @@ Lookup[mol@"StereochemistryElements","ChiralCenter"]]
			],
			Infinity
		] 
	];
	res
]

quantifyTransformationFunction[ HoldPattern[TransformationFunction[ mat_?MatrixQ ]] ] /; Dimensions[ mat] === {4,4} := 
	Module[{trans = mat[[;;3,4]],newmat},
		(
			newmat = mat;
			newmat[[;;3,4]] = Quantity[ newmat[[;;3,4]], "Angstroms"];
			TransformationFunction[newmat]
		)/; Max[trans] > 0 && FreeQ[mat,Quantity]
	]
	
quantifyTransformationFunction[arg_] := arg 


reflectionQ[ HoldPattern[TransformationFunction[ mat_?MatrixQ ]] ] /; Dimensions[ mat] === {4,4} := SameQ[
	Sign[ Det[ mat[[;;3,;;3]] ] ],
	-1
]

reflectionQ[___] := False

$rule = _Rule | _RuleDelayed | {(_Rule | _RuleDelayed)..}


(* ::Section::Closed:: *)
(*SetMetaInformation*)


iSetMetaInformation[mol_?MoleculeQ, meta_, OptionsPattern[] ] /; FreeQ[mol,meta] := 
	Molecule[mol, MetaInformation -> meta]
	
iSetMetaInformation[mol_?MoleculeQ,___] := mol

iSetMetaInformation[___] := $Failed

optionValue[func_, opts_, opt_] := OptionValue[{func,MoleculeModify}, FilterRules[opts, Options/@ {func,MoleculeModify}], opt]


(* ::Section:: *)
(*SetAtomProperty*)

iSetAtomProperty[prop_][mol_?MoleculeQ, rule:Rule[_Integer,_], opts:OptionsPattern[]] := iSetAtomProperty[prop][mol,{rule},opts]

iSetAtomProperty[prop_][mol_?MoleculeQ, rules:{Rule[_Integer,_] ..}, opts:OptionsPattern[]] := Module[
	{newAtoms, inputAtoms, atomsToDelete, newValues, res},
	inputAtoms = Keys @ rules;
	newValues = Values @ rules;
	atomsToDelete = atomNameMap[mol][inputAtoms];	
	Check[Catch[
		messageOnBadAtomReference[ mol["AtomIndex"], inputAtoms, atomsToDelete, "badatom"],
		"badatom"], Return[$Failed, Module];
	];
	
	newAtoms = replaceAtomProperty @@@ Thread[ {mol[[1,atomsToDelete]], prop, newValues}];
	If[
		prop === "MassNumber"
		,
		res = List @@ mol;
		res[[1,atomsToDelete]] = newAtoms;
		Molecule @@ res
		,
		iReplaceAtom[
			mol,
			Thread[ atomsToDelete -> newAtoms],
			opts
		]
	]
		
]



iSetAtomProperty["MassNumber"][mol_?MoleculeQ, opts:OptionsPattern[]] := Module[
	{nums},
	nums = AtomList[mol, _, {"AtomIndex", "MostAbundantMassNumber"}];
	iSetAtomProperty["MassNumber"][mol, Rule @@@ nums]
]

replaceAtomProperty[atom:Atom[sym_,rules___], prop_, val_] := Atom[
	sym,
	Sequence @@ FilterRules[ {rules}, Except[prop]],
	prop -> val
]; 

(* ::Section:: *)
(*SetAtomChirality*)


iSetAtomChirality[mol_?MoleculeQ, rule_Rule, opts:OptionsPattern[]] := iSetAtomChirality[ mol, {rule}, opts];
iSetAtomChirality[mol_, {}, OptionsPattern[]] := mol;
iSetAtomChirality[mol_?MoleculeQ, rules:{Rule[_Integer,Join[$AtomStereoTypes,Alternatives["Undefined"]]] ..}, opts:OptionsPattern[]] := Module[
	{stereoIn,stereoOut,inputAtoms,newValues},
	
	stereoIn = Replace[
		OptionValue[ Molecule, Options @ mol, StereochemistryElements],
		Except[ {__?AssociationQ}] :> {}
	];
	
	inputAtoms = Keys @ rules;
	
	Check[Catch[
		messageOnBadAtomReference[ mol["AtomIndex"], inputAtoms, inputAtoms, "badatom"],
		"badatom"], Return[$Failed, Module];
	];
	
	stereoOut = Join[
		DeleteCases[ stereoIn, KeyValuePattern[ "ChiralCenter" -> (Alternatives @@ inputAtoms)]],
		DeleteCases[ rules, HoldPattern[ _ -> "Undefined"]]
	];
	
	Molecule[
		mol[[1]],
		mol[[2]],
		Sequence @@ FilterRules[ Options[mol], Except[StereochemistryElements | AtomCoordinates | AtomDiagramCoordinates]],
		StereochemistryElements -> stereoOut
	]
		
]


iSetBondStereo[mol_?MoleculeQ, rule_Rule, opts:OptionsPattern[]] := iSetBondStereo[ mol, {rule}, opts];
iSetBondStereo[mol_, {}, OptionsPattern[]] := mol;
iSetBondStereo[mol_?MoleculeQ, rules:{Rule[{_Integer,_Integer}|_Bond,$BondStereoTypes] ..}, opts:OptionsPattern[]] := Module[
	{stereoIn,stereoOut,inputAtoms,newValues},
	
	stereoIn = Replace[
		OptionValue[ Molecule, Options @ mol, StereochemistryElements],
		Except[ {__?AssociationQ}] :> {}
	];
	
	inputAtoms = Replace[
		Keys @ rules,
		{
			b:Bond[_, "Double"] /; BondQ[mol,b] :> First @ b,
			{a_,b_} /; BondQ[mol, Bond[{a,b},"Double"]] :> {a,b},
			x_ :> (Message[MoleculeModify::bond, x]; Return[$Failed, Module])
		},
		{1}
	];
		
	newValues = Values @ rules;

	stereoOut = Join[
		DeleteCases[ stereoIn, KeyValuePattern[ "StereoBond" -> (Alternatives @@ inputAtoms)]],
		Thread[ inputAtoms -> newValues]
	];
	
	Molecule[
		mol[[1]],
		mol[[2]],
		Sequence @@ FilterRules[ Options[mol], Except[StereochemistryElements | AtomCoordinates | AtomDiagramCoordinates]],
		StereochemistryElements -> stereoOut
	]
		
]

(* ::Section::Closed:: *)
(*utilities*)

If[ !StringQ[#1], Set[ #1, #2]] & @@@ {
	{MoleculeModify::bnded, "Atoms `1` and `2` should be bonded."},
	{MoleculeModify::rngbnd, "The bond containing atoms `1` and `2` should not belong to a ring."},
	{MoleculeModify::rngbnds, "Bonds {`1`,`2`} and {`2`,`3`} should not both belong to a ring."}
	
}


geometryMessage[ msg_String, atoms_, rules_] := Catch[StringReplace[
	msg,
	{
		"atoms i and j must be bonded" :> (Message[ MoleculeModify::bnded, atoms[[1]], atoms[[2]]]; Throw[$Failed, "msg"]),
		"atoms j and k must be bonded" :> (Message[ MoleculeModify::bnded, atoms[[2]], atoms[[3]]]; Throw[$Failed, "msg"]),
		"bond (i,j) must not belong to a ring" :> (Message[ MoleculeModify::rngbnd, atoms[[1]], atoms[[2]]]; Throw[$Failed, "msg"]),
		"bond (j,k) must not belong to a ring" :> (Message[ MoleculeModify::rngbnd, atoms[[2]], atoms[[3]]]; Throw[$Failed, "msg"]),
		"bonds (i,j) and (j,k) must not both belong to a ring" :> (Message[ MoleculeModify::rngbnds, atoms[[1]], atoms[[2]], atoms[[3]]]; Throw[$Failed, "msg"])
	}
], "msg"]


removeCoordinatesAndStereo = DeleteCases[ 
	HoldPattern[ 
		Rule[
			AtomDiagramCoordinates | AtomCoordinates | StereochemistryElements,
			_
		]
	]
]

removeCoordinates = DeleteCases[ 
	HoldPattern[ 
		Rule[
			AtomDiagramCoordinates | AtomCoordinates,
			_
		]
	]
]

handleValencesOption[opts:{___?OptionQ}] := MatchQ[
	OptionValue[ MoleculeModify, FilterRules[opts, Options @ MoleculeModify], ValenceErrorHandling],
	True | Automatic
]
handleValencesOption[opts___?OptionQ] := handleValencesOption[{opts}]

includeHydrogens[opts:KeyValuePattern[IncludeHydrogens -> _]][molOpts_] := Join[
	FilterRules[molOpts, Except[IncludeHydrogens]],
	FilterRules[ opts, IncludeHydrogens]
]

includeHydrogens[___][arg_] := arg

messageOnBadAtomInput[ input_, atoms_, tag_] :=
	Replace[
		Thread[ {input, atoms} ],
		{inp_, atom:Except[Atom[$ElementsPattern, OptionsPattern[] ] ]} :> 
			(Message[ Molecule::atom, inp]; Throw[ $Failed, tag]),
		1
	]


checkMessage3D[ mol:Molecule[args__,opts:OptionsPattern[]] ] := With[
	{ coords = mol["AtomCoordinates",TargetUnits->None]},
	If[
		quantityArrayQ[ coords] || MatrixQ[ coords, NumericQ],
		True,
		Message[ MoleculeModify::nombed]; False
	]
]




removeStereoFromAtoms[ stereo:{__Association}, atoms : {__Integer} ] := With[
	{
		patt1 = KeyValuePattern[ "ChiralCenter" -> (Alternatives @@ atoms)],
		patt2 = KeyValuePattern[ "StereoBond" -> {OrderlessPatternSequence[Alternatives @@ atoms, _]}]
	},
	DeleteCases[ stereo, patt1 | patt2]
]

	
	
connectedHs[mol_, vert_] := With[
	{atoms = First @mol, graph = getMolGraph @ mol},
	connectedHs[graph, atoms, vert]
]

connectedHs[ graph_, atoms_, vert_] := With[ {
	connectedAtoms = AdjacencyList[graph, vert]},
	Pick[
		connectedAtoms,
		atoms[[ connectedAtoms]],
		Atom["H"]
	]
]


coordsFromExpression[mol_,Optional[dim_, "3D"]] := Lookup[
	Options[ mol] ,
	Switch[
		dim, 
		"2D", AtomDiagramCoordinates, 
		_,  AtomCoordinates
	]
] 

(*replaceMoleculeOption[ expr_, Rule[ option_, value_] ] := replaceMoleculeOption[ expr, option, value];*)
	
replaceMoleculeOption[expr_, option:(_Symbol|_String), value_:Nothing] := Module[ 
	{args, opts},
	
	With[ {expression = expr},
		{args, opts} = Internal`UnsafeQuietCheck[
			System`Private`Arguments[expression,{1,2}],
			Return[ expr, Module]
		];
	];
	
	opts = DeleteCases[ opts, HoldPattern[  option -> _ ] ];
	
	If[ 
		value =!= Nothing,
		AppendTo[ opts, option -> value]
	];
	
	Molecule[ Sequence @@ args, Sequence @@ opts ]
]


(*loadSMARTSString Throws an exception *)

loadSMARTSString[MoleculePattern[s_String]] := loadSMARTSString[s];
loadSMARTSString[s_String] := Block[ {nm = newInstance[]},
	Replace[
		nm["createMolFromSmarts", s],
		-1 :> ( Message[ MoleculePattern::invldtd, s]; Throw[ $Failed, $tag] )
	];
	nm
]

loadSMILESString[s_String] := Check[
	Block[{nm = newInstance[]},
		nm["createMolFromSmiles", s, 2(*TODO:why the magic number?*)];
		nm
	],
	Throw[$Failed, $tag]
]
  

quietlyRemoveBadStereocenters[atoms_, bonds_, stereo_] := Quiet @ Block[
	{m = Molecule[atoms, bonds], possible},
	possible = If[ 
		MoleculeQ[m],
		m["PossibleStereocenters"],
		Return[stereo, Block]
	];
	
	DeleteCases[
		stereo,
		KeyValuePattern[ "ChiralCenter" -> Except[(Alternatives @@ possible)]]
	]
]

quietlyRemoveBadStereocenters[ _, _, {}] := {}

moleculeArguments[mol_] := System`Private`Arguments[mol, 2]


fixH[True][mol_?MoleculeQ] := If[
	TrueQ[!mol["HasValenceErrors"] && mol["HasImplicitHydrogens"]]
	,
	Molecule[mol, IncludeHydrogens -> True]
	,
	mol
]

fixH[True][mols:{__?MoleculeQ}] := fixH[True] /@ mols

fixH[___][x_] := x



End[] (* End Private Context *)

