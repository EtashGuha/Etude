


Begin["Chemistry`Private`MoleculeSubstructureDump`"]



(* ::Section::Closed:: *)
(*MoleculePattern*)

MoleculePattern[mol_?MoleculeQ] := MoleculePattern[ mol, AtomList[mol, Except["H"], "AtomIndex"]]

MoleculePattern[ mol_?MoleculeQ, indices:{__Integer}] /; 
	AllTrue[indices, TrueQ @* Between[{1,Length[First[mol]] } ]] := Module[
	{map,atms,alt = Alternatives @@ indices, bnds},
	map = AssociationThread[ indices -> Range @ Length @ indices];
	atms = mol[[1,indices]];
	bnds = Cases[
		mol[[2]],
		Bond[ a:{alt,alt}, b___] :> Bond[ Lookup[map,a],b]
	];
	MoleculePattern[atms,bnds]
	
]


(* ::Section::Closed:: *)
(*ValidQuery*)


Chemistry`MoleculePatternQ[args___] := ValidQuery[args]

$IssueMessage = False;
$MessageHead = Null;

(* 
	ValidQuery will attempt to create a query molecule in the library, 
	and returns True if it is able to do so, False otherwise
*)

(* 
	MessageValidQuery[head] will call ValidQuery, and will issue a
	head::molpat message if the result is False.
*) 

MessageValidQuery[sym_][arg_] := Block[
	{$IssueMessage = True, $MessageHead = sym, $PattMol = arg},
	ValidQuery[ arg]
]

ValidQuery[ Molecule[{}, ___] ] := False
ValidQuery[ m_Molecule ] := MoleculeQ[m]

ValidQuery[mol_] := Switch[getCachedMol[mol,{IncludeHydrogens -> Automatic}],
	_iMolecule, 
		True,
	$Failed, 
		False,
	_, 
		validQuery[mol]
]


ValidQueryAtom[ atom_?ValidQuery] := getCachedMol[atom,{IncludeHydrogens -> Automatic}]["atomCount",True] === 1
ValidQueryAtom[___] := False

ValidQueryBond[ bond_?ValidQuery] := With[
	{im = getCachedMol[bond,{IncludeHydrogens -> Automatic}]},
	im["atomCount",True] === 2 && im["bondCount",True] === 1
] 
ValidQueryBond[___] := False

(* 	:validQuery: *)

validQuery[ Atom[ {_Integer}] ] := False
validQuery[ Atom[ {_Integer}] ] := False


validQuery[mol_] := Module[ 
	{nm = newInstance[], valid},
	valid = Catch[
		validateQuery[nm, mol],
		$tag
	];
	If[
		(TrueQ[valid] && checkMolecule[nm] ),
		cacheMolecule[ mol, nm];
		True,
		cacheMolecule[ mol, $Failed];
		False
	] 
];





(* :iQueryMolecule: *)

validateQuery[ imol_, query:MoleculePattern[smarts_String] ] := Module[
	{res,sma},
	res = imol["createMolFromSmarts", smarts];
	If[
		res === -1 && StringQ[sma = NameToSMILES[smarts]]
		,
		res = imol["createMolFromSmarts", sma]
	];
	res =!= -1
]


validateQuery[imol_, queryMol:MoleculePattern[{},bondsInput:{}]] := validateQuery[imol, MoleculePattern[""]]
validateQuery[imol_, queryMol:MoleculePattern[atomsInput_,bondsInput_]] := Module[
	{atoms,bonds},
	atoms = canonicalizeAtoms[atomsInput];
	atoms = Replace[
		makeQueryAtom /@ atoms,
		Except[ {{__?AssociationQ}..}] :> throwOnBadQuery[](*Return[False, Module]*)
	];
	bonds = makeQueryBond /@ bondsInput;
	imol["createQueryMolecule", {atoms, bonds}];
	True
]
validateQuery[imol_,queryMol:MoleculePattern[atomsInput_List] ] := validateQuery[ imol, MoleculePattern[ atomsInput, {}]]

$bondTypePattern = (_String | Verbatim[Blank[]] | Verbatim[Alternatives][$BondTypes..] | PatternSequence[]);
$queryAtomPattern = (_Atom | _String | Verbatim[Alternatives][(_Atom|_String)..] | Verbatim[Except][_String] | Verbatim[Except][Verbatim[Alternatives][__String]]);
validateQuery[imol_, queryAtom:$queryAtomPattern] := validateQuery[imol, MoleculePattern[{queryAtom},{}]];

(* _ChemicalBond *)
validateQuery[imol_, Verbatim[Blank][Bond] ] := validateQuery[ imol, MoleculePattern[ { _, _}, {Bond[{1,2}]} ] ];

(* Bond[ __] | Bond[ ___] *)
validateQuery[imol_, Bond[Verbatim[BlankSequence[]] | Verbatim[BlankNullSequence[]]] ] := 
	validateQuery[ imol, MoleculePattern[ { _, _}, {Bond[{1,2}]} ] ];
	
(* Bond[ _, type_] | Bond[ _List, type_] | Bond[ {_,_}, type_]  *)
validateQuery[imol_, Bond[ Verbatim[Blank[]] | Verbatim[ Blank[List]] | {Verbatim[Blank[]],Verbatim[Blank[]]}, type:$bondTypePattern] ] := 
	validateQuery[ imol, MoleculePattern[{_, _}, {Bond[ {1, 2}, type]} ] ]
	
(* Bond[ {atom1, atom2}, type] *)	
validateQuery[imol_, Bond[ {atom1:$queryAtomPattern, atom2:$queryAtomPattern}, type:$bondTypePattern] ] := 
	validateQuery[ imol, MoleculePattern[{atom1, atom2}, {Bond[ {1, 2}, type]} ] ]
	
(* Bond[ {atom1, _}, type] *)	
validateQuery[imol_, Bond[ {OrderlessPatternSequence[ atom1:$queryAtomPattern, Verbatim[Blank[]] | Verbatim[Blank[Integer]]]}, type:$bondTypePattern] ] := 
	validateQuery[ imol, MoleculePattern[{atom1, Atom[_]}, {Bond[ {1, 2}, type]} ] ]



validateQuery[___] := throwOnBadQuery[];

throwOnBadQuery[] := (
	If[
		TrueQ @ $IssueMessage && $MessageHead =!= Null,
		With[
			{sym = $MessageHead},
			Message[MessageName[ sym, "molpat"], $PattMol]
		]
	];
	Throw[$Failed, $tag]
)
makeQueryAtom[ Verbatim[Atom[_]]] := {<||>}
makeQueryAtom[ Verbatim[Atom[]]] := {<||>}

(*makeQueryAtom[Atom[el:$ElementsPattern, after___Rule]] := Atom[ "AtomicNumber" -> AtomicNumber[el], after]*)

makeQueryAtom[ el_String ] := makeQueryAtom[ Atom[ el]]
makeQueryAtom[ el:Verbatim[Alternatives][__String] ] := makeQueryAtom[ Atom[ el]]

(* Atom[_] and Atom[ _, opts] *)
makeQueryAtom[ Atom[ Verbatim[Blank[]], after___Rule] ] := makeQueryAtom[ Atom[ after] ];

(* Atom[ sym, opts] *)
makeQueryAtom[ Atom[ element_String, after___Rule] ] := With[
	{atNum = Replace[
		AtomicNumber[element],
		Except[_Integer] :> throwOnBadQuery[]
	]},
	makeQueryAtom[ Atom["AtomicNumber" -> atNum, after] ]
]
	
(* Atom[ sym1] | Atom[ sym2] *)	
makeQueryAtom[Verbatim[Alternatives][atoms:(Atom[_String]..)] ] := 
	makeQueryAtom[ Atom[ Alternatives @@ {atoms}[[All,1]] ] ];
	
(* Atom[ sym1, opts1] | Atom[ sym2, opts2] *)	
makeQueryAtom[Verbatim[Alternatives][atoms__Atom] ] := makeQueryAtom /@ {atoms}
	
(* Atom[ sym1 | sym2, opts] *)
makeQueryAtom[ Atom[ Verbatim[ Alternatives][syms__String], after___Rule] ] := With[
	{atNums = Replace[
		AtomicNumber[{syms}],
		Except[_Integer] :> throwOnBadQuery[],
		1
	]},
	makeQueryAtom[ Atom["AtomicNumber" -> (Alternatives @@ atNums), after] ]
]

(* Atom[Except["C"]] and Atom[Except["C" | "Fe"]] *)
makeQueryAtom[ patt: (Verbatim[Except][sym__String | Verbatim[ Alternatives][syms__String]])] := makeQueryAtom[Atom[patt]];
makeQueryAtom[ Atom[ Verbatim[Except][sym__String | Verbatim[ Alternatives][syms__String]], after___Rule] ] := With[
	{atNums = Replace[
		AtomicNumber[{sym,syms}],
		Except[_Integer] :> throwOnBadQuery[],
		1
	]},
	makeQueryAtom[ Atom["AtomicNumber" -> Except[(Alternatives @@ atNums)], after] ] 
]

$emptyRule = HoldPattern[Rule[_,Verbatim[Blank[]]]];
makeQueryAtom[ atm_Atom /; !FreeQ[atm,$emptyRule] ] := makeQueryAtom[ DeleteCases[atm, $emptyRule, Infinity] ];

makeQueryAtom[ Atom[ rules___Rule ] ] := Module[
	{transformedRules},
	transformedRules = atomQueryRules /@ {rules} 
]

makeQueryAtom[___] := throwOnBadQuery[];

$integerQueryRules = Apply[Alternatives,
	{
		"AtomicNumber", "OrbitalHybridization", "FormalCharge", "MassNumber", "ImplicitHydrogenCount",
		(*"ExplicitHydrogenCount",*) "HydrogenCount", (*"ChiralTag",*) "ImplicitValence", "ExplicitValence",
		"UnpairedElectronCount", "CIPRank", "CoordinationNumber", "HeavyAtomCoordinationNumber","PiElectronCount",
		"Valence", "OuterShellElectronCount", "AtomChirality"
	}
]; 
$doubleQueryRules = Alternatives @@ {"GasteigerPartialCharge", "MMFFPartialCharge",
	 "TopologicalStericEffectIndex", "GeometricStericEffectIndex", "AtomicMass"}

$booleanQueryRules = "AromaticAtomQ" | "RingMemberQ" | "HasImplicitHydrogens" | "UnsaturatedAtomQ"


atomQueryRules[ Rule[ prop:Join[$integerQueryRules,$doubleQueryRules], atnums_] ] := Module[
	{values, test, negated, queryType, valueType, intQuery},
	test = Switch[prop,
		"AtomicNumber", 
			validAtomicNumber,
		"FormalCharge", 
			IntegerQ,
		"MassNumber", 
			Internal`NonNegativeIntegerQ,
		_, 
			NumericQ
	];
	valueType = Switch[ prop,
		$integerQueryRules, 
			"Integer",
		$doubleQueryRules, 
			"Double"
	];
	intQuery = valueType === "Integer";
	queryType = valueType <> "SetQuery";
	values = Switch[prop,
		"AtomChirality",
			atnums /. chiralTypeLookup,
		"MassNumber",
			atnums /. None -> 0,
		"OrbitalHybridization",
			atnums /. hybridizationLookup ,
		_,
			atnums
	];
	values = Replace[
		values, 
		{
			el_?test :> {el},
			el:Verbatim[Alternatives][nums__?test] /; intQuery :> {nums},
			el:Verbatim[Except][num_?test]  /; intQuery :> (negated = True;{num}),
			el:Verbatim[Except][Verbatim[Alternatives][nums__?test]]  /; intQuery :> (negated = True;{nums}),
			If[
				prop === "AtomicNumber",
				el:$ElementsPattern :> AtomicNumber[el],
				Sequence @@ {}
			],
			el:Verbatim[GreaterThan][x_]:> (queryType = valueType <> "GreaterQuery"; x),
			el:Verbatim[GreaterEqualThan][x_]:> (queryType = valueType <> "GreaterEqualQuery"; x),
			el:Verbatim[LessThan][x_]:> (queryType = valueType <> "LessQuery"; x),
			el:Verbatim[LessEqualThan][x_]:> (queryType = valueType <> "LessEqualQuery"; x),
			el:Verbatim[Between][x_]:> (queryType = valueType <> "BetweenQuery"; x),
			_ :> throwOnBadQuery[]
		}
	];
	values = <|"Values" -> values|>;
	If[
		queryType === "DoubleSetQuery",
		throwOnBadQuery[]
	];
	If[
		MatchQ[queryType, "DoubleLessQuery" | "DoubleLessEqualQuery"],
		negated = True
	];
	values["Negated"] = TrueQ @ negated;
	values["Property"] = prop;
	values["QueryType"] = queryType;
	
	values
]


atomQueryRules[ Rule[ prop:$booleanQueryRules, val:(True | False)] ] := <|"Property" -> prop, "QueryType" -> "Boolean","Value" -> val |>


atomQueryRules[___] := throwOnBadQuery[]

$unknownBondPattern = Verbatim[Blank[]] | PatternSequence[]

makeQueryBond[ Bond[ atom1_Integer, atom2_Integer, $unknownBondPattern] ] := 
	<| "Atoms" ->{ atom1, atom2}, "BondType" -> 0 |>
	
makeQueryBond[ Bond[ {atom1_Integer, atom2_Integer}, $unknownBondPattern] ] := 
	<| "Atoms" ->{ atom1, atom2}, "BondType" -> 0 |>
	
makeQueryBond[ Bond[ {atom1_Integer, atom2_Integer}, type:$BondTypes ] ] := 
	<| "Atoms" ->{ atom1, atom2}, "BondType" -> bondOrderLookup[type] |>


makeQueryBond[___] := throwOnBadQuery[]


(* ::Section::Closed:: *)
(*MaximumCommonSubstructure*)


(*{"findMCS",
            {                
                {iMolecule,1},
                "Boolean", (*maximizeBonds*)
                Integer, (* timeout *)
                "Boolean", (*matchValences*)
                "Boolean",(*ringMatchesRingOnly*)
                "Boolean",(*completeRingsOnly*)
                "Boolean"(*matchChiralTag*)
            },
            "Void"
        },*)
        
        
Options[MaximumCommonSubstructure] = {"MaximizeBonds" -> True, "MatchValences" -> True, 
	"RingMatchesRingOnly" -> True, "CompleteRingsOnly" -> True, "MatchChirality" -> False,
	"Timeout" -> 3600, IncludeHydrogens -> False}
MaximumCommonSubstructure[ mols__Molecule /; AllTrue[{mols},MoleculeQ], opts:OptionsPattern[]] := 
	Module[{nm = newInstance[], maximizeBonds, matchValences, ringMatchesRingOnly, 
		completeRingsOnly, matchChirality, timeout, return, returnSmiles, molIds},
		
		maximizeBonds = TrueQ @ OptionValue @ "MaximizeBonds";
		matchValences = TrueQ @ OptionValue @ "MatchValences";
		ringMatchesRingOnly = TrueQ @ OptionValue @ "RingMatchesRingOnly";
		completeRingsOnly = TrueQ @ OptionValue @ "CompleteRingsOnly";
		matchChirality = TrueQ @ OptionValue @ "MatchChirality";
		timeout = OptionValue@ "Timeout" /. x:Except[_Integer] :> 3600;
		molIds = ManagedLibraryExpressionID @ getCachedMol[ #, 
			Switch[ OptionValue[ IncludeHydrogens],
				True | All, "AllAtoms",
				_, "NoHydrogens"
			]
		] & /@ {mols};
		
		
		return = nm[ "findMCS", molIds, maximizeBonds, timeout, matchValences, 
			ringMatchesRingOnly, completeRingsOnly, matchChirality];
	
		MoleculePattern[ return ] /; StringQ[return] && (StringLength[return]  > 0) 
		
	]
	
MaximumCommonSubstructure[___] := $Failed



(* ::Section:: *)
(*End package*)



End[] (* End Private Context *)


