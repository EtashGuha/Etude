

Begin["Chemistry`Private`CommonDump`"]



Unprotect["Chemistry`Common`*"];
ClearAll["Chemistry`Common`*"];


If[
	And[
		TrueQ["ReturnQuantities" /. ("DataOptions" /. SystemOptions["DataOptions"])],
		!TrueQ[Internal`$DisableQuantityUnits]
	]
	,
	quantity := quantity = Quantity;
	quantityArray := quantityArray = QuantityArray;
	quantityArrayQ := quantityArrayQ = QuantityArray`QuantityArrayQ
	,
	quantity[arg_,unit___] := arg;
	quantityArray[arg_,unit___] := arg;
	quantityArrayQ = Identity
];



makeEntityDownValue[symbol_] := SetDelayed[
	symbol[args:HoldPattern[PatternSequence[bef___,e:Entity["Chemical",_],aft___]]],
	With[
		{m = Molecule[e]},
		symbol[bef,m,aft] /; MoleculeQ[m]
	]
]

makeEntityDownValue[{symbol_}] := SetDelayed[
	symbol[{args:HoldPattern[PatternSequence[bef___,e:Entity["Chemical",_],aft___]]},opts___],
	With[
		{m = Molecule[e]},
		symbol[{bef,m,aft}, opts] /; MoleculeQ[m]
	]
]


getMolOption[ mol_Molecule, option_] := OptionValue[ Molecule, Options[mol], option]

(*
	called when a function is expecting `mol` to 
	be MoleculeQ and it isn't.
*)
messageNoMol[sym_,arg_] := Message[
	MessageName[sym, "mol"],
	arg
]


(*echo[x___]:=Echo[ #, x] &*)
echo[___] = Identity
(* 	:associationInvert: *)

associationInvert[assoc_Association] := AssociationThread[Values @ assoc, Keys @ assoc];

(* ************************************************************************* **

       Functions that go from an integer to a value are XXXMap
       Functions that go from a value to an integer are XXXLookup
       It is not always the case that both are needed.

** ************************************************************************* *)



numberElementMap = <|0->"R",1->"H",2->"He",3->"Li",4->"Be",5->"B",6->"C",7->"N",8->"O",9->"F",10->"Ne",
11->"Na",12->"Mg",13->"Al",14->"Si",15->"P",16->"S",17->"Cl",18->"Ar",19->"K",20->"Ca",
21->"Sc",22->"Ti",23->"V",24->"Cr",25->"Mn",26->"Fe",27->"Co",28->"Ni",29->"Cu",30->"Zn",
31->"Ga",32->"Ge",33->"As",34->"Se",35->"Br",36->"Kr",37->"Rb",38->"Sr",39->"Y",40->"Zr",
41->"Nb",42->"Mo",43->"Tc",44->"Ru",45->"Rh",46->"Pd",47->"Ag",48->"Cd",49->"In",50->"Sn",
51->"Sb",52->"Te",53->"I",54->"Xe",55->"Cs",56->"Ba",57->"La",58->"Ce",59->"Pr",60->"Nd",
61->"Pm",62->"Sm",63->"Eu",64->"Gd",65->"Tb",66->"Dy",67->"Ho",68->"Er",69->"Tm",70->"Yb",
71->"Lu",72->"Hf",73->"Ta",74->"W",75->"Re",76->"Os",77->"Ir",78->"Pt",79->"Au",80->"Hg",
81->"Tl",82->"Pb",83->"Bi",84->"Po",85->"At",86->"Rn",87->"Fr",88->"Ra",89->"Ac",90->"Th",
91->"Pa",92->"U",93->"Np",94->"Pu",95->"Am",96->"Cm",97->"Bk",98->"Cf",99->"Es",100->"Fm",
101->"Md",102->"No",103->"Lr",104->"Rf",105->"Db",106->"Sg",107->"Bh",108->"Hs",109->"Mt",
110->"Ds",111->"Rg",112->"Cn",113->"Nh",114->"Fl",115->"Mc",116->"Lv",117->"Ts",118->"Og",119->"D"|> ;
elementNumberMap := elemnetNumberMap = associationInvert[numberElementMap];

numberNameMap = <|
	1 -> "Hydrogen", 2 -> "Helium", 3 -> "Lithium", 4 -> "Beryllium", 5 -> "Boron", 6 -> "Carbon",
	7 -> "Nitrogen", 8 -> "Oxygen", 9 -> "Fluorine", 10 -> "Neon", 11 -> "Sodium", 12 -> "Magnesium",
	13 -> "Aluminum", 14 -> "Silicon", 15 -> "Phosphorus", 16 -> "Sulfur", 17 -> "Chlorine",
	18 -> "Argon", 19 -> "Potassium", 20 -> "Calcium", 21 -> "Scandium", 22 -> "Titanium",
	23 -> "Vanadium", 24 -> "Chromium", 25 -> "Manganese", 26 -> "Iron", 27 -> "Cobalt",
	28 -> "Nickel", 29 -> "Copper", 30 -> "Zinc", 31 -> "Gallium", 32 -> "Germanium",
	33 -> "Arsenic", 34 -> "Selenium", 35 -> "Bromine", 36 -> "Krypton", 37 -> "Rubidium",
	38 -> "Strontium", 39 -> "Yttrium", 40 -> "Zirconium", 41 -> "Niobium", 42 -> "Molybdenum",
	43 -> "Technetium", 44 -> "Ruthenium", 45 -> "Rhodium", 46 -> "Palladium", 47 -> "Silver",
	48 -> "Cadmium", 49 -> "Indium", 50 -> "Tin", 51 -> "Antimony", 52 -> "Tellurium",
	53 -> "Iodine", 54 -> "Xenon", 55 -> "Cesium", 56 -> "Barium", 57 -> "Lanthanum",
	58 -> "Cerium", 59 -> "Praseodymium", 60 -> "Neodymium", 61 -> "Promethium", 62 -> "Samarium",
	63 -> "Europium", 64 -> "Gadolinium", 65 -> "Terbium", 66 -> "Dysprosium", 67 -> "Holmium",
	68 -> "Erbium", 69 -> "Thulium", 70 -> "Ytterbium", 71 -> "Lutetium", 72 -> "Hafnium",
	73 -> "Tantalum", 74 -> "Tungsten", 75 -> "Rhenium", 76 -> "Osmium", 77 -> "Iridium",
	78 -> "Platinum", 79 -> "Gold", 80 -> "Mercury", 81 -> "Thallium", 82 -> "Lead",
	83 -> "Bismuth", 84 -> "Polonium", 85 -> "Astatine", 86 -> "Radon", 87 -> "Francium",
	88 -> "Radium", 89 -> "Actinium", 90 -> "Thorium", 91 -> "Protactinium", 92 -> "Uranium",
	93 -> "Neptunium", 94 -> "Plutonium", 95 -> "Americium", 96 -> "Curium", 97 -> "Berkelium",
	98 -> "Californium", 99 -> "Einsteinium", 100 -> "Fermium", 101 -> "Mendelevium",
	102 -> "Nobelium", 103 -> "Lawrencium", 104 -> "Rutherfordium", 105 -> "Dubnium",
	106 -> "Seaborgium", 107 -> "Bohrium", 108 -> "Hassium", 109 -> "Meitnerium", 110 -> "Darmstadtium",
	111 -> "Roentgenium", 112 -> "Copernicium", 113 -> "Nihonium", 114 -> "Flerovium",
	115 -> "Moscovium", 116 -> "Livermorium", 117 -> "Tennessine", 118 -> "Oganesson"
|> ;
nameNumberMap := nameNumberMap = associationInvert[numberNameMap];
elementCommonName[element_] := ToLowerCase[ numberNameMap[AtomicNumber[element]]]


$ElementsPattern = Alternatives @@ (Values @ numberElementMap);
numberElementLookup := (
	Unprotect @ numberElementLookup;
	numberElementLookup = With[
		{ dict = associationInvert[ numberElementMap]},
		Lookup[dict, #]& 
	];
	Protect[ numberElementLookup];
	numberElementLookup
)


validAtomicNumber[x:{__Integer}] := AllTrue[ x, Between[{0,118}] ];
validAtomicNumber[x_Integer] := Between[x, {0,118}] ;
validAtomicNumber[___] := False




$AtomInputPattern = HoldPattern[Entity["Element" | "Isotope", _] | _String | _Atom | Rule[_, _Entity | _String | _Atom]]
$AtomPattern = Atom[ $ElementsPattern, OptionsPattern[] ];
$AtomsPattern = {Atom[ $ElementsPattern, Rule["Name"|"FormalCharge"|"MassNumber"|"UnpairedElectronCount"|"HydrogenCount",_]...]...}
$AtomReferencePattern = _Integer | _String
$AtomInMoleculePattern = { _Molecule, $AtomReferencePattern}


$EmptyMolecule = HoldPattern[ Molecule[{},{},___]]

$BondTypes = ("Single" | "Double" | "Triple" | "Aromatic" | "Unspecified")
$PatternBondHeads = UndirectedEdge | TwoWayRule 

$BondInputPattern = Alternatives[
	Bond[ 
		{_Integer | $AtomInputPattern, _Integer | $AtomInputPattern}, 
		_String | PatternSequence[] 
	], 
	{_Integer, _Integer, _String | PatternSequence[]}
]

$BondPattern = Bond[ {_Integer , _Integer}, $BondTypes ];




Module[
	{ dict = Join[associationInvert[ numberElementMap], associationInvert[ numberNameMap]]},
	AtomicNumber[element:(_String | {__String})] := Lookup[ dict, Capitalize @ element, Missing["NotAvailable"]];
	AtomicNumber[Atom[element_String,___]] := Lookup[ dict, element, Missing["NotAvailable"]];
	AtomicNumber[HoldPattern[Entity[ "Element", element_String]] ] := AtomicNumber[ element];
	AtomicNumber[iso:HoldPattern[Entity[ "Isotope", _String]] ] := iso["AtomicNumber"];
	AtomicNumber[ list_List] := AtomicNumber /@ list
]

AtomicNumber[arg_] := (Message[ AtomicNumber::nas, arg]; Null /; False)


FromAtomicNumber[element:(_Integer | {___Integer})] := Lookup[ numberElementMap, element, Missing["NotAvailable"]];
FromAtomicNumber[arg_] := (Message[ FromAtomicNumber::nint, arg]; Null /; False)

(* 
  //! the bond's direction (for chirality)
  typedef enum {
    NONE = 0,    //!< no special style
    BEGINWEDGE,  //!< wedged: narrow at begin
    BEGINDASH,   //!< dashed: narrow at begin
    // FIX: this may not really be adequate
    ENDDOWNRIGHT,  //!< for cis/trans
    ENDUPRIGHT,    //!<  ditto
    EITHERDOUBLE,  //!< a "crossed" double bond
    UNKNOWN,       //!< intentionally unspecified stereochemistry
  } BondDir; 
 
*)

bondDirMap = {
    0 -> "None",
    1 -> "BeginWedge",
    2 -> "BeginDash",
    3 -> "EndDownRight",
    4 -> "EndUpRight",
    5 -> "EitherDouble",
    6 -> "Unknonwn",
    _Integer -> "Other"  (* unrecognized hybridization *)
  };
  

(*
  //! store hybridization
  typedef enum {
    UNSPECIFIED = 0,  //!< hybridization that hasn't been specified
    S,
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    OTHER  //!< unrecognized hybridization
  } HybridizationType;
*)

hybridizationMap = <|
    0 -> "Unspecified",
    1 -> "S",
    2 -> Style["sp",Italic],
    3 -> Superscript[Style["sp",Italic],2],
    4 -> Superscript[Style["sp",Italic],3],
    5 -> Row[{Superscript[Style["sp",Italic],3], Style["d",Italic]}],
    6 -> Row[{Superscript[Style["sp",Italic],3],Superscript[Style["d",Italic],2]}],
    7 -> "Other"  (* unrecognized hybridization *)
  |>;
hybridizationLookup = <|Rule[
		Row[
			{
				Superscript[Style["sp", Italic], 3],
				Style["d", Rule[FontSlant, Italic]]
			}
		],
		5
	],
	Rule[
		Row[
			{
				Superscript[Style["sp", Italic], 3],
				Superscript[Style["d", Italic], 2]
			}
		],
		6
	],
	Style["sp", Rule[FontSlant, Italic]] -> 2,
	Superscript[Style["sp", Italic], 2] -> 3,
	Superscript[Style["sp", Italic], 3] -> 4,
	
	"S" -> 1, "s" ->1, "sp" -> 2, "SP" -> 2, "sp2" -> 2, "SP2" -> 3, "SP3" -> 4, "sp3" -> 4,
	"SP3D" -> 5, "sp3d" -> 5, "SP3D2" -> 6, "sp3d2" -> 6, "Unspecified" -> 0
|> ;



(*
	typedef enum {
		AtomicNumber = 1,
		OrbitalHybridization,
		FormalCharge,
		MassNumber,
		ExplicitHydrogens,
		ImplicitHydrogens,
		TotalHydrogens,
		ChiralTag,
		ImplicitValence,
		ExplicitValence,
		RadicalElectrons,
		CIPRank,
		Aromatic
	} AtomProperty ;

*)

atomPropertyMap = AssociationThread[Range@Length@# , #] &@ {
	"AtomicNumber", "OrbitalHybridization", "FormalCharge", "MassNumber", "ExplicitHydrogenCount",
	"ImplicitHydrogenCount", "HydrogenCount", "ChiralTag", "ImplicitValence", "ExplicitValence",
	"UnpairedElectronCount", "CIPRank", "Aromatic"
};

atomPropertyLookup = With[{ dict = associationInvert[ atomPropertyMap]},
	Lookup[dict, #]&  
];


bondOrderMap = <| 1 -> "Single", 2 -> "Double", 3 -> "Triple", 
				12 -> "Aromatic", 0 -> "Unspecified"|>;
bondOrderLookup = Function[
	Lookup[
		<|
			"Single" -> 1, "Double" -> 2, "Triple" -> 3, "Aromatic" -> 12, 
			"Amide" -> 1, "Any" -> 0, "Unspecified" -> 0
		|>,
		#
	]
] 


	 
(*
  typedef enum {     // stereochemistry of double bonds
    STEREONONE = 0,  // no special style
    STEREOANY,       // intentionally unspecified
    // -- Put any true specifications about this point so
    // that we can do comparisons like if(bond->getStereo()>Bond::STEREOANY)
    STEREOZ,     // Z double bond
    STEREOE,     // E double bond
    STEREOCIS,   // cis double bond
    STEREOTRANS  // trans double bond
  } BondStereo;
  
  *)
bondStereoMap = <|0 -> "None", 1 -> "Unspecified", 2 -> "Together", 3 -> "Opposite", 4 -> "StereoCis", 5 -> "StereoTrans" |>
bondStereoLookup = Lookup[
	<| 
		"None" -> 0, "Unspecified" -> 1, 
		"Together" -> 2, "Opposite" -> 3, 
		"Cis" -> 2, "Trans" ->3,
		"Z" -> 2, "E" -> 3
	|>,
	#
]&
$BondStereoTypes = "E" | "Z" | "Together" | "Opposite" | "Cis" | "Trans"


(*
  //! store type of chirality
  typedef enum {
    CHI_UNSPECIFIED = 0,  //!< chirality that hasn't been specified
    CHI_TETRAHEDRAL_CW,   //!< tetrahedral: clockwise rotation (SMILES \@\@)
    CHI_TETRAHEDRAL_CCW,  //!< tetrahedral: counter-clockwise rotation (SMILES
                          //\@)
    CHI_OTHER             //!< some unrecognized type of chirality
  } ChiralType;
*)

chiralTagMap = <| 0 -> "Unspecified", 1 -> "Clockwise", 2 -> "Counterclockwise", 3 -> "Other" , 4 -> "R", 5 -> "S"|>
chiralTagLookup = With[{ dict = associationInvert[ chiralTagMap]},
	Lookup[dict, #]& 
];
$AtomStereoTypes = "R" | "S" | "Clockwise" | "Counterclockwise";

chiralTypeMap = <| 0 -> None, 1 -> "Unspecified", 2 -> "R", 3 -> "S"|>
chiralTypeLookup = associationInvert[ chiralTypeMap];

(*
 *      0 Protein, L amino acids (default)
 *      1 Protein, D amino acids
 *      2 RNA, no cap
 *      3 RNA, 5' cap
 *      4 RNA, 3' cap
 *      5 RNA, both caps
 *      6 DNA, no cap
 *      7 DNA, 5' cap
 *      8 DNA, 3' cap
 *      9 DNA, both caps
 *)

 $SequenceTypes = <|
	"LAminoAcid" -> 0, "DAminoAcid" -> 1, "NoCapRNA" -> 2, "5'CapRNA" -> 3, "3'CapRNA" -> 4,
	"BothCapsRNA" -> 5, "NoCapDNA" -> 6, "5'CapDNA" -> 7, "3'CapDNA" -> 8, "BothCapsDNA" -> 9
|>; 
		


(* 	:checkMolecule: *)
(* returns a LibraryFunctionError if the Molecule pointer is null *)

checkMolecule[ mol_?ManagedLibraryExpressionQ] := mol[ "isValidLibraryMol"];
checkMolecule[ ___] := False;


(* 	:rdmessageFix: *)

fixLibraryMessage = 
	StringReplace[
		{
			"# " ~~ x : NumberString ~~ " " :> 
				"# " ~~ IntegerString[ToExpression@x + 1] ~~ " ",
				
			"Can't kekulize mol.  Unkekulized atoms: " ~~ st__ :> 
				"Unable to Kekulize atoms " ~~ StringReplace[st, x : NumberString :> IntegerString[ToExpression@x + 1] ],
				
			"non-ring atom "~~x:NumberString~~" marked aromatic" :> 
				"Non-ring atom "~~IntegerString[FromDigits[x]+1]~~" marked as aromatic.",
				
			"Bond already exists." :> 
				"Duplicate bond detected."
		}
	];

addCompletion = 
 With[{cr = #}, FE`Evaluate[FEPrivate`AddSpecialArgCompletion[cr]]] &;

 
 
atomNameMap[mol_Molecule ] := Module[
	{imol = getCachedMol @ mol, names},
	(
		names = imol["getNames"];
		(
			ReplaceAll[ KeyValueMap[#2 -> FromDigits[#1] &] @ names ]
		) /; AssociationQ[names]
	
	) /; imol =!= Null
]

atomNameMap[___] := Identity


messageOnBadAtomReference[ vertices_, input_, atomsToValidate_, tag_] /; 
	!ContainsAll[ vertices, Flatten[ atomsToValidate]] :=
	Replace[
		Thread[ Flatten /@ {input, atomsToValidate} ],
		{inp_, atom_} /; !MemberQ[ vertices, atom] :> 
			(Message[ Molecule::atom, inp]; Throw[ $Failed, tag]),
		1
	]
	
hasDuplicates[list_] := Length[list] > Length[Union@list]

messageOnBadAtomReference[ vertices_, input_, atomsToValidate_, tag_] /;
	hasDuplicates[atomsToValidate] := (Message[ Molecule::atom, input]; Throw[ $Failed, tag])
	

messageOnNotEmbedded[ mol_Molecule, imol_iMolecule, tag_] := If[
	!has3DCoordinates[imol] &&
		!has3DCoordinates[mol] && 
		!generateDefault3DCoordinates[ imol, getMolOption[mol, AtomCoordinates]],
	Return[ Missing["NotAvailable"], Module];
	,
	If[
		!has3DCoordinates[imol],
		imol["addCoordinates",mol["AtomCoordinates",TargetUnits -> None]]
	];
]

messageOnAtomsDimension[ prop_, atoms_ /; Length[atoms]>0, atomsIn_, tag_] := Module[{dim},If[ 
	!MatchQ[ Dimensions @ atoms,
		dim = Switch[ prop,
			"InteratomicDistance" | "BondLength", {_,2},
			"BondAngle", {_,3},
			"TorsionAngle" | "OutOfPlaneAngle", {_,4},
			"CenterOfMass", {_,_}
		]
	] || AnyTrue[
		atoms,
		Not@*DuplicateFreeQ
	],
	Message[ MessageName[ tag, "blveclen"], First[atomsIn], Last @ dim];
	$Failed
]]



firstQuantity[HoldPattern[{x_Quantity,___}]] := x
firstQuantity[HoldPattern[x_?quantityArrayQ]] := Normal[ First @ x, QuantityArray]
firstQuantity[x_] := x


ihasImplicitHydrogens[mol_] := TrueQ @ (getCachedMol[mol]["hasImplicitHydrogens"])
hasImplicitHydrogens[mol_] := cachedEvaluation[ihasImplicitHydrogens, mol]

(* 	:has2DCoordinates: *)

has2DCoordinates[imol_iMolecule] := Quiet[ Internal`NonNegativeIntegerQ @ imol[ "get2DConformerIndex"] ] ;
has2DCoordinates[mol_Molecule] := MatrixQ[Lookup[ Options[ mol], AtomDiagramCoordinates] ] 
has2DCoordinates[___] := False

(* 	:has3DCoordinates: *)

has3DCoordinates[imol_iMolecule] := Quiet[ Internal`NonNegativeIntegerQ @ imol[ "get3DConformerIndex"] ];
has3DCoordinates[mol_Molecule] := MatrixQ[Lookup[ Options[ mol], AtomCoordinates] ] 
has3DCoordinates[___] := False



(* ************************************************************************* **

       System`BondQ
       
       
** ************************************************************************* *)

BondQ[args___] := Block[{argCheck = System`Private`Arguments[ BondQ[args], 2 ], res},
        iBondQ[args] /; argCheck =!= {}
]

iBondQ[ mol_?MoleculeQ, x:{_,_} /; MemberQ[x, _String] ] := Block[{res},
	BondQ[ mol, res ] /; FreeQ[ res = atomNameMap[mol][x], _String]
]

iBondQ[ mol_?MoleculeQ, x:Bond[_,___] /; MemberQ[x[[1]], _String] ] := Block[{res},
	res = atomNameMap[mol][x];
	BondQ[ mol, res ] /; FreeQ[res[[1]] , _String]
]

iBondQ[ mol_?MoleculeQ, {a1_Integer, a2_Integer}] := EdgeQ[
	getMolGraph[ mol, "AllAtoms"],
	UndirectedEdge[ a1, a2]
]

iBondQ[ mol_?MoleculeQ, Bond[{a1_Integer, a2_Integer}] ] := EdgeQ[
	getMolGraph[ mol, "AllAtoms"],
	UndirectedEdge[ a1, a2]
]

iBondQ[ mol_?MoleculeQ, Bond[{a1_Integer, a2_Integer}, type_String] ] := Block[
	{graph = getMolGraph[ mol, "AllAtoms"], edge, index},
	edge = UndirectedEdge[ a1, a2];
	(
		index = EdgeIndex[ graph, edge];
		(* there is a bond between a1 and a2, check it's type first against
			the bonds in the expression, then against kekulized versions thereof *)
		SameQ[
			type,
			Part[ BondList[mol], index, 2]
		]
	) /; EdgeQ[ graph, edge ]
]

makeEntityDownValue[iBondQ]

iBondQ[ arg1_ /; !MoleculeQ[arg1], ___] := (messageNoMol[BondQ,arg1]; Null /; False)
iBondQ[___] := False
SmilesToCanonicalSmiles[smi_String] := Module[
	{im = newInstance[], res},
	res /; StringQ[ res = im["smilesToCanonicalSmiles", smi] ]
]

SmilesToCanonicalSmiles[smi_] := smi

SmilesToCanonicalSmiles[___] := $Failed



End[] (* End Private Context *)
