


Begin["Chemistry`Private`SymmetrizerLinkDump`"]



Needs["JLink`"]



Options[SymmetryInformation] = {Tolerance -> 0.1}




(* ::Section::Closed:: *)
(*SymmetryInformation*)


SymmetryInformation[m_?MoleculeQ, atoms_:All, opts:OptionsPattern[]] := Module[
	{out,temp,strm,shortContext},
	
	Internal`WithLocalSettings[
		shortContext = OptionValue[LoadJavaClass, AllowShortContext];
		SetOptions[LoadJavaClass, {AllowShortContext -> False}];
		BeginJavaBlock[];
		out = getSystemOut[];
		temp = CreateTemporary[];
		strm = JavaNew["java.io.PrintStream", temp];
		java`lang`System`setOut[strm];
		,
		iSymmetryInformation[m, atoms, opts]
		,
		java`lang`System`setOut[out];
		strm @ close[];
		EndJavaBlock[];
		SetOptions[LoadJavaClass, {AllowShortContext -> shortContext}];
		safeDeleteFile[temp];
	]
		
]


iSymmetryInformation[m_, indices_, opts:OptionsPattern[] ] := Module[
   	{mol, natoms, coords, atomArray, pointGroupList, symInfo, mainPointGroup, res, symmetryElements, atoms, pointGroupName, tolerance},
   	
	natoms = Replace[ indices,
		{
			All :>  AtomCount @ m,
			{__Integer} :> Length @ indices
		}
	];
	atomArray = ConstantArray[0, {natoms, 5}];
	atomArray[[All, 1]] = m[{"AtomicNumber",indices}];
	atomArray[[All,2;;4]] = QuantityMagnitude @ m[{"AtomCoordinates",indices}];
	atomArray[[All,5]] = QuantityMagnitude @ m[{"AtomicMass",indices}];
	If[
		!MatchQ[atomArray, {{_Integer, __Real}..}]
		,
		Return[Missing["NotAvailable"], Module]
	];
			
	
	mol = JavaNew["net.webmo.symmetry.molecule.Molecule"];
	
	
	atoms = JavaNew["net.webmo.symmetry.molecule.Atom", ##] & @@@ atomArray;
	mol @ addAtom[#] & /@ atoms;
	
	tolerance = OptionValue[ SymmetryInformation, {opts}, Tolerance];
	
	symInfo = JavaNew["net.webmo.symmetry.Symmetry", mol, tolerance];
	pointGroupList = JavaObjectToExpression @ symInfo @ findAllPointGroups[];
	
	
	mainPointGroup = If[ 
		0 === Length @ pointGroupList
		,
		Return[ Missing @ "NotAvailable", Module]
		,
		First @ pointGroupList
	];
	pointGroupName = StringReplace[  
		mainPointGroup @ getName[],
		{
			"D_infinity_d" -> "D\[Infinity]h",
			"C_infinity_v" -> "C\[Infinity]v"
		}
	];
	res = <|
		"PointGroupDisplay" -> pointGroupMap[pointGroupName],
		"PointGroupString" -> pointGroupName
	|>;
	AssociateTo[res, "SymmetryEquivalentAtoms" -> equivalentAtomGroups[m] ];
	
	symmetryElements = mainPointGroup @ getElements[];
	AssociateTo[res, "SymmetryElements" -> Map[getSymmetryInfo] @ JavaObjectToExpression @ symmetryElements];
	res
]


getSystemOut[] := (
	LoadJavaClass["java.lang.System", AllowShortContext -> False];
	java`lang`System`out
);

safeDeleteFile[file_String?FileExistsQ] := DeleteFile[file];


getSymmetryInfo[element_] := getSymmetryInfo[ element, ClassName @ element];

getSymmetryInfo[element_ , "net.webmo.symmetry.elements.Reflection"] := Module[ 
	{res = <|"Operation" -> "Reflection"|>, pt, nrml},
	
	res["Name"] = symOperationName @ element @ getName[];
	res["Degree"] = element @ getDegree[];
	res["UniqueOperationsCount"] = element @ getNumUniqueOperations[];
	
	pt = fromPoint3D @ element @ getPoint[];
	nrml = fromPoint3D @ element @ getNormal[];
	
	res["SymmetryPlane"] = N @ Chop @ Hyperplane[nrml, pt];
	
	res
]
	
getSymmetryInfo[element_ ,"net.webmo.symmetry.elements.ProperRotation"] := Module[ 
	{res = <|"Operation" -> "Rotation"|>, pt, axis},
	
	res["Name"] = symOperationName @ element @ getName[];
	res["Degree"] = element @ getDegree[];
	res["UniqueOperationsCount"] = element @ getNumUniqueOperations[];
	
	pt = fromPoint3D @ element @ getPoint[];
	axis = fromPoint3D @ element @ getAxis[];
	
	res["RotationAxis"] = N @ Chop @ InfiniteLine[pt, axis];
	
	res
]

getSymmetryInfo[element_ , "net.webmo.symmetry.elements.Inversion"] := Module[ 
	{res = <|"Operation" -> "Inversion"|>, pt},
	res["Name"] = symOperationName @ element @ getName[];
	res["Degree"] = element @ getDegree[];
	res["UniqueOperationsCount"] = element @ getNumUniqueOperations[];
	
	pt = fromPoint3D @ element @ getPosition[];
	
	res["InversionCenter"] = N @ Chop @ Point[pt];
	
	res
]

	
getSymmetryInfo[element_ ,"net.webmo.symmetry.elements.ImproperRotation"] := Module[ 
	{res = <|"Operation" -> "ImproperRotation"|>, pt, axis},
	
	res["Name"] = symOperationName @ element @ getName[];
	res["Degree"] = element @ getDegree[];
	res["UniqueOperationsCount"] = element @ getNumUniqueOperations[];
	
	pt = fromPoint3D @ element @ getPoint[];
	axis = fromPoint3D @ element @ getAxis[];
	
	res["RotationAxis"] = N @ Chop @ InfiniteLine[pt, axis];
	
	res
]

equivalentAtomGroups[mol_ /; MoleculeQ[mol] ] := Module[ 
	{ranks, imol,indices},

	imol = getCachedMol[ mol];
	ranks = imol[ "rankAtoms", <|"BreakTies" -> False|> ];
		
	indices = mol["AtomIndex"];
	
	ranks = Thread[ {indices, ranks}];
	ranks = GatherBy[ ranks, Last];
	
	ranks[[All, All, 1]]	
]

With[{sub = Subscript[##]&, italic = Style[ #, Italic]&},
	pointGroupMap = <|
		"C1" -> sub[italic["C"], "1"],
		"Cs" -> sub[italic["C"], italic @ "s"],
		"Ci" -> sub[italic["C"], italic @ "i"],
		"C2" -> sub[italic["C"], "2"],
		"C3" -> sub[italic["C"], "3"],
		"C4" -> sub[italic["C"], "4"],
		"C5" -> sub[italic["C"], "5"],
		"C6" -> sub[italic["C"], "6"],
		"C7" -> sub[italic["C"], "7"],
		"C8" -> sub[italic["C"], "8"],
		"D2" -> sub[italic["D"], "2"],
		"D3" -> sub[italic["D"], "3"],
		"D4" -> sub[italic["D"], "4"],
		"D5" -> sub[italic["D"], "5"],
		"D6" -> sub[italic["D"], "6"],
		"C2v" -> sub[italic["C"], Row @ {"2", italic @ "v"}],
		"C3v" -> sub[italic["C"], Row @ {"3", italic @ "v"}],
		"C4v" -> sub[italic["C"], Row @ {"4", italic @ "v"}],
		"C5v" -> sub[italic["C"], Row @ {"5", italic @ "v"}],
		"C6v" -> sub[italic["C"], Row @ {"6", italic @ "v"}],
		"D2d" -> sub[italic["D"], Row @ {"2", italic @ "d"}],
		"D3d" -> sub[italic["D"], Row @ {"3", italic @ "d"}],
		"D4d" -> sub[italic["D"], Row @ {"4", italic @ "d"}],
		"D5d" -> sub[italic["D"], Row @ {"5", italic @ "d"}],
		"D6d" -> sub[italic["D"], Row @ {"6", italic @ "d"}],
		"C2h" -> sub[italic["C"], Row @ {"2", italic @ "h"}],
		"C3h" -> sub[italic["C"], Row @ {"3", italic @ "h"}],
		"C4h" -> sub[italic["C"], Row @ {"4", italic @ "h"}],
		"C5h" -> sub[italic["C"], Row @ {"5", italic @ "h"}],
		"C6h" -> sub[italic["C"], Row @ {"6", italic @ "h"}],
		"D2h" -> sub[italic["D"], Row @ {"2", italic @ "h"}],
		"D3h" -> sub[italic["D"], Row @ {"3", italic @ "h"}],
		"D4h" -> sub[italic["D"], Row @ {"4", italic @ "h"}],
		"D5h" -> sub[italic["D"], Row @ {"5", italic @ "h"}],
		"D6h" -> sub[italic["D"], Row @ {"6", italic @ "h"}],
		"S4" -> sub[italic["S"], "4"],
		"S6" -> sub[italic["S"], "6"],
		"S8" -> sub[italic["S"], "8"],
		"T" -> italic["T"],
		"Th" -> sub[italic["T"], italic @ "h"],
		"Td" -> sub[italic["T"], italic @ "d"],
		"O" -> italic["O"],
		"Oh" -> sub[italic["O"], italic @ "h"],
		"I" -> italic["I"],
		"Ih" -> sub[italic["I"], italic @ "h"],
		"C\[Infinity]v" -> sub[italic["C"], Row @ {"\[Infinity]", italic @ "v"}],
		"D\[Infinity]h" -> sub[italic["D"], Row @ {"\[Infinity]", italic @ "h"}]
	|> 
] ;

symOperationName = ReplaceAll @ {
	RuleDelayed[
		PatternTest[x_String,
			StringMatchQ[StringExpression["C" | "S", NumberString]]
		],
		Subscript[StringTake[x, 1], StringDrop[x, 1]]
	],
	"sigma" -> "\[Sigma]"
};

fromPoint3D = { # @ x, # @ y, # @ z} &;





End[] (* End Private Context *)


