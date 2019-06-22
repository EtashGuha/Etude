



Begin["Chemistry`Private`RDKitLinkDump`"]


Unprotect["Chemistry`RDKitLink`*"];
ClearAll["Chemistry`RDKitLink`*"];

Needs["JLink`"]


(* ************************************************************************* **

                        Find the libraries

** ************************************************************************* *)

With[
	{path = DirectoryName[$InputFileName]},
	$MolTemplateFile = FileNameJoin[ {path, "templates.mae"}];
	If[
		!FileExistsQ[$MolTemplateFile],
		$MolTemplateFile = "$Failed",
		$MolTemplateFile = DirectoryName[$MolTemplateFile]
	];
	Get @ FileNameJoin[ {path, "RDKLibraryFunctions.wl"}]
];


With[
	{parentDirectory = ParentDirectory @ DirectoryName[$InputFileName]},
	
	(* The location of the library directory is slightly different this is
	   in the layout vs if it's distributed as a paclet *)
	If[
		FileExistsQ @ FileNameJoin[ { parentDirectory, "LibraryResources"}],
		$ResourcePath = parentDirectory,
		$ResourcePath = ParentDirectory @ parentDirectory
	];
	If[
		!StringQ[$RDKitLibrary], (* because FileExistsQ is not gauranteed to return True or False *)
		$RDKitLibrary = FindLibrary["RDKitLink"];
	];
	
	
]

Quiet[LibraryUnload @ $RDKitLibrary];



(* ************************************************************************* **

                        loadChemLibraryFunction

** ************************************************************************* *)


loadChemLibraryFunction[class_String,{funName_String, args_List, return_, cSignature_, params_:{}}] := With[
	{
		symbol = classSymbols @ class,
		libargs = transformLibraryArguments @ Join[{Integer}, args],
		libFunName = StringJoin[class, "_", funName],
		libReturn = transformLibraryArguments @ return,
		postProcess = Switch[
			return,
			"RawJSON",readRawJSON,
			"ExpressionJSON", readExpressionJSON,
			_, Identity
		],
		preprocess = If[
			FreeQ[args, "RawJSON"],
			Sequence,
			Apply[
				Sequence,
				MapAt[
					Developer`WriteRawJSONString,
					(* When a library function takes a JSON object as input, the library will have
					   default values, so we can omit them. *)
					PadRight[{##},Length @ args, Null],
					Position[ args, "RawJSON"]
				]
			]&
		]
	},
	addLibrarySignature[funName, return, cSignature, class, params];
	
	If[
		$lazyLoading,
		symbol[idx_][funName, argumentsx___] := With[
			{fun = LibraryFunctionLoad[ $RDKitLibrary, libFunName, libargs, libReturn]},
			symbol[id_][ funName, arguments___] := postProcess @ fun[id, preprocess @ arguments];
			symbol[idx][funName, argumentsx]
		]
		,
		With[
			{fun = LibraryFunctionLoad[ $RDKitLibrary, libFunName, libargs, libReturn]},
			symbol[id_][funName, arguments___] := postProcess @ fun[id, preprocess @ arguments];
		]
	]
];

readRawJSON[x_String] := Developer`ReadRawJSONString[x];
readRawJSON[x_] := x

readExpressionJSON[x_String] := Developer`ReadRawJSONString[x];
readExpressionJSON[x_] := x

(* :classSymbols: *)

classSymbols = <| 
	"Molecule" -> iMolecule, 
	"RDKFingerprint" -> fingerprintFunction,
	"RDKForceField" -> forceFieldFunction,
	"MolPlotUtility" -> utility,
	"CacheManager" -> cacheManager
	|>
	

(* 	:transformLibraryArguments: *)

transformLibraryArguments = ReplaceAll[
	{
		"RawJSON" -> "UTF8String",
		"ExpressionJSON" -> "UTF8String",
		"iMolecule" -> Integer
	}
];
	
(* 	:$chemLibraryFunctions: *)

$chemLibraryFunctions = <| |>;


(* 	:addLibrarySignature: *)

addLibrarySignature[funName_,return_,cSignature_,class_, params_] := (
	$chemLibraryFunctions[funName] = <|
		"CSignature" -> cSignature,
		"Output" -> return,
		"Class" -> class
	|>;
	If[ 
		AssociationQ @ params, 
		$chemLibraryFunctions[funName, "Parameters"] = params
	];	
) 


(* 	:$lazyLoading: *)

$lazyLoading = False;

(* ************************************************************************* **

                        ManagedLibraryExpression constructors

** ************************************************************************* *)

newInstance[] := CreateManagedLibraryExpression["Molecule", iMolecule]
newInstance["RectangleBinPack"] := CreateManagedLibraryExpression["Molecule", rectangleBinPack]
newInstance["PlotUtility"] := CreateManagedLibraryExpression["MolPlotUtility", utility]
newInstance["CacheManager"] := CreateManagedLibraryExpression["CacheManager", cacheManager]
newInstance["Fingerprint"] := CreateManagedLibraryExpression["RDKFingerprint", fingerprintFunction]
newInstance["ForceField"] := CreateManagedLibraryExpression["RDKForceField", forceFieldFunction]


(* :iMoleculeList: *)


iMoleculeList = LibraryFunctionLoad[ $RDKitLibrary, 	"Molecule_get_collection", {}, {Integer, 1}]


(* ************************************************************************* **

                        Read in the list of library functions, 
                        write downvalues

** ************************************************************************* *)

Get[ 
	FileNameJoin[{
		DirectoryName @ $InputFileName,
		"RDKLibraryFunctions.wl"
	}]
];

loadChemLibraryFunction["Molecule", #]& /@ moleculeLibraryFunctions;
loadChemLibraryFunction["RDKFingerprint", #]& /@ rdFingerprintLibraryFunctions;
loadChemLibraryFunction["RDKForceField", #]& /@ rdForceFieldLibraryFunctions;
loadChemLibraryFunction["MolPlotUtility", #]& /@ molPlotUtilityFunctions;
loadChemLibraryFunction["CacheManager", #]& /@ cacheManagerFunctions;
deleteiMolecule := (
	Unprotect @ deleteiMolecule;
	deleteiMolecule = LibraryFunctionLoad[$RDKitLibrary, "deleteMoleculeInstance", {Integer}, Integer];
	Protect @ deleteiMolecule;
)




SetAttributes[
	{
		newInstance,
		iMolecule,
		iMoleculeList,
		deleteiMolecule,
		$chemLibraryFunctions,
		$MolTemplateFile
	},
    {ReadProtected, Protected}
]



End[] (* End Private Context *)
