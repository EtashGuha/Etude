


Begin["Chemistry`Private`MoleculeFileParseDump`"]


Options[ReadMoleculeFile] = {"ReturnMetaInformation" -> False,"Indices" -> All, IncludeMetaInformation -> True, "ReturnCount" -> False}

ReadMoleculeFile[args___] := Module[
	{argCheck = System`Private`Arguments[ ReadMoleculeFile[args], {1,2} ]},
	Catch[
			iReadMoleculeFile["File", Sequence @@ argCheck],
			$tag
	] /; argCheck =!= {}
]

ReadMoleculeFileString[args___] := Module[
	{argCheck = System`Private`Arguments[ ReadMoleculeFileString[args], 2 ]},
	Catch[
			iReadMoleculeFile["String", Sequence @@ argCheck],
			$tag
	] /; argCheck =!= {}
]


iReadMoleculeFile[ "File", {file:File[path_String], other___}, opts_] := iReadMoleculeFile["File", {path, other}, opts]

iReadMoleculeFile["File", {input_String, fileType_String:Automatic}, opts_] := Module[
	{res,absoluteFileName,type},
	absoluteFileName = FindFile[ input];
	If[ 
		absoluteFileName === $Failed || 
			!FileExistsQ[absoluteFileName],
		Message[ReadMoleculeFile::nffil, Import];
		Throw[{$Failed,$Failed}, $tag]
	];
	type = Replace[
		fileType,
		Automatic :> getFileType[absoluteFileName]
	];
	res = iReadMoleculeFile[{"File",absoluteFileName}, type, opts]
]

iReadMoleculeFile["String", {input_String, fileType_String}, opts_] := Module[
	{type},
	type = Replace[
		ToLowerCase @ fileType, 
		Except[ "mol" | "sdf" | "mol2"] :> Throw[ $Failed, $tag]
	];
	iReadMoleculeFile[{"String",input}, type, opts]
]

getFileType[file_] := Replace[
	ToLowerCase @ FileExtension[ file], 
	Except[ "mol" | "sdf" | "mol2"] :> Throw[ $Failed, $tag]
]




iReadMoleculeFile[ {type_, input_}, "sdf", opts:OptionsPattern[]] := Module[
	{nm = newInstance[], nstructs,  includeMeta, res, libOpts, parts, evaluatedParts},
	
	libOpts = If[
		type === "File",
		<|"File" -> True|>,
		Null
	];
	nstructs = Replace[
		nm["initSDFileParser", input, libOpts],
		Except[_Integer] :> Throw[{$Failed,$Failed}, $tag]
	];
	If[
		TrueQ @ OptionValue[ ReadMoleculeFile, {opts}, "ReturnCount"]
		,
		If[
			(* if the file has only one record, check if it is moleculeQ before returning the count *)
			nstructs > 0 && (nstructs > 1 || MoleculeQ[ First @ getMoleculeFromSDF[ nm, 1,False]])
			,
			Throw[ nstructs, $tag]
			,
			Throw[$Failed, $tag]
		]
	];
	
	parts = Lookup[ Flatten[{opts}], "Indices", All];
	evaluatedParts = Replace[
		System`ConvertersDump`Utilities`ReplaceSubelement["SDF", parts, nstructs, "ForceLists"-> True],
		Except[{__Integer}] :> Throw[$Failed, $tag]
	];
	
	includeMeta = OptionValue[ ReadMoleculeFile, Flatten @ {opts}, IncludeMetaInformation];
	
	res = Table[
		Catch[ getMoleculeFromSDF[ nm, n, includeMeta], $tag]
		,{n, evaluatedParts}
	];
	nm["clearSDFileParser"];
	res
]

getMoleculeFromSDF[ supplier_iMolecule, n_Integer, includeMeta_] := Module[
	{im, pickle, molOptions, coords, meta, mol, nonHeaderKeys},
	
	pickle = supplier[ "getMolFromSDFile", n];
	
	If[ !ByteArrayQ[ pickle], Return[{$Failed,$Failed}, Module]];
	
	im = newInstance[];
	im["loadPickle", pickle];
	If[ im["atomCount", True] < 1, Return[{$Failed,$Failed}, Module]];
	
	molOptions = getCoordinatesFromMol[ im];
	
	meta = supplier["getSDFileProperties"];
	nonHeaderKeys = DeleteCases[ Keys @ meta, "Header"];
	meta = MapAt[
		postProcessMetaData,
		meta,
		Thread @ {nonHeaderKeys}
	];
	If[
		includeMeta && Length[meta] > 0,
		AppendTo[ molOptions, MetaInformation -> meta]
	];
	mol = Molecule[ 
		If[ 
			!TrueQ[im["validProperties"]]
			, 
			Sequence @@ molDataFromLibrary @ im
			,
			im
		], 
		Sequence @@ molOptions];
	{mol, meta}
	
]

postProcessMetaData[x_?StringQ] := Replace[
	System`Convert`MolDump`ToNumber /@ StringSplit[
		(*https://bugs.wolfram.com/show?number=361946*)
		StringReplace[x, "\n$$$$"->""],
		"\n"
	],
	{y_} :> y
]

postProcessMetaData[x_] := x

getCoordinatesFromMol[ im_iMolecule] := Module[
	{coords3D, coords2D, meta},
	coords3D = If[ 
		im["get3DConformerIndex"] >= 0 && MatrixQ[coords3D = im["get3DCoordinates"] ],
		AtomCoordinates -> quantityArray[ coords3D, "Angstroms"],
		Nothing
	];
	coords2D = If[ 
		im["get2DConformerIndex"] >= 0 && 
			MatrixQ[coords2D = im["get2DCoordinates"] ] && True
			(*DuplicateFreeQ[coords2D]*),
		AtomDiagramCoordinates -> coords2D,
		Nothing
	];
	
	{coords2D, coords3D}
]

iReadMoleculeFile[ {type_, input_}, "mol", opts:OptionsPattern[]] := Module[
	{im = newInstance[], metaInfo, libOpts, data, expr, graph, conf, molOptions, mol, includeMeta},
	libOpts = If[
		type === "String",
		<|"File" -> False|>,
		Null
	];
	metaInfo = Replace[
		Quiet[im[ "createMolFromMolFile", input, libOpts]],
		{
			Null :> <||>,
			_LibraryFunctionError :> Throw[ LegacyImportMol[{type,input},opts], $tag],
			x_Association :> KeyMap[StringReplace["_"->""],x]
		}
	];
	
	molOptions = getCoordinatesFromMol[ im];
	includeMeta = Lookup[ Flatten[{opts}], "ReturnMetaInformation", True];
	mol = Molecule[ Sequence @@ molDataFromLibrary @ im, Sequence @@ molOptions];
	If[
		includeMeta,
		{mol, metaInfo},
		mol
	] /; MoleculeQ[mol]
]

LegacyImportMol[{type_, input_}, opts:OptionsPattern[]] := Module[{data, output, stream, rules, mol},
	stream = If[
		type === "String",
		StringToStream[ input],
		OpenRead[input]
	];
	data = ReadList[stream, Record, RecordSeparators -> {}];
	If[Less[Length @ data, 1],
		Message[Import::fmterr, "MOL"];
		Return[{$Failed,$Failed},Module]
	];
	data = data[[1]];
	output = System`Convert`MolDump`ParseMoldata[data, "MOL"];
	If[SameQ[output, $Failed], Return[{$Failed,$Failed},Module]];
	rules = Join[ {"Units"->"Picometers"}, 
		MapThread[Rule,
			{
				{
					"VertexTypes", "EdgeRules", "FormalCharges", "VertexCoordinates", "EdgeTypes",
					"Header", "MassNumbers"
				},
				output
			}
		]
	];
	
	mol = Molecule[rules];
	If[!MoleculeQ[mol], Return[{$Failed,$Failed},Module]];
	name = output[[-2]];
	{mol, <| "Name" -> output[[-2]] |>}
] 


iReadMoleculeFile[ {type_, input_}, "mol2", opts:OptionsPattern[]] := Module[
	{im = newInstance[], res, fileInput, libOpts, data, expr, graph, conf, molOptions, mol},
	libOpts =  If[
		type === "String"
		,
		fileInput = If[
			StringEndsQ[input, "\n"],
			input,
			input<>"\n"
		];
		<|"File" -> False|>,
		fileInput = input;
		Null
	];
	res =  Quiet[
		Check[
			(*
				Hav found several mol2 files in the wild using upper case element symbols,
				but the RDKit fails on these with messages like
				"Element 'CL' not found."
				Rather than subject every input mol2 file to a string replacement, do it only
				when the library complains 
			*)
			im[ "createMolFromMol2File", fileInput, libOpts] /. "Missing" -> Missing["NotAvailable"]
			,
			im[ "createMolFromMol2File", fixMol2String @ fileInput, libOpts] /. "Missing" -> Missing["NotAvailable"];
			,
			{Molecule::interr}
		]
	];
	If[ 
		!MatchQ[ res, _LibraryFunctionError] && checkMolecule[im]
		,
		im["setStereoFrom3D"];
		molOptions = getCoordinatesFromMol[ im];
		mol = Molecule[ Sequence @@ molDataFromLibrary @ im, Sequence @@ molOptions];
		,
		Message[Import::fmterr, "MOL2"];
		Return[ $Failed ];
	];

	{ res, mol}
]

fixMol2String := fixMol2String = With[
	{
		elems = Select[
			Values @ numberElementMap,
			Function @ Greater[StringLength @ #, 1]
		]
	},
	{upperCase = Map[ToUpperCase, elems]},
	{rules = Thread[upperCase -> elems]},
	StringReplace[
		RuleDelayed[
			StringExpression["@<TRIPOS>ATOM", Shortest[atoms__], "@<TRIPOS>BOND"],
			StringJoin["@<TRIPOS>ATOM",
				StringReplace[atoms, rules],
				"@<TRIPOS>BOND"
			]
		]
	] 
] 




iReadMoleculeFile[___] := {$Failed,$Failed}


backupMol2Import[ {type_, file_}] := Module[
	{data, res},
	data = System`Convert`MOL2Dump`LegacyImportMol2Data[{type, file}];
	 
	
	res = Quiet[Molecule[ Normal @ data] ];
	
	If[
		MoleculeQ @ res,
		res = convertImplicitHydrogensToUnpariedElectrons[res];
		{data, res},
		{data, $Failed}
	]
];


backupMol2Import[___] := $Failed;

convertImplicitHydrogensToUnpariedElectrons[ mol_?MoleculeQ] /; has3DCoordinates[mol] := Module[
	{atomcount = AtomCount[ mol], hcounts, res,
		coordCount = Length @ Lookup[ Options[ mol], AtomCoordinates]},
	res = mol;	
	If[atomcount === coordCount, Return[ res, Module] ];
	If[
		coordCount === Length[First @ res],
		hcounts = MoleculeValue[ res,"ImplicitHydrogenCount", IncludeHydrogens -> False];
		
		Do[
			If[ 
				hcounts[[i]] > 0,
				AppendTo[ res[[ 1, i]], "UnpairedElectronCount" -> hcounts[[i]]]
			]
			,
			{i, Length @ hcounts}
		];
	];
	res
]

Options[WriteMoleculeFile] = Options[WriteMoleculeFileString] = {
	"CoordinateDimension" -> Automatic,
	IncludeHydrogens -> True,
	"Name" -> None,
	"Header" -> "Created with the Wolfram Language : www.wolfram.com",
	"Kekulized" -> True
}

WriteMoleculeFile[args___] := Module[
	{argCheck = System`Private`Arguments[ WriteMoleculeFile[args], {2,3} ]},
	If[ argCheck === {}, Return[$Failed, Module]];
	Catch[
			iWriteMoleculeFile["File", Sequence @@ argCheck],
			$tag
	]
]

WriteMoleculeFileString[args___] := Module[
	{argCheck = System`Private`Arguments[ WriteMoleculeFileString[args], 2 ]},
	If[ argCheck === {}, Return[$Failed, Module]];
	Catch[
			iWriteMoleculeFile["String", Sequence @@ argCheck],
			$tag
	] 
]




iWriteMoleculeFile["File", {input_, file_String, fileType_String:Automatic}, opts_] := Module[
	{res,absoluteFileName,type},
	absoluteFileName = ExpandFileName[ file];
	If[ 
		absoluteFileName === $Failed ,
		Throw[{$Failed,$Failed}, $tag]
	];
	type = Replace[
		fileType,
		Automatic :> getFileType[absoluteFileName]
	];
	res = iWriteMoleculeFile[{"File", input, absoluteFileName}, type, opts]
]

iWriteMoleculeFile["String", {input_Molecule, fileType_String}, opts_] := Module[
	{type},
	type = Replace[
		ToLowerCase @ fileType, 
		Except[ "mol" | "sdf" | "mol2"] :> Throw[ $Failed, $tag]
	];
	iWriteMoleculeFile[{"String",input, Null}, type, opts]
]


writeMolFileFromRules[mol_, fname_] := Module[
	{vals},
	vals = MoleculeValue[ 
		mol, 
		{"VertexTypes", "EdgeRules", "EdgeTypes", "AtomCoordinates"}
	];
	If[
		MatchQ[ 
			vals, 
			{ atoms:{__String}, rules:{__Rule}, types:{__String}, coords_?MatrixQ} /; 
				Length[types] === Length[rules] && Length[atoms] === Length[coords]
		]
		,
		System`Convert`MolDump`LegacyExportMOL[ 
			fname,
			Thread[ {"VertexTypes", "EdgeRules", "EdgeTypes", "VertexCoordinates"} -> vals]
		];
		fname
		,
		$Failed
	]
] 

hasExportCoordinates[mol_,imol_,coordOption_, data_] := hasExportCoordinates[
	mol, imol, Which[
		has2DCoordinates[mol],
			2,
		has3DCoordinates[mol],
			3,
		True,
			If[ imol["hasImplicitHydrogens"], 2, 3 ]
		],
	data
]

hasExportCoordinates[mol_, imol_, dim:(2|3), data_] := Module[
	{prop,coords},
	prop = If[ dim === 2, "coords2D", "coords3D"];
	data["Dimension"] = dim;
	
	coords = getiMolProperty[ imol, #1, getMolOption[mol,#2] ]& @@ Switch[ dim,
		2, {"coords2D",  AtomDiagramCoordinates},
		3, {"coords3D", AtomCoordinates}
	];
	
	If[
		MatrixQ[ coords]
		,
		True
		,
		If[
			dim === 3
			,
			coords = getiMolProperty[ imol, "coords2D", getMolOption[mol,AtomDiagramCoordinates] ];
			data["Dimension"] = True;
			MatrixQ[ coords]
			,
			False
		]
	]
]

(*
	"CoordinateDimension" -> 3,
	IncludeHydrogens -> True,
	"Name" -> None,
	"Header" -> "Created with the Wolfram Language : www.wolfram.com"*)
iWriteMoleculeFile[ {returnType_, mol_?MoleculeQ, fname_}, type:"mol", opts:OptionsPattern[]] := Module[
	{libOpts, string, imol, hasCoords, res, molDataSymbol},
	If[
		mol["HasValenceErrors"]
		,
		Return[ writeMolFileFromRules[mol, fname], Module]
	];
	libOpts = <| "Kekulized" -> TrueQ[ OptionValue[ WriteMoleculeFile, opts, "Kekulized"]] |>;
	If[
		returnType === "File",
		libOpts["File"] = True;
		libOpts["Type"] = type;
		libOpts["FileName"] = fname;,
		libOpts["File"] = False;
	];
	If[
		StringQ @ (string = OptionValue[ WriteMoleculeFile, opts, "Name"]),
		libOpts["Name"] = string
	];
	If[
		StringQ @ (string = OptionValue[ WriteMoleculeFile, opts, "Header"]),
		libOpts["Header"] =  string
	];
	
	imol = getCachedMol[mol, Flatten@{opts}];
	
	hasCoords = hasExportCoordinates[ mol, imol, 
		OptionValue[ WriteMoleculeFile, opts, "CoordinateDimension"],
		molDataSymbol
	];
	
	libOpts["Dimension"] = molDataSymbol["Dimension"];

	(
		res = imol["exportMolFile", libOpts];
		res
	) /; hasCoords
]


iWriteMoleculeFile[ {"File", mols:{__?MoleculeQ}, fname_}, type:"sdf", opts:OptionsPattern[]] := Module[
	{imols, meta, nm = newInstance[], headers, dim, hasCoords, prop, opt, molDataSymbol},
	
	meta = Lookup[Options[#], MetaInformation, <||>] & /@ mols;
	headers = Replace[
		OptionValue[ WriteMoleculeFile, Flatten@{opts}, "Header"],
		{
			x_String :> ConstantArray[x, Length @ mols],
			Except[x:{__String} /; Length[x] === Length @ mols] :> (
				Message[Export::uneqlen, "Molecule", "Header"];
				Return[$Failed, Module]
			)
		}
	];
	meta = MapThread[
		Append[#1, "Header" -> #2]&,
		{meta, headers}
	];
	
	dim = OptionValue[ WriteMoleculeFile, opts, "CoordinateDimension"];
	

	imols = getCachedMol[#(*, "AllAtoms"*)] & /@ mols;
	hasCoords = MapThread[
		hasExportCoordinates[ #1, #2, 
			OptionValue[ WriteMoleculeFile, opts, "CoordinateDimension"],
			molDataSymbol
		]&,
		{mols,imols}
	];
	
			
	hasCoords = And @@ hasCoords;
	(*MoleculeValue[ mols, "AtomCoordinates"];*)
	Quiet[
		Check[
			nm[ "writeSDFile", fname, ManagedLibraryExpressionID /@ imols, meta],
			Message[Export::fmterr, "SDF"];
			$Failed
		],
		Developer`WriteRawJSONString::jsonstrictencoding
	] /; hasCoords
	
]

iWriteMoleculeFile[___] := $Failed


(* ::Section::Closed:: *)
(*XYZMolecule*)

Options[XYZMolecule] = {"InferBondTypes" -> True, "TimeConstraint" -> 10, "Name" -> {}, "OldMethod" -> False}



XYZMolecule[ atoms_, coords_, comments_, opts:OptionsPattern[]] := Block[
	{$AdjustHydrogens = False, mol},
	mol = Catch[
		Quiet @ iXYZMolecule[ atoms, coords, comments, opts],
		$tag
	];
	fixOrdering[mol, coords] /; MoleculeQ[mol]
]


iXYZMolecule[atoms_, coords_, comments_, opts:OptionsPattern[]] := Module[
	{res, bonds, check, pattern, unmatchedBondIndices, name},
	check[] := If[
		!res["HasImplicitHydrogens"] && !res["HasValenceErrors"],
		Throw[ res, $tag]
	];
	
	(*
		The strategy here is as follows:
		
		1) try setting all bonds to single
		2) try finding a molecule name or SMILES string in 
		   the comments, grab the bonds from there
		3) follow a set of heuristics to determine bond order
		4) if failed, set all bonds to "Unspecified" and return
	*)
	bonds = Replace[
		inferBonds[ atoms, QuantityMagnitude[coords] * 100, 40, 25],
		HoldPattern[Rule[a_,b_]]:> Bond[{a,b},"Single"],
		{1}
	];
	res = Molecule[ atoms, bonds, AtomCoordinates -> coords];	
	
	check[];
	pattern = res /. "Single" -> "Unspecified";
	If[
		!OptionValue[XYZMolecule,{opts},"InferBondTypes"]
		,
		Return[pattern, Module]
	];
	name = Replace[
		OptionValue[XYZMolecule,{opts},"Name"],
		{
			x_?StringQ :> {x},
			_ :> {}
		}
	];
	Replace[
		findTemplate[pattern, Join[ comments, name] ],
		x_?MoleculeQ :> (res = applyTemplateBonds[ x, pattern])
	];
	check[];
	

	Replace[
		TimeConstrained[findFullBonds[ res],OptionValue[XYZMolecule,{opts},"TimeConstraint"]],
		x_?MoleculeQ :> (res = x)
	];
	check[];
	unmatchedBondIndices = BondList[res,
		Bond[ {Atom["ImplicitHydrogenCount" -> GreaterThan[0]], Atom[ "AtomicNumber" -> GreaterThan[1]]}],
		"BondIndex"
	];
	res[[2,unmatchedBondIndices,2]] = "Unspecified";
	
	Molecule @@ res
]


stringExplode[string_] := Module[{list},
	list = SequenceCases[StringSplit @ StringTrim @ string,
		token_ /; Less[Length @ token, 2],
		Overlaps -> All
	];
	DeleteDuplicates[Map[StringTrim @* StringRiffle, list]]
] 

findTemplate[ patt_, comments:{__String}] := Catch[ Module[
	{check, atomcount = Length @ First @ patt, strings, res},
	check[mol_] := If[
		AtomCount[mol] === atomcount && MoleculeContainsQ[mol, patt],
		Throw[mol, "template"]
	];
	Replace[
		Molecule[#,IncludeHydrogens -> True] & /@ comments,
		x_?MoleculeQ :> check[x],
		{1}
	];
	Do[
		strings = stringExplode[comment];
		Replace[
			strings,
			name_ /; And[
				StringQ[ NameToSMILES[name]],
				check[Molecule[name,IncludeHydrogens -> True]]
			]
		]
		,
		{comment,comments}
	]
	
],"template"]

applyTemplateBonds[ template_, mol_] :=  Module[
	{res,map,bonds},
	res = mol;
	map = Replace[
		FindMoleculeSubstructure[ template, mol],
		{
			{x_?AssociationQ} :> Values[x],
			_ :> Return[ $Failed, Module]
		}
	];
	bonds = BondList[
		MoleculeModify[ template, {"RenumberAtoms", map}]
	];
	res[[2]] = bonds;
	res
	
]

OrderingToTarget[list_, sourceIds_, targetIds_] := Part[
	Part[list, Ordering @ sourceIds],
	Ordering @ Ordering @ targetIds
] 

fixOrdering[mol_, coords_] := Module[{molCoords, ordering},
	
	molCoords = QuantityMagnitude[ Lookup[ Options[mol], AtomCoordinates]];
	If[ molCoords === coords, Return[mol, Module]];
	
	ordering = Range @ Length @ molCoords;
	ordering = OrderingToTarget[ ordering, molCoords, QuantityMagnitude @ coords];
	Return[ MoleculeModify[ mol, {"RenumberAtoms", ordering}] ]
]

findFullBonds[m_?ConnectedMoleculeQ] := Catch[
	Module[
		{im = newInstance[], rxnSmarts, data, coords},
		im["createCopy", ManagedLibraryExpressionID @ getCachedMol[m],<|"QuickCopy" -> False|>];
		
		rxnSmarts = {
			"[ND3:1]([OD1h1:2])[OD1h1:3]>>[N+:1](=[O:2])[O-:3]",
			"[C;D3:1]-[O;D1:2]>>[C:1]=[O:2]",
			"[N;D4;+0:1]>>[N+1:1]",
			"[O;D3;+0:1]>>[O+1:1]",
			"[S;D4:1](-[O;D1;h1:2])-[O;D1;h1:3]>>[S:1](=[O:2])=[O:3]",
			"[C;D3;h1:1]-[C;D2;h2:2]-[C;D3;h1:3]>>[C:1]=[C:2]=[C:3]",
			"[C;D3;h1:1]-[C;D2;h2:2]-[O;D1;h1:3]>>[C:1]=[C:2]=[O:3]",
			"[N;D2;h1:1]-[C;D2;h2:2]-[S;D1:3]>>[N:1]=[C:2]=[S:3]",
			"[C;D2;h2:1]-[N;D1;h2:2]>>[C:1]#[N:2]",
			"[h2:1]-[h2:2]>>[*:1]=[*:2]",
			"[h1:1]-[h1:2]>>[*:1]=[*:2]",
			"[h1:1]=[h1:2]>>[*:1]#[*:2]"
		};
		
		Check[
			Replace[
				im["standardizeWithRxnSmartsList", rxnSmarts],
				_LibraryFunctionError :> Throw[$Failed,"mol"]
			]
			,
			Throw[$Failed,"mol"]
		];
		
		data = molDataFromLibrary @ im;
		coords = im["get3DCoordinates"];
		
		If[
			MatchQ[data, {__List}] && MatrixQ[coords] &&  (Length[coords] === Length[First @ data] )
			,
			Molecule[ 
				Sequence @@ data, 
				AtomCoordinates -> quantityArray[coords, "Angstroms"]
			]
			,
			$Failed
		] 
		
	],"mol"]

inferBonds := (
	ImportExport`MoleculePlot3D;
	ClearAll[inferBonds];
	inferBonds[args___] := Graphics`MoleculePlotDump`InferBonds[args];
	inferBonds
)
	

SetAttributes[
    {
        ReadMoleculeFile
    },
    {ReadProtected, Protected}
]


End[] (* End Private Context *)

