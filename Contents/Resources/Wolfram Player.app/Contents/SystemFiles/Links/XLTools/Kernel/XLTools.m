(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["XLTools`"]

(* ::Section:: *)
(* Usage *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

XLImport::usage = "XLImport[fmt_, elem_][path_, opts___] Parse XL file as WL Table."
XLImportMetadata::usage = "XLImportMetadata[fmt_, elem_][path_, opts___] Parse XL file metadata. Supports sheet names and images."

(* ::Section:: *)
(* Load Dependencies *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Begin["`Private`"]

Get[FileNameJoin[{FileNameDrop[$InputFileName, -2], "LibraryResources", "LibraryLinkUtilities.wl"}]];

$packageFile = $InputFileName;
$libraryFileName = Switch[$OperatingSystem, "Windows", "XLTools.dll", "MacOSX", "XLTools.dylib", "Unix", "XLTools.so", _, $Failed];
$libXLFileName = Switch[$OperatingSystem, "Windows", "libxl.dll", "MacOSX", "libxl.dylib", "Unix", "libxl.so", _, $Failed];

$library = FileNameJoin[{FileNameTake[$packageFile, {1,-3}], "LibraryResources", $SystemID, $libraryFileName}];
$libxl = FileNameJoin[{FileNameTake[$packageFile, {1,-3}], "LibraryResources", $SystemID, $libXLFileName}];
$InitXLTools = False;

InitXLTools[] :=
If[TrueQ[$InitXLTools],
	$InitXLTools
	,
	$InitXLTools =
		Catch[
			Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
				SafeLibraryLoad[$libxl];
				SafeLibraryLoad[$library];

				RegisterPacletErrors[$library, <|
					"AdapterNotLoaded" -> "Error loading LibraryLink functions.",
					"ArchiveNotFound" -> "Could not find specified archive file. See exception Parameters for file path.",
					"FileNotFound" -> "One or more of the files could not be located. See exception Parameters for file path(s).",
					"UnsupportedFormat" -> "Unsupported call.",
					"ReadError" -> "Error reading data.",
					"FormatError" -> "`FileName` is not a `Format` file.",
					"InvalidOption" -> "Invalid option."
				|>];

				(* Import *)
				$clearCache = SafeLibraryFunction["clearCache", {}, "Void", "Throws" -> True];
				$openWorkbook = SafeLibraryFunction["openWorkbook", {"UTF8String", "UTF8String"}, "Void", "Throws" -> True];
				$closeWorkbook = SafeLibraryFunction["closeWorkbook", {"UTF8String"}, "Void", "Throws" -> True];
				$getImages = SafeLibraryFunction["getImages", LinkObject, LinkObject, "Throws" -> True];
				$getSheetCount = SafeLibraryFunction["getSheetCount", {"UTF8String"}, Integer, "Throws" -> True];
				$getSheetNames = SafeLibraryFunction["getSheetNames", LinkObject, LinkObject];
				$getBookDimensions = SafeLibraryFunction["getBookDimensions", {"UTF8String"}, {Integer, 1}, "Throws" -> True];
				$parseXLFile = SafeLibraryFunction["parseXLFile", LinkObject, LinkObject, "Throws" -> True];
			];
		True
	]
]

ValidateOption = System`ConvertersDump`Utilities`ValidateOption;
ProcessTableHeaders = System`ConvertersDump`Utilities`ProcessTableHeaders;
TryCatch = System`ConvertersDump`Utilities`TryCatch;

(* ::Section:: *)
(* Converter *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

(* ::Subsection:: *)
(* Import *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Options[XLImportMetadata] = {
	"CheckFormat" -> True
}

(* Unsupported format *)
XLImportMetadata[format_, ___][___] := CreatePacletFailure["UnsupportedFormat", "Parameters" -> format];

XLImportMetadata[
	format : "XLS" | "XLSX",
	elem : "Images" | "Sheets" | "SheetCount" | "Dimensions"
][inFile_?StringQ, opts : OptionsPattern[]] :=
Block[{init, res, pos, chkFmt, dateObjHead, sNames, numSheets},
	If[!TrueQ[init = InitXLTools[]],
		Return@init;
	];

	chkFmt = ValidateOption[True, XLImport, "CheckFormat", _?BooleanQ, opts];

	(* Check if input file exists *)
	If[chkFmt && !FileExistsQ[inFile],
		Return@CreatePacletFailure["FileNotFound", "Parameters" -> inFile];
	];

	If[chkFmt && !ImportExport`FileFormatQ[inFile, format],
		Return@CreatePacletFailure["FormatError", "MessageParameters" -> <|"FileName" -> FileNameTake[inFile, -1], "Format" -> format|>]
	];

	TryCatch[
		$openWorkbook[format, inFile];

		Switch[elem,
			"Images",
				res = $getImages[inFile];

				Which[
					res === {},
						Return[{}, Block];
					,
					!MatchQ[res, {__String}],
						Return@CreatePacletFailure["ReadError"];
				];

				(* $getImages returns a list of Strings, where 1, 3, 5 ... are the formats, and
					2, 4, 6 ... are the bytestrings. 1 is the format of 2 and so on.
				 *)
				res = Replace[ImportString[Last@#, First@#] & /@ Partition[res, 2], x_?(!ImageQ[#]&) :> $Failed, {1}];
			,
			"Sheets",
				res = $getSheetNames[inFile];
			,
			"SheetCount",
				res = $getSheetCount[inFile];
			,
			"Dimensions",
				sNames = $getSheetNames[inFile];

				res = Partition[$getBookDimensions[inFile], 2];

				numSheets = Length@res;

				If[!ListQ[sNames] || !ListQ[res],
					Throw@CreatePacletFailure["ReadError"];
				];

				If[Length@sNames == numSheets == 0,
					Return[{}];
				];

				res = 
					If[Positive[numSheets] && Length[DeleteDuplicates@sNames] == numSheets,
						AssociationThread[sNames, res]
						,
						CreatePacletFailure["ReadError"]
					];

				If[FailureQ[res],
					Throw@res;
				];
		];

		res
		,
		(* Finally *)
		$closeWorkbook[inFile];
	]
]

Options[XLImport] = {
	"CheckFormat" -> True,
	"DateObjectHead" -> "DateObject",
	"EmptyField" -> "",
	"HeaderLines" -> 0,
	"SkipLines" -> 0
}

XLImport[
	format : "XLS" | "XLSX",
	elem : "Formulas" | "Sheets" | "Data" | "FormattedData" | "Dataset",
	sheets_: All,
	rows_: All,
	cols_: All
][inFile_?StringQ, opts : OptionsPattern[]] :=
Block[
	{
		init, dims, res, pos, chkFmt, $sheets, $rows, $cols, dateObjHead, 
		sNames, numSheets, ranges, rowLen, colLen, hLines, sLines, null
	},

	If[!TrueQ[init = InitXLTools[]],
		Return@init;
	];

	chkFmt 		= ValidateOption[True, XLImport, "CheckFormat", _?BooleanQ, opts];
	dateObjHead = ValidateOption[True, XLImport, "DateObjectHead", _, opts];
	hLines 	= ValidateOption["Import", XLImport, "HeaderLines",
					Automatic | 
					x_?Internal`NonNegativeIntegerQ | 
					{y_?Internal`NonNegativeIntegerQ, z_?Internal`NonNegativeIntegerQ}
				, opts];
	sLines 	= ValidateOption["Import", XLImport, "SkipLines",
					Automatic | 
					x_?Internal`NonNegativeIntegerQ | 
					{y_?Internal`NonNegativeIntegerQ, z_?Internal`NonNegativeIntegerQ}
				, opts];
	null = Hold["EmptyField"] /. Flatten @ {opts} /. "EmptyField" -> "";

	(* Check if input file exists *)
	If[chkFmt && !FileExistsQ[inFile],
		Return@CreatePacletFailure["FileNotFound", "Parameters" -> inFile];
	];

	If[chkFmt && !ImportExport`FileFormatQ[inFile, format],
		Return@CreatePacletFailure["FormatError", "MessageParameters" -> <|"FileName" -> FileNameTake[inFile, -1], "Format" -> format|>]
	];

	TryCatch[
		$openWorkbook[format, inFile];

		sNames = $getSheetNames[inFile];

		dims = Partition[$getBookDimensions[inFile], 2];

		numSheets = Length@dims;

		If[!ListQ[sNames] || !ListQ[dims],
			Throw@CreatePacletFailure["ReadError"];
		];

		If[Length@sNames == numSheets == 0,
			Return[{}];
		];

		dims = 
			If[Positive[numSheets] && Length[DeleteDuplicates@sNames] == numSheets,
				AssociationThread[sNames, dims]
				,
				CreatePacletFailure["ReadError"]
			];

		If[FailureQ[dims],
			Throw@dims;
		];

		(* Get list of positive ints for sheet indices *)
		$sheets = 
			Switch[sheets,
				(_Span | _Integer | "All" | All),

				System`ConvertersDump`Utilities`ReplaceSubelement[format, sheets, numSheets, "ForceLists" -> True]
				
				,
				(_List | _String),

				(* String to List *)
				pos = Replace[sheets, x : Except[_List] :> {x}];

				(* Neg to Pos (keep Neg for messages) *)
				pos = 
				Thread[pos -> 
					Replace[
						pos
						, 
						{x_Integer?Negative :> x + numSheets + 1}
						, 
						{1}
					]
				];
				
				Replace[
					pos
					,
					{
						(key_ -> x : Except[_?IntegerQ]) :> 
							(* For non ints, get the position in the dims assoc *)
							Replace[
								Position[Keys@dims, x]
								,
								{ 
									{} :> (Message[Import::noelem, key, format]; $Failed), 
									{{val_}} :> val
								}
							],
						(* For ints, issue a message for out of range values *)
						(key_ -> x_Integer?(# <= 0 || # > numSheets &)) :> (Message[Import::noelem, key, format]; $Failed),
						(key_ -> x_Integer) :> x
					}
					,
					{1}
				]
			]; (* Switch[sheets.. *)

		If[MatchQ[$sheet, {$Failed ..}],
			Return[$sheet];
		];

		ranges = Thread[$sheets -> getRanges[dims, format, rows, cols] /@ $sheets];

		If[MatchQ[ranges, {($Failed -> $Failed) ..}],
			Return[
				If[MatchQ[sheets, _Integer | _String] && !MatchQ[sheets, "All"],
					First[Last /@ ranges, $Failed]
					,
					Last /@ ranges
				]
			];
		];

		res = readSheet[inFile, elem, dateObjHead, Boole[MatchQ[cols, "All" | All]], format] /@ ranges;

		If[MatchQ[sheets, _Integer | _String] && !MatchQ[sheets, "All"], 
			res = First[res, $Failed];
			If[MatchQ[rows, _Integer],
				res = First[res, $Failed];
				If[MatchQ[cols, _Integer],
					res = First[res, $Failed];
				];
				,
				(* Rows are List *)
				If[MatchQ[cols, _Integer],
					res = First[#, $Failed] & /@ res;
				];
			];
			,
			(* Sheets are List *)
			{rowLen, colLen} = Length /@ Last[FirstCase[ranges, Except[$Failed -> $Failed]]];

			If[MatchQ[rows, _Integer],
				res = First[#, $Failed] & /@ res;
				If[MatchQ[cols, _Integer],
					res = First[#, $Failed] & /@ res;
				];
				,
				(* Rows are List *)

				(* 
					For List Rows and All cols, C code will put $Failed for out of range access.
					For List Rows and other cols, C code will put {$Failed.. } with number of cols				
				 *)

				(* This should *probably* be in the C code. Very low priority since this is working. *)
				If[MatchQ[cols, _Integer],
					res = Replace[Replace[res, x_List :> Flatten[x], {1}], $Failed -> ConstantArray[$Failed, rowLen], {1}];
					,
					(* Cols are List *)
					Switch[{rows, cols},
						{All, Except[All]},
						res = Replace[res, $Failed -> ConstantArray[$Failed, colLen], {1}];
						,
						{Except[All], All},
						res = Replace[res, $Failed -> ConstantArray[$Failed, rowLen], {1}];
						,
						{Except[All], Except[All]},
						res = Replace[res, $Failed -> ConstantArray[$Failed, {rowLen, colLen}], {1}];
					]
					
				];
			];
		];

		res = Developer`ToPackedArray[res /. Null :> ReleaseHold@null];

		If[MatchQ[sheets, _Integer | _String], 
			ProcessTableHeaders[res, sLines, hLines, "EmptyField" -> ReleaseHold@null, "Dataset" -> (elem === "Dataset")]
			,
			ProcessTableHeaders[#, sLines, hLines, "EmptyField" -> ReleaseHold@null, "Dataset" -> (elem === "Dataset")] & /@ res
		]
		,
		(* Finally *)
		$closeWorkbook[inFile];
	]
]

(* ::Subsection:: *)
(* Utility Functions *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

getRanges[___][fail_?FailureQ] := fail
getRanges[dims_, fmt_, rows_, cols_][iSheet_] :=
Block[{$rowDims, $colDims},
	{$rowDims, $colDims} = Replace[dims[[iSheet]], {0, 0} -> {1, 0}];

	{
		System`ConvertersDump`Utilities`ReplaceSubelement[fmt, rows, $rowDims, 0, "ForceLists" -> True],
		System`ConvertersDump`Utilities`ReplaceSubelement[fmt, cols, $colDims, 0, "ForceLists" -> True]
	}
]

readSheet[___][Rule[$Failed, $Failed]] := $Failed
readSheet[___][Rule[_, {{1}, {}}]] := {{}}
readSheet[filePath_, elem_, dateObjHead_, allCols_, format_][Rule[sheet_, {rows_, cols_}]] := 
Block[{res = $parseXLFile[filePath, elem, dateObjHead, allCols, sheet, rows, cols]},
	If[FailureQ[res],
		Message[Import::fmterr, format];
		Throw@res;
		,
		res
	]
]

End[]
EndPackage[]
