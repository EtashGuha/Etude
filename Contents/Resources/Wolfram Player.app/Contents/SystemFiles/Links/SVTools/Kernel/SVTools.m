(* ::Package:: *)

(* Mathematica Package *)

(* Author:
	Sean Cheren
	scheren@wolfram.com
 *)

(* Mathematica Version: 11.2 *)

BeginPackage["SVTools`"]

(* ::Section:: *)
(* Usage *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

SVImport::usage = 
"SVImport[\"format\", \"filePath\"] parses file as WL Table. \"format\" can be \"CSV\" or \"TSV\".
SVImport[\"format\", \"InputStream\"] reads and parses data from InputStream as WL Table. \"format\" can be \"CSV\" or \"TSV\"."

SVExport::usage = 
"SVExport[\"format\", File[\"filePath\"], \"data\"] exports data to filePath in the CSV format. \"format\" can be \"CSV\" or \"TSV\".
SVExport[\"format\", \"filePath\", \"data\"] exports data to filePath in the CSV format. \"format\" can be \"CSV\" or \"TSV\".
SVExport[\"format\", \"OutputStream\", \"data\"] exports data to OutputStream in the CSV format. \"format\" can be \"CSV\" or \"TSV\"."

SVGetDimensions::usage = 
"SVGetDimensions[\"format\", \"filePath\"] get the dimensions of a file.
SVImport[\"format\", \"InputStream\"] get the dimensions of CSV/TSV data from a stream."

CreateSVToolsException::usage = "CreateSVToolsException[type_String, errorCode_Integer, message_String, param_List, cause_] creates a CreateSVToolsException."
SVToolsException::usage = "SVToolsException is an exception object returned by CreateSVToolsException. SVToolsException[field] queries for the following properties: \"Type\", \"ErrorCode\", \"Message\", \"Parameters\", \"Cause\", \"ErrorID\""


(* ::Section:: *)
(* Load Dependencies *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Begin["`Private`"]

$packageFile = $InputFileName;
$libraryFileName = Switch[$OperatingSystem, "Windows", "SVTools.dll", "MacOSX", "SVTools.dylib", "Unix", "SVTools.so", _, $Failed];
$libcsvFileName = Switch[$OperatingSystem, "Windows", "csv.dll", "MacOSX", "libcsv.dylib", "Unix", "libcsv.so", _, $Failed];

$library = FileNameJoin[{FileNameTake[$packageFile, {1, -3}], "LibraryResources", $SystemID, $libraryFileName}];
$libcsv = FileNameJoin[{FileNameTake[$packageFile, {1, -3}], "LibraryResources", $SystemID, $libcsvFileName}];
$adapterLoaded = False;

loadAdapter[] := 
If[$libraryFileName === $Failed,
	Return@CreateSVToolsException["AdapterNotLoaded", "Parameters" -> "OS " <> $OperatingSystem <> " not supported."];
	,
	If[!$adapterLoaded,
		If[LibraryLoad[SVTools`Private`$libcsv] === $Failed,
			Return@CreateSVToolsException["AdapterNotLoaded", "Parameters" -> "Failed to load " <> SVTools`Private`$libcsv];
		];
		If[LibraryLoad[SVTools`Private`$library] =!= $Failed,
			$SVGetDimsFile = LibraryFunctionLoad[SVTools`Private`$library, "svGetDimsFile", {"UTF8String", "UTF8String", "UTF8String", Integer, Integer, "Boolean", Integer, "UTF8String"}, "NumericArray"];
			$SVGetDimsString = LibraryFunctionLoad[SVTools`Private`$library, "svGetDimsString", {"UTF8String", "UTF8String", {"NumericArray", "Constant"}, Integer, Integer, "Boolean", Integer, "UTF8String"}, "NumericArray"];
			$SVImport = LibraryFunctionLoad[SVTools`Private`$library, "svParse", {"UTF8String", {"NumericArray", "Constant"}, "Boolean", "UTF8String", Integer, Integer, Integer, Integer, "UTF8String", "UTF8String", "UTF8String"}, "Void"];
			$SVClearCache = LibraryFunctionLoad[SVTools`Private`$library, "svClearCache", LinkObject, LinkObject];
			$adapterLoaded = True;
			,
			Return@CreateSVToolsException["AdapterNotLoaded", "Parameters" -> "Failed to load " <> $libraryFileName];
		];
	];
];


(* ::Section:: *)
(* Exceptions *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

(* SVToolsException[
	type_String,
	errorCode_Integer, 
	message_String, 
	param_List, 
	cause_] := 
Association[
	"Type" -> type,
	"ErrorCode" -> errorCode,
	"Message" -> message,
	"Parameters" -> param,
	"Cause" -> cause]; *)

$errID = 0;

$SVToolsExceptionCoreLUT = Association[
	(* "ExceptionTemplate" -> 
		<|"Type" -> "ExceptionTemplate", "ErrorCode" -> 0, "Message" -> "ExceptionTemplate"|>, *)
	"AdapterNotLoaded" -> 
		<|"Type" -> "AdapterNotLoaded", "ErrorCode" -> 1, "Message" -> "Error loading LibraryLink functions. See exception Parameters for more details."|>,
	"Unsupported" -> 
		<|"Type" -> "Unsupported", "ErrorCode" -> 2, "Message" -> "Unsupported call."|>,
	"FileNotFound" -> 
		<|"Type" -> "FileNotFound", "ErrorCode" -> 3, "Message" -> "One or more of the files could not be located. See exception Parameters for file path(s)."|>,
	"FileExists" -> 
		<|"Type" -> "FileExists", "ErrorCode" -> 5, "Message" -> "File already exists. See exception Parameters for file path."|>,	
	"Bounds" -> 
		<|"Type" -> "Bounds", "ErrorCode" -> 6, "Message" -> "Requested row or col outside of data bounds."|>,
	"InvalidOption" -> 
		<|"Type" -> "InvalidOption", "ErrorCode" -> 7, "Message" -> "Invalid option."|>,
	"FileCreateErr" -> 
		<|"Type" -> "FileCreateErr", "ErrorCode" -> 8, "Message" -> "Could not create file. See exception Parameters for file path."|>,
	"FileOpenErr" -> 
		<|"Type" -> "FileOpenErr", "ErrorCode" -> 9, "Message" -> "Could not open file. See exception Parameters for file path."|>,
	"GetDimsErr" -> 
		<|"Type" -> "GetDimsErr", "ErrorCode" -> 10, "Message" -> "Error retrieving dimensions of file. See exception Parameters for file path."|>,
	"ReadError" -> 
		<|"Type" -> "ReadError", "ErrorCode" -> 11, "Message" -> "Error reading data from Stream."|>
];

SVToolsException /: 
	MakeBoxes[exc_SVToolsException, form : StandardForm | TraditionalForm] := 
		RowBox[{"SVToolsException", "[<", exc["ErrorID"], ">, ", exc["Type"], "]"}]

SVToolsException[assoc_Association][key_] := assoc[key]

Options[CreateSVToolsException] = {"Cause" -> None, "Parameters" -> {}};

CreateSVToolsException[errorType_String, opts:OptionsPattern[]]:=
Module[{res, params},
	params = Replace[OptionValue["Parameters"], x_ :> {x} /; Head[x] =!= List];
	res = Lookup[$SVToolsExceptionCoreLUT, errorType, Association["Type" -> "UnknownErrorType" , "ErrorCode" -> -1, "Message" -> "Unknown error."]];
	AssociateTo[res, {"Parameters" -> params, "Cause" -> OptionValue["Cause"], "ErrorID" -> ++$errID}];
	$SVClearCache[];
	Return[SVToolsException[res]];
];

ValidateOption = System`ConvertersDump`Utilities`ValidateOption;
ReplaceSubelement = System`ConvertersDump`Utilities`ReplaceSubelement;

(* ::Section:: *)
(* Converter *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)


(* ::Subsection:: *)
(* Import *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Options[SVImport] = {
	CharacterEncoding 		-> "UTF8ISOLatin1",
	"EmptyField" 			-> "", 
	"FillRows" 				-> True, 
	"IgnoreEmptyLines"		-> False,
	"SkipLines" 			-> 0,
	"TakeElements"			-> {All, All},
	"TextDelimiters"		-> "\"",
	"Legacy"			-> False
}

(* Unsupported call *)
SVImport[format_, ___][___] := CreateSVToolsException["Unsupported", "Parameters" -> format];

SVImport[format : "CSV" | "TSV", rows_:All, cols_:All][
	input 	: (_?StringQ | _InputStream),  
	opts 	: OptionsPattern[]] := 
CheckAbort[
Module[
	{
		res, numRows, maxCols, rowStart, rowEnd, colStart, colEnd, fRows, resNum, levelSpec,
		emptyField, incompleteField, dims, cacheToken, enc, ignoreEmptyLines, td, msg,
		emptyFieldLen, incompleteFieldLen, lastRow, rowList, colList, minrow, maxrow, 
		mincol, total, curColList, curRowLen, earlyFail = False, streamMode, data
	},
	(* Load LibraryLink functions *)
	If[!loadAdapter[],
		Return@CreateSVToolsException["AdapterNotLoaded"];
	];

	(* Get and check options *)
	(* True => doesn't issue messages *)
	enc					= ValidateOption[True, SVImport, CharacterEncoding, Alternatives @@ Join[System`$CharacterEncodings, {None, Automatic, "UTF8ISOLatin1"}], opts];
	emptyField			= ValidateOption[True, SVImport, "EmptyField", _String, opts];
	fRows 				= ValidateOption[True, SVImport, "FillRows", True | False, opts];
	ignoreEmptyLines	= ValidateOption[True, SVImport, "IgnoreEmptyLines", True | False, opts];
	td					= ValidateOption[True, SVImport, "TextDelimiters", Automatic | None | (t_String /; 0 <= StringLength[t] <= 1), opts];
	text 				= ValidateOption[True, SVImport, "Legacy", True | False, opts];


	(* 
		There are two CSV implementations: files and streams. When importing in streamMode, 
		importText will call code very similar to Import[.., "Text"], and if text is True will also
		apply the same \r | \r\n -> \n that Text import does. We need to keep using Text import for 
		"Unicode" due to the UCS2 support, and Automatic for the encoding detection. 

		The File implementation use c++ i/o and much faster parsing methods as well as encoding 
		transformations at the list construction level. This should always be preferred when possible. 
	 *)
	streamMode = (!StringQ[input] || MatchQ[enc, "Unicode" | Automatic] || text);

	If[streamMode,
		data = importText[input, enc, text];

		If[data === {{}},
			Return[{{}}];
		];

		If[!MatchQ[data, {_Integer..}],
			Return@CreateSVToolsException["ReadError"];
		];

		(* Data only imported in stream mode, in file mode the input path is passed to the parser directly. *)
		data = NumericArray[data, "UnsignedInteger8"];
	];

	cacheToken = StringReplace[CreateUUID[], "-" -> ""];

	(* The backend returns rectangular data. Import preserves the length of columns. 
			Thus, we must introduce a unique flag to remove after parsing the file *)
	incompleteField = If[!fRows, CreateUUID[], emptyField];

	emptyFieldLen = StringLength[emptyField];
	incompleteFieldLen = StringLength[incompleteField];

	lastRow = 
		Switch[rows,
			All,
			0
			,
			_Integer,
			If[Positive@rows, rows, 0]
			,
			_List,
			{minrow, maxrow} = MinMax[rows];
			If[Negative@minrow,
				0
				,
				maxrow
			]
			,
			_Span,
			If[Internal`PositiveIntegerQ[rows[[1]]] && Internal`PositiveIntegerQ[rows[[2]]], 
				Max[rows[[1]], rows[[2]]]
				, 
				0
			]
		];

	dims = 
		Quiet @ Normal @
			If[streamMode,
				$SVGetDimsString[format, cacheToken, data, emptyFieldLen, incompleteFieldLen, ignoreEmptyLines, lastRow, td]
				,
				$SVGetDimsFile[format, cacheToken, input, emptyFieldLen, incompleteFieldLen, ignoreEmptyLines, lastRow, td]
			];
	(* Now the relevant data has been cached by the parser.  *)

	If[dims =!= {{}} && MatchQ[dims, {{___Integer}.. }],
		{numRows, maxCols} = {Length@dims, Max[Length/@dims]};
		,
		Return@CreateSVToolsException["GetDimsErr"];
	];

	(* 
		For List partial access, (i.e. rows or columns {1,7,5}), we import the span 1;;7 initially, and clean up later.
		Negative values will result in importing all data and taking the part at the end.
	 *)

	rowList = ReplaceSubelement[format, rows, numRows, "ForceLists" -> True];

	(* In most cases, we need to clean up row by row and issue the message per row *)
	colList = Quiet@ReplaceSubelement[format, cols, maxCols, "ForceLists" -> True];

	If[MatchQ[rowList, {$Failed.. }],
		earlyFail = True;
		numRows = Length[rowList];
	];

	If[MatchQ[colList, {$Failed.. }],
		(* Since we're returning early and not cleaning up later, 
			we need to issue the noelem message here *)
		
		earlyFail = True;
		numRows = Length[rowList];

		If[MatchQ[cols, _Span],
			msg := Message[Import::someelem, #, format] &;
			,
			msg := Message[Import::noelem, #, format] &;
		];

		levelSpec = If[ListQ[cols], {1}, {0}];
		
		Do[
			Map[msg, cols, levelSpec]
			, 
			numRows
		];
	];

	If[earlyFail,
		$SVClearCache[];
		Return@Switch[{rows, cols},
			{_Integer, _List | _Span},
			List@ConstantArray[$Failed, Length[colList]]
			,
			{Except[_Integer], _Integer},
			List /@ ConstantArray[$Failed, numRows]
			,
			{Except[_Integer], _List | _Span},
			ConstantArray[$Failed, {numRows, Length[colList]}]
			,
			{Except[_Integer], All},
			ConstantArray[$Failed, numRows]
			,
			_,
			$Failed
		];
	];

	{rowStart, rowEnd} = MinMax[Replace[rowList, $Failed -> {}, {1}]];
	{colStart, colEnd} = 
		Switch[cols,
			All,
			{1, maxCols}
			,
			_Integer,
			If[Negative@cols, 
				{1, maxCols}
				, 
				{cols, cols}
			]
			,
			_List,
			mincol = Min[cols];
			If[Negative@mincol,
				{1, maxCols}
				,
				MinMax[Replace[colList, $Failed -> {}, {1}]]
			]
			,
			_Span,
			If[Internal`PositiveIntegerQ[cols[[1]]] && Internal`PositiveIntegerQ[cols[[2]]], 
				MinMax[Replace[colList, $Failed -> {}, {1}]]
				, 
				{1, maxCols}
			]
		];

	dims = dims[[rowStart ;; rowEnd, colStart ;; colEnd]];

	total = Total[dims, 2];
	
	If[total === 0,
		Return[Replace[dims, 0 -> Nothing, {2}]];
	];

	resNum = Developer`AllocateNumericArray["UnsignedInteger8", {total}];

	Quiet[$SVImport[format, resNum, ignoreEmptyLines, cacheToken, rowStart, rowEnd, colStart, colEnd, emptyField, incompleteField, td]];
	
	If[!Developer`NumericArrayQ[resNum],
		Return@CreateSVToolsException["ReadError"];
	];

	dims = Developer`ToPackedArray[dims];

	(*  StreamMode supports ImportString's possile Import of strings with ToCharacterCode > 255.
		Unicode also has CharacterCodes > 255, thus both of these were normalized to UTF8.
	 *)
	res = 
		Quiet @ If[enc === "UTF8ISOLatin1" || streamMode,
			Check[
				Internal`FromCharacterCodesToListOfStrings[resNum, dims, "UTF8"],
				Internal`FromCharacterCodesToListOfStrings[resNum, dims, "ISOLatin1"]
			]
			,
			(* Otherwise decode strings while constructing lists. *)
			Internal`FromCharacterCodesToListOfStrings[resNum, dims, enc]
		];

	If[!fRows,
		res = Replace[res, incompleteField :> Nothing, {2}]
	];

	(* 
		Cleanup list partial access. 
		For something like rows = {3,7,5}, we imported 3;;7, so now we need to take {1,5,3} from the resulting set.
		e.g. rows - minrow + 1
	 *)

	If[Positive@Length@res && MatchQ[rows, _List | _Span],
		If[rowStart > 1,
			rowList = Replace[rowList, x_Integer :> x - rowStart + 1, {1}];
		];
		res = 
			If[IntegerQ[#] && (1 <= # <= Length[res]),
				res[[#]]
				,
				$Failed
				,
				$Failed
			] & /@ rowList;
	];

	(* 
		Columns were slightly more complicated. The row might be ragged, and missing elements should
		be replaced with $Failed to be backward compatible. 
	 *)
	If[Positive@Length@res,
		Switch[cols,
			_?Internal`NegativeIntegerQ,
			res = 
				(* This message is issued here row by row. *)
				List@If[(Abs@cols <= Length[#]),
					#[[cols]]
					,
					Message[Import::noelem, cols, format];
					$Failed
					,
					Message[Import::noelem, cols, format];
					$Failed
				] & /@ res;
			,
			_Span,
			res = 
				Function[curRow,
					curRowLen = Length[curRow];
					If[colStart > 1,
						(* This message is issued here row by row  *)
						curColList = Replace[colList, x_Integer :> x - colStart + 1, {1}];
						,
						curColList = colList;
					];
					If[IntegerQ[#] && (1 <= # <= curRowLen),
						curRow[[#]]
						,
						Message[Import::someelem, cols, format];
						$Failed
						,
						Message[Import::someelem, cols, format];
						$Failed
					] & /@ curColList
				] /@ res;
			,
			{_?Internal`PositiveIntegerQ ..},
			res = 
				Function[curRow,
					curRowLen = Length[curRow];
					If[colStart > 1,
						curColList = Replace[cols, x_Integer :> x - colStart + 1, {1}];
						,
						curColList = colList;
					];
					(* This message is issued here row by row  *)
					If[IntegerQ[#] && (1 <= # <= curRowLen),
						curRow[[#]]
						,
						Message[Import::noelem, # + colStart - 1, format];
						$Failed
						,
						Message[Import::noelem, # + colStart - 1, format];
						$Failed
					] & /@ curColList
				] /@ res;
			,
			_List,
			res = 
				Function[curRow,
					(* This message is issued here row by row. *)
					If[IntegerQ[#] && (1 <= Abs@# <= Length[curRow]),
						curRow[[#]]
						,
						Message[Import::noelem, #, format];
						$Failed
						,
						Message[Import::noelem, #, format];
						$Failed
					] & /@ cols
				] /@ res;
		];
	];

	res
]
,
$SVClearCache[];
]


Options[SVGetDimensions] = {
	CharacterEncoding 		-> "UTF8ISOLatin1",
	"IgnoreEmptyLines"		-> False,
	"Summary"				-> False,
	"TextDelimiters" 		-> "\""
}

(* Unsupported call *)
SVGetDimensions[format_, ___] := CreateSVToolsException["Unsupported", "Parameters" -> format];

SVGetDimensions[
	format 	: "CSV" | "TSV",
	input 	: (_String | _InputStream), 
	opts 	: OptionsPattern[]] := 
Module[{data, fileMode, dims, enc, dataStrm, ignoreEmptyLines, summary},

	(* Load LibraryLink functions *)
	If[!loadAdapter[],
		Return@CreateSVToolsException["AdapterNotLoaded"];
	];

	enc					= ValidateOption[True, SVGetDimensions, CharacterEncoding, Alternatives @@ Join[System`$CharacterEncodings, {None, Automatic, "UTF8ISOLatin1"}], opts];
	ignoreEmptyLines	= ValidateOption[True, SVGetDimensions, "IgnoreEmptyLines", True | False, opts];
	td					= ValidateOption[True, SVGetDimensions, "TextDelimiters", t_String /; 0 <= StringLength[t] <= 1, opts];
	summary				= ValidateOption[True, SVGetDimensions, "Summary", _?BooleanQ, opts];

	dims = 
		If[StringQ[input],
			Quiet @ Normal @ $SVGetDimsFile[format, "", input, 0, 0, ignoreEmptyLines, 0, td]
			,
			data = importText[input, enc, False];

			If[!MatchQ[data, {_Integer..}],
				Return@CreateSVToolsException["ReadError"];
			];

			data = NumericArray[data, "UnsignedInteger8"];

			Quiet @ Normal @ $SVGetDimsString[format, "", data, 0, 0, ignoreEmptyLines, 0, td]
		];

	If[dims =!= {{}} && MatchQ[dims, {{___Integer}.. }],
		If[summary,
			Dataset@<|
				"Format" -> format,
				"FileSize" -> N@UnitConvert[Quantity[Total[dims, 2], "Bytes"], "Megabytes"],
				"RowCount" -> Length@dims,
				"MaxColumnCount" -> Max[Length/@dims]
			|>
			,
			dims
		]
		,
		CreateSVToolsException["GetDimsErr"]
	]
]


(* ::Subsection:: *)
(* Utility Functions *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

(* importText returns the character codes of the data *)
importText[input : (_?StringQ | _InputStream), "Unicode", _] :=
Module[{data, frst, bo, strm, fileMode},
	fileMode = StringQ[input];

	strm = 
		If[fileMode,
			OpenRead[input, BinaryFormat -> True]
			,
			input
		];

	bo = $ByteOrdering;
	data = BinaryRead[strm, "Character16", ByteOrdering -> bo];

	If[data === {}, 
		Return[{{}}];
	];

	(* Check for ByteOrderMark character:
	UTF-16 Big Endian		FE FF
	UTF-16 Little Endian	FF FE
	When data are exchange in the same byte order as they were in the memory of 
	the originating system, they may appear to be in the wrong byte order on the
	receiving system. In that situation, a BOM would look like 0xFFFE which is a
	noncharacter, allowing the receiving system to apply byte reversal before 
	processing the data. http://www.unicode.org/faq/utf_bom.html#BOM *)
	Switch[data,
		FromCharacterCode[16^^feff], Null,				(* Retain byte order  *)
		FromCharacterCode[16^^fffe], bo = -bo,			(* Reverse byte order *)
		_, SetStreamPosition[strm, 0]					(* No order indicated *)
	];
	data = Quiet[BinaryReadList[strm, "Character16", ByteOrdering -> bo]];

	(* Todo use try *)
	If[fileMode,
		Close[strm];
	];

	If[data === {}, 
		Return[{{}}];
	];

	If[data === $Failed,
		Return@CreateSVToolsException["ReadError"];
	];

	Quiet@Check[
		ToCharacterCode[StringJoin@data, "UTF8"]
		,
		ToCharacterCode[StringJoin@data, "ISOLatin1"]
	]
]

importText[input : (_?StringQ | _InputStream), encoding_, textImport_?BooleanQ] :=
Module[{str, originalStrmPos, bytes, strmPos, strm2, enc = encoding, fileMode},
	fileMode = StringQ[input];

	strm = 
		If[fileMode,
			OpenRead[input, BinaryFormat -> True]
			,
			input
		];

	originalStrmPos = StreamPosition[strm];	
	str = Read[strm, Record, RecordSeparators -> {}];
	If[str === $Failed,
		Return@CreateSVToolsException["ReadError"];
	];
	If[str === EndOfFile, 
		Return[{{}}];
	];
	(* apply EOF char and return conversions *)

	bytes = ToCharacterCode[str];

	Switch[enc,
		"UTF8ISOLatin1",
		(* If no explicit CharacterEncoding option was used, then use a fail-safe
			method.  First try UTF8 and fail over to ISOLatin1 if it fails.
		*)
		str = 
			Quiet[
				Check[
					FromCharacterCode[bytes, "UTF8"]
					,
					FromCharacterCode[bytes, "ISOLatin1"]
					,
					{$CharacterEncoding::utf8}
				]
				,
				{$CharacterEncoding::utf8}
			];
		,
		Automatic,
		(* If option is Automatic, use encoding classifier. *)
		enc = 
			Quiet[
				Catch[
					(*This runs on average 13.71 ms - March 5, 2018. Up to 600, to limit time spent classifying*)
					FindCharacterEncoding`FindCharacterEncoding[Take[bytes, UpTo[600]], "AllowUCS2"->True]
					,
					"noFeature"
					,
					Function[Null, "UTF8"]
				]
				, 
				(* The other messages should never be issued due to checks above. *)
				{FindCharacterEncoding`FindCharacterEncoding::insuffinfo}
			];

		If[!StringQ[enc], enc = "UTF8"];
		If[enc === "Unicode",
			(*UCS2*)
			(*Try re-using the stream for efficiency. If SetStreamPosition fails, e.g. for a non-seekable stream,
			  the returned position will not match the requested position*)
			(*Quiet without 2nd argument, as messages are rarely issued for setting position to 0 (most frequent value
			  for originalStrmPos), and for the case a message is issued, the return value will not be 0, and hence is 
			  dealt with*)
			strmPos = Quiet[SetStreamPosition[strm, originalStrmPos]];
			If[strmPos === originalStrmPos,
				Return[PlaintextImport[strm, CharacterEncoding->"Unicode"], Module];
				,
				Internal`WithLocalSettings[strm2 = StringToStream[str]
					,
					Return[PlaintextImport[strm2, CharacterEncoding->"Unicode"], Module];
					,
					Close[strm2]
				]			
			];
			,
			(*Non-UCS2 encoding*)
			(* 2-step check, in practice from tests, vast majority don't go through both, since model produces 
			very few UTF-8 false negatives, and the vast majority (90.5%) of files are UTF-8.
			*)
			str = 
				Quiet[		
					Check[			
						FromCharacterCode[bytes, enc],
						Check[		
							If[enc =!= "UTF8",
								FromCharacterCode[bytes, "UTF8"],
								(*This never issues Messages, since just returning bytes*)
								FromCharacterCode[bytes, "ISOLatin1"]
							]
							,
							FromCharacterCode[bytes, "ISOLatin1"]
						]
					]
					(* Deliberately not checking {$CharacterEncoding::utf8} since we don't know the message name 
						for the predicted encoding
					*)
			];			
		];
		,
		_,
		str = FromCharacterCode[bytes, enc]; 
	];

	(* Todo use try *)
	If[fileMode,
		Close[strm];
	];
	
	If[textImport,
		(* apply EOF char and return conversions *)
		str = StringReplace[str, {"\r\n"|"\r" -> "\n", FromCharacterCode@62371 -> "\n"}];
		
		(* Per bug 106165: we remove the byte-order mark used for Unicode, but may appear in UTF8 files and ISOLatin1. *)
		If[str =!= "" && MatchQ[First@ToCharacterCode[StringTake[str, 1]], 16^^feff | 16^^fffe], str = StringDrop[str, 1]];
	];

	(* To prevent returning any character codes > 255 *)
	Quiet@Check[
		ToCharacterCode[str, "UTF8"]
		,
		ToCharacterCode[str, "ISOLatin1"]
	]
]

(* ::Subsection:: *)
(* Export *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Options[SVExport] = {
	Alignment 				-> None,
	CharacterEncoding 		-> "UTF8",
	"EmptyField" 			-> "",
	"FillRows"				-> False,
	"Overwrite" 			-> False, 
	"TableHeadings"			-> None,
	"TextDelimiters" 		-> "\""
}

SVExport[
	format 	: "CSV" | "TSV", 
	outFile : (_File | _String | _OutputStream), 
	data 	: {__List}, 
	opts 	: OptionsPattern[]] :=
Module[{dest, destname, res, maxLength, overwrite, fileMode, recordSeparator, td, emptyField, head, align, lens, numcols, fill, enc},

	Switch[outFile,
		_File,
		dest = First[outFile];
		fileMode = True;
		,
		_String,
		dest = outFile;
		fileMode = True;
		,
		_OutputStream,
		dest = outFile;
		fileMode = False;
	];

	align		= ValidateOption[True, SVExport, Alignment, _, opts];
	enc			= ValidateOption[True, SVExport, CharacterEncoding, Alternatives @@ Join[System`$CharacterEncodings, {None, Automatic}], opts] /. Automatic -> "UTF8";
	emptyField 	= ValidateOption[True, SVExport, "EmptyField", _String, opts];
	fill 		= ValidateOption[True, SVExport, "FillRows", _?BooleanQ | Automatic, opts];
	overwrite 	= ValidateOption[True, SVExport, "Overwrite", _?BooleanQ, opts];
	head		= ValidateOption[True, SVExport, "TableHeadings", Automatic | None | {__} | {{__},{__}}, opts];
	td			= ValidateOption[True, SVExport, "TextDelimiters", Automatic | None | (t_String /; 0 <= StringLength[t] <= 1), opts];

	fill = Replace[fill, Automatic -> True];
	emptyField = StringReplace[emptyField, {"\"" -> "\"\"", "\[IndentingNewLine]" -> "\n"}];
	recordSeparator = Replace[format, {"CSV" -> ",", "TSV" -> "\t"}];

	If[fileMode,
		If[FileExistsQ[dest],
			If[overwrite,
				Quiet[DeleteFile[dest]];
				,
				Return@CreateSVToolsException["FileExists", "Parameters" -> outFile];
			]
		];

		If[FailureQ[Quiet[destname = CreateFile[dest]]],
			Return@CreateSVToolsException["FileCreateErr", "Parameters" -> outFile];
		];

		If[FailureQ[dest = OpenWrite[destname, BinaryFormat -> enc === "Unicode", CharacterEncoding -> enc]],
			Quiet[ImportExport`FileUtilities`DeleteFileOrDir[destname]];
			Return@CreateSVToolsException["FileOpenErr", "Parameters" -> outFile];
		]
	];

	(* Get rid of Nulls *)
	res = data /. Null -> "";

	If[head =!= None,
		res = AddHeader[head, res];
	];

	lens = Length /@ res;
	numcols = Max[lens];

	If[fill,
		missingCells = ConstantArray["", numcols - #] & /@ lens;
		res = MapThread[Join, {res, missingCells}];
	];

	res = 
		Replace[
			res
			, 
			{
				"" :> td <> emptyField <> td,
				cellData:(_Integer | _NumberForm) :> ToString[cellData],
				cellData_Real :> ToString[CForm@cellData, InputForm],
				cellData_Complex :> StringReplace[ToString[cellData, InputForm], " " -> ""],
				cellData_String :>  td <> StringReplace[cellData, {"\"" -> "\"\"", "\[IndentingNewLine]" -> "\n"}] <> td,
				cellData_ :>  td <> StringReplace[ToString[cellData, InputForm], {"\"" -> "\"\"", "\[IndentingNewLine]" -> "\n"}] <> td
			}
			, 
			{2}
		];

	If[align =!= None,
		res = PadTable[res, align, lens, numcols];
	];

	res = StringRiffle[#, recordSeparator] <> "\n" & /@ res;

	If[enc === "Unicode",
		BinaryWrite[dest, FromCharacterCode[16^^feff], "Character16", ByteOrdering -> $ByteOrdering];
		BinaryWrite[dest, res, "Character16", ByteOrdering -> $ByteOrdering];
		,
		Scan[WriteString[dest, #] &, res];
	];

	If[fileMode,
		Close[dest];
		destname
		,
		dest
	]
]

SVExport[format : "CSV" | "TSV", outFile : (_File | _String | _OutputStream), {}, rest___] := 
	SVExport[format, outFile, {{}}, rest]

SVExport[format : "CSV" | "TSV", outFile : (_File | _String | _OutputStream), data_List, rest___] := 
	SVExport[format, outFile, Replace[data, expr:Except[_List] :> {expr}, 1], rest]

SVExport[format : "CSV" | "TSV", outFile : (_File | _String | _OutputStream), data_, rest___] :=
	SVExport[format, outFile, {{data}}, rest]

(* Unsupported call *)
SVExport[format_, ___] := CreateSVToolsException["Unsupported", "Parameters" -> format] /; !StringMatchQ[format, "CSV" | "TSV"]


(* ::Subsection:: *)
(* Utility Functions *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

AddHeader[head_, data_] :=
Block[{rowhead, colhead, numrows, numcols, newdata = data, i},
	numrows = Length@data;
	numcols = Max[Length/@data];

	(* Format Header *)
	Switch[head,
		Automatic,
		{rowhead, colhead} =  {Range[numrows], Range[numcols]}
		,
		{{__}, {__}},
		{rowhead, colhead} = {PadRight[head[[1]], numrows, ""], PadRight[head[[2]], numcols, ""]}
		,
		{__},
		colhead = PadRight[head, numcols, ""];
		rowhead = None
		,
		_,
		(* In None of the above, return data *)
		Return[data]
	];

	If[rowhead =!= None && colhead =!= None,
		PrependTo[colhead, ""]
	];

	If[rowhead =!= None,
		newdata = MapThread[Prepend, {newdata, rowhead}]
	];

	If[colhead =!= None,
		newdata = Prepend[newdata, colhead]
	];

	newdata
]

PadField[str_String, _, _] := str

PadField[str_String, num_, Left] := str <> Table[" ", {num}]

PadField[str_String, num_, Right] := Table[" ", {num}] <> str

PadField[str_String, num_, Center] := 
	Table[" ", {Floor[num/2]}] <> str <> Table[" ", {Ceiling[num/2]}]

PadColumn[col_, colalign_] :=
Block[{lens},
	lens = StringLength /@ col;
	MapThread[PadField[#1, #2, colalign]&, {col, Max@lens - lens}]
]

PadTable[table_, colalign_, lens_, max_] := 
	MapThread[
		Take[#1, #2]&
		, 
		{
			Transpose[PadColumn[#, colalign]& /@ Transpose[Map[PadRight[#, max, ""]&, table]]]
			, 
			lens
		}
	]



End[]
EndPackage[]
