BeginPackage["ImageFileTools`RLE`"]

RLEGet::usage = "RLEGet[elem_][filename_] imports \"elem\" from a \"file\".\nPossible values for \"elem\" include \"Data\", \"Image\", \"ImageSize\", \"Rule\", \"Comments\", \"ImageCount\" and \"MetaInformation\"."

Begin["`Private`"]

$InitImageFileToolsRLE = False;

$ImageFileToolsBaseDirectory = FileNameDrop[$InputFileName, -2];

Get[FileNameJoin[{$ImageFileToolsBaseDirectory, "LibraryResources", "LibraryLinkUtilities.wl"}]];

$BaseLibraryDirectory = FileNameJoin[{$ImageFileToolsBaseDirectory, "LibraryResources", $SystemID}];
$ImageFileToolsLibraryRLE = "ImageFileTools";

InitImageFileToolsRLE[debug_ : False] :=
	If[TrueQ[$InitImageFileToolsRLE],
		$InitImageFileToolsRLE
		,
		$InitImageFileToolsRLE =
			Catch[
				Block[
					{
						$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]
					},

					SafeLibraryLoad[debug, $ImageFileToolsLibraryRLE];

					RegisterPacletErrors[$ImageFileToolsLibraryRLE, <|
						"FileAccessError" 	-> "Cannot find `FileName`.",
						"FormatError" 		-> "`FileName` cannot be interpreted as RLE file.",
						"OutOfMemory" 		-> "Not enough memory to import `FileName` with `ImageSize` dimensions."
					|>];

					$RLEStreamOpen 		= SafeLibraryFunction["RLEStreamOpen"    	, {"UTF8String"}	, {"Boolean"} , "Throws" -> True];

					$RLEGetImageCount 	= SafeLibraryFunction["RLEGetImageCount" 	, {}             	, {Integer}   , "Throws" -> True];
					$RLEGetMetadata 	= SafeLibraryFunction["RLEGetMetadata"   	, LinkObject    	, LinkObject  , "Throws" -> True];
					$RLEGetImageData 	= SafeLibraryFunction["RLEGetImageData"  	, {Integer}     	, {"RawArray"}, "Throws" -> True];
					$RLEGetImageObject 	= SafeLibraryFunction["RLEGetImageObject"	, {Integer}     	, {"Image"}   , "Throws" -> True];

					$RLEStreamClose		= SafeLibraryFunction["RLEStreamClose"   	, {}            	, {"Boolean"} , "Throws" -> True];

				];
				True
			]
	]

(*
	██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
	██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
	███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
	██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
	██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
	╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝
*)

ValidateOption = System`ConvertersDump`Utilities`ValidateOption;
TryCatch = System`ConvertersDump`Utilities`TryCatch;

Options[RLEOptions] = {
	"ValidateFileName" -> True
}

(* 
	This function throws but does not catch, thus, 
	it can be called only in a Catch[] context 
*)
checkForAvailableMemory[fileNameInput_, opts: OptionsPattern[]] :=
	Block[
		{
			dims, maxSize
		},

		dims = Quiet[RLEGet["ImageSize"][fileNameInput, opts]];
		If[FailureQ[dims],
			Throw[dims];
		];

		If[!MatchQ[dims, {_?Internal`PositiveIntegerQ, _?Internal`PositiveIntegerQ}],
			dims = First[ReverseSortBy[dims, Times @@ # &]]
		];

		maxSize = Times @@ dims;

		(* Fail to import if the image size is larger than 80% of available memory *)
		If[maxSize >= N[MemoryAvailable[] * 8 / 10],
			Throw[CreatePacletFailure["OutOfMemory", "MessageParameters" -> <|"ImageSize" -> dims, "FileName" -> FileNameTake[fileNameInput, -1]|>]];
		];
	];

(* 
	This function throws but does not catch, thus, 
	it can be called only in a Catch[] context 
*)
validateInit[] :=
	If[!TrueQ[$InitImageFileToolsRLE] && !TrueQ[InitImageFileToolsRLE[]],
		Throw[$InitImageFileToolsRLE]
	];

createCacheKey[fileName_String, prop_String, imageIndex_Integer : -1] :=
	Hash[
		{
			RLEstreamRead,
			{
				FileHash[fileName],
				prop,
				imageIndex
			}
		}
	]

(* 
	This function throws but does not catch, thus, 
	it can be called only in a Catch[] context 
*)
fileExistsQ[fileNameInput_String, opts: OptionsPattern[]] :=
	Block[
		{
			fileName, validateFileName
		},

		validateFileName = Quiet@ValidateOption[True, RLEOptions, "ValidateFileName", _?BooleanQ, opts];
		If[!TrueQ[validateFileName],
			Return[fileNameInput];
		];

		fileName = FindFile[fileNameInput];

		If[
			Or[FailureQ[fileName],
				!FileExistsQ[fileName]
			],
			Throw[CreatePacletFailure["FormatError", "MessageParameters" -> <|"FileName" -> FileNameTake[fileNameInput, -1], "Format" -> "RLE"|>]];
			,
			If[!SameQ[FileFormat[fileName], "RLE"],
				Throw[CreatePacletFailure["FileAccessError", "MessageParameters" -> <|"FileName" -> FileNameTake[fileNameInput, -1], "Format" -> "RLE"|>]];
			];
		];

		Return[fileName];
	];

(*
	 ██████╗ ███████╗████████╗████████╗███████╗██████╗ ███████╗
	██╔════╝ ██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
	██║  ███╗█████╗     ██║      ██║   █████╗  ██████╔╝███████╗
	██║   ██║██╔══╝     ██║      ██║   ██╔══╝  ██╔══██╗╚════██║
	╚██████╔╝███████╗   ██║      ██║   ███████╗██║  ██║███████║
 	 ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝
*)


RLEGetImageCount[fileNameInput_String, opts: OptionsPattern[]] :=
	Catch[Block[
		{
			key, cache, result, fileName = fileExistsQ[fileNameInput, opts]
		},

		key = createCacheKey[fileName, "ImageCount"];
		cache = Internal`CheckImageCache[key];
		If[FailureQ[cache],
			result =
				TryCatch[
					$RLEStreamOpen[fileName];
					$RLEGetImageCount[]
					,
					$RLEStreamClose[];
				];

			If[FailureQ[result],
				Throw[result];
			];

			Internal`SetImageCache[key, result];
			,
			result = cache;
		];

		Return[result];
	]]

RLEGetMetadata[fileNameInput_String, opts: OptionsPattern[]] :=
	Catch[Block[
		{
			key, cache, result, fileName = fileExistsQ[fileNameInput, opts]
		},

		key = createCacheKey[fileName, "MetaInformation"];
		cache = Internal`CheckImageCache[key];
		If[FailureQ[cache],
			result =
       			TryCatch[
					$RLEStreamOpen[fileName];
					$RLEGetMetadata[]
					,
					$RLEStreamClose[];
				];

			If[FailureQ[result],
				Throw[result];
			];

			Internal`SetImageCache[key, result];
			,
			result = cache;
		];

		result = Replace[result, "" -> Missing["NotAvailable"], {2}];

		Return[Replace[result, {x_} :> x]];
	]]

RLEGet[elem : "RawData" | "Data" | "Image" | "Graphics"][fileNameInput_String, opts: OptionsPattern[]] :=
	Catch[Block[
		{
			imageCount,
			result
		},

		validateInit[];

		imageCount = RLEGetImageCount[fileNameInput, opts];
		If[FailureQ[imageCount],
			Throw[imageCount];
		];

		checkForAvailableMemory[fileNameInput, opts];

		result =
			TryCatch[
				$RLEStreamOpen[fileExistsQ[fileNameInput, opts]];
				If[StringMatchQ[elem, "Image" | "Graphics"],
					$RLEGetImageObject /@ Range[imageCount]
					,
					$RLEGetImageData /@ Range[imageCount]
				]
				,
				$RLEStreamClose[];
			];

		If[!FailureQ[result],
			result = 
				Switch[elem,
					"Data",
					Normal /@ result
					,
					"Graphics",
					Replace[Image`ToGraphicsRaster[#, ImageType[#]], (opt_ -> _) :> (opt -> Automatic), {1}] & /@ result
					,
					_,
					result
				];
		];

		Return[Replace[result, {x_} :> x]];
	]]

RLEGet[elem : "ImageSize" | "Rule" | "Comments"][fileNameInput_String, opts: OptionsPattern[]] :=
	Catch[
		validateInit[];

		Replace[Lookup[RLEGetMetadata[fileNameInput, opts], elem], {x_} :> x]
	]

RLEGet[elem : "MetaInformation"][fileNameInput_String, opts: OptionsPattern[]] :=
	Catch[
		validateInit[];

		RLEGetMetadata[fileNameInput, opts]
	]

RLEGet[elem : "ImageCount"][fileNameInput_String, opts: OptionsPattern[]] :=
	Catch[
		validateInit[];

		RLEGetImageCount[fileNameInput, opts]
	]

RLEGet[elem_][fileNameInput_String, opts: OptionsPattern[]] := Return[$Failed]

End[]
EndPackage[]
