BeginPackage["TesseractTools`"]
Begin["`Private`"]

$InitTesseractTools = False;

$ThisDirectory = FileNameDrop[$InputFileName, -1]
$BaseLibraryDirectory = FileNameJoin[{ParentDirectory[$ThisDirectory], "LibraryResources", $SystemID}];
$TesseractToolsLibrary = "TesseractTools";

(*****************************)
(**Getting path to traindata**)
(*****************************)
baseDir = $BaseLibraryDirectory;
TrainDataPath = $BaseLibraryDirectory;
LanguageName = "eng";
finalPath = $Failed; (*Temporary location to be removed at the end of recognition*)
stringSplitter = TesseractTools`Private`$GetStringSplitter[]; (*To split string if many rectangles were specified as a mask*)

safeLibraryLoad[debug_, lib_] :=
	Quiet[
		Check[
			LibraryLoad[lib],
			If[TrueQ[debug],
				Print["Failed to load ", lib]
			];
			Throw[$InitTesseractTools = $Failed]
		]
	]
safeLibraryFunctionLoad[debug_, args___] :=
	Quiet[
		Check[
			LibraryFunctionLoad[$TesseractToolsLibrary, args],
			If[TrueQ[debug], (*LibraryFunctionError[]*)
				Print["Failed to load the function ", First[{args}], " from ", $TesseractToolsLibrary]
			];
			Throw[$InitTesseractTools = $Failed]
		]
	]

InitTesseractTools[debug_ : False] := If[TrueQ[$InitTesseractTools],
	$InitTesseractTools,
	$InitTesseractTools = Catch[
		Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
			safeLibraryLoad[debug, $TesseractToolsLibrary];

			(*Initialize Tesseract with a file or an image, language, language path, segmentation mode, engine mode, Tesseract options, region of interest, Leptonica usage need and the granularity*)
			(*This uses Leptonica to construct an image from a file path and pass the data to Tesseract*)
			$InitFile  = safeLibraryFunctionLoad[debug, "InitFile", {{"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}}, "Boolean"];
			(*This uses LibraryLink MImage to extract the image data and pass it to Tesseract*)
			$InitImage = safeLibraryFunctionLoad[debug, "InitImage", {{"Image", "Constant"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"Boolean"}, {"UTF8String"}}, "Boolean"];
			(*De initialize Tesseract before switching to another file*)
			$UnInit    = safeLibraryFunctionLoad[debug, "UnInit", {}, "Boolean"];

			(*
			 * All the functions below:
			 * 1. Have to be used in a context between $InitFile/$InitImage and $UnInit
             * 2. Perform the recognition/extraction based on parameters specified in $InitFile/$InitImage
			*)

			(*Recognize the text in the image or file and return a string, which will be split using special splitter if many regions of interest where specified*)
			$RecognizeTextImage = safeLibraryFunctionLoad[debug, "RecognizeTextImage", {}, "UTF8String"];
			$RecognizeTextFile  = safeLibraryFunctionLoad[debug, "RecognizeTextFile", {}, "UTF8String"];

			(*Extract the strength of the recognition as a strings representing list of reals*)
			$GetImageComponentsConfidence  = safeLibraryFunctionLoad[debug, "GetImageComponentsConfidence", {}, "UTF8String"];
			(*Extract the bounds of the recognition as a strings representing list of points*)
			$GetImageComponentsBoundingBox = safeLibraryFunctionLoad[debug, "GetImageComponentsBoundingBox", {}, "UTF8String"];

			(*The functions below are not documented functionality and were not stressed to pass all the tests, thus maybe not stabile*)

			(*Extract the orientation related information*)
			$GetOrientation    = safeLibraryFunctionLoad[debug, "GetOrientation", {_Integer}, "UTF8String"];
			(*Extract the strength of the recognition per character as a strings representing list of reals*)
			$GetCharactersConfidence       = safeLibraryFunctionLoad[debug, "GetCharactersConfidence", {}, "UTF8String"];
			(*Get the special splitter used for splitting text in case if many regions of interest were specified*)
			$GetStringSplitter = safeLibraryFunctionLoad[debug, "GetStringSplitter", {}, "UTF8String"];
		];
		True
	]
]

(*****************************)
(**Assisting functions**)
(*****************************)
formatString[asc_] :=
	Block[{$Context = "TesseractTools`"},
		If[StringMatchQ[ToString[asc], "N/A", IgnoreCase -> True],
			"n/a",
			If[StringContainsQ[ToString[asc], "{"],
				StringTake[
					StringReplace[
						ToString[asc],
						{"->" -> "-", WhitespaceCharacter -> ""}
					],
					{2, -2}
				]
				,
				StringReplace[
					ToString[asc],
					{"->" -> "-", WhitespaceCharacter -> "" , "True" -> "T", "False" -> "F"}
				]
			]
		]
	]

processRegion[rect_, width_] :=
	Block[{$Context = "TesseractTools`"},
		If[StringMatchQ[ToString[rect], "N/A", IgnoreCase -> True],
			"n/a",
			With[
				{
					tmp = Join[
						{rect[[1, 1]], Abs[width - rect[[1, 2]] - Abs[rect[[2, 2]] - rect[[1, 2]]]]},
						{rect[[2, 1]] - rect[[1, 1]], Abs[Abs[width - rect[[1, 2]]] - Abs[width - rect[[1, 2]] - Abs[rect[[2, 2]] - rect[[1, 2]]]]]}
					]
				},
				StringTake[
					StringJoin[
						{ToString@IntegerPart[tmp[[#]]], ","} & /@ Range[4]
					],
					{1, -2}
				]
			]
		]
	]

processRectangle[res_] :=
	Block[{$Context = "TesseractTools`"},
		Rectangle @@ Partition[#, 2] & /@ res
	]

processConfidence[res_] :=
	Block[{$Context = "TesseractTools`"},
		N[# / 100] & /@ res
	]

processRectangleAndConfidence[res_] :=
	Block[{$Context = "TesseractTools`"},
		res /. ((lhs_ -> rhs_) :> lhs ->
			Map[
				Which[
					First[#] === "Confidence" , First[#] -> Last[#] / 100,
					First[#] === "BoundingBox", First[#] -> Rectangle @@ Partition[Last[#], 2],
					First[#] === "Choices"    , First[#] -> {First[First[Last[#]]] -> Last[Last[Last[#]]] / 100},
					True                      , First[#] -> Map[First[#] -> Last[#] / 100 &, Last[#]]
				]&
				,
				rhs
			])
	]

(*Rule validation*)
validRuleQ[(Rule | RuleDelayed)[_, _]] = True
validRuleQ[_] = False

(*List validation*)
checkForList[l_] :=
	Block[{$Context = "TesseractTools`"},
		If[
			And[
				ListQ[l],
				(ListQ[First[l]] || Length[l] === 1)
			],
			checkForList[First[l]],
			l
		]
	]

(*****************************)
(**Overridden functions**)
(*****************************)
Init[FI_, rect_ : "N/A", opts_ : "N/A", seg_ : "N/A", engine_ : "N/A", convLept_ : True, wgran_ : "block", width_] :=
	Block[{$Context = "TesseractTools`"},
		If[Quiet[StringQ[FI]],
			$InitFile[
				FI,
				ToLowerCase[LanguageName],
				TrainDataPath,
				ToLowerCase[seg],
				ToLowerCase[engine],
				formatString[opts],
				With[
					{
						t = DeleteCases[rect, "N/A" | "n/a"]
					},
					If[Length[t] === 0 && MatchQ[Head[t], List],
						"n/a",
						StringTake[
							StringJoin @@ (StringJoin[#, ","]& /@ (processRegion[#, width]& /@ t)),
							{1, -2}
						]
					]
				],
				ToLowerCase[wgran]
			]
			,
			$InitImage[
				FI,
				ToLowerCase[LanguageName],
				TrainDataPath,
				ToLowerCase[seg],
				ToLowerCase[engine],
				formatString[opts],
				With[
					{
						t = DeleteCases[rect, "N/A" | "n/a"]
					},
					If[Length[t] === 0 && MatchQ[Head[t], List],
						"n/a",
						StringTake[
							StringJoin @@ (StringJoin[#, ","]& /@ (processRegion[#, width]& /@ t)),
							{1, -2}
						]
					]
				],
				convLept,
				ToLowerCase[wgran]
			]
		]
	]

UnInit[] :=
	Block[{$Context = "TesseractTools`"},
		$UnInit[]
	]

(*
 * Return the recognized text
*)
RecognizeText[isFile_ : False] :=
	Block[{$Context = "TesseractTools`"},
		If[isFile,
			$RecognizeTextFile[],
			$RecognizeTextImage[]
		]
	]

(*
 * Returns a list of rectangles
*)
GetImageComponentsRectangle[] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				res
			},

			res = $GetImageComponentsBoundingBox[];

			res =
				Quiet[
					processRectangle[
						ToExpression[
							StringReplace[
								res,
								{"\n" | "\n\n" -> "", "\\" -> ""}
							]
						]
					]
				];

			If[Quiet[ListQ[res]],
				res,
				$Failed
			]
		]]

(*
 * Returns a list of strengths
*)
GetImageComponentsConfidence[] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				res
			},

			res = $GetImageComponentsConfidence[];

			res =
				Quiet[
					processConfidence[
						ToExpression[
							StringReplace[
								res,
								{"\n" | "\n\n" -> "", "\\" -> ""}
							]
						]
					]
				];

			If[Quiet[ListQ[res]],
				res,
				$Failed
			]
		]
	]

GetOrientation[] := Block[{$Context = "TesseractTools`"}, GetOrientationInfo[1]];
GetDirection[] := Block[{$Context = "TesseractTools`"}, GetOrientationInfo[2]];
GetOrder[] := Block[{$Context = "TesseractTools`"}, GetOrientationInfo[3]];
GetDeskewAngle[] := Block[{$Context = "TesseractTools`"}, GetOrientationInfo[4]];
GetOrientationInfo[mode_ : 0] :=
	Block[{$Context = "TesseractTools`"},
		Block[
			{
				res
			},

			res = $GetOrientation[mode];
			ToExpression[res]
		]
	]

(*****************************)
(*******Post processing*******)
(*****************************)
recognizeText[
	image_,
	originalImage_,
	imageQ_
	,
	level_,
	props_
	,
	masking_,
	prior_,
	engine_,
	tesseractOptions_,
	rotation_,
	useLeptonica_
] :=
	Block[{$Context = "TesseractTools`"},
		Quiet[Module[
			{
				tmpImage, tmpRes = {}, xDim, yDim, dims, properties, maskingTesseract, scale = 1.
			},

			tmpImage = image;
			properties = DeleteDuplicates[props];

			If[imageQ,
			(*Image case*)
				If[
					And[
						imageQ,
						Internal`RealValuedNumericQ[rotation] (*not documented*)
					],
					tmpImage = ImageRotate[tmpImage, rotation];
					dims = ImageDimensions[tmpImage]
					,
					dims = ImageDimensions[image]
				];

				(*
					On Windows x86 Tesseract has problem analyzing the layout
					for any image that has at least one dimension smaller than 10,
					and for some images that have at least one dimension smaller than 50.
					This numbers are get experimentally.

					Tesseract forums recommend the minimal dimension of the input image
					be not less than 70.
					RandomImage[1, {69, 69}] crashes, while RandomImage[1, {70, 70}] does not.

					There are 2 solutions to this.
					(1) Since the crash happens in textord.cpp, line 257, we could
						edit the source code and update the according function to
						stop analyzing the layout if is_blocks.empty() instead of
						asserting.
						Since the textord.cpp is called from inside of Tesseract,
						and is not directly accessed by us, there seem to be no way
						to predict whether the computation of block is going to be empty.
	
					(2) Resize all images that have at least one dimension less than 50,
						so that both dimensions are 50 or more.
				*)

				scale = N[1 / Min[dims / 70]];
				If[scale > 1,
					tmpImage = ImageResize[tmpImage, Scaled[scale]];
				];
				,
			(*File case*)
				If[
					Or[
						StringContainsQ[$SystemID, "MacOSX"],
						StringMatchQ[FileExtension[tmpImage], "gif", IgnoreCase -> True]
					],
					tmpImage = Quiet[Import[FindFile[tmpImage]]];
					If[!Image`PossibleImageQ[tmpImage],
						Return[$Failed]
						,
						If[ImageInformation[tmpImage, Transparency],
							tmpImage = AlphaBlend[tmpImage]
						];
						tmpImage = Image[ColorConvert[tmpImage, "Grayscale"], "Byte"]
					]
				];

				dims = Import[image, "ImageSize"]
			];

			xDim = First[dims];
			yDim = Last[dims];

			maskingTesseract = masking;
			maskingTesseract = maskingTesseract /. All -> "N/A";

			(*Initializing the engine once for all properties to be extracted*)
			(*The initialization takes ~20 ms*)
			Init[tmpImage, maskingTesseract, tesseractOptions, prior, (*engine*)"N/A", useLeptonica, ToLowerCase[level], yDim];

			(*Gather all the properties recognized like {{x, x1, x2}, {y, y1, y2}}*)
			AppendTo[
				tmpRes,
				# -> extractInfo[tmpImage, originalImage, #, level, yDim, maskingTesseract, scale]& /@ properties
			];

			tmpRes = First[tmpRes];

			UnInit[];

			Select[
				tmpRes,
				!MatchQ[Last[#],
					{} | {" " -> " "} | " " | "N/A"
				]&
			];

			(*If for some reason nothing is recognized*)
			If[
				Or[
					tmpRes === {},
					(*... or the extracted features have different length*)
					And[
						Length[tmpRes] > 1,
						! SameQ @@ (Length /@ (Flatten /@ Values[tmpRes]))
					]
				],
				tmpRes = $Failed;
				Return[Missing["NotRecognized"]]
			];

			(*Clean up for temp language folder*)
			If[Quiet[DirectoryQ[finalPath]],
				DeleteDirectory[finalPath, DeleteContents -> True]
			];

			Return[Association[tmpRes]];
		]]
	]

extractInfo[im_, originalImg_, recMode_, wgran_, yDim_, masking_, scale_] :=
	Block[{$Context = "TesseractTools`"},
		Which[
			StringMatchQ[recMode, "Text"],

			With[
				{tmp = processText[wgran, masking]},

				If[ListQ[tmp],
					If[tmp === {},
						{Missing["NotRecognized"]}
						,
						If[Length[tmp] === 1 && ListQ[First[tmp]],
							First[tmp],
							tmp
						]
					]
					,
					{tmp}
				]
			]
			,
			StringMatchQ[recMode, "BoundingBox", IgnoreCase -> True],
			processBBPerLevel[yDim, wgran, masking, scale]
			,
			StringMatchQ[recMode, "Image", IgnoreCase -> True],
			With[{bb = processBBPerLevel[yDim, wgran, masking, scale]},
				If[bb === {Missing["NotRecognized"]},
					bb
					,						
					ImageTrim[originalImg, #] & /@ List @@@ bb						
				]
			]
			,
			StringMatchQ[recMode, "Strength", IgnoreCase -> True],
			Module[
				{
					tmp = processConfidencePerLevel[wgran, masking],
					tmpText, tmpTextDims
				},

				tmp =
					If[ListQ[First[tmp]],
						First /@ tmp
						,
						tmp
					];

				If[ListQ[masking] && Length[masking] > 1,
					(*
						Since we do initial initialization with specified masks,
						confidence comes back as one big list of numbers, which
						needs to be reshaped. This is only valid if multiple
						rectangles are specified as regions of interest.
					*)
					tmpTextDims = Dimensions[extractInfo[im, originalImg, "Text", wgran, yDim, masking, scale]];
					tmp = ArrayReshape[tmp, tmpTextDims];
				];

				tmp
			]
		]
	]

postProcessResultString[text_] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{tmpTxt = text}
			,
			
			Which[StringQ[tmpTxt],
				If[tmpTxt === stringSplitter, tmpTxt = " "];
				tmpTxt =
					Quiet[
						StringSplit[
							tmpTxt, (*StringReplace[tmpTxt, "\n\n" -> "\n"],*)
							stringSplitter
						]
					];
				tmpTxt = StringTrim /@ tmpTxt;
				tmpTxt =
					If[Length[tmpTxt] == 1,
						First[tmpTxt],
						tmpTxt
					]
				,
				ListQ[tmpTxt],
				tmpTxt
				,
				True, Missing["NotRecognized"]
			]
		]
	]

processText[wgran_, masking_] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				recText, tmpRecText, shouldBeSplit = True
			},

			recText = RecognizeText[];

			If[
				Or[
					recText === $Failed,
					MatchQ[tmpRes, LibraryFunctionError[_, _]],
					StringTrim[ToString[recText]] === ""
				],
				Return[Missing["NotRecognized"]]
			];

			If[StringContainsQ[recText, stringSplitter],
				shouldBeSplit = False;
			];

			tmpRecText = postProcessResultString[recText];

			If[
				And[!TrueQ[shouldBeSplit],
					SameQ[wgran, "Automatic"],
					MatchQ[masking, "N/A" | {"N/A"}]
				],
				tmpRecText = StringTrim[StringJoin[StringJoin[#, "\n\n"] & /@ tmpRecText]];
			];

			Return[# & /@ (checkForList /@ tmpRecText)]
		]
	]

processConfidencePerLevel[wgran_, masking_] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				asc = GetImageComponentsConfidence[]
			},

			If[
				Or[
					asc === $Failed,
					asc === {},
					And[
						And[
							Length[asc] === 1,
							Length[asc[[1]]] > 0
						],
						StringTrim[ToString[asc[[1, 1]]]] === ""
					]
				]
				,
				Return[{Missing["NotRecognized"]}]
			];

			If[Quiet[!ListQ[asc]], asc = {asc}];

			Return[
				If[StringMatchQ[wgran, "Automatic"],
					If[
						And[
							Length[asc] > 1,
							MatchQ[masking, "N/A" | {"N/A"}]
						],
						First[asc],
						asc
					]
					,
					asc
				]
			]
		]
	]

processBBPerLevel[yDim_, wgran_, masking_, scale_] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				res, newY1, newY2
			},

			res = GetImageComponentsRectangle[];

			If[
				Or[
					res === $Failed,
					res === {},
					And[
						Length[res] === 1,
						Length[res[[1]]] > 0,
						StringTrim[ToString[res[[1, 1]]]] === ""
					]
				]
				,
				Return[{Missing["NotRecognized"]}]
			];

			If[Quiet[!ListQ[res]], res = {res}];

			res = 			
				If[StringMatchQ[wgran, "Automatic"],
					If[
						And[
							Length[res] > 1,
							MatchQ[masking, "N/A" | {"N/A"}]
						],

						{Rectangle[{Min[res[[All, 1, 1]]], Min[res[[All, 1, 2]]]}, {Max[res[[All, 2, 1]]], Max[res[[All, 2, 2]]]}]},
						res
					]
					,
					res
				];

			If[scale > 1,
				res = Replace[res, rect:_Rectangle :> Map[Function[elem, Round[elem / scale]], rect], {1}];
			];

			res = res /. x_Rectangle :> Rectangle[{x[[1, 1]], Abs[yDim - x[[2, 2]]]}, {x[[2, 1]], Abs[yDim - x[[1, 2]]]}];

			Return[res];
		]
	]

(*****************************)
(***********Options***********)
(*****************************)

tesseractOptsWL := Block[{$Context = "TesseractTools`"}, StringJoin@*System`Capitalize /@ StringSplit[tesseractOpts, "_"]]
allOpts := Block[{$Context = "TesseractTools`"}, AssociationThread[tesseractOptsWL, tesseractOpts]]
getOptionName[name_] := Block[{$Context = "TesseractTools`"}, Lookup[allOpts, name, Missing["NotAvailable"]]]

parseOptions[opts_] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				options
			},

			options = Cases[Flatten[{options}], _Rule | _RuleDelayed];
			If[Length[options] > 0,
				options =
					Quiet[
						Block[
							{
								rs =
									DeleteMissing[
										With[
											{tmp = getOptionName[options[[#, 1]]]},
											If[!MatchQ[tmp, _Missing],
												tmp -> options[[#, 2]],
												Missing["NotAvailable"]
											]
										]& /@ Range[Length@Keys[options]]
									]
							},
							If[Length[rs] >= 1,
								rs,
								"N/A"
							]
						]
					]
				,
				options = "N/A"
			];

			Return[options]
		]
	]

(*****************************)
(******Language parsing*******)
(*****************************)

InitializeLanguageFromFilePath[file_, itt_] :=
	Block[{$Context = "TesseractTools`"},
		Block[
			{
				lan, path
			},

			If[!Quiet[FileExistsQ[FindFile[file]]],
				Return[$Failed]
			];

			lan = Quiet[First[StringSplit[FileNameTake[file, -1], "."]]];

			If[StringQ[lan],
				If[itt === 0,
					LanguageName = lan,
					LanguageName = LanguageName <> "+" <> lan
				]
				,
				Return[$Failed]
			];

			path = Quiet[FileNameTake[file, {1, -2}]];

			If[FileNameTake[path, -1] === "tessdata",
				path = Quiet[FileNameTake[path, {1, -2}]]
			];

			If[StringQ[path],
				TrainDataPath = path,
				Return[$Failed]
			]
		]
	]

getTessdataIndex[] :=
	Block[{$Context = "TesseractTools`"},
		If[ValueQ[$tessdataIndex],
			$tessdataIndex,
			Quiet[
				PacletManager`Package`getPacletWithProgress["TesseractTrainedDataIndex"];
				$tessdataIndex = Get[PacletManager`PacletResource["TesseractTrainedDataIndex", "index.m"]]
			]
		]
	]

parseLanguagesTesseract[lanIn_, orlanIn_] :=
	Block[{$Context = "TesseractTools`"},
		Module[
			{
				lan, originalLanguage,
				wrongLanguages = {}, wrongLanguagesPos = {}, trainedDataPathsList = {},
				itt = 0, index = 0, lanPacletName = "", lanPacletNamePath = $Failed,
				layoutPath = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "TesseractTools", "LibraryResources", "tessdata"}]
			},

			lan = lanIn;
			originalLanguage = orlanIn;

			wrongLanguages = Flatten[Cases[lan, {$Failed, _}, Infinity][[All, 2]]];

			If[
				Head[#] === File,
				If[Or[
					!FileExistsQ[#],
					Length[
						StringCases[
							FileNameTake[#],
							StringExpression[StartOfString, name__, ".traineddata", EndOfString] :> name
						]
					] === 0
				],
					AppendTo[wrongLanguages, #]
				]
			] & /@ lan;

			If[ContainsAny[lan, {"jpn"}] && $SystemID === "Windows",
				lan = DeleteCases[lan, "jpn"];
				AppendTo[wrongLanguages, "Japanese"];
			];
			If[ContainsAny[lan, {"hin"}],
				lan = DeleteCases[lan, "hin"];
				AppendTo[wrongLanguages, "Hindi"];
			];
			If[ContainsAny[lan, {"ara"}],
				lan = DeleteCases[lan, "ara"];
				AppendTo[wrongLanguages, "Arabic"];
			];

			With[
				{
					wrongLanguagesPosFailedToParse = Position[lan, {$Failed, _}, Infinity],
					wrongLanguagesPosRemovedOrNotPresent = (Position[lan, #, Infinity]& /@ wrongLanguages)[[All, All, 1]]
				},

				wrongLanguagesPos = DeleteDuplicates[Join[wrongLanguagesPosFailedToParse, wrongLanguagesPosRemovedOrNotPresent]];

				lan = Flatten[DeleteCases[lan, {$Failed, _}, Infinity]];
				lan = DeleteCases[lan, Alternatives @@ wrongLanguages];

				originalLanguage = DeleteCases[originalLanguage, Alternatives @@ wrongLanguages | _EntityClass]
			];

			Switch[
				Length[wrongLanguages],
				0, _,
				1, Message[TextRecognize::langwrngx, First[wrongLanguages]],
				2, Message[TextRecognize::langwrngn, Row@Riffle[wrongLanguages, " and "]],
				_, Message[TextRecognize::langwrngn, Row@Riffle[wrongLanguages, Append[ConstantArray[", ", Length[wrongLanguages] - 2], ", and "]]]
			];

			If[lan === {}, Return[$Failed]];
			wrongLanguages = {};

			If[Head[#] === File,
				Block[{lpath = Quiet[FindFile[#]], lanTrim, tspath, tmppath},

					lanTrim = Quiet@First[
						StringCases[
							FileNameTake[lpath],
							StringExpression[StartOfString, name__, ".traineddata", EndOfString] :> name],
						$Failed
					];

					If[lpath =!= $Failed && !SameQ[FileNameTake[lpath, {-2}], "tessdata"],
						If[FileExistsQ[lpath],

							tspath = FileNameJoin[{$TemporaryDirectory, "tessdata"}];
							If[!FileExistsQ[tspath],
								tspath = CreateDirectory[tspath]
							];

							tmppath = FileNameJoin[{tspath, FileNameTake[lpath, -1]}];
							If[!FileExistsQ[tmppath],
								lpath = Quiet@CopyFile[lpath, tmppath],

								Quiet@DeleteFile[tmppath];
								lpath = Quiet@CopyFile[lpath, tmppath]
							]
							,
							AppendTo[wrongLanguages, If[
								ListQ[originalLanguage],
								First@originalLanguage[[First[Position[lan, #]]]],
								originalLanguage
							]
							]
						]
					];

					If[lpath =!= $Failed,
						AppendTo[trainedDataPathsList, lpath];
						If[InitializeLanguageFromFilePath[lpath, itt] === $Failed,
							AppendTo[wrongLanguages, If[
								ListQ[originalLanguage],
								First@originalLanguage[[First[Position[lan, #]]]],
								originalLanguage
							]
							]
						]
						,
						If[itt === 0, LanguageName = lanTrim, LanguageName = LanguageName <> "+" <> lanTrim]
					];

					itt = itt + 1
				]
				,
				If[!FileExistsQ[FileNameJoin[{layoutPath, # <> ".traineddata"}]] && Quiet[FindFile[#]] === $Failed,
					index = getTessdataIndex[];
					If[
						Head[index] === Association
						,
						lanPacletName = Lookup[index, #];
						If[StringQ[lanPacletName]
							,
							Quiet[
								PacletManager`Package`getPacletWithProgress[lanPacletName];
								lanPacletNamePath = PacletManager`PacletResource[lanPacletName, #]
							];

							If[!StringQ[lanPacletNamePath],
								AppendTo[wrongLanguages, If[
									ListQ[originalLanguage],
									First@originalLanguage[[First[Position[lan, #]]]],
									originalLanguage
								]
								](* A tessdata file for the name exists, but it couldn't be located, or installed, for some reason. *)
							]
							,
							AppendTo[wrongLanguages, If[
								ListQ[originalLanguage],
								First@originalLanguage[[First[Position[lan, #]]]],
								originalLanguage
							]
							];(* No tessdata file exists for that name. *)
							lanPacletNamePath = $Failed
						]
						,
						AppendTo[wrongLanguages, If[
							ListQ[originalLanguage],
							First@originalLanguage[[First[Position[lan, #]]]],
							originalLanguage
						]
						];(* TesseractTrainedDataIndex paclet not present and could not be installed. *)
						lanPacletNamePath = $Failed
					];
					AppendTo[trainedDataPathsList, lanPacletNamePath];
					If[lanPacletNamePath =!= $Failed,
						If[InitializeLanguageFromFilePath[lanPacletNamePath, itt] === $Failed ,
							AppendTo[wrongLanguages, If[
								ListQ[originalLanguage],
								First@originalLanguage[[First[Position[lan, #]]]],
								originalLanguage
							]
							]
						],
						If[itt === 0,
							LanguageName = #,
							LanguageName = LanguageName <> "+" <> #
						]
					]
					,

					AppendTo[trainedDataPathsList, FileNameJoin[{FileNameDrop[baseDir, -1], "tessdata", # <> ".traineddata"}]];
					If[Quiet[FindFile[#]] === $Failed,
						If[InitializeLanguageFromFilePath[FileNameJoin[{FileNameDrop[baseDir, -1], "tessdata", #}], itt] === $Failed ,
							AppendTo[wrongLanguages, If[
								ListQ[originalLanguage],
								First@originalLanguage[[First[Position[lan, #]]]],
								originalLanguage
							]
							]
						],
						If[itt === 0,
							LanguageName = #,
							LanguageName = LanguageName <> "+" <> #
						]
					];
				];

				itt = itt + 1
			] & /@ lan;

			trainedDataPathsList = DeleteCases[trainedDataPathsList, $Failed];
			Switch[
				Length[wrongLanguages],
				0, _,
				1, Message[TextRecognize::langinvx, First[wrongLanguages]],
				2, Message[TextRecognize::langinvn, Row@Riffle[wrongLanguages, " and "]],
				_, Message[TextRecognize::langinvn, Row@Riffle[wrongLanguages, Append[ConstantArray[", ", Length[wrongLanguages] - 2], ", and "]]]
			];
			If[trainedDataPathsList === {}, Return[$Failed]];

			finalPath = Quiet[
				If[!Equal @@ (FileNameTake[#, {1, -2}] & /@ trainedDataPathsList) (*two or more languages are not in the same directory*)
					,
					Block[{rnd, rndDir, tmp},
						rnd = RandomInteger[{1000000, 9999999}];
						rndDir = FileNameJoin[{$TemporaryDirectory, "$" <> ToString[rnd], "tessdata"}]; (*temporary directory for storing all the language files in same directory*)
						If[DirectoryQ[rndDir], DeleteDirectory[rndDir, DeleteContents -> True]];
						CreateDirectory[rndDir];
						CopyFile[#, FileNameJoin[{rndDir, FileNameTake[#, -1]}]]& /@ trainedDataPathsList;
						TrainDataPath = rndDir;
						rndDir
					]
				]
			];

			Return[lan]
		]
	]

(*****************************)
(************Enums************)
(*****************************)

parseLanguageTesseract[expr_, language_] :=
	Block[{$Context = "TesseractTools`"},
		Replace[
			expr,
			{
				File[_?StringQ] -> expr,
				"afrikaans"                                             -> "afr",
				"amharic"                                               -> "amh",
				"arabic"                                                -> "ara",
				"assamese"                                              -> "asm",
				"azerbaijanicyrillic" | "cyrillicazerbaijani"           -> "aze_cyrl",
				"southazerbaijani" | "northazerbaijani" | "azerbaijani" -> "aze",
				"belarusian"                                            -> "bel",
				"bengali"                                               -> "ben",
				"tibetan"                                               -> "bod",
				"bosnian"                                               -> "bos",
				"bulgarian"                                             -> "bul",
				"catalan" | "valencian"                                 -> "cat",
				"cebuano"                                               -> "ceb",
				"czech"                                                 -> "ces",
				"mandarin" | "simplifiedchinese" |
	           	"chinesesimplified"                                     -> "chi_sim",
				"TraditionalChinese" | "chinesetraditional" 			-> "chi_tra",
				"chinese" 												-> {"chi_sim", "chi_tra"},
				"cherokee"                                              -> "chr",
				"welsh"                                                 -> "cym",
				"danishfraktur" | "frakturdanish"                       -> "dan_frak",
				"danish"                                                -> "dan",
				"germanfraktur" | "frakturgerman" | "fraktur"           -> "deu_frak",
				"german"                                                -> "deu",
				"dzongkha"                                              -> "dzo",
				"greekmodern" | "moderngreek" | "greek"                 -> "ell",
				"english" | Automatic                                   -> "eng",
				"englishmiddle" | "middleenglish"                       -> "enm",
				"esperanto"                                             -> "epo",
				"equation" | "math"                                     -> "equ",
				"estonian"                                              -> "est",
				"basque"                                                -> "eus",
				"persian" | "farsi"                                     -> "fas",
				"finnish"                                               -> "fin",
				"french"                                                -> "fra",
				"frankish"                                              -> "frk",
				"frenchMiddle" | "middlefrench"                         -> "frm",
				"irish"                                                 -> "gle",
				"galician"                                              -> "glg",
				"greekancient" | "ancientgreek"                         -> "grc",
				"gujarati"                                              -> "guj",
				"haitiancreolefrench" | "haitian" | "haitiancreole"     -> "hat",
				"hebrew"                                                -> "heb",
				"hindi"                                                 -> "hin",
				"croatian"                                              -> "hrv",
				"hungarian"                                             -> "hun",
				"inuktitut"                                             -> "iku",
				"indonesian"                                            -> "ind",
				"icelandic"                                             -> "isl",
				"italianold" | "olditalian"                             -> "ita_old",
				"italian"                                               -> "ita",
				"javanese"                                              -> "jav",
				"japanese"                                              -> "jpn",
				"kannada"                                               -> "kan",
				"georgianold" | "oldgeorgian"                           -> "kat_old",
				"georgian"                                              -> "kat",
				"kazakh"                                                -> "kaz",
				"khmer"                                                 -> "khm",
				"kirghiz"                                               -> "kir",
				"korean"                                                -> "kor",
				"kurdish"                                               -> "kur",
				"lao"                                                   -> "lao",
				"latin"                                                 -> "lat",
				"latvian"                                               -> "lav",
				"lithuanian"                                            -> "lit",
				"malayalam"                                             -> "mal",
				"marathi"                                               -> "mar",
				"macedonian"                                            -> "mkd",
				"maltese"                                               -> "mlt",
				"malay"                                                 -> "msa",
				"burmese"                                               -> "mya",
				"nepali"                                                -> "nep",
				"dutch"                                                 -> "nld",
				"norwegian"                                             -> "nor",
				"oriya"                                                 -> "ori",
				"punjabi"                                               -> "pan",
				"polish"                                                -> "pol",
				"portuguese"                                            -> "por",
				"pushto"                                                -> "pus",
				"romanian"                                              -> "ron",
				"russian"                                               -> "rus",
				"sanskrit"                                              -> "san",
				"sinhala" | "sinhalese"                                 -> "sin",
				"slovakFraktur"                                         -> "slk_frak",
				"slovak"                                                -> "slk",
				"slovenian"                                             -> "slv",
				"spanishold" | "oldspanish"                             -> "spa_old",
				"spanish" | "castilian"                                 -> "spa",
				"albanianarbereshe" | "albanian" |
				"albanianarb\[EDoubleDot]resh\[EDoubleDot]" |
                "albanianarvanitika" | "albaniangheg" | "albaniantosk"  -> "sqi",
				"serbianlatin" | "latinserbian"                         -> "srp_latn",
				"serbian"                                               -> "srp",
				"swahili"                                               -> "swa",
				"swedish"                                               -> "swe",
				"syriac"                                                -> "syr",
				"tamil"                                                 -> "tam",
				"telugu"                                                -> "tel",
				"tajik"                                                 -> "tgk",
				"tagalog"                                               -> "tgl",
				"thai"                                                  -> "tha",
				"tigrinya"                                              -> "tir",
				"turkish"                                               -> "tur",
				"uighur" | "uyghur"                                     -> "uig",
				"ukrainian"                                             -> "ukr",
				"urdu"                                                  -> "urd",
				"uzbekcyrillic" | "cyrillicuzbek"                       -> "uzb_cyrl",
				"uzbeknorthern" | "uzbek"                               -> "uzb",
				"vietnamese"                                            -> "vie",
				"yiddish"                                               -> "yid",
				_                                                       :> {$Failed, language}
			}
		]
	]

parseSegmentation[segmentation_] :=
	Block[{$Context = "TesseractTools`"},
		Replace[
			segmentation,
			{
				"SingleColumnVariableSize" | "Column"   -> "single_column",         (*Assume a single column of text of variable sizes. *)
				"SingleUniformBlock" | "Block"          -> "single_block",          (*Assume a single uniform block of text.*)
				"SingleLine" | "Line"                   -> "single_line",           (*Treat the image as a single text line.*)
				"SingleWord" | "Word"                   -> "single_word",           (*Treat the image as a single word.*)
				"SingleCharacter" | "Character"         -> "single_char",           (*Treat the image as a single character.*)
				"OSDOnly"                               -> "osd_only",              (*Orientation and script detection only.*)
				"AutoOnly"                              -> "auto_only",             (*Automatic page segmentation, but no OSD, or OCR.*)
				"Auto"                                  -> "auto",                  (*Fully automatic page segmentation, but no OSD.*)
				"AutoOSD"                               -> "auto_osd",              (*Automatic page segmentation with orientation and script detection. (OSD)*)
				"SingleBlockVertical" | "VerticalBlock" -> "sigle_block_vert_text", (*Assume a single uniform block of vertically aligned text.*)
				"SparseText"                            -> "sparse_text",           (*Find as much text as possible in no particular order.*)
				"SparseTextOSD"                         -> "sparse_text_osd",       (*Sparse text with orientation and script det.*)
				"RawLine"                               -> "raw_line",              (*Treat the image as a single text line, bypassing hacks that are Tesseract-specific.*)
				"Count"                                 -> "count",                 (*Number of enum entries.*)
				_                                       -> "N/A"
			}
		]
	]

parseEngineMode[engMode_] :=
	Block[{$Context = "TesseractTools`"},
		Replace[
			engMode,
			{
				"Tesseract" |
	            "TesseractOnly"         -> "tesseract_only"
				,
				"Cube" |
	            "CubeOnly"              -> "cube_only"
				,
				"TesseractCubeCombined" -> "tesseract_cube_combined"
				,
				_                       -> "N/A"
			}
		]
	]

(*****************************)
(******Additional lists*******)
(*****************************)

tesseractOpts := {"ambigs_debug_level", "applybox_debug", "applybox_exposure_pattern", "applybox_learn_chars_and_char_frags_mode", "applybox_learn_ngrams_mode", "applybox_page", "assume_fixed_pitch_char_segment",
	"bestrate_pruning_factor", "bidi_debug", "bland_unrej", "certainty_scale", "chop_center_knob", "chop_debug", "chop_enable", "chop_good_split", "chop_inside_angle", "chop_min_outline_area",
	"chop_min_outline_points", "chop_ok_split", "chop_overlap_knob", "chop_same_distance", "chop_sharpness_knob", "chop_split_dist_knob", "chop_split_length", "chop_vertical_creep", "chop_width_change_knob",
	"chop_x_y_weight", "chs_leading_punct", "chs_trailing_punct1", "chs_trailing_punct2", "classify_adapt_feature_threshold", "classify_adapt_proto_threshold", "classify_bln_numeric_mode",
	"classify_character_fragments_garbage_certainty_threshold", "classify_char_norm_range", "classify_class_pruner_multiplier", "classify_class_pruner_threshold", "classify_cp_angle_pad_loose",
	"classify_cp_angle_pad_medium", "classify_cp_angle_pad_tight", "classify_cp_cutoff_strength", "classify_cp_end_pad_loose", "classify_cp_end_pad_medium", "classify_cp_end_pad_tight", "classify_cp_side_pad_loose",
	"classify_cp_side_pad_medium", "classify_cp_side_pad_tight", "classify_debug_character_fragments", "classify_debug_level", "classify_enable_adaptive_debugger", "classify_enable_adaptive_matcher",
	"classify_enable_learning", "classify_font_name", "classify_integer_matcher_multiplier", "classify_learn_debug_str", "classify_learning_debug_level", "classify_max_norm_scale_x", "classify_max_norm_scale_y",
	"classify_max_slope", "classify_min_norm_scale_x", "classify_min_norm_scale_y", "classify_min_slope", "classify_misfit_junk_penalty", "classify_norm_adj_curl", "classify_norm_adj_midpoint", "classify_norm_method",
	"classify_num_cp_levels", "classify_pico_feature_length", "classify_pp_angle_pad", "classify_pp_end_pad", "classify_pp_side_pad", "classify_radius_gyr_max_exp", "classify_radius_gyr_max_man",
	"classify_radius_gyr_min_exp", "classify_radius_gyr_min_man", "classify_save_adapted_templates", "classify_training_file", "classify_use_pre_adapted_templates", "conflict_set_I_l_1", "crunch_accept_ok",
	"crunch_debug", "crunch_del_cert", "crunch_del_high_word", "crunch_del_low_word", "crunch_del_max_ht", "crunch_del_min_ht", "crunch_del_min_width", "crunch_del_rating", "crunch_early_convert_bad_unlv_chs",
	"crunch_early_merge_tess_fails", "crunch_include_numerals", "crunch_leave_accept_strings", "crunch_leave_lc_strings", "crunch_leave_ok_strings", "crunch_leave_uc_strings", "crunch_long_repetitions",
	"crunch_poor_garbage_cert", "crunch_poor_garbage_rate", "crunch_pot_garbage", "crunch_pot_indicators", "crunch_pot_poor_cert", "crunch_pot_poor_rate", "crunch_rating_max", "crunch_small_outlines_size",
	"crunch_terrible_garbage", "crunch_terrible_rating", "cube_debug_level", "dawg_debug_level", "debug_acceptable_wds", "debug_file", "debug_fix_space_level", "debug_x_ht_level", "devanagari_split_debugimage",
	"devanagari_split_debuglevel", "disable_character_fragments", "doc_dict_certainty_threshold", "doc_dict_enable", "doc_dict_pending_threshold", "docqual_excuse_outline_errs", "edges_boxarea", "edges_childarea",
	"edges_children_count_limit", "edges_children_fix", "edges_children_per_grandchild", "edges_debug", "edges_max_children_layers", "edges_max_children_per_outline", "edges_maxedgelength", "edges_min_nonhole",
	"edges_patharea_ratio", "edges_use_new_outline_complexity", "editor_dbwin_height", "editor_dbwin_name", "editor_dbwin_width", "editor_dbwin_xpos", "editor_dbwin_ypos", "editor_debug_config_file",
	"editor_image_blob_bb_color", "editor_image_menuheight", "editor_image_text_color", "editor_image_win_name", "editor_image_word_bb_color", "editor_image_xpos", "editor_image_ypos", "editor_word_height",
	"editor_word_name", "editor_word_width", "editor_word_xpos", "editor_word_ypos", "enable_new_segsearch", "equationdetect_save_bi_image", "equationdetect_save_merged_image", "equationdetect_save_seed_image",
	"equationdetect_save_spt_image", "file_type", "fixsp_done_mode", "fixsp_non_noise_limit", "fixsp_small_outlines_size", "force_word_assoc", "fragments_debug", "fragments_guide_chopper", "fx_debugfile",
	"gapmap_big_gaps", "gapmap_debug", "gapmap_no_isolated_quanta", "gapmap_use_ends", "heuristic_max_char_wh_ratio", "heuristic_segcost_rating_base", "heuristic_weight_rating", "heuristic_weight_seamcut",
	"heuristic_weight_width", "hyphen_debug_level", "il1_adaption_test", "image_default_resolution", "interactive_display_mode", "language_model_debug_level", "language_model_fixed_length_choices_depth",
	"language_model_min_compound_length", "language_model_ngram_nonmatch_score", "language_model_ngram_on", "language_model_ngram_order", "language_model_ngram_scale_factor",
	"language_model_ngram_space_delimited_language", "language_model_ngram_use_only_first_uft8_step", "language_model_penalty_case", "language_model_penalty_chartype", "language_model_penalty_font",
	"language_model_penalty_increment", "language_model_penalty_non_dict_word", "language_model_penalty_non_freq_dict_word", "language_model_penalty_punc", "language_model_penalty_script",
	"language_model_penalty_spacing", "language_model_use_sigmoidal_certainty", "language_model_viterbi_list_max_num_prunable", "language_model_viterbi_list_max_size", "load_bigram_dawg", "load_fixed_length_dawgs",
	"load_freq_dawg", "load_number_dawg", "load_punc_dawg", "load_system_dawg", "load_unambig_dawg", "matcher_avg_noise_size", "matcher_bad_match_pad", "matcher_clustering_max_angle_delta", "matcher_debug_flags",
	"matcher_debug_level", "matcher_debug_separate_windows", "matcher_good_threshold", "matcher_great_threshold", "matcher_min_examples_for_prototyping", "matcher_perfect_threshold", "matcher_permanent_classes_min",
	"matcher_rating_margin", "matcher_sufficient_examples_for_prototyping", "max_permuter_attempts", "max_viterbi_list_size", "m_data_sub_dir", "merge_fragments_in_matrix", "min_orientation_margin",
	"min_sane_x_ht_pixels", "ngram_permuter_activated", "numeric_punctuation", "ocr_devanagari_split_strategy", "ok_repeated_ch_non_alphanum_wds", "oldbl_corrfix", "oldbl_dot_error_size", "oldbl_holed_losscount",
	"oldbl_xhfix", "oldbl_xhfract", "outlines_2", "outlines_odd", "output_ambig_words_file", "pageseg_devanagari_split_strategy", "paragraph_debug_level", "permute_chartype_word", "permute_debug",
	"permute_fixed_length_dawg", "permute_only_top", "permute_script_word", "pitsync_fake_depth", "pitsync_joined_edge", "pitsync_linear_version", "pitsync_offset_freecut_fraction", "poly_debug",
	"poly_wide_objects_better", "prioritize_division", "quality_blob_pc", "quality_char_pc", "quality_min_initial_alphas_reqd", "quality_outline_pc", "quality_rej_pc", "quality_rowrej_pc", "rating_scale",
	"rej_1Il_trust_permuter_type", "rej_1Il_use_dict_word", "rej_alphas_in_number_perm", "rej_trust_doc_dawg", "rej_use_good_perm", "rej_use_sensible_wd", "rej_use_tess_accepted", "rej_use_tess_blanks",
	"rej_whole_of_mostly_reject_word_fract", "repair_unchopped_blobs", "save_alt_choices", "save_blob_choices", "save_doc_words", "save_raw_choices", "segment_adjust_debug", "segment_debug",
	"segment_nonalphabetic_script", "segment_penalty_dict_case_bad", "segment_penalty_dict_case_ok", "segment_penalty_dict_frequent_word", "segment_penalty_dict_nonword", "segment_penalty_garbage",
	"segment_penalty_ngram_best_choice", "segment_reward_chartype", "segment_reward_ngram_best_choice", "segment_reward_script", "segment_segcost_rating", "segsearch_debug_level", "segsearch_max_char_wh_ratio",
	"segsearch_max_f", "segsearch_max_futile_classifications", "segsearch_max_pain_points", "speckle_large_max_size", "speckle_large_penalty", "speckle_small_certainty", "speckle_small_penalty",
	"stopper_allowable_character_badness", "stopper_ambiguity_threshold_gain", "stopper_ambiguity_threshold_offset", "stopper_certainty_per_char", "stopper_debug_level", "stopper_no_acceptable_choices",
	"stopper_nondict_certainty_base", "stopper_phase2_certainty_rejection_offset", "stopper_smallword_size", "suspect_accept_rating", "suspect_constrain_1Il", "suspect_level", "suspect_rating_per_ch",
	"suspect_short_words", "suspect_space_level", "tess_bn_matching", "tess_cn_matching", "tessdata_manager_debug_level", "tessedit_adaption_debug", "tessedit_adapt_to_char_fragments", "tessedit_ambigs_training",
	"tessedit_bigram_debug", "tessedit_certainty_threshold", "tessedit_char_blacklist", "tessedit_char_whitelist", "tessedit_class_miss_scale", "tessedit_consistent_reps", "tessedit_create_boxfile",
	"tessedit_create_hocr", "tessedit_debug_block_rejection", "tessedit_debug_doc_rejection", "tessedit_debug_fonts", "tessedit_debug_quality_metrics", "tessedit_display_outwords", "tessedit_dont_blkrej_good_wds",
	"tessedit_dont_rowrej_good_wds", "tessedit_dump_choices", "tessedit_dump_pageseg_images", "tessedit_enable_bigram_correction", "tessedit_enable_doc_dict", "tessedit_fix_fuzzy_spaces", "tessedit_fix_hyphens",
	"tessedit_flip_0O", "tessedit_good_doc_still_rowrej_wd", "tessedit_good_quality_unrej", "tessedit_image_border", "tessedit_init_config_only", "tessedit_load_sublangs", "tessedit_lower_flip_hyphen",
	"tessedit_make_boxes_from_boxes", "tessedit_matcher_log", "tessedit_minimal_rejection", "tessedit_minimal_rej_pass1", "tessedit_ocr_engine_mode", "tessedit_ok_mode", "tessedit_override_permuter",
	"tessedit_page_number", "tessedit_pageseg_mode", "tessedit_prefer_joined_punct", "tessedit_preserve_blk_rej_perfect_wds", "tessedit_preserve_min_wd_len", "tessedit_preserve_row_rej_perfect_wds",
	"tessedit_redo_xheight", "tessedit_reject_bad_qual_wds", "tessedit_reject_block_percent", "tessedit_reject_doc_percent", "tessedit_rejection_debug", "tessedit_reject_mode", "tessedit_reject_row_percent",
	"tessedit_resegment_from_boxes", "tessedit_resegment_from_line_boxes", "tessedit_row_rej_good_docs", "tessedit_single_match", "tessedit_tess_adaption_mode", "tessedit_tess_adapt_to_rejmap",
	"tessedit_test_adaption", "tessedit_test_adaption_mode", "tessedit_train_from_boxes", "tessedit_training_tess", "tessedit_truncate_wordchoice_log", "tessedit_unrej_any_wd", "tessedit_upper_flip_hyphen",
	"tessedit_use_reject_spaces", "tessedit_whole_wd_rej_row_percent", "tessedit_word_for_word", "tessedit_write_block_separators", "tessedit_write_images", "tessedit_write_params_to_file", "tessedit_write_rep_codes",
	"tessedit_write_unlv", "tessedit_zero_kelvin_rejection", "tessedit_zero_rejection", "test_pt", "test_pt_x", "test_pt_y", "textord_all_prop", "textord_ascheight_mode_fraction", "textord_ascx_ratio_max",
	"textord_ascx_ratio_min", "textord_balance_factor", "textord_biased_skewcalc", "textord_blob_size_bigile", "textord_blob_size_smallile", "textord_blockndoc_fixed", "textord_blocksall_fixed",
	"textord_blocksall_prop", "textord_blocksall_testing", "textord_chopper_test", "textord_chop_width", "textord_debug_baselines", "textord_debug_block", "textord_debug_bugs", "textord_debug_images",
	"textord_debug_pitch_metric", "textord_debug_pitch_test", "textord_debug_printable", "textord_debug_tabfind", "textord_debug_xheights", "textord_descheight_mode_fraction", "textord_descx_ratio_max",
	"textord_descx_ratio_min", "textord_disable_pitch_test", "textord_dotmatrix_gap", "textord_dump_table_images", "textord_equation_detect", "textord_excess_blobsize", "textord_expansion_factor",
	"textord_fast_pitch_test", "textord_fix_makerow_bug", "textord_fix_xheight_bug", "textord_force_make_prop_words", "textord_fp_chop_error", "textord_fp_chopping", "textord_fp_chop_snap", "textord_fpiqr_ratio",
	"textord_fp_min_width", "textord_heavy_nr", "textord_interpolating_skew", "textord_linespace_iqrlimit", "textord_lms_line_trials", "textord_max_blob_overlaps", "textord_max_noise_size", "textord_max_pitch_iqr",
	"textord_min_blob_height_fraction", "textord_min_blobs_in_row", "textord_min_linesize", "textord_minxh", "textord_min_xheight", "textord_new_initial_xheight", "textord_noise_area_ratio", "textord_noise_debug",
	"textord_noise_rejrows", "textord_noise_rejwords", "textord_noise_sizefraction", "textord_noise_sncount", "textord_noise_translimit", "textord_no_rejects", "textord_occupancy_threshold", "textord_ocropus_mode",
	"textord_old_baselines", "textord_oldbl_debug", "textord_oldbl_jumplimit", "textord_oldbl_merge_parts", "textord_oldbl_paradef", "textord_oldbl_split_splines", "textord_old_xheight", "textord_overlap_x",
	"textord_parallel_baselines", "textord_pitch_cheat", "textord_pitch_range", "textord_pitch_rowsimilarity", "textord_pitch_scalebigwords", "textord_projection_scale", "textord_really_old_xheight",
	"textord_restore_underlines", "textord_show_blobs", "textord_show_boxes", "textord_show_expanded_rows", "textord_show_final_blobs", "textord_show_final_rows", "textord_show_fixed_cuts", "textord_show_fixed_words",
	"textord_show_initial_rows", "textord_show_initial_words", "textord_show_new_words", "textord_show_page_cuts", "textord_show_parallel_rows", "textord_show_row_cuts", "textord_show_tables",
	"textord_single_height_mode", "textord_skew_ile", "textord_skew_lag", "textord_skewsmooth_offset", "textord_skewsmooth_offset2", "textord_space_size_is_variable", "textord_spacesize_ratiofp",
	"textord_spacesize_ratioprop", "textord_spline_medianwin", "textord_spline_minblobs", "textord_spline_outlier_fraction", "textord_spline_shift_fraction", "textord_straight_baselines",
	"textord_tabfind_aligned_gap_fraction", "textord_tabfind_find_tables", "textord_tabfind_force_vertical_text", "textord_tabfind_only_strokewidths", "textord_tabfind_show_blocks", "textord_tabfind_show_color_fit",
	"textord_tabfind_show_columns", "textord_tabfind_show_finaltabs", "textord_tabfind_show_images", "textord_tabfind_show_initial_partitions", "textord_tabfind_show_initialtabs", "textord_tabfind_show_partitions",
	"textord_tabfind_show_reject_blobs", "textord_tabfind_show_strokewidths", "textord_tabfind_show_vlines", "textord_tabfind_vertical_horizontal_mix", "textord_tabfind_vertical_text",
	"textord_tabfind_vertical_text_ratio", "textord_tablefind_recognize_tables", "textord_tablefind_show_mark", "textord_tablefind_show_stats", "textord_tabvector_vertical_box_ratio",
	"textord_tabvector_vertical_gap_fraction", "textord_test_landscape", "textord_test_mode", "textord_testregion_bottom", "textord_testregion_left", "textord_testregion_right", "textord_testregion_top",
	"textord_test_x", "textord_test_y", "textord_underline_offset", "textord_underline_threshold", "textord_underline_width", "textord_use_cjk_fp_model", "textord_width_limit", "textord_width_smooth_factor",
	"textord_words_default_maxspace", "textord_words_default_minspace", "textord_words_default_nonspace", "textord_words_def_fixed", "textord_words_definite_spread", "textord_words_def_prop",
	"textord_words_initial_lower", "textord_words_initial_upper", "textord_words_maxspace", "textord_words_minlarge", "textord_words_min_minspace", "textord_words_pitchsd_threshold", "textord_wordstats_smooth_factor",
	"textord_words_veto_power", "textord_words_width_ile", "textord_xheight_error_margin", "textord_xheight_mode_fraction", "tosp_all_flips_fuzzy", "tosp_block_use_cert_spaces", "tosp_debug_level",
	"tosp_dont_fool_with_small_kerns", "tosp_enough_small_gaps", "tosp_enough_space_samples_for_median", "tosp_few_samples", "tosp_flip_caution", "tosp_flip_fuzz_kn_to_sp", "tosp_flip_fuzz_sp_to_kn",
	"tosp_force_wordbreak_on_punct", "tosp_fuzzy_kn_fraction", "tosp_fuzzy_limit_all", "tosp_fuzzy_space_factor", "tosp_fuzzy_space_factor1", "tosp_fuzzy_space_factor2", "tosp_fuzzy_sp_fraction", "tosp_gap_factor",
	"tosp_ignore_big_gaps", "tosp_ignore_very_big_gaps", "tosp_improve_thresh", "tosp_init_guess_kn_mult", "tosp_init_guess_xht_mult", "tosp_kern_gap_factor1", "tosp_kern_gap_factor2", "tosp_kern_gap_factor3",
	"tosp_large_kerning", "tosp_max_sane_kn_thresh", "tosp_min_sane_kn_sp", "tosp_narrow_aspect_ratio", "tosp_narrow_blobs_not_cert", "tosp_narrow_fraction", "tosp_near_lh_edge", "tosp_old_sp_kn_th_factor",
	"tosp_old_to_bug_fix", "tosp_old_to_constrain_sp_kn", "tosp_old_to_method", "tosp_only_small_gaps_for_kern", "tosp_only_use_prop_rows", "tosp_only_use_xht_gaps", "tosp_pass_wide_fuzz_sp_to_context",
	"tosp_recovery_isolated_row_stats", "tosp_redo_kern_limit", "tosp_rep_space", "tosp_row_use_cert_spaces", "tosp_row_use_cert_spaces1", "tosp_rule_9_test_punct", "tosp_sanity_method", "tosp_short_row",
	"tosp_silly_kn_sp_gap", "tosp_stats_use_xht_gaps", "tosp_table_fuzzy_kn_sp_ratio", "tosp_table_kn_sp_ratio", "tosp_table_xht_sp_ratio", "tosp_threshold_bias1", "tosp_threshold_bias2", "tosp_use_pre_chopping",
	"tosp_use_xht_gaps", "tosp_wide_aspect_ratio", "tosp_wide_fraction", "unlv_tilde_crunching", "unrecognised_char", "use_ambigs_for_adaption", "use_definite_ambigs_for_classifier", "use_new_state_cost",
	"use_only_first_uft8_step", "user_patterns_suffix", "user_words_suffix", "wordrec_blob_pause", "wordrec_debug_blamer", "wordrec_debug_level", "wordrec_display_all_blobs", "wordrec_display_all_words",
	"wordrec_display_segmentations", "wordrec_display_splits", "wordrec_enable_assoc", "wordrec_no_block", "wordrec_num_seg_states", "wordrec_run_blamer", "wordrec_worst_state", "words_default_fixed_limit",
	"words_default_fixed_space", "words_default_prop_nonspace", "words_initial_lower", "words_initial_upper", "word_to_debug", "word_to_debug_lengths", "x_ht_acceptance_tolerance", "x_ht_min_change"};


End[]
EndPackage[]