(* ::Package:: *)
Needs["IntegratedServices`"];

System`StringRiffle;
System`ServiceConnect;
System`ServiceObject;
System`URLRead;
System`Audio;
System`AudioStream;
System`CloudConnect;
ServiceConnections`ServiceConnections;
ServiceConnections`Private`$authenticatedservices;

(Unprotect[#];
 Clear[#])& /@ {
	System`WebAudioSearch,
	Audio`WebAudioSearchInformation
};

System`WebAudioSearch::invarg = "`arg` is not a valid search criterion."
System`WebAudioSearch::invprop = "`prop` is not a supported property for `method`."
System`WebAudioSearch::invvalue = "`value` is not a valid value for the search criterion `arg`."
System`WebAudioSearch::timeout = "Connection timed out.  Try again."
System`WebAudioSearch::invalidserviceobj = "ServiceConnect failed."
System`WebAudioSearch::invformat = "`format` is not a supported format."
System`WebAudioSearch::invservice = "`arg` is not a supported service."
System`WebAudioSearch::offline = "The Wolfram Language is currently configured not to use the Internet.";
System`WebAudioSearch::notauth = "WebAudioSearch requests are only valid when authenticated.  Try authenticating again.";
System`WebAudioSearch::nofe = "WebAudioSearch gallery view requires a front end.";
System`WebAudioSearch::longquery = "Query string is too long.";
System`WebAudioSearch::interr = "An internal error occurred."

BeginPackage["WebAudioSearch`"];
Begin["`Private`"];

$connTimeout = 30;
$thumbnailSize = 40;
$buttonSize = 18;
$buttonIconColor = Hue[.582, .644, .792, 0.5];
$buttonIconSelectedColor = Hue[.582, .644, .792, 0.8];
$durationTextColor = Hue[.582, .644, .792, 0.8];
$bottomBackground = RGBColor["#f6f6f6"];
$userFontColor = RGBColor["#8b8b8b"];
$licenseIconSpacing = 2;
$defaultWindowSize = 420;
$inactiveWindowSize = 320;
$defaultPaneSize = $defaultWindowSize - 14;
$services = {"Freesound"};
$formats = {"Dataset", "Samples", "Gallery", "GalleryMinimal", "TitleHyperlinks", "RandomSample"};
$supportedProperty = <|
	"Freesound" -> Sort@{"Player", "TitleHyperlink", "Title", "Duration", "SampleRate", "Channels", "PageLink", "SampleLink", "License", "Tags", "Username", "Sample"},
	"SoundCloud" -> Sort@{"Player", "TitleHyperlink", "Title", "Duration", "PageLink", "SampleLink", "License"}
|>;

$defaultFormat = "Dataset";
$defaultFormatNoFE = "TitleHyperlinks";

$propertyDataset = <|
	"Freesound" -> {"Player", "TitleHyperlink", "Title", "Duration", "SampleRate", "Channels", "PageLink", "SampleLink", "License"},
	"SoundCloud" -> {"Player", "TitleHyperlink", "Title", "Duration", "PageLink", "SampleLink", "License"}
|>;

$propertySamples = {"Sample", "SampleLink", "Title", "Username", "License", "PageLink"};
$propertyRandomSample = {"SampleLink", "Title", "Username", "License", "PageLink"};
$propertyTitleHyperlinks = {"TitleHyperlink", "Title", "PageLink"};
$propertyGallery = Sort @ {"SampleLink", "Username", "Title", "Duration", "License", "PageLink", "Image"};
$propertyGalleryMinimal = Sort @ {"SampleLink", "Username", "Title", "Duration", "License", "PageLink"};

$criterion = {"Duration", "BeatPerMinute", "SampleRate", "Channels", "Tag"};
$realCriterion = {"Duration"};
$integerCriterion = {"BeatPerMinute", "SampleRate", "Channels"};

$propertyConversion = <|
	"Freesound" -> {
		"image" -> "images",
		"title" -> "name",
		"beatperminute" -> "rhythm.bpm",
		"pagelink" -> "url",
		"samplelink" -> "previews"},
	"SoundCloud" -> {
		"image" -> "artwork_url",
		"pagelink" -> "permalink_url",
		"beatperminute" -> "bpm",
		"username" -> "user",
		"samplelink" -> "stream_url"}
|>;

$compositeProperty = {
	"TitleHyperlink" -> {"TitleHyperlink", "Title", "PageLink"},
	"Sample" -> {"Sample", "SampleLink", "Title", "Username", "License", "PageLink"},
	"Player" -> {"Player", "SampleLink"}

};

$criterionConversion = <|
	"Freesound" -> {
		"beatperminute" -> "rhythm.bpm"},
	"SoundCloud" -> {
	}
|>;

$logo = <|
	"Freesound" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Freesound-Logo.png"]]
|>

$logoWatermark = <|
	"Freesound" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Freesound-Watermark.png"]]
|>

$logoLicense = <|
	"CC" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC.png"]],
	"By" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-By.png"]],
	"Nc" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-Nc.png"]],
	"Zero" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-Zero.png"]],
	"Remix" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-Zero.png"]],
	"NcEU" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-NcEU.png"]],
	"NcJP" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-NcJP.png"]],
	"Share" -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "CC-Share.png"]]
|>

$result = <||>;
$resultCurrentPage = <||>;

(*WebAudioSearchInformation*)

Audio`WebAudioSearchInformation[service:Alternatives@@$services] := Association[
	"Properties" -> $supportedProperty[service]
]

Audio`WebAudioSearchInformation[] := $services

(*WebAudioSearch*)

Options[WebAudioSearch] = SortBy[#, First]& @{
	MaxItems -> 10
};

System`WebAudioSearch[___] := (getWebAudioSearchFailure["offline", <||>]) /; (!PacletManager`$AllowInternet)

System`WebAudioSearch[args___] := With[{connected = CloudConnect[]},
	If[System`$CloudConnected,
		System`WebAudioSearch[args]
		,
		Message[System`WebAudioSearch::notauth];
		connected
	]
] /; (!System`$CloudConnected)

System`WebAudioSearch[s___] :=
Block[
	{a, r},
	a = System`Private`Arguments[WebAudioSearch[s], {1, 3}];
	(If[Developer`UseFrontEnd[CurrentValue["UserInteractionEnabled"]],
		PrintTemporary[ProgressIndicator[Appearance -> "Necklace"]]];
	r
	) /; a =!={} && (r = iWebAudioSearch[Sequence@@a]) =!= $Failed
];

Options[iWebAudioSearch] = Options[WebAudioSearch];

iWebAudioSearch[{arg___}, opts:OptionsPattern[]] := $Failed

iWebAudioSearch[{arg_}, opts:OptionsPattern[]] :=
iWebAudioSearch[{"", arg}, opts]

iWebAudioSearch[{query_String}, opts:OptionsPattern[]] :=
	iWebAudioSearch[{query, Automatic}, opts]
iWebAudioSearch[{query:List[Repeated[_String]]}, opts:OptionsPattern[]] :=
	iWebAudioSearch[{StringRiffle[query, " "]}, opts]

iWebAudioSearch[{query_String, format : _?StringQ | _?ListQ}, opts:OptionsPattern[]] :=
	iWebAudioSearch[{query, format, Automatic}, opts]

iWebAudioSearch[{query:List[Repeated[_String]], format : _?StringQ | _?ListQ}, opts:OptionsPattern[]] :=
	iWebAudioSearch[{StringRiffle[query, " "], format}, opts]

iWebAudioSearch[{query_String, arg_}, opts:OptionsPattern[]] :=
iWebAudioSearch[{query, If[$CloudEvaluation || Developer`UseFrontEnd[CurrentValue["UserInteractionEnabled"]], $defaultFormat, $defaultFormatNoFE], arg}, opts]

iWebAudioSearch[{query:List[Repeated[_String]], arg_}, opts:OptionsPattern[]] :=
	iWebAudioSearch[{StringRiffle[query, " "], arg}, opts]


iWebAudioSearch[{query:List[Repeated[_String]], format : _?StringQ | _?ListQ, arg_}, opts:OptionsPattern[]] :=
	iWebAudioSearch[{StringRiffle[query, " "], format, arg}, opts]

iWebAudioSearch[{query_String, format : _?StringQ | _?ListQ, arg_}, opts:OptionsPattern[]] :=
Module[{queryParameter, next, method, max, property, total, enabled, internalFormat = format, propertySet = {}, criterion},
	max = OptionValue[ MaxItems];
	method = "Freesound";
	enabled = False;
	If[format === "RandomSample", max = 150];

	Catch[
		If [!Internal`PositiveMachineIntegerQ[max], Throw[getWebAudioSearchFailure["ioppm", {MaxItems, max}]]];
		Switch[format,
			_String,
			If[!GeneralUtilities`ValidPropertyQ[WebAudioSearch, format, Union[$formats, $supportedProperty[method]], "Type" -> "format"],
				Throw[getWebAudioSearchFailure["invformat", <|"format" -> format|>]]];
			If[MemberQ[$supportedProperty[method], format],
				internalFormat = "Set";
				propertySet = DeleteDuplicates[Flatten[(# /. $compositeProperty)& /@ DeleteDuplicates[Append[{format}, "TitleHyperlink"]]]]
			]
			,
			_List,
			If[!GeneralUtilities`ValidPropertyQ[WebAudioSearch, #, $supportedProperty[method], "Type" -> "property"],
				Throw[getWebAudioSearchFailure["invprop", <|"prop" -> #, "method" -> method|>]]] & /@ format;
			internalFormat = "Set";
			propertySet = DeleteDuplicates[Flatten[(# /. $compositeProperty)& /@ DeleteDuplicates[Append[format, "TitleHyperlink"]]]]
		];

		If[!validService[method],
			Throw[getWebAudioSearchFailure["invservice", <|"arg" -> method|>]]
		];
		If[StringLength[query] > 3000,
			Throw[getWebAudioSearchFailure["longquery", <||>]]
		];
		If[!$CloudEvaluation && !Developer`UseFrontEnd[CurrentValue["UserInteractionEnabled"]] && (format === "Gallery" || format === "GalleryMinimal"),
			Throw[getWebAudioSearchFailure["nofe", <||>]]
		];

		With[{propTemp = findStringSlot[arg]},
			If[!validCriterion[propTemp],
				Throw[getWebAudioSearchFailure["invarg", <|"arg" -> propTemp|>]]
			]
		];
		property = Switch[internalFormat,
			"Dataset", If[!$CloudEvaluation && Developer`UseFrontEnd[CurrentValue["UserInteractionEnabled"]], $propertyDataset[method], DeleteCases[$propertyDataset[method], "Player"]],
			"Samples", $propertySamples,
			"RandomSample", $propertyRandomSample,
			"TitleHyperlinks", $propertyTitleHyperlinks,
			"Gallery", $propertyGallery,
			"GalleryMinimal", $propertyGalleryMinimal,
			"Set", propertySet
		];

		criterion = validateCriterion[arg];

		queryParameter = Switch[#,
			"Freesound"
			,
			# -> <|
				{"query" -> query,
				"fields" -> "id,name,previews,url,images,duration,samplerate,download,channels,license,username,tags",
				"page_size" -> ToString[max]},
			constructSearchProperty[criterion, method]|>
			,
			"SoundCloud"
			,
			# -> <|
				{"q" -> query,
				"linked_partitioning" -> "1",
				"limit" -> ToString[max]},
			constructSearchProperty[criterion, method]|>
		] & /@ Audio`WebAudioSearchInformation[];

		{result, next, total} = Lookup[searchResource[method /. queryParameter, method, property, internalFormat], {"result", "next", "total"}];
		Switch[internalFormat,
			"Samples",
			#["Sample"]& /@ result,
			"RandomSample",
			With[{index = RandomInteger[Length[result] - 1] + 1},
				System`Audio[result[[index]]["SampleLink"],
					MetaInformation -> metaInfo[method,
						result[[index]]["Title"],
						result[[index]]["PageLink"],
						result[[index]]["Username"],
						result[[index]]["License"]]
				]
			],
			"Gallery",
			gallery[method, result, next, total, max, <|queryParameter|>, "Default", $propertyGallery, enabled],
			"GalleryMinimal",
			gallery[method, result, next, total, max, <|queryParameter|>, "Minimal", $propertyGalleryMinimal, enabled],
			"TitleHyperlinks",
			#["TitleHyperlink"]& /@ result,
			"Set",
			Association[Rule[#["TitleHyperlink"], Lookup[#, format]]& /@result],
			_,
			MapIndexed[(
				If[KeyExistsQ[#1, "SampleLink"],
					If[KeyExistsQ[#1, "SampleLink"], result[[First@#2, "SampleLink"]] = Iconize[result[[First@#2, "SampleLink"]]]];
				];
				result[[First@#2, "TitleHyperlink"]] = Hyperlink[result[[First@#2, "Title"]], result[[First@#2, "PageLink"]]]) &
				,
				result];
			Dataset[KeyDrop[result, {"Image", "Title", "PageLink"}]]
		]
	]
]

gallery[method_String, result_List, next_String, total_Integer, max_Integer, parameter_Association, mode_String, property_List, enabled_] :=
Module[{uuid = CreateUUID[]},
	AssociateTo[$result, {
		uuid -> <|
			"SearchParameter" -> parameter,
			"Mode" -> mode,
			"MaxPerPage" -> max,
			method ->
				<|"PageHistory" -> <|0 -> "null", 1 -> <|"Result" -> ((KeyDrop[#, "Attribution"]) &/@ result)|>|> |>
		|>
	}];

	AssociateTo[$resultCurrentPage, uuid -> <|
		"CurrentMethod" -> method,
		method -> <|
			"CurrentPage" -> 1,
			"Total" -> total,
			"PageHistory" -> <|
				0 -> "null",
				1 -> <|"Data" -><||>, "Selection" ->ConstantArray[False, Length[result]] |>
			|>
		|>
	|>
	];
	AppendTo[$result[uuid][method]["PageHistory"], Max[Keys[$result[uuid][method]["PageHistory"]]] + 1 -> next];

	Deploy[Pane[CustomGalleryDisplay[1, method, uuid, If[TrueQ[enabled], Automatic, None],
			parameter, max, property, mode]
		,
		ImageSize -> $defaultWindowSize]]
]

CustomGalleryDisplay[version: 1, method_String, uuid_String, appear_, parameter_Association, max_Integer, property_List, mode_String] :=
DynamicModule[{searchParameter, localProperty, localMode, localMax, box,
	sessionID = $SessionID,
	currentPage = 1,
	selection = $resultCurrentPage[uuid][method]["PageHistory"][1]["Selection"],
	currentMethod = $resultCurrentPage[uuid]["CurrentMethod"],
	downloading = False
	}
	,
	Dynamic[If[sessionID === $SessionID,
		Grid[{
			{
				Dynamic[Column[displayResult[uuid, Dynamic[currentPage], Dynamic[selection], Dynamic[currentMethod]],
					Dividers -> Center,
					FrameStyle -> RGBColor["#c7c7c7"],
					Spacings -> 0.8]],
				SpanFromLeft, SpanFromLeft
			}
			,
			controlBar[uuid, appear, Dynamic[currentPage], Dynamic[selection], Dynamic[currentMethod], Dynamic[downloading]]
			}
			,
			Spacings -> 0,
			FrameStyle -> RGBColor["#c7c7c7"],
			Frame -> {False, All},
			Alignment -> {{Left, Center, Right}, Baseline},
			ItemSize -> {{{ Scaled[.25], Fit, Scaled[.25]}}}
		],
		If[$VersionNumber < 12,
			Framed[
				Grid[{{Text[Style["Displaying this WebAudioSearch GUI requires a more recent version. Contact Wolfram Research to upgrade. ",
						11, RGBColor["#5f5f5f"], FontFamily -> "Helvetica", LineIndent -> 0]]
					,
					Hyperlink[Style["\[RightSkeleton]", FontSize -> 9], "http://www.wolfram.com/"]}}, Alignment -> {Left, Center}]
				,
				Background -> RGBColor["#f6f6f6"],
				FrameStyle -> RGBColor["#c7c7c7"],
				FrameMargins -> {{10, 10}, {Automatic, Automatic}}
			]
			,
			DynamicModule[{running = False},
				Dynamic[If[!TrueQ[running],
					Pane[Grid[{{
						Row[{Style["Inactive", FontFamily -> "Source Sans Pro", FontSize -> 12, FontColor -> RGBColor["#5f5f5f"]],
							Style[" WebAudioSearch " , FontFamily -> "Source Sans Pro", FontSize -> 12, FontColor -> RGBColor["#252525"]],
							Style["for ", FontFamily -> "Source Sans Pro", FontSize -> 12, FontColor -> RGBColor["#5f5f5f"]],
							Short[Style["\"" <> getQueryString[searchParameter, method]  <> "\"", FontFamily -> "Source Sans Pro", FontSize -> 12, FontColor -> RGBColor["#252525"], LineIndent -> 0], 1]
						}],
						Item[Button[
							PaneSelector[{
								False -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Refresh-Arrow.png"]],
								True -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Refresh-Arrow-Hover.png"]]
								},
								Dynamic[CurrentValue["MouseOver"]]
							]
							,
							running = True;
							With[{temp2 = Catch[searchResource[searchParameter[method], method, localProperty, "Gallery"]]},
								If[!FailureQ[temp2],
									NotebookWrite[box, ToBoxes[gallery[method, temp2["result"], temp2["next"], temp2["total"], localMax, searchParameter, localMode, localProperty, False]]]
									,
									running = False
								]
							]
							,
							Method -> "Queued",
							Appearance -> None,
							ImageSize -> Automatic
						], ItemSize -> Fit]
						}},
						Spacings -> 1,
						FrameStyle -> RGBColor["#c7c7c7"],
						Frame -> {False, All},
						Alignment -> {{Left, Right}, Center}
					], ImageSize -> $inactiveWindowSize]
					,
					Animator[Appearance -> "Necklace"]
				]]
			]
		]]
		,
		Initialization :> (System`WebAudioSearch;
			searchParameter = parameter;
			localMode = mode;
			localMax = max;
			localProperty = property;
			box = ParentBox@ParentBox@ParentBox[EvaluationBox[]];
			CurrentValue[EvaluationCell[], CellEditDuplicate] = False
		),
		Deinitialization :> (
		If[AssociationQ[$result] && !MissingQ[Lookup[$result, uuid]],
			If[!StringQ[#], RemoveAudioStream[#]] & /@
				Flatten@Values@Values@Select[
					$resultCurrentPage[uuid][method]["PageHistory"], AssociationQ][[All, "Data"]][[All, All]]
		]
	),
		TrackedSymbols :> {currentPage, currentMethod}
	]
]

CustomGalleryDisplay[version_, ___] := Framed[
	Row[{Text[Style[Dynamic[FEPrivate`FrontEndResource["FEStrings", "webAudioSearchVersionText"]], 11, RGBColor["#5f5f5f"], FontFamily -> "Helvetica"]]
		,
		Hyperlink[Style["\[RightSkeleton]", FontSize -> 9], "http://www.wolfram.com/"]}]
	,
	Background -> RGBColor["#f6f6f6"],
	FrameStyle -> RGBColor["#c7c7c7"],
	FrameMargins -> {{10, 10}, {Automatic, Automatic}}
]

switchMethod[uuid_String, method:Alternatives@@$services, Dynamic[currentMethod_], Dynamic[currentPage_], Dynamic[selection_]] :=
Module[{parameter, result, next, total, temp},
	$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Selection"] = selection;
	$resultCurrentPage[uuid]["CurrentMethod"] = method;
	If[Lookup[$result[uuid], method, None] === None,
		parameter = $result[uuid]["SearchParameter"][method];
		localProperty = Switch[$result[uuid]["Mode"], "Default", $propertyGallery, "Minimal", $propertyGalleryMinimal, _, $propertyGallery];
		{result, next, total} = Lookup[searchResource[parameter, method, localProperty, "Gallery"], {"result", "next", "total"}];

		AssociateTo[$result[uuid],
			method -> <|"PageHistory" -> <|0 -> "null", 1 -> <|"Result" -> ((KeyDrop[#, "Attribution"]) &/@ result)|>|> |>
		];

		AssociateTo[$resultCurrentPage[uuid], method -> <|
			"CurrentPage" -> 1,
			"Total" -> total,
			"PageHistory" -> <|
				0 -> "null",
				1 -> <|"Data" -><||>, "Selection" ->ConstantArray[False, Length[result]] |>
			|>
		|>];
		AppendTo[$result[uuid][method]["PageHistory"], Max[Keys[$result[uuid][method]["PageHistory"]]] + 1 -> next];
	];
	temp = $resultCurrentPage[uuid][method]["CurrentPage"];
	{currentMethod, currentPage, selection} = {method, temp, $resultCurrentPage[uuid][method]["PageHistory"][temp]["Selection"]}
]

getQueryString[parameter_Association, method_String] := Switch[method,
	"Freesound",
	parameter[method]["query"]
	,
	"SoundCloud",
	parameter[method]["q"]
	,
	_,
	""
]

getOriginalAudio[stream_] /; Audio`AudioStreamInternals`validAudioStreamQ[stream] && !Audio`AudioStreamInternals`validInputAudioStreamQ[stream] :=
	Audio`AudioStreamInternalsDump`$audioStreams["External", stream["ID"]]["Audio"]

copyButton[uuid_String, desktop:True, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_]] :=
		Button[
			"Copy"
			,
			CopyToClipboard[
				MapIndexed[
					(If[TrueQ[#1],
						If[StringQ[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]]],
							$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]] =
								AudioStream[System`Audio[
									$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]],
										MetaInformation -> metaInfo[
												currentMethod,
												$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["Title"],
												$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["PageLink"],
												$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["Username"],
												$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["License"]
										]
									]
								]
						];
						getOriginalAudio[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]]]
						,
						Unevaluated[Sequence[]]
					])&
					,
					selection
				]
			]
			,
			ImageSize -> All,
			BaseStyle->"DialogStyle",
			Enabled -> Dynamic[Or@@selection],
			Method -> "Queued"
		]

copyButton[uuid_String, desktop:False, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_]] :=
		Button[
			"Copy"
			,
			CellPrint[ExpressionCell[
				MapIndexed[
					(If[TrueQ[#1],
						If[StringQ[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]]],
							$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]] =
								System`Audio[
									$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]],
									MetaInformation -> metaInfo[
										currentMethod,
										$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["Title"],
										$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["PageLink"],
										$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["Username"],
										$result[uuid][currentMethod]["PageHistory"][currentPage]["Result"][[First[#2]]]["License"]
									]
								]
						];
						$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][First[#2]]
						,
						Unevaluated[Sequence[]]
					])&
					,
					selection
				],
				"Output"]
			]
			,
			ImageSize -> All,
			BaseStyle->"DialogStyle",
			Enabled -> Dynamic[Or@@selection],
			Method -> "Queued"
		]

controlBar[uuid_String, appear_, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_], Dynamic[downloading_]] :=
	{
		Item[Row[{Spacer[5],Dynamic@copyButton[uuid, !$CloudEvaluation, Dynamic[currentPage], Dynamic[selection], Dynamic[currentMethod]]}],
			Background -> $bottomBackground],
		Item[Dynamic@nextPreviousPageButton[uuid, Dynamic[currentPage], Dynamic[selection], Dynamic[currentMethod], Dynamic[downloading]],
			Background -> $bottomBackground],
		Item[Row[{
				PaneSelector[{
					True -> ProgressIndicator[Appearance -> "Percolate"],
					False -> $logoWatermark[currentMethod]}
					,
					Dynamic[Refresh[downloading]]
				],
				Spacer[5]}]
			,
			Background -> $bottomBackground
		]
	}

displayResult[uuid_String, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_]] := With[{index = currentPage, method = currentMethod},
	MapIndexed[(
		AssociateTo[$resultCurrentPage[uuid][method]["PageHistory"][index]["Data"], First[#2] -> #1["SampleLink"][[1]]];
		Item[dynamicSelect[uuid,
			#1["Image"],
			First[#2],
			#1["Title"],
			#1["Username"],
			#1["Duration"],
			Dynamic[currentPage],
			Dynamic[selection],
			Dynamic[currentMethod],
			If[ImageQ[#1["Image"]], $fakeItem, $fakeItemNoImage]]
			,
			Alignment -> Center]) &
		,
		$result[uuid][method]["PageHistory"][index]["Result"]
	]
]

displayResult[___] := {}

dynamicSelect[uuid_String, expr_, i_, title_, user_, duration_, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_], holder_] := Dynamic[
	itemButton[uuid, expr, i, title, user, duration, Dynamic[currentPage], Dynamic[selection], Dynamic[currentMethod]]
	,
	SynchronousUpdating -> False,
	SingleEvaluation -> True,
	TrackedSymbols :> {},
	CachedValue :> (ItemBox[DynamicBox[holder]])
]

$providerToSite[method_String] := Switch[
	method,
	"Freesound", "https://freesound.org",
	"SoundCloud", "https://soundcloud.com",
	_,
	method
]

metaInfo[provider_String, title_String, url_, user_String, license_] :=
<|
	"Provider" -> URL[$providerToSite[provider]],
	"Title" -> title,
	"URL" -> URL[url],
	"Username" -> user,
	"License" -> URL[license]
|>

formattedTime[x_?IntegerQ] :=
	With[{s = ToString[IntegerPart[Mod[x, 3600]/60]]}, If[StringLength[s] == 1, "0" <> s, s]] <> ":" <>
	With[{s = ToString[Mod[Mod[x, 3600], 60]]}, If[StringLength[s] == 1, "0" <> s, s]]

formattedDuration[x_?Developer`RealQ] := If[x>1,
	Quantity[N[Round[x, 10^-1]], "Seconds"]
	,
	Quantity[N[Round[x, 10^-3]], "Seconds"]
]

formattedSampleRate[x_?IntegerQ] := Quantity[N[x/1000.], "Kilohertz"]

highlightItem[duration_, playing:True|False] := Column[{
		Switch[playing,
			False, Mouseover[$pauseIcon[False], $pauseIcon[True]],
			True, Mouseover[$playIcon[False], $playIcon[True]]
		],
		Style[formattedTime[IntegerPart[duration]], $durationTextColor, Smaller]
	}
]

$fakeItem := TagBox[GridBox[{{
	TemplateBox[List[0], "Spacer1"],
	CheckboxBox[False],
	$fakeImage,
	ButtonBox["", ButtonFunction :> {}, Evaluator -> None,
		ImageSize -> {24, 24}, Enabled -> False, Appearance -> Automatic, Method -> "Preemptive"],
	ItemBox[TagBox[GridBox[{{
		StyleBox["\".........\"",
			HyphenationOptions -> {"HyphenationCharacter" -> "\[Ellipsis]"},
			LineIndent -> 0,
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "SemiBold",
			FontColor -> RGBColor["#5f5f5f"],
			StripOnInput -> False
		]},
		{StyleBox["\"......\"",
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "Regular",
			FontColor -> $userFontColor,
			StripOnInput -> False]}}
		,
		GridBoxAlignment -> {"Columns" -> {{Left}}},
		DefaultBaseStyle -> "Column",
		GridBoxItemSize -> {
			"Columns" -> {{Automatic}},
			"Rows" -> {{Automatic}}}], "Column"],
		ItemSize -> Fit,
		StripOnInput -> False]
	}},
	GridBoxAlignment -> {
		"Columns" -> {{Left}},
		"Rows" -> {{Center}}},
	AutoDelete -> False,
	GridBoxItemSize -> {
		"Columns" -> {{Automatic}},
		"Rows" -> {{Automatic}}}]
	,
	"Grid"
]

$fakeItemNoImage := TagBox[GridBox[{{
	TemplateBox[List[0], "Spacer1"],
	CheckboxBox[False],
	ButtonBox["", ButtonFunction :> {}, Evaluator -> None,
		ImageSize -> {24, 24}, Enabled -> False, Appearance -> Automatic, Method -> "Preemptive"],
	ItemBox[TagBox[GridBox[{{
		StyleBox["\".........\"",
			HyphenationOptions -> {"HyphenationCharacter" -> "\[Ellipsis]"},
			LineIndent -> 0,
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "SemiBold",
			FontColor -> RGBColor["#5f5f5f"],
			StripOnInput -> False
		]},
		{StyleBox["\"......\"",
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "Regular",
			FontColor -> $userFontColor,
			StripOnInput -> False]}}
		,
		GridBoxAlignment -> {"Columns" -> {{Left}}},
		DefaultBaseStyle -> "Column",
		GridBoxItemSize -> {
			"Columns" -> {{Automatic}},
			"Rows" -> {{Automatic}}}], "Column"],
		ItemSize -> Fit,
		StripOnInput -> False]
	}},
	GridBoxAlignment -> {
		"Columns" -> {{Left}},
		"Rows" -> {{Center}}},
	AutoDelete -> False,
	GridBoxItemSize -> {
		"Columns" -> {{Automatic}},
		"Rows" -> {{Automatic}}}]
	,
	"Grid"
]

itemButton[uuid_String, expr_, index_, title_, user_, duration_, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_]] := DynamicModule[{image},
Dynamic[
	Grid[{{
		Spacer[0],
		Checkbox[Dynamic[selection[[index]]]],
		If[ImageQ[image], image, Nothing],
		If[!$CloudEvaluation, Button[
			PaneSelector[{
				True -> highlightItemSample[True],
				False -> highlightItemSample[False]
				},
				Dynamic[
					(StringQ[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]] ||
						$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]["Status"] =!= "Playing")
				]
			]
			,
			If[StringQ[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]],
				$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index] =
					AudioStream[Audio[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]]];
				AudioPlay[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]]
				,
				If[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]["Status"] === "Playing",
					AudioPause[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]]
					,
					AudioPlay[$resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Data"][index]]
				];
			]
			,
			ImageSize -> {Automatic, 30}, Appearance -> None
		], Nothing],
		Item[Column[{
			Tooltip[Pane[Style[title,
				FontFamily -> "Source Sans Pro",
				FontSize -> 11,
				FontWeight -> "SemiBold",
				FontColor -> RGBColor["#5f5f5f"]], ImageSize -> {Automatic, 15}], title],
			Style["by " <> user,
				FontFamily -> "Source Sans Pro",
				FontSize -> 11,
				FontWeight -> "Regular",
				FontColor -> $userFontColor
			]}, Spacings -> 0]
			,
			ItemSize -> Fit
		],
		Item[Style[formattedTime[IntegerPart[duration]],
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "Regular",
			FontColor -> $userFontColor
			]
			,
			Alignment -> Right
		],
		Spacer[0]
		}},
		Alignment -> {Left, Center}
	]
	,
	Initialization :> (image = If[ImageQ[expr], ImageResize[expr, {$thumbnailSize, $thumbnailSize}], Null])
]]

constructSearchProperty[arg___, method:"Freesound"] :=
Module[{temp, result},
	temp = Flatten[{arg}];

	result = With[{argNum = Length[Keys[#]]},
				Switch[ argNum,
					2,
					{StringJoin[(#["Name"] /. $criterionConversion[method]),
						":",
						ToString[#["Low"]]],
					If[#["Name"] === "beatperminute", "descriptors_filter", "filter"]},
					3,
					{StringJoin[(#["Name"] /. $criterionConversion[method]),
						":[",
						ToString[#["Low"]],
						" TO ",
						ToString[#["High"]],
						"]"],
					If[#["Name"] === "beatperminute", "descriptors_filter", "filter"]}
				]]& /@ temp;
	If[temp === "", {}, GroupBy[result, Last, StringRiffle[#[[All, 1]], " "]&]]
]

constructSearchProperty[arg___, method:"SoundCloud"] :=
Module[{temp},
	temp = Flatten[{parseSearchParameter[arg]}];
	With[{argNum = Length[Keys[#]]},
		Switch[ argNum,
			2,
			{
				(#["Name"] /. $criterionConversion[method]) -> #["Low"]
			},
			3,
			{
				If[Internal`RealValuedNumericQ[#["Low"]], (#["Name"] /. $criterionConversion[method]) ~~ "[from]" -> ToString[#["Low"]], Unevaluated[Sequence[]]],
				If[Internal`RealValuedNumericQ[#["High"]], (#["Name"] /. $criterionConversion[method]) ~~ "[to]" -> ToString[#["High"]], Unevaluated[Sequence[]]]
			}
		]
	]& /@ temp
]

searchResource[query_Association, method: Alternatives@@$services, prop_List, format_String] :=
Module[{temp, command, ret},
	command = Switch[method,
		"Freesound", If[KeyExistsQ[query, "descriptors_filter"], "RawCombinedSearch", "RawTextSearch"],
		"SoundCloud", "RawTrackSearch"
	];
	temp = Quiet@Check[IntegratedServices`RemoteServiceExecute[Symbol["WebAudioSearch"], method, command, query],
		First[$MessageList]
	];
	Which[
		MatchQ[temp, $Canceled],
			Throw[getWebAudioSearchFailure["timeout", <||>]]
		,
		MatchQ[temp, HoldForm[MessageName[__]]],
			Message@@temp;
			Throw[Failure["WebAudioSearchFailure", <||>]]
		,
		FailureQ[temp],
			Throw[getWebAudioSearchFailure["interr", <||>]]
		,
		!FailureQ[temp],
			extractInfo[temp, method, prop, format]
		,
		True,
			Throw[temp]
	]
]

extractInfo[rule_Association, method: Alternatives@@$services, prop_List, format_String] :=
Module[{result, nextPage, total},
	{result, nextPage, total} = Switch[method,
		"Freesound",
			With[{next = Lookup[rule, "next", "next"], count = Lookup[rule, "count", "count"]},
				{rule["results"],
				(If[next === "next", rule["more"], next]) /. {Null -> "null"},
				If[count === "count", 0, rule["count"]]}
			]
		,
		"SoundCloud",
			{"collection" /. rule, "next_href" /. rule, 0}
		,
		_,
		{Normal[rule], "null", 0}
	];
	<|
		"next" -> nextPage,
		"result" -> (iExtractInfo[#, method, prop, format] & /@ result),
		"total" -> total
	|>
]

iExtractInfo[rule_Association, method: Alternatives@@$services, prop_List, format_String] :=
Module[{link, property, userName, userLink, image},
	property = <|(# -> (rule[(ToLowerCase[#] /. $propertyConversion[method])] )) & /@ prop |>;
	Switch[method,
		"Freesound"
		,
		link = rule["url"];
		userName = rule[("username" /. $propertyConversion[method])];
		userLink = StringReplace[link, n:userName ~~ ___ ~~ EndOfString :> n];
		If[KeyExistsQ[property, "License"] && format === "Dataset", property["License"] = Hyperlink[Lookup[$licenseRule, extractCClicense[property["License"]], $logoLicense["CC"]], property["License"]]];
		If[KeyExistsQ[property, "License"] && format === "Set", property["License"] = URL[property["License"]]];
		If[KeyExistsQ[property, "SampleLink"], property["SampleLink"] = URL["preview-lq-ogg" /. property["SampleLink"]]];
		If[KeyExistsQ[property, "SampleRate"] && (format === "Dataset" || format === "Set"), property["SampleRate"] = formattedSampleRate[IntegerPart[property["SampleRate"]]]];
		If[KeyExistsQ[property, "Duration"] && (format === "Dataset" || format === "Set"), property["Duration"] = formattedDuration[property["Duration"]]];
		If[KeyExistsQ[property, "Image"],
			image = URLRead["spectral_m" /. property["Image"]];
			property["Image"] = ImportString[FromCharacterCode@Normal[image[[1]]], "content-type" /. image["Headers"]]
			,
			property["Image"] = None
		];
		If[KeyExistsQ[property, "Player"], property["Player"] = sampleButton[property["SampleLink"]]];
		If[KeyExistsQ[property, "TitleHyperlink"], property["TitleHyperlink"] = Hyperlink[property["Title"], property["PageLink"]]];
		If[KeyExistsQ[property, "Sample"], property["Sample"] =
			System`Audio[property["SampleLink"],
				MetaInformation -> metaInfo[method, property["Title"], property["PageLink"], property["Username"], property["License"]]]]
		,
		"SoundCloud"
		,
		link = "permalink_url" /. rule;
		userName = "username" /. (("username" /. $propertyConversion[method]) /. rule);
		userLink = "permalink_url" /. (("username" /. $propertyConversion[method]) /. rule);
		If[KeyExistsQ[property, "Duration"], property["Duration"] = formattedDuration[property["Duration"] / 1000.]];
		If[KeyExistsQ[property, "Username"], property["Username"] = "username" /. property["Username"]];
		If[KeyExistsQ[property, "Image"],
			image = URLRead[ property["Image"] /. {Null -> "waveform_url"} /. rule];
			property["Image"] = ImportString[FromCharacterCode@Normal[image[[1]]], "content-type" /. image["Headers"]]
			,
			property["Image"] = None
		];

	];

	<|	property,
		"Attribution" -> Hyperlink[
			Column[{
				$logo[method],
				Style["  by " <> userName,
					FontFamily -> "Source Sans Pro",
					FontSize -> 11,
					FontWeight -> "Regular",
					FontColor -> $userFontColor
				]}
			],
			userLink
		]
	|>
]

highlightItemSample[playing:True|False] := Switch[playing,
	False
	,
	PaneSelector[{
		False -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio"}, "PlayButton-Single-Released.png"]],
		True -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio"}, "PlayButton-Single-Pause-Hover.png"]]
		},
		Dynamic[CurrentValue["MouseOver"]]
	]
	,
	True
	,
	PaneSelector[{
		False -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio"}, "PlayButton-Single-Default.png"]],
		True -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio"}, "PlayButton-Single-Default-Hover.png"]]
		},
		Dynamic[CurrentValue["MouseOver"]]
	]
]

highlightItemSample2[playing:True|False] := Switch[playing,
	False, Mouseover[$pauseIcon[False], $pauseIcon[True]],
	True, Mouseover[$playIcon[False], $playIcon[True]]
]

sampleButtonHelper[name_URL, Dynamic[play_]]:=With[{s = play}, If[
	Audio`AudioStreamInternals`validAudioStreamQ[play]
	,
	If[play["Status"] === "Playing", AudioPause[play], AudioPlay[play]]
	,
	play = AudioStream[Audio[name]]; AudioPlay[play]
]]

sampleButton[sampleLink_URL] :=
DynamicModule[{playing},
	Dynamic[Button[
		PaneSelector[{
			"Playing" -> highlightItemSample[False]}
			,
			Dynamic[playing["Status"]]
			,
			highlightItemSample[True]
		],
		sampleButtonHelper[sampleLink, Dynamic[playing]]
		,
		Appearance -> None,
		Method -> "Queued"
	]
]]

pageToString[uuid_String, method_String, page_] :=
Module[{total = $resultCurrentPage[uuid][method]["Total"],
	begin = ((page-1) * $result[uuid]["MaxPerPage"]) + 1,
	end, stringItems,
	currentNum = Length[$result[uuid][method]["PageHistory"][page]["Result"]]
	}
	,
	end = If[(page * $result[uuid]["MaxPerPage"]) > total && total =!= 0, total, (page - 1) * $result[uuid]["MaxPerPage"] + currentNum];
	begin = If[begin > end, end, begin];
	stringItems = {
		Style["  showing " <> ToString[begin] <> "-" <> ToString[end],
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "Regular",
			FontColor -> RGBColor["#828282"]
		],
		If[total > 0,
			{
				Style[" of ",
					FontFamily -> "Source Sans Pro",
					FontSize -> 11,
					FontWeight -> "Regular",
					FontColor -> RGBColor["#828282"]]
				,
				Style[total,
					FontFamily -> "Source Sans Pro",
					FontSize -> 11,
					FontWeight -> "Bold",
					FontColor -> RGBColor["#5f5f5f"]]
			}
			,
			Unevaluated[Sequence[]]
		],
		Style["  ",
			FontFamily -> "Source Sans Pro",
			FontSize -> 11,
			FontWeight -> "Regular"
		]
	};
	Apply[StringJoin, ToString[#, StandardForm] & /@ Flatten[stringItems]]
]

nextPreviousPageButton[uuid_String, Dynamic[currentPage_], Dynamic[selection_], Dynamic[currentMethod_], Dynamic[downloading_]] :=
DynamicModule[{previousResult, result, next, prop, localCurrentPage = currentPage,
	nextResult = $result[uuid][currentMethod]["PageHistory"][currentPage + 1],
	total = $resultCurrentPage[uuid][currentMethod]["Total"]
	}
	,
	Row[{
		Button[
			PaneSelector[{
				True -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Previous-Active.png"]],
				False -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Previous-Inactive.png"]]
				}
				,
				Dynamic[localCurrentPage =!= 1]
			]
			,
			previousResult = $result[uuid][currentMethod]["PageHistory"][localCurrentPage - 2];
			nextResult = $result[uuid][currentMethod]["PageHistory"][localCurrentPage];
			$resultCurrentPage[uuid][currentMethod]["PageHistory"][localCurrentPage]["Selection"] = selection;
			$resultCurrentPage[uuid][currentMethod]["CurrentPage"] = localCurrentPage - 1;
			localCurrentPage--;
			currentPage--;
			selection = $resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Selection"];
			,
			Enabled -> Dynamic[localCurrentPage =!= 1],
			ImageSize -> All,
			Appearance -> None,
			Method -> "Queued"
		],
		Dynamic[Item[Style[pageToString[uuid, currentMethod, localCurrentPage]]], TrackedSymbols :> {localCurrentPage}],
		Button[
			PaneSelector[{
				True -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Next-Active.png"]],
				False -> Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"Typeset", "Audio", "WebAudioSearch"}, "Next-Inactive.png"]]
				}
				,
				Dynamic[nextResult =!= "null"]
			]
			,
			prop = Switch[$result[uuid]["Mode"], "Default", $propertyGallery, "Minimal", $propertyGalleryMinimal, _, $propertyGallery];
			{result, next} = With[{tempNext = $result[uuid][currentMethod]["PageHistory"][localCurrentPage + 1]},
				If[AssociationQ[tempNext],
					{$result[uuid][currentMethod]["PageHistory"][localCurrentPage + 1], $result[uuid][currentMethod]["PageHistory"][localCurrentPage + 2]}
					,
					downloading = True;
					Lookup[searchResource[<|URLParse[tempNext, "Query"]|>, currentMethod, prop, "Gallery"], {"result", "next"}]
				]];
			downloading = False;
			If [StringQ[$result[uuid][currentMethod]["PageHistory"][localCurrentPage + 1]],
				$result[uuid][currentMethod]["PageHistory"][localCurrentPage + 1] = <|"Result" -> result|>;
				AppendTo[$resultCurrentPage[uuid][currentMethod]["PageHistory"],
					(localCurrentPage + 1) -> <|"Data" -> <||>, "Selection" -> ConstantArray[False, Length[result]] |>];
			];
			$result[uuid][currentMethod]["PageHistory"][localCurrentPage + 2] = next;
			nextResult = next;
			$resultCurrentPage[uuid][currentMethod]["PageHistory"][localCurrentPage]["Selection"] = selection;
			$resultCurrentPage[uuid][currentMethod]["CurrentPage"] = localCurrentPage + 1;
			localCurrentPage++;
			currentPage++;
			selection = $resultCurrentPage[uuid][currentMethod]["PageHistory"][currentPage]["Selection"];
			,
			Enabled -> Dynamic[nextResult =!= "null"],
			Appearance -> None,
			ImageSize -> All,
			Method -> "Queued"
		]
	}]
	,
	Initialization :> (
		nextResult = $result[uuid][currentMethod]["PageHistory"][currentPage + 1];
		total = $resultCurrentPage[uuid][currentMethod]["Total"];
	)
]

ClearAll[validateCriterion];
SetAttributes[validateCriterion, HoldAll];
validateCriterion[Automatic] := {}
validateCriterion[arg_Function] := Module[{exp},
	exp = getFunc[arg];
	exp = exp /. SameQ -> Equal;
	parseSearchParameter[Evaluate[exp]]
]

getFunc[Function[arg_]] := Hold[arg]

validateCriterion[s___] := Throw[getWebAudioSearchFailure["invarg", <|"arg" -> s|>]]

ClearAll[parseSearchParameter];
SetAttributes[parseSearchParameter, HoldAll];

parseSearchParameter[] := {}
parseSearchParameter[Hold[arg___]] := parseSearchParameter[arg]
parseSearchParameter[And[s___]] := ReleaseHold[ parseSearchParameter /@ Hold[s]]

parseSearchParameter[Greater[upper_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[upper], upper > 0] :=
parseSearchParameter[Less[Slot[s], upper]]

parseSearchParameter[Greater[upper_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[upper]] :=
parseSearchParameter[Less[Slot[s], upper]]

parseSearchParameter[Greater[Slot[s_], lower_]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[lower], lower > 0] :=
parseSearchParameter[Less[lower, Slot[s]]]

parseSearchParameter[Greater[Slot[s_], lower_]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[lower]] :=
parseSearchParameter[Less[lower, Slot[s]]]

parseSearchParameter[Greater[upper_, Slot[s_], lower_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $realCriterion],
		Internal`RealValuedNumericQ[upper],
		upper > 0,
		Internal`RealValuedNumericQ[lower],
		lower > 0
	] :=
	parseSearchParameter[Less[lower, Slot[s], upper]]

parseSearchParameter[Greater[upper_, Slot[s_], lower_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $integerCriterion],
		Internal`PositiveMachineIntegerQ[upper],
		Internal`PositiveMachineIntegerQ[lower]
	] :=
	parseSearchParameter[Less[lower, Slot[s], upper]]

parseSearchParameter[GreaterEqual[upper_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[upper], upper > 0] :=
parseSearchParameter[Less[Slot[s], upper]]

parseSearchParameter[GreaterEqual[upper_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[upper]] :=
parseSearchParameter[Less[Slot[s], upper]]

parseSearchParameter[GreaterEqual[Slot[s_], lower_]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[lower], lower > 0] :=
parseSearchParameter[Less[lower, Slot[s]]]

parseSearchParameter[GreaterEqual[Slot[s_], lower_]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[lower]] :=
parseSearchParameter[Less[lower, Slot[s]]]

parseSearchParameter[GreaterEqual[upper_, Slot[s_], lower_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $realCriterion],
		Internal`RealValuedNumericQ[upper],
		upper > 0,
		Internal`RealValuedNumericQ[lower],
		lower > 0
	] :=
	parseSearchParameter[Less[lower, Slot[s], upper]]

parseSearchParameter[GreaterEqual[upper_, Slot[s_], lower_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $integerCriterion],
		Internal`PositiveMachineIntegerQ[upper],
		Internal`PositiveMachineIntegerQ[lower]
	] :=
	parseSearchParameter[Less[lower, Slot[s], upper]]

parseSearchParameter[Less[lower_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[lower], lower > 0] :=
<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> "*"|>

parseSearchParameter[Less[lower_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[lower]] :=
<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> "*"|>

parseSearchParameter[Less[Slot[s_], upper_]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[upper], upper > 0] :=
<|"Name" -> ToLowerCase[s], "Low" -> "*", "High" -> upper|>

parseSearchParameter[Less[Slot[s_], upper_]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[upper]] :=
<|"Name" -> ToLowerCase[s], "Low" -> "*", "High" -> upper|>

parseSearchParameter[Less[lower_, Slot[s_], upper_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $realCriterion],
		Internal`RealValuedNumericQ[upper],
		upper > 0,
		Internal`RealValuedNumericQ[lower],
		lower > 0] :=
	<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> upper|>

parseSearchParameter[Less[lower_, Slot[s_], upper_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $integerCriterion],
		Internal`PositiveMachineIntegerQ[upper],
		Internal`PositiveMachineIntegerQ[lower]
	] :=
	<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> upper|>

parseSearchParameter[LessEqual[lower_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[lower], lower > 0] :=
<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> "*"|>

parseSearchParameter[LessEqual[lower_, Slot[s_]]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[lower]] :=
<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> "*"|>

parseSearchParameter[LessEqual[Slot[s_], upper_]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[upper], upper > 0] :=
<|"Name" -> ToLowerCase[s], "Low" -> "*", "High" -> upper|>

parseSearchParameter[LessEqual[Slot[s_], upper_]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[upper], upper > 0] :=
<|"Name" -> ToLowerCase[s], "Low" -> "*", "High" -> upper|>

parseSearchParameter[LessEqual[lower_, Slot[s_], upper_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $realCriterion],
		Internal`RealValuedNumericQ[upper],
		upper > 0,
		Internal`RealValuedNumericQ[lower],
		lower > 0] :=
	<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> upper|>

parseSearchParameter[LessEqual[lower_, Slot[s_], upper_]] /; Quiet@And[
		MatchQ[s, Alternatives @@ $integerCriterion],
		Internal`PositiveMachineIntegerQ[upper],
		Internal`PositiveMachineIntegerQ[lower]
	] :=
	<|"Name" -> ToLowerCase[s], "Low" -> lower, "High" -> upper|>

parseSearchParameter[Equal[Slot[s_], query_]] /; Quiet@And[MatchQ[s, Alternatives @@ $realCriterion], Internal`RealValuedNumericQ[query], query > 0] :=
<|"Name" -> ToLowerCase[s], "Low" -> query|>

parseSearchParameter[Equal[Slot[s_], query_]] /; Quiet@And[MatchQ[s, Alternatives @@ $integerCriterion], Internal`PositiveMachineIntegerQ[query]] :=
<|"Name" -> ToLowerCase[s], "Low" -> query|>

parseSearchParameter[Equal[Slot[s:"Tag"], query_String]] := <|"Name" -> ToLowerCase[s], "Low" -> query|>

parseSearchParameter[s___] := Throw[getWebAudioSearchFailure["invarg", <|"arg" -> HoldForm[s]|>]]

findStringSlot[Automatic] :={}
findStringSlot[fun_] := DeleteDuplicates[Cases[fun, Slot[s_String] :> s, {0, Infinity}, Heads -> True]]

validProperty[p_List, service:Alternatives@@$services] := validProperty[p, $propertyDataset[service]]
validProperty[p_String, service:Alternatives@@$services] := validProperty[p, $propertyDataset[service]]
validProperty[p_String, properties_List] := MemberQ[properties,p]
validProperty[p_List, properties_List] := VectorQ[p, validProperty[#, properties]&]
validProperty[___] := False
validCriterion[p_List] := validProperty[p, $criterion]
validService[s_String] := MemberQ[$services, s]

getWebAudioSearchFailure[msgName_String, params: _?AssociationQ | _?ListQ] :=
Failure["WebAudioSearchFailure",
	Association[
		RuleDelayed["MessageTemplate", MessageName[WebAudioSearch, msgName]],
		Rule["MessageParameters", params]
	]
]

getAuthenticatedServiceObject[service_String, starttime_, timeout_] :=
Module[{srvcObj},
	srvcObj = ServiceConnect[service];
	getAuthenticatedServiceObject[srvcObj,starttime,timeout];
	Switch[srvcObj,
		_ServiceConnect,
		Throw[getWebAudioSearchFailure["invalidserviceobj", <||>]],
		_ServiceObject,
		srvcObj,
		_,
		Throw[getWebAudioSearchFailure["invalidserviceobj", <||>]]
	]
]

getAuthenticatedServiceObject[service_ServiceObject, starttime_, timeout_] :=
(
	While[!ServiceConnections`Private`authenticatedServiceQ[service], timeoutHandler[service,starttime,timeout]; Pause[1]];
	service
)

timeoutHandler[service_, starttime_, timeout_:$connTimeout] :=
(
	If[UnixTime[Now] - starttime >= timeout,
		Throw[getWebAudioSearchFailure["timeout", <||>]]
	]
)

$licenseRule = <|
	"publicdomain/zero" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["Zero"]}, Spacer[$licenseIconSpacing]]],
	"licenses/by" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["By"]}, Spacer[$licenseIconSpacing]]],
	"licenses/nc" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["Nc"]}, Spacer[$licenseIconSpacing]]],
	"licenses/by-sa" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["By"], $logoLicense["Sa"]}, Spacer[$licenseIconSpacing]]],
	"licenses/by-nd" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["By"], $logoLicense["Nd"]}, Spacer[$licenseIconSpacing]]],
	"licenses/by-nc" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["By"], $logoLicense["Nc"]}, Spacer[$licenseIconSpacing]]],
	"licenses/by-nc-sa" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["By"], $logoLicense["Nc"], $logoLicense["Sa"]}, Spacer[$licenseIconSpacing]]],
	"licenses/by-nc-nd" -> Row[Riffle[{$logoLicense["CC"], $logoLicense["By"], $logoLicense["Nc"], $logoLicense["Nd"]}, Spacer[$licenseIconSpacing]]]
|>

extractCClicense[s_String] := StringReplace[s,
	StartOfString ~~ ___ ~~ "creativecommons.org/" ~~ u___ ~~ "/" ~~ ___ ~~ "/" :> u]

(* ::Section:: *)
(*Epilog*)

$fakeImage = GraphicsBox[{GrayLevel[0.25], RectangleBox[{0, 0}, RoundingRadius -> 0.1]}, ImageSize -> {40, 40} ]

End[]

EndPackage[]


SetAttributes[
{
	System`WebAudioSearch,
	Audio`WebAudioSearchInformation
},
    {Protected, ReadProtected}];
