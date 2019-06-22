(* Formatting *)

Format[AudioFileStreamTools`MetaTags[obj_AudioFileStreamTools`FileStreamObject, parts___, opts:OptionsPattern[(* $metaTagsOpts *)]], StandardForm] := 
(
	getMetaTagAssociation[obj, parts, opts]
)

(* Tag Retrieval *)

AudioFileStreamTools`MetaTags /: AudioFileStreamTools`MetaTags[obj_, parts1___, opts:OptionsPattern[(* $metaTagsOpts *)]][parts2__] /; (SameQ[Head[obj], AudioFileStreamTools`FileStreamObject]) := 
(
	getMetaTagAssociation[obj, parts1, parts2, opts]
)

AudioFileStreamTools`MetaTags /: Part[AudioFileStreamTools`MetaTags[obj_, parts1___, opts:OptionsPattern[(* $metaTagsOpts *)]], parts2__] /; (SameQ[Head[obj], AudioFileStreamTools`FileStreamObject]) := 
(
	getMetaTagAssociation[obj, parts1, parts2, opts]
)

AudioFileStreamTools`MetaTags /: f_[ante___, AudioFileStreamTools`MetaTags[obj_, parts___, opts:OptionsPattern[(* $metaTagsOpts *)]], post___] /; (SameQ[Head[obj], AudioFileStreamTools`FileStreamObject] && !MatchQ[f, Set|Unset|Part|Format|AppendTo|PrependTo|AssociateTo]) := 
(
	f[ante, getMetaTagAssociation[obj, parts, opts], post]
)

Options[getMetaTagAssociation] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic, "Level" -> {}};

getMetaTagAssociation[obj_AudioFileStreamTools`FileStreamObject, opts:OptionsPattern[]] := (
	DeleteCases[
		AssociationMap[Replace[getMetaTagAssociation[obj, #, opts], {Except[_?AssociationQ] -> Nothing, <||> -> Nothing}]&, Keys[$tagTypes]]
	, Nothing]
)

getMetaTagAssociation[obj_AudioFileStreamTools`FileStreamObject, tagType_, parts___, opts:OptionsPattern[]] := 
Module[{tag, tagNo = $Failed, keyf, res},
	tag = Check[
		If[(res = addNewTag[obj, tagType, FilterRules[{opts}, Options[addNewTag]]]) =!= $Failed, tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"];];
		If[tagNo =!= $Failed, getFullTag[tagNo, tagType, FilterRules[{opts}, Options[getFullTag]]], tagNo] /. $Failed -> Missing["KeyAbsent", tagType]
		,
		Missing["KeyAbsent", tagType]
	];
	If[!MissingQ[tag], 
		keyf = If[SameQ[tagType, "M4A"], translateTagKey[tagType, #]&, ToUpperCase[translateTagKey[tagType, #]]&];
		tag = Last@Quiet@FoldList[Module[{p},
							If[MatchQ[#1, _Missing], #1,
								Check[If[MatchQ[p = Part[##], _Missing], tagMessage["unsupported"];, p],
									Check[If[AssociationQ[#1] && !MatchQ[p = Part[KeyMap[keyf, #1], keyf[#2]], _Missing], p, tagMessage["unsupported"];],
										Missing["KeyAbsent", #2]
									]
								]
							]]&, tag, {parts}];
	];
	tag
]

Options[getFullTag] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic, "Level" -> {}};

getFullTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, opts:OptionsPattern[]] := getFullTag[getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"], tagType, opts] (* TODO: unused version of function *)
getFullTag[tagNo_, tagType_, opts:OptionsPattern[]] := Module[{tagPair, translate, upperCase},
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	translate = TrueQ[OptionValue@"TranslateKeys"] || (MatchQ[OptionValue@"MetaInformationInterpretation", Automatic|{(_Rule|_RuleDelayed)..}] && !SameQ[OptionValue@"TranslateKeys", False]);
	Return[AssociationMap[(getAllFramesForKeyForTag[tagNo, tagType, #, opts])&, getKeysForTag[tagNo, tagType, "TranslateKeys" -> translate]]];
]

Options[getAllFramesForKeyForTag] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic, "Level" -> {}};

getAllFramesForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, tagID_, opts:OptionsPattern[]] := getAllFramesForKeyForTag[getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"], tagType, tagID, opts]
getAllFramesForKeyForTag[tagNo_, tagType_, tagID_, opts:OptionsPattern[]] := 
Module[{retVal, tagPair, count, frames, type, tagElements, frame},
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	frames = If[MatchQ[OptionValue@"MetaInformationInterpretation", Automatic|{(_Rule|_RuleDelayed)..}],
		interpretRawTags[tagType, tagNo, "Context" -> {tagType, tagID, tagPair, tagNo}, FilterRules[{opts}, Options[interpretRawTags]]];
		(type = #[[2]]["Type"]; If[MissingQ[#[[2]]["IgnoreInterpretedFrame"]] && !MissingQ[frame = #[[2]]["InterpretedFrame"]], frame, fixID3v2Frame[tagType, tagID, #[[2]], "ExpandEmbeddedFrames" -> False]])& /@ getRawFramesForKeyForTag[tagType, tagID, tagPair]
		,
		(type = #[[2]]["Type"]; fixID3v2Frame[tagType, tagID, #[[2]], "ExpandEmbeddedFrames" -> False])& /@ getRawFramesForKeyForTag[tagType, tagID, tagPair]
	];
	Switch[tagType
		,"ID3v2",
		tagElements = $id3v2FramesAssociation[generalizeTagKey[tagID]];
		If[!MissingQ[tagElements] && tagElements["Singleton"], 
			If[frames === {}, None, First[frames]]
			, 
			(frame = #;
			If[AssociationQ[frame] && !MissingQ[frame["EmbeddedFrames"]], 
				AssociateTo[frame, "EmbeddedFrames" -> getFullTag[frame["EmbeddedFrames"], tagType, opts]];
			]; frame)& /@ frames
		]
		,"ID3v1", First[frames]
		,"APE", If[$APETypesAssociation[type] == "Binary", First[frames], frames]
		, "M4A",
			If[StringMatchQ[type, "*List"], frames, First[frames]]
		,_, frames
	]
]

Options[getRawFramesForKeyForTag] = {"CaseSensitive" -> Automatic};

getRawFramesForKeyForTag[tagType_, tagID_, tagPair_, opts:OptionsPattern[]] := Module[{uTagID, compareCase},
	compareCase = TrueQ[OptionValue@"CaseSensitive"] || (SameQ[OptionValue@"CaseSensitive", Automatic] && SameQ[tagType, "M4A"]);
	If[# === {}, #, First[#]]& @
	If[compareCase,
		Last[Reap[Scan[If[SameQ[tagID, #[[2]]["ID"]] || SameQ[tagID, #[[2]]["AliasID"]], Sow[#]]&, Normal@$rawTagContainer[tagType, tagPair[[1]], tagPair[[2]]]]]]
		,
		uTagID = ToUpperCase[tagID];
		Last[Reap[Scan[If[SameQ[uTagID, ToUpperCase[#[[2]]["ID"]]] || SameQ[tagID, #[[2]]["AliasID"]], Sow[#]]&, Normal@$rawTagContainer[tagType, tagPair[[1]], tagPair[[2]]]]]]
	]
]

Options[interpretRawTags] = Options[interpretRawTag] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic, "Context" -> Automatic};

interpretRawTag[tagType_, tag_, tagNo_, tagPair_, opts:OptionsPattern[]] := Module[{elem, newTag = tag, frame, f, context, interpretOpt},
	If[!AssociationQ[tag] || tag === <||>, Return[tag];];
	interpretOpt = OptionValue@"MetaInformationInterpretation";
	If[!MissingQ[newTag["AliasID"]] && SameQ[interpretOpt, None](* !(MatchQ[interpretOpt, Automatic|{(_Rule|_RuleDelayed)..}] && MissingQ[newTag["InterpretedFrame"]]) *), Return[newTag];];
	elem = getElementsAssociation[tagType, newTag["ID"]];
	(* If[MissingQ[elem], Return[newTag];]; *)
	If[!MissingQ[elem] && First@elem["Alias"] =!= None, AssociateTo[newTag, "AliasID" -> First@elem["Alias"]];];
	If[SameQ[tagType, "ID3v2"] && !MissingQ[newTag[["Frame", "EmbeddedFrames"]]], 
		interpretRawTags[tagType, newTag[["Frame", "EmbeddedFrames"]], opts];
	];
	If[!SameQ[interpretOpt, None], newTag = Delete[newTag, "IgnoreInterpretedFrame"];];
	If[MatchQ[interpretOpt, Automatic] && !MissingQ[elem] && !SameQ[elem["InterpretationFunction"], Identity],
		If[!MissingQ[newTag["InterpretedFrame"]] && TrueQ[newTag["DefaultInterpretation"]], Return[newTag];];
		frame = fixID3v2Frame[tagType, newTag["ID"], newTag, "ExpandEmbeddedFrames" -> False, FilterRules[{opts}, Options[fixID3v2Frame]]];
		context = Replace[OptionValue@"Context", Automatic -> {tagType, newTag["ID"], tagPair, tagNo}];
		AssociateTo[newTag, {"InterpretedFrame" -> elem["InterpretationFunction"][frame, context], "DefaultInterpretation" -> True}];
		,
		If[MatchQ[interpretOpt, {(_Rule|_RuleDelayed)..}],
			f = Last@First[Replace[If[SameQ[tagType, "M4A"],
									Select[interpretOpt, (MatchQ[First[#], newTag["ID"]] || MatchQ[First[#], newTag["AliasID"]])&]
									,Select[interpretOpt, (MatchQ[ToUpperCase@First[#], ToUpperCase@newTag["ID"]] || MatchQ[First[#], newTag["AliasID"]])&]
								], {} -> {{Identity}}]];
			If[!SameQ[f, Identity], 
				frame = fixID3v2Frame[tagType, newTag["ID"], newTag, "ExpandEmbeddedFrames" -> True, "MetaInformationInterpretation" -> None (* FilterRules[{opts}, Options[fixID3v2Frame]] *)];
				AssociateTo[newTag, {"InterpretedFrame" -> f[frame], "DefaultInterpretation" -> False}];
				,
				If[SameQ[tagType, "ID3v2"], (* Handle Frame Sub-Elements *)
					elem = DeleteCases[Keys[newTag["Frame"]], "Values"];
					f = Association[Select[interpretOpt, MemberQ[elem, First[#]]&]];
					If[f =!= <||>,
						frame = fixID3v2Frame[tagType, newTag["ID"], newTag, "ExpandEmbeddedFrames" -> True, "MetaInformationInterpretation" -> None (* FilterRules[{opts}, Options[fixID3v2Frame]] *)];
						If[AssociationQ[frame],
							AssociateTo[newTag, {"InterpretedFrame" -> Association[
								Map[(
									elem = f[#[[1]]];
									If[MissingQ[elem], #, #[[1]] -> elem[#[[2]]]]
								)&, Normal[frame]]
							], "DefaultInterpretation" -> False}];
						];
						,
						If[!MissingQ[newTag["InterpretedFrame"]], AssociateTo[newTag, "IgnoreInterpretedFrame"->True];];
					];
					,
					If[!MissingQ[newTag["InterpretedFrame"]], AssociateTo[newTag, "IgnoreInterpretedFrame"->True];];
				];
			];
			,
			If[MatchQ[interpretOpt, Automatic] && MissingQ[elem],
				If[!MissingQ[newTag["InterpretedFrame"]] && TrueQ[newTag["DefaultInterpretation"]], Return[newTag];];
				frame = fixID3v2Frame[tagType, newTag["ID"], newTag, "ExpandEmbeddedFrames" -> False, FilterRules[{opts}, Options[fixID3v2Frame]]];
				AssociateTo[newTag, {"InterpretedFrame" -> unknownStringToInterpretedValue[tagType, newTag["ID"], frame], "DefaultInterpretation" -> True}];
			];
		];
	];
	newTag
]

interpretRawTags[tagType_, tagNo_, opts:OptionsPattern[]] := Module[{tagPair, tags},
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	tags = $rawTagContainer[tagType, tagPair[[1]], tagPair[[2]]];
	$rawTagContainer[tagType, tagPair[[1]], tagPair[[2]]] = interpretRawTag[tagType, #, tagNo, tagPair, opts]& /@ tags
]

Options[fixID3v2Frame] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic, "ExpandEmbeddedFrames" -> True};

fixID3v2Frame[tagType:("ID3v1"|"Xiph"|"APE"|"M4A"), tagID_, elements_Association, opts:OptionsPattern[]] := elements["Frame"]
fixID3v2Frame[tagType:"ID3v2", tagID_, elements_Association, opts:OptionsPattern[]] := 
Module[{frame, key},
	Switch[generalizeTagKey[tagID]
		,"T***", First@elements["Frame", "Values"] (* "T***" frames are documented to be unique *)
		,"W***", elements["Frame", "URL"]
		,_, 
			frame = elements["Frame"];
			If[TrueQ[OptionValue@"ExpandEmbeddedFrames"],
				Join[frame, If[IntegerQ[key = ("EmbeddedFrames" /. frame)], <|"EmbeddedFrames" -> getFullTag[key, "ID3v2", FilterRules[{opts}, Options[getFullTag]]]|>, <||>]]
				,
				frame
			]
	]
]

(* Helper Functions *)

Options[getFrameCountForKeyForTag] = {"Level" -> {}};

getFrameCountForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, tagID_, opts:OptionsPattern[]] := getFrameCountForKeyForTag[getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"], tagType, tagID, opts]
getFrameCountForKeyForTag[tagNo_, tagType_, tagID_, opts:OptionsPattern[]] := 
Module[{tagPair, frames},
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	frames = getRawFramesForKeyForTag[tagType, tagID, tagPair];
	Length[frames]
]

Options[getKeysForTag] = {"ToUpperCase" -> Automatic, "TranslateKeys" -> False, "Level" -> {}};

getKeysForTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, opts:OptionsPattern[]] := getKeysForTag[getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"], tagType, opts]
getKeysForTag[tagNo_, tagType_, opts:OptionsPattern[]] := 
Module[{l, tagPair, keys, upperCase},
    tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	l = {};
	Map[(AppendTo[l, If[TrueQ[OptionValue@"TranslateKeys"] && !MissingQ[#["AliasID"]], #["AliasID"], #["ID"]]])&, $rawTagContainer[tagType, tagPair[[1]], tagPair[[2]]]];
	keys = DeleteDuplicates[l];
	upperCase = TrueQ[OptionValue@"ToUpperCase"] || (!TrueQ[OptionValue@"TranslateKeys"] && SameQ[OptionValue@"ToUpperCase", Automatic] && !SameQ[tagType,"M4A"]);
	If[upperCase, ToUpperCase /@ keys, keys]
]

Options[getTagsAssociation] = {"Level" -> {}};
getTagsAssociation[tagType_, streamID_, opts:OptionsPattern[]] := Switch[tagType,
	"ID3v1", getID3v1TagsAssociation[streamID],
	"ID3v2", getID3v2TagsAssociation[streamID, OptionValue@"Level"],
	"APE", getAPETagsAssociation[streamID],
	"Xiph", getXiphTagsAssociation[streamID],
	"M4A", getM4ATagsAssociation[streamID],
	_, $Failed
]

(* Retrieve Tags From Disk *)

getID3v2TagsAssociation[fsi_, level_] :=
Quiet[Module[{i, data, newLevel, currentMetaTagCount, type, data2, elem, tag, index, frameType},
	currentMetaTagCount = $metaTagCountID3v2;
	$fsiMetaTagKeys["ID3v2", fsi] = Append[$fsiMetaTagKeys["ID3v2", fsi], {"ID3v2", $metaTagCountID3v2}];
	$metaTagAssociation[{"ID3v2", $metaTagCountID3v2++}] = {fsi, level, False};
	frameIndexListID3v2[{}] = Normal@lfFileStreamGetID3v2FramesList[fsi];
	If[frameIndexListID3v2[level] === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3],
	    $rawTagContainer["ID3v2", fsi, level] = <||>;
	    Return[currentMetaTagCount];
	];

	If[Head[frameIndexListID3v2[level]] === LibraryFunctionError,
	    Return[$Failed];
	];

	i = 0; $rawTagContainer["ID3v2", fsi, level] = <|
	Map[( index = Append[level, i]; frameType = #; 
			i -> <|"Type" -> #,
				"Frame" -> DeleteCases[<|

					(data = 
							lfFileStreamGetID3v2ChapterFrameValues[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						"StartTime" -> Nothing[]
						,
						data = Normal@data;
						{"StartTime" -> data[[1]],
							If[data[[2]] =!= -1, "StartOffset" -> data[[2]], "StartOffset" -> Nothing[]]
							,
							"EndTime" -> data[[3]],
							If[data[[4]] =!= -1, "EndOffset" -> data[[4]], "EndOffset" -> Nothing[]]}]),

					"Language" -> (data = 
							lfFileStreamGetID3v2FrameLanguage[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"ChildElements" -> (data =
							lfFileStreamGetID3v2TableOFContentsFrameChildElementCount[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[],
						Map[(FromCharacterCode[Normal@lfFileStreamGetID3v2TableOFContentsFrameChildElementIdentifier[fsi, index, frameType, #]])&, Range[data] - 1]]),

					"Data" -> (data =
							lfFileStreamGetID3v2FrameData[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], toByteArray[Normal@data]]),

					"Owner" -> (data =
							lfFileStreamGetID3v2FrameOwner[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"URL" -> (data =
							lfFileStreamGetID3v2FrameURL[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"Identifier" -> (data =
							lfFileStreamGetID3v2FrameIdentifier[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"Ordered" -> (data =
							lfFileStreamGetID3v2FrameOrdered[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], If[SameQ[data, 0], False, True]]),

					"TopLevel" -> (data =
							lfFileStreamGetID3v2FrameTopLevel[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], If[SameQ[data, 0], False, True]]),

					"SynchedText" -> (data = 
							lfFileStreamGetID3v2FrameSynchedTextList[fsi, index, frameType];
					data2 = lfFileStreamGetID3v2FrameSynchedTextTimes[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError] || SameQ[Head[data2], LibraryFunctionError],
						Nothing[],
						data = FromCharacterCode[SplitBy[Normal@data, SameQ[#, 0] &][[1 ;; ;; 2]]];
						data2 = Normal@data2;
						If[!SameQ[Length[data], Length[data2]],
							Nothing[], <|Map[(data2[[#]] -> data[[#]]) &, Range[Length[data]]]|>]]),

					"LyricsType" -> (data =
							lfFileStreamGetID3v2FrameLyricsType[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], $lyricsTypes[data]]),

					"Channels" -> (data =
							lfFileStreamGetID3v2FrameChannels[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], 
						<|Map[($channelTypes[#] -> <|
							"PeakVolume" -> toByteArray[Normal@lfFileStreamGetID3v2FramePeakVolume[fsi, index, frameType, #]],
							"BitsRepresentingPeak" -> lfFileStreamGetID3v2FramePeakBits[fsi, index, frameType, #],
							"VolumeAdjustment" -> lfFileStreamGetID3v2FrameVolumeAdjustmentIndex[fsi, index, frameType, #]|>)&, Normal@data]|>]),

					"Rating" -> (data =
							lfFileStreamGetID3v2FrameRating[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], data]),

					"Counter" -> (data =
							lfFileStreamGetID3v2FrameCounter[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], data]),

					"Email" -> (data =
							lfFileStreamGetID3v2FrameEmail[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"Seller" -> (data =
							lfFileStreamGetID3v2FrameSeller[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"PurchaseDate" -> (data =
							lfFileStreamGetID3v2FramePurchaseDate[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"PricePaid" -> (data =
							lfFileStreamGetID3v2FramePricePaid[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"PictureType" -> (data =
							lfFileStreamGetID3v2FramePictureType[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], $pictureTypes[data]]),

					"Picture" -> (data =
							lfFileStreamGetID3v2FramePicture[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], toByteArray[Normal@data]]),

					"Object" -> (data =
							lfFileStreamGetID3v2FrameObject[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], toByteArray[Normal@data]]),

					"MimeType" -> (data =
							lfFileStreamGetID3v2FrameMimeType[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"FileName" -> (data =
							lfFileStreamGetID3v2FrameFileName[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"TimestampFormat" -> (data =
							lfFileStreamGetID3v2FrameTimeStampFormat[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], $eventTimestampFormats[data]]),

					"SynchedEvents" -> (data =
							lfFileStreamGetID3v2FrameSynchedEvents[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[],
						data = Normal@data;
						<|Map[(data[[(#*2) - 1]] -> $eventTimingCodes[data[[#*2]]]) &, Range[Length[data]/2]]|>]),

					"Text" -> (data =
							lfFileStreamGetID3v2FrameText[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"Description" -> (data =
							lfFileStreamGetID3v2FrameDescription[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[Normal@data]]),

					"EmbeddedFrames" -> (data =
							lfFileStreamGetID3v2ChapterFrameEmbeddedFramesList[fsi, index, frameType];
					If[SameQ[data, LibraryFunctionError["LIBRARY_FUNCTION_ERROR", 6]], (* This check needs to remain a specific check for LF Function Error, since this is later checked in the next recursive level to see if it is LF error 3: frameIndexListID3v2[level] *)
						Nothing[],
						newLevel = index;
						frameIndexListID3v2[newLevel] = Normal@data;
						getID3v2TagsAssociation[fsi, newLevel]]),

					"Values" -> (data =
							lfFileStreamGetID3v2FrameValues[fsi, index, frameType];
					If[SameQ[Head[data], LibraryFunctionError],
						Nothing[], FromCharacterCode[SplitBy[Normal@data, SameQ[#, 0] &][[1 ;; ;; 2]]]]),

					"Foo" -> Nothing[]|>, Nothing],

				"ID" -> FromCharacterCode[Normal@lfFileStreamGetID3v2FrameID[fsi, Append[level, i++], #]]
			|> )&, frameIndexListID3v2[level]]|>;
	$rawTagContainer["ID3v2", fsi, level]["Parent"] = None;
	Return[currentMetaTagCount];
], {Association::setps, LibraryFunction::dimerr}]

getID3v1TagsAssociation[fsi_] :=
Quiet[Module[{i, data, currentMetaTagCount, rawTag},
	currentMetaTagCount = $metaTagCountID3v1;
	i = 0; rawTag = <|
		DeleteCases[Map[i++ -> If[#["Frame"] === Nothing, Nothing, #]& @<|
			"Frame" -> (data =
					lfFileStreamGetID3v1Element[fsi, $ID3v1Keys[#]];
				If[SameQ[Head[data], LibraryFunctionError],
					Nothing[],
					If[$ID3v1Types[#] == "String", FromCharacterCode[Normal@data], First[Normal@data]]]),
			"ID" -> #|>&, Keys[$ID3v1Keys]], _Integer -> Nothing]|>;

	$fsiMetaTagKeys["ID3v1", fsi] = Append[$fsiMetaTagKeys["ID3v1", fsi], {"ID3v1", $metaTagCountID3v1}];
	$metaTagAssociation[{"ID3v1", $metaTagCountID3v1++}] = {fsi, {}, False};
	$rawTagContainer["ID3v1", fsi, {}] = rawTag;
	Return[currentMetaTagCount];
], {Association::setps}]

getAPETagsAssociation[fsi_] :=
Quiet[Module[{i, ii, key, type, data, currentMetaTagCount, APEKeys, APETypes, values},
	currentMetaTagCount = $metaTagCountAPE;
	$fsiMetaTagKeys["APE", fsi] = Append[$fsiMetaTagKeys["APE", fsi], {"APE", $metaTagCountAPE}];
	$metaTagAssociation[{"APE", $metaTagCountAPE++}] = {fsi, {}, False};
	APEKeys = lfFileStreamGetAPEItemKeys[fsi];

	If[APEKeys === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3],
		$rawTagContainer["APE", fsi, {}] = <||>;
		Return[currentMetaTagCount];
	];

	If[Head[APEKeys] === LibraryFunctionError,
		Return[$Failed];
	];

	APEKeys = SplitBy[Normal@APEKeys, SameQ[#, 0] &][[1 ;; ;; 2]];
	APETypes = Normal@lfFileStreamGetAPEItemTypes[fsi];
	i = ii = 0; $rawTagContainer["APE", fsi, {}] = <|
		Map[(key = APEKeys[[++i]]; 
			type = #;
			values = If[type === $APETypesAssociation["Text"],
						data = lfFileStreamGetAPEItemValues[fsi, stringToRawArray[key]];
						If[SameQ[Head[data], LibraryFunctionError],
							{}, FromCharacterCode[SplitBy[Normal@data, SameQ[#, 0] &][[1 ;; ;; 2]]]]
						,
						data = lfFileStreamGetAPEItemData[fsi, stringToRawArray[key]];
						If[SameQ[Head[data], LibraryFunctionError],
							{}, {toByteArray[Normal@data]}]
					 ];
			 If[values === {}, Nothing[], (ii++ -> <|"Type" -> type, "Frame" -> #, "ID" -> FromCharacterCode[key]|>)& /@ values]
			)&, APETypes]|>;
	Return[currentMetaTagCount];
], {Association::setps, LibraryFunction::dimerr}]

getXiphTagsAssociation[fsi_] :=
Quiet[Module[{i, data, currentMetaTagCount, XiphKeys, values, key},
	currentMetaTagCount = $metaTagCountXiph;
	$fsiMetaTagKeys["Xiph", fsi] = Append[$fsiMetaTagKeys["Xiph", fsi], {"Xiph", $metaTagCountXiph}];
	$metaTagAssociation[{"Xiph", $metaTagCountXiph++}] = {fsi, {}, False};

	XiphKeys = lfFileStreamGetXiphKeys[fsi];
	If[XiphKeys === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3],
		$rawTagContainer["Xiph", fsi, {}] = <||>;
		Return[currentMetaTagCount];
	];

	If[Head[XiphKeys] === LibraryFunctionError,
		Return[$Failed];
	];

	XiphKeys = FromCharacterCode[SplitBy[Normal@XiphKeys, SameQ[#, 0] &][[1 ;; ;; 2]]];
	i = 0; $rawTagContainer["Xiph", fsi, {}] = <|
		Map[(key = #;
			data = lfFileStreamGetXiphValues[fsi, stringToRawArray[key]];
			values = If[SameQ[Head[data],LibraryFunctionError],
						{}, FromCharacterCode[SplitBy[Normal@data, SameQ[#, 0] &][[1 ;; ;; 2]]]];
			If[values === {}, Nothing[], (i++ -> <|"Frame" -> #, "ID" -> key|>)& /@ values]
			)&, XiphKeys]|>;
	Return[currentMetaTagCount];
], {Association::setps, LibraryFunction::dimerr}]

getM4ACoverArtList[fsi_, rawKey_] := Quiet[Module[{data, values, fmt, n, picture, type},
	n = lfFileStreamGetM4AItemCoverArtN[fsi, rawKey];
	If[!PositiveIntegerQ[n], Return[{}];];
	(data = lfFileStreamGetM4AItemCoverArt[fsi, rawKey, #];
	fmt = lfFileStreamGetM4AItemCoverArtFormat[fsi, rawKey, #];
	picture = If[Head[data] =!= LibraryFunctionError, ("Picture" -> ByteArray[Normal[data]]), Nothing]; 
	type = If[Head[fmt] =!= LibraryFunctionError, ("MimeType" -> FromCharacterCode[Normal[fmt]]), Nothing];
	<|picture, type|>
	)& /@ (Range[n] - 1)
]]

getM4ATagsAssociation[fsi_] :=
Quiet[Module[{i, data, currentMetaTagCount, M4AKeys, values, key, rawKey, fmt, n, tmpdata, elements, ItemType, atomType},
	currentMetaTagCount = $metaTagCountM4A;
	$fsiMetaTagKeys["M4A", fsi] = Append[$fsiMetaTagKeys["M4A", fsi], {"M4A", $metaTagCountM4A}];
	$metaTagAssociation[{"M4A", $metaTagCountM4A++}] = {fsi, {}, False};

	M4AKeys = lfFileStreamGetM4AItemKeys[fsi];
	If[M4AKeys === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3],
		$rawTagContainer["M4A", fsi, {}] = <||>;
		Return[currentMetaTagCount];
	];
	If[Head[M4AKeys] === LibraryFunctionError,
		Return[$Failed];
	];

	M4AKeys = FromCharacterCode[SplitBy[Normal@M4AKeys, SameQ[#, 0] &][[1 ;; ;; 2]]];
	i = 0; $rawTagContainer["M4A", fsi, {}] = <|
		Map[(key = #; rawKey = stringToRawArray[key];
			values = {};
			If[!MissingQ[(elements = getElementsAssociation["M4A", key])],
				Switch[(itemType = elements["ItemType"]),
					"Boolean",
						data = lfFileStreamGetM4AItemBoolean[fsi, rawKey];
						If[Head[data] =!= LibraryFunctionError, values = {data};];
					,
					"SignedInteger"|"UnsignedInteger"|"LongInteger"|"Byte",
						data = lfFileStreamGetM4AItemInt[fsi, rawKey, $m4aItemTypesAssoc[elements["ItemType"]]];
						If[Head[data] =!= LibraryFunctionError, values = {data};];
					,
					"CoverArtList",
						values = getM4ACoverArtList[fsi, rawKey];
					,
					"ByteVectorList",
						data = lfFileStreamGetM4AItemBytes[fsi, rawKey];
						If[Head[data] =!= LibraryFunctionError, values = {ByteArray[Normal@data]};];
					,
					"IntegerPair",
						data = lfFileStreamGetM4AItemIntPair[fsi, rawKey];
						If[Head[data] =!= LibraryFunctionError, values = {Normal[data]};];
					,
					_,
						data = lfFileStreamGetM4AItemStrings[fsi, rawKey];
						If[Head[data] =!= LibraryFunctionError, values = FromCharacterCode@SplitBy[Normal@data, SameQ[#, 0]&][[1 ;; ;; 2]];];
						itemType = "StringList";
				];
				,
				Switch[(atomType = $m4aAtomTypesAssoc[lfFileStreamGetM4AItemType[fsi, rawKey]]),
					"Integer"|"QTUnsignedInteger32"|"QTSignedInteger64",
						data = lfFileStreamGetM4AItemInt[fsi, rawKey, -1];
						If[Head[data] =!= LibraryFunctionError, values = {data};];
						itemType = Switch[atomType, "Integer", "SignedInteger", "QTUnsignedInteger32", "UnsignedInteger", "QTSignedInteger64", "LongInteger"];
					,
					"UTF8String"|"UTF16String"|"URL",
						data = lfFileStreamGetM4AItemStrings[fsi, rawKey];
						If[data === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3], 
							data = lfFileStreamGetM4AItemBytes[fsi, rawKey];
						];
						If[Head[data] =!= LibraryFunctionError, values = FromCharacterCode@SplitBy[Normal@data, SameQ[#, 0]&][[1 ;; ;; 2]];];
						itemType = "StringList";
					,
					"JPEG"|"PNG"|"GIF"|"BMP",
						values = getM4ACoverArtList[fsi, rawKey];
						itemType = "CoverArtList";
					,
					_,
						n = lfFileStreamGetM4AItemCoverArtN[fsi, rawKey];
						If[!IntegerQ[n] || n == 0, 
							data = lfFileStreamGetM4AItemStrings[fsi, rawKey];
							If[data === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3], 
								data = lfFileStreamGetM4AItemBytes[fsi, rawKey];
								If[data === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3], 
									data = lfFileStreamGetM4AItemBoolean[fsi, rawKey];
									If[(data === LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3]) || (data =!= 0 && data =!= 1),
										data = lfFileStreamGetM4AItemInt[fsi, rawKey, $m4aItemTypesAssoc["Byte"]];
										If[Head[data] =!= LibraryFunctionError, values = {data};];
										itemType = "Byte";
										,
										If[Head[data] =!= LibraryFunctionError, values = {SameQ[data, 1]};];
										itemType = "Boolean";
									];
									,
									If[Head[data] =!= LibraryFunctionError, values = {ByteArray[Normal@data]};];
									itemType = "ByteVectorList";
								];
								,
								If[Head[data] =!= LibraryFunctionError, values = FromCharacterCode@SplitBy[Normal@data, SameQ[#, 0]&][[1 ;; ;; 2]];];
								itemType = "StringList";
							];
							,
							values = getM4ACoverArtList[fsi, rawKey];
							itemType = "CoverArtList";
						];
				];
			];
			values = Replace[values, _LibraryFunctionError -> {}];
			If[values === {}, 
				i++ -> <|"Frame" -> {}, "ID" -> key, "Type" -> itemType|>
				, 
				(i++ -> <|"Frame" -> #, "ID" -> key, "Type" -> itemType|>)& /@ values
			]
			)&, M4AKeys]|>;
	Return[currentMetaTagCount];
], {Association::setps, LibraryFunction::dimerr}]

