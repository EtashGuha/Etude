LoadMetaTagsResources[dir_] := 
(
	Get[FileNameJoin[{dir, "TagData.m"}]];
	Get[FileNameJoin[{dir, "InterpretationUtilities.m"}]];
	Get[FileNameJoin[{dir, "GetMetaTags.m"}]];
	Get[FileNameJoin[{dir, "SetMetaTags.m"}]];
)

Options[AudioFileStreamTools`MetaTags] = {"MetaInformationInterpretation" -> Automatic};
(* $metaTagsOpts = Join[Options[AudioFileStreamTools`MetaTags], {"TranslateKeys" -> Automatic}]; *)

$tagTypes = <|"ID3v2" -> 0, "ID3v1" -> 1, "Xiph" -> 2, "APE" -> 3, "M4A" -> 4|>;
$tagsKey = "Tags";

(*Set to True to have MetaTag ByteArrays always return empty. Used for testing, this should always be False in production. *)
$testByteArrays = False;

toByteArray[dat_] := Module[{},
    If[$testByteArrays, Return[ByteArray[]];];
    Return[ByteArray[dat]];
];

(* Book-keeping *)

$metaTagCountID3v2 = 1;
$metaTagCountID3v1 = 1;
$metaTagCountAPE = 1;
$metaTagCountXiph = 1;
$metaTagCountM4A = 1;

$metaTagAssociation = Association[];
$fsiMetaTagKeys = Association[
	"ID3v2" -> Association[], 
	"ID3v1" -> Association[], 
	"APE" -> Association[], 
	"Xiph" -> Association[], 
	"M4A" -> Association[]
];
$rootMetaTagAssociationID3v2 = Association[];
$rootMetaTagAssociationID3v1 = Association[];
$rootMetaTagAssociationAPE = Association[];
$rootMetaTagAssociationXiph = Association[];
$rootMetaTagAssociationM4A = Association[];

(* Common Utilties *)

getElementsAssociation[tagType_, tagID_] := Module[{uTagID},
	uTagID = If[SameQ[tagType,"M4A"], #, ToUpperCase[#]]& @ translateTagKey[tagType, tagID];
	Switch[tagType,
		"ID3v2", If[StringMatchQ[uTagID, RegularExpression["(W|T)(?!XXX)..."]], 	
					$id3v2ExtendedFramesAssociation[uTagID], 
					$id3v2FramesAssociation[uTagID]],
		"ID3v1", $id3v1ElementsAssociation[uTagID],
		"APE", $apeElementsAssociation[uTagID],
		"Xiph", $xiphElementsAssociation[uTagID],
		"M4A", $m4aElementsAssociation[uTagID],
		_, Missing["KeyAbsent", tagID]
	]
]

getRootMetaTagAssociation[tagType_, streamID_] := Switch[tagType,
	"ID3v1", $rootMetaTagAssociationID3v1[streamID],
	"ID3v2", $rootMetaTagAssociationID3v2[streamID],
	"APE", $rootMetaTagAssociationAPE[streamID],
	"Xiph", $rootMetaTagAssociationXiph[streamID],
	"M4A", $rootMetaTagAssociationM4A[streamID],
	_, Missing["KeyAbsent", tagType]
]

setRootMetaTagAssociation[tagType_, streamID_, val_] := Switch[tagType,
	"ID3v1", $rootMetaTagAssociationID3v1[streamID] = val;,
	"ID3v2", $rootMetaTagAssociationID3v2[streamID] = val;,
	"APE", $rootMetaTagAssociationAPE[streamID] = val;,
	"Xiph", $rootMetaTagAssociationXiph[streamID] = val;,
	"M4A", $rootMetaTagAssociationM4A[streamID] = val;,
	_, $Failed
]

(* Utility for issuing a variety of descriptive error messages without having a large number of Message tags to Quiet / Check *)
tagMessage[t0_] := tagMessage[t0, t0]
tagMessage[t0_, t1_, args___] :=
	Switch[t0
		,"noset",
			AudioFileStreamTools`MetaTags::noset =
			Switch[t1 
				,"WritePerm", "Unable to modify metadata. Please check your access permissions for the file or directory."
				,"NewTag", "Could not create new `1` tag."
				,"NewFrame", "Could not create new `1` frame for `2` tag."
				,"RemoveFrame", "Could not remove `1` frame for `2` tag."
				,"SetElement", "Could not set `1` frame element for `2` tag."
				,"SetNamedElement", "Could not set `1` frame `2` element for `3` tag."
				,"Reason", "Set operation could not be performed: `1`"
				,"NoUnset", "Unset operation could not be performed."
				,"noset", "Set operation could not be performed."
				,_, t1
			];
			Message[AudioFileStreamTools`MetaTags::noset, Sequence @@ {args}];
		,"invalid",
			AudioFileStreamTools`MetaTags::invalid =
			Switch[t1
				,"TagNotFound", "No `1` tag loaded for stream."
				,"FrameNotFound", "The specified frame was not found: \"`1`\"."
				,"Tag", "Invalid `1` tag."
				,"Frame", "Invalid `1` frame for `2` tag."
				,"Element", "Invalid element for `1` frame for `2` tag."
				,"NamedElement", "Invalid `1` element for `2` frame."
				,"?Tag", "Unrecognized tag type: \"`1`\"."
				,"?Frame", "Unrecognized frame for `1` tag: \"`2`\"."
				,"?Element", "Unrecognized element for `1` frame: \"`2`\"."
				,"?XFrame", "Invalid frame identifier for `1` tag: \"`2`\"."
				,"Singleton", "Only one `1` frame allowed for `2` tag."
				,"FileStreamObject", "`1` is not a valid FileStreamObject."
				,"invalid", "Incorrectly formatted tag."
				,_, t1
			];
			Message[AudioFileStreamTools`MetaTags::invalid, Sequence @@ {args}];
		,"unsupported",
			AudioFileStreamTools`MetaTags::unsupported =
			Switch[t1
				,"TagType", "This audio format does not support `1` tags."
				,"Operation", "This operation is not supported for `1` tags."
				,"InternetStream", "MetaData for Internet stream objects cannot be modified."
				,"InternetStreamMemory", "MetaTag objects are not yet available for streams using a memory buffer (streams not using a file specified with the \"FilePath\" option)."
				,"UnsetElement", "Individual frame elements cannot be removed."
				,"unsupported", "Unsupported operation."
				,_, t1
			];
			Message[AudioFileStreamTools`MetaTags::unsupported, Sequence @@ {args}];
]

Options[addNewTag] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic};

addNewTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, opts:OptionsPattern[]] :=
Module[{streamID, tagPair, key},
	streamID = getStreamID@obj;
	If[!KeyExistsQ[$openStreams, streamID], Message[AudioFileStreamTools`MetaTags::afstnostream, obj]; Return[$Failed]];
	If[getField[streamID, "InternetStreamType"] == "Memory", tagMessage["unsupported", "InternetStreamMemory"]; Return[$Failed]];
	If[MissingQ[(key = $fsiMetaTagKeys[tagType, streamID])], Return[getTagNo[streamID, tagType, True]];];
	If[key === $Failed (*MetaTags::unsupportedtagtype*), Return[$Failed];];
	tagPair = $metaTagAssociation[{tagType, Last[First[key]]}];
	If[tagPair[[3]], createNewTag[obj, tagType, opts], Last[First[key]]]
]

Options[createNewTag] = {"MetaInformationInterpretation" -> Automatic, "TranslateKeys" -> Automatic};

createNewTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, opts:OptionsPattern[]] :=
Module[{streamID, ret, prevKey},
	streamID = getStreamID@obj;
	If[!KeyExistsQ[$openStreams, streamID], Message[AudioFileStreamTools`MetaTags::afstnostream, obj]; Return[$Failed]];
	If[getField[streamID, "InternetStreamType"] == "Memory", tagMessage["unsupported", "InternetStreamMemory"]; Return[$Failed]];
	If[!MissingQ[$fsiMetaTagKeys[tagType, streamID]], KeyDropFrom[$metaTagAssociation, $fsiMetaTagKeys[tagType, streamID]];];
	$fsiMetaTagKeys[tagType, streamID] = {};
	lfFileStreamOpenTags[streamID];
	ret = getTagsAssociation[tagType, streamID];
	If[ret === $Failed,
		tagMessage["unsupported", "TagType", tagType];
		$fsiMetaTagKeys[tagType, streamID] = $Failed (*MetaTags::unsupportedtagtype*);
		lfFileStreamCloseTags[streamID, 0];
		Return[$Failed];
	];
	If[MissingQ[getRootMetaTagAssociation[tagType, streamID]],
		setRootMetaTagAssociation[tagType, streamID, ret];
		,
		prevKey = ret;
		ret = getRootMetaTagAssociation[tagType, streamID];
		$metaTagAssociation[{tagType, ret}] = $metaTagAssociation[{tagType, prevKey}];
		KeyDropFrom[$metaTagAssociation, {tagType, prevKey}];
	];
	interpretRawTags[tagType, ret, opts];
	lfFileStreamCloseTags[streamID, 0];
	Return[ret];
]

reloadTag[tag_] := Quiet[getMetaTagAssociation[AudioFileStreamTools`FileStreamObject[tag[[1]]], tag[[2]]]];

reloadAllTags[exceptTag_] := Module[{tagList},
	tagList = Map[({#[[2]][[1]], #[[1]][[1]]}) &,  Normal[$metaTagAssociation]];
	tagList = DeleteDuplicates[tagList];
	tagList = DeleteCases[tagList, exceptTag];
	Scan[reloadTag[#] &, tagList];
]

removeTagReferences[streamID_] := Module[{tagList = {}},
    Scan[If[#[[2]][[1]] == streamID, AppendTo[tagList, #[[1]]]] &, Normal[$metaTagAssociation]];
    tagList = DeleteDuplicates[tagList];
    KeyDropFrom[$metaTagAssociation, tagList];
]

getTagNo[streamID_, tagType_, createNewTagIfNecessary:(True|False), level_:{}] := 
Module[{tagsForStreamForType, tagPair, tagNo = $Failed, ret},
	tagsForStreamForType = $fsiMetaTagKeys[tagType, streamID];
	If[tagsForStreamForType === $Failed (*MetaTags::unsupportedtagtype*), Return[$Failed];];
	If[MissingQ[tagsForStreamForType],
		If[!createNewTagIfNecessary, Return[$Failed];];
		If[createNewTag[AudioFileStreamTools`FileStreamObject[streamID], tagType, "MetaInformationInterpretation" -> None] === $Failed, 
			Return[$Failed];
			,
			tagsForStreamForType = $fsiMetaTagKeys[tagType, streamID];
			If[MissingQ[tagsForStreamForType], Return[$Failed];];
		];
	];
	If[SameQ[tagType, "ID3v2"],
		Scan[(tagPair = $metaTagAssociation[#]; If[tagPair[[2]] === level || (tagPair[[2]] =!= {} && Take[tagPair[[2]], Length[level]] === level), tagNo = Last[#]; Return[];];) &, tagsForStreamForType];
		,
		tagNo = Last[First[tagsForStreamForType]];
	];
	tagNo
]

getTagNo[___] := $Failed


