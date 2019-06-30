(* Mutators *)

Language`SetMutationHandler[AudioFileStreamTools`MetaTags, AudioFileStreamTools`Private`MetaTagsHandler];
SetAttributes[AudioFileStreamTools`Private`MetaTagsHandler, HoldAllComplete];

(* Clear Tags *)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[obj_]]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject) := 
(
	Quiet[addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[$tagTypes]];
	Quiet[removeTag[obj, #]& /@ Keys[$tagTypes]];
)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[obj_, tagType_]]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject) := 
(
	Quiet[addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[$tagTypes]];
	removeTag[obj, tagType];
)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[obj_, tagType_, tagID_String]]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject) := 
(
	Quiet[addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[$tagTypes]];
	removeAllFramesForKeyForTag[obj, tagType, tagID];
)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[obj_, tagType_, tagID_String, n_Integer]]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject) := 
(
	Quiet[addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[$tagTypes]];
	removeFrameForKeyForTag[obj, tagType, tagID, n];
)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[obj_, tagType_, tagID_, n_Integer, property_String]]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject) := 
(
	tagMessage["unsupported", "UnsetElement"];
)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[obj_, tagType_, tagID_, property_String]]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject) := 
(
	tagMessage["unsupported", "UnsetElement"];
)

AudioFileStreamTools`Private`MetaTagsHandler[Unset[AudioFileStreamTools`MetaTags[___]]] := 
(
	tagMessage["noset", "NoUnset"];
)

(* Set Tags *)

AudioFileStreamTools`Private`MetaTagsHandler[Set[AudioFileStreamTools`MetaTags[obj_], tags_Association]] := 
Module[{res, tagsToRemove, existingTags, warn},
	If[# =!= {}, tagMessage["invalid", "?Tag", First[#]]; Return[tags];]& @ Select[Keys[tags], !MemberQ[Keys[$tagTypes], #]&];
	If[!(And@@(MapThread[validateTag, {Keys[tags], Values[tags]}])), Return[tags];];
	(* Verify that the tag type is supported: *)
	Check[
		res = addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[tags];
		,
		res = $Failed;
		warn = False;
	];
	If[!VectorQ[res, IntegerQ],
		If[warn, tagMessage["noset"]];
		,
		tagsToRemove = Select[Keys[$tagTypes], !MemberQ[Keys[tags], #]&];
		Quiet@addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ tagsToRemove;
		Quiet@removeTag[obj, #]& /@ tagsToRemove;
		existingTags = Quiet[Replace[getKeysForTag[obj, #], {{__} -> #, _ -> Nothing}]& /@ Keys[tags]];
		MapThread[(
			warn = True;
			Check[
				res = setTag[obj, #1, #2];
				,
				res = $Failed;
				warn = False;
			];
			If[res === $Failed,
				If[warn, tagMessage["noset"]];
				If[!MemberQ[existingTags, #1], Quiet@removeTag[obj, #1]];
			];)&, {Keys[tags], Values[tags]}];
	];
	tags
]

AudioFileStreamTools`Private`MetaTagsHandler[Set[AudioFileStreamTools`MetaTags[obj_, tagType_, parts___], value_]] := 
Module[{res, existingTags, warn = True},
	(* Verify that the tag type is supported: *)
	Check[
		res = addNewTag[obj, tagType, "MetaInformationInterpretation" -> None];
		,
		res = $Failed;
		warn = False;
	];
	If[res === $Failed,
		If[warn, tagMessage["noset"]];
		Return[value];
	];
	existingTags = Replace[getKeysForTag[obj, tagType], Except[_List] -> {}];
	Check[
		res = setTag[obj, tagType, parts, value];
		,
		res = $Failed;
		warn = False;
	];
	If[res === $Failed, 
		If[warn, tagMessage["noset"]];
		Which[
			{parts} === {} && existingTags === {},
				Quiet@removeTag[obj, tagType];
			,
			Length@{parts} > 0 && !MemberQ[existingTags, If[SameQ[tagType, "M4A"], #, ToUpperCase@#]& @ translateTagKey[tagType, First@{parts}]],
				Quiet@removeAllFramesForKeyForTag[obj, tagType, First@{parts}];
		];
	];
	value
]

(* Catch-all to issue MetaTags "Set cannot be performed" Message instead of default Set message *)
AudioFileStreamTools`Private`MetaTagsHandler[Set[AudioFileStreamTools`MetaTags[___], x_]] := 
(
	tagMessage["noset"];
	x
)

AudioFileStreamTools`Private`MetaTagsHandler[mutator_[AudioFileStreamTools`MetaTags[obj_, parts___, opts:OptionsPattern[AudioFileStreamTools`MetaTags(* $metaTagsOpts *)]], args___]] /; (Head[obj] === AudioFileStreamTools`FileStreamObject && MatchQ[mutator, AppendTo|PrependTo|AssociateTo]) := 
Module[{res, res2, tag, argsAssoc, warn = True},
	Quiet[Check[
		(* Verify that the tag type is supported: *)
		Check[
			Which[
				Length@{parts} > 0, 
					res = addNewTag[obj, First@{parts}, "MetaInformationInterpretation" -> None];
				, 
				AssociationQ[args] && Complement[Keys[args], Keys[$tagTypes]] === {},
					res = addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[args];
					If[MemberQ[res, $Failed], res = $Failed];
				,
				True,
					res = $Failed;
			];
			,
			res = $Failed;
			warn = False;
		];
		If[res === $Failed,
			If[warn, tagMessage["noset"]];
			Return[args];
		];
		tag = getMetaTagAssociation[obj, parts, "MetaInformationInterpretation" -> None, "TranslateKeys" -> (OptionValue[{opts}, "MetaInformationInterpretation"] =!= None)];
		res = Switch[mutator
			,AppendTo,
				Switch[tag
					,_List, Join[tag, {args}]
					,_Association, Join[tag, <|args|>]
					,_Missing, AppendTo[Missing["KeyAbsent", Last[tag]], args]
					,_, $Failed
				]
			,PrependTo,
				Switch[tag
					,_List, Join[{args}, tag]
					,_Association, Join[<|args|>, tag]
					,_Missing, PrependTo[Missing["KeyAbsent", Last[tag]], args]
					,_, $Failed
				]
			,AssociateTo,
				If[AssociationQ[tag],
					argsAssoc = <|Sequence@@If[MatchQ[{args}, {_Rule}], {args}, If[MatchQ[{args}, {{_Rule, __}}], args, {}]]|>;
					If[argsAssoc === <||>, $Failed, Join[tag, argsAssoc]]
					,
					$Failed
				]
			,_, $Failed
		];
		,
		(* the message might not be fatal *)
		If[res === $Failed || MissingQ[res] || Head[res] === Join || Head[res] === mutator, 
			tagMessage["noset"]; 
			Return[If[res === $Failed, args, res]];
		];
	];, All, {AudioFileStreamTools`MetaTags::noset}];
	If[res === $Failed || MissingQ[res] || Head[res] === Join || Head[res] === mutator, 
		tagMessage["noset"];
		,
		Check[
			res2 = setTag[obj, parts, res];
			,
			res2 = $Failed;
			warn = False;
		];
		If[res2 === $Failed,
			If[warn, tagMessage["noset"]];
			If[MatchQ[tags, {__} || <|__|>], Quiet@removeTag[obj, parts]];
		];
	];
	If[res === $Failed, args, res]
]

Options[setTag] = {"Level" -> {}};
setTag[obj_AudioFileStreamTools`FileStreamObject, tagType_String, tag_Association, opts:OptionsPattern[]] :=
Module[{tagNo, keysToRemove, keysToAdd, tagPair, removeTagOnFailure = False, ret = {}},
	If[!validateTag[tagType, tag], Return[$Failed];];
	If[(tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"]) =!= $Failed,
		keysToAdd = (If[SameQ[tagType,"M4A"], #, ToUpperCase[#]]& @ translateTagKey[tagType, #])& /@ Keys[tag];
		While[{} =!= (keysToRemove = Select[getKeysForTag[obj, tagType, opts], !MemberQ[keysToAdd, #]&]), Scan[removeAllFramesForKeyForTag[obj, tagType, #, opts]&, keysToRemove];];
		,
		removeTagOnFailure = True;
	];
	If[addNewTag[obj, tagType] === $Failed, Return[$Failed];];
	tagNo = getTagNo[getStreamID@obj, tagType, True, OptionValue@"Level"];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	MapThread[AppendTo[ret, setTag[obj, tagType, #1, #2, "Level" -> tagPair[[2]]]]&, {Keys[tag], Values[tag]}];
	If[MemberQ[ret, $Failed], If[removeTagOnFailure, removeTag[obj, tagType]]; Return[$Failed];];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v1", tagID_String, frame_, opts:OptionsPattern[]] :=
Module[{tagNo},
	If[!validateFrameElements[tagType, tagID, frame], Return[$Failed];];
	If[(tagNo = getTagNo[getStreamID@obj, tagType, True]) === $Failed, tagMessage["noset", "NewTag", tagType]; Return[$Failed];];
	If[setElementForFrameForKeyForTag[obj, tagType, tagID, frame] === $Failed, 
		removeFrameForKeyForTag[obj, tagType, tagID]; 
		tagMessage["noset", "SetElement", tagID, tagType]; 
		Return[$Failed];
	];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"APE", tagID_String, frame_, opts:OptionsPattern[]] :=
Module[{tagNo, frameType, removeTagOnFailure = False, removeFrameOnFailure = False, ret},
	If[$Failed === (frameType = getApeFrameType[tagID, frame]), Return[$Failed];];
	If[!validateFrameElements[tagType, tagID, frame], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[getFrameCountForKeyForTag[obj, tagType, tagID] === 0, 
		If[addFrameForKeyForTag[obj, tagType, tagID, frameType] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType];]; tagMessage["noset", "NewFrame", tagID, tagType]; Return[$Failed];];
		removeFrameOnFailure = True;
	];
	ret = If[frameType === "Binary",
		setElementForFrameForKeyForTag[obj, tagType, tagID, frame]
		, setElementForFrameForKeyForTag[obj, tagType, tagID, If[ListQ[frame], frame, {frame}], "MultipleElements" -> True]
	];
	If[ret === $Failed, If[removeTagOnFailure, removeTag[obj, tagType];, If[removeFrameOnFailure, removeFrameForKeyForTag[obj, tagType, tagID];];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[$Failed];];	
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"APE", tagID_String, n_Integer?Positive, frame_, opts:OptionsPattern[]] :=
Module[{tagNo, frameType, removeTagOnFailure = False, removeFrameOnFailure = False, frameList = frame},
	If[$Failed === (frameType = getApeFrameType[tagID, frame]), Return[$Failed];];
	If[!validateFrameElements[tagType, tagID, frame], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[getFrameCountForKeyForTag[obj, tagType, tagID] === 0, 
		If[addFrameForKeyForTag[obj, tagType, tagID, frameType] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType];]; tagMessage["noset", "NewFrame", tagID, tagType]; Return[$Failed];];
		removeFrameOnFailure = True;
		,
		frameList = getAllFramesForKeyForTag[obj, tagType, tagID, "MetaInformationInterpretation" -> None];
		If[Length[frameList] >= n, frameList[[n]] = frame, AppendTo[frameList, frame]];
	];
	If[setElementForFrameForKeyForTag[obj, tagType, tagID, frameList, "MultipleElements" -> True] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType];, If[removeFrameOnFailure, removeFrameForKeyForTag[obj, tagType, tagID];];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[$Failed];];	
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"Xiph", tagID_String, frame_, opts:OptionsPattern[]] :=
Module[{tagNo, removeTagOnFailure = False},
	If[!validateFrameElements[tagType, tagID, frame], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[setElementForFrameForKeyForTag[obj, tagType, tagID, frame, "MultipleElements" -> ListQ[frame]] === $Failed, 
		If[removeTagOnFailure, removeTag[obj, tagType];];
		tagMessage["noset", "SetElement", tagID, tagType];
		Return[$Failed];
	];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"Xiph", tagID_String, n_Integer?Positive, value_, opts:OptionsPattern[]] := 
Module[{tagNo, fr, removeTagOnFailure = False},
	If[!validateFrameElements[tagType, tagID, value], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	fr = getAllFramesForKeyForTag[obj, tagType, tagID, "MetaInformationInterpretation" -> None];
	If[n > Length[fr], AppendTo[fr, value], fr[[n]] = value];
	If[setElementForFrameForKeyForTag[obj, tagType, tagID, fr, "MultipleElements" -> True] === $Failed, 
		If[removeTagOnFailure, removeTag[obj, tagType];];
		tagMessage["noset", "SetElement", tagID, tagType];
		Return[$Failed];
	];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"M4A", tagID_String, frame_List, opts:OptionsPattern[]] /; !MatchQ[frame, {_Integer, _Integer}] := 
Module[{tagNo, removeTagOnFailure = False, res},
	If[!validateFrameElements[tagType, tagID, frame], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[!removeTagOnFailure, Quiet@removeAllFramesForKeyForTag[obj, tagType, tagID];];
	res = MapThread[setElementForFrameForKeyForTag[obj, tagType, tagID, #1, #2, opts]&, {Range[Length[frame]], frame}];
	If[MemberQ[res, $Failed], If[removeTagOnFailure, removeTag[obj, tagType];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[$Failed];];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"M4A", tagID_String, frame_, opts:OptionsPattern[]] := setTag[obj, tagType, tagID, 1, frame, True, opts]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"M4A", tagID_String, n_Integer?Positive, value_, removeExisting:(True|False):False, opts:OptionsPattern[]] := 
Module[{tagNo, removeTagOnFailure = False, res},
	If[!validateSingleFrameElement[tagType, tagID, n, value], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[!removeTagOnFailure && removeExisting, Quiet@removeAllFramesForKeyForTag[obj, tagType, tagID];];
	res = setElementForFrameForKeyForTag[obj, tagType, tagID, n, value];
	If[res === $Failed, If[removeTagOnFailure, removeTag[obj, tagType];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[$Failed];];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, frame:(_?AssociationQ|_?(MatchQ[#,{_?AssociationQ..}]&)), opts:OptionsPattern[]] :=
Module[{tagNo, frameList, frameCount = 0, res = Null},
	If[!validateFrame[tagType, tagID, frame], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	frameList = If[ListQ[frame], frame, {frame}];
	(* If tag exists, remove extra Frames. *)
	If[!removeTagOnFailure, While[getFrameCountForKeyForTag[obj, tagType, tagID, opts] > Length[frameList], removeFrameForKeyForTag[obj, tagType, tagID, opts];];];
	Scan[If[$Failed === setTag[obj, tagType, tagID, ++frameCount, #, opts], res = $Failed;]&, frameList];
	res
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, frame_, opts:OptionsPattern[]] :=
Module[{tagNo, removeTagOnFailure = False, removeFrameOnFailure = False, rawFrame},
	If[!validateFrame[tagType, tagID, frame], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[getFrameCountForKeyForTag[obj, tagType, tagID, opts] == 0, 
		If[addFrameForKeyForTag[obj, tagType, tagID, opts] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType, opts];]; tagMessage["noset", "NewFrame", tagID, tagType]; Return[$Failed];];
		removeFrameOnFailure = True;
	];
	If[setElementForFrameForKeyForTag[obj, tagType, tagID, frame, opts] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType, opts];, If[removeFrameOnFailure, removeAllFramesForKeyForTag[obj, tagType, tagID, opts];];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[$Failed];];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, n_Integer?Positive, elements_Association, opts:OptionsPattern[]] := 
Module[{tagNo, nFrame, fr = n, removeTagOnFailure = False, removeFrameOnFailure = False, res = Null},
	If[!validateFrame[tagType, tagID, elements], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[n > (nFrames = getFrameCountForKeyForTag[obj, tagType, tagID, opts]), 
		If[addFrameForKeyForTag[obj, tagType, tagID, opts] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType, opts];]; tagMessage["noset", "NewFrame", tagID, tagType]; Return[$Failed];];
		removeFrameOnFailure = True;
		fr = nFrames + 1;
	];
	Scan[(If[$Failed === setTag[obj, tagType, tagID, fr, Sequence@@#, opts], 
			res = $Failed; If[removeTagOnFailure, removeTag[obj, tagType, opts];, If[removeFrameOnFailure, removeFrameForKeyForTag[obj, tagType, tagID, fr, opts];];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[];
		];)&, Normal[elements]];
	res
]

setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, property_String, values__, opts:OptionsPattern[]] := setTag[obj, "ID3v2", tagID, 1, property, values, opts]
setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, n_Integer?Positive, property_String, values__, opts:OptionsPattern[]] := 
Module[{tagNo, fr, frameIndex = n, removeTagOnFailure = False, removeFrameOnFailure = False, res, tagPair, count = 0, uTagID, rawValues},
	If[!validateSingleFrameElement[tagType, tagID, n, property, values], Return[$Failed];];
	{tagNo, removeTagOnFailure} = setTagGetTagNo[obj, tagType, OptionValue@"Level"];
	If[tagNo === $Failed, Return[$Failed];];
	If[n > (fr = getFrameCountForKeyForTag[obj, tagType, tagID, opts]),
		If[addFrameForKeyForTag[obj, tagType, tagID, opts] === $Failed, If[removeTagOnFailure, removeTag[obj, tagType, opts];]; tagMessage["noset", "NewFrame", tagID, tagType]; Return[$Failed];];
		removeFrameOnFailure = True;
		frameIndex = fr + 1;
		tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"];
	];
	If[property === "EmbeddedFrames",
		tagPair = $metaTagAssociation[{tagType, tagNo}];
		res = getRawFramesForKeyForTag[tagType, tagID, tagPair];
		tagNo = res[[n, 2, "Frame", "EmbeddedFrames"]];
		tagPair = $metaTagAssociation[{tagType, tagNo}];
		res = setTag[obj, tagType, values, "Level" -> tagPair[[2]]];
		,
		res = setElementForFrameForKeyForTag[obj, tagType, tagID, frameIndex, property, values, opts];
	];
	If[res === $Failed, If[removeTagOnFailure, removeTag[obj, tagType, opts];, If[removeFrameOnFailure, removeFrameForKeyForTag[obj, tagType, tagID, n, opts];];]; tagMessage["noset", "SetElement", tagID, tagType]; Return[$Failed];];
]

setTag[obj_AudioFileStreamTools`FileStreamObject, t_, ___] /; (t =!= "ID3v2" && t =!= "ID3v1" && t =!= "APE" && t =!= "Xiph" && t =!= "M4A") := (tagMessage["invalid", "?Tag", t]; $Failed)
setTag[obj_AudioFileStreamTools`FileStreamObject, tagType:("ID3v2"|"ID3v1"|"APE"|"Xiph"|"M4A"), tagID_, ___] /; !StringQ[tagID] := (tagMessage["invalid", "?XFrame", tagType, tagID]; $Failed)
setTag[o_, ___] /; (Head[o] =!= AudioFileStreamTools`FileStreamObject) := (tagMessage["invalid", "FileStreamObject", o]; $Failed)
setTag[___] := $Failed

(* Tag Validation Functions *)

validateTag[tagType:("ID3v2"|"ID3v1"|"Xiph"|"APE"|"M4A"), tag_Association] := (
	If[# =!= {}, tagMessage["invalid", "?Frame", tagType, First[#]]; Return[False];]& @ Select[Keys[tag], !StringQ[#]&];
	Return[ And@@MapThread[validateFrame[tagType, ##]&, {Keys[tag], Values[tag]}]];
)
validateTag[tagType:("ID3v2"|"ID3v1"|"Xiph"|"APE"|"M4A"), tag_] /; !AssociationQ[tag] := (tagMessage["invalid", "Tag", tagType]; False)
validateTag[tagType_, ___] /; (tagType =!= "ID3v2" && tagType =!= "ID3v1" && tagType =!= "Xiph" && tagType =!= "APE" && tagType =!= "M4A") := (tagMessage["invalid", "?Tag", tagType]; False)
validateTag[___] := (tagMessage["invalid"]; False)

validateFrame[tagType:"ID3v2", tagID_String, frame_] :=
Module[{genTagKey = generalizeTagKey[tagID], tag, frameList},
	tag = $id3v2FramesAssociation[genTagKey];
	If[MissingQ[tag], tagMessage["invalid", "?Frame", tagType, tagID]; Return[False];];
	If[(genTagKey === "T***" || genTagKey === "W***"),
		If[!MissingQ[(tag = $id3v2ExtendedFramesAssociation[translateTagKey[tagType, tagID]])], 
			Return[If[!tag["ValidationFunction"][frame], tagMessage["invalid", "Element", tagID, tagType]; False, True]];
			,
			Return[If[!StringQ[frame], tagMessage["invalid", tagID<>" frame must be a String."]; False, True]];
		];
		,
		frameList = If[ListQ[frame], frame, {frame}];
		If[!(And@@(AssociationQ /@ frameList)), tagMessage["invalid", tagID<>" frame must be an Association or List of Associations."]; Return[False];];
		If[Length[frameList] > 1 && tag["Singleton"], tagMessage["invalid", "Singleton", tagID, tagType]; Return[False];];
		Return[And@@Map[And@@MapThread[(validateFrameElements[tagType, tagID, #1, #2])&, {Keys[#], Values[#]}]&, frameList]];
	];
]
validateFrame[tagType_, tagID_String, frame_] := validateFrameElements[tagType, tagID, frame]
validateFrame[tagType_, tagID_, frame_, ___] /; !StringQ[tagID] := (tagMessage["invalid", "?Frame", tagType, tagID]; False)
validateFrame[___] := (tagMessage["invalid"]; False)

validateFrameElements[tagType:"ID3v2", tagID_String, property:"EmbeddedFrames", values_Association] := validateTag[tagType, values]
validateFrameElements[tagType:"ID3v2", tagID_String, property:"EmbeddedFrames", values__] := validateFrame[tagType, values]
validateFrameElements[tagType:"ID3v2", tagID_String, property_String, value_] := Module[{genTagKey = generalizeTagKey[tagID]},
	If[MissingQ[$id3v2FramesAssociation[genTagKey]], tagMessage["invalid", "?Frame", tagType, tagID]; Return[False];];
	If[MissingQ[$id3v2ElementsAssociation[property]], tagMessage["invalid", "?Element", tagID, property]; Return[False];];
	If[(genTagKey === "T***" || genTagKey === "W***"),
		If[!MissingQ[(tag = $id3v2ExtendedFramesAssociation[translateTagKey[tagType, tagID]])], 
			Return[If[!tag["ValidationFunction"][frame], tagMessage["invalid", "Element", tagID, tagType]; False, True]];
			,
			Return[If[!StringQ[frame], tagMessage["invalid", tagID<>" frame must be a String."]; False, True]];
		];
		,
		(* TODO: Interpretation/Uninterpretation of unknown (unlisted) elements? *)
		Return[If[!$id3v2ElementsAssociation[property]["ValidationFunction"][value], tagMessage["invalid", "NamedElement", property, tagID]; False, True]];
	];
]
validateFrameElements[tagType:"ID3v1", tagID_?StringQ, frameElements_List] := (tagMessage["invalid", "Singleton", tagID, tagType]; False)
validateFrameElements[tagType:"M4A", tagID_?StringQ, fe_List] /; !MatchQ[fe, {_Integer,_Integer}] := Module[{elem, res},
	elem = getElementsAssociation[tagType, tagID];
	If[MissingQ[elem],
		If[!validateTagKey[tagType, tagID], tagMessage["invalid", "?XFrame", tagType, tagID]; Return[False];];
		res = validateUnknownStringOrInterpretedValue[tagType, tagID, fe];
		,
		If[!MissingQ[elem["ItemType"]] && !StringMatchQ[elem["ItemType"], "*List"], tagMessage["invalid", "Singleton", tagType, tagID]; Return[False];];
		If[!MissingQ[elem["ValidationFunction"]],
			res = VectorQ[fe, elem["ValidationFunction"][#]&];
			,
			If[!MissingQ[elem["ItemType"]],
				res = VectorQ[fe, validateM4AFrameByItemType[elem["ItemType"], #]&];
				,
				res = VectorQ[fe, StringQ];
			];
		];
	];
	Return[If[!res, tagMessage["invalid", "Element", tagID, tagType]; False, True]];
]
validateFrameElements[tagType:("ID3v1"|"Xiph"|"APE"|"M4A"), tagID_?StringQ, fe_] := Module[{elem, res},
	elem = getElementsAssociation[tagType, tagID];
	If[MissingQ[elem] || MissingQ[elem["ValidationFunction"]], 
		If[MissingQ[elem] && !validateTagKey[tagType, tagID], tagMessage["invalid", "?XFrame", tagType, tagID]; Return[False];];
		Switch[tagType,
			"ID3v1", tagMessage["invalid", "?Frame", tagType, tagID]; Return[False];,
			"APE", res = If[!MissingQ[elem], zOrZsQ[fe, StringQ], validateUnknownStringOrInterpretedValue[tagType, tagID, fe]];,
			"Xiph", res = If[!MissingQ[elem], zOrZsQ[fe, StringQ], validateUnknownStringOrInterpretedValue[tagType, tagID, fe]];,
			"M4A", res = If[!MissingQ[elem],
							If[!MissingQ[elem["ItemType"]], 
								validateM4AFrameByItemType[elem["ItemType"], fe],
								zOrZsQ[fe, StringQ]
							],
							validateUnknownStringOrInterpretedValue[tagType, tagID, fe]
						];
		];
		,
		res = zOrZsQ[fe, elem["ValidationFunction"][#]&];
	];
	Return[If[!TrueQ[res], tagMessage["invalid", "Element", tagID, tagType]; False, True]];
]
validateFrameElements[tagType:("ID3v1"|"Xiph"|"APE"|"M4A"), tagID_, ___] /; !StringQ[tagID] := (tagMessage["invalid", "?XFrame", tagType, tagID]; False)
validateFrameElements[tagType_, ___] /; (tagType =!= "ID3v2" && tagType =!= "ID3v1" && tagType =!= "Xiph" && tagType =!= "APE" && tagType =!= "M4A") := (tagMessage["invalid", "?Tag", tagType]; False)
validateFrameElements[___] := (tagMessage["invalid"]; False)

validateSingleFrameElement[tagType:"ID3v2", tagID_String, n_Integer, property_String, values__] := 
Module[{genTagKey = generalizeTagKey[tagID]},
	If[(genTagKey === "T***" || genTagKey === "W***"), tagMessage["invalid", "?Element", tagID, property]; Return[False];];
	If[n > 1 && $id3v2FramesAssociation[genTagKey]["Singleton"], tagMessage["invalid", "Singleton", tagID, tagType]; Return[False];];
	Return[validateFrameElements[tagType, tagID, property, values]];
]
validateSingleFrameElement[tagType:"M4A", tagID_String, n_Integer, frame_] :=
Module[{elem, res},
	elem = getElementsAssociation[tagType, tagID];
	If[!MissingQ[elem],
		If[n > 1 && !MissingQ[elem["ItemType"]] && !StringMatchQ[elem["ItemType"], "*List"], tagMessage["invalid", "Singleton", tagID, tagType]; Return[False];];
		res = If[!MissingQ[elem["ValidationFunction"]], elem["ValidationFunction"][frame], StringQ[frame]];
		Return[If[!res, tagMessage["invalid", "Element", tagID, tagType]; False, True]];
		,
		If[!validateTagKey[tagType, tagID], tagMessage["invalid", "?XFrame", tagType, tagID]; Return[False];];
		Return[validateUnknownStringOrInterpretedValue[tagType, tagID, frame]];
	];
]
validateSingleFrameElement[___] := (tagMessage["invalid"]; False)

m4aIllegalItemKeys = {"clip", "crgn", "matt", "kmat", "pnot", "ctab", "load", "imap", "tmcd", "chap", "sync", "scpt", "ssrc"};
apeIllegalItemKeys = ToUpperCase/@{"ID3", "TAG", "OggS", "MP+"};
(* case-sensitive *)
validateTagKey[tagType:"M4A", strKey_?StringQ] := (
	!MemberQ[m4aIllegalItemKeys, strKey] 
	&& (StringLength[strKey] == 4) 
	&& ((First[#] > 31 && Last[#] < 256)& @ MinMax[ToCharacterCode[strKey]]))
(* case-insensitive *)
validateTagKey[tagType:"ID3v2", strKey_?StringQ] := (
	(StringLength[strKey] == 4) 
	&& ((First[#] > 31 && Last[#] < 126)& @ MinMax[ToCharacterCode[strKey]]))
(* case-insensitive *)
validateTagKey[tagType:"Xiph", strKey_?StringQ] := (
	(((First[#] > 31 && Last[#] < 126)& @ MinMax[#]) && !MemberQ[#, 61])& @ ToCharacterCode[strKey])
(* case-insensitive *)
validateTagKey[tagType:"APE", strKey_?StringQ] := (
	!MemberQ[apeIllegalItemKeys, ToUpperCase@strKey] 
	&& ((((First[#] > 31 && Last[#] < 127)& @ MinMax[#]) && ((# > 1 && # < 256)& @ Length[#]))& @ ToCharacterCode[strKey]))
(* case-insensitive *)
validateTagKey[tagType:"ID3v1", strKey_?StringQ] := (
	MemberQ[Keys[$ID3v1Keys], ToUpperCase@strKey])
validateTagKey[___] := False

validateM4AFrameByItemType[itemType_, frame_] := Switch[itemType,
	"ByteVectorList",
		ByteArrayQ[frame]
	,
	"StringList",
		StringQ[frame]
	,
	"CoverArtList",
		ImageQ[frame] || ByteArrayQ[frame]
	,
	"Byte",
		Internal`NonNegativeIntegerQ[frame] && frame < 256
	,
	"Boolean",
		MatchQ[frame, True|False|0|1]
	,
	"SignedInteger",
		IntegerQ[frame]
	,
	"UnsignedInteger",
		Internal`NonNegativeIntegerQ[frame]
	,
	"LongInteger",
		IntegerQ[frame]
	,
	"IntegerPair",
		MatchQ[frame, {_Integer,_Integer}]
	,
	_,
		False
]

(* Removing Tags and parts of tags *)

Options[removeTag] = {"Level" -> {}};

removeTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, opts:OptionsPattern[]] := 
Module[{tagNo, tagPair, res, tag, baseTagPair, baseTagNo, streamID},
	tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	streamID = tagPair[[1]];
	If[getField[streamID, "InternetStream"] == True, tagMessage["unsupported", "InternetStream"]; Return[$Failed]];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];

	$metaTagAssociation[{tagType, tagNo}] = {streamID, tagPair[[2]], True};
	If[SameQ[tagType, "ID3v2"],
		baseTagNo = getTagNo[getStreamID@obj, tagType, False];
		baseTagPair = $metaTagAssociation[{tagType, baseTagNo}];
		$metaTagAssociation[{tagType, baseTagNo}] = {streamID, baseTagPair[[2]], True};
	];
	lfFileStreamOpenTags[streamID];
	res = lfFileStreamRemoveTag[streamID, $tagTypes[tagType]];
	lfFileStreamCloseTags[streamID, 1];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];
	If[res == LibraryFunctionError["LIBRARY_FUNCTION_ERROR", 6],
		Switch[tagType,
			"ID3v2",
				While[Keys[(tag = getMetaTagAssociation[obj, tagType, "MetaInformationInterpretation"->None, opts])] =!= {}, Scan[removeFrameForKeyForTag[obj, tagType, #, opts]&, Keys[tag]]];
			, "ID3v1",
				Scan[removeFrameForKeyForTag[obj, tagType, #]&, getKeysForTag[obj, tagType]];
			, _,
				Scan[removeAllFramesForKeyForTag[obj, tagType, #]&, getKeysForTag[obj, tagType]];
		];
	];
	reloadAllTags[{streamID, tagType}];
	Return[addNewTag[obj, tagType]];
]

Options[removeAllFramesForKeyForTag] = {"RefreshTags" -> True, "Level" -> {}};

removeAllFramesForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, tagID_, opts:OptionsPattern[]] := 
Module[{tagNo, tagPair, uTagID, level, streamID},
	level = OptionValue@"Level";
	tagNo = getTagNo[getStreamID@obj, tagType, False, level];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	uTagID = If[SameQ[tagType,"M4A"], #, ToUpperCase[#]]& @ translateTagKey[tagType, tagID];
	If[!MemberQ[getKeysForTag[obj, tagType, "Level"->level], uTagID], tagMessage["invalid", "FrameNotFound", tagID]; Return[$Failed]];
	If[SameQ[tagType, "Xiph"],
		lfFileStreamOpenTags[(streamID = getStreamID@obj)];
		lfFileStreamRemoveXiphKey[streamID, stringToRawArray[uTagID]];
		lfFileStreamCloseTags[streamID, 1];
		,
		While[True, 
			If[removeFrameForKeyForTag[obj, tagType, uTagID, (* "RefreshTags" -> False, *) "Level"->level] === $Failed, tagMessage["noset", "RemoveFrame", tagID, tagType]; Break[];];
			If[!MemberQ[getKeysForTag[obj, tagType, "Level"->level], uTagID], Break[];];
		];
	];
	If[TrueQ[OptionValue@"RefreshTags"], $metaTagAssociation[{tagType, tagNo}] = {tagPair[[1]], tagPair[[2]], True};];
	reloadAllTags[{tagPair[[1]], tagType}];
	Return[addNewTag[obj, tagType]];
]

Options[removeFrameForKeyForTag] = {"RefreshTags" -> True, "Level" -> {}};

removeFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, tagID_, opts:OptionsPattern[]] := removeFrameForKeyForTag[obj, tagType, tagID, 1, opts]
removeFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType_, tagID_, n_Integer, opts:OptionsPattern[]] := 
Module[{tagNo, count = 0, tagPair, rawFrames, index = None, values, uTagID, baseTagPair, baseTagNo, streamID, rawKey},
	tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	streamID = tagPair[[1]];
	If[getField[streamID, "InternetStream"] == True, tagMessage["unsupported", "InternetStream"]; Return[$Failed]];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];

	rawFrames = getRawFramesForKeyForTag[tagType, tagID, tagPair];
	If[Length[rawFrames] < n, tagMessage["invalid", "FrameNotFound", tagID]; Return[$Failed]];
	index = rawFrames[[n, 1]];

	$metaTagAssociation[{tagType, tagNo}] = {streamID, tagPair[[2]], True};
	If[SameQ[tagType, "ID3v2"],
		baseTagNo = getTagNo[getStreamID@obj, tagType, False];
		baseTagPair = $metaTagAssociation[{tagType, baseTagNo}];
		$metaTagAssociation[{tagType, baseTagNo}] = {streamID, baseTagPair[[2]], True};
	];

	uTagID = If[SameQ[tagType,"M4A"], #, ToUpperCase[#]]& @ translateTagKey[tagType, tagID];

	lfFileStreamOpenTags[streamID];
	Switch[tagType,
	"ID3v1",
		Switch[uTagID,
			"YEAR", lfFileStreamSetID3v1Element[streamID, $ID3v1Keys[uTagID], RawArray["UnsignedInteger32",{0}]];,
			"TRACK", lfFileStreamSetID3v1Element[streamID, $ID3v1Keys[uTagID], RawArray["UnsignedInteger32",{0}]];,
			"GENRE", lfFileStreamSetID3v1Element[streamID, $ID3v1Keys[uTagID], RawArray["UnsignedInteger32", {255}]];,
			_, lfFileStreamSetID3v1Element[streamID, $ID3v1Keys[uTagID], RawArray["UnsignedInteger8",{0}]];
		];
	,
	"ID3v2",
	    lfFileStreamRemoveID3v2Frame[streamID, AppendTo[tagPair[[2]], index]];
	,
	"APE",
		If[rawFrames[[n, 2, "Type"]] != $APETypesAssociation["Text"] || Length[rawFrames] == 1,
			lfFileStreamRemoveAPEItem[streamID, stringToRawArray[uTagID]];
			,
			lfFileStreamSetAPEItemValues[streamID, stringToRawArray[uTagID], stringToRawArray[(#[[2,"Frame"]])& /@ Delete[rawFrames,n]]];
		];
	,
	"M4A",
		lfFileStreamRemoveM4AItemKey[streamID, stringToRawArray[uTagID]];
	,
	"Xiph",
		rawKey = stringToRawArray[uTagID];
		values = lfFileStreamGetXiphValues[streamID, rawKey];
		If[SameQ[Head[values],LibraryFunctionError], tagMessage["invalid", "FrameNotFound", tagID]; Return[$Failed];];
		values = SplitBy[Normal@values, SameQ[#, 0]&][[1 ;; ;; 2]][[n]];
		lfFileStreamRemoveXiphKeyWithValue[streamID, rawKey, RawArray["UnsignedInteger8", values]];
	];
	lfFileStreamCloseTags[streamID, 1];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];
	If[TrueQ[OptionValue@"RefreshTags"], $metaTagAssociation[{tagType, tagNo}] = {tagPair[[1]], tagPair[[2]], True};];
	reloadAllTags[{streamID, tagType}];
	Return[addNewTag[obj, tagType]];
]

(* Adding frames to tags *)

(* TODO: I don't think AudioFileStreamTools should be modifying the casing of Item keys for tags, even if they are case-insensitive. 
Setting elements should just be smarter, and check if there exists a Key that matches ignoring case, and use it if so. *)

Options[addFrameForKeyForTag] = {"Level" -> {}};
addFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:("ID3v2"|"Xiph"|"M4A"), tagID_, opts:OptionsPattern[]] := 
Module[{tagPair, tagNo, baseTagPair, baseTagNo, streamID},
	If[(tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"]) === $Failed, Return[$Failed];];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	streamID = tagPair[[1]];
	If[getField[streamID, "InternetStream"] == True, tagMessage["unsupported", "InternetStream"]; Return[$Failed]];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];

	$metaTagAssociation[{tagType, tagNo}] = {streamID, tagPair[[2]], True};
	If[SameQ[tagType, "ID3v2"],
		baseTagNo = getTagNo[getStreamID@obj, tagType, False];
		baseTagPair = $metaTagAssociation[{tagType, baseTagNo}];
		$metaTagAssociation[{tagType, baseTagNo}] = {streamID, baseTagPair[[2]], True};
	];
	lfFileStreamOpenTags[streamID];
	Switch[tagType,
		"ID3v2",
			lfFileStreamAddID3v2Frame[streamID, tagPair[[2]], stringToRawArray[ToUpperCase[translateTagKey[tagType,tagID]]]];
		,
		"Xiph",
			lfFileStreamAddXiphKey[streamID, stringToRawArray[ToUpperCase[translateTagKey[tagType,tagID]]]];
		,
		"M4A",
			lfFileStreamSetM4AItem[streamID, stringToRawArray[(*translateTagKey[tagType,#]&@*)tagID], Sequence@@m4aItemToRawArray[{}]];(*TODO: why am I not translating?*)
	];
	lfFileStreamCloseTags[streamID, 1];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];
	reloadAllTags[{streamID, tagType}];
	addNewTag[obj, tagType];
]

addFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"APE", tagID_, frameType_, opts:OptionsPattern[]] := 
Module[{tagPair, typeEnum, tagNo, streamID},
	If[(tagNo = getTagNo[getStreamID@obj, tagType, False]) === $Failed, Return[$Failed];];
	typeEnum = $APETypesAssociation[frameType];
	If[MissingQ[typeEnum], tagMessage["invalid", "FrameNotFound", tagID]; Return[$Failed];];
	tagPair = $metaTagAssociation[{"APE", tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	streamID = tagPair[[1]];
	If[getField[streamID, "InternetStream"] == True, tagMessage["unsupported", "InternetStream"]; Return[$Failed]];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];

	$metaTagAssociation[{tagType, tagNo}] = {streamID, tagPair[[2]], True};
	lfFileStreamOpenTags[streamID];
	lfFileStreamAddAPEItem[streamID, stringToRawArray[ToUpperCase[translateTagKey[tagType,tagID]]], typeEnum];
	lfFileStreamCloseTags[streamID, 1];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];
	reloadAllTags[{streamID, "APE"}];
	addNewTag[obj, "APE"];
]

addFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, "ID3v1", ___] := (tagMessage["unsupported", "Operation", "ID3v1"]; $Failed)
addFrameForKeyForTag[___] := $Failed

(* Setting Tags on Disk *)

Options[setElementForFrameForKeyForTag] = {"Level" -> {}, "MultipleElements" -> False};

setElementForFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, inValue_, opts:OptionsPattern[]] := 
Module[{value, tagPair, tagNo},
	If[(tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"]) === $Failed, Return[$Failed];];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];

	value = uninterpretElementsForKey[tagType, tagID, inValue, "Context" -> {tagType, tagID, tagPair, tagNo}];
	If[value === $Failed,
		tagMessage["invalid", "Element", tagID, tagType];
		Return[$Failed];
	];
	Switch[First@Characters[ToUpperCase[translateTagKey[tagType, tagID]]],
		"T", setElementForFrameForKeyForTag[obj, tagType, tagID, 1, "Values", {value}, opts],
		"W", setElementForFrameForKeyForTag[obj, tagType, tagID, 1, "URL", value, opts],
		_, $Failed
	]
]

setElementForFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, property_String, value_, opts:OptionsPattern[]] := setElementForFrameForKeyForTag[obj, tagType, tagID, 1, property, value, opts]

setElementForFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:"ID3v2", tagID_String, n_Integer, property_String, inValue_, opts:OptionsPattern[]] := 
Module[{tagNo, count = 0, tagPair, index = None, frameType, data, ret, elements, uTagID, baseTagPair, baseTagNo, streamID, value},
	If[(tagNo = getTagNo[getStreamID@obj, tagType, False, OptionValue@"Level"]) === $Failed, Return[$Failed];];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	streamID = tagPair[[1]];
	If[getField[streamID, "InternetStream"] == True, tagMessage["unsupported", "InternetStream"]; Return[$Failed]];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];

	value = uninterpretElementsForKey[tagType, property, inValue, "Context" -> {tagType, tagID, tagPair, tagNo}];
	If[value === $Failed,
		tagMessage["invalid", "Element", tagID, tagType];
		Return[$Failed];
	];

	index = getRawFramesForKeyForTag[tagType, tagID, tagPair];
	If[Length[index] < n, tagMessage["invalid", "FrameNotFound", tagID]; Return[$Failed]];
	frameType = index[[n, 2, "Type"]];
	index = index[[n, 1]];

	$metaTagAssociation[{tagType, tagNo}] = {streamID, tagPair[[2]], True};
	baseTagNo = getTagNo[getStreamID@obj, tagType, False];
	baseTagPair = $metaTagAssociation[{tagType, baseTagNo}];
	$metaTagAssociation[{tagType, baseTagNo}] = {streamID, baseTagPair[[2]], True};
	lfFileStreamOpenTags[streamID];

	index = Append[tagPair[[2]], index];
	uTagID = ToUpperCase[translateTagKey[tagType, tagID]];
	If[!MissingQ[$id3v2FramesAssociation[uTagID]],
		elements = $id3v2FramesAssociation[uTagID]["Elements"];
		,
		Switch[Characters[uTagID][[1]],
			"T",
				elements = $id3v2FramesAssociation["T***"]["Elements"],
			"W",
				elements = $id3v2FramesAssociation["W***"]["Elements"],
			_,
				lfFileStreamCloseTags[streamID, 0];
				reloadAllTags[{streamID, tagType}];
				addNewTag[obj,"ID3v2"];
				tagMessage["invalid", "FrameNotFound", tagID];
				Return[$Failed]
		]
	];

	If[!MemberQ[elements, property],
		lfFileStreamCloseTags[streamID, 0];
		reloadAllTags[{streamID, tagType}];
		addNewTag[obj,"ID3v2"];
		tagMessage["invalid", "FrameNotFound", tagID];
		Return[$Failed]
	];

	Switch[property,
		"Channels",
		lfFileStreamClearID3v2FrameChannels[streamID, index, frameType];
		Scan[(
			data = <||>;
			data["BitsRepresentingPeak"] = Replace[#[[2]]["BitsRepresentingPeak"], _?MissingQ -> 0];
			data["VolumeAdjustment"] = Replace[#[[2]]["VolumeAdjustment"], _?MissingQ -> 0];
			data["PeakVolume"] = RawArray["UnsignedInteger8", Normal@Replace[#[[2]]["PeakVolume"], _?MissingQ -> {0}]];
			lfFileStreamSetID3v2FrameChannel[streamID, index, frameType, $channelTypes[#[[1]]], data["BitsRepresentingPeak"], data["VolumeAdjustment"], data["PeakVolume"]];
		)&, Normal[value]];
		,
		"ChildElements",
		lfFileStreamClearID3v2TableOFContentsFrameChildElements[streamID, index, frameType];
		Scan[(lfFileStreamSetID3v2TableOFContentsFrameChildElements[streamID, index, frameType, stringToRawArray[If[StringQ[#], #, Normal[#]]]])&, value];
		,
		"SynchedEvents",
		data = {};
		Scan[(AppendTo[data, #[[1]]]; AppendTo[data, If[StringQ[#], $eventTimingCodes[#], #]& @ #[[2]]])&, Normal[value]];
		data = RawArray["UnsignedInteger32", data];
		ret = lfFileStreamSetID3v2FrameSynchedEvents[streamID, index, frameType, data];
		,
		"SynchedText",
		lfFileStreamClearID3v2FrameSynchedText[streamID, index, frameType];
		ret = lfFileStreamSetID3v2FrameSynchedText[streamID, index, frameType, RawArray["UnsignedInteger32", Keys[value]], stringToRawArray[Values[value]]];
		(* Scan[(lfFileStreamSetID3v2FrameSynchedText[streamID, index, frameType, Print["setting time: ",#[[1]]]; #[[1]], stringToRawArray[#[[2]]]])&, Normal[value]]; *)
		,
		"Values",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameValues[streamID, index, frameType, data];
		,
		"Description",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameDescription[streamID, index, frameType, data];
		,
		"Language",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameLanguage[streamID, index, frameType, data];
		,
		"FileName",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameFileName[streamID, index, frameType, data];
		,
		"MimeType",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameMimeType[streamID, index, frameType, data];
		,
		"Picture",
		data = stringToRawArray[Normal[value]];
		ret = lfFileStreamSetID3v2FramePicture[streamID, index, frameType, data];
		,
		"Seller",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameSeller[streamID, index, frameType, data];
		,
		"PurchaseDate",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FramePurchaseDate[streamID, index, frameType, data];
		,
		"PricePaid",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FramePricePaid[streamID, index, frameType, data];
		,
		"Email",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameEmail[streamID, index, frameType, data];
		,
		"Object",
		data = stringToRawArray[Normal[value]];
		ret = lfFileStreamSetID3v2FrameObject[streamID, index, frameType, data];
		,
		"Owner",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameOwner[streamID, index, frameType, data];
		,
		"Data",
		data = stringToRawArray[Normal[value]];
		ret = lfFileStreamSetID3v2FrameData[streamID, index, frameType, data];
		,
		"Identifier",
		data = stringToRawArray[If[StringQ[value], value, Normal[value]]];
		ret = lfFileStreamSetID3v2FrameIdentifier[streamID, index, frameType, data];
		,
		"URL",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameURL[streamID, index, frameType, data];
		,
		"Text",
		data = stringToRawArray[value];
		ret = lfFileStreamSetID3v2FrameText[streamID, index, frameType, data];
		,
		"EndOffset",
		ret = lfFileStreamSetID3v2FrameEndOffset[streamID, index, frameType, value];
		,
		"StartOffset",
		ret = lfFileStreamSetID3v2FrameStartOffset[streamID, index, frameType, value];
		,
		"EndTime",
		ret = lfFileStreamSetID3v2FrameEndTime[streamID, index, frameType, value];
		,
		"StartTime",
		ret = lfFileStreamSetID3v2FrameStartTime[streamID, index, frameType, value];
		,
		"PictureType",
		data = If[StringQ[value], $pictureTypes[value], value];
		ret = lfFileStreamSetID3v2FramePictureType[streamID, index, frameType, data];
		,
		"LyricsType",
		data = If[StringQ[value], $lyricsTypes[value], value];
		ret = lfFileStreamSetID3v2FrameLyricsType[streamID, index, frameType, data];
		,
		"TimestampFormat",
		data =  If[StringQ[value], $eventTimestampFormats[value], value];
		ret = lfFileStreamSetID3v2FrameTimeStampFormat[streamID, index, frameType, data];
		,
		"Ordered",
		data = If[TrueQ[value], 1, 0];
		ret = lfFileStreamSetID3v2FrameOrdered[streamID, index, frameType, data];
		,
		"TopLevel",
		data = If[TrueQ[value], 1, 0];
		ret = lfFileStreamSetID3v2FrameTopLevel[streamID, index, frameType, data];
	];

	lfFileStreamCloseTags[streamID, 1];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];
	reloadAllTags[{streamID, tagType}];
	addNewTag[obj, "ID3v2"];
	Return[Null];
]

(* support for flattened tag structures *)
setElementForFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:("Xiph"|"ID3v1"|"APE"|"M4A"), tagID_String, data_, opts:OptionsPattern[]] := setElementForFrameForKeyForTag[obj, tagType, tagID, 1, data, opts]

setElementForFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:("Xiph"|"ID3v1"|"APE"|"M4A"), tagID_String, n_Integer, inData_, opts:OptionsPattern[]] := 
Module[{tagNo, count = 0, tagPair, index = None, frameType, ret, value, uTagID, streamID, rawKey, data, elem},
	If[(tagNo = getTagNo[getStreamID@obj, tagType, False]) === $Failed, Return[$Failed];];
	tagPair = $metaTagAssociation[{tagType, tagNo}];
	If[MissingQ[tagPair], tagMessage["invalid", "TagNotFound", tagType]; Return[$Failed];];
	streamID = tagPair[[1]];
	If[getField[streamID, "InternetStream"] == True, tagMessage["unsupported", "InternetStream"]; Return[$Failed]];
	If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; Return[$Failed];];

	data = If[TrueQ[OptionValue@"MultipleElements"],
		uninterpretElementsForKey[tagType, tagID, #, "Context" -> {tagType, tagID, tagPair, tagNo}]& /@ inData
		, uninterpretElementsForKey[tagType, tagID, inData, "Context" -> {tagType, tagID, tagPair, tagNo}]
	];
	If[data === $Failed || MemberQ[data, $Failed],
		tagMessage["invalid", "Element", tagID, tagType];
		Return[$Failed];
	];

	If[!SameQ[tagType, "M4A"],
		If[n != 1, tagMessage["unsupported", "Operation", tagType]; Return[$Failed];];
	];

	uTagID = If[SameQ[tagType,"M4A"], #, ToUpperCase[#]]& @ translateTagKey[tagType, tagID];
	If[SameQ[tagType, "APE"],
		index = getRawFramesForKeyForTag[tagType, uTagID, tagPair];
		If[Length[index] < n, tagMessage["invalid", "FrameNotFound", tagID]; Return[$Failed]];
		frameType = index[[n, 2, "Type"]];
	];

	$metaTagAssociation[{tagType, tagNo}] = {streamID, tagPair[[2]], True};
	lfFileStreamOpenTags[streamID];

	Switch[tagType,
		"Xiph",
			ret = If[!zOrZsQ[data, StringQ], $Failed, lfFileStreamSetXiphValues[streamID, stringToRawArray[uTagID], stringToRawArray[data]]];
		,
		"ID3v1",
			ret = If[uTagID === "YEAR" || uTagID === "TRACK" || uTagID === "GENRE",
				If[!IntegerQ[data], $Failed, lfFileStreamSetID3v1Element[streamID, $ID3v1Keys[uTagID], RawArray["UnsignedInteger32", {data}]]]
				,
				If[!StringQ[data], $Failed, lfFileStreamSetID3v1Element[streamID, $ID3v1Keys[uTagID], stringToRawArray[data]]]
			];
			If[ret === $Failed, tagMessage["invalid", "Element", tagID, tagType];];
		,
		"APE", 
			ret = If[frameType === $APETypesAssociation["Text"],
				If[!zOrZsQ[data, StringQ], $Failed, lfFileStreamSetAPEItemValues[streamID, stringToRawArray[uTagID], stringToRawArray[If[StringQ[data], {data}, data]]]]
				,
				If[!ByteArrayQ[data], $Failed, lfFileStreamSetAPEItemData[streamID, stringToRawArray[uTagID], RawArray["UnsignedInteger8", Normal@data]]]
	        ];
			If[ret === $Failed, tagMessage["invalid", "Element", tagID, tagType];];
		,
		"M4A",
			rawKey = stringToRawArray[uTagID];
			elem = getElementsAssociation["M4A", uTagID];
			value = If[MatchQ[data, {_Integer, _Integer}] || !ListQ[data], {data}, data];
			value = If[MissingQ[elem] || MissingQ[elem["ItemType"]], m4aItemToRawArray /@ value, m4aItemToRawArray[#, "ItemType"->elem["ItemType"]]& /@ value];
			If[MemberQ[value, $Failed], 
				ret = $Failed
				,
				If[MatchQ[data, {_Integer, _Integer}|_Integer|True|False],
					lfFileStreamSetM4AItem[streamID, rawKey, Sequence@@First[value]];
					,
					If[n == 1,
						MapThread[lfFileStreamSetM4AItemInList[streamID, rawKey, Sequence@@(#1), #2]&, {value, Range[Length[value]]-1}];
						,
						lfFileStreamSetM4AItemInList[streamID, rawKey, Sequence@@First[value], n-1];
					];
				];
			];
	];

	If[ret === $Failed,
		lfFileStreamCloseTags[streamID, 0];
		,
		lfFileStreamCloseTags[streamID, 1];
		If[lfStreamHasOperatingSystemWritePermissions[streamID] =!= 1, tagMessage["noset", "WritePerm"]; ret = $Failed;];
	];
	reloadAllTags[{streamID, tagType}];
	addNewTag[obj, tagType];
	ret
]

setElementForFrameForKeyForTag[obj_AudioFileStreamTools`FileStreamObject, tagType:("Xiph"|"ID3v1"|"APE"|"ID3v2"|"M4A"), ___] := (tagMessage["unsupported", "Operation", tagType]; $Failed)
setElementForFrameForKeyForTag[___] := (tagMessage["noset", "noset"]; $Failed)

(* Helper Functions *)

setTagGetTagNo[obj_AudioFileStreamTools`FileStreamObject, tagType_, level_] := Module[{streamID, tagNo, removeTagOnFailure = False},
	streamID = getStreamID@obj;
	If[(tagNo = getTagNo[streamID, tagType, False, level]) === $Failed,
		If[(tagNo = getTagNo[streamID, tagType, True, level]) === $Failed, tagMessage["noset", "NewTag", tagType];];
		removeTagOnFailure = True;
	];
	{tagNo, removeTagOnFailure}
]

getApeFrameType[tagID_, frame_] := (
	If[!MissingQ[$apeElementsAssociation[translateTagKey[tagType, tagID]]],
		"Text"
		,
		Switch[frame,
			_?stringOrLinkQ, "Text",
			_?stringOrQuantityQ, "Text",
			_?stringOrNumberQ, "Text",
			_?dateSpecQ, "Text",
			_?ListQ, "Text",
			_?ByteArrayQ, "Binary",
			_, $Failed
		]
	]
)

(* Attempt to infer the tagType from file format and tag keys *)
conformMetaTags[tags_Association, fileFormat_String, obj_:None] := 
Module[{tagKeys, existingTags = {}, hasId3v1KeysOnly},
	tagKeys = Keys@tags;
	If[!SubsetQ[Keys@$tagTypes, tagKeys],
		If[obj =!= None, 
			Quiet[addNewTag[obj, #, "MetaInformationInterpretation" -> None]& /@ Keys[$tagTypes]];
			existingTags = Quiet[Replace[getKeysForTag[obj, #], {{__} -> #, _ -> Nothing}]& /@ Keys[$tagTypes]]
		];
		Switch[fileFormat,
			"MP3",
				hasId3v1KeysOnly = SubsetQ[Keys@$ID3v1Keys, ToUpperCase[translateTagKey["ID3v1", #]]& /@ tagKeys];
				Which[
					hasId3v1KeysOnly && Complement[existingTags, {"ID3v1"}] === {},
						<|"ID3v1" -> tags|>
					,
					SubsetQ[Union[Keys@$id3v2FramesAssociation, Keys@$id3v2ExtendedFramesAssociation], 
							translateTagKey["ID3v2", #]& /@ tagKeys],
						<|"ID3v2" -> tags|>
					,
					True,
						<|"APE" -> tags|>
				]
			,
			"FLAC",
				hasId3v1KeysOnly = SubsetQ[Keys@$ID3v1Keys, ToUpperCase[translateTagKey["ID3v1", #]]& /@ tagKeys];
				If[hasId3v1KeysOnly && Complement[existingTags, {"ID3v1"}] === {},
					<|"ID3v1" -> tags|>
					,
					<|"Xiph" -> tags|>
				]
			,
			"M4A",
				<|"M4A" -> tags|>
			,
			"WAV"|"AIFF",
				<|"ID3v2" -> tags|>
			,
			"OGG",
				<|"Xiph" -> tags|>
			,
			_, 
				tags
		]
		,
		tags
	]
]

stringToRawArray[strKey_?StringQ] := (RawArray[If[Max[#] > 255, "UnsignedInteger16", "UnsignedInteger8"], #]& @ ToCharacterCode[strKey])
stringToRawArray[strKey_?ByteArrayQ] := RawArray["UnsignedInteger8", Normal[strKey]]
stringToRawArray[strKey_] /; VectorQ[strKey, Internal`NonNegativeIntegerQ] := (RawArray[If[Max[#] > 255, "UnsignedInteger16", "UnsignedInteger8"], #]& @ strKey)
stringToRawArray[strKey_] /; VectorQ[strKey, StringQ] := (RawArray[If[Max[#] > 255, "UnsignedInteger16", "UnsignedInteger8"], #]& @ Flatten[Riffle[ToCharacterCode[strKey], 0]])
stringToRawArray[strKey_?Developer`RawArrayQ] := strKey

Options[m4aItemToRawArray] = {"ItemType" -> Automatic, "AtomType" -> Automatic};
m4aItemToRawArray[data_, opts:OptionsPattern[]] := Quiet[Module[{itemType, atomType},
	Check[
		itemType = $m4aItemTypesAssoc[OptionValue@"ItemType"] /. _Missing -> Automatic;
		atomType = $m4aAtomTypesAssoc[OptionValue@"AtomType"] /. _Missing -> Automatic;
		Which[
			SameQ[OptionValue@"ItemType", "SignedInteger"] || SameQ[OptionValue@"AtomType", "Integer"],
				If[IntegerQ[data] && (data >= -2^31) && (data < 2^31), {RawArray["Integer32", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["SignedInteger"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Integer"])}, $Failed]
			,
			SameQ[OptionValue@"ItemType", "UnsignedInteger"] || SameQ[OptionValue@"AtomType", "QTUnsignedInteger32"],
				If[IntegerQ[data] && (data >= 0) && (data < 2^32), {RawArray["UnsignedInteger32", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["UnsignedInteger"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["QTUnsignedInteger32"])}, $Failed]
			,
			SameQ[OptionValue@"ItemType", "LongInteger"] || SameQ[OptionValue@"AtomType", "QTSignedInteger64"],
				If[IntegerQ[data] && (data >= -2^63) && (data < 2^64), {RawArray["Integer64", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["LongInteger"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["QTSignedInteger64"])}, $Failed]
			,
			SameQ[OptionValue@"ItemType", "Byte"] || SameQ[OptionValue@"AtomType", "Undefined"], 
				If[IntegerQ[data] && (data >= 0) && (data < 256), {RawArray["UnsignedInteger8", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["Byte"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}, $Failed]
			,
			SameQ[OptionValue@"ItemType", "StringList"] || SameQ[OptionValue@"AtomType", "UTF8String"] || SameQ[OptionValue@"AtomType", "UTF16String"] || SameQ[OptionValue@"AtomType", "URL"],
				If[StringQ[data] || (MatchQ[data, _URL|_File] && StringQ[data = First[data]]), If[Max[#] > 255, 
						{RawArray["UnsignedInteger16", #], (itemType /. Automatic -> $m4aItemTypesAssoc["StringList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["UTF16String"])}, 
						{RawArray["UnsignedInteger8", #], (itemType /. Automatic -> $m4aItemTypesAssoc["StringList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["UTF8String"])}
					]& @ ToCharacterCode[data], $Failed]
			,
			SameQ[OptionValue@"ItemType", "CoverArtList"] || SameQ[OptionValue@"AtomType", "JPEG"] || SameQ[OptionValue@"AtomType", "GIF"] || SameQ[OptionValue@"AtomType", "PNG"] || SameQ[OptionValue@"AtomType", "BMP"],
				Switch[data,
					_?ImageQ,
						{RawArray["UnsignedInteger8", Flatten@ToCharacterCode[ExportString[data, (OptionValue@"AtomType" /. Automatic -> "JPEG")]]], (itemType /. Automatic -> $m4aItemTypesAssoc["CoverArtList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["JPEG"])}
					,
					_?ByteArrayQ,
						{RawArray["UnsignedInteger8", Normal@data], (itemType /. Automatic -> $m4aItemTypesAssoc["CoverArtList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["JPEG"])}
					,
					_, $Failed
				]
			,
			SameQ[OptionValue@"ItemType", "Boolean"],
				If[MatchQ[data, True|False|0|1], {RawArray["UnsignedInteger8", {data /. {True->1,False->0}}], (itemType /. Automatic -> $m4aItemTypesAssoc["Boolean"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}, $Failed]
			,
			SameQ[OptionValue@"ItemType", "ByteVectorList"],
				If[ByteArrayQ[data], {RawArray["UnsignedInteger8", Normal@data], (itemType /. Automatic -> $m4aItemTypesAssoc["ByteVectorList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}, $Failed]
			,
			True,
				Switch[data,
					{},
						{RawArray["UnsignedInteger8", {0}], (itemType /. Automatic -> $m4aItemTypesAssoc["ByteVectorList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}
					,
					{_Integer, _Integer},
						{RawArray["Integer32", data], (itemType /. Automatic -> $m4aItemTypesAssoc["IntegerPair"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}
					,
					_String,
						If[Max[#] > 255, {RawArray["UnsignedInteger16", #], (itemType /. Automatic -> $m4aItemTypesAssoc["StringList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["UTF16String"])}, 
							{RawArray["UnsignedInteger8", #], (itemType /. Automatic -> $m4aItemTypesAssoc["StringList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["UTF8String"])}
						]& @ ToCharacterCode[data]
					,
					_Image,
						{RawArray["UnsignedInteger8", Flatten@ToCharacterCode[ExportString[data, (OptionValue@"AtomType" /. Automatic -> "JPEG")]]], (itemType /. Automatic -> $m4aItemTypesAssoc["CoverArtList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["JPEG"])}
					,
					_ByteArray,
						{RawArray["UnsignedInteger8", Normal@data], (itemType /. Automatic -> $m4aItemTypesAssoc["ByteVectorList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}
					,
					(True|False),
						{RawArray["UnsignedInteger8", {If[data,1,0]}], (itemType /. Automatic -> $m4aItemTypesAssoc["Boolean"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}
					,
					_URL|_File,
						{RawArray["UnsignedInteger16", ToCharacterCode[First[data]]], (itemType /. Automatic -> $m4aItemTypesAssoc["StringList"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["URL"])}
					,
					_Integer,
						If[(data >= 0) && (data < 256),
							{RawArray["UnsignedInteger8", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["Byte"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Undefined"])}
							,
							If[(data >= -2^31) && (data < 2^31), 
								{RawArray["Integer32", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["SignedInteger"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["Integer"])}
								,
								If[(data >= 0) && (data < 2^32),
									{RawArray["UnsignedInteger32", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["UnsignedInteger"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["QTUnsignedInteger32"])}
									,
									If[(data >= -2^63) && (data < 2^64),
										{RawArray["Integer64", {data}], (itemType /. Automatic -> $m4aItemTypesAssoc["LongInteger"]), (atomType /. Automatic -> $m4aAtomTypesAssoc["QTSignedInteger64"])}
										,
										$Failed
									]
								]
							]
						]
					,
					_,
						$Failed
				]
		]
	, $Failed]
]]

