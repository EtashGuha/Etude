BeginPackage["XMPTools`"]
Begin["`Private`"]

$InitXMPTools = False;

$XMPToolsLibrary = "XMPTools";
$XMPToolsBaseDirectory = FileNameDrop[$InputFileName, -2];
$BaseLibraryDirectory = FileNameJoin[{$XMPToolsBaseDirectory, "LibraryResources", $SystemID}];
$AdditionalFiles = Append[
	FileNameJoin[{$XMPToolsBaseDirectory, "Kernel", #}] & /@ {"Tags.m", "Exif.m", "XMP.m", "IPTC.m"},
	FileNameJoin[{$XMPToolsBaseDirectory, "LibraryResources", "LibraryLinkUtilities.wl"}]
];

Once[Get /@ $AdditionalFiles];

InitXMPTools[debug_ : False] :=
	If[TrueQ[$InitXMPTools],
		$InitXMPTools
		,
		$InitXMPTools =
			Catch[
				Block[
					{
						$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]
					},

					SafeLibraryLoad[debug, $XMPToolsLibrary];
					RegisterPacletErrors[$XMPToolsLibrary, <||>];
					(*Init*)
					$XMPInitialize = SafeLibraryFunction["XMPInitialize", {"UTF8String"}, True | False];
					$XMPUnInitialize = SafeLibraryFunction["XMPUnInitialize", {}, True | False];
					(*InitCopyMetaInformation*)
					$XMPInitializeMetaCopy = SafeLibraryFunction["XMPInitializeMetaCopy", {{"UTF8String"}, {"UTF8String"}, True | False}, True | False];
					$XMPUnInitializeMetaCopy = SafeLibraryFunction["XMPUnInitializeMetaCopy", {}, True | False];
					(*CopyMetaInformation*)
					$CopyMetaInformation = SafeLibraryFunction["CopyMetaInformation", {True | False, True | False, True | False, True | False}, True | False];
					(*Exif reading*)
					$ReadExifAllRaw = SafeLibraryFunction["ReadExifAllRaw", {True | False}, {"UTF8String"}];
					$ReadExifIndividualTag = SafeLibraryFunction["ReadExifIndividualTag", {"UTF8String"}, {"UTF8String"}];
					(*Exif writing*)
					$WriteExifInt = SafeLibraryFunction["WriteExifInt", {{"UTF8String"}, _Integer}, True | False];
					$WriteExifReal = SafeLibraryFunction["WriteExifReal", {{"UTF8String"}, _Real}, True | False];
					$WriteExifString = SafeLibraryFunction["WriteExifString", {{"UTF8String"}, {"UTF8String"}}, True | False];
					(*XMP reading*)
					$ReadXMPAll = SafeLibraryFunction["ReadXMPAll", {}, {"UTF8String"}];
					$ReadXMPAllRaw = SafeLibraryFunction["ReadXMPAllRaw", {}, {"UTF8String"}];
					(*XMP writing*)
					$WriteXMPNumber = SafeLibraryFunction["WriteXMPNumber", {{"UTF8String"}, _Integer}, True | False];
					$WriteXMPString = SafeLibraryFunction["WriteXMPString", {{"UTF8String"}, {"UTF8String"}}, True | False];
					$WriteXMPStructure = SafeLibraryFunction["WriteXMPStructure", {{"UTF8String"}, {"UTF8String"}, {"UTF8String"}}, True | False];
					(*IPTC reading*)
					$ReadIPTCAll = SafeLibraryFunction["ReadIPTCAll", {}, {"UTF8String"}];
					$ReadIPTCAllRaw = SafeLibraryFunction["ReadIPTCAllRaw", {}, {"UTF8String"}];
					$ReadIPTCIndividualTag = SafeLibraryFunction["ReadIPTCIndividualTag", {"UTF8String"}, {"UTF8String"}];
					(*IPTC writing*)
					$WriteIPTCInt = SafeLibraryFunction["WriteIPTCInt", {{"UTF8String"}, _Integer}, True | False];
					$WriteIPTCString = SafeLibraryFunction["WriteIPTCString", {{"UTF8String"}, {"UTF8String"}}, True | False];
				];
				True
			]
	]


(**************************)
(**************************)
(**************************)
(******INITIALIZATION******)
(**************************)
(**************************)
(**************************)

ExifAll = <||>;
XMPAll= <||>;
IPTCAll= <||>;

ExifRaw = <||>;
XMPRaw = <||>;
IPTCRaw = <||>;

MakerNote = <||>;

Init[on_, meta_String, fname___] := 
        Block[{$Context = "XMPTools`TempContext`"},
               If[on === False, $XMPUnInitialize[];
														Which[			 
	                                                        meta === "Exif",      ExifAll = $Failed
	                                                               ,
	                                                        meta === "ExifRaw",   ExifRaw = $Failed
	                                                               ,
	                                                        meta === "MakerNote", MakerNote = $Failed
	                                                        	   ,
															meta === "XMP",       XMPAll = $Failed
																   ,
														    meta === "XMPRaw",    XMPRaw = $Failed
																   ,
															meta === "IPTC",      IPTCAll = $Failed
															       ,
															meta === "IPTCRaw",   IPTCRaw = $Failed],
				If[!TrueQ@$XMPInitialize[Quiet@FindFile[fname]],
					Quiet[Which[			 
	                                                        meta === "Exif",      ExifAll = $Failed
	                                                               ,
	                                                        meta === "ExifRaw",   ExifRaw = $Failed
	                                                               ,
	                                                        meta === "MakerNote", MakerNote = $Failed
	                                                        	   ,
															meta === "XMP",       XMPAll = $Failed
																   ,
														    meta === "XMPRaw",    XMPRaw = $Failed
																   ,
															meta === "IPTC",      IPTCAll = $Failed
															       ,
															meta === "IPTCRaw",   IPTCRaw = $Failed]];
					 False
					,
					Quiet[
						Which[
						meta === "IPTC",      IPTCAll = ReadIPTC["All", False]
						   ,
				    	meta === "IPTCRaw",   IPTCRaw = ReadIPTC["AllRaw", False]
						   ,	
						meta === "Exif",      ExifAll = ReadExif["All", False]
					       ,
						meta === "ExifRaw",   ExifRaw = ReadExif["AllRaw", False]
					       ,
						meta === "MakerNote", MakerNote = ReadExif["MakerNote", False]
					       ,
						meta === "XMP",       XMPAll = ReadXMP["All", False]
					       ,
						meta === "XMPRaw",    XMPRaw = ReadXMP["AllRaw", False]
						]];
						True
				]				
			]
        ]


validatePossibleAssociation[tmp_]:=
	If[!MatchQ[tmp, LibraryFunctionError[_, _]] &&
		StringQ[tmp] &&
		(StringMatchQ[tmp, "<|"~~___~~"|>"] ||
		StringMatchQ[tmp, "{"~~___~~"}"]) &&
        StringLength[tmp] > 5
		,
		tmp
		,
		"<||>"
	]

validatePossibleString[tmp_]:=
	If[!MatchQ[tmp, LibraryFunctionError[_, _]] &&
		StringQ[tmp]
		,
		tmp
		,
		""
	]

(**************************)
(**************************)
(**************************)
(*********COPYMETA*********)
(**************************)
(**************************)
(**************************)

CopyMetaInformation[in_, out_, e_, x_, i_, c_, overwriteAllTags_:True] := 
                        Block[{$Context = "XMPTools`TempContext`"},
	                        Quiet[
	                        	Module[{inFPath, outFPath },
	                        		inFPath = FindFile[in];
	                            	outFPath = FindFile[out];
                                	If[!FileExistsQ[inFPath] || !FileExistsQ[outFPath], Return[$Failed]];
                                	If[$XMPInitializeMetaCopy[inFPath, outFPath, overwriteAllTags] =!= True, Return[$Failed]];
                                	If[$CopyMetaInformation[e, x, i, c] =!= True, Return[$Failed]];
                                	If[$XMPUnInitializeMetaCopy[] =!= True, Return[$Failed]];	 
                                	Return[True]
                                 ]
                             ]
                         ]
                         
GetXMP[tname_] := Block[{$Context = "XMPTools`TempContext`"},
	                  Module[{res = XMPAll, fin},
 					      fin = Quiet@Which[
 					      tname === "All",  Module[{x = ValidateXMP[res]}, If[AssociationQ[x], x, <||>]],
 					      tname === "Raw", XMPRaw,
 					      True, If[MatchQ[res[tname], $Failed[tname]] , $Failed, If[MatchQ[res[tname], Missing["KeyAbsent", tname]], LibraryFunctionError["LIBRARY_USER_ERROR",-2], res[tname]]]];
 					      Quiet[If[AssociationQ[fin], DeleteMissing[fin, Infinity], fin]]
 					   ]
                   ]
                   
GetIPTC[tname_] :=  Block[{$Context = "XMPTools`TempContext`"},
                        Module[{res = IPTCAll, fin},
 					        fin = Quiet@Which[
 					        tname === "All",  Module[{x = If[res === <||>, <||>, ValidateIPTC[res]]}, If[AssociationQ[x], x, <||>]],
 					        tname === "Raw", IPTCRaw,
 					        True, If[MatchQ[res[tname], $Failed[tname]] , $Failed, If[MatchQ[res[tname], Missing["KeyAbsent", tname]], LibraryFunctionError["LIBRARY_USER_ERROR",-2], res[tname]]]];
 					        Quiet[If[AssociationQ[fin], DeleteMissing[fin, Infinity], fin]]
 					    ]
                     ]
                     
GetExif[tname_] := Block[{$Context = "XMPTools`TempContext`"}, 
                       Module[{res = ExifAll, fin},
 					       fin = Quiet@Which[
 					           tname === "All", Module[{x = ValidateExif[res]}, If[AssociationQ[x], x, <||>]],
 					           tname === "Raw", ExifRaw,
 					           tname === "MakerNote", MakerNote,
 					           True, If[MatchQ[res[tname], $Failed[tname]] , $Failed, If[MatchQ[res[tname], Missing["KeyAbsent", tname]], LibraryFunctionError["LIBRARY_USER_ERROR",-2], res[tname]]]];
 					           Quiet[If[AssociationQ[fin], DeleteMissing[fin, Infinity], fin]]
 				        ]
                    ]
                    
End[]
EndPackage[]
