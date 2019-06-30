(* ::Package:: *)

(* :Title: LayoutDocsCollection.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.1 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


(* This file deals with the collection of documentation paclets describing the doc files that are in the
   Mathematica layout ($InstallationDirectory/Documentation). Because there are so many of them, and they are
   used in only a very specialized way, they are represented differently from all other paclets. They are
   not stored in memory as Mathematica expressions, but instead are stored in a file-based hashtable called
   PacletDB.m. The only operation that is performed on these paclets (or, at least, the only operation that needs
   to be fast) is lookup based on a linkbase-resourcename pair, like ["WolframMathematica", "ReferencePages/Symbols/Power"]
   coming from a URI like "paclet:ref/Power". They can also be looked up based on a context instead of a linkbase
   (for Packages symbols). Thus, we hash on the linkbase/resourcename and use that to quickly look up the PacletInfo
   data in the PacletDB.m file.
*)

Begin["`Package`"]


initializeLayoutDocsCollection

LDCfindForDocResource
LDCfindMatching

(* Carry this over from previous version of PM. webM sets this to False to avoid reading doc paclets. *)
$readDocPaclets


(* Used by WRI build process. *)
BuildDocDBFile


End[]  (* `Package` *)


(* Current context will be PacletManager`. *)

Begin["`LayoutDocsCollection`Private`"]


$layoutDocsCollection
$layoutDocsDirectory


(*****************************  Initialization  ******************************)

initializeLayoutDocsCollection[layoutDocsDir_String] :=
    (
        (* Defer doing any work until actual use. *)
        $layoutDocsDirectory = layoutDocsDir;  (* Typically $InstallationDirectory/Documentation. *)
        If[$pmMode === "ReadOnly",
            (* This ensures that it will never be populated, always remaining empty. *)
            $layoutDocsCollection = {},
        (* else *)
            $layoutDocsCollection = Null
        ]
    )


(*********************************  Getters  *********************************)

(* You must always call the initialize functions once in a session before calling these getters. *)


(* This incurs a small overhead (few hundredths of a second), so don't call it as part of PM startup
   process. Instead, defer until a doc paclet lookup is first needed.
*)
getLayoutDocsCollection[] :=
    (
        If[$layoutDocsCollection === Null,
            $layoutDocsCollection = {};
            If[$readDocPaclets =!= False,
                Module[{dbStrm, dbFile, serVersion, hashTable, lang, pacletStartPos},
                    (* Build $layoutDocsCollection by looping over all language subdirs of Documentation. *)
                    doForEach[dbFile, FileNames["PacletDB.m", ToFileName[$InstallationDirectory, "Documentation"], 2],
                        lang = FileNameTake[dbFile, {-2}];
                        using[{dbStrm = OpenRead[dbFile, DOSTextFormat -> False]},
                            serVersion = Read[dbStrm, Expression];
                            If[serVersion === 1,
                                hashTable = Developer`ToPackedArray[Read[dbStrm, Expression]];
                                pacletStartPos = StreamPosition[dbStrm];
                                AppendTo[$layoutDocsCollection, {lang, dbFile, pacletStartPos, hashTable}]
                            ]
                        ]
                    ]
                ]
            ]
        ];
        $layoutDocsCollection
    )


LDCfindForDocResource[linkBase:(_String | All), context:(_String | All),
                        expandedResourceName:(_String | All), language_String] :=
    Module[{docLayoutPCForRequestedLanguage, lBase, ctxt, hash, offsetPos, dbStrm, pacletExpr, resPath,
              collectionLanguage, collectionDBFile, collectionPacletStartPos, collectionHashTable, foundPacletAndPath},

        (* Single-word lookups are converted into a linkbase and empty resName. For user paclets, this supports
           resolving a paclet name to its main page. But this collection does not deal with such lookups, as they
           would be done via the search system.
        *)
        If[expandedResourceName == "",
            Return[{}]
        ];

        docLayoutPCForRequestedLanguage = Select[getLayoutDocsCollection[], (First[#] === language)&];
        If[Length[docLayoutPCForRequestedLanguage] == 0,
            (* We don't have docs for this language in the layout. *)
            Return[{}]
        ];

        (* Fall through result value. *)
        foundPacletAndPath = {};

        {collectionLanguage, collectionDBFile, collectionPacletStartPos, collectionHashTable} =
            First[docLayoutPCForRequestedLanguage];

        (* Now lookup via the hash, which is computed from something like WolframMathematica/ReferencePages/Symbols/Power.

           Note that we use the context as the linkbase if present. We get here with Context =!= All in only
           two situations: either it is a lookup of a symbol with a context (Audio`Foo) or it is a lookup
           of a message URI. In either case, we can assume that the context is the linkbase. This
           is strictly for this particular collection (docs in the layout indexed by the .db file).
           Every notebook is indexed by linkbase, and every context is the same as the linkbase.
           Therefore, we just take the linkbase to be the context.
        *)
        Which[
            context === "System`",
                lBase = "WolframMathematica";
                ctxt = All,
            StringQ[context],
                lBase = StringDrop[context, -1];
                ctxt = All,
            True,
                lBase = linkBase;
                ctxt = context
        ];
        hash = docHash[lBase <> "/" <> expandedResourceName];
        offsetPos = Quiet @ Cases[collectionHashTable, {hash, pos_} :> pos];
        If[Length[offsetPos] > 0,
            offsetPos = First[offsetPos];
            using[{dbStrm = OpenRead[collectionDBFile, DOSTextFormat->False]},
                SetStreamPosition[dbStrm, collectionPacletStartPos + offsetPos];
                pacletExpr = parsePacletInfo[Read[dbStrm, String]];
                (* Normal outcome: *)
                If[Head[pacletExpr] === List, pacletExpr = First[pacletExpr]]
            ]
        ];
        If[Head[pacletExpr] === Paclet,
            (* Location needs to be fixed up, since it is "fake" in the db file. *)
            pacletExpr = setLocation[pacletExpr, ToFileName[$layoutDocsDirectory, collectionLanguage]];
            resPath = PgetDocResourcePath[pacletExpr, lBase, ctxt, expandedResourceName, language];
            (*foundPacletAndPath = {{pacletExpr, ToFileName[{$layoutDocsDirectory, collectionLanguage, extraPath}, expandedResourceName <> ".nb"]}}*)
            foundPacletAndPath = {{pacletExpr, resPath}}
        ];
        foundPacletAndPath
    ]


(* This is very inefficient, as it reads the entire layout docs collection into memory, something that we 
   designed the collection to avoid doing. But, nothing in the internal operation of the PM ever triggers this,
   only a user doing PacletFind with IncludeDocPaclets->True. 
*)
LDCfindMatching[name_, version_, language_] :=
    Module[{paclets, coll, selectFunc, dbStrm,
              collectionLanguage, collectionDBFile, collectionPacletStartPos, collectionHashTable},
        paclets = 
	        Join @@
	        forEach[coll, Select[getLayoutDocsCollection[], (language === All || First[#] === language)&],
	            {collectionLanguage, collectionDBFile, collectionPacletStartPos, collectionHashTable} = coll;
	            using[{dbStrm = OpenRead[collectionDBFile, DOSTextFormat->False]},
	                SetStreamPosition[dbStrm, collectionPacletStartPos];
	                ReadList[dbStrm, Expression]
	            ]
	        ];
        selectFunc = Hold[];
        If[name =!= All,
            selectFunc = Join[selectFunc, Hold[StringMatchQ[getPIValue[#, "Name"], name]]]
        ];
        If[version =!= All && version =!= "",
            selectFunc = Join[selectFunc, Hold[StringMatchQ[getPIValue[#, "Version"], version]]]
        ];
        Select[paclets, Function[{heldSelectFunc}, Function[Null, And @@ heldSelectFunc] ] @ selectFunc]
    ]


(******************************  Building PacletDB.m file  ********************************)

(* This is used in looking up doc resources in the layoutdocscollection of paclets, which is a lookup
   into the PacletDB.m file.
   Get duplicate hashes without the Reverse; hash func is less sensitive to differences in the later
   chars of the string than earlier, so reverse the strings to put the small diffs between some res names
   at the beginning.
   Args to this function are long res names like "WolframMathematica/ReferencePages/Symbols/Table" and
   "XML/Tutorials/Overview".
   
   This s not a true hashmap-style lookup, in the sense that hash values are not being used to do an expedited lookup.
   The basic flow is that a resource name like "WolframMathematica/ReferencePages/Symbols/Table" is converted to
   an integer via a "hash" function, and this integer is then looked up in a table via linear search, to find 
   the Paclet expr that corresponds to that doc paclet. We could just as easily put the resource names into the
   table and not the integer hash values. But because the table is held in memory, we take advantage of the
   efficiency of a packed array of integers. We don't care that the lookup through the table is still linear
   search, because the speed is plenty fast enough.
   
   Quiet is used here onlyto silence potential unpacking messages on the packed array resulting from ToCharacterCode.
*)
docHash[str_String] := Quiet[Fold[docHashFunc, 5381, Reverse @ ToCharacterCode[ToLowerCase[str]]]]

(* DJBHash func from http://www.partow.net/programming/hashfunctions/#StringHashing. *)
docHashFunc[old_Integer, char_Integer] := Mod[BitShiftLeft[old, 5] + old + char, 4294967295]


$systemDocDirs = {"System/ReferencePages", "System/Guides", "System/Tutorials", "System/HowTos", "System/ExamplePages", "System/Workflows", "System/WorkflowGuides"};
$packagesDocDirs = {"Packages"};
(* The default version number of every doc paclet. Ideally, based on CVS revision numbers as in the past. *)
$pacletVersion = "10.0.0";

(* Not a run-time function; used only during build process, to create the PacletDB.m file.

   You can build directly out of an installed layout, with layoutTopLevelDir being simply $InstallationDirectory.
   More precisely, The dir you supply for layoutTopLevelDir merely needs a Documentation/lang/System and
   Documentation/lang/Packages hierarchy beneath it.
*)
BuildDocDBFile[destDir_String, layoutTopLevelDir_String, language_String] :=
    Module[{systemLinkBasesAndResNames, packagesLinkBasesAndResNames, linkBasesAndResNameLists, hashes,
             scratchFile, strm, startPositions, pos, pacletStartPos, destFile, allMsgs, resNameList},
        SetDirectory[FileNameJoin[{layoutTopLevelDir, "Documentation", language}]];
        systemLinkBasesAndResNames =
            Function[{fileName},
                If[MatchQ[FileNameSplit[fileName], {"System", "ReferencePages", "Messages", __}],
                    (* Messages notebooks need special handling, as they might document multiple messages (e.g., Block::lvset, Module::lvset).
                       Such notebooks will have a Resources section that contains multiple entries. That is, the Messages/Block/lvset paclet
                       will also include Messages/Module/lvset in its Resources list.
                    *)
                    allMsgs = getAllMessages[FileNameJoin[{Directory[], fileName}]];
                    If[allMsgs =!= $Failed,
                        resNameList = (StringJoin @@ Riffle[Join[{"ReferencePages", "Messages"}, #], "/"])& /@ allMsgs,
                    (* else *)
                        Print["WARNING: Messages notebook " <> fileName <>
                                 " has unexpected header; cannot determine if it contains multiple messages."];
                        resNameList = {StringJoin[Riffle[Rest[FileNameSplit[StringDrop[fileName, -3]]], "/"]]}
                    ];
                    {"WolframMathematica", resNameList},
                (* else *)
                    (* Not a Messages notebook. *)
                    {"WolframMathematica", 
                    (* 'Rest' strips off System/ prefix, StringDrop takes off ".nb" *)
                    {StringJoin[Riffle[Rest[FileNameSplit[StringDrop[fileName, -3]]], "/"]]}}
                ]
            ] /@ 
            Flatten[FileNames["*.nb", #, Infinity] & /@ $systemDocDirs];

        packagesLinkBasesAndResNames =
            Function[{fileName},
                {FileNameSplit[fileName][[2]],
                (* The [[5::]] drops Packages/Pkgname/Documentation/English *)
                {StringJoin[Riffle[FileNameSplit[StringDrop[fileName, -3]] [[5 ;;]], "/"]]}}
            ] /@ 
            Select[Flatten[FileNames["*.nb", #, Infinity] & /@ $packagesDocDirs],
                StringMatchQ[#, __~~$PathnameSeparator~~language~~$PathnameSeparator~~__]&
            ];     
        linkBasesAndResNameLists = Join[systemLinkBasesAndResNames, packagesLinkBasesAndResNames];
               
        (* Quick check for duplicate hashes. Duplicates would mean that when looking up doc X you would end up at doc Y. *)
        hashes = docHash /@ 
            Flatten[
                forEach[linkBaseAndResNameList, linkBasesAndResNameLists,
                    StringJoin[Riffle[#, "/"]]& /@ Thread[linkBaseAndResNameList]
                ]
            ];
        If[Length[hashes] =!= Length[Union[hashes]],
            (* Right now, this is firing with 1 duplicate because of bug 217106 (a trivial duplication of one line in a message notebook). *)
            Print["WARNING: Duplicate hashcodes: ", Length[hashes] - Length[Union[hashes]]]
        ];
        
        ResetDirectory[];
        scratchFile = Close[OpenWrite[]];
        strm = OpenWrite[scratchFile, DOSTextFormat -> False, CharacterEncoding -> "UTF8"];
        startPositions =
            forEach[linkBaseAndResNameList, linkBasesAndResNameLists,
                pos = StreamPosition[strm];
                WriteString[strm, makeDocPacletExpr[linkBaseAndResNameList, $pacletVersion, language]];
                WriteString[strm, "\n"];
                Table[pos, {Length[Last[linkBaseAndResNameList]]}]
            ] // Flatten;
        Close[strm];
        DeleteFile[scratchFile];
        strm = OpenWrite[scratchFile, DOSTextFormat -> False, CharacterEncoding -> "UTF8"];
        Write[strm, 1]; (* ser version *)
        Write[strm, Thread[{hashes, startPositions}]];
        pacletStartPos = StreamPosition[strm];
        Close[strm];
        DeleteFile[scratchFile];
        strm = OpenWrite[scratchFile, DOSTextFormat -> False, CharacterEncoding -> "UTF8"];
        Write[strm, 1]; (* ser version; change if, for example, the hash function changes. *)
        Write[strm, Thread[{hashes, startPositions}]];
        forEach[linkBaseAndResNameList, linkBasesAndResNameLists,
            WriteString[strm, makeDocPacletExpr[linkBaseAndResNameList, $pacletVersion, language]];
            WriteString[strm, "\n"];
        ];

        Close[strm];
        destFile = ToFileName[destDir, "PacletDB.m"];
        If[FileExistsQ[destFile],
            DeleteFile[destFile]
        ];
        CopyFile[scratchFile, destFile]
    ]

  
makeDocPacletExpr[{linkBase_String, resNames:{__String}}, version_String, language_String] :=
    Module[{name, resourceSpec, primaryMessage},
        primaryMessage = First[resNames];
        name = "SystemDocs_" <> language <> "_" <>
                 If[linkBase == "WolframMathematica", "", linkBase <> "_"] <> 
                   StringReplace[primaryMessage, "/" -> "_"];
        resourceSpec =
            If[linkBase == "WolframMathematica",
                If[StringMatchQ[First[resNames], "ReferencePages/Messages/*"],
                    (* Cannot use "#" matching in the custom path, because one notebook will be used for multiple
                       resources, most of which do not correspond to the actual file path.
                    *)
                    ToString[{#, "System/" <> primaryMessage <> ".nb"}& /@ resNames, InputForm],
                (* else *)
                    ToString[{{First[resNames], "System/#.nb"}}, InputForm]
                ],
            (* else *)
                ToString[{{First[resNames], "Packages/" <> linkBase <> "/Documentation/" <> language <> "/#.nb"}}, InputForm]
            ];
        "Paclet[\"Name\"->\"" <> name <> "\",\"Version\"->\"" <> version <> 
               "\",\"Extensions\"->{{\"Documentation\",\"LinkBase\"->\"" <> 
               linkBase <> "\",\"Language\"->\"" <> language <> "\",\"Root\"->\".\",\"Resources\"->" <> resourceSpec <>
         "}}]"
    ]
  
(* Messages notebooks are sometimes shared among multiple symbols that have a message with the same tag.
   For example, the Block/lvset.nb file is used to document the Block::lvset, Dialog::lvset and Module::lvset messages.
   This function determines the set of messages documented in a given notebook. It returns a list
   {{"Symbol", "tag"}, ...} with one entry for each message. The symbol names have had their context
   stripped off, so if the notebook displays a message as, say, Experimental`SomeFunc::foo, it will
   appear as just {"SomeFunc", "foo"} in the result. This code is sensitive to the format of the 
   Messages notebooks, and will need to be changed if the format changes.
*)
getAllMessages[file_String] :=
    Module[{msgCell, msgs},
        msgCell = Cases[Get[file], Cell[contents_String, "ObjectName"|"ObjectNameSmall", ___] :> contents, Infinity];
        If[Length[msgCell] != 1, 
            (* Should always be just one cell. *)
            Return[$Failed]
        ];
        msgCell = First[msgCell];
        msgs = StringSplit[msgCell, Whitespace];
        (* Convert "OptionalContext`SymbolName::tag" into {"SymbolName", "tag"}. *)
        Take[StringSplit[#, {"`", "::"}], -2]& /@ msgs
    ]

End[]

