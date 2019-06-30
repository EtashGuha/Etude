(* :Title: Paclet.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.1 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


Paclet::usage = "Paclet is an internal symbol."

(* TODO: These messages are weak. Mention that it returns a list of paclets, or $Failed. *)
CreatePaclet::usage = "CreatePaclet is a utility function to read the contents of a PacletInfo.m file into the appropriate internal Wolfram Language expression representation."

VerifyPaclet::usage = "VerifyPaclet is an internal symbol."


Begin["`Package`"]

createPacletsFromParentDirs
parsePacletInfo

getPIValue

PgetQualifiedName
PgetLocation
PgetKey
PisPacked
PhasLinkBase
PhasContext
PgetContexts
PgetExtensions

PgetDocResourcePath

PgetPathToRoot

PgetMainLink

PgetLoadingState
PgetPreloadData
PgetDeclareLoadData
PgetFunctionInformation


setLocation

$linkableExtensions



End[]  (* `Package` *)


(* Current context will be PacletManager`. *)

Begin["`Paclet`Private`"]


(*********************************  Reading PI files  ********************************)

(* ALL reading of PacletInfo.m files into Mathematica should be done by this function.
   The trick we use is to read the files in such a way that Paclet and the other symbols (Name, Version,
   etc.) are created in a private context. Then we turn them into strings, except for Paclet.
   Thus, this function returns an expression like
   {Paclet["Name"->"foo", "Version"->"1.0", ...]} where there are no symbols, only strings,
   except for System symbols like True, False, All.
   It returns either:
       - a Paclet expression that has passed the VerifyPaclet step
       - a list of Paclet expressions for a PacletSite or PacletGroup (not documented functionality)
       - $Failed
*)

CreatePaclet::badpi = "`1` is not a properly formatted PacletInfo.m or PacletInfo.wl file. You can use the VerifyPaclet function to get more detailed information about the error."
CreatePaclet::badppi = "The paclet `1` does not have a properly formatted PacletInfo.m or PacletInfo.wl file. You can use the VerifyPaclet function to get more detailed information about the error."
CreatePaclet::badarg = "`1` must be a .paclet file, a PacletInfo.m or PacletInfo.wl file, or a directory containing a PacletInfo.m or PacletInfo.wl file."
CreatePaclet::notfound = "Could not find specified file `1`."


CreatePaclet[path_String] :=
    Which[
        StringMatchQ[FileNameTake[path], "PacletInfo.m"] || StringMatchQ[FileNameTake[path], "PacletInfo.wl"],
            createPacletFromPIFile[path],
        StringMatchQ[path, "*.paclet"] || StringMatchQ[path, "*.cdf"],
            createPacletFromPackedFile[path],
        DirectoryQ[path],
            If[Length[#] > 0,
                CreatePaclet[First[#]],
            (* else *)
                Message[CreatePaclet::badarg, path];
                $Failed
            ]& @ FileNames[{"PacletInfo.m", "PacletInfo.wl"}, path],
        True,
            Message[CreatePaclet::badarg, path];
            $Failed
    ]

CreatePaclet[File[path_String]] := CreatePaclet[path]


createPacletFromPIFile[pathToPacletInfoFile_String, isPacletSiteData:(True | False):False] :=
    Module[{strm, piData, piExpr, msg},
        strm = Quiet[
                   (* Open the file, replacing OpenRead::noopen with CreatePaclet::noopen. *)
                   Check[OpenRead[pathToPacletInfoFile, BinaryFormat -> True],
                       msg = Hold[Message[CreatePaclet::noopen, pathToPacletInfoFile]],
                       OpenRead::noopen
                   ],
                   OpenRead::noopen
               ];
        If[Head[strm] === InputStream,
            piData = FromCharacterCode[BinaryReadList[strm], "UTF8"];
            Close[strm];
            piExpr = parsePacletInfo[piData];
            Switch[piExpr,
                {_Paclet},
                    (* Normal case; one Paclet expr in file. *)
                    If[VerifyPaclet[#, Verbose->False],
                        setLocation[#, ExpandFileName[FileNameDrop[pathToPacletInfoFile]]],
                    (* else *)
                        Message[CreatePaclet::badpi, pathToPacletInfoFile];
                        $Failed
                    ]& @ First[piExpr],
                {__Paclet},
                    (* Was a PacletGroup or PacletSite in file. This function is not documented to return a list, because I think
                       the whole PacletGroup idea is only for SystemDocs paclets and will not be described to users,
                       but in this case we return the list of paclets. Skip VerifyPaclet step as an optimization,
                       as the file ws probably created by a tool, not a user.
                    *)
                    (* For the sake of efficiency, don't bother with setLocation[] for site paclets. *)
                    If[isPacletSiteData,
                        piExpr,
                    (* else *)
                        setLocation[piExpr, ExpandFileName[FileNameDrop[pathToPacletInfoFile]]]
                    ],
                _,
                    Message[CreatePaclet::badpi, pathToPacletInfoFile];
                    $Failed
            ],
        (* else *)
            ReleaseHold[msg];  (* CreatePaclet::noopen Message issued here, outside of Quiet block. *)
            $Failed
        ]
    ]


createPacletFromPackedFile[pathToPacletFile_String] :=
    Module[{dataBuffer, piData, piExpr},
        If[FileType[pathToPacletFile] =!= File,
            Message[CreatePaclet::notfound, pathToPacletFile];
            Return[$Failed]
        ];
        Quiet[  (* Suppress messages from ZipGetFile; we will give our own message later *)
            dataBuffer = ZipGetFile[pathToPacletFile, "PacletInfo.m"];
            If[!ListQ[dataBuffer],
                dataBuffer = ZipGetFile[pathToPacletFile, "PacletInfo.wl"]
            ]
        ];
        If[ListQ[dataBuffer],
            piData = FromCharacterCode[dataBuffer, "UTF8"];
            piExpr = parsePacletInfo[piData];
            (* piExpr is either $Failed or {__Paclet}. *)
            Switch[piExpr,
                {_Paclet},
                    (* Normal case; one Paclet expr in file. *)
                    If[VerifyPaclet[#, Verbose->False],
                        setLocation[#, ExpandFileName[pathToPacletFile]],
                    (* else *)
                        $Failed
                    ]& @ First[piExpr],
                {__Paclet},
                    (* Was a PacletGroup or PacletSite in file. This function is not documented to return a list, because I think
                       the whole PacletGroup idea is only for SystemDocs paclets and will not be described to users,
                       but in this case we return the list of paclets. Skip VerifyPaclet step as an optimization,
                       as the file ws probably created by a tool, not a user.
                    *)
                    setLocation[piExpr, ExpandFileName[pathToPacletFile]],
                _,
                    Message[CreatePaclet::badppi, pathToPacletFile];
                    $Failed
            ],
        (* else *)
            Message[CreatePaclet::badppi, pathToPacletFile];
            $Failed
        ]
    ]


(* Worker function called when rebuilding paclet collections. Scans the given dirs to the given depth looking for
   PacletInfo.m files. Use depth 2 when giving a dir that will itself hold paclet dirs, as in "MyPacletSpace":
       MyPacletSpace/
            MyPaclet1/
               PacletInfo.m
               ...
            MyPaclet2/
               PacletInfo.m
               ...

   Returns a list of Paclet expresions.
*)
createPacletsFromParentDirs[parentDir:(_String | _File), depth_Integer] := createPacletsFromParentDirs[{parentDir}, depth]

createPacletsFromParentDirs[parentDirs:{(_String | _File)...}, depth_Integer] :=
    Cases[CreatePaclet /@ (Join @@ (FileNames[{"PacletInfo.m", "PacletInfo.wl"}, #, depth]& /@ parentDirs)), _Paclet]


(* All creation of paclets from PacletInfo.m data goes through this function. It returns
   either a list of one or more Paclet expressions, or $Failed. $Failed is returned if the
   paclet is not a syntactically correct Wolfram Language expression, or doesn't have head
   Paclet/PacletGroup/PacletSite. Verification of the paclet expression as legal in more subtle ways,
   an essential step, is left to the caller.
   It issues no messages.
*)

parsePacletInfo[piData_String] :=
    Block[{$Context = "PacletManager`Paclet`Private`", $ContextPath = {"System`"}, piExpr, strm, e},
        piExpr = $Failed;
        Quiet[
            using[{strm = StringToStream[piData]},
                While[!MatchQ[e = Read[strm, Expression],
                          EndOfFile | $Failed | _PacletManager`Paclet`Private`Paclet | _PacletManager`Paclet`Private`PacletGroup |
                                         _PacletManager`Paclet`Private`PacletSite],
                    Null (* Do nothing; keep reading. *)
                ]
            ];
            If[e =!= EndOfFile && e =!= $Failed,
                piExpr = e /. $pacletInfoSymbolConversions;
                If[Head[piExpr] === Paclet,
                    piExpr = {piExpr},
                (* else *)
                    (* Must have been PacletGroup or PacletSite. As a (foolish?) optimization, don't call VerifyPaclet
                       on Site or Group files, assuming them to be created by tools and not users.
                    *)
                    piExpr = List @@ piExpr
                ]
            ]
        ];
        piExpr
    ]


(* These are the symbols that might be encountered in a PacletInfo.m file. We create a set of
   rules that maps them to them to their string forms. It isn't essential that every symbol
   be here, but it is an optimization.
*)
$pacletInfoSymbols = {
    Name,
    Version,
    Extensions,
    Resources,
    SystemID,
    MathematicaVersion,  (* Deprecated; replaced by WolframVersion *)
    WolframVersion, 
    Qualifier,  (* Replacing old PlatformQualifier name, which is still supported. *)
    Internal,
    Root,
    BackwardCompatible,
    BuildNumber,
    Description,
    InstallFromDocRequest,
    ID,
    Creator,
    URL,
    Publisher,
    Support,
    Category,
    Thumbnail,
    Copyright,
    License,
    Loading,
    Language,
    Context,
    LinkBase,
    MainPage,
    Prepend,
    Symbols,
    FunctionInformation,
    HiddenImport,
    Alias,
    ProductName,
    Updating
}

(* We also allow these undocumented short forms. These are only used on the paclet server, and 
   I think this feature should be removed.
   TODO: Remove it. There are conflicts with some symbols used differently in different parts of the PI file.
*)
$pacletInfoShortRules = {
    N->"Name",
    V->"Version",
    B->"BuildNumber",
    PQ->"Qualifier",
    MV->"WolframVersion",
    S->"SystemID",
    BC->"BackwardCompatible",
    R->"Root",
    D->"Description",
    IDR->"InstallFromDocRequest",
    ID->"ID",
    CR->"Creator",
    U->"URL",
    P->"Publisher",
    SU->"Support",
    TH->"Thumbnail",
    CY->"Copyright",
    L->"License",
    IN->"Internal",
    LD->"Loading",
    E->"Extensions",
    
    "R"->"Resource",
    RS->"Resources",
    "K"->"Kernel",
    SY->"Symbols",
    C->"Context",
    "FE"->"FrontEnd",
    "D"->"Documentation",
    LB->"LinkBase",
    M->"MainPage",
    "DM"->"Demonstration"
}


$pacletInfoSymbolConversions =
    Dispatch[
        Join[
            Thread[$pacletInfoSymbols -> (ToString /@ $pacletInfoSymbols)],
            $pacletInfoShortRules,
            (* The first line below is a mapping of an old name to a newer replacement. Because _all_
               paclet-reading must be done by the functions in this file, this is the only place we need to make
               a change to support old names.
            *)
            {PacletManager`Paclet`Private`PlatformQualifier -> "Qualifier",
             (* I picked the wrong name originally (Context instead of Contexts) in Kernel extensions,
                and I think a common mistake will be to use Contexts instead. So help people out by converting.
             *)
             PacletManager`Paclet`Private`Contexts -> "Context",
             PacletManager`Paclet`Private`Paclet -> PacletManager`Paclet,
             PacletManager`Paclet`Private`PacletGroup -> PacletManager`Paclet`Private`PacletGroup,
             PacletManager`Paclet`Private`PacletSite -> PacletManager`Paclet`Private`PacletSite,
             List -> List,
             Rule -> Rule,
             True -> True,
             False -> False,
             (* Next two are for limited patterns on RHS: "ProductName" -> Except["Mathematica"] *)
             Except -> Except,
             Alternatives -> Alternatives, 
             All -> All,
             None -> None,
             Null -> Null,
             Automatic -> Automatic,
             (* For (incorrect) use of a real number for version instead of a string. *)
             v:(_Real | _Integer) :> ToString[v],
             (* Fallthrough for new symbols in a PI file that aren't yet listed in $pacletInfoSymbols. *)
             s_Symbol :> ToString[s]
            }
        ]
    ]


(***********************  Paclet Expression Representation  ************************)

General::pclt = "`1` does not refer to a known paclet."
General::pcltn = "No appropriate paclet with name `1` was found."
General::pcltnv = "No appropriate paclet with name `1` and version `2` was found."
General::pcltni = "No appropriate paclet with name `1` is installed."
General::pcltnvi = "No appropriate paclet with name `1` and version `2` is installed."

Paclet::selector = "Unknown selector `1` for Paclet expression."

Attributes[Paclet] = {ReadProtected}

(* The tests for StringQ["Name"] and StringQ["Version"] are just to prevent messages and pink boxes from showing up
   if a user manually enters a Paclet[] expression, or reads one directly from a PI.m file using Get. Such Paclet
   expressions are invalid, and it is an error to use one in a PacletManager function, but we can at least prevent
   error messages. One could imagine a user writing a program to read/write PI.m files; the Paclet[] expresions in
   M would never be used as true paclets (never passed in to a PM function), but they might be programmatically
   manipulated or interrogated. We might as well prevent copious error messages from showing up when users try this.
*)
Format[p_Paclet, OutputForm] := "Paclet[" <> p["Name"] <> ", " <> p["Version"] <> ", <>]" /; StringQ[p["Name"]] && StringQ[p["Version"]]
Format[p_Paclet, TextForm] := "Paclet[" <> p["Name"] <> ", " <> p["Version"] <> ", <>]" /; StringQ[p["Name"]] && StringQ[p["Version"]]

Paclet /: MakeBoxes[p_Paclet, fmt_] :=
    With[{literalName = "\"\<" <> p["Name"] <> "\>\"", literalVersion = "\"\<" <> p["Version"] <> "\>\""},
        InterpretationBox[RowBox[{"Paclet", "[", literalName, ",", literalVersion, ",", "<>", "]"}], p]
    ] /; StringQ[p["Name"]] && StringQ[p["Version"]]

(*
Paclet /: MakeBoxes[p_Paclet, fmt_] := BoxForm`ArrangeSummaryBox[Paclet, p, 
 summaryBoxIcon, {BoxForm`SummaryItem[{"Name: ", p["Name"]}], 
  BoxForm`SummaryItem[{"Version: ", p["Version"]}]}, {BoxForm`SummaryItem[{"Location: ", 
    p["Location"]}]}, fmt]
*)
(* TODO: I need to consider using this selector notation for non-PI properties like Location (already using),
   MainLink, Key, QualifiedName, Loading, Context, etc. I think that I should do this, and document the full list
   of selectors. Note that this (along with PacletInformation) is the only way for users to programmatically
   interrogate paclets for info. Thus, it needs to be complete. For example, how would you get the list of
   contexts supported by a paclet? p["Context"] is one way, but do I then have selectors for every piece of
   info? Even constructed ones like Contexts? What if a user wants to know if the documentation extension has
   Language->Japanese? This is why we have a p["PacletInfo"] selector that returns the full PacletInfo.m data.
*)
(* Selectors quietly return Null if paclet does not have that element, and if it doesn't have a default value.
   Currently not issuing a message for future-compatibility reasons.
*)
p_Paclet[selector_String] := getPIValue[p, selector]
p_Paclet[selectors:{__String}] := getPIValue[p, selectors]


(*********************************  getPIValue  ************************************)

(* TODO: Tension between getPIValue and selector defs above for user-level extraction of
   paclet info. Do I need both? Is it sensible to use defaults in the case of the unified function?
   [I think the selector style is for users, the getPIValue style is for me.]
*)

(* getPIValue is the function that extracts a value from the expression representation of a paclet.
   Best to keep this encapsulated so we can change the representation as desired.
*)
(* These can be made a little faster by eliminating the overuse of subst rules. Definitely see if
   a Dispatch list that has all fields->Null would be faster than the Thread business.
*)
getPIValue[paclet_Paclet, field_String] := field /. (List @@ paclet) /. $piDefaults /. field->Null
getPIValue[paclet_Paclet, fields:{__String}] := fields /. (List @@ paclet) /. $piDefaults /. Thread[fields->Null]
(* Separate rules for "Extensions" because replacing with $piDefaults will go inside Extensions value and replace subitems. *)
getPIValue[paclet_Paclet, "Extensions"] := "Extensions" /. (List @@ paclet) /. "Extensions" -> {}
getPIValue[paclet_Paclet, "QualifiedName"] := PgetQualifiedName[paclet]
(* Special rules to support older paclets that use "MathematicaVersion" instead of "WolframVersion". *)
getPIValue[paclet_Paclet, fields:{___String, "WolframVersion", ___String}] :=
    fields /. Replace[List @@ paclet, ("MathematicaVersion" -> v_) :> ("WolframVersion" -> v), {1}] /. $piDefaults /. Thread[fields->Null]
getPIValue[paclet_Paclet, "WolframVersion"] :=
    Block[{plist = List @@ paclet, v},
        v = "MathematicaVersion" /. plist;
        If[v === "MathematicaVersion", "WolframVersion" /. plist /. $piDefaults, v]
    ]
(* Gives the full PacletInfo.m data (note that the lhs of all rules are strings, not symbols as typically written in the PI.m file). *) 
getPIValue[paclet_Paclet, "PacletInfo"] := List @@ DeleteCases[paclet, "Location" -> _]

$linkableExtensions = "Documentation" | "Demonstration"


(* Only top-level fields go here, not fields that are part of Extensions elements. *)
$piDefaults = Dispatch[{
    "Extensions" -> {},
    "SystemID" -> All,
    "WolframVersion" -> "10+",
    "ProductName" -> All,
    "Qualifier" -> "",
    "Internal" -> False,
    "Root" -> ".",
    "BackwardCompatible" -> True,
    "BuildNumber" -> "",
    "Description" -> "",
    "InstallFromDocRequest" -> False,
    "ID" -> "",
    "Creator" -> "",
    "URL" -> "",
    "Publisher" -> "",
    "Support" -> "",
    "Category" -> "",
    "Thumbnail" -> "",
    "Copyright" -> "",
    "License" -> "",
    "Loading" -> Manual,
    "Updating" -> Manual
}]

(********************************  Paclet "object" methods  *************************)

(* TODO *)
PisPacked[paclet_] = False

PgetQualifiedName[paclet_] :=
    Block[{n, p, v},
        {n, p, v} = getPIValue[paclet, {"Name", "Qualifier", "Version"}];
        If[p == "",
            ExternalService`EncodeString[n, "UTF-8"] <> "-" <> v,
        (* else *)
            ExternalService`EncodeString[n, "UTF-8"] <> "-" <> p <> "-" <> v
        ]
    ]

PgetKey[paclet_] := {PgetQualifiedName[paclet], PgetLocation[paclet]}

PgetLocation[paclet_] := getPIValue[paclet, "Location"]

PhasContext[paclet_, context_String] :=
    Block[{listedContexts, contextPos},
        listedContexts = ReplaceList["Context", Flatten[Rest /@ 
                             cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID", "ProductName"}]]];
        (* This step handles cases where ctxt was specified as a {ctxt, path} pair. *)
        listedContexts = Flatten[Replace[listedContexts, {ctxt_String, path_String} :> ctxt, {2}]];
        Which[
            Length[listedContexts] == 0,
                False,
            MemberQ[listedContexts, context],
                True,
            True,
                (* If the context we are looking for is a subcontext, we only require that the parent context be named. *)
                contextPos = StringPosition[context, "`"];
                Length[contextPos] > 1 && MemberQ[listedContexts, StringTake[context, contextPos[[1,1]]]]
        ]
    ]

PgetContexts[paclet_] :=
    Module[{kernelExt},
        forEach[kernelExt, cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID", "ProductName"}],
            Replace[EXTgetProperty[kernelExt, "Context", {}], {ctxt_String, path_String} :> ctxt, {1}]
        ] // Flatten
    ]


PhasLinkBase[paclet_, linkBase_String] :=
    If[getPIValue[paclet, "Name"] == linkBase,
        True,
    (* else *)
        MemberQ[ReplaceList["LinkBase", Flatten[Rest /@ cullExtensionsFor[PgetExtensions[paclet, $linkableExtensions], {"WolframVersion", "SystemID", "ProductName"}]]], linkBase]
    ]


(* Always returns a list (empty if no extensions of requested type). *)
PgetExtensions[paclet_] := getPIValue[paclet, "Extensions"]
PgetExtensions[paclet_, extTypes:(_String | _Alternatives)] := Cases[PgetExtensions[paclet], {extTypes, ___}]

(* This is the function called to get a full path to a doc/demonstration notebook. The file is checked to
   ensure that it exists. It returns Null if the paclet cannot supply this resource. Cannot meaningfully
   be called on a paclet that is not locally installed.

   Called in two different circumstances. The main use is when we are looking up a doc path from a URI that
   has been decomposed into a linkBase/resName pair. The other use is when we are looking up a message doc page
   from a context and sym::tag (that has already been turned into the resourceName). In that case, the paclet can
   announce that it supplies docs for the given context either by having a Kernel extension with that Context,
   or, much more rarely, Doc extensions can announce that they contain docs for symbols in a context via their
   own Context property. An example of that would be a paclet that provided nothing but docs for a separate
   paclet that contained the code. The other paclet has the Kernel extension, and the doc-only paclet needs a way
   to say that it has docs for a specific context.
*)
PgetDocResourcePath[paclet_, linkBase_, context_, expandedResourceName_, lang_] :=
    Module[{linkBaseMatchesPacletName, pacletRootPath, extLinkBase, extLanguage,
              docRoot, resPath, fullPath, isContextBasedLookup},

        (* We are always doing either a linkBase-based lookup (most common) or a context-based lookup (only for message
           links), thus exactly one of linkBase and context must be a string.
        *)
        Assert[(StringQ[linkBase] || StringQ[context]) && !(StringQ[linkBase] && StringQ[context])];

        If[!systemIDMatches[getPIValue[paclet, "SystemID"]] ||
             !kernelVersionMatches[getPIValue[paclet, "WolframVersion"]] ||
                !languageMatches[lang, getPIValue[paclet, "Language"]] ||
                   !productNameMatches[getPIValue[paclet, "ProductName"]],
           Return[Null]
        ];

        linkBaseMatchesPacletName = linkBase === paclet["Name"];
        isContextBasedLookup = StringQ[context];

        Scan[
            Function[{ext},
                (* Check for match of linkbase, context, language. *)

                (* I'm not sure if Language->All is useful in extensions, but atm it is used in some of our own paclets.
                   Here I support it by pretending that Language->All was Language->"current requested lang".
                *)
                extLanguage = EXTgetProperty[ext, "Language", "English"] /. All -> lang;

                extLinkBase = EXTgetProperty[ext, "LinkBase"];
                If[(extLinkBase === Null && linkBaseMatchesPacletName || extLinkBase === linkBase ||
                    isContextBasedLookup && (PhasContext[paclet, context] || EXTgetProperty[ext, "Context"] === context)) &&
                       (lang === All || languageMatches[lang, extLanguage]),

                    (* We know this ext matches linkbase/context and language. Now, build all possible paths
                       and see if any of them point to an existing file. Return it immediately if we find one.
                    *)
                    pacletRootPath = PgetPathToRoot[paclet];
                    docRoot = EXTgetProperty[ext, "Root", "Documentation"];
                    resPath = EXTgetResourcePath[ext, expandedResourceName];
                    (* If there was a custom path in URL form, use it, otherwise construct a full path to the file. *)
                    If[StringMatchQ[resPath, "file:*"] || StringMatchQ[resPath, "http*:*"],
                        Return[resPath],
                    (* else *)
	                    (* Try with and without a language-specific subdir. *)
	                    fullPath = ToFileName[{pacletRootPath, docRoot, extLanguage}, resPath];
	                    If[FileExistsQ[fullPath],
	                        Return[ExpandFileName[fullPath]]
	                    ];
	                    fullPath = ToFileName[{pacletRootPath, docRoot}, resPath];
	                    If[FileExistsQ[fullPath],
	                       Return[ExpandFileName[fullPath]]
	                    ]
                    ]
                ]
            ],
            cullExtensionsFor[PgetExtensions[paclet, $linkableExtensions], {"WolframVersion", "SystemID", "ProductName"}]
        ]  (* Fall through to Scan's return value of Null if no file found. *)
    ]



PgetPathToRoot[paclet_Paclet] := ToFileName[PgetLocation[paclet], getPIValue[paclet, "Root"]]


(* Returns a paclet URI that this paclet will resolve to an actual file, or Null if no such link exists. This will be
   paclet:PacletName or paclet:SomeLinkBaseNamedInADocExtension.
*)
PgetMainLink[paclet_Paclet] :=
    Module[{name, extLinkBase},
        name = getPIValue[paclet, "Name"];
       (* First, try paclet:Name. The following call looks for a path when the linkbase is the paclet name. *)
        If[StringQ[PgetDocResourcePath[paclet, name, All, "", $Language]] ||
                $Language != "English" && StringQ[PgetDocResourcePath[paclet, name, All, "", "English"]],
            "paclet:" <> name,
        (* else *)
            (* Look at all doc exts and, for each linkBase they name, see if paclet:LinkBase will resolve to an actual
               file for this paclet. This Scan will either return a URI if found, or Null if not.
            *)
            Scan[
                Function[{docExt},
                    extLinkBase = EXTgetProperty[docExt, "LinkBase"];
                    If[extLinkBase =!= Null &&
                          (StringQ[PgetDocResourcePath[paclet, extLinkBase, All, "", $Language]] ||
                           $Language != "English" && StringQ[PgetDocResourcePath[paclet, extLinkBase, All, "", "English"]]),
                         Return["paclet:" <> extLinkBase]
                    ]
                ],
                cullExtensionsFor[PgetExtensions[paclet, "Documentation"], {"WolframVersion", "SystemID", "ProductName"}]
            ]
        ]
    ]


PgetLoadingState[paclet_Paclet] :=
    Block[{val},
        val = getPIValue[paclet, "Loading"];
        Switch[ToLowerCase[ToString[val]],
            "automatic",
                Automatic,
            "startup",
                "Startup",
            _,
                Manual
        ]
    ]

PgetPreloadData[paclet_Paclet] :=
    Block[{kernelExt, ctxt, result},  (* Block for speed only. *)
        result = {};
        forEach[kernelExt, cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID", "ProductName"}],
            forEach[ctxt, EXTgetProperty[kernelExt, "Context", {}],
                (* First[Flatten[{ctxt}]] here because ctxt could be "context`", {"context`"}, or {"context`", "path"}. *)
                If[ListQ[#] && StringQ[First[#]], AppendTo[result, First[#]]]& @ contextToFileName[paclet, First[Flatten[{ctxt}]]]
            ]
        ];
        result
    ]

PgetDeclareLoadData[paclet_Paclet] :=
    Block[{kernelExt, ctxts, firstCtxt, symbols, hiddenImport, result},  (* Block for speed only. *)
        result = {};
        forEach[kernelExt, cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID", "ProductName"}],
            ctxts = Flatten[{EXTgetProperty[kernelExt, "Context", {}]}];
            (* If you want to have a Symbols spec for autoloading, you probably have just one context
               named in the extension. Use separate exts for multiple contexts. If there is more than
               one context, we just assume that all the listed symbols are in the first-listed context.
            *)
            If[Length[ctxts] > 0,
                firstCtxt = First[Flatten[ctxts]];
                symbols = EXTgetProperty[kernelExt, "Symbols", {}];
                hiddenImport = EXTgetProperty[kernelExt, "HiddenImport"];
                Which[
                    StringQ[hiddenImport] && StringEndsQ[hiddenImport, "`"],
                        (* Was a context; use it unmodified, so nothing to do here. *)
                        Null,
                    MatchQ[hiddenImport, {_String}],
                        (* Developer mistakenly used a list {"context`"} instead of just a string. Handle it. *)
                        hiddenImport = First[hiddenImport],
                    hiddenImport === {} || hiddenImport === None,
                        (* None is not the same as False. None means "add no context to $Packages". *)
                        hiddenImport = None,
                    TrueQ[hiddenImport],
                        (* Was True; use it unmodified, so nothing to do here. *)
                        Null,
                    True,
                        (* Any other value is treated as False *)
                        hiddenImport = False                        
                ];
                If[Length[symbols] > 0,
                    (* The symbols in the list need their context prepended; the list we are appending should look
                       like {"ctxt`", {"ctxt`sym`", "ctxt`sym2",...}}. We don't want the user to need to prepend the context
                       in the PI file (they should be able to write Symbols->{"Foo"}, not Symbols->{"ctxt`Foo"}), so we
                       prepend it here. If the user has a specific context prepended in the PI file, we skip automatic prepending.
                    *)
                    AppendTo[result, {firstCtxt, hiddenImport, If[StringMatchQ[#, "*`*"], #, firstCtxt <> #]& /@ symbols}]
                ]
            ]
        ];
        result
    ]

(* Because our parsing scheme doesn't allow arbitrary M exprs in PI.m files (we want everything to be strings),
   it doesn't work to put FunctionInformation inline into the PI.m file. Instead, we use FunctionInformation->"file"
   (where the default is FunctionInformation.m).
*)
PgetFunctionInformation[paclet_Paclet] :=
    Block[{kernelExt, funcInfoFile, funcInfo, pacletRootPath, kernelRoot, strm, fullPath},  (* Block for speed only. *)
        result = {};
        forEach[kernelExt, cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID", "ProductName"}],
            funcInfoFile = EXTgetProperty[kernelExt, "FunctionInformation", "FunctionInformation.m"];
            If[StringQ[funcInfoFile],
                pacletRootPath = PgetPathToRoot[paclet];
                kernelRoot = EXTgetProperty[kernelExt, "Root"];
                fullPath = ToFileName[{pacletRootPath, kernelRoot}, funcInfoFile];
                If[FileExistsQ[fullPath],
                    using[{strm = OpenRead[fullPath]},
                        funcInfo = Read[strm, Expression]
                    ]
                ]
            ];
            If[MatchQ[funcInfo, {{_String, {_List...}}...}],
                result = result ~Join~ funcInfo
            ]
        ];
        result
    ]



(* Paclet expressions are never changed, except for one case--when they are first created from reading a
   PacletInfo.m file, they need the Location field added.
*)
setLocation[paclets:{___Paclet}, location_String] := setLocation[#, location]& /@ paclets

setLocation[paclet_Paclet, location_String] :=
    If[getPIValue[paclet, "Location"] === Null,
        (* Canonicalize location using ExpandFileName. *)
        Append[paclet, "Location" -> #],
    (* else *)
        paclet /. ("Location"->_) :> ("Location"->#)
    ]& @ If[StringMatchQ[location, "http*:*", IgnoreCase->True] || StringMatchQ[location, "file:*", IgnoreCase->True], location, ExpandFileName[location]]



(* Contains code for verifying the correctness of PacletInfo.m files. *)


(*********************************  VerifyPaclet  *************************************)

VerifyPaclet::path = "`1` must be a .paclet file, a PacletInfo.m or PacletInfo.wl file, or a directory containing a PacletInfo.m or PacletInfo.wl file."
VerifyPaclet::notfound = "Could not find specified file `1`."
VerifyPaclet::nopi = "The directory `1` does not have a PacletInfo.m or PacletInfo.wl file in it."
VerifyPaclet::noppi = "The paclet archive `1` does not appear to have a PacletInfo.m or PacletInfo.wl file in it."
VerifyPaclet::rules = "Invalid PacletInfo.m data: Paclet expression must be a list of rules."
VerifyPaclet::noname = "Invalid PacletInfo.m data: Paclet must have a Name field."
VerifyPaclet::novers = "Invalid PacletInfo.m data: Paclet must have a Version field."
VerifyPaclet::badvers = "Invalid PacletInfo.m data: Version number must be one or more blocks of digits separated by periods."
VerifyPaclet::badpi = "The contents of the PacletInfo.m file either is not a syntactically correct Wolfram Language expression, it does not have the head Paclet."
VerifyPaclet::extlist = "Invalid PacletInfo.m data: The Extensions specification must be a list."
VerifyPaclet::badext = "Invalid PacletInfo.m data: `1`."
VerifyPaclet::badextt = "Invalid PacletInfo.m data: An Extensions element of type `1` does not have the correct form. `2`."


Options[VerifyPaclet] = {Verbose -> True}


VerifyPaclet[p_Paclet, OptionsPattern[]] := 
    Module[{name, vers, verbose},
        verbose = OptionValue[Verbose];
        If[!MatchQ[p, _[(_String -> (_String | _Symbol | _List))..]],
            If[verbose, Message[VerifyPaclet::rules]];
            Return[False]
        ];
        {name, vers} = getPIValue[p, {"Name", "Version"}];
        If[!StringQ[name],
            If[verbose, Message[VerifyPaclet::noname]];
            Return[False]
        ];
        If[!StringQ[vers],
            If[verbose, Message[VerifyPaclet::novers]];
            Return[False]
        ];
        If[!StringMatchQ[vers, (DigitCharacter | ".")..],
            If[verbose, Message[VerifyPaclet::badvers]];
            Return[False]
        ];
        If[!verifyExtensions[p, verbose],
            Return[False]
        ];
        True
    ]


(* Although this version takes the Verbose option, it is not very useful. The only reason for the Verbose option
   is so VerifyPaclet[_Paclet], which is called internally by CreatePaclet, can be silenced so as to prevent
   distracting details from being shown to users who might know nothing about a paclet. But this version below is
   only called directly by users, and they would have no reason to silence it (and if they did, they would just
   wrap it in Quiet).
*)
VerifyPaclet[pacletFileOrDir_String, OptionsPattern[]] :=
    Module[{piData, strm, dataBuffer, verbose, piExpr},
        verbose = OptionValue[Verbose];
        piData = $Failed;
        Which[
            StringMatchQ[FileNameTake[pacletFileOrDir], "PacletInfo.m"] || StringMatchQ[FileNameTake[pacletFileOrDir], "PacletInfo.wl"],
                using[{strm = Quiet[OpenRead[pacletFileOrDir, BinaryFormat -> True]]},
                    If[Head[strm] === InputStream,
                        piData = FromCharacterCode[BinaryReadList[strm], "UTF8"]
                    ]
                ],
            StringMatchQ[pacletFileOrDir, "*.paclet"] || StringMatchQ[pacletFileOrDir, "*.cdf"],
                If[FileType[pacletFileOrDir] === File,
                    Quiet[
                        dataBuffer = ZipGetFile[pacletFileOrDir, "PacletInfo.m"];
                        If[!ListQ[dataBuffer],
                            dataBuffer = ZipGetFile[pacletFileOrDir, "PacletInfo.wl"]
                        ]
                    ];
                    If[ListQ[dataBuffer],
                        piData = FromCharacterCode[dataBuffer, "UTF8"],
                    (* else *)
                        If[verbose, Message[VerifyPaclet::noppi, pacletFileOrDir]]
                    ],
                (* else *)
                    If[verbose, Message[VerifyPaclet::notfound, pacletFileOrDir]]
                ],
            DirectoryQ[pacletFileOrDir],
                If[Length[#] > 0,
                    Return[VerifyPaclet[First[#], Verbose->verbose]],
                (* else *)
                    If[verbose, Message[VerifyPaclet::nopi, pacletFileOrDir]]
                ]& @ FileNames[{"PacletInfo.m", "PacletInfo.wl"}, pacletFileOrDir],
            True,
                Message[VerifyPaclet::path, pacletFileOrDir]
        ];
        If[StringQ[piData],
            piExpr = parsePacletInfo[piData];
            Switch[piExpr,
                {_Paclet},
                    VerifyPaclet[First[piExpr], Verbose->verbose],
                {__Paclet},
                    And @@ (VerifyPaclet[#, Verbose->verbose]& /@ piExpr),
                _,
                    If[verbose, Message[VerifyPaclet::badpi, pacletFileOrDir]];
                    False
            ],
        (* else *)
            (* Message already issued. *)
            False            
        ]
    ]


verifyExtensions[p_Paclet, verbose_] :=
    Module[{exts},
        exts = PgetExtensions[p];
        If[!ListQ[exts],
            If[verbose, Message[VerifyPaclet::extlist]];
            Return[False]
        ];
        If[!MatchQ[exts, {___List}],
            If[verbose, Message[VerifyPaclet::badext, "Each element of the Extensions list must itself be a list"]];
            Return[False]
        ];
        And @@ (verifyExtension[#, verbose]& /@ exts)
    ]

(*  TODO: Add some meat to these. But these are speed-sensitive operations. Use the verbose parameter to decide
    whether to do deep inspection.
*)
verifyExtension[{"Documentation", rest___Rule}, verbose_] = True  
verifyExtension[{"Kernel" | "Application", rest___Rule}, verbose_] = True  
verifyExtension[{"Resource", rest___Rule}, verbose_] = True  
verifyExtension[{t_String, rest___Rule}, verbose_] = True
  
verifyExtension[_, verbose_] :=
    (   
        If[verbose, Message[VerifyPaclet::badext, "Each Extensions element must be a list consisting of a string naming the extension type followed by a sequence of rules"]];
        False
    )
    
    
(*****************************  Summary Box Icon  ****************************)

summaryBoxIcon =
BoxData@GraphicsBox[
TagBox[RasterBox[CompressedData["
1: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
IearskUn+5zsghqVnCAAViz51vEfUy6Pfw==
"], {{0, 46}, {46, 0}}, {0, 255},
ColorFunction->RGBColor],
BoxForm`ImageTag[
     "Byte", ColorSpace -> "RGB", ImageSize -> Automatic, 
      Interleaving -> True, Magnification -> Automatic],
Selectable->False],
BaseStyle->"ImageGraphics",
ImageSize->Automatic,
ImageSizeRaw->{36, 36},
PlotRange->{{0, 46}, {0, 46}}]

        
End[]

