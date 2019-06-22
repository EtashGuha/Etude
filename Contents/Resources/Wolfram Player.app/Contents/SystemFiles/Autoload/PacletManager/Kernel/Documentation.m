(* ::Package:: *)

(* :Title: Documentation.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 6.0 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


Begin["`Package`"]

(* Called outside of this package, in buttons on failed fragment-loading messages. *)
resolveURL

convertResourceNameToLongForm

(* unused *)
doProgressPage
cancelDownload

getUpdatablePacletResource (* unused ? *)

initSearchCache

(* Functions called from InstalledPacletsAndAddOns page. *)
enabledStateRadioFunction
loadStateRadioFunction
loadButtonFunction
uninstallButtonFunction
updateButtonFunction
disableCheckboxFunction
enabledStateRadioEnabled
loadStateRadioEnabled
loadButtonState
uninstallButtonPresent
updateButtonEnabled
loadStateRadioPresent
disableCheckboxPresent



End[]  (* `Package` *)


(* Current context will be PacletManager`. *)

Begin["`Documentation`Private`"]


(*************************  Functions called from kernel's Documentation.m code  ***************************)

(* These are the functions from StartUp/Documentation.m that we add defs for.
   This is the public API defined for documentation lookups by the PM.
*)
Unprotect[Documentation`HelpLookupPacletURI]
Unprotect[Documentation`CreateMessageLink]
Unprotect[Documentation`ResolveLink]


(* Tells what file the PM thinks a paclet: link resolves to.
   Does allow downloads. Not asynchronous.
   Note that it works without a FE present.
*)
Documentation`ResolveLink[link_] :=
    Module[{language = Quiet[System`CurrentValue[$FrontEnd, Language]]},
        If[StringQ[language],
            Documentation`ResolveLink[link, language],
        (* else *)
            Documentation`ResolveLink[link, $Language]
        ]
    ]

Documentation`ResolveLink[link_, language_String] :=
    Documentation`ResolveLink[link, language, True]

Documentation`ResolveLink[link_, language_String, allowDownload:(True | False)] :=
    Module[{result},
        result = Quiet[resolveURL[link, language, {Null, Null} (*ignored*), Null&, !allowDownload, False, False]];
        If[ListQ[result],
            First[result],
        (* else *)
            Null
        ]
    ]


(*
  HelpLookupPacletURI

  Called for clicks on links to docs in front end.
  Resolve link then open result notebook in FE or browser.
  For InstallFromDocRequest paclets, will result in download from server. Otherwise
  will open in browser window to web docs.
  Note that this is asynchronous; the NotebookLocate is called later, probably on ServiceLink.

  in:  String uri (e.g. "paclet:ref/Partition")
  out: Null for async lookup, Notebook object for successful sync operation, $Failed for failure.
*)

Documentation`HelpLookupPacletURI::link = "Could not resolve paclet URL `1` to a document."

Documentation`HelpLookupPacletURI[link_String, feData_, language_, opts:OptionsPattern[]] :=
     Module[{feDataFixed, resolved, lang = language, completionFunc, nb},
         If[!StringQ[lang], lang = "English"];
         (* Resolve the FrontEnd`ButtonNotebook[] call right now in the kernel, because
            the FE gets the feData asynchronously and cannot resolve FrontEnd`ButtonNotebook[]
            at that time.
         *)
         feDataFixed = feData /. FrontEnd`ButtonNotebook[] :> ButtonNotebook[];
         completionFunc = OptionValue[Documentation`CompletionFunction];
         (* The AbortProtect is to prevent the FE's 6 second TimeConstrained for dynamic evals from
            aborting this operation. It doesn't do anything truly time-consuming, so it should
            be harmless to prevent aborts. One reason a 6-second TimeConstrained might kick
            in is that the search system takes several seconds to initialize the first time.
            Another reason is that prolonged swapping is always a possibility.
         *)
         resolved =
             AbortProtect[                  
                 (* We want to give the TextSearch system a chance to index legacy paclets that have only old-style search indices.
                    We do this here, on a first-search-per-session basis (enforced within ReindexLegacyPacletsAndSearch). That function
                    indexes within a ScheduledTask, and then that task re-calls HelpLookupPacletURI. Therefore we just return $future here
                    and let the second search finish asynchronously. The StringStartsQ test here is a poor-man's test to decide whether
                    this lookup is indeed a search. The full logic of whether a search is needed is complex and occurs within resolveURL,
                    but we at least want a simple test to avoid reindexing when the user is just browsing links.
                 *)
                 If[!StringStartsQ[link, "paclet:"] && TrueQ[DocumentationSearch`ReindexLegacyPacletsAndSearch[{link, feDataFixed, lang, opts}]],
                     $future,
                 (* else *)
                     resolveURL[link, lang, feDataFixed, completionFunc, False, TrueQ[$allowDialogs], True]
                 ]
             ];
         (*  Note that the resolveURL call above is made asynchronously, so the ListQ[resolved]
             branch here will never be entered. Nevertheless, I preserve it in case we ever
             decide to handle some or all cases synchronously.
        *)
         Which[
             resolved === $future,
                 (* Was async call. Arbitrarily choose to return Null. *)
                 Null,
             ListQ[resolved],
                 (* Was successful synchronous call. *)
                 nb = locateNotebook[resolved, feDataFixed];
                 If[Head[nb] === NotebookObject,
                     completionFunc[nb]
                 ];
                 nb,
             True,
                 (* Failure in async or sync call. This makes the front end display a dialog. *)
                 MathLink`CallFrontEnd[FrontEnd`ErrorMessage["ItemNotFound", link]]
         ]
     ]



(* This is the function that message-formatting code in the kernel calls to decide
   whether to create a link button for the message. This function returns either a "paclet:..."
   string suitable for the ButtonData of a link, or non-string to indicate that no link
   should be created. It will return a string iff such a link can be resolved by the
   PacletManager to a document. Will consider server paclets, but only if $AllowInternet and
   $AllowDocumentationUpdates are true, and if they are InstallFromDocRequest paclets.
*)
Documentation`CreateMessageLink[context_String, symbolName_String, msgTag_String, language_String] :=
    Module[{linkBase, result, ctxt},
        (* For System` symbols, use a LinkBase of WolframMathematica. For non-System` symbols,
           find all paclets that provide that context (context-based lookup means linkBase of null).
        *)
        If[context == "System`",
            linkBase = "WolframMathematica";
            ctxt = All,
        (* else *)
            linkBase = All;
            ctxt = context
        ];
        
        (* Note that we don't limit search to installed paclets. If we add a new message doc file to
           the server (a new file, not just a new version), we want it picked up and downloaded on demand.
           But we only want to build links for InstallFromDocRequest paclets, as only these can be
           downloaded and installed when the link is clicked.
        *)
        If[msgTag == "usage",
            result = createKnownGoodMessageURI[linkBase, ctxt, "ref/" <> symbolName, language];
            If[StringQ[result],
                Return[result]
            ];
            If[ctxt =!= All,
                result = createKnownGoodMessageURI[linkBase, ctxt, "ref/" <> StringReplace[ctxt, "`"->"/"] <> symbolName, language];
                If[StringQ[result],
                    Return[result]
                ]
            ],
        (* else *)
            (* Warning message, not usage message. For each symbol, try a tag associated with that symbol and with General.*)
            result = createKnownGoodMessageURI[linkBase, ctxt, "ref/message/" <> symbolName <> "/" <> msgTag, language];
            If[StringQ[result],
                Return[result]
            ];
            If[ctxt =!= All,
                result = createKnownGoodMessageURI[linkBase, ctxt, "ref/message/" <> StringReplace[ctxt, "`"->"/"] <> symbolName <> "/" <> msgTag, language];
                If[StringQ[result],
                    Return[result]
                ]
            ];
            (* This finds General messages for symbols in System` and non-System` contexts. *)
            result = createKnownGoodMessageURI["WolframMathematica", All, "ref/message/General/" <> msgTag, language];
            If[StringQ[result],
                Return[result]
            ];
            If[symbolName != "General",
                (* If no specific message notebook found, fall through to looking for the symbol's ref page.
                   But don't do this for the symbol General, as we don't want messages like General::foo
                   that don't have a message page creating a link to the General usage page.
                *)
                result = Documentation`CreateMessageLink[context (* yes, original context here *), symbolName, "usage", language];
                If[StringQ[result],
                    Return[result]
                ]
            ];
            (* One final attempt. Some older paclets have neither a Kernel extension nor a Context property in their Doc 
               extension. This means that they do not announce in any way that they might contain docs for symbols in any
               context, and thus these docs would not be found by any of the above lookups. Examples of such paclets are
               all the ones in <M dir>/AddOns/Applications. Newer paclets should have a Kernel extension or, if they have
               only docs, a Context property in their Doc extension. To support older paclets, here we try the context name
               as the linkbase (meaning that paclets whose name or linkbase is the same as their context will work).
            *)
            If[ctxt =!= All,
                result = createKnownGoodMessageURI[First[StringSplit[ctxt, "`"]], All, "ref/" <> symbolName, language];
                If[StringQ[result],
                    Return[result]
                ]
            ]
        ];
        (* Failed to find an appropriate URI. *)
        Null
    ]


Protect[Documentation`HelpLookupPacletURI]
Protect[Documentation`CreateMessageLink]
Protect[Documentation`ResolveLink]


(*************************************  Implementation  ************************************)

(* The first component of a paclet URI is the LinkBase. For convenience, though,
   we want to allow users to leave out the "WolframMathematica" LinkBase for built-in system
   docs. That is:
       paclet:ref/Partition
   instead of requiring
       paclet:WolframMathematica/ref/Partition
   Thus we look for one of the special categories of system docs as the first
   element of the URI and recognize that this is actually the beginning of the resourceName
   part and that there is an empty LinkBase component that should be defaulted to
   WolframMathematica. Add to this list as new categories (directories) are created for
   the root level of the docs hierarchy.

*)

(* Including ReferencePages, Guides, Tutorials, etc. allows us to support non-shortened links. *)
$specialLinkBases = {"ref", "guide", "tutorial", "note", "howto", "example", "workflow", "workflowguide", 
                     "ReferencePages", "Guides", "Tutorials", "Notes", "HowTos", "ExamplePages", "Workflows", "WorkflowGuides"}
                     

$hasCheckedForFallbackNonEnglishLang = False

(*
    Accepts:
        paclet:Guides/WolframRoot   (the typical use; default to WolframMathematica LinkBase))
        paclet://Guides/WolframRoot (legacy support for incorrect paclet:// urls)

        paclet:LinkBase/Guides/SomeGuide   (the typical use)
        paclet://LinkBase/Guides/SomeGuide (legacy support for incorrect paclet:// urls)

        paclet:SingleWord   (must match a LinkBase provided by a paclet; give that paclet's root guide page)
        SingleWord          (if matches a LinkBase, treat like paclet:SingleWord. Otherwise, treat as
                             a poor-man's search: look for any paclet (start with WolframMathematica) that
                             can provide this doc via Symbols, Guides, etc.)

        guide/WolframRoot
        Guides/WolframRoot     (/ but no paclet:, simply prepend with paclet:)

        paclet:SearchResult?query   (special treatment as search)

   Does not accept:
        http://anything
        Symbols/Partition  (if it has a / char but no paclet:, it must be a full URL except missing paclet:)


   Returns list:
    {
     "full path string or http URL to notebook",
     "fragment part",
     "paclet URL" (possibly modified from original to make legal}
    }

    or $Failed or $future.

   The first element is the full path string or http URL to a notebook,
   or $Failed if there is an error. This will simply prevent the NotebookLocate
   from firing because the first part is not a string. Also will issue message on error.
   The second element is the fragment part, if it exists (as in
   the "abc" part of paclet:Guides/Foo#abc, typically a cell tag).
   If no fragment part exists, None is used, which is what the front end expects
   in NotebookLocate.
*)
resolveURL[url_String, language_String, feData_, completionFunc_,
            localFilesOnly:(True | False), interactive:(True | False), async:(True | False)] :=
    Module[{trimmedURL, resource, resourceWithoutFragment, wordsOnly, fallThroughToSearch,
                fragPart, parts, linkBase, resourceName, context, pathOrURL,
                    symbolNamePieces, directHitResult, searchTextTranslated, directHitURI, searchHeaderData, langDirs},

        wordsOnly = False;
        context = All;

        (* First, trim whitespace from start and end *)
        trimmedURL = StringReplace[url, StartOfString ~~ WhitespaceCharacter ... ~~ Shortest[mid___] ~~ WhitespaceCharacter ... ~~ EndOfString :> mid];

        (* Treat all requests that are nothing but whitespace chars as a request for the root page. *)
        If[trimmedURL == "", trimmedURL = "paclet:guide/WolframRoot"];

        (* Some links should simply fail if not resolved, and not fall through to a search. In this category
           we put links that have an explicit paclet: prefix, and links that have a / in them (like someone entered
           ref/Foo).
        *)
        fallThroughToSearch = !looksLikePacletURI[trimmedURL];

        (* Drop paclet: at start. NOTE temporarily, handle either paclet: or paclet:// at beginning.
           The paclet: form is the one we will use, but at this point some links still have paclet://.
        *)
        resource =
            Which[
                StringMatchQ[trimmedURL, "paclet://*"],
                    (* paclet:// is not a legal prefix, but we can easily handle it if users mistakenly use it. *)
                    StringDrop[trimmedURL, 9],
                StringMatchQ[trimmedURL, "paclet:/*"],
                    (* paclet:/ is not a legal prefix, but we can easily handle it if users mistakenly use it. *)
                    StringDrop[trimmedURL, 8],
                StringMatchQ[trimmedURL, "paclet:*"],
                    StringDrop[trimmedURL, 7],
                StringMatchQ[trimmedURL, "* *"],
                    wordsOnly = True;
                    trimmedURL,
                looksLikePacletURI[trimmedURL],
                    (* Treat like a normal paclet: URL except user left out paclet: prefix. *)
                    trimmedURL,
                StringMatchQ[trimmedURL, "installedpacletsandaddons" | "installedaddons", IgnoreCase->True],
                    (* Entering InstalledAddOns should bypass a search (don't set wordsOnly in this case). *)
                    "installedpacletsandaddons",
                True,
                    (* "wordsOnly" URLs do not start with paclet and either have spaces or do not have a / in them.
                       They are like weak searches, where a user types just the name of, say,
                       a symbol or a guide page. WordsOnly is not a very good name for this, because we also
                       get here on gibberish strings like %[+]#, but that's OK, and in fact the search system
                       wants first crack at these in case they are operators.
                    *)
                    wordsOnly = True;
                    If[StringLength[trimmedURL] < 100 && !StringMatchQ[trimmedURL, "*`"],
                        (* Here we extract the context in case it's a symbol like Foo`Function.
                           For example, "a`b`c" --> "a`b`", or "abc" -> "". It doesn't matter
                           if it isn't a symbol name but happens to have a ` in it--it simply
                           won't be found by context-based lookup.
                        *)
                        symbolNamePieces = StringSplit[trimmedURL, "`"];
                        context = StringJoin @@ Riffle[Append[Most[symbolNamePieces], ""], "`"];
                        If[context == "", context = All];
                        Last[symbolNamePieces],
                    (* else *)
                        trimmedURL
                    ]
            ];

        (* In some cases, $Language is English but only non-English language docs are present in the layout. In such cases, we want to fall back to using the
           non-English language. See bug 299613 for the motivation for this feature.
        *)
        If[!$hasCheckedForFallbackNonEnglishLang,
            $hasCheckedForFallbackNonEnglishLang = True;
            langDirs = FileNameTake[#, -1]& /@ Select[FileNames["*", FileNameJoin[{$InstallationDirectory, "Documentation"}]], DirectoryQ];
            If[Length[langDirs] > 0 && !MemberQ[langDirs, "English"],
                $fallbackNonEnglishLang = First[langDirs]
            ]
        ];
        
        (* Want to give the search system first crack at resolving "direct hits", which are
           cases where he user types in a word that is considered by us as a synonym for a
           built-in symbol.
        *)
        searchHeaderData = Null;
        If[wordsOnly && !StringMatchQ[trimmedURL, "* *"],
            loadDocSearchPackage[];
            directHitResult = Symbol["DocumentationSearch`DirectHitSearch"][resource];
            If[MatchQ[directHitResult, {_String, _String}],
                {searchTextTranslated, directHitURI} = directHitResult;
                wordsOnly = False;
                searchHeaderData = {resource, searchTextTranslated};
                context = All;
                resource = directHitURI,
            (* else *)
                searchHeaderData = {resource, resource}
            ]
        ];

        (* Separate fragment part. *)
        If[StringMatchQ[resource, "#*"] || StringMatchQ[resource, "*#*#*"],
            (* If it starts with a # char or has more than one #, then # isn't being used as a fragment spec. *)
            resourceWithoutFragment = resource;
            fragPart = "",
        (* else *)
            parts = StringSplit[resource, "#"];
            resourceWithoutFragment = First[parts];
            If[Length[parts] > 1,
                fragPart = parts[[2]],
            (* else *)
                fragPart = ""
            ]
        ];

        (* Split resource into linkbase and resourcename. *)
        Which[
            StringQ[context],
                (* Contexts only appear when the user enters a symbol with context into the lookup field (Foo`Bar).
                   The PM can then function as a simple search system to find the ref page for that symbol based
                   on which paclets announce they carry docs for that context. Note that the direct-hit search system
                   has already had a crack at finding the full Foo`Bar string, and we only get here if it fails. 
                *)
                {linkBase, resourceName} = {All, "ref/" <> resourceWithoutFragment},
            wordsOnly,
                (* The PacletManager itself is not a search system. For a wordsOnly lookup, the only
                   thing to do is to treat the word as a linkbase, as the PM will not look in all paclets to
                   see if they define a symbol with this name (that is, see if they have this as a ref/name resource).
                   Lookups for built-in symbols and user-defined symbols that are properly indexed have already been done,
                   and if a hit was found we won't get here. This is only to support a user entering the name of a paclet
                   and the PM returning the link to the paclet's "main" page.
                *)
                {linkBase, resourceName} = {resourceWithoutFragment, ""},
            True,
                {linkBase, resourceName} = splitResource[resourceWithoutFragment]
        ];

        pathOrURL = resolveDocResource[linkBase, context, resourceName, language, localFilesOnly][[2]];
        If[StringQ[pathOrURL],
            {pathOrURL, fragPart, trimmedURL, searchHeaderData},
        (* else *)
            (* Lookup failed. If desired, do a search; otherwise fail. *)
            Which[
                (* Bit of a hack: Because "note/" docs removed from product, map them to the web. *)
                StringMatchQ[resource, "note/*SourceInformation", IgnoreCase->True],
                    {"http://reference.wolfram.com/mathematica/" <> resource <> ".html", "", "", Null},
                StringMatchQ[resource, "guide/installedpacletsandaddons" | "installedpacletsandaddons" | 
                                        "guide/installedaddons" | "installedaddons", IgnoreCase->True],
                    {generateNotebook["InstalledPacletsAndAddOns", "InstalledPacletsAndAddOns"], "", trimmedURL, Null},
                fallThroughToSearch,
                    {generateNotebook[resource, "Search"], "", trimmedURL, Null},
                True,
                    If[!interactive,
                        Message[Documentation`HelpLookupPacletURI::link, trimmedURL]
                    ];
                    $Failed
            ]
        ]
    ]


(* This is the function that is called to resolve a resource like ref/Foo or JLink/guide/CallingJava.
   Will fallback to English if the specified language is not found.
   It returns a list of two elements: {paclet, "path/to/file/known/to/exist"}, or {Null, Null}.
*)
resolveDocResource[linkBase:(_String | All), context:(_String | All), resourceName_String, language_String,
                    localFilesOnly:(True | False), allowRecursion:(True | False):True] :=
    Module[{expandedResName, matchingPacletsAndPaths},
        (* At least one must be All. We are either doing a context-based lookup (context != All)
           or a linkBase-based lookup (linkBase != All).
        *)
        Assert[linkBase === All || context === All];
        
        (* resourceName can be "", as happens when the user enters just the name of a paclet and we want to
           resolve this to the paclet's main page. In this case, the linkbase holds the name the user entered.
        *)
               
        (* Convert ref-guide-tutorial style URIs to ReferencePages-Guides-Tutorials style. *)
        expandedResName = convertResourceNameToLongForm[resourceName];

        matchingPacletsAndPaths = PCfindForDocResource["LinkBase" -> linkBase, "Context" -> context,
                                               "ResourceName" -> expandedResName, "Language" -> language];
        
        Which[
            Length[matchingPacletsAndPaths] > 0,
                First[matchingPacletsAndPaths],
            language != "English" && allowRecursion,
                (* Fall back to trying English-language file if correct language was not found. *)
                resolveDocResource[linkBase, context, resourceName, "English", localFilesOnly, False],
            language == "English" && StringQ[$fallbackNonEnglishLang] && allowRecursion,
                (* If search is English, but no English docs are present in systemdocs, try a non-English lang, if present. See bug 299613 for the motivation for this feature. *)
                resolveDocResource[linkBase, context, resourceName, $fallbackNonEnglishLang, localFilesOnly, False],
            True,
                (* Not found. *)
                {Null, Null}
        ]
    ]                                
                                    

(* Generates and manages caching of three types of dynamically-generated notebooks:
   search results, About notebooks, and the InstalledAddOns notebook.
*)
generateNotebook[query_String, type_String] :=
    Module[{encodedQuery, nb},
        encodedQuery = ExternalService`EncodeString[query];
        Switch[type,
             "Search" | "ForceSearch",
                  loadDocSearchPackage[];
                  nb = Symbol["DocumentationSearch`ExportSearchResults"][Symbol["DocumentationSearch`SearchDocumentation"][query], "Notebook"];
                  $searchInitialized = True,
            "InstalledPacletsAndAddOns",
                (* type = "InstalledPacletsAndAddOns" is a request for the dynamically-generated page. *)
                nb = Documentation`MakeInstalledAddOnsNotebook[getInstalledPaclets[], True],
            "About",
                (* query holds the paclet name *)
                nb = makePacletAboutNotebook[query]
        ];
        cacheSearchResult[encodedQuery, nb, type]
    ]


(* Utility function to call NotebookLocate properly. *)
locateNotebook[{path_String, fragPart:(_String | None), fixedURL_String, searchHeaderData_}, feData_] :=
    Module[{nb, completionFunc = Null&, urlString, trimmedPath, searchText, searchTextTranslated},
        If[MatchQ[searchHeaderData, {_String, _String}],
            (* If searchHeaderData is a list, then this page was found via direct-hit search (e.g., user
               typed in Plot and got directed to ref/Plot). We want a link to appear in header
               that lets user force a search on this term if a direct hit is not what they
               wanted. This is done by setting special TaggingRules that are used in the
               help viewer header cell to create a "search for all results containing ..." line.
            *)
            {searchText, searchTextTranslated} = searchHeaderData;
            (* Trim whitespace from ends of searchText. This is so that "integrate " doesn't
               look like a different word than "Integrate".
            *)
            searchText = StringReplace[searchText, StartOfString ~~ WhitespaceCharacter... ~~
                                ShortestMatch[chars___] ~~ WhitespaceCharacter... ~~ EndOfString :> chars];
            If[ToLowerCase[searchText] === ToLowerCase[searchTextTranslated],
                searchText = ""
            ];
            With[{searchText = searchText, searchTextTranslated = searchTextTranslated},
                completionFunc = (
                    FEPrivate`Set[CurrentValue[#, {TaggingRules, "SearchText"}], searchText];
                    FEPrivate`Set[CurrentValue[#, {TaggingRules, "SearchTextTranslated"}], searchTextTranslated];
                )&
            ];
        ];
        Which[
            treatLikeURL[path],
                urlString =
                    If[StringMatchQ[path, "http:*" | "file:*", IgnoreCase->True],
                        path,
                    (* else *)
                        trimmedPath = If[StringMatchQ[path, ("/" ~~ ___) | ("\\" ~~ ___)], StringDrop[path, 1], path];
                        (* I don't use FileNameJoin here because I want to force "/" chars as separators. *)
                        "file:///" <>
                            StringJoin[
                                Riffle[ExternalService`EncodeString /@ FileNameSplit[trimmedPath], "/"]
                            ]
                    ] <> If[StringQ[fragPart] && fragPart != "", "#" <> fragPart, ""];
                NotebookLocate[{URL[urlString], ""}, FrontEnd`HistoryData -> {feData, fixedURL}, FrontEnd`ReturnNotebookObject->True],
            StringQ[fragPart] && fragPart != "" && DigitQ[fragPart],
                (* NotebookLocate cannot handle CellIDs (we assume a CellID is intended when
                     frag is all digits). Therefore must also use NotebookFind to move to CellID.
                *)
                nb = NotebookLocate[{path, None}, FrontEnd`HistoryData -> {feData, fixedURL},
                                FrontEnd`ReturnNotebookObject->True, FrontEnd`CompletionFunction -> completionFunc];
                If[Head[nb] === NotebookObject,
                    NotebookFind[nb, fragPart, Next, CellID, AutoScroll->Top];
                    (* Give focus to input field. *)
                    MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[nb, {{"HelpViewerSearchField"}},
                                    FE`BoxOffset -> {FE`BoxChild[1]}], FE`SearchStart -> "StartFromFirstDockedCell"]]
                ];
                nb,
            StringMatchQ[path, "*.nbp"],
                (* Special treatment for Player files: don't open as Documentation Center notebooks. *)
                NotebookLocate[{path, fragPart}],
            True,
                (* No fragment specifier, or frag was a cell tag, not a cell id. *)
                nb = NotebookLocate[{path, fragPart}, FrontEnd`HistoryData -> {feData, fixedURL},
                            FrontEnd`ReturnNotebookObject->True , FrontEnd`CompletionFunction -> completionFunc];
                If[Head[nb] === NotebookObject,
                    (* Give focus to input field. *)
                    MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[nb, {{"HelpViewerSearchField"}},
                                    FE`BoxOffset -> {FE`BoxChild[1]}], FE`SearchStart -> "StartFromFirstDockedCell"]]
                ];
                nb
        ]
    ]



(* Splits a resource into a LinkBase and resourceName components.
   Resource comes in as something like
        "ref/Partition"
        "tutorial/WorkingWithLists"
        "JLink/ref/InstallJava"
*)
splitResource[resource_String] :=
    Module[{linkBase, resourceName, slashPos},
        slashPos = Flatten[StringPosition[resource, "/"]];
        If[slashPos =!= {},
            linkBase = StringTake[resource, First[slashPos] - 1];
            If[MemberQ[$specialLinkBases, linkBase],
                (* Wasn't a LinkBase; rather, the LinkBase was missing and the first
                   part of the URI was a reserved word like ref, guide, etc.
                *)
                linkBase = "WolframMathematica";
                resourceName = resource,
            (* else *)
                resourceName = StringDrop[resource, First[slashPos]]
            ],
        (* else *)
            linkBase = resource;
            resourceName = ""
        ];
        {linkBase, resourceName}
    ]


looksLikePacletURI[s_String] :=
    StringMatchQ[s, "paclet:*"] ||
        (StringMatchQ[s, (WordCharacter.. ~~ "/").. ~~ (WordCharacter | "$" | "#" | "." | "-" | "_")..] &&
            (* Special-case handling for the slashes in J/Link and .NET/Link. *)
            ToLowerCase[s] =!= "j/link" && ToLowerCase[s] =!= "net/link" && ToLowerCase[s] =!= ".net/link"
        )



(* Treat everything that isn't a .m or .nb file as a URL. What this means in practice is that
   it will get wrapped in URL[]. The FE will launch these files with the viewer program that
   a web browser would.
*)
treatLikeURL[s_String] :=
    StringMatchQ[s, "http:*", IgnoreCase->True] || 
       !StringMatchQ[s, __ ~~ "." ~~ ("m" | "nb" | "nbp" | "cdf"), IgnoreCase->True]


initSearchCache[dir_String] :=
    (
        $searchCacheDir = dir;
        (* Deactivating messages because at this point in time you often get a DeleteFile failure
           because the FE still holds onto the file handle for some reason.
        *)
        Quiet[
            DeleteFile /@ FileNames["*.nb", $searchCacheDir]
        ]
    )


cacheSearchResult[encodedQuery_String, nb_, type_String] :=
    Module[{nbFile = ToFileName[$searchCacheDir, nbNameFromSearchQuery[encodedQuery, type]], oldNbs, strm},
        (* Keep max of 20 search results. Scan through cache and delete oldest files
           down to the point where at most 20 remain.
        *)
        oldNbs = Developer`FileInformation /@ FileNames["*.nb", $searchCacheDir];
        oldNbs = Sort[oldNbs, ((Date /. #1) > (Date /. #2))&];
        oldNbs = Drop[oldNbs, Min[20, Length[oldNbs]]];
        (* Deactivating messages because at this point in time you often get a DeleteFile failure
           because the FE still holds onto the file handle for some reason. When that changes,
           remove the Quiet.
        *)
        Quiet[
            DeleteFile /@ (File /. oldNbs)
        ];
        (* Pruning of cache finished. Save out the new result notebook. Avoid using Export
           (Export[nbFile, nb, "NB"]) because trial version has Export removed. Manually
           open stream with PageWidth->Infinity as a workaround for 84439.
        *)
        strm = OpenWrite[nbFile, PageWidth->Infinity];
        Put[nb, strm];
        Close[strm];
        nbFile
    ]


(* Query string is already URL-encoded. The only char that could still exist in it
   that isn't safe for filenames is *. Cap the filename at 80 chars in case the user had
   a large amount of text selected in the notebook and inadvertantly did a search.
   This means that if you type a legitimate 90 char search string and then search again
   after changing the last few chars it won't work--you'll get the same search result
   notebook from the cache.
   
   The id part gives unique names to otherwise identical search notebooks, which avoids
   problems with the old file being open in the front end while we try to write out a new one, 
   which happens on repeat searches.
*)
nbNameFromSearchQuery[encodedQuery_String, type_String] :=
    Module[{id = "_" <> ToString[Round[SessionTime[] * 100]]},
        Switch[type,
            "InstalledPacletsAndAddOns",
               "InstalledPacletsAndAddOns" <> id <> ".nb",
            "About",
               (* encodedQuery will be paclet name (possibly encoded, ofcourse) *)
               encodedQuery <> "_About" <> id <> ".nb",
            _,
               "SearchResults_" <>
                  StringReplace[StringTake[encodedQuery, Min[StringLength[encodedQuery], 80]], "*" -> "%2A"] <> id <> ".nb"
        ]
    ]


(* Worker function used only by Documentation`CreateMessageLink.
   Returns a paclet: URI that is known to resolve to a file that exists, or Null.
*)
createKnownGoodMessageURI[linkBase:(_String | All), context:(_String | All), resourceName_String, language_String] :=
    Module[{paclet, path, expandedResourceName, ext, extLinkBase},
        
        (* We are always doing either a linkBase-based lookup (most common) or a context-based lookup (only for message
           links), thus exactly one of linkBase and context must be a string.
        *)
        Assert[(StringQ[linkBase] || StringQ[context]) && !(StringQ[linkBase] && StringQ[context])];
        
        {paclet, path} = resolveDocResource[linkBase, context, resourceName, language, False];
        
        If[StringQ[path],
            (* We know that the paclet provides an existing doc file that can be looked up by the supplied
               linkbase/context and resourceName. We now need to return a paclet: URI that will map to this file.
               For the case of a linkBase-based lookup, this is trivial--it will be simply paclet:linkBase/resourceName.
               But for a context-based lookup, we need to go back and generate a URI (linkBase-based, by definition)
               that will map to this file.
            *)
            If[StringQ[linkBase],
                "paclet:" <> If[linkBase == "WolframMathematica", "", linkBase <> "/"] <> resourceName,
            (* else *)
                expandedResourceName = convertResourceNameToLongForm[resourceName];
                (* Try each linkable extension and see if a URI made out of its LinkBase will resolve to a file. *)
                doForEach[ext, cullExtensionsFor[PgetExtensions[paclet, $linkableExtensions], {"WolframVersion", "SystemID"}],
                    extLinkBase = EXTgetProperty[ext, "LinkBase", paclet["Name"]];
                    If[StringQ[PgetDocResourcePath[paclet, extLinkBase, Null, expandedResourceName, language]],
                        Return["paclet:" <> extLinkBase <> "/" <> resourceName]
                    ];
                    (* The resolveDocResource call above might have fallen back to an english doc if the requested
                       language was something else. In that case, the PgetDocResourcePath[..., "non-english language"]
                       call we just did won't have found a path (it doesn't fallback to trying English). Thus we need
                       to also try PgetDocResourcePath[..., "English"] to see if it works.
                    *)
                    If[language != "English",
                        If[StringQ[PgetDocResourcePath[paclet, extLinkBase, Null, expandedResourceName, "English"]],
                            Return["paclet:" <> extLinkBase <> "/" <> resourceName]
                        ]
                    ]
                ]
                (* Accept default Null return value of Scan (from doForEach) if not found. *)
            ]
        ]
    ]


If[!ValueQ[$hasLoadedDocSearch], $hasLoadedDocSearch = False]

loadDocSearchPackage[] := 
    If[!$hasLoadedDocSearch,
        $hasLoadedDocSearch = True;
        Block[{$ContextPath = {"System`"}}, Needs["DocumentationSearch`"]]
    ]


(*************************  convertResourceNameToLongForm  ****************************)

(* Tests: /PacletManagerTest.nb#123460 *)

(* Used in lookup of docs. Converts a short-form doc resource name like ref/Foo into its long form,
   like ReferencePages/Symbols/Foo. The short forms are only a convenience to users--everywhere within
   the PacletManager we use the long forms, and this is the function that does the conversion.

   Any new conversion rules we create need to be encoded here. An example would be a new category
   under ReferencePages that needs special conversion, like ref/foo/blort --> ReferencePages/FooBar/blort.
   Any new top-level categories that need short forms beyond guide, tutorial, ref, and note would also
   have to be added here.
*)
    
convertResourceNameToLongForm[resourceName_String] :=
    Module[{result, parts, first, subdir},
        result = resourceName;
        parts = StringSplit[resourceName, "/"];
        
        (* Only one part means no conversion. *)
        If[Length[parts] < 2,
            Return[result]
        ];
        
        first = First[parts];
        Which[
            StringMatchQ[first, "ref", IgnoreCase->True],
                If[Length[parts] == 2,
                    result = "ReferencePages/Symbols/" <> parts[[2]],
                (* else *)
                    subdir = parts[[2]];
                    (* Currently, in all categories other than 'message', there should
                       only be one element in the URI after the category (e.g., ref/format/foo
                       and not ref/format/foo/bar), so that parts[[3]] should be the last element
                       in parts. But there is no reason to enforce that in the code here, so
                       we allow any number of elements after the category; they are just
                       appended to the resource name: ReferencePages/CategoryName/Foo/Bar/Baz.
                    *)
                    Which[
                        StringMatchQ[subdir, "message", IgnoreCase->True],
                            result = "ReferencePages/Messages/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "character"],
                            result = "ReferencePages/Characters/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "format", IgnoreCase->True],
                            result = "ReferencePages/Formats/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "menuitem", IgnoreCase->True],
                            result = "ReferencePages/MenuItems/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "net", IgnoreCase->True],
                            result = "ReferencePages/NET/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "program", IgnoreCase->True],
                            result = "ReferencePages/Programs/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "frontendobject", IgnoreCase->True],
                            result = "ReferencePages/FrontEndObjects/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "applescript", IgnoreCase->True],
                            result = "ReferencePages/AppleScript/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "method", IgnoreCase->True],
                            result = "ReferencePages/Methods/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "file", IgnoreCase->True],
                            result = "ReferencePages/Files/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "service", IgnoreCase->True],
                            result = "ReferencePages/Services/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "indicator", IgnoreCase->True],
                            result = "ReferencePages/Indicators/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "device", IgnoreCase->True],
                            result = "ReferencePages/Devices/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "widget", IgnoreCase->True],
                            result = "ReferencePages/Widgets/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "callback", IgnoreCase->True],
                            result = "ReferencePages/Callbacks/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "interpreter", IgnoreCase->True],
                            result = "ReferencePages/Interpreters/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "embeddingformat", IgnoreCase->True],
                            result = "ReferencePages/EmbeddingFormats/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "mlmodel", IgnoreCase->True],
                            result = "ReferencePages/MLModels/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "classifier", IgnoreCase->True],
                            result = "ReferencePages/Classifiers/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "predictor", IgnoreCase->True],
                            result = "ReferencePages/Predictors/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "netencoder", IgnoreCase->True],
                            result = "ReferencePages/NetEncoders/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "netdecoder", IgnoreCase->True],
                            result = "ReferencePages/NetDecoders/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "externalevaluationsystem", IgnoreCase->True],
                            result = "ReferencePages/ExternalEvaluationSystems/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "entity", IgnoreCase->True],
                            result = "ReferencePages/Entities/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "compiledtype", IgnoreCase->True],
                            result = "ReferencePages/CompiledTypes/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "resourcetype", IgnoreCase->True],
                            result = "ReferencePages/ResourceTypes/" <> resourceFromParts[parts, 3],
                        StringMatchQ[subdir, "databaseconnection", IgnoreCase->True],
                            result = "ReferencePages/DatabaseConnections/" <> resourceFromParts[parts, 3],
                        StringLength[subdir] > 0,
                            (* Here we have default handling for other categories (e.g., c, java, etc.)
                               Just convert the first character to upper case and leave the rest unmodified.
                            *)
                            result = "ReferencePages/" <>
                                     ToUpperCase[StringTake[subdir, 1]] <> StringDrop[subdir, 1] <>
                                     "/" <> resourceFromParts[parts, 3];                  
                    ]
                ],
            StringMatchQ[first, "guide", IgnoreCase->True],
                result = "Guides/" <> resourceFromParts[parts, 2],
            StringMatchQ[first, "tutorial", IgnoreCase->True],
                result = "Tutorials/" <> resourceFromParts[parts, 2],
            StringMatchQ[first, "note", IgnoreCase->True],
                result = "Notes/" <> resourceFromParts[parts, 2],
            StringMatchQ[first, "howto", IgnoreCase->True],
                result = "HowTos/" <> resourceFromParts[parts, 2],
            StringMatchQ[first, "example", IgnoreCase->True],
                result = "ExamplePages/" <> resourceFromParts[parts, 2],
            StringMatchQ[first, "workflow", IgnoreCase->True],
                result = "Workflows/" <> resourceFromParts[parts, 2],
            StringMatchQ[first, "workflowguide", IgnoreCase->True],
                result = "WorkflowGuides/" <> resourceFromParts[parts, 2]
        ];
        
        result
    ]
        

(* Private utiity that builds a /-separated string from component strings. *)
resourceFromParts[parts:{___String}, startIndex_Integer] := StringJoin[Riffle[Drop[parts, startIndex-1], "/"]]



(********************************  getUpdatablePacletResource  ******************************)

(* Used by front end to look for updates to special files. *)

(* UNUSED ??? *)

getUpdatablePacletResource[pacletName_String, resourceName_String, fallback_String] :=
    Module[{resourcePath},
        Quiet[
            PacletUpdate[pacletName];
            resourcePath = PacletResource[pacletName, resourceName];
            If[StringQ[resourcePath], resourcePath, fallback]
        ]
    ]


(********************************  InstalledPacletsAndAddOns List  ****************************)

(*
    Called from root guide page to get info on paclets for dynamically-built guide page.
*)
getInstalledPaclets[] :=
    Module[{paclets, userPacletInformation},
        (* Find all "user" paclets. That is, not ones in the layout or downloaded systemdocs. These are paclets
           that have no official entry point already in the standard docs. The code here is doing
           "takeLatestEnabledVersionOfEachPaclet" except that it doesn't drop disabled ones.
        *)
        paclets = Cases[groupByNameAndSortByVersion[PCfindMatching["Collections"->{"User", "Legacy", "Extra"}]], {a_, ___} :> a];
        (* Only keep ones for which we know we can create a link that will resolve to a document. *)
        paclets = Select[paclets, StringQ[PgetMainLink[#]]&];
        (* Get rid of any paclets that are upgrades to ones found already in the layout. Any
           such paclets already have their docs woven into main system.
        *) 
        paclets = Select[paclets, PCfindMatching["Name"-> getPIValue[#, "Name"], "Collections"->{"Layout"}] === {} &];
        userPacletInformation =
            {"Name" -> getPIValue[#, "Name"],
             "Version" -> getPIValue[#, "Version"],
             "Description" -> getPIValue[#, "Description"],
             (* Convert thumbnail to full pathname. OK if it doesn't exist. *)
             "Thumbnail" -> ToFileName[PgetPathToRoot[#], getPIValue[#, "Thumbnail"]],
             "Creator" -> getPIValue[#, "Creator"],
             "Publisher" -> getPIValue[#, "Publisher"],
             "Paclet" -> True,
             "URI" -> PgetMainLink[#],
             "HomePage" -> getPIValue[#, "URL"],
             "PacletKey" -> PgetKey[#]
            }& /@ paclets;
        (* Sort by name in alphabetical order. *)
        SortBy[userPacletInformation, First]
    ]
    


(****  Functions called from InstalledPacletsAndAddOns page  ****)

(* Global used to force Dynamic reevals. *)
$loadButtonChangeTrigger = 0;


loadStateRadioFunction[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        $loadButtonChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            getLoadingState[paclet],
        (* else *)
            Manual   (* Shouldn't get here. *)
        ] // ToString
    ]

loadStateRadioFunction /: Set[loadStateRadioFunction[key_], val_] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            PacletSetLoading[paclet, val]
        ];
        $loadButtonChangeTrigger++;
        val
    ]

(* This deprecated one-arg def is never used, to my knowledge. *)
loadStateRadioEnabled[key:{pacletQualifiedName_String, pacletLocation_String}] := loadStateRadioEnabled[key, Null]
loadStateRadioEnabled[key:{pacletQualifiedName_String, pacletLocation_String}, stateToEnable_] :=
    Module[{paclet, kernelExts, ext},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet && TrueQ[isEnabled[paclet]],
            (* Decide which of the three buttons to enable ("Manual", Automatic, or "Startup"). At the moment, the only
               decision is for Automatic, and we enable iff there is a Kernel extension that lists Symbols, as that is required
               to do auto-loading.
            *)
            If[stateToEnable =!= Automatic,
                True,
            (* else *)
                kernelExts = cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID"}];
                If[Length[kernelExts] > 0,
                    Or @@ forEach[ext, kernelExts,
                        Length[EXTgetProperty[ext, "Symbols", {}]] > 0
                    ],
                (* else *)
                    False
                ]
            ],
        (* else *)
            False
        ]
    ]

loadStateRadioPresent[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            Length[cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID"}]] > 0,
            (* TODO: Should also include paclets that have FE resources (and perhaps other types) but no code. *)
        (* else *)
            False   (* Should not happen *)
        ]
    ]

loadButtonFunction[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            Get /@ PgetPreloadData[paclet]
        ];
        $loadButtonChangeTrigger++
    ]

(* Returns a string describing the state of the button (what it reads followed by whether it is enabled or not). *)
loadButtonState[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet, isLoaded},
        $loadButtonChangeTrigger;
        (* Hit this so that we can disable when the paclet is uninstalled (would be better if whole section
           on the InstalledPacletsAndAddOnsPage could disappear, but don't know if that's possible.
        *)
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            isLoaded = And @@ (isContextCurrentlyLoaded /@ PgetContexts[paclet]);
            Which[
                isLoaded,
                    "LoadedDisabled",
                TrueQ[isEnabled[paclet]],
                    "LoadNowEnabled",
                True,
                    "LoadNowDisabled"
            ]
        ]
    ]

uninstallButtonFunction[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{dialogResult, paclet},
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            dialogResult = ChoiceDialog["Are you sure you want to uninstall the paclet named \"" <>
                                    paclet["Name"] <> "\"? This will remove it from your computer.",
                                    WindowTitle -> "Wolfram Engine"
                        ];
            If[dialogResult,
                PacletUninstall[paclet];
                (* Reload the page. *)
                Documentation`HelpLookup["paclet:guide/InstalledAddOns"]
            ]
        ];
    ]


uninstallButtonPresent[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            !StringMatchQ[pacletLocation, $InstallationDirectory ~~ __],
        (* else *)
            False    (* Shouldn't happen *)
        ]
    ]


updateButtonFunction[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        Head[paclet] === Paclet
    ]

updateButtonEnabled[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            MatchQ[PacletCheckUpdate[paclet], _Paclet | {__Paclet}],
        (* else *)
            False    (* Shouldn't happen *)
        ]
    ]

disableCheckboxFunction[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            !TrueQ[isEnabled[paclet]],
        (* else *)
            False    (* Shouldn't happen *)
        ]
    ]

disableCheckboxFunction /: Set[disableCheckboxFunction[key_], val_] :=
    Module[{paclet},
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            If[TrueQ[val], PacletEnable[paclet], PacletDisable[paclet]]
        ];
        $pacletDataChangeTrigger++;
        !TrueQ[isEnabled[paclet]]
    ]

disableCheckboxPresent[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    If[True,
        (* Enable/Disble not supported right now. *)
        False,
    (* else *)
        (* Old code. Restore eventually. *)
        Module[{paclet, hasContexts},
            $pacletDataChangeTrigger;
            paclet = getPacletFromKey[key];
            If[Head[paclet] === Paclet,
                (* TODO: This next line is broken; there is no PgetContexts that takes those args. *)
                hasContexts = (ListQ[#] && Length[#] > 0)& @ PgetContexts[pacletQualifiedName, pacletLocation];
                hasContexts && !MemberQ[$Path, ParentDirectory[pacletLocation]],
                (* TODO: Should also include paclets that have FE resources (and perhaps other types) but no code. *)
            (* else *)
                False  (* Shouldn't happen *)
            ]
        ]
    ]

(* THESE UNUSED ??? *)
enabledStateRadioFunction[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            isEnabled[paclet],
        (* else *)
            False  (* Shouldn't get here. *)
        ]
    ]
enabledStateRadioFunction /: Set[enabledStateRadioFunction[key_], val_] :=
    Module[{paclet},
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            If[TrueQ[val], PacletEnable[paclet], PacletDisable[paclet]]
        ];
        $pacletDataChangeTrigger++;
        val
    ]
enabledStateRadioEnabled[key:{pacletQualifiedName_String, pacletLocation_String}] :=
    Module[{paclet},
        $pacletDataChangeTrigger;
        paclet = getPacletFromKey[key];
        If[Head[paclet] === Paclet,
            Length[cullExtensionsFor[PgetExtensions[paclet, "Kernel" | "Application"], {"WolframVersion", "SystemID"}]] > 0,
            (* TODO: Should also include paclets that have FE resources (and perhaps other types) but no code. *)
        (* else *)
            False   (* Should not happen *)
        ]
    ]



isContextCurrentlyLoaded[ctxt_String] :=
    Module[{names},
        If[!MemberQ[$ContextPath, ctxt],
            False,
        (* else *)
            names = Names[ctxt <> "*"];
            If[Length[names] > 0,
                (* A package that locks its symbols will generate a message on the call to OwnValues and return $Failed.
                   Thus, Quet the message and accept True or $Failed as an indicator that package has already been loaded.
                *)
                Quiet[MatchQ[Function[sym, FreeQ[OwnValues[sym], Package`ActivateLoad], HoldFirst] @@ ToHeldExpression[First[names]], True | $Failed]],
            (* else *)
                (* No names in the context? Must be some weird manual $ContextPath changes. *)
                False
            ]
        ]
    ]


(* Currently, these button functions on the InsatalledPacletsAndAddOns page are the only place where
   we need to go backwards from a key to a paclet. This might be more generally useful in the future.
   The current functionality is very inefficient, and would need to be improved for uses not so
   directly tied to the user interface.
   Returns either a Paclet expression or Null (it should generally not be Null, though, as the key almost
   always refers to a known paclet).
*)
getPacletFromKey[key:{qualifiedName_String, location_String}] :=
    Module[{paclet},
        paclet = PCfindMatching["Location" -> location];
        If[MatchQ[paclet, {_Paclet}],
            paclet = First[paclet],
        (* else *)
            Null  (* Shouldn't get here. *)
        ]
    ]
  
 
End[]
