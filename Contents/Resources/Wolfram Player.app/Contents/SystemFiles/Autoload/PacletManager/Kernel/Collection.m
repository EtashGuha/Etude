(* :Title: Paclet.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.1 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


Begin["`Package`"]

PCinitialize
PCwrite
PCrebuild

PCfindMatching
PCfindForDocResource


$extraPacletDirs


End[]  (* `Package` *)


(* Current context will be PacletManager`. *)

Begin["`Collection`Private`"]

$pacletsPersistentFile

$userPacletCollection
$layoutPacletCollection
$legacyPacletCollection
$extraPacletCollection
$cloudExtraPacletCollection (* A special collection in Wolfram Cloud kernels, for the WolframApplications area; empty list in desktop. *)


(* Change this when you modify the serialization format of pacletData.pmd2. *)
$serializationVersion = 3;

(*****************************  Initialization  ******************************)

(* This is intended to only be called once in a "normal" session. It's not an error to call it more than once, but this is
   not a function that gets called to implement something like RebuildPacletData[]. (It _is_ called to implement
   RestartPacletManager[].)
   Returns True/False to indicate whether the collection data had to be rebuilt.
*)

PCinitialize[freshStart:(True | False)] :=
    executionProtect @
    Module[{foundFile, strm, serVersion, pc, oldHash, newHash, oldCreationDate, lockFile, hadToRebuild},
        (***** Try to read from serialized file. *****)
        (* Because the persistent file depends on paclets in the layout, it must distinguish
           between different installation directories.
        *)
        $pacletsPersistentFile = ToFileName[$userConfigurationDir, "pacletData" <> "_" <>
                                              getKernelVersionString[] <> "_" <>
                                                ToString[Hash[$InstallationDirectory]] <> ".pmd2"];
        foundFile = FileExistsQ[$pacletsPersistentFile] && !freshStart;
        If[foundFile,
            lockFile = ToFileName[$userTemporaryDir, FileNameTake[$pacletsPersistentFile] <> ".lock"];
            (* Writes might take a while, so long wait time before going ahead and forcefully acquiring the lock. *)
            If[acquireLock[lockFile, 3, True],
                using[{strm = OpenRead[$pacletsPersistentFile]},
                    serVersion = Read[strm, Expression];
                    If[serVersion <= $serializationVersion,
                        (* $legacyPacletCollection and $extraPacletCollection are never serialized. *)
                        (* If we are in the lifetime of a BeginPackage when this is called, PacletManager` would not be visible,
                           and the symbol Paclet would be created in the current context when these files are read. 
                           Therefore, make sure that PacletManager` is visible.
                        *)
                        Block[{$ContextPath = Prepend[$ContextPath, "PacletManager`"]},
                            pc = Read[strm, Expression];
                            If[ListQ[pc], $userPacletCollection = pc];
                            pc = Read[strm, Expression];
                            If[ListQ[pc], $layoutPacletCollection = pc]
                        ];
                        (* Newer versions of the file have an extra expression at the end: a hash of the paclet names in the
                           repository. If this is different from the last time it was written, force a rebuild. This
                           allows us to pick up new paclets that were installed by other instances of Mathematica that
                           share this repository.
                        *)
                        If[serVersion > 1,
                            oldHash = Read[strm, Expression];
                            If[IntegerQ[oldHash],
                                newHash = Hash[FileNames["*", $userRepositoryDir]];
                                If[newHash =!= oldHash,
                                    (* Setting this to a non-list value forces a rebuild below. *)
                                    $userPacletCollection=.
                                ]
                            ]
                        ];
                        (* The transition from ser=2 to ser=3 brings another new expression at end: the Wolfram Language
                          $CreationDate. This catches cases where pacletdata is invalid because a paclet has been added
                          to a development version of M not associated with a change in the version number.
                        *)
                        If[serVersion > 2,
                            oldCreationDate = Read[strm, Expression];
                            (* Use N@DateList@ here because of a bug/feature in how DateObject is written/read from a stream. *)
                            If[oldCreationDate =!= N[DateList[$CreationDate]],
                                (* Setting this to a non-list value forces a rebuild below. *)
                                $userPacletCollection=.
                            ]
                        ]
                    ]
                ];
                releaseLock[lockFile]
            ]
        ];

        (* $extraPacletDirs does not survive a RestartPacletManager[]. *)
        $extraPacletDirs = {};
        If[!foundFile || !ListQ[$userPacletCollection] || !ListQ[$layoutPacletCollection],
            (* Will get here if fresh start, or no file found, or couldn't be opened, or had bad ser version, or had bad data. *)
            If[MemberQ[$CommandLine, "-layoutpaclets"],
                PCrebuild["Collections" -> {"Layout"}];
                $userPacletCollection = {},
            (* else *)
                PCrebuild["Collections" -> {"User", "Layout"}]
            ];
            PCwrite[];
            hadToRebuild = True
        ];

        (* legacy, extra, cloudextra collections are always built from PI files. *)
        If[MemberQ[$CommandLine, "-layoutpaclets"],
            PCrebuild["Collections" -> {"Extra", "CloudExtra"}];
            $legacyPacletCollection = {},
        (* else *)
            PCrebuild["Collections" -> {"Legacy", "Extra", "CloudExtra"}]
        ];

        initializeLayoutDocsCollection[ToFileName[{$InstallationDirectory, "Documentation"}]];

        (* Never get past here without these being meaningful exprs. *)
        Assert[ListQ[$userPacletCollection] && ListQ[$layoutPacletCollection] && ListQ[$legacyPacletCollection] && 
                  ListQ[$extraPacletCollection] && ListQ[$cloudExtraPacletCollection]];

        TrueQ[hadToRebuild]
    ]



(*********************************  PCwrite  *********************************)

PCwrite[] /; $pmMode =!= "ReadOnly" :=
    executionProtect @
    Module[{strm, lockFile, hash},
        lockFile = ToFileName[$userTemporaryDir, FileNameTake[$pacletsPersistentFile] <> ".lock"];
        hash = Hash[FileNames["*", $userRepositoryDir]];
        (* This might be a long write, so long wait time before going ahead and forcefully acquiring the lock. *)
        If[acquireLock[lockFile, 3, True],
            using[{strm = OpenWrite[$pacletsPersistentFile]},
                Write[strm, $serializationVersion];
                Write[strm, $userPacletCollection];
                Write[strm, $layoutPacletCollection];
                Write[strm, hash];
                (* Use N@DateList@ here because of a bug/feature in how DateObject is written/read from a stream. *)
                Write[strm, N[DateList[$CreationDate]]]
            ];
            releaseLock[lockFile]
        ]
    ]


(********************************  PCrebuild  ******************************)

Options[PCrebuild] = {"UserRepositoryDir" -> Automatic, "SharedRepositoryDir" -> Automatic, "SystemPacletDirs" -> Automatic,
                            "ApplicationDirs" -> Automatic, "Collections" -> All}

PCrebuild[OptionsPattern[]] :=
    executionProtect @
    Module[{userRepositoryDir, sharedRepositoryDir, sysPacletDirs, appDirs, collections, lockFile, result},
        log[1, "Rebuilding"];
        result = True;
        {userRepositoryDir, sharedRepositoryDir, sysPacletDirs, appDirs, collections} =
              OptionValue[{"UserRepositoryDir", "SharedRepositoryDir", "SystemPacletDirs", "ApplicationDirs", "Collections"}];
        collections = Flatten[{collections}];
        If[collections === {All} || MemberQ[collections, "User"],
            lockFile = ToFileName[$userTemporaryDir, "repository.lock"];
            (* Wait a second to get the lock, but walk away if it can't be acquired. Not that big a deal if we can't write
               the data. If the lock is being held, then some other process is probably writing it anyway. Don't bother even
               trying to acquire the lock in the cloud, as it will fail, but the paclet repo is read-only anyway.
            *)
            If[$CloudEvaluation || acquireLock[lockFile, 3, False],
                (* Use depth 3 to allow paclets like:
                       FooPaclet-1.0.0/
                           FooPaclet/
                               PacletInfo.m
                   in addition to:
                       FooPaclet-1.0.0/
                           PacletInfo.m
                               FooPaclet/
                *)
                $userPacletCollection = createPacletsFromParentDirs[{
                                                 If[userRepositoryDir === Automatic, $userRepositoryDir, userRepositoryDir],
                                                 If[sharedRepositoryDir === Automatic, $sharedRepositoryDir, sharedRepositoryDir]},
                                                 3
                                          ];
                releaseLock[lockFile],
            (* else *)
                (* We can get here in cases where the lockfile still exists (meaning another M process is reading/writing
                   to the repository), or, perhaps more commonly, when $UserBasePacletsDirectory is not visible (e.g.,
                   when the user's home dir is not visible).
                *)
                If[FileInformation[$UserBasePacletsDirectory] === {},
                    Message[RebuildPacletData::basedir, $UserBasePacletsDirectory],
                (* else *)
                    Message[RebuildPacletData::lock]
                ];
                (* Make sure not to leave this function with the collection undefined. *)
                If[!ListQ[$userPacletCollection], $userPacletCollection = {}];
                result = False
            ]
        ];
        If[collections === {All} || MemberQ[collections, "Layout"],
            $layoutPacletCollection = createPacletsFromParentDirs[
                                             If[sysPacletDirs === Automatic, systemPacletDirs[], sysPacletDirs],
                                             2
                                      ]
        ];
        If[collections === {All} || MemberQ[collections, "Legacy"],
            $legacyPacletCollection = createPacletsFromParentDirs[
                                             If[appDirs === Automatic, applicationDirs[], appDirs],
                                             2
                                      ]
        ];
        If[collections === {All} || MemberQ[collections, "Extra"],
            $extraPacletCollection = createPacletsFromParentDirs[$extraPacletDirs, 2]
        ];
        If[collections === {All} || MemberQ[collections, "CloudExtra"],
            $cloudExtraPacletCollection = If[TrueQ[$CloudEvaluation], createPacletsFromParentDirs["/wolframcloud/userfiles/WolframApplications", 2], {}]
        ];
        (* TODO: Sanity check, issue messages, etc. ? *)
        result
    ]


(**********************************  PCfindMatching  *************************************)

(*
     Find paclets in this collection that match specified criteria.
     Result is an unsorted list of Paclet[...].
*)

Options[PCfindMatching] =
    {"Name" -> All, "Version" -> All, "Qualifier" -> All, "SystemID" -> Automatic, "WolframVersion" -> Automatic, "ProductName" -> Automatic,
     "Context" -> All, "Extension" -> All, "Location" -> All, "Internal" -> All, "Creator" -> All, "Publisher" -> All,
     "Collections" -> {"User", "Layout", "Legacy", "Extra" (* The other value is "LayoutDocs" *)},
     "Language" -> All, (* Language is weird here, as it is not a top-level field in paclets. But for searches into
                          the layoutdocs collection, it might be useful. Not used by any internal PM features, though. *)
     "Paclets" -> Null}

PCfindMatching[OptionsPattern[]] :=
    Module[{pc, selectFunc, name, qualifier, version, systemID, mVersion, productName, context, extension,
              location, internal, creator, publisher, collections, paclets, result},
        {name, version, qualifier, systemID, mVersion, productName, context, extension,
           location, internal, creator, publisher, collections, paclets} =
               OptionValue[{"Name", "Version", "Qualifier", "SystemID", "WolframVersion", "ProductName", "Context", "Extension",
                              "Location", "Internal", "Creator", "Publisher", "Collections", "Paclets"}];
        selectFunc = Hold[];
        (* Put tests that cull heavily in speed-critical lookups at the start. This is so that short-circuit evaluation
           can prevent later tests from being called at all. A good example is context lookups--most paclets will fail
           this first test, and we therefore avoid testing their MVersion.
        *)
        If[context =!= All,
            (* PhasContext will search for a parent context, so if the requested context is Foo`Bar`, we will
               find paclets that announce Foo`.
            *)
            With[{context = context}, selectFunc = Join[selectFunc, Hold[PhasContext[#, context]]]]
        ];
        If[extension =!= All,
            With[{extension = Alternatives @@ Flatten[{extension}]}, selectFunc = Join[selectFunc, Hold[Length[PgetExtensions[#, extension]] > 0]]]
        ];
        If[name =!= All,
            With[{name = name}, selectFunc = Join[selectFunc, Hold[StringMatchQ[getPIValue[#, "Name"], name]]]]
        ];
        If[qualifier =!= All,
            With[{qualifier = qualifier}, selectFunc = Join[selectFunc, Hold[getPIValue[#, "Qualifier"] === qualifier]]]
        ];
        If[version =!= All && version =!= "",
            With[{version = version}, selectFunc = Join[selectFunc, Hold[StringMatchQ[getPIValue[#, "Version"], version]]]]
        ];
        If[systemID === Automatic, systemID = $SystemID];
        If[systemID =!= All,
            With[{systemID = systemID}, selectFunc = Join[selectFunc, Hold[systemIDMatches[systemID, getPIValue[#, "SystemID"]]]]]
        ];
        If[mVersion === Automatic, mVersion = getKernelVersionStringComplete[]];
        If[mVersion =!= All,
            With[{mVersion = mVersion}, selectFunc = Join[selectFunc, Hold[kernelVersionMatches[mVersion, getPIValue[#, "WolframVersion"]]]]]
        ];
        If[productName === Automatic, productName = getProductName[]];
        If[productName =!= All,
            With[{productName = productName}, selectFunc = Join[selectFunc, Hold[productNameMatches[productName, getPIValue[#, "ProductName"]]]]]
        ];
        If[internal =!= All,
            With[{internal = internal}, selectFunc = Join[selectFunc, Hold[getPIValue[#, "Internal"] === internal]]]
        ];
        If[location =!= All,
            With[{location = location}, selectFunc = Join[selectFunc, Hold[PgetLocation[#] === location]]]
        ];
        (* TODO: Other criteria for selection. *)
        (* It is an important, although undocumented, feature that $extraPacletCollection gets favored in the case of a version number tie.
           In other words, if PacletFind["Foo"] returns two Foo paclets with version 1.0.0, then the one from $extraPC will be listed first.
           This is useful during development and during the WRI build system, when developers might modify a paclet but not increment the
           version number. They typically use PacletDirectoryAdd (which goes to $extraPC) to make these updated paclets available to the
           system, and we want them to be found in preference. Because of the way paclet sorting happens downstream of this function,
           being listed last below means they will show up first in the final list. In other words, make sure $extraPC is added last here.
        *) 
        Which[
            ListQ[paclets],
                pc = paclets,
            collections === All,
                pc = Join[$userPacletCollection, $layoutPacletCollection, $legacyPacletCollection, $extraPacletCollection, $cloudExtraPacletCollection],
            True,
                pc = Join[
                        If[MemberQ[collections, "User"], $userPacletCollection, {}],
                        If[MemberQ[collections, "Layout"], $layoutPacletCollection, {}],
                        If[MemberQ[collections, "Legacy"], $legacyPacletCollection, {}],
                        If[MemberQ[collections, "Extra"], $extraPacletCollection, {}],
                        If[MemberQ[collections, "CloudExtra"], $cloudExtraPacletCollection, {}]
                     ]
        ];
        (* TODO: Don't I want to make older versions of a paclet completely shadowed by newer ones? *)
        result = Select[pc, Function[{heldSelectFunc}, Function[Null, And @@ heldSelectFunc] ] @ selectFunc];

        (* LayoutDocsCollection is quite different from others and must be handled separately. *)
        If[MemberQ[collections, "LayoutDocs"] && paclets === Null && context === All && qualifier === All &&
              kernelVersionMatches[mVersion] && productNameMatches[productName] && 
                 (extension === All || MemberQ[Flatten[{extension}], "Documentation"]) &&
                    internal =!= True && creator === All && publisher === All && location === All,
            result = result ~Join~ LDCfindMatching[name, version, OptionValue["Language"]]
        ];

        result
    ]


(*******************************  PCfindForDocResource  ********************************)

(* Returns a list of {paclet, docPath} of paclets that are known to contain an existing file
   corresponding to this resource. Disabled paclets have been dropped, and only the highest version number
   of each like-named paclet is represented. Empty list if no matches.

   Does not fall through to English if requested language is not found.

   TODO: Is there any good reason this returns a list instead of just the best one? (First member of the list).
   Perhaps when there are server paclets involved, a list will be useful for callers to have more info, but
   I think there will still be no reason not to pick just one here.
*)

Options[PCfindForDocResource] =
    {"LinkBase" -> All, "Context" -> All, "ResourceName" -> All, "Language" -> "English"}

PCfindForDocResource[opts:OptionsPattern[]] :=
    Module[{linkBase, context, expandedResourceName, language},

        {linkBase, context, expandedResourceName, language} =
             OptionValue[{"LinkBase", "Context", "ResourceName", "Language"}];

        Assert[MatchQ[linkBase, _String | All]];  (* Typically, "WolframMathematica" *)
        Assert[MatchQ[context, _String | All]];   (* All, unless we are looking up a message URI. *)
        Assert[MatchQ[expandedResourceName, _String | All]];  (* e.g. "Guides/foo", or All if looking for a root guide page. *)
        Assert[MatchQ[language, _String]];  (* Desired language must be a string, never All *)
        (* Lookups are either linkbase-based or context-based. *)
        Assert[linkBase === All || context === All];

        (* Search among all collections.

           Note that we do no checking here to first decide if a paclet has a chance of being found in any
           given collection--we just blindly propagate the find request to all collections. Leave
           all optimization logic to the collections themselves.
        *)
        Join[
            LDCfindForDocResource[linkBase, context, expandedResourceName, language],
            MCfindForDocResource[$layoutPacletCollection, linkBase, context, expandedResourceName, language],
            MCfindForDocResource[$userPacletCollection, linkBase, context, expandedResourceName, language],
            MCfindForDocResource[$legacyPacletCollection, linkBase, context, expandedResourceName, language],
            MCfindForDocResource[$extraPacletCollection, linkBase, context, expandedResourceName, language],
            MCfindForDocResource[$cloudExtraPacletCollection, linkBase, context, expandedResourceName, language]
        ]
        (* Drop disabled and sort by version number. *)
        // Select[#, isEnabled[First[#]]&]&
        // Sort[#, versionGreater[getPIValue[First[#1], "Version"], getPIValue[First[#2], "Version"]]&]&
    ]


End[]

