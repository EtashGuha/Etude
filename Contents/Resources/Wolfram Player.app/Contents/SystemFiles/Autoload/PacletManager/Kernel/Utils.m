(* :Title: Utils.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 6.0 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


PacletNewerQ::usage = "PacletNewerQ is an internal symbol."


Begin["`Package`"]

log
$LogLevel

BuildPacletSiteFiles

versionGreater
versionCompare

getKernelVersionStringComplete
getKernelVersionString

getProductName

kernelVersionMatches
systemIDMatches
languageMatches
productNameMatches

sortPacletsByVersion
groupByNameAndSortByVersion
takeLatestEnabledVersionOfEachPaclet

cullExtensionsFor

acquireLock
releaseLock

getTaskData
setTaskData
freeTaskData

hasFrontEnd
hasLocalFrontEnd
isSubKernel

catchSystemException

differenceInDays

isNetworkSuccess

using
executionProtect
try
forEach
doForEach


End[]  (* `Package` *)



(* Current context will be PacletManager`. *)

Begin["`Utils`Private`"]


(**************************  Sorting and version comparison  ***************************)

(* Tests that the two paclets are the "same" (same name and qualifier), as well as the version number compare. 
   It's not obvious that this is the right thing to do. Perhaps just test the version, and let callers be responsible
   for deciding if two paclets are worth comparing?
*)
PacletNewerQ[p1_Paclet, p2_Paclet] :=
    getPIValue[p1, "Name"] == getPIValue[p2, "Name"] && 
        getPIValue[p1, "Qualifier"] == getPIValue[p2, "Qualifier"] && 
            versionGreater[p1["Version"], p2["Version"]]
(* Also allow direct version number comparison:*)
PacletNewerQ[v1_String, v2_String] := versionGreater[v1, v2]


(* Tests: /PacletManagerTest.nb#123461 *)

(* Sorts with highest version at front of list. *)

sortPacletsByVersion[paclets_List] :=
    Sort[paclets, versionGreater[getPIValue[#1, "Version"], getPIValue[#2, "Version"]]&]


(* Tests: /PacletManagerTest.nb#123462 *)

groupByNameAndSortByVersion[paclets_List] :=
    sortPacletsByVersion /@ GatherBy[paclets, getPIValue[#, "Name"]&]
    

(* Tests: /PacletManagerTest.nb#123463 *)

takeLatestEnabledVersionOfEachPaclet[paclets_List] :=
    Cases[groupByNameAndSortByVersion[dropDisabledPaclets[paclets]], {a_, ___} :> a]


(* Tests: /PacletManagerTest.nb#123464 *)

cullExtensionsFor[exts_List, properties:{__String}] :=
    Module[{checkSystemID, checkLanguage, checkMVersion, checkProductName},
        checkSystemID = MemberQ[properties, "SystemID"];
        checkLanguage = MemberQ[properties, "Language"];
        checkMVersion = MemberQ[properties, "WolframVersion"];
        checkProductName = MemberQ[properties, "ProductName"];
        Select[exts, (!checkMVersion || kernelVersionMatches[EXTgetProperty[#, "WolframVersion", "*"]]) &&
                     (!checkSystemID || MemberQ[Flatten[{EXTgetProperty[#, "SystemID", $SystemID]}], $SystemID]) &&
                     (!checkLanguage || languageMatches[EXTgetProperty[#, "Language", $Language]]) &&
                     (!checkProductName || productNameMatches[EXTgetProperty[#, "ProductName", All]]) &]
    ]
    

(* Tests: /PacletManagerTest.nb#123458 *)

(* versionGreater compares two version numbers and returns True if the first is
   greater (newer) than the second. False if equal or second is greater.
   Note that this provides comparisons for up to 5 digit blocks: 1.2.3.4.5.
*)
versionGreater[v1_String, v2_String] :=
    Not[
        OrderedQ[
            {PadRight[ToExpression[StringSplit[v1, "."]], 5], PadRight[ToExpression[StringSplit[v2, "."]], 5]}
        ]
    ]

(* Returns True, False, or Equal. PacletNewerQ and versionGreater don't work for the common need
   to know if they are equal, so this function is provided as a lower-level utility.
*)
versionCompare[v1_String, v2_String] :=
    Which[
        versionGreater[v1, v2],
            True,
        versionGreater[v2, v1],
            False,
        True,
            Equal
    ]


(******************************  getKernelVersionString  *****************************)

(* Tests: /PacletManagerTest.nb#123456 *)

(* Caches the value so it is computed only once. *)
getKernelVersionString[] := getKernelVersionString[] =
    StringJoin[
        If[StringMatchQ[#, "*."], # <> "0", #]& [ToString[$VersionNumber]],
        ".",
        ToString[$ReleaseNumber],
        ".",
        ToString[$MinorReleaseNumber]
    ];

getKernelVersionStringComplete[] := getKernelVersionStringComplete[] = 
    getKernelVersionString[] <> "." <> ToString[FromDigits[StringReverse[$PatchLevelID], 2]]


(*****************************  kernelVersionMatches  ********************************)

(* Tests: /PacletManagerTest.nb#123457 *)

(* Tells whether the given kernel version is compatible with the version spec given by a paclet.
   This is the only function that deals with versionSpec patterns like 8.1.* and 8+.
   This is a surpisingly expensive method, and since the vast majority of paclets will have one
   of a small class of kernel version specs (e.g., "8.0+"), we cache the results for
   quick lookup.

   kernelVersion is something like "8.1.0.0", but it could have up to 5 digit fields.
   versionSpec typically comes directly from a PacletInfo.m file. Could be "8+", "8.0.0.0", "8.1.*",
   "8.1.*,8.2.1", etc.
*)
kernelVersionMatches[versionSpec_String] := kernelVersionMatches[getKernelVersionStringComplete[], versionSpec]

kernelVersionMatches[kernelVersion_String, All | Null] = True

kernelVersionMatches[kernelVersion_String, versionSpec_String] := 
    Block[{key},  (* Block for speed only *)
        (* lookup in cache *)
        key = kernelVersion <> "?" <> versionSpec;
        If[# === True || # === False, Return[#]]& [$kernelVersionCache[key]];
        storeInCache[kernelVersion, versionSpec, key]
    ]
    
(* Private worker function. *)
storeInCache[kernelVersion_String, versionSpec_String, key_String] :=
    Block[{matches, components, specComponents, kernelComponents},  (* Block for speed only *)
        matches = False;
        (* If spec has comma-separated components, map over them. *)
        components = StringSplit[versionSpec, ","];
        If[Length[components] > 1,
            If[Or @@ (kernelVersionMatches[kernelVersion, StringReplace[#, " " -> ""]]& /@ components),
                matches = True
            ],
        (* else *)
            Which[
                StringMatchQ[versionSpec, "*+"],
                    (*  + can only come at end, and it cannot be used along with *. *)
                    matches = versionCompare[kernelVersion, StringDrop[versionSpec, -1]] =!= False,
                StringMatchQ[versionSpec, "*-"],
                    (*  - can only come at end, and it cannot be used along with *. *)
                    matches = versionCompare[kernelVersion, StringDrop[versionSpec, -1]] =!= True,
                True,
                    (* Does not end in +. Split at periods and compare each segment. Version specs can
                       be considered to end with an arbitrarily-long tail of "*" elements, and version numbers
                       can be considered to end with an arbitrarily long tail of "0" elements.
                    *)
                    specComponents = PadRight[StringSplit[versionSpec, "."], 5, "*"];
                    kernelComponents = PadRight[StringSplit[kernelVersion, "."], 5, "0"];
                    matches = And @@ Transpose[{kernelComponents, specComponents}] /. 
                                    {{_, "*"} -> True,
                                     {x_String, y_String} :> (ToExpression[x] === ToExpression[y])}
            ]
        ];
        $kernelVersionCache[key] = matches
    ]
    
    
(* Gets downvalues added to it. *)
Clear[$kernelVersionCache]
$kernelVersionCache[_] = Null


(*******************************  systemIDMatches  **********************************)

(* Tests: /PacletManagerTest.nb#123459 *)

(* Tells whether the given SystemID is compatible with the sysID specification given by a paclet.
   If called with one argument, it uses the current $SystemID value.
*)
systemIDMatches[sysIDSpec:(_String | {___String})] := systemIDMatches[$SystemID, sysIDSpec]

systemIDMatches[All | Null] = True
systemIDMatches[sysID_String, All | Null] = True

systemIDMatches[sysID_String, sysIDSpec:_String] := sysID == sysIDSpec
systemIDMatches[sysID_String, sysIDSpec:{___String}] := MemberQ[sysIDSpec, sysID]


(*******************************  languageMatches  **********************************)

(* Tests: /PacletManagerTest.nb#123470 *)

(* Tells whether the given language is compatible with the Language specification given by a paclet.
   If called with one argument, it uses the current $Language value.
*)
languageMatches[langSpec:(_String | {___String})] := languageMatches[$Language, langSpec]

languageMatches[All | Null] = True

languageMatches[lang_String, All | Null] = True
languageMatches[lang_String, langSpec_String] := lang == langSpec
languageMatches[lang_String, langSpec:{___String}] := MemberQ[langSpec, lang]


(*******************************  productNameMatches  **********************************)

(* Tells whether the current ProductIDName is compatible with the ProductName specification given by a paclet.
   If called with one argument, it uses the current ProductIDName value.
*)
$currentProductName = Lookup[$ProductInformation, "ProductIDName"]
getProductName[] = $currentProductName

productNameMatches[All | Null] = True
productNameMatches[productNameSpec_] := productNameMatches[$currentProductName, productNameSpec]

productNameMatches[pname_String, All | Null] = True
productNameMatches[pname_String, productNameSpec_String] := pname == productNameSpec
productNameMatches[pname_String, productNameSpec:{___String}] := MemberQ[productNameSpec, pname]
(* This next rule is for an arbitrary pattern, like Except["WolframFinancePlatform"] *)
productNameMatches[pname_String, productNameSpec_] := MatchQ[pname, productNameSpec]


(********************************  log  ***********************************)

(* Simple logging scheme intended mainly for debugging. Set PacletManager`Package`$LogLevel
   to an integer value to turn on logging. The first arg in the call to log is the
   "log level". If it is <= $LogLevel the message gets logged. Use the following as
   a general guide for log levels in calls to log:

   1   Information that might conceivably be useful to users. That is, assume that
       it isn't out of the question that a user might want to turn on logging at this level.

   2   This information is only for debugging. This would be a typical choice for $LogLevel
       if a developer was trying to follow what was happening internally.

   3   Verbose tracing-type information. A developer wants to know as much as possible
       about what was going on.
*)

(* So args are not evaled if they are not going to be logged. *)
SetAttributes[log, HoldRest]

log[level_Integer, args___] :=
    If[level <= $LogLevel, Print[args]]


(****************************************************************************)


(** Utility funcs for creation of a paclet site. These are not public. *)

Options[BuildPacletSiteFiles] = {Verbose->False, "WRI"->False, "MathematicaVersion"->""}

(* Call this on the top-level paclet site dir (the dir in which the PacletSite.m file will
   appear; this dir will have a Paclets subdir holding a flat or subdir structure of .paclet files.

    For example:

    BuildPacletSiteFile["d:\\webserver\\tomcat5\\webapps\\pacletserver\\testsite"]

    The fileList argument can be used to pass in the list of .paclet files to operate on. Passing {} means "use all".
*)
BuildPacletSiteFiles[dir_String, fileList:_List:{}, OptionsPattern[]] :=
    Module[{files, vbose, wri, mVersionSuffix},
        {vbose, wri, mVersionSuffix} = OptionValue[{Verbose, "WRI", "MathematicaVersion"}];
        files = If[Length[fileList] == 0, FileNames["*.paclet" | "*.cdf", ToFileName[dir, "Paclets"], Infinity], fileList];
        (* Cull out files that are not valid paclets. *)
        files = Select[files, If[VerifyPaclet[#], True, Print["***** File " <> # <> " is not a valid paclet. It will be excluded."]; False]&];
        buildSiteFiles[files, dir, "", vbose];
        (* On WRI server we have several variants of the PacletSite.m/mz files.
           Special server logic decides which to serve to clients. This keeps download sizes
           down and reduces memory and time on the client.
           Third-party paclet servers will not have such logic, so for those the code just
           lumps all paclets into a single PacletSite.m/mz set.

           For WRI servers, we always create the following:

               PacletSite.m/mz          contains all paclets

               PacletSite_x.x.x.m/mz    contains all paclets, trimmed for relevance to the given version
                                        For these version-specific files, each is relevant to all versions of
                                           Mathematica up to the next version for which a named file exists.
           
           Note that this version of the PM can only build a PacletSite_9x.m or later, as it creates version-specific files
           in a format that earlier versions of the PM cannot handle. To get PacletSite_8x.m, etc., you must use each
           specific version of M and its PM.                      
        *)
        If[wri && StringLength[mVersionSuffix] > 0,
            buildSiteFiles[files, dir, "_" <> mVersionSuffix, vbose]
        ]
    ]

(* Worker function. *)
buildSiteFiles[files:{___String}, outDir_String, mVersionSuffix_String, verbose:(True | False)] :=
    Module[{pacletFile, siteFile, strm, pacletWritingFunction},
        If[verbose, Print["Number of paclets placed into site file: ", Length[files]]];
        siteFile = ToFileName[outDir, "PacletSite" <> mVersionSuffix <> ".m"];
        (* We only use the mVersionSuffix for 9 and later PacletSite files. *)
        pacletWritingFunction = If[StringLength[mVersionSuffix] > 0, writePaclet9, writePacletCompatible];
        strm = OpenWrite[siteFile, CharacterEncoding->"UTF8"];
        WriteString[strm, "PacletSite["];
        pacletWritingFunction[strm, First[files]];
        doForEach[pacletFile, Rest[files],
            WriteString[strm, ","];
            pacletWritingFunction[strm, pacletFile]
        ];
        WriteString[strm, "]"];
        Close[strm];
        compressPacletSiteFile[siteFile];
    ]

(* Writes the contents of PacletSite.m in a format that is more verbose, but can be read by any version of M. *)
writePacletCompatible[strm_OutputStream, pacletFile_String] :=
    Module[{dataBuffer, piString, pacletStartPos},
        dataBuffer = Quiet[ZipGetFile[pacletFile, "PacletInfo.m"]];
        If[!ListQ[dataBuffer],
            dataBuffer = ZipGetFile[pacletFile, "PacletInfo.wl"]
        ];
        If[ListQ[dataBuffer],
            piString = FromCharacterCode[dataBuffer, "UTF8"];
            (* Trim comments from beginning (e.g., the ones typically placed there by Workbench). *)
            pacletStartPos = StringPosition[piString, "Paclet["];
            If[Length[pacletStartPos] == 1,
                (* PI.m file looks as we expect, with only one Paclet[ in it. Eliminate any junk at the beginning
                   the efficient, easy way.
                *)
                piString = StringDrop[piString, pacletStartPos[[1,1]] - 1],
            (* else *)
                (* PI.m file is unusual. Unlikely. Perhaps Paclet[ appears in a comment at the beginning, or there is a
                   space or linefeed between the t and the [. Use a slower way to eliminate any junk at the beginning.
                *)
                piString = StringReplace[piString, __ ~~ rest:("Paclet[" ~~ __ ~~ EndOfString) :> rest];
            ];           
            piString = StringReplace[piString, WhitespaceCharacter.. -> " "];
            WriteString[strm, piString],
        (* else *)
            (* This should never happen, as all paclets have passed VerifyPaclet step earlier. *)
            Print["WARNING: Bad paclet: ", pacletFile, ". It will be excluded"]
        ]
    ]

(* Writes the contents of PacletSite.m in a format that is much less verbose, but can only be read by M9 or later.
   The compatibility issue is that these paclets will be written out with strings instead of symbols for the LHS of rules
   (e.g., "Name"->"Foo" instead of Name->"Foo"), which is the in-memory representation of a Paclet expr. The
   motivation for a different way of writing PacletSite_9.m is that we can drop any content that is in the Paclet
   expression but not needed on the client. The important (and only, atm) case is the huge Resources sections of
   some data paclets, which can make up half the PacletSite.m file. Not only does this take download time, but site
   paclets are stored in memory in M, so there is a big memory cost.
*)
writePaclet9[strm_OutputStream, pacletFile_String] :=
    Module[{p},
        p = CreatePaclet[pacletFile];
        (* We have previously verified that p is a legit paclet, but test here again. *)
        If[Head[p] === Paclet,
            (* Get rid of any content that is not a rule (nothing like this present atm), and also Location field. *)
            p = DeleteCases[p, Except[_Rule] | ("Location"->_)];
            (* Get rid of the potentially large Resources sections. Of course this means that the client cannot see
               them, and thus cannot make any decisions about whether to download a paclet based on this info.
               That feature is never used at present.
            *)
            p = DeleteCases[p, "Resources"->_, Infinity];
            WriteString[strm, ToString[p, InputForm]],
        (* else *)
            (* This should never happen, as all paclets have passed VerifyPaclet step earlier. *)
            Print["WARNING: Bad paclet: ", pacletFile, ". It will be excluded"]
        ]
    ]


(* Compresses the given PacletSite.m file into PacletSite.mz in the same directory.
   PacletSite.m must be specified as a full path.
   Called automatically by BuildPacletSiteFile, so probably rarely used directly.
*)
compressPacletSiteFile[file_String] :=
    (
        Needs["JLink`"];
        JLink`InstallJava[];
        JLink`JavaBlock[
            Module[{fis, numRead, buf, zipStrm},
                fis = JLink`JavaNew["java.io.FileInputStream", file];
                zipStrm = JLink`JavaNew["java.util.zip.ZipOutputStream",
                            JLink`JavaNew["java.io.FileOutputStream", file <> "z"]];
                (* In a compressed .mz file, the entry is always called PacletSite.m, even though the
                   actual filename might have a language suffix: PacletSite_Japanese.mz.
                *)
                zipStrm@putNextEntry[JLink`JavaNew["java.util.zip.ZipEntry", "PacletSite.m"]];
                buf = JLink`JavaNew["[B", 100000];
                While[(numRead = fis@read[buf]) >= 0,
                    zipStrm@write[buf, 0, numRead]
                ];
                zipStrm@closeEntry[];
                zipStrm@close[];
                fis@close[];
            ]
        ]
    )


(*****************************  File locking  *****************************)

(* We need to avoid multiple instances of M on the same machine (like parallel kernels) from clobbering
   each other on writes and reads to various PM files. We also need to protect against a preemptive
   computation from interrupting a paclet read/write operation and performing another paclet operation.
   There are configuration files like managerData and pacletData, and the entire Repository directory itself.
   
   The locking scheme involves the use of a .lock file that guards a specific resource. Acquiring the lock
   corresponds to creating the file, and releasing it corresponds to deleting the file. The presence of the file
   is thus the indicator that aome other code is in the act of reading or writing the resource. This is not bulletproof,
   since there is a (very brief) period of time between testing for the existence of a file and then creating it. 
   
   There are 2 types of behavior:
      - wait for up to a certain amout of time trying to acquire, then give up
      - wait for up to a certain amout of time trying to acquire, then force the acquire to succeed
      
   Allowing the second, "force", behavior might seem a strange choice--we are trying to respect the integrity of
   our paclets and configuration files by enforcing a lock/unlock discipline, but if a caller requests a lock and
   finds it busy, it can just delete the lock file and forge ahead, re-creating the lock and reading/writing
   the files. The reason for this is that we want to absolutely avoid deadlock on a leftover lock file. For most uses,
   the configuration file read/write is very fast, and the lock file will only exist for a short period of time. If you
   end up waiting, say, 0.5 seconds to read a config file, then it's probably the case that the lock file you are
   looking at is orphaned.
   
   If acquireLock returns True, you _must_ call releaseLock.
*)

acquireLock[lockFile_String, timeout_, force_] :=
    Quiet @
    executionProtect @
    Module[{doAcquire, loopsRemaining, strm},
        doAcquire = True;
        If[FileExistsQ[lockFile],
            (* If the lock file exists, we wait for at most the timeout period. *)
            loopsRemaining = Quotient[timeout, 0.099];
            While[FileExistsQ[lockFile] && loopsRemaining-- > 0,
                Pause[0.1]
            ];
            If[FileExistsQ[lockFile],
                If[force,
                    DeleteFile[lockFile],
                (* else *)
                    (* This is the only branch that makes us not proceed with acquiring the lock. We waited for the
                       specified timeout and the lock file was still present, and the force parameter was False.
                    *)
                    doAcquire = False
                ]
            ]
        ];
        If[doAcquire,
            strm = OpenWrite[lockFile];
            If[Head[strm] === OutputStream,
                Close[strm];
                True,
            (* else *)
                False
            ],
        (* else *)
            False
        ]
    ]


releaseLock[lockFile_String] := Quiet @ executionProtect @ DeleteFile[lockFile]


(********************************  get/setTaskData  **********************************)

(* For HTTP download async tasks in a couple different places we need to associate data with the task,
   and also update this data as the task progresses (e.g., status code, progress percentage, etc.)
   It would be nice to be able to do this with the Options/SetOptions mechanism on the "UserData" option
   of URLSaveAsynchronous. Unfortunately, this won't work since I need the data to be queryable even after
   the async task is finished. I might want to change this scheme in the future, so I encapsulate it here.
   
   task[[2]] is the task id.
*)

(*
getTaskData[task_] := ReleaseHold[Options[task, "UserData"]]

setTaskData[task_, data_] := Function[{sym}, sym = data, HoldFirst] @@ Options[task, "UserData"]
*)
getTaskData[task_] := $taskData[task[[2]]]

setTaskData[task_, data_] := $taskData[task[[2]]] = data

freeTaskData[task_] := $taskData[task[[2]]] =.


(********************************  Utilities  **********************************)

(* Test as a string to avoid the symbol FrontEndObject forcing loading of System`FEDump, etc. *)
hasFrontEnd[] := ToString[Head[$FrontEnd]] === "FrontEndObject"

hasLocalFrontEnd[] := hasFrontEnd[] && MathLink`CallFrontEnd[FrontEnd`Value["$MachineID"]] === $MachineID

(* Evaluating $KernelID will trigger autoloading of Parallel packages, so only do it if Parallel` already loaded. *)
isSubKernel[] = MemberQ[$Packages, "Parallel`"] && IntegerQ[$KernelID] && $KernelID =!= 0


(* The plugin (any time Developer`$ProtectedMode is True) will cause SystemExceptino to be thrown
   in illegal operations (like writes to disallowed dirs). Sometiems we want to try operations and
   quietly fail if they are illegal instead of checking for protected mode and then bailing out
   ahead of time. The former strategy will silently allow code that was once illegal to function
   if restrictions are eased.
   catchSystemException will return $Failed if a SystemException was thrown.
*)
SetAttributes[catchSystemException, {HoldFirst}]

catchSystemException[expr_] :=
    Catch[
        Quiet[expr, General::sandbox],
        _SystemException,
        $Failed&
    ]
    
 
(* In case you are wondering why this function isn't just DateDifference (which it was in M9 and earlier), it is because a
   backward-incompatible change is being contemplated in that function, so I want to avoid using it.
*)
differenceInDays[d1_List, d2_List] := (AbsoluteTime[d2] - AbsoluteTime[d1])/86400

isNetworkSuccess[url_, statusCode_] :=
    Which[
        StringMatchQ[url, "http*"],
            statusCode === 200,
        StringMatchQ[url, "file:*"],
            statusCode === 0,
        StringMatchQ[url, "ftp:*"],
            200 <= statusCode <= 300,
        True,
            (* Probably can never even get here on an unknown protocol, and it probably
               doesn't matter whether we return True or False.
            *)
            True
    ]
    
    
(********************************  using  **********************************)

(* using is a utility that emulates the using keyword in C# and similar features in other languages.
   The idea is that you create/open a resource, and then it is guaranteed to be closed/released/etc.
   when the using function ends. The $usingActionsMap associates a cleanup function with each head.
   Currently, it only supports streams and Unprotect. For example:
   
   using[{strm = OpenRead["foo"], Unprotect["Power"]},
       ...
   ]
*)

(* Tests: /PacletManagerTest.nb#123480 *)

SetAttributes[using, HoldAll]

using[vars_List, expr_] :=
    Module[{cleanup},
        Internal`WithLocalSettings[
            cleanup = Flatten[vars],
            expr,
            Scan[(Head[#] /. $usingActionsMap)[#]&, cleanup]
        ]
    ]

$usingActionsMap = {
    InputStream -> Close,
    OutputStream -> Close,
    String -> Protect,
    _ -> (Null&)
}


(**********************************  try  ************************************)

(* Because of bug 214499, we need special handling of aborts if we want to ensure correct behavior
   for arbitrary 'finally' clauses. The 'using' function above does not need to be written this way because
   it has a specialized cleanup clause.
*)

(* Tests: /PacletManagerTest.nb#123481 *)

SetAttributes[try, HoldAll]

try[expr_, finally_] :=
    Block[{wasAborted = False, result},
        Internal`WithLocalSettings[
            Null,
            result = CheckAbort[expr, wasAborted = True],
            finally
        ];
        If[wasAborted, Abort[]];
        result
    ]


(***************************  executionProtect  *******************************)

SetAttributes[executionProtect, {HoldFirst}]

executionProtect[e_] := AbortProtect[PreemptProtect[e]]


(********************************  forEach  **********************************)

(* Loop like: forEach[var, expr, functionInvolvingVar]

   foEach and doForEach make some types of Map/Scan operations more readable by putting the
   expression before the function. If the function is long, Map/Scan require the reader to
   skim the whole body to find what is probably the most meaningful part--the expression being operated on.
   It also lets you write the function in non-pure function form, which is easier to read, and without having
   to write the clumsy Function[{x}, ...].
   They work if the expression is an atom: forEach[x, 42, Print[x]].
   This is similar to Java/C# notation: "for each x in expr do the following".
 *)

(* Tests: /PacletManagerTest.nb#123482 *)

SetAttributes[forEach, HoldAll]
SetAttributes[doForEach, HoldAll]

(* Using With prevents expr from being evaulated twice. The Function @@ Hold[sym, f] instead of just Function[sym, f] is to
   avoid problems that occur when using SetSystemOptions["StrictLexicalScoping" -> False].
*)

forEach[sym_Symbol, expr_, f_] := With[{e = expr}, (Function @@ Hold[sym, f]) /@ If[AtomQ[e], {e}, e]]

(* doForEach allows use of Return[] in f, because it uses Scan instead of Map. *)

doForEach[sym_Symbol, expr_, f_] := With[{e = expr}, (Function @@ Hold[sym, f]) ~Scan~ If[AtomQ[e], {e}, e]]


End[]
