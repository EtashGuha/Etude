(* :Title: Services.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 6.0 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


$PacletSite::usage = "$PacletSite gives the location of the main Wolfram Research paclet server."

PacletSite::usage = "PacletSite is an internal symbol."

PacletSites::usage = "PacletSites[] gives the list of currently-enabled paclet sites."

PacletSiteAdd::usage = "PacletSiteAdd is an internal symbol."

PacletSiteUpdate::usage = "PacletSiteUpdate is an internal symbol."

PacletSiteRemove::usage = "PacletSiteRemove is an internal symbol."


PacletFindRemote::usage = "PacletFindRemote is an internal symbol."


$AllowInternet::usage = "$AllowInternet specifies whether the Wolfram Language should attempt to use the Internet for certain operations. Set it to False to prevent Internet use."
$AllowDocumentationUpdates::usage = "$AllowDocumentationUpdates specifies whether the Wolfram Language should automatically update documentation notebooks from the Internet as newer ones become available. Set it to False to prevent the Wolfram Language from attempting to download newer notebooks as you browse the documentation."
$AllowDataUpdates::usage = "$AllowDataUpdates specifies whether the Wolfram Language should automatically update data collections (like CountryData, ElementData, etc.) as newer data become available. Set it to False to prevent the Wolfram Language from attempting to download updates."

$InternetProxyRules::usage = "$InternetProxyRules is a list of rules that control how the Wolfram Language accesses the Internet. Unless the first rule is UseProxy->True, the other rules, which specify protocol-specific proxies, are not used. Instead of modifying $InternetProxyRules directly, use the UseInternetProxy and SetInternetProxy functions, or the Help > Internet Connectivity dialog in the front end."
PacletManager`SetInternetProxy::usage = "SetInternetProxy[\"protocol\", {\"proxyHost\", port}] causes the Wolfram Language to use the specified proxy host and port when accessing URLs of the specified protocol (\"http\", for example). You can also use the Help > Internet Connectivity dialog in the front end to configure proxy settings."
UseInternetProxy::usage = "UseInternetProxy controls whether the Wolfram Language should use a proxy server when accessing the Internet. UseInternetProxy[Automatic] causes the Wolfram Language to attempt to use the proxy settings from your system or browser. UseInternetProxy[False] causes the Wolfram Language to connect directly to the Internet, bypassing a proxy server. UseInternetProxy[True] causes the Wolfram Language to use the proxy settings specified in $InternetProxyRules. You can also use the Help > Internet Connectivity dialog in the front end to configure proxy settings."


Begin["`Package`"]

initializeSites

$pacletSiteName
makePacletSite
errorStringFromHTTPStatusCode
errorDialog

passwordDialog

testConnectivity

$userAgent

(* A flag for users to prevent PM from putting up dialogs. *)
$allowDialogs


End[]  (* `Package` *)



(* Current context will be PacletManager`. *)

Begin["`Services`Private`"]


(***********************  Kernel-Wide Settings  **********************)

Unprotect[$AllowInternet, $AllowDocumentationUpdates, $AllowDataUpdates, $InternetProxyRules]

$AllowInternet::boolset = "Cannot set value of $AllowInternet to `1`; value must be True or False."
$AllowDocumentationUpdates::boolset = "Cannot set value of $AllowDocumentationUpdates to `1`; value must be True or False."
$AllowDataUpdates::boolset = "Cannot set value of $AllowDataUpdates to `1`; value must be True or False."


$AllowInternet /: Set[$AllowInternet, allow_] :=
    executionProtect[
        If[allow === True || allow === False,
            updateManagerData["AllowInternet" -> allow, "Write"->True],
        (* else *)
            Message[$AllowInternet::boolset, allow]
        ];
        allow
    ]

$AllowInternet := "AllowInternet" /. $managerData


$AllowDocumentationUpdates /: Set[$AllowDocumentationUpdates, allow_] :=
    (
        If[allow === True || allow === False,
            updateManagerData["IsDocAutoUpdate" -> allow, "Write"->True],
        (* else *)
            Message[$AllowDocumentationUpdates::boolset, allow]
        ];
        allow
    )

$AllowDocumentationUpdates := "IsDocAutoUpdate" /. $managerData

$AllowDataUpdates /: Set[$AllowDataUpdates, allow_] :=
    (
        If[allow === True || allow === False,
            updateManagerData["IsDataAutoUpdate" -> allow, "Write"->True],
        (* else *)
            Message[$AllowDataUpdates::boolset, allow]
        ];
        allow
    )

$AllowDataUpdates := "IsDataAutoUpdate" /. $managerData


$InternetProxyRules :=
    Module[{useProxy, httpProxyHost, httpProxyPort, httpsProxyHost, httpsProxyPort,
            ftpProxyHost, ftpProxyPort, socksProxyHost, socksProxyPort, useWPAD},
        {useProxy, httpProxyHost, httpProxyPort, httpsProxyHost, httpsProxyPort, ftpProxyHost, ftpProxyPort,
           socksProxyHost, socksProxyPort, useWPAD} =
                 {"UseProxy", "HTTPProxyHost", "HTTPProxyPort", "HTTPSProxyHost", "HTTPSProxyPort",
                       "FTPProxyHost", "FTPProxyPort", "SocksProxyHost", "SocksProxyPort", "UseWPAD"} /. $managerData;
        {"UseProxy" -> useProxy,
         "HTTP" -> If[httpProxyHost =!= Null && httpProxyPort =!= 0, {httpProxyHost, httpProxyPort}, {}],
         "HTTPS" -> If[httpsProxyHost =!= Null && httpsProxyPort =!= 0, {httpsProxyHost, httpsProxyPort}, {}],
         "FTP" -> If[ftpProxyHost =!= Null && ftpProxyPort =!= 0, {ftpProxyHost, ftpProxyPort}, {}],
         "Socks" -> If[socksProxyHost =!= Null && socksProxyPort =!= 0, {socksProxyHost, socksProxyPort}, {}],
         (* Coerce useWPAD to T/F in case it doesn't have a value. This is only to handle potential problems during the transition to serialization of this new property. *)
         "UseWPAD" -> If[useWPAD === False, False, True]
        }
    ]


Protect[$AllowInternet, $AllowDocumentationUpdates, $AllowDataUpdates, $InternetProxyRules]


(***********************  SetInternetProxy  **********************)

UseInternetProxy::val = "The argument to UseInternetProxy must be either True, False, or Automatic."

UseInternetProxy[val:(True | False | Automatic)] := updateManagerData["UseProxy" -> val, "Write"->True]

UseInternetProxy[_] := Null /; Message[UseInternetProxy::val]
UseInternetProxy[args___] := Null /; (ArgumentCountQ[UseInternetProxy, Length[{args}], 1, 1]; False)


(* J/Link's version of this function calls the one here in its implementation.
   Because JLink` might come before PacletManager` on $ContextPath, we also
   overload J/Link's version with defs that call us.
*)

PacletManager`SetInternetProxy::args = "Improper arguments to SetInternetProxy."
PacletManager`SetInternetProxy::prot = "Invalid protocol name `1`; must be one of \"HTTP\", \"HTTPS\", \"FTP\", or \"Socks\"."

JLink`SetInternetProxy[type_String, {}] := PacletManager`SetInternetProxy[type, {}]
PacletManager`SetInternetProxy[type_String, {}] :=
    Switch[ToLowerCase[type],
        "http",
            updateManagerData[{"HTTPProxyHost" -> Null, "HTTPProxyPort" -> 0}, "Write"->True],
        "https",
            updateManagerData[{"HTTPSProxyHost" -> Null, "HTTPSProxyPort" -> 0}, "Write"->True],
        "ftp",
            updateManagerData[{"FTPProxyHost" -> Null, "FTPProxyPort" -> 0}, "Write"->True],
        "socks",
            updateManagerData[{"SocksProxyHost" -> Null, "SocksProxyPort" -> 0}, "Write"->True],
        _,
            Message[PacletManager`SetInternetProxy::prot, type];
            $Failed
    ]

JLink`SetInternetProxy[type_String, {host_String, port:(_String | _Integer)}] :=
    PacletManager`SetInternetProxy[type, {host, port}]

PacletManager`SetInternetProxy[type_String, {host_String, port:(_String | _Integer)}] :=
    Switch[ToLowerCase[type],
        "http",
            updateManagerData[{"HTTPProxyHost" -> host, "HTTPProxyPort" -> ToExpression[port]}, "Write"->True],
        "https",
            updateManagerData[{"HTTPSProxyHost" -> host, "HTTPSProxyPort" -> ToExpression[port]}, "Write"->True],
        "ftp",
            updateManagerData[{"FTPProxyHost" -> host, "FTPProxyPort" -> ToExpression[port]}, "Write"->True],
        "socks",
            updateManagerData[{"SocksProxyHost" -> host, "SocksProxyPort" -> ToExpression[port]}, "Write"->True],
        _,
            Message[PacletManager`SetInternetProxy::prot, type];
            $Failed
    ]

PacletManager`SetInternetProxy[(Rule | RuleDelayed)["UseWPAD", val:(True | False)]] := updateManagerData["UseWPAD" -> val, "Write"->True]

PacletManager`SetInternetProxy[arg___] := Null /; Message[PacletManager`SetInternetProxy::args]
JLink`SetInternetProxy[arg___] :=         Null /; Message[PacletManager`SetInternetProxy::args]


(***********************************  Paclet Site Data  **********************************)

(* Delayed assignment so its evaluated value is never captured in the .mx file. *)
$userAgent := $userAgent =
    With[{versNumber = ToString[PacletManager`Information`$VersionNumber]},
        If[StringQ[$ProgramName], $ProgramName, "MathematicaProgram"] <> "/" <> getKernelVersionString[] <>
        " PM/" <> versNumber <> If[StringMatchQ[versNumber, "*."], "0.", "."] <>
        ToString[PacletManager`Information`$ReleaseNumber]
    ]


 (* $PacletSite = "http://127.0.0.1:8080/PacletServer" *)
$PacletSite = "http://pacletserver.wolfram.com"
$pacletSiteName = "Wolfram Research Paclet Server"

(*  Form of $pacletSiteData:

      List of lists, one for each site:
         {"URL", "Name", local:True|False, {last update date attempt}, {last successful update date}, {paclets}}
     The dates are in M list format, empty list if no date.

*)


initializeSites[readOnlyMode:(True | False)] :=
    (
        (* Note that the site file uses only the first digit of the M version number in its name: pacletSiteData_9.pmd2.
           This avoids spinning off multiple copies of this large file with every small upgrade to the M version.
           The site file is not highly dependent on the version of M requesting it (that it might differ at all is purely
           a server-side optimization).
        *)
        $pacletSitePersistentFile =
            ToFileName[$userConfigurationDir, "pacletSiteData_" <> First[StringSplit[getKernelVersionString[], "."]] <> ".pmd2"];
        $pacletSiteData =.;
        (* A scheduled task will deserialize the data in 2 seconds if it is not requested earlier. *)
        If[!readOnlyMode,
            RunScheduledTask[
                RemoveScheduledTask[$ScheduledTask];
                getPacletSiteData[];
                If[TrueQ[$AllowInternet] && !isSubKernel[], doWeeklyUpdate[]],
                {2}
            ]
        ]
    )


(* Because the site data is filled in after startup by a scheduled task, callers who want $pacletSiteData must
   always get it via this accessor, to be sure initialization has finished.
*)

getPacletSiteData[] :=
    If[ListQ[$pacletSiteData],
        $pacletSiteData,
    (* else *)
        If[$pmMode =!= "ReadOnly", readSiteData[], $pacletSiteData = {}];
        $pacletSiteData
    ]


(*********************************  Serialization  **********************************)

(* Serialization strategy:

   The file that holds downloaded data about site paclets is pacletSiteData_9.pmd2. Note that the filename only
   contains the first digit in the M version number, in contrast to other PM data files. This is to avoid spewing
   multiple copies of this large file as users go through a series of M version upgrades. This scheme works fine
   because the file is not tied to the precise version number of M. As an optimization, the server might send back
   a specialized set of data for a specific version of M, but it won't be any more fine-grained than the
   major version number.

   The read/write functions are guarded by executionProtect to ensure that they cannot be preempted. and also that
   they run to completion. This does not prevent multiple instances of the kernel running on the same machine from
   clobbering each other as they read/write the data, however. This is not uncommon, for example in gridMathematica.
   That problem is solved by each kernel opening a "lock" file before reading/writing, and then deleting the file
   when done. The code checks to see if the lockfile exists before trying to read or write, indicating that another
   kernel is in the process of reading/writing. If the file exists when a write is attempted, it waits for a short
   time, then fails if the lock file still exists. It isn't a critical problem if a write doesn't
   occur. If the lockfile exists when a read is initiated, the caller waits for a short time for the lockfile to
   go away. If it does, then the read proceeds. If not, no read occurs.
*)


$serializationVersion = 0

writeSiteData[tryAgain:(True | False):True] /; $pmMode =!= "ReadOnly" :=
    executionProtect @
    Quiet @
    Module[{lockFile, strm},
        lockFile = ToFileName[$userTemporaryDir, "pacletSiteData_" <> First[StringSplit[getKernelVersionString[], "."]] <> ".lock"];
        (* Wait a second to get the lock, but walk away if it can;tbe acquired. Not that big a deal if we can't write
           the data. If the lock is being held, then some other process is probably writing it anyway.
        *)
        If[acquireLock[lockFile, 1.5, False],
            using[{strm = OpenWrite[$pacletSitePersistentFile]},
                Write[strm, $serializationVersion];
                (* Write potentially calls Read (via getPacletSiteData[], so we need to prevent readSiteData
                   from choking when it sees the lockFile present. The $inWrite flag alerts readSiteData that
                   the lockFile presence is expected and it is safe to go ahead and read.
                *)
                Write[strm, Block[{$inWrite = True}, getPacletSiteData[]]]
            ];
            releaseLock[lockFile]
        ]
    ]

readSiteData[] /; $pmMode =!= "ReadOnly" :=
    executionProtect @
    Quiet @
    Module[{lockFile, strm, serVersion, data, useLockFile},
        useLockFile = !TrueQ[$inWrite];
        lockFile = ToFileName[$userTemporaryDir, "pacletSiteData_" <> First[StringSplit[getKernelVersionString[], "."]] <> ".lock"];
        If[useLockFile && acquireLock[lockFile, 1.5, False],
            strm = OpenRead[$pacletSitePersistentFile];
            If[Head[strm] === InputStream,
                serVersion = Read[strm, Expression];
                (* TODO: Respect serVersion. *)
                (* The Block here ensures that the only symbol in the file, Paclet, is interpreted as being PacletManager`Paclet. *)
                data =  Block[{$ContextPath = {"PacletManager`", "System`"}}, Read[strm, Expression]]
            ];
            Close[strm];
            releaseLock[lockFile]
        ];
        Which[
            MatchQ[data, {{_String, _String, True | False, _List, _List, _List}...}],
                (* Got good data from the file; use it. *)
                $pacletSiteData = data,
            MatchQ[$pacletSiteData, {{_String, _String, True | False, _List, _List, _List}...}],
                (* We didn't get good data from the file, but the existing data is good, so keep using it. *)
                Null,
            True,
                (* Reset. *)
                $pacletSiteData = {}
        ];
        (* Make sure that the default server is always present at the start of every session. *)
        If[Length[Cases[$pacletSiteData, {$PacletSite, __}]] == 0,
            PrependTo[$pacletSiteData, {$PacletSite, $pacletSiteName, False, {}, {}, {}}]
        ]
    ]


(**************************************  Public API  ***************************************)


PacletSites[] := PacletSite[#[[1]], #[[2]], "Local" -> #[[3]]]& /@ getPacletSiteData[]



Options[PacletSiteAdd] = {"Local" -> False, Prepend -> False}

PacletSiteAdd::badurl = "URL `1` is not a properly formed http or file URL."
General::nosite = "Site `1` is not an existing Paclet site."
General::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog."


(* Returns PacletSite xpression, or $Failed iff is obviously a bogus URL (e.g., htp:/foo). *)

PacletSiteAdd[siteURL_String, opts:OptionsPattern[]] := PacletSiteAdd[siteURL, "", opts]

PacletSiteAdd[site_PacletSite, opts:OptionsPattern[]] := PacletSiteAdd[site[[1]], site[[2]], Append[{opts}, site[[3]]]]

PacletSiteAdd[siteURL_String, name_String, opts:OptionsPattern[]] :=
    executionProtect @
    Module[{dataChanged, existingSite, isLocal, isPrepend},
        dataChanged = False;
        If[!(StringMatchQ[siteURL, "http*:*"] || StringMatchQ[siteURL, "file:*"]),
            Message[PacletSiteAdd::badurl, siteURL];
            Return[$Failed]
        ];
        {isLocal, isPrepend} = TrueQ /@ OptionValue[{"Local", Prepend}];
        If[StringMatchQ[siteURL, "file:*"], isLocal = True];
        existingSite = Cases[getPacletSiteData[], {siteURL, __}];
        If[Length[existingSite] > 0,
            existingSite = First[existingSite];
            Which[
                existingSite[[2]] == name && existingSite[[3]] == isLocal && !isPrepend,
                    (* Nothing to do; already exists in requested form. *)
                    Null,
                isPrepend,
                    (* Move the site to the front of the list, changing name and local attrs as necessary. *)
                    $pacletSiteData = Prepend[DeleteCases[getPacletSiteData[], {siteURL, __}], ReplacePart[existingSite, {2->name, 3->isLocal}]];
                    dataChanged = True,
                True,
                    (* Change name and local attrs as necessary, leaving the position unchanged. *)
                    $pacletSiteData = Replace[getPacletSiteData[], {siteURL, n_, loc_, d1_, d2_, p_} :>
                                                  {siteURL, name, isLocal, d1, d2, p}, {1}];
                    dataChanged = True
            ],
        (* else *)
            (* is new *)
            $pacletSiteData = If[isPrepend, Prepend, Append][$pacletSiteData, {siteURL, name, isLocal, {}, {}, {}}];
            dataChanged = True
        ];
        If[dataChanged,
            writeSiteData[]
        ];
        PacletSite[siteURL, name, "Local" -> isLocal]
    ]


(* Returns PacletSites[] left after removal. *)

PacletSiteRemove[site_PacletSite] := PacletSiteRemove[First[site]]

PacletSiteRemove[siteURL_String] :=
    executionProtect @
    Module[{originalData},
        originalData = getPacletSiteData[];
        $pacletSiteData = DeleteCases[originalData, {siteURL, __}];
        If[Length[$pacletSiteData] == Length[originalData],
            Message[PacletSiteRemove::nosite, siteURL],
        (* else *)
            writeSiteData[]
        ];
        PacletSites[]
    ]


(*
    When called synchronously, returns either a PacletSite expression representing a successfully updated site,
    or Null if no site update occurred. This could be because the site was not among your set of PacletSites[],
    or $AllowInternet was False, or there was some kind of error attempting the update. These situations issue
    appropriate messages.

    the reason that PacletSiteUpdate["newSite"] is an error instead of just adding and then updating the new site
    is that we don't want this function to inadvertantly add a site if an existing one was attempting to be updated
    and the URL was misspelled.
*)

PacletSiteUpdate::err = "An error occurred attempting to update paclet information from site `1`. `2`"


Options[PacletSiteUpdate] = Options[startPacletSiteUpdate] = {"Interactive"->False, Asynchronous->False, "Force"->True, "Timeout"->Automatic}

Attributes[PacletSiteUpdate] = {Listable}

PacletSiteUpdate[site_PacletSite, opts:OptionsPattern[]] := PacletSiteUpdate[First[site], opts]

PacletSiteUpdate[siteURL_String, opts:OptionsPattern[]] :=
    Module[{startResult, siteRec},
        (* Quietly reject attempts to call PacletSiteUpdate if in ReadOnly mode. *)
        If[$pmMode === "ReadOnly",
            Return[Null]
        ];
        (* A feature intended for WRI developers is to be able to call PacletSiteUpdate with "Force"->False and
           be able to skip the update if it has already been done by an automatic "once in a session" process.
           For example, the weekly update, or an update forced by the first call to a data paclet.
        *)
        If[OptionValue["Force"] === False,
            If[TrueQ[$checkedForUpdates],
                (* Do nothing; "fake" a return value. *)
                siteRec = First[Cases[getPacletSiteData[], {siteURL, __}]];
                Return[PacletSite[siteRec[[1]], siteRec[[2]], "Local" -> siteRec[[3]]]],
            (* else *)
                (* Proceed with the update, but prevent if from happening again in this session for other
                   "Force"->False callers.
                *)
                $checkedForUpdates = True
            ]
        ];
        (* To enable asynchronous behavior, we split the
           functionality for site updating into two parts. The first part starts the download, and the
           second part uncompresses and updates the data when the download finishes (this is called from
           the async callback function).
        *)
        startResult = startPacletSiteUpdate[siteURL, opts];
        Switch[startResult,
            _List,
                (* Was a synchronous call. Finish the update immediately. *)
                finishPacletSiteUpdate[startResult],
            _AsynchronousTaskObject,
                (* Return the task object. finishPacletSiteUpdate gets called in the task automatically when the download finishes. *)
                startResult,
            _,
                (* Get here in several "normal" circumstances, like AllowInternet is False.
                   Message will already have been issued.
                *)
                $Failed
        ]
    ]

startPacletSiteUpdate[siteURL_String, opts:OptionsPattern[]] :=
    Module[{siteRec, interactive, async, tempFile, task, url, statusCode},
        interactive = TrueQ[OptionValue["Interactive"]] && TrueQ[$allowDialogs];
        (* Will be True|False|number. *)
        async = OptionValue[Asynchronous];
        siteRec = Cases[getPacletSiteData[], {siteURL, __}];
        If[Length[siteRec] > 0,
            siteRec = First[siteRec];
            If[!TrueQ[siteRec[[3]]] && !TrueQ[$AllowInternet],
                If[interactive,
                    (* This message is actually defined for General, so need to use General to get the text as a string.
                       The StringForm is to convert the special sequence `.` into a single `.
                    *)
                    errorDialog[ToString[StringForm[General::offline]]],
                (* else *)
                    Message[PacletSiteUpdate::offline]
                ];
                Return[Null]
            ];
            (* First step is to update the "last attempted update date". *)
            $pacletSiteData = Replace[getPacletSiteData[], {siteURL, n_, loc_, d1_, d2_, p_} :> {siteURL, n, loc, Date[], d2, p}, {1}];
            url = If[siteURL == $PacletSite,
                      StringReplace[siteURL, "pacletserver" -> ("pacletserver" <> $wriPacletServerIndex)],
                  (* else *)
                      siteURL
                  ] <> "/PacletSite.mz";
            tempFile = ToFileName[$userTemporaryDir, "ps" <> ToString[$ProcessID] <> ".mz"];
            If[TrueQ[async] || NumberQ[async],
                PreemptProtect[
                    (* Use PreemptProtect to ensure that setTaskData[] gets called before siteUpdateCallback can fire. *)
                    task = URLSaveAsynchronous[url, tempFile, siteUpdateCallback[async],
                                  "UserAgent" -> $userAgent,
                                  "Headers" -> {"Mathematica-systemID" -> $SystemID,
                                                  "Mathematica-license" -> ToString[$LicenseID],
                                                     "Mathematica-mathID" -> ToString[$MachineID],
                                                        "Mathematica-language" -> ToString[$Language],
                                                           "Mathematica-activationKey" -> ToString[$ActivationKey]},
                                  BinaryFormat -> True, "Progress" -> False
                           ];
                    setTaskData[task, {siteURL, tempFile, interactive, async}]
                ];
                task,
            (* else *)
                (* Synchronous. *)
                statusCode =
                     URLSave[url, tempFile, "StatusCode",
                          "UserAgent" -> $userAgent,
                          "Headers" -> {"Mathematica-systemID" -> $SystemID,
                                          "Mathematica-license" -> ToString[$LicenseID],
                                             "Mathematica-mathID" -> ToString[$MachineID],
                                                "Mathematica-language" -> ToString[$Language],
                                                   "Mathematica-activationKey" -> ToString[$ActivationKey]},
                          BinaryFormat -> True
                     ];
                {siteURL, tempFile, interactive, async, If[IntegerQ[statusCode], statusCode, "Unknown error"]}
            ],
        (* else *)
            (* Site was not already present. *)
            If[interactive,
                errorDialog["Update failed because site " <> siteURL <> " is not an existing paclet site."],
            (* else *)
                Message[PacletSiteUpdate::nosite, siteURL]
            ];
            Null
        ]
    ]


(* The two ways a download can end are in a status code or an error. *)
siteUpdateCallback[async_][task_, "statuscode" | "error", {statusCodeOrErrorString_}] :=
    (
        setTaskData[task, Append[getTaskData[task], statusCodeOrErrorString]];
        Which[
            TrueQ[async],
                (* For async, we will never need to know the return value of finishPacletSiteUpdate, so we can call it
                   here; it's an "auto-finish".
                *)
                Quiet @ finishPacletSiteUpdate[task],
            NumberQ[async],
                (* If async value is a number, delay that many seconds before executing the finish. *)
                RunScheduledTask[Quiet @ finishPacletSiteUpdate[task]; RemoveScheduledTask[$ScheduledTask], {async}]
        ]
    )


finishPacletSiteUpdate[task_] :=
    Module[{siteURL, file, statusCodeOrErrorString, interactive, async, taskData},
        taskData = getTaskData[task];
        If[Length[taskData] == 4,
            (* It is possible for finishPacletSiteUpdate to be called after WaitAsynchronousTask times out, in which case the statusCodeOrErrorString
               value will not have been appened to the task data. This is a failure and there is nothing to do here.
            *)
            (* TODO: Should we issue a message here? PacletSiteUpdate is rarely called directly by users, so a message coming out of the blue might not be appropriate. *)
            Return[$Failed]
        ];
        {siteURL, file, interactive, async, statusCodeOrErrorString} = taskData;
        freeTaskData[task];
        finishPacletSiteUpdate[{siteURL, file, interactive, async, statusCodeOrErrorString}]
    ]


finishPacletSiteUpdate[{siteURL_, file_, interactive_, async_, statusCodeOrErrorString_}] :=
    Module[{success, data, sitePaclets, errorMsg},
        success = True;
        If[isNetworkSuccess[siteURL, statusCodeOrErrorString],
            (* Download appeared to work. *)
            data = ZipGetFile[file, "PacletSite.m"];
            DeleteFile[file];
            data = FromCharacterCode[data, "UTF8"];
            sitePaclets = parsePacletInfo[data];
            (* sitePaclets is either $Failed or {__Paclet}. *)
            If[MatchQ[sitePaclets, {__Paclet}],
                (* Store the data, updating the "last successful update date" field. *)
                $pacletSiteData = Replace[getPacletSiteData[], {siteURL, n_, loc_, d1_, d2_, p_} :>
                                              {siteURL, n, loc, d1, Date[], sitePaclets}, {1}];
                writeSiteData[];
                If[siteURL == $PacletSite,
                    updateManagerData[{"LastUpdatedSite" -> $PacletSite, "LastUpdatedSiteDate" -> Date[]}, "Write" -> True]
                ];
                If[interactive, errorDialog["Update from paclet site " <> siteURL <> " succeeded."]],
            (* else *)
                (* Bad data in site file, or perhaps corrupted download. *)
                If[interactive,
                    errorDialog["Update from paclet site " <> siteURL <> " failed."],
                (* else *)
                    Message[PacletSiteUpdate::err, siteURL, "Improperly-formatted data obtained from this paclet site."]
                ];
                success = False
            ],
        (* else *)
            (* Status code != 200 or is error string or $Failed. *)
            (* First do failover to different WRI server if appropriate. 502 and 504 are BAD_GATEWAY and GATEWAY_TIMEOUT,
               which you can get from a proxy server if the dest server is unreachable.
            *)
            If[siteURL == $PacletSite && !$alreadyTriedFailover &&
                   (StringQ[statusCodeOrErrorString] || statusCodeOrErrorString === 502 || statusCodeOrErrorString === 504),
                $wriPacletServerIndex = ToString[Mod[ToExpression[$wriPacletServerIndex], 6] + 1];
                $alreadyTriedFailover = True;
                (* Simply try again with new setting for server index. *)
                Return[PacletSiteUpdate[siteURL, "Interactive"->interactive, Asynchronous->async]]
            ];
            errorMsg =
                Which[
                    statusCodeOrErrorString === 404,
                        (* The PacletSite.mz file could not be found on the server. *)
                        "Does not appear to be a valid paclet site",
                    IntegerQ[statusCodeOrErrorString],
                        errorStringFromHTTPStatusCode[statusCodeOrErrorString],
                    True,
                        statusCodeOrErrorString
                ];
            If[interactive,
                errorDialog[{"Update from paclet site " <> siteURL <> " failed.", errorMsg}],
            (* else *)
                Message[PacletSiteUpdate::err, siteURL, errorMsg]
            ];
            success = False
        ];
        If[success,
            siteRec = First[Cases[getPacletSiteData[], {siteURL, __}]];
            PacletSite[siteRec[[1]], siteRec[[2]], "Local" -> siteRec[[3]]],
        (* else *)
            $Failed
        ]
    ]



testConnectivity[] :=
    Module[{success, msgLines},
        success = False;
        If[TrueQ[$AllowInternet],
            (* TODO: Better handling of different message types. Don't want a message from LibraryLink
               problem, or failure to write the pacletSiteData file, to look like a network issue.
            *)
            success = Check[PacletSiteUpdate[$PacletSite]; True, False];
            If[success,
                errorDialog["Connectivity test succeeded using pacletserver.wolfram.com."],
            (* else *)
                (* Pass False for the "typeset" arg here because we don't want the Internet Connectivity
                   button added to the dialog (users invoked this function from that dialog).
                *)
                msgLines = {"Connectivity test failed using pacletserver.wolfram.com.",
                            "\n", "The server or network might be unavailable, or ",
                            "you might need to configure proxy settings.",
                            Row[{"Click ", Button["here", Documentation`HelpLookup["paclet:tutorial/TroubleshootingInternetConnectivity", Null],
                                BaseStyle->"Link", Appearance->Frameless, Evaluator->Automatic], " for more information."}]};
                errorDialog[msgLines]
            ],
        (* else *)
            errorDialog[{"Connectivity test failed!", ToString[StringForm[General::offline]]}]
        ];
        success
    ]


(* This function is called from a scheduled task after startup to do a once-weekly update of $PacletSite. *)
doWeeklyUpdate[] :=
    Quiet @
    Module[{dates},
        dates = Cases[getPacletSiteData[], {$PacletSite, _, _, _, lastSuccessfulUpdate_, _} :> lastSuccessfulUpdate];
        If[Length[dates] == 0 || First[dates] === {} || differenceInDays[First[dates], Date[]] > 3,
            PacletSiteUpdate[$PacletSite, Asynchronous->True, "Force"->False]
        ]
    ]


(* Delayed eval so its value is never captured in the .mx file. *)
$wriPacletServerIndex := $wriPacletServerIndex = ToString[Random[Integer, {1,6}]]
$alreadyTriedFailover = False


(******************  Error/Warning Dialogs for Asynchronous Operations  ******************)

errorStringFromHTTPStatusCode[statusCode_Integer] :=
    Switch[statusCode,
        404,
           "File not found on server.",
        200,
            "Could not access server.",
        0,
            "Unknown error.",  (* Shouldn't happen. *)
        _?(#>600&),
             General::dlrefused,
        _,
            (* TODO: Expand this handling to give sensible messages for other HTTP codes. *)
            "Server returned HTTP status code " <> ToString[statusCode] <> "."
    ]


If[!ValueQ[$allowDialogs], $allowDialogs = True]

errorDialog[$Aborted] := errorDialog["The network operation was aborted."]

errorDialog[statusCode_Integer] := errorDialog[errorStringFromHTTPStatusCode[statusCode]]

errorDialog[line1_String] := errorDialog[{line1, ""}]

errorDialog[lines:{__}] :=
    If[$allowDialogs, MessageDialog[Column[lines], WindowTitle -> "Wolfram Paclet Manager"]]



(*************************************  PacletFindRemote  ****************************************)

Options[PacletFindRemote] = {"Location"->All, "SystemID"->Automatic, "WolframVersion"->Automatic,
                                "Extension"->All, "Creator"->All, "Publisher"->All, "Context"->All, "UpdateSites"->False}

PacletFindRemote[pacletName:(_String | All):All, opts:OptionsPattern[]] :=
    PacletFindRemote[{pacletName, All}, opts]

PacletFindRemote[{pacletName:(_String | All):All, pacletVersion:(_String | All):All}, opts:OptionsPattern[]] :=
    Module[{location, matchingPaclets, site},
        If[OptionValue["UpdateSites"], Quiet[PacletSiteUpdate /@ PacletSites[]]];
        location = OptionValue["Location"];
        If[location === All, location = _];
        matchingPaclets = Join @@
            forEach[site, Cases[getPacletSiteData[], {location, __}],
                setLocation[PCfindMatching["Paclets" -> Last[site], "Name" -> pacletName, "Version" -> pacletVersion,
                                            Sequence @@ DeleteCases[Flatten[{opts}], ("Location" -> _) | ("UpdateSites" -> _)]],
                            First[site]
                ]
            ];
        Flatten[groupByNameAndSortByVersion[matchingPaclets]]
    ]


End[]
