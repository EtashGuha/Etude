

(*Set to False to disable getting Operating System configured proxies for Windows and OSX through CURLLink.
This should only be disabled if a change in CURLLink breaks functionality.
Environment variable proxies will still function.*)
$enableOperatingSystemProxies = True;

(*Set to False to disable SoundCloud ServiceConnect functionality.*)
$enableSoundCloudFunctionality = False;

$CACERT;

$useCACERT = 1;

LoadInternetStreamResources[curlDir_] := 
(
	$CACERT = FileNameJoin[{curlDir, "SSL", "cacert.pem"}];
)

(*  Do not set CURLOPT_CAINFO, now that OSX version of libcurl uses
    darwinssl.
    This option was relevent when we were using openssl for https requests.
    https://stash.wolfram.com/projects/PAC/repos/curllink/commits/6c93dcd10c8823222963a2870010808edc4798d3

    For Mac we use OSX native Keychain, after switching to darwinssl, this is only relevent when using openssl.
*)
If[$SystemID=="MacOSX-x86-64",
    $useCACERT = 0;
];

$proxyEnvVars = <|"http_proxy" -> 1, "https_proxy" -> 2, "ftp_proxy" -> 3, "socks_proxy" -> 4, "all_proxy" -> 5, "no_proxy" -> 6|>;

(****************************************************************************)
(* Password Dialog code modified from CURLLink's HTTP.m *)

(* Old default Wolfram System password dialog *)
If[!ValueQ[$allowDialogs], $allowDialogs = True]
hasFrontEnd[] := ToString[Head[$FrontEnd]] === "FrontEndObject"
$pwdDlgResult;
$pwdDlgStandaloneRetries = 0;

passwordDialogStandalone[prompt1_, prompt2_, prompt3_] :=
    (
        Print[prompt1];
        Print[prompt2];
        Print[prompt3];
        $pwdDlgResult = {InputString["username: "], InputString["password (will echo as cleartext): "]};
        $pwdDlgStandaloneRetries++;
    )

passwordDialogFE[title_, prompt1_, prompt2_, prompt3_] :=
    Module[{cells, uname = "", pwd = "", createDialogResult},
        cells = {
            TextCell[prompt1, NotebookDefault, "DialogStyle", "ControlStyle"],
            TextCell[prompt2, NotebookDefault, "DialogStyle", "ControlStyle"],
            ExpressionCell[Grid[{ {TextCell["UserName:  "], InputField[Dynamic[uname], String, ContinuousAction -> True,
                ImageSize -> 200, BoxID -> "UserNameField"]}, {TextCell["Password:  "],
                InputField[Dynamic[pwd], String, ContinuousAction -> True,
                    ImageSize -> 200, FieldMasked -> True]}}], "DialogStyle", "ControlStyle"],
            TextCell[prompt3, NotebookDefault, "DialogStyle", "ControlStyle"],

            ExpressionCell[ Row[{DefaultButton[$pwdDlgResult = {uname, pwd};
            DialogReturn[], ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]]], Spacer[{2.5`, 42, 16}],
                CancelButton[$pwdDlgResult = $Canceled; DialogReturn[],
                    ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]]]}], TextAlignment -> Right] };
        createDialogResult = DialogInput[DialogNotebook[cells],
            WindowTitle -> title, WindowSize -> {400, FitAll}, Evaluator -> CurrentValue["Evaluator"],
            LineIndent -> 0, PrivateFontOptions -> {"OperatorSubstitution" -> False} ];
        If[createDialogResult === $Failed,
            Null,
        (* else *)
            MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[ FE`BoxReference[createDialogResult, {{"UserNameField"}},
                FE`BoxOffset -> {FE`BoxChild[1]}]]];
            $pwdDlgResult
        ]
    ]

coreDialog[url_String, prompt2_String, urlPrompt_String] :=
    Module[{title, prompt1, prompt3},
        title = "Authentication Required";
        Clear[$pwdDlgResult];
        Which[
            !TrueQ[$allowDialogs],
            Null,
            hasFrontEnd[],
        (* Use FE dialog box *)
            prompt1 = Row[{urlPrompt, Hyperlink[url, BaseStyle -> "ControlStyle"]}];
            prompt3 = "(These values are kept for this connection only.)";
            passwordDialogFE[title, prompt1, prompt2, prompt3],
            True,
            prompt1 = urlPrompt <> url;
            prompt3 = "(These values are kept for this connection only.)";
            passwordDialogStandalone[prompt1, prompt2, prompt3]
        ]
    ]

passwordDialog[url_String, lastAuthFailed_Integer] := coreDialog[url, "The server is requesting authentication.", "You are attempting to read from the URL:\n"]
proxyDialog[url_String, lastAuthFailed_Integer] := coreDialog[url, "The proxy server is requesting authentication.", "You are attempting to connect to the proxy server:\n"]
passwordDialog[url_String, 1] := coreDialog[url, "Your last authentication attempt failed.\nThe server is requesting re-authentication.", "You are attempting to read from the URL:\n"]
proxyDialog[url_String, 1] := coreDialog[url, "Your last authentication attempt failed.\nThe proxy server is requesting re-authentication.", "You are attempting to connect to the proxy server:\n"]

(****************************************************************************)

(*Internet Stream Functions:*)

(* Following URLFetchAsynchronous convention http://reference.wolfram.com/language/ref/URLFetchAsynchronous.html?q=URLFetchAsynchronous&lang=en *)
internetStreamInternalAsyncCallback[streamID_, asynchObj_, eventType_, data_] :=
    Module[{func, dlnow, dltotal},
        func = getField[streamID, "CallbackFunction"];
        If[eventType == "progress",
            dltotal = data[[1]];
            dlnow = dltotal - getField[streamID, "DownloadProgress"];
            setField[streamID, "DownloadProgress", dltotal];
            If[!(func === None),
                func[AudioFileStreamTools`FileStreamObject[streamID], eventType, {dlnow, dltotal}];
            ];
        ];
        If[eventType == "data",
            If[!(func === None),
                func[AudioFileStreamTools`FileStreamObject[streamID], eventType, data];
            ];
        ];
    ]

(****************************************************************************)
(* Proxy code modified from CURLLink's HTTP.m *)

extractScheme[url_String] := Module[{scheme, positions},
    positions = StringPosition[url, "://"];
    scheme = StringTake[url, {1, positions[[1]][[2]]}];
    Return[scheme];
];
addProxies[allProxies_, newProxies_] :=
    Module[{newProxyList = allProxies},
        Scan[If[! MemberQ[allProxies, #], AppendTo[newProxyList, #]] &,
            newProxies];
        Return[newProxyList];
    ];
getProxiesWrapper[url_String, flag_] := Module[{allProxies, scheme},
    If[flag == False,
        Return[{""}];
    ];

    scheme = extractScheme[url];
    If[flag == Automatic,
        allProxies = getProxies[url, flag];
        If[allProxies[[1]] == All,
            Return[{allProxies[[2]], removeDuplicateScheme["socks://" <> allProxies[[2]]]}];
        ];
        allProxies = addProxies[allProxies, getProxies["socks://a", flag]];
        If[allProxies == {}, Return[{""}];];
        Return[allProxies];
    ];
    If[flag == True,
        allProxies = getProxies[url, flag];
        If[allProxies == {}, Return[{""}];];
        Return[allProxies];
    ];

];

removeDuplicateScheme[url_String] :=
    Module[{positions, newurl = url},
        positions = StringPosition[newurl, "://"];
        While[Length[positions] >= 2,
            newurl = StringReplacePart[newurl, "", {1, positions[[1]][[2]]}];
            positions = StringPosition[newurl, "://"];
        ];
        Return[newurl];
    ];

URISplit[uri_String] :=
    Flatten[StringCases[uri,
        RegularExpression[
            "^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?"] ->
            {"Scheme" -> "$2", "Authority" -> "$4"}]]

URIJoin[uri : List[_Rule, _Rule]] :=
    Module[{scheme, authority},
        If[! Fold[And, True, Map[MatchQ[#, Rule[_String, _String]] &, uri]],
            Return[$Failed]];
        {scheme, authority} = Map[Last, uri];
        StringJoin[
            Cases[{If[scheme =!= "", StringJoin[scheme, ":"]],
                Which[authority =!= "" && scheme =!= "",
                    StringJoin["//", authority], authority === "" && scheme =!= "",
                    authority]}, Except[Null]]]]
getProxies[url_String, False] = {""}


tourl[{url_String, port_Integer}] := url <> ":" <> ToString[port]
tourl[{url_String}] := url
tourl[___] := Nothing
wlproxy[proto_String] :=
    Module[{url, scheme, fullurl}, fullurl = proto /. PacletManager`$InternetProxyRules;
    url = tourl[fullurl];
    If[fullurl =!= {}, scheme = URLParse[fullurl[[1]], "Scheme"]];
    If[scheme === None, url = ToLowerCase[proto] <> "://" <> url, url]]

(*when UseProxy->True,in PacletManager`$InternetProxyRules, and scheme isn't https or ftp,
 getProxies defaults to HTTP and Socks proxies given by PacletManager`$InternetProxyRules *)
getProxies[url_String, True] :=
    Module[{proxies, scheme = URLParse[url, "Scheme"]},
        proxies =
            Which[scheme === "https", {wlproxy["HTTPS"], wlproxy["Socks"]},
                scheme === "ftp", {wlproxy["FTP"], wlproxy["Socks"]},
                True, {wlproxy["HTTP"], wlproxy["Socks"]}];
        If[proxies === {}, {""}, proxies]]

getProxies[url_String, Automatic] := Module[{proxies, scheme},
    proxies = getSystemProxies[url, $OperatingSystem];
    If[Length[proxies] === 2 && proxies[[1]] === All,
    	scheme = extractScheme[url];
        proxies = {All, removeDuplicateScheme[scheme <> proxies[[2]]]};
        Return[proxies];
    ];
    proxies = Map[removeDuplicateScheme[#] &, proxies];
    Return[proxies];
];

getSystemProxies[url_String, "Windows"] :=
    Module[{rawProxies, proxies, proxyList = {}},
    	If[$enableOperatingSystemProxies === False, Return[{}];];
        rawProxies =
            If[(StringMatchQ[url, "http://*"] ||
                StringMatchQ[url, "https://*"] || StringMatchQ[url, "ftp://*"] ||
                StringMatchQ[url, "ftps://*"] ||
                StringMatchQ[url, "socks*://*"]),
                Quiet[Check[CURLLink`CURLGetProxies[url], {}],
                    LibraryFunction::strnull]
            (*else*),
                Quiet[Check[CURLLink`CURLGetProxies["http://" <> url], {}],
                    LibraryFunction::strnull]];
                    
        If[rawProxies == $Failed, Return[{All, {}}];];
        If[(!StringMatchQ[rawProxies, "http=*"]) && (!StringMatchQ[rawProxies, "https=*"]) && (!StringMatchQ[rawProxies, "ftp=*"]) && (!StringMatchQ[rawProxies, "socks=*"]),
        	rawProxies = "http=" <> rawProxies <> ";http=https://" <> rawProxies <>";https=" <> rawProxies <> ";https=http://" <> rawProxies <> ";ftp=" <> rawProxies <> ";socks=" <> rawProxies;
        ];
        rawProxies = StringSplit[rawProxies, ";"];
        If[StringMatchQ[url, "http://*"],
        	proxies = Select[rawProxies, StringMatchQ[#, "http=*"] &];
        	proxies = Map[StringReplace[#, "http=" -> "", 1] &, proxies];
			proxies = Map[("http://" <> #) &, proxies];
			Scan[AppendTo[proxyList, #] &, proxies];
        ];
        If[StringMatchQ[url, "https://*"],
        	proxies = Select[rawProxies, StringMatchQ[#, "https=*"] &];
        	proxies = Map[StringReplace[#, "https=" -> "", 1] &, proxies];
			proxies = Map[("https://" <> #) &, proxies];
			Scan[AppendTo[proxyList, #] &, proxies];
        ];
        If[StringMatchQ[url, "ftp://*"],
        	proxies = Select[rawProxies, StringMatchQ[#, "ftp=*"] &];
        	proxies = Map[StringReplace[#, "ftp=" -> "", 1] &, proxies];
			proxies = Map[("ftp://" <> #) &, proxies];
			Scan[AppendTo[proxyList, #] &, proxies];
        ];
        proxies = Select[rawProxies, StringMatchQ[#, "socks=*"] &];
    	proxies = Map[StringReplace[#, "socks=" -> "", 1] &, proxies];
		proxies = Map[("socks://" <> #) &, proxies];
		Scan[AppendTo[proxyList, #] &, proxies];
        Return[proxyList];
];

getSystemProxies[url_String, "MacOSX"] :=
    Module[{},
    	If[$enableOperatingSystemProxies === False, Return[{}];];
        If[(StringMatchQ[url, "http://*"] ||
            StringMatchQ[url, "https://*"] || StringMatchQ[url, "ftp://*"] ||
            StringMatchQ[url, "ftps://*"] ||
            StringMatchQ[url, "socks*://*"]),
            Flatten@{Quiet[
                Check[CURLLink`CURLGetProxies[
                    URIJoin[Flatten@{URISplit[url]}]], {}]]}
        (*else*),
            Flatten@{Quiet[
                Check[CURLLink`CURLGetProxies[
                    URIJoin[Flatten@{URISplit["http://" <> url]}]], {}]]}]]

getSystemProxies[url_String, _] := {}
buildProxy[{scheme_String, url_String}] :=
    If[StringMatchQ[url, scheme <> "://*"], url, scheme <> "://" <> url]
buildProxy[{url_String}] := url
buildProxy[url_String] := url


(****************************************************************************)

internetStreamChooseProxy[url_String, proxyType_] :=
    Module[{proxies = {}},
        If[proxyType == Automatic,
            proxies = DeleteDuplicates[getProxiesWrapper[ToLowerCase[url], ("UseProxy" /. PacletManager`$InternetProxyRules)]];
            proxies = Map[removeDuplicateScheme[#] &, proxies];
            Return[proxies];
        ];
        If[proxyType == "HTTP",
            proxies = DeleteDuplicates[getProxiesWrapper["http://a", ("UseProxy" /. PacletManager`$InternetProxyRules)]];
            proxies = Map[removeDuplicateScheme[#] &, proxies];
            Return[proxies];
        ];
        If[proxyType == "HTTPS",
            proxies = DeleteDuplicates[getProxiesWrapper["https://a", ("UseProxy" /. PacletManager`$InternetProxyRules)]];
            proxies = Map[removeDuplicateScheme[#] &, proxies];
            Return[proxies];
        ];
        If[proxyType == "Socks",
            proxies = DeleteDuplicates[getProxiesWrapper["socks://a", ("UseProxy" /. PacletManager`$InternetProxyRules)]];
            proxies = Map[removeDuplicateScheme[#] &, proxies];
            Return[proxies];
        ];
        If[proxyType == "FTP",
            proxies = DeleteDuplicates[getProxiesWrapper["ftp://a", ("UseProxy" /. PacletManager`$InternetProxyRules)]];
            proxies = Map[removeDuplicateScheme[#] &, proxies];
            Return[proxies];
        ];
        Return[$Failed];
    ]

addToAvailableProxies[availableProxies_, proxy_] :=
    If[proxy =!= "",
        Append[availableProxies, proxy]
        ,
        availableProxies
    ]

proxyErrorQ[error_Integer] = If[error < -100, True, False]

$downloadStatusMessagesAssociation = <|-1 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::unknownerror, #1]&), -2 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::invalidsslcertificate, #1]&),
    -3 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::authenticationrequired, #1]&), -4 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::filenotfound, #1]&),
    -5 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::forbidden, #1]&), -6 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::servernotfound, #1]&),
    -7 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::unsupportedprotocol, #1]&), -8 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::timedout, #1]&),
    -9 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::couldnotconnect, #1]&), -10 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::readerror, #1]&),

    -103 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxyunknownerror, #1]&) , -108 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxyinvalidsslcertificate, #1]&),
    -102 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxyauthenticationrequired, #1]&), -101 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxyservernotfound, #1]&),
    -107 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxyunsupportedprotocol, #1]&), -105 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxytimedout, #1]&),
    -104 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxycouldnotconnect, #1]&), -106 -> (Message[AudioFileStreamTools`InternetStreamOpenRead::proxyreaderror, #1]&)|>;

$downloadStatusAssociation = <|0 -> "InProgress", 1 -> "Complete",
    2 -> "PartiallyComplete", -1 -> "Aborted",
    -2 -> "AbortedInvalidCertificate", -3 -> "AuthenticationRequired",
    -4 -> "FileNotFound", -5 -> "Forbidden", -6 -> "ServerNotFound",
    -7 ->"UnsupportedProtocol", -8 -> "TimedOut",
    -9 -> "CouldNotConnect", -10 -> "ErrorReadingData",

    -101 -> "ProxyServerNotFound", -102 -> "ProxyAuthenticationRequired",
    -103 -> "ProxyAborted", -104 -> "ProxyCouldNotConnect",
    -105 -> "ProxyTimedOut", -106 -> "ProxyErrorReadingData",
    -107 -> "ProxyUnsupportedProtocol", -108 -> "ProxyAbortedInvalidCertificate"|>;

(* TODO: Assert file output format is the same as stream read-in format *)
(* "AudioFormat" option will be used if "FilePath" is not specified, and the format was not found from the formatServiceURL[] parser. *)
Options[AudioFileStreamTools`InternetStreamOpenRead] =
    {   "ContainerType" -> "RawArray", "FilePath" -> None,
        "AudioFormat" -> None, "DeleteFileOnClose" -> Automatic, "DeleteFileOnExit" -> Automatic,
        "DataUpdates" -> False, "VerifySSLCertificate" -> False,
        "UserName" -> None, "Password" -> None, "ConnectTimeout" -> 0,
        "ProxyUserName" -> None, "ProxyPassword" -> None, "ProxyType" -> Automatic, "ProxyOverride" -> None
    }; (* NOTE: "UserName" is correct, URLFetch uses "Username", but this is going to be changed to "UserName"?*)

$lfcLLproxyTypeAssociation = <|None -> 0, "GivenProxy" -> 1, 0 -> None, 1 -> "GivenProxy"|>;

AudioFileStreamTools`InternetStreamOpenRead[url_String, opts:OptionsPattern[]]:= AudioFileStreamTools`InternetStreamOpenRead[url, None, opts];
AudioFileStreamTools`InternetStreamOpenRead[url_String, func:Except[_Rule], opts:OptionsPattern[]]:=
    Module[{streamID, res, containerType, deleteOnClose = 1, deleteOnExit = 0, dataUpdates = 1, verifySSLCert = 1, useUsernamePassword = 0, callbackDataType,
        filePath, directory, formats, audioFormat, formattedURL, existingURL, hasWritePermissions, userName = "", password = "", timeout = 0,
        useProxyUsernamePassword = 0, proxyAddress = "", proxyUserName = "", proxyPassword = "", lfcLLproxyType = $lfcLLproxyTypeAssociation[None],
        availableProxies = {}, retriedStream, failureMessage, originalStreamType = 0},

        If[$pwdDlgStandaloneRetries >= 3,
            $pwdDlgStandaloneRetries = 0;
            Message[AudioFileStreamTools`InternetStreamOpenRead::authfailed];
            Return[$Failed];
        ];

        containerType = OptionValue["ContainerType"];
        formats = {".mp3", ".wav", ".ogg", ".oga", ".aif", ".aiff", ".flac"};
        If[!(containerType === "RawArray") && !(containerType === "MTensor"), Message[AudioFileStreamTools`InternetStreamOpenRead::invalidcontainer, containerType];Return[$Failed];];
        loadAdapter[];

        (* Append http:// to URL if url does not have a protocol specified. *)
        If[StringPosition[url, "://"] === {},
            formattedURL = "http://" <> url;
            ,
            formattedURL = url;
        ];

        {formattedURL, audioFormat} = formatServiceURL[formattedURL];
        If[formattedURL == $Failed,
            Return[$Failed];
        ];

        If[IntegerQ[OptionValue["ConnectTimeout"]] && OptionValue["ConnectTimeout"] > 0,
            timeout = OptionValue["ConnectTimeout"];
        ];

        If[OptionValue["VerifySSLCertificate"] == False, verifySSLCert = 0;];

        If[OptionValue["UserName"] =!= None || OptionValue["Password"] =!= None,
            If[OptionValue["UserName"] =!= None && OptionValue["Password"] =!= None,
                If[lfURLIsOpenedAsInternetStream[formattedURL] === 1,
                    Message[AudioFileStreamTools`InternetStreamOpenRead::usernamepasswordimmutable];
                    ,
                    userName = OptionValue["UserName"];
                    password = OptionValue["Password"];
                    useUsernamePassword = 1;
                ];
                ,
                Message[AudioFileStreamTools`InternetStreamOpenRead::usernamepasswordformat];
                Return[$Failed];
            ];
        ];

        If[OptionValue["ProxyUserName"] =!= None || OptionValue["ProxyPassword"] =!= None,
            If[OptionValue["ProxyUserName"] =!= None && OptionValue["ProxyPassword"] =!= None,
                If[lfURLIsOpenedAsInternetStream[formattedURL] === 1,
                    Message[AudioFileStreamTools`InternetStreamOpenRead::proxyusernamepasswordimmutable];
                    ,
                    proxyUserName = OptionValue["ProxyUserName"];
                    proxyPassword = OptionValue["ProxyPassword"];
                    useProxyUsernamePassword = 1;
                ];
                ,
                Message[AudioFileStreamTools`InternetStreamOpenRead::proxyusernamepasswordformat];
                Return[$Failed];
            ];
        ];

        If[OptionValue["ProxyOverride"] === None,
            Quiet[availableProxies = internetStreamChooseProxy[formattedURL, OptionValue["ProxyType"]];];

            If[availableProxies == $Failed,
                Message[AudioFileStreamTools`InternetStreamOpenRead::proxyconfigurationerror];
                Return[$Failed];
            ];
            
            availableProxies = DeleteCases[availableProxies, "http://"];
            availableProxies = DeleteCases[availableProxies, "https://"];
            availableProxies = DeleteCases[availableProxies, "socks*://"];

            If[availableProxies == {},
                availableProxies = {""};
            ];
            proxyAddress = availableProxies[[1]];
            ,
            availableProxies = {OptionValue["ProxyOverride"]};
            proxyAddress = availableProxies[[1]];
        ];

        If[("UseProxy" /. PacletManager`$InternetProxyRules) == Automatic && proxyAddress == "",
            availableProxies = {};

            If[extractScheme[formattedURL] == "http://",
                availableProxies = addToAvailableProxies[availableProxies, lfGetEnvironmentProxySettings[$proxyEnvVars["http_proxy"]]];
                availableProxies = addToAvailableProxies[availableProxies, lfGetEnvironmentProxySettings[$proxyEnvVars["socks_proxy"]]];
                availableProxies = addToAvailableProxies[availableProxies, lfGetEnvironmentProxySettings[$proxyEnvVars["all_proxy"]]];
            ];
            If[extractScheme[formattedURL] == "https://",
                availableProxies = addToAvailableProxies[availableProxies, lfGetEnvironmentProxySettings[$proxyEnvVars["https_proxy"]]];
                availableProxies = addToAvailableProxies[availableProxies, lfGetEnvironmentProxySettings[$proxyEnvVars["socks_proxy"]]];
                availableProxies = addToAvailableProxies[availableProxies, lfGetEnvironmentProxySettings[$proxyEnvVars["all_proxy"]]];
            ];

            If[availableProxies == {},
                availableProxies = {""};
            ];
            proxyAddress = availableProxies[[1]];

        ];

        If[proxyAddress != "",
            lfcLLproxyType = $lfcLLproxyTypeAssociation["GivenProxy"];
        ];

        (*Print["availableProxies: " <> ToString[availableProxies]];*)
        (*Print["proxyAddress: " <> proxyAddress];*)
        (*Print["lfcLLproxyType: " <> ToString[$lfcLLproxyTypeAssociation[lfcLLproxyType]]];*)

        originalStreamType = lfURLOpenedAsInternetStreamType[formattedURL];

        If[OptionValue["FilePath"] === None,
            Message[AudioFileStreamTools`InternetStreamOpenRead::filepathnonedeprecated];
            If[audioFormat == None,
                audioFormat = StringTake[formattedURL, Last[StringPosition[formattedURL, formats], 0]];
                If[audioFormat == ".flac",
                    Message[AudioFileStreamTools`InternetStreamOpenRead::flacnotsupported];
                    Return[$Failed];
                ];
                If[OptionValue["AudioFormat"] === None,
                    If[audioFormat == "", audioFormat = ".mp3"];
                    ,
                    audioFormat = OptionValue["AudioFormat"];
                ];
            ];

            If[(OptionValue["DeleteFileOnClose"] =!= Automatic || OptionValue["DeleteFileOnExit"] =!= Automatic),
                Message[AudioFileStreamTools`InternetStreamOpenRead::nodeletionoptions];
            ];
            streamID = lfInternetStreamOpenReadMemory[ formattedURL, audioFormat, $CACERT, $useCACERT, verifySSLCert, useUsernamePassword, userName, password, timeout, useProxyUsernamePassword, proxyUserName, proxyPassword, proxyAddress, lfcLLproxyType];
            If[Head[streamID] === LibraryFunctionError, Message[AudioFileStreamTools`InternetStreamOpenRead::openreadfail, formattedURL]; Return[$Failed]];
            setField[streamID, "FilePath", None];

            If[!IntegerQ[originalStreamType] || originalStreamType === 0,
                setField[streamID, "InternetStreamType", "Memory"];
                ,
                setField[streamID, "InternetStreamType", $originalStreamTypes[originalStreamType]];
            ];
            ,
            directory = FileNameTake[OptionValue["FilePath"], {1, -2}];
            If[directory === "", directory = Directory[];];
            filePath = ImportExport`FileUtilities`GetFilePath[directory];
            If[!DirectoryQ[directory] || filePath === $Failed,
                Message[AudioFileStreamTools`InternetStreamOpenRead::dirnex, directory];
                Return[$Failed];
            ];
            filePath = filePath <> FileNameTake[OptionValue["FilePath"]];
            audioFormat = "." <> Last[StringSplit[filePath, "."]];
            If[audioFormat == ".flac",
                Message[AudioFileStreamTools`InternetStreamOpenRead::flacnotsupported];
            ];

            If[$streamTypeAssoc[filePath] === "Write" || $streamTypeAssoc[filePath] === "Read", Message[AudioFileStreamTools`InternetStreamOpenRead::streamtypeconflict, "Internet", $streamTypeAssoc[filePath], filePath]; Return[$Failed];];

            If[(OptionValue["DeleteFileOnClose"] =!= Automatic || OptionValue["DeleteFileOnExit"] =!= Automatic) && (lfFileIsOpenedAsInternetStream[filePath] === 1 || lfURLIsOpenedAsInternetStream[formattedURL] === 1),
                Message[AudioFileStreamTools`InternetStreamOpenRead::setdelopts];
                ,
                If[OptionValue["DeleteFileOnClose"] == False,
                    deleteOnClose = 0;
                    If[OptionValue["DeleteFileOnExit"] == True, deleteOnExit = 1;];
                ];
            ];

            existingURL = lfInternetStreamFilePathGetURL[filePath];
            If[existingURL =!= "" && existingURL =!= formattedURL,
                Message[AudioFileStreamTools`InternetStreamOpenRead::openreadfailfilepath, filePath, formattedURL];
                Return[$Failed];
            ];

            streamID = lfInternetStreamOpenReadFile[ formattedURL, audioFormat, $CACERT, $useCACERT, filePath, deleteOnClose, deleteOnExit, verifySSLCert, useUsernamePassword, userName, password, timeout, useProxyUsernamePassword, proxyUserName, proxyPassword, proxyAddress, lfcLLproxyType];

            If[Head[streamID] === LibraryFunctionError, Message[AudioFileStreamTools`InternetStreamOpenRead::openreadfail, formattedURL]; Return[$Failed]];
            $streamTypeAssoc[filePath] = "Internet";

            If[!IntegerQ[originalStreamType] || originalStreamType === 0,
                setField[streamID, "InternetStreamType", "File"];
                setField[streamID, "FilePath", filePath];
                ,
                setField[streamID, "InternetStreamType", $originalStreamTypes[originalStreamType]];
                setField[streamID, "FilePath", lfInternetStreamURLGetFilePath[formattedURL]];
                Message[AudioFileStreamTools`InternetStreamOpenRead::filepathignored, filePath, formattedURL];
            ];
        ];

        hasWritePermissions = lfStreamHasOperatingSystemWritePermissions[streamID];
        If[Head[hasWritePermissions] === LibraryFunctionError, Message[AudioFileStreamTools`InternetStreamOpenRead::openreadfail, formattedURL]; lfFileStreamClose[streamID]; Return[$Failed]];
        If[hasWritePermissions =!= 1, Message[AudioFileStreamTools`InternetStreamOpenRead::nowriteperm, filePath, directory]; lfFileStreamClose[streamID]; Return[$Failed]];

        setField[streamID, "DataContainer", OptionValue["ContainerType"]];
        setField[streamID, "CallbackFunction", func];
        setField[streamID, "DataUpdates", OptionValue["DataUpdates"]];
        setField[streamID, "DownloadProgress", 0];
        setField[streamID, "InternetStream", True];
        setField[streamID, "URL", formattedURL];

        If[!(func === None) && OptionValue["DataUpdates"] == True, dataUpdates = 2];
        If[containerType == "MTensor", callbackDataType = 0];
        If[containerType == "RawArray", callbackDataType = 1];

        asyncObj = Internal`CreateAsynchronousTask[lfInternetStreamStartDownload, {streamID, dataUpdates, callbackDataType}, internetStreamInternalAsyncCallback[streamID, #, #2, #3] & ];
        setField[streamID, "AsyncObject", asyncObj];
        res = lfInternetStreamWaitForTransferInitialization[streamID];

        If[res < 0,
            AudioFileStreamTools`InternetStreamClose[AudioFileStreamTools`FileStreamObject[ streamID]];

            If[SameQ[$downloadStatusAssociation[res], "AuthenticationRequired"],
                passwordDialog[url, useUsernamePassword];
                If[$pwdDlgResult === $Canceled,
                    $downloadStatusMessagesAssociation[res][url];
                    Return[$Failed];
                    ,
                    Return[
                        AudioFileStreamTools`InternetStreamOpenRead[url, func,
                        "ContainerType" -> OptionValue["ContainerType"], "FilePath" -> OptionValue["FilePath"],
                        "AudioFormat" -> OptionValue["AudioFormat"], "DeleteFileOnClose" -> OptionValue["DeleteFileOnClose"], "DeleteFileOnExit" -> OptionValue["DeleteFileOnExit"],
                        "DataUpdates" -> OptionValue["DataUpdates"], "VerifySSLCertificate" -> OptionValue["VerifySSLCertificate"],
                        "UserName" -> $pwdDlgResult[[1]], "Password" -> $pwdDlgResult[[2]], "ConnectTimeout" -> OptionValue["ConnectTimeout"],
                        "ProxyUserName" -> OptionValue["ProxyUserName"], "ProxyPassword" -> OptionValue["ProxyPassword"], "ProxyType" -> OptionValue["ProxyType"], "ProxyOverride" -> OptionValue["ProxyOverride"]
                        ]
                    ];
                ];
            ];

            If[SameQ[$downloadStatusAssociation[res], "ProxyAuthenticationRequired"],
                proxyDialog[proxyAddress, useProxyUsernamePassword];
                If[$pwdDlgResult === $Canceled,
                    $downloadStatusMessagesAssociation[res][proxyAddress];
                    Return[$Failed];
                    ,
                    Return[
                        AudioFileStreamTools`InternetStreamOpenRead[url, func,
                            "ContainerType" -> OptionValue["ContainerType"], "FilePath" -> OptionValue["FilePath"],
                            "AudioFormat" -> OptionValue["AudioFormat"], "DeleteFileOnClose" -> OptionValue["DeleteFileOnClose"], "DeleteFileOnExit" -> OptionValue["DeleteFileOnExit"],
                            "DataUpdates" -> OptionValue["DataUpdates"], "VerifySSLCertificate" -> OptionValue["VerifySSLCertificate"],
                            "UserName" -> OptionValue["UserName"], "Password" -> OptionValue["Password"], "ConnectTimeout" -> OptionValue["ConnectTimeout"],
                            "ProxyUserName" -> $pwdDlgResult[[1]], "ProxyPassword" -> $pwdDlgResult[[2]], "ProxyType" -> OptionValue["ProxyType"], "ProxyOverride" -> OptionValue["ProxyOverride"]
                        ]
                    ];
                ];
            ];

            If[proxyErrorQ[res], (* Retry with other available proxy methods *)
                $downloadStatusMessagesAssociation[res][proxyAddress];
                availableProxies = Delete[availableProxies, 1];
                While[OptionValue["ProxyOverride"] === None && availableProxies =!= {},
                    (*Print["Retrying with " <> availableProxies[[1]]];*)
                    retriedStream = AudioFileStreamTools`InternetStreamOpenRead[url, func,
                        "ContainerType" -> OptionValue["ContainerType"], "FilePath" -> OptionValue["FilePath"],
                        "AudioFormat" -> OptionValue["AudioFormat"], "DeleteFileOnClose" -> OptionValue["DeleteFileOnClose"], "DeleteFileOnExit" -> OptionValue["DeleteFileOnExit"],
                        "DataUpdates" -> OptionValue["DataUpdates"], "VerifySSLCertificate" -> OptionValue["VerifySSLCertificate"],
                        "UserName" -> OptionValue["UserName"], "Password" -> OptionValue["Password"], "ConnectTimeout" -> OptionValue["ConnectTimeout"],
                        "ProxyUserName" -> OptionValue["ProxyUserName"], "ProxyPassword" -> OptionValue["ProxyPassword"], "ProxyType" -> OptionValue["ProxyType"], "ProxyOverride" -> availableProxies[[1]]
                    ];
                    If[retriedStream =!= $Failed, Return[retriedStream];];
                    availableProxies = Delete[availableProxies, 1];
                ];
                Return[$Failed];
            ];
        ];

        failureMessage = $downloadStatusMessagesAssociation[res];
        If[MissingQ[failureMessage], Return[AudioFileStreamTools`FileStreamObject[streamID]];];

        failureMessage[url];
        Return[$Failed];
    ]

AudioFileStreamTools`InternetStreamOpenRead[___]:= (Message[AudioFileStreamTools`InternetStreamOpenRead::invalidargs]; Return[$Failed]);

AudioFileStreamTools`InternetStreamGetBufferedRange[obj_AudioFileStreamTools`FileStreamObject] :=
    Module[{streamID, startPos, endPos, bitDepth},
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[AudioFileStreamTools`InternetStreamGetBufferedRange::afstnostream, obj]; Return[$Failed]];

        startPos = 1;
        endPos = getField[streamID, "DownloadProgress"];
        If[endPos == Missing["NotAvailable"], endPos = 1];

        Return[{startPos, endPos}]
    ]


AudioFileStreamTools`InternetStreamGetBufferedRange[___]:= (Message[AudioFileStreamTools`InternetStreamGetBufferedRange::invalidargs]; Return[$Failed]);

AudioFileStreamTools`InternetStreamDownloadStatus[obj_AudioFileStreamTools`FileStreamObject] :=
    Module[{streamID, res, resStr},
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[AudioFileStreamTools`InternetStreamDownloadStatus::afstnostream, obj]; Return[$Failed]];
        res = lfInternetStreamDownloadStatus[streamID];
        If[Head[res] == LibraryFunctionError, Message[AudioFileStreamTools`InternetStreamDownloadStatus::afstnostream, obj]; Return[$Failed]];

        resStr = $downloadStatusAssociation[res];
        If[MissingQ[resStr], Return[$Failed]; (* Need message *)];
        Return[resStr];
    ]

AudioFileStreamTools`InternetStreamDownloadStatus[___]:= (Message[AudioFileStreamTools`InternetStreamOpenRead::invalidargs]; Return[$Failed]);

AudioFileStreamTools`InternetStreamDownloadPercent[obj_AudioFileStreamTools`FileStreamObject] :=
    Module[{streamID, size, finalSize, frameCount, finalFrameCount, sizeDiv = -1, frameCountDiv = -1},
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[AudioFileStreamTools`InternetStreamDownloadPercent::afstnostream, obj]; Return[$Failed]];
        If[AudioFileStreamTools`InternetStreamDownloadStatus[obj] == "Complete", Return[N[1/1]];];
        size = lfInternetStreamCurrentDownloadSize[streamID];
        finalSize = lfInternetStreamFinalDownloadSize[streamID];
        If[size =!= LibraryFunctionError["LIBRARY_FUNCTION_ERROR", 6] && finalSize =!= LibraryFunctionError["LIBRARY_FUNCTION_ERROR", 6],
            If[finalSize >= size && finalSize > 0,
                (*Print[size];*)
                (*Print[finalSize];*)
                sizeDiv = N[size/finalSize];
            ]
        ];

        If[sizeDiv > 0, Return[sizeDiv];];

        frameCount = lfFileStreamGetMetaInformation[streamID, $metaInformationFields["FrameCount"]];
        finalFrameCount = lfFileStreamGetMetaInformation[streamID, $metaInformationFields["TotalFrameCount"]];
        If[frameCount =!= LibraryFunctionError["LIBRARY_FUNCTION_ERROR", 6] && finalFrameCount =!= LibraryFunctionError["LIBRARY_FUNCTION_ERROR", 6] && !MissingQ[frameCount] && !MissingQ[finalFrameCount],
            If[finalFrameCount >= frameCount && finalFrameCount > 0,
                frameCountDiv = N[frameCount/finalFrameCount];
            ]
        ];

        If[frameCountDiv >= 0, Return[frameCountDiv];];
        If[sizeDiv == 0, Return[sizeDiv];];

        Return[Missing["Indeterminate"]];
    ]

AudioFileStreamTools`InternetStreamDownloadStatus[___]:= (Message[AudioFileStreamTools`InternetStreamOpenRead::invalidargs]; Return[$Failed]);

AudioFileStreamTools`InternetStreamReadN[args___] := AudioFileStreamTools`FileStreamReadN[args];
AudioFileStreamTools`InternetStreamClose[args___] := AudioFileStreamTools`FileStreamClose[args];
AudioFileStreamTools`InternetStreamGetReadPosition[args___] := AudioFileStreamTools`FileStreamGetReadPosition[args];
AudioFileStreamTools`InternetStreamSetReadPosition[args___] := AudioFileStreamTools`FileStreamSetReadPosition[args];
AudioFileStreamTools`InternetStreamGetMetaInformation[args___] := AudioFileStreamTools`FileStreamGetMetaInformation[args];
AudioFileStreamTools`InternetStreamGetStreamInformation[args___] := AudioFileStreamTools`FileStreamGetStreamInformation[args];

(* TODO: Need to decide on where this parser should live, a seperate paclet? *)
formatServiceURL[url_] :=
    Quiet[Module[{parsedURL, accessToken, queryAssoc, soundCloudAccessToken, sc, trackURL,
        urlToResolve, resolveURL, rawURL},
        parsedURL = URLParse[url];
        If[$enableSoundCloudFunctionality && StringEndsQ[ToLowerCase[parsedURL["Domain"]] , "soundcloud.com"],
            Quiet[sc = ServiceConnect["SoundCloud"];];

            If[Check[ServiceConnections`ServiceInformation[sc], $Failed, {ServiceConnections`Private`ServiceInformation::nolink}] == $Failed,
            (* Throw non connected message *)
                Message[AudioFileStreamTools`InternetStreamOpenRead::soundcloudauthfailed];
                Return[{$Failed, None}];
            ];

            Check[
                soundCloudAccessToken = Select[Internal`CheckCache[{"OAuthTokens", "SoundCloud"}] /. OAuthSigning`OAuthToken -> List, SameQ[Head[#], OAuthSigning`Private`Token20] &][[1]][[1]];
                ,
                Message[AudioFileStreamTools`InternetStreamOpenRead::soundcloudauthfailed];
                Return[{$Failed, None}];
            ];
            If[!StringQ[soundCloudAccessToken],
                Message[AudioFileStreamTools`InternetStreamOpenRead::soundcloudauthfailed];
                Return[{$Failed, None}];
            ];

            (* If domain is not api, attempt to resolve *)

            If[! SameQ[ToLowerCase[parsedURL["Domain"]] , "api.soundcloud.com"],
                parsedURL["Query"] = {};
                parsedURL["Fragment"] = None;
                urlToResolve = URLBuild[parsedURL];
                Check[
                    rawURL = sc["RawResolve", "url" -> urlToResolve];
                    ,
                    (*Throw non resolve message *)
                    Message[AudioFileStreamTools`InternetStreamOpenRead::filenotfound, url];
                    Return[{$Failed, None}];
                ];
                If[!StringQ[rawURL],
                (*Throw non resolve message *)
                    Message[AudioFileStreamTools`InternetStreamOpenRead::filenotfound, url];
                    Return[{$Failed, None}];
                ];
                Check[
                    parsedURL = URLParse[rawURL];
                    parsedURL["Path"] = Append[parsedURL["Path"], "stream"];
                    ,
                    (*Throw non resolve message *)
                    Message[AudioFileStreamTools`InternetStreamOpenRead::filenotfound, url];
                    Return[{$Failed, None}];
                ];
            ];
            (* Valid API stream or download URL *)
            If[SameQ[parsedURL["Path"][[2]], "tracks"] && (SameQ[parsedURL["Path"][[4]], "stream"] || SameQ[parsedURL["Path"][[4]], "download"]),
                queryAssoc = <|parsedURL["Query"]|>;
                KeyDropFrom[queryAssoc, "access_token"];
                queryAssoc["oauth_token"] = soundCloudAccessToken;
                parsedURL["Query"] = Normal[queryAssoc];
                If[SameQ[parsedURL["Path"][[4]], "stream"], (* Stream URL *)
                    Return[{URLBuild[parsedURL], ".mp3"}];
                ];
                If[SameQ[parsedURL["Path"][[4]], "download"], (* Download URL *)
                    Return[{URLBuild[parsedURL], None}];
                ];
            ];

            Message[AudioFileStreamTools`InternetStreamOpenRead::invalidsoundcloudurl, url];
            Return[{$Failed, None}];

        ];
        Return[{url, None}];
    ];
    ,
    {Part::partw}
];
