(* ::Package:: *)

(* Mathematica package *)
BeginPackage["CloudObject`"]

Begin["`Private`"]

handleErrorDetails["max-viewers", extra_, head_] := Message[head::maxviewers, extra]
handleErrorDetails["unknown-user", extra_, head_] := Message[head::userunknown, extra]
handleErrorDetails["user-not-found", extra_, head_] := Message[head::invusr, extra]
handleErrorDetails["MemoryLimitExceeded", extra_, head_] := Message[head::memorylimit, extra]
handleErrorDetails["TimeLimitExceeded", extra_, head_] := Message[head::timelimit, extra]
handleErrorDetails[code_, extra_, head_] := Message[head::notparam]

(*TODO: handle various error codes*)
checkError[response_, msghd_Symbol:CloudObject] :=
    If[response === {$Failed, $Failed},
        True,
        With[{res = MatchQ[response, HTTPError[_Integer, ___]]},
            If[res,
                Switch[response,
                    HTTPError[400 | 412, {__Rule}, ___],
                        (* When the server returns a 400 or 412, it sometimes returns JSON data in the response body,
                         giving more details about the error. *)
                        handleErrorDetails[Lookup[response[[2]], "errorCode"], Lookup[response[[2]], "extraData"], msghd],
                    HTTPError[503, rules:{__Rule} /; KeyExistsQ[rules,"errorCode"] && KeyExistsQ[rules, "errorDetails"], ___],
                        handleErrorDetails[Lookup[response[[2]], "errorCode"], Lookup[response[[2]], "errorDetails"], msghd],
                    HTTPError[400, ___], Message[msghd::notparam],
                    HTTPError[401, ___], Message[msghd::notauth],(*might need a different message here*)
                    HTTPError[403, ___], Message[msghd::notperm],
                    HTTPError[404, ___], Message[msghd::cloudnf], (* TODO need a CloudObject to pass here *)
                    HTTPError[405, ___], Message[msghd::notmethod],
                    HTTPError[412, ___], Message[msghd::cloudprecondition],
                    HTTPError[429, ___], Message[msghd::rejreq],
                    HTTPError[500, ___], Message[msghd::srverr],
                    HTTPError[503, ___], Message[msghd::unavailable],
                    _, Message[msghd::cloudunknown]
                ]
            ];
            res
        ]
    ]

handleBadObjectUUID[uuid_, obj_CloudObject] :=
    ( handleBadObjectUUIDHelper[uuid, obj]; $Failed )

handleBadObjectUUIDForExecute[uuid_, obj_CloudObject] :=
    ( handleBadObjectUUIDHelper[uuid, obj]; {$Failed, $Failed} )

handleBadObjectUUIDHelper[uuid_, obj_CloudObject] :=
    Switch[uuid,
        (* not found *)
        None, Message[CloudObject::cloudnf, obj],
        (* unauthenticated or not permitted. getCloudAndUUID returns the error message *)
        $Failed, Null,
        (* anything else *)
        _, Message[CloudObject::srverr, obj]
    ]

fetchURL[cloud_, url_, elements_, options___] /; useUnauthenticatedRequestQ[cloud] := 
	URLFetch[url, elements, options]

fetchURL[cloud_, url_, elements_, options___] := 
    Block[{$CloudBase = cloud},
    	authenticatedURLFetch[url, elements, options]
    ]

(* The asynchronous version simply ignores the requested elements. It does not return anything, it just sets off the request. *)
fetchURLAsync[cloud_, url_, elements_, options___] /; useUnauthenticatedRequestQ[cloud] :=
    ( URLFetchAsynchronous[url, Null, options]; {200, {}, {}} )

fetchURLAsync[cloud_, url_, elements_, options___] := 
	Block[{$CloudBase = cloud},
    	authenticatedURLFetchAsynchronous[url, Null, options];
    	{200, {}, {}}
    ]

useUnauthenticatedRequestQ[cloud_] := And[TrueQ[$CloudEvaluation], cloud =!= $EvaluationCloudBase]

contentDisplay[list:{_Integer...}] := FromCharacterCode[list]
contentDisplay[value_String] := value
contentDisplay[expr_] := ToString[expr, InputForm]

preprocessContent[content_List] := FromCharacterCode[content, "UTF-8"]
preprocessContent[content_] := content

preprocessErrorContent[content_, "application/json"] :=
    Module[{data},
        data = importFromJSON[preprocessContent[content]];
        log["Error content: `1`", data, DebugLevel->2];
        data
    ]
preprocessErrorContent[content_, type_] := preprocessContent[content]

callServer[cloud_, url_, mimetype_: "text/plain", httpVerb_: "GET", body_: "", async_ : False] :=
    Module[{response, status, headers, content, callFunction, finalURL, contentType, 
    	localBody = Replace[body, {} -> ""](* work around for bug-342303 *)},
        log["Calling remote server: `1` `2` with MIME type `3`", httpVerb, url, mimetype];
        log["Decoded URL: `1`", URLDecode[url], DebugLevel->2];
        log["Request content: `1`", contentDisplay[localBody], DebugLevel->2];
        finalURL = url;
        callFunction = If[async, fetchURLAsync, fetchURL];
        response = callFunction[cloud, finalURL, {"StatusCode", "Headers", "ContentData"},
           "Method"->httpVerb,
           "Headers"->{
               "Content-Type"->mimetype,
               "Accept"->"application/vnd.wolfram.v1"
           },
           "Body"->localBody,
           "VerifyPeer"->False,
           "DisplayProxyDialog" -> False
        ];
        If[MatchQ[response,{_,_,_}], {status, headers, content} = response, If[MatchQ[response, _HTTPError], Return[response], Return[HTTPError[404]]]];
        log["Response status: `1`", status];
        If[headers =!= {},
           log["Response headers: `1`", headers, DebugLevel->2];
        ];
        log["Response content: `1`", contentDisplay[content], DebugLevel->2];
        contentType = contentTypeCheck[Lookup[Rule @@@ headers, "Content-Type"]];
        If[Not[And[status >= 200, status < 300]],
           content = preprocessErrorContent[content, contentType];
           Return[HTTPError[status, content, contentType]]
        ];
        {contentType, content}
    ]

getUUID[cloud_, path_] := Module[{pathString, uuid},
    pathString = JoinURL[path];
    uuid = responseToString @ execute[cloud, "GET", {"files"}, Parameters -> {"path" -> pathString}];
    log["UUID for path `1`: `2`", pathString, uuid];
    If[uuid === "", None, uuid]
]

getCloud[uri_] :=
    Module[{cloud, uuid, user, path, ext, extraPath, search},
        {cloud, uuid, user, path, ext, extraPath, search} = parseURI[uri];
        StringReplace[cloud, "://datadrop." -> "://www."]
    ]

getCloudAndUUID[obj : CloudObject[uri_String, ___]] := getCloudAndUUID[uri]
    
getCloudAndUUID[uri_String] :=
    Module[{cloud, uuid, user, path, ext, extraPath, search},
        {cloud, uuid, user, path, ext, extraPath, search} = parseURI[uri];
        If[uuid === None,
            uuid = getUUID[cloud, {user, path}],
        (* uuid set, check for path inside it (file inside an unnamed directory) *)
            If[extraPath =!= {},
                uuid = getUUID[cloud, {uuid, extraPath}]
            ]
        ];
        {cloud, uuid}
    ]

getCloudAndUUID[x_] := {None, None}

getCloudAndUUIDOrPath[obj : CloudObject[uri_String, ___]] := getCloudAndUUIDOrPath[uri]

getCloudAndUUIDOrPath[uri_String] :=
    Module[{cloud, uuid, user, path, ext, extraPath, search},
        {cloud, uuid, user, path, ext, extraPath, search} = parseURI[uri];
        If[extraPath === {},
            {cloud, uuid, If[path === None, None, Join[{user}, path]]},
        (* else: *)
            If[uuid === None,
            (* this will not actually happen, because extraPath is only set when uuid is set *)
                {cloud, None, Join[{user}, path, extraPath]},
            (* else *)
                {cloud, None, Join[{uuid}, extraPath]}
            ]
        ]
    ]

getCloudAndPathList[obj_CloudObject] :=
    Module[{cloud, uuid, path},
        {cloud, uuid, path} = getCloudAndUUIDOrPath[obj];
        {cloud, If[path === None, {uuid}, path]}
    ]

Options[execute] = {Parameters -> {}, Body -> {}, Type -> "text/plain", UseUUID -> True, 
    Asynchronous -> False, "ResponseFormat" -> "ByteList"};

(* supported values for "ResponseFormat": 
   "ByteList": returns List of byte-range integers
   "ByteString": returns a String, each character's code point is in the byte range
   future values:
   "ByteArray": returns a ByteArray expression
*)

(* perform the execute locally, we are already in the cloud *)
Options[executeInCloud] = Options[execute];
executeInCloud[cloud_String, method_String, path_List : {}, OptionsPattern[]] :=
    Module[{parameters, mimetype = OptionValue[Type], body = OptionValue[Body],
        bodyString, responseFormat = OptionValue["ResponseFormat"]},

        parameters = OptionValue[Parameters];

        log["Calling server `1` `2` with MIME type `3`, parameters `4`", method,
            JoinURL[path], mimetype, ToString[parameters, InputForm],
            DebugLevel -> 2];
        If[body =!= {},
            log["Content: `1`", body, DebugLevel->2];
        ];
        (* string is more efficient than a list of bytes in the Java server *)
        bodyString = If[ListQ[body], FromCharacterCode[body], body];

        $lastExecuteResult = CloudSystem`Private`writeCallPacketService[
            CloudSystem`CloudObject`DoCloudOperation[method, path, parameters,
                mimetype, bodyString
            ]
        ];
        log["Call packet result: `1`", ToString[$lastExecuteResult, InputForm], DebugLevel -> 2];

        Replace[
            $lastExecuteResult,
            {
                {type_, File[resultFile_String]} :>
                    {contentTypeCheck[type], 
                        If[responseFormat === "ByteString",
                            Replace[ReadString[resultFile], EndOfFile -> ""],
                        (* Else assume "ByteList" *)
                            BinaryReadList[resultFile]
                        ]},
                {type_, result_String} :> 
                    {contentTypeCheck[type], 
                        If[responseFormat === "ByteString",
                            result,
                        (* Else assume "ByteList" *)
                            ToCharacterCode[result]
                        ]
                    },
                {type_, result_List} :> {contentTypeCheck[type], result},
                HTTPError[status_Integer?Positive, content_, type_] :>
                    HTTPError[status, preprocessErrorContent[content, type], type],
                err:HTTPError[_Integer?Positive] :> err,
                _ :> HTTPError[500]
            }
        ]
    ]

(* make an HTTP request to perform the execute *)
Options[executeRemotely] = Options[execute];
executeRemotely[cloud_String, method_String, path_List : {}, OptionsPattern[]] := Module[{url},
    url = JoinURL[{cloud, path}] <> JoinURLSearch[OptionValue[Parameters]];
    callServer[cloud, url, OptionValue[Type], method, OptionValue[Body], OptionValue[Asynchronous]]
]

execute[cloud_String, method_String, path_List : {}, opts:OptionsPattern[]] :=
    If[TrueQ[System`$CloudEvaluation] && (cloud === $EvaluationCloudBase),
        executeInCloud[cloud, method, path, opts]
        ,
        executeRemotely[cloud, method, path, opts]
    ]

execute[obj_CloudObject, method : _String | Automatic : "GET", api_String : "files", subpath_List : {}, options : OptionsPattern[]] :=
    Module[{cloud, uuid, path, methodToUse, parameters, newOptions, optionBody, bodyIsCloudObject},
        If[OptionValue[UseUUID] === True,
            {cloud, uuid} = getCloudAndUUID[obj];
            If[!UUIDQ[uuid], Return[handleBadObjectUUIDForExecute[uuid, obj]]];
            log["Executing on UUID `1`", uuid];
            execute[cloud, method, {api, uuid, subpath}, options],
        (* else *)
            {cloud, uuid, path} = getCloudAndUUIDOrPath[obj];
            (* uuid can be None *)
            If[uuid === $Failed, Return[{$Failed, $Failed}]];
            If[method === Automatic,
                If[uuid === None,
                    methodToUse = "POST",
                    methodToUse = "PUT"
                ]
            ];
            parameters = OptionValue[Parameters];
            optionBody = OptionValue[Body];
            bodyIsCloudObject = Head[optionBody] === CloudObject;
            If[bodyIsCloudObject,
                Module[{srccloud, srcuuid},
                    {srccloud, srcuuid} = getCloudAndUUID[optionBody];
                    If[!UUIDQ[srcuuid], Return[handleBadObjectUUIDForExecute[srcuuid, optionBody]]];
                    parameters = Join[parameters, {"copyContentFrom" -> srcuuid}];
                ]
            ];
            newOptions = Join[If[bodyIsCloudObject, {Body -> {}}, {}], {options}];
            If[uuid === None,
                execute[cloud, methodToUse, {api, subpath}, Parameters -> Join[parameters, {"path" -> JoinURL[path]}], newOptions],
                execute[cloud, methodToUse, {api, uuid, subpath}, Parameters -> parameters, newOptions]
            ]
        ]
    ]

contentTypeCheck[type_] := Replace[type, Except[_String] -> "text/plain"]

responseToString[{type_, content_List}, head_] := FromCharacterCode[content, "UTF-8"]
responseToString[{type_, content_String}, head_] := content
responseToString[response_, head_] := $Failed /; checkError[response, head]
responseToString[response_] := responseToString[response, CloudObject]

responseToExpr[response_] := Replace[responseToString[response], r_String :> ToExpression[r]]

responseToStringList[response_, head_] := StringSplit[responseToString[response, head], "\n"]
responseToStringList[response_, head_] := $Failed /; checkError[response, head]

dumpBinary::usage = "dumpBinary[file, contents] writes binary contents to file."

dumpBinary[filename_, contents_String] := 
    (* We treat the contents as a "byte string", meaning it's already been encoded from characters 
        to bytes and stored in the string; hence we force the ISO8859-1 encoding here when 
        converting from string to ByteArray to preserve each code point as a byte value. *)
    dumpBinary[filename, StringToByteArray[contents, "ISO8859-1"]]

(* contents is expected to be either a list of bytes or a ByteArray *)
dumpBinary[filename_, contents_] := 
	With[{ostream = OpenWrite[filename, BinaryFormat -> True]},
	    BinaryWrite[ostream, contents];
	    Close[ostream];
	]

responseToFile::usage = "responseToFile[{type, content}, head] writes content to a temporary file and returns {tempfile, type}. The content can be a list of bytes, a string, or a ByteArray."

responseToFile[{type_, content:(_List | _String | _ByteArray)}, head_:CloudObject] := 
Module[{tempfilename = CreateTemporary[]},
    dumpBinary[tempfilename, content];
    {tempfilename, type}
]

responseToFile[response_, head_:CloudObject] := {$Failed, $Failed} /; checkError[response, head]

responseCheck[response_, head_, result_] :=
    If[checkError[response, head],
        $Failed,
        result
    ]
responseCheck[response_, head_ : CloudObject] := responseCheck[response, head, Null]

End[]

EndPackage[]
