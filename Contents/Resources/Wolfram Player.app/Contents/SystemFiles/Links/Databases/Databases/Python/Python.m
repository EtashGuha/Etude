(* Wolfram Language package *)
Package["Databases`Python`"]


PackageImport["Databases`"] (* DatabaseReference, RelationalDatabase *)
PackageImport["Databases`Common`"] (* DBRaise *)
PackageImport["PacletManager`"]


PackageExport["DBRunPython"]
PackageExport["$PythonProcess"]
PackageExport["DBKeepListening"]
PackageExport["DBClearPythonProcessCache"]

DBClearPythonProcessCache[] := Internal`DeleteCache[i$PythonProcess]


getResourceSettings[] := If[TrueQ[$DBResourceObject["Installed"]],
    <|
        "OraclePath" -> Replace[
            And[
                $DBResourceObject["Installed"],
                $DBResourceObject["PropertyExists"]["Paths", "Oracle"],
                $DBResourceObject["Paths", "Oracle"]
            ], 
            False -> None
        ]
    |>,
    (* else *)
    <| 
        "OraclePath" -> None 
    |>
]


getEnvironment[s_Association?AssociationQ] := getEnvironment[$SystemID, getResourceSettings[]]

getEnvironment["Windows-x86-64", s_Association?AssociationQ] :=
    Replace[
        s["OraclePath"],
        { 
            None -> <||>, 
            (* 
            ** It is important to put the Oracle-related libs location on PATH
            ** for Windows. Formally this may seem to not be required, since the 
            ** cx_Oracle lib is called from the same folder, and so this folder 
            ** is on Windows lib search path. However, if in any other place on
            ** PATH there is a different (older) version of Oracle libs, those 
            ** might be used instead, which causes a version error. With the 
            ** Oracle resource folder path prepended to PATH, this will not 
            ** happen.
            *)
            path_ :> <|"PATH" -> path|>
        }
    ]

getEnvironment[sid_String, _] := <||>

DefError @ getEnvironment


setEnvironment["PATH" -> path_String] := With[{currentPath = Environment["PATH"]},
    If[!StringContainsQ[currentPath, path],
        SetEnvironment["PATH" -> StringJoin[
            path, 
            If[StringMatchQ[$SystemID, "Windows"~~___], ";", ":"], 
            currentPath
        ]]
    ]
]

setEnvironment[key:Except["PATH"] -> value_] := SetEnvironment[key -> value]

setEnvironment[env_Association?AssociationQ] := Scan[setEnvironment, Normal @ env]

DefError @ setEnvironment


$PythonDistribution := 
    With[
        {
            location = PacletResource["Databases", "Python"],
            resourceSettings = getResourceSettings[] 
        }, {
            dist  = FileNameJoin @ {
                location,
                If[
                    StringMatchQ[$SystemID, "Windows" ~~ ___],
                    "distribution.exe",
                    "distribution"
                ]
            },
            debug = FileNameJoin @ {location, "run.py"},
            args  = {
                "listener_loop",
                If[TrueQ[$DBDebugMode], "--debug", Nothing],
                "--kernel-version", ToString[$VersionNumber],
                If[
                    resourceSettings["OraclePath"] =!= None
                    ,
                    Sequence @@ {"--extrapath", resourceSettings["OraclePath"]},
                    (* else *)
                    Nothing
                ]
            }
        },
        Which[
            FileExistsQ[dist],
                <| 
                    "Process" -> Join[{dist}, args], 
                    "Environment" -> getEnvironment[resourceSettings]
                |>,
            TrueQ[$DBDebugMode],
                 <| 
                    "Process" -> Join[{"python3", debug}, args], 
                    "Environment" -> getEnvironment[resourceSettings]
                |>,
            True,
                DBRaise @ Failure[
                    "DatabaseFailure", <|
                        "MessageTemplate" :> RelationalDatabase::nvldplatform,
                        "MessageParameters" -> <||>
                    |>
                ]
        ]
    ]

$MaxProcessAge := If[$DBDebugMode, 2, None]

(* 
** TODO: temporary workaround, revisit when ReadByteArray gets fixed 
**
** This function is needed to work around the apparent bug in ReadByteArray, 
** which in some cases leads to blocking, as per bug #368404. Reading just one 
** byte by BinaryRead unblocks ReadByteArray.
*)
readByteArray[p_, 1] := ByteArray[{BinaryRead[p]}]
readByteArray[p_, n_] := Join[
    readByteArray[p, 1],
    ReadByteArray[p, n-1]
]

DefError @ readByteArray


$PythonProcess := 
    With[{pdist = $PythonDistribution},
        setEnvironment[pdist["Environment"]];
        DBCheckAndSetCache[
            i$PythonProcess,
            <|"Process" -> StartProcess @ pdist["Process"], "StartTime" -> UnixTime[]|>,
            (* 
            ** this is invalidating the process cache if the process is dead there is no 
            ** method to check if the process is alive 
            *)
            If[
                NumberQ[$MaxProcessAge],
                Function @ If[ 
                    (* 
                    ** this is for debug, is the process has been alive for more then 
                    ** 1 sec we kill it so that code is reloaded 
                    *)
                    #StartTime + $MaxProcessAge < UnixTime[],
                    KillProcess[#Process];
                    False,
                    ProcessStatus[#Process] === "Running"          
                ],
                Function[ProcessStatus[#Process] === "Running"]        
            ]
        ]
    ]
    

byteArrayPartition[ba_ByteArray, len_:15000] := 
    (* 

        fix for https://bugs.wolfram.com/show?number=366128 
        stdin buffer is limited to few kbs, we need to split the bytearray in several
        chunks and for each message we need to wait for python to clear the buffer

    *)
    With[
        {byteLen = Length[ba]},
        If[
            byteLen <= len,
            (* short circuit, we don't want to re-create the bytearray *)
            {ba},
            Table[Take[ba, {k, UpTo[k + len - 1]}], {k, 1, byteLen, len}]
        ]
    ]



autoraise = Replace @ {
    f:$Failed|_Failure?FailureQ :> DBRaise[DBRunPython, "error_python_data_transfer"]
}

SetAttributes[pythonErrorHandler, HoldRest]
pythonErrorHandler[process_, code_] :=
    Replace[
        CheckAll[code, HoldComplete], {
            (* the python code returned a failure, we need to raise it *)
            _[res_Failure, Hold[]] :> 
                DBRaise @ res,

            (* everything was fine, we return the result*)
            _[res_, Hold[]] :> 
                res,

            (* Abort[] or Throw should just be propagated. in any case the error was 
               thrown in the middle of an IO operation, we cannot trust the process 
               to work on the next evaluation. we must 🔪 the process.
            *)
            _[res_, Hold[abort_]] :> (
                KillProcess[process]; 
                abort;
                (*
                ** If 'abort' is an Abort[] or exception, the line below will not
                ** fire. But if for whatever reason it is something else, we raise 
                ** an exception, since at this point anything here should mean 
                ** that we've got an error, and it should not pass silently. 
                *)
                DBRaise[pythonErrorHandler, "unhandled_error", Unevaluated @ {res, abort}]
            )
        }
    ]

(* constants for comunication *)
$padding    = 12
$okMsg      = ByteArray[{75}] (* K *)
$appendMode = ByteArray[{65}] (* A *)
$writeMode  = ByteArray[{87}] (* W *)
    
Options[DBRunPython] := {
    Path           -> None,
    Authentication -> None,
    "Schema"       -> None
}

DBRunPython[expr_] :=
    With[{
        process  = $PythonProcess["Process"], 
        messages = byteArrayPartition @ BinarySerialize[expr]
        },

        (*  
            Aborting the computation now would not kill the process 
            started with RunProcess this will result in an invalid state where 
            there is already data waiting to be read from stdout.

            If a Failure is returned at any point of this code, 
            autoraise will raise it and pythonErrorHandler will kill the process,.

            https://bugs.wolfram.com/show?number=363512
        *)

        pythonErrorHandler[
            process,
            (* 

                WRITE LOOP. 
                we serialize the payload using BinarySerialize and we split the message in chunks 
                that can be contained by stdin

                message format
                'W' or 'A' write or append mode. write is used to specify that the message is ended.
                '0000000123' number of bytes to read ahead. this number is always padded with $padding
                '....' actual bytes

            *)
            Apply[
                Function[
                    {bytes, isLast},
                    autoraise @ BinaryWrite[
                        process, Join[
                            If[isLast, $writeMode, $appendMode],
                            StringToByteArray[
                                StringPadLeft[IntegerString[Length[bytes]], $padding, "0"]
                            ],
                            bytes
                        ]
                    ];
                    If[
                        (* 
                            if this message is not the last we need to wait for python to write $okMsg on stdout
                            which means that stdin has been flushed
                        *)
                        And[
                            Not @ isLast,
                            Not @ autoraise[readByteArray[process, 1]] == $okMsg
                        ],
                        autoraise @ $Failed
                    ];
                ],
                Transpose @ {
                    messages,
                    Append[ConstantArray[False, Length[messages] - 1], True]
                },
                {1}
            ];
            (*
                READ LOOP
                after the message is decoded as WXF the code is evaluated

                at any point python can send a side effect (such as Print[...])
                that side effect is encoded as WXF, wrapped in DBKeepListening 
                and deserialized and evaluated in the kernel.

                if the expression is not DBKeepListening[...] such expression 
                is the result of the computation and is returned.

                if python can return an expression it means that the read/write
                script worked succesfully and we can re-use the interpreter 
                on the next evaluation.

                if this is not the case autoraise will throw a $Failed that is
                catched by pythonErrorHandler that will kill the current process and
                return an error (TODO we should probably log this event somehow).

                after that an eventual python side exception is raised, 
                but this failure is not an IO error which means 
                the interpeter is still comunicating correctly with the kernel.
        
            *)

            NestWhile[
                Function @ BinaryDeserialize[
                    autoraise @ readByteArray[
                        process,
                        FromDigits[ByteArrayToString[readByteArray[process, $padding]]]
                    ]
                ],
                DBKeepListening[],
                MatchQ[_DBKeepListening]
            ]
        ]
    ]

(* 
** the two argument version of DBRunPython is creating an env to use for the 
**command, which is a shortcut for WithEnvironment 
*)
DBRunPython[expr_, OptionsPattern[]] :=
    DBResourceCheckInstall[
        $DBResourceObject,
        Replace[
            OptionValue[Authentication], {
                connection_DatabaseReference :> (connection["Backend"] === "Oracle"),
                _ -> False
            }
        ]
    ] @
    DBRunPython[
        "WithEnvironment"[
            expr, <|
                "connection" -> Replace[
                    OptionValue[Authentication],
                    auth_DatabaseReference :> ReplaceAll[
                        auth[All], {
                            _Missing :> None,
                            f_Failure?FailureQ :> RuleCondition[DBRaise[f]]
                        }
                    ]
                ],
                "metadata"   -> Replace[
                    OptionValue["Schema"],
                    HoldPattern[RelationalDatabase[schema_, ___]] :> <|schema|>
                ],
                "path"       -> Replace[
                    OptionValue[Path],
                    s:_String|_File :> ExpandFileName[s]
                ],
                "timezone"   -> Replace[
                    $TimeZone,
                    t: Except[_Integer|_Real] :> TimeZoneOffset[t]
                ]
            |>
        ]
    ]