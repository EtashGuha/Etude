(* Wolfram Language package *)

Package["Databases`Schema`"]

PackageImport["Databases`"] (* DatabaseReference, RelationalDatabase *)
PackageImport["PacletManager`"]

PackageScope["getTypes"]
PackageScope["getTypeInfo"]
PackageScope["validateType"]

(* 
    Types are curated on the python side $Types is reading a dump that can be generated using 

    python run.py create_type_dump

    or 

    RunProcess[
        {"python", FileNameJoin[{PacletResource["Databases", "Python"], "run.py"}],
        "create_data_dump"}
    ]

*)

$types  = BinaryDeserialize[
    ReadByteArray @ FileNameJoin @ {
        PacletResource["Databases", "Data"], 
        "Types.wxf"
    }
]

SetAttributes[makeFailure, HoldFirst]
makeFailure[message_, params_] := 
    Failure[
        "DatabaseFailure", <|
            "MessageTemplate"   :> message,
            "MessageParameters" :> params,
            "FailureCode" :> "ValidationError"
        |>
    ]


getTypes[] := Lookup[$types, "wolfram"]
getTypes[backend_String] := Lookup[$types, ToLowerCase[backend]]
getTypes[schema : _RelationalDatabase] := getTypes[DatabaseReference[schema]]
getTypes[conn : $DBDatabaseReferencePattern] := getTypes[conn["Backend"]]
getTypes[_] := <||>

getTypeInfo[t_, info: _RelationalDatabase | $DBDatabaseReferencePattern | _String] :=
    getTypeInfo[t, getTypes[info]] 


getTypeInfo[type_String, info_Association: <||>] :=
    FirstCase[
        Lookup[{info, getTypes[]}, type],
        Except[_Missing],
        makeFailure[RelationalDatabase::typenvld, {type}]
    ]

getTypeInfo[{type_String, ___}, info_Association: <||>] :=
    getTypeInfo[type, info]

getTypeInfo[type_, ___] := makeFailure[RelationalDatabase::typenvld, {type}]

validateType[type_String, rest___] :=
    validateType[{type}, rest]

validateType[t: {type_String, ___}, rest___] :=
    Catch[
        Replace[
            getTypeInfo[type, rest], {
                res_Association :> With[
                    {l = Length[res["Arguments"]]},
                    If[
                        Length[t] - 1 > l,
                        makeFailure[
                            RelationalDatabase::typeargcount, {
                                type,
                                Length[t] - 1,
                                l
                            }
                        ], 
                        <|
                            "Type" -> res["Type"],
                            MapIndexed[
                                With[
                                    {value = If[
                                        First[#2] > Length[t] - 1,
                                        None,
                                        t[[First[#2] + 1]]
                                    ]},
                                    #Parameter -> Which[
                                        Not[#Required] && MatchQ[value, None | _Missing],
                                        #Default,
                                        Or[
                                            Not[#Required] && MatchQ[value, None | _Missing],
                                            #Test[value]
                                        ],
                                        value,
                                        True,
                                        Throw[#Message[value], validateType]
                                    ]
                                ] &,
                                Values @ res["Arguments"]
                            ]
                        |>
                    ]
                ]
            }
        ],
        validateType
    ]

validateType[type_, ___] := makeFailure[RelationalDatabase::typenvld, {type}]
