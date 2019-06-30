
BeginPackage["TypeFramework`Utilities`PrenexNormalForm`"]

ToPrenexNormalForm

Begin["`Private`"] 

Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]


(********************************************************)

ToPrenexNormalForm[t_] := toPrenexNormalForm[<||>, t, <||>] 

toPrenexNormalForm[st_, ts_?ListQ, opts_?AssociationQ] :=
    toPrenexNormalForm[st, #, opts]& /@ ts;

toPrenexNormalForm[st_, t_?TypeApplicationQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeApplication[
            toPrenexNormalForm[st, t["type"], opts],
            toPrenexNormalForm[st, t["arguments"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ]
    
toPrenexNormalForm[st_, t_?TypeQualifiedQ, opts_?AssociationQ] :=
    With[{
        body = toPrenexNormalForm[st, t["type"], opts]
    },
    With[{
        res = CreateTypeQualified[
            t["predicates"],
            body
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ]]
    
    
toPrenexNormalForm[st_, t_?TypeForAllQ, opts_?AssociationQ] :=
    With[{
        qual = t["type"]
    },
    With[{
        body = toPrenexNormalForm[st, qual["type"], opts]
    },
        If[TypeForAllQ[body] && Intersection[t["variables"], body["variables"], SameTest -> sameQ] === {},
            CreateTypeForAll[
                Join[t["variables"], body["variables"]],
                CreateTypeQualified[
                    Join[qual["predicates"], body["type"]["predicates"]],
                    body["type"]["type"]
                ]
            ]
            , (* Else *)
            t            
        ]
    ]];
toPrenexNormalForm[st_, t_?TypeArrowQ, opts_?AssociationQ] :=
    With[{
        fvs = Values[
            Join @@ Map[#["free"]&, t["arguments"]]
        ],
        theta2 = toPrenexNormalForm[st, t["result"], opts]
    },
        If[TypeForAllQ[theta2] && Intersection[theta2["variables"], fvs, SameTest -> sameQ] === {},
            CreateTypeForAll[
                theta2["variables"],
                CreateTypeArrow[
                    t["arguments"],
                    theta2["type"]
                ]
            ]
            , (* Else *)
            t
        ]
    ];
toPrenexNormalForm[st_, t_?TypeObjectQ, opts_] :=
    t;

toPrenexNormalForm[args___] :=
    TypeFailure["TypePrenexNormalForm", "Unrecognized call to ToPrenexNormalForm", args]



sameQ[a_, b_] := TrueQ[a["sameQ", b]]

End[];

EndPackage[];
