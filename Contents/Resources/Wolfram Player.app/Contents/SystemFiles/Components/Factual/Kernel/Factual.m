Begin["FactualAPI`"]

ServiceConnect::disc = "Due to restrictions added by Factual, this service is currently not available."

Begin["`Private`"]

(******************************* Factual *************************************)

factualdata[___]:=(
ServiceConnections`Private`$keyservices=DeleteCases[ServiceConnections`Private`$keyservices,"Factual"];
Message[ServiceConnect::disc,"Factual"];Throw[$Failed])

factualcookeddata[___]:=(Message[ServiceConnect::disc,"Factual"];Throw[$Failed])

factualsendmessage[___]:=(Message[ServiceConnect::disc,"Factual"];Throw[$Failed])

(* Authentication information *)

factualdata[]:= {
        "ServiceName"         -> "Factual",
        "URLFetchFun"        :> (With[ {params = Lookup[{##2},"Parameters",{}]},
                                     URLFetch[#1,
                                         Sequence@@FilterRules[{##2},Except["Parameters"]],
                                             "Parameters" -> params
                                             ]
                                 ]
                                &),
        "ClientInfo"        :> OAuthDialogDump`Private`MultipleKeyDialog["Factual",{"Key"->"KEY"},"https://www.factual.com/contact/new#free_api_access","http://www.factual.com/tos"],
        "Gets"                -> {"Places","Products","NutritionFacts","PlacesDataset","PlacesList"},
        "Posts"                -> {},
        "RawGets"            -> {"RawPlaces","RawProducts","RawNutritionFacts"},
        "RawPosts"            -> {},
        "Information"        -> "Factual connection for WolframLanguage"
}

(* Raw *)
factualdata["RawPlaces"] :=
    {
    "URL"                -> "http://api.v3.factual.com/t/places",
    "HTTPSMethod"        -> "GET",
    "Parameters"         -> {"geo","filters","sort","q","offset","limit"},
    "RequiredParameters"-> {},
    "ResultsFunction"    -> FImport
    }

factualdata["RawProducts"] :=
    {
    "URL"                -> "http://api.v3.factual.com/t/products-cpg",
    "HTTPSMethod"        -> "GET",
    "Parameters"        -> {"filters","sort","q","offset","limit"},
    "RequiredParameters"-> {"q"},
    "ResultsFunction"    -> FImport
    }

factualdata["RawRestaurantsUS"] :=
    {
    "URL"                -> "http://api.v3.factual.com/t/restaurants-us",
    "HTTPSMethod"        -> "GET",
    "Parameters"        -> {"geo","filters","sort","q","offset","limit"},
    "RequiredParameters"-> {"geo"},
    "ResultsFunction"    -> FImport
    }

factualdata["RawNutritionFacts"] :=
    {
    "URL"                -> "http://api.v3.factual.com/t/products-cpg-nutrition",
    "HTTPSMethod"        -> "GET",
    "Parameters"        -> {"filters","sort","q","offset","limit"},
    "RequiredParameters"-> {"q"},
    "ResultsFunction"    -> FImport
    }

(* Cooked *)

factualcookeddata["Places", id_,args_] :=
    Block[ {params,rawdata,geo,newParams = {},filters,filtersoptions,sortingoptions,rows,outputOptions = <|"Output" -> Dataset, "InterpretEntities" -> False|>,calls,residual,progress,totalresults,items = {},invalidParameters},
        params = Association[args];
        invalidParameters = Select[Keys@params,!MemberQ[{"Output","Query","MaxItems",MaxItems,"Location","Locality","Region","Radius","Sort"},#]&];
        If[Length[invalidParameters]>0,
    		(
    			Message[ServiceObject::noget,#,"Factual"]&/@invalidParameters;
    			Throw[$Failed]
    		)];
        params = Join[params,<|"StartIndex"->0|>];
        newParams = Join[newParams,{"offset"->"0","limit"->"20"}];

        If[ !Xor[KeyExistsQ[params,"MaxItems"],KeyExistsQ[params,MaxItems]],
            params = Join[params,<|"MaxItems"->20|>]
        ];
        If[ KeyExistsQ[params,MaxItems],
            params = KeyMap[# /. (MaxItems -> "MaxItems") &, params]
        ];
        If[ KeyExistsQ[params,"Output"],
            outputOptions = KeyTake[params,{"Output","InterpretEntities"}]
        ];

        If[ Or[KeyExistsQ[params,"Region"],KeyExistsQ[params,"Locality"]],
          filtersoptions = FGetFilters[params];
          newParams = Join[newParams,{"filters"->filtersoptions}]
        ];

        If[ KeyExistsQ[params,"Query"],
            newParams = Join[newParams,{"q"->params["Query"]}]
        ];

        If[ KeyExistsQ[params,"Location"],
            If[ And[IntegerQ[Round@QuantityMagnitude@UnitConvert[params["Radius"],"Meters"]],
            MatchQ[GeoPosition[params["Location"]], GeoPosition[{_?NumericQ, _?NumericQ}]]],
              geo = FGetGeoFilter[params];
              newParams = Join[newParams,{"geo"->geo}];
              If[!KeyExistsQ[params,"Sort"],
                 params = Join[params,<|"Sort"->"Distance"|>]
              ];
              sortingoptions = FGetSortFilter[params];
              newParams = Join[newParams,{"sort"->sortingoptions}],
              Throw[$Failed]
            ]
        ];


        items = FPaginationCalls[id,"RawPlaces",params,newParams];
        FCookedImport[items,outputOptions,"Places"]
    ]

factualcookeddata["PlacesDataset", id_,args_] :=
    Module[ {newParams = Join[args,
        {"Output"->Dataset}]},
        factualcookeddata["Places", id,newParams]
    ]

factualcookeddata["PlacesList", id_,args_] :=
    Module[ {newParams = Join[args,
        {"Output"->List}]},
        factualcookeddata["Places", id,newParams]
    ]

factualcookeddata["Products", id_,args_] :=
    Block[ {
            params,rawdata,geo,newParams = {},filters,rows,
            outputOptions = <|"Output" -> Dataset, "InterpretEntities" -> False|>,
            calls,residual,progress,totalresults,items = {}
            },
        params = Association[args];
        params = Join[params,<|"StartIndex"->0|>];
        If[ !KeyExistsQ[params,"MaxItems"],
            params = Join[params,<|"MaxItems"->20|>]
        ];
        If[ KeyExistsQ[params,"Output"],
            outputOptions = KeyTake[params,{"Output","InterpretEntities","Elements"}]
        ];
        If[ KeyExistsQ[params,"Query"],
            newParams = Join[newParams,{"q"->params["Query"]}]
        ];
        newParams = Join[newParams,{"offset"->"0","limit"->"20"}];
        items = FPaginationCalls[id,"RawProducts",params,newParams];
        FCookedImport[items,outputOptions,"Products"]
    ]

factualcookeddata["RestaurantsUS ", id_,args_] :=
    Block[ {params,rawdata,geo,newParams,filters,sortingoptions,rows,outputOptions = <|"Output" -> Dataset, "InterpretEntities" -> False|>,calls,residual,progress,totalresults,items = {}},
        params = Association[args];
        params = Join[params,<|"StartIndex"->0|>];
        If[ !KeyExistsQ[params,"MaxItems"],
            params = Join[params,<|"MaxItems"->20|>]
        ];
        If[ KeyExistsQ[params,"Output"],
            outputOptions = KeyTake[params,{"Output","InterpretEntities"}]
        ];
        If[ And[
            IntegerQ[Round@QuantityMagnitude@UnitConvert[params["Radius"],"Meters"]],
            MatchQ[GeoPosition[params["Location"]], GeoPosition[{_?NumericQ, _?NumericQ}]]
            ],
            ( geo = FGetGeoFilter[params];
              newParams = {"geo"->geo};
              If[ KeyExistsQ[params,"Query"],
                  newParams = Join[newParams,{"q"->params["Query"]}]
              ];
              If[ KeyExistsQ[params,"Sort"],
                  newParams = Join[newParams,{"sort"->params["Sort"]}],
                  params = Join[params,<|"Sort"->"Distance"|>]
              ];

              sortingoptions = FGetSortFilter[params];
              newParams = If[ KeyExistsQ[Association@newParams,"sort"],
                              newParams/.("sort"->__)->"sort"->sortingoptions,
                              Join[newParams,{"sort"->sortingoptions}]
                          ];
              newParams = Join[newParams,{"offset"->"0","limit"->"20"}];
              items = FPaginationCalls[id,"RawRestaurantsUS",params,newParams];
              FCookedImport[items,outputOptions,"RestaurantsUS"]
            ),
            Throw[$Failed]
        ]
    ]

factualcookeddata["NutritionFacts", id_,args_] :=
    Block[ {
            params,rawdata,geo,newParams = {},filters,sortingoptions,rows,
            outputOptions = <|"Output" -> Dataset, "InterpretEntities" -> False|>,
            calls,residual,progress,totalresults,items = {}
            },
        params = Association[args];
        params = Join[params,<|"StartIndex"->0|>];
        If[ !KeyExistsQ[params,"MaxItems"],
            params = Join[params,<|"MaxItems"->20|>]
        ];
        If[ KeyExistsQ[params,"Output"],
            outputOptions = KeyTake[params,{"Output","InterpretEntities"}]
        ];
        If[ KeyExistsQ[params,"Query"],
            newParams = Join[newParams,{"q"->params["Query"]}]
        ];
        newParams = Join[newParams,{"offset"->"0","limit"->"20"}];
        items = FPaginationCalls[id,"RawNutritionFacts",params,newParams];
        FCookedImport[items,outputOptions,"NutritionFacts"]
    ]

factualcookeddata[___] :=
    $Failed

factualsendmessage[___] :=
    $Failed


(* Utilities *)
getallparameters[str_] :=
    DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.factualdata[str]],
    ("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

{FactualAPI`Private`factualdata,FactualAPI`Private`factualcookeddata,FactualAPI`Private`factualsendmessage}
