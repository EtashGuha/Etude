(* Created with the Wolfram Language : www.wolfram.com *)

BeginPackage["FactualFunctions`"];

FCamelize::usage = "";
FCookedImport::usage = "";
FFormatNutritionFacts::usage = "";
FFormatPlaces::usage = "";
FFormatProducts::usage = "";
FFormatRestaurants::usage = "";
FGetBrackets::usage = "";
FGetFilters::usage = "";
FGetGeoFilter::usage = "";
FGetSortFilter::usage = "";
FImport::usage = "";
FInterval::usage = "";
FTimeObjects::usage = "";
FPaginationCalls::usage = "";

Begin["`Private`"];

FCamelize[string_] :=
    Module[ {split, partial},
        (
        split = StringSplit[string, {" ","_","-"}];
        partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
        StringJoin[partial]
        )
    ]

FCookedImport[res_,outputOptions_,request_] :=
    Module[ {listResult,links},
        listResult = Switch[request,
            "Places",FFormatPlaces[res,outputOptions],
            "Products",FFormatProducts[res,outputOptions],
            "RestaurantUS",FFormatRestaurants[res,outputOptions],
            "NutritionFacts",FFormatNutritionFacts[res,outputOptions]];
        Switch[outputOptions["Output"],
            Dataset,Dataset[Association/@listResult],
            List,listResult,
            "Images",
                (links = Cases[listResult // Normal, HoldPattern["Images" -> x_]:>x, \[Infinity]];
                 Quiet[If[ !MatchQ[#,_Missing],
                           DeleteCases[(Import /@#),$Failed],
                           #
                       ]& /@links])
        ]
    ]

FFormatRestaurants[list_,entitiesQ_] :=
    Module[ {localityList = {},adList = {}(*,
    		 infoProperties = {"name","latitude","longitude","address","$distance",
    		 	               "locality","region","category_labels","website"}*)},
        If[ entitiesQ["InterpretEntities"],
            (
            localityList = Rule[#,Interpreter["City"][#]]&/@DeleteDuplicates@("locality"/.list);
            adList = Rule[#,Interpreter["AdministrativeDivision"][#]]&/@DeleteDuplicates@("region"/.list);
            )
        ];
        {"Name" -> "name" /. #,
        "Location" -> GeoPosition[{"latitude" /. #, "longitude" /. #}],
        "Address" -> ("address" /. #)/."address"->Missing["NotAvailable"],
        "Distance" -> (Quantity["$distance" /. #, "Meters"])/."$distance"->Missing["NotAvailable"],
        "Locality" -> ("locality" /. #)/.Join[{"locality"->Missing["NotAvailable"]},localityList](*(Interpreter["City"]["locality" /. #])/."locality"->Missing["NotAvailable"]*),
        "AdministrativeDivision" ->  ("region" /. #)/.Join[{"region"->Missing["NotAvailable"]},adList],
        (*(Interpreter["AdministrativeDivision"]["region" /. #])/."region"->Missing["NotAvailable"],*)
        "Categories"->("category_labels"/.#)/."category_labels"->Missing["NotAvailable"],
        "WebSite" -> ("website"/.#)/."website"->Missing["NotAvailable"],
        "Phone" -> ("tel" /. #)/."tel"->Missing["NotAvailable"],
        "HourDisplay" -> ("hours_display" /. #)/."hours_display"->Missing["NotAvailable"],
        "Hours" -> (FTimeObjects["hours" /. #])/."hours"->Missing["NotAvailable"]} & /@list
    ]

FFormatPlaces[list_,entitiesQ_] :=
    Module[ {localityList = {},adList = {},preformated,distribution},
        If[ entitiesQ["InterpretEntities"],
            (
            localityList = Rule[#,Interpreter["City"][#]]&/@DeleteDuplicates@("locality"/.list);
            adList = Rule[#,Interpreter["AdministrativeDivision"][#]]&/@DeleteDuplicates@("region"/.list);
            )
        ];
        preformated={"Name" -> "name" /. #,
        "Location" -> GeoPosition[{"latitude" /. #, "longitude" /. #}],
        "Address" -> ("address" /. #)/."address"->Missing["NotAvailable"],
        "Distance" -> ("$distance" /. #) /."$distance"->Missing["NotAvailable"],
        "Locality" -> ("locality" /. #)/.Join[{"locality"->Missing["NotAvailable"]},localityList](*(Interpreter["City"]["locality" /. #])/."locality"->Missing["NotAvailable"]*),
        "AdministrativeDivision" ->  ("region" /. #)/.Join[{"region"->Missing["NotAvailable"]},adList],
        (*(Interpreter["AdministrativeDivision"]["region" /. #])/."region"->Missing["NotAvailable"],*)
        "Categories"->("category_labels"/.#)/."category_labels"->Missing["NotAvailable"],
        "WebSite" -> ("website"/.#)/."website"->Missing["NotAvailable"],
        "Phone" -> ("tel" /. #)/."tel"->Missing["NotAvailable"],
        "HourDisplay" -> ("hours_display" /. #)/."hours_display"->Missing["NotAvailable"],
        "Hours" -> (FTimeObjects["hours" /. #])/."hours"->Missing["NotAvailable"]} & /@list;
        distribution=Association[Rule @@@Tally[preformated[[All,4]][[All,2]]]];
        If[distribution[Missing["NotAvailable"]]===Length[preformated],
        	Drop[preformated,None,{4}],
          preformated[[All, 4]] = ("Distance" -> Quantity[#[[2]], "Meters"] &) /@ preformated[[All, 4]];
          preformated
        ]
    ]

FFormatProducts[list_,entitiesQ_] :=
    Module[ {manufacturerList = {}},
        If[ entitiesQ["InterpretEntities"],
            (
            manufacturerList = ReplaceAll[(Rule[#, Interpreter["Company"][#]] &)/@DeleteCases[DeleteDuplicates["manufacturer" /. list],"manufacturer"], (x_ -> _Failure) :> x -> x];
            )
        ];
        {"ProductName" -> "product_name" /. #,
        "UPC-A" -> (("upc" /. #)/."upc"->Missing["NotAvailable"]),
        "UPC-E" -> (("upc_e" /. #)/."upc_e"->Missing["NotAvailable"]),
        "EAN13" -> (("ean13" /. #)/."ean13"->Missing["NotAvailable"]),
        "Category" -> ("category" /. #)/."category"->Missing["NotAvailable"],
        "Brand" -> ("brand" /. #)/."brand"->Missing["NotAvailable"],
        "Manufacturer" -> ("manufacturer" /. #)/.Join[{"manufacturer"->Missing["NotAvailable"]},manufacturerList],
        "AVGPrice" -> (("avg_price"/.#)/."avg_price"->Missing["NotAvailable"])/.x_?NumericQ:>Quantity[x,"Dollars"],
        "Size" -> ("size" /. #)/."size"->Missing["NotAvailable"](*SemanticInterpretation/@(("size" /. #)/.{}->Missing["NotAvailable"])*),
        "Images"->("image_urls"/.#)/."image_urls"->Missing["NotAvailable"]} & /@list
    ]

FFormatNutritionFacts[list_,entitiesQ_] :=
    Module[ {data,nutritionList,generalInfoList,nutritionListwithCamelTitles,manufacturerList = {},
        ginfoList = {"product_name","upc","upc_e","ean13","category","brand","factual_id","manufacturer","avg_price","size","image_urls"}},
        If[ entitiesQ["InterpretEntities"],
            (
            manufacturerList = (Rule[#,Interpreter["Company"][#]]&/@DeleteDuplicates@("manufacturer"/.list));
            )
        ];
        nutritionList = Complement[#, Rule @@@ Transpose@{ginfoList, ginfoList /. #}]&/@list;
        nutritionListwithCamelTitles = Replace[nutritionList, Rule[a_, b_] :> Rule[FCamelize[a], b], Infinity];
        generalInfoList = {"ProductName" -> "product_name" /. #,
        "UPC-A" -> (("upc" /. #)/."upc"->Missing["NotAvailable"]),
        "UPC-E" -> (("upc_e" /. #)/."upc_e"->Missing["NotAvailable"]),
        "EAN13" -> (("ean13" /. #)/."ean13"->Missing["NotAvailable"]),
        "Category" -> ("category" /. #)/."category"->Missing["NotAvailable"],
        "Brand" -> ("brand" /. #)/."brand"->Missing["NotAvailable"],
        "Manufacturer" -> ("manufacturer" /. #)/.Join[{"manufacturer"->Missing["NotAvailable"]},manufacturerList],
        "AVGPrice" -> (("avg_price"/.#)/."avg_price"->Missing["NotAvailable"])/.x_?NumericQ:>Quantity[x,"Dollars"],
        "Size" -> ("size" /. #)/."size"->Missing["NotAvailable"](*SemanticInterpretation/@(("size" /. #)/.{}->Missing["NotAvailable"])*),
        "Images"->("image_urls"/.#)/."image_urls"->Missing["NotAvailable"]} & /@list;
        data = (Rule @@@ Transpose[{{"GeneralInfo", "NutritionFacts"}, #}]) & /@Transpose[{generalInfoList,nutritionListwithCamelTitles}];
        Replace[#,List[a__Rule] :> Association[a],Infinity]&/@data
    ]

FImport[$Failed] :=
    Throw[$Failed]

FImport[json_String] :=
    Module[ {res = ImportString[json,"JSON"],status},
        If[ res===$Failed,
            Throw[$Failed]
        ];
        status = "status"/.res;
        If[ StringMatchQ[status,"ok"],
            ("data"/.("response"/.res)),
            Message[ServiceExecute::apierr,"message"/.res];
            Throw[$Failed]
        ]
    ]

FInterval[list_] :=
    Interval[TimeObject /@ #] & /@ list;

FTimeObjects[list_] :=
    {FCamelize[#[[1]]] -> FInterval[#[[2]]]}&/@list;

FGetBrackets[list_] :=
    "[\"" <> StringJoin @@ Riffle[ToString /@ list, "\",\""] <> "\"]";

FGetGeoFilter[params_Association] :=
    "{\"$circle\":{\"$center\":"<>
    FGetBrackets[GeoPosition[params["Location"]][[1]]]<>",\"$meters\":" <>
    ToString@Round@QuantityMagnitude@UnitConvert[params["Radius"], "Meters"] <> "}}"

FGetFilters[params_Association] :=
    Module[ {country,region,locality,and,state=params["Region"]["StateAbbreviation"], city=params["Locality"]["Name"],json},
        country={{"country" -> {"$eq" -> "US"}}};
        region=If[!MatchQ[state, Missing[__]["StateAbbreviation"]],
          {{"region" -> {"$eq" -> ToLowerCase[state]}}},{}
          ];
        locality=If[!MatchQ[city, Missing[__]["Name"]],
          {{"locality" -> {"$eq" -> ToLowerCase[city]}}},{}
          ];
        and = {"$and"->Join[country,region,locality]};
        ExportString[and,"JSON"]
      ]

FGetSortFilter[params_] :=
    If[ MatchQ[params["Sort"],"Distance"],
        "$distance",
        ToLowerCase@params["Sort"]
    ]

FPaginationCalls[id_,prop_,p_,np_] :=
    Module[ {calls,residual,progress=0,rawdata,totalresults,items = {},params = p,newParams = np},
        calls = Quotient[params["MaxItems"], 20];
        residual = params["MaxItems"] - (calls*20);
        PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
        If[ calls > 0,
            (
              (
                rawdata = FImport[KeyClient`rawkeydata[id,prop,newParams]];
                totalresults = Length@rawdata;
                items = Join[items, rawdata];
                progress = progress +1;
                params["StartIndex"] = (#+1)*20+1;
                newParams = newParams/.("offset"->__)->"offset"->ToString@params["StartIndex"]
              )& /@ Range[0,calls-1]
            )
        ];
        If[ residual > 0,
            (
              params["MaxItems"] = residual;
              newParams = newParams/.("offset"->__)->"offset"->ToString[params["StartIndex"]];
              newParams = newParams/.("limit"->__)->"limit"->ToString@params["MaxItems"];
              rawdata = FImport[KeyClient`rawkeydata[id,prop,newParams]];
              totalresults = Length@rawdata;
              items = Join[items, rawdata]
            )
        ];
        items
    ]

End[];

EndPackage[];
