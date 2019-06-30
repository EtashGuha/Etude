
Unprotect[System`WebImageSearch];
Clear[System`WebImageSearch];

BeginPackage["WebSearch`"]

Begin["`Private`"] (* Begin Private Context *)

System`WebImageSearch::nomethod = "The method `1` is not available for WebImageSearch."
System`WebImageSearch::invopt = "`1` is not a valid value for option `2`.";
System`WebImageSearch::invoptws = "The option `1` is not available for WebImageSearch."
System`WebImageSearch::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";
System`WebImageSearch::notauth = "WebImageSearch requests are only valid when authenticated. Please try authenticating again.";

ISServiceName[name_]:=
	Switch[name,
		"Google"|"GoogleCustomSearch","GoogleCustomSearch",
		"Bing"|"BingSearch","BingSearch",
(*		"Flickr"|"flickr","Flickr", *)
		_, Message[System`WebImageSearch::nomethod, name]; Throw["exception"]
    ]


Options[WebImageSearch] = SortBy[{
    Method -> "Bing",
    "MaxItems"-> 10,
		MaxItems-> 10,
    "Site"-> Null,
    "Description"-> Null,
    "Country"-> Null,
    "Location"-> Null,
    "Elements"-> Null,
    "ImageFilters"-> Null,
    Language-> $Language,
    AllowAdultContent-> False,
		SortedBy->None
  },(ToString[#[[1]]]&)];

System`WebImageSearch[___] := (Message[System`WebImageSearch::offline]; $Failed)/;(!PacletManager`$AllowInternet)

System`WebImageSearch[args___]:= With[{connected = CloudConnect[]},
    If[$CloudConnected,
        System`WebImageSearch[args],
        Message[System`WebImageSearch::notauth];
        connected
    ]
]/; !$CloudConnected

System`WebImageSearch[args__]:= With[
	{res = Catch[WebImageSearch1[args]]},

	res /; !MatchQ[res, "exception"]
]

WebImageSearch1[query_,elem_String:"Elements",maxItems_Integer:10,opt : OptionsPattern[WebImageSearch]] :=
	Block[ {engine,requestParameters,response,rParams,type,query2,opt2 = List@opt,invalidParameters,result, sf},
		invalidParameters = Select[Keys@opt2,!MemberQ[{Method,MaxItems,SortedBy, "MaxItems", "Site", "Description", "Country", "Location", "Elements", "ImageFilters", Language, AllowAdultContent}, #]&];
		If[ Length[invalidParameters]>0,
        Message[System`WebImageSearch::invoptws,#]&/@invalidParameters;
        Throw["exception"]
		];
    If[ !MemberQ[{"Thumbnail", "Thumbnails", "Image", "Images", "PageTitle", "PageTitles", "ImageHyperlink", "ImageHyperlinks","PageHyperlink", "PageHyperlinks", "FileFormat", "FileFormats", "Elements"}, elem],
        WSReturnMessage[Symbol["WebImageSearch"], "nval2", "Elements"];
      	Throw["exception"]
    ];
		engine = ISServiceName[OptionValue[Method]];
		query2 = WebSearch`WSGetFormattedQuery[query,engine, Symbol["WebImageSearch"]];
		If[ !KeyExistsQ[opt2, AllowAdultContent],
		opt2 = Join[opt2, {"ContentFiltering"->"High"}],
		opt2 = ReplaceAll[opt2, (AllowAdultContent->x_):>("ContentFiltering"->(x/.{False->"High",True->"Off",
										 val_ :> (Message[System`WebImageSearch::invopt, val, AllowAdultContent]; Throw["exception"])}))];
		];

		If[ !KeyExistsQ[opt2, MaxItems],
				opt2 = Join[opt2, {MaxItems->maxItems}]
		];

		sf = Association[opt2][SortedBy];
		opt2 = Normal@KeyDrop[opt2,SortedBy];

		result = Switch[engine,
			"BingSearch",
			(
			Get["BingSearch`"];
			requestParameters = BingSearchFunctions`BSFormatRequestParameters[FilterRules[Join[{opt2},{"Query"->query2,"SearchType"->"Image"}],Except[Method]]];
			response = BSPaginationCalls[Symbol["WebImageSearch"], requestParameters,True];
			If[!StringMatchQ[elem,"Elements"],
				opt2 = Join[{"Elements"->elem},{opt}]
			];
			If[ !MatchQ[response, False | $Failed],
				BingSearchFunctions`BSCookedImport[response,requestParameters,opt2],
				Return[$Failed]
			]
			),
			"GoogleCustomSearch",
			(
			Get["GoogleCustomSearch`"];
			rParams = GoogleCustomSearchFunctions`GCSFormatRequestParameters[FilterRules[Join[{opt2},{"Query"->query2,"SearchType"->"Image"}],Except[Method]]];
			{requestParameters,type} = rParams;
			response = GCSPaginationCalls[Symbol["WebImageSearch"], requestParameters,True];
			If[KeyExistsQ[requestParameters,"Elements"],
				If[MatchQ[Lookup[requestParameters,"Elements"],Default],
				requestParameters=KeyDrop[requestParameters,"Elements"]
				]
			];
			If[!StringMatchQ[elem,"Elements"],
				requestParameters= Join[requestParameters,{"Elements"->elem}]
			];
			If[ !MatchQ[response, False | $Failed],
				GoogleCustomSearchFunctions`GCSCookedImport[response,Join[Normal@requestParameters,{opt}],type],
				Throw["exception"]
			]
			)
		];

		If[!MissingQ[sf],SortBy[result, sf],result]

	]

WebImageSearch1[___]:=Throw["exception"]

SetAttributes[System`WebImageSearch,{Protected, ReadProtected}];
End[]; (* End Private Context *)
EndPackage[];
