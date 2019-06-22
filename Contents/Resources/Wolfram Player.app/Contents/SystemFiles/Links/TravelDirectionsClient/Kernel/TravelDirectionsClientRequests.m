(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["TravelDirectionsClient`"]
(* Exported symbols added here with SymbolName::usage *) 

System`Private`NewContextPath[{"System`","ExternalService`"}];

(Unprotect[#];
 Clear[#];)&/@{
    System`TravelDirections,
    System`TravelDirectionsData,
    System`TravelDistance,
    System`TravelDistanceList,
    System`TravelTime,
    System`TravelMethod
};

System`Quantity;
System`GeoPath;

Begin["GIS`TravelDump`"];
    
(*
    Return nice Quantity magnitudes and taking into account UnitSystem. Avoid calling UnitConvert in the simplest cases.
    Copied from GIS/GeoFunctions.m
*)
GeoUnitConvert[HoldPattern@Quantity[x_, "Meters", opts___], "Metric" | "SI"] :=
    If[ NumericQ[x] && Abs[x]<1000,
        Quantity[x, "Meters", opts],
        Quantity[x/1000, "Kilometers", opts]
    ];
GeoUnitConvert[HoldPattern@Quantity[x_, "Meters", opts___], "SIBase"] :=
    Quantity[x, "Meters", opts];
metersperfoot = 381/1250;
meterspermile = 5280*metersperfoot;
GeoUnitConvert[HoldPattern@ Quantity[x_, "Meters", opts___], "Imperial"] :=
    If[ NumericQ[x] && Abs[x]<=(1000*metersperfoot),
        Quantity[x/metersperfoot , "Feet", opts],
        Quantity[x/meterspermile , "Miles", opts]
    ];
kilometersperfoot = 381/1250/1000;
kilometerspermile = 5280*kilometersperfoot;
GeoUnitConvert[HoldPattern@ Quantity[x_, "Kilometers", opts___], "Imperial"] :=
    If[ NumericQ[x] && Abs[x]<=(1000*kilometersperfoot),
        Quantity[x/kilometersperfoot , "Feet", opts],
        Quantity[x/kilometerspermile , "Miles", opts]
    ];
GeoUnitConvert[HoldPattern@Quantity[x_, "Gallons", opts___], "Imperial"] :=
    Quantity[x, "Gallons", opts];
GeoUnitConvert[q_Quantity, system_] :=
    System`UnitConvert[q, system];

If[$VersionNumber < 10.4,

TimeConvert[t_Quantity] :=
    Module[ {x = UnitConvert[t, MixedRadix["Years", "Months", "Days", "Hours", "Minutes", "Seconds"]]},
        Quantity[Select[x[[1]], # > 0 &], Pick[x[[2]], # > 0 & /@ x[[1]]]]
    ]

,

TimeConvert[t_Quantity] :=
	Module[{x = UnitConvert[t, MixedUnit[{"Years", "Months", "Days", "Hours", "Minutes", "Seconds"}]]},
		Quantity[
			MixedMagnitude@Select[x[[1, 1]], # > 0 &],
			MixedUnit@Pick[x[[2, 1]], # > 0 & /@ x[[1, 1]]]]]

]

TimeConvert[t : Quantity[0|0., u_]] := Quantity[0, u]


(* Decoder for polylines as defined in https://developers.google.com/maps/documentation/utilities/polylinealgorithm
 *)

decodePolyline[encoded_String] := Module[{deltas},
	deltas = With[{x = FromDigits[#, 32]},
		If[OddQ[x], -BitShiftRight[x, 1] - 1, BitShiftRight[x, 1]]
	] & /@ Reverse[BitAnd[ToCharacterCode@StringCases[encoded, RegularExpression["[_-~]*[ -\\^]"]] - 63, BitNot@32], 2];

	Rest@FoldList[Plus, {0, 0}, Partition[deltas / 100000., 2]]
]

pathType = If[$VersionNumber < 10.3, "Rhumb", "TravelPath"];

Options[TravelDirections] = {TravelMethod -> "Driving", UnitSystem :> $UnitSystem};

Options[TravelDirectionsData] = {UnitSystem :> $UnitSystem};

TravelDirections::servermessage = TravelTime::servermessage = TravelDistance::servermessage = TravelDistanceList::servermessage = "Server message: `1`";

cachedKey = Null;
cachedValue = Null;

replaceWithDefault[prop_, rules_, default_] := Replace[prop, Flatten[{rules, _ -> default}]]

callGetDirections[fun_, locations_List, opts_:{}] :=
    Module[ {type, res, key, messages},
        type = Switch[OptionValue[TravelDirections, opts, TravelMethod],
            "Driving", "fastest",
            "Walking", "pedestrian",
            "Biking"|"Bicycle", "bicycle",
            _, Null
        ];
        If[ type == Null,
            Return[$Failed]
        ];

	key={type, locations};
	If[key == cachedKey,
		Return@cachedValue];

	res = ReleaseHold@Internal`MWACompute["MWADirections",
		{"GetDirections", {locations, "Type" -> type, "RouteShapeFormat" -> "Encoded"}},
		"ContextPath"->{"Internal`MWASymbols`","System`","TravelDirectionsClient`Private`"}];
	If[ListQ@res, messages = replaceWithDefault["Messages", res, $Failed] /. WolframAlphaClient`Private`$FromMWARules];
	If[ListQ@res, res = replaceWithDefault["Result", res, $Failed]];
	If[ListQ@messages,
		Scan[(# /. {
			{"noroute", ___} :> Message[fun::noroute, OptionValue[TravelDirections, opts, TravelMethod], locations],
			{"noloc", loc_} :> Message[fun::noloc, loc],
			m_ :> Message[fun::servermessage, m]
		})&, messages];
	];
	If[ListQ@res,
		res = res /. {
			("BoundingBox" -> {a_List, b_List}) :> ("BoundingBox" -> {GeoPosition@a, GeoPosition@b}),
			("Shape" -> encoded_String) :> ("Shape" -> decodePolyline@encoded)
		};
		shape = "Shape" /. res;
		If[ListQ@shape,
			indexes = "RouteIndexes" /. ("Maneuvers" /. ("Legs" /. res));
			res = res /. {
				("Shape" -> _) :> ("Shape" -> Function[l, shape[[#[[1]] ;; #[[2]]]] & /@ l] /@ (indexes + 1)),
				("RouteIndexes" -> {a_, b_}) :> Sequence[
					"StartingPoint" -> shape[[a+1]],
					"EndingPoint" -> shape[[b+1]],
					"Path" -> GeoPath[GeoPosition@shape[[a+1;;b+1]], pathType]
				]
		}];
		res = Association@res;
		cachedValue = res;
		cachedKey = key;
        ];
        res
    ]

travelDirectionsInternal[fun_, locations:{(_String|_Entity|_GeoPosition)..}, opts:OptionsPattern[TravelDirections]] :=
    callGetDirections[fun, locations, {opts}] /. res_Association :> TravelDirectionsData[locations, res, opts]

TravelDirections[locations:{(_String|_Entity|_GeoPosition)..}, opts:OptionsPattern[]] :=
    callGetDirections[TravelDirections, locations, {opts}] /. res_Association :> TravelDirectionsData[locations, res, opts]

TravelDirections[arg_, OptionsPattern[]] := $Failed /; (
	Switch[arg,
		{_,__},
		Message[TravelDirections::arginvll, arg],
		_,
		Message[TravelDirections::arginvlt, arg]
	];
	False
)

TravelDirections[args___] := $Failed /; (System`Private`Arguments[TravelDirections[args], {1, 2}, List, List@args /. {
		{_, "TravelDistance", ___} :> Options@TravelDirectionsData,
		_ -> {}
	}]; False);

TravelDirectionsData[_List, route_Association, defaultOpts___]["TravelDistance", opts:OptionsPattern[TravelDirectionsData]] :=
	GeoUnitConvert[
		Quantity["Distance"/.route,"Kilometers"],
		If[MatchQ[{opts}, {___,_[UnitSystem,_],___}],
			OptionValue[TravelDirectionsData, {opts}, UnitSystem],
			OptionValue[TravelDirections, {defaultOpts}, UnitSystem]
		]
	]

TravelDirectionsData[_List, route_Association, ___]["TravelTime", opts:OptionsPattern[TravelDirectionsData]] :=
	TimeConvert@Quantity[With[{t = "Time"/.route},
		Round[t, Which[
			t > 600,	60,
			t > 120,	15,
			True   ,	1
		]]
	], "Seconds"]

TravelDirectionsData[_List, route_Association, ___]["ManeuverGrid", opts:OptionsPattern[TravelDirectionsData]] :=
    Grid[({
    Text["Narrative"/.#],
    With[ {km = "Distance" /. #},
        If[ km > 0,
            TraditionalForm@GeoUnitConvert[Quantity[SetAccuracy[km, Which[km > 1, 1, km > .1, 4, True, 7]], "Kilometers"], OptionValue[TravelDirectionsData, {opts}, UnitSystem]],
            Null
        ]
    ]
    })& /@ Flatten["Maneuvers"/.("Legs"/.route),1],
    Alignment -> {{Left, "."}}, Frame->All, FrameStyle -> LightGray
    ]

sortRow[row_] :=
    SortBy[row, (#[[1]] /. {
    "Description" -> 1,
    "Distance" -> 2,
    "Time" -> 3,
    "ManeuverType" -> 4,
    "StartingPoint" -> 5,
    "EndingPoint" -> 6
    }) &]

TravelDirectionsData[_List, route_Association, ___]["Dataset", opts:OptionsPattern[TravelDirectionsData]] :=
    Dataset[Association@sortRow@# & /@ (Flatten["Maneuvers"/.("Legs"/.route),1] /. {
    ("Narrative" -> x_) :> ("Description" -> x),
    ("StreetInvolved" -> x_) :> ("Street" -> x),
    ("Distance" -> d_) :> ("Distance" -> GeoUnitConvert[Quantity[d, "Kilometers"], OptionValue[TravelDirectionsData, {opts}, UnitSystem]]),
    ("Time" -> t_) :> ("Time" -> Quantity[t, "Seconds"]),
    (prop:("StartingPoint"|"EndingPoint") -> p_) :> (prop -> GeoPosition@p)
    })]

TravelDirectionsData[l_List, route_Association, ___]["TravelPath"] :=
	GeoPath[GeoPosition@Flatten["Shape" /. route, 1], pathType]

TravelDirectionsData[_List, __]["Properties"] = {
    "Dataset", "ManeuverGrid", "TravelPath",
    "TravelDistance", "TravelTime"
}

TravelDirectionsData[_List, route_Association, ___][prop__] := (
	Message[TravelDirectionsData::invprop, First@List@prop];
	Null /; False
)

TravelDirections[locations:{(_String|_Entity|_GeoPosition)..}, property_, opts:(OptionsPattern@TravelDirections | OptionsPattern@TravelDirectionsData)] :=
    TravelDirections[locations, FilterRules[{opts}, Options@TravelDirections]] /. directions_TravelDirectionsData :> directions[property, Sequence@@FilterRules[{opts}, Options@TravelDirectionsData]]

TravelDirections[locations_, property_, OptionsPattern[]] := $Failed /; (
	Switch[locations,
		{_,__},
		Message[TravelDirections::arginvll, locations],
		_,
		Message[TravelDirections::arginvlt, locations]
	];
	False
)

Options[TravelDistance] = Options@TravelDirections;

TravelDistance[locations:{(_String|_Entity|_GeoPosition)..}, opts:OptionsPattern[]] :=
	Replace[travelDirectionsInternal[TravelDistance, locations, opts],
		td_TravelDirectionsData :> td["TravelDistance"]
	]

TravelDistance[arg_, opts:OptionsPattern[]] := $Failed /; (
	Switch[arg,
		{_,__},
		Message[TravelDistance::arginvll, arg],
		_,
		Message[TravelDistance::arginvlt, arg]
	];
	False
)

TravelDistance[loc1:(_String|_Entity|_GeoPosition), loc2:(_String|_Entity|_GeoPosition), opts:OptionsPattern[]] :=
	TravelDistance[{loc1, loc2}, opts]

TravelDistance[arg1:(_String|_Entity|_GeoPosition), arg2_, opts:OptionsPattern[]] := $Failed /; (
	Message[TravelDistance::arginv, arg2];
	False
)

TravelDistance[arg1:Except[_String|_Entity|_GeoPosition], arg2_, opts:OptionsPattern[]] := $Failed /; (
	Message[TravelDistance::arginv, arg1];
	False
)

TravelDistance[args___] := $Failed /; (System`Private`Arguments[TravelDistance[args], {1, 2}]; False);

Options[TravelDistanceList] = Options@TravelDirections;

TravelDistanceList[{_String|_Entity|_GeoPosition}, opts:OptionsPattern[]] := {}

TravelDistanceList[locations:{(_String|_Entity|_GeoPosition)..}, opts:OptionsPattern[]] := Module[{td},
	td = travelDirectionsInternal[TravelDistanceList, locations, opts];
	If[Head@td =!= TravelDirectionsData,
		Return[td]];
	QuantityArray[GeoUnitConvert[
		Quantity[#,"Kilometers"],
		OptionValue[TravelDirections, {opts}, UnitSystem]
	]& /@ ("Distance" /. td[[2]]["Legs"])]
]

TravelDistanceList[arg_, OptionsPattern[]] := $Failed /; (
	Switch[arg,
		{_,__},
		Message[TravelDistanceList::arginvll, arg],
		_,
		Message[TravelDistanceList::arginvlt, arg]
	];
	False
)

TravelDistanceList[args___] := $Failed /; (System`Private`Arguments[TravelDistanceList[args], 1]; False);

Options[TravelTime] = Options@TravelDirections;

TravelTime[locations:{(_String|_Entity|_GeoPosition)..}, opts:OptionsPattern[]] :=
	Replace[travelDirectionsInternal[TravelTime, locations, opts],
		td_TravelDirectionsData :> td["TravelTime"]
	]

TravelTime[arg_, OptionsPattern[]] := $Failed /; (
	Switch[arg,
		{_,__},
		Message[TravelTime::arginvll, arg],
		_,
		Message[TravelTime::arginvlt, arg]
	];
	False
)

TravelTime[loc1:(_String|_Entity|_GeoPosition), loc2:(_String|_Entity|_GeoPosition), opts:OptionsPattern[]] :=
	TravelTime[{loc1, loc2}, opts]

TravelTime[arg1:(_String|_Entity|_GeoPosition), arg2_, opts:OptionsPattern[]] := $Failed /; (
	Message[TravelTime::arginv, arg2];
	False
)

TravelTime[arg1:Except[_String|_Entity|_GeoPosition], arg2_, opts:OptionsPattern[]] := $Failed /; (
	Message[TravelTime::arginv, arg1];
	False
)

TravelTime[args___] := $Failed /; (System`Private`Arguments[TravelTime[args], {1, 2}]; False);

BoxForm`MakeConditionalTextFormattingRule[TravelDirectionsData];

TravelDirectionsData /:
MakeBoxes[
    td : TravelDirectionsData[locations:{_, __}, data_Association, ___],
    fmt_
] /; BoxForm`UseIcons && And@@(NumericQ /@ Lookup[data, {"Distance", "Time"}]) :=
Module[ {icon, alwaysGrid, sometimesGrid},
    icon = BoxForm`GenericIcon[System`TravelDirections];
    alwaysGrid = {
        BoxForm`SummaryItem[{"Start: ", locations[[1]]}],
        Sequence@@(BoxForm`SummaryItem[{"Via: ", #}]& /@ locations[[2;;-2]]),
        BoxForm`SummaryItem[{"End: ", locations[[-1]]}]
    };
    sometimesGrid = {
        BoxForm`SummaryItem[{"Total distance: ", td["TravelDistance"]}],
        BoxForm`SummaryItem[{"Total time: ", td["TravelTime"]}]
    };
    BoxForm`ArrangeSummaryBox[TravelDirectionsData, td, icon, alwaysGrid, sometimesGrid, fmt, "Interpretable" -> True]
];
    
End[] (* end of "GIS`TravelDump`" *)

(*--------------------------------------------------------------------------------------*)
Protect[TravelDirections]
Protect[TravelDirectionsData]
Protect[TravelDistance]
Protect[TravelDistanceList]
Protect[TravelTime]
Protect[TravelMethod]
EndPackage[]
