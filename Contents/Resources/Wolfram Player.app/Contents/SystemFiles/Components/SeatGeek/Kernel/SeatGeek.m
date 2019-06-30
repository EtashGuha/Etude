Begin["SeatGeek`"]

Begin["`Private`"]

(******************************* SeatGeek *************************************)

(* Authentication information *)

seatgeekdata[]:={
		"ServiceName" 		-> "SeatGeek",
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
        		URLFetch[#1,{"StatusCode","Content"},
        		Sequence@@FilterRules[{##2},Except["Parameters"|"Headers"]],
        		"Parameters"->Cases[params, Except[Rule["apikey", _]]],
        		"Headers" -> {}]]&)
        	,
        "ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["SeatGeek",{"Client ID"->"client_id"},"https://seatgeek.com/account/develop","https://seatgeek.com/api-terms"],
	 	"Gets"				-> {"EventDetails","EventList","EventDataset","PerformerDetails","VenueDetails",
	 							"PerformerList","PerformerDataset","VenueList","VenueDataset"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawEvents","RawEvent","RawPerformers","RawPerformer","RawVenues","RawVenue"},
	 	"RawPosts"			-> {},
 		"Information"		-> "Import SeatGeek API data to the Wolfram Language"
}

(**** Raw Requests ****)

seatgeekdata["RawEvents"] := {
        "URL"				-> "http://api.seatgeek.com/2/events",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> Join[{"format","callback","geoip","lat","lon","range","per_page","page","sort","q","id"},
        							Flatten[Outer[StringJoin[#1, ".", #2] &, {"listing_count", "average_price", "lowest_price","highest_price"}, {"lt", "gt","lte","gte"}]],
        							Flatten[Outer[StringJoin["performers", #1, ".", #2] &, {"", "[home_team]", "[away_team]", "[primary]", "[any]"}, {"id", "slug"}]],
        							StringJoin["venue.", #] & /@ {"name", "address", "extended_address","city", "postal_code", "state", "country", "location", "url", "score", "id"},
        							Flatten[Outer[StringJoin[#1, ".", #2] &, {"datetime_local", "datetime_utc"}, {"gt", "gte", "lt", "lte"}]], {"datetime_local", "datetime_utc"},
        							StringJoin["taxonomies.", #] & /@ {"name", "id", "parent_id"}],
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

seatgeekdata["RawEvent"] := {
        "URL"				-> (ToString@StringForm["http://api.seatgeek.com/2/events/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"event_id"},
        "RequiredParameters"-> {"event_id"},
        "ResultsFunction"	-> formatresults
    }

seatgeekdata["RawPerformers"] := {
        "URL"				-> "http://api.seatgeek.com/2/performers",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> Join[{"format","callback","per_page","page","sort","slug","q","id"},StringJoin["taxonomies.", #] & /@ {"name", "id", "parent_id"}],
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

seatgeekdata["RawPerformer"] := {
        "URL"				-> (ToString@StringForm["http://api.seatgeek.com/2/performers/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"performer_id"},
        "RequiredParameters"-> {"performer_id"},
        "ResultsFunction"	-> formatresults
    }

seatgeekdata["RawVenues"] := {
        "URL"				-> "http://api.seatgeek.com/2/venues",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"format","callback","geoip","lat","lon","range","per_page","page","sort","city","state","country","postal_code","q","id"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> formatresults
    }

seatgeekdata["RawVenue"] := {
        "URL"				-> (ToString@StringForm["http://api.seatgeek.com/2/venues/`1`", #]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"venue_id"},
        "RequiredParameters"-> {"venue_id"},
        "ResultsFunction"	-> formatresults
    }

seatgeekdata[___]:=$Failed

(**** Cooked Requests ****)

seatgeekcookeddata["EventDetails", id_, args_] := Module[{rawdata, invalidParameters,eventId},
		invalidParameters = Select[Keys[args],!MemberQ[{"EventID"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"EventDetails"]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[KeyExistsQ[args,"EventID"],
			(
			eventId = "EventID" /. args;
			Switch[eventId,
			_String | _Integer,
				(
					rawdata = KeyClient`rawkeydata[id,"RawEvent",{"event_id"->ToString[eventId]}];
					Association[formateventdetails[formatresults[rawdata]]]
				),
			_List && Count[eventId,(_String|_Integer)]==Length[eventId] && Length[eventId] > 0,
				(
					eventId = ToString /@ eventId;
					eventId = StringJoin[Riffle[eventId,","]];
					rawdata = KeyClient`rawkeydata[id,"RawEvents",{"id"->eventId}];
					Association /@ (formateventdetails/@("events" /. formatresults[rawdata]))
				),
			_,
				(
					Message[ServiceExecute::nval,"EventID","SeatGeek"];
					Throw[$Failed]
				)
			]),
			(
				Message[ServiceExecute::nparam,"EventID"];
				Throw[$Failed]
			)
		]

]

seatgeekcookeddata[prop:("EventList"|"EventDataset"), id_, args_] := Module[{rawdata, invalidParameters,params={"format"->"json"},location, latitude, longitude, coordinates,
													point, radius, defaultRadius="30mi", priceRange, minPrice, maxPrice, lcRange, minCount, maxCount, performers,
													performerType, dateRange, startDate, endDate, limit, maxPerPage=100, startIndex, startPage, calls,
													argsCopy,result, total, sort, sortVal, sortDir, query},
	invalidParameters = Select[Keys[args],!MemberQ[{"Location","AveragePriceRange","HighestPriceRange","LowestPriceRange","ListingCountRange","Performer","Date",
													"MaxItems",MaxItems,"StartIndex","SortBy","Query"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];

	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	(* geo search *)
	If[KeyExistsQ[args,"Location"],
		(
			location = "Location" /. args;

			(* this handles the case where the user gives a GeoPosition representation for more than one point e.g. polygons *)
			If[MatchQ[Head[location],GeoPosition] && MatchQ[Head[QuantityMagnitude[Latitude[location], "AngularDegrees"]],List],
				location=GeoBoundingBox[location]];

			Switch[location,
				_String,
				(
					(* IP address *)
					If[StringMatchQ[location,RegularExpression["[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"]],
						params = Append[params,"geoip"->location],
						(* US/Canada zip code *)
						params = Append[params,"geoip"->location]
					]
				),
				Entity["ZIPCode",_], (* US zip code *)
				(
					params = Append[params,"geoip"->EntityValue[location, "Name"]]
				),
				True, (* current location (from user's IP) *)
				(
					params = Append[params,"geoip"-> "true"]
				),
				_GeoPosition, (* radial search, default radius 30 miles *)
				(
					latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
					longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;

					params = Join[params,{"lat"->latitude,"lon"->longitude}]
				),
				_Entity,
				(
					Switch[EntityTypeName[location],
						"Country",
						(
							latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
							longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;

							params = Join[params,{"lat"->latitude,"lon"->longitude}]
						),
						"City",
						(
							latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
							longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;

							params = Join[params,{"lat"->latitude,"lon"->longitude}]
						),
						_,
						(
							coordinates = LatitudeLongitude[location];
							If[MatchQ[Head[coordinates],List],
							(
								latitude = coordinates[[1,1]] // ToString;
								longitude = coordinates[[2,1]] // ToString;

								params = Join[params,{"lat"->latitude,"lon"->longitude}]
							),
							(
								Message[ServiceExecute::nval,"Location","SeatGeek"];
								Throw[$Failed]
							)]
						)
					]
				),
				_GeoDisk,
				(
					Switch[location,
						GeoDisk[],
						(
							point = $GeoLocation;
							radius = defaultRadius;
						),
						GeoDisk[_],
						(
							point = location[[1]];
							radius = defaultRadius;
						),
						GeoDisk[_,_,___],
						(
							point = location[[1]];
							radius = location[[2]];
							If[Internal`RealValuedNumericQ[radius],
								radius = radius / 1000, (* GeoDisk assumes that the quantity representing the radius is in meters *)
								radius = QuantityMagnitude[radius, "Kilometers"];
								radius = ToString[Round[radius]] ~~ "km"
							]
						)
					];

					latitude = QuantityMagnitude[Latitude[point], "AngularDegrees"] //ToString;
					longitude = QuantityMagnitude[Longitude[point], "AngularDegrees"] //ToString;

					params = Join[params, {"lat" -> latitude, "lon" -> longitude, "range" -> radius}]
				),
				_, (* unrecognized Location specification *)
				(
					Message[ServiceExecute::nval,"Location","SeatGeek"];
					Throw[$Failed]
				)
			]
		)
	];

	If[KeyExistsQ[args,"AveragePriceRange"],
		(
			priceRange = "AveragePriceRange" /. args;
			If[MatchQ[priceRange,{x_,y_}/;(IntegerQ[x]||MatchQ[x,-Infinity])&&(IntegerQ[y]||MatchQ[y,Infinity])],
				(
					minPrice = priceRange[[1]];
					maxPrice = priceRange[[2]];

					If[IntegerQ[minPrice], params = Append[params, "average_price.gte" -> ToString[minPrice]]];

					If[IntegerQ[maxPrice], params = Append[params, "average_price.lte" -> ToString[maxPrice]]];
				),
				(
					Message[ServiceExecute::nval,"AveragePriceRange","SeatGeek"];
					Throw[$Failed]
				)
			]
		)
	];

	If[KeyExistsQ[args,"HighestPriceRange"],
		(
			priceRange = "HighestPriceRange" /. args;
			If[MatchQ[priceRange,{x_,y_}/;(IntegerQ[x]||MatchQ[x,-Infinity])&&(IntegerQ[y]||MatchQ[y,Infinity])],
				(
					minPrice = priceRange[[1]];
					maxPrice = priceRange[[2]];

					If[IntegerQ[minPrice], params = Append[params, "highest_price.gte" -> ToString[minPrice]]];

					If[IntegerQ[maxPrice], params = Append[params, "highest_price.lte" -> ToString[maxPrice]]];
				),
				(
					Message[ServiceExecute::nval,"HighestPriceRange","SeatGeek"];
					Throw[$Failed]
				)
			]
		)
	];

	If[KeyExistsQ[args,"LowestPriceRange"],
		(
			priceRange = "LowestPriceRange" /. args;
			If[MatchQ[priceRange,{x_,y_}/;(IntegerQ[x]||MatchQ[x,-Infinity])&&(IntegerQ[y]||MatchQ[y,Infinity])],
				(
					minPrice = priceRange[[1]];
					maxPrice = priceRange[[2]];

					If[IntegerQ[minPrice], params = Append[params, "lowest_price.gte" -> ToString[minPrice]]];

					If[IntegerQ[maxPrice], params = Append[params, "lowest_price.lte" -> ToString[maxPrice]]];
				),
				(
					Message[ServiceExecute::nval,"LowestPriceRange","SeatGeek"];
					Throw[$Failed]
				)
			]
		)
	];

	If[KeyExistsQ[args,"ListingCountRange"],
		(
			lcRange = "ListingCountRange" /. args;
			If[MatchQ[lcRange,{x_,y_}/;(IntegerQ[x]||MatchQ[x,-Infinity])&&(IntegerQ[y]||MatchQ[y,Infinity])],
				(
					minCount = lcRange[[1]];
					maxCount = lcRange[[2]];

					If[IntegerQ[minCount], params = Append[params, "listing_count.gte" -> ToString[minCount]]];

					If[IntegerQ[maxCount], params = Append[params, "listing_count.lte" -> ToString[maxCount]]];
				),
				(
					Message[ServiceExecute::nval,"ListingCountRange","SeatGeek"];
					Throw[$Failed]
				)
			]
		)
	];

	If[KeyExistsQ[args,"Performer"],
		(
			performers = "Performer" /. args;
			performerType = "any";
			If[MatchQ[performers,{_String,___Rule}],
			(
				performerType = If[KeyExistsQ[performers,"Type"],"Type"/.Rest[performers],"Any"];
				performers = First[performers];
				performerType = ReplaceAll[performerType,{"Any"->"any","Primary"->"primary","HomeTeam"->"home_team","AwayTeam"->"away_team"}];
			)];
			If[MatchQ[performers,_String],
			(
				If[StringMatchQ[performers,RegularExpression["[0-9]+"]],
				(
					(* performer's id *)
					params = Append[params, "performers[" ~~ performerType ~~ "].id" -> performers]
				),
				(
					(* performer's slug *)
					params = Append[params, "performers[" ~~ performerType ~~ "].slug" -> performers]
				)]
			),
			(
				Message[ServiceExecute::nval,"Performers","SeatGeek"];
				Throw[$Failed]
			)]
		)
	];

	If[KeyExistsQ[args,"Date"],
	(
		dateRange = "Date" /. args;

		Switch[dateRange,
			_String,
			(
				startDate = DateObject[dateRange];
				endDate = startDate;
			),
			_DateObject,
			(
				startDate = dateRange;
				endDate = startDate;
			),
			List[_,_],
			(
				startDate = dateRange[[1]];
				endDate = dateRange[[2]];

				Switch[Head[startDate],
					String,
					startDate = DateObject[startDate],
					DateObject,
					startDate = startDate
				];
				Switch[Head[endDate],
					String,
					endDate = DateObject[endDate],
					DateObject,
					endDate = endDate
				];
			),
			Interval[{_DateObject,_DateObject}],
			(
				startDate = dateRange /. Interval[{f_,t_}]:>f;
				endDate = dateRange /. Interval[{f_,t_}]:>t;
			),
			_,
			(
				Message[ServiceExecute::nval,"Date","SeatGeek"];
				Throw[$Failed]
			)
		];

		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"Date","SeatGeek"];
			Throw[$Failed]
		)];

		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"Date","SeatGeek"];
			Throw[$Failed]
		)];

		startDate = DateString[startDate,{"Year", "-", "Month", "-", "Day"}];

		endDate = DateString[endDate,{"Year", "-", "Month", "-", "Day"}];

		params = Join[params,{Rule["datetime_utc.gte",startDate],Rule["datetime_utc.lte",endDate]}];

	)];

	If[KeyExistsQ[args,"SortBy"],
	(
		sort = "SortBy" /. args;
		If[Head[sort]===String,
		(
			(* Default descending *)
			sort = {sort,"Descending"}
		)];
		If[MatchQ[sort, {_String, _String}],
		(
			If[sort[[2]] == "Ascending", sortDir = "asc",
			(
				If[sort[[2]] == "Descending",
					sortDir = "desc",
					(
						Message[ServiceExecute::nval,"SortBy","SeatGeek"];
						Throw[$Failed]
					)
				]
			)];
			Switch[sort[[1]],
				"Date",
				sortVal = "datetime_local",
				"AnnounceDate",
				sortVal = "announce_date",
				"Id",
				sortVal = "id",
				"Score",
				sortVal = "score",
				_,
				(
					Message[ServiceExecute::nval,"SortBy","SeatGeek"];
					Throw[$Failed]
				)
			];
		)];
		params = Append[params,Rule["sort",sortVal ~~ "." ~~ sortDir]];
	)];

	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"q" -> query]
		)
	];

	If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","SeatGeek"];
				Throw[$Failed]
			)];
	),
		limit = maxPerPage;
	];

	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(
				Message[ServiceExecute::nval,"StartIndex","SeatGeek"];
				Throw[$Failed]
			)];
		),
		startIndex = 0
	];

	(* regularized startIndex *)
	startPage = 1 + Quotient[startIndex,maxPerPage];
	startIndex = Mod[startIndex, maxPerPage];
	total = startIndex + limit;

	If[total <= maxPerPage,
	(
		params = Join[params,{"per_page" -> ToString[total],"page" -> ToString[startPage]}];
		rawdata = KeyClient`rawkeydata[id,"RawEvents",params];
		result = "events" /. formatresults[rawdata];
	),
	(
		calls = Ceiling[total / maxPerPage*1.];
		params = Join[params,{"per_page" -> ToString[maxPerPage],"page" -> ToString[startPage]}];
		result = {};
		(
			params = ReplaceAll[params,Rule["page",_]->Rule["page",ToString[startPage + #]]];
			rawdata = KeyClient`rawkeydata[id,"RawEvents",params];
			result = Join[result, "events" /.formatresults[rawdata]];

		)& /@ Range[0,calls-1];
	)];
	result = (formatevent /@ result[[startIndex + 1;;Min[total,Length[result]]]]);
	If[prop=="EventList",
		result,
		(
			If[Length[result]==0,
				Dataset[Association[]],
				Dataset[Association /@ result]
			]
		)
	]
]

seatgeekcookeddata["PerformerDetails", id_, args_] := Module[{rawdata, invalidParameters,pId},
		invalidParameters = Select[Keys[args],!MemberQ[{"PerformerID"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"PerformerDetails"]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[KeyExistsQ[args,"PerformerID"],
			(
			pId = "PerformerID" /. args;
			Switch[pId,
			_String | _Integer,
				(
					rawdata = KeyClient`rawkeydata[id,"RawPerformer",{"performer_id"->ToString[pId]}];
					Association[formatperformerdetails[formatresults[rawdata]]]
				),
			_List && Count[pId,(_String|_Integer)]==Length[pId] && Length[pId] > 0,
				(
					pId = ToString /@ pId;
					pId = StringJoin[Riffle[pId,","]];
					rawdata = KeyClient`rawkeydata[id,"RawPerformers",{"id"->pId}];
					Association /@ (formatperformerdetails/@("performers" /. formatresults[rawdata]))
				),
			_,
				(
					Message[ServiceExecute::nval,"PerformerID","SeatGeek"];
					Throw[$Failed]
				)
			]),
			(
				Message[ServiceExecute::nparam,"PerformerID"];
				Throw[$Failed]
			)
		]

]

seatgeekcookeddata[prop:("PerformerList"|"PerformerDataset"), id_, args_] := Module[{rawdata, invalidParameters,params={"format"->"json"},
													limit, maxPerPage=100, startIndex, startPage, calls,
													argsCopy,result, total, sort, sortVal, sortDir, query},
	invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","SortBy","Query"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];

	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	If[KeyExistsQ[args,"SortBy"],
	(
		sort = "SortBy" /. args;
		If[Head[sort]===String,
		(
			(* Default descending *)
			sort = {sort,"Descending"}
		)];
		If[MatchQ[sort, {_String, _String}],
		(
			If[sort[[2]] == "Ascending", sortDir = "asc",
			(
				If[sort[[2]] == "Descending",
					sortDir = "desc",
					(
						Message[ServiceExecute::nval,"SortBy","SeatGeek"];
						Throw[$Failed]
					)
				]
			)];
			Switch[sort[[1]],
				"Id",
				sortVal = "id",
				"Score",
				sortVal = "score",
				_,
				(
					Message[ServiceExecute::nval,"SortBy","SeatGeek"];
					Throw[$Failed]
				)
			];
		)];
		params = Append[params,Rule["sort",sortVal ~~ "." ~~ sortDir]];
	)];

	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"q" -> query]
		)
	];

	If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","SeatGeek"];
				Throw[$Failed]
			)];
	),
		limit = maxPerPage;
	];

	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(
				Message[ServiceExecute::nval,"StartIndex","SeatGeek"];
				Throw[$Failed]
			)];
		),
		startIndex = 0
	];

	(* regularized startIndex *)
	startPage = 1 + Quotient[startIndex,maxPerPage];
	startIndex = Mod[startIndex, maxPerPage];
	total = startIndex + limit;

	If[total <= maxPerPage,
	(
		params = Join[params,{"per_page" -> ToString[total],"page" -> ToString[startPage]}];
		rawdata = KeyClient`rawkeydata[id,"RawPerformers",params];
		result = "performers" /. formatresults[rawdata];
	),
	(
		calls = Ceiling[total / maxPerPage*1.];
		params = Join[params,{"per_page" -> ToString[maxPerPage],"page" -> ToString[startPage]}];
		result = {};
		(
			params = ReplaceAll[params,Rule["page",_]->Rule["page",ToString[startPage + #]]];
			rawdata = KeyClient`rawkeydata[id,"RawPerformers",params];
			result = Join[result, "performers" /.formatresults[rawdata]];

		)& /@ Range[0,calls-1];
	)];
	result = (formatperformer /@ result[[startIndex + 1;;Min[total,Length[result]]]]);
	If[prop=="PerformerList",
		result,
		(
			If[Length[result]==0,
				Dataset[Association[]],
				Dataset[Association /@ result]
			]
		)
	]
]

seatgeekcookeddata["VenueDetails", id_, args_] := Module[{rawdata, invalidParameters,venueId},
		invalidParameters = Select[Keys[args],!MemberQ[{"VenueID"},#]&];

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"VenueDetails"]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[KeyExistsQ[args,"VenueID"],
			(
			venueId = "VenueID" /. args;
			Switch[venueId,
			_String | _Integer,
				(
					rawdata = KeyClient`rawkeydata[id,"RawVenue",{"venue_id"->ToString[venueId]}];
					Association[formatvenuedetails[formatresults[rawdata]]]
				),
			_List && Count[venueId,(_String|_Integer)]==Length[venueId] && Length[venueId] > 0,
				(
					venueId = ToString /@ venueId;
					venueId = StringJoin[Riffle[venueId,","]];
					rawdata = KeyClient`rawkeydata[id,"RawVenues",{"id"->venueId}];
					Association /@ (formatvenuedetails/@("venues" /. formatresults[rawdata]))
				),
			_,
				(
					Message[ServiceExecute::nval,"VenueID","SeatGeek"];
					Throw[$Failed]
				)
			]),
			(
				Message[ServiceExecute::nparam,"VenueID"];
				Throw[$Failed]
			)
		]

]

seatgeekcookeddata[prop:("VenueList"|"VenueDataset"), id_, args_] := Module[{rawdata, invalidParameters,params={"format"->"json"},location, latitude, longitude, coordinates,
													point, radius, defaultRadius="30mi",limit, maxPerPage=100, startIndex, startPage, calls,
													argsCopy,result, total, sort, sortVal, sortDir, query},
	invalidParameters = Select[Keys[args],!MemberQ[{"Location","MaxItems",MaxItems,"StartIndex","SortBy","Query"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];

	argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
	(* geo search *)
	If[KeyExistsQ[args,"Location"],
		(
			location = "Location" /. args;

			(* this handles the case where the user gives a GeoPosition representation for more than one point e.g. polygons *)
			If[MatchQ[Head[location],GeoPosition] && MatchQ[Head[QuantityMagnitude[Latitude[location], "AngularDegrees"]],List],
				location=GeoBoundingBox[location]];

			Switch[location,
				_String,
				(
					(* IP address *)
					If[StringMatchQ[location,RegularExpression["[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"]],
						params = Append[params,"geoip"->location],
						(* US/Canada zip code *)
						params = Append[params,"geoip"->location]
					]
				),
				Entity["ZIPCode",_], (* US zip code *)
				(
					params = Append[params,"geoip"->EntityValue[location, "Name"]]
				),
				True, (* current location (from user's IP) *)
				(
					params = Append[params,"geoip"-> "true"]
				),
				_GeoPosition, (* radial search, default radius 30 miles *)
				(
					latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
					longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;

					params = Join[params,{"lat"->latitude,"lon"->longitude}]
				),
				_Entity,
				(
					Switch[EntityTypeName[location],
						"Country",
						(
							latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
							longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;

							params = Join[params,{"lat"->latitude,"lon"->longitude}]
						),
						"City",
						(
							latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
							longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;

							params = Join[params,{"lat"->latitude,"lon"->longitude}]
						),
						_,
						(
							coordinates = LatitudeLongitude[location];
							If[MatchQ[Head[coordinates],List],
							(
								latitude = coordinates[[1,1]] // ToString;
								longitude = coordinates[[2,1]] // ToString;

								params = Join[params,{"lat"->latitude,"lon"->longitude}]
							),
							(
								Message[ServiceExecute::nval,"Location","SeatGeek"];
								Throw[$Failed]
							)]
						)
					]
				),
				_GeoDisk,
				(
					Switch[location,
						GeoDisk[],
						(
							point = $GeoLocation;
							radius = defaultRadius;
						),
						GeoDisk[_],
						(
							point = location[[1]];
							radius = defaultRadius;
						),
						GeoDisk[_,_,___],
						(
							point = location[[1]];
							radius = location[[2]];
							If[Internal`RealValuedNumericQ[radius],
								radius = radius / 1000, (* GeoDisk assumes that the quantity representing the radius is in meters *)
								radius = QuantityMagnitude[radius, "Kilometers"];
								radius = ToString[Round[radius]] ~~ "km"
							]
						)
					];

					latitude = QuantityMagnitude[Latitude[point], "AngularDegrees"] //ToString;
					longitude = QuantityMagnitude[Longitude[point], "AngularDegrees"] //ToString;

					params = Join[params, {"lat" -> latitude, "lon" -> longitude, "range" -> radius}]
				),
				_, (* unrecognized Location specification *)
				(
					Message[ServiceExecute::nval,"Location","SeatGeek"];
					Throw[$Failed]
				)
			]
		)
	];

	If[KeyExistsQ[args,"SortBy"],
	(
		sort = "SortBy" /. args;
		If[Head[sort]===String,
		(
			(* Default descending *)
			sort = {sort,"Descending"}
		)];
		If[MatchQ[sort, {_String, _String}],
		(
			If[sort[[2]] == "Ascending", sortDir = "asc",
			(
				If[sort[[2]] == "Descending",
					sortDir = "desc",
					(
						Message[ServiceExecute::nval,"SortBy","SeatGeek"];
						Throw[$Failed]
					)
				]
			)];
			Switch[sort[[1]],
				"Id",
				sortVal = "id",
				"Score",
				sortVal = "score",
				_,
				(
					Message[ServiceExecute::nval,"SortBy","SeatGeek"];
					Throw[$Failed]
				)
			];
		)];
		params = Append[params,Rule["sort",sortVal ~~ "." ~~ sortDir]];
	)];

	If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			params = Append[params,"q" -> query]
		)
	];

	If[KeyExistsQ[argsCopy,MaxItems],
		(
			limit = MaxItems /. argsCopy;
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","SeatGeek"];
				Throw[$Failed]
			)];
	),
		limit = maxPerPage;
	];

	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = "StartIndex" /. args;
			If[!IntegerQ[startIndex],
			(
				Message[ServiceExecute::nval,"StartIndex","SeatGeek"];
				Throw[$Failed]
			)];
		),
		startIndex = 0
	];

	(* regularized startIndex *)
	startPage = 1 + Quotient[startIndex,maxPerPage];
	startIndex = Mod[startIndex, maxPerPage];
	total = startIndex + limit;
	If[total <= maxPerPage,
	(
		params = Join[params,{"per_page" -> ToString[total],"page" -> ToString[startPage]}];
		rawdata = KeyClient`rawkeydata[id,"RawVenues",params];
		result = "venues" /. formatresults[rawdata];
	),
	(
		calls = Ceiling[total / maxPerPage*1.];
		params = Join[params,{"per_page" -> ToString[maxPerPage],"page" -> ToString[startPage]}];
		result = {};
		(
			params = ReplaceAll[params,Rule["page",_]->Rule["page",ToString[startPage + #]]];
			rawdata = KeyClient`rawkeydata[id,"RawVenues",params];
			result = Join[result, "venues" /.formatresults[rawdata]];

		)& /@ Range[0,calls-1];
	)];
	If[MatchQ[result,{}],
		Dataset[Association[]],
		(result = (formatvenue /@ result[[If[startIndex>=Length[result],0,startIndex]+1 ;;Min[total,Length[result]]]]);
		If[prop=="VenueList",
			result,
			(
				If[Length[result]==0,
					Dataset[Association[]],
					Dataset[Association /@ result]
				]
			)
		])
	]
]

seatgeekcookeddata[___]:=$Failed

seatgeeksendmessage[___]:=$Failed

(* Utilities *)
getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.seatgeekdata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]


(*formatresults[rawdata_] := ImportString[ToString[rawdata,CharacterEncoding->"UTF-8"],"JSON"]*)
formatresults[rawdata_] := ImportString[ToString[rawdata[[2]],CharacterEncoding->"UTF-8"],"JSON"]

formatevent[e_] := Module[{fields, orderList, dtutc, dtlocal, doutc, dolocal, tz,
						stats, listingcount,avgprice,lowprice,highprice,fieldnames},
  (
   fields = FilterRules[e, {"performers", "datetime_local", "datetime_utc","score", "id", "stats", "type", "title", "url", "venue"}];
   fields = ReplaceAll[fields, Rule["performers", d_] :> Rule["performers", FilterRules[#, {"id", "name"}] & /@ d]];
   fields = ReplaceAll[fields, Rule["venue", d_] :> Rule["venue", FilterRules[d, {"id", "name", "display_location"}]]];

   fields = ReplaceAll[fields,Rule["performers",p_]:>Rule["performers",Association/@p]];
   fields = ReplaceAll[fields,Rule["venue",v_]:>Rule["venue",Association[v]]];

   (* format dates considering time zones *)
   dtutc = "datetime_utc" /. fields;
   dtlocal = "datetime_local" /. fields;
   doutc = DateObject[dtutc];
   dolocal = DateObject[dtlocal];
   tz = QuantityMagnitude[dolocal - doutc];
   doutc = DateObject[dtutc, "TimeZone"->0];
   dolocal = DateObject[dtlocal, "TimeZone" -> tz];
   fields = ReplaceAll[fields, Rule["datetime_local", d_] :> Rule["datetime_local", dolocal]];
   fields = DeleteCases[fields, Rule["datetime_utc", _], Infinity];
   fieldnames = {"id","datetime_local","title","type","url","score","performers","venue"};

   (* Stats: Break into separate keys, and add appropriate units to the price values *)
   If[KeyExistsQ[fields,"stats"],
   	(
   		stats = "stats" /. fields;

   		If[KeyExistsQ[stats,"listing_count"],
   		(
   			listingcount = "listing_count" /. stats;
   			fields = Append[fields, Rule["listing_count",listingcount]];
   			fieldnames = Append[fieldnames,"listing_count"];
   		)];
   		If[KeyExistsQ[stats,"average_price"],
   		(
   			avgprice = "average_price" /. stats;
   			If[!MatchQ[avgprice,Null],
   				fields = Append[fields, Rule["average_price",Quantity[avgprice, "USDollars"]]];
   				fieldnames = Append[fieldnames,"average_price"];
   			]
   		)];
   		If[KeyExistsQ[stats,"lowest_price"],
   		(
   			lowprice = "lowest_price" /. stats;
   			If[!MatchQ[lowprice,Null],
   				fields = Append[fields, Rule["lowest_price",Quantity[lowprice, "USDollars"]]];
				fieldnames = Append[fieldnames,"lowest_price"];
   			]
   		)];
   		If[KeyExistsQ[stats,"highest_price"],
   		(
   			highprice = "highest_price" /. stats;
   			If[!MatchQ[highprice,Null],
   				fields = Append[fields, Rule["highest_price",Quantity[highprice, "USDollars"]]];
   				fieldnames = Append[fieldnames,"highest_price"];
   			]
   		)];
   		fields = DeleteCases[fields, Rule["stats", _], Infinity];
   )];

   orderList = Thread[fieldnames -> Range[Length[fieldnames]]];
   fields = SortBy[fields, (#[[1]] /. orderList&)];

   fields = ReplaceAll[fields, Rule[x_,y_] :> Rule[camelCase[x],y]];
   fields = ReplaceAll[fields, Rule["Id",x_] :> Rule["ID",x]];
   fields = ReplaceAll[fields, Rule["DatetimeLocal",x_] :> Rule["Date",x]];
   fields
  )]

formateventdetails[e_] := Module[{fields, dtutc, dtlocal, doutc, dolocal, tz,
						stats, listingcount,avgprice,lowprice,highprice},
  (
   fields = e;
   fields = ReplaceAll[fields, Rule["performers", d_] :> Rule["performers", FilterRules[#, {"id", "name"}] & /@ d]];
   fields = ReplaceAll[fields, Rule["venue", d_] :> Rule["venue", FilterRules[d, {"id", "name", "display_location"}]]];

   fields = ReplaceAll[fields,Rule["performers",p_]:>Rule["performers",Association/@p]];
   fields = ReplaceAll[fields,Rule["venue",v_]:>Rule["venue",Association[v]]];
   fields = ReplaceAll[fields,Rule["taxonomies",p_]:>Rule["taxonomies",Association/@p]];

   (* format dates considering time zones *)
   If[KeyExistsQ[fields,"datetime_utc"]&&KeyExistsQ[fields,"datetime_local"],
   	dtutc = "datetime_utc" /. fields;
   	dtlocal = "datetime_local" /. fields;
   	doutc = DateObject[dtutc];
   	dolocal = DateObject[dtlocal];
   	tz = QuantityMagnitude[dolocal - doutc];
   	doutc = DateObject[dtutc, "TimeZone"->0];
   	dolocal = DateObject[dtlocal, "TimeZone" -> tz];
   ];

   fields = ReplaceAll[fields, Rule["datetime_local", d_] :> Rule["datetime_local", dolocal]];
   fields = ReplaceAll[fields, Rule["datetime_utc", _]:> Rule["datetime_utc", doutc]];
   fields = ReplaceAll[fields, Rule["visible_until_utc", d_] :> Rule["visible_until_utc", DateObject[d,"TimeZone"->0]]];
   fields = ReplaceAll[fields, Rule["announce_date", d_] :> Rule["announce_date", DateObject[d,"TimeZone"->0]]];
   fields = ReplaceAll[fields, Rule["created_at", d_] :> Rule["created_at", DateObject[d,"TimeZone"->0]]];
   (* Stats: Break into separate keys, and add appropriate units to the price values *)
   If[KeyExistsQ[fields,"stats"],
   	(
   		stats = "stats" /. fields;

   		If[KeyExistsQ[stats,"listing_count"],
   		(
   			listingcount = "listing_count" /. stats;
   			fields = Append[fields, Rule["listing_count",listingcount]];
   		)];
   		If[KeyExistsQ[stats,"average_price"],
   		(
   			avgprice = "average_price" /. stats;
   			If[!MatchQ[avgprice,Null],
   				fields = Append[fields, Rule["average_price",Quantity[avgprice, "USDollars"]]];
   			]
   		)];
   		If[KeyExistsQ[stats,"lowest_price"],
   		(
   			lowprice = "lowest_price" /. stats;
   			If[!MatchQ[lowprice,Null],
   				fields = Append[fields, Rule["lowest_price",Quantity[lowprice, "USDollars"]]];
   			]
   		)];
   		If[KeyExistsQ[stats,"highest_price"],
   		(
   			highprice = "highest_price" /. stats;
   			If[!MatchQ[highprice,Null],
   				fields = Append[fields, Rule["highest_price",Quantity[highprice, "USDollars"]]];
   			]
   		)];
   		fields = DeleteCases[fields, Rule["stats", _], Infinity];
   )];

   fields = ReplaceAll[fields, Rule[x_,y_] :> Rule[camelCase[x],y]];
   fields = ReplaceAll[fields, Rule["Id",x_] :> Rule["ID",x]];
   fields = ReplaceAll[fields, Rule["DatetimeLocal",x_] :> Rule["Date",x]];
   fields = ReplaceAll[fields, Rule["DatetimeUtc",x_] :> Rule["DateUTC",x]];
   fields = ReplaceAll[fields, Rule["VisibleUntilUtc",x_] :> Rule["VisibleUntilUTC",x]];
   fields
  )]

formatperformer[e_] := Module[{fields, orderList, fieldnames},
  (
   fields = FilterRules[e, {"url","name","type","score","slug","id","genres"}];

   fields = ReplaceAll[fields,Rule["genres",g_]:>Rule["genres",Association/@g]];

   fieldnames = {"id","slug","name","type","url","score","genres"};

   orderList = Thread[fieldnames -> Range[Length[fieldnames]]];

   fields = SortBy[fields, (#[[1]] /. orderList&)];

   fields = ReplaceAll[fields, Rule[x_,y_] :> Rule[camelCase[x],y]];
   fields = ReplaceAll[fields, Rule["Id",x_] :> Rule["ID",x]];
   fields
  )]

formatperformerdetails[e_] := Module[{fields},
  (
   fields = e;

   fields = ReplaceAll[fields,Rule["genres",g_]:>Rule["genres",Association/@g]];
   fields = ReplaceAll[fields,Rule["taxonomies",t_]:>Rule["taxonomies",Association/@t]];
   fields = ReplaceAll[fields,Rule["links",l_]:>Rule["links",Association/@l]];
   fields = ReplaceAll[fields,Rule["stats",s_]:>Rule["stats",Association[s]]];
   fields = ReplaceAll[fields,Rule["images",i_]:>Rule["images",Association[i]]];

   fields = ReplaceAll[fields, Rule[x_,y_] :> Rule[camelCase[x],y]];
   fields = ReplaceAll[fields, Rule["Id",x_] :> Rule["ID",x]];
   fields
  )]

formatvenue[e_] := Module[{fields, orderList, fieldnames, city, cityI, location, country, pc, countryE},
  (
   fields = FilterRules[e, {"url","city","location","name","country","score","postal_code","slug","state","id"}];

   (*If[KeyExistsQ[fields,"city"],
   	city = "city" /. fields;
   	cityI = Interpreter["City"][city];
   	If[Head[cityI]===Entity,
   		fields = ReplaceAll[fields,Rule["city",i_]:>Rule["city",cityI]]
   	]
   ];*)

   If[KeyExistsQ[fields,"location"] && ("location" /. fields)=!=Null,
   	location = "location" /. fields;
   	fields = ReplaceAll[fields,Rule["location",i_]:>Rule["location",GeoPosition[{"lat"/.location,"lon"/.location}]]]
   ];

   If[KeyExistsQ[fields,"country"] && ("country" /. fields)=!=Null,
   	country = "country" /. fields;
   	If[KeyExistsQ[countrycodealignment,ToUpperCase[country]],
   		country = ToUpperCase[country] /. countrycodealignment,
   		(
   			countryE = Interpreter["Country"][country];
   			If[MatchQ[countryE,_Entity],country = countryE]
   		)
   	];
   	fields = ReplaceAll[fields,Rule["country",i_]:>Rule["country",country]]
   ];

   If[KeyExistsQ[fields,"postal_code"] && ("postal_code" /. fields)=!=Null,
   	pc = "postal_code" /. fields;
   	pc = ZIPCodeData[pc];
   	If[MatchQ[pc,_Entity], fields = ReplaceAll[fields,Rule["postal_code",i_]:>Rule["postal_code",ZIPCodeData[pc]]]]
   ];

   fields = ReplaceAll[fields,(Rule["state",i_]/;i=!=Null):>Rule["state",ToUpperCase[i]/.usstatealignment]];

   fieldnames = {"id","slug","name","location","city","state","country","postal_code","score","url"};

   orderList = Thread[fieldnames -> Range[Length[fieldnames]]];

   fields = SortBy[fields, (#[[1]] /. orderList&)];

   fields = ReplaceAll[fields, Rule[x_,y_] :> Rule[camelCase[x],y]];
   fields = ReplaceAll[fields, Rule["Id",x_] :> Rule["ID",x]];
   fields
  )]

formatvenuedetails[e_] := Module[{fields, city, cityI, location, country, countryE},
  (
   fields = e;

   (*If[KeyExistsQ[fields,"city"],
   	city = "city" /. fields;
   	cityI = Interpreter["City"][city];
   	If[Head[cityI]===Entity,
   		fields = ReplaceAll[fields,Rule["city",i_]:>Rule["city",cityI]]
   	]
   ];*)

   If[KeyExistsQ[fields,"location"],
   	location = "location" /. fields;
   	fields = ReplaceAll[fields,Rule["location",i_]:>Rule["location",GeoPosition[{"lat"/.location,"lon"/.location}]]]
   ];

   If[KeyExistsQ[fields,"country"],
   	country = "country" /. fields;
   	If[KeyExistsQ[countrycodealignment,ToUpperCase[country]],
   		country = ToUpperCase[country] /. countrycodealignment,
   		(
   			countryE = Interpreter["Country"][country];
   			If[MatchQ[countryE,_Entity],country = countryE]
   		)
   	];
   	fields = ReplaceAll[fields,Rule["country",i_]:>Rule["country",country]]
   ];

   fields = ReplaceAll[fields,Rule["postal_code",i_]:>Rule["postal_code",ZIPCodeData[i]]];
   fields = ReplaceAll[fields,Rule["state",i_]:>Rule["state",ToUpperCase["i"]/.usstatealignment]];
   fields = ReplaceAll[fields,Rule["stats",s_]:>Rule["stats",Association[s]]];
   fields = ReplaceAll[fields,Rule["links",s_]:>Rule["links",Association[s]]];

   fields = ReplaceAll[fields, Rule[x_,y_] :> Rule[camelCase[x],y]];
   fields = ReplaceAll[fields, Rule["Id",x_] :> Rule["ID",x]];
   fields
  )]

camelCase[text_] := Module[{split, partial}, (
	(*text = ToLowerCase[text];*)
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    partial = StringJoin[partial];
    partial = StringReplace[partial,RegularExpression["[Uu][Rr][Ll]"]->"URL"];
    partial
    )]

usstatealignment = {"AL" -> Entity[
   "AdministrativeDivision", {"Alabama", "UnitedStates"}],
 "AK" -> Entity["AdministrativeDivision", {"Alaska", "UnitedStates"}],
  "AZ" -> Entity[
   "AdministrativeDivision", {"Arizona", "UnitedStates"}],
 "AR" -> Entity[
   "AdministrativeDivision", {"Arkansas", "UnitedStates"}],
 "CA" -> Entity[
   "AdministrativeDivision", {"California", "UnitedStates"}],
 "CO" -> Entity[
   "AdministrativeDivision", {"Colorado", "UnitedStates"}],
 "CT" -> Entity[
   "AdministrativeDivision", {"Connecticut", "UnitedStates"}],
 "DE" -> Entity[
   "AdministrativeDivision", {"Delaware", "UnitedStates"}],
 "FL" -> Entity[
   "AdministrativeDivision", {"Florida", "UnitedStates"}],
 "GA" -> Entity[
   "AdministrativeDivision", {"Georgia", "UnitedStates"}],
 "HI" -> Entity["AdministrativeDivision", {"Hawaii", "UnitedStates"}],
  "ID" -> Entity["AdministrativeDivision", {"Idaho", "UnitedStates"}],
  "IL" -> Entity[
   "AdministrativeDivision", {"Illinois", "UnitedStates"}],
 "IN" -> Entity[
   "AdministrativeDivision", {"Indiana", "UnitedStates"}],
 "IA" -> Entity["AdministrativeDivision", {"Iowa", "UnitedStates"}],
 "KS" -> Entity["AdministrativeDivision", {"Kansas", "UnitedStates"}],
  "KY" -> Entity[
   "AdministrativeDivision", {"Kentucky", "UnitedStates"}],
 "LA" -> Entity[
   "AdministrativeDivision", {"Louisiana", "UnitedStates"}],
 "ME" -> Entity["AdministrativeDivision", {"Maine", "UnitedStates"}],
 "MD" -> Entity[
   "AdministrativeDivision", {"Maryland", "UnitedStates"}],
 "MA" -> Entity[
   "AdministrativeDivision", {"Massachusetts", "UnitedStates"}],
 "MI" -> Entity[
   "AdministrativeDivision", {"Michigan", "UnitedStates"}],
 "MN" -> Entity[
   "AdministrativeDivision", {"Minnesota", "UnitedStates"}],
 "MS" -> Entity[
   "AdministrativeDivision", {"Mississippi", "UnitedStates"}],
 "MO" -> Entity[
   "AdministrativeDivision", {"Missouri", "UnitedStates"}],
 "MT" -> Entity[
   "AdministrativeDivision", {"Montana", "UnitedStates"}],
 "NE" -> Entity[
   "AdministrativeDivision", {"Nebraska", "UnitedStates"}],
 "NV" -> Entity["AdministrativeDivision", {"Nevada", "UnitedStates"}],
  "NH" -> Entity[
   "AdministrativeDivision", {"NewHampshire", "UnitedStates"}],
 "NJ" -> Entity[
   "AdministrativeDivision", {"NewJersey", "UnitedStates"}],
 "NM" -> Entity[
   "AdministrativeDivision", {"NewMexico", "UnitedStates"}],
 "NY" -> Entity[
   "AdministrativeDivision", {"NewYork", "UnitedStates"}],
 "NC" -> Entity[
   "AdministrativeDivision", {"NorthCarolina", "UnitedStates"}],
 "ND" -> Entity[
   "AdministrativeDivision", {"NorthDakota", "UnitedStates"}],
 "OH" -> Entity["AdministrativeDivision", {"Ohio", "UnitedStates"}],
 "OK" -> Entity[
   "AdministrativeDivision", {"Oklahoma", "UnitedStates"}],
 "OR" -> Entity["AdministrativeDivision", {"Oregon", "UnitedStates"}],
  "PA" -> Entity[
   "AdministrativeDivision", {"Pennsylvania", "UnitedStates"}],
 "RI" -> Entity[
   "AdministrativeDivision", {"RhodeIsland", "UnitedStates"}],
 "SC" -> Entity[
   "AdministrativeDivision", {"SouthCarolina", "UnitedStates"}],
 "SD" -> Entity[
   "AdministrativeDivision", {"SouthDakota", "UnitedStates"}],
 "TN" -> Entity[
   "AdministrativeDivision", {"Tennessee", "UnitedStates"}],
 "TX" -> Entity["AdministrativeDivision", {"Texas", "UnitedStates"}],
 "UT" -> Entity["AdministrativeDivision", {"Utah", "UnitedStates"}],
 "VT" -> Entity[
   "AdministrativeDivision", {"Vermont", "UnitedStates"}],
 "VA" -> Entity[
   "AdministrativeDivision", {"Virginia", "UnitedStates"}],
 "WA" -> Entity[
   "AdministrativeDivision", {"Washington", "UnitedStates"}],
 "WV" -> Entity[
   "AdministrativeDivision", {"WestVirginia", "UnitedStates"}],
 "WI" -> Entity[
   "AdministrativeDivision", {"Wisconsin", "UnitedStates"}],
 "WY" -> Entity[
   "AdministrativeDivision", {"Wyoming", "UnitedStates"}]}

countrycodealignment = {"AE" -> Entity["Country", "UnitedArabEmirates"],
 "AF" -> Entity["Country", "Afghanistan"],
 "AG" -> Entity["Country", "AntiguaBarbuda"],
 "AI" -> Entity["Country", "Anguilla"],
 "AL" -> Entity["Country", "Albania"],
 "AM" -> Entity["Country", "Armenia"],
 "AO" -> Entity["Country", "Angola"],
 "AR" -> Entity["Country", "Argentina"],
 "AS" -> Entity["Country", "AmericanSamoa"],
 "AT" -> Entity["Country", "Austria"],
 "AU" -> Entity["Country", "Australia"],
 "AW" -> Entity["Country", "Aruba"],
 "AZ" -> Entity["Country", "Azerbaijan"],
 "BA" -> Entity["Country", "BosniaHerzegovina"],
 "BB" -> Entity["Country", "Barbados"],
 "BD" -> Entity["Country", "Bangladesh"],
 "BE" -> Entity["Country", "Belgium"],
 "BF" -> Entity["Country", "BurkinaFaso"],
 "BG" -> Entity["Country", "Bulgaria"],
 "BH" -> Entity["Country", "Bahrain"],
 "BI" -> Entity["Country", "Burundi"],
 "BJ" -> Entity["Country", "Benin"],
 "BM" -> Entity["Country", "Bermuda"],
 "BN" -> Entity["Country", "Brunei"],
 "BO" -> Entity["Country", "Bolivia"],
 "BR" -> Entity["Country", "Brazil"],
 "BS" -> Entity["Country", "Bahamas"],
 "BT" -> Entity["Country", "Bhutan"],
 "BW" -> Entity["Country", "Botswana"],
 "BY" -> Entity["Country", "Belarus"],
 "BZ" -> Entity["Country", "Belize"],
 "CA" -> Entity["Country", "Canada"],
 "CC" -> Entity["Country", "CocosKeelingIslands"],
 "CD" -> Entity["Country", "DemocraticRepublicCongo"],
 "CF" -> Entity["Country", "CentralAfricanRepublic"],
 "CG" -> Entity["Country", "RepublicCongo"],
 "CH" -> Entity["Country", "Switzerland"],
 "CI" -> Entity["Country", "IvoryCoast"],
 "CK" -> Entity["Country", "CookIslands"],
 "CL" -> Entity["Country", "Chile"],
 "CM" -> Entity["Country", "Cameroon"],
 "CN" -> Entity["Country", "China"],
 "CO" -> Entity["Country", "Colombia"],
 "CR" -> Entity["Country", "CostaRica"],
 "CU" -> Entity["Country", "Cuba"],
 "CV" -> Entity["Country", "CapeVerde"],
 "CW" -> Entity["Country", "Curacao"],
 "CX" -> Entity["Country", "ChristmasIsland"],
 "CY" -> Entity["Country", "Cyprus"],
 "CZ" -> Entity["Country", "CzechRepublic"],
 "DE" -> Entity["Country", "Germany"],
 "DJ" -> Entity["Country", "Djibouti"],
 "DK" -> Entity["Country", "Denmark"],
 "DM" -> Entity["Country", "Dominica"],
 "DO" -> Entity["Country", "DominicanRepublic"],
 "DZ" -> Entity["Country", "Algeria"],
 "EC" -> Entity["Country", "Ecuador"],
 "EE" -> Entity["Country", "Estonia"],
 "EG" -> Entity["Country", "Egypt"],
 "EH" -> Entity["Country", "WesternSahara"],
 "ER" -> Entity["Country", "Eritrea"],
 "ES" -> Entity["Country", "Spain"],
 "ET" -> Entity["Country", "Ethiopia"],
 "FI" -> Entity["Country", "Finland"],
 "FJ" -> Entity["Country", "Fiji"],
 "FK" -> Entity["Country", "FalklandIslands"],
 "FM" -> Entity["Country", "Micronesia"],
 "FO" -> Entity["Country", "FaroeIslands"],
 "FR" -> Entity["Country", "France"],
 "GA" -> Entity["Country", "Gabon"],
 "GB" -> Entity["Country", "UnitedKingdom"],
 "GD" -> Entity["Country", "Grenada"],
 "GE" -> Entity["Country", "Georgia"],
 "GF" -> Entity["Country", "FrenchGuiana"],
 "GG" -> Entity["Country", "Guernsey"],
 "GH" -> Entity["Country", "Ghana"],
 "GI" -> Entity["Country", "Gibraltar"],
 "GL" -> Entity["Country", "Greenland"],
 "GM" -> Entity["Country", "Gambia"],
 "GN" -> Entity["Country", "Guinea"],
 "GP" -> Entity["Country", "Guadeloupe"],
 "GQ" -> Entity["Country", "EquatorialGuinea"],
 "GR" -> Entity["Country", "Greece"],
 "GT" -> Entity["Country", "Guatemala"],
 "GU" -> Entity["Country", "Guam"],
 "GW" -> Entity["Country", "GuineaBissau"],
 "GY" -> Entity["Country", "Guyana"],
 "HK" -> Entity["Country", "HongKong"],
 "HN" -> Entity["Country", "Honduras"],
 "HR" -> Entity["Country", "Croatia"],
 "HT" -> Entity["Country", "Haiti"],
 "HU" -> Entity["Country", "Hungary"],
 "ID" -> Entity["Country", "Indonesia"],
 "IE" -> Entity["Country", "Ireland"],
 "IL" -> Entity["Country", "Israel"],
 "IM" -> Entity["Country", "IsleOfMan"],
 "IN" -> Entity["Country", "India"],
 "IQ" -> Entity["Country", "Iraq"], "IR" -> Entity["Country", "Iran"],
  "IS" -> Entity["Country", "Iceland"],
 "IT" -> Entity["Country", "Italy"],
 "JE" -> Entity["Country", "Jersey"],
 "JM" -> Entity["Country", "Jamaica"],
 "JO" -> Entity["Country", "Jordan"],
 "JP" -> Entity["Country", "Japan"],
 "KE" -> Entity["Country", "Kenya"],
 "KG" -> Entity["Country", "Kyrgyzstan"],
 "KH" -> Entity["Country", "Cambodia"],
 "KI" -> Entity["Country", "Kiribati"],
 "KM" -> Entity["Country", "Comoros"],
 "KN" -> Entity["Country", "SaintKittsNevis"],
 "KP" -> Entity["Country", "NorthKorea"],
 "KR" -> Entity["Country", "SouthKorea"],
 "KW" -> Entity["Country", "Kuwait"],
 "KY" -> Entity["Country", "CaymanIslands"],
 "KZ" -> Entity["Country", "Kazakhstan"],
 "LA" -> Entity["Country", "Laos"],
 "LB" -> Entity["Country", "Lebanon"],
 "LC" -> Entity["Country", "SaintLucia"],
 "LI" -> Entity["Country", "Liechtenstein"],
 "LK" -> Entity["Country", "SriLanka"],
 "LR" -> Entity["Country", "Liberia"],
 "LS" -> Entity["Country", "Lesotho"],
 "LT" -> Entity["Country", "Lithuania"],
 "LU" -> Entity["Country", "Luxembourg"],
 "LV" -> Entity["Country", "Latvia"],
 "LY" -> Entity["Country", "Libya"],
 "MA" -> Entity["Country", "Morocco"],
 "MC" -> Entity["Country", "Monaco"],
 "MD" -> Entity["Country", "Moldova"],
 "ME" -> Entity["Country", "Montenegro"],
 "MG" -> Entity["Country", "Madagascar"],
 "MH" -> Entity["Country", "MarshallIslands"],
 "MK" -> Entity["Country", "Macedonia"],
 "ML" -> Entity["Country", "Mali"],
 "MM" -> Entity["Country", "Myanmar"],
 "MN" -> Entity["Country", "Mongolia"],
 "MO" -> Entity["Country", "Macau"],
 "MP" -> Entity["Country", "NorthernMarianaIslands"],
 "MQ" -> Entity["Country", "Martinique"],
 "MR" -> Entity["Country", "Mauritania"],
 "MS" -> Entity["Country", "Montserrat"],
 "MT" -> Entity["Country", "Malta"],
 "MU" -> Entity["Country", "Mauritius"],
 "MV" -> Entity["Country", "Maldives"],
 "MW" -> Entity["Country", "Malawi"],
 "MX" -> Entity["Country", "Mexico"],
 "MY" -> Entity["Country", "Malaysia"],
 "MZ" -> Entity["Country", "Mozambique"],
 "NA" -> Entity["Country", "Namibia"],
 "NC" -> Entity["Country", "NewCaledonia"],
 "NE" -> Entity["Country", "Niger"],
 "NF" -> Entity["Country", "NorfolkIsland"],
 "NG" -> Entity["Country", "Nigeria"],
 "NI" -> Entity["Country", "Nicaragua"],
 "NL" -> Entity["Country", "Netherlands"],
 "NO" -> Entity["Country", "Norway"],
 "NP" -> Entity["Country", "Nepal"],
 "NR" -> Entity["Country", "Nauru"],
 "NU" -> Entity["Country", "Niue"],
 "NZ" -> Entity["Country", "NewZealand"],
 "OM" -> Entity["Country", "Oman"],
 "PA" -> Entity["Country", "Panama"],
 "PE" -> Entity["Country", "Peru"],
 "PF" -> Entity["Country", "FrenchPolynesia"],
 "PG" -> Entity["Country", "PapuaNewGuinea"],
 "PH" -> Entity["Country", "Philippines"],
 "PK" -> Entity["Country", "Pakistan"],
 "PL" -> Entity["Country", "Poland"],
 "PN" -> Entity["Country", "PitcairnIslands"],
 "PR" -> Entity["Country", "PuertoRico"],
 "PS" -> Entity["Country", "WestBank"],
 "PT" -> Entity["Country", "Portugal"],
 "PW" -> Entity["Country", "Palau"],
 "PY" -> Entity["Country", "Paraguay"],
 "QA" -> Entity["Country", "Qatar"],
 "RE" -> Entity["Country", "Reunion"],
 "RO" -> Entity["Country", "Romania"],
 "RS" -> Entity["Country", "Serbia"],
 "RU" -> Entity["Country", "Russia"],
 "RW" -> Entity["Country", "Rwanda"],
 "SA" -> Entity["Country", "SaudiArabia"],
 "SB" -> Entity["Country", "SolomonIslands"],
 "SC" -> Entity["Country", "Seychelles"],
 "SD" -> Entity["Country", "Sudan"],
 "SE" -> Entity["Country", "Sweden"],
 "SG" -> Entity["Country", "Singapore"],
 "SH" -> Entity["Country", "SaintHelena"],
 "SI" -> Entity["Country", "Slovenia"],
 "SJ" -> Entity["Country", "Svalbard"],
 "SK" -> Entity["Country", "Slovakia"],
 "SL" -> Entity["Country", "SierraLeone"],
 "SM" -> Entity["Country", "SanMarino"],
 "SN" -> Entity["Country", "Senegal"],
 "SO" -> Entity["Country", "Somalia"],
 "SR" -> Entity["Country", "Suriname"],
 "SS" -> Entity["Country", "SouthSudan"],
 "ST" -> Entity["Country", "SaoTomePrincipe"],
 "SV" -> Entity["Country", "ElSalvador"],
 "SX" -> Entity["Country", "SintMaarten"],
 "SY" -> Entity["Country", "Syria"],
 "SZ" -> Entity["Country", "Swaziland"],
 "TC" -> Entity["Country", "TurksCaicosIslands"],
 "TD" -> Entity["Country", "Chad"], "TG" -> Entity["Country", "Togo"],
  "TH" -> Entity["Country", "Thailand"],
 "TJ" -> Entity["Country", "Tajikistan"],
 "TK" -> Entity["Country", "Tokelau"],
 "TL" -> Entity["Country", "EastTimor"],
 "TM" -> Entity["Country", "Turkmenistan"],
 "TN" -> Entity["Country", "Tunisia"],
 "TO" -> Entity["Country", "Tonga"],
 "TR" -> Entity["Country", "Turkey"],
 "TT" -> Entity["Country", "TrinidadTobago"],
 "TV" -> Entity["Country", "Tuvalu"],
 "TW" -> Entity["Country", "Taiwan"],
 "TZ" -> Entity["Country", "Tanzania"],
 "UA" -> Entity["Country", "Ukraine"],
 "UG" -> Entity["Country", "Uganda"],
 "US" -> Entity["Country", "UnitedStates"],
 "UY" -> Entity["Country", "Uruguay"],
 "UZ" -> Entity["Country", "Uzbekistan"],
 "VA" -> Entity["Country", "VaticanCity"],
 "VC" -> Entity["Country", "SaintVincentGrenadines"],
 "VE" -> Entity["Country", "Venezuela"],
 "VG" -> Entity["Country", "BritishVirginIslands"],
 "VI" -> Entity["Country", "UnitedStatesVirginIslands"],
 "VN" -> Entity["Country", "Vietnam"],
 "VU" -> Entity["Country", "Vanuatu"],
 "WF" -> Entity["Country", "WallisFutuna"],
 "WS" -> Entity["Country", "Samoa"],
 "YE" -> Entity["Country", "Yemen"],
 "YT" -> Entity["Country", "Mayotte"],
 "ZA" -> Entity["Country", "SouthAfrica"],
 "ZM" -> Entity["Country", "Zambia"],
 "ZW" -> Entity["Country", "Zimbabwe"],
 "AD" -> Entity["Country", "Andorra"],
 "MF" -> Entity["Country", "SintMaarten"],
 "PM" -> Entity["Country", "SaintPierreMiquelon"]}

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{SeatGeek`Private`seatgeekdata,SeatGeek`Private`seatgeekcookeddata,SeatGeek`Private`seatgeeksendmessage}
