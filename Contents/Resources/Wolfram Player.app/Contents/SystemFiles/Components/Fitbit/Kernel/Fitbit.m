Begin["Fitbit`"]

Get["FitbitFunctions.m"]

ServiceExecute::fbtnd="No data available for the given date range."
ServiceExecute::fbrate="The rate limit of your account has been exceeded. Please retry after `1` seconds."
ServiceExecute::fbrate2="The rate limit of your account would be exceeded. Please retry with a range of up to `1` days."
ServiceExecute::fbintr="`1` request is only available for users with compatible trackers."
ServiceExecute::fbdrng="`1` should be a date that precedes `2`."
ServiceExecute::fbdval="`1` is not a valid date specification."

Begin["`Private`"]

$RateLimitQ = False;
$RateLimitResetTime = None;
$RateLimitRemaining = 150;

(******************************* Fitbit *************************************)

(* Authentication information *)

fitbitdata[]:= If[OAuthClient`Private`$UseChannelFramework,
	{
		"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "Fitbit",
		"AuthorizeEndpoint"		-> "https://www.fitbit.com/oauth2/authorize",
		"AccessEndpoint"		-> "https://api.fitbit.com/oauth2/token",
		"RedirectURI"			-> "WolframConnectorChannelListen",
		"Blocking"				-> False,
		"RedirectURLFunction"	-> (#1&),
		"AuthorizationFunction"	-> "Fitbit",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"AccessTokenRequestor"	-> "HTTPBasic",
		"RefreshAccessTokenFunction" -> "HTTPBasic",
		"VerifierLabel"			-> "code",
		"VerifyPeer"			-> True,
		"AuthenticationDialog"	:> "WolframConnectorChannel",
		"ClientInfo"			-> {"Wolfram", "Token"},
		"RequestFormat"			->(Block[{params=Lookup[{##2},"Parameters",{}], method=Lookup[{##2},"Method","GET"], body=Lookup[{##2},"BodyData",""],
											headers = Lookup[{##2},"Headers",{}], auth},
									auth = Lookup[params,"access_token",""];
									URLRead[HTTPRequest[#1,	<|"Headers" -> Join[{"Authorization" -> "Bearer " <> auth}, headers], 
										Method -> method, "Query" -> KeyDrop["access_token"][params], "Body" -> body|>],
										{"Body", "Headers", "StatusCode"}, "CredentialsProvider" -> None]
									]&),
		"Gets"					-> Join[{"FoodList", "ActivityData","SleepData", "SleepList","SleepCalendar","SleepDensityTimeline","UserData"},
									$timeseriesprops,$timeseriesplots],
		"Posts"					-> {"RecordWeight"},
		"RawGets"				-> Join[{"RawUserData", "RawWeight", "RawBodyFat", "RawMeasurements", "RawFood", "RawWater", "RawActivity", "RawSleep", "RawFoodUnit"},
									$rawtimeseriesprops],
		"RawPosts"				-> {"RawLogWeight","RawLogBodyFat","RawLogMeasurement","RawLogFood"},
		"Scope"					-> {"activity+heartrate+location+nutrition+profile+settings+sleep+social+weight"},
		"Information"			-> "A service for finding and receiving data from a Fitbit account"
	}
    ,
	{
		"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "Fitbit",
		"AuthorizeEndpoint"		-> "https://www.fitbit.com/oauth2/authorize",
		"AccessEndpoint"		-> "https://api.fitbit.com/oauth2/token",
		"RedirectURI"			-> "https://www.wolfram.com/oauthlanding?service=Fitbit",
		"AuthorizationFunction"	-> "Fitbit",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"AccessTokenRequestor"	-> "HTTPBasic",
		"RefreshAccessTokenFunction" -> "HTTPBasic",
		"VerifierLabel"			-> "code",
		"VerifyPeer"			-> True,
		"AuthenticationDialog"	:> (OAuthClient`tokenOAuthDialog[#, "Fitbit", fitbiticon]&),
		"ClientInfo"			-> {"Wolfram", "Token"},
		"RequestFormat"			-> (Block[{params=Lookup[{##2},"Parameters",{}], method=Lookup[{##2},"Method","GET"], body=Lookup[{##2},"BodyData",""],
											headers = Lookup[{##2},"Headers",{}], auth},
									auth = Lookup[params,"access_token",""];
									URLRead[HTTPRequest[#1,	<|"Headers" -> Join[{"Authorization" -> "Bearer " <> auth}, headers], 
										Method -> method, "Query" -> KeyDrop["access_token"][params], "Body" -> body|>],
										{"Body", "Headers", "StatusCode"}, "CredentialsProvider" -> None]
									]&),
		"Gets"					-> Join[{"FoodList", "ActivityData", "SleepData", "SleepList","SleepCalendar","SleepDensityTimeline","UserData"},
									$timeseriesprops,$timeseriesplots],
		"Posts"					-> {"RecordWeight"},
		"RawGets"				-> Join[{"RawUserData", "RawWeight", "RawBodyFat", "RawMeasurements", "RawFood", "RawWater", "RawActivity", "RawSleep", "RawFoodUnit"},
									$rawtimeseriesprops],
		"RawPosts"				-> {"RawLogWeight","RawLogBodyFat","RawLogMeasurement","RawLogFood"},
		"Scope"					-> {"activity+heartrate+location+nutrition+profile+settings+sleep+social+weight"},
		"Information"			-> "A service for finding and receiving data from a Fitbit account"
	}    
]
    
fitbitdata["RawUserData"]={
	"URL"				-> "https://api.fitbit.com/1/user/-/profile.json",
	"HTTPSMethod"		-> "GET",
	"Headers" 			-> {"Accept-Language"->"en_US"},
	"ResultsFunction"	-> fitbitimport
}

fitbitdata["RawMeasurements"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/body/date/`1`.json", formatDate[##]]&),
	"Headers" 			-> {"Accept-Language"->"en_US"},
	"PathParameters" 	-> {"Date"},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
}
       
fitbitdata["RawWeight"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/body/log/weight/date/`1`.json", formatDate[##]]&),
	"Headers" 			-> {"Accept-Language"->"en_US"},
	"PathParameters" 	-> {"Date"},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
}

fitbitdata["RawBodyFat"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/body/log/fat/date/`1`.json", formatDate[##]]&),
	"PathParameters" 	-> {"Date","StartDate","EndDate"},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
}
           
fitbitdata["RawFood"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/foods/log/date/`1`.json", formatDate[##]]&),
	"HTTPSMethod"		-> "GET",
	"PathParameters" 	-> {"Date"},
	"ResultsFunction"	-> fitbitimport
}
           
fitbitdata["RawWater"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/foods/log/water/date/`1`.json", formatDate[##]]&),
	"HTTPSMethod"		-> "GET",
	"PathParameters" 	-> {"Date"},
	"ResultsFunction"	-> fitbitimport
}
    
fitbitdata["RawActivity"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/activities/date/`1`.json", formatDate[##]]&),
	"PathParameters" 	-> {"Date"},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
}

fitbitdata["RawSleep"]={
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/sleep/date/`1`.json", formatDate[##]]&),
	"PathParameters" 	-> {"Date"},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
} 
           
fitbitdata["RawFoodUnit"]={
	"URL"				-> "https://api.fitbit.com/1/foods/units.json",
	"PathParameters" 	-> {},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
} 
        
fitbitdata["RawLogFood"]={
	"URL"				-> "https://api.fitbit.com/1/user/-/foods/log.json",
	"BodyData"			-> {"foodID","foodName","calories","brandName","mealTypeId","unitId","amount","date"},
	"RequiredParameters"-> {"mealTypeId","unitId","amount","date"},
	"HTTPSMethod"		-> "POST",
	"ResultsFunction"	-> fitbitimport
} 
                   
fitbitdata["RawLogWeight"]={
	"URL"				-> "https://api.fitbit.com/1/user/-/body/log/weight.json",
	"Headers" 			-> {"Accept-Language"->"en_US"},
	"Parameters"		-> {"weight","date","time"},
	"RequiredParameters"-> {"weight","date"},
	"HTTPSMethod"		-> "POST",
	"ResultsFunction"	-> fitbitimport
} 
             
fitbitdata["RawLogBodyFat"]={
	"URL"				-> "https://api.fitbit.com/1/user/-/body/log/fat.json",
	"BodyData"			-> {"fat","date"},
	"RequiredParameters"-> {"fat","date"},
	"HTTPSMethod"		-> "POST",
	"ResultsFunction"	-> fitbitimport
} 
    
fitbitdata["RawLogMeasurement"]={
	"URL"				-> "https://api.fitbit.com/1/user/-/body.json",
	"Headers" 			-> {"Accept-Language"->"en_US"},
	"BodyData"			-> {"bicep","calf","chest","fat","forearm","hips","neck","thigh","waist","weight","date"},
	"RequiredParameters"-> {"bicep"|"calf"|"chest"|"fat"|"forearm"|"hips"|"neck"|"thigh"|"waist"|"weight"},
	"HTTPSMethod"		-> "POST",
	"ResultsFunction"	-> fitbitimport
} 

fitbitdata[prop:(Alternatives@@$rawtimeseriesprops)] := {
	"URL"				-> (ToString@StringForm["https://api.fitbit.com/1/user/-/"<>Lookup[$timeseriesmapping,prop]<>"/date/`1`/`2`.json",formatDate[#1],formatDate[#2]]&),
	"Headers"			-> {"Accept-Language"->"en_US"},
	"PathParameters"	-> {"StartDate", "EndDate"},
	"RequiredParameters"-> {"StartDate", "EndDate"},
	"HTTPSMethod"		-> "GET",
	"ResultsFunction"	-> fitbitimport
} 

fitbitdata["icon"]=fitbiticon

fitbitdata[___]:=$Failed

(* a function for importing the raw data - usually json or xml - from the service *)

fitbitimport[$Failed]:=(Message[ServiceExecute::serror];Throw[$Failed])

fitbitimport[data_Association]:=Block[{status,headers,response,time,rate},

	status = data["StatusCode"];
	headers = data["Headers"];
	response = ImportString[data["Body"],"RawJSON"];

	Switch[status,
				429,
					$RateLimitQ = True;
					$RateLimitRemaining = 0;
					time = ToExpression@Lookup[headers,"retry-after"];
					$RateLimitResetTime = UnixTime[] + time;
					Message[ServiceExecute::fbrate,time];
					Throw[$Failed]
					,
				s : Except[200 | 201 | 202] /; IntegerQ[s],
					rate = ToExpression@Lookup[headers,"fitbit-rate-limit-remaining"];
					If[$RateLimitQ,
						$RateLimitRemaining = rate,
						$RateLimitRemaining = Max[Min[rate,$RateLimitRemaining-1],0]
					];
					If[SameQ[$RateLimitRemaining,0], $RateLimitQ = True, $RateLimitQ = False];
					time = ToExpression@Lookup[headers,"fitbit-rate-limit-reset"];
					$RateLimitResetTime = UnixTime[] + time;
					Message[ServiceExecute::apierr, response["errors"][[1, "message"]] ];
					Throw[$Failed]
					,
				200 | 201 | 202,
					rate = ToExpression@Lookup[headers,"fitbit-rate-limit-remaining"];
					If[$RateLimitQ,
						$RateLimitRemaining = rate,
						$RateLimitRemaining = Max[Min[rate,$RateLimitRemaining-1],0]
					];
					If[SameQ[$RateLimitRemaining,0], $RateLimitQ = True, $RateLimitQ = False];
					time = ToExpression@Lookup[headers,"fitbit-rate-limit-reset"];
					$RateLimitResetTime = UnixTime[] + time;
					response
					,
				_,
					Throw[$Failed]
		]
]

fitbitimport[___]:=Throw[$Failed]

(****** Cooked Properties ******)

fitbitcookeddata["UserData",id_,args_]:=Block[{invalidParameters,data},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	invalidParameters = Select[Keys[args],!MemberQ[{},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	data = fitbitimport[OAuthClient`rawoauthdata[id,"RawUserData",args]];
    data = Replace[data["user"], asoc_?AssociationQ :> KeyMap[StringReplace[Capitalize[#],{"Id"->"ID","OffsetFromUTCMillis"->"TimeZoneOffset"}]&][asoc], {0, Infinity}];
    data = KeyDrop[data, {"AutoStrideEnabled", "Avatar150", "DisplayNameSetting", "DistanceUnit", "Corporate", "CorporateAdmin",
    	"FoodsLocale",  "GlucoseUnit", "HeightUnit", "MfaEnabled", "Locale", "StrideLengthRunningType", "StrideLengthWalkingType",
    	"SwimUnit", "WaterUnit", "WaterUnitName", "WeightUnit"}];
    data = MapAt[Quantity[#, "Years"]&, data, {Key["Age"]}];
    If[KeyExistsQ[data,"Avatar"], AssociateTo[data, "Avatar" -> Quiet[Check[Import@#, Missing["NotAvailable"]]]& @ data["Avatar"]]];
    data = MapAt[UnitConvert[Quantity[#, "Milliseconds"], "Hours"]&, data, {Key["TimeZoneOffset"]}];
    data = MapAt[DateObject,data,{{Key["DateOfBirth"]}, {Key["MemberSince"]}}];
    data = MapAt[ToExpression@Capitalize@ToLowerCase[#]&, data, {Key["StartDayOfWeek"]}];  
    data = MapAt[UnitConvert[Quantity[#, "Inches"], MixedUnit[{"Feet", "Inches"}]]&, data, {Key["Height"]}];
    data = MapAt[UnitConvert[Quantity[#, "Inches"], "Feet"]&, data, {{Key["StrideLengthRunning"]}, {Key["StrideLengthWalking"]}}];
    data = MapAt[Quantity[#, "Pounds"]&, data, {Key["Weight"]}];
    data = MapAt[Map[formatBadge], data, {Key["TopBadges"]}];
    Dataset[data]
]

fitbitcookeddata["Bedtimes",id_,args_] :=Block[{rawdata, data},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	rawdata = getTimeSeriesRawdata["Bedtimes",id, args];
	data = fitbitimport@rawdata;
	data = (DateObject[Key["dateTime"][#]] -> If[Key["value"][#] === "", Missing["NotAvailable"], TimeObject[Key["value"][#]]])& /@ data["sleep-startTime"];
	Dataset[Association[data]]
]

fitbitcookeddata[prop:(Alternatives@@$timeseriesprops),id_,args_] :=Block[{data},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	data = getTimeSeriesRawdata[prop,id, args];
	If[MatchQ[prop, "FloorsTimeSeries"|"ElevationTimeSeries"],
		If[SameQ[data["StatusCode"],400],
			Message[ServiceExecute::fbintr,prop];
			Throw[$Failed],
			data = fitbitimport@data
		]
		,
		data = fitbitimport@data
	];
	data = formatTimeSeriesData[data];
	TimeSeries[data]
]

fitbitcookeddata[prop:(Alternatives@@$timeseriesplots),id_,args_] :=Block[{data, opts},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	opts = FilterRules[args,Except["StartDate"|"EndDate"]];
	data = getTimeSeriesRawdata[StringReplace[prop,"Plot"->"TimeSeries"],id,FilterRules[args,"StartDate"|"EndDate"]];
	If[MatchQ[prop, "FloorsPlot"|"ElevationPlot"],
		If[SameQ[data["StatusCode"],400],
			Message[ServiceExecute::fbintr,prop];
			Throw[$Failed],
			data = fitbitimport@data
		]
		,
		data = fitbitimport@data
	];
	data = formatTimeSeriesData[data];
	DateListPlot[TimeSeries@data, Sequence@@opts, Filling->Axis]
]

fitbitcookeddata["SleepList",id_,args_]:=Block[{rawdata, data},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	rawdata = getSleepRawdata[id, args];
	data = Key["sleep"] /@ fitbitimport /@ rawdata;
	data = KeyTake[#, {"awakeningsCount", "duration", "efficiency", "minutesAfterWakeup", "minutesAsleep",
			"minutesAwake", "minutesToFallAsleep", "startTime", "timeInBed"}] & /@ data;
	(Map[KeyMap[StringReplace["Minutes" -> "Time"]]] /@ Map[KeyMap[Capitalize]] /@
		MapAt[Quantity[#, "Percent"] &, {All, Key["efficiency"]}] /@
		MapAt[UnitConvert[Quantity[#, "Milliseconds"], MixedUnit[{"Hours", "Minutes", "Seconds"}]]&, {All, Key["duration"]}] /@
		MapAt[Quantity[#, "Minutes"] &, {{All, Key["minutesAfterWakeup"]}, {All, Key["minutesAsleep"]},
		{All, Key["minutesAwake"]}, {All, Key["minutesToFallAsleep"]}, {All, Key["timeInBed"]}}] /@
		MapAt[readDate, {All, Key["startTime"]}] /@ data)
]

fitbitcookeddata["SleepData",id_,args_]:=Block[{invalidParameters, date, data, sleepdata, summary},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	invalidParameters = Select[Keys[args],!MemberQ[{"Date"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	date = Lookup[args, "Date", Message[ServiceExecute::nparam,"Date"]; Throw[$Failed]];
	If[!DateObjectQ[Quiet[DateObject[date]]], Message[ServiceExecute::nval,"Date","Fitbit"]; Throw[$Failed]];

	data = fitbitimport@OAuthClient`rawoauthdata[id, "RawSleep", "Date" -> date];
	sleepdata = Key["sleep"]@data;
	summary = Key["summary"]@data;
	sleepdata = (KeyMap[StringReplace[Capitalize[#],"Minutes"->"Time"]&] /@
					MapAt[readDate, {Key["startTime"]}] /@ 
					MapAt[Quantity[#, "Minutes"] &, {{Key["minutesAfterWakeup"]}, {Key["minutesAsleep"]}, {Key["minutesAwake"]}, {Key["timeInBed"]}}] /@
					MapAt[Quantity[#, "Percent"] &, {Key["efficiency"]}] /@
					MapAt[UnitConvert[Quantity[#, "Millieconds"], MixedUnit[{"Hour", "Minute", "Second"}]] &, Key["duration"]] /@ 
 					KeyDrop[{"awakeCount", "awakeDuration", "dateOfSleep", "isMainSleep", "logId",
 						"minuteData", "minutesToFallAsleep", "restlessCount", "restlessDuration"}] /@ sleepdata);
 	summary = (MapAt[UnitConvert[Quantity[#, "Minutes"], MixedUnit[{"Hours", "Minutes"}]]&, {{Key["TotalTimeAsleep"]}, {Key["TotalTimeInBed"]}}] @
 					KeyMap[StringReplace[Capitalize[#],"Minutes"->"Time"]&] @ RotateLeft @ summary);
 	Dataset[<|"Summary" -> summary, "SleepData" -> sleepdata|>]
]

fitbitcookeddata["SleepCalendar",id_,args_]:=Block[{rawdata,data,starts,ends,durations,effs,minEfficiency,maxEfficiency,period,ndays,t0,i},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	rawdata = getSleepRawdata[id, args];
	data = Key["sleep"] /@ fitbitimport /@ rawdata;
	starts = readDate[#,Identity]& /@ Flatten[ Map[Key["startTime"]] /@ data ];
	durations = Flatten[ Map[ToExpression[Key["timeInBed"][#]]&] /@ data ];
	effs = Flatten[ Map[ToExpression[Key["efficiency"][#]]&] /@ data ];
	data = Cases[Transpose[{starts, durations, effs}], {_, Repeated[_?NumberQ, 2]}];
	If[data === {}, Throw[Graphics[]]];
	minEfficiency = Min @ data[[All, 3]];
	maxEfficiency = Max @ data[[All, 3]];
	starts = data[[All, 1]];
	durations = data[[All, 2]];
	ends = starts; ends[[All, 5]] = ends[[All, 5]] + durations;
	period = DayRange @@ DateBounds[Join[starts,ends]];
	ndays = Length[period];
	t0 = AbsoluteTime@First@period;
	Graphics[
		makeSleepBox[#, minEfficiency, maxEfficiency, t0]& /@ data,
		AspectRatio -> 1/GoldenRatio,
		Axes -> True,
		AxesOrigin->{0,0},
		Ticks -> {Range[-2, 24], Table[{i+.5,DateString[DatePlus[t0,{i,"Day"}], {"MonthNameShort", " ", "Day", " ", "Year"}]},{i,0,ndays-1}]}
	]
]

fitbitcookeddata["SleepDensityTimeline",id_,args_]:=Block[{rawdata,data,starts,ends,durations,effs,minEfficiency,maxEfficiency,period,ndays},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	rawdata = getSleepRawdata[id, args];
	data = Key["sleep"] /@ fitbitimport /@ rawdata;
	data = DeleteCases[data,{}];
	starts = readDate[#,Identity]& /@ Flatten[ Map[Key["startTime"]] /@ data ];
	durations = Flatten[ Map[ToExpression[Key["timeInBed"][#]]&] /@ data ];
	effs = Flatten[ Map[ToExpression[Key["efficiency"][#]]&] /@ data ];
	data = Cases[Transpose[{starts, durations, effs}], {_, Repeated[_?NumberQ, 2]}];
	If[data === {}, Throw[Graphics[{},Axes -> True,AxesOrigin->{0,0},Ticks -> {Range[-2, 24], {1}}]]];
	minEfficiency = Min @ data[[All, 3]];
	maxEfficiency = Max @ data[[All, 3]];
	starts = data[[All, 1]];
	durations = data[[All, 2]];
	ends = starts; ends[[All, 5]] = ends[[All, 5]] + durations;
	period = DayRange @@ DateBounds[Join[starts,ends]];
	ndays = Length[period];
	Graphics[
		makeSleepBox[#, minEfficiency, maxEfficiency, AbsoluteTime[#[[1,1;;3]]],True]&/@data,
		Axes -> True,
		AxesOrigin->{0,0},
		Ticks -> {Range[-2, 24], {1}}
	]
]

fitbitcookeddata["FoodList",id_,args_]:=Block[{invalidParameters,date,data},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	invalidParameters = Select[Keys[args],!MemberQ[{"Date"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	date = Lookup[args, "Date", Today];
	If[!DateObjectQ[Quiet[DateObject[date]]], Message[ServiceExecute::nval,"Date","Fitbit"]; Throw[$Failed]];

	data = fitbitimport@OAuthClient`rawoauthdata[id, "RawFood", "Date" -> date];
	data = Map[Key["loggedFood"]]@Lookup[data,"foods",{}];
	data = KeyTake[{"logId", "logDate", "nutritionalValues", "amount","foodID","unit","brand","calories","mealTypeId","name","units"}] /@ data;
	Replace[data, asoc_Association :> KeyMap[StringReplace[Capitalize[#], "Id" -> "ID"] &][asoc], Infinity]
]

fitbitcookeddata["ActivityData",id_,args_]:=Block[{invalidParameters,date,data,distances},
	
	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	invalidParameters = Select[Keys[args],!MemberQ[{"Date"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	date = Lookup[args, "Date", Today];
	If[!DateObjectQ[Quiet[DateObject[date]]], Message[ServiceExecute::nval,"Date","Fitbit"]; Throw[$Failed]];

	data = fitbitimport@OAuthClient`rawoauthdata[id, "RawActivity", "Date" -> date];
	distances = Lookup[data["summary"], "distances", {}];
	distances = Association[(Capitalize@#["activity"] -> Quantity[#["distance"], "Miles"])& /@ distances];
	data = KeyDrop[data["summary"], "distances"];
	If[KeyExistsQ[data, "elevation"], data = MapAt[Quantity[#, "Feet"] &, data, {Key["elevation"]}]];
	If[KeyExistsQ[data, "restingHeartRate"], data = MapAt[Quantity[#, Times[Power["Minutes",-1],IndependentUnit["beats"]]]&, data, {Key["restingHeartRate"]}]];
	If[KeyExistsQ[data, "heartRateZones"], data = MapAt[Map[formatHeartRateZone], data, {Key["heartRateZones"]}]];
	data = MapAt[Quantity[#, "LargeCalories"]&, data, {{Key["activityCalories"]},{Key["caloriesBMR"]},{Key["caloriesOut"]},{Key["marginalCalories"]}}];
	data = MapAt[Quantity[#, "Minutes"]&, data, {{Key["fairlyActiveMinutes"]},{Key["lightlyActiveMinutes"]},{Key["sedentaryMinutes"]},{Key["veryActiveMinutes"]}}];
	data = KeyMap[Capitalize]@data;
	Dataset@Join[data, distances]
]

fitbitcookeddata["RecordWeight",id_,args_]:=Block[{invalidParameters,date,time,weight,data},

	If[$RateLimitQ && TrueQ[$RateLimitResetTime>UnixTime[]],
		Message[ServiceExecute:fbrate,$RateLimitResetTime-UnixTime[]];
		Throw[$Failed] 
	];

	invalidParameters = Select[Keys[args],!MemberQ[{"Date","Weight"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];

	date = Lookup[args, "Date", Now];
	If[!DateObjectQ[Quiet[DateObject[date]]], Message[ServiceExecute::nval,"Date","Fitbit"]; Throw[$Failed]];
	{date,time} = formatDateTime@date;

	weight = Lookup[args, "Weight", Message[ServiceExecute::nparam, "Weight"]; Throw[$Failed]];
	If[QuantityQ[weight],
		If[Quiet[CompatibleUnitQ[weight, "Pound"]],
			weight = ToString@QuantityMagnitude@UnitConvert[weight,"Pounds"],
			Message[ServiceExecute::nval,"Weight","Fitbit"];
			Throw[$Failed]
		]
	];
	
	If[NumericQ[weight],
		If[weight < 1500,
			weight = ToString@N@weight,
			Message[ServiceExecute::nval,"Weight","Fitbit"];
			Throw[$Failed]
		]
	];
	
	If[!StringQ[weight] || !StringMatchQ[weight, (DigitCharacter|".")..],
		Message[ServiceExecute::nval,"Weight","Fitbit"];
		Throw[$Failed]
	];

	data = fitbitimport[OAuthClient`rawoauthdata[id, "RawLogWeight", {"weight"->weight,"date"->date}]];
	data["weightLog"]["weight"]
]

fitbitcookeddata[___]:=$Failed

(* Send Message *)
fitbitsendmessage[___]:=$Failed

fitbiticon=Image[RawArray["Byte", {{{255, 255, 255, 0}, {255, 255, 255, 3}, {255, 255, 255, 0}, {255, 255, 255, 15}, {254, 254, 
  254, 94}, {253, 253, 253, 170}, {253, 253, 253, 220}, {253, 253, 253, 246}, {253, 253, 253, 255}, {253, 253, 253, 
  255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, 
  {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 
  253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 255}, {253, 253, 253, 246}, {253, 253, 
  253, 220}, {253, 253, 253, 170}, {254, 254, 254, 94}, {255, 255, 255, 15}, {255, 255, 255, 0}, {255, 255, 255, 3}, 
  {255, 255, 255, 0}}, {{255, 255, 255, 3}, {255, 255, 255, 0}, {254, 254, 254, 76}, {252, 252, 252, 229}, {252, 252, 
  252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 
  255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, 
  {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 
  252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 
  252, 255}, {252, 252, 252, 255}, {252, 252, 252, 255}, {252, 252, 252, 229}, {254, 254, 254, 76}, {255, 255, 255, 
  0}, {255, 255, 255, 3}}, {{255, 255, 255, 0}, {254, 254, 254, 73}, {251, 251, 251, 254}, {251, 251, 251, 255}, 
  {251, 251, 251, 250}, {251, 251, 251, 252}, {251, 251, 251, 254}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 
  251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 
  251, 255}, {249, 251, 251, 255}, {249, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 
  255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, 
  {251, 251, 251, 254}, {251, 251, 251, 252}, {251, 251, 251, 250}, {251, 251, 251, 255}, {251, 251, 251, 254}, {254, 
  254, 254, 73}, {255, 255, 255, 0}}, {{255, 255, 255, 13}, {251, 251, 251, 234}, {250, 250, 250, 255}, {251, 251, 
  251, 252}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 
  255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {250, 250, 250, 255}, 
  {251, 251, 251, 255}, {255, 254, 254, 255}, {255, 254, 254, 255}, {251, 251, 251, 255}, {250, 250, 250, 255}, {251, 
  251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 
  251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 255}, {251, 251, 251, 252}, {250, 250, 250, 
  255}, {251, 251, 251, 234}, {255, 255, 255, 13}}, {{253, 253, 253, 93}, {249, 249, 249, 255}, {250, 250, 250, 251}, 
  {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 
  250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {249, 250, 250, 255}, {250, 250, 
  250, 255}, {250, 250, 250, 255}, {180, 229, 229, 255}, {180, 229, 229, 255}, {250, 250, 250, 255}, {250, 250, 250, 
  255}, {249, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, 
  {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 250, 250, 255}, {250, 
  250, 250, 251}, {249, 249, 249, 255}, {253, 253, 253, 93}}, {{251, 251, 251, 172}, {248, 248, 248, 254}, {249, 249, 
  249, 252}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 
  255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {246, 248, 248, 255}, 
  {255, 252, 252, 255}, {163, 223, 223, 255}, {45, 188, 188, 255}, {45, 188, 188, 255}, {163, 223, 223, 255}, {255, 
  252, 252, 255}, {246, 248, 248, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 
  249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 255}, {249, 249, 249, 
  255}, {249, 249, 249, 252}, {248, 248, 248, 254}, {251, 251, 251, 172}}, {{248, 248, 248, 222}, {248, 248, 248, 
  255}, {248, 248, 248, 254}, {248, 248, 248, 255}, {247, 247, 247, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, 
  {248, 248, 248, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, {244, 
  246, 246, 255}, {255, 251, 251, 255}, {134, 214, 214, 255}, {43, 188, 188, 255}, {43, 188, 188, 255}, {134, 214, 
  214, 255}, {255, 251, 251, 255}, {244, 246, 246, 255}, {247, 247, 247, 255}, {248, 248, 248, 255}, {248, 248, 248, 
  255}, {248, 248, 248, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, {248, 248, 248, 255}, 
  {248, 248, 248, 255}, {248, 248, 248, 254}, {247, 247, 247, 255}, {248, 248, 248, 222}}, {{246, 246, 246, 248}, 
  {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 
  246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {244, 246, 246, 255}, {243, 246, 
  246, 255}, {245, 246, 246, 255}, {250, 248, 248, 255}, {227, 241, 241, 255}, {112, 207, 207, 255}, {112, 207, 207, 
  255}, {227, 241, 241, 255}, {250, 248, 248, 255}, {245, 245, 245, 255}, {243, 243, 243, 255}, {244, 244, 244, 255}, 
  {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 
  246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 255}, {246, 246, 246, 248}}, {{245, 
  245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 
  245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {246, 245, 245, 255}, {255, 249, 249, 
  255}, {255, 249, 249, 255}, {247, 246, 246, 255}, {244, 245, 245, 255}, {251, 247, 247, 255}, {255, 251, 251, 255}, 
  {255, 251, 251, 255}, {251, 247, 247, 255}, {244, 244, 244, 255}, {248, 247, 247, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {246, 246, 246, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 
  245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 255}, {245, 245, 245, 
  255}}, {{244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 
  255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {245, 244, 244, 255}, 
  {168, 223, 223, 255}, {155, 219, 219, 255}, {237, 242, 242, 255}, {246, 245, 245, 255}, {240, 243, 243, 255}, {159, 
  220, 220, 255}, {159, 220, 220, 255}, {240, 242, 242, 255}, {246, 246, 246, 255}, {236, 237, 237, 255}, {149, 149, 
  149, 255}, {164, 164, 164, 255}, {245, 245, 245, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 
  255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, {244, 244, 244, 255}, 
  {244, 244, 244, 255}}, {{243, 243, 243, 255}, {243, 243, 243, 255}, {243, 243, 243, 255}, {243, 243, 243, 255}, 
  {243, 243, 243, 255}, {243, 243, 243, 255}, {243, 243, 243, 255}, {240, 242, 242, 255}, {254, 246, 246, 255}, {167, 
  222, 222, 255}, {45, 188, 188, 255}, {43, 188, 188, 255}, {134, 213, 213, 255}, {255, 249, 249, 255}, {149, 217, 
  217, 255}, {44, 188, 188, 255}, {43, 188, 188, 255}, {149, 215, 215, 255}, {255, 255, 255, 255}, {127, 129, 129, 
  255}, {31, 31, 31, 255}, {33, 33, 33, 255}, {163, 163, 163, 255}, {254, 254, 254, 255}, {240, 240, 240, 255}, {243, 
  243, 243, 255}, {243, 243, 243, 255}, {243, 243, 243, 255}, {243, 243, 243, 255}, {243, 243, 243, 255}, {243, 243, 
  243, 255}, {243, 243, 243, 255}}, {{241, 241, 241, 255}, {241, 241, 241, 255}, {241, 241, 241, 255}, {241, 241, 
  241, 255}, {241, 241, 241, 255}, {241, 241, 241, 255}, {241, 241, 241, 255}, {238, 240, 240, 255}, {255, 245, 245, 
  255}, {152, 217, 217, 255}, {43, 188, 188, 255}, {45, 188, 188, 255}, {119, 208, 208, 255}, {255, 248, 248, 255}, 
  {134, 212, 212, 255}, {44, 188, 188, 255}, {44, 188, 188, 255}, {134, 211, 211, 255}, {255, 255, 255, 255}, {111, 
  113, 113, 255}, {32, 32, 32, 255}, {31, 31, 31, 255}, {148, 148, 148, 255}, {255, 255, 255, 255}, {238, 238, 238, 
  255}, {241, 241, 241, 255}, {241, 241, 241, 255}, {241, 241, 241, 255}, {241, 241, 241, 255}, {241, 241, 241, 255}, 
  {241, 241, 241, 255}, {241, 241, 241, 255}}, {{240, 240, 240, 255}, {240, 240, 240, 255}, {240, 240, 240, 255}, 
  {240, 240, 240, 255}, {240, 240, 240, 255}, {237, 237, 237, 255}, {237, 237, 237, 255}, {238, 239, 239, 255}, {242, 
  241, 241, 255}, {233, 238, 238, 255}, {132, 209, 209, 255}, {118, 205, 205, 255}, {220, 234, 234, 255}, {245, 242, 
  242, 255}, {227, 236, 236, 255}, {124, 207, 207, 255}, {124, 207, 207, 255}, {226, 235, 235, 255}, {245, 245, 245, 
  255}, {219, 219, 219, 255}, {110, 110, 110, 255}, {126, 126, 126, 255}, {233, 233, 233, 255}, {242, 242, 242, 255}, 
  {238, 238, 238, 255}, {237, 237, 237, 255}, {237, 237, 237, 255}, {240, 240, 240, 255}, {240, 240, 240, 255}, {240, 
  240, 240, 255}, {240, 240, 240, 255}, {240, 240, 240, 255}}, {{239, 239, 239, 255}, {239, 239, 239, 255}, {239, 
  239, 239, 255}, {238, 238, 238, 255}, {239, 239, 239, 255}, {250, 250, 250, 255}, {252, 252, 252, 255}, {243, 243, 
  243, 255}, {237, 237, 237, 255}, {243, 241, 241, 255}, {255, 254, 254, 255}, {255, 255, 255, 255}, {246, 243, 243, 
  255}, {236, 237, 237, 255}, {245, 242, 242, 255}, {255, 254, 254, 255}, {255, 254, 254, 255}, {245, 242, 242, 255}, 
  {236, 237, 237, 255}, {247, 247, 247, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {243, 243, 243, 255}, {237, 
  237, 237, 255}, {243, 243, 243, 255}, {252, 252, 252, 255}, {250, 250, 250, 255}, {239, 239, 239, 255}, {239, 239, 
  239, 255}, {239, 239, 239, 255}, {239, 239, 239, 255}, {239, 239, 239, 255}}, {{237, 237, 237, 255}, {237, 237, 
  237, 255}, {237, 237, 237, 255}, {237, 237, 237, 255}, {239, 239, 239, 255}, {152, 152, 152, 255}, {124, 124, 124, 
  255}, {218, 218, 218, 255}, {242, 242, 242, 255}, {234, 234, 234, 255}, {141, 143, 143, 255}, {127, 129, 129, 255}, 
  {224, 225, 225, 255}, {242, 242, 242, 255}, {229, 230, 230, 255}, {134, 136, 136, 255}, {134, 136, 136, 255}, {229, 
  230, 230, 255}, {242, 242, 242, 255}, {223, 223, 223, 255}, {127, 127, 127, 255}, {142, 142, 142, 255}, {234, 234, 
  234, 255}, {242, 242, 242, 255}, {218, 218, 218, 255}, {124, 124, 124, 255}, {152, 152, 152, 255}, {239, 239, 239, 
  255}, {237, 237, 237, 255}, {237, 237, 237, 255}, {237, 237, 237, 255}, {237, 237, 237, 255}}, {{236, 236, 236, 
  255}, {236, 236, 236, 255}, {233, 233, 233, 255}, {248, 248, 248, 255}, {167, 167, 167, 255}, {34, 34, 34, 255}, 
  {32, 32, 32, 255}, {101, 101, 101, 255}, {255, 255, 255, 255}, {149, 149, 149, 255}, {32, 32, 32, 255}, {32, 32, 
  32, 255}, {117, 117, 117, 255}, {255, 255, 255, 255}, {133, 133, 133, 255}, {31, 31, 31, 255}, {31, 31, 31, 255}, 
  {133, 133, 133, 255}, {255, 255, 255, 255}, {116, 116, 116, 255}, {32, 32, 32, 255}, {32, 32, 32, 255}, {150, 150, 
  150, 255}, {255, 255, 255, 255}, {101, 101, 101, 255}, {32, 32, 32, 255}, {34, 34, 34, 255}, {167, 167, 167, 255}, 
  {248, 248, 248, 255}, {233, 233, 233, 255}, {236, 236, 236, 255}, {236, 236, 236, 255}}, {{234, 234, 234, 255}, 
  {234, 234, 234, 255}, {232, 232, 232, 255}, {246, 246, 246, 255}, {166, 166, 166, 255}, {34, 34, 34, 255}, {33, 33, 
  33, 255}, {100, 100, 100, 255}, {255, 255, 255, 255}, {148, 148, 148, 255}, {32, 32, 32, 255}, {32, 32, 32, 255}, 
  {116, 116, 116, 255}, {255, 255, 255, 255}, {132, 132, 132, 255}, {32, 32, 32, 255}, {32, 32, 32, 255}, {132, 132, 
  132, 255}, {255, 255, 255, 255}, {115, 115, 115, 255}, {32, 32, 32, 255}, {32, 32, 32, 255}, {149, 149, 149, 255}, 
  {255, 255, 255, 255}, {100, 100, 100, 255}, {33, 33, 33, 255}, {34, 34, 34, 255}, {166, 166, 166, 255}, {246, 246, 
  246, 255}, {232, 232, 232, 255}, {235, 235, 235, 255}, {234, 234, 234, 255}}, {{233, 233, 233, 255}, {233, 233, 
  233, 255}, {233, 233, 233, 255}, {233, 233, 233, 255}, {235, 235, 235, 255}, {149, 149, 149, 255}, {121, 121, 121, 
  255}, {214, 214, 214, 255}, {239, 238, 238, 255}, {229, 230, 230, 255}, {139, 140, 140, 255}, {125, 126, 126, 255}, 
  {219, 220, 220, 255}, {238, 238, 238, 255}, {225, 225, 225, 255}, {131, 133, 133, 255}, {131, 133, 133, 255}, {225, 
  225, 225, 255}, {238, 238, 238, 255}, {219, 219, 219, 255}, {125, 125, 125, 255}, {139, 139, 139, 255}, {229, 229, 
  229, 255}, {238, 238, 238, 255}, {214, 214, 214, 255}, {121, 121, 121, 255}, {149, 149, 149, 255}, {235, 235, 235, 
  255}, {233, 233, 233, 255}, {233, 233, 233, 255}, {233, 233, 233, 255}, {233, 233, 233, 255}}, {{232, 232, 232, 
  255}, {232, 232, 232, 255}, {232, 232, 232, 255}, {231, 231, 231, 255}, {232, 232, 232, 255}, {243, 243, 243, 255}, 
  {245, 245, 245, 255}, {236, 236, 236, 255}, {230, 230, 230, 255}, {237, 234, 234, 255}, {253, 246, 246, 255}, {253, 
  247, 247, 255}, {239, 236, 236, 255}, {229, 230, 230, 255}, {237, 235, 235, 255}, {255, 247, 247, 255}, {255, 247, 
  247, 255}, {237, 235, 235, 255}, {229, 229, 229, 255}, {240, 240, 240, 255}, {254, 254, 254, 255}, {253, 253, 253, 
  255}, {237, 237, 237, 255}, {230, 230, 230, 255}, {236, 236, 236, 255}, {245, 245, 245, 255}, {243, 243, 243, 255}, 
  {232, 232, 232, 255}, {231, 231, 231, 255}, {232, 232, 232, 255}, {232, 232, 232, 255}, {232, 232, 232, 255}}, 
  {{230, 230, 230, 255}, {230, 230, 230, 255}, {230, 230, 230, 255}, {230, 230, 230, 255}, {230, 230, 230, 255}, 
  {228, 228, 228, 255}, {227, 227, 227, 255}, {228, 229, 229, 255}, {234, 231, 231, 255}, {216, 227, 227, 255}, {113, 
  202, 202, 255}, {109, 201, 201, 255}, {210, 225, 225, 255}, {238, 232, 232, 255}, {218, 227, 227, 255}, {121, 204, 
  204, 255}, {121, 204, 204, 255}, {217, 226, 226, 255}, {239, 236, 236, 255}, {204, 205, 205, 255}, {97, 97, 97, 
  255}, {111, 111, 111, 255}, {219, 219, 219, 255}, {233, 233, 233, 255}, {228, 228, 228, 255}, {227, 227, 227, 255}, 
  {228, 228, 228, 255}, {230, 230, 230, 255}, {230, 230, 230, 255}, {230, 230, 230, 255}, {230, 230, 230, 255}, {230, 
  230, 230, 255}}, {{229, 229, 229, 255}, {229, 229, 229, 255}, {229, 229, 229, 255}, {229, 229, 229, 255}, {229, 
  229, 229, 255}, {229, 229, 229, 255}, {229, 229, 229, 255}, {226, 228, 228, 255}, {241, 231, 231, 255}, {129, 207, 
  207, 255}, {45, 189, 189, 255}, {46, 189, 189, 255}, {117, 205, 205, 255}, {252, 234, 234, 255}, {129, 208, 208, 
  255}, {44, 189, 189, 255}, {44, 189, 189, 255}, {129, 206, 206, 255}, {251, 242, 242, 255}, {100, 102, 102, 255}, 
  {35, 35, 35, 255}, {32, 32, 32, 255}, {133, 133, 133, 255}, {242, 242, 242, 255}, {226, 226, 226, 255}, {229, 229, 
  229, 255}, {229, 229, 229, 255}, {229, 229, 229, 255}, {229, 229, 229, 255}, {229, 229, 229, 255}, {229, 229, 229, 
  255}, {229, 229, 229, 255}}, {{228, 228, 228, 255}, {228, 228, 228, 255}, {228, 228, 228, 255}, {228, 228, 228, 
  255}, {228, 228, 228, 255}, {228, 228, 228, 255}, {228, 228, 228, 255}, {225, 227, 227, 255}, {238, 230, 230, 255}, 
  {146, 210, 210, 255}, {46, 190, 190, 255}, {45, 189, 189, 255}, {134, 208, 208, 255}, {249, 232, 232, 255}, {142, 
  210, 210, 255}, {45, 190, 190, 255}, {45, 190, 190, 255}, {142, 208, 208, 255}, {249, 240, 240, 255}, {118, 120, 
  120, 255}, {32, 32, 32, 255}, {35, 35, 35, 255}, {150, 150, 150, 255}, {239, 239, 239, 255}, {225, 225, 225, 255}, 
  {228, 228, 228, 255}, {228, 228, 228, 255}, {228, 228, 228, 255}, {228, 228, 228, 255}, {228, 228, 228, 255}, {228, 
  228, 228, 255}, {228, 228, 228, 255}}, {{226, 226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {226, 
  226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {227, 227, 
  227, 255}, {225, 226, 226, 255}, {153, 211, 211, 255}, {149, 211, 211, 255}, {223, 226, 226, 255}, {229, 227, 227, 
  255}, {223, 226, 226, 255}, {150, 211, 211, 255}, {150, 211, 211, 255}, {223, 225, 225, 255}, {230, 229, 229, 255}, 
  {220, 220, 220, 255}, {140, 140, 140, 255}, {152, 152, 152, 255}, {227, 227, 227, 255}, {227, 227, 227, 255}, {226, 
  226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}, {226, 226, 
  226, 255}, {226, 226, 226, 255}, {226, 226, 226, 255}}, {{225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 
  225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 
  255}, {225, 225, 225, 255}, {226, 225, 225, 255}, {236, 227, 227, 255}, {237, 228, 228, 255}, {227, 225, 225, 255}, 
  {224, 225, 225, 255}, {230, 226, 226, 255}, {244, 229, 229, 255}, {244, 229, 229, 255}, {230, 226, 226, 255}, {223, 
  224, 224, 255}, {227, 227, 227, 255}, {238, 238, 238, 255}, {237, 237, 237, 255}, {226, 226, 226, 255}, {225, 225, 
  225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 
  255}, {225, 225, 225, 255}, {225, 225, 225, 255}, {225, 225, 225, 255}}, {{224, 224, 224, 248}, {224, 224, 224, 
  255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, 
  {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {221, 223, 223, 255}, {221, 223, 223, 255}, {222, 
  224, 224, 255}, {228, 225, 225, 255}, {207, 221, 221, 255}, {106, 201, 201, 255}, {106, 201, 201, 255}, {207, 221, 
  221, 255}, {228, 225, 225, 255}, {222, 223, 223, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {224, 224, 224, 
  255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, 
  {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 255}, {224, 224, 224, 248}}, {{226, 226, 226, 222}, 
  {222, 222, 222, 255}, {223, 223, 223, 254}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 
  223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 
  223, 255}, {220, 222, 222, 255}, {235, 225, 225, 255}, {124, 205, 205, 255}, {46, 189, 189, 255}, {46, 190, 190, 
  255}, {124, 205, 205, 255}, {235, 225, 225, 255}, {220, 222, 222, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, 
  {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 255}, {223, 
  223, 223, 255}, {223, 223, 223, 255}, {223, 223, 223, 254}, {222, 222, 222, 255}, {226, 226, 226, 222}}, {{232, 
  232, 232, 172}, {220, 220, 220, 254}, {222, 222, 222, 252}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 
  222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 
  255}, {222, 222, 222, 255}, {219, 221, 221, 255}, {232, 224, 224, 255}, {148, 209, 209, 255}, {48, 190, 190, 255}, 
  {47, 190, 190, 255}, {148, 208, 208, 255}, {232, 224, 224, 255}, {219, 221, 221, 255}, {222, 222, 222, 255}, {222, 
  222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 
  222, 255}, {222, 222, 222, 255}, {222, 222, 222, 255}, {222, 222, 222, 252}, {220, 220, 220, 254}, {232, 232, 232, 
  172}}, {{241, 241, 241, 93}, {218, 218, 218, 255}, {221, 221, 221, 251}, {221, 221, 221, 255}, {221, 221, 221, 
  255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, 
  {221, 221, 221, 255}, {221, 221, 221, 255}, {220, 221, 221, 255}, {221, 221, 221, 255}, {222, 221, 221, 255}, {161, 
  210, 210, 255}, {161, 210, 210, 255}, {222, 221, 221, 255}, {221, 221, 221, 255}, {220, 221, 221, 255}, {221, 221, 
  221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 
  255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 255}, {221, 221, 221, 251}, {218, 218, 218, 255}, 
  {242, 242, 242, 93}}, {{253, 253, 253, 13}, {222, 222, 222, 234}, {219, 219, 219, 255}, {220, 220, 220, 252}, {220, 
  220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 
  220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 
  255}, {230, 221, 221, 255}, {230, 222, 222, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, 
  {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 
  220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 255}, {220, 220, 220, 252}, {219, 219, 219, 255}, {222, 222, 
  222, 234}, {253, 253, 253, 13}}, {{255, 255, 255, 0}, {244, 244, 244, 73}, {217, 217, 217, 254}, {218, 218, 218, 
  255}, {220, 220, 220, 250}, {219, 219, 219, 252}, {219, 219, 219, 254}, {219, 219, 219, 255}, {219, 219, 219, 255}, 
  {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 
  219, 219, 255}, {217, 218, 218, 255}, {217, 218, 218, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 
  219, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 219, 255}, {219, 219, 219, 
  255}, {219, 219, 219, 254}, {219, 219, 219, 252}, {220, 220, 220, 250}, {218, 218, 218, 255}, {217, 217, 217, 254}, 
  {244, 244, 244, 73}, {255, 255, 255, 0}}, {{255, 255, 255, 3}, {255, 255, 255, 0}, {243, 243, 243, 76}, {220, 220, 
  220, 229}, {215, 215, 215, 255}, {216, 216, 216, 255}, {217, 217, 217, 255}, {218, 218, 218, 255}, {218, 218, 218, 
  255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, 
  {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 
  218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 218, 255}, {218, 218, 
  218, 255}, {217, 217, 217, 255}, {216, 216, 216, 255}, {215, 215, 215, 255}, {220, 220, 220, 229}, {243, 243, 243, 
  76}, {255, 255, 255, 0}, {255, 255, 255, 3}}, {{255, 255, 255, 0}, {255, 255, 255, 3}, {255, 255, 255, 0}, {253, 
  253, 253, 15}, {240, 240, 240, 94}, {229, 229, 229, 170}, {222, 222, 222, 220}, {218, 218, 218, 246}, {217, 217, 
  217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 
  255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, 
  {218, 218, 218, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {217, 217, 217, 255}, {218, 
  218, 218, 246}, {222, 222, 222, 220}, {229, 229, 229, 170}, {240, 240, 240, 94}, {253, 253, 253, 15}, {255, 255, 
  255, 0}, {255, 255, 255, 3}, {255, 255, 255, 0}}}], "Byte", ColorSpace -> "RGB", Interleaving -> True];
  
End[] (* End Private Context *)
           		
End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{Fitbit`Private`fitbitdata,Fitbit`Private`fitbitcookeddata,Fitbit`Private`fitbitsendmessage}
