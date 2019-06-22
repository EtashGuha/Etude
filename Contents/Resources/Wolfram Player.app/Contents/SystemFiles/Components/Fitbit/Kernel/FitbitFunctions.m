BeginPackage["FitbitFunctions`"] 

$timeseriesprops::usage="";
$rawtimeseriesprops::usage="";
$timeseriespaths::usage="";
$timeseriesplots::usage="";
$timeseriesmapping::usage="";
getTimeSeriesRawdata::usage="";
formatTimeSeriesData::usage="";
getSleepRawdata::usage="";
makeSleepBox::usage="";
formatHeartRateZone::usage="";
formatBadge::usage="";
formatDate::usage="";
formatDateTime::usage="";
readDate::usage="";

Begin["`Private`"]

$timeseriesprops={"CaloriesInTimeSeries", "WaterTimeSeries", "CaloriesTimeSeries", "CaloriesBMRTimeSeries", "StepsTimeSeries", "DistanceTimeSeries", 
 "FloorsTimeSeries", "ElevationTimeSeries", "MinutesSedentaryTimeSeries", "MinutesLightlyActiveTimeSeries", "MinutesFairlyActiveTimeSeries",
 "MinutesVeryActiveTimeSeries", "ActivityCaloriesTimeSeries", (* "CaloriesTimeSeries", "StepsTimeSeries", "DistanceTimeSeries", "MinutesSedentaryTimeSeries", 
 "MinutesLightlyActiveTimeSeries", "MinutesFairlyActiveTimeSeries", "MinutesVeryActiveTimeSeries", "ActivityCaloriesTimeSeries",*) "Bedtimes",
 "TimeInBedTimeSeries", "MinutesAsleepTimeSeries", "AwakeningsCountTimeSeries", "MinutesAwakeTimeSeries", "MinutesToFallAsleepTimeSeries",
 "MinutesAfterWakeupTimeSeries", "SleepEfficiencyTimeSeries", "WeightTimeSeries", "BMITimeSeries", "BodyFatTimeSeries"}

$rawtimeseriesprops = ("Raw"<>#&/@$timeseriesprops)

$timeseriespaths={"foods/log/caloriesIn", "foods/log/water", "activities/calories", "activities/caloriesBMR", "activities/steps", "activities/distance", 
 "activities/floors", "activities/elevation", "activities/minutesSedentary", "activities/minutesLightlyActive", "activities/minutesFairlyActive",
 "activities/minutesVeryActive", "activities/activityCalories", (* "activities/tracker/calories", "activities/tracker/steps","activities/tracker/distance",
 "activities/tracker/floors", "activities/tracker/elevation", "activities/tracker/minutesSedentary", "activities/tracker/minutesLightlyActive",
 "activities/tracker/minutesFairlyActive", "activities/tracker/minutesVeryActive", "activities/tracker/activityCalories",*) "sleep/startTime",
 "sleep/timeInBed", "sleep/minutesAsleep", "sleep/awakeningsCount", "sleep/minutesAwake", "sleep/minutesToFallAsleep", "sleep/minutesAfterWakeup",
 "sleep/efficiency", "body/weight", "body/bmi", "body/fat"}

$timeseriesplots = DeleteCases[StringReplace[$timeseriesprops, "TimeSeries" -> "Plot"], "Bedtimes"]

$timeseriesmapping = Thread[$rawtimeseriesprops->$timeseriespaths]

getTimeSeriesRawdata[prop_,id_,args_]:=Block[{invalidParameters, rawprop = ("Raw" <> prop), start, end, rawdata},

	invalidParameters = Select[Keys[args],!MemberQ[{"StartDate", "EndDate"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	start = Lookup[args, "StartDate", Message[ServiceExecute::nparam,"StartDate"]; Throw[$Failed]];
	If[!DateObjectQ[Quiet[DateObject[start]]], Message[ServiceExecute::nval,"StartDate","Fitbit"]; Throw[$Failed]];
	end = Lookup[args, "EndDate", DateString["ISODate"]];
	If[!DateObjectQ[Quiet[DateObject[end]]], Message[ServiceExecute::nval,"EndDate","Fitbit"]; Throw[$Failed]];

	If[ !TrueQ[First@DateDifference[start, end] > 0],
		Message[ServiceExecute::fbdrng,start,end];
		Throw[$Failed]
	];

	OAuthClient`rawoauthdata[id, rawprop, {"StartDate"-> start,"EndDate"-> end}]

];

formatTimeSeriesData[data_]:= MapThread[{AbsoluteTime@#1,ToExpression@#2}&, {Key["dateTime"]/@#[[1]], Key["value"]/@#[[1]]}]& @ data

getSleepRawdata[id_,args_]:=Block[{invalidParameters, rawdata, start, end, ndays, n, day},

	invalidParameters = Select[Keys[args],!MemberQ[{"StartDate", "EndDate"},#]&];

	If[ Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Fitbit"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	start = Lookup[args, "StartDate", Message[ServiceExecute::nparam,"StartDate"]; Throw[$Failed]];
	If[!DateObjectQ[Quiet[DateObject[start]]], Message[ServiceExecute::nval,"StartDate","Fitbit"]; Throw[$Failed]];
	end = Lookup[args, "EndDate", DateString["ISODate"]];
	If[!DateObjectQ[Quiet[DateObject[end]]], Message[ServiceExecute::nval,"EndDate","Fitbit"]; Throw[$Failed]];

	ndays = First[DateDifference[start, end]];

	Which[	!TrueQ[ndays > 0],
				Message[ServiceExecute::fbdrng,start,end];
				Throw[$Failed],
			TrueQ[ndays > Fitbit`Private`$RateLimitRemaining],
				Message[ServiceExecute::fbrate2, Fitbit`Private`$RateLimitRemaining];
				Throw[$Failed]
	];

	rawdata = Table[OAuthClient`rawoauthdata[id, "RawSleep", "Date" -> day], {day, DayRange[start, end]}]
]

makeSleepBox[data_, minEfficiency_, maxEfficiency_, t0_, timelineQ_:False] :=
	Block[{day, duration = data[[2]], efficiency, xmin, xmax, tQ = Boole[timelineQ]},
		xmin = Dot[{1, 1/60, 1/1440}, Take[First[data], -3]];
		xmax = xmin + (duration/60);
		efficiency = Rescale[Last[data], {maxEfficiency, minEfficiency}];
		day = (AbsoluteTime[Take[First[data], 3]] - t0)/86400;
		{
			Opacity[0.95-tQ/2],
			ColorData["TemperatureMap"][efficiency],
			Tooltip[
				If[xmax>24,
					If[xmax>48, Throw[$Failed]];
					{Rectangle[{0, day + 1.05 - tQ}, {xmax - 24, day + 1.95 - tQ}],
					Rectangle[{xmin, day + .05}, {24, day + 0.95}]}
					,
					Rectangle[{xmin, day + .05}, {xmax, day + 0.95}]
				],
				ToString@NumberForm[N@UnitConvert[Quantity[duration, "Minutes"], "Hour"], {6, 2}]
			]
		}
	]

formatHeartRateZone[data_]:=Block[{var = data},
	var = KeyMap[Capitalize]@var;
	var = KeyTake[var, {"Name", "Min", "Max", "CaloriesOut", "Minutes"}];
	var = MapAt[Quantity[#, "LargeCalories"]&, {{Key["CaloriesOut"]}, {Key["Max"]}, {Key["Min"]}}]@var;
	MapAt[Quantity[#, "Minutes"]&, {Key["Minutes"]}] @ var
]

formatBadge[data_Association]:= Block[{var = data, gradient, images, descriptions, sharing},
	gradient = Map[RGBColor["#"<>#]&]@Reverse@(KeyMap[StringReplace["BadgeGradient"->""]][KeySelect[var, StringMatchQ["BadgeGradient*"]]]);
	images = KeySortBy[ToExpression[StringDrop[#,-2]]&]@KeyMap[StringReplace["Image"->""]][KeySelect[var, StringMatchQ["Image*"]]];
	descriptions = Sort[KeySelect[var, StringMatchQ["*Description*"]]];
	sharing = KeyMap[StringReplace[{("Share" | (DigitCharacter .. ~~ "px")) -> ""}]][KeySelect[var, StringMatchQ["Share*"]]];
	var = Join[ KeyTake[var, {"Name", "EncodedID"}], <|"DateTime" -> DateObject[var["DateTime"]]|>,
				KeyTake[var, {"Category", "BadgeType"}],
				<|"Descriptions" -> descriptions, "Images"-> images, "Share" -> sharing, "BadgeGradient"-> gradient|>,
				KeyTake[var, {"EarnedMessage", "TimesAchieved", "ShortName", "Unit", "Value", "Cheers"}]
			]
]

formatDate[per:("1d"|"7d"|"30d"|"1w"|"1m"|"3m"|"6m"|"1y"|"max"|"today")]:=per
formatDate[date_]:=DateString[date, "ISODate"]
formatDate[dates__]:=StringJoin[Riffle[formatDate[#]&/@{dates},"/"]]

formatDateTime[date_]:=StringSplit[DateString[date, "ISODateTime"], "T"]

readDate[date_,form_:DateObject]:=form[DateList[{date,{"Year", "-", "Month", "-", "Day"}}]]/;StringFreeQ[date,"T"]
readDate[date_,form_:DateObject]:=form[DateList[{date,{"Year", "-", "Month", "-", "Day", "T", "Hour", ":", "Minute", ":","Second", ".", "Millisecond"}}]]

End[]

EndPackage[]
