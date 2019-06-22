(* Frame Interpretation *)

(* Any interpretation will be done when the tag values are retrieved from Library Functions rather than right before the tag is presented to the user. 
 	This way, the values can be stored in their interpreted form rather than reinterpreted every time. 
 	Interpretation functions should expect as input the raw tag value, and should have a clear inverse that can return the original data unmodified. *)

$numberStringPatt = (WhitespaceCharacter... ~~ (NumberString|(DigitCharacter..~~"/"~~DigitCharacter..)) ~~ WhitespaceCharacter...);
$unitsPatt = (LetterCharacter ~~ (LetterCharacter|WhitespaceCharacter)...);
$unitDurationPatt = ("*seconds"|"sec"|"s"|"ms"|"minutes"|"min"|"hours"|"hrs"|"hr"|"ns"|"\[Micro]s");
$unitBeatsPerMinutePatt = RegularExpression["(bpm)|(beats( per | *\/ *)min(ute){0,1})"];

(* Interpretation/UnInterpretation for Frames which are probably common but nonetheless do not appear 
in the list of formally-supported frames and thus would not otherwise have a default interpretation function. *)
unknownStringToInterpretedValue[tagType_, tagID_, s_] := Quiet[Module[{uTagID},
	uTagID = ToUpperCase[tagID];
	Check[
		Which[
			StringMatchQ[uTagID, "*DATE*"], numberOrStringToDateObject[s],
			StringMatchQ[uTagID, "*YEAR*"], numberOrStringToDateObject[s, "DateGranularity" -> "Year"],
			StringMatchQ[uTagID, "*GAIN*"]||StringMatchQ[uTagID, "*VOLUME*"], numberOrStringToQuantity[s, "dB"],
			possibleTrackNumberStringQ[uTagID], interpretTrackNumber[s],
			True, s
		]
	, s]
]]

unknownInterpretedValueToString[tagType_, tagID_, s_] := Quiet[Module[{uTagID, res},
	uTagID = ToUpperCase[tagID];
	Check[
		Which[
			possibleTrackNumberStringQ[uTagID], res = uninterpretTrackNumber[s];,
			True, res = $Failed;
		];
		If[!StringQ[res], interpretedValueToString[s, "DateStringElements" -> "FromDateObject"], res]
	, $Failed]
]]

possibleTrackNumberStringQ[uTagID_] := (StringMatchQ[uTagID, "TRACK"|"DISK"|"DISC"] || StringMatchQ[uTagID, "TRACKN*"|"DISKN*"|"DISCN*"])

(* Test whether an expression could be an interpreted value *)
validateUnknownStringOrInterpretedValue[tagType_, tagID_, s_] := 
(
	zOrZsQ[s,
		(stringOrLinkQ[#]
		|| NumberQ[#]
		|| BooleanQ[#]
		|| QuantityQ[#]
		|| DateObjectQ[#]
		|| TimeObjectQ[#]
		|| MatchQ[#, {_Integer, _Integer}]
		|| (tagType === "APE" && ByteArrayQ[#]))&]
)

(* The main UnInterpretation function for converting various Wolfram-Language expressions into Strings, Integers, et c. *)
Options[interpretedValueToString] = SortBy[{
	"DateParts" -> All, 
	"DateStringElements" -> None,
	"Prefix" -> None, 
	"Suffix" -> None, 
	"StringValidationFunction" -> (True&),
	"Separator" -> "", 
	"NumberFormatFunction" -> Identity, (* ex: (IntegerString[#,10,2]&) *)
	"KeepQuantityUnits" -> False,
	"KeepFractionalForm" -> False,
	"Boolean" -> False, (* {"True","False"}|{"1","0"}|... *)
	"ConversionUnits" -> None
}, ToString];

Options[iinterpretedValueToString] = Options[interpretedValueToString];

interpretedValueToString[s_, opts:OptionsPattern[]] := Quiet[Module[{res},
	Check[
		If[!StringQ[(res = iinterpretedValueToString[s,opts])], 
			$Failed
			, 
			res = (OptionValue@"Prefix" /. None -> "") <> res <> (OptionValue@"Suffix" /. None -> "");
			If[!OptionValue["StringValidationFunction"][res], $Failed, res]
		]
	, $Failed]
]]
interpretedValueToString[___, opts:OptionsPattern[]] := $Failed

iinterpretedValueToString[s:(True|False), opts:OptionsPattern[]] := (
	If[VectorQ[OptionValue@"Boolean", StringQ] && (2 == Length[OptionValue@"Boolean"]), 
		If[s, First[OptionValue@"Boolean"], Last[OptionValue@"Boolean"]]
		,
		If[s,"1","0"]
	]
)
iinterpretedValueToString[s_?StringQ, opts:OptionsPattern[]] := s
iinterpretedValueToString[s_?QuantityQ, opts:OptionsPattern[]] := (
	If[TrueQ[OptionValue@"KeepQuantityUnits"], 
		If[SameQ[Identity, OptionValue@"NumberFormatFunction"], 
			QuantityUnits`ToQuantityShortString[#], 
			iinterpretedValueToString[QuantityMagnitude[#],opts]<>QuantityUnit[#]
		], 
		iinterpretedValueToString[QuantityMagnitude[#],opts]
	]& @ If[SameQ[None, OptionValue@"ConversionUnits"], s, UnitConvert[s, OptionValue@"ConversionUnits"]]
)
iinterpretedValueToString[s_?IntegerQ, opts:OptionsPattern[]] := (
	If[VectorQ[OptionValue@"Boolean", StringQ] && (2 == Length[OptionValue@"Boolean"]), 
		If[s > 0, First[OptionValue@"Boolean"], Last[OptionValue@"Boolean"]]
		,
		ToString[OptionValue["NumberFormatFunction"][s]]
	]
)
iinterpretedValueToString[s_?NumberQ, opts:OptionsPattern[]] /; !MatchQ[s,_Rational] := ToString[OptionValue["NumberFormatFunction"][DecimalForm[N@s]]]
iinterpretedValueToString[s_Rational, opts:OptionsPattern[]] /; NumberQ[s] := (
	If[TrueQ[OptionValue@"KeepFractionalForm"], 
		iinterpretedValueToString[Numerator[s],opts] <> "/" <> iinterpretedValueToString[Denominator[s],opts], 
		iinterpretedValueToString[N@s, opts]
	]
)
iinterpretedValueToString[File[s_?StringQ], opts:OptionsPattern[]] := s
iinterpretedValueToString[URL[s_?StringQ], opts:OptionsPattern[]] := s
iinterpretedValueToString[s_?((DateObjectQ[#] || TimeObjectQ[#])&), opts:OptionsPattern[]] := (
	If[SameQ[All, OptionValue@"DateParts"], 
		Which[
			SameQ[None, OptionValue@"DateStringElements"], 
				DateString[s],
			SameQ["FromDateObject", OptionValue@"DateStringElements"], 
				Replace[s, {DateObject[_, "Day", ___] -> DateString[s, {"Day", " ", "MonthNameShort", " ", "Year"}], DateObject[_, "Year", ___] -> DateString[s, "Year"], _ -> DateString[s]}],
			True, 
				DateString[s, OptionValue@"DateStringElements"]
		]
		,
		iinterpretedValueToString[Flatten@{DateValue[s, OptionValue@"DateParts", Integer]}, opts]
	]
)
iinterpretedValueToString[s_?(VectorQ[#,StringQ]&), opts:OptionsPattern[]] := StringJoin[Riffle[s, OptionValue@"Separator"]]
iinterpretedValueToString[s_?(VectorQ[#, (NumberQ[#] || StringQ[#] || dateSpecQ[#])&]&), opts:OptionsPattern[]] := iinterpretedValueToString[iinterpretedValueToString[#,opts]& /@ s, opts]
iinterpretedValueToString[___, opts:OptionsPattern[]] := $Failed

(* UnInterpret a Quantity expression (or number) into a number or string *)
Options[quantityToNumeric] = SortBy[{"ConversionUnits" -> None, "ToInteger" -> False}, ToString];
quantityToNumeric[s_?QuantityQ, opts:OptionsPattern[]] := Quiet[Module[{m},
	Check[
		m = QuantityMagnitude[If[SameQ[None, OptionValue@"ConversionUnits"], s, UnitConvert[s, OptionValue@"ConversionUnits"]]];
		If[TrueQ[OptionValue@"ToInteger"], IntegerPart[m], DecimalForm[N[m]]]
	, $Failed]
]]
quantityToNumeric[s_?NumberQ, opts:OptionsPattern[]] := Quiet[
	Check[
		If[TrueQ[OptionValue@"ToInteger"], IntegerPart[s], DecimalForm[N[s]]]
	, $Failed]
]

(* Interpret a string as a File or URL *)
Options[stringToLink] = {"Wrapper" -> Automatic};
stringToLink[s_?StringQ, opts:OptionsPattern[]] := Quiet[
	Check[
		If[SameQ[OptionValue@"Wrapper", Automatic], 
			If[StringStartsQ[s, "file:"], 
				File[StringDrop[s, 5]], 
				If[StringContainsQ[s, "://"], URL[s], s]
			]
			, 
			OptionValue["Wrapper"][s]
		]
	, s]
]
stringToLink[s_, opts:OptionsPattern[]] := s

(* Test whether an expression is a possible File or URL *)
Options[stringOrLinkQ] = {"Wrapper" -> Automatic};
stringOrLinkQ[s_?StringQ, opts:OptionsPattern[]] := True
stringOrLinkQ[s_URL, opts:OptionsPattern[]] := (SameQ[OptionValue@"Wrapper", Automatic] || SameQ[OptionValue@"Wrapper", URL])
stringOrLinkQ[s_File, opts:OptionsPattern[]] := (SameQ[OptionValue@"Wrapper", Automatic] || SameQ[OptionValue@"Wrapper", File])
stringOrLinkQ[___] := False

(* Interpret a string as a number (or boolean value) *)
Options[stringToNumeric] = {"Boolean" -> False};
stringToNumeric[s_?StringQ, opts:OptionsPattern[]] /; StringMatchQ[s, $numberStringPatt] := (If[TrueQ[OptionValue@"Boolean"], Replace[#, x_?IntegerQ :> (x>0)], #]& @ ToExpression[s])
stringToNumeric[s_, opts:OptionsPattern[]] := s

byteCountToQuantity[nb_?NumberQ] := With[{b = N@nb},
Piecewise[{
	{Quantity[b * 1., "Bytes"], b < 500.},
	{Quantity[b/1000., "Kilobytes"], 500. <= b < 500000.},
	{Quantity[b/1000^2., "Megabytes"],500000. <= b < 5.*10^8},
	{Quantity[b/1000^3., "Gigabytes"], 5.*10^8 <= b < 5.*10^11},
	{Quantity[b/1000^4., "Terabytes"], 5.*10^11 <= b}
}]]
secondsToQuantity[ns_?NumberQ] := With[{s = N@ns},
Piecewise[{
	{Quantity[0,"Seconds"], s == 0.},
	{Quantity[s*10^6, "Microseconds"], s < 5.*10^-4},
	{Quantity[s*10^3, "Milliseconds"], 5*10^-4 <= s < 0.5},
	{DateObject[s] - DateObject[0], 0.5 <= s}
}]]

(* Interpret a string or number as a Quantity expression *)
(* stringToQuantityDB[s_] /; StringMatchQ[s, $numberStringPatt ~~ "db"|"decibels", IgnoreCase -> True] := Quantity[ToExpression@First@StringSplit[s, "db", IgnoreCase -> True], "dB"] *)
Options[numberOrStringToQuantity] = {"ConversionUnits" -> None};
numberOrStringToQuantity[s_?StringQ, opts:OptionsPattern[]] /; StringMatchQ[s, $numberStringPatt ~~ $unitsPatt] := Quiet[
Module[{u,q,m},
	Check[
		Replace[
			q = Quantity[ToExpression[StringTake[s,#]], StringTrim@StringDrop[s,#]]& @ Last@First@StringPosition[s, NumberString, Overlaps -> False];
			Switch[(u = OptionValue@"ConversionUnits"),
				"AutomaticDuration", secondsToQuantity[QuantityMagnitude[#, "Seconds"]],
				"AutomaticFileSize", byteCountToQuantity[QuantityMagnitude[#, "Bytes"]],
				_?StringQ, UnitConvert[q, u],
				_, q
			]
		, Except[_?QuantityQ] -> s]
	, s]
]]
numberOrStringToQuantity[s_?StringQ, units_, opts:OptionsPattern[]] /; StringMatchQ[s,$numberStringPatt] := numberOrStringToQuantity[ToExpression[s], units, opts]
numberOrStringToQuantity[s_?NumberQ, units_, opts:OptionsPattern[]] := Quiet[Module[{m},
	Check[
		Replace[
			Switch[OptionValue@"ConversionUnits",
				"AutomaticDuration", secondsToQuantity[QuantityMagnitude[#, "Seconds"]],
				"AutomaticFileSize", byteCountToQuantity[QuantityMagnitude[#, "Bytes"]],
				_?StringQ, UnitConvert[#, OptionValue@"ConversionUnits"],
				_, #
			]& @ Quantity[s, units]
		, Except[_?QuantityQ] -> s]
	, s]
]]
numberOrStringToQuantity[s_?StringQ, units_, opts:OptionsPattern[]] /; StringMatchQ[s, $numberStringPatt ~~ $unitsPatt] := Quiet[
Module[{u,q,m},
	Check[
		Replace[
			{q,u} = ({ToExpression[StringTake[s,#]], StringTrim@StringDrop[s,#]}& @ Last@First@StringPosition[s, NumberString, Overlaps -> False]);
			Switch[OptionValue@"ConversionUnits",
				"AutomaticDuration", secondsToQuantity[QuantityMagnitude[#, "Seconds"]],
				"AutomaticFileSize", byteCountToQuantity[QuantityMagnitude[#, "Bytes"]],
				_?StringQ, UnitConvert[#, OptionValue@"ConversionUnits"],
				_, UnitConvert[#, units]
			]& @ Quantity[q, u]
		, Except[_?QuantityQ] -> s]
	, s]
]]
numberOrStringToQuantity[s_,___] := s

(* Interpret a string or number as a Date expression *)
Options[numberOrStringToDateObject] = SortBy[{"DatePattern" -> "*", "DateElements" -> All, "DateGranularity" -> None, "DateObjectOpts" -> {}}, ToString];
numberOrStringToDateObject[s_?StringQ, opts:OptionsPattern[]] := Quiet[
	Check[
		Replace[
			If[StringQ[OptionValue@"DateGranularity"], DateObject[#, OptionValue@"DateGranularity"], #]& @
				If[StringMatchQ[s, OptionValue@"DatePattern"] /. Except[True|False] -> False,
					If[SameQ[All, OptionValue@"DateElements"],
						DateObject[s, OptionValue@"DateObjectOpts"],
						DateObject[{s, OptionValue@"DateElements"}, OptionValue@"DateObjectOpts"]
					],
					DateObject[s, OptionValue@"DateObjectOpts"]
				]
		, Except[_?DateObjectQ] -> s]
	, s, {DateObject::str}]
]
numberOrStringToDateObject[s_?NumberQ, opts:OptionsPattern[]] := Quiet[
	Check[
		Replace[
			If[!SameQ[None, OptionValue@"DateGranularity"], DateObject[#, OptionValue@"DateGranularity"], #]& @ 
				If[s < 10^4, DateObject[{IntegerPart[s]}, OptionValue@"DateObjectOpts"] (* Year *), DateObject[s, OptionValue@"DateObjectOpts"] (* AbsoluteTime *)]
		, Except[_?DateObjectQ] -> s]
	, s]
]
numberOrStringToDateObject[s_, opts:OptionsPattern[]] := s

(* Interpret a string or number as a Time expression *)
Options[stringToTimeObject] = Options[numberOrStringToDateObject];
stringToTimeObject[s_, opts:OptionsPattern[]] := Quiet[
	Check[
		Replace[
			Replace[
				numberOrStringToDateObject[s, opts]
			, dt_?DateObjectQ :> TimeObject[dt]]
		, Except[_?TimeObjectQ] -> s]
	, s]
]

(* Interpret a string or number as a TimeStamp/Duration expression *)
stringToTimeStamp[s_?StringQ] /; StringMatchQ[s, $numberStringPatt] := ToExpression[s] (* Could be either Samples or Seconds *)
stringToTimeStamp[s_?StringQ] /; StringMatchQ[s, WhitespaceCharacter...~~DigitCharacter..~~":"~~NumberString~~WhitespaceCharacter...] := Quiet[
	Check[
		Quantity[ToExpression[First[#]] + ToExpression[Last[#]]/60., "Minutes"]& @ StringSplit[s, ":"]
	, s]
]
stringToTimeStamp[s_?StringQ] /; StringMatchQ[s, WhitespaceCharacter...~~DigitCharacter..~~":"~~DigitCharacter..~~":"~~NumberString~~WhitespaceCharacter...] := Quiet[
	Check[
		Quantity[ToExpression[First[#]] + ToExpression[#[[2]]]/60. + ToExpression[Last[#]]/3600., "Hours"]& @ StringSplit[s, ":"]
	, s]
]
stringToTimeStamp[s_] := s

(* UnInterpret a number into a formatted string *)
Options[numberToPaddedString] = SortBy[{"LeadingDigits" -> 2, "TrailingDigits" -> All}, ToString];
numberToPaddedString[s_?IntegerQ, opts:OptionsPattern[]] := Quiet[Check[IntegerString[s,10,OptionValue@"LeadingDigits"], $Failed]]
numberToPaddedString[s_?NumberQ, opts:OptionsPattern[]] := Quiet[
	Check[
		IntegerString[IntegerPart[s], 10, OptionValue@"LeadingDigits"] <> 
			Replace[FractionalPart[s], {0->"", f_ :> "." <> StringTake[Last[StringSplit[ToString@DecimalForm[N[f]],"."]], OptionValue@"TrailingDigits"]}]
	, $Failed]
]

(* UnInterpret a TimeStamp/Duration expression into a string *)
timeStampToString[s_?QuantityQ] := Module[{m},
	Check[
		Switch[QuantityUnit[s],
			"Hours",
				m = QuantityMagnitude[s];
				ToString[IntegerPart[m]]<>":"<>numberToPaddedString[IntegerPart[FractionalPart[m]*60]]<>":"<>numberToPaddedString[FractionalPart[FractionalPart[m]*60]*60]
			,
			"Minutes",
				m = QuantityMagnitude[s];
				ToString[IntegerPart[m]]<>":"<>numberToPaddedString[FractionalPart[m]*60]
			,
			x_/;StringMatchQ[x, RegularExpression@"[A-Za-z]*(S|s)econds"],
				ToString[DecimalForm[N@QuantityMagnitude[UnitConvert[s,"Seconds"]]]]
			,
			_, $Failed
		]
	, $Failed]
]
timeStampToString[s_?NumberQ] := Quiet[Check[ToString[DecimalForm[N@s]], $Failed]]
timeStampToString[s_?StringQ] := s
timeStampToString[s_] := $Failed

(* Test whether an expression is a possible TimeStamp/Duration value *)
stringOrTimeStampQ[s_?NumberQ] /; Positive[s] := True
stringOrTimeStampQ[s_?StringQ] /; StringMatchQ[s, $numberStringPatt|((DatePattern[{"Minute","Second"}, ":"]|DatePattern[{"Hour","Minute","Second"},":"]) ~~ ("" | ("." ~~ DigitCharacter ..)))] := True
stringOrTimeStampQ[s_] := stringOrQuantityQ[s, "UnitStringPattern"->$unitDurationPatt, "AllowNumericValuesTest"->(Positive[#]&)]

(* Test whether a string is a possible Quantity value *)
Options[stringOrQuantityQ] = SortBy[{"UnitStringPattern" -> "*", "AllowNumericValuesTest" -> (False&), "UnitPattern" -> _}, ToString];
stringOrQuantityQ[s_?StringQ, opts:OptionsPattern[]] /; StringMatchQ[s, $numberStringPatt] := True
stringOrQuantityQ[s_?StringQ, opts:OptionsPattern[]] /; StringMatchQ[s, $numberStringPatt ~~ $unitsPatt] := Quiet[
	Check[
		StringMatchQ[StringTrim@StringDrop[s, Last@First@StringPosition[s, NumberString, Overlaps -> False]], OptionValue@"UnitStringPattern", IgnoreCase -> True]
	, False]
]
stringOrQuantityQ[s_, opts:OptionsPattern[]] := numberOrQuantityQ[s, FilterRules[{opts},Options[numberOrQuantityQ]], "NumberTest"->OptionValue@"AllowNumericValuesTest"]

(* Test whether a number is a possible Quantity value *)
Options[numberOrQuantityQ] = SortBy[{"UnitStringPattern" -> "*", "NumberTest" -> (True&), "UnitPattern" -> _}, ToString];
numberOrQuantityQ[s_?QuantityQ, opts:OptionsPattern[]] := Quiet[
	Check[
		If[StringQ[#], 
			StringMatchQ[#, OptionValue@"UnitStringPattern", IgnoreCase->True],
			MatchQ[#, OptionValue@"UnitPattern"]
		]& @ QuantityUnit[s]
	, False]
]
numberOrQuantityQ[s_?NumberQ, opts:OptionsPattern[]] := Quiet[
	Check[
		Replace[OptionValue["NumberTest"][s], Except[True|False] -> False]
	, False]
]
numberOrQuantityQ[___] := False

(* Test whether an expression is a possible valid numeric string or number *)
Options[stringOrNumberQ] = SortBy[{"NumberTest" -> None, "Fractional" -> False}, ToString];
stringOrNumberQ[s_?StringQ, opts:OptionsPattern[]] /; StringMatchQ[s, $numberStringPatt] := Quiet[
	Check[
		stringOrNumberQ[Replace[ToExpression[s], _String -> False], opts]
	, False]
]
stringOrNumberQ[s_?NumberQ, opts:OptionsPattern[]] := Quiet[
	Check[
		Replace[
			And[
				If[TrueQ[OptionValue@"Fractional"], (Positive[s] && (IntegerQ[s] || MatchQ[s, _Rational])), True]
				,
				If[SameQ[None, OptionValue["NumberTest"]], True, OptionValue["NumberTest"][s]]
			]
		, Except[True|False] -> False]
	, False]
]
stringOrNumberQ[___] := False

(* Test whether an expression is a possible Boolean value or string representation thereof *)
Options[stringOrBooleanQ] = {"AllowNumericValues" -> True}
stringOrBooleanQ[s:(True|False)] := True
stringOrBooleanQ[s_?StringQ] /; StringMatchQ[s, RegularExpression["((T|t)rue)|((F|f)alse)"]] := True
stringOrBooleanQ[s_] := stringOrNumberQ[s, "NumberTest" -> ((TrueQ[OptionValue@"AllowNumericValues"] && (SameQ[#,0] || SameQ[#,1]))&)]

numberToBoolean[s_?IntegerQ] := (s > 0)
numberToBoolean[s_] := s

booleanToNumber[s_?IntegerQ] := If[s > 0, 1, 0]
booleanToNumber[s_] := Quiet[Check[If[TrueQ[s],1,0], $Failed]]

numberOrBooleanQ[s:(True|False|0|1)] := True
numberOrBooleanQ[___] := False

(* Interpret a string as a Synchronised lyrics association with TimeStamps *)
(* Perhaps we should save the original max granularity (hours|minutes|seconds), as well as the original seconds precision (for RHS padding) *)
interpretLyrics[s_?StringQ] := Quiet[Module[{timeStamps, parts0, parts1, values, timeStrings, qtimes, desc}, 
	Check[
		timeStamps = StringPosition[s, WhitespaceCharacter...~~(DigitCharacter..~~":")..~~NumberString~~WhitespaceCharacter..., Overlaps -> False];
		desc = StringTake[s, First[First[timeStamps]] - 1];
		parts0 = Partition[Join[timeStamps, {{StringLength[s] + 1}}], 2, 1];
		parts1 = {Last[#[[1]]] + 1, First[#[[2]]] - 1}& /@ parts0;
		values = StringTake[s, #]& /@ parts1;
		timeStrings = StringTrim[StringTake[s, #]]& /@ timeStamps;
		qtimes = stringToTimeStamp[#]& /@ timeStrings;
		If[VectorQ[qtimes, QuantityQ], qtimes = UnitConvert[#, "Seconds"]& /@ qtimes];
		(* TimeSeries[N[QuantityMagnitude[#]]& /@ qtimes, {values}, MetaInformation -> {"Description"->desc}] *)
		Join[<|"Description"->desc|>, AssociationThread[qtimes,values]]
	, s]
]]
interpretLyrics[s_] := s

(* UnInterpret a Synchronised lyrics association with TimeStamps into a string *)
uninterpretLyrics[s_?AssociationQ] := Quiet[Module[{l0, l1}, 
	Check[
		StringJoin[
			l0 = SortBy[Normal@Delete[s,"Description"], First];
			l1 = Join[{Replace[s["Description"], _Missing -> Nothing]}, StringJoin[timeStampToString[UnitConvert[#[[1]], "Hours"]], " ", #[[2]]]& /@ l0];
			StringJoin[Riffle[l1, "\n"]]
		]
	, $Failed]
]]
uninterpretLyrics[s_?StringQ] := s
uninterpretLyrics[s_?(VectorQ[#,StringQ]&)] := s
uninterpretLyrics[s_] := $Failed

(* Interpret a string as a track number *)
interpretTrackNumber[s_?StringQ] /; StringMatchQ[s, $numberStringPatt] := Quiet[
	Check[ 
		Which[
			MatchQ[#, {_?Internal`NonNegativeIntegerQ}], First@#,
			MatchQ[#, {_?Internal`NonNegativeIntegerQ, _?Internal`NonNegativeIntegerQ}], #,
			True, s
		]& @ ToExpression[StringSplit[s, "/"]]
	, s]
]
interpretTrackNumber[s_] := s

(* UnInterpret a track number into a string or number. A TrackNumber can be in a form such as {N, Out-Of-M}. *)
Options[uninterpretTrackNumber] = {"ToString" -> True}; (* use "ToString" -> False for conversion to Integer *)
uninterpretTrackNumber[s_?Internal`NonNegativeIntegerQ, opts:OptionsPattern[]] := If[TrueQ[OptionValue@"ToString"], ToString[s], s]
uninterpretTrackNumber[s:{num_?Internal`NonNegativeIntegerQ, den_?Internal`NonNegativeIntegerQ}, opts:OptionsPattern[]] := Quiet[
	Check[
		If[TrueQ[OptionValue@"ToString"], ToString[num]<>"/"<>ToString[den], num]
	, $Failed]
]
uninterpretTrackNumber[s_?StringQ] /; StringMatchQ[s, $numberStringPatt] := Quiet[
	Check[
		If[TrueQ[OptionValue@"ToString"],
			s
			, 
			Which[
				MatchQ[#, {_?Internal`NonNegativeIntegerQ..}], First@#,
				True, $Failed
			]& @ ToExpression[StringSplit[s, "/"]]
		]
	, $Failed]
]
uninterpretTrackNumber[___] := $Failed

(* Test whether an expression is a possible track number specification *)
stringOrTrackNumberQ[_?StringQ] := True
stringOrTrackNumberQ[_?Internal`NonNegativeIntegerQ | {_?Internal`NonNegativeIntegerQ, _?Internal`NonNegativeIntegerQ}] := True
stringOrTrackNumberQ[___] := False

(* Test whether a string is a possible Date or Time specification *)
dateTimeStringQ[s_?StringQ] := Quiet[Check[DateObjectQ[DateObject[s]], False, {DateObject::str}]]
dateTimeStringQ[s_] := False

(* Test whether an expression is a possible Date value *)
Options[dateSpecQ] = SortBy[{"AllowTimeObject" -> False, "StringTest" -> (True&), "NumberTest" -> (True&)}, ToString];
dateSpecQ[s_, opts:OptionsPattern[]] := (
	(dateTimeStringQ[s] && OptionValue["StringTest"][s]) 
	|| (NumberQ[s] && Positive[s] && OptionValue["NumberTest"][s]) 
	|| DateObjectQ[s] 
	|| If[TrueQ[OptionValue@"AllowTimeObject"], TimeObjectQ[s], False])

(* Test whether an expression is some type or a list of some type *)
Options[zOrZsQ] = {"Level" -> 1};
zOrZsQ[s_, z_, opts:OptionsPattern[]] := VectorQ[Flatten[{s}, OptionValue@"Level"], z]

(* Utility for flattening a list and then mapping over it *)
Options[flattenMap] = {"Level" -> 1};
flattenMap[f_, s_, opts:OptionsPattern[]] := (f /@ Flatten[{s}, OptionValue@"Level"])

transformGenre[g_?IntegerQ] := ($genreTypes[g] /. _Missing -> $genreTypes[255])
transformGenre[g_?StringQ] := ($genreTypes[g] /. _Missing -> 255)

dateToIntegerYear[s_?DateObjectQ] := Quiet[
	Check[
		DateValue[s, "Year", Integer]
	, $Failed]
]
dateToIntegerYear[s_?IntegerQ] := s
dateToIntegerYear[s_?StringQ] := Quiet[
	Check[
		Replace[
			DateValue[numberOrStringToDateObject[s,"DateGranularity"->"Year"], "Year", Integer]
		, Except[_?IntegerQ] -> $Failed]
	, $Failed]
]
dateToIntegerYear[___] := $Failed

Options[transformID3v2Frames] = Options[transformID3v2Elements] = {"ToRawForm" -> False};

transformID3v2Frames[elements_Association, context_, opts:OptionsPattern[]] := Module[{elem},
	Association[Map[(
			elem = If[StringMatchQ[#[[1]], RegularExpression["(W|T)(?!XXX)..."]], $id3v2ExtendedFramesAssociation[#[[1]]], $id3v2FramesAssociation[#[[1]]]];
			If[MissingQ[elem], 
				#, 
				#[[1]] -> If[TrueQ[OptionValue@"ToRawForm"],
							If[SameQ[Identity, elem["InterpretationFunction"]], #[[2]], elem["InterpretationInverse"][#[[2]], context]]
							,
							If[SameQ[Identity, elem["InterpretationFunction"]], #[[2]], elem["InterpretationFunction"][#[[2]], context]]
						]
			]
		)&, Normal[elements]]
	]
]

transformID3v2Elements[elements_Association, context_, opts:OptionsPattern[]] := Module[{elem},
	Association[Map[(
		elem = $id3v2ElementsAssociation[#[[1]]];
		If[MissingQ[elem], 
			#, 
			#[[1]] -> If[TrueQ[OptionValue@"ToRawForm"], 
						If[SameQ[Identity, elem["InterpretationFunction"]], #[[2]], elem["InterpretationInverse"][#[[2]], context]]
						, 
						If[SameQ[Identity, elem["InterpretationFunction"]], #[[2]], elem["InterpretationFunction"][#[[2]], context]]
					]
		])&, Normal[elements]]
	]
]

byteArrayToImage[s_?ByteArrayQ, context:{tagType:("ID3v2"|"M4A"), tagID_, tagPair_, tagNo_, ___}] := Quiet[Module[{frame, format = "JPEG"},
	Check[
		frame = $rawTagContainer[tagType, tagPair[[1]], tagPair[[2]], tagNo];
		If[AssociationQ[frame] && !MissingQ[frame["MimeType"]], format = frame["MimeType"];];
		Replace[ImportString[FromCharacterCode@Normal[s], format], Except[_?ImageQ] -> s]
	, s]
]]
byteArrayToImage[s_?AssociationQ, context_] := Quiet[Module[{frame, format},
	Check[
		frame = s["Picture"];
		If[ImageQ[frame], frame,
			format = s["MimeType"];
			If[MissingQ[frame] || MissingQ[format], 
				s
				,
				Replace[ImportString[FromCharacterCode@Normal[frame], format], Except[_?ImageQ] -> s]
			]
		]
	, s]
]]
byteArrayToImage[s_,___] := s

imageToByteArray[s_?ImageQ, context:{tagType:("ID3v2"|"M4A"), tagID_, tagPair_, tagNo_, ___}] := Quiet[Module[{data, frame, format = "JPEG"},
	Check[
		frame = $rawTagContainer[tagType, tagPair[[1]], tagPair[[2]], tagNo];
		If[AssociationQ[frame] && !MissingQ[frame["MimeType"]], format = frame["MimeType"];];
		data = ByteArray[ToCharacterCode@ExportString[s, format]];
		If[ByteArrayQ[data], data, $Failed]
	, $Failed]
]]
imageToByteArray[s_?AssociationQ, context_] := Quiet[Module[{data, frame, format},
	Check[
		frame = s["Picture"];
		If[ByteArrayQ[frame], frame,
			format = s["MimeType"];
			If[MissingQ[frame] || MissingQ[format], 
				$Failed
				,
				data = ByteArray[ToCharacterCode@ExportString[frame, format]];
				If[ByteArrayQ[data], data, $Failed]
			]
		]
	, $Failed]
]]
imageToByteArray[s_?ByteArrayQ, _] := s
imageToByteArray[___] := $Failed

stringToPricePaidQuantity[s_?StringQ, context:{tagType:"ID3v2", tagID_, tagPair_, tagNo_, rawForm___}] := Quiet[
	Check[
		If[SameQ[{rawForm}, {"ToRawForm"}], s, 
			Quantity[ToExpression@StringDrop[s, 3], StringTake[s, 3]]
		]
	, s]
]
stringToPricePaidQuantity[s_,___] := s

pricePaidQuantityToString[s_?QuantityQ, ___] := Quiet[Module[{unit, val},
	Check[
		unit = QuantityUnit[s];
		val = ToString@NumberForm[QuantityMagnitude[s], 2];
		If[unit === "USDollars", 
			"USD",
			Block[{PacletManager`$AllowInternet = False},
				First[StringSplit[First@QuantityUnits`Private`getConversionRatio[QuantityUnits`Private`getCUF0@unit, QuantityUnits`Private`getCUF1@"USDollars"], "/"]]
			]
		] <> val
	, $Failed]
]]
pricePaidQuantityToString[s_?StringQ,___] /; StringMatchQ[s, RegularExpression@"^([A-Z]{3,3}|[0-9]{3,3})[0-9]*([.][0-9]*){0,1}$"] := s
pricePaidQuantityToString[___] := $Failed

numberListToTimestampFormatQuantity[s_List, context:{tagType:"ID3v2", tagID_, tagPair_, tagNo_, ___}] := Quiet[Module[{frame, format},
	frame = $rawTagContainer[tagType, tagPair[[1]], tagPair[[2]], tagNo];
	format = If[(AssociationQ[frame] && !MissingQ[frame["TimestampFormat"]]), frame["TimestampFormat"], "AbsoluteMilliseconds"];
	If[format === "AbsoluteMilliseconds", 
		If[IntegerQ[#], 
			Check[
				Quantity[#/1000., "Seconds"]
			, #]
		, #]& /@ s
	, s]
]]
numberListToTimestampFormatQuantity[s_,___] := s

(* Frame UnInterpretation *)

Options[uninterpretElementsForKey] = {"Context" -> {}};

uninterpretElementsForKey[tagType_, tagID_, val_, opts:OptionsPattern[]] := Module[{uTagID, elements},
	uTagID = If[SameQ[tagType, "M4A"], #, ToUpperCase[#]]& @ translateTagKey[tagType, tagID];
	Switch[tagType,
		"ID3v2", elements = ($id3v2FramesAssociation[uTagID] /. _Missing -> $id3v2ExtendedFramesAssociation[uTagID] /. _Missing -> $id3v2ElementsAssociation[tagID])
		,
		"ID3v1", elements = ($id3v1ElementsAssociation[uTagID])
		,
		"Xiph", elements = ($xiphElementsAssociation[uTagID])
		,
		"APE", elements = ($apeElementsAssociation[uTagID])
		,
		"M4A", elements = ($m4aElementsAssociation[uTagID])
	];
	If[MissingQ[elements],
		Switch[tagType,
			"APE",
				If[StringQ[val] || ByteArrayQ[val], val, unknownInterpretedValueToString[tagType, uTagID, val(*, "AlternateOutput"->{ByteArray}*)]]
			,
			_,
				If[StringQ[val], val, unknownInterpretedValueToString[tagType, uTagID, val]]
		]
		,
		If[SameQ[elements["InterpretationFunction"], Identity], 
			val,
			elements["InterpretationInverse"][elements["InterpretationFunction"][val, OptionValue@"Context"], Append[OptionValue@"Context","ToRawForm"]]
		]
	]
]
