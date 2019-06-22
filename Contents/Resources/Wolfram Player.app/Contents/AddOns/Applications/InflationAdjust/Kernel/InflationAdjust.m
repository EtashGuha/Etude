
Begin["System`InflationAdjust`Private`"]

Get[FileNameJoin[{DirectoryName[$InputFileName], "CPIData.m"}]]

$QueryTimeout = Automatic;
$tag = "InflationAdjustCatchThrowFlag";

(*********************************        Within the same currency; No currency conversions         *************************************)
Unprotect[InflationAdjust];
Clear[InflationAdjust];

Options[InflationAdjust] = {InflationMethod -> Automatic}

(* Invalid Number of Arguments *)
InflationAdjust[args___] := (ArgumentCountQ[InflationAdjust,Length[DeleteCases[{args},_Rule,Infinity]],1,2];Null/;False)


(* 1: No source date, or target and source dates are the same: Identity *) 
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_String]]?QuantityQ, opts:OptionsPattern[]] := Quantity[q, DatedUnit[SourceUnit]]
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_, SourceDate_?DateListQ]]?QuantityQ, SourceDate_?DateListQ, opts:OptionsPattern[]] := Quantity[q, DatedUnit[SourceUnit, trimDate[SourceDate]]]

(* 2: Source unit in String form *)
InflationAdjust[Quantity[q_, SourceUnit_String]?QuantityQ, rest___] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, currentDate[]]], rest]

(* 3: No Source date *)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_String]]?QuantityQ, rest___] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, currentDate[]]], rest]
(* 3.5: Source date is DateObject*)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_String,d_DateObject]]?QuantityQ, rest___] :=
	With[{du = ToDatedUnit[SourceUnit,d]}, InflationAdjust[Quantity[q,du],rest]]
(* 4: Source year, not full date *)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_String, SourceYear_Integer]]?QuantityQ, rest___] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, {SourceYear}]], rest]

(* 5: No target date or Autoamatic *)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_, SourceDate_?DateListQ]]?QuantityQ, opts:OptionsPattern[]] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, SourceDate]], currentDate[], opts]
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_, SourceDate_?DateListQ]]?QuantityQ, Automatic, opts:OptionsPattern[]] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, SourceDate]], currentDate[], opts]

(* 6: Target year, not full date *)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_String, SourceYear_]]?QuantityQ, TargetDate_Integer, opts:OptionsPattern[]] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, SourceYear]], {TargetDate}, opts]

(* 7: Main function with no currency conversion involved *)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_String, SourceDate_?DateListQ]]?QuantityQ, TargetDate_?DateListQ, opts:OptionsPattern[]] := Module[
	{startvalue, endvalue, startdate, enddate, SDate, TDate, UnitCode, cpi = OptionValue[InflationAdjust, InflationMethod]},
	
	(* Check validity of the SourceUnits first *)
	If[!QuantityUnits`Private`couldBeACurrencyQ[SourceUnit], Message[InflationAdjust::notcur, SourceUnit]; Return[Quantity[q, DatedUnit[SourceUnit, SourceDate]]]];
	UnitCode = QuantityUnits`Private`FindCurrencyUnitValue[SourceUnit];
	If[!MatchQ[UnitCode, {{_String}, _?(NumberQ[#]&)}], Message[InflationAdjust::curnc, SourceUnit]; Return[Missing["NotAvailable"]]];
	
	(* Make sure the SourceUnit is covered in cpi *)
	cpi = Switch[cpi, 
				Automatic, CPIData[UnitCode[[1, 1]]],
				_?TimeSeriesQ, cpi,
				_?TemporalData`TemporalDataQ, Normal[cpi],
				{__Rule}, With[{CPI = UnitCode[[1, 1]]/.cpi}, (*Print[CPI//Short]; *)If[TimeSeriesQ[CPI], CPI, Message[InflationAdjust::badcpi, cpi]; Return[Missing["NotAvailable"]]]],
				_, Message[InflationAdjust::badcpi, cpi]; Return[Missing["NotAvailable"]]
				] /. d_DateObject :> System`InflationAdjust`Private`getDateFromDateObject[d]; (*Print[UnitCode[[1, 1]], "   ",cpi//Short];*)
	(* 17 currencies that have no cpi coverage *)
	If[!ListQ[cpi], Message[InflationAdjust::curnc, SourceUnit]; Return[Missing["NotAvailable"]]];
	
	(* Make sure the dates are covered in the cpi timeseries *)
	startdate = cpi[[1, 1, 1]];
	enddate = Max[AbsoluteTime[cpi[[-1, 1, 1]]], AbsoluteTime[DateList[][[1]]]];
	{SDate, TDate} = DateToDateList /@{SourceDate, TargetDate};
	If[!IntervalMemberQ[Interval[{startdate, enddate}], SDate[[1]]], Message[InflationAdjust::datenc, SourceUnit, startdate, enddate]; Return[Missing["NotAvailable"]]]; 
	If[!IntervalMemberQ[Interval[{startdate, enddate}], TDate[[1]]], Message[InflationAdjust::datenc, SourceUnit, startdate, enddate]; Return[Missing["NotAvailable"]]];
	
	(* Construct interpolation function and *)
	cpi 		= DateListInterpolation[cpi, InterpolationOrder -> 1]; (*Print@cpi;*)
	{startvalue, endvalue} = Quiet[cpi[#, InterpolatingFunction::dmval], InterpolatingFunction::dmval]&/@{SDate, TDate};
	
	(* If the interpolation is successful and we have a value back, proceed and otherwise, just return the value *)
	Which[
		StringQ[startvalue], startvalue, 
		StringQ[endvalue], endvalue, 
		True, Quantity[q * (endvalue/startvalue), DatedUnit[SourceUnit, trimDate[TargetDate]]]
	]
]

(*********************************         various patterns for second argument with DateObject         *************************************)
InflationAdjust[arg_, dObj_DateObject, opts:OptionsPattern[]] := InflationAdjust[arg, getDateFromDateObject[dObj], opts]
InflationAdjust[arg_, DatedUnit[unit_, dObj_DateObject], opts:OptionsPattern[]] := InflationAdjust[arg, DatedUnit[unit, getDateFromDateObject[dObj]],opts]

getDateFromDateObject[HoldPattern[dObj:DateObject[date_List, __]]] := With[{r=DateList[dObj]}, If[ListQ[r], Take[r, Length[date]], date]]
getDateFromDateObject[HoldPattern[dObj:DateObject[date_List, ___?OptionQ]]] := With[{r=DateList[dObj]}, If[ListQ[r], Take[r, Length[date]], date]]
getDateFromDateObject[HoldPattern[dObj:DateObject[date_List, TimeObject[time_List, ___], ___?OptionQ]]] := With[{r=DateList[dObj]}, If[ListQ[r], Take[r, Length[date]+Length[time]], Join[date,time]]]


(*********************************         With Currency Conversions         *************************************)

(* 8: Source and Target currency the same *)
InflationAdjust[Quantity[q_, DatedUnit[SourceUnit_, SourceDate_]]?QuantityQ, DatedUnit[SourceUnit_, TargetDate_(*_?DateListQ|_Integer*)], opts:OptionsPattern[]] := 
	InflationAdjust[Quantity[q, DatedUnit[SourceUnit, SourceDate]], TargetDate, opts]

(* 9: No target date, but unit *)
InflationAdjust[q:Quantity[_, _DatedUnit]?QuantityQ, DatedUnit[TargetUnit_String], opts:OptionsPattern[]] := 
	InflationAdjust[q, DatedUnit[TargetUnit, currentDate[]], opts]
InflationAdjust[q_?QuantityQ, TargetUnit_String, opts:OptionsPattern[]] := 
	If[QuantityUnits`Private`couldBeACurrencyQ[TargetUnit], InflationAdjust[q, DatedUnit[TargetUnit, currentDate[]], opts], Message[InflationAdjust::notcur, TargetUnit]; $Failed]

(* 10: Target year, not full date *)
InflationAdjust[q:Quantity[_, _DatedUnit]?QuantityQ, DatedUnit[TargetUnit_, TargetDate_Integer], opts:OptionsPattern[]] := 
	InflationAdjust[q, DatedUnit[TargetUnit, {TargetDate}], opts]

(* 11: DatedUnit as second argument *)
InflationAdjust[q:Quantity[_, _DatedUnit]?QuantityQ, DatedUnit[TargetUnit_String, TargetDate_?DateListQ], opts:OptionsPattern[]] := 
	If[QuantityUnits`Private`couldBeACurrencyQ[TargetUnit], InflationAdjust[q, Quantity[1, DatedUnit[TargetUnit, TargetDate]], opts], Message[InflationAdjust::notcur, TargetUnit]; $Failed]

(* 12: Target Quantity, but not with full date *)
InflationAdjust[q1:Quantity[_, _DatedUnit]?QuantityQ, Quantity[q2_, DatedUnit[TargetUnit_, TargetDate_Integer]]?QuantityQ, opts:OptionsPattern[]]:=
	InflationAdjust[q1, Quantity[q2, DatedUnit[TargetUnit, {TargetDate}]], opts]
	
(* 13: Main function with currency conversion involved *)
InflationAdjust[q1:Quantity[_, _DatedUnit]?QuantityQ, Quantity[q2_, DatedUnit[TargetUnit_, TargetDate: _?DateListQ|_Integer]]?QuantityQ, opts:OptionsPattern[]]:= Module[
			{SourceUnitTargetDate = InflationAdjust[q1, TargetDate, opts]},
	(*Print@iInflationAdjust[q1, TargetDate, opts];*)
			(* Don't bother currency convert in target date, if we have not been able to time convert the source currency over time: might need revision  *)
			If[!QuantityQ[SourceUnitTargetDate] || !QuantityUnits`Private`couldBeACurrencyQ[QuantityUnit[SourceUnitTargetDate]], Return[Missing["NotAvailable"]]];
	(*Print@SourceUnitTargetDate;*)
			iCurrencyConvert[SourceUnitTargetDate, DatedUnit[TargetUnit, TargetDate]]
]


Clear@iCurrencyConvert
iCurrencyConvert[Quantity[q_, DatedUnit[SourceUnit_, SourceDate_]], DatedUnit[TargetUnit_String, TargetDate_List]] := Catch[Module[
	{SourceBase = QuantityUnits`Private`FindCurrencyUnitValue[SourceUnit], TargetBase = QuantityUnits`Private`FindCurrencyUnitValue[TargetUnit], api, dates=DateExpand[TargetDate], value},

	If[!MatchQ[SourceBase, {{_String}, _?(NumberQ[#]&)}], Message[InflationAdjust::notcur, SourceUnit]; Return[Missing["NotAvailable"]]];
	If[!MatchQ[TargetBase, {{_String}, _?(NumberQ[#]&)}], Message[InflationAdjust::notcur, TargetUnit]; Return[Missing["NotAvailable"]]];
	(*Print[SourceBase, "    ", TargetBase];*)
	
	value = q*SourceBase[[2]]/TargetBase[[2]];
	(*Print@value;*)
	api = If[SourceBase[[1, 1]] === TargetBase[[1, 1]], SourceBase[[2]], "Result"/.ReleaseHold[APICompute["CurrencyConversionMean", Join[{SourceBase[[1, 1]], TargetBase[[1, 1]]}, dates]]]]; (*Print@api;*)
	Switch[api,
		_?NumberQ, value*Quantity[api, DatedUnit[TargetUnit, trimDate[TargetDate]]],
		"noconn", Quantity[value/ (DatedUnit[TargetUnit, TargetDate] / DatedUnit[SourceUnit, TargetDate]), DatedUnit[TargetUnit, trimDate[TargetDate]]],
		_, Message[InflationAdjust::insfferd, SourceUnit, TargetUnit, Replace[TargetDate, {y_Integer} -> y]]; Missing["NotAvailable"]
		]
], $tag]

iCurrencyConvert[___] := $Failed



(*********************************         Compound Source Unit          *************************************)

InflationAdjust[Quantity[q_, SourceUnit_]?QuantityQ, opts:OptionsPattern[]] := InflationAdjust[Quantity[q, SourceUnit], currentDate[], opts]

InflationAdjust[Quantity[q_, SourceUnit_?(FreeQ[#, DatedUnit]&)]?QuantityQ, DatedUnit[TargetUnit_, TargetDate: _?DateListQ|_Integer], opts:OptionsPattern[]] := 
	QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, SourceUnit/.u_String :> ToDatedUnit[u]], DatedUnit[TargetUnit, trimDate[TargetDate]], opts] /;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit]

InflationAdjust[Quantity[q_, SourceUnit_]?QuantityQ, DatedUnit[TargetUnit_, TargetDate: _?DateListQ|_Integer], opts:OptionsPattern[]] := 
	QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, SourceUnit], DatedUnit[TargetUnit, trimDate[TargetDate]], opts] /;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit]

InflationAdjust[Quantity[q_, SourceUnit_?(FreeQ[#, DatedUnit]&)]?QuantityQ, TargetDate: _?DateListQ|_Integer, opts:OptionsPattern[]] := With[
	{target = Cases[SourceUnit, u_String?(QuantityUnits`Private`couldBeACurrencyQ[#]&) :> ToDatedUnit[u, trimDate[TargetDate]]]},
	QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, SourceUnit/.u_String :> ToDatedUnit[u]], Sequence@@target, opts] /;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit]
]

InflationAdjust[Quantity[q_, SourceUnit_]?QuantityQ, TargetDate: _?DateListQ|_Integer, opts:OptionsPattern[]] := With[
	{target = Cases[SourceUnit, DatedUnit[u_, _] :> DatedUnit[u, trimDate[TargetDate]]]}, 
	QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, SourceUnit], Sequence@@target, opts] /;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit]
]

InflationAdjust[q1_?QuantityQ, Quantity[q2_, DatedUnit[TargetUnit_, TargetDate: _?DateListQ|_Integer]]?QuantityQ, opts:OptionsPattern[]]:= 
	InflationAdjust[q1, DatedUnit[TargetUnit, TargetDate], opts]



(*********************************         Compound Target Unit          *************************************)

InflationAdjust[Quantity[q_, SourceUnit_]?QuantityQ, TargetUnit_?(Count[#, DatedUnit, Infinity, Heads -> True] == 1 && !QuantityQ[#] &), opts:OptionsPattern[]]/;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit] := 
	Module[
		{rest, intermediate, res},
		rest = DeleteCases[TargetUnit, _DatedUnit, Infinity];
		intermediate = QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, SourceUnit], First@Cases[TargetUnit, _DatedUnit], opts];
		res = UnitConvert[intermediate, rest];
		If[QuantityQ[res], res, $Failed]
		]

(*InflationAdjust[Quantity[q_, SourceUnit_]?QuantityQ, TargetUnit_?(QuantityUnits`Private`couldBeACurrencyQ[#] &), opts:OptionsPattern[]]/;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit] := 
	Module[
		{currency, rest, intermediate, res},
		currency = Select[TargetUnit, QuantityUnits`Private`couldBeACurrencyQ[#] &];
		rest = Select[TargetUnit, !QuantityUnits`Private`couldBeACurrencyQ[#] &]; Print@rest;
		intermediate = InflationAdjust[Quantity[q, SourceUnit], DatedUnit[currency], opts]; Print@FullForm[intermediate];
		res = UnitConvert[intermediate, rest];
		If[QuantityQ[res], res, $Failed]
		]*)

InflationAdjust[Quantity[q_, SourceUnit_]?QuantityQ, TargetUnit_?(QuantityUnits`Private`couldBeACurrencyQ[#] &), opts:OptionsPattern[]]/;
	(	QuantityUnits`Private`couldBeACurrencyQ[SourceUnit] && 
		Length[Cases[TargetUnit, _?(! QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}]] == 1 &&
		Length[Cases[TargetUnit, _?(QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}]] == 1
	):= With[{
			other = First@Cases[TargetUnit, _?(! QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}], 
			currency = First@Cases[TargetUnit, _?(QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}]
			}, (*Print@currency;*)
		UnitConvert[InflationAdjust[Quantity[q, SourceUnit], DatedUnit[currency], opts], other]
	]


(*********************************         TimeSeries         *************************************)

InflationAdjust[ts_?QuantityTimeSeriesQ, rest___, opts:OptionsPattern[]] := InflationAdjust[#, rest, opts] &/@ ts
	
InflationAdjust[td_TemporalData, rest___, opts:OptionsPattern[]] := 
	td["Caller"][Replace[td["Paths"], {t_Integer, q_Quantity} :> InflationAdjust[{DateList[t], q}, rest, opts], {2}]]

(* Single element timeseries with no date *)
InflationAdjust[{SourceDate: _?DateListQ|_Integer, Quantity[q_, DatedUnit[SourceUnit_String]|SourceUnit_String]?QuantityQ}, rest___, opts:OptionsPattern[]] := 
	{SourceDate, InflationAdjust[Quantity[q, DatedUnit[SourceUnit, SourceDate]], rest, opts]}

(* Single element timeseries with date, covers compound units as well *)
InflationAdjust[{Date: _?DateListQ|_Integer, Quantity[q_, SourceUnit_?(!FreeQ[#, DatedUnit]&)]?QuantityQ}, rest___, opts:OptionsPattern[]] := 
	{Date, InflationAdjust[Quantity[q, SourceUnit], rest, opts]}

(* Single element timeseries with no date and no target for compound units *)
InflationAdjust[{SourceDate: _?DateListQ|_Integer, Quantity[q_, DatedUnit[SourceUnit_]|SourceUnit_]?QuantityQ}, opts:OptionsPattern[]] := With[
	{
		source = SourceUnit/. u_String?(QuantityUnits`Private`couldBeACurrencyQ[#]&) :> ToDatedUnit[u, SourceDate],
		target = Cases[SourceUnit, u_String?(QuantityUnits`Private`couldBeACurrencyQ[#]&) :> DatedUnit[u, currentDate[]]]
	},
	{SourceDate, QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, source], Sequence@@target, opts]} /;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit]
]

(* Single element timeseries with no date with target for compound units *)
InflationAdjust[{SourceDate: _?DateListQ|_Integer, Quantity[q_, DatedUnit[SourceUnit_]|SourceUnit_]?QuantityQ}, target: _DatedUnit | _String | _?DateListQ | _Integer, opts:OptionsPattern[]] := With[
	{source = SourceUnit/. u_String?(QuantityUnits`Private`couldBeACurrencyQ[#]&) :> ToDatedUnit[u, SourceDate]},
	{SourceDate, QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, source], target, opts]} /;QuantityUnits`Private`couldBeACurrencyQ[SourceUnit]
]

InflationAdjust[{SourceDate: _?DateListQ|_Integer, Quantity[q_, SourceUnit_]?QuantityQ}, TargetUnit_, opts:OptionsPattern[]]/;
	(	QuantityUnits`Private`couldBeACurrencyQ[SourceUnit] && 
		Length[Cases[TargetUnit, _?(! QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}]] == 1 &&
		Length[Cases[TargetUnit, _?(QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}]] == 1
	):= With[{
			other = First@Cases[TargetUnit, _?(! QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}], 
			currency = First@Cases[TargetUnit, _?(QuantityUnits`Private`couldBeACurrencyQ[#] && KnownUnitQ[#] &), {-1}],
			source = SourceUnit/. u_String?(QuantityUnits`Private`couldBeACurrencyQ[#]&) :> ToDatedUnit[u, SourceDate]
			},
		{SourceDate, UnitConvert[QuantityUnits`Private`InflationAdjustUnitConvert[Quantity[q, source], currency, opts], other]} 
	]

(*********************************             Currency Conversion Helper Functions           *******************************************)

WolframAlpha;(*load WolframAlphaClient`*)

Attributes[APICompute] = {HoldAll};
APICompute[type_, args_] := With[{res=Internal`MWACompute[type,args,"ContextPath" -> {"Internal`MWASymbols`", "System`", 
  "System`InflationAdjust`Private`"},"MessageHead"->InflationAdjust]},
  If[SameQ[res,$Failed],"Result"->"noconn",res]
  ]

(*********************************             General Helper Functions           *******************************************)

ToDatedUnit[unit_String] := DatedUnit[unit, currentDate[]] /;QuantityUnits`Private`couldBeACurrencyQ[unit]
ToDatedUnit[unit_String, Date_?DateListQ] := DatedUnit[unit, Date] /;QuantityUnits`Private`couldBeACurrencyQ[unit]
ToDatedUnit[unit_String, Date_Integer] := DatedUnit[unit, {Date}] /;QuantityUnits`Private`couldBeACurrencyQ[unit]
ToDatedUnit[unit_String, Date_?DateObjectQ] := With[{d = DateList[Date]}, DatedUnit[unit,d] /;And[QuantityUnits`Private`couldBeACurrencyQ[unit],DateListQ[d]]]
ToDatedUnit[unit_String, ___] := unit
(*ToDatedUnit[unit_DatedUnit] := unit*)

trimDate[l_List] := Take[l, UpTo[3]]
trimDate[any__] := any

PreferredTargetUnit[units_List] := With[{unitsonly = DeleteDuplicates[units/.DatedUnit[u_, ___] :> u]},
	SortBy[{#, Length@CPIData[#]}&/@unitsonly, Last][[-1, 1]]
]

currentDate[] := DateList[][[;; 1]]

Clear[DateListQ]
DateListQ[x_List] := Internal`PossibleDateQ[x]
DateListQ[__] := False

DateExpand[date_] := date/.{
	{y_} :> {{y, 1, 1}, {y, 12, 31}},
	{y_, m_} :> {{y, m, 1}, {y, m, 31}},
	any_ :> {any, any}
}

TimeSeriesQ[x_] := MatchQ[x, {{_?DateListQ | _Integer, _?NumberQ} ..}]
QuantityTimeSeriesQ[x_] := MatchQ[x, {{_?DateListQ | _Integer, _?QuantityQ} ..}]

yearsToSeconds[ys_] := Module[{res},
    (* if there were 365 days per year then there would be 31536000 seconds in a year *)
    res = ys - 1900;
    res = res*31536000 + (Quotient[res, 4, 1] - Quotient[res, 100, 1] + Quotient[res, 400, -299]) 86400
  ];
DateListToSeconds[dls:{___List}] := Internal`DateListToSeconds /@ dls;
DateListToSeconds[dl_] := With[{res=Internal`DateListToSeconds[dl]}, res /; NumberQ[res]];

DateListToSeconds[{y_}] := yearsToSeconds[y];

DateListToSeconds[{y_, m_}] :=
  yearsToSeconds[y] + (UnitStep[m-2] 31 + (* KroneckerDelta is not listable *)
  UnitStep[m-3] (28 + (1-Unitize[Mod[y, 4]] + Unitize[Mod[y, 100]] - Unitize[Mod[y, 400]])) +
  UnitStep[m-4] 31 + UnitStep[m-5] 30 + UnitStep[m-6] 31 + UnitStep[m-7] 30 + UnitStep[m-8] 31 +
  UnitStep[m-9] 31 + UnitStep[m-10] 30 + UnitStep[m-11] 31 + UnitStep[m-12] 30)86400

(* day 1 has an offset of 0 seconds *)
DateListToSeconds[{y_, m_, d_}] := DateListToSeconds[{y, m}] + (d-1) 86400;
DateListToSeconds[{y_, m_, d_, h_}] := DateListToSeconds[{y,m}] + Total[ {d, h} {86400, 3600} ] - 86400;
DateListToSeconds[{y_, m_, d_, h_, min_}] := DateListToSeconds[{y,m}] + Total[ {d, h, min} {86400, 3600, 60} ] - 86400;
DateListToSeconds[{y_, m_, d_, h_, min_, s_}] := DateListToSeconds[{y, m}] + Total[ SetAccuracy[{d, h, min, s}, Accuracy[s]] * {86400, 3600, 60, 1} ] - 86400;

Options[DateListInterpolation] = {Sequence@@Options[Interpolation], "SecondsOffSet" -> None};
DateListInterpolation[datelist:{{_List, _} ...}, opts:OptionsPattern[]] :=
Module[{interpolationfn, data},
	data = datelist;
	data[[All, 1]] = DateListToSeconds[data[[All, 1]]];
	With[{offset = OptionValue["SecondsOffSet"]}, 
		If[MatchQ[offset, _?NumberQ], data[[All, 1]] += offset]];
    data = Tally[data, First[#1] == First[#2]&][[All,1]];
	interpolationfn = Interpolation[data, Sequence@@FilterRules[{opts}, Options[Interpolation]]];
	data = With[{i=interpolationfn}, i[AbsoluteTime[#]]&];
	data
];

DateToDateList[d_?DateListQ] := Switch[d,
	{_}, Join[d, {7, 1}], (* The more precise would be {DateList[][[1]], 6, 27} *)
	{_, _}, Join[d, {15}],
	_, d[[;;3]]
]
	
DateInput[date_List] := Switch[Length@date,
	1, DateString[date, {"Year"}],
	2, DateString[date, {"Month", "/", "Year"}],
	3, DateString[date, {"Month", "/", "Day", "/", "Year"}],
	_, DateString[date[[;;3]], {"Month", "/", "Day", "/", "Year"}]
] 

SetAttributes[InflationAdjust,{ReadProtected,Protected}];

End[];

