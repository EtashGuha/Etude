(* ::Package:: *)

(*:Name: Calendar Package *)

(*:Context: Calendar` *)

(*:Title: Unifying Calendar Computations by Considering a Calendar 
          as a Mixed Radix Representation Generalized to Lists.
*)

(*:Summary:
This package provides functions for basic calendar operations.
The Gregorian, Julian, Islamic, and Jewish calendars are supported.
*)

(*:Author: Ilan Vardi *)

(*:Mathematica Version: 5.1 *)

(*:Package Version: 1.1 *)

(*:History: V1.0, Ilan Vardi
		V1.0.1, minor revisions, John M. Novak, February 1992.
		V1.0.2 more minor revisions, John M. Novak, April 1994.
			-- added ability of externally visible functions to use
			   six argument form of Date[], based on suggestion by
			   Larry Calmer
        V1.1 added the Jewish calendar to fix issues with JewishNewYear,
             John M. Novak, December 2004.
        Altered context to fit new paclet system for M--6,
             Brian Van Vertloo, January 2007.
*)

(*:Keywords: Calendar, Julian, Gregorian, Islamic, Digits.
*)

(*:Requirements: None. *)

(* :Copyright: Copyright 1992-2007, Wolfram Research, Inc.*)

(*:Warnings: A date is written as {y, m, d} where y is the year,
             m is the month, and d is the day. 

             The computations can be done either in the Gregorian,
             Julian, Jewish, or Islamic calendars. The Gregorian calendar is 
             the calendar commonly in use today and this calendar is 
             taken to be the default if no calendar is specified.

             Great Britain and its colonies switched from the Julian
             calendar to the Gregorian calendar in September, 1752.
             In these countries {1752, 9, 2} was followed by 
             {1752, 9, 14}. The default calendar for dates on
             or before {1752, 9, 2} is taken to be the Julian calendar.
             This requires making some adjustments to DayOfWeek, 
             DaysBetween, and DaysPlus. For example,
             DaysBetween[{1752, 9, 2}, {1752, 9, 14}] will return 1.
             
             Catholic countries switched from the Julian to the Gregorian
             calendar in October 1582, so that {1582, 10, 4} was followed
             by {1582, 10, 15}. This change to the Package can be made
             quite easily.

             The algorithm for the Julian calendar will fail for the year
             4 and earlier since the leap years from 40 B.C. to the
             year 4 did not follow a regular pattern (see Bickerman's book),
             and also the year zero does not exist (it is taken to be 
             1 B.C.). The first valid Julian date is therefore {4, 3, 1}.

             The implementation for the Jewish calendar is based on the
             Dershowitz and Reingold paper, and isn't using the same
             mixed-radix implementation as the other calendars.

			 Input to functions that accept a date can also take the
			 date in {year, month, day, hour, minute, second} form
			 (such as that returned by Mathematica's Date[] command).
			 Output will also be returned in this form, but no calculations
			 are performed based on hour, minute, and second.

*)

(*:Source:  Ilan Vardi, Computational Recreations in Mathematica,
            Addison-Wesley 1991, Chapter 3.

            E.R. Berlekamp, J.H. Conway, and R.K. Guy, Winning Ways,
            Volume 2, Academic Press 1982, pages 795-800.

            W.A. Schocken, The Calculated Confusion of the Calendar,
            Vantage Press, New York 1976.

            E.J. Bickerman, The Chronology of the Ancient World,
            Revised Edition, Thames and Hudson, London 1980.
            
            Nachum Dershowitz and Edward M. Reingold, "Calendrical
            Calculations", Software - Practice and Experience 20 (1990), 899-928.
*)

(*:Limitations: This package is meant to show how Mathematica can be
                used to give a unified treatment of a problem usually
                done using specialized hacks. This means that these 
                functions can be speeded up somewhat. For example, 
                DayOfWeek can be computed efficiently for the Gregorian
                calendar by using an algorithm of Reverend Zeller. See
                Computational Recreations in Mathematica, Problem 3.1.
                
                Calculations involving the Jewish calendar with dates
                before the year 1 Gregorian are not trustworthy.
*)

(*:Discussion: This package was written in order to give a unified
               treatment of the basic calendar operations. This is
               done by considering the calendar as a mixed radix 
               positional number system where the radices are lists.
               This requires a further generalization as the radix must
               actually be a tree. This is necessary since, for example,
               the numbers of days in a month depend on what year it is.
               The calendars are quite regular, so they are most 
               compactly represented as trees of functions. See 
               Chapter 3 of Computational Recreations in Mathematica
               for details.

               The Julian calendar is the simplest calendar, consisting
               of the usual Western calendar with leap years every 
               four years. This calendar gives a year that is slightly 
               too long. It was replaced with the Gregorian calendar
               in Catholic countries in 1582 and by Britain and its 
               colonies in 1752. It is still used to compute Greek 
               Orthodox holidays such as Easter.

               The Gregorian calendar is the calendar presently in 
               use in the western world. It differs from the Julian
               calendar in eliminating leap years for centuries not
               divisible by 400. In other words, 1900 was not a leap 
               year, but the year 2000 will be a leap year.

               The Islamic calendar is a purely lunar calendar, and a year
               has 354 or 355 days. The months do not correspond to the 
               solar year, and migrate over the solar year following a 
               30 year cycle. The names of the months are the following:

               Muharram, Safar, Rabia I, Rabia II, Jumada I, Jumada II, 
               Rajab, Sha'ban, Ramadan, Shawwal, Dhu al-Qada, Dhu al-Hijah

               The function computing Easter is taken directly from Winning Ways.
               
               It should be noted that Edward M. Reingold and Nachum Dershowitz's
               "Calendrical Calculations: The Millenium Edition" (Cambridge
               University Press, 2001) provides a much more extensive and
               accurate implementation of calendrical computations.
*)

BeginPackage["Calendar`"]

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"Calendar`"],
StringMatchQ[#,StartOfString~~"Calendar`*"]&]//ToExpression;
];

If[Not@ValueQ[DayOfWeek::usage],DayOfWeek::usage = "DayOfWeek[{y, m, d}] gives the day of the \
week for year y, month m, and day d. The default \
calendar is the usual American calendar, but can be changed with the \
Calendar option. The date can also be given in {y, m, d, h, m, s} form."];

If[Not@ValueQ[DaysBetween::usage],DaysBetween::usage = "DaysBetween[{y1,m1, d1}, {y2,m2,d2}] gives \
the number of days between the dates {y1, m1, d1} and {y2, m2, d2}. \
The default calendar is the usual American calendar, but can be \
changed with the Calendar option. \
Dates can also be given in {y, m, d, h, m, s} form."];

If[Not@ValueQ[DaysPlus::usage],DaysPlus::usage = "DaysPlus[{y, m, d}, n] gives the date n days \
after {y, m, d}. The default calendar is the usual \
American calendar, but can be changed with the \
Calendar option. The date can also be given in {y, m, d, h, m, s} \
form."];

If[Not@ValueQ[CalendarChange::usage],CalendarChange::usage = "CalendarChange[date, calendar1, calendar2] \
converts a date in calendar1 to a date in calendar2."];

If[Not@ValueQ[DateQ::usage],DateQ::usage = "DateQ[date] indicates whether date is valid in the \
current calendar. The Calendar option may be used to specify which \
calendar is used."];

If[Not@ValueQ[Calendar::usage],Calendar::usage = "Calendar is an option for calendar functions \
indicating which calendar system to use: Gregorian, Julian, Islamic, \
or Jewish. If set to Automatic, the Julian calendar is used \
before 2 September 1752, and the Gregorian calendar afterwards."];

If[Not@ValueQ[Gregorian::usage],Gregorian::usage =
"Gregorian specifyies that the Gregorian calendar is to be used. \
It is a value for the Calendar option and can be used as an \
argument to CalendarChange. It is assumed that the date must \
be {1752, 9, 14} or later."];

If[Not@ValueQ[Julian::usage],Julian::usage =
"Julian specifyies that the Julian calendar is to be used. \
It is a value for the Calendar option and can be used as an \
argument to CalendarChange. This calendar is only valid starting {4, 3, 1}."];

If[Not@ValueQ[Islamic::usage],Islamic::usage = 
"Islamic specifyies that the Islamic calendar is to be used. \
It is a value for the Calendar option and can be used as an \
argument to CalendarChange. This calendar began on {622, 7, 16} Julian, \
or {1, 1, 1} in the Islamic calendar (the Hejira)."];

If[Not@ValueQ[Jewish::usage],Jewish::usage = 
"Jewish specifyies that the Jewish calendar is to be used. \
It is a value for the Calendar option and can be used as an \
argument to CalendarChange."];

If[Not@ValueQ[EasterSunday::usage],EasterSunday::usage = "EasterSunday[year] computes the date of Easter \
Sunday in the Gregorian calendar according to the Gregorian \
calculation."];

If[Not@ValueQ[EasterSundayGreekOrthodox::usage],EasterSundayGreekOrthodox::usage = "EasterSundayGreekOrthodox[year] computes the \
date of Easter Sunday according to the Greek Orthodox church. The \
result is given as a Gregorian date."];

If[Not@ValueQ[JewishNewYear::usage],JewishNewYear::usage = "JewishNewYear[y] gives the date of the Jewish \
New Year occurring in Christian year y, 1900 <= y < 2100. Add 3761 to \
Christian year y to get the corresponding new Jewish Year."];

(*
If[Not@ValueQ[Sunday::usage],Sunday::usage = "Sunday is a day of the week."];

If[Not@ValueQ[Monday::usage],Monday::usage = "Monday is a day of the week."];

If[Not@ValueQ[Tuesday::usage],Tuesday::usage = "Tuesday is a day of the week."];

If[Not@ValueQ[Wednesday::usage],Wednesday::usage = "Wednesday is a day of the week."];

If[Not@ValueQ[Thursday::usage],Thursday::usage = "Thursday is a day of the week."];

If[Not@ValueQ[Friday::usage],Friday::usage = "Friday is a day of the week."];

If[Not@ValueQ[Saturday::usage],Saturday::usage = "Saturday is a day of the week."];
*)

Unprotect[DayOfWeek, DaysBetween, DaysPlus, CalendarChange, Calendar, 
        Julian, Gregorian, Islamic, Jewish, EasterSunday, JewishNewYear,
        EasterSundayGreekOrthodox]

Begin["`Private`"]

$knowncalendars = {Julian, Gregorian, Islamic, Jewish};

validCalendarQ[cal_Symbol] := MemberQ[$knowncalendars, cal]
validCalendarQ[_] := False

Options[DayOfWeek]:= {Calendar -> Automatic}

Options[DaysBetween]:= {Calendar -> Automatic}

Options[DaysPlus]:= {Calendar -> Automatic}

DayOfWeek::cal = DaysBetween::cal = DaysPlus::cal =
"The value `1` for Calendar is not a known type of calendar. Using \
Automatic instead.";

DayOfWeek::baddate = DaysBetween::baddate =
     DaysPlus::baddate = CalendarChange::baddate =
"The date `1` is not valid for the calendar `2`.";

DayOfWeek::datestruct = DaysBetween::datestruct =
     DaysPlus::datestruct = CalendarChange::datestruct =
"To be a valid date, the argument `1` must have the form \
{y, m, d} or {y, m, d, h, mn, s}, where y, m, and d are all integers.";

DayOfWeekNumber[date_List, opts___]:= 
    Block[{calendar = Calendar /. Flatten[{opts, Options[DayOfWeek]}]},
        If[!(validCalendarQ[calendar] || calendar === Automatic),
            Message[DayOfWeek::cal, calendar];
            calendar = Automatic
        ];
        If[calendar === Automatic,
            calendar = If[OrderedQ[{date, {1752, 9, 2}}], Julian, Gregorian]
        ];
        Mod[DateToNumber[date, calendar] + DayOfWeekInit[calendar], 7]
    ]

DayOfWeek[date_List?(Length[#] == 6 &), opts___?OptionQ] :=
	DayOfWeek[Take[date, 3], opts]

dayofweekobsmsgflag = True; 

DayOfWeek[date_, opts___?OptionQ]/;datecheckQ[date, DayOfWeek, opts] :=
 (
        If[TrueQ[dayofweekobsmsgflag],
            Message[General::obsfun, Calendar`DayOfWeek, DayName];
            dayofweekobsmsgflag = False;
        ];
	With[{day = DayOfWeekNumber[date, opts]},
 {
  System`Sunday, System`Monday, System`Tuesday, System`Wednesday, System`Thursday, System`Friday, System`Saturday
  }[[
      1 + day
    ]]/; NumberQ[day]
 ])        

DayOfWeekInit[Gregorian] = 0

DayOfWeekInit[calendar_]:= DayOfWeekInit[calendar] = 
 Mod[3 - 
     DateToNumber[
     CalendarChange[{1990, 10, 31}, Gregorian, calendar], 
     calendar], 
     7]

DaysBetween[date1_List, date2_List, opts___]:= (
	0 /; date1 == date2)

DaysBetween[date1_List?(Length[#] == 6 &), rest___] :=
   DaysBetween[Take[date1, 3], rest]

DaysBetween[date1_List, date2_List?(Length[#] == 6 &), rest___] :=
   DaysBetween[date1, Take[date2, 3], rest]

daysbetweenobsmsgflag = True;

DaysBetween[date1_, date2_, opts___?OptionQ]/;
           (datecheckQ[date1, DaysBetween, opts] &&
            datecheckQ[date2, DaysBetween, opts]) :=
    (
        If[TrueQ[daysbetweenobsmsgflag],
            Message[General::obsfun, Calendar`DaysBetween, DayCount];
            daysbetweenobsmsgflag = False;
        ];
	Module[{calendar = Calendar /. Flatten[{opts, Options[DaysBetween]}],
            d1, d2, s},
        If[!(validCalendarQ[calendar] || calendar === Automatic),
            Message[DaysBetween::cal, calendar];
            calendar = Automatic
        ];
        If[calendar === Automatic,
            If[OrderedQ[{date2, date1}],
                d1 = date2; d2 = date1; s = -1,
                d1 = date1; d2 = date2; s = 1
            ];
            If[OrderedQ[{d1, {1752, 9, 2}}] && 
                    OrderedQ[{{1752, 9, 14}, d2}], 
                s * (1 + DaysBetween[d1, {1752, 9, 2}, Calendar -> Julian] +
                    DaysBetween[{1752, 9, 14}, d2, Calendar -> Gregorian]),
                s * DaysBetween[d1, d2, Calendar -> 
                          If[OrderedQ[{d1, {1752, 9, 2}}], 
                             Julian, Gregorian]]
            ], 
            DateToNumber[date2, calendar] - DateToNumber[date1, calendar]
        ]
    ])

DaysPlus[date_List?(Length[#] == 6 &), n_Integer, rest___?OptionQ] :=
    (
	With[{res = DaysPlus[Take[date, 3], n, rest]},
        Join[res, Take[date, -3]]/;ListQ[res]
    ])

daysplusobsmsgflag = True;

DaysPlus[date_, n_Integer, opts___?OptionQ]/;datecheckQ[date, DaysPlus, opts] := 
 (
	Block[{calendar = Calendar /. Flatten[{opts, Options[DaysPlus]}], d},
        If[TrueQ[daysplusobsmsgflag],
            Message[General::obsfun, Calendar`DaysPlus, DayPlus];
            daysplusobsmsgflag = False;
        ];
        If[!(validCalendarQ[calendar] || calendar === Automatic),
            Message[DaysPlus::cal, calendar];
            calendar = Automatic
        ];
        If[calendar === Automatic || !validCalendarQ[calendar], 
           If[OrderedQ[{date, {1752, 9, 2}}],
              d = DaysPlus[date, n, Calendar -> Julian];
              If[OrderedQ[{{1752, 9, 3}, d}],
                 CalendarChange[d, Julian, Gregorian],
                 d],
              d = DaysPlus[date, n, Calendar -> Gregorian];
              If[OrderedQ[{d, {1752, 9, 13}}],
                 CalendarChange[d, Gregorian, Julian],
                 d]
              ],
            NumberToDate[DateToNumber[date, calendar] + n, calendar]
           ]
        ])

CalendarChange[date_List?(Length[#] == 6 &), rest___] :=
    With[{res = CalendarChange[Take[date, 3], rest]},
        Join[res, Take[date, -3]]/;ListQ[res]
    ]

CalendarChange[date_, calendar1_?validCalendarQ, calendar1_]/;
       datecheckQ[date, CalendarChange, Calendar -> calendar1] :=
    date

CalendarChange[date_, calendar1_?validCalendarQ, calendar2_?validCalendarQ]/;
       datecheckQ[date, CalendarChange, Calendar -> calendar1] := 
 (
	NumberToDate[DateToNumber[date, calendar1] + 
       CalendarChangeInit[calendar1, calendar2], calendar2])

Options[DateQ] := {Calendar -> Automatic}

DateQ[{y_, m_, d_, r___}, opts___] :=
    (
	Module[{cal = Calendar/.Flatten[{opts, Options[DateQ]}]},
        dateQ[y, m, d, cal]
    ])

DateQ[any___] := False (* fallthrough *)

(* internal-use only version that emits message on failure *)
datecheckQ[{y_Integer, m_Integer, d_Integer} |
               {y_Integer, m_Integer, d_Integer, _, _, _}, func_, opts___] :=
    Module[{cal = Calendar/.Flatten[{opts, Options[DateQ]}], res},
        res = dateQ[y, m, d, cal];
        If[!res,
            Message[MessageName[func, "baddate"], {y, m, d}, cal]
        ];
        res
   ]

datecheckQ[any_, func_, opts___] :=
    (Message[MessageName[func, "datestruct"], any]; False)

dateQ[0, _, _, Automatic] := False (* none of the calendars have year 0 *)

dateQ[y_, m_, d_, Automatic] :=
    Which[OrderedQ[{{y, m, d}, {1752, 9, 2}}], dateQ[y, m, d, Julian],
          OrderedQ[{{1752, 9, 14}, {y, m, d}}], dateQ[y, m, d, Gregorian],
          True, False (* remaining days don't exist in the Automatic calendar *)
    ]

dateQ[any___] := False (* fallthrough *)

(* dateQ for rest with each specific calendar's functions *)

EasterSunday[y_Integer]:= 
      (
	Block[{paschal, golden, c, h, t},
            golden= Mod[y,19] +1;
            h = Quotient[y,100];
            c = - h + Quotient[h,4] + Quotient[8(h +11),25];
            t = DaysPlus[{y, 4, 19}, - Mod[11 golden +c, 30],
                         Calendar -> Gregorian];
            paschal = If[(t[[3]] == 19) || (t[[3]] == 18 && golden >11), 
                          t - {0,0,1} , t];
            DaysPlus[paschal, 
             7 - DayOfWeekNumber[paschal, Calendar -> Gregorian]]
           ])

EasterSundayGreekOrthodox[y_Integer]:= 
      (
	Block[{paschal, golden, c, h, t},
            golden= Mod[y,19] +1;
            h = Quotient[y,100];
            c = 3;
            t = DaysPlus[{y, 4, 19}, - Mod[11 golden +c, 30],
                         Calendar -> Gregorian];
            paschal = If[(t[[3]] == 19) || (t[[3]] == 18 && golden >11), 
                          t - {0,0,1} , t];
            CalendarChange[DaysPlus[paschal, 
                    7 - DayOfWeekNumber[paschal, Calendar -> Julian]],
                    Julian, Gregorian]
           ])

(* Generalization of Digits to mixed list radices *)

MyDigits[n_, list_List, path_]:= 
         Block[{md = MyDigitsInit[n, list, path]},
                If[Last[md] != 0, 
                   md,
                   MapAt[# + 1&,
                         MyDigitsInit[n-1, list, path], 
                         {-1}]
               ]]

MyDigitsInit[n_, {}, _]:= {n}

MyDigitsInit[n_, list_List, path_]:= 
   Block[{r = MyQuotient[n, First[list][path]]}, 
   Prepend[MyDigits[MyMod[n, First[list] [path]], 
                        Rest[list], Append[path, r]],
           MyQuotient[n, First[list] [path]]]]



DigitsToNumber[date_, list_, path_]:= 
 1 + date[[1]] list[[1]][path+1][[1]] + Last[date] + 
 (Plus @@  
  (Fold[Plus, 0, Take[list[[#]][path+1], date[[#]]]]& /@
        Range[2, Length[list]]))

MyDigits[n_, b_List]:= {n} /; b == {} ||  0 < n < Last[b] 

MyDigits[n_, b_List]:= 
Append[MyDigits[Quotient[n, Last[b]], Drop[b, -1]],
       Mod[n, Last[b]]]

DigitsToNumber[{n_}, b_]:= n

DigitsToNumber[digits_List, b_]:=
DigitsToNumber[Drop[digits, -1], 
               Drop[b, -1]] Last[b] + Last[digits]

(* Quotient and Mod generalized to lists *)

MyQuotient[n_Integer, list_List]:= 
  Quotient[n, First[list]] + 1 /; Length[list] == 1

MyQuotient[n_Integer, list_List]:= 
   Block[{s = First[list], q = 1}, 
          While[n > s, q++; s += list[[q]]]; q] /; Length[list] > 1

MyMod[n_Integer, list_List]:= 
   Mod[n, First[list]] /; Length[list] == 1

MyMod[n_Integer, list_List]:= 
   n - Fold[Plus, 0, Take[list, MyQuotient[n, list]-1]] /; Length[list] > 1



(* Julian calendar *)

JulianCalendar = {JulianFourYears, JulianYears, JulianMonths}

JulianFourYears[_]:= {1461}

JulianYears[_]:= {365, 365, 365, 366}

JulianMonths[path_List]:= 
{31, 28 + Quotient[path[[2]], 4], 
 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}


NumberToDate[n_, Julian]:=  
Prepend[Drop[#, 2], DigitsToNumber[Take[# - 1, 2], {4}] + 1]& @
        MyDigits[n, JulianCalendar, {}]

DateToNumber[date_, Julian]:= 
Block[{d = Join[MyDigits[First[date]-1, {4}] , 
                Rest[date]-1]},
       d = Join[Table[0, {4 - Length[d]}], d];
       DigitsToNumber[d, JulianCalendar, d]]

dateQ[y_, 2, d_, Julian] := (* special leap-year handling *)
   1 <= d <= If[Mod[y, 4] === 0, 29, 28]

dateQ[y_, m_, d_, Julian]/; 1 <= m <= 12 := 
    1 <= d <= {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}[[m]]

(* Gregorian calendar *)

GregorianCalendar = 
{GregorianFourCenturies, GregorianCentury, 
 GregorianFourYears, GregorianYears, GregorianMonths}

GregorianFourCenturies[_]:= {146097}

GregorianCentury[_]:= {36524, 36524, 36524, 36525}

GregorianFourYears[path_]:= 
 Append[Table[1461, {24}], 1460 + Quotient[path[[2]], 4]]

GregorianYears[path_]:= 
 {365, 365, 365, 366 - 
 (1-Quotient[path[[2]], 4]) Quotient[path[[3]], 25]}

GregorianMonths[path_]:= 
{31, 28 + Quotient[path[[4]], 4] * (1 - 
   (1 - Quotient[path[[2]], 4]) Quotient[path[[3]], 25]),
 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

NumberToDate[n_, Gregorian]:= 
Prepend[Drop[#, 4], DigitsToNumber[Take[#-1, 4], {4,25,4}] + 1]& @
 MyDigits[n, GregorianCalendar, {}]

DateToNumber[date_, Gregorian]:= 
Block[{d = Join[MyDigits[First[date]-1, {4, 25, 4}] , Rest[date]-1]},
       d = Join[Table[0, {6 - Length[d]}], d];
       DigitsToNumber[d, GregorianCalendar, d]]

dateQ[y_, 2, d_, Gregorian] := (* special leap-year handling *)
   1 <= d <= If[Mod[y, 4] === 0 && !MemberQ[{100, 200, 300}, Mod[y, 400]], 29, 28]

dateQ[y_, m_, d_, Gregorian]/; 1 <= m <= 12 := 
    1 <= d <= {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}[[m]]

CalendarChangeInit[Gregorian, Julian] = 2

CalendarChangeInit[Julian, Gregorian] = -2


(* Islamic calendar *)


IslamicCalendar = {IslamicThirtyYears, 
                   IslamicYears, 
                   IslamicMonths}

IslamicThirtyYears[_]:= {30 354 + 11}

IslamicYears[_]:= 354 +
   {0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,
    1,0,1,0,0,1,0,0,1,0,1,0,0,1,0}

IslamicMonths[path_]:= 
  {30, 29, 30, 29, 30, 29, 30, 29, 30, 29, 30, 29 + 
   {0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,
    1,0,1,0,0,1,0,0,1,0,1,0,0,1,0}[[path[[2]]]]}


NumberToDate[n_, Islamic]:= 
Prepend[Drop[#, 2], DigitsToNumber[Take[#-1, 2], {30}] + 1]& @
 MyDigits[n, IslamicCalendar, {}]

DateToNumber[date_, Islamic]:= 
Block[{d = Join[MyDigits[First[date]-1, {30}], 
                Rest[date]-1]},
       d = Join[Table[0, {4 - Length[d]}], d];
       DigitsToNumber[d, IslamicCalendar, d]]

dateQ[y_, 12, d_, Islamic] := (* leap year handling *)
    1 <= d <= If[Mod[11 y + 14, 30] < 11, 30, 29]

dateQ[y_, m_, d_, Islamic] :=
    (1 <= m <= 12) && (1 <= d <= If[OddQ[m], 30, 29])

CalendarChangeInit[Islamic, Julian] = 
  DateToNumber[{622, 7, 15}, Julian]

CalendarChangeInit[Julian, Islamic] = 
-CalendarChangeInit[Islamic, Julian]

CalendarChangeInit[Gregorian, Islamic] = 
-DateToNumber[CalendarChange[{622, 7, 15}, Julian, Gregorian], 
              Gregorian]

CalendarChangeInit[Islamic, Gregorian] = 
-CalendarChangeInit[Gregorian, Islamic] 


(* Jewish Calendar *)
(* This is implemented after the algorithms in Dershowitz and Reingold's
   paper "Calendrical Calculations" from Software-Practice and Experience 20 (1990),
   pp. 899-928. Because it's not a direct mixed-radix implementation like the
   other calendars, the base absolute number is the same as that used for the
   Gregorian calendar, which makes for some convenient translations. --JMN 12/04 *)
JewishLeapYearQ[y_] := Mod[7 y + 1, 19] < 7

JewishCalendarElapsedDays[y_] :=
    Module[{monthsgoneby, partsgoneby, day, parts, altday},
        monthsgoneby = 235 * Quotient[y - 1, 19] +
                       12 * Mod[y - 1, 19] +
                       Quotient[7 * Mod[y - 1, 19] + 1, 19];
        partsgoneby = 13753 * monthsgoneby + 5604;
        day = 29 * monthsgoneby + Quotient[partsgoneby, 25920] + 1;
        parts = Mod[partsgoneby, 25920];
        altday = If[parts >= 19440 ||
                      (Mod[day, 7] === 2 && 
                         parts >= 9924 &&
                         !JewishLeapYearQ[y]) ||
                      (Mod[day, 7] === 1 &&
                         parts >= 16789 &&
                         JewishLeapYearQ[y - 1]),
                     day + 1, day];
        If[MemberQ[{0, 3, 5}, Mod[altday, 7]],
            altday + 1,
            altday
        ]
    ]

DaysInJewishYear[y_] := 
    JewishCalendarElapsedDays[y + 1] - JewishCalendarElapsedDays[y]

LongHeshvanYearQ[y_] := Mod[DaysInJewishYear[y], 10] === 5

ShortKislevYearQ[y_] := Mod[DaysInJewishYear[y], 10] ===  3

LastMonthOfJewishYear[y_] := If[JewishLeapYearQ[y], 13, 12]

LastDayOfJewishMonth[m_, y_] :=
    If[MemberQ[{2,4,6,10,13}, m] ||
            (m === 12 && !JewishLeapYearQ[y]) ||
            (m === 8 && !LongHeshvanYearQ[y]) ||
            (m === 9) && ShortKislevYearQ[y],
        29, 30]

dateQ[y_, m_, d_, Jewish] :=
    (1 <= m <= LastMonthOfJewishYear[y]) && (1 <= d <= LastDayOfJewishMonth[m, y])

(* utility function to sum values of a function until a condition on the
   counter is met *)
dosum[f_, t_, s_] :=
    Module[{count = s, sum = 0}, While[t[count], sum += f[count]; count++]; sum]

DateToNumber[{y_, m_, d_, ___}, Jewish] :=
    d +
    If[m < 7, 
        dosum[LastDayOfJewishMonth[#, y]&, (# <= LastMonthOfJewishYear[y])&, 7] +
            dosum[LastDayOfJewishMonth[#, y]&,(# < m)&, 1],
        dosum[LastDayOfJewishMonth[#, y]&, (# < m)&, 7]
    ] +
    JewishCalendarElapsedDays[y] -
    1373429

NumberToDate[n_, Jewish] :=
    Module[{year, month, day},
        year = Quotient[n + 1373429, 366];
        While[n >= DateToNumber[{year + 1, 7, 1}, Jewish], year++];
        month = If[n < DateToNumber[{year, 1, 1}, Jewish], 7, 1];
        While[n > DateToNumber[{year, month,
                                LastDayOfJewishMonth[month, year]}, Jewish],
              month++
        ];
        day = n - (DateToNumber[{year, month, 1}, Jewish] - 1);
        {year, month, day}
    ]

(* The 'absolute' numbering system being used is the same as the Gregorian
   calendar. *)
CalendarChangeInit[Jewish, Gregorian] = 0;
CalendarChangeInit[Gregorian, Jewish] = 0;

CalendarChangeInit[Islamic, Jewish] = 
-CalendarChangeInit[Gregorian, Islamic]
CalendarChangeInit[Jewish, Islamic] = 
-CalendarChangeInit[Islamic, Gregorian]

CalendarChangeInit[Julian, Jewish] = 
-CalendarChangeInit[Gregorian, Julian]
CalendarChangeInit[Jewish, Julian] = 
-CalendarChangeInit[Julian, Gregorian]

JewishNewYear[y_] := (
	CalendarChange[{y + 3761, 7, 1}, Jewish, Gregorian])

End[]   (* Calendar`Private` *)

Protect[DayOfWeek, DaysBetween, DaysPlus, CalendarChange, Calendar, 
        Julian, Gregorian, Islamic, Jewish, EasterSunday, JewishNewYear,
        EasterSundayGreekOrthodox]
        
       

EndPackage[]   (* Calendar` *)



