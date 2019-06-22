
(************************************************************************)
BeginPackage["XMLSchema`DateString`", {"JLink`"}];
(************************************************************************)

DateString::usage = "DateString[spec, elems] gives a string representing elements elems of the date or time specification spec."

(* More work/ideas need to be discussed here *)
(*
DateStringRefreshTime::usage = "DateStringRefreshTime[elems] returns the minimum time in seconds that the elems formats require for use with Refresh."
*)

FromDateString::usage = "FromDateString[str, elems] returns a date expression by parsing a string representing elements elems."

(************************************************************************)
Begin["`Private`"]
(************************************************************************)

(* Loaded for calling Miscellaneous`Calendar`Private`DayOfWeekNumber *)
Needs["Miscellaneous`Calendar`"];

(**********************************************
   DateString
 **********************************************)
 
DateString[] := DateString[ Date[]];
DateString[{}] := DateString[ Date[]];
  
DateString[elem_String] := DateString[{elem}];
 
DateString[{elems__String}] := DateString[Date[], {elems}];
 
DateString[d_?NumberQ] := DateString[ ToDate[d]];
DateString[d_?dateListPatternQ] := DateString[d, {"DateTime"}];
 
DateString[d_?NumberQ, elem_String] := DateString[ ToDate[d], elem];
DateString[d_?dateListPatternQ, elem_String] := DateString[d, {elem}];
 
DateString[d_?NumberQ, {elems__String}] := DateString[ ToDate[d], {elems}];
 
DateString[d_?dateListPatternQ, {elems__String}] := Block[{l = toLocale[$DateStringLocale], dd = ToDate[FromDate[d]]},
   StringJoin[ Flatten[convertDateStringForms[dd, l, #]& /@ {elems}]]
   ];
    

(**********************************************
   DateStringRefreshTime
 **********************************************)
 
DateStringRefreshTime[] := DateStringRefreshTime["DateTime"];
DateStringRefreshTime[{}] := DateStringRefreshTime["DateTime"];
DateStringRefreshTime[elem_String] := DateStringRefreshTime[{elem}];

DateStringRefreshTime[{elems__String}] := Module[{rt},
  rt = Flatten[ findRefreshTime /@ {elems}];
  If[ Length[rt] > 0, Min[rt], 1]
  ];
  
DateStringRefreshTime[e___] := 1


(**********************************************
   FromDateString
 **********************************************)
 
FromDateString[str_String] := FromDateString[str, "DateTime"];
 
FromDateString[str_String, elem_String] := FromDateString[str, {elem}];
 
FromDateString[str_String, {elems__String}] := JavaBlock[Module[{l = toLocale[$DateStringLocale], jForm, sd, d, c, yearFree},
  InstallJava[];
  jForm = Flatten[ convertToJavaForms[l, #]& /@ {elems}];
  jForm = jForm //. {{a___, convertStringLiteral[s1_], convertStringLiteral[s2_], b___} :> {a, convertStringLiteral[StringJoin[s1,s2]], b}};
  jForm = jForm /. {convertStringLiteral[s_] :> StringJoin["'", StringReplace[s, "'" -> "''"], "'"]};
  yearFree = !MemberQ[jForm, "yyyy" | "yy"];
  jForm = StringJoin[ jForm];
  sd = JavaNew["java.text.SimpleDateFormat",  jForm];
  d = Block[{$JavaExceptionHandler = Null&}, sd @ parse[str]];
  If[ d === Null || d === $Failed, Return[$Failed]];
  c = JavaNew["java.util.GregorianCalendar"];
  c @ setTime[d];
  {If[yearFree, 1900, c @ get[Calendar`YEAR]], c @ get[Calendar`MONTH] + 1, c @ get[Calendar`DATE], 
    c @ get[Calendar`HOURUOFUDAY], c @ get[Calendar`MINUTE], c @ get[Calendar`SECOND] + (c @ get[Calendar`MILLISECOND])/1000.}
  ]
 ];
 

convertToJavaForms[l_, "DateTime"] := {convertToJavaForms[l, "Date"], " ", convertToJavaForms[l, "Time"]};

convertToJavaForms[l_String?(StringMatchQ[#, "de*", IgnoreCase->True]&), "Date"] := 
  {convertToJavaForms[l,"Day"], ".", convertToJavaForms[l,"Month"], ".", convertToJavaForms[l,"Year"]};
  
convertToJavaForms[l_, "Date"] := 
  {convertToJavaForms[l,"MonthName"], " ", convertToJavaForms[l,"ShortDay"], ", ", convertToJavaForms[l,"Year"]};
  
convertToJavaForms[l_String?(StringMatchQ[#, "de*", IgnoreCase->True]&), "Time"] := 
  {convertToJavaForms[l,"Hour24"], ":", convertToJavaForms[l,"Minute"], ":", convertToJavaForms[l,"Second"]};
  
convertToJavaForms[l_, "Time"] := {convertToJavaForms[l,"ShortHour12"], ":", convertToJavaForms[l,"Minute"], ":", convertToJavaForms[l,"Second"], 
    " ", convertToJavaForms[l,"AMPM"]};

convertToJavaForms[l_, "Year"] = "yyyy";
convertToJavaForms[l_, "ShortYear"] = "yy";
convertToJavaForms[l_, "Month"] = "MM";
convertToJavaForms[l_, "ShortMonth"] = "M";
convertToJavaForms[l_, "Day"] = "dd";
convertToJavaForms[l_, "ShortDay"] = "d";

(* We need to decide how Hour decides to match either Hour24 or Hour12 *)
convertToJavaForms[l_, "Hour"] := convertToJavaForms[l, "Hour24"];
convertToJavaForms[l_, "ShortHour"] := convertToJavaForms[l, "ShortHour24"];

convertToJavaForms[l_, "Hour12"] = "hhh";
convertToJavaForms[l_, "ShortHour12"] = "h";
convertToJavaForms[l_, "Hour24"] = "HHH";
convertToJavaForms[l_, "ShortHour24"] = "H";
convertToJavaForms[l_, "Minute"] = "mm";
convertToJavaForms[l_, "ShortMinute"] = "m";
convertToJavaForms[l_, "Second"] = "ss";

convertToJavaForms[l_, "Millisecond"] = "SSS";
convertToJavaForms[l_, "ShortMillisecond"] = "S";

convertToJavaForms[l_, "TimeZone"] = "ZZZ";
  
convertToJavaForms[l_, "Quarter"] = "";
convertToJavaForms[l_, "QuarterName"] = "Quarter";
convertToJavaForms[l_, "ShortQuarterName"] = "Q";

convertToJavaForms[l_, "MonthName"] = "MMMM";
convertToJavaForms[l_, "ShortMonthName"] = "MM";
convertToJavaForms[l_, "MonthNameInitial"] = "M";

convertToJavaForms[l_, "DayName"] = "EEEE";
convertToJavaForms[l_, "ShortDayName"] = "EE";
convertToJavaForms[l_, "DayNameInitial"] = "E";

convertToJavaForms[l_, "AMPM"] = "a";

convertToJavaForms[l_, "TimeZoneName"] = "zzzz";
convertToJavaForms[l_, "ShortTimeZoneName"] = "z";

convertToJavaForms[l_, str_String] := convertStringLiteral[str];

convertToJavaForms[l_, expr___] := expr;

 
(**********************************************
   Internal utility functions
 **********************************************)
 
(* These are the default choices for "Date", "Time", and "DateTime" 
   If a locale as a specific different default date/time format include it here.
*)

convertDateStringForms[d_, l_String?(StringMatchQ[#, "de*", IgnoreCase->True]&), "Date"] := 
  {convertDateStringForms[d,l,"Day"], ".", convertDateStringForms[d,l,"Month"], ".", convertDateStringForms[d,l,"Year"]};
  
convertDateStringForms[d_, l_, "Date"] := 
  {convertDateStringForms[d,l,"MonthName"], " ", convertDateStringForms[d,l,"ShortDay"], ", ", convertDateStringForms[d,l,"Year"]};
  
convertDateStringForms[d_, l_String?(StringMatchQ[#, "de*", IgnoreCase->True]&), "Time"] := 
  {convertDateStringForms[d,l,"Hour24"], ":", convertDateStringForms[d,l,"Minute"], ":", convertDateStringForms[d,l,"Second"]};
  
convertDateStringForms[d_, l_, "Time"] := 
  {convertDateStringForms[d,l,"ShortHour12"], ":", convertDateStringForms[d,l,"Minute"], ":", convertDateStringForms[d,l,"Second"], 
    " ", convertDateStringForms[d,l,"AMPM"]};
  
convertDateStringForms[d_, l_, "DateTime"] := 
  {convertDateStringForms[d,l,"Date"], " ", convertDateStringForms[d,l,"Time"]};



convertDateStringForms[{y_, __}, l_, "Year"] := ToString[y];
convertDateStringForms[{y_, __}, l_, "ShortYear"] := 
   StringTake[ ToString[PaddedForm[y, 4, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]], -2];
   
convertDateStringForms[{_, m_,__}, l_, "Month"] := ToString[PaddedForm[m, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]];
convertDateStringForms[{_, m_,__}, l_, "ShortMonth"] := ToString[m];

convertDateStringForms[{_,_,d_,__}, l_, "Day"] := ToString[PaddedForm[d, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]];
convertDateStringForms[{_,_,d_,__}, l_, "ShortDay"] := ToString[d];

(* We need to decide how Hour decides to match either Hour24 or Hour12 *)
convertDateStringForms[d_, l_, "Hour"] := convertDateStringForms[d,l,"Hour24"];
convertDateStringForms[d_, l_, "ShortHour"] := convertDateStringForms[d,l,"ShortHour24"];

convertDateStringForms[{_,_,_,h_,__}, l_, "Hour12"] := Block[{hh = Mod[h,12]},
		If[ hh == 0, hh = 12];
		ToString[PaddedForm[hh, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]]
    ];
convertDateStringForms[{_,_,_,h_,__}, l_, "ShortHour12"] := Block[{hh = Mod[h,12]},
		If[ hh == 0, hh = 12];
		ToString[hh]
    ];
convertDateStringForms[{_,_,_,h_,__}, l_, "Hour24"] := ToString[PaddedForm[h, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]];
convertDateStringForms[{_,_,_,h_,__}, l_, "ShortHour24"] := ToString[h];

convertDateStringForms[{__,m_,_}, l_, "Minute"] := ToString[PaddedForm[m, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]];
convertDateStringForms[{__,m_,_}, l_, "ShortMinute"] := ToString[m];
convertDateStringForms[{__,s_}, l_, "Second"] := ToString[PaddedForm[Floor[s], 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]];

convertDateStringForms[{__,s_}, l_, "Millisecond"] := 
  ToString[PaddedForm[ Floor[FractionalPart[s] 1000], 3, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]];
convertDateStringForms[{__,s_}, l_, "ShortMillisecond"] := 
  ToString[Floor[FractionalPart[s] 1000]];

convertDateStringForms[{y_, m_, d_, h_, mi_, s_}, l_, "TimeZone"] := JavaBlock[
  Module[{off,tz, hh, mm},
     InstallJava[];
     LoadJavaClass["java.util.TimeZone"];
     tz = TimeZone`getDefault[];
     off = tz @ getOffset[1, y, m-1,d, 1+Miscellaneous`Calendar`Private`DayOfWeekNumber[{y,m,d}], 
       Floor[(h 3600 + m 60 + s) 1000]];
     hh = off/(1000 60 60.);
     mm = Abs[Floor[FractionalPart[hh] 60]];
     hh = IntegerPart[hh];
     StringJoin["GMT", 
        ToString[PaddedForm[hh, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"-", "+"}, SignPadding -> True]], ":",
        ToString[PaddedForm[mm, 2, NumberPadding -> {"0", "0"}, NumberSigns -> {"", ""}]]
        ]
     ]
  ];
  
  
computeQuarter[m_] := Quotient[m, 3, 1] + 1;

convertDateStringForms[{_, m_,__}, l_, "Quarter"] := ToString[computeQuarter[m]];

(* With the use of the word Quarter and Q here we may want to add locale specific alternate formats for the form below *)

convertDateStringForms[{_, m_,__}, l_, "QuarterName"] := StringJoin["Quarter ", ToString[computeQuarter[m]]];
convertDateStringForms[{_, m_,__}, l_, "ShortQuarterName"] := StringJoin["Q",  ToString[computeQuarter[m]]];


(**********************************************
   These need checks for localized values
 **********************************************)

convertDateStringForms[{_, m_,__}, l_, "MonthName"] := 
  getCachedMonths[l][[Mod[m,12,1]]];
convertDateStringForms[{_, m_,__}, l_, "ShortMonthName"] := 
  getCachedShortMonths[l][[Mod[m,12,1]]];
convertDateStringForms[d_, l_, "MonthNameInitial"] := StringTake[ convertDateStringForms[d,l,"MonthName"], 1];

convertDateStringForms[{y_,m_,d_,__}, l_, "DayName"] := Block[{dw = 1+Miscellaneous`Calendar`Private`DayOfWeekNumber[{y,m,d}]},
   getCachedWeekdays[l][[dw]]
   ];
convertDateStringForms[{y_,m_,d_,__}, l_, "ShortDayName"] := Block[{dw = 1+Miscellaneous`Calendar`Private`DayOfWeekNumber[{y,m,d}]},
   getCachedShortWeekdays[l][[dw]]
   ];
convertDateStringForms[d_, l_, "DayNameInitial"] := StringTake[ convertDateStringForms[d,l,"DayName"], 1];

convertDateStringForms[{_,_,_,h_,__}, l_, "AMPM"] := Block[{ap = getCachedAMPM[l]},
  If[h < 12, First[ap], Last[ap]]
  ];
  
(* Name needs to check if date uses daylight savings time in locale and whether it is active to get name right 

   For TimeZone,  get the default TimeZone and call
    getDisplayName(boolean daylight, TimeZone`LONG | TimeZone`SHORT, Locale locale)
    by determining if date is in inDaylightTime(Date d)
    do we need to call useDaylightTime() first to see or will in always do this?
*)
convertDateStringForms[{y_, m_, d_, h_, mi_, s_}, l_, "TimeZoneName"] := JavaBlock[
  Module[{loc,c,tz},
     InstallJava[];
     loc = JavaNew["java.util.Locale", l];
     c = JavaNew["java.util.GregorianCalendar", y, m-1,d,h,mi,Floor[s]];
     LoadJavaClass["java.util.TimeZone"];
     tz = TimeZone`getDefault[];
     tz @ getDisplayName[ tz @ inDaylightTime[c @ getTime[]], TimeZone`LONG, loc]
     ]
  ];
convertDateStringForms[{y_, m_, d_, h_, mi_, s_}, l_, "ShortTimeZoneName"] := JavaBlock[
  Module[{loc,c,tz},
     InstallJava[];
     loc = JavaNew["java.util.Locale", l];
     c = JavaNew["java.util.GregorianCalendar", y, m-1,d,h,mi,Floor[s]];
     LoadJavaClass["java.util.TimeZone"];
     tz = TimeZone`getDefault[];
     tz @ getDisplayName[ tz @ inDaylightTime[c @ getTime[]], TimeZone`SHORT, loc]
     ]
  ];
 

convertDateStringForms[d_, l_, e_] := e;
 
 
 
findRefreshTime["Date"] := findRefreshTime["Day"];
findRefreshTime["Time"] := findRefreshTime["Second"];
findRefreshTime["DateTime"] := findRefreshTime["Second"];
findRefreshTime["Year" | "ShortYear"] = 60 60 24 28 12;
findRefreshTime["MonthName" | "ShortMonthName" | "Month" | "ShortMonth"] = 60 60 24 28;
findRefreshTime["DayName" | "ShortDayName" | "Day" | "ShortDay"] = 60 60 24;
findRefreshTime["AMPM" | "Hour" | "Hour12" | "Hour24"] = 60 60;
findRefreshTime["Minute" | "ShortMinute"] = 60;
findRefreshTime["Second"] = 1;
findRefreshTime[e___] := {};


dateListPatternQ[{year_Integer, month_Integer, day_Integer, hour_Integer, min_Integer, sec_}] := True;
dateListPatternQ[___] := False;

(* Localized values*)

getCachedMonths[l_] := Block[{r = cachedMonths[l]},
   If[ Head[r] === cachedMonths, 
      loadLocaleData[l];
      r = cachedMonths[l];
      ];
   r
   ];
getCachedShortMonths[l_] := Block[{r = cachedShortMonths[l]},
   If[ Head[r] === cachedShortMonths, 
      loadLocaleData[l];
      r = cachedShortMonths[l];
      ];
   r
   ];
getCachedWeekdays[l_] := Block[{r = cachedWeekdays[l]},
   If[ Head[r] === cachedWeekdays, 
      loadLocaleData[l];
      r = cachedWeekdays[l];
      ];
   r
   ];
getCachedShortWeekdays[l_] := Block[{r = cachedShortWeekdays[l]},
   If[ Head[r] === cachedShortWeekdays, 
      loadLocaleData[l];
      r = cachedShortWeekdays[l];
      ];
   r
   ];
getCachedAMPM[l_] := Block[{r = cachedAMPM[l]},
   If[ Head[r] === cachedAMPM, 
      loadLocaleData[l];
      r = cachedAMPM[l];
      ];
   r
   ];
   
cachedMonths["en"] = 
 {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
cachedShortMonths["en"] = 
 {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
cachedWeekdays["en"] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
cachedShortWeekdays["en"] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
cachedAMPM["en"] = {"AM", "PM"};

(* When loading from Java, construct a java.util.Locale and if found
   create a java.text.DateFormatSymbols and ask for
   getMonths[], getShortMonths[], getWeekdays[] (+1 offset), getShortWeekdays[] (+1 offset), getAmPmStrings[]

*)
loadLocaleData[l_String] := JavaBlock[
  Module[{loc, ds},
    InstallJava[];
    loc = JavaNew["java.util.Locale", l];
    ds = JavaNew["java.text.DateFormatSymbols", loc];
    cachedMonths[l] = ds @ getMonths[];
    cachedShortMonths[l] = ds @ getShortMonths[];
    cachedWeekdays[l] = Drop[ds @ getWeekdays[], 1];
    cachedShortWeekdays[l] = Drop[ds @ getShortWeekdays[], 1];
    cachedAMPM[l] = ds @ getAmPmStrings[];
    ]
  ];

toLocale[{ls__}] := First[toLocale /@ {ls}];

(SetDelayed[toLocale[#1[[2]]], Evaluate[#1[[1]]]]&) /@ 
{
  {"aa","Afar"}, {"ab","Abkhazian"}, {"af","Afrikaans"}, {"am","Amharic"}, {"ar","Arabic"}, 
  {"as","Assamese"}, {"ay","Aymara"}, {"az","Azerbaijani"}, {"ba","Bashkir"}, {"be","Byelorussian"}, 
  {"bg","Bulgarian"}, {"bh","Bihari"}, {"bi","Bislama"}, {"bn","Bengali"}, {"bn","Bangla"}, 
  {"bo","Tibetan"}, {"br","Breton"}, {"ca","Catalan"}, {"co","Corsican"}, {"cs","Czech"}, 
  {"cy","Welsh"}, {"da","Danish"}, {"de","German"}, {"dz","Bhutani"}, {"el","Greek"}, 
  {"en","English"}, {"eo","Esperanto"}, {"es","Spanish"}, {"et","Estonian"}, 
  {"eu","Basque"}, {"fa","Persian"}, {"fi","Finnish"}, {"fj","Fiji"}, 
  {"fo","Faroese"}, {"fr","French"}, {"fy","Frisian"}, {"ga","Irish"}, 
  {"gd","Scots Gaelic"}, {"gl","Galician"}, {"gn","Guarani"}, {"gu","Gujarati"}, 
  {"ha","Hausa"}, {"he","Hebrew"}, {"hi","Hindi"}, {"hr","Croatian"}, {"hu","Hungarian"}, 
  {"hy","Armenian"}, {"ia","Interlingua"}, {"id","Indonesian"}, {"ie","Interlingue"}, 
  {"ik","Inupiak"}, {"is","Icelandic"}, {"it","Italian"}, {"iu","Inuktitut"}, {"ja","Japanese"}, 
  {"jw","Javanese"}, {"ka","Georgian"}, {"kk","Kazakh"}, {"kl","Greenlandic"}, 
  {"km","Cambodian"}, {"kn","Kannada"}, {"ko","Korean"}, {"ks","Kashmiri"}, 
  {"ku","Kurdish"}, {"ky","Kirghiz"}, {"la","Latin"}, {"ln","Lingala"}, 
  {"lo","Laothian"}, {"lt","Lithuanian"}, {"lv","Lettish"}, {"lv","Latvian"}, 
  {"mg","Malagasy"}, {"mi","Maori"}, {"mk","Macedonian"}, {"ml","Malayalam"}, 
  {"mn","Mongolian"}, {"mo","Moldavian"}, {"mr","Marathi"}, {"ms","Malay"}, 
  {"mt","Maltese"}, {"my","Burmese"}, {"na","Nauru"}, {"ne","Nepali"}, {"nl","Dutch"}, 
  {"no","Norwegian"}, {"oc","Occitan"}, {"om","Oromo"}, {"om","Afan"}, {"or","Oriya"}, 
  {"pa","Punjabi"}, {"pl","Polish"}, {"ps","Pushto"}, {"ps","Pashto"}, {"pt","Portuguese"}, 
  {"qu","Quechua"}, {"rm","Rhaeto-Romance"}, {"rn","Kirundi"}, {"ro","Romanian"}, 
  {"ru","Russian"}, {"rw","Kinyarwanda"}, {"sa","Sanskrit"}, {"sd","Sindhi"}, 
  {"sg","Sangho"}, {"sh","Serbo-Croatian"}, {"si","Sinhalese"}, {"sk","Slovak"}, 
  {"sl","Slovenian"}, {"sm","Samoan"}, {"sn","Shona"}, {"so","Somali"}, {"sq","Albanian"}, 
  {"sr","Serbian"}, {"ss","Siswati"}, {"st","Sesotho"}, {"su","Sundanese"}, {"sv","Swedish"}, 
  {"sw","Swahili"}, {"ta","Tamil"}, {"te","Telugu"}, {"tg","Tajik"}, {"th","Thai"}, 
  {"ti","Tigrinya"}, {"tk","Turkmen"}, {"tl","Tagalog"}, {"tn","Setswana"}, {"to","Tonga"}, 
  {"tr","Turkish"}, {"ts","Tsonga"}, {"tt","Tatar"}, {"tw","Twi"}, {"ug","Uighur"}, 
  {"uk","Ukrainian"}, {"ur","Urdu"}, {"uz","Uzbek"}, {"vi","Vietnamese"}, {"vo","Volapuk"}, 
  {"wo","Wolof"}, {"xh","Xhosa"}, {"yi","Yiddish"}, {"yo","Yoruba"}, {"za","Zhuang"}, 
  {"zh","Chinese"}, {"zu","Zulu"}
};

toLocale[s_String] := s;
toLocale[___] := "en";

If[ !ValueQ[$DateStringLocale],
  $DateStringLocale = toLocale[$Language];
  ];


(************************************************************************)
End[]  
(************************************************************************)

(************************************************************************)
EndPackage[]
(************************************************************************)


