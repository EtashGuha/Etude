(* Mathematica Package *)

(Unprotect[#]; Clear[#])& /@ {System`Databin}

BeginPackage["DataDropClient`"]
(* Exported symbols added here with SymbolName::usage *)  

System`Databin
DataDropClient`$databinUpValueFunctions

Begin["`Private`"] (* Begin Private Context *) 

(databin_Databin)[requestargs___]:=Catch[datadropExecute[databin,requestargs]]
Databin[db_Databin,rest___]:=Databin[getBinID[db],rest]

(*************** Options/Settings *************)
System`Databin/:HoldPattern[Options][db_Databin]:=Catch[databinsettings[db]]
System`Databin/:HoldPattern[Options][db_Databin, key_]:=Catch[databinsettings[db, key]]

databinsettings[db_Databin, rest___]:=databinsettings[getBinID[db], rest]
databinsettings[id_, rest___]:=Normal@getBinSettings[id,rest]

System`Databin/:HoldPattern[SetOptions[db_Databin, rest___]]:=Catch[setdatabinsettings[db, rest]]

setdatabinsettings[db_Databin, rest___]:=setdatabinsettings[getBinID[db], rest]
setdatabinsettings[id_, as_Association]:=Block[{settings, res},
	settings=validateDatabinParams[as];
	res=apifun["EditBinSettings",Join[Association[{"Bin"->id}],settings]];
	If[Quiet[KeyExistsQ[res,"Settings"]],
		res["Settings"]
		,
		errorcheck[res]
	]
]
setdatabinsettings[id_, opts___?OptionQ]:=setdatabinsettings[id,Association[{opts}]]
setdatabinsettings[___]:=(Message[Databin::optas];$Failed)

(************* UpValues *******************)
databinUpValueFunctionsMap=DeleteDuplicates@Flatten[ReleaseHold[{BarChart, BoxWhiskerChart, DateListPlot, DistributionChart, 
EventSeries, ExponentialMovingAverage, Histogram, ListLinePlot, 
ListPlot, MinimumTimeIncrement, MovingAverage, MovingMap, 
MovingMedian, PieChart, ProbabilityPlot, ProbabilityScalePlot, 
QuantilePlot, RegularlySampledQ, SmoothHistogram, TemporalData, 
TimeSeriesAggregate, TimeSeriesMap, TimeSeriesMapThread, 
TimeSeriesModelFit, TimeSeriesResample, TimeSeriesRescale, 
TimeSeriesShift, TimeSeriesWindow,
RandomProcesses`TemporalDataDump`$DescriptiveStatisticsList,
 RandomProcesses`TemporalDataDump`$ValueFunctionList,
 RandomProcesses`TemporalDataDump`$FittingFunctionList,
 RandomProcesses`TemporalDataDump`$ActsOnStatesList,
 RandomProcesses`TemporalDataDump`$filteringFunctions1,
 RandomProcesses`TemporalDataDump`$filteringFunctions2,
 TimeValue, EffectiveInterest, Cashflow, ListPlay, ListAnimate, 
  SampledSoundList, Interpolation, Standardize, Ratios, Accumulate, FindPeaks,
  Differences, ListCorrelate, ListConvolve}]];
excluded={PearsonCorrelationTest,PillaiTraceTest, SpearmanRankTest, WilksWTest}

databinUpValueFunctionsThread={TimeSeriesThread,
LocationEquivalenceTest, VarianceEquivalenceTest, SiegelTukeyTest, LogRankTest, MannWhitneyTest,
ConoverTest}
	
mapUpValue[f_]:=Databin/:f[before___,db_Databin,after___]:=Map[f[before,#,after]&,db["EventSeries"]]
mapUpValue/@Complement[databinUpValueFunctionsMap,databinUpValueFunctionsThread,excluded]

threadUpValue[f_]:=Databin/:f[before___,db_Databin,after___]:=f[before,Values[db["EventSeries"]],after]
threadUpValue/@databinUpValueFunctionsThread

Databin/:HoldPattern[TimeSeries][db_Databin]:=db["TimeSeries"]
Databin/:HoldPattern[Values][db_Databin]:=db["Values"]
Databin/:HoldPattern[Normal][db_Databin]:=db["Entries"]
Databin/:HoldPattern[Keys][db_Databin]:=db["Keys"]
Databin/:HoldPattern[Get][db_Databin]:=db["FullEntries"]
Databin/:HoldPattern[DeleteFile][db_Databin]:=DataDropClient`DeleteDatabin[db]
Databin/:HoldPattern[System`DeleteObject][db_Databin]:=DataDropClient`DeleteDatabin[db]

databinUpValueFunctionsCustom={Get,Values,Keys,Normal}

Databin/:HoldPattern[Dataset][db_Databin]:=db["Dataset"]

DataDropClient`$databinUpValueFunctions=Join[databinUpValueFunctionsMap,databinUpValueFunctionsThread,databinUpValueFunctionsCustom]

(************ Typesetting *****************)
(* Stolen from Itai's Elided Forms code *)
makeRow[{a_,b_}, fmt_] := RawBoxes @ RowBox[{
	StyleBox[MakeBoxes[a, fmt], FontColor->GrayLevel[0.5]],
	"\[InvisibleSpace]",
	TagBox[MakeBoxes[b, fmt], "SummaryItem", Selectable ->True]
}]

Databin/:
MakeBoxes[databin_Databin, form:StandardForm|TraditionalForm] := (
Catch[With[{id=databinID[databin], icon=databinIcon},
	If[StringQ[id],
		With[{fullid=loaddatabin[id], name=getBinName[id]},
			With[{above=databinAboveFoldTypesetting[fullid,name,databin],
				below=databinBelowFoldTypesetting[fullid,id,name, databin]},
				If[$VersionNumber>=10&&StringQ[fullid],
					BoxForm`ArrangeSummaryBox[
						(* Head *)Databin, 
						(* Interpretation *)databin, 
						(* Icon *)icon, 
						(* Column or Grid *)
						above
						,
						(* Plus Box Column or Grid *)
						below, 
						form]
						,
					ToBoxes[$Failed]
				]
			]
		],
			
		ToBoxes[$Failed]
	]
]])

databinAboveFoldTypesetting[fullid_,None|"Unnamed", databin_]:=databinaboveFoldTypesetting[fullid, databin, limitedbinQ[databin], getShortBinID[fullid]]

databinaboveFoldTypesetting[fullid_, databin_, True, shortid_]:={
	BoxForm`SummaryItem[{"Short ID: ", shortid}],
    BoxForm`SummaryItem[{"Total entry count: ", typesetEntryCount[fullid]}],
    typesetlimits[getBinLimits[databin]]
    }/;validShortIDQ[shortid]

databinaboveFoldTypesetting[fullid_, databin_, True, shortid_]:={
    BoxForm`SummaryItem[{"UUID: ", fullid}],
    BoxForm`SummaryItem[{"Total entry count: ", typesetEntryCount[fullid]}],
    typesetlimits[getBinLimits[databin]]
    }
    
databinaboveFoldTypesetting[fullid_, databin_, False, shortid_]:={
    BoxForm`SummaryItem[{"Short ID: ", shortid}],
    BoxForm`SummaryItem[{"Entry count: ", typesetEntryCount[fullid]}]
    }/;validShortIDQ[shortid]
    
databinaboveFoldTypesetting[fullid_, databin_, False, shortid_]:={
    BoxForm`SummaryItem[{"UUID: ", fullid}],
    BoxForm`SummaryItem[{"Entry count: ", typesetEntryCount[fullid]}]
    }
    
validShortIDQ[str_String]:=TrueQ[StringLength[str]<20]
validShortIDQ[_]:=False
    
databinAboveFoldTypesetting[fullid_,name_, databin_]:=If[limitedbinQ[databin],
    {BoxForm`SummaryItem[{"Name: ", name}],
        BoxForm`SummaryItem[{"Total entry count: ", typesetEntryCount[fullid]}],
        typesetlimits[getBinLimits[databin]]},
    {BoxForm`SummaryItem[{"Name: ", name}],
        BoxForm`SummaryItem[{"Entry count: ", typesetEntryCount[fullid]}]}
]	
	
    
typesetEntryCount[fullid_]:=Dynamic[Lookup[Replace[
    datadropclientcache[{"DatabinStats", fullid}],Except[_Association]->{},{0}],"EntryCount",Missing[]]]
    
    
databinBelowFoldTypesetting[fullid_,id_,name:(None|"Unnamed"), databin_]:=
        {
            With[{shurl=getBinURL[fullid]},
                checksummaryItem["ShortURL: ",Hyperlink[shurl],StringQ[shurl]]
            ],
            BoxForm`SummaryItem[{"UUID: ", fullid}],
            BoxForm`SummaryItem[{"ShortURL: ", Hyperlink[getBinURL[fullid]]}],
            BoxForm`SummaryItem[{"Creator: ", getCreator[id,"Creator"]/.None->Style[None,Gray]}],
            BoxForm`SummaryItem[{"Owner: ", getCreator[id,"Owner"]/.None->Style[None,Gray]}],
            BoxForm`SummaryItem[{"Creation date: ", DateString[getCreationDate[fullid]]}],
            BoxForm`SummaryItem[{"Latest date: ", Dynamic[
                Replace[Lookup[Replace[datadropclientcache[{"DatabinLatest", fullid}],Except[_Association]->{},{0}],"Timestamp",Missing[]],
                    date_DateObject:>DateString[date],{0}]]}], 
            BoxForm`SummaryItem[{If[limitedbinQ[databin],"Total size: ","Size: "], typesetSize[fullid]}],
            BoxForm`SummaryItem[{"Latest: ", Dynamic[
                Lookup[Replace[datadropclientcache[{"DatabinLatest", fullid}],Except[_Association]->{},{0}],"Data",Missing[]]]}],
            Sequence@@(If[#=!=None,{BoxForm`SummaryItem[{"ExpirationDate: ", #}]},{}]&@(
              getExpirationDate[fullid]/.date_DateObject:>DateString[date]))
                
        }

databinBelowFoldTypesetting[fullid_,id_,_, databin_]:=With[{shortid=getShortBinID[fullid]},
	Join[
		If[validShortIDQ[shortid],
		    {
		        BoxForm`SummaryItem[{"Short ID: ", getShortBinID[fullid]}],
		        BoxForm`SummaryItem[{"UUID: ", fullid}],
		        With[{shurl=getBinURL[fullid]},
	                checksummaryItem["ShortURL: ",Hyperlink[shurl],StringQ[shurl]]
	            ]
		    },
		    {
		        BoxForm`SummaryItem[{"UUID: ", fullid}]
		    }
		],        
        {
            BoxForm`SummaryItem[{"Short ID: ", getShortBinID[fullid]}],
            BoxForm`SummaryItem[{"UUID: ", fullid}],
            BoxForm`SummaryItem[{"ShortURL: ", Hyperlink[getBinURL[fullid]]}],
            BoxForm`SummaryItem[{"Creator: ", getCreator[id,"Creator"]/.None->Style[None,Gray]}],
            BoxForm`SummaryItem[{"Owner: ", getCreator[id,"Owner"]/.None->Style[None,Gray]}],
            BoxForm`SummaryItem[{"Creation date: ", DateString[getCreationDate[fullid]]}],
            BoxForm`SummaryItem[{"Latest date: ", Dynamic[
                Replace[Lookup[Replace[datadropclientcache[{"DatabinLatest", fullid}],Except[_Association]->{},{0}],"Timestamp",Missing[]],
                    date_DateObject:>DateString[date],{0}]]}], 
        	BoxForm`SummaryItem[{If[limitedbinQ[databin],"Total size: ","Size: "], typesetSize[fullid]}],
            BoxForm`SummaryItem[{"Latest: ", Dynamic[
                Lookup[Replace[datadropclientcache[{"DatabinLatest", fullid}],Except[_Association]->{},{0}],"Data",Missing[]]]}],
            Sequence@@(If[#=!=None,{BoxForm`SummaryItem[{"ExpirationDate: ", #}]},{}]&@(
              getExpirationDate[fullid]/.date_DateObject:>DateString[date]))
                
        }
	]
]
        
typesetlimits[{All,keys_}]:=typesetkeys[{keys}]
typesetlimits[{range_,All}]:=typesetlimits[{range}]
typesetlimits[{range_,keys_}]:=Sequence[typesetlimits[{range}],typesetkeys[{keys}]]

typesetlimits[{n_Integer}]:=BoxForm`SummaryItem[{"Selection: ", Row[{"first ",n}]," entries"}]/;n>0
typesetlimits[{n_Integer}]:=BoxForm`SummaryItem[{"Selection: ", Row[{"latest ",-n}]," entries"}]
typesetlimits[{{n_Integer, m_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: entries ", n}],BoxForm`SummaryItem[{" to ", m}]}]/;(n>0)&&(m>0)
typesetlimits[{{n_Integer, m_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: entries ", n}],BoxForm`SummaryItem[{" to last ", -m}]}]/;n>0
typesetlimits[{{n_Integer, m_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: last ", -n}],BoxForm`SummaryItem[{" entries to entry ", m}]}]/;m>0
typesetlimits[{{n_Integer, m_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: last ", -n}],BoxForm`SummaryItem[{" to ", -m," entries"}]}]
typesetlimits[{{n_Integer, m_Integer, step_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: entries ", n}],
	BoxForm`SummaryItem[{" to ", m}],BoxForm`SummaryItem[{" step ", step}]}]/;(n>0)&&(m>0)
typesetlimits[{{n_Integer, m_Integer, step_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: entries ", n}],
    BoxForm`SummaryItem[{" to last ", -m}],BoxForm`SummaryItem[{" step ", step}]}]/;(n>0)
typesetlimits[{{n_Integer, m_Integer, step_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: last ", -n}],
    BoxForm`SummaryItem[{" entries to entry ", m}],BoxForm`SummaryItem[{" step ", step}]}]/;(m>0)
typesetlimits[{{n_Integer, m_Integer, step_Integer}}]:=Row[{BoxForm`SummaryItem[{"Selection: last ", -n}],
    BoxForm`SummaryItem[{" to ", -m}],BoxForm`SummaryItem[{" entries step ", step}]}]
typesetlimits[{date_DateObject}]:=BoxForm`SummaryItem[{"Selection: ", Row[{"since ",DateString@date}]}]
typesetlimits[{{date1_DateObject, date2_DateObject}}]:=Row[{BoxForm`SummaryItem[{"Selection: ", DateString@date1}],BoxForm`SummaryItem[{" to ", DateString@date2}]}]
typesetlimits[{dates:{_DateObject..}}]:=With[{sorted=Sort[dates,Less]},
	Row[{BoxForm`SummaryItem[{"Selection: ", DateString@First[sorted]}],BoxForm`SummaryItem[{" to ", DateString@Last[sorted]}]}]]
typesetlimits[{q_Quantity}]:=BoxForm`SummaryItem[{"Selection: ", Row[{"latest ",q}]}]
typesetlimits[{str_String}]:=BoxForm`SummaryItem[{"Selection: ", Row[{"latest ",str}]}]/;validTimeStringQ[str]

typesetlimits[_]:=BoxForm`SummaryItem[{"Range limited", ""}]

typesetkeys[{key:(_String|_Key)}]:=BoxForm`SummaryItem[{"Key: ", key}]
typesetkeys[{keys:{(_String|_Key)..}}]:=BoxForm`SummaryItem[{"Keys: ", keys}]
typesetkeys[_]:=BoxForm`SummaryItem[{"Key limited", ""}]

typesetSize[fullid_]:=Dynamic[If[NumberQ[#],Round[#,0.1],#]&@Lookup[Replace[datadropclientcache[{"DatabinStats", fullid}],Except[_Association]->{},{0}],"Size",Missing[]]]

checksummaryItem[label_,content_,True]:=BoxForm`SummaryItem[{label, content}]
checksummaryItem[___]:=Sequence[]

databinIcon=Graphics[{Thickness[0.05555555555555555], 
  Style[{FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 
        0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 
        0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 
        0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 
        0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1,
         3, 3}}}, {{{15.646999999999998, 5.933000000000001}, {13.585, 
        7.263}, {15.533, 8.458}, {17.332, 
        7.363999999999999}, {15.646999999999998, 
        5.933000000000001}}, {{12.697, 3.425}, {10.245, 
        5.212}, {12.491, 6.591}, {14.725999999999999, 
        5.1499999999999995}, {12.697, 3.425}}, {{8.666, 
        0.}, {5.978000000000001, 2.285}, {8.666, 
        4.244}, {11.354000000000001, 2.285}, {8.666, 0.}}, {{4.635, 
        3.425}, {2.606, 5.1499999999999995}, {4.841, 6.591}, {7.087, 
        5.212}, {4.635, 3.425}}, {{0., 7.363999999999999}, {1.799, 
        8.458}, {3.7470000000000003, 7.263}, {1.6840000000000002, 
        5.933000000000001}, {0., 7.363999999999999}}, {{5.319, 
        10.537999999999998}, {7.096, 10.29}, {8.462000000000002, 
        18.787}, {8.504, 19.044999999999998}, {8.783, 
        19.046}, {8.825000000000001, 
        18.787999999999997}, {10.219999999999999, 10.293}, {12.012, 
        10.537999999999998}, {12.185, 10.537999999999998}, {12.282, 
        10.338}, {12.175, 10.202}, {8.89, 6.2}, {8.758, 
        6.039000000000001}, {8.512, 6.04}, {8.381, 6.202}, {5.156, 
        10.202}, {5.049, 10.338}, {5.146, 10.537999999999998}, {5.319,
         10.537999999999998}}}]}, 
   FaceForm[
    RGBColor[0.44721600000000006, 0.519288, 0.528528, 
     1.]]]}, {Background -> GrayLevel[0.93], Axes -> False, 
  AspectRatio -> 1, 
  ImageSize -> {Automatic, 
    Dynamic[3.5*(CurrentValue["FontCapHeight"]/
        AbsoluteCurrentValue[Magnification])]}, Frame -> True, 
  FrameTicks -> None, PlotRangePadding -> 2.5,
  FrameStyle -> Directive[Thickness[Tiny], GrayLevel[0.55]], 
  ImageSize -> {18., 19.}, PlotRange -> {{0., 18.}, {0., 19.}}, 
  AspectRatio -> Automatic}]

End[] (* End Private Context *)

EndPackage[]


SetAttributes[{Databin},
   {ReadProtected, Protected}
];
