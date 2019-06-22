(* Mathematica Package *)


BeginPackage["DataDropClient`"]
(* Exported symbols added here with SymbolName::usage *)  

$readapi="Read";
Begin["`Private`"] (* Begin Private Context *) 

datadropRecent[args__]:=Block[{$readapi="Recent"},
	datadropRead[args]
]

datadropFullRecords[args__]:=Block[{res=datadropRead[args]},
	formatMetadata[res]
]

datadropLatestDate[id_, arg___]:=With[{res=datadropLatest[id, arg]},
	Replace[Quiet[Lookup[res,"Timestamp",Missing[]]],Except[_DateObject]->Missing[],{0}]
]

datadropLatest[id_, arg___]:=First[Switch[{arg},
	{},datadropRecent[id,Association[{"Count"->1}]],
	{_Association},datadropRecent[id,Join[arg,Association[{"Count"->1}]]],
	{_List},datadropRecent[id,Association[{"Parameters"->arg,"Count"->1}]],
	{_},datadropRecent[id,Association[{"Parameters"->{arg},"Count"->1}]]
]]

readarguments[f1_, f2_]:=(f1[id_, rest___]:=f2[id,formatbinlimits[rest]])

readarguments[datadropRead,datadropread];
readarguments[datadropSeries,datadropseries];
readarguments[datadropTimeSeries,datadroptimeSeries];
readarguments[datadropData,datadropdata];

datadropRead[___]:=$Failed

datadropread[id_, as_]:=With[{res=datadropRecent[id,as],samplesize=5},
	If[Length[res]>=samplesize,
		Message[Databin::readcon]
	];
	res
]/;!ddCloudConnectedQ[]&&$readapi=!="Recent"

datadropread[id_,as_]:=With[{res=datadropread0[id,addreadauth[id, as]]},
	storetoken[as, id, "Read",res];
	res
]

datadropread0[id_,as_]:=Block[{res=cachedRead[id, $readapi,as]},
	If[Quiet[KeyExistsQ[res,"Drops"]],res=Lookup[res,"Drops"]];
	If[res==={},Return[res]];
	If[MatchQ[res,{_Association..}],
		storelatest[id, res];
		toDateObjects[res,"Full"]
		,
		errorcheck[res]
	]
]

datadropentries[id_, as_, req_]:=Block[{res=cachedRead[id, "Entries",Join[Association["IncludePaging"->True],as]]},
	If[Quiet[KeyExistsQ[res,"Drops"]],res=Lookup[res,"Drops"]];
	If[res==={},Return[res]];
	If[ListQ[res],
		entriesResults[res, req]
		,
		errorcheck[res]
	]
]

entriesResults[res_, "Entries"]:=res
entriesResults[res_, "Values"]:=transposeEntries[res]
entriesResults[res:{{_,_}..}, "Timestamps"]:=DateObject[Last[#]]&/@res
entriesResults[res:{{_,_}..}, req:("TimeInterval"|"StartTime"|"EndTime")]:=Block[{times=Last/@res},
	Switch[Length[times],
		1, Switch[req,
			"TimeInterval",DateObject/@times,
			"StartTime",DateObject[times[[1]]],
			"EndTime",DateObject[times[[-1]]]
		],
		0, Missing[],
		_,
		times=Sort[times,Less];
		Switch[req,
			"TimeInterval",DateObject/@times[[{1,-1}]],
			"StartTime",DateObject[times[[1]]],
			"EndTime",DateObject[times[[-1]]]
		]
	]
]

entriesResults[{}, _]:=Missing[]
entriesResults[___]:=(Message[Databin::invres];$Failed)

transposeEntries[{}]:=Association[]
transposeEntries[res:{_Association..}]:=mergeJoin[res]
transposeEntries[res_List]:=res/;FreeQ[res,_Association,{1}]
transposeEntries[res_List]:=transposeEntries[Cases[Replace[res,  x : Except[_Association] :>  Association[{None -> x}], {1}], _Association, {1}]]
transposeEntries[_]:=$Failed

datadropdataset[id_,as_]:=Block[{entries},
	entries=datadropentries[id,Join[as,Association["IncludeTimestamps"->True]],"Entries"];
	If[ListQ[entries],
		Switch[Length[entries],
			0,Dataset[{}],
			_,formatdataset[Transpose[entries],entries]
		]
	]
]

formatdataset[{data:{_Association...},times_},_]:=Block[{set},
	set=Join[KeyUnion[data],Association["Timestamp"->DateObject[#]]&/@times,2];
	Dataset[set]
]

formatdataset[{data_,times_},entries_]:=Block[{},
	Dataset[entries]
]

formatdataset[___]:=(Message[Databin::invres];$Failed)

datadropseries[{id_,f_},as_Association]:=Module[{entries=datadropexecute0[id,"Entries", Join[Association["IncludeTimestamps"->True],as]]},
	If[ListQ[entries],
		formattingfunction[f][entries]
		,
		Missing[]
	]
]

makeseries[drops_, f_]:=Module[{data,times},
	If[Length[drops]<2,
		Message[Databin::seriesn];
		Return[$Failed]		
	];
	data=Lookup[drops,"Data"];
	times=Lookup[drops,"Timestamp"];
	makeseries0[data,times,f]
]

makeseries0[data0_,times_,Automatic]:=Block[{keys, data},
	makeseries1[times,data0, EventSeries]
]/;FreeQ[data0,_Association]

makeseries0[data0_,times_,f_]:=Block[{keys, data},
	data=Cases[Replace[data0,  x : Except[_Association] :>  Association[{None -> x}], {1}], _Association, {1}];
	keys=Union[Flatten[getKeys/@data]];
	data=Transpose[Lookup[data,keys]];
	Association[MapThread[#1->makeseries1[times,#2, f/.{"EventSeries"->EventSeries,
		Automatic->EventSeries,"TimeSeries"->TimeSeries}]&,{keys,data}]]
]

makeseries1[times_,data_, f_]:=With[{series=makeseries2[times,data,f]},
    If[MatchQ[Head[series],TimeSeries|EventSeries|TemporalData],
        series,
        $Failed
    ]
]
makeseries2[times_,data_, f_]:=TemporalData`SetDatesQ[f[data,{times}],True]/;FreeQ[data,_Missing]
makeseries2[times_,data_, f_]:=TemporalData`SetDatesQ[f[DeleteCases[Transpose[{times,data}],{_,_Missing}]],True]
(* Data *)

datadropdata[id_, as_Association]:=Block[{dataformatting, entries,$converttimezones},
	dataformatting=getBinFormatting[id];
	entries=Switch[dataformatting,
		"EventSeries"|"TimeSeries"|Automatic,
		$converttimezones=True;
		datadropexecute0[id,"Entries", Join[Association["IncludeTimestamps"->True],as]]
		,
		"Entries"|"Values",
		$converttimezones=False;
		datadropexecute0[id,"Entries", as]
		,		
		_,
		datadropRead[id, Join[Association[{"Count"->All}],as]]
	];
	If[ListQ[entries],
		formattingfunction[dataformatting][entries]
		,
		(Message[Databin::invres];$Failed)
	]
]

formattingfunction[format_][timevaluepairs_]:=Module[{data, times, res},
	If[timevaluepairs==={},
		Message[Databin::empty];Return[{}]
	];
	{data, times}=Transpose[timevaluepairs];
	If[MatchQ[times,{_DateObject..}],
		times=AbsoluteTime/@times
	];
	res=makeseries0[data, times, format];
	res
]/;MatchQ[format,("EventSeries"|"TimeSeries"|Automatic)]


formattingfunction["Values"][entries_]:=If[entries==={},
		Association[],
		mergeJoin[entries]
	]

formattingfunction[f_Function]:=f

(* utilities *)
addreadauth[id_,as_]:=Block[{token},
		token=readauth[id];
		If[token===None||KeyExistsQ[as,"Authorization"],
			as,
			Join[Association[{"Authorization"->token}],as]
		]
]

formatMetadata[drops_]:=Module[{res},
	res=If[MatchQ[drops,{_Association...}],MapAt[Quantity[N[#/1000],"Kilobytes"]&,drops,{All,"Size"}],drops];
	If[MatchQ[res,{_Association...}],res,drops]
]

mergeJoin[entries_]:=With[{allkeys = DeleteDuplicates[Flatten[Keys /@ entries]]},
	Association[(# -> DeleteMissing[Lookup[entries, #]]) & /@ allkeys]
]

End[] (* End Private Context *)

EndPackage[]
