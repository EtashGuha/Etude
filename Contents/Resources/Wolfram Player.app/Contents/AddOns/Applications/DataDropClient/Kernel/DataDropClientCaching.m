(* Mathematica Package *)


BeginPackage["DataDropClient`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

$CacheDatabinData=True;
$CacheDataDropReferences=True;

ddCache=Association[];

addToCacheData[id_, req_,res_]:=With[{paging=Lookup[res,"Paging",Association[]]},
    ddCache[{id,req,"Data"}]=Lookup[res,"Drops",Missing[]];
    ddCache[{id,req,"StartIndex"}]=Lookup[paging,"Start",Missing[]];
    ddCache[{id,req,"EndIndex"}]=Lookup[paging,"End",Missing[]];
]

importnewEntries[req_, res_, as_]:=
    Which[req==="Read",
        MapAt[importfullentries,res,"Drops"],
        req==="Recent",
        importfullentries@res
        ,
        TrueQ[Lookup[as,"IncludeTimestamps",False]],
        MapAt[importentries,res,"Drops"],
        True,
        res]

importandaddToCache[req_, res_, id_, as_]:=Module[{imported},
    Which[req==="Read",
        imported=MapAt[importfullentries,res,"Drops"];
        addToCacheData[id, req,imported]
        ,
        TrueQ[Lookup[as,"IncludeTimestamps",False]],
        imported=MapAt[importentries,res,"Drops"];
        addToCacheData[id, req,imported],
        True,
        imported=res];
    imported
]

cacheStart[id_]:=Min[DeleteMissing[{ddCache[{id,"Read","StartIndex"}],ddCache[{id,"Entries","StartIndex"}]}]]
cacheEnd[id_]:=Min[DeleteMissing[{ddCache[{id,"Read","EndIndex"}],ddCache[{id,"Entries","EndIndex"}]}]]

cachedRead[id_, req_, as_]:=cachedRead[id, req, Append[as,"ResponseFormat"->"Text"]]/;$SystemWordLength=!=64&&(!KeyExistsQ[as,"ResponseFormat"])

cachedRead[id_,req_,as_]:=noCacheRead[id,req,as]/;(Lookup[as,"Parameters",All]=!=All)||(!FreeQ[Keys[as], "StartTime"|"EndTime"|"StepSize"])
cachedRead[id_,"Recent",as_]:=With[{res=apifun["Recent",Join[Association[{"Bin"->id}],as]]},
     If[MatchQ[res,{_Association...}],
       importnewEntries["Recent", res, as],
       $Failed
    ]
]
cachedRead[id_,req:Except["Read" | "Entries"],as_]:=apifun[req,Join[Association[{"Bin"->id}],as]]

cachedRead[id_, req_, as_]:=Block[{cache, cachestart, cacheend,newData, res},
    cachestart=ddCache[{id,req,"StartIndex"}];
    cacheend=ddCache[{id,req,"EndIndex"}];
    If[IntegerQ[cachestart]||IntegerQ[cacheend],
        getMergedData[id, req, as, {cachestart, cacheend}],
        res=apifun[req,Join[Association[{"Bin"->id}],as]];
        If[KeyExistsQ[res,"Drops"],
            res=importandaddToCache[req, res, id, as];
            res
            ,
            $Failed
        ]
    ]
]


getMergedData[id_, req_, as_, {cachestart_, cacheend_}]:=Block[{newstart, newend, cacheResultQ=True,cacheImportFunction=Identity,res},
    {newstart, newend}=getStartandEnd[Lookup[as,{"StartIndex","EndIndex","Count"},None],{cachestart, cacheend}];
    If[req==="Entries"&&!TrueQ[Lookup[as,"IncludeTimestamps",False]],
        cacheResultQ=False;
        cacheImportFunction=(Map[First,#]&)
    ];
    
    If[newend==="Recent",
        res=apifun[req,Join[Association[{"Bin"->id}],as]];
        Return@If[KeyExistsQ[res,"Drops"],
            res=importnewEntries[req, res, as];
            checkAppendLatestToCache[id,res,req,{cachestart,cacheend},{cacheResultQ,cacheImportFunction}];
            res
            ,
            $Failed
        ]
        
    ];
    If[cacheend+1>=newstart>=cachestart&&newend>cacheend,
        Return[appendNewData[id, req, Join[KeyDrop[as,"Count"],
            Association["StartIndex"->cacheend+1,"EndIndex"->newend/.Infinity->"Latest"]],{cacheResultQ,cacheImportFunction}, {cachestart,newstart}]]
    ];
    If[cachestart-1<=newend<=cacheend&&newstart<cachestart,
        Return[prependNewData[id, req, Join[KeyDrop[as,"Count"],
            Association["StartIndex"->newstart,"EndIndex"->cachestart-1]],{cacheResultQ,cacheImportFunction},{newend,newstart}]]
    ];
    If[cachestart<=newstart&&newend<=cacheend,
        Return[cacheImportFunction@ddCache[{id,req,"Data"}][[newstart-cachestart+1;;newend-cachestart+1]]]
    ];
    If[cachestart>newstart&&newend>cacheend,
        res=apifun[req,Join[Association[{"Bin"->id}],as]];
        Return@If[KeyExistsQ[res,"Drops"],
            importandaddToCache[req, res, id, as];
            res
            ,
            $Failed
        ]
    ];
    apifun[req,Join[Association[{"Bin"->id}],as]]
]

appendNewData[id_, req_, as_,{cacheResultQ_,cacheImportFunction_}, {cachestart_,newstart_}]:=Block[
    {res, cache, alldata,$CacheDatabinData=False,resultend},
    res=apifun[req,Join[Association[{"Bin"->id}],as]];
    res=importnewEntries[req, res, as];
    cache=cacheImportFunction@ddCache[{id,req,"Data"}];
    resultend=Lookup[res,"Paging",Association[]]["End"];
    If[KeyExistsQ[res,"Drops"],
        alldata=Join[cache, res["Drops"]];
        If[cacheResultQ&&IntegerQ[resultend],
            ddCache[{id,req,"Data"}]=alldata;
            ddCache[{id,req,"EndIndex"}]=resultend;
        ];
        Association["Drops"->alldata[[newstart-cachestart+1;;-1]]]
        ,
        $Failed
    ]
]

prependNewData[id_, req_, as_,{cacheResultQ_,cacheImportFunction_}, {newend_,newstart_}]:=Block[
    {res, cache, alldata,$CacheDatabinData=False, resultstart},
    res=apifun[req,Join[Association[{"Bin"->id}],as]];
    res=importnewEntries[req, res, as];
    cache=cacheImportFunction@ddCache[{id,req,"Data"}];
    resultstart=Lookup[res,"Paging",Association[]]["Start"];
    If[KeyExistsQ[res,"Drops"],
        alldata=Join[res["Drops"],cache];
        If[cacheResultQ&&IntegerQ[resultstart],
            ddCache[{id,req,"Data"}]=alldata;
            ddCache[{id,req,"StartIndex"}]=resultstart;
        ];
        Association["Drops"->alldata[[1;;newend-newstart+1]]]
        ,
        $Failed
    ]
]

checkAppendLatestToCache[id_,res_,req_ {cachestart_,cacheend_},{cacheResultQ_,cacheImportFunction_}]:=Block[{resultend,resultstart,cache,alldata,newdrops},
    resultend=Lookup[res,"Paging",Association[]]["End"];
    resultstart=Lookup[res,"Paging",Association[]]["Start"];
    If[IntegerQ[resultend]&&IntegerQ[resultstart],
        If[resultstart-1<=cacheend<resultend,
            If[resultstart>cachestart,
                cache=ddCache[{id,req,"Data"}];
                cache=cacheImportFunction[cache[[1;;resultstart-1]]];
                alldata=Join[cache, res["Drops"]];
               ,
               alldata=res["Drops"]
            ];
            ddCache[{id,req,"Data"}]=alldata;
            ddCache[{id,req,"EndIndex"}]=resultend;         
        ]
    ]
]/;cacheResultQ

getStartandEnd[{start_,end_,count_}, {cachestart_, cacheend_}]:=Module[{},
    Switch[{start, end, count},
        {None, None, None},(* default *)
        {1, Infinity},
        {_Integer, _Integer,_},
        {start, end},
        {_Integer, _,_Integer},
        {start,start+count-1},
        {_,_Integer,_Integer},
        {Max[{end-count+1,1}], end},
        {_Integer, _,_},
        {start, Infinity},
        {_,_Integer,_},
        {1, end},
        {None,None,_Integer},
        {-count,"Recent"},
        _,
        {1, Infinity}
    ]
    
]

checkClearCache[id_,as_]:=If[cacheStart[id]<=as["Index"]<=cacheEnd[id],
        clearDDCache[id]
    ]/;KeyExistsQ[as,"Index"]

checkClearCache[id_,as_]:=Block[{start=cacheStart[id],end=cacheEnd[id]},
    If[(Lookup[as,"EndIndex",0]<start)||(Lookup[as,"StartIndex",Infinity]>end),
        Null,
        clearDDCache[id]
    ]
]/;KeyExistsQ[as,"StartIndex"]

checkClearCache[id_,_]:=clearDDCache[id]

clearDDCache[]:=ddCache=Association[];
clearDDCache[bin_]:=(KeyDropFrom[ddCache,Join @@ Outer[{bin, #1, #2} &, {"Entries", "Read"}, {"Data", 
   "StartIndex", "EndIndex"}]];Null)


cacheDataDropReference[ref:referencePattern, res_Image]:=(ref=res)/;$CacheDataDropReferences


noCacheRead[id_,req_,as_]:=Block[{res},
    res=apifun[req,Join[Association[{"Bin"->id}],as]];
    importnewEntries[req, res, as]
]

End[] (* End Private Context *)

EndPackage[]
