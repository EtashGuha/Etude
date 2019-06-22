(* Mathematica Package *)


(Unprotect[#]; Clear[#])& /@ {System`DatabinRemove}

BeginPackage["DataDropClient`"]
System`DatabinRemove
DataDropClient`DeleteDatabin

Begin["`Private`"] (* Begin Private Context *) 


System`DatabinRemove[args___]:=Catch[deleteEntry[args]]

deleteEntry[bin_Databin,rest___]:=deleteentry[bin,getBinID[bin],rest]
deleteEntry[str_String,rest___]:=Block[{bin},
    Check[bin=Databin[str],
        (Message[DatabinAdd::nobin,str];Return[$Failed])
    ];
    deleteEntry[bin,rest]
]

deleteEntry[first_,___]:=(Message[Databin::nobin,first];$Failed)
deleteEntry[___]:=$Failed


deleteentry[bin_,id_String,span_Span]:=deleteentry0[bin, id, Association["StartIndex"->First[span],"EndIndex"->Last[span]]]
deleteentry[bin_,id_String,{start_Integer,end:(_Integer|All)}]:=deleteentry0[bin, id, Association["StartIndex"->start,"EndIndex"->end]]
deleteentry[bin_,id_String,int_Integer]:=deleteentry0[bin, id, Association["Index"->int]]
deleteentry[bin_,id_String,str_String]:=deleteentry0[bin, id, Association["DataID"->str]]/;StringLength[str]>30

deleteentry0[bin_, id_, as_Association]:=Block[{res},
	res=apifun["DeleteBinData",Join[Association["Bin"->id],as]];
	If[KeyExistsQ[res,"DataID"],
		updatebininfo[id,res];
		checkClearCache[id,as];
		bin
		,
		errorcheck[res]
	]
	
]

updatebininfo[id_,res_]:=Block[{info, old},
    info=Lookup[res,"Information",datadropclientcache[{"DatabinStats", id}]];
    If[Quiet[NumberQ[Lookup[info,"Size"]]],
        info=Normal@MapAt[Quantity[N[#/1000],"Kilobytes"]&,If[ListQ[info],Association,Identity]@info,"Size"]
    ];
    old=datadropclientcache[{"DatabinStats", id}]/.$Failed->{};
    info=Merge[{old,info},Last];
    datadropclientcache[{"DatabinStats", id}]=info;
    (* Check if the latest entry is the one that was deleted *)
]   
    

deleteentry[_,_,rest___]:=(Message[Databin::invent,{rest}];$Failed)

DataDropClient`DeleteDatabin[args___]:=Catch[deleteDatabin[args]]

deleteDatabin[bin_Databin]:=deleteDatabin[getBinID[bin]]

deleteDatabin[id_String]:=Block[{res,DataDropClient`Private`$DataDropHTTPCodes = 200},
	res=apifun["DeleteBin",Join[Association["Bin"->id]]];
	If[KeyExistsQ[res,"Message"],
		clearAllBinCache[id];
		,
		$Failed
	]
]

deleteDatabin[___]:=$Failed

clearAllBinCache[id_]:=Block[{},
	Quiet[
	datadropclientcache[{"DatabinStats", id}]=.;
	datadropclientcache[{"DatabinIDs", id}]=.;
	datadropclientcache[{"DatabinLatest", id}]=.;
	expirationDate[id]=.;
	getCreationDate[id]=.;
	clearDDCache[id];
	$loadeddatabins=DeleteCases[$loadeddatabins, id]
	,
	Unset::norep
	]
]


End[] (* End Private Context *)

EndPackage[]

SetAttributes[{DatabinRemove},
   {ReadProtected, Protected}
];