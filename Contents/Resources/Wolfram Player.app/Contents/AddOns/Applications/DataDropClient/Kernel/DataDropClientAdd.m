(* Mathematica Package *)

(Unprotect[#]; Clear[#])& /@ {System`DatabinAdd,DataDropClient`DatabinAddAsynchronous,System`DatabinUpload}

BeginPackage["DataDropClient`"]

System`DatabinAdd
DataDropClient`DatabinAddAsynchronous
System`DatabinUpload
$placeholder="DataDropPlaceholderKey";

Begin["`Private`"] (* Begin Private Context *) 

$AsynchDatabinAdd=False;

System`DatabinAdd[first_,rest__]:=Catch[databinAdd[first, rest]]

System`DatabinAdd[bin_Databin][rest___]:=Catch[databinAdd[bin, rest]]

System`DatabinAdd[str_String]:=Block[{bin},
	Check[bin=Databin[str],
        (Message[DatabinAdd::nobin,str];Return[$Failed])
    ];
    DatabinAdd[bin]
]

databinAdd[bin_Databin,rest___]:=datadropAdd[bin,getBinID[bin],rest]
databinAdd[str_String,rest___]:=Block[{bin},
	Check[bin=Databin[str],
		(Message[DatabinAdd::nobin,str];Return[$Failed])
	];
	databinAdd[bin,rest]
]

databinAdd[first_,___]:=(Message[DatabinAdd::nobin,first];$Failed)
databinAdd[___]:=$Failed


datadropAdd[bin_Databin,id_,args_]:=datadropAdd[bin,id,Association[args]]/;MatchQ[args,{_Rule...}]
datadropAdd[bin_Databin,id_,args___]:=datadropAdd[bin,id,Association[{args}]]/;MatchQ[{args},{_Rule...}]

datadropAdd[bin_Databin,id_, as_Association]:=datadropadd[bin,id,formatGeoLocation[as]]
datadropAdd[bin_Databin,id_, value_]:=datadropadd[bin,id,Association[{$placeholder->value}]]

datadropAdd[id_, expr_]:=datadropAdd[Databin[id],id,expr]

datadropAdd[___]:=$Failed

datadropadd[bin_,id_,as_]:=With[{res=datadropadd0[bin,id,addwriteauth[id, as]]},
	storetoken[as, id, "Add",res];
	res
]

(* Add *)
datadropadd0[bin_,id_,as_]:=Block[{res=apifun["Add",Join[Association[{"Bin"->id}],as]]},
	If[TrueQ[datadropaddHandleResponse[id, res, as]],
		bin
		,
		Throw[$Failed]
	]
]


datadropadd0[bin_,id_,as_]:=Block[{
	res=asynchapifun["Add",Join[Association[{"Bin"->id}],as],datadropaddHandleResponse[id, #, as]&]},
	If[Head[res]===AsynchronousTaskObject,
		bin
		,
		Throw[$Failed]
	]
]/;$AsynchDatabinAdd

datadropaddHandleResponse[id_, resp_, as_]:=Block[{info, old},
	If[Quiet[KeyExistsQ[resp,"Data"]],
		info=Lookup[resp,"Information",datadropclientcache[{"DatabinStats", id}]];
		If[MatchQ[info,"None"|None],
			datadropclientcache[{"DatabinStats", id}]=If[AssociationQ[datadropclientcache[{"DatabinStats", id}]],
				Join[datadropclientcache[{"DatabinStats", id}],AssociationMap[Missing["NotAvailable"]&,{"Size","EntryCount","LatestTimestamp"}]],
				Association[]
			]
			,
           	If[Quiet[NumberQ[Lookup[info,"Size"]]],
                info=Normal@MapAt[Quantity[N[#/1000],"Kilobytes"]&,If[ListQ[info],Association,Identity]@info,"Size"]
            ];
            old=datadropclientcache[{"DatabinStats", id}]/.$Failed->{};
		    info=Merge[{old,info},Last];
		    datadropclientcache[{"DatabinStats", id}]=info
		];
		storelatest[id, {datadropaddhandleResponse[resp,as]}];
		True
		,
		Catch[errorcheck[resp,"Add"]]
	]
]

datadropaddhandleResponse[resp_,as_]:=removeplaceholders[Association["Data"->KeyDrop[as,{"Bin","Authorization","Timestamp"}],"Timestamp"->fromString[as["Timestamp"]]]]/;resp["Data"]===Automatic

datadropaddhandleResponse[resp_,_]:=KeyTake[removeplaceholders[resp],{"Data","Timestamp"}]

fromString[str_String]:=ToExpression[str]
fromString[expr_]:=expr

(* utilities *)
addwriteauth[id_,as_]:=Block[{token},
		token=writeauth[id];
		If[token===None||KeyExistsQ[as,"Authorization"],
			as,
			Join[Association[{"Authorization"->token}],as]
		]
]

removeplaceholders[data_,key_:"Data"]:=MapAt[If[Quiet[Keys[#]==={$placeholder}],#[$placeholder],#]&,data,key]

formatGeoLocation[as_]:=as/;!KeyExistsQ[as,"GeoLocation"]

formatGeoLocation[as_]:=Append[as,"GeoLocation"->formatgeoLocation[Lookup[as,"GeoLocation"]]]

formatgeoLocation[str_String]:=str
formatgeoLocation[geo_GeoPosition]:=geo
formatgeoLocation[ent_Entity]:=With[{geo=EntityValue[ent,"Coordinates"]},
	If[Head[geo]==GeoPosition,geo,
		Message[DatabinAdd::invgeo];Throw[$Failed]
	]
]

formatgeoLocation[expr_]:=(Message[DatabinAdd::invgeo];Throw[$Failed])

DataDropClient`DatabinAddAsynchronous[args___]:=Catch[databinAddAsync[args]]

databinAddAsync[bin_Databin,rest___]:=Block[{$AsynchDatabinAdd=True},
	databinAdd[bin, rest]	
]

databinAddAsync[___]:=$Failed

(*********** Upload ************)

System`DatabinUpload[args___]:=Catch[databinUpload[args]]

databinUpload[bin_Databin,rest___]:=databinupload[bin,getBinID[bin], rest]
databinUpload[str_String,rest___]:=Block[{bin},
    Check[bin=Databin[str],
        (Message[DatabinAdd::nobin,first];Return[$Failed])
    ];
    databinUpload[bin,rest]
]

databinUpload[first_,___]:=(Message[DatabinAdd::nobin,first];$Failed)
databinUpload[___]:=$Failed

databinupload[bin_,_,{},___]:=bin
databinupload[bin_,id_,entries_,opts___?OptionQ]:=Block[{res},
	res=databinupload0[id, entries];
	If[Quiet[KeyExistsQ[res,"LastEntry"]],
        cacheuploadresults[id,res];
        bin
        ,
        errorcheck[res,"Upload"]
    ]
]/;MatchQ[entries,(_List|_TemporalData|_EventSeries|_TimeSeries)]

cacheuploadresults[id_,res0_]:=Block[{res=res0, info, old},
	info=Lookup[res,"Information",datadropclientcache[{"DatabinStats", id}]];
    res=removeplaceholders[res,"LastEntry"];
   	If[Quiet[NumberQ[Lookup[info,"Size"]]],
        info=Normal@MapAt[Quantity[N[#/1000],"Kilobytes"]&,If[ListQ[info],Association,Identity]@info,"Size"]
    ];
    old=datadropclientcache[{"DatabinStats", id}]/.$Failed->{};
    info=Merge[{old,info},Last];
    datadropclientcache[{"DatabinStats", id}]=info;
    storelatest[id, {KeyMap[(# /. "LastEntry" -> "Data") &,KeyTake[res,{"LastEntry","Timestamp"}]]}];
]

databinupload[___]:=(Message[DatabinUpload::uplist];Throw[$Failed])

databinupload0[id_, entries_]:=apifun["Upload",Association[{"Bin"->id,"Entries"->entries}]]/;FreeQ[entries,_Image]||ByteCount[entries]<$UncompressedImageLimit

databinupload0[id_, entries_]:=Block[{entries1,rules, res},
	{entries1,rules}=separateimages[entries];
	If[Length[rules]>0,
		apifun["Upload",Association[{"Bin"->id,"Entries"->entries1,
			"DataDropReferences"->StringJoin[Riffle[Last/@rules,","]],
			Sequence@@(Reverse/@rules)
			}]]
		,
		apifun["Upload",Association[{"Bin"->id,"Entries"->entries1}]]
	]
]
	
	
separateimages[entries_]:=Block[{images, n, rules},
	images=DeleteDuplicates[Cases[entries,_Image,Infinity]];
	n=Length[images];
	If[n<1,Return[{entries, {}}]];
	rules=(#->CreateUUID[])&/@images;
	{Replace[entries,rules,Infinity],rules}
]


End[]

EndPackage[]

SetAttributes[{DatabinAdd, DatabinUpload,DataDropClient`DatabinAddAsynchronous},
   {ReadProtected, Protected}
];


