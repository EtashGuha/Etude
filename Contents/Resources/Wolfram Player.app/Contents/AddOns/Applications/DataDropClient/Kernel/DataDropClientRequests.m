(* Mathematica Package *)


BeginPackage["DataDropClient`"]
(* Exported symbols added here with SymbolName::usage *)  


Begin["`Private`"] (* Begin Private Context *) 

$authenticatedrequests={"Data","Report", "FullEntries","WebForm","Values","Entries",
	"EventSeries","TimeSeries","Timestamps","TimeInterval","StartTime","EndTime","GeoLocations"};
$nonauthenticatedrequests={"Add","Latest","LatestDate","Recent","ShortID",
	"UUID","Name","Creator","CreationDate","Information","Keys","URL","ShortURL", 
	"Permissions", "ExpirationDate", "LatestTimestamp","EntryCount","Size",
	"Administrators","Owner","Class","Interpretation"};
$undocumentedrequests={"Administrator"}

$nonauthenticatedapinames={"BinIDs","BinInfo","BinStats","LoadBin","ExpectedParameters"};

(*** Get api locations and create cloud objects *)
$DataDropClientRequests:=If[ddCloudConnectedQ[],
	Union[$authenticatedrequests,$nonauthenticatedrequests],
	Sort[$nonauthenticatedrequests]
]


datadropExecute[args___]:=Catch[datadropexecute[args]]

datadropexecute[databin_Databin,range_]:=Databin[getBinID[databin],range]/;validRangeQ[range]
datadropexecute[databin_Databin,range_, keys_]:=Databin[getBinID[databin],range, keys]/;validRangeQ[range]&&validKeysQ[keys]

datadropexecute[id_Databin,request_,args_]:=datadropexecute[id,request,Association[args]]/;MatchQ[args,{_Rule...}]
datadropexecute[id_Databin,request_,args___]:=datadropexecute[id,request,Association[{args}]]/;MatchQ[{args},{_Rule...}]

datadropexecute[databin_Databin,request_String,rest___]:=
	datadropexecute[getBinID[databin],request,rest]/;!limitedbinQ[databin]

validRangeQ[range_]:=(MatchQ[range,(All|_Integer|_DateObject|_Quantity|{_Integer..}|{_DateObject..})]||validTimeStringQ[range])
validKeysQ[keys_]:=MatchQ[keys,(All|_String|_Key)|{(_String|_Key)..}]

datadropexecute[databin_Databin,request_String,rest___]:=Block[{id, limits, res},
	id=getBinID[databin];
	limits=formatBinLimits[databin,request, {rest}];
	If[ListQ[limits],
		limits=Sequence@@limits
	];
	res=datadropexecute[id,request,limits];
	formatlimitedbinResponse[{databin,id}, res,request,limits]
]

datadropexecute[id_,"Add",rest___]:=datadropAdd[id,rest];
datadropexecute[id_,"FullEntries",rest___]:=datadropFullRecords[id,rest];
datadropexecute[id_,"GeoLocations",rest___]:=takemetadata[datadropFullRecords[id,rest],"GeoLocation"];
datadropexecute[id_,"Latest",rest___]:=datadropLatest[id,rest];
datadropexecute[id_,"LatestDate"|"LatestTimestamp",rest___]:=datadropLatestDate[id,rest];
datadropexecute[id_,"Recent",rest___]:=datadropRecent[id,rest];
datadropexecute[id_,"Data",rest___]:=datadropData[id,rest];
datadropexecute[id_,f:("EventSeries"|"TimeSeries"),rest___]:=datadropSeries[{id,f},rest];
datadropexecute[id_,"Embed",rest___]:=datadropEmbed[id, rest]


datadropexecute[id_,prop:("Keys"|"Interpretation"),___]:=With[{res=apifun["ExpectedParameters",Join[Association[{"Bin"->id}],Association[]]]},
	If[ListQ[res],
		If[prop==="Keys",
			First/@res,
			res
		],
		errorcheck[res]			
	]
]


datadropexecute[id_,"Information",as_Association]:=Join[getBinSettings[id],datadropexecute[id,"BinStats",as]];

datadropexecute[id_,request_,as_Association]:=datadropexecute1[id,request,as]

datadropexecute[___]:=(Message[Databin::invreq];$Failed)


datadropexecute1[id_,"WebForm",as_]:=With[{res=datadropexecute0[id, "WebForm",as]},
	If[KeyExistsQ[res,"URL"],
		CloudObject["URL"/.res],
		Message[Databin::invres];
		$Failed
	]
]


(* Local requests *)
datadropexecute1[id_,"UUID",_]:=(id)
datadropexecute1[id_,"ShortID",_]:=getShortBinID[id]
datadropexecute1[id_,"Name",_]:=getBinName[id]
datadropexecute1[id_,"CreationDate",_]:=getCreationDate[id]
datadropexecute1[id_,"Requests",_]:=$DataDropClientRequests
datadropexecute1[id_,"Creator",_]:=getCreator[id]
datadropexecute1[id_,"Owner",_]:=getCreator[id,"Owner"]
datadropexecute1[id_,"ShortURL",_]:=With[{url=getBinURL[id]},
	If[StringQ[url],Hyperlink[url],url]]
datadropexecute1[id_,"URL",_]:=With[{url=getBinURL[id,"Long"]},
    If[StringQ[url],Hyperlink[url],url]]
datadropexecute1[id_,"ClearCache",_]:=clearDDCache[id]


datadropexecute1[id_,"Administrator",rest___]:=datadropexecute1[id,"Administrators",rest]
datadropexecute1[id_,Permissions,rest___]:=datadropexecute1[id,"Permissions",rest]
datadropexecute1[id_,"Permissions",rest___]:=With[{info=datadropexecute[id,"Information",Association[{}]]},
	KeyTake[info,Permissions]
]

inforequests={"Class","Administrators"}
datadropexecute1[id_,prop:(Alternatives@@inforequests),_]:=datadropinforequest[id, prop]

statsrequests={"ExpirationDate", "Size", "EntryCount","LatestTimestamp"}
datadropexecute1[id_,prop:(Alternatives@@statsrequests),_]:=datadropdstatsrequest[id, prop]

datadropinforequest[id_,prop_]:=With[{info=datadropexecute[id,"BinInfo",Association[{}]]},
	Lookup[info,prop,default[prop]]
]

datadropdstatsrequest[id_,prop_]:=With[{info=datadropexecute[id,"BinStats",Association[{}]]},
	Lookup[info,prop,default[prop]]
]

default["Administrators"]={}
default["Size"]=Quantity[0,"Kilobytes"]
default["Class"]=default["ExpirationDate"]=None
default[_]:=Missing[]

datadropexecute1[id_,req_,as_]:=datadropexecuteToken[id, req,addreadauth[id, as]]/;MemberQ[{"Dashboard"},req]
datadropexecute1[id_String,req_,as_]:=datadropexecute0[id, req,as]

datadropexecute1[___]:=(Message[Databin::invreq];$Failed)

datadropexecuteToken[id_,req_,as_]:=With[{res=datadropexecute0[id, req, as]},
	storetoken[as, id, req,res];
	res
]

(* Requests that require special handling *)
datadropexecute0[id_,"BinStats",as_]:=getBinStats0[id]
datadropexecute0[id_,"WebReport",as_]:=datadropclientdashboard[id, as]
datadropexecute0[id_,"Report",as_]:=datadropclientlocaldashboard[id, as]
datadropexecute0[id_,"Entries",as_]:=datadropentries[id, as, "Entries"]
datadropexecute0[id_,req:("Entries"|"Values"),as_]:=datadropentries[id, as, req]
datadropexecute0[id_,req:("Timestamps"|"TimeInterval"|"StartTime"|"EndTime"),as_]:=Block[{$converttimezones=True},
	datadropentries[id, Join[as,Association["IncludeTimestamps"->True]], req]
]
datadropexecute0[id_,"Dataset",as_]:=datadropdataset[id, as]
(* No special handling needed, since only the BinID is required *)
datadropexecute0[id_String,req_,as_]:=Block[{DataDropClient`Private`$DataDropHTTPCodes = 200},
	apifun[req,Join[Association[{"Bin"->id}],as]]
]

datadropexecute0[___]:=(Message[Databin::invreq];$Failed)

storetoken[as_, id_, req_,res_]:=Switch[req,
	"Add",
	writeauth[id]=as["Authorization"],
	"Read"|"Recent"|"WebReport"|"Data"|"FullEntries",
	readauth[id]=as["Authorization"],
	_,Null
]/;KeyExistsQ[as,"Authorization"]&&validresultQ[res]

storetoken[___]:=Null

validresultQ[res_]:=FreeQ[res, "error" | "Error" | $Failed]


datadropclientdashboard[id_, as_]:=(Message[Databin::unav,"WebReport"];$Failed)

datadropclientlocaldashboard[id_, as_]:=If[ddCloudConnectedQ[],
	With[{res=apifun["Dashboard",Join[Association[{"Bin"->id,"Deployed"->False,"EscapeLongCharacters"->True}],as]]},
		If[Head[res]===Notebook,
			NotebookPut[fixLongChars[res]],
			errorcheck[res]
		]
	],
	Message[Databin::dashcon];$Failed
]

takemetadata[{},key_]:={}

takemetadata[entries_,key_]:=takesourcedata[Lookup[entries,"SourceInformation"],key]/;MemberQ[{"TimeRecorded", "TimeGiven", "Authenticated", 
	"GeoLocation", "SourceType", "SourceDetails"},key]

takemetadata[entries_,key_]:=Lookup[entries,key]

takesourcedata[sourceinfo_,key_]:=Lookup[sourceinfo,key]
	
formatlimitedbinResponse[{bin_, id_},res_,"Add"|"Upload",limits_]:=bin

formatlimitedbinResponse[{bin_, id_},res_,"BinStats",limits_]:=res

formatlimitedbinResponse[_,res_,_,_]:=res

spanFromLeftEscape="DDSpanFromLeft";
rightguillmetEscape="DDRightGuillemet";
rightAssociationEscape="DDRightAssociation";
leftAssociationEscape="DDLeftAssociation";
ruleEscape="DDRule";
invSpaceEscape="DDinvSpace";

fixLongChars[nbdata_]:=Replace[nbdata,{
	spanFromLeftEscape -> "\[SpanFromLeft]", 
	ruleEscape -> "\[Rule]", 
 	rightguillmetEscape -> "\"View Databin URL \[RightGuillemet]\"", 
   	leftAssociationEscape -> "\[LeftAssociation]", 
 	rightAssociationEscape -> "\[RightAssociation]", 
 	invSpaceEscape -> "\[InvisibleSpace]"}, Infinity]



End[] (* End Private Context *)

EndPackage[]
