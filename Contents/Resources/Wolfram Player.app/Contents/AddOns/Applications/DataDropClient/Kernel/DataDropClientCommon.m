(* ::Package:: *)

(* Mathematica Package *)

(Unprotect[#]; Clear[#])& /@ {System`Databin,System`Databins}

BeginPackage["DataDropClient`"]
(* Exported symbols added here with SymbolName::usage *)  
System`Databin
System`Databins

Begin["`Private`"] (* Begin Private Context *) 
$loadeddatabins={};
$BinLoadLimit=30;
$UncompressedImageLimit=10^4;
$converttimezones=False;
$ImportDataDropReferences=True;

$readRequests={"Data", "FullEntries","Values","Entries",
    "EventSeries","TimeSeries","Timestamps","TimeInterval","StartTime","EndTime","GeoLocations","Latest","LatestDate","Recent",
    "LatestTimestamp","Dataset"};    
    
$datadropclientversion=With[{info=PacletManager`PacletInformation["DataDropClient"]},
	If[ListQ[info],
	   ("Version" /. info)/."Version"->"0"
	   ,
	   "0"
	]	
]

(**** Utilities ****)
databinID[databin_]:=First[databin]/;Length[databin]>0
databinID[databin_]:=$Failed

getBinID[id_String]:=With[{cached=datadropclientcache[{"DatabinIDs", id}]},
	If[cached=!=$Failed,
		Lookup[cached,"UUID",$Failed],
		getBinID0[id,"UUID"]
	]
]

getBinID[databin_]:=With[{id=databinID[databin]},
	If[StringQ[id],
		getBinID[id]
		,
		$Failed
	]
]

getShortBinID[id_String]:=With[{cached=datadropclientcache[{"DatabinIDs", id}]},
	If[cached=!=$Failed,
		Lookup[cached,"ShortID",None],
		getBinID0[id,"ShortID"]
	]
]

getShortBinID[databin_]:=With[{id=databinID[databin]},
	If[StringQ[id],
		getShortBinID[id]
		,
		$Failed
	]
]

(* Bin Name can change, we should have an update mechanism *)
getBinName[id_String]:=With[{cached=datadropclientcache[{"DatabinIDs", id}]},
	If[cached=!=$Failed,
		Lookup[cached,"Name",Missing[]],
		getBinID0[id,"Name"]
	]
]

getBinName[databin_]:=getBinName[databinID[databin]]

getBinURL[id_String, rest___]:=With[{cached=datadropclientcache[{"DatabinIDs", id}]},
	If[cached=!=$Failed&&{rest}=!={"Long"},
		Lookup[cached,"ShortURL",Missing[]]
		,
		buildbinURL[id,rest]
	]
]

buildbinURL[id_String]:=With[{shortid=getShortBinID[id]},
	If[StringQ[shortid],
		shorturlbase[]<>shortid
		,
		binurlbase[]<>id
	]
]

buildbinURL[id_String,"Long"]:=With[{uuid=getBinID[id]},
	binurlbase[]<>uuid
]

getBinSettings[id_]:=getBinSettings[id,All]
getBinSettings[id_,keys_]:=Block[{res=apifun["BinInfo",Association[{"Bin"->id}]]},
	If[Quiet[KeyExistsQ[res,"Bin"]],
		If[Quiet[KeyExistsQ[res,"CreationDate"]],
			res=MapAt[timezoneconvert,res,"CreationDate"]
		];
		If[Quiet[KeyExistsQ[res,"ExpirationDate"]],
			res=MapAt[timezoneconvertCheck,res,"ExpirationDate"]
		];
		(* res=Normal[res]; *)
		If[keys===All,
			res,
			Options[Normal@res,keys]
		]
		,
		errorcheck[res]
	]
]

getBinFormatting[id_]:=getBinFormatting[id]=With[{settings=getBinSettings[id]},
	If[Quiet[KeyExistsQ[settings,"DataFormatting"]],
		settings["DataFormatting"]
		,
		Automatic
	]
]

(* Token management *)
readauth[_]:=None
writeauth[_]:=None

loaddatabin[id_]:=getBinID[id]/;MemberQ[$loadeddatabins,id]

(* get all the needed values for to represent an existing databin, store them and return the uuid *)
loaddatabin[id_]:=Block[{res=apifun["LoadBin",Association[{"Bin"->id}]],uuid},
	If[Head[res]===Association,
		storebin[id,res]
		,
		errorcheck[res]
	]
]

storebin[as_Association]:=storebin[Quiet[Lookup[Lookup[as,"BinIDs",{}],"UUID",$Failed]],as]
storebin[$Failed,___]:=$Failed
storebin[id_,res_Association]:=Module[{uuid,shortid},
	If[Quiet[KeyExistsQ[res,"BinIDs"]],
		{uuid,shortid}=storeBinID0[id,{"UUID","ShortID"},res["BinIDs"]];
		storeBinStats0[uuid,res["BinStats"]];
		storeRecent[uuid,importentries@res["Recent"]];
		
		$loadeddatabins=Join[$loadeddatabins,{uuid,shortid}];
		uuid
		,
		(* Error handling *)
		errorcheck[res]
	]
]
storebin[$Failed,___]:=$Failed

loaddatabins[{}]:={}
loaddatabins[ids_List]:=Block[{ids1,res},
	ids1=Complement[ids,$loadeddatabins];
	res=apifun["LoadBin",Association[{"Bins"->ids1}]];
	If[ListQ[res],
		storebin/@res,
		errorcheck[res]
	]
]/;Length[ids]<=$BinLoadLimit

loaddatabins[ids_List]:=Join@@(loaddatabins/@Partition[ids,$BinLoadLimit,$BinLoadLimit, 1, {}])

getBinID0[id_, key_]:=Module[{res=apifun["BinIDs",Association[{"Bin"->id}]]},
	If[KeyExistsQ[res,"UUID"],
		storeBinID0[id,key, res],
		error["nobin",id];Throw[$Failed]
	]
]

storeBinID0[id_,key_, res_]:=If[MatchQ[res,_Association|_List],
		If[KeyExistsQ[res,"UUID"],
			datadropclientcache[{"DatabinIDs", id}]=datadropclientcache[{"DatabinIDs", res["UUID"]}]=res/.$Failed->None;
			If[key===All,
				res,
				Lookup[res,key,Missing[]]
			]
			,
			error["nobin",id]
		],
		error["nobin",id]
	]
	
creationDate[_]:=Missing[]
expirationDate[_]:=None

getCreationDate[id_]:=With[{cached=creationDate[id]},
	If[Head[cached]===DateObject,
		cached,
		getCreationDate0[id]
	]
]

getExpirationDate[id_]:=With[{cached=expirationDate[id]},
	If[Head[cached]===DateObject,
		cached,
		getExpirationDate0[id]
	]
]

getRecent[id_]:=With[{res=getRecent0[id]},
	storeRecent[id,res]
]
storeRecent[id_,res_]:=If[ListQ[res],
		If[res==={},
			datadropclientcache[{"DatabinLatest", id}]:={};
			,
			datadropclientcache[{"DatabinLatest", id}]:=Quiet[MapAt[timezoneconvert,Last[res],{"Timestamp"}]];
		],
		datadropclientcache[{"DatabinLatest", id}]:={};
	]

getCreationDate0[id_]:=Block[{new,cached=datadropclientcache[{"DatabinStats", id}]},
	If[Quiet[KeyExistsQ[cached,"CreationDate"]],
		Lookup[cached,"CreationDate",None],
		new=getBinStats0[id];
		If[Quiet[KeyExistsQ[new,"CreationDate"]],
			timezoneconvert[Lookup[new,"CreationDate",None]],
			None
		]
	]
]

getExpirationDate0[id_]:=Block[{new,cached=datadropclientcache[{"DatabinStats", id}]},
	If[Quiet[KeyExistsQ[cached,"ExpirationDate"]],
		Lookup[cached,"ExpirationDate",None],
		None
	]
]

getCreator[id_,role_:"Creator"]:=With[{cached=binroles[id,role]},
	If[Head[cached]===String,
		cached,
		getCreator0[id,role]
	]
]


getCreator0[id_, role_]:=Block[{new,cached=datadropclientcache[{"DatabinStats", id}]},
	If[Quiet[KeyExistsQ[cached,role]],
		Lookup[cached,role,None],
		new=getBinStats0[id];
		If[Quiet[KeyExistsQ[new,role]],
			Lookup[new,role,None],
			None
		]
	]
]

getBinStats0[id_]:=With[{res=apifun["BinStats",Association[{"Bin"->id}]]},
	storeBinStats0[id,res]
]
storeBinStats0[id_,res0_]:=Module[{shortid=getShortBinID[id],uuid=getBinID[id], temp, res=res0},
	If[MatchQ[res,_Association|_List],
		temp=Quiet[MapAt[timezoneconvert,res,{{"LatestTimestamp"},{"CreationDate"}}]];
		If[KeyExistsQ[temp,"ExpirationDate"],temp=Quiet[MapAt[timezoneconvertCheck,temp,"ExpirationDate"]]];
		temp=Quiet[MapAt[Quantity[N[#/1000],"Kilobytes"]&,temp,"Size"]];
		If[Head[temp]=!=MapAt,res=temp];
		datadropclientcache[{"DatabinStats", shortid}]=datadropclientcache[{"DatabinStats", uuid}]=res;
		res
		,
		error["nobin",id]
	]
]

getRecent0[id_]:=With[{res=apifun["Recent",Association[{"Bin"->id}]]},
	If[ListQ[res],
		importfullentries@res,
		error["nobin",id]
	]
	
]

storelatest[_, {}]:=Null
storelatest[id_, res_]:=Block[{new, pos,times},
	times=Quiet[AbsoluteTime/@Lookup[res,"Timestamp",0]];
	times=Replace[times, Except[_?NumberQ] :> 0, {1}];
	pos=First[Ordering[times,-1]];
	If[times[[pos]]>AbsoluteTime[Lookup[datadropclientcache[{"DatabinLatest", id}],"Timestamp",0]/._Missing->0],
		datadropclientcache[{"DatabinLatest", id}]=
			importDataDropReferences[Quiet[MapAt[timezoneconvert,res[[pos]],{"Timestamp"}]]];
	]
]

(************************* User bins ************************)
System`Databins[args___]:=Catch[databins[args]]

databins[]:=(
	ddCloudConnect[];
	If[ddCloudConnectedQ[],
		databins[],
		Message[Databins::cloudc];$Failed
	])/;!ddCloudConnectedQ[]
	
databins[args___]:=(
	ddCloudConnect[];
	If[ddCloudConnectedQ[],
		databins[args],
		Message[Databins::cloudc];$Failed
	])/;!ddCloudConnectedQ[]

databins[]:=getUserBins[]

databins[___]:=$Failed

getUserBins[]:=getuserbins[$WolframID]
getUserBins[as:(_List|_Association)]:=getuserbins[Lookup[as,"WolframID",$WolframID]]
getUserBins[str_String]:=getuserbins[str]
getUserBins[None]:=(Message[Databins::cloudc];$Failed)

getuserbins[user_]:=Block[{res,as,ids, loadeduuids=Select[$loadeddatabins, StringLength[#] > 20 &]},
	res=apifun["GetUserBinsInfo",Association[{"WolframID"->user,"Omit"->loadeduuids}]];
	If[res===Association[],
		Return[{}]
	];
	If[ListQ[res],
		as=res["Bins"];
		ids=Lookup[res,"BinIDs",{{}}];
		ids=Lookup[ids,"UUID",{}];
        storebin/@res;
		Map[memberDatabin[$loadeddatabins,#]&,Join[loadeduuids,ids],{1}]
		,
		If[Quiet[KeyExistsQ[res,"Message"]],
			Message[Databins::apierr, res["Message"]],
			Message[Databins::err1]
		];
		$Failed
	]
]

getUserBins[___]:=$Failed

memberDatabin[list_,str_String]:=Databin[str]/;MemberQ[list,str]
memberDatabin[___]:=Sequence[]

(****************** limited databin ****************)
limitedbinQ[Databin[_,rest___]]:=Switch[{rest},
	{All}|{},False,
	_, True
]
limitedbinQ[_]:=False

endlimitedBinQ[db_Databin]:=False/;!limitedbinQ[db]
endlimitedBinQ[Databin[_,limits___]]:=With[{res=formatbinLimits[limits,"Read"]},
    endlimitedbinQ[res]
]
endlimitedbinQ[as_]:=FreeQ[Keys[as],"EndTime"|"EndIndex"|"Count"]


getBinLimits[Databin[_,limits___]]:={limits}

formatBinLimits[db_Databin, request_, rest_]:=formatbinLimits[getBinLimits[db],request, rest]

formatbinLimits[_,"Add",rest_]:=rest
formatbinLimits[_,"Upload",rest_]:=rest
formatbinLimits[_,"Embed",rest_]:=rest

formatbinLimits[{limits___},Alternatives@@$readRequests,{rest___}]:=Block[{binlimits, requestparams, params},
	binlimits=formatbinlimits[limits];
	requestparams=formatbinlimits[rest];
	params=Merge[{binlimits, requestparams},Identity];
	params=mapAt[Min,params, "Count"];
    params=mapAt[Max,params, "StartIndex"];
    params=mapAt[Min,params, "EndIndex"];
    params=mapAt[LCM@@#&,params, "StepSize"];
    params=mapAt[Max,params, "StartTime"];
    params=mapAt[Max,params, "EndTime"];
    params=mapAt[commonParameters,params, "Parameters"];
	params
]

mapAt[f_,as_,part_]:=If[KeyExistsQ[as,part],MapAt[f,as,part],as]

commonParameters[{key_}]:=key
commonParameters[{first_, rest___}]:=Intersection[Flatten[{first}], Flatten[{rest}]]

formatbinLimits[_,_,rest_]:=rest 

formatbinlimits[range_,keys_]:=Join[formatbinlimits[range],formatbinkeys[keys]]

formatbinlimits[]=Association[];
formatbinlimits[All]=Association[];
formatbinlimits[as_Association]:=as;
formatbinlimits[n_Integer]:=If[n>0,Association["StartIndex"->1,"Count"->n],Association["Count"->-n]]
formatbinlimits[q_Quantity]:=Association["StartTime"->checkquantity[q]]
formatbinlimits[date_DateObject]:=Association["StartTime"->date]
formatbinlimits[l:{_DateObject..}]:=With[{sorted=Sort[l,Less]},
    Association[{"StartTime"->First[sorted], "EndTime"->Last[sorted]}]
]
formatbinlimits[{start_Integer, end_Integer}]:=Association[{"StartIndex"->start, "EndIndex"->end}]
formatbinlimits[{i_Integer}]:=Association[{"StartIndex"->i, "Count"->1}]
formatbinlimits[{start_Integer, end_Integer, step_Integer}]:=Association[{"StartIndex"->start, "EndIndex"->end,"StepSize"->step}]
formatbinlimits[str_String]:=formatbinlimits[Quantity[1,str]]/;validTimeStringQ[str]

formatbinlimits[expr_]:=(Message[Databin::invlimit,expr];Throw[$Failed])
formatbinlimits[___]:=Throw[$Failed]


formatbinkeys[key:(_String|_Key)]:=Association[{"Parameters"->{key}}]
formatbinkeys[keys:{(_String|_Key)..}]:=Association[{"Parameters"->keys}]

formatbinkeys[expr_]:=(Message[Databin::invkey,expr];Throw[$Failed])
formatbinkeys[___]:=Throw[$Failed]

checkquantity[q_]:=Association[{"StartTime"->DatePlus[Now, -q]}]/;UnitDimensions[q]==={{"TimeUnit",1}} 
checkquantity[___]:=Throw[$Failed]

validTimeStringQ["Minute"|"Hour"|"Day"|"Week"|"Month"|"Year"]=True
validTimeStringQ[_]:=False

(************************* Utilities ************************)
preparedata[data_]:=data/.HoldPattern[Rule[a_,b_]]:>Rule[a,ToString[b,InputForm,CharacterEncoding->"UTF8"]]

(******************)
referencePattern=(_DataDropReference`ImageReference)

$importeddatadropreferencecount=0;
importDataDropReferences[as_Association] := Block[{as1=as},
    as1["Drops"]=importDataDropReferences[as1["Drops"]];
    as1
]/;KeyExistsQ[as,"Drops"]

importDataDropReferences[as_Association] := Block[{n},
    (*$importeddatadropreferencecount=0;
    n=Count[as,referencePattern,3];
    If[n===0,Return[as]];
    PrintTemporary[ProgressIndicator[Dynamic[$importeddatadropreferencecount], {0,n}]];*)
    Map[importDataDropReference, as, 3]
]


importDataDropReferences[data_List] := Block[{n},
    $importeddatadropreferencecount=0;
	n=Count[data,referencePattern,3];
	If[n===0,Return[data]];
	PrintTemporary[ProgressIndicator[Dynamic[$importeddatadropreferencecount], {0,n}]];
    Map[importDataDropReference, data, 3]
]

importDataDropReferences[data_] := data
importDataDropReference[ref:referencePattern] :=With[
	{res=CloudImport[CloudObject[$CloudBase<>"/objects/" <> First[ref]]]},
	cacheDataDropReference[ref, res];
	$importeddatadropreferencecount += 1;
	res		
]
  
importDataDropReference[expr_] := expr
 
 
 checkWarnings[as_Association]:=as/;!KeyExistsQ[as,"Warnings"]
 checkWarnings[as_Association]:=(Message[Databin::apiwarn,#]&/@Lookup[as,"Warnings",{}]; as)
 checkWarnings[expr_]:=expr
(****************)


getDates[data_,"Full"] := Join[data[[All, Key["Timestamp"]]],
  data[[All, Key["SourceInformation"], Key["TimeRecorded"]]],
  data[[All, Key["SourceInformation"], Key["TimeGiven"]]]
  ]

getDates[data_,"Small"] := data[[All, Key["Timestamp"]]]/;KeyExistsQ[First[data],"Timestamp"]
getDates[data_,"Small"] := data[[All, 2]]/;Length[First[data]]>0
getDates[data_,"Small"] := {}

setDates[data_, newdates_,"Full"] := Block[{n = Length[data], newdata = data},
  newdata[[All, Key["Timestamp"]]] = newdates[[;; n]];
  newdata[[All, Key["SourceInformation"], Key["TimeRecorded"]]] = newdates[[n + 1 ;; 2 n]];
  newdata[[All, Key["SourceInformation"], Key["TimeGiven"]]] = newdates[[2 n + 1 ;;]];
  newdata
  ]
  
setDates[data_, newdates_,"Small"] := Block[{n = Length[data], newdata = data},
  newdata[[All, Key["Timestamp"]]] = newdates[[;; n]];
  newdata
  ]/;KeyExistsQ[First[data],"Timestamp"]
  
setDates[data_, newdates_,"Small"] := Block[{n = Length[data], newdata = data},
  newdata[[All, 2]] = newdates[[;; n]];
  newdata
  ]/;Length[First[data]]===2  
  
setDates[data_, _,"Small"] := data  
  
convertTimeZones[data_, type_] := Block[{dates},
  dates = getDates[data,type];
  dates = Replace[dates,l_List:>convertTimeZone[l],{1}];
  setDates[data, dates,type]
  ]
  
convertTimeZone[missing_Missing]:=missing
convertTimeZone[gmtList_]:=Block[{tz = $TimeZone, $TimeZone = 0}, AbsoluteTime[gmtList, TimeZone -> tz]]

importfullentries[{}]:={}
importfullentries[drops_]:=convertTimeZones[drops,"Full"]

importentries[{}]:={}
importentries[drops_]:=convertTimeZones[drops,"Small"]
   

timezoneconvertCheck[date_DateObject]:=date/;FreeQ[date,_TimeObject]
timezoneconvertCheck[x_]:=timezoneconvert[x]
timezoneconvert[x:(_Missing|None),___]:=x
timezoneconvert[x_DateObject]:=DateObject[DateAndTime`DateObjectToDateList[x], TimeZone -> $TimeZone]
timezoneconvert[x_List]:=timezoneconvert[DateObject[x, TimeZone -> 0]]
timezoneconvert[x_Integer]:=timezoneconvert[DateObject[x, TimeZone -> 0]]
timezoneconvert[_]:=Missing[]

toDateObjects[data_, type_]:=Block[{dates},
  dates = getDates[data,type];
  dates = dateObject/@dates;
  setDates[data, dates,type]
  ]
  
dateObject[expr_]:=Quiet[Check[DateObject[expr], expr]]
(******************)

makedatabin[shortid_,id_,name_, url_]:=(
	datadropclientcache[{"DatabinIDs", id}]=datadropclientcache[{"DatabinIDs", shortid}]=Association[{"UUID"->id, "ShortID"->shortid,"Name"->name,"ShortURL"->url}];
	datadropclientcache[{"DatabinStats", id}]=datadropclientcache[{"DatabinStats", shortid}]=Association[{
		"ExpirationDate"->Replace[expirationDate[id], Except[_DateObject] :> None, {0}],
		"EntryCount"->0, "Size"->Quantity[0,"Kilobytes"],"LatestTimestamp"->None,"CreationDate"->timezoneconvert[DateObject[]],"Creator"->$WolframID,"Owner"->$WolframID}];
	System`Databin[shortid])

datadropclientcache[{"DatabinStats", _}]:={}
datadropclientcache[{"DatabinLatest", _}]:={}
datadropclientcache[___]:=$Failed


getKeys[as:(_List|_Association)]:=Keys[as]
getKeys[___]:={}

DataDropClient`DatabinQ[args___]:=Catch[databinQ[args]]

databinQ[db_Databin]:=possibleDatabinIDQ[databinID[db]]
databinQ[link_Hyperlink]:=databinq[First[link]]
databinQ[str_String]:=databinq[str]
databinQ[___]:=False

possibleDatabinIDQ[str_String]:=possibleShortIDQ[str]||uuidQ[str]
possibleDatabinIDQ[_]:=False

possibleShortIDQ[str_]:=5<StringLength[str]<14
uuidQ[str_]:=StringLength[str]>35

databinq[str_]:=TrueQ[If[StringFreeQ[str,"://"],
	KeyExistsQ[apifun["BinIDs",Association["Bin"->str]],"UUID"],
	If[StringFreeQ[str,shorturldomain[]],
		StringMatchQ[str,datadropbinform[]],
		StringMatchQ[URLExpand[str],datadropbinform[]]
	]
]]

errorcheck[res_HTTPResponse, name_:None]:=(Message[Databin::invres];Throw[$Failed])
		
errorcheck[res_, name_:None]:=(If[Quiet[KeyExistsQ[res,"Message"]],
			ddclientErrorMessage[res,name],
			Message[Databin::invres]
		];Throw[$Failed])
		
	
ddclientErrorMessage[res_,name_]:=Block[{message, links=Lookup[res, "Hyperlinks",{}], strs, i},
	message=Lookup[res,"Message",""];
	strs=Keys[links];
	links=KeyValueMap[Hyperlink, Association@links];
	message=StringReplace[message,Table[strs[[i]]->"`"<>ToString[i]<>"`",{i,Length[strs]}]];
	With[{head=Switch[name,
		"Create",CreateDatabin,
		"Databins",Databins,
		_,Databin
		]},
		head::apierrl=message;
		Message[head::apierrl,Sequence@@links]
	]
]/;KeyExistsQ[res,"Hyperlinks"]

ddclientErrorMessage[res_,name_]:=Message[Databin::apierr,ddclienterrormessage[Lookup[res,"Message",""], name]]
ddclienterrormessage["The specified API is not available.", name_]:="The specified request "<>ToString[name]<>" is not available for this databin"
ddclienterrormessage[str_,_]:=str

(* Messages *)
Databin::nobin="The specified databin `1` could not be found."
DatabinAdd::nobin="A Databin is expected instead of `1`."
CopyDatabin::nobin="A Databin is expected instead of `1`."
CopyDatabin::limit="The limited databin can not be copied. Try copying the full databin.";
Databin::readcon="Connect to the Wolfram cloud using CloudConnect to read the full databin; returning only a recent sample."
Databin::dashcon="Connect to the Wolfram cloud using CloudConnect to get the databin report."
CreateDatabin::nocr="A Wolfram Data Drop databin could not be created, please try again later."
CreateDatabin::apierr="`1`"
CreateDatabin::apierr2="`1` `2`"
Databin::optas="The options should be given as a list of rules or an Association."
Databin::apierr="`1`"
Databin::apiwarn="`1`";
Databin::asyncf="The asynchronous request to Data Drop was not successful.";
Databins::apierr="`1`"
Databins::err1="The databins service is unavailable. Please try again later."
Databins::cloudc="Connect to the Wolfram cloud using CloudConnect to see your bin ids."
Databin::cloudc="Connect to the Wolfram cloud using CloudConnect to make this request."
CreateDatabin::cloudc="Connect to the Wolfram cloud using CloudConnect to create a databin"
Databin::notav="Wolfram Data Drop is not yet available; check back soon"
Databin::timeout1="The request timed-out. Try to split up the request using the Count and StartIndex parameters."
Databin::timeout2="The request timed-out."
Databin::seriesn="A series can not be created for the databin with less than two data entries."
Databin::invres="A valid response could not be obtained. Please try again later."
Databin::invreq="The request is invalid. See the Databin documentation page for valid request formats."
Databin::empty="The databin has no entries."
Databin::unav="The request `1` is not currently available."
DatabinUpload::uplist="The data entries should be a list or an EventSeries."
DatabinAdd::limit="Data cannot be added to a bin with an end limit. Try using the full databin.";
DatabinUpload::limit="Data cannot be added to a bin with an end limit. Try using the full databin.";
Databin::invlimit="The limit specification `1` is not valid.";
Databin::invkey="The key specification `1` is not valid.";
Databin::invent="An index or entry id was expected instead of `1`.";
DatabinAdd::invgeo="The value given for \"GeoLocation\" in not valid. Try using a GeoPosition."
Databin::ddcb1="The current $DataDropBase can not be authenticated with the current $CloudBase."

(* Results *)
error["nobin",id_]:=(Message[Databin::nobin,id];$Failed)
error["create"]:=(Message[Databin::nocr];$Failed)
error["create", "You have reached the limit for bins without a Wolfram Cloud subscription. Sign up for an account at http://www.wolframcloud.com"]:=(
	Message[CreateDatabin::apierr2,"You have reached the limit for bins without a Wolfram Cloud subscription. Sign up for an account at",Hyperlink["http://www.wolframcloud.com"]];
	$Failed)
error["create", msg_]:=(Message[CreateDatabin::apierr,msg];$Failed)

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{Databins, Databin},
   {ReadProtected, Protected}
];
