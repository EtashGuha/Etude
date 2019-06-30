Begin["RunKeeperOAuth`"]

Begin["`Private`"]

(******************************* Fitbit *************************************)

(* Authentication information *)

runkeeperdata[]:=
	If[TrueQ[OAuthClient`Private`$UseChannelFramework],{
		"ServiceName"		-> "RunKeeper",
		"OAuthVersion"		-> "2.0",
		"AuthorizeEndpoint"	-> "https://runkeeper.com/apps/authorize",
		"AccessEndpoint"	-> "https://runkeeper.com/apps/token",
		"RedirectURI"		-> "WolframConnectorChannelListen",
		"Blocking"			-> False,
		"VerifierLabel"		-> "code",
		"ClientInfo"		-> {"Wolfram","Token"}, 
		"AuthenticationDialog" :> "WolframConnectorChannel",
		"RequestFormat"		-> {"Headers","Bearer"},
		"Gets"				-> Join[{"UserID","UserData","FitnessActivity",
			"AnimatedPathMap","Path","PathList","PathMap","PathsMap","PathGoogleMap"},
 									userendpointnames],
		"Posts"				-> {},
		"RawGets"			-> Join[{"RawUserData","RawFitnessActivity","RawBackgroundActivity"},
 									rawuserendpoints],
		"RawPosts"			-> {},
		"Information"		-> "A service for accessing data from a RunKeeper account"
    }
    ,
    {
		"ServiceName"		-> "RunKeeper",
		"OAuthVersion"		-> "2.0",
		"AuthorizeEndpoint"	-> "https://runkeeper.com/apps/authorize",
		"AccessEndpoint"	-> "https://runkeeper.com/apps/token",
		"RedirectURI"		-> "https://www.wolfram.com/oauthlanding?service=RunKeeper",
		"VerifierLabel"		-> "code",
		"ClientInfo"		-> {"Wolfram","Token"},
		"AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "RunKeeper",rkicon]&),
		"RequestFormat"		-> {"Headers","Bearer"},
		"Gets"				-> Join[{"UserID","UserData","FitnessActivity",
			"AnimatedPathMap","Path","PathList","PathMap","PathsMap","PathGoogleMap"},
                                    userendpointnames],
		"Posts"				-> {},
		"RawGets"			-> Join[{"RawUserData","RawFitnessActivity","RawBackgroundActivity"},
                                    rawuserendpoints],
		"RawPosts"			-> {},
		"Information"		-> "A service for accessing data from a RunKeeper account"
    }
]
    
(* a function for importing the raw data - usually json or xml - from the service *)
runkeeperimport[$Failed,___]:=Throw[$Failed]
runkeeperimport[json_String,camelQ_:False,formatting_:{}]:=Block[{res=ImportString[json,"JSON"]},
	If[res===$Failed,Throw[$Failed]];
	If[FreeQ[res,_["errors",_]],
		If[TrueQ[camelQ],res=Replace[(res/.formatting),HoldPattern[Rule[a_String,b_]]:>Rule[camelcase[a],b],Infinity]];
		Switch[res,
			_Rule|{_Rule...},Association@res,
			{{_Rule...}...},Association/@res,
			_,res
		],
		Message[ServiceExecute::apierr,"message"/.("errors"/.res)];
		Throw[$Failed]
	]
]

runkeeperimport[raw_,___]:=raw

(* This is the only documented static url.
	All other urls should be requested from this url *)
baseurl="https://api.runkeeper.com"

(* current and default values, updated whenever possible *)
$updateduserendpoints=False;
{"RawProfile", "RawSettings", "RawFitnessActivities", 
"RawStrengthTrainingActivities", "RawBackgroundActivities", 
"RawSleep", "RawNutrition", "RawWeight", "RawGeneralMeasurements", 
"RawDiabetes", "RawRecords", "RawTeam"}
endpointurl["RawProfile"]="/profile";
endpointurl["RawSettings"]="/settings";
endpointurl["RawFitnessActivities"]="/fitnessActivities";
endpointurl["RawStrengthTrainingActivities"]="/strengthTrainingActivities";
endpointurl["RawBackgroundActivities"]="/backgroundActivities";
endpointurl["RawSleep"]="/sleep";
endpointurl["RawNutrition"]="/nutrition";
endpointurl["RawWeight"]="/weight";
endpointurl["RawGeneralMeasurements"]="/generalMeasurements";
endpointurl["RawDiabetes"]="/diabetes";
endpointurl["RawRecords"]="/records";
endpointurl["RawTeam"]="/team";
endpointurl["RawChangeLog"]="/changeLog";

camelcase[l_List, rest___]:=camelcase[#,rest]&/@l
camelcase[str_String, separators_:{"_"}]:=StringReplace[
 StringReplace[
  StringReplace[str, 
   Thread[separators -> " "]], {WordBoundary ~~ word_ :> 
    ToUpperCase[word]}], {"Id"~~WordBoundary->"ID",WhitespaceCharacter -> "","Url"~~WordBoundary->"URL","Urls"~~WordBoundary->"URLs"}]

userendpoints={"profile", "settings", "fitness_activities", "strength_training_activities", "background_activities", "sleep", 
"nutrition", "weight", "general_measurements", "diabetes", "records" (*,"change_log" *), "team"};
userendpointnames=camelcase/@userendpoints;
rawuserendpoints="Raw"<>#&/@userendpointnames;

(* Raw *)
runkeeperdata["RawUserData"] = {
        "URL"				  	-> baseurl<>"/user",
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		->	runkeeperimport
    }

runkeeperdata[prop:(Alternatives@@rawuserendpoints)]:={
        "URL"				  	-> baseurl<>endpointurl[prop],
        "HTTPSMethod"			-> "GET",
        "Parameters"			-> {"noEarlierThan","noLaterThan","modifiedNoEarlierThan","modifiedNoLaterThan"},
        "ResultsFunction"		->	runkeeperimport
    }
    
runkeeperdata["RawFitnessActivity"] = {
        "URL"				  	-> (ToString@StringForm[baseurl<>endpointurl["RawFitnessActivities"]<>"/`1`", #1]&),
        "PathParameters" 		-> {"ActivityID"},
   		"RequiredParameters"	-> {"ActivityID"},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		->	runkeeperimport
    }
runkeeperdata["RawBackgroundActivity"] = {
        "URL"				  	-> (ToString@StringForm[baseurl<>endpointurl["RawBackgroundActivities"]<>"/`1`", #1]&),
        "PathParameters" 		-> {"ActivityID"},
   		"RequiredParameters"	-> {"ActivityID"},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		->	runkeeperimport
    }
     
runkeeperdata["icon"]=rkicon
   
runkeeperdata[___]:=$Failed
(****** Cooked Properties ******)

setpath[_,Missing|Missing[___]|$Failed]:=Null
setpath[prop:(Alternatives@@rawuserendpoints),url_]:=(endpointurl[prop]=url)

updateendpoints[data_]:=(MapThread[setpath[#1,Lookup[data,#2,Missing[]]]&,{rawuserendpoints,userendpoints}];
	$updateduserendpoints=True)

runkeepercookeddata["UserID",id_,args_]:=Module[
	{rawdata, params,data},
	params=filterparameters[args,getallparameters["RawUserData"]];
	rawdata=OAuthClient`rawoauthdata[id,"RawUserData",params];
	data=runkeeperimport[rawdata];
	If[!$updateduserendpoints,updateendpoints[data]];
	Lookup[data,"userID",Missing["NotAvailable"]]
]  

runkeepercookeddata["UserData",id_,args_]:=Association@FilterRules[Normal[runkeepercookeddata["Profile",id, args]],{"Location","Profile","Birthday","Gender","Name","AthleteType"}]


runkeepercookeddata[prop:(Alternatives@@userendpointnames),id_,args_]:=Module[
	{rawdata,data, params},
	If[!$updateduserendpoints,runkeepercookeddata["UserID",id]];
	params=filterparameters[args,getallparameters["Raw"<>prop]];
	params=fixdates[params];
	rawdata=OAuthClient`rawoauthdata[id,"Raw"<>prop, params];
	data=runkeeperimport[rawdata,True];
	parseuserdata[data,prop]
]  

runkeepercookeddata["FitnessActivity",id_,args_]:=Module[
	{rawdata, params,fields={"Source","Uri","Climb","Type","Duration","TotalDistance","TotalCalories"}},
	params=filterparameters[args,getallparameters["RawFitnessActivity"]];
	If[FreeQ[params,"ActivityID"],Message[ServiceExecute::nparam,"ActivityID"];Throw[$Failed]];
	params=params/.HoldPattern[Rule[a_,b_]]:>Rule[a,ToString[b]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFitnessActivity",params];
	Association@(Thread[fields->(Lookup[
		runkeeperimport[rawdata,True,{fval["start_time"->readDate],quant["duration"->"Seconds"],
			quant["climb"->"Meters"],
			quant["total_distance"->"Meters"],
			fval["uri"->(getID[#]&)]}],
		#,
		Null
	]&/@fields)]/.{"Uri"->"ActivityID",HoldPattern[Rule[_,Null|Missing|Missing[___]]]->Sequence@@{}})
] 
 
getpathdata[id_,args_]:=Block[{params,rawdata,data,path,dists,t0,times},
	
	params=filterparameters[args,getallparameters["RawFitnessActivity"]];
	If[FreeQ[params,"ActivityID"],Message[ServiceExecute::nparam,"ActivityID"];Throw[$Failed]];
	params=params/.HoldPattern[Rule[a_,b_]]:>Rule[a,ToString[b]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFitnessActivity",params];
	data=runkeeperimport[rawdata];
	path=Lookup[data,"path",{}];
	If[path==={},
		Throw[Missing["NotAvailable"]]
	];
	dists={"timestamp","distance"}/.Lookup[data,"distance",{}];
	t0=readDate[Lookup[data,"start_time",0]];
	data={"latitude","longitude","altitude","timestamp"}/.path/."altitude"->0;
	
	times=data[[All,-1]];
	times=t0+(Quantity[#,"Seconds"]&/@times);
	
	data[[All,-1]]=times;
	{data,dists}
]
  
runkeepercookeddata[prop:("Path"|"PathList"|"PathMap"|"AnimatedPathMap"|"PathGoogleMap"),id_,args_]:=Module[
	{data,t, times, dists, mapopts},
	mapopts=FilterRules[args,{"PathMarkers",First /@ Options[Animate]}];
	
	{data,dists}=getpathdata[id,args];
	times=data[[All,-1]];
	
	Switch[prop,
		"Path",
			TimeSeries[Transpose[{times,GeoPosition[Most[#]]&/@data}]],
		"PathList",GeoPosition/@data,
		"PathMap",
			data=Cases[data,{_?NumberQ,_?NumberQ,__}];
			data=SortBy[data,Last];
			t=AbsoluteTime/@data[[All,-1]];
			data=data[[All,1;;2]];
			createMap[data,t,dists,Sequence@@mapopts],
		"AnimatedPathMap"	
			,
			data=Cases[data,{_?NumberQ,_?NumberQ,__}];
			data=SortBy[data,Last];
			t=AbsoluteTime/@data[[All,-1]];
			data=data[[All,1;;2]];
			createAnimatedMap[data,t,dists,Sequence@@mapopts],
		"PathGoogleMap",
			data=Cases[data,{_?NumberQ,_?NumberQ,__}];
			data=SortBy[data,Last];
			googleMapPath[data[[All,1;;2]]]
		]
] 


runkeepercookeddata["PathsMap",id_,args_]:=Block[
	{rawdata, params,data, urls,rawpathdata,pathdata, paths={},OAuthClient`$CacheResults=True,
	dists={},distdata,tdata,t={},mapopts,result},
	mapopts=FilterRules[args,{"PathMarkers"}];
	params=filterparameters[args,getallparameters["RawFitnessActivities"]];
	params=params/.HoldPattern[Rule[a_,b_]]:>Rule[a,ToString[b]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFitnessActivities",params];
	data=runkeeperimport[rawdata];
	urls={"uri","has_path"}/.data["items"];
	urls=First/@Cases[urls,{_,True},{1}];
	urls=getID/@urls;
	(	
		result = Catch[getpathdata[id,{"ActivityID"->#}]];
		If[MatchQ[result,List[_,_]],
		(
			{pathdata,distdata}= result;
			pathdata=Cases[pathdata,{_?NumberQ,_?NumberQ,__}];
			pathdata=SortBy[pathdata,Last];
			tdata=AbsoluteTime/@pathdata[[All,-1]];
			paths=Join[paths,{pathdata[[All,1;;2]]}];
			dists=Join[dists,{distdata}];
			t=Join[t,{tdata}];
		)]
	)&/@urls;
	createMap[paths,t,dists,Sequence@@mapopts]
] 

runkeepercookeddata[args___]:=($Failed)

(* Send Message *)
runkeepersendmessage[___]:=$Failed

(*** Service specific utilites ****)
fval[_[label_,fun_]]:=(Rule[label,value_]:>Rule[label,fun[value]])

quant[l_List]:=quant/@l
quant[_[label_,unit_]]:=(Rule[label,value_]:>Rule[label,Quantity[value, unit]])

filterparameters[given:{(_Rule|_RuleDelayed)...},accepted_,separators_:{"_"}]:=Module[{camel=camelcase[accepted,separators]},
	Cases[given,HoldPattern[Rule|RuleDelayed][Alternatives@@Join[accepted, camel],_],Infinity]/.Thread[camel->accepted]
]
filterparameters[___]:=Throw[$Failed]

getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.runkeeperdata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

makepath[latlongs_] := 
 StringJoin @@ (ToString /@ 
    Flatten[Riffle[Insert[#, ",", 2] & /@ latlongs, "|"]])

googleMapPath[latlongs_] := Block[
  {baseurl = "http://maps.googleapis.com/maps/api/staticmap?", 
   midpoint, center, size, path, url, len, sample = 1},
  midpoint = Mean /@ Transpose[latlongs];
  center = 
   "center=" <> ToString[midpoint[[1]]] <> "," <> 
    ToString[midpoint[[2]]];
  size = "size=500x400";
  path = "path=color:0x0000ff|weight:5|" <> makepath[latlongs];
  len = StringLength[path];
  If[len > 1800,
   sample = Ceiling[len/1800];
   path = 
    "path=color:0x0000ff|weight:5|" <> 
     makepath[latlongs[[1 ;; -1 ;; sample]]];
   ];
  url = StringJoin[baseurl, size, "&", path, "&sensor=true"];
  Import[url]
  ]

llpattern={_?NumberQ,_?NumberQ}

createMap[latlongs:{{llpattern..}..},times0_,dists_,opts___]:=Module[
	{range1, range2, center,lines,markers,colors,i, n=Length[latlongs],size},
	range1=Through@{Min,Max}@latlongs[[All,All,1]];
	range2=Through@{Min,Max}@latlongs[[All,All,2]];
	size=Norm[Subtract @@@ {range1, range1}];
	range1+=.1 size {-1,1};
	range2+=.1 size {-1,1};
	center=Point[{Mean[range2],Mean[range1]}];
	colors=Table[ColorData[1][i],{i,n}];
	markers=
		Flatten[MapThread[getMapMarkers["PathMarkers"/.{opts},#1,#2,#3,
			First[GeoDistance[Sequence @@ Transpose[{range1, range2}]]]/(Max[n,3] * 10^3),#4]&,
			{times0, dists, latlongs,colors}]];

	lines=Flatten[MapIndexed[{colors[[#2[[1]]]],Line[GeoPosition[ #1]]}&,latlongs]];
	
	GeoGraphics[{{Thickness[Large], lines},markers}, 
	 GeoBackground -> GeoStyling["StreetMap"], 
	 GeoRange -> {range1, range2}]
]

createAnimatedMap[latlongs:{llpattern..}, times0_,dists_, opts___]:=Module[
	{range1, range2, bg,line,t,size},
	range1=Through@{Min,Max}@latlongs[[All,1]];
	range2=Through@{Min,Max}@latlongs[[All,2]];
	size=Norm[Subtract @@@ {range1, range1}];
	range1+=.1 size {-1,1};
	range2+=.1 size {-1,1};
	bg=GeoGraphics[{}, 
	 GeoBackground -> GeoStyling["StreetMap"], 
	 GeoRange -> {range1, range2}, GeoRangePadding -> .0025];
	line=latlongs;
	line=GeoGridPosition[GeoPosition[line], "Mercator"][[1]];
	
	With[{background=bg, animationline=line, times=Union[times0]},
		If[Length[times]<2,Throw[$Failed]];
		Animate[
			With[{pos=First[Flatten@Position[times,_?(#>t&),1,1]]},
				Show[{Graphics @@ background, 
					Graphics[{Red,PointSize[Large],Point[animationline[[pos]]],Blue,Line[animationline[[;;pos]]]}]}]
			]
			,
			{{t,times[[1]],"time"},times[[1]],times[[-2]]},opts]
	]
]

createMap[latlongs:{llpattern..},times:{_?NumberQ..},dists:{llpattern..},rest___]:=createMap[{latlongs}, {times},{dists},rest]
createMap[{},___]:={}
createMap[___]:=$Failed

getMapMarkers[type_, times0_, dists_, latlongs_, scale_, color_]:=Module[
	{times, t0, tf, delt, markers, ts, pos, dtimes, ds,labels, targetn=5, maxn=15, minn=1, range},
	Switch[type,
		"Time", 
			times=times0-times0[[1]];
			dtimes=dists[[All,1]];
			t0=dtimes[[1]];
			tf=dtimes[[-1]];
			range=tf-t0;
			delt=First@Nearest[{3600, 600, 60, 10, 1},range/targetn];
			delt=If[maxn>(range/delt)>minn,delt,range/targetn];
			ts=Range[t0,tf,delt];
			pos=Flatten@Table[Position[times,_?(#>=i&),1,1],{i,ts}];
			markers=latlongs[[pos]];
			labels=DateString[#,"Time"]&/@Take[ts,Length[markers]]
			,
		"Distance",
			times=times0-times0[[1]];
			range=dists[[-1,-1]];
			delt=First@Nearest[{5280, 1000, 100, 10, 1},range/targetn];
			delt=If[maxn>(range/delt)>minn,delt,range/targetn];
			ds=Range[0,dists[[-1,-1]],delt];
			pos=Flatten@Table[Position[dists[[All,-1]],_?(#>=i&),1,1],{i,ds}];
			ts=dists[[pos,1]];
			pos=Flatten@Table[Position[times,_?(#>=i&),1,1],{i,ts}];
			markers=latlongs[[pos]];
			labels=Quantity[#,"Feet"]&/@Take[ds,Length[markers]]
		,
		_,markers={};labels={}
	];
	

	MapThread[Tooltip[GeoMarker[#1,"Scale"->scale,"Color"->color],#2]&,{markers,labels}]
	
]

formatDate[date_]:=DateString[date, {"Year", "Month", "Day"}]
readDate[date_,form_:DateObject]:=form[date]
readTime[sec_,form_:TimeObject]:=form[{0,0,sec}]

toRunkeeperDate[date_]:=DateString[date, {"Year", "-", "Month", "-", "Day"}]
fixdates[params_]:=params/.HoldPattern[Rule][a:("noEarlierThan"|"noLaterThan"|"modifiedNoEarlierThan"|"modifiedNoLaterThan"),date_]:>Rule[a,toRunkeeperDate[date]]


getID[uri_]:=With[{split=StringSplit[uri,"/"]},
	ToExpression[Last[split]]
]

(******* parse user data ************)
parseuserdata[data_,"BackgroundActivities"]:=With[{items=data["Items"]},
	(items/.{
		fval["Uri"->getID],
		fval["Timestamp"->readDate]}
		)/."Uri"->"ActivityID"
]
parseuserdata[data_,"FitnessActivities"]:=With[{items=data["Items"]},
	Association/@((items/.{
		fval["Uri"->getID],
		quant["Duration"->"Seconds"],
		quant["TotalDistance"->"Meters"],
		fval["StartTime"->readDate]}
		)/."Uri"->"ActivityID")
]
parseuserdata[data_,"StrengthTrainingActivities"]:=With[{items=data["Items"]},
	Association/@((items/.{
		fval["Uri"->getID],
		quant["Duration"->"Seconds"],
		fval["StartTime"->readDate]}
		)/."Uri"->"ActivityID")
]
parseuserdata[data_,"Sleep"]:=With[{items=data["Items"]},
	(items/.{
		fval["Uri"->getID],
		fval["Timestamp"->readDate],
		quant["TotalSleep"->"Minutes"],
		quant["Rem"->"Minutes"],
		quant["Deep"->"Minutes"],
		quant["Light"->"Minutes"],
		quant["Awake"->"Minutes"]}
		)/.{"Uri"->"SleepID","Rem"->"REM"}
]
parseuserdata[data_,"Nutrition"]:=With[{items=data["Items"]},
	(items/.{
		fval["Uri"->getID],
		fval["Timestamp"->readDate],
		quant["Fat"->"Gram"],
		quant["Calories"->"Gram"],
		quant["Carbohydrates"->"Gram"],
		quant["Fiber"->"Gram"],
		quant["Protein"->"Gram"],
		quant["Sodium"->"Milligram"],
		quant["Water"->"Fluid Ounces"],
		quant["Awake"->"Minutes"]}
		)/.{"Uri"->"NutritionID"}
]
parseuserdata[data_,"GeneralMeasurements"]:=With[{items=data["Items"]},
	(items/.{
		fval["Uri"->getID],
		fval["Timestamp"->readDate],
		quant["Systolic"->"nnHG"],
		quant["Diastolic"->"Gram"],
		quant["TotalCholesterol"->"mg/dL"],
		quant["Hdl"->"mg/dL"],
		quant["Ldl"->"mg/dL"],
		quant["VitaminD"->"ng/dL"]}
		)/.{"Uri"->"MeasurementID","Hdl"->"HDL","Ldl"->"LDL"}
]
parseuserdata[data_,"Diabetes"]:=With[{items=data["Items"]},
	(items/.{
		fval["Uri"->getID],
		fval["Timestamp"->readDate]}
		)/.{"Uri"->"DiabetesID"}
]
parseuserdata[data_,"Team"]:=With[{items=data["Items"]},
	(items/.{fval["Url"->getID]}
		)/."Url"->"TeamID"
]
parseuserdata[data_,"Weight"]:=With[{items=data["Items"]},
	(items/.{
		fval["Uri"->getID],
		quant["Weight"->"Kilograms"],
		fval["Timestamp"->readDate]}
		)/."Uri"->"WeightID"
]
parseuserdata[data_,"Records"]:=Module[{},
	Cases[data,_?(Total["Value"/.Lookup[#,"Stats",{{"Value"->0}}]/."Value"->0]=!=0&)]
]
parseuserdata[data_,_]:=data

getgeolocation[data_]:=If[FreeQ[data,"Latitude"]||FreeQ[data,"Longitude"],data,
	Join[FilterRules[data,Except["Latitude"|"Longitude"|"Altitude"]],{"GeoPosition"->GeoPosition[{"Latitude","Longitude","Altitude"}/.data/."Altitude"->0]}]]

getgeolocations[data_]:=getgeolocation/@data

rkicon=Image[RawArray["Byte", {{{138, 176, 220, 1}, {185, 208, 234, 0}, {183, 206, 233, 1}, {185, 207, 234, 67}, {183, 207, 
  233, 131}, {182, 205, 232, 156}, {182, 205, 232, 155}, {182, 205, 233, 155}, {182, 205, 232, 155}, {182, 205, 232, 
  155}, {181, 205, 232, 155}, {181, 205, 232, 155}, {181, 205, 232, 155}, {181, 205, 232, 155}, {181, 205, 232, 155}, 
  {181, 204, 231, 155}, {180, 204, 231, 155}, {180, 204, 231, 155}, {180, 204, 231, 155}, {180, 204, 231, 155}, {180, 
  204, 231, 155}, {179, 203, 230, 155}, {179, 203, 230, 155}, {179, 203, 230, 155}, {179, 203, 230, 155}, {179, 202, 
  230, 155}, {179, 202, 230, 155}, {179, 202, 229, 127}, {179, 202, 228, 61}, {176, 200, 227, 0}, {185, 207, 229, 0}, 
  {133, 171, 215, 1}}, {{195, 215, 237, 0}, {255, 255, 255, 26}, {255, 255, 255, 193}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 251, 255}, {255, 253, 248, 183}, {255, 255, 
  251, 17}, {186, 206, 229, 1}}, {{182, 206, 233, 5}, {252, 253, 254, 202}, {247, 250, 252, 255}, {255, 255, 255, 
  250}, {229, 236, 241, 251}, {184, 205, 220, 252}, {180, 204, 220, 252}, {180, 204, 220, 252}, {180, 204, 220, 252}, 
  {179, 203, 220, 252}, {178, 203, 220, 252}, {178, 203, 219, 252}, {178, 202, 219, 252}, {177, 201, 219, 252}, {177, 
  201, 219, 252}, {176, 200, 219, 252}, {175, 200, 218, 252}, {175, 199, 218, 252}, {175, 199, 217, 252}, {175, 199, 
  217, 252}, {174, 199, 217, 252}, {173, 198, 217, 252}, {173, 198, 217, 252}, {172, 197, 216, 252}, {172, 197, 216, 
  252}, {170, 195, 216, 252}, {174, 198, 216, 252}, {217, 227, 234, 251}, {253, 251, 250, 251}, {241, 244, 246, 255}, 
  {252, 251, 247, 185}, {174, 198, 224, 2}}, {{183, 206, 233, 79}, {255, 255, 255, 255}, {255, 255, 255, 250}, {198, 
  214, 226, 255}, {108, 156, 190, 255}, {105, 160, 197, 255}, {107, 163, 201, 255}, {106, 162, 201, 255}, {106, 162, 
  202, 255}, {104, 160, 201, 255}, {103, 160, 200, 255}, {102, 160, 200, 255}, {101, 159, 199, 255}, {101, 158, 199, 
  255}, {99, 158, 199, 255}, {100, 156, 199, 255}, {98, 156, 198, 255}, {98, 155, 198, 255}, {96, 153, 196, 255}, 
  {92, 151, 194, 255}, {96, 153, 196, 255}, {94, 152, 196, 255}, {93, 151, 196, 255}, {92, 150, 196, 255}, {91, 150, 
  196, 255}, {91, 150, 195, 255}, {90, 147, 193, 255}, {90, 144, 187, 255}, {179, 201, 219, 255}, {251, 250, 249, 
  250}, {255, 254, 248, 255}, {176, 199, 225, 62}}, {{182, 205, 232, 152}, {255, 255, 255, 253}, {228, 235, 241, 
  251}, {105, 153, 189, 255}, {114, 168, 207, 255}, {123, 178, 217, 255}, {122, 177, 218, 255}, {121, 177, 218, 255}, 
  {121, 177, 218, 255}, {120, 175, 217, 255}, {119, 175, 217, 255}, {118, 174, 216, 255}, {117, 173, 216, 255}, {117, 
  174, 216, 255}, {115, 172, 216, 255}, {115, 171, 215, 255}, {114, 171, 214, 255}, {108, 165, 210, 255}, {113, 166, 
  207, 255}, {123, 172, 208, 255}, {104, 160, 204, 255}, {109, 165, 210, 255}, {110, 165, 212, 255}, {109, 166, 212, 
  255}, {107, 165, 212, 255}, {106, 164, 211, 255}, {106, 162, 210, 255}, {99, 156, 205, 255}, {87, 143, 188, 255}, 
  {213, 223, 232, 251}, {255, 255, 250, 254}, {175, 198, 225, 131}}, {{180, 204, 231, 171}, {255, 255, 255, 255}, 
  {184, 205, 220, 252}, {97, 152, 193, 255}, {120, 175, 215, 255}, {117, 173, 216, 255}, {116, 173, 216, 255}, {116, 
  173, 216, 255}, {114, 172, 215, 255}, {113, 171, 215, 255}, {114, 170, 214, 255}, {112, 170, 215, 255}, {110, 169, 
  214, 255}, {111, 168, 214, 255}, {110, 167, 213, 255}, {110, 168, 213, 255}, {106, 163, 209, 255}, {121, 168, 208, 
  255}, {224, 233, 240, 255}, {245, 248, 249, 255}, {158, 190, 216, 255}, {94, 152, 200, 255}, {104, 161, 209, 255}, 
  {101, 160, 209, 255}, {102, 158, 209, 255}, {100, 158, 208, 255}, {98, 157, 208, 255}, {102, 158, 208, 255}, {82, 
  142, 193, 255}, {162, 189, 213, 251}, {255, 255, 251, 255}, {174, 197, 225, 149}}, {{180, 203, 230, 170}, {255, 
  255, 255, 255}, {173, 197, 217, 252}, {96, 152, 195, 255}, {115, 171, 215, 255}, {112, 170, 214, 255}, {112, 169, 
  214, 255}, {111, 168, 214, 255}, {109, 168, 214, 255}, {109, 167, 213, 255}, {109, 166, 212, 255}, {107, 166, 213, 
  255}, {107, 164, 212, 255}, {107, 164, 210, 255}, {109, 163, 210, 255}, {104, 161, 208, 255}, {81, 142, 193, 255}, 
  {158, 188, 214, 255}, {255, 255, 254, 255}, {255, 255, 255, 255}, {180, 201, 221, 255}, {85, 141, 192, 255}, {101, 
  156, 207, 255}, {97, 155, 208, 255}, {97, 154, 207, 255}, {96, 153, 206, 255}, {95, 152, 206, 255}, {97, 154, 206, 
  255}, {79, 138, 194, 255}, {154, 183, 211, 251}, {255, 255, 250, 255}, {173, 197, 224, 148}}, {{179, 202, 229, 
  170}, {255, 255, 255, 255}, {172, 196, 215, 252}, {92, 147, 193, 255}, {111, 166, 213, 255}, {108, 165, 213, 255}, 
  {108, 166, 212, 255}, {108, 165, 211, 255}, {107, 163, 211, 255}, {106, 162, 211, 255}, {104, 161, 210, 255}, {104, 
  160, 208, 255}, {104, 160, 209, 255}, {100, 157, 205, 255}, {88, 145, 196, 255}, {104, 153, 198, 255}, {139, 174, 
  206, 255}, {183, 202, 219, 255}, {254, 253, 251, 255}, {224, 230, 236, 255}, {98, 142, 181, 255}, {88, 140, 191, 
  255}, {95, 150, 202, 255}, {92, 148, 202, 255}, {92, 149, 203, 255}, {92, 149, 203, 255}, {92, 147, 202, 255}, {93, 
  148, 205, 255}, {75, 132, 192, 255}, {153, 181, 210, 251}, {255, 255, 249, 255}, {172, 196, 223, 148}}, {{179, 202, 
  229, 170}, {255, 255, 255, 255}, {170, 194, 213, 252}, {88, 144, 192, 255}, {108, 163, 212, 255}, {104, 160, 210, 
  255}, {104, 160, 209, 255}, {104, 159, 209, 255}, {102, 159, 208, 255}, {101, 157, 208, 255}, {101, 157, 208, 255}, 
  {100, 156, 206, 255}, {90, 146, 199, 255}, {96, 147, 195, 255}, {159, 187, 213, 255}, {228, 235, 240, 255}, {255, 
  253, 251, 255}, {251, 250, 248, 255}, {249, 248, 247, 255}, {225, 230, 235, 255}, {145, 172, 198, 255}, {90, 138, 
  186, 255}, {84, 137, 192, 255}, {89, 141, 195, 255}, {88, 141, 197, 255}, {87, 142, 200, 255}, {86, 141, 199, 255}, 
  {89, 142, 200, 255}, {71, 127, 189, 255}, {149, 178, 208, 251}, {255, 255, 247, 255}, {172, 195, 223, 148}}, {{178, 
  201, 229, 170}, {255, 255, 255, 255}, {167, 191, 212, 252}, {84, 139, 188, 255}, {104, 158, 208, 255}, {101, 156, 
  208, 255}, {100, 154, 206, 255}, {99, 154, 206, 255}, {99, 154, 205, 255}, {97, 153, 205, 255}, {98, 152, 205, 
  255}, {89, 145, 198, 255}, {124, 164, 204, 255}, {227, 234, 239, 255}, {255, 255, 255, 255}, {251, 250, 249, 255}, 
  {242, 243, 245, 255}, {244, 244, 245, 255}, {244, 244, 244, 255}, {247, 247, 246, 255}, {255, 255, 252, 255}, {109, 
  146, 185, 255}, {154, 179, 206, 255}, {186, 204, 223, 255}, {72, 125, 186, 255}, {85, 138, 196, 255}, {84, 137, 
  196, 255}, {86, 138, 198, 255}, {68, 122, 186, 255}, {149, 175, 206, 251}, {255, 255, 246, 255}, {171, 194, 222, 
  148}}, {{177, 201, 228, 170}, {255, 255, 255, 255}, {164, 189, 211, 252}, {79, 134, 186, 255}, {99, 152, 206, 255}, 
  {96, 150, 204, 255}, {95, 150, 203, 255}, {94, 149, 204, 255}, {94, 148, 203, 255}, {93, 148, 202, 255}, {94, 147, 
  203, 255}, {78, 132, 191, 255}, {193, 209, 227, 255}, {249, 248, 247, 255}, {157, 179, 204, 255}, {110, 143, 179, 
  255}, {223, 228, 234, 255}, {248, 247, 246, 255}, {242, 242, 242, 255}, {238, 239, 240, 255}, {254, 253, 250, 255}, 
  {166, 184, 205, 255}, {207, 215, 223, 255}, {180, 198, 218, 255}, {65, 118, 179, 255}, {81, 132, 193, 255}, {76, 
  129, 193, 255}, {77, 129, 193, 255}, {57, 111, 181, 255}, {140, 168, 203, 251}, {255, 254, 244, 255}, {168, 193, 
  219, 148}}, {{176, 200, 227, 170}, {255, 255, 255, 255}, {163, 186, 209, 252}, {75, 129, 183, 255}, {95, 147, 203, 
  255}, {92, 146, 202, 255}, {92, 145, 201, 255}, {90, 144, 201, 255}, {90, 143, 199, 255}, {89, 142, 200, 255}, {90, 
  142, 200, 255}, {71, 124, 186, 255}, {198, 212, 228, 255}, {171, 190, 211, 255}, {54, 104, 160, 255}, {73, 117, 
  167, 255}, {211, 219, 228, 255}, {249, 248, 246, 255}, {240, 240, 240, 255}, {242, 242, 242, 255}, {223, 227, 231, 
  255}, {243, 243, 242, 255}, {228, 230, 233, 255}, {65, 110, 167, 255}, {46, 101, 171, 255}, {50, 107, 181, 255}, 
  {47, 106, 181, 255}, {48, 107, 180, 255}, {29, 89, 168, 255}, {127, 156, 195, 251}, {255, 252, 241, 255}, {165, 
  190, 218, 148}}, {{176, 199, 226, 170}, {255, 255, 254, 255}, {161, 184, 208, 252}, {72, 124, 180, 255}, {91, 143, 
  199, 255}, {89, 141, 199, 255}, {87, 139, 198, 255}, {87, 138, 198, 255}, {85, 138, 196, 255}, {83, 137, 197, 255}, 
  {85, 137, 195, 255}, {70, 122, 184, 255}, {151, 176, 209, 255}, {187, 202, 219, 255}, {63, 110, 168, 255}, {83, 
  124, 174, 255}, {219, 225, 231, 255}, {242, 242, 241, 255}, {239, 240, 239, 255}, {227, 229, 232, 255}, {87, 117, 
  157, 255}, {242, 241, 238, 255}, {120, 148, 184, 255}, {22, 79, 152, 255}, {45, 99, 172, 255}, {44, 99, 176, 255}, 
  {44, 101, 177, 255}, {48, 103, 179, 255}, {31, 87, 165, 255}, {128, 156, 194, 251}, {255, 250, 240, 255}, {165, 
  189, 218, 148}}, {{175, 198, 226, 170}, {255, 255, 253, 255}, {159, 180, 206, 252}, {68, 118, 179, 255}, {87, 137, 
  195, 255}, {84, 134, 196, 255}, {83, 132, 194, 255}, {82, 132, 193, 255}, {81, 132, 194, 255}, {83, 131, 193, 255}, 
  {84, 133, 193, 255}, {68, 119, 183, 255}, {107, 143, 189, 255}, {234, 236, 238, 255}, {47, 92, 155, 255}, {116, 
  145, 183, 255}, {252, 248, 244, 255}, {228, 231, 233, 255}, {251, 248, 244, 255}, {160, 176, 199, 255}, {23, 70, 
  136, 255}, {111, 142, 183, 255}, {43, 91, 159, 255}, {44, 94, 168, 255}, {46, 98, 174, 255}, {47, 99, 174, 255}, 
  {47, 98, 176, 255}, {50, 101, 176, 255}, {31, 84, 164, 255}, {127, 153, 192, 251}, {255, 249, 240, 255}, {164, 188, 
  217, 148}}, {{174, 198, 225, 170}, {255, 255, 252, 255}, {156, 178, 204, 252}, {64, 113, 174, 255}, {83, 131, 192, 
  255}, {79, 129, 194, 255}, {79, 128, 192, 255}, {79, 128, 191, 255}, {78, 126, 190, 255}, {69, 121, 187, 255}, {61, 
  114, 183, 255}, {41, 97, 172, 255}, {74, 116, 176, 255}, {229, 232, 234, 255}, {79, 115, 165, 255}, {171, 186, 206, 
  255}, {248, 245, 242, 255}, {233, 234, 235, 255}, {227, 229, 231, 255}, {57, 94, 147, 255}, {34, 82, 152, 255}, 
  {34, 84, 161, 255}, {45, 94, 168, 255}, {46, 97, 172, 255}, {46, 96, 172, 255}, {48, 96, 173, 255}, {48, 96, 174, 
  255}, {50, 98, 174, 255}, {31, 83, 162, 255}, {128, 151, 191, 251}, {255, 248, 239, 255}, {163, 187, 217, 148}}, 
  {{174, 196, 224, 170}, {255, 255, 251, 255}, {154, 175, 202, 252}, {60, 107, 171, 255}, {80, 127, 190, 255}, {78, 
  126, 189, 255}, {72, 120, 187, 255}, {61, 113, 183, 255}, {51, 105, 180, 255}, {44, 99, 176, 255}, {43, 98, 175, 
  255}, {45, 98, 173, 255}, {43, 93, 166, 255}, {53, 98, 165, 255}, {36, 81, 148, 255}, {198, 206, 217, 255}, {241, 
  240, 237, 255}, {233, 233, 234, 255}, {221, 224, 227, 255}, {84, 114, 157, 255}, {29, 74, 146, 255}, {49, 93, 167, 
  255}, {47, 93, 170, 255}, {47, 93, 172, 255}, {47, 94, 171, 255}, {47, 94, 172, 255}, {47, 94, 171, 255}, {51, 95, 
  172, 255}, {32, 80, 160, 255}, {126, 150, 190, 251}, {253, 247, 238, 255}, {162, 186, 216, 148}}, {{171, 195, 223, 
  170}, {255, 255, 249, 255}, {152, 172, 201, 252}, {57, 103, 169, 255}, {69, 117, 184, 255}, {56, 106, 178, 255}, 
  {48, 99, 175, 255}, {45, 97, 175, 255}, {44, 96, 176, 255}, {47, 98, 175, 255}, {47, 98, 173, 255}, {48, 97, 172, 
  255}, {47, 95, 170, 255}, {39, 86, 162, 255}, {49, 89, 154, 255}, {210, 215, 222, 255}, {235, 235, 236, 255}, {234, 
  235, 236, 255}, {236, 237, 237, 255}, {239, 238, 237, 255}, {125, 147, 183, 255}, {36, 80, 152, 255}, {47, 92, 166, 
  255}, {48, 92, 169, 255}, {47, 93, 169, 255}, {48, 92, 169, 255}, {47, 92, 169, 255}, {50, 94, 169, 255}, {32, 76, 
  157, 255}, {126, 149, 188, 251}, {252, 246, 237, 255}, {162, 186, 215, 148}}, {{171, 194, 222, 170}, {255, 255, 
  248, 255}, {146, 167, 196, 252}, {37, 86, 158, 255}, {50, 99, 174, 255}, {45, 94, 173, 255}, {47, 96, 173, 255}, 
  {48, 96, 174, 255}, {47, 97, 174, 255}, {47, 95, 173, 255}, {47, 94, 172, 255}, {46, 95, 172, 255}, {50, 96, 171, 
  255}, {33, 78, 156, 255}, {119, 143, 183, 255}, {243, 241, 237, 255}, {235, 236, 236, 255}, {207, 211, 218, 255}, 
  {215, 218, 224, 255}, {237, 237, 235, 255}, {217, 220, 224, 255}, {43, 82, 149, 255}, {44, 88, 162, 255}, {47, 91, 
  167, 255}, {46, 91, 167, 255}, {47, 89, 166, 255}, {46, 89, 164, 255}, {51, 91, 165, 255}, {32, 75, 155, 255}, 
  {124, 147, 186, 251}, {251, 244, 236, 255}, {161, 185, 214, 148}}, {{170, 193, 221, 170}, {255, 255, 246, 255}, 
  {138, 159, 191, 252}, {27, 76, 152, 255}, {51, 97, 172, 255}, {48, 95, 172, 255}, {48, 94, 171, 255}, {47, 92, 170, 
  255}, {46, 93, 171, 255}, {48, 93, 170, 255}, {48, 92, 169, 255}, {47, 92, 170, 255}, {49, 93, 166, 255}, {31, 73, 
  147, 255}, {184, 193, 209, 255}, {248, 246, 241, 255}, {196, 202, 212, 255}, {55, 81, 129, 255}, {205, 209, 216, 
  255}, {247, 245, 239, 255}, {132, 150, 181, 255}, {34, 72, 145, 255}, {47, 87, 161, 255}, {46, 87, 164, 255}, {45, 
  87, 163, 255}, {47, 87, 162, 255}, {47, 86, 163, 255}, {50, 89, 163, 255}, {31, 73, 150, 255}, {124, 145, 184, 
  251}, {249, 243, 235, 255}, {160, 184, 213, 148}}, {{168, 192, 219, 170}, {255, 255, 244, 255}, {138, 158, 190, 
  252}, {29, 75, 151, 255}, {49, 94, 170, 255}, {46, 92, 169, 255}, {47, 92, 169, 255}, {46, 91, 169, 255}, {46, 91, 
  168, 255}, {47, 90, 166, 255}, {46, 90, 166, 255}, {46, 88, 163, 255}, {29, 69, 146, 255}, {97, 123, 170, 255}, 
  {237, 237, 235, 255}, {224, 226, 228, 255}, {44, 73, 128, 255}, {115, 133, 166, 255}, {255, 254, 245, 255}, {145, 
  159, 185, 255}, {28, 65, 133, 255}, {44, 81, 154, 255}, {45, 84, 159, 255}, {46, 85, 160, 255}, {46, 85, 161, 255}, 
  {46, 85, 160, 255}, {45, 84, 159, 255}, {49, 86, 160, 255}, {31, 70, 148, 255}, {125, 143, 182, 251}, {248, 242, 
  234, 255}, {160, 184, 213, 148}}, {{168, 191, 219, 170}, {255, 253, 243, 255}, {137, 156, 188, 252}, {30, 73, 149, 
  255}, {50, 93, 167, 255}, {48, 90, 165, 255}, {47, 89, 165, 255}, {46, 89, 165, 255}, {47, 88, 164, 255}, {47, 87, 
  163, 255}, {46, 87, 161, 255}, {37, 76, 148, 255}, {136, 154, 188, 255}, {241, 240, 236, 255}, {226, 227, 228, 
  255}, {78, 102, 149, 255}, {39, 69, 128, 255}, {236, 235, 233, 255}, {161, 173, 194, 255}, {29, 63, 131, 255}, {44, 
  77, 148, 255}, {46, 82, 155, 255}, {46, 82, 157, 255}, {44, 82, 157, 255}, {46, 81, 155, 255}, {44, 82, 156, 255}, 
  {44, 81, 156, 255}, {48, 85, 157, 255}, {31, 67, 144, 255}, {123, 142, 180, 251}, {247, 241, 233, 255}, {159, 183, 
  212, 148}}, {{166, 190, 218, 170}, {255, 253, 242, 255}, {137, 155, 187, 252}, {29, 71, 145, 255}, {48, 89, 164, 
  255}, {47, 87, 163, 255}, {48, 87, 162, 255}, {47, 86, 161, 255}, {46, 85, 161, 255}, {49, 88, 162, 255}, {32, 71, 
  146, 255}, {107, 130, 173, 255}, {255, 253, 244, 255}, {199, 205, 213, 255}, {54, 82, 136, 255}, {25, 57, 124, 
  255}, {192, 198, 209, 255}, {180, 188, 203, 255}, {30, 62, 129, 255}, {41, 75, 144, 255}, {46, 80, 153, 255}, {45, 
  80, 154, 255}, {46, 82, 154, 255}, {45, 80, 154, 255}, {46, 79, 153, 255}, {45, 79, 152, 255}, {44, 79, 152, 255}, 
  {48, 80, 154, 255}, {30, 65, 139, 255}, {122, 141, 177, 251}, {246, 240, 232, 255}, {158, 182, 211, 148}}, {{166, 
  190, 217, 170}, {255, 252, 241, 255}, {136, 153, 185, 252}, {28, 67, 143, 255}, {49, 87, 161, 255}, {46, 85, 159, 
  255}, {46, 85, 159, 255}, {45, 83, 159, 255}, {48, 84, 157, 255}, {33, 70, 146, 255}, {41, 74, 140, 255}, {216, 
  219, 223, 255}, {191, 197, 208, 255}, {51, 79, 135, 255}, {17, 52, 123, 255}, {139, 153, 184, 255}, {239, 238, 232, 
  255}, {62, 87, 141, 255}, {33, 65, 134, 255}, {46, 79, 150, 255}, {44, 77, 150, 255}, {45, 79, 151, 255}, {44, 78, 
  151, 255}, {45, 78, 151, 255}, {45, 77, 149, 255}, {45, 77, 149, 255}, {45, 76, 147, 255}, {48, 78, 149, 255}, {29, 
  62, 136, 255}, {122, 138, 174, 251}, {245, 239, 231, 255}, {157, 181, 210, 148}}, {{164, 188, 217, 170}, {255, 250, 
  242, 255}, {135, 151, 184, 252}, {29, 64, 140, 255}, {49, 85, 159, 255}, {46, 83, 155, 255}, {45, 82, 157, 255}, 
  {46, 82, 155, 255}, {42, 75, 149, 255}, {68, 96, 154, 255}, {169, 179, 198, 255}, {205, 208, 215, 255}, {38, 66, 
  128, 255}, {37, 68, 136, 255}, {31, 63, 134, 255}, {163, 174, 195, 255}, {217, 219, 221, 255}, {43, 70, 132, 255}, 
  {39, 71, 141, 255}, {44, 76, 148, 255}, {44, 75, 147, 255}, {44, 75, 148, 255}, {44, 74, 147, 255}, {42, 75, 146, 
  255}, {43, 75, 146, 255}, {43, 74, 147, 255}, {42, 74, 144, 255}, {48, 76, 145, 255}, {28, 59, 134, 255}, {120, 
  136, 173, 251}, {244, 238, 230, 255}, {157, 181, 210, 148}}, {{164, 188, 217, 170}, {255, 248, 240, 255}, {135, 
  151, 182, 252}, {28, 65, 136, 255}, {48, 82, 153, 255}, {45, 79, 152, 255}, {44, 78, 153, 255}, {46, 79, 151, 255}, 
  {31, 65, 138, 255}, {170, 179, 199, 255}, {255, 253, 243, 255}, {97, 116, 156, 255}, {25, 56, 125, 255}, {48, 78, 
  145, 255}, {35, 67, 138, 255}, {65, 91, 146, 255}, {197, 201, 209, 255}, {57, 81, 141, 255}, {35, 66, 137, 255}, 
  {43, 75, 145, 255}, {44, 74, 146, 255}, {43, 73, 144, 255}, {42, 72, 143, 255}, {42, 72, 143, 255}, {42, 73, 142, 
  255}, {41, 71, 143, 255}, {42, 71, 141, 255}, {47, 74, 143, 255}, {27, 57, 129, 255}, {119, 135, 172, 251}, {243, 
  237, 229, 255}, {156, 180, 209, 148}}, {{163, 187, 216, 170}, {253, 247, 240, 255}, {133, 148, 179, 252}, {28, 61, 
  132, 255}, {47, 78, 150, 255}, {45, 77, 151, 255}, {44, 77, 149, 255}, {45, 77, 147, 255}, {26, 59, 132, 255}, 
  {148, 160, 187, 255}, {215, 217, 219, 255}, {46, 72, 129, 255}, {38, 68, 135, 255}, {43, 73, 143, 255}, {43, 72, 
  142, 255}, {35, 65, 135, 255}, {51, 78, 139, 255}, {43, 72, 137, 255}, {40, 69, 138, 255}, {42, 70, 141, 255}, {42, 
  70, 141, 255}, {41, 71, 141, 255}, {41, 71, 140, 255}, {42, 70, 141, 255}, {42, 69, 140, 255}, {41, 69, 139, 255}, 
  {41, 70, 139, 255}, {44, 71, 140, 255}, {26, 55, 126, 255}, {117, 133, 168, 251}, {241, 236, 228, 255}, {155, 179, 
  208, 148}}, {{162, 186, 215, 171}, {252, 247, 239, 255}, {145, 156, 184, 252}, {26, 56, 127, 255}, {46, 77, 146, 
  255}, {42, 74, 146, 255}, {43, 74, 145, 255}, {45, 76, 145, 255}, {25, 56, 130, 255}, {131, 144, 178, 255}, {147, 
  158, 182, 255}, {21, 50, 119, 255}, {43, 72, 139, 255}, {41, 70, 141, 255}, {42, 71, 140, 255}, {42, 71, 139, 255}, 
  {37, 65, 135, 255}, {39, 68, 136, 255}, {41, 69, 138, 255}, {41, 69, 139, 255}, {40, 69, 138, 255}, {40, 68, 138, 
  255}, {40, 68, 137, 255}, {41, 69, 136, 255}, {40, 67, 135, 255}, {41, 67, 136, 255}, {39, 67, 135, 255}, {44, 70, 
  137, 255}, {24, 52, 121, 255}, {125, 138, 170, 251}, {240, 235, 227, 255}, {154, 178, 207, 149}}, {{163, 187, 215, 
  155}, {243, 239, 233, 253}, {198, 203, 212, 251}, {37, 63, 126, 255}, {37, 66, 135, 255}, {46, 74, 144, 255}, {45, 
  74, 143, 255}, {45, 74, 142, 255}, {42, 70, 138, 255}, {54, 79, 141, 255}, {59, 82, 141, 255}, {40, 66, 133, 255}, 
  {44, 71, 137, 255}, {45, 71, 139, 255}, {44, 71, 139, 255}, {43, 72, 138, 255}, {46, 70, 138, 255}, {44, 69, 138, 
  255}, {43, 70, 138, 255}, {43, 70, 136, 255}, {43, 70, 136, 255}, {42, 68, 136, 255}, {42, 68, 135, 255}, {42, 69, 
  135, 255}, {42, 68, 134, 255}, {43, 68, 133, 255}, {43, 68, 133, 255}, {38, 60, 129, 255}, {30, 54, 119, 255}, 
  {179, 186, 199, 251}, {232, 228, 222, 253}, {155, 179, 207, 133}}, {{164, 188, 215, 84}, {233, 230, 227, 255}, 
  {231, 231, 230, 250}, {157, 167, 189, 255}, {35, 61, 125, 255}, {27, 53, 124, 255}, {28, 55, 126, 255}, {27, 55, 
  125, 255}, {28, 54, 125, 255}, {24, 50, 123, 255}, {23, 49, 122, 255}, {26, 53, 123, 255}, {26, 53, 124, 255}, {26, 
  53, 123, 255}, {26, 52, 124, 255}, {26, 52, 123, 255}, {26, 51, 122, 255}, {25, 51, 122, 255}, {26, 52, 122, 255}, 
  {26, 52, 120, 255}, {25, 52, 120, 255}, {25, 50, 121, 255}, {25, 50, 119, 255}, {25, 50, 120, 255}, {26, 49, 119, 
  255}, {25, 49, 117, 255}, {23, 48, 117, 255}, {31, 53, 117, 255}, {138, 148, 175, 255}, {220, 221, 220, 250}, {222, 
  218, 216, 255}, {155, 179, 207, 67}}, {{161, 186, 214, 8}, {232, 229, 227, 210}, {216, 219, 222, 255}, {229, 230, 
  230, 250}, {192, 198, 208, 252}, {136, 149, 178, 252}, {130, 143, 174, 252}, {130, 143, 175, 252}, {129, 143, 174, 
  252}, {131, 142, 174, 252}, {130, 143, 174, 252}, {129, 141, 173, 252}, {129, 143, 173, 252}, {129, 142, 173, 252}, 
  {128, 141, 172, 252}, {128, 140, 172, 252}, {129, 140, 172, 252}, {127, 140, 171, 252}, {128, 140, 170, 252}, {127, 
  139, 170, 252}, {127, 139, 171, 252}, {126, 138, 170, 252}, {126, 138, 170, 252}, {127, 138, 169, 252}, {127, 138, 
  169, 252}, {125, 137, 168, 252}, {130, 142, 171, 252}, {181, 187, 198, 252}, {219, 221, 219, 250}, {206, 210, 213, 
  255}, {220, 218, 215, 195}, {154, 178, 207, 3}}, {{170, 191, 215, 0}, {246, 238, 228, 34}, {231, 228, 226, 205}, 
  {230, 227, 224, 255}, {238, 234, 228, 255}, {245, 240, 232, 255}, {244, 239, 231, 255}, {244, 239, 231, 255}, {244, 
  238, 231, 255}, {243, 238, 230, 255}, {242, 237, 229, 255}, {242, 237, 229, 255}, {242, 236, 229, 255}, {242, 236, 
  228, 255}, {241, 236, 228, 255}, {240, 235, 227, 255}, {240, 235, 227, 255}, {240, 235, 227, 255}, {239, 234, 226, 
  255}, {239, 234, 226, 255}, {239, 233, 226, 255}, {238, 233, 225, 255}, {237, 232, 224, 255}, {237, 232, 224, 255}, 
  {236, 231, 224, 255}, {236, 232, 224, 255}, {236, 231, 223, 255}, {229, 224, 219, 255}, {220, 217, 214, 255}, {220, 
  217, 215, 194}, {233, 225, 216, 24}, {162, 182, 207, 1}}, {{122, 161, 206, 1}, {169, 190, 214, 0}, {160, 184, 212, 
  5}, {161, 184, 212, 79}, {159, 183, 211, 145}, {157, 181, 210, 170}, {157, 181, 210, 170}, {157, 181, 210, 170}, 
  {157, 181, 209, 170}, {157, 180, 209, 170}, {156, 180, 209, 170}, {156, 180, 209, 170}, {156, 180, 208, 170}, {156, 
  180, 208, 170}, {155, 179, 208, 170}, {155, 178, 207, 170}, {154, 178, 207, 170}, {154, 178, 207, 170}, {154, 178, 
  207, 170}, {154, 178, 207, 170}, {154, 178, 206, 170}, {153, 177, 206, 170}, {153, 177, 205, 170}, {153, 177, 206, 
  170}, {153, 176, 205, 170}, {152, 176, 205, 170}, {152, 176, 205, 169}, {153, 177, 205, 143}, {155, 178, 206, 76}, 
  {153, 177, 206, 2}, {162, 182, 206, 0}, {117, 156, 201, 1}}}], "Byte", ColorSpace -> "RGB", Interleaving -> True];

End[] (* End Private Context *)
           		
End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{RunKeeperOAuth`Private`runkeeperdata,RunKeeperOAuth`Private`runkeepercookeddata,RunKeeperOAuth`Private`runkeepersendmessage}
