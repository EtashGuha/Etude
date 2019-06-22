(* ::Package:: *)

Begin["Tools`GeoModelData`GeoFieldModelData`Private`"]

$ProtectedSymbols = {
	System`GeogravityModelData,
	System`GeomagneticModelData
};
	
Unprotect@@$ProtectedSymbols;
$tag = "GeoFieldModelDataCatchThrowTag";

getResourceFile[file_String] := If[
	FileExistsQ[#],
	Get[#],
	Message[GeogravityModelData::init];Throw[$Failed,"MissingResource"]
]&[FileNameJoin[{DirectoryName[$InputFileName],file}]];

(***load models***)
AbortProtect[getResourceFile/@{"geomagneticmodels.m","geogravitymodels.m"}];

(*Components*)
$GeoVectorComponents={"Magnitude","NorthComponent","EastComponent","DownComponent","HorizontalComponent","Declination","Inclination","Potential",All};
$ivproperties={All,Min,Max,Mean,StandardDeviation,Interval, System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ};
$rangespec={"Day", "Week", "Month", "Year", All};
(*Constants*)
$GM=3986004.415*10^8;(*for the model adopted gravity mass constant,in m^3/s^2*)
(*EarthRadiusMeters=6378136.3;(*in m*)*)
$EarthRadius=(*6371.2*)6378136.30/1000;
$omegaEarth=1458423/20000000000.;
(*coordinate conversion utilities*)
GeocentricComponentsToGeodeticComponents[{x_,y_,z_},phi_,phi2_]:=With[{psi=phi2-phi},{x Cos[psi]-z Sin[psi],y,x Sin[psi]+z Cos[psi]}];
GeodeticToGeocentric[{height_,phi_,lambda_,t_}]:=Module[
	{xyz=First@GeoPositionXYZ[GeoPosition[{phi,lambda,height}]]},
		{Norm[xyz],ArcTan[Sqrt[xyz[[1]]^2 + xyz[[2]]^2], xyz[[3]]]/.Indeterminate->Sign[phi]*(Pi/2)*0.99999,lambda Degree,t}];

(* fix for inventive breakage by users/QA *)
nowGeoUsage := If[DateObjectQ[#], #, $Failed]&[Now]

(***INPUT CODE***)
(***INPUT TESTS***)
loclistQ[GeoPosition[input_List,___]]:=VectorQ[input,Function[{x},MatchQ[x,{_?(NumericQ[#]||MatchQ[#,_Quantity]&), _?(NumericQ[#]||MatchQ[#,_Quantity]&)}|
	{_?(NumericQ[#]||MatchQ[#,_Quantity]&), _?(NumericQ[#]||MatchQ[#,_Quantity]&),_?(NumericQ[#]||MatchQ[#,_Quantity]&)}]]]
loclistQ[{pt1_,pt2_}]:=False
loclistQ[input_List]:=VectorQ[input,Function[{x},MemberQ[{GeoDisk,Polygon,Rectangle,GeoPosition,Point,Entity,EntityClass,GeoVariant,Association},Head[x]]||
	MatchQ[x,{_?(NumericQ[#]||MatchQ[#,_Quantity]&), _?(NumericQ[#]||MatchQ[#,_Quantity]&)}|
		{_?(NumericQ[#]||MatchQ[#,_Quantity]&), _?(NumericQ[#]||MatchQ[#,_Quantity]&),_?(NumericQ[#]||MatchQ[#,_Quantity]&)}]]]
loclistQ[input_]:=False

(* datespecQ *)
datepattern1 := {"DayNameShort", " ", "Day", " ", "MonthNameShort", 
   " ", "Year", " ", "Hour", ":", "Minute"};
datepattern2 := {"DayNameShort", " ", "Day", " ", "MonthNameShort", 
   " ", "Year", " ", "Hour", ":", "Minute", ":", "Second"};

datespecQ[date_] := Module[{datelist},
	Which[
		MatchQ[date,_DateObject],
		True,
		StringQ[date],
   		Quiet[
   			datelist = DateList[{date, datepattern1}];
    		If[Not[ListQ[datelist]], datelist = DateList[{date, datepattern2}]];
     		ListQ[datelist], 
     		{DateString::str}
     	],
     	ListQ[date],
   		Quiet[DateObjectQ[DateObject[date]]],
   		True,
   		False
   	]
];

datelistQ[input_List]:=Which[
	MatchQ[input,{_?datespecQ,_?datespecQ}],False,
	MatchQ[input,{_?datespecQ,_?datespecQ,_?(MemberQ[$rangespec,#]&)}],False,
	VectorQ[input,Function[{x},datespecQ[x]]],True,
	True,False
];
datelistQ[___]:=False;

(***INPUT FORMATTING***)
degreeConvert[x_]:=Module[{temp},
	If[NumericQ[x],
		x,
		temp=UnitConvert[x,"AngularDegrees"];
		If[temp===$Failed,temp,QuantityMagnitude[temp]]
	]
];

(* option checking code *)
optionCheck[opt_,function_]:=Module[{invalid=FilterRules[opt,Except[Options[function]]],mopt=Method/.opt,invm},
	Which[function===System`GeogravityModelData&&Not[MemberQ[{Automatic, Method}, mopt]],
		invm=DeleteCases[Flatten[{mopt}],"ModelDegree"|"IncludeRotation"->_];
		If[Length[invm]>0,Message[System`GeogravityModelData::moptx,First@invm,mopt,"ModelDegree"]],
		function===System`GeomagneticModelData&&Not[MemberQ[{Automatic, Method}, mopt]],
		invm=DeleteCases[Flatten[{mopt}],"Model"->_];
		If[Length[invm]>0,Message[System`GeomagneticModelData::moptx,First@invm,mopt,"Model"]]
	];
	If[Length[invalid]>0,Message[function::optx,First@invalid,function]];
];

(*GeoElevationData code for getting base height: elevation=distance from mean sea level*)
(*generate list of points for regions, using zero in case of timeouts*)
geoElevation[{longitude1_,longitude2_},{latitude1_,latitude2_},zoom_,type_]:=Module[
	{z=zoom,rows,columns,points,lat1=Min[latitude1,latitude2],lat2=Max[latitude1,latitude2],
		long1=Min[longitude1,longitude2],long2=Max[longitude1,longitude2],array,l},
	array={Sort@{lat1,lat2},Sort@{long1,long2}};
	If[MatchQ[zoom,Automatic],
		z=GeoGraphics`EstimateProjectedZooms[Reverse[array], {128, 64}, "Wolfram"];
		If[ListQ[z],z=Max[z]]; (*if estimated zoom is a list, take the larger value as a first approximation*)
		If[!IntegerQ[z],z=0];
		l=Length[Flatten[GIS`GeoPositionArray[array,GeoZoomLevel->z][[1]],1]];
		If[l>5000,
			z=z-Ceiling[Log[4,l/5000.]]
		]
	];
	If[z<1,
		points=Reverse/@Quiet[GIS`GeoPositionArray[array,GeoZoomLevel->z],{GeoPosition::invpos}];
		points=Quiet[GeoElevationData[points,Automatic,"GeoPosition"],
			{GeoElevationData::datap,GeoElevationData::timeout,GeoElevationData::data,
			GeoElevationData::nonopt}],
		points=Quiet[GeoElevationData[{GeoPosition[{lat1,long1}],GeoPosition[{lat2,long2}]},
			Automatic,"GeoPosition",GeoZoomLevel->z],
			{GeoElevationData::datap,GeoElevationData::timeout,GeoElevationData::data,
			GeoElevationData::nonopt}]
	];
	If[MatchQ[points,GeoPosition[List[_List..],___]],
		points[[1]],
		Message[type::elev];
		points=Reverse@Quiet[GIS`GeoPositionArray[array, GeoZoomLevel -> z], {GeoPosition::invpos}][[1]];
		rows=Length[points];
		If[rows===0,Return[{}],columns=Length[points[[1]]]];
		If[columns===0,Return[{}]];
		Table[Append[points[[lat,long]],0],{lat,rows},{long,columns}]
	]
];

(* determine time *)
(* TODO: add check for dated entities *)
getTime[Automatic]:=Module[{res = nowGeoUsage}, 
 	If[DateObjectQ[res], 
 		res, 
  		Message[System`GeomagneticModelData::now]; $Failed
  	]
];
getTime[x_]:=Module[{res = If[DateObjectQ[x],x,DateObject[x]]}, 
 	If[DateObjectQ[res], 
 		res, 
  		Message[System`GeomagneticModelData::time]; $Failed
  	]
];
getTime[___]:=(Message[System`GeomagneticModelData::time];$Failed);

getModel[Automatic]:=Automatic;
getModel[x:{Rule[_,_]..}]:=Module[{model="Model"/.x},
	Which[MemberQ[{"Model",Automatic},model],
		Automatic,
		MemberQ[{"IGRF","WMM", "GUFM1"},model],
		model,
		True,
		Message[System`GeomagneticModelData::model];Automatic
	]
];
getModel[x_Rule]:=getModel[{x}];
getModel[___]:=(Message[System`GeomagneticModelData::model];Automatic);

(***OUTPUT FORMATTING***)
(*component code: calculation from feild and unit assignment*)
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"NorthComponent"]:=north;
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"EastComponent"]:=east;
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"DownComponent"]:=vert;
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"HorizontalComponent"]:=Norm[{north,east}];
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"Declination"]:=ArcTan[north,east]*180/Pi;
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"Inclination"]:=ArcTan[Norm[{north,east}],vert]*180/Pi;
geoComponent[{vert_?NumericQ,north_?NumericQ,east_?NumericQ},"Magnitude"]:=Norm[{vert,north,east}];
geoComponent[pot_?NumericQ,"Potential"]:=pot;
geoComponent[x_,_]:=x;

geoComponentUnit[val_?NumericQ,"NorthComponent"|"EastComponent"|"DownComponent"|"HorizontalComponent"|"Magnitude"|
	"Potential",unit_]:=Quantity[val,unit];
geoComponentUnit[val_?NumericQ,"Declination"|"Inclination",unit_]:=Quantity[val,"AngularDegrees"];
geoComponentUnit[val_?(ArrayQ[#, 1|2, NumericQ]&),"NorthComponent"|"EastComponent"|"DownComponent"|"HorizontalComponent"|"Magnitude"|
	"Potential",unit_]:=QuantityArray[val,unit];
geoComponentUnit[val_?(ArrayQ[#, 1|2, NumericQ]&),"Declination"|"Inclination",unit_]:=QuantityArray[val,"AngularDegrees"];
geoComponentUnit[val_?(ArrayQ[#, 1|2]&),_,_]:=val;
geoComponentUnit[_,_,_]:=Missing["NotAvailable"];

(* Association splitting *)
Options[iGeoModelDataAssociation]:={Method->Automatic,GeoZoomLevel->Automatic};
iGeoModelDataAssociation[loc_,comp_,iv_,func_,opt:OptionsPattern[]]:=Module[{date,location},
	date=If[KeyExistsQ[loc,"Date"],loc["Date"],nowGeoUsage];
	location=If[KeyExistsQ[loc,"Location"],loc["Location"],$GeoLocation];
	If[Not[datespecQ[date]]&&Not[datelistQ[date]],Return[$Failed]];
	If[func===System`GeomagneticModelData,
		iGeomagneticModelData[location,comp,iv,"Time"->date,Sequence@@FilterRules[{opt},Options[func]]],
		iGeogravityModelData[location,comp,iv,Sequence@@FilterRules[{opt},Options[func]]]
	]
];

(***GEOGRAVITY DATA***)
(*zero tide system*)

(* for future reference, additional factor of 2 for PList and If[m==0,1/Sqrt[2],1] factor in signcoefflist
 are necessary adjustments to bring potential values in line with ICGEM reference source *)
PList[n_, theta_] := 2*Sqrt[2 Pi]*SpecialFunctions`SphericalHarmonicYTriangularArray[n, N[-Pi/2+theta]];
signcoefflist[o_] := Developer`ToPackedArray[N[Flatten[Table[Table[If[m==0,1/Sqrt[2],1],{m,0,n-1}], {n, 1, o + 1}]]]];

cfCos = Compile[{lam, {n, _Integer}}, With[{c = lam (*Pi/180*)}, Table[Cos[c j], {j, 0, n}]]];
cfSin = Compile[{lam, {n, _Integer}}, With[{c = lam (*Pi/180*)}, Table[Sin[c j], {j, 0, n}]]];

(*Geogravity potential and derivatives *)
VEGM[{r_, th_, lam_}] := -$GM/r*(1 + (rlist (coslist Cnlist + sinlist Snlist)).PGravity);
VEGMDr[{r_, th_, lam_}] := $GM*(-1/r^2 + (rplist (coslist Cnlist + sinlist Snlist)).PGravity);
VEGMDlam[{r_, th_, lam_}] := $GM/r*((rlist (-sinlist Cnlist + coslist Snlist) mlist).PGravity);
VEGMDtheta[{r_, th_, lam_}] := -$GM/r*(r^2*Tan[th]*((rplist (coslist Cnlist + sinlist Snlist)).PGravity) + 
    Sec[th] ((rlist ((coslist Cnlist + sinlist Snlist) mnlist)).PGravity2));
iGeogravityField[{r_, th_, lam_}] := Module[{}, 
	{
		VEGMDr[{r, th + $MachineEpsilon, lam}], 
		-VEGMDtheta[{r, th + $MachineEpsilon, lam}]/r, 
		VEGMDlam[{r, th + $MachineEpsilon, lam}]*Sec[th+$MachineEpsilon]/r, 
		VEGM[{r, th, lam}]
	}];

(*Geogravity component calculation*)
Options[calculateComponent]:={"Rotation"->True};
calculateComponent["Array",{r_,th_,lam_},lat_,opts:OptionsPattern[]]:=Module[{result=iGeogravityField[{r,th,lam}],potential,
		rotQ=OptionValue["Rotation"],a=r},
	potential=result[[-1]]+If[rotQ,-1/2$omegaEarth^2 a^2 Cos[th]^2,0];
	result=result[[;;3]]-If[rotQ,{-$omegaEarth^2 r Cos[th]^2, -$omegaEarth^2 r Cos[th] Sin[th],0},{0,0,0}];
	result=GeocentricComponentsToGeodeticComponents[result[[{2,3,1}]],lat,th];
	result={-1,1,1}*result[[{3,1,2}]];
	{result,potential}
]
calculateComponent[All,{r_,th_,lam_},lat_,opts:OptionsPattern[]]:=Module[{result=iGeogravityField[{r,th,lam}],potential,
		rotQ=OptionValue["Rotation"]},
	potential=Quantity[result[[-1]]+If[rotQ,-1/2$omegaEarth^2 r^2 Cos[th]^2,0],"Joules"/"Kilograms"];
	result=result[[;;3]]-If[rotQ,{-$omegaEarth^2 r Cos[th]^2, -$omegaEarth^2 r Cos[th] Sin[th],0},{0,0,0}];
	result=GeocentricComponentsToGeodeticComponents[result[[{2,3,1}]],lat,th];
	result={-1,1,1}*result[[{3,1,2}]];
	Association[Rule@@@Append[Map[{#,geoComponentUnit[geoComponent[result,#],#,"Meters"/"Seconds"^2]}&,Most[$GeoVectorComponents]],{"Potential",potential}]]
];
calculateComponent["Potential",{r_,th_,lam_},lat_,opts:OptionsPattern[]]:=Module[{pot=VEGM[{r,th,lam}],
	rotQ=OptionValue["Rotation"]},
	geoComponent[pot+If[rotQ,-1/2 $omegaEarth^2 r^2 Cos[th]^2,0],"Potential"]
];
calculateComponent[comp_,{r_,th_,lam_},lat_,opts:OptionsPattern[]]:=Module[
	{a=r,result={VEGMDr[{r,th*Pi/180+$MachineEpsilon,lam+$MachineEpsilon}],
		-VEGMDtheta[{r,th*Pi/180+$MachineEpsilon,lam+$MachineEpsilon}]/r,
		VEGMDlam[{r,th*Pi/180+$MachineEpsilon,lam+$MachineEpsilon}]*Sec[th*Pi/180+$MachineEpsilon]/r},rotQ=OptionValue["Rotation"]},
	result=result[[;;3]]-If[rotQ,{-$omegaEarth^2 r Cos[th]^2, -$omegaEarth^2 r Cos[th] Sin[th],0},{0,0,0}];
	result=GeocentricComponentsToGeodeticComponents[result[[{2,3,1}]],lat,th];
	result={-1,1,1}*result[[{3,1,2}]];
	geoComponent[result,comp]
];

(*Internal GeogravityModelData Code*)
Options[iGeogravityModelData]:={Method->Automatic,GeoZoomLevel->Automatic};
iGeogravityModelData[loc_,comp_,iv_,opts:OptionsPattern[]]:=Module[{location,userzQ,tempht,
	zoom=OptionValue[GeoZoomLevel],o="ModelDegree"/.(OptionValue[Method]/.Automatic->{"ModelDegree"->60}),
		spin="IncludeRotation"/.(OptionValue[Method]/.Automatic->{"IncludeRotation"->True})},
	Quiet[If[DataPaclets`GeoLocationQ[loc,GeogravityModelData],location=DataPaclets`GeoLocationFormat[loc,GeogravityModelData],
		Message[System`GeogravityModelData::loc,loc];Return[$Failed]],{Quantity::unkunit}];
	If[location===$Failed,Return[$Failed]];
	If[MatchQ[location,{{_?NumericQ,_},{_,_},_}],
		If[location[[1,1]]==location[[2,1]]||location[[1,2]]==location[[2,2]],
			Message[System`GeogravityModelData::invreg,loc];Return[$Failed]
	]];
	With[{l=If[MatchQ[loc,_GeoPosition],loc[[1]],loc]},
		userzQ=Which[QuantityQ[l[[-1]]],CompatibleUnitQ[l[[-1]],Quantity["Meters"]],
			MatchQ[l, {_?(Length[#]>1&), _?NumericQ} | {_?(Length[#]>1&), _Quantity}]&&Length[l]===2&&ListQ[l],True,
			Length[l]>2&&Not[TrueQ[l[[-1]]]],True,
			True,False
		]
	];
	If[userzQ,
		tempht=(loc /. GeoPosition[x_List,___] :> x)[[-1]];
		If[Not[TrueQ[Element[location[[3]], Reals]]],
			Message[System`GeogravityModelData::invelev,tempht];
			Return[$Failed]
		];
		If[TrueQ[(location[[3]]<-10000||location[[3]]>600000)],
			tempht=location[[3]]/1000;
			Message[System`GeogravityModelData::elevrng,tempht];
			Return[Missing["NotAvailable"]]
		];
	];
	If[MatchQ[location,{_,_,__}],
		If[Not[NumericQ[location[[3]]]||TrueQ[location[[3]]]],Message[System`GeogravityModelData::elev];location[[3]]=0]];
	If[Not[MemberQ[{True,False,"IncludeRotation"},spin]],Message[System`GeogravityModelData::rotation,spin]];
	If[Not[BooleanQ[spin]], spin = True];
	If[o==="ModelDegree",o=60];
	If[o>360||Not[IntegerQ[o]]||o<2,Message[System`GeogravityModelData::range,o];Return[Missing["NotAvailable"]]];
	If[Not[IntegerQ[zoom]||MatchQ[zoom,Automatic]],Message[System`GeomagneticModelData::zoom,zoom];zoom=1];
	If[iv===All,
		iGeogravityModelCompute[location,comp,GeoZoomLevel->zoom, includeRotation->spin, modelDegree->o],
		iGeogravityModelCompute[location,comp,iv,GeoZoomLevel->zoom, includeRotation->spin, modelDegree->o]
	]
];

Clear[iGeogravityModelCompute]
Options[iGeogravityModelCompute]:={includeRotation->True,GeoZoomLevel->Automatic, modelDegree->60,"Unitless"->False};
iGeogravityModelCompute[{latitudedegrees_,longitudedegrees_,elevationmeters_,___},comp_,iv:System`GeoVector|System`GeoVectorENU|System`GeoVectorXYZ,opts:OptionsPattern[]]:=Module[
	{data,quantity,position,east,north,up},
	data=iGeogravityModelCompute[{latitudedegrees,longitudedegrees,elevationmeters},All,opts,"Unitless"->True];
	(*positions and components*)
	position={latitudedegrees,longitudedegrees,elevationmeters};
	east=data[[1,3]];
	north=data[[1,2]];
	up=-data[[1,1]];
	If[!FreeQ[{position,east,north,up},Missing|$Failed],Return[Missing["NotAvailable"]]];
	If[MemberQ[{"Potential","Magnitude"},comp],
		Missing["NotApplicable"],
		Switch[iv,
			System`GeoVectorENU,
				System`GeoVectorENU[GeoPosition[position]->QuantityArray[{east,north,up},"Meters"/"Seconds"^2]],
			System`GeoVector,
				data=System`GeoVectorENU[GeoPosition[position]->QuantityArray[{east,north,up},"Meters"/"Seconds"^2]];
				System`GeoVector[data],
			System`GeoVectorXYZ,
				data=System`GeoVectorXYZ[System`GeoVectorENU[GeoPosition[position]->{east,north,up}]];
				System`GeoVectorXYZ[GeoPosition[position]->QuantityArray[data["Vector"],"Meters"/"Seconds"^2]],
			_,$Failed
		]
	]
];
iGeogravityModelCompute[{{lat1_,long1_},{lat2_,long2_},elevation_},comp_,iv:System`GeoVector|System`GeoVectorENU|System`GeoVectorXYZ,opts:OptionsPattern[]]:=Module[
	{data,quantity,positions,east,north,up},
	data=iGeogravityModelCompute[{{lat1,long1},{lat2,long2},elevation},All,opts,"Unitless"->True];
	(*positions and components*)
	positions=Flatten[data["Position"],1];
	east=Flatten[data["EastComponent"]];
	north=Flatten[data["NorthComponent"]];
	up=Flatten[-data["DownComponent"]];
	If[!FreeQ[{positions,east,north,up},Missing|$Failed],Return[Missing["NotAvailable"]]];
	If[MemberQ[{"Potential","Magnitude"},comp],
		Missing["NotApplicable"],
		Switch[iv,
			System`GeoVectorENU,
				System`GeoVectorENU[GeoPosition[positions]->
					QuantityArray[Transpose[{east,north,up}],"Meters"/"Seconds"^2]],
			System`GeoVector,
				data=System`GeoVectorENU[GeoPosition[positions]->
					QuantityArray[Transpose[{east,north,up}],"Meters"/"Seconds"^2]];
				System`GeoVector[data],
			System`GeoVectorXYZ,
				data=System`GeoVectorXYZ[System`GeoVectorENU[GeoPosition[positions]->Transpose[{east,north,up}]]];
				System`GeoVectorXYZ[GeoPosition[positions]->QuantityArray[data["Vector"],"Meters"/"Seconds"^2]],
			_,$Failed
		]
	]
];
iGeogravityModelCompute[{{lat1_,long1_},{lat2_,long2_},elevation_},comp_String,iv_?(MemberQ[$ivproperties,#]&),opts:OptionsPattern[]]:=Module[
	{data,data2,quantity},
	data=iGeogravityModelCompute[{{lat1,long1},{lat2,long2},elevation},comp,opts];
	data2=data;
	data=DeleteCases[DeleteCases[Flatten[QuantityMagnitude[data]],_Missing],_QuantityMagnitude];
	Which[iv===All,
			data2,
		data==={},
			Missing["NotAvailable"],
		iv===Interval,
			quantity=QuantityUnit[data2];
			If[MatchQ[quantity,_List],quantity=quantity[[1,1]]];
			data=Sort[data][[{-1,1}]];
			Quantity[Interval[data],quantity],
		True,
			quantity=QuantityUnit[data2];
			If[MatchQ[quantity,_List],quantity=quantity[[1,1]]];
			Quantity[iv[data],quantity]
	]
];
iGeogravityModelCompute[{{lat1_,long1_},{lat2_,long2_},elevation_},comp_?(StringQ[#]||MatchQ[#,All]&),opts:OptionsPattern[]]:=Module[
	{lam,th,r,inputs,lat,L,PGravityAll,elevationmeters=elevation,last,result,potential,
		o=OptionValue[modelDegree],spin=OptionValue[includeRotation],
		(*setting higher precision to avoid cancellation errors under high exponents*)
		a=SetPrecision[$EarthRadius*1000,16],
		(* zoom symbols *)points,zoom,rows,columns},
	zoom=OptionValue[GeoZoomLevel];
	If[Not[IntegerQ[zoom]||MatchQ[zoom,Automatic]],Message[System`GeomagneticModelData::zoom, zoom];zoom=1];
	last=sumlist[[o+1]]+o;
	(* get elevations *)
	If[TrueQ[elevation],
		points=geoElevation[{long1,long2},{lat1,lat2},zoom,System`GeogravityModelData];
		rows=Length[points];
		If[rows===0,Return[Missing["NotAvailable"]],columns=Length[points[[1]]]];
		If[columns===0,Return[Missing["NotAvailable"]]],
		If[MatchQ[zoom,Automatic],zoom=GeoGraphics`EstimateProjectedZooms[{Sort@{long1, long2},Sort@{lat1, lat2}}, {128, 64}, "Wolfram"]];
		points=Quiet[GIS`GeoPositionArray[{Sort@{lat1, lat2}, Sort@{long1, long2}}, GeoZoomLevel -> zoom], {GeoPosition::invpos}][[1]];
		rows=Length[points];
		If[rows===0,Return[Missing["NotAvailable"]],columns=Length[points[[1]]]];
		If[columns===0,Return[Missing["NotAvailable"]]];
		If[elevationmeters<-10000||elevationmeters>600000,
			Message[System`GeogravityModelData::elevrng,elevation];
			Return[Table[Missing["NotAvailable"],{i,rows},{j,columns}]],
			points=Table[Append[points[[i,j]],0],{i,rows},{j,columns}]
		]
	];
	(* core code *)
	If[o===360,
		Cnlist=CnList;Snlist=SnList;mlist=mList;mnlist=mnList,
		Cnlist=CnList[[;;last-3]];Snlist=SnList[[;;last-3]];mlist=mList[[;;last-3]];mnlist=mnList[[;;last-3]]
	];
	DynamicModule[{progress = 0,text},
		If[TrueQ[$Notebooks],
			text=Row[{"Calculating ",Dynamic[progress]," of ",ToString[rows*columns]," values...."}];
			PrintTemporary[Internal`LoadingPanel[text]]];
		result=Table[progress=rows*(i-1);
			lam=(points[[1,i,2]](*+180*));
			L=cfCos[lam Degree,o];coslist=Developer`ToPackedArray@Flatten[Table[Take[L,n+1],{n,2,o}]];
			L=cfSin[lam Degree,o];sinlist=Developer`ToPackedArray@Flatten[Table[Take[L,n+1],{n,2,o}]];
			Table[
				lat=points[[j,1,1]];
				inputs=GeodeticToGeocentric[{points[[j,i,3]],lat,lam,0}];
				th=inputs[[2]];
				(*setting higher precision to avoid cancellation errors under high exponents*)
				r=SetPrecision[inputs[[1]],16];
				rlist=Developer`ToPackedArray@Flatten[Table[(a/r)^n ConstantArray[1,{n+1}],{n,2,o}]];
				rplist=Developer`ToPackedArray@Flatten[Table[-(n+1)(a^n/r^(n+2)) ConstantArray[1,{n+1}],{n,2,o}]];
				PGravityAll=Developer`ToPackedArray[N[PList[o+1,th] signcoefflist[o+1]]];
				PGravity=PGravityAll[[4;;last]];
				PGravity2=Flatten[Table[PGravityAll[[sumlist[[n+2]];;sumlist[[n+2]]+n]],{n,2,o}]];
				calculateComponent[If[comp===All,"Array",comp],{r,th,lam Degree},lat Degree,"Rotation"->spin],
			{j,rows}],
		{i,columns}];
		progress=rows*columns
	];
	result=Reverse[Transpose[result]];
	result=N[result];(*ensure output results are MachinePrecision*)
	Which[
		TrueQ[OptionValue["Unitless"]]&&comp===All,
		result = result[[All, All, 1]];
		result = Association[Rule @@@ Map[Function[{type}, {type, 
			Map[geoComponent[#, type] &, result, {2}]}], 
        	Most[$GeoVectorComponents]]];
		Append[result,"Position"->QuantityArray[points, 
			{"AngularDegrees", "AngularDegrees", "Meters"}]],
		
		comp===All,
		potential = QuantityArray[result[[All, All, 2]], "Joules"/"Kilograms"];
		result = result[[All, All, 1]];
		result = Association[Rule @@@ Append[Map[Function[{type}, {type, geoComponentUnit[
        	Map[geoComponent[#, type] &, result, {2}], type, "Meters"/"Seconds"^2]}], 
        	Most[$GeoVectorComponents]], {"Potential", potential}]];
		Append[result,"Position"->QuantityArray[points, 
			{"AngularDegrees", "AngularDegrees", "Meters"}]],
		comp==="Potential",
		geoComponentUnit[result,comp,"Joules"/"Kilograms"],
		True,
		geoComponentUnit[result,comp,"Meters"/"Seconds"^2]
	]
];
iGeogravityModelCompute[{latitudedegrees_,longitudedegrees_,elevationmeters_,___},comp_?(StringQ[#]||MatchQ[#,All]&),opts:OptionsPattern[]]:=Module[
	{inputs=GeodeticToGeocentric[{elevationmeters,latitudedegrees,((*180+*)longitudedegrees),0}],
		(*^ for some reason we don't adjust longitude degrees for gravity but do for magnetism, at least each are consistent with external sources*)
		r,lam,th,last,
		o=OptionValue[modelDegree],spin=OptionValue[includeRotation],
		(*setting higher precision to avoid cancellation errors under high exponents*)
		a=SetPrecision[$EarthRadius*1000,16],
		L,PGravityAll,result},
	last=sumlist[[o+1]]+o;
	(*setting higher precision to avoid cancellation errors under high exponents*)
	r=SetPrecision[inputs[[1]],16];th=inputs[[2]];lam=inputs[[3]];
	If[o===360,
		Cnlist=CnList;Snlist=SnList;mlist=mList;mnlist=mnList,
		Cnlist=CnList[[;;last-3]];Snlist=SnList[[;;last-3]];mlist=mList[[;;last-3]];mnlist=mnList[[;;last-3]]
	];
	L=cfCos[lam,o];coslist=Developer`ToPackedArray@Flatten[Table[Take[L,n+1],{n,2,o}]];
	L=cfSin[lam,o];sinlist=Developer`ToPackedArray@Flatten[Table[Take[L,n+1],{n,2,o}]];
	rlist=Developer`ToPackedArray@Flatten[Table[(a/r)^n ConstantArray[1,{n+1}],{n,2,o}]];
	rplist=Developer`ToPackedArray@Flatten[Table[-(n+1)(a^n/r^(n+2)) ConstantArray[1,{n+1}],{n,2,o}]];
	PGravityAll=Developer`ToPackedArray[N[PList[o+1,th] signcoefflist[o+1]]];
	PGravity=PGravityAll[[4;;last]];
	PGravity2=Flatten[Table[PGravityAll[[sumlist[[n+2]];;sumlist[[n+2]]+n]],{n,2,o}]];
	result=calculateComponent[If[TrueQ[OptionValue["Unitless"]],"Array",comp],{r,th,lam},latitudedegrees Degree,"Rotation"->spin];
	result=N[result];(*ensure output results are MachinePrecision*)
	Which[comp===All,
		result,
		comp==="Potential",
		geoComponentUnit[result,comp,"Joules"/"Kilograms"],
		True,
		geoComponentUnit[result,comp,"Meters"/"Seconds"^2]
	]
];
iGeogravityModelCompute[___]:=$Failed;

(*********************************INPUT CODE*************************)
Options[System`GeogravityModelData] := {GeoZoomLevel -> Automatic, Method -> Automatic};
System`GeogravityModelData[loc_?loclistQ,args___]:=Module[
	{res=System`GeogravityModelData[#,args]&/@If[MatchQ[loc,_GeoPosition],loc[[1]],loc]},
	Which[
    	FreeQ[{args}, GeoVector | GeoVectorENU | GeoVectorXYZ] || MatchQ[res, _GeoVector | _GeoVectorENU | _GeoVectorXYZ],
    	    res,  (*head is GeoVector*then we don't need to combine into one GeoVector**)
 
    	MatchQ[res, {_Missing ..}],
    	    Missing["NotAvailable"],
 	
    	res = If[MatchQ[#, _GeoVector | _GeoVectorENU | _GeoVectorXYZ], #[[1]], #] & /@ res;
    	If[MatchQ[res, {__Rule}], res = GeoPosition[res[[All, 1]]] -> QuantityArray[res[[All, 2]]]];
    	!FreeQ[{args}, GeoVectorENU],
    	    GeoVectorENU[res],
    	!FreeQ[{args}, GeoVectorXYZ],
    	    GeoVectorXYZ[res],
    	True,
    	    GeoVector[res]
 	]
]
System`GeogravityModelData[loc_Association,comp_?(MemberQ[$GeoVectorComponents,#]&),iv_?(MemberQ[$ivproperties,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,comp,iv,System`GeogravityModelData,opt]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
System`GeogravityModelData[loc_,comp_?(MemberQ[$GeoVectorComponents,#]&),iv_?(MemberQ[$ivproperties,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeogravityModelData[loc,comp,iv,Sequence@@FilterRules[{opt},Options[System`GeogravityModelData]]]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];

System`GeogravityModelData[loc_Association,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,All,iv,System`GeogravityModelData,opt]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
System`GeogravityModelData[loc_Association,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,comp,All,System`GeogravityModelData,opt]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
System`GeogravityModelData[loc_,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeogravityModelData[loc,All,iv,opt]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
System`GeogravityModelData[loc_,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeogravityModelData[loc,comp,All,Sequence@@FilterRules[{opt},Options[System`GeogravityModelData]]]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];

System`GeogravityModelData[loc_Association,opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,All,All,System`GeogravityModelData,opt]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
System`GeogravityModelData[comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=
	System`GeogravityModelData[$GeoLocation,comp,opt];
System`GeogravityModelData[loc_?(MatchQ[#, Except[_Rule]] &),opt:OptionsPattern[]]:=With[
	{res=iGeogravityModelData[loc,All,All,Sequence@@FilterRules[{opt},Options[System`GeogravityModelData]]]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
System`GeogravityModelData[opt:OptionsPattern[]]:=With[
	{res=iGeogravityModelData[$GeoLocation,All,All,Sequence@@FilterRules[{opt},Options[System`GeogravityModelData]]]},
	(optionCheck[{opt},System`GeogravityModelData];res)/;res=!=$Failed
];
  
(***GEOMAGNETIC COEFFICIENTS***)
(*source http://www.ngdc.noaa.gov/geomag/WMM/soft.shtml*)
(*also http://www.ngdc.noaa.gov/IAGA/vmod/igrf.html*)

intervSpline[tknts_, time_, nspl_]:=Module[{},
	If[time<tknts[[4]]||time>tknts[[nspl+1]],Return[$Failed]];
	First@FirstPosition[tknts,_?(time<=#&)]
];
bspline[tknts_, t_, nspl_, nleft_]:=Module[{spl=ConstantArray[1.0,4],
	deltar=ConstantArray[0,4],deltal=ConstantArray[0,4], saved,term},
	(*calculate splines of order 4*)
	Do[ 
		deltar[[j]]=tknts[[nleft+j]]-t;
		deltal[[j]]=t-tknts[[nleft+1-j]];
		saved=0.0;
		Do[
			term=spl[[i]]/(deltar[[i]]+deltal[[j+1-i]]);
			spl[[i]]=saved+deltar[[i]]*term;
			saved=deltal[[j+1-i]]*term,
		{i,1,j}];
		spl[[j+1]]=saved,
	{j,1,3}];
	spl
];

PMagnetic[m_,n_,x_]:=If[m>0&&n>=m,Sqrt[2 (n-m)!/(n+m)!],1] LegendreP[n,m,x];
gMagnetic[m_,n_][t_]:=gMagnetic[m,n,0]+t gMagnetic[m,n,1];
hMagnetic[m_,n_][t_]:=hMagnetic[m,n,0]+t hMagnetic[m,n,1];
gSpline[m_,n_][t_?NumericQ]:=Module[{nleft=intervSpline[tknts, t, nsplines],spl,k=klist[0,m,n]},
	spl=bspline[tknts, t, nsplines, nleft];
	Sum[spl [[j]]*gt[[j+nleft-4,k]],{j,1,4}]
];
hSpline[0,n_][t_]:=0;
hSpline[m_,n_][t_?NumericQ]:=Module[{nleft=intervSpline[tknts, t, nsplines],spl,k=klist[1,m,n]},
	spl=bspline[tknts, t, nsplines, nleft];
	Sum[spl [[j]]*gt[[j+nleft-4,k]],{j,1,4}]
];

pickYear["GUFM1",year_]:=If[year<1590||year>1990,
		Message[System`GeomagneticModelData::modelrng,"GUFM1",1590,1990];$Failed,
		{year,"GUFM1"}
	];
pickYear["IGRF",year_]:=Module[{baseyear=Floor[year,5]},
	If[baseyear<minyear||baseyear>maxyearIGRF,
		Message[System`GeomagneticModelData::modelrng,"IGRF",minyear,maxyearIGRF];$Failed,
		baseyear
	]
];
pickYear["WMM",year_]:=Module[{baseyear=Floor[year,5]},
	If[baseyear<WMMyear||baseyear>WMMyear,
		Message[System`GeomagneticModelData::modelrng,"WMM",WMMyear,WMMyear+5];$Failed,
		{baseyear,"WMM"}
	]
];
pickYear[Automatic,year_]:=pickYear[pickMagneticModel[year],year];
pickMagneticModel[year_]:=Module[{baseyear=Floor[year,5]},
	Which[baseyear<minyear,
		"GUFM1",
		baseyear<WMMyear,
		"IGRF",
		True,
		"WMM"
	]
];

(***Pole Finding Code***)
Options[iPoleFinder]:={"Altitude"->0};
(*geomagnetic codes*)
iPoleFinder[date_,"GUFM1"]:=Module[{year=QuantityMagnitude@DateDifference[DateObject[{1, 0, 0}], date, "Year"],
	g0t,h1t,g1t,phi,lambda},
	If[year===$Failed,Return[$Failed]];
	g0t=gSpline[0,1][year];
	g1t=gSpline[1,1][year];
	h1t=hSpline[1,1][year];
	phi=-ArcTan[Sqrt[h1t^2+g1t^2]/g0t];
	lambda=ArcTan[h1t/g1t];
	{Pi/2-phi,lambda}
];
iPoleFinder[date_,model_]:=Module[{yeardiff, baseyear=pickYear[model,First[DateList[date]]],
	g0t,h1t,g1t,phi,lambda},
	If[baseyear===$Failed,Return[$Failed]];
	If[MatchQ[baseyear,_List],
		yeardiff=First[DateDifference[{baseyear[[1]],1},date,"Year"]];
		baseyear=baseyear[[2]];
		If[MatchQ[baseyear,"GUFM1"],
			baseyear=QuantityMagnitude@DateDifference[DateObject[{1, 0, 0}], date, "Year"];
			g0t=gSpline[0,1][baseyear];
			g1t=gSpline[1,1][baseyear];
			h1t=hSpline[1,1][baseyear],
			g0t=g0[baseyear]+yeardiff*(g0["WMM+"]-g0[baseyear])/5;
			g1t=g1[baseyear]+yeardiff*(g1["WMM+"]-g1[baseyear])/5;
			h1t=h1[baseyear]+yeardiff*(h1["WMM+"]-h1[baseyear])/5
		],
		yeardiff=First[DateDifference[{baseyear,1},date,"Year"]];
		g0t=g0[baseyear]+yeardiff*(g0[baseyear+5]-g0[baseyear])/5;
		g1t=g1[baseyear]+yeardiff*(g1[baseyear+5]-g1[baseyear])/5;
		h1t=h1[baseyear]+yeardiff*(h1[baseyear+5]-h1[baseyear])/5
	];
	phi=-ArcTan[Sqrt[h1t^2+g1t^2]/g0t];
	lambda=ArcTan[h1t/g1t];
	{Pi/2-phi,lambda}
];
(*dip pole codes*)
eastcomp[long_?NumericQ,lat_?NumericQ,date_,Automatic]:=With[{coord=adjustlonglat[{lat,long}]},QuantityMagnitude[
	System`GeomagneticModelData[coord,date,"EastComponent","Method"->{"Time"->date}]]];
northcomp[long_?NumericQ,lat_?NumericQ,date_,Automatic]:=With[{coord=adjustlonglat[{lat,long}]},QuantityMagnitude[
	System`GeomagneticModelData[coord,date,"NorthComponent","Method"->{"Time"->date}]]];
eastcomp[long_?NumericQ,lat_?NumericQ,date_,alt_]:=With[{coord=adjustlonglat[{lat,long}]},QuantityMagnitude[
	System`GeomagneticModelData[coord,date,"EastComponent","Method"->{"Time"->date}]]];
northcomp[long_?NumericQ,lat_?NumericQ,date_,alt_]:=With[{coord=adjustlonglat[{lat,long}]},QuantityMagnitude[
	System`GeomagneticModelData[coord,date,"NorthComponent","Method"->{"Time"->date}]]];

adjustlonglat[{lat_,long_}]:=Module[{res},
	(*adjust for latitudes >90 or <-90*)
	res=Which[
		lat>90,{180-lat,long-180},
		lat<-90,{-180-lat,long-180},
		True,{lat,long}
	];
	If[res[[2]]<-180,res[[2]]=res[[2]]+360];
	If[res[[2]]>180,res[[2]]=res[[2]]-360];
	res
]

(*geomagnetic poles*)
$geopoles:={"NorthGeomagneticPole","SouthGeomagneticPole"};
iPoleFinder["NorthGeomagneticPole",date_,model_,opt:OptionsPattern[]]:=Module[{result=iPoleFinder[date,model]},
	If[result===$Failed,Return[Missing["NotAvailable"]]];
	result=UnitConvert[Quantity[#,"Radians"],"AngularDegrees"]&/@result;
	result=QuantityMagnitude@result;
	If[MatchQ[result,{_?NumericQ,_?NumericQ}],
		If[result[[2]]>180,result=result-{0,360}];
		If[result[[2]]<-180,result=result+{0,360}];
		GeoPosition[N@result],
		Missing["NotAvailable"]
	]
];
iPoleFinder["SouthGeomagneticPole",date_,model_,opt:OptionsPattern[]]:=Module[{result=iPoleFinder[date,model]},
	If[result===$Failed,Return[Missing["NotAvailable"]]];
	result=UnitConvert[Quantity[#,"Radians"],"AngularDegrees"]&/@{-result[[1]],Pi+result[[2]]};
	result=QuantityMagnitude@result;
	If[MatchQ[result,{_?NumericQ,_?NumericQ}],
		If[result[[2]]>180,result=result-{0,360}];
		If[result[[2]]<-180,result=result+{0,360}];
		GeoPosition[N@result],
		Missing["NotAvailable"]
	]
];

(*dip poles*)
$dippoles:={"NorthModelDipPole","SouthModelDipPole","NorthModelDipPoleGeoElevation","SouthModelDipPoleGeoElevation"};
iPoleFinder[comp_?(MemberQ[$dippoles,#]&),date_,model_,opt:OptionsPattern[]]:=Module[
	{north0,long,south0,lat,sol,alt=OptionValue["Altitude"],baseyear=pickYear[model,First[DateList[date]]]},
	If[baseyear===$Failed,Return[$Failed]];
	north0={{long,-70},{lat,70,0,90-$MachineEpsilon}};
	south0={{long,70},{lat,-70,-90+$MachineEpsilon,0}};
	With[{limits=If[StringMatchQ[comp,"North"~~__],north0,south0]},
		Quiet[sol=FindRoot[{eastcomp[long,lat,date,alt],northcomp[long,lat,date,alt]},limits];
			If[Complement[$MessageList,{FindRoot::lstol,FindRoot::nlnum}]=!={},
				Message[System`GeomagneticModelData::erpole,comp]
			],
			{FindRoot::reged,FindRoot::nlnum,FindRoot::lstol,GeoElevationData::timeout, GeoElevationData::data}]
	];
	sol={lat,long}/.sol;
	If[MatchQ[sol,{_?NumericQ,_?NumericQ}],
		sol=adjustlonglat[sol];(*hypothetically a latitude exceeding-90 or 90 could be returned*)
		If[Norm[sol - Reverse@If[StringMatchQ[comp, "North" ~~ __], north0[[All, 2]], south0[[All, 2]]]] < 10^-15,
			Missing["NotAvailable"], (*TODO: error message for utter failure of GeoElevation?*)
			GeoPosition[N@sol]
		],
		Missing["NotAvailable"]
	]
];	
iPoleFinder[___]:=$Failed;
	
(*MAIN GEOMAGNETIC CODE*)
(***FIELD CODES***)
GeomagneticField[elevation_,lattitudedegrees_,longitudedegrees_,yearsafter_,baseyear_]:=Module[{
	input={1/1000,1,1,1}*GeodeticToGeocentric[{elevation,lattitudedegrees,(180+longitudedegrees),yearsafter}],result},
	If[Pi/2-Abs[input[[2]]]<10^-6,input[[2]]=Sign[input[[2]]]*1.5707963`];
	result=GeocentricComponentsXX[baseyear]@@input;
	{1,-1,1}*GeocentricComponentsToGeodeticComponents[result,lattitudedegrees Degree,input[[2]]]];
GeomagneticField2[elevation_,lattitudedegrees_,longitudedegrees_,yearsafter_,baseyear_]:=Module[{pot,
	input={1/1000,1,1,1}*GeodeticToGeocentric[{elevation,lattitudedegrees,(180+longitudedegrees),yearsafter}],result},
	pot=GeomagneticPotential[baseyear]@@input;
	If[Pi/2-Abs[input[[2]]]<10^-6,input[[2]]=Sign[input[[2]]]*1.5707963`];
	result=GeocentricComponentsXX[baseyear]@@input;
	Append[{1,-1,1}*GeocentricComponentsToGeodeticComponents[result,lattitudedegrees Degree,input[[2]]],pot]
];

(***SORTING AND FORMATTING CODE***)
Options[iGeomagneticModelData]:={GeoZoomLevel->Automatic, Method->Automatic,"Time"->Automatic};
iGeomagneticModelData[loc_,comp_,iv_,opt:OptionsPattern[]]:=Module[{location,userzQ,tempht,
	time=getTime[OptionValue["Time"]],model=getModel[OptionValue[Method]],
	zoom=OptionValue[GeoZoomLevel]},
	If[time===$Failed,Return[$Failed]];
	Quiet[If[DataPaclets`GeoLocationQ[loc,GeomagneticModelData],location=DataPaclets`GeoLocationFormat[loc,GeomagneticModelData],
		Message[System`GeomagneticModelData::loc,loc];Return[$Failed]],{Quantity::unkunit}];
	If[location===$Failed,Return[$Failed]];
	If[MatchQ[location,{{_?NumericQ,_},{_,_}}],
		If[location[[1,1]]==location[[2,1]]||location[[1,2]]==location[[2,2]],
			Message[System`GeomagneticModelData::invreg,loc];Return[$Failed]
	]];
	With[{l=If[MatchQ[loc,_GeoPosition],loc[[1]],loc]},
		userzQ=Which[QuantityQ[l[[-1]]],CompatibleUnitQ[l[[-1]],Quantity["Meters"]],
			MatchQ[l, {_?(Length[#]>1&), _?NumericQ} | {_?(Length[#]>1&), _Quantity}]&&Length[l]===2,True,
			Length[l]>2&&Not[TrueQ[l[[-1]]]],True,
			True,False
		]
	];
	If[userzQ,
		tempht=(loc /. GeoPosition[x_List] :> x)[[-1]];
		If[Not[TrueQ[Element[location[[3]], Reals]]],
			Message[System`GeomagneticModelData::invelev,tempht];
			Return[$Failed]
		];
		If[TrueQ[(location[[3]]<-10000||location[[3]]>600000)],
			tempht=location[[3]]/1000;
			Message[System`GeomagneticModelData::elevrng,tempht];
			Return[Missing["NotAvailable"]]
		];
	];
	If[MatchQ[location,{_,_,_}],
		If[Not[NumericQ[location[[-1]]]||TrueQ[location[[-1]]]],Message[System`GeomagneticModelData::elev];location[[-1]]=0]];
	If[Not[NumericQ[zoom]||MatchQ[zoom,Automatic]],Message[System`GeomagneticModelData::zoom,zoom];zoom=1];
	If[iv===All,
		iGeomagneticModelCompute[location,comp,time,model,GeoZoomLevel->zoom],
		iGeomagneticModelCompute[location,comp,iv,time,model,GeoZoomLevel->zoom]
	]
];
iGeomagneticModelData[pole_,opt:OptionsPattern[]]:=Module[{
	time=getTime[OptionValue["Time"]],model=getModel[OptionValue[Method]]},
	If[time===$Failed,Return[$Failed]];
	If[MemberQ[{"NorthModelDipPoleGeoElevation","SouthModelDipPoleGeoElevation"},pole],
		iPoleFinder[pole,time,model,"Altitude"->Automatic],
		iPoleFinder[pole,time,model,"Altitude"->0]
	]
];

(***CALCULATING CODE***)
Options[iGeomagneticModelCompute]:={GeoZoomLevel->Automatic, "Unitless"->False};
iGeomagneticModelCompute[{lat_?NumericQ,long_?NumericQ,elev_},Array,date_,model_,opt:OptionsPattern[]]:=Module[{pot,field,
	baseyear=pickYear[model,First[DateList[date]]],yeardiff},
	If[baseyear===$Failed,Return[Missing["NotAvailable"]]];
	If[MatchQ[baseyear,_List],
		yeardiff=If[MatchQ["GUFM1",baseyear[[2]]],baseyear[[1]],First[DateDifference[{baseyear[[1]],1},date,"Year"]]];
		baseyear=baseyear[[2]],
		yeardiff=First[DateDifference[{baseyear,1},date,"Year"]];
	];
	field=GeomagneticField2[elev,lat,long,yeardiff,baseyear];
	pot=field[[-1]];
	field=RotateRight[field[[;;3]]];
	{field,pot}
]
iGeomagneticModelCompute[{lat_?NumericQ,long_?NumericQ,elev_},comp_,iv:System`GeoVector|System`GeoVectorENU|System`GeoVectorXYZ,date_DateObject,model_,opts:OptionsPattern[]]:=Module[
	{data,quantity,position,east,north,up},
	data=iGeomagneticModelCompute[{lat,long,elev},Array,date,model,opts];
	(*positions and components*)
	position={lat,long,elev};
	east=data[[1,3]];
	north=data[[1,2]];
	up=-data[[1,1]];
	If[!FreeQ[{position,east,north,up},Missing|$Failed],Return[Missing["NotAvailable"]]];
	If[MemberQ[{"Potential","Magnitude"},comp],
		Missing["NotApplicable"],
		Switch[iv,
			System`GeoVectorENU,
				System`GeoVectorENU[GeoPosition[position]->QuantityArray[{east,north,up},"Nanoteslas"]],
			System`GeoVector,
				data=System`GeoVectorENU[GeoPosition[position]->QuantityArray[{east,north,up},"Nanoteslas"]];
				System`GeoVector[data],
			System`GeoVectorXYZ,
				data=System`GeoVectorXYZ[System`GeoVectorENU[GeoPosition[position]->{east,north,up}]];
				System`GeoVectorXYZ[GeoPosition[position]->QuantityArray[data["Vector"],"Nanoteslas"]],
			_,$Failed
		]
	]
]
iGeomagneticModelCompute[{lat_?NumericQ,long_?NumericQ,elev_},comp_String,date_DateObject,model_,opt:OptionsPattern[]]:=Module[{field,
	baseyear=pickYear[model,First[DateList[date]]],yeardiff,unitlessQ=OptionValue["Unitless"]},
	If[baseyear===$Failed,Return[Missing["NotAvailable"]]];
	If[MatchQ[baseyear,_List],
		yeardiff=If[MatchQ["GUFM1",baseyear[[2]]],baseyear[[1]],First[DateDifference[{baseyear[[1]],1},date,"Year"]]];
		baseyear=baseyear[[2]],
		yeardiff=First[DateDifference[{baseyear,1},date,"Year"]];
	];
	If[comp==="Potential",
		field={1/1000,1,1,1}*GeodeticToGeocentric[{elev,lat,(180+long),yeardiff}];
		field=GeomagneticPotential[baseyear]@@field,
		field=GeomagneticField[elev,lat,long,yeardiff,baseyear];
		field=RotateRight[field]
	];
	If[unitlessQ,
		geoComponent[field,comp],
		If[comp==="Potential",
			geoComponentUnit[geoComponent[field,comp],comp,"Nanovolts"*"Seconds"/"Meters"],
			geoComponentUnit[geoComponent[field,comp],comp,"Nanoteslas"]
		]
	]
];
iGeomagneticModelCompute[{lat_?NumericQ,long_?NumericQ,elev_},All,date_,model_,opt:OptionsPattern[]]:=Module[{pot,field,
	baseyear=pickYear[model,First[DateList[date]]],yeardiff},
	If[baseyear===$Failed,Return[Missing["NotAvailable"]]];
	If[MatchQ[baseyear,_List],
		yeardiff=If[MatchQ["GUFM1",baseyear[[2]]],baseyear[[1]],First[DateDifference[{baseyear[[1]],1},date,"Year"]]];
		baseyear=baseyear[[2]],
		yeardiff=First[DateDifference[{baseyear,1},date,"Year"]];
	];
	field=GeomagneticField2[elev,lat,long,yeardiff,baseyear];
	pot=field[[-1]];
	field=RotateRight[field[[;;3]]];
	Association@@Rule@@@Append[Map[{#,geoComponentUnit[geoComponent[field,#],#,"Nanoteslas"]}&,$GeoVectorComponents[[;;-3]]],
		{"Potential",geoComponentUnit[geoComponent[pot,"Potential"],"Potential","Nanovolts"*"Seconds"/"Meters"]}]
];
iGeomagneticModelCompute[{{lat1_,long1_},{lat2_,long2_},elevation_},comp_?(StringQ[#]||MatchQ[#,All]&),iv:System`GeoVector|System`GeoVectorENU|System`GeoVectorXYZ,date_DateObject,model_,opt:OptionsPattern[]]:=Module[
	{data,east,north,up,positions},
	data=iGeomagneticModelCompute[{{lat1,long1},{lat2,long2},elevation},All,date,model,opt,"Unitless"->True];
	(*positions and components*)
	positions=Flatten[Reverse[data["Position"]],1];
	east=Flatten[data["EastComponent"]];
	north=Flatten[data["NorthComponent"]];
	up=Flatten[-data["DownComponent"]];
	If[!FreeQ[{positions,east,north,up},Missing|$Failed],Return[Missing["NotAvailable"]]];
	If[MemberQ[{"Potential","Magnitude"},comp],
		Missing["NotApplicable"],
		Switch[iv,
			System`GeoVectorENU,
				System`GeoVectorENU[GeoPosition[positions]->
					QuantityArray[Transpose[{east,north,up}],"Nanoteslas"]],
			System`GeoVector,
				data=System`GeoVectorENU[GeoPosition[positions]->
					QuantityArray[Transpose[{east,north,up}],"Nanoteslas"]];
				System`GeoVector[data],
			System`GeoVectorXYZ,
				data=System`GeoVectorXYZ[System`GeoVectorENU[GeoPosition[positions]->Transpose[{east,north,up}]]];
				System`GeoVectorXYZ[GeoPosition[positions]->QuantityArray[data["Vector"],"Nanoteslas"]],
			_,$Failed
		]
	]
]
iGeomagneticModelCompute[{{lat1_,long1_},{lat2_,long2_},elevation_},comp_String,iv_,date_DateObject,model_,opt:OptionsPattern[]]:=Module[
	{data,data2,quantity},
	data=iGeomagneticModelCompute[{{lat1,long1},{lat2,long2},elevation},comp,date,model,opt];
	data2=data;
	data=DeleteCases[DeleteCases[Flatten[QuantityMagnitude[data]],_Missing],_QuantityMagnitude];
	Which[iv===All,
			data2,
		data==={},
			Missing["NotAvailable"],
		iv===Interval,
			quantity=QuantityUnit[data2];
			If[MatchQ[quantity,_List],quantity=quantity[[1,1]]];
			data=Sort[data][[{-1,1}]];
			Quantity[Interval[data],quantity],
			True,
			quantity=QuantityUnit[data2];
			If[MatchQ[quantity,_List],quantity=quantity[[1,1]]];
			Quantity[iv[data],quantity]
	]
];
iGeomagneticModelCompute[{{lat1_,long1_},{lat2_,long2_},elevation_},comp_?(StringQ[#]||MatchQ[#,All]&),date_DateObject,model_,opt:OptionsPattern[]]:=Module[
	{lam,zoom,points,rows,columns,elevationmeters=elevation,
		lat,input,pot,baseyear=pickYear[model,First[DateList[date]]],yeardiff,result,quantity},
	zoom=OptionValue[GeoZoomLevel];
	If[Not[IntegerQ[zoom]||zoom===Automatic],Message[System`GeomagneticModelData::zoom, zoom];zoom=1];
	If[TrueQ[elevation],
		points=geoElevation[{long1,long2},{lat1,lat2},zoom,System`GeomagneticModelData];
		rows=Length[points];
		If[rows===0,Return[Missing["NotAvailable"]],columns=Length[points[[1]]]];
		If[columns===0,Return[Missing["NotAvailable"]]],
		If[MatchQ[zoom,Automatic],zoom=GeoGraphics`EstimateProjectedZooms[{Sort@{long1, long2},Sort@{lat1, lat2}}, {128, 64}, "Wolfram"]];
		points=Quiet[GIS`GeoPositionArray[{Sort@{lat1, lat2}, Sort@{long1, long2}}, GeoZoomLevel -> zoom], {GeoPosition::invpos}][[1]];
		rows=Length[points];
		If[rows===0,Return[Missing["NotAvailable"]],columns=Length[points[[1]]]];
		If[columns===0,Return[Missing["NotAvailable"]]];
		If[elevationmeters<-10000||elevationmeters>600000,
			Message[System`GeomagneticModelData::elevrng,elevation];
			Return[Table[Missing["NotAvailable"],{i,rows},{j,columns}]],
			points=Table[Append[points[[i,j]],0],{i,rows},{j,columns}]
		]
	];
	If[baseyear===$Failed,Return[Table[Missing["NotAvailable"],{j,columns},{i,rows}]]];
	Which[comp==="Potential"&&MatchQ[baseyear,_List],
		yeardiff=If[MatchQ["GUFM1",baseyear[[2]]],baseyear[[1]],First[DateDifference[{baseyear[[1]],1},date,"Year"]]];
		baseyear=baseyear[[2]],
		comp==="Potential",
		yeardiff=First[DateDifference[{baseyear,1},date,"Year"]]
	];
	DynamicModule[{text,progress=0},
		If[TrueQ[$Notebooks],
			text=Row[{"Calculating ",Dynamic[progress]," of ",ToString[rows*columns]," values...."}];
			PrintTemporary[Internal`LoadingPanel[text]]];
		result=Table[progress=rows*(i-1);
			lam=points[[1,i,2]];
			Table[lat=points[[j,1,1]];
				If[comp==="Potential",
					input={1/1000,1,1,1}*GeodeticToGeocentric[{points[[j,i,3]],lat,(180+lam),yeardiff}];
					pot=GeomagneticPotential[baseyear]@@input;
					geoComponent[pot,"Potential"],
					iGeomagneticModelCompute[{lat,lam,points[[j,i,3]]},If[comp===All,Array,comp],date,model,"Unitless"->True]
				],
			{j,rows}],
		{i,columns}];
		progress=rows*columns;
	];
	result=Reverse[Transpose[result]];
	Which[TrueQ[OptionValue["Unitless"]]&&comp===All,
		result = result[[All, All, 1]];
		result = Association[Rule @@@ Map[
			Function[{type}, {type, Map[geoComponent[#, type] &, result, {2}]}],
			{"EastComponent","NorthComponent","DownComponent"}]];
		Append[result,"Position"->QuantityArray[points, 
			{"AngularDegrees", "AngularDegrees", "Meters"}]],
		
		comp===All,
		pot = QuantityArray[result[[All, All, 2]], "Nanovolts"*"Seconds"/"Meters"];
		result = result[[All, All, 1]];
		result = Association[Rule @@@ Append[Map[Function[{type}, {type, geoComponentUnit[
        	Map[geoComponent[#, type] &, result, {2}], type, "Nanoteslas"]}], 
        	$GeoVectorComponents[[;;-3]]], {"Potential", pot}]];
		Append[result,"Position"->QuantityArray[points, 
			{"AngularDegrees", "AngularDegrees", "Meters"}]],
			
		True,
		quantity=If[comp==="Potential","Nanovolts"*"Seconds"/"Meters","Nanoteslas"];
		geoComponentUnit[result,comp,quantity]
	]
];
iGeomagneticModelCompute[___]:=$Failed;

(***GeomagneticModelData System Code***)
Options[System`GeomagneticModelData]:={GeoZoomLevel->Automatic, Method->Automatic};
System`GeomagneticModelData[loc_?loclistQ,args___]:=Module[
	{res=System`GeomagneticModelData[#,args]&/@If[MatchQ[loc,_GeoPosition],loc[[1]],loc]},
	Which[
    	FreeQ[{args}, GeoVector | GeoVectorENU | GeoVectorXYZ] || MatchQ[res, _GeoVector | _GeoVectorENU | _GeoVectorXYZ],
    	    res,  (*head is GeoVector*then we don't need to combine into one GeoVector**)
 
    	MatchQ[res, {_Missing ..}],
    	    Missing["NotAvailable"],
 	
    	res = If[MatchQ[#, _GeoVector | _GeoVectorENU | _GeoVectorXYZ], #[[1]], #] & /@ res;
    	If[MatchQ[res, {__Rule}], res = GeoPosition[res[[All, 1]]] -> QuantityArray[res[[All, 2]]]];
    	!FreeQ[{args}, GeoVectorENU],
    	    GeoVectorENU[res],
    	    
    	!FreeQ[{args}, GeoVectorXYZ],
    	    GeoVectorXYZ[res],
 	
    	True,
    	    GeoVector[res]
 	]
]
(* four args !!*)
System`GeomagneticModelData[loc_,date_?datespecQ,comp_?(MemberQ[$GeoVectorComponents,#]&),iv_?(MemberQ[$ivproperties,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,comp,iv,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,datelist_?datelistQ,comp_?(MemberQ[$GeoVectorComponents,#]&),iv_?(MemberQ[$ivproperties,#]&),opt:OptionsPattern[]]:=
	System`GeomagneticModelData[loc,#,comp,iv,opt]&/@datelist;

(*three args*)
System`GeomagneticModelData[loc_Association,comp_?(MemberQ[$GeoVectorComponents,#]&),iv_?(MemberQ[$ivproperties,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,comp,iv,System`GeomagneticModelData,opt]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,date_?datespecQ,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,All,iv,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,date_?datespecQ,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,comp,All,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,comp_?(MemberQ[$GeoVectorComponents,#]&),iv_?(MemberQ[$ivproperties,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,comp,iv,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,datelist_?datelistQ,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=
	System`GeomagneticModelData[loc,#,comp,opt]&/@datelist;

(*two args*)
System`GeomagneticModelData[loc_Association,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,comp,All,System`GeomagneticModelData,opt]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_Association,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,All,iv,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[datelist_?datelistQ,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=
	System`GeomagneticModelData[#,comp,opt]&/@datelist;
System`GeomagneticModelData[datelist_?datelistQ,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=
	System`GeomagneticModelData[#,iv,opt]&/@datelist;
System`GeomagneticModelData[date_?datespecQ,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[$GeoLocation,comp,All,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[date_?datespecQ,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[$GeoLocation,All,iv,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,comp,All,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,All,iv,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[pole_?(MemberQ[Join[$geopoles,$dippoles],#]&),date_?datespecQ,opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[pole,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[pole_?(MemberQ[Join[$geopoles,$dippoles],#]&),datelist_?datelistQ,opt:OptionsPattern[]]:=
	System`GeomagneticModelData[pole,#,opt]&/@datelist;
System`GeomagneticModelData[loc_,date_?datespecQ,opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,All,All,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_,datelist_?datelistQ,opt:OptionsPattern[]]:=System`GeomagneticModelData[loc,#,opt]&/@datelist;


(*single arg*)
System`GeomagneticModelData["Properties"]:=Sort[Join[$geopoles,$dippoles]];
System`GeomagneticModelData[loc_Association,opt:OptionsPattern[]]:=With[
	{res=iGeoModelDataAssociation[loc,All,All,System`GeomagneticModelData,opt]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[comp_?(MemberQ[$GeoVectorComponents,#]&),opt:OptionsPattern[]]:=System`GeomagneticModelData[$GeoLocation,comp,opt];
System`GeomagneticModelData[date_DateObject?datespecQ,opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[$GeoLocation,All,All,"Time"->date,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[datelist_?datelistQ,opt:OptionsPattern[]]:=System`GeomagneticModelData[#,opt]&/@datelist;
System`GeomagneticModelData[pole_?(MemberQ[Join[$geopoles,$dippoles],#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[pole,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[iv_?(MemberQ[{System`GeoVector,System`GeoVectorENU,System`GeoVectorXYZ},#]&),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[$GeoLocation,All,iv,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];
System`GeomagneticModelData[loc_?(MatchQ[#, Except[_Rule]] &),opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[loc,All,All,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];

(*no args*)
System`GeomagneticModelData[opt:OptionsPattern[]]:=With[
	{res=iGeomagneticModelData[$GeoLocation,All,All,Sequence@@FilterRules[{opt},Options[System`GeomagneticModelData]]]},
	(optionCheck[{opt},System`GeomagneticModelData];res)/;res=!=$Failed
];

With[{s=$ProtectedSymbols},SetAttributes[s,{ReadProtected}]];
Protect@@$ProtectedSymbols;

End[]; (* End Private Context *)
