Begin["DataPaclets`HumanGrowthDataDump`"];

$ProtectedSymbols = {
	System`HumanGrowthData
};

Unprotect@@$ProtectedSymbols;

$resourceDirectory=DirectoryName[$InputFileName](*NotebookDirectory[]*);

getResourceFile[file_String] := If[
	FileExistsQ[#],
	Get[#],
	Message[HumanGrowthData::init];Throw[$Failed,"MissingResource"]
]&[FileNameJoin[{$resourceDirectory,file}]]

(*AbortProtect[getResourceFile/@{"growthcurves.m"}];*)

(* Get Data *)
rawGrowthData=Import[FileNameJoin[{$resourceDirectory,"growthcurves.m"}]];

(*build LMS functions*)
(* sex, property, ethnicity, location (last two not used currently) *)
(HumanGrowthDataLMS[#[[1]], #[[2]], #[[3]],#[[4]]] =
  Interpolation[Developer`ToPackedArray@Transpose[Map[Function[{x}, Flatten[x, 1]], #[[{-2, -1}]]]]])&/@rawGrowthData;

$sexes={"Female","Male",All};
$enthnicities=Union[rawGrowthData[[All,3]]];
$countries=Union[rawGrowthData[[All,4]]];

(*extract valid ranges*)
Which[StringMatchQ[#[[2]],__~~"Age"],
	hgdDomain["Age", #[[1]], #[[2]], #[[3]],#[[4]]]=Quantity[Interval[#], "Months"]&@#[[5, 1, {1, -1}]],
	StringMatchQ[#[[2]],__~~"Length"],
	hgdDomain["Length", #[[1]], #[[2]], #[[3]],#[[4]]]=Quantity[Interval[#], "Centimeters"]&@#[[5, 1, {1, -1}]],
	True,
	hgdDomain["Height", #[[1]], #[[2]], #[[3]],#[[4]]]=Quantity[Interval[#], "Centimeters"]&@#[[5, 1, {1, -1}]]]&/@rawGrowthData;

(* LMS to Percentile or ZScore*)
(* if z is too big and l is negative, then this goes complex, the 1+lsz went through zero, so just use that *)
HumanGrowthDataFromLMS[z_, {l_, m_, s_}, "FromZScore"] := 
    Replace[If[l =!= 0., m*(1 + l*s*z)^(1/l), m*Exp[s*z]], _Complex -> If[l<0,Infinity,0]];
 
HumanGrowthDataFromLMS[p_, {l_, m_, s_}, "FromPercentile"] := 
    HumanGrowthDataFromLMS[PercentileToZ[p], {l, m, s}, "FromZScore"]
 
HumanGrowthDataFromLMS[val_, {l_, m_, s_}, "ToZScore"] := 
    If[l =!= 0., ((val/m)^l - 1)/(l*s), Log[val/m]/s]
 
HumanGrowthDataFromLMS[val_, {l_, m_, s_}, "ToPercentile"] := 
    ZToPercentile[HumanGrowthDataFromLMS[val, {l, m, s}, "ToZScore"]]
 
PercentileToZ[perc_] := Sqrt[2.]*InverseErf[0.02*perc - 1.]
ZToPercentile[z_] := 50*(Erf[z/Sqrt[2.]] + 1)

(* precompute some frequently used ones *)
PercentileToZ[0.1] = -3.0902323061678136;
PercentileToZ[2] = -2.053748910631822;
PercentileToZ[3] = -1.8807936081512506;
PercentileToZ[4] = -1.7506860712521704;
PercentileToZ[5] = -1.6448536269514724;
PercentileToZ[10] = -1.2815515655446001;
PercentileToZ[25] = -0.6744897501960818;
PercentileToZ[50] = 0.;
PercentileToZ[75] = 0.6744897501960818;
PercentileToZ[90] = 1.2815515655446001;
PercentileToZ[95] = 1.6448536269514737;
PercentileToZ[96] = 1.75068607125217;
PercentileToZ[97] = 1.8807936081512506;
PercentileToZ[98] = 2.053748910631822;
PercentileToZ[99.9] = 3.090232306167847;
ZToPercentile[-3] = 0.13498980316301035;
ZToPercentile[-2] = 2.275013194817921;
ZToPercentile[-1] = 15.865525393145708;
ZToPercentile[0] = 50;
ZToPercentile[1] = 84.1344746068543;
ZToPercentile[2] = 97.72498680518208;
ZToPercentile[3] = 99.86501019683699;
(* done precomputing *)

$hgdProperties = {
    "Height",
    "Length",
    "HeadCircumference",
    "Weight",
    "BMI",(*
    "Percentile",
    "ZScore",*)
    "Milestones",
    "NextMilestones",
    "PreviousMilestones"
};

propQuantity[Infinity|0|_?Negative,_,_]:=Missing["NotAvailable"]
propQuantity[x_,"BMI","Metric"]:=Quantity[x,"Kilograms"/"Meters"^2]
propQuantity[x_,"ZScore",_]:=x
propQuantity[x_,"Percentile",_]:=Quantity[x,"Percent"]
propQuantity[x_,"Age",_]:=Quantity[x,"Months"]

propQuantity[x_,"Height"|"Length"|"HeadCircumference","Imperial"]:=UnitConvert[Quantity[x,"Centimeters"],"Inches"]
propQuantity[x_,"Weight","Imperial"]:=UnitConvert[Quantity[x,"Kilograms"],"Pounds"]
propQuantity[x_,"BMI","Imperial"]:=UnitConvert[Quantity[x*703,"Kilograms"/"Meters"^2],"Pounds"/"Inches"^2]

propQuantity[x_,"Height"|"Length"|"HeadCircumference",_]:=Quantity[x,"Centimeters"]
propQuantity[x_,"Weight",_]:=Quantity[x,"Kilograms"]

propQuantity[x_,{"Height"|"Length"|"HeadCircumference","ProbabilityDensity"},"Imperial"]:=UnitConvert[Quantity[x,"Centimeters"^(-1)],"Inches"^(-1)]
propQuantity[x_,{"Height"|"Length"|"HeadCircumference","ProbabilityDensity"},_]:=Quantity[x,"Centimeters"^(-1)]
propQuantity[x_,{"Weight","ProbabilityDensity"},"Imperial"]:=UnitConvert[Quantity[x,"Kilograms"^(-1)],"Pounds"^(-1)]
propQuantity[x_,{"Weight","ProbabilityDensity"},_]:=Quantity[x,"Kilograms"^(-1)]
propQuantity[x_,{"BMI","ProbabilityDensity"},"Imperial"]:=UnitConvert[Quantity[x/703,"Meters"^2/"Kilograms"],"Inches"^2/"Pounds"]
propQuantity[x_,{"BMI","ProbabilityDensity"},_]:=Quantity[x,"Meters"^2/"Kilograms"]

quantitypart[q_,"Height"|"Length"|"HeadCircumference"]:=QuantityMagnitude@UnitConvert[q,"Centimeters"]
quantitypart[q_,"Weight"]:=QuantityMagnitude@UnitConvert[q,"Kilograms"]
quantitypart[q_,"BMI"]:=Module[{units=QuantityUnit[q],metricQ},
	metricQ=QuantityMagnitude@UnitConvert[Quantity[1,units],"Kilograms"/"Meters"^2];
	metricQ=IntegerQ[metricQ]||IntegerQ[1/metricQ];
	If[metricQ,
		QuantityMagnitude@UnitConvert[q,"Kilograms"/"Meters"^2],
		QuantityMagnitude@UnitConvert[q,"Pounds"/"Inches"^2]
	]
]

(* set up mapping from "entity tuples" to the corresponding curve(s) *)
hgdCurve["Height", "Weight"] = {"WeightFromStature"};
hgdCurve["Age", "Weight"] = {"InfantWeightFromAge", "WeightFromAge"};
hgdCurve["Age", "Height"] = {"StatureFromAge"};
hgdCurve["Length", "Weight"] = {"InfantWeightFromLength"};
hgdCurve["Age", "Length"] = {"InfantLengthFromAge"};
hgdCurve["Age", "HeadCircumference"] = {"InfantHeadCircumferenceFromAge"};
hgdCurve["Age", "BMI"] = {"BMIFromAge"};

hgageQ[x_Quantity]:=Module[{age=x/.HoldPattern[ThreadDepth -> _]:>ThreadDepth->Infinity},
	If[ListQ[age],
		hgageQ[age],
		If[CompatibleUnitQ[x,Quantity["Months"]],TrueQ[Element[QuantityMagnitude[x,"Days"], Reals]],False]
	]
]
hgageQ[x_DateObject]:=CompatibleUnitQ[Now-x,Quantity["Months"]]
hgageQ[x_?(VectorQ[#,NumericQ]&)]:=Module[{age=If[Length[x]<=6,DateObject[x],$Failed]},
	hgageQ[age]
]
hgageQ[{}]:=False
hgageQ[x_List] := If[hgageQ[x[[1]]],And @@ (hgageQ /@ x),False]
hgageQ[___]:=False

fgtestQ[x_DateObject]:=Module[{time=QuantityMagnitude[Now-x]},
	If[TrueQ[time<0],Message[System`HumanGrowthData::fdate,x];True,False]
]
fgtestQ[x_Quantity]:=Module[{age=x/.HoldPattern[ThreadDepth -> _]:>ThreadDepth->Infinity},
	If[ListQ[age],
		fgtestQ[age],
		age=QuantityMagnitude[x];
		If[TrueQ[age<0],Message[System`HumanGrowthData::negage,x];True,False]
	]
]
fgtestQ[x_?(VectorQ[#,NumericQ]&)]:=Module[{age=DateObject[x]},
	fgtestQ[age]
]
fgtestQ[x_List] := And @@ (fgtestQ /@ x)
fgtestQ[___]:=False

propQ[All]:=True
propQ[_String]:=True
propQ[_EntityProperty]:=True
propQ[___]:=False

indexQ[_Real]:=True
indexQ[_Integer]:=True
indexQ[HoldPattern[Quantity[_,"Percent",___]]]:=True
indexQ[x_Association]:=KeyExistsQ[x,"Percentage"]||KeyExistsQ[x,"ZScore"]
indexQ[___]:=False

quantitycheckQ[q_Quantity,"Height"|"Length"|"HeadCircumference"]:=CompatibleUnitQ[q,Quantity["Feet"]]
quantitycheckQ[q_Quantity,"Weight"]:=CompatibleUnitQ[q,Quantity["Kilograms"]]
quantitycheckQ[q_Quantity,"BMI"]:=CompatibleUnitQ[q,Quantity["Kilograms"/"Meters"^2]]
(*quantitycheckQ[q_NumericQ,"BMI"]:=True*)
quantitycheckQ[___]:=False

hgentityformat[e_Quantity]:={e,All,$Failed};
hgentityformat[e_Association]:=Module[
	{age=If[KeyExistsQ[e,"Age"],e["Age"]/.(ThreadDepth->_)->ThreadDepth->Infinity,$Failed],
		ethnicity=If[KeyExistsQ[e,"Ethnicity"],e["Ethnicity"],$Failed],
		gender=If[KeyExistsQ[e,"Gender"],e["Gender"],All]/.Entity["Gender",x_]:>x},
	If[MemberQ[{age,gender},$Failed],
		{$Failed,$Failed,$Failed},
		{age,gender,ethnicity}
	]
]
hgentityformat[e_]:={$Failed,$Failed,$Failed}

ageformat[e_List]:=ageformat/@e
ageformat[e_DateList]:=ageformat[DateObject[e]]
ageformat[e_DateObject]:=Now-e
ageformat[e_]:=e

indexformat[z_Association, All, f_]:=Which[
	KeyExistsQ[z,"ZScore"],indexformat[z["ZScore"],"ZScore", f],
	KeyExistsQ[z,"Percentage"],indexformat[z["Percentage"], All, f],
	True,$Failed
]
indexformat[z_?NumericQ,"ZScore",f_]:=If[z>=5||z<=-5,
	Message[f::zscore,z];False,
	z
]
indexformat[Quantity[z_,"Percent"],All,f_]:=Module[{zScore=z},
	If[z>=100||z<=0,
		Message[f::percentile,Quantity[z,"Percent"],Quantity[0,"Percent"],Quantity[100,"Percent"]];
		Return[False]
	];
	PercentileToZ[zScore]
]
indexformat[z_?NumericQ,_,f_]:=Module[{zScore=z},
	If[z>=1||z<=0,
		Message[f::percentile,z,0,1];
		Return[False]
	];
	PercentileToZ[zScore*100]
]
indexformat[___]:=$Failed

Options[System`HumanGrowthData]=SortBy[#,ToString]&@{UnitSystem:>$UnitSystem,Method->Automatic};
System`HumanGrowthData["Properties",opt:OptionsPattern[]] :=Sort[$hgdProperties]
System`HumanGrowthData[e_List,opt:OptionsPattern[]]:=System`HumanGrowthData[#,opt]&/@e
System`HumanGrowthData[e_,opt:OptionsPattern[]] :=With[{res = oHumanGrowthData[e,All,
		Sequence @@ FilterRules[{opt}, Options[System`HumanGrowthData]]]},
    res /; res =!= $Failed
];
System`HumanGrowthData[e_List,p_?(Not[MatchQ[#,_Rule]]&),opt:OptionsPattern[]]:=System`HumanGrowthData[#,p,opt]&/@e
System`HumanGrowthData[e_,p_?(Not[MatchQ[#,_Rule]]&),opt:OptionsPattern[]] :=With[
	{res = If[StringQ[p]||p===All,oHumanGrowthData[e,p,Sequence @@ FilterRules[{opt}, Options[System`HumanGrowthData]]],
		If[MatchQ[p,_Rule],$Failed,oHumanGrowthData[e,All,p,
			Sequence @@ FilterRules[{opt}, Options[System`HumanGrowthData]]]]]},
    res /; res =!= $Failed
];
System`HumanGrowthData[e_List,p_,z_?(Not[MatchQ[#,_Rule]]&),opt:OptionsPattern[]]:=System`HumanGrowthData[#,p,z,opt]&/@e
System`HumanGrowthData[e_,p_,z_?(Not[MatchQ[#,_Rule]]&),opt:OptionsPattern[]] :=With[{res = oHumanGrowthData[e,p,z,
		Sequence @@ FilterRules[{opt}, Options[System`HumanGrowthData]]]},
    res /; res =!= $Failed
];
System`HumanGrowthData[args___,opt:OptionsPattern[]]:=With[{res = If[1<=Length[{args}]<=3,$Failed,
	(Message[General::argb, System`HumanGrowthData, Length[{args}], 1, 3]; $Failed),$Failed]},
    res /; res =!= $Failed
];

Options[oHumanGrowthData]={UnitSystem->$UnitSystem,Method->Automatic};
oHumanGrowthDataEC[entity_,opt:OptionsPattern[]]:=Module[{age,sex,ethnicity,e},
	If[MatchQ[entity, Except[_List]],
		{age,sex,ethnicity}=hgentityformat[entity];
		If[MemberQ[{age,sex},$Failed],Return[$Failed]];
		If[Not[hgageQ[age]],Message[System`HumanGrowthData::notage, age];Return[$Failed]];
		If[fgtestQ[age],Return[$Failed]];
		If[MemberQ[$sexes,sex],entity,
			Message[System`HumanGrowthData::notsex,sex,Entity["Gender", "Male"],Entity["Gender", "Female"]];Return[$Failed]],
		Which[MatchQ[entity,{_Association..}],
			e=oHumanGrowthData/@entity;
			If[FreeQ[e[[All,;;2]],$Failed],entity,$Failed],
			True,
			Message[System`HumanGrowthData::notent, entity, System`HumanGrowthData];Return[$Failed]
		]
	]
]
oHumanGrowthData[entity_,property_?propQ,opt:OptionsPattern[]]:=Module[{e=If[entity==="Properties",$Failed,oHumanGrowthDataEC[entity]],
	p=If[MemberQ[Append[$hgdProperties,All],property],property,Message[System`HumanGrowthData::notprop, property, System`HumanGrowthData];$Failed],
	age,sex,ethnicity,nationality=OptionValue[Method]/.Automatic->("Model"->"CDC"),unit=OptionValue[UnitSystem]},
	If[MemberQ[{e,p},$Failed],Return[$Failed]];
	{age,sex,ethnicity}=hgentityformat[entity];
	If[Not[MemberQ[{"Imperial","Metric"},unit]],Message[System`HumanGrowthData::unit, unit];Return[$Failed]];
	age=ageformat[age];
	ethnicity=Automatic;
	nationality=If[MatchQ[nationality,_Rule]||MatchQ[nationality,{_Rule..}],
		("Model"/.nationality)/.{"Model"->"UnitedStates","CDC"->"UnitedStates"},
		"UnitedStates"
	];
	If[sex===All,
		If[p===All,
			Association["Female"->Association[(#->oHGDCompute[{age,"Female", ethnicity, nationality,Interval},#,unit,False]&)/@$hgdProperties],
				"Male"->Association[(#->oHGDCompute[{age,"Male", ethnicity, nationality,Interval},#,unit,False]&)/@$hgdProperties]
			],
			Association["Female"->oHGDCompute[{age,"Female", ethnicity, nationality,Interval},p,unit,True],
				"Male"->oHGDCompute[{age,"Male", ethnicity, nationality,Interval},p,unit,True]
			]
		],
		If[p===All,
			Association[(#->oHGDCompute[{age,sex, ethnicity, nationality,Interval},#,unit,False]&)/@$hgdProperties],
			oHGDCompute[{age,sex, ethnicity, nationality,Interval},p,unit,True]
		]
	]
]
oHumanGrowthData[entity_,property_?propQ,index_?indexQ,opt:OptionsPattern[]]:=Module[{e=If[entity==="Properties",$Failed,oHumanGrowthDataEC[entity]],
	p=If[MemberQ[Append[$hgdProperties,All],property],property,Message[System`HumanGrowthData::notprop, property, System`HumanGrowthData];$Failed],
	age,sex,ethnicity, nationality=OptionValue[Method]/.Automatic->("Model"->"CDC"),zscore,unit=OptionValue[UnitSystem]},
	If[MemberQ[{e,p},$Failed],Return[$Failed]];
	zscore=indexformat[index,All,System`HumanGrowthData];
	If[zscore===$Failed,Message[System`HumanGrowthData::invindex,index];Return[$Failed]];
	If[zscore===False,Return[Missing["NotAvailable"]]];
	{age,sex,ethnicity}=hgentityformat[entity];
	If[Not[MemberQ[{"Imperial","Metric"},unit]],Message[System`HumanGrowthData::unit, unit];Return[$Failed]];
	age=ageformat[age];
	ethnicity=Automatic;
	nationality=If[MatchQ[nationality,_Rule]||MatchQ[nationality,{_Rule..}],
		("Model"/.nationality)/.{"Model"->"UnitedStates","CDC"->"UnitedStates"},
		"UnitedStates"
	];
	If[sex===All,
		If[p===All,
			Association["Female"->Association[(#->oHGDCompute[{age,"Female", ethnicity, nationality,zscore},#,unit,False]&)/@$hgdProperties],
				"Male"->Association[(#->oHGDCompute[{age,"Male", ethnicity, nationality,zscore},#,unit,False]&)/@$hgdProperties]
			],
			Association["Female"->oHGDCompute[{age,"Female", ethnicity, nationality,zscore},p,unit,True],
				"Male"->oHGDCompute[{age,"Male", ethnicity, nationality,zscore},p,unit,True]
			]
		],
		If[p===All,
			Association[(#->oHGDCompute[{age,sex, ethnicity, nationality,zscore},#,unit,False]&)/@$hgdProperties],
			oHGDCompute[{age,sex, ethnicity, nationality,zscore},p,unit,True]
		]
	]
]
oHumanGrowthData[entity_,property_?propQ,func_String,opt:OptionsPattern[]]:=Module[{arg,
	e=If[entity==="Properties",$Failed,oHumanGrowthDataEC[entity]],
	p=If[MemberQ[Append[$hgdProperties,All],property],property,Message[System`HumanGrowthData::notprop, property, System`HumanGrowthData];$Failed],
	age,sex,ethnicity, nationality=OptionValue[Method]/.Automatic->("Model"->"CDC"),unit=OptionValue[UnitSystem]},
	arg=If[MemberQ[{"StandardDeviation","Distribution"},func],func,Message[System`HumanGrowthData::ncomp, func];$Failed];
	If[MemberQ[{e,p,arg},$Failed],Return[$Failed]];
	{age,sex,ethnicity}=hgentityformat[entity];
	If[Not[MemberQ[{"Imperial","Metric"},unit]],Message[System`HumanGrowthData::unit, unit];Return[$Failed]];
	age=ageformat[age];
	ethnicity=Automatic;
	nationality=If[MatchQ[nationality,_Rule]||MatchQ[nationality,{_Rule..}],
		("Model"/.nationality)/.{"Model"->"UnitedStates","CDC"->"UnitedStates"},
		"UnitedStates"
	];
	If[sex===All,
		Association["Female"->oHGDCompute[{age,"Female", ethnicity, nationality,arg},p,unit,True],
			"Male"->oHGDCompute[{age,"Male", ethnicity, nationality,arg},p,unit,True]
		],
		oHGDCompute[{age,sex, ethnicity, nationality,arg},p,unit,True]
	]
]
oHumanGrowthData[entity_,property_?propQ,quantity_,opt:OptionsPattern[]]:=Module[{e=If[entity==="Properties",$Failed,oHumanGrowthDataEC[entity]],
	p=If[MemberQ[Append[$hgdProperties,All],property],property,Message[System`HumanGrowthData::notprop, property, System`HumanGrowthData];$Failed],
	age,sex,ethnicity, nationality=OptionValue[Method]/.Automatic->("Model"->"CDC"),unit=OptionValue[UnitSystem]},
	If[MemberQ[{e,p},$Failed],Return[$Failed]];
	If[Not[quantitycheckQ[quantity,p]],If[p===All,
			Message[System`HumanGrowthData::invindex,quantity],
			Message[System`HumanGrowthData::punit, quantity, p]
		];Return[$Failed]
	];
	If[p===All,Return[$Failed]];(*should never use this*)
	{age,sex,ethnicity}=hgentityformat[entity];
	If[Not[MemberQ[{"Imperial","Metric"},unit]],Message[System`HumanGrowthData::unit, unit];Return[$Failed]];
	age=ageformat[age];
	ethnicity=Automatic;
	nationality=If[MatchQ[nationality,_Rule]||MatchQ[nationality,{_Rule..}],
		("Model"/.nationality)/.{"Model"->"UnitedStates","CDC"->"UnitedStates"},
		"UnitedStates"
	];
	If[sex===All,
		Association["Female"->oHGDCompute[{age,"Female", ethnicity, nationality,quantity},p,unit,True],
			"Male"->oHGDCompute[{age,"Male", ethnicity, nationality,quantity},p,unit,True]
		],
		oHGDCompute[{age,sex, ethnicity, nationality,quantity},p,unit,True]
	]
]
oHumanGrowthData[args___]:=(System`Private`Arguments[System`HumanGrowthData[args], {1,3}];$Failed)

Clear[oHGDCompute]
oHGDCompute[{age_,sex_, ethnicity_, nationality_, zscore_},p:"Milestones",unit_,ms_]:=Module[
	{physical,cognitive,language,social,a=Ceiling[N[QuantityMagnitude[age,"Days"]]], gendered,ml=sex/.masterindex},
	(* get data *)
	physical = a/.MilestoneData["Physical"];
	cognitive = a/.MilestoneData["Cognitive"];
	language = a/.MilestoneData["Language"];
	social = a/.MilestoneData["Social"];
	gendered = a/.(sex/.gendereventData);
	physical=Flatten[{physical,gendered,a/.MilestoneData["Teeth"]}];
	ml=Sort[DeleteCases[Flatten[#],_?NumericQ]]&/@{physical,cognitive,language,social};
    Association[MapThread[Rule,{{"Physical","Cognitive","Language","Social"},ml/.{}->Missing["NotAvailable"]}]]
]
oHGDCompute[{age_,sex_, ethnicity_, nationality_, zscore_},p:"NextMilestones"|"PreviousMilestones",unit_,ms_]:=Module[
	{physical,cognitive,language,social,a=Ceiling[N[QuantityMagnitude[age,"Days"]]], gendered,pl=sex/.physicalindex},
	(*get periods for each category so no category is missing at young ages*)
	a=selectperiod[#,a,p]&/@{pl,cognitiveindex,languageindex,socialindex};
	(* get data *)
	physical = a[[1]]/.MilestoneData["Physical"];
	cognitive = a[[2]]/.MilestoneData["Cognitive"];
	language = a[[3]]/.MilestoneData["Language"];
	social = a[[4]]/.MilestoneData["Social"];
	gendered = a[[1]]/.(sex/.gendereventData);
	physical=Flatten[{physical,gendered,a[[1]]/.MilestoneData["Teeth"]}];
	pl=Sort[DeleteCases[Flatten[#],_?NumericQ]]&/@{physical,cognitive,language,social};
    Association[MapThread[Rule,{{"Physical","Cognitive","Language","Social"},pl/.{}->Missing["NotAvailable"]}]]
]

selectperiod[list_,a_,"NextMilestones"]:=Module[{res=Select[list, a < # &, 1]},If[res==={},0,res]]
selectperiod[list_,a_,"PreviousMilestones"]:=Module[{res=Select[Reverse[list], a > # &, 2]},If[res==={},0,Last[res]]]

oHGDCompute[{age_,sex_, ethnicity_, nationality_, zscore_},p_,unit_,ms_]:=Module[{curve,range,pos,a},
	(*determine curve if multiple and cut out cases that out of range *)
	curve=hgdCurve["Age", p];
	range=hgdDomain["Age", sex, #, ethnicity, nationality]&/@curve;
	a=QuantityMagnitude[UnitConvert[age,"Months"]];
	
	(* get data *)
	If[ListQ[a],(* to return missings for values out of bounds *)
		(pos=FirstPosition[Function[{x},IntervalMemberQ[x,#]]/@QuantityMagnitude[range],True];
			If[MemberQ[{{}, Missing["NotFound"]}, pos],
				Missing["NotAvailable"],
				pos=curve[[First@pos]];
				oHGDResult[{sex, pos, ethnicity, nationality,p,unit,zscore},#]
			])&/@a,
		pos=FirstPosition[IntervalMemberQ[#,age]&/@range,True];
		If[MemberQ[{{}, Missing["NotFound"]}, pos],
			If[ms,Message[System`HumanGrowthData::range, "Age", Chop/@IntervalUnion@@range]];
			Return[Missing["NotAvailable"]],
			curve=curve[[First@pos]]
		];
		oHGDResult[{sex, curve, ethnicity, nationality,p,unit,zscore},a]
	]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,"Weight",unit_,"StandardDeviation"},age_]:=Module[{data,values,cdf,
	sol,mu,sigma,
	ratio=QuantityMagnitude@UnitConvert[propQuantity[1,"Weight",unit],QuantityUnit[propQuantity[1,"Weight",unit]]]},
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	values = Table[{HumanGrowthDataFromLMS[x, data, "FromPercentile"],x},{x, 0.1, 99.9,0.1}];
	cdf = Interpolation[{#1*ratio, #2/100} & @@@ values];
	With[{v1=QuantityMagnitude@values[[1,1]]*ratio,v2=QuantityMagnitude@values[[-1,1]]*ratio},
		sol = FindFit[Table[{x, cdf'[x]}, {x, v1, v2, 0.1}], 
  			PDF[LogNormalDistribution[mu, sigma], x], {mu, sigma}, x];
  		propQuantity[sigma/.sol,"Weight",unit]
	]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,property_,unit_,"StandardDeviation"},age_]:=Module[{data,values,sol},
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	sol=HumanGrowthDataFromLMS[0, data, "FromZScore"];
	values=HumanGrowthDataFromLMS[1, data, "FromZScore"]-sol;
	propQuantity[values,property,unit]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,"Weight",unit_,"Distribution"},age_]:=Module[{data,values,cdf,
	sol,mu,sigma,
	ratio=QuantityMagnitude@UnitConvert[propQuantity[1,"Weight",unit],QuantityUnit[propQuantity[1,"Weight",unit]]]},
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	values = Table[{HumanGrowthDataFromLMS[x, data, "FromPercentile"],x},{x, 0.1, 99.9,0.1}];
	cdf = Interpolation[{#1*ratio, #2/100} & @@@ values];
	With[{v1=QuantityMagnitude@values[[1,1]]*ratio,v2=QuantityMagnitude@values[[-1,1]]*ratio},
		sol = FindFit[Table[{x, cdf'[x]}, {x, v1, v2, 0.1}], 
  			PDF[LogNormalDistribution[mu, sigma], x], {mu, sigma}, x];
  		LogNormalDistribution[mu, sigma]/.sol
	]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,property_,unit_,"Distribution"},age_]:=Module[{data,values,sol,
	ratio=QuantityMagnitude@UnitConvert[propQuantity[1,property,unit],QuantityUnit[propQuantity[1,property,unit]]]},
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	sol=HumanGrowthDataFromLMS[0, data, "FromZScore"];
	values=HumanGrowthDataFromLMS[1, data, "FromZScore"]-sol;
	NormalDistribution[sol*ratio, values*ratio]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,p_,unit_,quantity_Quantity},age_]:=Module[{data,values,cdf,q},
	q=quantitypart[quantity,p];
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	values = Table[{HumanGrowthDataFromLMS[x, data, "FromPercentile"],x},{x, 0.1, 99.9,0.1}];
	If[IntervalMemberQ[Interval[values[[{1,-1},1]]],q],
		cdf = Interpolation[{#1, #2/100} & @@@ values];
		Association["Percentile"->Quantity[cdf[q]*100,"Percent"],"ProbabilityDensity"->propQuantity[cdf'[q],{p,"ProbabilityDensity"},unit]],
		Message[System`HumanGrowthData::qrange,quantity,propQuantity[values[[1,1]],p,unit],propQuantity[values[[-1,1]],p,unit]];
		Missing["NotAvailable"]
	]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,p_,unit_,Interval},age_]:=Module[{data,pm1,p1},
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	pm1=HumanGrowthDataFromLMS[-1, data, "FromZScore"];
	p1=HumanGrowthDataFromLMS[1, data, "FromZScore"];
	propQuantity[Interval[{pm1,p1}],p,unit]
]
oHGDResult[{sex_, curve_, ethnicity_, nationality_,p_,unit_,zscore_?NumericQ},age_]:=Module[{data},
	data=HumanGrowthDataLMS[sex, curve, ethnicity, nationality][age];
	data=HumanGrowthDataFromLMS[zscore, data, "FromZScore"];
	propQuantity[data,p,unit]
]

myFirst[x_,___]:=x

generateMilestoneRanges[q_Association]:=generateMilestoneRanges[q["TypicalAgeRange"]]
generateMilestoneRanges[q_Quantity]:=Module[{mag,unit=QuantityUnit[q]},
	mag=If[CompatibleUnitQ[q,"Days"],Round[QuantityMagnitude[N[q],"Days"]],Return[$Failed]];
	unit=unit/.{"Years"->365,"Months"->30,_->0}; (*determine addition*)
	If[MatchQ[mag,_Interval],
		mag=First[mag];
		Alternatives@@Range[mag[[1]],mag[[2]]+unit],
		Alternatives@@Range[mag,mag+unit]
	]
]
generateMilestoneRanges[___]:=$Failed

MilestoneData["Teeth"]=({generateMilestoneRanges[#]->#}&/@{
	<|"TypicalAgeRange" -> Quantity[Interval[{6, 10}], "Months"], "Milestones" -> {"first teeth (primary lower middle incisors) erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{8, 12}], "Months"], "Milestones" -> {"primary upper middle incisors erupt"}|>,
	<|"TypicalAgeRange" -> Quantity[Interval[{9, 13}], "Months"], "Milestones" -> {"primary upper lateral incisors erupt"}|>,
	<|"TypicalAgeRange" -> Quantity[Interval[{10, 16}], "Months"], "Milestones" -> {"primary lower lateral incisors erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{13, 19}], "Months"], "Milestones" -> {"primary upper first molars erupt"}|>,
	<|"TypicalAgeRange" -> Quantity[Interval[{14, 18}], "Months"], "Milestones" -> {"primary lower first molars erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{16, 22}], "Months"], "Milestones" -> {"primary upper canines erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{17, 23}], "Months"], "Milestones" -> {"primary lower canines erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{23, 31}], "Months"], "Milestones" -> {"primary bottom second molars erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{25, 33}], "Months"], "Milestones" -> {"primary top second molars erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{6, 7}], "Years"], "Milestones" -> {"permanent lower and upper first molars erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{7, 8}], "Years"], "Milestones" -> {"permanent lower lateral incisors erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{8, 9}], "Years"], "Milestones" -> {"permanent upper lateral incisors erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{9, 10}], "Years"], "Milestones" -> {"permanent lower canines erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{9, 11}], "Years"], "Milestones" -> {"primary upper and lower first molars shed"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{9, 12}], "Years"], "Milestones" -> {"primary lower canines shed"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{10, 11}], "Years"], "Milestones" -> {"permanent upper first premolars (first bicuspids) erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{10, 12}], "Years"], "Milestones" -> {"permanent lower first premolars (first bicuspids) erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{11, 12}], "Years"], "Milestones" -> {"permanent lower second premolars (second bicuspids) erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{11, 13}], "Years"], "Milestones" -> {"permanent lower second molars erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{12, 13}], "Years"], "Milestones" -> {"permanent upper second molars erupt"}|>, 
	<|"TypicalAgeRange" -> Quantity[Interval[{17, 21}], "Years"], "Milestones" -> {"permanent upper and lower third molars (wisdom teeth) erupt"}|>});

MilestoneData["Physical"]=({generateMilestoneRanges[#]->#}&/@{
	<|"TypicalAgeRange" -> Quantity[Interval[{2, 5}], "Days"], "Milestones" -> {"eats well", 
		"sucks, swallows and breathes without difficulty"}|>,
   	<|"TypicalAgeRange" -> Quantity[Interval[{2, 4}], "Weeks"], "Milestones" -> {"awake for one-hour periods",
   		"extremities move equally", "lifts chin off surface"}|>,
	<|"TypicalAgeRange" -> Quantity[1, "Months"], "Milestones" -> {"lifts head while on stomach"}|>,
	<|"TypicalAgeRange" -> Quantity[2, "Months"], "Milestones" -> {"diminished newborn reflexes", 
		"follows objects across field of vision", "gaze follows past midline", "hands open 50% of the time", 
		"holds head up for short periods of time", "lifts head and chest off surface", 
		"makes smoother movements with arms and legs", "symmetric movement"}|>,
    <|"TypicalAgeRange" -> Quantity[3, "Months"], "Milestones" -> {"holds head steady"}|>,
	<|"TypicalAgeRange" -> Quantity[4, "Months"], "Milestones" -> {"bears weight on legs", "brings hands together", 
		"brings hands to mouth", "can hold a toy and shake it, as well as swing at dangling toys", 
		"no head lag when pulled to sitting position", "rolls from front onto back", "sleeps through the night", 
		"uses arms to push chest off surface", "when lying on stomach, pushes up to elbows"}|>,
    <|"TypicalAgeRange" -> Quantity[5, "Months"], "Milestones" -> {"plays with hands and feet"}|>,
	<|"TypicalAgeRange" -> Quantity[6, "Months"], "Milestones" -> {"rakes small objects",
		"rocks back and forth, sometimes crawling backward before moving forward", 
		"rolls over in both directions (front to back, back to front)", "sits briefly leaning forward", 
		"stands when placed", "when standing, supports weight on legs and might bounce"}|>, 
	<|"TypicalAgeRange" -> Quantity[7, "Months"], "Milestones" -> {"drags objects to self", "sits without support"}|>,
	<|"TypicalAgeRange" -> Quantity[9, "Months"], "Milestones" -> {"can get into sitting position", "crawls and/or creeps", 
		"feeds self with fingers", "immature pencil grasp", "pulls to stand", "stands, holding on"}|>, 
    <|"TypicalAgeRange" -> Quantity[10, "Months"], "Milestones" -> {"crawls well", "pincer grasp"}|>,
	<|"TypicalAgeRange" -> Quantity[11, "Months"], "Milestones" -> {"cruises", "stands alone for a few seconds"}|>,
	<|"TypicalAgeRange" -> Quantity[12, "Months"], "Milestones" -> {"drinks from a cup", "gets to a sitting position without help",
		"neat pincer grasp", "rolls a ball back to examiner", "stands well alone", "walks holding onto furniture", "walks unassisted"}|>, 
	<|"TypicalAgeRange" -> Quantity[15, "Months"], "Milestones" -> {"plays with ball", "walks backward", 
		"walks well, stoops and climbs stairs"}|>,
	<|"TypicalAgeRange" -> Quantity[16, "Months"], "Milestones" -> {"drinks with minimal spilling", "turns pages of a book"}|>,
	<|"TypicalAgeRange" -> Quantity[18, "Months"], "Milestones" -> {"can help undress self", "dumps raisins from bottle", 
		"eats with a spoon", "pulls toys while walking", "turns pages without ripping them", "walks up steps"}|>, 
   	<|"TypicalAgeRange" -> Quantity[19, "Months"], "Milestones" -> {"runs", "throws a ball", "uses a spoon and fork"}|>,
   	<|"TypicalAgeRange" -> Quantity[20, "Months"], "Milestones" -> {"dumps object in imitation", "takes off own clothes"}|>,
	<|"TypicalAgeRange" -> Quantity[21, "Months"], "Milestones" -> {"walks up stairs"}|>,
	<|"TypicalAgeRange" -> Quantity[22, "Months"], "Milestones" -> {"kicks ball forward"}|>,
	<|"TypicalAgeRange" -> Quantity[24, "Months"], "Milestones" -> {"balances on one foot for one second", 
		"climbs onto and down from furniture without help", "jumps in place", "runs well", "stands on tiptoe", 
		"throws ball overhand", "turns single pages", "walks up and down stairs holding on"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{25, 26}], "Months"], "Milestones" -> {"walks with smooth heel-to-toe motion"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{27, 28}], "Months"], "Milestones" -> {"jumps with both feet", "opens doors"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{29, 30}], "Months"], "Milestones" -> {"draws vertical line"}|>, 
    <|"TypicalAgeRange" -> Quantity[30, "Months"], "Milestones" -> {"brushes teeth with help"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{31, 32}], "Months"], "Milestones" -> {"draws circles"}|>,
	<|"TypicalAgeRange" -> Quantity[36, "Months"], "Milestones" -> {"balances on one foot for 10 seconds", "buttons clothes", 
		"climbs well", "pedals a tricycle", "runs easily", "toilet-trained during day", 
		"walks up and down stairs, one foot on each step"}|>, 
	<|"TypicalAgeRange" -> Quantity[48, "Months"], "Milestones" -> {"age when toilet trained", "brushes teeth independently",
		"catches a bounced ball most of the time", "hops", "pours, cuts with supervision and mashes own food"}|>,
	<|"TypicalAgeRange" -> Quantity[60, "Months"], "Milestones" -> {"able to tie a knot", "can do a somersault", 
		"can use the toilet on his/her own", "has proper pencil grasp", "skips", 
		"stands on one foot for 10 seconds or longer", "swings and climbs", 
		"uses a fork and spoon, and sometimes a table knife", "zips clothes"}|>});
	
MilestoneData["Cognitive"]=({generateMilestoneRanges[#]->#}&/@{
	<|"TypicalAgeRange" -> Quantity[Interval[{2, 5}], "Days"], "Milestones" -> {"follows faces"}|>,
   	<|"TypicalAgeRange" -> Quantity[Interval[{2, 4}], "Weeks"], "Milestones" -> {"fixes on faces"}|>,
	<|"TypicalAgeRange" -> Quantity[1, "Months"], "Milestones" -> {"stares at faces"}|>,
	<|"TypicalAgeRange" -> Quantity[2, "Months"], "Milestones" -> {
		"begins to follow things with eyes and recognize people at a distance", "can indicate boredom", "pays attention to faces"}|>,
    <|"TypicalAgeRange" -> Quantity[3, "Months"], "Milestones" -> {"visually tracks moving objects"}|>,
	<|"TypicalAgeRange" -> Quantity[4, "Months"], "Milestones" -> {"easily distracted/excited by discovery of outside world", 
		"follows moving things with eyes from side to side", "lets you know if happy or sad", "reaches for toy with one hand", 
		"recognizes familiar people and things at a distance", "responds to affection", 
		"uses hands and eyes together, such as seeing a toy and reaching for it", "visual track 180 degrees", "watches faces closely"}|>, 
    <|"TypicalAgeRange" -> Quantity[5, "Months"], "Milestones" -> {"distinguishes between bold colors"}|>,
	<|"TypicalAgeRange" -> Quantity[6, "Months"], "Milestones" -> {"brings things to mouth", "passes objects from hand to hand", 
		"puts objects in mouth", "shows curiosity about things and tries to get things that are out of reach", 
		"visually explores surroundings"}|>,
	<|"TypicalAgeRange" -> Quantity[9, "Months"], "Milestones" -> {"looks at books", "looks for things he/she sees you hide", 
		"moves things smoothly from one hand to the other", "object permanence", 
		"picks up small objects between thumb and index finger", "watches the path of something as it falls"}|>,
    <|"TypicalAgeRange" -> Quantity[11, "Months"], "Milestones" -> {"correctly identifies mother and father"}|>,
	<|"TypicalAgeRange" -> Quantity[12, "Months"], "Milestones" -> {"bangs two things together", "copies gestures", 
		"explores things in different ways", "finds hidden things easily", "follows gaze", 
		"follows simple directions like \"pick up the toy\"", "imitates simple daily tasks", "lets things go without help", 
		"looks at the right picture or thing when it's named", "pokes with index finger", "puts things in a container", 
		"scribbles spontaneously", "starts to use things correctly; for example, drinks from a cup, brushes hair", 
		"will \"read\" board books on own"}|>,
	<|"TypicalAgeRange" -> Quantity[14, "Months"], "Milestones" -> {"eats with fingers", "empties containers of contents", 
		"imitates others"}|>,
	<|"TypicalAgeRange" -> Quantity[15, "Months"], "Milestones" -> {"has definite preferences", 
		"points to body part on request", "stacks two blocks", "understands and follows simple commands"}|>,
	<|"TypicalAgeRange" -> Quantity[18, "Months"], "Milestones" -> {"can follow one-step verbal commands without any gestures", 
		"knows what ordinary things are for", "names favorite book", "points to get the attention of others", "scribbles well", 
		"shows interest in a doll or stuffed animal by pretending to feed", "stacks three to four blocks"}|>,
    <|"TypicalAgeRange" -> Quantity[21, "Months"], "Milestones" -> {"sets simple goals"}|>, 
    <|"TypicalAgeRange" -> Quantity[22, "Months"], "Milestones" -> {"follows two-step commands"}|>,
	<|"TypicalAgeRange" -> Quantity[23, "Months"], "Milestones" -> {"names simple pictures in a book"}|>,
	<|"TypicalAgeRange" -> Quantity[24, "Months"], "Milestones" -> {"begins to sort shapes and colors", "colors with crayons", 
		"completes sentences and rhymes in familiar books", "finds things even when hidden under two or three covers", 
		"plays simple make-believe games", "stacks five or more blocks"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{25, 26}], "Months"], "Milestones" -> {"stacks six blocks"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{29, 30}], "Months"], "Milestones" -> {"washes and dries hands"}|>,
    <|"TypicalAgeRange" -> Quantity[30, "Months"], "Milestones" -> {"knows correct animal sounds", "points to six body parts"}|>,
	<|"TypicalAgeRange" -> Quantity[36, "Months"], "Milestones" -> {"builds towers of more than six blocks", 
		"can work toys with buttons, levers and moving parts", "can copy a cross", "can draw a person", 
		"comprehends cold, hungry and tired", "copies a circle with pencil or crayon", "counts to five", 
		"does puzzles with three or four pieces", "feeds self", "plays make-believe with dolls, animals and people", 
		"recognizes three or four colors", "screws and unscrews jar lids or turns door handles", 
		"turns book pages one at a time", "understands what \"two\" means"}|>,
	<|"TypicalAgeRange" -> Quantity[48, "Months"], "Milestones" -> {"counts to seven", 
		"draws a person with two to four body parts", "knows about food and appliances in home", "matches colors and shapes", 
		"names some colors and numbers", "plays board or card games", "remembers parts of a story", 
		"starts to copy some capital letters", "starts to understand time", 
		"tells you what he/she thinks is going to happen next in a book", "understands the idea of counting", 
		"understands the idea of \"same\" and \"different\"", "uses scissors"}|>, 
    <|"TypicalAgeRange" -> Quantity[60, "Months"], "Milestones" -> {"can draw a person with at least six body parts", 
    	"can print some letters or numbers", "copies a triangle and other geometric shapes", "counts 10 or more things", 
    	"is attentive", "knows about things used every day, like money and food", 
    	"listens to and follows instructions from authority figures", "recognizes letters of alphabet"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{6, 11}], "Years"], "Milestones" -> {"reading and doing math at grade level"}|>, 
   	<|"TypicalAgeRange" ->Quantity[Interval[{18, 21}], "Years"], "Milestones" -> {"demonstrates abstract thinking", 
   		"expresses philosophical/idealistic thoughts"}|>});
	
MilestoneData["Language"]=({generateMilestoneRanges[#]->#}&/@{
	<|"TypicalAgeRange" -> Quantity[Interval[{2, 5}], "Days"], "Milestones" -> {"turns to voice"}|>,
   	<|"TypicalAgeRange" -> Quantity[Interval[{2, 4}], "Weeks"], "Milestones" -> {"responds to sound"}|>,
	<|"TypicalAgeRange" -> Quantity[2, "Months"], "Milestones" -> {"attentive to voices", "coos, makes gurgling sounds", 
		"turns head toward sounds", "vocalizes"}|>, 
	<|"TypicalAgeRange" -> Quantity[4, "Months"], "Milestones" -> {"babbles", "coos when talked to", "copies sounds heard", 
		"cries in different ways to show hunger, pain or being tired", "indicates pleasure and displeasure", 
		"turns toward voices"}|>, 
	<|"TypicalAgeRange" -> Quantity[6, "Months"], "Milestones" -> {"begins to use consonant sounds", 
		"enjoys vocal turn-taking", "imitates sounds", "makes sounds to show joy and displeasure", "makes vowel sounds", 
		"responds to own name", "responds to sounds by making sounds"}|>, 
    <|"TypicalAgeRange" -> Quantity[8, "Months"], "Milestones" -> {"says \"mama\" and \"dada\" to both parents"}|>,
    <|"TypicalAgeRange" -> Quantity[9, "Months"], "Milestones" -> {"copies sounds and gestures of others", "jabbers", 
    	"makes a lot of different sounds", "makes purposeful sounds", "understands \"no\"", "uses fingers to point at things"}|>, 
    <|"TypicalAgeRange" -> Quantity[10, "Months"], "Milestones" -> {"waves goodbye"}|>, 
    <|"TypicalAgeRange" -> Quantity[12, "Months"], "Milestones" -> {"indicates wants with gestures", 
    	"makes sounds with changes in tone", "responds to simple spoken requests", "says \"mama\" or \"dada\" specifically", 
    	"tries to say words you say", "uses simple gestures, like shaking head \"no\""}|>, 
    <|"TypicalAgeRange" -> Quantity[13, "Months"], "Milestones" -> {"uses two words skillfully"}|>, 
    <|"TypicalAgeRange" -> Quantity[14, "Months"], "Milestones" -> {"uses three words regularly", "uses words to communicate"}|>,
    <|"TypicalAgeRange" -> Quantity[15, "Months"], "Milestones" -> {"says \"no\"", "listens to a story"}|>,
    <|"TypicalAgeRange" -> Quantity[17, "Months"], "Milestones" -> {"uses six words regularly"}|>,
    <|"TypicalAgeRange" -> Quantity[18, "Months"], "Milestones" -> {"combines two different words", "names two colors", 
    	"points to show someone what he/she wants", "says and shakes head \"no\"", "vocabulary of 7 to 20 words"}|>, 
    <|"TypicalAgeRange" -> Quantity[23, "Months"], "Milestones" -> {"uses 50 to 70 words"}|>,
    <|"TypicalAgeRange" -> Quantity[24, "Months"], "Milestones" -> {"follows simple instructions", 
    	"knows names of familiar people", "names at least six body parts", "points to things in a book", 
    	"points to things or pictures when they are named", "repeats words overheard in conversation", 
    	"uses two- to three-word sentences", "uses plurals"}|>, 
    <|"TypicalAgeRange" -> Quantity[30, "Months"], "Milestones" -> {"other people understand half of spoken words"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{31, 32}], "Months"], "Milestones" -> {"recites own name"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{33, 34}], "Months"], "Milestones" -> {"carries on a simple conversation", 
    	"names one friend"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{35, 36}], "Months"], "Milestones" -> {"describes how objects are used", 
    	"names two actions", "uses three to four words in a sentence"}|>,
    <|"TypicalAgeRange" -> Quantity[36, "Months"], "Milestones" -> {"can name most familiar things", 
    	"can speak multiple sentences", "carries on a conversation using two to three sentences", 
    	"follows instructions with two or three steps", "most spoken words are understandable", 
    	"says first name, age and sex", "understands words like \"in\", \"on\" and \"under\"", "uses prepositions", 
    	"uses pronouns"}|>,
	<|"TypicalAgeRange" -> Quantity[48, "Months"], "Milestones" -> {"asks/answers using question words", 
		"cooperates with other children", "enjoys doing new things", "follows two unrelated directions", 
		"is more and more creative with make-believe play", "names four colors", "names three shapes", 
		"often can't tell what's real and what's make-believe", "plays \"Mom\" and \"Dad\"", 
		"talks about what he/she likes and is interested in", "would rather play with other children than by him/herself"}|>,
	<|"TypicalAgeRange" -> Quantity[60, "Months"], "Milestones" -> {"articulates clearly", "says name and address",
		 "speaks very clearly", "tells a simple story using full sentences", "uses future tense"}|>});	
	
MilestoneData["Social"]=({generateMilestoneRanges[#]->#}&/@{
	<|"TypicalAgeRange" -> Quantity[Interval[{2, 5}], "Days"], "Milestones" -> {"calms to voice"}|>,
   	<|"TypicalAgeRange" -> Quantity[Interval[{2, 4}], "Weeks"], "Milestones" -> {"can be calmed", "starting to smile"}|>,
	<|"TypicalAgeRange" -> Quantity[2, "Months"], "Milestones" -> {"can briefly calm self", "has a social smile", 
		"looks at parents"}|>, 
    <|"TypicalAgeRange" -> Quantity[4, "Months"], "Milestones" -> {"calms self", 
    	"copies some movements and facial expressions", "elicits attention", 
    	"likes to play with people and might cry when playing stops", "shows interest in mirror images", 
    	"smiles and laughs"}|>,
	<|"TypicalAgeRange" -> Quantity[6, "Months"], "Milestones" -> {"distinguishes emotions by tone of voice", 
		"enjoys personal interactions", "knows familiar faces", "likes to look at self in a mirror", 
		"likes to play with others", "plays peekaboo", "responds to other people's emotions and often seems happy"}|>, 
    <|"TypicalAgeRange" -> Quantity[9, "Months"], "Milestones" -> {"has favorite toys", "may be afraid of strangers", 
    	"may be clingy with familiar adults", "plays pat-a-cake", "seeks parents for comfort"}|>, 
    <|"TypicalAgeRange" -> Quantity[12, "Months"], "Milestones" -> {"cries when parent leaves", 
    	"has favorite things and people", "imitates others' activities", "is shy or nervous with strangers", 
    	"offers a book to read", "prefers primary caregiver", "puts out arm or leg to help with dressing", 
    	"repeats sounds or actions to get attention", "shows fear in some situations"}|>, 
    <|"TypicalAgeRange" -> Quantity[15, "Months"], "Milestones" -> {"brings and shows toys", "has a wide range of emotions", 
    	"initiates social interactions", "tries to do what parents do"}|>,
    <|"TypicalAgeRange" -> Quantity[16, "Months"], "Milestones" -> {"becomes attached to a toy", "has temper tantrums"}|>, 
 	<|"TypicalAgeRange" -> Quantity[17, "Months"], "Milestones" -> {"enjoys pretend games", "likes riding toys"}|>,
    <|"TypicalAgeRange" -> Quantity[18, "Months"], "Milestones" -> {"explores alone but with parent close by", 
    	"helps with simple tasks", "laughs in response to others", "likes to hand things to others as play", 
		"may be afraid of strangers", "may cling to caregivers in new situations", "points to show others something interesting", 
		"shows affection to familiar people"}|>, 
    <|"TypicalAgeRange" -> Quantity[19, "Months"], "Milestones" -> {"helps around the house"}|>,
    <|"TypicalAgeRange" -> Quantity[20, "Months"], "Milestones" -> {"feeds doll"}|>, 
    <|"TypicalAgeRange" -> Quantity[24, "Months"], "Milestones" -> {"copies others", "gets excited when with other children", 
    	"mimics parents' activities", "parallel play (alongside children)", "plays interactively with other children", 
    	"separates from parent easily", "shows defiant behavior", "shows more and more independence"}|>, 
    <|"TypicalAgeRange" -> Quantity[36, "Months"], "Milestones" -> {"copies adults and friends", 
    	"dresses without supervision", "dresses with supervision", "may get upset with major changes in routine", 
    	"separates easily from mom and dad", "shows affection for friends without prompting", 
    	"shows a wide range of emotions", "shows concern for crying friend", "takes turns in games", 
    	"understands the idea of \"mine\" and \"his\" or \"hers\""}|>, 
    <|"TypicalAgeRange" -> Quantity[48, "Months"], "Milestones" -> {"can say first and last name", 
    	"can distinguish fantasy from reality", "does not object to major change in routine", "interacts with peers", 
    	"knows some basic rules of grammar, such as correctly using gendered pronouns", "plays board and card games", 
    	"sings a song or says a poem from memory", "tells stories"}|>, 
    <|"TypicalAgeRange" -> Quantity[60, "Months"], "Milestones" -> {"able to tell a lie", "dresses without help", 
    	"friendly or hostile/aggressive behavior", "has friends", "is aware of gender", 
    	"is sometimes demanding and sometimes very cooperative", "likes to sing, dance and act", 
    	"more likely to agree with rules", "plays interactive games with peers", "shows more independence", 
    	"wants to be like friends", "wants to please friends"}|>,
    <|"TypicalAgeRange" -> Quantity[Interval[{6, 11}], "Years"], "Milestones" ->  {"appropriate behavior at home", 
    	"appropriate behavior at school", "appropriate behavior playing with friend", "can manage anger", 
    	"communicates full range of emotions", "communicates with teacher", "completes school work", 
    	"displays preadolescent behavior", "does chores at home when asked", "eats healthy food and snacks", 
    	"engages in after-school activities", "gets along well with family and friends", "has a positive self-image", 
    	"increasingly independent", "increasingly responsible", "pride in achievements", "shows interest in school", 
    	"talks about what goes on in school"}|>, 
   	<|"TypicalAgeRange" -> Quantity[Interval[{12, 14}], "Years"], "Milestones" -> {"takes appropriate risks"}|>, 
   	<|"TypicalAgeRange" ->Quantity[Interval[{18, 21}], "Years"], "Milestones" -> {"has close relationships"}|>});

gendereventData={
	All->({generateMilestoneRanges[#]->#}&/@{
		<|"TypicalAgeRange" -> Quantity[Interval[{10, 11}], "Years"], "Milestones" -> {"puberty begins in females"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{15, 17}], "Years"], "Milestones" -> {"puberty ends in females"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{11, 12}], "Years"], "Milestones" -> {"puberty begins in males"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{16, 17}], "Years"], "Milestones" -> {"puberty ends in males"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{12, 13}], "Years"], "Milestones" -> {"onset of menstruation"}|>, 
		<|"TypicalAgeRange" -> Quantity[13, "Years"], "Milestones" -> {"first ejaculation"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{49, 52}], "Years"], "Milestones" -> {"menopause occurs"}|>}),
	"Male"->({generateMilestoneRanges[#]->#}&/@{
		<|"TypicalAgeRange" -> Quantity[Interval[{11, 12}], "Years"], "Milestones" -> {"puberty begins in males"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{16, 17}], "Years"], "Milestones" -> {"puberty ends in males"}|>, 
		<|"TypicalAgeRange" -> Quantity[13, "Years"], "Milestones" -> {"first ejaculation"}|>}),
	"Female"->({generateMilestoneRanges[#]->#}&/@{
		<|"TypicalAgeRange" -> Quantity[Interval[{10, 11}], "Years"], "Milestones" -> {"puberty begins in females"}|>, 
		<|"TypicalAgeRange" -> Quantity[Interval[{15, 17}], "Years"], "Milestones" -> {"puberty ends in females"}|>,
		<|"TypicalAgeRange" -> Quantity[Interval[{12, 13}], "Years"], "Milestones" -> {"onset of menstruation"}|>,
		<|"TypicalAgeRange" -> Quantity[Interval[{49, 52}], "Years"], "Milestones" -> {"menopause occurs"}|>})
};

masterindex=Module[{initial=Flatten[{Join[{MilestoneData["Physical"],MilestoneData["Teeth"]}],MilestoneData["Cognitive"],
		MilestoneData["Language"],MilestoneData["Social"]}],gendered=(gendereventData/.Alternatives -> myFirst)},
	initial=(initial/.Alternatives -> myFirst)[[All,1]];
	{All->Union[Cases[{initial,Flatten[(All/.gendered)][[All,1]]},_Integer, Infinity]],
		"Male"->Union[Cases[{initial,Flatten[(All/.gendered)][[All,1]]},_Integer, Infinity]],
		"Female"->Union[Cases[{initial,Flatten[(All/.gendered)][[All,1]]},_Integer, Infinity]]
	}
]
physicalindex=Module[{initial=Flatten[Join[{MilestoneData["Physical"],MilestoneData["Teeth"]}]],
	gendered=(gendereventData/.Alternatives -> myFirst)},
	initial=(initial/.Alternatives -> myFirst)[[All,1]];
	{All->Union[Cases[{initial,Flatten[(All/.gendered)][[All,1]]},_Integer, Infinity]],
		"Male"->Union[Cases[{initial,Flatten[(All/.gendered)][[All,1]]},_Integer, Infinity]],
		"Female"->Union[Cases[{initial,Flatten[(All/.gendered)][[All,1]]},_Integer, Infinity]]
	}
]
cognitiveindex=Module[{initial=Flatten[MilestoneData["Cognitive"]]},
	initial=(initial/.Alternatives -> myFirst)[[All,1]];
	Union[Cases[initial,_Integer, Infinity]]
]
languageindex=Module[{initial=Flatten[MilestoneData["Language"]]},
	initial=(initial/.Alternatives -> myFirst)[[All,1]];
	Union[Cases[initial,_Integer, Infinity]]
]
socialindex=Module[{initial=Flatten[MilestoneData["Social"]]},
	initial=(initial/.Alternatives -> myFirst)[[All,1]];
	Union[Cases[initial,_Integer, Infinity]]
]

With[{s=$ProtectedSymbols},SetAttributes[s,{ReadProtected}]];
Protect@@$ProtectedSymbols;

getResourceFile["FetalGrowthData.m"];

End[]; (* End Private Context *)