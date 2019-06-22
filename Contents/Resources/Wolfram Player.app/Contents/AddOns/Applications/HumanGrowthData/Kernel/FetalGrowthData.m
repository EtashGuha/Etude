Begin["DataPaclets`FetalGrowthDataDump`"];

$ProtectedSymbols = {
	System`FetalGrowthData
};

Unprotect@@$ProtectedSymbols;

$propertyList = {"AbdominalCircumference","AmnioticFluidIndex","BiparietalDiameter","CrownRumpLength","FemurLength","HeartRate",
	"Weight","HeadCircumference","MeanAbdominalDiameter","Milestones","MilestonesNextWeek","MilestonesPreviousWeek",
	"OccipitalFrontalDiameter","ProjectedWeightBirth","TotalHeartbeats"};
$zscoreprops = {"AbdominalCircumference","BiparietalDiameter","CrownRumpLength","FemurLength","Weight","HeadCircumference",
	"MeanAbdominalDiameter","OccipitalFrontalDiameter","ProjectedWeightBirth",All};

(* INPUT TESTING *)
(*argument checking code *)
pregnancyageQ[x_Quantity]:=Module[{age=x/.HoldPattern[ThreadDepth -> _]:>ThreadDepth->Infinity},
	If[ListQ[age],
		pregnancyageQ[age],
		CompatibleUnitQ[x,Quantity["Month"]]
		(*If[CompatibleUnitQ[x,Quantity["Month"]],
			Quantity[-20, "Days"]<=x<=Quantity[330, "Days"],
			False
		]*)
	]
]
pregnancyageQ[x_DateObject]:=(Quantity[-20, "Days"]<=(Quantity[40, "Weeks"] - (x - Now))<=Quantity[330, "Days"])
pregnancyageQ[x_?(VectorQ[#,NumericQ]&)]:=Module[{age=If[Length[x]<=6,DateObject[x],DateObject[{3000}]]},
	pregnancyageQ[age]
]
pregnancyageQ[{}]:=False
pregnancyageQ[x_List] := If[pregnancyageQ[First[x]],And @@ (pregnancyageQ /@ x),False]
pregnancyageQ[___]:=False

fgentityformat[e_Association]:=If[KeyExistsQ[e,"Age"],e["Age"],e]
fgentityformat[e_]:=e
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
 
PercentileToZ[perc_] := Sqrt[2.]*InverseErf[0.02*perc - 1.]

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

Options[System`FetalGrowthData]={UnitSystem:>$UnitSystem};
System`FetalGrowthData["Properties",opt:OptionsPattern[]] :=Sort[$propertyList]
System`FetalGrowthData[e_,opt:OptionsPattern[]] :=With[{res = oFetalGrowthData[e,All,
		Sequence @@ FilterRules[{opt}, Options[System`FetalGrowthData]]]},
    res /; res =!= $Failed
];
System`FetalGrowthData[e_,p_,opt:OptionsPattern[]] :=With[
	{res = If[StringQ[p]||p===All,oFetalGrowthData[e,p,
			Sequence @@ FilterRules[{opt}, Options[System`FetalGrowthData]]],
		If[MatchQ[p,_Rule],$Failed,oFetalGrowthData[e,All,p,
				Sequence @@ FilterRules[{opt}, Options[System`FetalGrowthData]]]]]},
    res /; res =!= $Failed
];
System`FetalGrowthData[e_,p_,z_,opt:OptionsPattern[]] :=With[{res = oFetalGrowthData[e,p,z,
		Sequence @@ FilterRules[{opt}, Options[System`FetalGrowthData]]]},
    res /; res =!= $Failed
];
System`FetalGrowthData[args___,opt:OptionsPattern[]]:=With[{res = If[1<=Length[{args}]<=3,$Failed,
	(Message[General::argb, System`FetalGrowthData, Length[{args}], 1, 3]; $Failed),$Failed]},
    res /; res =!= $Failed
];

Options[oFetalGrowthData]={UnitSystem->$UnitSystem};
oFetalGrowthData[entity_,opt:OptionsPattern[]]:=Module[
	{e=fgentityformat[entity]},
	If[pregnancyageQ[e],
		entity,
		Message[System`FetalGrowthData::notage,e];Return[$Failed]
	]
]
oFetalGrowthData[entity_,property_,opt:OptionsPattern[]]:=Module[{
	age=If[entity==="Properties",Message[System`FetalGrowthData::notage,entity];Return[$Failed],
		oFetalGrowthData[entity]/.{(ThreadDepth->_)->(ThreadDepth->Infinity),e_Association:>fgentityformat[e]}],
	p=property/.{EntityProperty["Pregnancy", x_]->x},res,
	units=OptionValue[UnitSystem]},
	If[age===$Failed,Return[$Failed],age=formatAge[age]];
	p=If[MemberQ[$propertyList,p]||p===All,
		p,
		Message[System`FetalGrowthData::notprop, property, System`FetalGrowthData];Return[$Failed]
	];
	If[MemberQ[{age,p},$Failed],Return[$Failed]];
	If[Not[MemberQ[{"Imperial","Metric"},units]],Message[System`FetalGrowthData::unit, units];Return[$Failed]];
	If[p===All,
		Association[(#->oFetalGrowthProperties[age,#,units])&/@$propertyList],
		res=oFetalGrowthProperties[age,p,units];
		If[Not[FreeQ[res,_Missing]||ListQ[age]],
			ageInRangeQ[p,age,True]];
		res
	]
]
oFetalGrowthData[entity_,property_,z_,opt:OptionsPattern[]]:=Module[{
	age=If[entity==="Properties",Message[System`FetalGrowthData::notage,entity];Return[$Failed],
		oFetalGrowthData[entity]/.{(ThreadDepth->_)->(ThreadDepth->Infinity),e_Association:>fgentityformat[e]}],
	p=property/.{EntityProperty["Pregnancy", x_]->x},
	units=OptionValue[UnitSystem],result,zScore},
	zScore=indexformat[z,All,System`FetalGrowthData];
	If[age===$Failed,Return[$Failed],age=formatAge[age]];
	p=If[MemberQ[$propertyList,p]||p===All,
		p,
		Message[System`FetalGrowthData::notprop, property, System`FetalGrowthData];Return[$Failed]
	];
	If[Not[MemberQ[$zscoreprops,p]],Message[System`FetalGrowthData::nindex,p]];
	If[MemberQ[{age,p},$Failed],Return[$Failed]];
	If[zScore===$Failed,Message[System`FetalGrowthData::invindex,z];Return[$Failed]];
	If[zScore===False,Return[Missing["NotAvailable"]]];
	If[Not[MemberQ[{"Imperial","Metric"},units]],Message[System`FetalGrowthData::unit, units];Return[$Failed]];
	If[p===All,
		Association[(#->oFetalGrowthProperties[age,#,units,zScore])&/@$propertyList],
		result=oFetalGrowthProperties[age,p,units,zScore];
		If[Not[FreeQ[result,_Missing]||ListQ[age]],
			ageInRangeQ[p,age,True]];
		result
	]
]
oFetalGrowthData[args___,opt:OptionsPattern[]]:=(System`Private`Arguments[System`FetalGrowthData[args], {1,3}];$Failed)


oFetalGrowthProperties[age_,p_,units_]:=Module[{result},
	If[ListQ[age],
		result=If[ageInRangeQ[p,#,False],
			If[MemberQ[$zscoreprops,p],Interval[{propertyValue[p, #, -1],propertyValue[p, #, 1]}],propertyValue[p, #, 0]],Missing["NotAvailable"]]&/@age;
		result=If[FreeQ[#,_Missing],#,Missing["NotAvailable"]]&/@result,
		result=If[ageInRangeQ[p,age,False],
			If[MemberQ[$zscoreprops,p],Interval[{propertyValue[p, age, -1],propertyValue[p, age, 1]}],propertyValue[p, age, 0]],Missing["NotAvailable"]];
		If[Not[FreeQ[result,_Missing]],result=Missing["NotAvailable"]]
	];
	If[Length[result]<2,
		propertyQuantity[If[ListQ[result]&&Not[MemberQ[{"Milestones","MilestonesPreviousWeek","MilestonesNextWeek"},p]],
			First[result],result],p,units],
		propertyQuantity[#,p,units]&/@result]
]
oFetalGrowthProperties[age_,p_,units_,zScore_]:=Module[{result},
	If[ListQ[age],
		result=If[ageInRangeQ[p,#,False],
			propertyValue[p, #, zScore],Missing["NotAvailable"]]&/@age;
		result=If[FreeQ[#,_Missing],#,Missing["NotAvailable"]]&/@result,
		result=If[ageInRangeQ[p,age,False],
			propertyValue[p, age, zScore],Missing["NotAvailable"]];
		If[Not[FreeQ[result,_Missing]],result=Missing["NotAvailable"]]
	];
	If[Length[result]<2,propertyQuantity[If[ListQ[result]&&Not[MemberQ[{"Milestones","MilestonesPreviousWeek","MilestonesNextWeek"},p]],
			First[result],result],p,units],propertyQuantity[#,p,units]&/@result]
]

(* format age *)
formatAge[age_Quantity]:=If[QuantityMagnitude[age,"Days"]<0,UnitConvert[Quantity[40, "Weeks"]-age,"Weeks"],UnitConvert[age,"Weeks"]]
formatAge[age_List]:=formatAge/@age
formatAge[duedate_DateObject]:=UnitConvert[Quantity[40, "Weeks"]-(duedate-Now),"Weeks"]

(*checks for in range *)
pregnancyDataAgeBound["CrownRumpLength", "Min"] = Quantity[40/7,"Weeks"]; (* true value is 40/7 *)
pregnancyDataAgeBound["CrownRumpLength", "Max"] = Quantity[14,"Weeks"];
pregnancyDataAgeBound["AmnioticFluidIndex", "Min"] = Quantity[16,"Weeks"];
pregnancyDataAgeBound["AmnioticFluidIndex", "Max"] = Quantity[42,"Weeks"];(*
pregnancyDataAgeBound["WeightGain", "Min"] = Quantity[8,"Weeks"]; (* combined ranges *)
pregnancyDataAgeBound["WeightGain", "Max"] = Quantity[40,"Weeks"];*)
pregnancyDataAgeBound["Milestones", "Min"] = Quantity[0,"Weeks"]; 
pregnancyDataAgeBound["Milestones", "Max"] = Quantity[44,"Weeks"];
pregnancyDataAgeBound["MilestonesPreviousWeek", "Min"] = Quantity[1,"Weeks"]; 
pregnancyDataAgeBound["MilestonesPreviousWeek", "Max"] = Quantity[45,"Weeks"];
pregnancyDataAgeBound["MilestonesNextWeek", "Min"] = Quantity[0,"Weeks"];
pregnancyDataAgeBound["MilestonesNextWeek", "Max"] = Quantity[45,"Weeks"];
pregnancyDataAgeBound["AbdominalCircumference"|"BiparietalDiameter"|"FemurLength"|"HeartRate"|"Weight"|
	"HeadCircumference"|"MeanAbdominalDiameter"|"OccipitalFrontalDiameter"|"TotalHeartbeats"|"ProjectedWeightBirth", "Min"] = Quantity[16,"Weeks"];
pregnancyDataAgeBound["AbdominalCircumference"|"BiparietalDiameter"|"FemurLength"|"HeartRate"|"Weight"|
	"HeadCircumference"|"MeanAbdominalDiameter"|"OccipitalFrontalDiameter"|"TotalHeartbeats"|"ProjectedWeightBirth", "Max"] = Quantity[39,"Weeks"];
ageInRangeQ[property_String, age_Quantity,m_] := Module[{min=pregnancyDataAgeBound[property, "Min"],max=pregnancyDataAgeBound[property, "Max"]},
	If[LessEqual[min, age, max],
		True,
		If[m,Message[System`FetalGrowthData::notterm,min,max]];False
	]
]

(*
FetalGrowthData["ZScore", "NormalRange"] = Quantity[Interval[{-5, 5}], Unit[PureUnities]];
FetalGrowthData["Age", "NormalRange"] = Quantity[Interval[{-20, 330}], Unit[Days]]; (* recall gestational age extends negatively and may as well allow overdue babies *)
*)




(*TODO: convert over, include imperial/metric*)
propertyQuantity[x_Missing,_,_]:=x
propertyQuantity[x_,"AbdominalCircumference"|"BiparietalDiameter"|"FemurLength"|"HeadCircumference"|"MeanAbdominalDiameter"|
	"OccipitalFrontalDiameter"|"CrownRumpLength","Metric"]:=Quantity[x,"Millimeters"]
propertyQuantity[x_,"AbdominalCircumference"|"BiparietalDiameter"|"FemurLength"|"HeadCircumference"|"MeanAbdominalDiameter"|
	"OccipitalFrontalDiameter"|"CrownRumpLength","Imperial"]:=UnitConvert[Quantity[x,"Millimeters"],"Inches"]
propertyQuantity[x_,"TotalHeartbeats",_]:=Quantity[x,IndependentUnit["Beats"]]
propertyQuantity[x_,"HeartRate",_]:=Quantity[x,IndependentUnit["Beats"]/"Minutes"]
propertyQuantity[x_,"Weight"|"ProjectedWeightBirth","Imperial"]:=UnitConvert[Quantity[N@x,"Grams"],"Pounds"]
propertyQuantity[x_,"Weight"|"ProjectedWeightBirth","Metric"]:=Quantity[x,"Grams"](*
propertyQuantity[x_,"WeightGain","Imperial"]:=Quantity[x,"Pounds"]
propertyQuantity[x_,"WeightGain","Metric"]:=UnitConvert[Quantity[N[x],"Pounds"],"Grams"]*)
propertyQuantity[x_,"AmnioticFluidIndex","Imperial"]:=UnitConvert[Quantity[x,"Millimeters"],"Inches"]
propertyQuantity[x_,"AmnioticFluidIndex","Metric"]:=Quantity[x/10.,"Centimeters"]
propertyQuantity[x_,_,_]:=x

propertyValue[_,Except[_Quantity],_]:=Missing["NotAvailable"]

(*** CrownRumpLength ***) (* Millimeters *)
(* The standard deviation term for this property was computed by subtracting the 5th percentile from th 95th of the dataset and pretending the 
   distribution is symmetric, even though it is not...  *)
propertyValue["CrownRumpLength", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(-4.672431393785568+0.1639522044390383*age-0.0005652279959168598*age^2) + (-9.08918-0.259622*age+0.0124029*age^2);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
]; 

(*** HeadCircumference ***) (* Millimeters *)
propertyValue["HeadCircumference", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(4.857+.03213*age) + (-106+2.174*age-.000007626*age^3);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
]; 

(*** FemurLength ***) (* Millimeters *)
propertyValue["FemurLength", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(3.822-175.6/age) + (-38.77+.6042*age-.0007116*age^2);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
];

(*** AbdominalCircumference ***) (* Millimeters *)
propertyValue["AbdominalCircumference", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(1.179+.0679*age) + (-89.39+1.719*age-.000002516*age^3);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
];

(*** MeanAbdominalDiameter ***) (* Millimeters *)
propertyValue["MeanAbdominalDiameter", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(.4414+.0213*age) + (-27.46+.5473*age-.000000801*age^3);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
];

(*** BiparietalDiameter ***) (* Millimeters *)
propertyValue["BiparietalDiameter", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(1.648+.00933*age) + (-28.04+.597*age-.00000187*age^3);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
];

(*** OccipitalFrontalDiameter ***) (* Millimeters *)
propertyValue["OccipitalFrontalDiameter", a_Quantity, zScore_]:=Module[{age = QuantityMagnitude[a, "Days"],res},
    res=zScore*(1.266+.01738*age) + (-39.08+.7791*age-.000002926*age^3);
    If[NumericQ[res],
    	res,
    	Missing["NotAvailable"]
    ]
];

(*** FetalWeight && ProjectedWeightBirth ***) (* returns grams *)
propertyValue["Weight",a_Quantity, zScore_] := Module[{ac, fl},
    ac = 0.1 propertyValue["AbdominalCircumference", a,zScore]; (* should return centimeters *)
    fl = 0.1 propertyValue["FemurLength", a,zScore];
    If[NumericQ[ac]&&NumericQ[fl],
    	fetalWeight[ac, fl],
    	Missing["NotAvailable"]
    ]
];
propertyValue["ProjectedWeightBirth",_Quantity, zScore_]:=propertyValue["Weight",Quantity[40,"Weeks"], zScore]

fetalWeight[abdominalCircumference_, femurLength_, opts:OptionsPattern[]] := With[
    {
        AC = abdominalCircumference,
        FL = femurLength
    },
    10^(1.3598 + 0.051*AC + 0.1844*FL - 0.0037*AC*FL)
];

(*** FetalHeartRate ***)
(* heart rate in beat per minute, kind of lame *)
propertyValue["HeartRate",age_Quantity,_] := Module[{weeks = QuantityMagnitude[age, "Weeks"],res},
	res=Piecewise[
    	{
    	    {37 + 19 weeks, 7 >= weeks >= 4},
    	    {5480/31 - (30 weeks)/31, 39 >= weeks > 7}
    	},
    	0
	];
	If[res===0,
        Missing["NotAvailable"],
        N@res
    ]
];

(*** TotalHeartbeats ***)
propertyValue["TotalHeartbeats",age_Quantity,_]:= Module[{weeks = QuantityMagnitude[age, "Weeks"],res},
	If[weeks<4,Return[Missing["NotAvailable"]]];
	res=Integrate[10080*propertyValue["HeartRate",Quantity[a,"Weeks"],0], {a, 4, weeks}];
	If[NumericQ[res],
		res,
        Missing["NotAvailable"]
    ]
]

(*** AmnioticFluidIndex function ***)
propertyValue["AmnioticFluidIndex", age_Quantity,_] := Module[
    {weeks = QuantityMagnitude[age, "Weeks"],res},
    res=aFIInterpolation[weeks][[{1,3}]];
    If[MatchQ[res,{_?NumericQ,_?NumericQ}],
    	Interval[N@res],
        Missing["NotAvailable"]
    ]
];

aFIInterpolation = Interpolation[{{16, {79, 121, 185}}, {17, {83, 127, 194}}, {18, {87, 133, 
   202}}, {19, {90, 137, 207}}, {20, {93, 141, 212}}, {21, {95, 143, 
   214}}, {22, {97, 145, 216}}, {23, {98, 146, 218}}, {24, {98, 147, 
   219}}, {25, {97, 147, 221}}, {26, {97, 147, 223}}, {27, {95, 146, 
   226}}, {28, {94, 146, 228}}, {29, {92, 145, 231}}, {30, {90, 145, 
   234}}, {31, {88, 144, 238}}, {32, {86, 144, 242}}, {33, {83, 143, 
   245}}, {34, {81, 142, 248}}, {35, {79, 140, 249}}, {36, {77, 138, 
   249}}, {37, {75, 135, 244}}, {38, {73, 132, 239}}, {39, {72, 127, 
   226}}, {40, {71, 123, 214}}, {41, {70, 116, 194}}, {42, {69, 110, 
   175}}}];

(*** WeightGain ***)(* pounds *)
(*propertyValue["WeightGain",age_Quantity,_]:=Module[
    {weeks=QuantityMagnitude[age,"Weeks"],min,max},
    min = weightGainLowerBound[weeks];
    max = weightGainUpperBound[weeks];
    Which[min===0,
    	0,
    	FreeQ[{min,max},$Failed],
    	Interval[{min,max}],
    	True,
        Missing["NotAvailable"]
    ]
]

weightGainLowerBound[weeks_] := Piecewise[{
	{0,weeks<8},
    {-0.002604*weeks^3 + 0.1517857*weeks^2 - 1.81994*weeks + 6.4178571, 8 <= weeks < 16},
    {0.000342*weeks^3 - 0.041126*weeks^2 + 2.3628096*weeks - 23.16739, 16 <= weeks < 32},
    {0.0052083*weeks^3 - 0.616071*weeks^2 + 24.33631*weeks - 297.05, 32 <= weeks <= 40},
    {$Failed,40<weeks}
}];*)
(*
weightGainUpperBound[weeks_] := Piecewise[{
	{0,weeks<8},
    {0.0130208*weeks^3 - 0.379464*weeks^2 + 4.0550595*weeks - 12.58214, 8 <= weeks < 16},
    {-0.000789*weeks^3 + 0.0324675 * weeks^2 + 1.1341089*weeks - 14.77814, 16 <= weeks < 32},
    {-4.9*10^(-16)*weeks^3 - 0.040179*weeks^2 + 3.3303571*weeks - 36.45, 32 <= weeks <= 40},
    {$Failed,40<weeks}
}];*)

(*** Milestones ***)
propertyValue["Milestones",age_Quantity,_] := Module[
    {day=Floor[QuantityMagnitude[age,"Days"]],events},
    events = day/.eventData;
    If[NumericQ[events],
        Missing["NotAvailable"],
        Flatten[{events}]
    ]
];
propertyValue["MilestonesNextWeek",age_Quantity,z_]:=propertyValue["Milestones",age+Quantity["Weeks"],z]
propertyValue["MilestonesPreviousWeek",age_Quantity,z_]:=propertyValue["Milestones",age-Quantity["Weeks"],z]

(*Milestone Events*)
(*Indexed by Day*)
eventData = {0->{"cervical flexure formation begins","embryonic period begins"},
     1->{"zygote formation begins"},
     2->{"genome activation occurs","blastomeres begin rapidly dividing","early pregnancy factor is detectable within the blood","zygote divides into two blastomeres"},
     3->{"compaction occurs","embryo enters the uterus"},
     4->{"embryonic disc formation occurs","embryo emerges from the zona pellucida","blastocyte is now free floating"},
     5->{"implantation begins"},
     7->{"amniotic cavity formation begins","chorion forms","extraembryonic mesoderm formation begins","inner cell mass divides into the hypoblast and epiblast",
     		"placenta begins to form","syncytiotrophoblast and cytotrophoblast form"},
     8->{"human chorionic gonadotropin detectable in mother's blood and urine"},
     9->{"cells in womb become engorged with nutrients","exocoelomic membrane forms","trophoblastic lacunae forms","primary yolk sac forms"},
     11->{"lacunar vascular circle forms"},
     12->{"implantation completes","primary umbilical vesicle forms from extraembryonic endoderm","primary yolk sac turns into the secondary yolk sac"},
     13->{"blood vessels form in the villi","primordial blood vessels form","chorionic villi form"},
     14->{"embryonic epiblast gives rise to primitive streak and embryonic ectoderm","primitive node forms"},
     15->{"stems cell divide into the ectoderm, endoderm, and mesoderm","prechordal plate forms","primitive groove appears"},
     16->{"secondary villi forms"},
     17->{"notochordal process forms and fuses with the endoderm"},
     18->{"brain begins to appear","neural ectoderm forms","neural groove and neural folds form","neural plate forms","notochordal and neurenteric canals form",
     	"notochordal plate forms","connecting stalk forms","notochordal pit forms"},
     20->{"otic disc forms","forebrain, midbrain, and hindbrain become more prominent","neural groove deepens substantially","primary neuromeres appear",
     	"cephalic and caudal folds appear","primitive streak reaches neurenteric canal","somite segments 1-3 form"},
     22->{"primary head vein forms","optic sulcus appears","otic pits form","otic plates form","chiasmatic plate appears","mesencephalic flexure appears",
     	"neural tube begins to appear","foregut and hindgut form","telencephalon appears","hyoid arch forms","mandibular arch and maxillary process form",
     	"somites pairs 4-12 form"},
     24->{"optic vesicles form","lamina terminalis forms","neural crest production and migration continues","neurohypophysial primordia appears",
     	"neuropore begin to close off","somite pairs 13-20 form"},
     26->{"internal carotid arteries form","rostral and caudal cardinal veins along brain and spinal cord begin to feed the common cardinal veins",
     	"ectodermal ring formation completes","hypoglossal nucleus appears","lower spinal cord formation begins","mamillary recess forms",
     	"marginal layer in rhombencephalon forms","mesencephalic flexure orients at 90 degrees","neural cord is present within the caudal eminence","neural tube closes",
     	"neurofibrils form in rhombencephalon","primary neurulation completes","sulcus limitans forms in the midbrain","somite pairs 21-29 form",
     	"embryo becomes progressively c-shaped "},
     28->{"anterior, middle, and posterior cerebral plexuses form","primary head veins drain cerebral plexuses and feed the precardinal veins","vertebral arteries form",
     	"otic invagination appears","retinal and lens discs form","adenohypophysial pouch appears","cerebellum forms","common afferent tract forms","fourth ventricle forms",
     	"isthmus rhombencephali appears","oculomotor nucleus forms in mesencephalon","terminal-vomeronasal neural crest appears",
     	"amnion surrounds the connecting stalk and vitelline stalk","amnion surrounds the embryo","trochlear nucleus forms in the isthmus","umbilical cord emerges",
     	"upper and lower limb buds appear"},
     31->{"eyes begin to appear"},
     32->{"lens and retina invaginate to form the optic cup","primordium of cochlear duct forms","adenohypophysis forms","all 16 secondary neuromeres are present",
     	"cerebral hemispheres appear and begin rapid growth","lateral ventricles begin to form in the brain","primordium appears in alar plate of rhombomere 1",
     	"oculomotor nerve forms","di-telencephalic sulcus forms","dorsal and ventral thalami form","dorsal funiculus forms","mamillary region appears",
     	"medial and lateral longitudinal fasciculi appear","pontine flexure forms","preoptic sulcus extends between optic evaginations","rhombic lip appears",
     	"synencephalon forms","tegmentum forms","medial portion of the tentorium cerebelli appear","terminal-vomeronasal crest makes contact with the brain",
     	"torus hemisphericus appears","velum transversum appears","ventral longitudinal fasciculus appears","ventral segment of hyoid arch subdivides"},
     33->{"right and left neural processes begin to form","vertebrae well appears","vertebral centra appears","blood vessels penetrate the diencephalon",
     	"capsule appears around the lens","endolymphatic duct of the ear forms","lens body appears","optic chiasm forms",
     	"primordium of antitragus emerges from ventral subsegment of the hyoid arch","adult lamina terminalis forms","amygdaloid area appears",
     	"five main sections of brain become prominent","cerebellar plate forms","fibers of dorsal funiculus reach level C1",
     	"first axodendritic synapses form in the cervical spinal cord","first nerve fibers appear","habenular nucleus forms","habenulo-interpeduncular tract appears",
     	"hypothalamic sulcus appears","hypothalamus forms","lateral striatal ridge appears","lateral ventricular eminence appears","locus caeruleus forms",
     	"median striatal ridge appears","olfactory fibers reach the brain","optic groove appears","postoptic recess appears","primary meninx surrounds most of the brain",
     	"primordium of epiphysis forms","superior colliculi and its commissure form","superior medullary velum appears","supramamillary commissure appears",
     	"synapses form among motor neurons in the spinal cord","tectobulbar tract appears","tentorium forms","third ventricle forms","trigemino-cerebellar tract forms",
     	"trochlear nerve root forms","hand plate emerges from distal upper limb bud"},
     35->{"subtle movement occurs"},
     37->{"spinal accessory nerve forms","anterior, middle, and posterior cerebral arteries form","formation of the circle of Willis progresses","mesencephalic artery forms",
     	"myelencephalic artery forms","nasal pits appear ventral and separated","olfactory fibers connect nasal pits with the brain","olfactory tubercle appears",
     	"pigment in retina is visible","primordium of cochlear pouch forms","area epithelialis appears","primordial plexiform layer forms in the future temporal lobe",
     	"cajal-retzius cells appear","commissure of the trochlear nerve forms","dorsal funiculus fibers reach the medulla oblongata","epiphysis cerebri appears",
     	"glial cells identifiable adjacent to neurons","greater petrosal nerve forms","infundibular recess and infundibulum appear","large interventricular foramen appears",
     	"median forebrain bundle forms","olfactory tubercle forms","pontine flexure deepens","posterior commissure forms","recurrent laryngeal nerve forms",
     	"reticular formation become more defined","retinal fissure closes","superior laryngeal nerve appears","frontonasal prominence forms",
     	"second pharyngeal arch becomes more prominent","third pharyngeal arch recedes"},
     40->{"nerve cells begin differentiating"},
     41->{"anterior choroid artery appears","auditory ossicles identifiable in mesenchyme","cell islands in olfactory tubercle appear","crescentic lens cavity forms",
     	"capillaries between adenohypophysis and hypothalamus form","commissure of the oculomotor nerves forms","cortical nucleus in amygdaloid body forms",
     	"dentate and isthmic nuclei form in the cerebellum","dura begins forming in the basal area","septal nucleus begins to form",
     	"frontal and temporal poles of cerebral hemispheres become prominent","gustatory fibers separate from the common afferent tract","hemispheric stalk appears",
     	"interventricular foramen appears","somite pairs 38 and 39 appear","spinal cord reaches the caudal tip of the body","subarachnoid space forms",
     	"synapses form in the spinal cord between interneurons and primary afferent neurons"},
     42->{"face withdraws from touch around the mouth","cerebral cortex becomes prominent","typical time of 1st ultrasound",
     	"typical time of the 1st round of prenatal blood tests","typical time of 1st prenatal pap smear","typical time of rubella immunity testing",
     	"typical time of sexually transmitted disease testing"},
     44->{"mesenchyme thickens","nerve fibers are present in the retina","one to three semicircular ducts form","optic fibers form",
     	"the retina's outer lamina becomes heavily pigmented","vomeronasal nerve and ganglion form","archistriatum forms",
     	"dentate nucleus forms in the internal cerebellar swellings","pineal recess emerges","brainwave activity begins","cerebrospinal fluid production begins",
     	"choroid plexuses present in fourth and lateral ventricles","corpus striatum extends to preoptic sulcus","four amygdaloid nuclei form",
     	"choroid folds appear in the fourth ventricle","hippocampus reaches olfactory region","neurohypophysis walls appear folded","nucleus ambiguus of the vagus forms",
     	"prosencephalic septum appears","red nucleus appears","substantia nigra forms","supraoptic commissure forms"},
     47->{"mandibular nerve innervates the premuscle mass of the mastication muscles","circle of Willis formation completes","common carotid arteries form",
     	"primitive cavernous sinus drains primitive maxillary and supraorbital veins","cochlear duct tip grows upward","nerve fibers begin extending from the retina",
     	"optic fibers enter the chiasmatic plate","lateral lobes of pars tuberalis form","internal capsule formation begins","dorsal and ventral cochlear nuclei appear",
     	"lateral recesses form in the fourth ventricle","ganglion of nervus terminalis forms","habenular commissure forms","gyrus dentatus of the hippocampus appears",
     	"interpeduncular fossa appears","lemniscal decussation forms","lower limb nerves are prominent","medial accessory olivary nucleus forms",
     	"neurohypophyseal bud appears","nuclei of the forebrain septum appears","nucleus accumbens appears","occipital pole of cerebral hemispheres becomes prominent",
     	"subcommissural organ forms"},
     49->{"head begins to rotate","legs begin to move"},
     50->{"optic commissure forms","optic fibers extend to optic chiasm","inferior colliculus forms","cerebral hemispheres expand beyond lamina terminalis",
     	"cerebral peduncles form","cuneate and gracile decussating fibers appear","interpeduncular groove appears","medial septal nucleus appears",
     	"nigrostriatal fibers appears","nucleus of diagonal band forms","sacrocaudal spinal cord formation completes","cuneate and gracile decussating fibers appear",
     	"septum verum appears","spinothalamic tract forms"},52->{"anterior and posterior choroid arteries form","fibers of the optic nerve reach the brain",
     	"anterior and inferior horns of lateral ventricle become prominent","lateral ventricle appear C-shaped","glial cells are present within the cranial nerves",
     	"global pallidus internus forms","insula forms","lateral olfactory tract forms","pyramidal cells in hippocampus appear"},
     54->{"intradural veins begin to form","claustrum appears","putamen appears","commissural plate thickens","cortical plate expands rapidly",
     	"folds appear in the roof of the third ventricle","spinothalamic tract appears","sulcus dorsalis appears","superior colliculus appears",
     	"thalamocortical fibers appear"},
     56->{"neurons begin to synapse in the cerebral cortex","complex response to touch emerges","hand-to-face contact begins","mouth opens and closes","squinting begins",
     	"embryo floats and rolls over in the womb","nerves reach the intestinal loop","peristalsis begins in the large intestine","breathing motions begin",
     	"blood supply patterns to the brain begin to normalize","superior sagittal sinus appears","cranial nerve distribution pattern normalizes",
     	"optic tract reaches ventral portion of the lateral geniculate body","caudate nucleus and putamen appear within the corpus striatum","cerebellar commissures appear",
     	"cerebral hemispheres cover lateral portion of the diencephalon","choroid plexus becomes lobular","dura lines the vertebral canal","greater palatine nerve forms",
     	"grey and white matter appear more prominently","hippocampus reaches the temporal pole","inferior and superior cerebellar peduncles form",
     	"principal nucleus of inferior olivary nuclei forms","pyramidal decussations appear","handedness emerges","subarachnoid space appears","suprapineal recess appears",
     	"suprascapular nerve forms","vermis of cerebellum appears","embryo reaches approximately 1 billion cells","embryonic period ends"},
     57->{"fetal period begins"},
     63->{"hip and knee bend if sole of the foot is touched","drinking becomes routine","thumb sucking occurs","sighing occurs","stretching occurs","head movement occurs",
     	"mouth opening occurs","tongue movement occurs","face, hands, and feet sense touch","rapid weight gain begins"},
     67->{"yawning occurs"},70->{"eyes begin to roll downward","commissure of the fornix appears"},
     77->{"facial expressions occur","inner and outer hair cells form"},
     84->{"hand-to-mouth touching occurs frequently","taste buds become prominent in the mouth","myelination begins in the spinal cord"},
     98->{"light touch to mouth evokes a turn toward stimulus","four lobes of cerebral cortex prominent","cerebellum resembles adult structure","fat deposits form in cheeks"},
     105->{"crown-heel length is 19.5 cm"},
     112->{"hormonal stress response occurs in reaction to invasive procedures","typical time of 2nd ultrasound","typical time of alpha-fetoprotein screening",
     	"typical time of hepatitis B screening"},
     119->{"discrete layers of retina are prominent","cerebral cortex produces measurable activity"},
     126->{"speaking motion in larynx occurs","corpus callosum formation completes"},
     133->{"daily cycles occur in biological rhythms","sulci on the surface of the cerebral hemispheres appear"},
     140->{"hearing and response to sound occur"},
     147->{"survival outside the womb is possible"},
     154->{"behavioral states begin to occur"},
     168->{"blink-startle response occurs"},
     175->{"rods and cones form","sense of taste is present","typical time for 2nd round of prenatal blood tests","typical time period of glucose tolerance test"},
     182->{"sense of smell is present"},
     189->{"pupils react to light"},
     196->{"ability to distinguish between sound frequencies is present"},
     210->{"breathing motions occur","typical time of 3rd prenatal ultrasound"},
     224->{"music preferences begin","taste preference begins"},
     245->{"amniotic fluid volume peaks","start of time period for checking cervical dilation"},
     252->{"typical time of 3rd round of prenatal blood tests","typical time of fetal non-stress test","typical time of vaginal discharge culture"},
     259->{"fetal intake of amniotic fluid increases","typical time of 4th prenatal ultrasound","typical time of flowmetry"},
     266->{"spinal cord formation completes","fetus initiates labor","fetus is ready for birth"}
};

With[{s=$ProtectedSymbols},SetAttributes[s,{ReadProtected}]];
Protect@@$ProtectedSymbols;

End[]; (* End Private Context *)