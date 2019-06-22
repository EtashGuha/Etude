(* Mathematica package *)

Begin["Tools`MathematicaFormulas`FormulaData`Private`"];

Clear[System`PlanckRadiationLaw];

$exposedprops=Sort@{"MaxFrequency",(*"MaxFrequencyColor",*)"MaxWavelength",(*formerly "MaxWavelengthColor"*)"Color","MeanFrequency"(*,"MeanFrequencyColor",*)
	"MeanWavelength",(*"MeanWavelengthColor",*)"SpectralPlot"}
$props={"MaxFrequency","MaxFrequencyColor","MaxWavelength","Color","MeanFrequency","MeanFrequencyColor",
	"MeanWavelength","MeanWavelengthColor","SpectralPlot"}

System`PlanckRadiationLaw["Properties"]:=$exposedprops
System`PlanckRadiationLaw[t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),"Properties"]:=$exposedprops
System`PlanckRadiationLaw[t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),prop_?(MemberQ[$props,#]&)]:=With[
	{res=iPlanckRadiationLaw[t,prop]},
		res/;res=!=$Failed]

System`PlanckRadiationLaw[t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),
        w_?(CompatibleUnitQ[#,Quantity[1,"Feet"]]&)]:=With[
	{res=iPlanckRadiationLaw[t,w,"Wavelength"]},
		res/;res=!=$Failed]

System`PlanckRadiationLaw[t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),
        w_?(CompatibleUnitQ[#,Quantity[1,"Hertz"]]&)]:=With[
	{res=iPlanckRadiationLaw[t,w,"Frequency"]},
		res/;res=!=$Failed]

System`PlanckRadiationLaw[t:QuantityVariable[_,"Temperature"],
	w:QuantityVariable[_,type_?(MemberQ[{"Frequency","Wavelength"},#]&)]]:=With[
	{res=iPlanckRadiationLaw[t,w,type]},res/;res=!=$Failed]
System`PlanckRadiationLaw[t:t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),
	w:QuantityVariable[_,type_?(MemberQ[{"Frequency","Wavelength"},#]&)]]:=With[
	{res=iPlanckRadiationLaw[t,w,type]},res/;res=!=$Failed]
System`PlanckRadiationLaw[t:QuantityVariable[_,"Temperature"],w_?(CompatibleUnitQ[#,Quantity[1,"Hertz"]]&)]:=With[
	{res=iPlanckRadiationLaw[t,w,"Frequency"]},res/;res=!=$Failed]
System`PlanckRadiationLaw[t:QuantityVariable[_,"Temperature"],w_?(CompatibleUnitQ[#,Quantity[1,"Feet"]]&)]:=With[
	{res=iPlanckRadiationLaw[t,w,"Wavelength"]},res/;res=!=$Failed]
	
System`PlanckRadiationLaw[t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),
        {w1_?(CompatibleUnitQ[#,Quantity[1,"Feet"]]&),w2_?(CompatibleUnitQ[#,Quantity[1,"Feet"]]&)}]:=With[
	{res=iPlanckRadiationLaw[t,{w1,w2},"Wavelength"]},
		res/;res=!=$Failed]

System`PlanckRadiationLaw[t_?(CompatibleUnitQ[#,Quantity[1,"Kelvins"]]&),
        {w1_?(CompatibleUnitQ[#,Quantity[1,"Hertz"]]&),w2_?(CompatibleUnitQ[#,Quantity[1,"Hertz"]]&)}]:=With[
	{res=iPlanckRadiationLaw[t,{w1,w2},"Frequency"]},
		res/;res=!=$Failed]

System`PlanckRadiationLaw[args___] := (CheckArgsAndIssueMessages[args];Null/;False)

CheckArgsAndIssueMessages[args___]:=Catch[
	If[Not[ArgumentCountQ[PlanckRadiationLaw,Length[{args}],2,2]],Throw[$Failed,$tag]];
	If[Not[CompatibleUnitQ[First[{args}],Quantity[1,"Kelvins"]]],Message[PlanckRadiationLaw::notemp,First[{args}]];Throw[$Failed,$tag]];
	With[{second=Part[{args},2]},
		Which[
			QuantityQ[second],
				If[Not[
					Or[CompatibleUnitQ[second,Quantity[1,"Feet"]],
						CompatibleUnitQ[second,Quantity[1,"Hertz"]]
					]], Message[PlanckRadiationLaw::wlfrq,second];Throw[$Failed,$tag]],
			MatchQ[second,{_?QuantityQ,_?QuantityQ}],
				If[CompatibleUnitQ@@second,
					If[Not[Or[
						CompatibleUnitQ[First[second],Quantity[1,"Feet"]],
						CompatibleUnitQ[First[second],Quantity[1,"Hertz"]]]],
					Message[PlanckRadiationLaw::wlfrq,second];Throw[$Failed,$tag]],
					Message[Quantity::compat,Sequence@@second];Throw[$Failed,$tag]],
			StringQ[second],
				If[UnsameQ[second,"SpectralPlot"],Message[PlanckRadiationLaw::wlfrq,second];Throw[$Failed,$tag]],
			True,
				Message[PlanckRadiationLaw::wlfrq,second];Throw[$Failed,$tag]
		]
	]
	, $tag]
	
iPlanckRadiationLaw[t_,"SpectralPlot"]:=Module[{temperature=UnitConvert[t,"Kelvins"],
	lpeak=0.0028978,factor1,factor2},
   If[Not[NumericQ[QuantityMagnitude[t]]],Return[$Failed]];
   lpeak = lpeak/QuantityMagnitude[temperature];
   factor2=QuantityMagnitude@UnitConvert[Quantity[2., "PlanckConstant"*"SpeedOfLight"^2]/
		(Quantity[1,"Meters"]^5),"Watts"/"Meters"^3];
   factor1=Quantity[1, ("PlanckConstant"*"SpeedOfLight")/"BoltzmannConstant"]/
         (temperature*Quantity[1,"Meters"]);
   Quiet[Plot[factor2/((-1 + E^(factor1/w))*w^5),{w,0,10*lpeak},Frame->True,Axes->False,PlotRange -> All,
     FrameLabel->{Quantity[None,"Meters"],Quantity[None,"Watts"/"Meters"^3/"Steradians"]}],
     {General::munfl}]

]
iPlanckRadiationLaw[t_,"MaxFrequency"]:=Module[{temperature=UnitConvert[t,"Kelvins"]},
   	If[Not[NumericQ[QuantityMagnitude[t]]],Return[$Failed]];
	If[QuantityMagnitude[temperature] > 10^20, Return[Missing["NotAvailable"]]];
	UnitConvert[Quantity[1, "BoltzmannConstant"/"PlanckConstant"]*
		(3*temperature + temperature ProductLog[-(3/E^3)]), "Hertz"]
]
iPlanckRadiationLaw[t_,"MaxWavelength"]:=Module[{temperature=UnitConvert[t,"Kelvins"]},
   	If[Not[NumericQ[QuantityMagnitude[t]]],Return[$Failed]];
	If[QuantityMagnitude[temperature] > 10^20, Return[Missing["NotAvailable"]]];
	UnitConvert[Quantity[1, ("PlanckConstant"*"SpeedOfLight")/"BoltzmannConstant"]/
		(5*temperature + temperature ProductLog[-(5/E^5)]), "Nanometers"]
]
iPlanckRadiationLaw[t_,"MeanFrequency"]:=Module[{temperature=UnitConvert[t,"Kelvins"]},
   	If[Not[NumericQ[QuantityMagnitude[t]]],Return[$Failed]];
	If[QuantityMagnitude[temperature] > 10^20, Return[Missing["NotAvailable"]]];
	UnitConvert[Quantity[360/Pi^4*Zeta[5], "BoltzmannConstant"/"PlanckConstant"]*temperature, "Hertz"]
]
iPlanckRadiationLaw[t_,"MeanWavelength"]:=Module[{temperature=UnitConvert[t,"Kelvins"]},
   	If[Not[NumericQ[QuantityMagnitude[t]]],Return[$Failed]];
	If[QuantityMagnitude[temperature] > 10^20, Return[Missing["NotAvailable"]]];
	UnitConvert[Quantity[30/Pi^4*Zeta[3], "SpeedOfLight"*"PlanckConstant"/"BoltzmannConstant"]/temperature, "Nanometers"]
]
iPlanckRadiationLaw[t_,"Color"]:=Module[{wavelength=iPlanckRadiationLaw[t,"MaxWavelength"]},
	If[QuantityMagnitude[t] > 10^20, Return[Missing["NotAvailable"]]];
	If[Not[QuantityQ[wavelength]],Return[$Failed]];
	wavelength=QuantityMagnitude[wavelength,"Nanometers"];
	ColorData["VisibleSpectrum"][wavelength]
]
iPlanckRadiationLaw[t_,"MaxFrequencyColor"]:=Module[{wavelength, f=iPlanckRadiationLaw[t,"MaxFrequency"]},
	If[QuantityMagnitude[t] > 10^20, Return[Missing["NotAvailable"]]];
	If[Not[QuantityQ[f]],Return[$Failed]];
	wavelength=QuantityMagnitude[UnitConvert[Quantity[1,"SpeedOfLight"]/f,"Nanometers"]];
	ColorData["VisibleSpectrum"][wavelength]
]
iPlanckRadiationLaw[t_,"MeanWavelengthColor"]:=Module[{wavelength=iPlanckRadiationLaw[t,"MeanWavelength"]},
	If[QuantityMagnitude[t] > 10^20, Return[Missing["NotAvailable"]]];
	If[Not[QuantityQ[wavelength]],Return[$Failed]];
	wavelength=QuantityMagnitude[wavelength,"Nanometers"];
	ColorData["VisibleSpectrum"][wavelength]
]
iPlanckRadiationLaw[t_,"MeanFrequencyColor"]:=Module[{wavelength, f=iPlanckRadiationLaw[t,"MeanFrequency"]},
	If[QuantityMagnitude[t] > 10^20, Return[Missing["NotAvailable"]]];
	If[Not[QuantityQ[f]],Return[$Failed]];
	wavelength=QuantityMagnitude[UnitConvert[Quantity[1,"SpeedOfLight"]/f,"Nanometers"]];
	ColorData["VisibleSpectrum"][wavelength]
]
iPlanckRadiationLaw[t_,{w1_,w2_},"Wavelength"]:=Module[{temperature=QuantityMagnitude[UnitConvert[t,"Kelvins"]],
   	wavelength1=QuantityMagnitude[UnitConvert[w1,"Meters"]],wavelength2=QuantityMagnitude[UnitConvert[w2,"Meters"]],
   	factor=Quantity[1, ("PlanckConstant"*"SpeedOfLight")/"BoltzmannConstant"]/(Quantity[1,"Kelvins"]*Quantity[1,"Meters"])},
   	If[Not[NumericQ[QuantityMagnitude[t]]&&(NumericQ[QuantityMagnitude[w1]]||Abs[QuantityMagnitude[w1]]===Infinity)&&
   		(NumericQ[QuantityMagnitude[w2]]||Abs[QuantityMagnitude[w2]]===Infinity)],Return[$Failed]];
   	factor=SetPrecision[factor,$MachinePrecision];
   	UnitConvert[Quantity[2, "PlanckConstant"*"SpeedOfLight"^2/"Steradians"]*Quantity[1,"Meters"]^-4*
     	Quiet[NIntegrate[1/((-1 + E^(factor/(temperature*w)))*w^5),{w,wavelength1,wavelength2}],{NIntegrate::ncvb}],
     	"Watts"/"Meters"^2/"Steradians"]
]
iPlanckRadiationLaw[t_,{f1_,f2_},"Frequency"]:=Module[{temperature=QuantityMagnitude[UnitConvert[t,"Kelvins"]],
 	frequency1=QuantityMagnitude[UnitConvert[f1,"Hertz"]],
   	frequency2=QuantityMagnitude[UnitConvert[f2,"Hertz"]],
   	factor=Quantity[1, "PlanckConstant"*"Hertz"/"BoltzmannConstant"]/Quantity[1,"Kelvins"]},
   	If[Not[NumericQ[QuantityMagnitude[t]]&&(NumericQ[QuantityMagnitude[f1]]||Abs[QuantityMagnitude[f1]]===Infinity)&&
   		(NumericQ[QuantityMagnitude[f2]]||Abs[QuantityMagnitude[f2]]===Infinity)],
   		Return[$Failed]];
   	factor=SetPrecision[factor,$MachinePrecision];
   	UnitConvert[Quantity[2, "PlanckConstant"*"Hertz"^4/"SpeedOfLight"^2/"Steradians"]*
   		Quiet[NIntegrate[f^3/((-1 + E^(factor*f/temperature))),{f,frequency1,frequency2}],{NIntegrate::ncvb}],
   		"Watts"/"Meters"^2/"Steradians"]
]

iPlanckRadiationLaw[t_,w_,"Wavelength"]:=Module[
	{temperature=If[MatchQ[t,_QuantityVariable],t,UnitConvert[t,"Kelvins"]],wavelength=w},
   If[Not[NumericQ[QuantityMagnitude[t]]||MatchQ[t,_QuantityVariable]],Return[$Failed]];
   If[Not[NumericQ[QuantityMagnitude[w]]||Abs[QuantityMagnitude[w]]===Infinity||MatchQ[w,_QuantityVariable]],Return[$Failed]];
   If[FreeQ[{t,w},QuantityVariable],
   	UnitConvert[Quantity[2, "PlanckConstant"*"SpeedOfLight"^2/"Steradians"]/
     	((-1 + E^(Quantity[1, ("PlanckConstant"*"SpeedOfLight")/"BoltzmannConstant"]/
     	(temperature*wavelength)))*wavelength^5),"Watts"/"Meters"^3/"Steradians"],
     Quantity[2, "PlanckConstant"*"SpeedOfLight"^2/"Steradians"]/
     	((-1 + E^(Quantity[1, ("PlanckConstant"*"SpeedOfLight")/"BoltzmannConstant"]/
     	(temperature*wavelength)))*wavelength^5)
   ]
]
iPlanckRadiationLaw[t_,f_,"Frequency"]:=Module[{temperature=If[MatchQ[t,_QuantityVariable],t,UnitConvert[t,"Kelvins"]]},
   If[Not[NumericQ[QuantityMagnitude[t]]||MatchQ[t,_QuantityVariable]],Return[$Failed]];
   If[Not[NumericQ[QuantityMagnitude[f]]||Abs[QuantityMagnitude[f]]===Infinity||MatchQ[f,_QuantityVariable]],Return[$Failed]];
   If[FreeQ[{t,f},QuantityVariable],
   	UnitConvert[Quantity[2, "PlanckConstant"/"SpeedOfLight"^2/"Steradians"]*f^3/
     	(-1 + E^(Quantity[1, "PlanckConstant"/"BoltzmannConstant"]*f/
     	temperature)),"Watts"/"Meters"^2/"Hertz"/"Steradians"],
     Quantity[2, "PlanckConstant"/"SpeedOfLight"^2/"Steradians"]*f^3/
     	(-1 + E^(Quantity[1, "PlanckConstant"/"BoltzmannConstant"]*f/
     	temperature))
   ]
]

iPlanckRadiationLaw[___]:=$Failed

End[];