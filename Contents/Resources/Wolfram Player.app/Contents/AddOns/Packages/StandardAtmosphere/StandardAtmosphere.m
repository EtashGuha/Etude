(* ::Package:: *)

(* :Title: StandardAtmosphere *)

(* :Author: Barbara Ercolano *)

(* :Summary: 
This package provides functions giving the values of atmospheric 
quantities for given altitudes and plots of these quantities versus the 
geometric altitude *)

(* :Context: StandardAtmosphere` *)

(* :Copyright: Copyright 1997-2007, Wolfram Research, Inc. *)

(* :Source: The Handbook of Chemistry and Physics, 73rd edition, 1992-1993 *)

(* :Mathematica Version: 3.0 *)

(* :History:
   Version 1.0 by Barbara Ercolano, 1997
   Version 1.1 changes by ECM, 1998
   Edited to fit the new paclet structure, 2006
*)

(* :Package Version: 1.1 *)


BeginPackage["StandardAtmosphere`", "Units`"];

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"StandardAtmosphere`"],
StringMatchQ[#,StartOfString~~"StandardAtmosphere`*"]&]//ToExpression;
];

If[Not@ValueQ[CollisionFrequency::usage],CollisionFrequency::usage=
"CollisionFrequency[altitude] gives the collision frequency at the \
specified altitude."];

If[Not@ValueQ[DynamicViscosity::usage],DynamicViscosity::usage=
"DynamicViscosity[altitude] gives the coefficient of dynamic viscosity at \
the specified altitude."];

If[Not@ValueQ[GravityAcceleration::usage],GravityAcceleration::usage=
"GravityAcceleration[altitude] gives the 45 degree latitude acceleration of \
gravity at the specified altitude."];

If[Not@ValueQ[KinematicViscosity::usage],KinematicViscosity::usage=
"KinematicViscosity[altitude] gives the kinematic viscosity at the \
specified altitude."];

If[Not@ValueQ[KineticTemperature::usage],KineticTemperature::usage=
"KineticTemperature[altitude] gives the kinetic temperature at the \
specified altitude."];

If[Not@ValueQ[MeanDensity::usage],MeanDensity::usage=
"MeanDensity[altitude] gives the mean density of air at the specified \
altitude."];

If[Not@ValueQ[MeanFreePath::usage],MeanFreePath::usage=
"MeanFreePath[altitude] gives the mean free path at the specified \
altitude."];

If[Not@ValueQ[MeanMolecularWeight::usage],MeanMolecularWeight::usage=
"MeanMolecularWeight[altitude] gives the mean molecular weight at the \
specified altitude."];

If[Not@ValueQ[MeanParticleSpeed::usage],MeanParticleSpeed::usage=
"MeanParticleSpeed[altitude] gives the mean particle speed at the specified \
altitude."];

If[Not@ValueQ[NumberDensity::usage],NumberDensity::usage=
"NumberDensity[altitude] gives the total number density of the mixture of \
neutral atmospheric gas particles at the specified altitude."];

If[Not@ValueQ[Pressure::usage],Pressure::usage=
"Pressure[altitude] gives the total atmospheric pressure at the \
specified altitude."];

If[Not@ValueQ[PressureScaleHeight::usage],PressureScaleHeight::usage=
"PressureScaleHeight[altitude] gives the local pressure scale height of the \
mixture of gases comprising the atmosphere at the specified altitude."];

If[Not@ValueQ[SoundSpeed::usage],SoundSpeed::usage=
"SoundSpeed[altitude] gives the speed of sound at the specified altitude."];

If[Not@ValueQ[ThermalConductivityCoefficient::usage],ThermalConductivityCoefficient::usage=
"ThermalConductivityCoefficient[altitude] gives the coefficient of thermal \
conductivity at the specified altitude."];


If[Not@ValueQ[AtmosphericPlot::usage],AtmosphericPlot::usage=
"AtmosphericPlot[property, options] plots the specified property as a \
function of geometric altitude. Properties are \
CollisionFrequency, DynamicViscosity, GravityAcceleration, \
KinematicViscosity, KineticTemperature, MeanDensity, MeanFreePath, \
MeanMolecularWeight, MeanParticleSpeed, NumberDensity, Pressure, \
PressureScaleHeight, SoundSpeed, and ThermalConductivityCoefficient."];


Options[AtmosphericPlot] = Options[Plot];


Begin["`Private`"];

Pressure[alt_] :=
	Module[{p},
	  (
		p
	  ) /; ((p = atmosphericDataInterpolation[Pressure, alt]) =!= $Failed)
	]

CollisionFrequency[alt_] :=
	Module[{cf},
	  (
		cf
	  ) /; ((cf = atmosphericDataInterpolation[CollisionFrequency, alt]) =!= $Failed)
	]

DynamicViscosity[alt_] :=
	Module[{dv},
	  (
		dv
	  ) /; ((dv = atmosphericDataInterpolation[DynamicViscosity, alt]) =!= $Failed)
	]

GravityAcceleration[alt_] :=
	Module[{ga},
	  (
		ga
	  ) /; ((ga = atmosphericDataInterpolation[GravityAcceleration, alt]) =!= $Failed)
	]

KinematicViscosity[alt_] :=
	Module[{kv},
	  (
		kv
	  ) /; ((kv = atmosphericDataInterpolation[KinematicViscosity, alt]) =!= $Failed)
	]

KineticTemperature[alt_] := 
	Module[{kt},
	  (
		kt
	  ) /; ((kt = atmosphericDataInterpolation[KineticTemperature, alt]) =!= $Failed)
	]

MeanDensity[alt_] :=
	Module[{md},
	  (
		md
	  ) /; ((md = atmosphericDataInterpolation[MeanDensity, alt]) =!= $Failed)
	]

MeanFreePath[alt_] :=
	Module[{mfp},
	  (
		mfp
	  ) /; ((mfp = atmosphericDataInterpolation[MeanFreePath, alt]) =!= $Failed)
	]

MeanMolecularWeight[alt_] :=
	Module[{mmw},
	  (
		mmw
	  ) /; ((mmw = atmosphericDataInterpolation[MeanMolecularWeight, alt]) =!= $Failed)
	]

MeanParticleSpeed[alt_] :=
	Module[{mps},
	  (
		mps
	  ) /; ((mps = atmosphericDataInterpolation[MeanParticleSpeed, alt]) =!= $Failed)
	]

PressureScaleHeight[alt_] :=
	Module[{psh},
	  (
		psh
	  ) /; ((psh = atmosphericDataInterpolation[PressureScaleHeight, alt]) =!= $Failed)
	]

NumberDensity[alt_] :=
	Module[{nd},
	  (
		nd
	  ) /; ((nd = atmosphericDataInterpolation[NumberDensity, alt]) =!= $Failed)
	]

SoundSpeed[alt_] :=
	Module[{ss},
	  (
		ss
	  ) /; ((ss = atmosphericDataInterpolation[SoundSpeed, alt]) =!= $Failed)
	]

ThermalConductivityCoefficient[alt_] :=
	Module[{tcc},
	  (
		tcc
	  ) /; ((tcc = atmosphericDataInterpolation[ThermalConductivityCoefficient, alt]) =!= $Failed)
	]


$Alts={-5000,-4500,-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,0,500,1000,
    1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,
    8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,
    14500,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,
    26000,27000,28000,29000,30000,31000,32000,33000,34000,35000,36000,38000,
    40000,42000,44000,46000,48000,50000,52000,54000,56000,58000,60000,65000,
    70000,75000,80000,85000,85500,86000,90000,95000,100000,110000,120000,
    130000,140000,150000,160000,170000,180000,190000,200000,210000,220000,
    240000,260000,280000,300000,320000,340000,360000,380000,400000,450000,
    500000,550000,600000,650000,700000,750000,800000,850000,900000,950000,
    1000000};

altitudeHm=
  Meter {-5004, -4503, -4003, -3502, -3001, -2501, -2001, -1500, -1000, -500, 
      0, 500, 1000, 1500, 1999, 2499, 2999, 3498, 3997, 4497, 4996, 5495, 
      5994, 6493, 6992, 7491, 7990, 8489, 8987, 9486, 9984, 10483, 10981, 
      11479, 11977, 12475, 12973, 13471, 13969, 14467,14965,  15960, 16955, 
      17949, 18943, 19937, 20931, 21924, 22917, 23910, 24902, 25894, 26886, 
      27877, 28868, 29859, 30850, 31840, 32830, 33819, 34808, 35797,  3774, 
      39750, 41724, 43698, 45669, 47640, 49610, 51578, 53545, 55511, 57476, 
      59539, 64342, 69238, 74125, 79006, 83878, 84365, 84852, 88744, 93610, 
      98451, 108129, 117777, 127395, 136983, 146542, 156072, 165572, 175043, 
      184486, 193899, 203270, 212641, 231268, 249784, 268187, 286480, 304663, 
      322738, 340705, 358565, 376320, 420250, 463540, 506202, 548252, 589701, 
      630563, 670850, 710574, 749747, 788380, 826484, 864071};

iGravityAcceleration["Data"]= {9.8221, 9.8295, 9.8190, 9.8175, 
    9.8159, 9.8144, 9.8128, 9.8113,9.8097, 9.8082, 9.8066, 9.8051, 9.8036, 
    9.8020, 9.8005, 9.7989, 9.7974,9.7959, 9.7943,9.7928, 9.7912, 9.7897, 
    9.7882, 9.7866, 9.7851, 9.7836, 9.7820, 9.7805, 9.7789, 9.7774, 9.7759, 
    9.7743, 9.7728, 9.7713, 9.7697, 9.7682, 9.7667, 9.7651, 9.7636, 9.7621, 
    9.7605, 9.7575, 9.7544, 9.7513, 9.7483, 9.7452, 9.7422, 9.7391, 9.7361, 
    9.7330, 9.7300, 9.7269, 9.7239, 9.7208, 9.7178, 9.7147, 9.7117, 9.7087, 
    9.7056, 9.7026, 9.6995, 9.6965, 9.6904, 9.6844, 9.6783, 9.6723, 9.6662, 
    9.6602, 9.6542, 9.6482, 9.6421, 9.6361, 9.6316, 9.6241, 9.6091, 9.5942, 
    9.5793, 9.5644, 9.5496, 9.5481, 9.5466, 9.5348, 9.5200, 9.5052, 9.4759, 
    9.4466, 9.4175, 9.3886, 9.3597, 9.3310, 9.3024, 9.2740, 9.2457, 9.2175, 
    9.1895, 9.1615, 9.1061, 9.0511, 8.9966, 8.9427, 8.8892, 8.8361, 8.7836, 
    8.7315, 8.6799, 8.5529, 8.4286, 8.3070, 8.1880, 8.0716, 7.9576, 7.8460, 
    7.7368, 7.6298, 7.5250, 7.4224, 7.3218};
iGravityAcceleration["OldUnits"] = Units`Meter/Units`Second^2;
iGravityAcceleration["Quantity"] = "Meters"/"Seconds"^2;

iPressureScaleHeight["Data"] = {9371.8, 9278.2, 9184.5, 9090.8, 8997.1, 
    8903.4, 8809.6, 8715.5, 8622.1, 8528.3, 8434.5, 8340.7, 8246.9, 8153.0, 
    8059.2, 7965.3, 7871.4, 7777.5, 7683.6, 7589.7, 7495.7, 7401.8, 7307.8, 
    7213.8, 7119.8, 7025.8, 6931.7, 6837.7, 6743.6, 6649.5, 6555.4, 6461.3, 
    6367.2, 6364.6, 6365.6, 6366.6, 6367.6, 6368.6, 6369.6, 6370.6, 6371.6, 
    6373.6, 6375.6, 6377.6, 6379.6, 6381.6, 6411.0, 6442.3, 6473.6, 6504.9, 
    6536.2, 6567.5, 6598.9, 6630.2, 6661.6, 6692.0, 6724.3, 6755.7, 6831.2, 
    6915.4, 6999.5, 7083.7, 7252.1, 7420.6, 7589.2, 7757.9, 7926.7, 8042.4, 
    8047.4, 8004.3, 7845.3, 7686.2, 7527.0, 7367.8, 6969.1, 6569.9, 6244.9, 
    5961.7, 5678.0, 5649.6, 5621, 5636, 5727, 6009, 7723, 12091, 16288, 
    20025, 23380, 26414, 29175, 31703, 34030, 36183, 38113, 40043, 43405, 
    46346, 48925, 51193, 53199, 54996, 56637, 58178, 59678, 63644, 68785, 
    76427, 88244, 105992, 130630, 161074, 193862, 224737, 250895, 271754, 
    288203};
iPressureScaleHeight["OldUnits"] = Units`Meter;
iPressureScaleHeight["Quantity"] = "Meters";

AtmosphericPlot[PressureScaleHeight, opts___?OptionQ]:=
 ListPlot[
    Transpose[{iPressureScaleHeight["Data"], $Alts/1000.}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"PressureScaleHeight, Km",
                 " Altitude, Km"},
    PlotRange->All]



iNumberDensity["Data"] = {4.0151 10^25, 3.8445 10^25, 3.6795 10^25, 
    3.5201 10^25, 3.3660 10^25, 3.2171 10^25, 3.1017 10^25, 2.9346 10^25, 
    2.8007 10^25, 2.6715 10^25, 2.5470 10^25, 2.4269 10^25, 2.3113 10^25, 
    2.2000 10^25, 2.0928 10^25,1.9897 10^25, 1.8905 10^25, 1.7952 10^25, 
    1.7036 10^25, 1.6156 10^25, 1.5312 10^25, 1.4502 10^25, 1.3725 10^25, 
    1.2980 10^25, 1.2267 10^25, 1.1585 10^25, 1.0932 10^25, 1.0308 10^25, 
    9.7110 10^24, 9.1413 10^24, 8.5976 10^24, 8.0790 10^24, 7.5854 10^24, 
    7.0157 10^24, 6.4857 10^24, 5.9958 10^24, 5.5430 10^24, 5.1244 10^24, 
    4.7375 10^24, 4.3799 10^24, 4.0493 10^24, 3.4612 10^24, 2.9587 10^24, 
    2.5292 10^24, 2.1622 10^24,1.8486 10^24, 1.5742 10^24, 1.3413 10^24, 
    1.1437 10^24, 9.7591 10^23, 8.3341 10^23, 7.1225 10^23, 6.0916 10^23, 
    5.2138 10^23, 4.4657 10^23, 3.8278 10^23, 3.2833 10^23, 2.8133 10^23, 
    2.4062 10^23, 2.0558 10^23, 1.7597 10^23, 1.5090 10^23, 1.1158 10^23, 
    8.3077 10^22, 6.2266 10^22, 4.6965 10^22, 3.5640 10^22, 2.7376 10^22, 
    2.1351 10^22, 1.6750 10^22, 1.3286 10^22, 1.0488 10^22, 8.2390 10^21, 
    6.4387 10^21, 3.3934 10^21, 1.7222 10^21, 8.3003 10^20, 3.8378 10^20, 
    1.7090 10^20, 1.5727 10^20, 1.447 10^20, 7.116 10^19, 2.920 10^19, 
    1.189 10^19, 2.144 10^18, 5.107 10^17, 1.930 10^17, 9.322 10^16, 
    5.186 10^16, 3.162 10^16, 2.055 10^16, 1.400 10^16, 9.887 10^15, 
    7.182 10^15, 5.611 10^15, 4.040 10^15, 2.420 10^15, 1.515 10^15, 
    9.807 10^14,  6.509 10^14, 4.405 10^14, 3.029 10^14, 2.109 10^14, 
    1.485 10^14, 1.056 10^14, 4.678 10^13, 2.192 10^13, 1.097 10^13, 
    5.950 10^12, 3.540 10^12, 2.311 10^12, 1.637 10^12, 1.234 10^12, 
    9.717  10^11, 7.876 10^11, 6.505 10^11, 5.442 10^11}  ;
iNumberDensity["OldUnits"] = Units`Meter^(-3);
iNumberDensity["Quantity"] = "Meters"^(-3);

AtmosphericPlot[NumberDensity, opts___?OptionQ]:=
 ListPlot[
    Transpose[{Log[10,iNumberDensity["Data"]], $Alts/1000}], 
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"Log[10, NumberDensity]",
                 "Altitude, Km"},
    PlotRange->All]

AtmosphericPlot[GravityAcceleration, opts___?OptionQ]:=
 ListPlot[
   Transpose[{iGravityAcceleration["Data"], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"GravityAcceleration, m/s^2",
                 "Altitude, Km"},
    Axes->False,
    PlotRange->All]

iMeanParticleSpeed["Data"] = {484.15,481.69, 479.22, 476.73, 
    474.23, 471.71, 469.69, 466.65, 464.09, 461.53, 458.94, 456.35, 453.74, 
    451.12, 448.48, 445.82, 443.15, 440.47, 437.76, 435.05, 432.31, 429.56, 
    426.79, 424.00, 421.20, 418.37, 415.53, 412.67, 409.79, 406.89, 403.97, 
    401.03, 398.07, 397.95, 397.95,397.95, 397.95, 397.95 , 397.95 , 397.95 , 
    397.95 , 397.95 , 397.95 , 397.95 , 397.95 , 397.95, 398.81, 399.72, 
    400.62, 401.53, 402.43, 403.33, 404.23, 405.12, 406.01, 406.91, 407.79,  
    408.68, 410.90, 413.35, 415.79, 418.22, 423.03, 427.78, 432.48, 437.13, 
    441.72, 444.79, 444.79, 443.46, 438.90, 434.29, 429.63, 424.93, 412.95, 
    400.64, 390.30, 381.05, 371.59, 370.63, 369.7, 369.9, 372.6, 381.4, 
    431.7, 539.3, 625.0, 691.9, 746.5, 792.2, 831.3, 865.3, 895.1, 921.6, 
    944.05, 966.5, 1003.2, 1033.5, 1058.7, 1079.7, 1097.4, 1112.4, 1125.5, 
    1137.4, 1148.5, 1177.4, 1215.0, 1271.5, 1356.4, 1476.0, 1627.0, 1793.9, 
    1954.3, 2089.6, 2192.6, 2266.4, 2318.1};
iMeanParticleSpeed["OldUnits"] = Units`Meter/Units`Second;
iMeanParticleSpeed["Quantity"] = "Meters"/"Seconds";

AtmosphericPlot[MeanParticleSpeed, opts___?OptionQ]:=
  ListPlot[
    Transpose[{iMeanParticleSpeed["Data"], $Alts/1000}],
    opts,
    Axes->False,
    Joined->True,
    Frame->True,
    FrameLabel->{"MeanParticleSpeed, m/s",
                 "Altitude, Km"},
   PlotRange->All]

iCollisionFrequency["Data"] = {1.506 10^10, 1.0961 10^10, 
    1.0437 10^10, 9.9328 10^9, 9.4481 10^9, 8.9824 10^9, 8.6231 10^9, 
    8.1056 10^9, 7.6934 10^9, 7.2980 10^9, 6.9189 10^9, 6.5555 10^9, 
    6.2075 10^9, 5.8743 10^9, 5.5554 10^9, 5.2504 10^9, 4.9588 10^9, 
    4.6802  10^9, 4.4141 10^9, 4.1602 10^9, 3.9180 10^9, 3.6871 10^9, 
    3.4671 10^9, 3.2577 10^9, 3.0584 10^9, 2.8689 10^9, 2.6888 10^9, 
    2.5178 10^9, 2.3555 10^9, 2.2016 10^9, 2.0558 10^9, 1.9177 10^9, 
    1.7871 10^9, 1.6525 10^9, 1.5277 10^9, 1.4123 10^9, 1.3056  10^9, 
    1.2070 10^9, 1.1159 10^9, 1.0317 10^9, 9.5380 10^8, 8.1528 10^8, 
    6.9691 10^8, 5.9576 10^8, 5.0931 10^8, 4.3543 10^8, 3.7161 10^8, 
    3.1733 10^8, 2.7119 10^8, 2.3194 10^8, 1.9852 10^8, 1.7004 10^8, 
    1.4575 10^8, 1.2502 10^8, 1.0732 10^8, 9.2192 10^7, 7.9251 10^7, 
    6.8175 10^7, 5.8522 10^7, 5.0297 10^7, 4.3307 10^7, 3.7356 10^7, 
    2.7939 10^7, 2.1036 10^7, 1.5939 10^7, 1.2152 10^7, 9.3182 10^6, 
    7.2075 10^6, 5.6201 10^6, 4.3966 10^6, 3.4515 10^6, 2.6961 10^6, 
    2.0952 10^6, 1.6195 10^6, 8.2945 10^5, 4.0839 10^5, 1.9175 10^5, 
    8.6559 10^4, 9.8858 10^3, 3.4501 10^3, 3.17 10^3, 1.56 10^3, 6.44 10^2, 
    2.68 10^2, 5.48 10^1, 1.63 10^1, 7.1 10^0, 3.8 10^0, 2.3 10^0, 1.5 10^0, 
    1. 10^0, 7.2 10^-1, 5.2 10^-1, 3.9 10^-1, 3.1 10^-1, 2.3 10^-1, 
    1.4 10^-1, 9.3 10^-2, 6.1 10^-2, 4.2 10^-2, 2.9 10^-2, 2. 10^-2, 
    1.4 10^-2, 1 10^-2, 7.2 10^-3, 3.3 10^-3, 1.6 10^-3, 8.3 10^-4, 
    4.8 10^-4, 3.1 10^-4, 2.2 10^-4, 1.7 10^-4, 1.4 10^-4, 1.2 10^-4, 
    1. 10^-4, 8.7 10^-5, 7.5 10^-6} ;
iCollisionFrequency["OldUnits"] = Units`Second^-1;
iCollisionFrequency["Quantity"] = "Seconds"^-1;

AtmosphericPlot[CollisionFrequency, opts___?OptionQ]:=
  ListPlot[
    Transpose[{Log[10, iCollisionFrequency["Data"]], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"Log[10, CollisionFrequency]",
                 "Altitude, Km"},
   PlotRange->All,
   Axes->False]

iMeanFreePath["Data"] = {4.2078 10^-8, 4.395 10^-8, 4.5915 10^-8, 
    4.7995 10^-8, 5.0193 10^-8, 5.2515 10^-8, 5.4469 10^-8, 5.7571 10^-8, 
    6.0324 10^-8, 6.3240 10^-8, 6.6332 10^-8, 6.9613 10^-8, 7.3095 10^-8, 
    7.6795 10^-8, 8.0728 10^-8, 8.4912 10^-8, 8.9367 10^-8, 9.4113 10^-8, 
    9.9173 10^-8, 1.0457 10^-7, 1.1034 10^-7, 1.1650 10^-7, 1.2310 10^-7, 
    1.3016 10^-7, 1.3772 10^-7, 1.4583 10^-7, 1.5454 10^-7, 1.6390 10^-7, 
    1.7397 10^-7, 1.8482 10^-7, 1.9651 10^-7, 2.0912 10^-7, 2.2274 10^-7, 
    2.4081 10^-7, 2.6049 10^-7, 2.8178 10^-7, 3.0479 10^-7, 3.2969 10^-7, 
    3.5662 10^-7, 3.8574 10^-7, 4.1723 10^-7, 4.8812 10^-7, 5.7102 10^-7, 
    6.6797 10^-7, 7.8135 10^-7, 9.1393 10^-7, 1.0732 10^-6, 1.2596 10^-6, 
    1.4772 10^-6, 1.7312 10^-6, 2.0272 10^-6, 2.3720 10^-6, 2.7734 10^-6, 
    3.2404 10^-6, 3.7832 10^-6, 4.4137 10^-6, 5.1456 10^-6, 5.9945 10^-6, 
    7.0212 10^-6, 8.2182 10^-6, 9.6010 10^-6, 1.1196 10^-5, 1.5141 10^-5, 
    2.0336 10^-5, 2.7133 10^-5, 3.5973 10^-5, 4.7404 10^-5, 6.1713 10^-5, 
    7.9130 10^-5, 1.0086 10^-4, 1.2716 10^-4, 1.6108 10^-4, 2.0506 10^-4, 
    2.6239 10^-4, 4.9787 10^-4, 9.8104 10^-4, 2.0354 10^-3, 4.4022 10^-3, 
    9.8858 10^-3, 1.0743 10^-2, 1.17 10^-2, 2.37 10^-2, 5.79 10^-2, 
    1.42 10^-1, 7.88 10^-1,   3.31, 8.8, 18, 33, 53, 82, 1.2 10^2, 1.7 10^2, 
    2.4 10^2,3.3 10^2, 4.2 10^2, 7 10^2, 1.1 10^3, 1.7 10^3, 2.6 10^3, 
    3.8 10^3, 5.6 10^3, 8. 10^3, 1.1 10^4, 1.6 10^4, 3.6 10^4, 7.7 10^4, 
    1.5 10^5, 2.8 10^5, 4.8 10^5, 7.3 10^5, 1. 10^6, 1.4 10^6, 1.7 10^6, 
    2.1 10^6, 2.6 10^6, 3.1 10^6} ;
iMeanFreePath["OldUnits"] = Units`Meter;
iMeanFreePath["Quantity"] = "Meters";

AtmosphericPlot[MeanFreePath, opts___?OptionQ]:=
  ListPlot[
     Transpose[{Log[10, iMeanFreePath["Data"]], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"Log[10, MeanFreePath]",
                 "Altitude, Km"},
    PlotRange->All,
    Axes->False]

iMeanMolecularWeight["Data"] = {28.964, 28.964, 28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  28.964,  
    28.964,  28.964,  28.964,  28.964,  28.964, 28.95, 28.91, 28.73, 28.40, 
    27.27, 26.20, 25.44, 24.75, 24.10, 23.49, 22.90, 22.34, 21.81, 21.30, 
    20.835, 20.37, 19.56, 18.85, 18.24, 17.73, 17.29, 16.91, 16.57, 16.27, 
    15.98, 15.25, 14.33, 13.09, 11.51, 9.72, 8.00, 6.58, 5.54, 4.85, 4.4, 
    4.12, 3.94};
iMeanMolecularWeight["OldUnits"] = Units`Kilogram/(Units`Kilo Units`Mole);
iMeanMolecularWeight["Quantity"] = "Kilograms"/"Kilomoles";

AtmosphericPlot[MeanMolecularWeight, opts___?OptionQ]:=
 ListPlot[  Transpose[{iMeanMolecularWeight["Data"], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"MolecularWeight, Kg/kmol",
                 "Altitude, Km"},
    PlotRange -> All,
    Axes->False]

iKineticTemperature["Data"] = {320.676, 317.421, 314.166, 310.913, 307.659, 
    304.406, 301.154, 297.902, 294.651, 291.400, 288.150, 284.900, 281.651, 
    278.402, 275.154, 271.906, 268.659, 265.413, 262.166, 258.921, 255.676, 
    252.431, 249.187, 245.943, 242.700, 239.457, 236.215, 232.974, 229.733, 
    226.492, 223.252, 220.013, 216.774, 216.650, 216.650, 216.650, 216.650, 
    216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 216.650, 
    216.650, 217.581, 218.574, 219.567, 220.560, 221.552, 222.544, 223.536, 
    224.527, 225.518, 226.509, 227.500, 228.490, 230.973, 233.473, 236.513, 
    239.282, 244.818, 250.350, 255.878, 261.403, 266.925, 270.650, 270.650, 
    269.031, 263.524, 258.019, 252.518, 247.021, 233.292, 219.585, 208.399, 
    198.639, 188.893, 187.920, 186.87, 186.87, 188.42, 195.08, 240.00, 
    360.00, 469.27, 559.63, 634.39, 696.29, 747.57,  790.07, 825.16, 854.56, 
    878.84, 899.01, 929.73, 950.99, 965.75, 976.01, 983.16, 988.15, 991.65, 
    994.10, 995.83, 998.22, 999.24, 999.67, 999.85, 999.93, 999.97, 999.00, 
    999.99, 1000, 1000, 1000, 1000};
iKineticTemperature["OldUnits"] = Units`Kelvin;
iKineticTemperature["Quantity"] = "Kelvins";

AtmosphericPlot[KineticTemperature, opts___?OptionQ]:=ListPlot[
    Transpose[{iKineticTemperature["Data"], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    Axes->False,
    FrameLabel->{"KineticTemperature, K",
                 "Altitude, Km"},
    PlotRange->All]

iPressure["Data"]= {1.7776 10^3, 1.6848 10^3, 1.5959 10^3, 1.5109 10^3, 
    1.4297 10^3, 1.3520 10^3, 1.2778 10^3, 1.2069 10^3, 1.1393 10^3, 
    1.0747 10^3, 1.01325 10^3, 9.5461 10^2, 8.9876 10^2, 8.4559 10^2, 
    7.9501 10^2, 7.4691 10^2, 7.0121 10^2, 6.5780 10^2, 6.1660 10^2, 
    5.7752 10^2, 5.4048 10^2,  5.0539 10^2, 4.7217 10^2, 4.4075 10^2, 
    4.1105 10^2, 3.8299 10^2, 3.5651 10^2, 3.3154 10^2, 3.0800 10^2, 
    2.8584 10^2, 2.6499 10^2, 2.4540 10^2, 2.2699 10^2, 2.0984 10^2, 
    1.9299 10^2, 1.7934 10^2, 1.6579 10^2, 1.5327 10^2, 1.4170 10^2, 
    1.3100 10^2, 1.2111 10^2, 1.0352 10^2, 8.8497 10^1, 7.5652 10^1, 
    6.4674 10^1, 5.5293 10^1, 4.7289 10^1, 4.0475 10^1, 3.4668 10^1, 
    2.9717 10^1, 2.5492 10^1, 2.1883 10^1, 1.8799 10^1, 1.6161 10^1, 
    1.3904 10^1, 1.1970 10^1, 1.0312 10^1, 8.8906 , 7.6730 , 6.6341 , 
    5.7459 , 4.9852 ,3.7713, 2.8713, 2.1996, 1.6949, 1.3134, 1.0229, 
    7.9779 10^-1, 6.2214 10^-1, 4.8337 10^-1, 3.7362 10^-1, 2.8723 10^-1, 
    2.1958 10^-1, 1.0929 10^-1, 5.2205 10^-2, 2.3881 10^-2, 1.0524 10^-2, 
    4.4568 10^-3, 4.0802 10^-3, 3.7338 10^-3, 1.8359 10^-3, 7.5966 10^-4, 
    3.2011 10^-4, 7.1042 10^-5, 2.5382 10^-5, 1.2505 10^-5,7.2028 10^-6, 
    4.5422 10^-6, 3.0395 10^-6, 2.1210 10^-6, 1.5279 10^-6, 1.1266 10^-6, 
    8.4736 10^-7, 6.4756 10^-7, 5.0149 10^-7, 3.1059 10^-7, 1.9894 10^-7, 
    1.3076 10^-7, 8.7704 10^-8, 5.9796 10^-8, 4.1320 10^-8, 2.8878 10^-8, 
    2.0384 10^-8, 1.4518 10^-8, 6.4468 10^-9, 3.0236 10^-9, 1.4137 10^-9, 
    8.2130 10^-10, 4.8865 10^-10, 3.1908 10^-10, 2.2599 10^-10, 
    1.7036 10^-10, 1.3415 10^-10, 1.0873 10^-10, 8.9816 10^-11, 
    7.5138 10^-11}  ;
iPressure["OldUnits"] = Units`Milli*Units`Bar;
iPressure["Quantity"] = "Millibars";

AtmosphericPlot[Pressure, opts___?OptionQ]:=
  ListPlot[
    Transpose[{Log[10, iPressure["Data"]], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"Log[10, Pressure]",
                 "Altitude, Km"},
    PlotRange->All, 
    Axes->False]

(* values from the 'U.S. Standard Atmosphere, 1976' model *)
iMeanDensity["Data"] = {1.9311, 1.8491, 1.7697, 1.6930, 1.6189, 
    1.5473, 1.4782, 1.4114, 1.3470, 1.2849, 1.2250, 1.1673, 1.1117, 1.0581, 
    1.0066, 9.5695 10^-1, 9.0925 10^-1, 8.6340 10^-1, 8.1935 10^-1, 
    7.7704 10^-1, 7.3643 10^-1, 6.9747 10^-1, 6.6011 10^-1, 6.2431 10^-1, 
    5.9002 10^-1, 5.5719 10^-1, 5.2579 10^-1, 4.9576 10^-1, 4.6706 10^-1, 
    4.3966 10^-1, 4.1351 10^-1, 3.8857 10^-1, 3.6480 10^-1, 3.3743 10^-1, 
    3.1194 10^-1, 2.8838 10^-1, 2.6660 10^-1, 2.4646 10^-1, 2.2786 10^-1, 
    2.1066 10^-1, 1.9476 10^-1, 1.6647 10^-1, 1.4230 10^-1, 1.2165 10^-1, 
    1.0400 10^-1, 8.8910 10^-2,7.5715 10^-2, 6.4510 10^-2, 5.5006 10^-2, 
    4.6938 10^-2, 4.0084 10^-2, 3.4257 10^-2, 2.9298 10^-2, 2.5076 10^-2, 
    2.1478 10^-2, 1.8410 10^-2, 1.5792 10^-2, 1.3555 10^-2, 1.1573 10^-2, 
    9.8874 10^-3, 8.4634   10^-3, 7.2579 10^-3, 5.3666 10^-3, 3.9957 10^-3, 
    2.9948 10^-3, 2.2589 10^-3, 1.7142 10^-3, 1.3167 10^-3, 1.0269 10^-3, 
    8.0562 10^-4, 6.3901 10^-4, 5.0445 10^-4, 3.9627 10^-4, 3.0968 10^-4, 
    1.6321 10^-4, 8.2829 10^-5, 3.9921 10^-5, 1.8458 10^-5, 8.2196 10^-6, 
    7.5641 10^-6,6.958 10^-6, 3.416 10^-6, 1.393 10^-6, 5.604 10^-7, 
    9.708 10^-8,   2.222 10^-8, 8.152 10^-9, 3.831 10^-9, 2.076 10^-9, 
    1.233 10^-9, 7.815 10^-10, 5.194 10^-10, 3.581 10^-10, 2.541 10^-10, 
    1.846 10^-10, 1.367 10^-10, 7.858 10^-11, 4.742 10^-11, 2.971 10^-11, 
    1.916 10^-11, 1.264 10^-11, 8.503 10^-12, 5.805 10^-12, 4.013 10^-12, 
    2.803 10^-12, 1.184 10^-12, 5.215 10^-13, 2.384 10^-13, 1.137 10^-13, 
    5.712 10^-14, 3.070 10^-14, 1.788 10^-14, 1.136 10^-14, 7.824 10^-15, 
    5.759 10^-15, 4.453 10^-15, 3.561 10^-15 };
iMeanDensity["OldUnits"] = Units`Kilogram/Units`Meter^3;
iMeanDensity["Quantity"] = "Kilograms"/"Meters"^3;

AtmosphericPlot[MeanDensity, opts___?OptionQ]:=
  ListPlot[
    Transpose[{Log[10, iMeanDensity["Data"]], $Alts/1000}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"Log[10, MeanDensity]",
                 "Altitude, Km"},
    RotateLabel->True, 
    PlotRange->All,
    Axes->False]

iSoundSpeed["Data"] = {358.99, 357.16, 355.32, 353.48, 351.63, 
    349.76, 347.89, 346.00, 344.11, 342.21, 340.29, 338.37, 336.43, 334.49, 
    332.53, 330.56, 328.58, 326.59, 324.59, 322.57, 320.55, 318.50, 316.45, 
    314.39, 312.31, 310.21, 308.11, 305.98, 303.85, 301.7, 299.53, 297.35, 
    295.15, 295.07, 295.07, 295.07, 295.07, 295.07, 295.07, 295.07, 295.07, 
    295.07, 295.07, 295.07, 295.07, 295.07, 295.07, 296.38, 297.05, 297.72, 
    298.39, 299.06, 299.72, 300.39, 301.05, 301.71, 302.37, 303.02, 304.67, 
    306.49, 308.30, 310.10,313.67, 317.19, 320.67, 324.12, 327.52, 329.80, 
    329.80, 328.81, 325.43, 322.01, 318.56, 315.07, 306.19, 297.06, 289.40, 
    282.54, 275.52, 274.81} ;
iSoundSpeed["OldUnits"] = Units`Meter/Units`Second;
iSoundSpeed["Quantity"] = "Meters"/"Seconds";

AtmosphericPlot[SoundSpeed, opts___?OptionQ]:=
  ListPlot[
    Transpose[{iSoundSpeed["Data"], Take[$Alts/1000, Length[iSoundSpeed["Data"]]]}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"SoundSpeed, m/s",
                 "Altitude, Km"},
    PlotRange->All,
    Axes->False];

iDynamicViscosity["Data"]= {19.422, 19.273, 19.123, 
      18.972, 18.820, 18.668, 18.515, 18.361, 18.206, 18.050, 17.894, 17.737, 
      17.579, 17.420, 17.260, 17.099, 16.938, 16.775, 16.612, 16.448, 16.282, 
      16.116, 15.949, 15.781, 15.612, 15.442, 15.271, 15.099, 14.926, 14.752, 
      14.577, 14.400, 14.223, 14.216, 14.216, 14.216, 14.216, 14.216, 14.216,
      14.216, 14.216, 14.216, 14.216, 14.216, 14.216, 14.216, 14.267, 
      14.322, 14.376, 14.430, 14.484, 14.538, 14.592, 14.646, 14.699, 14.753, 
      14.806, 14.859, 14.992, 15.140, 15.287, 15.433, 15.723, 16.009, 16.293, 
      16.573, 16.851, 17.037, 17.037, 16.956, 16.600, 16.402, 16.121, 15.837, 
      15.116, 14.377, 13.759, 13.208, 12.647, 12.590} 10^-6;
iDynamicViscosity["OldUnits"] = Units`Newton Units`Second/ Units`Meter^2;
iDynamicViscosity["Quantity"] = "Newtons"*"Seconds"/"Meters"^2;

AtmosphericPlot[DynamicViscosity, opts___?OptionQ]:=
  ListPlot[
    Transpose[{10^6 iDynamicViscosity["Data"],
	 Take[$Alts/1000, Length[iDynamicViscosity["Data"]]]}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"DynamicViscosity, 10^-6 N s/m^2",
                 "Altitude, Km"},
    PlotRange->All,
    Axes->False];

iKinematicViscosity["Data"] = {1.0058 10^-5, 1.0423 10^-5, 
    1.0806 10^-5, 1.1206 10^-5, 1.1225 10^-5, 1.2065 10^-5, 1.2525 10^-5, 
    1.3009 10^-5, 1.3516 10^-5, 1.4048 10^-5, 1.4607 10^-5, 1.5195 10^-5, 
    1.5813 10^-5, 1.6463 10^-5, 1.7147 10^-5, 1.7868 10^-5, 1.8628 10^-5, 
    1.9429 10^-5, 2.0275 10^-5, 2.1167 10^-5, 2.2110 10^-5, 2.3107 10^-5, 
    2.4161 10^-5, 2.5278 10^-5, 2.6461 10^-5, 2.7714 10^-5, 2.9044 10^-5, 
    3.0457 10^-5, 3.1957 10^-5, 3.3553 10^-5, 3.5251 10^-5, 3.7060 10^-5, 
    3.8988 10^-5, 4.2131 10^-5, 4.5574 10^-5, 4.9297 10^-5, 5.3325 10^-5, 
    5.7680 10^-5, 6.2391 10^-5, 6.7485 10^-5, 7.2995 10^-5, 8.5397 10^-5, 
    9.9901 10^-5, 1.1686 10^-4,1.3670 10^-4, 1.5989 10^-4, 1.8843 10^-4, 
    2.2201 10^-4, 2.6135 10^-4, 3.0743 10^-4, 3.6135 10^-4, 4.2439 10^-4, 
    4.9805 10^-4, 5.8405 10^-4, 6.8437 10^-4, 8.0134 10^-4, 9.3759 10^-4, 
    1.0962 10^-3, 1.2955 10^-3, 1.5312 10^-3, 1.8062 10^-3, 2.1264 10^-3, 
    2.9297 10^-3, 4.0066 10^-3, 5.4404 10^-3, 7.3371 10^-3, 9.8305 10^-3, 
    1.2939 10^-2, 1.6591 10^-2, 2.1047  10^-2, 2.6104 10^-2, 3.2514 10^-2, 
    4.0682 10^-2, 5.1141 10^-2, 9.2617 10^-2, 1.7357 10^-1, 3.4465 10^-1, 
    7.1557 10^-1, 1.5386, 1.6645}  ;
iKinematicViscosity["OldUnits"] = (Units`Meter^2)/Units`Second;
iKinematicViscosity["Quantity"] = ("Meters"^2)/"Seconds";

AtmosphericPlot[KinematicViscosity, opts___?OptionQ]:=
  ListPlot[
    Transpose[{Log[10, iKinematicViscosity["Data"]],
                   Take[$Alts/1000, Length[iKinematicViscosity["Data"]]]}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"Log[10, KinematicViscosity]",
                 "Altitude, Km"},
   PlotRange->All,
   Axes->False]

iThermalConductivityCoefficient["Data"] = {2.7882, 2.7634, 
    2.7384, 2.7134, 2.6884, 2.6632, 2.6380, 2.6126, 2.5872, 2.5618, 2.5326, 
    2.5106, 2.4849, 2.4591, 2.4333, 2.4073, 2.3813, 2.3552, 2.3290, 2.3028, 
    2.2765, 2.2500, 2.2236, 2.1970, 2.1703, 2.1436, 2.1168, 2.0899, 2.0630, 
    2.0359, 2.0088, 1.9816, 1.9543, 1.9533, 1.9533, 1.9533, 1.9533,1.9533, 
    1.9533, 1.9533, 1.9533, 1.9533, 1.9533, 1.9533, 1.9533, 1.9533, 1.9611, 
    1.9695, 1.9778, 1.9862, 1.9945, 2.0029, 2.0112, 2.0195, 2.0278, 2.0361, 
    2.0443, 2.0526, 2.0733, 2.0963, 2.1193, 2.1422, 2.1878, 2.2331, 2.2781, 
    2.3229, 2.3764, 2.3973, 2.3973, 2.3843, 2.3400, 2.2955, 2.2508, 2.2058, 
    2.0926, 1.9780, 1.8834, 1.8001, 1.7126, 1.7078};
iThermalConductivityCoefficient["OldUnits"] =
    Units`Joule/(Units`Meter Units`Second Units`Kelvin);
iThermalConductivityCoefficient["Quantity"] =
    "Joules"/("Meters"*"Seconds"*"Kelvins");

AtmosphericPlot[ThermalConductivityCoefficient, opts___?OptionQ]:=
 ListPlot[
    Transpose[{iThermalConductivityCoefficient["Data"],
	 Take[$Alts/1000, Length[iThermalConductivityCoefficient["Data"]]]}],
    opts,
    Joined->True,
    Frame->True,
    FrameLabel->{"ThermalConductivityCoefficient, J/(m s K)",
                 "Altitude, Km"},
   PlotRange->All,
   Axes->False]

Scan[
 (
  MessageName[#, "cnvrt"] =
	"Unable to convert altitude `1` to a numerical multiple of Meter.";
  MessageName[#, "rng"] =
	"The altitude `1` falls outside of the supported range from `2` to `3` Meters."
 )&,
	{CollisionFrequency, DynamicViscosity, GravityAcceleration, 
   KinematicViscosity, KineticTemperature, MeanDensity, MeanFreePath, 
   MeanParticleSpeed, MeanMolecularWeight, NumberDensity, Pressure, 
   PressureScaleHeight, SoundSpeed, ThermalConductivityCoefficient}
]

(* utility to accomodate both old & new style unit output, depending on input *)
fromMode[num_, iobj_, "Quantity"] :=
    Quantity[num, iobj["Quantity"]]
(* unfortunately, to maintain complete compatability, 'Numeric' must emit
   old units -- because '0' and '0.0' work in the old package (due to
   special property of multiplication by 0 killing the units). *)
fromMode[num_, iobj_, "OldUnits" | "Numeric"] :=
    num * iobj["OldUnits"]

toMode[a_Quantity] := {QuantityMagnitude[UnitConvert[a, "Meters"]], "Quantity"}
toMode[a_?(!FreeQ[#, _Symbol]&)] :=
    {Units`Convert[a, Units`Meter]/Units`Meter, "OldUnits"}
toMode[a_?NumericQ] := {a, "Numeric"}
toMode[any_] := {$Failed, $Failed}

atmosphericDataInterpolation[object_, altitude_] :=
   Module[{iobject = Symbol["StandardAtmosphere`Private`i"<>SymbolName[object]],
           n, alts, low, high, pos, a, p, a1, a2, d1, d2, alt, data, mode},
     {alt, mode} = toMode[altitude];
     If[!NumberQ[N[alt]],
	 Message[MessageName[object, "cnvrt"], altitude];
	 Return[$Failed]];
     data = iobject["Data"];
     n = Length[data];
     alts = Take[$Alts, n];
     {low, high} = {First[alts], Last[alts]};
     If[alt < low || alt > high,
	Message[MessageName[object, "rng"], altitude, low, high];
	Return[$Failed]
     ];
     pos = Position[alts, a_ /; a>=alt, 1, 1];
     If[MatchQ[pos, {{1}}] && alt==low,
        Return[fromMode[First[data], iobject, mode]]];
     p = First[First[pos]];
     {a1, a2} = alts[[{p-1, p}]];
     {d1, d2} = data[[{p-1, p}]];
     fromMode[(d1 + (alt-a1)/(a2-a1) * (d2-d1)), iobject, mode]
   ]

End[];

EndPackage[];

(* :Examples:
	error examples:
		CollisionFrequency[1]
		MeanParticleSpeed[-6000 Meter]
		GravityAcceleration[1000001 Meter]		
		DynamicViscosity[85501 Meter]
*)

(* :Examples:
	AtmosphericPlot[CollisionFrequency]
	AtmosphericPlot[DynamicViscosity]
		(* DynamicViscosity does not decrease monotonically
			with increasing altitude. *)
	AtmosphericPlot[GravityAcceleration]
	AtmosphericPlot[KinematicViscosity]
	AtmosphericPlot[KineticTemperature]
		(* KineticTemperature does not increase monotonically
                        with increasing altitude. *)
	AtmosphericPlot[MeanDensity]
	AtmosphericPlot[MeanFreePath]
	AtmosphericPlot[MeanMolecularWeight]
	AtmosphericPlot[MeanParticleSpeed]
		(* MeanParticleSpeed does not increase monotonically
                        with increasing altitude. *)
	AtmosphericPlot[NumberDensity]
	AtmosphericPlot[Pressure]
	AtmosphericPlot[PressureScaleHeight]
		(* PressureScaleHeight does not increase monotonically
                        with increasing altitude. *)
	AtmosphericPlot[SoundSpeed]
		(* SoundSpeed does not decrease monotonically 
			with increasing altitude. *)
	AtmosphericPlot[ThermalConductivityCoefficient]
		(* ThermalConductivityCoefficient does not decrease
                        monotonically with increasing altitude. *)
*)

(* :Examples:
	Plot[ Evaluate[CollisionFrequency[x Meter] Second],
			{x, -5000, 1000000}]
	Plot[ Evaluate[DynamicViscosity[x Meter] Meter^2/(Newton Second)],
			{x, -5000, 85500}]
	Plot[ Evaluate[GravityAcceleration[x Meter] Second^2/Meter],
			{x, -5000, 1000000}]
	Plot[ Evaluate[KinematicViscosity[x Meter] Second/Meter^2],
			{x, -5000, 85500}]
	Plot[ Evaluate[KineticTemperature[x Meter] 1/Kelvin],
			{x, -5000, 1000000}]
	Plot[ Evaluate[MeanDensity[x Meter] Meter^3/Kilogram],
			{x, -5000, 1000000}]
	Plot[ Evaluate[MeanFreePath[x Meter] 1/Meter],
			{x, -5000, 1000000}]
	Plot[ Evaluate[MeanMolecularWeight[x Meter] Kilo Mole/Kilogram],
			{x, -5000, 1000000}]
	Plot[ Evaluate[MeanParticleSpeed[x Meter] Second/Meter], 
        		{x, -5000, 1000000},
	   AxesLabel -> {"m", "m/sec"}]
	Plot[ Evaluate[NumberDensity[x Meter] Meter^3],
			{x, -5000, 1000000}]
	Plot[ Evaluate[Pressure[x Meter] 1/(Milli Bar)],
			{x, -5000, 1000000}]
	Plot[ Evaluate[PressureScaleHeight[x Meter] 1/Meter],
			{x, -5000, 1000000}]
	Plot[ Evaluate[SoundSpeed[x Meter] Second/Meter],
			{x, -5000, 85500}]
	Plot[ Evaluate[ThermalConductivityCoefficient[x Meter]
			 (Meter Second Kelvin)/Joule],
			{x, -5000, 85500}]
*)

