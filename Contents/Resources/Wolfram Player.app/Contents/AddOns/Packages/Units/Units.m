(* ::Package:: *)

(* :Title: Units *)

(* :Author: Stephen Wolfram *)

(* :Summary:
This package provides functions for conversion between units.
It also provides additional units beyond base SI units and derived
SI units found in SIUnits`.
*)

(* :Package Version: 1.6.1 *)

(* :Copyright: Copyright 1987-2007, Wolfram Research, Inc. *)

(* :Context: Units` *)

(* :History: 
	Version 1.1 by Stephen Wolfram (Wolfram Research), February 1987.
	Revised by ECM (Wolfram Research), November 1990, March 1995. 
	Revised by John Novak (Wolfram Research), September 1996. 
	Revised by ECM (Wolfram Research), April, November 1997.
	Version 1.5 by John Novak (Wolfram Research), February 1998
            - adds revisions of values to SolarMass and AstronomicalUnit
	      made by Barbara Ercolano, July 1997.
	Version 1.6 by ECM, Convert improvements, July 1998.
    Version 1.6.1 by John M. Novak - fixes to some units, April 1999.
    Edited to fit the new paclet structure, Sept. 2006.
 *) 

(* :Keywords: *)

(* :Source: 
	CRC Handbook of Chemistry and Physics, 69th Edition, 1988-1989.
	CRC Handbook of Chemistry and Physics, 75th Edition, 1995.
	Kaye & Laby, Allen, World Almanac, etc.
    Taylor, Barry N., Guide for the Use of the International System of
        Units, National Institute of Standards and Technology Special
        Publication 811, 1995 Edition.
*)

(* :Warning: Makes use of the system symbols
	 Span, Degree, Circle, Point, Last, Drop, Cup, Gamma, and Byte. *)

(* :Mathematica Version: 3.0 *)

(* :Limitation: In most cases, a unit must be entered with the metric
		prefix separated from the root by a space (e.g., kilometer is
		expressed as Kilo Meter).
		The exception is any unit fundamental to one of the supported
		systems of measurement (e.g., kilogram is the fundamental unit
		of length in the MKS system and may be expressed as either
		Kilogram or Kilo Gram). *)

(* :Discussion: *)

BeginPackage["Units`"]

Get[ToFileName["Units","SIUnits.m"]]

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"Units`"],
StringMatchQ[#,StartOfString~~"Units`*"]&]//ToExpression;
];

If[Not@ValueQ[Convert::usage],Convert::usage = 
"Convert[old, new] converts old to a form involving the combination of units \
new."];

If[Not@ValueQ[ConvertTemperature::usage],ConvertTemperature::usage =
"ConvertTemperature[temp, old, new] converts temp from the old scale to the new \
scale."];

If[Not@ValueQ[SI::usage],SI::usage = "SI[expr] converts expr to SI units (International System)."];

If[Not@ValueQ[MKS::usage],MKS::usage = "MKS[expr] converts expr to MKS units (meter/kilogram/second)."];

If[Not@ValueQ[CGS::usage],CGS::usage = "CGS[expr] converts expr to CGS units (centimeter/gram/second)."];


(** non-SI fundamental units usage **)

If[Not@ValueQ[Bit::usage],Bit::usage = "Bit is the fundamental unit of information."];

(** CGS units usage **)

If[Not@ValueQ[Centimeter::usage],Centimeter::usage = "Centimeter is the fundamental CGS unit of length."];

If[Not@ValueQ[Gram::usage],Gram::usage = "Gram is the fundamental CGS unit of mass."];

(** SI prefixes usage **)

Map[(If[Not@StringQ[Evaluate[#[[1]]]::"usage"], Evaluate[#[[1]]]::"usage" =
	StringJoin[ToString[#[[1]]], " is the SI unit prefix denoting 10^", 
		ToString[#[[2]]],"."]])&,
    Transpose[{
	{Yotta,Zetta,Exa,Peta,Tera,Giga,Mega,Kilo,Hecto,Deca,Deci,Centi,Milli,
	Micro,Nano,Pico,Femto,Atto,Zepto,Yocto},
	{24,21,18,15,12,9,6,3,2,1,-1,-2,-3,-6,-9,-12,-15,-18,-21,-24}	
    }]
]

(** supplementary SI units usage **)

If[Not@ValueQ[Radian::usage],Radian::usage = "Radian is a dimensionless measure of plane angle."];
If[Not@ValueQ[Steradian::usage],Steradian::usage = "Steradian is a dimensionless measure of solid angle."];

(** other usage messages **)

Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit multiplier."]])&,
	{Percent,Gross,Dozen,BakersDozen,
	ArcMinute,ArcSecond,RightAngle,Quadrant,Grade}
	]

$NewMessage[ Degree, "usage"]
If[StringQ[Degree::usage],
Degree::usage = StringJoin[ Degree::usage, " It is also a unit multiplier."]
]

$NewMessage[ Circle, "usage"]
If[StringQ[Circle::usage],
Circle::usage = StringJoin[ Circle::usage, " It is also a unit multiplier."]
]

If[Not@ValueQ[AU::usage],AU::usage = "AU or AstronomicalUnit is a unit of length."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of length."]])&,
	{Angstrom,XUnit,Fermi,Micron,LightYear,Parsec,AstronomicalUnit,Didot,
	DidotPoint,Cicero,Inch,Mil,Caliber,Hand,Cubit,Ell,Pica,PrintersPoint,
	Foot,Feet,Rope,Fathom,Cable,StatuteMile,NauticalMile,
	Skein,Stadion,AstronomicalUnit,Yard,Bolt,Furlong,Stadium,
	Pole,Perch,SurveyMile,Mile,League,Link,Rod,Chain}
	]

$NewMessage[ Span, "usage"]
If[StringQ[Span::usage],
Span::usage = StringJoin[ Span::usage, " It is also a unit of length."]
]

$NewMessage[ Point, "usage"]
If[StringQ[Point::usage],
Point::usage = StringJoin[ Point::usage,
    " It is also a unit of length corresponding to a computer point, or 1/72 of an inch."]
]

Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of area."]])&,
	{Barn,Hectare,Acre,Are,Rood,Section,Township}
	]
If[Not@ValueQ[Gallon::usage],Gallon::usage = "Gallon is a US volume unit."];
If[Not@ValueQ[UKGallon::usage],UKGallon::usage = "UKGallon (imperial gallon) is a British volume unit."];
If[Not@ValueQ[ImperialGallon::usage],ImperialGallon::usage = "ImperialGallon is a British volume unit."];
If[Not@ValueQ[UKPint::usage],UKPint::usage = "UKPint (imperial pint) is a British volume unit."];
If[Not@ValueQ[ImperialPint::usage],ImperialPint::usage = "ImperialPint is a British volume unit."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of volume."]])&,
	{Stere,Barrel,Cord,RegisterTon,BoardFoot,Liter,WineBottle,
	Firkin,Jeroboam,Bucket,Puncheon,Butt,Hogshead,Tun,Pint,
	FluidOunce,Minim,Shot,Jigger,Pony,FluidDram,Tablespoon,Teaspoon,
	Gill,Noggin,Quart,Fifth,Magnum,Peck,Bushel,Seam,Bag,Omer,Ephah}
	]

$NewMessage[ Cup, "usage"]
If[StringQ[Cup::usage],
Cup::usage = StringJoin[ Cup::usage, " It is also a unit of volume."]
]

$NewMessage[ Last, "usage"]
If[StringQ[Last::usage],
Last::usage = StringJoin[ Last::usage, " It is also a unit of volume."]
]

$NewMessage[ Drop, "usage"]
If[StringQ[Drop::usage],
Drop::usage = StringJoin[ Drop::usage, " It is also a unit of volume."]
]

If[Not@ValueQ[Kayser::usage],Kayser::usage = "Kayser is a unit of inverse length."];
If[Not@ValueQ[Diopter::usage],Diopter::usage = "Diopter is a unit of inverse length."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of time."]])&,
	{Minute,Hour,Day,Week,Fortnight,Year,Month,Decade,Century,Millennium,
	TropicalYear,SiderealYear,SiderealSecond}
	]
If[Not@ValueQ[Knot::usage],Knot::usage = "Knot is a unit of speed."];
If[Not@ValueQ[Gravity::usage],Gravity::usage = "Gravity is a measure of acceleration due to gravity."];
If[Not@ValueQ[Gal::usage],Gal::usage = "Gal is the derived CGS measure of acceleration due to gravity."];
If[Not@ValueQ[AMU::usage],AMU::usage = "AMU or AtomicMassUnit is a unit of mass."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
        StringJoin[ToString[#], " is a unit of mass."]])&,
  {AtomicMassUnit,
   Dalton, (* same as AtomicMassUnit *)
   Geepound, (* "Gee" stands for gravitational acceleration ... same as Slug *)
   MetricTon, (* same as Tonne *)
   Quintal, Slug, SolarMass, Tonne}      
                             ]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
       StringJoin[ToString[#], " is a unit of weight."]])&,
   {AssayTon, AvoirdupoisOunce, AvoirdupoisPound, Bale,
    Carat, Cental, Drachma, Grain, GrossHundredweight,
    Hundredweight, Libra, LongTon, Mina, NetHundredweight, Obolos, Ounce,
    Pennyweight, Pondus, Pound, Shekel, ShortHundredweight,
     ShortTon, Stone, Talent, Ton, TroyOunce, Wey}
                             ]
If[Not@ValueQ[Denier::usage],Denier::usage = "Denier is a unit of fineness for yarn or thread."];
If[Not@ValueQ[Dyne::usage],Dyne::usage = "Dyne is the derived CGS unit of force."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = 
	StringJoin[ToString[#], " is a unit of force."]])&,
	{Poundal,TonForce,PoundForce,PoundWeight,KilogramForce,
	KilogramWeight,GramWeight}
	]
If[Not@ValueQ[PSI::usage],PSI::usage = "PSI (pounds per square inch) is a unit of pressure."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of pressure."]])&,
	{Atmosphere,InchMercury,Barye,Bar,Torr,MillimeterMercury,
	 PoundsPerSquareInch}
	]
If[Not@ValueQ[Erg::usage],Erg::usage = "Erg is the derived CGS unit of energy."];
If[Not@ValueQ[BTU::usage],BTU::usage = "BTU or BritishThermalUnit is a unit of energy."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of energy."]])&,
	{ElectronVolt,Calorie,Rydberg,BritishThermalUnit,Therm}
	]
If[Not@ValueQ[Horsepower::usage],Horsepower::usage = "Horsepower is a unit of power."];
If[Not@ValueQ[ChevalVapeur::usage],ChevalVapeur::usage = "ChevalVapeur is a unit of power."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" =
	StringJoin[ToString[#], " is a unit of temperature."]])&,
	{Centigrade,Celsius,Fahrenheit,Rankine}
	]
If[Not@ValueQ[Poise::usage],Poise::usage = "Poise is the derived CGS unit of absolute viscosity."];
If[Not@ValueQ[Reyn::usage],Reyn::usage = "Reyn is a unit of absolute viscosity."];
If[Not@ValueQ[Stokes::usage],Stokes::usage = "Stokes is the derived CGS unit of kinematic viscosity."];
If[Not@ValueQ[Rhes::usage],Rhes::usage = "Rhes is a unit of viscosity."];
If[Not@ValueQ[Talbot::usage],Talbot::usage = "Talbot is a unit of luminous energy (quantity of light)."];
If[Not@ValueQ[Lumerg::usage],Lumerg::usage = "Lumerg is a unit of luminous energy (quantity of light)."];
If[Not@ValueQ[Phot::usage],Phot::usage = "Phot is the derived CGS unit of illumination (illuminance)."];
If[Not@ValueQ[FootCandle::usage],FootCandle::usage = "FootCandle is a unit of illumination (illuminance)."];
If[Not@ValueQ[Hefner::usage],Hefner::usage = "Hefner is a unit of luminous intensity."];
If[Not@ValueQ[Candle::usage],Candle::usage = "Candle is a unit of luminous intensity."];
If[Not@ValueQ[Stilb::usage],Stilb::usage = 
	"Stilb is the derived CGS unit of luminance (photometric brightness)."];
If[Not@ValueQ[Nit::usage],Nit::usage = "Nit is a unit of luminance (photometric brightness)."];
If[Not@ValueQ[Lambert::usage],Lambert::usage = "Lambert is a unit of luminance (photometric brightness)."];
If[Not@ValueQ[Apostilb::usage],Apostilb::usage = "Apostilb is a unit of luminance (photometric brightness)."];
If[Not@ValueQ[Rutherford::usage],Rutherford::usage = "Rutherford is a unit of radioactivity."];
If[Not@ValueQ[Curie::usage],Curie::usage = "Curie is a unit of radioactivity."];
If[Not@ValueQ[Rad::usage],Rad::usage = "Rad is a unit of absorbed dose of radiation."];
If[Not@ValueQ[Rontgen::usage],Rontgen::usage = "Rontgen is a unit of exposure to X or gamma radiation."];
If[Not@ValueQ[Roentgen::usage],Roentgen::usage = "Roentgen is a unit of exposure to X or gamma radiation."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of electric current."]])&,
	{Abampere,Statampere,Biot}
	]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of electric resistance."]])&,{Abohm,Statohm}
	]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of electric conductance."]])&,{Mho,Abmho}
	]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of electric charge."]])&,{Abcoulomb,Statcoulomb}
	]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of electric capacitance."]])&,{Abfarad,Statfarad}
	]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of inductance."]])&, {Abhenry,Stathenry}
	]
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of electric potential difference."]])&,{Abvolt,Statvolt}
	]
If[Not@ValueQ[Gauss::usage],Gauss::usage = "Gauss is the derived CGS unit of magnetic flux density."];

$NewMessage[ Gamma, "usage"]
If[StringQ[Gamma::usage],
Gamma::usage = StringJoin[ Gamma::usage, " It is also a unit of magnetic flux density."]
]

If[Not@ValueQ[Gilbert::usage],Gilbert::usage = "Gilbert is a unit of magnetomotive force."];
If[Not@ValueQ[Oersted::usage],Oersted::usage = "Oersted is the derived CGS unit of magnetic intensity."];
If[Not@ValueQ[Maxwell::usage],Maxwell::usage = "Maxwell is the derived CGS unit of magnetic flux."];
If[Not@ValueQ[BohrMagneton::usage],BohrMagneton::usage = "BohrMagneton is a unit of magnetic moment."];
If[Not@ValueQ[NuclearMagneton::usage],NuclearMagneton::usage = "NuclearMagneton is a unit of magnetic moment."];
Map[(If[Not@StringQ[Evaluate[#]::"usage"], Evaluate[#]::"usage" = StringJoin[ToString[#],
	" is a unit of information."]])&,{Nibble,Baud}
	]
$NewMessage[ Byte, "usage"]
If[StringQ[Byte::usage],
Byte::usage = StringJoin[ Byte::usage, " It is also a unit of information."]
]

(* ************************************************************** *)
Begin["`Private`"]

myConvert[x_, uniqueUnits_, new_] :=
  Module[{temp1, temp2, temp3, tempUniqueUnits},
	(* Collect is needed for cases like old = x Meter + 2 Meter . *)
	temp1 = Collect[x, uniqueUnits];
	temp2 = SI [ temp1/new ];
	tempUniqueUnits = UniqueCasesUnit[temp2];
	temp3 = Collect[temp2, tempUniqueUnits];
	temp3 //. $ToFundamental
  ]
	

(* unit conversion *)

Convert[old_, new_] :=
	Module[{x, y, t1, oldUU, t1UU},
	  (
		oldUU = UniqueCasesUnit[old];
		If[MatchQ[old, Power[x_, y_Rational]],
			x = old[[1]];
			y = old[[2]];
			t1 = myConvert[x, oldUU, new^(1/y)];
			t1UU = UniqueCasesUnit[t1];
			t1 = PowerExpand[t1^y, t1UU],
			t1 = myConvert[old, oldUU, new];
			t1UU = UniqueCasesUnit[t1];
			t1 = PowerExpand[t1, t1UU]
			
		];
		If[HasUnitQ[N[t1]],
			Message[Convert::incomp, old, new];
			Return[old],
			If[Apply[Or,
				Map[Position[new,#]!={}&,
					{Centigrade,Celsius,Kelvin,
						Fahrenheit,Rankine}]],
				Message[Convert::temp]
			];
			Return[t1 new]
		]
	  ) /; If[Head[old] === List,
		  Message[Convert::old, old];  False, True] &&
	       If[Head[new] === List,
                  Message[Convert::new, new];  False, True]	
	] 

HasUnitQ[expr_] :=(
     Count[expr, _Symbol?(
             (
		MatchQ[#, _Symbol] &&
	        (
                   (* check for units having "Units`" or
                       "SIUnits`" contexts *)
                   MemberQ[{ "Units`","SIUnits`"},
                 	Context[#]] ||
                   (* check for units having "System`" context *)
                   MemberQ[{Span, Cup, Degree, Circle, Point, Last, Drop, Gamma, Byte}, #]
	        )
	     ) 		&), Infinity] =!= 0)	

UniqueCasesUnit[expr_] :=
	Union[Cases[expr, _Symbol?(
             (
		MatchQ[#, _Symbol] &&
	        (
             	    (* check for units having "Units`" or
                       "SIUnits`" contexts *)
                    MemberQ[{ "Units`","SIUnits`"},
                 	Context[#]] ||
                    (* check for units having "System`" context *)
                    MemberQ[{Span, Cup, Degree, Circle, Point, Last, Drop, Gamma, Byte}, #]
                )
	      )		&), Infinity]]


Convert::old = "Argument `` should be a simple expression involving units, \
not a list."
Convert::new = "Argument `` should be a simple expression involving units, \
not a list."
Convert::incomp = "Incompatible units in `1` and `2`."
Convert::temp =
"Warning: Convert[old,new] converts units of temperature. \
ConvertTemperature[temp,old,new] converts absolute temperature." 

SI[expr_] := expr //. $ToIntermediate //. $SIPrefixes //. $ToSI
MKS := SI
CGS[expr_] := MKS[expr] //. $ToCGS


(* temperature conversion *)

validScales = {Celsius, Centigrade, Fahrenheit, Kelvin, Rankine};

ValidTargetQ[target_] :=
	If[MemberQ[validScales, target],
	   True,
	   Message[ConvertTemperature::invtarg, target];
	   False ]

ConvertTemperature::invtarg = "`` is an invalid target scale."

ConvertTemperature::invscal = "`` is an invalid scale for the temperature."

ConvertTemperature[x_, target_, target_] = x

ConvertTemperature[x_, old_, new_] :=
	$Failed /; !MemberQ[validScales, old] &&
		   (Message[ConvertTemperature::invscal, old]; False)	

ConvertTemperature[x_, Celsius, target_] := 
	ConvertTemperature[27315/100+x, Kelvin, target] /; ValidTargetQ[target]

ConvertTemperature[x_, Centigrade, target_] :=  
	ConvertTemperature[x, Celsius, target] /; ValidTargetQ[target]

ConvertTemperature[x_, Fahrenheit, target_] := 
	ConvertTemperature[(x-32) 5/9 + 27315/100, Kelvin, target] /;
						ValidTargetQ[target]

ConvertTemperature[x_, Rankine, target_] := 
	ConvertTemperature[x 5/9, Kelvin, target] /; ValidTargetQ[target]

ConvertTemperature[x_, Kelvin, Celsius] := x-27315/100

ConvertTemperature[x_, Kelvin, Centigrade] := x-27315/100

ConvertTemperature[x_, Kelvin, Fahrenheit] := (x-27315/100) 9/5 + 32

ConvertTemperature[x_, Kelvin, Rankine] := x 9/5


(* *************************************************************** *)

(** fundamental SI unit symbols **)

Map[(Evaluate[#[[1]]]::"symbol" = #[[2]])&,
	{{Meter,"m"},{Kilogram,"kg"},{Second,"s"},{Ampere,"A"},{Kelvin,"K"},
	 {Mole,"mol"},{Candela,"cd"}}
	 ]

(** derived SI unit symbols **)

Map[(Evaluate[#[[1]]]::"symbol" = #[[2]])&,
	{{Radian,"rad"},{Steradian,"sr"},{Newton,"N"},{Pascal,"Pa"},{Joule,"J"},
	{Watt,"W"},{Coulomb,"C"},{Volt,"V"},{Ohm,"Omega"},{Siemens,"S"},
	{Farad,"F"},{Weber,"Wb"},{Henry,"H"},{Tesla,"T"},{Henry,"H"},
	{Lumen,"lm"},{Lux,"lx"},{Hertz,"Hz"},{Becquerel,"Bq"},{GrayDose,"Gy"}}
	]

(** non-SI fundamental units symbols **)

Bit::symbol = "bit"

(** fundamental CGS and derived CGS unit symbols **) 

Map[(Evaluate[#[[1]]]::"symbol" = #[[2]])&,
	{{Centimeter,"cm"},{Gram,"g"},{Gal,"gal"},{Dyne,"dyn"},{Erg,"erg"},
	{Poise,"P"},{Stokes,"St"},{Phot,"ph"},{Stilb,"sb"},{Gauss,"G"},
	{Oersted,"Oe"},{Maxwell,"Mx"}}
	]

(** SI prefix symbols **)

Map[(Evaluate[#[[1]]]::"symbol" = #[[2]])&,
	{{Yotta, "Y"}, {Zetta, "Z"}, {Exa,"E"},{Peta,"P"},{Tera,"T"},{Giga,"G"},
     {Mega,"M"},{Kilo,"k"},
	 {Hecto,"h"},{Deca,"da"},{Deci,"d"},{Centi,"c"},{Milli,"m"},
	 {Micro,"mu"},{Nano,"n"},{Pico,"p"},{Femto,"f"},{Atto,"a"}, {Zepto, "z"},
     {Yocto, "y"}}
	 ]

(** supplementary SI unit symbols **)

Radian::symbol = "rad"
Steradian::symbol = "sr"

(** some ToSI unit symbols **)

Map[(Evaluate[#[[1]]]::"symbol" = #[[2]])&,
	{{Inch,"in"},{Foot,"ft"},{Yard,"yd"},{Mile,"mi"},{NauticalMile,"nm"},
	 {AstronomicalUnit,"AU"},{LightYear,"ly"},{Parsec,"pc"},
	 {Barn,"b"},{Hectare,"ha"},{Stere,"st"},{Liter,"l"},
	 {Gallon,"gal"},{UKGallon,"impgal"},{ImperialGallon,"impgal"},
	 {Pint,"pt"},{UKPint,"ipt"},{ImperialPint,"ipt"},
	 {Barrel,"bbl"},{Minute,"min"},{Hour,"h"},{Day,"d"},{Year,"yr"},
	 {Gravity,"g"},{Curie,"Ci"},{Roentgen,"R"},{Rontgen,"R"},{Rad,"rad"},
	 {Gram,"g"},{Pound,"lb"},{Hundredweight,"cwt"},{Ounce,"oz"},
	 {Atmosphere,"atm"},{Tonne,"t"},{Bar,"bar"},{BTU,"Btu"},
	 {BritishThermalUnit,"Btu"},
	 {Poundal,"pdl"},{PoundForce,"lbf"},
	 {ElectronVolt,"eV"}}
	 ]

(* ************************************************************** *)

$SIPrefixes =
	Dispatch[{
    Yotta -> 10^24,
    Zetta -> 10^21,
	Exa -> 10^18,
	Peta -> 10^15,
	Tera -> 10^12,
	Giga -> 10^9,
	Mega -> 10^6,
	Kilo -> 10^3,
	Hecto -> 100,
	Deca -> 10,
	Deci -> 10^-1,
	Centi -> 10^-2,
	Milli -> 10^-3,
	Micro -> 10^-6,
	Nano -> 10^-9,
	Pico -> 10^-12,
	Femto -> 10^-15,
	Atto -> 10^-18,
    Zepto -> 10^-21,
    Yocto -> 10^-24
	}]		

$ToFundamental =    (* conversions to fundamental SI units *)
	Dispatch[{
        Radian -> 1,
        Steradian -> Radian^2,
        Newton -> Meter Kilogram Second^-2,
        Pascal -> Newton/Meter^2,
        Joule -> Newton Meter,
        Watt -> Joule/Second,
        Coulomb -> Ampere Second,
        Volt -> Watt/Ampere,
	Ohm -> Volt/Ampere,
        Siemens -> Ampere/Volt,
        Farad -> Coulomb/Volt,
        Weber -> Volt Second,
        Henry -> Ohm Second,
        Tesla -> Weber/Meter^2,
        Lumen -> Candela Steradian,  
        Lux -> Lumen/Meter^2,
        Hertz -> Second^-1,
	Becquerel -> 1/Second,
	GrayDose -> Joule/Kilogram
	}]

$ToIntermediate =		(* conversions to intermediate units *)
	Dispatch[{
(* multipliers *)
		Percent -> 1/100,
		Gross -> 144,
		Dozen -> 12,
		BakersDozen -> 13,
		Mole -> 6.0221367 10.^23,
	(* ArcMinute *)
		ArcMinute -> Degree/60,
		ArcSecond -> ArcMinute/60,
	(* RightAngle *)
		Quadrant -> RightAngle,
		Grade -> RightAngle/100,
(* length *)
	(* AstronomicalUnit *)
		AU -> AstronomicalUnit,
	(* Didot *)
		DidotPoint -> Didot,
		Cicero -> 12 Didot,
	(* Inch *)
		Mil -> Inch/1000,
		Caliber -> Inch/100,
		Hand -> 4 Inch,
		Link -> 7.92 Inch,
		Span -> 9 Inch,
		Cubit -> 18 Inch, 
		Ell -> 45 Inch,
	(* Point *)
		Point -> 1/72 Inch,
		PrintersPoint -> 0.013837 Inch,
		Pica -> 12 Point,
	(* Foot *)
		Foot -> 12 Inch, (* NOTE: Foot -> 30.48*10^-2 Meter *)
		Feet -> Foot,
		Fathom -> 6 Foot,
		Rope -> 20 Foot,
		Chain -> 66 Foot,
		Cable -> 720 Foot,
		Skein -> 360 Foot,
		Stadion -> 622 Foot,
	(* Yard *)
		Yard -> 3 Foot, (* NOTE: Yard -> 0.9144 Meter *)
		Bolt -> 40 Yard,
		Furlong -> 220 Yard,
		Stadium -> 202 Yard,
	(* Rod *)
		Rod -> 5.5 Yard,	(* NOTE *)
		Pole -> Rod,
		Perch -> Rod,
		SurveyMile -> 320 Rod,
	(* Mile *)
		Mile -> 5280 Foot, (* NOTE: Mile -> 1.609344*10^3 Meter *)
		StatuteMile -> Mile,
		League -> 3 Mile,
(* area *)
	(* Acre *)
		Acre -> 43560 Foot^2, (* NOTE: Acre -> 0.404686 Hectare *)
		Rood -> Acre/4,
	(* Section *)
		Section -> Mile^2,
		Township -> 36 Section,
(* volume *)
		Cord -> 128 Foot^3,
		RegisterTon -> 100 Foot^3,
		BoardFoot -> 144 Inch^3,
	(* Liter *)
		UKPint -> 0.568261 Liter,
		ImperialPint -> 0.568261 Liter,
		WineBottle -> 0.7576778 Liter,
		Last -> 2909.414 Liter,
	(* UKGallon *)
		UKGallon -> 4.54609 Liter,
		ImperialGallon -> 4.54609 Liter,
		Firkin -> 9 UKGallon,
	(* Gallon *)
		Gallon -> 4 Quart, (* NOTE: Gallon -> 3.78541 Liter *)
		Jeroboam -> 4/5 Gallon,
		Bucket -> 4 Gallon,
		Puncheon -> 84 Gallon,
	(* Butt *)
		Butt -> 126 Gallon,
	(* Hogshead *)
		Hogshead -> Butt/2,
		Tun -> 4 Hogshead,
	(* Pint *)
		Pint -> 2 Cup,
		Cup -> 0.236588 Liter,	(* NOTE: Cup -> Pint/2 *)
	(* FluidOunce *)
		FluidOunce -> 1/16 Pint,
		Minim -> FluidOunce/480,
		Shot -> FluidOunce,
		Jigger -> 1.5 Shot,
		Pony -> 0.5 Jigger,
	(* FluidDram *)
		FluidDram -> 1/8 FluidOunce,
		Tablespoon -> 4 FluidDram,
		Teaspoon -> Tablespoon/3,
	(* Gill *)
		Gill -> 1/4 Pint,
		Noggin -> Gill,
	(* Quart *)
		Quart -> 2 Pint,
		Fifth -> 4/5 Quart,
		Magnum -> 2 Quart, 
	(* Peck *)
		Peck -> 8.810 Liter,
	(* Bushel *)
		Bushel -> 4 Peck,
		Seam -> 8 Bushel,
		Bag -> 3 Bushel,
	(* Omer *)
		Omer -> 0.45 Peck,
		Ephah -> 10 Omer,
(* time *)
		Hour -> 60 Minute,
		Day -> 24 Hour,
		Week -> 7 Day,
		Fortnight -> 2 Week,
		Year -> 365 Day,
		Month -> Year/12,
		Decade -> 10 Year,
		Century -> 100 Year,
		Millennium -> 1000 Year,
		TropicalYear -> 365.24219 Day,
		SiderealYear -> 365.25636 Day,
(* speed *)
		Knot -> NauticalMile/Hour,
(* mass and weight *)
	(* Gram *)
		Quintal -> 100000 Gram,
		SolarMass -> 1.9891*10^33 Gram, (* CRC Handbook of Chemistry and Physics, 80th ed., p. 14-1 *)
		AssayTon -> 29.167 Gram,
		Grain -> 64.799*10^-3 Gram,
		Carat -> 0.2 Gram,
		Shekel -> 14.1 Gram,
		Obolos -> 715.38*10^-3 Gram,
		Drachma -> 4.2923 Gram,
		Libra -> 325.971 Gram,
		TroyOunce -> 31.103 Gram,
		Pennyweight -> 1.555 Gram,
	(* Tonne *)
		Tonne -> 10^6 Gram,
		MetricTon -> Tonne,
	(* AMU *)
		AMU -> 1.6605402*10^-24 Gram,
		AtomicMassUnit -> AMU,
		Dalton -> AMU,
	(* Pound *)
		Pound -> 16 Ounce, (* NOTE: Pound -> 0.45359237 Kilogram *)
		AvoirdupoisPound -> Pound,
		Pondus -> 0.71864 Pound,
		Stone -> 14 Pound,
		Wey -> 252 Pound,
		Bale -> 500 Pound,
		LongTon -> 2240 Pound,
		Cental -> 100 Pound,
		ShortTon -> 2000 Pound,
		Ton -> 2000 Pound,
		NetHundredweight -> 100 Pound,
		ShortHundredweight -> 100 Pound,
	(* Hundredweight *)
		Hundredweight -> 112 Pound,
		GrossHundredweight -> Hundredweight,
	(* Mina *)
		Mina -> 0.9463 Pound,
		Talent -> 60 Mina,
	(* Ounce *)
		Ounce -> 28.3495 Gram,
		AvoirdupoisOunce -> Ounce,
	(* Slug *)
		Geepound -> Slug,
(* fineness *)
	(* Denier *)
		Denier -> 1/9000 Gram/Meter,
(* force *)
		Dyne -> 10^-5 Newton,
		Poundal -> 0.138255 Newton,
		TonForce -> 9.96402*10^3 Newton,
	(* PoundForce *)
		PoundForce -> 4.44822 Newton,
		PoundWeight -> PoundForce,
	(* KilogramForce *)
		KilogramForce -> 9.80665 Newton,
		KilogramWeight -> KilogramForce,
		GramWeight -> KilogramWeight/1000,
(* pressure *)
	(* Pascal *)					
		Atmosphere -> 0.101325*10^6 Pascal,
		InchMercury -> 3.38639*10^3 Pascal,
		Barye -> Pascal/10,
		Bar -> 10^5 Pascal,
		PSI -> 6.894757 10^3 Pascal,
	        PoundsPerSquareInch -> PSI,
		
	(* Torr *)
		Torr -> 1.33322*10^2 Pascal,
		MillimeterMercury -> Torr,
(* energy *)
	(* Erg *)
		Rydberg -> 2.1799*10^-11 Erg,
	(* BTU *)
		BritishThermalUnit -> BTU,
		Therm -> 10^5 BTU,
(* temperature *)
		Centigrade -> Kelvin,
		Celsius -> Centigrade,
		Fahrenheit -> 5/9 Kelvin,
		Rankine -> Fahrenheit,
(* viscosity *)
	(* Poise *)
		Poise -> 0.1 Pascal Second,
		Reyn -> 6.89476*10^4 Poise,
		Rhes -> Poise^-1,	
(* light *)
		Phot -> 10^4 Lux,
		FootCandle -> Lux Meter^2/Foot^2,	
	(* Lambert *)
		Apostilb -> 10^-4 Lambert,
	(* Talbot *)
		Lumerg -> Talbot,   
(* radioactivity *)
		Rad -> 0.01 GrayDose,
		Curie -> 37*10^9 Becquerel,
		Rontgen -> 0.258*10^-3 Coulomb/Kilogram,
		Roentgen -> Rontgen,
(* electric *)
	(* Mho *)
		Abmho -> 10^9 Mho,
(* magnetic *)
		Gauss -> 10^-4 Tesla,
		Gamma -> 10^-9 Tesla,
		BohrMagneton -> 0.92740154*10^-20 Erg/Gauss,
		NuclearMagneton -> 5.0507866*10^-24 Erg/Gauss,
(* information *)
		Byte -> 8 Bit,
		Nibble -> 4 Bit,
		Baud -> Bit/Second
	}]			(* end of $ToIntermediate list *)



$ToSI =		(* conversions to basic SI units *)
	Dispatch[{
(* multipliers *)
		Degree -> Pi/180 Radian,
		Circle -> 2 Pi Radian,
	(* RightAngle *)
		RightAngle -> Pi/2 Radian,
(* length *)
		Centimeter -> 10^-2 Meter,
		Angstrom -> 10^-10 Meter,
		XUnit -> 0.1002*10^-12 Meter,
		Fermi -> 10^-15 Meter,
		Micron -> 10^-6 Meter,
		NauticalMile -> 1.852*10^3 Meter,
		LightYear -> 9.46073*10^15 Meter, (* per NIST Special Pub. 811 - 1995 Ed. *)
		Parsec -> 30857*10^12 Meter,
	(* AstronomicalUnit *)
		AstronomicalUnit -> 1.4959787066*10^11 Meter, 
	(* Didot *)
		Didot -> Meter/2660,
	(* Inch *)
		Inch -> 254/10000 Meter, (* as per NIST Special Publication 811 - 1995 Edition *)
(* area *)
		Barn -> 1*10^-28 Meter^2,
	(* Hectare *)
		Hectare -> 1*10^4 Meter^2,
        Are -> 100 Meter^2, (* per NIST Special Pub. 811 - 1995 Ed. *)
(* volume *)
		Stere -> Meter^3,
		Barrel -> 0.1590 Meter^3,
		Drop -> 0.03*10^-6 Meter^3,
	(* Liter *)
		Liter -> 10^-3 Meter^3,
(* inverse length *)
		Kayser -> 100 Meter^-1,
		Diopter -> Meter^-1,
(* time *)
		Minute -> 60 Second,
		SiderealSecond -> 0.9972696 Second,
(* acceleration *)
		Gravity -> 9.80665 Meter/Second^2,
		Gal -> 10^-2 Meter/Second^2,
(* mass *)
	(* Gram *)
		Gram -> Kilogram/1000,
	(* Slug *)
		Slug -> 14.5939 Kilogram,
(* energy *)
		ElectronVolt -> 0.1602176487*10^-18 Joule,
		Calorie -> 4.1868 Joule,
	(* Erg *)
		Erg -> 10^-7 Joule,
	(* BTU *)
		BTU -> 1.05506*10^3 Joule,
(* power *)
		Horsepower -> 0.745700*10^3 Watt,
		ChevalVapeur -> 0.735499*10^3 Watt,
(* viscosity *)
		Stokes -> 10^-4 Meter^2/Second,
(* light *)
		Stilb -> 10^4 Candela/Meter^2,
		Nit -> Candela/Meter^2,
		Hefner -> 0.92 Candela,
		Candle -> Candela,
	(* Lambert *)
		Lambert -> (10^4/Pi) Lumen/Meter^2,
	(* Talbot *)
		Talbot -> Lumen Second,
(* radioactivity *)
		Rutherford -> 10^6/Second,
(* electric *)
	(* Ampere *)
		Amp -> Ampere, 
		Abampere -> 10 Ampere,
		Statampere -> 3.335635*10^-10 Ampere,
		Gilbert -> 10/(4 Pi) Ampere,
		Biot -> 10 Ampere,
	(* Ohm *)
		Abohm -> 10^-9 Ohm,
		Statohm -> 8.987584*10^11 Ohm,
	(* Mho *)
		Mho -> 1/Ohm,
	(* Coulomb *)
		Abcoulomb -> 10 Coulomb,
		Statcoulomb -> 3.335635*10^-10 Coulomb,
	(* Farad *)
		Abfarad -> 10^9 Farad,
		Statfarad -> 1.112646*10^-12 Farad,
	(* Henry *)
		Abhenry -> 10^-9 Henry,
		Stathenry -> 8.987584*10^11 Henry,
	(* Volt *)
		Abvolt -> 10^-8 Volt,
		Statvolt -> 299.7930 Volt,
(* magnetic *)
		Oersted -> 1/(4 Pi) 10^3 Ampere/Meter,
		Maxwell -> 10^-8 Weber
	}]			(* end of $ToSI list *)


$ToCGS =		(* conversions to basic CGS units *) 
	Dispatch[{
		Meter -> 100 Centimeter,
		Kilogram -> 1000 Gram,
		Newton -> 100000 Dyne,
		Joule -> 10000000 Erg,
		Pascal Second -> 10 Poise,
		Pascal -> 10 Dyne/Centimeter^2, (* Barye *)
		Meter^2 / Second -> 10000 Stokes,
		Meter / Second^2 -> 100 Gal,
		Lux -> 10^-4 Phot,
		Candela / Meter^2 -> 10^-4 Stilb,
		Tesla -> 10^4 Gauss,
		Ampere / Meter -> (4 Pi)/1000 Oersted,
		Weber -> 10^8 Maxwell
	}]	
	

End[ ] (* Private` *)

EndPackage[]
