(* ::Package:: *)

(* :Title: BlackBodyRadiation *)

(* :Author: Barbara Ercolano *)

(* :Summary:
This package provides functions giving the basic properties of black body 
radiation at a specified temperature, 
including the total power emitted, 
the peak wavelength and and the power emitted by a narrow band about the 
peak. The package also provides a function for plotting a blackbody 
profile at a specified temperature.
*)

(* :Context: BlackBodyRadiation` *)

(* :Copyright: Copyright 1997-2007, Wolfram Research, Inc. *)

(* :Source: *)

(* :Mathematica Version: 3.0 *)

(* :History:
   Version 1.0 by Barbara Ercolano, 1997
   Version 1.1 changes by ECM, 1998
   Edited to fit the new paclet system, 2006
*)

(* :Package Version: 1.1 *)

BeginPackage["BlackBodyRadiation`", "Units`"];

If[Not@ValueQ[PeakWavelength::usage],PeakWavelength::usage = 
"PeakWavelength[temperature] gives the wavelength of the maximum emission \
of a blackbody at the specified temperature."];

If[Not@ValueQ[BlackBodyProfile::usage],BlackBodyProfile::usage=   
"BlackBodyProfile[temperature1, temperature2, ...] plots blackbody \
spectral distribution profiles at the specified temperatures."];  

If[Not@ValueQ[TotalPower::usage],TotalPower::usage=   
"TotalPower[temperature] gives the total radiative power emitted by a \
black body at the specified temperature."];

If[Not@ValueQ[MaxPower::usage],MaxPower::usage=
"MaxPower[temperature, wavelengthband] gives the radiative power emitted by \
a blackbody in the specified wavelength band about the peak \
wavelength at the specified temperature."];

Begin["`Private`"]


MaxPower::wvl = "Value `1` is not an appropriate wavelength band.";

PeakWavelength::temp =
BlackBodyProfile::temp =
TotalPower::temp =
MaxPower::temp =
  "Value `1` is not an appropriate temperature.";

convertToKelvin[T_Quantity] := QuantityMagnitude[UnitConvert[T, "Kelvins"]]
(* old-style unit compatability *)
convertToKelvin[T_] := Module[{a},
		Replace[T, {a_.*( Celsius|Centigrade) :> N[a] + 273.15,
		 a_. * Fahrenheit :>  (N[a]-32)*5/9 + 273.15,
		a_. * Rankine :> N[a] *5/9, a_. * Kelvin :> N[a]}]]

planckConstant =
   QuantityMagnitude[UnitConvert[
       Quantity["PlanckConstant"],
       "Joules" "Seconds"]];
speedOfLight =
   QuantityMagnitude[UnitConvert[
       Quantity["SpeedOfLight"],
       "Meters"/"Seconds"]];
boltzmannConstant =
   QuantityMagnitude[UnitConvert[
       Quantity["BoltzmannConstant"],
       "Joules"/"Kelvins"]];
(* following matches CODATA 2006 - WeinWavelengthDisplacementConstant *)
(* we only need the numeric value in the following, so not expressed as
  a Quantity. Value not currently available directly from Quantity framework. *)
weinConstant = 2.8977685*^-3 (* "Meters"*"Kelvins" *);

pf1 = (2 Pi planckConstant  speedOfLight^2);
pf2 = planckConstant speedOfLight/boltzmannConstant;

planckFunction[t_, x_] := (pf1/ x^5)/(Exp[pf2/(x*t)]-1)

peakWavelength[t_] := weinConstant/t;

PeakWavelength[T_Quantity] :=
    Module[{t},
        Quantity[peakWavelength[QuantityMagnitude[t]], "Meters"]/;
            (t = UnitConvert[T, "Kelvins"];
             If[QuantityQ[t] && QuantityUnit[t] === "Kelvins",
                True,
                Message[PeakWavelength::temp, T]; False,
                False]
            )
    ]
(* old-style units compatability *)
PeakWavelength[T_] := Module[{t},peakWavelength[t] Meter
	  /; (t=convertToKelvin[T];
				 If[MatchQ[t, _Real | _Integer], True,
			   Message[PeakWavelength::temp, T]; False, False])]

maxline[t_]:=  Module[{p=peakWavelength[t],q},
		q = planckFunction[t,p];
		Graphics[{Dashing[{0.01, 0.01}], Line[{{p, 0},  {p, q}}],
			Text[t K, {p, q}, {0, -1}]}]]

Options[BlackBodyProfile] =
    Options[Plot];
SetOptions[BlackBodyProfile,
    PlotRange -> All,
    Axes -> True,
    AxesLabel -> {"Wavelength, m", "Intensity, W/m^3"},
    AspectRatio -> 1/GoldenRatio,
    PlotRangeClipping -> True]

BlackBodyProfile[Ts__?(!OptionQ[#]&), opts___?OptionQ] :=
    Module[{ts, tmp},
      (* throw/catch pair to prevent floods of messages; only one
         message per bad input *)
        ts = Catch[Map[If[!MatchQ[tmp = convertToKelvin[#], _Real | _Integer] || tmp <= 0.0,
                 Message[BlackBodyProfile::temp, #];Throw[$Failed],
                 tmp]&, {Ts}]];
      (* return unevaluated if invalid *)
        blackbodyprofilecore[ts, opts]/; ts =!= $Failed
    ]

(* note: following design is revised a bit from original, to allow Plot's new
   multiple-curve handling capabilities to be used. The downside is that the
   maxline lines are no longer interleaved with the curves. I don't think this
   is fatal, and helps us let a lot of options work transparently. --JMN 01.07 *)
blackbodyprofilecore[ts_, opts___?OptionQ] :=
    Module[{peak, popts, plt, x},
        peak = Max[Map[peakWavelength, ts]];
        (* Complement passes only non-default opts -- but user-specified opts should
           have precedence even if they are default-valued. *)
        popts = FilterRules[{opts, Complement[Options[BlackBodyProfile],Options[Plot]]},
                             Options[Plot]];
        plt = Plot[Evaluate[Map[planckFunction[#, x]&, ts]], {x, 0, 6 * peak},
                   Evaluate[popts]];
        Show[plt, Map[maxline, ts]]
    ]
 
(* original code, see below.
totalPower[t_]:=
  Module[{x, slwcon, ovfl, ncvb,  nint},
	slwcon = (Head[NIntegrate::slwcon] === $Off);  Off[NIntegrate::slwcon];
 	ovfl = (Head[General::ovfl] === $Off); Off[General::ovfl];
	ncvb = (Head[NIntegrate::ncvb] === $Off); Off[NIntegrate::ncvb];
	nint = NIntegrate[planckFunction[t, x], {x, 0, Infinity}] Watt/Meter^2;
	If[!slwcon, On[NIntegrate::slwcon]];
	If[!ovfl, On[General::ovfl]];
	If[!ncvb, On[NIntegrate::ncvb]];
	nint
  ]
*)
(* The total power is given directly by the Stephan-Boltzmann law;
   no need to compute the integral numerically. We can verify the
   result by symbolic integration. Note that because we want to
   express the units in Watt/Meter^2, we strip the units from the
   computation of \[Sigma] by the use of First[Cases[..., _?NumberQ]],
   which is marginally cleaner than multiplying by the inverse of the
   units involved in this instance. The units have already been
   stripped from t elsewhere in the code. *)
totalPower[t_] :=
      UnitConvert[2 Pi^5 Quantity["BoltzmannConstant"]^4/(15 Quantity["SpeedOfLight"]^2 Quantity["PlanckConstant"]^3) * t^4,
     "Watts"/"Meters"^2]

TotalPower[T_Quantity] :=
 Module[{t},
   (
   totalPower[t]
   ) /; (t=UnitConvert[T, "Kelvins"];
         If[QuantityQ[t] && QuantityUnit[t] === "Kelvins", True,
            Message[TotalPower::temp, T]; False, False])
 ]
(* old-style unit compatability *)
TotalPower[T_]:=
 Module[{t},
   (
   QuantityMagnitude[totalPower[Quantity[t, "Kelvins"]]] * Watt/Meter^2
   ) /; (t=convertToKelvin[T];
	 If[MatchQ[t, _Real | _Integer], True,
	    Message[TotalPower::temp, T]; False, False])
 ]

MaxPower[T_Quantity, delambda_Quantity] :=
 Module[{t,delam},
  (
  0.657548 (delam/peakWavelength[QuantityMagnitude[t]]) totalPower[t]
  )/;
  (t = UnitConvert[T, "Kelvins"];
   delam = QuantityMagnitude[UnitConvert[delambda, "Meters"]];
   If[QuantityQ[t] && QuantityUnit[t] === "Kelvins",
        If[NumericQ[delam], True,
            Message[MaxPower::wvl, delambda]; False,
             False],
       Message[MaxPower::temp, T]; False,
       False])
 ]
(* old-style unit compatability *)
MaxPower[T_, delambda_] :=
 Module[{t,delam},
  (
  0.657548 (delam/peakWavelength[t]) TotalPower[t]
  ) /;
   (t=convertToKelvin[T];
    delam = N[Convert[delambda, Meter]/Meter];
    If[MatchQ[t, _Real | _Integer],
        If[MatchQ[delam, _Real | _Integer], True,
            Message[MaxPower::wvl, delambda]; False,
             False],
       Message[MaxPower::temp, T]; False,
       False])
 ]

End[];

EndPackage[];
