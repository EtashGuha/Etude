(* ::Package:: *)

(* :Title: SIUnits *)

(* :Author: Wolfram Research *)

(* :Summary: 
This package introduces base SI and derived SI units.
(SI stands for the International System of units).
*)

(* :Package Version: 1.1 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc. *)

(* :Context: Units` *)

(* :History:
	ECM (Wolfram Research), November 1990
	Revised April 1997 
    Fit into paclet structure for Mathematica 6, September 2006 
       Second and Newton were exported, as they are no longer System`. *)

(* :Keywords: *)

(* :Source:
	CRC Handbook of Chemistry & Physics, 69th edition, 1988-1989 *)

(* :Warning: Makes use of the system symbol
         Newton. *)

(* :Mathematica Version: 3.0-6.0 *)

(* :Limitation: None. *)

(* :Discussion: *)



(** fundamental SI units usage **)

Map[(If[Not@StringQ[Evaluate[#[[1]]]::"usage"], Evaluate[#[[1]]]::"usage" =
   StringJoin[ToString[#[[1]]], " is the fundamental SI unit of ", #[[2]]]])&,
	{{Meter,"length."},{Kilogram,"mass."},
	 {Ampere,"electric current."},{Kelvin,"thermodynamic temperature."},
	 {Mole,"amount of substance."},
	 {Candela,"luminous intensity (candlepower)."},{Second,"time."}}
	]

(*$NewMessage[ Second, "usage"]; (* reset usage message of Second to the
					System` usage *)
If[StringQ[Second::usage],
Second::usage = StringJoin[ Second::usage,
	 "   It is also the fundamental SI unit of time."]
]*)

(*$NewMessage[ Newton, "usage"]; (* reset usage message of Newton to the
					System` usage *)
If[StringQ[Newton::usage],
Newton::usage = StringJoin[ Newton::usage,
	 "   It is also the derived SI unit of force."]
]*)

(** derived SI units usage **)

Map[(If[Not@StringQ[Evaluate[#[[1]]]::"usage"], Evaluate[#[[1]]]::"usage" =
	StringJoin[ToString[#[[1]]], " is the derived SI unit of ", #[[2]]]])&,
	{{Pascal,"pressure."},{Joule,"energy."},
	{Watt,"power."},{Coulomb,"electric charge."},
	{Volt,"electric potential difference."},{Ohm,"electric resistance."},
	{Siemens,"electric conductance."},{Farad,"electric capacitance."},
	{Weber,"magnetic flux."},{Henry,"inductance."},
	{Tesla,"magnetic flux density."},{Lumen,"luminous flux."},
	{Lux,"illumination (illuminance)."},{Hertz,"frequency."},
	{Becquerel,"radioactivity."},{GrayDose,"absorbed dose of radiation."},
	{Newton,"force."}}
	]
If[Not@ValueQ[Amp::usage],Amp::usage =
	"Amp, short for Ampere, is the fundamental SI unit of electric current."];

